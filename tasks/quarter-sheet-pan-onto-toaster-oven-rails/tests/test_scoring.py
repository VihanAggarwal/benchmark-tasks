"""Offline tests for layout.py, scoring.py and task.py: synthetic pan states (mechkit AssemblyStates from the task's own
spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
    import tiltgrip  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
DEG = math.radians(1.0)
NEAR_RIM_X = -L.PAN_D / 2 + L.BEAD_W / 2  # the near rim's bead centre (pan frame x): where a hand on it would be
RIM_Y = 0.12


def pan_at(trailing, y=0.0, z=L.RAIL1_TOP, yaw=0.0, pitch=0.0, roll=0.0):
    """The pan with its trailing edge at x ``trailing`` (m) and its underside's centre at ``z`` over the table top, yawed,
    pitched (positive tips the leading end down) and rolled about its own centre."""
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.PAN_PART, (trailing + L.PAN_D / 2, y, L.TOP + z, *q), source="test")


def home_pan(**kw):
    return pan_at(L.HOME_X0, **kw)


def pan_pitched_on_leading_edge(x_lead, deg):
    """The pan resting on its leading underside edge at the lower rails' top, its trailing end ``deg`` degrees up."""
    q = mk.quat_from_ypr(0.0, math.radians(deg), 0.0)
    r = mk.quat_to_matrix(q)
    pivot = np.array([x_lead, 0.0, L.TOP + L.RAIL1_TOP])
    origin = pivot - r @ np.array([L.PAN_D / 2, 0.0, 0.0])
    return mk.state_from_poses(T.PAN_PART, (*origin, *q), source="test")


def start():
    return T.PAN_PART.rest_state()


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD]), opening)


FAR = {"left": arm((0.15, 0.45, L.TOP + 0.25)), "right": arm((0.15, -0.45, L.TOP + 0.25))}


def run(state, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(state, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.PAN_D, 0.2413, places=4)  # 9-1/2 in
        self.assertAlmostEqual(L.PAN_W, 0.3302, places=4)  # 13 in
        self.assertAlmostEqual(L.PAN_H, 0.0254, places=4)  # 1 in
        self.assertAlmostEqual(L.PAN_MASS, 0.20)

    def test_clearances_in_the_oven(self):
        self.assertAlmostEqual(L.CW / 2 - L.PAN_W / 2, 0.0069, places=4)  # 6.9 mm from each wall
        self.assertAlmostEqual(L.PAN_W / 2 - L.RAIL_Y, 0.0111, places=4)  # 11 mm over each rail
        self.assertAlmostEqual(L.CD - L.PAN_D, 0.0037, places=4)  # 3.7 mm shorter than the cavity is deep
        self.assertGreater(L.PAN_W / 2 - L.RAIL_Y, L.CW / 2 - L.PAN_W / 2)  # a wall stops the pan before a rail loses it
        pan_top = L.RAIL1_TOP + L.PAN_H
        self.assertGreater(L.RAIL2_TOP - L.RAIL_T - pan_top, 0.03)  # the upper rails are over the pan's rim by 3 cm or more
        self.assertGreater(L.ROOF_H - pan_top, 0.08)

    def test_start_is_outside_the_oven_and_the_open_door(self):
        """The review copy's points: the pan's start overlaps neither the oven nor the open drop-down door in front of
        it. The pan's leading edge is 5.7 mm short of the door's free edge (x 0.255) and 176 mm short of the face."""
        pan_lo, pan_hi = T.PAN_PART.world_bounds()
        oven_lo, oven_hi = T.OVEN.world_bounds()
        door_lo, door_hi = T.DOOR.world_bounds()
        self.assertAlmostEqual(oven_lo[0] - pan_hi[0], 0.1757, places=4)
        self.assertAlmostEqual(door_lo[0] - pan_hi[0], 0.0057, places=4)
        for other in ((oven_lo, oven_hi), (door_lo, door_hi)):
            self.assertFalse(bool(np.all(pan_hi > other[0]) and np.all(other[1] > pan_lo)))
        self.assertAlmostEqual(float(pan_lo[2]), L.TOP, places=6)  # lies on the table
        self.assertAlmostEqual(float(pan_hi[2]) - L.TOP, L.PAN_H, places=6)
        # between the two arm columns, as far from the arm bases as the door leaves room for
        self.assertGreaterEqual(float(pan_lo[0]), 0.0)
        self.assertLessEqual(float(abs(pan_lo[1]) + 0), L.PAN_W / 2 + 1e-9)
        self.assertLess(L.PAN_W / 2, 0.305 - 0.12)

    def test_open_door(self):
        """The door lies open flat in front of the oven: hinged along the front bottom edge, its free edge 170 mm out,
        as wide as the oven, its inner face level with the oven floor's top; fixed, and clear of the pan's path."""
        lo, hi = T.DOOR.world_bounds()
        self.assertIsNone(T.DOOR.mass)
        self.assertAlmostEqual(float(hi[0]), L.OVEN_X, places=6)  # the hinge line is the oven's front face
        self.assertAlmostEqual(float(hi[0] - lo[0]), 0.170, places=6)
        self.assertAlmostEqual(float(hi[1] - lo[1]), L.CW + 2 * L.WALL_T, places=6)
        self.assertAlmostEqual(float(hi[2]) - L.TOP, L.FLOOR_H, places=6)
        self.assertAlmostEqual(float(lo[2]), L.TOP, places=6)
        # the carried pan (underside 27 mm over the lower rails) and the pushing hand pass well over it
        self.assertGreater(L.RAIL1_TOP - L.DOOR_T, 0.05)

    def test_oven_geometry(self):
        lo, hi = T.OVEN.world_bounds()
        self.assertAlmostEqual(float(lo[0]), L.OVEN_X, places=6)
        self.assertAlmostEqual(float(hi[0]), L.BACK_X + L.WALL_T, places=6)
        self.assertAlmostEqual(float(hi[2]) - L.TOP, L.ROOF_H + L.ROOF_T, places=6)
        self.assertIsNone(T.OVEN.mass)  # fixed
        # rails: four plates, 4 mm thick, 18 mm deep, tops at the two heights
        rails = [s for s in T.OVEN.shapes if abs(s.size[2] - L.RAIL_T) < 1e-9]
        self.assertEqual(len(rails), 4)
        tops = sorted(round(s.center[2] + s.size[2] / 2, 6) for s in rails)
        self.assertEqual(tops, [L.RAIL1_TOP, L.RAIL1_TOP, L.RAIL2_TOP, L.RAIL2_TOP])
        for s in rails:
            self.assertAlmostEqual(s.size[1], L.RAIL_D)
            self.assertAlmostEqual(abs(s.center[1]) + s.size[1] / 2, L.CW / 2)

    def test_pan_shapes_fill_the_footprint(self):
        lo, hi = T.PAN_PART.local_bounds()
        self.assertAlmostEqual(float(hi[0] - lo[0]), L.PAN_D)
        self.assertAlmostEqual(float(hi[1] - lo[1]), L.PAN_W)
        self.assertAlmostEqual(float(hi[2] - lo[2]), L.PAN_H)
        self.assertAlmostEqual(float(lo[2]), 0.0)
        beads = [s for s in T.PAN_PART.shapes if abs(s.size[2] - L.BEAD_H) < 1e-9]
        self.assertEqual(len(beads), 4)
        near = [s for s in beads if s.center[0] < -L.PAN_D / 2 + L.BEAD_W]
        self.assertEqual(len(near), 1)
        self.assertAlmostEqual(near[0].size[0], L.BEAD_W)  # the grippers close on 16 mm
        self.assertAlmostEqual(near[0].center[0], NEAR_RIM_X)

    def test_home_position(self):
        self.assertAlmostEqual(L.HOME_X0 + L.PAN_D, L.BACK_X)
        self.assertGreater(L.HOME_X0, L.OVEN_X)  # the pan's trailing edge is inside the face when it is home
        r = S.read(home_pan())
        self.assertAlmostEqual(S.leading_gap(r), 0.0, places=9)
        self.assertAlmostEqual(S.trailing_out(r), L.OVEN_X - L.HOME_X0, places=9)


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(home_pan())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        d = r.details
        self.assertTrue(d["on_rails"] and d["in_oven"] and d["level"] and d["home"])

    def test_tier_table_is_one_score_per_state(self):
        """Every combination of the six booleans gives exactly one of 0, 0.25, 0.5, 1.0 and the first rule that holds."""
        import itertools
        for latched, settled, rails, inside, lvl, is_home, is_near in itertools.product((False, True), repeat=7):
            got = S.tier(latched, settled, rails, inside, lvl, is_home, is_near)
            if latched or not settled or not rails or not inside:
                want = 0.0
            elif lvl and is_home:
                want = 1.0
            elif lvl and is_near:
                want = 0.5
            else:
                want = 0.25
            self.assertEqual(got, want)

    def test_off_centre_and_not_home_is_one_tier(self):
        """The review copy asked which tier applies when the pan is off centre and not fully home: off-centre alone never
        changes the score (a wall stops it 7 mm out); not home and level and near is 0.5."""
        r = run(pan_at(L.HOME_X0 - 0.020, y=0.006))
        self.assertEqual(r.score, 0.5)
        r = run(home_pan(y=0.006))
        self.assertEqual(r.score, 1.0)

    def test_not_home_but_near(self):
        for short in (0.006, 0.020, 0.040):
            r = run(pan_at(L.HOME_X0 - short))
            self.assertEqual(r.score, 0.5, short)
            self.assertFalse(r.details["home"])

    def test_home_window_edge(self):
        """Home is the trailing edge at or inside the face (the review copy's point: 4 mm from the back wall left the
        trailing edge 0.3 mm out, so the door could not shut)."""
        self.assertEqual(run(pan_at(L.OVEN_X + 0.0001)).score, 1.0)  # leading edge 3.6 mm from the wall
        self.assertEqual(run(pan_at(L.OVEN_X - 0.0001)).score, 0.5)  # 0.1 mm out of the face
        self.assertEqual(run(pan_at(L.OVEN_X + 0.0005, yaw=0.5 * DEG)).score, 0.5)  # a corner yawed out of the face

    def test_near_window_edge(self):
        self.assertEqual(run(pan_at(L.OVEN_X - 0.0399)).score, 0.5)  # trailing edge 39.9 mm out
        self.assertEqual(run(pan_at(L.OVEN_X - 0.0401)).score, 0.25)

    def test_far_out_is_a_quarter_until_the_centre_is_outside(self):
        self.assertEqual(run(pan_at(L.OVEN_X - 0.060)).score, 0.25)
        self.assertEqual(run(pan_at(L.OVEN_X - 0.085)).score, 0.25)
        self.assertEqual(run(pan_at(L.OVEN_X - 0.095)).score, 0.0)  # centre less than 30 mm inside the face

    def test_tilted_on_the_rails_is_a_quarter(self):
        r = run(pan_pitched_on_leading_edge(L.BACK_X, 2.5))
        self.assertTrue(r.details["on_rails"])
        self.assertFalse(r.details["level"])
        self.assertEqual(r.score, 0.25)
        r = run(pan_pitched_on_leading_edge(L.BACK_X, 1.5))
        self.assertEqual(r.score, 1.0)

    def test_tilted_too_far_is_not_on_the_rails(self):
        self.assertEqual(run(pan_pitched_on_leading_edge(L.BACK_X, 3.0)).score, 0.0)  # trailing end 12.6 mm up

    def test_start_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)
        self.assertFalse(S.Tracker().update(start(), FAR).success)

    def test_all_scores_are_in_the_set(self):
        for dx in np.linspace(-0.12, 0.01, 27):
            self.assertIn(run(pan_at(L.HOME_X0 + dx), steps=WIN + 1).score, (0.0, 0.25, 0.5, 1.0))


class NotOnTheRails(unittest.TestCase):
    def test_on_the_cavity_floor(self):
        self.assertEqual(run(home_pan(z=L.FLOOR_H)).score, 0.0)

    def test_on_the_upper_rails(self):
        self.assertEqual(run(home_pan(z=L.RAIL2_TOP)).score, 0.0)

    def test_between_the_rails(self):
        self.assertEqual(run(home_pan(z=0.105)).score, 0.0)

    def test_sunk_into_the_lower_rails(self):
        self.assertEqual(run(home_pan(z=L.RAIL1_TOP - 0.005)).score, 0.0)
        self.assertEqual(run(home_pan(z=L.RAIL1_TOP - 0.0025)).score, 1.0)

    def test_one_rail_lost(self):
        self.assertEqual(run(home_pan(y=0.0125)).score, 0.0)  # its right edge is off the right rail
        self.assertEqual(run(home_pan(y=-0.0125)).score, 0.0)
        self.assertEqual(run(home_pan(y=0.0080)).score, 1.0)  # still 3 mm over it

    def test_rolled_onto_one_edge(self):
        self.assertEqual(run(home_pan(roll=3 * DEG)).score, 0.0)

    def test_diagonal_across_the_cavity(self):
        """A pan yawed 12 degrees would have its corners inside the walls: not a pose a rigid pan can take."""
        r = run(home_pan(yaw=12 * DEG))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(S.inside_walls(S.read(home_pan(yaw=12 * DEG))))

    def test_yawed_as_far_as_the_walls_allow(self):
        """About 1.5 degrees: its corners still clear the walls, and it is home and level."""
        self.assertTrue(S.inside_walls(S.read(pan_at(L.HOME_X0 - 0.010, yaw=1.5 * DEG))))
        self.assertEqual(run(pan_at(L.HOME_X0 - 0.010, yaw=1.5 * DEG)).score, 0.5)

    def test_through_the_back_wall(self):
        self.assertEqual(run(pan_at(L.HOME_X0 + 0.010)).score, 0.0)

    def test_on_the_table_in_front(self):
        self.assertEqual(run(pan_at(0.30, z=0.0)).score, 0.0)

    def test_turned_over(self):
        """A pan upside down on the lower rails, home: its rim is on the rails, not its underside."""
        q = mk.quat_from_ypr(0.0, 0.0, math.pi)
        r = mk.quat_to_matrix(q)
        origin = np.array([L.HOME_CX, 0.0, L.TOP + L.RAIL1_TOP + L.PAN_H]) - r @ np.array([0.0, 0.0, 0.0])
        flipped = mk.state_from_poses(T.PAN_PART, (*origin, *q), source="test")
        res = run(flipped)
        self.assertEqual(res.score, 0.0)
        self.assertGreater(res.details["tilt_deg"], 170.0)

    def test_standing_on_its_leading_edge(self):
        q = mk.quat_from_ypr(0.0, math.pi / 2, 0.0)
        r = mk.quat_to_matrix(q)
        origin = np.array([L.BACK_X, 0.0, L.TOP + L.RAIL1_TOP]) - r @ np.array([L.PAN_D / 2, 0.0, 0.0])
        self.assertEqual(run(mk.state_from_poses(T.PAN_PART, (*origin, *q), source="test")).score, 0.0)


class Latches(unittest.TestCase):
    def test_fell_off_the_table_then_set_home_is_zero(self):
        tr = S.Tracker()
        run(pan_at(0.30, z=-0.05), steps=3, tracker=tr)
        self.assertTrue(tr.latched)
        self.assertEqual(run(home_pan(), tracker=tr).score, 0.0)

    def test_on_the_oven_floor_is_not_a_fall(self):
        tr = S.Tracker()
        run(home_pan(z=L.FLOOR_H), steps=3, tracker=tr)
        self.assertFalse(tr.latched)
        self.assertEqual(run(home_pan(), tracker=tr).score, 1.0)

    def test_lifted_pan_is_not_a_fall(self):
        tr = S.Tracker()
        run(pan_at(0.2, z=0.2), steps=3, tracker=tr)
        self.assertFalse(tr.latched)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_pan(self):
        rim = (L.HOME_X0 + NEAR_RIM_X + L.PAN_D / 2, RIM_Y, L.TOP + L.RAIL1_TOP + 0.0245)
        arms = {"left": arm(rim, 0.085), "right": FAR["right"]}
        r = run(home_pan(), arms=arms)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["released_1s"])

    def test_hand_clear(self):
        near = (L.HOME_X0 - 0.15, RIM_Y, L.TOP + L.RAIL1_TOP + 0.10)
        self.assertEqual(run(home_pan(), arms={"left": arm(near), "right": FAR["right"]}).score, 1.0)

    def test_finger_zone_edge(self):
        """A closed hand's zone is 15.7 mm round its axis: 25 mm from the pan is not clear, 50 mm is."""
        base = np.array([L.HOME_X0 - 0.06, 0.0, L.TOP + L.RAIL1_TOP + 0.013 + 0.012])
        for gap, want in ((0.025, 0.0), (0.050, 1.0)):
            tip_x = L.HOME_X0 - gap
            tcp = np.array([tip_x, RIM_Y, L.TOP + L.RAIL1_TOP + 0.012 + 0.013])
            a = rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.0)
            self.assertEqual(run(home_pan(), arms={"left": a, "right": FAR["right"]}).score, want, gap)

    def test_parked_hands_are_clear(self):
        self.assertEqual(run(home_pan()).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(home_pan(), arms=None).score, 0.0)
        self.assertEqual(run(home_pan(), arms={"left": FAR["left"]}).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        rim = (L.HOME_X0 + NEAR_RIM_X + L.PAN_D / 2, RIM_Y, L.TOP + L.RAIL1_TOP + 0.0245)
        run(home_pan(), arms={"left": arm(rim, 0.085), "right": FAR["right"]}, steps=5, tracker=tr)
        self.assertEqual(run(home_pan(), tracker=tr, steps=WIN - 1).score, 0.0)
        self.assertEqual(tr.update(home_pan(), FAR).score, 1.0)

    def test_moving_pan(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):  # creeping 0.5 mm a step: 10 mm over the window, still on every rule
            res = tr.update(pan_at(L.HOME_X0 - 0.020 + 0.0005 * i), FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_just_inside_the_still_window(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):  # 0.2 mm a step: 4 mm over the window
            res = tr.update(pan_at(L.HOME_X0 - 0.020 + 0.0002 * i), FAR)
        self.assertEqual(res.score, 0.5)

    def test_turning_pan(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):  # 0.04 degrees a step about the vertical: 0.8 degrees over the window
            res = tr.update(home_pan(yaw=0.04 * i * DEG), FAR)
        self.assertEqual(res.score, 0.0)

    def test_slow_small_turn_is_still(self):
        tr = S.Tracker()
        for i in range(WIN + 3):  # 0.2 degrees a step: 4 degrees over the window, under the 5 degree limit
            tr.update(home_pan(yaw=0.2 * i * DEG), FAR)
        self.assertLessEqual(tr.motion()["turn"], L.STILL_TURN_DEG)

    def test_still_limits_are_5_mm_and_5_degrees(self):
        self.assertEqual((L.STILL_MOVE, L.STILL_TURN_DEG), (0.005, 5.0))

    def test_rocking_pan_is_not_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 3):
            res = tr.update(home_pan(roll=(3.0 if i % 2 else -3.0) * DEG), FAR)
        self.assertEqual(res.score, 0.0)


class Exploits(unittest.TestCase):
    def test_holding_the_pan_home_earns_nothing(self):
        """Hands that keep the pan against the back wall (or hover 20 mm from it) never release: 0."""
        rim = (L.HOME_X0 + NEAR_RIM_X + L.PAN_D / 2, RIM_Y, L.TOP + L.RAIL1_TOP + 0.0245)
        for opening in (0.085, 1.0):
            arms = {"left": arm(rim, opening), "right": arm((rim[0], -rim[1], rim[2]), opening)}
            self.assertEqual(run(home_pan(), arms=arms).score, 0.0)
        hover = (L.HOME_X0 + 0.100, RIM_Y, L.TOP + L.RAIL1_TOP + 0.080)
        self.assertEqual(run(home_pan(), arms={"left": arm(hover), "right": FAR["right"]}).score, 0.0)

    def test_pan_left_on_the_upper_rails_earns_nothing(self):
        self.assertEqual(run(home_pan(z=L.RAIL2_TOP)).score, 0.0)

    def test_pan_resting_on_one_rail_and_a_wall_earns_nothing(self):
        self.assertEqual(run(home_pan(roll=6 * DEG, z=0.085)).score, 0.0)

    def test_doing_nothing_earns_nothing(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_pan_on_the_table_beside_the_oven_earns_nothing(self):
        self.assertEqual(run(pan_at(L.HOME_X0, y=0.40, z=0.0)).score, 0.0)

    def test_only_the_final_state_counts_but_a_fall_is_latched(self):
        tr = S.Tracker()
        run(home_pan(), steps=5, tracker=tr)
        run(pan_at(0.3, z=-0.04), steps=2, tracker=tr)
        self.assertEqual(run(home_pan(), tracker=tr).score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update(None, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_state(self):
        bad = mk.state_from_poses(T.PAN_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        r = S.Tracker().update(bad, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_nan_window_restarts_the_second(self):
        tr = S.Tracker()
        run(home_pan(), tracker=tr)
        bad = mk.state_from_poses(T.PAN_PART, (float("inf"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(tr.update(bad, FAR).score, 0.0)
        self.assertEqual(tr.update(home_pan(), FAR).score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(home_pan(), tracker=tr).score, 1.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(pan_at(0.3, z=-0.05), steps=2, tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            st = pan_at(rng.uniform(0.1, 0.5), y=rng.uniform(-0.05, 0.05), z=rng.uniform(-0.02, 0.2),
                        yaw=rng.uniform(-0.5, 0.5), pitch=rng.uniform(-0.3, 0.3), roll=rng.uniform(-0.3, 0.3))
            r = tr.update(st, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))

    def test_success_needs_the_full_score(self):
        for dx in (0.0, 0.01, 0.05):
            r = run(pan_at(L.HOME_X0 - dx))
            self.assertEqual(r.success, r.score == 1.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        self.assertEqual([o.name for o in T.OBJECTS], ["oven", "oven_door", "pan"])
        self.assertIsNone(T.OVEN.mass)
        self.assertAlmostEqual(T.PAN_PART.mass, L.PAN_MASS)
        self.assertEqual(T.PAN_PART.surface_friction(), L.PAN_FRICTION)

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("lift", "push", "carry", "roof", "lip", "grip", "mm", "step", "pitch", "close"):
            self.assertNotIn(word, low)
        self.assertEqual(json.loads((HERE / "task.json").read_text())["instruction"], T.INSTRUCTION)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            self.assertEqual(set(plan), {"left", "right"})
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ + 60, run_limit)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_the_demo_is_within_half_the_run_limit(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        plan = T.plans("")
        self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ, run_limit / 2)

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "short", {"left", "right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def test_both_hands_run_on_one_clock_in_step(self):
        plan = T.plans("")
        self.assertEqual([w.duration for w in plan["left"]], [w.duration for w in plan["right"]])
        self.assertEqual([w.label for w in plan["left"]], [w.label for w in plan["right"]])
        for wl, wr in zip(plan["left"], plan["right"]):
            self.assertAlmostEqual(wl.pos[0], wr.pos[0], places=9)
            self.assertAlmostEqual(wl.pos[1], -wr.pos[1], places=9)
            self.assertAlmostEqual(wl.pos[2], wr.pos[2], places=9)

    def test_grip_geometry(self):
        self.assertAlmostEqual(-2 * T.GRIP, L.BEAD_W - 0.008)  # 8 mm commanded across the 16 mm bead
        self.assertAlmostEqual(T.grip_z(0.0) - T.PAD, T.TIPS_UP)
        self.assertLess(T.TIPS_UP, L.PAN_H - L.BEAD_H)  # the pad tips start under the bead
        self.assertGreater(T.TIPS_UP, L.PAN_FT + 0.005)  # and clear the pan's floor by more than the contact offset
        self.assertLess(-T.RELEASE * 2, 0.030)
        self.assertGreater(-T.RELEASE * 2, L.BEAD_W + 0.008)  # the let-go gap clears the bead by 5 mm each side
        self.assertAlmostEqual(T.end_x(L.PAN_X0) - L.PAN_X0, L.END_GRIP)
        self.assertAlmostEqual(L.SIDE_Y, L.PAN_W / 2 - L.BEAD_W / 2)
        # the pads (10 mm each side of the TCP along x) are on the side bead, clear of the near rim's bead
        self.assertGreater(L.END_GRIP - 0.010, L.BEAD_W)
        self.assertEqual(T.YAW_END, 0.0)  # closing along y, across the side bead
        side = [s for s in T.PAN_PART.shapes if abs(s.size[2] - L.BEAD_H) < 1e-9 and abs(s.center[1]) > 0.1]
        self.assertEqual(len(side), 2)
        for s in side:
            self.assertAlmostEqual(abs(s.center[1]), L.SIDE_Y)
            self.assertAlmostEqual(s.size[1], L.BEAD_W)
            self.assertLess(s.center[0] - s.size[0] / 2, L.GRIP_X - 0.010)

    def test_carry_keeps_the_hands_out_of_the_oven(self):
        """The hands let go with the gripper body (about 40 mm each side of the TCP along x when it closes along y)
        15 mm or more outside the face, and the pan's centre is 25 mm or more inside it (it will not tip)."""
        xe = T.end_x(T.STOP_X0)
        self.assertLessEqual(xe + 0.040 + 0.015, L.OVEN_X + 1e-9)
        self.assertGreater(T.STOP_X0 + L.PAN_D / 2, L.OVEN_X + 0.025)
        # carried 27 mm over the rails: the pan's underside clears them, its rim clears the upper rails by 8 mm or more
        self.assertGreaterEqual(T.CARRY_U - L.RAIL1_TOP, 0.027 - 1e-9)
        self.assertGreater(L.RAIL2_TOP - L.RAIL_T - (T.CARRY_U + L.PAN_H), 0.008)

    def test_hands_22_cm_apart(self):
        plan = T.plans("")
        for i, (wl, wr) in enumerate(zip(plan["left"], plan["right"])):
            self.assertGreaterEqual(wl.pos[1] - wr.pos[1], 0.20, wl.label)

    def test_push_geometry(self):
        """The closed, pitched hand's front-lower pad corner is on the trailing face 12 mm over the pan's underside,
        and the push ends 6 mm into the back wall's stop."""
        for yaw in (T.YAW_L, T.YAW_R):
            tcp = T.push_tcp(yaw, 0.30, T.PUSH_UP, 0.11)
            rot = tiltgrip.rotation(yaw, T.PUSH_PITCH)
            c = rot[:, 0] * (1.0 if rot[0, 0] >= 0 else -1.0)
            corner = tcp + T.PAD * rot[:, 2] + T.PAD_THICK * c
            self.assertAlmostEqual(float(corner[0]), 0.30, places=9)
            self.assertAlmostEqual(float(corner[2]), L.RAIL1_TOP + T.PUSH_UP, places=9)
            self.assertAlmostEqual(float(corner[1]), 0.11, places=9)
            pad = np.array([tcp + rot[:, 2] * a + c * t for a in (-0.025, 0.013) for t in (0.0, 0.008)])
            self.assertLessEqual(float(pad[:, 0].max()), 0.30 + 1e-9)  # nothing of the block is ahead of the corner
            self.assertGreater(float(pad[:, 2].min()), L.FLOOR_H)  # clear of the oven floor
        self.assertAlmostEqual(T.OVERTRAVEL, 0.010)
        self.assertLessEqual(T.PUSH_UP + 0.0, L.PAN_H)  # on the 25 mm wall
        self.assertGreater(T.PUSH_UP, L.PAN_FT)

    def test_push_hand_stays_clear_of_the_roof(self):
        """The pitched hand's housing is behind the pad: at the end of the push everything of the hand that is under
        the roof is the pad block, below the upper rails and between the rails (|y| 0.11 + 0.0135 < 0.154)."""
        self.assertLess(T.PUSH_Y + rigstate.FINGER_HALF_WIDTH, L.RAIL_Y)
        end = T.push_tcp(T.YAW_L, L.HOME_X0 + T.OVERTRAVEL, T.PUSH_UP, T.PUSH_Y)
        rot = tiltgrip.rotation(T.YAW_L, T.PUSH_PITCH)
        for back in (0.0, 0.025, 0.07, 0.144):
            housing = end - rot[:, 2] * back
            if housing[0] > L.OVEN_X:
                self.assertLess(float(housing[2]), L.RAIL2_TOP, back)

    def test_variants(self):
        self.assertEqual(set(T.VARIANTS), {"short", "half"})
        for name, out in (("short", 0.020), ("half", 0.070)):
            for side, wps in T.plans(name).items():
                push = [w for w in wps if w.label.startswith("push")]
                self.assertEqual(len(push), 1)
                yaw = T.YAW_L if side == "left" else T.YAW_R
                want = T.push_tcp(yaw, L.OVEN_X - out, T.PUSH_UP, (1 if side == "left" else -1) * T.PUSH_Y)
                self.assertTrue(np.allclose(np.array(push[0].pos) - [0, 0, L.TOP], want))


class OfflineModel(unittest.TestCase):
    """The offline stand-in for the physics (never used in the simulator)."""

    def setUp(self):
        self.m = T.OfflineModel()

    def hands(self, x, z, opening=0.085, dy=0.0):
        tcp = lambda side: np.array([x, side * (L.SIDE_Y + dy), L.TOP + z])  # noqa: E731
        return {"left": arm(tcp(1), opening), "right": arm(tcp(-1), opening)}

    def move(self, x0, z0, x1, z1, opening=0.085, steps=40):
        """Both hands from (x0, z0) to (x1, z1) in small steps, as a control loop sees them."""
        for i in range(steps + 1):
            f = i / steps
            self.m.step(self.hands(x0 + f * (x1 - x0), z0 + f * (z1 - z0), opening))

    def test_both_hands_carry_the_pan_by_its_ends(self):
        g = T.end_x(L.PAN_X0)
        self.move(g, 0.0245, g, 0.1)
        self.move(g, 0.1, g + 0.1, 0.1)
        self.assertAlmostEqual(self.m.x, L.PAN_CX + 0.1, places=6)
        self.assertAlmostEqual(self.m.z, 0.1 - 0.0245, places=6)

    def test_hands_on_the_near_rim_do_not_carry_it(self):
        """Only the two short ends are a carrying grip here (the demo's); hands elsewhere leave it where it is."""
        lip = L.PAN_X0 + L.BEAD_W / 2
        for z in (0.0245, 0.1):
            self.m.step({"left": arm((lip, RIM_Y, L.TOP + z), 0.085), "right": arm((lip, -RIM_Y, L.TOP + z), 0.085)})
        self.assertEqual(self.m.z, 0.0)

    def test_one_hand_does_not_hold_it(self):
        g = T.end_x(L.PAN_X0)
        hands = self.hands(g, 0.0245)
        hands["right"] = FAR["right"]
        for z in (0.0245, 0.1):
            hands["left"] = arm((g, L.SIDE_Y, L.TOP + z), 0.085)
            self.m.step(hands)
        self.assertEqual(self.m.z, 0.0)

    def test_open_hands_do_not_hold_it(self):
        g = T.end_x(L.PAN_X0)
        self.m.step(self.hands(g, 0.0245, opening=1.0))
        self.m.step(self.hands(g, 0.1, opening=1.0))
        self.assertEqual(self.m.z, 0.0)

    def test_let_go_over_the_rails_drops_onto_them(self):
        xe = T.end_x(T.STOP_X0)
        g = T.end_x(L.PAN_X0)
        top = L.RAIL1_TOP + 0.015 + 0.0245
        self.move(g, 0.0245, g, top)
        self.move(g, top, xe, top)
        self.m.step(self.hands(xe, top, opening=0.3))
        self.assertAlmostEqual(self.m.z, L.RAIL1_TOP)
        self.assertAlmostEqual(self.m.x, T.STOP_X0 + L.PAN_D / 2, places=6)

    def test_let_go_in_front_drops_to_the_table(self):
        g = T.end_x(L.PAN_X0)
        self.move(g, 0.0245, g, 0.1)
        self.move(g, 0.1, g + 0.05, 0.1)
        self.m.step(self.hands(g + 0.05, 0.1, opening=0.3))
        self.assertEqual(self.m.z, 0.0)

    def test_let_go_at_the_upper_rails(self):
        xe = T.end_x(T.STOP_X0)
        g = T.end_x(L.PAN_X0)
        top = L.RAIL2_TOP + 0.004 + 0.0245
        self.move(g, 0.0245, g, top)
        self.move(g, top, xe, top)
        self.m.step(self.hands(xe, top, opening=0.3))
        self.assertAlmostEqual(self.m.z, L.RAIL2_TOP)

    def pusher(self, x_corner, up=T.PUSH_UP, y=0.11):
        out = {}
        for side, yaw in (("left", T.YAW_L), ("right", T.YAW_R)):
            tcp = T.push_tcp(yaw, x_corner, up, (1 if side == "left" else -1) * y)
            rot = tiltgrip.rotation(yaw, T.PUSH_PITCH)
            out[side] = rigstate.Arm(tcp + np.array([0.0, 0.0, L.TOP]), tcp + np.array([0.0, 0.0, L.TOP]) + rot[:, 2] * T.PAD,
                                     0.0)
        return out

    def test_a_pitched_closed_hand_pushes_the_pan_in(self):
        self.m.x, self.m.z = T.STOP_X0 + L.PAN_D / 2, L.RAIL1_TOP
        self.m.step(self.pusher(T.STOP_X0 - 0.010))
        self.assertAlmostEqual(self.m.x, T.STOP_X0 + L.PAN_D / 2)  # not touching yet
        self.m.step(self.pusher(T.STOP_X0 + 0.030))
        self.assertAlmostEqual(self.m.x, T.STOP_X0 + 0.030 + L.PAN_D / 2, places=9)

    def test_the_back_wall_stops_it(self):
        self.m.x, self.m.z = T.STOP_X0 + L.PAN_D / 2, L.RAIL1_TOP
        for x in np.linspace(T.STOP_X0 - 0.010, L.HOME_X0 + T.OVERTRAVEL, 30):
            self.m.step(self.pusher(float(x)))
        self.assertAlmostEqual(self.m.x, L.HOME_CX, places=9)

    def test_a_hand_above_the_pan_does_not_push(self):
        self.m.x, self.m.z = T.STOP_X0 + L.PAN_D / 2, L.RAIL1_TOP
        self.m.step(self.pusher(T.STOP_X0 + 0.03, up=0.06))
        self.assertAlmostEqual(self.m.x, T.STOP_X0 + L.PAN_D / 2)

    def test_a_straight_down_closed_hand_does_not_push(self):
        self.m.x, self.m.z = T.STOP_X0 + L.PAN_D / 2, L.RAIL1_TOP
        tcp = np.array([T.STOP_X0 + 0.02, 0.11, L.TOP + L.RAIL1_TOP + 0.03])
        self.m.step({"left": arm(tcp, 0.0), "right": arm(tcp * [1, -1, 1], 0.0)})
        self.assertAlmostEqual(self.m.x, T.STOP_X0 + L.PAN_D / 2)

    def test_state_at_rest_matches_the_spawn(self):
        a, b = self.m.state(), T.PAN_PART.rest_state()
        self.assertTrue(np.allclose(a.pos, b.pos))
        self.assertTrue(np.allclose(a.origin, b.origin))

    def test_unreadable_arms_change_nothing(self):
        self.m.step(None)
        self.m.step({"left": FAR["left"]})
        self.assertEqual((self.m.x, self.m.z), (L.PAN_CX, 0.0))


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("x 0.425 m", "0.344 m wide", "0.245 m deep", "x 0.670 m", "0.020 m above", "0.190 m above",
                      "18 mm deep and 4 mm thick", "z 0.075 m", "z 0.140 m", "0.308 m apart", "9-1/2 x 13 x 1 in",
                      "0.241 m deep", "0.330 m wide", "0.20 kg", "16 mm wide", "x 0.008 m", "x 0.249 m",
                      "6 mm short", "11 mm", "7 mm from each wall", "3.7 mm shorter", "x 0.255 m", "open",
                      "0.170 m long"):
            self.assertIn(token, setup)
        for token in ("3 mm below or 12 mm above", "z 0.075 m", "2 mm or more along 50 mm", "30 mm inside",
                      "2 degrees", "no part of the pan in front of the front face", "40 mm in front", "30 mm or more",
                      "under 5 mm", "5 degrees", "no visible motion", "20 mm below the table top", "first rule that holds",
                      "Possible scores: 0, 0.25, 0.5, 1.0", "1.0 level and home", "0.5 level, not home, near",
                      "0.25 otherwise", "1.0 held for 1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        self.assertAlmostEqual(L.OVEN_X, 0.425)
        self.assertAlmostEqual(L.BACK_X, 0.670)
        self.assertAlmostEqual(L.PAN_X0 + L.PAN_D, 0.249, places=3)
        self.assertAlmostEqual(L.OVEN_X - L.DOOR_L, 0.255)
        scores = d.get("scores", "")  # before the GPU run: TBD; after it: the real Dream-1 run and demo 1.0
        self.assertTrue(scores.startswith("TBD") or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?",
                                                                 scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_simplifications(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("Rigid bodies", "4 mm thick", "bead", "est", "door", "heating elements", "wire rack"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("every week", "two short ends", "near rim", "door shut", "rail"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertLess(len(d["instruction"]), 300)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tiltgrip.py", "rigstate.py",
                  "mechkit.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("—|–|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                if f.startswith("tests/") or (f == "README.md" and line.startswith("|")):  # tables run long
                    continue
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
