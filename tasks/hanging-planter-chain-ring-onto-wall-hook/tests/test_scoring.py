"""Offline tests for layout.py, scoring.py and task.py: synthetic planter states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
TOP = L.TOP
SPEC = T.PLANTER_PART


def ry(theta):
    return mk.quat_from_ypr(0.0, theta, 0.0)


def planter(ring=(0.31, L.HOOK_Y, L.RING_HUNG_Z), tilt=0.0, yaw=0.0, ring_roll=0.0, bail_extra=0.0, nan=False):
    """The planter hanging plumb from a ring centre ``ring`` (table-top frame): bail and ring upright, the pot turned
    ``tilt`` rad about y on the bail's ear axis (the bail joint takes the turn out), ``ring_roll`` the ring's own
    joint about x, ``yaw`` the whole thing turned about z."""
    cx, cy, cz = ring
    apex = np.array([cx, cy, cz - L.RING_DZ])
    ear = apex - np.array([0.0, 0.0, L.BAIL_RISE])
    r = mk.quat_to_matrix(ry(tilt))
    origin = ear - r @ np.array([0.0, 0.0, L.EAR_Z])
    q = mk.quat_from_ypr(yaw, tilt, 0.0)
    if yaw:  # turn about the vertical through the apex
        rz = mk.quat_to_matrix(mk.quat_from_ypr(yaw, 0.0, 0.0))
        origin = apex + rz @ (origin - apex)
    joints = {"bail": -tilt + bail_extra, "ring": ring_roll}
    pose = (origin[0], origin[1], TOP + origin[2], *q)
    if nan:
        pose = (float("nan"),) + tuple(pose[1:])
    return mk.state_from_poses(SPEC, pose, joints=joints, source="test")


def standing(x=L.POT_X, y=L.POT_Y, z=0.0, tilt=0.0):
    """The planter standing on the table as at the start (or ``z`` up, or lying turned ``tilt`` about y)."""
    return mk.state_from_poses(SPEC, (x, y, TOP + z, *ry(tilt)), source="test")


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


FAR = {"left": arm((0.10, 0.45, TOP + 0.30)), "right": arm((0.10, -0.45, TOP + 0.30))}


def near(side="left"):
    return {**FAR, side: arm((0.31, L.HOOK_Y + 0.02, TOP + 0.18), 0.45)}


def run(state, arms=FAR, n=WIN + 6, tracker=None):
    t = tracker or S.Tracker()
    res = None
    for _ in range(n):
        res = t.update({S.PLANTER: state}, arms)
    return t, res


class LayoutTests(unittest.TestCase):
    def test_hung_geometry(self):
        self.assertAlmostEqual(L.RING_HUNG_Z, 0.174, places=6)
        self.assertAlmostEqual(L.HANG_BASE_Z, 0.0275, places=6)
        self.assertGreaterEqual(L.HANG_BASE_Z - L.CLEAR, 0.007)  # 7.5 mm to spare over the 20 mm bar
        self.assertAlmostEqual(L.RING_Z0, 0.1465, places=6)
        self.assertAlmostEqual(L.POT_H, 0.080)

    def test_ring_lip_clearance(self):
        """The lip's top and the bar stand inside the hole with the ring's centre 4 mm over the bar's axis (threading),
        and cannot when the ring hangs (the ring cannot come back over the lip)."""
        for dz, ok in ((L.RING_THREAD_Z - L.HOOK_Z, True), (L.RING_HUNG_DZ, False)):
            lo, hi = dz - L.RING_IN, dz + L.RING_IN
            inside = lo <= -L.BAR_R and hi >= L.BAR_R + L.LIP_UP
            self.assertEqual(inside, ok)
        self.assertGreaterEqual(L.RING_IN - (L.RING_THREAD_Z - L.HOOK_Z) - L.BAR_R, 0.010)  # 12 mm below the bar
        self.assertGreaterEqual(L.RING_IN + (L.RING_THREAD_Z - L.HOOK_Z) - L.BAR_R - L.LIP_UP, 0.0099)  # 10 mm over the lip

    def test_spec_geometry_through_the_hole(self):
        """From the declared shapes: the bar's axis is 4 mm (the bar's radius) from the ring's wire when hung, and the
        lip's top is clear of the ring's wire while threading."""
        hung = planter()
        d = hung.distance(np.array([[0.31, L.HOOK_Y, TOP + L.HOOK_Z]]), bodies=("ring",))
        self.assertTrue(np.all(np.abs(d - L.BAR_R) < 0.0005), d)
        thread = planter(ring=(0.31, L.HOOK_Y, L.RING_THREAD_Z))
        lip = np.array([[0.31, L.HOOK_Y, TOP + L.LIP_Z1], [0.31, L.HOOK_Y, TOP + L.LIP_Z0]])
        self.assertTrue(np.all(thread.distance(lip, bodies=("ring",)) - L.LIP_R >= 0.005))
        bar_bottom = np.array([[0.31, L.HOOK_Y, TOP + L.HOOK_Z - L.BAR_R]])
        self.assertGreaterEqual(float(thread.distance(bar_bottom, bodies=("ring",))[0]), 0.008)

    def test_bail_does_not_overlap_the_pot_rim(self):
        s = standing()
        self.assertGreaterEqual(L.BAIL_Y - L.BAIL_WIRE / 2, L.RIM_R)
        # the bail's side wires clear the pot's cylinders
        pts = np.array([[L.POT_X, L.POT_Y + sy * (L.BAIL_Y - L.BAIL_WIRE / 2), TOP + z]
                        for sy in (-1, 1) for z in (L.EAR_Z, L.APEX_Z)])
        self.assertTrue(np.all(s.distance(pts, bodies=("base",)) > 0.0005))

    def test_hand_clearances_at_the_grasp(self):
        """Pad tips (13 mm under the TCP) over the bail's top bar when the gripper holds the ring."""
        tip_z = T.GRASP_TCP - T.PAD_TIP
        self.assertGreaterEqual(tip_z - (L.APEX_Z + L.BAIL_WIRE / 2), 0.010)
        self.assertAlmostEqual(2 * (L.RING_APOTHEM + L.RING_WIRE / 2), 0.050)  # 50 mm across the flats
        self.assertTrue(0.016 <= T.RING_OD <= 0.080)  # the grip range
        self.assertLess((L.POT_MASS + L.BAIL_MASS + L.RING_MASS) * L.G, 5.0)  # an arm's lift (about 5 N)

    def test_reach(self):
        """The TCP heights above the table at the threading and the hang (the reach table: 0.215 at x 0.30, y 0.20)."""
        self.assertLessEqual(T.TRANSIT_TCP, 0.205)
        self.assertLessEqual(T.GRASP_TCP, 0.185)
        self.assertGreaterEqual(L.WALL_X0 - (0.31 + 0.033), 0.05)  # the gripper housing stays 5 cm off the panel

    def test_wall_and_start(self):
        self.assertGreater(L.POT_X - L.RIM_R, 0.19)  # nothing under the arms' start pose (x < 0.2 near the bases)
        self.assertLess(L.POT_Y + L.RIM_R + L.BAIL_Y, 0.45)
        self.assertGreater(L.BAR_X0, L.POT_X + 0.02)  # the ring starts short of the lip
        self.assertAlmostEqual(L.BAR_X1 - L.BAR_X0, 0.150, places=6)
        self.assertGreater(L.WALL_X0 - L.PLATE_X0, 0)
        self.assertLessEqual(L.HOOK_Z + L.PLATE_H / 2, L.WALL_H)

    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["wall", "planter"])
        self.assertIsNone(T.OBJECTS[0].mass)
        self.assertAlmostEqual(T.PLANTER_PART.mass, 0.25)
        self.assertEqual([j.name for j in T.PLANTER_PART.joints], ["bail", "ring"])
        self.assertEqual(len(T.RING.shapes), 8)


class TierTests(unittest.TestCase):
    def test_full_score(self):
        t, res = run(planter())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["on_bar"] and res.details["clear"] and res.details["plumb"])
        self.assertAlmostEqual(res.details["pot_low_mm"], 27.5, places=1)
        self.assertAlmostEqual(res.details["bar_off_mm"], 16.0, places=1)

    def test_ring_anywhere_on_the_bar_behind_the_lip(self):
        for x, want in ((0.290, 1.0), (0.31, 1.0), (0.37, 1.0), (0.40, 0.25), (0.42, 0.0)):
            _, res = run(planter(ring=(x, L.HOOK_Y, L.RING_HUNG_Z)))
            self.assertEqual(res.score, want, (x, res.details))  # 0.40: the pot's rim is within 3 mm of the panel

    def test_half_for_a_leaning_pot(self):
        _, res = run(planter(tilt=math.radians(10.5)))
        self.assertEqual((res.score, res.success), (0.5, False))
        self.assertFalse(res.details["plumb"])
        self.assertTrue(res.details["clear"])
        _, res = run(planter(tilt=math.radians(9)))
        self.assertEqual(res.score, 1.0)

    def test_half_for_a_swinging_pot(self):
        t = S.Tracker()
        res = None
        for i in range(WIN + 10):
            tilt = math.radians(8) * math.sin(i / 3.0)  # a swing of 8 degrees: plumb at the end of the window or not
            res = t.update({S.PLANTER: planter(tilt=tilt)}, FAR)
        self.assertFalse(res.details["still_1s"])
        self.assertEqual(res.score, 0.5)

    def test_quarter_for_a_pot_swung_down_to_the_table(self):
        """The ring on the bar but the pot's rim within 20 mm of the table (the pot hanging 20 degrees out): 0.25."""
        _, res = run(planter(tilt=math.radians(20)))
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["clear"])
        self.assertLess(res.details["pot_low_mm"], 20.0)

    def test_zero_pot_on_the_table_ring_off_the_bar(self):
        _, res = run(standing())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["on_bar"])

    def test_nothing_scores_zero(self):
        t = S.Tracker()
        self.assertEqual(t.update({S.PLANTER: standing()}, FAR).score, 0.0)


class OnBarTests(unittest.TestCase):
    def check(self, state, expect_on):
        _, res = run(state)
        self.assertEqual(res.details["on_bar"], expect_on, res.details)
        self.assertEqual(res.score > 0, expect_on)

    def test_in_front_of_the_lip(self):
        self.check(planter(ring=(0.265, L.HOOK_Y, L.RING_HUNG_Z)), False)
        self.check(planter(ring=(0.270, L.HOOK_Y, L.RING_THREAD_Z)), False)

    def test_on_the_lip_itself(self):
        self.check(planter(ring=(L.LIP_X, L.HOOK_Y, L.RING_HUNG_Z)), False)

    def test_just_behind_the_lip(self):
        self.check(planter(ring=(L.BAR_BEHIND_LIP + 0.001, L.HOOK_Y, L.RING_HUNG_Z)), True)

    def test_held_up_beside_the_bar(self):
        """The ring held 25 mm over the bar's axis (the bar below the hole), or 30 mm high: not hanging."""
        self.check(planter(ring=(0.31, L.HOOK_Y, L.HOOK_Z + 0.025)), False)
        self.check(planter(ring=(0.31, L.HOOK_Y, L.HOOK_Z + 0.004)), False)  # threading height: held, not hung
        self.check(planter(ring=(0.31, L.HOOK_Y, L.HOOK_Z - L.HUNG_BELOW + 0.001)), False)
        self.check(planter(ring=(0.31, L.HOOK_Y, L.HOOK_Z - L.HUNG_BELOW - 0.001)), True)

    def test_off_to_the_side(self):
        self.check(planter(ring=(0.31, L.HOOK_Y + 0.022, L.RING_HUNG_Z)), False)
        self.check(planter(ring=(0.31, L.HOOK_Y + 0.010, L.RING_HUNG_Z)), True)

    def test_ring_edge_on_to_the_bar(self):
        self.check(planter(yaw=math.radians(80)), False)
        self.check(planter(yaw=math.radians(30)), True)

    def test_past_the_plate(self):
        self.check(planter(ring=(L.BAR_X1 + 0.002, L.HOOK_Y, L.RING_HUNG_Z)), False)

    def test_ring_rolled_about_its_axis_still_on(self):
        self.check(planter(ring_roll=0.15), True)
        self.check(planter(ring_roll=0.5), False)  # swung out of the bar's line (the ring's centre moves 11 mm)


class WallTests(unittest.TestCase):
    def test_pot_touching_the_panel_is_not_clear(self):
        x = L.WALL_X0 - L.RIM_R + 0.001  # the rim's far edge 1 mm into the panel's face
        _, res = run(planter(ring=(x, L.HOOK_Y, L.RING_HUNG_Z)))
        self.assertFalse(res.details["clear"])
        self.assertEqual(res.score, 0.25)

    def test_pot_clear_of_the_panel(self):
        x = L.WALL_X0 - L.RIM_R - 0.004
        _, res = run(planter(ring=(x, L.HOOK_Y, L.RING_HUNG_Z)))
        self.assertTrue(res.details["clear"])
        self.assertEqual(res.score, 1.0)


class ReleaseTests(unittest.TestCase):
    def test_hand_on_the_ring_scores_zero(self):
        t, res = run(planter(), near())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_hand_30_mm_away_is_released_and_29_is_not(self):
        s = planter()

        def hand_at(gap):
            """An open left hand over the pot whose finger zone is ``gap`` from the planter, found by bisection on x."""
            lo, hi = 0.0, 0.31
            for _ in range(60):
                mid = (lo + hi) / 2
                a = {"left": arm(np.array([mid, L.HOOK_Y, TOP + L.HOOK_Z + 0.06])), "right": FAR["right"]}
                if rigstate.hands_gap(a, s.distance) < gap:
                    hi = mid
                else:
                    lo = mid
            return {"left": arm(np.array([lo, L.HOOK_Y, TOP + L.HOOK_Z + 0.06])), "right": FAR["right"]}

        for gap, ok in ((0.0302, True), (0.0298, False)):
            a = hand_at(gap)
            self.assertAlmostEqual(rigstate.hands_gap(a, s.distance), gap, places=4)
            _, res = run(s, a)
            self.assertEqual(res.details["released_1s"], ok, (gap, res.details["hands_gap_mm"]))

    def test_right_hand_counts_too(self):
        _, res = run(planter(), near("right"))
        self.assertEqual(res.score, 0.0)

    def test_hands_unreadable(self):
        for arms in (None, {"left": FAR["left"]}, {"left": arm((float("nan"), 0, 0)), "right": FAR["right"]}):
            _, res = run(planter(), arms)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.details["released_1s"])

    def test_released_only_after_a_whole_second(self):
        t = S.Tracker()
        for _ in range(WIN - 1):
            res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 0.0)
        res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 0.5)  # 20 steps released but the stillness window needs 21 readings
        res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 1.0)

    def test_hand_touches_for_one_step_resets_the_second(self):
        t, _ = run(planter(), n=WIN + 5)
        res = t.update({S.PLANTER: planter()}, near())
        self.assertEqual(res.score, 0.0)
        for _ in range(WIN - 1):
            res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 0.0)
        res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 1.0)


class StillnessTests(unittest.TestCase):
    def moving(self, drift_mm=0.0, tilt_deg=0.0, ring_drift_mm=0.0, ring_roll=0.0, yaw_deg=0.0):
        t = S.Tracker()
        res = None
        for i in range(WIN + 1):
            f = i / WIN
            ring = (0.31 + ring_drift_mm / 1000 * f, L.HOOK_Y, L.RING_HUNG_Z)
            s = planter(ring=ring, tilt=math.radians(tilt_deg) * f, ring_roll=ring_roll * f, yaw=math.radians(yaw_deg) * f)
            if drift_mm:
                o = s.origin
                s = mk.state_from_poses(SPEC, (o[0] + drift_mm / 1000 * f, o[1], o[2], *s.quat_xyzw),
                                        joints=s.joints, source="test")
            res = t.update({S.PLANTER: s}, FAR)
        return res

    def test_pot_move_limit(self):
        self.assertEqual(self.moving(drift_mm=4.5).score, 1.0)
        res = self.moving(drift_mm=5.5)
        self.assertEqual((res.score, res.details["still_1s"]), (0.5, False))

    def test_pot_turn_limit(self):
        self.assertEqual(self.moving(yaw_deg=4.5).score, 1.0)  # turning about the vertical through the ring: no move
        res = self.moving(yaw_deg=5.5)
        self.assertEqual(res.score, 0.5)
        self.assertGreater(res.details["motion"]["pot_turn_deg"], 5.0)

    def test_pot_tilt_moves_its_origin_more_than_5_mm_first(self):
        """The pot turns about its ears 76 mm above its origin: 3 degrees move the origin 4 mm (still), 4.5 degrees 6 mm."""
        self.assertEqual(self.moving(tilt_deg=3.0).score, 1.0)
        res = self.moving(tilt_deg=4.5)
        self.assertEqual(res.score, 0.5)
        self.assertGreater(res.details["motion"]["pot_move_mm"], 5.0)
        self.assertLess(res.details["motion"]["pot_turn_deg"], 5.0)

    def test_ring_move_limit(self):
        self.assertEqual(self.moving(ring_drift_mm=4.5).score, 1.0)
        self.assertEqual(self.moving(ring_drift_mm=5.8).score, 0.5)

    def test_ring_turn_limit(self):
        self.assertEqual(self.moving(ring_roll=math.radians(4.5)).score, 1.0)
        self.assertEqual(self.moving(ring_roll=math.radians(5.5)).score, 0.5)

    def test_pot_swinging_under_a_still_ring_is_not_full_score(self):
        """The review copy's point: the ring and the pot's tilt at the end can look still while the pot swings. The pot's
        own origin and turn are read over the whole second."""
        t = S.Tracker()
        res = None
        for i in range(WIN + 1):
            tilt = math.radians(8.0) * math.sin(2 * math.pi * i / WIN)  # a full period: ends level, plumb, where it began
            res = t.update({S.PLANTER: planter(tilt=tilt)}, FAR)
        self.assertTrue(res.details["plumb"])
        self.assertAlmostEqual(res.details["ring_mm"][0], 310.0, places=1)
        self.assertFalse(res.details["still_1s"])
        self.assertEqual(res.score, 0.5)

    def test_a_pendulum_beyond_5_mm_even_if_it_returns(self):
        t = S.Tracker()
        res = None
        for i in range(WIN + 1):
            s = planter(tilt=math.radians(3.0) * math.sin(math.pi * i / WIN))
            res = t.update({S.PLANTER: s}, FAR)
        self.assertEqual(res.score, 1.0)  # 3 degrees about the ears: the pot's origin moves under 5 mm, turns under 5

    def test_a_slow_creep_over_5_mm(self):
        self.assertEqual(self.moving(drift_mm=5.2).score, 0.5)

    def test_window_slides(self):
        t = S.Tracker()
        for i in range(WIN + 1):  # swinging through the first half
            t.update({S.PLANTER: planter(tilt=math.radians(10) * math.sin(i))}, FAR)
        res = None
        for _ in range(WIN + 2):
            res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 1.0)

    def test_too_few_readings(self):
        _, res = run(planter(), n=WIN)
        self.assertFalse(res.details["still_1s"])
        self.assertEqual(res.score, 0.5)  # released? no: 20 steps released, needs 21 readings for the window
        # (released needs WIN steps; the window needs WIN + 1 readings, so the 21st step is the first 1.0)


class LatchTests(unittest.TestCase):
    def test_falling_off_the_table_latches(self):
        t = S.Tracker()
        t.update({S.PLANTER: standing(z=-0.05)}, FAR)
        self.assertTrue(t.latched)
        res = None
        for _ in range(WIN + 6):
            res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(t.latched)

    def test_lying_on_the_floor(self):
        t, res = run(standing(z=-0.7475))
        self.assertTrue(t.latched)
        self.assertEqual(res.score, 0.0)

    def test_tipped_on_the_table_is_not_a_fall(self):
        t, res = run(standing(tilt=math.radians(90), z=0.046))
        self.assertFalse(t.latched)
        self.assertEqual(res.score, 0.0)

    def test_dropped_and_picked_up_again_stays_zero(self):
        t = S.Tracker()
        t.update({S.PLANTER: standing(z=-0.03)}, FAR)
        for _ in range(3 * WIN):
            res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 0.0)


class InvalidStateTests(unittest.TestCase):
    def test_nan_pose(self):
        _, res = run(planter(nan=True), n=3)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])

    def test_nan_joint(self):
        s = planter()
        bad = mk.state_from_poses(SPEC, (*np.array(s.origin), *s.quat_xyzw), joints={"bail": float("nan")}, source="test")
        _, res = run(bad, n=3)
        self.assertEqual(res.score, 0.0)

    def test_missing_state_and_wrong_type(self):
        t = S.Tracker()
        self.assertEqual(t.update({}, FAR).score, 0.0)
        self.assertEqual(t.update(None, FAR).score, 0.0)
        self.assertEqual(t.update({S.PLANTER: object()}, FAR).score, 0.0)
        self.assertEqual(t.bad_steps, 3)

    def test_a_blow_up_restarts_the_windows(self):
        t, res = run(planter(), n=WIN + 5)
        self.assertEqual(res.score, 1.0)
        t.update({S.PLANTER: planter(nan=True)}, FAR)
        res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 0.0)  # the one-second windows start again
        for _ in range(WIN + 1):
            res = t.update({S.PLANTER: planter()}, FAR)
        self.assertEqual(res.score, 1.0)

    def test_infinite_values(self):
        s = planter()
        inf = mk.state_from_poses(SPEC, (float("inf"), 0.0, 0.8, *s.quat_xyzw), source="test")
        _, res = run(inf, n=3)
        self.assertEqual(res.score, 0.0)


class ExploitTests(unittest.TestCase):
    def test_hold_it_on_the_bar_by_hand(self):
        """Ring hung but a gripper still around it: nothing earned."""
        _, res = run(planter(), near())
        self.assertEqual(res.score, 0.0)

    def test_leave_the_pot_standing_with_the_ring_on_it(self):
        """The start state, hands clear: the ring is on its own bail, far from the bar."""
        _, res = run(standing())
        self.assertEqual(res.score, 0.0)

    def test_pot_hung_by_something_else_ring_dropped_below_the_bar(self):
        """The planter hanging by its bail on the bar with the ring swung out of the way: the ring is not on the bar."""
        s = planter(ring=(0.31, L.HOOK_Y, L.RING_HUNG_Z + 0.060))  # the ring 60 mm over the bar's axis (the bail is on it)
        _, res = run(s)
        self.assertEqual(res.score, 0.0)

    def test_ring_lying_on_the_bar_top_with_the_pot_on_the_table(self):
        """The planter on the table (its bail tipped over), the ring dragged across and left lying on the bar: the ring's
        hole is not on the bar and the ring cannot hang."""
        s = planter(ring=(0.31, L.HOOK_Y, L.HOOK_Z + L.BAR_R + L.RING_APOTHEM + L.RING_WIRE / 2 - 0.005))
        _, res = run(s)
        self.assertEqual(res.score, 0.0)

    def test_pot_leaned_against_the_panel_with_the_ring_loose(self):
        s = standing(x=L.WALL_X0 - L.RIM_R - 0.001, y=L.HOOK_Y, z=0.0)
        _, res = run(s)
        self.assertEqual(res.score, 0.0)

    def test_pot_on_the_floor_but_the_ring_up_the_bar_is_a_fall(self):
        t, res = run(planter(ring=(0.31, L.HOOK_Y, L.RING_HUNG_Z - 0.8)))
        self.assertTrue(t.latched)
        self.assertEqual(res.score, 0.0)

    def test_reset_detected_from_a_jump(self):
        t = S.Tracker()
        run(planter(), tracker=t, n=WIN + 10)
        self.assertEqual(t.resets, 0)
        res = t.update({S.PLANTER: standing()}, FAR)  # the simulator put the planter back on its mark
        self.assertEqual(t.resets, 1)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(t.steps, 1)

    def test_a_pot_slowly_returned_to_its_mark_is_no_reset(self):
        t = S.Tracker()
        t.update({S.PLANTER: standing(x=L.POT_X + 0.03)}, FAR)
        t.update({S.PLANTER: standing()}, FAR)  # 30 mm in one step
        self.assertEqual(t.resets, 0)

    def test_latch_cleared_by_a_reset(self):
        t = S.Tracker()
        t.update({S.PLANTER: standing(z=-0.05)}, FAR)
        self.assertTrue(t.latched)
        t.update({S.PLANTER: planter()}, FAR)
        t.update({S.PLANTER: standing()}, FAR)  # a reset
        self.assertFalse(t.latched)


class TaskTests(unittest.TestCase):
    def test_demo_plans(self):
        os.environ.pop("DEMO_HOLD_ONLY", None)
        os.environ.pop("DEMO_IDLE_ARM", None)
        os.environ.pop("DEMO_VARIANT", None)
        plan = T.demo()
        self.assertEqual(set(plan), {"left"})
        steps = int(round(sum(w.duration for w in plan["left"]) * HZ))
        self.assertLessEqual(steps, 475)  # under half the 1000-step limit with the scorer's second and the 1 s hold
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_IDLE_ARM")
        os.environ["DEMO_VARIANT"] = "bogus"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT")

    def test_gripper_closes_on_the_ring_not_through_it(self):
        self.assertGreater(-T.GRIP * 2, 0.030)
        self.assertLess(-T.GRIP * 2, T.RING_OD)

    def test_offline_model_hangs_the_ring_on_release(self):
        m = T.OfflineModel()
        tcp = np.array([L.POT_X, L.POT_Y, TOP + T.GRASP_TCP])
        arms = {"left": arm(tcp, 0.45), "right": FAR["right"]}
        m.step(arms)
        self.assertIsNotNone(m.grip)
        for z in (0.20, 0.199):
            m.step({"left": arm(np.array([0.31, L.HOOK_Y, TOP + z]), 0.45), "right": FAR["right"]})
        for z in (0.19, 0.18, 0.17):  # the hand lowers past where the ring stops on the bar
            m.step({"left": arm(np.array([0.31, L.HOOK_Y, TOP + z]), 0.45), "right": FAR["right"]})
        self.assertAlmostEqual(m.ring[2], L.RING_HUNG_Z, places=6)
        m.step({"left": arm(np.array([0.31, L.HOOK_Y, TOP + 0.17]), 1.0), "right": FAR["right"]})
        self.assertIsNone(m.grip)
        self.assertAlmostEqual(m.ring[2], L.RING_HUNG_Z, places=6)

    def test_offline_model_drops_the_planter_short_of_the_lip(self):
        m = T.OfflineModel()
        m.step({"left": arm(np.array([L.POT_X, L.POT_Y, TOP + T.GRASP_TCP]), 0.45), "right": FAR["right"]})
        m.step({"left": arm(np.array([0.25, L.HOOK_Y, TOP + 0.199]), 0.45), "right": FAR["right"]})
        m.step({"left": arm(np.array([0.25, L.HOOK_Y, TOP + 0.199]), 1.0), "right": FAR["right"]})
        self.assertAlmostEqual(m.ring[2], L.RING_Z0, places=6)

    def test_offline_demo_end_to_end(self):
        """Play the demo's waypoints through the offline model and the scorer (the checker does this with real IK)."""
        plan = T.left_plan()
        t = S.Tracker()
        m = T.OfflineModel()
        res = None
        steps = 0
        pos = np.array([0.20, 0.30, 0.19])
        opening = 1.0
        for w in plan:
            n = max(1, int(round(w.duration * HZ)))
            target = np.array(w.pos)
            tgt_open = min(1.0, max(0.0, w.finger / -0.04695))
            p0, o0 = pos.copy(), opening
            for i in range(1, n + 1):
                f = i / n
                pos = p0 + (target - p0) * f
                opening = o0 + (tgt_open - o0) * f
                tcp = np.array([pos[0], pos[1], pos[2]])
                arms = {"left": arm(tcp, opening), "right": FAR["right"]}
                m.step(arms)
                res = t.update(m.states(), arms)
                steps += 1
        for _ in range(WIN + 2):  # the policy holds the last pose
            arms = {"left": arm(pos, opening), "right": FAR["right"]}
            m.step(arms)
            res = t.update(m.states(), arms)
        self.assertEqual(res.score, 1.0, res.details)
        self.assertTrue(res.success)


class TextTests(unittest.TestCase):
    def test_field_limits_and_no_forbidden_text(self):
        spec = json.loads((HERE / "task.json").read_text())
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(spec[k]), n, k)
        for f in ("task.json", "README.md", "task.py", "scoring.py", "layout.py"):
            text = (HERE / f).read_text()
            self.assertNotIn(chr(0x2014), text, f)
            self.assertNotIn(chr(0x2013), text, f)
            self.assertIsNone(re.search(r"(?i)claude|anthropic|T-70|vihan", text), f)

    def test_no_pycache(self):
        self.assertTrue(sys.dont_write_bytecode)


if __name__ == "__main__":
    unittest.main()
