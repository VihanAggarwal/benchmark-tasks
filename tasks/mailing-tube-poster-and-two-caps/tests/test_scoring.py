"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
IN_Z = L.TUBE_Z - (L.TUBE_R_IN - L.ROLL_R)  # the roll's axis when it lies on the bore's floor


def tube_state(dy=0.0, dx=0.0, dz=0.0, yaw=0.0):
    q = mk.quat_from_ypr(yaw, 0.0, 0.0)
    return mk.state_from_poses(T.TUBE_PART, (L.BOARD_X + dx, dy, L.TOP + L.TUBE_Z + dz, *q), source="test")


def roll_state(y=0.0, dx=0.0, z=IN_Z, yaw=0.0):
    q = mk.quat_from_ypr(yaw, 0.0, 0.0)
    return mk.state_from_poses(T.ROLL_PART, (L.BOARD_X + dx, y, L.TOP + z, *q), source="test")


def cap_state(side, gap=0.0, dx=0.0, dz=0.0, turn_deg=0.0, flipped=False, end=None, dy_tube=0.0):
    """A cap on the tube's end (+1: left, -1: right; by default its own side) ``gap`` outside it, ``turn_deg`` off
    the tube's axis about the vertical."""
    end = end if end is not None else (1 if side == "a" else -1)
    yaw = (0.0 if end > 0 else math.pi) + (math.pi if flipped else 0.0) + math.radians(turn_deg)
    q = mk.quat_from_ypr(yaw, 0.0, 0.0)
    spec = T.CAP_A_PART if side == "a" else T.CAP_B_PART
    return mk.state_from_poses(spec, (L.BOARD_X + dx, dy_tube + end * (L.TUBE_END + gap), L.TOP + L.TUBE_Z + dz, *q),
                               source="test")


def scene(roll_y=0.0, **kw):
    """The finished tube: the roll in, a cap on each end, unless overridden."""
    return {S.TUBE: kw.get("tube", tube_state()), S.ROLL: kw.get("roll", roll_state(roll_y)),
            S.CAP_A: kw.get("cap_a", cap_state("a")), S.CAP_B: kw.get("cap_b", cap_state("b"))}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.TUBE_L, 0.4572, places=4)  # 18 in
        self.assertAlmostEqual(L.TUBE_ID, 0.051, places=4)  # 2 in tube: about 51 mm bore
        self.assertAlmostEqual(L.TUBE_OD, 0.054, places=4)
        self.assertAlmostEqual(L.TUBE_Z, 0.032, places=4)  # 5 mm board + 27 mm
        self.assertAlmostEqual(L.CAP_DEPTH, 0.018, places=4)

    def test_roll_fits_and_caps_clear_it(self):
        self.assertLess(L.ROLL_R, L.TUBE_R_IN)  # 6.5 mm of play
        self.assertLess(L.CAP_PLUG_R, L.TUBE_R_IN)
        self.assertLess(L.CAP_NOSE_R, L.CAP_PLUG_R)
        # both plugs fit inside the tube beside the roll when it sits centred
        self.assertGreater(L.TUBE_END - L.ROLL_L / 2, L.CAP_DEPTH)
        # the roll's near end as found is outside the left mouth
        self.assertGreater(L.ROLL_LEAD_Y, L.TUBE_END)
        # the roll's far end is on the table
        self.assertLess(L.ROLL_Y0 + L.ROLL_L / 2, L.TABLE_Y)

    def test_flange_overhangs_tube_so_hand_is_clear(self):
        self.assertGreater(L.CAP_FLANGE_R - L.TUBE_R_OUT, 0.002)

    def test_tube_wall_closed(self):
        # adjacent staves overlap: width covers a side of the 16-gon
        side = 2 * L.TUBE_R_IN * math.tan(math.pi / L.TUBE_STAVES)
        self.assertGreater(L.TUBE_STAVE_W, side)
        self.assertEqual(len(T.tube_shapes()), 16)

    def test_rests_hold_caps_level(self):
        self.assertAlmostEqual(L.CAP_Z0 - L.CAP_PLUG_R - L.REST_PLAT_T, 0.0005, places=4)
        self.assertGreater(L.REST_RAIL_X - L.REST_RAIL_W / 2, L.CAP_FLANGE_R)

    def test_start_layout(self):
        res = run(start(), steps=5)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertFalse(res.details["roll_in"])
        self.assertEqual(res.details["ends_capped"], 0)

    def test_start_states_are_where_layout_says(self):
        r = S.read(start())
        for n, o in (("tube", L.tube_origin()), ("roll", L.roll_origin()), ("cap_a", L.cap_origin("a")),
                     ("cap_b", L.cap_origin("b"))):
            self.assertLess(float(np.linalg.norm(r.pos[n] - np.array(o))), 1e-6, n)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertTrue(res.details["roll_in"])
        self.assertEqual(res.details["ends_capped"], 2)

    def test_roll_anywhere_in_window_is_full(self):
        # both caps' plugs clear the roll: its ends within 15 mm of the mouths allows +-19.6 mm of centring (380 mm)
        lim = L.TUBE_END - L.ROLL_END_IN - L.ROLL_L / 2
        for y in (-lim + 0.0005, 0.0, lim - 0.0005):
            self.assertEqual(run(scene(y)).score, 1.0, y)

    def test_one_cap_each_side(self):
        for keep, other in (("cap_a", "cap_b"), ("cap_b", "cap_a")):
            far = cap_state("a" if other == "cap_a" else "b", gap=0.08)
            res = run(scene(**{other.lower(): far}))
            self.assertEqual(res.score, 0.5, other)
            self.assertFalse(res.success)

    def test_no_cap(self):
        res = run(scene(cap_a=cap_state("a", gap=0.1), cap_b=cap_state("b", gap=0.1)))
        self.assertEqual(res.score, 0.25)

    def test_nothing_done_is_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_success_needs_full(self):
        self.assertFalse(run(scene(cap_a=cap_state("a", gap=0.05))).success)


class RollIn(unittest.TestCase):
    def test_roll_end_too_near_the_mouth(self):
        # trailing end 10 mm inside the left mouth: under the 15 mm rule
        y = L.TUBE_END - 0.010 - L.ROLL_L / 2
        res = run(scene(y))
        self.assertFalse(res.details["roll_in"])
        self.assertEqual(res.score, 0.0)

    def test_roll_end_exactly_inside_limit(self):
        y = L.TUBE_END - L.ROLL_END_IN - L.ROLL_L / 2
        self.assertTrue(run(scene(y)).details["roll_in"])
        self.assertFalse(run(scene(y + 0.002)).details["roll_in"])

    def test_roll_on_the_far_side(self):
        y = -(L.TUBE_END - 0.010 - L.ROLL_L / 2)
        self.assertEqual(run(scene(y)).score, 0.0)

    def test_roll_beside_the_tube(self):
        res = run(scene(roll=roll_state(0.0, dx=0.06, z=L.ROLL_Z0)))
        self.assertEqual(res.score, 0.0)

    def test_roll_lying_on_top_of_the_tube(self):
        res = run(scene(roll=roll_state(0.0, z=L.TUBE_Z + L.TUBE_R_OUT + L.ROLL_R)))
        self.assertEqual(res.score, 0.0)

    def test_roll_radial_limit(self):
        # the roll lies 6.5 mm under the axis on the bore's floor; 8 mm is the limit (the 16-gon's corners allow 7.0)
        self.assertTrue(run(scene(roll=roll_state(0.0, dx=0.004))).details["roll_in"])
        self.assertFalse(run(scene(roll=roll_state(0.0, dx=0.006))).details["roll_in"])
        self.assertFalse(run(scene(roll=roll_state(0.0, z=L.TUBE_Z + 0.012))).details["roll_in"])

    def test_roll_turned_across_the_tube(self):
        res = run(scene(roll=roll_state(0.0, yaw=math.radians(90.0))))
        self.assertEqual(res.score, 0.0)

    def test_roll_tilted_so_one_end_is_outside(self):
        # centred but yawed 3 degrees: the ends are 10 mm off the axis
        res = run(scene(roll=roll_state(0.0, yaw=math.radians(3.0))))
        self.assertFalse(res.details["roll_in"])

    def test_roll_not_in_means_zero_whatever_the_caps(self):
        self.assertEqual(run(scene(roll=roll_state(0.4, z=L.ROLL_Z0))).score, 0.0)


class CapRules(unittest.TestCase):
    def test_gap_limits(self):
        ok = run(scene(cap_a=cap_state("a", gap=L.CAP_GAP - 0.0001))).score
        self.assertEqual(ok, 1.0)
        self.assertEqual(run(scene(cap_a=cap_state("a", gap=L.CAP_GAP + 0.0005))).score, 0.5)

    def test_cap_pushed_in_limit(self):
        self.assertEqual(run(scene(cap_a=cap_state("a", gap=-L.CAP_PEN + 0.0002))).score, 1.0)
        self.assertEqual(run(scene(cap_a=cap_state("a", gap=-L.CAP_PEN - 0.001))).score, 0.5)

    def test_radial_limit(self):
        self.assertEqual(run(scene(cap_a=cap_state("a", dz=0.002))).score, 1.0)
        self.assertEqual(run(scene(cap_a=cap_state("a", dz=0.004))).score, 0.5)
        self.assertEqual(run(scene(cap_b=cap_state("b", dx=0.004))).score, 0.5)

    def test_tilt_limit(self):
        self.assertEqual(run(scene(cap_a=cap_state("a", turn_deg=2.0))).score, 1.0)
        self.assertEqual(run(scene(cap_a=cap_state("a", turn_deg=7.0))).score, 0.5)

    def test_plug_must_be_in_the_bore(self):
        # a cap turned 4 degrees about its flange centre: the flange passes, the plug's tip is 1.3 mm off the axis
        self.assertTrue(S.cap_seated_at(S.read(scene(cap_a=cap_state("a", turn_deg=4.0))), S.CAP_A, 1))
        # a cap 3 mm up at the flange and level: the flange centre fails first
        self.assertFalse(S.cap_seated_at(S.read(scene(cap_a=cap_state("a", dz=0.003))), S.CAP_A, 1))

    def test_flipped_cap_does_not_count(self):
        self.assertEqual(run(scene(cap_a=cap_state("a", flipped=True))).score, 0.5)

    def test_cap_beside_the_tube_end(self):
        self.assertEqual(run(scene(cap_a=cap_state("a", dx=0.05))).score, 0.5)

    def test_each_end_counts_once(self):
        both_left = scene(cap_b=cap_state("b", end=1))  # both caps on the left end
        res = run(both_left)
        self.assertEqual(res.details["ends_capped"], 1)
        self.assertEqual(res.score, 0.5)

    def test_either_cap_on_either_end(self):
        res = run(scene(cap_a=cap_state("a", end=-1), cap_b=cap_state("b", end=1)))
        self.assertEqual(res.score, 1.0)

    def test_tube_may_be_anywhere_on_the_board_if_the_caps_follow(self):
        s = scene(tube=tube_state(dy=0.03), roll=roll_state(0.03), cap_a=cap_state("a", dy_tube=0.03),
                  cap_b=cap_state("b", dy_tube=0.03))
        self.assertEqual(run(s).score, 1.0)

    def test_tube_turned_with_everything_on_it(self):
        # the tube yawed 5 degrees with the roll and caps following it rigidly: still 1.0
        yaw = math.radians(5.0)
        c, s_ = math.cos(yaw), math.sin(yaw)
        o = np.array([L.BOARD_X, 0.0])

        def moved(state, spec, extra_yaw=0.0):
            p = np.asarray(state.origin, dtype=float)
            xy = o + np.array([[c, -s_], [s_, c]]) @ (p[:2] - o)
            q = mk.matrix_to_quat(mk.axis_angle_matrix((0, 0, 1), yaw) @ state.rotation())
            return mk.state_from_poses(spec, (xy[0], xy[1], p[2], *q), source="test")

        base = scene()
        s = {S.TUBE: moved(base[S.TUBE], T.TUBE_PART), S.ROLL: moved(base[S.ROLL], T.ROLL_PART),
             S.CAP_A: moved(base[S.CAP_A], T.CAP_A_PART), S.CAP_B: moved(base[S.CAP_B], T.CAP_B_PART)}
        self.assertEqual(run(s).score, 1.0)


class HandsOffAndStill(unittest.TestCase):
    def test_hand_on_the_tube_is_zero(self):
        a = {"left": arm((L.BOARD_X, 0.10, L.TOP + L.TUBE_Z), 0.4), "right": FAR["right"]}
        res = run(scene(), arms=a)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_hand_on_a_cap_is_zero(self):
        a = {"left": FAR["left"], "right": arm((L.BOARD_X, -(L.TUBE_END + 0.0025), L.TOP + L.TUBE_Z), 0.5)}
        self.assertEqual(run(scene(), arms=a).score, 0.0)

    def test_hand_just_clear_ok(self):
        # finger zone 30 mm or more away: fully open hand hovering high above
        a = {"left": arm((L.BOARD_X, 0.0, L.TOP + 0.20)), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=a).score, 1.0)

    def test_hand_near_for_one_step_in_the_window_zeroes_it(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        near = {"left": arm((L.BOARD_X, 0.10, L.TOP + L.TUBE_Z), 0.4), "right": FAR["right"]}
        tr.update(scene(), near)
        self.assertEqual(tr.last.score, 0.0)
        for _ in range(WIN - 1):
            tr.update(scene(), FAR)
            self.assertEqual(tr.last.score, 0.0)
        tr.update(scene(), FAR)
        self.assertEqual(tr.last.score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)
        bad = {"left": arm((np.nan, 0.45, L.TOP + 0.25)), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=bad).score, 0.0)
        self.assertEqual(run(scene(), arms={"left": FAR["left"]}).score, 0.0)

    def test_needs_a_full_second_of_clear_steps(self):
        tr = S.Tracker()
        for _ in range(WIN - 1):
            tr.update(scene(), FAR)
        self.assertEqual(tr.last.score, 0.0)
        tr.update(scene(), FAR)
        tr.update(scene(), FAR)
        self.assertEqual(tr.last.score, 1.0)

    def test_still_needs_one_second(self):
        tr = S.Tracker()
        for k in range(WIN + 5):
            res = tr.update(scene(tube=tube_state(dy=0.0005 * k)), FAR)  # drifting 10 mm/s
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_creep_under_1_mm_is_still(self):
        tr = S.Tracker()
        for k in range(WIN + 5):
            res = tr.update(scene(tube=tube_state(dy=0.00002 * k)), FAR)
        self.assertEqual(res.score, 1.0)

    def test_a_turning_part_is_not_still(self):
        tr = S.Tracker()
        for k in range(WIN + 5):
            res = tr.update(scene(roll=roll_state(0.0, yaw=math.radians(0.15 * k))), FAR)
        self.assertEqual(res.score, 0.0)  # 0.15 deg a step is 3 deg over the window (and the ends leave the axis)

    def test_a_rolling_cap_is_not_still(self):
        tr = S.Tracker()
        for k in range(WIN + 5):
            res = tr.update(scene(cap_a=cap_state("a", turn_deg=0.15 * k)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_motion_resets_the_window(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        self.assertEqual(tr.last.score, 1.0)
        tr.update(scene(tube=tube_state(dy=0.01)), FAR)
        self.assertEqual(tr.last.score, 0.0)


class Latch(unittest.TestCase):
    def test_fall_latches_even_if_it_comes_back(self):
        tr = S.Tracker()
        down = scene(cap_a=cap_state("a", dz=-0.12))
        tr.update(down, FAR)
        self.assertTrue(tr.latched)
        run(scene(), tracker=tr)
        self.assertEqual(tr.last.score, 0.0)
        self.assertIsNotNone(tr.last.details["latch"]["F1"])

    def test_off_the_table_latches(self):
        tr = S.Tracker()
        tr.update(scene(roll=roll_state(0.70)), FAR)
        self.assertTrue(tr.latched)

    def test_tube_on_the_floor(self):
        tr = S.Tracker()
        tr.update(scene(tube=tube_state(dz=-0.9)), FAR)
        self.assertTrue(tr.latched)

    def test_no_latch_on_the_board(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        self.assertFalse(tr.latched)


class BadStates(unittest.TestCase):
    def test_nan_state_is_zero_then_recovers(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = scene()
        o = np.array(bad[S.CAP_A].origin, dtype=float)
        o[1] = np.nan
        bad[S.CAP_A] = mk.AssemblyState(
            pos=bad[S.CAP_A].pos, quat_xyzw=bad[S.CAP_A].quat_xyzw, lin_vel=bad[S.CAP_A].lin_vel,
            ang_vel=bad[S.CAP_A].ang_vel, low=bad[S.CAP_A].low, high=bad[S.CAP_A].high, origin=o,
            joints={}, joint_vel={}, links={}, source="test", spec=T.CAP_A_PART)
        res = tr.update(bad, FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        for _ in range(WIN):
            self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # a second of valid readings is needed again
        self.assertEqual(tr.update(scene(), FAR).score, 1.0)

    def test_missing_part_is_zero(self):
        s = scene()
        s.pop(S.ROLL)
        self.assertEqual(run(s).score, 0.0)
        self.assertEqual(run({}).score, 0.0)
        self.assertEqual(run(None).score, 0.0)

    def test_empty_quaternion_is_zero(self):
        s = scene()
        q = np.array([np.nan, 0.0, 0.0, 1.0])
        s[S.TUBE] = mk.AssemblyState(
            pos=s[S.TUBE].pos, quat_xyzw=q, lin_vel=s[S.TUBE].lin_vel, ang_vel=s[S.TUBE].ang_vel, low=s[S.TUBE].low,
            high=s[S.TUBE].high, origin=s[S.TUBE].origin, joints={}, joint_vel={}, links={}, source="test",
            spec=T.TUBE_PART)
        self.assertEqual(run(s).score, 0.0)


class Exploits(unittest.TestCase):
    def test_caps_on_an_empty_tube(self):
        # the draft's exploit: both caps seated, the roll still on the table
        s = scene(roll=roll_state(0.435, z=L.ROLL_Z0))
        res = run(s)
        self.assertEqual(res.score, 0.0)

    def test_both_caps_on_the_same_end_cannot_score_full(self):
        self.assertEqual(run(scene(cap_b=cap_state("b", end=1))).score, 0.5)

    def test_holding_the_finished_tube_scores_nothing(self):
        a = {"left": arm((L.BOARD_X, 0.0, L.TOP + L.TUBE_Z), 0.45), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=a).score, 0.0)

    def test_caps_beside_the_tube_at_the_right_height(self):
        s = scene(cap_a=cap_state("a", dx=0.07), cap_b=cap_state("b", dx=-0.07))
        self.assertEqual(run(s).score, 0.25)

    def test_caps_lying_on_the_rest(self):
        s = scene(cap_a=T.CAP_A_PART.rest_state(), cap_b=T.CAP_B_PART.rest_state())
        self.assertEqual(run(s).score, 0.25)

    def test_roll_poked_through_with_a_cap_nowhere(self):
        # the roll pushed clean through to the far side, hanging out of the right end
        y = -(L.TUBE_END - 0.005 - L.ROLL_L / 2)
        self.assertEqual(run(scene(y, cap_b=cap_state("b", gap=0.3))).score, 0.0)

    def test_a_thrown_cap_seated_then_tube_lost(self):
        tr = S.Tracker()
        tr.update(scene(tube=tube_state(dz=-0.2)), FAR)
        run(scene(), tracker=tr)
        self.assertEqual(tr.last.score, 0.0)

    def test_reset_teleport_clears_the_latch(self):
        tr = S.Tracker()
        tr.update(scene(tube=tube_state(dz=-0.2)), FAR)
        self.assertTrue(tr.latched)
        res = tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(res.score, 0.0)

    def test_reset_after_success_scores_zero_again(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        self.assertEqual(tr.last.score, 1.0)
        tr.update(start(), FAR)
        self.assertEqual(tr.last.score, 0.0)
        self.assertEqual(tr.steps, 1)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        T._OFFLINE.reset()
        self.m = T.OfflineModel()

    def arms(self, left, right):
        return {"left": arm(*left), "right": arm(*right)}

    def test_roll_gripped_and_dropped_in_line(self):
        m = self.m
        far = (0.1, -0.45, L.TOP + 0.25)
        # closed on the roll at its far end
        m.step(self.arms(((0.30, 0.60, L.TOP + L.ROLL_Z0), 0.3), (far, 1.0)))
        self.assertIn("roll", m.grip)
        # carried in line to 0.247, then let go
        m.step(self.arms(((0.30, 0.247, L.TOP + L.TUBE_Z + 0.002), 0.3), (far, 1.0)))
        self.assertAlmostEqual(m.roll[1], 0.247 + 0.025 - L.ROLL_L / 2, places=3)
        m.step(self.arms(((0.30, 0.247, L.TOP + L.TUBE_Z + 0.002), 0.9), (far, 1.0)))
        self.assertTrue(m.on_tube["roll"])
        self.assertAlmostEqual(m.roll[2], IN_Z, places=4)

    def test_roll_not_in_line_falls_to_the_table(self):
        m = self.m
        far = (0.1, -0.45, L.TOP + 0.25)
        m.step(self.arms(((0.30, 0.60, L.TOP + L.ROLL_Z0), 0.3), (far, 1.0)))
        m.step(self.arms(((0.30, 0.50, L.TOP + 0.10), 0.3), (far, 1.0)))  # the roll lifted 100 mm
        m.step(self.arms(((0.30, 0.50, L.TOP + 0.10), 0.9), (far, 1.0)))
        self.assertFalse(m.on_tube["roll"])
        self.assertAlmostEqual(m.roll[2], L.ROLL_Z0, places=4)

    def test_cap_push_moves_the_roll(self):
        m = self.m
        m.roll = np.array([L.BOARD_X, 0.272 - L.ROLL_L / 2, IN_Z])
        m.on_tube["roll"] = True
        m.rel["roll"] = m.roll[1]
        far = (0.1, -0.45, L.TOP + 0.25)
        m.step(self.arms(((L.CAP_X0, 0.3475, L.TOP + L.CAP_Z0), 0.5), (far, 1.0)))
        gap = L.CAP_FLANGE_T / 2
        m.step(self.arms(((L.BOARD_X, 0.25 + gap, L.TOP + L.TUBE_Z + 0.001), 0.5), (far, 1.0)))
        # the nose tip at 0.25 - 0.018 = 0.232 pushes the roll's trailing end from 0.272 to 0.232
        self.assertAlmostEqual(m.roll[1] + L.ROLL_L / 2, 0.232, places=3)

    def test_cap_stops_on_the_tube_end(self):
        m = self.m
        far = (0.1, -0.45, L.TOP + 0.25)
        m.step(self.arms(((L.CAP_X0, 0.3475, L.TOP + L.CAP_Z0), 0.5), (far, 1.0)))
        self.assertIn("a", m.grip)
        m.step(self.arms(((L.BOARD_X, 0.20, L.TOP + L.TUBE_Z + 0.001), 0.5), (far, 1.0)))
        self.assertAlmostEqual(m.cap["a"][1], L.TUBE_END, places=4)

    def test_unreadable_arms_change_nothing(self):
        m = self.m
        before = m.roll.copy()
        m.step(None)
        self.assertTrue(np.allclose(before, m.roll))


class Demo(unittest.TestCase):
    def tearDown(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
            os.environ.pop(k, None)

    def test_demo_has_both_arms_within_the_run_limit(self):
        plan = T.demo()
        self.assertEqual(set(plan), {"left", "right"})
        total = max(T._duration(p) for p in plan.values())
        limit = int(re.match(r"(\d+)", json.loads((HERE / "task.json").read_text())["run_limit"]).group(1))
        self.assertLess(total * HZ, limit)
        self.assertLess(total, 55.0)
        self.assertAlmostEqual(T._duration(plan["left"]), T._duration(plan["right"]), places=6)

    def test_hold_only_and_idle(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        os.environ.pop("DEMO_HOLD_ONLY")
        os.environ["DEMO_IDLE_ARM"] = "left"
        self.assertEqual(set(T.demo()), {"right"})
        os.environ["DEMO_IDLE_ARM"] = "right"
        self.assertEqual(set(T.demo()), {"left"})

    def test_variants(self):
        os.environ["DEMO_VARIANT"] = "short"
        self.assertEqual(set(T.demo()), {"left", "right"})
        os.environ["DEMO_VARIANT"] = "nonsense"
        with self.assertRaises(ValueError):
            T.demo()

    def test_every_waypoint_is_sane(self):
        for arm_name, plan in T.plans().items():
            for w in plan:
                self.assertGreater(w.duration, 0.0)
                self.assertTrue(-0.0470 <= w.finger <= 0.0, w.label)
                self.assertGreaterEqual(w.pos[2] - L.TOP, 0.013, w.label)  # pad tips off the table
                self.assertTrue(all(math.isfinite(v) for v in w.pos))

    def test_gripper_moves_take_at_least_1_5_s(self):
        for plan in T.plans().values():
            prev = T.OPEN
            for w in plan:
                if abs(w.finger - prev) > 0.02:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
                prev = w.finger

    def test_pad_clears_the_tube_while_pressing(self):
        # the flange (60 mm) holds the pads 3 mm clear of the tube's 54 mm outside
        self.assertGreaterEqual(L.CAP_FLANGE_R - L.TUBE_R_OUT, 0.003 - 1e-9)

    def test_roll_slide_ends_pads_outside_the_mouth(self):
        end_pad_front = T.ROLL_END_Y - rigstate.FINGER_HALF_WIDTH
        self.assertGreater(end_pad_front, L.TUBE_END)
        # and the roll's trailing end is then within a cap's reach (the cap's nose pushes it the rest)
        trailing = T.ROLL_END_Y + 0.025
        self.assertGreater(trailing, L.TUBE_END - L.CAP_DEPTH)
        self.assertLess(trailing - (L.TUBE_END - L.CAP_DEPTH), 0.1)

    def test_roll_is_carried_by_its_middle_then_pushed_by_its_far_end(self):
        # a pinch holds no torque about its closing axis: the carry hold must be on the roll's middle (GPU run: held
        # 25 mm from its end, the near end hung on the table and jammed on the mouth's ring)
        plan = T.roll_moves()
        grips = [w for w in plan if w.label in ("grip the roll", "grip the far end")]
        self.assertEqual([w.label for w in grips], ["grip the roll", "grip the far end"])
        self.assertAlmostEqual(grips[0].pos[1], L.ROLL_Y0, places=6)
        # the carry puts the roll's axis within the bore's play of the tube's axis, the roll level
        carry = [w for w in plan if w.label == "slide the roll into the tube"][0]
        self.assertLess(abs((carry.pos[2] - L.TOP) - L.TUBE_Z), 0.003)
        # after the first slide the far end sticks out and the second hold is 25 mm in from it
        y_far = T.ROLL_END_Y + L.ROLL_L / 2
        self.assertAlmostEqual(grips[1].pos[1], y_far - T.ROLL_TAIL_IN, places=6)
        # the second hold ends with the far end 43 mm outside the mouth, as the cap push assumes
        self.assertAlmostEqual(T.ROLL_END_Y + T.ROLL_TAIL_IN, L.TUBE_END + 0.043, delta=0.002)
        # the hand comes down where the roll has rolled to, within the open pads' play
        self.assertLess(abs(T.ROLL_DRIFT_X - (L.BOARD_X - 0.017)), 0.005)

    def test_cap_press_travel(self):
        y_stage = T.cap_y("a", L.CAP_Y0["a"] - L.TUBE_END)
        self.assertAlmostEqual(y_stage, L.CAP_Y0["a"] + L.CAP_FLANGE_T / 2, places=4)
        self.assertAlmostEqual(T.cap_y("a", 0.0), L.TUBE_END + L.CAP_FLANGE_T / 2, places=4)
        self.assertAlmostEqual(T.cap_y("b", 0.0), -(L.TUBE_END + L.CAP_FLANGE_T / 2), places=4)

    def test_variant_gaps_make_a_wrong_press(self):
        for name, (ga, gb) in T.VARIANT_GAP.items():
            self.assertTrue(ga >= 0.0 and gb >= 0.0, name)
            self.assertTrue(ga > L.CAP_GAP or gb > L.CAP_GAP, name)


class Texts(unittest.TestCase):
    def setUp(self):
        self.tj = json.loads((HERE / "task.json").read_text())

    def test_fields_and_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.tj[k]), n, k)

    def test_no_em_dash_or_team(self):
        for p in (HERE / "task.json", HERE / "README.md", HERE / "task.py", HERE / "layout.py", HERE / "scoring.py"):
            if not p.exists():
                continue
            t = p.read_text()
            self.assertNotIn(chr(0x2014), t, p.name)
            self.assertNotRegex(t.lower(), r"t-70|vihan|aggarwal|claude|anthropic", p.name)

    def test_scoring_text_matches_layout(self):
        s = self.tj["scoring"]
        for needle in ("15 mm", "8 mm", "5 degrees", "1.5 mm", "2.5 mm", "3 mm", "30 mm", "1 mm", "2 degrees",
                       "0, 0.25, 0.5, 1.0"):
            self.assertIn(needle, s, needle)
        self.assertAlmostEqual(L.ROLL_END_IN, 0.015)
        self.assertAlmostEqual(L.ROLL_RADIAL, 0.008)
        self.assertAlmostEqual(L.CAP_TILT_DEG, 5.0)
        self.assertAlmostEqual(L.CAP_GAP, 0.0015)
        self.assertAlmostEqual(L.CAP_RADIAL, 0.0025)
        self.assertAlmostEqual(L.CAP_NOSE_RADIAL, 0.003)
        self.assertAlmostEqual(L.LET_GO, 0.030)

    def test_setup_text_matches_layout(self):
        s = self.tj["setup"]
        for needle in ("457 mm", "51 mm bore", "54 mm outside", "0.12 kg", "380 mm", "0.06 kg", "12 g", "x 170 mm",
                       "+345 mm", "-345 mm", "16.5 mm", "60 mm flange", "18 mm deep"):
            self.assertIn(needle, s, needle)
        self.assertAlmostEqual(L.ROLL_LEAD_Y - L.TUBE_END, 0.0165, places=3)


if __name__ == "__main__":
    unittest.main()
