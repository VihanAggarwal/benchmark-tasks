"""Offline tests for layout.py, scoring.py, pourplan.py and task.py: synthetic part states (mechkit AssemblyStates built
from the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import dataclasses
import json
import math
import os
import pathlib
import re
import sys
import types
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
    import pourplan as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
KIT_DIR = HERE.parents[1] / "tools" / "kit"
ML = 1e-6


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def bottle_start():
    return T.BOTTLE.rest_state()


def cup_start():
    return T.CUP.rest_state()


def bottle_tilted(theta, anchor):
    origin, r = P.bottle_pose(anchor, theta)
    return mk.state_from_poses(T.BOTTLE, (*origin, *P.matrix_quat(r)), source="test")


def cup_at(x=L.CUP_X, y=L.CUP_Y, lift=0.0, tilt_deg=0.0, vel=None):
    q = P.matrix_quat(P.rx(math.radians(tilt_deg)))
    return mk.state_from_poses(T.CUP, (x, y, L.CUP_Z + lift, *q), vel, source="test")


def ramp(a, b, n):
    return [a + (b - a) * P.smoothstep((i + 1) / n) for i in range(n)]


def pour(hold_s, cup_fn=None, tracker=None, theta=P.THETA_HOLD, anchor=None, settle=True, end_cup=None,
         end_bottle=None, end_hands=True):
    """The bottle brought upright over the cup, tilted to ``theta`` over 2.5 s, held ``hold_s``, tilted back over
    2.5 s, then (if ``settle``) put back on its mark with the cup set on the pan, still and hands clear for 2 s.
    ``cup_fn(step)`` gives the cup's state at each pouring step (default: held up level 45 mm over the pan)."""
    tr = tracker or S.Tracker()
    anchor = P.pour_anchor(P.THETA_HOLD) if anchor is None else anchor
    cup_fn = cup_fn or (lambda i: cup_at(lift=P.CUP_LIFT))
    thetas = ramp(0.0, theta, 50) + [theta] * int(round(hold_s * HZ)) + ramp(theta, 0.0, 50) + [0.0] * 20
    res = None
    for i, th in enumerate(thetas):
        res = tr.update(bottle_tilted(th, anchor), cup_fn(i), True)
    if settle:  # lower both back onto their marks over 2 s (a jump of 8 cm in a step would read as a reset)
        for k in range(1, 41):
            f = k / 40.0
            a = anchor + (P.grip_start() - anchor) * f
            res = tr.update(bottle_tilted(0.0, a), cup_at(lift=P.CUP_LIFT * (1.0 - f)), True)
        for _ in range(int(2 * HZ)):
            res = tr.update(end_bottle or bottle_start(), end_cup or cup_start(), end_hands)
    return tr, res


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(2 * L.BODY_R, 0.049)
        self.assertAlmostEqual(L.NECK_TOP, 0.105)
        self.assertAlmostEqual(2 * L.LIP_R, 0.022)
        self.assertAlmostEqual(L.CUP_OUTER, 0.036)
        self.assertAlmostEqual(L.CUP_H, 0.038)
        self.assertAlmostEqual(L.RHO, 1300.0)
        self.assertAlmostEqual(L.MU, 0.15)
        self.assertAlmostEqual(L.FILL, 118e-6)

    def test_bottle_holds_118_ml_and_the_bore_is_bigger(self):
        top = S.volume_below(L.NECK_TOP + 1.0, (0, 0, 0), (0, 0, 1.0))
        self.assertGreater(top, 118 * ML)
        self.assertLess(top, 140 * ML)
        zs = S.surface_height(L.FILL, (0, 0, 0), (0, 0, 1.0))
        self.assertLess(zs, L.NECK_TOP)  # 118 mL stands below the lip: nothing comes out of an upright bottle

    def test_total_mass_is_syrup_plus_bottle(self):
        self.assertAlmostEqual(L.FILL * L.RHO + 0.016, 0.1693, places=3)
        self.assertAlmostEqual(L.BOTTLE_MASS, 0.17)

    def test_cup_ten_ml_line(self):
        self.assertAlmostEqual(10 * ML / L.CUP_AREA * 1000, 9.77, places=2)
        self.assertGreater(L.CUP_INNER ** 2 * (L.CUP_H - L.CUP_FLOOR), 30 * ML)  # a 30 mL cup
        self.assertAlmostEqual(L.CUP_FLOOR + 10 * ML / L.CUP_AREA, 0.0118, places=4)

    def test_dose_windows_in_grams(self):
        self.assertAlmostEqual(9.0 * 1.3, 11.7)
        self.assertAlmostEqual(11.0 * 1.3, 14.3)
        self.assertAlmostEqual(8.0 * 1.3, 10.4)
        self.assertAlmostEqual(12.0 * 1.3, 15.6)
        self.assertAlmostEqual(4.0 * 1.3, 5.2)
        self.assertAlmostEqual(15.0 * 1.3, 19.5)
        self.assertAlmostEqual(0.5 * 1.3, 0.65)

    def test_parts_spawn_where_they_rest(self):
        b, c = bottle_start(), cup_start()
        self.assertAlmostEqual(float(b.low[2]), L.TOP, places=4)
        self.assertAlmostEqual(float(c.low[2]), L.PAN_TOP, places=4)
        self.assertTrue(S.Tracker.at_start(b, c))
        self.assertAlmostEqual(float(b.origin[0]), L.BOTTLE_X)
        self.assertAlmostEqual(float(b.origin[1]), L.BOTTLE_Y)

    def test_upright_bottle_pours_nothing_and_a_tilted_one_does(self):
        up = S.pour_rate(L.FILL, (0, 0, 0), (0, 0, 1.0))
        self.assertEqual(up[0], 0.0)
        for deg, flowing in ((20, False), (50, False), (76, True), (110, True)):
            th = math.radians(deg)
            origin, r = P.bottle_pose(np.zeros(3), th)
            q = S.pour_rate(L.FILL, origin, r[:, 2])[0]
            self.assertEqual(q > 0.0, flowing, deg)
        origin, r = P.bottle_pose(np.zeros(3), P.THETA_HOLD)
        self.assertAlmostEqual(S.pour_rate(L.FILL, origin, r[:, 2])[0] / ML, 3.0, places=3)  # capped by the neck

    def test_the_stream_window_is_a_third_of_a_second(self):
        # 3 mL/s: 2 mL (the 9 to 11 window) is 0.67 s, so a hold timed within +-0.33 s scores 1.0
        self.assertAlmostEqual((L.WINDOW[1] - L.WINDOW[0]) * ML / L.Q_CAP, 0.667, places=2)

    def test_pour_point_is_over_the_cup(self):
        a = P.pour_anchor(P.THETA_HOLD)
        origin, r = P.bottle_pose(a, P.THETA_HOLD)
        low = S.lip_low_point(origin, r[:, 2])
        self.assertAlmostEqual(float(low[0]), L.CUP_X, places=6)
        self.assertAlmostEqual(float(low[1]), L.CUP_Y, places=6)
        self.assertGreater(float(low[2]), L.CUP_Z + P.CUP_LIFT + L.CUP_H + 0.02)  # the stream falls at least 2 cm


class Tiers(unittest.TestCase):
    def test_full(self):
        tr, res = pour(T.POUR_HOLD)
        self.assertEqual(res.score, 1.0, res.details)
        self.assertTrue(res.success)
        self.assertGreater(res.details["cup_ml"], 9.3)
        self.assertLess(res.details["cup_ml"], 10.7)
        self.assertEqual(res.details["spilled_ml"], 0.0)
        self.assertEqual(res.details["unread_ml"], 0.0)

    def test_hold_time_tiers(self):
        want = {0.0: 0.0, 0.5: 0.25, 1.5: 0.5, 2.2: 1.0, 3.0: 0.25, 3.5: 0.25, 6.0: 0.0}
        for hold, score in want.items():
            _, res = pour(hold)
            self.assertEqual(res.score, score, (hold, res.details["cup_ml"]))

    def test_no_pour_scores_zero(self):
        tr = S.Tracker()
        res = None
        for _ in range(60):
            res = tr.update(bottle_start(), cup_start(), True)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_start_scores_zero_and_does_not_succeed(self):
        res = S.Tracker().update(bottle_start(), cup_start(), True)
        self.assertEqual((res.score, res.success), (0.0, False))

    def test_window_edges(self):
        tr = S.Tracker()
        want = [(3.99, 0.0), (4.0, 0.25), (7.99, 0.25), (8.0, 0.5), (8.99, 0.5), (9.0, 1.0), (10.0, 1.0), (11.0, 1.0),
                (11.01, 0.5), (12.0, 0.5), (12.01, 0.25), (14.9, 0.25)]
        for dose, score in want:
            tr.v_cup = tr.v_peak = dose * ML + (-1e-12 if dose in (11.0, 12.0) else 1e-12)
            self.assertEqual(tr.tier(True), score, dose)

    def test_overdose_latch(self):
        tr = S.Tracker()
        tr.v_cup = 10 * ML
        tr.v_peak = 15.5 * ML
        self.assertEqual(tr.tier(True), 0.0)
        tr.v_peak = 14.9 * ML
        self.assertEqual(tr.tier(True), 1.0)

    def test_not_placed_is_three_quarters(self):
        up = bottle_tilted(0.0, P.grip_start() + np.array([0.0, 0.0, 0.02]))  # upright, lifted 20 mm
        tilted = bottle_tilted(math.radians(30), P.grip_start())
        cases = {
            "hands still near": dict(end_hands=False),
            "cup left up in the air": dict(end_cup=cup_at(lift=0.03)),
            "cup off the pan's middle": dict(end_cup=cup_at(x=L.CUP_X + 0.03)),
            "bottle lifted": dict(end_bottle=up),
            "bottle left tilted": dict(end_bottle=tilted),
        }
        for name, kw in cases.items():
            _, res = pour(T.POUR_HOLD, **kw)
            self.assertEqual(res.score, 0.75, (name, res.details))
            self.assertFalse(res.success, name)
        _, res = pour(T.POUR_HOLD)
        self.assertEqual(res.score, 1.0)

    def test_poured_with_the_cup_on_the_pan_is_half(self):
        _, res = pour(T.POUR_HOLD, cup_fn=lambda i: cup_start())
        self.assertEqual(res.score, 0.5)
        self.assertGreater(res.details["unread_s"], 0.15)
        self.assertGreater(res.details["unread_ml"], 5.0)

    def test_poured_with_the_cup_tilted_is_half(self):
        _, res = pour(T.POUR_HOLD, cup_fn=lambda i: cup_at(lift=P.CUP_LIFT, tilt_deg=8.0))
        self.assertEqual(res.score, 0.5)

    def test_missed_stream_over_three_ml_is_half(self):
        # the cup is 6 cm to the side until the stream has run 1.5 s, then comes under the lip: 3.8 mL on the table
        _, res = pour(3.2, cup_fn=lambda i: cup_at(y=L.CUP_Y + (0.06 if i < 65 else 0.0), lift=P.CUP_LIFT))
        self.assertGreater(res.details["spilled_ml"], 3.0)
        self.assertLessEqual(res.score, 0.5)

    def test_small_spill_is_three_quarters(self):
        tr = S.Tracker()
        tr.v_cup = tr.v_peak = 10 * ML
        tr.spilled = 1.0 * ML
        self.assertEqual(tr.tier(True), 0.75)
        tr.spilled = 0.5 * ML
        self.assertEqual(tr.tier(True), 1.0)
        tr.spilled = 3.0 * ML
        self.assertEqual(tr.tier(True), 0.75)
        tr.spilled = 3.1 * ML
        self.assertEqual(tr.tier(True), 0.5)
        tr.spilled = 10.1 * ML
        self.assertEqual(tr.tier(True), 0.0)

    def test_unread_threshold(self):
        tr = S.Tracker()
        tr.v_cup = tr.v_peak = 10 * ML
        tr.unread_s = 0.15
        self.assertEqual(tr.tier(True), 1.0)
        tr.unread_s = 0.2
        self.assertEqual(tr.tier(True), 0.5)

    def test_unread_is_counted_in_seconds_of_stream(self):
        tr = S.Tracker()
        tr.v_cup = tr.v_peak = 10 * ML
        for steps, score in ((3, 1.0), (4, 0.5)):
            tr.unread_s = 0.0
            for _ in range(steps):
                tr.unread_s += tr.dt
            self.assertEqual(tr.tier(True), score, steps)

    def test_success_needs_stillness(self):
        tr, _ = pour(T.POUR_HOLD)
        moving = mk.state_from_poses(T.BOTTLE, (L.BOTTLE_X, L.BOTTLE_Y, L.TOP, 0, 0, 0, 1), [0.05, 0, 0, 0, 0, 0],
                                     source="test")
        res = tr.update(moving, cup_start(), True)
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.success)
        spinning = mk.state_from_poses(T.CUP, (L.CUP_X, L.CUP_Y, L.CUP_Z, 0, 0, 0, 1), [0, 0, 0, 0, 0, 1.0], source="test")
        self.assertFalse(tr.update(bottle_start(), spinning, True).success)
        self.assertTrue(tr.update(bottle_start(), cup_start(), True).success)


class Latches(unittest.TestCase):
    def test_overpour_cannot_be_undone(self):
        tr, res = pour(6.0, cup_fn=lambda i: cup_at(lift=P.CUP_LIFT))
        self.assertEqual(res.score, 0.0)
        self.assertGreater(res.details["peak_ml"], 15.0)

    def test_tipping_the_cup_out_is_zero_for_good(self):
        tr, _ = pour(T.POUR_HOLD, settle=False)
        res = tr.update(bottle_start(), cup_at(lift=0.05, tilt_deg=70.0), True)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(res.details["dumped"])
        for _ in range(40):
            res = tr.update(bottle_start(), cup_start(), True)
        self.assertEqual(res.score, 0.0)  # an empty cup back on the scale does not forgive it

    def test_emptying_the_bottle_on_the_table_is_zero(self):
        _, res = pour(8.0, cup_fn=lambda i: cup_at(y=0.2))
        self.assertGreater(res.details["spilled_ml"], 10.0)
        self.assertEqual(res.score, 0.0)

    def test_cup_tilted_over_thirty_degrees_catches_nothing(self):
        _, res = pour(T.POUR_HOLD, cup_fn=lambda i: cup_at(lift=P.CUP_LIFT, tilt_deg=35.0))
        self.assertEqual(res.details["cup_ml"], 0.0)
        self.assertGreater(res.details["spilled_ml"], 5.0)

    def test_ledger_is_conserved(self):
        tr, res = pour(T.POUR_HOLD, cup_fn=lambda i: cup_at(y=L.CUP_Y + (0.06 if i < 20 else 0.0), lift=P.CUP_LIFT))
        total = tr.v_cup + tr.spilled + tr.v_bottle
        self.assertAlmostEqual(total / ML, 118.0, places=6)

    def test_ledger_is_not_reset_by_a_jump_away_from_the_start(self):
        tr, _ = pour(T.POUR_HOLD, settle=False)
        before = tr.v_cup
        tr.update(bottle_start(), cup_at(x=0.2, y=0.2), True)  # a big jump, but not onto the start marks
        self.assertEqual(tr.resets, 0)
        self.assertAlmostEqual(tr.v_cup, before)


class ReleaseAndHands(unittest.TestCase):
    def _hands(self, arms):
        return T._hands_clear(arms, bottle_start(), cup_start())

    def test_far_hands_are_clear(self):
        self.assertTrue(self._hands(FAR))

    def test_hand_on_the_cup(self):
        arms = {"left": arm((L.CUP_X, L.CUP_Y, L.TOP + 0.03), 0.4), "right": FAR["right"]}
        self.assertFalse(self._hands(arms))

    def test_hand_on_the_bottle(self):
        arms = {"left": FAR["left"], "right": arm((L.BOTTLE_X, L.BOTTLE_Y, L.TOP + 0.03), 0.4)}
        self.assertFalse(self._hands(arms))

    def test_unreadable_arms_are_never_clear(self):
        self.assertFalse(self._hands(None))
        bad = {"left": arm((float("nan"), 0.1, 0.8)), "right": FAR["right"]}
        self.assertFalse(self._hands(bad))
        self.assertFalse(self._hands({"left": FAR["left"]}))


class Robustness(unittest.TestCase):
    def test_nan_state(self):
        bad = mk.state_from_poses(T.BOTTLE, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        tr = S.Tracker()
        res = tr.update(bad, cup_start(), True)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertEqual(tr.v_bottle, L.FILL)

    def test_nan_quaternion_and_inf(self):
        good = cup_start()
        bad_quat = types.SimpleNamespace(**{k: getattr(good, k) for k in
                                            ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")})
        bad_quat.quat_xyzw = np.array([0.0, 0.0, 0.0, float("nan")])
        zero_quat = types.SimpleNamespace(**vars(bad_quat))
        zero_quat.quat_xyzw = np.zeros(4)
        inf_pos = mk.state_from_poses(T.CUP, (float("inf"), 0, 0.75, 0, 0, 0, 1), source="test")
        for bad in (bad_quat, zero_quat, inf_pos):
            res = S.Tracker().update(bottle_start(), bad, True)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.details["valid_state"])

    def test_missing_or_garbage_states(self):
        for garbage in (object(), None, {}):
            res = S.Tracker().update(garbage, cup_start(), True)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)

    def test_nan_mid_pour_does_not_move_the_ledger(self):
        tr, _ = pour(1.0, settle=False)
        v = tr.v_cup
        bad = mk.state_from_poses(T.BOTTLE, (float("nan"),) * 3 + (0, 0, 0, 1), source="test")
        tr.update(bad, cup_start(), True)
        self.assertEqual(tr.v_cup, v)

    def test_teleport_back_to_the_start_resets(self):
        tr, _ = pour(T.POUR_HOLD, settle=False)
        self.assertGreater(tr.v_cup, 0.0)
        tr.update(bottle_start(), cup_start(), True)  # the simulator put both back on their marks in one step
        self.assertEqual(tr.resets, 1)
        self.assertEqual(tr.v_cup, 0.0)
        self.assertEqual(tr.v_bottle, L.FILL)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            th = rng.uniform(-3.2, 3.2)
            anchor = P.grip_start() + rng.uniform(-0.2, 0.2, 3)
            b = bottle_tilted(th, anchor)
            c = cup_at(x=L.CUP_X + rng.uniform(-0.1, 0.1), y=L.CUP_Y + rng.uniform(-0.1, 0.1),
                       lift=rng.uniform(0, 0.1), tilt_deg=rng.uniform(0, 180))
            r = tr.update(b, c, bool(rng.integers(0, 2)))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertGreaterEqual(tr.v_bottle, 0.0)


class Exploits(unittest.TestCase):
    """Things a policy might try instead of pouring a dose."""

    def test_pour_it_on_the_pan_without_lifting_the_cup(self):
        _, res = pour(T.POUR_HOLD, cup_fn=lambda i: cup_start())
        self.assertLessEqual(res.score, 0.5)

    def test_overfill_then_tip_out_to_get_back_to_ten(self):
        tr, _ = pour(4.5, settle=False)
        tr.update(bottle_start(), cup_at(lift=0.06, tilt_deg=80.0), True)
        for _ in range(60):
            res = tr.update(bottle_start(), cup_start(), True)
        self.assertEqual(res.score, 0.0)

    def test_fill_a_little_then_top_up_is_just_the_total(self):
        tr, _ = pour(0.5, settle=False)
        tr, res = pour(0.4, tracker=tr)
        self.assertEqual(res.score, 1.0, res.details)

    def test_hold_the_cup_forever(self):
        tr, _ = pour(T.POUR_HOLD, settle=False)
        res = None
        for _ in range(60):
            res = tr.update(bottle_start(), cup_at(lift=0.04), True)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.success)

    def test_bottle_laid_down_empties_without_a_dose(self):
        tr = S.Tracker()
        res = None
        for _ in range(200):
            res = tr.update(bottle_tilted(math.radians(100), P.grip_start()), cup_start(), True)
        self.assertEqual(res.score, 0.0)

    def test_tilting_below_the_lip_pours_nothing(self):
        tr = S.Tracker()
        for _ in range(200):
            res = tr.update(bottle_tilted(math.radians(30), P.pour_anchor()), cup_at(lift=P.CUP_LIFT), True)
        self.assertEqual(tr.poured, 0.0)
        self.assertEqual(res.score, 0.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["scale_pan", T.CUP_NAME, T.BOTTLE_NAME])
        self.assertIsNone(T.SCALE.mass)  # fixed
        self.assertEqual(T.SCENE, "station")
        self.assertEqual(T.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(T.SCORE_OVER, "end")

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        self.assertIn("10 mL", T.INSTRUCTION)
        self.assertNotIn("tilt", T.INSTRUCTION.lower())
        self.assertNotIn("second", T.INSTRUCTION.lower())

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        plan = {"left": T.left_plan(), "right": T.right_plan()}
        for a, wps in plan.items():
            for w in wps:
                self.assertTrue(all(math.isfinite(c) for c in w.pos))
                self.assertGreater(w.duration, 0.0)
                self.assertLessEqual(w.finger, 0.0)
                self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (a, w.label))
            for x, y in zip(wps, wps[1:]):
                if abs(x.finger - y.finger) > 0.005:
                    self.assertGreaterEqual(y.duration, 1.5 - 1e-9, y.label)  # gripper moves take 1.5 s or more
        self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ + 80, run_limit)

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)
        self.assertEqual(set(T.demo()), {"left", "right"})

    def test_pour_hold_env(self):
        os.environ["DEMO_POUR_HOLD"] = "3.0"
        try:
            self.assertEqual(T.pour_hold(), 3.0)
            os.environ["DEMO_POUR_HOLD"] = "junk"
            self.assertEqual(T.pour_hold(), T.POUR_HOLD)
        finally:
            os.environ.pop("DEMO_POUR_HOLD")

    def test_demo_timing(self):
        """The left hand holds the cup up the whole time the bottle is tilted, and sets it down after the bottle is
        upright again."""
        left, right = T.left_plan(), T.right_plan()

        def span(wps, label):
            t = 0.0
            for w in wps:
                if w.label == label:
                    return t, t + w.duration
                t += w.duration
            raise KeyError(label)

        tilt0 = span(right, "tilt to pour")[0]
        upright = span(right, "tilt back upright")[1]
        lifted = span(left, "lift it 45 mm")[1]
        down0 = span(left, "set it on the scale")[0]
        self.assertLessEqual(lifted, tilt0)
        self.assertGreaterEqual(down0, upright - 1e-9)
        self.assertAlmostEqual(T.POUR_HOLD, 2.2)

    def test_demo_pour_dose_is_in_the_window(self):
        """The demo's tilt in, hold and tilt out, through the scorer: the dose it makes for each hold time."""
        for hold, lo, hi in ((2.2, 9.0, 11.0), (1.9, 8.0, 9.5), (2.5, 10.5, 12.0)):
            _, res = pour(hold)
            self.assertTrue(lo <= res.details["cup_ml"] <= hi, (hold, res.details["cup_ml"]))

    def test_grip_geometry(self):
        self.assertAlmostEqual(-2 * T.BOTTLE_GRIP, 2 * L.BODY_R - 0.008)
        self.assertAlmostEqual(-2 * T.CUP_GRIP, L.CUP_OUTER - 0.008)
        self.assertGreater(-T.OPEN * 2, 2 * L.BODY_R + 0.02)  # open gripper clears the bottle by 1 cm each side

    def test_hands_apart_at_the_pour(self):
        a = P.pour_anchor(P.THETA_HOLD)
        right = P.right_tcp(a, P.THETA_HOLD)
        left = P.left_tcp(np.array([L.CUP_X, L.CUP_Y, L.CUP_Z + P.CUP_LIFT]))
        self.assertGreater(float(np.linalg.norm(right - left)), 0.09)


class SyrupDisplay(unittest.TestCase):
    """The declared display rule: an amber plate in the cup rides at the ledger's level; the 10 mL line is drawn."""

    def test_target_follows_the_ledger(self):
        self.assertEqual(T.syrup_target(0.0), 0.0)
        self.assertAlmostEqual(T.syrup_target(10 * ML), 10 * ML / L.CUP_AREA - L.SYRUP_T, places=9)
        self.assertAlmostEqual(T.syrup_target(10 * ML) + L.SYRUP_T + L.CUP_FLOOR, L.LINE_Z, places=9)  # at the line
        self.assertLess(T.syrup_target(5 * ML), T.syrup_target(10 * ML))

    def test_target_is_bounded_and_safe(self):
        self.assertEqual(T.syrup_target(1.0), L.SYRUP_RANGE)  # a cubic metre stops at the brim
        for bad in (-1e-6, float("nan"), float("inf"), None, "x", [1]):
            self.assertEqual(T.syrup_target(bad), 0.0)
        self.assertEqual(T.syrup_target(1e-12), 0.0)  # under the plate's own thickness

    def test_joint_and_line(self):
        j = T.SYRUP_PLATE
        self.assertEqual(j.kind, "prismatic")
        self.assertGreater(j.spring, 0.0)  # a target does nothing without a spring
        self.assertAlmostEqual(L.LINE_Z, 0.0118, places=4)
        self.assertEqual(len(T.LINE_STRIPS), 2)
        ys = sorted(round(s.center[1], 6) for s in T.LINE_STRIPS)
        self.assertEqual(ys, [round(-(L.CUP_OUTER / 2 + L.LINE_T / 2), 6), round(L.CUP_OUTER / 2 + L.LINE_T / 2, 6)])
        for s in T.LINE_STRIPS:  # not on the faces the left hand grips (fingers close along x)
            self.assertLess(abs(s.center[0]), 1e-9)

    def test_plate_stays_inside_the_cup_and_the_cup_still_rests_on_the_pan(self):
        lo = T.CUP.rest_state()
        self.assertAlmostEqual(float(lo.low[2]), L.PAN_TOP, places=4)
        self.assertLess(L.CUP_INNER - 0.0004, L.CUP_INNER)
        self.assertGreater(L.SYRUP_RANGE, 15 * ML / L.CUP_AREA)  # travel covers the 15 mL overdose level


class OfflineModel(unittest.TestCase):
    def test_alias(self):
        self.assertIs(T.OfflineModel, T.OfflineBottle)

    def test_without_arms_the_state_is_unchanged(self):
        m = T.OfflineModel()
        b = bottle_start()
        self.assertIs(m.state(b, None), b)

    def test_held_bottle_follows_the_hand_tilt(self):
        """Bring the right hand through the demo's tilt: the model's bottle ends up at the demo's tilt."""
        m = T.OfflineModel()
        a = P.pour_anchor(P.THETA_HOLD)
        centre0 = np.array([a[0], a[1], a[2] + 0.04])
        res = None
        for th in ramp(0.0, P.THETA_HOLD, 20):
            tcp = P.right_tcp(a, th)
            tips = tcp + 0.013 * P.right_approach(th)
            origin, r = P.bottle_pose(a, th)
            centre = origin + r @ np.array([0.0, 0.0, L.NECK_TOP / 2])
            carried = mk.state_from_poses(T.BOTTLE, (*origin, *P.matrix_quat(r)), source="offline")
            carried = dataclasses.replace(carried, pos=np.asarray(centre, dtype=float))
            arms = {"left": FAR["left"], "right": rigstate.Arm(tcp, tips, 0.2)}
            res = m.state(carried, arms)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(S.tilt_deg(res.quat_xyzw), math.degrees(P.THETA_HOLD), delta=3.0)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("4 fl oz (118 mL)", "x 300 mm, y -180 mm", "49 mm across", "82 mm tall", "28 mm across",
                      "105 mm", "22 mm opening", "2 mm walls", "118 mL of syrup", "170 g", "1.30 g/mL", "150 mPa s",
                      "30 mL (1 fl oz)", "36 mm across and 38 mm tall", "8 g", "10 mL", "100 x 100 mm", "14 mm high",
                      "x 300 mm, y 0", "11.8 mm", "3 mL/s", "76 degrees", "32 mm", "amber plate", "dark 2 mm strip"):
            self.assertIn(token, setup)
        for token in ("15 mL", "45 degrees", "10 mL", "9.0 to 11.0 mL", "0.5 mL", "8.0 to 12.0 mL",
                      "at least 4 mL", "0.15 s", "4 such frames", "20 mm", "5 degrees", "15 mm", "3 mm", "30 mm",
                      "1 cm/s", "0.5 rad/s",
                      "Possible scores: 0, 0.25, 0.5, 0.75, 1.0", "11.7 to 14.3 g", "10.4 to 15.6 g", "0.65 g",
                      "0.1 g", "3 mL", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        self.assertAlmostEqual(L.WINDOW[0], 9.0)
        self.assertAlmostEqual(L.WINDOW[1], 11.0)
        self.assertAlmostEqual(L.WIDE[0], 8.0)
        self.assertAlmostEqual(L.WIDE[1], 12.0)
        self.assertAlmostEqual(L.OVERDOSE_ML, 15.0)
        self.assertAlmostEqual(L.PARTIAL_ML, 4.0)
        self.assertAlmostEqual(L.CLEAN_SPILL_ML, 0.5)
        self.assertAlmostEqual(L.UNREAD_S, 0.15)
        self.assertAlmostEqual(L.SPILL_CAP_ML, 3.0)
        self.assertAlmostEqual(L.SPILL_ZERO_ML, 10.0)
        self.assertAlmostEqual(L.HELD_CLEAR, 0.020)
        self.assertAlmostEqual(L.HELD_TILT_DEG, 5.0)
        self.assertAlmostEqual(L.DUMP_TILT_DEG, 45.0)
        self.assertAlmostEqual(L.CUP_SET_XY, 0.015)
        self.assertAlmostEqual(L.CUP_SET_Z, 0.003)
        self.assertAlmostEqual(L.CUP_SET_TILT_DEG, 3.0)
        self.assertAlmostEqual(L.BOTTLE_SET_XY, 0.030)
        self.assertAlmostEqual(L.BOTTLE_SET_TILT_DEG, 5.0)
        self.assertAlmostEqual(L.LET_GO, 0.030)
        scores = d.get("scores", "")  # before the GPU run: TBD; after it: the real Dream-1 run and demo 1.0
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_ledger_and_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("volume ledger", "3.0 mL/s", "est", "square", "No drip tail", "150 mPa s", "1.30 g/mL",
                      "Declared rule", "amber plate", "display only"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("parent", "dosing cup", "3 mL/s", "cannot be poured back"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "pourplan.py", "tiltgrip.py",
                  "rigstate.py", "mechkit.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("\u2014|\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "pourplan.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            else:
                self.assertNotIn("\u2014", text)
                self.assertNotIn("\u2013", text)
            for i, line in enumerate(text.splitlines() if f.endswith(".py") else [], 1):
                self.assertLessEqual(len(line), 125, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
