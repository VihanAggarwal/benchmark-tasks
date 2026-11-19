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
        self.assertAlmostEqual(L.CUP_OUTER, 0.040)
        self.assertAlmostEqual(L.CUP_INNER, 0.036)
        self.assertAlmostEqual(L.CUP_H, 0.038)
        self.assertAlmostEqual(L.RHO, 1310.0)  # Simple Syrup NF, 1.313 g/mL
        self.assertAlmostEqual(L.MU, 0.15)
        self.assertAlmostEqual(L.FILL, 118e-6)

    def test_bottle_holds_118_ml_and_the_bore_is_bigger(self):
        top = S.volume_below(L.NECK_TOP + 1.0, (0, 0, 0), (0, 0, 1.0))
        self.assertGreater(top, 118 * ML)
        self.assertLess(top, 140 * ML)
        zs = S.surface_height(L.FILL, (0, 0, 0), (0, 0, 1.0))
        self.assertLess(zs, L.NECK_TOP)  # 118 mL stands below the lip: nothing comes out of an upright bottle

    def test_total_mass_is_syrup_plus_bottle(self):
        self.assertAlmostEqual(L.FILL * L.RHO + 0.016, 0.1706, places=3)  # 171 g, 170 g in the simulation
        self.assertAlmostEqual(L.BOTTLE_MASS, 0.17)

    def test_cup_ten_ml_line(self):
        self.assertAlmostEqual(10 * ML / L.CUP_AREA * 1000, 9.77, places=2)
        self.assertGreater(L.CUP_AREA * (L.CUP_H - L.CUP_FLOOR), 30 * ML)  # a 30 mL cup
        self.assertAlmostEqual(L.CUP_FLOOR + 10 * ML / L.CUP_AREA, 0.0118, places=4)

    def test_dose_windows_in_grams(self):
        g = L.RHO / 1000.0  # g/mL: the gram figures in task.json and the README
        for mL, grams in ((9.0, 11.8), (11.0, 14.4), (8.0, 10.5), (12.0, 15.7), (4.0, 5.2), (15.0, 19.7), (3.0, 3.9),
                          (10.0, 13.1)):
            self.assertAlmostEqual(round(mL * g, 1), grams, places=9, msg=mL)
        self.assertAlmostEqual(round(0.5 * g, 2), 0.66, places=9)
        text = (HERE / "task.json").read_text() + (HERE / "README.md").read_text()
        for figure in ("11.8 to 14.4 g", "10.5 to 15.7", "5.2 g", "19.7 g", "0.66 g", "3.9 g", "13.1 g", "1.31 g/mL"):
            self.assertIn(figure, text)
        for old in ("1.30 g/mL", "11.7 to 14.3", "19.5 g", "0.65 g"):
            self.assertNotIn(old, text)

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

    def test_the_stream_window_is_two_thirds_of_a_second(self):
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

    def test_stated_tilt_is_the_plan_tilt(self):
        deg = round(math.degrees(P.THETA_HOLD))
        self.assertEqual(deg, 76)
        self.assertIn(f"tilt it to {deg} degrees", T.right_plan.__doc__)
        for f in ("task.py", "README.md", "task.json"):
            for m in re.finditer(r"tilt(?:s|ed)? (?:it )?(?:to|of) (\d+) degrees", (HERE / f).read_text()):
                self.assertEqual(int(m.group(1)), deg, (f, m.group(0)))

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
        lifted = span(left, "lift it 32 mm")[1]
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

    def test_demo_holds_the_cup_off_centre_and_low_but_up(self):
        """The left hand holds the cup 32 mm up (12 mm over the 20 mm 'held up' rule) and 6 mm toward +y, away from the
        bottle, so the top camera sees the syrup clear of the neck: the lip's lowest point then falls 6 mm from the
        cup's axis, well inside the 14 mm catch circle, and the dose and score are those of a centred cup."""
        self.assertAlmostEqual(P.CUP_LIFT, 0.032)
        self.assertAlmostEqual(P.CUP_HOLD_DY, 0.006)
        self.assertGreaterEqual(P.CUP_LIFT - L.HELD_CLEAR, 0.010)
        origin, rot = P.bottle_pose(P.pour_anchor(P.THETA_HOLD), P.THETA_HOLD)
        low = S.lip_low_point(origin, rot[:, 2])
        off = float(np.hypot(low[0] - L.CUP_X, low[1] - (L.CUP_Y + P.CUP_HOLD_DY)))
        self.assertLess(off, L.CUP_INNER_HALF - L.CATCH_MARGIN - 0.007)
        self.assertGreater(float(low[2]) - (L.CUP_Z + P.CUP_LIFT + L.CUP_H), 0.04)  # the lip 4 cm or more over the rim
        hold = T.left_plan()[4]
        self.assertEqual(hold.label, "hold it level while the syrup goes in")
        np.testing.assert_allclose(hold.pos, P.left_tcp(np.array([L.CUP_X, L.CUP_Y + P.CUP_HOLD_DY,
                                                                  L.CUP_Z + P.CUP_LIFT])), atol=1e-12)
        _, centred = pour(T.POUR_HOLD)
        _, shifted = pour(T.POUR_HOLD, cup_fn=lambda i: cup_at(y=L.CUP_Y + P.CUP_HOLD_DY, lift=P.CUP_LIFT))
        self.assertEqual(shifted.score, 1.0)
        self.assertAlmostEqual(shifted.details["cup_ml"], centred.details["cup_ml"], places=6)
        self.assertEqual(shifted.details["spilled_ml"], 0.0)

    def test_hands_apart_at_the_pour(self):
        a = P.pour_anchor(P.THETA_HOLD)
        right = P.right_tcp(a, P.THETA_HOLD)
        left = P.left_tcp(np.array([L.CUP_X, L.CUP_Y, L.CUP_Z + P.CUP_LIFT]))
        self.assertGreater(float(np.linalg.norm(right - left)), 0.09)


def _height(shape):
    return shape.height if shape.kind == "cylinder" else shape.size[2]


def _yaw(shape):
    """A shape's turn about z (rad); every shape of the cup is turned about z only."""
    r = shape.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _rect(shape):
    """(centre xy, unit axes (2x2, columns), half extents xy, z0, z1) of a box turned about z."""
    c = np.asarray(shape.center, dtype=float)
    yaw = _yaw(shape)
    axes = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    h = np.asarray(shape.size, dtype=float) / 2
    return c[:2], axes, h[:2], c[2] - h[2], c[2] + h[2]


def _boxes_overlap(s, u, tol=1e-12):
    """True when two z-turned boxes share volume (separating axes on their footprints, then their heights)."""
    cs, As, hs, zs0, zs1 = _rect(s)
    cu, Au, hu, zu0, zu1 = _rect(u)
    if min(zs1, zu1) - max(zs0, zu0) <= tol:
        return False
    d = cu - cs
    for axis in (As[:, 0], As[:, 1], Au[:, 0], Au[:, 1]):
        rs = hs[0] * abs(axis @ As[:, 0]) + hs[1] * abs(axis @ As[:, 1])
        ru = hu[0] * abs(axis @ Au[:, 0]) + hu[1] * abs(axis @ Au[:, 1])
        if abs(axis @ d) >= rs + ru - tol:
            return False
    return True


def _in_frame(s, k):
    """A box's centre in panel k's frame (radial, along the panel, z)."""
    phi = T.panel_angle(k)
    c = np.asarray(s.center, dtype=float)
    return np.array([c[0] * math.cos(phi) + c[1] * math.sin(phi), -c[0] * math.sin(phi) + c[1] * math.cos(phi), c[2]])


class SyrupDisplay(unittest.TestCase):
    """The declared display rule: an amber disc in the cup rides at the ledger's level; the 10 mL line is drawn."""

    def plate_top(self, v):  # the disc's top over the cup's base for v m^3 in the cup
        p = T.SYRUP_PLATE.shapes[0]
        return p.center[2] + _height(p) / 2 + T.syrup_target(v)

    def test_target_follows_the_ledger(self):
        self.assertEqual(T.syrup_target(0.0), 0.0)
        self.assertAlmostEqual(T.syrup_target(10 * ML), 10 * ML / L.CUP_AREA + L.SYRUP_SINK, places=9)
        self.assertAlmostEqual(self.plate_top(10 * ML), L.LINE_Z, places=9)  # the disc's top at the 10 mL line
        for ml in (1.0, 5.0, 15.0, 30.0):
            self.assertAlmostEqual(self.plate_top(ml * ML), L.CUP_FLOOR + ml * ML / L.CUP_AREA, places=9)
        self.assertLess(T.syrup_target(5 * ML), T.syrup_target(10 * ML))

    def test_target_is_bounded_and_safe(self):
        self.assertEqual(T.syrup_target(1.0), L.SYRUP_RANGE)  # a cubic metre stops at the brim
        self.assertAlmostEqual(self.plate_top(1.0), L.CUP_H - L.CUP_SEAM, places=9)  # 0.1 mm under the rim
        for bad in (-1e-6, float("nan"), float("inf"), None, "x", [1]):
            self.assertEqual(T.syrup_target(bad), 0.0)
        self.assertLess(T.syrup_target(1e-12), 1e-4)

    def test_empty_cup_shows_no_syrup(self):
        """Empty, the disc is parked wholly inside the solid round floor: nothing amber shows in an empty cup."""
        p = T.SYRUP_PLATE.shapes[0]
        floor = T.SHELL[0]
        self.assertEqual((floor.kind, p.kind), ("cylinder", "cylinder"))
        self.assertEqual(floor.opacity, 1.0)
        self.assertLess(p.radius, floor.radius)
        self.assertGreater(p.center[2] - p.height / 2, floor.center[2] - floor.height / 2)  # strictly inside
        self.assertLess(p.center[2] + p.height / 2, floor.center[2] + floor.height / 2)
        self.assertAlmostEqual(self.plate_top(0.0), L.CUP_FLOOR - L.SYRUP_SINK, places=12)
        self.assertGreater(p.center[2] - p.height / 2, 0.0001)  # clear of the cup's base: never near the pan
        self.assertAlmostEqual(p.height, 0.0018, places=12)  # the thicker 1.8 mm edge
        self.assertLess(p.center[2] + p.height / 2, L.CUP_FLOOR)  # parked top under 2.0 mm
        self.assertGreater(p.center[2] - p.height / 2, 0.0)  # parked bottom over 0
        self.assertAlmostEqual(p.center[2] - p.height / 2, 0.00015, places=9)  # 0.15 mm over the base

    def test_joint_and_line(self):
        j = T.SYRUP_PLATE
        self.assertEqual(j.kind, "prismatic")
        self.assertGreater(j.spring, 0.0)  # a target does nothing without a spring
        self.assertAlmostEqual(L.LINE_Z, 0.0118, places=4)
        self.assertEqual(len(T.LINE_STRIPS), 6)  # outside on the top camera's half, not gripped, no numeral
        self.assertEqual(len(T.INNER_LINE_STRIPS), 14)  # inside on the rest but the four numeral panels
        for s in T.LINE_STRIPS:  # not on the panels the left hand grips (fingers close along x), on the -x half
            self.assertLess(abs(math.cos(_yaw(s))), math.cos(math.radians(2 * 15.0)) + 1e-9)
            self.assertLess(math.cos(_yaw(s)), 1e-9)

    def test_plate_stays_inside_the_cup_and_the_cup_still_rests_on_the_pan(self):
        lo = T.CUP.rest_state()
        self.assertAlmostEqual(float(lo.low[2]), L.PAN_TOP, places=4)
        self.assertLess(L.SYRUP_R, L.CUP_INNER_HALF)
        self.assertGreater(self.plate_top(1.0) - self.plate_top(0.0), 15 * ML / L.CUP_AREA)  # covers 15 mL


class CupLook(unittest.TestCase):
    """The look: a round cup of 24 clear flat panels on a solid white round floor, graduations at 5 to 30 mL: rings at
    10 (red), 20 and 30 mL round it, ticks at 5, 15 and 25 mL facing the arms."""

    WALLS = 1 + 24 + 24  # the shell: the floor, the 24 hidden lower panels and the 24 lips, the first base shapes

    def marks(self):
        return T.OUTER_MARKS + T.INNER_MARKS

    def info(self):
        return [(s, k, ml, False) for s, (k, ml) in zip(T.OUTER_MARKS, T.OUTER_INFO)] + \
               [(s, k, ml, True) for s, (k, ml) in zip(T.INNER_MARKS, T.INNER_INFO)]

    def test_round_cup_keeps_the_volume_scale(self):
        """24 panels whose inside faces stand 18 mm from the axis: 10.24 cm^2 inside, as the square cup's 32 x 32 mm,
        so every graduation stands where it did."""
        self.assertEqual(L.CUP_SIDES, 24)
        self.assertAlmostEqual(L.CUP_AREA * 1e4, 10.237, places=3)
        self.assertAlmostEqual(L.CUP_AREA, math.pi * 0.018 ** 2, delta=0.012 * L.CUP_AREA)  # within 1.2 % of a circle
        self.assertGreater(L.CUP_AREA * (L.CUP_H - L.CUP_FLOOR), 30 * ML)  # holds 30 mL (36.9 mL to the brim)

    def test_mark_heights_are_volume_over_inner_area(self):
        self.assertEqual(L.GRAD_ML, (5.0, 10.0, 15.0, 20.0, 25.0, 30.0))
        for ml, z in zip(L.GRAD_ML, L.GRAD_Z):
            self.assertAlmostEqual(z, L.CUP_FLOOR + ml * ML / L.CUP_AREA, delta=5e-5)
        self.assertEqual([round(z * 1000, 1) for z in L.GRAD_Z], [6.9, 11.8, 16.7, 21.5, 26.4, 31.3])  # the README
        self.assertAlmostEqual(L.GRAD_Z[1], L.LINE_Z, delta=5e-5)
        for s, k, ml, inside in self.info():
            self.assertAlmostEqual(s.center[2], L.GRAD_Z[L.GRAD_ML.index(ml)], places=12)
        self.assertLess(L.GRAD_Z[-1] + L.TICK_W / 2, L.CUP_H)  # 30 mL is below the rim
        gaps = np.diff(L.GRAD_Z)
        self.assertGreater(float(gaps.min()), (L.LINE_W + L.TICK_W) / 2 + 0.001)  # marks 1 mm or more apart

    def test_counts_and_panels(self):
        """Each ring once round the cup: outside on the 6 panels of the top camera's half that the hand does not grip
        and that carry no numeral, inside on 14 others (the far half and the three gripped near panels), on neither
        face of the four numeral panels; ticks at 5, 15 and 25 mL outside on TICK_PANELS_POS; no ticks inside."""
        self.assertEqual(L.RING_ML, (10.0, 20.0, 30.0))
        self.assertEqual(L.TICK_ML, (5.0, 15.0, 25.0))
        self.assertEqual(sorted(L.RING_ML + L.TICK_ML), list(L.GRAD_ML))
        self.assertEqual(T.OUTER_RING_PANELS, (8, 9, 10, 14, 15, 16))
        self.assertEqual(T.INNER_RING_PANELS, (0, 1, 2, 3, 4, 5, 11, 12, 13, 19, 20, 21, 22, 23))
        self.assertEqual(T.OUTER_TICK_PANELS, (3, 4, 5, 19, 20, 21))
        self.assertEqual(T.OUTER_TICK_PANELS, L.TICK_PANELS_POS)
        self.assertEqual(T.NUMERAL_PANELS, (6, 7, 17, 18))
        self.assertEqual(len(T.OUTER_MARKS), 3 * 6 + 3 * 6)
        self.assertEqual(len(T.INNER_MARKS), 3 * 14)
        self.assertEqual(len(T.INNER_LINE_STRIPS), 14)
        self.assertEqual({ml for k, ml in T.INNER_INFO}, set(L.RING_ML))
        for ml in L.RING_ML:  # every ring drawn once: each panel but the numeral panels carries it on exactly one
            inner = [k for k, m in T.INNER_INFO if m == ml]  # face, so the top camera never sees it twice
            outer = [k for k, m in T.OUTER_INFO if m == ml]
            self.assertEqual(sorted(inner + outer), [k for k in range(24) if k not in T.NUMERAL_PANELS])
        for k in T.OUTER_RING_PANELS:  # outside only where the top camera (at -x) looks straight at the face
            self.assertLess(math.cos(T.panel_angle(k)), 1e-9)
        self.assertEqual(T.CUP.shapes[:self.WALLS], T.SHELL)
        self.assertEqual(T.CUP.shapes[self.WALLS:], self.marks() + T.NUMERALS)
        for s, k, ml, inside in self.info():  # each strip faces its panel, sunk 0.3 mm, 0.3 (in) or 0.6 (out) proud
            self.assertAlmostEqual(math.cos(_yaw(s) - T.panel_angle(k)), 1.0, places=12)
            c = _in_frame(s, k)
            self.assertAlmostEqual(c[1], 0.0, places=12)
            r0, r1 = c[0] - s.size[0] / 2, c[0] + s.size[0] / 2
            if inside:
                self.assertAlmostEqual(r0, L.CUP_INNER_HALF - L.MARK_IN_T, places=12)
                self.assertAlmostEqual(r1, L.CUP_INNER_HALF + L.MARK_EMBED, places=12)
            else:
                self.assertAlmostEqual(r0, L.CUP_OUTER / 2 - L.MARK_EMBED, places=12)
                self.assertAlmostEqual(r1, L.CUP_OUTER / 2 + L.LINE_T, places=12)

    def test_every_mark_is_inside_its_panel_and_clear_of_the_others(self):
        """The sunk 0.3 mm of each mark lies inside its own see-through panel, clear of its edges (one piece with it);
        no mark shares volume with another mark, with any other panel or with a lip."""
        panels = T.WALLS
        for s, k, ml, inside in self.info():
            c = _in_frame(s, k)
            self.assertLessEqual(L.MARK_L / 2, L.PANEL_L / 2 - 1e-4)  # 0.12 mm inside the panel's ends
            z0, z1 = c[2] - s.size[2] / 2, c[2] + s.size[2] / 2
            self.assertGreater(z0, L.CUP_FLOOR + L.CUP_SEAM)
            self.assertLess(z1, L.CUP_H - L.RIM_H - L.CUP_SEAM)  # under the lip: on the hidden lower part
            hosts = [j for j, w in enumerate(panels) if _boxes_overlap(s, w)]
            self.assertEqual(hosts, [k], (k, ml, inside))
            self.assertFalse(any(_boxes_overlap(s, lip) for lip in T.LIPS))
        marks = self.marks() + T.NUMERALS
        for i in range(len(marks)):
            for j in range(i + 1, len(marks)):
                self.assertFalse(_boxes_overlap(marks[i], marks[j]), (i, j))

    def test_panels_make_a_closed_ring_without_overlaps(self):
        """24 panels, one every 15 degrees, inside faces 18 mm and outside faces 20 mm from the axis, 0.1 mm clear of
        each other at the inner corners and of the floor, each a hidden lower part up to 35.4 mm and a lip from 35.5
        to 38.0 mm of the same radial span and width; none shares volume with another."""
        panels = T.WALLS
        self.assertEqual(len(panels), 24)
        self.assertEqual(len(T.LIPS), 24)
        self.assertEqual(T.SHELL, (T.SHELL[0],) + T.WALLS + T.LIPS)
        for k, (w, lip) in enumerate(zip(panels, T.LIPS)):
            for part in (w, lip):
                c = _in_frame(part, k)
                self.assertAlmostEqual(c[0] - part.size[0] / 2, L.CUP_INNER_HALF, places=12)
                self.assertAlmostEqual(c[0] + part.size[0] / 2, L.CUP_OUTER / 2, places=12)
                self.assertAlmostEqual(c[1], 0.0, places=12)
                self.assertAlmostEqual(part.size[1], 2 * L.CUP_INNER_HALF * math.tan(math.pi / 24) - L.CUP_SEAM,
                                       places=12)
            self.assertAlmostEqual(w.center[2] + w.size[2] / 2, L.CUP_H - L.RIM_H - L.CUP_SEAM, places=12)
            self.assertAlmostEqual(w.center[2] - w.size[2] / 2, L.CUP_FLOOR + L.CUP_SEAM, places=12)
            self.assertAlmostEqual(lip.center[2] - lip.size[2] / 2, L.CUP_H - L.RIM_H, places=12)
            self.assertAlmostEqual(lip.center[2] + lip.size[2] / 2, L.CUP_H, places=12)
        self.assertAlmostEqual(L.CUP_H - L.RIM_H, 0.0355, places=12)  # the seam at 35.5 mm
        self.assertLess(P.CUP_GRIP_Z, L.CUP_H - L.RIM_H - 0.015)  # the pads grip at 19 mm, far below the seam
        shell = panels + T.LIPS
        for i in range(len(shell)):
            for j in range(i + 1, len(shell)):
                self.assertFalse(_boxes_overlap(shell[i], shell[j]), (i, j))
        # the inner corners: neighbours' inside edges 0.1 mm apart along the ring
        e0 = np.array([L.CUP_INNER_HALF, L.PANEL_L / 2])
        phi = 2 * math.pi / 24
        e1 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]) @ np.array(
            [L.CUP_INNER_HALF, -L.PANEL_L / 2])
        self.assertAlmostEqual(float(np.linalg.norm(e1 - e0)), L.CUP_SEAM, delta=1e-6)
        floor = T.SHELL[0]
        self.assertEqual(floor.kind, "cylinder")
        self.assertAlmostEqual(floor.radius, L.CUP_OUTER / 2, places=12)
        self.assertAlmostEqual(floor.center[2] + floor.height / 2, L.CUP_FLOOR, places=12)
        self.assertAlmostEqual(floor.center[2] - floor.height / 2, 0.0, places=12)

    def test_no_two_faces_coincide(self):
        """See-through walls show coincident faces as flicker. Shapes turned to different panels have no parallel
        side faces in common (15 degree steps) and share no volume (above); every flat top or bottom face of the cup
        stands at a height of its own wherever two footprints overlap, at any fill level of the disc."""
        shapes = [s for s in T.CUP.shapes if s.kind == "box"]
        for group in (T.WALLS, T.LIPS, T.NUMERALS, T.OUTER_MARKS, T.INNER_MARKS):  # lips and numerals included
            self.assertTrue(all(s in shapes for s in group))
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                a, b = shapes[i], shapes[j]
                if abs(math.sin(2 * (_yaw(a) - _yaw(b)))) > 1e-9:
                    continue  # not parallel or at right angles: no faces on one plane over an area
                yaw = _yaw(a)  # the same panel's direction (or 90 degrees on): compare in a's frame
                rot = lambda c: np.array([c[0] * math.cos(yaw) + c[1] * math.sin(yaw),  # noqa: E731
                                          -c[0] * math.sin(yaw) + c[1] * math.cos(yaw), c[2]])
                ca, cb = rot(np.asarray(a.center)), rot(np.asarray(b.center))
                ha = np.asarray(a.size) / 2
                hb = np.asarray(b.size) / 2 if abs(math.cos(_yaw(a) - _yaw(b))) > 0.5 else \
                    np.asarray((b.size[1], b.size[0], b.size[2])) / 2
                for ax in range(3):
                    rest = [m for m in range(3) if m != ax]
                    for fa in (ca[ax] - ha[ax], ca[ax] + ha[ax]):
                        for fb in (cb[ax] - hb[ax], cb[ax] + hb[ax]):
                            if abs(fa - fb) < 1e-9:
                                self.assertFalse(all(min(ca[m] + ha[m], cb[m] + hb[m]) -
                                                     max(ca[m] - ha[m], cb[m] - hb[m]) > 1e-9 for m in rest), (i, j))
        floor, p = T.SHELL[0], T.SYRUP_PLATE.shapes[0]
        flats = {round(floor.center[2] + floor.height / 2, 9), round(floor.center[2] - floor.height / 2, 9)}
        for s in shapes:
            self.assertFalse({round(s.center[2] - s.size[2] / 2, 9), round(s.center[2] + s.size[2] / 2, 9)} & flats)
        for ml in (0.0, 0.5, 2.0, 5.0, 9.0, 10.0, 11.0, 15.0, 30.0, 40.0):
            z = p.center[2] + T.syrup_target(ml * ML)
            disc = {round(z - p.height / 2, 9), round(z + p.height / 2, 9)}
            self.assertFalse(disc & flats, ml)
            for s in T.INNER_MARKS:  # the disc's faces never on an inside mark's top or bottom
                self.assertFalse(disc & {round(s.center[2] - s.size[2] / 2, 9), round(s.center[2] + s.size[2] / 2, 9)})

    def test_ten_ml_line_is_the_longest_and_thickest(self):
        for group, info in ((T.OUTER_MARKS, T.OUTER_INFO), (T.INNER_MARKS, T.INNER_INFO)):
            line = [s for s, (k, ml) in zip(group, info) if ml == L.DOSE_ML]
            others = [s for s, (k, ml) in zip(group, info) if ml != L.DOSE_ML]
            for ml in L.GRAD_ML:  # no graduation is drawn longer than the 10 mL line
                self.assertLessEqual(sum(1 for k, m in info if m == ml), len(line))
            self.assertGreaterEqual(len(line), 6)  # round the cup, not one face
            self.assertGreater(min(s.size[2] for s in line), max(s.size[2] for s in others) + 1e-4)
        self.assertEqual(len(T.LINE_STRIPS) + len(T.INNER_LINE_STRIPS), 20)  # 24 panels less the four numeral panels
        for s in self.marks():
            self.assertGreaterEqual(s.size[0], 0.0006 - 1e-12)  # 0.3 or 0.6 mm proud plus 0.3 mm sunk
            self.assertLessEqual(s.size[0], 0.0009 + 1e-12)
            self.assertAlmostEqual(s.size[1], L.MARK_L, places=12)

    def test_marks_are_drawn_for_the_cameras(self):
        """Bolder than real print: the 10 mL line red and 6 mm tall (the 9 to 11 mL window its middle third), the 20
        and 30 mL rings near black and 4 mm tall, rings round the cup; the 5, 15 and 25 mL ticks near black, 1.5 mm
        tall, over 45 degrees of arc (about 15 mm outside). At the top camera's 1.8 px/mm in the 2880 px screenshot,
        about 0.8 px per mm of a wall's height: the red line 4.8 px, the rings 3.2 px."""
        self.assertAlmostEqual(L.LINE_W, 0.006, places=12)
        self.assertAlmostEqual(L.RING_W, 0.004, places=12)
        self.assertGreaterEqual(L.TICK_W, 0.0015 - 1e-12)
        lo_z, hi_z = (L.CUP_FLOOR + v * ML / L.CUP_AREA for v in L.WINDOW)  # the window: the line's middle third
        self.assertAlmostEqual((hi_z - lo_z) / L.LINE_W, 1 / 3, delta=0.02)
        self.assertAlmostEqual((lo_z + hi_z) / 2, L.LINE_Z, places=9)
        self.assertEqual(L.TICK_PANELS, 3)
        self.assertAlmostEqual(L.TICK_PANELS * 2 * math.pi / 24 * L.CUP_OUTER / 2, 0.0157, delta=5e-4)
        self.assertEqual(L.LINE_RED, (0.85, 0.05, 0.05))
        r, g, b = L.LINE_RED
        self.assertGreater(r - max(g, b), 0.7)  # a strongly saturated red
        for pale in (L.CLEAR_EDGE, L.CUP_BASE, L.PAN_WHITE, L.CUP_WALL_FALLBACK):
            self.assertGreater(sum(abs(a - c) for a, c in zip(L.LINE_RED, pale)), 1.5)  # far from the clear cup
            self.assertGreater(sum(abs(a - c) for a, c in zip(L.INK, pale)), 1.9)  # near black on pale
        for pale in (L.CUP_BASE, L.PAN_WHITE, L.CUP_WALL_FALLBACK):
            self.assertGreater(sum(abs(a - c) for a, c in zip(L.INK, pale)), 2.4)  # near black on white
        self.assertGreater(sum(abs(a - c) for a, c in zip(L.LINE_RED, L.SYRUP_GOLD)), 0.6)  # and from the syrup
        for s, k, ml, inside in self.info():
            line = ml == L.DOSE_ML
            self.assertEqual(s.color, L.LINE_RED if line else L.INK)
            self.assertAlmostEqual(s.size[2], T.mark_height(ml), places=12)
            self.assertAlmostEqual(s.size[2], L.LINE_W if line else (L.RING_W if ml in L.RING_ML else L.TICK_W),
                                   places=12)
        self.assertAlmostEqual(L.GRAD_Z[-1] + L.RING_W / 2, 0.0333, delta=5e-5)  # the 30 mL ring: 2 mm under the lip
        self.assertAlmostEqual(L.GRAD_Z[3] - L.RING_W / 2, 0.0195, delta=5e-5)  # the 20 mL ring: 19.5 to 23.5 mm
        self.assertAlmostEqual(L.GRAD_Z[3] + L.RING_W / 2, 0.0235, delta=5e-5)
        self.assertAlmostEqual(L.GRAD_Z[5] - L.RING_W / 2, 0.0293, delta=5e-5)
        z = dict(zip(L.GRAD_ML, L.GRAD_Z))
        for a, b in zip(L.GRAD_ML, L.GRAD_ML[1:]):  # neighbouring marks 1 mm or more apart
            self.assertGreater(z[b] - z[a] - (T.mark_height(a) + T.mark_height(b)) / 2, 0.001)

    def test_no_mark_on_a_gripped_panel(self):
        """The left hand's fingers close along x on the panels facing +-x (outer faces at x +-20 mm): no outside mark
        on those panels or their neighbours, every mark 1 mm or more inside the gripped faces' planes."""
        self.assertAlmostEqual(abs(P.left_approach()[0]), 0.0, places=9)  # the hand comes in along y, fingers on x
        for k, ml in T.OUTER_INFO:
            self.assertGreater(min(T._ring_dist(k, 0), T._ring_dist(k, 12)), 1)
        for k, v, d, n in T.NUMERAL_INFO:
            self.assertGreater(min(T._ring_dist(k, 0), T._ring_dist(k, 12)), 1)
        for s in self.marks() + T.NUMERALS:
            c, A, h, _, _ = _rect(s)
            corners = [c + A @ np.array([sx * h[0], sy * h[1]]) for sx in (-1, 1) for sy in (-1, 1)]
            self.assertLess(max(abs(float(q[0])) for q in corners), L.CUP_OUTER / 2 - 0.001)
        lo, hi = T.CUP.local_bounds()
        self.assertAlmostEqual(float(hi[0] - lo[0]), L.CUP_OUTER, places=6)  # x extent: the gripped panels only
        self.assertAlmostEqual(float(hi[1] - lo[1]), L.CUP_OUTER + 2 * L.LINE_T, places=6)  # the 10 mL line at +-y
        self.assertAlmostEqual(float(hi[2] - lo[2]), L.CUP_H, places=9)
        self.assertEqual(T.CUP.mass, L.CUP_MASS)
        self.assertEqual(T.CUP.surface_friction(), mk.DEFAULT_SURFACE_FRICTION)

    def test_inner_marks_do_not_collide_with_the_plate(self):
        plan = mk.usd_plan(T.CUP, "/World/cup")
        self.assertFalse(plan["self_collisions"])  # the disc (a link) never collides with the cup's own body
        self.assertEqual([j["name"] for j in plan["joints"]], ["syrup"])
        disc = T.SYRUP_PLATE.shapes[0]
        self.assertAlmostEqual(disc.radius, L.SYRUP_R, places=12)
        for s, k, ml, inside in self.info():  # every inside mark's face is a flat 17.7 mm out: 0.1 mm clear
            if inside:
                face = _in_frame(s, k)[0] - s.size[0] / 2
                self.assertAlmostEqual(face - disc.radius, L.CUP_SEAM, places=12)

    def test_walls_hidden_lips_opaque_no_opacity(self):
        """The clear wall is not rendered: every lower panel is hidden and still a collider, its size and pose those
        of 2b's panel below the lip seam; the lips are drawn and opaque; nothing anywhere has an opacity below 1."""
        r = L.CUP_INNER_HALF + L.CUP_WALL / 2
        h_2b = L.CUP_H - L.CUP_FLOOR - L.CUP_SEAM  # 2b: one panel from 2.1 mm up to the 38 mm rim
        for k, (w, lip) in enumerate(zip(T.WALLS, T.LIPS)):
            old = T._ring(k, r, L.CUP_H - h_2b / 2, (L.CUP_WALL, L.PANEL_L, h_2b))
            self.assertTrue(w.hidden)
            self.assertEqual(w.opacity, 1.0)
            self.assertEqual(w.color, L.CUP_WALL_FALLBACK)
            np.testing.assert_allclose(w.center[:2], old.center[:2], atol=1e-15)
            self.assertEqual(w.quat, old.quat)
            self.assertEqual(w.size[:2], old.size[:2])
            self.assertAlmostEqual(w.center[2] - w.size[2] / 2, old.center[2] - old.size[2] / 2, places=12)
            self.assertAlmostEqual(w.center[2] + w.size[2] / 2, 0.0355 - L.CUP_SEAM, places=12)
            self.assertFalse(lip.hidden)
            self.assertEqual(lip.opacity, 1.0)
            self.assertEqual(lip.color, L.CLEAR_EDGE)
            np.testing.assert_allclose(lip.center[:2], old.center[:2], atol=1e-15)
            self.assertEqual(lip.size[:2], old.size[:2])
        for obj in (T.CUP, T.BOTTLE):
            for s in obj.shapes:
                self.assertEqual(s.opacity, 1.0)
        for s in T.SYRUP_PLATE.shapes:
            self.assertEqual(s.opacity, 1.0)
            self.assertFalse(s.hidden)
        hidden = [s for s in T.CUP.shapes if s.hidden]
        self.assertEqual(hidden, list(T.WALLS))  # only the 24 lower panels
        self.assertFalse(any(s.hidden for s in T.BOTTLE.shapes))
        self.assertFalse(hasattr(L, "CUP_OPACITY"))
        self.assertFalse(hasattr(L, "CLEAR"))
        self.assertEqual(T.SHELL[0].color, L.CUP_BASE)  # the opaque pale base disc
        self.assertEqual(T.CUP.color, L.CUP_BASE)
        self.assertEqual(L.CUP_WALL_FALLBACK, L.PAN_WHITE)  # if the hide fails: a white cup, never grey
        self.assertEqual(T.SCALE.color, L.PAN_WHITE)
        self.assertEqual(T.SYRUP_PLATE.shapes[0].color, L.SYRUP_GOLD)
        self.assertGreater(sum(L.SYRUP_GOLD) - sum(L.AMBER), 0.7)  # the syrup far lighter than the amber bottle
        plan = mk.usd_plan(T.CUP, "/World/cup")
        by_body = {}
        for s in plan["shapes"]:
            by_body.setdefault(s["body"], []).append((s["opacity"], s["hidden"]))
        self.assertEqual(by_body["base"][:self.WALLS], [(1.0, False)] + [(1.0, True)] * 24 + [(1.0, False)] * 24)
        self.assertEqual(set(by_body["base"][self.WALLS:]), {(1.0, False)})
        self.assertEqual(by_body["syrup"], [(1.0, False)])
        self.assertEqual(len(plan["shapes"]), len(T.CUP.shapes) + 1)  # every shape, hidden or not, is a collider
        self.assertEqual(T.CUP.mass, L.CUP_MASS)  # mass at assembly level: 8 g whatever the shape count


class Numerals(unittest.TestCase):
    """Printed numerals 10, 20 and 30 beside the lines on the panels facing each arm (seven-segment digits)."""

    def test_counts_and_colours(self):
        self.assertEqual(len(T.NUMERALS), 60)  # 8 + 11 + 11 per side
        per_value = {}
        for k, v, d, n in T.NUMERAL_INFO:
            per_value[v] = per_value.get(v, 0) + n
        self.assertEqual(per_value, {10: 16, 20: 22, 30: 22})
        i = 0
        for k, v, d, n in T.NUMERAL_INFO:
            for s in T.NUMERALS[i:i + n]:
                self.assertEqual(s.color, L.LINE_RED if v == 10 else L.INK)
                self.assertFalse(s.hidden)
                self.assertEqual(s.opacity, 1.0)
            i += n
        self.assertEqual(T.DIGIT_SEGMENTS, {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd"})

    def segs(self):
        i = 0
        for k, v, d, n in T.NUMERAL_INFO:
            for name, s in zip(T.DIGIT_SEGMENTS[d], T.NUMERALS[i:i + n]):
                yield k, v, d, name, s
            i += n

    def test_each_segment_fits_its_panel(self):
        """Inside the panel's 4.64 mm width, between 2 mm and the 35.5 mm lip seam, standing on the outside face
        (0.6 mm proud, 0.3 mm sunk), on its own panel only; no two segments touch."""
        for k, v, d, name, s in self.segs():
            self.assertAlmostEqual(math.cos(_yaw(s) - T.panel_angle(k)), 1.0, places=12)
            c = _in_frame(s, k)
            self.assertLessEqual(abs(c[1]) + s.size[1] / 2, L.PANEL_L / 2 + 1e-12)
            z0, z1 = c[2] - s.size[2] / 2, c[2] + s.size[2] / 2
            self.assertGreater(z0, 0.002)
            self.assertLess(z1, L.CUP_H - L.RIM_H)
            self.assertAlmostEqual(c[0] - s.size[0] / 2, L.CUP_OUTER / 2 - L.MARK_EMBED, places=12)
            self.assertAlmostEqual(c[0] + s.size[0] / 2, L.CUP_OUTER / 2 + L.NUM_PROUD, places=12)
            hosts = [j for j, w in enumerate(T.WALLS) if _boxes_overlap(s, w)]
            self.assertEqual(hosts, [k])
            self.assertFalse(any(_boxes_overlap(s, lip) for lip in T.LIPS))
        tops = [(_in_frame(s, k)[2] + s.size[2] / 2) for k, v, d, name, s in self.segs() if v == 30]
        self.assertAlmostEqual(max(tops), 0.0348, delta=5e-5)  # the 30 under the 35.5 mm lip
        self.assertEqual(L.NUM_Z, {10: L.GRAD_Z[1], 20: L.GRAD_Z[3], 30: L.GRAD_Z[5]})
        segs = list(self.segs())
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                self.assertFalse(_boxes_overlap(segs[i][4], segs[j][4]), (segs[i][:4], segs[j][:4]))
        for k, v, d, name, s in segs:  # within a digit: 0.1 mm gaps (the boxes are never closer)
            for k2, v2, d2, name2, s2 in segs:
                if (k2, v2) != (k, v) or name2 == name:
                    continue
                a, b = _in_frame(s, k), _in_frame(s2, k)
                gap = max(abs(a[1] - b[1]) - (s.size[1] + s2.size[1]) / 2, abs(a[2] - b[2]) - (s.size[2] + s2.size[2]) / 2)
                self.assertGreaterEqual(gap, 0.0001 - 1e-12, (k, v, name, name2))

    def test_segment_sizes(self):
        for k, v, d, name, s in self.segs():
            if name in "agd":
                self.assertAlmostEqual(s.size[1], 0.004, places=12)
                self.assertAlmostEqual(s.size[2], 0.001, places=12)
            else:
                self.assertAlmostEqual(s.size[1], 0.001, places=12)
                self.assertAlmostEqual(s.size[2], 0.0018, places=12)
            self.assertAlmostEqual(s.size[0], L.NUM_PROUD + L.MARK_EMBED, places=12)

    def test_reading_order(self):
        """For a reader outside the cup on each side, the tens digit is on the left: project both digit panels'
        centres on the reader's right axis (+y reader: -x; -y reader: +x). Within a digit the b and c strokes sit at
        +t, the panel's +tangential axis, which is the outside reader's right."""
        right = {"+y": np.array([-1.0, 0.0]), "-y": np.array([1.0, 0.0])}
        for side, (tens, units) in L.NUMERAL_PANELS.items():
            ct = np.array([math.cos(T.panel_angle(tens)), math.sin(T.panel_angle(tens))])
            cu = np.array([math.cos(T.panel_angle(units)), math.sin(T.panel_angle(units))])
            self.assertLess(float(ct @ right[side]), float(cu @ right[side]), side)
            phi = T.panel_angle(tens)  # the reader's right is the panel's +t: forward (-radial) x up
            fwd = -np.array([math.cos(phi), math.sin(phi), 0.0])
            r3 = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            np.testing.assert_allclose(r3, (-math.sin(phi), math.cos(phi), 0.0), atol=1e-12)
            self.assertGreater(float(r3[:2] @ right[side]), math.cos(math.radians(15.0)) - 1e-9)
        self.assertEqual(L.NUMERAL_PANELS, {"+y": (6, 7), "-y": (17, 18)})
        for k, v, d, name, s in self.segs():
            t = _in_frame(s, k)[1]
            if name in "bc":
                self.assertAlmostEqual(t, L.NUM_SEG_T, places=12)
            elif name in "fe":
                self.assertAlmostEqual(t, -L.NUM_SEG_T, places=12)
            else:
                self.assertAlmostEqual(t, 0.0, places=12)
        for k, v, d, n in T.NUMERAL_INFO:  # tens digit on the tens panel
            side = "+y" if k in L.NUMERAL_PANELS["+y"] else "-y"
            tens, units = L.NUMERAL_PANELS[side]
            self.assertEqual(d, v // 10 if k == tens else 0)

    def test_rings_skip_the_numeral_panels(self):
        for k, ml in T.OUTER_INFO + T.INNER_INFO:
            if ml in L.RING_ML:
                self.assertNotIn(k, T.NUMERAL_PANELS)
        for k in T.OUTER_TICK_PANELS:
            self.assertNotIn(k, T.NUMERAL_PANELS)
        for k in T.NUMERAL_PANELS:  # nothing but the numerals on those panels
            self.assertNotIn(k, [j for j, ml in T.OUTER_INFO + T.INNER_INFO])

    def test_nothing_proud_on_the_gripped_panels(self):
        """No mark, numeral or lip stands outside the outer face of a gripped panel (0, 12) or of its neighbours (1,
        11, 13, 23): the fingers close there on the bare wall."""
        near = (0, 1, 11, 12, 13, 23)
        on_near = 0
        for s in T.OUTER_MARKS + T.INNER_MARKS + T.NUMERALS + T.LIPS:
            for k in near:
                if abs(math.cos(_yaw(s) - T.panel_angle(k)) - 1.0) > 1e-12:
                    continue  # not turned to panel k
                c = _in_frame(s, k)
                if abs(c[1]) > L.PANEL_L:
                    continue  # turned that way but not on that panel
                on_near += 1
                self.assertLessEqual(c[0] + s.size[0] / 2, L.CUP_OUTER / 2 + 1e-12, k)
        self.assertEqual(on_near, len(near) + 3 * len([k for k in near if k in T.INNER_RING_PANELS]))  # lips + inside


class BottleLooks(unittest.TestCase):
    """Looks only: the bottle's label and opening."""

    def test_label_and_opening_are_thin_looks(self):
        body, neck, label, opening = T.BOTTLE.shapes
        self.assertAlmostEqual(label.radius - L.BODY_R, 0.0003, places=12)  # 0.3 mm proud of the body
        self.assertLess(label.center[2] + label.height / 2, L.BODY_H)
        self.assertGreater(label.center[2] - label.height / 2, 0.0)
        self.assertLess(label.center[2] - label.height / 2, P.GRIP_Z)  # the hand grips the label, as a person does
        self.assertAlmostEqual(opening.radius, L.LIP_R, places=12)
        self.assertAlmostEqual(opening.center[2] + opening.height / 2, L.NECK_TOP + 0.0001, places=12)
        lo, hi = T.BOTTLE.local_bounds()
        self.assertAlmostEqual(float(hi[2]), L.NECK_TOP + 0.0001, places=9)
        self.assertAlmostEqual(float(lo[2]), 0.0, places=12)


class MechkitOpacity(unittest.TestCase):
    """This task's mechkit copy adds a look opacity; the default 1.0 leaves every shape, plan and look unchanged."""

    def test_default_is_solid_and_unchanged(self):
        a = mk.box((0.0, 0.0, 0.01), (0.02, 0.02, 0.02), (0.1, 0.2, 0.3))
        b = mk.box((0.0, 0.0, 0.01), (0.02, 0.02, 0.02), (0.1, 0.2, 0.3), opacity=1.0)
        self.assertEqual(a.opacity, 1.0)
        self.assertEqual(a, b)
        c = mk.cylinder((0.0, 0.0, 0.01), 0.01, 0.02, "Z", (0.1, 0.2, 0.3))
        self.assertEqual(c, mk.cylinder((0.0, 0.0, 0.01), 0.01, 0.02, "Z", (0.1, 0.2, 0.3), opacity=1.0))
        self.assertEqual(mk.open_box((0, 0, 0.02), (0.05, 0.05, 0.04), 0.002),
                         mk.open_box((0, 0, 0.02), (0.05, 0.05, 0.04), 0.002, opacity=1.0))
        positional = mk.Shape("box", (0.0, 0.0, 0.01), (0.02, 0.02, 0.02), 0.0, 0.0, "Z", (0.1, 0.2, 0.3), None)
        self.assertEqual(positional, a)  # the old positional fields still line up

    def test_other_plans_are_unchanged(self):
        plan = mk.usd_plan(T.BOTTLE, "/World/bottle")
        self.assertEqual({s["opacity"] for s in plan["shapes"]}, {1.0})
        shown = mk.assembly("cup_shown", origin=(L.CUP_X, L.CUP_Y, L.CUP_Z), mass=L.CUP_MASS, joints=[T.SYRUP_PLATE],
                            shapes=[dataclasses.replace(s, hidden=False) for s in T.CUP.shapes], color=L.CUP_BASE)
        p0 = mk.usd_plan(shown, "/World/c")
        p1 = mk.usd_plan(T.CUP, "/World/c")
        strip = lambda p: [{k: v for k, v in s.items() if k != "hidden"} for s in p["shapes"]]  # noqa: E731
        self.assertEqual(strip(p0), strip(p1))  # colliders, places and colours: identical
        self.assertEqual({k: v for k, v in p0.items() if k != "shapes"}, {k: v for k, v in p1.items() if k != "shapes"})

    def test_opacity_is_validated(self):
        for bad in (0.0, -0.1, 1.5, float("nan"), "x", None):
            with self.assertRaises(ValueError):
                mk.box((0, 0, 0), (0.01, 0.01, 0.01), opacity=bad)
        with self.assertRaises(ValueError):
            mk.Shape("box", (0, 0, 0), (0.01, 0.01, 0.01), opacity=2.0).check()
        self.assertEqual(mk.box((0, 0, 0), (0.01, 0.01, 0.01), opacity=0.35).opacity, 0.35)

    def test_look_passes_opacity_only_below_one(self):
        src = (HERE / "mechkit.py").read_text()
        self.assertIn("cfg.opacity = float(opacity)", src)
        self.assertIn("if opacity < 1.0:", src)
        self.assertIn("_see_through(stage, path, float(opacity))", src)
        self.assertEqual(mk.FRACTIONAL_CUTOUT_SETTING, "/rtx/raytracing/fractionalCutoutOpacity")
        self.assertLess(mk.SEE_THROUGH_ROUGHNESS, mk.LOOK_ROUGHNESS)

    def test_see_through_never_raises_offline(self):
        """Without pxr shaders or carb (offline) it writes nothing and returns 0; it never stops the scene."""
        import contextlib
        import io

        class NoStage:
            def GetPrimAtPath(self, path):
                raise RuntimeError("no stage")

        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(mk._see_through(NoStage(), "/World/cup/Looks/color_1", 0.55), 0)
        self.assertIn("0 shader(s) written", err.getvalue())
        self.assertIn("fractional cut-out opacity UNKNOWN", err.getvalue())  # no renderer offline: not claimed ON

    def test_see_through_log_says_on_or_off(self):
        """The GPU log line ends in ON only when the renderer reads the setting back as true."""
        import contextlib
        import io

        class NoStage:
            def GetPrimAtPath(self, path):
                raise RuntimeError("no stage")

        for stored, word in ((True, "ON"), (False, "OFF"), (None, "UNKNOWN")):
            class Settings:
                def get(self, key, _v=stored):
                    return _v

                def set_bool(self, key, value):
                    pass

            carb = types.ModuleType("carb")
            carb.settings = types.SimpleNamespace(get_settings=lambda s=Settings(): s)
            saved = sys.modules.get("carb")
            sys.modules["carb"] = carb
            try:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    mk._see_through(NoStage(), "/World/cup/Looks/color_1", 0.55)
            finally:
                if saved is None:
                    sys.modules.pop("carb", None)
                else:
                    sys.modules["carb"] = saved
            line = [x for x in err.getvalue().splitlines() if "see-through look" in x]
            self.assertEqual(len(line), 1)
            self.assertTrue(line[0].endswith(f"fractional cut-out opacity {word}"), line[0])
            self.assertIn("opacity 0.55", line[0])

    def test_see_through_writes_the_shader(self):
        """With pxr (when installed): opacity, threshold 0 and roughness land on the UsdPreviewSurface shader."""
        try:
            from pxr import Sdf, Usd, UsdShade
        except ImportError:
            self.skipTest("pxr not installed")
        import contextlib
        import io

        stage = Usd.Stage.CreateInMemory()
        UsdShade.Material.Define(stage, "/Looks/m")
        sh = UsdShade.Shader.Define(stage, "/Looks/m/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(mk._see_through(stage, "/Looks/m", 0.55), 1)
        self.assertAlmostEqual(sh.GetInput("opacity").Get(), 0.55, places=6)
        self.assertEqual(sh.GetInput("opacityThreshold").Get(), 0.0)
        self.assertAlmostEqual(sh.GetInput("roughness").Get(), mk.SEE_THROUGH_ROUGHNESS, places=6)


class MechkitHidden(unittest.TestCase):
    """This task's mechkit copy adds a hidden look: the collider is kept, nothing is drawn."""

    def test_default_not_hidden_and_validated(self):
        a = mk.box((0.0, 0.0, 0.01), (0.02, 0.02, 0.02), (0.1, 0.2, 0.3))
        self.assertFalse(a.hidden)
        self.assertEqual(a, mk.box((0.0, 0.0, 0.01), (0.02, 0.02, 0.02), (0.1, 0.2, 0.3), hidden=False))
        h = mk.box((0.0, 0.0, 0.01), (0.02, 0.02, 0.02), (0.1, 0.2, 0.3), hidden=True)
        self.assertTrue(h.hidden)
        self.assertEqual(dataclasses.replace(h, hidden=False), a)  # nothing else changes
        self.assertTrue(mk.cylinder((0, 0, 0.01), 0.01, 0.02, "Z", hidden=True).hidden)
        self.assertTrue(all(s.hidden for s in mk.open_box((0, 0, 0.02), (0.05, 0.05, 0.04), 0.002, hidden=True)))
        for bad in ("yes", 1, None, 0.0):
            with self.assertRaises(ValueError):
                mk.box((0, 0, 0), (0.01, 0.01, 0.01), hidden=bad)
        with self.assertRaises(ValueError):
            mk.Shape("box", (0, 0, 0), (0.01, 0.01, 0.01), hidden="no").check()

    def test_hide_mode_env(self):
        saved = os.environ.pop("MECHKIT_HIDE_MODE", None)
        try:
            self.assertEqual(mk.hide_mode(), "visibility")
            os.environ["MECHKIT_HIDE_MODE"] = "purpose"
            self.assertEqual(mk.hide_mode(), "purpose")
            os.environ["MECHKIT_HIDE_MODE"] = "junk"
            self.assertEqual(mk.hide_mode(), "visibility")
        finally:
            os.environ.pop("MECHKIT_HIDE_MODE", None)
            if saved is not None:
                os.environ["MECHKIT_HIDE_MODE"] = saved

    def _fake_pxr(self, calls):
        class Imageable:
            def __init__(self, geom):
                self.geom = geom

            def MakeInvisible(self):
                calls.append(("invisible", self.geom))

            def CreatePurposeAttr(self, value):
                calls.append(("purpose", self.geom, value))

        usdgeom = types.SimpleNamespace(Imageable=Imageable, Tokens=types.SimpleNamespace(guide="guide"))
        pxr = types.ModuleType("pxr")
        pxr.UsdGeom = usdgeom
        return pxr

    def test_hide_writes_visibility_then_purpose_only_on_request(self):
        import contextlib
        import io

        for mode, want in (("visibility", [("invisible", "g")]),
                           ("purpose", [("invisible", "g"), ("purpose", "g", "guide")])):
            calls = []
            saved = sys.modules.get("pxr")
            sys.modules["pxr"] = self._fake_pxr(calls)
            try:
                self.assertTrue(mk._hide("g", mode))
            finally:
                if saved is None:
                    sys.modules.pop("pxr", None)
                else:
                    sys.modules["pxr"] = saved
            self.assertEqual(calls, want, mode)
        if "pxr" not in sys.modules:  # offline without pxr: never raises, says so, reports no hide
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertFalse(mk._hide("g"))
            self.assertIn("hide not written", err.getvalue())

    def test_spawner_keeps_collider_and_both_materials(self):
        """In the spawner a hidden shape gets its collider first, then the hide, then the physics material and its
        visual material (the fallback look); one stderr line per assembly counts the hidden shapes."""
        src = (HERE / "mechkit.py").read_text()
        body = src[src.index("def _author("):]
        body = body[:body.index("\ndef ")]
        order = ["UsdPhysics.CollisionAPI.Apply(prim)", "physx.collider(prim, plan)", "_hide(geom)",
                 "bind_physics_material(s[\"path\"]", "bind_visual_material(s[\"path\"]"]
        at = [body.index(x) for x in order]
        self.assertEqual(at, sorted(at))
        self.assertNotIn("continue", body[at[0]:at[-1]])
        self.assertIn("shapes drawn invisible (colliders kept)", body)
        plan = mk.usd_plan(T.CUP, "/World/cup")
        self.assertEqual(sum(1 for s in plan["shapes"] if s["hidden"]), 24)


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
        """rigstate.py is the kit's byte for byte. mechkit.py is the kit's plus the look opacity and the hidden look:
        every top-level statement is the same except the module docstring, their helpers (and constants) and the six
        definitions they touch."""
        f = "rigstate.py"
        if (KIT_DIR / f).is_file():
            self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)
        if not (KIT_DIR / "mechkit.py").is_file():
            return
        import ast

        touched = {"Shape", "box", "cylinder", "open_box", "usd_plan", "_author"}
        added = {"_opacity", "_with_opacity", "_see_through", "SEE_THROUGH_ROUGHNESS", "FRACTIONAL_CUTOUT_SETTING",
                 "_hidden", "_with_look", "HIDE_MODES", "hide_mode", "_hide"}

        def name(n):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                return n.targets[0].id
            return getattr(n, "name", None)

        def tops(path, skip):
            out = []
            for i, n in enumerate(ast.parse(path.read_text()).body):
                if i == 0 and isinstance(n, ast.Expr):
                    continue  # the module docstring
                if name(n) in skip:
                    continue
                out.append(ast.dump(n))
            return out

        ours = ast.parse((HERE / "mechkit.py").read_text()).body
        self.assertTrue(added <= {name(n) for n in ours})
        self.assertEqual(tops(HERE / "mechkit.py", touched | added), tops(KIT_DIR / "mechkit.py", touched))

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("4 fl oz (118 mL)", "x 300 mm, y -180 mm", "49 mm across", "82 mm tall", "28 mm across",
                      "105 mm", "22 mm opening", "2 mm walls", "118 mL of syrup", "170 g", "1.31 g/mL", "150 mPa s",
                      "30 mL (1 fl oz)", "40 mm across and 38 mm tall", "36 mm inside", "8 g", "10 mL",
                      "100 x 100 mm", "14 mm high", "x 300 mm, y 0", "11.8 mm", "3 mL/s", "76 degrees",
                      "within 14 mm of its axis", "golden disc", "see-through", "round",
                      "24 flat panels drawn clear (not rendered)", "polypropylene", "5 to 30 mL",
                      "rings at 10 mL (red, 6 mm",
                      "20 and 30 mL (near black, 4 mm)", "1.5 mm ticks at 5, 15 and 25 mL",
                      "rolled lip drawn pale grey", "base pale clear", "9 to 11 mL is its middle third",
                      "printed numerals 10, 20 and 30 (7 mm)", "white platform",
                      "bolder than real print", "Simple Syrup NF", "shelf-stable", "child-resistant",
                      "not on the table", "white paper label"):
            self.assertIn(token, setup)
        for stale in ("made fresh", "within 8 h", "square", "granulated sugar", "grey,", "tint)", "amber disc",
                      "percent opaque", "untinted", "solid white floor"):
            self.assertNotIn(stale, setup)
        for token in ("15 mL", "45 degrees", "10 mL", "9.0 to 11.0 mL", "0.5 mL", "8.0 to 12.0 mL",
                      "at least 4 mL", "0.15 s", "4 such frames", "20 mm", "5 degrees", "15 mm", "3 mm", "30 mm",
                      "1 cm/s", "0.5 rad/s",
                      "Possible scores: 0, 0.25, 0.5, 0.75, 1.0", "11.8 to 14.4 g", "10.5 to 15.7 g", "0.66 g",
                      "0.1 g", "3 mL", "1 s", "1.31 g/mL"):
            self.assertIn(token, scoring)
        self.assertEqual(round(2 * L.CUP_INNER_HALF * 1000), 36)
        self.assertEqual(round((L.CUP_INNER_HALF - L.CATCH_MARGIN) * 1000), 14)
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
        for token in ("volume ledger", "3.0 mL/s", "(est)", "24 flat wall panels", "No drip tail", "150 mPa s", "1.31 g/mL",
                      "Declared rule", "golden disc", "display only", "drawn clear (not rendered)",
                      "numerals bolder than print"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("parent", "dosing cup", "3 mL/s", "cannot be poured back", "child-resistant cap",
                      "9 to 11 mL window is about two thirds of a second", "eye level"):
            self.assertIn(token, ctx)
        self.assertNotIn("third of a second", ctx.replace("two thirds of a second", ""))
        self.assertNotIn("label says", ctx)  # no claim about what labels say

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
