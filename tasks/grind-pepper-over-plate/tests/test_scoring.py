"""Offline tests for layout.py, scoring.py and task.py: synthetic mill states (mechkit AssemblyStates from the task's
own spec), synthetic plate states and synthetic arm snapshots (rigstate.Arm); no simulator. They need numpy and the
pure-Python yam_arena package (yam_arena.task_file, yam_arena.demo) on the path:

    PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v
"""

import json
import math
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
from yam_arena.task_file import ObjectState  # noqa: E402

WIN = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 20 steps
START = np.array([*L.MILL_START, L.TOP])
OVER = np.array([*L.PLATE_CENTRE, L.TOP + 0.035])
AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}


def holding(base) -> dict:
    """Both grippers on the mill (left on the body, right on the crown), closed."""
    b = np.asarray(base, dtype=float)
    left = b + np.array([0.0, 0.01, 0.026])
    right = b + np.array([0.0, 0.0, 0.089])
    return {"left": rigstate.Arm(left, left + np.array([0.0, -0.008, -0.009]), 0.0),
            "right": rigstate.Arm(right, right - np.array([0.0, 0.0, 0.013]), 0.0)}


def quat_tilt_x(deg: float) -> tuple:
    a = math.radians(deg) / 2
    return (math.sin(a), 0.0, 0.0, math.cos(a))


def mill(base, head_rad: float = 0.0, tilt: float = 0.0):
    return mk.state_from_poses(T.MILL_SPEC, (*np.asarray(base, dtype=float), *quat_tilt_x(tilt)), np.zeros(6),
                               joints={S.HEAD: head_rad})


def plate(dx: float = 0.0, tilt: float = 0.0) -> ObjectState:
    c = np.array([L.PLATE_CENTRE[0] + dx, L.PLATE_CENTRE[1], L.TOP + L.PLATE_H * L.MM / 2])
    h = np.array([L.PLATE_R, L.PLATE_R, L.PLATE_H / 2]) * L.MM
    return ObjectState(pos=c, quat_xyzw=np.array(quat_tilt_x(tilt)), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=c - h, high=c + h)


class Run:
    """Feeds a Tracker step by step and keeps the head's angle as the simulator would report it."""

    def __init__(self, wrap_reports: bool = False) -> None:
        self.t = S.Tracker()
        self.head = 0.0
        self.wrap = wrap_reports

    def step(self, base, arms, d_head: float = 0.0, tilt: float = 0.0, p: ObjectState | None = None):
        self.head += d_head
        reported = S.wrap(self.head) if self.wrap else self.head
        return self.t.update({S.MILL: mill(base, reported, tilt), S.PLATE: p or plate()}, arms)

    def carry(self, a, b, n: int = 20):
        r = None
        for k in range(1, n + 1):
            r = self.step(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * k / n, holding(np.asarray(a)))
        return r

    def grind(self, turns: float, base=OVER, sense: float = -1.0, per_step_deg: float = 7.5):
        steps = int(round(turns * 360.0 / per_step_deg))
        r = None
        for _ in range(steps):
            r = self.step(base, holding(base), sense * math.radians(per_step_deg))
        return r

    def stand_and_leave(self, spot=START, n: int = 40):
        self.carry(OVER, spot)
        r = None
        for _ in range(n):
            r = self.step(spot, AWAY)
        return r


class TestScoring(unittest.TestCase):
    def full_run(self, turns: float, spot=START, **kw):
        run = Run(**kw)
        for _ in range(3):
            run.step(START, AWAY)
        run.carry(START, OVER)
        run.grind(turns)
        return run, run.stand_and_leave(spot)

    def test_five_turns_then_stood_back_is_full_score(self):
        _, r = self.full_run(5.0)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["turns"], 5)

    def test_reported_angle_wrapped_to_one_turn_still_counts(self):
        _, r = self.full_run(5.0, wrap_reports=True)
        self.assertEqual(r.details["turns"], 5)
        self.assertEqual(r.score, 1.0)

    def test_more_than_five_turns_is_still_full(self):
        _, r = self.full_run(7.2)
        self.assertEqual(r.score, 1.0)

    def test_partial_turns_score_per_whole_turn(self):
        for turns, want in ((0.9, 0.0), (1.0, 0.15), (3.5, 0.45), (4.9, 0.6)):
            _, r = self.full_run(turns)
            self.assertAlmostEqual(r.score, want, msg=f"{turns} turns")

    def test_turning_the_wrong_way_grinds_nothing(self):
        run = Run()
        run.carry(START, OVER)
        r = run.grind(6.0, sense=+1.0)
        self.assertEqual(r.details["turns"], 0)
        self.assertEqual(r.score, 0.0)

    def test_back_and_forth_counts_only_the_clockwise_part(self):
        run = Run()
        run.carry(START, OVER)
        for _ in range(4):
            run.grind(1.0)
            run.grind(1.0, sense=+1.0)
        self.assertEqual(run.t.last.details["turns"], 4)

    def test_grinding_beside_the_plate_or_resting_on_it_counts_nothing(self):
        run = Run()
        r = run.grind(6.0, base=START)  # standing on its start spot
        self.assertEqual(r.details["turns"], 0)
        resting = np.array([*L.PLATE_CENTRE, L.TOP + L.PLATE_H * L.MM])  # on the plate's top: not held clear
        r = run.grind(6.0, base=resting)
        self.assertEqual(r.details["turns"], 0)
        edge = np.array([L.PLATE_CENTRE[0] + 0.10, L.PLATE_CENTRE[1], L.TOP + 0.035])  # over the rim, not the middle
        r = run.grind(6.0, base=edge)
        self.assertEqual(r.details["turns"], 0)

    def test_grinding_tilted_past_30_deg_counts_nothing(self):
        run = Run()
        for _ in range(int(6 * 360 / 7.5)):
            run.step(OVER, holding(OVER), math.radians(-7.5), tilt=40.0)
        self.assertEqual(run.t.last.details["turns"], 0)

    def test_five_turns_left_on_the_plate_is_0_75(self):
        on_plate = np.array([*L.PLATE_CENTRE, L.TOP + L.PLATE_H * L.MM])
        _, r = self.full_run(5.0, spot=on_plate)
        self.assertAlmostEqual(r.score, 0.75)

    def test_five_turns_still_held_is_0_75(self):
        run = Run()
        run.carry(START, OVER)
        run.grind(5.0)
        run.carry(OVER, START)
        r = None
        for _ in range(40):
            r = run.step(START, holding(START))
        self.assertAlmostEqual(r.score, 0.75)

    def test_unreadable_arms_never_give_full_score(self):
        run = Run()
        run.carry(START, OVER)
        run.grind(5.0)
        run.carry(OVER, START)
        r = None
        for _ in range(40):
            r = run.step(START, None)
        self.assertAlmostEqual(r.score, 0.75)

    def test_dropping_the_mill_latches_zero(self):
        run = Run()
        run.carry(START, OVER)
        run.grind(5.0)
        run.step(OVER, holding(OVER), tilt=75.0)
        r = run.stand_and_leave()
        self.assertEqual(r.score, 0.0)
        self.assertIn("dropped", r.details["failed"])

    def test_knocking_the_plate_latches_zero(self):
        run = Run()
        run.carry(START, OVER)
        run.grind(5.0)
        run.step(OVER, holding(OVER), p=plate(dx=0.04))
        r = run.stand_and_leave()
        self.assertEqual(r.score, 0.0)
        self.assertIn("plate", r.details["failed"])

    def test_mill_off_the_table_latches_zero(self):
        run = Run()
        run.step(START + np.array([0.0, 0.0, -0.05]), AWAY)
        r = run.stand_and_leave()
        self.assertEqual(r.score, 0.0)

    def test_invalid_step_breaks_the_count_and_the_windows(self):
        run = Run()
        run.carry(START, OVER)
        run.grind(2.0)
        r = run.t.update({S.PLATE: plate()}, holding(OVER))  # the mill's state missing
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        run.grind(3.0)
        self.assertEqual(run.t.last.details["turns"], 4)  # the step across the gap is not counted

    def test_teleport_back_to_start_resets_the_tally(self):
        run = Run()
        run.carry(START, OVER)
        run.grind(5.0)
        r = run.step(START, AWAY)
        self.assertEqual(run.t.resets, 1)
        self.assertEqual(r.details["turns"], 0)

    def test_tiers(self):
        self.assertEqual(S.tier(5, False, True), 1.0)
        self.assertEqual(S.tier(9, False, False), 0.75)
        self.assertEqual(S.tier(5, True, True), 0.0)
        self.assertEqual(S.tier(0, False, True), 0.0)


class TestSpec(unittest.TestCase):
    def test_mill_is_one_articulation_with_a_free_spindle(self):
        self.assertEqual([j.name for j in T.MILL_SPEC.joints], [S.HEAD])
        j = T.MILL_SPEC.joints[0]
        self.assertTrue(j.continuous)
        plan = mk.usd_plan(T.MILL_SPEC, "/World/envs/env_0/mill")
        self.assertIsNone(plan["joints"][0]["lower"])
        self.assertAlmostEqual(plan["joints"][0]["actuator"]["friction"], L.GRIND_FRICTION)

    def test_mill_size_and_mass_match_the_product(self):
        lo, hi = T.MILL_SPEC.local_bounds()
        self.assertAlmostEqual((hi[2] - lo[2]) * 1000, L.HEIGHT, places=3)
        self.assertAlmostEqual((hi[0] - lo[0]) * 1000, 59.0, delta=0.6)
        self.assertAlmostEqual(L.M_BODY + L.M_HEAD - 0.025, 0.190, places=3)  # 190 g empty + 25 g of peppercorns

    def test_objects_start_apart_and_on_the_table(self):
        gap = math.hypot(L.MILL_START[0] - L.PLATE_CENTRE[0], L.MILL_START[1] - L.PLATE_CENTRE[1])
        self.assertGreater(gap, (L.PLATE_R + 29.5) * L.MM + 0.02)
        self.assertGreaterEqual(math.hypot(*np.subtract(L.MILL_START, L.PLATE_CENTRE)), L.STAND_OFF_PLATE)


class TestDemo(unittest.TestCase):
    def test_demo_turns_the_head_at_least_five_turns_clockwise(self):
        right = T.plans()["right"]
        turned = 0.0
        for a, b in zip(right, right[1:]):
            if a.finger == T.CROWN_GRIP and b.finger == T.CROWN_GRIP:
                turned += max(0.0, float(a.yaw) - float(b.yaw))
        self.assertGreaterEqual(math.degrees(turned), 5 * 360.0)

    def test_left_holds_until_right_is_off_the_crown(self):
        p = T.plans()
        right_off = 0.0
        for w in p["right"]:
            right_off += w.duration
            if w.label == "up off the crown":
                break
        t, hold_end = 0.0, None
        for w in p["left"]:
            t += w.duration
            if w.label == "hold the body while the head is turned":
                hold_end = t
        self.assertGreater(hold_end, right_off)

    def test_variants_exist(self):
        for v in T.VARIANTS:
            self.assertIn("left", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
            self.assertTrue(meta.get(k), k)
        self.assertEqual(meta["instruction"], T.INSTRUCTION)
        self.assertIn(meta["category"], ("Impact on jobs", "Impact on AI timelines", "Safety and alignment",
                                         "Societal relatability", "Other"))


if __name__ == "__main__":
    unittest.main()
