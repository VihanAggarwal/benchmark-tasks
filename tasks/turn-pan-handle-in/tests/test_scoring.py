"""Offline tests for layout.py, scoring.py and task.py: synthetic pan states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm); no simulator. They need numpy and the pure-Python yam_arena
package (yam_arena.task_file, yam_arena.demo) on the path:

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

START = np.array(L.pan_start())
AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}


def holding(yaw_deg: float) -> dict:
    tcp = T.grip_tcp(math.radians(yaw_deg))
    return {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.0), "right": AWAY["right"]}


def quat(yaw_deg: float, tilt_deg: float = 0.0) -> tuple:
    """Yaw about z, then tilt about the pan's own x (the handle's axis rolls)."""
    y, t = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(y), math.cos(y)])
    qx = np.array([math.sin(t), 0.0, 0.0, math.cos(t)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def pan(yaw_deg: float, offset=(0.0, 0.0, 0.0), tilt: float = 0.0):
    return mk.state_from_poses(T.PAN_SPEC, (*(START + np.asarray(offset)), *quat(yaw_deg, tilt)), np.zeros(6))


class Run:
    def __init__(self) -> None:
        self.t = S.Tracker()

    def step(self, yaw, arms, offset=(0.0, 0.0, 0.0), tilt=0.0):
        return self.t.update({S.PAN: pan(yaw, offset, tilt)}, arms)

    def swing(self, a, b, n=30, offset=(0.0, 0.0, 0.0)):
        r = None
        for k in range(1, n + 1):
            y = a + (b - a) * k / n
            r = self.step(y, holding(y), offset)
        return r

    def leave(self, yaw, n=40, offset=(0.0, 0.0, 0.0)):
        r = None
        for _ in range(n):
            r = self.step(yaw, AWAY, offset)
        return r


class TestGeometry(unittest.TestCase):
    def test_start_handle_overhangs_the_left_edge(self):
        inside, clear, problem = L.safe_plan(L.handle_world_xy(START[:2], L.PAN_START_YAW))
        self.assertFalse(inside)
        self.assertIn("edge", problem)

    def test_the_obvious_directions_are_unsafe(self):
        for d, why in ((0.0, "left rear"), (-20.0, "left rear"), (-90.0, "right front"), (180.0, "edge"),
                       (135.0, "edge"), (45.0, "edge")):
            inside, clear, problem = L.safe_plan(L.handle_world_xy(START[:2], math.radians(d)))
            self.assertFalse(inside and clear, d)
            self.assertIn(why, problem, d)

    def test_safe_sector(self):
        ys = L.safe_yaws(1.0)
        main = [d for d in ys if -80.0 <= d <= -30.0]
        self.assertEqual(min(main), -70.0)
        self.assertEqual(max(main), -37.0)
        self.assertGreaterEqual(len(main), 30)


class TestScoring(unittest.TestCase):
    def test_turned_into_the_safe_sector_and_let_go_is_full(self):
        run = Run()
        run.leave(90.0, n=3)
        run.swing(90.0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_turned_back_over_the_hot_element_is_zero(self):
        run = Run()
        run.swing(90.0, 0.0)
        r = run.leave(0.0)
        self.assertEqual(r.score, 0.0)
        self.assertIn("hot", r.details["problem"])

    def test_turned_to_the_middle_over_the_lit_element_is_zero(self):
        run = Run()
        run.swing(90.0, -95.0)
        r = run.leave(-95.0)
        self.assertEqual(r.score, 0.0)

    def test_untouched_is_zero(self):
        run = Run()
        r = run.leave(90.0)
        self.assertEqual(r.score, 0.0)

    def test_still_holding_is_half(self):
        run = Run()
        run.swing(90.0, -54.0)
        r = None
        for _ in range(40):
            r = run.step(-54.0, holding(-54.0))
        self.assertEqual(r.score, 0.5)

    def test_off_centre_between_30_and_60_mm_is_half(self):
        run = Run()
        off = (0.03, -0.025, 0.0)  # 39 mm
        run.swing(90.0, -54.0, offset=off)
        r = run.leave(-54.0, offset=off)
        self.assertEqual(r.score, 0.5)

    def test_slid_off_the_burner_latches_zero(self):
        run = Run()
        run.swing(90.0, -54.0, offset=(0.07, 0.0, 0.0))
        r = run.leave(-54.0)
        self.assertEqual(r.score, 0.0)
        self.assertIn("slid", r.details["failed"])

    def test_tipped_latches_zero(self):
        run = Run()
        run.swing(90.0, 0.0)
        run.step(0.0, holding(0.0), tilt=12.0)
        run.swing(0.0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 0.0)
        self.assertIn("spilled", r.details["failed"])

    def test_lifted_latches_zero(self):
        run = Run()
        run.step(90.0, holding(90.0), offset=(0.0, 0.0, 0.03))
        run.swing(90.0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 0.0)

    def test_small_lift_and_tilt_in_passing_are_allowed(self):
        run = Run()
        run.step(90.0, holding(90.0), offset=(0.0, 0.0, 0.015), tilt=6.0)
        run.swing(90.0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 1.0)

    def test_unreadable_arms_never_full(self):
        run = Run()
        run.swing(90.0, -54.0)
        r = None
        for _ in range(40):
            r = run.step(-54.0, None)
        self.assertEqual(r.score, 0.5)

    def test_missing_state_scores_zero_and_restarts_windows(self):
        run = Run()
        run.swing(90.0, -54.0)
        run.leave(-54.0, n=15)
        r = run.t.update({}, AWAY)
        self.assertEqual(r.score, 0.0)
        r = run.leave(-54.0, n=15)
        self.assertEqual(r.score, 0.5)  # the 20-step window started again
        r = run.leave(-54.0, n=10)
        self.assertEqual(r.score, 1.0)

    def test_a_failure_survives_a_jump_back_to_the_start(self):
        # Only a real episode reset (rigstate.on_reset) clears the latch: the pan put back where it started, however
        # fast, then turned in properly, still scores 0.
        run = Run()
        run.step(90.0, holding(90.0), tilt=12.0)  # tilted past the limit: failed
        run.swing(90.0, -54.0)
        run.step(90.0, AWAY)  # straight back to the starting pose in one step
        r = run.leave(-54.0, n=40)
        self.assertIsNotNone(r.details["failed"])
        self.assertEqual(r.score, 0.0)
        run.t.reset()  # what the episode reset calls
        self.assertIsNone(run.t.last.details.get("failed"))


class TestDemo(unittest.TestCase):
    def test_demo_aims_inside_the_safe_sector_and_lets_go(self):
        left = T.plans()["left"]
        closed = [w for w in left if w.finger == T.CLOSED]
        aim = math.degrees(float(closed[-1].yaw))
        self.assertAlmostEqual(aim, T.TARGET_YAW_DEG - T.OVERSHOOT_DEG, places=6)
        safe = L.safe_yaws(1.0)
        self.assertIn(T.TARGET_YAW_DEG, safe)
        self.assertIn(round(aim), safe)  # even with no lag at all the handle ends safe
        self.assertEqual(left[-1].finger, T.OPEN)

    def test_demo_turn_follows_the_circle_about_the_pan(self):
        r0 = math.hypot(*(L.grip_point()[:2]))
        for w in T.plans()["left"]:
            if w.finger == T.CLOSED and "slide the pan" not in w.label and w.label != "settle":
                d = math.hypot(w.pos[0] - START[0], w.pos[1] - START[1])
                self.assertAlmostEqual(d, r0, places=6)

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertIn("left", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("x")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"):
            self.assertTrue(meta.get(k), k)
        self.assertEqual(meta["instruction"], T.INSTRUCTION)
        self.assertIn("37 and 70 degrees", meta["scoring"])


if __name__ == "__main__":
    unittest.main()
