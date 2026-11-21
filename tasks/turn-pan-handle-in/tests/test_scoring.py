"""Offline tests for layout.py, scoring.py and task.py: synthetic pan states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm); no simulator. They need numpy and the pure-Python yam_arena
package (yam_arena.task_file, yam_arena.demo) on the path: the real one, or the offline stand-in found next to this
folder (tools/offline/yam_stub), which is put on the path when present:

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


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    try:
        import yam_arena  # noqa: F401  (the real package wins when it is installed)
    except ModuleNotFoundError:
        for _p in (OFFLINE / "yam_stub", OFFLINE):
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

START = np.array(L.pan_start())
Y0 = math.degrees(L.PAN_START_YAW)  # 150: the handle toward the cook's side, past the front edge
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
    def test_start_handle_sticks_out_past_the_front_edge(self):
        pts = L.handle_world_xy(START[:2], L.PAN_START_YAW)
        inside, clear, problem = L.safe_plan(pts)
        self.assertFalse(inside)
        self.assertIn("edge", problem)
        x0, x1, y0, y1 = L.cook_bounds()
        self.assertGreater(x0 - pts[:, 0].min(), 0.14)  # 149 mm past the front edge (the cook's side)
        self.assertLess(pts[:, 1].max(), y1)  # and only the front edge: inside the left edge, its full width
        self.assertGreater(pts[:, 1].min(), y0)
        self.assertLess(pts[:, 0].max(), x1)

    def test_start_handle_clears_the_left_arm_base(self):
        if OFFLINE is None:
            self.skipTest("tools/offline not found")
        import yamkin as K
        base = np.array([0.0, 0.305, 0.75])
        caps = [c for c in K.arm().capsules(K.READY, base) if c[0] in ("base", "link1")]
        pts = L.handle_world_xy(START[:2], L.PAN_START_YAW)[:L.HANDLE_SAMPLES]
        z = START[2] + L.handle_points()[:, 2]
        gap = min(float(np.linalg.norm(P - (a + np.clip((P - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1) * (b - a))))
                  - r for P in (np.array([x, y, zz]) for (x, y), zz in zip(pts, z)) for _, a, b, r in caps)
        self.assertGreater(gap - L.HANDLE_W / 2, 0.06)  # about 73 mm: the handle never touches the base

    def test_the_cooktop_clears_the_arm_bases(self):
        x0, _, y0, y1 = L.cook_bounds()
        self.assertGreaterEqual(x0, 0.10)  # the bases reach about 46 mm forward of their line

    def test_the_obvious_directions_are_unsafe(self):
        for d, why in ((0.0, "left rear"), (-20.0, "left rear"), (-90.0, "right front"), (180.0, "edge"),
                       (150.0, "edge"), (135.0, "edge"), (45.0, "edge"), (-113.0, "back")):
            inside, clear, problem = L.safe_plan(L.handle_world_xy(START[:2], math.radians(d)))
            self.assertFalse(inside and clear, d)
            self.assertIn(why, problem, d)

    def test_safe_sector(self):
        ys = L.safe_yaws(1.0)
        self.assertEqual(min(ys), -68.0)  # one sector only: nothing else anywhere round the pan
        self.assertEqual(max(ys), -41.0)
        self.assertEqual(len(ys), 28)

    def test_a_handle_still_pointing_forward_is_not_turned_in(self):
        # at -113 deg (toward the front right) the handle ends 12 mm inside the front edge and clear of the hot
        # rings, but it still points at the cook's side: not turned in
        pts = L.handle_world_xy(START[:2], math.radians(-113.0))
        x0, x1, y0, y1 = L.cook_bounds()
        self.assertTrue(np.all((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)))
        inside, clear, problem = L.safe_plan(pts)
        self.assertTrue(clear)
        self.assertFalse(inside)
        self.assertIn("back", problem)


class TestHandleWidth(unittest.TestCase):
    def test_a_handle_edge_past_the_cooktop_is_unsafe(self):
        # at -69 deg the handle's centre line clears the rules, but one of its side edges does not
        centre_only = L.handle_points()
        x, y, _ = L.pan_start()
        c, s_ = math.cos(math.radians(-69.0)), math.sin(math.radians(-69.0))
        xy = np.stack([x + c * centre_only[:, 0] - s_ * centre_only[:, 1],
                       y + s_ * centre_only[:, 0] + c * centre_only[:, 1]], axis=1)
        self.assertTrue(all(L.safe_plan(xy)[:2]))  # the centre line alone would pass
        full = L.handle_world_xy((x, y), math.radians(-69.0))
        self.assertFalse(all(L.safe_plan(full)[:2]))  # the full width does not


class TestScoring(unittest.TestCase):
    def test_turned_into_the_safe_sector_and_let_go_is_full(self):
        run = Run()
        run.leave(Y0, n=3)
        run.swing(Y0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_turned_back_over_the_hot_element_is_zero(self):
        run = Run()
        run.swing(Y0, 0.0)
        r = run.leave(0.0)
        self.assertEqual(r.score, 0.0)
        self.assertIn("hot", r.details["problem"])

    def test_turned_to_the_middle_over_the_lit_element_is_zero(self):
        run = Run()
        run.swing(Y0, -95.0)
        r = run.leave(-95.0)
        self.assertEqual(r.score, 0.0)

    def test_untouched_is_zero(self):
        run = Run()
        r = run.leave(Y0)
        self.assertEqual(r.score, 0.0)

    def test_still_holding_is_half(self):
        run = Run()
        run.swing(Y0, -54.0)
        r = None
        for _ in range(40):
            r = run.step(-54.0, holding(-54.0))
        self.assertEqual(r.score, 0.5)

    def test_off_centre_between_30_and_60_mm_is_half(self):
        run = Run()
        off = (0.03, -0.025, 0.0)  # 39 mm
        run.swing(Y0, -54.0, offset=off)
        r = run.leave(-54.0, offset=off)
        self.assertEqual(r.score, 0.5)

    def test_slid_off_the_burner_latches_zero(self):
        run = Run()
        run.swing(Y0, -54.0, offset=(0.07, 0.0, 0.0))
        r = run.leave(-54.0)
        self.assertEqual(r.score, 0.0)
        self.assertIn("slid", r.details["failed"])

    def test_tipped_latches_zero(self):
        run = Run()
        run.swing(Y0, 0.0)
        run.step(0.0, holding(0.0), tilt=12.0)
        run.swing(0.0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 0.0)
        self.assertIn("spilled", r.details["failed"])

    def test_lifted_latches_zero(self):
        run = Run()
        run.step(Y0, holding(Y0), offset=(0.0, 0.0, 0.03))
        run.swing(Y0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 0.0)

    def test_small_lift_and_tilt_in_passing_are_allowed(self):
        run = Run()
        run.step(Y0, holding(Y0), offset=(0.0, 0.0, 0.015), tilt=6.0)
        run.swing(Y0, -54.0)
        r = run.leave(-54.0)
        self.assertEqual(r.score, 1.0)

    def test_unreadable_arms_never_full(self):
        run = Run()
        run.swing(Y0, -54.0)
        r = None
        for _ in range(40):
            r = run.step(-54.0, None)
        self.assertEqual(r.score, 0.5)

    def test_missing_state_scores_zero_and_restarts_windows(self):
        run = Run()
        run.swing(Y0, -54.0)
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
        run.step(Y0, holding(Y0), tilt=12.0)  # tilted past the limit: failed
        run.swing(Y0, -54.0)
        run.step(Y0, AWAY)  # straight back to the starting pose in one step
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

    def test_demo_replayed_rigidly_scores_full(self):
        # The demo's own waypoints at 20 Hz, the pan following the closed gripper rigidly (no lag, no slip): the
        # handle ends at the aim (-60, inside the sector), the slide leaves the pan 25 mm off centre (< 30 mm), and
        # once let go the scorer gives 1.0 well inside the 800-step run limit.
        left = T.plans()["left"]
        t_end = sum(w.duration for w in left) + 3.0
        hz = L.CONTROL_HZ
        grip_local = L.grip_point()[:2]
        pos = np.array(T.grip_tcp(L.PAN_START_YAW)) + np.array([0.0, 0.0, 0.2])
        yaw, finger = L.PAN_START_YAW, T.OPEN
        segs, t0 = [], 0.0
        for w in left:
            segs.append((t0, t0 + w.duration, pos, np.array(w.pos), yaw, w.yaw, finger, w.finger))
            t0, pos, yaw, finger = t0 + w.duration, np.array(w.pos), w.yaw, w.finger
        run = Run()
        pan_xy, pan_yaw, held = START[:2].copy(), Y0, None
        first = None
        for k in range(int(t_end * hz)):
            t = k / hz
            for a, b, p0, p1, y0, y1, f0, f1 in segs:
                if t < b:
                    s_ = (t - a) / (b - a)
                    s_ = s_ * s_ * (3 - 2 * s_)
                    p, yv, f = p0 + s_ * (p1 - p0), y0 + s_ * (y1 - y0), f0 + s_ * (f1 - f0)
                    break
            else:
                p, yv, f = p1, y1, f1
            closed = abs(f - T.CLOSED) < 1e-6
            if closed and held is None:
                held = True
            if held and not closed:
                held = False
            if held:
                c, s2 = math.cos(yv), math.sin(yv)
                pan_yaw = math.degrees(yv)
                pan_xy = p[:2] - np.array([c * grip_local[0] - s2 * grip_local[1], s2 * grip_local[0] + c * grip_local[1]])
            opening = float(np.clip(f / T.OPEN, 0.0, 1.0))
            arms = {"left": rigstate.Arm(p, p - np.array([0.0, 0.0, 0.013]), opening), "right": AWAY["right"]}
            r = run.t.update({S.PAN: pan(pan_yaw, (pan_xy[0] - START[0], pan_xy[1] - START[1], 0.0))}, arms)
            if r.success and first is None:
                first = run.t.steps
        self.assertEqual(r.score, 1.0, r.details)
        self.assertIsNotNone(first)
        self.assertLess(first, 600)
        self.assertLess(r.details["centre_mm"], 30.0)
        self.assertIsNone(r.details["failed"])

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
        self.assertIn("41 and 68 degrees", meta["scoring"])


if __name__ == "__main__":
    unittest.main()
