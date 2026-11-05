"""Offline checks of task.py's scripted demo against the station's limits (tools/offline/check_task.py is not
available here). Generic: it reads only task.py's OBJECTS, demo() and the run limit in task.json.

    python3 -m unittest discover -s tests -v

Limits checked (yam_arena station, yam_scripted policy):
  * every loose object weighs 0.3 kg or less; fixed objects have no mass;
  * every commanded finger value lies in [-0.04695, -0.008] (a commanded gap of 16 to 94 mm), and every value other
    than fully open is a grip 8 mm narrower than some loose object's width (a gap of 16 to 80 mm);
  * every waypoint that changes the finger value lasts at least 1.5 s;
  * every TCP target, and every point of the plan sampled at 20 Hz, lies inside that arm's reach, MARGIN inside
    every limit: x 0.10 to 0.50 m in front of the bases; toward the midline the reach shrinks: at |y| <= 0.05,
    x <= 0.40 at table height and x <= 0.30 at 16 cm up or higher (linear in height between), interpolated linearly
    in |y| between 0.05 and 0.305; an arm never crosses the midline;
  * the two TCPs, sampled on one clock at 20 Hz from the ready poses and interpolated linearly between waypoints,
    are never within 0.20 m (plus MARGIN) of each other in y while within 0.05 m of each other in x;
  * no TCP target lower than the table top + 13 mm (the pad tips at the table);
  * every object lies on the table (|y| <= 0.55, 0.05 <= x <= 0.55);
  * the demo's length plus 3 s of settling is under the run limit;
  * DEMO_HOLD_ONLY=1 gives an empty plan and DEMO_IDLE_ARM=left|right drops that arm.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

TOP = 0.7475
HZ = 20.0
OPEN = -0.04695
FINGER_MIN, FINGER_MAX = -0.04695, -0.008
MAX_MASS = 0.3
MIN_CHANGE_S = 1.5
MARGIN = 0.02  # the demo keeps at least 2 cm inside every reach limit and the TCP separation
READY = {"left": (0.323, 0.305, TOP + 0.26), "right": (0.323, -0.305, TOP + 0.26)}  # yam_arena ready poses
BASE_Y = {"left": 0.305, "right": -0.305}


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def load_task():
    stub = _stub_dir()
    if stub is not None and str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    return importlib.import_module("task")


def run_limit_s() -> float:
    text = json.loads((HERE / "task.json").read_text())["run_limit"]
    steps = int(re.match(r"\s*([\d,]+)\s*steps", text).group(1).replace(",", ""))
    return steps / HZ


def reach_x_max(arm: str, y: float, z: float) -> float:
    """The furthest forward this arm reaches at (y, z); -inf past the midline."""
    d = y if arm == "left" else -y  # distance from the midline toward the arm's own side
    if d < 0:
        return -np.inf
    t = float(np.clip((z - TOP) / 0.16, 0.0, 1.0))
    mid = 0.40 + (0.30 - 0.40) * t  # the limit at |y| <= 0.05
    if d <= 0.05:
        return mid
    if d >= 0.305:
        return 0.50
    return mid + (0.50 - mid) * (d - 0.05) / (0.305 - 0.05)


def sample_plan(plan: dict, hz: float = HZ, hold_s: float = 0.0) -> dict:
    """Both arms on one clock: per arm, arrays of TCP positions (n x 3), finger values (n) and yaw (n) at every
    control step from t = 0 (the ready pose, fingers open) to the end of the longer plan plus ``hold_s``. Positions
    and finger values move linearly over each waypoint's duration; an arm without a plan holds its ready pose."""
    total = max([sum(w.duration for w in wps) for wps in plan.values()] + [0.0]) + hold_s
    n = int(round(total * hz)) + 1
    out = {}
    for arm in ("left", "right"):
        knots_t, knots_p, knots_f, knots_y = [0.0], [np.array(READY[arm], float)], [OPEN], [0.0]
        t = 0.0
        for w in plan.get(arm, []):
            t += w.duration
            knots_t.append(t)
            knots_p.append(np.array(w.pos, float))
            knots_f.append(float(w.finger))
            knots_y.append(float(w.yaw))
        ts = np.arange(n) / hz
        kt = np.array(knots_t)
        kp = np.array(knots_p)
        pos = np.stack([np.interp(ts, kt, kp[:, i]) for i in range(3)], axis=1)
        fin = np.interp(ts, kt, np.array(knots_f))
        yaw = np.interp(ts, kt, np.array(knots_y))
        out[arm] = {"pos": pos, "finger": fin, "yaw": yaw}
    out["t"] = np.arange(n) / hz
    return out


def object_widths(objects) -> list[float]:
    """Graspable widths of the loose objects (box sides along x and y, cylinder and sphere diameters)."""
    widths = []
    for o in objects:
        if getattr(o, "mass", None) is None:
            continue
        if hasattr(o, "size"):
            widths += [float(o.size[0]), float(o.size[1])]
        elif hasattr(o, "radius"):
            widths.append(2 * float(o.radius))
    return widths


def extents(o) -> tuple[np.ndarray, np.ndarray]:
    p = np.array(o.pos, float)
    if hasattr(o, "size"):
        h = np.array(o.size, float) / 2
        yaw = float(getattr(o, "yaw", 0.0))
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        h = np.array([c * h[0] + s * h[1], s * h[0] + c * h[1], h[2]])
    elif hasattr(o, "height"):
        h = np.array([o.radius, o.radius, o.height / 2], float)
    else:
        h = np.array([o.radius] * 3, float)
    return p - h, p + h


class DemoPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        for key in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(key, None)
        cls.plan = cls.task.demo()

    def test_both_arms_planned(self):
        self.assertEqual(set(self.plan), {"left", "right"})
        for wps in self.plan.values():
            self.assertTrue(wps)
            for w in wps:
                self.assertGreater(w.duration, 0.0, w.label)
                self.assertEqual(len(w.pos), 3, w.label)

    def test_masses(self):
        for o in self.task.OBJECTS:
            m = getattr(o, "mass", None)
            if m is not None:
                self.assertGreater(m, 0.0, o.name)
                self.assertLessEqual(m, MAX_MASS, o.name)

    def test_object_names_unique(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))

    def test_finger_values_and_grips(self):
        widths = object_widths(self.task.OBJECTS)
        for arm, wps in self.plan.items():
            for w in wps:
                self.assertGreaterEqual(w.finger, FINGER_MIN - 1e-9, (arm, w.label))
                self.assertLessEqual(w.finger, FINGER_MAX + 1e-9, (arm, w.label))
                if abs(w.finger - OPEN) > 1e-9:
                    gap = -2 * w.finger
                    self.assertGreaterEqual(gap, 0.016 - 1e-9, (arm, w.label))
                    self.assertLessEqual(gap, 0.080 + 1e-9, (arm, w.label))
                    self.assertTrue(any(abs(gap - (wd - 0.008)) < 5e-4 for wd in widths),
                                    f"{arm} '{w.label}': a {gap * 1000:.1f} mm grip is not 8 mm under any part")

    def test_finger_changes_are_slow(self):
        for arm, wps in self.plan.items():
            prev = OPEN
            for w in wps:
                if abs(w.finger - prev) > 1e-9:
                    self.assertGreaterEqual(w.duration, MIN_CHANGE_S - 1e-9, (arm, w.label))
                prev = w.finger

    def _check_reach(self, arm, p, what):
        x, y, z = (float(v) for v in p)
        self.assertGreaterEqual(x, 0.10 + MARGIN - 1e-9, f"{arm} {what}: x {x:.3f} too close to the base")
        lim = reach_x_max(arm, y, z)
        self.assertLessEqual(x, lim - MARGIN + 1e-9, f"{arm} {what}: x {x:.3f} beyond reach {lim:.3f} - margin")

    def test_reach_of_targets(self):
        for arm, wps in self.plan.items():
            for w in wps:
                self._check_reach(arm, w.pos, w.label)

    def test_reach_sampled(self):
        s = sample_plan(self.plan)
        for arm in ("left", "right"):
            if arm not in self.plan:
                continue
            for i, p in enumerate(s[arm]["pos"]):
                self._check_reach(arm, p, f"t={s['t'][i]:.2f}s")

    def test_tcp_separation(self):
        s = sample_plan(self.plan)
        dx = np.abs(s["left"]["pos"][:, 0] - s["right"]["pos"][:, 0])
        dy = np.abs(s["left"]["pos"][:, 1] - s["right"]["pos"][:, 1])
        close = (dx < 0.05) & (dy < 0.20 + MARGIN)
        self.assertFalse(bool(close.any()), f"TCPs too close at t = {s['t'][np.argmax(close)]:.2f} s")

    def test_no_target_below_pad_tips_at_table(self):
        for arm, wps in self.plan.items():
            for w in wps:
                self.assertGreaterEqual(w.pos[2], TOP + 0.013 - 1e-9, (arm, w.label))

    def test_objects_on_table(self):
        for o in self.task.OBJECTS:
            lo, hi = extents(o)
            self.assertGreaterEqual(lo[0], 0.05 - 1e-9, o.name)
            self.assertLessEqual(hi[0], 0.55 + 1e-9, o.name)
            self.assertLessEqual(max(abs(lo[1]), abs(hi[1])), 0.55 + 1e-9, o.name)
            self.assertGreaterEqual(lo[2], TOP - 1e-6, o.name)

    def test_fits_run_limit(self):
        total = max(sum(w.duration for w in wps) for wps in self.plan.values())
        self.assertLess(total + 3.0, run_limit_s())

    def test_switches(self):
        try:
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(self.task.demo(), {})
            os.environ.pop("DEMO_HOLD_ONLY")
            for idle, keep in (("left", {"right"}), ("right", {"left"})):
                os.environ["DEMO_IDLE_ARM"] = idle
                self.assertEqual(set(self.task.demo()), keep)
        finally:
            os.environ.pop("DEMO_HOLD_ONLY", None)
            os.environ.pop("DEMO_IDLE_ARM", None)


if __name__ == "__main__":
    unittest.main()
