"""Offline check of the scripted demo against the simulator's and the station's limits (tools/offline/check_task.py
is not available here). Generic: it reads only task.py's OBJECTS and demo(), task.json's run_limit and the station
constants below, so it can be copied to another task folder unchanged.

    python3 -m unittest discover -s tests -v

Limits checked (yam_arena station, yam_scripted policy):
  * every loose object weighs 0.3 kg or less; fixed objects have no mass; names are unique;
  * every commanded finger value lies in [-0.04695, -0.008] (a commanded gap of 16 to 94 mm); every value other than
    fully open is either a grip 8 mm narrower than some loose part's width (a gap of 16 to 80 mm) or the fully
    closed value -0.008 (a closed hand that holds nothing, as when it presses or pushes);
  * every waypoint that changes the finger value lasts at least 1.5 s, does not move the arm, and follows a still
    'settle' waypoint of at least 0.5 s; a descent straight into that settle pose covers at most 3.2 cm at no more
    than 2.5 cm/s (the last 2-3 cm are lowered slowly);
  * every TCP target, and every point of the plan sampled at 20 Hz, lies inside that arm's reach, MARGIN inside every
    limit: x 0.10 to 0.50 m in front of the bases; toward the midline the reach shrinks: at |y| <= 0.05, x <= 0.40 at
    table height and x <= 0.30 at 16 cm up or higher (linear in height between), interpolated linearly in |y|
    between 0.05 and 0.305; an arm never crosses the midline nor goes outboard of its own base;
  * the two TCPs, sampled on one clock at 20 Hz from the ready poses and interpolated linearly between waypoints,
    are never within 0.20 m (plus MARGIN) of each other in y while within 0.05 m (plus MARGIN) of each other in x;
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

TOP = 0.7475  # station table top
HZ = 20.0
OPEN = -0.04695  # finger joint value fully open (94 mm)
FINGER_MIN, FINGER_MAX = -0.04695, -0.008  # commanded gap 16..94 mm
CLOSED = -0.008  # fully closed, holding nothing
GRIP_GAP_MIN, GRIP_GAP_MAX = 0.016, 0.080  # a grip on a part: 16..80 mm commanded
GRIP_UNDER = 0.008  # a grip is commanded 8 mm narrower than the part
MIN_FINGER_S = 1.5  # every open or close lasts at least this long
SETTLE_S = 0.5  # and is preceded by a still 'settle' waypoint at least this long
SLOW_DROP, SLOW_SPEED = 0.032, 0.025  # the move into a settle pose before a finger change: <= 3.2 cm, <= 2.5 cm/s
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
READY = {"left": (0.323, 0.305, TOP + 0.26), "right": (0.323, -0.305, TOP + 0.26)}  # yam_arena ready poses
BASE_Y = {"left": 0.305, "right": -0.305}
MARGIN = 0.02  # the plan keeps at least this far inside every reach and clearance limit
SEP_X, SEP_Y = 0.05, 0.20  # two TCPs within SEP_X in x need SEP_Y between them in y
MAX_LOOSE_MASS = 0.3
TABLE_X, TABLE_Y = (0.05, 0.55), 0.55
SETTLE_AFTER_S = 3.0  # the demo plus this must fit in the run limit


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def reach_x(arm: str, y: float, z: float) -> float:
    """The furthest x this arm's TCP reaches at (y, z): 0.40 at the midline at table height, 0.30 there at 16 cm up
    (linear between), rising linearly to 0.50 at the arm's own base line (|y| 0.305); -inf past the midline."""
    toward = y if arm == "left" else -y  # distance from the midline on the arm's own side
    if toward < -1e-9:
        return -np.inf
    f_z = min(max((z - TOP) / 0.16, 0.0), 1.0)
    mid = 0.40 + (0.30 - 0.40) * f_z
    f_y = min(max((toward - 0.05) / (0.305 - 0.05), 0.0), 1.0)
    return mid + (0.50 - mid) * f_y


def part_widths(objects):
    """Every width a loose part offers the fingers: a box's two horizontal sides, a cylinder's or sphere's diameter."""
    out = []
    for o in objects:
        if getattr(o, "mass", None) is None:
            continue
        if hasattr(o, "size"):
            out += [float(o.size[0]), float(o.size[1])]
        else:
            out.append(2 * float(o.radius))
    return out


def extents(o):
    """World-aligned (lo, hi) of an object (a yawed box by its turned outline)."""
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


def sample_plan(plan: dict, hz: float = HZ) -> dict:
    """Both arms on one clock: per arm, the TCP position (n x 3) at every control step from t = 0 (the ready pose) to
    the end of the longer plan, moving linearly over each waypoint's duration; an arm without a plan holds its ready
    pose."""
    total = max([sum(w.duration for w in wps) for wps in plan.values()] + [0.0])
    n = int(round(total * hz)) + 1
    ts = np.arange(n) / hz
    out = {"t": ts}
    for arm in ("left", "right"):
        kt, kp, t = [0.0], [np.array(READY[arm], float)], 0.0
        for w in plan.get(arm, []):
            t += w.duration
            kt.append(t)
            kp.append(np.array(w.pos, float))
        kp = np.array(kp)
        out[arm] = np.stack([np.interp(ts, kt, kp[:, i]) for i in range(3)], axis=1)
    return out


def run_limit_steps() -> int:
    data = json.loads((HERE / "task.json").read_text())
    m = re.match(r"\s*([\d,]+)\s+steps", data["run_limit"])
    return int(m.group(1).replace(",", ""))


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class DemoPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        for key in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(key, None)
        cls.plan = cls.task.demo()

    def test_both_arms_planned(self):
        self.assertEqual(set(self.plan), {"left", "right"})
        for side, wps in self.plan.items():
            self.assertTrue(wps)
            for w in wps:
                self.assertGreater(w.duration, 0.0, f"{side} {w.label}")
                self.assertEqual(len(w.pos), 3, f"{side} {w.label}")

    def test_masses_and_names(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for o in self.task.OBJECTS:
            m = getattr(o, "mass", None)
            self.assertTrue(m is None or 0 < m <= MAX_LOOSE_MASS, f"{o.name}: mass {m}")

    def test_objects_on_the_table(self):
        for o in self.task.OBJECTS:
            lo, hi = extents(o)
            self.assertGreaterEqual(lo[0], TABLE_X[0] - 1e-9, o.name)
            self.assertLessEqual(hi[0], TABLE_X[1] + 1e-9, o.name)
            self.assertLessEqual(max(abs(lo[1]), abs(hi[1])), TABLE_Y + 1e-9, o.name)
            self.assertGreaterEqual(lo[2], TOP - 1e-6, o.name)

    def test_fingers(self):
        widths = part_widths(self.task.OBJECTS)
        for side, wps in self.plan.items():
            prev = OPEN
            for i, w in enumerate(wps):
                where = f"{side} {i} '{w.label}'"
                self.assertGreaterEqual(w.finger, FINGER_MIN - 1e-9, where)
                self.assertLessEqual(w.finger, FINGER_MAX + 1e-9, where)
                if abs(w.finger - OPEN) > 1e-9 and abs(w.finger - CLOSED) > 1e-9:
                    gap = -2 * w.finger
                    self.assertTrue(GRIP_GAP_MIN - 1e-9 <= gap <= GRIP_GAP_MAX + 1e-9, f"{where}: gap {gap}")
                    self.assertTrue(any(abs(gap - (wd - GRIP_UNDER)) < 5e-4 for wd in widths),
                                    f"{where}: a {gap * 1000:.1f} mm grip matches no part")
                if abs(w.finger - prev) > 1e-9:
                    self.assertGreaterEqual(w.duration, MIN_FINGER_S - 1e-9, where)
                    self.assertGreater(i, 0, f"{where}: a finger change needs a settle waypoint before it")
                    s = wps[i - 1]
                    self.assertEqual(tuple(s.pos), tuple(w.pos), f"{where}: the fingers move while the arm moves")
                    self.assertAlmostEqual(s.finger, prev)
                    self.assertGreaterEqual(s.duration, SETTLE_S - 1e-9, f"{where}: no settle before it")
                    a = wps[i - 3].pos if i >= 3 else READY[side]
                    if i >= 2:
                        m = wps[i - 2]
                        drop = a[2] - m.pos[2]
                        if drop > 1e-9 and tuple(m.pos) == tuple(s.pos):
                            self.assertLessEqual(drop, SLOW_DROP + 1e-9, f"{side} {i - 2} '{m.label}'")
                            self.assertLessEqual(drop / m.duration, SLOW_SPEED + 1e-9, f"{side} {i - 2} '{m.label}'")
                    elif i == 1:  # the settle itself is the first waypoint: the arm must already be there
                        self.assertEqual(tuple(s.pos), tuple(READY[side]), f"{where}: the settle moves the arm")
                prev = w.finger

    def _check_reach(self, side, p, where):
        x, y, z = (float(v) for v in p)
        toward = y if side == "left" else -y
        self.assertGreaterEqual(toward, -1e-9, f"{side} {where}: crosses the midline")
        self.assertLessEqual(abs(y), abs(BASE_Y[side]) + 1e-9, f"{side} {where}: outboard of its base")
        lim = reach_x(side, y, z)
        self.assertGreaterEqual(x, 0.10 + MARGIN - 1e-9, f"{side} {where}: x {x:.3f}")
        self.assertLessEqual(x, lim - MARGIN + 1e-9, f"{side} {where}: x {x:.3f}, limit {lim:.3f}")

    def test_reach_of_targets_and_height(self):
        for side, wps in self.plan.items():
            for i, w in enumerate(wps):
                self._check_reach(side, w.pos, f"{i} '{w.label}'")
                self.assertGreaterEqual(w.pos[2], TOP + PAD_TIPS - 1e-9, f"{side} {i} '{w.label}': below the table")

    def test_reach_sampled(self):
        s = sample_plan(self.plan)
        for side in self.plan:
            for k, p in enumerate(s[side]):
                self._check_reach(side, p, f"t={s['t'][k]:.2f}s")

    def test_two_grippers_apart(self):
        s = sample_plan(self.plan)
        dx = np.abs(s["left"][:, 0] - s["right"][:, 0])
        dy = np.abs(s["left"][:, 1] - s["right"][:, 1])
        close = (dx < SEP_X + MARGIN) & (dy < SEP_Y + MARGIN - 1e-9)
        k = int(np.argmax(close))
        self.assertFalse(bool(close.any()), f"t={s['t'][k]:.2f}s: TCPs {s['left'][k]} {s['right'][k]}")

    def test_fits_the_run_limit(self):
        total = max(sum(w.duration for w in wps) for wps in self.plan.values())
        self.assertLess((total + SETTLE_AFTER_S) * HZ, run_limit_steps())

    def test_switches(self):
        try:
            for key, value, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                                     ("DEMO_HOLD_ONLY", "1", set())):
                os.environ[key] = value
                try:
                    self.assertEqual(set(self.task.demo()), want)
                finally:
                    os.environ.pop(key, None)
            self.assertEqual(set(self.task.demo()), {"left", "right"})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY", None)
            os.environ.pop("DEMO_IDLE_ARM", None)


if __name__ == "__main__":
    unittest.main()
