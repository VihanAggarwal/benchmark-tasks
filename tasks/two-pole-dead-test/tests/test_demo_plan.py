"""Offline check of the scripted demo against the simulator's and the station's limits (tools/offline/check_task.py
is not available here). Generic: it reads only task.py's OBJECTS and demo(), task.json's run_limit and the station
constants below, so it can be copied to another task folder unchanged.

    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import math
import os
import pathlib
import re
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

TOP = 0.7475  # station table top
HZ = 20.0
OPEN = -0.04695  # finger joint value fully open (94 mm)
FINGER_MIN, FINGER_MAX = -0.04695, -0.008  # commanded gap 16..94 mm
GRIP_GAP_MIN, GRIP_GAP_MAX = 0.016, 0.080  # a grip on a part: 16..80 mm commanded
GRIP_UNDER = 0.008  # a grip is commanded 8 mm narrower than the part
MIN_FINGER_S = 1.5  # every open or close lasts at least this long
SETTLE_S = 0.5  # and is preceded by a still 'settle' waypoint at least this long
SLOW_DROP, SLOW_SPEED = 0.032, 0.025  # the move into a grip or release pose: at most 3.2 cm down, at most 2.5 cm/s
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
READY = {"left": (0.323, 0.305, TOP + 0.26), "right": (0.323, -0.305, TOP + 0.26)}
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


def reach_x(toward_mid_y: float, dz: float) -> float:
    """The furthest x a TCP reaches at a lateral distance ``toward_mid_y`` from the midline on its own side and a
    height ``dz`` above the table: 0.40 at the midline at table height, 0.30 there at 16 cm up (linear between),
    rising linearly to 0.50 at the arm's own base line (|y| 0.305)."""
    f_z = min(max(dz / 0.16, 0.0), 1.0)
    mid = 0.40 + (0.30 - 0.40) * f_z
    f_y = min(max((toward_mid_y - 0.05) / (0.305 - 0.05), 0.0), 1.0)
    return mid + (0.50 - mid) * f_y


def part_widths(objects):
    """Every width a loose part offers the fingers: a box's two horizontal sides, a cylinder's diameter."""
    out = []
    for o in objects:
        if getattr(o, "mass", None) is None:
            continue
        if hasattr(o, "size"):
            out += [float(o.size[0]), float(o.size[1])]
        else:  # a cylinder (upright) or a sphere: its diameter
            out.append(2 * float(o.radius))
    return out


def sample(plan_side, start, hz=HZ):
    """(time, (x, y, z), finger) at every control step, moving linearly from each waypoint's start to its target."""
    pos, fin = tuple(start), OPEN
    out, t = [], 0.0
    for w in plan_side:
        n = max(1, int(round(w.duration * hz)))
        for i in range(1, n + 1):
            f = i / n
            p = tuple(a + (b - a) * f for a, b in zip(pos, w.pos))
            out.append((t + w.duration * f, p, fin + (w.finger - fin) * f))
        pos, fin, t = tuple(w.pos), w.finger, t + w.duration
    return out


def run_limit_steps():
    data = json.loads((HERE / "task.json").read_text())
    m = re.match(r"\s*([\d,]+)\s+steps", data["run_limit"])
    return int(m.group(1).replace(",", ""))


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class DemoPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.plan = cls.task.demo()

    def test_masses(self):
        for o in self.task.OBJECTS:
            m = getattr(o, "mass", None)
            self.assertTrue(m is None or 0 < m <= MAX_LOOSE_MASS, f"{o.name}: mass {m}")

    def test_objects_on_the_table(self):
        for o in self.task.OBJECTS:
            x, y, _ = o.pos
            if hasattr(o, "size"):
                hx, hy = o.size[0] / 2, o.size[1] / 2
                if getattr(o, "yaw", 0.0):
                    r = math.hypot(hx, hy)
                    hx = hy = r
            else:
                hx = hy = float(o.radius)
            self.assertGreaterEqual(x - hx, TABLE_X[0] - 1e-9, o.name)
            self.assertLessEqual(x + hx, TABLE_X[1] + 1e-9, o.name)
            self.assertLessEqual(abs(y) + hy, TABLE_Y + 1e-9, o.name)

    def test_fingers(self):
        widths = part_widths(self.task.OBJECTS)
        for side, wps in self.plan.items():
            prev = OPEN
            for i, w in enumerate(wps):
                self.assertGreaterEqual(w.finger, FINGER_MIN - 1e-9, f"{side} {i} {w.label}")
                self.assertLessEqual(w.finger, FINGER_MAX + 1e-9, f"{side} {i} {w.label}")
                if abs(w.finger - OPEN) > 1e-9:
                    gap = -2 * w.finger
                    self.assertTrue(GRIP_GAP_MIN - 1e-9 <= gap <= GRIP_GAP_MAX + 1e-9, f"{side} {i}: gap {gap}")
                    self.assertTrue(any(abs(gap - (wd - GRIP_UNDER)) < 5e-4 for wd in widths),
                                    f"{side} {i} {w.label}: a {gap * 1000:.1f} mm grip matches no part")
                if abs(w.finger - prev) > 1e-9:
                    self.assertGreaterEqual(w.duration, MIN_FINGER_S - 1e-9, f"{side} {i} {w.label}")
                    self.assertGreater(i, 0)
                    s = wps[i - 1]
                    self.assertEqual(tuple(s.pos), tuple(w.pos), f"{side} {i}: the fingers move while the arm moves")
                    self.assertAlmostEqual(s.finger, prev)
                    self.assertGreaterEqual(s.duration, SETTLE_S - 1e-9, f"{side} {i}: no settle before {w.label}")
                    if i >= 2:  # the move into the settle pose: slow over the last 2-3 cm
                        a = wps[i - 3].pos if i >= 3 else READY[side]
                        m = wps[i - 2]
                        drop = a[2] - m.pos[2]
                        if drop > 1e-9 and tuple(m.pos) == tuple(s.pos):
                            self.assertLessEqual(drop, SLOW_DROP + 1e-9, f"{side} {i - 2} {m.label}")
                            self.assertLessEqual(drop / m.duration, SLOW_SPEED + 1e-9, f"{side} {i - 2} {m.label}")
                prev = w.finger

    def test_reach_and_height(self):
        for side, wps in self.plan.items():
            for i, w in enumerate(wps):
                x, y, z = w.pos
                toward = y if side == "left" else -y  # distance from the midline on the arm's own side
                self.assertGreaterEqual(toward, -1e-9, f"{side} {i} {w.label}: crosses the midline")
                self.assertLessEqual(abs(y), abs(BASE_Y[side]) + 1e-9, f"{side} {i} {w.label}: outboard of its base")
                lim = reach_x(toward, z - TOP)
                self.assertGreaterEqual(x, 0.10 + MARGIN - 1e-9, f"{side} {i} {w.label}: x {x}")
                self.assertLessEqual(x, lim - MARGIN + 1e-9, f"{side} {i} {w.label}: x {x:.3f}, limit {lim:.3f}")
                self.assertGreaterEqual(z, TOP + PAD_TIPS - 1e-9, f"{side} {i} {w.label}: pads below the table")

    def test_two_grippers_apart(self):
        if len(self.plan) < 2:
            self.skipTest("one arm")
        a, b = (sample(self.plan[s], READY[s]) for s in ("left", "right"))
        end = max(a[-1][0], b[-1][0])
        n = int(round(end * HZ))

        def at(seq, k, side):
            return seq[min(k, len(seq) - 1)][1] if seq else READY[side]

        for k in range(n):
            pl, pr = at(a, k, "left"), at(b, k, "right")
            if abs(pl[0] - pr[0]) < SEP_X + MARGIN:
                gap = abs(pl[1] - pr[1])
                self.assertGreaterEqual(gap, SEP_Y + MARGIN - 1e-9, f"step {k}: TCPs {pl} {pr}")

    def test_fits_the_run_limit(self):
        total = max(sum(w.duration for w in wps) for wps in self.plan.values())
        self.assertLess((total + SETTLE_AFTER_S) * HZ, run_limit_steps())

    def test_switches(self):
        for key, value, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                                 ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = value
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)
        self.assertEqual(set(self.task.demo()), {"left", "right"})


if __name__ == "__main__":
    unittest.main()
