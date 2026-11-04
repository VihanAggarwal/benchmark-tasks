"""Offline checks of the scripted demo and the objects against the simulator's and the station's limits, no
simulator (the program's tools/offline/check_task.py is not available here). The task file is imported through the
offline yam_arena stand-in (tools/offline/yam_stub, found by walking up from this folder).

Generic: the limits below are the station's, not this task's, so the file can be dropped into another task folder
unchanged (it reads TOP from the task's layout.py, and the objects and demo from task.py).

    python3 -m unittest discover -s tests -v
"""

import importlib
import math
import os
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


# -- the station's limits --------------------------------------------------------------------------------------
HZ = 20.0
RUN_LIMIT_STEPS = 1200  # the program's default run limit (60 s)
SETTLE_S = 3.0  # the demo's total time plus this must fit the run limit
MAX_LOOSE_MASS = 0.3  # kg
FINGER_OPEN = -0.04695  # finger joint value fully open (gap 94 mm)
FINGER_MIN, FINGER_MAX = -0.04695, -0.008  # commanded joint values: a gap of 94 .. 16 mm
GRIP_NARROWER = 0.008  # a grip of a part commands a gap this much narrower than the part
GRIP_GAP_MIN, GRIP_GAP_MAX = 0.016, 0.080  # a gripped part is then 24 .. 88 mm across
FINGER_MOVE_MIN_S = 1.5  # every waypoint that opens or closes the fingers lasts at least this
PAD_TIPS = 0.013  # the pad tips are this far below the TCP
TCP_MIN_ABOVE_TOP = 0.013  # no TCP target lower than this above the table top (tips at the table)
REACH_X_MIN, REACH_X_MAX = 0.10, 0.50  # each arm reaches the table this far in front of its own base
BASE_Y = 0.305  # the arm bases stand at y +-0.305
MID_Y = 0.05  # at |y| <= this the reach shrinks to ...
MID_X_LOW, MID_X_HIGH = 0.40, 0.30  # ... this at table height, and this at z >= TOP + MID_Z
MID_Z = 0.16  # (interpolated linearly in |y| between MID_Y and BASE_Y)
TCP_SEP_Y = 0.20  # the two TCPs need this much between them in y ...
TCP_SEP_X = 0.05  # ... whenever they are within this in x
TABLE_Y, TABLE_X0, TABLE_X1 = 0.55, 0.05, 0.55  # every object within the table


def _max_x(y: float, z: float, top: float) -> float:
    """The furthest x an arm reaches at this |y| and height: the full 0.50 outboard of |y| 0.305, shrinking to
    0.40 (table height) or 0.30 (16 cm up) at the midline."""
    mid = MID_X_LOW if z < top + MID_Z else MID_X_HIGH
    ay = abs(y)
    if ay >= BASE_Y:
        return REACH_X_MAX
    if ay <= MID_Y:
        return mid
    u = (ay - MID_Y) / (BASE_Y - MID_Y)
    return mid + u * (REACH_X_MAX - mid)


def _objects_extent(o):
    """World-aligned bounding box (lo, hi) of a task-file object at its yaw (yaw only turns boxes)."""
    pos = np.array(o.pos, float)
    if hasattr(o, "size"):
        sx, sy, sz = (float(v) for v in o.size)
        yaw = float(getattr(o, "yaw", 0.0) or 0.0)
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey = (sx * c + sy * s) / 2, (sx * s + sy * c) / 2
        half = np.array([ex, ey, sz / 2])
    else:
        r = float(o.radius)
        h = float(getattr(o, "height", 2 * r))
        half = np.array([r, r, h / 2])
    return pos - half, pos + half


def sample(plan, ready, hz=HZ):
    """Each arm's TCP and finger target per control step on one clock, interpolating linearly between waypoints
    from the ready pose; a finished arm holds its last target until the longer plan ends."""
    total = max((sum(w.duration for w in wps) for wps in plan.values()), default=0.0)
    n = int(round(total * hz))
    out = {}
    for side, wps in plan.items():
        pos, fin = np.array(ready[side], float), FINGER_OPEN
        segs = []
        for w in wps:
            segs.append((pos, fin, np.array(w.pos, float), float(w.finger), float(w.duration)))
            pos, fin = np.array(w.pos, float), float(w.finger)
        rows = []
        for i in range(n + 1):
            t = i / hz
            cur = (pos, fin)
            acc = 0.0
            for p0, f0, p1, f1, d in segs:
                if t <= acc + d:
                    u = (t - acc) / d if d > 0 else 1.0
                    cur = ((1 - u) * p0 + u * p1, (1 - u) * f0 + u * f1)
                    break
                acc += d
            rows.append(cur)
        out[side] = rows
    return out, n


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class DemoPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        for key in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(key, None)
        cls.task = importlib.import_module("task")
        cls.L = importlib.import_module("layout")
        cls.TOP = float(cls.L.TOP)
        cls.plan = cls.task.demo()
        cls.ready = {"left": (0.323, BASE_Y, cls.TOP + 0.26), "right": (0.323, -BASE_Y, cls.TOP + 0.26)}
        cls.loose = [o for o in cls.task.OBJECTS if o.mass is not None]

    # -- objects -------------------------------------------------------------------------------------------------
    def test_object_masses(self):
        for o in self.task.OBJECTS:
            if o.mass is not None:
                self.assertGreater(o.mass, 0.0, o.name)
                self.assertLessEqual(o.mass, MAX_LOOSE_MASS, o.name)
                self.assertTrue(hasattr(o, "size"), f"{o.name}: a loose part must be a box, cylinder or sphere")

    def test_objects_within_the_table(self):
        for o in self.task.OBJECTS:
            lo, hi = _objects_extent(o)
            self.assertGreaterEqual(lo[0], TABLE_X0 - 1e-9, o.name)
            self.assertLessEqual(hi[0], TABLE_X1 + 1e-9, o.name)
            self.assertLessEqual(abs(lo[1]), TABLE_Y + 1e-9, o.name)
            self.assertLessEqual(abs(hi[1]), TABLE_Y + 1e-9, o.name)
            self.assertGreaterEqual(lo[2], self.TOP - 1e-6, f"{o.name} below the table top")

    def test_object_types_and_names(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)), "object names must be unique")
        kinds = {type(o).__name__ for o in self.task.OBJECTS}
        self.assertTrue(kinds <= {"Box", "Cylinder", "Sphere"}, kinds)

    # -- waypoints -----------------------------------------------------------------------------------------------
    def _waypoints(self):
        for side, wps in self.plan.items():
            for i, w in enumerate(wps):
                yield side, i, w

    def test_finger_values_within_the_commanded_range(self):
        for side, i, w in self._waypoints():
            self.assertGreaterEqual(w.finger, FINGER_MIN - 1e-9, f"{side}[{i}] {w.label}")
            self.assertLessEqual(w.finger, FINGER_MAX + 1e-9, f"{side}[{i}] {w.label}")
            self.assertGreater(w.duration, 0.0, f"{side}[{i}] {w.label}")

    def test_grips_are_8mm_narrower_than_a_part(self):
        """Every finger value other than fully open or fully closed must be a grip of a loose part: a gap 8 mm
        narrower than one of its horizontal sides (so 16 .. 80 mm)."""
        sides = set()
        for o in self.loose:
            if hasattr(o, "size"):
                sides |= {round(float(o.size[0]), 6), round(float(o.size[1]), 6)}
            else:
                sides.add(round(2 * float(o.radius), 6))
        for side, i, w in self._waypoints():
            gap = -2 * w.finger
            if abs(w.finger - FINGER_MIN) < 1e-9 or abs(w.finger - FINGER_MAX) < 1e-9:
                continue  # fully open, or closed on nothing
            self.assertGreaterEqual(gap, GRIP_GAP_MIN - 1e-9, f"{side}[{i}] {w.label}: gap {gap * 1000:.1f} mm")
            self.assertLessEqual(gap, GRIP_GAP_MAX + 1e-9, f"{side}[{i}] {w.label}: gap {gap * 1000:.1f} mm")
            self.assertTrue(any(abs(gap - (s - GRIP_NARROWER)) < 1e-6 for s in sides),
                            f"{side}[{i}] {w.label}: a {gap * 1000:.1f} mm gap grips no loose part 8 mm wider")

    def test_finger_moves_take_at_least_1_5s(self):
        for side, wps in self.plan.items():
            fin = FINGER_OPEN
            for i, w in enumerate(wps):
                if abs(w.finger - fin) > 1e-9:
                    self.assertGreaterEqual(w.duration, FINGER_MOVE_MIN_S - 1e-9,
                                            f"{side}[{i}] {w.label}: fingers move in {w.duration} s")
                fin = w.finger

    def test_tcp_targets_above_the_table(self):
        for side, i, w in self._waypoints():
            self.assertGreaterEqual(w.pos[2], self.TOP + TCP_MIN_ABOVE_TOP - 1e-9,
                                    f"{side}[{i}] {w.label}: TCP {w.pos[2] - self.TOP:.4f} above the top")

    def test_tcp_targets_within_reach(self):
        for side, i, w in self._waypoints():
            x, y, z = (float(v) for v in w.pos)
            self.assertGreaterEqual(x, REACH_X_MIN - 1e-9, f"{side}[{i}] {w.label}: x {x}")
            self.assertLessEqual(x, _max_x(y, z, self.TOP) + 1e-9,
                                 f"{side}[{i}] {w.label}: x {x} beyond reach at y {y}, z {z - self.TOP:.3f} up")
            # each arm's own side: it reaches across the midline only a little
            own = 1.0 if side == "left" else -1.0
            self.assertGreaterEqual(own * y, -0.18 - 1e-9, f"{side}[{i}] {w.label}: y {y} too far across")
            self.assertLessEqual(abs(y), TABLE_Y, f"{side}[{i}] {w.label}: y {y} off the table")

    def test_grasps_are_straight_down_with_yaw_only(self):
        for side, i, w in self._waypoints():
            yaw = float(getattr(w, "yaw", 0.0) or 0.0)
            self.assertLessEqual(abs(yaw), math.pi + 1e-9, f"{side}[{i}] {w.label}")

    # -- the two arms together -------------------------------------------------------------------------------------
    def test_tcps_keep_apart(self):
        rows, n = sample(self.plan, self.ready)
        if len(rows) < 2:
            self.skipTest("one arm only")
        worst = None
        for i in range(n + 1):
            (pl, _), (pr, _) = rows["left"][i], rows["right"][i]
            if abs(pl[0] - pr[0]) <= TCP_SEP_X:
                dy = abs(pl[1] - pr[1])
                if worst is None or dy < worst[0]:
                    worst = (dy, i / HZ)
                self.assertGreaterEqual(dy, TCP_SEP_Y - 1e-9,
                                        f"at {i / HZ:.2f} s the TCPs are {dy * 1000:.0f} mm apart in y, within 5 cm in x")
        self.assertIsNotNone(worst)

    def test_total_time_fits_the_run_limit(self):
        total = max(sum(w.duration for w in wps) for wps in self.plan.values())
        self.assertLess((total + SETTLE_S) * HZ, RUN_LIMIT_STEPS,
                        f"demo {total:.1f} s plus {SETTLE_S} s settle exceeds {RUN_LIMIT_STEPS} steps")
        self.assertGreater(total, 5.0)

    def test_switches(self):
        try:
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(self.task.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY", None)
        for idle, keep in (("left", "right"), ("right", "left")):
            try:
                os.environ["DEMO_IDLE_ARM"] = idle
                self.assertEqual(set(self.task.demo()), {keep})
            finally:
                os.environ.pop("DEMO_IDLE_ARM", None)
        self.assertEqual(set(self.task.demo()), {"left", "right"})


if __name__ == "__main__":
    unittest.main()
