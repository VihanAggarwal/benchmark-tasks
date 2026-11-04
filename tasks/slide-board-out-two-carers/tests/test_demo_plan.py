"""Offline checks of the scripted demo against the simulator's and the station's limits (no simulator; the
program's tools/offline/check_task.py is not available here).  python3 -m unittest discover -s tests -v

Generic: it reads only ``task.OBJECTS``, ``task.demo()``, ``task.SUCCESS_HOLD_S`` and, if present,
``task.RUN_LIMIT_STEPS`` (default 1,200) and ``layout.TOP`` (default 0.7475), so it can be dropped into another task
folder next to the same offline stub. The checks and their numbers are the station's rules:

- every loose object's mass is at most 0.3 kg; a fixed object has none;
- every commanded finger value is within [-0.04695, -0.008] (a gap of 16..94 mm), and every grip of a part is
  8 mm narrower than one of that part's horizontal sides (a gap of 16..80 mm);
- every waypoint that changes the finger value (opening or closing) lasts at least 1.5 s;
- every TCP target is in its arm's reach: x within 0.10..0.50 m, shrinking toward the midline (at |y| <= 0.05,
  x <= 0.40 at table height and x <= 0.30 at 16 cm up or higher, interpolated linearly in |y| between 0.05 and
  0.305 and in z between 1.3 cm and 16 cm up);
- the two TCPs are never within 0.20 m of each other in y while within 0.05 m in x (both plans sampled on one
  clock at 20 Hz, interpolated linearly between waypoints, from the ready pose);
- no TCP target is below TOP + 0.013 (the pad tips at the table);
- every object is within the table (|y| <= 0.55, 0.05 <= x <= 0.55);
- the demo's total time plus 3 s is under the run limit;
- DEMO_HOLD_ONLY=1 gives an empty plan and DEMO_IDLE_ARM=left|right drops that arm.
"""

import importlib
import os
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

OPEN = -0.04695  # the finger joint value with the pads 94 mm apart
FINGER_MIN, FINGER_MAX = -0.04695, -0.008  # commanded gaps of 94 mm .. 16 mm
GRIP_UNDERSIZE = 0.008  # a part is gripped with the fingers commanded this much narrower than it
GRIP_GAP_MAX = 0.080  # the widest part the pads can grip (gap 16..80 mm)
MAX_LOOSE_MASS = 0.3  # kg
MIN_GRIP_S = 1.5  # an open or close lasts at least this long
PAD_TIPS = 0.013  # the pad tips are this far below the TCP
REACH_X = (0.10, 0.50)  # an arm covers this much of the table in front of its own base
MID_X_LOW, MID_X_HIGH = 0.40, 0.30  # ... but at the midline only this far at table height / 16 cm up
MID_Y, BASE_Y = 0.05, 0.305  # the midline band, and the arm bases' |y|
HIGH_Z = 0.16  # above the table top: where the midline reach is MID_X_HIGH
TCP_SEP_Y, TCP_SAME_X = 0.20, 0.05  # two grippers within TCP_SAME_X in x need TCP_SEP_Y between them in y
TABLE_X, TABLE_Y = (0.05, 0.55), 0.55  # the table's usable extent
SETTLE_S = 3.0  # added to the demo's time before the run limit
HZ = 20.0
READY = {"left": (0.323, 0.305, 0.26), "right": (0.323, -0.305, 0.26)}  # each arm's ready pose (z above TOP)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _env(**kv):
    """Run a block with these environment variables set, restoring them after."""
    class _Ctx:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kv}
            for k, v in kv.items():
                os.environ[k] = v

        def __exit__(self, *exc):
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _Ctx()


def sample(wps, ready, hz=HZ):
    """(t, x, y, z, finger) at each control step of a plan, interpolated linearly between waypoints, starting
    from the ready pose at t 0."""
    times = [0.0]
    pts = [np.array(ready, float)]
    fingers = [OPEN]
    for w in wps:
        times.append(times[-1] + float(w.duration))
        pts.append(np.array(w.pos, float))
        fingers.append(float(w.finger))
    times, pts, fingers = np.array(times), np.array(pts), np.array(fingers)
    t = np.arange(0.0, times[-1] + 1e-9, 1.0 / hz)
    xyz = np.stack([np.interp(t, times, pts[:, i]) for i in range(3)], 1)
    return t, xyz, np.interp(t, times, fingers)


def reach_x_max(y, z, top):
    """The largest x an arm reaches at (|y|, z): 0.50 at and beyond its base's |y|, less toward the midline, and
    less again higher up."""
    fy = float(np.clip((abs(y) - MID_Y) / (BASE_Y - MID_Y), 0.0, 1.0))
    fz = float(np.clip((z - top - PAD_TIPS) / (HIGH_Z - PAD_TIPS), 0.0, 1.0))
    mid = MID_X_LOW + (MID_X_HIGH - MID_X_LOW) * fz
    return mid + (REACH_X[1] - mid) * fy


def box_extent(o):
    """World-aligned half extents of a task-file object from its size (Box), radius and height (Cylinder) or
    radius (Sphere)."""
    if hasattr(o, "size"):
        sx, sy, sz = o.size
        yaw = float(getattr(o, "yaw", 0.0) or 0.0)
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        return np.array([sx * c + sy * s, sx * s + sy * c, sz]) / 2
    if hasattr(o, "height"):
        return np.array([o.radius, o.radius, o.height / 2])
    return np.array([o.radius] * 3)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class DemoPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        try:
            cls.top = float(importlib.import_module("layout").TOP)
        except (ImportError, AttributeError):
            cls.top = 0.7475
        cls.run_limit = int(getattr(cls.task, "RUN_LIMIT_STEPS", 1200))
        with _env(DEMO_HOLD_ONLY="0", DEMO_IDLE_ARM=""):
            cls.plan = cls.task.demo()
        cls.ready = {a: (x, y, cls.top + z) for a, (x, y, z) in READY.items()}

    def test_objects_are_loose_or_fixed_within_the_mass_limit_and_on_the_table(self):
        for o in self.task.OBJECTS:
            with self.subTest(object=o.name):
                if o.mass is not None:
                    self.assertGreater(o.mass, 0.0)
                    self.assertLessEqual(o.mass, MAX_LOOSE_MASS)
                lo, hi = np.array(o.pos) - box_extent(o), np.array(o.pos) + box_extent(o)
                self.assertGreaterEqual(lo[0], TABLE_X[0] - 1e-9)
                self.assertLessEqual(hi[0], TABLE_X[1] + 1e-9)
                self.assertLessEqual(max(abs(lo[1]), abs(hi[1])), TABLE_Y + 1e-9)
                self.assertGreaterEqual(lo[2], self.top - 1e-6)  # nothing below the table top

    def test_finger_values_and_grip_sizes(self):
        sides = sorted({round(float(s), 6) for o in self.task.OBJECTS if o.mass is not None
                        for s in (o.size[:2] if hasattr(o, "size") else (2 * o.radius,))})
        for arm, wps in self.plan.items():
            for i, w in enumerate(wps):
                with self.subTest(arm=arm, waypoint=i, label=w.label):
                    self.assertGreaterEqual(w.finger, FINGER_MIN - 1e-9)
                    self.assertLessEqual(w.finger, FINGER_MAX + 1e-9)
                    if w.finger > OPEN + 1e-9:  # a grip: 8 mm narrower than a loose part's side, 16..80 mm
                        gap = -2 * w.finger
                        self.assertLessEqual(gap, GRIP_GAP_MAX + 1e-9)
                        self.assertTrue(any(abs(gap + GRIP_UNDERSIZE - s) < 1e-6 for s in sides),
                                        f"gap {gap * 1000:.1f} mm is not 8 mm under a loose part's side {sides}")

    def test_opening_and_closing_take_at_least_1_5_s(self):
        for arm, wps in self.plan.items():
            prev = OPEN
            for i, w in enumerate(wps):
                with self.subTest(arm=arm, waypoint=i, label=w.label):
                    if abs(w.finger - prev) > 1e-9:
                        self.assertGreaterEqual(w.duration, MIN_GRIP_S - 1e-9)
                prev = w.finger

    def test_targets_within_reach_and_above_the_table(self):
        for arm, wps in self.plan.items():
            for i, w in enumerate(wps):
                x, y, z = (float(v) for v in w.pos)
                with self.subTest(arm=arm, waypoint=i, label=w.label):
                    self.assertGreater(w.duration, 0.0)
                    self.assertGreaterEqual(z, self.top + PAD_TIPS - 1e-9)
                    self.assertGreaterEqual(x, REACH_X[0] - 1e-9)
                    self.assertLessEqual(x, reach_x_max(y, z, self.top) + 1e-9)
                    own = 1.0 if arm == "left" else -1.0
                    self.assertGreaterEqual(y * own, -BASE_Y, "a target on the far side of the other arm's base")

    def test_tcps_keep_apart(self):
        if len(self.plan) < 2:
            self.skipTest("one arm only")
        tl, pl, _ = sample(self.plan["left"], self.ready["left"])
        tr, pr, _ = sample(self.plan["right"], self.ready["right"])
        n = max(len(tl), len(tr))
        pl = np.vstack([pl, np.repeat(pl[-1:], n - len(pl), 0)])  # an arm that finishes first holds its last pose
        pr = np.vstack([pr, np.repeat(pr[-1:], n - len(pr), 0)])
        same_x = np.abs(pl[:, 0] - pr[:, 0]) <= TCP_SAME_X
        sep_y = pl[:, 1] - pr[:, 1]  # the left arm is at larger y
        close = same_x & (sep_y < TCP_SEP_Y)
        self.assertFalse(close.any(), f"TCPs too close at t {np.nonzero(close)[0] / HZ}")
        self.assertTrue((sep_y > 0).all(), "the arms cross")

    def test_duration_within_the_run_limit(self):
        total = max(sum(w.duration for w in wps) for wps in self.plan.values())
        self.assertLess((total + SETTLE_S) * HZ, self.run_limit)
        self.assertGreater(self.task.SUCCESS_HOLD_S, 0.0)
        self.assertLess((total + self.task.SUCCESS_HOLD_S + SETTLE_S) * HZ, self.run_limit)

    def test_switches(self):
        with _env(DEMO_HOLD_ONLY="1"):
            self.assertEqual(self.task.demo(), {})
        for idle, left in (("left", {"right"}), ("right", {"left"})):
            with _env(DEMO_HOLD_ONLY="0", DEMO_IDLE_ARM=idle):
                plan = self.task.demo()
                self.assertEqual(set(plan), left & set(self.plan))
                for arm in plan:
                    self.assertEqual([w.label for w in plan[arm]], [w.label for w in self.plan[arm]])


if __name__ == "__main__":
    unittest.main()
