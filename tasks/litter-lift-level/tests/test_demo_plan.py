"""Offline checks of the scripted demo against the simulator's and station's limits (no simulator: the plan's
numbers only). Generic: it reads task.py's OBJECTS and demo() through the offline yam_arena stub and knows nothing
of this task but the table top. Reusable by any task folder laid out like this one.

    python3 -m unittest discover -s tests -v

Limits checked (from the program's build notes):
- every loose object's mass <= 0.3 kg; a fixed object has no mass;
- every object stands within the table (|y| <= 0.55, 0.05 <= x <= 0.55);
- every commanded finger value is within [-0.04695, -0.008] (a gap of 16..94 mm), and every grip (a finger value
  the plan closes to) is 8 mm narrower than some loose part's horizontal size it could be holding (gap 16..80 mm);
- every waypoint that changes the finger value lasts >= 1.5 s;
- every TCP target is within its arm's reach: x within 0.10..0.50 m, and near the midline the reach shrinks: at
  |y| <= 0.05, x <= 0.40 at table height and x <= 0.30 when z >= TOP + 0.16, interpolated linearly in |y| between
  0.05 and 0.305 (at |y| >= 0.305 the full 0.50);
- no TCP target below TOP + 0.013 (the pad tips at the table);
- the two TCPs are never within 0.20 m of each other in y while within 0.05 m of each other in x (both plans
  sampled on one clock at 20 Hz, interpolating linearly between waypoints);
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

TOP = 0.7475  # the station's table top
HZ = 20.0
RUN_LIMIT_STEPS = 1200
HOLD_AFTER_S = 3.0  # the settle after the plan before the scripted policy stops
FINGER_OPEN = -0.04695  # the finger joint value with the pads 94 mm apart
FINGER_MIN_GAP, FINGER_MAX_GAP = 0.016, 0.0939
GRIP_MAX_GAP = 0.080  # a part gripped 8 mm narrower than its size: the size is at most 88 mm, the gap 80
GRIP_NARROWER = 0.008
GRIP_TOL = 1e-6
MAX_LOOSE_MASS = 0.3
TABLE_Y, TABLE_X = 0.55, (0.05, 0.55)
REACH_X = (0.10, 0.50)
MID_Y, BASE_Y = 0.05, 0.305  # the reach shrinks toward the midline between these |y|
MID_X_LOW, MID_X_HIGH = 0.40, 0.30  # the midline's x limit at table height, and from HIGH_Z up
HIGH_Z = TOP + 0.16
MIN_TCP_Z = TOP + 0.013  # the pad tips at the table
MIN_TCP_Y_GAP, TCP_X_NEAR = 0.20, 0.05
MIN_FINGER_MOVE_S = 1.5
GRIPPER_RATE_LIMIT = 0.05  # the scripted policy's per-step limit on the finger command (fraction of fully open)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def load_task():
    sys.path.insert(0, str(_stub_dir()))
    if "task" in sys.modules:
        return importlib.reload(sys.modules["task"])
    return importlib.import_module("task")


def object_extent(o):
    """(half size along x, half size along y, height) of a stub object, whatever its kind."""
    if hasattr(o, "size"):
        sx, sy, sz = o.size
        c, s = abs(np.cos(getattr(o, "yaw", 0.0))), abs(np.sin(getattr(o, "yaw", 0.0)))
        return (sx * c + sy * s) / 2, (sx * s + sy * c) / 2, sz
    if hasattr(o, "height"):  # a cylinder
        return o.radius, o.radius, o.height
    return o.radius, o.radius, 2 * o.radius  # a sphere


def horizontal_sizes(o):
    """The sizes a gripper could close across: a box's x and y sides, a cylinder's or sphere's diameter."""
    if hasattr(o, "size"):
        return [o.size[0], o.size[1]]
    return [2 * o.radius]


def reach_x_limit(y: float, z: float) -> float:
    """The largest x an arm reaches at (|y|, z): the full 0.50 outboard of its base, shrinking toward the midline."""
    mid = MID_X_LOW if z < HIGH_Z else MID_X_HIGH
    ay = abs(y)
    if ay <= MID_Y:
        return mid
    if ay >= BASE_Y:
        return REACH_X[1]
    f = (ay - MID_Y) / (BASE_Y - MID_Y)
    return mid + f * (REACH_X[1] - mid)


def arm_sign(arm: str) -> float:
    return 1.0 if arm == "left" else -1.0


def sample(plan, hz=HZ):
    """Each arm's TCP position and finger value at every control step of the demo, both on one clock (an arm with
    no plan, or whose plan has ended, holds its last target; before its first waypoint it holds that waypoint's
    target, which is how the policy starts from the ready pose: a straight move to the first target)."""
    total = max((sum(w.duration for w in wps) for wps in plan.values()), default=0.0)
    n = int(round(total * hz)) + 1
    out = {}
    for arm, wps in plan.items():
        pos, fin = np.zeros((n, 3)), np.zeros(n)
        if not wps:
            continue
        t0, prev = 0.0, None
        times = np.arange(n) / hz
        for w in wps:
            t1 = t0 + w.duration
            a = np.array(w.pos if prev is None else prev.pos, float)
            b = np.array(w.pos, float)
            fa = w.finger if prev is None else prev.finger
            m = (times >= t0) & (times <= t1)
            f = np.clip((times[m] - t0) / max(w.duration, 1e-9), 0.0, 1.0)
            pos[m] = a + np.outer(f, b - a)
            fin[m] = fa + f * (w.finger - fa)
            t0, prev = t1, w
        m = times > t0
        pos[m], fin[m] = np.array(prev.pos, float), prev.finger
        out[arm] = (pos, fin)
    return out


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class DemoPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        old = {k: os.environ.pop(k, None) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
        try:
            cls.plan = cls.task.demo()
        finally:
            os.environ.update({k: v for k, v in old.items() if v is not None})
        cls.loose = [o for o in cls.task.OBJECTS if o.mass is not None]

    def waypoints(self):
        for arm, wps in self.plan.items():
            prev_finger = FINGER_OPEN  # each arm starts fully open
            for i, w in enumerate(wps):
                yield arm, i, w, prev_finger
                prev_finger = w.finger

    # -- objects --------------------------------------------------------------------------------------------------
    def test_objects_are_light_enough_and_on_the_table(self):
        for o in self.task.OBJECTS:
            with self.subTest(o.name):
                if o.mass is not None:
                    self.assertGreater(o.mass, 0.0)
                    self.assertLessEqual(o.mass, MAX_LOOSE_MASS)
                hx, hy, hz = object_extent(o)
                self.assertGreaterEqual(o.pos[0] - hx, TABLE_X[0] - 1e-9)
                self.assertLessEqual(o.pos[0] + hx, TABLE_X[1] + 1e-9)
                self.assertLessEqual(abs(o.pos[1]) + hy, TABLE_Y + 1e-9)
                self.assertGreaterEqual(o.pos[2] - hz / 2, TOP - 1e-6)  # nothing below the table top

    def test_object_names_are_unique(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))

    # -- fingers ----------------------------------------------------------------------------------------------------
    def test_finger_values_are_within_the_gripper(self):
        for arm, i, w, _ in self.waypoints():
            with self.subTest(arm=arm, i=i, label=w.label):
                gap = -2 * w.finger
                self.assertGreaterEqual(gap, FINGER_MIN_GAP - GRIP_TOL)
                self.assertLessEqual(gap, FINGER_MAX_GAP + GRIP_TOL)

    def test_grips_are_8mm_narrower_than_a_loose_part(self):
        sizes = sorted({round(s, 6) for o in self.loose for s in horizontal_sizes(o)})
        for arm, i, w, prev in self.waypoints():
            if w.finger >= prev + 1e-9:  # a close: the value is -(gap)/2, so a narrower gap is a larger value
                gap = -2 * w.finger
                with self.subTest(arm=arm, i=i, label=w.label, gap_mm=round(gap * 1000, 1)):
                    self.assertGreaterEqual(gap, FINGER_MIN_GAP - GRIP_TOL)
                    self.assertLessEqual(gap, GRIP_MAX_GAP + GRIP_TOL)
                    self.assertTrue(any(abs(gap - (s - GRIP_NARROWER)) < GRIP_TOL for s in sizes),
                                    f"gap {gap * 1000:.1f} mm is not 8 mm under a part size in {sizes}")

    def test_finger_moves_are_slow_enough(self):
        # every change of the finger target: at least 1.5 s, and under the CLI's 0.05-per-step gripper limiter at
        # 20 Hz (a smoothstep peaks at 1.5 times its mean rate)
        for arm, i, w, prev in self.waypoints():
            if abs(w.finger - prev) > 1e-9:
                with self.subTest(arm=arm, i=i, label=w.label):
                    self.assertGreaterEqual(w.duration, MIN_FINGER_MOVE_S - 1e-9)
                    change = abs(w.finger - prev) / abs(FINGER_OPEN)
                    self.assertLessEqual(1.5 * change / (w.duration * HZ), GRIPPER_RATE_LIMIT)

    # -- reach --------------------------------------------------------------------------------------------------------
    def test_targets_are_within_reach(self):
        for arm, i, w, _ in self.waypoints():
            x, y, z = w.pos
            with self.subTest(arm=arm, i=i, label=w.label, pos=tuple(round(v, 3) for v in w.pos)):
                self.assertGreaterEqual(x, REACH_X[0] - 1e-9)
                self.assertLessEqual(x, REACH_X[1] + 1e-9)
                self.assertGreaterEqual(z, MIN_TCP_Z - 1e-9)
                self.assertLessEqual(abs(y), TABLE_Y + 1e-9)
                # an arm reaches across the midline only a little: its targets stay on its own side or near it
                self.assertGreaterEqual(y * arm_sign(arm), -MID_Y - 1e-9)
                self.assertLessEqual(x, reach_x_limit(y, z) + 1e-9)

    def test_tcps_keep_apart(self):
        arms = sample(self.plan)
        if len(arms) < 2:
            self.skipTest("one arm only")
        (pl, _), (pr, _) = arms["left"], arms["right"]
        near_x = np.abs(pl[:, 0] - pr[:, 0]) <= TCP_X_NEAR
        gap_y = np.abs(pl[:, 1] - pr[:, 1])
        bad = np.flatnonzero(near_x & (gap_y < MIN_TCP_Y_GAP - 1e-9))
        self.assertEqual(len(bad), 0, f"TCPs too close at steps {bad[:10].tolist()} (y gap {gap_y[bad[:1]]} m)")

    def test_sampled_path_stays_within_reach(self):
        # the interpolated path between waypoints, not only the waypoints, stays within reach and above the table
        for arm, (pos, fin) in sample(self.plan).items():
            with self.subTest(arm=arm):
                self.assertTrue(np.all(pos[:, 2] >= MIN_TCP_Z - 1e-9))
                self.assertTrue(np.all(pos[:, 0] >= REACH_X[0] - 1e-9))
                lim = np.array([reach_x_limit(y, z) for _, y, z in pos])
                self.assertTrue(np.all(pos[:, 0] <= lim + 1e-9))
                self.assertTrue(np.all(-2 * fin >= FINGER_MIN_GAP - 1e-6))
                self.assertTrue(np.all(-2 * fin <= FINGER_MAX_GAP + 1e-6))

    # -- timing ---------------------------------------------------------------------------------------------------------
    def test_demo_fits_the_run_limit(self):
        total = max(sum(w.duration for w in wps) for wps in self.plan.values())
        self.assertLess((total + HOLD_AFTER_S) * HZ, RUN_LIMIT_STEPS)
        for arm, wps in self.plan.items():
            for w in wps:
                self.assertGreater(w.duration, 0.0, (arm, w.label))

    def test_switches(self):
        old = {k: os.environ.pop(k, None) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
        try:
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(self.task.demo(), {})
            os.environ.pop("DEMO_HOLD_ONLY")
            for arm, other in (("left", "right"), ("right", "left")):
                os.environ["DEMO_IDLE_ARM"] = arm
                plan = self.task.demo()
                self.assertNotIn(arm, plan)
                self.assertIn(other, plan)
            os.environ.pop("DEMO_IDLE_ARM")
            self.assertEqual(set(self.task.demo()), set(self.plan))
        finally:
            for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
                os.environ.pop(k, None)
            os.environ.update({k: v for k, v in old.items() if v is not None})


if __name__ == "__main__":
    unittest.main()
