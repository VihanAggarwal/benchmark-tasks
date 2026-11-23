"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's own
specs), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import pathlib
import re
import sys
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
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")


def gel_pose(side, inward=True, dy=0.0, dz=0.0, dx=0.0):
    """A cassette upright on the ``side`` gasket (short plate inward unless ``inward`` is False)."""
    s = L.arm_sign(side)
    # short plate on local -y; inward means local -y points toward the centre (-s in world y)
    yaw = 0.0 if s > 0 else math.pi
    if not inward:
        yaw += math.pi
    r = mk.quat_to_matrix(mk.quat_from_ypr(yaw, 0.0, 0.0))
    o = np.array([L.FX + dx, s * (L.GY + L.CS_T / 2) + s * dy, L.L0 + dz])
    return o, r


def states(a=None, b=None, q_pos=0.0, q_neg=0.0):
    out = {}
    for name, pose, start in ((S.GEL_A, a, L.A_START), (S.GEL_B, b, L.B_START)):
        if pose is None:
            o = np.array(start[:3])
            r = mk.quat_to_matrix(mk.quat_from_ypr(start[3], 0.0, 0.0))
        else:
            o, r = pose
        out[name] = mk.state_from_poses(T.SPECS[name], (o[0], o[1], L.TOP + o[2], *mk.matrix_to_quat(r)),
                                        source="offline")
    out[S.FRAME] = mk.state_from_poses(T.FRAME_PART, (L.FX, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                       joints={"arm_pos": q_pos, "arm_neg": q_neg}, source="offline")
    return out


def raw(st):
    r = S.read(st)
    return S.raw_score(r)[0]


class Geometry(unittest.TestCase):
    def test_constants(self):
        self.assertAlmostEqual(L.CS_W, 0.100)
        self.assertAlmostEqual(L.CS_H, 0.080)
        self.assertAlmostEqual(L.CS_T, 0.0046)
        self.assertAlmostEqual(L.PIV_Y - L.AW / 2, L.GY + L.CS_T)  # shut, the arm lies on the cassette's face
        self.assertGreater(L.NOTCH_Z, L.L0 + L.CS_HS)
        self.assertLess(L.NOTCH_Z - (L.L0 + L.CS_HS), L.NOTCH_LOW)

    def test_credits_sum_to_one(self):
        self.assertAlmostEqual(2 * (L.CREDIT_AGAINST + L.CREDIT_SEATED + L.CREDIT_ARM), 1.0)

    def test_rotations_proper(self):
        for side in S.SIDES:
            for inward in (True, False):
                _, r = gel_pose(side, inward)
                self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=6)


class Credit(unittest.TestCase):
    def test_start_is_zero(self):
        self.assertEqual(raw(states()), 0.0)

    def test_full(self):
        st = states(gel_pose("pos"), gel_pose("neg"), L.Q_CLOSED, L.Q_CLOSED)
        self.assertAlmostEqual(raw(st), 1.0)

    def test_seated_no_arms(self):
        self.assertAlmostEqual(raw(states(gel_pose("pos"), gel_pose("neg"))), 0.6)

    def test_reversed_earns_nothing(self):
        st = states(gel_pose("pos", inward=False), gel_pose("neg"), L.Q_CLOSED, L.Q_CLOSED)
        self.assertAlmostEqual(raw(st), 0.5)

    def test_high_is_against_not_seated(self):
        st = states(gel_pose("pos", dz=0.006), gel_pose("neg"))
        self.assertAlmostEqual(raw(st), 0.45)

    def test_gap_fails(self):
        st = states(gel_pose("pos", dy=0.003), None)
        self.assertEqual(raw(st), 0.0)

    def test_arm_not_shut(self):
        st = states(gel_pose("pos"), gel_pose("neg"), L.Q_CLOSED - math.radians(5), L.Q_CLOSED)
        self.assertAlmostEqual(raw(st), 0.8)

    def test_wrong_side(self):
        o, r = gel_pose("neg")
        st = states((o, r), None, L.Q_CLOSED, L.Q_CLOSED)  # gel A on the -y gasket
        self.assertAlmostEqual(raw(st), 0.5)


class TrackerRules(unittest.TestCase):
    def test_dropped_latches_zero(self):
        tr = S.Tracker()
        o = np.array([0.3, 0.2, -0.5])
        floor = (o, np.eye(3))
        tr.update(states(floor, None), None)
        st = states(gel_pose("pos"), gel_pose("neg"), L.Q_CLOSED, L.Q_CLOSED)
        for _ in range(40):
            res = tr.update(st, None)
        self.assertIsNotNone(tr.dropped)
        self.assertEqual(res.score, 0.0)

    def test_unreadable_is_zero(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, None).score, 0.0)


class ClipRule(unittest.TestCase):
    def test_latches_only_present(self):
        t, latched = T.clip_targets(L.Q_LATCH + 0.001, 0.0, False, True)
        self.assertTrue(latched)
        t, latched = T.clip_targets(L.Q_LATCH + 0.001, 0.0, False, False)
        self.assertFalse(latched)

    def test_cam_pushes_back(self):
        q = L.Q_CAM0 + 0.02
        t, latched = T.clip_targets(q, 0.0, False, True)
        self.assertFalse(latched)
        self.assertLess(t, q)


class DemoText(unittest.TestCase):
    def test_demo_env(self):
        import os
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(T.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_task_json(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json")
        d = json.loads(p.read_text())
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        text = json.dumps(d) + (HERE / "README.md").read_text()
        self.assertIsNone(re.search("[–—]", text))
        self.assertIsNone(re.search(r"(?i)t-?70\b", text))


if __name__ == "__main__":
    unittest.main()
