"""Offline tests for layout.py, scoring.py and task.py: synthetic states for the pieces (yam_arena ObjectStates), the
dustpan and the brush (mechkit AssemblyStates from the task's own specs) and synthetic arm snapshots; no simulator.

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

AWAY = {a: rigstate.Arm(np.array([0.08, s * 0.30, L.TOP + 0.30]), np.array([0.08, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
START = {name: np.array([x, y]) for name, (x, y), _, _ in L.SHARDS}
# a spot on the pan's floor for each piece (world), spread across it
IN_PAN = {name: np.array([L.PAN_LIP_X - 0.050 - 0.012 * (i % 4), -0.060 + 0.035 * (i // 2 % 4)])
          for i, (name, _, _, _) in enumerate(L.SHARDS)}


def yaw_q(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def tilt_q(deg):
    a = math.radians(deg) / 2
    return (math.sin(a), 0.0, 0.0, math.cos(a))


def piece(name, xy, z=None):
    sz = S.SIZES[name]
    c = np.array([xy[0], xy[1], L.TOP + sz[2] / 2 if z is None else z])
    return ObjectState(pos=c, quat_xyzw=np.array(yaw_q(0.0)), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                       low=c - np.array(sz) / 2, high=c + np.array(sz) / 2)


def pan_state(tilt=0.0, dz=0.0):
    q = np.array(yaw_q(math.pi))
    if tilt:
        t = np.array(tilt_q(tilt))
        x1, y1, z1, w1 = q
        x2, y2, z2, w2 = t
        q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    return mk.state_from_poses(T.PAN_SPEC, (L.PAN_LIP_X, L.PAN_Y, L.TOP + dz, *q), np.zeros(6))


def brush_state():
    return mk.state_from_poses(T.BRUSH_SPEC, (*L.BRUSH_START, L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6))


def states(where, pan=None, dz=0.0):
    out = {S.PAN: pan or pan_state(dz=dz), S.BRUSH: brush_state()}
    for name in S.NAMES:
        xy = where.get(name, START[name])
        z = (L.TOP + dz + L.PAN_FLOOR * L.MM + S.SIZES[name][2] / 2 if name in where and where[name] is IN_PAN[name]
             else None)
        out[name] = piece(name, xy, z)
    return out


def swept(k):
    """The first k pieces in the pan, the rest where they started."""
    return {name: IN_PAN[name] for name in S.NAMES[:k]}


class TestScoring(unittest.TestCase):
    def run_steps(self, where, arms=AWAY, n=40, tracker=None, pan=None, dz=0.0):
        t = tracker or S.Tracker()
        r = None
        for _ in range(n):
            r = t.update(states(where, pan, dz), arms)
        return t, r

    def test_start_is_zero(self):
        _, r = self.run_steps({})
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["in_pan"], 0)

    def test_all_in_pan_and_left_is_full(self):
        _, r = self.run_steps(swept(8))
        self.assertEqual(r.details["in_pan"], 8)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_tiers(self):
        for k, want in ((3, 0.0), (4, 0.25), (5, 0.25), (6, 0.5), (7, 0.5)):
            _, r = self.run_steps(swept(k))
            self.assertEqual(r.score, want, k)

    def test_all_in_but_pan_not_set_down_is_half(self):
        _, r = self.run_steps(swept(8), dz=0.006)  # held 6 mm off the table, the pieces in it
        self.assertEqual(r.details["in_pan"], 8)
        self.assertFalse(r.details["pan_flat"])
        self.assertEqual(r.score, 0.5)

    def test_pan_tilted_is_not_flat(self):
        self.assertFalse(S.pan_flat(S.read(states({}, pan=pan_state(tilt=8.0)))))

    def test_all_in_but_still_held_is_half(self):
        tcp = np.array([T.BACK_WALL_X, L.PAN_Y + 0.05, L.TOP + 0.041])
        holding = {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.0), "right": AWAY["right"]}
        _, r = self.run_steps(swept(8), arms=holding)
        self.assertEqual(r.score, 0.5)

    def test_touching_a_piece_latches_zero(self):
        t = S.Tracker()
        p = START["shard_d"]
        tcp = np.array([p[0], p[1], L.TOP + 0.016])
        touching = {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.0), "right": AWAY["right"]}
        t.update(states({}), touching)
        _, r = self.run_steps(swept(8), tracker=t)
        self.assertEqual(r.score, 0.0)
        self.assertIn("touched shard_d", r.details["failed"])

    def test_a_piece_off_the_table_latches_zero(self):
        t = S.Tracker()
        where = swept(7)
        where["shard_h"] = np.array([0.5, 0.0])
        st = states(where)
        st["shard_h"] = piece("shard_h", (0.9, 0.0), z=L.TOP - 0.3)
        t.update(st, AWAY)
        _, r = self.run_steps(swept(8), tracker=t)
        self.assertEqual(r.score, 0.0)
        self.assertIn("fell off", r.details["failed"])

    def test_unreadable_arms_latch_zero(self):
        t = S.Tracker()
        t.update(states({}), None)
        _, r = self.run_steps(swept(8), tracker=t)
        self.assertEqual(r.score, 0.0)

    def test_piece_beside_the_pan_does_not_count(self):
        where = swept(8)
        where["shard_a"] = np.array([L.PAN_LIP_X - 0.06, L.PAN_Y + 0.12])  # beside the side wall
        _, r = self.run_steps(where)
        self.assertEqual(r.details["in_pan"], 7)

    def test_the_brush_and_pan_grips_do_not_count_as_touching(self):
        """The demo's grips are far enough from the pieces' start and from the pan's floor."""
        for tcp in (T.pan_tcp(), T.brush_tcp(L.BRUSH_START, 0.0)):
            arms = {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.0), "right": AWAY["right"]}
            for where in ({}, swept(8)):
                _, r = self.run_steps(where, arms=arms, n=2)
                self.assertIsNone(r.details["failed"])


class TestDemo(unittest.TestCase):
    def test_left_holds_until_the_brush_is_away(self):
        p = T.plans()
        t_back = T._time_of(p["right"], "back over its spot")
        t_let_go = T._time_of(p["left"], "let go")
        self.assertGreater(t_let_go, t_back)

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertIn("right", T.plans(v))
        self.assertNotIn("left", T.plans("no_hold"))
        with self.assertRaises(ValueError):
            T.plans("x")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)
        self.assertLessEqual(len(meta["fidelity"]), 500)


if __name__ == "__main__":
    unittest.main()
