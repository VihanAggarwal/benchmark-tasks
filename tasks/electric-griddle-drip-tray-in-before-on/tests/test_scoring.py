"""Offline tests for layout.py, scoring.py, offline_model.py and task.py: synthetic griddle states (mechkit
AssemblyStates from the task's own spec) and synthetic arm snapshots, no simulator.
    python3 -m pytest tests -q
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import pathlib
import sys
import unittest

sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
for p in HERE.parents:
    d = p / "tools" / "offline"
    if (d / "yam_stub" / "yam_arena").is_dir():
        for _p in (d, d / "yam_stub"):
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
        break

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import offline_model as OM  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable")

KIT_DIR = HERE.parents[1] / "tools" / "kit"


def gstate(travel=0.0, dial=0.0):
    origin, quat = T.GRIDDLE_PART.spawn_pose()
    return {S.GRIDDLE: mk.state_from_poses(T.GRIDDLE_PART, (*origin, *quat),
                                           joints={S.TRAY_JOINT: travel, S.DIAL_JOINT: math.radians(dial)},
                                           source="test")}


class ReadTests(unittest.TestCase):
    def test_reads_travel_and_dial(self):
        for travel in (0.0, 0.1, L.TRAVEL):
            for dial in (0.0, 45.0, 90.0, 120.0):
                r = S.read(gstate(travel, dial))
                self.assertAlmostEqual(r.travel, travel, places=6)
                self.assertAlmostEqual(r.dial, dial, places=4)

    def test_bad_states(self):
        self.assertIsNone(S.read(None))
        self.assertIsNone(S.read({}))
        self.assertIsNone(S.read({S.GRIDDLE: object()}))


class TierTests(unittest.TestCase):
    def test_ladder(self):
        self.assertEqual(S.tier(True, True, True, True, True, L.TRAVEL), 0.0)
        self.assertEqual(S.tier(False, True, True, True, True, L.TRAVEL), 1.0)
        self.assertEqual(S.tier(False, True, True, True, False, L.TRAVEL), 0.75)
        self.assertEqual(S.tier(False, True, False, False, True, L.TRAVEL), 0.5)
        self.assertEqual(S.tier(False, True, True, False, True, L.TRAVEL), 0.5)
        self.assertEqual(S.tier(False, False, False, False, True, 0.15), 0.25)
        self.assertEqual(S.tier(False, False, False, False, True, 0.05), 0.0)

    def test_thresholds(self):
        self.assertTrue(S.is_seated(L.TRAVEL - 0.0099))
        self.assertFalse(S.is_seated(L.TRAVEL - 0.0101))
        self.assertFalse(S.is_on(14.9))
        self.assertTrue(S.is_on(15.0))
        self.assertTrue(S.is_set(76.0) and S.is_set(104.0))
        self.assertFalse(S.is_set(74.0) or S.is_set(106.0))


def feed(tracker, seq):
    res = None
    for travel, dial in seq:
        res = tracker.update(gstate(travel, dial), None)
    return res


class TrackerTests(unittest.TestCase):
    def test_on_before_seated_latches(self):
        tr = S.Tracker()
        seq = [(0.0, a) for a in range(0, 91, 5)] + [(t / 100, 90.0) for t in range(0, round(L.TRAVEL * 100) + 1)] + [(L.TRAVEL, 90.0)] * 50
        res = feed(tr, seq)
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(tr.unsafe_at)

    def test_tray_first_then_dial(self):
        tr = S.Tracker()
        seq = [(t / 100, 0.0) for t in range(0, round(L.TRAVEL * 100) + 1)] + [(L.TRAVEL, float(a)) for a in range(0, 91, 5)]
        seq += [(L.TRAVEL, 90.0)] * 40
        res = feed(tr, seq)
        self.assertIsNone(tr.unsafe_at)
        self.assertEqual(res.score, 0.75)  # no arms readable: never "released"

    def test_partial(self):
        self.assertEqual(feed(S.Tracker(), [(0.15, 0.0)] * 5).score, 0.25)
        self.assertEqual(feed(S.Tracker(), [(L.TRAVEL, 0.0)] * 5).score, 0.5)
        self.assertEqual(feed(S.Tracker(), [(0.0, 0.0)] * 5).score, 0.0)

    def test_teleport_resets(self):
        tr = S.Tracker()
        feed(tr, [(0.0, 60.0)])
        self.assertIsNotNone(tr.unsafe_at)
        feed(tr, [(0.0, 0.0)])
        self.assertIsNone(tr.unsafe_at)
        self.assertEqual(tr.resets, 1)


class DemoTests(unittest.TestCase):
    def test_variants_and_plans(self):
        p = T.plans("")
        self.assertEqual(set(p), {"left", "right"})
        self.assertGreaterEqual(sum(w.duration for w in p["left"]), sum(w.duration for w in p["right"]))
        for v in T.VARIANTS:
            T.plans(v)
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_push_overshoots_stop(self):
        ys = [w.pos[1] for w in T._tray_moves()]
        self.assertGreater(max(ys), T.TAB_Y_SEATED + 0.015)

    def test_dial_turn_in_set_window(self):
        self.assertTrue(S.is_set(math.degrees(T.DIAL_TURN)))

    def test_commanded_yaw(self):
        plan = T._dial_moves()
        self.assertIsNone(OM.commanded_yaw(plan, 0.5))
        self.assertAlmostEqual(OM.commanded_yaw(plan, 1e6), T.YAW_DIAL_SET)


class FileTests(unittest.TestCase):
    def test_kit_copies(self):
        for name in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / name).exists():
                self.assertEqual((HERE / name).read_bytes(), (KIT_DIR / name).read_bytes(), name)

    def test_task_json(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        for k in ("title", "context", "setup", "scoring", "fidelity", "reset_notes"):
            self.assertNotIn("—", d[k])
            self.assertNotIn("–", d[k])
        self.assertNotIn("drip", d["instruction"].lower())


if __name__ == "__main__":
    unittest.main()
