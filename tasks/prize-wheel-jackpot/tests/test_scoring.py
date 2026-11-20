"""Offline tests for layout.py, scoring.py and task.py: synthetic wheel turns (the spindle's joint, wrapped as the
simulator may report it) and synthetic arm snapshots (rigstate.Arm); no simulator.

    PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v
"""

import json
import math
import pathlib
import sys
import types
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
AT_PEG = {"left": AWAY["left"],
          "right": rigstate.Arm(T.push_tcp(), T.push_tcp() - [0, 0, 0.013], 0.0)}


def wheel(turn):
    wrapped = (turn + math.pi) % (2 * math.pi) - math.pi
    return {S.STAND: types.SimpleNamespace(joints={S.WHEEL: wrapped})}


def spin(total_deg, touch_deg=20.0, steps=200, rest=40):
    """Touched while turning to ``touch_deg``, then a free coast (decaying) to ``total_deg``, then still."""
    t = S.Tracker()
    r = None
    for k in range(10):
        r = t.update(wheel(math.radians(touch_deg * k / 9)), AT_PEG)
    a0, a1 = math.radians(touch_deg), math.radians(total_deg)
    for k in range(1, steps + 1):
        f = 1 - math.exp(-5.0 * k / steps)
        r = t.update(wheel(a0 + (a1 - a0) * f / (1 - math.exp(-5.0))), AWAY)
    for _ in range(rest):
        r = t.update(wheel(a1), AWAY)
    return r


class TestWheel(unittest.TestCase):
    def test_jackpot_needs_a_turn_and_a_quarter(self):
        self.assertEqual(L.segment_under_pointer(0.0), 0)
        self.assertEqual(L.segment_under_pointer(math.radians(450)), L.JACKPOT)
        self.assertEqual(L.segment_under_pointer(math.radians(90)), L.JACKPOT)

    def test_the_demo_peg_is_reached_from_outside_the_rim(self):
        tcp = T.push_tcp()
        self.assertGreater(math.hypot(tcp[1] - L.CENTRE[1], tcp[2] - L.TOP - L.CENTRE[2]), L.R)


class TestScoring(unittest.TestCase):
    def test_a_coast_onto_jackpot_is_full(self):
        r = spin(450.0)
        self.assertEqual(r.details["segment"], L.JACKPOT)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_neighbours_and_misses(self):
        self.assertEqual(spin(480.0).score, 0.5)
        self.assertEqual(spin(420.0).score, 0.5)
        self.assertEqual(spin(540.0).score, 0.25)

    def test_walking_it_round_to_jackpot_is_nothing(self):
        self.assertEqual(spin(90.0, touch_deg=80.0).score, 0.0)

    def test_less_than_a_full_free_turn_is_nothing(self):
        r = spin(450.0, touch_deg=120.0)  # the coast is 330 deg
        self.assertEqual(r.score, 0.0)

    def test_stopping_it_by_hand_counts_from_the_stop(self):
        t = S.Tracker()
        for k in range(10):
            t.update(wheel(math.radians(2.0 * k)), AT_PEG)
        for k in range(1, 100):
            t.update(wheel(math.radians(18.0 + 4.0 * k)), AWAY)
        for _ in range(30):  # caught and held at 450 deg by the hand
            r = t.update(wheel(math.radians(450.0)), AT_PEG)
        for _ in range(30):
            r = t.update(wheel(math.radians(450.0)), AWAY)
        self.assertEqual(r.score, 0.0)

    def test_not_at_rest_yet_scores_nothing(self):
        r = spin(450.0, rest=0)
        self.assertFalse(r.details["at_rest"])

    def test_unreadable_state(self):
        r = S.Tracker().update({}, AWAY)
        self.assertFalse(r.details["valid_state"])


class TestDemo(unittest.TestCase):
    def test_variants(self):
        for v in ("", *T.VARIANTS):
            self.assertIn("right", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
