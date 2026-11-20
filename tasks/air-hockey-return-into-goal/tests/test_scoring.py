"""Offline tests for layout.py, scoring.py and task.py: synthetic puck and mallet states sampled at the 20 Hz control
rate and synthetic arm snapshots (rigstate.Arm); no simulator.

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
DT = 1.0 / L.CONTROL_HZ
PUCK_Z = L.TOP + L.SURFACE + L.PUCK_H / 2
MALLET_Z = L.TOP + L.SURFACE + L.MALLET_H / 2


def state(puck, vel, mallet, mallet_lift=0.0):
    return {S.PUCK: types.SimpleNamespace(origin=np.array([puck[0], puck[1], PUCK_Z]),
                                          lin_vel=np.array([vel[0], vel[1], 0.0])),
            S.MALLET: types.SimpleNamespace(origin=np.array([mallet[0], mallet[1], MALLET_Z + mallet_lift]))}


def holding(mallet, opening=0.26, other=AWAY["right"]):
    tcp = np.array([mallet[0], mallet[1], MALLET_Z + 0.03])
    return {"left": rigstate.Arm(tcp, tcp - [0, 0, 0.013], opening), "right": other}


def rally(out_vel, contact=(0.36, 0.08), in_speed=0.8, after_s=1.2, mallet_gap=0.005, lift=0.0, arms=None,
          strike=0.5, opening=0.26):
    """The puck comes down at ``in_speed``, meets the mallet at ``contact`` and leaves at ``out_vel``."""
    t = S.Tracker()
    c = np.array(contact)
    d = np.array(out_vel) / (np.linalg.norm(out_vel) or 1.0)
    mallet = c - d * (L.PUCK_R + L.MALLET_R + mallet_gap)
    r = None
    hands = (lambda m: arms) if arms is not None else (lambda m: holding(m, opening))
    for k in range(10, 0, -1):  # the mallet swings in at ``strike`` m/s
        m = mallet - d * strike * DT * k
        r = t.update(state(c + [in_speed * DT * k, 0.0], (-in_speed, 0.0), m), hands(m))
    p = c.copy()
    for _ in range(int(after_s / DT)):
        r = t.update(state(p, out_vel, mallet, lift), hands(mallet))
        p = p + np.array(out_vel) * DT
        mallet = mallet + d * 0.001
    return r


def toward_goal(speed=1.2, contact=(0.36, 0.08)):
    d = np.array([L.X1 + 0.05 - contact[0], 0.0 - contact[1]])
    return d / np.linalg.norm(d) * speed


class TestScoring(unittest.TestCase):
    def test_a_hit_into_the_far_goal_is_full(self):
        r = rally(toward_goal())
        self.assertTrue(r.details["won"])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_a_return_that_misses_the_goal_is_half(self):
        r = rally((1.0, 0.0))  # straight back up the left: into the end rail beside the goal
        self.assertTrue(r.details["returned"])
        self.assertFalse(r.details["won"])
        self.assertEqual(r.score, 0.5)

    def test_a_weak_hit_is_a_quarter(self):
        self.assertEqual(rally((0.2, 0.0), after_s=0.5).score, 0.25)

    def test_a_mallet_not_held_is_not_a_hit(self):
        self.assertFalse(rally(toward_goal(), arms=AWAY).details["hit"])
        self.assertFalse(rally(toward_goal(), opening=0.95).details["hit"])  # pushed with open fingers

    def test_a_rebound_off_the_near_rail_is_not_a_hit(self):
        r = rally(toward_goal(), contact=(L.X0 + L.PUCK_R + 0.002, 0.08))
        self.assertFalse(r.details["hit"])

    def test_a_mallet_held_still_in_the_way_is_not_a_hit(self):
        r = rally(toward_goal(), strike=0.0)
        self.assertFalse(r.details["hit"])
        self.assertEqual(r.score, 0.0)

    def test_the_puck_turning_away_from_the_mallet_is_not_a_hit(self):
        r = rally(toward_goal(), mallet_gap=0.08)
        self.assertFalse(r.details["hit"])

    def test_a_hand_on_the_puck_scores_nothing(self):
        p = np.array([0.36, 0.08, PUCK_Z])
        arms = holding(np.array([0.30, 0.08]), other=rigstate.Arm(p + [0, 0, 0.02], p + [0, 0, 0.007], 0.0))
        t = S.Tracker()
        for k in range(10, 0, -1):
            r = t.update(state((0.36 + 0.04 * k, 0.08), (-0.8, 0.0), (0.30, 0.08)), arms)
        self.assertTrue(r.details["hand"])
        self.assertEqual(r.score, 0.0)

    def test_conceding_scores_nothing(self):
        t = S.Tracker()
        r = None
        for k in range(30):
            r = t.update(state((0.40 - 0.02 * k, 0.0), (-0.4, 0.0), (0.26, 0.125)), AWAY)
        self.assertTrue(r.details["conceded"])
        self.assertEqual(r.score, 0.0)

    def test_the_mallet_over_the_centre_line_scores_nothing(self):
        t = S.Tracker()
        r = t.update(state((0.70, 0.08), (0.0, 0.0), (L.CENTRE_X - L.MALLET_R + 0.002, 0.0)), AWAY)
        self.assertTrue(r.details["over"])
        r = rally(toward_goal())
        self.assertFalse(r.details["over"])

    def test_unreadable_state(self):
        r = S.Tracker().update({}, AWAY)
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_a_straight_block_misses_the_goal(self):
        self.assertGreater(abs(L.PUCK_START[1]) - L.PUCK_R, L.GOAL_HALF)

    def test_the_hand_on_the_mallet_is_clear_of_the_puck_at_contact(self):
        gap = L.PUCK_R + L.MALLET_R  # TCP over the mallet's centre, the puck touching its rim
        self.assertGreater(gap, L.PUCK_R + L.HAND_CLEAR)


class TestDemo(unittest.TestCase):
    def test_variants(self):
        for v in ("", *T.VARIANTS):
            self.assertIn("left", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
