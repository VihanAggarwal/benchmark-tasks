"""Offline tests for layout.py, scoring.py and task.py: synthetic machine and ball states and synthetic arm snapshots
(rigstate.Arm); no simulator.

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

AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
SWING = L.FLIPPER_UP - L.FLIPPER_REST


def machine(flip=0.0):
    return mk.state_from_poses(T.MACHINE_SPEC, (L.CAB_FRONT_X, L.CAB_CY, L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                               joints={S.RIGHT_FLIPPER: flip * SWING})


def ball(u, v, vel_u=0.0):
    p = L.world(L.pf_point(u, v, L.BALL_R))
    vel = np.array([vel_u * math.cos(L.SLOPE), 0.0, vel_u * math.sin(L.SLOPE)])
    return ObjectState(p, np.array([0.0, 0.0, 0.0, 1.0]), vel, np.zeros(3), p - L.BALL_R, p + L.BALL_R)


def play(path, arms=AWAY):
    """path: (u, v, vel_u, flip) per step."""
    t = S.Tracker()
    r = None
    for u, v, vu, f in path:
        r = t.update({S.MACHINE: machine(f), S.BALL: ball(u, v, vu)}, arms)
    return t, r


START = (L.PLUNGER_REST_U + L.BALL_R, L.LANE_V, 0.0, 0.0)
OUT = [(0.36, -0.10, -0.1, 0.0), (0.25, -0.09, -0.4, 0.0), (0.12, -0.05, -0.4, 0.0), (0.09, -0.03, -0.1, 0.0)]
SHOT = [(0.16, -0.06, 0.6, 1.0), (0.25, -0.08, 0.4, 1.0), (0.34, -0.07, 0.1, 1.0)]
DRAIN = [(0.05, 0.0, -0.3, 0.0), (0.02, 0.0, -0.2, 0.0)]


class TestScoring(unittest.TestCase):
    def test_resting_in_the_lane_is_zero(self):
        _, r = play([START] * 10)
        self.assertFalse(r.details["launched"])
        self.assertEqual(r.score, 0.0)

    def test_launched_then_drained_is_quarter(self):
        _, r = play([START] + OUT + DRAIN)
        self.assertTrue(r.details["launched"] and r.details["drained"])
        self.assertEqual(r.score, 0.25)

    def test_flipper_shot_into_the_top_lanes_is_full(self):
        _, r = play([START] + OUT + SHOT)
        self.assertTrue(r.details["flipped"] and r.details["top"])
        self.assertEqual(r.score, 1.0)

    def test_flipper_shot_short_of_the_top_is_half(self):
        _, r = play([START] + OUT + SHOT[:2])
        self.assertEqual(r.score, 0.5)

    def test_ball_rolling_up_without_a_flipper_is_not_a_shot(self):
        up = [(0.16, -0.06, 0.6, 0.0), (0.25, -0.08, 0.4, 0.0), (0.34, -0.07, 0.1, 0.0)]
        _, r = play([START] + OUT + up)
        self.assertFalse(r.details["flipped"])
        self.assertEqual(r.score, 0.25)

    def test_a_fast_launch_with_a_flipper_held_up_is_not_a_shot(self):
        lane = [(u, L.LANE_V, 1.5, 1.0) for u in (0.06, 0.12, 0.2, 0.3)]  # up the lane, the right flipper fired
        top = [(0.36, -0.12, 0.4, 1.0), (0.40, -0.10, 0.1, 1.0)]  # round the top, past the top lanes' line
        _, r = play([START] + lane + top)
        self.assertTrue(r.details["launched"])
        self.assertFalse(r.details["flipped"])
        self.assertEqual(r.score, 0.25)

    def test_a_raised_flipper_far_from_the_ball_is_not_a_shot(self):
        far = [(0.16, 0.10, 0.6, 1.0), (0.25, 0.10, 0.4, 1.0), (0.34, 0.10, 0.1, 1.0)]  # rolling up the left side
        out_left = [(0.36, 0.10, -0.1, 0.0), (0.25, 0.12, -0.4, 0.0)]
        _, r = play([START] + out_left + far)
        self.assertFalse(r.details["flipped"])

    def test_a_rebound_off_a_flipper_held_up_is_not_a_shot(self):
        held = [(u, v, vu, 1.0) for u, v, vu, _ in OUT]  # the right flipper held up the whole way down
        bounce = [(0.14, -0.05, 0.5, 1.0), (0.22, -0.06, 0.4, 1.0), (0.34, -0.07, 0.1, 1.0)]
        _, r = play([START] + [(0.36, -0.10, 0.0, 1.0)] * 10 + held + bounce)
        self.assertFalse(r.details["flipped"])
        self.assertEqual(r.score, 0.25)

    def test_nothing_counts_after_a_drain(self):
        _, r = play([START] + OUT + DRAIN + SHOT)
        self.assertEqual(r.score, 0.25)

    def test_a_hand_on_the_playfield_is_zero(self):
        over = L.world(L.pf_point(0.2, 0.0, 0.03))
        hand = {"left": AWAY["left"], "right": rigstate.Arm(over, over - [0, 0, 0.013], 0.0)}
        t = S.Tracker()
        t.update({S.MACHINE: machine(), S.BALL: ball(*START[:2])}, hand)
        r = None
        for u, v, vu, f in OUT + SHOT:
            r = t.update({S.MACHINE: machine(f), S.BALL: ball(u, v, vu)}, AWAY)
        self.assertIn("playfield", r.details["failed"])
        self.assertEqual(r.score, 0.0)

    def test_unreadable_state(self):
        r = S.Tracker().update({S.MACHINE: machine()}, AWAY)
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_ball_starts_in_the_lane_against_the_plunger(self):
        u, v, h = L.to_pf(L.ball_start())
        self.assertAlmostEqual(v, L.LANE_V, places=6)
        self.assertAlmostEqual(h, L.BALL_R, places=6)
        self.assertAlmostEqual(u, L.PLUNGER_REST_U + L.BALL_R, places=6)

    def test_drain_gap_lets_the_ball_through(self):
        gap = 2 * (L.FLIPPER_V - L.FLIPPER_L * math.cos(-L.FLIPPER_REST))
        self.assertGreater(gap, 2 * L.BALL_R)

    def test_the_hands_work_outside_the_playfield(self):
        for p in (T.knob_tcp(), T.knob_tcp(T.PULL), T.button_tcp("right"), T.button_tcp("right", L.BUTTON_TRAVEL)):
            self.assertFalse(S.over_playfield(p))


class TestDemo(unittest.TestCase):
    def test_demo_presses_the_button_all_the_way_after_launch(self):
        p = T.plans()["right"]
        press = next(w for w in p if w.label.startswith("press"))
        self.assertLess(press.pos[2], T.button_tcp("right", L.BUTTON_TRAVEL)[2])
        t, times = 0.0, {}
        for w in p:
            t += w.duration
            times.setdefault(w.label, t)
        self.assertGreater(times["press: the flipper strikes"], times["let go: the plunger launches the ball"] + 1.5)
        self.assertEqual(p[-1].finger, T.OPEN)

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertIn("right", T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
