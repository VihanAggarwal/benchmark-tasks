"""Offline tests for layout.py, scoring.py and task.py: synthetic table and ball states and synthetic arm snapshots
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
B0 = np.array(L.ball_start())


def table(slide=0.0, turn=0.0):
    return mk.state_from_poses(T.TABLE_SPEC, (L.NEAR_X, L.CAB_CY, L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
                               joints={S.SLIDE: slide, S.TURN: turn})


def ball(p):
    p = np.asarray(p, dtype=float)
    return ObjectState(p, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), p - L.BALL_R, p + L.BALL_R)


def run(path, turns=None, arms=AWAY, tail=30, slide=0.0):
    t = S.Tracker()
    r = None
    turns = turns or [0.0] * len(path)
    for p, q in zip(path, turns):
        r = t.update({S.TABLE: table(slide, q), S.BALL: ball(p)}, arms)
    for _ in range(tail):
        r = t.update({S.TABLE: table(slide, turns[-1]), S.BALL: ball(path[-1])}, arms)
    return t, r


def shot(to_x, to_y, n=20):
    end = np.array([to_x, to_y, B0[2]])
    return [B0 + (end - B0) * k / n for k in range(n + 1)]


GOAL_Y = L.goal_line_y() - L.BALL_R - 0.02


class TestScoring(unittest.TestCase):
    def test_nothing_is_zero(self):
        _, r = run([B0])
        self.assertEqual(r.score, 0.0)

    def test_goal_and_let_go_is_full(self):
        _, r = run(shot(L.FIELD_CX + 0.038, GOAL_Y))
        self.assertTrue(r.details["goal"])
        self.assertEqual(r.score, 1.0)

    def test_goal_still_holding_is_three_quarters(self):
        h = T.tcp(0.0)
        holding = {"left": AWAY["left"], "right": rigstate.Arm(h, h + np.array([0.013, 0.0, 0.0]), 0.3)}
        _, r = run(shot(L.FIELD_CX + 0.038, GOAL_Y), arms=holding)
        self.assertEqual(r.score, 0.75)

    def test_on_target_miss_is_half(self):
        _, r = run(shot(L.FIELD_CX + L.GOAL_W / 2 + 0.05, L.goal_line_y() + L.BALL_R))
        self.assertFalse(r.details["goal"])
        self.assertEqual(r.score, 0.5)

    def test_wide_miss_with_progress_is_quarter(self):
        _, r = run(shot(L.FIELD_CX + 0.20, B0[1] - 0.20))
        self.assertEqual(r.score, 0.25)
        _, r = run(shot(L.FIELD_CX + 0.04, B0[1] - 0.10))
        self.assertEqual(r.score, 0.0)

    def test_spin_latches_zero(self):
        path = shot(L.FIELD_CX + 0.038, GOAL_Y)
        turns = [math.radians(-30.0) * k for k in range(len(path))]  # 600 deg one way, reported wrapped
        turns = [((q + math.pi) % (2 * math.pi)) - math.pi for q in turns]
        _, r = run(path, turns)
        self.assertIn("spun", r.details["failed"])
        self.assertEqual(r.score, 0.0)

    def test_back_and_forth_turns_are_not_a_spin(self):
        path = shot(L.FIELD_CX + 0.038, GOAL_Y)
        turns = [math.radians(60.0 * math.sin(k)) for k in range(len(path))]
        _, r = run(path, turns)
        self.assertIsNone(r.details["failed"])

    def test_hand_in_the_game_latches_zero(self):
        inside = np.array([L.FIELD_CX, L.rod_y(5) - 0.05, L.TOP + 0.08])
        arms = {"left": AWAY["left"], "right": rigstate.Arm(inside, inside - [0, 0, 0.013], 1.0)}
        t = S.Tracker()
        t.update({S.TABLE: table(), S.BALL: ball(B0)}, arms)
        _, r = run(shot(L.FIELD_CX + 0.038, GOAL_Y), tail=30)
        r = t.update({S.TABLE: table(), S.BALL: ball(shot(L.FIELD_CX + 0.038, GOAL_Y)[-1])}, AWAY)
        self.assertIn("playfield", r.details["failed"])
        self.assertEqual(r.score, 0.0)

    def test_own_goal_is_zero(self):
        _, r = run(shot(L.FIELD_CX, L.FIELD_Y1 + L.BALL_R + 0.02))
        self.assertEqual(r.score, 0.0)

    def test_unreadable_state(self):
        r = S.Tracker().update({S.TABLE: table()}, AWAY)
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_lane_past_the_goalie_and_inside_the_posts(self):
        x = L.BALL_OFFSET_X
        self.assertGreater(abs(x), L.FOOT[1] / 2 + L.BALL_R)  # clear of the goalie's foot
        self.assertLess(abs(x), L.GOAL_W / 2 - L.BALL_R)  # inside the posts
        for _, side, men, spacing in L.RODS[6:]:  # the opponent's defence and goalie in front of the goal
            for mx in L.men_x(men, spacing):
                self.assertGreater(abs(L.FIELD_CX + x - mx), L.FOOT[1] / 2 + L.BALL_R)

    def test_slide_reaches_the_ball(self):
        self.assertLessEqual(abs(L.BALL_OFFSET_X), L.SLIDE)

    def test_ball_rests_on_the_playfield_in_front_of_the_forward(self):
        self.assertAlmostEqual(B0[2], L.TOP + L.FLOOR_Z + L.BALL_R)
        self.assertLess(B0[1], L.rod_y(L.rod_index(L.SHOOTER)))


class TestDemo(unittest.TestCase):
    def test_demo_slides_to_the_ball_and_strikes_without_spinning(self):
        p = T.plans()["right"]
        slide = next(w for w in p if w.label.startswith("slide the rod"))
        self.assertAlmostEqual(slide.pos[0] - T.tcp(0.0)[0], L.BALL_OFFSET_X)
        self.assertLess(abs(T.WIND_DEG - T.STRIKE_DEG), 360.0)
        self.assertEqual(p[-1].finger, T.OPEN)

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertIn("right", T.plans(v))


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
