"""Offline tests for layout.py, scoring.py and task.py: synthetic paddle and ball states (an analytic bouncing ball,
sampled at the 20 Hz control rate) and synthetic arm snapshots (rigstate.Arm); no simulator.

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

G = 9.81
DT = 1.0 / L.CONTROL_HZ
AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
HELD = L.TOP + 0.140  # the blade's centre, held up
YAW_Q = (0.0, 0.0, math.sin(L.PADDLE_YAW / 2), math.cos(L.PADDLE_YAW / 2))


def paddle(z=HELD, vz=0.0):
    return mk.state_from_poses(T.PADDLE_SPEC, (0.36, 0.0, z, *YAW_Q), np.array([0.0, 0.0, vz, 0.0, 0.0, 0.0]))


def ball(gap, vel, z=HELD, dx=0.0):
    p = np.array([0.36 + dx, 0.0, z + L.BLADE_T / 2 + L.BALL_R + gap])
    return ObjectState(p, np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, vel]), np.zeros(3),
                       p - L.BALL_R, p + L.BALL_R)


def flight_samples(drop, rises, settle=0.4):
    """(gap, vel) every control step: the ball falls from ``drop`` above the face, then leaves each touch rising to
    the next height in ``rises``; after the last it lies still on the face."""
    gap, vel, t, out, k, h = drop, 0.0, 0.0, [], 0, 1e-4
    left = list(rises)
    end = None
    while end is None or t < end:
        vel -= G * h
        gap += vel * h
        if gap <= 0.0 and vel < 0.0:
            gap = 0.0
            if left:
                vel = math.sqrt(2 * G * left.pop(0))
            else:
                vel = 0.0
                end = t + settle if end is None else end
        t += h
        if t >= k * DT:
            out.append((gap, vel))
            k += 1
    return out


def held_by_hand(gap):
    """A hand holding the ball ``gap`` above the face (the drop's start)."""
    c = np.array([0.36, 0.0, HELD + L.BLADE_T / 2 + L.BALL_R + gap])
    return {"left": rigstate.Arm(c + [0, 0, 0.008], c, 0.4), "right": AWAY["right"]}


def play(drop, rises, z=HELD, dx=0.0, hold_steps=4, extra=None):
    t = S.Tracker()
    r = None
    for _ in range(hold_steps):
        r = t.update({S.PADDLE: paddle(z), S.BALL: ball(drop, 0.0, z, dx)}, held_by_hand(drop))
    for i, (gap, vel) in enumerate(flight_samples(drop, rises)):
        arms = extra(i, gap) if extra else AWAY
        r = t.update({S.PADDLE: paddle(z), S.BALL: ball(gap, vel, z, dx)}, arms)
    return t, r


class TestScoring(unittest.TestCase):
    def test_ten_bounces_in_a_row_is_full(self):
        _, r = play(0.12, [0.15] * 10)
        self.assertEqual(r.details["best"], 10)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_partial_runs(self):
        for n, want in ((9, 0.5), (6, 0.5), (5, 0.25), (4, 0.25), (3, 0.0), (1, 0.0), (0, 0.0)):
            with self.subTest(n=n):
                _, r = play(0.12, [0.15] * n)
                self.assertEqual(r.details["best"], n)
                self.assertEqual(r.score, want)

    def test_low_bounces_do_not_count(self):
        _, r = play(0.12, [0.08] * 12)
        self.assertEqual(r.details["best"], 0)
        self.assertEqual(r.score, 0.0)

    def test_a_low_bounce_breaks_the_run(self):
        _, r = play(0.12, [0.15] * 4 + [0.06] + [0.15] * 4)
        self.assertEqual(r.details["best"], 4)
        self.assertEqual(r.score, 0.25)

    def test_bounces_need_a_drop_from_a_hand_first(self):
        _, r = play(0.12, [0.15] * 10, hold_steps=0)  # the ball was never held: launched off the face, say
        self.assertEqual(r.details["best"], 0)
        self.assertFalse(r.details["from_hand"])

    def test_a_flight_over_250_mm_breaks_the_run(self):
        _, r = play(0.12, [0.15] * 3 + [0.30] + [0.15] * 3)
        self.assertEqual(r.details["best"], 3)
        self.assertEqual(r.score, 0.0)

    def test_one_high_tap_then_a_still_paddle_reaches_no_tier(self):
        e2 = L.RESTITUTION ** 2
        _, r = play(0.12, [L.MAX_RISE * e2 ** k for k in range(8)])
        self.assertLess(r.details["best"], L.TIERS[-1][0])
        self.assertEqual(r.score, 0.0)

    def test_a_still_paddle_under_the_highest_allowed_drop_scores_nothing(self):
        rises = [L.MAX_DROP * (L.RESTITUTION ** 2) ** k for k in range(1, 8)]
        _, r = play(L.MAX_DROP, rises)
        self.assertLessEqual(r.details["best"], 1)
        self.assertEqual(r.score, 0.0)

    def test_bounces_after_a_drop_from_too_high_do_not_count(self):
        _, r = play(0.25, [0.18, 0.15, 0.15, 0.15])
        self.assertTrue(r.details["high_drop"])
        self.assertEqual(r.details["best"], 0)

    def test_a_high_drop_counts_from_the_let_go_point_even_while_the_fingers_open(self):
        def fingers(i, gap):  # the opening pads stay 25 mm from the ball's surface for the first 0.15 s of the fall
            if i < 3:
                c = np.array([0.36, 0.0, HELD + L.BLADE_T / 2 + L.BALL_R + gap])
                return {"left": rigstate.Arm(c + [0, 0, 0.040], c + [0, 0.045, 0.0], 0.8), "right": AWAY["right"]}
            return AWAY

        _, r = play(0.25, [0.15] * 10, extra=fingers)
        self.assertTrue(r.details["high_drop"])
        self.assertEqual(r.details["best"], 0)

    def test_the_paddle_must_be_held_up_off_its_block(self):
        _, r = play(0.12, [0.15] * 10, z=L.TOP + L.BLOCK_SIZE[2] + L.BLADE_T / 2)
        self.assertEqual(r.details["best"], 0)
        self.assertEqual(r.score, 0.0)

    def test_touches_off_the_blade_do_not_count(self):
        _, r = play(0.12, [0.15] * 10, dx=L.BLADE_R + 0.01)
        self.assertEqual(r.details["best"], 0)

    def test_a_hand_touching_the_ball_restarts_the_run(self):
        n = len(flight_samples(0.12, [0.15] * 4))

        def poke(i, gap):  # a hand at the ball once, just after the fourth bounce's touch
            return held_by_hand(gap) if i == n - 6 else AWAY

        _, r = play(0.12, [0.15] * 9, extra=poke)
        self.assertLess(r.details["best"], 5)  # the flight after the hand is a new drop: nine became four and four
        self.assertEqual(r.score, 0.25)

    def test_bounces_under_the_blade_do_not_count(self):
        below = -(L.BLADE_T + 2 * L.BALL_R + 0.01)  # the ball's bottom this far "above" the face: under the blade
        t = S.Tracker()
        t.update({S.PADDLE: paddle(), S.BALL: ball(0.05, 0.0)}, held_by_hand(0.05))
        r = None
        for _ in range(3):
            for gap, vel in flight_samples(0.12, [0.15])[:-1]:
                r = t.update({S.PADDLE: paddle(), S.BALL: ball(below + gap, vel)}, AWAY)
        self.assertEqual(r.details["best"], 0)
        self.assertNotEqual(r.details["last_touch"], "face")

    def test_the_ball_landing_on_the_table_ends_the_run_and_needs_a_new_drop(self):
        t, _ = play(0.12, [0.15] * 3)
        table_gap = L.TOP - (HELD + L.BLADE_T / 2)  # the ball's bottom on the table, "above" the face
        r = None
        for gap, vel in ((table_gap + 0.02, -2.0), (table_gap, 0.0)):
            r = t.update({S.PADDLE: paddle(), S.BALL: ball(gap, vel, dx=0.20)}, AWAY)
        self.assertEqual(r.details["last_touch"], "off the face")
        self.assertFalse(r.details["from_hand"])
        for gap, vel in flight_samples(0.12, [0.15] * 6):  # bounced on the face again without a new drop
            r = t.update({S.PADDLE: paddle(), S.BALL: ball(gap, vel)}, AWAY)
        self.assertEqual(r.details["best"], 3)

    def test_a_paddle_snapped_away_under_the_ball_is_not_a_touch(self):
        t = S.Tracker()
        t.update({S.PADDLE: paddle(), S.BALL: ball(0.05, 0.0)}, held_by_hand(0.05))
        r = None
        for k, (gap, vb, vp) in enumerate([(0.05, -0.5, 0.0), (0.03, -0.99, -0.5), (0.03, -1.48, -2.5),
                                           (0.04, -1.97, -3.0), (0.05, -2.46, -3.5)]):
            r = t.update({S.PADDLE: paddle(HELD - 0.02 * k, vp), S.BALL: ball(gap, vb, HELD - 0.02 * k)}, AWAY)
        self.assertNotEqual(r.details["last_touch"], "face")
        self.assertTrue(r.details["dropping"])

    def test_unreadable_state(self):
        r = S.Tracker().update({S.PADDLE: paddle()}, AWAY)
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_the_ball_sits_in_its_stand(self):
        inner = L.STAND_OUTER - 2 * L.STAND_WALL
        self.assertLess(inner, 2 * L.BALL_R)
        self.assertGreater(L.BALL_START[2] - L.BALL_R, L.TOP + L.STAND_WALL)

    def test_the_paddle_settles_onto_its_block(self):
        self.assertAlmostEqual(L.PADDLE_REST[2] - L.BLADE_T / 2, L.TOP + L.BLOCK_SIZE[2], places=9)
        self.assertLess(L.PADDLE_REST[2], L.LIFTED + L.TOP)

    def test_passive_bounces_die_below_the_count_height(self):
        self.assertLess(L.MAX_DROP * L.RESTITUTION ** 4, L.MIN_RISE)
        self.assertLess(L.MAX_RISE * L.RESTITUTION ** 6, L.MIN_RISE)  # at most 3 in a row after the highest flight
        self.assertGreater(L.TIERS[-1][0], 3)


class TestDemo(unittest.TestCase):
    def test_the_drop_is_over_the_blade_and_low_enough(self):
        c = T.drop_centre()
        gap = c[2] - T.HOLD[2] - L.BLADE_T / 2 - L.BALL_R
        self.assertLessEqual(gap, L.MAX_DROP)
        self.assertAlmostEqual(float(np.hypot(*(c - T.HOLD)[:2])), 0.0, places=9)
        self.assertGreater(T.HOLD[2] - T.AMP, L.TOP + L.LIFTED)

    def test_the_paddle_is_held_level_in_a_pinch_on_the_handle(self):
        r = T._side_rotation()
        self.assertAlmostEqual(abs(r[2, 0]), 1.0, places=9)  # the pads close vertically, on the handle's flats
        self.assertAlmostEqual(r[2, 2], 0.0, places=9)  # the hand comes in level

    def test_variants(self):
        for v in T.VARIANTS:
            self.assertEqual(set(T.plans(v)), {"left", "right"})
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
