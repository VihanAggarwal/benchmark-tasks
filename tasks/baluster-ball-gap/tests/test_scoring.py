"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, down_on_guard, evaluate, lowest, off_table, openings, outline, rotation, stands, tilt_deg, valid,
)

MM = 0.001
R, H = L.BAL_R, L.BAL_H


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A baluster's state: its world bounding box worked out from the turned cylinder."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = (H / 2) * np.abs(a) + R * np.sqrt(np.clip(1 - a * a, 0, 1))
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def bal(y, x=L.RAIL_X, bottom=L.RAIL_TOP, tilt=0.0, axis="x", **kw):
    """A baluster at (x, y) with its lowest point at ``bottom``, tilted ``tilt`` degrees about the world ``axis``
    (90: lying down)."""
    t = math.radians(tilt)
    z = bottom + (H / 2) * abs(math.cos(t)) + R * abs(math.sin(t))
    return state((x, y, z), about(axis, tilt), **kw)


def start():
    """The starting layout: every baluster upright on the table at its mark (order of L.NAMES)."""
    return [bal(y, x, L.TOP) for _, x, y in L.BALUSTERS]


S0 = start()


def guard(ys, xs=None, rest=None):
    """Balusters standing on the rail at ``ys`` (and ``xs``), the rest of the six left at their starting marks
    (or given in ``rest``)."""
    xs = [L.RAIL_X] * len(ys) if xs is None else xs
    on = [bal(y, x) for x, y in zip(xs, ys, strict=True)]
    rest = S0[len(ys):] if rest is None else rest
    return on + list(rest)


def ys_from_openings(gaps):
    """The y of balusters on the centre line that leave ``gaps`` (mm, left post to right post; they must add up to
    356 - 20 n)."""
    assert abs(sum(gaps) - (356 - 20 * (len(gaps) - 1))) < 1e-6, sum(gaps)
    out, y = [], L.POST_FACE
    for g in gaps[:-1]:
        y -= g * MM + R
        out.append(y)
        y -= R
    return out


FIVE = list(L.even_centres(5))
SIX = list(L.even_centres(6))


def score_of(bals):
    return evaluate(bals)[0]


def glide(a, b, n):
    """n frames moving every baluster of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None):
    """From the start, every baluster moved gradually to the guard (or ``final``), then held there."""
    final = guard(FIVE) if final is None else final
    return [S0] * 5 + glide(S0, final, n) + [final] * 10


def run(frames, hz=20.0):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


class Geometry(unittest.TestCase):
    def test_tilt_and_lowest_point(self):
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=6)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("x", 7)), 7.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)  # upside down: either end may be down
        self.assertAlmostEqual(tilt_deg(about("y", 90)), 90.0, places=4)
        for tilt in (0, 3, 30, 60, 90, 120, 180):
            s = bal(0.0, bottom=L.RAIL_TOP + 0.004, tilt=tilt)
            self.assertAlmostEqual(lowest(s), L.RAIL_TOP + 0.004, places=9, msg=tilt)
            self.assertAlmostEqual(lowest(s), s.low[2], places=9)  # agrees with the bounding box
        self.assertAlmostEqual(lowest(S0[0]), L.TOP, places=9)

    def test_outline(self):
        up = outline(bal(0.05))
        self.assertTrue(np.allclose(np.hypot(up[:, 0] - L.RAIL_X, up[:, 1] - 0.05), R))
        flat = outline(bal(0.0, tilt=90, axis="x"))  # lying along y
        self.assertAlmostEqual(flat[:, 1].max() - flat[:, 1].min(), H, places=6)
        self.assertAlmostEqual(flat[:, 0].max() - flat[:, 0].min(), 2 * R, places=6)

    def test_stands_tolerances(self):
        self.assertTrue(stands(bal(0.0)))
        self.assertTrue(stands(bal(0.0, tilt=4.9)))
        self.assertFalse(stands(bal(0.0, tilt=5.1)))
        self.assertTrue(stands(bal(0.0, bottom=L.RAIL_TOP + 0.0019)))
        self.assertFalse(stands(bal(0.0, bottom=L.RAIL_TOP + 0.0021)))  # held just above the rail
        self.assertTrue(stands(bal(0.0, bottom=L.RAIL_TOP - 0.0019)))
        self.assertFalse(stands(bal(0.0, bottom=L.RAIL_TOP - 0.0021)))
        self.assertTrue(stands(bal(0.0, x=L.RAIL_X + 0.0199)))  # over the rail's top, overhanging its edge
        self.assertFalse(stands(bal(0.0, x=L.RAIL_X - 0.0201)))
        self.assertFalse(stands(bal(0.0, bottom=L.TOP)))  # on the table
        self.assertFalse(stands(bal(L.POST_FACE + 0.001)))
        self.assertFalse(stands(S0[0]))

    def test_openings(self):
        self.assertEqual(openings([]), [L.RAIL_LEN])
        ops = openings([bal(y) for y in FIVE])
        self.assertEqual(len(ops), 6)
        self.assertTrue(np.allclose(ops, (356 - 100) / 6 * MM))
        # order does not matter; a neighbour's opening is the clear distance between centres less a diameter
        ops = openings([bal(0.0, x=L.RAIL_X + 0.01), bal(0.05, x=L.RAIL_X - 0.01)])
        self.assertAlmostEqual(ops[0], L.POST_FACE - 0.05 - R)
        self.assertAlmostEqual(ops[1], math.hypot(0.02, 0.05) - 2 * R)
        self.assertAlmostEqual(ops[2], L.POST_FACE - R)

    def test_even_centres_arithmetic(self):
        for n, want in ((4, 55.2), (5, 42.7), (6, 33.7)):
            ops = openings([bal(y) for y in L.even_centres(n)])
            self.assertTrue(np.allclose(ops, ops[0]))
            self.assertAlmostEqual(ops[0] / MM, want, places=1)
        self.assertTrue(np.allclose(L.DEMO_YS, (0.125333, 0.062667, 0.0, -0.062667, -0.125333), atol=1e-6))

    def test_valid(self):
        self.assertTrue(valid(bal(0.0)))
        self.assertFalse(valid(bal(float("nan"))))
        self.assertFalse(valid(state((0.3, 0, 0.8), quat=(0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        self.assertFalse(valid(bal(0.0, vel=(0, float("inf"), 0))))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["openings_mm"], [356.0])
        self.assertEqual(d["passing"], 1)

    def test_five_or_six_even_score_one(self):
        self.assertEqual(score_of(guard(FIVE)), 1.0)
        self.assertEqual(score_of(guard(SIX)), 1.0)
        self.assertEqual(score_of(guard(FIVE[::-1], rest=S0[:1])), 1.0)  # any baluster in any place
        upside_down = [bal(y, tilt=180) for y in FIVE] + S0[5:]
        self.assertEqual(score_of(upside_down), 1.0)  # either end down

    def test_four_can_never_pass(self):
        self.assertEqual(score_of(guard(list(L.even_centres(4)))), 0.0)  # five openings of 55.2 mm
        _, d = evaluate(guard(list(L.even_centres(4))))
        self.assertEqual(d["passing"], 5)
        rng = np.random.default_rng(3)
        for _ in range(300):  # however four are spread, at least one opening passes
            w = rng.dirichlet(np.ones(5)) * 276
            score, d = evaluate(guard(ys_from_openings(list(w))))
            self.assertGreaterEqual(d["passing"], 1)
            self.assertIn(score, (0.0, 0.25))

    def test_four_with_one_wide_gap_scores_a_quarter(self):
        self.assertEqual(score_of(guard(ys_from_openings([48, 48, 48, 48, 84]))), 0.25)
        self.assertEqual(score_of(guard(ys_from_openings([84, 48, 48, 48, 48]))), 0.25)  # at a post
        self.assertEqual(score_of(guard(ys_from_openings([40, 40, 60, 40, 96]))), 0.0)  # two pass

    def test_uneven_without_a_passing_gap_scores_half(self):
        self.assertEqual(score_of(guard(ys_from_openings([35, 50, 35, 50, 36, 50]))), 0.5)
        self.assertEqual(score_of(guard(ys_from_openings([30, 40, 30, 40, 30, 40, 26]))), 0.5)  # six

    def test_spread_limit(self):
        # five: openings 38.0 + 48.0 (+ four of 42.5): spread 10.0 mm -> 1.0; 10.2 mm -> 0.5
        self.assertEqual(score_of(guard(ys_from_openings([38.0, 48.0, 42.5, 42.5, 42.5, 42.5]))), 1.0)
        self.assertEqual(score_of(guard(ys_from_openings([37.9, 48.1, 42.5, 42.5, 42.5, 42.5]))), 0.5)

    def test_ball_limit(self):
        # five with one opening of 50.9 mm passes nothing; 51.0 passes the ball
        self.assertEqual(score_of(guard(ys_from_openings([50.9, 40.0, 42.0, 41.0, 41.0, 41.1]))), 0.5)
        _, d = evaluate(guard(ys_from_openings([51.0, 41.0, 41.0, 41.0, 41.0, 41.0])))
        self.assertEqual(d["passing"], 1)
        self.assertEqual(score_of(guard(ys_from_openings([51.0, 41.0, 41.0, 41.0, 41.0, 41.0]))), 0.25)
        self.assertEqual(score_of(guard(ys_from_openings([56, 40, 40, 40, 40, 40]))), 0.25)  # five, one passing
        self.assertEqual(score_of(guard(ys_from_openings([66, 30, 30, 30, 30, 30, 20]))), 0.25)  # six

    def test_fewer_than_four_score_zero(self):
        for n in (1, 2, 3):
            self.assertEqual(score_of(guard(list(L.even_centres(n)))), 0.0, n)
        # three bunched at one end: exactly one opening passes, but fewer than four stand
        three = guard(ys_from_openings([45, 45, 45, 161]))
        _, d = evaluate(three)
        self.assertEqual(d["passing"], 1)
        self.assertEqual(score_of(three), 0.0)
        self.assertEqual(score_of(guard([0.10])), 0.0)  # one opening of 68 mm and one of 268 passes twice

    def test_balusters_elsewhere_are_ignored(self):
        on_post = bal(L.POST_Y, x=L.RAIL_X, bottom=L.TOP + L.POST_SIZE[2])  # standing on top of a post
        in_air = bal(0.0, x=0.20, bottom=L.TOP + 0.05)  # held in the air
        self.assertEqual(score_of(guard(FIVE, rest=[on_post])), 1.0)
        self.assertEqual(score_of(guard(FIVE, rest=[in_air])), 1.0)
        knocked = bal(-0.27, x=0.20, bottom=L.TOP - 0.30, tilt=90, axis="y", vel=(0, 0, -2))
        self.assertTrue(off_table(knocked))
        self.assertEqual(score_of(guard(FIVE, rest=[knocked])), 1.0)
        lying_on_table = bal(-0.30, x=0.20, bottom=L.TOP, tilt=90, axis="y")
        self.assertEqual(score_of(guard(FIVE, rest=[lying_on_table])), 1.0)

    def test_a_baluster_down_on_the_guard_scores_zero(self):
        cases = {
            "lying across the rail": bal(0.0, x=0.30, bottom=L.RAIL_TOP, tilt=90, axis="y"),  # along x, across it
            "lying along the rail's top": bal(0.15, x=0.30, bottom=L.RAIL_TOP, tilt=90, axis="x"),
            "leaning against a post, bottom on the rail": bal(0.155, bottom=L.RAIL_TOP, tilt=-12, axis="x"),
            "leaning against a neighbour": bal(0.03, bottom=L.RAIL_TOP, tilt=14, axis="x"),
            "on the table, against the rail's side": bal(0.0, x=L.RAIL_X + L.RAIL_W / 2 + R + 0.001,
                                                         bottom=L.TOP, tilt=90, axis="x"),
            "on the table, leaning on a post's front": bal(0.198, x=0.335, bottom=L.TOP, tilt=-10, axis="y"),
            "half on the rail, rolled off its edge": bal(-0.10, x=0.315, bottom=L.TOP + 0.004, tilt=60, axis="y"),
        }
        for what, s in cases.items():
            with self.subTest(what):
                self.assertTrue(down_on_guard(s), what)
                score, d = evaluate(guard(FIVE, rest=[s]))
                self.assertEqual(score, 0.0)
                self.assertEqual(d["down_on_guard"], [L.NAMES[5]])

    def test_not_down_on_the_guard(self):
        cases = {
            "lying on the table 5 mm from the rail": bal(0.0, x=L.RAIL_X + L.RAIL_W / 2 + R + 0.005, bottom=L.TOP,
                                                         tilt=90, axis="x"),
            "lying across a post's top": bal(0.198, x=0.30, bottom=L.TOP + L.POST_SIZE[2], tilt=90, axis="y"),
            "carried tilted high over the rail": bal(0.0, bottom=L.RAIL_TOP + 0.03, tilt=20),
            "upright on the table in front of the rail": bal(0.0, x=0.35, bottom=L.TOP),
            "standing": bal(0.0),
        }
        for what, s in cases.items():
            with self.subTest(what):
                self.assertFalse(down_on_guard(s), what)

    def test_success_needs_stillness(self):
        moving = guard(FIVE, rest=[bal(-0.27, x=0.14, bottom=L.TOP, vel=(0.02, 0, 0))])
        r, _ = run([moving])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        self.assertFalse(r.details["all_still"])
        spinning = [bal(FIVE[0], spin=(0, 0, 0.6))] + [bal(y) for y in FIVE[1:]] + S0[5:]
        r, _ = run([spinning])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        falling_off = bal(-0.27, x=0.20, bottom=L.TOP - 0.30, tilt=90, axis="y", vel=(0, 0, -2))
        r, _ = run([guard(FIVE, rest=[falling_off])])  # one that fell off the table is not waited for
        self.assertTrue(r.success)
        r, _ = run([guard(SIX, rest=[])])
        self.assertTrue(r.success)

    def test_score_is_the_state_at_hand(self):
        r, t = run(build() + [guard(FIVE[:4], rest=S0[4:])])
        self.assertEqual(t.best, 1.0)
        self.assertIn(1.0, t.first_at)
        self.assertEqual(r.score, 0.25)  # one baluster taken away again: four standing, the last state counts


class Exploits(unittest.TestCase):
    def test_laying_balusters_flat_across_the_gaps(self):
        flat = [bal(y, x=0.30, bottom=L.RAIL_TOP, tilt=90, axis="x") for y in (0.13, 0.04, -0.05, -0.14)]
        self.assertEqual(score_of(flat + S0[4:]), 0.0)
        mixed = guard(FIVE[:4], rest=[bal(FIVE[4], x=0.30, bottom=L.RAIL_TOP, tilt=90, axis="y"), S0[5]])
        self.assertEqual(score_of(mixed), 0.0)

    def test_four_zigzagged_across_the_rail_still_pass(self):
        ys = list(L.even_centres(4))
        xs = [L.RAIL_X + (0.0195 if i % 2 else -0.0195) for i in range(4)]
        self.assertEqual(score_of(guard(ys, xs)), 0.0)  # zigzag only widens the openings

    def test_pairs_side_by_side_count_as_two(self):
        ys = [0.09, 0.09, 0.0, 0.0, -0.09, -0.09]
        xs = [0.29, 0.31] * 3
        score, d = evaluate(guard(ys, xs))
        self.assertEqual(d["n_standing"], 6)
        self.assertEqual(score, 0.0)  # three places only: four openings of 72-78 mm pass

    def test_stacking_a_baluster_on_another(self):
        stacked = bal(FIVE[2], bottom=L.RAIL_TOP + H)
        self.assertFalse(stands(stacked))
        self.assertFalse(down_on_guard(stacked))
        # it is not on the rail: the guard scores as the four below it do (one 105 mm opening: 0.25)
        self.assertEqual(score_of(guard(FIVE[:4], rest=[stacked, S0[5]])), score_of(guard(FIVE[:4], rest=S0[4:])))
        self.assertEqual(score_of(guard(FIVE[:4], rest=[stacked, S0[5]])), 0.25)

    def test_standing_them_on_the_table_in_front_of_the_rail(self):
        front = [bal(y, x=0.335, bottom=L.TOP) for y in FIVE]
        self.assertEqual(score_of(front + S0[5:]), 0.0)

    def test_balusters_held_just_off_the_rail(self):
        held = guard(FIVE[:4], rest=[bal(FIVE[4], bottom=L.RAIL_TOP + 0.0015), S0[5]])
        self.assertEqual(score_of(held), 1.0)  # within 2 mm counts (see fidelity)
        held = guard(FIVE[:4], rest=[bal(FIVE[4], bottom=L.RAIL_TOP + 0.003), S0[5]])
        self.assertEqual(score_of(held), 0.25)  # 3 mm up it does not: four standing, its 105 mm gap passes

    def test_one_arm_alone(self):
        """Gripping from above, one arm fetches only its own three balusters: three standing scores 0."""
        left = [bal(y) for y in FIVE[:3]] + S0[3:]
        self.assertEqual(score_of(left), 0.0)
        right = S0[:3] + [bal(y) for y in (-0.03, -0.0627, -0.1253)]
        self.assertEqual(score_of(right), 0.0)
        best_three = [bal(y) for y in ys_from_openings([45, 45, 45, 161])] + S0[3:]
        self.assertEqual(score_of(best_three), 0.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for f in build():
            t.update(f)
        self.assertEqual(t.best, 1.0)
        r = t.update(S0)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        for f in build():
            r = t.update(f)
        self.assertEqual(r.score, 1.0)

    def test_teleport_back_to_jittered_marks_resets(self):
        t = Tracker()
        for f in build():
            t.update(f)
        rng = np.random.default_rng(1)
        jittered = [bal(s.pos[1] + rng.uniform(-0.03, 0.03), s.pos[0] + rng.uniform(-0.03, 0.03), L.TOP) for s in S0]
        t.update(jittered)
        self.assertEqual(t.best, 0.0)

    def test_putting_things_back_gradually_does_not_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 40) + [S0] * 10
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(t.steps, len(frames))

    def test_a_jump_elsewhere_does_not_reset(self):
        scattered = [bal(y, 0.45, L.TOP) for y in (0.5, 0.4, 0.3, -0.3, -0.4, -0.5)]
        r, t = run(build() + [scattered] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.0)

    def test_a_baluster_falling_back_onto_its_mark_is_not_a_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 30)
        (_, x0, y0) = L.BALUSTERS[0]
        falling = bal(y0, x0, L.TOP + 0.10, vel=(0, 0, -1.4))
        _, t = run(frames + [[falling, *S0[1:]], S0])
        self.assertEqual(t.best, 1.0)
        resting = bal(y0, x0, L.TOP + 0.10)  # the same 10 cm jump by a baluster that was still: a teleport
        _, t = run(frames + [[resting, *S0[1:]], S0])
        self.assertEqual(t.best, 0.0)

    def test_marks_need_every_baluster_upright_on_its_mark(self):
        self.assertTrue(Tracker.at_start(S0))
        (_, x0, y0) = L.BALUSTERS[0]
        self.assertFalse(Tracker.at_start([bal(y0, x0, L.TOP, tilt=3), *S0[1:]]))
        self.assertFalse(Tracker.at_start([bal(y0, x0, L.TOP + 0.01), *S0[1:]]))
        self.assertFalse(Tracker.at_start([*S0[:5], bal(-0.27 + 0.05, 0.14, L.TOP)]))
        self.assertFalse(Tracker.at_start([*S0[:5], bal(-0.27, 0.14, L.TOP, vel=(0, 0, 0.05))]))


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the four scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        good = guard(FIVE)
        return [
            [bal(nan), *good[1:]],
            [good[0], state((0.3, 0, L.TOP + 0.05), quat=(nan, 0, 0, nan)), *good[2:]],
            [*good[:5], bal(-0.27, 0.14, L.TOP, vel=(0, inf, 0))],
            [*good[:5], bal(-0.27, 0.14, L.TOP, spin=(nan, 0, 0))],
            [state((0.3, 0.0, L.TOP + 0.05), quat=(0, 0, 0, 0)), *good[1:]],
            [SimpleNamespace(pos=np.zeros(3)), *good[1:]],
        ]

    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        for frame in self.odd():
            with self.subTest():
                t = Tracker()
                for f in build():
                    t.update(f)
                prev, first, best = t.prev, dict(t.first_at), t.best
                r = t.update(frame)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.bad_steps, 1)
                self.assertIs(t.prev, prev)
                self.assertEqual(t.first_at, first)
                self.assertEqual(t.best, best)
                json.dumps(r.details, allow_nan=False)
                r = t.update(guard(FIVE))
                self.assertEqual(r.score, 1.0)

    def test_far_and_strange_states(self):
        rng = np.random.default_rng(5)
        for _ in range(80):
            bals = [state(rng.uniform(-2, 2, 3), tuple(rng.normal(size=4))) for _ in range(6)]
            r = Tracker().update(bals)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(r.details, allow_nan=False)

    def test_random_states_near_the_guard(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(400):
            n = int(rng.integers(3, 7))
            ys = list(L.even_centres(n) + rng.normal(0, 0.006, n))
            bals = [bal(y, x=L.RAIL_X + rng.normal(0, 0.006), bottom=L.RAIL_TOP + rng.normal(0, 0.0015),
                        tilt=abs(rng.normal(0, 3))) for y in ys] + S0[n:]
            score, d = evaluate(bals)
            seen.add(score)
            self.assertIn(score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(d, allow_nan=False)
            if score == 1.0:
                self.assertEqual(d["passing"], 0)
                self.assertLessEqual(d["spread_mm"], 10.05)
                self.assertGreaterEqual(d["n_standing"], 5)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})  # every tier turns up


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, bals):
        return dict(zip(self.task.BAL_NAMES, bals, strict=True))

    def test_objects_match_the_layout(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.BAL_NAMES, L.NAMES)
        for spec, s in zip(self.task.BALUSTERS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((spec.radius, spec.height, spec.mass), (L.BAL_R, L.BAL_H, L.BAL_MASS))
        loose = [o.name for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(self.task.OBJECTS), 1 + 2 + 6)
        rail = self.task.RAIL
        self.assertAlmostEqual(rail.pos[1] + rail.size[1] / 2, L.POST_FACE)  # the rail runs face to face
        for post in self.task.POSTS:
            self.assertAlmostEqual(abs(post.pos[1]) - post.size[1] / 2, L.POST_FACE)

    def test_demo_builds_five_even(self):
        ys = sorted(self.task.PLACE_TO.values(), reverse=True)
        self.assertTrue(np.allclose(ys, L.even_centres(5)))
        self.assertEqual(score_of([bal(self.task.PLACE_TO[n]) if n in self.task.PLACE_TO else s
                                   for n, s in zip(L.NAMES, S0, strict=True)]), 1.0)
        for arm, names in self.task.ORDER.items():
            for n in names:  # each arm places only its own side's balusters, on its own side of the rail
                self.assertTrue(n.startswith(f"baluster_{arm[0]}"))
                self.assertGreaterEqual((1 if arm == "left" else -1) * self.task.PLACE_TO[n], 0.0)

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                self.assertTrue(self.task.success(self.objects(guard(FIVE))))
                self.assertEqual(self.task.score(self.objects(guard(FIVE))), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        for key in ("step", "score", "success", "standing", "n_standing", "down_on_guard", "openings_mm",
                    "passing", "spread_mm", "all_still", "bad_steps", "first_step", "baluster_l1", "baluster_r3"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["openings_mm"], [42.7] * 6)
        self.assertEqual(rows[-1]["baluster_l1"]["bottom_mm"], 0.0)

    def test_trace_survives_a_bad_path(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(self.objects(S0)))
        finally:
            os.environ.pop("TASK_TRACE", None)

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    @staticmethod
    def at_rail(wps):
        """(start, end, y) of each spell a gripper spends at the rail: from the start of 'forward over the rail'
        to the end of 'back behind the posts'."""
        out, t, t0 = [], 0.0, None
        for w in wps:
            if w.label.startswith("forward over the rail"):
                t0 = t
            t += w.duration
            if w.label.startswith("back behind the posts"):
                out.append((t0, t, w.pos[1]))
        return out

    def test_grippers_at_the_rail_stay_apart(self):
        """Whenever both grippers are at the rail at once, they are at least 0.20 m apart along it."""
        plan = self.task.demo()
        left, right = self.at_rail(plan["left"]), self.at_rail(plan["right"])
        self.assertEqual((len(left), len(right)), (3, 2))
        for a0, a1, ya in left:
            for b0, b1, yb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400)  # inside the run limit with the policy's 3 s hold

    def test_carried_balusters_clear_the_guard(self):
        """Carried at HOVER_Z the baluster's bottom clears the standing balusters' and posts' tops; it reaches the
        rail low, along its own line, from behind the posts."""
        t = self.task
        self.assertGreaterEqual(t.HOVER_Z - t.GRIP_UP - (L.TOP + L.BAL_H), 0.015)
        self.assertGreaterEqual(t.HOVER_Z - t.GRIP_UP - (L.TOP + L.POST_SIZE[2]), 0.0)
        self.assertGreaterEqual(t.LOW_Z - t.GRIP_UP - L.RAIL_TOP, 0.015)
        self.assertLessEqual(t.LANE_X + L.BAL_R + 0.018, L.RAIL_X - L.POST_SIZE[0] / 2)  # pads clear of the posts
        self.assertGreaterEqual(t.UP_Z - 0.013 - (L.RAIL_TOP + L.BAL_H), 0.010)  # pad tips over its top
        self.assertGreater(-2 * t.OPEN, L.BAL_D + 0.015)
        self.assertLess(-2 * t.OPEN / 2 + 0.008, 0.060 - L.BAL_R - 0.015)  # open pads clear neighbours at the start

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_success_hold_outlasts_every_release(self):
        """A baluster counts as standing while still gripped (within 2 mm), so the success hold must be longer than
        the settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        plan = self.task.demo()
        n = 0
        for arm, wps in plan.items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly stand"):
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertTrue(settle.label == "settle" and let_go.label.startswith("let go"), arm)
                    self.assertLessEqual(settle.duration + let_go.duration + 0.5, self.task.SUCCESS_HOLD_S)
                    self.assertGreaterEqual(let_go.duration, 1.5)
                    n += 1
        self.assertEqual(n, 5)


if __name__ == "__main__":
    unittest.main()
