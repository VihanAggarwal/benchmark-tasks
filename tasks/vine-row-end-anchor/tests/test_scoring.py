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
    Tracker, evaluate, is_set, lowest, off_table, offset_reach, rotation, row_score, tilt_deg, valid,
)

MM = 0.001
H = {r: L.row_axis(r)[2] for r in L.ROW_NAMES}  # left 0.100, right 0.150


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def qmul(a, b):
    (x1, y1, z1, w1), (x2, y2, z2, w2) = a, b
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A block's state: its world bounding box worked out from the turned box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ np.array(L.ANCHOR_HALF)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def block(x, y, bottom=L.TOP, yaw=0.0, tilt=0.0, axis="x", **kw):
    """A block at (x, y) with its lowest point at ``bottom``, turned ``yaw`` degrees about the vertical after being
    tilted ``tilt`` degrees about the world ``axis``."""
    q = qmul(about("z", yaw), about(axis, tilt))
    ext = np.abs(rotation(q)) @ np.array(L.ANCHOR_HALF)
    return state((x, y, bottom + ext[2]), q, **kw)


def row_point(row, reach=None, offset=0.0):
    """The point ``reach`` beyond the row's end post's centre along its line (default: the post's height) and
    ``offset`` to the left of the line (seen looking along it, away from the row)."""
    (ex, ey), (ux, uy), h = L.row_axis(row)
    r = h if reach is None else reach
    return ex + r * ux - offset * uy, ey + r * uy + offset * ux


def at_row(row, reach=None, offset=0.0, **kw):
    return block(*row_point(row, reach, offset), **kw)


S0 = [block(*L.START[n]) for n in L.NAMES]
DONE = [at_row("left"), at_row("right")]


def score_of(blocks):
    return evaluate(blocks)[0]


def glide(a, b, n):
    """n frames moving every block of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None, hold=30):
    """From the start, both blocks moved gradually to their anchor points (or ``final``), then held there."""
    final = DONE if final is None else final
    return [S0] * 5 + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


class Geometry(unittest.TestCase):
    def test_lowest_point_and_tilt(self):
        s = block(0.2, 0.0)
        self.assertAlmostEqual(lowest(s), L.TOP)
        self.assertAlmostEqual(tilt_deg(s.quat_xyzw), 0.0)
        upside_down = block(0.2, 0.0, tilt=180.0)
        self.assertAlmostEqual(tilt_deg(upside_down.quat_xyzw), 0.0, places=5)  # top and bottom are alike
        self.assertAlmostEqual(lowest(upside_down), L.TOP)
        lying = block(0.2, 0.0, tilt=90.0, axis="y")  # on a 30 x 40 mm side: centre 15 mm up
        self.assertAlmostEqual(tilt_deg(lying.quat_xyzw), 90.0, places=5)
        self.assertAlmostEqual(float(lying.pos[2]) - L.TOP, 0.015)
        self.assertAlmostEqual(tilt_deg(block(0.2, 0.0, tilt=7.0).quat_xyzw), 7.0, places=5)

    def test_set(self):
        self.assertTrue(is_set(block(0.2, 0.0)))
        self.assertTrue(is_set(block(0.2, 0.0, tilt=180.0)))  # either end up
        self.assertTrue(is_set(block(0.2, 0.0, yaw=37.0)))  # square: its heading does not matter
        self.assertTrue(is_set(block(0.2, 0.0, bottom=L.TOP + 2.9 * MM)))
        self.assertFalse(is_set(block(0.2, 0.0, bottom=L.TOP + 3.2 * MM)))  # held above the table
        self.assertTrue(is_set(block(0.2, 0.0, tilt=9.9)))
        self.assertFalse(is_set(block(0.2, 0.0, tilt=10.2)))
        self.assertFalse(is_set(block(0.2, 0.0, tilt=90.0, axis="y")))  # lying on a side
        self.assertFalse(is_set(block(0.2, 0.0, bottom=L.TOP + L.ANCHOR_SIZE[2])))  # standing on the other block

    def test_offset_and_reach(self):
        for row in L.ROW_NAMES:
            (ex, ey), _, h = L.row_axis(row)
            off, reach = offset_reach(at_row(row), row)
            self.assertAlmostEqual(off, 0.0)
            self.assertAlmostEqual(reach, h)
            off, reach = offset_reach(at_row(row, reach=0.07, offset=-0.012), row)
            self.assertAlmostEqual(off, 0.012)
            self.assertAlmostEqual(reach, 0.07)
            self.assertAlmostEqual(offset_reach(block(ex, ey), row)[1], 0.0)  # at the end post
            line = [ln for r, _, _, ln in L.ROWS if r == row][0]
            off, reach = offset_reach(block(*line), row)  # the line post: on the line, the wrong way
            self.assertAlmostEqual(off, 0.0)
            self.assertAlmostEqual(reach, -L.POST_GAP)

    def test_targets(self):
        """The anchor points: on the line through each row's two posts, H beyond the end post, toward the arms."""
        self.assertAlmostEqual(H["left"], 0.100)
        self.assertAlmostEqual(H["right"], 0.150)
        self.assertAlmostEqual(L.POST_GAP, 0.130507, places=6)
        self.assertAlmostEqual(L.ROW_ANGLE_DEG, 15.10, places=2)
        want = {"left": (0.22345, 0.22605), "right": (0.22518, -0.14092)}
        for row, (end, h, line) in ((r, (e, hh, ln)) for r, e, hh, ln in L.ROWS):
            tx, ty = L.TARGETS[row]
            self.assertAlmostEqual(tx, want[row][0], places=5)
            self.assertAlmostEqual(ty, want[row][1], places=5)
            self.assertAlmostEqual(math.hypot(tx - end[0], ty - end[1]), h)
            # collinear with the two posts, the end post between the anchor and the line post
            cross = (end[0] - line[0]) * (ty - line[1]) - (end[1] - line[1]) * (tx - line[0])
            self.assertAlmostEqual(cross, 0.0)
            self.assertAlmostEqual(math.hypot(tx - line[0], ty - line[1]), h + L.POST_GAP)
        self.assertGreater(L.TARGETS["left"][1] - L.TARGETS["right"][1], 0.36)  # the place points 0.367 m apart

    def test_off_table(self):
        self.assertFalse(off_table(block(0.2, 0.0)))
        self.assertFalse(off_table(block(0.2, 0.0, bottom=L.TOP - 9 * MM)))
        self.assertTrue(off_table(block(0.95, 0.0, bottom=L.TOP - 0.2)))

    def test_valid(self):
        self.assertTrue(valid(S0[0]))
        self.assertFalse(valid(block(float("nan"), 0.0)))
        self.assertFalse(valid(state((0.2, 0.0, L.TOP + 0.02), quat=(0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        self.assertFalse(valid(state((0.2, 0.0, L.TOP + 0.02), vel=(0, float("inf"), 0))))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual(score, 0.0)
        self.assertEqual((d["left_row"], d["right_row"]), (0.0, 0.0))
        r = Tracker().update(S0)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_both_rows_right_scores_one(self):
        score, d = evaluate(DONE)
        self.assertEqual(score, 1.0)
        self.assertEqual((d["left"]["anchor"], d["right"]["anchor"]), ("anchor_left", "anchor_right"))
        self.assertEqual(d["left"]["reach_mm"], 100.0)
        self.assertEqual(d["right"]["reach_mm"], 150.0)

    def test_one_row_right_scores_half(self):
        self.assertEqual(score_of([at_row("left"), S0[1]]), 0.5)
        self.assertEqual(score_of([S0[0], at_row("right")]), 0.5)

    def test_one_right_other_loose_scores_three_quarters(self):
        self.assertEqual(score_of([at_row("left"), at_row("right", reach=H["right"] + 0.013)]), 0.75)
        self.assertEqual(score_of([at_row("left", offset=0.017), at_row("right")]), 0.75)

    def test_both_loose_scores_half(self):
        self.assertEqual(score_of([at_row("left", reach=H["left"] - 0.012),
                                   at_row("right", offset=-0.015, reach=H["right"] + 0.011)]), 0.5)

    def test_one_loose_scores_a_quarter(self):
        self.assertEqual(score_of([at_row("left", offset=0.012), S0[1]]), 0.25)
        self.assertEqual(score_of([S0[0], at_row("right", reach=H["right"] - 0.014)]), 0.25)

    def test_tight_and_loose_limits(self):
        for row in L.ROW_NAMES:
            h = H[row]
            cases = (
                ((0.0, 9.9), 0.5), ((0.0, 10.1), 0.25), ((0.0, -9.9), 0.5), ((0.0, -10.1), 0.25),
                ((9.9, 0.0), 0.5), ((10.1, 0.0), 0.25), ((-9.9, 0.0), 0.5), ((-10.1, 0.0), 0.25),
                ((0.0, 14.9), 0.25), ((0.0, 15.1), 0.0), ((0.0, -14.9), 0.25), ((0.0, -15.1), 0.0),
                ((19.9, 0.0), 0.25), ((20.1, 0.0), 0.0), ((-19.9, 14.9), 0.25), ((9.9, 9.9), 0.5),
                ((10.1, 9.9), 0.25), ((20.1, 14.9), 0.0),
            )
            for (off_mm, err_mm), want in cases:
                with self.subTest(row=row, offset=off_mm, err=err_mm):
                    s = at_row(row, reach=h + err_mm * MM, offset=off_mm * MM)
                    self.assertEqual(row_score(s, row)[0], want)

    def test_every_tier_turns_up(self):
        seen = set()
        for left in (S0[0], at_row("left", offset=0.015), at_row("left")):
            for right in (S0[1], at_row("right", reach=H["right"] + 0.013), at_row("right")):
                seen.add(score_of([left, right]))
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_either_block_may_serve_either_row(self):
        swapped = [at_row("right"), at_row("left")]
        score, d = evaluate(swapped)
        self.assertEqual(score, 1.0)
        self.assertEqual((d["left"]["anchor"], d["right"]["anchor"]), ("anchor_right", "anchor_left"))

    def test_a_block_counts_for_one_row_only(self):
        both_left = [at_row("left"), at_row("left", offset=0.008)]  # two blocks side by side on the left point
        self.assertEqual(score_of(both_left), 0.5)
        # The matching takes the better pairing: one block right on the left row, the other loose on the right.
        mixed = [at_row("right", reach=H["right"] + 0.012), at_row("left")]
        self.assertEqual(score_of(mixed), 0.75)

    def test_a_block_off_the_table_scores_zero(self):
        fallen = block(0.95, -0.2, bottom=L.TOP - 0.3)
        self.assertEqual(score_of([at_row("left"), fallen]), 0.0)
        self.assertEqual(score_of([fallen, at_row("right")]), 0.0)
        self.assertEqual(evaluate([fallen, at_row("right")])[1]["off_table"], ["anchor_left"])

    def test_a_block_not_set_scores_zero_for_its_row(self):
        for bad in (at_row("right", tilt=90.0, axis="y"),  # lying on a side on the point
                    at_row("right", bottom=L.TOP + 0.005),  # held 5 mm above it
                    at_row("right", tilt=14.0)):  # leaning
            with self.subTest():
                self.assertEqual(score_of([at_row("left"), bad]), 0.5)

    def test_success_needs_stillness(self):
        r, _ = run(build())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        frames = [S0] * 5 + glide(S0, DONE, 20)
        r, _ = run(frames)  # just arrived: the last 0.5 s still has the move in it
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        jitter = [DONE] * 20 + [[at_row("left", offset=(-1) ** i * 0.0006), DONE[1]] for i in range(12)]
        r, _ = run(jitter)  # 1.2 mm a step back and forth: 13 mm of path over 0.5 s, not still
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)

    def test_score_is_the_state_at_hand(self):
        moved_on = [at_row(r, reach=H[r] + 0.040) for r in L.ROW_NAMES]  # both pushed 40 mm farther out after 1.0
        r, t = run(build() + glide(DONE, moved_on, 5))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(t.first_at[1.0], 5 + 20)


class Mistakes(unittest.TestCase):
    """The wrong rules named in task.json's scoring, with the numbers given there."""

    def test_straight_back_along_x(self):
        blocks = []
        for row in L.ROW_NAMES:
            (ex, ey), _, h = L.row_axis(row)
            blocks.append(block(ex - h, ey))
        _, d = evaluate(blocks)
        self.assertAlmostEqual(d["left"]["offset_mm"], 26.1, delta=0.1)
        self.assertAlmostEqual(d["right"]["offset_mm"], 39.1, delta=0.1)
        self.assertEqual(score_of(blocks), 0.0)

    def test_heights_swapped(self):
        blocks = [at_row("left", reach=H["right"]), at_row("right", reach=H["left"])]
        self.assertEqual(score_of(blocks), 0.0)

    def test_the_line_posts_height(self):
        blocks = [at_row(r, reach=L.LINE_POST_H[r]) for r in L.ROW_NAMES]
        _, d = evaluate(blocks)
        self.assertAlmostEqual(d["left"]["reach_err_mm"], 20.0)
        self.assertAlmostEqual(d["right"]["reach_err_mm"], 20.0)
        self.assertEqual(score_of(blocks), 0.0)

    def test_the_post_spacing(self):
        blocks = [at_row("left", reach=L.POST_GAP), at_row("right", reach=L.POST_GAP)]
        _, d = evaluate(blocks)
        self.assertAlmostEqual(d["left"]["reach_err_mm"], 30.5, delta=0.1)
        self.assertAlmostEqual(d["right"]["reach_err_mm"], 19.5, delta=0.1)
        self.assertEqual(score_of(blocks), 0.0)

    def test_one_distance_for_both_rows(self):
        self.assertEqual(score_of([at_row("left", reach=0.100), at_row("right", reach=0.100)]), 0.5)
        self.assertEqual(score_of([at_row("left", reach=0.150), at_row("right", reach=0.150)]), 0.5)
        self.assertEqual(score_of([at_row("left", reach=0.125), at_row("right", reach=0.125)]), 0.0)

    def test_the_wrong_side_of_the_post(self):
        beyond_line = [at_row(r, reach=-(L.POST_GAP + H[r])) for r in L.ROW_NAMES]  # past the line post
        self.assertEqual(score_of(beyond_line), 0.0)
        between = [at_row(r, reach=-L.POST_GAP / 2, offset=0.03) for r in L.ROW_NAMES]  # beside the wire
        self.assertEqual(score_of(between), 0.0)
        # The left row's 100 mm measured the wrong way from the end post, toward the line post: on the line, under
        # the wire, between the posts (reach -100 mm).
        wrong_way = [at_row("left", reach=-H["left"]), S0[1]]
        self.assertEqual(evaluate(wrong_way)[1]["left"]["reach_mm"], -100.0)
        self.assertEqual(score_of(wrong_way), 0.0)


class Exploits(unittest.TestCase):
    def test_stacking_one_block_on_the_other(self):
        base = at_row("left")
        on_top = at_row("left", bottom=L.TOP + L.ANCHOR_SIZE[2])
        self.assertEqual(score_of([base, on_top]), 0.5)  # only the one on the table counts
        self.assertEqual(score_of([on_top, S0[1]]), 0.0)

    def test_laying_blocks_on_their_sides_at_the_points(self):
        lying = [at_row(r, tilt=90.0, axis="x") for r in L.ROW_NAMES]
        self.assertEqual(score_of(lying), 0.0)

    def test_holding_blocks_just_above_the_points(self):
        held = [at_row(r, bottom=L.TOP + 0.004) for r in L.ROW_NAMES]
        self.assertEqual(score_of(held), 0.0)
        # The scorer cannot see the grippers: held still within 3 mm of the table counts (task.json's fidelity).
        self.assertEqual(score_of([at_row(r, bottom=L.TOP + 0.002) for r in L.ROW_NAMES]), 1.0)

    def test_sweeping_a_block_off_the_table(self):
        self.assertEqual(score_of([at_row("left"), block(0.3, -0.70, bottom=L.TOP - 0.5)]), 0.0)

    def test_leaning_a_block_against_a_post(self):
        (ex, ey), (ux, uy), _ = L.row_axis("right")
        leaning = block(ex + 0.03 * ux, ey + 0.03 * uy, tilt=20.0, axis="y")
        self.assertEqual(score_of([at_row("left"), leaning]), 0.5)

    def test_one_arm_alone(self):
        self.assertEqual(score_of([at_row("left"), S0[1]]), 0.5)
        self.assertEqual(score_of([S0[0], at_row("right")]), 0.5)
        # One arm with only its own block: wherever it goes, at most one row.
        for row in L.ROW_NAMES:
            self.assertEqual(score_of([at_row(row), S0[1]]), 0.5)


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
        self.assertFalse(r.success)
        for f in build():
            r = t.update(f)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_putting_things_back_gradually_does_not_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 40) + [S0] * 10
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(t.steps, len(frames))

    def test_a_jump_elsewhere_does_not_reset(self):
        scattered = [block(0.45, 0.3), block(0.45, -0.3)]
        r, t = run(build() + [scattered] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.0)

    def test_a_block_falling_back_onto_its_spot_is_not_a_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 30)
        x0, y0 = L.START["anchor_left"]
        falling = block(x0, y0, bottom=L.TOP + 0.10, vel=(0, 0, -1.4))
        _, t = run(frames + [[falling, S0[1]], S0])
        self.assertEqual(t.best, 1.0)
        resting = block(x0, y0, bottom=L.TOP + 0.10)  # the same 10 cm jump by a block that was still: a teleport
        _, t = run(frames + [[resting, S0[1]], S0])
        self.assertEqual(t.best, 0.0)

    def test_the_start_needs_both_blocks_on_their_spots(self):
        self.assertTrue(Tracker.at_start(S0))
        x0, y0 = L.START["anchor_left"]
        self.assertTrue(Tracker.at_start([block(x0 + 0.03, y0 - 0.02), S0[1]]))  # a seeded reset's jitter
        self.assertFalse(Tracker.at_start([block(x0, y0, tilt=3), S0[1]]))
        self.assertFalse(Tracker.at_start([block(x0, y0, bottom=L.TOP + 0.01), S0[1]]))
        self.assertFalse(Tracker.at_start([S0[0], at_row("right")]))
        self.assertFalse(Tracker.at_start([S0[0], block(*L.START["anchor_right"], vel=(0, 0, 0.05))]))


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        good = DONE
        return [
            [block(nan, 0.2), good[1]],
            [good[0], state((0.2, 0, L.TOP + 0.02), quat=(nan, 0, 0, nan))],
            [good[0], at_row("right", vel=(0, inf, 0))],
            [good[0], at_row("right", spin=(nan, 0, 0))],
            [state((0.2, 0.0, L.TOP + 0.02), quat=(0, 0, 0, 0)), good[1]],
            [SimpleNamespace(pos=np.zeros(3)), good[1]],
        ]

    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        for frame in self.odd():
            with self.subTest():
                t = Tracker()
                for f in build():
                    t.update(f)
                prev, first, best, hist = t.prev, dict(t.first_at), t.best, list(t.history)
                r = t.update(frame)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.bad_steps, 1)
                self.assertIs(t.prev, prev)
                self.assertEqual(t.first_at, first)
                self.assertEqual(t.best, best)
                self.assertEqual(len(t.history), len(hist))
                json.dumps(r.details, allow_nan=False)
                r = t.update(DONE)
                self.assertEqual(r.score, 1.0)
                self.assertTrue(r.success)

    def test_far_and_strange_states(self):
        rng = np.random.default_rng(5)
        for _ in range(120):
            blocks = [state(rng.uniform(-2, 2, 3), tuple(rng.normal(size=4))) for _ in range(2)]
            r = Tracker().update(blocks)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)

    def test_random_states_near_the_anchor_points(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(800):
            blocks = []
            for row in L.ROW_NAMES:
                blocks.append(at_row(row, reach=H[row] + rng.normal(0, 0.010), offset=rng.normal(0, 0.010),
                                     bottom=L.TOP + abs(rng.normal(0, 0.0012)), yaw=rng.uniform(-45, 45),
                                     tilt=abs(rng.normal(0, 3.0))))
            score, d = evaluate(blocks)
            seen.add(score)
            self.assertIn(score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(d, allow_nan=False)
            if score == 1.0:
                for row in L.ROW_NAMES:
                    self.assertLessEqual(d[row]["offset_mm"], 10.05)
                    self.assertLessEqual(d[row]["reach_err_mm"], 10.05)
                    self.assertTrue(d[row]["set"])
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})  # every tier turns up


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

    def objects(self, blocks):
        return dict(zip(self.task.ANCHOR_NAMES, blocks, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.ANCHOR_NAMES, L.NAMES)
        self.assertEqual(len(t.OBJECTS), 4 + 2 + 2)
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        for spec, s in zip(t.ANCHORS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((tuple(spec.size), spec.mass, spec.yaw), (L.ANCHOR_SIZE, L.ANCHOR_MASS, 0.0))
        posts = {o.name: o for o in t.POSTS}
        for row, end, h, line in L.ROWS:
            ep, lp, wire = posts[f"end_post_{row}"], posts[f"line_post_{row}"], [w for w in t.WIRES
                                                                                  if w.name == f"wire_{row}"][0]
            self.assertTrue(np.allclose(ep.pos[:2], end) and np.allclose(lp.pos[:2], line))
            self.assertAlmostEqual(ep.size[2], h)
            self.assertAlmostEqual(lp.size[2], L.LINE_POST_H[row])
            for o in (ep, lp, wire):
                self.assertIsNone(o.mass)
                self.assertAlmostEqual(o.yaw, L.ROW_YAW)  # turned along the row
            for o in (ep, lp):
                self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)  # standing on the table
            # The wire runs along the row between the posts' facing sides, 0.5 mm short of each, under both tops.
            self.assertTrue(np.allclose(wire.pos[:2], ((end[0] + line[0]) / 2, (end[1] + line[1]) / 2)))
            self.assertAlmostEqual(wire.size[0] / 2 + L.POST_SIDE / 2 + 0.0005, L.POST_GAP / 2)
            self.assertLess(wire.pos[2] + wire.size[2] / 2, L.TOP + min(h, L.LINE_POST_H[row]))
        for row in L.ROW_NAMES:  # each line post 20 mm lower than its row's end post
            self.assertAlmostEqual(H[row] - L.LINE_POST_H[row], 0.020)
        # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
        for o in t.OBJECTS:
            self.assertLess(abs(o.pos[1]) + max(o.size[:2]) / 2, 0.45, o.name)

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                for _ in range(11):
                    ok = self.task.success(self.objects(DONE))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 12)
        for key in ("step", "score", "success", "left_row", "right_row", "left", "right", "off_table",
                    "all_still", "bad_steps", "first_step", "anchor_left", "anchor_right"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["left"]["offset_mm"], 0.0)
        self.assertEqual(rows[-1]["right"]["reach_mm"], 150.0)
        self.assertEqual(rows[-1]["anchor_left"]["left_reach_mm"], 100.0)
        self.assertEqual(rows[-1]["anchor_left"]["path_mm"], 0.0)
        self.assertTrue(rows[-1]["anchor_right"]["set"])
        self.assertIsNone(rows[0]["anchor_left"]["path_mm"])

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(self.objects(S0)))
        finally:
            os.environ.pop("TASK_TRACE", None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                self.assertFalse(self.task.success(self.objects([block(float("nan"), 0.0), S0[1]])))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                row = json.loads(f.readline())
        self.assertFalse(row["valid_state"])

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_sets_both_anchors(self):
        """Each arm picks its own block where it stands and lets go of it over its own row's anchor point."""
        t = self.task
        plan = t.demo()
        final = list(S0)
        for arm, wps in plan.items():
            name, row = t.PLAN[arm]
            self.assertEqual(row, arm)
            grip = [w for w in wps if w.label == f"grip {name}"][0]
            self.assertTrue(np.allclose(grip.pos[:2], L.START[name]))
            let_go = [w for w in wps if w.label == "let go of it"][0]
            self.assertTrue(np.allclose(let_go.pos[:2], L.TARGETS[row]))
            self.assertAlmostEqual(let_go.pos[2] - t.GRIP_UP, L.TOP + t.DROP)
            final[L.NAMES.index(name)] = block(*let_go.pos[:2])
        self.assertEqual(score_of(final), 1.0)

    @staticmethod
    def spells(wps):
        """(start, end, x, y) of each waypoint, on the plan's clock."""
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w.pos[0], w.pos[1]))
            t += w.duration
        return out

    def test_grippers_stay_apart(self):
        """The two grippers are always at least 0.20 m apart in y; the demo fits the run limit."""
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, _, ya in left:
            for b0, b1, _, yb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 1200)  # inside the run limit with the policy's 3 s hold

    def test_grip_geometry(self):
        t = self.task
        self.assertGreater(-2 * t.OPEN, L.ANCHOR_SIZE[1] + 0.060)  # 32 mm either side of a block when open
        self.assertAlmostEqual(-2 * t.GRIP, L.ANCHOR_SIZE[1] - 0.008)  # the grip is commanded 8 mm narrower
        self.assertGreaterEqual(t.PICK_Z - 0.013 - L.TOP, 0.008)  # pad tips 9 mm over the table
        # The pads (from 25 mm above the tool point to their tips 13 mm below it) close on the block's lower part
        # and cover at least 25 mm of its 40 mm height.
        tips, top = t.GRIP_UP - 0.013, t.GRIP_UP + 0.025
        self.assertLessEqual(tips, L.ANCHOR_SIZE[2] / 2)
        self.assertGreaterEqual(min(top, L.ANCHOR_SIZE[2]) - tips, 0.025)
        # Seen from above, the open pads over an anchor point (8 mm thick outside the 94 mm gap along y, 20 mm wide
        # along x) clear every post's turned 25 x 25 mm square by more than 70 mm.
        half_x, half_y = 0.010, -t.OPEN + 0.008
        c, s = math.cos(L.ROW_YAW), math.sin(L.ROW_YAW)

        def to_square(p, centre):
            dx, dy = p[0] - centre[0], p[1] - centre[1]
            u, v = c * dx + s * dy, -s * dx + c * dy
            return math.hypot(max(abs(u) - L.POST_SIDE / 2, 0.0), max(abs(v) - L.POST_SIDE / 2, 0.0))

        def to_rect(p, tx, ty):
            return math.hypot(max(abs(p[0] - tx) - half_x, 0.0), max(abs(p[1] - ty) - half_y, 0.0))

        for row in L.ROW_NAMES:
            tx, ty = L.TARGETS[row]
            for _, end, _, line in L.ROWS:
                for centre in (end, line):
                    h = L.POST_SIDE / 2
                    corners = [(centre[0] + c * a - s * b, centre[1] + s * a + c * b) for a in (-h, h) for b in (-h, h)]
                    pads = [(tx + a, ty + b) for a in (-half_x, half_x) for b in (-half_y, half_y)]
                    gap = min(min(to_rect(p, tx, ty) for p in corners), min(to_square(p, centre) for p in pads))
                    self.assertGreater(gap, 0.070, (row, centre))

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
        """A block counts as set while still gripped (within 3 mm), so the success hold must be longer than the
        settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        plan = self.task.demo()
        n = 0
        for arm, wps in plan.items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly set it"):
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertTrue(settle.label == "settle" and let_go.label.startswith("let go"), arm)
                    self.assertLessEqual(settle.duration + let_go.duration + 0.5, self.task.SUCCESS_HOLD_S)
                    self.assertGreaterEqual(let_go.duration, 1.5)
                    n += 1
        self.assertEqual(n, 2)


def _yamkin():
    for p in HERE.parents:
        if (p / "tools" / "offline" / "yamkin.py").is_file():
            return p / "tools" / "offline"
    return None


@unittest.skipIf(_yamkin() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class Reach(unittest.TestCase):
    """The README's one-arm claim, by our IK model: gripping straight down, neither arm reaches the other's block."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_yamkin()))
        cls.K = importlib.import_module("yamkin")

    def test_own_block_and_row_reachable_far_block_not(self):
        K = self.K
        bases = K.BASES["station"]
        for arm, own, far in (("left", "anchor_left", "anchor_right"), ("right", "anchor_right", "anchor_left")):
            for z in (0.022, 0.120):
                with self.subTest(arm=arm, z=z):
                    self.assertTrue(K.reachable((*L.START[own], L.TOP + z), 0.0, bases[arm])[0])
                    self.assertTrue(K.reachable((*L.TARGETS[arm], L.TOP + z), 0.0, bases[arm])[0])
                    self.assertFalse(K.reachable((*L.START[far], L.TOP + z), 0.0, bases[arm])[0])


if __name__ == "__main__":
    unittest.main()
