"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import itertools
import json
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
    BAR_HALF, Tracker, across_form, bolster_top, crossing, end_covers, evaluate, heading_err_deg, laid, off_table,
    on_bolsters, on_edge, rests_on_bolster, rotation, spacing, tilt_deg, underside, valid,
)

BH = L.BAR_SIZE[2]  # the bar's section (16 mm)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def qmul(a, b):
    """The turn ``b`` followed by the turn ``a`` (quaternions xyzw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0), size=L.BAR_SIZE):
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half  # the world-aligned bounding box of the turned box
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext, size=tuple(size))


def bar_at(x, y=0.0, bottom=L.BOLSTER_TOP, yaw=0.0, roll=0.0, **kw):
    """A bar lying flat with its underside at ``bottom``: along y at ``yaw`` 0 (degrees, about the vertical), rolled
    ``roll`` degrees about its own length (a multiple of 90 puts another long face down)."""
    q = qmul(about("z", yaw), about("y", roll))
    return state((x, y, bottom + BH / 2), q, **kw)


def start():
    """The starting layout: every bar along x on its racks (order: left inner, left outer, right)."""
    return [bar_at(L.BAR_START_X, y, L.RACK_TOP, 90.0) for _, y in L.BARS]


# The demo's mat: left inner at x 0.26, left outer at 0.16, right at 0.36, all centred on y 0.
MAT = dict(zip((n for n, _ in L.BARS), (L.MAT_XS[1], L.MAT_XS[0], L.MAT_XS[2]), strict=True))


def mat(xs=None, ys=(0.0, 0.0, 0.0), yaws=(0.0, 0.0, 0.0), bottom=L.BOLSTER_TOP, bottoms=None, **kw):
    """Three bars laid along y on the bolsters (default: the demo's mat, in the order of ``L.BARS``)."""
    xs = tuple(MAT.values()) if xs is None else xs
    bottoms = (bottom,) * 3 if bottoms is None else bottoms
    return [bar_at(x, y, b, yaw, **kw) for x, y, yaw, b in zip(xs, ys, yaws, bottoms, strict=True)]


S0 = start()


def score_of(bars):
    return evaluate(bars)[0]


def glide(a, b, n):
    """n frames moving every bar of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None):
    """From the start, every bar moved gradually into the mat (or ``final``), then held there."""
    final = mat() if final is None else final
    return [S0] * 5 + glide(S0, final, n) + [final] * 10


def run(frames, hz=20.0):
    """Feed frames (lists of three bar states) to a fresh tracker; return the last result and the tracker."""
    t = Tracker(hz)
    r = None
    for bars in frames:
        r = t.update(bars)
    return r, t


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_and_heading_either_way_round(self):
        self.assertAlmostEqual(tilt_deg(about("x", 6)), 6.0, places=4)  # one end raised 6 degrees
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("y", 90)), 0.0, places=4)  # rolled onto another long face
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 45)), 45.0, places=4)  # rolled onto an edge (arris)
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 90.0, places=4)  # standing on an end
        self.assertAlmostEqual(heading_err_deg(bar_at(0.2), "y"), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(bar_at(0.2, yaw=180), "y"), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(bar_at(0.2, yaw=-2.5), "y"), 2.5, places=6)
        self.assertAlmostEqual(heading_err_deg(bar_at(0.2, yaw=178), "y"), 2.0, places=6)
        self.assertAlmostEqual(heading_err_deg(S0[0], "x"), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(S0[0], "y"), 90.0, places=6)
        self.assertAlmostEqual(heading_err_deg(state((0.2, 0, 0.9), about("x", 90)), "y"), 90.0)  # on its end

    def test_underside_is_the_lowest_long_face(self):
        for roll in (0, 90, 180, 270):
            u = underside(bar_at(0.2, 0.0, L.BOLSTER_TOP, 0.0, roll))
            self.assertAlmostEqual(u.c[2], L.BOLSTER_TOP, places=9)
            self.assertTrue(np.allclose(u.n, (0, 0, 1)))
        u = underside(state((0.2, 0, 0.8), about("x", 4)))
        self.assertAlmostEqual(float(np.ptp(u.corners[:, 2])), 0.4 * np.sin(np.radians(4)), places=6)

    def test_rests_on_bolster_tolerances(self):
        for i in (0, 1):
            self.assertTrue(rests_on_bolster(bar_at(0.2), i))
            self.assertTrue(rests_on_bolster(bar_at(0.2, bottom=L.BOLSTER_TOP + 0.0029), i))  # held just above
            self.assertTrue(rests_on_bolster(bar_at(0.2, bottom=L.BOLSTER_TOP - 0.0029), i))  # pressed in
            self.assertFalse(rests_on_bolster(bar_at(0.2, bottom=L.BOLSTER_TOP + 0.0031), i))
            self.assertFalse(rests_on_bolster(bar_at(0.2, bottom=L.TOP), i))  # on the table beside them
        # over the end of the bolsters: at least half the 16 mm width (96 square mm of 192) must be over them
        self.assertTrue(rests_on_bolster(bar_at(L.BOLSTER_X0 + 0.0005), 0))
        self.assertFalse(rests_on_bolster(bar_at(L.BOLSTER_X0 - 0.0005), 0))
        self.assertTrue(rests_on_bolster(bar_at(L.BOLSTER_X1 - 0.0005), 1))
        self.assertFalse(rests_on_bolster(bar_at(L.BOLSTER_X1 + 0.0005), 1))
        # a bar along y shifted 30 cm sideways misses the far bolster
        self.assertFalse(rests_on_bolster(bar_at(0.2, y=0.30), 1))

    def test_on_bolsters_and_laid(self):
        self.assertTrue(on_bolsters(bar_at(0.2)) and laid(bar_at(0.2)))
        self.assertTrue(laid(bar_at(0.2, yaw=2.9)))
        turned = bar_at(0.2, yaw=3.1)
        self.assertTrue(on_bolsters(turned))
        self.assertFalse(laid(turned))
        self.assertTrue(on_bolsters(bar_at(0.2, yaw=30)))  # across both, but not along y
        self.assertFalse(laid(bar_at(0.2, yaw=30)))
        self.assertTrue(laid(bar_at(0.2, roll=4)))  # rolled 4 degrees about its length: still flat (under 5)
        self.assertFalse(on_bolsters(bar_at(0.2, roll=6)))  # rolled 6 degrees: not flat
        self.assertFalse(on_bolsters(state((0.2, 0, L.BOLSTER_TOP + BH / 2), about("x", 6))))  # an end raised 6 deg
        self.assertFalse(laid(state((0.2, 0, L.BOLSTER_TOP + 0.0113), about("y", 45))))  # on an edge: tilted 45

    def test_crossing_area_matches_sampling(self):
        rng = np.random.default_rng(3)
        for _ in range(25):
            s = bar_at(rng.uniform(0.11, 0.41), rng.uniform(-0.05, 0.05), L.BOLSTER_TOP, rng.uniform(-60, 60))
            for i in (0, 1):
                area, gap = crossing(bolster_top(i), underside(s))
                u, v = np.meshgrid(np.linspace(-0.5, 0.5, 81), np.linspace(-0.5, 0.5, 2001))
                pts = np.stack([u.ravel() * L.BAR_SIZE[0], v.ravel() * L.BAR_SIZE[1], np.zeros(u.size)], 1)
                world = pts @ rotation(s.quat_xyzw).T + s.pos
                inside = ((np.abs(world[:, 0] - L.BOLSTER_X) <= L.BOLSTER_LEN / 2)
                          & (np.abs(world[:, 1] - L.BOLSTER_YS[i]) <= L.BOLSTER_W / 2))
                want = inside.mean() * L.BAR_SIZE[0] * L.BAR_SIZE[1]
                self.assertAlmostEqual(area, want, delta=0.03 * L.BAR_SIZE[0] * L.BOLSTER_W)
                if gap is not None:
                    self.assertAlmostEqual(gap, 0.0, places=9)

    def test_end_covers(self):
        self.assertTrue(np.allclose(end_covers(bar_at(0.2)), (0.030, 0.030)))
        self.assertTrue(np.allclose(end_covers(bar_at(0.2, y=0.012)), (0.018, 0.042)))
        turned = end_covers(bar_at(0.2, yaw=2.0))  # its corners reach a little further
        reach = 0.2 * np.cos(np.radians(2)) + 0.008 * np.sin(np.radians(2))
        self.assertTrue(np.allclose(turned, (L.SLAB_Y - reach, L.SLAB_Y - reach)))
        self.assertLess(end_covers(bar_at(0.2, y=0.035))[0], 0.0)  # past the left form's face

    def test_spacing_is_read_at_each_bolster(self):
        a, b = bar_at(0.16), bar_at(0.26)
        self.assertTrue(np.allclose(spacing(a, b), (0.1, 0.1)))
        self.assertTrue(np.allclose(spacing(b, a), (0.1, 0.1)))
        shift = 0.12 * np.tan(np.radians(2.0))
        g = spacing(a, bar_at(0.26, yaw=2.0))  # same centres, but the tape along each bolster reads differently
        self.assertAlmostEqual(min(g), 0.1 - shift, places=9)
        self.assertAlmostEqual(max(g), 0.1 + shift, places=9)
        self.assertTrue(np.allclose(spacing(a, bar_at(0.26, y=0.05)), (0.1, 0.1)))  # sliding along y: no change

    def test_off_table_edge_and_form(self):
        self.assertTrue(off_table(bar_at(0.3, 0.7, bottom=0.0)))
        self.assertFalse(off_table(bar_at(0.3, 0.5, bottom=L.TOP)))
        self.assertTrue(on_edge(state((0.3, 0.45, L.TOP + 0.0114), qmul(about("z", 90), about("y", 45)))))
        self.assertFalse(on_edge(state((0.3, 0.5, L.TOP + 0.02), about("x", 29))))
        self.assertTrue(on_edge(state((0.3, 0.5, L.TOP + 0.21), about("x", 90))))
        for s in S0:
            self.assertFalse(across_form(s))
        self.assertFalse(across_form(bar_at(0.2)))
        self.assertTrue(across_form(bar_at(0.2, y=0.045, bottom=L.FORM_TOP)))  # its end on the left side form
        self.assertTrue(across_form(bar_at(0.26, 0.0, L.BOLSTER_TOP, 90.0)))  # along x over the front/back forms
        self.assertTrue(across_form(bar_at(0.095, 0.0, L.FORM_TOP)))  # along y on top of the front form
        self.assertFalse(across_form(bar_at(0.2, y=0.045, bottom=L.FORM_TOP + 0.021)))  # carried over it
        self.assertFalse(across_form(bar_at(0.2, y=0.32, bottom=L.TOP, yaw=90.0)))  # on the table outside

    def test_valid(self):
        self.assertTrue(valid(S0[0]))
        self.assertFalse(valid(bar_at(float("nan"))))
        self.assertFalse(valid(state((0.2, 0, 0.8), (0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([S0] * 20)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_mat_scores_one(self):
        r, t = run(build())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["mat_order"], [1, 0, 2])  # left outer, left inner, right along x
        self.assertEqual(r.details["gaps_mm"], [[100.0, 100.0], [100.0, 100.0]])
        self.assertEqual(r.details["cover_mm"], [30.0, 30.0, 30.0])
        self.assertEqual(t.best, 1.0)
        self.assertIn("1.0", r.details["first_step"])

    def test_any_bar_anywhere_either_way_round_any_face_down(self):
        for xs in itertools.permutations((0.16, 0.26, 0.36)):
            self.assertEqual(score_of(mat(xs)), 1.0, xs)
        self.assertEqual(score_of(mat((0.14, 0.24, 0.34), ys=(0.008, -0.009, 0.0))), 1.0)  # shifted mat
        self.assertEqual(score_of(mat((0.29, 0.19, 0.39))), 1.0)
        self.assertEqual(score_of(mat(yaws=(180.0, 0.0, -179.0))), 1.0)
        self.assertEqual(score_of([bar_at(x, roll=r) for x, r in zip((0.16, 0.26, 0.36), (90, 180, 270))]), 1.0)

    def test_spacing_limits(self):
        for gap, want in ((0.1049, 1.0), (0.0951, 1.0), (0.1051, 0.75), (0.0949, 0.75), (0.1119, 0.75),
                          (0.0881, 0.75), (0.1121, 0.5), (0.0879, 0.5)):
            self.assertEqual(score_of(mat((0.16, 0.26, 0.26 + gap))), want, gap)
        self.assertEqual(score_of(mat((0.13, 0.23, 0.39))), 0.5)  # 100 and 160 mm
        self.assertEqual(score_of(mat((0.13, 0.243, 0.356))), 0.25)  # both gaps 113 mm: no pair within 5 mm
        self.assertEqual(score_of(mat((0.16, 0.268, 0.376))), 0.75)  # both 108 mm

    def test_spacing_is_checked_at_both_bolsters(self):
        # centres exactly 100 mm apart, but the middle bar turned 2.5 deg (still 'along y'): the tape along each
        # bolster reads 100 -+ 5.2 mm on either side of it
        bars = mat((0.16, 0.26, 0.36), yaws=(0.0, 2.5, 0.0))
        score, d = evaluate(bars)
        self.assertEqual(d["laid"], [True, True, True])
        self.assertAlmostEqual(d["spacing_dev_mm"], 5.2, places=1)
        self.assertEqual(score, 0.75)
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), yaws=(0.0, 2.0, 0.0))), 1.0)  # 4.2 mm

    def test_cover_limits(self):
        for dy, want in ((0.0099, 1.0), (0.0101, 0.75), (-0.0199, 0.75)):  # covers 20.1, 19.9, 10.1 mm
            self.assertEqual(score_of(mat(ys=(0.0, 0.0, dy))), want, dy)
        # cover 9.9 mm: that bar is not right. As the end bar (x 0.36) the other two still make a pair (0.5); as
        # the middle bar (x 0.26) it breaks both pairs (0.25)
        bad = 0.0201
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), ys=(0.0, 0.0, bad))), 0.5)
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), ys=(0.0, bad, 0.0))), 0.25)
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), ys=(0.0099, -0.0099, 0.0099))), 1.0)

    def test_heading_limit(self):
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), yaws=(0.0, 0.0, 2.9))), 0.75)  # 6 mm off at the bolsters
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), yaws=(0.0, 0.0, 3.1))), 0.5)  # not laid: a pair left
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), yaws=(0.0, 3.1, 0.0))), 0.25)
        self.assertEqual(score_of(mat((0.16, 0.26, 0.36), yaws=(10.0, 10.0, 10.0))), 0.0)  # parallel, but skewed

    def test_two_bars_score_half_one_bar_a_quarter(self):
        left_only = [bar_at(0.26), bar_at(0.16), S0[2]]  # the right bar still on its rack
        self.assertEqual(score_of(left_only), 0.5)
        right_only = [S0[0], S0[1], bar_at(0.36)]
        self.assertEqual(score_of(right_only), 0.25)
        self.assertEqual(score_of([bar_at(0.16), bar_at(0.36), S0[2]]), 0.25)  # 200 mm apart, the middle missing
        self.assertEqual(score_of([bar_at(0.16, y=0.015), S0[1], S0[2]]), 0.0)  # cover 15 mm
        self.assertEqual(score_of([bar_at(0.16, y=0.0099), S0[1], S0[2]]), 0.25)

    def test_a_pair_must_be_neighbours_in_the_mat(self):
        self.assertEqual(score_of(mat((0.16, 0.21, 0.26))), 0.25)  # 0.16 and 0.26 are 100 mm apart, a bar between
        # a bar on the bolsters but turned 10 degrees between two right ones also breaks the pair
        self.assertEqual(score_of(mat((0.16, 0.21, 0.26), yaws=(0.0, 10.0, 0.0))), 0.25)
        # ... but a bar between them that is not on the bolsters (held 10 mm above them) does not
        self.assertEqual(score_of(mat((0.16, 0.21, 0.26), bottoms=(L.BOLSTER_TOP, L.BOLSTER_TOP + 0.010,
                                                                   L.BOLSTER_TOP))), 0.5)

    def test_a_bar_off_the_table_on_its_edge_or_across_a_form_scores_zero(self):
        two = [bar_at(0.26), bar_at(0.16)]
        self.assertEqual(score_of(two + [S0[2]]), 0.5)
        self.assertEqual(score_of(two + [bar_at(0.3, -0.75, bottom=0.0)]), 0.0)  # fell off the table
        edge = qmul(about("z", 90), about("y", 45))  # lying along x outside the forms, rolled onto an arris
        self.assertEqual(score_of(two + [state((0.3, -0.45, L.TOP + 0.0114), edge)]), 0.0)
        self.assertEqual(score_of(two + [state((0.3, -0.5, L.TOP + 0.20), about("x", 90))]), 0.0)  # on its end
        self.assertEqual(score_of(two + [bar_at(0.36, y=-0.045, bottom=L.FORM_TOP)]), 0.0)  # on the side form
        self.assertEqual(score_of(two + [bar_at(0.32, 0.0, L.FORM_TOP, 90.0)]), 0.0)  # along x on the forms
        # the full mat with one bar 36 mm too far left: its end 6 mm over the left form (and resting on it: the
        # form is 5 mm taller than the bolsters, so the bar tips about 0.8 degrees onto the right bolster)
        lean = np.degrees(np.arctan2(L.FORM_H - L.BOLSTER_H, L.SLAB_Y + 0.12))
        q = about("x", lean)
        over = state((0.36, 0.036, L.BOLSTER_TOP + BH / 2 + 0.0026), q)
        self.assertTrue(across_form(over))
        self.assertLess(end_covers(over)[0], 0.0)
        self.assertEqual(score_of(two + [over]), 0.0)

    def test_success_needs_stillness(self):
        moving = mat()
        moving[2] = bar_at(0.36, vel=(0.0, 0.02, 0.0))
        r, _ = run(build() + [moving])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        spinning = mat()
        spinning[0] = bar_at(0.26, spin=(0.0, 0.0, 0.8))
        r, _ = run(build() + [spinning])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        slow = mat()
        slow[1] = bar_at(0.16, vel=(0.009, 0.0, 0.0), spin=(0.4, 0.0, 0.0))
        r, _ = run(build() + [slow])
        self.assertTrue(r.success)

    def test_score_is_the_state_at_hand(self):
        r, t = run(build() + [[bar_at(0.26), bar_at(0.16), bar_at(0.30, -0.45, L.TOP, 90.0)]] * 5)
        self.assertEqual(r.score, 0.5)  # the finished mat knocked apart at the end scores what is left
        self.assertEqual(t.best, 1.0)


class Exploits(unittest.TestCase):
    def test_bars_along_x_score_zero(self):
        # a 400 mm bar along x cannot lie inside the 320 mm long slab: laid along a bolster it rests on the front
        # and back forms (on or across a form: 0)
        along = [bar_at(L.BOLSTER_X, y, L.FORM_TOP, 90.0) for y in (0.12, -0.12)]
        self.assertTrue(all(across_form(s) for s in along))
        self.assertEqual(score_of(along + [S0[2]]), 0.0)
        self.assertEqual(score_of(along + [bar_at(0.26)]), 0.0)

    def test_bars_piled_on_one_another_count_once(self):
        bottom = bar_at(0.26)
        on_top = bar_at(0.26, bottom=L.BOLSTER_TOP + BH)
        third = bar_at(0.26, bottom=L.BOLSTER_TOP + 2 * BH)
        self.assertEqual(score_of([bottom, on_top, third]), 0.25)
        cross = bar_at(0.26, 0.0, L.BOLSTER_TOP + BH, 30.0)  # turned 30 degrees, lying on the laid bar
        self.assertFalse(across_form(cross))
        self.assertEqual(score_of([bottom, S0[1], cross]), 0.25)

    def test_bars_held_just_off_the_bolsters_count_as_resting(self):
        # the scorer sees only the bars (fidelity): held still within 3 mm counts; 3.5 mm up does not
        self.assertEqual(score_of(mat(bottom=L.BOLSTER_TOP + 0.0025)), 1.0)
        self.assertEqual(score_of(mat(bottom=L.BOLSTER_TOP + 0.0035)), 0.0)

    def test_a_skewed_mat_scores_nothing(self):
        # parallel bars 100 mm apart measured square to them, but the whole mat turned 12 degrees: not along y
        skewed = mat((0.155, 0.257, 0.360), yaws=(12.0, 12.0, 12.0))
        self.assertFalse(any(across_form(s) for s in skewed))
        self.assertTrue(all(on_bolsters(s) for s in skewed))
        self.assertEqual(score_of(skewed), 0.0)

    def test_one_arm_alone(self):
        # the left arm alone lays its two bars (0.5); the right arm alone its one (0.25); the left arm cannot
        # get credit for a third bar laid 200 mm from one of its own
        self.assertEqual(score_of([bar_at(0.26), bar_at(0.16), S0[2]]), 0.5)
        self.assertEqual(score_of([S0[0], S0[1], bar_at(0.36)]), 0.25)
        self.assertEqual(score_of([bar_at(0.36), bar_at(0.16), S0[2]]), 0.25)

    def test_a_bar_dropped_onto_one_bolster_scores_zero(self):
        # one end on the table at y 0 inside the slab, the bar rising over the right bolster's edge (10 deg) and
        # on over the right form: not on both bolsters, and its outline crosses the form near the table (0)
        lean = np.arctan2(L.BOLSTER_H, 0.12 - L.BOLSTER_W / 2)
        q = about("x", -np.degrees(lean))
        end = rotation(q) @ np.array([0, BAR_HALF[1], -BAR_HALF[2]])
        dropped = state(np.array([0.26, 0.0, L.TOP]) - end, q)
        self.assertAlmostEqual(float(dropped.low[2]), L.TOP, places=6)
        self.assertFalse(rests_on_bolster(dropped, 0))
        self.assertFalse(on_bolsters(dropped))
        self.assertTrue(across_form(dropped))
        self.assertEqual(score_of([dropped, S0[1], S0[2]]), 0.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for frame in build():
            t.update(frame)
        self.assertEqual(t.best, 1.0)
        r = t.update(S0)  # every bar jumps back onto its mark in one step: a new episode
        self.assertEqual(t.best, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        for frame in build():  # and the new episode can score 1.0 again
            r = t.update(frame)
        self.assertEqual(r.score, 1.0)

    def test_teleport_back_to_jittered_marks_resets(self):
        t = Tracker()
        for frame in build():
            t.update(frame)
        rng = np.random.default_rng(1)
        jittered = [bar_at(s.pos[0] + rng.uniform(-0.03, 0.03), s.pos[1] + rng.uniform(-0.03, 0.03), L.RACK_TOP,
                           90.5) for s in S0]
        t.update(jittered)
        self.assertEqual(t.best, 0.0)

    def test_putting_things_back_gradually_does_not_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 40) + [S0] * 10
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)  # back at the start, but it got there gradually: the same episode
        self.assertEqual(t.steps, len(frames))

    def test_a_jump_elsewhere_does_not_reset(self):
        scattered = [bar_at(0.3, y, L.TOP, 20.0) for y in (0.5, -0.4, -0.5)]
        r, t = run(build() + [scattered] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.0)

    def test_a_bar_falling_back_onto_its_mark_is_not_a_reset(self):
        # one bar dropped from about 20 cm: 10 cm above its mark it falls at 1.4 m/s, so the last step's 10 cm
        # (more than RESET_JUMP plus a free fall from rest) fits its velocity; the other bars are already back
        frames = build()
        frames += glide(frames[-1], S0, 30)
        x0, y0 = S0[0].pos[:2]
        falling = bar_at(x0, y0, L.RACK_TOP + 0.10, 90.0, vel=(0, 0, -1.4))
        _, t = run(frames + [[falling, *S0[1:]], S0])
        self.assertEqual(t.best, 1.0)
        # the same 10 cm jump by a bar that was still is a teleport: a new episode
        resting = bar_at(x0, y0, L.RACK_TOP + 0.10, 90.0)
        _, t = run(frames + [[resting, *S0[1:]], S0])
        self.assertEqual(t.best, 0.0)

    def test_marks_need_every_bar_square_on_its_rack(self):
        self.assertTrue(Tracker.at_start(S0))
        self.assertFalse(Tracker.at_start([bar_at(L.BAR_START_X, L.BARS[0][1], L.RACK_TOP, 80.0), *S0[1:]]))
        self.assertFalse(Tracker.at_start([bar_at(L.BAR_START_X, L.BARS[0][1], L.TOP, 90.0), *S0[1:]]))
        self.assertFalse(Tracker.at_start([*S0[:2], bar_at(L.BAR_START_X, L.BARS[2][1] + 0.05, L.RACK_TOP, 90.0)]))
        self.assertFalse(Tracker.at_start([*S0[:2], bar_at(L.BAR_START_X, L.BARS[2][1], L.RACK_TOP, 90.0,
                                                           vel=(0, 0, 0.05))]))


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        good = mat()
        return [
            [bar_at(nan), *good[1:]],
            [good[0], state((0.2, 0, L.TOP + 0.04), quat=(nan, 0, 0, nan)), good[2]],
            [*good[:2], bar_at(0.36, vel=(0, inf, 0))],
            [*good[:2], bar_at(0.36, spin=(nan, 0, 0))],
            [state((0.26, 0.0, L.TOP + 0.03), quat=(0, 0, 0, 0)), *good[1:]],
            [SimpleNamespace(pos=np.zeros(3)), *good[1:]],  # attributes missing
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
                r = t.update(mat())  # the next good step scores the mat again
                self.assertEqual(r.score, 1.0)

    def test_far_and_strange_states(self):
        rng = np.random.default_rng(5)
        for _ in range(60):
            bars = [state(rng.uniform(-2, 2, 3), tuple(rng.normal(size=4))) for _ in range(3)]
            t = Tracker()
            r = t.update(bars)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)

    def test_random_states_near_the_mat(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            bars = [bar_at(x + rng.normal(0, 0.006), rng.normal(0, 0.01), L.BOLSTER_TOP + rng.normal(0, 0.002),
                           rng.normal(0, 3), rng.choice((0, 90, 180)))
                    for x in (0.16, 0.26, 0.36)]
            score, d = evaluate(bars)
            self.assertIn(score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(d, allow_nan=False)
            if score == 1.0:
                self.assertTrue(all(d["good"]))
                self.assertLessEqual(d["spacing_dev_mm"], 5.0)


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

    def objects(self, bars):
        return dict(zip(self.task.BAR_NAMES, bars, strict=True))

    def test_objects_match_the_layout(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.BAR_NAMES, tuple(n for n, _ in L.BARS))
        for spec, s in zip(self.task.BARS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual(tuple(spec.size), L.BAR_SIZE)
            self.assertEqual(spec.mass, L.BAR_MASS)
        loose = [o.name for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(self.task.BAR_NAMES))
        self.assertEqual(len(self.task.OBJECTS), 4 + 2 + 4 + 3)

    def test_mat_places_are_inside_the_cover_and_on_the_bolsters(self):
        xs = sorted(self.task.PLACE_TO.values())
        self.assertTrue(np.allclose(np.diff(xs), L.SPACING))
        for x in xs:
            self.assertTrue(laid(bar_at(x)))
        self.assertGreaterEqual(L.SLAB_Y - L.BAR_HALF_LEN, L.COVER_FULL + 0.008)  # 30 mm, 10 mm to spare

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                self.assertTrue(self.task.success(self.objects(mat())))
                self.assertEqual(self.task.score(self.objects(mat())), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        for key in ("step", "score", "success", "mat_order", "gaps_mm", "spacing_dev_mm", "cover_mm", "laid",
                    "on_bolsters", "heading_err_deg", "bad_steps", "first_step", "bar_left_inner", "bar_right"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["score"], 1.0)
        self.assertEqual(rows[-1]["bar_right"]["covers_mm"], [30.0, 30.0])

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
    def times(wps, label_start):
        out, t = [], 0.0
        for w in wps:
            if w.label.startswith(label_start):
                out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_one_bar_over_the_mat_at_a_time(self):
        """The left arm carries its first bar over the mat only after the right arm has let go and is up."""
        plan = self.task.demo()
        right_up = self.times(plan["right"], "up")[0][1]
        left_carries = self.times(plan["left"], "carry")
        self.assertEqual(len(left_carries), 2)
        self.assertEqual(len(self.times(plan["right"], "carry")), 1)
        self.assertGreaterEqual(left_carries[0][0] + 1e-9, right_up)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400)  # inside the run limit with the policy's 3 s hold

    def test_instruction_is_the_same_everywhere(self):
        """task.py, task.json and both README commands give the model the same instruction."""
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_success_hold_outlasts_every_release(self):
        """A bar counts as resting while still gripped (within 3 mm), so the success hold must be longer than the
        settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        plan = self.task.demo()
        n = 0
        for arm, wps in plan.items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly set"):
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertTrue(settle.label == "settle" and let_go.label.startswith("let go"), arm)
                    self.assertLessEqual(settle.duration + let_go.duration + 0.5, self.task.SUCCESS_HOLD_S)
                    self.assertGreaterEqual(let_go.duration, 1.5)  # the gripper's rate limit needs 1.5 s
                    n += 1
        self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
