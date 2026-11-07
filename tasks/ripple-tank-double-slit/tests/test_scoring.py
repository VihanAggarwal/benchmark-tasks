"""Offline tests for scoring.py and task.py: synthetic bar states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    HALF, Tracker, axis_err_deg, bar_info, ends, evaluate, lowest, openings, outline, rotation, tilt_deg, valid,
)

HZ = 20.0
TOP = L.TOP
FLOOR = TOP + L.FLOOR_TOP
LONG_HALF = L.LONG_SIZE[1] / 2  # 35 mm
SHORT_HALF = L.SHORT_SIZE[1] / 2  # 15 mm
EXAMPLE_GAP = 0.001  # task.json's 1.0 example: both long bars 1 mm from their walls, the short bar centred


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def state(name, center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like bar: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ HALF[name]
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def bar(name, x, y, turn=0.0, lift=0.0, tilt=0.0, tilt_axis="x", flip=False, base=FLOOR, quat=None):
    """A bar standing (length along y) centred at (x, y) with its lowest point at ``base + lift``; turned ``turn``
    degrees about the vertical, tilted ``tilt`` degrees about its own x (one end up) or y axis (leaning toward the
    beach), turned upside down with ``flip``; or with an explicit ``quat``."""
    if quat is None:
        quat = about("z", turn)
        if flip:
            quat = qmul(quat, about("x", 180.0))
        if tilt:
            quat = qmul(quat, about(tilt_axis, tilt))
    s = state(name, (x, y, 0.0), quat)
    return state(name, (x, y, base + lift - (s.low[2] - s.pos[2])), quat)


def by_gaps(gl=EXAMPLE_GAP, gr=EXAMPLE_GAP, dc=0.0, x=L.LINE_X, **kw):
    """The bars in order A, C, B across the tank: A ``gl`` from the left wall, B ``gr`` from the right wall, C ``dc``
    left of the centre line (m)."""
    return {"bar_a": bar("bar_a", x, L.TANK_HALF - gl - LONG_HALF, **kw),
            "bar_b": bar("bar_b", x, -(L.TANK_HALF - gr - LONG_HALF), **kw),
            "bar_c": bar("bar_c", x, dc, **kw)}


def target():
    return by_gaps()


def start():
    return {n: bar(n, x, y, base=TOP) for n, (x, y) in L.BAR_START.items()}


def short_at_left_wall(c_gap=0.0, slit1=0.015, slit2=0.015):
    """The wrong order: C against the left wall, bar A in the middle, bar B against the right wall."""
    yc = L.TANK_HALF - c_gap - SHORT_HALF
    ya = yc - SHORT_HALF - slit1 - LONG_HALF
    yb = ya - LONG_HALF - slit2 - LONG_HALF
    return {"bar_a": bar("bar_a", L.LINE_X, ya), "bar_b": bar("bar_b", L.LINE_X, yb),
            "bar_c": bar("bar_c", L.LINE_X, yc)}


def flags(bars):
    s, d = evaluate(bars)
    return s, (d["slits_ok"], d["centres_ok"], d["walls_ok"]), d


def run(seq, t=None):
    t = Tracker(HZ) if t is None else t
    r = None
    for bars in seq:
        r = t.update(bars)
    return r, t


def still(bars, n=12):
    return [bars] * n


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        bars = start()
        self.assertTrue(Tracker.at_start(bars))
        for n, s in bars.items():
            self.assertAlmostEqual(lowest(s, n), 0.0, places=9)
            i = bar_info(s, n)
            self.assertFalse(i["on_floor"])  # on the table, 3 mm below the floor's top
            self.assertFalse(i["inside"])
            self.assertFalse(i["stands"])
        self.assertEqual(evaluate(bars)[0], 0.0)

    def test_angles_fold(self):
        def q(quat):
            return SimpleNamespace(quat_xyzw=quat)

        self.assertAlmostEqual(tilt_deg(q(about("x", 7.0))), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(q(about("x", 180.0))), 0.0, places=6)  # upside down: the same bar
        self.assertAlmostEqual(tilt_deg(q(about("x", 173.0))), 7.0, places=6)
        self.assertAlmostEqual(axis_err_deg(q(about("z", 4.0))), 4.0, places=6)
        self.assertAlmostEqual(axis_err_deg(q(about("z", 176.0))), 4.0, places=6)  # either end either way
        self.assertAlmostEqual(axis_err_deg(q(about("z", 90.0))), 90.0, places=6)
        on_end = q(about("x", 90.0))  # standing on an end: no length seen from above
        self.assertAlmostEqual(axis_err_deg(on_end), 90.0, places=6)
        self.assertAlmostEqual(tilt_deg(on_end), 90.0, places=6)

    def test_outline_and_ends(self):
        s = bar("bar_a", 0.3, 0.06)
        poly = outline(s, "bar_a")
        self.assertEqual(poly.shape, (4, 2))
        np.testing.assert_allclose(poly[:, 1].min(), 0.06 - LONG_HALF, atol=1e-9)
        np.testing.assert_allclose(poly[:, 0].max(), 0.31, atol=1e-9)
        e = ends(s, "bar_a")
        np.testing.assert_allclose(sorted(e[:, 1]), [0.06 - LONG_HALF, 0.06 + LONG_HALF], atol=1e-9)
        turned = outline(bar("bar_a", 0.3, 0.0, turn=10.0), "bar_a")  # its shadow, not its box
        self.assertEqual(turned.shape, (4, 2))
        tilted = outline(bar("bar_a", 0.3, 0.0, turn=10.0, tilt=8.0), "bar_a")
        self.assertEqual(tilted.shape, (6, 2))

    def test_inside_allows_wall_contact_not_a_prop(self):
        y_touch = L.TANK_HALF - LONG_HALF
        self.assertTrue(bar_info(bar("bar_a", 0.3, y_touch), "bar_a")["inside"])
        self.assertTrue(bar_info(bar("bar_a", 0.3, y_touch + 0.0029), "bar_a")["inside"])  # sunk 2.9 mm into it
        self.assertFalse(bar_info(bar("bar_a", 0.3, y_touch + 0.0035), "bar_a")["inside"])

    def test_openings_of_the_target(self):
        infos = {n: bar_info(s, n) for n, s in target().items()}
        op = openings(infos)
        self.assertEqual(op["order"], ["bar_a", "bar_c", "bar_b"])
        for k, want in (("left_gap", 0.001), ("slit_1", 0.014), ("slit_2", 0.014), ("right_gap", 0.001)):
            self.assertAlmostEqual(op[k]["width"], want, places=9)
        self.assertAlmostEqual(op["slit_1"]["centre"] - op["slit_2"]["centre"], 0.044, places=9)

    def test_touching_bars_leave_a_zero_slit(self):
        bars = by_gaps(gl=0.0, dc=L.TANK_HALF - 2 * LONG_HALF - SHORT_HALF)  # C against A
        s, (S, C, W), d = flags(bars)
        self.assertEqual(d["slits_mm"][0], 0.0)
        self.assertFalse(S)

    def test_validity(self):
        good = bar("bar_a", 0.3, 0.06)
        self.assertTrue(valid(good))
        self.assertFalse(valid(None))
        for key, val in (("pos", np.array([np.nan, 0, 0])), ("quat_xyzw", np.zeros(4)),
                         ("lin_vel", np.array([np.inf, 0, 0])), ("ang_vel", np.array([0, np.nan, 0])),
                         ("low", np.array([0, 0, np.nan])), ("pos", np.zeros(2))):
            bad = SimpleNamespace(**vars(good))
            setattr(bad, key, val)
            self.assertFalse(valid(bad), key)
        missing = SimpleNamespace(pos=good.pos, quat_xyzw=good.quat_xyzw)
        self.assertFalse(valid(missing))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(still(start()))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_target_scores_one(self):
        s, f, d = flags(target())
        self.assertEqual(s, 1.0)
        self.assertEqual(d["slits_mm"], [14.0, 14.0])
        self.assertEqual(d["slit_sep_mm"], 44.0)
        self.assertEqual(d["wall_gaps_mm"], [1.0, 1.0])
        r, _ = run(still(target()))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_pressed_against_both_walls_scores_one(self):
        s, _, d = flags(by_gaps(0.0, 0.0))
        self.assertEqual(s, 1.0)
        self.assertEqual(d["slits_mm"], [15.0, 15.0])
        self.assertEqual(d["slit_sep_mm"], 45.0)

    def test_demo_margins(self):
        """With both long bars against their walls (where the demo drives them), the short bar may land up to 5 mm
        off the centre line; with the short bar centred, either long bar may stop up to 5 mm short of its wall."""
        for dc in (-0.005, -0.003, 0.0, 0.003, 0.005):
            self.assertEqual(flags(by_gaps(0.0, 0.0, dc))[0], 1.0, dc)
        for gl, gr in ((0.005, 0.0), (0.0, 0.005), (0.005, 0.005), (0.003, 0.002)):
            self.assertEqual(flags(by_gaps(gl, gr, 0.0))[0], 1.0, (gl, gr))
        # the offline proxy's end state: both long bars drawn 1.5 mm into their walls (no contacts offline)
        bars = {n: bar(n, L.LINE_X, L.PLACE_Y[n]) for n in L.BAR_NAMES}
        s, _, d = flags(bars)
        self.assertEqual((s, d["wall_gaps_mm"], d["slits_mm"]), (1.0, [0.0, 0.0], [16.5, 16.5]))

    def test_all_eight_combinations(self):
        """Every combination of S, C and W, the score 0.25 more for each; every tier from 0.25 to 1.0 occurs."""
        cases = {
            (True, True, True): by_gaps(),
            (True, True, False): by_gaps(0.006, 0.0, -0.003),  # bar A 6 mm off its wall
            (False, True, True): by_gaps(0.0, 0.0, 0.006),  # short bar 6 mm off centre: slits 9 and 21 mm
            (True, False, True): short_at_left_wall(),  # wrong order: slits 15 and 15 mm, 85 mm apart
            (False, True, False): by_gaps(0.006, 0.0, 0.004),  # slits 5 and 19 mm, wall gap 6 mm
            (False, False, True): short_at_left_wall(slit1=0.0, slit2=0.030),
            (True, False, False): short_at_left_wall(c_gap=0.006, slit1=0.012, slit2=0.012),
            (False, False, False): by_gaps(0.010, 0.010),  # slits 5 mm, centres 35 mm apart, gaps 10 mm
        }
        for want, bars in cases.items():
            s, got, d = flags(bars)
            self.assertTrue(d["line"], want)
            self.assertEqual(got, want)
            self.assertEqual(s, 0.25 * (1 + sum(want)), want)

    def test_the_text_examples(self):
        """The examples in task.json's scoring, number for number."""
        s, _, d = flags(by_gaps(0.006, 0.0, -0.003))
        self.assertEqual((s, d["slits_mm"], d["wall_gaps_mm"]), (0.75, [12.0, 12.0], [6.0, 0.0]))
        s, _, d = flags(by_gaps(0.0, 0.0, 0.006))
        self.assertEqual((s, d["slits_mm"]), (0.75, [9.0, 21.0]))
        s, _, d = flags(short_at_left_wall())
        self.assertEqual((s, d["slits_mm"], d["slit_sep_mm"], d["slit_mid_mm"]), (0.75, [15.0, 15.0], 85.0, 20.0))
        s, _, d = flags(by_gaps(0.006, 0.0, 0.004))
        self.assertEqual((s, d["slits_mm"]), (0.5, [5.0, 19.0]))
        s, _, d = flags(by_gaps(0.010, 0.010))
        self.assertEqual((s, d["slits_mm"], d["slit_sep_mm"]), (0.25, [5.0, 5.0], 35.0))

    def test_limits_are_inclusive(self):
        # S: a slit of exactly 10 and exactly 20 mm (C 5 mm left of centre, walls closed)
        s, (S, C, W), d = flags(by_gaps(0.0, 0.0, 0.005))
        self.assertEqual(d["slits_mm"], [10.0, 20.0])
        self.assertTrue(S)
        s, (S, _, _), _ = flags(by_gaps(0.0, 0.0, 0.0052))
        self.assertFalse(S)
        # W: a wall gap of exactly 5 mm, and 5.2 mm
        self.assertTrue(flags(by_gaps(0.005, 0.0, -0.0025))[1][2])
        self.assertFalse(flags(by_gaps(0.0052, 0.0, -0.0025))[1][2])

    def test_slit_minimum_alone(self):
        # One slit just under 10 mm, the other well inside the window: S is lost on the narrow one alone.
        s, (S, C, W), d = flags(by_gaps(0.0, 0.001, 0.0052))
        self.assertEqual(d["slits_mm"], [9.8, 19.2])
        self.assertFalse(S)
        self.assertTrue(C and W)
        self.assertTrue(flags(by_gaps(0.0, 0.001, 0.005))[1][0])  # 10.0 and 19.0 mm
        # The openings always add up to 30 mm (170 mm of bar in 200 mm), so a slit over 20 mm leaves the other under
        # 10 mm: the 20 mm limit never decides S on its own, but it is the real-world window and is stated.
        for dc in (0.0, 0.003, 0.0052, 0.008):
            d = flags(by_gaps(0.0, 0.0, dc))[2]
            self.assertAlmostEqual(sum(d["slits_mm"]) + sum(d["wall_gaps_mm"]), 30.0, places=6)

    def test_midpoint_limit_alone(self):
        # Walls closed, the short bar far off centre: the spacing stays 45 mm, so C turns on the midpoint alone.
        s, (S, C, W), d = flags(by_gaps(0.0, 0.0, 0.0105))
        self.assertEqual((d["slit_sep_mm"], d["slit_mid_mm"]), (45.0, 5.25))
        self.assertFalse(C)
        s, (S, C, W), d = flags(by_gaps(0.0, 0.0, 0.0099))
        self.assertEqual(d["slit_mid_mm"], 4.95)
        self.assertTrue(C)

    def test_overlapping_outlines_read_as_a_closed_slit(self):
        # A simulator state with the short bar sunk 2 mm into bar A: the slit between them is 0, not 2 mm.
        bars = by_gaps(0.0, 0.0, L.TANK_HALF - 2 * LONG_HALF - SHORT_HALF + 0.002)
        d = flags(bars)[2]
        self.assertEqual(d["slits_mm"][0], 0.0)

    def test_centring_limits(self):
        # C's midpoint: the slits shifted together by moving the whole line sideways (A and B off one wall each)
        s, (S, C, W), d = flags(by_gaps(0.0, 0.008, 0.004))  # gaps 0 and 8, C 4 mm left: midpoint +4.0 mm
        self.assertAlmostEqual(d["slit_mid_mm"], 4.0, places=6)
        self.assertTrue(C)
        s, (S, C, W), d = flags(short_at_left_wall(slit1=0.012, slit2=0.018))
        self.assertGreater(abs(d["slit_mid_mm"]), 5.0)
        self.assertFalse(C)

    def test_score_is_read_at_hand_and_needs_no_history(self):
        t = Tracker(HZ)
        self.assertEqual(t.result(target()).score, 1.0)
        self.assertFalse(t.result(target()).success)  # still is not known yet

    def test_success_needs_stillness(self):
        seq = still(target(), 11)
        moving = by_gaps(dc=0.0025)  # C 2.5 mm across in one step
        r, _ = run(seq + [moving])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r, _ = run(seq + [moving] * 11)
        self.assertTrue(r.success)
        r, _ = run(still(target(), 10))  # 0.5 s not seen yet
        self.assertFalse(r.success)

    def test_score_is_idempotent(self):
        t = Tracker(HZ)
        run(still(target()), t)
        a, b = t.result(target()), t.result(target())
        self.assertEqual((a.score, a.success, a.details), (b.score, b.success, b.details))
        self.assertEqual(t.steps, 12)


class ZeroConditions(unittest.TestCase):
    def assertZero(self, bars):
        s, d = evaluate(bars)
        self.assertFalse(d["line"])
        self.assertEqual(s, 0.0)

    def test_a_bar_left_on_the_table(self):
        bars = target()
        bars["bar_b"] = bar("bar_b", *L.BAR_START["bar_b"], base=TOP)
        self.assertZero(bars)

    def test_a_bar_outside_the_tank_at_the_line(self):
        bars = target()
        bars["bar_a"] = bar("bar_a", L.LINE_X, 0.150, base=TOP)  # beyond the left wall, on the table
        self.assertZero(bars)

    def test_lying_on_a_side_or_standing_on_an_end(self):
        for q in (about("y", 90.0), qmul(about("z", 0.0), about("x", 90.0))):
            bars = target()
            bars["bar_c"] = bar("bar_c", L.LINE_X, 0.0, quat=q)
            self.assertZero(bars)

    def test_tilt_limit(self):
        for axis in ("x", "y"):
            bars = target()
            bars["bar_c"] = bar("bar_c", L.LINE_X, 0.0, tilt=9.5, tilt_axis=axis)
            self.assertTrue(evaluate(bars)[1]["line"], axis)
            bars["bar_c"] = bar("bar_c", L.LINE_X, 0.0, tilt=10.5, tilt_axis=axis)
            self.assertZero(bars)

    def test_raised_off_the_floor(self):
        bars = target()
        bars["bar_a"] = bar("bar_a", L.LINE_X, bars["bar_a"].pos[1], lift=0.0019)
        self.assertEqual(evaluate(bars)[0], 1.0)
        bars["bar_a"] = bar("bar_a", L.LINE_X, bars["bar_a"].pos[1], lift=0.0025)
        self.assertZero(bars)

    def test_turned_off_the_line(self):
        bars = target()
        bars["bar_b"] = bar("bar_b", L.LINE_X, bars["bar_b"].pos[1] + 0.002, turn=4.5)
        self.assertTrue(evaluate(bars)[1]["line"])
        bars["bar_b"] = bar("bar_b", L.LINE_X, bars["bar_b"].pos[1] + 0.003, turn=5.5)
        self.assertZero(bars)

    def test_out_of_line_along_x(self):
        bars = target()
        bars["bar_c"] = bar("bar_c", L.LINE_X + 0.0059, 0.0)
        self.assertTrue(evaluate(bars)[1]["line"])
        bars["bar_c"] = bar("bar_c", L.LINE_X + 0.0065, 0.0)
        self.assertZero(bars)

    def test_distance_from_the_dipper(self):
        for x, ok in ((0.2905, True), (0.3095, True), (0.2895, False), (0.3105, False)):
            s, d = evaluate(by_gaps(x=x))
            self.assertEqual(d["line"], ok, x)
            self.assertEqual(s, 1.0 if ok else 0.0)

    def test_a_missing_or_broken_bar(self):
        bars = target()
        del bars["bar_c"]
        r, t = run(still(target(), 3) + [bars])
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])


class Exploits(unittest.TestCase):
    def test_short_bar_at_a_wall_is_not_centred(self):
        # Two good 15 mm slits and closed walls, but the slits sit 20 mm off the centre line: 0.75, never 1.0.
        r, _ = run(still(short_at_left_wall()))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_closing_a_wall_gap_by_propping_a_bar_on_the_wall(self):
        # Bar A 6 mm short of the left wall, its left end lifted onto the wall's top: the shadow reaches over the
        # wall and closes the gap, its lowest corner still on the floor and tilted under 10 degrees: not standing.
        tilt = math.degrees(math.asin((L.WALL_H - L.FLOOR_T) / (2 * LONG_HALF)))
        self.assertLess(tilt, L.TILT_DEG)
        q = qmul(about("z", 0.0), about("x", tilt))  # +y end up
        a = bar("bar_a", L.LINE_X, L.TANK_HALF - LONG_HALF + 0.004, quat=q)
        info = bar_info(a, "bar_a")
        self.assertTrue(info["on_floor"])
        self.assertFalse(info["inside"])
        bars = by_gaps(0.006, 0.0, 0.0)
        bars["bar_a"] = a
        self.assertEqual(evaluate(bars)[0], 0.0)

    def test_skewing_a_long_bar_to_reach_the_wall(self):
        # A long bar turned 8 degrees spans more of the tank from above (and so closes its wall gap): out of line.
        bars = by_gaps(0.006, 0.0, -0.003)
        bars["bar_a"] = bar("bar_a", L.LINE_X, L.TANK_HALF - 0.006 - LONG_HALF, turn=8.0)
        self.assertEqual(evaluate(bars)[0], 0.0)

    def test_a_staggered_line_is_not_a_line(self):
        # The short bar 10 mm in front of the others: in a real tank it makes a third opening round its ends.
        bars = target()
        bars["bar_c"] = bar("bar_c", L.LINE_X - 0.010, 0.0)
        self.assertEqual(evaluate(bars)[0], 0.0)

    def test_bars_laid_flat_do_not_count(self):
        flat = qmul(about("z", 0.0), about("y", 90.0))  # on a 70 x 25 mm face, 20 mm tall
        bars = {n: bar(n, s.pos[0], s.pos[1], quat=flat) for n, s in target().items()}
        self.assertEqual(evaluate(bars)[0], 0.0)

    def test_stacking_or_resting_on_another_bar(self):
        bars = target()
        bars["bar_c"] = bar("bar_c", L.LINE_X, bars["bar_a"].pos[1], base=FLOOR + L.LONG_SIZE[2])  # on bar A
        self.assertEqual(evaluate(bars)[0], 0.0)

    def test_only_two_bars_in_the_tank(self):
        bars = by_gaps(0.0, 0.0, 0.0)
        bars["bar_c"] = bar("bar_c", *L.BAR_START["bar_c"], base=TOP)  # the short bar never moved
        r, _ = run(still(bars))
        self.assertEqual(r.score, 0.0)

    def test_closing_every_opening_is_not_a_two_slit_barrier(self):
        # All three bars pushed together against the left wall: no slits, a 30 mm gap at the right wall.
        bars = {"bar_a": bar("bar_a", L.LINE_X, L.TANK_HALF - LONG_HALF),
                "bar_c": bar("bar_c", L.LINE_X, L.TANK_HALF - 2 * LONG_HALF - SHORT_HALF),
                "bar_b": bar("bar_b", L.LINE_X, L.TANK_HALF - 3 * LONG_HALF - 2 * SHORT_HALF)}
        s, (S, C, W), d = flags(bars)
        self.assertEqual(d["slits_mm"], [0.0, 0.0])
        self.assertEqual(d["wall_gaps_mm"], [0.0, 30.0])
        self.assertLessEqual(s, 0.25)


class Robustness(unittest.TestCase):
    def test_either_way_up_and_either_end(self):
        bars = {n: bar(n, s.pos[0], s.pos[1], turn=180.0, flip=True) for n, s in target().items()}
        r, _ = run(still(bars))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_any_bar_in_any_place(self):
        b = target()
        swapped = {"bar_a": bar("bar_a", L.LINE_X, b["bar_b"].pos[1]),
                   "bar_b": bar("bar_b", L.LINE_X, b["bar_a"].pos[1]), "bar_c": b["bar_c"]}
        self.assertEqual(evaluate(swapped)[0], 1.0)

    def test_contact_jitter_keeps_success(self):
        rng = np.random.default_rng(0)
        seq = still(target(), 12)
        for _ in range(100):
            seq.append(by_gaps(EXAMPLE_GAP + rng.uniform(-3e-4, 3e-4), EXAMPLE_GAP + rng.uniform(-3e-4, 3e-4),
                               rng.uniform(-3e-4, 3e-4)))
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker(HZ)
        run(still(target(), 12), t)
        before = {n: list(map(np.copy, h)) for n, h in t.history.items()}
        good = target()["bar_a"]
        bads = []
        for key, val in (("pos", np.array([np.nan, 0, 0])), ("quat_xyzw", np.zeros(4)),
                         ("lin_vel", np.array([np.inf, 0, 0])), ("pos", np.array([0, np.inf, 0]))):
            s = SimpleNamespace(**vars(good))
            setattr(s, key, val)
            bads.append(s)
        bads.append(None)
        for s in bads:
            bars = target()
            bars["bar_a"] = s
            r = t.update(bars)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(t.result(bars).score, 0.0)
        self.assertEqual(t.bad_steps, len(bads))
        for n, h in t.history.items():
            for a, b in zip(h, before[n]):
                np.testing.assert_array_equal(a, b)
        r = t.update(target())  # the record of the bars was kept: still success at once
        self.assertTrue(r.success)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(1)
        t = Tracker(HZ)
        for _ in range(300):
            bars = {}
            for n in L.BAR_NAMES:
                q = rng.normal(size=4)
                bars[n] = state(n, rng.uniform([-0.2, -0.5, 0.5], [0.8, 0.5, 1.2]), q / np.linalg.norm(q))
            r = t.update(bars)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_held_just_above_the_floor_counts_as_standing(self):
        # The scorer reads only the bars (fidelity says so): a bar held still 1.5 mm up is standing.
        bars = target()
        bars["bar_c"] = bar("bar_c", L.LINE_X, 0.0, lift=0.0015)
        self.assertEqual(evaluate(bars)[0], 1.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker(HZ)
        run(still(target(), 15), t)
        self.assertTrue(t.last.success)
        r = t.update(start())
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(t.first_full_step)

    def test_teleport_to_a_jittered_start_resets(self):
        t = Tracker(HZ)
        run(still(target(), 15), t)
        jit = {n: bar(n, x + 0.02, y - 0.02, turn=5.0, base=TOP) for n, (x, y) in L.BAR_START.items()}
        t.update(jit)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker(HZ)
        run(still(target(), 15), t)
        a, b = target(), start()
        for k in range(1, 41):
            f = k / 40
            t.update({n: bar(n, *((1 - f) * a[n].pos[:2] + f * b[n].pos[:2]), base=TOP) for n in L.BAR_NAMES})
        self.assertEqual(t.steps, 55)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = Tracker(HZ)
        run(still(target(), 15), t)
        t.update(by_gaps(0.010, 0.010))  # a big move, but not back to the start
        self.assertEqual(t.steps, 16)

    def test_first_steps_are_recorded(self):
        t = Tracker(HZ)
        r, _ = run(still(start(), 5) + still(target(), 12), t)
        self.assertEqual(r.details["first_line_step"], 6)
        self.assertEqual(r.details["first_full_step"], 6)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace, the demo, the task.json text
    and the offline checker on the whole folder."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")

    def setUp(self):
        self.task._TRACKER.reset()

    def test_objects(self):
        objs = self.task.OBJECTS
        names = [o.name for o in objs]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in objs if o.mass is not None], list(L.BAR_NAMES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.5)
        spec = {o.name: o for o in objs}
        for n, (x, y) in L.BAR_START.items():
            np.testing.assert_allclose(spec[n].pos, (x, y, TOP + L.BAR_SIZE[n][2] / 2))
            self.assertEqual(tuple(spec[n].size), L.BAR_SIZE[n])
            self.assertLessEqual(spec[n].mass, 0.3)
            # the arms' start pose covers the table near each base; nothing of the task lies there
            self.assertLess(abs(y) + L.BAR_SIZE[n][1] / 2, 0.45)
        left, right = spec["tank_wall_left"], spec["tank_wall_right"]
        self.assertAlmostEqual(left.pos[1] - left.size[1] / 2, L.TANK_HALF)
        self.assertAlmostEqual(right.pos[1] + right.size[1] / 2, -L.TANK_HALF)
        self.assertAlmostEqual(spec["tank_wall_near"].pos[0] + spec["tank_wall_near"].size[0] / 2, L.TANK_X0)
        self.assertAlmostEqual(spec["tank_wall_far"].pos[0] - spec["tank_wall_far"].size[0] / 2, L.TANK_X1)
        floor = spec["tank_floor"]
        self.assertAlmostEqual(floor.pos[2] + floor.size[2] / 2, FLOOR)
        self.assertAlmostEqual(spec["dipper"].pos[0], L.DIPPER_X)
        for n in names:
            if n not in L.BAR_NAMES:
                self.assertIsNone(spec[n].mass, n)
        # the bars fill 170 of the 200 mm: 30 mm for the two slits
        self.assertAlmostEqual(2 * L.TANK_HALF - 2 * L.LONG_SIZE[1] - L.SHORT_SIZE[1], 0.030)

    def test_hooks(self):
        t = self.task
        self.assertFalse(t.success(start()))
        self.assertEqual(t.score(start()), 0.0)
        t._TRACKER.reset()
        ok = False
        for _ in range(12):
            ok = t.success(target())
        self.assertTrue(ok)
        self.assertEqual(t.score(target()), 1.0)
        self.assertFalse(t.success({"bar_a": target()["bar_a"]}))  # missing bars: no crash, 0
        self.assertEqual(t.score({"bar_a": target()["bar_a"]}), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        tmp = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(tmp)
            self.task.success(target())
            row = json.loads(tmp.read_text().splitlines()[-1])
            for key in ("step", "score", "success", "line", "slits_mm", "slit_sep_mm", "slit_mid_mm", "wall_gaps_mm",
                        "inline_spread_mm", "moved_mm", "bar_a", "bar_a_pos", "bar_c_quat", "order"):
                self.assertIn(key, row)
            for key in ("y_mm", "low_mm", "tilt_deg", "axis_err_deg", "beyond_dipper_mm"):
                self.assertIn(key, row["bar_b"])
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(target()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
            if tmp.exists():
                tmp.unlink()
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        t_of = {a: np.cumsum([w.duration for w in plan[a]]) for a in plan}

        def when(arm, start_of):
            i = next(i for i, w in enumerate(plan[arm]) if w.label.startswith(start_of))
            return t_of[arm][i] - plan[arm][i].duration, t_of[arm][i]

        # one gripper in the tank at a time: b goes in once the left has risen from a, c once the right has from b
        self.assertGreaterEqual(when("right", "carry bar_b")[0], when("left", "straight up")[1] - 1e-9)
        b_up = [t_of["right"][i] for i, w in enumerate(plan["right"]) if w.label.startswith("straight up")][0]
        self.assertGreaterEqual(when("left", "carry bar_c")[0], b_up - 1e-9)
        # every open or close takes 1.5 s or more; every final set-down is slow
        for wps in plan.values():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
            for w in wps:
                if w.label.startswith("slowly down"):
                    self.assertGreaterEqual(w.duration, 1.0, w.label)
        # each long bar comes down 6 mm in from its wall, slides out below the wall's top, along y, toward a point
        # 1.5 mm past the wall's face (the wall stops it; under 2 mm, so the offline checker drops it on the floor)
        for arm, name in (("left", "bar_a"), ("right", "bar_b")):
            i = next(i for i, w in enumerate(plan[arm]) if w.label.startswith(f"slide {name}"))
            w, prev = plan[arm][i], plan[arm][i - 1]
            self.assertEqual(w.finger, t.GRIP)
            self.assertAlmostEqual(prev.pos[1], L.SET_Y[name])
            self.assertAlmostEqual(w.pos[1], L.PLACE_Y[name])
            self.assertEqual((prev.pos[0], prev.pos[2]), (w.pos[0], w.pos[2]))
            self.assertAlmostEqual(w.pos[2], t.SLIDE_Z)
            bottom = w.pos[2] - t.GRIP_UP - TOP
            self.assertLess(bottom, L.WALL_H - 0.005)  # the bar's bottom 8 mm below the wall's top
            self.assertGreater(bottom, L.FLOOR_TOP)
            self.assertAlmostEqual(abs(w.pos[1]) + LONG_HALF, L.TANK_HALF + L.PUSH_PAST)
            self.assertLess(L.PUSH_PAST, 0.002)
            self.assertLess(L.PUSH_PAST, L.INSIDE_TOL)
            self.assertAlmostEqual(abs(w.pos[1]) - abs(prev.pos[1]), L.SET_WALL_GAP + L.PUSH_PAST)
        # the pads (8 mm thick, 36 mm apart) stay in front of and behind the line, clear of the bars already set
        self.assertGreater(-t.RELEASE - L.LONG_SIZE[0] / 2, 0.005)
        # every grip across a bar's 20 mm thickness, commanded 8 mm narrower
        self.assertAlmostEqual(-2 * t.GRIP, L.LONG_SIZE[0] - 0.008)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_task_json_matches_the_task(self):
        d = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "reset_notes",
                  "context", "keywords"):
            self.assertTrue(d.get(k), k)
        self.assertEqual(d["instruction"], self.task.INSTRUCTION)
        # the line's distance is stated the way the scorer reads it: bar centres from the dipper's centre line
        self.assertIn("120 mm beyond it (centre line to centre line)", d["instruction"])
        self.assertAlmostEqual(L.LINE_X - L.DIPPER_X, 0.120)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{d["instruction"]}"'), 2)
        self.assertIn("run dry", d["fidelity"])
        self.assertEqual(d["category"], "Impact on jobs")
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        s = d["scoring"]
        for phrase in ("2 mm of the tank floor", "10 degrees", "5 degrees", "6 mm", "110 to 130 mm", "x 29.0 to 31.0 cm",
                       "10 to 20 mm", "40 to 50 mm", "within 5 mm", "5 mm or less", "3 mm allowed", "2.5 s",
                       "Possible scores: 0, 0.25, 0.5, 0.75, 1.0"):
            self.assertIn(phrase, s)
        for x, y in L.BAR_START.values():
            for k in ("setup", "reset_notes"):
                self.assertIn(f"x {x * 100:.1f} cm and y {y * 100:+.1f} cm", d[k])
        self.assertIn("1,800 steps", d["run_limit"])
        self.assertIn("70 mm long, 20 mm thick and 25 mm tall", d["setup"])
        self.assertIn("95 g", d["setup"])
        self.assertIn("41 g", d["setup"])

    def test_offline_checker_predicts_one(self):
        """tools/offline/check_task.py --strict --one-arm on this folder (no reach table, for time). Run in-process
        with sys.modules and sys.path restored afterwards (the checker reloads the task's helper modules)."""
        saved_modules, saved_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(_offline_dir()))
        try:
            import check_task
            rep, _, code = check_task.check(HERE, strict=True, one_arm=True, reach=False, require_json=True)
            self.assertEqual(code, 0, rep.fails)
            self.assertEqual(rep.data["playback"]["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(rep.data["one_arm"], {"left_only": 0.0, "right_only": 0.0})
            self.assertEqual(rep.data["playback"]["proxy"]["interference"], [])
            check_task._forget_task_modules()
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path


if __name__ == "__main__":
    unittest.main()
