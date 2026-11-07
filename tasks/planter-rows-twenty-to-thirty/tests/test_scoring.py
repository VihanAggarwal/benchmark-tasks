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
    Tracker, covers_mark, evaluate, lowest, off_table, on_stand, on_toolbar, rotation, set_on_bar, tilt_deg,
    valid, yaw_err_deg,
)

MM = 0.001
T0, T1 = L.TARGETS  # 95.25 and 285.75 mm from the mark


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
    """A unit's state: its world bounding box worked out from the turned box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ np.array(L.UNIT_HALF)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def unit(y, x=L.BAR_X, bottom=L.BAR_TOP, yaw=0.0, tilt=0.0, axis="y", **kw):
    """A unit at (x, y) with its lowest point at ``bottom``, turned ``yaw`` degrees about the vertical after being
    tilted ``tilt`` degrees about the world ``axis``."""
    q = qmul(about("z", yaw), about(axis, tilt))
    ext = np.abs(rotation(q)) @ np.array(L.UNIT_HALF)
    return state((x, y, bottom + ext[2]), q, **kw)


def on_stand_state(sign, dx=0.0, dy=0.0, **kw):
    return unit(sign * L.STAND_Y + dy, x=L.STAND_X + dx, bottom=L.STAND_TOP, **kw)


def start():
    """The starting layout: every unit square on the toolbar at its 20 in mark (order of L.NAMES)."""
    return [unit(y) for _, y in L.UNITS]


S0 = start()
NAME_INDEX = {n: i for i, n in enumerate(L.NAMES)}


def layout_of(left_bar=(T0, T1), right_bar=(T0, T1), left_spare="stand", right_spare="stand", **over):
    """Units l1, l3 (left) and r1, r3 (right) on the toolbar at the given distances from the mark (each side's
    first entry goes to its inner unit), the middle units l2 and r2 as spares ('stand', 'table', 'start' or a state).
    ``over``: name -> state replaces that unit."""
    out = list(S0)
    for (inner, outer), sign, tag in ((left_bar, 1.0, "l"), (right_bar, -1.0, "r")):
        if inner is not None:
            out[NAME_INDEX[f"unit_{tag}1"]] = unit(sign * inner)
        if outer is not None:
            out[NAME_INDEX[f"unit_{tag}3"]] = unit(sign * outer)
    for spare, sign, tag in ((left_spare, 1.0, "l"), (right_spare, -1.0, "r")):
        i = NAME_INDEX[f"unit_{tag}2"]
        if spare == "stand":
            out[i] = on_stand_state(sign)
        elif spare == "table":
            out[i] = unit(sign * 0.38, x=0.40, bottom=L.TOP)
        elif spare == "start":
            out[i] = S0[i]
        else:
            out[i] = spare
    for n, s in over.items():
        out[NAME_INDEX[n]] = s
    return out


DONE = layout_of()


def score_of(units):
    return evaluate(units)[0]


def glide(a, b, n):
    """n frames moving every unit of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None, hold=30):
    """From the start, every unit moved gradually to the four-row layout (or ``final``), then held there."""
    final = DONE if final is None else final
    return [S0] * 5 + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


class Geometry(unittest.TestCase):
    def test_lowest_point_and_angles(self):
        s = unit(0.1)
        self.assertAlmostEqual(lowest(s), L.BAR_TOP)
        self.assertAlmostEqual(tilt_deg(s.quat_xyzw), 0.0)
        tipped = unit(0.1, tilt=90.0, axis="y")  # tipped over forward onto its end: the long side vertical
        self.assertAlmostEqual(lowest(tipped), L.BAR_TOP)
        self.assertAlmostEqual(tilt_deg(tipped.quat_xyzw), 90.0, places=5)
        upside_down = unit(0.1, tilt=180.0, axis="x")
        self.assertAlmostEqual(tilt_deg(upside_down.quat_xyzw), 0.0, places=5)  # top and bottom are alike
        self.assertAlmostEqual(lowest(upside_down), L.BAR_TOP)
        self.assertAlmostEqual(tilt_deg(unit(0.1, tilt=4.0, axis="x").quat_xyzw), 4.0, places=5)

    def test_squareness(self):
        self.assertAlmostEqual(yaw_err_deg(unit(0.1, yaw=7.5).quat_xyzw), 7.5, places=5)
        self.assertAlmostEqual(yaw_err_deg(unit(0.1, yaw=-7.5).quat_xyzw), 7.5, places=5)
        self.assertAlmostEqual(yaw_err_deg(unit(0.1, yaw=180.0).quat_xyzw), 0.0, places=5)  # either end forward
        self.assertAlmostEqual(yaw_err_deg(unit(0.1, yaw=175.0).quat_xyzw), 5.0, places=5)
        self.assertAlmostEqual(yaw_err_deg(unit(0.1, yaw=90.0).quat_xyzw), 90.0, places=5)
        standing_on_end = unit(0.1, tilt=90.0, axis="y")  # long side vertical
        self.assertEqual(yaw_err_deg(standing_on_end.quat_xyzw), 90.0)

    def test_on_toolbar_and_set(self):
        self.assertTrue(set_on_bar(unit(T0)))
        self.assertTrue(set_on_bar(unit(T0, bottom=L.BAR_TOP + 0.0029)))  # held 2.9 mm up: counts (fidelity)
        self.assertFalse(on_toolbar(unit(T0, bottom=L.BAR_TOP + 0.0035)))
        self.assertTrue(set_on_bar(unit(T0, x=L.BAR_X + 0.0119)))
        self.assertFalse(set_on_bar(unit(T0, x=L.BAR_X + 0.0125)))  # 12.5 mm off the centre line
        self.assertTrue(on_toolbar(unit(T0, x=L.BAR_X + 0.0125)))  # ... but still resting on the bar
        self.assertFalse(on_toolbar(unit(T0, x=L.BAR_X + 0.021)))  # centre past the bar's edge
        self.assertFalse(on_toolbar(unit(0.355)))  # past the bar's end
        self.assertTrue(set_on_bar(unit(T0, yaw=7.9)))
        self.assertFalse(set_on_bar(unit(T0, yaw=8.5)))
        self.assertTrue(on_toolbar(unit(T0, yaw=90.0)))  # turned across the bar: on it, not set
        self.assertTrue(set_on_bar(unit(T0, tilt=4.9, axis="x")))
        self.assertFalse(set_on_bar(unit(T0, tilt=5.5, axis="x")))
        lying = unit(T0, tilt=90.0, axis="x")  # on its side, on the bar
        self.assertTrue(on_toolbar(lying))
        self.assertFalse(set_on_bar(lying))
        on_another = unit(T0, bottom=L.BAR_TOP + L.UNIT_SIZE[2])  # stacked on a unit on the bar
        self.assertFalse(on_toolbar(on_another))
        self.assertFalse(on_toolbar(unit(T0, x=0.40, bottom=L.TOP)))  # on the table

    def test_mark(self):
        self.assertTrue(covers_mark(unit(0.0)))
        self.assertTrue(covers_mark(unit(0.0229)))
        self.assertTrue(covers_mark(unit(-0.0229, yaw=90.0)))
        self.assertFalse(covers_mark(unit(0.0231)))
        self.assertFalse(covers_mark(unit(0.0, bottom=L.BAR_TOP + 0.05)))  # held above the mark
        self.assertFalse(covers_mark(unit(0.0, x=0.23, bottom=L.TOP + L.HITCH_SIZE[2])))  # on the hitch tongue

    def test_stand(self):
        self.assertTrue(on_stand(on_stand_state(1), L.STAND_Y))
        self.assertFalse(on_stand(on_stand_state(1), -L.STAND_Y))
        self.assertTrue(on_stand(on_stand_state(1, dx=0.044, dy=-0.044), L.STAND_Y))  # centre over its corner
        self.assertFalse(on_stand(on_stand_state(1, dx=0.046), L.STAND_Y))
        self.assertFalse(on_stand(on_stand_state(1, tilt=90.0, axis="x"), L.STAND_Y))  # lying on its side
        self.assertTrue(on_stand(on_stand_state(1, yaw=90.0), L.STAND_Y))  # turned is fine on a stand
        self.assertFalse(on_stand(unit(L.STAND_Y, x=L.STAND_X, bottom=L.STAND_TOP + 0.004), L.STAND_Y))

    def test_off_table(self):
        self.assertFalse(off_table(unit(0.4, x=0.4, bottom=L.TOP)))
        self.assertTrue(off_table(unit(0.4, x=0.4, bottom=L.TOP - 0.011)))
        self.assertTrue(off_table(unit(0.8, x=0.4, bottom=0.02)))

    def test_valid(self):
        self.assertTrue(valid(S0[0]))
        nan = float("nan")
        self.assertFalse(valid(unit(nan)))
        self.assertFalse(valid(state((0.3, 0, 0.8), quat=(0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        self.assertFalse(valid(state((0.3, 0, 0.8), vel=(0, float("inf"), 0))))

    def test_target_arithmetic(self):
        """Four rows at 30 in, symmetric, none on the centre: half and one and a half spacings from the mark; the six
        20 in rows started at half, one and a half and two and a half 20 in spacings."""
        self.assertAlmostEqual(T0, 0.09525)
        self.assertAlmostEqual(T1, 0.28575)
        self.assertAlmostEqual(T1 - T0, L.NEW_PITCH)
        self.assertAlmostEqual(2 * T0, L.NEW_PITCH)  # the two middle rows straddle the mark
        ys = sorted(y for _, y in L.UNITS)
        self.assertTrue(np.allclose(np.diff(ys), L.OLD_PITCH))
        self.assertAlmostEqual(6 * L.OLD_PITCH, 4 * L.NEW_PITCH)  # both span 120 in at 1:4
        # The old inner and outer positions are 31.75 mm from the new ones: outside the 25 mm band of 0.25.
        self.assertAlmostEqual(T0 - 0.0635, 0.03175)
        self.assertAlmostEqual(0.3175 - T1, 0.03175)
        self.assertGreater(0.03175, L.LOOSE)
        self.assertAlmostEqual(L.MARK_COVER, 0.023)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual(score, 0.0)
        self.assertEqual((d["left"]["n_on_toolbar"], d["right"]["n_on_toolbar"]), (3, 3))
        r, _ = run([S0] * 20)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_both_sides_done_scores_one(self):
        score, d = evaluate(DONE)
        self.assertEqual(score, 1.0)
        self.assertEqual((d["left_side"], d["right_side"]), (0.5, 0.5))
        r, _ = run(build())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_one_side_done_scores_half(self):
        self.assertEqual(score_of(layout_of(right_bar=(0.0635, 0.3175), right_spare="start")), 0.5)  # right as found
        self.assertEqual(score_of(layout_of(left_bar=(0.0635, 0.3175), left_spare="start")), 0.5)

    def test_one_side_done_other_within_25mm_scores_three_quarters(self):
        self.assertEqual(score_of(layout_of(right_bar=(T0 + 0.020, T1 - 0.015))), 0.75)
        self.assertEqual(score_of(layout_of(right_spare="table")), 0.75)  # right in place, its spare left on the table

    def test_both_within_25mm_scores_half(self):
        self.assertEqual(score_of(layout_of(left_spare="table", right_spare="table")), 0.5)
        self.assertEqual(score_of(layout_of(left_bar=(T0 - 0.024, T1 + 0.024), right_bar=(T0 + 0.024, T1 - 0.024))),
                         0.5)

    def test_one_side_within_25mm_scores_a_quarter(self):
        self.assertEqual(score_of(layout_of(left_spare="table", right_bar=(0.0635, 0.3175), right_spare="start")),
                         0.25)
        self.assertEqual(score_of(layout_of(left_bar=(T0 + 0.015, T1), right_bar=(0.0635, 0.3175),
                                            right_spare="start")), 0.25)

    def test_a_quarter_needs_one_job_done_in_full(self):
        """0.25 needs the rows within 25 mm and either both within 10 mm (stand empty) or the stand filled. A row 10
        to 25 mm off with the spare not on its stand earns nothing for that side."""
        for off, spare, want in ((0.015, "stand", 0.75), (0.015, "table", 0.5), (0.0249, "table", 0.5),
                                 (0.0099, "table", 0.75), (0.0, "table", 0.75)):
            with self.subTest(off=off, spare=spare):
                self.assertEqual(score_of(layout_of(left_bar=(T0 + off, T1), left_spare=spare)), want)
                self.assertEqual(score_of(layout_of(left_bar=(T0, T1 - off), left_spare=spare)), want)
        _, d = evaluate(layout_of(left_bar=(T0 + 0.015, T1), left_spare="table"))
        self.assertEqual(d["left_side"], 0.0)
        self.assertFalse(d["left"]["stand_filled"])
        self.assertTrue(d["left"]["all_set"])

    def test_tight_and_loose_limits(self):
        for off, want in ((0.0099, 1.0), (0.0101, 0.75), (0.0249, 0.75), (0.0251, 0.5)):
            with self.subTest(off=off):
                self.assertEqual(score_of(layout_of(left_bar=(T0 + off, T1))), want)
                self.assertEqual(score_of(layout_of(left_bar=(T0, T1 - off))), want)
                self.assertEqual(score_of(layout_of(left_bar=(T0 - off, T1 + off))), want)

    def test_the_old_pitch_scores_zero(self):
        """Spares off but the four left where they were (63.5 and 317.5 mm: 31.75 mm off each target): 0."""
        self.assertEqual(score_of(layout_of(left_bar=(0.0635, 0.3175), right_bar=(0.0635, 0.3175))), 0.0)

    def test_four_at_the_old_20in_positions_score_zero(self):
        """Four rows kept at 20 in (63.5 and 190.5 mm), the outer units to the stands: 0."""
        units = list(S0)
        for tag, sign in (("l", 1.0), ("r", -1.0)):
            units[NAME_INDEX[f"unit_{tag}3"]] = on_stand_state(sign)
        self.assertEqual(score_of(units), 0.0)

    def test_three_units_on_a_side_score_zero_for_it(self):
        self.assertEqual(score_of(layout_of(left_spare=unit(0.19))), 0.5)  # left spare left on the bar
        self.assertEqual(score_of(layout_of(left_spare=unit(0.19, tilt=90.0, axis="x"))), 0.5)  # lying on it

    def test_a_unit_on_the_mark_scores_zero(self):
        """Three rows at 30 in (0 and +-190.5 mm), the odd-row layout: 0. So is any unit left covering the mark."""
        units = list(S0)
        units[NAME_INDEX["unit_l1"]] = unit(0.0)
        units[NAME_INDEX["unit_l2"]] = unit(L.NEW_PITCH)
        units[NAME_INDEX["unit_r2"]] = unit(-L.NEW_PITCH)
        units[NAME_INDEX["unit_l3"]] = on_stand_state(1)
        units[NAME_INDEX["unit_r3"]] = on_stand_state(-1)
        units[NAME_INDEX["unit_r1"]] = unit(-0.38, x=0.40, bottom=L.TOP)
        score, d = evaluate(units)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["covering_mark"], ["unit_l1"])
        # A perfect layout with a seventh row on the mark cannot be built (six units), so: done left, a unit on
        # the mark taken from the right.
        on_mark = layout_of(right_bar=(T0, None), right_spare="stand", unit_r3=unit(-0.010))
        self.assertEqual(score_of(on_mark), 0.0)
        self.assertEqual(score_of(layout_of(right_bar=(T0, None), unit_r3=unit(-0.0235))), 0.5)  # next to it: right 0

    def test_a_unit_off_the_table_scores_zero(self):
        dropped = layout_of(left_spare=unit(0.5, x=0.4, bottom=0.05))
        score, d = evaluate(dropped)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["off_table"], ["unit_l2"])

    def test_a_unit_not_set_square_scores_zero_for_its_side(self):
        for bad in (unit(T1, yaw=12.0), unit(T1, yaw=90.0), unit(T1, tilt=8.0, axis="x"),
                    unit(T1, x=L.BAR_X + 0.015), unit(T1, tilt=90.0, axis="x")):
            with self.subTest():
                self.assertEqual(score_of(layout_of(unit_l3=bad)), 0.5)

    def test_wrong_side_stand(self):
        """Both spares on the left stand (side by side): the right side's stand is empty, so the right scores 0.25."""
        both = layout_of(left_spare=on_stand_state(1, dy=0.020), right_spare=on_stand_state(1, dy=-0.020))
        self.assertEqual(score_of(both), 0.75)

    def test_every_tier_turns_up(self):
        seen = {score_of(f) for f in (S0, layout_of(left_spare="table", right_bar=(0.0635, 0.3175), right_spare="start"),
                                      layout_of(right_bar=(0.0635, 0.3175), right_spare="start"),
                                      layout_of(right_spare="table"), DONE)}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_any_unit_may_take_any_place(self):
        """The units are identical: l1 and l3 swapped, or the left middle unit set as a row and the left outer one on
        the stand, still score 1.0."""
        swapped = layout_of(unit_l1=unit(T1), unit_l3=unit(T0))
        self.assertEqual(score_of(swapped), 1.0)
        other = layout_of(left_spare=unit(T1), unit_l3=on_stand_state(1))
        self.assertEqual(score_of(other), 1.0)

    def test_success_needs_stillness(self):
        frames = build(hold=0)
        moving = [[state(s.pos + np.array([0, 0.0012 * (k % 2), 0]), s.quat_xyzw) for s in DONE] for k in range(20)]
        r, _ = run(frames + moving)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)  # every unit shuttling 1.2 mm a step: 11 mm of path in 0.5 s
        r, _ = run(frames + [DONE] * 9)
        self.assertFalse(r.success)  # 0.5 s of stillness not yet seen
        r, _ = run(frames + [DONE] * 11)
        self.assertTrue(r.success)
        r, _ = run([DONE] * 5)  # under 0.5 s of history: not still yet
        self.assertFalse(r.success)

    def test_score_is_the_state_at_hand(self):
        r, t = run(build() + [S0[:1] + list(DONE[1:])] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.5)  # l1 back at 63.5 mm: the left side has three units


class Exploits(unittest.TestCase):
    def test_sweeping_the_spares_off_the_table(self):
        """Pushing the middle units off the table instead of standing them on their stands: 0."""
        self.assertEqual(score_of(layout_of(left_spare=unit(0.2, x=0.7, bottom=0.0),
                                            right_spare=unit(-0.2, x=0.7, bottom=0.0))), 0.0)

    def test_knocking_the_spares_off_the_bar_and_nudging_the_rows(self):
        """The cheapest near-miss: each middle unit pushed off the bar onto the table, each inner unit pushed out and
        each outer one in by 7 mm (24.75 mm short of its target, inside the 25 mm band). Neither job is done in full:
        0. Done with the rows pushed all the way (within 10 mm) it is the rows-done partial, 0.25 a side."""
        nudged = layout_of(left_bar=(0.0635 + 0.007, 0.3175 - 0.007), right_bar=(0.0635 + 0.007, 0.3175 - 0.007),
                           left_spare="table", right_spare="table")
        score, d = evaluate(nudged)
        self.assertEqual(score, 0.0)
        self.assertLessEqual(d["left"]["inner_off_mm"], 25.0)
        self.assertEqual(score_of(layout_of(left_spare="table", right_spare="table")), 0.5)

    def test_stacking_a_spare_on_a_row(self):
        """The spare left on top of a unit on the bar is off the toolbar but not on its stand: 0.25 for that side."""
        stacked = unit(T1, bottom=L.BAR_TOP + L.UNIT_SIZE[2])
        self.assertEqual(score_of(layout_of(left_spare=stacked)), 0.75)

    def test_spares_on_the_hitch_or_the_table(self):
        on_hitch = unit(0.0, x=L.HITCH_X, bottom=L.TOP + L.HITCH_SIZE[2])
        self.assertEqual(score_of(layout_of(left_spare=on_hitch)), 0.75)
        self.assertEqual(score_of(layout_of(left_spare="table", right_spare="table")), 0.5)

    def test_spare_laid_on_its_stand(self):
        self.assertEqual(score_of(layout_of(left_spare=on_stand_state(1, tilt=90.0, axis="x"))), 0.75)

    def test_rows_on_the_table_in_front_of_the_bar(self):
        """All six taken off the bar and four stood on the table at the right distances: no units on the toolbar."""
        units = [unit(y, x=0.40, bottom=L.TOP) for y in (T0, T1, -T0, -T1)] + [on_stand_state(1), on_stand_state(-1)]
        self.assertEqual(score_of(units), 0.0)

    def test_rows_turned_across_the_bar(self):
        units = layout_of(**{n: unit(y * (1 if n[5] == "l" else -1), yaw=90.0)
                             for n, y in (("unit_l1", T0), ("unit_l3", T1), ("unit_r1", T0), ("unit_r3", T1))})
        self.assertEqual(score_of(units), 0.0)

    def test_rows_held_above_the_bar(self):
        """Units held 5 mm above their places are not on the toolbar: each side has none there, 0."""
        held = layout_of(**{n: unit(y, bottom=L.BAR_TOP + 0.005)
                            for n, y in (("unit_l1", T0), ("unit_l3", T1), ("unit_r1", -T0), ("unit_r3", -T1))})
        self.assertEqual(score_of(held), 0.0)

    def test_moving_a_unit_across_the_mark(self):
        """The right arm takes the left inner unit to the right side: the right side then has three units, the left
        one; both 0."""
        units = layout_of(left_bar=(None, T1), unit_l1=unit(-0.15))
        self.assertEqual(score_of(units), 0.0)

    def test_one_arm_alone(self):
        """Gripping from above, one arm resets only its own side: 0.5. The right arm can also reach the left inner
        unit (to y +99 mm), but taking it off or moving it leaves the left side 0."""
        right_only = layout_of(left_bar=(0.0635, 0.3175), left_spare="start")
        self.assertEqual(score_of(right_only), 0.5)
        moved = layout_of(left_bar=(T0, 0.3175), left_spare="start")  # left inner moved to 95.25 by the right arm
        self.assertEqual(score_of(moved), 0.5)
        removed = layout_of(left_bar=(None, 0.3175), left_spare="start", unit_l1=on_stand_state(-1, dy=0.03))
        self.assertEqual(score_of(removed), 0.5)  # left: 190.5 and 317.5 mm, 95 mm off the inner target


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
        scattered = [unit(y, x=0.45, bottom=L.TOP) for y in (0.5, 0.4, 0.3, -0.3, -0.4, -0.5)]
        r, t = run(build() + [scattered] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.0)

    def test_a_unit_falling_back_onto_its_mark_is_not_a_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 30)
        y0 = L.UNITS[0][1]
        falling = unit(y0, bottom=L.BAR_TOP + 0.10, vel=(0, 0, -1.4))
        _, t = run(frames + [[falling, *S0[1:]], S0])
        self.assertEqual(t.best, 1.0)
        resting = unit(y0, bottom=L.BAR_TOP + 0.10)  # the same 10 cm jump by a unit that was still: a teleport
        _, t = run(frames + [[resting, *S0[1:]], S0])
        self.assertEqual(t.best, 0.0)

    def test_marks_need_every_unit_seated_on_its_mark(self):
        self.assertTrue(Tracker.at_start(S0))
        y0 = L.UNITS[0][1]
        self.assertFalse(Tracker.at_start([unit(y0, tilt=3, axis="x"), *S0[1:]]))
        self.assertFalse(Tracker.at_start([unit(y0, bottom=L.BAR_TOP + 0.01), *S0[1:]]))
        self.assertFalse(Tracker.at_start([*S0[:5], on_stand_state(-1)]))
        self.assertFalse(Tracker.at_start([*S0[:5], unit(-0.3175, vel=(0, 0, 0.05))]))


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        good = DONE
        return [
            [unit(nan), *good[1:]],
            [good[0], state((0.3, 0, L.TOP + 0.05), quat=(nan, 0, 0, nan)), *good[2:]],
            [*good[:5], unit(-0.2, vel=(0, inf, 0))],
            [*good[:5], unit(-0.2, spin=(nan, 0, 0))],
            [state((0.3, 0.0, L.TOP + 0.05), quat=(0, 0, 0, 0)), *good[1:]],
            [SimpleNamespace(pos=np.zeros(3)), *good[1:]],
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
        for _ in range(80):
            units = [state(rng.uniform(-2, 2, 3), tuple(rng.normal(size=4))) for _ in range(6)]
            r = Tracker().update(units)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)

    def test_random_states_near_the_layout(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(600):
            units = []
            for s in DONE:
                x, y, _ = s.pos
                on_bar = abs(x - L.BAR_X) < 0.05
                units.append(unit(y + rng.normal(0, 0.012), x=x + rng.normal(0, 0.004),
                                  bottom=(L.BAR_TOP if on_bar else L.STAND_TOP) + abs(rng.normal(0, 0.0012)),
                                  yaw=rng.normal(0, 3), tilt=abs(rng.normal(0, 1.5)), axis="x"))
            score, d = evaluate(units)
            seen.add(score)
            self.assertIn(score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(d, allow_nan=False)
            if score == 1.0:
                for side in ("left", "right"):
                    self.assertLessEqual(d[side]["inner_off_mm"], 10.05)
                    self.assertLessEqual(d[side]["outer_off_mm"], 10.05)
                    self.assertTrue(d[side]["stand_filled"])
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

    def objects(self, units):
        return dict(zip(self.task.UNIT_NAMES, units, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.UNIT_NAMES, L.NAMES)
        for spec, s in zip(t.UNITS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((tuple(spec.size), spec.mass), (L.UNIT_SIZE, L.UNIT_MASS))
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 3 + 1 + 2 + 6)
        # The toolbar's three boxes meet with one flush top, end to end from y -350 to +350 mm.
        pieces = sorted([*t.BAR_HALVES, t.MARK], key=lambda b: b.pos[1])
        edges = [(b.pos[1] - b.size[1] / 2, b.pos[1] + b.size[1] / 2) for b in pieces]
        self.assertAlmostEqual(edges[0][0], -L.BAR_HALF_LEN)
        self.assertAlmostEqual(edges[-1][1], L.BAR_HALF_LEN)
        for (_, a1), (b0, _) in zip(edges, edges[1:]):
            self.assertAlmostEqual(a1, b0)
        for b in pieces:
            self.assertAlmostEqual(b.pos[2] + b.size[2] / 2, L.BAR_TOP)
            self.assertAlmostEqual(b.pos[0], L.BAR_X)
        self.assertAlmostEqual(t.HITCH.pos[0] + t.HITCH.size[0] / 2, L.BAR_X - L.BAR_W / 2)  # butts the bar
        # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
        for o in t.OBJECTS:
            self.assertLess(abs(o.pos[1]) + o.size[1] / 2, 0.45, o.name)

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
        for key in ("step", "score", "success", "left_side", "right_side", "left", "right", "covering_mark",
                    "off_table", "all_still", "bad_steps", "first_step", "unit_l1", "unit_r3"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["left"]["inner_off_mm"], 0.0)
        self.assertEqual(rows[-1]["unit_l1"]["from_mark_mm"], 95.2)
        self.assertEqual(rows[-1]["unit_l1"]["path_mm"], 0.0)
        self.assertTrue(rows[-1]["unit_l1"]["set"])
        self.assertIsNone(rows[0]["unit_l1"]["path_mm"])

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
                self.assertFalse(self.task.success(self.objects([unit(float("nan")), *S0[1:]])))
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

    def test_demo_builds_the_four_row_layout(self):
        t = self.task
        final = list(S0)
        for arm, steps in t.PLAN.items():
            sign = 1.0 if arm == "left" else -1.0
            for name, where in steps:
                self.assertTrue(name.startswith(f"unit_{arm[0]}"))  # each arm resets only its own side
                if where == "stand":
                    final[NAME_INDEX[name]] = on_stand_state(sign)
                else:
                    self.assertGreater(sign * where, 0.0)
                    final[NAME_INDEX[name]] = unit(where)
        self.assertEqual(score_of(final), 1.0)

    @staticmethod
    def spells(wps):
        """(start, end, y) of each waypoint, on the plan's clock."""
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w.pos[1], w.pos[0]))
            t += w.duration
        return out

    def test_grippers_stay_apart(self):
        """Whenever both grippers work over the toolbar at once, they are at least 0.20 m apart along it."""
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, ya, xa in left:
            for b0, b1, yb, xb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400)  # inside the run limit with the policy's 3 s hold

    def test_carried_units_clear_the_others(self):
        t = self.task
        self.assertGreaterEqual(t.HOVER_Z - t.GRIP_UP - (L.BAR_TOP + L.UNIT_SIZE[2]), 0.030)  # over standing tops
        self.assertGreaterEqual(t.SHIFT_Z - t.GRIP_UP - L.BAR_TOP, 0.025)  # off the bar when moved along it
        self.assertGreaterEqual(t.PICK_Z - 0.013 - L.BAR_TOP, 0.012)  # pad tips over the bar's top
        self.assertGreaterEqual(t.STAND_Z - 0.013 - L.STAND_TOP, 0.012)  # ... and over the stand's
        self.assertGreater(-2 * t.OPEN, L.UNIT_SIZE[1] + 0.040)  # 29 mm either side of a unit when open
        self.assertAlmostEqual(-2 * t.GRIP, L.UNIT_SIZE[1] - 0.008)  # the grip is commanded 8 mm narrower
        # Open pads (gap plus two 8 mm pads) clear a neighbour 127 mm away at the start by more than 40 mm.
        self.assertGreater(L.OLD_PITCH - L.UNIT_HALF[1] - (-t.OPEN + 0.008), 0.040)
        self.assertLess(t.GRIP_UP - 0.013, L.UNIT_SIZE[2])  # the pads close on the unit, not over it
        self.assertGreater(t.GRIP_UP + 0.025, L.UNIT_SIZE[2])

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
        """A unit counts as set while still gripped (within 3 mm), so the success hold must be longer than the
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
        self.assertEqual(n, 6)


if __name__ == "__main__":
    unittest.main()
