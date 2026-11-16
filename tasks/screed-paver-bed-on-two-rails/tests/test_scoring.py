"""Offline tests for layout.py, scoring.py and task.py: synthetic board poses and arm snapshots, no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline stand-in's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
import sys
import unittest
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker, tier  # noqa: E402

try:
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    T = None

HZ = L.CONTROL_HZ
WIN = int(round(L.STILL_WINDOW_S * HZ))


# -- synthetic states ----------------------------------------------------------------------------------------------
def quat(roll=0.0, pitch=0.0, yaw=0.0):
    """xyzw of Rz(yaw) Ry(pitch) Rx(roll), angles in degrees."""
    r, p, y = (math.radians(a) / 2 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy])


def board(x=L.BD_REST_X, y=0.0, sole=L.BD_REST_Z, roll=0.0, pitch=0.0, yaw=0.0, vel=(0.0, 0.0, 0.0)):
    """The board with the centre of its underside at (x, y) and ``sole`` above the table top, turned as given."""
    q = quat(roll, pitch, yaw)
    r = S.rotation(q)
    c = np.array([x, y, L.TOP + sole]) + r @ np.array([0.0, 0.0, L.BD_Z / 2])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=c, high=c)


def arm(tcp, opening=0.44, down=0.013):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, down]), opening=opening)


def arms(left=(0.15, 0.285, L.TOP + 0.10), right=(0.15, -0.285, L.TOP + 0.10), opening=0.44):
    return {"left": arm(left, opening), "right": arm(right, opening)}


AWAY = arms()


class Run:
    """Feeds one episode to a fresh tracker."""

    def __init__(self):
        self.t = Tracker(HZ)
        self.r = None

    def feed(self, b, a=None, n=1):
        for _ in range(n):
            self.r = self.t.update({S.BOARD: b}, AWAY if a is None else a)
        return self.r

    def glide(self, b0, b1, n, a=None):
        for k in range(1, n + 1):
            f = k / n
            self.feed(SimpleNamespace(pos=b0.pos + (b1.pos - b0.pos) * f, quat_xyzw=b1.quat_xyzw,
                                      lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=b0.low, high=b0.high), a)
        return self.r

    def pull(self, sole=L.RAIL_H, x0=L.BD_START_X, x1=L.BD_END_X, step=0.00075, **kw):
        x = x0
        n = int(round((x0 - x1) / step))
        for k in range(n + 1):
            self.feed(board(x0 + (x1 - x0) * k / n, sole=sole, **kw))
        return self.r

    def to_rails(self, sole=L.RAIL_H, lift=L.BD_REST_Z + 0.01):
        """From the blocks to the rails' far end: lift, carry, lower."""
        self.feed(board(), n=5)
        self.glide(board(), board(sole=lift), 20)
        self.glide(board(sole=lift), board(L.BD_START_X, sole=lift), 80)
        self.glide(board(L.BD_START_X, sole=lift), board(L.BD_START_X, sole=sole), 40)

    def to_blocks(self, sole=L.RAIL_H, lift=L.BD_REST_Z + 0.01, x_from=L.BD_END_X):
        self.glide(board(x_from, sole=sole), board(x_from, sole=lift), 40)
        self.glide(board(x_from, sole=lift), board(sole=lift), 40)
        self.glide(board(sole=lift), board(), 30)

    def whole(self, sole=L.RAIL_H, hold=2 * WIN, x1=L.BD_END_X, **kw):
        """The demo's course with everything away at the end."""
        self.to_rails(sole)
        self.pull(sole, x1=x1, **kw)
        self.feed(board(x1, sole=sole), n=10)
        self.to_blocks(sole, x_from=x1)
        return self.feed(board(), n=hold)


def whole_run(**kw):
    run = Run()
    run.whole(**kw)
    return run


class Layout(unittest.TestCase):
    def test_rails_fill_the_cell_columns_and_the_sole_is_one_cell(self):
        for j, side in zip(L.RAIL_COLS, (-1, 1)):
            lo, hi = L.BED_Y0 + j * L.CELL, L.BED_Y0 + (j + 1) * L.CELL
            self.assertAlmostEqual(side * L.RAIL_Y - L.RAIL_W / 2, lo, places=9)  # the rail's near face is the column's
            self.assertAlmostEqual(side * L.RAIL_Y + L.RAIL_W / 2, hi, places=9)
        self.assertAlmostEqual(L.BD_X, L.CELL, places=9)
        self.assertAlmostEqual(L.RAIL_W, L.CELL, places=9)

    def test_start_heights(self):
        h = np.array(L.start_heights())
        sand = h[S._SAND]
        self.assertEqual(sand.size, 112)
        self.assertAlmostEqual(float(sand.max()), 0.045)
        self.assertAlmostEqual(float(sand.min()), 0.019)
        self.assertEqual(int((h == L.MOUND_H).sum()), 9)
        self.assertEqual(int((h == L.HOLLOW_H).sum()), 9)
        self.assertTrue(np.all(h[:, list(L.RAIL_COLS)] == L.BED_TARGET))
        self.assertTrue(np.all(h[L.UNDER_BOARD_I, :] == L.BED_TARGET))
        mass = float(sand.sum()) * L.CELL ** 2 * L.SAND_DENSITY
        self.assertTrue(3.0 < mass < 4.0, mass)
        # the mound and the hollow stay off the rail columns and the board's start column
        for i in L.MOUND_I:
            self.assertNotEqual(i, L.UNDER_BOARD_I)
        for j in list(L.MOUND_J) + list(L.HOLLOW_J):
            self.assertNotIn(j, L.RAIL_COLS)

    def test_the_board_starts_clear_of_sand_and_walls(self):
        b = board(L.BD_START_X)
        self.assertLess(L.BD_START_X + L.BD_X / 2, L.INSIDE_X1 - 0.010)  # 15 mm to the far wall
        self.assertLess(L.BD_L / 2, L.INSIDE_Y - 0.010)  # 15 mm to each side wall
        under = [i for i in range(L.NX) if abs(L.cell_x(i) - L.BD_START_X) <= L.BD_X / 2]
        self.assertEqual(under, [L.UNDER_BOARD_I])
        inside, _ = S.under_sole(b.pos - np.array([0.0, 0.0, L.TOP]), S.rotation(b.quat_xyzw), S._X, S._Y)
        self.assertEqual(np.nonzero(inside.any(axis=1))[0].tolist(), [L.UNDER_BOARD_I])

    def test_the_blocks_are_off_the_pull_path(self):
        # the pull runs the board's sole between x 200 and 400 mm; the blocks' far faces end 30 mm short of the bed
        self.assertLess(L.STOP_X + L.STOP_SIZE[0] / 2 + L.BD_X / 2, L.BED_X0)
        # and they stand outside the rails (y 100..125) and clear of the grips (y 166..194)
        self.assertGreater(L.STOP_Y - L.STOP_SIZE[1] / 2, L.RAIL_Y + L.RAIL_W / 2 - 1e-9)
        self.assertLess(L.STOP_Y + L.STOP_SIZE[1] / 2, 0.180 - 0.0135)

    def test_the_rest_height_clears_the_mound(self):
        self.assertGreater(L.BD_REST_Z, L.MOUND_H)  # the board on its blocks is above the highest sand
        self.assertGreater(L.TALL_MAX, L.BED_TARGET + L.TOL_LOOSE)  # 'tall' admits everything 'loose' admits

    def test_start_scores_zero_and_is_not_success(self):
        run = Run()
        r = run.feed(board(), n=3 * WIN)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertTrue(r.details["stowed"])
        self.assertFalse(r.details["tall"])
        self.assertAlmostEqual(r.details["max_depth_mm"], 45.0)


class SandModel(unittest.TestCase):
    def test_a_straight_pull_levels_the_bed_and_conserves_the_sand(self):
        run = Run()
        run.to_rails()
        v0 = run.t.sand.volume()
        run.pull()
        s = run.t.sand
        self.assertAlmostEqual(s.deviation(), 0.0, places=9)
        self.assertAlmostEqual(s.max_depth(), 0.025, places=9)
        self.assertAlmostEqual(s.volume(), v0, places=9)  # heap and spoil count: nothing appears or vanishes
        self.assertGreater(s.spoil, 0.0)  # the surplus went over the near edge
        self.assertEqual(float(s.heap.sum()), 0.0)

    def test_volume_is_conserved_on_every_step(self):
        run = Run()
        run.to_rails()
        v0 = run.t.sand.volume()
        x = L.BD_START_X
        while x > L.BD_END_X:
            run.feed(board(x))
            self.assertAlmostEqual(run.t.sand.volume(), v0, places=9)
            x -= 0.0011

    def test_the_hollow_is_filled_from_the_heap(self):
        run = Run()
        run.to_rails()
        run.pull(x1=0.2375)  # until the sole is over the hollow's last cell column
        s = run.t.sand
        for i in L.HOLLOW_I:
            for j in L.HOLLOW_J:
                self.assertAlmostEqual(float(s.h[i, j]), 0.025, places=9)

    def test_there_is_enough_sand_for_the_hollow_and_not_far_too_much(self):
        # the hollow columns carry 3 x 11 mm in by the time the hollow starts; it takes 3 x 6 mm
        run = Run()
        run.to_rails()
        run.pull(x1=0.3125)
        j = L.HOLLOW_J[0]
        self.assertAlmostEqual(float(run.t.sand.heap[j]), 3 * (L.START_H - L.BED_TARGET), places=9)
        self.assertGreater(3 * (L.START_H - L.BED_TARGET), 3 * (L.BED_TARGET - L.HOLLOW_H))

    def test_a_board_lifted_mid_bed_drops_its_heap_as_a_hump(self):
        run = Run()
        run.to_rails()
        run.pull(x1=0.2875)
        run.glide(board(0.2875), board(0.2875, sole=0.060), 10)
        run.feed(board(0.2875, sole=0.060), n=5)
        s = run.t.sand
        self.assertGreater(s.max_depth(), 0.045)  # the ridge it carried now sits on one cell
        run.glide(board(0.2875, sole=0.060), board(L.BD_END_X, sole=0.060), 20)
        run.glide(board(L.BD_END_X, sole=0.060), board(L.BD_END_X, sole=L.RAIL_H), 10)
        self.assertGreater(run.t.sand.deviation(), 0.003)  # not level any more: the board has to go back over it

    def test_a_board_held_high_cuts_to_where_it_is_and_fills_no_hollow(self):
        run = Run()
        run.to_rails(sole=0.030)
        run.pull(sole=0.030)
        s = run.t.sand
        self.assertLess(s.max_depth(), 0.0301)
        self.assertAlmostEqual(float(s.h[1, 8]), L.HOLLOW_H, places=9)  # the hollow: 5 mm high fills nothing
        # 5 mm high is above the 3 mm the board may ride, so it is not 'bridged'
        self.assertEqual(run.r.details["pull_mm"], 0.0)

    def test_a_tilted_board_cuts_a_wedge(self):
        # one end 9 mm above the other over its 400 mm: the sole is 25 mm on the centre line and 25 +- 4.5 mm at
        # the ends of the bed's width
        roll = math.degrees(math.atan(0.0200))  # 20 mm per metre of y
        run = Run()
        run.to_rails()
        run.glide(board(L.BD_START_X, sole=L.RAIL_H), board(L.BD_START_X, sole=L.RAIL_H, roll=roll), 10)
        run.pull(roll=roll)
        s = run.t.sand
        col_lo, col_hi = s.h[:, 0], s.h[:, 15]
        self.assertAlmostEqual(float(col_hi[3] - col_lo[3]), 0.020 * 0.375, places=4)  # 7.5 mm end to end
        self.assertAlmostEqual(s.deviation(), 0.020 * 0.1875, places=4)

    def test_skew_leaves_the_bed_ends_uncut_or_cut_unevenly(self):
        run = Run()
        run.to_rails()
        run.glide(board(L.BD_START_X, sole=L.RAIL_H), board(L.BD_START_X, sole=L.RAIL_H, yaw=12.0), 10)
        run.pull(yaw=12.0)
        self.assertGreater(run.t.sand.deviation(), 0.003)  # a 12 degree skew leaves the bed's edge columns high

    def test_a_fast_pull_cannot_skip_cells(self):
        slow, fast = Run(), Run()
        slow.to_rails()
        fast.to_rails()
        slow.pull(step=0.0005)
        for k in range(1, 4):  # the whole 175 mm in three control steps (a physics glitch)
            fast.feed(board(L.BD_START_X + (L.BD_END_X - L.BD_START_X) * k / 3, sole=L.RAIL_H))
        self.assertLess(float(np.abs(fast.t.sand.h - slow.t.sand.h).max()), 1e-9)

    def test_pulling_away_from_the_robot_piles_the_heap_at_the_far_wall(self):
        run = Run()
        run.feed(board(), n=3)
        run.glide(board(), board(sole=0.065), 20)
        run.glide(board(sole=0.065), board(L.BD_END_X, sole=0.065), 40)
        run.glide(board(L.BD_END_X, sole=0.065), board(L.BD_END_X, sole=L.RAIL_H), 30)
        v0 = run.t.sand.volume()
        run.pull(x0=L.BD_END_X, x1=L.BD_START_X)
        s = run.t.sand
        self.assertAlmostEqual(s.volume(), v0, places=9)
        run.glide(board(L.BD_START_X), board(L.BD_START_X, sole=0.065), 20)
        self.assertGreater(s.max_depth(), 0.032)  # nothing went to the tray: the far wall's cells are deep
        self.assertFalse(run.r.details["tall"])
        self.assertEqual(run.r.score, 0.0)

    def test_the_board_on_its_side_cuts_nothing(self):
        run = Run()
        run.to_rails()
        run.feed(board(L.BD_START_X, sole=0.0, roll=80.0), n=3)
        self.assertEqual(run.t.sand.touching.sum(), 0)


class Tiers(unittest.TestCase):
    def test_the_whole_course_scores_one(self):
        run = whole_run()
        self.assertEqual(run.r.score, 1.0)
        self.assertTrue(run.r.success)
        d = run.r.details
        self.assertEqual(d["cells_off_3mm"], 0)
        self.assertAlmostEqual(d["pull_mm"], 175.0, places=1)
        self.assertTrue(d["stowed"] and d["released_1s"] and d["still_1s"])

    def test_the_demo_trims_by_the_arm_press(self):
        # the demo commands the sole 2 mm under the rails: the board rests on them (25 mm); the offline stand-in
        # and the real rails agree, so the result is the same as with the sole on the rails
        run = whole_run(sole=0.025)
        self.assertEqual(run.r.score, 1.0)

    def test_up_to_3mm_high_or_low_is_still_one(self):
        for sole in (0.022, 0.0235, 0.0265, 0.028):  # up to 3 mm either way (the heap still covers the hollow)
            run = whole_run(sole=sole)
            self.assertEqual(run.r.score, 1.0, sole)

    def test_a_wedge_of_375_mm_scores_075(self):
        roll = math.degrees(math.atan(0.0200))
        run = Run()
        run.to_rails()
        run.glide(board(L.BD_START_X, sole=L.RAIL_H), board(L.BD_START_X, sole=L.RAIL_H, roll=roll), 10)
        run.pull(roll=roll)
        run.feed(board(L.BD_END_X, roll=roll), n=10)
        run.glide(board(L.BD_END_X, sole=L.RAIL_H, roll=roll), board(L.BD_END_X, sole=0.065), 20)
        run.to_blocks(sole=0.065)
        r = run.feed(board(), n=2 * WIN)
        self.assertTrue(0.003 < r.details["dev_mm"] / 1000 <= 0.004, r.details["dev_mm"])
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_a_wedge_of_5_mm_scores_half(self):
        roll = math.degrees(math.atan(0.0265))  # 4.97 mm at the bed's edge columns, 2.98 over each rail
        run = Run()
        run.to_rails()
        run.glide(board(L.BD_START_X, sole=L.RAIL_H), board(L.BD_START_X, sole=L.RAIL_H, roll=roll), 10)
        run.pull(roll=roll)
        run.glide(board(L.BD_END_X, sole=L.RAIL_H, roll=roll), board(L.BD_END_X, sole=0.065), 20)
        run.to_blocks(sole=0.065)
        r = run.feed(board(), n=2 * WIN)
        self.assertTrue(0.004 < r.details["dev_mm"] / 1000 <= 0.006, r.details["dev_mm"])
        self.assertEqual(r.score, 0.5)

    def test_level_but_left_on_the_bed_scores_half(self):
        run = Run()
        run.to_rails()
        run.pull()
        r = run.feed(board(L.BD_END_X), n=2 * WIN)
        self.assertEqual(r.details["cells_off_3mm"], 0)
        self.assertFalse(r.details["stowed"])
        self.assertEqual(r.score, 0.5)

    def test_level_and_stowed_but_not_hands_off_scores_075(self):
        run = Run()
        run.to_rails()
        run.pull()
        run.feed(board(L.BD_END_X), n=10)
        run.to_blocks()
        near = arms(left=(0.15, 0.18, L.TOP + 0.097))  # still on the board's end
        r = run.feed(board(), near, n=2 * WIN)
        self.assertEqual(r.details["cells_off_3mm"], 0)
        self.assertFalse(r.details["released_1s"])
        self.assertEqual(r.score, 0.75)

    def test_moving_in_the_last_second_is_not_still(self):
        run = Run()
        run.to_rails()
        run.pull()
        run.feed(board(L.BD_END_X), n=10)
        run.to_blocks()
        run.feed(board(), n=2 * WIN)
        r = run.feed(board(L.BD_REST_X + 0.0006), n=1)  # a 0.6 mm shift
        r = run.feed(board(L.BD_REST_X + 0.0012), n=1)
        self.assertFalse(r.details["still_1s"])  # 1.2 mm from where the window started
        self.assertEqual(r.score, 0.75)
        r = run.feed(board(L.BD_REST_X + 0.0012), n=WIN + 1)
        self.assertEqual(r.score, 1.0)

    def test_a_turn_over_03_degrees_in_the_last_second_is_not_still(self):
        run = whole_run()
        r = run.feed(board(yaw=0.2), n=1)
        r = run.feed(board(yaw=0.4), n=1)
        self.assertFalse(r.details["still_1s"])
        self.assertEqual(r.score, 0.75)

    def test_hands_30mm_and_not_less(self):
        # the right hand's closed finger zone (radius 16 mm) beside the board's end: the end face is at y -200 mm
        for gap, expect in ((0.026, 0.75), (0.034, 1.0)):
            run = Run()
            run.whole(hold=0)
            r_zone = rigstate.zone_radius(0.0)
            y = -(0.200 + gap + r_zone)
            a = arms(right=(0.15, y, L.TOP + L.BD_REST_Z + 0.025 + 0.013), opening=0.0)
            r = run.feed(board(), a, n=2 * WIN)
            self.assertEqual(r.score, expect, gap)

    def test_a_hand_left_over_the_bed_is_not_hands_off(self):
        run = Run()
        run.whole(hold=0)
        a = arms(left=(0.30, 0.10, L.TOP + 0.10))
        r = run.feed(board(), a, n=2 * WIN)
        self.assertFalse(r.details["released_1s"])
        self.assertEqual(r.score, 0.75)

    def test_tier_table(self):
        t = lambda **k: tier(**{**dict(latched=False, pull_ok=True, cut=True, dev=0.0, is_stowed=True, settled=True,
                                       pressed=False), **k})
        self.assertEqual(t(), 1.0)
        self.assertEqual(t(dev=0.0030), 1.0)
        self.assertEqual(t(dev=0.0031), 0.75)
        self.assertEqual(t(dev=0.0040), 0.75)
        self.assertEqual(t(dev=0.0041), 0.5)
        self.assertEqual(t(dev=0.0060), 0.5)
        self.assertEqual(t(dev=0.0061), 0.25)
        self.assertEqual(t(dev=0.020), 0.25)
        self.assertEqual(t(is_stowed=False), 0.5)
        self.assertEqual(t(settled=False), 0.75)
        self.assertEqual(t(settled=False, is_stowed=False), 0.5)
        self.assertEqual(t(pressed=True), 0.5)
        self.assertEqual(t(pressed=True, dev=0.02), 0.25)
        self.assertEqual(t(cut=False), 0.0)
        self.assertEqual(t(pull_ok=False), 0.0)
        self.assertEqual(t(latched=True), 0.0)

    def test_a_bed_cut_but_overcut_in_one_cell_scores_a_quarter(self):
        run = whole_run()
        run.t.sand.h[3, 5] = 0.015  # a divot 10 mm under the target (a board that dipped at the end would leave it)
        r = run.feed(board(), n=1)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_one_high_cell_anywhere_is_caught_at_every_tier(self):
        # the review copy's point: nine probes would miss it; every cell is read
        for i, j in ((0, 0), (3, 15), (7, 7), (4, 8)):
            run = whole_run()
            run.t.sand.h[i, j] = 0.029
            r = run.feed(board(), n=1)
            self.assertEqual(r.score, 0.75, (i, j))
            run.t.sand.h[i, j] = 0.033
            r = run.feed(board(), n=1)
            self.assertEqual(r.score, 0.25, (i, j))  # one cell 8 mm high: only the bottom tier, not zero
            self.assertFalse(r.details["tall"])
            self.assertTrue(r.details["cut"])

    def test_cut_needs_three_quarters_of_the_cells(self):
        run = whole_run()
        k = 0
        for i in range(L.NX):
            for j in range(L.NY):
                if j in L.RAIL_COLS:
                    continue
                if k < 112 - L.CUT_CELLS:
                    run.t.sand.h[i, j] = 0.034  # 9 mm high
                k += 1
        r = run.feed(board(), n=1)
        self.assertEqual(r.details["cells_within_6mm"], L.CUT_CELLS)
        self.assertEqual(r.score, 0.25)
        run.t.sand.h[7, 7] = 0.034
        r = run.feed(board(), n=1)
        self.assertEqual(r.details["cells_within_6mm"], L.CUT_CELLS - 1)
        self.assertEqual(r.score, 0.0)

    def test_the_start_is_far_from_cut(self):
        r = Run().feed(board(), n=1)
        self.assertEqual(r.details["cells_within_6mm"], 14 + 9)  # the column under the board's start and the hollow
        self.assertFalse(r.details["cut"])

    def test_no_pull_no_score(self):
        run = whole_run(x1=L.BD_START_X - 0.140)  # 140 mm only
        self.assertEqual(run.r.details["pull_mm"], 140.0)
        self.assertEqual(run.r.score, 0.0)


class Zero(unittest.TestCase):
    def test_doing_nothing_scores_zero(self):
        self.assertEqual(Run().feed(board(), n=200).score, 0.0)

    def test_the_board_alone_on_the_rails_scores_zero(self):
        run = Run()
        run.to_rails()
        self.assertEqual(run.feed(board(L.BD_START_X, sole=L.RAIL_H), n=60).score, 0.0)

    def test_a_board_through_the_table_latches_zero(self):
        run = Run()
        run.to_rails()
        run.feed(board(L.BD_START_X, sole=0.03, roll=40.0))  # one end 98 mm under the table top: off it
        run.feed(board(L.BD_START_X, sole=L.RAIL_H), n=3)
        self.assertEqual(run.t.latch["F1"], run.t.steps - 3)
        run.pull()
        run.feed(board(L.BD_END_X), n=5)
        run.to_blocks()
        r = run.feed(board(), n=3 * WIN)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_a_board_tipped_over_on_the_table_is_not_a_fall(self):
        run = whole_run()
        self.assertEqual(run.r.score, 1.0)
        r = run.feed(board(0.10, sole=L.BD_X / 2, pitch=90.0), n=3 * WIN)  # on its 50 mm face on the table
        self.assertIsNone(run.t.latch["F1"])
        self.assertEqual(r.details["pressed_cells"], 0)
        self.assertEqual(r.score, 0.5)  # the bed is level, the board not stowed

    def test_a_board_tipped_onto_the_finished_bed_presses_it(self):
        run = Run()
        run.to_rails()
        run.pull()
        run.feed(board(L.BD_END_X), n=5)
        run.to_blocks()
        run.feed(board(), n=5)
        # dropped on its 50 mm face across the rails over the finished cells (its lowest face 25 mm up)
        run.glide(board(), board(0.30, sole=L.RAIL_H + L.BD_X / 2, pitch=90.0), 10)
        r = run.feed(board(0.30, sole=L.RAIL_H + L.BD_X / 2, pitch=90.0), n=5)
        self.assertGreater(r.details["pressed_cells"], 0)
        run.glide(board(0.30, sole=L.RAIL_H + L.BD_X / 2, pitch=90.0), board(sole=0.08), 10)
        r = run.feed(board(), n=3 * WIN)
        self.assertTrue(r.details["stowed"])
        self.assertEqual(r.score, 0.5)  # pressed caps it

    def test_a_board_tipped_mid_pull_drops_its_heap(self):
        run = Run()
        run.to_rails()
        v0 = run.t.sand.volume()
        run.pull(x1=0.30)
        self.assertGreater(float(run.t.sand.heap.sum()), 0.0)
        run.feed(board(0.30, sole=L.RAIL_H + L.BD_X / 2, pitch=90.0))
        self.assertEqual(float(run.t.sand.heap.sum()), 0.0)
        self.assertAlmostEqual(run.t.sand.volume(), v0, places=9)

    def test_tilt_just_under_30_degrees_does_not_latch(self):
        run = Run()
        run.to_rails()
        run.feed(board(L.BD_START_X, sole=0.20, roll=29.0), n=2)
        self.assertIsNone(run.t.latch["F1"])

    def test_a_board_knocked_off_the_table_latches_zero(self):
        run = whole_run()
        self.assertEqual(run.r.score, 1.0)
        run.feed(board(0.5, 0.7, sole=-0.15), n=1)
        self.assertEqual(run.r.score, 0.0)
        run.feed(board(), n=3 * WIN)
        self.assertEqual(run.t.last.score, 0.0)

    def test_a_board_that_missed_the_rails_is_not_bridged(self):
        # laid 40 mm to one side (y), one end over a rail only
        run = Run()
        run.to_rails()
        for k in range(200):
            run.feed(board(L.BD_START_X - 0.00088 * k, y=0.150, sole=L.RAIL_H))
        self.assertEqual(run.r.details["pull_mm"], 0.0)
        self.assertEqual(run.r.score, 0.0)

    def test_a_board_along_x_is_not_bridged(self):
        run = Run()
        run.to_rails()
        run.pull(yaw=90.0)
        self.assertEqual(run.r.details["pull_mm"], 0.0)

    def test_a_pull_broken_by_a_lift_does_not_add_up(self):
        run = Run()
        run.to_rails()
        run.pull(x1=0.3000)  # 87.5 mm
        run.feed(board(0.30, sole=0.06), n=3)  # lifted off for a moment
        run.pull(x0=0.30, x1=0.2125)  # another 87.5 mm
        self.assertAlmostEqual(run.r.details["pull_mm"], 87.5, places=1)
        self.assertEqual(run.r.score, 0.0)


class Exploits(unittest.TestCase):
    def test_a_hand_in_the_finished_sand_caps_at_half(self):
        run = Run()
        run.to_rails()
        run.pull()
        run.feed(board(L.BD_END_X), n=10)
        run.to_blocks()
        # a pad tip dipped into a finished cell (x 300, y 62) after the board had passed it
        dip = arms(left=(0.300, 0.0625, L.TOP + 0.025 + 0.013 - 0.003))
        run.feed(board(), dip, n=2)
        r = run.feed(board(), AWAY, n=3 * WIN)
        self.assertGreater(r.details["pressed_cells"], 0)
        self.assertEqual(r.details["cells_off_3mm"], 0)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)

    def test_a_pad_tip_2mm_over_the_finished_sand_still_presses_and_5mm_does_not(self):
        for clear, pressed in ((0.0015, True), (0.005, False)):
            run = Run()
            run.to_rails()
            run.pull()
            run.feed(board(L.BD_END_X), n=10)
            run.to_blocks()
            a = arms(right=(0.300, -0.0625, L.TOP + 0.025 + clear + 0.013))
            run.feed(board(), a, n=2)
            r = run.feed(board(), AWAY, n=3 * WIN)
            self.assertEqual(r.details["pressed_cells"] > 0, pressed, clear)

    def test_a_hand_in_the_sand_before_the_board_has_passed_is_not_a_press(self):
        run = Run()
        run.to_rails()
        a = arms(left=(0.300, 0.0625, L.TOP + 0.030))  # in the unscreeded sand ahead of the board
        run.feed(board(L.BD_START_X, sole=L.RAIL_H), a, n=3)
        run.pull()
        run.feed(board(L.BD_END_X), n=10)
        run.to_blocks()
        r = run.feed(board(), n=3 * WIN)
        self.assertEqual(r.details["pressed_cells"], 0)
        self.assertEqual(r.score, 1.0)

    def test_the_board_passing_a_cell_again_clears_the_press(self):
        run = Run()
        run.to_rails()
        run.pull()
        run.feed(board(L.BD_END_X), n=10)
        a = arms(left=(0.300, 0.0625, L.TOP + 0.020))
        run.feed(board(L.BD_END_X), a, n=2)
        self.assertGreater(run.t.sand.pressed.sum(), 0)
        run.feed(board(L.BD_END_X), AWAY, n=2)
        run.glide(board(L.BD_END_X), board(L.BD_END_X, sole=0.065), 10)
        run.glide(board(L.BD_END_X, sole=0.065), board(L.BD_START_X, sole=0.065), 40)
        run.glide(board(L.BD_START_X, sole=0.065), board(L.BD_START_X, sole=L.RAIL_H), 20)
        run.pull()  # screed it again: the cell is passed again
        self.assertEqual(run.t.sand.pressed.sum(), 0)

    def test_unreadable_arms_after_the_pass_count_as_pressed(self):
        run = Run()
        run.to_rails()
        run.pull()
        run.feed(board(L.BD_END_X), n=10)
        run.to_blocks()
        r = run.t.update({S.BOARD: board()}, None)
        self.assertTrue(r.details["blind"])
        r = run.feed(board(), n=3 * WIN)
        self.assertEqual(r.score, 0.5)

    def test_skimming_the_rails_a_little_high_cuts_nothing_for_the_bed(self):
        # 4 mm over the rails: not bridged, so no pull; and the bed it cut sits 4 mm high
        run = Run()
        run.to_rails(sole=0.029)
        run.pull(sole=0.029)
        self.assertEqual(run.r.details["pull_mm"], 0.0)
        self.assertEqual(run.r.score, 0.0)

    def test_pressing_the_board_through_the_rails_is_not_a_trick(self):
        # a sole 4 mm under the rails' tops is not bridged: the board cannot be pushed into the bed for a clean cut
        run = Run()
        run.to_rails(sole=0.021)
        run.pull(sole=0.021)
        self.assertEqual(run.r.details["pull_mm"], 0.0)
        self.assertEqual(run.r.score, 0.0)

    def test_dragging_the_board_with_the_sand_ahead_and_dumping_it_elsewhere_gains_nothing(self):
        # screed 87 mm, lift, put the board down at the near end beside the bed's edge and push the heap off by hand:
        # the sand model only ever changes under the board, so the bed stays as the board left it
        run = Run()
        run.to_rails()
        run.pull(x1=0.30)
        run.feed(board(0.30, sole=0.07), n=5)  # lifted: the heap drops ahead of the board, once
        before = run.t.sand.h.copy()
        run.feed(board(0.45, 0.4, sole=0.07), arms(left=(0.30, 0.0, L.TOP + 0.020)), n=30)
        self.assertTrue(np.array_equal(run.t.sand.h, before))

    def test_lifting_the_board_off_and_never_pulling_scores_zero_even_with_the_bed_level_on_paper(self):
        run = Run()
        run.t.sand.h[:] = 0.025  # a level bed handed to the scorer, but nothing was ever pulled
        r = run.feed(board(), n=3 * WIN)
        self.assertEqual(r.score, 0.0)

    def test_pushing_the_board_through_the_bed_from_the_wrong_end_levels_nothing_extra(self):
        run = Run()
        run.glide(board(), board(sole=0.065), 10)
        run.glide(board(sole=0.065), board(0.30, sole=0.065), 30)
        run.glide(board(0.30, sole=0.065), board(0.30, sole=L.RAIL_H), 20)
        run.pull(x0=0.30, x1=0.2125)
        s = run.t.sand
        self.assertGreaterEqual(float(s.h[5, 4]), L.MOUND_H)  # east of where the board went down: never cut
        self.assertGreaterEqual(float(s.h[6, 4]), L.MOUND_H)
        self.assertGreaterEqual(float(s.h[5, 8]), L.START_H)
        run.feed(board(L.BD_END_X), n=10)
        run.to_blocks(x_from=L.BD_END_X)
        r = run.feed(board(), n=3 * WIN)
        self.assertEqual(r.score, 0.0)  # half the bed was never screeded: not even 'tall'


class OddStates(unittest.TestCase):
    def test_nan_and_missing_observations(self):
        bad = [
            None, SimpleNamespace(pos=np.array([np.nan, 0, 0.8]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                                  lin_vel=np.zeros(3), ang_vel=np.zeros(3)),
            SimpleNamespace(pos=np.array([0.15, 0, 0.8]), quat_xyzw=np.array([0, 0, 0, np.inf]),
                            lin_vel=np.zeros(3), ang_vel=np.zeros(3)),
            SimpleNamespace(pos=np.array([0.15, 0, 0.8]), quat_xyzw=np.zeros(4), lin_vel=np.zeros(3),
                            ang_vel=np.zeros(3)),
            SimpleNamespace(pos=np.array([0.15, 0]), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                            ang_vel=np.zeros(3)),
        ]
        run = Run()
        run.to_rails()
        run.pull(x1=0.30)
        before = run.t.sand.h.copy()
        for b in bad:
            r = run.t.update({S.BOARD: b}, AWAY)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
        self.assertTrue(np.array_equal(run.t.sand.h, before))
        self.assertEqual(run.t.bad_steps, len(bad))
        self.assertEqual(run.t.update({}, AWAY).score, 0.0)
        self.assertEqual(run.t.update(None, AWAY).score, 0.0)

    def test_a_blow_up_in_the_last_second_restarts_the_stillness_window(self):
        run = whole_run()
        self.assertEqual(run.r.score, 1.0)
        run.t.update({S.BOARD: None}, AWAY)
        r = run.feed(board(), n=WIN - 2)
        self.assertNotEqual(r.score, 1.0)
        r = run.feed(board(), n=WIN + 2)
        self.assertEqual(r.score, 1.0)

    def test_nan_arms_are_not_released(self):
        run = Run()
        run.whole(hold=0)
        nan_arms = {"left": rigstate.Arm(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3), opening=0.4),
                    "right": AWAY["right"]}
        r = run.feed(board(), nan_arms, n=3 * WIN)
        self.assertFalse(r.details["released_1s"])
        self.assertTrue(r.details["blind"])  # an arm it could not see after the pass: counted as pressed
        self.assertEqual(r.score, 0.5)

    def test_no_crash_and_json_clean_details(self):
        run = whole_run()
        json.dumps(run.r.details)
        r = run.t.update({S.BOARD: None}, None)
        json.dumps(r.details)
        r = run.t.update({S.BOARD: board()}, None)
        json.dumps(r.details)

    def test_a_speed_boosted_unit_quaternion_is_normalised(self):
        b = board()
        b.quat_xyzw = b.quat_xyzw * 3.0
        self.assertEqual(Run().feed(b).details["valid_state"], True)


class Reset(unittest.TestCase):
    def test_teleport_back_to_the_mark_resets_the_episode(self):
        run = whole_run()
        self.assertEqual(run.r.score, 1.0)
        run.t.latch["F1"] = 7
        far = board(0.30, sole=0.07)
        run.feed(far, n=2)
        run.feed(board(), n=1)  # 15 cm back to its mark in one step, with nothing to explain it
        self.assertEqual(run.t.steps, 1)
        self.assertIsNone(run.t.latch["F1"])
        self.assertEqual(run.t.best_pull, 0.0)
        self.assertAlmostEqual(run.t.sand.max_depth(), 0.045)
        self.assertEqual(run.t.resets, 1)

    def test_putting_the_board_back_slowly_does_not_reset(self):
        run = whole_run()
        n = run.t.steps
        run.glide(board(), board(sole=0.07), 10)
        run.glide(board(sole=0.07), board(), 10)
        self.assertEqual(run.t.steps, n + 20)
        self.assertEqual(run.t.resets, 0)

    def test_dropping_the_board_onto_its_blocks_from_height_is_not_a_reset(self):
        run = whole_run()
        steps = run.t.steps
        g, dt = L.G, 1 / HZ
        h, v = 0.40, 0.0
        run.feed(board(sole=L.BD_REST_Z + h), n=2)
        while h > 0:
            v, h = v + g * dt, h - v * dt - 0.5 * g * dt * dt
            if h > 0:
                run.feed(board(sole=L.BD_REST_Z + h, vel=(0, 0, -v)))
        run.feed(board())
        self.assertEqual(run.t.resets, 0)
        self.assertGreater(run.t.steps, steps)

    def test_rigstate_reset_callback_clears_the_tracker(self):
        if T is None:
            self.skipTest("no yam_arena stub")
        T._TRACKER.sand.h[:] = 0.0
        rigstate.offline_reset()
        self.assertAlmostEqual(T._TRACKER.sand.max_depth(), 0.045)


@unittest.skipIf(T is None, "yam_arena is not importable: run next to tools/offline")
class TaskFile(unittest.TestCase):
    def setUp(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
            os.environ.pop(k, None)

    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(len(set(names)), len(names))
        board_spec = T.OBJECTS[0]
        self.assertEqual(board_spec.name, "board")
        self.assertIsNotNone(board_spec.mass)
        self.assertTrue(all(o.mass is None for o in T.OBJECTS[1:]))
        self.assertAlmostEqual(board_spec.pos[2], L.TOP + L.BD_REST_Z + L.BD_Z / 2)
        # the drawn sand sits below the sole and the rails do not overlap the slabs
        self.assertLess(L.SAND_DRAW_H, L.RAIL_H - 0.004)

    def test_demo_plans(self):
        plans = T.demo()
        self.assertEqual(set(plans), {"left", "right"})
        dur = [sum(w.duration for w in p) for p in plans.values()]
        self.assertEqual(dur[0], dur[1])
        self.assertLess(dur[0] * HZ, 1200)  # within half the run limit
        left, right = plans["left"], plans["right"]
        for a, b in zip(left, right):  # mirror images: same x and z and finger on one clock, y and yaw mirrored
            self.assertEqual(a.duration, b.duration)
            self.assertAlmostEqual(a.pos[0], b.pos[0])
            self.assertAlmostEqual(a.pos[2], b.pos[2])
            self.assertAlmostEqual(a.pos[1], -b.pos[1])
            self.assertEqual(a.finger, b.finger)

    def test_idle_arm_and_hold_only_and_variants(self):
        os.environ["DEMO_IDLE_ARM"] = "left"
        self.assertEqual(set(T.demo()), {"right"})
        os.environ["DEMO_IDLE_ARM"] = "right"
        self.assertEqual(set(T.demo()), {"left"})
        os.environ.pop("DEMO_IDLE_ARM")
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        os.environ.pop("DEMO_HOLD_ONLY")
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            self.assertEqual(set(T.demo()), {"left", "right"})
        os.environ["DEMO_VARIANT"] = "nonsense"
        with self.assertRaises(ValueError):
            T.demo()

    def test_the_offline_support_matches_the_rails_and_blocks(self):
        self.assertEqual(T._support(0.30), L.RAIL_H)
        self.assertEqual(T._support(L.STOP_X), L.STOP_H)
        self.assertEqual(T._support(0.10), 0.0)
        # a board commanded 2 mm under the rails is held at them (offline only)
        s = T._offline_board(board(0.30, sole=0.023))
        self.assertAlmostEqual(s.pos[2] - L.TOP - L.BD_Z / 2, L.RAIL_H)

    def test_success_and_score_hooks_feed_the_tracker(self):
        T._TRACKER.reset()
        rigstate.offline_reset()
        objs = {"board": board()}
        self.assertFalse(T.success(objs))
        self.assertEqual(T.score(objs), 0.0)
        self.assertEqual(T._TRACKER.steps, 1)
        self.assertEqual(T.score(objs), 0.0)
        self.assertEqual(T._TRACKER.steps, 1)  # score() does not advance it


class TaskJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((HERE / "task.json").read_text())

    def test_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.spec[k]), n, k)

    def test_no_em_dashes_and_no_team_or_ai_text(self):
        text = json.dumps(self.spec) + (HERE / "README.md").read_text()
        self.assertNotIn("—", text)
        self.assertNotIn("–", text)
        for bad in ("T-70", "Claude", "Anthropic", "Co-Authored", "Generated with", "VihanAggarwal", "robocurve"):
            self.assertNotIn(bad, text)

    def test_the_text_states_the_numbers_the_code_uses(self):
        s = self.spec["scoring"] + self.spec["setup"]
        for needle in ("25 mm", "84 of the 112", "150 mm", "30 mm", "6 mm", "4 mm", "3 mm", "112", "0.25", "0.75", "1 s",
                       "0.3 degrees"):
            self.assertIn(needle, s)
        self.assertEqual(L.CUT_CELLS, 84)
        self.assertEqual((L.TOL_LOOSE, L.TOL_MED, L.TOL_TIGHT), (0.006, 0.004, 0.003))
        self.assertEqual(L.BRIDGE_PULL, 0.150)

    def test_instruction_and_scores_field(self):
        if T is not None:
            self.assertEqual(self.spec["instruction"], T.INSTRUCTION)
        self.assertTrue(self.spec["scores"].startswith("TBD") or "Dream-1" in self.spec["scores"])

    def test_run_limit_covers_twice_the_demo(self):
        m = re.search(r"([\d,]+) steps", self.spec["run_limit"])
        limit = int(m.group(1).replace(",", ""))
        if T is not None:
            demo_steps = sum(w.duration for w in T.demo()["left"]) * HZ
            self.assertGreaterEqual(limit, 2 * demo_steps)


if __name__ == "__main__":
    unittest.main()
