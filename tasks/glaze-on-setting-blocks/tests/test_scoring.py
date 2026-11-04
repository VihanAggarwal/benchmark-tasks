"""Offline tests for scoring.py: synthetic object states, no simulator.  python3 -m unittest discover -s tests -v"""

import contextlib
import importlib.util
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
    Tracker, block_on_sill, block_set_on, block_yaw_err_deg, gap_to_block, in_channel, on_block, rotation,
    share_under, standing, touches_sill, unit_tilt_deg,
)

UH = L.UNIT_SIZE[2]
CX, CY = L.FRAME_X, L.FRAME_Y
HZ = L.CONTROL_HZ


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def compose(*quats):
    """The rotation of applying ``quats`` in turn (world axes), as a quaternion (xyzw)."""
    r = np.eye(3)
    for q in quats:
        r = rotation(q) @ r
    w = np.sqrt(max(1e-12, 1 + np.trace(r))) / 2
    return (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w


def state(center, size, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half  # the world-aligned bounding box of the turned box
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext, size=tuple(size))


def _at(size, x, y, low, quat, **kw):
    """A box whose lowest corner is at height ``low``."""
    drop = float(np.abs(rotation(quat)[2]) @ (np.array(size) / 2))
    return state((x, y, low + drop), size, quat, **kw)


def unit_at(x=CX, y=CY, low=L.SET_BLOCK_TOP, quat=(0, 0, 0, 1), **kw):
    return _at(L.UNIT_SIZE, x, y, low, quat, **kw)


def block_at(x, y, low=L.MARK_TOP, quat=(0, 0, 0, 1), **kw):
    return _at(L.BLOCK_SIZE, x, y, low, quat, **kw)


U_HOME = unit_at(L.UNIT_X, L.UNIT_Y, L.TOP)
B1_HOME = block_at(*L.BLOCK_STARTS[0], L.TRAY_TOP)
B2_HOME = block_at(*L.BLOCK_STARTS[1], L.TRAY_TOP)
B1_SET = block_at(CX, L.MARK_YS[0])  # on the right mark
B2_SET = block_at(CX, L.MARK_YS[1])  # on the left mark
U_ON = unit_at()  # standing on both blocks, centred
U_OVER = unit_at(low=L.STOP_TOP + 0.012)  # held over the channel, 12 mm above the stops
U_SILL = unit_at(low=L.SILL_TOP)  # standing on the bare sill (no blocks)
CARRY = L.STOP_TOP + 0.015


def glide(a, b, n):
    """n states moving a's centre to b's in equal steps, with the matching velocity (the simulator moves things
    gradually); a's orientation."""
    v = (b.pos - a.pos) * HZ / n
    return [state((1 - i / n) * a.pos + (i / n) * b.pos, a.size, a.quat_xyzw, vel=v) for i in range(1, n + 1)]


def hold(s, n):
    return [s] * n


def path(*points, n=20):
    """Glides through a list of states, n steps each."""
    out = []
    for a, b in zip(points, points[1:]):
        out += glide(a, b, n)
    return out


def move_block(home, end):
    """Lift a block off the tray, carry it over the near stop and lower it to ``end``."""
    up = block_at(home.pos[0], home.pos[1], CARRY)
    lane = block_at(0.25, end.pos[1], CARRY)
    over = block_at(end.pos[0], end.pos[1], CARRY)
    final = block_at(end.pos[0], end.pos[1], end.low[2], end.quat_xyzw)
    return path(home, up, lane, over, final) + [final]


def move_unit(end, start=U_HOME):
    """Lift the unit 18 mm clear of the stops, bring it beside the frame and over the channel, lower it to ``end``."""
    lift = L.STOP_TOP + 0.018
    up = unit_at(start.pos[0], start.pos[1], lift)
    lane = unit_at(0.25, CY, lift)
    over = unit_at(CX, CY, lift)
    entry = unit_at(end.pos[0], end.pos[1], end.low[2] + 0.020)
    final = state(end.pos, end.size, end.quat_xyzw)
    return path(start, up, lane, over, entry, n=20) + glide(entry, final, 50) + [final]


def run(units, b1s, b2s, tracker=None):
    """Feed matching unit / block states (lists of equal length) to a fresh tracker; return the last result."""
    assert len(units) == len(b1s) == len(b2s)
    t = tracker or Tracker(HZ)
    r = None
    for u, a, b in zip(units, b1s, b2s):
        r = t.update(u, a, b)
    return r, t


def procedure(b1_end=B1_SET, b2_end=B2_SET, unit_end=U_ON, settle=70):
    """The intended sequence: block 1 to its mark, block 2 to its mark, then the unit into the channel onto them,
    then everything still for ``settle`` steps."""
    m1, m2 = move_block(B1_HOME, b1_end), move_block(B2_HOME, b2_end)
    us = hold(U_HOME, 5 + len(m1) + len(m2))
    b1 = hold(B1_HOME, 5) + m1 + hold(m1[-1], len(m2))
    b2 = hold(B2_HOME, 5 + len(m1)) + m2
    mu = move_unit(unit_end)
    us += mu + hold(mu[-1], settle)
    b1 += hold(b1[-1], len(mu) + settle)
    b2 += hold(b2[-1], len(mu) + settle)
    return us, b1, b2


def with_tail(seq, tail, n):
    """Replace the last n states of a sequence by ``tail`` (a state)."""
    return seq[:-n] + hold(tail, n)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_plain_box(self):
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("y", 2.5))), 2.5, places=4)
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("x", 180))), 0.0, places=4)  # on its other long edge
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("x", 177))), 3.0, places=4)
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("z", 40))), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("y", 90))), 90.0, places=4)  # lying on a face
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("x", 90))), 90.0, places=4)  # stood on its end

    def test_block_yaw_has_two_ways_round(self):
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 10)), 10.0, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 175)), 5.0, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 90)), 90.0, places=4)  # 30 mm across: a different block
        self.assertAlmostEqual(block_yaw_err_deg(about("x", 180)), 0.0, places=4)  # flipped onto its other face

    def test_touching_the_sill(self):
        self.assertTrue(touches_sill(U_SILL))
        self.assertTrue(touches_sill(unit_at(low=L.SILL_TOP + 0.0019)))
        self.assertFalse(touches_sill(unit_at(low=L.SILL_TOP + 0.0021)))  # 2.1 mm up
        self.assertFalse(touches_sill(U_ON))  # on the blocks: 9 mm up
        self.assertFalse(touches_sill(U_HOME))  # low, but on the table outside the frame
        self.assertFalse(touches_sill(unit_at(0.25, 0.0, L.TOP)))  # on the table beside the frame
        # tipped about x on one block: its far corner comes down to 1.5 mm above the sill
        q = about("x", 4.5)
        self.assertTrue(touches_sill(unit_at(low=L.SILL_TOP + 0.0015, quat=q)))
        self.assertFalse(touches_sill(unit_at(low=L.SILL_TOP + 0.003, quat=q)))

    def test_in_the_channel(self):
        self.assertTrue(in_channel(U_ON))
        self.assertTrue(in_channel(unit_at(CX + 0.0085)))  # its face 0.5 mm into the far stop: slack
        self.assertFalse(in_channel(unit_at(CX + 0.0095)))
        self.assertTrue(in_channel(unit_at(y=0.020)))  # its end at the jamb's face
        self.assertFalse(in_channel(unit_at(y=0.0215)))
        self.assertFalse(in_channel(unit_at(0.25, 0.0, L.STOP_TOP + 0.012)))  # beside the frame
        self.assertTrue(standing(U_ON))
        self.assertTrue(standing(unit_at(quat=about("y", 2.9))))
        self.assertFalse(standing(unit_at(quat=about("y", 3.1))))

    def test_block_on_the_sill(self):
        self.assertTrue(block_on_sill(B1_SET))
        self.assertTrue(block_on_sill(block_at(CX, 0.0, L.SILL_TOP)))  # on the bare sill between the marks
        self.assertTrue(block_on_sill(block_at(CX, L.MARK_YS[0], quat=about("x", 180))))  # other face up
        self.assertFalse(block_on_sill(block_at(CX, L.MARK_YS[0], L.SILL_TOP + 0.0035)))  # held above it
        self.assertFalse(block_on_sill(block_at(CX, L.MARK_YS[0], L.SILL_TOP, quat=about("y", 90))))  # on edge
        self.assertFalse(block_on_sill(B1_HOME))  # on the tray
        self.assertFalse(block_on_sill(block_at(CX, 0.125)))  # centre past the jamb's face

    def test_block_set_on_a_mark(self):
        self.assertEqual(block_set_on(B1_SET, L.BLOCK_TIGHT), 0)
        self.assertEqual(block_set_on(B2_SET, L.BLOCK_TIGHT), 1)
        self.assertEqual(block_set_on(block_at(CX + 0.005, L.MARK_YS[1] + 0.006), L.BLOCK_TIGHT), 1)  # 7.8 mm
        self.assertIsNone(block_set_on(block_at(CX, L.MARK_YS[1] + 0.0082), L.BLOCK_TIGHT))
        self.assertEqual(block_set_on(block_at(CX, L.MARK_YS[1] + 0.0082), L.BLOCK_LOOSE), 1)
        self.assertEqual(block_set_on(block_at(CX, L.MARK_YS[0], quat=about("z", 14.5)), L.BLOCK_TIGHT), 0)
        self.assertIsNone(block_set_on(block_at(CX, L.MARK_YS[0], quat=about("z", 15.5)), L.BLOCK_TIGHT))
        self.assertIsNone(block_set_on(block_at(CX, 0.0, L.SILL_TOP), L.BLOCK_LOOSE))  # between the marks

    def test_resting_on_a_block(self):
        self.assertTrue(on_block(U_ON, B1_SET) and on_block(U_ON, B2_SET))
        self.assertAlmostEqual(gap_to_block(U_ON, B1_SET), 0.0, places=9)
        self.assertTrue(on_block(unit_at(low=L.SET_BLOCK_TOP + 0.0019), B1_SET))
        self.assertFalse(on_block(unit_at(low=L.SET_BLOCK_TOP + 0.0021), B1_SET))  # held above it
        self.assertTrue(on_block(unit_at(low=L.SET_BLOCK_TOP - 0.0019), B1_SET))  # solver penetration
        self.assertFalse(on_block(unit_at(low=L.SET_BLOCK_TOP - 0.0021), B1_SET))
        self.assertFalse(on_block(U_ON, B1_HOME))
        self.assertAlmostEqual(share_under(U_ON, B1_SET), 1.0, places=6)
        # the unit 8 mm toward the far stop and the block 4 mm toward the near one: 12 of its 24 mm under the unit
        b = block_at(CX - 0.004, L.MARK_YS[0])
        self.assertAlmostEqual(share_under(unit_at(CX + 0.008), b), 0.5, places=6)
        self.assertTrue(on_block(unit_at(CX + 0.0079), b))
        self.assertFalse(on_block(unit_at(CX + 0.0081), b))

    def test_share_under_matches_sampling(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            u = unit_at(CX + rng.uniform(-0.008, 0.008), rng.uniform(-0.02, 0.02),
                        quat=compose(about("z", rng.uniform(-4, 4)), about("y", rng.uniform(-2, 2))))
            b = block_at(CX + rng.uniform(-0.015, 0.015), L.MARK_YS[0] + rng.uniform(-0.01, 0.01),
                         quat=about("z", rng.uniform(-60, 60)))
            s, t_ = np.meshgrid(np.linspace(-0.5, 0.5, 121), np.linspace(-0.5, 0.5, 151))
            top = np.stack([s.ravel() * L.BLOCK_SIZE[0], t_.ravel() * L.BLOCK_SIZE[1], np.zeros(s.size)], 1)
            world = top @ rotation(b.quat_xyzw).T + b.pos
            r = rotation(u.quat_xyzw)
            under = np.array(u.pos) - r[:, 2] * UH / 2  # centre of the unit's underside
            local = (world[:, :2] - under[:2]) @ np.linalg.inv(r[:2, :2]).T  # into the underside's own axes
            inside = (np.abs(local[:, 0]) <= L.UNIT_SIZE[0] / 2) & (np.abs(local[:, 1]) <= L.UNIT_SIZE[1] / 2)
            self.assertAlmostEqual(share_under(u, b), inside.mean(), delta=0.02)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(hold(U_HOME, 20), hold(B1_HOME, 20), hold(B2_HOME, 20))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = run(*procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertFalse(t.touched)
        self.assertAlmostEqual(r.details["min_low_in_channel_mm"], 9.0, places=3)

    def test_either_way_up_scores_one(self):
        flipped_unit = unit_at(quat=about("x", 180))
        flipped_block = block_at(CX, L.MARK_YS[0], quat=about("y", 180))
        r, _ = run(*procedure(b1_end=flipped_block, unit_end=flipped_unit))
        self.assertEqual(r.score, 1.0)

    def test_limits_for_the_blocks(self):
        for dy, want in ((0.0079, 1.0), (0.0081, 0.75), (0.0199, 0.75), (0.0201, 0.5)):
            b2 = block_at(CX, L.MARK_YS[1] + dy)
            r, _ = run(*procedure(b2_end=b2))
            self.assertEqual(r.score, want, dy)
        r, _ = run(*procedure(b1_end=block_at(CX + 0.0055, L.MARK_YS[0] - 0.0055)))  # 7.8 mm diagonally
        self.assertEqual(r.score, 1.0)

    def test_limits_for_the_unit(self):
        for dy, want in ((0.0079, 1.0), (-0.0081, 0.75), (0.0149, 0.75), (-0.0151, 0.5)):
            r, _ = run(*procedure(unit_end=unit_at(y=dy)))
            self.assertEqual(r.score, want, dy)

    def test_a_turned_block_scores_half(self):
        r, _ = run(*procedure(b1_end=block_at(CX, L.MARK_YS[0], quat=about("z", 20))))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["blocks_set"])

    def test_blocks_at_the_ends_instead_of_the_quarter_points_score_half(self):
        r, _ = run(*procedure(b1_end=block_at(CX, -0.09), b2_end=block_at(CX, 0.09, L.SILL_TOP)))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["on_block"], [True, True])

    def test_standing_on_one_block_scores_half(self):
        # the second block left on the tray; the unit held level on the first (a released one would tip: below)
        us, b1, b2 = procedure()
        r, _ = run(us, b1, hold(B2_HOME, len(us)))
        self.assertEqual(r.details["on_block"], [True, False])
        self.assertEqual(r.score, 0.5)

    def test_on_one_block_and_held_tilted_above_the_other_scores_half(self):
        # both blocks set, the unit held 2.5 degrees off level: down on the right-hand block, 4.4 mm over the other
        q = about("x", 2.5)
        u0 = state((CX, CY, L.SET_BLOCK_TOP + UH / 2), L.UNIT_SIZE, q)
        tilted = state(u0.pos - np.array([0, 0, gap_to_block(u0, B1_SET)]), L.UNIT_SIZE, q)
        self.assertTrue(standing(tilted) and on_block(tilted, B1_SET))
        self.assertGreater(gap_to_block(tilted, B2_SET), 0.004)
        self.assertFalse(touches_sill(tilted))
        r, _ = run(*procedure(unit_end=tilted))
        self.assertEqual(r.details["on_block"], [True, False])
        self.assertTrue(r.details["blocks_set"])
        self.assertEqual(r.score, 0.5)

    def test_unit_too_far_off_centre_scores_half(self):
        r, _ = run(*procedure(unit_end=unit_at(y=0.020)))  # its end against the left jamb
        self.assertEqual(r.score, 0.5)

    def test_blocks_set_only_scores_quarter(self):
        us, b1, b2 = procedure()
        r, _ = run(hold(U_HOME, len(us)), b1, b2)  # the right arm alone
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_blocks_set_unit_held_above_scores_quarter(self):
        us, b1, b2 = procedure(unit_end=unit_at(low=L.SET_BLOCK_TOP + 0.005))
        r, _ = run(us, b1, b2)
        self.assertEqual(r.score, 0.25)

    def test_blocks_set_unit_leaning_scores_quarter(self):
        us, b1, b2 = procedure()
        leaning = unit_at(CX - 0.004, low=L.SET_BLOCK_TOP, quat=about("y", 6))  # its top against the near stop
        r, _ = run(with_tail(us, leaning, 70), b1, b2)
        self.assertFalse(r.details["standing"])
        self.assertEqual(r.score, 0.25)

    def test_blocks_set_loosely_and_unit_not_on_them_scores_zero(self):
        us, b1, b2 = procedure(b2_end=block_at(CX, L.MARK_YS[1] + 0.012))
        r, _ = run(hold(U_HOME, len(us)), b1, b2)
        self.assertEqual(r.score, 0.0)

    def test_one_block_set_only_scores_zero(self):
        us, b1, b2 = procedure()
        r, _ = run(hold(U_HOME, len(us)), b1, hold(B2_HOME, len(us)))
        self.assertEqual(r.score, 0.0)

    def test_unit_lowered_onto_the_bare_sill_scores_zero(self):
        # the left arm alone: no blocks, the unit lowered 2 mm further than their tops would be
        mu = move_unit(U_SILL)
        r, t = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_touching_the_sill_scores_zero_whatever_follows(self):
        # lowered onto the sill first, lifted out, then the blocks and the unit put in correctly
        dip = move_unit(U_SILL)
        out = path(U_SILL, U_OVER, unit_at(0.25, CY, L.STOP_TOP + 0.018), unit_at(L.UNIT_X, L.UNIT_Y, L.TOP))
        us, b1, b2 = procedure()
        pre = dip + out
        r, t = run(pre + us, hold(B1_HOME, len(pre)) + b1, hold(B2_HOME, len(pre)) + b2)
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_lying_down_or_on_end_scores_zero(self):
        us, b1, b2 = procedure()
        for q in (about("y", 90), about("x", 90), about("y", 31)):
            fallen = unit_at(0.20, 0.20, L.TOP, quat=q)  # outside the frame, so it never touched the sill
            r, t = run(with_tail(us, fallen, 30), b1, b2)
            self.assertFalse(t.touched)
            self.assertTrue(r.details["lying_down"])
            self.assertEqual(r.score, 0.0)

    def test_unit_off_the_table_scores_zero(self):
        us, b1, b2 = procedure()
        r, _ = run(with_tail(us, unit_at(0.20, 0.70, 0.0), 30), b1, b2)
        self.assertTrue(r.details["off_table"])
        self.assertEqual(r.score, 0.0)

    def test_success_needs_stillness(self):
        us, b1, b2 = procedure(settle=0)
        moving = state(U_ON.pos, U_ON.size, vel=(0, 0, -0.012))
        r, _ = run(us + hold(moving, 70), b1 + hold(B1_SET, 70), b2 + hold(B2_SET, 70))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        spinning = state(B2_SET.pos, B2_SET.size, spin=(0, 0, 0.6))
        r, _ = run(us + hold(U_ON, 70), b1 + hold(B1_SET, 70), b2 + hold(spinning, 70))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)

    def test_score_is_read_from_the_state_now(self):
        # result() of the same tracker on other states: the record (touched) is kept, the rest is re-read
        r, t = run(*procedure())
        self.assertEqual(t.result(U_HOME, B1_SET, B2_SET).score, 0.25)
        self.assertEqual(t.result(U_ON, B1_SET, B2_SET).score, 1.0)


class Exploits(unittest.TestCase):
    def test_a_unit_stood_on_one_block_tips_onto_the_sill(self):
        # let go on the right-hand block only: it tips about that block's edge until its far end is on the sill
        us, b1, b2 = procedure()
        pivot_y = L.MARK_YS[0] + L.BLOCK_SIZE[1] / 2
        theta = np.degrees(np.arctan2(L.SET_BLOCK_TOP - L.SILL_TOP, 0.10 - pivot_y))
        tipped = unit_at(low=L.SILL_TOP, quat=about("x", theta))
        r, t = run(us + glide(U_ON, tipped, 4) + hold(tipped, 40), b1 + hold(B1_SET, 44),
                   hold(B2_HOME, len(us) + 44))
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_a_momentary_dip_onto_the_sill_counts(self):
        # rocked so a corner is down on the sill for one control step, then back on both blocks
        us, b1, b2 = procedure()
        rocked = unit_at(low=L.SILL_TOP + 0.001, quat=about("x", 4.5))
        k = len(us) - 60
        r, t = run(us[:k] + [rocked] + us[k + 1:], b1, b2)
        self.assertEqual(t.touched_at, k + 1)
        self.assertEqual(r.score, 0.0)

    def test_both_blocks_stacked_on_one_mark_score_zero(self):
        # a taller support on one mark: the upper block is not on the sill, the unit is 8 mm above the lower one
        stacked = block_at(CX, L.MARK_YS[0], L.SET_BLOCK_TOP)
        us, b1, b2 = procedure(b2_end=stacked, unit_end=unit_at(low=L.SET_BLOCK_TOP + L.BLOCK_SIZE[2]))
        r, _ = run(us, b1, b2)
        self.assertEqual(r.details["on_block"], [False, False])
        self.assertEqual(r.score, 0.0)

    def test_both_blocks_crowded_onto_one_mark_score_half_at_best(self):
        # end to end, 15 mm either side of the right-hand mark: each within 20 mm of it, none on the left mark
        b1, b2 = block_at(CX, L.MARK_YS[0] - 0.015), block_at(CX, L.MARK_YS[0] + 0.015, L.SILL_TOP)
        self.assertEqual((block_set_on(b1, L.BLOCK_LOOSE), block_set_on(b2, L.BLOCK_LOOSE)), (0, 0))
        r, _ = run(*procedure(b1_end=b1, b2_end=b2, unit_end=unit_at()))  # held level on them
        self.assertFalse(r.details["blocks_set_loose"])
        self.assertEqual(r.score, 0.5)

    def test_a_block_stood_on_edge_is_not_a_setting_block(self):
        on_edge = block_at(CX, L.MARK_YS[1], L.MARK_TOP, quat=about("y", 90))  # 24 mm tall
        us, b1, b2 = procedure(b2_end=on_edge, unit_end=unit_at(low=L.MARK_TOP + L.BLOCK_SIZE[0]))
        r, _ = run(us, b1, b2)
        self.assertFalse(r.details["block_on_sill"][1])
        self.assertEqual(r.score, 0.0)

    def test_the_unit_on_the_blocks_while_they_are_still_on_the_tray(self):
        # a 'set' needs the sill: standing the unit on the blocks on the tray (or held over it) is worth nothing
        on_tray = unit_at(L.TRAY_X, L.TRAY_Y, L.TRAY_TOP + L.BLOCK_SIZE[2])
        self.assertFalse(on_block(on_tray, B1_HOME))
        r, _ = run(hold(U_HOME, 5) + hold(on_tray, 70), hold(B1_HOME, 75), hold(B2_HOME, 75))
        self.assertEqual(r.score, 0.0)

    def test_dropping_the_unit_in_without_blocks_then_adding_them_scores_zero(self):
        # glass first, then the blocks pushed in underneath: it has already been down on the sill
        mu = move_unit(U_SILL)
        lifted = unit_at(low=L.SET_BLOCK_TOP + 0.003)
        us = mu + glide(U_SILL, lifted, 20)
        m1, m2 = move_block(B1_HOME, B1_SET), move_block(B2_HOME, B2_SET)
        us += hold(lifted, len(m1) + len(m2)) + glide(lifted, U_ON, 10) + hold(U_ON, 70)
        b1 = hold(B1_HOME, len(mu) + 20) + m1 + hold(B1_SET, len(m2) + 80)
        b2 = hold(B2_HOME, len(mu) + 20 + len(m1)) + m2 + hold(B2_SET, 80)
        r, t = run(us, b1, b2)
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_laying_the_unit_flat_across_the_stops_scores_zero(self):
        across = unit_at(CX, 0.0, L.STOP_TOP, quat=about("y", 90))
        us, b1, b2 = procedure()
        r, t = run(with_tail(us, across, 40), b1, b2)
        self.assertFalse(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_held_still_within_2mm_counts_as_resting(self):
        # the scorer sees no gripper (stated in fidelity): held 1.5 mm above both blocks is resting on them
        r, _ = run(*procedure(unit_end=unit_at(low=L.SET_BLOCK_TOP + 0.0015)))
        self.assertEqual(r.score, 1.0)


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        return [
            (unit_at(x=nan), B1_SET, B2_SET),
            (state(U_ON.pos, U_ON.size, quat=(nan, 0, 0, nan)), B1_SET, B2_SET),
            (U_ON, block_at(CX, inf), B2_SET),
            (U_ON, B1_SET, state(B2_SET.pos, B2_SET.size, vel=(0, nan, 0))),
            (state(U_ON.pos, U_ON.size, quat=(0, 0, 0, 0)), B1_SET, B2_SET),
            (state(U_ON.pos, U_ON.size, spin=(inf, 0, 0)), B1_SET, B2_SET),
            (unit_at(50.0, -40.0, -10.0, quat=about("y", 123)), block_at(-3.0, 8.0, 100.0, quat=about("x", 77)),
             block_at(1e6, -1e6, -1e6)),
            (unit_at(quat=about("x", 180)), block_at(CX, -0.05, quat=about("y", 180)), B2_HOME),
        ]

    def test_no_crash_and_clean_details(self):
        for u, a, b in self.odd():
            t = Tracker()
            t.update(U_HOME, B1_HOME, B2_HOME)
            r = t.update(u, a, b)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)
            json.dumps(t.result(u, a, b).details, allow_nan=False)

    def test_blow_ups_score_zero_and_leave_the_tracker_alone(self):
        for k, (u, a, b) in enumerate(self.odd()[:6]):
            us, b1, b2 = procedure()
            i = len(us) - 80
            t = Tracker()
            for j, (uu, aa, bb) in enumerate(zip(us, b1, b2)):
                r = t.update(*((u, a, b) if j == i else (uu, aa, bb)))
                if j == i:
                    self.assertEqual(r.score, 0.0, k)
                    self.assertFalse(r.details["valid_state"])
            self.assertEqual(r.score, 1.0, k)
            self.assertTrue(r.success, k)
            self.assertEqual(t.bad_steps, 1)
            self.assertFalse(t.touched)

    def test_random_states_give_a_listed_score(self):
        rng = np.random.default_rng(11)
        t = Tracker()
        for _ in range(300):
            def rnd(size, centre):
                q = rng.normal(size=4)
                return state(np.array(centre) + rng.normal(scale=0.03, size=3), size, q / np.linalg.norm(q),
                             vel=rng.normal(scale=0.01, size=3))
            r = t.update(rnd(L.UNIT_SIZE, U_ON.pos), rnd(L.BLOCK_SIZE, B1_SET.pos), rnd(L.BLOCK_SIZE, B2_SET.pos))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        mu = move_unit(U_SILL)
        r, t = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        self.assertTrue(t.touched)
        r = t.update(U_HOME, B1_HOME, B2_HOME)  # everything jumps back in one step: a new episode
        self.assertFalse(t.touched)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        r, _ = run(*procedure(), tracker=t)  # and the new episode can score 1.0
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        mu = move_unit(U_SILL)
        _, t = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        u = unit_at(L.UNIT_X + 0.03, L.UNIT_Y - 0.02, L.TOP, quat=about("z", 8))
        a = block_at(L.BLOCK_STARTS[0][0] - 0.02, L.BLOCK_STARTS[0][1] + 0.03, L.TRAY_TOP, quat=about("z", -10))
        b = block_at(L.BLOCK_STARTS[1][0] + 0.01, L.BLOCK_STARTS[1][1] - 0.03, L.TRAY_TOP)
        t.update(u, a, b)
        self.assertFalse(t.touched)
        self.assertEqual(t.steps, 1)

    def test_teleport_after_the_demo_resets(self):
        _, t = run(*procedure())
        t.touched = True  # whatever happened in that episode
        t.update(U_HOME, B1_HOME, B2_HOME)
        self.assertEqual(t.steps, 1)
        self.assertFalse(t.touched)

    def test_putting_things_back_slowly_does_not_reset(self):
        mu = move_unit(U_SILL)
        back = path(U_SILL, U_OVER, unit_at(0.25, CY, L.STOP_TOP + 0.018), U_HOME)
        us = mu + back + hold(U_HOME, 10)
        r, t = run(us, hold(B1_HOME, len(us)), hold(B2_HOME, len(us)))
        self.assertTrue(t.touched)  # back at the start, gradually: the same episode
        self.assertEqual(t.steps, len(us))

    def test_dropping_the_unit_back_onto_its_start_is_not_a_reset(self):
        # lowered onto the sill, carried back high over its start and let fall: its last sampled step before
        # landing is more than RESET_JUMP, but its speed then explains it
        mu = move_unit(U_SILL)
        high = unit_at(L.UNIT_X, L.UNIT_Y, L.TOP + 0.40)
        us = mu + path(U_SILL, U_OVER, unit_at(L.UNIT_X, L.UNIT_Y, L.STOP_TOP + 0.012), high)
        g, dt = L.GRAVITY, 1 / HZ
        h, v, last = 0.40, 0.0, None
        while True:  # free fall, sampled every control step
            v, h = v + g * dt, h - v * dt - 0.5 * g * dt * dt
            if h <= 0:
                break
            last = unit_at(L.UNIT_X, L.UNIT_Y, L.TOP + h, vel=(0, 0, -v))
            us.append(last)
        self.assertGreater(float(np.linalg.norm(last.pos - U_HOME.pos)), L.RESET_JUMP)
        us.append(U_HOME)  # landed on its start, still
        r, t = run(us, hold(B1_HOME, len(us)), hold(B2_HOME, len(us)))
        self.assertTrue(t.touched)
        self.assertEqual(t.steps, len(us))

    def test_a_jump_elsewhere_is_not_a_reset(self):
        mu = move_unit(U_SILL)
        _, t = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        t.update(unit_at(0.20, -0.10, L.TOP), B1_HOME, B2_HOME)  # a 10 cm jump, but not to its start
        self.assertTrue(t.touched)
        self.assertEqual(t.steps, len(mu) + 1)

    def test_the_blocks_must_be_back_on_the_tray_too(self):
        us, b1, b2 = procedure()
        _, t = run(us, b1, b2)
        t.touched = True
        t.update(U_HOME, B1_SET, B2_SET)  # only the unit jumps back: not the starting layout
        self.assertTrue(t.touched)


STUB = HERE.parent.parent / "tools" / "offline" / "yam_stub"


def load_task():
    """task.py, imported through the offline stand-in for yam_arena."""
    if str(STUB) not in sys.path:
        sys.path.insert(0, str(STUB))
    spec = importlib.util.spec_from_file_location("glaze_task_under_test", HERE / "task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(STUB.is_dir(), "needs the offline yam_arena stub (tools/offline/yam_stub)")
class DemoTiming(unittest.TestCase):
    """The scripted demo's timings against the simulator's limits (no simulator: the plan's numbers only)."""

    HOLD_S = 3.0  # yam_scripted's default hold after the plan before it stops the episode (-P hold_s)

    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        old = {k: os.environ.pop(k, None) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
        try:
            cls.plan = cls.task.demo()
        finally:
            os.environ.update({k: v for k, v in old.items() if v is not None})

    def segments(self, arm):
        """(start time, end time, previous waypoint or None, waypoint) for each waypoint of an arm's plan."""
        t, prev, out = 0.0, None, []
        for w in self.plan[arm]:
            out.append((t, t + w.duration, prev, w))
            t, prev = t + w.duration, w
        return out

    def test_grips_and_releases_are_slow_enough(self):
        # every change of the finger target: at least 1.5 s, and under the CLI's 0.05-per-step gripper limiter at
        # 20 Hz (smoothstep peaks at 1.5 times its mean rate)
        for arm in ("left", "right"):
            finger = self.task.OPEN  # each arm starts open
            for _, _, _, w in self.segments(arm):
                change = abs(w.finger - finger) / abs(self.task.OPEN)
                if change > 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertLessEqual(1.5 * change / (w.duration * HZ), 0.05, (arm, w.label))
                finger = w.finger

    def test_descents_into_the_channel_are_slow(self):
        # any waypoint that ends over the channel at least 1 cm lower than it started takes 1.5 s or more
        for arm in ("left", "right"):
            for _, _, prev, w in self.segments(arm):
                if prev is None:
                    continue
                over = abs(w.pos[0] - CX) < L.CHANNEL_HALF_W and abs(w.pos[1] - CY) < L.CHANNEL_HALF_L
                if over and prev.pos[2] - w.pos[2] > 0.01:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))

    def test_trims_are_zero_and_the_releases_are_on_the_marks(self):
        self.assertEqual(self.task.PLACE_TRIM, (0.0, 0.0))
        self.assertEqual(self.task.BLOCK_TRIM, ((0.0, 0.0), (0.0, 0.0)))
        lets = [w for w in self.plan["right"] if w.label == "let go of the block"]
        self.assertEqual([tuple(np.round(w.pos[:2], 6)) for w in lets], [(CX, L.MARK_YS[0]), (CX, L.MARK_YS[1])])
        unit = [w for w in self.plan["left"] if w.label == "let go"]
        self.assertEqual(len(unit), 1)
        self.assertEqual(tuple(np.round(unit[0].pos[:2], 6)), (CX, CY))

    @staticmethod
    def _smoothstep_at(s):
        """The fraction of a smoothstep segment's time at which it has covered ``s`` of its way."""
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mid * mid * (3 - 2 * mid) < s else (lo, mid)
        return lo

    def _hold_end(self, sag):
        """With the arm sagging ``sag`` under the unit (none offline): when the unit first rests on the blocks, when
        the 3 s success hold is up, the commanded finger gap (m) then, and when the release ends."""
        seg = {w.label: (t0, t1) for t0, t1, _, w in self.segments("left")}
        d0, d1 = seg["lower the last 2 cm slowly"]
        r0, r1 = seg["let go"]
        grip, let_go = -2 * self.task.UNIT_GRIP, -2 * self.task.UNIT_LET_GO
        drop = self.task.U_PLACE_Z - L.SET_BLOCK_TOP - (self.task.UNIT_TCP_UP)  # 2 mm let-go gap, commanded
        entry = self.task.U_ENTRY_Z - self.task.U_PLACE_Z  # the slow last 20 mm
        if sag <= drop:  # it rests only once the pads let go (opened to the unit's thickness)
            rest = r0 + self._smoothstep_at((L.UNIT_SIZE[0] - grip) / (let_go - grip)) * (r1 - r0)
        else:  # it rests when the commanded bottom is ``sag`` above the blocks' tops
            rest = d0 + self._smoothstep_at(1 - (sag - drop) / entry) * (d1 - d0)
        end = rest + self.task.SUCCESS_HOLD_S
        tau = min(1.0, max(0.0, (end - r0) / (r1 - r0)))
        return rest, end, grip + (let_go - grip) * tau * tau * (3 - 2 * tau), r1

    def test_success_cannot_hold_while_the_gripper_holds_the_glass(self):
        # the estimated 5.6 mm of sag: the unit rests before the slow descent ends; the 3 s are up only after the
        # release has ended (fingers fully open)
        rest, end, gap, released = self._hold_end(0.0056)
        self.assertLess(rest, released - 2.0)
        self.assertGreater(end, released)
        self.assertAlmostEqual(gap, -2 * self.task.UNIT_LET_GO, places=6)
        # 15 mm of sag: the pads at least 6 mm clear of each face when the 3 s are up
        self.assertGreater(self._hold_end(0.015)[2], L.UNIT_SIZE[0] + 0.012)

    def test_success_comes_before_the_policy_stops_and_the_run_limit(self):
        stop = max(sum(w.duration for w in self.plan[a]) for a in ("left", "right")) + self.HOLD_S
        for sag in (0.0, 0.0056, 0.015):
            _, end, _, _ = self._hold_end(sag)
            self.assertLess(end, stop - 1.0, sag)  # a second of margin
        self.assertLessEqual(round(stop * HZ), 1200)  # task.json: run limit 1,200 steps


@unittest.skipUnless(STUB.is_dir(), "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py itself, imported through the offline stand-in for yam_arena (not part of a submission)."""

    @contextlib.contextmanager
    def env(self, **kw):
        old = {k: os.environ.get(k) for k in kw}
        os.environ.update({k: v for k, v in kw.items() if v is not None})
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def load(self):
        return load_task()

    def start(self):
        return {"glass_unit": U_HOME, "block_1": B1_HOME, "block_2": B2_HOME}

    def test_demo_switches(self):
        task = self.load()
        with self.env(DEMO_IDLE_ARM=None, DEMO_HOLD_ONLY=None):
            self.assertEqual(set(task.demo()), {"left", "right"})
        with self.env(DEMO_IDLE_ARM="left", DEMO_HOLD_ONLY=None):
            self.assertEqual(set(task.demo()), {"right"})
        with self.env(DEMO_IDLE_ARM="right", DEMO_HOLD_ONLY=None):
            self.assertEqual(set(task.demo()), {"left"})
        with self.env(DEMO_HOLD_ONLY="1", DEMO_IDLE_ARM=None):
            self.assertEqual(task.demo(), {})

    def test_the_arms_are_sequenced(self):
        # the left arm starts over the channel only after the right arm has parked
        task = self.load()
        with self.env(DEMO_IDLE_ARM=None, DEMO_HOLD_ONLY=None):
            plan = task.demo()
        t, starts = 0.0, {}
        for w in plan["left"]:
            starts.setdefault(w.label, t)
            t += w.duration
        self.assertGreaterEqual(starts["beside the frame, level with the channel"] + 1e-9,
                                sum(w.duration for w in plan["right"]))

    def test_hooks_and_trace(self):
        task = self.load()
        with tempfile.TemporaryDirectory() as d:
            with self.env(TASK_TRACE=d):  # a directory: the trace cannot be written, the trial goes on
                self.assertFalse(task.success(self.start()))
            trace = os.path.join(d, "trace.jsonl")
            with self.env(TASK_TRACE=trace):
                self.assertFalse(task.success(self.start()))
            with open(trace) as f:
                row = json.loads(f.read().splitlines()[-1])
            for key in ("step", "score", "touched", "min_low_in_channel_mm", "on_block", "gap_mm", "block_mark_mm",
                        "unit_pos", "block_1_pos"):
                self.assertIn(key, row)
        self.assertEqual(task.score(self.start()), 0.0)
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("glass_unit", names)


if __name__ == "__main__":
    unittest.main()
