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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    corners,
    Tracker, block_on_sill, block_set_on, block_yaw_err_deg, gap_to_block, hand_gap, in_channel, let_go_now,
    on_block, rotation, share_under, sill_gaps, standing, touches_sill, unit_tilt_deg,
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


def arm(tcp, opening=1.0):
    """A gripper pointing straight down (pad-tip point 13 mm below the TCP), as rigstate reports it."""
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=float(opening))


AWAY = {"left": arm((0.32, 0.305, 1.0)), "right": arm((0.32, -0.305, 1.0))}  # both up at the ready pose, open
GRIP_OPENING = (L.UNIT_SIZE[0] - 0.008) / rigstate.FULL_GAP  # the demo's 16 mm grip on the unit


def holding_unit(u):
    """The left gripper closed on the unit's top edge (the demo's grip: TCP 16 mm below the top), the right away."""
    top = float(corners_top(u))
    return {"left": arm((u.pos[0], u.pos[1], top - 0.016), GRIP_OPENING), "right": AWAY["right"]}


def holding_block(b):
    """The right gripper closed across a block (pad tips 2.5 mm above its bottom), the left away."""
    return {"left": AWAY["left"], "right": arm((b.pos[0], b.pos[1], b.low[2] + 0.0155), 0.022 / rigstate.FULL_GAP)}


def corners_top(s):
    return s.high[2]


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


def run(units, b1s, b2s, tracker=None, arms=AWAY):
    """Feed matching unit / block states (lists of equal length) to a fresh tracker; return the last result.
    ``arms``: what rigstate reports each step (a list, one per step), or the same every step (None: unreadable).
    By default both grippers are up and away (let go of everything)."""
    assert len(units) == len(b1s) == len(b2s)
    arms_seq = arms if isinstance(arms, list) else [arms] * len(units)
    assert len(arms_seq) == len(units)
    t = tracker or Tracker(HZ)
    r = None
    for u, a, b, g in zip(units, b1s, b2s, arms_seq):
        r = t.update(u, a, b, g)
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
        # the rule is contact: a part of the unit inside the channel 0.5 mm or less above the surface under it
        self.assertEqual(L.SILL_TOUCH_GAP, 0.0005)
        self.assertTrue(touches_sill(U_SILL))
        self.assertFalse(touches_sill(U_ON))  # on the blocks: 9 mm up
        self.assertFalse(touches_sill(U_HOME))  # low, but on the table outside the frame
        self.assertFalse(touches_sill(unit_at(0.25, 0.0, L.TOP)))  # on the table beside the frame
        self.assertFalse(touches_sill(U_OVER))

    def test_a_level_unit_on_the_two_marks_is_a_touch(self):
        # lowered with no blocks it rests on the marks along its bottom edges, its corners 1 mm above the bare sill
        u = unit_at(low=L.MARK_TOP)
        self.assertTrue(touches_sill(u))
        corner_gaps = [p[2] - L.SILL_TOP for p in corners(u, np.array(L.UNIT_SIZE) / 2) if p[2] < L.SILL_TOP + 0.002]
        self.assertTrue(corner_gaps and all(abs(g - 0.001) < 1e-9 for g in corner_gaps))
        self.assertAlmostEqual(float(sill_gaps(u).min()), 0.0, places=9)

    def test_bottom_edge_over_a_mark_half_a_millimetre(self):
        self.assertTrue(touches_sill(unit_at(low=L.MARK_TOP + 0.0004)))  # 0.4 mm above a mark's top
        self.assertFalse(touches_sill(unit_at(low=L.MARK_TOP + 0.0006)))  # 0.6 mm: a near miss

    def test_contact_on_a_mark_outline_between_edge_samples(self):
        # a pose whose lowest point over a mark lies on the mark's outline between two edge samples (true gap 0.038 mm)
        pos, q = (0.3, 0.0045, 0.840757), (0.241922, 0, 0, 0.970296)  # quaternion xyzw
        u = SimpleNamespace(pos=np.array(pos), quat_xyzw=np.array(q), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3))
        self.assertTrue(touches_sill(u))
        self.assertLess(float(sill_gaps(u).min()), 0.0001)
        up = SimpleNamespace(**{**u.__dict__, "pos": np.array(pos) + np.array([0, 0, 0.0007])})
        self.assertFalse(touches_sill(up))  # raised 0.7 mm: a near miss

    def test_two_degree_roll_with_the_bottom_face_on_a_mark_edge(self):
        for sign in (1, -1):
            q = about("y", 2.0 * sign)  # a roll about the unit's long axis
            r = rotation(q)
            corners_local = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * (
                np.array(L.UNIT_SIZE) / 2)
            w = corners_local @ r.T
            low_x = w[w[:, 2] == w[:, 2].min()][0, 0]  # x of the low edge relative to the centre
            side = -1 if low_x < 0 else 1
            # the low edge sits exactly on the mark's outer side (x), at the mark's top, over the mark (y = its centre)
            x = L.FRAME_X + side * L.MARK_SIZE[0] / 2 - low_x
            u = unit_at(x, L.MARK_YS[1], low=L.MARK_TOP, quat=q)
            self.assertTrue(touches_sill(u))
            self.assertFalse(touches_sill(unit_at(x, L.MARK_YS[1], low=L.MARK_TOP + 0.0007, quat=q)))

    def test_tilted_unit_whose_low_corner_is_near_the_sill(self):
        # tipped 3 degrees about x so its far end is down, clear of the marks: only that end is near the sill
        q = about("x", 3.0)
        self.assertTrue(touches_sill(unit_at(low=L.SILL_TOP, quat=q)))  # the low corner reaches the sill
        self.assertTrue(touches_sill(unit_at(low=L.SILL_TOP + 0.0004, quat=q)))  # 0.4 mm above the bare sill
        self.assertFalse(touches_sill(unit_at(low=L.SILL_TOP + 0.0006, quat=q)))  # 0.6 mm: not
        self.assertFalse(touches_sill(unit_at(low=L.SILL_TOP + 0.0010, quat=q)))  # 1.0 mm over the bare sill: near miss
        # the marks are not what it touched: at the mark nearer the low end it is over 1 mm above the mark's top
        self.assertGreater(float(sill_gaps(unit_at(low=L.SILL_TOP + 0.0010, quat=q)).min()), 0.0005)

    def test_tipped_onto_one_block_is_not_a_touch(self):
        q = about("x", 4.5)
        self.assertFalse(touches_sill(unit_at(low=L.SILL_TOP + 0.003, quat=q)))
        self.assertTrue(touches_sill(unit_at(low=L.SILL_TOP, quat=q)))

    def test_edges_are_sampled_every_5mm_or_finer(self):
        from scoring import _EDGE_LOCAL
        self.assertEqual(L.TOUCH_STEP, 0.005)
        for a, size in enumerate(L.UNIT_SIZE):  # along each axis consecutive samples on an edge are <= 5 mm apart
            along = np.unique(np.round(_EDGE_LOCAL[:, a], 9))
            self.assertLessEqual(float(np.diff(along).max()), 0.005 + 1e-9, a)
        self.assertGreaterEqual(len(_EDGE_LOCAL), 200)

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
        self.assertTrue(block_on_sill(block_at(CX, L.MARK_YS[0], L.MARK_TOP + 0.0009)))  # 0.9 mm over its mark
        self.assertFalse(block_on_sill(block_at(CX, L.MARK_YS[0], L.MARK_TOP + 0.0011)))  # 1.1 mm: above it
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
        self.assertTrue(on_block(unit_at(low=L.SET_BLOCK_TOP + 0.0009), B1_SET))
        self.assertFalse(on_block(unit_at(low=L.SET_BLOCK_TOP + 0.0011), B1_SET))  # 1.1 mm above it: hovering
        self.assertTrue(on_block(unit_at(low=L.SET_BLOCK_TOP - 0.0009), B1_SET))  # solver penetration
        self.assertFalse(on_block(unit_at(low=L.SET_BLOCK_TOP - 0.0011), B1_SET))
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
        # both on the bare sill 90 mm from the centre (off the marks), the unit standing on them
        ends = block_at(CX, -0.09, L.SILL_TOP), block_at(CX, 0.09, L.SILL_TOP)
        r, _ = run(*procedure(b1_end=ends[0], b2_end=ends[1], unit_end=unit_at(low=L.SILL_TOP + L.BLOCK_SIZE[2])))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["on_block"], [True, True])

    def test_standing_on_one_block_scores_half(self):
        # the second block left on the tray; the unit let go level on the first (the scorer's geometry; in physics
        # a unit let go on one quarter-point block tips onto the sill: Exploits below)
        us, b1, b2 = procedure()
        r, _ = run(us, b1, hold(B2_HOME, len(us)))
        self.assertEqual(r.details["on_block"], [True, False])
        self.assertEqual(r.score, 0.5)

    def test_on_one_block_and_tilted_above_the_other_scores_half(self):
        # both blocks set, the unit let go 2.5 degrees off level: down on the right-hand block, 4.4 mm over the other
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
        r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(holding_unit(us[-1]), 70))
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
        # the left arm alone: no blocks, the unit lowered onto the sill
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

    def test_resting_needs_stillness_over_the_last_second(self):
        # judged from the poses (a path window), not from the reported velocity
        us, b1, b2 = procedure(settle=0)
        creep = [unit_at(y=-0.0021 + 0.0021 * i / 19) for i in range(20)]  # 2.1 mm along the channel in 1 s
        r, _ = run(us + creep, b1 + hold(B1_SET, 20), b2 + hold(B2_SET, 20))
        self.assertEqual(r.details["still"][0], False)
        self.assertEqual(r.score, 0.25)  # the blocks are set; the unit is not resting
        creep = [unit_at(y=-0.0019 + 0.0019 * i / 19) for i in range(20)]  # 1.9 mm: still enough
        r, _ = run(us + creep, b1 + hold(B1_SET, 20), b2 + hold(B2_SET, 20))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        turning = [block_at(CX, L.MARK_YS[1], quat=about("z", 2.1 * i / 19)) for i in range(20)]  # 2.1 degrees
        r, _ = run(us + hold(U_ON, 20), b1 + hold(B1_SET, 20), b2 + turning)
        self.assertEqual(r.details["still"], [True, True, False])
        self.assertEqual(r.score, 0.5)  # resting on the other block only; the blocks are not both set
        self.assertFalse(r.success)

    def test_a_reported_velocity_alone_does_not_stop_a_rest(self):
        # simulator jitter in the velocity with the pose unchanged: still (the GPU demo's velocities flickered)
        us, b1, b2 = procedure(settle=0)
        jitter = state(U_ON.pos, U_ON.size, vel=(0, 0, -0.012))
        r, _ = run(us + hold(jitter, 25), b1 + hold(B1_SET, 25), b2 + hold(B2_SET, 25))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

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
        rocked = unit_at(low=L.SILL_TOP, quat=about("x", 4.5))
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



class LetGo(unittest.TestCase):
    """The review's point: a unit held, or hovering, within reach of the blocks earns none of the resting tiers.
    Resting needs the unit and the block let go (both finger zones 30 mm or more from each on every step of the last
    1 s), still over it, and the support within 1 mm. Unreadable arms are never let go."""

    def held_at_end(self, unit_end, n=70, who=holding_unit):
        us, b1, b2 = procedure(unit_end=unit_end)
        return run(us, b1, b2, arms=hold(AWAY, len(us) - n) + hold(who(us[-1]), n))

    def test_held_on_the_blocks_earns_only_the_blocks(self):
        # the unit exactly on both blocks, perfectly still, but the left gripper still closed on its top edge
        r, _ = self.held_at_end(U_ON)
        self.assertEqual(r.details["on_block"], [False, False])
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_held_hovering_just_above_the_blocks_earns_only_the_blocks(self):
        for low in (0.0005, 0.0015, 0.002):
            r, _ = self.held_at_end(unit_at(low=L.SET_BLOCK_TOP + low))
            self.assertEqual(r.score, 0.25, low)
            self.assertFalse(r.success, low)

    def test_held_off_centre_or_on_one_block_earns_nothing_for_the_unit(self):
        r, _ = self.held_at_end(unit_at(y=0.012))  # would be 0.75 let go
        self.assertEqual(r.score, 0.25)
        us, b1, b2 = procedure()
        r, _ = run(us, b1, hold(B2_HOME, len(us)), arms=hold(AWAY, len(us) - 70) + hold(holding_unit(us[-1]), 70))
        self.assertEqual(r.score, 0.0)  # would be 0.5 let go

    def test_a_gap_over_1mm_is_not_resting(self):
        # let go, but 1.1 mm above both blocks (as a unit wedged or held some other way would be): not resting
        r, _ = run(*procedure(unit_end=unit_at(low=L.SET_BLOCK_TOP + 0.0011)))
        self.assertEqual(r.score, 0.25)
        r, _ = run(*procedure(unit_end=unit_at(low=L.SET_BLOCK_TOP + 0.0009)))
        self.assertEqual(r.score, 1.0)

    def test_let_go_must_last_the_whole_last_second(self):
        us, b1, b2 = procedure()
        hand = holding_unit(us[-1])
        for free, want in ((19, 0.25), (20, 1.0)):
            r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 40) + hold(hand, 40 - free) + hold(AWAY, free))
            self.assertEqual(r.score, want, free)

    def test_the_30mm_finger_zone_clearance(self):
        # an open gripper straight over the unit: its zone reaches zone_radius(1.0) = 56.6 mm round the pad tips
        top = U_ON.high[2]
        for gap, want in ((0.029, False), (0.031, True)):
            tips_z = top + rigstate.zone_radius(1.0) + gap
            hands = {"left": arm((CX, CY, tips_z + 0.013), 1.0), "right": AWAY["right"]}
            self.assertAlmostEqual(hand_gap(hands, U_ON, L.UNIT_SIZE), gap, places=6)
            self.assertEqual(let_go_now(hands, U_ON, L.UNIT_SIZE), want)
            us, b1, b2 = procedure()
            r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(hands, 70))
            self.assertEqual(r.score, 1.0 if want else 0.25, gap)

    def test_an_open_hand_round_the_glass_has_not_let_go(self):
        # fingers wide open but still either side of the top edge: not let go, whatever the opening
        us, b1, b2 = procedure()
        top = U_ON.high[2]
        for opening in (0.5, 1.0):
            hands = {"left": arm((CX, CY, top - 0.016), opening), "right": AWAY["right"]}
            self.assertFalse(let_go_now(hands, U_ON, L.UNIT_SIZE))
            r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(hands, 70))
            self.assertEqual(r.details["on_block"], [False, False], opening)
            self.assertLess(r.score, 0.5, opening)  # 0.25, or 0 where the wide-open zone also reaches a block

    def test_unreadable_arms_are_never_let_go(self):
        nan = float("nan")
        good = AWAY
        broken = [
            None,  # no snapshot
            {"left": good["left"]},  # a side missing
            {"left": arm((nan, 0.3, 1.0)), "right": good["right"]},  # a non-finite TCP
            {"left": good["left"], "right": rigstate.Arm(tcp=good["right"].tcp, tips=np.array([0.3, np.inf, 1.0]),
                                                          opening=1.0)},  # a non-finite pad-tip point
            {"left": good["left"], "right": arm((0.32, -0.305, 1.0), nan)},  # a non-finite opening
            {"left": SimpleNamespace(tcp=(0.3, 0.3)), "right": good["right"]},  # a malformed arm
            "garbage",
        ]
        us, b1, b2 = procedure()
        for k, bad in enumerate(broken):
            self.assertFalse(let_go_now(bad, U_ON, L.UNIT_SIZE), k)
            self.assertIsNone(hand_gap(bad, U_ON, L.UNIT_SIZE), k)
            r, t = run(us, b1, b2, arms=bad)  # unreadable for the whole trial: nothing is ever let go
            self.assertEqual(r.score, 0.0, k)  # not even the blocks are set
            self.assertFalse(r.success, k)
            self.assertEqual(t.arms_unreadable, len(us), k)
            r, t = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(bad, 70))  # unreadable at the end only
            self.assertEqual(r.score, 0.0, k)

    def test_one_unreadable_step_restarts_the_count(self):
        us, b1, b2 = procedure(settle=90)
        n = len(us)
        for gap_at, want in ((n - 10, 0.0), (n - 20, 0.0), (n - 21, 1.0)):
            arms = hold(AWAY, n)
            arms[gap_at] = None
            r, t = run(us, b1, b2, arms=arms)
            self.assertEqual(r.score, want, gap_at)
            self.assertEqual(t.arms_unreadable, 1)

    def test_a_distance_that_cannot_be_computed_is_not_let_go(self):
        bad = SimpleNamespace(pos=np.array([0.3, 0.0]), quat_xyzw=np.array([0, 0, 0, 1.0]))  # a 2-vector centre
        self.assertFalse(let_go_now(AWAY, bad, L.UNIT_SIZE))
        self.assertEqual(hand_gap(AWAY, bad, L.UNIT_SIZE), -np.inf)

    def test_a_held_block_is_not_set(self):
        # no unit brought over; the right gripper still closed on block 1 on its mark at the end: not set (0, not
        # the 0.25 both set and let go would earn)
        us, b1, b2 = procedure()
        n = len(us)
        r, _ = run(hold(U_HOME, n), b1, b2)
        self.assertEqual(r.score, 0.25)
        r, _ = run(hold(U_HOME, n), b1, b2, arms=hold(AWAY, n - 70) + hold(holding_block(B1_SET), 70))
        self.assertFalse(r.details["blocks_set"])
        self.assertEqual(r.score, 0.0)
        # held 0.5 mm over its mark (inside the band a block let go there could sit in): not set either
        held_up = block_at(CX, L.MARK_YS[0], L.MARK_TOP + 0.0005)
        self.assertTrue(block_on_sill(held_up))
        r, _ = run(hold(U_HOME, n), b1[:-70] + hold(held_up, 70), b2,
                   arms=hold(AWAY, n - 70) + hold(holding_block(held_up), 70))
        self.assertEqual(r.score, 0.0)

    def test_gripped_again_or_lifted_after_resting_loses_the_resting_tiers(self):
        # let go and resting (1.0, success), then gripped again: in place, or lifted 5 mm off the blocks and held
        us, b1, b2 = procedure()
        r, t = run(us, b1, b2)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        lifted = unit_at(low=L.SET_BLOCK_TOP + 0.005)
        for end in (U_ON, lifted):
            tail = glide(U_ON, end, 10) + hold(end, 30)
            r2, _ = run(tail, hold(b1[-1], len(tail)), hold(b2[-1], len(tail)), tracker=t,
                        arms=[holding_unit(u) for u in tail])
            self.assertEqual(r2.details["on_block"], [False, False])
            self.assertEqual(r2.score, 0.25)
            self.assertFalse(r2.success)
            t = run(us, b1, b2)[1]
        # lifted, then put back down and let go: no credit until it has been let go for the whole last 1 s
        back = glide(lifted, U_ON, 10)
        tail = glide(U_ON, lifted, 10) + back + hold(U_ON, 19)
        hands = [holding_unit(u) for u in tail[:20]] + hold(AWAY, 19)
        r3, t = run(tail, hold(b1[-1], len(tail)), hold(b2[-1], len(tail)), tracker=t, arms=hands)
        self.assertEqual(r3.score, 0.25)
        r3, _ = run([U_ON], [b1[-1]], [b2[-1]], tracker=t, arms=[AWAY])
        self.assertEqual(r3.score, 1.0)

    def test_the_success_signal_needs_everything_let_go(self):
        r, _ = self.held_at_end(U_ON, n=200)
        self.assertFalse(r.success)
        r, _ = run(*procedure())
        self.assertTrue(r.success)


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

    def test_a_touch_is_latched_when_only_a_block_is_unreadable(self):
        t = Tracker()
        t.update(U_HOME, B1_HOME, B2_HOME)
        bad_block = state(B2_SET.pos, B2_SET.size, vel=(0, float("nan"), 0))
        r = t.update(unit_at(low=L.SILL_TOP), B1_SET, bad_block)  # the unit's own state is fine: it sits on the sill
        self.assertTrue(t.touched)
        self.assertEqual(t.bad_steps, 1)
        self.assertEqual(r.score, 0.0)
        r, _ = run([U_ON] * 3, [B1_SET] * 3, [B2_SET] * 3, tracker=t)
        self.assertTrue(r.details["touched"])
        self.assertEqual(r.score, 0.0)

    def test_blow_ups_score_zero_and_leave_the_tracker_alone(self):
        for k, (u, a, b) in enumerate(self.odd()[:6]):
            us, b1, b2 = procedure()
            i = len(us) - 80
            t = Tracker()
            for j, (uu, aa, bb) in enumerate(zip(us, b1, b2)):
                r = t.update(*((u, a, b) if j == i else (uu, aa, bb)), AWAY)
                if j == i:
                    self.assertEqual(r.score, 0.0, k)
                    self.assertFalse(r.details["valid_state"])
            self.assertEqual(r.score, 1.0, k)
            self.assertTrue(r.success, k)
            self.assertEqual(t.bad_steps, 1)
            self.assertFalse(t.touched)

    def test_a_blow_up_restarts_the_let_go_and_stillness_counts(self):
        us, b1, b2 = procedure(settle=90)
        bad = (unit_at(x=float("nan")), B1_SET, B2_SET)
        for after, want in ((19, 0.0), (20, 1.0)):
            i = len(us) - after - 1
            t, r = Tracker(), None
            for j, (uu, aa, bb) in enumerate(zip(us, b1, b2)):
                r = t.update(*(bad if j == i else (uu, aa, bb)), AWAY)
            self.assertEqual(r.score, want, after)

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


OFFLINE = HERE.parent.parent / "tools" / "offline"
STUB = OFFLINE / "yam_stub"


def load_task():
    """task.py, imported through the offline yam_arena stub."""
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

    def commanded(self, arm, sag=0.0):
        """Each control step's commanded gripper (rigstate.Arm) for one arm, through the stub ArmPlan's own
        interpolation; ``sag`` lowers every TCP (a tracking error or payload sag the scorer must survive)."""
        if str(OFFLINE) not in sys.path:
            sys.path.insert(0, str(OFFLINE))
        import yamkin
        from yam_arena.demo import ArmPlan

        plan = ArmPlan(list(self.plan[arm]), yamkin.BASES["station"][arm])
        out = []
        for k in range(round(self.stop() * HZ) + 1):
            pos, rot, finger, _ = plan.target(k / HZ)
            pos = np.asarray(pos, float) - np.array([0.0, 0.0, sag])
            tips = pos + self.task.PAD * np.asarray(rot)[:, 2]
            out.append(rigstate.Arm(tcp=pos, tips=tips, opening=float(np.clip(finger / self.task.OPEN, 0, 1))))
        return out

    def stop(self) -> float:
        """When the scripted policy stops: the longer plan, then its hold."""
        return max(sum(w.duration for w in self.plan[a]) for a in ("left", "right")) + self.HOLD_S

    def free_from(self, sag=0.0):
        """The first control step from which, to the policy's stop, the unit on the blocks (U_ON) and both blocks
        on their marks are let go by both grippers (commanded poses)."""
        left, right = self.commanded("left", sag), self.commanded("right", sag)
        first = None
        for k, (a, b) in enumerate(zip(left, right)):
            hands = {"left": a, "right": b}
            ok = all(let_go_now(hands, s, size) for s, size in
                     ((U_ON, L.UNIT_SIZE), (B1_SET, L.BLOCK_SIZE), (B2_SET, L.BLOCK_SIZE)))
            first = (first if first is not None else k) if ok else None
        return first, {"left": left[-1], "right": right[-1]}

    def test_the_demo_lets_go_and_clears_the_glass_before_the_policy_stops(self):
        # from the step everything is let go: 1 s of let-go window, then the 1 s success hold, with a second to spare
        stop = round(self.stop() * HZ)
        for sag in (0.0, 0.0056, 0.015):
            first, end = self.free_from(sag)
            self.assertIsNotNone(first, sag)
            success_at = first + L.WINDOW_STEPS - 1 + round(self.task.SUCCESS_HOLD_S * HZ)
            self.assertLess(success_at, stop - HZ, sag)
            # and at the end the nearer finger zone is far past the 30 mm rule
            self.assertGreater(hand_gap(end, U_ON, L.UNIT_SIZE), 2 * L.LET_GO_CLEAR, sag)
        # the release ends before the retreat starts: everything let go comes after the fingers are fully open
        seg = {w.label: (t0, t1) for t0, t1, _, w in self.segments("left")}
        self.assertGreater(self.free_from()[0] / HZ, seg["let go"][1])

    def test_straight_up_alone_would_not_count_as_let_go(self):
        # why the demo goes back over the unit's starting spot: 18 mm over the glass, the open zone still overlaps it
        seg = {w.label: (t0, t1, w) for t0, t1, _, w in self.segments("left")}
        w = seg["straight up"][2]
        hands = {"left": arm(w.pos, -w.finger / -self.task.OPEN), "right": AWAY["right"]}
        self.assertFalse(let_go_now(hands, U_ON, L.UNIT_SIZE))
        self.assertEqual(seg["back over its starting spot, clear of the glass"][2].pos, self.task.RETREAT_L)

    def test_the_run_limit_covers_the_demo(self):
        self.assertLessEqual(round(self.stop() * HZ), 1200)  # task.json: run limit 1,200 steps


@unittest.skipUnless(STUB.is_dir(), "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py itself, imported through the offline yam_arena stub (not part of a submission)."""

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

    def test_task_json_agrees_with_the_code(self):
        task = self.load()
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], task.INSTRUCTION)
        self.assertEqual(meta["category"], "Impact on jobs")
        sc = meta["scoring"]
        self.assertEqual((L.LET_GO_CLEAR, L.WINDOW_S, L.WINDOW_STEPS), (0.030, 1.0, 20))
        self.assertEqual((L.STILL_MOVE, L.STILL_TURN_DEG, L.ON_BLOCK_GAP), (0.002, 2.0, 0.001))
        self.assertEqual(task.SUCCESS_HOLD_S, 1.0)
        for phrase in ("30 mm", "last 20 control steps (1 s)", "within 2 mm and 2 degrees", "within 1 mm of the block",
                       "unreadable", "for 1 s", "in contact with the sill's top or a mark's top",
                       "0.5 mm or less above the surface under it", "chalk"):
            self.assertIn(phrase, sc, phrase)
        for stale in ("within 2 mm of the block", "for 3 s", "3 mm above the sill", "cannot see the grippers",
                      "less than 2 mm above the sill", "within 2 mm of the sill"):
            self.assertNotIn(stale, sc + meta["fidelity"], stale)
        readme = (HERE / "README.md").read_text()
        self.assertIn(task.INSTRUCTION, readme)
        self.assertNotIn("cannot see the grippers", readme)
        for stale in ("within 2 mm of the sill", "less than 2 mm above the sill"):
            self.assertNotIn(stale, readme, stale)
        self.assertIn("0.5 mm or less above the surface under it", readme)

    def test_the_instruction_states_the_goal_only(self):
        task = self.load()
        meta = json.loads((HERE / "task.json").read_text())
        for text in (task.INSTRUCTION, meta["title"]):
            low = text.lower()
            for word in (" then ", "first", "before", "after", "straight down", "lower "):
                self.assertNotIn(word, low, (word, text))
        self.assertLess(len(task.INSTRUCTION), 150)
        self.assertIn("let go", task.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
