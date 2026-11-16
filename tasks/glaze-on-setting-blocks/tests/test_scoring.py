"""Offline tests for scoring.py, the frame and the demo: synthetic object states, no simulator.
python3 -m unittest discover -s tests -v"""

import contextlib
import importlib.util
import json
import math
import os
import pathlib
import re
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
    Tracker, block_on_sill, block_set_on, block_yaw_err_deg, box_separation, corners, gap_to_block, hand_gap,
    in_rebate, let_go_now, on_block, rotation, share_under, sill_gap, standing, stop_gap, touches_sill,
    unit_tilt_deg,
)

UH = L.UNIT_SIZE[2]
CX, CY = L.UNIT_SET_X, L.FRAME_Y  # the unit's centre standing against the stop (1 mm play), centred
HZ = L.CONTROL_HZ
SET_PIVOT = 0.2865  # the demo's: where the unit's room-side bottom edge is set on the blocks (task.SET_PIVOT_X)
LANE_X = L.FRAME_ROOM_X - 0.035  # in front of the frame


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


def leaning(deg, pivot_x=SET_PIVOT, pivot_z=L.SET_BLOCK_TOP, y=CY):
    """The unit tilted ``deg`` degrees with its top toward the room, its room-side bottom edge at (pivot_x, pivot_z)."""
    q = about("y", -deg)
    c = np.array([pivot_x, y, pivot_z]) - rotation(q) @ np.array([-L.UNIT_SIZE[0] / 2, 0.0, -UH / 2])
    return state(c, L.UNIT_SIZE, q)


def arm(tcp, opening=1.0):
    """A gripper pointing straight down (pad-tip point 13 mm below the TCP), as rigstate reports it."""
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=float(opening))


AWAY = {"left": arm((0.32, 0.305, 1.0)), "right": arm((0.32, -0.305, 1.0))}  # both up at the ready pose, open
GRIP_OPENING = (L.UNIT_SIZE[0] - 0.008) / rigstate.FULL_GAP  # the demo's 16 mm grip on the unit


def holding_unit(u):
    """The left gripper closed on the unit's top edge (TCP 10 mm below the top), the right away."""
    return {"left": arm((u.pos[0], u.pos[1] + 0.075, u.high[2] - 0.010), GRIP_OPENING), "right": AWAY["right"]}


def holding_block(b):
    """The right gripper closed across a block (pad tips 2.5 mm above its bottom), the left away."""
    return {"left": AWAY["left"], "right": arm((b.pos[0], b.pos[1], b.low[2] + 0.0155), 0.022 / rigstate.FULL_GAP)}


U_HOME = unit_at(L.UNIT_X, L.UNIT_Y, L.TOP)
B1_HOME = block_at(*L.BLOCK_STARTS[0], L.TRAY_TOP)
B2_HOME = block_at(*L.BLOCK_STARTS[1], L.TRAY_TOP)
B1_SET = block_at(L.MARK_X, L.MARK_YS[0])  # on the right mark
B2_SET = block_at(L.MARK_X, L.MARK_YS[1])  # on the left mark
U_ON = unit_at()  # standing on both blocks, centred, 1 mm from the outside stop
U_MARKS = unit_at(low=L.MARK_TOP)  # standing on the two bare marks (no blocks)


def glide(a, b, n):
    """n states moving a's centre to b's in equal steps, with the matching velocity; a's orientation."""
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
    """Lift a block off the tray, bring it in front of the frame level with its mark, in under the head 10 mm over
    the mark, and down to ``end``."""
    carry = L.TOP + 0.040
    up = block_at(home.pos[0], home.pos[1], carry)
    lane = block_at(LANE_X, home.pos[1], carry)
    front = block_at(LANE_X, end.pos[1], carry)
    entry = block_at(LANE_X, end.pos[1], end.low[2] + 0.010)
    over = block_at(end.pos[0], end.pos[1], end.low[2] + 0.010)
    final = block_at(end.pos[0], end.pos[1], end.low[2], end.quat_xyzw)
    return path(home, up, lane, front, entry, over, final) + [final]


def turn(fn, a, b, n=20):
    """n states fn(angle) for angles from a to b (degrees)."""
    return [fn(a + (b - a) * i / n) for i in range(1, n + 1)]


def move_unit(end=U_ON, set_x=SET_PIVOT, upright_x=None, start=U_HOME):
    """The room-side insertion: lift the unit, bring it upright in front of the frame, tilt its top 48 degrees toward
    the room, bring its bottom edge in over the blocks and set it on them, turn it upright about that edge (under the
    head), slide it out to ``end``."""
    top = L.SET_BLOCK_TOP
    lift = unit_at(start.pos[0], start.pos[1], L.TOP + 0.030)
    front_x = set_x - 0.075
    front = unit_at(front_x + L.UNIT_SIZE[0] / 2, CY, top + 0.018)
    out = path(start, lift, front)
    out += turn(lambda d: leaning(d, front_x, top + 0.018), 0, 48)
    out += path(leaning(48, front_x, top + 0.018), leaning(48, set_x - 0.035, top + 0.008),
                leaning(48, set_x, top + 0.008), leaning(48, set_x, top))
    out += turn(lambda d: leaning(d, set_x, top), 48, 0, n=40)
    upright = leaning(0, set_x, top) if upright_x is None else unit_at(upright_x)
    final = state(end.pos, end.size, end.quat_xyzw)
    return out + glide(upright, final, 20) + [final]


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


def procedure(b1_end=B1_SET, b2_end=B2_SET, unit_end=U_ON, settle=70, **kw):
    """The intended sequence: block 1 to its mark, block 2 to its mark, then the unit in from the room side onto
    them and upright against the stop, then everything still for ``settle`` steps."""
    m1, m2 = move_block(B1_HOME, b1_end), move_block(B2_HOME, b2_end)
    us = hold(U_HOME, 5 + len(m1) + len(m2))
    b1 = hold(B1_HOME, 5) + m1 + hold(m1[-1], len(m2))
    b2 = hold(B2_HOME, 5 + len(m1)) + m2
    mu = move_unit(unit_end, **kw)
    us += mu + hold(mu[-1], settle)
    b1 += hold(b1[-1], len(mu) + settle)
    b2 += hold(b2[-1], len(mu) + settle)
    return us, b1, b2


def with_tail(seq, tail, n):
    """Replace the last n states of a sequence by ``tail`` (a state)."""
    return seq[:-n] + hold(tail, n)


def frame_separation(s, size):
    """The smallest separation (m) between the box of state ``s`` and the frame's boxes."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, float)
    return min(box_separation(c, r, np.array(size) / 2, centre, np.eye(3), np.array(sz) / 2)
               for centre, sz in L.FRAME_BOXES.values())


class Frame(unittest.TestCase):
    """The revised setup: a complete frame (sill, jambs, head) with an outside stop and an open room side."""

    def box(self, name):
        (cx, cy, cz), (sx, sy, sz) = L.FRAME_BOXES[name]
        return np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2]), np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])

    def test_the_opening_and_its_clearances(self):
        lo_l, _ = self.box("jamb_left")
        _, hi_r = self.box("jamb_right")
        self.assertAlmostEqual(lo_l[1] - hi_r[1], 0.240)  # 240 mm between the jambs
        self.assertAlmostEqual(lo_l[1] - U_ON.high[1], 0.020)  # 20 mm at each end
        self.assertAlmostEqual(self.box("head")[0][2] - U_ON.high[2], 0.006)  # 6 mm at the head
        self.assertAlmostEqual(stop_gap(U_ON), 0.001)  # against the outside stop, 1 mm play
        for name in ("sill", "head", "jamb_left", "jamb_right"):  # real section: 60 mm deep, 20 mm thick
            lo, hi = self.box(name)
            self.assertAlmostEqual(hi[0] - lo[0], 0.060, msg=name)
            self.assertAlmostEqual(min(hi[1] - lo[1], hi[2] - lo[2]), 0.020, msg=name)

    def test_the_outside_stop_runs_round_the_opening_15mm_into_it(self):
        sill, head = self.box("stop_sill"), self.box("stop_head")
        self.assertAlmostEqual(sill[1][2] - L.SILL_TOP, 0.015)
        self.assertAlmostEqual(L.SOFFIT - head[0][2], 0.015)
        for name in ("stop_jamb_left", "stop_jamb_right"):
            lo, hi = self.box(name)
            self.assertAlmostEqual(hi[1] - lo[1], 0.015, msg=name)
        for name in ("stop_sill", "stop_head", "stop_jamb_left", "stop_jamb_right"):  # all on the outside, at STOP_X
            self.assertAlmostEqual(self.box(name)[0][0], L.STOP_X, msg=name)
        # the unit standing against it is held by the sill stop (6 mm) and the head stop (9 mm) on its outside face
        self.assertAlmostEqual(sill[1][2] - U_ON.low[2], 0.006)
        self.assertAlmostEqual(U_ON.high[2] - head[0][2], 0.009)

    def test_the_room_side_is_open_and_the_bead_lies_on_the_table(self):
        # nothing of the frame is room-side of its room face; the 12 mm bead space in front of the unit is empty
        for name in L.FRAME_BOXES:
            self.assertGreaterEqual(self.box(name)[0][0], L.FRAME_ROOM_X - 1e-12, name)
        self.assertAlmostEqual(U_ON.low[0] - L.FRAME_ROOM_X, 0.012)
        self.assertEqual(L.BEAD_SIZE[1], 0.012)
        self.assertAlmostEqual(L.BEAD_POS[2] - L.BEAD_SIZE[2] / 2, L.TOP)  # lying on the table
        bead = state(L.BEAD_POS, L.BEAD_SIZE)
        self.assertGreater(frame_separation(bead, L.BEAD_SIZE), 0.02)  # beside the frame, not in it

    def test_the_unit_cannot_be_lowered_in_from_above(self):
        # straight down from above the head onto the blocks, the unit passes through the head
        for low in np.arange(L.HEAD_TOP + 0.002, L.SET_BLOCK_TOP - 1e-9, -0.002):
            if frame_separation(unit_at(low=low), L.UNIT_SIZE) < 0:
                break
        else:
            self.fail("lowered from above without meeting the head")
        self.assertLess(frame_separation(unit_at(low=L.SET_BLOCK_TOP + 0.010), L.UNIT_SIZE), 0)

    def test_the_room_side_path_clears_the_frame_and_the_sill(self):
        us, b1, b2 = procedure()
        for i, u in enumerate(us):
            self.assertGreaterEqual(frame_separation(u, L.UNIT_SIZE), -1e-9, i)
            self.assertFalse(touches_sill(u), i)
        r, t = run(us, b1, b2)
        self.assertEqual(r.score, 1.0)
        self.assertGreater(t.min_sill_gap, 0.005)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_plain_box(self):
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("y", 2.5))), 2.5, places=4)
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("x", 180))), 0.0, places=4)  # on its other long edge
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("x", 177))), 3.0, places=4)
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("z", 40))), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(unit_tilt_deg(unit_at(quat=about("y", 90))), 90.0, places=4)  # lying on a face
        self.assertAlmostEqual(unit_tilt_deg(leaning(48)), 48.0, places=4)

    def test_block_yaw_is_the_same_after_a_quarter_turn(self):
        # the blocks are 30 x 30 mm: turned 90 degrees one is indistinguishable, so it must score the same
        self.assertEqual(L.BLOCK_SIZE[0], L.BLOCK_SIZE[1])
        for d in (0, 5, 10, 14, 20, 30, 44):
            base = block_yaw_err_deg(about("z", d))
            for k in (1, 2, 3, -1):
                self.assertAlmostEqual(block_yaw_err_deg(about("z", d + 90 * k)), base, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 90)), 0.0, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 100)), 10.0, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 45)), 45.0, places=4)

    def test_block_set_equally_when_turned_a_quarter(self):
        for d, want in ((0, 0), (10, 0), (14, 0), (20, None), (40, None)):
            for k in (0, 1, -1, 2):
                blk = block_at(L.MARK_X, L.MARK_YS[0], quat=about("z", d + 90 * k))
                self.assertEqual(block_set_on(blk, 0.008), want, (d, k))

    def test_block_yaw_has_two_ways_round(self):
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 10)), 10.0, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("z", 175)), 5.0, places=4)
        self.assertAlmostEqual(block_yaw_err_deg(about("x", 180)), 0.0, places=4)  # flipped onto its other face

    def test_box_separation(self):
        eye = np.eye(3)
        self.assertAlmostEqual(box_separation((0, 0, 0), eye, (1, 1, 1), (3.5, 0, 0), eye, (1, 1, 1)), 1.5)
        self.assertAlmostEqual(box_separation((0, 0, 0), eye, (1, 1, 1), (2, 0, 0), eye, (1, 1, 1)), 0.0)
        self.assertAlmostEqual(box_separation((0, 0, 0), eye, (1, 1, 1), (1.7, 0, 0), eye, (1, 1, 1)), -0.3)
        # a box tipped about y over a slab: the separation is its lowest corner's height
        u = leaning(30, pivot_z=L.SILL_TOP + 0.004)
        r, c = rotation(u.quat_xyzw), u.pos
        sill_c, sill_s = L.SILL_BOX
        sep = box_separation(c, r, np.array(L.UNIT_SIZE) / 2, sill_c, eye, np.array(sill_s) / 2)
        self.assertAlmostEqual(sep, 0.004, places=9)

    def test_contact_is_a_touch_and_an_air_gap_is_not(self):
        # the rule is contact (no air gap), the event a chalk print shows: 0.3 mm of air is NOT a touch
        self.assertEqual(L.SILL_TOUCH_GAP, 0.0)
        self.assertTrue(touches_sill(U_MARKS))  # standing on the two marks
        self.assertTrue(touches_sill(unit_at(low=L.MARK_TOP - 0.0002)))  # pressed 0.2 mm into them
        self.assertFalse(touches_sill(unit_at(low=L.MARK_TOP + 0.0003)))  # a 0.3 mm air gap
        self.assertFalse(touches_sill(unit_at(low=L.MARK_TOP + 0.00001)))  # 0.01 mm of air: still no contact
        self.assertAlmostEqual(sill_gap(unit_at(low=L.MARK_TOP + 0.0003)), 0.0003, places=9)
        # over the bare sill, between the marks: a corner tipped down to the sill's top
        q = about("x", 3.0)  # its far end down, clear of the marks (which are at y +-0.05)
        self.assertTrue(touches_sill(unit_at(low=L.SILL_TOP, quat=q)))
        self.assertFalse(touches_sill(unit_at(low=L.SILL_TOP + 0.0003, quat=q)))
        # any face of the sill counts: the bottom edge on the sill's room-side edge, or 0.3 mm in front of it
        self.assertTrue(touches_sill(leaning(48, pivot_x=L.FRAME_ROOM_X, pivot_z=L.SILL_TOP)))
        self.assertFalse(touches_sill(leaning(48, pivot_x=L.FRAME_ROOM_X - 0.0003, pivot_z=L.SILL_TOP)))
        self.assertTrue(touches_sill(leaning(48, pivot_x=L.FRAME_ROOM_X, pivot_z=L.SILL_TOP - 0.005)))  # its face

    def test_on_the_blocks_tilted_or_upright_is_not_a_touch(self):
        self.assertFalse(touches_sill(U_ON))  # 8 mm over the marks
        self.assertFalse(touches_sill(leaning(48)))  # leaning on its bottom edge on the blocks
        self.assertGreater(sill_gap(leaning(48)), 0.0079)
        self.assertFalse(touches_sill(U_HOME))  # on the table outside the frame
        self.assertFalse(touches_sill(unit_at(LANE_X, CY, L.TOP)))  # on the table in front of the frame

    def test_standing_in_the_rebate(self):
        self.assertTrue(in_rebate(U_ON) and standing(U_ON))
        self.assertTrue(in_rebate(unit_at(CX - 0.0019)))  # 2.9 mm from the stop
        self.assertFalse(in_rebate(unit_at(CX - 0.0021)))  # 3.1 mm: not against the stop
        self.assertTrue(in_rebate(unit_at(CX + 0.0019)))  # 0.9 mm into the stop: slack
        self.assertFalse(in_rebate(unit_at(CX + 0.0021)))
        self.assertTrue(in_rebate(unit_at(y=0.020)))  # its end at the jamb's face
        self.assertFalse(in_rebate(unit_at(y=0.0215)))
        self.assertFalse(in_rebate(unit_at(LANE_X, CY, L.TOP)))  # in front of the frame
        self.assertTrue(standing(unit_at(quat=about("y", 2.0))))
        self.assertFalse(standing(unit_at(CX - 0.002, quat=about("y", -3.1))))  # leaning 3.1 degrees
        self.assertFalse(standing(leaning(48)))

    def test_block_on_the_sill(self):
        self.assertTrue(block_on_sill(B1_SET))
        self.assertTrue(block_on_sill(block_at(L.MARK_X, 0.0, L.SILL_TOP)))  # on the bare sill between the marks
        self.assertTrue(block_on_sill(block_at(L.MARK_X, L.MARK_YS[0], quat=about("x", 180))))  # other face up
        self.assertTrue(block_on_sill(block_at(L.MARK_X, L.MARK_YS[0], L.MARK_TOP + 0.0009)))
        self.assertFalse(block_on_sill(block_at(L.MARK_X, L.MARK_YS[0], L.MARK_TOP + 0.0011)))  # held above it
        self.assertFalse(block_on_sill(block_at(L.MARK_X, L.MARK_YS[0], L.SILL_TOP, quat=about("y", 90))))  # on edge
        self.assertFalse(block_on_sill(B1_HOME))  # on the tray
        self.assertFalse(block_on_sill(block_at(L.MARK_X, 0.125)))  # centre past the jamb's face
        self.assertFalse(block_on_sill(block_at(L.FRAME_ROOM_X - 0.001, 0.0, L.SILL_TOP)))  # centre off the sill

    def test_block_set_on_a_mark(self):
        self.assertEqual(block_set_on(B1_SET, L.BLOCK_TIGHT), 0)
        self.assertEqual(block_set_on(B2_SET, L.BLOCK_TIGHT), 1)
        self.assertEqual(block_set_on(block_at(L.MARK_X + 0.005, L.MARK_YS[1] + 0.006), L.BLOCK_TIGHT), 1)
        self.assertIsNone(block_set_on(block_at(L.MARK_X, L.MARK_YS[1] + 0.0082), L.BLOCK_TIGHT))
        self.assertEqual(block_set_on(block_at(L.MARK_X, L.MARK_YS[1] + 0.0082), L.BLOCK_LOOSE), 1)
        self.assertEqual(block_set_on(block_at(L.MARK_X, L.MARK_YS[0], quat=about("z", 14.5)), L.BLOCK_TIGHT), 0)
        self.assertIsNone(block_set_on(block_at(L.MARK_X, L.MARK_YS[0], quat=about("z", 15.5)), L.BLOCK_TIGHT))
        self.assertIsNone(block_set_on(block_at(L.MARK_X, 0.0, L.SILL_TOP), L.BLOCK_LOOSE))  # between the marks

    def test_resting_on_a_block(self):
        self.assertTrue(on_block(U_ON, B1_SET) and on_block(U_ON, B2_SET))
        self.assertAlmostEqual(gap_to_block(U_ON, B1_SET), 0.0, places=9)
        self.assertTrue(on_block(unit_at(low=L.SET_BLOCK_TOP + 0.0009), B1_SET))
        self.assertFalse(on_block(unit_at(low=L.SET_BLOCK_TOP + 0.0011), B1_SET))  # 1.1 mm above it: hovering
        self.assertTrue(on_block(unit_at(low=L.SET_BLOCK_TOP - 0.0009), B1_SET))  # solver penetration
        self.assertFalse(on_block(U_ON, B1_HOME))
        # the unit's outside face at x 0.312, the block from 0.279 to 0.309: 21 of its 30 mm under the unit
        self.assertAlmostEqual(share_under(U_ON, B1_SET), 0.7, places=6)
        b = block_at(L.MARK_X - 0.006, L.MARK_YS[0])  # 6 mm toward the room: 15 of 30 mm under the unit
        self.assertAlmostEqual(share_under(U_ON, b), 0.5, places=6)
        self.assertTrue(on_block(U_ON, block_at(L.MARK_X - 0.0058, L.MARK_YS[0])))
        self.assertFalse(on_block(U_ON, block_at(L.MARK_X - 0.0062, L.MARK_YS[0])))
        self.assertFalse(on_block(leaning(48), B1_SET))  # leaning on its edge: not resting on the block's top

    def test_share_under_matches_sampling(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            u = unit_at(CX + rng.uniform(-0.008, 0.008), rng.uniform(-0.02, 0.02),
                        quat=compose(about("z", rng.uniform(-4, 4)), about("y", rng.uniform(-2, 2))))
            b = block_at(L.MARK_X + rng.uniform(-0.015, 0.015), L.MARK_YS[0] + rng.uniform(-0.01, 0.01),
                         quat=about("z", rng.uniform(-60, 60)))
            s, t_ = np.meshgrid(np.linspace(-0.5, 0.5, 151), np.linspace(-0.5, 0.5, 151))
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

    def test_standing_short_of_the_stop_within_3mm_still_scores_one(self):
        # set 2.5 mm short of the stop and never slid out (the demo's nominal set point): 2.5 mm from the stop
        us, b1, b2 = procedure(unit_end=leaning(0, SET_PIVOT))
        r, _ = run(us, b1, b2)
        self.assertAlmostEqual(r.details["stop_gap_mm"], 2.5, places=3)
        self.assertEqual(r.score, 1.0)

    def test_either_way_up_scores_one(self):
        flipped_unit = unit_at(quat=about("x", 180))
        flipped_block = block_at(L.MARK_X, L.MARK_YS[0], quat=about("y", 180))
        r, _ = run(*procedure(b1_end=flipped_block, unit_end=flipped_unit))
        self.assertEqual(r.score, 1.0)

    def test_limits_for_the_blocks(self):
        for dy, want in ((0.0079, 1.0), (0.0081, 0.75), (0.0199, 0.75), (0.0201, 0.5)):
            r, _ = run(*procedure(b2_end=block_at(L.MARK_X, L.MARK_YS[1] + dy)))
            self.assertEqual(r.score, want, dy)
        r, _ = run(*procedure(b1_end=block_at(L.MARK_X + 0.0055, L.MARK_YS[0] - 0.0055)))  # 7.8 mm diagonally
        self.assertEqual(r.score, 1.0)

    def test_limits_for_the_unit(self):
        for dy, want in ((0.0079, 1.0), (-0.0081, 0.75), (0.0149, 0.75), (-0.0151, 0.5)):
            r, _ = run(*procedure(unit_end=unit_at(y=dy)))
            self.assertEqual(r.score, want, dy)

    def test_a_turned_block_scores_half(self):
        r, _ = run(*procedure(b1_end=block_at(L.MARK_X, L.MARK_YS[0], quat=about("z", 20))))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["blocks_set"])

    def test_blocks_at_the_ends_instead_of_the_quarter_points_score_half(self):
        ends = block_at(L.MARK_X, -0.09, L.SILL_TOP), block_at(L.MARK_X, 0.09, L.SILL_TOP)
        us, b1, b2 = procedure(b1_end=ends[0], b2_end=ends[1], unit_end=unit_at(low=L.SILL_TOP + L.BLOCK_SIZE[2]))
        r, _ = run(us, b1, b2)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["on_block"], [True, True])

    def test_standing_on_one_block_scores_half(self):
        us, b1, b2 = procedure()
        r, _ = run(us, b1, hold(B2_HOME, len(us)))
        self.assertEqual(r.details["on_block"], [True, False])
        self.assertEqual(r.score, 0.5)

    def test_unit_too_far_off_centre_scores_half(self):
        r, _ = run(*procedure(unit_end=unit_at(y=0.020)))  # its end against the left jamb
        self.assertEqual(r.score, 0.5)

    def test_blocks_set_only_scores_quarter(self):
        us, b1, b2 = procedure()
        r, _ = run(hold(U_HOME, len(us)), b1, b2)  # the right arm alone (the left arm idle)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_one_block_set_only_scores_zero(self):
        us, b1, b2 = procedure()
        r, _ = run(hold(U_HOME, len(us)), b1, hold(B2_HOME, len(us)))
        self.assertEqual(r.score, 0.0)


class WrongWays(unittest.TestCase):
    """The ways the review's setup and this one can be got wrong: each earns less than 1.0."""

    def test_the_unit_set_straight_on_the_sill_scores_zero(self):
        # the left arm alone (no blocks): the unit's bottom edge comes down on the marks
        mu = move_unit(U_MARKS, upright_x=CX)
        r, t = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)
        # set down tilted on the bare sill, before it is ever turned upright: a touch on that step
        tilted = leaning(48, SET_PIVOT, L.MARK_TOP)
        front = L.FRAME_ROOM_X - 0.060
        us = path(U_HOME, unit_at(front + 0.012, CY, L.TOP + 0.03))
        us += turn(lambda d: leaning(d, front, L.TOP + 0.03), 0, 48)
        us += path(leaning(48, front, L.TOP + 0.03), leaning(48, SET_PIVOT, L.MARK_TOP + 0.004)) + [tilted]
        r, t = run(us, hold(B1_HOME, len(us)), hold(B2_HOME, len(us)))
        self.assertEqual(t.touched_at, len(us))

    def test_touching_the_sill_scores_zero_whatever_follows(self):
        mu = move_unit(U_MARKS, upright_x=CX)
        back = path(U_MARKS, unit_at(CX - 0.03, CY, L.TOP + 0.02), U_HOME)
        us, b1, b2 = procedure()
        pre = mu + back
        r, t = run(pre + us, hold(B1_HOME, len(pre)) + b1, hold(B2_HOME, len(pre)) + b2)
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_the_unit_left_tilted_scores_quarter(self):
        # let go while still leaning on its edge (the push stopped part way): the blocks only
        for deg in (20, 3.5):
            us, b1, b2 = procedure()
            r, _ = run(with_tail(us, leaning(deg, SET_PIVOT), 70), b1, b2)
            self.assertFalse(r.details["standing"], deg)
            self.assertEqual(r.score, 0.25, deg)
        # leaning over 30 degrees counts as down (left like that, it falls back toward the room)
        r, _ = run(with_tail(us, leaning(48, SET_PIVOT), 70), b1, b2)
        self.assertTrue(r.details["lying_down"])
        self.assertEqual(r.score, 0.0)

    def test_the_unit_held_upright_scores_quarter(self):
        us, b1, b2 = procedure()
        r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(holding_unit(us[-1]), 70))
        self.assertEqual(r.details["on_block"], [False, False])
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_standing_on_the_blocks_short_of_the_stop_scores_quarter(self):
        # upright on both blocks but 5 mm short of the outside stop (pushed upright, never against it)
        us, b1, b2 = procedure(unit_end=unit_at(CX - 0.004))
        r, _ = run(us, b1, b2)
        self.assertFalse(r.details["in_rebate"])
        self.assertEqual(r.score, 0.25)

    def test_blocks_off_their_marks(self):
        for dx, want in ((0.012, 0.75), (-0.012, 0.75), (0.025, 0.5)):
            b = block_at(L.MARK_X, L.MARK_YS[0] + dx)
            r, _ = run(*procedure(b1_end=b))
            self.assertEqual(r.score, want, dx)
        r, _ = run(*procedure(b1_end=block_at(L.MARK_X, L.MARK_YS[0] + 0.012),
                              b2_end=block_at(L.MARK_X, L.MARK_YS[1] - 0.012)))
        self.assertEqual(r.score, 0.75)

    def test_one_arm_idle(self):
        us, b1, b2 = procedure()
        r, _ = run(hold(U_HOME, len(us)), b1, b2)  # left arm idle: the blocks only
        self.assertEqual(r.score, 0.25)
        mu = move_unit(U_MARKS, upright_x=CX)  # right arm idle: no blocks, the unit comes down on the marks
        r, _ = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        self.assertEqual(r.score, 0.0)

    def test_a_momentary_dip_onto_the_sill_counts(self):
        us, b1, b2 = procedure()
        k = len(us) - 60
        rocked = unit_at(low=L.SILL_TOP, quat=about("x", 4.5))  # a corner down on the sill for one step
        r, t = run(us[:k] + [rocked] + us[k + 1:], b1, b2)
        self.assertEqual(t.touched_at, k + 1)
        self.assertEqual(r.score, 0.0)

    def test_both_blocks_stacked_on_one_mark_score_zero(self):
        stacked = block_at(L.MARK_X, L.MARK_YS[0], L.SET_BLOCK_TOP)
        us, b1, b2 = procedure(b2_end=stacked, unit_end=unit_at(low=L.SET_BLOCK_TOP + L.BLOCK_SIZE[2]))
        r, _ = run(us, b1, b2)
        self.assertEqual(r.details["on_block"], [False, False])
        self.assertEqual(r.score, 0.0)

    def test_the_unit_on_the_blocks_while_they_are_still_on_the_tray(self):
        on_tray = unit_at(L.TRAY_X, L.TRAY_Y, L.TRAY_TOP + L.BLOCK_SIZE[2])
        self.assertFalse(on_block(on_tray, B1_HOME))
        r, _ = run(hold(U_HOME, 5) + hold(on_tray, 70), hold(B1_HOME, 75), hold(B2_HOME, 75))
        self.assertEqual(r.score, 0.0)

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


class LetGo(unittest.TestCase):
    """A unit held, or hovering, within reach of the blocks earns none of the resting tiers. Resting needs the unit
    and the block let go (both finger zones 30 mm or more from each on every step of the last 1 s), still over it,
    and the support within 1 mm. Unreadable arms are never let go."""

    def held_at_end(self, unit_end, n=70, who=holding_unit):
        us, b1, b2 = procedure(unit_end=unit_end)
        return run(us, b1, b2, arms=hold(AWAY, len(us) - n) + hold(who(us[-1]), n))

    def test_held_on_the_blocks_earns_only_the_blocks(self):
        r, _ = self.held_at_end(U_ON)
        self.assertEqual(r.details["on_block"], [False, False])
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_held_hovering_just_above_the_blocks_earns_only_the_blocks(self):
        for low in (0.0005, 0.0015, 0.002):
            r, _ = self.held_at_end(unit_at(low=L.SET_BLOCK_TOP + low))
            self.assertEqual(r.score, 0.25, low)

    def test_a_gap_over_1mm_is_not_resting(self):
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
        top = U_ON.high[2]
        for gap, want in ((0.029, False), (0.031, True)):
            tips_z = top + rigstate.zone_radius(1.0) + gap
            hands = {"left": arm((CX, CY, tips_z + 0.013), 1.0), "right": AWAY["right"]}
            self.assertAlmostEqual(hand_gap(hands, U_ON, L.UNIT_SIZE), gap, places=6)
            self.assertEqual(let_go_now(hands, U_ON, L.UNIT_SIZE), want)
            us, b1, b2 = procedure()
            r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(hands, 70))
            self.assertEqual(r.score, 1.0 if want else 0.25, gap)

    def test_a_fingertip_still_on_its_face_has_not_let_go(self):
        # the closed fingertip that pushed it upright, still touching its room face: not let go
        us, b1, b2 = procedure()
        tip = np.array([U_ON.low[0] - 0.002, -0.045, L.SET_BLOCK_TOP + 0.045])
        hands = {"left": AWAY["left"], "right": rigstate.Arm(tcp=tip + np.array([-0.01, 0, 0.008]), tips=tip,
                                                              opening=0.0)}
        self.assertFalse(let_go_now(hands, U_ON, L.UNIT_SIZE))
        r, _ = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(hands, 70))
        self.assertEqual(r.details["on_block"], [False, False])
        self.assertLess(r.score, 0.5)

    def test_unreadable_arms_are_never_let_go(self):
        nan = float("nan")
        good = AWAY
        broken = [
            None,
            {"left": good["left"]},
            {"left": arm((nan, 0.3, 1.0)), "right": good["right"]},
            {"left": good["left"], "right": rigstate.Arm(tcp=good["right"].tcp, tips=np.array([0.3, np.inf, 1.0]),
                                                          opening=1.0)},
            {"left": good["left"], "right": arm((0.32, -0.305, 1.0), nan)},
            {"left": SimpleNamespace(tcp=(0.3, 0.3)), "right": good["right"]},
            "garbage",
        ]
        us, b1, b2 = procedure()
        for k, bad in enumerate(broken):
            self.assertFalse(let_go_now(bad, U_ON, L.UNIT_SIZE), k)
            self.assertIsNone(hand_gap(bad, U_ON, L.UNIT_SIZE), k)
            r, t = run(us, b1, b2, arms=bad)
            self.assertEqual(r.score, 0.0, k)
            self.assertEqual(t.arms_unreadable, len(us), k)
            r, t = run(us, b1, b2, arms=hold(AWAY, len(us) - 70) + hold(bad, 70))
            self.assertEqual(r.score, 0.0, k)

    def test_one_unreadable_step_restarts_the_count(self):
        us, b1, b2 = procedure(settle=90)
        n = len(us)
        for gap_at, want in ((n - 10, 0.0), (n - 20, 0.0), (n - 21, 1.0)):
            arms = hold(AWAY, n)
            arms[gap_at] = None
            r, t = run(us, b1, b2, arms=arms)
            self.assertEqual(r.score, want, gap_at)

    def test_a_distance_that_cannot_be_computed_is_not_let_go(self):
        bad = SimpleNamespace(pos=np.array([0.3, 0.0]), quat_xyzw=np.array([0, 0, 0, 1.0]))
        self.assertFalse(let_go_now(AWAY, bad, L.UNIT_SIZE))
        self.assertEqual(hand_gap(AWAY, bad, L.UNIT_SIZE), -np.inf)

    def test_a_held_block_is_not_set(self):
        us, b1, b2 = procedure()
        n = len(us)
        r, _ = run(hold(U_HOME, n), b1, b2, arms=hold(AWAY, n - 70) + hold(holding_block(B1_SET), 70))
        self.assertFalse(r.details["blocks_set"])
        self.assertEqual(r.score, 0.0)

    def test_gripped_again_after_resting_loses_the_resting_tiers(self):
        us, b1, b2 = procedure()
        r, t = run(us, b1, b2)
        self.assertTrue(r.success)
        tail = hold(U_ON, 30)
        r2, _ = run(tail, hold(b1[-1], 30), hold(b2[-1], 30), tracker=t, arms=[holding_unit(u) for u in tail])
        self.assertEqual(r2.score, 0.25)
        self.assertFalse(r2.success)


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        return [
            (unit_at(x=nan), B1_SET, B2_SET),
            (state(U_ON.pos, U_ON.size, quat=(nan, 0, 0, nan)), B1_SET, B2_SET),
            (U_ON, block_at(L.MARK_X, inf), B2_SET),
            (U_ON, B1_SET, state(B2_SET.pos, B2_SET.size, vel=(0, nan, 0))),
            (state(U_ON.pos, U_ON.size, quat=(0, 0, 0, 0)), B1_SET, B2_SET),
            (state(U_ON.pos, U_ON.size, spin=(inf, 0, 0)), B1_SET, B2_SET),
            (unit_at(50.0, -40.0, -10.0, quat=about("y", 123)), block_at(-3.0, 8.0, 100.0, quat=about("x", 77)),
             block_at(1e6, -1e6, -1e6)),
        ]

    def test_no_crash_and_clean_details(self):
        for u, a, b in self.odd():
            t = Tracker()
            t.update(U_HOME, B1_HOME, B2_HOME)
            r = t.update(u, a, b)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)

    def test_a_touch_is_latched_when_only_a_block_is_unreadable(self):
        t = Tracker()
        t.update(U_HOME, B1_HOME, B2_HOME)
        bad_block = state(B2_SET.pos, B2_SET.size, vel=(0, float("nan"), 0))
        r = t.update(U_MARKS, B1_SET, bad_block)
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_blow_ups_score_zero_and_leave_the_tracker_alone(self):
        for k, (u, a, b) in enumerate(self.odd()[:6]):
            us, b1, b2 = procedure()
            i = len(us) - 80
            t = Tracker()
            for j, (uu, aa, bb) in enumerate(zip(us, b1, b2)):
                r = t.update(*((u, a, b) if j == i else (uu, aa, bb)), AWAY)
            self.assertEqual(r.score, 1.0, k)
            self.assertEqual(t.bad_steps, 1)

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
        mu = move_unit(U_MARKS, upright_x=CX)
        r, t = run(mu, hold(B1_HOME, len(mu)), hold(B2_HOME, len(mu)))
        self.assertTrue(t.touched)
        t.update(U_HOME, B1_HOME, B2_HOME)  # everything jumps back in one step: a new episode
        self.assertFalse(t.touched)
        self.assertEqual(t.steps, 1)
        r, _ = run(*procedure(), tracker=t)
        self.assertEqual(r.score, 1.0)

    def test_putting_things_back_slowly_does_not_reset(self):
        mu = move_unit(U_MARKS, upright_x=CX)
        back = path(U_MARKS, unit_at(CX - 0.03, CY, L.TOP + 0.02), U_HOME)
        us = mu + back + hold(U_HOME, 10)
        r, t = run(us, hold(B1_HOME, len(us)), hold(B2_HOME, len(us)))
        self.assertTrue(t.touched)
        self.assertEqual(t.steps, len(us))

    def test_the_blocks_must_be_back_on_the_tray_too(self):
        us, b1, b2 = procedure()
        _, t = run(us, b1, b2)
        t.touched = True
        t.update(U_HOME, B1_SET, B2_SET)
        self.assertTrue(t.touched)


OFFLINE = HERE.parent.parent / "tools" / "offline"
STUB = OFFLINE / "yam_stub"


def load_task():
    """task.py, imported through the offline yam_arena stub."""
    for p in (str(STUB), str(OFFLINE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("glaze_task_under_test", HERE / "task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame_boxes():
    return {k: (np.array(c) - np.array(s) / 2, np.array(c) + np.array(s) / 2) for k, (c, s) in L.FRAME_BOXES.items()}


def point_box(p, lo, hi):
    """Distance from points (N, 3) to an axis-aligned box (negative inside: the depth)."""
    q = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    out = np.linalg.norm(q, axis=-1)
    inside = np.all((p >= lo) & (p <= hi), axis=-1)
    return np.where(inside, -np.min(np.minimum(p - lo, hi - p), axis=-1), out)


@unittest.skipUnless(STUB.is_dir(), "needs the offline yam_arena stub (tools/offline/yam_stub)")
class Demo(unittest.TestCase):
    """The scripted demo against the frame and the limits (no simulator: the plan's numbers and the offline
    kinematics and capsule model of tools/offline)."""

    HOLD_S = 3.0  # yam_scripted's default hold after the plan before it stops the episode (-P hold_s)

    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        old = {k: os.environ.pop(k, None) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
        try:
            cls.plan = cls.task.demo()
        finally:
            os.environ.update({k: v for k, v in old.items() if v is not None})
        import yamkin
        cls.K = yamkin

    def segments(self, arm):
        t, prev, out = 0.0, None, []
        for w in self.plan[arm]:
            out.append((t, t + w.duration, prev, w))
            t, prev = t + w.duration, w
        return out

    def stop(self) -> float:
        return max(sum(w.duration for w in self.plan[a]) for a in ("left", "right")) + self.HOLD_S

    def commanded(self, arm):
        """Each control step's commanded gripper (rigstate.Arm) and rotation for one arm (stub ArmPlan)."""
        from yam_arena.demo import ArmPlan

        plan = ArmPlan(list(self.plan[arm]), self.K.BASES["station"][arm])
        out = []
        for k in range(round(self.stop() * HZ) + 1):
            pos, rot, finger, label = plan.target(k / HZ)
            pos = np.asarray(pos, float)
            tips = pos + self.task.PAD * np.asarray(rot)[:, 2]
            out.append((rigstate.Arm(tcp=pos, tips=tips, opening=float(np.clip(finger / self.task.OPEN, 0, 1))),
                        np.asarray(rot), label))
        return out

    def test_grips_and_releases_are_slow_enough(self):
        for arm in ("left", "right"):
            finger = self.task.OPEN
            for _, _, _, w in self.segments(arm):
                change = abs(w.finger - finger) / abs(self.task.OPEN)
                if change > 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertLessEqual(1.5 * change / (w.duration * HZ), 0.05, (arm, w.label))
                finger = w.finger

    def test_set_downs_are_slow(self):
        seg = {(a, w.label): w for a in ("left", "right") for _, _, _, w in self.segments(a)}
        self.assertGreaterEqual(seg[("right", "slowly down onto the mark")].duration, 2.0)
        self.assertGreaterEqual(seg[("left", "set the bottom edge on the blocks")].duration, 2.0)
        for label in ("push it upright to 38 degrees", "push it upright to 0 degrees"):
            self.assertGreaterEqual(seg[("right", label)].duration, 1.5, label)

    def test_trims_are_zero_and_the_releases_are_on_the_marks(self):
        self.assertEqual(self.task.PLACE_TRIM, (0.0, 0.0))
        self.assertEqual(self.task.BLOCK_TRIM, ((0.0, 0.0), (0.0, 0.0)))
        lets = [w for w in self.plan["right"] if w.label == "let go of the block"]
        want = [self.task.block_tcp(L.MARK_X, my, L.MARK_TOP + 0.0015) for my in L.MARK_YS]
        for w, p in zip(lets, want):
            np.testing.assert_allclose(w.pos, p, atol=1e-9)
        self.assertEqual(len(lets), 2)

    def test_the_real_pad_faces_cover_the_blocks_sides(self):
        # GPU 2026-11-14: sized by a 27 mm rectangle, the pitched pads' real faces (15 mm wide at the tip) passed
        # 0.3 mm over a block's top and closed on nothing. Now, at each pick and each release on a mark, the real
        # faces' lowest corner is 1 to 3 mm over the block's bottom and their lower edge, from the tip back to the
        # TCP level, at least 4 mm below its top
        t = self.task
        rot = t.tiltgrip.rotation(0.0, t.BLOCK_PITCH)
        down = -rot[:, 1] if rot[2, 1] > 0 else rot[:, 1]
        bottoms = {"slowly down beside its 30 mm sides": L.TRAY_TOP, "let go of the block": L.MARK_TOP + 0.0015}
        seen = 0
        for w in self.plan["right"]:
            if w.label not in bottoms:
                continue
            seen += 1
            tcp, z0 = np.asarray(w.pos), bottoms[w.label]
            low = t.pad_face_corners(tcp, rot, L.BLOCK_SIZE[1])[:, 2].min() - z0
            self.assertGreaterEqual(low, 0.001, w.label)
            self.assertLessEqual(low, 0.003, w.label)
            for a, hw in t.PAD_FACE[1:]:
                edge = tcp + rot[:, 2] * a + down * hw
                self.assertLessEqual(edge[2] - z0, L.BLOCK_SIZE[2] - 0.004, (w.label, a))
        self.assertEqual(seen, 4)

    def test_the_handover_order(self):
        t = self.task
        self.assertLess(t.T_SET, t.T_TOUCH)  # the unit is on the blocks before the fingertip arrives
        self.assertLess(t.T_TOUCH, t.T_UNLOAD)  # the fingertip is on it before the left arm gives it up
        ends = {w.label: t1 for _, t1, _, w in self.segments("left")}
        self.assertAlmostEqual(ends["back out along the glass"], t.T_LEFT_OUT, places=6)
        starts = {w.label: t0 for t0, _, _, w in self.segments("right")}
        self.assertGreaterEqual(starts["push it upright to 38 degrees"], t.T_LEFT_OUT)
        # when the push starts, the left hand's pads (opened) are 15 mm or more clear of the leaning unit
        left = self.commanded("left")
        a, rot, _ = left[round(t.T_PUSH * HZ)]
        corners16 = t.pad_corners(a.tcp, rot, a.opening * rigstate.FULL_GAP)
        pts = np.array([p + (q - p) * f for p in corners16 for q in corners16 for f in np.linspace(0, 1, 6)])
        u = leaning(48, t.SET_PIVOT_X)
        d = rigstate.box_distances(pts, u.pos, L.UNIT_SIZE, rotation(u.quat_xyzw))
        self.assertGreater(float(d.min()), 0.015)

    def test_the_arms_stay_13cm_apart(self):
        left, right = self.commanded("left"), self.commanded("right")
        d = min(float(np.linalg.norm(a[0].tcp - b[0].tcp)) for a, b in zip(left, right))
        self.assertGreaterEqual(d, 0.130)

    def test_15mm_clear_of_the_head_and_the_jambs(self):
        # every waypoint of both arms, by the capsule model (fingers as 8 x 27 x 38 mm pads); the left hand holding
        # the unit tilted at its loaded (sagged) pose, as the simulator will put it; and, if there were no sag at
        # all, the commanded pose still 10 mm or more clear
        K = self.K
        boxes = {k: v for k, v in frame_boxes().items() if k != "sill"}
        sag_r = self.task._rotvec(np.radians(self.task.SAG_ROT_DEG))
        worst = {}
        for arm in ("left", "right"):
            seed = None
            for _, _, _, w in self.segments(arm):
                rot = getattr(w.yaw, "rotation", None)
                rot = K.grip_rotation(float(w.yaw)) if rot is None else rot
                compensated = arm == "left" and rot[2, 2] > -0.95 and not np.allclose(rot, self.task._ROT_U)
                q, err, _ = K.solve(np.asarray(w.pos), K.matrix_to_quat_xyzw(rot),
                                    seed if seed is not None else K.READY, K.BASES["station"][arm], iters=200)
                seed = q
                self.assertLess(err, 0.001, (arm, w.label))
                d = self.clearance(arm, q, w.finger, boxes)
                if compensated:  # holding the unit tilted: the simulator puts the hand where the sag takes it
                    worst["no sag"] = min(worst.get("no sag", 1.0), d)
                    pos, r = np.asarray(w.pos) + np.asarray(self.task.SAG), sag_r @ rot
                    q2, err, _ = K.solve(pos, K.matrix_to_quat_xyzw(r), q, K.BASES["station"][arm], iters=200)
                    self.assertLess(err, 0.001, (arm, w.label))
                    d = self.clearance(arm, q2, w.finger, boxes)
                worst[arm] = min(worst.get(arm, 1.0), d)
        self.assertGreaterEqual(worst["left"], 0.015)
        self.assertGreaterEqual(worst["right"], 0.015)
        self.assertGreaterEqual(worst["no sag"], 0.010)

    def clearance(self, arm, q, finger, boxes):
        K = self.K
        caps = K.arm().capsules(q, K.BASES["station"][arm], finger)
        p, r = K.tcp_pose(q, K.BASES["station"][arm])
        gap = K.finger_gap(finger)
        pts = []
        for side in (-1, 1):
            c0 = p + r[:, 0] * side * (gap / 2 + 0.004)
            for u in (-0.004, 0.004):
                for v in np.linspace(-0.0135, 0.0135, 5):
                    for w in np.linspace(-0.025, 0.013, 8):
                        pts.append(c0 + r[:, 0] * u + r[:, 1] * v + r[:, 2] * w)
        pts = np.array(pts)
        best = 1.0
        for lo, hi in boxes.values():
            best = min(best, float(point_box(pts, lo, hi).min()))
            for link, a, b, rad in caps:
                if link in K.FINGER_LINKS:
                    continue
                seg = a + np.linspace(0, 1, 12)[:, None] * (b - a)
                best = min(best, float(point_box(seg, lo, hi).min()) - rad)
        return best

    def test_the_payload_sag_constants(self):
        # SAG is the static estimate J K^-1 J^T of the unit's weight and moment at the set pose (joints 80/80/80 and
        # 10/10/10 N m/rad): recomputed here
        K, t = self.K, self.task
        rot = t.tiltgrip.rotation(t.YAW_L, t.TILT)
        tcp = t.left_tcp(t.TILT, t.SET_PIVOT_X, L.SET_BLOCK_TOP, 0.0)
        q, err, _ = K.solve(tcp, K.matrix_to_quat_xyzw(rot), K.READY, K.BASES["station"]["left"], iters=300)
        p, _, jl, ja = K.arm().tcp_pose_jacobian(q, K.BASES["station"]["left"])
        jac = np.vstack([jl, ja])
        comp = jac @ np.diag(1 / np.array([80, 80, 80, 10, 10, 10.0])) @ jac.T
        c, _ = t.unit_pose(t.TILT, t.SET_PIVOT_X, L.SET_BLOCK_TOP, 0.0)
        f = np.array([0, 0, -L.UNIT_MASS * 9.81])
        d = comp @ np.r_[f, np.cross(c - p, f)]
        np.testing.assert_allclose(d[:3], t.SAG, atol=0.0005)
        np.testing.assert_allclose(np.degrees(d[3:]), t.SAG_ROT_DEG, atol=0.3)

    def test_the_leaning_unit_is_held_by_the_fingertip_and_the_push_is_light(self):
        # the unit leaning 48 degrees on its bottom edge (on the blocks, from y -0.065 to +0.065) and on the
        # fingertip: turning about the line through one block's end and the fingertip must lift its centre of mass
        t = self.task
        th, up = math.radians(t.PUSH_UP[0][0]), t.PUSH_UP[0][1]
        u = np.array([-math.sin(th), 0, math.cos(th)])
        m = np.array([math.cos(th), 0, math.sin(th)])
        com = 0.040 * u + 0.012 * m
        tip = up * u + np.array([0, t.PUSH_Y, 0])
        for qy, by in ((0.065, -0.065), (-0.065, 0.065), (0.035, -0.065), (-0.035, 0.065)):
            q, b = np.array([0, qy, 0.0]), np.array([0, by, 0.0])
            w = (tip - q) / np.linalg.norm(tip - q)
            for sign in (1, -1):
                vb, vc = np.cross(sign * w, b - q), np.cross(sign * w, com - q)
                self.assertFalse(vb[2] > 1e-9 and vc[2] < -1e-9, (qy, sign))
        # the force on the fingertip that holds it leaning: well under 8 N (and under 1 N)
        weight = L.UNIT_MASS * 9.81
        for deg, s in t.PUSH_UP:
            a = math.radians(deg)
            lean = 0.040 * math.sin(a) - 0.012 * math.cos(a)
            self.assertLess(weight * max(lean, 0.0) / s, 1.0, deg)

    def test_the_demo_lets_go_and_clears_everything_before_the_policy_stops(self):
        left, right = self.commanded("left"), self.commanded("right")
        first = None
        for k, (a, b) in enumerate(zip(left, right)):
            hands = {"left": a[0], "right": b[0]}
            ok = all(let_go_now(hands, s, size) for s, size in
                     ((U_ON, L.UNIT_SIZE), (B1_SET, L.BLOCK_SIZE), (B2_SET, L.BLOCK_SIZE)))
            first = (first if first is not None else k) if ok else None
        self.assertIsNotNone(first)
        success_at = first + L.WINDOW_STEPS - 1 + round(self.task.SUCCESS_HOLD_S * HZ)
        stop = round(self.stop() * HZ)
        self.assertLess(success_at, stop - HZ)
        end = {"left": left[-1][0], "right": right[-1][0]}
        self.assertGreater(hand_gap(end, U_ON, L.UNIT_SIZE), 2 * L.LET_GO_CLEAR)

    def test_the_run_limit_covers_the_demo(self):
        meta = json.loads((HERE / "task.json").read_text())
        limit = int(re.search(r"([\d,]+) steps", meta["run_limit"]).group(1).replace(",", ""))
        self.assertLessEqual(round(self.stop() * HZ), limit)


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

    def start(self):
        return {"glass_unit": U_HOME, "block_1": B1_HOME, "block_2": B2_HOME}

    def test_demo_switches(self):
        task = load_task()
        with self.env(DEMO_IDLE_ARM=None, DEMO_HOLD_ONLY=None):
            self.assertEqual(set(task.demo()), {"left", "right"})
        with self.env(DEMO_IDLE_ARM="left", DEMO_HOLD_ONLY=None):
            self.assertEqual(set(task.demo()), {"right"})
        with self.env(DEMO_IDLE_ARM="right", DEMO_HOLD_ONLY=None):
            self.assertEqual(set(task.demo()), {"left"})
        with self.env(DEMO_HOLD_ONLY="1", DEMO_IDLE_ARM=None):
            self.assertEqual(task.demo(), {})

    def test_hooks_and_trace(self):
        task = load_task()
        with tempfile.TemporaryDirectory() as d:
            trace = os.path.join(d, "trace.jsonl")
            with self.env(TASK_TRACE=trace):
                self.assertFalse(task.success(self.start()))
            with open(trace) as f:
                row = json.loads(f.read().splitlines()[-1])
            for key in ("step", "score", "touched", "sill_gap_mm", "min_sill_gap_mm", "stop_gap_mm", "in_rebate",
                        "on_block", "gap_mm", "block_mark_mm", "unit_pos", "block_1_pos"):
                self.assertIn(key, row)
        self.assertEqual(task.score(self.start()), 0.0)
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for name in ("glass_unit", "block_1", "block_2", "sill", "head", "jamb_left", "jamb_right", "stop_sill",
                     "stop_head", "glazing_bead"):
            self.assertIn(name, names)
        bead = next(o for o in task.OBJECTS if o.name == "glazing_bead")
        self.assertIsNone(bead.mass)  # a fixed prop

    def test_task_json_agrees_with_the_code(self):
        task = load_task()
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], task.INSTRUCTION)
        self.assertEqual(meta["category"], "Impact on jobs")
        sc = meta["scoring"]
        for phrase in ("30 mm", "last 20 control steps (1 s)", "within 2 mm and 2 degrees", "within 1 mm of the block",
                       "unreadable", "for 1 s", "0.0 mm or less", "air gap", "chalk", "within 3 mm of the outside stop",
                       "tilt under 3 degrees"):
            self.assertIn(phrase, sc, phrase)
        for stale in ("0.5 mm or less", "channel", "near miss is not a touch (0.5"):
            self.assertNotIn(stale, sc, stale)
        fid = meta["fidelity"]
        for stale in ("from above", "cut down", "bottom rail"):
            self.assertNotIn(stale, fid, stale)
        for phrase in ("bead", "sealant"):
            self.assertIn(phrase, fid, phrase)
        pat = r"^Dream-1 run: [0-9.]+ \([0-9]+ steps\); demo: 1\.0 \([0-9]+ steps\)$"
        self.assertRegex(meta.get("scores", ""), pat)
        readme = (HERE / "README.md").read_text()
        self.assertIn("## Results in simulation", readme)
        self.assertIn("0.0 mm or less", readme)
        self.assertNotIn("0.5 mm or less above the surface under it", readme)

    def test_the_instruction_states_the_goal_only(self):
        task = load_task()
        meta = json.loads((HERE / "task.json").read_text())
        self.assertLess(len(task.INSTRUCTION), 300)
        low = task.INSTRUCTION.lower()
        for word in (" then ", "first", "before", "after", "tilt", "push", "lower", "slide", "lean", "room side"):
            self.assertNotIn(word, low, word)
        for word in ("blocks", "marks", "let go", "never touch the sill", "frame"):
            self.assertIn(word, low, word)
        self.assertNotIn(" then ", meta["title"].lower())

    def test_text_rules(self):
        """No dashes of the em or en kind, lines of 120 characters or less, no AI attribution."""
        bad = re.compile("—|–|co-authored|claude|anthropic", re.IGNORECASE)
        mine = ("task.json", "README.md", "REVISION.md", "task.py", "layout.py", "scoring.py", "offline_unit.py",
                "tiltgrip.py", "rigstate.py")
        for name in mine:
            text = (HERE / name).read_text()
            m = bad.search(text)
            self.assertIsNone(m, f"{name}: {m.group(0) if m else ''}")
        for f in [HERE / n for n in mine if n != "task.json"] + sorted((HERE / "tests").glob("*.py")):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                self.assertLessEqual(len(line), 120, f"{f.name}:{i}")


if __name__ == "__main__":
    unittest.main()
