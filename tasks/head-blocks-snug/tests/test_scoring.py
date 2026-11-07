"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic head and block states and synthetic
arm snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
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
    HEAD, NAMES, Tracker, corners, evaluate, footprint, gap, head_low, heading_deg, let_go, point_box_distance,
    released, rise, rotation, segment_polygon_distance, side_of, tilt_deg, valid, zone_gap, zone_radius,
)

HZ = 20.0
TIPS = 0.013  # pad tips below the TCP
R = L.HEAD_R
HW = L.BLOCK_HALF[1]  # half the block's 50 mm width (across, toward the head)
BL, BR = "head_block_left", "head_block_right"
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
OPEN = 1.0
GRIP = L.BLOCK_SIZE[0] / L.FULL_GAP  # rigstate opening closed on a block's 70 mm length


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


def _ns(c, quat, ext, vel, spin):
    c = np.array(c, float)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def block_state(center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like block: centre, orientation, velocities, world bounding box."""
    ext = np.abs(rotation(quat)) @ np.array(L.BLOCK_HALF, float)
    return _ns(center, quat, ext, vel, spin)


def block(x, y, yaw=0.0, lean=0.0, lift=0.0, flat=False, turned=False, on_table=False, vel=(0, 0, 0),
          spin=(0, 0, 0)):
    """A block with its centre over (x, y), turned ``yaw`` degrees about the vertical and leaned ``lean`` degrees
    about its 70 mm axis (positive: its top toward -y), its lowest point ``lift`` above the board's top (or the
    table's, ``on_table``). ``flat`` lays it on its 70 x 80 mm face; ``turned`` turns it 90 degrees (its 50 mm side
    along the board)."""
    q = about("x", lean)
    if flat:
        q = qmul(about("x", 90.0), q)
    q = qmul(about("z", yaw + (90.0 if turned else 0.0)), q)
    probe = block_state((x, y, 0.0), q)
    base = L.TOP if on_table else L.BOARD_TOP
    z = base + lift - float(corners(probe)[:, 2].min())
    return block_state((x, y, z), q, vel, spin)


def head(x=L.HEAD_X, y=L.HEAD_Y, tilt=0.0, lift=0.0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """The head with its centre over (x, y), its axis tilted ``tilt`` degrees about x, its lowest point ``lift``
    above the board's top."""
    q = about("x", tilt)
    a = rotation(q)[:, 2]
    ext = R * np.sqrt(np.clip(1 - a * a, 0, 1)) + (L.HEAD_H / 2) * np.abs(a)
    probe = _ns((x, y, 0.0), q, ext, vel, spin)
    z = L.BOARD_TOP + lift - (head_low(probe) + L.BOARD_TOP)
    return _ns((x, y, z), q, ext, vel, spin)


def beside(side, g, x=L.HEAD_X, h=None, **kw):
    """A block standing upright beside the head (+1 left, -1 right) with a gap ``g`` from it."""
    h = head() if h is None else h
    return block(x, float(h.pos[1]) + side * (R + g + HW), **kw)


def start(name):
    x, y = L.BLOCKS[name]
    return block(x, y, on_table=True)


def blocks(**placed):
    """Both blocks: those given where given, the other on its starting mark."""
    out = {n: start(n) for n in NAMES}
    out.update(placed)
    return out


def done(h=None):
    """The demo's end: each block 2 mm from its side of the head, centred on it in x."""
    return blocks(**{BL: beside(1, 0.002, h=h), BR: beside(-1, 0.002, h=h)})


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def score_of(b, h=None, arms="ready"):
    arms = snap() if arms == "ready" else arms
    return Tracker(HZ).update(b, head() if h is None else h, arms).score


def trial(*steps, arms="ready"):
    """Feed the starting layout and then each (blocks, head) step to one tracker; the last step's score."""
    t = Tracker(HZ)
    t.update(blocks(), head(), snap())
    r = None
    for b, h in steps:
        r = t.update(b, h, snap() if arms == "ready" else arms)
    return r.score


def grip_tcp(s):
    """The TCP of a gripper holding a block at its top (pad tips 25 mm below it)."""
    return np.array([float(s.pos[0]), float(s.pos[1]), float(s.high[2]) - 0.025 + TIPS])


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        b, h = blocks(), head()
        self.assertTrue(Tracker.at_start(b, h))
        for n in NAMES:
            self.assertAlmostEqual(rise(b[n]), -L.BOARD_T, places=9)  # standing on the table, beside the board
            self.assertAlmostEqual(tilt_deg(b[n].quat_xyzw), 0.0, places=6)
            self.assertAlmostEqual(gap(b[n], h), L.BLOCK_Y - HW - R, places=9)
        self.assertAlmostEqual(head_low(h), 0.0, places=9)
        self.assertAlmostEqual(float(h.pos[2]), L.HEAD_Z, places=9)
        self.assertEqual(score_of(b), 0.0)

    def test_gap_upright(self):
        h = head()
        for side, g in ((1, 0.002), (-1, 0.0123), (1, 0.0)):
            self.assertAlmostEqual(gap(beside(side, g), h), g, places=9)
        self.assertAlmostEqual(gap(beside(1, -0.003), h), -0.003, places=9)  # 3 mm inside the outline

    def test_gap_yawed_matches_hand_computation(self):
        h = head()
        for yaw in (5.0, -12.0, 30.0):
            s = block(0.205, 0.090, yaw=yaw)
            r = rotation(s.quat_xyzw)[:2, :2]
            d = r.T @ (np.array([L.HEAD_X, L.HEAD_Y]) - np.array(s.pos[:2]))
            want = float(np.linalg.norm(np.maximum(np.abs(d) - np.array(L.BLOCK_HALF[:2]), 0.0))) - R
            self.assertAlmostEqual(gap(s, h), want, places=9)

    def test_gap_leaning_uses_the_nearest_point(self):
        h = head()
        upright = block(0.20, R + 0.008 + HW)
        lean_in = block(0.20, R + 0.008 + HW, lean=6.0)  # its top toward -y, toward the head
        lean_out = block(0.20, R + 0.008 + HW, lean=-6.0)
        self.assertLess(gap(lean_in, h), gap(upright, h))
        self.assertAlmostEqual(gap(lean_in, h), float(footprint(lean_in)[:, 1].min()) - R, places=9)
        self.assertLess(gap(lean_out, h), gap(upright, h))  # its bottom inner edge swings in a little too

    def test_gap_to_a_tilted_head_uses_its_axis(self):
        h = head(tilt=8.0)  # top toward -y: the axis seen from above runs from y +7 mm to -7 mm
        a = rotation(h.quat_xyzw)[:, 2] * (L.HEAD_H / 2)
        s = beside(1, 0.010)
        self.assertAlmostEqual(gap(s, h), 0.010 - abs(float(a[1])), places=9)

    def test_side_is_taken_from_the_head(self):
        h = head(y=0.006)
        self.assertEqual(side_of(block(0.20, 0.003, lift=L.HEAD_H), h), -1)  # left of the midline, right of the head
        self.assertEqual(side_of(block(0.20, 0.007, lift=L.HEAD_H), h), 1)
        self.assertEqual(side_of(beside(-1, 0.002, h=h), h), -1)

    def test_segment_polygon_distance(self):
        sq = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        self.assertEqual(segment_polygon_distance((0.5, 0.5), (0.6, 0.6), sq), 0.0)  # inside
        self.assertEqual(segment_polygon_distance((-1.0, 0.5), (2.0, 0.5), sq), 0.0)  # crossing
        self.assertAlmostEqual(segment_polygon_distance((-2.0, 0.5), (-1.0, 0.5), sq), 1.0)
        self.assertAlmostEqual(segment_polygon_distance((2.0, 2.0), (2.0, 2.0), sq), math.sqrt(2.0))
        self.assertAlmostEqual(segment_polygon_distance((0.5, 3.0), (3.0, 0.5), sq), (3.5 - 2.0) / math.sqrt(2.0))

    def test_tilt_and_heading_are_folded(self):
        self.assertAlmostEqual(tilt_deg(about("x", 180.0)), 0.0, places=6)  # upside down is the same block
        self.assertAlmostEqual(tilt_deg(about("y", 170.0)), 10.0, places=6)
        self.assertAlmostEqual(heading_deg(about("z", 180.0)), 0.0, places=6)  # either end forward
        self.assertAlmostEqual(heading_deg(about("z", -14.0)), 14.0, places=6)
        self.assertAlmostEqual(heading_deg(about("z", 97.0)), 83.0, places=6)
        self.assertAlmostEqual(tilt_deg(block(0.2, 0.2, flat=True).quat_xyzw), 90.0, places=6)

    def test_head_low(self):
        self.assertAlmostEqual(head_low(head(lift=0.004)), 0.004, places=9)
        h = head(tilt=10.0)
        self.assertAlmostEqual(head_low(h), 0.0, places=9)
        a = rotation(h.quat_xyzw)[:, 2]
        self.assertAlmostEqual(float(h.pos[2]) - L.BOARD_TOP,
                               (L.HEAD_H / 2) * abs(float(a[2])) + R * math.sqrt(1 - float(a[2]) ** 2), places=9)

    def test_finger_zone_and_let_go(self):
        s = beside(1, 0.002)
        arm = SimpleNamespace(tcp=grip_tcp(s), tips=grip_tcp(s) - [0, 0, TIPS], opening=GRIP)
        self.assertLess(zone_gap(arm, s), 0.0)
        self.assertFalse(let_go(arm, s))  # closed on it
        arm.opening = 0.9
        self.assertTrue(let_go(arm, s))  # 90 % open holds nothing
        arm.opening = 0.89
        self.assertFalse(let_go(arm, s))
        arm = SimpleNamespace(tcp=arm.tcp + [0, 0, 0.10], tips=arm.tips + [0, 0, 0.10], opening=GRIP)
        self.assertTrue(let_go(arm, s))  # closed, but well clear above it
        self.assertAlmostEqual(zone_radius(SimpleNamespace(opening=1.0)), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135))
        self.assertEqual(point_box_distance(s.pos, s), 0.0)


class Tiers(unittest.TestCase):
    def test_demo_end_scores_one(self):
        r = Tracker(HZ).update(done(), head(), snap())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual((r.details["left_side"], r.details["right_side"]), (0.5, 0.5))

    def test_snug_and_close_score_075(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.002), BR: beside(-1, 0.012)})), 0.75)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.015), BR: beside(-1, 0.0)})), 0.75)

    def test_one_snug_or_two_close_score_half(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.004)})), 0.5)
        self.assertEqual(score_of(blocks(**{BR: beside(-1, 0.003)})), 0.5)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.010), BR: beside(-1, 0.014)})), 0.5)

    def test_one_close_scores_quarter(self):
        self.assertEqual(score_of(blocks(**{BR: beside(-1, 0.009)})), 0.25)

    def test_every_tier_is_reachable(self):
        cases = {
            0.0: blocks(),
            0.25: blocks(**{BL: beside(1, 0.010)}),
            0.5: blocks(**{BL: beside(1, 0.002)}),
            0.75: blocks(**{BL: beside(1, 0.002), BR: beside(-1, 0.010)}),
            1.0: done(),
        }
        for want, b in cases.items():
            self.assertEqual(score_of(b), want)

    def test_gap_limits_are_5_and_15_mm(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.005)})), 0.5)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.0051)})), 0.25)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.015)})), 0.25)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.0151)})), 0.0)

    def test_touching_counts_as_snug(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.0), BR: beside(-1, -0.002)})), 1.0)

    def test_angle_limits(self):
        h = head()
        # a lean or a turn swings an edge or a corner toward the head: the gap is measured where it really is
        for off, kw, ok in ((0.020, {"lean": -10.0}, True), (0.020, {"lean": -10.5}, False),
                            (0.020, {"lean": 10.0}, True), (0.012, {"yaw": 15.0}, True),
                            (0.012, {"yaw": -16.0}, False)):
            s = block(0.20, R + off + HW, **kw)
            self.assertTrue(0.005 < gap(s, h) <= 0.015, (kw, gap(s, h)))
            self.assertEqual(score_of(blocks(**{BL: s})), 0.25 if ok else 0.0, kw)
        # a block turned within the limit is measured at its nearest corner
        s = block(0.20, R + 0.004 + HW, yaw=12.0)
        self.assertGreater(gap(s, h), 0.0)
        self.assertEqual(score_of(blocks(**{BL: s})), 0.5 if gap(s, h) <= 0.005 else 0.25)

    def test_must_stand_on_the_board(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.002, lift=0.003)})), 0.5)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.002, lift=0.0035)})), 0.0)  # held above it
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.002, lift=-0.0035)})), 0.0)

    def test_must_be_centred_on_the_head_along_x(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.002, x=L.HEAD_X + 0.025)})), 0.5)
        self.assertEqual(score_of(blocks(**{BL: beside(1, 0.002, x=L.HEAD_X - 0.026)})), 0.0)

    def test_sides_follow_the_head(self):
        h = head(x=0.203, y=0.004)  # pushed 5 mm, within the 8 mm allowance
        b = blocks(**{BL: beside(1, 0.002, x=0.203, h=h), BR: beside(-1, 0.002, x=0.203, h=h)})
        self.assertEqual(trial((b, h)), 1.0)

    def test_success_needs_everything_still(self):
        t = Tracker(HZ)
        r = t.update(blocks(**{BL: beside(1, 0.002, vel=(0.02, 0, 0)), BR: beside(-1, 0.002)}), head(), snap())
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r = t.update(done(), head(spin=(0, 0, 0.6)), snap())
        self.assertFalse(r.success)
        self.assertTrue(t.update(done(), head(), snap()).success)

    def test_score_is_idempotent(self):
        t = Tracker(HZ)
        arms = snap()
        t.update(done(), head(), arms)
        a = t.result(done(), head(), arms).score
        b = t.result(done(), head(), arms).score
        self.assertEqual((a, b, t.steps), (1.0, 1.0, 1))


class Release(unittest.TestCase):
    def setUp(self):
        self.b = done()

    def test_a_held_block_does_not_count(self):
        arms = snap(left_tcp=grip_tcp(self.b[BL]), left_open=GRIP)
        r = Tracker(HZ).update(self.b, head(), arms)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["blocks"][BL]["let_go"])

    def test_open_fingers_on_the_block_have_let_go(self):
        self.assertEqual(score_of(self.b, arms=snap(left_tcp=grip_tcp(self.b[BL]), left_open=0.95)), 1.0)

    def test_closed_gripper_clear_of_the_blocks(self):
        tcp = grip_tcp(self.b[BL]) + [0.0, 0.0, 0.08]
        self.assertEqual(score_of(self.b, arms=snap(left_tcp=tcp, left_open=0.0)), 1.0)

    def test_either_gripper_can_hold_either_block(self):
        arms = snap(right_tcp=grip_tcp(self.b[BL]), right_open=GRIP)
        self.assertEqual(score_of(self.b, arms=arms), 0.5)
        arms = snap(left_tcp=grip_tcp(self.b[BR]), left_open=GRIP)
        self.assertEqual(score_of(self.b, arms=arms), 0.5)


class ZeroConditions(unittest.TestCase):
    def test_head_pushed_more_than_8_mm(self):
        for dx, dy, want in ((0.0, -0.0081, 0.0), (0.008, 0.0, 1.0), (0.0057, 0.0057, 0.0), (0.0056, 0.0056, 1.0)):
            h = head(x=L.HEAD_X + dx, y=dy)
            b = blocks(**{BL: beside(1, 0.002, h=h), BR: beside(-1, 0.002, h=h)})
            self.assertEqual(trial((b, h)), want, (dx, dy))

    def test_a_push_counts_even_if_the_head_is_put_back(self):
        t = Tracker(HZ)
        t.update(blocks(), head(), snap())
        t.update(blocks(**{BL: beside(1, 0.0, h=head(y=-0.012))}), head(y=-0.012), snap())  # shoved 12 mm
        r = t.update(done(), head(), snap())  # and back on its marks
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["head_disturbed"])
        self.assertEqual(t.first_disturbed_step, 2)
        self.assertEqual(r.details["worst_shift_mm"], 12.0)

    def test_head_lifted(self):
        self.assertEqual(trial((done(), head(lift=0.010)), (done(), head())), 1.0)
        self.assertEqual(trial((done(), head(lift=0.011)), (done(), head())), 0.0)

    def test_head_tipped(self):
        t = Tracker(HZ)
        t.update(done(), head(tilt=10.5), snap())
        self.assertEqual(t.update(done(), head(), snap()).score, 0.0)
        t = Tracker(HZ)
        t.update(done(), head(tilt=9.5), snap())
        self.assertEqual(t.update(done(), head(), snap()).score, 1.0)

    def test_a_block_resting_on_the_head(self):
        on = block(L.HEAD_X, 0.0, lift=L.HEAD_H)
        self.assertEqual(score_of(blocks(**{BL: on, BR: beside(-1, 0.002)})), 0.0)

    def test_a_block_partly_over_the_head(self):
        over = block(L.HEAD_X, R + HW - 0.006, lift=L.HEAD_H)  # 6 mm inside the outline, on the head's rim
        self.assertEqual(score_of(blocks(**{BL: over, BR: beside(-1, 0.002)})), 0.0)

    def test_a_block_held_in_the_air_over_the_head(self):
        s = block(L.HEAD_X, 0.0, lift=0.15)
        arms = snap(left_tcp=grip_tcp(s), left_open=GRIP)
        self.assertEqual(score_of(blocks(**{BL: s, BR: beside(-1, 0.002)}), arms=arms), 0.0)

    def test_up_to_5_mm_inside_the_outline_is_touching(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, -0.005), BR: beside(-1, 0.002)})), 1.0)
        self.assertEqual(score_of(blocks(**{BL: beside(1, -0.0051), BR: beside(-1, 0.002)})), 0.0)


class Exploits(unittest.TestCase):
    def test_both_blocks_on_one_side_count_once(self):
        b = blocks(**{BL: beside(1, 0.002, x=0.185), BR: beside(1, 0.002, x=0.220)})
        self.assertEqual(score_of(b), 0.5)

    def test_a_block_lying_flat_beside_the_head(self):
        s = block(L.HEAD_X, R + 0.002 + L.BLOCK_SIZE[2] / 2, flat=True)
        self.assertAlmostEqual(gap(s, head()), 0.002, places=9)
        self.assertEqual(score_of(blocks(**{BL: s})), 0.0)

    def test_a_block_turned_with_its_narrow_side_to_the_head(self):
        s = block(L.HEAD_X, R + 0.002 + L.BLOCK_HALF[0], turned=True)
        self.assertAlmostEqual(gap(s, head()), 0.002, places=9)
        self.assertEqual(score_of(blocks(**{BL: s})), 0.0)

    def test_a_block_beside_the_neck_or_crown_is_not_beside_the_head(self):
        self.assertEqual(score_of(blocks(**{BL: block(L.HEAD_X + 0.06, R + 0.002 + HW)})), 0.0)
        self.assertEqual(score_of(blocks(**{BR: block(L.HEAD_X - 0.05, -(R + 0.002 + HW))})), 0.0)

    def test_a_block_on_the_table_beside_the_board(self):
        s = block(L.HEAD_X, L.BOARD_HALF_W + HW + 0.002, on_table=True)
        self.assertEqual(score_of(blocks(**{BL: s})), 0.0)

    def test_pushing_the_head_into_a_block(self):
        # the block is put down 20 mm out and the head is shoved over to it: the head moved, 0
        s = beside(1, 0.020)
        h = head(y=0.020)
        self.assertAlmostEqual(gap(s, h), 0.0, places=9)
        self.assertEqual(trial((blocks(**{BL: s, BR: beside(-1, 0.002, h=h)}), h)), 0.0)

    def test_holding_both_blocks_in_place_never_letting_go(self):
        b = done()
        arms = snap(left_tcp=grip_tcp(b[BL]), left_open=GRIP, right_tcp=grip_tcp(b[BR]), right_open=GRIP)
        self.assertEqual(score_of(b, arms=arms), 0.0)

    def test_a_block_stacked_on_the_other(self):
        lower = beside(1, 0.002)
        upper = block(L.HEAD_X, float(lower.pos[1]), lift=L.BLOCK_SIZE[2])
        self.assertEqual(score_of({BL: lower, BR: upper}), 0.5)

    def test_squeezing_the_head_without_moving_it_is_snug(self):
        self.assertEqual(score_of(blocks(**{BL: beside(1, -0.001), BR: beside(-1, -0.001)})), 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker(HZ)
        t.update(done(), head(x=L.HEAD_X + 0.003), snap())
        saved = (t.worst_shift, t.worst_lift, t.worst_tilt, t.head_start, t.first_full_step)
        good = done()
        bad_states = []
        for key, val in (("pos", [np.nan, 0, 0]), ("quat_xyzw", [0, 0, 0, 0]), ("lin_vel", [np.inf, 0, 0]),
                         ("ang_vel", [0, np.nan, 0]), ("low", [np.nan] * 3), ("high", [np.inf] * 3)):
            s = SimpleNamespace(**vars(good[BL]))
            setattr(s, key, np.array(val, float))
            bad_states.append(({**good, BL: s}, head()))
            hs = SimpleNamespace(**vars(head()))
            setattr(hs, key, np.array(val, float))
            bad_states.append((good, hs))
        missing = SimpleNamespace(**{k: v for k, v in vars(good[BL]).items() if k != "low"})
        bad_states.append(({**good, BL: missing}, head()))
        for b, h in bad_states:
            r = t.update(b, h, snap())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual((t.worst_shift, t.worst_lift, t.worst_tilt, t.head_start, t.first_full_step), saved)
        self.assertEqual(t.bad_steps, len(bad_states))
        self.assertEqual(t.update(done(), head(x=L.HEAD_X + 0.003), snap()).score, 1.0)
        self.assertFalse(valid(SimpleNamespace(pos=[0, 0], quat_xyzw=[0, 0, 0, 1], lin_vel=[0] * 3,
                                               ang_vel=[0] * 3, low=[0] * 3, high=[0] * 3)))

    def test_a_missing_block_or_head_scores_zero(self):
        t = Tracker(HZ)
        self.assertEqual(t.update({BL: beside(1, 0.002)}, head(), snap()).score, 0.0)
        self.assertEqual(t.update(done(), None, snap()).score, 0.0)
        self.assertEqual(t.update(None, head(), snap()).score, 0.0)
        self.assertEqual(t.bad_steps, 3)

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(7)
        t = Tracker(HZ)
        for _ in range(300):
            def rq():
                v = np.array([rng.gauss(0, 1) for _ in range(4)])
                return v / np.linalg.norm(v)
            b = {n: block_state((rng.uniform(0, 0.6), rng.uniform(-0.4, 0.4), rng.uniform(0.70, 1.0)), rq())
                 for n in NAMES}
            h = head(x=rng.uniform(0.1, 0.3), y=rng.uniform(-0.1, 0.1), tilt=rng.uniform(-90, 90))
            r = t.update(b, h, snap(left_open=rng.random(), right_open=rng.random()))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_arms_unavailable_taken_as_let_go_and_counted(self):
        t = Tracker(HZ)
        r = t.update(done(), head(), None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertFalse(r.details["arms_available"])
        bad = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3), opening=0.5),
               "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3), opening=0.5)}
        t.update(done(), head(), bad)
        self.assertEqual(t.arms_unavailable, 2)
        self.assertTrue(released(None, done()[BL]))

    def test_offline_feed_reaches_the_scorer(self):
        b = done()
        arms = snap(left_tcp=grip_tcp(b[BL]), left_open=GRIP)
        self.assertEqual(rigstate.arms()["left"].opening, GRIP)
        self.assertEqual(Tracker(HZ).update(b, head(), arms).score, 0.5)


class Reset(unittest.TestCase):
    def played(self):
        t = Tracker(HZ)
        t.update(blocks(), head(), snap())
        t.update(done(), head(y=0.010), snap())  # the head shoved 10 mm for a step
        t.update(done(), head(), snap())
        self.assertEqual(t.first_disturbed_step, 2)
        self.assertGreater(t.worst_shift, 0.009)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.played()
        r = t.update(blocks(), head(), snap())  # everything back on its mark in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.worst_shift, 0.0)
        self.assertIsNone(t.first_disturbed_step)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.update(done(), head(), snap()).score, 1.0)

    def test_teleport_to_a_jittered_start_resets_and_measures_from_there(self):
        t = self.played()
        b = {n: block(L.BLOCKS[n][0] + 0.02, L.BLOCKS[n][1] - 0.01, on_table=True) for n in NAMES}
        h = head(x=L.HEAD_X + 0.02, y=-0.01)
        t.update(b, h, snap())
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.head_start, (L.HEAD_X + 0.02, -0.01))
        self.assertEqual(t.update(done(h), h, snap()).score, 1.0)  # the head has not moved from where it began

    def test_gradual_return_does_not_reset(self):
        t = self.played()
        for k in range(1, 41):
            f = k / 40
            b = blocks(**{BL: block(L.HEAD_X, R + 0.002 + HW + f * (L.BLOCK_Y - R - 0.002 - HW),
                                    lift=-f * L.BOARD_T),
                          BR: block(L.HEAD_X, -(R + 0.002 + HW + f * (L.BLOCK_Y - R - 0.002 - HW)),
                                    lift=-f * L.BOARD_T)})
            t.update(b, head(), snap())
        self.assertTrue(Tracker.at_start(b, head()))
        self.assertEqual(t.first_disturbed_step, 2)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.played()
        t.update(blocks(**{BL: block(0.6, 0.4, on_table=True)}), head(), snap())
        self.assertEqual(t.first_disturbed_step, 2)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.played()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual(t.steps, 0)
            self.assertIsNone(t.head_start)
            self.assertEqual(t.worst_shift, 0.0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)

    def test_evaluate_without_history(self):
        score, ev = evaluate(done(), head(), snap())
        self.assertEqual(score, 1.0)
        score, ev = evaluate(done(), head(), snap(), worst=(0.009, 0.0, 0.0))
        self.assertEqual(score, 0.0)
        self.assertTrue(ev["head_disturbed"])


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objs(self, b, h):
        return {**b, HEAD: h}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = {o.name: o for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(sorted(loose), sorted([*NAMES, HEAD]))
        self.assertTrue(all(o.mass <= 0.30 for o in loose.values()))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        # the spawned head and blocks are the starting state the scorer expects
        b = {o.name: block_state(o.pos, about("z", math.degrees(o.yaw))) for o in self.task.BLOCKS}
        hp = self.task.HEAD_PART
        self.assertEqual((hp.radius, hp.height), (R, L.HEAD_H))
        self.assertTrue(Tracker.at_start(b, head(x=hp.pos[0], y=hp.pos[1])))
        self.assertAlmostEqual(hp.pos[2], L.HEAD_Z)
        # the head stands free on the board, 10 mm short of the neck; the blocks stand on the table off the board
        self.assertAlmostEqual(self.task.NECK.pos[0] - self.task.NECK.size[0] / 2 - (L.HEAD_X + R), 0.010)
        for o in self.task.BLOCKS:
            self.assertGreater(abs(o.pos[1]) - L.BLOCK_HALF[1], L.BOARD_HALF_W)
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)
        self.assertGreater(L.BOARD_HALF_W, R + 0.002 + 2 * HW)  # a placed block stands wholly on the board

    def test_instruction_matches_task_json(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertFalse(self.task.success(self.objs(blocks(), head())))  # arms None after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertTrue(self.task.success(self.objs(done(), head())))
        self.assertEqual(self.task.score(self.objs(done(), head())), 1.0)
        self.assertEqual(self.task.score(self.objs(done(), head(y=0.009))), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertIsNone(tr.head_start)

    def test_trace_rows(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                rigstate.offline_reset()
                snap()
                self.task.success(self.objs(done(), head()))
                rigstate.offline_reset()
                self.task.success(self.objs(done(), head()))  # no arms
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        r = rows[0]
        for key in ("episode", "step", "score", "success", "left_side", "right_side", "head_disturbed",
                    "over_head", "resting", "head_shift_mm", "head_lift_mm", "head_tilt_deg", "worst_shift_mm",
                    "worst_lift_mm", "worst_tilt_deg", "blocks", "opening", "arms_unavailable", "first_full_step",
                    "first_disturbed_step", "best_score", "pos", "left_tcp", "right_tcp"):
            self.assertIn(key, r)
        blk = r["blocks"][BL]
        for key in ("side", "credit", "snug", "beside", "let_go", "gap_mm", "tilt_deg", "heading_deg", "rise_mm",
                    "dx_mm"):
            self.assertIn(key, blk)
        self.assertEqual(blk["gap_mm"], 2.0)
        self.assertIn(HEAD, r["pos"])
        self.assertIn("rigstate_error", rows[1])

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objs(blocks(), head())))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        task = self.task
        plan = task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        grip_up = L.BLOCK_SIZE[2] - 0.025 + TIPS
        for arm, sign in (("left", 1), ("right", -1)):
            wps = plan[arm]
            # gripped across its 70 mm length (fingers along x), 10 mm off its centre away from the head
            grip = [w for w in wps if w.label.startswith("grip the block")][0]
            self.assertAlmostEqual(abs(grip.yaw), math.pi / 2)
            self.assertAlmostEqual(grip.finger, -(L.BLOCK_SIZE[0] - 0.008) / 2)
            self.assertAlmostEqual(grip.pos[1], sign * (L.BLOCK_Y + 0.010))
            self.assertAlmostEqual(grip.pos[2] - grip_up, L.TOP)
            # carried in with its bottom 8 mm above the board, set down 2 mm from the head, bottom 1 mm up
            carry = [w for w in wps if w.label.startswith("carry it in")][0]
            self.assertAlmostEqual(carry.pos[2] - grip_up, L.BOARD_TOP + 0.008)
            self.assertAlmostEqual(abs(carry.pos[1]) - 0.010 - HW - R, 0.012)
            down = [w for w in wps if w.label.startswith("set it down")][0]
            self.assertAlmostEqual(abs(down.pos[1]) - 0.010 - HW - R, 0.002)
            self.assertAlmostEqual(down.pos[2] - grip_up, L.BOARD_TOP + 0.001)
            self.assertAlmostEqual(down.pos[0], L.HEAD_X)
            for w in wps:
                self.assertAlmostEqual(w.yaw, task.YAW[arm])
                self.assertGreaterEqual(sign * w.pos[1], R + 0.002 + HW)  # the TCP never over the head's side
        # the two grippers are never within 0.20 m of each other in y while both are below 12 cm
        times = sorted({round(sum(w.duration for w in plan[a][:k + 1]), 6) for a in plan
                        for k in range(len(plan[a]))})

        def at(arm, t):
            acc, prev = 0.0, None
            for w in plan[arm]:
                if acc + w.duration >= t - 1e-9:
                    return w.pos if prev is None else tuple(
                        p + (q - p) * min(1.0, (t - acc) / w.duration) for p, q in zip(prev, w.pos))
                acc += w.duration
                prev = w.pos
            return plan[arm][-1].pos

        for k in range(int(max(times) * HZ) + 1):
            t = k / HZ
            lp, rp = at("left", t), at("right", t)
            if lp[2] < L.TOP + 0.12 and rp[2] < L.TOP + 0.12:
                self.assertGreaterEqual(lp[1] - rp[1], 0.20, t)
        # the left gripper is up and moving away before the right block comes in
        t_left_up = sum(w.duration for w in plan["left"][:-1])
        i = [k for k, w in enumerate(plan["right"]) if w.label.startswith("carry it in")][0]
        self.assertLessEqual(t_left_up, sum(w.duration for w in plan["right"][:i]))
        self.assertLessEqual(sum(w.duration for w in plan["right"]) * HZ, 1200 - 100)
        # every close and open takes 1.5 s, and so do the slow descents onto a block and onto the board
        for a in plan:
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger or (prev.pos[2] > w.pos[2] and w.pos[2] < L.TOP + 0.09):
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
