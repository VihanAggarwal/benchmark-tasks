"""Offline tests for scoring.py, standin.py and the hooks in task.py: synthetic foot and block states and synthetic
arm snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import re
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
import standin  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, arm_clearance, block_front, block_rel_heel, block_tilt_deg, foot_tilt_deg, heel_toe, point_box_distance,
    rotation, sole_corners, toe_out_deg,
)

HZ = 20.0
DT = 1.0 / HZ
WINDOW = 11  # samples in 0.5 s
TIPS = L.PAD_TIPS
FAR_L = np.array([0.250, 0.300, L.TOP + 0.150])  # the demo's parking spots
FAR_R = np.array([0.250, -0.300, L.TOP + 0.150])
PLATE = L.PLATE_TOP
HB = L.BLOCK_SIZE[2]


def quat(heading=0.0, pitch=0.0, roll=0.0):
    """(x, y, z, w) of Rz(heading) Ry(pitch) Rx(roll), degrees; +pitch turns the toe down."""
    h, p, r = (math.radians(v) for v in (heading, pitch, roll))
    rz = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return standin._quat(rz @ ry @ rx)


def foot(heel=(0.1904, 0.0, PLATE), heading=-5.0, pitch=0.0, roll=0.0, vel=(0, 0, 0), source="sim"):
    """A foot state by its heel point H (= the base frame's origin) and its turn."""
    return SimpleNamespace(origin=np.array(heel, float), quat_xyzw=quat(heading, pitch, roll),
                           lin_vel=np.array(vel, float), ang_vel=np.zeros(3), source=source)


def block(centre=(0.220, 0.0), z=None, heading=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    c = np.array([centre[0], centre[1], PLATE + HB / 2 if z is None else z], float)
    return SimpleNamespace(pos=c, quat_xyzw=quat(heading, pitch, roll), lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3))


def pitched(mid_ap=0.030, toe_out=5.0, heel_y=0.0, b1=0.0598):
    """The foot resting heel on the block's front edge (b1 along the foot from H) and toe on the plate, its middle
    mid_ap in front of the laser line, its heel point heel_y off the centre line."""
    th = math.asin(HB / (L.FOOT_LEN - b1))
    half = L.FOOT_LEN / 2 * math.cos(th)
    a = math.radians(toe_out)
    hx = L.LASER_X + mid_ap - half * math.cos(a)
    return foot((hx, heel_y, PLATE + HB + b1 * math.sin(th)), -toe_out, math.degrees(th))


def flat(mid_ap=0.030, toe_out=5.0, heel_y=0.0, z=PLATE):
    a = math.radians(toe_out)
    hx = L.LASER_X + mid_ap - L.FOOT_LEN / 2 * math.cos(a)
    return foot((hx, heel_y, z), -toe_out, 0.0)


def on_toe(heel_h, toe_t=0.0, mid_ap=0.030, toe_out=5.0):
    """The foot pitched with its heel point heel_h and its toe point toe_t above the plate, middle mid_ap in front
    of the laser line (whatever holds it there: the scorer reads only the pose)."""
    p = math.asin((heel_h - toe_t) / L.FOOT_LEN)
    a = math.radians(toe_out)
    hx = L.LASER_X + mid_ap - L.FOOT_LEN / 2 * math.cos(p) * math.cos(a)
    return foot((hx, 0.0, PLATE + heel_h), -toe_out, math.degrees(p))


FOOT_START = foot((*L.FOOT_ORIGIN[:2], L.TOP), 0.0)
BLOCK_START = block(L.BLOCK_START, z=L.TOP + HB / 2)
GOAL_FOOT, GOAL_BLOCK = pitched(), block()
NO_BLOCK = block((0.120, 0.300), z=L.TOP + HB / 2)  # left on the table, off the plate


def snap(left_tcp=FAR_L, right_tcp=FAR_R, left_open=1.0, right_open=1.0):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def run(f, b, arms="far", n=WINDOW, tracker=None):
    t = tracker or Tracker(HZ)
    a = snap() if arms == "far" else arms
    r = None
    for _ in range(n):
        r = t.update(f, b, a)
    return r, t


def score_of(f, b, arms="far"):
    return run(f, b, arms)[0].score


def adapter_top_point(f):
    o, r = np.asarray(f.origin), rotation(f.quat_xyzw)
    return o + r @ np.array([L.ADAPTER_CENTER[0], 0.0, L.FOOT_TOP])


def along_foot(f, along, side=0.0):
    """The point (x, y) seen from above `along` in front of H along H -> T and `side` to its left."""
    h, t = heel_toe(f)
    u = (t[:2] - h[:2]) / np.linalg.norm(t[:2] - h[:2])
    return tuple(h[:2] + u * along + np.array([-u[1], u[0]]) * side)


def rest(f, b):
    """The foot as standin.py lets it come to rest on the block and the plate."""
    o, q = standin.rest_pose(f.origin, f.quat_xyzw, b)
    return SimpleNamespace(origin=o, quat_xyzw=q, lin_vel=np.zeros(3))


def finger_on_forefoot(f):
    """A wide-open right gripper beside the foot (exploit E2): its pad-tip point 40 mm outside the shell's right side
    face, level with the forefoot's top, so one finger (47 to 57 mm from the TCP) lies 18 to 28 mm inside the
    shell's outline, on the forefoot. Only the TCP and the pad-tip point are read."""
    o, r = np.asarray(f.origin), rotation(f.quat_xyzw)
    tips = o + r @ np.array([0.150, -(L.FOOT_W / 2 + 0.040), L.FORE_SIZE[2]])
    return snap(right_tcp=tips + np.array([0.0, 0.0, TIPS]), right_open=1.0)


class Geometry(unittest.TestCase):
    def test_heel_toe_and_corners(self):
        f = foot((0.2, 0.0, PLATE), 0.0)
        h, t = heel_toe(f)
        np.testing.assert_allclose(h, (0.2, 0.0, PLATE))
        np.testing.assert_allclose(t, (0.42, 0.0, PLATE))
        c = sole_corners(f)
        np.testing.assert_allclose(sorted(c[:, 1]), [-0.035, -0.035, 0.035, 0.035])

    def test_toe_out_sign(self):
        self.assertAlmostEqual(toe_out_deg(foot(heading=-5.0)), 5.0, places=6)  # toward -y: out, for a right foot
        self.assertAlmostEqual(toe_out_deg(foot(heading=+5.0)), -5.0, places=6)

    def test_foot_tilt_not_folded_block_tilt_folded(self):
        self.assertAlmostEqual(foot_tilt_deg(foot(roll=180.0)), 180.0, places=4)
        self.assertAlmostEqual(block_tilt_deg(block(roll=180.0)), 0.0, places=4)
        self.assertAlmostEqual(block_tilt_deg(block(roll=90.0)), 90.0, places=4)

    def test_point_box_distance(self):
        r = np.eye(3)
        self.assertEqual(point_box_distance((0, 0, 0), (0, 0, 0), np.array([0.01, 0.01, 0.01]), r), 0.0)
        self.assertAlmostEqual(point_box_distance((0, 0, 0.03), (0, 0, 0), np.array([0.01, 0.01, 0.01]), r), 0.02)

    def test_goal_pose_matches_the_demo_numbers(self):
        h, t = heel_toe(GOAL_FOOT)
        self.assertAlmostEqual((h[0] + t[0]) / 2 - L.LASER_X, 0.030, places=6)
        self.assertAlmostEqual((h[2] - PLATE) * 1000, 27.5, delta=0.2)
        self.assertAlmostEqual(t[2], PLATE, places=6)
        self.assertAlmostEqual(foot_tilt_deg(GOAL_FOOT), 7.17, delta=0.05)
        along, side = block_rel_heel(GOAL_FOOT, GOAL_BLOCK)
        self.assertTrue(along >= -0.030 and abs(side) <= 0.035)
        self.assertLess(block_front(GOAL_FOOT, GOAL_BLOCK), 0.065)  # its front corner 25 mm inside the 90 mm limit


class StandIn(unittest.TestCase):
    """The offline resting model (standin.py)."""

    def rest(self, f, b):
        o, q = standin.rest_pose(f.origin, f.quat_xyzw, b)
        return SimpleNamespace(origin=o, quat_xyzw=q, lin_vel=np.zeros(3))

    def test_carried_pose_is_left_alone(self):
        f = flat(z=PLATE + 0.025)
        o, q = standin.rest_pose(f.origin, f.quat_xyzw, GOAL_BLOCK)
        np.testing.assert_allclose(o, f.origin)
        np.testing.assert_allclose(q, f.quat_xyzw)

    def test_heel_on_the_block_tips_toe_down(self):
        r = self.rest(flat(z=PLATE), GOAL_BLOCK)  # dropped flat onto the plate (the checker's column)
        h, t = heel_toe(r)
        self.assertAlmostEqual(foot_tilt_deg(r), 7.2, delta=0.2)
        self.assertAlmostEqual(t[2], PLATE, delta=1e-6)
        self.assertTrue(0.027 < h[2] - PLATE < 0.029)
        self.assertAlmostEqual(toe_out_deg(r), 5.0, places=4)
        self.assertEqual(score_of(r, GOAL_BLOCK), 1.0)

    def test_no_block_lies_flat(self):
        r = self.rest(flat(z=PLATE), NO_BLOCK)
        self.assertAlmostEqual(foot_tilt_deg(r), 0.0, places=6)
        self.assertEqual(score_of(r, NO_BLOCK), 0.75)

    def test_block_under_the_toe_tips_heel_down(self):
        b = block((0.370, -0.015))
        r = self.rest(flat(z=PLATE), b)
        h, t = heel_toe(r)
        self.assertAlmostEqual(h[2], PLATE, delta=1e-6)
        self.assertGreater(t[2] - PLATE, 0.020)

    def test_block_under_the_middle_lies_flat_on_it(self):
        b = block((0.290, -0.008))
        r = self.rest(flat(z=PLATE), b)
        h, t = heel_toe(r)
        self.assertAlmostEqual(h[2] - PLATE, HB, delta=1e-6)
        self.assertAlmostEqual(t[2] - PLATE, HB, delta=1e-6)

    def test_at_the_start_on_the_table(self):
        o, q = standin.rest_pose(FOOT_START.origin, FOOT_START.quat_xyzw, BLOCK_START)
        np.testing.assert_allclose(o, FOOT_START.origin, atol=1e-9)


class Tiers(unittest.TestCase):
    def test_goal_scores_one_and_succeeds(self):
        r, _ = run(GOAL_FOOT, GOAL_BLOCK)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_every_tier_is_reachable(self):
        cases = {
            1.0: (GOAL_FOOT, GOAL_BLOCK),
            0.75: (flat(), NO_BLOCK),  # no block: A and B
            0.5: (flat(toe_out=0.0), NO_BLOCK),  # A only
            0.25: (flat(mid_ap=0.045, toe_out=0.0), NO_BLOCK),  # the gate only
            0.0: (FOOT_START, BLOCK_START),  # nothing done
        }
        for want, (f, b) in cases.items():
            with self.subTest(want=want):
                self.assertEqual(score_of(f, b), want)

    def test_each_check_alone(self):
        self.assertEqual(score_of(pitched(mid_ap=0.045, toe_out=0.0), GOAL_BLOCK), 0.5)  # C only
        self.assertEqual(score_of(pitched(toe_out=0.0), GOAL_BLOCK), 0.75)  # A and C
        self.assertEqual(score_of(pitched(mid_ap=0.045), GOAL_BLOCK), 0.75)  # B and C
        self.assertEqual(score_of(flat(mid_ap=0.045), NO_BLOCK), 0.5)  # B only

    def test_forward_band(self):
        for ap, ok in ((0.0249, False), (0.0251, True), (0.0349, True), (0.0351, False), (0.010, False)):
            with self.subTest(ap=ap):
                self.assertEqual(score_of(pitched(mid_ap=ap), GOAL_BLOCK), 1.0 if ok else 0.75)

    def test_toe_out_band(self):
        for deg, ok in ((1.9, False), (2.1, True), (7.9, True), (8.1, False), (-5.0, False), (0.0, False)):
            with self.subTest(deg=deg):
                self.assertEqual(score_of(pitched(toe_out=deg), GOAL_BLOCK), 1.0 if ok else 0.75)

    def test_toe_out_ruler_equivalent(self):
        lo, hi = L.TOE_OUT_RULER
        self.assertAlmostEqual(lo * 218, 7.6, delta=0.1)
        self.assertAlmostEqual(hi * 218, 30.6, delta=0.1)

    def test_block_flat_on_the_plate(self):
        for tilt, ok in ((9.5, True), (10.5, False)):
            with self.subTest(tilt=tilt):
                b = block(pitch=tilt, z=PLATE + 0.5 * (HB * math.cos(math.radians(tilt))
                                                      + L.BLOCK_SIZE[0] * math.sin(math.radians(tilt))))
                self.assertEqual(score_of(GOAL_FOOT, b), 1.0 if ok else 0.75)
        for lift, ok in ((0.0029, True), (0.0031, False), (-0.0029, True)):
            with self.subTest(lift=lift):
                self.assertEqual(score_of(GOAL_FOOT, block(z=PLATE + HB / 2 + lift)), 1.0 if ok else 0.75)

    def test_block_under_the_heel(self):
        # the block's centre within 35 mm sideways and no more than 30 mm behind H; every corner at most 90 mm in
        # front of H. Square to the foot (turned -5 deg with it) the front corners are 30 mm ahead of the centre ...
        for along, side, ok in ((0.0595, 0.0, True), (0.0605, 0.0, False), (-0.0295, 0.0, True),
                                (-0.0305, 0.0, False), (0.028, 0.0345, True), (0.028, 0.0355, False),
                                (0.028, -0.0355, False)):
            with self.subTest(along=along, side=side):
                b = block(along_foot(GOAL_FOOT, along, side), heading=-5.0)
                self.assertEqual(score_of(GOAL_FOOT, b), 1.0 if ok else 0.75)
        # ... set square to the plate as the demo sets it, 32.9 mm; turned 90 deg, 35 mm; turned 45 deg, 46.0 mm
        for heading, reach in ((0.0, 0.0329), (85.0, 0.035), (40.0, 0.0460)):
            for front, ok in ((0.0895, True), (0.0905, False)):
                with self.subTest(heading=heading, front=front):
                    b = block(along_foot(GOAL_FOOT, front - reach), heading=heading)
                    self.assertAlmostEqual(block_front(GOAL_FOOT, b), front, delta=0.0001)
                    self.assertEqual(score_of(GOAL_FOOT, b), 1.0 if ok else 0.75)

    def test_heel_and_toe_heights(self):
        for heel_h, ok in ((0.0175, False), (0.0185, True), (0.0395, True), (0.0405, False)):
            with self.subTest(heel_h=heel_h):
                self.assertEqual(score_of(on_toe(heel_h), GOAL_BLOCK), 1.0 if ok else 0.75)
        for toe_t, ok in ((0.0039, True), (0.0041, False), (-0.0039, True), (-0.0041, False)):
            with self.subTest(toe_t=toe_t):
                f = on_toe(0.0275, toe_t)
                self.assertAlmostEqual(heel_toe(f)[1][2] - PLATE, toe_t, places=9)
                self.assertEqual(score_of(f, GOAL_BLOCK), 1.0 if ok else 0.75)


class Gate(unittest.TestCase):
    """Every zero condition, at its boundary."""

    def test_sole_over_the_plate(self):
        # the toe corners just inside / just past the plate's front edge (x 0.450); square, so A fails: 0.25
        for corner_x, ok in ((0.4495, True), (0.4505, False)):
            with self.subTest(corner_x=corner_x):
                f = foot((corner_x - L.FOOT_LEN, 0.0, PLATE), 0.0)
                self.assertAlmostEqual(max(sole_corners(f)[:, 0]), corner_x, places=9)
                self.assertEqual(score_of(f, NO_BLOCK), 0.25 if ok else 0.0)
        # turned 25 deg: a toe corner passes the plate's side at y +0.100 (and G3, 20 deg, fails as well)
        f = foot((0.20, 0.0, PLATE), 25.0)
        self.assertGreater(max(sole_corners(f)[:, 1]), L.PLATE_HALF_Y)
        self.assertEqual(score_of(f, NO_BLOCK), 0.0)
        # G1's sideways limit on its own (E3): H 15 mm off the line and turned 15 deg the same way, so G3 and G4
        # hold, but a toe corner is 5.7 mm past the plate's side
        for heel_y, heading in ((0.015, 15.0), (-0.015, -15.0)):
            with self.subTest(heel_y=heel_y, heading=heading):
                f = foot((0.20, heel_y, PLATE), heading)
                self.assertAlmostEqual(np.abs(sole_corners(f)[:, 1]).max(), 0.1057, delta=0.0002)
                r, _ = run(f, NO_BLOCK)
                d = r.details
                self.assertEqual((d["g1_sole_over_plate"], d["g3_toes_forward"], d["g4_heel_on_line"]),
                                 (False, True, True))
                self.assertEqual(r.score, 0.0)

    def test_sole_down(self):
        for tilt, ok in ((14.5, True), (15.5, False)):
            with self.subTest(tilt=tilt):
                f = foot((0.20, 0.0, PLATE), -5.0, roll=tilt)
                self.assertEqual(score_of(f, NO_BLOCK) > 0, ok)
        self.assertEqual(score_of(foot((0.20, 0.0, PLATE + 0.058), -5.0, roll=180.0), NO_BLOCK), 0.0)  # upside down

    def test_toes_forward(self):
        for heading, ok in ((-19.5, True), (-20.5, False), (19.5, True), (20.5, False)):
            with self.subTest(heading=heading):
                f = foot((0.200, -0.012 if heading > 0 else 0.012, PLATE), heading)
                self.assertTrue(all(abs(c[1]) <= L.PLATE_HALF_Y for c in sole_corners(f)))  # G1 still met
                self.assertEqual(score_of(f, NO_BLOCK) > 0, ok)
        # 30 deg could never decide: turned that far with H on the line, a toe corner is off the plate
        f = foot((0.200, -0.015, PLATE), 22.5)
        self.assertGreater(max(sole_corners(f)[:, 1]), L.PLATE_HALF_Y)
        self.assertEqual(score_of(foot((0.420, 0.0, PLATE), 180.0), NO_BLOCK), 0.0)  # backwards

    def test_heel_on_the_centre_line(self):
        for y, ok in ((0.0145, True), (0.0155, False), (-0.0145, True), (-0.0155, False)):
            with self.subTest(y=y):
                self.assertEqual(score_of(pitched(heel_y=y), block((0.220, y))), 1.0 if ok else 0.0)

    def test_released(self):
        top = adapter_top_point(GOAL_FOOT)
        for gap, ok in ((0.0195, False), (0.0205, True)):
            with self.subTest(gap=gap):
                tcp = top + np.array([0, 0, gap + TIPS])  # the pad-tip point is the lower one
                self.assertEqual(score_of(GOAL_FOOT, GOAL_BLOCK, snap(right_tcp=tcp)), 1.0 if ok else 0.0)
        b = GOAL_BLOCK.pos + np.array([0.0, 0.0, HB / 2 + 0.019 + TIPS])
        self.assertEqual(score_of(GOAL_FOOT, GOAL_BLOCK, snap(left_tcp=b)), 0.0)  # left gripper on the block
        self.assertEqual(score_of(GOAL_FOOT, GOAL_BLOCK, None), 0.0)  # arms unreadable: not let go

    def test_opened_but_not_moved_clear(self):
        # E1: the gripper opened fully (94 mm) round the 46 mm adapter but not lifted away has let go of the foot but
        # not moved clear, which the instruction asks for: 0 until both points are 20 mm clear
        spec = json.loads((HERE / "task.json").read_text())
        self.assertTrue(spec["instruction"].endswith("Then let go and move both grippers clear of the foot and the "
                                                     "block."))
        top = adapter_top_point(GOAL_FOOT)
        r, _ = run(GOAL_FOOT, GOAL_BLOCK, snap(right_tcp=top - np.array([0, 0, 0.009]), right_open=1.0), n=12)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["g5_released"])
        self.assertTrue(r.details["a_forward"] and r.details["b_toe_out"] and r.details["c_heel_on_block"])
        for gap, want in ((0.0195, 0.0), (0.0205, 1.0)):  # opened and lifted: the pad tips this far above the adapter
            with self.subTest(gap=gap):
                a = snap(right_tcp=top + np.array([0, 0, gap + TIPS]), right_open=1.0)
                self.assertEqual(run(GOAL_FOOT, GOAL_BLOCK, a, n=12)[0].score, want)

    def test_still(self):
        t = Tracker(HZ)
        a = snap()
        for i in range(WINDOW - 1):  # less than 0.5 s seen: not still yet
            r = t.update(GOAL_FOOT, GOAL_BLOCK, a)
        self.assertEqual(r.score, 0.0)
        r = t.update(GOAL_FOOT, GOAL_BLOCK, a)
        self.assertEqual(r.score, 1.0)
        # the foot creeping sideways 0.21 mm a step: 2.1 mm over 0.5 s; 0.19 mm a step: 1.9 mm
        for creep, ok in ((0.00021, False), (0.00019, True)):
            with self.subTest(creep=creep):
                t = Tracker(HZ)
                for i in range(30):
                    f = pitched()
                    f.origin = f.origin + np.array([0, creep * i, 0])
                    r = t.update(f, GOAL_BLOCK, a)
                self.assertEqual(r.score, 1.0 if ok else 0.0)
                self.assertEqual(r.details["g6_still"], ok)
        # the block sliding 2.5 mm over 0.5 s
        t = Tracker(HZ)
        for i in range(30):
            r = t.update(GOAL_FOOT, block((0.220, 0.00025 * i)), a)
        self.assertEqual(r.score, 0.0)

    def test_success_holds_only_when_released_and_still(self):
        t = Tracker(HZ)
        held = adapter_top_point(GOAL_FOOT) - np.array([0, 0, 0.009])
        for _ in range(40):
            r = t.update(GOAL_FOOT, GOAL_BLOCK, snap(right_tcp=held, right_open=0.4))
        self.assertFalse(r.success)
        r = t.update(GOAL_FOOT, GOAL_BLOCK, snap())
        self.assertTrue(r.success)


class Exploits(unittest.TestCase):
    """Unintended strategies earn nothing, or only what the text says."""

    def test_holding_the_foot_in_place_instead_of_letting_go(self):
        held = adapter_top_point(GOAL_FOOT) - np.array([0, 0, 0.009])
        self.assertEqual(score_of(GOAL_FOOT, GOAL_BLOCK, snap(right_tcp=held, right_open=0.40)), 0.0)

    def test_finger_resting_on_the_foot_is_not_seen(self):
        # E2: G5 reads the TCP and the pad-tip point only. A wide-open finger lying on the forefoot of a foot that
        # rests where it is by itself (heel on the block, toe on the plate) changes nothing: 1.0, as task.json's
        # fidelity note says
        a = finger_on_forefoot(GOAL_FOOT)
        self.assertAlmostEqual(arm_clearance(a["right"], GOAL_FOOT, GOAL_BLOCK), 0.040, delta=0.0005)
        self.assertEqual(score_of(GOAL_FOOT, GOAL_BLOCK, a), 1.0)

    def test_finger_cannot_pin_a_foot_balanced_on_a_turned_block(self):
        # E2's harmful case, closed by the 90 mm corner limit: a block turned on the plate whose front reaches past
        # the centre of mass (93.9 mm from H) leaves the foot lying flat on it, toe in the air; a finger pinning the
        # toe down to the plate would make it look aligned. The old window (block centre at most 60 mm in front of
        # H) let that score 1.0; now the corners are over 90 mm in front of H: no (C), 0.75.
        f = pitched(b1=0.095)  # pivoted over the block's front edge, past the centre of mass: it needs a hold
        for heading, along in ((85.0, 0.0595), (40.0, 0.050)):
            with self.subTest(heading=heading):
                b = block(along_foot(f, along), heading=heading)
                self.assertLessEqual(block_rel_heel(f, b)[0], 0.060)
                self.assertGreater(block_front(f, b), 0.0939)
                r, _ = run(f, b, finger_on_forefoot(f))
                self.assertTrue(r.details["gate"])
                self.assertFalse(r.details["c_heel_on_block"])
                self.assertEqual(r.score, 0.75)
                self.assertEqual(score_of(rest(flat(), b), b), 0.75)  # let alone it lies flat on the block: no (C)

    def test_block_front_past_90_mm_gets_no_c_even_if_the_foot_balances(self):
        # finding 5, a stated limit: a block square to the foot with its centre 62.5 mm in front of H (front edge
        # 92.5 mm) still lets the foot rest toe-down by itself (9.0 deg, H 34.5 mm), but (C) needs 90 mm: 0.75;
        # 59.5 mm (front edge 89.5 mm) scores 1.0
        for along, want in ((0.0595, 1.0), (0.0625, 0.75)):
            with self.subTest(along=along):
                b = block(along_foot(flat(), along), heading=-5.0)
                r = rest(flat(), b)
                h, t = heel_toe(r)
                self.assertGreater(foot_tilt_deg(r), 8.5)
                self.assertAlmostEqual(t[2], PLATE, delta=1e-6)
                self.assertEqual(score_of(r, b), want)

    def test_foot_dragged_across_the_table_to_the_plate_edge(self):
        # pushed along the table from its start until it meets the plate's side: still on the table, not on it
        f = foot((0.190, -0.100 - 0.0352, L.TOP), 0.0)
        self.assertEqual(score_of(f, GOAL_BLOCK), 0.0)

    def test_foot_on_its_side_at_the_right_spot(self):
        f = pitched()
        g = foot(f.origin + np.array([0, 0, 0.035]), -5.0, roll=90.0)
        self.assertEqual(score_of(g, GOAL_BLOCK), 0.0)

    def test_block_under_the_toe(self):
        b = block((0.370, -0.015))
        self.assertEqual(score_of(rest(flat(), b), b), 0.75)

    def test_block_under_the_middle(self):
        b = block((0.290, -0.008))
        self.assertEqual(score_of(rest(flat(), b), b), 0.75)

    def test_block_stood_on_end_under_the_heel(self):
        # on its 60 x 20 mm face: 70 mm tall, its axis horizontal; the heel would sit about 76 mm up
        b = block(z=PLATE + 0.035, roll=90.0)
        f = pitched(b1=0.0598)
        th = math.asin(0.070 / (L.FOOT_LEN - 0.0598))
        g = foot((f.origin[0], 0.0, PLATE + 0.070 + 0.0598 * math.sin(th)), -5.0, math.degrees(th))
        self.assertEqual(score_of(g, b), 0.0)  # tipped 25.9 deg: not sole down

    def test_the_wrong_part_on_the_plate(self):
        # the block set where the foot belongs, the foot left where it lay
        self.assertEqual(score_of(FOOT_START, block((0.300, 0.0))), 0.0)

    def test_heel_off_the_line_by_more_than_a_heel_half(self):
        self.assertEqual(score_of(pitched(heel_y=-0.030), block((0.220, -0.030))), 0.0)

    def test_toed_in_or_out_to_the_left(self):
        self.assertEqual(score_of(pitched(toe_out=-5.0), GOAL_BLOCK), 0.75)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = run(GOAL_FOOT, GOAL_BLOCK)
        self.assertEqual(r.score, 1.0)
        before = {k: list(v) for k, v in t.paths.items()}
        prev = t.prev
        bads = []
        for key, val in (("origin", [math.nan, 0, 0]), ("origin", [math.inf, 0, 0]), ("quat_xyzw", [0, 0, 0, 0]),
                         ("lin_vel", [0, math.nan, 0])):
            f = SimpleNamespace(**vars(GOAL_FOOT))
            setattr(f, key, np.array(val, float))
            bads.append((f, GOAL_BLOCK))
        missing = SimpleNamespace(quat_xyzw=GOAL_FOOT.quat_xyzw, lin_vel=np.zeros(3))  # no origin
        bads += [(missing, GOAL_BLOCK), (GOAL_FOOT, SimpleNamespace(pos=[0, 0, 1])), (None, None),
                 (GOAL_FOOT, block(vel=(math.nan, 0, 0)))]
        for f, b in bads:
            r = t.update(f, b, snap())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
        self.assertEqual(t.bad_steps, len(bads))
        self.assertIs(t.prev, prev)
        for k, v in t.paths.items():
            self.assertEqual(len(v), len(before[k]))
        r = t.update(GOAL_FOOT, GOAL_BLOCK, snap())
        self.assertEqual(r.score, 1.0)  # the record was kept

    def test_random_states_do_not_crash(self):
        rng = random.Random(7)
        t = Tracker(HZ)
        for _ in range(2000):
            f = foot((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 2)), rng.uniform(-360, 360),
                     rng.uniform(-180, 180), rng.uniform(-180, 180), vel=[rng.uniform(-5, 5) for _ in range(3)])
            b = block((rng.uniform(-1, 1), rng.uniform(-1, 1)), z=rng.uniform(0, 2), heading=rng.uniform(-180, 180),
                      pitch=rng.uniform(-180, 180), roll=rng.uniform(-180, 180))
            arms = snap([rng.uniform(-1, 1) for _ in range(3)], [rng.uniform(-1, 1) for _ in range(3)]) \
                if rng.random() < 0.8 else None
            r = t.update(f, b, arms)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            o, q = standin.rest_pose(f.origin, f.quat_xyzw, b)
            self.assertTrue(np.all(np.isfinite(o)) and np.all(np.isfinite(q)))

    def test_result_is_idempotent(self):
        r, t = run(GOAL_FOOT, GOAL_BLOCK)
        a = snap()
        steps = t.steps
        for _ in range(5):
            self.assertEqual(t.result(GOAL_FOOT, GOAL_BLOCK, a).score, 1.0)
        self.assertEqual(t.steps, steps)

    def test_arms_unavailable_are_counted(self):
        _, t = run(GOAL_FOOT, GOAL_BLOCK, arms=None, n=5)
        self.assertEqual(t.arms_unavailable, 5)
        arms = snap()
        arms["left"] = SimpleNamespace(tcp=np.array([math.nan, 0, 0]), tips=np.zeros(3), opening=1.0)
        r = t.update(GOAL_FOOT, GOAL_BLOCK, arms)
        self.assertEqual(t.arms_unavailable, 6)
        self.assertEqual(r.score, 0.0)


class Resets(unittest.TestCase):
    def placed(self):
        _, t = run(GOAL_FOOT, GOAL_BLOCK, n=30)
        return t

    def test_teleport_back_to_the_start_resets(self):
        t = self.placed()
        t.update(FOOT_START, BLOCK_START, snap())
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.paths["heel"]), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.placed()
        t.update(foot((L.FOOT_ORIGIN[0] + 0.02, L.FOOT_ORIGIN[1] - 0.03, L.TOP), 3.0),
                 block((L.BLOCK_START[0] - 0.03, L.BLOCK_START[1] + 0.02), z=L.TOP + HB / 2), snap())
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        t = self.placed()
        a = snap()
        p0, p1 = GOAL_FOOT.origin, np.array([*L.FOOT_ORIGIN[:2], L.TOP])
        b0, b1 = GOAL_BLOCK.pos, BLOCK_START.pos
        for i in range(1, 121):
            s = i / 120
            f = foot(p0 + (p1 - p0) * s, -5.0 * (1 - s), foot_tilt_deg(GOAL_FOOT) * (1 - s), vel=(p1 - p0) / 6.0)
            t.update(f, block(tuple((b0 + (b1 - b0) * s)[:2]), z=float((b0 + (b1 - b0) * s)[2]),
                              vel=(b1 - b0) / 6.0), a)
        self.assertEqual(t.steps, 150)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.placed()
        t.update(foot((0.30, -0.25, L.TOP), 40.0), GOAL_BLOCK, snap())  # knocked far, not to the start
        self.assertEqual(t.steps, 31)

    def test_a_dropped_foot_is_not_a_teleport(self):
        # let go 15 cm above its start, it falls fast onto its start: its velocity explains the motion
        t = self.placed()
        z, v = L.TOP + 0.15, 0.0
        t.update(foot((*L.FOOT_ORIGIN[:2], z), 0.0), BLOCK_START, snap())
        n = t.steps
        while z > L.TOP:
            v += 9.81 * DT
            z = max(L.TOP, z - v * DT)
            t.update(foot((*L.FOOT_ORIGIN[:2], z), 0.0, vel=(0, 0, -v)), BLOCK_START, snap())
            n += 1
        self.assertEqual(t.steps, n)

    def test_rigstate_reset_hook(self):
        t = self.placed()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            rigstate.offline_reset()
            self.assertEqual(t.steps, 0)
            self.assertIsNone(rigstate.arms())
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, stand-in switch and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.mk = importlib.import_module("mechkit")

    def objects(self, f, b, source="sim"):
        st = self.mk.state_from_poses(self.task.FOOT, (*f.origin, *f.quat_xyzw), None, source=source)
        return {"foot": st, "heel_block": b}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")
        com = self.task.FOOT.center_of_mass()
        self.assertAlmostEqual(com[0], L.FOOT_COM_S, places=3)
        self.assertAlmostEqual(com[2], 0.0168, places=3)
        self.assertEqual(self.task.FOOT.kind, "rigid")
        tops = {round(o.pos[2] + o.size[2] / 2, 6) for o in self.task.PLATE}
        self.assertEqual(tops, {round(PLATE, 6)})  # flush: the lines are inlaid

    def test_offline_box_round_trip(self):
        c, size, yaw = self.task.FOOT.offline_box()
        np.testing.assert_allclose(c, (0.275, -0.380, L.TOP + 0.029))
        st = SimpleNamespace(pos=np.array(c), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                             ang_vel=np.zeros(3))
        s = self.task.FOOT.offline_state(st)
        np.testing.assert_allclose(s.origin, L.FOOT_ORIGIN, atol=1e-9)
        self.assertEqual(s.source, "offline")

    def test_hooks_use_the_standin_only_offline(self):
        rigstate.offline_reset()
        dropped = flat(z=PLATE)  # what the checker hands over once it has dropped the column
        objs_off = self.objects(dropped, GOAL_BLOCK, source="offline")
        objs_sim = self.objects(dropped, GOAL_BLOCK, source="sim")
        for _ in range(WINDOW):
            snap()
            self.task.success(objs_off)
        self.assertEqual(self.task.score(objs_off), 1.0)  # rested toe-down on the block
        self.assertEqual(self.task.score(objs_sim), 0.75)  # the same pose read as real: flat, not on the block

    def test_reset_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        for _ in range(3):
            snap()
            self.task.success(self.objects(GOAL_FOOT, GOAL_BLOCK))
        self.assertEqual(tr.steps, 3)
        rigstate.offline_reset()
        self.assertEqual(tr.steps, 0)
        self.task.success(self.objects(GOAL_FOOT, GOAL_BLOCK))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        self.mk.offline_reset()  # mechkit's episode counter moves: a second reset signal
        snap()
        self.task.success(self.objects(GOAL_FOOT, GOAL_BLOCK))
        self.assertEqual(tr.steps, 1)

    def test_start_scores_zero(self):
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects(FOOT_START, BLOCK_START)))
        self.assertEqual(self.task.score(self.objects(FOOT_START, BLOCK_START)), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objects(FOOT_START, BLOCK_START)))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left_done = sum(w.duration for w in plan["left"])
        carry_start = sum(w.duration for w in plan["right"][:7])
        self.assertIn("carry it over the plate", plan["right"][7].label)
        self.assertGreaterEqual(carry_start, left_done - 1e-9)  # the left arm is parked before the foot comes over
        self.assertAlmostEqual(sum(w.duration for w in plan["right"]), 33.0)
        for w in plan["right"][:7]:
            self.assertEqual(w.yaw, 0.0)
        self.assertAlmostEqual(plan["right"][9].yaw, -math.radians(5.0))
        # let go with the sole 7 mm above the block's top, more than the estimated 6.6 mm payload sag
        self.assertGreater(self.task.RELEASE_GAP, 0.0066)
        self.assertAlmostEqual(plan["right"][11].pos[2] - L.TOP, 0.082, places=6)
        self.assertEqual(plan["right"][11].finger, self.task.OPEN)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_is_identical_everywhere(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        quoted = re.findall(r'--instruction "([^"]*)"', readme)
        self.assertGreaterEqual(len(quoted), 2)
        for q in quoted:
            self.assertEqual(q, self.task.INSTRUCTION)

    def test_task_json_fields(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertIn("demo: 1.0", spec["scores"])
        self.assertNotIn("media", spec)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(spec["run_limit"], "2,400 steps (2 minutes at 20 Hz)")
        self.assertIn("Possible scores: 0, 0.25, 0.5, 0.75, 1.0", spec["scoring"])
        mm = lambda v: f"{v * 1000:g} mm"  # noqa: E731
        needles = (
            f"{(L.AP_TARGET - L.AP_TOL) * 1000:g} to {(L.AP_TARGET + L.AP_TOL) * 1000:g} mm",
            f"{L.TOE_OUT_MIN_DEG:g} to {L.TOE_OUT_MAX_DEG:g} degrees",
            f"{L.HEEL_H_MIN * 1000:g} to {L.HEEL_H_MAX * 1000:g} mm", f"within {mm(L.TOE_T_TOL)}",
            f"within {mm(L.HEEL_SIDE_MAX)} of the centre line", f"{mm(L.RELEASE_CLEAR)} or more",
            f"less than {mm(L.STILL_PATH)}", f"{L.STILL_WINDOW_S:g} s",
            f"within {L.FOOT_TILT_MAX_DEG:g} degrees of vertical", f"within {L.HEADING_MAX_DEG:g} degrees of +x",
            f"within {L.BLOCK_TILT_MAX_DEG:g} degrees of vertical", f"within {mm(L.BLOCK_BOTTOM_TOL)}",
            f"within {mm(L.BLOCK_SIDE_MAX)}", f"{mm(-L.BLOCK_ALONG_MIN)} behind H",
            f"more than {mm(L.BLOCK_FRONT_MAX)} in front of H", "7.6 to 30.6 mm", "3.5 to 14.1 %",
            "let go and moved clear", "centre of mass, 94 mm from H",
        )
        self.assertAlmostEqual(L.TOE_OUT_RULER[0] * 100, 3.5, delta=0.05)
        self.assertAlmostEqual(L.TOE_OUT_RULER[1] * 100, 14.1, delta=0.05)
        for name, text in (("task.json", spec["scoring"]), ("README.md", (HERE / "README.md").read_text())):
            text = " ".join(text.split())
            for needle in needles:
                self.assertIn(needle, text, f"{name}: {needle}")


if __name__ == "__main__":
    unittest.main()
