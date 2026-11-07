"""Offline tests for scoring.py (and the rigstate hooks and demo in task.py): synthetic part states and synthetic arm
snapshots, no simulator.  python3 -m unittest discover -s tests -v"""

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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, box_distance, gauge_set, gaps, hands_off, lowest, point_box_distance, rail_tilt_deg, rail_yaw_deg,
    rotation, seat_offsets, seat_x, seated, shim_away, shim_contacts, shim_in_right_joint, shim_on_track,
    shim_tilt_deg, tier, tip_distances, valid,
)

TIPS = L.PAD_TIPS
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
MM = 0.001


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    """Quaternion product a * b (xyzw): turn b first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def box(center, size, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def _placed(x, y, low, size, quat, **kw):
    """A box of ``size`` turned by ``quat`` with its lowest corner ``low`` above the table."""
    b = box((x, y, 0.0), size, quat)
    return box((x, y, L.TOP + low - b.low[2]), size, quat, **kw)


def rail(y=L.MID_Y0, x=L.RAIL_X, low=L.SLEEPER_TOP, yaw=0.0, quat=None, **kw):
    """The middle length (lowest corner ``low`` above the table; seated on the sleepers by default)."""
    return _placed(x, y, low, L.MID_SIZE, about("z", yaw) if quat is None else quat, **kw)


def rail_gap(left_gap, **kw):
    """The middle length square and seated with its left joint ``left_gap`` (m); the right one is 20 mm minus it."""
    return rail(L.FIX_END_Y - L.HALF_LEN - left_gap, **kw)


def shim(x=L.SHIM_X0, y=L.SHIM_Y0, low=0.0, quat=None, **kw):
    return _placed(x, y, low, L.SHIM_SIZE, Q0 if quat is None else quat, **kw)


def arm(tcp, opening=1.0):
    tcp = np.array(tcp, float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


R_START = rail()  # crept 10 mm: left joint shut (0 mm), right joint 20 mm
R_SET = rail(0.0)  # slid 10 mm right: both joints 10 mm
S_START = shim()
S_UP = shim(low=0.070)  # carried
JOINT_SEAT_Y = -L.FIX_END_Y + L.SHIM_THICK / 2  # -0.160: the shim's face on the fixed rail end's face
S_IN_JOINT = shim(L.RAIL_X, JOINT_SEAT_Y, 0.0015)  # standing in the right joint against the fixed end, held
S_HOME = shim(low=0.0005)  # put back down in front of the track
PARKED = {"left": arm((0.200, 0.220, L.TOP + 0.14)), "right": arm((0.150, -0.250, L.TOP + 0.14))}
READY = {"left": arm((0.323, 0.305, L.TOP + 0.26)), "right": arm((0.323, -0.305, L.TOP + 0.26))}
RAIL_HELD = (L.RAIL_W - 0.008) / L.FULL_GAP
SHIM_HELD = (L.SHIM_SIZE[0] - 0.008) / L.FULL_GAP
ON_RAIL = arm((L.RAIL_X, 0.125, L.TOP + 0.040), RAIL_HELD)  # the left gripper still closed on the middle length
ON_SHIM_JOINT = arm((L.RAIL_X, JOINT_SEAT_Y, L.TOP + 0.0315), SHIM_HELD)
ON_SHIM_HOME = arm((L.SHIM_X0, L.SHIM_Y0, L.TOP + 0.0305), SHIM_HELD)
BOTH_HOLDING = {"left": ON_RAIL, "right": ON_SHIM_JOINT}


def run(segments, hz=20.0):
    """Feed (rail, shim, arms, steps) segments to a fresh tracker; return the last result and the tracker."""
    t = Tracker(hz)
    r = None
    for ra, sh, arms, n in segments:
        for _ in range(n):
            r = t.update(ra, sh, arms)
    return r, t


def slide(a, b, n, size):
    """n states moving a's centre to b's in equal steps (with b's orientation)."""
    return [box((1 - (i / n)) * a.pos + (i / n) * b.pos, size, b.quat_xyzw) for i in range(1, n + 1)]


def gauge_steps():
    """The method, as segments: the shim carried into the right joint against the fixed rail end (2 mm a step),
    then the middle length slid 10 mm onto it (0.5 mm a step). The tracker records 'gauged' on the last step."""
    segs = [(R_START, S_START, READY, 5)]
    segs += [(R_START, s, READY, 1) for s in slide(S_START, S_IN_JOINT, 50, L.SHIM_SIZE)]
    segs += [(r, S_IN_JOINT, BOTH_HOLDING, 1) for r in slide(R_START, R_SET, 20, L.MID_SIZE)]
    return segs


def primed(hz=20.0):
    """A tracker that has been through the method: 'gauged' is set, the rail at 10 mm, the shim in the joint."""
    _, t = run(gauge_steps(), hz)
    assert t.gauged
    return t


def end_score(ra, sh=S_HOME, arms=PARKED, n=20, gauged=True):
    """Score of an end state held for n control steps, reached by moving both parts to it over 50 steps (no
    teleport) from the joint set on the gauge (``gauged``) or from the start (the shim never used)."""
    t = primed() if gauged else run([(R_START, S_START, READY, 5)])[1]
    r0, s0 = (R_SET, S_IN_JOINT) if gauged else (R_START, S_START)
    for ra_i, sh_i in zip(slide(r0, ra, 50, L.MID_SIZE), slide(s0, sh, 50, L.SHIM_SIZE)):
        t.update(ra_i, sh_i, arms)
    r = None
    for _ in range(n):
        r = t.update(ra, sh, arms)
    assert t.gauged == gauged
    return r


class Geometry(unittest.TestCase):
    def test_start_gaps(self):
        left, right = gaps(R_START)
        self.assertAlmostEqual(left, 0.0, places=9)
        self.assertAlmostEqual(right, 0.020, places=9)
        self.assertAlmostEqual(L.START_LEFT_GAP, 0.0, places=9)
        self.assertAlmostEqual(L.START_RIGHT_GAP, 0.020, places=9)
        self.assertEqual(tuple(round(g, 9) for g in gaps(R_SET)), (0.010, 0.010))
        # the shim goes into the open joint with 5 mm clear of each rail end
        self.assertAlmostEqual((L.START_RIGHT_GAP - L.SHIM_THICK) / 2, 0.005, places=9)

    def test_gaps_are_the_narrowest_point_and_fold_end_for_end(self):
        for yaw in (2.0, -2.0, 182.0, 178.0):
            left, right = gaps(rail(0.0, yaw=yaw))
            want = L.FIX_END_Y - (L.HALF_LEN * math.cos(math.radians(2)) + L.RAIL_W / 2 * math.sin(math.radians(2)))
            self.assertAlmostEqual(left, want, places=9, msg=yaw)
            self.assertAlmostEqual(right, want, places=9, msg=yaw)
            self.assertLess(left + right, L.GAP_TOTAL)
        self.assertAlmostEqual(rail_yaw_deg(rail(0.0, yaw=182.0)), 2.0, places=6)
        self.assertAlmostEqual(rail_yaw_deg(rail(0.0, yaw=-3.0)), -3.0, places=6)

    def test_seated_at_its_bounds(self):
        self.assertTrue(seated(R_START))
        self.assertTrue(seated(R_SET))
        # pushed against a shoulder (5 mm of play): the centre line 5 mm off at both seats
        self.assertTrue(seated(rail(0.0, x=L.RAIL_X - L.SEAT_PLAY)))
        self.assertTrue(seated(rail(0.0, x=L.RAIL_X + 0.006)))  # exactly 6 mm: within
        self.assertFalse(seated(rail(0.0, x=L.RAIL_X + 0.0061)))
        # turned as far as the shoulders allow (3.1 degrees): the centre line 4.3 mm off at the seats
        yawed = rail(0.0, yaw=3.09)
        self.assertTrue(seated(yawed))
        self.assertTrue(all(o < 0.0045 for o in seat_offsets(yawed)))
        # height: lowest corner 16 to 20 mm up
        for low, want in ((0.016, True), (0.020, True), (0.0159, False), (0.0201, False), (0.028, False)):
            self.assertEqual(seated(rail(0.0, low=low)), want, msg=low)
        # tilt: its 25 mm axis within 3 degrees of vertical, either way up
        for quat, want in ((about("y", 3.0), True), (about("y", -3.0), True), (about("y", 3.1), False),
                           (about("x", 2.9), True), (about("x", 3.1), False), (about("y", 180.0), True),
                           (about("y", 90.0), False)):
            self.assertEqual(seated(rail(0.0, quat=quat)), want, msg=str(quat))
        self.assertAlmostEqual(rail_tilt_deg(rail(0.0, quat=about("x", 180))), 0.0, places=5)

    def test_seat_x_follows_the_centre_line(self):
        r = rail(0.0, yaw=2.0)
        t = math.tan(math.radians(2.0))
        self.assertAlmostEqual(seat_x(r, 0.08), L.RAIL_X - 0.08 * t, places=9)
        self.assertAlmostEqual(seat_x(r, -0.08), L.RAIL_X + 0.08 * t, places=9)
        self.assertTrue(math.isnan(seat_x(rail(0.0, yaw=90.0), 0.08)))  # turned across the track
        self.assertFalse(seated(rail(0.0, yaw=90.0)))

    def test_shim_away(self):
        self.assertTrue(shim_away(S_START))  # it starts in front of the track
        self.assertTrue(shim_away(S_HOME))
        self.assertFalse(shim_away(S_IN_JOINT))
        lying = shim(quat=about("x", 90))  # knocked flat on its 60 x 45 mm face
        self.assertTrue(shim_away(lying))
        self.assertAlmostEqual(shim_tilt_deg(lying), 90.0, places=5)
        half = L.SHIM_SIZE[0] / 2
        self.assertTrue(shim_away(shim(L.CESS_X - half - 0.0001)))  # far corner 0.1 mm short of x 0.160
        self.assertFalse(shim_away(shim(L.CESS_X - half)))  # exactly on the sleepers' near ends: not below
        for low, want in ((0.003, True), (0.0031, False), (-0.003, True), (-0.004, False)):
            self.assertEqual(shim_away(shim(low=low)), want, msg=low)

    def test_shim_on_track_bounds(self):
        half_x, half_y = L.SHIM_SIZE[0] / 2, L.SHIM_SIZE[1] / 2
        self.assertFalse(shim_on_track(S_START))
        self.assertFalse(shim_on_track(S_HOME))
        self.assertTrue(shim_on_track(S_IN_JOINT))
        # x 16.0 to 42.0 cm, inclusive
        self.assertTrue(shim_on_track(shim(L.TRACK_X[0] - half_x)))  # its far corners exactly at x 16.0 cm
        self.assertFalse(shim_on_track(shim(L.TRACK_X[0] - half_x - 0.0001)))
        self.assertTrue(shim_on_track(shim(L.TRACK_X[1] + half_x, -0.16)))  # near corners exactly at 42.0 cm
        self.assertFalse(shim_on_track(shim(L.TRACK_X[1] + half_x + 0.0001, -0.16)))
        # |y| below 42.0 cm (strict)
        self.assertTrue(shim_on_track(shim(0.29, -(L.TRACK_Y + half_y - 0.0001))))
        self.assertFalse(shim_on_track(shim(0.29, -(L.TRACK_Y + half_y))))
        self.assertFalse(shim_on_track(shim(0.29, L.TRACK_Y + half_y)))
        # lower than 45 mm above the table (strict): held 44.9 mm up over the joint is on it, 45 mm up is not
        self.assertTrue(shim_on_track(shim(L.RAIL_X, JOINT_SEAT_Y, 0.0449)))
        self.assertFalse(shim_on_track(shim(L.RAIL_X, JOINT_SEAT_Y, 0.045)))
        self.assertFalse(shim_on_track(shim(L.RAIL_X, JOINT_SEAT_Y, 0.100)))

    def test_shim_in_right_joint_bounds(self):
        self.assertTrue(shim_in_right_joint(R_START, S_IN_JOINT))
        self.assertTrue(shim_in_right_joint(R_SET, S_IN_JOINT))
        self.assertFalse(shim_in_right_joint(R_SET, S_HOME))
        # standing on (or held just above) the table: lowest corner within 3 mm
        self.assertTrue(shim_in_right_joint(R_START, shim(L.RAIL_X, -0.155, 0.003)))
        self.assertFalse(shim_in_right_joint(R_START, shim(L.RAIL_X, -0.155, 0.0031)))
        # across the rail's line: some of it at x 21.0 to 23.0 cm
        half_x = L.SHIM_SIZE[0] / 2
        self.assertTrue(shim_in_right_joint(R_START, shim(L.RAIL_LINE_X[1] + half_x, -0.155)))
        self.assertFalse(shim_in_right_joint(R_START, shim(L.RAIL_LINE_X[1] + half_x + 0.0001, -0.155)))
        self.assertTrue(shim_in_right_joint(R_START, shim(L.RAIL_LINE_X[0] - half_x, -0.155)))
        self.assertFalse(shim_in_right_joint(R_START, shim(L.RAIL_LINE_X[0] - half_x - 0.0001, -0.155)))
        # its centre between the fixed rail end's face (y -16.5 cm) and the middle length's right end
        end = L.MID_Y0 - L.HALF_LEN  # -0.145 at the start
        self.assertTrue(shim_in_right_joint(R_START, shim(L.RAIL_X, end)))
        self.assertFalse(shim_in_right_joint(R_START, shim(L.RAIL_X, end + 0.0001)))
        self.assertTrue(shim_in_right_joint(R_START, shim(L.RAIL_X, -L.FIX_END_Y)))
        self.assertFalse(shim_in_right_joint(R_START, shim(L.RAIL_X, -L.FIX_END_Y - 0.0001)))
        # the left joint is not the right joint
        self.assertFalse(shim_in_right_joint(rail_gap(0.012), shim(L.RAIL_X, L.FIX_END_Y - 0.006)))

    def test_box_distance(self):
        a = ((0.0, 0.0, 0.0), Q0, (0.02, 0.02, 0.02))
        self.assertAlmostEqual(box_distance(*a, (0.0, 0.025, 0.0), Q0, (0.02, 0.01, 0.02)), 0.010, places=9)
        self.assertAlmostEqual(box_distance(*a, (0.03, 0.03, 0.0), Q0, (0.02, 0.02, 0.02)), math.sqrt(2) * 0.01,
                               places=9)  # corner to corner
        self.assertEqual(box_distance(*a, (0.005, 0.0, 0.0), about("z", 30), (0.02, 0.02, 0.02)), 0.0)  # overlap
        # an edge-to-edge case: two bars crossed at right angles, one 7 mm above the other
        self.assertAlmostEqual(box_distance((0, 0, 0), Q0, (0.1, 0.01, 0.01), (0, 0, 0.017), about("z", 90),
                                            (0.1, 0.01, 0.01)), 0.007, places=8)
        # a turned box against a face: corner of the turned one nearest
        d = box_distance((0, 0, 0), about("z", 45), (0.02, 0.02, 0.02), (0.0, 0.03, 0.0), Q0, (0.04, 0.02, 0.04))
        self.assertAlmostEqual(d, 0.02 - 0.01 * math.sqrt(2), places=8)

    def test_gauge_set(self):
        self.assertFalse(gauge_set(R_START, S_IN_JOINT))  # in the joint, but the rail 10 mm off it
        self.assertTrue(gauge_set(R_SET, S_IN_JOINT))  # the rail drawn onto it: both faces touching
        f, r = shim_contacts(R_SET, S_IN_JOINT)
        self.assertAlmostEqual(f, 0.0, places=9)
        self.assertAlmostEqual(r, 0.0, places=9)
        # within 1 mm of the fixed rail end and of the middle length
        for dy, want in ((0.001, True), (0.0011, False)):
            # the shim 'dy' off the fixed end, the rail on it (the right joint 10 + dy mm)
            self.assertEqual(gauge_set(rail(dy), shim(L.RAIL_X, JOINT_SEAT_Y + dy, 0.0015)), want, msg=dy)
            # the shim on the fixed end, the rail 'dy' short of it
            self.assertEqual(gauge_set(rail(dy), shim(L.RAIL_X, JOINT_SEAT_Y, 0.0015)), want, msg=("rail", dy))
        # the right joint outside 8.5 to 11.5 mm when the shim fills it (a thicker gauge would): not gauged
        self.assertFalse(gauge_set(rail(0.0016), shim(L.RAIL_X, JOINT_SEAT_Y + 0.0008, 0.0015)))
        # a shim turned 3 degrees, still square enough: the right joint 10.6 mm at its narrowest
        sh = shim(L.RAIL_X, JOINT_SEAT_Y + 0.0006, 0.0, quat=about("z", 3.0))
        ra = rail(0.0006)
        self.assertTrue(gauge_set(ra, sh))
        self.assertTrue(8.5 <= gaps(ra)[1] * 1000 <= 11.5)
        # off the table (3.1 mm up) does not count; held 3 mm up does
        self.assertTrue(gauge_set(R_SET, shim(L.RAIL_X, JOINT_SEAT_Y, 0.003)))
        self.assertFalse(gauge_set(R_SET, shim(L.RAIL_X, JOINT_SEAT_Y, 0.0031)))

    def test_tip_distances(self):
        d = tip_distances(PARKED, R_SET, S_HOME)
        self.assertGreater(min(d["left"]), 0.10)
        self.assertGreater(min(d["right"]), 0.10)
        self.assertEqual(tip_distances({"left": ON_RAIL, "right": PARKED["right"]}, R_SET, S_HOME)["left"][0], 0.0)
        # a point beside a turned rail: the distance is to its real outline, not its bounding box
        r = rail(0.0, yaw=3.0)
        n = np.array([math.cos(math.radians(3.0)), math.sin(math.radians(3.0)), 0.0])
        p = np.array(r.pos) + n * (L.RAIL_W / 2 + 0.040)
        self.assertAlmostEqual(point_box_distance(p, r, L.MID_SIZE), 0.040, places=9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([(R_START, S_START, READY, 30)])
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["left_gap_mm"], 0.0)
        self.assertFalse(r.details["gauged"])

    def test_full_procedure_scores_one(self):
        segs = gauge_steps()
        segs += [(R_SET, S_IN_JOINT, {"left": PARKED["left"], "right": ON_SHIM_JOINT}, 20)]
        r, t = run(segs)
        self.assertEqual(r.score, 0.25)  # the shim still in the joint
        self.assertTrue(t.gauged)
        # set on the step the rail came within 1 mm of the shim (slid 9 mm, 0.5 mm a step: the right joint 11 mm)
        self.assertEqual(t.gauged_step, 5 + 50 + 18)
        segs += [(R_SET, s, {"left": PARKED["left"], "right": ON_SHIM_HOME}, 1)
                 for s in slide(S_IN_JOINT, S_UP, 50, L.SHIM_SIZE)]
        segs += [(R_SET, s, {"left": PARKED["left"], "right": ON_SHIM_HOME}, 1)
                 for s in slide(S_UP, S_HOME, 40, L.SHIM_SIZE)]
        segs += [(R_SET, S_HOME, {"left": PARKED["left"], "right": ON_SHIM_HOME}, 20)]
        r, t = run(segs)
        self.assertEqual(r.score, 0.5)  # the right gripper still on the shim
        r2, t2 = run(segs + [(R_SET, S_HOME, PARKED, 20)])
        self.assertEqual(r2.score, 1.0)
        self.assertTrue(r2.success)
        self.assertEqual((r2.details["left_gap_mm"], r2.details["right_gap_mm"]), (10.0, 10.0))
        self.assertEqual(t2.arms_unavailable, 0)
        self.assertEqual(t2.steps, 5 + 50 + 20 + 20 + 50 + 40 + 20 + 20)  # no reset on the way

    def test_every_tier_is_reachable(self):
        seen = {end_score(rail_gap(g * MM)).score for g in (10.0, 7.0, 4.0, 0.0)}
        self.assertEqual(seen, {1.0, 0.5, 0.25, 0.0})
        self.assertEqual(end_score(R_SET, gauged=False).score, 0.5)  # the joints set by eye
        self.assertEqual(end_score(R_SET, S_IN_JOINT, {"left": PARKED["left"], "right": ON_SHIM_JOINT}).score, 0.25)

    def test_exhaustive_left_gap_sweep(self):
        """Square and seated, shim put away, hands off, still: as the left gap goes from 0 to 20 mm the score is
        0, 0.25, 0.5, 1.0, 0.5, 0.25, 0 when gauged, changing at 3, 6, 8.5, 11.5, 14 and 17 mm (each bound
        inclusive); 0, 0.25, 0.5, 0.25, 0 when the gauge was not used; 0 then 0.25 from 3 mm with the shim left in
        the right joint."""
        def edges(**kw):
            out, prev = [], None
            for i in range(0, 2001):
                ra = rail_gap(i / 100 * MM)
                args = dict(in_joint=False, on_track=False, away=True, clear=True, still=True, gauged=True)
                args.update(kw)
                sc = tier(seated(ra), *gaps(ra), **args)
                if sc != prev:
                    out.append((round(i / 100, 2), sc))
                    prev = sc
            return out

        self.assertEqual(edges(), [(0.0, 0.0), (3.0, 0.25), (6.0, 0.5), (8.5, 1.0), (11.51, 0.5), (14.01, 0.25),
                                   (17.01, 0.0)])
        self.assertEqual(edges(gauged=False), [(0.0, 0.0), (3.0, 0.25), (6.0, 0.5), (14.01, 0.25), (17.01, 0.0)])
        self.assertEqual(edges(in_joint=True, on_track=True, away=False), [(0.0, 0.0), (3.0, 0.25), (17.01, 0.0)])
        self.assertEqual(edges(on_track=True, away=False), [(0.0, 0.0)])

    def test_band_bounds_through_the_tracker(self):
        for g, want in ((8.5, 1.0), (11.5, 1.0), (8.49, 0.5), (11.51, 0.5), (6.0, 0.5), (14.0, 0.5),
                        (5.99, 0.25), (14.01, 0.25), (3.0, 0.25), (17.0, 0.25), (2.99, 0.0), (17.01, 0.0)):
            self.assertEqual(end_score(rail_gap(g * MM)).score, want, msg=g)

    def test_quarter_is_the_shut_joint_opened_part_way(self):
        r = end_score(rail(L.MID_Y0 - 0.004), gauged=False)  # slid 4 mm: gaps 4 and 16
        self.assertEqual(r.score, 0.25)
        self.assertEqual((r.details["left_gap_mm"], r.details["right_gap_mm"]), (4.0, 16.0))

    def test_yawed_against_the_shoulders_reads_the_narrowest_point(self):
        r = end_score(rail(0.0, yaw=3.09))
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(r.details["left_gap_mm"], 9.69, places=1)

    def test_gripper_still_on_a_part_is_half(self):
        self.assertEqual(end_score(R_SET, S_HOME, {"left": ON_RAIL, "right": PARKED["right"]}).score, 0.5)
        self.assertEqual(end_score(R_SET, S_HOME, {"left": PARKED["left"], "right": ON_SHIM_HOME}).score, 0.5)

    def test_hands_off_needs_60_mm(self):
        top = L.TOP + L.SLEEPER_TOP + L.RAIL_H  # the rail's top face
        for h, want in ((0.060, 1.0), (0.0599, 0.5), (0.080, 1.0)):
            above = {"left": arm((L.RAIL_X, 0.05, top + h + TIPS)), "right": PARKED["right"]}
            r = end_score(R_SET, S_HOME, above)
            self.assertAlmostEqual(r.details["left_tip_to_rail_mm"], h * 1000, places=6)
            self.assertEqual(r.score, want, msg=h)
        shim_top = L.TOP + L.SHIM_SIZE[2] + 0.0005
        for h, want in ((0.060, 1.0), (0.0599, 0.5)):
            above = {"left": PARKED["left"], "right": arm((L.SHIM_X0, L.SHIM_Y0, shim_top + h + TIPS))}
            self.assertEqual(end_score(R_SET, S_HOME, above).score, want, msg=h)
        self.assertTrue(hands_off(PARKED, R_SET, S_HOME))

    def test_zero_conditions(self):
        for ra, sh, why in (
            (rail_gap(0.0029), S_HOME, "slid under 3 mm: the left joint 2.9 mm"),
            (rail(0.0, x=0.231, low=L.SLEEPER_TOP + L.SHOULDER_SIZE[2]), S_HOME, "the rail up on a shoulder"),
            (rail(0.0, quat=about("y", 90)), S_HOME, "the rail on its side"),
            (rail(0.0, low=0.030), S_HOME, "the rail held up out of its seats"),
            (rail(0.0, quat=about("x", 6.0)), S_HOME, "the rail tipped, one end up"),
            (rail_gap(0.0185), S_HOME, "slid too far: the right joint 1.5 mm"),
            (R_SET, shim(0.29, -0.25, L.SLEEPER_TOP), "the shim on a sleeper"),
            (R_SET, shim(0.29, -0.16), "the shim on the table between the rails"),
            (R_SET, shim(0.36, -0.16, L.SLEEPER_TOP + L.RAIL_H, quat=about("x", 90)), "the shim on the far rail"),
            (R_SET, shim(L.RAIL_X, -0.16, L.SLEEPER_TOP + L.RAIL_H, quat=about("x", 90)),
             "the shim lying on the rail tops over the joint"),
            (R_SET, shim(0.14, -0.16, 0.0005), "the shim's far end 1 cm in between the sleepers"),
            (R_SET, shim(L.RAIL_X, L.FIX_END_Y - 0.005), "the shim in the left joint"),
        ):
            self.assertEqual(end_score(ra, sh).score, 0.0, msg=why)
        self.assertEqual(run([(R_START, S_HOME, PARKED, 30)])[0].score, 0.0)  # nothing moved: the left joint shut

    def test_shim_left_in_the_right_joint_is_a_quarter(self):
        held = {"left": PARKED["left"], "right": ON_SHIM_JOINT}
        self.assertEqual(end_score(R_SET, S_IN_JOINT, held).score, 0.25)  # stopped with the gauge still held in
        self.assertEqual(end_score(R_SET, S_IN_JOINT, BOTH_HOLDING).score, 0.25)  # both grippers still on
        self.assertEqual(end_score(R_SET, shim(L.RAIL_X, JOINT_SEAT_Y, 0.0)).score, 0.25)  # let go, left standing
        # with the shim in the joint the rail must still have opened the shut joint 3 mm
        self.assertEqual(end_score(rail_gap(0.0029), shim(L.RAIL_X, JOINT_SEAT_Y, 0.0), gauged=False).score, 0.0)
        self.assertEqual(end_score(rail_gap(0.003), shim(L.RAIL_X, JOINT_SEAT_Y, 0.0), gauged=False).score, 0.25)

    def test_still_needs_half_a_second_of_history(self):
        t = Tracker()
        for k in range(1, 13):
            r = t.update(R_SET, S_HOME, PARKED)
            self.assertEqual(r.details["still"], k >= 11, msg=k)
            self.assertEqual(r.score, 0.5)  # never gauged


class Stillness(unittest.TestCase):
    def test_velocity_chatter_is_ignored(self):
        """A part held still reports 10-45 mm/s of chatter: stillness is read from its path, never its velocity."""
        t = primed()
        for i in range(30):
            v = (0.045 * (-1) ** i, 0.03, -0.02)
            r = t.update(rail(0.0, vel=v, spin=(0.4, 0, 0)), shim(low=0.0005, vel=v), PARKED)
        self.assertTrue(r.details["still"])
        self.assertEqual(r.score, 1.0)

    def test_small_jitter_is_still(self):
        t = primed()
        rng = np.random.default_rng(0)
        for _ in range(30):
            d = rng.uniform(-0.00005, 0.00005, 2)
            r = t.update(rail(d[1], x=L.RAIL_X + d[0]), S_HOME, PARKED)
        self.assertTrue(r.details["still"])
        self.assertEqual(r.score, 1.0)

    def test_creeping_rail_is_not_still(self):
        t = primed()
        start = rail(0.00125)
        for _ in range(15):
            t.update(start, S_HOME, PARKED)
        for i in range(1, 12):  # 0.25 mm per step: 2.5 mm in 0.5 s, sliding at 5 mm/s, inside the 1.0 band
            r = t.update(rail(0.00125 - 0.00025 * i), S_HOME, PARKED)
        self.assertFalse(r.details["still"])
        self.assertEqual(r.score, 0.5)
        for _ in range(11):
            r = t.update(rail(-0.0015), S_HOME, PARKED)
        self.assertEqual((r.details["left_gap_mm"], r.details["right_gap_mm"]), (11.5, 8.5))
        self.assertEqual(r.score, 1.0)  # stopped: still again after 0.5 s

    def test_vibrating_in_place_is_not_still(self):
        t = primed()
        for i in range(25):  # 0.3 mm back and forth every step: 3 mm of path in 0.5 s, going nowhere
            r = t.update(rail(0.00015 * (-1) ** i), S_HOME, PARKED)
        self.assertFalse(r.details["still"])
        self.assertEqual(r.score, 0.5)

    def test_path_bound_is_strict(self):
        t = primed()
        for _ in range(12):
            t.update(R_SET, S_HOME, PARKED)
        step = 0.0002  # ten steps of 0.2 mm: exactly 2 mm, not under it
        for i in range(1, 11):
            r = t.update(rail(-step * i), S_HOME, PARKED)
        self.assertAlmostEqual(r.details["rail_path_mm"], 2.0, places=6)
        self.assertFalse(r.details["still"])


class Exploits(unittest.TestCase):
    def test_setting_the_joints_by_eye_without_the_gauge(self):
        """Review E1: the shim never touched, the rail slid 10 mm (or 8.6 or 11.4) by eye and let go: 0.5, not
        1.0."""
        segs = [(R_START, S_START, READY, 10)]
        segs += [(ra, S_START, {"left": ON_RAIL, "right": READY["right"]}, 1)
                 for ra in slide(R_START, R_SET, 40, L.MID_SIZE)]
        segs += [(R_SET, S_START, PARKED, 30)]
        r, t = run(segs)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.gauged)
        for s in (0.0086, 0.0114, 0.0090):
            self.assertEqual(end_score(rail(L.MID_Y0 - s), S_START, gauged=False).score, 0.5, msg=s)

    def test_shim_held_off_the_track_is_half_whatever_its_height(self):
        """Review E2: the joint set on the gauge, the shim lifted out and still held over the cess: 0.5 at 3.0 mm
        up and at 3.1 mm up alike (a 0.1 mm change in a held part's height no longer swings the score to 0)."""
        for low in (0.002, 0.003, 0.0031, 0.010, 0.060):
            held = {"left": PARKED["left"], "right": arm((L.SHIM_X0, L.SHIM_Y0, L.TOP + low + 0.030), SHIM_HELD)}
            self.assertEqual(end_score(R_SET, shim(low=low), held).score, 0.5, msg=low)
        over_joint = {"left": PARKED["left"], "right": arm((L.RAIL_X, JOINT_SEAT_Y, L.TOP + 0.13), SHIM_HELD)}
        self.assertEqual(end_score(R_SET, shim(L.RAIL_X, JOINT_SEAT_Y, 0.10), over_joint).score, 0.5)

    def test_run_limit_mid_procedure_is_not_below_a_nudge(self):
        """Review E3: stopped with the rail drawn onto the held shim scores 0.25, as much as a 4 mm nudge with the
        shim never touched (it was 0 against 0.25)."""
        mid = end_score(R_SET, S_IN_JOINT, BOTH_HOLDING).score
        nudge = end_score(rail(L.MID_Y0 - 0.004), S_START, gauged=False).score
        self.assertEqual((mid, nudge), (0.25, 0.25))

    def test_holding_the_rail_in_place_instead_of_letting_go(self):
        self.assertEqual(end_score(R_SET, S_HOME, {"left": ON_RAIL, "right": PARKED["right"]}).score, 0.5)

    def test_wedging_the_shim_across_the_wide_joint(self):
        """Turned about 20 degrees (or leaning about 17), the 10 mm shim touches both ends of the 20 mm joint
        without the rail moving. That is no gauge reading: the right joint must be 8.5 to 11.5 mm when the shim
        fills it, so 'gauged' stays unset and setting the joints by eye afterwards scores 0.5."""
        wedged = shim(L.RAIL_X, -0.155, 0.0, quat=about("z", 20.0))
        leaning = shim(L.RAIL_X, -0.1528, 0.0, quat=about("x", 17.5))
        for sh in (wedged, leaning):
            self.assertTrue(shim_in_right_joint(R_START, sh))
            self.assertTrue(all(d <= L.GAUGE_TOUCH for d in shim_contacts(R_START, sh)))
            self.assertFalse(gauge_set(R_START, sh))
            segs = [(R_START, S_START, READY, 5)]
            segs += [(R_START, s, READY, 1) for s in slide(S_START, sh, 50, L.SHIM_SIZE)]
            segs += [(R_START, sh, READY, 10)]
            segs += [(R_START, s, READY, 1) for s in slide(sh, S_HOME, 50, L.SHIM_SIZE)]
            segs += [(ra, S_HOME, {"left": ON_RAIL, "right": READY["right"]}, 1)
                     for ra in slide(R_START, R_SET, 40, L.MID_SIZE)]
            segs += [(R_SET, S_HOME, PARKED, 30)]
            r, t = run(segs)
            self.assertFalse(t.gauged)
            self.assertEqual(r.score, 0.5)

    def test_shim_stood_against_the_rail_end_instead_of_the_fixed_end(self):
        """The shim put in the open joint against the middle length's end (10 mm short of the fixed rail end), then
        taken out and the rail slid by eye: never a gauge reading, 0.5."""
        against_rail = shim(L.RAIL_X, L.MID_Y0 - L.HALF_LEN - L.SHIM_THICK / 2)
        self.assertTrue(shim_in_right_joint(R_START, against_rail))
        self.assertFalse(gauge_set(R_START, against_rail))
        segs = [(R_START, S_START, READY, 5)]
        segs += [(R_START, s, READY, 1) for s in slide(S_START, against_rail, 50, L.SHIM_SIZE)]
        segs += [(R_START, s, READY, 1) for s in slide(against_rail, S_HOME, 50, L.SHIM_SIZE)]
        segs += [(ra, S_HOME, {"left": ON_RAIL, "right": READY["right"]}, 1)
                 for ra in slide(R_START, R_SET, 40, L.MID_SIZE)]
        segs += [(R_SET, S_HOME, PARKED, 30)]
        r, t = run(segs)
        self.assertFalse(t.gauged)
        self.assertEqual(r.score, 0.5)

    def test_shim_left_standing_in_the_joint(self):
        self.assertEqual(end_score(R_SET, S_IN_JOINT).score, 0.25)
        self.assertEqual(end_score(R_SET, shim(L.RAIL_X, JOINT_SEAT_Y, 0.0)).score, 0.25)  # let go, on the table

    def test_dragging_it_all_the_way_over(self):
        """Slid until the right joint shuts: the left gap is 20 mm, the right 0: 0."""
        self.assertEqual(end_score(rail_gap(0.020)).score, 0.0)
        self.assertEqual(end_score(rail_gap(0.0165)).score, 0.25)  # stopped 3.5 mm short of shutting it: 0.25

    def test_lifting_it_out_of_its_seats(self):
        held = rail(0.0, low=0.025)
        self.assertEqual(end_score(held, S_HOME, {"left": ON_RAIL, "right": PARKED["right"]}).score, 0.0)

    def test_knocking_the_rail_off_its_seats(self):
        knocked = rail(0.0, x=0.250, low=0.0, quat=about("z", 20))  # off the sleepers, on the table
        self.assertEqual(end_score(knocked).score, 0.0)

    def test_shim_held_touching_the_table_in_front(self):
        """Held, not put down: within 3 mm of the table counts as put away, but the gripper is on it (0.5)."""
        held = shim(low=0.002)
        self.assertEqual(end_score(R_SET, held, {"left": PARKED["left"], "right": ON_SHIM_HOME}).score, 0.5)

    def test_wrong_place_for_the_shim(self):
        """Left on the track: 0 (review E4: centred in front but its far end 1 cm in between the sleepers is on the
        track). Off the track but not on the table in front of it: at most 0.5."""
        for sh, want, why in (
            (shim(0.29, -0.16), 0.0, "between the rails"),
            (shim(0.14, -0.16, 0.0005), 0.0, "its far end 1 cm in between the sleepers"),
            (shim(L.RAIL_X, -0.16, 0.0449), 0.0, "held 1.9 mm above the rail tops over the joint"),
            (shim(0.46, -0.16), 0.5, "on the table behind the track"),
            (shim(0.30, -0.70, -0.40), 0.5, "dropped off the table"),
            (shim(0.12, -0.50, 0.0005), 1.0, "on the table in front of the track, beyond the rail ends"),
            (shim(0.129, -0.16, 0.0005), 1.0, "in front, its far end 1 mm short of the sleeper ends"),
        ):
            self.assertEqual(end_score(R_SET, sh).score, want, msg=why)

    def test_gaming_the_band_edge_by_turning(self):
        """Turning the rail does not widen a gap: the narrowest point is read, so a turned rail 2 mm short of the
        band stays short."""
        short = rail(L.FIX_END_Y - L.HALF_LEN - 0.0065, yaw=3.0)  # left gap under 6.5 mm at its narrowest
        r = end_score(short)
        self.assertLess(r.details["left_gap_mm"], 6.5)
        self.assertEqual(r.score, 0.5)

    def test_unreadable_arms_are_counted(self):
        t = primed()
        for _ in range(20):
            r = t.update(R_SET, S_HOME, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 20)  # only the None steps (the method's steps had both arms)
        self.assertFalse(r.details["arms_available"])
        bad = {"left": arm((np.nan, 0, 1)), "right": PARKED["right"]}
        _, t = run([(R_SET, S_HOME, bad, 12)])
        self.assertEqual(t.arms_unavailable, 12)


class Robustness(unittest.TestCase):
    def test_nan_inf_zero_quaternion_and_missing_fields_score_zero_and_leave_the_tracker(self):
        t = primed()
        for _ in range(15):
            t.update(R_SET, S_HOME, PARKED)
        window = [p.copy() for p in t.window["rail"]]
        steps, gauged_step = t.steps, t.gauged_step
        nan = box((np.nan, 0, 1), L.MID_SIZE)
        inf = box((0.3, np.inf, 1), L.SHIM_SIZE)
        zero = SimpleNamespace(**{**vars(R_SET), "quat_xyzw": np.zeros(4)})
        missing = SimpleNamespace(pos=R_SET.pos, quat_xyzw=R_SET.quat_xyzw)
        for ra, sh in ((nan, S_HOME), (R_SET, inf), (zero, S_HOME), (missing, S_HOME), (None, S_HOME)):
            r = t.update(ra, sh, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.bad_steps, 5)
        self.assertEqual(t.steps, steps + 5)
        self.assertEqual([p.tolist() for p in t.window["rail"]], [p.tolist() for p in window])
        self.assertEqual((t.gauged, t.gauged_step), (True, gauged_step))
        self.assertEqual(t.update(R_SET, S_HOME, PARKED).score, 1.0)  # picks up where it was

    def test_valid(self):
        self.assertTrue(valid(R_SET))
        self.assertFalse(valid(SimpleNamespace(pos=(0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(**{**vars(R_SET), "pos": np.zeros(2)})))

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(1)
        t = Tracker()
        for i in range(400):
            q = rng.normal(size=4) if i % 7 else np.zeros(4)
            ra = box(rng.uniform(-1, 1, 3), L.MID_SIZE, q if np.linalg.norm(q) > 0 else Q0)
            ra.quat_xyzw = q
            sh = box(rng.uniform(-1, 1, 3), L.SHIM_SIZE, about("z", rng.uniform(0, 360)))
            arms = None if i % 5 == 0 else {s: arm(rng.uniform(-1, 1, 3), rng.uniform(0, 1)) for s in ("left", "right")}
            r = t.update(ra, sh, arms)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
        # random states near the joint, where the contact distances are worked out
        for i in range(300):
            ra = rail(rng.uniform(-0.012, 0.012), x=L.RAIL_X + rng.uniform(-0.006, 0.006), yaw=rng.uniform(-4, 4))
            sh = box((L.RAIL_X + rng.uniform(-0.04, 0.04), rng.uniform(-0.17, -0.13), L.TOP + rng.uniform(0.0, 0.03)),
                     L.SHIM_SIZE, rng.normal(size=4))
            r = t.update(ra, sh, PARKED)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))

    def test_score_is_idempotent(self):
        t = primed()
        for _ in range(15):
            t.update(R_SET, S_HOME, PARKED)
        steps = t.steps
        a = t.result(R_SET, S_HOME, PARKED)
        b = t.result(R_SET, S_HOME, PARKED)
        self.assertEqual((a.score, a.details), (b.score, b.details))
        self.assertEqual(a.score, 1.0)
        self.assertEqual(t.steps, steps)
        c = t.result(R_SET, S_IN_JOINT, PARKED)  # result() never sets the record either
        self.assertEqual(c.score, 0.25)
        self.assertEqual(t.steps, steps)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        """The demo ends with the shim back on its own place, so only the rail jumps (10 mm) at a reset; the
        'gauged' record is cleared with everything else."""
        t = primed()
        for _ in range(30):
            t.update(R_SET, S_HOME, None)
        self.assertEqual(t.arms_unavailable, 30)
        self.assertTrue(t.gauged)
        r = t.update(R_START, S_START, READY)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertEqual(len(t.window["rail"]), 1)
        self.assertEqual((t.gauged, t.gauged_step), (False, None))
        self.assertEqual(r.score, 0.0)

    def test_jittered_teleport_resets(self):
        t = primed()
        for _ in range(20):
            t.update(R_SET, shim(0.30, -0.45), PARKED)
        t.update(rail(L.MID_Y0 - 0.0015, x=L.RAIL_X + 0.001), shim(L.SHIM_X0 + 0.002, L.SHIM_Y0 + 0.001), READY)
        self.assertEqual(t.steps, 1)
        self.assertFalse(t.gauged)

    def test_putting_things_back_slowly_does_not_reset(self):
        t = Tracker()
        for ra in slide(R_START, R_SET, 40, L.MID_SIZE) + slide(R_SET, R_START, 40, L.MID_SIZE):
            t.update(ra, S_START, READY)
        self.assertEqual(t.steps, 80)
        t = primed()
        steps = t.steps
        for ra, sh in zip(slide(R_SET, R_START, 40, L.MID_SIZE), slide(S_IN_JOINT, S_START, 40, L.SHIM_SIZE)):
            t.update(ra, sh, READY)
        self.assertEqual(t.steps, steps + 40)
        self.assertTrue(t.gauged)

    def test_a_jump_elsewhere_does_not_reset(self):
        t = Tracker()
        for _ in range(10):
            t.update(R_SET, S_HOME, PARKED)
        t.update(rail(L.MID_Y0 + 0.009), S_HOME, PARKED)  # a 19 mm jump, but not back onto the start
        self.assertEqual(t.steps, 11)
        t.update(R_START, shim(0.30, -0.45), PARKED)  # the rail back on its start, the shim elsewhere
        self.assertEqual(t.steps, 12)


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
        cls.spec = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()

    @staticmethod
    def feed(arms):
        rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in arms.items()})

    @staticmethod
    def objs(ra, sh):
        return {"middle_rail": ra, "shim": sh}

    def test_objects_and_constants(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), 15)
        self.assertEqual(len(set(names)), 15)
        loose = {o.name: o.mass for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, {"middle_rail": 0.25, "shim": 0.21})
        self.assertEqual(self.task.SCENE, "station")
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(self.spec["run_limit"], "3,000 steps (2.5 minutes at 20 Hz)")
        self.assertIn("demo: 1.0", self.spec["scores"])
        for text in ("310 mm long, 0.25 kg", "45 mm tall, 0.21 kg", "the left joint is shut (0 mm) and the right "
                     "joint 20 mm", "5 mm clear of each rail end", "2 mm above the rail tops"):
            self.assertIn(text, self.spec["setup"])
        for text in ("8.5 to 11.5 mm", "6 to 14 mm", "at least 3 mm", "at least 60 mm", "shorter than 2 mm",
                     "within 6 mm of x 22.0 cm", "16 to 20 mm", "within 3 degrees", "within 3 mm of the table top",
                     "below 16.0 cm", "x 16.0 to 42.0 cm", "below 42.0 cm", "lower than 45 mm", "x 21.0 to 23.0 cm",
                     "within 1 mm of both", "3, 6, 8.5, 11.5, 14 and 17 mm", "Possible scores: 0, 0.25, 0.5, 1.0."):
            self.assertIn(text, self.spec["scoring"])
        self.assertNotIn("cannot be read", self.spec["scoring"])  # an implementation note: it is in the README
        self.assertIn("cannot read the arms", self.readme)

    def test_instruction_is_the_same_everywhere(self):
        self.assertEqual(self.task.INSTRUCTION, self.spec["instruction"])
        self.assertEqual(self.readme.count(f'--instruction "{self.task.INSTRUCTION}"'), 2)
        for text in (self.task.INSTRUCTION, json.dumps(self.spec), self.readme):
            self.assertNotIn("\u2014", text)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(PARKED)
        for _ in range(12):  # the joints right, but never set on the gauge: 0.5, no success
            self.task.success(self.objs(R_SET, S_HOME))
        self.assertFalse(self.task.success(self.objs(R_SET, S_HOME)))
        self.assertEqual(self.task.score(self.objs(R_SET, S_HOME)), 0.5)
        rigstate.offline_reset()
        self.assertFalse(tr.gauged)
        for ra, sh, arms, n in gauge_steps():
            self.feed(arms)
            for _ in range(n):
                self.task.success(self.objs(ra, sh))
        self.assertTrue(tr.gauged)
        self.feed(PARKED)
        for ra, sh in zip([R_SET] * 50, slide(S_IN_JOINT, S_HOME, 50, L.SHIM_SIZE)):
            self.task.success(self.objs(ra, sh))
        for _ in range(12):
            self.task.success(self.objs(R_SET, S_HOME))
        self.assertTrue(self.task.success(self.objs(R_SET, S_HOME)))
        self.assertEqual(self.task.score(self.objs(R_SET, S_HOME)), 1.0)
        self.assertEqual(self.task.score(self.objs(R_SET, S_HOME)), 1.0)  # score() changes nothing
        self.feed({"left": ON_RAIL, "right": PARKED["right"]})
        self.assertFalse(self.task.success(self.objs(R_SET, S_HOME)))
        self.assertEqual(self.task.score(self.objs(R_SET, S_HOME)), 0.5)
        rigstate.offline_reset()  # rigstate's reset hook: the tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(tr.gauged)
        self.assertFalse(self.task.success(self.objs(R_START, S_START)))  # arms None after a reset
        self.assertEqual(tr.arms_unavailable, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed(READY)
            self.assertFalse(self.task.success(self.objs(R_START, S_START)))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_timing_and_targets(self):
        T = self.task
        plan = T.demo()

        def ends(side):
            t, out = 0.0, {}
            for w in plan[side]:
                t += w.duration
                out.setdefault(w.label, t)
            return out

        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if w.finger != prev.finger:  # every close and every release: >= 1.5 s, in place, after a settle
                    self.assertGreaterEqual(w.duration, 1.5, msg=(side, w.label))
                    self.assertEqual(tuple(w.pos), tuple(prev.pos), msg=(side, w.label))
                    self.assertEqual(prev.label, "settle", msg=(side, w.label))
                    self.assertGreaterEqual(prev.duration, 0.5)
                if w.label.startswith("slowly"):  # slow final descents
                    self.assertGreaterEqual(w.duration, 1.5, msg=(side, w.label))
            self.assertTrue(all(abs(w.yaw - (T.YAW_L if side == "left" else T.YAW_R)) < 1e-12 for w in wps))
        self.assertAlmostEqual(T.SHIM_GRIP, -(0.060 - 0.008) / 2)
        self.assertAlmostEqual(T.RAIL_GRIP, -(0.020 - 0.008) / 2)
        left, right = ends("left"), ends("right")
        self.assertAlmostEqual(T.RIGHT_SEATED_S, 19.7)
        # the slide starts only after the shim is on the fixed rail end's face
        slide_start = left["slide it right until it bears on the shim"] - 4.0
        self.assertGreaterEqual(slide_start, right["onto the fixed rail end's face"] + 0.5 - 1e-9)
        # the shim leaves the joint only after the left gripper has let go and risen clear of the rail
        self.assertGreaterEqual(right["hold it while the rail is slid onto it and let go"],
                                left["straight up, clear of the rail"] - 1e-9)
        # the shim goes into the joint centred, 5 mm clear of each rail end, and ends on the fixed end's face
        self.assertAlmostEqual(T.JOINT_Y, -0.155)
        rail_end = L.MID_Y0 - L.HALF_LEN
        self.assertAlmostEqual(rail_end - (T.JOINT_Y + L.SHIM_THICK / 2), 0.005)
        self.assertAlmostEqual((T.JOINT_Y - L.SHIM_THICK / 2) + L.FIX_END_Y, 0.005)
        self.assertAlmostEqual(T.SEAT_Y - L.SHIM_THICK / 2, -L.FIX_END_Y)
        self.assertAlmostEqual(T.SHIM_SET_Z - L.TOP - T.SHIM_HANG, 0.0015)  # within the gauge rule's 3 mm
        # the slide is commanded 27 mm (the rail stops on the shim after 10) and eased back to 10 mm before letting go
        self.assertAlmostEqual(T.TRAVEL_TO_SHIM, 0.010)
        self.assertAlmostEqual(T.SLIDE_CMD, 0.027)
        self.assertAlmostEqual(T.SLIDE_CMD - T.TRAVEL_TO_SHIM, 0.017)
        self.assertAlmostEqual(T.EASE_TO, 0.010)
        # the shim is lifted out of the joint until its underside is 2 mm above the rail tops: off the track
        self.assertAlmostEqual(T.SHIM_ENTRY_Z - T.SHIM_HANG - L.TOP, L.TRACK_Z)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertAlmostEqual(total, 46.3)
        self.assertLess((total + 3.0) * 20, 3000 / 3)  # the run limit is over 3x the demo


if __name__ == "__main__":
    unittest.main()
