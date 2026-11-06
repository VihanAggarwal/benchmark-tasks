"""Offline tests for scoring.py (and the rigstate hooks and demo in task.py): synthetic object states and synthetic
arm snapshots, no simulator.  python3 -m unittest discover -s tests -v"""

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
    Tracker, finger_gaps, finger_zone, inset, lowest, off_centre, print_face_up, print_tilt_deg, riser_flat,
    riser_on_board, riser_tilt_deg, riser_under_print, riser_yaw_deg, rotation, turned_deg, valid, zone_gap,
    zone_radius,
)

TIPS = 0.013  # pad tips below the TCP
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
R_H = L.RISER_SIZE[2] / 2
P_H = L.PRINT_SIZE[2] / 2
BX, BY = L.BOARD_X, L.BOARD_Y
OPEN = 1.0
PRINT_HELD = (L.PRINT_SIZE[1] - 0.008) / L.FULL_GAP  # rigstate opening while closed on the print's 70 mm


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


def riser(x, y, low=0.0, yaw=0.0, quat=None, **kw):
    """The riser lying flat (or turned by ``quat``) with its lowest point ``low`` above the table."""
    q = about("z", yaw) if quat is None else quat
    b = box((x, y, 0.0), L.RISER_SIZE, q)
    return box((x, y, L.TOP + low - (b.low[2] - 0.0)), L.RISER_SIZE, q, **kw)


def print_(x, y, low=0.0, yaw=0.0, quat=None, **kw):
    q = about("z", yaw) if quat is None else quat
    b = box((x, y, 0.0), L.PRINT_SIZE, q)
    return box((x, y, L.TOP + low - (b.low[2] - 0.0)), L.PRINT_SIZE, q, **kw)


R_START = riser(L.RISER_X, L.RISER_Y)
P_START = print_(L.PRINT_X, L.PRINT_Y)
R_SET = riser(BX, BY, 0.005, yaw=-90)  # lengthwise along x, as the demo leaves it
R_ACROSS = riser(BX, BY, 0.005)  # laid as it lies: its 70 mm across the print
P_FLOAT = print_(BX, BY, 0.015)
R_UP = riser(L.RISER_X, L.RISER_Y, 0.09)  # carried
P_UP = print_(L.PRINT_X, L.PRINT_Y, 0.09)


def arm(tcp, opening=OPEN):
    tcp = np.array(tcp, float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


PARKED = {"left": arm((0.240, 0.300, L.TOP + 0.12)), "right": arm((0.240, -0.300, L.TOP + 0.12))}
READY = {"left": arm((0.323, 0.305, L.TOP + 0.26)), "right": arm((0.323, -0.305, L.TOP + 0.26))}
HOLDING = {"left": arm((BX, BY, L.TOP + 0.015 + 0.002 + TIPS), PRINT_HELD), "right": PARKED["right"]}
HOVERING = {"left": arm((BX, BY, L.TOP + 0.025 + 0.010 + TIPS), OPEN), "right": PARKED["right"]}


def run(segments, hz=20.0):
    """Feed (riser, print, arms, steps) segments to a fresh tracker; return the last result and the tracker."""
    t = Tracker(hz)
    r = None
    for ri, pr, arms, n in segments:
        for _ in range(n):
            r = t.update(ri, pr, arms)
    return r, t


def slide(a, b, n, size):
    """n states moving a's centre to b's in equal steps (with b's orientation)."""
    return [box((1 - (i / n)) * a.pos + (i / n) * b.pos, size, b.quat_xyzw) for i in range(1, n + 1)]


def end_score(ri, pr, arms=PARKED, n=20):
    """Score of a still end state held for n control steps after a normal procedure."""
    r, _ = run([(R_START, P_START, READY, 5), (R_UP, P_UP, READY, 5), (ri, pr, arms, n)])
    return r


class Geometry(unittest.TestCase):
    def test_inset_nominal_and_across(self):
        self.assertAlmostEqual(inset(P_FLOAT, R_SET), 0.020, places=6)
        self.assertAlmostEqual(inset(P_FLOAT, riser(BX, BY, 0.005, yaw=90)), 0.020, places=6)
        self.assertAlmostEqual(inset(P_FLOAT, R_ACROSS), 0.0, places=6)  # flush with the long edges: it shows
        self.assertAlmostEqual(inset(P_FLOAT, riser(BX + 0.012, BY, 0.005, yaw=-90)), 0.008, places=6)
        self.assertAlmostEqual(inset(P_FLOAT, riser(BX, BY + 0.025, 0.005, yaw=-90)), -0.005, places=6)  # past an edge
        self.assertAlmostEqual(inset(print_(BX, BY, 0.015, yaw=90), R_ACROSS), 0.020, places=6)  # in the print's frame

    def test_riser_yaw_reported(self):
        self.assertAlmostEqual(riser_yaw_deg(R_START), 0.0)
        self.assertAlmostEqual(riser_yaw_deg(R_SET), -90.0)
        self.assertAlmostEqual(end_score(R_SET, P_FLOAT).details["riser_yaw_deg"], -90.0)

    def test_finger_gaps_take_the_nearer_part(self):
        right = arm((BX + 0.07, BY - 0.05, L.TOP + 0.10))  # high over the board's corner
        g = finger_gaps({"left": PARKED["left"], "right": right}, R_SET, P_FLOAT)
        self.assertAlmostEqual(g["right"], min(zone_gap(right, R_SET, L.RISER_SIZE),
                                               zone_gap(right, P_FLOAT, L.PRINT_SIZE)))
        self.assertGreater(g["left"], L.CLEAR_GAP)
        # the right arm still closed on the riser at the board, the print still on its mark: the riser counts
        on_riser = arm((BX, BY, L.TOP + 0.005 + 0.002 + TIPS), (L.RISER_SIZE[0] - 0.008) / L.FULL_GAP)
        g = finger_gaps({"left": PARKED["left"], "right": on_riser}, R_SET, P_START)
        self.assertLess(g["right"], 0.0)
        self.assertGreater(zone_gap(on_riser, P_START, L.PRINT_SIZE), L.CLEAR_GAP)

    def test_turned_folds_to_0_90(self):
        for yaw, want in ((0, 0), (5, 5), (-5, 5), (175, 5), (180, 0), (90, 90), (-90, 90), (100, 80)):
            self.assertAlmostEqual(turned_deg(print_(BX, BY, 0.015, yaw=yaw)), want, places=5)

    def test_off_centre_and_under(self):
        self.assertAlmostEqual(off_centre(print_(BX + 0.006, BY - 0.008, 0.015)), 0.010, places=6)
        self.assertTrue(riser_under_print(P_FLOAT, R_SET))
        self.assertFalse(riser_under_print(print_(BX + 0.06, BY, 0.015), R_SET))

    def test_riser_flat_either_face_and_on_board(self):
        self.assertTrue(riser_flat(R_SET))
        flipped = riser(BX, BY, 0.005, quat=qmul(about("z", -90), about("x", 180)))
        self.assertTrue(riser_flat(flipped))
        self.assertAlmostEqual(riser_tilt_deg(flipped), 0.0, places=4)
        self.assertTrue(riser_flat(riser(BX, BY, 0.005, quat=about("x", 9))))
        self.assertFalse(riser_flat(riser(BX, BY, 0.005, quat=about("x", 11))))
        self.assertFalse(riser_flat(riser(BX, BY, 0.005, quat=about("x", 90))))  # on its edge
        self.assertTrue(riser_on_board(R_SET))
        edge = L.BOARD_X + L.BOARD_HALF_X - L.RISER_SIZE[1] / 2 - 1e-6  # the riser's end exactly at the board's front edge
        self.assertTrue(riser_on_board(riser(edge, BY, 0.005, yaw=-90)))
        self.assertFalse(riser_on_board(riser(edge + 0.002, BY, 0.005, yaw=-90)))

    def test_print_face_up(self):
        self.assertTrue(print_face_up(P_FLOAT))
        self.assertTrue(print_face_up(print_(BX, BY, 0.015, quat=about("y", 2.5))))
        self.assertFalse(print_face_up(print_(BX, BY, 0.015, quat=about("y", 3.5))))
        down = print_(BX, BY, 0.015, quat=about("x", 180))
        self.assertFalse(print_face_up(down))
        self.assertAlmostEqual(print_tilt_deg(down), 180.0, places=4)

    def test_lowest(self):
        self.assertAlmostEqual(lowest(R_SET, L.RISER_SIZE), 0.005, places=9)
        self.assertAlmostEqual(lowest(P_FLOAT, L.PRINT_SIZE), 0.015, places=9)
        tipped = print_(BX, BY, 0.005, quat=about("x", 22))
        self.assertAlmostEqual(lowest(tipped, L.PRINT_SIZE), 0.005, places=9)

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 1), OPEN)), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135))
        pts, r = finger_zone(arm((0.3, 0.0, 1.0)))
        z = pts[:, 2]
        self.assertAlmostEqual(z.min(), 1.0 - TIPS)  # the zone ends at the tips' plane ...
        self.assertAlmostEqual(z.max(), 1.0 - TIPS + L.FINGER_LEN)  # ... and runs 38 mm up the gripper
        self.assertAlmostEqual(float(np.max(np.hypot(pts[:, 0] - 0.3, pts[:, 1]))), r)
        self.assertLess(zone_gap(HOLDING["left"], P_FLOAT, L.PRINT_SIZE), 0.0)  # closed on the print
        self.assertGreater(zone_gap(PARKED["left"], P_FLOAT, L.PRINT_SIZE), L.CLEAR_GAP)

    def test_zone_is_flat_at_the_tips(self):
        """Straight above the print the gap is the tips' height over its face, however far the gripper is open
        (a rounded cap would add up to 57 mm below the tips, where there are no pads)."""
        for opening in (1.0, PRINT_HELD, 0.0):
            for h in (0.025, 0.030, 0.035, 0.080):
                a = arm((BX, BY, L.TOP + 0.025 + h + TIPS), opening)
                self.assertAlmostEqual(zone_gap(a, P_FLOAT, L.PRINT_SIZE), h, places=9, msg=(opening, h))

    def test_zone_beside_and_at_a_corner(self):
        r = zone_radius(arm((0, 0, 0), OPEN))
        side = arm((BX, BY + L.PRINT_HALF_Y + r + 0.012, L.TOP + 0.020 + TIPS))  # tips level with the print's side
        self.assertAlmostEqual(zone_gap(side, P_FLOAT, L.PRINT_SIZE), 0.012, places=6)
        # tips 20 mm above the face, axis 10 mm past a corner diagonally: the rim's edge is the nearest point
        c = np.array([BX + L.PRINT_HALF_X, BY + L.PRINT_HALF_Y])
        d = (r + 0.010) / math.sqrt(2)
        corner = arm((c[0] + d, c[1] + d, L.TOP + 0.025 + 0.020 + TIPS))
        self.assertAlmostEqual(zone_gap(corner, P_FLOAT, L.PRINT_SIZE), math.hypot(0.010, 0.020), places=4)

    def test_zone_follows_the_turned_part(self):
        """The gap is taken from the part's real outline, not its world bounding box: beside a print turned 8
        degrees, 35 mm from its long side, it is 35 mm."""
        turned = print_(BX, BY, 0.015, yaw=8)
        r = zone_radius(arm((0, 0, 0), OPEN))
        h = math.radians(8)
        n = np.array([-math.sin(h), math.cos(h)])  # the print's +v side's outward normal
        p = np.array([BX, BY]) + n * (L.PRINT_HALF_Y + r + 0.035)
        a = arm((p[0], p[1], L.TOP + 0.020 + TIPS))
        self.assertAlmostEqual(zone_gap(a, turned, L.PRINT_SIZE), 0.035, places=6)

    def test_tilted_gripper_zone(self):
        """A closed gripper tilted 45 degrees about x, leaning away from the print, its tips 40 mm out from the
        print's side: the nearest point of the zone is on the tips' rim, r cos 45 nearer to the print than the
        tips and (placed here) level with the middle of the print's side."""
        r = math.hypot(L.FINGER_THICK, L.FINGER_HALF_WIDTH)  # closed
        k = math.sqrt(0.5)
        tips = np.array([BX, BY + L.PRINT_HALF_Y + 0.040, L.TOP + 0.020 - r * k])
        u = np.array([0.0, k, k])  # from the tips back up the gripper, away from the print
        a = SimpleNamespace(tcp=tips + TIPS * u, tips=tips, opening=0.0)
        self.assertAlmostEqual(zone_radius(a), r)
        self.assertAlmostEqual(zone_gap(a, P_FLOAT, L.PRINT_SIZE), 0.040 - r * k, places=4)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([(R_START, P_START, READY, 30)])
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = run([(R_START, P_START, READY, 10), (R_UP, P_UP, READY, 30), (R_SET, P_UP, READY, 40),
                    (R_SET, P_FLOAT, HOLDING, 20), (R_SET, P_FLOAT, PARKED, 20)])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.details["inset_mm"], 20.0)
        self.assertEqual(t.arms_unavailable, 0)

    def test_riser_only_scores_quarter(self):  # the right arm alone
        self.assertEqual(end_score(R_SET, P_START).score, 0.25)

    def test_riser_laid_across_shows_half(self):  # laid as it lies, print centred: its ends are flush
        r = end_score(R_ACROSS, P_FLOAT)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["hidden"])

    def test_print_quarter_turned_over_an_unturned_riser_half(self):
        r = end_score(R_ACROSS, print_(BX, BY, 0.015, yaw=90))
        self.assertTrue(r.details["hidden"])  # the riser is hidden, but the print is not square to the board
        self.assertEqual(r.score, 0.5)

    def test_off_centre_tiers(self):
        self.assertEqual(end_score(R_SET, print_(BX + 0.005, BY, 0.015)).score, 1.0)
        self.assertEqual(end_score(R_SET, print_(BX + 0.0065, BY, 0.015)).score, 0.75)
        self.assertEqual(end_score(R_SET, print_(BX, BY + 0.009, 0.015)).score, 0.75)  # 9 mm: inset 11 mm
        self.assertEqual(end_score(R_SET, print_(BX + 0.013, BY, 0.015)).score, 0.5)  # 13 mm along x: inset 7 mm
        far = riser(BX + 0.013, BY, 0.005, yaw=-90)  # riser and print both 13 mm off: inset 20 mm, off centre
        self.assertEqual(end_score(far, print_(BX + 0.013, BY, 0.015)).score, 0.5)
        near = riser(BX + 0.011, BY, 0.005, yaw=-90)
        self.assertEqual(end_score(near, print_(BX + 0.011, BY, 0.015)).score, 0.75)

    def test_turn_tiers(self):
        self.assertEqual(end_score(R_SET, print_(BX, BY, 0.015, yaw=2.5)).score, 1.0)
        self.assertEqual(end_score(R_SET, print_(BX, BY, 0.015, yaw=5)).score, 0.75)
        self.assertEqual(end_score(R_SET, print_(BX, BY, 0.015, yaw=-7.5)).score, 0.75)
        self.assertEqual(end_score(R_SET, print_(BX, BY, 0.015, yaw=10)).score, 0.5)
        self.assertEqual(end_score(R_SET, print_(BX, BY, 0.015, yaw=180)).score, 1.0)  # half a turn is square

    def test_inset_under_10_mm_half(self):
        r = end_score(riser(BX + 0.011, BY, 0.005, yaw=-90), P_FLOAT)  # riser 11 mm forward: inset 9 mm
        self.assertAlmostEqual(r.details["inset_mm"], 9.0)
        self.assertEqual(r.score, 0.5)
        r = end_score(riser(BX + 0.009, BY, 0.005, yaw=-90), P_FLOAT)
        self.assertEqual(r.score, 1.0)

    def test_riser_turned_part_way(self):
        self.assertEqual(end_score(riser(BX, BY, 0.005, yaw=-45), P_FLOAT).score, 0.5)  # a corner shows
        self.assertEqual(end_score(riser(BX, BY, 0.005, yaw=-80), P_FLOAT).score, 1.0)  # still 10 mm in

    def test_gripper_still_holding_or_touching_three_quarters(self):
        self.assertEqual(end_score(R_SET, P_FLOAT, HOLDING).score, 0.75)
        self.assertEqual(end_score(R_SET, P_FLOAT, HOVERING).score, 0.75)
        near_riser = {"left": PARKED["left"], "right": arm((BX + 0.07, BY - 0.05, L.TOP + 0.03))}
        self.assertEqual(end_score(R_SET, P_FLOAT, near_riser).score, 0.75)

    def test_clear_needs_30_mm(self):
        """An open gripper beside the print, its finger tips level with it: 25 mm away is not clear, 35 mm is."""
        r_open = zone_radius(arm((0, 0, 0), OPEN))
        for gap, want in ((0.025, 0.75), (0.035, 1.0)):
            y = L.PRINT_HALF_Y + r_open + gap  # the zone's axis this far from the print's side (y +35 mm)
            beside = {"left": arm((BX, BY + y, L.TOP + 0.020 + TIPS)), "right": PARKED["right"]}
            r = end_score(R_SET, P_FLOAT, beside)
            self.assertAlmostEqual(r.details["left_finger_gap_mm"], gap * 1000, places=3)
            self.assertEqual(r.score, want, msg=gap)

    def test_clear_straight_above_the_print(self):
        """An open gripper straight above the print: tips 35 mm above its face are clear, 25 mm are not, whatever
        the opening (the zone ends at the tips' plane; the old capsule needed about 87 mm when fully open)."""
        for opening in (1.0, PRINT_HELD, 0.3):
            for h, want in ((0.035, 1.0), (0.025, 0.75)):
                above = {"left": arm((BX, BY, L.TOP + 0.025 + h + TIPS), opening), "right": PARKED["right"]}
                r = end_score(R_SET, P_FLOAT, above)
                self.assertAlmostEqual(r.details["left_finger_gap_mm"], h * 1000, places=6)
                self.assertEqual(r.score, want, msg=(opening, h))
                self.assertEqual(r.details["clear"], want == 1.0)

    def test_let_go_lift_a_little_and_stop_scores_one(self):
        """Let go of the print, lift the open gripper 5 cm straight up and stop there: clear, 1.0 (the written
        rule; the capsule zone used to keep it at 0.75)."""
        release_z = L.TOP + 0.015 + 0.001 + 0.002 + TIPS  # the demo lets go 1 mm above the riser's top
        segs = [(R_START, P_START, READY, 5), (R_UP, P_UP, READY, 5), (R_SET, P_UP, READY, 20),
                (R_SET, P_FLOAT, HOLDING, 20), (R_SET, P_FLOAT, {**HOLDING, "left": arm((BX, BY, release_z))}, 10)]
        for dz in np.linspace(0.0, 0.05, 11)[1:]:
            segs.append((R_SET, P_FLOAT, {**HOLDING, "left": arm((BX, BY, release_z + dz))}, 1))
        stop = {**HOLDING, "left": arm((BX, BY, release_z + 0.05))}
        r, _ = run(segs + [(R_SET, P_FLOAT, stop, 25)])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.details["left_finger_gap_mm"], 50.0 - 7.0, places=6)  # tips 43 mm above the face

    def test_float_height_band(self):
        for low, want in ((0.0135, 1.0), (0.0165, 1.0), (0.0125, 0.25), (0.0175, 0.25), (0.022, 0.25)):
            self.assertEqual(end_score(R_SET, print_(BX, BY, low)).score, want, msg=low)
        for low, want in ((0.0035, 1.0), (0.0065, 1.0), (0.0025, 0.0), (0.0075, 0.0)):
            self.assertEqual(end_score(riser(BX, BY, low, yaw=-90), print_(BX, BY, low + 0.010)).score, want,
                             msg=low)

    def test_not_floating_quarter(self):
        aside = riser(BX - 0.06, BY, 0.005)  # set on the board, but not under the print
        for ri, p in (
            (aside, print_(BX + 0.03, BY, 0.005)),  # print laid on the bare board
            (R_SET, print_(BX, BY, 0.030)),  # held above the riser
            (R_SET, print_(BX, BY, 0.005, quat=about("x", 22))),  # tipped, one edge on the board
            (R_SET, print_(BX, BY, 0.015, quat=about("x", 180))),  # face down on the riser
            (R_SET, print_(BX, BY, 0.015, quat=about("y", 4))),  # tilted 4 degrees
            (R_SET, print_(BX + 0.07, BY, 0.015)),  # so far off that the riser's centre is not under it
        ):
            self.assertEqual(end_score(ri, p).score, 0.25, msg=str(p.pos))

    def test_riser_not_set_scores_zero(self):
        for ri in (
            riser(L.RISER_X, L.RISER_Y - 0.0),  # left on the table
            riser(0.20, -0.05),  # on the table in front of the board
            riser(BX, BY, 0.005, quat=qmul(about("z", -90), about("x", 90))),  # on its edge
            riser(BX, BY, 0.005, quat=about("y", 90)),  # on its end
            riser(L.BOARD_X + L.BOARD_HALF_X, BY, 0.005, yaw=-90),  # hanging past the board's front edge
            riser(BX, BY, 0.015, yaw=-90),  # on something 15 mm up (the print)
        ):
            self.assertEqual(end_score(ri, P_FLOAT).score, 0.0, msg=str(ri.pos))

    def test_left_arm_alone_scores_zero(self):
        self.assertEqual(end_score(R_START, print_(BX, BY, 0.005)).score, 0.0)

    def test_off_table_scores_zero(self):
        self.assertEqual(end_score(R_SET, print_(0.6, 0.0, -0.4)).score, 0.0)
        self.assertEqual(end_score(riser(0.6, 0.0, -0.4), P_FLOAT).score, 0.0)
        self.assertEqual(end_score(R_SET, print_(0.4, 0.0, -0.009, quat=about("y", 30))).score, 0.25)  # over the edge

    def test_still_moving_riser_scores_zero(self):
        t = Tracker()
        for _ in range(20):
            t.update(R_SET, P_START, READY)
        for i in range(12):  # 0.6 mm per step: 6 mm in 0.5 s, sliding at 12 mm/s
            r = t.update(riser(BX + 0.0006 * i, BY, 0.005, yaw=-90), P_START, READY)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["riser_at_rest"])
        for _ in range(11):
            r = t.update(riser(BX + 0.0066, BY, 0.005, yaw=-90), P_START, READY)
        self.assertEqual(r.score, 0.25)  # stopped: at rest again after 0.5 s


class Boundaries(unittest.TestCase):
    """A value exactly on a bound the text states as inclusive ('or less', 'or more', 'within', 'to') lands in the
    tier the text gives it, not one lower by floating-point rounding."""

    def test_off_centre_exactly_6_and_12_mm(self):
        r = end_score(R_SET, print_(BX + 0.006, BY, 0.015))
        self.assertGreater(off_centre(print_(BX + 0.006, BY, 0.015)), 0.006)  # 6.000000000000005 mm in floating point
        self.assertEqual(r.score, 1.0)
        self.assertEqual(end_score(R_SET, print_(BX - 0.0036, BY + 0.0048, 0.015)).score, 1.0)  # 6 mm diagonally
        both = riser(BX + 0.012, BY, 0.005, yaw=-90)  # riser moved with the print: inset stays 20 mm
        self.assertEqual(end_score(both, print_(BX + 0.012, BY, 0.015)).score, 0.75)
        self.assertEqual(end_score(both, print_(BX + 0.0121, BY, 0.015)).score, 0.5)
        self.assertEqual(end_score(R_SET, print_(BX + 0.0061, BY, 0.015)).score, 0.75)

    def test_turned_exactly_3_and_8_degrees(self):
        for yaw, want in ((3.0, 1.0), (-3.0, 1.0), (177.0, 1.0), (3.1, 0.75), (8.0, 0.75), (-8.0, 0.75),
                          (172.0, 0.75), (8.1, 0.5)):
            self.assertEqual(end_score(R_SET, print_(BX, BY, 0.015, yaw=yaw)).score, want, msg=yaw)

    def test_inset_exactly_10_mm(self):
        for dx in (0.010, -0.010):
            r = end_score(riser(BX + dx, BY, 0.005, yaw=-90), P_FLOAT)
            self.assertAlmostEqual(r.details["inset_mm"], 10.0)
            self.assertEqual(r.score, 1.0, msg=dx)
        self.assertEqual(end_score(riser(BX, BY - 0.010, 0.005, yaw=-90), P_FLOAT).score, 1.0)  # along y
        self.assertEqual(end_score(riser(BX + 0.0101, BY, 0.005, yaw=-90), P_FLOAT).score, 0.5)

    def test_clear_exactly_30_mm(self):
        for h, want in ((0.030, 1.0), (0.0299, 0.75)):
            above = {"left": arm((BX, BY, L.TOP + 0.025 + h + TIPS)), "right": PARKED["right"]}
            self.assertEqual(end_score(R_SET, P_FLOAT, above).score, want, msg=h)

    def test_flat_and_face_up_on_their_bounds(self):
        self.assertTrue(riser_flat(riser(BX, BY, 0.005, quat=qmul(about("z", -90), about("x", 10.0)))))
        self.assertTrue(riser_flat(riser(BX, BY, 0.005, quat=about("y", 10.0))))
        self.assertFalse(riser_flat(riser(BX, BY, 0.005, quat=about("y", 10.1))))
        self.assertTrue(print_face_up(print_(BX, BY, 0.015, quat=about("x", 3.0))))
        self.assertTrue(print_face_up(print_(BX, BY, 0.015, quat=about("y", -3.0))))
        self.assertFalse(print_face_up(print_(BX, BY, 0.015, quat=about("x", 3.1))))

    def test_height_bands_on_their_bounds(self):
        for low, want in ((0.013, 1.0), (0.017, 1.0), (0.0129, 0.25), (0.0171, 0.25)):
            self.assertEqual(end_score(R_SET, print_(BX, BY, low)).score, want, msg=low)
        for low, want in ((0.003, 1.0), (0.007, 1.0), (0.0029, 0.0), (0.0071, 0.0)):
            self.assertEqual(end_score(riser(BX, BY, low, yaw=-90), print_(BX, BY, low + 0.010)).score, want,
                             msg=low)

    def test_riser_on_the_board_edge(self):
        edge = L.BOARD_X + L.BOARD_HALF_X - L.RISER_SIZE[1] / 2  # the riser's end exactly on the board's front edge
        self.assertTrue(riser_on_board(riser(edge, BY, 0.005, yaw=-90)))
        side = L.BOARD_Y + L.BOARD_HALF_Y - L.RISER_SIZE[0] / 2
        self.assertTrue(riser_on_board(riser(BX, side, 0.005, yaw=-90)))
        self.assertFalse(riser_on_board(riser(edge + 0.0001, BY, 0.005, yaw=-90)))


class Stillness(unittest.TestCase):
    def test_velocity_chatter_is_ignored(self):
        """A part held still reports 10-45 mm/s of chatter: stillness is read from its path, never its velocity."""
        t = Tracker()
        for i in range(30):
            v = (0.045 * (-1) ** i, 0.03, -0.02)
            r = t.update(R_SET, print_(BX, BY, 0.015, vel=v, spin=(0.4, 0, 0)), HOLDING)
        self.assertTrue(r.details["print_at_rest"])
        self.assertEqual(r.score, 0.75)

    def test_small_jitter_is_still(self):
        t = Tracker()
        rng = np.random.default_rng(0)
        for _ in range(30):
            d = rng.uniform(-0.0002, 0.0002, 2)
            r = t.update(R_SET, print_(BX + d[0], BY + d[1], 0.015), PARKED)
        self.assertTrue(r.details["print_at_rest"])
        self.assertEqual(r.score, 1.0)

    def test_creep_and_turn_are_not_still(self):
        t = Tracker()
        for i in range(20):  # 0.35 degrees per step: 3.5 degrees in 0.5 s
            r = t.update(R_SET, print_(BX, BY, 0.015, yaw=0.35 * i), PARKED)
        self.assertFalse(r.details["print_at_rest"])
        self.assertEqual(r.score, 0.25)

    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for k in range(1, 12):
            r = t.update(R_SET, P_FLOAT, PARKED)
            self.assertEqual(r.score, 1.0 if k >= 11 else 0.0, msg=k)

    def test_success_after_landing_needs_half_a_second(self):
        t = Tracker()
        for _ in range(20):
            t.update(R_SET, P_UP, READY)
        scores = [t.update(R_SET, P_FLOAT, PARKED).score for _ in range(12)]
        self.assertEqual(scores[:10], [0.25] * 10)
        self.assertEqual(scores[10:], [1.0, 1.0])


class Exploits(unittest.TestCase):
    def test_riser_on_top_of_the_print(self):
        self.assertEqual(end_score(riser(BX, BY, 0.015, yaw=-90), print_(BX, BY, 0.005)).score, 0.0)
        self.assertEqual(end_score(riser(L.PRINT_X, L.PRINT_Y, 0.010), P_START).score, 0.0)

    def test_print_on_a_riser_on_the_table(self):
        self.assertEqual(end_score(riser(0.20, -0.15, 0.0, yaw=-90), print_(0.20, -0.15, 0.010)).score, 0.0)

    def test_riser_stood_on_end_under_the_print(self):
        tall = riser(BX, BY, 0.005, quat=about("y", 90))
        self.assertEqual(end_score(tall, print_(BX, BY, 0.035)).score, 0.0)

    def test_print_held_still_at_float_height(self):
        self.assertEqual(end_score(R_SET, P_FLOAT, HOLDING).score, 0.75)

    def test_turning_the_print_instead_of_the_riser(self):
        for yaw in (90, -90, 45):
            self.assertEqual(end_score(R_ACROSS, print_(BX, BY, 0.015, yaw=yaw)).score, 0.5, msg=yaw)

    def test_shifting_the_print_to_cover_an_unturned_riser(self):
        for dy in (-0.010, 0.010, 0.030):
            self.assertEqual(end_score(R_ACROSS, print_(BX, BY + dy, 0.015)).score, 0.5, msg=dy)

    def test_dropped_into_place_is_scored_on_its_merits(self):
        r, _ = run([(R_START, P_START, READY, 5), (riser(BX, BY, 0.05, yaw=-90), P_START, READY, 3),
                    (R_SET, P_START, READY, 15), (R_SET, print_(BX, BY, 0.06), PARKED, 2),
                    (R_SET, P_FLOAT, PARKED, 15)])
        self.assertEqual(r.score, 1.0)  # nothing in the scene breaks: no whole-trial rule

    def test_print_pushed_onto_the_riser_half_on(self):
        tipped = print_(BX + 0.05, BY, 0.005, quat=about("y", 12))
        self.assertEqual(end_score(R_SET, tipped).score, 0.25)

    def test_unreadable_arms_are_counted(self):
        r, t = run([(R_SET, P_FLOAT, None, 20)])
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 20)
        self.assertFalse(r.details["arms_available"])

    def test_one_gripper_unreadable_is_unreadable(self):
        bad = {"left": arm((np.nan, 0, 1)), "right": PARKED["right"]}
        r, t = run([(R_SET, P_FLOAT, bad, 12)])
        self.assertEqual(t.arms_unavailable, 12)


class Robustness(unittest.TestCase):
    def test_nan_and_zero_quaternion_score_zero_and_leave_the_tracker(self):
        t = Tracker()
        for _ in range(15):
            t.update(R_SET, P_FLOAT, PARKED)
        window = len(t.window["print"]), list(t.window["print"])[-1][0].copy()
        nan = box((np.nan, 0, 1), L.PRINT_SIZE)
        zero = SimpleNamespace(**{**vars(P_FLOAT), "quat_xyzw": np.zeros(4)})
        for bad in (nan, zero):
            r = t.update(R_SET, bad, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        r = t.update(box((0.3, np.inf, 1), L.RISER_SIZE), P_FLOAT, PARKED)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 3)
        self.assertEqual((len(t.window["print"]), list(t.window["print"])[-1][0].tolist()),
                         (window[0], window[1].tolist()))
        self.assertEqual(t.update(R_SET, P_FLOAT, PARKED).score, 1.0)  # picks up where it was

    def test_valid(self):
        self.assertTrue(valid(P_FLOAT))
        self.assertFalse(valid(SimpleNamespace(pos=(0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(**{**vars(P_FLOAT), "pos": np.zeros(2)})))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for _ in range(30):
            t.update(R_SET, P_FLOAT, None)
        self.assertEqual(t.arms_unavailable, 30)
        r = t.update(R_START, P_START, READY)  # both parts jump back to their marks in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertEqual(len(t.window["print"]), 1)
        self.assertEqual(r.score, 0.0)

    def test_putting_things_back_slowly_does_not_reset(self):
        t = Tracker()
        for ri in slide(R_START, R_SET, 40, L.RISER_SIZE) + slide(R_SET, R_START, 40, L.RISER_SIZE):
            t.update(ri, P_START, READY)
        self.assertEqual(t.steps, 80)

    def test_a_jump_elsewhere_does_not_reset(self):
        t = Tracker()
        for _ in range(10):
            t.update(R_START, P_START, READY)
        t.update(R_SET, P_START, READY)  # a jump, but not back onto the starting marks
        self.assertEqual(t.steps, 11)


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

    @staticmethod
    def feed(arms):
        rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in arms.items()})

    def test_objects_and_constants(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(names, ["backing_board", "riser", "print"])
        mass = {o.name: o.mass for o in self.task.OBJECTS}
        self.assertEqual(mass, {"backing_board": None, "riser": L.RISER_MASS, "print": L.PRINT_MASS})
        self.assertEqual((L.RISER_MASS, L.PRINT_MASS), (0.015, 0.022))  # mat board 0.71 g/cm3, gatorboard + print
        self.assertIn("70 x 30 x 10 mm, 15 g", self.spec["setup"])
        self.assertIn("110 x 70 x 10 mm, 22 g", self.spec["setup"])
        self.assertEqual(self.task.SCENE, "station")
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(self.task.INSTRUCTION, self.spec["instruction"])
        self.assertEqual(self.spec["run_limit"], "1,800 steps (1.5 minutes at 20 Hz)")

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(PARKED)
        for _ in range(12):
            self.task.success({"riser": R_SET, "print": P_FLOAT})
        self.assertTrue(self.task.success({"riser": R_SET, "print": P_FLOAT}))
        self.assertEqual(self.task.score({"riser": R_SET, "print": P_FLOAT}), 1.0)
        self.feed(HOLDING)
        self.assertFalse(self.task.success({"riser": R_SET, "print": P_FLOAT}))
        self.assertEqual(self.task.score({"riser": R_SET, "print": P_FLOAT}), 0.75)
        rigstate.offline_reset()  # rigstate's reset hook: the tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success({"riser": R_START, "print": P_START}))  # arms None after a reset
        self.assertEqual(tr.arms_unavailable, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed(READY)
            self.assertFalse(self.task.success({"riser": R_START, "print": P_START}))
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
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if w.finger != prev.finger:  # every close and every release: >= 1.5 s, in place, after a settle
                    self.assertGreaterEqual(w.duration, 1.5, msg=(side, w.label))
                    self.assertEqual(tuple(w.pos), tuple(prev.pos), msg=(side, w.label))
                    self.assertEqual(prev.label, "settle", msg=(side, w.label))
                    self.assertGreaterEqual(prev.duration, 0.5)
                if "slowly" in w.label:  # the last 2 cm of each place over at least 2 s
                    self.assertAlmostEqual(prev.pos[2] - w.pos[2], T.ENTRY)
                    self.assertGreaterEqual(w.duration, 2.0)
            carry = {w.pos[2] - L.TOP for w in wps if w.label.startswith(("carry", "lift", "to the wait"))}
            self.assertTrue(all(0.10 - 1e-9 <= z <= 0.14 for z in carry), msg=carry)
        self.assertAlmostEqual(T.RISER_GRIP, -(0.030 - 0.008) / 2)
        self.assertAlmostEqual(T.PRINT_GRIP, -(0.070 - 0.008) / 2)
        # the left arm leaves its wait point only once the right arm has parked
        right_end = sum(w.duration for w in plan["right"])
        i = next(k for k, w in enumerate(plan["left"]) if w.label == "carry it over the board")
        self.assertGreaterEqual(sum(w.duration for w in plan["left"][:i]), right_end - 1e-9)
        # the parts are let go 1 mm above their supports
        self.assertAlmostEqual(T.RISER_PLACE_Z - T.PAD_TIPS - T.GRIP_UP, L.BOARD_TOP + 0.001)
        self.assertAlmostEqual(T.PRINT_PLACE_Z - T.PAD_TIPS - T.GRIP_UP, L.RISER_TOP_ON_BOARD + 0.001)
        # the right arm picks at yaw +pi/2 and turns to yaw 0 on the way: the riser ends lengthwise along x
        self.assertAlmostEqual(plan["right"][0].yaw, math.pi / 2)
        self.assertAlmostEqual(plan["right"][-1].yaw, 0.0)
        # both of the right arm's quarter turns are slow enough to keep joint 6 under 1.2 rad/s (check_task: 1.15)
        names = ("over the riser", "carry it over the board, turning it lengthwise")
        turns = [w for w in plan["right"] if w.label in names]
        self.assertEqual([w.duration for w in turns], [2.8, 3.5])
        self.assertAlmostEqual(T.RIGHT_PARKED_S, 20.0)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * 20, 1800)


if __name__ == "__main__":
    unittest.main()
