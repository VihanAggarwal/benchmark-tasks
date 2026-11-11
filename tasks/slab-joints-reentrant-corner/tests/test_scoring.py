"""Offline tests for scoring.py and task.py: synthetic strip states and synthetic arm snapshots fed through
rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Bar, Tracker, axis_of, counted, end_meets, evaluate, inside, joint_lines, lowest, panels, rotation, snapped,
    strip_info, tilt_deg, valid, zone_gap, zone_radius,
)

HZ = 20.0
TOP = L.TOP
TIPS = 0.013  # pad tips below the TCP
BASE = TOP + L.SUBBASE_H  # the sub-base's top (world z)
W = L.STRIP_W
MID_X = (L.LONG[0] + L.LONG[1]) / 2  # 0.320
CY = L.CORNER[1]  # 0.120
J2_Y = CY + W / 2 + 0.004 + L.SHORT_LEN / 2  # 0.1745: the short strip 4 mm from J1 and from the left form
LEFT = L.FACES_Y[0][0]  # 0.220: the left form's inside face
L_PARK = np.array([0.20, 0.30, TOP + 0.15])
R_PARK = np.array([0.20, -0.30, TOP + 0.15])
NAMES = L.STRIP_NAMES
LONG_NAMES = ("strip_left_long", "strip_right_inner", "strip_right_outer")


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


def raw(name, center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like strip: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    half = np.array((L.STRIPS[name][0] / 2, W / 2, L.STRIP_T / 2))
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def strip(name, x, y, axis="x", turn=0.0, roll=0.0, lift=0.0, floor=None, edge=False, on_end=False):
    """A strip lying flat with its length along ``axis`` ('x' or 'y'), centred at (x, y), turned ``turn`` degrees
    about the vertical, rolled ``roll`` degrees about its length, its lowest point ``lift`` above ``floor`` (default
    the sub-base's top). ``edge``: lying on a 12 mm edge; ``on_end``: standing on an end."""
    q = about("z", (0.0 if axis == "x" else 90.0) + turn)
    if edge:
        q = qmul(q, about("x", 90.0))
    if on_end:
        q = qmul(q, about("y", 90.0))
    if roll:
        q = qmul(q, about("x", roll))
    s = raw(name, (x, y, 0.0), q)
    z = (BASE if floor is None else floor) + lift - (s.low[2] - s.pos[2])
    return raw(name, (x, y, z), q)


def on_table(name):
    """A strip on its starting mark on the table."""
    _, _, x, y = L.STRIPS[name]
    return strip(name, x, y, floor=TOP)


def slab(j1=CY, j2=(L.CORNER[0], J2_Y), j3=0.0, j4=-0.120, **over):
    """The four strips: the left long one across the long leg on y ``j1`` (out of the inside corner), the short one
    along y at ``j2`` (between the legs), the right ones across the long leg on y ``j3`` and ``j4``. Pass None to
    leave a strip on the table; ``over`` replaces any strip by name."""
    out = {
        "strip_left_long": strip("strip_left_long", MID_X, j1) if j1 is not None else on_table("strip_left_long"),
        "strip_left_short": (strip("strip_left_short", j2[0], j2[1], "y") if j2 is not None
                             else on_table("strip_left_short")),
        "strip_right_outer": strip("strip_right_outer", MID_X, j3) if j3 is not None else on_table("strip_right_outer"),
        "strip_right_inner": strip("strip_right_inner", MID_X, j4) if j4 is not None else on_table("strip_right_inner"),
    }
    out.update(over)
    return out


def start():
    return {n: on_table(n) for n in NAMES}


def snap(left_tcp=L_PARK, left_open=1.0, right_tcp=R_PARK, right_open=1.0):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def score_of(strips, arms="park"):
    if arms == "park":
        arms = snap()
    return evaluate(strips, arms)[0]


def details_of(strips, arms="park"):
    if arms == "park":
        arms = snap()
    return evaluate(strips, arms)[1]


def settle(tracker, strips, n=12, arms="park"):
    """Feed the same state n steps (still)."""
    res = None
    for _ in range(n):
        res = tracker.update(strips, snap() if arms == "park" else arms)
    return res


def lines_of(*segments):
    """Joint lines from (axis, across, from, to) tuples, for panels()."""
    return list(segments)


class Geometry(unittest.TestCase):
    def test_rule_numbers(self):
        self.assertAlmostEqual(L.AREA, 0.0672)
        self.assertAlmostEqual(L.LONG[1] - L.LONG[0], 0.120)
        self.assertAlmostEqual(L.RETURN[3] - L.RETURN[2], 0.100)
        self.assertAlmostEqual(L.LONG[1] - L.LONG[0] - L.LONG_LEN, 0.008)  # 4 mm of play at each end
        slot = L.FACES_Y[0][0] - (CY + W / 2)  # from a strip out of the corner to the left form
        self.assertAlmostEqual(slot, 0.091)
        self.assertAlmostEqual(slot - L.SHORT_LEN, 0.008)  # the short strip: 4 mm at each end
        self.assertLess(L.RETURN[3] - L.RETURN[2], L.LONG_LEN)  # a long strip does not fit across the return leg
        # the short strip falls well short across the long leg (18.5 mm a side, 10.5 mm past the limit)
        self.assertAlmostEqual((0.120 - L.SHORT_LEN) / 2, 0.0185)
        self.assertGreater((0.120 - L.SHORT_LEN) / 2 - L.END_TOL, 0.010)
        # two strips make at most three panels, and three good panels cover at most 3 x 132^2 mm2 < the slab
        self.assertGreater(L.AREA, 3 * L.MAX_SIDE ** 2)
        self.assertAlmostEqual(L.MAX_SIDE * L.SCALE, 3.3)
        # rigid PVC (about 1.38 g/cm3): the stated masses match the bars' volumes within 1.5 g
        for length, mass, _, _ in L.STRIPS.values():
            self.assertAlmostEqual(mass, length * L.STRIP_W * L.STRIP_T * 1380, delta=0.0015)

    def test_forms_meet_without_overlap_and_their_faces_are_the_scorers(self):
        boxes = [(n, x0, x1, y0, y1) for n, x0, x1, y0, y1 in L.FORMS]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                ox = min(a[2], b[2]) - max(a[1], b[1])
                oy = min(a[4], b[4]) - max(a[3], b[3])
                self.assertFalse(ox > 1e-9 and oy > 1e-9, (a[0], b[0]))
        f = {n: (x0, x1, y0, y1) for n, x0, x1, y0, y1 in L.FORMS}
        self.assertAlmostEqual(f["form_far"][0], L.LONG[1])
        self.assertAlmostEqual(f["form_left"][2], L.LONG[3])
        self.assertAlmostEqual(f["form_near"][1], L.LONG[0])
        self.assertAlmostEqual(f["form_near"][3], CY)
        self.assertAlmostEqual(f["form_return_side"][3], L.RETURN[2])
        self.assertAlmostEqual(f["form_return_end"][1], L.RETURN[0])
        self.assertAlmostEqual(f["form_right_end"][3], L.LONG[2])
        # the walls the scorer slides ends into are those faces
        self.assertIn((L.LONG[1], L.LONG[2], L.LONG[3]), L.FACES_X)
        self.assertIn((L.LONG[0], L.LONG[2], CY), L.FACES_X)
        self.assertIn((CY, L.RETURN[0], L.RETURN[1]), L.FACES_Y)

    def test_start_layout(self):
        for n in NAMES:
            info = strip_info(on_table(n), n, snap())
            self.assertFalse(info["inside"], n)
            self.assertFalse(info["candidate"], n)
        self.assertEqual(score_of(start()), 0.0)

    def test_tilt_axis_lowest(self):
        n = "strip_left_long"
        self.assertAlmostEqual(tilt_deg(strip(n, MID_X, CY)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(strip(n, MID_X, CY, roll=180)), 0.0, places=5)  # upside down: still flat
        self.assertAlmostEqual(tilt_deg(strip(n, MID_X, CY, roll=4)), 4.0, places=5)
        self.assertAlmostEqual(tilt_deg(strip(n, MID_X, CY, edge=True)), 90.0, places=5)
        self.assertEqual(axis_of(strip(n, MID_X, CY))[0], "x")
        self.assertEqual(axis_of(strip(n, MID_X, CY, turn=180))[0], "x")
        ax, turn = axis_of(strip(n, MID_X, CY, "y", turn=-3))
        self.assertEqual(ax, "y")
        self.assertAlmostEqual(turn, 3.0, places=5)
        self.assertEqual(axis_of(strip(n, MID_X, CY, on_end=True)), (None, 90.0))
        self.assertAlmostEqual(lowest(strip(n, MID_X, CY, lift=0.002), n), L.SUBBASE_H + 0.002)

    def test_snap(self):
        self.assertEqual(snapped("x", CY + 0.006), CY)
        self.assertEqual(snapped("x", CY - 0.006), CY)
        self.assertAlmostEqual(snapped("x", CY + 0.0065), CY + 0.0065)
        self.assertEqual(snapped("y", L.CORNER[0] - 0.005), L.CORNER[0])
        self.assertAlmostEqual(snapped("y", 0.200), 0.200)

    def test_finger_zone(self):
        arms = snap()
        self.assertAlmostEqual(zone_radius(arms["left"]), math.hypot(L.FULL_GAP / 2 + 0.010, 0.020))
        arms = snap(left_open=0.0)
        self.assertAlmostEqual(zone_radius(arms["left"]), math.hypot(0.010, 0.020))


class Panels(unittest.TestCase):
    def test_no_joints_one_l_shaped_panel(self):
        p = panels([])
        self.assertEqual(len(p), 1)
        self.assertFalse(p[0]["rect"])
        self.assertAlmostEqual(p[0]["area"], L.AREA)

    def test_the_right_layout_five_good_panels(self):
        p = panels(lines_of(("x", CY, 0.260, 0.380), ("y", 0.260, CY, LEFT), ("x", 0.0, 0.260, 0.380),
                            ("x", -0.120, 0.260, 0.380)))
        self.assertEqual(len(p), 5)
        for q in p:
            self.assertTrue(q["good"], q)
            self.assertAlmostEqual(q["long"], 0.120)
        # three 120 mm squares and two 120 x 100 mm panels (the return leg and the long leg's end beside it)
        self.assertEqual(sorted(round(q["short"], 4) for q in p), [0.1, 0.1, 0.12, 0.12, 0.12])

    def test_a_line_between_the_legs_off_the_corner_still_rectangles(self):
        # between the legs at x 0.250 (in the return leg): panels 110 x 100 and 130 x 100, both good
        p = panels(lines_of(("x", CY, 0.260, 0.380), ("y", 0.250, CY, LEFT)))
        rects = sorted((round(q["long"], 4), round(q["short"], 4), q["good"]) for q in p if q["rect"])
        self.assertIn((0.11, 0.1, True), rects)
        self.assertIn((0.13, 0.1, True), rects)

    def test_a_dangling_line_divides_nothing(self):
        p = panels(lines_of(("x", 0.0, 0.300, 0.380)))  # from the far form, stopping short: no new panel
        self.assertEqual(len(p), 1)

    def test_aspect_and_length_limits_inclusive(self):
        p = {round(q["short"], 4): q["good"] for q in panels(lines_of(("x", -0.160, 0.260, 0.380)))}
        self.assertTrue(p[0.08])  # 120 x 80: exactly 1.5
        p = [q for q in panels(lines_of(("x", -0.1605, 0.260, 0.380))) if q["rect"]]
        self.assertFalse(p[0]["good"])  # 120 x 79.5
        p = [q for q in panels(lines_of(("x", -0.108, 0.260, 0.380))) if q["rect"]]
        self.assertTrue(p[0]["good"])  # 132 x 120
        p = [q for q in panels(lines_of(("x", -0.1075, 0.260, 0.380))) if q["rect"]]
        self.assertFalse(p[0]["good"])  # 132.5 x 120


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(start()), 0.0)

    def test_the_right_layout_scores_one(self):
        d = details_of(slab())
        self.assertEqual(d["good_panels"], 5)
        self.assertEqual(score_of(slab()), 1.0)
        for n in NAMES:
            self.assertTrue(d[n]["counts"], n)
        self.assertEqual(d["strip_left_short"]["ends_meet"], ["strip_left_long", "form"])
        self.assertEqual(d["strip_left_long"]["ends_meet"], ["form", "form"])
        self.assertEqual(d["strip_left_short"]["end_gaps_mm"], [4.0, 4.0])

    def test_every_tier_as_presented(self):
        cases = {
            "nothing laid": (slab(None, None, None, None), 0.0),
            "one strip 120 mm from the right end": (slab(None, None, None, -0.120), 0.2),
            "out of the corner + between the legs": (slab(CY, (0.260, J2_Y), None, None), 0.4),
            "two across at the right end": (slab(None, None, 0.0, -0.120), 0.4),
            "corner pair + one across at y 0": (slab(CY, (0.260, J2_Y), 0.0, None), 0.6),
            "three across, none between the legs": (slab(CY, None, 0.0, -0.120), 0.6),
            "all four, one across 30 mm off": (slab(j3=-0.030), 0.8),
            "the right layout": (slab(), 1.0),
        }
        for label, (s, want) in cases.items():
            self.assertAlmostEqual(score_of(s), want, msg=label)
        self.assertEqual(sorted({w for _, w in cases.values()}), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test_one_arm_each_scores_point_four(self):
        # left alone: its long strip out of the corner and the short one between the legs
        self.assertAlmostEqual(score_of(slab(CY, (0.260, J2_Y), None, None)), 0.4)
        # right alone: its two long strips across the right part of the long leg
        self.assertAlmostEqual(score_of(slab(None, None, 0.0, -0.120)), 0.4)
        # two strips can never make three good panels: whatever two lines, at most 0.4
        rng = random.Random(1)
        for _ in range(200):
            ya, yb = rng.uniform(-0.23, 0.23), rng.uniform(-0.23, 0.23)
            if abs(ya - yb) < 0.02:
                continue
            s = slab(None, None, ya, yb)
            self.assertLessEqual(score_of(s), 0.4 + 1e-9)

    def test_cross_strip_tolerance(self):
        for y3, want in ((0.012, 1.0), (-0.012, 1.0), (0.0125, 0.8), (-0.0125, 0.8)):
            self.assertAlmostEqual(score_of(slab(j3=y3)), want, msg=y3)
        for y4, want in ((-0.108, 1.0), (-0.132, 1.0), (-0.1075, 0.8), (-0.1325, 0.8)):
            self.assertAlmostEqual(score_of(slab(j4=y4)), want, msg=y4)

    def test_the_corner_strip_within_six_mm_of_the_corner(self):
        # above the corner by 6 mm: taken to run out of it (the short strip moved up to clear it)
        s = slab(CY + 0.006, (0.260, J2_Y + 0.003))
        self.assertEqual(score_of(s), 1.0)
        # 7 mm above: the corner is open, the return leg runs into the panel below it: L-shaped, 0.6
        s = slab(CY + 0.007, (0.260, J2_Y + 0.0035))
        d = details_of(s)
        self.assertTrue(all(d[n]["counts"] for n in NAMES))
        self.assertAlmostEqual(score_of(s), 0.6)
        # 7 and 8 mm below: the strip between the legs then runs out of the corner, every panel a rectangle
        s = slab(CY - 0.007, (0.260, J2_Y - 0.0035))
        self.assertEqual(score_of(s), 1.0)
        s = slab(CY - 0.008, (0.260, J2_Y - 0.004))  # the short strip 8 mm from it and from the left form
        self.assertEqual(score_of(s), 1.0)
        # 9 mm below, the short strip cannot reach within 8 mm of both it and the left form
        s = slab(CY - 0.009, (0.260, J2_Y - 0.0045))
        self.assertFalse(details_of(s)["strip_left_short"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.6)

    def test_between_the_legs_where_it_may_go(self):
        # it meets the corner strip (its end within 8 mm of the corner strip's outline) from x 24.9 to 27.2 cm, and
        # both panels it separates stay within 132 mm (at x 27.2 cm the return leg's panel is exactly 132 mm long)
        for x in (0.249, 0.254, 0.260, 0.266, 0.271, 0.272):
            self.assertEqual(score_of(slab(j2=(x, J2_Y))), 1.0, x)
        # 13 mm into the return leg: 8.9 mm from the corner strip's end, 13 mm from the return's side form
        d = details_of(slab(j2=(0.247, J2_Y)))
        self.assertFalse(d["strip_left_short"]["counts"])
        self.assertAlmostEqual(score_of(slab(j2=(0.247, J2_Y))), 0.6)
        # 12.5 and 13 mm into the long leg: it counts, but the return leg's panel is then over 132 mm long
        self.assertAlmostEqual(score_of(slab(j2=(0.2725, J2_Y))), 0.8)
        self.assertAlmostEqual(score_of(slab(j2=(0.273, J2_Y))), 0.8)

    def test_end_gaps_eight_mm_inclusive(self):
        for dy, want in ((0.004, 1.0), (-0.004, 1.0), (0.0045, 0.6), (-0.0045, 0.6)):
            s = slab(j2=(0.260, J2_Y + dy))
            self.assertAlmostEqual(score_of(s), want, msg=dy)

    def test_flat_within_five_degrees(self):
        s = slab()
        s["strip_right_outer"] = strip("strip_right_outer", MID_X, 0.0, roll=5)
        self.assertEqual(score_of(s), 1.0)
        s["strip_right_outer"] = strip("strip_right_outer", MID_X, 0.0, roll=6)
        self.assertAlmostEqual(score_of(s), 0.6)  # J1-to-J4 is 240 mm: two good panels lost, one bad made

    def test_square_within_five_degrees(self):
        s = slab()
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.120, turn=5)
        self.assertEqual(score_of(s), 1.0)
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.120, turn=6)
        self.assertAlmostEqual(score_of(s), 0.6)

    def test_on_the_sub_base_within_three_mm(self):
        s = slab()
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.120, lift=0.003)
        self.assertEqual(score_of(s), 1.0)
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.120, lift=0.0035)
        self.assertAlmostEqual(score_of(s), 0.6)

    def test_inside_the_slab_within_two_mm(self):
        n = "strip_right_inner"
        self.assertTrue(inside(strip(n, MID_X, -0.240 + W / 2 - 0.002), n))
        self.assertFalse(inside(strip(n, MID_X, -0.240 + W / 2 - 0.0025), n))

    def test_let_go(self):
        s = slab()
        over_j3 = np.array([MID_X, 0.0, BASE + L.STRIP_T + 0.010])  # gripper just above J3, closed on it
        self.assertAlmostEqual(score_of(s, snap(right_tcp=over_j3, right_open=0.2)), 0.6)
        high = np.array([MID_X, 0.0, TOP + 0.10])  # risen after letting go (open 40 mm): clear
        self.assertEqual(score_of(s, snap(right_tcp=high, right_open=0.43)), 1.0)
        d = details_of(s, snap(right_tcp=high, right_open=0.43))
        self.assertGreater(d["strip_right_outer"]["right_zone_mm"], 20.0)

    def test_arms_unavailable_taken_as_met(self):
        self.assertEqual(score_of(slab(), None), 1.0)

    def test_success_needs_stillness_and_holds(self):
        tr = Tracker(HZ)
        res = settle(tr, slab(), 10)
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.success)  # 0.5 s not seen yet
        res = tr.update(slab(), snap())
        self.assertTrue(res.success)
        self.assertEqual(tr.first_full_step, 1)
        moving = slab(j3=0.003)
        res = tr.update(moving, snap())
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.success)  # J3 jumped 3 mm in the last 0.5 s
        res = settle(tr, moving, 11)
        self.assertTrue(res.success)

    def test_score_is_idempotent(self):
        tr = Tracker(HZ)
        settle(tr, slab(j3=-0.030), 3)
        a = tr.result(slab(j3=-0.030), snap()).score
        b = tr.result(slab(j3=-0.030), snap()).score
        self.assertEqual(a, b)
        self.assertEqual(tr.steps, 3)


class ZeroConditions(unittest.TestCase):
    def test_strips_left_on_the_table(self):
        self.assertEqual(score_of(start()), 0.0)

    def test_lying_on_an_edge_or_standing_on_end(self):
        s = {n: strip(n, *L.STRIPS[n][2:], floor=TOP) for n in NAMES}
        self.assertEqual(score_of(s), 0.0)
        s = slab()
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.120, edge=True)
        s["strip_right_outer"] = strip("strip_right_outer", MID_X, 0.0, edge=True)
        s["strip_left_long"] = strip("strip_left_long", MID_X, CY, edge=True)
        s["strip_left_short"] = strip("strip_left_short", 0.260, J2_Y, "y", on_end=True)
        self.assertEqual(score_of(s), 0.0)

    def test_on_top_of_a_form(self):
        s = slab(None, None, None, None)
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.245, floor=TOP + L.FORM_H)
        self.assertEqual(score_of(s), 0.0)
        info = strip_info(s["strip_right_inner"], "strip_right_inner", snap())
        self.assertFalse(info["candidate"])

    def test_held_in_the_air_over_its_line(self):
        s = slab(None, None, None, None)
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, -0.120, lift=0.030)
        self.assertEqual(score_of(s), 0.0)

    def test_a_strip_along_a_form_divides_nothing(self):
        # along the left form in the long leg: its near end points into the return leg, 124 mm from the return's end
        s = slab(None, None, None, None)
        s["strip_right_inner"] = strip("strip_right_inner", MID_X, LEFT - W / 2 - 0.001)
        d = details_of(s)
        self.assertFalse(d["strip_right_inner"]["counts"])
        self.assertEqual(score_of(s), 0.0)
        # along the right end form, 1 mm off it: it counts (both ends 4 mm from a form) but cuts off a 10 mm
        # sliver, not a good panel
        s["strip_right_outer"] = strip("strip_right_outer", MID_X, -0.240 + W / 2 + 0.001)
        d = details_of(s)
        self.assertTrue(d["strip_right_outer"]["counts"])
        self.assertEqual(d["good_panels"], 0)
        self.assertEqual(score_of(s), 0.0)
        # along the return's end form a long strip does not fit (112 mm in 100 mm); the short one, anywhere across the
        # return leg, is 8.5 mm from each form at best
        s = slab(None, None, None, None)
        s["strip_left_long"] = strip("strip_left_long", 0.140 + W / 2 + 0.001, (CY + LEFT) / 2, "y")
        s["strip_left_short"] = strip("strip_left_short", 0.140 + W / 2 + 0.001 + 0.040, (CY + LEFT) / 2, "y")
        d = details_of(s)
        self.assertFalse(d["strip_left_long"]["inside"])
        self.assertFalse(d["strip_left_short"]["counts"])
        self.assertEqual(d["strip_left_short"]["end_gaps_mm"], [8.5, 8.5])
        self.assertEqual(score_of(s), 0.0)


class Exploits(unittest.TestCase):
    def test_stacked_strips(self):
        s = slab(None, None, None, -0.120)
        s["strip_right_outer"] = strip("strip_right_outer", MID_X, -0.120, floor=BASE + L.STRIP_T)
        d = details_of(s)
        self.assertFalse(d["strip_right_outer"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.2)

    def test_four_across_the_long_leg_none_between_the_legs(self):
        # the short strip cannot span the long leg (18.5 mm from a form at best): three across, 0.6 at most
        s = slab(CY, None, 0.0, -0.120)
        s["strip_left_short"] = strip("strip_left_short", MID_X, -0.060)
        d = details_of(s)
        self.assertFalse(d["strip_left_short"]["counts"])
        self.assertEqual(d["strip_left_short"]["end_gaps_mm"], [18.5, 18.5])
        self.assertAlmostEqual(score_of(s), 0.6)
        rng = random.Random(7)
        for _ in range(200):  # any three lines across the long leg and none between the legs: at most 0.6
            ys = sorted(rng.uniform(-0.22, 0.22) for _ in range(3))
            if min(b - a for a, b in zip(ys, ys[1:])) < 0.02:
                continue
            s = slab(ys[2], None, ys[1], ys[0])
            self.assertLessEqual(score_of(s), 0.6 + 1e-9)

    def test_short_strip_between_the_legs_without_the_corner_strip(self):
        # nothing for its lower end to meet within 8 mm: it divides nothing
        s = slab(None, (0.260, J2_Y), 0.0, -0.120)
        self.assertFalse(details_of(s)["strip_left_short"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.4)

    def test_a_long_strip_does_not_fit_between_the_legs(self):
        # 112 mm across the return leg's 100 mm: it lies 6 mm over a form at each end, outside the slab
        s = slab(None, None, 0.0, -0.120)
        s["strip_left_long"] = strip("strip_left_long", 0.260, (CY + LEFT) / 2, "y")
        d = details_of(s)
        self.assertFalse(d["strip_left_long"]["inside"])
        self.assertFalse(d["strip_left_long"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.4)
        # the short strip centred between the legs with no corner strip: 8.5 mm from each form, it divides nothing
        s["strip_left_short"] = strip("strip_left_short", 0.260, (CY + LEFT) / 2, "y")
        d = details_of(s)
        self.assertEqual(d["strip_left_short"]["end_gaps_mm"], [8.5, 8.5])
        self.assertFalse(d["strip_left_short"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.4)

    def test_held_still_on_its_line(self):
        s = slab()
        at_j1 = np.array([MID_X, CY, BASE + 0.016])  # the left gripper still closed round J1
        d = details_of(s, snap(left_tcp=at_j1, left_open=0.19))
        self.assertFalse(d["strip_left_long"]["let_go"])
        self.assertTrue(d["strip_left_short"]["let_go"])
        self.assertFalse(d["strip_left_short"]["counts"])  # its lower end has nothing to meet without J1
        self.assertAlmostEqual(score_of(s, snap(left_tcp=at_j1, left_open=0.19)), 0.4)

    def test_a_dropped_candidate_takes_those_meeting_it_with_it(self):
        s = slab()
        s["strip_left_long"] = strip("strip_left_long", MID_X, CY, turn=6)  # out of square: not a candidate
        d = details_of(s)
        self.assertFalse(d["strip_left_long"]["counts"])
        self.assertFalse(d["strip_left_short"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.4)

    def test_a_strip_meeting_only_a_strip_that_fails_its_ends_falls_too(self):
        # the corner strip slid 5 mm toward the near form: 1 mm into it (contact) but 9 mm short of the far form. It
        # is dropped, and then the short strip, whose lower end only met it, is dropped on the next pass
        s = slab()
        s["strip_left_long"] = strip("strip_left_long", MID_X - 0.005, CY)
        d = details_of(s)
        self.assertFalse(d["strip_left_long"]["counts"])
        self.assertEqual(d["strip_left_long"]["end_gaps_mm"], [-1.0, 9.0])
        self.assertFalse(d["strip_left_short"]["counts"])
        self.assertAlmostEqual(score_of(s), 0.4)

    def test_the_score_never_passes_one(self):
        rng = random.Random(3)
        for _ in range(300):
            s = slab(rng.choice([None, rng.uniform(-0.22, 0.22)]), rng.choice([None, (rng.uniform(0.15, 0.37),
                                                                                       rng.uniform(-0.18, 0.18))]),
                     rng.choice([None, rng.uniform(-0.22, 0.22)]), rng.choice([None, rng.uniform(-0.22, 0.22)]))
            v = score_of(s)
            self.assertIn(round(v, 6), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))


class EndsAndLines(unittest.TestCase):
    def test_end_meets(self):
        j1 = Bar("a", "x", MID_X, CY, CY, L.LONG_LEN / 2)
        j2 = Bar("b", "y", J2_Y, 0.260, 0.260, L.SHORT_LEN / 2)
        g, what, to = end_meets(j1, -1, [j1, j2])
        self.assertEqual((what, to), ("form", 0.260))
        self.assertAlmostEqual(g, 0.004)
        g, what, to = end_meets(j2, -1, [j1, j2])
        self.assertEqual((what, to), ("a", CY))
        self.assertAlmostEqual(g, 0.004)
        g, what, to = end_meets(j2, 1, [j1, j2])
        self.assertEqual((what, to), ("form", LEFT))
        keep, ends = counted([j1, j2])
        self.assertEqual({b.name for b in keep}, {"a", "b"})
        self.assertEqual(sorted(joint_lines(keep, ends)),
                         sorted([("x", CY, 0.260, 0.380), ("y", 0.260, CY, LEFT)]))

    def test_an_end_in_contact_may_be_up_to_two_mm_in(self):
        far = L.LONG[1]
        for into, met in ((0.001, True), (0.002, True), (0.0025, False)):
            b = Bar("a", "x", far + into - 0.056, 0.0, 0.0, 0.056)
            g, what, to = end_meets(b, 1, [b])
            self.assertEqual(what == "form" and g <= L.END_TOL, met, into)
            if met:
                self.assertAlmostEqual(to, far)
        j1 = Bar("a", "x", MID_X, CY, CY, L.LONG_LEN / 2)
        j2 = Bar("b", "y", CY + W / 2 - 0.001 + L.SHORT_LEN / 2, 0.260, 0.260, L.SHORT_LEN / 2)  # 1 mm into J1
        g, what, to = end_meets(j2, -1, [j1, j2])
        self.assertEqual((what, to), ("a", CY))
        self.assertAlmostEqual(g, 0.0)

    def test_two_strips_meeting_at_a_corner_hold_each_other(self):
        # slid 5 mm toward the far form, the corner strip's near end is 9 mm from the near form but level with the
        # short strip's end, 4 mm below it: each meets the other, and the lines are those of the right layout
        s = slab()
        s["strip_left_long"] = strip("strip_left_long", MID_X + 0.005, CY)
        d = details_of(s)
        self.assertEqual(d["strip_left_long"]["ends_meet"], ["strip_left_short", "form"])
        self.assertEqual(score_of(s), 1.0)

    def test_end_to_end_strips(self):
        a = Bar("a", "x", 0.140 + 0.004 + 0.056, 0.180, 0.180, 0.056)
        b = Bar("b", "x", 0.380 - 0.004 - 0.056, 0.180, 0.180, 0.056)
        keep, ends = counted([a, b])
        self.assertEqual(len(keep), 2)
        self.assertEqual(ends["a"][1][1], "b")


class Arms(unittest.TestCase):
    def test_unavailable_counted(self):
        tr = Tracker(HZ)
        tr.update(slab(), None)
        tr.update(slab(), snap())
        self.assertEqual(tr.arms_unavailable, 1)

    def test_offline_feed_reaches_the_scorer(self):
        arms = snap(left_tcp=(0.1, 0.2, 0.9))
        np.testing.assert_allclose(arms["left"].tcp, (0.1, 0.2, 0.9))

    def test_zone_gap_sign(self):
        s = strip("strip_left_long", MID_X, CY)
        self.assertLess(zone_gap(snap(left_tcp=(MID_X, CY, BASE + 0.016), left_open=0.19)["left"], s,
                                 "strip_left_long"), 0.0)
        self.assertGreater(zone_gap(snap()["left"], s, "strip_left_long"), 0.05)

    def test_bad_arms_are_unreadable(self):
        bad = {"left": SimpleNamespace(tcp=[np.nan, 0, 0], tips=[0, 0, 0], opening=1.0),
               "right": SimpleNamespace(tcp=[0, 0, 0], tips=[0, 0, 0], opening=1.0)}
        self.assertEqual(score_of(slab(), bad), 1.0)  # taken as unavailable: let go met
        self.assertEqual(score_of(slab(), {"left": None}), 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr = Tracker(HZ)
        settle(tr, slab(), 12)
        hist = {n: len(h) for n, h in tr.history.items()}
        prev = dict(tr.prev)
        for bad in ("pos", "quat_xyzw", "lin_vel", "low"):
            s = slab()
            v = np.array(getattr(s["strip_right_inner"], bad), float)
            v[0] = np.nan
            setattr(s["strip_right_inner"], bad, v)
            res = tr.update(s, snap())
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
            self.assertEqual(tr.result(s, snap()).score, 0.0)
        s = slab()
        s["strip_right_inner"].quat_xyzw = np.zeros(4)
        self.assertEqual(tr.update(s, snap()).score, 0.0)
        s = slab()
        del s["strip_right_inner"]
        self.assertEqual(tr.update(s, snap()).score, 0.0)
        self.assertEqual(tr.bad_steps, 6)
        self.assertEqual({n: len(h) for n, h in tr.history.items()}, hist)
        self.assertEqual(set(tr.prev), set(prev))
        self.assertTrue(tr.update(slab(), snap()).success)  # the record carried on unharmed
        self.assertFalse(valid(None))
        self.assertFalse(valid(SimpleNamespace(pos=[0, 0])))

    def test_unnormalised_quaternion(self):
        s = slab()
        s["strip_right_inner"].quat_xyzw = s["strip_right_inner"].quat_xyzw * 3.0
        self.assertEqual(score_of(s), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(5)
        tr = Tracker(HZ)
        for _ in range(300):
            s = {}
            for n in NAMES:
                q = rng.normal(size=4)
                c = rng.uniform((0.0, -0.6, TOP - 0.05), (0.6, 0.6, TOP + 0.3))
                s[n] = raw(n, c, q)
            res = tr.update(s, snap())
            self.assertIn(round(res.score, 6), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr = Tracker(HZ)
        settle(tr, slab(), 12)
        self.assertEqual(tr.steps, 12)
        tr.update(start(), snap())
        self.assertEqual(tr.steps, 1)
        self.assertIsNone(tr.first_full_step)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker(HZ)
        _, _, x0, y0 = L.STRIPS["strip_right_inner"]
        for k in range(30):  # one strip carried back to its start 8 mm per step, the others there already
            f = 1 - k / 29
            s = start()
            s["strip_right_inner"] = strip("strip_right_inner", x0 + 0.09 * f, y0 + 0.22 * f, floor=TOP)
            tr.update(s, snap())
        self.assertEqual(tr.steps, 30)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr = Tracker(HZ)
        settle(tr, start(), 3)
        tr.update(slab(), snap())
        self.assertEqual(tr.steps, 4)

    def test_rigstate_reset_hook_runs_registered_callbacks(self):
        saved = dict(rigstate._RESET_CALLBACKS)
        hits = []
        try:
            rigstate.on_reset(lambda: hits.append(1))
            rigstate.offline_reset()
            self.assertEqual(hits, [1])
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook, the demo, the
    task.json text, and the offline checker on the whole folder."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")

    def test_objects(self):
        objs = self.task.OBJECTS
        names = [o.name for o in objs]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in objs if o.mass is not None], list(NAMES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        spec = {o.name: o for o in objs}
        for n, (length, mass, x, y) in L.STRIPS.items():
            np.testing.assert_allclose(spec[n].pos, (x, y, TOP + L.STRIP_T / 2))
            self.assertEqual(tuple(spec[n].size), (length, W, L.STRIP_T))
            self.assertLessEqual(spec[n].mass, 0.3)
        for n, x0, x1, y0, y1 in L.FORMS:
            o = spec[n]
            self.assertAlmostEqual(o.pos[0] - o.size[0] / 2, x0)
            self.assertAlmostEqual(o.pos[1] + o.size[1] / 2, y1)
            self.assertIsNone(o.mass)
        for n, x0, x1, y0, y1 in L.SUBBASES:
            o = spec[n]
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, BASE)
            self.assertAlmostEqual(o.size[0], x1 - x0)
        # the arms' start pose covers the table near each base (|y| > 0.45 m, x < 0.2 m); nothing of the task there
        for s in objs:
            self.assertLess(abs(s.pos[1]) + s.size[1] / 2, 0.45, s.name)  # every box here is square to the axes

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(start()))
        self.assertEqual(self.task.score(start()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(slab()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        for _ in range(11):
            snap()
            ok = self.task.success(slab())
        self.assertTrue(ok)
        self.assertEqual(self.task.score(slab()), 1.0)
        self.assertFalse(self.task.success({"strip_left_long": strip("strip_left_long", MID_X, CY)}))
        self.assertEqual(self.task.score({"strip_left_long": strip("strip_left_long", MID_X, CY)}), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        tmp = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(tmp)
            rigstate.offline_reset()
            snap()
            self.task.success(slab())
            row = json.loads(tmp.read_text().splitlines()[-1])
            for key in ("score", "good_panels", "panels", "lines_mm", "still", "moved_mm", "strip_left_short",
                        "strip_left_short_pos", "left_tcp", "left_opening", "arms_unavailable", "first_full_step"):
                self.assertIn(key, row)
            for key in ("counts", "why", "line_mm", "line_raw_mm", "end_gaps_mm", "ends_meet", "tilt_deg",
                        "turn_deg", "low_mm", "inside", "let_go", "left_zone_mm"):
                self.assertIn(key, row["strip_left_short"])
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(slab()))
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
        # every open or close takes 1.5 s or more; every final set-down is slow
        for wps in plan.values():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
            for w in wps:
                if w.label.startswith("slowly down"):
                    self.assertGreaterEqual(w.duration, 2.0)
                    self.assertAlmostEqual(w.pos[2] - t.GRIP_UP, BASE + 0.0005)
        places = {arm: [w for w in plan[arm] if w.label.startswith("slowly down")] for arm in plan}
        self.assertEqual([tuple(np.round(w.pos[:2], 4)) for w in places["left"]],
                         [(MID_X, CY), (0.260, round(J2_Y, 4))])
        self.assertEqual([tuple(np.round(w.pos[:2], 4)) for w in places["right"]], [(MID_X, -0.12), (MID_X, 0.0)])
        self.assertAlmostEqual(places["left"][1].yaw, -math.pi / 2)
        self.assertAlmostEqual(t.J2_GAP, 0.004)
        # grips 8 mm narrower than the 18 mm width; open 11 mm a side
        self.assertAlmostEqual(-t.GRIP * 2, W - 0.008)
        self.assertAlmostEqual(-t.PICK_OPEN * 2 - W, 0.022)
        # the pad tips stay at least 3 mm off the table; a carried strip clears the forms by 50 mm or more
        self.assertGreaterEqual(t.PICK_Z - TIPS - TOP, 0.003 - 1e-9)
        self.assertGreaterEqual(t.CARRY_Z - t.GRIP_UP - (TOP + L.FORM_H), 0.050)

        # the left arm lays J2 only after the right arm has risen from J3
        t_right_up = sum(w.duration for w in plan["right"][:-1])
        t_left_carry = sum(w.duration for w in plan["left"][:next(i for i, w in enumerate(plan["left"])
                                                                  if w.label.startswith("carry the short"))])
        self.assertGreaterEqual(t_left_carry, t_right_up - 1e-9)
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
        self.assertEqual(d["category"], "Impact on jobs")
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        s = d["scoring"]
        for phrase in ("132 mm", "1.5 times", "8 mm", "6 mm", "5 degrees", "3 mm", "2 mm", "20 mm", "0.2",
                       "Possible scores: 0, 0.2, 0.4, 0.6, 0.8, 1.0"):
            self.assertIn(phrase, s)
        for _, _, x, y in L.STRIPS.values():
            self.assertIn(f"x {x * 100:.1f} cm, y {y * 100:+.1f} cm", d["setup"])
            self.assertIn(f"x {x * 100:.1f} cm, y {y * 100:+.1f} cm", d["reset_notes"])
        self.assertIn("1,600 steps", d["run_limit"])
        self.assertIn("TBD", d["scores"])

    def test_offline_checker_predicts_one(self):
        """tools/offline/check_task.py --strict --one-arm on this folder (no reach table, for time). Run in-process
        with sys.modules and sys.path restored afterwards (the checker reloads the task's helper modules)."""
        saved_modules, saved_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(_offline_dir()))
        try:
            import check_task
            rep, _, code = check_task.check(HERE, strict=True, one_arm=True, reach=False)
            self.assertEqual(code, 0, rep.fails)
            self.assertEqual(rep.data["playback"]["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(rep.data["one_arm"], {"left_only": 0.4, "right_only": 0.4})
            check_task._forget_task_modules()
        finally:
            for name in [n for n in sys.modules if n not in saved_modules]:
                del sys.modules[name]
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path


if __name__ == "__main__":
    unittest.main()
