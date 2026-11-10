"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import itertools
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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, bottom_gap, corners, evaluate, held, highest, lowest, marker_on_side, on_grass_outline, placed, quad,
    reach, rotation, square_score, standing, tilt_deg, upwind_sides, valid,
)

MM = 0.001
GX0, GX1, GY0, GY1 = L.GRASS


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A beacon's state: its world bounding box worked out from the turned cylinder."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = np.array([L.BEACON_H / 2 * abs(a[k]) + L.BEACON_RADIUS * math.sqrt(max(0.0, 1 - a[k] ** 2))
                    for k in range(3)])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def beacon(x, y, bottom=L.MAT_TOP, tilt=0.0, axis="x", **kw):
    """A beacon centred at (x, y) with its lowest point at ``bottom``, tilted ``tilt`` degrees about ``axis``."""
    q = about(axis, tilt)
    az = abs(rotation(q)[2, 2])
    half = L.BEACON_H / 2 * az + L.BEACON_RADIUS * math.sqrt(max(0.0, 1 - az * az))
    return state((x, y, bottom + half), q, **kw)


def start():
    return [beacon(*L.START[n]) for n in L.NAMES]


S0 = start()


def square_xy(side, cx=None, cy=None, turn=0.0):
    """Corners (x, y) of a square ``side`` m on a side centred at (cx, cy), turned ``turn`` degrees."""
    cx = (GX0 + GX1) / 2 if cx is None else cx
    cy = 0.0 if cy is None else cy
    r = side / math.sqrt(2)
    th = math.radians(turn)
    return [(cx + r * math.cos(th + math.pi / 4 + k * math.pi / 2), cy + r * math.sin(th + math.pi / 4 + k * math.pi / 2))
            for k in range(4)]


def upwind_mid(side, cx=None, cy=None, turn=0.0, out=0.030, t=0.5):
    """The point ``out`` m outside the upwind (left, +y) side of that square, a share ``t`` along it."""
    cx = (GX0 + GX1) / 2 if cx is None else cx
    cy = 0.0 if cy is None else cy
    th = math.radians(turn)
    n = np.array([-math.sin(th), math.cos(th)])  # the +y side's outward normal, turned
    e = np.array([math.cos(th), math.sin(th)])  # along it
    p = np.array([cx, cy]) + n * (side / 2 + out) + e * (t - 0.5) * side
    return tuple(p)


def zone(corner_xy, blue_xy, **over):
    """Four orange beacons at the corners, the blue at ``blue_xy``; ``over``: name -> state replaces that beacon."""
    out = [beacon(x, y) for x, y in corner_xy] + [beacon(*blue_xy)]
    for name, s in over.items():
        out[L.NAMES.index(name)] = s
    return out


# The demo's zone: 292 x 292 mm, its left (upwind) side at y +146 mm, the blue 30 mm outside its middle.
DEMO_CORNERS = [(0.148, 0.146), (0.440, 0.146), (0.148, -0.146), (0.440, -0.146)]
DEMO_BLUE = (0.294, 0.176)
DONE = zone(DEMO_CORNERS, DEMO_BLUE)
# A 260 mm zone (between 90 and 100 ft) centred on the field's x middle, y 0.
SMALL = 0.260


def score_of(bs):
    return evaluate(bs)[0]


def score_of_arms(bs, arms):
    return evaluate(bs, arms)[0]


def glide(a, b, n):
    """n frames moving every beacon of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None, hold=30):
    """From the start, every beacon moved gradually to the demo's zone (or ``final``), then held there."""
    final = DONE if final is None else final
    return [S0] + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0, arms=None):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


ON = 0.028 / L.FULL_GAP  # rigstate's opening with the fingers closed on a 28 mm beacon


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    """A rigstate.arms() snapshot: each given arm's pad-tip point (x, y, z) and opening (0 closed .. 1 open)."""
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


# Both arms parked where the demo leaves them: over their own side's emptied road or strip, pad tips 107 mm up.
PARKED = snap(left=(0.200, 0.250, L.TOP + 0.107), right=(0.200, -0.250, L.TOP + 0.107))


class Geometry(unittest.TestCase):
    def test_lowest_point_and_tilt(self):
        s = beacon(0.3, 0.0)
        self.assertAlmostEqual(lowest(s), L.MAT_TOP)
        self.assertAlmostEqual(bottom_gap(s), 0.0)
        self.assertAlmostEqual(tilt_deg(s.quat_xyzw), 0.0)
        lying = beacon(0.3, 0.0, tilt=90.0)
        self.assertAlmostEqual(lying.pos[2] - L.MAT_TOP, L.BEACON_RADIUS)  # on its side: the radius up
        self.assertAlmostEqual(tilt_deg(lying.quat_xyzw), 90.0)
        self.assertAlmostEqual(lowest(lying), L.MAT_TOP)
        leaning = beacon(0.3, 0.0, tilt=45.0)  # resting on its rim: corner 19.8 mm below the centre
        self.assertAlmostEqual(lowest(leaning), L.MAT_TOP)
        self.assertAlmostEqual(leaning.pos[2] - L.MAT_TOP, L.BEACON_RADIUS * math.sqrt(2))
        self.assertAlmostEqual(highest(leaning) - L.MAT_TOP, 2 * L.BEACON_RADIUS * math.sqrt(2))  # its top rim
        self.assertAlmostEqual(highest(lying) - L.MAT_TOP, 2 * L.BEACON_RADIUS)
        self.assertAlmostEqual(highest(s) - L.MAT_TOP, L.BEACON_H)
        flipped = beacon(0.3, 0.0, tilt=180.0)  # either end up
        self.assertAlmostEqual(tilt_deg(flipped.quat_xyzw), 0.0)
        self.assertAlmostEqual(lowest(flipped), L.MAT_TOP)
        self.assertAlmostEqual(tilt_deg(beacon(0.3, 0.0, tilt=7.0, axis="y").quat_xyzw), 7.0)

    def test_placed(self):
        self.assertTrue(placed(beacon(0.3, 0.0)))
        self.assertTrue(placed(beacon(0.3, 0.0, tilt=180.0)))
        self.assertTrue(placed(beacon(0.3, 0.0, tilt=10.0)))
        self.assertFalse(placed(beacon(0.3, 0.0, tilt=10.5)))
        self.assertFalse(placed(beacon(0.3, 0.0, tilt=90.0)))  # lying on its side
        self.assertTrue(placed(beacon(0.3, 0.0, bottom=L.MAT_TOP + 3 * MM)))
        self.assertFalse(placed(beacon(0.3, 0.0, bottom=L.MAT_TOP + 3.5 * MM)))  # held up
        self.assertFalse(placed(beacon(0.3, 0.0, bottom=L.MAT_TOP + L.BEACON_H)))  # stacked on another
        for name in L.NAMES:  # the start: on the road or the strip, not the grass
            self.assertFalse(placed(beacon(*L.START[name])), name)
        # The outline: centre on the edges counts, 1 mm past them does not.
        for x, y in ((GX0, 0.0), (GX1, 0.0), (0.3, GY0), (0.3, GY1)):
            self.assertTrue(on_grass_outline(beacon(x, y)))
        for x, y in ((GX0 - MM, 0.0), (GX1 + MM, 0.0), (0.3, GY0 - MM), (0.3, GY1 + MM)):
            self.assertFalse(placed(beacon(x, y)))

    def test_valid(self):
        self.assertTrue(valid(beacon(0.3, 0.0)))
        for bad in (state((np.nan, 0, 0.77)), state((0.3, 0, 0.77), vel=(np.inf, 0, 0)),
                    state((0.3, 0, 0.77), quat=(0, 0, 0, 0)), SimpleNamespace(pos=(0.3, 0, 0.77))):
            self.assertFalse(valid(bad))

    def test_corners_any_order(self):
        """The orange beacons are joined in order of angle round their centroid, whatever their names."""
        base = corners([(*p, 0.0) for p in DEMO_CORNERS])
        for perm in itertools.permutations(DEMO_CORNERS):
            q = corners([(*p, 0.0) for p in perm])
            self.assertEqual({tuple(np.round(r, 6)) for r in q}, {tuple(np.round(r, 6)) for r in base})
            z = quad([(*p, 0.0) for p in perm])
            self.assertTrue(z["convex"])
            for s in z["sides"]:
                self.assertAlmostEqual(s, 0.292)
            for a in z["angles"]:
                self.assertAlmostEqual(a, 90.0)

    def test_dart_is_not_convex(self):
        z = quad([(0.15, 0.15, 0), (0.45, 0.15, 0), (0.30, -0.15, 0), (0.30, 0.10, 0)])
        self.assertFalse(z["convex"])
        self.assertEqual(square_score(z), 0.0)

    def test_scale_and_layout_arithmetic(self):
        self.assertAlmostEqual(100 * 0.3048 / 110, L.SIDE_FULL, delta=0.0005)
        self.assertAlmostEqual(90 * 0.3048 / 110, L.SIDE_PART, delta=0.0005)
        self.assertEqual(L.WIND_FROM, (0.0, 1.0))
        # The windsock sleeve runs from the mast toward -y (downwind), its sections tapering.
        self.assertLess(L.SLEEVE_Y0 - len(L.SLEEVE_SIDES) * L.SLEEVE_LEN, L.MAST_Y)
        self.assertEqual(list(L.SLEEVE_SIDES), sorted(L.SLEEVE_SIDES, reverse=True))
        self.assertGreater(L.SLEEVE_Y0 - len(L.SLEEVE_SIDES) * L.SLEEVE_LEN, L.ROAD[3])  # its tip beyond the road
        self.assertGreaterEqual(len(L.SLEEVE_SIDES) * L.SLEEVE_LEN, 0.060)  # long enough to read from the top camera
        # Mats abut: the road and strip along the field's left and right edges.
        self.assertAlmostEqual(L.ROAD[2], GY1)
        self.assertAlmostEqual(L.STRIP[3], GY0)
        # The demo's zone fits on the grass with the blue on it too.
        for x, y in [*DEMO_CORNERS, DEMO_BLUE]:
            self.assertTrue(on_grass_outline(beacon(x, y)))
        # The largest zone the field holds with room for the marker (left side <= y +180 mm): 335 x 350 mm.
        self.assertGreater(GX1 - GX0, L.SIDE_FULL)
        self.assertGreater(GY1 - L.MARK_NEAR - GY0, L.SIDE_FULL)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["orange_not_placed"], list(L.ORANGE))

    def test_demo_zone_scores_one(self):
        score, d = evaluate(DONE)
        self.assertEqual(score, 1.0)
        self.assertEqual((d["square"], d["marker"]), (0.5, 0.5))
        self.assertEqual(d["sides_mm"], [292.0] * 4)
        self.assertAlmostEqual(d["blue_out_mm"], 30.0)
        self.assertAlmostEqual(d["blue_along"], 0.5)
        self.assertEqual(d["upwind_normal_deg"], 0.0)

    def test_blue_beside_a_corner_or_too_near_or_far_scores_three_quarters(self):
        for t in (0.0, 0.1, 0.3, 0.7, 0.95, 1.0):
            self.assertEqual(score_of(zone(DEMO_CORNERS, (0.148 + t * 0.292, 0.176))), 0.75, t)
        for out in (0.001, 0.005, 0.009, 0.056, 0.070):  # a 280 mm zone, its upwind side at y +115 mm
            bs = zone(square_xy(0.280, cy=-0.025), upwind_mid(0.280, cy=-0.025, out=out))
            self.assertTrue(placed(bs[4]))
            self.assertEqual(score_of(bs), 0.75, out)

    def test_ninety_foot_zone(self):
        """A 260 mm zone: 0.25 for the square, plus the marker."""
        c = square_xy(SMALL)
        self.assertEqual(score_of(zone(c, upwind_mid(SMALL))), 0.75)  # marker mid-side
        self.assertEqual(score_of(zone(c, upwind_mid(SMALL, t=0.1))), 0.5)  # marker beside a corner
        self.assertEqual(score_of(zone(c, (0.2975, -SMALL / 2 - 0.030))), 0.25)  # marker downwind

    def test_marker_downwind_inside_or_beside_another_side(self):
        cx = (GX0 + GX1) / 2
        for blue in ((cx, -0.146 - 0.020), (cx, 0.0), (0.440 + 0.020, 0.0), (0.148 - 0.012, 0.0)):
            bs = zone(DEMO_CORNERS, blue)
            score, d = evaluate(bs)
            self.assertEqual(score, 0.5, blue)
            self.assertEqual(d["marker"], 0.0)

    def test_marker_past_a_corner_or_too_far_out(self):
        self.assertEqual(score_of(zone(DEMO_CORNERS, (0.148 - 0.010, 0.176))), 0.5)  # t < 0
        self.assertEqual(score_of(zone(DEMO_CORNERS, (0.440 + 0.010, 0.176))), 0.5)  # t > 1
        # 100 mm out is the far end of the 0.25 band; on this field only a 90 ft zone leaves room for it on the
        # grass (its upwind side 100 mm inside the field's left edge).
        c = square_xy(SMALL, cy=-0.040)
        self.assertEqual(score_of(zone(c, upwind_mid(SMALL, cy=-0.040, out=0.100))), 0.5)
        self.assertEqual(score_of(zone(c, upwind_mid(SMALL, cy=-0.040, out=0.099))), 0.5)
        z = quad([(*p, 0.0) for p in c])
        (i,), normals = upwind_sides(z)
        for out, want in ((0.100, 0.25), (0.101, 0.0), (0.0, 0.0), (-0.010, 0.0)):
            self.assertEqual(marker_on_side(z, i, normals[i], upwind_mid(SMALL, cy=-0.040, out=out))[2], want, out)

    def test_marker_bands(self):
        """10 and 55 mm out and 1/3 and 2/3 along are inside the full band; just outside them is partial."""
        for out, want in ((0.010, 1.0), (0.0099, 0.75), (0.055, 1.0), (0.0551, 0.75)):
            bs = zone(square_xy(0.280, cy=-0.02), upwind_mid(0.280, cy=-0.02, out=out))
            self.assertTrue(all(placed(b) for b in bs))
            self.assertEqual(score_of(bs), want, out)
        for t, want in ((1 / 3, 1.0), (1 / 3 - 0.002, 0.75), (2 / 3, 1.0), (2 / 3 + 0.002, 0.75)):
            bs = zone(square_xy(0.280, cy=-0.02), upwind_mid(0.280, cy=-0.02, t=t))
            self.assertEqual(score_of(bs), want, t)

    def test_side_bands(self):
        for side, want in ((0.277, 1.0), (0.2765, 0.75), (0.249, 0.75), (0.2485, 0.0), (0.166, 0.0)):
            bs = zone(square_xy(side, cy=-0.02), upwind_mid(side, cy=-0.02))
            self.assertTrue(all(placed(b) for b in bs))
            self.assertEqual(score_of(bs), want, side)

    def test_corner_bands(self):
        """Parallelograms leaning along y (the field's longer way), the blue 30 mm outside the middle of the upwind
        side: corners 80/100 degrees full, 79/101 and 70/110 partial, 69/111 nothing."""
        def para(angle_deg, s):
            a = math.radians(angle_deg)
            v = np.array([s * math.sin(a), s * math.cos(a)])  # the second side, at angle_deg to the first (along y)
            p0 = np.array([0.0, 0.0])
            pts = [p0, p0 + (0.0, s), p0 + (0.0, s) + v, p0 + v]
            ys = [p[1] for p in pts]
            shift = np.array([(GX0 + GX1) / 2 - (s * math.sin(a)) / 2, GY0 + 0.004 - min(ys)])
            pts = [tuple(p + shift) for p in pts]
            z = quad([(*p, 0.0) for p in pts])
            up, normals = upwind_sides(z)
            i = up[0]
            mid = (z["corners"][i] + z["corners"][(i + 1) % 4]) / 2 + normals[i] * 0.030
            bs = zone(pts, tuple(mid))
            self.assertTrue(all(placed(b) for b in bs), (angle_deg, s))  # the test's shape fits on the grass
            return bs
        for ang, s, want in ((90, 0.285, 1.0), (80, 0.285, 1.0), (100, 0.285, 1.0), (79, 0.285, 0.75),
                             (101, 0.285, 0.75), (70, 0.255, 0.75), (110, 0.255, 0.75), (69, 0.255, 0.0),
                             (111, 0.255, 0.0)):
            self.assertEqual(score_of(para(ang, s)), want, ang)

    def test_corner_limits_bind_on_both_sides(self):
        """One corner of a square moved along its diagonal: pulled in, its corner opens past 100 degrees while
        the others stay at 80 or more (partial square); pushed out, it closes under 70 degrees while the others stay
        at 110 or less (no square)."""
        def moved(side, cx, cy, u):
            a = side / 2
            c = [(cx - a, cy + a), (cx + a, cy + a), (cx - a, cy - a), (cx + a + u, cy - a - u)]  # far right moves
            return c, quad([(*p, 0.0) for p in c])
        c, z = moved(0.320, (GX0 + GX1) / 2, -0.005, -0.030)  # pulled 42 mm in along the diagonal
        self.assertGreater(max(z["angles"]), 100.0)
        self.assertLess(max(z["angles"]), 110.0)
        self.assertGreaterEqual(min(z["angles"]), 80.0)
        self.assertGreaterEqual(min(z["sides"]), L.SIDE_FULL)
        bs = zone(c, ((GX0 + GX1) / 2, -0.005 + 0.160 + 0.030))
        self.assertTrue(all(placed(b) for b in bs))
        self.assertEqual(score_of(bs), 0.75)  # 0.25 square (a 102 degree corner) + 0.5 marker
        c, z = moved(0.260, 0.262, 0.033, 0.068)  # pushed 96 mm out along the diagonal
        self.assertLess(min(z["angles"]), 70.0)
        self.assertLessEqual(max(z["angles"]), 110.0)
        self.assertGreaterEqual(min(z["sides"]), L.SIDE_PART)
        bs = zone(c, (0.262, 0.0))
        self.assertTrue(all(placed(b) for b in bs))
        self.assertEqual(score_of(bs), 0.0)

    def test_a_larger_zone_with_square_corners_is_full(self):
        """'At least 100 by 100 ft': a 320 x 300 mm zone with square corners counts as full."""
        c = [(0.140, 0.170), (0.460, 0.170), (0.140, -0.130), (0.460, -0.130)]
        self.assertEqual(score_of(zone(c, (0.300, 0.185))), 1.0)

    def test_a_turned_zone_still_counts(self):
        """Turned 10 degrees (the field holds a 280 mm zone turned up to about 13), the side nearest +y is the
        upwind one."""
        c = square_xy(0.280, cy=0.0, turn=10.0)
        blue = upwind_mid(0.280, cy=0.0, turn=10.0)
        self.assertTrue(all(on_grass_outline(beacon(x, y)) for x, y in [*c, blue]))
        score, d = evaluate(zone(c, blue))
        self.assertEqual(score, 1.0)
        self.assertAlmostEqual(d["upwind_normal_deg"], 10.0, places=4)

    def test_an_equal_tie_lets_either_side_count(self):
        """A square turned 45 degrees (it does not fit on this field: the rule is checked directly) has two sides
        equally near the wind; the blue beside either counts."""
        c = square_xy(0.30, cx=0.0, cy=0.0, turn=45.0)
        z = quad([(*p, 0.0) for p in c])
        up, normals = upwind_sides(z)
        self.assertEqual(len(up), 2)
        for i in up:
            a, b = z["corners"][i], z["corners"][(i + 1) % 4]
            mid = (a + b) / 2 + normals[i] * 0.030
            self.assertEqual(marker_on_side(z, i, normals[i], mid)[2], 0.5)

    def test_whole_field_leaves_no_room_for_the_marker(self):
        """Corners at the field's corners: the blue cannot stand on the grass outside the upwind side."""
        c = [(GX0, GY1), (GX1, GY1), (GX0, GY0), (GX1, GY0)]
        self.assertEqual(score_of(zone(c, (0.2975, GY1 + 0.030))), 0.5)  # blue on the road: not placed
        self.assertEqual(score_of(zone(c, (0.2975, GY1 - 0.010))), 0.5)  # blue inside the zone

    def test_any_orange_beacon_may_be_any_corner(self):
        for perm in itertools.permutations(DEMO_CORNERS):
            self.assertEqual(score_of(zone(perm, DEMO_BLUE)), 1.0)

    def test_every_outcome_listed_in_task_json(self):
        """The 'Every outcome' list in task.json's scoring, state by state (each one standing on the grass)."""
        cx = (GX0 + GX1) / 2
        small = square_xy(SMALL)
        # A 250 mm rhombus leaning along y with 65 and 115 degree corners (it just fits on the grass).
        v = (0.250 * math.sin(math.radians(65)), 0.250 * math.cos(math.radians(65)))
        rhombus = [(0.180, -0.168), (0.180, 0.082), (0.180 + v[0], 0.082 + v[1]), (0.180 + v[0], -0.168 + v[1])]
        n = (-math.sin(math.radians(25)), math.cos(math.radians(25)))  # its upwind side's outward direction
        rh_blue = (0.180 + v[0] / 2 + 0.030 * n[0], 0.082 + v[1] / 2 + 0.030 * n[1])
        listed = [
            ("292 mm zone, blue 30 mm out mid upwind", zone(DEMO_CORNERS, DEMO_BLUE), 1.0),
            ("blue beside a corner of that side", zone(DEMO_CORNERS, (0.148 + 0.25 * 0.292, 0.176)), 0.75),
            ("blue 5 mm out", zone(DEMO_CORNERS, (cx, 0.151)), 0.75),
            ("280 mm zone, blue 70 mm out", zone(square_xy(0.280, cy=-0.025),
                                                 upwind_mid(0.280, cy=-0.025, out=0.070)), 0.75),
            ("260 mm zone, blue mid upwind", zone(small, upwind_mid(SMALL)), 0.75),
            ("292 mm zone, blue downwind", zone(DEMO_CORNERS, (cx, -0.166)), 0.5),
            ("292 mm zone, blue inside", zone(DEMO_CORNERS, (cx, 0.0)), 0.5),
            ("260 mm zone, blue beside a corner", zone(small, upwind_mid(SMALL, t=0.15)), 0.5),
            ("260 mm zone, blue downwind", zone(small, (cx, -SMALL / 2 - 0.030)), 0.25),
            ("166 mm zone", zone(square_xy(0.166), upwind_mid(0.166)), 0.0),
            ("250 mm zone with 65 degree corners", zone(rhombus, rh_blue), 0.0),
            ("orange as the marker", zone(DEMO_CORNERS[:3] + [DEMO_BLUE], DEMO_CORNERS[3]), 0.0),
            ("orange left on the road", zone(DEMO_CORNERS, DEMO_BLUE, orange_2=beacon(*L.START["orange_2"])), 0.0),
        ]
        for what, bs, want in listed:
            self.assertEqual(score_of(bs), want, what)
            if what not in ("orange left on the road",):
                self.assertTrue(all(placed(b) for b in bs), what)
        # ... and the two that read the arms: a beacon still gripped (the demo's grip, closed on it) on its spot.
        bx, by = DEMO_BLUE
        self.assertEqual(score_of_arms(DONE, snap(left=(bx, by, L.MAT_TOP + 0.008), left_open=ON)), 0.5)
        ox, oy = DEMO_CORNERS[3]
        self.assertEqual(score_of_arms(DONE, snap(right=(ox, oy, L.MAT_TOP + 0.008), right_open=ON)), 0.0)

    def test_the_field_limits_the_zones_turn(self):
        """A 277 mm zone fits (corner centres on the grass) turned up to about 14 degrees, a 249 mm one about 27:
        so the upwind side is always the zone's left side, and no 'faces the wind' rule is needed."""
        def fits(side, turn):
            return any(all(on_grass_outline(beacon(x, y)) for x, y in square_xy(side, cy=cy, turn=turn))
                       for cy in np.arange(-0.02, 0.025, 0.005))
        self.assertTrue(fits(0.277, 13.5))
        self.assertFalse(fits(0.277, 14.5))
        self.assertTrue(fits(0.249, 26.5))
        self.assertFalse(fits(0.249, 27.5))

    def test_every_score_turns_up(self):
        c = square_xy(SMALL)
        cases = {
            1.0: zone(DEMO_CORNERS, DEMO_BLUE),
            0.75: zone(DEMO_CORNERS, (0.160, 0.176)),
            0.5: zone(DEMO_CORNERS, (0.294, -0.176)),
            0.25: zone(c, (0.2975, -SMALL / 2 - 0.030)),
            0.0: zone(square_xy(0.166), upwind_mid(0.166)),
        }
        for want, bs in cases.items():
            self.assertEqual(score_of(bs), want)

    def test_success_needs_stillness(self):
        r, _ = run(build(hold=10))  # 10 still frames: 0.5 s of paths
        self.assertTrue(r.success)
        r, _ = run(build(hold=5))
        self.assertFalse(r.success)
        self.assertEqual(r.score, 1.0)
        # One beacon creeping 0.6 mm a step (6 mm per 0.5 s) is not still; 0.4 mm a step (4 mm) is.
        for creep, want in ((0.0006, False), (0.0004, True)):
            frames = [S0] + glide(S0, DONE, 20)
            for k in range(15):
                f = list(DONE)
                f[0] = beacon(0.148 + creep * k, 0.146)
                frames.append(f)
            r, _ = run(frames)
            self.assertEqual(r.success, want, creep)

    def test_score_is_the_state_at_hand(self):
        r, t = run(build())
        self.assertEqual(r.score, 1.0)
        r = t.update(zone(DEMO_CORNERS, (0.294, -0.176)))  # the blue moved downwind (not a reset)
        self.assertEqual(r.score, 0.5)
        r = t.update(zone(DEMO_CORNERS[:3] + [(0.300, 0.0)], DEMO_BLUE))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)
        self.assertLessEqual(t.first_at[1.0], 21)  # reached on (or a step before) the glide's last frame


class ZeroConditions(unittest.TestCase):
    def test_an_orange_beacon_left_on_the_road_or_strip(self):
        for k in range(4):
            name = L.ORANGE[k]
            bs = zone(DEMO_CORNERS, DEMO_BLUE, **{name: beacon(*L.START[name])})
            self.assertEqual(score_of(bs), 0.0, name)

    def test_an_orange_beacon_tipped_held_up_or_off_the_grass(self):
        for s in (beacon(0.148, 0.146, tilt=90.0), beacon(0.148, 0.146, tilt=12.0),
                  beacon(0.148, 0.146, bottom=L.MAT_TOP + 0.005), beacon(0.148, GY1 + 0.005),
                  beacon(0.148, 0.146, bottom=L.TOP - 0.300)):
            self.assertEqual(score_of(zone(DEMO_CORNERS, DEMO_BLUE, orange_1=s)), 0.0)

    def test_too_small_or_not_square(self):
        self.assertEqual(score_of(zone(square_xy(0.166), upwind_mid(0.166))), 0.0)  # 60 ft
        self.assertEqual(score_of(zone(square_xy(0.248), upwind_mid(0.248))), 0.0)
        # A long thin rectangle: two sides far too short.
        c = [(0.15, 0.10), (0.45, 0.10), (0.15, -0.10), (0.45, -0.10)]
        self.assertEqual(score_of(zone(c, (0.30, 0.13))), 0.0)

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        r, t = run(build(hold=10))
        self.assertTrue(r.success)
        hist = len(t.history)
        bad = list(DONE)
        bad[2] = state((np.nan, 0.0, 0.77))
        r = t.update(bad)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(r.details["bad_beacons"], ["orange_3"])
        self.assertEqual(t.bad_steps, 1)
        self.assertEqual(len(t.history), hist)
        r = t.update(DONE)  # the next good step carries on
        self.assertTrue(r.success)
        for broken in (state((0.3, 0, 0.77), quat=(0, 0, 0, 0)), state((0.3, 0, 0.77), spin=(0, np.inf, 0))):
            r = t.update([broken, *DONE[1:]])
            self.assertEqual(r.score, 0.0)
        self.assertEqual(t.result(DONE[:4]).score, 0.0)  # a beacon missing


class LetGo(unittest.TestCase):
    """A beacon counts only once let go: no gripper's pad-tip point lower than 10 mm above its top within that
    gripper's finger reach of its axis (79 mm fully open, 50 mm closed on a beacon)."""

    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0), 0.0794, places=4)
        self.assertAlmostEqual(L.release_reach(ON), 0.0502, places=4)
        self.assertAlmostEqual(L.release_reach(0.0), 0.0414, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=float("nan"))), L.release_reach(1.0))  # unreadable
        self.assertAlmostEqual(reach(SimpleNamespace()), L.release_reach(1.0))

    def test_parked_arms_let_every_beacon_go(self):
        score, d = evaluate(DONE, PARKED)
        self.assertEqual((score, d["held"]), (1.0, []))

    def test_a_beacon_still_gripped_does_not_count(self):
        x, y = DEMO_BLUE
        grip = snap(left=(x, y, L.MAT_TOP + 0.008), left_open=ON, right=PARKED["right"].tips)  # the demo's grip
        self.assertTrue(standing(DONE[4]))
        self.assertTrue(held(DONE[4], grip))
        score, d = evaluate(DONE, grip)
        self.assertEqual((score, d["blue_placed"], d["held"]), (0.5, False, ["blue"]))
        x, y = DEMO_CORNERS[0]
        score, d = evaluate(DONE, snap(left=(x, y, L.MAT_TOP + 0.008), left_open=ON))
        self.assertEqual((score, d["orange_not_placed"]), (0.0, ["orange_1"]))
        # the right arm gripping orange_3 while the left, readable too, is parked: each arm is checked
        x, y = DEMO_CORNERS[2]
        score, d = evaluate(DONE, snap(left=PARKED["left"].tips, right=(x, y, L.MAT_TOP + 0.008), right_open=ON))
        self.assertEqual((score, d["orange_not_placed"]), (0.0, ["orange_3"]))

    def test_an_open_gripper_round_a_beacon_still_holds_it(self):
        x, y = DEMO_BLUE
        self.assertTrue(held(DONE[4], snap(left=(x, y, L.MAT_TOP + 0.020))))

    def test_a_finger_propping_a_beacon_is_holding_it(self):
        x, y = DEMO_BLUE
        low = L.MAT_TOP + 0.015
        self.assertTrue(held(DONE[4], snap(left=(x, y - 0.071, low))))  # fully open, a finger's outside face on it
        self.assertTrue(held(DONE[4], snap(left=(x + 0.079, y, low))))  # just inside the 79 mm open reach
        self.assertFalse(held(DONE[4], snap(left=(x + 0.080, y, low))))  # just outside it: no finger can touch
        self.assertTrue(held(DONE[4], snap(left=(x + 0.041, y, low), left_open=0.0)))  # closed: 41 mm
        self.assertFalse(held(DONE[4], snap(left=(x + 0.042, y, low), left_open=0.0)))

    def test_pad_tips_10_mm_over_the_top_have_let_go(self):
        x, y = DEMO_BLUE
        top = highest(DONE[4])
        self.assertAlmostEqual(top, L.MAT_TOP + L.BEACON_H)
        self.assertTrue(held(DONE[4], snap(left=(x, y, top + 0.0099))))
        self.assertFalse(held(DONE[4], snap(left=(x, y, top + 0.0101))))
        self.assertEqual(evaluate(DONE, snap(left=(x, y, top + 0.011)))[0], 1.0)  # the demo's gripper rising away

    def test_a_tilted_beacon_is_measured_along_its_axis(self):
        """The reach is measured to the beacon's axis at the pad tips' height: for a beacon leaning 10 degrees toward
        -y that point is 3.7 mm toward -y from its centre."""
        b = beacon(0.294, 0.176, tilt=10.0, axis="x")  # its top leans toward -y
        self.assertTrue(standing(b))
        h = highest(b) + 0.005
        shift = (h - float(b.pos[2])) * math.tan(math.radians(10.0))
        self.assertAlmostEqual(shift, 0.0037, places=4)
        x, on_axis_y = float(b.pos[0]), float(b.pos[1]) - shift
        self.assertTrue(held(b, snap(left=(x, on_axis_y - 0.0785, h))))  # 82 mm from the centre, 78.5 from the axis
        self.assertFalse(held(b, snap(left=(x, on_axis_y + 0.0800, h))))  # 76 mm from the centre, 80 from the axis

    def test_success_needs_every_beacon_let_go(self):
        x, y = DEMO_BLUE
        r, _ = run(build(hold=30), arms=snap(left=(x, y, L.MAT_TOP + 0.008), left_open=ON))
        self.assertEqual((r.score, r.success), (0.5, False))
        r, _ = run(build(hold=30), arms=PARKED)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        frames = build(hold=10)
        r, t = run(frames, arms=None)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.arms_unavailable, len(frames))
        self.assertEqual(r.details["arms_unavailable_by_side"], {"left": len(frames), "right": len(frames)})
        self.assertEqual(r.details["arms_readable"], [])

    def test_one_unreadable_arm_never_hides_the_other(self):
        x, y = DEMO_CORNERS[2]  # orange_3, gripped by the right arm while the left cannot be read
        right = SimpleNamespace(tips=np.array((x, y, L.MAT_TOP + 0.008)), opening=ON)
        frames = build(hold=10)
        r, t = run(frames, arms={"left": SimpleNamespace(tips=np.full(3, np.nan), opening=1.0), "right": right})
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["orange_not_placed"], ["orange_3"])
        self.assertEqual(r.details["arms_readable"], ["right"])
        self.assertEqual(t.arms_unavailable_by_side, {"left": len(frames), "right": 0})
        self.assertEqual(t.arms_unavailable, len(frames))
        self.assertEqual(t.update(DONE, {"right": right}).score, 0.0)  # the left arm missing altogether

    def test_rigstate_reset_hook_resets_the_tracker(self):
        r, t = run(build(hold=10))
        self.assertEqual(t.best, 1.0)
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            rigstate.offline_reset()
            self.assertEqual((t.steps, t.best, t.first_at), (0, 0.0, {}))
            self.assertIsNone(rigstate.arms())
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        r, t = run(build(hold=10))
        self.assertEqual(t.best, 1.0)
        r = t.update(S0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(len(t.history), 1)

    def test_gradual_return_is_not_a_reset(self):
        r, t = run(build(hold=10) + glide(DONE, S0, 30))
        self.assertEqual(t.best, 1.0)
        self.assertGreater(t.steps, 30)

    def test_jump_elsewhere_is_not_a_reset(self):
        r, t = run(build(hold=10))
        moved = zone(square_xy(SMALL), upwind_mid(SMALL))
        t.update(moved)
        self.assertEqual(t.best, 1.0)

    def test_jittered_reset_counts(self):
        r, t = run(build(hold=10))
        jit = [beacon(L.START[n][0] + 0.02, L.START[n][1] - 0.02) for n in L.NAMES]
        t.update(jit)
        self.assertEqual(t.steps, 1)


class Exploits(unittest.TestCase):
    def test_orange_as_the_wind_marker_and_blue_at_a_corner(self):
        """Three oranges and the blue at the corners, the fourth orange outside the upwind side: 0."""
        bs = zone(DEMO_CORNERS[:3] + [DEMO_BLUE], DEMO_CORNERS[3])
        self.assertEqual(score_of(bs), 0.0)

    def test_blue_stacked_on_an_orange_or_held_up(self):
        on_top = beacon(0.148, 0.146, bottom=L.MAT_TOP + L.BEACON_H)
        self.assertEqual(score_of(zone(DEMO_CORNERS, (0.148, 0.146), blue=on_top)), 0.5)
        held = beacon(*DEMO_BLUE, bottom=L.MAT_TOP + 0.010)
        self.assertEqual(score_of(zone(DEMO_CORNERS, DEMO_BLUE, blue=held)), 0.5)

    def test_blue_on_the_road_beside_a_zone_at_the_field_edge(self):
        """The zone's upwind side on the field's edge, the blue on the road beyond it: not on the grass."""
        c = [(0.150, 0.185), (0.440, 0.185), (0.150, -0.105), (0.440, -0.105)]
        self.assertEqual(score_of(zone(c, (0.295, 0.215))), 0.5)

    def test_corner_on_the_road(self):
        c = [(0.150, 0.200), (0.440, 0.146), (0.150, -0.146), (0.440, -0.146)]
        self.assertEqual(score_of(zone(c, DEMO_BLUE)), 0.0)

    def test_beacons_lying_down_on_the_marks(self):
        bs = [beacon(x, y, tilt=90.0) for x, y in DEMO_CORNERS] + [beacon(*DEMO_BLUE, tilt=90.0)]
        self.assertEqual(score_of(bs), 0.0)

    def test_holding_beacons_on_their_spots_earns_nothing_for_them(self):
        """Both grippers closed on beacons standing on their spots at the end: those beacons are not placed."""
        (lx, ly), (rx, ry) = DEMO_CORNERS[0], DEMO_CORNERS[2]
        both = snap(left=(lx, ly, L.MAT_TOP + 0.008), right=(rx, ry, L.MAT_TOP + 0.008), left_open=ON, right_open=ON)
        self.assertEqual(score_of_arms(DONE, both), 0.0)
        bx, by = DEMO_BLUE
        blue = snap(left=(bx, by, L.MAT_TOP + 0.008), left_open=ON)
        self.assertEqual(score_of_arms(DONE, blue), 0.5)

    def test_two_oranges_side_by_side_at_one_corner(self):
        c = [(0.148, 0.146), (0.176, 0.146), (0.148, -0.146), (0.440, -0.146)]
        self.assertEqual(score_of(zone(c, DEMO_BLUE)), 0.0)

    def test_random_states_score_only_listed_values(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(400):
            c = [(rng.uniform(GX0 - 0.02, GX1 + 0.02), rng.uniform(GY0 - 0.02, GY1 + 0.02)) for _ in range(4)]
            if rng.random() < 0.5:
                side = rng.uniform(0.24, 0.30)
                c = [(x + rng.normal(0, 0.006), y + rng.normal(0, 0.006))
                     for x, y in square_xy(side, cy=rng.uniform(-0.03, 0.03), turn=rng.uniform(-8, 8))]
            blue = (rng.uniform(GX0, GX1), rng.uniform(0.10, GY1 + 0.02))
            s = score_of(zone(c, blue))
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertGreaterEqual(len(seen), 4)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _kin_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yamkin.py").is_file():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, beacons):
        return dict(zip(self.task.BEACON_NAMES, beacons, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.BEACON_NAMES, L.NAMES)
        for spec, s in zip(t.BEACONS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((spec.radius, spec.height, spec.mass), (L.BEACON_RADIUS, L.BEACON_H, L.BEACON_MASS))
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 3 + 1 + 1 + 5 + 5)
        self.assertEqual([b.color for b in t.SLEEVE],
                         [L.SOCK_ORANGE, L.WHITE, L.SOCK_ORANGE, L.WHITE, L.SOCK_ORANGE])  # orange and white bands
        # The mats' tops are flush; the fire engine stands on the strip; the sleeve hangs from the mast's top.
        for m in (t.GRASS, t.ROAD, t.STRIP):
            self.assertAlmostEqual(m.pos[2] + m.size[2] / 2, L.MAT_TOP)
        self.assertAlmostEqual(t.ENGINE.pos[2] - t.ENGINE.size[2] / 2, L.MAT_TOP)
        x0, x1, y0, y1 = L.STRIP
        self.assertTrue(x0 <= t.ENGINE.pos[0] - t.ENGINE.size[0] / 2 and t.ENGINE.pos[0] + t.ENGINE.size[0] / 2 <= x1)
        self.assertTrue(y0 <= t.ENGINE.pos[1] - t.ENGINE.size[1] / 2 and t.ENGINE.pos[1] + t.ENGINE.size[1] / 2 <= y1)
        self.assertAlmostEqual(t.SLEEVE[0].pos[1] + L.SLEEVE_LEN / 2, L.MAST_Y - L.MAST_RADIUS)
        for b in t.SLEEVE:
            self.assertLess(b.pos[2] + b.size[2] / 2, L.TOP + L.MAST_H)
        # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m),
        # and every object is on the table.
        for o in t.OBJECTS:
            half_y = o.size[1] / 2 if hasattr(o, "size") else o.radius
            self.assertLess(abs(o.pos[1]) + half_y, 0.45, o.name)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()  # no arms read (as offline): the let-go rule is skipped
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
        for key in ("step", "score", "success", "square", "marker", "sides_mm", "angles_deg", "convex",
                    "upwind_side", "upwind_normal_deg", "blue_out_mm", "blue_along", "orange_not_placed",
                    "blue_placed", "all_still", "bad_steps", "first_step", "orange_1", "blue"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["blue_out_mm"], 30.0)
        self.assertEqual(rows[-1]["blue"]["path_mm"], 0.0)
        self.assertTrue(rows[-1]["orange_4"]["placed"])
        self.assertIsNone(rows[0]["orange_1"]["path_mm"])
        self.assertEqual(rows[0]["orange_not_placed"], list(L.ORANGE))
        self.assertIn("rigstate_error", rows[-1])
        self.assertFalse(rows[-1]["blue"]["held"])

    def test_hooks_read_the_arms(self):
        """task.py passes rigstate's arms to the scorer: the blue still gripped is not placed; once the gripper has
        risen clear and parked it is, and the trace shows both arms."""
        def feed(left_tips, left_open):
            rigstate.offline_feed({"left": {"tcp": np.add(left_tips, (0, 0, 0.013)), "tips": left_tips,
                                            "opening": left_open},
                                   "right": {"tcp": PARKED["right"].tcp, "tips": PARKED["right"].tips,
                                             "opening": 1.0}})
        bx, by = DEMO_BLUE
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                for _ in range(12):
                    feed((bx, by, L.MAT_TOP + 0.008), ON)
                    self.assertFalse(self.task.success(self.objects(DONE)))
                self.assertEqual(self.task.score(self.objects(DONE)), 0.5)
                feed(tuple(PARKED["left"].tips), 1.0)
                self.assertTrue(self.task.success(self.objects(DONE)))
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
                rigstate.offline_reset()
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(rows[0]["blue"]["held"])
        self.assertEqual(rows[0]["held"], ["blue"])
        self.assertAlmostEqual(rows[0]["left_opening"], ON)
        self.assertEqual(rows[-1]["held"], [])
        self.assertEqual(rows[-1]["arms_readable"], ["left", "right"])
        self.assertEqual(rows[-1]["arms_unavailable"], 0)

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
                self.assertFalse(self.task.success(self.objects([state((np.nan, 0, 0.77)), *S0[1:]])))
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

    def test_demo_builds_the_zone_and_each_arm_works_its_own_side(self):
        t = self.task
        final = list(S0)
        for arm, steps in t.PLAN.items():
            sign = 1.0 if arm == "left" else -1.0
            for name, (x, y) in steps:
                self.assertGreater(sign * L.START[name][1], 0.0)  # each arm fetches only its own side's beacons
                self.assertGreater(sign * y, 0.0)
                final[L.NAMES.index(name)] = beacon(x, y)
        self.assertEqual(score_of(final), 1.0)
        z = quad([s.pos for s in final[:4]])
        self.assertGreaterEqual(min(z["sides"]) - L.SIDE_FULL, 0.015 - 1e-9)  # 15 mm over the 100 ft minimum
        # With one arm's beacons left where they start, two or three orange beacons stay off the grass: 0.
        for idle in ("left", "right"):
            part = list(final)
            for name, _ in t.PLAN[idle]:
                part[L.NAMES.index(name)] = S0[L.NAMES.index(name)]
            self.assertEqual(score_of(part), 0.0)

    @staticmethod
    def spells(wps):
        """(start, end, x, y) of each waypoint, on the plan's clock."""
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w.pos[0], w.pos[1]))
            t += w.duration
        return out

    def test_grippers_stay_apart(self):
        """The two grippers are always at least 0.20 m apart in y, and the demo fits the run limit."""
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, _, ya in left:
            for b0, b1, _, yb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400)

    def test_carried_beacons_clear_everything(self):
        t = self.task
        bottom = t.HOVER_Z - t.GRIP_UP
        self.assertGreaterEqual(bottom - (L.MAT_TOP + L.BEACON_H), 0.030)  # over a standing beacon's top
        self.assertGreaterEqual(bottom - (L.MAT_TOP + L.ENGINE_SIZE[2]), 0.030)  # over the fire engine
        self.assertGreaterEqual(t.PICK_Z - 0.013 - L.MAT_TOP, 0.005)  # pad tips over the road
        self.assertAlmostEqual(-2 * t.GRIP, 2 * L.BEACON_RADIUS - 0.008)  # commanded 8 mm narrower
        self.assertLess(t.GRIP_UP - 0.013, L.BEACON_H)  # the pads close on the beacon, not over it
        self.assertGreater(t.GRIP_UP + 0.009, L.BEACON_H - 0.005)
        # Open pads (gap plus two 8 mm pads, 27 mm wide along x) clear a neighbour 60 mm away on the road.
        xs = sorted(L.START[n][0] for n in L.NAMES if L.START[n][1] > 0)
        self.assertGreater(min(b - a for a, b in zip(xs, xs[1:])) - 0.0135 - L.BEACON_RADIUS, 0.025)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_task_json_states_the_thresholds(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("277 mm", "249 mm", "80 to 100 degrees", "70 to 110 degrees", "10 to 55 mm", "100 mm",
                  "10 degrees", "3 mm", "1/3 to 2/3", "5 mm", "0.5 s", "3 s", "10 mm above its top", "79 mm",
                  "50 mm"):
            self.assertIn(s, text)
        setup = spec["setup"]
        for s in ("x 13.0 to 46.5 cm, y -17.0 to +19.0 cm", "28 mm across and 28 mm tall, 40 g",
                  "x 20.0 and 26.0 cm, y +22.5 cm", "x 20.0 and 24.0 cm, y -21.5 cm", "x 32.0 cm, y +22.5 cm",
                  "x 45.0 cm, y +33.0 cm"):
            self.assertIn(s, setup)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(spec["run_limit"], "2,400 steps (2 minutes at 20 Hz)")

    def test_success_hold_outlasts_every_release(self):
        """A beacon counts as placed while still gripped (within 3 mm), so the success hold must be longer than
        the settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        plan = self.task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly stand it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    self.assertLess(sum(a.duration for a in after), self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 5)


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the other side's starting beacons."""

    def test_other_sides_beacons_are_out_of_reach(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        for arm, names in (("left", ("orange_3", "orange_4")), ("right", ("orange_1", "orange_2", "blue"))):
            base = K.BASES["station"][arm]
            for name in names:
                x, y = L.START[name]
                for yaw in np.linspace(-math.pi, math.pi, 13):
                    ok = K.reachable((x, y, L.MAT_TOP + 0.020), float(yaw), base)[0]
                    self.assertFalse(ok, (arm, name, yaw))
        # ... while each arm reaches its own side's beacons and corners.
        for arm, names in (("left", ("orange_1", "orange_2", "blue")), ("right", ("orange_3", "orange_4"))):
            base = K.BASES["station"][arm]
            for name in names:
                x, y = L.START[name]
                self.assertTrue(K.reachable((x, y, L.MAT_TOP + 0.020), 0.0, base)[0], (arm, name))


if __name__ == "__main__":
    unittest.main()
