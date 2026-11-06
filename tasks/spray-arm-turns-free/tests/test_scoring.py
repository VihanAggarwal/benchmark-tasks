"""Offline tests for scoring.py, offline_model.py and the hooks in task.py: synthetic item states, spray arm angles
and arm snapshots (fed through rigstate.offline_feed where the hooks are tested), no simulator.
python3 -m unittest discover -s tests -v"""

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
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = L.CONTROL_HZ
TIPS = 0.013
HUB = np.array(L.HUB)
PARKED = {"left": np.array([0.30, 0.305, L.TOP + 0.20]), "right": np.array([0.30, -0.305, L.TOP + 0.20])}


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def item(name, xy, tilt=0.0, lift=0.0, on="floor", quat=None):
    """An ObjectState-like cylinder standing (tilted ``tilt`` degrees about x) with its lowest point ``lift`` above
    the rack floor (``on`` "floor") or the table (``on`` "table")."""
    r, h = L.SIZE[name]
    q = about("x", tilt) if quat is None else np.asarray(quat, float)
    t = math.radians(tilt)
    base = L.FLOOR_TOP if on == "floor" else L.TOP
    zc = base + lift + 0.5 * h * abs(math.cos(t)) + r * abs(math.sin(t))
    c = np.array([xy[0], xy[1], zc])
    ext = np.array([r, r, h / 2])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=c - ext, high=c + ext)


def start_items():
    return {n: item(n, L.START[n][:2], on="table") for n in L.ITEMS}


def place_xy(n):
    return L.JAR_PLACE[n] if n in L.JARS else L.BOWL_PLACE[n]


def loaded_items(**over):
    """Every item standing on the rack floor where the demo puts it; ``over`` replaces items by name."""
    out = {n: item(n, place_xy(n)) for n in L.ITEMS}
    out.update(over)
    return out


def at_hub(name, r, phi_deg=135.0, **kw):
    """An item whose centre is ``r`` from the hub's axis at angle ``phi_deg``."""
    a = math.radians(phi_deg)
    return item(name, (HUB[0] + r * math.cos(a), HUB[1] + r * math.sin(a)), **kw)


def leaning(name, r, phi_deg, tilt, inward=True):
    """An item standing on the rack floor with its centre ``r`` from the hub's axis at ``phi_deg``, its axis tilted
    ``tilt`` degrees so that its top leans toward the hub (``inward``) or away from it, its lowest rim on the floor."""
    s = at_hub(name, r, phi_deg=phi_deg)
    radial = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg)), 0.0])
    axis = np.cross(radial, [0.0, 0.0, 1.0]) * (1.0 if inward else -1.0)  # turning about it tips the top inward
    h = math.radians(tilt) / 2
    rr, hh = L.SIZE[name]
    t = math.radians(tilt)
    s.pos[2] = L.FLOOR_TOP + 0.5 * hh * math.cos(t) + rr * math.sin(t)
    s.quat_xyzw = np.array([*(math.sin(h) * axis), math.cos(h)])
    return s


def box_gap(p, lo, hi):
    """Horizontal distance from the point ``p`` to the rectangle (lo, hi) (0 inside)."""
    d = np.maximum(np.maximum(np.asarray(lo) - p, p - np.asarray(hi)), 0.0)
    return float(np.hypot(*d))


HOLDER_BOXES = [(np.array(xy) - np.array(L.HOLDER[:2]) / 2, np.array(xy) + np.array(L.HOLDER[:2]) / 2)
                for xy in L.HOLDER_XY.values()]


def jar_spots(others=(), step=0.0025):
    """Every upright jar centre (on a ``step`` grid) that fits on the rack floor: inside the walls, clear of the
    holders, the hub and the given (centre, radius) items."""
    out = []
    for x in np.arange(L.INNER_X[0] + L.JAR_R, L.INNER_X[1] - L.JAR_R + 1e-9, step):
        for y in np.arange(L.INNER_Y[0] + L.JAR_R, L.INNER_Y[1] - L.JAR_R + 1e-9, step):
            p = np.array([x, y])
            if any(box_gap(p, lo, hi) < L.JAR_R for lo, hi in HOLDER_BOXES):
                continue
            if np.hypot(*(p - HUB)) < L.HUB_R + L.JAR_R:
                continue
            if any(np.hypot(*(p - np.asarray(c))) < L.JAR_R + r for c, r in others):
                continue
            out.append(p)
    return out


def brute_side(s, n=720, m=400):
    """The jar's side surface sampled densely: the smallest horizontal distance from the hub's axis of its points at
    the heights the arm and its cap pass."""
    rot = S.rotation(s.quat_xyzw)
    u, v, a = rot[:, 0], rot[:, 1], rot[:, 2]
    th = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    hs = np.linspace(-L.JAR_H / 2, L.JAR_H / 2, m)
    ring = L.JAR_R * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
    pts = (np.asarray(s.pos)[None, None, :] + hs[:, None, None] * a[None, None, :] + ring[None, :, :]).reshape(-1, 3)
    z = pts[:, 2] - L.TOP
    band = pts[(z >= L.CLEAR_Z[0]) & (z <= L.CLEAR_Z[1])]
    return float(np.min(np.hypot(band[:, 0] - HUB[0], band[:, 1] - HUB[1])))


def moved(s, d):
    """The same state shifted by ``d`` (m, world)."""
    d = np.asarray(d, float)
    return SimpleNamespace(pos=s.pos + d, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel,
                           low=s.low + d, high=s.high + d)


def arm(tcp, opening=1.0):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


def arms(left=None, right=None):
    return {"left": left if left is not None else arm(PARKED["left"]),
            "right": right if right is not None else arm(PARKED["right"])}


def holding(s, opening=0.53):
    """A straight-down gripper closed on an item, its finger tips on the item's axis 12 mm above its middle."""
    return arm(np.asarray(s.pos) + np.array([0.0, 0.0, 0.012 + TIPS]), opening)


def tips_at(xy, z, opening=1.0):
    return arm((xy[0], xy[1], z + TIPS), opening)


class Film:
    """A trial, one control step per frame: (items, spray arm angle (rad), arms)."""

    def __init__(self, theta=0.0):
        self.frames = []
        self.theta = theta

    def add(self, n, items, a=None, theta=None):
        th = self.theta if theta is None else theta
        for _ in range(n):
            self.frames.append((items, th, a if a is not None else arms()))
        self.theta = th
        return self

    def turn(self, deg, n, items, a=None):
        """Turn the spray arm ``deg`` degrees in ``n`` equal steps (the items as given)."""
        a0 = self.theta
        for i in range(1, n + 1):
            self.frames.append((items, a0 + math.radians(deg) * i / n, a if a is not None else arms()))
        self.theta = a0 + math.radians(deg)
        return self


def play(film, tracker=None):
    t = tracker or Tracker(HZ)
    r = None
    for items, th, a in film.frames:
        r = t.update(items, th, a)
    return r, t


def procedure(deg=200.0, steps=100, settle=30):
    """The whole job: the items on the table, then loaded, the grippers away, a turn of ``deg``, still."""
    f = Film().add(5, start_items())
    f.add(10, loaded_items())
    f.turn(deg, steps, loaded_items())
    f.add(settle, loaded_items())
    return f


# -- geometry ---------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_loaded_tilt_boundary(self):
        for n in L.ITEMS:
            self.assertTrue(S.loaded(n, item(n, place_xy(n), tilt=9.9)), n)
            self.assertFalse(S.loaded(n, item(n, place_xy(n), tilt=10.1)), n)

    def test_either_end_up(self):
        for n in L.ITEMS:
            s = item(n, place_xy(n), quat=about("x", 180.0))
            self.assertAlmostEqual(S.tilt_deg(s), 0.0, places=6)
            self.assertTrue(S.loaded(n, s))
            s2 = item(n, place_xy(n), tilt=175.0)  # upside down and 5 degrees off
            self.assertAlmostEqual(S.tilt_deg(s2), 5.0, places=6)

    def test_loaded_bottom_boundary(self):
        for n in L.ITEMS:
            self.assertTrue(S.loaded(n, item(n, place_xy(n), lift=0.0049)), n)
            self.assertFalse(S.loaded(n, item(n, place_xy(n), lift=0.0051)), n)
            self.assertTrue(S.loaded(n, item(n, place_xy(n), lift=-0.0049)), n)  # sunk a little: still on it
            self.assertFalse(S.loaded(n, item(n, place_xy(n), lift=-0.0051)), n)

    def test_bottom_of_a_tilted_cylinder(self):
        s = item("jar_left", (0.2, 0.1), tilt=8.0, lift=0.002)
        self.assertAlmostEqual(S.bottom("jar_left", s) - L.FLOOR_TOP, 0.002, places=9)
        self.assertAlmostEqual(S.top("jar_left", s) - S.bottom("jar_left", s),
                               L.JAR_H * math.cos(math.radians(8)) + 2 * L.JAR_R * math.sin(math.radians(8)), 9)

    def test_inside_the_walls_boundary(self):
        b = "bowl_left"
        for x, y, want in ((L.INNER_X[0] + 0.0005, 0.0, True), (L.INNER_X[0] - 0.0005, 0.0, False),
                           (L.INNER_X[1] - 0.0005, 0.0, True), (L.INNER_X[1] + 0.0005, 0.0, False),
                           (0.27, L.INNER_Y[1] - 0.0005, True), (0.27, L.INNER_Y[1] + 0.0005, False),
                           (0.27, L.INNER_Y[0] + 0.0005, True), (0.27, L.INNER_Y[0] - 0.0005, False)):
            self.assertEqual(S.loaded(b, item(b, (x, y))), want, (x, y))

    def test_clear_radius_boundary(self):
        self.assertAlmostEqual(L.CLEAR_R, 0.1405)
        self.assertAlmostEqual(L.CLEAR_R, L.R_TIP + L.JAR_R + L.CLEAR_MARGIN, delta=0.0001)
        self.assertTrue(S.jar_clear(at_hub("jar_left", 0.1406)))
        self.assertFalse(S.jar_clear(at_hub("jar_left", 0.1404)))
        self.assertTrue(S.jar_clear(at_hub("jar_left", 0.1415)))
        self.assertFalse(S.jar_clear(at_hub("jar_left", 0.1395)))

    def test_held_margin(self):
        """A gripper is on an item when its fingers (a disc round the tips point out to a finger's outer corner:
        16.8 mm shut, 58.5 mm fully open) come within 20 mm of the item's side, with the tips below its top."""
        self.assertAlmostEqual(S.finger_reach(arm(PARKED["left"], 0.0)), 0.0168, places=4)
        self.assertAlmostEqual(S.finger_reach(arm(PARKED["left"], 1.0)), 0.0585, places=4)
        self.assertAlmostEqual(S.finger_reach(arm(PARKED["left"], 0.5)), math.hypot(0.0939 / 4 + 0.010, 0.0135))
        s = item("jar_left", L.JAR_PLACE["jar_left"])
        c = s.pos
        z = c[2]  # tips at the jar's middle height, below its top
        for opening in (0.0, 0.5, 1.0):
            edge = L.JAR_R + S.finger_reach(arm(PARKED["left"], opening)) + L.HELD_MARGIN
            for d in ((1.0, 0.0), (0.0, -1.0), (0.6, 0.8)):  # every direction alike
                d = np.array(d)
                self.assertTrue(S.held_by(tips_at(c[:2] + d * (edge - 0.0005), z, opening), "jar_left", s))
                self.assertFalse(S.held_by(tips_at(c[:2] + d * (edge + 0.0005), z, opening), "jar_left", s))
        self.assertTrue(S.held_by(tips_at(c[:2], S.top("jar_left", s) - 0.001), "jar_left", s))
        self.assertFalse(S.held_by(tips_at(c[:2], S.top("jar_left", s) + 0.001), "jar_left", s))  # above its top

    def test_the_demo_places(self):
        for j in L.JARS:
            x, y = L.JAR_PLACE[j]
            self.assertGreaterEqual(math.hypot(x - HUB[0], y - HUB[1]) - L.CLEAR_R, 0.022)  # 22.1 mm to spare
            self.assertGreaterEqual(x - L.JAR_R - L.INNER_X[0], 0.0099)  # 10 mm from the near wall
            self.assertGreaterEqual(L.INNER_Y[1] - (abs(y) + L.JAR_R), 0.0099)  # and from the side wall
            self.assertTrue(S.jar_clear(item(j, (x, y))))
        for b in L.BOWLS:
            x, y = L.BOWL_PLACE[b]
            p = np.array([x, y])
            self.assertLess(math.hypot(x - HUB[0], y - HUB[1]) - L.BOWL_R, L.R_TIP - 0.03)  # under the sweep
            self.assertGreaterEqual(L.INNER_Y[1] - (abs(y) + L.BOWL_R), 0.0099)  # 10 mm from the side wall
            self.assertGreaterEqual(min(box_gap(p, lo, hi) for lo, hi in HOLDER_BOXES) - L.BOWL_R, 0.034)
            # the open far finger (fingers closing along x) 13 mm from the holder's face
            far_finger = x + L.FULL_GAP / 2 + L.FINGER_THICK
            self.assertGreaterEqual(min(lo[0] for lo, hi in HOLDER_BOXES) - far_finger, 0.012)
            for j in L.JARS:  # 50 mm from the jars
                self.assertGreaterEqual(np.hypot(*(p - np.array(L.JAR_PLACE[j]))) - L.BOWL_R - L.JAR_R, 0.049)
        self.assertLess(L.FLOOR_T + L.BOWL_H, L.BAR_LOW - 0.03)  # a bowl's top 41 mm under the arm
        self.assertGreater(L.FLOOR_T + L.JAR_H, L.BAR_LOW + 0.03)  # a jar's top 39 mm above its underside
        self.assertGreater(L.BAR_LOW, L.HUB_TOP)

    def test_holders_leave_only_the_near_corners_for_the_jars(self):
        """With the holders in the far corners, the only places an upright jar can stand clear of the sweep are the
        two near corners; anywhere in the rack's far part its centre is at most 131.2 mm from the hub."""
        spots = jar_spots()
        self.assertGreater(len(spots), 1000)
        clear = [p for p in spots if S.jar_clear(item("jar_left", p))]
        self.assertTrue(clear)
        for p in clear:
            d = p - HUB
            self.assertTrue(d[0] < -0.064 and abs(d[1]) > 0.064, p)  # a near corner
        far = [float(np.hypot(*(p - HUB))) for p in spots if p[0] > HUB[0]]
        self.assertAlmostEqual(max(far), math.hypot(L.HOLDER_XY["holder_left"][0] - L.HOLDER[0] / 2 - L.JAR_R - HUB[0],
                                                    L.INNER_Y[1] - L.JAR_R), places=3)
        self.assertLess(max(far), L.CLEAR_R - 0.009)  # 131.2 mm: 9.3 mm short
        # the far corners themselves are taken: a bowl or jar there would stand inside a holder
        for lo, hi in HOLDER_BOXES:
            for xy in ((0.380, 0.110), (0.380, -0.110), (0.395, 0.125), (0.395, -0.125)):
                if lo[1] < xy[1] < hi[1]:
                    self.assertEqual(box_gap(np.array(xy), lo, hi), 0.0)
        # the holders: fixed boxes standing on the floor plate against the far wall and a side wall, under the arm
        self.assertAlmostEqual(min(lo[0] for lo, hi in HOLDER_BOXES), 0.335)
        self.assertAlmostEqual(max(hi[0] for lo, hi in HOLDER_BOXES), L.INNER_X[1])
        self.assertLess(L.HOLDER_TOP, L.BAR_LOW - 0.03)

    def test_bowls_in_the_near_corners_leave_a_jar_nowhere_clear(self):
        """A bowl in each near corner takes the room the jars need: no place is left for a jar clear of the sweep,
        so the best is 0.25. (Short items belong under the arm.)"""
        corners = [((L.INNER_X[0] + L.BOWL_R, L.INNER_Y[1] - L.BOWL_R), L.BOWL_R),
                   ((L.INNER_X[0] + L.BOWL_R, L.INNER_Y[0] + L.BOWL_R), L.BOWL_R)]
        spots = jar_spots(corners)
        self.assertTrue(spots)
        self.assertFalse([p for p in spots if S.jar_clear(item("jar_left", p))])
        best = max(spots, key=lambda p: float(np.hypot(*(p - HUB))))
        self.assertLess(float(np.hypot(*(best - HUB))), L.CLEAR_R - 0.003)  # 3.7 mm short at best (136.8 mm)
        # one bowl in a near corner: that corner has no clear place left, the other near corner still does
        spots = jar_spots(corners[:1])
        clear = [p for p in spots if S.jar_clear(item("jar_left", p))]
        self.assertTrue(clear)
        self.assertTrue(all(p[1] < 0 for p in clear))
        # both jars fit in one near corner only pressed against its walls, with under 3 mm to spare each
        pair = [(np.array([L.INNER_X[0] + L.JAR_R + 0.001, HUB[1] - 0.071]),
                 np.array([HUB[0] - 0.071, L.INNER_Y[0] + L.JAR_R + 0.001]))]
        a, b = pair[0]
        self.assertGreaterEqual(float(np.hypot(*(a - b))), 2 * L.JAR_R)
        for p in (a, b):
            self.assertTrue(S.jar_clear(item("jar_left", p)))
            self.assertLess(float(np.hypot(*(p - HUB))) - L.CLEAR_R, 0.003)
        # scored: bowls in the near corners, the jars as far out as they now fit
        bl = item("bowl_left", corners[0][0])
        br = item("bowl_right", corners[1][0])
        spots = jar_spots(corners)
        jl = max((p for p in spots if p[1] > 0), key=lambda p: float(np.hypot(*(p - HUB))))
        jr = max((p for p in spots if p[1] < 0), key=lambda p: float(np.hypot(*(p - HUB))))
        its = loaded_items(bowl_left=bl, bowl_right=br, jar_left=item("jar_left", jl), jar_right=item("jar_right", jr))
        r, t = play(Film().add(5, its).turn(200.0, 60, its).add(20, its))
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.turned)

    def test_a_leaning_jar_is_judged_at_the_arm_height(self):
        """The reviewer's case: a jar centred 140.6 mm out but leaning 9.9 degrees toward the hub reaches 1.7 mm into
        the sweep at the arm's height: not clear. Leaning away, or upright, it is."""
        lean_in = leaning("jar_left", 0.1406, 135.0, 9.9)
        self.assertTrue(S.loaded("jar_left", lean_in))
        self.assertGreaterEqual(S.hub_distance(lean_in), L.CLEAR_R)
        self.assertLess(S.jar_side_at_arm(lean_in), L.R_TIP)  # inside the 110.45 mm circle
        self.assertFalse(S.jar_clear(lean_in))
        lean_out = leaning("jar_left", 0.1406, 135.0, 9.9, inward=False)
        self.assertTrue(S.jar_clear(lean_out))
        self.assertGreater(S.jar_side_at_arm(lean_out), 0.1406 - L.JAR_R)  # leaning out: further out at that height
        up = at_hub("jar_left", 0.1406)
        self.assertAlmostEqual(S.jar_side_at_arm(up), 0.1406 - L.JAR_R, places=9)
        # the side rule's boundary, by the lean: centre 145 mm out, leaning in; clear until its side at the top of
        # the arm's band (10.9 cm up) comes within 115.5 mm of the axis
        for tilt, want in ((5.0, True), (7.0, False)):
            j = leaning("jar_left", 0.145, 135.0, tilt)
            self.assertEqual(S.jar_clear(j), want, (tilt, S.jar_side_at_arm(j)))
        # against a brute-force measure of the side: points on the jar's surface at the arm's heights
        for j in (lean_in, lean_out, leaning("jar_left", 0.145, 60.0, 7.0), leaning("jar_left", 0.16, -120.0, 3.0)):
            self.assertLessEqual(S.jar_side_at_arm(j), brute_side(j) + 1e-6)
            self.assertGreater(S.jar_side_at_arm(j), brute_side(j) - 0.0005)
        lying = item("jar_left", (0.2, 0.0), tilt=90.0)
        self.assertEqual(S.jar_side_at_arm(lying), -math.inf)
        self.assertFalse(S.jar_clear(lying))
        # scored: loaded, turned, the leaning jar in the sweep at the arm's height: 0.25, no turn counted
        its = loaded_items(jar_left=lean_in)
        r, t = play(Film().add(5, its).turn(200.0, 60, its).add(20, its))
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.turned)
        its = loaded_items(jar_left=lean_out)
        self.assertEqual(play(Film().add(5, its).turn(200.0, 60, its).add(20, its))[0].score, 1.0)


# -- tiers ------------------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().add(20, start_items()))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(t.turned)
        self.assertEqual(t.stretches, 1)

    def test_every_listed_tier_is_reachable(self):
        got = set()
        got.add(play(Film().add(10, start_items()))[0].score)  # 0: not loaded
        got.add(play(Film().add(10, loaded_items(jar_left=at_hub("jar_left", 0.130))))[0].score)  # 0.25
        got.add(play(Film().add(10, loaded_items()))[0].score)  # 0.5: loaded, clear, no turn
        got.add(play(procedure())[0].score)  # 1.0
        self.assertEqual(got, {0.0, 0.25, 0.5, 1.0})

    def test_loaded_and_clear_without_a_turn_is_half(self):
        r, _ = play(Film().add(40, loaded_items()))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)

    def test_the_latch_needs_180_degrees(self):
        r, _ = play(procedure(deg=179.0))
        self.assertEqual(r.score, 0.5)
        r, t = play(procedure(deg=181.0))
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(deg=-181.0))  # either way
        self.assertEqual(r.score, 1.0)

    def test_a_turn_before_loading_counts_for_nothing(self):
        f = Film().add(5, start_items()).turn(300.0, 100, start_items())
        f.add(10, loaded_items()).add(30, loaded_items())
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.turned)

    def test_a_turn_with_one_item_not_yet_loaded_counts_for_nothing(self):
        part = loaded_items(bowl_right=item("bowl_right", L.START["bowl_right"][:2], on="table"))
        f = Film().add(5, part).turn(300.0, 100, part).add(10, loaded_items())
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_back_and_forth_does_not_add_up(self):
        f = Film().add(5, loaded_items())
        for _ in range(3):
            f.turn(150.0, 40, loaded_items()).turn(-150.0, 40, loaded_items())
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertAlmostEqual(math.degrees(t.best_turn), 150.0, places=6)

    def test_a_jar_moved_11_mm_mid_turn_restarts_the_stretch(self):
        f = Film().add(5, loaded_items()).turn(120.0, 40, loaded_items())
        nudged = loaded_items(jar_left=moved(item("jar_left", L.JAR_PLACE["jar_left"]), (-0.011, 0.0, 0.0)))
        f.turn(120.0, 40, nudged).add(30, nudged)
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.stretches, 2)
        # 9 mm keeps the stretch: the same turns latch
        f = Film().add(5, loaded_items()).turn(120.0, 40, loaded_items())
        nudged = loaded_items(jar_left=moved(item("jar_left", L.JAR_PLACE["jar_left"]), (-0.009, 0.0, 0.0)))
        f.turn(120.0, 40, nudged).add(30, nudged)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.stretches, 1)

    def test_the_10_mm_rule_is_from_the_stretch_start_not_per_step(self):
        """Creeping 1 mm a step adds up: after 11 steps the stretch ends."""
        f = Film().add(5, loaded_items())
        base = item("bowl_left", L.BOWL_PLACE["bowl_left"])
        for i in range(1, 41):
            f.turn(5.0, 1, loaded_items(bowl_left=moved(base, (0.001 * i, 0.0, 0.0))))
        r, t = play(f)
        self.assertFalse(t.turned)
        self.assertGreater(t.stretches, 1)

    def test_a_latched_turn_then_a_jar_moved_into_the_sweep_is_quarter(self):
        f = procedure()
        f.add(30, loaded_items(jar_right=at_hub("jar_right", 0.125, phi_deg=-135.0)))
        r, t = play(f)
        self.assertGreaterEqual(math.degrees(t.best_turn), 180.0)
        self.assertEqual(t.first_half_step, 15 + 90)  # the half turn was made ...
        self.assertFalse(t.turned)  # ... but not with the load as it stands now
        self.assertEqual(r.score, 0.25)

    def test_the_load_must_stay_as_it_was_turned(self):
        """Placing then disturbing: moving an item more than 10 mm after the half turn (even to another allowed
        place) ends the stretch the turn was made in: 0.5 until the arm is turned half a turn again."""
        for name, d, want in (("jar_left", (-0.0, 0.011, 0.0), 0.5), ("jar_left", (0.0, 0.009, 0.0), 1.0),
                              ("bowl_right", (0.011, 0.0, 0.0), 0.5), ("bowl_right", (0.0, 0.0095, 0.0), 1.0)):
            f = procedure()
            s = moved(item(name, place_xy(name)), d)
            self.assertTrue(S.loaded(name, s) and (name not in L.JARS or S.jar_clear(s)))
            f.add(30, loaded_items(**{name: s}))
            self.assertEqual(play(f)[0].score, want, (name, d))
        # turned half a turn again with the load as it now stands: 1.0 again
        now = loaded_items(jar_left=moved(item("jar_left", L.JAR_PLACE["jar_left"]), (0.0, 0.011, 0.0)))
        f = procedure().add(30, now)
        self.assertEqual(play(f)[0].score, 0.5)
        f.turn(-200.0, 60, now).add(20, now)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.stretches, 2)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f)
        items, th, a = f.frames[-1]
        once = [t.result(items, a, th).score for _ in range(5)]
        self.assertEqual(once, [1.0] * 5)
        self.assertEqual(t.steps, len(f.frames))

    def test_success_needs_half_a_second_still(self):
        f = Film().add(5, loaded_items()).turn(200.0, 60, loaded_items())
        base = item("bowl_left", L.BOWL_PLACE["bowl_left"])
        for i in range(1, 9):  # the bowl creeping 0.8 mm a step for 8 steps after the turn: 6.4 mm in 0.4 s
            f.add(1, loaded_items(bowl_left=moved(base, (0.0008 * i, 0.0, 0.0))))
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        f.add(L.STILL_STEPS + 1, loaded_items(bowl_left=moved(base, (0.0064, 0.0, 0.0))))
        r, _ = play(f)
        self.assertTrue(r.success)


# -- zeros and exploits -----------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_holding_a_jar_at_the_end_scores_zero(self):
        f = procedure()
        jar = item("jar_left", L.JAR_PLACE["jar_left"])
        f.add(20, loaded_items(), arms(left=holding(jar)))
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_holding_a_jar_just_above_the_floor_scores_zero(self):
        """Holding instead of placing: a jar held 4 mm up counts as standing on the floor, but the gripper is on it."""
        f = Film().add(5, loaded_items())
        jar = item("jar_right", L.JAR_PLACE["jar_right"], lift=0.004)
        held = loaded_items(jar_right=jar)
        f.turn(200.0, 60, held, arms(right=holding(jar))).add(20, held, arms(right=holding(jar)))
        r, t = play(f)
        self.assertFalse(t.turned)  # the turn was made with a gripper on the load: it never counts
        self.assertEqual(r.score, 0.0)

    def test_a_turn_made_while_the_last_item_is_still_in_the_gripper_counts_for_nothing(self):
        """The reviewer's P1: the last jar set 4 mm above the floor and still gripped while the other arm turns the
        arm 185 degrees; then it is let go (it drops 4 mm) and the gripper leaves: 0.5, not 1.0."""
        jar_up = item("jar_right", L.JAR_PLACE["jar_right"], lift=0.004)
        held_items = loaded_items(jar_right=jar_up)
        grip = arms(right=holding(jar_up))
        self.assertTrue(S.loaded("jar_right", jar_up))
        f = Film().add(5, loaded_items(jar_right=item("jar_right", L.START["jar_right"][:2], on="table")))
        f.add(5, held_items, grip).turn(185.0, 60, held_items, grip)
        f.add(5, loaded_items(), grip)  # let go: it drops 4 mm, the fingers still round it
        f.add(30, loaded_items())  # the gripper away
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)
        self.assertEqual(t.stretches, 1)  # the only stretch starts once the gripper is clear
        self.assertEqual(t.best_turn, 0.0)
        # the same with the turn made after the gripper has left: 1.0
        f.turn(185.0, 60, loaded_items()).add(30, loaded_items())
        self.assertEqual(play(f)[0].score, 1.0)

    def test_an_open_finger_resting_on_a_jar_at_the_end_scores_zero(self):
        """The reviewer's P2: loaded and turned, then an open gripper with one finger against a jar's side (the
        point between its tips 72 mm from the jar's centre): 0. Shut fingers 3 cm off the side are on it too, a
        gripper 2.1 cm clear of its fingers' reach is not."""
        base = loaded_items()
        c = base["jar_left"].pos
        f = procedure()
        f.add(30, base, arms(left=tips_at((c[0] + 0.025 + 0.047, c[1]), L.TOP + 0.080, opening=1.0)))
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["held"], ["left:jar_left"])
        f = procedure().add(30, base, arms(left=tips_at((c[0] + 0.030 + 0.025, c[1]), L.TOP + 0.080, opening=0.0)))
        self.assertEqual(play(f)[0].score, 0.0)
        clear = L.JAR_R + S.finger_reach(arm(PARKED["left"], 1.0)) + 0.021
        f = procedure().add(30, base, arms(left=tips_at((c[0] - clear * 0.6, c[1] + clear * 0.8), L.TOP + 0.080)))
        self.assertEqual(play(f)[0].score, 1.0)

    def test_a_hand_on_the_load_during_the_turn_ends_the_stretch(self):
        """Steadying a jar with a gripper while the arm turns (it does not move): the stretch ends when the gripper
        touches it and starts again once it has let go, so the turn counts only from then."""
        jar = item("jar_left", L.JAR_PLACE["jar_left"])
        f = Film().add(5, loaded_items()).turn(100.0, 30, loaded_items())
        f.turn(100.0, 30, loaded_items(), arms(left=holding(jar))).turn(100.0, 30, loaded_items())
        f.add(30, loaded_items())
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.stretches, 2)
        self.assertAlmostEqual(math.degrees(t.best_turn), 100.0, places=6)

    def test_turn_move_an_item_turn_again(self):
        """The reviewer's second-turn case: a half turn, a jar moved 15 mm (still clear), another half turn the other
        way: 1.0, counted in the second stretch (the real trial redraws the outlines after a move of more than 10
        mm and counts the cap from there)."""
        base = loaded_items()
        jm = moved(base["jar_left"], (-0.010, 0.011, 0.0))
        self.assertTrue(S.jar_clear(jm))
        it2 = loaded_items(jar_left=jm)
        f = Film().add(5, base).turn(185.0, 60, base).add(5, it2).turn(-185.0, 60, it2).add(30, it2)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.stretches, 2)
        # without the second turn: 0.5
        f = Film().add(5, base).turn(185.0, 60, base).add(35, it2)
        self.assertEqual(play(f)[0].score, 0.5)

    def test_a_jar_in_the_sweep_that_is_tipped_scores_zero(self):
        jar = at_hub("jar_left", 0.120)
        f = Film().add(5, loaded_items(jar_left=jar)).turn(60.0, 20, loaded_items(jar_left=jar))
        tipped = item("jar_left", jar.pos[:2] + [0.03, 0.0], tilt=90.0)  # knocked over by the arm
        f.turn(200.0, 60, loaded_items(jar_left=tipped)).add(20, loaded_items(jar_left=tipped))
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.turned)
        # stood up again where it was: 0.25 (still in the sweep, and the turn never counted)
        f.add(20, loaded_items(jar_left=jar))
        r, _ = play(f)
        self.assertEqual(r.score, 0.25)

    def test_a_jar_lying_under_the_arm_is_not_loaded(self):
        """Gaming the height: a jar on its side (50 mm high) would pass under the arm, but it is not standing."""
        lying = item("jar_left", (0.27, 0.06), tilt=90.0)
        self.assertLess(S.top("jar_left", lying) - L.TOP, L.BAR_LOW)
        f = Film().add(5, loaded_items(jar_left=lying)).turn(200.0, 60, loaded_items(jar_left=lying))
        r, t = play(f.add(20, loaded_items(jar_left=lying)))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.turned)

    def test_stacked_or_propped_items_are_not_loaded(self):
        bowl = item("bowl_left", L.BOWL_PLACE["bowl_left"])
        on_bowl = item("jar_left", L.BOWL_PLACE["bowl_left"], lift=L.BOWL_H)
        r, _ = play(Film().add(10, loaded_items(jar_left=on_bowl, bowl_left=bowl)))
        self.assertEqual(r.score, 0.0)
        propped = item("jar_left", (L.INNER_X[0] + 0.03, 0.12), tilt=14.0)  # leaning on the wall
        r, _ = play(Film().add(10, loaded_items(jar_left=propped)))
        self.assertEqual(r.score, 0.0)
        on_wall = item("jar_left", (L.INNER_X[0] + 0.003, 0.12), lift=L.WALL_H)
        r, _ = play(Film().add(10, loaded_items(jar_left=on_wall)))
        self.assertEqual(r.score, 0.0)

    def test_the_wrong_part_in_the_corner(self):
        """The bowl in the corner and the jar beside the hub: the jar is in the sweep (0.25 at best, no turn counts)."""
        swap = loaded_items(jar_left=item("jar_left", L.BOWL_PLACE["bowl_left"]),
                            bowl_left=item("bowl_left", L.JAR_PLACE["jar_left"]))
        f = Film().add(5, swap).turn(200.0, 60, swap).add(20, swap)
        r, t = play(f)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.turned)

    def test_a_jar_in_the_sweep_pushed_clear_by_the_turn_counts_only_from_then(self):
        """A jar half a centimetre into the sweep, shoved 8 mm out by the arm (under the 10 mm stretch rule): the
        turn counts only once both jars are clear."""
        inside = at_hub("jar_left", 0.1325)
        out = at_hub("jar_left", 0.1415)
        f = Film().add(5, loaded_items(jar_left=inside)).turn(90.0, 30, loaded_items(jar_left=inside))
        f.turn(100.0, 30, loaded_items(jar_left=out)).add(20, loaded_items(jar_left=out))
        r, t = play(f)
        self.assertEqual(r.score, 0.5)
        # the stretch starts on the first step with the jar clear, one 3.3 degree step into the second turn
        self.assertAlmostEqual(math.degrees(t.best_turn), 100.0 * 29 / 30, places=6)

    def test_dragging_the_load_round_with_the_arm(self):
        """Every item dragged 2 mm a step as the arm turns: the stretch keeps ending, no half turn is made."""
        f = Film().add(5, loaded_items())
        for i in range(1, 61):
            f.turn(4.0, 1, {n: moved(s, (0.0, 0.002 * i, 0.0)) for n, s in loaded_items().items()})
        r, t = play(f)
        self.assertFalse(t.turned)

    def test_pushing_items_off_the_floor_or_out_of_the_rack(self):
        out = item("bowl_right", (0.44, -0.05), on="table")  # pushed over the far wall onto the table
        r, _ = play(procedure().add(20, loaded_items(bowl_right=out)))
        self.assertEqual(r.score, 0.0)


# -- arms -------------------------------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_unreadable_arms_use_the_last_snapshot(self):
        jar = item("jar_left", L.JAR_PLACE["jar_left"])
        f = procedure()
        f.add(5, loaded_items(), arms(left=holding(jar)))  # last readable: holding the jar
        f.frames += [(loaded_items(), f.theta, None)] * 20
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 20)
        f2 = procedure()
        f2.frames += [(loaded_items(), f2.theta, None)] * 20  # last readable: away
        self.assertEqual(play(f2)[0].score, 1.0)

    def test_arms_never_readable_count_as_clear(self):
        f = procedure()
        f.frames = [(i, th, None) for i, th, _ in f.frames]
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(f.frames))

    def test_non_finite_arms_count_as_unreadable(self):
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0), "right": arm(PARKED["right"])}
        self.assertFalse(S.arms_valid(bad))
        self.assertFalse(S.arms_valid({"left": arm(PARKED["left"])}))
        self.assertFalse(S.arms_valid({"left": arm(PARKED["left"], math.nan), "right": arm(PARKED["right"])}))
        self.assertFalse(S.arms_valid({"left": SimpleNamespace(tips=PARKED["left"], opening=None),
                                       "right": arm(PARKED["right"])}))


# -- robustness -------------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        f = procedure()
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        good = loaded_items()
        for bad_item in (SimpleNamespace(pos=np.array([np.nan, 0, 0]), quat_xyzw=np.array([0, 0, 0, 1.0])),
                         SimpleNamespace(pos=np.array([0.2, 0, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0])),
                         SimpleNamespace(pos=np.array([0.2, 0.1, 0.8]), quat_xyzw=np.zeros(4)),
                         SimpleNamespace(pos=np.array([0.2, 0.1, 0.8]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                                         lin_vel=np.array([np.nan, 0, 0])),
                         SimpleNamespace(quat_xyzw=np.array([0, 0, 0, 1.0])),
                         None):
            before = (t.turned, t.stretch, t.theta, len(t.history))
            r = t.update(loaded_items(jar_left=bad_item), f.theta, arms())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual((t.turned, t.stretch, t.theta, len(t.history)), before)
        for yaw in (math.nan, math.inf, None, "x"):
            r = t.update(good, yaw, arms())
            self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 10)
        r = t.update(good, f.theta, arms())  # and it carries on where it was
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.result({"jar_left": good["jar_left"]}, arms(), f.theta).score, 0.0)  # items missing
        self.assertEqual(t.result(None, arms(), f.theta).score, 0.0)

    def test_wrapped_readings_unwrap_through_plus_minus_pi(self):
        """The link yaw is read in (-pi, pi]: a turn from 100 to 300 degrees passes the wrap and still counts."""
        f = Film().add(5, loaded_items(), theta=math.radians(100.0))
        for i in range(1, 81):
            a = math.radians(100.0 + 200.0 * i / 80)
            f.add(1, loaded_items(), theta=math.atan2(math.sin(a), math.cos(a)))
        f.add(20, loaded_items())
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(math.degrees(t.theta), 300.0, places=6)
        # the other way, from -100 to -300
        f = Film().add(5, loaded_items(), theta=math.radians(-100.0))
        for i in range(1, 81):
            a = math.radians(-100.0 - 200.0 * i / 80)
            f.add(1, loaded_items(), theta=math.atan2(math.sin(a), math.cos(a)))
        r, t = play(f)
        self.assertAlmostEqual(math.degrees(t.theta), -300.0, places=6)
        self.assertTrue(t.turned)

    def test_millimetre_wobble_is_tolerated(self):
        rng = np.random.default_rng(3)
        f = Film().add(5, loaded_items())
        for i in range(1, 101):
            wob = {n: moved(s, rng.uniform(-0.0004, 0.0004, 3)) for n, s in loaded_items().items()}
            f.turn(2.0, 1, wob)
        f.add(20, loaded_items())
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.stretches, 1)

    def test_random_states_do_not_crash(self):
        rng = random.Random(7)
        t = Tracker(HZ)
        for _ in range(400):
            its = {}
            for n in L.ITEMS:
                q = np.array([rng.uniform(-1, 1) for _ in range(4)])
                its[n] = SimpleNamespace(pos=np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 2)]),
                                         quat_xyzw=q)
            tips = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 2)])
            a = {"left": rigstate.Arm(tips + [0, 0, TIPS], tips, rng.random()), "right": arm(PARKED["right"])}
            r = t.update(its, rng.uniform(-10, 10), a if rng.random() < 0.8 else None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(r.details)


# -- reset ------------------------------------------------------------------------------------------------------------
class Reset(unittest.TestCase):
    def latched(self):
        r, t = play(procedure())
        self.assertTrue(t.turned)
        return t, Film(theta=math.radians(200.0))

    def test_teleport_back_to_start_resets(self):
        t, _ = self.latched()
        r = t.update(start_items(), 0.0, arms())  # everything back on its start spot, the arm along x: new episode
        self.assertFalse(t.turned)
        self.assertEqual((t.steps, t.stretches, t.best_turn), (1, 0, 0.0))
        self.assertIsNone(t.first_half_step)
        self.assertEqual(r.score, 0.0)
        r, _ = play(Film().add(30, loaded_items()), t)  # and the next episode is scored on its own: no turn yet
        self.assertEqual(r.score, 0.5)

    def test_teleport_to_a_jittered_start_resets(self):
        t, _ = self.latched()
        jit = {n: moved(s, (0.003, -0.002, 0.0)) for n, s in start_items().items()}
        t.update(jit, math.radians(2.0), arms())
        self.assertEqual((t.steps, t.stretches, t.best_turn), (1, 0, 0.0))
        self.assertIsNone(t.first_half_step)

    def test_gradual_return_does_not_reset(self):
        t, _ = self.latched()
        n0 = t.steps
        f = Film(theta=math.radians(200.0))
        a, b = loaded_items(), start_items()
        for i in range(1, 101):
            f.add(1, {n: moved(a[n], (b[n].pos - a[n].pos) * i / 100) for n in L.ITEMS})
        for i in range(1, 41):
            f.add(1, b, theta=math.radians(200.0 * (1 - i / 40)))
        play(f, t)
        self.assertEqual(t.steps, n0 + 140)  # back at the start, but it got there gradually: same episode
        self.assertGreaterEqual(math.degrees(t.best_turn), 200.0 - 1e-6)
        self.assertIsNotNone(t.first_half_step)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t, _ = self.latched()
        n0 = t.steps
        far = {n: moved(s, (0.05, 0.0, 0.0)) for n, s in start_items().items()}
        t.update(far, 0.0, arms())
        self.assertEqual((t.steps, t.stretches), (n0 + 1, 1))
        self.assertIsNotNone(t.first_half_step)
        t2, _ = self.latched()
        t2.update(start_items(), math.radians(200.0), arms())  # home, but the arm is not back along x
        self.assertEqual(t2.steps, n0 + 1)
        self.assertIsNotNone(t2.first_half_step)

    def test_rigstate_reset_hook_resets_a_tracker(self):
        t, _ = self.latched()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertFalse(t.turned)
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


# -- the offline stand-in for the arm's joint -------------------------------------------------------------------------
class OfflineModel(unittest.TestCase):
    def at(self, phi_deg, h, r=L.PUSH_R):
        a = math.radians(phi_deg)
        return tips_at((HUB[0] + r * math.cos(a), HUB[1] + r * math.sin(a)), L.TOP + h)

    def sweep(self, m, a, b, h=0.075, side="left", n=60, r=L.PUSH_R):
        for i in range(n + 1):
            m.update({side: self.at(a + (b - a) * i / n, h, r), ("right" if side == "left" else "left"):
                      arm(PARKED["right" if side == "left" else "left"])})

    def test_push_anticlockwise(self):
        m = offline_model.ArmModel()
        self.sweep(m, 135.0, 240.0)
        half = math.degrees(math.atan(L.PUSH_HALF / L.PUSH_R))
        self.assertAlmostEqual(math.degrees(m.theta), 240.0 + half - 180.0, places=6)  # the red end at 78.4 deg
        self.assertGreater(m.pushes, 0)

    def test_push_clockwise(self):
        m = offline_model.ArmModel()
        self.sweep(m, -135.0, -240.0)
        half = math.degrees(math.atan(L.PUSH_HALF / L.PUSH_R))
        self.assertAlmostEqual(math.degrees(m.theta), -(240.0 + half - 180.0), places=6)

    def test_no_push_out_of_the_band(self):
        for h, r in ((0.122, L.PUSH_R), (0.030, L.PUSH_R), (0.075, 0.115), (0.075, 0.015)):
            m = offline_model.ArmModel()
            self.sweep(m, 135.0, 240.0, h=h, r=r)
            self.assertEqual(m.theta, 0.0, (h, r))

    def test_the_demo_strokes(self):
        """Push 1 from 135 to 240 degrees, over the arm, push 2 from 40 to 240: about 258 degrees in all."""
        m = offline_model.ArmModel()
        self.sweep(m, 135.0, 240.0)
        self.sweep(m, 240.0, 225.0, n=10)
        self.sweep(m, 225.0, 40.0, h=0.122)
        self.sweep(m, 40.0, 240.0, n=80)
        half = math.degrees(math.atan(L.PUSH_HALF / L.PUSH_R))
        self.assertAlmostEqual(math.degrees(m.theta), 240.0 + half, places=6)

    def test_stops_and_reset_and_bad_arms(self):
        m = offline_model.ArmModel()
        for k in range(5):
            self.sweep(m, 135.0 + 360 * k, 495.0 + 360 * k, n=120)
        self.assertAlmostEqual(m.theta, L.ARM_UPPER)
        m.update(None)
        m.update({"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0)})
        self.assertAlmostEqual(m.theta, L.ARM_UPPER)
        m.reset()
        self.assertEqual((m.theta, m.pushes), (0.0, 0))


# -- the task file -----------------------------------------------------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: the objects, the hooks, the reset hooks, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def spray(self, theta=0.0, source="sim", origin=None):
        spec = self.task.SPRAY
        pose = (*(spec.pos if origin is None else origin), 0.0, 0.0, 0.0, 1.0)
        return self.mk.state_from_poses(spec, pose, joints={"arm": theta}, source=source)

    def objs(self, items, theta=0.0, source="sim"):
        return {**items, "spray": self.spray(theta, source)}

    def feed(self, a):
        rigstate.offline_feed({k: {"tcp": v.tcp, "tips": v.tips, "opening": v.opening} for k, v in a.items()})

    def test_objects(self):
        objs = {o.name: o for o in self.task.OBJECTS}
        self.assertEqual(list(objs), ["rack_floor", "rack_wall_near", "rack_wall_far", "rack_wall_left",
                                      "rack_wall_right", "holder_left", "holder_right", "spray", "jar_left",
                                      "jar_right", "bowl_left", "bowl_right"])
        for n, (lo, hi) in zip(("holder_left", "holder_right"), HOLDER_BOXES):  # where the tests think they are
            o = objs[n]
            self.assertIsNone(o.mass)
            self.assertTrue(np.allclose(np.array(o.pos[:2]) - np.array(o.size[:2]) / 2, lo))
            self.assertTrue(np.allclose(np.array(o.pos[:2]) + np.array(o.size[:2]) / 2, hi))
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP + L.FLOOR_T)  # on the floor plate
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, L.TOP + L.HOLDER_TOP)
        sp = objs["spray"]
        self.assertTrue(sp.fixed_base)
        self.assertEqual([(j.name, j.kind, j.axis) for j in sp.joints], [("arm", "revolute", "Z")])
        j = sp.joints[0]
        self.assertTrue(-2 * math.pi < j.lower < -math.pi - 2.5 and math.pi + 2.5 < j.upper < 2 * math.pi)
        self.assertEqual((j.spring, j.friction), (0.0, 0.0))
        self.assertAlmostEqual(j.damping, L.ARM_DAMPING)
        for n in ("rack_floor", "rack_wall_near", "rack_wall_far", "rack_wall_left", "rack_wall_right", "holder_left",
                  "holder_right"):
            self.assertIsNone(objs[n].mass)
            self.assertGreaterEqual(min(objs[n].size), 0.004)
        for n in L.ITEMS:
            o = objs[n]
            self.assertTrue(0.005 <= o.mass <= 0.3)
            self.assertTrue(0.016 <= 2 * o.radius <= 0.080)  # graspable widths
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP)  # standing on the table
        # the walls' inner faces are where the scorer says, and they do not overlap each other
        near, far = objs["rack_wall_near"], objs["rack_wall_far"]
        self.assertAlmostEqual(near.pos[0] + near.size[0] / 2, L.INNER_X[0])
        self.assertAlmostEqual(far.pos[0] - far.size[0] / 2, L.INNER_X[1])
        left, right = objs["rack_wall_left"], objs["rack_wall_right"]
        self.assertAlmostEqual(left.pos[1] - left.size[1] / 2, L.INNER_Y[1])
        self.assertAlmostEqual(right.pos[1] + right.size[1] / 2, L.INNER_Y[0])
        self.assertAlmostEqual(left.pos[0] - left.size[0] / 2, near.pos[0] + near.size[0] / 2)
        # the arm: its underside over the hub, its ends' corners sweep R_TIP
        st = self.spray(theta=1.0)
        lo, hi = st.link_bounds("arm")
        self.assertAlmostEqual(lo[2], L.TOP + L.BAR_LOW)
        self.assertAlmostEqual(hi[2], L.TOP + L.CAP_TOP)
        blo, bhi = sp.local_bounds(bodies=("base",))
        self.assertAlmostEqual(bhi[2], L.HUB_TOP)
        self.assertAlmostEqual(blo[2], L.FLOOR_T)
        self.assertAlmostEqual(L.R_TIP, 0.11045, places=5)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, L.SUCCESS_HOLD_S)

    def test_hooks_read_the_link_yaw_in_the_simulator(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        for i in range(0, 41):  # 0 to 200 degrees: the link pose, read through the wrap at 180
            self.task.success(self.objs(loaded_items(), theta=math.radians(5.0 * i)))
        self.assertAlmostEqual(tr.last.details["turn_deg"], 200.0, places=3)
        self.assertTrue(tr.turned)
        self.assertEqual(self.task.score(self.objs(loaded_items(), theta=math.radians(200.0))), 1.0)

    def test_offline_states_use_the_model(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertEqual(self.task._MODEL.theta, 0.0)
        for i in range(31):  # the left gripper's tips swept from 135 to 200 degrees round the hub, in the push band
            a = math.radians(135.0 + 65.0 * i / 30)
            self.feed(arms(left=tips_at((HUB[0] + L.PUSH_R * math.cos(a), HUB[1] + L.PUSH_R * math.sin(a)),
                                        L.TOP + 0.075, 0.0)))
            self.task.success(self.objs(loaded_items(), source="offline"))
        half = math.degrees(math.atan(L.PUSH_HALF / L.PUSH_R))
        self.assertAlmostEqual(tr.last.details["turn_deg"], 200.0 + half - 180.0, delta=0.01)
        self.task.score(self.objs(loaded_items(), source="offline"))  # score() reads the model without moving it
        self.assertAlmostEqual(math.degrees(self.task._MODEL.theta), 200.0 + half - 180.0, places=3)
        rigstate.offline_reset()  # rigstate's reset resets the model too
        self.assertEqual(self.task._MODEL.theta, 0.0)

    def test_broken_spray_state_scores_zero(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        bad = self.spray()
        bad.links["arm"] = np.array([0.27, 0.0, L.TOP, 0.0, 0.0, 0.0, 0.0])  # zero quaternion
        self.assertFalse(self.task.success({**loaded_items(), "spray": bad}))
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(self.task.score({**loaded_items(), "spray": bad}), 0.0)
        tilted = self.spray()
        tilted.links["arm"] = np.array([0.27, 0.0, L.TOP, *about("y", 60.0)])
        self.assertEqual(self.task.score({**loaded_items(), "spray": tilted}), 0.0)
        self.assertEqual(self.task.score({**loaded_items(), "spray": SimpleNamespace(links={})}), 0.0)
        self.assertEqual(self.task.score(loaded_items()), 0.0)  # no spray state at all
        self.assertEqual(self.task.score({"spray": self.spray()}), 0.0)  # no items

    def test_reset_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        self.task.success(self.objs(start_items()))
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.task.success(self.objs(start_items()))  # arms None right after a reset
        self.assertEqual(tr.arms_unavailable, 1)
        self.feed(arms())
        self.task.success(self.objs(start_items()))
        self.assertEqual(tr.steps, 2)
        self.task._MODEL.theta = 0.5
        self.mk.offline_reset()  # mechkit's reset counter: the second, independent signal
        self.task.success(self.objs(start_items()))
        self.assertEqual(tr.steps, 1)
        self.assertEqual(self.task._MODEL.theta, 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed(arms())
            self.assertFalse(self.task.success(self.objs(start_items())))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_switches(self):
        task = self.task
        self.assertEqual(set(task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_plan(self):
        task = self.task
        plan = task.demo()

        def times(wps):
            out, t = [], 0.0
            for w in wps:
                t += w.duration
                out.append((t - w.duration, t, w))
            return out

        left, right = times(plan["left"]), times(plan["right"])
        # every grip and release takes 1.5 s after a settle of 0.5 s or more at the same place; slow descents
        for wps in (plan["left"], plan["right"]):
            for a, b in zip(wps, wps[1:]):
                if abs(b.finger - a.finger) > 0.01 and not b.label.startswith("shut the fingers"):
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertTrue(a.label == "settle" and a.duration >= 0.5, (a.label, b.label))
                    self.assertEqual(tuple(a.pos), tuple(b.pos))
                if b.label.endswith("slowly"):
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertLessEqual(a.pos[2] - b.pos[2], 0.050, b.label)
        # the left arm pushes only once the right arm has parked
        parked = [t1 for t0, t1, w in right if w.label.startswith("park")][0]
        first_push = min(t0 for t0, t1, w in left if w.label.startswith("down beside the spray arm"))
        self.assertGreaterEqual(first_push, parked)
        # the two arms are never both low in the rack within 10 cm of each other along x
        def low_in_rack(seq, t):
            for t0, t1, w in seq:
                if t0 <= t <= t1:
                    return w.pos if w.pos[2] < L.TOP + 0.10 and L.INNER_X[0] < w.pos[0] < L.INNER_X[1] \
                        and abs(w.pos[1]) < L.INNER_Y[1] else None
            return None
        for k in range(int(parked * 10)):
            pl, pr = low_in_rack(left, k / 10), low_in_rack(right, k / 10)
            if pl is not None and pr is not None:
                self.assertGreater(abs(pl[0] - pr[0]), 0.10, k / 10)
        # the pushes stay on the circle at the push height, with shut fingers
        for t0, t1, w in left:
            if w.label.startswith("push"):
                r = math.hypot(w.pos[0] - HUB[0], w.pos[1] - HUB[1])
                self.assertAlmostEqual(r, L.PUSH_R, places=9)
                self.assertAlmostEqual(w.pos[2], L.TOP + task.PUSH_Z)
                self.assertEqual(w.finger, 0.0)
        # the turn the strokes make, by the offline model: past 180 with room to spare, short of the stops
        half = math.degrees(math.atan(L.PUSH_HALF / L.PUSH_R))
        total = task.PUSH2[1] + half
        self.assertTrue(220.0 <= total <= math.degrees(L.ARM_UPPER) - 60.0, total)

    def test_text_agrees(self):
        """task.py, task.json and both README commands give the model the same instruction; task.json is complete
        and agrees with layout.py on the numbers that matter."""
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        self.assertIn("demo: 1.0", meta["scores"])
        self.assertNotIn("media", meta)
        for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                    "fidelity", "reset_notes"):
            self.assertTrue(meta[key], key)
        steps = int(meta["run_limit"].split()[0].replace(",", ""))
        self.assertIn(f"--max-steps {steps}", readme)
        demo_steps = math.ceil(max(sum(w.duration for w in v) for v in self.task.demo().values()) * HZ)
        self.assertTrue(3 * demo_steps <= steps <= 6 * demo_steps)
        sc = meta["scoring"]
        for text in ("14.05 cm", "11.05 cm", "11.55 cm", "8.5 to 10.9 cm", "10 degrees", "5 mm", "10 mm",
                     "180 degrees", "2 cm", "1.68 cm", "5.85 cm", "Possible scores: 0, 0.25, 0.5, 1.0", "0.5 s", "1 s"):
            self.assertIn(text, sc)
        for text in ("8.5 x 8.5 cm", "5 cm tall", "33.5 to 42.0", "6.5 to 15.0"):
            self.assertIn(text, meta["setup"])
        self.assertIn("355 degrees", meta["fidelity"])
        self.assertAlmostEqual(math.degrees(L.ARM_UPPER), 355.2, places=1)
        for text in (json.dumps(meta), readme):
            self.assertNotIn("—", text)  # no em-dashes
            self.assertNotIn("–", text)


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo played through our offline checker's kinematics (the arm's angle from offline_model)."""

    def test_demo_predicts_one_and_each_arm_alone_zero(self):
        import subprocess
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("TASK_TRACE", None)
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--one-arm", "--json", "-"],
                              capture_output=True, text=True, env=env, timeout=1200)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        rep = json.loads(proc.stdout)
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 3600)
        self.assertEqual(rep["one_arm"], {"left_only": 0.0, "right_only": 0.0})
        for side in ("left", "right"):
            self.assertLess(rep["playback"]["arms"][side]["worst_ik_mm"], 0.5)
        self.assertGreater(rep["playback"]["min_arm_clearance_m"], 0.05)
        self.assertEqual(proxy["final_hooks"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
