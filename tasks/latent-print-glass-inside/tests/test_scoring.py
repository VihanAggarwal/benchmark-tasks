"""Offline tests for scoring.py, offline_model.py and the hooks in task.py: synthetic glass states and synthetic arm
snapshots (rigstate.Arm, or fed through rigstate.offline_feed), no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import offline_model as M  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
HH = L.GLASS_HALF_H
DEPTH = 0.042  # the demo's tips below the rim while gripping
NEAR, CONTACT, GRIP, OPEN = (L.NEAR_CLOSED_GAP / L.FULL_GAP, L.CONTACT_GAP / L.FULL_GAP, L.GRIP_GAP / L.FULL_GAP,
                             1.0)  # rigstate openings
READY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def glass(x, y, bottom, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like glass whose lowest point is at ``bottom`` (world z), centred on (x, y)."""
    r = S.rotation(quat)
    half = np.array([L.GLASS_HALF, L.GLASS_HALF, HH])
    ext = np.abs(r) @ half
    c = np.array([x, y, bottom + ext[2]], float)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


ON_MARK = glass(L.START_X, L.START_Y, L.MARK_TOP)
ON_SHEET = glass(L.SHEET_X, L.SHEET_Y, L.SHEET_TOP)
IN_BOX = glass(L.BOX_X, L.BOX_Y, L.TOP)


def arm(tips, opening=NEAR, tilt=None):
    """A rigstate arm with its tip point at ``tips``, straight down (or ``tilt`` = (axis, deg) about the tips)."""
    tips = np.array(tips, float)
    up = np.array([0.0, 0.0, TIPS])
    if tilt is not None:
        up = S.rotation(about(*tilt)) @ up
    return rigstate.Arm(tcp=tips + up, tips=tips, opening=float(opening))


def tips_in(g, depth=DEPTH, dx=0.0, dy=0.0):
    """The tip point ``depth`` below the rim of glass ``g``, (dx, dy) off its axis."""
    return np.array([g.pos[0] + dx, g.pos[1] + dy, g.high[2] - depth])


def ready(side, opening=OPEN):
    return arm(READY[side] - np.array([0, 0, TIPS]), opening)


def arms(left=None, right=None):
    return {"left": left or ready("left"), "right": right or ready("right")}


def moved(g, dx=0.0, dy=0.0, dz=0.0, vel=None):
    v = (0, 0, 0) if vel is None else vel
    return glass(g.pos[0] + dx, g.pos[1] + dy, g.low[2] + dz, g.quat_xyzw, vel=v)


def path(g0, g1, n, holder=None):
    """n steps carrying the glass from g0 to g1 at a steady speed; ``holder`` ('left'/'right') holds it inside."""
    out = []
    a, b = np.array(g0.pos, float), np.array(g1.pos, float)
    v = (b - a) / (n * DT)
    for i in range(1, n + 1):
        c = a + (b - a) * i / n
        g = glass(c[0], c[1], c[2] - HH, g0.quat_xyzw, vel=v)
        ar = arms(**{holder: arm(tips_in(g), CONTACT)}) if holder else arms()
        out.append((g, ar))
    return out


def run(frames, tracker=None):
    t = tracker or Tracker(HZ)
    r = None
    for g, ar in frames:
        r = t.update(g, ar)
    return r, t


def hold(g, ar, n):
    return [(g, ar)] * n


def procedure(*, skip_sheet=False, end_arms=None, release=True):
    """The task done right, as states: the left arm grips the glass from inside on its mark, carries it to the
    sheet and lets go; the right arm grips it there, carries it into the box and lets go."""
    f = hold(ON_MARK, arms(), 10)
    f += hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), NEAR)), 10)
    f += hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), CONTACT)), 10)
    up = moved(ON_MARK, dz=0.075)
    f += path(ON_MARK, up, 30, "left")
    if skip_sheet:
        over_box = glass(L.BOX_X, L.BOX_Y, L.TOP + 0.085)
        f += path(up, over_box, 40, "left")  # (no arm reaches this; it only tests the scorer)
        f += path(over_box, IN_BOX, 40, "left")
        holder = "left"
    else:
        over = glass(L.SHEET_X, L.SHEET_Y, L.SHEET_TOP + 0.015)
        f += path(up, over, 40, "left")
        f += path(over, ON_SHEET, 30, "left")
        f += hold(ON_SHEET, arms(left=arm(tips_in(ON_SHEET), CONTACT)), 10)
        f += hold(ON_SHEET, arms(left=arm(tips_in(ON_SHEET), NEAR)), 10)
        f += hold(ON_SHEET, arms(left=arm(tips_in(ON_SHEET, depth=-0.03), NEAR)), 10)
        f += hold(ON_SHEET, arms(right=arm(tips_in(ON_SHEET), NEAR)), 10)
        f += hold(ON_SHEET, arms(right=arm(tips_in(ON_SHEET), CONTACT)), 10)
        up = moved(ON_SHEET, dz=0.085)
        f += path(ON_SHEET, up, 30, "right")
        over_box = glass(L.BOX_X, L.BOX_Y, L.TOP + 0.085)
        f += path(up, over_box, 40, "right")
        f += path(over_box, IN_BOX, 40, "right")
        holder = "right"
    f += hold(IN_BOX, arms(**{holder: arm(tips_in(IN_BOX), CONTACT)}), 10)
    if release:
        f += hold(IN_BOX, arms(**{holder: arm(tips_in(IN_BOX), NEAR)}), 10)
        f += hold(IN_BOX, end_arms or arms(**{holder: arm(tips_in(IN_BOX, depth=-0.047), NEAR)}), 30)
    return f


class Geometry(unittest.TestCase):
    def test_glass_shape(self):
        self.assertAlmostEqual(S.lowest(ON_MARK), L.MARK_TOP)
        self.assertAlmostEqual(L.INNER_HALF, 0.031)
        self.assertAlmostEqual(L.CONTACT_GAP, 0.042)
        poly = S.outline(ON_MARK)
        self.assertEqual(len(poly), 4)
        self.assertAlmostEqual(np.ptp(poly[:, 0]), L.GLASS_OUTER)
        self.assertAlmostEqual(S.tilt_deg(about("x", 7)), 7.0, places=4)
        self.assertAlmostEqual(S.tilt_deg(about("y", 180)), 180.0, places=4)  # upside down is not upright
        loc = S.to_local(ON_MARK, ON_MARK.pos + np.array([0, 0, HH]))
        self.assertTrue(np.allclose(loc, [0, 0, HH]))
        # a tilted glass's lowest point is its lowest corner, and its outline grows
        t = glass(0.3, 0.0, L.TOP + 0.05, about("x", 8))
        self.assertAlmostEqual(S.lowest(t), L.TOP + 0.05)
        self.assertGreater(np.ptp(S.outline(t)[:, 1]), L.GLASS_OUTER)

    def test_mass_and_size_are_buildable(self):
        vol = L.GLASS_OUTER ** 2 * L.GLASS_H - (2 * L.INNER_HALF) ** 2 * (L.GLASS_H - L.GLASS_FLOOR)
        self.assertAlmostEqual(vol * 1190, L.GLASS_MASS, delta=0.01)  # acrylic, 1.19 g/cm3
        self.assertLessEqual(L.GLASS_MASS, 0.3)
        self.assertGreaterEqual(min(L.GLASS_WALL, L.GLASS_FLOOR, L.BOX_WALL), 0.004)

    def test_fingers_fit_the_opening(self):
        # across the closing axis the 40 mm fingers have 11 mm each side; nearly closed, 16 mm along it
        self.assertAlmostEqual(L.INNER_HALF - L.FINGER_HALF_WIDTH, 0.011)
        self.assertAlmostEqual(L.INNER_HALF - (L.NEAR_CLOSED_GAP / 2 + L.FINGER_THICK), 0.016)
        # commanded 4 mm per finger past the walls: about 20 N each at 5000 N/m, under the 50 N cap
        self.assertAlmostEqual((L.GRIP_GAP - L.CONTACT_GAP) / 2 * 5000, 20.0)
        # the commanded grip is well inside the gripper's range
        self.assertLess(L.GRIP_GAP, L.FULL_GAP - 0.03)

    def test_surfaces(self):
        self.assertTrue(S.standing_on(ON_MARK, S.MARK, L.MARK_TOP))
        self.assertTrue(S.on_sheet(ON_SHEET))
        self.assertTrue(S.in_box(IN_BOX))
        self.assertFalse(S.in_box(ON_SHEET))
        self.assertFalse(S.on_sheet(moved(ON_SHEET, dz=0.004)))  # 4 mm above it: not standing on it
        self.assertTrue(S.on_sheet(moved(ON_SHEET, dz=0.0025)))
        edge = 0.070 - 0.037  # 33 mm: the glass's edge on the 140 mm sheet's edge
        self.assertTrue(S.on_sheet(moved(ON_SHEET, dx=edge - 0.0005)))
        self.assertFalse(S.on_sheet(moved(ON_SHEET, dx=edge + 0.0005)))
        self.assertFalse(S.on_sheet(glass(L.SHEET_X, L.SHEET_Y, L.SHEET_TOP, about("y", 6))))
        self.assertTrue(S.in_box(moved(IN_BOX, dy=0.0155 - 0.001)))
        self.assertFalse(S.in_box(moved(IN_BOX, dy=0.0155 + 0.001)))  # over the 105 mm floor's edge

    def test_box_walls(self):
        self.assertAlmostEqual(S.box_gap(IN_BOX), 0.0155)  # 15.5 mm round the glass
        near = moved(IN_BOX, dx=0.0155 - 0.0015)  # 1.5 mm from the front wall
        self.assertTrue(S.touches_box(near))
        self.assertFalse(S.touches_box(moved(IN_BOX, dx=0.0155 - 0.0025)))
        on_wall = glass(L.BOX_X + L.BOX_INNER_HALF, L.BOX_Y, L.TOP + 0.055)  # resting across the front wall's top
        self.assertTrue(S.touches_box(on_wall))
        self.assertTrue(S.touches_box(glass(L.BOX_X + L.BOX_INNER_HALF, L.BOX_Y, L.TOP + 0.055 + 0.0015)))
        over = glass(L.BOX_X + L.BOX_INNER_HALF, L.BOX_Y, L.TOP + 0.055 + 0.0025)  # passing 2.5 mm over it
        self.assertFalse(S.touches_box(over))
        outside = glass(L.BOX_X + L.BOX_OUTER_HALF + L.GLASS_HALF + 0.001, L.BOX_Y, L.TOP + 0.01)
        self.assertTrue(S.touches_box(outside))  # 1 mm from its outside face

    def test_set_down_rule(self):
        self.assertFalse(S.set_down_off_limits(ON_MARK))
        self.assertFalse(S.set_down_off_limits(ON_SHEET))
        self.assertFalse(S.set_down_off_limits(IN_BOX))
        bare = glass(0.40, 0.15, L.TOP)
        self.assertTrue(S.set_down_off_limits(bare))
        self.assertTrue(S.set_down_off_limits(moved(bare, dz=0.0035)))
        self.assertFalse(S.set_down_off_limits(moved(bare, dz=0.0045)))
        edge = L.SHEET_SIZE[0] / 2 - L.GLASS_HALF
        self.assertTrue(S.set_down_off_limits(moved(ON_SHEET, dy=edge + 0.001)))  # over the sheet's edge
        mark_edge = 0.055 - 0.037  # 18 mm of slack on the 110 mm mark
        self.assertFalse(S.set_down_off_limits(moved(ON_MARK, dx=mark_edge - 0.001)))
        self.assertTrue(S.set_down_off_limits(moved(ON_MARK, dx=mark_edge + 0.001)))

    def test_off_table(self):
        self.assertFalse(S.off_table(ON_MARK))
        self.assertTrue(S.off_table(glass(0.3, 0.70, L.TOP + 0.05)))
        self.assertTrue(S.off_table(glass(0.3, 0.2, L.TOP - 0.02)))


def approach(tips, a, opening):
    """A rigstate arm whose axis (tool point to tips) is ``a`` (any direction)."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    tips = np.asarray(tips, float)
    return rigstate.Arm(tcp=tips - TIPS * a, tips=tips, opening=float(opening))


def tilted(deg, toward=(1.0, 0.0)):
    """Axis tipped ``deg`` from straight down, its tips leaning toward the horizontal direction ``toward``."""
    t = math.radians(deg)
    h = np.array([toward[0], toward[1], 0.0]) / math.hypot(*toward)
    return h * math.sin(t) + np.array([0.0, 0.0, -math.cos(t)])


def real_finger_points(arm_, turn, g, n=(3, 7, 9), reach=0.038, per_finger=False):
    """Points on both real finger blocks (their lowest ``reach``) at the TRUE turn ``turn`` (rad, from the glass's x
    side as the scorer measures it), in the glass's frame: the oracle the exploit tests judge against."""
    r = S.rotation(g.quat_xyzw)
    tips = (np.asarray(arm_.tips, float) - np.asarray(g.pos, float)) @ r
    a = S.axis_of(arm_) @ r
    ex = np.array([1.0, 0.0, 0.0]) - a * a[0]
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([0.0, 1.0, 0.0]) - a * a[1]
    ex /= np.linalg.norm(ex)
    u = math.cos(turn) * ex + math.sin(turn) * np.cross(a, ex)
    v = np.cross(a, u)
    gap = S.finger_gap(arm_)
    fingers = []
    for sign in (-1, 1):
        out = []
        for k in np.linspace(gap / 2, gap / 2 + 0.010, n[0]):
            for w in np.linspace(-0.020, 0.020, n[1]):
                for up in np.linspace(0.0, reach, n[2]):
                    out.append(tips + sign * k * u + w * v - up * a)
        fingers.append(np.array(out))
    return fingers if per_finger else np.concatenate(fingers)


def touches_outside(pts):
    """Some real finger point below the rim (down to 10 mm under the bottom) is beyond the inner walls by more than
    the 5 mm allowance and within 10 mm of the glass: on its outside, in its outer wall, or within 1 cm of it."""
    m = np.max(np.abs(pts[:, :2]), axis=1)
    band = (pts[:, 2] <= HH) & (pts[:, 2] >= -HH - 0.010)
    return bool(np.any(band & (m > 0.031 + 0.005) & (m <= 0.037 + 0.010)))


def impossible(fingers):
    """A real finger (its whole 70 mm, as ``real_finger_points(..., reach=0.070, per_finger=True)``) through the
    acrylic: below the rim both inside the opening and more than 5 mm into or beyond the walls (the straight path
    between two such points stays below the rim and crosses the wall), or in the floor or under the bottom while also
    in the opening. The walls stop real fingers there, so no rule is asked of such poses."""
    for pts in fingers:
        m = np.max(np.abs(pts[:, :2]), axis=1)
        below = pts[:, 2] <= HH
        inside = below & (m < 0.031) & (pts[:, 2] > -HH + 0.006)
        if np.any(inside) and np.any(below & (m > 0.036) & (pts[:, 2] >= -HH - 0.010)):
            return True
        if np.any(inside) and np.any((m < 0.037) & (pts[:, 2] < -HH + 0.006) & (pts[:, 2] > -HH - 0.010)):
            return True
    return False


class Fingers(unittest.TestCase):
    """The finger rule: below the rim, both fingers inside the opening at some turn, or at every turn at least 10 mm
    from the glass."""

    def status(self, a, g=ON_MARK):
        return S.finger_check(a, g)["status"]

    def test_internal_grip_is_inside(self):
        for op in (0.0, NEAR, CONTACT):
            self.assertEqual(self.status(arm(tips_in(ON_MARK), op)), "inside")
        # pressing the walls with up to 5 mm of the block in them still counts (fingers on the walls); the fingers'
        # outer faces 31 mm from the axis at 42 mm, 36 mm at 52 mm
        self.assertEqual(self.status(arm(tips_in(ON_MARK), (0.052 - 0.0004) / L.FULL_GAP)), "inside")
        self.assertEqual(self.status(arm(tips_in(ON_MARK), (0.052 + 0.0004) / L.FULL_GAP)), "outside")

    def test_above_the_rim_is_not_judged(self):
        self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=-0.001), OPEN)), "above")
        self.assertEqual(self.status(ready("left")), "above")

    def test_outside_grip_breaks_it(self):
        external = arm(tips_in(ON_MARK, depth=0.030), (L.GLASS_OUTER - 0.008) / L.FULL_GAP)  # pads across 74 mm
        self.assertEqual(self.status(external), "outside")
        self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=0.030), 0.074 / L.FULL_GAP)), "outside")

    def test_pinching_the_rim_breaks_it(self):
        x = L.INNER_HALF + L.GLASS_WALL / 2  # over the wall's middle, fingers closed on its 6 mm
        pinch = arm(tips_in(ON_MARK, depth=0.010, dx=x), 0.0)
        self.assertEqual(self.status(pinch), "outside")
        self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=0.002, dx=x), 0.0)), "outside")

    def test_one_cm_clearance_at_every_turn(self):
        # closed fingers (numbers as task.json states them: the glass 37 mm from its axis to its outside, 10 mm
        # clearance; fingers 10 mm thick outside the gap and 40 mm wide, so they sweep 22.4 mm round the axis)
        reach = math.hypot(0.010, 0.020)
        for d, want in ((0.037 + 0.010 + reach + 0.0005, "away"), (0.037 + 0.010 + reach - 0.0005, "outside")):
            self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=0.03, dx=d), 0.0)), want, d)
            self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=0.03, dy=-d), 0.0)), want, d)
        # off a corner the distance is to the corner (diagonally)
        k = (0.037 + 0.010) + (reach + 0.0005) / math.sqrt(2)
        self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=0.03, dx=k, dy=k), 0.0)), "away")
        # fully open: they sweep 60.4 mm round the axis, so the tip point must be 107.4 mm off below the rim
        reach = math.hypot(0.0939 / 2 + 0.010, 0.020)
        for d, want in ((0.037 + 0.010 + reach + 0.0005, "away"), (0.037 + 0.010 + reach - 0.0005, "outside")):
            self.assertEqual(self.status(arm(tips_in(ON_MARK, depth=0.03, dx=d), OPEN)), want, d)

    def test_turn_is_searched_for_inside(self):
        # fingers 44 mm apart span 64 x 40 mm: they fit with the closing axis near x or near y, not at 45 deg
        out = S.finger_check(arm(tips_in(ON_MARK), 0.044 / L.FULL_GAP), ON_MARK)
        self.assertEqual(out["status"], "inside")
        self.assertLess(out["turns_ok"], len(S.HEADINGS))
        self.assertEqual(len(S.HEADINGS), 36)  # 0 .. 175 deg
        # the same fingers 10 mm off the axis both ways fit at no turn: a finger is in the wall
        self.assertEqual(self.status(arm(tips_in(ON_MARK, dx=0.010, dy=0.010), 0.044 / L.FULL_GAP)), "outside")

    def test_turned_glass(self):
        turned = glass(L.START_X, L.START_Y, L.MARK_TOP, about("z", 30))
        self.assertEqual(self.status(arm(tips_in(turned), CONTACT), turned), "inside")

    def test_finger_blocks_follow_the_gripper(self):
        a = arm(tips_in(ON_MARK), NEAR, tilt=("x", 30))
        corners, seg, ax, radius = S.finger_blocks(a, ON_MARK)
        self.assertEqual(corners.shape, (36, 2, 8, 3))
        self.assertAlmostEqual(float(np.linalg.norm(seg[1] - seg[0])), L.FINGER_LEN)
        self.assertAlmostEqual(radius, math.hypot(0.005 + 0.010, 0.020))
        for block in corners[7]:  # each block is a 10 x 40 mm finger running 70 mm up to its carriage
            d = np.sort([np.linalg.norm(block[i] - block[0]) for i in (1, 2, 4)])
            self.assertTrue(np.allclose(d, [0.010, 0.040, 0.070]), d)
        # straight down at turn 0 the closing axis is the glass's x side
        c0, _, _, _ = S.finger_blocks(arm(tips_in(ON_MARK), CONTACT), ON_MARK)
        loc = c0[0, 1]
        self.assertAlmostEqual(float(np.max(np.abs(loc[:, 0]))), 0.031)
        self.assertAlmostEqual(float(np.max(np.abs(loc[:, 1]))), 0.020)

    def test_under_the_base_counts(self):
        lifted = moved(ON_MARK, dz=0.05)
        under = arm(np.array([lifted.pos[0] + 0.03, lifted.pos[1], lifted.low[2] - 0.005]), 0.0)
        self.assertEqual(self.status(under, lifted), "outside")
        # a sideways gripper under the lifted glass: its 40 mm-wide fingers sweep 22.4 mm round its axis, so with
        # the axis 5 mm under the bottom they count; 33 mm under they are more than 10 mm away at any turn
        for gap, want in ((0.005, "outside"), (0.033, "away")):
            flat = arm(np.array([lifted.pos[0], lifted.pos[1], lifted.low[2] - gap]), 0.0, tilt=("x", 90))
            self.assertEqual(self.status(flat, lifted), want, gap)
        # a gripper pointing up, its closed fingers pressing the bottom from below in the middle (within the
        # opening's outline, but under the floor): on the outside
        for d in (0.0, 0.005):
            up = approach(np.array([lifted.pos[0], lifted.pos[1], lifted.low[2] - d]), (0.0, 0.0, 1.0), 0.0)
            self.assertEqual(self.status(up, lifted), "outside", d)
        # fingers in the opening's outline but below the inner floor are in its floor or under it, not inside
        down = approach(np.array([lifted.pos[0], lifted.pos[1], lifted.low[2] - 0.005]), (0.0, 0.0, -1.0), 0.0)
        self.assertEqual(self.status(down, lifted), "outside")

    def test_tilted_legit_grips(self):
        """An internal grip at the contact gap stays 'inside' with the gripper tipped up to 8 degrees in any
        direction and 16 along the glass's sides (wrist sag is under 1), or its tip point 7 mm off the glass's axis
        in any direction (the pads are said to sit up to 8 mm off the commanded point; the fingers centre the glass
        along the closing axis); tipped 20 degrees along a side it no longer fits."""
        for t, towards in ((4, ((1, 0), (0, 1), (1, 1), (-1, 1))), (8, ((1, 0), (0, 1), (1, 1), (-1, 1))),
                           (16, ((1, 0), (0, 1), (-1, 0)))):
            for toward in towards:
                a = approach(tips_in(ON_SHEET), tilted(t, toward), CONTACT)
                self.assertEqual(self.status(a, ON_SHEET), "inside", (t, toward))
        self.assertEqual(self.status(approach(tips_in(ON_SHEET), tilted(20, (1, 0)), CONTACT), ON_SHEET), "outside")
        for deg in range(0, 360, 15):
            d = (0.007 * math.cos(math.radians(deg)), 0.007 * math.sin(math.radians(deg)))
            self.assertEqual(self.status(arm(tips_in(ON_SHEET, dx=d[0], dy=d[1]), CONTACT), ON_SHEET), "inside", deg)

    def test_holding_and_let_go(self):  # holding() is reported in the trace only
        self.assertTrue(S.holding(arm(tips_in(ON_SHEET), CONTACT), ON_SHEET))
        self.assertFalse(S.holding(arm(tips_in(ON_SHEET), NEAR), ON_SHEET))
        self.assertFalse(S.holding(arm(tips_in(ON_SHEET, depth=-0.005), CONTACT), ON_SHEET))
        self.assertFalse(S.let_go(arm(tips_in(ON_SHEET), NEAR), ON_SHEET))
        self.assertFalse(S.let_go(arm(tips_in(ON_SHEET, depth=-0.0195), NEAR), ON_SHEET))
        self.assertTrue(S.let_go(arm(tips_in(ON_SHEET, depth=-0.0205), NEAR), ON_SHEET))
        self.assertTrue(S.let_go(arm(tips_in(ON_SHEET, depth=0.02, dx=0.061), 0.0), ON_SHEET))
        self.assertFalse(S.let_go(arm(tips_in(ON_SHEET, depth=0.02, dx=0.059), 0.0), ON_SHEET))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(hold(ON_MARK, arms(), 20))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = run(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(t.examined and t.lifted)
        self.assertEqual(t.breaches, {})

    def test_score_is_idempotent(self):
        r, t = run(procedure())
        g, ar = procedure()[-1]
        steps = t.steps
        self.assertEqual(t.result(g, ar).score, t.result(g, ar).score)
        self.assertEqual(t.steps, steps)

    def test_gripper_still_inside_scores_075(self):
        r, _ = run(procedure(release=False))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        r, _ = run(procedure(end_arms=arms(right=arm(tips_in(IN_BOX, depth=0.03), NEAR))))  # closed, still inside
        self.assertEqual(r.score, 0.75)

    def test_skipping_the_sheet_scores_075(self):
        r, t = run(procedure(skip_sheet=True))
        self.assertFalse(t.examined)
        self.assertEqual(r.score, 0.75)

    def test_on_the_sheet_scores_half(self):
        f = procedure()
        cut = next(i for i, (g, ar) in enumerate(f) if S.on_sheet(g) and ar["right"].opening == CONTACT)
        r, _ = run(f[:cut] + hold(ON_SHEET, arms(), 20))
        self.assertEqual(r.score, 0.5)

    def test_moving_on_the_sheet_is_not_standing_on_it(self):
        f = procedure()
        cut = next(i for i, (g, ar) in enumerate(f) if S.on_sheet(g) and ar["right"].opening == CONTACT)
        slid = moved(ON_SHEET, vel=(0.02, 0.0, 0.0))
        r, _ = run(f[:cut] + hold(ON_SHEET, arms(), 5) + [(slid, arms())])
        self.assertEqual(r.score, 0.25)
        # brushing over the sheet without stopping on it does not count as having stood on it
        up = moved(ON_MARK, dz=0.075)
        low = glass(L.SHEET_X, L.SHEET_Y + 0.02, L.SHEET_TOP + 0.002)
        g = hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), CONTACT)), 5) + path(ON_MARK, up, 20, "left")
        g += path(up, low, 30, "left") + path(low, moved(low, dy=-0.04), 10, "left")
        _, t = run(g)
        self.assertFalse(t.examined)

    def test_lifted_and_put_back_scores_quarter(self):
        up = moved(ON_MARK, dz=0.03)
        f = hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), CONTACT)), 5) + path(ON_MARK, up, 20, "left")
        f += path(up, ON_MARK, 20, "left") + hold(ON_MARK, arms(), 20)
        r, t = run(f)
        self.assertTrue(t.lifted)
        self.assertEqual(r.score, 0.25)

    def test_lift_threshold(self):
        f = hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), CONTACT)), 5)
        for dz, want in ((0.020 - 0.001 - 0.0005, False), (0.020 - 0.001 + 0.0005, True)):  # 20 mm over the table
            up = moved(ON_MARK, dz=dz)
            _, t = run(f + path(ON_MARK, up, 10, "left"))
            self.assertEqual(t.lifted, want)

    def test_hanging_in_the_box_is_not_in_it(self):
        f = procedure(release=False)
        hang = moved(IN_BOX, dz=0.010)
        r, _ = run(f + path(IN_BOX, hang, 5, "right") + hold(hang, arms(right=arm(tips_in(hang), CONTACT)), 20))
        self.assertEqual(r.score, 0.25)

    def test_success_needs_stillness(self):
        f = procedure()
        g, ar = f[-1]
        r, _ = run(f + [(moved(g, vel=(0.0, 0.02, 0.0)), ar)])
        self.assertEqual(r.score, 0.25)  # moving: not 'standing' in the box this step
        self.assertFalse(r.success)

    def test_tilted_in_the_box(self):
        f = procedure()
        _, ar = f[-1]
        r, _ = run(f + [(glass(L.BOX_X, L.BOX_Y, L.TOP, about("x", 6)), ar)])
        self.assertLess(r.score, 0.75)


class Zeros(unittest.TestCase):
    """Every whole-trial zero, and that it is never undone."""

    def test_fingers_on_the_outside(self):
        external = arm(tips_in(ON_MARK, depth=0.03), (L.GLASS_OUTER - 0.008) / L.FULL_GAP)
        r, t = run(hold(ON_MARK, arms(left=external), 3) + procedure())
        self.assertEqual(r.score, 0.0)
        self.assertIn("left_fingers_outside", t.breaches)

    def test_other_gripper_near_the_outside(self):
        f = procedure()
        i = next(k for k, (g, ar) in enumerate(f) if S.on_sheet(g))
        g, _ = f[i]
        beside = arm(np.array([g.pos[0], g.pos[1] - L.GLASS_HALF - 0.015, g.low[2] + 0.04]), 0.0)
        f.insert(i, (g, arms(right=beside)))
        r, t = run(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_fingers_outside", t.breaches)

    def test_touching_a_box_wall(self):
        f = procedure()
        i = next(k for k, (g, _) in enumerate(f) if S.in_box(g))
        g, ar = f[i]
        f.insert(i, (moved(g, dy=L.BOX_INNER_HALF - L.GLASS_HALF - 0.001, dz=0.02), ar))
        r, t = run(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("touched_box", t.breaches)

    def test_tipped(self):
        for deg, broken in ((9.5, False), (10.5, True)):
            g = glass(L.START_X, L.START_Y, L.TOP + 0.05, about("y", deg))
            _, t = run(hold(g, arms(left=arm(tips_in(g), CONTACT)), 3))
            self.assertEqual("tipped" in t.breaches, broken, deg)

    def test_set_down_on_the_bare_table(self):
        bare = glass(0.33, 0.15, L.TOP)
        f = procedure()
        f.insert(60, (bare, arms(left=arm(tips_in(bare), CONTACT))))
        r, t = run(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("set_down_off_limits", t.breaches)

    def test_dropped(self):
        up = moved(ON_MARK, dz=0.05)
        lift = hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), CONTACT)), 3) + path(ON_MARK, up, 10, "left")
        # (glass speed, tips' speed down with it, the zero?): the glass falling while the tips stay put, or are
        # nowhere near it, is dropped; the gripper lowering it fast is not
        for v, tips_v, inside, broken in ((-0.46, 0.0, True, True), (-0.44, 0.0, True, False),
                                         (-0.50, 0.40, True, False), (-0.50, 0.30, True, True),
                                         (-0.50, 0.50, False, True)):
            g = moved(up, dz=-0.01, vel=(0.0, 0.0, v))
            tp = tips_in(up) - np.array([0.0, 0.0, tips_v * DT])
            if not inside:
                tp = tp + np.array([0.1, 0.0, 0.0])
            _, t = run(lift + [(g, arms(left=arm(tp, NEAR)))])
            self.assertEqual("dropped" in t.breaches, broken, (v, tips_v, inside))

    def test_off_the_table(self):
        r, t = run(hold(ON_MARK, arms(), 2) + [(glass(0.3, 0.70, L.TOP - 0.3), arms())])
        self.assertIn("off_table", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_zero_is_never_undone(self):
        f = procedure()
        i = 75  # on its way to the sheet, past the mark's edge: lowered onto the bare table and lifted again
        g = f[i][0]
        low = glass(g.pos[0], g.pos[1], L.TOP)
        self.assertFalse(S.wholly_on(S.outline(low), S.MARK))
        r, t = run(f[:i] + path(g, low, 20, "left") + path(low, g, 20, "left") + f[i:])
        self.assertIn("set_down_off_limits", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)


class Exploits(unittest.TestCase):
    """Unintended strategies: each must score 0 or little."""

    def test_ordinary_outside_grip_and_carry(self):
        g0 = ON_MARK
        outside = (L.GLASS_OUTER - 0.008) / L.FULL_GAP  # pads closed across the 74 mm outside
        f = hold(g0, arms(left=arm(tips_in(g0, depth=0.04), outside)), 10)
        over = glass(L.SHEET_X, L.SHEET_Y, L.SHEET_TOP)
        f += [(g, arms(left=arm(tips_in(g, depth=0.04), outside))) for g, _ in path(g0, over, 40)]
        f += hold(over, arms(), 20)
        r, t = run(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("left_fingers_outside", t.breaches)

    def test_pinch_the_rim(self):
        x = L.INNER_HALF + L.GLASS_WALL / 2
        f = hold(ON_MARK, arms(left=arm(tips_in(ON_MARK, depth=0.012, dx=x), 0.0)), 5) + procedure()
        r, _ = run(f)
        self.assertEqual(r.score, 0.0)

    def test_push_it_from_outside(self):
        f = hold(ON_MARK, arms(), 3)
        g = ON_MARK
        for i in range(30):  # closed fingers against its side, sliding it across the mark toward the sheet
            g = moved(g, dy=-0.002, vel=(0.0, -0.04, 0.0))
            pusher = arm(np.array([g.pos[0], g.pos[1] + L.GLASS_HALF + 0.011, g.low[2] + 0.02]), 0.0)
            f.append((g, arms(left=pusher)))
        r, t = run(f)
        self.assertEqual(r.score, 0.0)
        self.assertTrue({"left_fingers_outside", "set_down_off_limits"} <= set(t.breaches))

    def test_drag_it_from_inside_across_the_table(self):
        f = hold(ON_MARK, arms(left=arm(tips_in(ON_MARK), CONTACT)), 5)
        g = ON_MARK
        for i in range(60):  # gripped from inside but slid along the table, never lifted, to the sheet
            g = glass(g.pos[0], g.pos[1] - 0.005, L.TOP + (0.001 if i < 5 else 0.0), vel=(0.0, -0.1, 0.0))
            f.append((g, arms(left=arm(tips_in(g), CONTACT))))
        r, t = run(f + hold(ON_SHEET, arms(), 20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("set_down_off_limits", t.breaches)

    def test_drop_it_into_the_box(self):
        f = procedure(release=False)
        cut = next(i for i, (g, _) in enumerate(f) if g.low[2] < L.TOP + 0.08 and abs(g.pos[1] - L.BOX_Y) < 1e-6)
        f = f[:cut]
        g = f[-1][0]
        z, v = g.low[2], 0.0
        while z > L.TOP:  # let go 3 cm over the walls: it falls in
            v -= 9.81 * DT
            z = max(L.TOP, z + v * DT)
            f.append((glass(L.BOX_X, L.BOX_Y, z, vel=(0, 0, v if z > L.TOP else 0.0)),
                      arms(right=arm(np.array([L.BOX_X, L.BOX_Y, L.TOP + 0.20]), NEAR))))
        r, t = run(f + hold(IN_BOX, arms(), 30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dropped", t.breaches)

    def test_tip_it_into_the_box(self):
        f = procedure(release=False)
        tipped = glass(L.BOX_X, L.BOX_Y, L.TOP, about("x", 90))
        r, t = run(f + hold(tipped, arms(), 20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("tipped", t.breaches)

    def test_set_it_on_a_box_wall(self):
        f = procedure(release=False)
        cut = next(i for i, (g, _) in enumerate(f) if abs(g.pos[1] - L.BOX_Y) < 1e-6)
        on_wall = glass(L.BOX_X - L.BOX_INNER_HALF, L.BOX_Y, L.BOX_TOP)
        r, t = run(f[:cut] + hold(on_wall, arms(), 20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("touched_box", t.breaches)

    def test_hold_it_over_the_box(self):
        f = procedure(release=False)
        cut = next(i for i, (g, _) in enumerate(f) if abs(g.pos[1] - L.BOX_Y) < 1e-6)
        g, ar = f[cut]
        r, _ = run(f[:cut + 1] + hold(g, ar, 40))
        self.assertEqual(r.score, 0.25)


class ReviewExploits(unittest.TestCase):
    """The exploits an adversarial review found against an earlier version of the finger rule, and a randomised check
    of the rule against the real finger blocks at their true turn."""

    def side_grip_trial(self, tilt_down_deg):
        """Each arm grips the glass's OUTSIDE from the side, its gripper (nearly) horizontal and pointing +x, the
        fingers closing along y onto the two outer +-y walls (74 mm at contact), its axis 1 mm over the rim, so the
        40 mm-wide fingers, now vertical, squeeze the top of the outer walls; carried to the sheet, then into the box."""
        th = math.radians(tilt_down_deg)
        a = np.array([math.cos(th), 0.0, -math.sin(th)])
        ext = 0.074 / L.FULL_GAP
        off = np.array([0.010, 0.0, HH + 0.001])
        t = Tracker(HZ)

        def carry(g0, g1, n, side):
            for g, _ in path(g0, g1, n):
                t.update(g, arms(**{side: approach(g.pos + off, a, ext)}))
            return g1

        for _ in range(10):
            t.update(ON_MARK, arms(left=approach(ON_MARK.pos + off - [0.05, 0, 0], a, OPEN)))
        for _ in range(20):
            t.update(ON_MARK, arms(left=approach(ON_MARK.pos + off, a, ext)))
        g = carry(ON_MARK, moved(ON_MARK, dz=0.075), 30, "left")
        g = carry(g, glass(L.SHEET_X, L.SHEET_Y, L.SHEET_TOP + 0.015), 50, "left")
        g = carry(g, ON_SHEET, 30, "left")
        for _ in range(20):
            t.update(ON_SHEET, arms(left=approach(ON_SHEET.pos + off - [0.08, 0, 0], a, OPEN)))
        for _ in range(20):
            t.update(ON_SHEET, arms(right=approach(ON_SHEET.pos + off, a, ext)))
        g = carry(ON_SHEET, moved(ON_SHEET, dz=0.085), 30, "right")
        g = carry(g, glass(L.BOX_X, L.BOX_Y, L.TOP + 0.085), 50, "right")
        g = carry(g, IN_BOX, 40, "right")
        r = None
        for _ in range(30):
            r = t.update(IN_BOX, arms(right=approach(IN_BOX.pos + off + [0, 0, 0.03], a, OPEN)))
        return r, t

    def test_side_grip_on_the_outside_under_the_rim(self):
        for tilt in (0.0, 20.0, 45.0):
            r, t = self.side_grip_trial(tilt)
            self.assertEqual(r.score, 0.0, tilt)
            self.assertIn("left_fingers_outside", t.breaches)
            self.assertIn("right_fingers_outside", t.breaches)

    def test_open_finger_back_on_the_outer_wall(self):
        # fully (or 80 / 70 mm) open, straight down beside the glass, true closing axis along x: the near finger's
        # outer face on the +x outer wall 20 mm below the rim
        for gap in (0.0939, 0.080, 0.070):
            x = 0.037 + gap / 2 + 0.010
            a = arm(ON_MARK.pos + np.array([x, 0.0, HH - 0.020]), gap / L.FULL_GAP)
            self.assertEqual(S.finger_check(a, ON_MARK)["status"], "outside", gap)
            self.assertTrue(touches_outside(real_finger_points(a, 0.0, ON_MARK)))

    def test_push_it_on_its_mark_with_an_open_finger(self):
        t = Tracker(HZ)
        for _ in range(5):
            t.update(ON_MARK, arms())
        xoff = 0.037 + 0.0939 / 2 + 0.010
        r = None
        for i in range(30):  # pushed 15 mm along -x, staying on its mark (no set-down rule broken)
            g = glass(L.START_X - 0.015 * i / 29, L.START_Y, L.MARK_TOP, vel=(-0.01, 0, 0))
            r = t.update(g, arms(left=arm(g.pos + np.array([xoff, 0.0, HH - 0.020]), OPEN)))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(sorted(t.breaches), ["left_fingers_outside"])

    def test_tilted_open_gripper_just_over_the_rim(self):
        # tipped 20-40 deg along its closing axis, fully open, tip point 2 mm over the rim: wherever its low finger
        # really reaches the outer wall or within 10 mm of it, the rule is broken (it was once 'above')
        hits = 0
        for tilt in (20, 30, 40):
            for dx in np.arange(-0.030, 0.0301, 0.002):
                a = approach(ON_MARK.pos + np.array([dx, 0.0, HH + 0.002]), tilted(tilt), OPEN)
                if touches_outside(real_finger_points(a, 0.0, ON_MARK)):
                    hits += 1
                    self.assertEqual(S.finger_check(a, ON_MARK)["status"], "outside", (tilt, dx))
        self.assertGreater(hits, 20)

    def test_random_poses_against_the_real_fingers(self):
        """Grippers anywhere near the glass, tipped 0-90 deg, any opening, any TRUE turn: whenever a real finger is
        on the glass's outside (or within 10 mm of it) below the rim, the rule is broken."""
        rng = np.random.default_rng(11)
        g = ON_MARK
        flagged = 0
        for _ in range(2500):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(0.0, 0.11)
            tips = g.pos + np.array([rad * math.cos(ang), rad * math.sin(ang), rng.uniform(-HH - 0.03, HH + 0.05)])
            a = approach(tips, tilted(rng.uniform(0, 90), (math.cos(ang + rng.uniform(-2, 2)),
                                                            math.sin(ang + rng.uniform(-2, 2)))),
                         rng.uniform(0, 1))
            turn = rng.uniform(0, math.pi)
            pts = real_finger_points(a, turn, g, n=(2, 5, 7))
            if not touches_outside(pts) or impossible(real_finger_points(a, turn, g, n=(2, 5, 12), reach=0.070,
                                                                          per_finger=True)):
                continue
            flagged += 1
            self.assertEqual(S.finger_check(a, g)["status"], "outside", (tips - g.pos, S.axis_of(a), a.opening))
        self.assertGreater(flagged, 250)

    def test_stale_reading_while_carrying(self):
        """rigstate's own snapshot is taken before the physics step, one control step older than the glass; task.py
        reads the arms again after the step, and if it cannot, carries a gripper inside the glass one step on with
        it. At the demo's peak carry speed (0.23 m/s: 11.4 mm per step) and the glass held 6 mm off the gripper's
        axis across the closing axis, the stale reading would break rule (1); carried on, it does not."""
        task = _task()
        v = np.array([0.0, -0.228, 0.0])
        g = glass(L.SHEET_X + 0.006, 0.10, L.TOP + 0.075, vel=v)  # 6 mm off across the closing axis (11 mm room)
        tips_now = np.array([L.SHEET_X, 0.10, g.high[2] - DEPTH])
        stale = arms(left=arm(tips_now - v * DT, CONTACT))
        self.assertEqual(S.finger_check(stale["left"], g)["status"], "outside")
        moved_on = task._carried_along(stale, g)
        self.assertTrue(np.allclose(moved_on["left"].tips, tips_now))
        self.assertEqual(S.finger_check(moved_on["left"], g)["status"], "inside")
        self.assertIs(moved_on["right"], stale["right"])  # a gripper not in the glass is left as read

    def test_arms_are_read_after_the_step_when_they_can_be(self):
        from unittest import mock
        task = _task()
        fresh = arms(left=arm(tips_in(ON_SHEET), CONTACT))
        rigstate.offline_reset()
        rigstate.offline_feed({s: {"tcp": READY[s], "tips": READY[s] - [0, 0, TIPS], "opening": 1.0} for s in ARMS_})
        with mock.patch.object(task.mk, "env", return_value=object()), \
                mock.patch.object(task.mk, "step", return_value=0), \
                mock.patch.object(task.rigstate, "_snapshot", return_value=fresh):
            got, how = task._sim_arms(ON_SHEET)
        self.assertIs(got, fresh)
        self.assertEqual(how, "after_step")
        with mock.patch.object(task.mk, "env", return_value=object()), \
                mock.patch.object(task.mk, "step", return_value=0), \
                mock.patch.object(task.rigstate, "_snapshot", side_effect=RuntimeError("no scene")), \
                mock.patch("sys.stderr", new_callable=__import__("io").StringIO) as err:
            task._WARNED.discard("fresh")
            got, how = task._sim_arms(ON_SHEET)
        self.assertEqual(how, "before_step")
        self.assertIn("could not read the arms after the step", err.getvalue())

    def test_no_arm_reading_at_all_fails_closed(self):
        from unittest import mock
        task = _task()
        rigstate.offline_reset()  # rigstate has no snapshot, mechkit no env
        sim = task.mk.state_from_poses(task.GLASS, (L.BOX_X, L.BOX_Y, L.TOP, 0, 0, 0, 1.0), source="sim")
        with mock.patch("sys.stderr", new_callable=__import__("io").StringIO) as err:
            task._WARNED.discard("none")
            for _ in range(30):
                task.success({"glass": sim})
        self.assertIn("the arms cannot be read", err.getvalue())
        self.assertEqual(task._TRACKER.arms_unavailable, 30)
        self.assertEqual(task.score({"glass": sim}), 0.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_record_alone(self):
        _, t = run(procedure()[:80])
        steps, prev, lifted = t.steps, t.prev, t.lifted
        bad = [
            SimpleNamespace(**{**vars(ON_SHEET), "pos": np.array([math.nan, 0.0, 0.8])}),
            SimpleNamespace(**{**vars(ON_SHEET), "lin_vel": np.array([math.inf, 0.0, 0.0])}),
            SimpleNamespace(**{**vars(ON_SHEET), "quat_xyzw": np.zeros(4)}),
            SimpleNamespace(pos=ON_SHEET.pos),  # missing attributes
            None,
        ]
        for b in bad:
            r = t.update(b, arms())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
        self.assertEqual(t.bad_steps, len(bad))
        self.assertEqual(t.steps, steps + len(bad))
        self.assertIs(t.prev, prev)
        self.assertEqual(t.lifted, lifted)
        self.assertEqual(t.breaches, {})

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker(HZ)
        for _ in range(400):
            q = rng.normal(size=4)
            g = glass(*rng.uniform([-0.3, -0.8, 0.0], [1.0, 0.8, 1.2]), quat=q / np.linalg.norm(q),
                      vel=rng.normal(scale=0.3, size=3))
            ar = arms(left=arm(rng.uniform([-0.2, -0.7, 0.6], [1.0, 0.7, 1.2]), rng.uniform(0, 1),
                               tilt=("x", rng.uniform(-90, 90))),
                      right=arm(g.pos + rng.normal(scale=0.03, size=3), rng.uniform(0, 1)))
            r = t.update(g, ar)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_arms_unavailable(self):
        """Fail closed: unreadable arms on more than 1 % of the steps score 0 (the finger rule could not be judged);
        a stray unreadable step is not judged, and 'let go' is taken as met on it."""
        f = procedure()
        r, t = run([(g, None) for g, _ in f])
        self.assertEqual(t.arms_unavailable, len(f))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["arms_blind"])
        n = len(f)
        few = [(g, None) if i in (100, 200) else (g, ar) for i, (g, ar) in enumerate(f)]  # 2 of n > 200 steps
        self.assertLess(2, 0.01 * n)
        r, t = run(few)
        self.assertEqual((t.arms_unavailable, r.score), (2, 1.0))
        many = [(g, None) if i % 50 == 7 else (g, ar) for i, (g, ar) in enumerate(f)]  # 2 % of the steps
        r, t = run(many)
        self.assertEqual(r.score, 0.0)
        bad = arms(left=arm([math.nan, 0, 0]))
        r, t = run([(g, bad) for g, _ in f[:5]])
        self.assertEqual(t.arms_unavailable, 5)

    def test_robust_to_a_few_mm_of_sag_and_offset(self):
        f = procedure()
        g, ar = f[-1]
        for dx, dy in ((0.005, 0.0), (0.0, -0.008), (-0.006, 0.006)):
            r, _ = run(f[:-1] + [(moved(g, dx=dx, dy=dy), ar)] * 25)
            self.assertEqual(r.score, 1.0, (dx, dy))
        # the gripper 8 mm off the glass's axis while gripping it: still inside
        self.assertEqual(S.finger_check(arm(tips_in(ON_SHEET, dx=0.008), CONTACT), ON_SHEET)["status"], "inside")


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, t = run(procedure())
        self.assertTrue(t.examined)
        t.update(ON_MARK, arms())
        self.assertFalse(t.examined)
        self.assertEqual(t.steps, 1)

    def test_jittered_teleport_resets(self):
        _, t = run(procedure()[:150])
        jittered = glass(L.START_X + 0.02, L.START_Y - 0.025, L.MARK_TOP)
        t.update(jittered, arms())
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        up = moved(ON_MARK, dz=0.03)
        f = path(ON_MARK, up, 20, "left") + path(up, ON_MARK, 20, "left")
        _, t = run(f)
        self.assertTrue(t.lifted)
        self.assertEqual(t.steps, 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = run(procedure()[:150])
        n = t.steps
        t.update(glass(0.5, 0.2, L.TOP + 0.1), arms())
        self.assertEqual(t.steps, n + 1)


ARMS_ = ("left", "right")


def _task():
    stub = _stub_dir()
    if stub is None:
        raise unittest.SkipTest("needs the offline yam_arena stub (tools/offline/yam_stub)")
    if str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    return importlib.import_module("task")


def _stub_dir():
    for p in HERE.parents:
        stub = p / "tools" / "offline" / "yam_stub"
        if stub.is_dir():
            return stub
    return None


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


class Model(unittest.TestCase):
    """offline_model: the glass follows a gripper that went in nearly closed and opened against the walls."""

    def run_model(self, lefts, start=ON_MARK):
        """Feed the model one left arm per step (the right one at its ready pose)."""
        m = M.GlassModel(HZ)
        outs = [m.step(start, arms(left=a)) for a in lefts]
        return m, outs

    def test_internal_grip_carries_and_releases(self):
        above = arm(tips_in(ON_MARK, depth=-0.01), NEAR)
        inside = [arm(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(0.0, DEPTH, 10)]
        opening = [arm(tips_in(ON_MARK), o) for o in np.linspace(NEAR, GRIP, 10)]
        lift = [arm(tips_in(ON_MARK) + [0, 0, dz], GRIP) for dz in np.linspace(0, 0.05, 10)]
        m, outs = self.run_model([above, *inside, *opening, *lift])
        self.assertEqual(m.holder, "left")
        g, seen = outs[-1]
        self.assertAlmostEqual(S.lowest(g), L.MARK_TOP + 0.05, places=6)
        self.assertAlmostEqual(seen["left"].opening, CONTACT)  # the walls stop the fingers
        self.assertEqual(S.finger_check(seen["left"], g)["status"], "inside")
        # back down onto the mark, fingers closed: it stays there
        down = [arm(tips_in(ON_MARK) + [0, 0, dz], GRIP) for dz in np.linspace(0.05, 0.0, 10)]
        release = [arm(tips_in(ON_MARK), o) for o in np.linspace(GRIP, NEAR, 10)]
        out = [arm(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(DEPTH, -0.03, 10)]
        for a in down + release + out:
            g, seen = m.step(ON_MARK, arms(left=a))
        self.assertIsNone(m.holder)
        self.assertAlmostEqual(S.lowest(g), L.MARK_TOP, places=6)
        self.assertTrue(np.allclose(g.lin_vel, 0.0))

    def run_left(self, tips_list, start=ON_MARK):
        m = M.GlassModel(HZ)
        out = None
        for tp, op in tips_list:
            out = m.step(start, arms(left=arm(tp, op)))
        return m, out

    def test_an_open_gripper_over_the_glass_moves_nothing(self):
        seq = [(tips_in(ON_MARK, depth=d), OPEN) for d in np.linspace(-0.02, 0.03, 10)]
        seq += [(tips_in(ON_MARK, depth=0.03), GRIP)] * 5 + [(tips_in(ON_MARK, depth=0.03) + [0, 0, 0.05], GRIP)] * 5
        m, (g, _) = self.run_left(seq)
        self.assertIsNone(m.holder)
        self.assertAlmostEqual(S.lowest(g), L.MARK_TOP)

    def test_too_shallow_does_not_grip(self):
        seq = [(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(-0.01, 0.020, 8)]
        seq += [(tips_in(ON_MARK, depth=0.020), o) for o in np.linspace(NEAR, GRIP, 8)]
        seq += [(tips_in(ON_MARK, depth=0.020) + [0, 0, 0.04], GRIP)] * 5
        m, (g, _) = self.run_left(seq)
        self.assertIsNone(m.holder)
        self.assertAlmostEqual(S.lowest(g), L.MARK_TOP)

    def test_opening_before_going_in_does_not_grip(self):
        # fingers opened to 50 mm (outer faces 35 mm from the axis, over the 31 mm walls) or 60 mm above the rim,
        # then lowered: they would land on the rim, they do not go in, nothing is held; at 60 mm they are also on
        # the walls for the scorer (beyond its 5 mm allowance)
        for gap, status in ((L.GRIP_GAP, None), (0.060, "outside")):
            op = gap / L.FULL_GAP
            seq = [(tips_in(ON_MARK, depth=-0.01), op)] + [(tips_in(ON_MARK, depth=d), op)
                                                            for d in np.linspace(0.0, DEPTH, 8)]
            seq += [(tips_in(ON_MARK) + [0, 0, 0.04], op)] * 5
            m, (g, seen) = self.run_left(seq)
            self.assertIsNone(m.holder)
            self.assertFalse(m.entered["left"])
            self.assertAlmostEqual(S.lowest(g), L.MARK_TOP)
            if status:
                self.assertEqual(S.finger_check(arm(tips_in(ON_MARK), op), ON_MARK)["status"], status)

    def test_drop_falls_with_its_real_speed(self):
        seq = [(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(-0.01, DEPTH, 8)]
        seq += [(tips_in(ON_MARK), o) for o in np.linspace(NEAR, GRIP, 8)]
        seq += [(tips_in(ON_MARK) + [0, 0, dz], GRIP) for dz in np.linspace(0, 0.06, 10)]
        m = M.GlassModel(HZ)
        for tp, op in seq:
            m.step(ON_MARK, arms(left=arm(tp, op)))
        tp = tips_in(ON_MARK) + [0, 0, 0.06]
        speeds = []
        for _ in range(8):
            g, _ = m.step(ON_MARK, arms(left=arm(tp, NEAR)))
            speeds.append(-float(g.lin_vel[2]))
        self.assertIsNone(m.holder)
        self.assertGreater(max(speeds), L.DROP_SPEED)
        self.assertAlmostEqual(S.lowest(g), L.MARK_TOP)

    def test_lets_go_as_soon_as_the_fingers_leave_the_walls(self):
        seq = [(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(-0.01, DEPTH, 8)]
        seq += [(tips_in(ON_MARK), o) for o in np.linspace(NEAR, GRIP, 8)]
        seq += [(tips_in(ON_MARK) + [0, 0, 0.02], GRIP)] * 3
        for gap, held in ((0.0405, True), (0.0395, False)):  # contact at 42 mm; let go under 40 mm
            m, _ = self.run_left(seq + [(tips_in(ON_MARK) + [0, 0, 0.02], gap / L.FULL_GAP)])
            self.assertEqual(m.holder == "left", held, gap)

    def test_cannot_push_it_through_its_support(self):
        seq = [(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(-0.01, DEPTH, 8)]
        seq += [(tips_in(ON_MARK), o) for o in np.linspace(NEAR, GRIP, 8)]
        seq += [(tips_in(ON_MARK) - [0, 0, 0.01], GRIP)] * 3
        m, (g, _) = self.run_left(seq)
        self.assertAlmostEqual(S.lowest(g), L.MARK_TOP)
        self.assertEqual(m.holder, "left")

    def test_same_step_is_computed_once(self):
        m = M.GlassModel(HZ)
        ar = arms()
        a = m.step(ON_MARK, ar)
        b = m.step(ON_MARK, ar)
        self.assertIs(a, b)
        self.assertEqual(m.steps, 1)

    def test_checker_carrying_it_wins(self):
        m = M.GlassModel(HZ)
        m.step(ON_MARK, arms())
        carried = moved(ON_MARK, dz=0.03)
        g, _ = m.step(carried, arms())
        self.assertIs(g, carried)

    def test_support_heights(self):
        self.assertAlmostEqual(M.GlassModel.support(ON_MARK.pos), L.MARK_TOP)
        self.assertAlmostEqual(M.GlassModel.support(ON_SHEET.pos), L.SHEET_TOP)
        self.assertAlmostEqual(M.GlassModel.support(IN_BOX.pos), L.TOP)
        self.assertAlmostEqual(M.GlassModel.support((L.BOX_X + L.BOX_INNER_HALF, L.BOX_Y, 0.9)), L.BOX_TOP)
        self.assertAlmostEqual(M.GlassModel.support((0.3, 0.9, 0.9)), 0.0)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hooks, demo and text."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def state(self, source="sim", origin=None, quat=(0.0, 0.0, 0.0, 1.0)):
        spec = self.task.GLASS
        pose = (*(spec.pos if origin is None else origin), *quat)
        return self.mk.state_from_poses(spec, pose, source=source)

    def feed(self, left=None, right=None):
        snap = {}
        for side, a in (("left", left or ready("left")), ("right", right or ready("right"))):
            snap[side] = {"tcp": a.tcp, "tips": a.tips, "opening": a.opening}
        rigstate.offline_feed(snap)

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(names, ["start_mark", "exam_sheet", "box_front", "box_back", "box_left", "box_right",
                                 "glass"])
        g = self.task.GLASS
        self.assertEqual(g.kind, "rigid")
        self.assertFalse(g.fixed_base)
        self.assertAlmostEqual(g.mass, L.GLASS_MASS)
        lo, hi = g.world_bounds()
        self.assertTrue(np.allclose(lo, [L.START_X - L.GLASS_HALF, L.START_Y - L.GLASS_HALF, L.MARK_TOP]))
        self.assertTrue(np.allclose(hi, [L.START_X + L.GLASS_HALF, L.START_Y + L.GLASS_HALF, L.MARK_TOP + L.GLASS_H]))
        # its state's centre is its geometric centre (what the scorer assumes)
        st = self.state()
        self.assertTrue(np.allclose(st.pos, ON_MARK.pos))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertGreaterEqual(self.task.SUCCESS_HOLD_S, 1.0)
        for w in self.task.BOX_WALLS:
            self.assertIsNone(w.mass)
            self.assertAlmostEqual(w.size[2], L.BOX_HEIGHT)

    def test_checker_state_is_marked_offline(self):
        """The offline checker's state for the glass is mechkit's offline_state: source 'offline'."""
        spec = self.task.GLASS
        centre, size, yaw = spec.offline_box()
        plain = SimpleNamespace(pos=np.array(centre), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                                ang_vel=np.zeros(3), low=np.array(centre) - np.array(size) / 2,
                                high=np.array(centre) + np.array(size) / 2)
        st = spec.offline_state(plain)
        self.assertEqual(st.source, "offline")
        self.assertTrue(np.allclose(st.origin, [L.START_X, L.START_Y, L.MARK_TOP]))

    def test_simulator_states_are_scored_as_they_are(self):
        rigstate.offline_reset()
        self.feed()
        far = self.state(origin=(L.BOX_X, L.BOX_Y, L.TOP))
        self.task.success({"glass": far})
        self.assertTrue(self.task._TRACKER.last.details["in_box"])  # the state itself, not the model
        self.assertEqual(self.task._MODEL.steps, 0)

    def test_offline_states_use_the_model(self):
        rigstate.offline_reset()
        start = self.state(source="offline")
        lefts = [arm(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(-0.01, DEPTH, 10)]
        lefts += [arm(tips_in(ON_MARK), o) for o in np.linspace(NEAR, GRIP, 10)]
        lefts += [arm(tips_in(ON_MARK) + [0, 0, dz], GRIP) for dz in np.linspace(0, 0.04, 10)]
        for a in lefts:
            self.feed(left=a)
            self.task.success({"glass": start})
            self.task.score({"glass": start})  # same step: the model does not advance twice
        self.assertEqual(self.task._MODEL.steps, len(lefts))
        self.assertEqual(self.task._MODEL.holder, "left")
        self.assertTrue(self.task._TRACKER.lifted)
        self.assertEqual(self.task._TRACKER.breaches, {})

    def test_offline_release_in_the_air_is_a_drop(self):
        rigstate.offline_reset()
        start = self.state(source="offline")
        lefts = [arm(tips_in(ON_MARK, depth=d), NEAR) for d in np.linspace(-0.01, DEPTH, 10)]
        lefts += [arm(tips_in(ON_MARK), o) for o in np.linspace(NEAR, GRIP, 10)]
        lefts += [arm(tips_in(ON_MARK) + [0, 0, dz], GRIP) for dz in np.linspace(0, 0.06, 10)]
        top = tips_in(ON_MARK) + [0, 0, 0.06]
        lefts += [arm(top, o) for o in np.linspace(GRIP, NEAR, 10)] + [arm(top, NEAR)] * 10
        for a in lefts:
            self.feed(left=a)
            self.task.success({"glass": start})
        self.assertIsNone(self.task._MODEL.holder)
        self.assertIn("dropped", self.task._TRACKER.breaches)
        self.assertEqual(self.task.score({"glass": start}), 0.0)

    def test_broken_glass_state_scores_zero(self):
        rigstate.offline_reset()
        self.feed()
        bad = self.state(origin=(math.nan, 0.0, 0.8))
        self.assertFalse(self.task.success({"glass": bad}))
        self.assertEqual(self.task._TRACKER.bad_steps, 1)
        self.assertEqual(self.task.score({"glass": bad}), 0.0)
        self.assertFalse(self.task.success({}))  # missing glass
        self.assertEqual(self.task._TRACKER.bad_steps, 2)

    def test_reset_hooks(self):
        tr, model = self.task._TRACKER, self.task._MODEL
        rigstate.offline_reset()
        self.feed()
        self.task.success({"glass": self.state(source="offline")})
        self.assertEqual(tr.steps, 1)
        self.assertEqual(model.steps, 1)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker and model start over
        self.assertEqual((tr.steps, model.steps), (0, 0))
        self.feed()
        self.task.success({"glass": self.state()})
        self.task.success({"glass": self.state()})
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's reset counter: the second, independent signal
        self.task.success({"glass": self.state()})
        self.assertEqual(tr.steps, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed()
            self.assertFalse(self.task.success({"glass": self.state()}))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

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

    def test_demo_timing_and_grip_geometry(self):
        task = self.task
        plan = task.demo()
        for side, wps in plan.items():
            prev_f, t = None, 0.0
            for i, w in enumerate(wps):
                t += w.duration
                if prev_f is not None and abs(w.finger - prev_f) > 0.01:  # an open or a close
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))
                    if i > 1:  # a settle (the same pose held) before every grip and release
                        self.assertEqual(wps[i - 1].pos, w.pos, (side, w.label))
                        self.assertGreaterEqual(wps[i - 1].duration, 0.5, (side, w.label))
                prev_f = w.finger
        # tips 42 mm below the rim when gripping: the measured lowest 38 mm of the fingers (their 10 mm-thick band)
        # inside with 4 mm to spare, carriages 28 mm and the gripper's body 41 mm over the rim, tips 32 mm over the
        # inner floor; and deep enough for offline_model to count the grip
        self.assertAlmostEqual(task.TIP_DEPTH, 0.042)
        self.assertGreaterEqual(task.TIP_DEPTH, L.FINGER_LEN + 0.004)
        self.assertGreaterEqual(0.070 - task.TIP_DEPTH, 0.025)
        self.assertGreaterEqual(0.083 - task.TIP_DEPTH, 0.040)
        self.assertGreaterEqual(L.GLASS_H - L.GLASS_FLOOR - task.TIP_DEPTH, 0.030)
        self.assertGreaterEqual(task.TIP_DEPTH, M.MIN_DEPTH + 0.010)
        self.assertAlmostEqual(task.HOLD + task.TIP_DEPTH - task.PAD_TIPS, L.GLASS_H)
        # carried over the box's walls with 30 mm to spare
        self.assertAlmostEqual(task.CARRY_R - task.HOLD - L.BOX_TOP, 0.030)
        # the right arm sets off for the sheet only once the left arm is out of the glass and 14 cm up
        def ends(wps):
            out, t = {}, 0.0
            for w in wps:
                t += w.duration
                out.setdefault(w.label, t)
            return out
        le, re_ = ends(plan["left"]), ends(plan["right"])
        start_r = re_["over the glass as the left arm moves away"] - 1.5
        self.assertGreaterEqual(start_r, le["straight up, out of the glass"])

    def test_instruction_is_the_same_everywhere(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{self.task.INSTRUCTION}"'), 2)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)


_EXPLOIT_RUNNER = r'''
import json, math, sys
sys.path.insert(0, sys.argv[1])
import check_task as C, scenegeom as G, yamkin as K
from pathlib import Path
task_dir = Path(sys.argv[2]); name = sys.argv[3]
task = C.load_task(task_dir / "task.py", task_dir)
from yam_arena.demo import Waypoint
import layout as L
top = K.TABLE_TOP["station"]
x, y, b = L.START_X, L.START_Y, L.MARK_TOP
HOVER = top + 0.14
if name == "outside_grip":  # an ordinary top-down grip across the glass's outside, carried to the sheet
    w = -(L.GLASS_OUTER - 0.008) / 2
    plan = {"left": [Waypoint(2.0, (x, y, HOVER), -0.04695, "over"), Waypoint(1.5, (x, y, b + 0.045), -0.04695, "down"),
                     Waypoint(0.6, (x, y, b + 0.045), -0.04695, "settle"), Waypoint(1.5, (x, y, b + 0.045), w, "grip"),
                     Waypoint(1.5, (x, y, top + 0.12), w, "lift"),
                     Waypoint(2.2, (L.SHEET_X, L.SHEET_Y, top + 0.12), w, "carry"),
                     Waypoint(1.5, (L.SHEET_X, L.SHEET_Y, L.SHEET_TOP + 0.047), w, "down"),
                     Waypoint(1.5, (L.SHEET_X, L.SHEET_Y, L.SHEET_TOP + 0.047), -0.04695, "open"),
                     Waypoint(1.5, (L.SHEET_X, L.SHEET_Y, HOVER), -0.04695, "up")]}
elif name == "open_into_it":  # fingers opened wide before going in: they straddle the walls, nothing is held
    plan = {"left": [Waypoint(2.0, (x, y, HOVER), -0.025, "over, 50 mm open"),
                     Waypoint(2.0, (x, y, b + 0.045), -0.025, "down onto it"),
                     Waypoint(1.5, (x, y, top + 0.12), -0.025, "lift")]}
else:
    raise SystemExit("unknown exploit")
solids = [G.solid_from_spec(o, top) for o in task.objects]
out = C.Playback(task, plan, solids, hz=20.0, hold_s=3.0).run()["proxy"]
print(json.dumps({"score": out["predicted_task_score"], "events": [e["event"] for e in out["events"]],
                  "moved": sorted(out["moved_objects"]), "final": out["final_hooks"]}))
'''


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo, and two exploits, played through our offline checker's kinematics (the glass moved by
    offline_model.py)."""

    def env(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("TASK_TRACE", None)
        env.pop("DEMO_IDLE_ARM", None)
        env.pop("DEMO_HOLD_ONLY", None)
        return env

    def test_demo_predicts_one_and_each_arm_alone_less(self):
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--one-arm", "--json", "-"],
                              capture_output=True, text=True, env=self.env(), timeout=1800)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        rep = json.loads(proc.stdout)
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 800)
        # the checker's own grasp proxy never moved the glass: only offline_model did
        self.assertEqual(proxy["moved_objects"], {})
        self.assertFalse([e for e in proxy["events"] if e["event"] in ("grasp", "release", "drop")])
        self.assertEqual(rep["one_arm"], {"left_only": 0.5, "right_only": 0.0})
        for side in ("left", "right"):
            self.assertLess(rep["playback"]["arms"][side]["worst_ik_mm"], 0.5)
        self.assertGreater(rep["playback"]["min_tcp_distance_m"], 0.20)

    def exploit(self, name):
        proc = subprocess.run([sys.executable, "-c", _EXPLOIT_RUNNER, str(_checker().parent), str(HERE), name],
                              capture_output=True, text=True, env=self.env(), timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_outside_grip_through_the_checker(self):
        out = self.exploit("outside_grip")
        self.assertIn("grasp", out["events"])  # the checker's own proxy carries the glass by its outside ...
        self.assertIn("glass", out["moved"])
        self.assertEqual(out["score"], 0.0)  # ... and the finger rule scores it 0

    def test_open_fingers_onto_it_through_the_checker(self):
        out = self.exploit("open_into_it")
        self.assertEqual(out["score"], 0.0)
        self.assertEqual(out["moved"], [])


if __name__ == "__main__":
    unittest.main()
