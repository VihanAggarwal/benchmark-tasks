"""Offline tests for scoring.py and task.py: synthetic bar states and gripper snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    Tracker,
    along,
    arms_valid,
    decide,
    ends,
    flat_err_deg,
    heading_change_deg,
    heading_deg,
    hands_off,
    lowest,
    point_distance,
    valid,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
SETTLE = 2 * WINDOW + 2  # steps that fill both stillness windows
TIERS = (0.0, 0.25, 0.5, 0.75, 1.0)


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def bar(name, x, y, yaw_deg, lift=0.0, pitch=0.0, roll=0.0, vel=(0.0, 0.0, 0.0)):
    """A bar's state centred at (x, y), its length turned yaw_deg from +x, lying flat on the floor (``lift`` m up
    from there), then pitched / rolled about its centre."""
    q = quat(yaw_deg, pitch, roll)
    p = np.array([x, y, L.TOP + L.BAR_W / 2 + lift])
    half = np.array([L.BAR_LEN[name] / 2, L.BAR_W / 2, L.BAR_W / 2])
    return SimpleNamespace(pos=p, quat_xyzw=q, lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3),
                           low=p - half.max(), high=p + half.max())


def start(name):
    x, y, yaw = L.BAR_START[name]
    return bar(name, x, y, yaw)


# Side-wall bar: along x, centre y -0.299 (6 mm off the kerb); compliant for the 48 in bar with cx 0.311..0.335.
def side(name="bar_48", cx=0.323, cy=-0.299, yaw=0.0, **kw):
    return bar(name, cx, cy, yaw, **kw)


# Rear-wall bar: along y, centre x 0.439 (6 mm off the kerb); compliant for the 40 in bar with cy -0.231..-0.215.
def rear(name="bar_40", cx=0.439, cy=-0.223, yaw=90.0, **kw):
    return bar(name, cx, cy, yaw, **kw)


DEMO = [side(), rear()]
CORNER_40 = rear(cy=L.SIDE_FACE_Y + L.BAR_LEN["bar_40"] / 2)  # pushed into the corner: -0.235


def arm(tips, opening=1.0, up=0.013):
    t = np.asarray(tips, dtype=float)
    return SimpleNamespace(tcp=t + np.array([0.0, 0.0, up]), tips=t, opening=opening)


AWAY = {"left": arm((0.20, 0.40, L.TOP + 0.20)), "right": arm((0.20, -0.40, L.TOP + 0.15))}


def holding(s, opening=L.BAR_W / L.FULL_GAP):
    """The right gripper closed on a bar at its middle, pad tips 8 mm above its underside."""
    tips = np.array([s.pos[0], s.pos[1], s.pos[2] - L.BAR_W / 2 + 0.008])
    return {"left": AWAY["left"], "right": arm(tips, opening)}


def feed(bars, arms=AWAY, n=SETTLE, tr=None):
    tr = tr or Tracker(HZ)
    res = None
    for _ in range(n):
        res = tr.update(bars, arms)
    return tr, res


def final(bars, arms=AWAY):
    return feed(bars, arms)[1]


class Geometry(unittest.TestCase):
    """The plan's numbers, the windows and the demo's aims."""

    def test_rule_numbers(self):
        self.assertAlmostEqual(L.CL_Y, -0.247)
        self.assertAlmostEqual(L.CL_Y - L.SIDE_FACE_Y, 17 * L.IN)
        self.assertAlmostEqual(L.REAR_FACE_X - L.SIDE_REAR_MAX, 0.407)
        self.assertAlmostEqual(L.REAR_FACE_X - L.SIDE_EXTENT_MIN, 0.239)
        self.assertAlmostEqual(L.CL_Y - L.REAR_WALL_SIDE_MIN, -0.295)
        self.assertAlmostEqual(L.CL_Y + L.REAR_OPEN_SIDE_MIN, -0.151)
        self.assertAlmostEqual(L.BAR_LEN["bar_48"], 0.192)
        self.assertAlmostEqual(L.BAR_LEN["bar_40"], 0.160)
        self.assertLess(L.BAR_LEN["bar_40"], L.SIDE_LEN_MIN)  # the 40 in bar can never be the side bar
        self.assertGreaterEqual(L.BAR_LEN["bar_48"], L.SIDE_LEN_MIN)
        self.assertGreaterEqual(L.BAR_LEN["bar_40"], L.REAR_LEN_MIN)
        self.assertAlmostEqual(L.TANK[1] - L.BOWL_X + L.BOWL_R, 28 * L.IN)  # the toilet is 28 in deep
        self.assertAlmostEqual(L.TANK[3] - L.TANK[2], 18 * L.IN)

    def test_centred_rear_bar_fails_the_open_side(self):
        """The trap: the 40 in bar centred on the toilet is stopped by the side kerb and reaches only 23 in."""
        centred_cy = max(L.CL_Y, L.SIDE_FACE_Y + L.BAR_LEN["bar_40"] / 2)
        self.assertAlmostEqual(centred_cy, -0.235)
        open_past = centred_cy + L.BAR_LEN["bar_40"] / 2 - L.CL_Y
        self.assertAlmostEqual(open_past / L.IN, 23.0)
        self.assertLess(open_past, L.REAR_OPEN_SIDE_MIN)

    def test_targets_mid_window(self):
        tx, ty = L.TARGET["bar_48"]
        self.assertAlmostEqual(tx, 0.321)
        self.assertAlmostEqual(ty, -0.297)
        tx, ty = L.TARGET["bar_40"]
        self.assertAlmostEqual(tx, 0.437)
        self.assertAlmostEqual(ty, -0.223)
        for s in DEMO:
            d = along(s, "bar_48" if s is DEMO[0] else "bar_40")
            self.assertTrue(d["compliant"], d)

    def test_windows(self):
        """Each bar's compliant window for its centre, edge to edge, and the margins round the demo's aims."""
        for cx, ok in ((0.311, True), (0.3105, False), (0.335, True), (0.3355, False)):
            self.assertEqual(along(side(cx=cx), "bar_48")["compliant"], ok, cx)
        for cy, ok in ((-0.306, True), (-0.3065, False), (-0.290, True), (-0.2895, False)):
            self.assertEqual(along(side(cy=cy), "bar_48")["wall"] == "side", ok, cy)
        for cy, ok in ((-0.231, True), (-0.2315, False), (-0.215, True), (-0.2145, False)):
            self.assertEqual(along(rear(cy=cy), "bar_40")["compliant"], ok, cy)
        for cx, ok in ((0.430, True), (0.4295, False), (0.446, True), (0.4465, False)):
            self.assertEqual(along(rear(cx=cx), "bar_40")["wall"] == "rear", ok, cx)

    def test_corner_clearance(self):
        """The side bar's rear end and the rear bar's front face stay >= 6 mm apart whether each bar lands on its
        trimmed aim or 4 mm short of it."""
        task = load_task()
        for b48 in (task.place_point("bar_48"), L.TARGET["bar_48"]):
            for b40 in (task.place_point("bar_40"), L.TARGET["bar_40"]):
                gap = (b40[0] - L.BAR_W / 2) - (b48[0] + L.BAR_LEN["bar_48"] / 2)
                self.assertGreaterEqual(gap, 0.006, (b48, b40))
                s, r = side(cx=b48[0], cy=b48[1]), rear(cx=b40[0], cy=b40[1])
                self.assertTrue(along(s, "bar_48")["compliant"])
                self.assertTrue(along(r, "bar_40")["compliant"])

    def test_kerb_clearance(self):
        """Laid on its trimmed aim (no shortfall) at the commanded heading (the -1.15 degree bias, with none of the
        GPU's counter-clockwise turn) or with up to 1.7 degrees of that turn, no corner of either bar comes within
        2 mm of its kerb's face, so a bar cannot come down on a kerb."""
        task = load_task()
        for dyaw in (-1.15, -0.55, 0.55):
            for name, wall_yaw in (("bar_48", 0.0), ("bar_40", 90.0)):
                cx, cy = task.place_point(name)
                a = math.radians(wall_yaw + dyaw)
                ax, ay = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
                corners = [np.array([cx, cy]) + i * L.BAR_LEN[name] / 2 * ax + j * L.BAR_W / 2 * ay
                           for i in (-1, 1) for j in (-1, 1)]
                if name == "bar_48":
                    clear = min(c[1] for c in corners) - L.SIDE_FACE_Y
                else:
                    clear = L.REAR_FACE_X - max(c[0] for c in corners)
                self.assertGreaterEqual(clear, 0.002, (name, dyaw, clear))

    def test_yaw_error_tolerated(self):
        """A bar laid 1.7 degrees off moves an end by under 3 mm; every demo aim stays compliant from -2 to +2."""
        task = load_task()
        for dyaw in (-2.0, -1.15, 0.0, 1.7, 2.0):
            for aim in ("trimmed", "target"):
                p48 = task.place_point("bar_48") if aim == "trimmed" else L.TARGET["bar_48"]
                p40 = task.place_point("bar_40") if aim == "trimmed" else L.TARGET["bar_40"]
                self.assertTrue(along(side(cx=p48[0], cy=p48[1], yaw=dyaw), "bar_48")["compliant"], (dyaw, aim))
                self.assertTrue(along(rear(cx=p40[0], cy=p40[1], yaw=90 + dyaw), "bar_40")["compliant"],
                                (dyaw, aim))
        self.assertLess(96 * math.sin(math.radians(1.7)), 3.0)

    def test_start_layout(self):
        for name in L.NAMES:
            s = start(name)
            self.assertIsNone(along(s, name)["wall"], name)
        self.assertEqual(final([start(n) for n in L.NAMES]).score, 0.0)
        self.assertTrue(Tracker.at_start([start(n) for n in L.NAMES]))


class Tiers(unittest.TestCase):
    """Every tier, as the spec lists them, and the decision list's order."""

    def test_demo_layout_scores_one(self):
        tr, res = feed(DEMO)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertEqual(res.details["tier_reason"], "both_walls_compliant")

    def test_not_yet_still_for_one_second(self):
        """Counted (let go and settled for 0.5 s) but not yet still for 1 s: 0.75."""
        _, res = feed(DEMO, n=WINDOW + 1)
        self.assertEqual(res.score, 0.75)
        _, res = feed(DEMO, n=2 * WINDOW + 1)
        self.assertEqual(res.score, 1.0)
        _, res = feed(DEMO, n=WINDOW)  # not even 0.5 s seen: nothing counts
        self.assertEqual(res.score, 0.0)

    def test_rear_bar_in_the_corner(self):
        res = final([side(), CORNER_40])
        self.assertEqual(res.score, 0.75)
        self.assertEqual(res.details["bar_40"]["why_not"], "open_side_under_24_in")

    def test_rear_bar_left_where_it_started(self):
        self.assertEqual(final([side(), start("bar_40")]).score, 0.5)

    def test_rear_compliant_side_bare(self):
        self.assertEqual(final([start("bar_48"), rear()]).score, 0.5)

    def test_both_on_walls_both_wrong(self):
        res = final([side(cx=0.300), CORNER_40])  # side bar's rear end 59 mm from the rear wall
        self.assertEqual(res.score, 0.25)
        self.assertEqual(res.details["bar_48"]["why_not"], "rear_end_over_12_in")

    def test_one_bar_on_a_wall_none_compliant(self):
        self.assertEqual(final([start("bar_48"), CORNER_40]).score, 0.25)
        self.assertEqual(final([start("bar_48"), side("bar_40", cx=0.370)]).score, 0.25)

    def test_nothing_moved(self):
        self.assertEqual(final([start(n) for n in L.NAMES]).score, 0.0)

    def test_bars_swapped(self):
        """The 48 in bar on the rear wall meets the rear rule; the 40 in bar on the side wall is under 42 in."""
        res = final([rear("bar_48", cy=-0.219), side("bar_40", cx=0.370)])
        self.assertEqual(res.score, 0.75)
        self.assertTrue(res.details["bar_48"]["compliant"])
        self.assertEqual(res.details["bar_40"]["why_not"], "under_42_in")

    def test_side_bar_beside_the_bowl(self):
        """Shortcut 3: the side bar pushed back to the rear wall, its front end only 192 mm out."""
        pushed = side(cx=L.REAR_FACE_X - L.BAR_LEN["bar_48"] / 2)  # 0.359
        res = final([pushed, rear()])
        self.assertEqual(res.score, 0.75)
        self.assertEqual(res.details["bar_48"]["why_not"], "under_54_in_out")
        self.assertEqual(final([pushed, CORNER_40]).score, 0.25)

    def test_last_bar_still_held(self):
        """The full demo but the gripper still closed on the rear bar: it is on no wall."""
        self.assertEqual(final(DEMO, holding(DEMO[1])).score, 0.5)

    def test_decision_list_order(self):
        def w(sh, so, rh, ro, still):
            return {"side": {"has": sh, "ok": so}, "rear": {"has": rh, "ok": ro}, "still_1s": still}

        self.assertEqual(decide(w(True, True, True, True, True))[0], 1.0)
        self.assertEqual(decide(w(True, True, True, True, False))[0], 0.75)
        self.assertEqual(decide(w(True, True, True, False, True))[0], 0.75)
        self.assertEqual(decide(w(True, False, True, True, True))[0], 0.75)
        self.assertEqual(decide(w(True, True, False, False, True))[0], 0.5)
        self.assertEqual(decide(w(False, False, True, True, True))[0], 0.5)
        self.assertEqual(decide(w(True, False, True, False, True))[0], 0.25)
        self.assertEqual(decide(w(True, False, False, False, True))[0], 0.25)
        self.assertEqual(decide(w(False, False, False, False, True))[0], 0.0)


class OnAWall(unittest.TestCase):
    """Every condition a bar needs to count as along a wall, each limit from both sides."""

    def test_tilted(self):
        """One end propped up: tilted 4.9 degrees with its low corner on the floor it still counts, 5.5 not."""
        def propped(p):
            lift = L.BAR_LEN["bar_48"] / 2 * math.sin(math.radians(p)) + L.BAR_W / 2 * (math.cos(math.radians(p)) - 1)
            return side(pitch=p, lift=lift)

        self.assertAlmostEqual(lowest(propped(4.9), "bar_48"), L.TOP, places=9)
        self.assertEqual(along(propped(4.9), "bar_48")["wall"], "side")
        self.assertIsNone(along(side(pitch=4.9), "bar_48")["wall"])  # pitched about its centre: an end sunk 8 mm
        d = along(propped(5.5), "bar_48")
        self.assertIsNone(d["wall"])
        self.assertEqual(d["why_not"], "not_flat_on_floor")

    def test_on_an_edge(self):
        self.assertIsNone(along(side(roll=45.0), "bar_48")["wall"])
        self.assertEqual(along(side(roll=90.0), "bar_48")["wall"], "side")  # another long face down: alike

    def test_standing_on_an_end(self):
        s = side(pitch=90.0, lift=L.BAR_LEN["bar_48"] / 2 - L.BAR_W / 2)
        self.assertIsNone(along(s, "bar_48")["wall"])
        self.assertTrue(math.isnan(heading_deg(s)))

    def test_floor_height(self):
        self.assertEqual(along(side(lift=0.0029), "bar_48")["wall"], "side")
        self.assertEqual(along(side(lift=-0.0029), "bar_48")["wall"], "side")
        self.assertIsNone(along(side(lift=0.0035), "bar_48")["wall"])
        self.assertIsNone(along(side(lift=-0.0035), "bar_48")["wall"])  # sunk / fallen through
        self.assertIsNone(along(side(lift=-0.30), "bar_48")["wall"])  # fallen off the table
        self.assertIsNone(along(side(lift=L.BAR_W), "bar_48")["wall"])  # laid on the other bar

    def test_on_the_paint(self):
        """The rear bar lies across the 0.5 mm painted tank: still on the floor."""
        self.assertEqual(along(rear(lift=L.PAINT_T), "bar_40")["wall"], "rear")

    def test_on_the_kerb(self):
        """Propped on a 4 mm kerb (one side up on it): tilted and up, it counts for nothing."""
        tilt = math.degrees(math.asin(L.KERB_H / L.BAR_W))
        s = side(cy=L.SIDE_FACE_Y + 0.004, roll=-tilt, lift=L.KERB_H / 2)
        self.assertIsNone(along(s, "bar_48")["wall"])
        r = rear(cx=L.REAR_FACE_X + L.KERB_W / 2, lift=L.KERB_H)  # right on top of the rear kerb
        self.assertIsNone(along(r, "bar_40")["wall"])

    def test_parallel(self):
        self.assertTrue(along(side(yaw=4.9), "bar_48")["compliant"])
        self.assertTrue(along(side(yaw=180.0 - 4.9), "bar_48")["compliant"])
        self.assertIsNone(along(side(yaw=5.5), "bar_48")["wall"])
        self.assertEqual(along(side(yaw=5.5), "bar_48")["why_not"], "not_parallel_to_a_wall")
        self.assertTrue(along(rear(yaw=94.9), "bar_40")["compliant"])
        self.assertTrue(along(rear(yaw=-85.1), "bar_40")["compliant"])
        self.assertIsNone(along(rear(yaw=95.5), "bar_40")["wall"])
        self.assertIsNone(along(side(yaw=45.0), "bar_48")["wall"])

    def test_wall_gap(self):
        f = L.SIDE_FACE_Y + L.BAR_W / 2
        self.assertEqual(along(side(cy=f - 0.0009), "bar_48")["wall"], "side")  # pressed to the kerb
        self.assertIsNone(along(side(cy=f - 0.0015), "bar_48")["wall"])
        self.assertEqual(along(side(cy=f + 0.0149), "bar_48")["wall"], "side")
        self.assertIsNone(along(side(cy=f + 0.016), "bar_48")["wall"])
        self.assertIsNone(along(side(cy=L.SIDE_FACE_Y - L.KERB_W - L.BAR_W / 2), "bar_48")["wall"])  # outside
        g = L.REAR_FACE_X - L.BAR_W / 2
        self.assertEqual(along(rear(cx=g + 0.0009), "bar_40")["wall"], "rear")
        self.assertIsNone(along(rear(cx=g + 0.0015), "bar_40")["wall"])
        self.assertEqual(along(rear(cx=g - 0.0149), "bar_40")["wall"], "rear")
        self.assertIsNone(along(rear(cx=g - 0.016), "bar_40")["wall"])
        self.assertIsNone(along(rear(cx=L.REAR_FACE_X + L.KERB_W + L.BAR_W / 2), "bar_40")["wall"])  # behind it

    def test_beside_the_kerb_only(self):
        """A bar in line with a wall but past the kerb's end is not along that wall."""
        self.assertEqual(along(rear(cy=0.005 + L.SPAN_TOL - 0.080), "bar_40")["wall"], "rear")
        self.assertIsNone(along(rear(cy=0.005 + L.SPAN_TOL - 0.079), "bar_40")["wall"])
        self.assertIsNone(along(rear(cy=0.10), "bar_40")["wall"])
        self.assertEqual(along(side("bar_40", cx=0.205 - L.SPAN_TOL + 0.080), "bar_40")["wall"], "side")
        self.assertIsNone(along(side("bar_40", cx=0.205 - L.SPAN_TOL + 0.079), "bar_40")["wall"])
        self.assertEqual(final([side(), rear(cy=0.10)]).score, 0.5)

    def test_side_rule_limits(self):
        cx_rear = L.REAR_FACE_X - L.SIDE_REAR_MAX - L.BAR_LEN["bar_48"] / 2  # rear end exactly 12 in off
        self.assertTrue(along(side(cx=cx_rear), "bar_48")["compliant"])
        self.assertFalse(along(side(cx=cx_rear - 0.001), "bar_48")["compliant"])
        cx_front = L.REAR_FACE_X - L.SIDE_EXTENT_MIN + L.BAR_LEN["bar_48"] / 2  # front end exactly 54 in out
        self.assertTrue(along(side(cx=cx_front), "bar_48")["compliant"])
        self.assertFalse(along(side(cx=cx_front + 0.001), "bar_48")["compliant"])

    def test_rear_rule_limits(self):
        h = L.BAR_LEN["bar_40"] / 2
        cy_wall = L.CL_Y - L.REAR_WALL_SIDE_MIN + h  # wall-side end exactly 12 in past the centreline
        self.assertTrue(along(rear(cy=cy_wall), "bar_40")["compliant"])
        self.assertEqual(along(rear(cy=cy_wall + 0.001), "bar_40")["why_not"], "wall_side_under_12_in")
        cy_open = L.CL_Y + L.REAR_OPEN_SIDE_MIN - h  # open end exactly 24 in past it
        self.assertTrue(along(rear(cy=cy_open), "bar_40")["compliant"])
        self.assertEqual(along(rear(cy=cy_open - 0.001), "bar_40")["why_not"], "open_side_under_24_in")

    def test_bar_ends_either_way_round(self):
        a = along(side(yaw=0.0), "bar_48")
        b = along(side(yaw=180.0), "bar_48")
        self.assertEqual((a["wall"], a["compliant"]), (b["wall"], b["compliant"]))
        self.assertAlmostEqual(a["rear_end_from_rear_wall_mm"], b["rear_end_from_rear_wall_mm"])


class LetGoAndStill(unittest.TestCase):
    """Counted = let go for 0.5 s and settled; the 1.0 also needs both bars still for 1 s."""

    def test_held_in_place(self):
        res = final(DEMO, holding(DEMO[0]))
        self.assertEqual(res.score, 0.5)  # the rear wall alone counts
        self.assertFalse(res.details["bar_48"]["released"])

    def test_open_gripper_resting_on_it(self):
        """An open gripper pressing the bar down or propping it: within its finger zone, not let go."""
        s = DEMO[1]
        tips = np.array([s.pos[0], s.pos[1], s.pos[2] + L.BAR_W / 2 + 0.002])
        res = final(DEMO, {"left": AWAY["left"], "right": arm(tips, 1.0)})
        self.assertFalse(res.details["bar_40"]["released"])
        self.assertEqual(res.score, 0.5)

    def test_released_needs_the_whole_half_second(self):
        tr, _ = feed(DEMO, n=SETTLE)
        for _ in range(3):
            tr.update(DEMO, holding(DEMO[1]))
        for k in range(WINDOW):  # let go again: counted only once 0.5 s has passed with hands off
            res = tr.update(DEMO, AWAY)
            self.assertEqual(res.details["bar_40"]["released"], False, k)
        self.assertTrue(tr.update(DEMO, AWAY).details["bar_40"]["released"])

    def test_hands_off_limit(self):
        s = DEMO[0]
        r = zone_radius(SimpleNamespace(opening=1.0))
        for above, ok in ((0.0105, True), (0.0095, False)):
            tips = np.array([s.pos[0], s.pos[1], s.pos[2] + L.BAR_W / 2 + r + above])
            off, _ = hands_off(s, "bar_48", {"left": AWAY["left"], "right": arm(tips, 1.0)})
            self.assertEqual(off, ok, above)

    def test_zone_radius(self):
        self.assertAlmostEqual(zone_radius(SimpleNamespace(opening=1.0)) * 1000, 56.6, delta=0.1)
        self.assertAlmostEqual(zone_radius(SimpleNamespace(opening=0.0)) * 1000, 15.7, delta=0.1)

    def test_point_distance(self):
        s = DEMO[0]
        self.assertEqual(point_distance(s.pos, s, "bar_48"), 0.0)
        p = np.array(s.pos) + np.array([L.BAR_LEN["bar_48"] / 2 + 0.010, 0.0, 0.0])
        self.assertAlmostEqual(point_distance(p, s, "bar_48"), 0.010)
        self.assertGreater(zone_gap(AWAY["right"], s, "bar_48"), 0.05)

    def test_moving(self):
        tr, _ = feed(DEMO)
        res = None
        for k in range(WINDOW + 1):  # sliding 0.3 mm a step: 3 mm per 0.5 s
            res = tr.update([side(cx=0.323 + 0.0003 * k), DEMO[1]])
        self.assertFalse(res.details["bar_48"]["settled"])
        self.assertEqual(res.score, 0.5)

    def test_slow_creep_under_the_limit(self):
        tr, _ = feed(DEMO)
        res = None
        for k in range(SETTLE):  # 0.1 mm a step: 1 mm per 0.5 s
            res = tr.update([side(cx=0.323 + 0.0001 * k), DEMO[1]])
        self.assertTrue(res.details["bar_48"]["settled"])
        self.assertEqual(res.score, 1.0)

    def test_turning(self):
        tr, _ = feed(DEMO)
        res = None
        for k in range(WINDOW + 1):  # 0.12 degrees a step about its centre: 1.2 degrees in 0.5 s
            res = tr.update([side(yaw=0.12 * k), DEMO[1]])
        self.assertFalse(res.details["bar_48"]["settled"])
        self.assertLess(res.score, 1.0)

    def test_earlier_motion_blocks_full_credit_only(self):
        """Settled in the last 0.5 s but moving in the 0.5 s before it: counted (0.75), not still for 1 s."""
        tr = Tracker(HZ)
        for k in range(WINDOW + 1):
            tr.update([side(cx=0.320 + 0.0003 * k), DEMO[1]], AWAY)
        res = None
        for _ in range(WINDOW):
            res = tr.update([side(cx=0.320 + 0.0003 * WINDOW), DEMO[1]], AWAY)
        self.assertTrue(res.details["bar_48"]["counted"])
        self.assertFalse(res.details["bar_48"]["still_1s"])
        self.assertEqual(res.score, 0.75)

    def test_reported_velocity_is_ignored(self):
        """Stillness is judged on positions: a chattering reported velocity does not matter."""
        bars = [side(vel=(0.3, -0.2, 0.1)), rear(vel=(-0.4, 0.0, 0.2))]
        self.assertEqual(final(bars).score, 1.0)


class Exploits(unittest.TestCase):
    """Strategies other than fitting the bars to the code earn nothing (or only what the code says)."""

    def test_rear_bar_centred_on_the_toilet(self):
        self.assertEqual(final([side(), CORNER_40]).score, 0.75)

    def test_both_bars_on_the_side_wall(self):
        """Both bars along the side wall, end to end: the side wall is compliant, the rear wall bare."""
        b40 = side("bar_40", cx=0.2265 - 0.003 - L.BAR_LEN["bar_40"] / 2 + 0.083)
        res = final([side(), b40])
        self.assertEqual(res.score, 0.5)

    def test_stacked_in_the_corner(self):
        res = final([side(), rear(lift=L.BAR_W)])  # the rear bar laid on top of something
        self.assertEqual(res.score, 0.5)

    def test_held_correctly_but_never_released(self):
        self.assertLess(final(DEMO, holding(DEMO[1])).score, 1.0)
        self.assertLess(final(DEMO, holding(DEMO[0])).score, 1.0)

    def test_crosswise_across_the_corner(self):
        self.assertEqual(final([side(yaw=45.0), rear(yaw=45.0)]).score, 0.0)

    def test_random_states_score_a_tier(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(400):
            bars = []
            for name in L.NAMES:
                bars.append(bar(name, rng.uniform(0.20, 0.47), rng.uniform(-0.33, 0.01), rng.choice(
                    [0.0, 90.0, 180.0, rng.uniform(0, 180)]) + rng.uniform(-3, 3), lift=rng.choice([0.0, 0.0, 0.01]),
                    roll=rng.choice([0.0, 0.0, 30.0])))
            res = final(bars)
            self.assertIn(res.score, TIERS)
            self.assertEqual(res.success, res.score == 1.0)
            seen.add(res.score)
            if res.score > 0:  # every positive score has a bar along a wall
                self.assertTrue(res.details["side_wall"]["has"] or res.details["rear_wall"]["has"])
        self.assertIn(0.0, seen)


class BrokenStates(unittest.TestCase):
    """NaN, inf, zero quaternions, missing fields, unreadable arms, resets."""

    def test_invalid_states(self):
        good = side()
        cases = []
        for key, val in (("pos", np.array([np.nan, 0.0, 0.0])), ("quat_xyzw", np.zeros(4)),
                         ("lin_vel", np.array([np.inf, 0.0, 0.0])), ("ang_vel", np.array([0.0, np.nan, 0.0])),
                         ("pos", np.array([0.0, 0.0])), ("quat_xyzw", None)):
            st = SimpleNamespace(**vars(good))
            setattr(st, key, val)
            cases.append(st)
        missing = SimpleNamespace(pos=good.pos, quat_xyzw=good.quat_xyzw)
        cases.append(missing)
        for st in cases:
            self.assertFalse(valid(st))
            tr, _ = feed(DEMO)
            n_hist = len(tr.history)
            res = tr.update([st, DEMO[1]], AWAY)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
            self.assertFalse(res.details["valid_state"])
            self.assertEqual(len(tr.history), n_hist)  # the record is left as it was
            self.assertEqual(tr.bad_steps, 1)
            self.assertEqual(tr.update(DEMO, AWAY).score, 1.0)  # and the next good step scores as before

    def test_wrong_number_of_bars(self):
        tr = Tracker(HZ)
        self.assertEqual(tr.update([side()], AWAY).score, 0.0)
        self.assertEqual(tr.bad_steps, 1)

    def test_arms_unreadable(self):
        tr, res = feed(DEMO, arms=None)
        self.assertEqual(res.score, 1.0)
        self.assertEqual(tr.arms_unavailable, SETTLE)
        broken = {"left": AWAY["left"], "right": SimpleNamespace(tcp=[np.nan] * 3, tips=[0, 0, 0], opening=0.5)}
        self.assertFalse(arms_valid(broken))
        self.assertFalse(arms_valid({"left": AWAY["left"]}))
        self.assertTrue(arms_valid(AWAY))

    def test_teleport_back_to_start_resets(self):
        tr, _ = feed(DEMO)
        self.assertEqual(tr.best, 1.0)
        res = tr.update([start(n) for n in L.NAMES], AWAY)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(tr.best, 0.0)
        self.assertEqual(res.score, 0.0)
        # the new trial needs its own 0.5 s before anything counts
        _, res = feed(DEMO, n=WINDOW, tr=tr)
        self.assertEqual(res.score, 0.0)

    def test_gradual_return_is_not_a_reset(self):
        tr, _ = feed(DEMO)
        steps = tr.steps
        a, b = np.array(DEMO[0].pos[:2]), np.array(L.BAR_START["bar_48"][:2])
        for k in range(1, 41):  # carried back over 2 s
            p = a + (b - a) * k / 40
            tr.update([bar("bar_48", p[0], p[1], 90.0 * k / 40), DEMO[1]], AWAY)
        self.assertEqual(tr.steps, steps + 40)

    def test_explicit_reset(self):
        tr, _ = feed(DEMO)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.arms_unavailable, tr.bad_steps), (0, 0, 0, 0))
        self.assertIsNone(tr.prev)

    def test_result_is_idempotent(self):
        tr, res = feed(DEMO)
        n = len(tr.history)
        for _ in range(3):
            self.assertEqual(tr.result(DEMO, AWAY).score, res.score)
        self.assertEqual(len(tr.history), n)


class Helpers(unittest.TestCase):
    def test_heading(self):
        self.assertAlmostEqual(heading_deg(side(yaw=0.0)), 0.0, places=6)
        self.assertAlmostEqual(heading_deg(side(yaw=-1.0)), 179.0, places=6)
        self.assertAlmostEqual(heading_change_deg(179.0, 1.0), 2.0)
        self.assertAlmostEqual(heading_change_deg(90.0, 0.0), 90.0)
        self.assertEqual(heading_change_deg(float("nan"), 0.0), 90.0)

    def test_flat_and_lowest(self):
        s = side()
        self.assertAlmostEqual(flat_err_deg(s), 0.0, places=6)
        self.assertAlmostEqual(lowest(s, "bar_48"), L.TOP, places=9)
        self.assertAlmostEqual(flat_err_deg(side(roll=30.0)), 30.0, places=6)
        r = side(roll=45.0)
        self.assertAlmostEqual(lowest(r, "bar_48"), L.TOP + L.BAR_W / 2 - L.BAR_W / math.sqrt(2), places=9)

    def test_ends(self):
        a, b = ends(side(), "bar_48")
        np.testing.assert_allclose(sorted([a[0], b[0]]), [0.227, 0.419], atol=1e-9)
        a, b = ends(rear(), "bar_40")
        np.testing.assert_allclose(sorted([a[1], b[1]]), [-0.303, -0.143], atol=1e-9)


def load_task():
    stub = None
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            stub = d
            break
    if stub is None:
        raise unittest.SkipTest("needs the offline yam_arena stub (tools/offline/yam_stub)")
    if str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    return importlib.import_module("task")


class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signal, the demo,
    task.json and README."""

    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        import rigstate
        cls.rig = rigstate

    def objs(self, bars):
        return dict(zip(L.NAMES, bars, strict=True))

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], list(L.NAMES))
        for o in loose:
            self.assertLessEqual(o.mass, 0.30)
            self.assertAlmostEqual(o.size[0], L.BAR_LEN[o.name])
            self.assertAlmostEqual(o.pos[2], L.TOP + L.BAR_W / 2)
        self.assertAlmostEqual(t.REAR_KERB.pos[0] - t.REAR_KERB.size[0] / 2, L.REAR_FACE_X)
        self.assertAlmostEqual(t.SIDE_KERB.pos[1] + t.SIDE_KERB.size[1] / 2, L.SIDE_FACE_Y)
        self.assertAlmostEqual(t.BOWL.pos[1], L.CL_Y)
        self.assertAlmostEqual(t.TANK.pos[1], L.CL_Y)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        # everything inside x 0.205..0.465, |y| <= 0.325: clear of what the ready arms hide in the screenshot
        for o in t.OBJECTS:
            size = (2 * o.radius, 2 * o.radius) if hasattr(o, "radius") else o.size[:2]
            c, s, yaw = o.pos, size, getattr(o, "yaw", 0.0) or 0.0
            hx = abs(math.cos(yaw)) * s[0] / 2 + abs(math.sin(yaw)) * s[1] / 2
            hy = abs(math.sin(yaw)) * s[0] / 2 + abs(math.cos(yaw)) * s[1] / 2
            self.assertGreaterEqual(c[0] - hx, 0.205 - 1e-9, o.name)
            self.assertLessEqual(c[0] + hx, 0.465 + 1e-9, o.name)
            self.assertLessEqual(max(abs(c[1] - hy), abs(c[1] + hy)), 0.325 + 1e-9, o.name)

    def test_hooks_feed_the_tracker(self):
        t = self.task
        t._TRACKER.reset()
        self.feed_arms(AWAY)
        ok = False
        for _ in range(SETTLE + 20):
            ok = t.success(self.objs(DEMO))
        self.assertTrue(ok)
        self.assertEqual(t.score(self.objs(DEMO)), 1.0)
        t._TRACKER.reset()

    def test_reset_callback(self):
        t = self.task
        self.feed_arms(AWAY)
        for _ in range(SETTLE):
            t.success(self.objs(DEMO))
        self.assertGreater(t._TRACKER.steps, 0)
        self.rig.offline_reset()
        self.assertEqual(t._TRACKER.steps, 0)

    def test_trace(self):
        t = self.task
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.feed_arms(AWAY)
                t.success(self.objs(DEMO))
            finally:
                del os.environ["TASK_TRACE"]
            row = json.loads(pathlib.Path(path).read_text().splitlines()[-1])
        for k in ("score", "success", "tier_reason", "side_wall", "rear_wall", "bar_48", "bar_40", "right_tips",
                  "openings", "arms_unavailable", "episode"):
            self.assertIn(k, row)
        for k in ("wall", "gap_mm", "heading_deg", "path_mm", "released", "zone_gap_mm", "centre_mm"):
            self.assertIn(k, row["bar_48"])

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(list(plan), ["right"])
        wps = plan["right"]
        self.assertAlmostEqual(sum(w.duration for w in wps), 34.5)
        for w in wps:
            if w.label.startswith(("grip it", "let go")):
                self.assertGreaterEqual(w.duration, 1.5, w.label)  # every open / close takes >= 1.5 s
        places = [w for w in wps if w.label == "let go of it"]
        self.assertEqual(len(places), 2)
        np.testing.assert_allclose(places[0].pos[:2], t.place_point("bar_48"))
        np.testing.assert_allclose(places[1].pos[:2], t.place_point("bar_40"))
        self.assertAlmostEqual(math.degrees(places[0].yaw), -1.15)
        self.assertAlmostEqual(math.degrees(places[1].yaw), 88.85)
        self.assertAlmostEqual(t.DROP, 0.002)
        self.assertAlmostEqual(t.PLACE_Z - t.PAD - 0.008 - L.TOP, t.DROP)
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(t.demo(), {})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(list(t.demo()), ["right"])
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(t.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_place_trims(self):
        t = self.task
        for name in L.NAMES:
            dx, dy = t.PLACE_TRIM[name]
            self.assertAlmostEqual(math.hypot(dx, dy), 0.004)
        np.testing.assert_allclose(t.place_point("bar_48"), (0.3227, -0.3006), atol=2e-4)
        np.testing.assert_allclose(t.place_point("bar_40"), (0.4389, -0.2265), atol=2e-4)

    def test_instruction_and_task_json(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        for s in ("1 in = 4 mm", "rear wall", "side wall", "40 in and 48 in", "2010 ADA Standards", "1-1/2 in",
                  "let go"):
            self.assertIn(s, spec["instruction"])
        for k, n in {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150,
                     "fidelity": 490, "reset_notes": 580}.items():
            self.assertLessEqual(len(spec[k]), n, k)
        self.assertIn("1,200 steps", spec["run_limit"])
        setup = spec["setup"]
        for s in ("45.5", "-31.5", "-24.7", "192", "160", "58 g", "48 g", "aluminium", "4 mm tall"):
            self.assertIn(s, setup)
        scoring = spec["scoring"]
        for s in ("42 in", "12 in", "54 in", "36 in", "24 in", "1.0", "0.75", "0.5", "0.25", "0 ", "10 mm"):
            self.assertIn(s, scoring)

    def test_text_hygiene(self):
        bad = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")  # en and em dashes
        for p in [*HERE.glob("*.py"), *HERE.glob("*.md"), *HERE.glob("*.json"), *HERE.glob("tests/*.py")]:
            text = p.read_text()
            self.assertIsNone(bad.search(text), p.name)
            if p.suffix in (".json", ".md"):
                self.assertFalse(_ident_hits(text), p.name)



def _ident_hits(text):
    """Team-identifying words found in text. They are kept as sha256 prefixes so this file names nobody."""
    import hashlib
    import re as _re
    banned = {"54622c861585ec4f", "7a5e09c6c81cf86f", "a4cde7cecd46cd77", "b1fc1d0d7b2f62fa", "c70eca6b0f88f44d", "c857d09db23e6822", "d807637b43a9a01c", "da3a1991ce3a0d58", "db6017dfdfe82073", "f0556d0df0fd581f"}
    found = []
    for word in set(_re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())):
        for w in {word, *word.split("-")}:
            if hashlib.sha256(w.encode()).hexdigest()[:16] in banned:
                found.append(w)
    return found


if __name__ == "__main__":
    unittest.main()
