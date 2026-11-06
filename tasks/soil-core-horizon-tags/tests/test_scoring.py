"""Offline tests for scoring.py (and the hooks in task.py): synthetic tag states and synthetic arm snapshots fed
through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import itertools
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
    Tracker, held, nearest_boundary, on_tray, path_length, reach, side_of_core, tagged_boundaries, tilt_deg,
    tip_offset, upright, valid,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad-tip point below the TCP in a straight-down grip
B = dict((name, y) for name, _, y in L.BOUNDARIES)  # boundary name -> world y
BY = L.BOUNDARY_Y
NAMES = L.TAG_NAMES
PX = 0.360  # the demo's place x: 25 mm in front of the core's near edge
FAR_X = 0.460  # 25 mm beyond its far edge
TOUCH_NEAR = L.CORE_X0 - L.TAG_RADIUS  # 0.375: pressed against the near edge
TOUCH_FAR = L.CORE_X1 + L.TAG_RADIUS  # 0.445
READY_TIPS = {"left": (0.323, 0.305, L.TOP + 0.25), "right": (0.323, -0.305, L.TOP + 0.25)}


def about_x(deg):
    h = math.radians(deg) / 2
    return np.array([math.sin(h), 0.0, 0.0, math.cos(h)])


def tag(x, y, bottom=L.TRAY_TOP, tilt=0.0, vel=(0.0, 0.0, 0.0), quat=None):
    """An ObjectState-like tag: lowest point at ``bottom``, tilted ``tilt`` degrees about x, with its world-aligned
    bounding box as the simulator reports it."""
    q = about_x(tilt) if quat is None else np.array(quat, float)
    x_, y_, z_, w_ = q / (np.linalg.norm(q) or 1.0)
    u = np.array([2 * (x_ * z_ + y_ * w_), 2 * (y_ * z_ - x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)])
    ext = np.abs(u) * L.TAG_HEIGHT / 2 + L.TAG_RADIUS * np.sqrt(np.clip(1 - u * u, 0, 1))
    c = np.array([x, y, bottom + ext[2]])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c - ext, high=c + ext)


def start_tags():
    return {name: tag(x, y, bottom=L.TOP) for name, x, y in L.TAG_STARTS}


def layout(**placed):
    """The start layout with some tags moved: layout(tag_l1=tag(...), ...)."""
    out = start_tags()
    out.update(placed)
    return out


OPEN, ON_TAG, SHUT = 1.0, 0.020 / L.FULL_GAP, 0.0  # rigstate openings: fully open, closed on a tag, fully closed


def snap(left=None, right=None, left_open=OPEN, right_open=OPEN):
    """Feed one synthetic arm snapshot (pad-tip points and openings) through rigstate, as the offline checker
    does."""
    feed = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        tips = np.array(READY_TIPS[side] if tips is None else tips, float)
        feed[side] = {"tcp": tips + np.array([0.0, 0.0, TIPS]), "tips": tips, "opening": op}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def play(tags, n=12, tracker=None, arms="ready"):
    """Hold one layout for n control steps (the arms parked high unless given); return the last Result."""
    t = tracker or Tracker()
    a = snap() if arms == "ready" else arms
    r = None
    for _ in range(n):
        r = t.update(tags, a)
    return r


def at(*ks, x=PX, off=0.0):
    """Tags at boundaries ks (indices 0..3), the left arm's tags first, then the right arm's."""
    return {NAMES[i]: tag(x, BY[k] + off) for i, k in enumerate(ks)}


ALL_FOUR = at(0, 1, 2, 3)


class Geometry(unittest.TestCase):
    def test_layout_numbers(self):
        self.assertEqual([round(y, 3) for y in BY], [0.234, 0.107, -0.034, -0.154])
        self.assertEqual([round(d * 1000) for _, d, _ in L.BOUNDARIES], [46, 173, 314, 434])
        self.assertGreaterEqual(min(b - a for a, b in zip(L.BOUNDARY_DEPTHS, L.BOUNDARY_DEPTHS[1:])), 0.120)
        self.assertAlmostEqual(L.NEAR_BAND[0], 0.340)
        self.assertAlmostEqual(L.FAR_BAND[1], 0.480)
        # the boxes of the core abut exactly and span it; Bt and BC are two boxes each
        segs = L.SEGMENTS
        self.assertEqual(segs[0][1], 0.0)
        self.assertAlmostEqual(segs[-1][2], L.CORE_LEN)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a[2], b[1])
        self.assertEqual([s[3] for s in segs], ["A", "E", "Bt", "Bt", "BC", "BC", "C"])
        joints = {round(s[2], 3) for s in segs[:-1]}
        self.assertEqual(joints - {round(d, 3) for d in L.BOUNDARY_DEPTHS}, {0.245, 0.374})
        # every counted tag stands wholly on the tray
        self.assertGreaterEqual(L.NEAR_BAND[0] - L.TAG_RADIUS, L.TRAY_X0)
        self.assertLessEqual(L.FAR_BAND[1] + L.TAG_RADIUS, L.TRAY_X1)

    def test_tilt_is_folded(self):
        self.assertAlmostEqual(tilt_deg(tag(PX, 0.0)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(tag(PX, 0.0, tilt=180.0)), 0.0, places=5)  # the other end down
        self.assertAlmostEqual(tilt_deg(tag(PX, 0.0, tilt=170.0)), 10.0, places=5)
        self.assertAlmostEqual(tilt_deg(tag(PX, 0.0, tilt=90.0)), 90.0, places=5)

    def test_upright_is_10_degrees(self):
        self.assertTrue(upright(tag(PX, 0.0, tilt=9.5)))
        self.assertTrue(upright(tag(PX, 0.0, tilt=-9.5)))
        self.assertFalse(upright(tag(PX, 0.0, tilt=10.5)))
        self.assertTrue(upright(tag(PX, 0.0, tilt=171.0)))

    def test_on_the_tray_is_3_mm(self):
        self.assertTrue(on_tray(tag(PX, 0.0)))
        self.assertTrue(on_tray(tag(PX, 0.0, bottom=L.TRAY_TOP + 0.0029)))
        self.assertTrue(on_tray(tag(PX, 0.0, bottom=L.TRAY_TOP - 0.0029)))
        self.assertFalse(on_tray(tag(PX, 0.0, bottom=L.TRAY_TOP + 0.0031)))
        self.assertFalse(on_tray(tag(0.30, 0.0, bottom=L.TOP)))  # on the table
        self.assertFalse(on_tray(tag(L.CORE_X, 0.0, bottom=L.CORE_TOP_Z)))  # on the core
        self.assertFalse(on_tray(tag(PX, 0.0, bottom=L.TRAY_TOP + L.TAG_HEIGHT)))  # on another tag

    def test_beside_the_core(self):
        for x, want in ((0.3395, None), (0.3405, "near"), (PX, "near"), (TOUCH_NEAR, "near"), (0.385, "near"),
                        (0.386, None), (L.CORE_X, None), (0.434, None), (0.435, "far"), (TOUCH_FAR, "far"),
                        (FAR_X, "far"), (0.4795, "far"), (0.4805, None)):
            self.assertEqual(side_of_core(tag(x, 0.0)), want, x)

    def test_nearest_boundary(self):
        k, off = nearest_boundary(tag(PX, B["B2"] + 0.004))
        self.assertEqual(k, 1)
        self.assertAlmostEqual(off, 0.004)
        k, off = nearest_boundary(tag(PX, B["B3"] - 0.007))
        self.assertEqual(k, 2)
        self.assertAlmostEqual(off, -0.007)

    def test_tip_offset_follows_a_tilted_axis(self):
        t = tag(PX, 0.0, tilt=8.0)  # top leans toward -y
        u = np.array([0.0, -math.sin(math.radians(8.0)), math.cos(math.radians(8.0))])
        top_on_axis = t.pos + u * (L.TAG_HEIGHT / 2)
        arm = SimpleNamespace(tips=top_on_axis)
        xy, above = tip_offset(arm, t)
        self.assertAlmostEqual(xy, 0.0, places=6)
        self.assertLess(above, 0.0)

    def test_path_length(self):
        self.assertAlmostEqual(path_length([np.zeros(3), np.array([0.003, 0, 0]), np.array([0.003, 0.004, 0])]),
                               0.007)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = play(start_tags())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertTrue(Tracker.at_start(start_tags()))

    def test_all_four_scores_one(self):
        r = play(layout(**ALL_FOUR))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["tagged"], ["B1", "B2", "B3", "B4"])

    def test_every_tier(self):
        for ks, want in (((), 0.0), ((0,), 0.25), ((0, 1), 0.5), ((0, 1, 2), 0.75), ((0, 1, 2, 3), 1.0),
                         ((3,), 0.25), ((1, 3), 0.5), ((0, 2, 3), 0.75)):
            r = play(layout(**at(*ks)))
            self.assertEqual(r.score, want, ks)
            self.assertEqual(r.success, want == 1.0)

    def test_one_tag_15_mm_off_scores_075(self):
        tags = layout(**ALL_FOUR)
        tags["tag_r2"] = tag(PX, BY[3] + 0.015)
        self.assertEqual(play(tags).score, 0.75)

    def test_tolerance_is_10_mm_either_way(self):
        for off, want in ((0.0095, 1.0), (-0.0095, 1.0), (0.0105, 0.75), (-0.0105, 0.75)):
            tags = layout(**ALL_FOUR)
            tags["tag_l2"] = tag(PX, BY[1] + off)
            self.assertEqual(play(tags).score, want, off)

    def test_only_one_arms_tags_scores_half(self):
        self.assertEqual(play(layout(**at(0, 1))).score, 0.5)  # left arm alone: B1 and B2
        right = {"tag_r1": tag(PX, BY[2]), "tag_r2": tag(PX, BY[3])}
        self.assertEqual(play(layout(**right)).score, 0.5)  # right arm alone: B3 and B4

    def test_two_tags_at_one_boundary_count_once(self):
        tags = layout(tag_l1=tag(PX, BY[0] - 0.012), tag_l2=tag(PX, BY[0] + 0.009))
        tags["tag_l1"] = tag(PX, BY[0] - 0.005)
        r = play(tags)
        self.assertEqual(r.score, 0.25)
        self.assertEqual(r.details["tagged"], ["B1"])

    def test_pressed_against_the_core_and_far_side_count(self):
        tags = layout(tag_l1=tag(TOUCH_NEAR, BY[0]), tag_l2=tag(TOUCH_FAR, BY[1]), tag_r1=tag(FAR_X, BY[2]),
                      tag_r2=tag(0.341, BY[3]))
        r = play(tags)
        self.assertEqual(r.score, 1.0)
        self.assertEqual([r.details["tags"][n]["side"] for n in NAMES], ["near", "far", "far", "near"])

    def test_too_far_from_the_core(self):
        tags = layout(**ALL_FOUR)
        tags["tag_l1"] = tag(0.335, BY[0])  # 50 mm in front of the near edge
        tags["tag_r1"] = tag(0.485, BY[2])  # 50 mm beyond the far edge
        self.assertEqual(play(tags).score, 0.5)

    def test_score_is_idempotent(self):
        t = Tracker()
        tags = layout(**at(0, 1, 2))
        play(tags, tracker=t)
        a = snap()
        self.assertEqual(t.result(tags, a).score, t.result(tags, a).score)
        self.assertEqual(t.result(tags, a).score, 0.75)


class Release(unittest.TestCase):
    """(e) A tag counts only once let go and clear: no gripper's pad-tip point lower than 10 mm above its top (56 mm
    above the table for a tag standing on the tray) with the tag's axis within that gripper's finger reach. The
    gripper's yaw is not read, so the reach is the widest over every yaw: hypot(half the gap + 10 mm, 20 mm) plus
    the tag's 10 mm radius plus 5 mm, i.e. 75 mm fully open, 43 mm closed on a tag, 37 mm fully closed."""

    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(OPEN), 0.07536, places=5)
        self.assertAlmostEqual(L.release_reach(ON_TAG), 0.04328, places=5)
        self.assertAlmostEqual(L.release_reach(SHUT), 0.03736, places=5)
        self.assertAlmostEqual(L.release_reach(1.7), L.release_reach(OPEN))  # clipped to 0..1
        self.assertAlmostEqual(L.release_reach(-0.2), L.release_reach(SHUT))
        for bad in (float("nan"), float("inf"), None):  # an unreadable opening: the widest reach
            self.assertAlmostEqual(reach(SimpleNamespace(opening=bad)), L.release_reach(OPEN))
        self.assertAlmostEqual(reach(SimpleNamespace()), L.release_reach(OPEN))

    def test_a_held_tag_does_not_count(self):
        tags = layout(**ALL_FOUR)
        grip = (PX, BY[3], L.TRAY_TOP + 0.012)  # pads closed on the tag, tips 12 mm above its bottom
        r = play(tags, arms=snap(right=grip, right_open=ON_TAG))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        self.assertTrue(r.details["tags"]["tag_r2"]["held"])
        self.assertEqual(r.details["reach_mm"], {"left": 75.4, "right": 43.3})

    def test_success_needs_the_grippers_clear(self):
        t = Tracker()
        tags = layout(**ALL_FOUR)
        top = L.TRAY_TOP + L.TAG_HEIGHT
        r = play(tags, tracker=t, arms=snap(left=(PX, BY[1], top + 0.009)))  # tips 9 mm above the tag's top
        self.assertFalse(r.success)
        self.assertEqual(r.score, 0.75)
        r = play(tags, n=1, tracker=t, arms=snap(left=(PX, BY[1], top + 0.011)))  # 11 mm above: let go
        self.assertTrue(r.success)
        self.assertEqual(r.score, 1.0)

    def test_an_open_finger_propping_a_tag_is_holding_it(self):
        # The review's case: the open gripper (pads 94 mm apart) offset along its closing axis (y here) so that one
        # finger's inside face touches the tag (axis 37 mm from the pad-tip point) or its outside face does (67 mm).
        tags = layout(**ALL_FOUR)
        low = L.TRAY_TOP + 0.020
        inside = L.FULL_GAP / 2 - L.TAG_RADIUS  # 36.95 mm
        outside = L.FULL_GAP / 2 + L.FINGER_THICK + L.TAG_RADIUS  # 66.95 mm
        for off in (inside, outside, -inside):
            r = play(tags, arms=snap(left=(PX, BY[0] + off, low)))
            self.assertEqual(r.score, 0.75, off)
            self.assertTrue(r.details["tags"]["tag_l1"]["held"], off)
        # the same along x (yaw unknown, so every direction is alike)
        self.assertEqual(play(tags, arms=snap(left=(PX - inside, BY[0], low))).score, 0.75)

    def test_open_reach_is_75_mm(self):
        tags = layout(**ALL_FOUR)
        low = L.TRAY_TOP + 0.020
        self.assertEqual(play(tags, arms=snap(left=(PX, BY[0] + 0.0745, low))).score, 0.75)
        self.assertEqual(play(tags, arms=snap(left=(PX, BY[0] + 0.0765, low))).score, 1.0)
        self.assertEqual(play(tags, arms=snap(left=(PX - 0.0765, BY[0], low))).score, 1.0)

    def test_the_reach_follows_the_opening(self):
        tags = layout(**ALL_FOUR)
        low = L.TRAY_TOP + 0.020
        for op, inside, outside in ((SHUT, 0.0365, 0.0385), (ON_TAG, 0.0425, 0.0445), (0.5, 0.0530, 0.0550)):
            self.assertEqual(play(tags, arms=snap(left=(PX, BY[0] + inside, low), left_open=op)).score, 0.75, op)
            self.assertEqual(play(tags, arms=snap(left=(PX, BY[0] + outside, low), left_open=op)).score, 1.0, op)
        nan = snap(left=(PX, BY[0] + 0.070, low), left_open=float("nan"))  # unreadable opening: fully open
        self.assertEqual(play(tags, arms=nan).score, 0.75)

    def test_either_arm_counts(self):
        tags = layout(**ALL_FOUR)
        over_b2 = (PX, BY[1], L.TRAY_TOP + 0.030)
        self.assertEqual(play(tags, arms=snap(right=over_b2)).score, 0.75)  # the other arm's gripper also counts

    def test_the_demo_retreat_is_clear(self):
        tags = layout(**ALL_FOUR)
        tips = L.TOP + 0.100 - TIPS  # TCP 10 cm up: tips 87 mm above the table
        r = play(tags, arms=snap(left=(PX, BY[1], tips), right=(PX, BY[3], tips)))
        self.assertTrue(r.success)

    def test_neighbouring_boundaries_are_out_of_reach(self):
        # an open gripper low at one boundary (placing its tag) never counts as holding the tag at the next one
        self.assertGreater(min(abs(a - b) for a, b in zip(BY, BY[1:])), L.release_reach(OPEN) + 0.040)
        tags = layout(**ALL_FOUR)
        for k in range(4):
            r = play(tags, arms=snap(left=(PX, BY[k], L.TRAY_TOP + 0.012)))
            self.assertEqual(r.score, 0.75, k)
            self.assertEqual([n for n in NAMES if r.details["tags"][n]["held"]], [NAMES[k]])

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        t = Tracker()
        r = play(layout(**ALL_FOUR), tracker=t, arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)
        self.assertFalse(r.details["arms_available"])
        self.assertIsNone(r.details["reach_mm"])
        broken = {"left": SimpleNamespace(tips=np.array([np.nan, 0, 0])), "right": snap()["right"]}
        self.assertFalse(held(tag(PX, BY[0]), broken))


class Stillness(unittest.TestCase):
    """(c) At rest = the centre's path over the last 0.5 s (10 control steps) under 3 mm; velocity is never read."""

    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        tags = layout(**ALL_FOUR)
        r = play(tags, n=10, tracker=t)
        self.assertEqual(r.score, 0.0)  # 10 positions, 9 moves: not yet 0.5 s
        r = play(tags, n=1, tracker=t)
        self.assertEqual(r.score, 1.0)

    def test_velocity_chatter_is_ignored(self):
        tags = {n: tag(PX, BY[i], vel=(0.045, -0.03, 0.02)) for i, n in enumerate(NAMES)}
        self.assertEqual(play(layout(**tags)).score, 1.0)

    def test_a_moving_tag_does_not_count_until_it_stops(self):
        t = Tracker()
        rest = layout(**at(0, 1, 2))
        y0 = BY[3] + 0.030
        play(dict(rest, tag_r2=tag(PX, y0)), n=11, tracker=t)  # the other three placed and still
        r = None
        for i in range(1, 16):  # sliding 2 mm per step (40 mm/s) to B4, through the 10 mm window
            tags = dict(rest, tag_r2=tag(PX, y0 - 0.002 * i))
            r = t.update(tags, snap())
            self.assertEqual(r.score, 0.75)
        self.assertAlmostEqual(tags["tag_r2"].pos[1], BY[3])
        for _ in range(8):
            r = t.update(tags, snap())
        self.assertEqual(r.score, 0.75)  # 8 still steps: the last two moves (4 mm) are still in the window
        r = t.update(tags, snap())
        self.assertEqual(r.score, 1.0)  # 9: one 2 mm move left in the last 0.5 s

    def test_a_slow_creep_under_3_mm_counts(self):
        t = Tracker()
        rest = layout(**at(0, 1, 2))
        r = None
        for i in range(20):
            r = t.update(dict(rest, tag_r2=tag(PX, BY[3] + 0.0002 * i)), snap())  # 2 mm per 0.5 s
        self.assertEqual(r.score, 1.0)

    def test_a_shiver_is_measured_as_path(self):
        t = Tracker()
        rest = layout(**at(0, 1, 2))
        r = None
        for i in range(20):  # +-0.2 mm back and forth: 0.4 mm per step, 4 mm per 0.5 s, though it goes nowhere
            r = t.update(dict(rest, tag_r2=tag(PX, BY[3] + (0.0002 if i % 2 else -0.0002))), snap())
        self.assertEqual(r.score, 0.75)


class NotCounted(unittest.TestCase):
    """Tags that count nowhere: lying down, tilted, stacked, on the core, on the table, inside the core's outline."""

    def check(self, bad, want=0.75):
        tags = layout(**at(0, 1, 2))
        tags["tag_r2"] = bad
        r = play(tags)
        self.assertEqual(r.score, want)
        self.assertFalse(r.details["tags"]["tag_r2"]["placed"])

    def test_lying_down(self):
        self.check(tag(PX, BY[3], tilt=90.0))

    def test_tilted_over_10_degrees(self):
        self.check(tag(PX, BY[3], tilt=15.0))

    def test_on_the_core(self):
        self.check(tag(L.CORE_X, BY[3], bottom=L.CORE_TOP_Z))
        self.check(tag(0.390, BY[3], bottom=L.CORE_TOP_Z))  # standing on the core's near edge

    def test_on_the_table(self):
        self.check(tag(0.300, BY[3], bottom=L.TOP))

    def test_stacked(self):
        # the upper of two stacked tags never counts; the lower one does if it is placed
        tags = layout(**at(0, 1, 2))
        tags["tag_r2"] = tag(PX, BY[2], bottom=L.TRAY_TOP + L.TAG_HEIGHT)
        r = play(tags)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["tags"]["tag_r2"]["on_tray"])
        tags = layout(tag_l1=tag(PX, BY[0], bottom=L.TRAY_TOP + L.TAG_HEIGHT))
        self.assertEqual(play(tags).score, 0.0)

    def test_inside_the_cores_outline(self):
        self.check(tag(0.400, BY[3]))

    def test_beyond_the_ends_of_the_core(self):
        tags = layout(tag_l1=tag(PX, 0.290), tag_r2=tag(PX, -0.290))
        self.assertEqual(play(tags).score, 0.0)


# -- blind layouts ----------------------------------------------------------------------------------------------
# Depth along the core in mm from its top end (the ground surface): the core spans 0..560, the tray -30..690.
CORE = (0.0, 560.0)
TRAY = (-30.0, 690.0)
DEPTHS = [d * 1000.0 for d in L.BOUNDARY_DEPTHS]
NOISE = 3.0  # mm of placement error a blind rule is also checked with


def split_points(a, b, n):  # the points dividing a..b into n equal parts
    return [a + (b - a) * k / n for k in range(1, n)]


def part_centres(a, b, n):  # the centres of n equal parts
    return [a + (b - a) * (k + 0.5) / n for k in range(n)]


def end_to_end(a, b, n):  # n points spread from one end to the other
    return [a + (b - a) * k / (n - 1) for k in range(n)]


def grid(start, step, sign):  # every ``step`` from ``start`` (the start included) while on the tray
    out, k = [], 0
    while TRAY[0] - 1e-9 <= start + sign * step * k <= TRAY[1] + 1e-9:
        out.append(start + sign * step * k)
        k += 1
    return out


def fours(pts):
    """The ways a blind rule with more points than tags can be followed with four tags: any four consecutive."""
    return [pts] if len(pts) <= 4 else [pts[i:i + 4] for i in range(len(pts) - 3)]


def reached(pts, tol=10.0):
    """The boundaries (indices) within ``tol`` mm of any of the points: pure geometry, no scorer."""
    return {k for k, d in enumerate(DEPTHS) if any(abs(p - d) <= tol for p in pts)}


def scored(pts):
    """The full scorer on four tags placed exactly at these depths beside the core (more than four: four at a time
    is what a trial allows, so the caller passes four)."""
    placed = {NAMES[i]: tag(PX, L.y_at(p / 1000)) for i, p in enumerate(pts)}
    return play(layout(**placed)).score


def simple_layouts():
    """Blind rules that score 0: for the core and for the tray, the points dividing it into 2 to 6 equal parts, the
    centres of 1 to 4 equal parts and 2 to 5 points end to end; a tag every 10 or 20 cm from either end of the
    core or from the tray's bottom end (the end itself included); the ends."""
    out = {}
    for what, (a, b) in (("core", CORE), ("tray", TRAY)):
        for n in range(2, 7):
            out[f"{what} split into {n}"] = split_points(a, b, n)
        for n in range(1, 5):
            out[f"centres of {n} parts of the {what}"] = part_centres(a, b, n)
        for n in range(2, 6):
            out[f"{n} points along the {what} end to end"] = end_to_end(a, b, n)
    for what, start, sign in (("core's top", CORE[0], 1), ("core's bottom", CORE[1], -1),
                              ("tray's bottom", TRAY[1], -1)):
        for step in (100.0, 200.0):
            out[f"{step / 10:.0f} cm grid from the {what}"] = grid(start, step, sign)
    out["the core's ends"] = list(CORE)
    out["the tray's ends"] = list(TRAY)
    return out


def few_layouts():
    """Blind rules that tag one boundary (0.25), or two with a few mm of error: the centres of 5 equal parts of the
    core or the tray, and a tag every 10 or 20 cm from the tray's top end."""
    return {
        "centres of 5 parts of the core": part_centres(*CORE, 5),  # 56, 168, 280, 392, 504
        "centres of 5 parts of the tray": part_centres(*TRAY, 5),  # 42, 186, 330, 474, 618
        "10 cm grid from the tray's top": grid(TRAY[0], 100.0, 1),  # -30, 70, 170, ...
        "20 cm grid from the tray's top": grid(TRAY[0], 200.0, 1),  # -30, 170, 370, 570
    }


def dense_layouts():
    """Rules with many more points than there are tags: the centres of 6 to 12 equal parts, the points dividing it
    into 7 to 12 parts, 6 to 12 points end to end (of the core or the tray), and a 5 cm grid from any end of
    either."""
    out = {}
    for what, (a, b) in (("core", CORE), ("tray", TRAY)):
        for n in range(6, 13):
            out[f"centres of {n} parts of the {what}"] = part_centres(a, b, n)
            out[f"{n} points along the {what} end to end"] = end_to_end(a, b, n)
        for n in range(7, 13):
            out[f"{what} split into {n}"] = split_points(a, b, n)
    for what, start, sign in (("core top", CORE[0], 1), ("core bottom", CORE[1], -1),
                              ("tray top", TRAY[0], 1), ("tray bottom", TRAY[1], -1)):
        out[f"5 cm grid from the {what}"] = grid(start, 50.0, sign)
    return out


class BlindLayouts(unittest.TestCase):
    """Tags put down by a rule that does not look at the colours."""

    def test_layout_frame_matches(self):
        self.assertAlmostEqual(L.y_at(TRAY[0] / 1000), L.TRAY_Y_LEFT)
        self.assertAlmostEqual(L.y_at(TRAY[1] / 1000), L.TRAY_Y_RIGHT)
        self.assertAlmostEqual(L.y_at(CORE[1] / 1000), L.CORE_Y_TOP - L.CORE_LEN)

    def test_simple_rules_score_zero_and_miss_by_13_mm(self):
        worst = min(abs(p - d) for pts in simple_layouts().values() for p in pts for d in DEPTHS)
        self.assertAlmostEqual(worst, 13.0)  # so they score 0 even with a placement error under 3 mm
        for name, pts in simple_layouts().items():
            self.assertEqual(reached(pts, 10.0 + NOISE - 1e-6), set(), name)
            for four in fours(pts):
                self.assertEqual(scored(four), 0.0, (name, four))

    def test_the_ten_cm_grid_from_the_top(self):
        placed = {NAMES[i]: tag(PX, L.y_at(0.1 * (i + 2))) for i in range(4)}  # 200, 300, 400, 500 mm
        self.assertEqual(play(layout(**placed)).score, 0.0)
        placed = {NAMES[i]: tag(PX, L.y_at(0.1 * (i + 1))) for i in range(4)}  # 100 .. 400 mm
        self.assertEqual(play(layout(**placed)).score, 0.0)

    def test_the_reviewed_blind_rules(self):
        # The review's three: the first four centres of 5 equal parts of the core (B2 5 mm off, B1 10.0 mm off,
        # right at the tolerance) and four points of a 10 cm grid from the tray's top end (B2 3 mm off).
        five = part_centres(*CORE, 5)
        self.assertEqual([round(p) for p in five], [56, 168, 280, 392, 504])
        self.assertIn(scored(five[:4]), (0.25, 0.5))
        self.assertEqual(scored([-30.0, 70.0, 170.0, 270.0]), 0.25)
        self.assertEqual(scored([70.0, 170.0, 270.0, 370.0]), 0.25)

    def test_few_rules_tag_one_boundary_or_two_with_error(self):
        want = {  # boundaries within 10 mm of all points (exactly placed), and within 13 mm (3 mm of error)
            "centres of 5 parts of the core": ({0, 1}, {0, 1}),  # B1 at exactly 10.0 mm
            "centres of 5 parts of the tray": ({0}, {0, 1}),  # B1 4 mm, B2 13.0 mm
            "10 cm grid from the tray's top": ({1}, {1}),  # B2 3 mm; every other point 24 mm or more off
            "20 cm grid from the tray's top": ({1}, {1}),
        }
        for name, pts in few_layouts().items():
            exact, noisy = want[name]
            self.assertEqual(reached(pts, 10.0 + 1e-6), exact, name)
            self.assertEqual(reached(pts, 10.0 + NOISE), noisy, name)
            for four in fours(pts):
                self.assertLessEqual(scored(four), 0.5, (name, four))
        for name in ("centres of 5 parts of the tray", "10 cm grid from the tray's top",
                     "20 cm grid from the tray's top"):
            self.assertEqual(max(scored(f) for f in fours(few_layouts()[name])), 0.25, name)

    def test_few_rules_with_placement_error_stay_at_or_under_half(self):
        # every tag of every four-tag choice moved by -3, 0 or +3 mm along the core
        for name, pts in few_layouts().items():
            for four in fours(pts):
                for shift in itertools.product((-NOISE, 0.0, NOISE), repeat=len(four)):
                    self.assertLessEqual(len(reached([p + e for p, e in zip(four, shift)])), 2, (name, four))

    def test_dense_rules_reach_at_most_two_boundaries(self):
        worst_all, worst_four = 0, 0
        for name, pts in dense_layouts().items():
            fake = {f"t{i}": tag(PX, L.y_at(p / 1000)) for i, p in enumerate(pts)}
            n = len(tagged_boundaries({k: True for k in fake}, fake))  # all of the rule's points at once
            self.assertLessEqual(n, 2, name)
            worst_all = max(worst_all, n)
            for four in fours(pts):  # what four tags can do, with 3 mm of placement error
                worst_four = max(worst_four, len(reached(four, 10.0 + NOISE)))
        self.assertEqual(worst_all, 2)
        self.assertEqual(worst_four, 2)

    def test_the_vetted_layout_had_an_even_split_exploit(self):
        # Why the boundaries are at 46/173/314/434 mm: with the earlier 65/171/318/483 mm, four tags at the
        # centres of four equal parts of the core (70, 210, 350, 490 mm) tagged B1 and B4 (0.5).
        old = [65.0, 171.0, 318.0, 483.0]
        hits = sum(1 for d in old if any(abs(p - d) <= 10 for p in part_centres(*CORE, 4)))
        self.assertEqual(hits, 2)
        hits = sum(1 for d in DEPTHS if any(abs(p - d) <= 10 for p in part_centres(*CORE, 4)))
        self.assertEqual(hits, 0)

    def test_why_b1_and_b2_stay_where_they_are(self):
        # B2: the only whole-mm depth 120 mm or more from B1 and B3 that every score-0 rule misses by 13 mm
        pts = [p for v in simple_layouts().values() for p in v]
        ok = [d for d in range(0, 561) if min(abs(p - d) for p in pts) >= 13.0 - 1e-9]
        self.assertEqual([d for d in ok if DEPTHS[0] + 120 <= d <= DEPTHS[2] - 120], [173])
        # B1: any shallower depth that clears the 5-part centre at 56 mm lets some denser rule reach three
        dense = list(dense_layouts().values())
        for b1 in range(13, 46):
            self.assertGreaterEqual(min(abs(p - b1) for p in pts), 13.0 - 1e-9)
            d = [b1, *DEPTHS[1:]]
            self.assertEqual(max(sum(1 for x in d if any(abs(p - x) <= 10 for p in v)) for v in dense), 3, b1)

    def test_every_four_tag_choice_from_a_dense_rule_is_capped(self):
        # spot check by the full scorer: the 5 cm grid from the core's bottom, any four of its points
        pts = grid(CORE[1], 50.0, -1)
        best = 0.0
        for combo in itertools.combinations(pts, 4):
            placed = {NAMES[i]: tag(PX, L.y_at(p / 1000)) for i, p in enumerate(combo)}
            best = max(best, play(layout(**placed), n=11).score)
        self.assertLessEqual(best, 0.5)


class Robustness(unittest.TestCase):
    def test_nan_scores_zero_and_keeps_the_record(self):
        t = Tracker()
        tags = layout(**ALL_FOUR)
        play(tags, tracker=t)
        hist = {n: list(t.history[n]) for n in NAMES}
        bad = dict(tags, tag_l1=tag(PX, BY[0]))
        bad["tag_l1"].pos = np.array([np.nan, 0.0, 0.0])
        r = t.update(bad, snap())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual({n: list(t.history[n]) for n in NAMES}, hist)
        self.assertEqual(t.invalid_steps, 1)
        self.assertEqual(t.update(tags, snap()).score, 1.0)  # the next good step carries on

    def test_zero_quaternion_and_missing_tag(self):
        self.assertFalse(valid(tag(PX, 0.0, quat=(0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        tags = layout(**ALL_FOUR)
        tags["tag_r1"] = tag(PX, BY[2], quat=(0, 0, 0, 0))
        self.assertEqual(Tracker().update(tags, snap()).score, 0.0)
        tags = layout(**ALL_FOUR)
        del tags["tag_r1"]
        self.assertEqual(Tracker().update(tags, snap()).score, 0.0)

    def test_teleport_back_to_the_start_is_a_new_episode(self):
        t = Tracker()
        play(layout(**ALL_FOUR), tracker=t)
        steps = t.steps
        r = t.update(start_tags(), snap())
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.steps, 1)
        self.assertGreater(steps, 1)
        self.assertEqual(len(t.history["tag_l1"]), 1)

    def test_putting_one_tag_back_is_not_a_reset(self):
        t = Tracker()
        play(layout(**at(0, 1)), tracker=t)
        r = t.update(layout(tag_l1=tag(PX, BY[0])), snap())  # tag_l2 is back on its mark, tag_l1 is not
        self.assertEqual(r.score, 0.25)
        self.assertGreater(t.steps, 1)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = Tracker()
        play(layout(**ALL_FOUR), tracker=t)
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            rigstate.offline_reset()
            self.assertEqual(t.steps, 0)
            self.assertIsNone(rigstate.arms())
            self.assertEqual(t.result(layout(**ALL_FOUR), None).score, 0.0)  # no history yet
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


class Exploits(unittest.TestCase):
    """Unintended strategies and what they earn."""

    def test_holding_tags_at_the_boundaries_earns_nothing_for_them(self):
        # each gripper holds a tag standing at a boundary; the other two tags are not placed
        tags = layout(tag_l1=tag(PX, BY[0]), tag_r1=tag(PX, BY[2]))
        r = play(tags, arms=snap(left=(PX, BY[0], L.TRAY_TOP + 0.012), right=(PX, BY[2], L.TRAY_TOP + 0.012)))
        self.assertEqual(r.score, 0.0)

    def test_pressing_a_tag_down_from_above_is_holding_it(self):
        tags = layout(**ALL_FOUR)
        top = L.TRAY_TOP + L.TAG_HEIGHT
        self.assertEqual(play(tags, arms=snap(left=(PX + 0.012, BY[0], top))).score, 0.75)

    def test_one_tag_cannot_cover_two_boundaries(self):
        mid = (BY[2] + BY[3]) / 2
        self.assertEqual(play(layout(tag_r1=tag(PX, mid))).score, 0.0)
        gap = min(abs(a - b) for a, b in zip(BY, BY[1:]))
        self.assertGreater(gap, 2 * L.TAG_TOL)

    def test_tags_spread_over_the_whole_tray_width(self):
        # tags at the boundary depths but out at the tray's front edge, off the band beside the core
        tags = layout(**at(0, 1, 2, 3, x=L.TRAY_X0 + L.TAG_RADIUS))
        self.assertEqual(play(tags).score, 0.0)

    def test_a_drop_that_lands_upright_is_still_a_mark(self):
        t = Tracker()
        rest = layout(**at(0, 1, 2))
        z, v = 0.05, 0.0
        r = None
        while z > 0:  # free fall from 5 cm onto the tray at B4
            v += 9.81 * DT
            z = max(0.0, z - v * DT)
            r = t.update(dict(rest, tag_r2=tag(PX, BY[3], bottom=L.TRAY_TOP + z, vel=(0, 0, -v))), snap())
        self.assertLess(r.score, 1.0)
        r = play(dict(rest, tag_r2=tag(PX, BY[3])), n=11, tracker=t)
        self.assertEqual(r.score, 1.0)

    def test_tags_left_tilted_against_the_core(self):
        tags = layout(**ALL_FOUR)
        tags["tag_l1"] = tag(0.377, BY[0], tilt=8.0)  # leaning on the core at 8 deg: still a standing mark
        tags["tag_l2"] = tag(0.377, BY[1], tilt=12.0)  # 12 deg: not upright
        self.assertEqual(play(tags).score, 0.75)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, tags=None):
        return dict(tags if tags is not None else start_tags())

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None], list(NAMES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        core = [o for o in self.task.OBJECTS if o.name.startswith("core_") and o.name != "core_tray"]
        self.assertEqual(len(core), 7)
        for o in core:  # equal tops, on the tray
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, L.CORE_TOP_Z)
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TRAY_TOP)
        span = sorted((o.pos[1] - o.size[1] / 2, o.pos[1] + o.size[1] / 2) for o in core)
        for (_, hi), (lo, _) in zip(span, span[1:]):
            self.assertAlmostEqual(hi, lo)
        self.assertAlmostEqual(span[0][0], L.CORE_Y_TOP - L.CORE_LEN)
        self.assertAlmostEqual(span[-1][1], L.CORE_Y_TOP)
        # colour changes only at the four boundaries
        by_y = sorted(core, key=lambda o: -o.pos[1])
        changes = [round(a.pos[1] - a.size[1] / 2, 4) for a, b in zip(by_y, by_y[1:]) if a.color != b.color]
        self.assertEqual(changes, [round(y, 4) for y in BY])
        for o in self.task.TAGS:
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP)
            self.assertEqual(o.color, L.TAG_WHITE)  # white, against the black tray; no horizon is near white
        self.assertIn("white tag", self.task.INSTRUCTION)
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        placed = layout(**ALL_FOUR)
        ok = [self.task.success(self.objects(placed)) for _ in range(11)]
        self.assertTrue(ok[-1])
        self.assertEqual(self.task.score(self.objects(placed)), 1.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertEqual(self.task.score(self.objects(placed)), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        span = {a: sum(w.duration for w in plan[a]) for a in plan}
        self.assertAlmostEqual(span["left"], 30.0)
        self.assertAlmostEqual(span["right"], 30.0)
        places = {a: [w.pos for w in plan[a] if w.label == "let go"] for a in plan}
        self.assertEqual([round(p[1], 3) for p in places["left"]], [0.234, 0.107])
        self.assertEqual([round(p[1], 3) for p in places["right"]], [-0.034, -0.154])
        for a in plan:
            for p in places[a]:
                self.assertAlmostEqual(p[0], PX)
                self.assertAlmostEqual(p[2] - 0.025, L.TRAY_TOP + 0.001)  # tag bottom 1 mm above the tray
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every close and open takes 1.5 s
                if w.pos[2] < prev.pos[2]:
                    self.assertGreaterEqual(w.duration, 1.5)  # slow final descents
                if w.label == "let go" or w.label.startswith("grip"):
                    self.assertEqual(prev.label, "settle")
                self.assertEqual(w.yaw, 0.0)
                self.assertLessEqual(w.pos[2], L.TOP + 0.100 + 1e-9)
            grips = [w.finger for w in plan[a] if w.label.startswith("grip")]
            self.assertTrue(all(abs(g + 0.006) < 1e-12 for g in grips))  # 12 mm on the 20 mm tag
        # the two arms' grippers are never closer than 0.26 m in y at the same moment of the plan
        for wl, wr in zip(plan["left"], plan["right"]):
            if abs(wl.pos[0] - wr.pos[0]) < 1e-9:
                self.assertGreater(wl.pos[1] - wr.pos[1], 0.26)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
