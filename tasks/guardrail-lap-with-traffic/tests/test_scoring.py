"""Offline tests for scoring.py (and the hooks in task.py): synthetic panel states and synthetic arm snapshots, no
simulator.  python3 -m unittest discover -s tests -v"""

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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    PANELS,
    Tracker,
    corners,
    downstream_gaps,
    finger_clearance,
    heading_err_deg,
    length_heading_deg,
    outline,
    point_box_distance,
    sample_ys,
    slice_x,
    tilt_deg,
    valid,
    zone_radius,
)

HZ = 20.0
TOP = L.TOP
LISTED = {0.0, 0.25, 0.5, 1.0}
HT = L.PANEL_HALF[0]  # 9 mm: half the thickness
HL = L.PANEL_HALF[1]  # 150 mm: half the length
FACE = L.POST_FACE_X


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def about(axis, deg):
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def panel(x, y, turn=0.0, tilt=0.0, low_z=0.0, lean_axis="y"):
    """A panel as the simulator reports it: centre (x, y), its length turned ``turn`` degrees from the y axis
    (anticlockwise seen from above: its -y end farther from the arms), leaning ``tilt`` degrees about its length
    (or about another body axis), its lowest point ``low_z`` above the table."""
    q = qmul(about("z", turn), about(lean_axis, tilt))
    s = SimpleNamespace(pos=np.array([x, y, 0.0]), quat_xyzw=q)
    s.pos = np.array([x, y, TOP + low_z - corners(s)[:, 2].min()])
    c = corners(s)
    s.lin_vel, s.ang_vel, s.low, s.high = np.zeros(3), np.zeros(3), c.min(axis=0), c.max(axis=0)
    return s


def u_pose(yl, x_left_back, x_back_right_post, **kw):
    """An upstream panel turned anticlockwise, given its back-left corner (x_left_back, yl) (its largest y) and
    where its back face crosses the right post's line (y -0.240)."""
    th = math.atan2(x_back_right_post - x_left_back, yl - L.RIGHT_LINE_Y)
    a = np.array([-math.sin(th), math.cos(th)])  # toward its +y end
    n = np.array([math.cos(th), math.sin(th)])  # toward its back face
    c = np.array([x_left_back, yl]) - HL * a - HT * n
    return panel(c[0], c[1], math.degrees(th), **kw)


def wrong_u(yl, x_front_left, turn=-6.0, **kw):
    """An upstream panel turned clockwise (its right end nearer the arms), given its front-left corner (its largest
    y): tucked behind D's end, lapped against traffic."""
    th = math.radians(turn)
    a = np.array([-math.sin(th), math.cos(th)])
    n = np.array([math.cos(th), math.sin(th)])
    c = np.array([x_front_left, yl]) - HL * a + HT * n
    return panel(c[0], c[1], turn, **kw)


def d_pose(gap=0.0, right_end=-0.025, **kw):
    """A downstream panel along y, its back face ``gap`` in front of the posts' faces, its right end at y."""
    return panel(FACE - gap - HT, right_end + HL, **kw)


def arm(p, opening=1.0):
    p = np.asarray(p, dtype=float)
    return SimpleNamespace(tcp=p, tips=p - np.array([0.0, 0.0, 0.013]), opening=opening)


PARK = {"left": arm((0.20, 0.30, TOP + 0.15)), "right": arm((0.20, -0.30, TOP + 0.15))}
START_L = panel(*L.PANEL_START["panel_left"], turn=90.0)
START_R = panel(*L.PANEL_START["panel_right"], turn=90.0)
D_OK = d_pose()  # flush on the left and middle posts, right end 25 mm past the middle post's centre
U_OK = panel(0.2894, -0.1302, 4.72)  # the demo's place: overlap 45 mm, lap gap 2.7..5.6 mm, 2.5 mm off the post
HOLD_U = {"left": PARK["left"], "right": arm((0.2894, -0.1302, TOP + 0.0335), 0.2)}  # still gripping U
HOLD_D = {"left": arm((0.301, 0.125, TOP + 0.0335), 0.2), "right": PARK["right"]}
PANEL_TOP = TOP + L.PANEL_SIZE[2]
U_NORMAL = np.array([math.cos(math.radians(4.72)), math.sin(math.radians(4.72))])  # U_OK's front-to-back direction


def pair(d, u, swap=False):
    return {"panel_right": d, "panel_left": u} if swap else {"panel_left": d, "panel_right": u}


def run(seq, tr=None):
    """Feed (D, U, arms) triples to a tracker (a fresh one unless given); return the last result and it."""
    tr = Tracker(HZ) if tr is None else tr
    r = None
    for d, u, arms in seq:
        r = tr.update(pair(d, u), arms)
    return r, tr


def settled(d, u, arms=PARK, n=12, tr=None):
    """Both panels held still for n steps (0.6 s at 20 Hz: enough history to be still)."""
    return run([(d, u, arms)] * n, tr)


def score_of(d, u, arms=PARK):
    return settled(d, u, arms)[0].score


def details_of(d, u, arms=PARK):
    return settled(d, u, arms)[0].details


def moved(s, dx=0.0, dy=0.0, dz=0.0):
    out = SimpleNamespace(**vars(s))
    out.pos = s.pos + np.array([dx, dy, dz])
    out.low, out.high = s.low + np.array([dx, dy, dz]), s.high + np.array([dx, dy, dz])
    return out


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        for s, y in ((START_L, 0.40), (START_R, -0.40)):
            self.assertAlmostEqual(tilt_deg(s), 0.0)
            self.assertAlmostEqual(heading_err_deg(s), 90.0)  # lengths along x
            self.assertAlmostEqual(float(s.low[2]), TOP)
            h = outline(s)
            self.assertAlmostEqual(float(h[:, 0].min()), 0.160)
            self.assertAlmostEqual(float(h[:, 0].max()), 0.460)
            self.assertAlmostEqual(float(np.abs(h[:, 1]).min()), 0.391)
            self.assertTrue(Tracker.at_start("panel_left" if y > 0 else "panel_right", s))

    def test_slices_of_a_turned_panel(self):
        h = outline(U_OK)
        front, back = slice_x(h, L.RIGHT_LINE_Y)
        self.assertAlmostEqual(FACE - back, 0.0025, places=4)  # 2.5 mm off the right post
        self.assertAlmostEqual(back - front, L.PANEL_SIZE[0] / math.cos(math.radians(4.72)), places=5)
        self.assertAlmostEqual(float(h[:, 1].max()), 0.020, places=4)  # its back-left corner
        self.assertIsNone(slice_x(h, 0.03))  # beyond its end
        self.assertIsNotNone(slice_x(h, float(h[:, 1].max())))  # at its very end: a point

    def test_heading_is_folded_and_signed(self):
        self.assertAlmostEqual(heading_err_deg(panel(0.3, 0.0, 5.0)), 5.0)
        self.assertAlmostEqual(heading_err_deg(panel(0.3, 0.0, -5.0)), 5.0)
        self.assertAlmostEqual(heading_err_deg(panel(0.3, 0.0, 185.0)), 5.0)  # end for end: the same panel
        self.assertAlmostEqual(length_heading_deg(U_OK), 4.72, places=3)
        self.assertAlmostEqual(length_heading_deg(panel(0.3, 0.0, 180.0 + 4.72)), 4.72, places=3)
        self.assertAlmostEqual(heading_err_deg(panel(0.3, 0.0, tilt=90.0, lean_axis="x")), 90.0)  # on its end

    def test_tilt_either_edge_down(self):
        self.assertAlmostEqual(tilt_deg(panel(0.3, 0.0, tilt=180.0)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(panel(0.3, 0.0, tilt=10.0)), 10.0, places=6)
        self.assertAlmostEqual(tilt_deg(panel(0.3, 0.0, tilt=-10.0)), 10.0, places=6)

    def test_point_box_distance(self):
        self.assertAlmostEqual(point_box_distance((0.2894, -0.1302, TOP + 0.085), U_OK), 0.040, places=6)
        self.assertEqual(point_box_distance((0.2894, -0.1302, TOP + 0.020), U_OK), 0.0)

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), math.hypot(0.0939 / 2 + 0.010, 0.020), places=9)
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), 0.0604, places=4)  # 60.4 mm wide open
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 0.0)), math.hypot(0.010, 0.020), places=9)
        # straight above U: the clearance is the tips' height over its top, whatever the opening
        for op in (0.0, 1.0):
            a = arm((0.2894, -0.1302, PANEL_TOP + 0.013 + 0.025), op)
            self.assertAlmostEqual(finger_clearance(a, {"u": U_OK}), 0.025, places=6)
        # beside U's front face at mid height, wide open: the axis distance less the zone's radius
        c = np.array([0.2894, -0.1302]) - (HT + 0.100) * U_NORMAL
        a = arm((c[0], c[1], TOP + 0.03), 1.0)
        self.assertAlmostEqual(finger_clearance(a, {"u": U_OK}), 0.100 - zone_radius(a), delta=0.0002)
        # a tilted gripper works too (the axis from the tips back to the tool point)
        tilted = SimpleNamespace(tcp=np.array([0.2894, -0.1302, PANEL_TOP + 0.060]),
                                 tips=np.array([0.2894, -0.1302 + 0.009, PANEL_TOP + 0.060 - 0.010]), opening=0.0)
        self.assertGreater(finger_clearance(tilted, {"u": U_OK}), 0.0)

    def test_stretch_readings(self):
        ys = sample_ys(-0.025, 0.020)  # the demo's 45 mm stretch
        self.assertAlmostEqual(ys[0], -0.020)
        self.assertAlmostEqual(ys[-1], 0.015)
        self.assertTrue(all(0 < b - a <= L.SAMPLE_STEP + 1e-12 for a, b in zip(ys, ys[1:])))
        self.assertEqual(sample_ys(-0.025, -0.017), [-0.021])  # under 10 mm: the middle only
        self.assertEqual(len(sample_ys(-0.025, -0.015)), 1)

    def test_downstream_gaps(self):
        self.assertTrue(np.allclose(downstream_gaps(outline(D_OK)), [0.0, 0.0]))
        self.assertTrue(np.allclose(downstream_gaps(outline(d_pose(gap=-0.002))), [-0.002, -0.002]))
        self.assertEqual(downstream_gaps(outline(U_OK))[0], None)  # U does not reach the left post


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = settled(START_L, START_R)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["tier"], "no_downstream_panel")

    def test_the_demo_splice_scores_one(self):
        r, _ = settled(D_OK, U_OK)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        d = r.details
        self.assertEqual((d["D"], d["U"]), ("panel_left", "panel_right"))
        self.assertAlmostEqual(d["overlap_mm"], 45.0, delta=0.1)
        self.assertAlmostEqual(d["mid_mm"], -2.5, delta=0.1)
        self.assertAlmostEqual(d["right_post_gap_mm"], 2.5, delta=0.1)
        self.assertGreaterEqual(d["lap_gap_min_mm"], 2.0)
        self.assertLessEqual(d["lap_gap_max_mm"], 6.0)

    def test_the_offline_pressed_pose_scores_one(self):
        # the kinematic check has no contacts: it carries D 2 mm into the posts (the demo presses it there)
        self.assertEqual(score_of(d_pose(gap=-0.002), U_OK), 1.0)

    def test_every_tier_is_reachable(self):
        seq = [(START_L, START_R, PARK)] * 12  # nothing done
        seq += [(D_OK, START_R, HOLD_D)] * 12  # D set and still held (D in place: standing counts held or not)
        seq += [(D_OK, START_R, PARK)] * 12  # D let go, U untouched
        seq += [(D_OK, U_OK, HOLD_U)] * 12  # U lapped on the road side, still gripped
        seq += [(D_OK, U_OK, PARK)] * 12  # both let go
        tr = Tracker(HZ)
        seen = [tr.update(pair(d, u), a).score for d, u, a in seq]
        self.assertEqual([seen[11], seen[23], seen[35], seen[47], seen[59]], [0.0, 0.25, 0.25, 0.5, 1.0])
        self.assertEqual(set(seen), LISTED)

    def test_the_half_score_example(self):
        # U parallel to D, 7 mm in front of it, 25 mm off the right post: lapped on the road side, not on the post
        u = panel(FACE - 0.018 - 0.007 - HT, -0.130)
        d = details_of(D_OK, u)
        self.assertAlmostEqual(d["lap_gap_min_mm"], 7.0)
        self.assertAlmostEqual(d["right_post_gap_mm"], 25.0)
        self.assertEqual(score_of(D_OK, u), 0.5)

    def test_only_full_score_is_success(self):
        for d, u, arms in ((D_OK, U_OK, HOLD_U), (D_OK, START_R, PARK), (START_L, START_R, PARK)):
            r, _ = settled(d, u, arms)
            self.assertFalse(r.success)

    def test_score_is_idempotent(self):
        r, tr = settled(D_OK, U_OK)
        steps, n = tr.steps, len(tr.history)
        for _ in range(3):
            self.assertEqual(tr.result(pair(D_OK, U_OK), PARK).score, r.score)
        self.assertEqual((tr.steps, len(tr.history)), (steps, n))

    def test_panels_are_interchangeable(self):
        tr = Tracker(HZ)
        for _ in range(12):
            r = tr.update(pair(D_OK, U_OK, swap=True), PARK)
        self.assertEqual(r.score, 1.0)
        self.assertEqual((r.details["D"], r.details["U"]), ("panel_right", "panel_left"))

    def test_released_edge(self):
        # lifted straight up off U, wide open: released once the finger tips are 10 mm over its top
        at = {"left": PARK["left"], "right": arm((0.2894, -0.1302, PANEL_TOP + 0.013 + 0.010))}
        near = {"left": PARK["left"], "right": arm((0.2894, -0.1302, PANEL_TOP + 0.013 + 0.0097))}
        self.assertEqual(score_of(D_OK, U_OK, at), 1.0)
        self.assertEqual(score_of(D_OK, U_OK, near), 0.5)
        left_near = {"left": arm((0.301, 0.125, PANEL_TOP + 0.013 + 0.0097)), "right": PARK["right"]}
        self.assertEqual(score_of(D_OK, U_OK, left_near), 0.5)  # the other gripper counts too
        # beside U's front face at mid height, wide open: 10 mm clear of the zone's 60.4 mm radius
        rr = zone_radius(arm((0, 0, 0), 1.0))
        for off, want in ((rr + 0.0103, 1.0), (rr + 0.0097, 0.5)):
            c = np.array([0.2894, -0.1302]) - (HT + off) * U_NORMAL
            self.assertEqual(score_of(D_OK, U_OK, {"left": PARK["left"], "right": arm((c[0], c[1], TOP + 0.03))}),
                             want, off)

    def test_unreadable_arms_are_not_released(self):
        # 1.0 needs the grippers read: a rigstate failure while a hand still holds a panel must not end the trial
        r, tr = settled(D_OK, U_OK, None)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)
        self.assertFalse(r.details["released"])
        self.assertEqual(tr.arms_unavailable, 12)
        bad_tcp = {"left": arm((math.nan, 0.0, 0.0)), "right": PARK["right"]}
        no_tips = {"left": SimpleNamespace(tcp=PARK["left"].tcp, opening=1.0), "right": PARK["right"]}
        bad_open = {"left": arm(PARK["left"].tcp, math.inf), "right": PARK["right"]}
        for bad in (bad_tcp, no_tips, bad_open, {"left": PARK["left"]}, "arms"):
            self.assertEqual(score_of(D_OK, U_OK, bad), 0.5)


class DownstreamPlace(unittest.TestCase):
    """Score 0 without a panel standing against the left and middle posts, its right end past the middle post."""

    def test_back_edge_window(self):
        for gap, ok in ((0.008, True), (0.0082, False), (-0.003, True), (-0.0032, False)):
            d = details_of(d_pose(gap=gap), START_R)
            self.assertEqual(d["panel_left"]["in_downstream_place"], ok, gap)
            self.assertEqual(score_of(d_pose(gap=gap), START_R), 0.25 if ok else 0.0)

    def test_right_end_window(self):
        # D must reach the middle post's centre (y 0) and run at most 55 mm past it
        for end, ok in ((0.0, True), (0.0003, False), (-0.012, True), (-0.055, True), (-0.0553, False)):
            self.assertEqual(score_of(d_pose(right_end=end), START_R), 0.25 if ok else 0.0, end)

    def test_must_bear_on_both_posts(self):
        # turned about the middle post so it stands 10 mm off the left post: not in the downstream place
        th = math.degrees(math.atan2(0.010, L.POST_SPACING))
        d = panel(FACE - HT / math.cos(math.radians(th)) - 0.125 * math.tan(math.radians(th)), 0.125, th)
        gaps = details_of(d, START_R)["panel_left"]
        self.assertAlmostEqual(gaps["gap_middle_post_mm"], 0.0, delta=0.05)
        self.assertAlmostEqual(gaps["gap_left_post_mm"], 10.0, delta=0.05)
        self.assertEqual(score_of(d, U_OK), 0.0)

    def test_standing_edges(self):
        self.assertEqual(score_of(d_pose(tilt=7.8, gap=0.004), START_R), 0.25)
        self.assertEqual(score_of(d_pose(tilt=8.2, gap=0.004), START_R), 0.0)
        self.assertEqual(score_of(d_pose(low_z=0.003), START_R), 0.25)
        self.assertEqual(score_of(d_pose(low_z=0.0032), START_R), 0.0)  # held up off the table
        self.assertEqual(score_of(d_pose(tilt=90.0), START_R), 0.0)  # lying on a broad face

    def test_still_needs_half_a_second(self):
        r, _ = settled(D_OK, U_OK, n=10)
        self.assertEqual(r.score, 0.0)
        r, _ = settled(D_OK, U_OK, n=11)
        self.assertEqual(r.score, 1.0)

    def test_a_moving_panel_is_not_standing(self):
        seq = [(moved(D_OK, dy=0.00025 * k), U_OK, PARK) for k in range(12)]  # 0.25 mm a step: 2.5 mm in 0.5 s
        self.assertEqual(run(seq)[0].score, 0.0)
        seq = [(moved(D_OK, dy=0.00015 * k), U_OK, PARK) for k in range(12)]  # 1.5 mm in 0.5 s: still
        self.assertEqual(run(seq)[0].score, 1.0)

    def test_jitter_within_2_mm_is_still(self):
        seq = [(moved(D_OK, dx=0.0009 * (-1) ** k), U_OK, PARK) for k in range(12)]
        self.assertEqual(run(seq)[0].score, 1.0)

    def test_a_turn_in_place_is_not_still(self):
        seq = [(panel(0.2894, -0.1302, 4.72 + 0.25 * (k - 11)), D_OK, PARK) for k in range(12)]
        r, _ = run([(d, u, a) for u, d, a in seq])  # U turning 0.25 deg a step: 2.5 deg in 0.5 s
        self.assertEqual(r.score, 0.25)
        seq = [(D_OK, panel(0.2894, -0.1302, 4.72 + 0.15 * (k - 11)), PARK) for k in range(12)]
        self.assertEqual(run(seq)[0].score, 1.0)


class Lap(unittest.TestCase):
    """Lapped on the road side (0.5 / 1.0), lapped the wrong way (0), or neither (0.25)."""

    D2 = d_pose(right_end=-0.045)  # D reaching 45 mm past the middle post's centre: room to tuck U behind it

    def test_wrong_lap_scores_zero(self):
        u = wrong_u(-0.014, 0.3125)
        d = details_of(self.D2, u)
        self.assertTrue(d["wrong_lap"])
        self.assertEqual(d["tier"], "wrong_lap")
        self.assertAlmostEqual(d["overlap_mm"], 31.0, delta=0.05)
        self.assertEqual(score_of(self.D2, u), 0.0)
        # the right part of a tucked panel still stands in front of the right post, as it would have to
        self.assertLessEqual(d["right_post_gap_mm"], 12.0)

    def test_wrong_lap_gap_edges(self):
        tan6 = math.tan(math.radians(6.0))
        top = FACE + 0.015 + 0.005 * tan6  # the reading nearest U's end exactly 15 mm behind D's back edge
        self.assertEqual(score_of(self.D2, wrong_u(-0.014, top)), 0.0)
        self.assertEqual(score_of(self.D2, wrong_u(-0.014, top + 0.0003)), 0.25)
        low = FACE - 0.001 + 0.026 * tan6  # the reading nearest D's end exactly 1 mm into D (contact)
        self.assertEqual(score_of(self.D2, wrong_u(-0.014, low)), 0.0)
        self.assertEqual(score_of(self.D2, wrong_u(-0.014, low - 0.0003)), 0.25)

    def test_wrong_lap_reach_edge(self):
        # U's end tucked behind D's: up to 2 mm short of D's end still counts (contact), more is short of the splice
        self.assertEqual(score_of(self.D2, wrong_u(-0.040, 0.3125)), 0.0)  # overlap 5 mm: one paired reading
        self.assertEqual(score_of(self.D2, wrong_u(-0.0405, 0.3125)), 0.0)  # 4.5 mm
        self.assertEqual(score_of(self.D2, wrong_u(-0.045, 0.3125)), 0.0)  # end to end
        self.assertEqual(score_of(self.D2, wrong_u(-0.047, 0.3125)), 0.0)  # 2 mm short
        d = details_of(self.D2, wrong_u(-0.0473, 0.3125))
        self.assertEqual((d["tier"], d["wrong_reach_y_mm"]), ("downstream_only", -47.0))
        self.assertEqual(score_of(self.D2, wrong_u(-0.0473, 0.3125)), 0.25)  # 2.3 mm short

    def test_short_wrong_lap_gap_edges(self):
        # one paired reading: U's front edge 5 mm inside its end against D's back edge 5 mm inside D's end
        tan6 = math.tan(math.radians(6.0))
        top = FACE + 0.015 + 0.005 * tan6
        self.assertEqual(score_of(self.D2, wrong_u(-0.043, top)), 0.0)
        self.assertEqual(score_of(self.D2, wrong_u(-0.043, top + 0.0003)), 0.25)
        low = FACE - 0.001 + 0.005 * tan6
        self.assertEqual(score_of(self.D2, wrong_u(-0.043, low)), 0.0)
        self.assertEqual(score_of(self.D2, wrong_u(-0.043, low - 0.0003)), 0.25)

    def test_wrong_lap_against_the_middle_post(self):
        # D's end short of the middle post's right side: U's end can only tuck in behind D's line against the
        # post's side (y -12.5 mm); that is the wrong-way joint, 0; 2 mm farther right it is short of the splice
        for dr in (0.0, -0.005, -0.010, -0.012):
            self.assertEqual(score_of(d_pose(right_end=dr), wrong_u(-0.0126, FACE + 0.0005)), 0.0, dr)
            self.assertEqual(score_of(d_pose(right_end=dr), wrong_u(-0.0145, FACE + 0.0005)), 0.0, dr)
            self.assertEqual(score_of(d_pose(right_end=dr), wrong_u(-0.0148, FACE + 0.0005)), 0.25, dr)
        # such a U is physically possible: its end right of the middle post's side (y -12.5 mm), its front-left
        # corner behind D's back face, and its back edge in front of the right post's face all across that post
        h = outline(wrong_u(-0.0126, FACE + 0.0005))
        corner = h[int(np.argmax(h[:, 1]))]
        self.assertLess(float(corner[1]), -0.0125)
        self.assertGreater(float(corner[0]), FACE)
        for y in (-0.2275, -0.240, -0.2525):
            self.assertLess(slice_x(h, y)[1], FACE)

    def test_wrong_lap_counts_held_or_leaning_but_not_lifted(self):
        self.assertEqual(score_of(self.D2, wrong_u(-0.014, 0.3125, low_z=0.0048)), 0.0)
        self.assertEqual(score_of(self.D2, wrong_u(-0.014, 0.3125, low_z=0.0052)), 0.25)
        # leaning 12 degrees forward onto D's back (its outline's front edge then 4.5 mm farther forward)
        lean = wrong_u(-0.014, 0.3125 + 0.0045, tilt=-12.0)
        self.assertTrue(details_of(self.D2, lean)["wrong_lap"])
        self.assertEqual(score_of(self.D2, lean), 0.0)

    def test_lap_gap_edges(self):
        for gap, want in ((0.015, 0.5), (0.0153, 0.25), (-0.001, 0.5), (-0.0013, 0.25)):
            u = panel(FACE - 0.018 - gap - HT, -0.130)  # parallel to D, gap in front of it
            self.assertEqual(score_of(D_OK, u), want, gap)

    def test_lap_overlap_edges(self):
        x = FACE - 0.018 - 0.005 - HT  # parallel, 5 mm in front
        for left_end, want in ((-0.015, 0.5), (-0.0153, 0.25), (0.075, 0.5), (0.0753, 0.25)):
            self.assertEqual(score_of(D_OK, panel(x, left_end - HL)), want, left_end)

    def test_lap_turn_edge(self):
        # a short (12 mm) lap with U turned steeply: within 15 degrees it counts as lapped, past it not
        for turn, want in ((14.5, 0.5), (15.5, 0.25)):
            t = math.radians(turn)
            a, n = np.array([-math.sin(t), math.cos(t)]), np.array([math.cos(t), math.sin(t)])
            c = np.array([0.287, -0.013]) - HL * a - HT * n
            self.assertEqual(score_of(D_OK, panel(c[0], c[1], turn)), want, turn)

    def test_lapped_needs_U_standing(self):
        self.assertEqual(score_of(D_OK, panel(0.2894, -0.1302, 4.72, tilt=9.0)), 0.25)  # leaning on D
        self.assertEqual(score_of(D_OK, panel(0.2894, -0.1302, 4.72, low_z=0.004), HOLD_U), 0.25)  # held up
        self.assertEqual(score_of(D_OK, panel(0.2894, -0.1302, 4.72, low_z=0.003), HOLD_U), 0.5)


class FullSplice(unittest.TestCase):
    """1.0: lapped on the road side, overlap 30-60 mm, centred within 15 mm, on the right post, both let go."""

    def test_overlap_edges(self):
        for yl, want in ((0.005, 1.0), (0.0047, 0.5), (0.035, 1.0), (0.0353, 0.5)):
            self.assertEqual(score_of(D_OK, u_pose(yl, 0.286, 0.3055)), want, yl)

    def test_centre_edge(self):
        self.assertEqual(score_of(d_pose(right_end=-0.035), u_pose(0.005, 0.286, 0.3055)), 1.0)  # middle -15 mm
        self.assertEqual(score_of(d_pose(right_end=-0.0353), u_pose(0.0047, 0.286, 0.3055)), 0.5)  # -15.3 mm
        d = details_of(d_pose(right_end=-0.0353), u_pose(0.0047, 0.286, 0.3055))
        self.assertAlmostEqual(d["overlap_mm"], 40.0, delta=0.01)

    def test_right_post_edges(self):
        for xb, want in ((FACE - 0.012, 1.0), (FACE - 0.0123, 0.5), (FACE + 0.001, 1.0), (FACE + 0.0013, 0.5)):
            self.assertEqual(score_of(D_OK, u_pose(0.020, 0.286, xb)), want, xb)

    def test_not_let_go_is_half(self):
        self.assertEqual(score_of(D_OK, U_OK, HOLD_U), 0.5)
        self.assertEqual(score_of(D_OK, U_OK, HOLD_D), 0.5)

    def test_centred_splice_with_D_short_of_the_post_side(self):
        # review finding: a splice right in every other way, D's end 0-12 mm right of the middle post's centre
        # (not past its right side): 1.0 (it was 0, a rule the instruction did not state)
        self.assertEqual(score_of(d_pose(right_end=-0.005), u_pose(0.025, FACE - 0.022, FACE - 0.003)), 1.0)
        for ov, mid in ((0.030, 0.010), (0.040, 0.010), (0.045, 0.012), (0.030, 0.015), (0.030, 0.003),
                        (0.030, 0.000), (0.060, -0.015)):  # D's end from y 0 to y -45 mm
            d, u = d_pose(right_end=mid - ov / 2), u_pose(mid + ov / 2, FACE - 0.022, FACE - 0.003)
            self.assertEqual(score_of(d, u), 1.0, (ov, mid))


class Exploits(unittest.TestCase):
    """Unintended ways to a score, and what they earn."""

    def test_tucked_behind_the_wrong_way(self):
        # the classic error: the upstream panel's end behind the downstream one's, facing oncoming traffic
        self.assertEqual(score_of(d_pose(right_end=-0.045), wrong_u(-0.014, 0.3125)), 0.0)

    def test_upstream_panel_first_across_the_middle_post(self):
        u_first = panel(FACE - HT, -0.110)  # flat against the right and middle posts
        d_after = panel(FACE - 0.018 - HT, 0.170)  # then the other in front of it at the splice
        self.assertEqual(score_of(d_after, u_first), 0.0)

    def test_one_panel_against_the_right_posts_only(self):
        self.assertEqual(score_of(START_L, panel(FACE - HT, -0.110)), 0.0)

    def test_holding_instead_of_placing(self):
        self.assertEqual(score_of(D_OK, U_OK, HOLD_U), 0.5)
        both = {"left": HOLD_D["left"], "right": HOLD_U["right"]}
        self.assertEqual(score_of(D_OK, U_OK, both), 0.5)
        self.assertEqual(score_of(D_OK, panel(0.2894, -0.1302, 4.72, low_z=0.02), HOLD_U), 0.25)  # in the air

    def test_U_lying_flat_in_front_of_the_splice(self):
        self.assertEqual(score_of(D_OK, panel(0.270, -0.130, tilt=90.0)), 0.25)

    def test_U_stood_on_top_of_D(self):
        self.assertEqual(score_of(D_OK, panel(0.301, -0.010, low_z=L.PANEL_SIZE[2])), 0.25)

    def test_butt_joint_and_short_of_the_splice(self):
        self.assertEqual(score_of(D_OK, panel(FACE - HT, -0.025 - HL - 0.001)), 0.25)  # end to end
        self.assertEqual(score_of(D_OK, panel(0.276, -0.190)), 0.25)  # 15 mm short of D's end

    def test_U_out_in_the_lane(self):
        self.assertEqual(score_of(D_OK, panel(FACE - 0.018 - 0.020 - HT, -0.130)), 0.25)  # 20 mm in front

    def test_doubled_over_D(self):
        self.assertEqual(score_of(D_OK, panel(0.276, 0.125)), 0.25)  # wholly in front of D
        self.assertEqual(score_of(D_OK, panel(0.276, 0.200)), 0.25)  # lapped at D's other end, off the posts

    def test_D_short_of_the_middle_post(self):
        # D must reach the middle post's centre: one ending 5 mm short of it bears on the post's left half only
        self.assertEqual(score_of(d_pose(right_end=0.005), u_pose(0.040, 0.286, 0.3055)), 0.0)
        self.assertEqual(score_of(d_pose(right_end=0.0), u_pose(0.040, 0.286, 0.3055)), 0.5)  # 40 mm, middle +20

    def test_wrong_way_end_jammed_on_the_post_side(self):
        # review finding: D's end 0-5 mm past the middle post's side, U's end jammed on that side behind D's back
        # face, turned clear of the right post (overlap -0.6 to 4.9 mm): the wrong-way joint, 0 (it was 0.25)
        for dr in (-0.012, -0.015, -0.0165, -0.0175, -0.018):
            self.assertEqual(score_of(d_pose(right_end=dr), wrong_u(-0.0126, FACE + 0.0005, turn=-6.0)), 0.0, dr)

    def test_open_finger_left_on_U(self):
        # review finding: the right gripper open, its tool point 45 mm in front of U's face at mid height, so its far
        # finger touches U (the tool point alone looked 45 mm clear): not released, 0.5 (it was 1.0)
        c = np.array([0.2894, -0.1302]) - 0.054 * U_NORMAL
        arms = {"left": PARK["left"], "right": arm((c[0], c[1], TOP + 0.03), 1.0)}
        self.assertEqual(score_of(D_OK, U_OK, arms), 0.5)
        self.assertFalse(details_of(D_OK, U_OK, arms)["released"])

    def test_opened_but_not_lifted(self):
        # the fingers opened round U at the release height: not released, 0.5
        self.assertEqual(score_of(D_OK, U_OK, {"left": PARK["left"], "right": arm((0.2894, -0.1302, TOP + 0.0335))}),
                         0.5)

    def test_D_knocked_over_after_the_splice(self):
        r, tr = settled(D_OK, U_OK)
        self.assertEqual(r.score, 1.0)
        r, _ = settled(panel(0.280, 0.125, tilt=90.0), U_OK, tr=tr)
        self.assertEqual(r.score, 0.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        _, tr = settled(D_OK, U_OK)
        steps, n, prev = tr.steps, len(tr.history), tr.prev
        bad = [
            SimpleNamespace(pos=np.array([math.nan, 0.0, 0.8]), quat_xyzw=D_OK.quat_xyzw),
            SimpleNamespace(pos=D_OK.pos, quat_xyzw=np.array([0.0, 0.0, math.inf, 1.0])),
            SimpleNamespace(pos=D_OK.pos, quat_xyzw=np.zeros(4)),
            SimpleNamespace(pos=D_OK.pos),
            SimpleNamespace(pos=np.zeros(2), quat_xyzw=D_OK.quat_xyzw),
            None,
        ]
        for b in bad:
            self.assertFalse(valid(b))
            r = tr.update(pair(b, U_OK), PARK)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
        self.assertEqual(tr.update({"panel_left": D_OK}, PARK).score, 0.0)  # a panel missing
        self.assertEqual(tr.update(None, PARK).score, 0.0)
        self.assertEqual(len(tr.history), n)
        self.assertIs(tr.prev, prev)
        self.assertEqual(tr.steps, steps + len(bad) + 2)
        self.assertEqual(tr.result(pair(D_OK, U_OK), PARK).score, 1.0)  # its record is intact

    def test_unnormalised_quaternion(self):
        u = SimpleNamespace(**vars(U_OK))
        u.quat_xyzw = U_OK.quat_xyzw * 3.0
        self.assertEqual(score_of(D_OK, u), 1.0)

    def test_random_states_score_a_listed_tier(self):
        rng = random.Random(7)
        tr = Tracker(HZ)
        for _ in range(400):
            st = []
            for _ in range(2):
                q = np.array([rng.gauss(0, 1) for _ in range(4)])
                if rng.random() < 0.5:  # mostly upright, near the posts
                    q = about("z", rng.uniform(-30, 30) + rng.choice((0.0, 90.0, 180.0)))
                st.append(SimpleNamespace(pos=np.array([rng.uniform(0.2, 0.35), rng.uniform(-0.3, 0.3),
                                                        TOP + rng.uniform(0.0, 0.06)]), quat_xyzw=q))
            arms = rng.choice((PARK, HOLD_U, None))
            r = tr.update(pair(*st), arms)
            self.assertIn(r.score, LISTED)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, tr = settled(D_OK, U_OK)
        tr.update(pair(START_L, START_R), PARK)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.history), 1)

    def test_jittered_start_resets(self):
        _, tr = settled(D_OK, U_OK)
        tr.update(pair(panel(0.33, 0.38, 93.0), panel(0.29, -0.42, 87.0)), PARK)
        self.assertEqual(tr.steps, 1)

    def test_one_panel_put_back_while_the_other_never_moved(self):
        _, tr = settled(D_OK, START_R)
        tr.update(pair(START_L, START_R), PARK)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        _, tr = settled(D_OK, START_R)
        a, b = D_OK.pos, START_L.pos
        for k in range(1, 31):
            f = k / 30  # under 1 cm a step
            s = panel(*(a[:2] + f * (b[:2] - a[:2])), turn=90.0 * f)
            tr.update(pair(s, START_R), PARK)
        self.assertEqual(tr.steps, 12 + 30)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, tr = settled(D_OK, U_OK)
        tr.update(pair(panel(0.20, 0.30, 90.0), U_OK), PARK)  # 25 cm in one step, but not onto its start
        self.assertEqual(tr.steps, 13)

    def test_explicit_reset(self):
        _, tr = settled(D_OK, U_OK)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.arms_unavailable), (0, 0, 0))
        self.assertIsNone(tr.prev)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.tf = importlib.import_module("yam_arena.task_file")
        cls.meta = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()

    def setUp(self):
        self.task._TRACKER.reset()
        rigstate._STATE["arms"] = None

    def obs(self, d, u):
        return {n: self.tf.ObjectState(s.pos, s.quat_xyzw, s.lin_vel, s.ang_vel, s.low, s.high)
                for n, s in pair(d, u).items()}

    def feed(self, arms):
        rigstate.offline_feed({a: {"tcp": arms[a].tcp, "tips": arms[a].tips, "opening": arms[a].opening}
                               for a in ("left", "right")})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], list(PANELS))
        for o in loose:
            self.assertLessEqual(o.mass, 0.3)
            self.assertEqual(tuple(o.size), L.PANEL_SIZE)
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, TOP)
            self.assertAlmostEqual(o.yaw, math.pi / 2)
        for p in t.POST_OBJECTS:
            self.assertAlmostEqual(p.pos[0] - p.size[0] / 2, FACE)
            self.assertEqual(p.mass, None)
        # the arrows point +y: each head's two strokes meet at the tip, 20 mm past the arrow's centre
        for i, yc in enumerate(L.ARROW_YS):
            a, b = (next(o for o in t.OBJECTS if o.name == f"arrow{i}_head_{k}") for k in "ab")
            self.assertAlmostEqual(a.pos[1], b.pos[1])
            self.assertGreater(a.pos[1], yc)
            self.assertAlmostEqual(a.yaw, -b.yaw)
            corner = np.array(a.pos[:2]) + np.array([L.HEAD_DX, L.HEAD_DY])
            self.assertTrue(np.allclose(corner, [L.ARROW_X, yc + L.ARROW_TIP]))

    def test_hooks(self):
        t = self.task
        self.assertFalse(t.success(self.obs(START_L, START_R)))
        self.assertEqual(t.score(self.obs(START_L, START_R)), 0.0)
        t._TRACKER.reset()
        self.feed(PARK)
        ok = [t.success(self.obs(D_OK, U_OK)) for _ in range(12)]
        self.assertFalse(ok[0])
        self.assertTrue(ok[-1])
        self.assertEqual(t.score(self.obs(D_OK, U_OK)), 1.0)
        self.feed(HOLD_U)
        self.assertEqual(t.score(self.obs(D_OK, U_OK)), 0.5)
        self.assertEqual(t.score({"panel_left": None}), 0.0)

    def test_rigstate_reset_starts_a_new_trial(self):
        t = self.task
        self.feed(PARK)
        for _ in range(12):
            t.success(self.obs(D_OK, U_OK))
        self.assertEqual(t._TRACKER.steps, 12)
        rigstate.offline_reset()
        self.assertEqual(t._TRACKER.steps, 0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.feed(PARK)
                for _ in range(12):
                    self.task.success(self.obs(D_OK, U_OK))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "tier", "D", "U", "overlap_mm", "mid_mm",
                          "lap_gap_min_mm", "lap_gap_max_mm", "wrong_gap_min_mm", "wrong_reach_y_mm",
                          "right_post_gap_mm", "u_turn_deg", "released", "left_finger_clear_mm",
                          "right_finger_clear_mm", "panel_left", "panel_right_pos", "left_tcp", "left_opening",
                          "arms_unavailable"):
                    self.assertIn(k, row)
                for k in ("standing", "still", "moved_mm", "tilt_deg", "low_mm", "gap_left_post_mm",
                          "gap_middle_post_mm", "right_end_y_mm", "in_downstream_place"):
                    self.assertIn(k, row["panel_left"])
                self.assertTrue(row["success"])
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.task.success(self.obs(D_OK, U_OK))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm_name, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes at least 1.5 s, standing still
                    self.assertGreaterEqual(w.duration, 1.5)
                    self.assertEqual(tuple(prev.pos), tuple(w.pos))
                if w.pos[2] < prev.pos[2] and w.pos[2] - TOP < 0.07:  # slow final descents
                    self.assertGreaterEqual(w.duration, 1.5)
            grip = [w for w in wps if w.label.startswith("grip it")][0]
            self.assertAlmostEqual(grip.finger, -(0.018 - 0.008) / 2)
            self.assertEqual(grip.yaw, 0.0)  # fingers along y across the panel's thickness at its start
        left, right = plan["left"], plan["right"]
        place_d = [w for w in left if w.label == "let go"][0]
        place_u = [w for w in right if w.label == "let go"][0]
        self.assertAlmostEqual(place_d.pos[0] + HT - FACE, 0.002)  # pressed 2 mm into the posts
        self.assertAlmostEqual(place_d.pos[1] - HL, -0.025)  # right end 25 mm past the middle post's centre
        self.assertAlmostEqual(place_d.yaw, -math.pi / 2)
        self.assertAlmostEqual(place_u.yaw - math.pi / 2, math.radians(4.72), places=3)
        self.assertAlmostEqual(place_u.pos[2] - TOP, 0.0335)
        # review finding: each panel moves sideways into place with its bottom 8 mm up (about 5 mm with the arm's
        # 3 mm sag), then comes straight down over 1.5 s, so it never drags on the table while moving sideways
        for wps in (left, right):
            i = [w.label for w in wps].index("slowly straight down")
            side, down = wps[i - 1], wps[i]
            self.assertAlmostEqual(side.pos[2] - TOP - 0.032, 0.008)
            self.assertEqual(tuple(side.pos[:2]), tuple(down.pos[:2]))
            self.assertGreaterEqual(down.duration, 1.5)
            self.assertAlmostEqual(down.pos[2] - TOP - 0.032, 0.0015)
        # the right arm comes down to the splice only once the left hand has let go and lifted
        t_left_up = sum(w.duration for w in left[:-1])
        t_right_down = sum(w.duration for w in right[:[w.label for w in right].index("down in front of D's end")])
        self.assertAlmostEqual(t_left_up, t_right_down)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_is_the_same_everywhere(self):
        ins = self.task.INSTRUCTION
        self.assertEqual(self.meta["instruction"], ins)
        commands = re.findall(r'--instruction "([^"]*)"', self.readme)
        self.assertGreaterEqual(len(commands), 2)
        for c in commands:
            self.assertEqual(c, ins)
        for s in ("middle post", "3 to 6 cm", "direction of traffic", "white arrows", "road side", "let go",
                  "lift both grippers clear"):
            self.assertIn(s, ins)

    def test_task_json_lists_the_scores(self):
        self.assertIn("Possible scores: 0, 0.25, 0.5, 1.0.", self.meta["scoring"])
        self.assertIn("demo: 1.0", self.meta["scores"])
        self.assertNotIn("media", self.meta)
        self.assertEqual(self.meta["run_limit"], "2,400 steps (2 minutes at 20 Hz)")


if __name__ == "__main__":
    unittest.main()
