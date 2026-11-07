"""Offline tests for scoring.py (and the hooks in task.py): synthetic flag states and synthetic arm snapshots fed
through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
    Tracker, base_centre, held, in_target, is_background, item_at, on_ground, path_length, reach, tilt_deg,
    tip_offset, upright, valid,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad-tip point below the TCP in a straight-down grip
NAMES = L.FLAG_NAMES
CORE = {s: tuple(L.STAINS[s]["core"][:2]) for s in L.STAIN_NAMES}  # core centres
BG = L.BACKGROUND_POINT
READY_TIPS = {"left": (0.323, 0.305, L.TOP + 0.25), "right": (0.323, -0.305, L.TOP + 0.25)}


def about_x(deg):
    h = math.radians(deg) / 2
    return np.array([math.sin(h), 0.0, 0.0, math.cos(h)])


def surface_top(x, y):
    """Height of the top surface under a point: core or streak 3 mm, halo 2 mm, sheet 1 mm, bare table 0."""
    for parts in L.STAINS.values():
        if any(L.rect_distance(x, y, r) == 0.0 for r in (parts["core"], parts["streak"])):
            return L.CORE_TOP
    for parts in L.STAINS.values():
        if any(L.rect_distance(x, y, r) == 0.0 for r in parts["halo"]):
            return L.HALO_TOP
    return L.SHEET_TOP if L.rect_distance(x, y, L.SHEET) == 0.0 else L.TOP


def flag(x, y, bottom=None, tilt=0.0, vel=(0.0, 0.0, 0.0), quat=None):
    """An ObjectState-like flag whose lower end is centred at (x, y), its lowest point at ``bottom`` (default: the
    surface under it), tilted ``tilt`` degrees about x, with its world-aligned bounding box as the simulator
    reports it."""
    q = about_x(tilt) if quat is None else np.array(quat, float)
    n = np.linalg.norm(q) or 1.0
    x_, y_, z_, w_ = q / n
    u = np.array([2 * (x_ * z_ + y_ * w_), 2 * (y_ * z_ - x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)])
    up = u if u[2] >= 0 else -u
    ext = np.abs(u) * L.FLAG_HEIGHT / 2 + L.FLAG_RADIUS * np.sqrt(np.clip(1 - u * u, 0, 1))
    bottom = surface_top(x, y) if bottom is None else bottom
    c = np.array([x + up[0] * L.FLAG_HEIGHT / 2, y + up[1] * L.FLAG_HEIGHT / 2, bottom + ext[2]])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c - ext, high=c + ext)


def start_flags():
    return {name: flag(x, y) for name, x, y in L.FLAG_STARTS}


def layout(**moved):
    """The start layout with some flags moved: layout(flag_l1=flag(...), ...)."""
    out = start_flags()
    out.update(moved)
    return out


ALL_FOUR = {"flag_l2": flag(*CORE["S1"]), "flag_l1": flag(*BG), "flag_r2": flag(*CORE["S2"]),
            "flag_r1": flag(*CORE["S3"])}
OPEN, ON_FLAG, SHUT = 1.0, 0.025 / 0.0939, 0.0  # rigstate openings: fully open, closed on a 25 mm flag, fully closed


def snap(left=None, right=None, left_open=OPEN, right_open=OPEN):
    """Feed one synthetic arm snapshot (pad-tip points and openings) through rigstate, as the offline checker
    does."""
    feed = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        tips = np.array(READY_TIPS[side] if tips is None else tips, float)
        feed[side] = {"tcp": tips + np.array([0.0, 0.0, TIPS]), "tips": tips, "opening": op}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def play(flags, n=12, tracker=None, arms="ready"):
    """Hold one layout for n control steps (the arms parked high unless given); return the last Result."""
    t = tracker or Tracker()
    a = snap() if arms == "ready" else arms
    r = None
    for _ in range(n):
        r = t.update(flags, a)
    return r


def decoys(stain):
    """The points a policy might aim at instead of the darkest spot: the main lobe's centre, the streak's centre,
    the halo's area centroid and its bounding-box centre."""
    lobes = L.STAINS[stain]["halo"]
    areas = [r[2] * r[3] for r in lobes]
    cx = sum(a * r[0] for a, r in zip(areas, lobes)) / sum(areas)
    cy = sum(a * r[1] for a, r in zip(areas, lobes)) / sum(areas)
    b = [L.rect_bounds(r) for r in lobes]
    bbox = ((min(v[0] for v in b) + max(v[1] for v in b)) / 2, (min(v[2] for v in b) + max(v[3] for v in b)) / 2)
    return {"main_lobe": lobes[0][:2], "streak": L.STAINS[stain]["streak"][:2], "centroid": (cx, cy), "bbox": bbox}


def target_rect(stain):
    x0, x1, y0, y1 = L.target(stain)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def luminance(rgb):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def background_grid(step=0.001):
    """Every point of a 1 mm grid over the sheet that counts as background."""
    x0, x1, y0, y1 = L.rect_bounds(L.SHEET)
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    return np.array([(x, y) for x in xs for y in ys if is_background(x, y)])


class Geometry(unittest.TestCase):
    def test_sheets_stack_and_abut(self):
        self.assertEqual(tuple(round(v, 3) for v in L.rect_bounds(L.SHEET)), (0.08, 0.46, -0.31, 0.31))
        for s, parts in L.STAINS.items():
            a, b = (L.rect_bounds(r) for r in parts["halo"])
            # the two lobes touch along an edge and do not overlap
            ox = min(a[1], b[1]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[2], b[2])
            self.assertAlmostEqual(min(ox, oy), 0.0, places=9, msg=s)
            self.assertGreater(max(ox, oy), 0.0, s)
            main = a
            for part in ("streak", "core"):
                p = L.rect_bounds(parts[part])
                self.assertTrue(main[0] - 1e-9 <= p[0] and p[1] <= main[1] + 1e-9
                                and main[2] - 1e-9 <= p[2] and p[3] <= main[3] + 1e-9, (s, part))
            st, co = L.rect_bounds(parts["streak"]), L.rect_bounds(parts["core"])
            ox = min(st[1], co[1]) - max(st[0], co[0])
            oy = min(st[3], co[3]) - max(st[2], co[2])
            self.assertAlmostEqual(min(ox, oy), 0.0, places=9, msg=s)  # streak and core abut
            self.assertGreater(max(ox, oy), 0.0, s)
            self.assertAlmostEqual(parts["core"][2], 0.050)
            self.assertAlmostEqual(parts["core"][3], 0.050)
            x0, x1, y0, y1 = L.target(s)
            self.assertAlmostEqual(x1 - x0, 0.040)
            self.assertAlmostEqual(y1 - y0, 0.040)
            for lobe in parts["halo"]:  # every halo lies on the sheet
                self.assertEqual(L.rect_distance(lobe[0], lobe[1], L.SHEET), 0.0)

    def test_the_core_is_the_darkest_tone(self):
        lum = [luminance(c) for c in (L.CORE_BLACK, L.STREAK_BROWN, L.HALO_TAN, L.SHEET_GREY)]
        self.assertEqual(lum, sorted(lum))
        self.assertLess(lum[0] * 8, lum[1])  # the core is about 9 times darker than the streak beside it
        self.assertLess(lum[1] * 3, lum[2])  # the streak about 3 times darker than the halo
        self.assertLess(lum[2] * 1.5, lum[3])  # the halo about 1.6 times darker than the sheet
        for s, parts in L.STAINS.items():  # the streak is larger than the core and nearer the stain's middle
            self.assertGreater(parts["streak"][2] * parts["streak"][3], parts["core"][2] * parts["core"][3])

    def test_decoys_miss_the_targets(self):
        for s in L.STAIN_NAMES:
            d = decoys(s)
            dist = {k: L.rect_distance(*p, target_rect(s)) for k, p in d.items()}
            self.assertGreaterEqual(dist["main_lobe"], 0.025 - 1e-9, s)
            self.assertGreaterEqual(dist["streak"], 0.040 - 1e-9, s)
            self.assertTrue(0.040 <= dist["centroid"] <= 0.048, (s, dist["centroid"]))
            self.assertTrue(0.050 - 1e-9 <= dist["bbox"] <= 0.060 + 1e-9, (s, dist["bbox"]))

    def test_stain_gaps(self):
        def gap(a, b):
            return min(math.hypot(max(p[0] - q[1], q[0] - p[1], 0.0), max(p[2] - q[3], q[2] - p[3], 0.0))
                       for p in map(L.rect_bounds, L.STAINS[a]["halo"]) for q in map(L.rect_bounds, L.STAINS[b]["halo"]))
        self.assertAlmostEqual(gap("S1", "S2"), 0.045)
        self.assertAlmostEqual(gap("S2", "S3"), 0.075)
        self.assertGreater(gap("S1", "S3"), 0.30)

    def test_the_background_pocket(self):
        pts = background_grid()
        self.assertGreater(len(pts), 3000)
        # one pocket by the sheet's near edge in the middle: x 0.095..0.155, y -0.084..+0.046
        self.assertAlmostEqual(pts[:, 0].min(), 0.095, places=3)
        self.assertLessEqual(pts[:, 0].max(), 0.155 + 1e-9)
        self.assertGreaterEqual(pts[:, 1].min(), -0.084 - 1e-9)
        self.assertLessEqual(pts[:, 1].max(), 0.0465)
        col = pts[np.abs(pts[:, 0] - 0.115) < 1e-6]
        self.assertAlmostEqual(col[:, 1].min(), -0.063, delta=0.0015)
        self.assertAlmostEqual(col[:, 1].max(), 0.034, delta=0.0015)
        # the demo's background point: 177 mm from S1, 190 mm from S2, 208 mm from S3; 20 mm inside the pocket
        self.assertEqual([round(L.stain_distance(*BG, s) * 1000) for s in L.STAIN_NAMES], [177, 190, 208])
        self.assertTrue(is_background(*BG))
        self.assertAlmostEqual(BG[0] - (L.rect_bounds(L.SHEET)[0] + L.SHEET_INSET), 0.020)
        for dx, dy in ((-0.0199, 0.0), (0.0, 0.0279), (0.0, -0.0659), (0.0399, 0.0)):  # 20 mm or more inside
            self.assertTrue(is_background(BG[0] + dx, BG[1] + dy), (dx, dy))

    def test_no_point_between_two_stains_is_background(self):
        pts = background_grid()
        # nothing between the stains, beyond them or at the far side of the sheet
        self.assertTrue(np.all(pts[:, 0] < 0.16))
        for a, b in (((0.42, 0.105), (0.42, -0.105)), ((0.30, 0.10), (0.30, -0.13))):
            for f in np.linspace(0, 1, 21):
                p = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
                self.assertFalse(is_background(*p), p)

    def test_flag_starts_are_off_the_sheet_and_far_from_the_pocket(self):
        pts = background_grid()
        sx0, sx1, sy0, sy1 = L.rect_bounds(L.SHEET)
        for name, x, y in L.FLAG_STARTS:
            self.assertGreaterEqual(abs(y) - L.FLAG_RADIUS - sy1, 0.0775 - 1e-9, name)  # 77.5 mm off the sheet
            self.assertGreater(min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)), 0.333, name)
            self.assertGreaterEqual(L.nearest_stain_distance(x, y), 0.120 - 1e-9, name)
            self.assertIsNone(item_at(x, y), name)
            # the other arm's base is 0.705 m away in y: each arm reaches only its own two flags
            self.assertAlmostEqual(abs(y) + 0.305, 0.705)

    def test_tilt_and_base(self):
        self.assertAlmostEqual(tilt_deg(flag(*BG)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(flag(*BG, tilt=180.0)), 0.0, places=5)  # the other end down
        self.assertAlmostEqual(tilt_deg(flag(*BG, tilt=171.0)), 9.0, places=5)
        self.assertTrue(upright(flag(*BG, tilt=9.5)))
        self.assertTrue(upright(flag(*BG, tilt=-9.5)))
        self.assertFalse(upright(flag(*BG, tilt=10.5)))
        f = flag(0.30, 0.10, tilt=8.0)
        self.assertTrue(np.allclose(base_centre(f)[:2], (0.30, 0.10)))  # the base, not the leaning middle
        self.assertGreater(abs(f.pos[1] - 0.10), 0.004)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(f.pos) - base_centre(f))), L.FLAG_HEIGHT / 2)

    def test_on_the_ground_is_5_mm(self):
        self.assertTrue(on_ground(flag(*BG)))  # on the sheet, 1 mm
        self.assertTrue(on_ground(flag(*CORE["S1"])))  # on a core, 3 mm
        self.assertTrue(on_ground(flag(0.2, 0.4)))  # on the table
        self.assertTrue(on_ground(flag(*BG, bottom=L.TOP + 0.0049)))
        self.assertFalse(on_ground(flag(*BG, bottom=L.TOP + 0.0051)))  # held up 4 mm over the sheet
        self.assertFalse(on_ground(flag(*CORE["S1"], bottom=L.CORE_TOP + L.FLAG_HEIGHT)))  # on another flag
        self.assertFalse(on_ground(flag(*BG, bottom=L.TOP - 0.10)))  # fallen off the table

    def test_target_inset_is_5_mm_each_side(self):
        for s in L.STAIN_NAMES:
            x0, x1, y0, y1 = L.rect_bounds(L.STAINS[s]["core"])
            cx, cy = CORE[s]
            for p, want in (((x0 + 0.0051, cy), True), ((x0 + 0.0049, cy), False), ((x1 - 0.0051, cy), True),
                            ((x1 - 0.0049, cy), False), ((cx, y0 + 0.0051), True), ((cx, y0 + 0.0049), False),
                            ((cx, y1 - 0.0051), True), ((cx, y1 - 0.0049), False)):
                self.assertEqual(in_target(*p, s), want, (s, p))
                self.assertEqual(item_at(*p), s if want else None, (s, p))

    def test_background_needs_150_mm_and_15_mm_inside_the_sheet(self):
        # straight in front of S2's halo (x 0.305) along y 0: 150 mm away at x 0.155
        self.assertTrue(is_background(0.1549, 0.0))
        self.assertFalse(is_background(0.1551, 0.0))
        x0 = L.rect_bounds(L.SHEET)[0]
        self.assertTrue(is_background(x0 + 0.0151, 0.0))
        self.assertFalse(is_background(x0 + 0.0149, 0.0))
        self.assertFalse(is_background(0.05, 0.0))  # the bare table in front of the sheet: not the site
        self.assertFalse(is_background(0.20, 0.45))  # the bare table beside it

    def test_wholly_on_the_sheet_means_15_mm_inside_its_edges(self):
        # the instruction asks for a flag wholly on clean grey sheet; the scorer asks for its base centre 15 mm
        # inside the sheet's edges: the flag's 12.5 mm radius with 2.5 mm to spare
        self.assertAlmostEqual(L.SHEET_INSET - L.FLAG_RADIUS, 0.0025)
        x0 = L.rect_bounds(L.SHEET)[0]
        for x in (x0 + 0.0005, x0 + 0.010, x0 + 0.0125, x0 + 0.0149):  # on the sheet, over 150 mm from every halo
            self.assertGreater(L.nearest_stain_distance(x, 0.0), L.BACKGROUND_MIN)
            self.assertFalse(is_background(x, 0.0), x)
            self.assertEqual(play(layout(flag_l1=flag(x, 0.0))).score, 0.0, x)
        self.assertEqual(play(layout(flag_l1=flag(x0 + 0.0151, 0.0))).score, 0.25)

    def test_no_point_meets_two_items(self):
        for s in L.STAIN_NAMES:
            x0, x1, y0, y1 = L.target(s)
            for x in np.linspace(x0, x1, 9):
                for y in np.linspace(y0, y1, 9):
                    self.assertFalse(is_background(x, y))
                    self.assertEqual([t for t in L.STAIN_NAMES if in_target(x, y, t)], [s])

    def test_path_length(self):
        self.assertAlmostEqual(path_length([np.zeros(3), np.array([0.003, 0, 0]), np.array([0.003, 0.004, 0])]),
                               0.007)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = play(start_flags())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertTrue(Tracker.at_start(start_flags()))

    def test_all_four_scores_one(self):
        r = play(layout(**ALL_FOUR))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["items"], ["S1", "S2", "S3", "background"])

    def test_every_tier(self):
        spots = {"S1": ("flag_l2", flag(*CORE["S1"])), "S2": ("flag_r2", flag(*CORE["S2"])),
                 "S3": ("flag_r1", flag(*CORE["S3"])), "background": ("flag_l1", flag(*BG))}
        for items, want in (((), 0.0), (("S1",), 0.25), (("background",), 0.25), (("S1", "S3"), 0.5),
                            (("S2", "background"), 0.5), (("S1", "S2", "S3"), 0.75),
                            (("S1", "S2", "background"), 0.75), (("S1", "S2", "S3", "background"), 1.0)):
            r = play(layout(**dict(spots[i] for i in items)))
            self.assertEqual(r.score, want, items)
            self.assertEqual(r.success, want == 1.0, items)

    def test_two_flags_on_one_core_count_once(self):
        x0, x1, y0, y1 = L.target("S1")
        flags = layout(flag_l1=flag(x0 + 0.002, y0 + 0.002), flag_l2=flag(x1 - 0.002, y1 - 0.002))
        self.assertEqual(play(flags).score, 0.25)

    def test_two_background_flags_count_once(self):
        flags = layout(flag_l1=flag(*BG), flag_r1=flag(0.130, -0.050))
        self.assertTrue(is_background(0.130, -0.050))
        self.assertEqual(play(flags).score, 0.25)

    def test_one_arms_two_flags_score_half(self):
        self.assertEqual(play(layout(flag_l2=flag(*CORE["S1"]), flag_l1=flag(*BG))).score, 0.5)  # left alone
        self.assertEqual(play(layout(flag_r2=flag(*CORE["S2"]), flag_r1=flag(*CORE["S3"]))).score, 0.5)  # right

    def test_flags_anywhere_in_the_targets_count(self):
        flags = {}
        for name, s in (("flag_l2", "S1"), ("flag_r2", "S2"), ("flag_r1", "S3")):
            x0, x1, y0, y1 = L.target(s)
            flags[name] = flag(x0 + 0.001, y1 - 0.001)  # a corner of the target
        flags["flag_l1"] = flag(0.150, -0.030)
        self.assertEqual(play(layout(**flags)).score, 1.0)

    def test_score_is_idempotent(self):
        t = Tracker()
        flags = layout(**ALL_FOUR)
        play(flags, tracker=t)
        arms = snap()
        a, b = t.result(flags, arms), t.result(flags, arms)
        self.assertEqual((a.score, a.success, t.steps), (b.score, b.success, t.steps))


class Release(unittest.TestCase):
    """(d) let go and clear: no gripper's pad-tip point lower than 10 mm above the flag's top with the flag's axis
    within that gripper's finger reach horizontally (over every yaw)."""

    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0), 0.0779, places=4)  # fully open
        self.assertAlmostEqual(L.release_reach(0.025 / 0.04695), 0.0578, places=4)  # the demo's 50 mm opening
        self.assertAlmostEqual(L.release_reach(ON_FLAG), 0.0476, places=4)  # closed on a flag
        self.assertAlmostEqual(L.release_reach(SHUT), 0.0399, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=float("nan"))), L.release_reach(1.0))  # unreadable

    def test_a_flag_held_on_its_core_does_not_count(self):
        x, y = CORE["S1"]
        f = flag(x, y)
        tips = (x, y, L.CORE_TOP + 0.001 + 0.040 - TIPS)  # the demo's grip: pad tips 27 mm above its bottom
        self.assertTrue(held(f, snap(left=tips, left_open=ON_FLAG)))
        r = play(layout(**ALL_FOUR), arms=snap(left=tips, left_open=ON_FLAG))
        self.assertEqual(r.score, 0.75)

    def test_an_open_gripper_round_the_flag_is_still_holding_it(self):
        x, y = CORE["S2"]
        tips = (x, y, L.CORE_TOP + 0.028)
        self.assertTrue(held(flag(x, y), snap(right=tips, right_open=0.5)))

    def test_an_open_finger_propping_a_flag_is_holding_it(self):
        x, y = BG
        f = flag(x, y)
        low = L.SHEET_TOP + 0.020
        # fully open, one finger's inside face against the flag: pad-tip point 47 - 12.5 = 34.5 mm from its axis
        self.assertTrue(held(f, snap(left=(x, y + 0.0345, low))))
        # the finger's outside face against the flag: 47 + 10 + 12.5 = 69.5 mm away
        self.assertTrue(held(f, snap(left=(x, y + 0.0695, low))))
        self.assertTrue(held(f, snap(left=(x + 0.077, y, low))))  # just inside the 78 mm open reach
        self.assertFalse(held(f, snap(left=(x + 0.079, y, low))))  # just outside it: no finger can touch

    def test_the_reach_follows_the_opening(self):
        x, y = BG
        f = flag(x, y)
        low = L.SHEET_TOP + 0.020
        self.assertTrue(held(f, snap(left=(x + 0.045, y, low), left_open=ON_FLAG)))
        self.assertFalse(held(f, snap(left=(x + 0.050, y, low), left_open=ON_FLAG)))
        self.assertTrue(held(f, snap(left=(x + 0.050, y, low), left_open=OPEN)))

    def test_10_mm_above_the_top_is_clear(self):
        x, y = BG
        f = flag(x, y)
        top = L.SHEET_TOP + L.FLAG_HEIGHT
        self.assertTrue(held(f, snap(left=(x, y, top + 0.0099))))  # pressing on its top counts as holding
        self.assertFalse(held(f, snap(left=(x, y, top + 0.0101))))

    def test_tip_offset_follows_a_tilted_axis(self):
        f = flag(0.30, 0.10, tilt=8.0)  # top leans toward -y
        u = np.array([0.0, -math.sin(math.radians(8.0)), math.cos(math.radians(8.0))])
        p = base_centre(f) + u * 0.040
        xy, above = tip_offset(SimpleNamespace(tips=p), f)
        self.assertAlmostEqual(xy, 0.0, places=6)
        self.assertLess(above, 0.0)

    def test_either_arm_counts(self):
        x, y = CORE["S3"]
        tips = (x, y, L.CORE_TOP + 0.03)
        self.assertTrue(held(flag(x, y), snap(right=tips)))
        self.assertTrue(held(flag(x, y), snap(left=tips)))

    def test_the_demo_end_pose_is_clear(self):
        # after the last 'straight up' the TCPs hold 10 cm above the table over the last two flags, 50 mm open
        flags = layout(**ALL_FOUR)
        half = 0.025 / 0.04695
        arms = snap(left=(*BG, L.TOP + 0.100 - TIPS), right=(*CORE["S3"], L.TOP + 0.100 - TIPS), left_open=half,
                    right_open=half)
        r = play(flags, arms=arms)
        self.assertEqual(r.score, 1.0)
        self.assertGreater(r.details["flags"]["flag_l1"]["tip_above_mm"], 20.0)
        self.assertGreater(r.details["flags"]["flag_r1"]["tip_above_mm"], 20.0)

    def test_neighbouring_spots_are_out_of_reach(self):
        # a gripper still low over one spot, fully open, cannot touch a flag on any other spot: the three targets
        # and the background pocket are at least 110 mm apart
        pocket = background_grid(0.002)
        areas = {s: [(x, y) for x in np.linspace(*L.target(s)[:2], 9) for y in np.linspace(*L.target(s)[2:], 9)]
                 for s in L.STAIN_NAMES}
        areas["background"] = [tuple(p) for p in pocket]
        names = list(areas)
        gaps = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pa, pb = np.array(areas[a]), np.array(areas[b])
                gaps.append(min(np.hypot(*(pa - q).T).min() for q in pb))
        self.assertGreater(min(gaps), 0.110)
        self.assertGreater(min(gaps), L.release_reach(1.0))

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        t = Tracker()
        r = play(layout(**ALL_FOUR), tracker=t, arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)
        self.assertEqual(r.details["arms_unavailable_by_side"], {"left": 12, "right": 12})
        self.assertFalse(r.details["arms_available"])
        self.assertEqual(r.details["arms_readable"], [])
        self.assertIsNone(r.details["reach_mm"])
        # left unreadable, right readable and parked far from every flag: still 1.0, and the step is counted
        t = Tracker()
        bad = {"left": SimpleNamespace(tips=np.array([np.nan, 0, 0])), "right": SimpleNamespace(tips=np.zeros(3))}
        r = play(layout(**ALL_FOUR), tracker=t, arms=bad)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)
        self.assertEqual(r.details["arms_unavailable_by_side"], {"left": 12, "right": 0})
        self.assertEqual(r.details["arms_readable"], ["right"])
        self.assertIsNone(r.details["reach_mm"]["left"])

    def test_one_unreadable_arm_never_hides_the_other(self):
        # review exploit: the left gripper closed on the S1 flag (opening 0.27, pad tips 27 mm up) while the right
        # arm's pad-tip point is NaN or its key is missing. Each arm is checked on its own, so S1 is still held.
        x, y = CORE["S1"]
        tips = np.array((x, y, L.CORE_TOP + 0.027))
        gripping = SimpleNamespace(tcp=tips + (0, 0, TIPS), tips=tips, opening=0.27)
        nan = SimpleNamespace(tcp=np.full(3, np.nan), tips=np.full(3, np.nan), opening=1.0)
        far = SimpleNamespace(tcp=np.array(READY_TIPS["right"]) + (0, 0, TIPS), tips=np.array(READY_TIPS["right"]),
                              opening=1.0)
        self.assertEqual(play(layout(**ALL_FOUR), arms={"left": gripping, "right": far}).score, 0.75)  # both read
        for right in (nan, None, SimpleNamespace(opening=1.0), SimpleNamespace(tips=np.zeros(2), opening=1.0)):
            arms = {"left": gripping} if right is None else {"left": gripping, "right": right}
            self.assertTrue(held(ALL_FOUR["flag_l2"], arms))
            t = Tracker()
            r = play(layout(**ALL_FOUR), tracker=t, arms=arms)
            self.assertEqual(r.score, 0.75, right)
            self.assertFalse(r.details["flags"]["flag_l2"]["planted"])
            self.assertEqual(t.arms_unavailable, 12)
            self.assertEqual(r.details["arms_unavailable_by_side"], {"left": 0, "right": 12})
            self.assertEqual(r.details["arms_readable"], ["left"])
        # the mirror: the right gripper closed on the S2 flag, the left arm unreadable
        x, y = CORE["S2"]
        tips = np.array((x, y, L.CORE_TOP + 0.027))
        gripping = SimpleNamespace(tcp=tips + (0, 0, TIPS), tips=tips, opening=0.27)
        self.assertEqual(play(layout(**ALL_FOUR), arms={"left": nan, "right": gripping}).score, 0.75)
        self.assertEqual(play(layout(**ALL_FOUR), arms={"right": gripping}).score, 0.75)


class Stillness(unittest.TestCase):
    """(c) at rest: the base centre's path over the last 0.5 s (10 control steps) under 2 mm."""

    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        flags = layout(**ALL_FOUR)
        rs = [t.update(flags, snap()) for _ in range(11)]
        self.assertEqual(rs[9].score, 0.0)  # 10 samples: not yet 0.5 s
        self.assertEqual(rs[10].score, 1.0)

    def test_velocity_chatter_is_ignored(self):
        flags = layout(**ALL_FOUR)
        flags["flag_l2"] = flag(*CORE["S1"], vel=(0.04, -0.03, 0.02))  # still, but the velocity reads 45 mm/s
        self.assertEqual(play(flags).score, 1.0)

    def test_a_sliding_flag_counts_only_once_it_stops(self):
        t = Tracker()
        rest = layout(flag_l2=flag(*CORE["S1"]), flag_r2=flag(*CORE["S2"]), flag_r1=flag(*CORE["S3"]))
        x, y = BG
        r = None
        for i in range(20):  # sliding at 60 mm/s into the pocket
            r = t.update(dict(rest, flag_l1=flag(x + 0.003 * (i - 20), y)), snap())
        self.assertEqual(r.score, 0.75)
        rs = [t.update(dict(rest, flag_l1=flag(x, y)), snap()) for _ in range(11)]
        self.assertEqual(rs[-2].score, 0.75)  # 10 steps still: the window still holds the last 3 mm step
        self.assertEqual(rs[-1].score, 1.0)

    def test_a_creep_is_measured_as_path(self):
        rest = layout(flag_l2=flag(*CORE["S1"]), flag_r2=flag(*CORE["S2"]), flag_r1=flag(*CORE["S3"]))
        for per_step, want in ((0.00019, 1.0), (0.00021, 0.75)):
            t = Tracker()
            r = None
            for i in range(30):
                r = t.update(dict(rest, flag_l1=flag(BG[0] + per_step * i, BG[1])), snap())
            self.assertEqual(r.score, want, per_step)
        t = Tracker()  # a 0.15 mm shiver back and forth adds up: 10 x 0.3 mm = 3 mm of path
        r = None
        for i in range(30):
            r = t.update(dict(rest, flag_l1=flag(BG[0] + 0.0003 * (i % 2), BG[1])), snap())
        self.assertEqual(r.score, 0.75)


class NotCounted(unittest.TestCase):
    """Flags that earn nothing, whatever spot they are on."""

    def score_with(self, f, name="flag_l2"):
        return play(layout(**{name: f})).score

    def test_lying_down_on_a_core(self):
        self.assertEqual(self.score_with(flag(*CORE["S1"], tilt=90.0)), 0.0)

    def test_tilted_over_10_degrees(self):
        self.assertEqual(self.score_with(flag(*CORE["S1"], tilt=12.0)), 0.0)
        self.assertEqual(self.score_with(flag(*CORE["S1"], tilt=9.0)), 0.25)

    def test_hovering_or_held_above_the_core(self):
        self.assertEqual(self.score_with(flag(*CORE["S1"], bottom=L.CORE_TOP + 0.003)), 0.0)

    def test_on_the_halo_or_streak_off_the_core(self):
        for s in L.STAIN_NAMES:
            for kind, p in decoys(s).items():
                self.assertIsNone(item_at(*p), (s, kind))
                self.assertEqual(self.score_with(flag(*p)), 0.0, (s, kind))

    def test_between_stains_or_off_the_sheet(self):
        for p in ((0.40, 0.08), (0.36, 0.105), (0.30, -0.14), (0.05, 0.0), (0.50, 0.0), (0.20, 0.45)):
            self.assertIsNone(item_at(*p), p)
            self.assertEqual(self.score_with(flag(*p)), 0.0, p)


class Robustness(unittest.TestCase):
    def test_nan_scores_zero_and_keeps_the_record(self):
        t = Tracker()
        flags = layout(**ALL_FOUR)
        play(flags, tracker=t)
        hist = {n: [tuple(p) for p in t.history[n]] for n in NAMES}
        bad = dict(flags, flag_l1=flag(*BG))
        bad["flag_l1"].pos = np.array([np.nan, 0.0, 0.0])
        r = t.update(bad, snap())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual({n: [tuple(p) for p in t.history[n]] for n in NAMES}, hist)
        self.assertEqual(t.invalid_steps, 1)
        self.assertEqual(t.result(bad, snap()).score, 0.0)
        self.assertEqual(t.update(flags, snap()).score, 1.0)  # the next good step carries on

    def test_zero_quaternion_infinite_box_and_missing_flag(self):
        self.assertFalse(valid(flag(*BG, quat=(0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        inf = flag(*BG)
        inf.high = np.array([np.inf, 0.0, 0.0])
        self.assertFalse(valid(inf))
        flags = layout(**ALL_FOUR)
        flags["flag_r1"] = flag(*CORE["S3"], quat=(0, 0, 0, 0))
        self.assertEqual(Tracker().update(flags, snap()).score, 0.0)
        flags = layout(**ALL_FOUR)
        del flags["flag_r1"]
        self.assertEqual(Tracker().update(flags, snap()).score, 0.0)
        self.assertEqual(Tracker().update(None, snap()).score, 0.0)

    def test_teleport_back_to_the_start_is_a_new_episode(self):
        t = Tracker()
        play(layout(**ALL_FOUR), tracker=t)
        self.assertGreater(t.steps, 1)
        r = t.update(start_flags(), snap())
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history["flag_l1"]), 1)

    def test_putting_one_flag_back_is_not_a_reset(self):
        t = Tracker()
        play(layout(flag_l2=flag(*CORE["S1"]), flag_l1=flag(*BG)), tracker=t)
        r = t.update(layout(flag_l1=flag(*BG)), snap())  # flag_l2 is back on its mark, flag_l1 is not
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

    def test_holding_flags_on_the_spots_earns_nothing_for_them(self):
        flags = layout(**ALL_FOUR)
        lx, ly = CORE["S1"]
        rx, ry = CORE["S2"]
        arms = snap(left=(lx, ly, L.CORE_TOP + 0.027), right=(rx, ry, L.CORE_TOP + 0.027), left_open=ON_FLAG,
                    right_open=ON_FLAG)
        self.assertEqual(play(flags, arms=arms).score, 0.5)

    def test_nudging_a_flag_outboard_is_not_background(self):
        # the old layout's exploit: clean ground just outboard of the flag starts. Now they start off the sheet
        for name, x, y in L.FLAG_STARTS:
            for dy in (0.01, 0.02, 0.05, 0.10):
                ny = y + math.copysign(dy, y)
                self.assertEqual(play(layout(**{name: flag(x, ny)})).score, 0.0, (name, dy))
            for dy in (-0.01, -0.02, -0.05, -0.08):  # inboard, onto the sheet's side margin
                ny = y + math.copysign(dy, y)
                self.assertEqual(play(layout(**{name: flag(x, ny)})).score, 0.0, (name, dy))

    def test_aiming_at_the_middle_of_each_stain(self):
        for kind in ("main_lobe", "streak", "centroid", "bbox"):
            flags = layout(flag_l2=flag(*decoys("S1")[kind]), flag_r2=flag(*decoys("S2")[kind]),
                           flag_r1=flag(*decoys("S3")[kind]), flag_l1=flag(*BG))
            self.assertEqual(play(flags).score, 0.25, kind)  # only the background point

    def test_aiming_at_the_middle_with_a_placement_error_still_misses(self):
        for s in L.STAIN_NAMES:
            cx, cy = decoys(s)["main_lobe"]
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                p = (cx + 0.020 * math.cos(a), cy + 0.020 * math.sin(a))  # 20 mm off the lobe's centre
                self.assertNotEqual(item_at(*p), s, (s, p))

    def test_stacking_flags(self):
        x, y = CORE["S1"]
        flags = layout(flag_l2=flag(x, y), flag_l1=flag(x, y, bottom=L.CORE_TOP + L.FLAG_HEIGHT))
        r = play(flags)
        self.assertEqual(r.score, 0.25)  # the bottom one counts, the one on top of it does not
        self.assertFalse(r.details["flags"]["flag_l1"]["planted"])
        bx, by = BG
        flags = layout(flag_l1=flag(bx, by, bottom=L.SHEET_TOP + L.FLAG_HEIGHT))  # alone on top of nothing
        self.assertEqual(play(flags).score, 0.0)

    def test_a_flag_dropped_onto_a_core_counts_once_it_stands_still(self):
        t = Tracker()
        rest = layout(flag_l1=flag(*BG), flag_r2=flag(*CORE["S2"]), flag_r1=flag(*CORE["S3"]))
        z, v = 0.05, 0.0
        r = None
        while z > 0:  # free fall from 5 cm onto the S1 core
            v += 9.81 * DT
            z = max(0.0, z - v * DT)
            r = t.update(dict(rest, flag_l2=flag(*CORE["S1"], bottom=L.CORE_TOP + z, vel=(0, 0, -v))), snap())
        self.assertLess(r.score, 1.0)
        r = play(dict(rest, flag_l2=flag(*CORE["S1"])), n=11, tracker=t)
        self.assertEqual(r.score, 1.0)  # how it got there is not scored; it stands on the darkest spot

    def test_one_flag_lying_across_two_items(self):
        # a flag lying down with its middle on a core earns nothing; nothing lies close enough to meet two items
        self.assertEqual(play(layout(flag_l2=flag(*CORE["S1"], tilt=90.0))).score, 0.0)

    def test_spraying_all_four_flags_over_one_stain(self):
        x0, x1, y0, y1 = L.target("S2")
        flags = {n: flag(x, y) for n, (x, y) in zip(NAMES, ((x0, y0), (x1, y0), (x0, y1), (x1, y1)))}
        self.assertEqual(play(flags).score, 0.25)

    def test_the_near_edge_of_the_sheet_counts_only_in_the_pocket(self):
        x = L.rect_bounds(L.SHEET)[0] + L.SHEET_INSET + 0.002
        ys = np.linspace(-0.29, 0.29, 59)
        good = [y for y in ys if is_background(x, y)]
        self.assertTrue(good)
        self.assertGreaterEqual(min(good), -0.085)
        self.assertLessEqual(max(good), 0.047)


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

    def test_objects(self):
        objs = {o.name: o for o in self.task.OBJECTS}
        self.assertEqual(len(objs), 17)
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None], list(NAMES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        tops = {"site_sheet": L.SHEET_TOP}
        for s in L.STAIN_NAMES:
            k = s.lower()
            tops.update({f"{k}_halo_0": L.HALO_TOP, f"{k}_halo_1": L.HALO_TOP, f"{k}_streak": L.CORE_TOP,
                         f"{k}_core": L.CORE_TOP})
            self.assertEqual(objs[f"{k}_core"].color, L.CORE_BLACK)
            self.assertEqual(objs[f"{k}_streak"].color, L.STREAK_BROWN)
            self.assertEqual(tuple(objs[f"{k}_core"].pos[:2]), CORE[s])
        for name, top in tops.items():
            o = objs[name]
            self.assertIsNone(o.mass, name)
            self.assertAlmostEqual(o.size[2], 0.001)
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, top, msg=name)
        for name, x, y in L.FLAG_STARTS:
            o = objs[name]
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP)
            self.assertEqual((o.radius, o.height, o.mass), (L.FLAG_RADIUS, L.FLAG_HEIGHT, L.FLAG_MASS))
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        self.assertIn("darkest spot", self.task.INSTRUCTION)
        self.assertIn("wholly on clean grey sheet at least 15 cm from the edge of every stain, pale halo included",
                      self.task.INSTRUCTION)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)
        self.assertIn("27 g", spec["setup"])
        self.assertAlmostEqual(L.FLAG_MASS, 0.027)

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertEqual(tr.steps, 0)
        objects = start_flags()
        self.assertFalse(self.task.success(objects))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        placed = layout(**ALL_FOUR)
        ok = [self.task.success(placed) for _ in range(11)]
        self.assertTrue(ok[-1])
        self.assertFalse(any(ok[:-1]))
        self.assertEqual(self.task.score(placed), 1.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertEqual(self.task.score(placed), 0.0)

    def test_one_arm_missing_or_nan_through_the_hooks(self):
        # rigstate gives both arms or None; a partial snapshot must neither crash the hooks nor free a held flag
        import tempfile
        trace = pathlib.Path(tempfile.mkdtemp()) / "trace.jsonl"
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(trace)
        x, y = CORE["S1"]
        tips = np.array((x, y, L.CORE_TOP + 0.027))
        left = {"tcp": tips + (0, 0, TIPS), "tips": tips, "opening": 0.27}
        try:
            for right in (None, {"tcp": np.full(3, np.nan), "tips": np.full(3, np.nan), "opening": 1.0}):
                rigstate.offline_reset()
                placed = layout(**ALL_FOUR)
                for _ in range(12):
                    rigstate.offline_feed({"left": left} if right is None else {"left": left, "right": right})
                    self.assertFalse(self.task.success(placed))
                self.assertEqual(self.task.score(placed), 0.75)
                row = json.loads(trace.read_text().splitlines()[-1])
                self.assertEqual(row["arms_readable"], ["left"])
                self.assertEqual(row["arms_unavailable"], 12)
                if right is None:
                    self.assertIsNone(row["right_tips"])
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
            rigstate.offline_reset()

    def test_carry_paths_cross_no_placed_flag(self):
        # at the travel height a carried flag's bottom is 60 mm up, below the top of a standing flag (61 mm on the
        # sheet, 63 mm on a core), so no carry may pass over a flag standing at a place point or a start
        travel_bottom = self.task.TRAVEL_Z - self.task.GRIP_UP - L.TOP
        self.assertAlmostEqual(travel_bottom, 0.060)
        self.assertLess(travel_bottom, L.SHEET_TOP - L.TOP + L.FLAG_HEIGHT)
        plan = self.task.demo()
        places = [w.pos[:2] for a in plan for w in plan[a] if w.label == "let go"]
        starts = [(x, y) for _, x, y in L.FLAG_STARTS]

        def seg_dist(p, a, b):
            p, a, b = (np.asarray(v, float) for v in (p, a, b))
            t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
            return float(np.linalg.norm(p - (a + t * (b - a))))

        n = 0
        for a in plan:
            for prev, w in zip(plan[a], plan[a][1:]):
                if not w.label.startswith("carry"):
                    continue
                n += 1
                for q in places + starts:
                    if np.allclose(q, w.pos[:2]) or np.allclose(q, prev.pos[:2]):
                        continue  # its own place point or start
                    self.assertGreater(seg_dist(q, prev.pos[:2], w.pos[:2]), 2 * L.FLAG_RADIUS + 0.030, (a, w.label, q))
        self.assertEqual(n, 4)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(start_flags()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for a in plan:
            self.assertAlmostEqual(sum(w.duration for w in plan[a]), 26.4)
        places = {a: [w.pos for w in plan[a] if w.label == "let go"] for a in plan}
        want = {"left": [(CORE["S1"], L.CORE_TOP), (BG, L.SHEET_TOP)],
                "right": [(CORE["S2"], L.CORE_TOP), (CORE["S3"], L.CORE_TOP)]}
        for a in plan:
            for p, (xy, top) in zip(places[a], want[a]):
                self.assertAlmostEqual(p[0], xy[0])
                self.assertAlmostEqual(p[1], xy[1])
                self.assertAlmostEqual(p[2] - 0.040, top + 0.001)  # flag bottom 1 mm above the surface
                item = item_at(p[0], p[1])
                self.assertIsNotNone(item)
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every close and open takes 1.5 s
                if w.label == "let go" or w.label.startswith("grip"):
                    self.assertEqual(prev.label, "settle")
                if w.label.startswith("lower the last"):
                    self.assertAlmostEqual(prev.pos[2] - w.pos[2], 0.020)
                    self.assertGreaterEqual(w.duration, 1.2)  # 17 mm/s at most, slower at the ends
                self.assertEqual(w.yaw, 0.0)
                self.assertLessEqual(w.pos[2], L.TOP + 0.100 + 1e-9)
            grips = [w.finger for w in plan[a] if w.label.startswith("grip")]
            self.assertTrue(all(abs(g + 0.0085) < 1e-12 for g in grips))  # 17 mm on the 25 mm flag
            self.assertTrue(all(abs(w.finger + 0.025) < 1e-12 for w in plan[a] if w.label == "let go"))
        # every demo place point lies 20 mm inside its target, or 20 mm inside the background pocket
        for s in L.STAIN_NAMES:
            self.assertAlmostEqual(L.rect_distance(*CORE[s], L.STAINS[s]["core"]), 0.0)
        for key, val, want_arms in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                                    ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want_arms)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
