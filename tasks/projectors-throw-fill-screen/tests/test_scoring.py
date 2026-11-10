"""Offline tests for scoring.py and the hooks in task.py: synthetic projector states, no simulator.
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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    WINDOW, Tracker, heading_rad, lens_point, let_go, lowest, nearest_screen, over_table, picture_width, read_arms,
    released, square_err_deg, throw_distance, tilt_deg, valid, width_err, zone_gap, zone_radius,
)

TABLE = L.TABLE_TOP_W
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}
BLACK, WHITE = L.PROJECTORS
SL, SR = L.SCREENS


def quat(heading=0.0, pitch=0.0, roll=0.0):
    """(x, y, z, w) of Rz(heading) Ry(pitch) Rx(roll), degrees."""
    h, p, r = (math.radians(v) / 2 for v in (heading, pitch, roll))
    qz = np.array([0.0, 0.0, math.sin(h), math.cos(h)])
    qy = np.array([0.0, math.sin(p), 0.0, math.cos(p)])
    qx = np.array([math.sin(r), 0.0, 0.0, math.cos(r)])

    def mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

    return mul(mul(qz, qy), qx)


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def proj(x, y, heading=0.0, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0), floor=TABLE):
    """A projector state as mechkit reports it: frame origin (the middle of the body's bottom face) at (x, y), turned
    `heading` degrees and tilted by `pitch`/`roll`, resting with its lowest point `lift` above `floor` (the projector
    table's top unless given); low/high from its 31 x 34 x 14 mm outline (lens included)."""
    q = quat(heading, pitch, roll)
    local = np.array([(u, v, z) for u in (-L.FOOT_BACK, L.FOOT_FRONT) for v in (-L.FOOT_V, L.FOOT_V)
                      for z in (0.0, L.BODY_H)])
    pts = local @ rot(q).T
    o = np.array([x, y, floor + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


def placed(name, screen, dist=None, off=0.0, heading=0.0, **kw):
    """A projector standing on the table facing the wall (turned `heading` degrees), the centre of its lens's front
    face `dist` from the screens' plane (default: the distance its ratio needs) and `off` to the left of the screen's
    centre line."""
    dist = L.throw_for(name) if dist is None else dist
    lens = np.array([L.SCREEN_FACE_X - dist, L.SCREEN_Y[screen] + off])
    h = math.radians(heading)
    o = lens - L.LENS_FRONT_U * np.array([math.cos(h), math.sin(h)])
    s = proj(float(o[0]), float(o[1]), heading, **kw)
    if abs(heading) < 1e-12 and not kw:
        assert abs(throw_distance(s) - dist) < 1e-12 and abs(lens_point(s)[1] - lens[1]) < 1e-12
    return s


def width_dist(name, frac):
    """The throw distance that gives this projector a picture `frac` wider than the screen (frac -0.05 = 5 % narrower)."""
    return L.RATIO[name] * L.SCREEN_W * (1.0 + frac)


def arm(tips, opening=1.0):
    """A rigstate-like arm pointing straight down, its finger-tip point at ``tips`` (the TCP 13 mm above it)."""
    t = np.asarray(tips, dtype=float)
    return rigstate.Arm(tcp=t + [0.0, 0.0, 0.013], tips=t, opening=float(opening))


PARKED = {"left": arm((0.200, 0.300, L.TOP + 0.137)), "right": arm((0.200, -0.300, L.TOP + 0.137))}  # demo's park
GRIPPED = L.BODY_W / L.FULL_GAP  # rigstate's opening with the pads on the 34 mm body: 0.362


def holding(p, side="left", opening=GRIPPED, other=None):
    """Arms with ``side``'s gripper around projector ``p`` as the demo grips it (tips 4 mm above its base, at its
    origin), the other arm parked (or ``other``)."""
    tips = np.asarray(p.origin, dtype=float) + [0.0, 0.0, 0.004]
    arms = dict(PARKED)
    arms[side] = arm(tips, opening)
    if other is not None:
        arms["right" if side == "left" else "left"] = other
    return arms


START_B = proj(*L.START[BLACK][:2], math.degrees(L.START[BLACK][2]))
START_W = proj(*L.START[WHITE][:2], math.degrees(L.START[WHITE][2]))
GOOD_B = placed(BLACK, SL)
GOOD_W = placed(WHITE, SR)


def settled(black, white, t=None, n=WINDOW + 1, arms=PARKED):
    """Feed the same pair n times (half a second still), the grippers parked open unless ``arms`` is given; return
    (result, tracker)."""
    t = Tracker() if t is None else t
    r = None
    for _ in range(n):
        r = t.update(black, white, arms)
    return r, t


def scored(black, white=GOOD_W) -> float:
    return settled(black, white)[0].score


def why(black, white=GOOD_W, which=BLACK):
    return settled(black, white)[0].details[which]["why"]


class Geometry(unittest.TestCase):
    def test_lens_point_and_throw(self):
        s = proj(0.20, 0.10, 0.0)
        np.testing.assert_allclose(lens_point(s), [0.20 + L.LENS_FRONT_U, 0.10, TABLE + L.LENS_Z], atol=1e-12)
        self.assertAlmostEqual(throw_distance(s), L.SCREEN_FACE_X - 0.218, places=12)
        self.assertAlmostEqual(picture_width(BLACK, s), throw_distance(s) / 1.5, places=12)
        self.assertAlmostEqual(picture_width(WHITE, s), throw_distance(s), places=12)
        s = proj(0.20, 0.10, 90.0)  # turned: the lens's centre moves sideways
        np.testing.assert_allclose(lens_point(s)[:2], [0.20, 0.10 + L.LENS_FRONT_U], atol=1e-12)

    def test_heading_tilt_height(self):
        for h in (0.0, 30.0, -90.0, 180.0):
            self.assertAlmostEqual(math.remainder(math.degrees(heading_rad(proj(0.3, 0.1, h))) - h, 360.0), 0.0)
        self.assertAlmostEqual(square_err_deg(proj(0.3, 0.1, -7.0)), 7.0)
        self.assertAlmostEqual(square_err_deg(proj(0.3, 0.1, 180.0)), 180.0)
        self.assertAlmostEqual(tilt_deg(proj(0.3, 0.1, 0.0, pitch=4.0)), 4.0, places=6)
        self.assertAlmostEqual(tilt_deg(proj(0.3, 0.1, 0.0, roll=180.0)), 180.0, places=6)
        self.assertAlmostEqual(lowest(proj(0.3, 0.1, 0.0, lift=0.002)), 0.002, places=12)

    def test_the_rule_numbers(self):
        self.assertAlmostEqual(L.throw_for(BLACK), 0.240, places=12)  # 1.5 x 1.6 m = 2.4 m
        self.assertAlmostEqual(L.throw_for(WHITE), 0.160, places=12)  # 1.0 x 1.6 m = 1.6 m
        np.testing.assert_allclose(L.TARGET[BLACK][:2], (0.210, 0.120), atol=1e-12)
        np.testing.assert_allclose(L.TARGET[WHITE][:2], (0.290, -0.120), atol=1e-12)
        for name, screen in ((BLACK, SL), (WHITE, SR)):
            s = proj(*L.TARGET[name])
            self.assertAlmostEqual(picture_width(name, s), L.SCREEN_W, places=12)
            self.assertEqual(nearest_screen(s), (screen, nearest_screen(s)[1]))
            self.assertAlmostEqual(nearest_screen(s)[1], 0.0, places=12)
            self.assertTrue(over_table(s))
        # every place a band allows is on the table: the loose band's ends, either screen
        for name in L.PROJECTORS:
            for frac in (-L.LOOSE_TOL, L.LOOSE_TOL):
                for screen in L.SCREENS:
                    for off in (-L.CENTRE_LOOSE, L.CENTRE_LOOSE):
                        self.assertTrue(over_table(placed(name, screen, width_dist(name, frac), off)), (name, frac))

    def test_start_layout_scores_zero(self):
        self.assertTrue(Tracker.at_start(BLACK, START_B))
        self.assertTrue(Tracker.at_start(WHITE, START_W))
        r, _ = settled(START_B, START_W)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details[BLACK]["why"], "lens_not_toward_the_wall")
        self.assertEqual(r.details[WHITE]["why"], "lens_not_toward_the_wall")
        for s in (START_B, START_W):
            self.assertTrue(over_table(s))
            self.assertAlmostEqual(square_err_deg(s), 90.0)
            self.assertTrue(np.all(np.abs(s.high[:2]) < 0.45))  # clear of the arms' start-pose shadow
        # lenses toward the middle, 264 mm apart
        self.assertAlmostEqual(lens_point(START_B)[1] - lens_point(START_W)[1], 0.264, places=9)

    def test_demo_target(self):
        r, _ = settled(GOOD_B, GOOD_W)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        db, dw = r.details[BLACK], r.details[WHITE]
        self.assertEqual((db["throw_mm"], db["picture_mm"]), (240.0, 160.0))
        self.assertEqual((dw["throw_mm"], dw["picture_mm"]), (160.0, 160.0))
        self.assertEqual((db["screen"], dw["screen"]), (SL, SR))
        self.assertTrue(db["counted"] and dw["counted"])
        np.testing.assert_allclose(GOOD_B.origin[:2], L.TARGET[BLACK][:2], atol=1e-12)
        np.testing.assert_allclose(GOOD_W.origin[:2], L.TARGET[WHITE][:2], atol=1e-12)


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        far_b = placed(BLACK, SL, 0.260)  # 20 mm too far: picture 173 mm (+8.3 %)
        far_w = placed(WHITE, SR, 0.180)  # 20 mm too far: 180 mm (+12.5 %)
        back_w = placed(WHITE, SR, 0.160 + 2 * L.LENS_FRONT_U, heading=180.0)  # turned the wrong way: lens to the back
        self.assertEqual(scored(START_B, START_W), 0.0)
        self.assertEqual(scored(GOOD_B, GOOD_W), 1.0)
        self.assertEqual(scored(GOOD_B, far_w), 0.75)
        self.assertEqual(scored(far_b, GOOD_W), 0.75)
        self.assertEqual(scored(far_b, far_w), 0.5)
        self.assertEqual(scored(GOOD_B, back_w), 0.5)
        self.assertEqual(why(GOOD_B, back_w, WHITE), "lens_not_toward_the_wall")
        self.assertEqual(scored(START_B, far_w), 0.25)
        self.assertEqual(scored(GOOD_B, START_W), 0.5)
        # swapped distances: the black at 160 mm (picture 107 mm), the white at 240 mm (240 mm): nothing
        sw_b, sw_w = placed(BLACK, SL, 0.160), placed(WHITE, SR, 0.240)
        self.assertEqual(scored(sw_b, sw_w), 0.0)
        self.assertEqual(why(sw_b, sw_w, BLACK), "picture_too_narrow")
        self.assertEqual(why(sw_b, sw_w, WHITE), "picture_too_wide")

    def test_either_projector_may_serve_either_screen(self):
        self.assertEqual(scored(placed(BLACK, SR), placed(WHITE, SL)), 1.0)

    def test_width_band_edges(self):
        for name, screen in ((BLACK, SL), (WHITE, SR)):
            other = GOOD_W if name == BLACK else GOOD_B
            for sign in (-1, 1):
                for frac, inside, outside in ((L.FULL_TOL, 1.0, 0.75), (L.LOOSE_TOL, 0.75, 0.5)):
                    a = placed(name, screen, width_dist(name, sign * frac) - sign * 1e-5)  # 0.01 mm inside
                    b = placed(name, screen, width_dist(name, sign * frac) + sign * 1e-4)  # 0.1 mm outside
                    pair_a = (a, other) if name == BLACK else (other, a)
                    pair_b = (b, other) if name == BLACK else (other, b)
                    self.assertEqual(scored(*pair_a), inside, (name, sign, frac))
                    self.assertEqual(scored(*pair_b), outside, (name, sign, frac))
        # in millimetres: the black's full band is 228-252 mm, the white's 152-168 mm
        self.assertEqual(scored(placed(BLACK, SL, 0.2285)), 1.0)
        self.assertEqual(scored(placed(BLACK, SL, 0.2275)), 0.75)
        self.assertEqual(scored(GOOD_B, placed(WHITE, SR, 0.1675)), 1.0)
        self.assertEqual(scored(GOOD_B, placed(WHITE, SR, 0.1685)), 0.75)
        self.assertEqual(scored(placed(BLACK, SL, 0.2755)), 0.75)
        self.assertEqual(scored(placed(BLACK, SL, 0.2765)), 0.5)
        self.assertEqual(scored(GOOD_B, placed(WHITE, SR, 0.1365)), 0.75)
        self.assertEqual(scored(GOOD_B, placed(WHITE, SR, 0.1355)), 0.5)

    def test_square_edges(self):
        for h, want in ((4.9, 1.0), (-4.9, 1.0), (5.1, 0.75), (-9.9, 0.75), (10.1, 0.5), (-10.1, 0.5)):
            self.assertEqual(scored(placed(BLACK, SL, heading=h)), want, h)
        self.assertEqual(why(placed(BLACK, SL, heading=7.0)), "off_square")

    def test_centre_edges(self):
        for off, want in ((0.0059, 1.0), (-0.0059, 1.0), (0.0061, 0.75), (-0.0119, 0.75), (0.0121, 0.5),
                          (-0.0121, 0.5)):
            self.assertEqual(scored(placed(BLACK, SL, off=off)), want, off)
        self.assertEqual(why(placed(BLACK, SL, off=0.009)), "off_centre")
        self.assertEqual(why(placed(BLACK, SL, off=0.020)), "not_on_a_centre_line")
        # the centre line is read at the lens, not the body: turned -4.9 degrees with the lens 5.5 mm off the line,
        # the middle of its base is 7.0 mm off it
        s = placed(BLACK, SL, off=0.0055, heading=-4.9)
        self.assertAlmostEqual(float(s.origin[1]) - L.SCREEN_Y[SL], 0.0055 + L.LENS_FRONT_U * math.sin(math.radians(4.9)))
        self.assertEqual(scored(s), 1.0)
        self.assertEqual(scored(placed(BLACK, SL, off=0.0065, heading=4.9)), 0.75)  # base 5.0 mm off, lens 6.5
        # half way between the screens counts for neither
        self.assertEqual(why(placed(BLACK, SL, off=-0.120)), "not_on_a_centre_line")

    def test_only_a_full_score_is_success(self):
        for b, w in ((GOOD_B, GOOD_W), (GOOD_B, START_W), (placed(BLACK, SL, 0.260), GOOD_W)):
            r, _ = settled(b, w)
            self.assertEqual(r.success, r.score == 1.0)

    def test_every_tier_is_reachable(self):
        seen = {scored(START_B, START_W), scored(START_B, placed(WHITE, SR, 0.180)), scored(GOOD_B, START_W),
                scored(GOOD_B, placed(WHITE, SR, 0.180)), scored(GOOD_B, GOOD_W)}
        self.assertEqual(seen, TIERS)

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(GOOD_B, GOOD_W)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.result(START_B, GOOD_W, PARKED).score, 0.5)  # result() reads the pose it is given
        self.assertEqual(t.result(GOOD_B, GOOD_W, PARKED).score, 1.0)

    def test_random_states_map_to_a_tier(self):
        rnd = random.Random(7)
        for _ in range(400):
            ps = []
            for name in L.PROJECTORS:
                ps.append(proj(rnd.uniform(0.15, 0.40), rnd.uniform(-0.22, 0.22), rnd.uniform(-180, 180),
                               lift=rnd.choice((0.0, 0.0, 0.002, 0.01)), pitch=rnd.choice((0.0, 0.0, 3.0, 20.0))))
            r, _ = settled(*ps)
            self.assertIn(r.score, TIERS)
            self.assertEqual(r.success, r.score == 1.0)


class NothingForTheProjector(unittest.TestCase):
    """Conditions under which a projector earns nothing (the other one keeps its credit)."""

    def test_tipped_upside_down_or_on_its_side(self):
        self.assertEqual(why(placed(BLACK, SL, pitch=6.0)), "not_upright")
        self.assertEqual(scored(placed(BLACK, SL, pitch=4.5)), 1.0)
        self.assertEqual(why(placed(BLACK, SL, roll=180.0)), "not_upright")
        self.assertEqual(why(placed(BLACK, SL, roll=90.0)), "not_upright")
        self.assertEqual(scored(placed(BLACK, SL, roll=90.0)), 0.5)

    def test_lifted_or_off_the_table(self):
        self.assertEqual(why(placed(BLACK, SL, lift=0.004)), "not_on_the_table")  # held 4 mm up
        self.assertEqual(scored(placed(BLACK, SL, lift=0.0029)), 1.0)  # 2.9 mm: within 3 mm
        floor = proj(0.21, 0.12, 0.0, floor=L.TOP)  # on the station table below, not the projector table
        self.assertEqual(why(floor), "not_on_the_table")
        beyond = proj(0.40, 0.12, 0.0, floor=L.TOP)  # between the table and the wall, on the floor
        self.assertEqual(why(beyond), "not_on_the_table")
        self.assertFalse(over_table(proj(0.16, 0.12)))
        # overhanging the table's far edge or a side, at the table's height (held there, or about to fall off)
        self.assertEqual(why(proj(L.TABLE_X[1] + 0.002, 0.12, 0.0)), "not_on_the_table")
        self.assertEqual(why(proj(0.21, L.TABLE_Y[1] + 0.002, 0.0)), "not_on_the_table")
        self.assertEqual(why(proj(L.TABLE_X[1] - 0.002, 0.12, 0.0)), "picture_too_narrow")  # its base still over it

    def test_on_top_of_the_other_projector(self):
        top = proj(*L.TARGET[BLACK][:2], 0.0, lift=L.BODY_H)
        self.assertEqual(why(top, GOOD_B, BLACK), "not_on_the_table")

    def test_moving(self):
        t = Tracker()
        for i in range(WINDOW + 1):
            r = t.update(placed(BLACK, SL, 0.240 + 0.001 * i), GOOD_W, PARKED)  # 1 mm per step: 10 mm in 0.5 s
        self.assertEqual(r.details[BLACK]["why"], "moving")
        self.assertEqual(r.score, 0.5)

    def test_lens_backwards_or_sideways(self):
        self.assertEqual(why(placed(BLACK, SL, heading=180.0)), "lens_not_toward_the_wall")
        self.assertEqual(why(placed(BLACK, SL, heading=90.0)), "lens_not_toward_the_wall")
        self.assertEqual(why(placed(BLACK, SL, heading=15.0)), "lens_not_toward_the_wall")


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for _ in range(WINDOW):
            r = t.update(GOOD_B, GOOD_W, PARKED)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.update(GOOD_B, GOOD_W, PARKED).score, 1.0)

    def test_slow_creep_under_5_mm_is_still(self):
        t = Tracker()
        for i in range(WINDOW + 1):
            r = t.update(placed(BLACK, SL, 0.240 + 0.0004 * i), GOOD_W, PARKED)  # 4 mm in 0.5 s
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(WINDOW + 1):
            r = t.update(placed(BLACK, SL, 0.240 + (0.0006 if i % 2 else 0.0)), GOOD_W, PARKED)  # 6 mm of to and fro
        self.assertEqual(r.details[BLACK]["why"], "moving")

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        x, y, _ = L.TARGET[BLACK]
        for i in range(WINDOW + 1):
            r = t.update(proj(x, y, 0.0 if i % 2 else 1.5), GOOD_W, PARKED)  # 1.5 degrees to and fro about its base's middle
        self.assertEqual(r.details[BLACK]["why"], "moving")  # 10 x 0.65 mm at the outline's corner
        t = Tracker()
        for i in range(WINDOW + 1):
            r = t.update(proj(x, y, 0.0 if i % 2 else 1.0), GOOD_W, PARKED)  # 10 x 0.43 mm: still
        self.assertEqual(r.score, 1.0)


class Exploits(unittest.TestCase):
    def test_held_just_above_the_table_earns_nothing(self):
        b, w = placed(BLACK, SL, lift=0.002), placed(WHITE, SR, lift=0.002)
        r, _ = settled(b, w, arms=holding(b, "left", other=holding(w, "right")["right"]))
        self.assertEqual(r.score, 0.0)
        self.assertEqual({r.details[n]["why"] for n in L.PROJECTORS}, {"not_let_go"})
        # the same poses with both grippers open (or away): within the 3 mm contact tolerance, they count
        self.assertEqual(settled(b, w)[0].score, 1.0)
        self.assertEqual(settled(b, w, arms=holding(b, "left", opening=1.0))[0].score, 1.0)

    def test_both_projectors_on_one_screen_count_once(self):
        r, _ = settled(GOOD_B, placed(WHITE, SL, off=0.0, dist=0.160))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details[SR]["credit"], 0.0)
        self.assertEqual(r.details[SL]["projector"], BLACK)
        self.assertFalse(r.details[WHITE]["counted"])

    def test_a_second_projector_on_a_served_screen_adds_nothing(self):
        # whichever comes first, a screen counts only its best projector
        part_b = placed(BLACK, SL, 0.260)  # 0.25
        full_w = placed(WHITE, SL, off=0.0, dist=0.160)  # 0.5 at the same screen
        r, _ = settled(part_b, full_w)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details[SL]["projector"], WHITE)
        self.assertFalse(r.details[BLACK]["counted"])
        r, _ = settled(GOOD_B, placed(WHITE, SL, off=0.0, dist=0.180))  # 0.5 then 0.25
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details[SL]["projector"], BLACK)

    def test_the_right_distance_for_the_other_ratio(self):
        # the black projector at the white one's distance (picture 107 mm) and vice versa: nothing
        self.assertEqual(scored(placed(BLACK, SL, 0.160), START_W), 0.0)
        self.assertEqual(scored(START_B, placed(WHITE, SR, 0.240)), 0.0)

    def test_backwards_at_the_mirrored_distance(self):
        # turned the wrong way round with its body where a correct one would be: the lens points at the arms
        s = proj(*L.TARGET[BLACK][:2], 180.0)
        self.assertEqual(why(s), "lens_not_toward_the_wall")
        self.assertEqual(scored(s, START_W), 0.0)

    def test_pushed_against_the_wall_edge_of_the_table(self):
        s = placed(BLACK, SL, L.SCREEN_FACE_X - L.TABLE_X[1] + 0.002)  # lens over the table's far edge: 100 mm
        self.assertEqual(why(s), "picture_too_narrow")


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        t = Tracker()
        settled(GOOD_B, GOOD_W, t)
        bad = []
        for field_name in ("origin", "quat_xyzw", "pos", "low"):
            s = proj(*L.TARGET[BLACK])
            setattr(s, field_name, np.full(np.asarray(getattr(s, field_name)).shape, np.nan))
            bad.append(s)
        z = proj(*L.TARGET[BLACK])
        z.quat_xyzw = np.zeros(4)
        bad.append(z)
        inf = proj(*L.TARGET[BLACK])
        inf.lin_vel = np.array([np.inf, 0, 0])
        bad.append(inf)
        hist = [list(t.history[n]) for n in L.PROJECTORS]
        for s in bad:
            self.assertFalse(valid(s))
            r = t.update(s, GOOD_W, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual([len(list(t.history[n])) for n in L.PROJECTORS], [len(h) for h in hist])
        self.assertEqual(t.update(GOOD_B, GOOD_W, PARKED).score, 1.0)  # back to normal at once

    def test_missing_fields_or_objects(self):
        s = SimpleNamespace(origin=np.zeros(3), quat_xyzw=np.array([0, 0, 0, 1.0]))
        self.assertFalse(valid(s))
        self.assertFalse(valid(None))
        t = Tracker()
        self.assertEqual(t.update(None, GOOD_W, PARKED).score, 0.0)
        self.assertEqual(t.result(GOOD_B, None, PARKED).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = placed(BLACK, SL)
        s.quat_xyzw = np.asarray(s.quat_xyzw) * 3.0
        self.assertEqual(scored(s), 1.0)

    def test_lens_pointing_straight_up(self):
        s = proj(*L.TARGET[BLACK][:2], 0.0, pitch=-90.0)
        self.assertTrue(math.isnan(heading_rad(s)))
        self.assertEqual(square_err_deg(s), 180.0)
        self.assertEqual(why(s), "not_upright")


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        r, t = settled(GOOD_B, GOOD_W)
        self.assertEqual(r.score, 1.0)
        n = t.steps
        t.update(START_B, START_W, PARKED)
        self.assertEqual(t.steps, 1)
        self.assertLess(t.steps, n)
        self.assertEqual(t.update(GOOD_B, GOOD_W, PARKED).score, 0.0)  # half a second must pass again

    def test_teleport_to_a_jittered_start_resets(self):
        _, t = settled(GOOD_B, GOOD_W)
        jb = proj(L.START[BLACK][0] + 0.02, L.START[BLACK][1] - 0.02, math.degrees(L.START[BLACK][2]) + 5.0)
        t.update(jb, START_W, PARKED)
        self.assertEqual(t.steps, 1)

    def test_reset_when_one_projector_had_not_moved(self):
        _, t = settled(GOOD_B, START_W)
        t.update(START_B, START_W, PARKED)
        self.assertEqual(t.steps, 1)

    def test_one_projector_carried_home_is_not_a_reset(self):
        t = Tracker()
        settled(GOOD_B, GOOD_W, t)
        n = t.steps
        t.update(GOOD_B, START_W, PARKED)  # only the white one jumps: not both at their marks
        self.assertEqual(t.steps, n + 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        settled(START_B, START_W, t)
        a, b = np.array(L.TARGET[BLACK][:2]), np.array(L.START[BLACK][:2])
        for k in np.linspace(0.0, 1.0, 40):
            p = a + k * (b - a)
            t.update(proj(p[0], p[1], math.degrees(L.START[BLACK][2])), START_W, PARKED)
        self.assertGreater(t.steps, 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = Tracker()
        settled(START_B, START_W, t)
        n = t.steps
        t.update(GOOD_B, GOOD_W, PARKED)
        self.assertEqual(t.steps, n + 1)

    def test_explicit_reset(self):
        _, t = settled(GOOD_B, GOOD_W)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertTrue(all(len(t.history[n]) == 0 for n in L.PROJECTORS))
        self.assertEqual(t.update(GOOD_B, GOOD_W, PARKED).score, 0.0)


class LetGo(unittest.TestCase):
    """A projector counts only once both grippers have let go of it: at least 90 % open, or the finger zone more than
    10 mm from it. Unreadable arms: nothing is let go."""

    def test_zone_radius(self):
        self.assertAlmostEqual(zone_radius(1.0), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(0.0), math.hypot(0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(2.0), zone_radius(1.0))  # clipped

    def test_held_by_either_gripper_earns_nothing(self):
        for side in ("left", "right"):
            r, _ = settled(GOOD_B, GOOD_W, arms=holding(GOOD_B, side))
            self.assertEqual(r.score, 0.5, side)
            self.assertEqual(r.details[BLACK]["why"], "not_let_go")
            self.assertFalse(r.details[BLACK]["let_go"])
            self.assertTrue(r.details[WHITE]["let_go"])
            self.assertFalse(r.success)

    def test_opening_edge(self):
        for opening, want in ((L.GRIPPER_OPEN - 0.001, 0.5), (L.GRIPPER_OPEN, 1.0), (1.0, 1.0)):
            self.assertEqual(settled(GOOD_B, GOOD_W, arms=holding(GOOD_B, opening=opening))[0].score, want, opening)

    def test_closed_gripper_beside_it(self):
        """A closed gripper pointing down beside the black projector's left side (y +): let go only when its finger
        zone (15.7 mm round its axis) is more than 10 mm from the body."""
        side_y = L.SCREEN_Y[SL] + L.BODY_W / 2
        z = L.TABLE_TOP_W + L.BODY_H / 2
        for extra, want in ((-0.0005, 0.5), (0.0005, 1.0)):
            a = arm((L.TARGET[BLACK][0], side_y + zone_radius(0.0) + L.LET_GO_GAP + extra, z), 0.0)
            arms = dict(PARKED, left=a)
            self.assertAlmostEqual(zone_gap(read_arms(arms)[0], GOOD_B), L.LET_GO_GAP + extra, places=9)
            self.assertEqual(settled(GOOD_B, GOOD_W, arms=arms)[0].score, want, extra)

    def test_closed_gripper_far_away_has_let_go(self):
        arms = {"left": arm((0.200, 0.300, L.TOP + 0.137), 0.0), "right": arm((0.200, -0.300, L.TOP + 0.137), 0.0)}
        self.assertEqual(settled(GOOD_B, GOOD_W, arms=arms)[0].score, 1.0)

    def test_the_lens_counts_as_the_projector(self):
        """A closed gripper on the lens alone (in front of the body) holds it."""
        tip = lens_point(GOOD_B) + [0.002, 0.0, 0.0]
        arms = dict(PARKED, left=arm(tip, 0.2))
        self.assertLess(zone_gap(read_arms(arms)[0], GOOD_B), 0.0)
        self.assertEqual(settled(GOOD_B, GOOD_W, arms=arms)[0].details[BLACK]["why"], "not_let_go")

    def test_distances_follow_the_projectors_turn(self):
        """The black projector at its start (lens toward -y): its lens tip touches y 0.132, its back face is at
        y 0.163."""
        from scoring import shape_distances
        z = L.TABLE_TOP_W + L.LENS_Z
        d = shape_distances([(0.300, 0.150 - L.LENS_FRONT_U, z), (0.300, 0.150 + L.BODY_D / 2 + 0.005, z),
                             (0.300 + L.BODY_W / 2 + 0.004, 0.150, z)], START_B)
        np.testing.assert_allclose(d, [0.0, 0.005, 0.004], atol=1e-9)

    def test_the_zone_runs_up_the_fingers(self):
        """A closed gripper straight over the projector, its finger tips 30 mm above the top: the zone (15.7 mm
        round the axis from the tips upward) is 14.3 mm clear, so it has let go; 25 mm above: 9.3 mm, held."""
        top = L.TABLE_TOP_W + L.BODY_H
        for above, want in ((0.030, True), (0.025, False)):
            arms = dict(PARKED, left=arm((*GOOD_B.origin[:2], top + above), 0.0))
            self.assertEqual(released(read_arms(arms), GOOD_B), want, above)

    def test_unreadable_arms_let_nothing_go(self):
        bad = [None, {}, {"left": PARKED["left"]}, "arms",
               dict(PARKED, left=rigstate.Arm(tcp=PARKED["left"].tcp, tips=PARKED["left"].tips, opening=math.nan)),
               dict(PARKED, right=rigstate.Arm(tcp=np.array([np.inf, 0, 0]), tips=PARKED["right"].tips, opening=1.0)),
               dict(PARKED, right=SimpleNamespace(tcp=np.zeros(2), tips=np.zeros(3), opening=1.0))]
        for arms in bad:
            self.assertIsNone(read_arms(arms))
            self.assertFalse(released(read_arms(arms), GOOD_B))
            r, _ = settled(GOOD_B, GOOD_W, arms=arms)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["arms_read"])
            self.assertEqual({r.details[n]["why"] for n in L.PROJECTORS}, {"not_let_go"})
        # back to readable arms: counts again at once (stillness history is unaffected)
        _, t = settled(GOOD_B, GOOD_W, arms=None)
        self.assertEqual(t.update(GOOD_B, GOOD_W, PARKED).score, 1.0)

    def test_let_go_is_checked_after_standing_and_reported(self):
        r, _ = settled(placed(BLACK, SL, lift=0.010), GOOD_W, arms=holding(GOOD_B))
        self.assertEqual(r.details[BLACK]["why"], "not_on_the_table")  # lifted 10 mm: the first reason given
        r, _ = settled(GOOD_B, GOOD_W)
        self.assertTrue(r.details["arms_read"])
        self.assertEqual(r.details["openings"], {"left": 1.0, "right": 1.0})
        self.assertEqual(set(r.details[BLACK]["zone_gap_mm"]), {"left", "right"})
        self.assertTrue(let_go(read_arms(PARKED)[0], GOOD_B))


def _stub_dir():
    """tools/offline/yam_stub, found by walking up from this task (the repository layout)."""
    for parent in HERE.parents:
        d = parent / "tools" / "offline" / "yam_stub"
        if d.is_dir():
            return d
    return None


def _boxes_overlap(a, b) -> float:
    """Overlap volume of two axis-aligned boxes given as (centre, size)."""
    (ca, sa), (cb, sb) = a, b
    d = [min(ca[i] + sa[i] / 2, cb[i] + sb[i] / 2) - max(ca[i] - sa[i] / 2, cb[i] - sb[i] / 2) for i in range(3)]
    return max(d[0], 0.0) * max(d[1], 0.0) * max(d[2], 0.0)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the
    demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        sys.path.insert(0, str(_stub_dir().parent))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit
        cls.specs = {BLACK: cls.task.PROJECTOR_BLACK, WHITE: cls.task.PROJECTOR_WHITE}

    def setUp(self):
        rigstate.offline_feed({a: {"tcp": PARKED[a].tcp, "tips": PARKED[a].tips, "opening": 1.0} for a in PARKED})

    def sim(self, black, white):
        """Synthetic states turned into what mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in ((BLACK, black), (WHITE, white))}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], [BLACK, WHITE])
        self.assertTrue(all(o.mass <= 0.30 for o in loose))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        # the scorer's planes and lines are the objects': screen faces, centre lines, the table's top
        for box, s in zip(t.SCREEN_BOXES, L.SCREENS):
            self.assertAlmostEqual(box.pos[0] - box.size[0] / 2, L.SCREEN_FACE_X)
            self.assertAlmostEqual(box.pos[1], L.SCREEN_Y[s])
            self.assertAlmostEqual(box.size[1], L.SCREEN_W)
            self.assertLessEqual(box.pos[0] + box.size[0] / 2, L.WALL_X[0] + 1e-12)  # flat on the wall's face
        for box, s in zip(t.MARKS, L.SCREENS):
            self.assertAlmostEqual(box.pos[1], L.SCREEN_Y[s])
            self.assertLess(box.pos[2] + box.size[2] / 2, t.SCREEN_BOXES[0].pos[2] - t.SCREEN_BOXES[0].size[2] / 2)
        tb = t.PROJECTOR_TABLE
        self.assertAlmostEqual(tb.pos[2] + tb.size[2] / 2, L.TABLE_TOP_W)
        self.assertAlmostEqual(tb.pos[0] - tb.size[0] / 2, L.TABLE_X[0])
        self.assertAlmostEqual(tb.pos[0] + tb.size[0] / 2, L.TABLE_X[1])
        self.assertAlmostEqual(tb.pos[1] + tb.size[1] / 2, L.TABLE_Y[1])
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, o.size, 0.0)
            hx = abs(math.cos(yaw)) * size[0] / 2 + abs(math.sin(yaw)) * size[1] / 2
            hy = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            if c[0] - hx < 0.2 - 1e-9:
                self.assertLess(abs(c[1]) + hy, 0.45, o.name)

    def test_projector_geometry_matches_the_scorer(self):
        for name, spec in self.specs.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-L.FOOT_BACK, -L.FOOT_V, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [L.FOOT_FRONT, L.FOOT_V, L.BODY_H], atol=1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0] - self.task.GRIP_U), 0.001)  # the grip is within 1 mm of the centre of mass
            self.assertAlmostEqual(com[1], 0.0, places=12)
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [*L.START[name][:2], L.TABLE_TOP_W], atol=1e-12)
            self.assertTrue(Tracker.at_start(name, rest))
        shapes = [(c, s) for _, c, s, _ in L.SHAPES]
        self.assertLess(_boxes_overlap(*shapes), 1e-12)
        body, lens = shapes
        self.assertAlmostEqual(lens[0][0] - lens[1][0] / 2, body[0][0] + body[1][0] / 2)  # on the front face
        self.assertAlmostEqual(lens[0][0] + lens[1][0] / 2, L.LENS_FRONT_U)
        self.assertAlmostEqual(lens[0][2], L.LENS_Z)
        self.assertLess(lens[1][1], body[1][1])  # narrower than the body: between the fingers of a grip across it
        # each lens contrasts with its body (seen from the camera)
        for name in L.PROJECTORS:
            b, le = np.array(L.COLOURS[L.BODY_COLOUR[name]]), np.array(L.COLOURS[L.LENS_COLOUR[name]])
            self.assertGreater(abs(b.mean() - le.mean()), 0.5, name)
        sim = self.sim(GOOD_B, GOOD_W)
        self.assertAlmostEqual(float(sim[BLACK].low[2]), L.TABLE_TOP_W, places=9)
        self.assertEqual(scored(sim[BLACK], sim[WHITE]), 1.0)
        self.assertAlmostEqual(throw_distance(sim[BLACK]), 0.240, places=9)

    def test_offline_state_path(self):
        st = {}
        for name, spec in self.specs.items():
            c, _, yaw = spec.offline_box()
            box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                                  ang_vel=np.zeros(3))
            st[name] = spec.offline_state(box)
            self.assertEqual(st[name].source, "offline")
            np.testing.assert_allclose(st[name].origin, [*L.START[name][:2], L.TABLE_TOP_W], atol=1e-12)
        self.assertEqual(scored(st[BLACK], st[WHITE]), 0.0)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START_B, START_W)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(GOOD_B, GOOD_W)
        for _ in range(WINDOW):
            self.task.success(done)
        self.assertTrue(self.task.success(done))
        self.assertEqual(self.task.score(done), 1.0)
        self.assertEqual(self.task.score(done), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(self.task.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        self.assertEqual(self.task.score({BLACK: done[BLACK]}), 0.0)  # a missing projector scores 0
        # the hooks read rigstate's arms: a gripper still closed on the black projector, or no arms at all, and it
        # does not count
        h = holding(GOOD_B)
        rigstate.offline_feed({a: {"tcp": h[a].tcp, "tips": h[a].tips, "opening": h[a].opening} for a in h})
        for _ in range(WINDOW):
            self.assertFalse(self.task.success(done))
        self.assertEqual(self.task.score(done), 0.5)
        rigstate.offline_reset()
        self.assertIsNone(rigstate.arms())
        self.assertEqual(self.task.score(done), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(GOOD_B, GOOD_W))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "valid_state", BLACK, WHITE, SL, SR,
                          "origins", "quats", "low_z", "vels", "source", "episode", "arms_read", "openings",
                          "left_tcp", "right_tcp", "left_tips", "right_tips"):
                    self.assertIn(k, row)
                for k in ("credit", "why", "screen", "throw_mm", "picture_mm", "width_err_pc", "square_err_deg",
                          "nearest_screen", "centre_off_mm", "origin_mm", "tilt_deg", "lowest_mm", "on_table",
                          "path_mm", "let_go", "zone_gap_mm", "counted"):
                    self.assertIn(k, row[BLACK])
                self.assertEqual(row[BLACK]["throw_mm"], 240.0)
                self.assertEqual(row[WHITE]["picture_mm"], 160.0)
                self.assertEqual(row[BLACK]["why"], "moving")  # one step seen: not yet still
                rigstate.offline_reset()  # no arms: the row says why
                self.task.success(self.sim(GOOD_B, GOOD_W))
                row = json.loads(path.read_text().splitlines()[-1])
                self.assertIn("rigstate_error", row)
                self.assertFalse(row["arms_read"])
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(GOOD_B, GOOD_W)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes 1.5 s, standing still
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertEqual(tuple(prev.pos), tuple(w.pos), (arm, w.label))
                if w.pos[2] < prev.pos[2] and w.pos[2] < t.APPROACH_Z - 1e-9:  # slow last descents
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.03 / 1.5 + 1e-9, (arm, w.label))
            name = t.SIDE[arm]
            grip = [w for w in wps if w.label == "grip across the body"][0]
            np.testing.assert_allclose(grip.pos[:2], L.START[name][:2], atol=1e-12)
            put = [w for w in wps if w.label == "let go"][0]
            np.testing.assert_allclose(put.pos[:2], L.TARGET[name][:2], atol=1e-12)
            # the wrist turns the lens to the wall: +90 degrees for the black (lens -y -> +x), -90 for the white
            self.assertAlmostEqual(put.yaw - grip.yaw, L.TARGET[name][2] - L.START[name][2])
            self.assertAlmostEqual(abs(math.degrees(grip.yaw)), 90.0)
            self.assertAlmostEqual(put.yaw, 0.0)
        self.assertLess(plan["left"][4].yaw, 0.0)  # the left arm turns from -90, the right from +90
        self.assertGreater(plan["right"][4].yaw, 0.0)
        self.assertAlmostEqual(t.GRIP, -0.013)  # 26 mm commanded on the 34 mm body
        self.assertAlmostEqual(t.PLACE_Z - t.GRIP_Z, 0.002)
        self.assertAlmostEqual(t.GRIP_Z - 0.013 - L.TABLE_TOP_W, 0.004)  # pad tips 4 mm above the table
        self.assertLess(t.GRIP_Z - 0.013, L.TABLE_TOP_W + L.BODY_H)  # ... and below the body's top

    def test_demo_counts_only_after_letting_go(self):
        """Replays the demo's place: with the fingers closed on the projector (lowered and settled) the projector
        does not count; once 'let go' has opened them (94 mm: opening 1.0) it does."""
        t = self.task
        for arm_name, wps in t.demo().items():
            name = t.SIDE[arm_name]
            lo = [w for w in wps if w.label == "lower it slowly"][0]
            go = [w for w in wps if w.label == "let go"][0]
            p = placed(name, SL if name == BLACK else SR)
            for w, want in ((lo, False), (go, True)):
                tips = np.array(w.pos) - [0.0, 0.0, 0.013]
                if w.finger == t.GRIP:
                    opening = L.BODY_W / L.FULL_GAP  # the pads stop on the 34 mm body
                else:
                    opening = w.finger / rigstate.FINGER_OPEN
                arms = dict(PARKED)
                arms[arm_name] = arm(tips, opening)
                self.assertEqual(released(read_arms(arms), p), want, (arm_name, w.label))
        self.assertAlmostEqual(t.OPEN / rigstate.FINGER_OPEN, 1.0)

    @staticmethod
    def _sample(wps, dt=0.05):
        """(t, TCP position, yaw) of an arm's plan, interpolated as ArmPlan does (smoothstep in time)."""
        out, t, prev = [], 0.0, None
        for w in wps:
            if prev is not None:
                n = max(1, int(round(w.duration / dt)))
                for i in range(1, n + 1):
                    s = i / n
                    k = 3 * s * s - 2 * s * s * s
                    p = (1 - k) * np.array(prev.pos) + k * np.array(w.pos)
                    yaw = prev.yaw + k * math.remainder(w.yaw - prev.yaw, 2 * math.pi)
                    out.append((t + s * w.duration, p, yaw, w.label))
            t += w.duration
            prev = w
        return out

    def test_carry_path_stays_over_the_table_and_apart(self):
        """Replays both arms' carries (the projector rigid in the grip, turning with the wrist): each stays over the
        table, the two outlines stay apart, and the lens ends at the wall."""
        t = self.task
        paths = {}
        for arm in ("left", "right"):
            wps = t.demo()[arm]
            a = [i for i, w in enumerate(wps) if w.label == "lift 20 mm"][0]
            b = [i for i, w in enumerate(wps) if w.label == "lower it slowly"][0]
            name = t.SIDE[arm]
            h0, y0 = L.START[name][2], wps[a].yaw
            seq = []
            for tt, p, yaw, _ in self._sample(wps[a:b + 1]):
                h = h0 + (yaw - y0)
                seq.append((round(tt, 3), proj(p[0], p[1], math.degrees(h))))
            paths[arm] = seq
            self.assertAlmostEqual(math.remainder(h - L.TARGET[name][2], 2 * math.pi), 0.0, places=9)
        gap = 1.0
        for (tl, pl), (tr, pr) in zip(paths["left"], paths["right"]):
            self.assertAlmostEqual(tl, tr)  # both arms on one clock: the same moment
            for p in (pl, pr):
                self.assertTrue(over_table(p))
                self.assertGreater(p.low[0], L.TABLE_X[0] + 0.020)
                self.assertLess(p.high[0], L.TABLE_X[1])
                self.assertLess(abs(p.high[1]), L.TABLE_Y[1] - 0.030)
            gap = min(gap, float(pl.low[1] - pr.high[1]))
        self.assertGreater(gap, 0.18)

    def test_tcps_stay_apart(self):
        t = self.task
        plan = t.demo()
        sl, sr = self._sample(plan["left"]), self._sample(plan["right"])
        self.assertEqual(len(sl), len(sr))
        self.assertGreaterEqual(min(float(np.linalg.norm(a[1] - b[1])) for a, b in zip(sl, sr)), 0.25)

    def test_one_arm_switches(self):
        t = self.task
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule_and_matches_everywhere(self):
        ins = self.task.INSTRUCTION
        for s in ("one projector per screen", "160 mm wide", "1.6 m", "1:10", "1.5:1", "1.0:1", "black", "white",
                  "throw ratio x picture width", "from the front of the lens", "upright on the table",
                  "square to the wall", "centre line", "just fills the screen"):
            self.assertIn(s, ins)
        self.assertLessEqual(len(ins), 490)
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], ins)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{ins}"'), 2)

    def test_task_json_fields(self):
        spec = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "context", "setup", "scoring", "run_limit", "fidelity",
                  "reset_notes", "scores"):
            self.assertTrue(spec.get(k), k)
        self.assertEqual(spec["category"], "Impact on jobs")
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150,
                  "fidelity": 490, "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(spec[k]), n, k)
        self.assertIn("1,200 steps", spec["run_limit"])
        self.assertIn("demo", spec["scores"])

    def test_task_json_states_the_codes_numbers(self):
        spec = json.loads((HERE / "task.json").read_text())
        sc, su = spec["scoring"], spec["setup"]
        for s in ("5 degrees", "within 3 mm", "5 mm over the last 0.5 s", "10 degrees", "6 mm", "12 mm", "5 %",
                  "15 %", "152 to 168 mm", "136 to 184 mm", "240 mm", "160 mm", "0.25", "0.5", "0.75", "1.0", "3 s",
                  "x 46.8"):
            self.assertIn(s, sc)
        self.assertEqual((L.TILT_MAX_DEG, L.TABLE_TOL, L.STILL_PATH), (5.0, 0.003, 0.005))
        self.assertEqual((L.SQUARE_DEG, L.AIM_DEG, L.CENTRE_TIGHT, L.CENTRE_LOOSE), (5.0, 10.0, 0.006, 0.012))
        self.assertEqual((L.FULL_TOL, L.LOOSE_TOL), (0.05, 0.15))
        for s in ("x 46.8", "x 47.0 to 48.0", "y -25.0 to +25.0", "190 mm", "160 mm wide", "100 mm tall",
                  "y +12.0", "y -12.0", "x 17.0 to 37.0", "y -20.0 to +20.0", "75 mm", "50 g", "34 mm", "26 mm",
                  "14 mm", "5 mm", "x 30.0", "y +15.0", "y -15.0", "1.5:1", "1.0:1"):
            self.assertIn(s, su)
        nums = {float(v) for v in re.findall(r"(\d+(?:\.\d+)?) mm", sc)}
        self.assertTrue({3.0, 5.0, 6.0, 12.0, 240.0, 160.0} <= nums)
        # 'let go': the rule's numbers as task.json and the README state them
        for s in ("let go by both grippers", "90 % open", "fingers 84.5 mm apart", "more than 10 mm", "38 mm up",
                  "radius 16 mm closed, 57 mm open", "nothing is let go", "still held"):
            self.assertIn(s, sc)
        self.assertEqual((L.GRIPPER_OPEN, L.LET_GO_GAP, L.FINGER_LEN), (0.9, 0.010, 0.038))
        self.assertAlmostEqual(L.GRIPPER_OPEN * L.FULL_GAP, 0.0845, places=4)
        self.assertEqual((round(zone_radius(0.0) * 1000), round(zone_radius(1.0) * 1000)), (16, 57))
        readme = (HERE / "README.md").read_text()
        for s in ("90 % open", "84.5 mm", "more than 10 mm", "16 mm closed, 57 mm fully open", "rigstate.py"):
            self.assertIn(s, readme)
        self.assertNotIn("counts as standing", spec["fidelity"] + readme)

    def test_neither_arm_reaches_the_other_projector(self):
        """README 'One arm alone': straight down, from the table up to the carry height, neither arm reaches the
        other arm's projector where it starts, at any yaw (joint 6 allows +-120 degrees)."""
        import yamkin as K
        for arm, other in (("left", WHITE), ("right", BLACK)):
            base = K.BASES["station"][arm]
            x, y, _ = L.START[other]
            for dy in (0.0, -math.copysign(L.LENS_FRONT_U, y)):  # its centre and its lens's tip (toward the middle)
                for z in (0.085, self.task.GRIP_Z - L.TOP, self.task.CARRY_Z - L.TOP):
                    for yaw in range(-120, 121, 15):
                        cell, _ = K.reach_cell((x, y + dy, L.TOP + z), math.radians(yaw), base, L.TOP, None)
                        self.assertEqual(cell, 0, (arm, dy, z, yaw))

    def test_the_wrong_turn_is_as_reachable_as_the_right_one(self):
        """The direction of the quarter turn is the policy's choice: gripped at +90 degrees instead of -90 (the left
        arm), the same wrist turn to 0 points the black projector's lens at the arms."""
        import yamkin as K
        for arm, name, wrong in (("left", BLACK, 90.0), ("right", WHITE, -90.0)):
            base = K.BASES["station"][arm]
            x, y, _ = L.START[name]
            cell, _ = K.reach_cell((x, y, self.task.GRIP_Z), math.radians(wrong), base, L.TOP, None)
            self.assertEqual(cell, 2, arm)
            tx, ty, _ = L.TARGET[name]
            cell, _ = K.reach_cell((tx, ty, self.task.GRIP_Z), 0.0, base, L.TOP, None)
            self.assertEqual(cell, 2, arm)
        backwards = proj(*L.TARGET[BLACK][:2], 180.0)
        self.assertEqual(why(backwards), "lens_not_toward_the_wall")


if __name__ == "__main__":
    unittest.main()
