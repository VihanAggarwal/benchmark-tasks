"""Offline tests for scoring.py and task.py: synthetic wedge states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    WEDGES,
    Tracker,
    credit,
    deck_margin,
    evaluate,
    wedge_ends,
    facing_rad,
    placement,
    tilt_deg,
    valid,
    wedge_corners,
    wedge_pose,
)

HZ = 20.0
WINDOW = 10  # 0.5 s at 20 Hz (written out, not from layout.STILL_WINDOW_S)
TOP = 0.7475
ON = TOP + 0.003  # the deck's top: a wedge standing on the deck (written out, not from layout.DECK_TOP_W)
STANDS = {"left": (0.390, 0.150), "right": (0.390, -0.150)}  # the mic stands' centres (straight under the heads)
STOOLS = {"left": (0.430, 0.150), "right": (0.430, -0.150)}  # the singers' stools
NULLS = {"left": 180.0, "right": 126.0}  # cardioid, supercardioid
DECK = (0.140, 0.470, -0.340, 0.340)  # x0, x1, y0, y1


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


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _corners(c, s):
    return [(c[0] + a * s[0] / 2, c[1] + b * s[1] / 2, c[2] + e * s[2] / 2) for a in (-1, 1) for b in (-1, 1)
            for e in (-1, 1)]


# The wedge's shapes, written out again here from the task's description (not from scoring.py or layout.py): a black
# cabinet 45 mm deep (local x, the way the grille faces), 60 mm wide and 30 mm tall, stepped: a full-depth slab 12 mm
# tall, a rear block 20 mm deep to the full 30 mm, and a grey grille 56 mm wide over the step's top (25 mm deep, 2 mm
# thick) and up the rear block's front face (2 mm thick, from 14 to 28 mm up). Base frame at the footprint's middle.
LOCAL = np.array(
    _corners((0.0, 0.0, 0.006), (0.045, 0.060, 0.012))
    + _corners((-0.0125, 0.0, 0.021), (0.020, 0.060, 0.018))
    + _corners((0.010, 0.0, 0.013), (0.025, 0.056, 0.002))
    + _corners((-0.0015, 0.0, 0.021), (0.002, 0.056, 0.014))
)


def _sorted(points):
    """Points as a sorted list of tuples, rounded to 1e-9 m so float noise cannot change the order."""
    return sorted(tuple(round(float(v), 9) for v in p) for p in np.asarray(points))


def wdg(cx, cy, heading=0.0, bottom=ON, pitch=0.0, roll=0.0):
    """A wedge as mechkit reports it: ``heading`` is the way its grille faces, seen from above (0: +x, upstage; 90:
    +y, left). Its base frame's origin is over (cx, cy) (the middle of its footprint) and its lowest corner at
    ``bottom`` (world z)."""
    q = quat(heading, pitch, roll)
    r = _rot(q)
    o = np.array([cx, cy, 0.0])
    o[2] = bottom - float((LOCAL @ r.T)[:, 2].min())
    c = o + LOCAL @ r.T
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(c.min(0) + c.max(0)) / 2, lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3), low=c.min(0), high=c.max(0), source="test")


def at(singer, r, beta, side=None, aim_off=0.0, **kw):
    """A wedge r from that singer's stand, beta degrees round from her mic's front (+x), on the +y side (side +1) or
    the -y side (-1; default: the right singer's outer side, -y), its grille aimed at her stool turned aim_off."""
    sx, sy = STANDS[singer]
    if side is None:
        side = -1.0 if singer == "right" else 1.0
    a = math.radians(beta) * side
    x, y = sx + r * math.cos(a), sy + r * math.sin(a)
    px, py = STOOLS[singer]
    return wdg(x, y, math.degrees(math.atan2(py - y, px - x)) + aim_off, **kw)


def state(**kw):
    """The finished layout: the left wedge 110 mm straight behind the cardioid, the right one 110 mm out at 126
    degrees from the supercardioid (outer side), each aimed at its singer's stool."""
    s = {"wedge_left": at("left", 0.110, 180.0), "wedge_right": at("right", 0.110, 126.0)}
    s.update(kw)
    return s


START = {
    "wedge_left": wdg(0.220, 0.390, 90.0, bottom=TOP),
    "wedge_right": wdg(0.220, -0.390, -90.0, bottom=TOP),
}


def feed(tr, states):
    r = None
    for st in states:
        r = tr.update(st)
    return r


def settled(st, n=WINDOW + 1):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n)


def scored(st):
    return settled(st)[1].score


def detail(st):
    return settled(st)[1].details


# An independent scorer, from the task's text (for the random-state cross-check).
def ref_credit(w):
    r = _rot(w.quat_xyzw)
    c = np.asarray(w.origin) + LOCAL @ r.T
    if abs(c[:, 2].min() - ON) > 0.003 or math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2])))) > 10.0:
        return None, 0.0
    if not (np.all(c[:, 0] >= DECK[0]) and np.all(c[:, 0] <= DECK[1]) and np.all(c[:, 1] >= DECK[2])
            and np.all(c[:, 1] <= DECK[3])):
        return None, 0.0
    ox, oy = w.origin[0], w.origin[1]
    s = min(STANDS, key=lambda k: math.hypot(ox - STANDS[k][0], oy - STANDS[k][1]))
    sx, sy = STANDS[s]
    dist = math.hypot(ox - sx, oy - sy)
    beta = math.degrees(math.acos(max(-1.0, min(1.0, (ox - sx) / dist)))) if dist > 0 else 0.0
    off = abs(beta - NULLS[s])
    fx, fy = r[0, 0], r[1, 0]
    tx, ty = STOOLS[s][0] - ox, STOOLS[s][1] - oy
    cosang = (fx * tx + fy * ty) / (math.hypot(fx, fy) * math.hypot(tx, ty))
    aim = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    if 0.080 <= dist <= 0.150 and off <= 15.0 and aim <= 20.0:
        return s, 0.5
    if 0.060 <= dist <= 0.200 and off <= 30.0 and aim <= 45.0:
        return s, 0.25
    return s, 0.0


def ref_score(st):
    best = {"left": 0.0, "right": 0.0}
    for n in ("wedge_left", "wedge_right"):
        s, c = ref_credit(st[n])
        if s is not None:
            best[s] = max(best[s], c)
    return best["left"] + best["right"]


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(Tracker.at_start(START))
        self.assertEqual(scored(START), 0.0)
        d = detail(START)
        for n in WEDGES:
            self.assertFalse(d[n]["on_pose"])  # on the table in the wings, not on the deck
            self.assertLess(d[n]["deck_margin_mm"], 0.0)
        self.assertEqual(d["zero_reason"], "left_no_wedge,right_no_wedge")

    def test_placement_measures(self):
        r, beta, aim = placement(at("left", 0.110, 180.0), "left")
        self.assertAlmostEqual(r, 0.110, places=9)
        self.assertAlmostEqual(beta, 180.0, places=6)
        self.assertAlmostEqual(aim, 0.0, places=6)
        r, beta, aim = placement(at("right", 0.100, 126.0), "right")
        self.assertAlmostEqual(r, 0.100, places=9)
        self.assertAlmostEqual(beta, 126.0, places=6)
        self.assertAlmostEqual(aim, 0.0, places=6)
        _, beta, _ = placement(at("right", 0.100, 126.0, side=1.0), "right")  # the inner side: the same angle
        self.assertAlmostEqual(beta, 126.0, places=6)
        _, beta, _ = placement(at("left", 0.100, 90.0), "left")
        self.assertAlmostEqual(beta, 90.0, places=6)
        _, _, aim = placement(at("left", 0.110, 180.0, aim_off=-33.0), "left")
        self.assertAlmostEqual(aim, 33.0, places=6)
        _, _, aim = placement(at("left", 0.110, 180.0, aim_off=180.0), "left")  # grille to the audience
        self.assertAlmostEqual(aim, 180.0, places=6)

    def test_facing_and_tilt(self):
        self.assertAlmostEqual(math.degrees(facing_rad(wdg(0.3, 0.1, 40.0))), 40.0, places=6)
        self.assertAlmostEqual(tilt_deg(wdg(0.3, 0.1, 0.0, pitch=7.0)), 7.0, places=6)
        self.assertIsNone(facing_rad(wdg(0.3, 0.1, 0.0, pitch=-90.0)))  # on its back, grille straight up
        self.assertAlmostEqual(tilt_deg(wdg(0.3, 0.1, 0.0, roll=180.0)), 180.0, places=6)

    def test_front_and_back_points(self):
        front, back = wedge_ends(wdg(0.30, -0.10, 30.0))
        self.assertAlmostEqual(float(np.hypot(*(front - back)[:2])), 0.045, places=9)  # 45 mm apart, along its depth
        heading = math.degrees(math.atan2(front[1] - back[1], front[0] - back[0]))
        self.assertAlmostEqual(heading, 30.0, places=6)  # the front point is on the grille's side

    def test_deck_margin_on_every_edge(self):
        self.assertAlmostEqual(deck_margin([[0.300, 0.0]]), 0.160, places=9)
        for pt, want in (((0.135, 0.0), -0.005), ((0.475, 0.0), -0.005), ((0.300, 0.345), -0.005),
                         ((0.300, -0.345), -0.005), ((0.145, 0.0), 0.005), ((0.300, -0.335), 0.005)):
            self.assertAlmostEqual(deck_margin([pt]), want, places=9, msg=str(pt))
        downstage = wdg(0.155, 0.150, 0.0)  # over the deck's front edge: its back hangs 7.5 mm past it
        self.assertAlmostEqual(deck_margin(wedge_corners(downstage)[:, :2]), -0.0075, places=9)
        self.assertFalse(wedge_pose(downstage)["on_pose"])

    def test_scorer_shape_matches_layout(self):
        np.testing.assert_allclose(_sorted(wedge_corners(wdg(0.0, 0.0, 0.0, bottom=0.0))), _sorted(LOCAL), atol=1e-9)
        self.assertEqual((L.DECK_X0, L.DECK_X1, L.DECK_Y0, L.DECK_Y1), DECK)
        for s in ("left", "right"):
            self.assertEqual(L.SINGERS[s][:2], STANDS[s])
            np.testing.assert_allclose(L.STOOLS[s], STOOLS[s], atol=1e-12)
            self.assertEqual(L.SINGERS[s][3], NULLS[s])
        self.assertAlmostEqual(L.DECK_TOP_W, ON)

    def test_credit_bands(self):
        self.assertEqual(credit(0.110, 0.0, 0.0), 0.5)
        for r, want in ((0.0795, 0.25), (0.0805, 0.5), (0.1495, 0.5), (0.1505, 0.25), (0.0595, 0.0), (0.0605, 0.25),
                        (0.1995, 0.25), (0.2005, 0.0)):
            self.assertEqual(credit(r, 0.0, 0.0), want, r)
        for off, want in ((14.9, 0.5), (15.1, 0.25), (29.9, 0.25), (30.1, 0.0)):
            self.assertEqual(credit(0.110, off, 0.0), want, off)
        for aim, want in ((19.9, 0.5), (20.1, 0.25), (44.9, 0.25), (45.1, 0.0)):
            self.assertEqual(credit(0.110, 0.0, aim), want, aim)
        self.assertEqual(credit(0.070, 20.0, 30.0), 0.25)
        self.assertEqual(credit(0.070, 31.0, 0.0), 0.0)


class Tiers(unittest.TestCase):
    def test_finished_scores_one(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["zero_reason"], "")
        self.assertEqual(r.details["left"]["best"], "wedge_left")
        self.assertEqual(r.details["right"]["best"], "wedge_right")

    def test_either_wedge_for_either_singer_and_either_side(self):
        swapped = {"wedge_left": at("right", 0.110, 126.0), "wedge_right": at("left", 0.110, 180.0)}
        self.assertEqual(scored(swapped), 1.0)
        self.assertEqual(scored(state(wedge_right=at("right", 0.110, 126.0, side=1.0))), 1.0)  # inner null
        self.assertEqual(scored(state(wedge_left=at("left", 0.090, 168.0, side=-1.0))), 1.0)  # 12 deg off, -y side

    def test_three_quarters(self):
        _, r = settled(state(wedge_right=at("right", 0.110, 145.0)))
        self.assertEqual(r.score, 0.75)  # 19 deg off its null
        self.assertFalse(r.success)  # success (which ends the trial early) is 1.0 only
        self.assertFalse(settled(state(wedge_right=at("right", 0.110, 180.0)))[1].success)
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, aim_off=30.0))), 0.75)  # aimed 30 off
        self.assertEqual(scored(state(wedge_left=at("left", 0.170, 180.0))), 0.75)  # 1.7 m away

    def test_half(self):
        # the usual mistake: both wedges straight behind their mics (54 degrees off the supercardioid's null)
        self.assertEqual(scored(state(wedge_right=at("right", 0.110, 180.0))), 0.5)
        self.assertEqual(scored(state(wedge_right=START["wedge_right"])), 0.5)  # one singer only
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, aim_off=180.0))), 0.5)  # facing the audience
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 155.0), wedge_right=at("right", 0.110, 100.0))),
                         0.5)  # both 25-26 degrees off: 0.25 + 0.25

    def test_quarter(self):
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 150.5, aim_off=30.0),
                                      wedge_right=START["wedge_right"])), 0.25)
        self.assertEqual(scored(state(wedge_left=START["wedge_left"], wedge_right=at("right", 0.065, 126.0))), 0.25)

    def test_none_scores_zero(self):
        self.assertEqual(scored(START), 0.0)
        both_wrong = state(wedge_left=at("left", 0.110, 90.0), wedge_right=at("right", 0.110, 180.0))
        self.assertEqual(scored(both_wrong), 0.0)

    def test_every_tier_is_reachable(self):
        layouts = {
            1.0: state(),
            0.75: state(wedge_right=at("right", 0.110, 145.0)),
            0.5: state(wedge_right=at("right", 0.110, 180.0)),
            0.25: state(wedge_left=at("left", 0.110, 160.0), wedge_right=at("right", 0.110, 180.0)),
            0.0: START,
        }
        for want, st in layouts.items():
            self.assertEqual(scored(st), want)
            self.assertEqual(ref_score(st), want)

    def test_score_is_idempotent(self):
        tr, r = settled(state())
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(state()).score, 1.0)
        self.assertEqual(tr.steps, n)

    def test_random_states_match_the_text(self):
        """Random wedges over and around the stage: the score is always one of 0, 0.25, ..., 1.0 and equals an
        independent scorer written from the task's text."""
        rng = random.Random(7)
        for _ in range(600):
            st = {}
            for n in WEDGES:
                if rng.random() < 0.6:  # near a singer's null band
                    s = rng.choice(("left", "right"))
                    w = at(s, rng.uniform(0.04, 0.23), rng.uniform(80.0, 180.0), side=rng.choice((-1.0, 1.0)),
                           aim_off=rng.uniform(-60.0, 60.0), bottom=ON + rng.choice((0.0, 0.0, 0.002, 0.005)),
                           pitch=rng.choice((0.0, 0.0, 5.0, 12.0)))
                else:
                    w = wdg(rng.uniform(0.10, 0.50), rng.uniform(-0.40, 0.40), rng.uniform(-180, 180),
                            bottom=rng.choice((ON, TOP)))
                st[n] = w
            got = scored(st)
            self.assertIn(got, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertEqual(got, ref_score(st), st)


class Edges(unittest.TestCase):
    def test_distance_edges(self):
        for r, want in ((0.0795, 0.75), (0.0805, 1.0), (0.1495, 1.0), (0.1505, 0.75), (0.0595, 0.5), (0.0605, 0.75),
                        (0.1995, 0.75), (0.2005, 0.5)):
            self.assertEqual(scored(state(wedge_left=at("left", r, 180.0))), want, r)

    def test_null_edges(self):
        for beta, want in ((165.1, 1.0), (164.9, 0.75), (150.1, 0.75), (149.9, 0.5)):
            self.assertEqual(scored(state(wedge_left=at("left", 0.110, beta))), want, beta)
        for beta, want in ((111.1, 1.0), (110.9, 0.75), (140.9, 1.0), (141.1, 0.75), (96.1, 0.75), (95.9, 0.5),
                           (155.9, 0.75), (156.1, 0.5)):
            self.assertEqual(scored(state(wedge_right=at("right", 0.110, beta))), want, beta)

    def test_aim_edges(self):
        for off, want in ((19.9, 1.0), (-20.1, 0.75), (44.9, 0.75), (-45.1, 0.5)):
            self.assertEqual(scored(state(wedge_right=at("right", 0.110, 126.0, aim_off=off))), want, off)

    def test_height_edges(self):
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, bottom=ON + 0.0029))), 1.0)
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, bottom=ON + 0.0031))), 0.5)
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, bottom=ON - 0.0031))), 0.5)

    def test_tilt_edges(self):
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, pitch=9.9))), 1.0)
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, roll=10.1))), 0.5)

    def test_deck_edge(self):
        # the whole full band fits on the deck on the outer side of the supercardioid's null: at 149 mm and 111-141
        # degrees the outline stays more than 1 cm inside the deck's edge
        for beta in (111.5, 126.0, 140.5):
            edge = at("right", 0.149, beta)
            self.assertGreater(deck_margin(wedge_corners(edge)[:, :2]), 0.010, beta)
            self.assertEqual(scored(state(wedge_right=edge)), 1.0, beta)
        # further out (part band), the right wedge at 126 degrees 190 mm out hangs over the deck's edge: not placed
        far = at("right", 0.190, 126.0)
        self.assertLess(deck_margin(wedge_corners(far)[:, :2]), 0.0)
        self.assertEqual(scored(state(wedge_right=far)), 0.5)
        near = at("right", 0.170, 126.0)
        self.assertGreater(deck_margin(wedge_corners(near)[:, :2]), 0.0)
        self.assertEqual(scored(state(wedge_right=near)), 0.75)

    def test_two_wedges_at_one_singer_count_once(self):
        both_left = {"wedge_left": at("left", 0.110, 180.0), "wedge_right": at("left", 0.110, 165.5, side=-1.0)}
        self.assertEqual(scored(both_left), 0.5)
        d = detail(both_left)
        self.assertEqual(sorted(d["left"]["wedges"]), sorted(WEDGES))
        worse_first = {"wedge_left": at("left", 0.110, 160.0), "wedge_right": at("left", 0.110, 178.0, side=-1.0)}
        self.assertEqual(scored(worse_first), 0.5)  # the better one counts, whichever it is
        self.assertEqual(detail(worse_first)["left"]["best"], "wedge_right")

    def test_a_wedge_serves_the_nearer_stand(self):
        mid = wdg(0.300, 0.010, 90.0)  # between the singers, a little to the left: serves the left singer only
        self.assertEqual(wedge_pose(mid)["singer"], "left")
        self.assertEqual(scored({"wedge_left": mid, "wedge_right": START["wedge_right"]}), 0.0)


class ZeroConditions(unittest.TestCase):
    def test_carried_across_unturned_is_the_trap(self):
        """Set in the right places but facing the way they started (each to its wing): aimed 90 and about 130
        degrees off: 0."""
        unturned = {"wedge_left": wdg(*at("left", 0.110, 180.0).origin[:2], 90.0),
                    "wedge_right": wdg(*at("right", 0.110, 126.0).origin[:2], -90.0)}
        self.assertEqual(scored(unturned), 0.0)
        d = detail(unturned)
        self.assertAlmostEqual(d["wedge_left"]["aim_deg"], 90.0, places=1)
        self.assertGreater(d["wedge_right"]["aim_deg"], 125.0)

    def test_in_front_of_the_mic_or_beside_it(self):
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 0.0, aim_off=180.0))), 0.5)  # upstage, by the stool
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 90.0))), 0.5)  # to the side

    def test_still_in_the_wings_or_on_the_table(self):
        self.assertEqual(scored(state(wedge_left=START["wedge_left"])), 0.5)
        on_table = wdg(0.120, 0.150, 0.0, bottom=TOP)  # downstage of the deck, on the table
        self.assertEqual(scored(state(wedge_left=on_table)), 0.5)

    def test_held_above_the_deck(self):
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, bottom=ON + 0.010))), 0.5)
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, bottom=ON + 0.050),
                                      wedge_right=at("right", 0.110, 126.0, bottom=ON + 0.050))), 0.0)

    def test_sunk_into_the_deck_is_not_standing(self):
        self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, bottom=ON - 0.005))), 0.5)


class Exploits(unittest.TestCase):
    def test_on_its_back_side_front_or_upside_down(self):
        for kw in ({"pitch": -90.0}, {"pitch": 90.0}, {"roll": 90.0}, {"roll": 180.0}, {"pitch": 45.0}):
            self.assertEqual(scored(state(wedge_left=at("left", 0.110, 180.0, **kw))), 0.5, kw)

    def test_stacked_on_the_other_wedge(self):
        lower = at("left", 0.110, 180.0)
        upper = at("left", 0.110, 180.0, bottom=ON + 0.030)
        self.assertEqual(scored({"wedge_left": lower, "wedge_right": upper}), 0.5)
        self.assertFalse(detail({"wedge_left": lower, "wedge_right": upper})["wedge_right"]["placed"])

    def test_held_still_resting_on_the_deck_counts(self):
        """The scorer reads only the wedges (stated in fidelity): a wedge resting on the deck and not moving counts,
        whether or not a gripper still touches it. The trial runs 3 s past the first 1.0 so the demo ends with the
        grippers open."""
        self.assertEqual(scored(state()), 1.0)


class Stillness(unittest.TestCase):
    def test_moving_wedge_does_not_count(self):
        tr = Tracker(HZ)
        a = at("left", 0.110, 180.0)
        for i in range(WINDOW + 1):  # sliding 1 mm per step along y
            o = a.origin
            r = tr.update(state(wedge_left=wdg(o[0], o[1] + 0.001 * i, 0.0)))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["wedge_left"]["at_rest"])
        r = feed(tr, [state()] * (WINDOW + 1))
        self.assertEqual(r.score, 1.0)

    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [state()] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["moved_mm"]["wedge_left"])
        self.assertEqual(tr.update(state()).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        rng = random.Random(3)
        o = at("left", 0.110, 180.0).origin
        seq = [state(wedge_left=wdg(o[0] + rng.uniform(-0.0008, 0.0008), o[1] + rng.uniform(-0.0008, 0.0008), 0.0))
               for _ in range(WINDOW + 5)]
        self.assertEqual(feed(tr, seq).score, 1.0)

    def test_rest_edge_is_2_mm_in_half_a_second(self):
        o = at("left", 0.110, 180.0).origin
        for total, want in ((0.0019, 1.0), (0.0021, 0.5)):  # a slow creep along y over the last 0.5 s
            tr = Tracker(HZ)
            seq = [state(wedge_left=wdg(o[0], o[1] + total * i / WINDOW, 0.0)) for i in range(WINDOW + 1)]
            self.assertEqual(feed(tr, seq).score, want, total)

    def test_a_turn_in_place_counts_as_moving(self):
        tr = Tracker(HZ)
        o = at("left", 0.110, 180.0).origin
        seq = [state(wedge_left=wdg(o[0], o[1], -10.0 + 2.0 * i)) for i in range(WINDOW + 1)]  # 0.5 s, 20 degrees
        r = feed(tr, seq)
        self.assertEqual(r.score, 0.5)
        self.assertGreater(r.details["moved_mm"]["wedge_left"], 2.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        good = state()
        for bad in (np.array([np.nan, 0.0, 0.0]), np.array([np.inf, 0.0, 0.0])):
            w = SimpleNamespace(**{**vars(good["wedge_left"]), "origin": bad})
            r = tr.update(state(wedge_left=w))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        w = SimpleNamespace(**{**vars(good["wedge_right"]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(tr.update(state(wedge_right=w)).score, 0.0)
        w = SimpleNamespace(**{**vars(good["wedge_right"]), "pos": np.array([0.3, np.nan, 0.77])})
        self.assertEqual(tr.update(state(wedge_right=w)).score, 0.0)
        self.assertEqual(tr.update(good).score, 1.0)  # the history from before the blow-ups is intact

    def test_wedge_without_origin_or_missing(self):
        w = SimpleNamespace(**{k: v for k, v in vars(state()["wedge_left"]).items() if k != "origin"})
        self.assertFalse(valid(state(wedge_left=w)))
        self.assertEqual(scored(state(wedge_left=w)), 0.0)
        self.assertFalse(valid({"wedge_left": state()["wedge_left"]}))
        self.assertEqual(Tracker(HZ).update({"wedge_left": state()["wedge_left"]}).score, 0.0)
        self.assertFalse(valid(None))

    def test_unnormalised_quaternion(self):
        w = state()["wedge_right"]
        w2 = SimpleNamespace(**{**vars(w), "quat_xyzw": w.quat_xyzw * 3.0})
        self.assertEqual(scored(state(wedge_right=w2)), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(11)
        tr = Tracker(HZ)
        for _ in range(300):
            st = {n: wdg(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-360, 360), bottom=rng.uniform(0, 2),
                         pitch=rng.uniform(-180, 180), roll=rng.uniform(-180, 180)) for n in WEDGES}
            r = tr.update(st)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
        on_stool = wdg(STOOLS["left"][0], STOOLS["left"][1], 0.0)  # its middle right over the stool's centre
        self.assertEqual(placement(on_stool, "left")[2], 180.0)

    def test_evaluate_without_rest_is_never_above_zero(self):
        score, why, d = evaluate(state(), {})
        self.assertEqual(score, 0.0)
        self.assertEqual(why, "left_no_wedge,right_no_wedge")
        self.assertEqual(evaluate(state(), {n: True for n in WEDGES})[0], 1.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        r = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["moved_mm"]["wedge_left"])

    def test_reset_then_a_new_trial(self):
        tr, _ = settled(state())
        tr.update(START)
        self.assertEqual(feed(tr, [state()] * WINDOW).score, 0.0)  # its new history is not long enough yet
        self.assertEqual(tr.update(state()).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(state())
        jit = {"wedge_left": wdg(0.240, 0.365, 92.0, bottom=TOP), "wedge_right": wdg(0.200, -0.410, -88.0, bottom=TOP)}
        self.assertTrue(Tracker.at_start(jit))
        tr.update(jit)
        self.assertEqual(tr.steps, 1)

    def test_turned_start_is_not_the_start(self):
        turned = {"wedge_left": wdg(0.220, 0.390, 0.0, bottom=TOP), "wedge_right": START["wedge_right"]}
        self.assertFalse(Tracker.at_start(turned))

    def test_lifted_over_the_start_is_not_the_start(self):
        lifted = {"wedge_left": wdg(0.220, 0.390, 90.0, bottom=TOP + 0.02), "wedge_right": START["wedge_right"]}
        self.assertFalse(Tracker.at_start(lifted))

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(state())
        n = tr.steps
        a, b = state(), START
        for i in range(1, 41):  # back to the start over 2 s, never more than a few mm a step
            f = i / 40
            st = {k: wdg(*((1 - f) * a[k].origin[:2] + f * b[k].origin[:2]), 0.0 if k == "wedge_left" else 40.0,
                         bottom=TOP if f == 1 else ON) for k in WEDGES}
            tr.update(st)
        tr.update(START)
        self.assertEqual(tr.steps, n + 41)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(state())
        n = tr.steps
        tr.update({"wedge_left": wdg(0.30, 0.0, 0.0), "wedge_right": wdg(0.30, -0.10, 0.0)})
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(state())
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(tr.update(state()).score, 0.0)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


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

    def sim(self, st):
        """Synthetic states turned into what yam_arena and mechkit would report from the simulator."""
        out = dict(st)
        for n, spec in (("wedge_left", self.task.WEDGE_LEFT), ("wedge_right", self.task.WEDGE_RIGHT)):
            out[n] = self.mk.state_from_poses(spec, (*st[n].origin, *st[n].quat_xyzw), source="sim")
        return out

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), sorted(WEDGES))
        for o in loose:
            self.assertLessEqual(o.mass, 0.30)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        by = {o.name: o for o in t.OBJECTS}
        deck = by["stage_deck"]
        self.assertAlmostEqual(deck.pos[2] + deck.size[2] / 2, ON)
        np.testing.assert_allclose([deck.pos[0] - deck.size[0] / 2, deck.pos[0] + deck.size[0] / 2,
                                    deck.pos[1] - deck.size[1] / 2, deck.pos[1] + deck.size[1] / 2], DECK, atol=1e-12)
        for s, (sx, sy) in STANDS.items():
            for part in ("stand_base", "stand_pole", "mic_clip", "mic_head"):  # all on the stand's vertical axis
                self.assertAlmostEqual(by[f"{part}_{s}"].pos[0], sx)
                self.assertAlmostEqual(by[f"{part}_{s}"].pos[1], sy)
            handle = by[f"mic_handle_{s}"]
            self.assertLess(handle.pos[0], sx)  # the handle trails toward the audience: the mic points +x
            self.assertAlmostEqual(handle.pos[2], by[f"mic_head_{s}"].pos[2])
            self.assertAlmostEqual(by[f"stand_base_{s}"].pos[2] - by[f"stand_base_{s}"].height / 2, ON)
            stool = by[f"stool_{s}"]
            self.assertAlmostEqual(stool.pos[0], STOOLS[s][0])
            self.assertAlmostEqual(stool.pos[1], STOOLS[s][1])
            self.assertAlmostEqual(stool.pos[2] - stool.height / 2, ON)
            self.assertAlmostEqual(by[f"mic_head_{s}"].pos[2] - ON, 0.115)  # seated mouth height, 1.15 m
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None), 0.0)
            if size is None:  # cylinders and spheres
                half = getattr(o, "radius", 0.0)
            else:
                half = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            self.assertLess(abs(c[1]) + half, 0.45, o.name)

    def test_scorer_geometry_matches_mechkit(self):
        for spec, name in ((self.task.WEDGE_LEFT, "wedge_left"), (self.task.WEDGE_RIGHT, "wedge_right")):
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-0.0225, -0.030, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [0.0225, 0.030, 0.030], atol=1e-12)
            corners = np.array([c for sh in spec.shapes for c in _corners(sh.center, sh.size)])
            np.testing.assert_allclose(_sorted(corners), _sorted(LOCAL), atol=1e-9)
            grille = [sh for sh in spec.shapes if tuple(sh.color) == L.GRILLE_GREY]
            self.assertEqual(len(grille), 2)
            for sh in grille:  # the grille is on the front half (local +x, the way the scorer says it faces)
                self.assertGreaterEqual(sh.center[0] - sh.size[0] / 2, -0.0025 - 1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0] - self.task.GRIP_X), 0.0005)  # gripped over its centre of mass
            self.assertAlmostEqual(com[1], 0.0)
            rest = spec.rest_state()
            x, y, heading = L.WEDGES[name]
            np.testing.assert_allclose(rest.origin, [x, y, TOP], atol=1e-12)
            a = _rot(rest.quat_xyzw)[:, 0]
            self.assertAlmostEqual(math.atan2(a[1], a[0]), heading)
        rest = {"wedge_left": self.task.WEDGE_LEFT.rest_state(), "wedge_right": self.task.WEDGE_RIGHT.rest_state()}
        self.assertTrue(Tracker.at_start(rest))
        sim = self.sim(state())
        np.testing.assert_allclose(sim["wedge_left"].low[2], ON, atol=1e-9)
        self.assertEqual(scored(sim), 1.0)

    def test_offline_state_path(self):
        spec = self.task.WEDGE_RIGHT
        box = SimpleNamespace(pos=np.array(spec.offline_box()[0]), quat_xyzw=quat(-90.0), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [0.220, -0.390, TOP], atol=1e-12)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(state())
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

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(state()))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "zero_reason", "moved_mm", "wedge_left",
                          "wedge_right", "left", "right", "wedge_left_origin", "wedge_right_quat", "wedge_left_front",
                          "wedge_right_back", "wedge_left_source"):
                    self.assertIn(k, row)
                for k in ("singer", "r_mm", "beta_deg", "null_off_deg", "aim_deg", "tilt_deg", "gap_mm",
                          "deck_margin_mm", "on_pose", "fit", "at_rest", "placed", "credit"):
                    self.assertIn(k, row["wedge_left"])
                for k in ("wedges", "best", "credit"):
                    self.assertIn(k, row["right"])
                self.assertAlmostEqual(row["wedge_left"]["beta_deg"], 180.0, places=1)
                self.assertAlmostEqual(row["wedge_right"]["beta_deg"], 126.0, places=1)
                self.assertEqual(row["wedge_right"]["singer"], "right")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(state())))  # no crash
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
            # every close and open takes 1.5 s, and so does every last 20-30 mm down to a grip or a place
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                if prev.pos[2] > w.pos[2] and min(abs(w.pos[2] - t.PICK_Z), abs(w.pos[2] - t.PLACE_Z)) < 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertLessEqual(prev.pos[2] - w.pos[2], t.APPROACH + 1e-9)
                # no wrist move of more than a quarter turn in one waypoint (the way round is never ambiguous)
                self.assertLessEqual(abs(w.yaw - prev.yaw), math.pi / 2 + 1e-9, (arm, w.label))
            grip = [w for w in wps if w.label == "grip the wedge"][0]
            self.assertLess(abs(grip.finger), (0.060 - 0.007) / 2)  # commanded 8 mm narrower than the wedge
            self.assertGreater(abs(wps[0].finger), 0.060 / 2 + 0.010)  # opened 1 cm clear on each side
        # each wedge ends where the demo aims it, turned by exactly what the wrist turned, and that is full credit
        for arm, name, singer in (("left", "wedge_left", "left"), ("right", "wedge_right", "right")):
            wps = plan[arm]
            grip = [w for w in wps if w.label == "grip the wedge"][0]
            place = [w for w in wps if w.label == "slowly set the wedge on the deck"][0]
            sx, sy, sh = L.WEDGES[name]
            np.testing.assert_allclose(grip.pos[:2], t.grip_point(sx, sy, sh), atol=1e-12)
            self.assertAlmostEqual(grip.yaw, sh)  # gripped across its width: the wrist's yaw is the grille's heading
            heading = sh + (place.yaw - grip.yaw)
            tx, ty, th = t.target(singer)
            self.assertAlmostEqual(heading, th)
            origin = (place.pos[0] - t.GRIP_X * math.cos(th), place.pos[1] - t.GRIP_X * math.sin(th))
            np.testing.assert_allclose(origin, (tx, ty), atol=1e-12)
            w = wdg(origin[0], origin[1], math.degrees(heading))
            r, beta, aim = placement(w, singer)
            self.assertAlmostEqual(r, 0.110, places=9)
            self.assertAlmostEqual(abs(beta - NULLS[singer]), 0.0, places=6)
            self.assertAlmostEqual(aim, 0.0, places=6)
            self.assertGreater(deck_margin(wedge_corners(w)[:, :2]), 0.020)
        self.assertLess(t.target("right")[1], -0.150)  # the supercardioid's wedge on its outer side, as documented
        self.assertAlmostEqual(math.degrees(t.YAW_LEFT_PLACE - t.YAW_LEFT_PICK), -90.0)
        self.assertAlmostEqual(math.degrees(t.YAW_RIGHT_PLACE - t.YAW_RIGHT_PICK), 90.0 + 40.38, places=1)
        # pad tips clear of the deck when letting go
        self.assertGreater(t.PLACE_Z - 0.013 - ON, 0.005)
        old = dict(os.environ)
        try:
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(set(t.demo()), {"right"})
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(set(t.demo()), {"left"})
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(t.demo(), {})
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_either_arm_can_set_its_wedge_turned_or_not(self):
        """Which way a wedge faces is the policy's choice, not the kinematics': each arm can set its wedge at its
        target turned or unturned, with the whole interpolated path inside the joint limits. A wedge gripped across
        its width can be held from either side (wrist yaw h or h - 180 degrees): unturned, the left wedge is set down
        gripped from its other side (9.8 degrees from the nearest joint limit; with the demo's grip joint 6 would end
        0.4 degrees from its limit)."""
        import yamkin as K
        from yam_arena.demo import ArmPlan

        t = self.task
        bases = {"left": (0.0, 0.305, 0.75), "right": (0.0, -0.305, 0.75)}
        for arm, name, singer in (("left", "wedge_left", "left"), ("right", "wedge_right", "right")):
            sx, sy, sh = L.WEDGES[name]
            tx, ty, th = t.target(singer)
            for turn in (True, False):
                heading = th if turn else sh
                pick = t.grip_point(sx, sy, sh)
                place = t.grip_point(tx, ty, heading)
                results = []
                for flip in (0.0, -math.copysign(math.pi, sh)):  # the demo's grip, or the wedge held the other way
                    g0, g1 = sh + flip, heading + flip
                    p = ArmPlan(t._pick(pick, g0) + t._place(pick, place, g1, g0), bases[arm])
                    worst_margin = math.inf
                    for i in range(int(p.duration * HZ) + 1):
                        p.action7(i / HZ)
                        worst_margin = min(worst_margin, float(np.degrees(K.joint_margins(p.q)).min()))
                    results.append((p.worst_ik_mm < 1.0 and p.worst_ik_deg < 0.5 and worst_margin > 5.0, flip,
                                    round(worst_margin, 1)))
                self.assertTrue(any(ok for ok, _, _ in results), (arm, turn, results))
                if turn:
                    self.assertTrue(results[0][0] and results[0][2] > 10.0, (arm, results))  # the demo's own grip

    def test_arms_stay_clear_of_the_mics(self):
        """The YAM's wrist housing juts ahead of the gripper: over the whole demo every arm link (capsule proxy,
        rounder and larger than the simulator's hulls) stays at least 20 mm from each mic's handle, head, clip and
        pole."""
        import yamkin as K
        from yam_arena.demo import ArmPlan

        def mic(s):
            sx, sy = STANDS[s]
            z = TOP + L.HEAD_Z
            return [("handle", np.array([sx - 0.040, sy, z]), np.array([sx - 0.008, sy, z]), 0.0064),
                    ("head", np.array([sx, sy, z]), np.array([sx, sy, z]), 0.008),
                    ("pole", np.array([sx, sy, TOP]), np.array([sx, sy, TOP + L.POLE_TOP + 0.006]), 0.0099)]

        bases = {"left": np.array((0.0, 0.305, 0.75)), "right": np.array((0.0, -0.305, 0.75))}
        for arm, wps in self.task.demo().items():
            p = ArmPlan(wps, bases[arm])
            worst = math.inf
            for i in range(int(p.duration * HZ) + 1):
                p.action7(i / HZ)
                caps = K.arm().capsules(p.q, bases[arm])
                for s in STANDS:
                    worst = min(worst, K.capsule_clearance(caps, mic(s))[0])
            self.assertGreater(worst, 0.020, arm)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("1:10", "floor wedge for each seated singer", "null", "feed back", "black cardioid",
                  "directly behind", "180 degrees", "silver supercardioid", "126 degrees", "either side", "base down",
                  "80 to 150 mm", "mic stand", "grey grille", "stool"):
            self.assertIn(s, ins)
        self.assertLessEqual(len(ins), 490)
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], ins)
        import re
        cmds = re.findall(r'--instruction "([^"]*)"', (HERE / "README.md").read_text())
        self.assertEqual(len(cmds), 2)
        for c in cmds:
            self.assertEqual(c, ins)


if __name__ == "__main__":
    unittest.main()
