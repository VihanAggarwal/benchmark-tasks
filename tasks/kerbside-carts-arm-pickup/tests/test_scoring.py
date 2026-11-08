"""Offline tests for scoring.py and the hooks in task.py: synthetic cart states, no simulator.
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
    WINDOW, Tracker, clearances, facing_err_deg, footprint, heading_rad, lowest, on_driveway, setback, tilt_deg,
    valid,
)

TOP = L.TOP
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}
LID_TOP = L.CART_H + L.ARROW_T  # the arrow's top: the highest point of a cart


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


def cart(x, y, heading=0.0, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    """A cart state as mechkit reports it: frame origin (the middle of the body's bottom face) at (x, y), turned
    `heading` degrees and tilted by `pitch`/`roll`, resting with its lowest point `lift` above the verge (a tilted
    cart rests on an edge); low/high from its 79 x 58 x 102 mm box."""
    q = quat(heading, pitch, roll)
    local = np.array([(u, v, z) for u in (-L.FOOT_BACK, L.FOOT_FRONT) for v in (-L.FOOT_V, L.FOOT_V)
                      for z in (0.0, LID_TOP)])
    pts = local @ rot(q).T
    o = np.array([x, y, TOP + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


def kerbed(y, back=L.TARGET_SETBACK, heading=0.0, **kw):
    """A cart with its origin at y, turned `heading`, placed in x so that its footprint's most forward point is
    `back` behind the kerb face."""
    probe = cart(0.0, y, heading, **kw)
    s = cart(L.KERB_FACE_X - back - footprint(probe)[:, 0].max(), y, heading, **kw)
    assert abs(setback(s) - back) < 1e-9
    return s


START_L = cart(*L.START["cart_left"][:2], math.degrees(L.START["cart_left"][2]))
START_R = cart(*L.START["cart_right"][:2], math.degrees(L.START["cart_right"][2]))
LEFT = kerbed(L.TARGET["cart_left"][1])
RIGHT = kerbed(L.TARGET["cart_right"][1])
NEAR_MAILBOX_R = kerbed(-0.045)  # right of the mailbox, 8 mm from it: standing, facing, at the kerb, not clear


def settled(left, right, t=None, n=WINDOW + 1):
    """Feed the same pair n times (half a second still); return (result, tracker)."""
    t = Tracker() if t is None else t
    r = None
    for _ in range(n):
        r = t.update(left, right)
    return r, t


def scored(left, right=RIGHT) -> float:
    return settled(left, right)[0].score


def why(left, right=RIGHT, which="cart_left"):
    return settled(left, right)[0].details[which]["why"]


def zero_reason(left, right=RIGHT):
    return settled(left, right)[0].details["zero_reason"]


class Geometry(unittest.TestCase):
    def test_footprint_upright(self):
        fp = footprint(cart(0.3, 0.1, 0.0))
        self.assertAlmostEqual(fp[:, 0].max() - fp[:, 0].min(), 0.079, places=9)
        self.assertAlmostEqual(fp[:, 1].max() - fp[:, 1].min(), 0.058, places=9)
        self.assertAlmostEqual(fp[:, 0].max(), 0.3 + L.FOOT_FRONT, places=9)  # the front: the lid's opening side
        fp = footprint(cart(0.3, 0.1, 90.0))  # facing +y: 79 mm along y
        self.assertAlmostEqual(fp[:, 1].max(), 0.1 + L.FOOT_FRONT, places=9)
        self.assertAlmostEqual(fp[:, 1].min(), 0.1 - L.FOOT_BACK, places=9)

    def test_heading_tilt_height(self):
        for h in (0.0, 30.0, -90.0, 180.0):
            self.assertAlmostEqual(math.remainder(math.degrees(heading_rad(cart(0.3, 0.1, h))) - h, 360.0), 0.0)
        self.assertAlmostEqual(facing_err_deg(cart(0.3, 0.1, -20.0)), 20.0)
        self.assertAlmostEqual(facing_err_deg(cart(0.3, 0.1, 180.0)), 180.0)
        self.assertAlmostEqual(tilt_deg(cart(0.3, 0.1, 0.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(cart(0.3, 0.1, 0.0, roll=180.0)), 180.0, places=6)
        self.assertAlmostEqual(lowest(cart(0.3, 0.1, 0.0, lift=0.004)), 0.004, places=12)

    def test_the_stretches(self):
        lo, hi = L.stretch("left")
        self.assertAlmostEqual(lo, 0.128, places=9)
        self.assertAlmostEqual(hi, 0.168, places=9)
        lo, hi = L.stretch("right")
        self.assertAlmostEqual(hi, -0.128, places=9)
        self.assertAlmostEqual(lo, -0.1858, places=4)
        for y in np.linspace(*L.stretch("left"), 9)[1:-1]:
            self.assertEqual(scored(kerbed(float(y)), RIGHT), 1.0, y)
        for y in np.linspace(*L.stretch("right"), 9)[1:-1]:
            self.assertEqual(scored(LEFT, kerbed(float(y))), 1.0, y)
        # just outside each end: too close to the mailbox, the hydrant or the car
        self.assertEqual(why(kerbed(0.1275)), "too_close_to_mailbox")
        self.assertEqual(why(kerbed(0.1685)), "too_close_to_hydrant")
        self.assertEqual(why(LEFT, kerbed(-0.1275), "cart_right"), "too_close_to_mailbox")
        self.assertEqual(why(LEFT, kerbed(-0.1865), "cart_right"), "too_close_to_car")

    def test_start_layout_scores_zero(self):
        self.assertTrue(Tracker.at_start("cart_left", START_L))
        self.assertTrue(Tracker.at_start("cart_right", START_R))
        r, _ = settled(START_L, START_R)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["zero_reason"])
        self.assertEqual(r.details["cart_left"]["why"], "not_facing_the_street")
        self.assertEqual(r.details["cart_right"]["why"], "not_facing_the_street")
        for s in (START_L, START_R):
            fp = footprint(s)
            self.assertTrue(np.all(np.abs(fp[:, 1]) < 0.45))  # clear of the arms' start-pose shadow
            self.assertGreater(setback(s), 0.10)  # near the house, far from the kerb
            self.assertFalse(on_driveway(s))
        gap = footprint(START_L)[:, 1].min() - footprint(START_R)[:, 1].max()
        self.assertAlmostEqual(gap, 0.187, places=9)  # fronts toward each other, 187 mm apart

    def test_demo_target(self):
        r, _ = settled(LEFT, RIGHT)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        dl, dr = r.details["cart_left"], r.details["cart_right"]
        self.assertAlmostEqual(dl["setback_mm"], 12.0)
        self.assertAlmostEqual(dr["setback_mm"], 12.0)
        np.testing.assert_allclose(LEFT.origin[:2], (L.TARGET_X, 0.150), atol=1e-12)
        np.testing.assert_allclose(RIGHT.origin[:2], (L.TARGET_X, -0.150), atol=1e-12)
        cl, cr = clearances(LEFT, RIGHT), clearances(RIGHT, LEFT)
        self.assertAlmostEqual(cl["mailbox"], 0.113, places=9)
        self.assertAlmostEqual(cl["hydrant"], 0.109, places=9)
        self.assertAlmostEqual(cr["mailbox"], 0.113, places=9)
        self.assertAlmostEqual(cr["car"], math.hypot(0.032, 0.121), places=9)  # 125 mm, to the car's corner
        self.assertAlmostEqual(cl["other_cart"], 0.242, places=9)
        self.assertGreater(min(min(cl.values()), min(cr.values())) - L.CLEAR, 0.017)  # 18 mm or more to spare


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        self.assertEqual(scored(START_L, START_R), 0.0)  # the start
        self.assertEqual(scored(LEFT, START_R), 0.5)  # one set out, the other left at the house
        self.assertEqual(scored(START_L, RIGHT), 0.5)
        self.assertEqual(scored(LEFT, NEAR_MAILBOX_R), 0.75)  # both set out, one within 91 mm of the mailbox
        self.assertEqual(scored(LEFT, RIGHT), 1.0)
        both_left = (kerbed(0.150), kerbed(0.050))  # both in the stretch left of the mailbox: too close
        r, _ = settled(*both_left)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["cart_left"]["why"], "too_close_to_other_cart")
        self.assertEqual(r.details["cart_right"]["credit"], 0.25)
        self.assertEqual(scored(kerbed(0.150, heading=180.0), RIGHT), 0.5)  # handle to the street: 0 for that cart
        self.assertEqual(why(kerbed(0.150, heading=180.0)), "not_facing_the_street")
        self.assertEqual(scored(NEAR_MAILBOX_R, START_L), 0.25)
        self.assertEqual(scored(kerbed(0.045), NEAR_MAILBOX_R), 0.5)  # both beside the mailbox: 0.25 + 0.25

    def test_either_cart_may_go_on_either_side(self):
        self.assertEqual(scored(RIGHT, LEFT), 1.0)

    def test_setback_band_edges(self):
        self.assertEqual(scored(kerbed(0.150, back=0.0)), 1.0)  # front against the kerb face
        self.assertEqual(scored(kerbed(0.150, back=0.0299)), 1.0)
        self.assertEqual(scored(kerbed(0.150, back=0.0301)), 0.5)
        self.assertEqual(why(kerbed(0.150, back=0.0301)), "not_at_the_kerb")
        self.assertEqual(why(kerbed(0.150, back=0.080)), "not_at_the_kerb")  # 80 mm back: toward the house
        self.assertEqual(scored(kerbed(0.150, back=-0.0019)), 1.0)  # contact tolerance at the kerb face
        self.assertEqual(scored(kerbed(0.150, back=-0.0021)), 0.0)
        self.assertEqual(zero_reason(kerbed(0.150, back=-0.0021)), "a_cart_on_the_kerb_or_in_the_road")

    def test_facing_band_edges(self):
        for h, want in ((24.9, 1.0), (-24.9, 1.0), (25.1, 0.5), (-25.1, 0.5), (90.0, 0.5), (-90.0, 0.5),
                        (180.0, 0.5)):
            self.assertEqual(scored(kerbed(0.150, heading=h)), want, h)
        self.assertEqual(why(kerbed(0.150, heading=25.1)), "not_facing_the_street")
        # turned, a cart's front corner is what reaches the kerb: the set-back is measured to it
        s = kerbed(0.150, heading=20.0, back=0.0)
        self.assertAlmostEqual(setback(s), 0.0, places=9)
        self.assertEqual(scored(s), 1.0)

    def test_clearance_edges(self):
        d = L.CLEAR
        for eps, want in ((0.0001, 1.0), (-0.0001, 0.75)):
            to_mailbox = kerbed(L.MAILBOX_RECT[3] + d + eps + L.FOOT_V)
            to_hydrant = kerbed(L.HYDRANT_XY[1] - L.HYDRANT_R - d - eps - L.FOOT_V)
            dx = L.CAR_RECT[0] - (L.KERB_FACE_X - L.TARGET_SETBACK)
            to_car = kerbed(L.CAR_RECT[3] + math.sqrt((d + eps) ** 2 - dx ** 2) + L.FOOT_V)
            self.assertEqual(scored(to_mailbox, RIGHT), want, eps)
            self.assertEqual(scored(to_hydrant, RIGHT), want, eps)
            self.assertEqual(scored(LEFT, to_car), want, eps)
        self.assertEqual(why(kerbed(L.MAILBOX_RECT[3] + d - 0.0001 + L.FOOT_V)), "too_close_to_mailbox")
        # between the carts (no layout of two carts at the kerb is clear of both and of each other: the function)
        a = cart(0.25, 0.10, 0.0)
        for eps, ok in ((0.0001, True), (-0.0001, False)):
            b = cart(0.25, 0.10 - 2 * L.FOOT_V - d - eps, 0.0)
            self.assertEqual(clearances(a, b)["other_cart"] >= d, ok)
        self.assertAlmostEqual(clearances(a, cart(0.25, 0.20, 0.0))["other_cart"], 0.042, places=9)
        self.assertEqual(clearances(a, cart(0.25, 0.12, 0.0))["other_cart"], 0.0)  # overlapping

    def test_only_a_full_score_is_success(self):
        for left, right, want in ((LEFT, NEAR_MAILBOX_R, 0.75), (LEFT, START_R, 0.5), (NEAR_MAILBOX_R, START_L, 0.25)):
            r, _ = settled(left, right)
            self.assertEqual(r.score, want)
            self.assertFalse(r.success, want)
        self.assertTrue(settled(LEFT, RIGHT)[0].success)

    def test_every_tier_is_reachable(self):
        options = (START_L, LEFT, NEAR_MAILBOX_R, RIGHT, kerbed(0.150, heading=90.0))
        seen = set()
        for i, a in enumerate(options):
            for b in options[i + 1:]:
                seen.add(scored(a, b))
        self.assertEqual(seen, TIERS)

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(LEFT, RIGHT)
        self.assertEqual(t.result(LEFT, RIGHT).score, 1.0)
        self.assertEqual(t.result(LEFT, RIGHT).score, 1.0)  # result() does not advance the tracker
        self.assertEqual(t.steps, WINDOW + 1)

    def test_random_states_map_to_a_tier(self):
        rng = random.Random(7)
        for _ in range(300):
            states = [cart(rng.uniform(0.15, 0.42), rng.uniform(-0.3, 0.45), rng.uniform(-180, 180),
                           lift=rng.choice((0.0, 0.0, 0.002, 0.01)), pitch=rng.choice((0.0, 0.0, 5.0, 30.0, 90.0)))
                      for _ in range(2)]
            self.assertIn(scored(*states), TIERS)


class ZeroConditions(unittest.TestCase):
    def test_a_cart_knocked_over_zeroes_the_trial(self):
        for p in (45.1, 60.0, 90.0, -90.0):
            self.assertEqual(scored(kerbed(0.150, pitch=p), RIGHT), 0.0, p)
            self.assertEqual(zero_reason(kerbed(0.150, back=0.050, pitch=p)), "a_cart_knocked_over")
        self.assertEqual(scored(cart(0.25, 0.15, 0.0, roll=180.0), RIGHT), 0.0)  # upside down
        self.assertEqual(scored(START_L, cart(0.22, -0.13, 90.0, roll=90.0)), 0.0)  # knocked over at the house
        # tilted 10.1 to 45 degrees (leaning on something): that cart earns nothing, the trial is not zeroed
        self.assertEqual(scored(kerbed(0.150, pitch=9.9), RIGHT), 1.0)
        self.assertEqual(why(kerbed(0.150, pitch=10.1)), "not_upright")
        self.assertEqual(scored(kerbed(0.150, back=0.020, pitch=10.1), RIGHT), 0.5)
        self.assertEqual(scored(kerbed(0.150, back=0.020, roll=44.9), RIGHT), 0.5)

    def test_on_the_kerb_or_in_the_road_zeroes_the_trial(self):
        in_road = cart(0.470, 0.150, 0.0, lift=L.ROAD_H)
        self.assertEqual(scored(in_road, RIGHT), 0.0)
        self.assertEqual(zero_reason(in_road), "a_cart_on_the_kerb_or_in_the_road")
        on_kerb = cart(0.400, 0.150, 0.0, lift=L.KERB_H)
        self.assertEqual(zero_reason(on_kerb), "a_cart_on_the_kerb_or_in_the_road")
        # standing in the road side-on, or past the kerb line across the driveway: the same
        self.assertEqual(scored(cart(0.450, -0.100, 90.0, lift=L.ROAD_H), LEFT), 0.0)
        self.assertEqual(zero_reason(cart(0.380, 0.450, 0.0, lift=L.DRIVE_H)), "a_cart_on_the_kerb_or_in_the_road")
        # leaning against the kerb (its base's front edge at the kerb face), its top over it: the shadow is past
        lean = cart(L.KERB_FACE_X - L.FOOT_FRONT * math.cos(math.radians(8.0)), 0.150, 0.0, pitch=8.0)
        self.assertLess(setback(lean), -L.KERB_TOL)
        self.assertEqual(zero_reason(lean), "a_cart_on_the_kerb_or_in_the_road")

    def test_on_the_driveway(self):
        s = cart(0.300, 0.450, 0.0, lift=L.DRIVE_H)
        self.assertTrue(on_driveway(s))
        self.assertEqual(why(s), "on_the_driveway")
        self.assertEqual(scored(s, RIGHT), 0.5)
        edge = kerbed(L.DRIVE_Y[0] - L.FOOT_V - 0.0005)  # 0.5 mm short of the driveway: off it
        self.assertFalse(on_driveway(edge))
        self.assertTrue(on_driveway(kerbed(L.DRIVE_Y[0] - L.FOOT_V + 0.0005)))

    def test_off_the_verge(self):
        self.assertEqual(scored(kerbed(0.150, lift=0.0029)), 1.0)
        self.assertEqual(why(kerbed(0.150, lift=0.0031)), "not_on_the_verge")
        self.assertEqual(why(kerbed(0.150, lift=0.025)), "not_on_the_verge")  # carried 25 mm up
        self.assertEqual(scored(kerbed(0.150, lift=-0.0029)), 1.0)  # contact settling
        self.assertEqual(why(kerbed(0.150, lift=-0.0031)), "not_on_the_verge")  # sunk: a broken state

    def test_moving(self):
        t = Tracker()
        for i in range(30):  # sliding along the kerb at 20 mm/s: 1 mm a step, 10 mm over 0.5 s
            r = t.update(kerbed(0.140 + 0.001 * i), RIGHT)
        self.assertEqual(r.details["cart_left"]["why"], "moving")
        self.assertEqual(r.score, 0.5)

    def test_left_at_the_house_or_side_on(self):
        self.assertEqual(why(cart(0.22, 0.15, 0.0)), "not_at_the_kerb")  # facing the street but 131 mm back
        self.assertEqual(why(kerbed(0.150, heading=90.0)), "not_facing_the_street")  # side-on at the kerb


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for _ in range(WINDOW):
            self.assertEqual(t.update(LEFT, RIGHT).score, 0.0)
        self.assertEqual(t.update(LEFT, RIGHT).score, 1.0)

    def test_slow_creep_under_5_mm_is_still(self):
        t = Tracker()
        for i in range(40):  # 0.4 mm a step: 4 mm over 0.5 s
            r = t.update(kerbed(0.140 + 0.0004 * i), RIGHT)
        self.assertTrue(r.details["cart_left"]["standing"])
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(30):  # 0.6 mm back and forth every step: 6 mm of path over 0.5 s, though it goes nowhere
            r = t.update(kerbed(0.150 + (0.0006 if i % 2 else 0.0)), RIGHT)
        self.assertFalse(r.details["cart_left"]["standing"])
        self.assertEqual(r.score, 0.5)

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        for i in range(30):  # 1.5 degree a step about its middle: 1.3 mm a step at its corners
            r = t.update(kerbed(0.150, back=0.020, heading=1.5 * (i if i < 15 else 30 - i)), RIGHT)
        self.assertEqual(r.details["cart_left"]["why"], "moving")


class Exploits(unittest.TestCase):
    def test_held_still_just_above_the_verge_counts_as_standing(self):
        # The scorer sees only the carts (task.json fidelity): held still 2 mm up at the place, a cart counts.
        self.assertEqual(scored(kerbed(0.150, lift=0.002), kerbed(-0.150, lift=0.002)), 1.0)

    def test_both_carts_in_one_stretch(self):
        self.assertEqual(scored(kerbed(0.160), kerbed(0.040)), 0.5)
        self.assertEqual(scored(kerbed(-0.140), kerbed(-0.040)), 0.5)

    def test_on_its_back_with_the_lid_toward_the_street(self):
        s = cart(0.30, 0.15, 0.0, pitch=90.0)  # lying on its front, the lid facing +x (toward the street)
        self.assertGreater(np.array([0.0, 0.0, 1.0]) @ rot(s.quat_xyzw)[0], 0.99)
        self.assertEqual(scored(s, RIGHT), 0.0)
        self.assertEqual(zero_reason(s), "a_cart_knocked_over")

    def test_stacked_on_the_other_cart(self):
        top = cart(L.TARGET_X, 0.150, 0.0, lift=LID_TOP)
        r, _ = settled(top, LEFT)
        self.assertEqual(r.details["cart_left"]["why"], "not_on_the_verge")
        self.assertEqual(r.details["cart_right"]["why"], "too_close_to_other_cart")
        self.assertEqual(r.score, 0.25)

    def test_beside_the_car_or_on_the_far_side_of_the_hydrant(self):
        self.assertEqual(why(LEFT, kerbed(-0.350), "cart_right"), "too_close_to_car")
        self.assertEqual(scored(LEFT, kerbed(-0.350)), 0.75)
        self.assertEqual(why(kerbed(0.330)), "too_close_to_hydrant")
        self.assertEqual(why(cart(0.30, 0.450, 0.0, lift=L.DRIVE_H)), "on_the_driveway")  # past the hydrant

    def test_pushed_against_the_post_or_against_the_hydrant(self):
        self.assertEqual(why(kerbed(0.037)), "too_close_to_mailbox")  # 0 mm from the post's line
        self.assertEqual(why(kerbed(0.259)), "too_close_to_hydrant")

    def test_parked_far_back_out_of_everything(self):
        self.assertEqual(scored(cart(0.250, 0.150, 0.0), cart(0.250, -0.150, 0.0)), 0.0)  # 104 mm back


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = settled(LEFT, RIGHT)
        self.assertEqual(r.score, 1.0)
        n = {k: len(v) for k, v in t.history.items()}
        bad = []
        for k in ("origin", "quat_xyzw", "pos", "lin_vel", "low"):
            s = kerbed(0.150)
            setattr(s, k, np.full_like(np.asarray(getattr(s, k), dtype=float), np.nan))
            bad.append(s)
        z = kerbed(0.150)
        z.quat_xyzw = np.zeros(4)
        bad.append(z)
        inf = kerbed(0.150)
        inf.origin = np.array([np.inf, 0.15, TOP])
        bad.append(inf)
        for s in bad:
            self.assertFalse(valid(s))
            for pair in ((s, RIGHT), (LEFT, s)):
                res = t.update(*pair)
                self.assertEqual(res.score, 0.0)
                self.assertEqual(res.details["zero_reason"], "invalid_state")
                self.assertEqual({k: len(v) for k, v in t.history.items()}, n)
        self.assertEqual(t.update(LEFT, RIGHT).score, 1.0)  # the record was left as it was

    def test_missing_fields_or_objects(self):
        s = SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3))  # no origin
        self.assertFalse(valid(s))
        self.assertEqual(Tracker().update(s, RIGHT).score, 0.0)
        self.assertEqual(Tracker().update(None, RIGHT).score, 0.0)
        self.assertEqual(Tracker().update(LEFT, None).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = kerbed(0.150)
        s.quat_xyzw = s.quat_xyzw * 3.0
        self.assertEqual(scored(s), 1.0)

    def test_front_pointing_straight_up(self):
        s = cart(0.30, 0.15, 0.0, pitch=-90.0)
        self.assertAlmostEqual(tilt_deg(s), 90.0, places=6)
        self.assertEqual(facing_err_deg(s), 180.0)
        self.assertEqual(scored(s), 0.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, t = settled(LEFT, RIGHT)
        r = t.update(START_L, START_R)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history["cart_left"]), 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        _, t = settled(LEFT, RIGHT)
        t.update(cart(0.25, 0.10, -90.0), cart(0.19, -0.16, 90.0))
        self.assertEqual(t.steps, 1)

    def test_reset_when_one_cart_had_not_moved(self):
        _, t = settled(LEFT, START_R)
        t.update(START_L, START_R)  # the left cart jumps home in one step, the right one never left: a reset
        self.assertEqual(t.steps, 1)

    def test_one_cart_carried_home_is_not_a_reset(self):
        _, t = settled(LEFT, START_R)
        t.update(cart(0.28, 0.14, -90.0), START_R)  # 62 mm from its mark: not on it yet
        n = t.steps
        t.update(START_L, START_R)  # back on its mark, 61 mm in the step: under the 80 mm jump limit
        self.assertEqual(t.steps, n + 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = np.array(L.START["cart_left"][:2]), np.array([0.26, 0.16])
        for i in range(20):
            p = b + (a - b) * i / 19
            t.update(cart(p[0], p[1], -90.0), START_R)
        self.assertEqual(t.steps, 20)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = settled(START_L, START_R)
        t.update(LEFT, RIGHT)
        self.assertEqual(t.steps, WINDOW + 2)

    def test_explicit_reset(self):
        _, t = settled(LEFT, RIGHT)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(LEFT, RIGHT).score, 0.0)  # half a second of history again


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
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
        cls.specs = {"cart_left": cls.task.CART_LEFT, "cart_right": cls.task.CART_RIGHT}

    def sim(self, left, right):
        """Synthetic states turned into what mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in (("cart_left", left), ("cart_right", right))}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], ["cart_left", "cart_right"])
        self.assertTrue(all(o.mass <= 0.30 for o in loose))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        kerb = t.KERB
        self.assertAlmostEqual(kerb.pos[0] - kerb.size[0] / 2, L.KERB_FACE_X)
        self.assertAlmostEqual(kerb.pos[1] + kerb.size[1] / 2, L.DRIVE_Y[0])  # dropped at the driveway
        # the scorer's outlines are the objects': mailbox (covering its post), hydrant, car
        mb, post, hyd, car = t.MAILBOX, t.POST, t.HYDRANT, t.CAR
        self.assertAlmostEqual(mb.pos[0] - mb.size[0] / 2, L.MAILBOX_RECT[0])
        self.assertAlmostEqual(mb.pos[1] + mb.size[1] / 2, L.MAILBOX_RECT[3])
        self.assertLessEqual(post.pos[0] + post.radius, L.MAILBOX_RECT[1] + 1e-12)
        self.assertGreaterEqual(post.pos[0] - post.radius, L.MAILBOX_RECT[0] - 1e-12)
        self.assertLessEqual(post.radius, L.MAILBOX_RECT[3] + 1e-12)
        self.assertEqual((hyd.pos[0], hyd.pos[1], hyd.radius), (*L.HYDRANT_XY, L.HYDRANT_R))
        self.assertLessEqual(t.HYDRANT_BONNET.radius, hyd.radius)
        self.assertAlmostEqual(car.pos[1] + car.size[1] / 2, L.CAR_RECT[3])
        self.assertAlmostEqual(car.pos[0] - car.size[0] / 2, L.CAR_RECT[0])
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None), 0.0)
            if size is None:
                size = (2 * o.radius,) * 3
            hx = abs(math.cos(yaw)) * size[0] / 2 + abs(math.sin(yaw)) * size[1] / 2
            hy = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            if c[0] - hx < 0.2 - 1e-9:
                self.assertLess(abs(c[1]) + hy, 0.45, o.name)

    def test_cart_geometry_matches_the_scorer(self):
        for name, spec in self.specs.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-L.FOOT_BACK, -L.FOOT_V, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [L.FOOT_FRONT, L.FOOT_V, LID_TOP], atol=1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0] - self.task.GRIP_U), 0.002)  # the grip is within 2 mm of the centre of mass
            self.assertAlmostEqual(com[1], 0.0, places=12)
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [*L.START[name][:2], TOP], atol=1e-12)
            self.assertTrue(Tracker.at_start(name, rest))
        shapes = [(c, s) for _, c, s, _ in L.CART_SHAPES]
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                self.assertLess(_boxes_overlap(shapes[i], shapes[j]), 1e-12, (L.CART_SHAPES[i][0],
                                                                              L.CART_SHAPES[j][0]))
        # every part is within the 58 mm width: the grip across the body spans the whole cart
        for _, c, s, _ in L.CART_SHAPES:
            self.assertLessEqual(abs(c[1]) + s[1] / 2, L.FOOT_V + 1e-12)
        # the handle and the wheels are at the back, the arrow points to the front
        names = {n: (c, s) for n, c, s, _ in L.CART_SHAPES}
        for n in ("handle", "wheel_left", "wheel_right"):
            self.assertAlmostEqual(names[n][0][0] - names[n][1][0] / 2, -L.FOOT_BACK)
        head = max(c[0] + s[0] / 2 for n, (c, s) in names.items() if n.startswith("arrow"))
        tail = min(c[0] - s[0] / 2 for n, (c, s) in names.items() if n.startswith("arrow"))
        self.assertGreater(head, 0.0)
        self.assertLess(tail, 0.0)
        self.assertGreater(head + tail, 0.0)  # the arrow head is the forward end
        sim = self.sim(LEFT, RIGHT)
        self.assertAlmostEqual(float(sim["cart_left"].low[2]), TOP, places=9)
        self.assertEqual(scored(sim["cart_left"], sim["cart_right"]), 1.0)

    def test_offline_state_path(self):
        st = {}
        for name, spec in self.specs.items():
            c, _, yaw = spec.offline_box()
            box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                                  ang_vel=np.zeros(3))
            st[name] = spec.offline_state(box)
            self.assertEqual(st[name].source, "offline")
            np.testing.assert_allclose(st[name].origin, [*L.START[name][:2], TOP], atol=1e-12)
        self.assertEqual(scored(st["cart_left"], st["cart_right"]), 0.0)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START_L, START_R)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(LEFT, RIGHT)
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
        self.assertEqual(self.task.score({"cart_left": done["cart_left"]}), 0.0)  # a missing cart scores 0

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(LEFT, RIGHT))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "valid_state", "cart_left", "cart_right",
                          "origins", "quats", "low_z", "vels", "source"):
                    self.assertIn(k, row)
                for k in ("credit", "why", "standing", "facing_err_deg", "setback_mm", "on_driveway", "clear_mm",
                          "nearest", "origin_mm", "tilt_deg", "lowest_mm", "path_mm"):
                    self.assertIn(k, row["cart_left"])
                self.assertEqual(row["cart_left"]["setback_mm"], 12.0)
                self.assertEqual(row["cart_left"]["why"], "moving")  # one step seen: not yet still
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(LEFT, RIGHT)))  # no crash
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
            # the grip is where the cart is, at its start and at its place, and the yaw turns it to face the street
            cart_name = t.SIDE[arm]
            grip = [w for w in wps if w.label == "grip across the body"][0]
            np.testing.assert_allclose(grip.pos[:2], L.START[cart_name][:2], atol=1e-12)
            put = [w for w in wps if w.label == "let go"][0]
            np.testing.assert_allclose(put.pos[:2], L.TARGET[cart_name][:2], atol=1e-12)
            self.assertAlmostEqual(put.yaw - grip.yaw, L.TARGET[cart_name][2] - L.START[cart_name][2])
            self.assertAlmostEqual(abs(math.degrees(grip.yaw)), 90.0)
            self.assertLessEqual(abs(math.degrees(grip.yaw)), 120.0)  # joint 6 limit
        self.assertLess(plan["left"][4].yaw, 0.0)  # the left arm turns from -90, the right from +90
        self.assertGreater(plan["right"][4].yaw, 0.0)
        self.assertAlmostEqual(t.GRIP, -0.025)  # 50 mm commanded on the 58 mm body
        self.assertAlmostEqual(t.PLACE_Z - t.GRIP_Z, 0.002)
        # the gripper's body (35 mm above the tool point) clears the lid arrow at the grip
        self.assertGreater(t.GRIP_Z - TOP + 0.035 - LID_TOP, 0.008)

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

    def test_carry_path_clears_the_street_and_the_other_cart(self):
        """Replays both arms' carries (the cart rigid in the grip, turning with the wrist) and checks the lifted
        carts' footprints against the kerb face, the mailbox, the hydrant, the car and each other."""
        t = self.task
        paths = {}
        for arm in ("left", "right"):
            wps = t.demo()[arm]
            a = [i for i, w in enumerate(wps) if w.label == "lift 25 mm"][0]
            b = [i for i, w in enumerate(wps) if w.label == "lower it slowly"][0]
            name = t.SIDE[arm]
            h0, y0 = L.START[name][2], wps[a].yaw
            fps = []
            for tt, p, yaw, _ in self._sample(wps[a:b + 1]):
                h = h0 + (yaw - y0)
                c = cart(p[0] - t.GRIP_U * math.cos(h), p[1] - t.GRIP_U * math.sin(h), math.degrees(h))
                fps.append((round(tt, 3), c))
            paths[arm] = fps
            self.assertAlmostEqual(math.remainder(h - L.TARGET[name][2], 2 * math.pi), 0.0, places=9)
        worst = {"kerb": 1.0, "street": 1.0, "carts": 1.0}
        for (tl, cl), (tr, cr) in zip(paths["left"], paths["right"]):
            self.assertAlmostEqual(tl, tr)  # both arms on one clock: the same moment
            for c, other in ((cl, cr), (cr, cl)):
                worst["kerb"] = min(worst["kerb"], setback(c))
                d = clearances(c, other)
                worst["street"] = min(worst["street"], d["mailbox"], d["hydrant"], d["car"])
                worst["carts"] = min(worst["carts"], d["other_cart"])
        self.assertGreater(worst["kerb"], 0.0119)  # never closer to the kerb than the place (12 mm)
        self.assertGreater(worst["street"], 0.095)  # README: at least 97 mm from the mailbox, hydrant and car
        self.assertGreater(worst["carts"], 0.175)  # and the carts at least 180 mm apart

    def test_tcps_stay_apart(self):
        t = self.task
        plan = t.demo()
        sl, sr = self._sample(plan["left"]), self._sample(plan["right"])
        self.assertEqual(len(sl), len(sr))
        self.assertGreaterEqual(min(abs(a[1][1] - b[1][1]) for a, b in zip(sl, sr)), 0.20)

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
        for s in ("lid opening toward the street", "white arrow", "25 degrees", "0 to 30 mm", "1:10",
                  "not on it", "91 mm (3 feet)", "the other cart", "the mailbox and its post", "the hydrant",
                  "the parked car", "off the driveway", "handle and wheels toward the house"):
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
        for s in ("10 degrees", "within 3 mm", "5 mm over the last 0.5 s", "25 degrees", "0 to 30 mm",
                  "2 mm past", "91 mm", "45 degrees", "0.25", "0.5", "0.75", "1.0", "3 s"):
            self.assertIn(s, sc)
        self.assertEqual(L.TILT_STAND_DEG, 10.0)
        self.assertEqual(L.FLOOR_TOL, 0.003)
        self.assertEqual(L.STILL_PATH, 0.005)
        self.assertEqual(L.FACING_DEG, 25.0)
        self.assertEqual(L.SETBACK_MAX, 0.030)
        self.assertEqual(L.KERB_TOL, 0.002)
        self.assertEqual(L.CLEAR, 0.091)
        self.assertEqual(L.TIPPED_DEG, 45.0)
        for s in ("x 39.0", "12 mm tall", "y +36.0", "16 mm across", "105 mm", "x 37.0", "24 mm across",
                  "55 mm tall", "x 36.5", "y +30.0", "41.0", "-30.0", "120 g", "73 mm", "58 mm", "95 mm",
                  "101 mm", "79 mm", "x 22.0", "y +13.0", "y -13.0", "187 mm"):
            self.assertIn(s, su)
        nums = {float(v) for v in re.findall(r"(\d+(?:\.\d+)?) mm", spec["scoring"])}
        self.assertTrue({3.0, 5.0, 30.0, 2.0, 91.0} <= nums)

    def test_neither_arm_reaches_the_far_place(self):
        """README 'One arm alone': straight down, at the grip height, each arm's reach ends short of the other
        arm's place, at every yaw (joint 6 allows +-120 degrees)."""
        import yamkin as K
        for arm, other in (("left", "cart_right"), ("right", "cart_left")):
            base = K.BASES["station"][arm]
            x, y, _ = L.TARGET[other]
            for z in (0.068, 0.078):
                for yaw in range(-120, 121, 15):
                    cell, _ = K.reach_cell((x, y, TOP + z), math.radians(yaw), base, TOP, None)
                    self.assertEqual(cell, 0, (arm, z, yaw))
            # ... but it reaches the kerb just past the mailbox on the far side (the README's one-arm 0.75)
            cell, _ = K.reach_cell((L.TARGET_X, -0.045 if arm == "left" else 0.045, TOP + 0.078), 0.0, base, TOP, None)
            self.assertEqual(cell, 2, arm)
        self.assertAlmostEqual(clearances(LEFT, NEAR_MAILBOX_R)["other_cart"], 0.137, places=9)
        self.assertEqual(scored(LEFT, NEAR_MAILBOX_R), 0.75)


if __name__ == "__main__":
    unittest.main()
