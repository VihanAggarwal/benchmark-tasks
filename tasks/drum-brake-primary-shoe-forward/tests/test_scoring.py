"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import itertools
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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, bottom_gap, evaluate, fits, flat_on_plate, footprint_distance, held, highest, lowest, mid_arc,
    rotation, seat_error, tilt_deg, valid, wrap_deg, yaw_deg,
)

MM = 0.001
SEAT_Z = L.TOP + L.PLATE_H + L.PAINT_H  # a fitted shoe's underside (on the seat paint)
RF, RR = ("right_plate", "front"), ("right_plate", "rear")
LF, LR = ("left_plate", "front"), ("left_plate", "rear")
SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}


def quat(yaw=0.0, tilt=0.0, axis="x"):
    """Quaternion (xyzw): a turn of ``yaw`` degrees about z, after a tilt of ``tilt`` degrees about a body axis."""
    def q_axis(v, deg):
        h = math.radians(deg) / 2
        return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])

    def mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    v = {"x": (1, 0, 0), "y": (0, 1, 0)}[axis]
    return tuple(mul(q_axis((0, 0, 1), yaw), q_axis(v, tilt)))


def shoe(x, y, yaw=0.0, bottom=SEAT_Z, tilt=0.0, axis="x", vel=(0, 0, 0), spin=(0, 0, 0)):
    """A shoe's state: its mid-arc point at (x, y) with its lowest point at ``bottom``, turned ``yaw`` degrees,
    tilted ``tilt`` degrees about its body ``axis``. pos/low/high are its bounding box (unused by the scorer except
    for validity)."""
    q = quat(yaw, tilt, axis)
    probe = SimpleNamespace(origin=np.zeros(3), quat_xyzw=np.array(q))
    o = np.array([x, y, bottom - lowest(probe)])
    r = rotation(q)
    c = o + r @ np.array([-0.012, 0.0, L.SHOE_TOP / 2])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(q, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - 0.05, high=c + 0.05, origin=o)


def seated(key, dx=0.0, dy=0.0, dyaw=0.0, **kw):
    x, y, yaw = L.SEATS[key]
    return shoe(x + dx, y + dy, yaw + dyaw, **kw)


def start():
    return [shoe(x, y, yaw, bottom=L.TOP) for _, x, y, yaw in (L.SHOES[n] for n in L.NAMES)]


S0 = start()
# Order of layout.NAMES: secondary_1, primary_1, primary_2, secondary_2.
CORRECT = {"secondary_1": RR, "primary_1": RF, "primary_2": LF, "secondary_2": LR}


def fitted(assign, **offsets):
    """Shoes in the order of layout.NAMES, each in the seat ``assign[name]`` (or on its start mark if None);
    ``offsets[name]`` = dict(dx, dy, dyaw, ...) moves one."""
    out = []
    for n, s0 in zip(L.NAMES, S0, strict=True):
        key = assign.get(n)
        out.append(s0 if key is None else seated(key, **offsets.get(n, {})))
    return out


DONE = fitted(CORRECT)
PARKED = {"left": rigstate.Arm(tcp=np.array([0.20, 0.25, L.TOP + 0.15]),
                               tips=np.array([0.20, 0.25, L.TOP + 0.137]), opening=0.47),
          "right": rigstate.Arm(tcp=np.array([0.20, -0.25, L.TOP + 0.15]),
                                tips=np.array([0.20, -0.25, L.TOP + 0.137]), opening=0.47)}
ON = 12 / 93.9  # rigstate opening with the pads closed on a shoe (12 mm commanded)


def run(shoes, steps=25, arms=PARKED, tracker=None):
    """Feed the same state ``steps`` times; the last Result."""
    t = tracker or Tracker()
    res = None
    for _ in range(steps):
        res = t.update(shoes, arms)
    return res


def final(shoes, arms=PARKED):
    return run(shoes, arms=arms).score


class Geometry(unittest.TestCase):
    def test_shoe_dimensions(self):
        self.assertAlmostEqual(L.R_MID - L.SHOE_HALF_W, 0.036)
        self.assertAlmostEqual(L.R_MID + L.SHOE_HALF_W, 0.056)
        # The end disc's edge is 70 degrees round along the centre line: 140 degrees end to end.
        self.assertAlmostEqual(L.END_DISC_DEG + math.degrees(math.asin(L.SHOE_HALF_W / L.R_MID)), 70.0)
        # Every disc is centred on the centre line; neighbours overlap (no gaps along the bar).
        cs = sorted(L.shoe_disc_centres(), key=lambda p: math.atan2(p[1], p[0] + L.R_MID))
        for x, y in cs:
            self.assertAlmostEqual(math.hypot(x - L.CURVE_C[0], y - L.CURVE_C[1]), L.R_MID)
        pts = [(0.0, 0.0)] + cs
        pts.sort(key=lambda p: math.atan2(p[1], p[0] + L.R_MID))
        for a_, b_ in zip(pts, pts[1:]):
            self.assertLess(math.dist(a_, b_), 2 * L.SHOE_HALF_W - 0.008)
        # The grip block's faces are 20 mm apart radially; its outer face is the outermost point on its line.
        self.assertAlmostEqual(L.GRIP_BLOCK[0], 2 * L.SHOE_HALF_W)

    def test_lining_lengths_differ_clearly(self):
        for kind, deg in (("primary", 90.0), ("secondary", 130.0)):
            ang = L.lining_angles(kind)
            edge = math.degrees(math.asin(L.LINING_DOT_R / L.LINING_R))
            self.assertAlmostEqual(ang[-1] + edge, deg / 2)
            self.assertAlmostEqual(ang[0], -ang[-1])
            self.assertLessEqual(max(b - a for a, b in zip(ang, ang[1:])), L.LINING_STEP_DEG + 1e-9)
            s = shoe(0.0, 0.0, 0.0, bottom=0.0)
            for x, y in L.lining_centres(kind):  # on the shoe's top, inside its outer edge (to 0.5 mm at the ends)
                r = math.hypot(x - L.CURVE_C[0], y - L.CURVE_C[1])
                self.assertLessEqual(r + L.LINING_DOT_R, L.R_MID + L.SHOE_HALF_W - 0.0005 + 1e-9)
                self.assertGreaterEqual(r - L.LINING_DOT_R, L.R_MID)
                for a in np.linspace(0, 2 * math.pi, 13):
                    p = (x + L.LINING_DOT_R * math.cos(a), y + L.LINING_DOT_R * math.sin(a), 0.0)
                    self.assertLess(footprint_distance(s, p), 0.0005)
        # Along the outer edge (56 mm) the bands are 88 and 127 mm long: 39 mm apart.
        self.assertGreater(math.radians(40) * L.LINING_R, 0.035)

    def test_seats_follow_the_mirror(self):
        # Left rear brake: front of the car toward +y; right rear brake: toward -y.
        self.assertGreater(L.SEATS[LF][1], L.PLATES["left_plate"][1])
        self.assertLess(L.SEATS[RF][1], L.PLATES["right_plate"][1])
        for key, (x, y, yaw) in L.SEATS.items():
            px, py, _ = L.PLATES[key[0]]
            # the seat's centre of curvature is the plate's centre
            cx = x + L.CURVE_C[0] * math.cos(math.radians(yaw))
            cy = y + L.CURVE_C[0] * math.sin(math.radians(yaw))
            self.assertAlmostEqual(cx, px)
            self.assertAlmostEqual(cy, py)

    def test_fitted_shoes_clear_post_block_and_each_other(self):
        """With the whole shoe (not just its grip block): every shoe in its seat stays 10 mm or more from its
        plate's anchor post and 7 mm or more from its adjuster block, and the two shoes on a plate are 25 mm apart,
        on the plate (inside its 60 mm radius) everywhere."""
        for plate, (px, py, _) in L.PLATES.items():
            a, b = seated((plate, "front")), seated((plate, "rear"))
            post = np.array([px + L.ANCHOR_DX, py, SEAT_Z + 0.004])
            self.assertGreater(footprint_distance(a, post) - L.ANCHOR_R, 0.010)
            self.assertGreater(footprint_distance(b, post) - L.ANCHOR_R, 0.010)
            bx0, by0 = px + L.ADJ_DX - L.ADJ_SIZE[0] / 2, py - L.ADJ_SIZE[1] / 2
            block = [(bx0 + u * L.ADJ_SIZE[0], by0 + v * L.ADJ_SIZE[1]) for u in np.linspace(0, 1, 7)
                     for v in np.linspace(0, 1, 7)]
            for s in (a, b):
                self.assertGreater(min(footprint_distance(s, (x, y, SEAT_Z)) for x, y in block), 0.007)
            # The two shoes: sample one's footprint, measure to the other.
            r = rotation(a.quat_xyzw)
            for d in np.linspace(-70, 70, 29):
                for rad in (0.036, 0.046, 0.056):
                    p = a.origin + r @ np.array([L.CURVE_C[0] + rad * math.cos(math.radians(d)),
                                                 rad * math.sin(math.radians(d)), 0.0])
                    self.assertGreater(footprint_distance(b, p), 0.025)
                    self.assertLess(math.hypot(p[0] - px, p[1] - py), L.PLATE_R)

    def test_start_row_is_clear_and_mirror_symmetric(self):
        for (n1, s1), (n2, s2) in itertools.combinations(zip(L.NAMES, S0), 2):
            self.assertGreater(footprint_distance(s1, s2.origin), 0.10, (n1, n2))
        for n in L.NAMES:
            kind, x, y, yaw = L.SHOES[n]
            mirror = [m for m in L.NAMES if math.isclose(L.SHOES[m][2], -y)]
            self.assertEqual(L.SHOES[mirror[0]][0], kind)
        # Every shoe at the start: its whole outline in front of the plates (x 0.29 on), clear of the screenshot's
        # covered corners (|y| > 0.45 m with x < 0.2 m).
        for s in S0:
            r = rotation(s.quat_xyzw)
            for d in np.linspace(-70, 70, 15):
                for rad in (0.036, 0.056):
                    p = s.origin + r @ np.array([L.CURVE_C[0] + rad * math.cos(math.radians(d)),
                                                 rad * math.sin(math.radians(d)), 0.0])
                    self.assertLess(p[0], L.PLATE_X - L.PLATE_R - 0.02)
                    self.assertGreater(p[0], 0.21)
                    self.assertLess(abs(p[1]), 0.39)

    def test_pose_helpers(self):
        s = seated(LF)
        self.assertAlmostEqual(yaw_deg(s), 90.0)
        self.assertAlmostEqual(tilt_deg(s), 0.0)
        self.assertAlmostEqual(bottom_gap(s), L.PAINT_H)
        self.assertAlmostEqual(highest(s) - lowest(s), L.SHOE_TOP)
        self.assertTrue(flat_on_plate(s))
        up = seated(LF, tilt=180.0)  # lining down
        self.assertAlmostEqual(tilt_deg(up), 180.0)
        self.assertAlmostEqual(bottom_gap(up), L.PAINT_H)
        self.assertFalse(flat_on_plate(up))
        self.assertEqual(wrap_deg(190.0), -170.0)
        self.assertEqual(footprint_distance(s, s.origin), 0.0)
        self.assertAlmostEqual(footprint_distance(s, s.origin + np.array([0.0, 0.030, 0.0])), 0.020)  # 30 mm out
        px, py, _ = L.PLATES["left_plate"]
        self.assertAlmostEqual(footprint_distance(s, (px, py, SEAT_Z)), L.R_MID - L.SHOE_HALF_W)

    def test_valid(self):
        self.assertTrue(valid(S0[0]))
        for bad in (dict(pos=np.array([np.nan, 0, 0])), dict(quat_xyzw=np.zeros(4)), dict(origin=None),
                    dict(lin_vel=np.array([np.inf, 0, 0])), dict(origin=np.zeros(2))):
            s = SimpleNamespace(**{**vars(S0[0]), **bad})
            self.assertFalse(valid(s), bad)
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        res = run(S0)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_correct_fit_scores_one_after_a_second(self):
        t = Tracker()
        scores = [t.update(DONE, PARKED).score for _ in range(25)]
        w = t.window
        self.assertEqual(scores[:w], [0.0] * w)  # nothing settled before 0.5 s of steps
        self.assertEqual(scores[w:2 * w], [0.75] * w)  # counted, but not yet still for 1 s
        self.assertEqual(set(scores[2 * w:]), {1.0})
        self.assertTrue(t.update(DONE, PARKED).success)

    def test_one_shoe_eleven_mm_off_or_ten_degrees_turned(self):
        for off in (dict(dx=0.011), dict(dy=-0.011), dict(dyaw=10.0), dict(dyaw=-14.0), dict(dx=0.009, dyaw=9.0)):
            self.assertEqual(final(fitted(CORRECT, primary_1=off)), 0.75, off)

    def test_fine_and_coarse_edges(self):
        for name, key in CORRECT.items():
            for off, want in ((dict(dx=0.0079), 1.0), (dict(dy=-0.0079), 1.0), (dict(dyaw=7.9), 1.0),
                              (dict(dyaw=-7.9), 1.0), (dict(dx=0.0081), 0.75), (dict(dyaw=8.1), 0.75),
                              (dict(dx=0.0139), 0.75), (dict(dyaw=-14.9), 0.75),
                              (dict(dx=0.0141), 0.5), (dict(dyaw=15.1), 0.5)):
                self.assertEqual(final(fitted(CORRECT, **{name: off})), want, (name, off))

    def test_mirror_copy_scores_half(self):
        """The right plate fitted like the left (primary on its +y side, which is its rear): 0.5."""
        copy = dict(CORRECT, primary_1=RR, secondary_1=RF)
        self.assertEqual(final(fitted(copy)), 0.5)
        copy = dict(CORRECT, primary_2=LR, secondary_2=LF)  # or the left fitted like the right
        self.assertEqual(final(fitted(copy)), 0.5)

    def test_one_plate_right_the_other_bare_or_half_done(self):
        self.assertEqual(final(fitted({"primary_1": RF, "secondary_1": RR})), 0.5)
        self.assertEqual(final(fitted({"primary_2": LF, "secondary_2": LR})), 0.5)
        self.assertEqual(final(fitted(dict(CORRECT, primary_2=None))), 0.5)
        self.assertEqual(final(fitted(dict(CORRECT, secondary_2=None, primary_2=LR))), 0.5)

    def test_both_plates_primary_rearward(self):
        swap = {"primary_1": RR, "secondary_1": RF, "primary_2": LR, "secondary_2": LF}
        self.assertEqual(final(fitted(swap)), 0.25)

    def test_one_shoe_or_none(self):
        self.assertEqual(final(fitted({"primary_1": RF})), 0.0)
        self.assertEqual(final(fitted({"secondary_2": LR})), 0.0)
        self.assertEqual(final(S0), 0.0)

    def test_two_shoes_seated_but_no_plate_right(self):
        self.assertEqual(final(fitted({"primary_1": RF, "primary_2": LF})), 0.25)
        self.assertEqual(final(fitted({"secondary_1": RF, "primary_1": RR})), 0.25)
        self.assertEqual(final(fitted({"primary_1": LF, "secondary_1": LR})), 0.5)  # any primary, any plate

    def test_every_assignment_of_shoes_to_seats(self):
        """All 24 ways of putting the four shoes in the four seats: 1.0 only for the right one, 0.5 when exactly one
        plate is right, 0.25 otherwise; never anything else."""
        seats = [RF, RR, LF, LR]
        seen = {}
        for perm in itertools.permutations(seats):
            assign = dict(zip(L.NAMES, perm, strict=True))
            got = final(fitted(assign))
            right = all((L.SHOES[n][0] == "primary") == (k[1] == "front") for n, k in assign.items())
            plates = [all((L.SHOES[n][0] == "primary") == (k[1] == "front") for n, k in assign.items() if k[0] == p)
                      for p in ("left_plate", "right_plate")]
            want = 1.0 if right else 0.5 if sum(plates) == 1 else 0.25
            self.assertEqual(got, want, assign)
            seen[got] = seen.get(got, 0) + 1
        # the primaries are alike and the secondaries are alike: 4 assignments are right
        self.assertEqual(seen, {1.0: 4, 0.5: 8, 0.25: 12})

    def test_every_score_turns_up(self):
        got = {final(DONE), final(fitted(CORRECT, primary_1=dict(dx=0.011))),
               final(fitted({"primary_1": RF, "secondary_1": RR})),
               final(fitted({"primary_1": RR, "secondary_1": RF})), final(S0)}
        self.assertEqual(got, SCORES)

    def test_score_is_the_state_at_hand(self):
        t = Tracker()
        run(DONE, tracker=t)
        moved = fitted(CORRECT, secondary_2=dict(dx=0.011))
        self.assertEqual(t.result(moved, PARKED).score, 0.75)  # same history, the state at hand decides seating
        self.assertEqual(t.result(S0, PARKED).score, 0.0)

    def test_scores_tracked_and_first_steps(self):
        t = Tracker()
        run(DONE, tracker=t)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(t.first_at[0.75], t.window + 1)
        self.assertEqual(t.first_at[1.0], 2 * t.window + 1)


class NotSeated(unittest.TestCase):
    def check_breaks_one(self, **pose):
        s = fitted(CORRECT)
        s[1] = seated(RF, **pose)  # primary_1
        return final(s)

    def test_concave_side_out(self):
        self.assertEqual(self.check_breaks_one(dyaw=180.0), 0.5)
        self.assertFalse(fits(seated(RF, dyaw=180.0), RF, "coarse"))

    def test_upside_down_tilted_or_lifted(self):
        self.assertEqual(self.check_breaks_one(tilt=180.0), 0.5)  # lining down
        self.assertEqual(self.check_breaks_one(tilt=6.0), 0.5)
        self.assertEqual(self.check_breaks_one(tilt=6.0, axis="y"), 0.5)
        self.assertEqual(self.check_breaks_one(tilt=4.5), 1.0)
        self.assertEqual(self.check_breaks_one(bottom=SEAT_Z + 0.0035), 0.5)  # on the anchor post, on another shoe
        self.assertEqual(self.check_breaks_one(bottom=L.TOP + L.PLATE_H - 0.0035), 0.5)  # sunk through
        self.assertEqual(self.check_breaks_one(bottom=SEAT_Z + 0.0020), 1.0)

    def test_stacked_on_the_other_shoe(self):
        s = fitted(dict(CORRECT, secondary_1=None))
        s[0] = seated(RF, bottom=SEAT_Z + L.SHOE_TOP)  # the secondary lying on the primary
        self.assertEqual(final(s), 0.5)

    def test_a_seat_holding_two_shoes_holds_none(self):
        shoes = fitted(CORRECT)
        counted = [True] * 4
        two = [seated(RF), seated(RF, dx=0.003), shoes[2], shoes[3]]
        score, d = evaluate(two, counted, [True] * 4)
        self.assertEqual(score, 0.5)
        self.assertEqual(d["seats_fine"]["right_plate:front"], ["secondary_1", "primary_1"])
        self.assertFalse(d["plate_coarse"]["right_plate"])
        # both primaries in the right front seat (the primary listed first): that seat holds neither
        both = [seated(RR), seated(RF), seated(RF, dx=0.003), seated(LR)]
        score, d = evaluate(both, counted, [True] * 4)
        self.assertEqual(score, 0.25)
        self.assertFalse(d["plate_coarse"]["right_plate"])

    def test_moving_shoes_are_not_counted(self):
        t = Tracker()
        for k in range(25):  # one shoe creeping 0.25 mm a step (5 mm in 1 s) stays uncounted
            s = fitted(CORRECT, primary_2=dict(dx=0.00025 * k - 0.003))
            res = t.update(s, PARKED)
        self.assertEqual(res.score, 0.5)
        t = Tracker()
        for k in range(25):  # one turning 0.15 degrees a step (1.5 degrees in 0.5 s)
            res = t.update(fitted(CORRECT, primary_2=dict(dyaw=0.15 * k - 2.0)), PARKED)
        self.assertEqual(res.score, 0.5)
        t = Tracker()
        for k in range(25):  # settled under the limits: 1 mm and 0.5 degrees of drift in 1 s count
            res = t.update(fitted(CORRECT, primary_2=dict(dx=0.00005 * k, dyaw=0.02 * k)), PARKED)
        self.assertEqual(res.score, 1.0)

    def test_still_for_one_second_is_needed_for_one(self):
        t = Tracker()
        run(DONE, steps=15, tracker=t)  # settled 0.5 s but not 1 s
        self.assertEqual(t.update(DONE, PARKED).score, 0.75)
        t = Tracker()
        for k in range(10):  # a knock 0.75 s ago: settled now but not in the window before
            t.update(fitted(CORRECT, secondary_2=dict(dx=0.0003 * k)), PARKED)
        for _ in range(11):
            res = t.update(fitted(CORRECT, secondary_2=dict(dx=0.0027)), PARKED)
        self.assertEqual(res.score, 0.75)
        for _ in range(10):
            res = t.update(fitted(CORRECT, secondary_2=dict(dx=0.0027)), PARKED)
        self.assertEqual(res.score, 1.0)

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        t = Tracker()
        run(DONE, tracker=t)
        before = (len(t.history), t.best)
        for bad in (dict(pos=np.array([np.nan, 0, 0])), dict(quat_xyzw=np.zeros(4)), dict(origin=None),
                    dict(ang_vel=np.array([0, np.inf, 0]))):
            s = list(DONE)
            s[2] = SimpleNamespace(**{**vars(s[2]), **bad})
            res = t.update(s, PARKED)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
            self.assertFalse(res.details["valid_state"])
        self.assertEqual((len(t.history), t.best), before)
        self.assertEqual(t.bad_steps, 4)
        self.assertEqual(t.update(DONE, PARKED).score, 1.0)
        self.assertEqual(t.update(DONE[:3], PARKED).score, 0.0)  # a shoe missing


class LetGo(unittest.TestCase):
    def arms_at(self, side, tips, opening, other=PARKED):
        arms = dict(other)
        arms[side] = rigstate.Arm(tcp=np.add(tips, (0, 0, 0.013)), tips=np.asarray(tips, float), opening=opening)
        return arms

    def test_reach_numbers(self):
        self.assertAlmostEqual(L.finger_reach(1.0), math.hypot(0.0939 / 2 + 0.010, 0.020) + 0.005)
        self.assertAlmostEqual(L.finger_reach(0.0), math.hypot(0.010, 0.020) + 0.005)
        self.assertGreater(L.finger_reach(44 / 93.9), 0.042)

    def test_parked_arms_let_every_shoe_go(self):
        for s in DONE + S0:
            self.assertFalse(held(s, PARKED))

    def test_last_shoe_still_gripped_in_its_seat(self):
        """Shortcut 5: the full fit, but the right arm still grips its last shoe in its seat: the right plate is not
        correct (0.5); once the gripper has opened and risen clear it is (1.0 after a second)."""
        x, y, _ = L.SEATS[RR]
        grip = self.arms_at("right", (x, y, SEAT_Z + 0.004), ON)
        self.assertTrue(held(DONE[0], grip))
        self.assertEqual(final(DONE, arms=grip), 0.5)
        opened = self.arms_at("right", (x, y, SEAT_Z + 0.004), 44 / 93.9)  # open but not lifted: still on it
        self.assertEqual(final(DONE, arms=opened), 0.5)
        t = Tracker()
        run(DONE, steps=30, arms=grip, tracker=t)
        lifted = self.arms_at("right", (x, y, SEAT_Z + L.SHOE_TOP + 0.011), 44 / 93.9)
        scores = [t.update(DONE, lifted).score for _ in range(12)]
        self.assertEqual(scores[:10], [0.5] * 10)  # held within the last 0.5 s: not yet counted
        self.assertEqual(scores[10:], [1.0, 1.0])

    def test_a_shoe_held_now_is_never_counted_whatever_the_history(self):
        x, y, _ = L.SEATS[RR]
        t = Tracker()
        run(DONE, tracker=t)  # a second of history with the arms parked
        grip = self.arms_at("right", (x, y, SEAT_Z + 0.004), ON)
        self.assertEqual(t.result(DONE, grip).score, 0.5)
        self.assertEqual(t.result(DONE, PARKED).score, 1.0)

    def test_the_other_shoe_on_the_plate_is_not_held_by_a_gripper_at_its_neighbour(self):
        x, y, _ = L.SEATS[RR]
        arms = self.arms_at("right", (x, y, SEAT_Z + 0.004), 44 / 93.9)
        self.assertFalse(held(DONE[1], arms))  # primary_1 in the right front seat
        self.assertTrue(held(DONE[0], arms))

    def test_a_finger_propping_a_shoe_end_holds_it(self):
        s = seated(LF)
        r = rotation(s.quat_xyzw)
        end = s.origin + r @ np.array([L.CURVE_C[0] + L.R_MID * math.cos(math.radians(70)),
                                       L.R_MID * math.sin(math.radians(70)) + 0.020, 0.010])
        self.assertTrue(held(s, self.arms_at("left", end, 0.0)))
        far = end + r @ np.array([0.0, 0.030, 0.0])
        self.assertFalse(held(s, self.arms_at("left", far, 0.0)))

    def test_pad_tips_10_mm_over_the_top_have_let_go(self):
        s = seated(LF)
        top = highest(s)
        self.assertTrue(held(s, self.arms_at("left", (*s.origin[:2], top + 0.0095), 1.0)))
        self.assertFalse(held(s, self.arms_at("left", (*s.origin[:2], top + 0.0105), 1.0)))

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        t = Tracker()
        res = run(DONE, arms=None, tracker=t)
        self.assertEqual(res.score, 1.0)
        self.assertEqual(t.arms_unavailable, 25)
        self.assertEqual(res.details["arms_readable"], [])

    def test_one_unreadable_arm_never_hides_the_other(self):
        x, y, _ = L.SEATS[RR]
        arms = {"left": None, "right": rigstate.Arm(tcp=np.array([x, y, SEAT_Z + 0.017]),
                                                    tips=np.array([x, y, SEAT_Z + 0.004]), opening=ON)}
        t = Tracker()
        res = run(DONE, arms=arms, tracker=t)
        self.assertEqual(res.score, 0.5)
        self.assertEqual(t.arms_unavailable_by_side, {"left": 25, "right": 0})
        broken = {"left": SimpleNamespace(tips=[np.nan, 0, 0], opening=1.0), "right": arms["right"]}
        self.assertEqual(final(DONE, arms=broken), 0.5)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = Tracker()
        run(DONE, tracker=t)
        fired = []
        rigstate.on_reset(lambda: fired.append(1))
        rigstate.offline_reset()
        self.assertTrue(fired)


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        t = Tracker()
        run(DONE, tracker=t)
        res = t.update(S0, PARKED)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history), 1)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(res.score, 0.0)

    def test_gradual_return_is_not_a_reset(self):
        t = Tracker()
        run(S0, steps=3, tracker=t)
        for k in range(1, 6):
            s = [shoe(*L.SHOES[n][1:3], bottom=L.TOP) for n in L.NAMES]
            s[0] = shoe(0.250 + 0.001 * k, -0.330, bottom=L.TOP, vel=(0.02, 0, 0))
            t.update(s, PARKED)
        self.assertEqual(t.steps, 8)

    def test_jump_elsewhere_is_not_a_reset(self):
        t = Tracker()
        run(S0, steps=3, tracker=t)
        t.update(DONE, PARKED)
        self.assertEqual(t.steps, 4)

    def test_jittered_reset_counts(self):
        t = Tracker()
        run(DONE, tracker=t)
        jit = [shoe(L.SHOES[n][1] + 0.02, L.SHOES[n][2] - 0.02, 0.0, bottom=L.TOP) for n in L.NAMES]
        t.update(jit, PARKED)
        self.assertEqual(t.steps, 1)


class Exploits(unittest.TestCase):
    def test_nearest_seat_for_each_shoe(self):
        """Taking each shoe straight forward to the nearer seat of the nearer plate: every primary lands on a rear
        seat. 0.25."""
        assign = {}
        for n in L.NAMES:
            _, x, y, _ = L.SHOES[n]
            assign[n] = min(L.SEAT_KEYS, key=lambda k: math.hypot(L.SEATS[k][0] - x, L.SEATS[k][1] - y))
        self.assertEqual(sorted(assign.values()), sorted(L.SEAT_KEYS))
        self.assertTrue(all(assign[n][1] == "rear" for n in L.PRIMARIES))
        self.assertEqual(final(fitted(assign)), 0.25)

    def test_copying_one_plate_onto_the_other(self):
        """The same layout on both plates (in the table frame): one plate is always backwards. 0.5."""
        for prim_y in (+1, -1):
            assign = {}
            for p in ("left_plate", "right_plate"):
                py = L.PLATES[p][1]
                k_prim = min(((p, "front"), (p, "rear")), key=lambda k: -prim_y * (L.SEATS[k][1] - py))
                k_sec = (p, "rear") if k_prim[1] == "front" else (p, "front")
                pr = "primary_2" if p == "left_plate" else "primary_1"
                se = "secondary_2" if p == "left_plate" else "secondary_1"
                assign[pr], assign[se] = k_prim, k_sec
            self.assertEqual(final(fitted(assign)), 0.5, prim_y)

    def test_random_swaps(self):
        rng = random.Random(3)
        for _ in range(40):
            names = list(L.NAMES)
            rng.shuffle(names)
            assign = dict(zip(names, [RF, RR, LF, LR], strict=True))
            got = final(fitted(assign))
            plates = [all((L.SHOES[n][0] == "primary") == (k[1] == "front") for n, k in assign.items()
                          if k[0] == p) for p in ("left_plate", "right_plate")]
            want = 1.0 if all(plates) else 0.5 if any(plates) else 0.25
            self.assertEqual(got, want)

    def test_holding_shoes_in_their_seats_earns_nothing_for_them(self):
        lx, ly, _ = L.SEATS[LF]
        rx, ry, _ = L.SEATS[RF]
        arms = {"left": rigstate.Arm(tcp=np.array([lx, ly, SEAT_Z + 0.017]), tips=np.array([lx, ly, SEAT_Z + 0.004]),
                                     opening=ON),
                "right": rigstate.Arm(tcp=np.array([rx, ry, SEAT_Z + 0.017]),
                                      tips=np.array([rx, ry, SEAT_Z + 0.004]), opening=ON)}
        # Gripping both primaries in their seats: only the two secondaries (really fitted) count, no plate is right.
        self.assertEqual(final(DONE, arms=arms), 0.25)
        self.assertEqual(final(fitted({"primary_1": RF, "primary_2": LF}), arms=arms), 0.0)

    def test_shoes_dropped_or_standing_anywhere_else_score_nothing(self):
        on_bench = [shoe(0.30, y, 90.0, bottom=L.TOP) for y in (-0.33, -0.11, 0.11, 0.33)]
        self.assertEqual(final(on_bench), 0.0)
        on_edge = [seated(k, tilt=90.0) for k in (RR, RF, LF, LR)]
        self.assertEqual(final(on_edge), 0.0)

    def test_random_states_score_only_listed_values(self):
        rng = np.random.default_rng(7)
        seats = list(L.SEAT_KEYS)
        for _ in range(300):
            shoes = []
            for n in L.NAMES:
                if rng.random() < 0.7:
                    k = seats[rng.integers(4)]
                    shoes.append(seated(k, dx=float(rng.normal(0, 0.008)), dy=float(rng.normal(0, 0.008)),
                                        dyaw=float(rng.normal(0, 10.0)) + (180.0 if rng.random() < 0.05 else 0.0),
                                        tilt=float(abs(rng.normal(0, 3.0)))))
                else:
                    _, x, y, yaw = L.SHOES[n]
                    shoes.append(shoe(x, y, yaw, bottom=L.TOP))
            res = run(shoes)
            self.assertIn(res.score, SCORES)
            self.assertEqual(res.success, res.score == 1.0)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yamkin.py").is_file():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub and kinematics (tools/offline)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        sys.path.insert(0, str(_offline_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, shoes):
        return dict(zip(self.task.SHOE_NAMES, shoes, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SHOE_NAMES, L.NAMES)
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 3 * 2 + 4)
        for spec in t.SHOES:
            kind, x, y, yaw = L.SHOES[spec.name]
            self.assertEqual(spec.mass, L.SHOE_MASS)
            self.assertTrue(np.allclose(spec.pos, (x, y, L.TOP)))
            self.assertAlmostEqual(spec.yaw, math.radians(yaw))
            browns = [s for s in spec.shapes if s.color == L.LINING_BROWN]
            self.assertEqual(len(browns), len(L.lining_angles(kind)))
            self.assertEqual(len(spec.shapes) - len(browns), 13)  # the block and twelve discs
            lo, hi = spec.local_bounds()
            self.assertTrue(np.allclose(lo, (-0.046 + 0.046 * math.cos(math.radians(L.END_DISC_DEG)) - 0.010,
                                             -(0.046 * math.sin(math.radians(L.END_DISC_DEG)) + 0.010), 0.0),
                                        atol=0.0005))
            self.assertAlmostEqual(hi[0], 0.010)
            self.assertAlmostEqual(hi[2], L.SHOE_TOP)
            # The real shoe's state, read the way the simulator reports it, gives the layout's mid-arc point.
            st = spec.rest_state()
            self.assertTrue(np.allclose(st.origin, (x, y, L.TOP)))
            self.assertAlmostEqual(yaw_deg(st), yaw)
        for plate in ("left_plate", "right_plate"):
            spec = next(o for o in t.OBJECTS if o.name == plate)
            self.assertIsNone(spec.mass)
            lo, hi = spec.world_bounds()
            self.assertAlmostEqual(hi[2], L.TOP + L.PLATE_H + L.PAINT_H)
        # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
        for o in t.OBJECTS:
            lo, hi = (o.world_bounds() if hasattr(o, "world_bounds") else
                      (np.subtract(o.pos, (o.size if hasattr(o, "size") else (2 * o.radius,) * 3)),
                       np.add(o.pos, (o.size if hasattr(o, "size") else (2 * o.radius,) * 3))))
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)

    def test_offline_box_is_the_grip_block_and_rebuilds_the_frame(self):
        spec = self.task.SHOES[1]
        centre, size, yaw = spec.offline_box()
        self.assertTrue(np.allclose(centre, (0.250, -0.110, L.TOP + L.SHOE_TOP / 2)))
        self.assertEqual(size[:2], (0.020, 0.016))
        moved = SimpleNamespace(pos=np.array([0.35, -0.2, L.TOP + 0.0145]), quat_xyzw=np.array(quat(-90.0)),
                                lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))
        st = spec.offline_state(moved)
        self.assertTrue(np.allclose(st.origin, (0.35, -0.2, L.TOP + 0.0145 - L.SHOE_TOP / 2)))
        self.assertAlmostEqual(yaw_deg(st), -90.0)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()  # no arms read (as offline): the let-go rule is skipped
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                for _ in range(21):
                    ok = self.task.success(self.objects(DONE))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 22)
        for key in ("step", "score", "success", "line", "plate_fine", "plate_coarse", "seats_fine", "seats_coarse",
                    "n_seated", "counted", "all_still_1s", "shoes", "bad_steps", "first_step", "rigstate_error"):
            self.assertIn(key, rows[-1])
        sh = rows[-1]["shoes"]["primary_1"]
        for key in ("mid_mm", "yaw_deg", "tilt_deg", "bottom_mm", "held", "counted", "still_1s", "nearest_seat",
                    "off_seat_mm", "yaw_err_deg", "path_mm", "yaw_range_deg"):
            self.assertIn(key, sh)
        self.assertEqual(sh["nearest_seat"], "right_plate:front")
        self.assertEqual(sh["off_seat_mm"], 0.0)
        self.assertIsNone(rows[0]["shoes"]["primary_1"]["path_mm"])

    def test_hooks_read_the_arms(self):
        x, y, _ = L.SEATS[LF]

        def feed(left_tips, left_open):
            rigstate.offline_feed({"left": {"tcp": np.add(left_tips, (0, 0, 0.013)), "tips": left_tips,
                                            "opening": left_open},
                                   "right": {"tcp": PARKED["right"].tcp, "tips": PARKED["right"].tips,
                                             "opening": 0.47}})
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                for _ in range(25):
                    feed((x, y, SEAT_Z + 0.004), ON)
                    self.assertFalse(self.task.success(self.objects(DONE)))
                self.assertEqual(self.task.score(self.objects(DONE)), 0.5)
                for _ in range(21):
                    feed(tuple(PARKED["left"].tips), 0.47)
                    ok = self.task.success(self.objects(DONE))
                self.assertTrue(ok)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(rows[0]["shoes"]["primary_2"]["held"])
        self.assertFalse(rows[-1]["shoes"]["primary_2"]["held"])
        self.assertIn("left_tips", rows[-1])

    def test_mechkit_reset_counter_resets_the_tracker(self):
        mk = sys.modules["mechkit"]
        self.task._TRACKER.reset()
        rigstate.offline_reset()
        for _ in range(25):
            self.task.success(self.objects(DONE))
        self.assertEqual(self.task._TRACKER.best, 1.0)
        mk.offline_reset()
        self.task.success(self.objects(DONE))
        self.assertEqual(self.task._TRACKER.steps, 1)

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = "/nonexistent/dir/trace.jsonl"
        try:
            s = list(DONE)
            s[0] = SimpleNamespace(**{**vars(s[0]), "pos": np.array([np.nan, 0, 0])})
            self.assertFalse(self.task.success(self.objects(s)))
        finally:
            os.environ.pop("TASK_TRACE", None)

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for env, want in ((("DEMO_IDLE_ARM", "left"), {"right"}), (("DEMO_IDLE_ARM", "right"), {"left"}),
                          (("DEMO_HOLD_ONLY", "1"), set())):
            os.environ[env[0]] = env[1]
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(env[0])

    def test_demo_fits_each_plate_with_its_own_arm(self):
        t = self.task
        for arm, moves in t.PLAN.items():
            side = arm
            for name, seat in moves:
                self.assertTrue(seat[0].startswith(side))
                self.assertTrue((L.SHOES[name][2] > 0) == (arm == "left"))
                want = "front" if L.SHOES[name][0] == "primary" else "rear"
                self.assertEqual(seat[1], want)
                # picked at the yaw that turns it onto the seat at gripper yaw 0 (+ the trim)
                turn = math.degrees(0.0 - t.pick_yaw(name, seat))
                self.assertAlmostEqual(wrap_deg(L.SHOES[name][3] + turn), L.SEATS[seat][2])
                px, py = t.place_point(name, seat, arm)
                sx, sy, _ = L.SEATS[seat]
                # led PLACE_LEAD along the carry and GRIP_INSET in toward the plate's centre: the shoe rides
                # GRIP_INSET outward of the TCP, so it lands PLACE_LEAD on from its seat
                _, x0, y0, _ = L.SHOES[name]
                d = np.array([sx - x0, sy - y0]) / math.hypot(sx - x0, sy - y0)
                out = np.array([math.cos(math.radians(L.SEATS[seat][2])), math.sin(math.radians(L.SEATS[seat][2]))])
                lands = np.array([px, py]) + t.GRIP_INSET * out
                np.testing.assert_allclose(lands, np.array([sx, sy]) + t.PLACE_LEAD * d, atol=1e-12)
                self.assertLess(math.hypot(px - sx, py - sy), t.PLACE_LEAD + t.GRIP_INSET)
                # picked GRIP_INSET inside the mid-arc point
                kx, ky = t.pick_point(name)
                self.assertAlmostEqual(math.hypot(kx - x0, ky - y0), t.GRIP_INSET)
                self.assertLess(kx, x0)
        self.assertEqual(sorted(s for moves in t.PLAN.values() for _, s in moves), sorted(L.SEAT_KEYS))
        self.assertAlmostEqual(math.degrees(t.YAW_TRIM), -1.15)
        # Every wait at a close or a release is at least 1.5 s; each settle before them at least 0.5 s.
        for arm, wps in t.demo().items():
            for a, b in zip(wps, wps[1:]):
                if a.finger != b.finger:
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertGreaterEqual(a.duration, 0.3, a.label)

    def test_open_fingers_clear_the_whole_shoe(self):
        """With the whole shoe (the checker sees only its grip block): at the pick and at the place the open pads
        (8 mm thick, 40 mm wide, centred on the closing axis) stay 3 mm or more from every part of the shoe, and from
        the other shoe on the plate; closed, they reach the shoe's faces only (the block and its two neighbours)."""
        t = self.task
        gap = -2 * t.OPEN
        for arm, moves in t.PLAN.items():
            for name, seat in moves:
                kind, x, y, yaw = L.SHOES[name]
                px, py = t.place_point(name, seat, arm)
                other = (seat[0], "rear" if seat[1] == "front" else "front")
                kx, ky = t.pick_point(name)
                syaw = math.radians(L.SEATS[seat][2])
                gx, gy = px + t.GRIP_INSET * math.cos(syaw), py + t.GRIP_INSET * math.sin(syaw)
                for where, (cx, cy), psi, s in (
                        ("pick", (kx, ky), t.pick_yaw(name, seat), shoe(x, y, yaw, bottom=L.TOP)),
                        ("place", (px, py), 0.0, shoe(gx, gy, L.SEATS[seat][2])),  # in the gripper (GPU)
                        ("place, offline", (px, py), 0.0, shoe(px, py, L.SEATS[seat][2])),  # centred (checker)
                        ("place, the plate's other seat", (px, py), 0.0, seated(other))):
                    close = np.array([math.cos(psi + math.pi / 2), math.sin(psi + math.pi / 2)])
                    across = np.array([-close[1], close[0]])
                    worst = math.inf
                    for side in (-1, 1):
                        for u in np.linspace(0, 0.008, 3):
                            for v in np.linspace(-0.020, 0.020, 9):
                                p = np.array([cx, cy]) + close * side * (gap / 2 + u) + across * v
                                worst = min(worst, footprint_distance(s, (*p, 0.0)))
                    self.assertGreater(worst, 0.006, (name, where, worst))
                # closed on the shoe (12 mm commanded), the pads press only on its faces at mid-arc
                s = shoe(x, y, yaw, bottom=L.TOP)
                self.assertEqual(footprint_distance(s, (x, y, 0.0)), 0.0)

    def test_pick_centres_the_jaws_on_the_shoe(self):
        """Closing from 50 mm, the 40 mm wide pads meet the whole shoe (flat outer block face, concave inner side met
        at the pads' edges) within 1 mm of the same moment: the TCP, GRIP_INSET inside the mid-arc point, is within
        0.5 mm of the middle of what the pads close on, so the shoe is not pushed along the bench."""
        t = self.task
        s = shoe(0.0, 0.0, 0.0, bottom=L.TOP)
        out, inn = [], []
        for v in np.linspace(-L.FINGER_HALF_WIDTH, L.FINGER_HALF_WIDTH, 81):
            xs = [u for u in np.linspace(-0.030, 0.015, 901) if footprint_distance(s, (u, v, 0.0)) == 0.0]
            if xs:
                out.append(max(xs))
                inn.append(min(xs))
        outer, inner = max(out), min(inn)
        self.assertAlmostEqual(outer, L.SHOE_HALF_W, delta=1e-4)
        self.assertLess(inner, -0.015)  # the inner pad meets the concave side's horns first
        self.assertAlmostEqual((outer + inner) / 2, -t.GRIP_INSET, delta=0.0005)
        gap = -2 * t.OPEN
        for clear in (gap / 2 - (outer + t.GRIP_INSET), gap / 2 + (inner + t.GRIP_INSET)):
            self.assertGreater(clear, 0.010)

    def test_grippers_stay_apart(self):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        from yam_arena.demo import ArmPlan
        sys.path.insert(0, str(_offline_dir()))
        import yamkin as K
        plans = {a: ArmPlan(w, np.asarray(K.BASES["station"][a], float)) for a, w in self.task.demo().items()}
        end = max(p.duration for p in plans.values())
        closest = math.inf
        for k in range(int(end * 20) + 1):
            pos = {a: np.asarray(p.target(k / 20)[0], float) for a, p in plans.items()}
            closest = min(closest, abs(pos["left"][1] - pos["right"][1]))
        self.assertGreater(closest, 0.30)

    def test_instruction_and_thresholds_in_task_json(self):
        with open(HERE / "task.json") as f:
            tj = json.load(f)
        self.assertEqual(tj["instruction"], self.task.INSTRUCTION)
        for text in ("8 mm", "8 degrees", "14 mm", "15 degrees", "5 degrees", "3 mm", "2 mm", "0.5 s", "1 s",
                     "1,200 steps"):
            self.assertIn(text, tj["scoring"] + tj["run_limit"], text)
        for text in ("0.04 kg", "120 mm", "10 mm", "12 by 14 mm", "90 degrees", "130 degrees"):
            self.assertIn(text, tj["setup"], text)
        for k in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes"):
            for bad in (chr(0x2014), chr(0x2013)):
                self.assertNotIn(bad, tj[k])

    def test_success_hold_outlasts_every_release(self):
        """Every release is followed by a lift of at least 1 s before the next action, and 1.0 needs the shoe let go
        0.5 s and still 1 s: success cannot fire while fingers are on a shoe."""
        self.assertGreaterEqual(self.task.SUCCESS_HOLD_S, 1.0)


@unittest.skipIf(_offline_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the other plate's seats."""

    def test_other_plate_out_of_reach(self):
        sys.path.insert(0, str(_offline_dir()))
        import yamkin as K
        z = L.TOP + L.PLATE_H + L.PAINT_H + 0.018
        for arm, plate in (("left", "right_plate"), ("right", "left_plate")):
            base = K.BASES["station"][arm]
            for side in ("front", "rear"):
                x, y, _ = L.SEATS[(plate, side)]
                for yaw in np.linspace(-math.pi, math.pi, 13):
                    self.assertFalse(K.reachable((x, y, z), float(yaw), base)[0], (arm, plate, side, yaw))
            own = "left_plate" if arm == "left" else "right_plate"
            for side in ("front", "rear"):
                x, y, _ = L.SEATS[(own, side)]
                self.assertTrue(K.reachable((x, y, z), 0.0, base)[0], (arm, own, side))


if __name__ == "__main__":
    unittest.main()
