"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    TIERS, Tracker, bottom_gap, clock, decide, held, in_transfer, in_window, lowest, rotation, standing, tilt_deg,
    valid,
)

MM = 0.001
D, A, C = L.NAMES


def _stub_dir():
    """tools/offline/yam_stub (the offline yam_arena stand-in) in a parent folder, or None (e.g. in the public
    task repository, where the tests that need it are skipped)."""
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _kin_dir():
    """tools/offline (yamkin, the arm's kinematics) in a parent folder, or None."""
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yamkin.py").is_file():
            return d
    return None


NEEDS_STUB = unittest.skipIf(_stub_dir() is None or _kin_dir() is None,
                             "needs the offline yam_arena stub and yamkin (tools/offline)")


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(math.sin(h) * np.array(v, float)) + (math.cos(h),)


def obj(name, x, y, bottom=L.TOP, tilt=0.0, axis="x", yaw=0.0, vel=(0, 0, 0)):
    """One object's state, its lowest point at ``bottom``, tilted ``tilt`` degrees about ``axis`` after ``yaw``."""
    zq = np.array(about("z", yaw))
    tq = np.array(about(axis, tilt))
    # q = tilt * yaw (Hamilton product, xyzw)
    x1, y1, z1, w1 = tq
    x2, y2, z2, w2 = zq
    q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    probe = SimpleNamespace(pos=np.zeros(3), quat_xyzw=q)
    half = -lowest(name, probe)
    c = np.array([x, y, bottom + half])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c - 0.03, high=c + 0.03)


def at(name, phi, r, **kw):
    return obj(name, *L.clock_xy(phi, r), **kw)


def start():
    return [obj(n, *L.START[n][:2]) for n in L.NAMES]


DEMO = [at(n, *L.DEMO_AT[n]) for n in L.NAMES]


def layout_of(d=None, a=None, c=None):
    """The demo's end layout with some objects replaced (each an object state, or a (phi, r) pair)."""
    out = list(DEMO)
    for i, (n, v) in enumerate(zip(L.NAMES, (d, a, c))):
        if v is None:
            continue
        out[i] = at(n, *v) if isinstance(v, tuple) else v
    return out


def run(frames, arms=None, tracker=None):
    t = tracker or Tracker()
    res = None
    for f in frames:
        res = t.update(f, arms)
    return res, t


def settled(objs, n=30, arms=None):
    """``objs`` held still for ``n`` control steps (1.5 s)."""
    return run([objs] * n, arms)


def score_of(objs, arms=None, n=30):
    return settled(objs, n, arms)[0].score


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    """A rigstate snapshot: each arm's pad-tip point (x, y, z world) or None (parked high)."""
    park = {"left": (0.40, 0.30, L.TOP + 0.107), "right": (0.25, -0.30, L.TOP + 0.137)}
    tips = {"left": left or park["left"], "right": right or park["right"]}
    return {s: rigstate.Arm(tcp=np.array(tips[s]) + [0, 0, 0.013], tips=np.array(tips[s], float),
                            opening=left_open if s == "left" else right_open) for s in ("left", "right")}


class Geometry(unittest.TestCase):
    def test_clock_orientation_is_the_supine_patients(self):
        # A supine patient, head toward -y, face up (+z): their left is (head direction) x (face direction) = -x.
        head, face = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        left = np.cross(head, face)
        self.assertTrue(np.allclose(left, [-1, 0, 0]))
        x3, y3 = L.clock_xy(90.0, 0.1)
        self.assertLess(x3, L.MOUTH[0])  # 3 o'clock is the patient's left
        self.assertAlmostEqual(y3, L.MOUTH[1])
        x12, y12 = L.clock_xy(0.0, 0.1)
        self.assertLess(y12, L.MOUTH[1])  # 12 behind the head (-y)
        x6, y6 = L.clock_xy(180.0, 0.1)
        self.assertGreater(y6, L.MOUTH[1])  # 6 toward the feet, over the chair
        self.assertTrue(L.CHAIR[2] <= y6 <= L.CHAIR[3] and L.CHAIR[0] <= x6 <= L.CHAIR[1])
        for phi in (0, 30, 45, 90, 135, 200, 270, 330, 359):
            self.assertAlmostEqual(L.clock_of(*L.clock_xy(phi, 0.15))[0], phi, places=6)
            self.assertAlmostEqual(L.clock_of(*L.clock_xy(phi, 0.15))[1], 0.15, places=9)
        self.assertEqual([L.hours(p) for p in (0, 15, 45, 247.5, 270, 352.5, 105)],
                         ["12:00", "12:30", "1:30", "8:15", "9:00", "11:45", "3:30"])

    def test_windows_sit_inside_the_zones(self):
        h = 30.0  # degrees per hour
        self.assertTrue(0 * h <= L.LH_DENTIST[0] and L.LH_DENTIST[1] <= 5 * h)  # operator 12 to 5
        self.assertTrue(8 * h <= L.LH_ASSISTANT[0] and L.LH_ASSISTANT[1] <= 10 * h)  # assistant 8 to 10
        self.assertTrue(8 * h <= L.LH_CART[0] and L.LH_CART[1] <= 12 * h)  # assistant or static, 8 to 12
        self.assertEqual(L.TRANSFER, (5 * h, 8 * h))
        self.assertTrue(7 * h <= L.RH_DENTIST[0] and L.RH_DENTIST[1] <= 12 * h)  # right-handed operator 7 to 12
        self.assertTrue(2 * h <= L.RH_ASSISTANT[0] and L.RH_ASSISTANT[1] <= 4 * h)  # right-handed assistant 2 to 4
        # the right-handed windows are the left-handed ones' mirror images about the 12-6 axis, give or take a margin
        for lh, rh in ((L.LH_DENTIST, L.RH_DENTIST), (L.LH_ASSISTANT, L.RH_ASSISTANT)):
            self.assertLessEqual(abs((360 - lh[1]) - rh[0]), 15.0)
        for w in (L.LH_DENTIST, L.LH_ASSISTANT, L.LH_CART, L.RH_DENTIST, L.RH_ASSISTANT):
            self.assertTrue(w[1] <= L.TRANSFER[0] or w[0] >= L.TRANSFER[1])  # none overlaps the transfer zone

    def test_lowest_point_and_tilt(self):
        for n in L.NAMES:
            s = obj(n, 0.3, 0.0)
            self.assertAlmostEqual(bottom_gap(n, s), 0.0, places=9)
            self.assertAlmostEqual(s.pos[2], L.TOP + L.HEIGHT[n] / 2, places=9)
            self.assertTrue(standing(n, s))
            t = obj(n, 0.3, 0.0, tilt=9.5, axis="y")
            self.assertAlmostEqual(tilt_deg(t.quat_xyzw), 9.5, places=6)
            self.assertAlmostEqual(bottom_gap(n, t), 0.0, places=9)
            self.assertTrue(standing(n, t))
            self.assertFalse(standing(n, obj(n, 0.3, 0.0, tilt=10.5)))
            self.assertFalse(standing(n, obj(n, 0.3, 0.0, tilt=90)))  # on its side
            self.assertTrue(standing(n, obj(n, 0.3, 0.0, tilt=180)))  # either end up
            self.assertTrue(standing(n, obj(n, 0.3, 0.0, bottom=L.TOP + 0.0049)))
            self.assertFalse(standing(n, obj(n, 0.3, 0.0, bottom=L.TOP + 0.0055)))
            self.assertFalse(standing(n, obj(n, 0.3, 0.0, bottom=L.TOP - 0.0055)))
        # a cart turned about z keeps its height; one rolled onto its 45 mm side is 45 tall but not upright
        self.assertTrue(standing(C, obj(C, 0.3, 0.0, yaw=37)))
        rolled = obj(C, 0.3, 0.0, tilt=90, axis="x")
        self.assertAlmostEqual(bottom_gap(C, rolled), 0.0, places=9)
        self.assertFalse(standing(C, rolled))

    def test_valid(self):
        good = DEMO[0]
        self.assertTrue(valid(good))
        self.assertFalse(valid(SimpleNamespace(**{**vars(good), "pos": np.array([np.nan, 0, 0])})))
        self.assertFalse(valid(SimpleNamespace(**{**vars(good), "quat_xyzw": np.zeros(4)})))
        self.assertFalse(valid(SimpleNamespace(**{**vars(good), "lin_vel": np.array([np.inf, 0, 0])})))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))

    def test_demo_targets_have_margins(self):
        for n, (phi, r) in L.DEMO_AT.items():
            w = {D: L.LH_DENTIST, A: L.LH_ASSISTANT, C: L.LH_CART}[n]
            self.assertGreaterEqual(min(phi - w[0], w[1] - phi), 22.5)  # 39 mm of arc or more at 100 mm
            self.assertGreaterEqual(min(r - L.R_IN, L.R_OUT - r), 0.020)

    def test_start_and_demo_layout_clear_of_the_chair(self):
        def gap(rect, x, y, half):
            x0, x1, y0, y1 = rect
            dx, dy = max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)
            return math.hypot(dx, dy) - half
        for n in L.NAMES:
            for x, y in (L.START[n][:2], L.clock_xy(*L.DEMO_AT[n])):
                for rect in (L.CHAIR, L.HEADREST):
                    self.assertGreater(gap(rect, x, y, L.half_width(n)), 0.025, (n, x, y))
            self.assertGreater(L.clock_of(*L.START[n][:2])[1], L.R_OUT + 0.1)
        # the screenshot: the arms' ready pose covers roughly |y| > 0.45 m at x < 0.2 m
        for n in L.NAMES:
            x, y, _ = L.START[n]
            self.assertFalse(abs(y) > 0.45 and x < 0.2)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        res, _ = settled(start())
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertEqual(res.details["score_line"], 6)

    def test_demo_layout_scores_one(self):
        res, _ = settled(DEMO)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertEqual(res.details["score_line"], 2)

    def test_cart_left_at_start_or_behind_the_head(self):
        self.assertEqual(score_of(layout_of(c=obj(C, *L.START[C][:2]))), 0.75)
        self.assertEqual(score_of(layout_of(c=(15.0, 0.150))), 0.75)  # 12:30, the right-handed static zone
        self.assertEqual(score_of(layout_of(c=(30.0, 0.180))), 0.75)  # 1:00
        self.assertEqual(score_of(layout_of(c=(0.0, 0.150))), 0.75)  # 12:00
        self.assertEqual(score_of(layout_of(c=(300.0, 0.250))), 0.75)  # right side but 250 mm out

    def test_cart_anywhere_on_the_assistants_side(self):
        for phi in (248.0, 270.0, 300.0, 330.0, 352.0):
            for r in (0.085, 0.130, 0.195):
                self.assertEqual(score_of(layout_of(c=(phi, r))), 1.0, (phi, r))

    def test_assistants_stool_left_at_start_or_at_three(self):
        self.assertEqual(score_of(layout_of(a=obj(A, *L.START[A][:2]))), 0.5)
        self.assertEqual(score_of(layout_of(a=(90.0, 0.100))), 0.5)
        self.assertEqual(score_of(layout_of(a=(320.0, 0.100))), 0.5)  # 10:40, outside 8:15-9:45

    def test_textbook_right_handed_layout(self):
        rh = [at(D, 330.0, 0.100), at(A, 90.0, 0.100), at(C, 30.0, 0.130)]  # dentist 11, assistant 3, cart 1
        res, _ = settled(rh)
        self.assertEqual(res.score, 0.25)
        self.assertEqual(res.details["score_line"], 5)
        rh_cart_on_lh_side = [at(D, 330.0, 0.100), at(A, 90.0, 0.100), at(C, 270.0, 0.150)]
        self.assertEqual(score_of(rh_cart_on_lh_side), 0.25)

    def test_one_stool_alone(self):
        self.assertEqual(score_of([at(D, 45, 0.1), obj(A, *L.START[A][:2]), obj(C, *L.START[C][:2])]), 0.5)
        self.assertEqual(score_of([obj(D, *L.START[D][:2]), at(A, 270, 0.1), obj(C, *L.START[C][:2])]), 0.0)
        self.assertEqual(score_of([obj(D, *L.START[D][:2]), at(A, 270, 0.1), at(C, 330, 0.13)]), 0.0)
        self.assertEqual(score_of([at(D, 330, 0.1), obj(A, *L.START[A][:2]), obj(C, *L.START[C][:2])]), 0.0)

    def test_window_edges(self):
        for name, w, slot in ((D, L.LH_DENTIST, "d"), (A, L.LH_ASSISTANT, "a")):
            mid = (w[0] + w[1]) / 2
            for phi, r, inside in ((w[0] + 0.2, 0.1, True), (w[0] - 0.2, 0.1, False), (w[1] - 0.2, 0.1, True),
                                   (w[1] + 0.2, 0.1, False), (mid, 0.0805, True), (mid, 0.0795, False),
                                   (mid, 0.1995, True), (mid, 0.2005, False)):
                s = at(name, phi, r)
                self.assertEqual(in_window(name, s, w), inside, (name, phi, r))
        for phi, r, inside in ((247.7, 0.13, True), (247.3, 0.13, False), (352.3, 0.13, True),
                               (352.7, 0.13, False)):
            self.assertEqual(score_of(layout_of(c=(phi, r))), 1.0 if inside else 0.75, (phi, r))

    def test_full_score_needs_one_second_still(self):
        moving = [layout_of(c=at(C, 330.0, 0.13 + k * 0.0005)) for k in range(12)]  # 0.5 mm a step: 5.5 mm
        res, t = run(moving)
        self.assertEqual(res.score, 0.0)  # not settled
        for _ in range(10):
            res = t.update(DEMO)
        self.assertEqual(res.score, 0.0)  # the last 0.5 s still holds the step back onto the mark
        res = t.update(DEMO)
        self.assertEqual(res.score, 0.75)  # settled for 0.5 s, not yet 1 s
        self.assertFalse(res.success)
        for _ in range(9):
            res = t.update(DEMO)
        self.assertEqual(res.score, 0.75)
        res = t.update(DEMO)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_every_listed_outcome(self):
        cases = {
            1.0: [DEMO],
            0.75: [layout_of(c=obj(C, *L.START[C][:2])), layout_of(c=(15.0, 0.15))],
            0.5: [layout_of(a=obj(A, *L.START[A][:2])), layout_of(a=(90.0, 0.1))],
            0.25: [[at(D, 330, 0.1), at(A, 90, 0.1), at(C, 30, 0.13)]],
            0.0: [[at(D, 270, 0.1), at(A, 60, 0.1), at(C, 330, 0.13)],  # swapped
                  layout_of(c=(210.0, 0.15)), layout_of(a=(210.0, 0.15)),  # transfer zone
                  layout_of(d=obj(D, *L.clock_xy(45, 0.1), bottom=L.TOP + L.CHAIR_H)),  # on the chair top
                  start()],
        }
        for want, layouts in cases.items():
            for lay in layouts:
                self.assertEqual(score_of(lay), want)

    def test_every_score_turns_up(self):
        seen = {score_of(lay) for lay in (DEMO, layout_of(c=(15.0, 0.15)), layout_of(a=(90.0, 0.1)),
                                          [at(D, 330, 0.1), at(A, 90, 0.1), at(C, 30, 0.13)], start())}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_decision_list_is_first_match(self):
        base = dict(released=True, settled=True, still_1s=True, transfer=False, lh_dentist=True, lh_assistant=True,
                    lh_cart=True, rh_dentist=False, rh_assistant=False)
        self.assertEqual(decide(base), (1.0, 2))
        self.assertEqual(decide({**base, "still_1s": False}), (0.75, 3))
        self.assertEqual(decide({**base, "transfer": True}), (0.0, 1))
        self.assertEqual(decide({**base, "released": False}), (0.0, 1))
        self.assertEqual(decide({**base, "settled": False}), (0.0, 1))
        self.assertEqual(decide({**base, "lh_cart": False}), (0.75, 3))
        self.assertEqual(decide({**base, "lh_assistant": False, "rh_assistant": True}), (0.5, 4))
        self.assertEqual(decide({**base, "lh_dentist": False, "rh_dentist": True, "rh_assistant": True,
                                 "lh_assistant": False}), (0.25, 5))
        self.assertEqual(decide({**base, "lh_dentist": False}), (0.0, 6))

    def test_score_hook_reads_the_state_at_hand(self):
        _, t = settled(DEMO)
        self.assertEqual(t.result(DEMO).score, 1.0)
        self.assertEqual(t.best, 1.0)
        res = t.update(start())  # a jump back that is not a reset (the objects move): not settled now
        self.assertEqual(res.score, 0.0)


class ZeroConditions(unittest.TestCase):
    def test_object_in_the_transfer_zone(self):
        for n, i in ((C, 2), (D, 0), (A, 1)):
            for phi in (152.0, 180.0, 210.0, 238.0):
                lay = list(DEMO)
                lay[i] = at(n, phi, 0.15) if phi != 180.0 else obj(n, *L.clock_xy(phi, 0.15),
                                                                     bottom=L.TOP + L.CHAIR_H)
                res, _ = settled(lay)
                self.assertEqual(res.score, 0.0, (n, phi))
                self.assertIn(n, res.details["in_transfer_zone"])
        self.assertTrue(in_transfer(at(C, 210.0, 0.199)))
        self.assertFalse(in_transfer(at(C, 210.0, 0.201)))
        self.assertFalse(in_transfer(at(C, 148.0, 0.15)))
        self.assertFalse(in_transfer(at(C, 242.0, 0.15)))
        # even a tipped or lifted object in the transfer zone zeroes the trial
        self.assertEqual(score_of(layout_of(c=at(C, 210.0, 0.15, tilt=90))), 0.0)

    def test_not_settled(self):
        frames = [layout_of(d=obj(D, *np.add(L.clock_xy(45, 0.1), (k * 0.0003, 0)))) for k in range(30)]
        res, _ = run(frames)  # creeping 0.3 mm per step: 3 mm per 0.5 s
        self.assertEqual(res.score, 0.0)
        self.assertIn(D, res.details["not_settled"])
        frames = [layout_of(d=obj(D, *np.add(L.clock_xy(45, 0.1), (k * 0.0001, 0)))) for k in range(30)]
        self.assertEqual(run(frames)[0].score, 1.0)  # 1 mm per 0.5 s: under 2 mm in both windows, still

    def test_too_early(self):
        res, _ = run([DEMO] * 10)  # under 0.5 s of history
        self.assertEqual(res.score, 0.0)

    def test_tipped_or_lifted_stool_does_not_count(self):
        self.assertEqual(score_of(layout_of(d=at(D, 45, 0.1, tilt=90))), 0.0)
        self.assertEqual(score_of(layout_of(a=at(A, 270, 0.1, tilt=60, axis="y"))), 0.5)
        self.assertEqual(score_of(layout_of(d=at(D, 45, 0.1, bottom=L.TOP + 0.01))), 0.0)  # held up

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        _, t = settled(DEMO)
        steps = len(t.history)
        bad = list(DEMO)
        bad[1] = SimpleNamespace(**{**vars(DEMO[1]), "pos": np.array([np.nan, 0.0, 0.8])})
        res = t.update(bad)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertEqual(t.bad_steps, 1)
        self.assertEqual(len(t.history), steps)
        bad[1] = SimpleNamespace(**{**vars(DEMO[1]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(t.update(bad).score, 0.0)
        self.assertEqual(t.update(DEMO[:2]).score, 0.0)  # an object missing
        self.assertEqual(t.update(DEMO).score, 1.0)  # the history before the blow-up still counts


class Released(unittest.TestCase):
    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0, D) * 1000, 90.4, places=1)
        self.assertAlmostEqual(L.release_reach(1.0, C) * 1000, 102.9, places=1)
        self.assertAlmostEqual(L.release_reach(0.0, D) * 1000, 52.4, places=1)

    def test_parked_arms_release_everything(self):
        self.assertEqual(score_of(DEMO, arms=snap()), 1.0)

    def test_cart_still_gripped_scores_zero(self):
        cx, cy = L.clock_xy(*L.DEMO_AT[C])
        arms = snap(right=(cx, cy, L.TOP + 0.013), right_open=37 / 93.9)
        res, _ = settled(DEMO, arms=arms)
        self.assertEqual(res.score, 0.0)
        self.assertIn(C, res.details["not_released"])

    def test_open_gripper_still_round_it_or_propping_it(self):
        cx, cy = L.clock_xy(*L.DEMO_AT[C])
        self.assertEqual(score_of(DEMO, arms=snap(right=(cx, cy, L.TOP + 0.013))), 0.0)
        self.assertEqual(score_of(DEMO, arms=snap(right=(cx + 0.07, cy, L.TOP + 0.02))), 0.0)  # a finger beside it
        self.assertEqual(score_of(DEMO, arms=snap(right=(cx, cy, L.TOP + 0.045 + 0.011))), 1.0)  # 11 mm over the top
        self.assertEqual(score_of(DEMO, arms=snap(right=(cx + 0.11, cy, L.TOP + 0.02))), 1.0)  # beyond reach

    def test_released_needs_half_a_second_clear(self):
        cx, cy = L.clock_xy(*L.DEMO_AT[C])
        holding = snap(right=(cx, cy, L.TOP + 0.02))
        _, t = settled(DEMO, arms=holding)
        for k in range(10):
            res = t.update(DEMO, snap())
            self.assertEqual(res.score, 0.0, k)  # the gripper was round it within the last 0.5 s
        res = t.update(DEMO, snap())
        self.assertEqual(res.score, 1.0)

    def test_unreadable_arms(self):
        res, t = settled(DEMO, arms=None)
        self.assertEqual(res.score, 1.0)
        self.assertEqual(t.arms_unavailable, 30)
        one = snap(right=(*L.clock_xy(*L.DEMO_AT[C]), L.TOP + 0.02))
        one["left"] = rigstate.Arm(tcp=np.full(3, np.nan), tips=np.full(3, np.nan), opening=1.0)
        res, t = settled(DEMO, arms=one)
        self.assertEqual(res.score, 0.0)  # the readable right arm still counts
        self.assertEqual(t.arms_unavailable_by_side, {"left": 30, "right": 0})

    def test_held_is_checked_per_object(self):
        dx, dy = L.clock_xy(*L.DEMO_AT[D])
        arms = snap(right=(dx, dy, L.TOP + 0.011))
        self.assertTrue(held(D, DEMO[0], arms))
        self.assertFalse(held(A, DEMO[1], arms))
        self.assertFalse(held(C, DEMO[2], arms))


class Resets(unittest.TestCase):
    def test_teleport_back_to_start_is_a_new_episode(self):
        _, t = settled(DEMO)
        self.assertEqual(t.best, 1.0)
        res = t.update(start())
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(res.score, 0.0)

    def test_gradual_return_is_not_a_reset(self):
        _, t = settled(DEMO)
        n = t.steps
        for k in range(1, 6):
            frac = k / 5
            t.update([obj(nm, *(np.array(L.clock_xy(*L.DEMO_AT[nm])) * (1 - frac)
                                + np.array(L.START[nm][:2]) * frac)) for nm in L.NAMES])
        self.assertEqual(t.steps, n + 5)
        self.assertEqual(t.best, 1.0)

    def test_jump_elsewhere_is_not_a_reset(self):
        _, t = settled(DEMO)
        t.update(layout_of(c=(15.0, 0.15)))
        self.assertEqual(t.best, 1.0)

    @NEEDS_STUB
    def test_rigstate_reset_hook_resets_the_task_tracker(self):
        task = load_task()
        objects = {n: s for n, s in zip(L.NAMES, DEMO)}
        for _ in range(30):
            task.success(objects)
        self.assertEqual(task.score(objects), 1.0)
        rigstate.offline_reset()
        self.assertEqual(task._TRACKER.steps, 0)
        self.assertEqual(task.score(objects), 0.0)


class Exploits(unittest.TestCase):
    """The spec's shortcut checks and other unintended strategies."""

    def test_right_handed_habit_scores_a_quarter_at_most(self):
        self.assertEqual(score_of([at(D, 330, 0.1), at(A, 90, 0.1), at(C, 30, 0.13)]), 0.25)
        self.assertEqual(score_of([at(D, 300, 0.15), at(A, 75, 0.15), at(C, 330, 0.13)]), 0.25)

    def test_stools_swapped(self):
        self.assertEqual(score_of([at(D, 270, 0.1), at(A, 60, 0.1), at(C, 330, 0.13)]), 0.0)
        self.assertEqual(score_of([at(D, 270, 0.1), at(A, 45, 0.1), at(C, 330, 0.13)]), 0.0)

    def test_cart_in_the_transfer_zone(self):
        self.assertEqual(score_of(layout_of(c=(210.0, 0.15))), 0.0)
        self.assertEqual(score_of(layout_of(c=(15.0, 0.15))), 0.75)

    def test_cart_still_in_the_gripper_at_the_end(self):
        cx, cy = L.clock_xy(*L.DEMO_AT[C])
        self.assertEqual(score_of(DEMO, arms=snap(right=(cx, cy, L.TOP + 0.013), right_open=0.4)), 0.0)

    def test_nothing_moved(self):
        self.assertEqual(score_of(start()), 0.0)

    def test_hedging_both_layouts_is_impossible(self):
        # one dentist's stool cannot be in both operator windows; parking it on the chair between them earns nothing
        self.assertEqual(score_of(layout_of(d=obj(D, *L.clock_xy(0.0, 0.03), bottom=L.TOP + L.CHAIR_H))), 0.0)
        self.assertEqual(score_of(layout_of(d=(0.0, 0.1))), 0.0)  # 12:00: neither operator window

    def test_stacked_or_on_the_chair(self):
        on_cart = obj(A, *L.clock_xy(330, 0.13), bottom=L.TOP + L.CART_SIZE[2])
        self.assertEqual(score_of(layout_of(a=on_cart)), 0.5)
        self.assertEqual(score_of(layout_of(c=obj(C, *L.clock_xy(300, 0.10), bottom=L.TOP + L.A_STOOL_H))), 0.75)

    def test_lying_on_the_right_spots(self):
        self.assertEqual(score_of([at(D, 45, 0.1, tilt=90), at(A, 270, 0.1, tilt=90), at(C, 330, 0.13, tilt=90)]),
                         0.0)

    def test_random_states_score_only_listed_values(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(300):
            lay = []
            for n in L.NAMES:
                phi, r = rng.uniform(0, 360), rng.uniform(0.0, 0.35)
                lay.append(at(n, phi, r, tilt=rng.choice((0.0, 0.0, 5.0, 30.0, 90.0)),
                              bottom=L.TOP + rng.choice((0.0, 0.0, 0.0, 0.003, 0.04))))
            s = score_of(lay, n=22)
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertGreaterEqual(len(seen), 3)


# -- the task file ---------------------------------------------------------------------------------------------


def load_task():
    for p in (str(_stub_dir()), str(_kin_dir())):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "task" in sys.modules:
        return importlib.reload(sys.modules["task"])
    return importlib.import_module("task")


@NEEDS_STUB
class TaskFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("DEMO_IDLE_ARM", None)
        os.environ.pop("DEMO_HOLD_ONLY", None)
        cls.task = load_task()
        cls.spec = json.loads((HERE / "task.json").read_text())

    def test_objects_match_the_layout(self):
        by = {o.name: o for o in self.task.OBJECTS}
        self.assertEqual(set(by), {"patient_chair", "headrest", "mouth_mark", D, A, C})
        for n in L.NAMES:
            self.assertAlmostEqual(by[n].mass, L.MASS[n])
            self.assertLessEqual(by[n].mass, 0.3)
            self.assertAlmostEqual(by[n].pos[2] - L.HEIGHT[n] / 2, L.TOP)  # standing on the floor
            self.assertEqual(tuple(by[n].pos[:2]), L.START[n][:2])
        for n in ("patient_chair", "headrest", "mouth_mark"):
            self.assertIsNone(by[n].mass)
        self.assertGreater(L.A_STOOL_H, L.D_STOOL_H)  # the assistant's is the taller stool
        self.assertEqual(tuple(by["mouth_mark"].pos[:2]), L.MOUTH)
        self.assertAlmostEqual(by[C].size[1], 0.045)

    def test_hooks_and_trace(self):
        task = load_task()
        objects = {n: s for n, s in zip(L.NAMES, DEMO)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                results = [task.success(objects) for _ in range(30)]
            finally:
                os.environ.pop("TASK_TRACE")
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(results[-1])
        self.assertEqual(task.score(objects), 1.0)
        self.assertEqual(len(rows), 30)
        last = rows[-1]
        for k in ("score", "success", "score_line", "lh_dentist", "lh_assistant", "lh_cart", "transfer",
                  "not_released", "not_settled", "first_step", "arms_unavailable"):
            self.assertIn(k, last)
        self.assertEqual(last[D]["clock"], "1:30")
        self.assertEqual(last[A]["clock"], "9:00")
        self.assertEqual(last[C]["clock"], "11:00")
        self.assertAlmostEqual(last[D]["r_mm"], 100.0, places=1)

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        task = load_task()
        os.environ["TASK_TRACE"] = "/nonexistent/dir/trace.jsonl"
        try:
            bad = {n: s for n, s in zip(L.NAMES, DEMO)}
            bad[A] = SimpleNamespace(**{**vars(DEMO[1]), "pos": np.array([np.nan, 0, 0])})
            self.assertFalse(task.success(bad))
            self.assertEqual(task.score(bad), 0.0)
        finally:
            os.environ.pop("TASK_TRACE")

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for arm, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = arm
            try:
                self.assertEqual(set(self.task.demo()), {other})
            finally:
                os.environ.pop("DEMO_IDLE_ARM")
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(self.task.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")

    def test_place_points_land_in_their_windows(self):
        for arm, names in self.task.PLAN.items():
            for n in names:
                px, py = self.task.place_point(n, arm)
                phi, r = L.clock_of(px, py)
                w = {D: L.LH_DENTIST, A: L.LH_ASSISTANT, C: L.LH_CART}[n]
                self.assertTrue(w[0] + 20 <= phi <= w[1] - 20 and L.R_IN + 0.015 <= r <= L.R_OUT - 0.05, n)
                tx, ty = self.task.target(n)
                self.assertAlmostEqual(math.hypot(px - tx, py - ty), self.task.LEAD, places=9)

    def test_demo_timing(self):
        plan = self.task.demo()
        dur = {a: sum(w.duration for w in wps) for a, wps in plan.items()}
        self.assertAlmostEqual(dur["right"], 32.8, places=6)
        self.assertAlmostEqual(dur["left"], 25.8, places=6)

        def carry_start(wps, k):
            t, seen = 0.0, 0
            for w in wps:
                if w.label == "carry it over its place":
                    if seen == k:
                        return t
                    seen += 1
                t += w.duration
        self.assertAlmostEqual(carry_start(plan["left"], 0), 16.5, places=6)  # after the right rose from the cart
        self.assertAlmostEqual(carry_start(plan["right"], 1), 23.5, places=6)
        for wps in plan.values():
            for a, b in zip(wps, wps[1:]):
                if a.finger != b.finger:
                    self.assertGreaterEqual(b.duration, 1.5)  # every open/close takes 1.5 s

    def test_instruction_is_the_same_everywhere(self):
        self.assertEqual(self.spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        self.assertIn(self.task.INSTRUCTION, readme)

    def test_task_json_states_the_thresholds(self):
        s = self.spec["scoring"]
        for text in ("80 to 200 mm", "12:30 to 3:30", "8:15 to 9:45", "8:15 to 11:45", "8:30 to 11:30",
                     "2:15 to 3:45", "5 to 8 o'clock", "10 degrees", "5 mm of the floor", "2 mm", "0.5 s",
                     "10 mm above", "1.0", "0.75", "0.5", "0.25", "1,200 steps", "2 s"):
            self.assertIn(text, s)
        self.assertIn("demo: 1.0", self.spec["scores"])
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.0)

    def test_success_hold_outlasts_the_last_release(self):
        # with the arms unreadable the cart would count from the end of its slow set-down; settle + open = 2.0 s
        self.assertGreater(self.task.SUCCESS_HOLD_S + L.STILL_WINDOW_S * L.STILL_WINDOWS_FOR_FULL, 0.5 + 1.5)

    def test_text_is_plain(self):
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py"):
            text = (HERE / f).read_text()
            self.assertNotIn("\u2014", text)
            self.assertNotIn("\u2013", text)
            self.assertTrue(text.isascii(), f)


@NEEDS_STUB
class OneArmReach(unittest.TestCase):
    """Straight down, at any heading, neither arm reaches the other side's starting objects."""

    def test_other_sides_objects_are_out_of_reach(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        bases = K.BASES["station"]
        for arm, names in (("left", (D, C)), ("right", (A,))):
            for n in names:
                x, y, _ = L.START[n]
                for yd in range(-180, 180, 30):
                    for h in (0.024, 0.035, 0.06):
                        ok = K.reachable((x, y, L.TOP + h), math.radians(yd), bases[arm])[0]
                        self.assertFalse(ok, (arm, n, yd, h))


if __name__ == "__main__":
    unittest.main()
