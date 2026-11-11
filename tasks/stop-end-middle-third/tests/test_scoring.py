"""Offline tests for scoring.py and task.py: synthetic stop-end states and synthetic arm snapshots fed through
rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, beam_clearance, evaluate, girder_of, lowest, rotation, square_err_deg, stop_info, tilt_deg, valid,
    zone_gap, zone_radius,
)

HZ = 20.0
TOP = L.TOP
TIPS = 0.013  # pad tips below the TCP
T = L.STOP_SIZE[0]  # 25 mm thick, along x when placed
GOOD_X = (L.BAND_LO + L.MID_HI) / 2  # 0.3175: the middle of the band
MID_SPAN = (L.COL_X[0] + L.COL_X[1]) / 2  # 0.290: the obvious spot
GY = L.GIRDER_Y
L_PARK = np.array([0.22, 0.32, TOP + 0.15])
R_PARK = np.array([0.22, -0.32, TOP + 0.15])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def state(center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like stop-end: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ np.array(L.STOP_HALF, float)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def stop(x, y=GY, turn=0.0, lift=0.0, lean_x=0.0, lean_y=0.0, flip=False, lie=None):
    """A stop-end standing upright with its length across a girder (along y), centred at (x, y): turned ``turn``
    degrees about the vertical, its lowest point ``lift`` above the table, leaned ``lean_x`` degrees about world x
    (top toward -y) or ``lean_y`` about world y, upside down with ``flip``. ``lie``: 'face' lying on a 40 x 60 face,
    'edge' on a 25 x 60 face, 'end_on' standing on a 25 x 40 end but with its length along x."""
    q = about("z", turn)
    if lie == "face":
        q = qmul(q, about("y", 90.0))
    elif lie == "edge":
        q = qmul(q, about("x", 90.0))
    elif lie == "end_on":
        q = about("z", 90.0 + turn)
    if flip:
        q = qmul(q, about("x", 180.0))
    if lean_x:
        q = qmul(about("x", lean_x), q)
    if lean_y:
        q = qmul(about("y", lean_y), q)
    s = state((x, y, 0.0), q)
    z = TOP + lift - (s.low[2] - s.pos[2])  # its lowest point at TOP + lift
    return state((x, y, z), q)


def bay(xl=GOOD_X, xr=GOOD_X, **kw):
    """Both stop-ends standing across their girders at xl (left) and xr (right)."""
    return {"stop_end_left": stop(xl, GY, **kw), "stop_end_right": stop(xr, -GY, **kw)}


def start():
    return {n: stop(x, y, turn=math.degrees(yaw)) for n, (x, y, yaw) in L.STOP_START.items()}


def snap(left_tcp=L_PARK, left_open=1.0, right_tcp=R_PARK, right_open=1.0):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def score_of(stops, arms="park"):
    if arms == "park":
        arms = snap()
    return evaluate(stops, arms)[0]


def settle(tracker, stops, n=12, arms="park"):
    """Feed the same state n steps (still)."""
    res = None
    for _ in range(n):
        res = tracker.update(stops, snap() if arms == "park" else arms)
    return res


def at_x_min(xmin, y=GY, **kw):
    """A square stop-end whose outline starts at xmin along x."""
    return stop(xmin + T / 2, y, **kw)


class Geometry(unittest.TestCase):
    def test_rule_numbers(self):
        self.assertAlmostEqual(L.SPAN, 0.300)  # 4.5 m at 1:15
        self.assertAlmostEqual(L.MID_LO, 0.240)
        self.assertAlmostEqual(L.MID_HI, 0.340)
        self.assertAlmostEqual(L.BEAM_FAR_FACE - L.BEAM_NEAR_FACE, L.BEAM_W)
        self.assertAlmostEqual(L.OFFSET, 0.040)
        self.assertAlmostEqual(L.BAND_LO, 0.295)
        # the near side of the cross beam is outside the middle third: only one band meets both rules
        self.assertLess(L.BEAM_NEAR_FACE - L.OFFSET, L.MID_LO)
        # the band is 45 mm; a square 25 mm stop-end has 20 mm of play in it
        self.assertAlmostEqual(L.MID_HI - L.BAND_LO - T, 0.020)
        # the obvious mid-span spot is in the middle third but too near the beam
        self.assertAlmostEqual(MID_SPAN, 0.290)
        self.assertLess(MID_SPAN - T / 2 - L.BEAM_FAR_FACE, L.OFFSET)

    def test_forms_meet_without_overlap(self):
        forms = {n: (x, y, sx, sy) for n, x, y, sx, sy in L.FORMS}
        o = forms["form_left_outer"]
        self.assertAlmostEqual(o[1] - o[3] / 2, L.TROUGH_OUT)
        self.assertAlmostEqual(o[0] - o[2] / 2, L.COL_FACE_X[0])
        self.assertAlmostEqual(o[0] + o[2] / 2, L.COL_FACE_X[1])
        near, far = forms["form_left_inner_near"], forms["form_left_inner_far"]
        self.assertAlmostEqual(near[1] + near[3] / 2, L.TROUGH_IN)
        self.assertAlmostEqual(near[0] + near[2] / 2, L.BEAM_NEAR_FACE)  # the gap is the cross beam's trough
        self.assertAlmostEqual(far[0] - far[2] / 2, L.BEAM_FAR_FACE)
        bn, bf = forms["beam_form_near"], forms["beam_form_far"]
        self.assertAlmostEqual(bn[0] + bn[2] / 2, L.BEAM_NEAR_FACE)
        self.assertAlmostEqual(bf[0] - bf[2] / 2, L.BEAM_FAR_FACE)
        self.assertAlmostEqual(bn[3] / 2, near[1] - near[3] / 2)  # the cross beam's forms end at the girders' forms
        self.assertAlmostEqual(2 * L.TROUGH_HALF - L.STOP_SIZE[1], 0.010)  # 5 mm each side when centred

    def test_start_layout(self):
        for n, s in start().items():
            i = stop_info(s, snap())
            self.assertIsNone(i["girder"], n)
            self.assertEqual(i["score"], 0.0)
            # 60 mm clear of the outer forms, and outside the start pose's shadow on the camera
            self.assertGreaterEqual(abs(float(s.pos[1])) - 0.0236 - (L.TROUGH_OUT + L.FORM_T), 0.060)
            self.assertLess(float(np.max(np.abs(s.high[1]))), 0.45)
        self.assertEqual(score_of(start()), 0.0)

    def test_tilt_square_lowest(self):
        s = stop(GOOD_X)
        self.assertAlmostEqual(tilt_deg(s), 0.0)
        self.assertAlmostEqual(square_err_deg(s), 0.0)
        self.assertAlmostEqual(lowest(s), 0.0, places=9)
        self.assertAlmostEqual(tilt_deg(stop(GOOD_X, lean_x=7.0)), 7.0, places=6)
        self.assertAlmostEqual(square_err_deg(stop(GOOD_X, turn=-8.0)), 8.0, places=6)
        self.assertAlmostEqual(square_err_deg(stop(GOOD_X, turn=172.0)), 8.0, places=6)
        self.assertAlmostEqual(tilt_deg(stop(GOOD_X, flip=True)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(stop(GOOD_X, lie="face")), 90.0, places=6)
        self.assertAlmostEqual(square_err_deg(stop(GOOD_X, lie="end_on")), 90.0, places=6)
        self.assertAlmostEqual(lowest(stop(GOOD_X, lift=0.004)), 0.004, places=9)

    def test_girder_of_and_beam_clearance(self):
        self.assertEqual(girder_of(stop(GOOD_X)), "left")
        self.assertEqual(girder_of(stop(GOOD_X, -GY)), "right")
        self.assertIsNone(girder_of(stop(0.245, 0.0)))  # the cross beam's trough
        self.assertIsNone(girder_of(stop(0.140, GY, lift=0.040)))  # on a column
        self.assertIsNone(girder_of(stop(0.300, 0.250)))  # outside the outer form
        self.assertAlmostEqual(beam_clearance(0.295, 0.320), 0.040)
        self.assertAlmostEqual(beam_clearance(0.180, 0.205), 0.030)
        self.assertEqual(beam_clearance(0.240, 0.265), 0.0)
        self.assertAlmostEqual(beam_clearance(L.BEAM_FAR_FACE + 0.0005, 0.2805), 0.0005)  # just past the face
        self.assertAlmostEqual(beam_clearance(0.2095, L.BEAM_NEAR_FACE - 0.0005), 0.0005)
        self.assertEqual(beam_clearance(0.230, 0.236), 0.0)

    def test_finger_zone(self):
        arms = snap(left_open=1.0, right_open=0.0)
        self.assertAlmostEqual(zone_radius(arms["left"]), math.hypot(L.FULL_GAP / 2 + 0.010, 0.020))
        self.assertAlmostEqual(zone_radius(arms["left"]), 0.060, delta=0.001)
        self.assertAlmostEqual(zone_radius(arms["right"]), 0.022, delta=0.001)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(start()), 0.0)

    def test_both_in_the_band_scores_one(self):
        self.assertEqual(score_of(bay()), 1.0)
        self.assertEqual(score_of(bay(flip=True)), 1.0)  # either end down

    def test_both_at_mid_span_scores_half(self):
        score, d = evaluate(bay(MID_SPAN, MID_SPAN), snap())
        self.assertEqual(score, 0.5)
        self.assertEqual((d["girder_left"], d["girder_right"]), (0.25, 0.25))
        self.assertIn("too near", d["stop_end_left"]["why"])

    def test_one_right_one_at_mid_span(self):
        self.assertEqual(score_of(bay(GOOD_X, MID_SPAN)), 0.75)
        self.assertEqual(score_of(bay(MID_SPAN, GOOD_X)), 0.75)

    def test_one_right_other_left_on_the_table(self):
        b = bay()
        b["stop_end_right"] = start()["stop_end_right"]
        self.assertEqual(score_of(b), 0.5)
        b["stop_end_right"] = stop(MID_SPAN, -GY, lie="face")
        self.assertEqual(score_of(b), 0.5)
        b = bay(MID_SPAN)
        b["stop_end_right"] = start()["stop_end_right"]
        self.assertEqual(score_of(b), 0.25)

    def test_every_total_is_reachable(self):
        spots = {0.5: GOOD_X, 0.25: MID_SPAN, 0.0: 0.200}  # band, middle third near the beam, near a column
        seen = set()
        for a, xa in spots.items():
            for b, xb in spots.items():
                got = score_of(bay(xa, xb))
                self.assertEqual(got, a + b)
                seen.add(got)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_band_limits_inclusive(self):
        for g, y in (("left", GY), ("right", -GY)):
            b = bay()
            name = f"stop_end_{g}"
            b[name] = at_x_min(L.BAND_LO, y)  # exactly 40 mm from the beam's face
            self.assertEqual(score_of(b), 1.0)
            b[name] = at_x_min(L.BAND_LO - 0.0005, y)  # 39.5 mm: middle third only
            self.assertEqual(score_of(b), 0.75)
            b[name] = at_x_min(L.MID_HI - T, y)  # its far face exactly on the third point
            self.assertEqual(score_of(b), 1.0)
            b[name] = at_x_min(L.MID_HI - T + 0.0005, y)  # 0.5 mm past it: outside the middle third
            self.assertEqual(score_of(b), 0.5)
            b[name] = at_x_min(L.MID_LO, y)  # its near face on the near third point: middle third
            self.assertEqual(score_of(b), 0.75)
            b[name] = at_x_min(L.MID_LO - 0.0005, y)
            self.assertEqual(score_of(b), 0.5)

    def test_position_is_the_whole_stop_end_not_its_centre(self):
        # centre at 0.300 is in the band, but its near face is 32.5 mm from the beam: middle third only
        self.assertEqual(score_of(bay(0.300, GOOD_X)), 0.75)
        # centre at 0.335 is in the middle third, its far face 7.5 mm past it: nothing
        self.assertEqual(score_of(bay(0.335, GOOD_X)), 0.5)

    def test_square_within_ten_degrees(self):
        self.assertEqual(score_of(bay(turn=9.0)), 1.0)
        self.assertEqual(score_of(bay(turn=-9.0)), 1.0)
        self.assertEqual(score_of(bay(turn=11.0)), 0.0)
        # turned, its outline widens along x (31 mm at 9 degrees): 8 mm toward the beam it leaves the band
        self.assertEqual(score_of(bay(GOOD_X - 0.008, GOOD_X, turn=9.0)), 0.75)

    def test_upright_within_ten_degrees(self):
        self.assertEqual(score_of(bay(lean_x=9.0)), 1.0)
        self.assertEqual(score_of(bay(lean_y=9.0)), 1.0)
        self.assertEqual(score_of(bay(lean_x=11.0)), 0.0)
        self.assertEqual(score_of(bay(lean_y=-11.0)), 0.0)

    def test_on_the_floor_within_five_mm(self):
        self.assertEqual(score_of(bay(lift=0.0049)), 1.0)
        self.assertEqual(score_of(bay(lift=0.0051)), 0.0)

    def test_in_the_trough(self):
        b = bay()
        b["stop_end_left"] = stop(GOOD_X, GY + 0.005)  # touching the outer form's face
        self.assertEqual(score_of(b), 1.0)
        b["stop_end_left"] = stop(GOOD_X, GY + 0.0085)  # 3.5 mm into the form: not in the trough
        self.assertEqual(score_of(b), 0.5)

    def test_let_go(self):
        b = bay()
        x_top = TOP + L.STOP_SIZE[2]
        # a closed gripper with its finger zone 25 mm above the top: let go
        zr = zone_radius(snap(right_open=0.0)["right"])
        arms = snap(right_tcp=[GOOD_X, -GY, x_top + zr + 0.025 + TIPS], right_open=0.0)
        self.assertEqual(score_of(b, arms), 1.0)
        # 15 mm above: not let go, that girder 0
        arms = snap(right_tcp=[GOOD_X, -GY, x_top + zr + 0.015 + TIPS], right_open=0.0)
        score, d = evaluate(b, arms)
        self.assertEqual(score, 0.5)
        self.assertEqual(d["stop_end_right"]["why"], "not let go")
        self.assertAlmostEqual(d["stop_end_right"]["right_zone_mm"], 15.0, delta=0.1)
        self.assertEqual(d["girder_left"], 0.5)

    def test_arms_unavailable_taken_as_met(self):
        self.assertEqual(score_of(bay(), None), 1.0)
        self.assertEqual(evaluate(bay(), None)[1]["arms_available"], False)

    def test_success_needs_stillness_and_holds(self):
        tr = Tracker(HZ)
        res = tr.update(bay(), snap())
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.success)  # 0.5 s of history not seen yet
        res = settle(tr, bay(), 10)
        self.assertTrue(res.success)
        self.assertEqual(res.details["first_full_step"], 1)
        # sliding 0.5 mm a step along the trough: still in the band, but not still
        for k in range(12):
            res = tr.update(bay(GOOD_X + 0.0005 * k), snap())
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.success)
        self.assertFalse(res.details["still"])

    def test_score_is_idempotent(self):
        tr = Tracker(HZ)
        settle(tr, bay(GOOD_X, MID_SPAN), 12)
        a = tr.result(bay(GOOD_X, MID_SPAN), snap()).score
        b = tr.result(bay(GOOD_X, MID_SPAN), snap()).score
        self.assertEqual(a, b)
        self.assertEqual(a, 0.75)
        self.assertEqual(tr.steps, 12)


class ZeroConditions(unittest.TestCase):
    def check_left_zero(self, s, why=None):
        b = bay()
        b["stop_end_left"] = s
        score, d = evaluate(b, snap())
        self.assertEqual(score, 0.5)
        self.assertEqual(d["girder_left"], 0.0)
        if why:
            self.assertEqual(d["stop_end_left"]["why"], why)

    def test_lying_down_in_the_band(self):
        self.check_left_zero(stop(GOOD_X, lie="face"), "not upright")
        self.check_left_zero(stop(GOOD_X, lie="edge"), "not upright")

    def test_standing_along_the_trough(self):
        # its 40 mm length along x fits the band (x 0.2975-0.3375), but it does not close the trough
        self.check_left_zero(stop(GOOD_X, lie="end_on"), "not square")

    def test_near_a_column_or_beyond_the_middle_third(self):
        self.check_left_zero(stop(0.200), "outside the middle third")
        self.check_left_zero(stop(0.380), "outside the middle third")
        self.check_left_zero(stop(0.180), "outside the middle third")  # 40 mm on the near side of the beam

    def test_in_the_cross_beams_trough(self):
        self.check_left_zero(stop(0.245, 0.120), "not over a girder")

    def test_in_the_junction(self):
        # astride the cross beam's mouth: partly outside the middle third
        self.check_left_zero(stop(0.245), "outside the middle third")
        # just inside the middle third at the mouth: middle third, too near the beam
        b = bay()
        b["stop_end_left"] = at_x_min(L.MID_LO)
        self.assertEqual(score_of(b), 0.75)

    def test_on_top_of_a_form_or_column(self):
        self.check_left_zero(stop(GOOD_X, GY + 0.030, lift=L.FORM_H), "not over a girder")
        self.check_left_zero(stop(GOOD_X, GY + 0.020, lift=L.FORM_H), "not on the floor")
        self.check_left_zero(stop(L.COL_X[0], GY, lift=L.COL_SIZE[2]), "not over a girder")
        # lying across both side forms' tops, over the trough
        self.check_left_zero(stop(GOOD_X, GY, lie="edge", lift=L.FORM_H), "not upright")

    def test_held_in_the_air_over_the_band(self):
        b = bay()
        b["stop_end_left"] = stop(GOOD_X, lift=0.030)
        tcp = [GOOD_X, GY, TOP + 0.030 + 0.052]
        score, d = evaluate(b, snap(left_tcp=tcp, left_open=0.18))
        self.assertEqual(score, 0.5)
        self.assertEqual(d["girder_left"], 0.0)

    def test_astride_the_inner_form(self):
        self.check_left_zero(stop(GOOD_X, GY - 0.012), "outside the trough")


class Exploits(unittest.TestCase):
    def test_both_stop_ends_in_one_girder(self):
        # two 25 mm stop-ends do not both fit in the 45 mm band; a girder scores by the worst of its stop-ends
        self.assertGreater(2 * T, L.MID_HI - L.BAND_LO)
        b = {"stop_end_left": at_x_min(L.MID_HI - T), "stop_end_right": stop(MID_SPAN)}
        score, d = evaluate(b, snap())
        self.assertEqual(score, 0.25)
        self.assertEqual((d["girder_left"], d["girder_right"]), (0.25, 0.0))
        b = {"stop_end_left": stop(GOOD_X), "stop_end_right": stop(0.380)}
        self.assertEqual(score_of(b), 0.0)

    def test_a_second_stop_end_lying_in_a_good_girder(self):
        b = bay()
        b["stop_end_right"] = stop(0.380, GY, lie="face")  # lying in the left trough beside the good one
        score, d = evaluate(b, snap())
        self.assertEqual(score, 0.0)
        self.assertEqual(d["girder_left"], 0.0)

    def test_held_still_on_the_floor_in_the_band(self):
        b = bay()
        tcp = [GOOD_X, GY, TOP + 0.052]
        score, d = evaluate(b, snap(left_tcp=tcp, left_open=17 / 93.9))
        self.assertEqual(score, 0.5)
        self.assertEqual(d["stop_end_left"]["why"], "not let go")

    def test_open_gripper_hovering_round_it(self):
        b = bay()
        tcp = [GOOD_X, GY, TOP + 0.052]
        self.assertEqual(score_of(b, snap(left_tcp=tcp, left_open=45 / 93.9)), 0.5)

    def test_set_at_mid_span_then_slid_into_the_band_counts(self):
        tr = Tracker(HZ)
        settle(tr, bay(MID_SPAN, MID_SPAN), 12)
        self.assertEqual(tr.last.score, 0.5)
        for k in range(1, 11):
            tr.update(bay(MID_SPAN + 0.00275 * k, MID_SPAN + 0.00275 * k), snap())
        res = settle(tr, bay(GOOD_X, GOOD_X), 12)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)


class Arms(unittest.TestCase):
    def test_unavailable_counted(self):
        tr = Tracker(HZ)
        tr.update(bay(), None)
        tr.update(bay(), snap())
        self.assertEqual(tr.arms_unavailable, 1)

    def test_offline_feed_reaches_the_scorer(self):
        arms = snap(left_tcp=[0.3, 0.0, TOP + 0.05], left_open=0.25)
        self.assertAlmostEqual(arms["left"].opening, 0.25)
        np.testing.assert_allclose(arms["left"].tips, [0.3, 0.0, TOP + 0.05 - TIPS])

    def test_zone_gap_sign(self):
        arms = snap(left_tcp=[GOOD_X, GY, TOP + 0.052], left_open=0.18)
        self.assertLess(zone_gap(arms["left"], stop(GOOD_X)), 0.0)

    def test_bad_arms_are_unreadable(self):
        arms = snap()
        broken = {"left": arms["left"], "right": SimpleNamespace(tcp=[np.nan, 0, 0], tips=[0, 0, 0], opening=1.0)}
        self.assertEqual(evaluate(bay(), broken)[1]["arms_available"], False)
        self.assertEqual(evaluate(bay(), {"left": arms["left"]})[1]["arms_available"], False)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr = Tracker(HZ)
        settle(tr, bay(), 12)
        hist = {n: len(h) for n, h in tr.history.items()}
        for bad in (np.nan, np.inf):
            b = bay()
            b["stop_end_right"] = SimpleNamespace(**{**vars(b["stop_end_right"]), "pos": np.array([bad, 0.0, 0.0])})
            res = tr.update(b, snap())
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
        b = bay()
        b["stop_end_left"] = SimpleNamespace(**{**vars(b["stop_end_left"]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(tr.update(b, snap()).score, 0.0)
        b = bay()
        del b["stop_end_left"]
        self.assertEqual(tr.update(b, snap()).score, 0.0)
        self.assertEqual(tr.update({"stop_end_left": None}, snap()).score, 0.0)
        b = bay()
        b["stop_end_left"] = SimpleNamespace(pos=[0.3, 0.2, 0.8])  # missing fields
        self.assertEqual(tr.update(b, snap()).score, 0.0)
        self.assertEqual(tr.bad_steps, 6)
        self.assertEqual({n: len(h) for n, h in tr.history.items()}, hist)
        self.assertTrue(tr.update(bay(), snap()).success)  # the record of the trial is intact
        self.assertFalse(valid(SimpleNamespace(pos=[0, 0, 0])))
        self.assertEqual(tr.result({"stop_end_left": None}, snap()).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = stop(GOOD_X)
        s2 = SimpleNamespace(**{**vars(s), "quat_xyzw": 3.0 * np.asarray(s.quat_xyzw)})
        self.assertEqual(score_of({"stop_end_left": s2, "stop_end_right": stop(GOOD_X, -GY)}), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(7)
        tr = Tracker(HZ)
        seen = set()
        for _ in range(500):
            b = {}
            for n in L.STOP_NAMES:
                q = np.array([rng.uniform(-1, 1) for _ in range(4)])
                q /= np.linalg.norm(q)
                if rng.random() < 0.6:
                    q = about("z", rng.uniform(-15, 15))
                y = rng.choice([GY, -GY]) + rng.uniform(-0.03, 0.03)
                b[n] = state((rng.uniform(0.12, 0.45), y, TOP + 0.030 + rng.uniform(0.0, 0.01)), q)
            res = tr.update(b, snap() if rng.random() < 0.8 else None)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(res.score)
        self.assertTrue({0.0, 0.25}.issubset(seen))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr = Tracker(HZ)
        settle(tr, bay(), 12)
        self.assertEqual(tr.steps, 12)
        tr.update(start(), snap())
        self.assertEqual(tr.steps, 1)
        self.assertIsNone(tr.first_full_step)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker(HZ)
        x0, y0, yaw0 = L.STOP_START["stop_end_left"]
        for k in range(30):  # the left stop-end carried back to its start 4 mm per step, the other there already
            f = 1 - k / 29
            b = start()
            b["stop_end_left"] = stop(x0 + 0.1175 * f, y0 - 0.12 * f, turn=math.degrees(yaw0))
            tr.update(b, snap())
        self.assertEqual(tr.steps, 30)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr = Tracker(HZ)
        settle(tr, start(), 3)
        tr.update(bay(), snap())
        self.assertEqual(tr.steps, 4)

    def test_rigstate_reset_hook_runs_registered_callbacks(self):
        saved = dict(rigstate._RESET_CALLBACKS)
        hits = []
        try:
            rigstate.on_reset(lambda: hits.append(1))
            rigstate.offline_reset()
            self.assertEqual(hits, [1])
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook, the demo, the
    task.json text, and the offline checker on the whole folder."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")

    def test_objects(self):
        objs = self.task.OBJECTS
        names = [o.name for o in objs]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in objs if o.mass is not None], list(L.STOP_NAMES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        spec = {o.name: o for o in objs}
        for n, (x, y, yaw) in L.STOP_START.items():
            np.testing.assert_allclose(spec[n].pos, (x, y, TOP + L.STOP_HALF[2]))
            self.assertEqual(tuple(spec[n].size), L.STOP_SIZE)
            self.assertAlmostEqual(spec[n].yaw, yaw)
            self.assertLessEqual(spec[n].mass, 0.3)
        # the forms' faces are the scorer's faces
        o, i = spec["form_left_outer"], spec["form_left_inner_far"]
        self.assertAlmostEqual(o.pos[1] - o.size[1] / 2, L.TROUGH_OUT)
        self.assertAlmostEqual(i.pos[1] + i.size[1] / 2, L.TROUGH_IN)
        self.assertAlmostEqual(i.pos[0] - i.size[0] / 2, L.BEAM_FAR_FACE)
        ro = spec["form_right_outer"]
        self.assertAlmostEqual(ro.pos[1] + ro.size[1] / 2, -L.TROUGH_OUT)
        bf = spec["beam_form_far"]
        self.assertAlmostEqual(bf.pos[0] - bf.size[0] / 2, L.BEAM_FAR_FACE)
        c = spec["column_near_left"]
        self.assertAlmostEqual(c.pos[0] + c.size[0] / 2, L.COL_FACE_X[0])
        self.assertAlmostEqual(spec["column_far_left"].pos[0] - spec["column_far_left"].size[0] / 2,
                               L.COL_FACE_X[1])
        for n, s in spec.items():
            if n not in L.STOP_NAMES:
                self.assertIsNone(s.mass, n)
        # the arms' start pose covers the table near each base; nothing of the task lies there
        for s in objs:
            self.assertLess(abs(s.pos[1]) + max(s.size[:2]) / 2, 0.45)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(start()))
        self.assertEqual(self.task.score(start()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(bay()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        for _ in range(11):
            snap()
            ok = self.task.success(bay())
        self.assertTrue(ok)
        self.assertEqual(self.task.score(bay()), 1.0)
        self.assertFalse(self.task.success({"stop_end_left": stop(GOOD_X)}))  # missing: no crash, 0
        self.assertEqual(self.task.score({"stop_end_left": stop(GOOD_X)}), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        tmp = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(tmp)
            rigstate.offline_reset()
            snap()
            self.task.success(bay())
            row = json.loads(tmp.read_text().splitlines()[-1])
            for key in ("score", "girder_left", "girder_right", "still", "moved_mm", "stop_end_left",
                        "stop_end_left_pos", "left_tcp", "left_opening", "arms_unavailable", "first_full_step"):
                self.assertIn(key, row)
            for key in ("why", "xmin_mm", "xmax_mm", "beam_clear_mm", "tilt_deg", "square_err_deg", "low_mm",
                        "let_go", "left_zone_mm"):
                self.assertIn(key, row["stop_end_right"])
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(bay()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
            if tmp.exists():
                tmp.unlink()
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        # mirror images, on one clock
        for wl, wr in zip(plan["left"], plan["right"]):
            self.assertAlmostEqual(wl.duration, wr.duration)
            self.assertAlmostEqual(wl.pos[0], wr.pos[0])
            self.assertAlmostEqual(wl.pos[1], -wr.pos[1])
            self.assertAlmostEqual(wl.yaw, -wr.yaw)
        # every open or close takes 1.5 s or more; every final set-down is slow
        for wps in plan.values():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
            for w in wps:
                if w.label.startswith("slowly down"):
                    self.assertGreaterEqual(w.duration, 2.0)
        # the place: the middle of the band, square (yaw -90 left, +90 right), the stop-end wholly in the band
        for arm, yaw in (("left", -math.pi / 2), ("right", math.pi / 2)):
            w = next(w for w in plan[arm] if w.label.startswith("slowly down"))
            self.assertAlmostEqual(w.yaw, yaw)
            self.assertAlmostEqual(abs(w.pos[1]), GY)
            self.assertGreaterEqual(w.pos[0] - T / 2, L.BAND_LO + 0.009)
            self.assertLessEqual(w.pos[0] + T / 2, L.MID_HI - 0.009)
            self.assertAlmostEqual(w.pos[2] - t.GRIP_Z, 0.0005)
            pick = next(w for w in plan[arm] if w.label.startswith("grip"))
            self.assertAlmostEqual(pick.yaw - yaw, L.STOP_START[f"stop_end_{arm}"][2])
            # the slow set-down starts after a pause with the stop-end's bottom at least 5 mm over the forms' tops,
            # so it enters the trough (5 mm clear of each side form) slowly, not at the end of the faster drop
            k = plan[arm].index(w)
            before, pause = plan[arm][k - 2], plan[arm][k - 1]
            self.assertEqual(pause.label, "settle")
            self.assertAlmostEqual(pause.pos[2], before.pos[2])
            bottom = pause.pos[2] - t.GRIP_Z  # the stop-end's bottom above the table (gripped with it on the table)
            self.assertGreaterEqual(bottom - L.FORM_H, 0.005)
            self.assertGreaterEqual(w.duration, 2.5)
        # the pad tips at the place stay well over the forms; a carried stop-end clears the columns
        self.assertGreaterEqual(t.GRIP_Z - TIPS - (TOP + L.FORM_H), 0.010)
        self.assertGreaterEqual(t.CARRY_Z - t.GRIP_Z - (TOP + L.COL_SIZE[2]) + TOP, 0.030)
        # the opened fingers stay inside the trough's length, clear of the stop-end by 10 mm a side
        self.assertAlmostEqual(-t.RELEASE * 2 - T, 0.020)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_task_json_matches_the_task(self):
        d = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "reset_notes",
                  "context", "keywords"):
            self.assertTrue(d.get(k), k)
        self.assertEqual(d["instruction"], self.task.INSTRUCTION)
        self.assertEqual(d["category"], "Impact on jobs")
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        s = d["scoring"]
        for phrase in ("x 24.0 to 34.0 cm", "x 29.5 cm", "10 degrees", "5 mm", "20 mm", "0.25", "the worst",
                       "Possible scores: 0, 0.25, 0.5, 0.75, 1.0"):
            self.assertIn(phrase, s)
        for x, y, _ in L.STOP_START.values():
            self.assertIn(f"x {x * 100:.1f} cm, y {y * 100:+.1f} cm", d["setup"])
            self.assertIn(f"x {x * 100:.1f} cm, y {y * 100:+.1f} cm", d["reset_notes"])
        self.assertIn("1,200 steps", d["run_limit"])

    def test_offline_checker_predicts_one(self):
        """tools/offline/check_task.py --strict --one-arm on this folder (no reach table, for time). Run in-process
        with sys.modules and sys.path restored afterwards (the checker reloads the task's helper modules)."""
        saved_modules, saved_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(_offline_dir()))
        try:
            import check_task
            rep, _, code = check_task.check(HERE, strict=True, one_arm=True, reach=False)
            self.assertEqual(code, 0, rep.fails)
            self.assertEqual(rep.data["playback"]["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(rep.data["one_arm"], {"left_only": 0.5, "right_only": 0.5})
            check_task._forget_task_modules()
        finally:
            for name in [n for n in sys.modules if n not in saved_modules]:
                del sys.modules[name]
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path


if __name__ == "__main__":
    unittest.main()
