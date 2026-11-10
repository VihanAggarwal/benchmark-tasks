"""Offline tests for scoring.py (and task.py against the offline yam_arena stand-in when it is there): synthetic lens
and screen states, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import itertools
import json
import math
import os
import pathlib
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
    element_facts,
    frame,
    judge,
    optic_point,
    rider_corners,
    rotation,
    square_err_deg,
    tilt_deg,
    valid,
)

WINDOW = int(round(L.REST_WINDOW_S * L.CONTROL_HZ))  # 10: an element is at rest after 11 still samples
BX = L.BENCH_X


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1.0, 0], [-math.sin(a), 0, math.cos(a)]])


def mat_quat(r):
    """Rotation matrix -> unit quaternion (x, y, z, w), by the largest of the four candidate terms (Shepperd), so it
    stays exact near 180-degree turns."""
    r = np.asarray(r, dtype=float)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > max(r[0, 0], r[1, 1], r[2, 2]):
        s = 2 * math.sqrt(1 + tr)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] >= r[2, 2]:
        s = 2 * math.sqrt(1 - r[0, 0] + r[1, 1] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2 * math.sqrt(1 - r[0, 0] - r[1, 1] + r[2, 2])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def corners(name, o, r):
    """Every corner of every shape of an element whose base frame is at o, turned r."""
    pts = []
    for c, s in L.BOXES[name]:
        h = np.array(s) / 2
        for sx, sy, sz in itertools.product((-1, 1), repeat=3):
            pts.append(o + r @ (np.array(c) + h * np.array([sx, sy, sz])))
    return np.array(pts)


def elem(name, x, y, yaw=0.0, z=L.BENCH_TOP, roll=0.0, pitch=0.0, vel=(0, 0, 0), origin=True, rot=None,
         rest=False, rest_on=L.BENCH_TOP):
    """An element's state as mechkit reports it: base frame at (x, y, z) (the centre of the rider's underside),
    turned yaw about the vertical after rolling about local x (tipping toward or away from the lamp) and pitching
    about local y (degrees); pos/low/high the world box around every shape. rest=True: z is replaced so that the
    element's lowest point is on ``rest_on`` (a tilted element resting there, not sunk into it)."""
    r = rz(yaw) @ rx(roll) @ ry(pitch) if rot is None else rot
    q = mat_quat(r)
    np.testing.assert_allclose(rotation(q), r, atol=1e-9)
    o = np.array([x, y, z], dtype=float)
    if rest:
        o[2] += rest_on - corners(name, o, r)[:, 2].min()
    pts = corners(name, o, r)
    low, high = pts.min(axis=0), pts.max(axis=0)
    st = SimpleNamespace(pos=(low + high) / 2, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                         low=low, high=high)
    if origin:
        st.origin = o
    return st


def lens_at(u_mm, dx=0.0, **kw):
    """The lens with its centre u mm from the arrow along the bench."""
    return elem("lens", BX + dx, L.OBJ_Y - u_mm / 1000, **kw)


def screen_at(d_mm, dx=0.0, **kw):
    """The screen with its white face d mm from the arrow along the bench (facing the lamp unless yaw says so)."""
    yaw = kw.pop("yaw", 0.0)
    face = L.FACE_Y * math.cos(math.radians(yaw))  # the face's y offset from the rider's centre, turned
    return elem("screen", BX + dx, L.OBJ_Y - d_mm / 1000 - face, yaw=yaw, **kw)


LENS_START = elem("lens", *L.START["lens"])
SCREEN_START = elem("screen", *L.START["screen"])
LENS_DONE = lens_at(200)
SCREEN_DONE = screen_at(400)


def pair(lens, screen):
    return {"lens": lens, "screen": screen}


def run(states, tr=None):
    """Feed the tracker a list of {element: state} steps; return the tracker and the last result."""
    tr = tr or Tracker()
    res = None
    for s in states:
        res = tr.update(s)
    return tr, res


def settled(lens, screen, n=WINDOW + 1, tr=None):
    return run([pair(lens, screen)] * n, tr)


def score_of(lens, screen):
    return settled(lens, screen)[1].score


class Geometry(unittest.TestCase):
    def test_riders_rest_on_the_bench(self):
        for name, st in (("lens", LENS_START), ("screen", SCREEN_START)):
            r, o = frame(name, st)
            np.testing.assert_allclose(rider_corners(name, r, o)[:, 2], L.BENCH_TOP, atol=1e-12)
            self.assertTrue(Tracker.at_start(name, st))

    def test_frame_without_origin_matches(self):
        # without mechkit's origin the frame comes from the box centre: exact for an upright element at any heading
        # (both are symmetric about their vertical centre line), approximate once tilted
        for name, st in (("lens", LENS_DONE), ("screen", SCREEN_START), ("lens", elem("lens", BX, 0.0, 33.0)),
                         ("screen", elem("screen", BX, -0.1, -71.0)), ("screen", screen_at(400, yaw=180.0))):
            bare = elem(name, *st.origin, rot=rotation(st.quat_xyzw), origin=False)
            np.testing.assert_allclose(frame(name, bare)[1], st.origin, atol=1e-9)

    def test_square_folding(self):
        # the lens is the same either way round; the screen's white face must face the lamp
        for h, e in ((0, 0), (180, 0), (90, 90), (175, 5), (-170, 10), (95, 85), (-5, 5), (360, 0)):
            self.assertAlmostEqual(square_err_deg("lens", rz(h)), e, places=9)
        for h, e in ((0, 0), (180, 180), (90, 90), (175, 175), (-170, 170), (-5, 5), (5, 5), (360, 0)):
            self.assertAlmostEqual(square_err_deg("screen", rz(h)), e, places=9)

    def test_tilt(self):
        for roll, pitch, t in ((0, 0, 0), (5, 0, 5), (0, -7, 7), (90, 0, 90), (180, 0, 180)):
            self.assertAlmostEqual(tilt_deg(rx(roll) @ ry(pitch)), t, places=9)

    def test_optic_points(self):
        # the lens's centre is on the axis, over its rider's centre; the screen's face is 11 mm toward the lamp
        np.testing.assert_allclose(optic_point("lens", *frame("lens", LENS_DONE)),
                                   [BX, L.OBJ_Y - 0.200, L.BENCH_TOP + L.AXIS_Z], atol=1e-12)
        np.testing.assert_allclose(optic_point("screen", *frame("screen", SCREEN_DONE)),
                                   [BX, L.OBJ_Y - 0.400, L.BENCH_TOP + L.AXIS_Z], atol=1e-12)
        self.assertAlmostEqual(element_facts("lens", LENS_DONE)["from_object_mm"], 200.0, places=6)
        self.assertAlmostEqual(element_facts("screen", SCREEN_DONE)["from_object_mm"], 400.0, places=6)

    def test_lens_law(self):
        self.assertAlmostEqual(L.image_distance(0.200), 0.200, places=12)
        self.assertAlmostEqual(L.image_distance(0.150), 0.300, places=12)
        self.assertAlmostEqual(L.image_distance(0.300), 0.150, places=12)
        # object-to-screen distance u + v* is smallest, 4f, at u = 2f: no real image on a screen nearer than 400 mm
        us = np.linspace(0.101, 0.600, 5000)
        total = us + np.array([L.image_distance(u) for u in us])
        self.assertAlmostEqual(float(total.min()), 4 * L.F, places=5)
        self.assertAlmostEqual(float(us[total.argmin()]), 2 * L.F, delta=0.001)

    def test_start_layout(self):
        f_l, f_s = element_facts("lens", LENS_START), element_facts("screen", SCREEN_START)
        self.assertAlmostEqual(f_l["from_object_mm"], 50.0, places=6)  # inside f: no real image anywhere
        self.assertAlmostEqual(f_s["from_object_mm"], 280.0, places=6)  # under 4f: no lens position focuses on it
        for u in np.linspace(0.1005, 0.279, 400):  # every lens position between the arrow and the screen
            v = 0.280 - u
            self.assertGreater(abs(v - L.image_distance(u)), L.SHARP_TOL)
        # the elements and the lamp do not overlap: rider edges 30 mm from the slide, 170 mm apart
        self.assertGreater(L.OBJ_Y - (L.START["lens"][1] + L.RIDER_LENS[1] / 2), 0.025)
        self.assertGreater(L.START["lens"][1] - L.RIDER_LENS[1] / 2 - (L.START["screen"][1] + L.RIDER_SCREEN[1] / 2),
                           0.15)

    def test_bench_holds_every_tier(self):
        # the screen can go to 450 mm (0.5 tier, magnified twice) with its rider still on the bench
        rider_end = L.OBJ_Y - 0.450 - L.FACE_Y - L.RIDER_SCREEN[1] / 2
        self.assertGreater(rider_end, L.BENCH_Y0)
        self.assertTrue(element_facts("screen", screen_at(450))["on_bench"])

    def test_standing_tolerance(self):
        self.assertTrue(element_facts("lens", lens_at(200, z=L.BENCH_TOP + 0.0029))["on_bench"])
        self.assertFalse(element_facts("lens", lens_at(200, z=L.BENCH_TOP + 0.0031))["on_bench"])
        self.assertFalse(element_facts("lens", lens_at(200, z=L.BENCH_TOP - 0.0031))["on_bench"])
        # leaning on one edge of its rider it still rests on the bench (the lean is judged by its tilt)
        for roll, pitch in ((4.0, 0.0), (0.0, 4.9), (8.0, 0.0)):
            self.assertTrue(element_facts("screen", screen_at(400, roll=roll, pitch=pitch, rest=True))["on_bench"])
        # held up with its lowest corner 3.1 mm above the bench: not resting on it
        self.assertFalse(element_facts("screen", screen_at(400, pitch=3.0, rest=True,
                                                           rest_on=L.BENCH_TOP + 0.0031))["on_bench"])
        # over the bench: its centre within the bench's width and length
        self.assertFalse(element_facts("lens", lens_at(200, dx=0.031))["on_bench"])
        self.assertFalse(element_facts("screen", elem("screen", BX, L.BENCH_Y0 - 0.001))["on_bench"])


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        tr, res = settled(LENS_START, SCREEN_START)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertFalse(res.details["sharp"])
        self.assertTrue(res.details["elements"]["lens"]["counts"] and res.details["elements"]["screen"]["counts"])

    def test_same_size_sharp_scores_one(self):
        tr, res = settled(LENS_DONE, SCREEN_DONE)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.details["u_mm"], 200.0)
        self.assertAlmostEqual(res.details["v_mm"], 200.0)
        self.assertAlmostEqual(res.details["v_star_mm"], 200.0)
        self.assertAlmostEqual(res.details["magnification"], 1.0)

    def test_six_mm_either_way_still_scores_one(self):
        # at u = 2f the object-to-image distance is at its minimum, so small errors in both barely blur the image
        for du, dd in itertools.product(range(-6, 7), repeat=2):
            self.assertEqual(score_of(lens_at(200 + du), screen_at(400 + dd)), 1.0, (du, dd))

    def test_sharp_at_another_size_scores_half(self):
        self.assertEqual(score_of(lens_at(150), screen_at(450)), 0.5)  # magnified twice
        self.assertEqual(score_of(lens_at(300), screen_at(450)), 0.5)  # half size
        self.assertEqual(score_of(lens_at(175), screen_at(175 + 700 / 3)), 0.5)  # 1.33 times
        # lens at its mark but the screen focused for 190 mm: 1.11 times, sharp, not same size
        self.assertEqual(score_of(lens_at(190), screen_at(190 + 1000 * L.image_distance(0.190))), 0.5)
        # ... with the screen 9 mm nearer, still sharp and within 10 %: 1.0
        self.assertEqual(score_of(lens_at(190), screen_at(190 + 1000 * L.image_distance(0.190) - 9)), 1.0)

    def test_sharpness_band(self):
        v_star = 200.0
        for dv, sharp in ((9.9, True), (-9.9, True), (10.1, False), (-10.1, False)):
            res = settled(lens_at(200), screen_at(200 + v_star + dv))[1]
            self.assertEqual(res.details["sharp"], sharp, dv)
            self.assertEqual(res.score, 1.0 if sharp else 0.25, dv)  # the lens is still at its mark: 0.25

    def test_same_size_band(self):
        for mag, want in ((0.9 + 1e-6, 1.0), (1.1 - 1e-6, 1.0), (0.89, 0.5), (1.11, 0.5)):
            score, d = judge(True, True, 0.2, 0.2 + 0.2 * mag)
            if d["sharp"]:
                self.assertEqual(score, want, mag)
        # judged on v / u from where the screen actually stands
        score, d = judge(True, True, 0.195, 0.195 + 0.212)
        self.assertTrue(d["sharp"] and d["magnification"] > 1.08)
        self.assertEqual(score, 1.0)

    def test_one_element_at_its_mark(self):
        self.assertEqual(score_of(lens_at(200), SCREEN_START), 0.25)  # the screen too near for any image
        self.assertEqual(score_of(LENS_START, screen_at(400)), 0.25)  # the lens inside f: no real image
        self.assertEqual(score_of(lens_at(190), SCREEN_START), 0.25)
        self.assertEqual(score_of(lens_at(210), SCREEN_START), 0.25)
        self.assertEqual(score_of(LENS_START, screen_at(390)), 0.25)
        self.assertEqual(score_of(LENS_START, screen_at(410)), 0.25)
        self.assertEqual(score_of(lens_at(200), screen_at(370)), 0.25)  # both near their marks, image blurred

    def test_marks_edges(self):
        self.assertEqual(score_of(lens_at(189), SCREEN_START), 0.0)
        self.assertEqual(score_of(lens_at(211), SCREEN_START), 0.0)
        self.assertEqual(score_of(LENS_START, screen_at(389)), 0.0)
        self.assertEqual(score_of(LENS_START, screen_at(411)), 0.0)

    def test_nothing_right(self):
        self.assertEqual(score_of(lens_at(120), screen_at(300)), 0.0)
        self.assertEqual(score_of(lens_at(100), screen_at(430)), 0.0)  # at f: the image is at infinity

    def test_every_tier_is_reachable(self):
        cases = {
            0.0: (LENS_START, SCREEN_START),
            0.25: (lens_at(200), SCREEN_START),
            0.5: (lens_at(150), screen_at(450)),
            1.0: (LENS_DONE, SCREEN_DONE),
        }
        for want, (a, b) in cases.items():
            self.assertEqual(score_of(a, b), want)

    def test_tolerances_within_counting(self):
        self.assertEqual(score_of(lens_at(200, dx=0.0079), screen_at(400, dx=-0.0079)), 1.0)
        self.assertEqual(score_of(elem("lens", BX, 0.0, 4.9), elem("screen", BX, -0.211, -4.9)), 1.0)
        self.assertEqual(score_of(elem("lens", BX, 0.0, 180.0), SCREEN_DONE), 1.0)  # the lens either way round
        self.assertEqual(score_of(lens_at(200, roll=3.0, rest=True), screen_at(400, pitch=3.0, rest=True)), 1.0)

    def test_counting_limits(self):
        for lens, counts in ((lens_at(200, dx=0.0085), False), (elem("lens", BX, 0.0, 5.5), False),
                             (lens_at(200, pitch=6.0, rest=True), False), (lens_at(200, pitch=4.5, rest=True), True)):
            res = settled(lens, SCREEN_DONE)[1]
            self.assertEqual(res.details["elements"]["lens"]["counts"], counts)
            self.assertEqual(res.score, 1.0 if counts else 0.25)  # the screen still at its mark


class Zeros(unittest.TestCase):
    def test_lens_tipped(self):
        for roll in (11.0, 30.0, 90.0):
            res = settled(lens_at(200, roll=roll, rest=True), SCREEN_DONE)[1]
            self.assertEqual(res.score, 0.0, roll)
            self.assertTrue(res.details["down"])

    def test_screen_tipped_with_the_lens_at_its_mark(self):
        self.assertEqual(score_of(lens_at(200), screen_at(400, pitch=12.0, rest=True)), 0.0)
        self.assertEqual(score_of(lens_at(200), screen_at(380, roll=-90.0, rest=True)), 0.0)  # lying face up

    def test_upside_down(self):
        self.assertEqual(score_of(lens_at(200, roll=180.0, z=L.BENCH_TOP + L.LENS_H), SCREEN_DONE), 0.0)

    def test_off_the_bench(self):
        # standing upright on the table beside the bench, 6 mm below its top
        self.assertEqual(score_of(lens_at(200, dx=-0.06, z=L.TOP), SCREEN_DONE), 0.0)
        self.assertEqual(score_of(LENS_DONE, screen_at(400, dx=0.07, z=L.TOP)), 0.0)
        self.assertTrue(settled(LENS_DONE, screen_at(400, dx=0.07, z=L.TOP))[1].details["down"])

    def test_tilt_between_count_and_zero(self):
        # leaning 7 degrees: does not count, not tipped
        res = settled(lens_at(200, roll=7.0, rest=True), SCREEN_DONE)[1]
        self.assertFalse(res.details["down"])
        self.assertEqual(res.score, 0.25)
        res = settled(lens_at(200, roll=9.9, rest=True), SCREEN_DONE)[1]
        self.assertFalse(res.details["down"])
        res = settled(lens_at(200, roll=10.1, rest=True), SCREEN_DONE)[1]
        self.assertTrue(res.details["down"])

    def test_zero_while_moving(self):
        tr, _ = settled(LENS_DONE, SCREEN_DONE)
        res = tr.update(pair(lens_at(200, roll=40.0, rest=True), SCREEN_DONE))
        self.assertEqual(res.score, 0.0)


class Rest(unittest.TestCase):
    def test_success_needs_half_a_second_at_rest(self):
        tr = Tracker()
        for i in range(WINDOW):
            self.assertFalse(tr.update(pair(LENS_DONE, SCREEN_DONE)).success, i)
        self.assertTrue(tr.update(pair(LENS_DONE, SCREEN_DONE)).success)

    def test_score_without_rest_history(self):
        self.assertEqual(Tracker().result(pair(LENS_DONE, SCREEN_DONE)).score, 0.0)

    def test_a_moving_element_does_not_count(self):
        steps = [pair(lens_at(200 - 3 * (WINDOW - i)), SCREEN_DONE) for i in range(WINDOW + 1)]
        res = run(steps)[1]
        self.assertFalse(res.details["elements"]["lens"]["at_rest"])
        self.assertEqual(res.score, 0.25)

    def test_jitter_counts_as_rest(self):
        steps = [pair(lens_at(200 + (0.8 if i % 2 else -0.8)), SCREEN_DONE) for i in range(WINDOW + 1)]
        self.assertEqual(run(steps)[1].score, 1.0)

    def test_a_3mm_shift_in_the_window_breaks_rest(self):
        steps = [pair(lens_at(203), SCREEN_DONE)] + [pair(LENS_DONE, SCREEN_DONE)] * WINDOW
        self.assertFalse(run(steps)[1].details["elements"]["lens"]["at_rest"])
        steps = [pair(LENS_DONE, SCREEN_DONE)] * (WINDOW + 1) + [pair(lens_at(203), SCREEN_DONE)]
        self.assertFalse(run(steps)[1].details["elements"]["lens"]["at_rest"])

    def test_a_rocking_element_is_not_at_rest(self):
        # rocking 3 degrees on its rider: the top moves 4.8 mm
        steps = [pair(LENS_DONE, screen_at(400, roll=(3.0 if i % 2 else 0.0), rest=True)) for i in range(WINDOW + 1)]
        self.assertFalse(run(steps)[1].details["elements"]["screen"]["at_rest"])

    def test_turning_on_the_spot_is_not_at_rest(self):
        steps = [pair(elem("lens", BX, 0.0, 1.0 * i), SCREEN_DONE) for i in range(WINDOW + 1)]
        self.assertFalse(run(steps)[1].details["elements"]["lens"]["at_rest"])

    def test_score_is_idempotent(self):
        tr, res = settled(LENS_DONE, SCREEN_DONE)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(pair(LENS_DONE, SCREEN_DONE)).score, 1.0)
        self.assertEqual(tr.steps, n)


class Exploits(unittest.TestCase):
    def test_screen_turned_round(self):
        # the white face away from the lamp: the image falls on the black back; only the lens counts
        res = settled(LENS_DONE, screen_at(400, yaw=180.0))[1]
        self.assertFalse(res.details["elements"]["screen"]["counts"])
        self.assertEqual(res.score, 0.25)
        self.assertEqual(score_of(lens_at(150), screen_at(450, yaw=180.0)), 0.0)

    def test_screen_between_the_lamp_and_the_lens(self):
        # the order swapped: screen face 200 mm from the arrow, lens 400 mm (v < 0)
        res = settled(lens_at(400), screen_at(200))[1]
        self.assertFalse(res.details["sharp"])
        self.assertEqual(res.score, 0.0)

    def test_held_just_above_the_bench(self):
        res = settled(lens_at(200, z=L.BENCH_TOP + 0.005), screen_at(400, z=L.BENCH_TOP + 0.005))[1]
        self.assertFalse(res.details["down"])
        self.assertEqual(res.score, 0.0)

    def test_virtual_image_position(self):
        # the lens inside its focal length (magnifying-glass position) never gives a sharp image on the screen
        for u in (50, 80, 99, 100):
            self.assertLess(score_of(lens_at(u), screen_at(400)), 0.5, u)

    def test_off_centre_pair(self):
        self.assertEqual(score_of(lens_at(200, dx=0.015), screen_at(400, dx=-0.015)), 0.0)

    def test_both_on_top_of_the_lamp_box(self):
        on_lamp = L.BENCH_TOP + L.LAMP_SIZE[2]
        res = settled(elem("lens", BX, L.LAMP_Y, z=on_lamp), elem("screen", BX, L.LAMP_Y, z=on_lamp))[1]
        self.assertFalse(res.details["down"])
        self.assertEqual(res.score, 0.0)

    def test_lying_lens_moved_to_the_mark(self):
        # pushed over and slid to 2f: tipped, so 0 whatever the screen does
        self.assertEqual(score_of(lens_at(200, pitch=90.0, rest=True), SCREEN_DONE), 0.0)

    def test_screen_tilted_to_fake_the_distance(self):
        # leaning the screen does not move its face's axis point far: 9 deg back moves it 7 mm, and it is tipped past
        # the 5 deg count limit
        res = settled(LENS_DONE, screen_at(400, roll=-9.0, rest=True))[1]
        self.assertFalse(res.details["elements"]["screen"]["counts"])
        self.assertEqual(res.score, 0.25)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(LENS_DONE, SCREEN_DONE)
        n = tr.steps
        hist = [len(h) for h in tr.hist.values()]
        for bad in ("nan_pos", "inf_vel", "zero_quat", "nan_origin", "short_pos"):
            b = elem("lens", BX, 0.0)
            if bad == "nan_pos":
                b.pos = np.array([np.nan, 0, 0])
            elif bad == "inf_vel":
                b.lin_vel = np.array([np.inf, 0, 0])
            elif bad == "zero_quat":
                b.quat_xyzw = np.zeros(4)
            elif bad == "nan_origin":
                b.origin = np.array([np.nan, 0, 0])
            else:
                b.pos = np.array([0.3, 0.0])
            self.assertFalse(valid(b), bad)
            res = tr.update(pair(b, SCREEN_DONE))
            self.assertEqual(res.score, 0.0, bad)
            self.assertFalse(res.success)
            self.assertEqual([len(h) for h in tr.hist.values()], hist)
        self.assertEqual(tr.steps, n + 5)
        self.assertEqual(tr.update(pair(LENS_DONE, SCREEN_DONE)).score, 1.0)  # the record carries on

    def test_missing_element_key(self):
        res = Tracker().update({"lens": LENS_DONE})
        self.assertEqual(res.score, 0.0)
        self.assertFalse(valid(None))

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker()
        for _ in range(300):
            a = elem("lens", *rng.uniform([0.0, -0.4, 0.6], [0.6, 0.4, 1.0]), rot=rotation(rng.normal(size=4)))
            b = elem("screen", *rng.uniform([0.0, -0.4, 0.6], [0.6, 0.4, 1.0]), rot=rotation(rng.normal(size=4)))
            res = tr.update(pair(a, b))
            self.assertIn(res.score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(res.details)

    def test_origin_or_box_centre_give_the_same_score(self):
        for a, b in ((LENS_DONE, SCREEN_DONE), (lens_at(150), screen_at(450)), (elem("lens", BX, 0.0, 4.0),
                                                                                 screen_at(400, yaw=4.0))):
            bare = [elem(n, *s.origin, rot=rotation(s.quat_xyzw), origin=False) for n, s in (("lens", a),
                                                                                            ("screen", b))]
            self.assertEqual(score_of(a, b), score_of(*bare))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(LENS_DONE, SCREEN_DONE)
        self.assertGreater(tr.steps, WINDOW)
        res = tr.update(pair(LENS_START, SCREEN_START))
        self.assertEqual(tr.steps, 1)
        self.assertEqual(res.score, 0.0)

    def test_carrying_back_slowly_does_not_reset(self):
        tr, _ = settled(lens_at(60), screen_at(290))
        n = tr.steps
        tr.update(pair(LENS_START, SCREEN_START))  # 10 mm each: not a jump
        self.assertEqual(tr.steps, n + 1)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(LENS_START, SCREEN_START)
        n = tr.steps
        tr.update(pair(LENS_DONE, SCREEN_DONE))
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(LENS_DONE, SCREEN_DONE)
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertIsNone(tr.prev)
        self.assertEqual(sum(len(h) for h in tr.hist.values()), 0)


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
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim(self, lens, screen):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        out = {}
        for name, st, spec in (("lens", lens, self.task.LENS), ("screen", screen, self.task.SCREEN)):
            out[name] = self.mk.state_from_poses(spec, (*st.origin, *st.quat_xyzw), source="sim")
        return out

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in self.task.OBJECTS if o.mass is not None), ["lens", "screen"])
        self.assertAlmostEqual(self.task.LENS.mass, L.LENS_MASS)
        self.assertAlmostEqual(self.task.SCREEN.mass, L.SCREEN_MASS)
        for spec in (self.task.LENS, self.task.SCREEN):
            self.assertLessEqual(spec.mass, 0.30)
        self.assertEqual(len(self.task.LENS.shapes), len(L.LENS_BOXES))
        self.assertEqual(len(self.task.SCREEN.shapes), len(L.SCREEN_BOXES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.5)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)
        self.assertEqual(len([n for n in names if n.startswith("scale_tick_")]), L.TICK_COUNT)

    def test_scorer_geometry_matches_mechkit(self):
        lo, hi = self.task.LENS.local_bounds()
        np.testing.assert_allclose(lo, [-0.025, -0.020, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.025, 0.020, L.LENS_H], atol=1e-12)
        lo, hi = self.task.SCREEN.local_bounds()
        np.testing.assert_allclose(lo, [-0.050, -0.020, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.050, 0.020, L.SCREEN_H], atol=1e-12)
        for spec, name in ((self.task.LENS, "lens"), (self.task.SCREEN, "screen")):
            rest = spec.rest_state()
            self.assertTrue(Tracker.at_start(name, rest))
            f = element_facts(name, rest)
            self.assertTrue(f["on_bench"] and f["centred"])
        st = self.sim(LENS_DONE, SCREEN_DONE)
        for n, ref in (("lens", LENS_DONE), ("screen", SCREEN_DONE)):
            np.testing.assert_allclose(st[n].low, ref.low, atol=1e-9)
            np.testing.assert_allclose(st[n].high, ref.high, atol=1e-9)

    def test_shapes_do_not_overlap(self):
        # PhysX spreads the mass over the colliders' volume, overlaps counting twice: none overlap
        for boxes in (L.LENS_BOXES, L.SCREEN_BOXES):
            for (ca, sa), (cb, sb) in itertools.combinations(boxes, 2):
                gap = np.abs(np.array(ca) - np.array(cb)) - (np.array(sa) + np.array(sb)) / 2
                self.assertGreaterEqual(float(gap.max()), -1e-12, (ca, cb))

    def test_elements_stand_stably(self):
        for spec, rider in ((self.task.LENS, L.RIDER_LENS), (self.task.SCREEN, L.RIDER_SCREEN)):
            com = spec.center_of_mass()
            np.testing.assert_allclose(com[:2], 0.0, atol=1e-12)
            # tipping toward or away from the lamp needs a lean past atan(20 / com height): over 18 degrees
            self.assertGreater(math.degrees(math.atan2(rider[1] / 2, com[2])), 18.0, spec.name)
            self.assertGreater(math.degrees(math.atan2(rider[0] / 2, com[2])), 25.0, spec.name)

    def test_fixed_parts_do_not_overlap_the_elements_or_each_other_at_start(self):
        fixed = [o for o in self.task.OBJECTS if o.mass is None]
        for o in fixed:
            lo = np.array(o.pos) - np.array(o.size) / 2
            self.assertGreaterEqual(lo[2], L.TOP - 1e-12, o.name)
        for spec in (self.task.LENS, self.task.SCREEN):
            glo, ghi = spec.world_bounds()
            for o in fixed:
                if o.name == "optical_bench":
                    continue  # they stand on it
                lo, hi = np.array(o.pos) - np.array(o.size) / 2, np.array(o.pos) + np.array(o.size) / 2
                self.assertTrue(np.any(glo > hi) or np.any(lo > ghi), (spec.name, o.name))
            self.assertAlmostEqual(glo[2], L.BENCH_TOP)

    def test_ticks_mark_every_50mm_from_the_arrow(self):
        ticks = sorted((o for o in self.task.OBJECTS if o.name.startswith("scale_tick_")), key=lambda o: -o.pos[1])
        for k, o in enumerate(ticks):
            self.assertAlmostEqual(L.OBJ_Y - o.pos[1], 0.050 * k)
            self.assertLess(o.pos[0] + o.size[0] / 2, L.BENCH_X - L.BENCH_W / 2)  # beside the bench, not under riders

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.LENS.offline_box()[0]), quat_xyzw=LENS_START.quat_xyzw,
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.LENS.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [*L.START["lens"], L.BENCH_TOP], atol=1e-12)
        self.assertTrue(Tracker.at_start("lens", st))

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(LENS_START, SCREEN_START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(LENS_DONE, SCREEN_DONE)
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
                self.task.success(self.sim(LENS_DONE, SCREEN_DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "down", "u_mm", "v_mm", "v_star_mm", "focus_err_mm",
                          "magnification", "sharp", "same_size", "lens_at_mark", "screen_at_mark", "elements",
                          "lens_state", "screen_state"):
                    self.assertIn(k, row)
                for k in ("counts", "at_rest", "on_bench", "tilt_deg", "square_err_deg", "centred", "tipped",
                          "off_bench", "centre_mm", "off_centre_mm", "from_object_mm", "rider_low_mm",
                          "rider_high_mm"):
                    self.assertIn(k, row["elements"]["screen"])
                self.assertAlmostEqual(row["u_mm"], 200.0)
                self.assertAlmostEqual(row["v_mm"], 200.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(LENS_DONE, SCREEN_DONE)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        for env, arms in (({}, {"left", "right"}), ({"DEMO_IDLE_ARM": "left"}, {"right"}),
                          ({"DEMO_IDLE_ARM": "right"}, {"left"}), ({"DEMO_HOLD_ONLY": "1"}, set())):
            old = {k: os.environ.pop(k, None) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
            os.environ.update(env)
            try:
                self.assertEqual(set(self.task.demo()), arms, env)
            finally:
                for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
                    os.environ.pop(k, None)
                    if old[k] is not None:
                        os.environ[k] = old[k]
        plan = self.task.demo()
        for side, wps in plan.items():
            prev = None
            for w in wps:
                if prev is not None and w.finger != prev.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))  # every open and close >= 1.5 s
                if prev is not None and w.pos[2] < prev.pos[2] and w.pos[2] < L.BENCH_TOP + 0.10:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))  # final descents are slow
                prev = w
        # grips 8 mm narrower than the 22 mm holder and board; opened wide enough round them
        self.assertAlmostEqual(-2 * self.task.LENS_GRIP, L.HOLDER_T - 0.008)
        self.assertAlmostEqual(-2 * self.task.SCREEN_GRIP, L.BOARD_T - 0.008)
        self.assertGreaterEqual(-2 * self.task.LENS_OPEN, L.HOLDER_T + 0.025)
        self.assertGreaterEqual(-2 * self.task.SCREEN_OPEN, L.BOARD_T + 0.030)
        # the lens is held by its holder's top bar, the pad tips above the glass, with pad on the bar
        tips = self.task.GRIP_UP["lens"] - 0.013
        self.assertGreater(tips - L.LENS_Z1, 0.005)
        self.assertGreater(L.HOLDER_Z1 - tips, 0.015)
        self.assertGreater(L.SCREEN_H - (self.task.GRIP_UP["screen"] - 0.013), 0.020)
        # opened over the lens at its start, the pads (8 mm thick) stay clear of the lamp's slide
        pad_far = L.START["lens"][1] - self.task.LENS_OPEN + 0.008
        self.assertGreater(L.OBJ_Y - pad_far, 0.015)
        # the finger pads never come within 8 mm of an element they are not holding (the offline checker skips
        # mechkit assemblies): each pad an 8 mm x 20 mm box from 25 mm above the tool point to the tips 13 mm below
        # it, fingers closing along y (yaw 0), tested against every box of both elements at their start and their
        # place, the arm's own element only before its grip closes and after its fingers have opened
        self.assertEqual(self.task.YAW, 0.0)

        def boxes(name, xy):
            return [(np.array(c) + [xy[0], xy[1], L.BENCH_TOP], np.array(s) / 2) for c, s in L.BOXES[name]]

        def pad_gap(pos, gap, bx):
            worst = math.inf
            for sign in (-1, 1):
                c = np.array([pos[0], pos[1] + sign * (gap / 2 + 0.004), pos[2] + 0.006])
                h = np.array([0.010, 0.004, 0.019])
                for bc, bh in bx:
                    worst = min(worst, float(np.max(np.abs(c - bc) - h - bh)))
            return worst

        places = {n: (L.AIM[n][0] + self.task.PLACE_TRIM[n][0], L.AIM[n][1] + self.task.PLACE_TRIM[n][1])
                  for n in L.ELEMENTS}
        for side, wps in plan.items():
            own = self.task.ARM_OF[side]
            other = "screen" if own == "lens" else "lens"
            fixed = boxes(other, L.START[other]) + boxes(other, places[other])
            phase = "before"  # own element at its start until the grip closes, at its place once the fingers open
            closest = math.inf
            for a, b in zip(wps, wps[1:]):
                if b.finger > a.finger:
                    phase = "held"
                for s in np.linspace(0.0, 1.0, 101):
                    pos = np.array(a.pos) + s * (np.array(b.pos) - np.array(a.pos))
                    gap = -2 * (a.finger + s * (b.finger - a.finger))
                    bx = list(fixed)
                    if phase == "before":
                        bx += boxes(own, L.START[own])
                    elif phase == "placed":
                        bx += boxes(own, places[own])
                    closest = min(closest, pad_gap(pos, gap, bx))
                if b.finger < a.finger:
                    phase = "placed"
            self.assertGreaterEqual(closest, 0.008, side)
        # the place points: lens 200 mm from the arrow, screen face 400 mm, both on the centre line
        for side, name, d in (("left", "lens", 0.200), ("right", "screen", 0.400)):
            low = [w for w in plan[side] if w.label.startswith("slowly lower")]
            self.assertEqual(len(low), 1)
            self.assertAlmostEqual(low[0].pos[0], L.BENCH_X)
            face = L.FACE_Y if name == "screen" else 0.0
            self.assertAlmostEqual(L.OBJ_Y - (low[0].pos[1] + face), d)
        # the left arm carries the lens toward the middle only once the right arm is on its way back
        left, right = plan["left"], plan["right"]

        def t_end(wps, prefix):
            i = next(k for k, w in enumerate(wps) if w.label.startswith(prefix))
            return sum(w.duration for w in wps[:i + 1])
        t_left_carry_starts = t_end(left, "carry it") - next(w for w in left if w.label.startswith("carry it")).duration
        self.assertGreater(t_left_carry_starts, t_end(right, "straight up") - 1.0)
        # rider lifted clear of the bench while carried, and set down slowly from 2 mm
        self.assertGreater(self.task.LIFT, 0.015)
        self.assertLessEqual(self.task.DROP, 0.003)


if __name__ == "__main__":
    unittest.main()
