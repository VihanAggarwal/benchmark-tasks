"""Offline tests for scoring.py (and task.py against the offline yam_arena stand-in when it is there): synthetic
shield states, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
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
    frame,
    overhang_at,
    overhang_credit,
    panel_corners,
    rotation,
    shield_facts,
    turn_deg,
    valid,
)

WINDOW = int(round(L.REST_WINDOW_S * L.CONTROL_HZ))  # 10: the shield is at rest after 11 still samples
AIM_X = L.PAN_X1 + L.OVERHANG_AIM - L.LIP_FACE_X  # 0.360: the shield's centre for a 60 mm overhang


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


def corners(o, r):
    """(48, 3): every corner of every box of a shield whose base frame is at o, turned r."""
    pts = []
    for c, s, _ in L.SHIELD_BOXES:
        h = np.array(s) / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(o + r @ (np.array(c) + h * np.array([sx, sy, sz])))
    return np.array(pts)


def shield(x, y, yaw=0.0, z=L.TOP, roll=0.0, pitch=0.0, vel=(0, 0, 0), origin=True, rot=None, rest=False):
    """A shield state as mechkit reports it: base frame at (x, y, z) (the centre of its footprint, on the panels'
    undersides), turned yaw about the vertical after rolling about local x and pitching about local y (degrees);
    pos/low/high the world box around every shape. rest=True: z is replaced so that the shield's lowest point is on
    the counter (a tilted shield resting on it, not sunk into it)."""
    r = rz(yaw) @ rx(roll) @ ry(pitch) if rot is None else rot
    q = mat_quat(r)
    np.testing.assert_allclose(rotation(q), r, atol=1e-9)
    o = np.array([x, y, z], dtype=float)
    if rest:
        o[2] += L.TOP - corners(o, r)[:, 2].min()
    pts = corners(o, r)
    low, high = pts.min(axis=0), pts.max(axis=0)
    st = SimpleNamespace(pos=(low + high) / 2, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                         low=low, high=high)
    if origin:
        st.origin = o
    return st


def at(d_mm, y=0.0, yaw=0.0, **kw):
    """A shield standing on the counter with its lip d_mm past the pans' edge (on its centre line), centre at y."""
    return shield(L.PAN_X1 + d_mm / 1000 - L.LIP_FACE_X, y, yaw, **kw)


DONE = at(60.0)  # as the demo leaves it: 60 mm overhang, centred
START = shield(*L.START)


def run(states, tr=None):
    """Feed the tracker a list of states; return the tracker and the last result."""
    tr = tr or Tracker()
    res = None
    for s in states:
        res = tr.update(s)
    return tr, res


def settled(s, n=WINDOW + 1, tr=None):
    return run([s] * n, tr)


def score_of(s):
    return settled(s)[1].score


class Geometry(unittest.TestCase):
    def test_panels_flat_on_the_counter(self):
        r, o = frame(DONE)
        np.testing.assert_allclose(panel_corners(r, o)[:, 2], L.TOP, atol=1e-12)
        f = shield_facts(DONE)
        self.assertTrue(f["standing"] and f["square"] and f["spans_row"])
        self.assertAlmostEqual(f["tilt_deg"], 0.0, places=6)

    def test_frame_without_origin_matches_for_any_heading(self):
        for yaw in (0.0, 7.0, 90.0, 180.0, -33.0):
            a, b = shield(0.30, 0.02, yaw), shield(0.30, 0.02, yaw, origin=False)
            np.testing.assert_allclose(frame(a)[1], frame(b)[1], atol=1e-9)

    def test_turn_reading(self):
        self.assertAlmostEqual(turn_deg(rz(0)), 0.0)
        self.assertAlmostEqual(turn_deg(rz(9)), 9.0)
        self.assertAlmostEqual(abs(turn_deg(rz(180))), 180.0)
        self.assertAlmostEqual(turn_deg(rz(-12)), -12.0)

    def test_overhang_readings(self):
        f = shield_facts(DONE)
        for d in f["overhang"]:
            self.assertAlmostEqual(d, 0.060, places=9)
        # a shield set 30 mm off centre reads the same overhang at both ends
        for d in shield_facts(at(60.0, y=0.030))["overhang"]:
            self.assertAlmostEqual(d, 0.060, places=9)
        # turned 5 deg anticlockwise seen from above: the lip's left (+y) end swings toward the staff side and
        # reads nearer, the right end further out; the mean grows by 40 mm x (1 / cos - 1) (the lip is 40 mm in front
        # of the centre)
        left, right = shield_facts(at(60.0, yaw=5.0))["overhang"]
        a = math.radians(5.0)
        self.assertAlmostEqual(right - left, 2 * L.ROW_HALF * math.tan(a), places=9)
        self.assertAlmostEqual((left + right) / 2, 0.060 + L.LIP_FACE_X * (1 / math.cos(a) - 1), places=9)

    def test_overhang_needs_the_lip_over_each_end(self):
        r, o = frame(at(60.0, y=0.0))
        self.assertIsNotNone(overhang_at(r, o, L.ROW_HALF + 0.040))  # 42 mm past the row's end: still under the lip
        self.assertIsNone(overhang_at(r, o, L.LIP_HALF + 0.001))
        r, o = frame(at(60.0, y=0.200))  # slid along the bar: the lip runs y +0.044 .. +0.356
        self.assertIsNone(overhang_at(r, o, -L.ROW_HALF))
        self.assertIsNotNone(overhang_at(r, o, L.ROW_HALF))
        r, o = frame(at(60.0, y=0.280))  # slid past the row: the lip starts at y +0.124
        self.assertIsNone(overhang_at(r, o, -L.ROW_HALF))
        self.assertIsNone(overhang_at(r, o, L.ROW_HALF))
        r, o = frame(at(60.0, yaw=90.0))
        self.assertIsNone(overhang_at(r, o, L.ROW_HALF))  # turned along the bar: the lip runs along x

    def test_start_layout(self):
        self.assertTrue(Tracker.at_start(START))
        f = shield_facts(START)
        self.assertTrue(f["standing"] and f["square"] and f["spans_row"])
        for d in f["overhang"]:
            self.assertAlmostEqual(d, L.START[0] + L.LIP_FACE_X - L.PAN_X1, places=9)  # -130 mm: short of the pans
        self.assertEqual(score_of(START), 0.0)
        # carried straight forward from the start it would end 30 mm off centre
        self.assertGreater(abs(L.START[1]), L.CENTRE_TOL)
        # the start clears the pans: its lip 49 mm behind their staff-side edge; and stays out of the arms' shadow
        # in the setup screenshot (roughly |y| > 0.45 m and x < 0.2 m)
        self.assertGreater(L.PAN_X0 - (L.START[0] + L.LIP_FACE_X), 0.040)
        self.assertLess(abs(L.START[1]) + L.SHIELD_W / 2, 0.45)

    def test_panels_pass_beside_the_row(self):
        # centred, the panels' inner faces are 42 mm outside the row's ends; the lip's underside 40 mm over the food
        self.assertAlmostEqual(L.PANEL_INNER - L.ROW_HALF, 0.042, places=9)
        self.assertAlmostEqual(L.LIP_Z0 - L.PAN_TOP, 0.040, places=9)
        # the tray slide stops the lip at 94 mm: outside every band, so 75-90 mm is reachable and 94 is not credited
        self.assertAlmostEqual(L.SLIDE_X0 - L.PAN_X1, 0.094, places=9)
        self.assertGreater(L.SLIDE_X0 - L.PAN_X1, L.OVERHANG_PART[1])

    def test_standing_tolerance(self):
        self.assertTrue(shield_facts(at(60.0, z=L.TOP + 0.003))["standing"])
        self.assertFalse(shield_facts(at(60.0, z=L.TOP + 0.0031))["standing"])
        self.assertTrue(shield_facts(at(60.0, z=L.TOP - 0.002))["standing"])  # sunk a little into the counter

    def test_sunk_or_flipped_is_not_standing(self):
        self.assertFalse(shield_facts(at(60.0, z=L.TOP - 0.005))["standing"])  # 5 mm into the counter: a blow-up
        # upside down with its panels' feet level with the counter (the body below it): the tilt rule
        s = at(60.0, roll=180.0)
        r, o = frame(s)
        np.testing.assert_allclose(panel_corners(r, o)[:, 2], L.TOP, atol=1e-12)
        self.assertFalse(shield_facts(s)["standing"])
        self.assertEqual(score_of(s), 0.0)

    def test_layout_matches_the_written_setup(self):
        # the numbers task.json gives (cm): pans x 25.9-34.0, row y +-11.4, food 25 mm up, slide x 43.4-44.4,
        # shield 80 x 320 x 119 mm, glass 90-94 mm up, caps 296 mm apart, start centre (17.0, +3.0)
        self.assertEqual((L.PAN_X0, L.PAN_X1), (0.259, 0.340))
        self.assertAlmostEqual(L.ROW_HALF, 0.114)
        self.assertAlmostEqual(L.PAN_TOP, 0.025)
        self.assertAlmostEqual(L.SLIDE_X0, 0.434)
        self.assertAlmostEqual(L.SLIDE_X0 + L.SLIDE_SIZE[0], 0.444)
        self.assertEqual((L.SHIELD_D, L.SHIELD_W), (0.080, 0.320))
        self.assertAlmostEqual(L.SHIELD_H, 0.119)
        self.assertAlmostEqual(L.GLASS_Z0, 0.090)
        self.assertAlmostEqual(2 * L.CAP_Y, 0.296)
        self.assertEqual(L.START, (0.170, 0.030))
        self.assertEqual((L.OVERHANG_FULL, L.OVERHANG_PART, L.CENTRE_TOL), ((0.045, 0.075), (0.025, 0.090), 0.015))
        self.assertEqual((L.STAND_TOL, L.TILT_MAX_DEG, L.TURN_MAX_DEG), (0.003, 5.0, 10.0))
        self.assertEqual((L.REST_WINDOW_S, L.REST_MOVE), (0.5, 0.003))
        self.assertEqual(WINDOW, 10)  # 0.5 s at 20 Hz: 11 samples

    def test_mass_matches_the_materials(self):
        # task.json: acrylic glass, panels and lip (1.19 g/cm3) and two hollow caps of 0.5 mm stainless sheet (7.9
        # g/cm3, open underneath): 0.25 kg. Solid steel caps would make it 0.51 kg, which the text must not imply.
        acrylic = sum(np.prod(s) for _, s, k in L.SHIELD_BOXES if k != "STEEL") * 1190.0
        cx, cy, cz = L.CAP_SIZE
        cap = (cx * cy + 2 * cx * cz + 2 * cy * cz) * 0.0005 * 7900.0
        self.assertAlmostEqual(acrylic, 0.228, delta=0.001)
        self.assertAlmostEqual(acrylic + 2 * cap, L.SHIELD_MASS, delta=0.010)
        self.assertEqual(L.SHIELD_MASS, 0.25)
        js = json.loads((HERE / "task.json").read_text())
        self.assertIn("one rigid body, 0.25 kg", js["setup"])
        self.assertIn("hollow stainless end cap", js["setup"])

    def test_turn_limit_and_the_band(self):
        self.assertLess(L.overhang_band_deg_limit(), L.TURN_MAX_DEG)  # the band itself bounds the turn to 7.5 deg


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(START), 0.0)

    def test_demo_place_scores_one(self):
        tr, res = settled(DONE)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertEqual(res.details["overhang_left_mm"], 60.0)
        self.assertEqual(res.details["overhang_right_mm"], 60.0)

    def test_the_spec_outcomes(self):
        cases = [
            (at(60.0), 1.0),
            (at(60.0, y=0.030), 0.75),
            (at(35.0), 0.5),
            (at(35.0, y=0.030), 0.25),
            (at(85.0), 0.5),
            (at(85.0, y=-0.030), 0.25),
            (at(-130.0, y=0.030), 0.0),
        ]
        for s, want in cases:
            self.assertEqual(score_of(s), want)

    def test_band_edges(self):
        for d, want in ((45.0, 1.0), (75.0, 1.0), (44.9, 0.5), (75.1, 0.5), (25.0, 0.5), (90.0, 0.5),
                        (24.9, 0.0), (90.1, 0.0), (10.0, 0.0), (0.0, 0.0), (94.0, 0.0)):
            self.assertEqual(score_of(at(d)), want, d)

    def test_centring_edge(self):
        self.assertEqual(score_of(at(60.0, y=0.015)), 1.0)
        self.assertEqual(score_of(at(60.0, y=-0.015)), 1.0)
        self.assertEqual(score_of(at(60.0, y=0.0151)), 0.75)
        self.assertEqual(score_of(at(60.0, y=-0.040)), 0.75)  # panel 2 mm off the row's end: still on the counter

    def test_centring_alone_earns_nothing(self):
        self.assertEqual(score_of(at(20.0)), 0.0)
        self.assertEqual(score_of(at(-130.0)), 0.0)

    def test_both_ends_must_be_in_the_band(self):
        # turned so that one end reads under 45 mm and the other over 75 mm: 0.25 band; centred 0.5
        t = math.degrees(math.atan(0.032 / (2 * L.ROW_HALF)))
        for yaw in (t, -t):
            s = at(60.0, yaw=yaw)
            lo, hi = sorted(shield_facts(s)["overhang"])
            self.assertLess(lo, 0.045)
            self.assertGreater(hi, 0.075)
            self.assertEqual(score_of(s), 0.5)
        # turned 5 deg about the aim: about 50 and 70 mm (task.json), both in the full band
        for yaw in (5.0, -5.0):
            lo, hi = sorted(shield_facts(at(60.0, yaw=yaw))["overhang"])
            self.assertAlmostEqual(lo, 0.050, delta=0.0005)
            self.assertAlmostEqual(hi, 0.070, delta=0.0005)
        self.assertEqual(score_of(at(60.0, yaw=5.0)), 1.0)
        self.assertEqual(score_of(at(60.0, yaw=-5.0)), 1.0)

    def test_overhang_credit_table(self):
        self.assertEqual(overhang_credit([0.060, 0.060]), 0.75)
        self.assertEqual(overhang_credit([0.046, 0.074]), 0.75)
        self.assertEqual(overhang_credit([0.044, 0.060]), 0.25)
        self.assertEqual(overhang_credit([0.030, 0.089]), 0.25)
        self.assertEqual(overhang_credit([0.024, 0.060]), 0.0)
        self.assertEqual(overhang_credit([0.060, 0.091]), 0.0)
        self.assertEqual(overhang_credit([None, 0.060]), 0.0)

    def test_every_tier_is_reachable(self):
        got = {score_of(s) for s in (START, at(35.0, y=0.03), at(35.0), at(60.0, y=0.03), at(60.0))}
        self.assertEqual(got, {0.0, 0.25, 0.5, 0.75, 1.0})


class Zeros(unittest.TestCase):
    def test_a_panel_on_a_pan(self):
        # set 50 mm toward the right end, the right panel lands on the croutons pan: tilted 4.5 deg (under the 5 deg
        # limit), but that panel's feet are 25 mm up
        roll = math.degrees(math.asin(L.PAN_TOP / (2 * L.PANEL_Y)))
        s = at(60.0, y=-0.050, roll=-roll, rest=True)
        f = shield_facts(s)
        self.assertLess(f["tilt_deg"], L.TILT_MAX_DEG)
        self.assertFalse(f["standing"])
        self.assertGreater(f["panel_high_mm"], 20.0)
        self.assertEqual(score_of(s), 0.0)

    def test_held_above_the_counter(self):
        self.assertEqual(score_of(at(60.0, z=L.TOP + 0.005)), 0.0)
        self.assertEqual(score_of(at(60.0, z=L.TOP + 0.030)), 0.0)

    def test_tilted(self):
        self.assertEqual(score_of(at(60.0, pitch=6.0, rest=True)), 0.0)
        self.assertEqual(score_of(at(60.0, roll=6.0, rest=True)), 0.0)

    def test_turned_past_ten_degrees(self):
        # turned 11 deg about a point chosen so both readings stay in 25-90 mm: still 0
        for yaw in (11.0, -11.0):
            s = at(57.0, yaw=yaw)
            lo, hi = sorted(shield_facts(s)["overhang"])
            self.assertTrue(0.025 <= lo and hi <= 0.090, (lo, hi))
            self.assertFalse(shield_facts(s)["square"])
            self.assertEqual(score_of(s), 0.0)

    def test_backwards(self):
        # lip to the staff side, the shield's front (its back now) 60 mm past the pans
        s = shield(L.PAN_X1 + 0.060 - L.LIP_FACE_X, 0.0, 180.0)
        self.assertFalse(shield_facts(s)["square"])
        self.assertEqual(score_of(s), 0.0)

    def test_not_spanning_the_row(self):
        s = at(60.0, y=0.290)
        self.assertFalse(shield_facts(s)["spans_row"])
        self.assertEqual(score_of(s), 0.0)

    def test_upside_down_or_on_its_side(self):
        self.assertEqual(score_of(at(60.0, roll=180.0, rest=True)), 0.0)
        self.assertEqual(score_of(at(60.0, pitch=90.0, rest=True)), 0.0)
        self.assertEqual(score_of(at(60.0, roll=90.0, rest=True)), 0.0)

    def test_moving(self):
        tr = Tracker()
        res = None
        for i in range(WINDOW + 5):
            res = tr.update(at(40.0 + 2.0 * i))  # 2 mm per step: 40 mm/s
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["at_rest"])


class Rest(unittest.TestCase):
    def test_success_needs_half_a_second_at_rest(self):
        tr = Tracker()
        for _ in range(WINDOW):
            self.assertFalse(tr.update(DONE).success)
        self.assertTrue(tr.update(DONE).success)

    def test_score_without_rest_history(self):
        self.assertEqual(Tracker().result(DONE).score, 0.0)

    def test_jitter_counts_as_rest(self):
        states = [at(60.0 + (0.8 if i % 2 else -0.8), y=0.0005 * (i % 3)) for i in range(WINDOW + 1)]
        self.assertEqual(run(states)[1].score, 1.0)

    def test_a_4mm_shift_in_the_window_breaks_rest(self):
        states = [at(56.0)] + [at(60.0)] * WINDOW
        self.assertEqual(run(states)[1].score, 0.0)
        states = [at(57.0)] + [at(60.0)] * WINDOW  # 3 mm: at rest
        self.assertEqual(run(states)[1].score, 1.0)

    def test_turning_on_the_spot_is_not_at_rest(self):
        states = [at(60.0, yaw=0.5 * i) for i in range(WINDOW + 1)]  # 5 deg in 0.5 s: panel feet move 14 mm
        self.assertEqual(run(states)[1].score, 0.0)

    def test_score_is_idempotent(self):
        tr, res = settled(DONE)
        self.assertEqual(tr.result(DONE).score, 1.0)
        self.assertEqual(tr.result(DONE).score, 1.0)
        self.assertEqual(tr.steps, WINDOW + 1)


class Exploits(unittest.TestCase):
    def test_pushed_against_the_tray_slide(self):
        self.assertEqual(score_of(at(94.0)), 0.0)

    def test_propped_on_the_tray_slide(self):
        # pushed onto the slide: the panels' front ends ride up on it (10 mm), the shield tilts back
        pitch = -math.degrees(math.asin(L.SLIDE_SIZE[2] / L.SHIELD_D))
        s = at(98.0, pitch=pitch, rest=True)
        self.assertFalse(shield_facts(s)["standing"])
        self.assertEqual(score_of(s), 0.0)

    def test_slid_along_the_bar_beyond_the_food(self):
        for y in (0.275, -0.300):
            self.assertEqual(score_of(at(60.0, y=y)), 0.0)

    def test_turned_along_the_bar(self):
        self.assertEqual(score_of(shield(0.38, 0.0, 90.0)), 0.0)

    def test_backwards_and_off_centre(self):
        self.assertEqual(score_of(shield(0.36, 0.02, 180.0)), 0.0)

    def test_hovering_over_the_place(self):
        tr, res = settled(at(60.0, z=L.TOP + 0.010))
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["standing"])

    def test_slid_into_place_counts(self):
        # sliding it along the counter instead of carrying it is a real way of putting it back
        states = [at(-130.0 + 10.0 * i, y=0.030 - 0.002 * i) for i in range(16)] + [at(20.0, y=0.0)]
        states += [at(60.0)] * (WINDOW + 1)
        self.assertEqual(run(states)[1].score, 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(DONE)
        n = tr.steps
        for bad in (
            shield(0.36, 0.0, vel=(math.nan, 0, 0)),
            SimpleNamespace(pos=np.array([math.inf, 0, 0]), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                            ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3)),
            SimpleNamespace(pos=np.zeros(3), quat_xyzw=np.zeros(4), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3)),
            SimpleNamespace(pos=np.zeros(3), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                            ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3), origin=np.array([math.nan] * 3)),
        ):
            hist = list(tr.hist)
            res = tr.update(bad)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
            self.assertFalse(res.details["valid_state"])
            self.assertEqual(len(tr.hist), len(hist))
            for a, b in zip(tr.hist, hist):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(tr.steps, n + 4)
        self.assertEqual(tr.update(DONE).score, 1.0)  # the record survived

    def test_missing_fields(self):
        self.assertFalse(valid(None))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        self.assertEqual(Tracker().update(SimpleNamespace(pos=np.zeros(3))).score, 0.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker()
        for _ in range(300):
            r = rz(rng.uniform(-180, 180)) @ rx(rng.uniform(-180, 180)) @ ry(rng.uniform(-90, 90))
            s = shield(rng.uniform(-0.2, 0.8), rng.uniform(-0.6, 0.6), rot=r, z=L.TOP + rng.uniform(-0.05, 0.3))
            res = tr.update(s)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_origin_or_box_centre_give_the_same_score(self):
        for s, s2 in ((at(60.0), at(60.0, origin=False)), (at(35.0, y=0.02), at(35.0, y=0.02, origin=False))):
            self.assertEqual(score_of(s), score_of(s2))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(DONE)
        tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.hist), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(DONE)
        tr.update(shield(L.START[0] + 0.02, L.START[1] - 0.02, 3.0))
        self.assertEqual(tr.steps, 1)

    def test_carrying_back_slowly_does_not_reset(self):
        tr, _ = settled(DONE)
        n = tr.steps
        path = np.linspace(0.0, 1.0, 40)
        for t in path:
            tr.update(shield(AIM_X + t * (L.START[0] - AIM_X), t * L.START[1]))
        self.assertEqual(tr.steps, n + len(path))

    def test_a_jump_onto_the_mark_turned_or_lifted_is_not_a_reset(self):
        for s in (shield(*L.START, 90.0), shield(*L.START, z=L.TOP + 0.02)):
            tr, _ = settled(DONE)
            n = tr.steps
            tr.update(s)
            self.assertEqual(tr.steps, n + 1)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(DONE)
        n = tr.steps
        tr.update(at(10.0, y=0.1))
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(DONE)
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(tr.result(DONE).score, 0.0)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _aabb_gap(lo1, hi1, lo2, hi2) -> float:
    """Euclidean distance between two axis-aligned boxes (0 if they touch or overlap)."""
    d = np.maximum(0.0, np.maximum(lo2 - hi1, lo1 - hi2))
    return float(np.linalg.norm(d))


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim(self, st):
        """A synthetic state turned into the AssemblyState mechkit would report from the simulator."""
        return {"food_shield": self.mk.state_from_poses(self.task.SHIELD, (*st.origin, *st.quat_xyzw), source="sim")}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None], ["food_shield"])
        self.assertAlmostEqual(self.task.SHIELD.mass, L.SHIELD_MASS)
        self.assertLessEqual(self.task.SHIELD.mass, 0.30)
        self.assertEqual(len(self.task.SHIELD.shapes), len(L.SHIELD_BOXES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 3.0)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)
        js = json.loads((HERE / "task.json").read_text())
        self.assertEqual(js["instruction"], self.task.INSTRUCTION)

    def test_scorer_geometry_matches_mechkit(self):
        lo, hi = self.task.SHIELD.local_bounds()
        np.testing.assert_allclose(lo, [-L.SHIELD_D / 2, -L.SHIELD_W / 2, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.SHIELD_D / 2, L.SHIELD_W / 2, L.SHIELD_H], atol=1e-12)
        np.testing.assert_allclose((lo + hi) / 2, L.SHIELD_BBOX_CENTRE, atol=1e-12)
        rest = self.task.SHIELD.rest_state()
        self.assertTrue(Tracker.at_start(rest))
        self.assertTrue(shield_facts(rest)["standing"])
        st = self.sim(DONE)["food_shield"]
        np.testing.assert_allclose(st.low, DONE.low, atol=1e-9)
        np.testing.assert_allclose(st.high, DONE.high, atol=1e-9)

    def test_shield_stands_stably(self):
        com = self.task.SHIELD.center_of_mass()
        self.assertAlmostEqual(com[1], 0.0, places=12)
        self.assertLess(abs(com[0]), 0.006)  # the lip pulls it 5 mm toward the customers
        # tipping forward or back needs a lean past atan((40 - 5) / 80) = 23.6 deg; sideways far more
        self.assertGreater(math.degrees(math.atan2(L.SHIELD_D / 2 - abs(com[0]), com[2])), 20.0)
        self.assertGreater(math.degrees(math.atan2(L.SHIELD_W / 2, com[2])), 60.0)

    def test_parts_do_not_overlap_at_start(self):
        fixed = [o for o in self.task.OBJECTS if o.mass is None]
        boxes = []
        for o in fixed:
            lo, hi = np.array(o.pos) - np.array(o.size) / 2, np.array(o.pos) + np.array(o.size) / 2
            self.assertGreaterEqual(lo[2], L.TOP - 1e-12, o.name)
            boxes.append((o.name, lo, hi))
        slo, shi = self.task.SHIELD.world_bounds()
        for name, lo, hi in boxes:
            self.assertGreater(_aabb_gap(slo, shi, lo, hi), 0.040, name)
        for i, (a, alo, ahi) in enumerate(boxes):
            for b, blo, bhi in boxes[i + 1:]:
                overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
                self.assertFalse(np.all(overlap > 1e-9), (a, b))  # touching (food on its pan) is fine

    def test_offline_proxy_assumptions(self):
        # the offline checker grasps the shield's 80 mm deep bounding box (under the open gripper's 94 mm) and, let go
        # with the feet 2 mm up, ignores supports more than 20 mm above its bottom: the pans' 25 mm tops
        self.assertLess(L.SHIELD_D, 0.094 - 0.010)
        self.assertGreater(L.PAN_TOP, self.task.DROP + 0.020)
        self.assertGreater(self.task.LIFT, L.PAN_TOP)

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.SHIELD.offline_box()[0]), quat_xyzw=START.quat_xyzw,
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.SHIELD.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [*L.START, L.TOP], atol=1e-12)
        self.assertTrue(Tracker.at_start(st))

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(DONE)
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
                self.task.success(self.sim(DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "valid_state", "at_rest", "standing", "square",
                          "spans_row", "overhang_left_mm", "overhang_right_mm", "overhang_credit", "centred",
                          "centre_mm", "turn_deg", "tilt_deg", "panel_low_mm", "panel_high_mm", "shield_state"):
                    self.assertIn(k, row)
                self.assertAlmostEqual(row["overhang_left_mm"], 60.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(DONE)))  # no crash
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
        left, right = plan["left"], plan["right"]
        self.assertEqual(len(left), len(right))
        gz = L.TOP + self.task.GRIP_UP
        for a, b in zip(left, right):  # one clock, mirrored: same timing, grips, heights, x; y mirrored about the row
            self.assertEqual((a.duration, a.finger, a.label), (b.duration, b.finger, b.label))
            self.assertAlmostEqual(a.pos[0], b.pos[0])
            self.assertAlmostEqual(a.pos[2], b.pos[2])
            self.assertAlmostEqual(a.yaw, -b.yaw)
            self.assertGreaterEqual(a.pos[1] - b.pos[1], 0.20)  # two grippers at one x: >= 0.20 m apart
        for side, wps in plan.items():
            prev = None
            for w in wps:
                if prev is not None and w.finger != prev.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))  # every open and close >= 1.5 s
                if prev is not None and w.pos[2] < prev.pos[2] and w.pos[2] < gz + 0.010:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))  # final descents are slow
                prev = w
        # the grip: 8 mm narrower than the cap's 30 mm depth, fingers along x; open past the 80 mm bounding box
        self.assertAlmostEqual(-2 * self.task.CAP_GRIP, L.CAP_SIZE[0] - 0.008)
        self.assertGreater(-2 * self.task.OPEN, L.SHIELD_D)
        # the pad tips stay above the glass (5 mm) with 20 mm of the 25 mm cap between them
        tips = self.task.GRIP_UP - 0.013
        self.assertAlmostEqual(tips - L.CAP_Z0, 0.005)
        self.assertGreater(L.CAP_Z0 + L.CAP_SIZE[2] - tips, 0.015)
        # the place: lip 60 mm past the pans' edge, centred
        place = [w for w in left if w.label == "slowly set it on the counter"][0]
        self.assertAlmostEqual(place.pos[0] + L.LIP_FACE_X - L.PAN_X1, L.OVERHANG_AIM)
        self.assertAlmostEqual(place.pos[1] - L.CAP_Y, 0.0)
        self.assertAlmostEqual(place.pos[2] - L.TOP - self.task.GRIP_UP, self.task.DROP)
        self.assertEqual(self.task.PLACE_TRIM, (0.0, 0.0))

    def test_demo_carry_clears_the_pans_and_the_slide(self):
        """Replays the left arm's planned tool point while the shield is gripped (both grips move together) and
        tests the shield's six boxes, as they are, against every pan, food top and the tray slide."""
        from yam_arena.demo import ArmPlan
        plan = self.task.demo()["left"]
        arm = ArmPlan(plan, (0.0, 0.305, 0.75))
        t_close = sum(w.duration for w in plan[:5])  # the grip is closed at the end of waypoint 4
        t_open = sum(w.duration for w in plan[:12])  # it starts to open after waypoint 11
        fixed = []
        for o in self.task.OBJECTS:
            if o.mass is None:
                fixed.append((o.name, np.array(o.pos) - np.array(o.size) / 2, np.array(o.pos) + np.array(o.size) / 2))
        worst = (math.inf, None, None)
        for t in np.arange(t_close, t_open + 1e-9, 0.05):
            tcp = arm.target(float(t))[0]
            o = np.array([tcp[0], tcp[1] - L.CAP_Y, tcp[2] - self.task.GRIP_UP])
            for c, s, _ in L.SHIELD_BOXES:
                lo, hi = o + np.array(c) - np.array(s) / 2, o + np.array(c) + np.array(s) / 2
                for name, flo, fhi in fixed:
                    g = _aabb_gap(lo, hi, flo, fhi)
                    if g < worst[0]:
                        worst = (g, name, round(float(t), 2))
        self.assertGreater(worst[0], 0.010, worst)
        # set down: the lip's outer face 34 mm short of the tray slide
        end = arm.target(t_open)[0]
        self.assertAlmostEqual(L.SLIDE_X0 - (end[0] + L.LIP_FACE_X), 0.034, places=6)


if __name__ == "__main__":
    unittest.main()
