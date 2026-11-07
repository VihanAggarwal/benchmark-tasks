"""Offline tests for scoring.py (and task.py against the offline yam_arena stand-in when it is there): synthetic gate
states, no simulator.  python3 -m unittest discover -s tests -v"""

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
    foot_corners,
    frame,
    gate_facts,
    in_path,
    obb_hits_aabb,
    path_gap,
    rotation,
    square_err_deg,
    valid,
)

WINDOW = int(round(L.REST_WINDOW_S * L.CONTROL_HZ))  # 10: a gate is at rest after 11 still samples
TX = L.TRACK_X


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
    """(56, 3): every corner of every shape of a gate whose base frame is at o, turned r."""
    pts = []
    for c, s in L.GATE_BOXES:
        h = np.array(s) / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(o + r @ (np.array(c) + h * np.array([sx, sy, sz])))
    return np.array(pts)


def gate(x, y, yaw=0.0, z=L.TOP, roll=0.0, pitch=0.0, vel=(0, 0, 0), origin=True, rot=None, rest=False):
    """A gate state as mechkit reports it: base frame at (x, y, z) (the centre between the feet, on their
    undersides), turned yaw about the vertical after rolling about its crossbar (local x) and pitching about local y
    (degrees); pos/low/high the world box around every shape. rest=True: z is replaced so that the gate's lowest
    point is on the bench (a tilted gate resting on it, not sunk into it)."""
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


SET_L = gate(TX, 0.100, 180.0)  # as the demo leaves them: centred, square, 200 mm apart
SET_R = gate(TX, -0.100, 0.0)
START_L = gate(*L.START["gate_left"], 90.0)
START_R = gate(*L.START["gate_right"], 90.0)
BESIDE = gate(0.20, -0.10, 0.0)  # standing on the bench beside the track, square, not over it
ON_TRACK = gate(TX, -0.10, 90.0, z=L.TOP + L.TRACK_H)  # lowered without turning: standing on the track's top
ACROSS_SIDE = gate(TX, -0.10, 0.0, z=L.TOP + 0.010, roll=90.0)  # fallen on its side across the track


def pair(left, right):
    return {"gate_left": left, "gate_right": right}


def run(states, tr=None):
    """Feed the tracker a list of {gate: state} steps; return the tracker and the last result."""
    tr = tr or Tracker()
    res = None
    for s in states:
        res = tr.update(s)
    return tr, res


def settled(left, right, n=WINDOW + 1, tr=None):
    return run([pair(left, right)] * n, tr)


class Geometry(unittest.TestCase):
    def test_obb_hits_aabb(self):
        lo, hi = np.zeros(3), np.ones(3)
        eye = np.eye(3)
        self.assertTrue(obb_hits_aabb((0.5, 0.5, 0.5), eye, (0.1, 0.1, 0.1), lo, hi))
        self.assertTrue(obb_hits_aabb((1.05, 0.5, 0.5), eye, (0.1, 0.1, 0.1), lo, hi))
        self.assertFalse(obb_hits_aabb((1.2, 0.5, 0.5), eye, (0.1, 0.1, 0.1), lo, hi))
        self.assertTrue(obb_hits_aabb((1.1, 0.5, 0.5), eye, (0.1, 0.1, 0.1), lo, hi))  # touching counts
        # a box turned 45 deg near a corner: its AABB overlaps the corner, the box itself does not
        r = rz(45)
        self.assertFalse(obb_hits_aabb((1.13, 1.13, 0.5), r, (0.1, 0.1, 0.1), lo, hi))
        self.assertTrue(obb_hits_aabb((1.05, 1.05, 0.5), r, (0.1, 0.1, 0.1), lo, hi))
        # edge-against-edge in 3-D: two boxes turned about different axes
        r2 = rx(45) @ ry(45)
        self.assertFalse(obb_hits_aabb((1.2, 1.2, 1.2), r2, (0.1, 0.1, 0.1), lo, hi))
        self.assertTrue(obb_hits_aabb((1.0, 1.0, 1.0), r2, (0.1, 0.1, 0.1), lo, hi))

    def test_feet_at_rest_on_the_bench(self):
        r, o = frame(SET_L)
        np.testing.assert_allclose(foot_corners(r, o)[:, 2], L.TOP, atol=1e-12)
        self.assertEqual(len(foot_corners(r, o)), 8)

    def test_frame_without_origin_matches(self):
        for st in (SET_L, START_R, gate(TX, 0.0, 33.0)):
            bare = gate(*st.origin, rot=rotation(st.quat_xyzw), origin=False)
            np.testing.assert_allclose(frame(bare)[1], st.origin, atol=1e-9)

    def test_square_folding(self):
        for h, e in ((0, 0), (180, 0), (90, 90), (175, 5), (-170, 10), (95, 85), (-5, 5), (360, 0)):
            self.assertAlmostEqual(square_err_deg(rz(h)), e, places=9)

    def test_set_gate_facts(self):
        f = gate_facts(SET_L)
        self.assertTrue(f["standing"] and f["straddling"] and f["in_section"])
        self.assertFalse(f["in_path"])
        self.assertAlmostEqual(f["square_err_deg"], 0.0, places=9)
        self.assertAlmostEqual(f["path_gap_mm"], (L.INNER_HALF - L.PATH_HALF_W) * 1000, places=6)  # 12 mm

    def test_crossbar_passes_over_the_path(self):
        # the crossbar's underside is 74 mm up, 17 mm above the path's 57 mm top
        self.assertGreater(L.CROSSBAR_Z0 - L.PATH_TOP, 0.015)
        self.assertGreater(L.PATH_TOP, L.CARD_TOP)
        self.assertGreater(L.PATH_HALF_W, L.GLIDER_HALF_W)
        self.assertFalse(in_path(*frame(SET_R)))

    def test_off_centre_limit(self):
        # square and on the bench, a gate leaves the path clear up to 12 mm off the centre line
        for off, blocked in ((0.0119, False), (-0.0119, False), (0.0121, True), (-0.0121, True), (0.03, True)):
            st = gate(TX + off, -0.10, 0.0)
            self.assertEqual(in_path(*frame(st)), blocked, off)
            self.assertEqual(path_gap(*frame(st)) < 0, blocked, off)

    def test_turned_gates_and_the_path(self):
        # centred, the inner foot corner is 34 cos a - 20 sin a from the centre line: clear up to about 26 deg
        self.assertFalse(in_path(*frame(gate(TX, -0.10, 10.0))))
        self.assertFalse(in_path(*frame(gate(TX, -0.10, 25.0))))
        self.assertTrue(in_path(*frame(gate(TX, -0.10, 30.0))))
        self.assertTrue(in_path(*frame(gate(TX, -0.10, 90.0))))

    def test_on_track_and_fallen_are_in_the_path(self):
        self.assertTrue(in_path(*frame(ON_TRACK)))
        self.assertTrue(in_path(*frame(ACROSS_SIDE)))
        self.assertFalse(gate_facts(ON_TRACK)["standing"])

    def test_beyond_the_buffers_is_not_the_path(self):
        # turned along the track, the gate spans 46 mm either side of its centre: wholly past the track's end
        self.assertFalse(in_path(*frame(gate(TX, L.PATH_Y1 + 0.047, 90.0, z=L.TOP + L.BUFFER_SIZE[2]))))
        self.assertTrue(in_path(*frame(gate(TX, L.PATH_Y1 + 0.045, 90.0, z=L.TOP + L.BUFFER_SIZE[2]))))

    def test_start_layout(self):
        for name, st in (("gate_left", START_L), ("gate_right", START_R)):
            f = gate_facts(st)
            self.assertTrue(f["standing"])
            self.assertFalse(f["straddling"])
            self.assertFalse(f["in_path"])
            self.assertTrue(Tracker.at_start(name, st))
            # the starting places are outside the screenshot's arm shadow (|y| > 0.45 m with x < 0.20 m)
            self.assertLess(abs(st.high[1]), 0.45)

    def test_section_bounds(self):
        # in front of the parked glider: a gate at the section's end clears its front by 9 mm (crossbar 20 mm deep)
        self.assertAlmostEqual(L.GLIDER_FRONT_Y - (L.SECTION_Y1 + L.CROSSBAR_SIZE[1] / 2), 0.009, places=9)
        self.assertGreater(L.SECTION_Y0 - L.CROSSBAR_SIZE[1] / 2, L.TRACK_Y0)
        # far end: with the glider stopped against the far buffer, its card's back edge is 10 mm past the windows
        self.assertAlmostEqual(L.CARD_BACK_AT_STOP, -0.165, places=9)
        self.assertAlmostEqual((L.SECTION_Y0 - L.WINDOW_SIZE[1] / 2) - L.CARD_BACK_AT_STOP, 0.010, places=9)
        self.assertTrue(gate_facts(gate(TX, L.SECTION_Y1, 0.0))["in_section"])
        self.assertFalse(gate_facts(gate(TX, L.SECTION_Y1 + 0.001, 0.0))["in_section"])
        self.assertTrue(gate_facts(gate(TX, L.SECTION_Y0, 0.0))["in_section"])
        self.assertFalse(gate_facts(gate(TX, L.SECTION_Y0 - 0.001, 0.0))["in_section"])

    def test_standing_tolerance(self):
        self.assertTrue(gate_facts(gate(TX, 0.1, 0.0, z=L.TOP + 0.0029))["standing"])
        self.assertFalse(gate_facts(gate(TX, 0.1, 0.0, z=L.TOP + 0.0031))["standing"])
        self.assertTrue(gate_facts(gate(TX, 0.1, 0.0, z=L.TOP - 0.0029))["standing"])
        # rocked on one edge of its feet (about the crossbar): the other edge 40 sin 6 = 4.2 mm up
        self.assertFalse(gate_facts(gate(TX, 0.1, 0.0, roll=6.0, rest=True))["standing"])
        self.assertTrue(gate_facts(gate(TX, 0.1, 0.0, roll=4.0, rest=True))["standing"])  # 2.8 mm
        # tilted across the track (about the track's direction): one foot's outer edge 92 sin 2.3 = 3.7 mm up
        self.assertFalse(gate_facts(gate(TX, 0.1, 0.0, pitch=2.3, rest=True))["standing"])
        self.assertTrue(gate_facts(gate(TX, 0.1, 0.0, pitch=1.8, rest=True))["standing"])  # 2.9 mm


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        _, res = settled(START_L, START_R)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_both_set_200_apart_scores_one(self):
        _, res = settled(SET_L, SET_R)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertEqual(res.details["n_set"], 2)
        self.assertAlmostEqual(res.details["spacing_mm"], 200.0)

    def test_full_spacing_band(self):
        for d in (0.190, 0.195, 0.205, 0.210):
            _, res = settled(gate(TX, 0.10, 0.0), gate(TX, 0.10 - d, 180.0))
            self.assertEqual(res.score, 1.0, d)

    def test_partial_spacing_band(self):
        for d in (0.180, 0.1899, 0.2101, 0.220):
            _, res = settled(gate(TX, 0.12, 0.0), gate(TX, 0.12 - d, 0.0))
            self.assertEqual(res.score, 0.75, d)
            self.assertFalse(res.success)

    def test_two_set_at_the_wrong_spacing_score_half(self):
        for d in (0.05, 0.1799, 0.2201, 0.26):
            _, res = settled(gate(TX, 0.12, 0.0), gate(TX, 0.12 - d, 0.0))
            self.assertEqual(res.score, 0.5, d)

    def test_tolerances_within_set(self):
        # 10 deg off square and 7 mm off the centre line; square and 11 mm off; at the ends of the section: set
        _, res = settled(gate(TX + 0.007, 0.10, 190.0), gate(TX - 0.007, -0.10, -10.0))
        self.assertEqual(res.score, 1.0)
        _, res = settled(gate(TX + 0.011, 0.10, 180.0), gate(TX - 0.011, -0.10, 0.0))
        self.assertEqual(res.score, 1.0)
        # 10 deg off square, the inner foot corner is 34 cos 10 - 20 sin 10 = 30.01 mm from the gate's centre: 8.0 mm
        # off leaves it 0.01 mm outside the path's 22 mm, 8.1 mm off puts it inside (0)
        _, res = settled(gate(TX + 0.0080, 0.10, 190.0), SET_R)
        self.assertEqual(res.score, 1.0)
        _, res = settled(gate(TX + 0.0081, 0.10, 190.0), SET_R)
        self.assertEqual(res.score, 0.0)
        _, res = settled(gate(TX, L.SECTION_Y1, 0.0), gate(TX, L.SECTION_Y1 - 0.2, 0.0))
        self.assertEqual(res.score, 1.0)
        _, res = settled(gate(TX, L.SECTION_Y0 + 0.2, 180.0), gate(TX, L.SECTION_Y0, 0.0))
        self.assertEqual(res.score, 1.0)

    def test_square_limit(self):
        # within 10 degrees of square: set; 12 degrees (still clear of the path when centred): not set
        for turn, n in ((9.9, 2), (-9.9, 2), (12.0, 1), (-12.0, 1), (168.0, 1), (190.0, 2)):
            _, res = settled(SET_L, gate(TX, -0.10, turn))
            self.assertFalse(res.details["blocked"], turn)
            self.assertEqual(res.details["n_set"], n, turn)
            self.assertEqual(res.score, 1.0 if n == 2 else 0.25, turn)

    def test_one_set(self):
        cases = {
            "other at its start": START_R,
            "other beside the track": BESIDE,
            "other turned 15 deg": gate(TX, -0.10, 15.0),
            "other past the section": gate(TX, L.SECTION_Y0 - 0.006, 0.0),
            "other rocked on its feet": gate(TX, -0.10, 0.0, roll=6.0, rest=True),
            "other held 4 mm up": gate(TX, -0.10, 0.0, z=L.TOP + 0.004),
        }
        for why, other in cases.items():
            _, res = settled(SET_L, other)
            self.assertEqual(res.score, 0.25, why)
            self.assertEqual(res.details["n_set"], 1, why)

    def test_right_alone(self):
        _, res = settled(START_L, SET_R)
        self.assertEqual(res.score, 0.25)

    def test_none_set(self):
        _, res = settled(BESIDE, gate(0.20, 0.10, 0.0))
        self.assertEqual(res.score, 0.0)

    def test_every_tier_is_reachable(self):
        seen = set()
        for a, b in ((START_L, START_R), (SET_L, START_R), (SET_L, gate(TX, -0.05, 0.0)),
                     (SET_L, gate(TX, -0.085, 0.0)), (SET_L, SET_R)):
            seen.add(settled(a, b)[1].score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})


class Zeros(unittest.TestCase):
    def test_gate_on_the_track_zeroes_everything(self):
        _, res = settled(SET_L, ON_TRACK)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(res.details["blocked"])
        self.assertTrue(res.details["gates"]["gate_right"]["in_path"])
        self.assertTrue(res.details["gates"]["gate_left"]["set"])

    def test_gate_fallen_across_the_track(self):
        _, res = settled(ACROSS_SIDE, SET_R)
        self.assertEqual(res.score, 0.0)

    def test_gate_too_far_off_centre(self):
        # straddling, both feet on the bench, but a foot 13 mm off: inside the glider's path
        st = gate(TX + 0.013, -0.10, 0.0)
        f = gate_facts(st)
        self.assertTrue(f["standing"] and f["straddling"])
        _, res = settled(SET_L, st)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["gates"]["gate_right"]["set"])  # a gate in the path is never 'set'
        self.assertEqual(res.details["n_set"], 1)

    def test_gate_turned_into_the_path(self):
        _, res = settled(SET_L, gate(TX, -0.10, 35.0))
        self.assertEqual(res.score, 0.0)

    def test_gate_in_the_path_while_moving_still_zeroes(self):
        tr, _ = settled(SET_L, SET_R)
        res = tr.update(pair(SET_L, gate(TX, -0.10, 90.0, z=L.TOP + 0.03)))  # lowered onto the track, any speed
        self.assertEqual(res.score, 0.0)


class Rest(unittest.TestCase):
    def test_success_needs_half_a_second_at_rest(self):
        tr, res = settled(SET_L, SET_R, n=WINDOW)
        self.assertEqual(res.score, 0.0)  # 10 samples span 0.45 s: neither gate is at rest yet, so neither is set
        self.assertEqual(res.details["n_set"], 0)
        res = tr.update(pair(SET_L, SET_R))
        self.assertEqual(res.score, 1.0)

    def test_score_without_rest_history(self):
        _, res = settled(SET_L, SET_R, n=1)
        self.assertEqual(res.score, 0.0)

    def test_a_moving_gate_does_not_count(self):
        states = [pair(SET_L, gate(TX, -0.10 + 0.0004 * i, 0.0)) for i in range(30)]  # creeping 8 mm/s
        _, res = run(states)
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["gates"]["gate_right"]["at_rest"])

    def test_jitter_counts_as_rest(self):
        states = [pair(gate(TX, 0.10 + (0.0006 if i % 2 else -0.0006), 180.0), SET_R) for i in range(30)]
        _, res = run(states)
        self.assertEqual(res.score, 1.0)

    def test_a_3mm_shift_in_the_window_breaks_rest(self):
        states = [pair(SET_L, SET_R)] * 15 + [pair(SET_L, gate(TX, -0.097, 0.0))] * 5
        tr, res = run(states)
        self.assertFalse(res.details["gates"]["gate_right"]["at_rest"])
        self.assertEqual(res.score, 0.25)
        _, res = run([pair(SET_L, gate(TX, -0.097, 0.0))] * 6, tr)
        self.assertEqual(res.score, 1.0)  # 197 mm apart once it has settled again

    def test_a_gate_turning_on_the_spot_is_not_at_rest(self):
        states = [pair(SET_L, gate(TX, -0.10, 0.6 * i)) for i in range(20)]  # 12 deg/s about its centre
        _, res = run(states)
        self.assertFalse(res.details["gates"]["gate_right"]["at_rest"])

    def test_score_is_idempotent(self):
        tr, res = settled(SET_L, SET_R)
        n = tr.steps
        self.assertEqual(tr.result(pair(SET_L, SET_R)).score, 1.0)
        self.assertEqual(tr.result(pair(SET_L, SET_R)).score, 1.0)
        self.assertEqual(tr.steps, n)


class Exploits(unittest.TestCase):
    def test_lowered_without_turning(self):
        # both gates set down along the track, not turned square: they stand on it and stop the run
        _, res = settled(gate(TX, 0.10, 90.0, z=L.TOP + L.TRACK_H), gate(TX, -0.10, 90.0, z=L.TOP + L.TRACK_H))
        self.assertEqual(res.score, 0.0)

    def test_beside_the_track_at_the_right_spacing(self):
        # square, on the bench, 200 mm apart, but beside the track instead of over it
        for x in (0.20, 0.36):
            _, res = settled(gate(x, 0.10, 0.0), gate(x, -0.10, 0.0))
            self.assertEqual(res.score, 0.0, x)
            self.assertFalse(res.details["blocked"], x)

    def test_one_leg_each_side_of_a_buffer_is_not_the_track_section(self):
        # straddling the far end beyond the section (over the buffer's line): not set
        _, res = settled(SET_L, gate(TX, -0.225, 0.0))
        self.assertEqual(res.score, 0.25)

    def test_where_the_glider_stops_is_not_the_section(self):
        # square over the track between the section's end and the far buffer: the glider would come to rest with its
        # card in this gate's beam (card -0.205 .. -0.165 against the buffer), so the gate is not set
        for y in (-0.155, -0.18, -0.20):
            _, res = settled(SET_L, gate(TX, y, 0.0))
            self.assertEqual(res.score, 0.25, y)
            self.assertFalse(res.details["gates"]["gate_right"]["in_section"], y)
            self.assertFalse(res.details["blocked"], y)

    def test_straddling_the_parked_glider(self):
        _, res = settled(gate(TX, L.GLIDER_Y, 0.0), SET_R)
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["gates"]["gate_left"]["in_path"])
        self.assertFalse(res.details["gates"]["gate_left"]["in_section"])

    def test_held_just_above_the_bench(self):
        _, res = settled(gate(TX, 0.10, 0.0, z=L.TOP + 0.005), gate(TX, -0.10, 0.0, z=L.TOP + 0.005))
        self.assertEqual(res.score, 0.0)

    def test_upside_down_beside_the_track(self):
        st = gate(0.20, -0.10, 0.0, z=L.TOP + L.GATE_H, roll=180.0)  # crossbar on the bench, feet up
        self.assertAlmostEqual(st.low[2], L.TOP, places=9)
        _, res = settled(SET_L, st)
        self.assertEqual(res.score, 0.25)

    def test_leaning_on_the_track(self):
        # one foot on the bench, the other on the track's top (tilted 18 deg across it): not standing, in the path
        st = gate(TX - 0.030, -0.10, 0.0, pitch=-18.0, rest=True)
        feet = gate_facts(st)["feet"]
        self.assertLess(abs(feet[1, 0] - TX), L.TRACK_HALF_W)  # the raised foot is over the track
        self.assertAlmostEqual(feet[1, 2] - L.TOP, L.TRACK_H, delta=0.002)
        self.assertFalse(gate_facts(st)["standing"])
        self.assertTrue(gate_facts(st)["in_path"])
        _, res = settled(SET_L, st)
        self.assertEqual(res.score, 0.0)

    def test_held_over_the_track_is_not_set_and_does_not_block(self):
        # held square astride the track with its feet 40 mm up: legs outside the path, crossbar above it; not set
        st = gate(TX, -0.10, 0.0, z=L.TOP + 0.040)
        self.assertFalse(gate_facts(st)["in_path"])
        _, res = settled(SET_L, st)
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["blocked"])

    def test_upside_down_across_the_track(self):
        # turned over and laid across the track, crossbar down on the track's top: in the path, 0
        st = gate(TX, -0.10, 0.0, z=L.TOP + L.TRACK_H + L.GATE_H, roll=180.0)
        self.assertAlmostEqual(st.low[2], L.TOP + L.TRACK_H, places=9)
        self.assertTrue(gate_facts(st)["in_path"])
        _, res = settled(SET_L, st)
        self.assertEqual(res.score, 0.0)

    def test_slid_on_from_the_far_end_counts(self):
        # set over the far buffer's line and slid along the bench to its place: the end state is all that is scored
        states = [pair(SET_L, gate(TX, -0.24 + 0.005 * i, 0.0)) for i in range(29)]  # to y -0.10
        _, res = run(states + [pair(SET_L, gate(TX, -0.10, 0.0))] * (WINDOW + 1))
        self.assertEqual(res.score, 1.0)
        # the opening between the feet (68 mm) and under the crossbar (74 mm) passes over the 40 x 35 mm buffer
        self.assertGreater(2 * L.INNER_HALF, L.BUFFER_SIZE[0])
        self.assertGreater(L.CROSSBAR_Z0, L.BUFFER_SIZE[2])

    def test_both_gates_at_one_spot_score_half_at_most(self):
        _, res = settled(gate(TX, 0.0, 0.0), gate(TX, 0.0, 180.0))  # overlapping (impossible): spacing 0
        self.assertEqual(res.score, 0.5)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(SET_L, SET_R)
        hist = {n: list(tr.hist[n]) for n in L.GATES}
        for bad in ("nan_pos", "inf_vel", "zero_quat", "nan_origin", "missing", "none"):
            st = gate(TX, -0.10, 0.0)
            if bad == "nan_pos":
                st.pos = np.array([np.nan, 0, 0])
            elif bad == "inf_vel":
                st.lin_vel = np.array([np.inf, 0, 0])
            elif bad == "zero_quat":
                st.quat_xyzw = np.zeros(4)
            elif bad == "nan_origin":
                st.origin = np.array([np.nan, 0, 0])
            elif bad == "missing":
                del st.low
            else:
                st = None
            self.assertFalse(valid(st), bad)
            res = tr.update(pair(SET_L, st))
            self.assertEqual(res.score, 0.0, bad)
            self.assertFalse(res.success, bad)
            for n in L.GATES:
                self.assertEqual(len(tr.hist[n]), len(hist[n]), bad)
                for a, b in zip(tr.hist[n], hist[n]):
                    np.testing.assert_array_equal(a, b)
        res = tr.update(pair(SET_L, SET_R))
        self.assertEqual(res.score, 1.0)  # the record survived the bad steps

    def test_missing_gate_key(self):
        tr = Tracker()
        res = tr.update({"gate_left": SET_L})
        self.assertEqual(res.score, 0.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker()
        for _ in range(300):
            x, y, z = rng.uniform([0.0, -0.6, 0.6], [0.6, 0.6, 1.0])
            a = gate(x, y, yaw=rng.uniform(-180, 180), z=z, roll=rng.uniform(-180, 180), pitch=rng.uniform(-90, 90))
            x, y, z = rng.uniform([0.0, -0.6, 0.6], [0.6, 0.6, 1.0])
            b = gate(x, y, yaw=rng.uniform(-180, 180), z=z)
            res = tr.update(pair(a, b))
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(res.details)

    def test_origin_or_box_centre_give_the_same_score(self):
        bare_l = gate(TX, 0.10, 180.0, origin=False)
        bare_r = gate(TX, -0.10, 0.0, origin=False)
        _, res = settled(bare_l, bare_r)
        self.assertEqual(res.score, 1.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(SET_L, SET_R)
        n = tr.steps
        res = tr.update(pair(START_L, START_R))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        self.assertEqual(len(tr.hist["gate_left"]), 1)
        self.assertEqual(res.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(SET_L, SET_R)
        sl, sr = L.START["gate_left"], L.START["gate_right"]
        tr.update(pair(gate(sl[0] + 0.02, sl[1] - 0.02, 75.0), gate(sr[0] - 0.02, sr[1] + 0.01, 100.0)))
        self.assertEqual(tr.steps, 1)

    def test_carrying_back_slowly_does_not_reset(self):
        tr, _ = settled(START_L, START_R)
        sl = L.START["gate_left"]
        n = tr.steps
        for i in range(1, 30):  # lifted a little and set back where it was, 1 mm per step
            tr.update(pair(gate(sl[0], sl[1] - 0.001 * min(i, 30 - i), 90.0), START_R))
        self.assertEqual(tr.steps, n + 29)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(START_L, START_R)
        n = tr.steps
        tr.update(pair(SET_L, START_R))
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(SET_L, SET_R)
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

    def sim(self, left, right):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        out = {}
        for name, st, spec in (("gate_left", left, self.task.GATE_LEFT), ("gate_right", right, self.task.GATE_RIGHT)):
            out[name] = self.mk.state_from_poses(spec, (*st.origin, *st.quat_xyzw), source="sim")
        return out

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in self.task.OBJECTS if o.mass is not None), ["gate_left", "gate_right"])
        for g in (self.task.GATE_LEFT, self.task.GATE_RIGHT):
            self.assertAlmostEqual(g.mass, L.GATE_MASS)
            self.assertLessEqual(g.mass, 0.30)
            self.assertEqual(len(g.shapes), len(L.GATE_BOXES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.5)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)

    def test_scorer_geometry_matches_mechkit(self):
        lo, hi = self.task.GATE_LEFT.local_bounds()
        np.testing.assert_allclose(lo, [-0.046, -0.020, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.046, 0.020, L.GATE_H], atol=1e-12)
        np.testing.assert_allclose((lo + hi) / 2, L.GATE_BBOX_CENTRE, atol=1e-12)
        for spec, name in ((self.task.GATE_LEFT, "gate_left"), (self.task.GATE_RIGHT, "gate_right")):
            rest = spec.rest_state()
            self.assertTrue(Tracker.at_start(name, rest))
            f = gate_facts(rest)
            self.assertTrue(f["standing"])
            self.assertFalse(f["straddling"] or f["in_path"])
        st = self.sim(SET_L, SET_R)
        for n, ref in (("gate_left", SET_L), ("gate_right", SET_R)):
            np.testing.assert_allclose(st[n].low, ref.low, atol=1e-9)
            np.testing.assert_allclose(st[n].high, ref.high, atol=1e-9)

    def test_gate_stands_stably(self):
        com = self.task.GATE_LEFT.center_of_mass()
        np.testing.assert_allclose(com[:2], 0.0, atol=1e-12)
        # tipping along the track needs a lean past atan(20 / 50.6) = 21.6 deg; across it, past 42 deg
        self.assertGreater(math.degrees(math.atan2(L.FOOT_SIZE[1] / 2, com[2])), 20.0)
        self.assertGreater(math.degrees(math.atan2(L.LEG_X + L.FOOT_SIZE[0] / 2, com[2])), 40.0)

    def test_fixed_parts_do_not_overlap_the_gates_or_each_other_at_start(self):
        fixed = [o for o in self.task.OBJECTS if o.mass is None]
        for o in fixed:
            lo = np.array(o.pos) - np.array(o.size) / 2
            self.assertGreaterEqual(lo[2], L.TOP - 1e-12, o.name)
        for spec in (self.task.GATE_LEFT, self.task.GATE_RIGHT):
            glo, ghi = spec.world_bounds()
            for o in fixed:
                lo, hi = np.array(o.pos) - np.array(o.size) / 2, np.array(o.pos) + np.array(o.size) / 2
                self.assertTrue(np.any(glo > hi) or np.any(lo > ghi), (spec.name, o.name))

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.GATE_LEFT.offline_box()[0]), quat_xyzw=START_L.quat_xyzw,
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.GATE_LEFT.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [*L.START["gate_left"], L.TOP], atol=1e-12)
        self.assertTrue(Tracker.at_start("gate_left", st))

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START_L, START_R)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(SET_L, SET_R)
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
                self.task.success(self.sim(SET_L, SET_R))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "blocked", "n_set", "spacing_mm", "gates",
                          "gate_left_state", "gate_right_state"):
                    self.assertIn(k, row)
                for k in ("set", "at_rest", "standing", "straddling", "square_err_deg", "in_section", "in_path",
                          "centre_mm", "off_centre_mm", "path_gap_mm", "foot_low_mm", "foot_high_mm"):
                    self.assertIn(k, row["gates"]["gate_right"])
                self.assertAlmostEqual(row["spacing_mm"], 200.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(SET_L, SET_R)))  # no crash
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
                if prev is not None and w.pos[2] < prev.pos[2] and w.pos[2] < L.TOP + 0.09:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))  # final descents are slow
                prev = w
        # the grip: 8 mm narrower than the crossbar's 20 mm depth; open wide enough for it
        self.assertAlmostEqual(-2 * self.task.GATE_GRIP, L.CROSSBAR_SIZE[1] - 0.008)
        self.assertGreater(-2 * self.task.GATE_OPEN, L.CROSSBAR_SIZE[1] + 0.030)
        # the pad tips stay below the crossbar's underside and above the window tops
        tips = self.task.GRIP_UP - 0.013
        self.assertLess(tips, L.CROSSBAR_Z0)
        self.assertGreater(tips, L.WINDOW_Z + L.WINDOW_SIZE[2] / 2)
        # carried over the track and the glider with room to spare
        self.assertGreater(self.task.CARRY_Z - L.TOP - self.task.GRIP_UP, L.CARD_TOP)
        # both place points are in the section, 200 mm apart, the two grippers never over the track together
        ys = sorted(w.pos[1] for side in plan for w in plan[side] if w.label.startswith("slowly lower"))
        self.assertAlmostEqual(ys[1] - ys[0], 0.200)
        left = self.task._arm("left")
        t_left_up = sum(w.duration for w in left[:left.index(next(w for w in left if w.label == "straight up")) + 1])
        right = plan["right"]
        t_right_over = sum(w.duration for w in right[:right.index(
            next(w for w in right if w.label.startswith("carry it in front"))) + 1])
        self.assertGreater(t_right_over, t_left_up)  # the right gate starts across only once the left hand is up
        # the carry: feet well over the track and the buffers; turned on the way to x 0.20, clear of the glider's side
        feet_up = self.task.CARRY_Z - L.TOP - self.task.GRIP_UP
        self.assertGreater(feet_up - L.TRACK_H, 0.030)
        self.assertGreater(feet_up - L.BUFFER_SIZE[2], 0.020)
        reach = math.hypot(L.LEG_X + L.FOOT_SIZE[0] / 2, L.FOOT_SIZE[1] / 2)  # the gate's corners, turning
        self.assertLess(self.task.VIA_X + reach, L.TRACK_X - L.GLIDER_HALF_W - 0.005)  # 9 mm to spare
        for side, wps in plan.items():
            via = [w for w in wps if w.label.startswith("carry it in front")]
            self.assertEqual(len(via), 1)
            self.assertAlmostEqual(via[0].pos[0], self.task.VIA_X)
        # the place yaws turn each gate square: start heading 90 deg, turned by (place - pick)
        for side in ("left", "right"):
            turned = 90.0 + math.degrees(self.task.YAW_PLACE - self.task.YAW_PICK[side])
            self.assertAlmostEqual(square_err_deg(rz(turned)), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
