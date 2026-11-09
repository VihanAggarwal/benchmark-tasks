"""Offline tests for scoring.py and the hooks in task.py: synthetic walking-frame states, no simulator.
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
    WINDOW, Tracker, band, feet_heights, footprint, heading_rad, tilt_deg, toe_gap, turn_deg, valid,
)

TOP = L.TOP
TIERS = {0.0, 0.25, 0.5, 1.0}


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


def frame(cx, cy, heading, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    """A frame state as mechkit reports it: origin (footprint centre) at (cx, cy), turned `heading` degrees and
    tilted by `pitch`/`roll`, resting with its lowest point `lift` above the floor; low/high from its 56 x 70 x 106
    mm box."""
    q = quat(heading, pitch, roll)
    local = np.array([(su * L.FOOT_U, sv * L.FOOT_V, z) for su in (-1, 1) for sv in (-1, 1) for z in (0.0, L.FRAME_H)])
    pts = local @ rot(q).T
    o = np.array([cx, cy, TOP + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


def placed(turn=0.0, off=0.0, d=0.030, **kw):
    """A frame in front of him turned `turn` degrees from square, its centre `off` from his centre line (+ toward
    +y), its nearest footprint corner `d` in front of his toes."""
    h = L.heading_of(turn)
    cx = L.TOE_X - d - max(dx for dx, _ in L.corner_offsets(h))
    return frame(cx, L.Y_C + off, math.degrees(h), **kw)


START = frame(L.FRAME_START[0], L.FRAME_START[1], -90.0)
TARGET = placed()


def settled(s, t=None, n=WINDOW + 1):
    """Feed the same state n times (half a second still); return (result, tracker)."""
    t = Tracker() if t is None else t
    r = None
    for _ in range(n):
        r = t.update(s)
    return r, t


def scored(s) -> float:
    return settled(s)[0].score


def reason(s):
    return settled(s)[0].details["zero_reason"]


class Geometry(unittest.TestCase):
    def test_turn_definition(self):
        for heading, want in ((180.0, 0.0), (165.0, 15.0), (195.0, 15.0), (-150.0, 30.0), (90.0, 90.0),
                              (-90.0, 90.0), (0.0, 180.0), (10.0, 170.0)):
            self.assertAlmostEqual(turn_deg(frame(0.3, 0.0, heading)), want, places=6, msg=heading)
        self.assertAlmostEqual(math.degrees(L.heading_of(0.0)), 180.0)

    def test_corner_offsets_match_the_footprint(self):
        for turn in (0.0, -12.0, 25.0, 90.0, 180.0):
            s = placed(turn)
            off = L.corner_offsets(L.heading_of(turn))
            np.testing.assert_allclose(footprint(s), s.origin[:2] + np.array(off), atol=1e-12)

    def test_toe_gap_feet_and_tilt(self):
        self.assertAlmostEqual(toe_gap(TARGET), 0.030, places=9)
        self.assertAlmostEqual(toe_gap(placed(22.0, 0.010, 0.047)), 0.047, places=9)
        np.testing.assert_allclose(feet_heights(TARGET), 0.0, atol=1e-12)
        np.testing.assert_allclose(feet_heights(frame(0.3, 0.0, 180.0, lift=0.004)), 0.004, atol=1e-12)
        self.assertAlmostEqual(tilt_deg(frame(0.3, 0.0, 180.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(frame(0.3, 0.0, 180.0, roll=180.0)), 180.0, places=6)
        # tipped 3 degrees onto one pair of feet: the other pair's lowest points (their inner edges, 48 mm from the
        # far edge of the feet it stands on) are 48 sin 3 = 2.5 mm up
        f = feet_heights(frame(0.3, 0.0, 0.0, pitch=3.0))
        self.assertAlmostEqual(float(f.max()), (2 * L.FOOT_U - L.TUBE) * math.sin(math.radians(3.0)), places=6)
        self.assertAlmostEqual(float(f.min()), 0.0, places=9)

    def test_start_layout_scores_zero(self):
        self.assertTrue(Tracker.at_start(START))
        r, _ = settled(START)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "not_square_side_on_or_crossbar_toward_him")
        self.assertAlmostEqual(r.details["turn_deg"], 90.0)
        self.assertGreater(abs(START.origin[1]) - L.FOOT_U, L.CENTRE_LOOSE + 0.050)  # far off his centre line too
        fp = footprint(START)
        self.assertTrue(np.all(np.abs(fp[:, 1]) < 0.45))  # clear of the arms' start-pose shadow
        self.assertTrue(np.all(fp[:, 0] > 0.20))

    def test_demo_target(self):
        r, _ = settled(TARGET)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        d = r.details
        self.assertAlmostEqual(d["turn_deg"], 0.0)
        self.assertAlmostEqual(d["off_centre_mm"], 0.0)
        self.assertAlmostEqual(d["toe_gap_mm"], 30.0)
        self.assertEqual((d["square"], d["centred"], d["toe_gap"]), ("full", "full", "full"))
        np.testing.assert_allclose(TARGET.origin[:2], L.TARGET_CENTER, atol=1e-12)


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        cases = ((0.0, 0.0, 0.030, 1.0), (20.0, 0.0, 0.030, 0.5), (0.0, 0.035, 0.030, 0.5), (0.0, 0.0, 0.070, 0.5),
                 (0.0, 0.0, 0.005, 0.5), (20.0, 0.035, 0.030, 0.25), (0.0, 0.035, 0.070, 0.25),
                 (20.0, 0.035, 0.070, 0.0), (90.0, 0.0, 0.030, 0.0), (180.0, 0.0, 0.030, 0.0),
                 (0.0, 0.060, 0.030, 0.0), (0.0, 0.0, 0.120, 0.0), (0.0, 0.0, -0.020, 0.0))
        for turn, off, d, want in cases:
            self.assertEqual(scored(placed(turn, off, d)), want, (turn, off, d))
        for turn, off, d, want in cases:  # mirror images: turned the other way, off to the other side
            self.assertEqual(scored(placed(-turn, -off, d)), want, (-turn, -off, d))

    def test_band_function(self):
        self.assertEqual(band(15.0, (0, 15), (0, 30)), "full")
        self.assertEqual(band(15.01, (0, 15), (0, 30)), "loose")
        self.assertEqual(band(30.01, (0, 15), (0, 30)), "out")
        self.assertEqual(band(float("nan"), (0, 15), (0, 30)), "out")

    def test_square_edges(self):
        for turn, want in ((14.9, 1.0), (15.1, 0.5), (29.9, 0.5), (30.1, 0.0), (-14.9, 1.0), (-15.1, 0.5),
                           (-30.1, 0.0)):
            self.assertEqual(scored(placed(turn)), want, turn)
        self.assertEqual(reason(placed(30.1)), "not_square_side_on_or_crossbar_toward_him")

    def test_centred_edges(self):
        for off, want in ((0.0199, 1.0), (0.0201, 0.5), (0.0499, 0.5), (0.0501, 0.0), (-0.0199, 1.0),
                          (-0.0201, 0.5), (-0.0501, 0.0)):
            self.assertEqual(scored(placed(0.0, off)), want, off)
        self.assertEqual(reason(placed(0.0, 0.0501)), "off_to_the_side")

    def test_toe_gap_edges(self):
        for d, want in ((0.0101, 1.0), (0.0099, 0.5), (0.0499, 1.0), (0.0501, 0.5), (0.0001, 0.5), (0.0999, 0.5)):
            self.assertEqual(scored(placed(0.0, 0.0, d)), want, d)
        self.assertEqual(reason(placed(0.0, 0.0, -0.0001)), "past_his_toes")
        self.assertEqual(reason(placed(0.0, 0.0, 0.1001)), "too_far_from_his_toes")

    def test_only_a_full_score_is_success(self):
        for turn, off, d, want in ((20.0, 0.0, 0.030, 0.5), (0.0, 0.035, 0.070, 0.25), (20.0, 0.035, 0.070, 0.0)):
            r, _ = settled(placed(turn, off, d))
            self.assertEqual(r.score, want)
            self.assertFalse(r.success, (turn, off, d))
        self.assertTrue(settled(TARGET)[0].success)

    def test_every_tier_is_reachable(self):
        seen = {scored(placed(t, o, d)) for t in (0.0, 22.0) for o in (0.0, 0.035) for d in (0.030, 0.075)}
        seen.add(scored(START))
        self.assertEqual(seen, TIERS)

    def test_three_loose_bands_score_zero_but_are_not_a_zero_reason(self):
        r, _ = settled(placed(25.0, 0.040, 0.080))
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["zero_reason"])
        self.assertEqual(r.details["n_loose"], 3)

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(TARGET)
        self.assertEqual(t.result(TARGET).score, 1.0)
        self.assertEqual(t.result(TARGET).score, 1.0)  # result() does not advance the tracker
        self.assertEqual(t.steps, WINDOW + 1)

    def test_random_states_map_to_a_tier(self):
        rng = random.Random(7)
        for _ in range(400):
            s = frame(rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.3), rng.uniform(-180, 180),
                      lift=rng.choice((0.0, 0.0, 0.002, 0.01)), pitch=rng.choice((0.0, 0.0, 2.0, 30.0)))
            self.assertIn(scored(s), TIERS)
        for _ in range(400):  # near the place
            s = placed(rng.uniform(-35, 35), rng.uniform(-0.06, 0.06), rng.uniform(-0.01, 0.11))
            self.assertIn(scored(s), TIERS)


class ZeroConditions(unittest.TestCase):
    def test_side_on_or_crossbar_toward_him(self):
        for turn in (90.0, -90.0, 180.0, 135.0, -45.0):
            s = placed(turn)
            self.assertEqual(scored(s), 0.0, turn)
            self.assertEqual(reason(s), "not_square_side_on_or_crossbar_toward_him", turn)

    def test_off_to_the_side(self):
        for off in (0.051, -0.051, 0.120, -0.200):
            self.assertEqual(reason(placed(0.0, off)), "off_to_the_side", off)

    def test_around_his_feet(self):
        # pushed in so that its back legs pass his toes, either side of his slippers (it is open at the back)
        for d in (-0.001, -0.020, -0.030):
            self.assertEqual(reason(placed(0.0, 0.0, d)), "past_his_toes", d)

    def test_too_far(self):
        self.assertEqual(reason(placed(0.0, 0.0, 0.101)), "too_far_from_his_toes")
        self.assertEqual(reason(placed(0.0, 0.0, 0.200)), "too_far_from_his_toes")

    def test_not_on_its_four_feet(self):
        self.assertEqual(scored(placed(lift=0.0029)), 1.0)
        self.assertEqual(reason(placed(lift=0.0031)), "not_on_its_four_feet")  # held or resting above the floor
        self.assertEqual(reason(placed(lift=0.020)), "not_on_its_four_feet")  # carried 20 mm up
        self.assertEqual(scored(placed(lift=-0.0029)), 1.0)  # contact settling into the floor
        self.assertEqual(reason(placed(lift=-0.0031)), "not_on_its_four_feet")  # sunk: a broken state
        self.assertEqual(reason(placed(roll=90.0)), "not_on_its_four_feet")  # on its side: two feet down
        self.assertEqual(reason(placed(pitch=90.0)), "not_on_its_four_feet")  # on its front or back
        self.assertEqual(reason(placed(pitch=-90.0)), "not_on_its_four_feet")
        self.assertEqual(reason(placed(roll=180.0)), "not_on_its_four_feet")  # upside down

    def test_one_pair_of_feet_lifted(self):
        # standing on one pair of feet, the other pair a little up: each foot's lowest point (the lifted feet's
        # inner edge, 48 mm along its depth or 62 mm across from the far edge it pivots on) within 3 mm counts
        du, dv = 2 * L.FOOT_U - L.TUBE, 2 * L.FOOT_V - L.TUBE
        for pitch_sign in (1, -1):  # on its back feet, or on its front feet
            a = math.degrees(math.asin(0.0029 / du))
            b = math.degrees(math.asin(0.0031 / du))
            self.assertEqual(scored(placed(pitch=pitch_sign * a)), 1.0)
            self.assertEqual(reason(placed(pitch=pitch_sign * b)), "not_on_its_four_feet")
        for roll_sign in (1, -1):  # on one side's two feet
            self.assertEqual(scored(placed(roll=roll_sign * math.degrees(math.asin(0.0029 / dv)))), 1.0)
            self.assertEqual(reason(placed(roll=roll_sign * math.degrees(math.asin(0.0031 / dv)))),
                             "not_on_its_four_feet")

    def test_moving(self):
        t = Tracker()
        for i in range(30):  # sliding toward him at 20 mm/s: 1 mm a step, 10 mm over 0.5 s
            r = t.update(placed(0.0, 0.0, 0.050 - 0.001 * i))
        self.assertEqual(r.details["zero_reason"], "moving")
        self.assertEqual(r.score, 0.0)


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for _ in range(WINDOW):
            r = t.update(TARGET)
            self.assertEqual(r.score, 0.0)
        self.assertEqual(t.update(TARGET).score, 1.0)

    def test_slow_creep_under_5_mm_is_still(self):
        t = Tracker()
        for i in range(40):  # 0.4 mm a step: 4 mm over 0.5 s
            r = t.update(placed(0.0, 0.0, 0.040 - 0.0004 * i))
        self.assertTrue(r.details["still"])
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(30):  # 0.6 mm back and forth every step: 6 mm of path over 0.5 s, though it goes nowhere
            r = t.update(placed(0.0, 0.0006 if i % 2 else 0.0))
        self.assertFalse(r.details["still"])

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        for i in range(30):  # 1.5 degrees a step about its centre: 1.2 mm a step at its corners
            r = t.update(frame(0.29, 0.0, 180.0 + 1.5 * (i if i < 15 else 30 - i)))
        self.assertFalse(r.details["still"])


class Exploits(unittest.TestCase):
    def test_held_still_just_above_the_floor_counts_as_set_down(self):
        # The scorer sees only the frame (task.json fidelity): held still 2 mm up at the place it counts as set down.
        self.assertEqual(scored(placed(lift=0.002)), 1.0)

    def test_the_classic_error_crossbar_toward_him(self):
        # Same place, same distance, but turned round: he would have to step round it or pull on the crossbar.
        self.assertEqual(scored(placed(180.0, 0.0, 0.030)), 0.0)
        self.assertEqual(scored(placed(170.0, 0.0, 0.010)), 0.0)

    def test_left_beside_his_chair(self):
        # Side-on along the chair's side or square but beside him: 0.
        self.assertEqual(scored(frame(0.420, -0.090, 0.0)), 0.0)
        self.assertEqual(scored(frame(0.420, 0.090, 180.0)), 0.0)
        self.assertEqual(scored(frame(0.300, -0.090, 180.0)), 0.0)

    def test_wrapped_around_his_feet(self):
        self.assertEqual(scored(placed(0.0, 0.0, -0.033)), 0.0)  # back legs level with his shins

    def test_far_out_in_the_room(self):
        self.assertEqual(scored(placed(0.0, 0.0, 0.150)), 0.0)

    def test_on_its_side_or_upside_down_at_the_place(self):
        self.assertEqual(scored(placed(roll=90.0)), 0.0)
        self.assertEqual(scored(placed(roll=180.0)), 0.0)

    def test_two_legs_on_his_slippers(self):
        # Pushed back so its back feet stand on his slippers (9 mm tall): tipped and off the floor.
        tip = math.degrees(math.atan2(0.009, 2 * L.FOOT_U))
        s = placed(0.0, 0.0, -0.010, pitch=tip)
        self.assertEqual(scored(s), 0.0)

    def test_a_frame_standing_level_where_it_scores_is_clear_of_him_and_his_chair(self):
        fixed = [(xs, ys, zs) for _, xs, ys, zs, _ in L.RESIDENT_BOXES] + [(xs, ys, zs) for _, xs, ys, zs in
                                                                              L.CHAIR_BOXES]
        rng = random.Random(3)
        n = 0
        for _ in range(3000):  # bounded: a broken scorer that zeroes everything must fail, not hang
            if n >= 300:
                break
            s = placed(rng.uniform(-30, 30), rng.uniform(-0.05, 0.05), rng.uniform(0.0, 0.1))
            if scored(s) == 0.0 and settled(s)[0].details["zero_reason"] is not None:
                continue
            n += 1
            r = rot(s.quat_xyzw)
            for _, c, size, _ in L.FRAME_SHAPES:
                w = s.origin + r @ np.array(c)
                half = np.abs(r) @ (np.array(size) / 2)
                lo, hi = w - half, w + half
                for xs, ys, zs in fixed:
                    ov = [min(hi[0], xs[1]) - max(lo[0], xs[0]), min(hi[1], ys[1]) - max(lo[1], ys[0]),
                          min(hi[2] - TOP, zs[1]) - max(lo[2] - TOP, zs[0])]
                    self.assertFalse(all(v > 1e-9 for v in ov), (s.origin, xs, ys, zs))
        self.assertEqual(n, 300)  # nearly every placement in these ranges scores above 0


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = settled(TARGET)
        self.assertEqual(r.score, 1.0)
        n = len(t.history)
        bad = []
        for k in ("origin", "quat_xyzw", "pos", "lin_vel", "low"):
            s = placed()
            setattr(s, k, np.full_like(np.asarray(getattr(s, k), dtype=float), np.nan))
            bad.append(s)
        z = placed()
        z.quat_xyzw = np.zeros(4)
        bad.append(z)
        inf = placed()
        inf.origin = np.array([np.inf, 0.0, TOP])
        bad.append(inf)
        for s in bad:
            self.assertFalse(valid(s))
            res = t.update(s)
            self.assertEqual(res.score, 0.0)
            self.assertEqual(res.details["zero_reason"], "invalid_state")
            self.assertFalse(res.success)
            self.assertEqual(len(t.history), n)
        self.assertEqual(t.update(TARGET).score, 1.0)  # the record was left as it was

    def test_missing_fields(self):
        s = SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                            low=np.zeros(3), high=np.zeros(3))  # no origin
        self.assertFalse(valid(s))
        self.assertEqual(Tracker().update(s).score, 0.0)
        self.assertEqual(Tracker().update(None).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = placed()
        s.quat_xyzw = s.quat_xyzw * 3.0
        self.assertEqual(scored(s), 1.0)

    def test_forward_pointing_straight_up(self):
        s = placed(pitch=-90.0)
        self.assertTrue(math.isnan(heading_rad(s)) or abs(tilt_deg(s) - 90.0) < 1e-6)
        self.assertEqual(scored(s), 0.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, t = settled(TARGET)
        r = t.update(START)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history), 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        _, t = settled(TARGET)
        t.update(frame(L.FRAME_START[0] + 0.03, L.FRAME_START[1] - 0.03, -90.0))
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = np.array(L.FRAME_START), np.array([0.25, -0.12])
        for i in range(20):
            p = b + (a - b) * i / 19
            t.update(frame(p[0], p[1], -90.0))
        self.assertEqual(t.steps, 20)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = settled(START)
        t.update(TARGET)
        self.assertEqual(t.steps, WINDOW + 2)

    def test_explicit_reset(self):
        _, t = settled(TARGET)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(TARGET).score, 0.0)  # half a second of history again


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _seg_point(a, b, p) -> float:
    a, b, p = (np.asarray(v, dtype=float) for v in (a, b, p))
    ab = b - a
    k = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(a + k * ab - p))


def _poly_rect_distance(poly, xs, ys) -> float:
    """Distance seen from above between a convex quadrilateral (corners in order) and an axis-aligned rectangle
    ((x0, x1), (y0, y1)); 0 if they overlap."""
    rect = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]

    def normals(q):
        return [np.array([q[i][1] - q[(i + 1) % 4][1], q[(i + 1) % 4][0] - q[i][0]]) for i in range(4)]

    if not any(max(np.dot(n, v) for v in poly) < min(np.dot(n, v) for v in rect)
               or max(np.dot(n, v) for v in rect) < min(np.dot(n, v) for v in poly)
               for n in normals(poly) + normals(rect)):
        return 0.0
    return min(min(_seg_point(poly[i], poly[(i + 1) % 4], q) for i in range(4) for q in rect),
               min(_seg_point(rect[i], rect[(i + 1) % 4], q) for i in range(4) for q in poly))


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

    def sim(self, s):
        """A synthetic state turned into what mechkit would report from the simulator."""
        return {"walking_frame": self.mk.state_from_poses(self.task.WALKING_FRAME, (*s.origin, *s.quat_xyzw),
                                                          source="sim")}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], ["walking_frame"])
        self.assertLessEqual(t.WALKING_FRAME.mass, 0.30)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None), 0.0)
            if size is None:
                size = (2 * o.radius,) * 3
            half = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            self.assertLess(abs(c[1]) + half, 0.45, o.name)
        # the fixed parts do not interpenetrate, and every one stands on the floor or on another
        boxes = [(o.name, np.array(o.pos), np.array(o.size)) for o in t.OBJECTS if o.mass is None and hasattr(o, "size")]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                self.assertLess(_boxes_overlap(boxes[i][1:], boxes[j][1:]), 1e-12, (boxes[i][0], boxes[j][0]))
        # his toes are on the toe line, and his feet sit inside the frame's open back (they fit between its legs)
        toes = min(xs[0] for n, xs, _, _, _ in L.RESIDENT_BOXES if n.endswith("slipper"))
        self.assertAlmostEqual(toes, L.TOE_X)
        feet_y = max(max(abs(ys[0]), abs(ys[1])) for n, _, ys, _, _ in L.RESIDENT_BOXES if n.endswith("slipper"))
        self.assertLess(feet_y, L.FOOT_V - L.TUBE)

    def test_frame_geometry_matches_the_scorer(self):
        spec = self.task.WALKING_FRAME
        lo, hi = spec.local_bounds()
        np.testing.assert_allclose(lo, [-L.FOOT_U, -L.FOOT_V, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.FOOT_U, L.FOOT_V, L.FRAME_H], atol=1e-12)
        shapes = [(c, s) for _, c, s, _ in L.FRAME_SHAPES]
        self.assertEqual(len(shapes), 17)
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                self.assertLess(_boxes_overlap(shapes[i], shapes[j]), 1e-12, (L.FRAME_SHAPES[i][0],
                                                                              L.FRAME_SHAPES[j][0]))
        com = spec.center_of_mass()
        self.assertLess(abs(com[0] - L.GRIP_U), 0.001)  # the grip is within 1 mm of the centre of mass
        self.assertAlmostEqual(com[1], 0.0, places=12)
        self.assertLess(com[2], L.FRAME_H - 0.030)  # it hangs below the grip
        rest = spec.rest_state()
        np.testing.assert_allclose(rest.origin, [L.FRAME_START[0], L.FRAME_START[1], TOP], atol=1e-12)
        self.assertTrue(Tracker.at_start(rest))
        sim = self.sim(TARGET)["walking_frame"]
        self.assertAlmostEqual(float(sim.low[2]), TOP, places=9)
        self.assertEqual(scored(sim), 1.0)
        # the feet: four ferrules at the outer corners, their bottoms on the floor
        ferrules = [s for s in L.FRAME_SHAPES if s[0].startswith("ferrule")]
        self.assertEqual(len(ferrules), 4)
        for _, c, s, _ in ferrules:
            self.assertAlmostEqual(c[2] - s[2] / 2, 0.0)
            self.assertAlmostEqual(abs(c[0]) + s[0] / 2, L.FOOT_U)
            self.assertAlmostEqual(abs(c[1]) + s[1] / 2, L.FOOT_V)
        # the crossbar is at the front (+u) and nothing closes the back: the open side
        cross = [s for s in L.FRAME_SHAPES if s[0] == "crossbar"][0]
        self.assertGreater(cross[1][0], 0.0)
        for name, c, s, _ in L.FRAME_SHAPES:
            if abs(c[1]) + s[1] / 2 < L.FOOT_V - L.TUBE - 1e-9 + s[1]:  # spans between the two sides
                self.assertGreater(c[0] - s[0] / 2, -1e-9, name)
        # the grip: the pads (20 mm wide) wholly on the black handgrips, which are 70 mm apart outside
        grips = [s for s in L.FRAME_SHAPES if s[0].startswith("handgrip")]
        self.assertEqual(len(grips), 2)
        for _, c, s, _ in grips:
            self.assertAlmostEqual(abs(c[1]) + s[1] / 2, L.FOOT_V)
            self.assertLessEqual(c[0] - s[0] / 2, L.GRIP_U - 0.010)
            self.assertGreaterEqual(c[0] + s[0] / 2, L.GRIP_U + 0.010)
            self.assertAlmostEqual(c[2] + s[2] / 2, L.FRAME_H)

    def test_offline_state_path(self):
        spec = self.task.WALKING_FRAME
        c, _, yaw = spec.offline_box()
        box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.FRAME_START[0], L.FRAME_START[1], TOP], atol=1e-12)
        self.assertEqual(scored(st), 0.0)

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(TARGET)
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
                self.task.success(self.sim(TARGET))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "zero_reason", "set_down", "on_feet", "still",
                          "square", "centred", "toe_gap", "n_loose", "turn_deg", "off_centre_mm", "toe_gap_mm",
                          "centre_mm", "feet_mm", "tilt_deg", "path_mm", "frame_origin", "frame_quat", "source"):
                    self.assertIn(k, row)
                self.assertEqual(row["toe_gap_mm"], 30.0)
                self.assertEqual(row["zero_reason"], "moving")  # one step seen: not yet still
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(TARGET)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"right"})  # one arm; the left holds its ready pose
        wps = plan["right"]
        for prev, w in zip(wps, wps[1:]):
            if prev.finger != w.finger:  # every close and open takes 1.5 s, standing still
                self.assertGreaterEqual(w.duration, 1.5, w.label)
                self.assertEqual(tuple(prev.pos), tuple(w.pos), w.label)
            if w.pos[2] < prev.pos[2] and w.pos[2] <= t.CARRY_Z + 1e-9:  # slow last descents
                self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.02 / 1.5 + 1e-9, w.label)
        # the grips are where the frame is: across its handgrips at its start and at the place
        np.testing.assert_allclose(t.START_GRIP, (L.FRAME_START[0] + L.GRIP_U * math.cos(L.FRAME_START_HEADING),
                                                  L.FRAME_START[1] + L.GRIP_U * math.sin(L.FRAME_START_HEADING)),
                                   atol=1e-12)
        np.testing.assert_allclose(t.PLACE_GRIP, (L.TARGET_CENTER[0] - L.GRIP_U, L.TARGET_CENTER[1]), atol=1e-12)
        # turning the grip by (YAW_PLACE - YAW_START) turns the frame from its start heading to square
        self.assertAlmostEqual(math.remainder(L.FRAME_START_HEADING + (t.YAW_PLACE - t.YAW_START) - L.TARGET_HEADING,
                                              2 * math.pi), 0.0, places=12)
        # the fingers close across the frame's width (its local v axis) at both ends: yaw = heading (mod pi)
        for yaw, h in ((t.YAW_START, L.FRAME_START_HEADING), (t.YAW_PLACE, L.TARGET_HEADING)):
            self.assertAlmostEqual(math.remainder(yaw - h, math.pi), 0.0, places=12)
        # pad tips beside the handgrips: 4 mm below their underside, above the side rails and the crossbar
        tips = t.GRIP_Z - 0.013 - L.TOP
        self.assertLess(tips, L.FRAME_H - L.TUBE - 0.003)
        self.assertGreater(tips, 0.078 + 0.010)
        self.assertAlmostEqual(t.PLACE_Z - t.GRIP_Z, 0.002)
        self.assertEqual(-2 * t.GRIP, 2 * L.FOOT_V - 0.008)

    def test_carry_path_clears_him_and_his_chair(self):
        """Replays the carry (positions and yaw interpolated as ArmPlan does, the frame rigid in the grip) and
        checks the lifted frame's footprint, seen from above, against every part of him and of the chair, and that
        it is set down where the target is."""
        t = self.task
        wps = t._right()
        grab = [i for i, w in enumerate(wps) if w.label == "lift 20 mm"][0]
        put = [i for i, w in enumerate(wps) if w.label == "lower it slowly"][0]
        self.assertEqual(wps[grab].pos[2], t.CARRY_Z)
        self.assertTrue(all(w.pos[2] == t.CARRY_Z for w in wps[grab:put]))  # carried with its feet 20 mm up
        worst = {}
        last = None
        for a, b in zip(wps[grab:put + 1], wps[grab + 1:put + 1]):
            for s in np.linspace(0.0, 1.0, 101):
                k = 3 * s * s - 2 * s * s * s
                p = (1 - k) * np.array(a.pos) + k * np.array(b.pos)
                yaw = a.yaw + k * math.remainder(b.yaw - a.yaw, 2 * math.pi)
                heading = L.FRAME_START_HEADING + (yaw - t.YAW_START)
                cx, cy = p[0] - L.GRIP_U * math.cos(heading), p[1] - L.GRIP_U * math.sin(heading)
                fl, fr, bl, br = [(cx + dx, cy + dy) for dx, dy in L.corner_offsets(heading)]
                poly = [fl, fr, br, bl]
                for name, xs, ys, *_ in (*L.RESIDENT_BOXES, *L.CHAIR_BOXES):
                    worst[name] = min(worst.get(name, 1.0), _poly_rect_distance(poly, xs, ys))
                last = (cx, cy, heading)
        self.assertGreater(min(worst["left_slipper"], worst["right_slipper"]), 0.027)  # README: 28 mm
        him = [n for n, *_ in L.RESIDENT_BOXES if not n.endswith("slipper")]
        self.assertGreater(min(worst[n] for n in him), 0.050)  # README: 51 mm (his shins)
        self.assertGreater(min(worst[n] for n, *_ in L.CHAIR_BOXES), 0.063)  # README: 64 mm (the chair's arm)
        np.testing.assert_allclose(last[:2], L.TARGET_CENTER, atol=1e-9)
        self.assertAlmostEqual(math.remainder(last[2] - L.TARGET_HEADING, 2 * math.pi), 0.0, places=9)

    def test_one_arm_switches(self):
        t = self.task
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", set()),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule_and_matches_everywhere(self):
        ins = self.task.INSTRUCTION
        for s in ("four feet", "directly in front of him", "centred on him", "square to his chair",
                  "open side toward him", "back legs 10 to 50 mm", "1:8", "toes"):
            self.assertIn(s, ins)
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


if __name__ == "__main__":
    unittest.main()
