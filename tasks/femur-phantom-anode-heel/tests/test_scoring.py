"""Offline tests for scoring.py and task.py: synthetic phantom, marker and gripper states, no simulator.
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
    FIELD_TOP_W,
    Tracker,
    align_err_deg,
    centre_offset,
    field_margin,
    flat_on_table,
    hip_toward_cathode,
    hull,
    marker_clearance,
    on_floor,
    phantom_lowest,
    phantom_point_distance,
    phantom_tilt_deg,
    poly_distance,
    valid_marker,
    valid_phantom,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FULL = 2 * WINDOW + 1  # 1 s of history
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}


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


def raw_ph(origin, q, vel=(0.0, 0.0, 0.0)):
    """A phantom state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o.copy(),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.1, high=o + 0.1)


def ph(cx=L.CROSS[0], cy=L.CROSS[1], yaw=0.0, pitch=0.0, roll=0.0, lift=0.0, base=FIELD_TOP_W):
    """The phantom with its centre over (cx, cy), turned (yaw, pitch, roll), its lowest point ``lift`` above
    ``base`` (default: the light field's top). yaw 0: along x, hip end toward +x (the cathode)."""
    q = quat(yaw, pitch, roll)
    st = raw_ph((cx, cy, 0.0), q)
    dz = base + lift - phantom_lowest(st)
    return raw_ph((cx, cy, dz), q)


def mk_marker(x, y, bottom=FIELD_TOP_W, yaw=0.0, tilt=0.0, vel=(0.0, 0.0, 0.0)):
    """A marker box whose lowest corner is at ``bottom``, tilted ``tilt`` degrees about x."""
    q = quat(yaw, 0.0, tilt)
    hx, hy, hz = np.asarray(L.MARKER_SIZE) / 2
    t = math.radians(tilt)
    z = bottom + hz * math.cos(t) + hy * abs(math.sin(t))
    p = np.array([x, y, z])
    return SimpleNamespace(pos=p, quat_xyzw=q, lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3),
                           low=p - 0.02, high=p + 0.02)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


PARK_L = arm((0.20, 0.25, L.TOP + 0.15), 0.43)
PARK_R = arm((0.20, -0.25, L.TOP + 0.16), 0.64)
FAR = {"left": PARK_L, "right": PARK_R}

START_PH = ph(L.PH_START[0], L.PH_START[1], 90.0, base=L.TOP + L.CART_H)
R_TRAY = mk_marker(*L.R_START, bottom=L.TOP + L.TRAY_H)
L_TRAY = mk_marker(*L.L_START, bottom=L.TOP + L.TRAY_H)
CORRECT = ph()
R_OK = mk_marker(0.270, 0.050)


def feed(tr, frames, arms=FAR):
    r = None
    for f in frames:
        r = tr.update(*f, arms)
    return r


def settled(p=CORRECT, r=R_OK, lm=L_TRAY, arms=FAR, n=FULL):
    tr = Tracker(HZ)
    return tr, feed(tr, [(p, r, lm)] * n, arms)


def scored(p=CORRECT, r=R_OK, lm=L_TRAY, arms=FAR):
    return settled(p, r, lm, arms)[1].score


def why(p=CORRECT, r=R_OK, lm=L_TRAY, arms=FAR):
    return settled(p, r, lm, arms)[1].details


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(valid_phantom(START_PH))
        self.assertTrue(Tracker.at_start(START_PH))
        self.assertFalse(on_floor(START_PH))
        self.assertAlmostEqual(phantom_lowest(START_PH), L.TOP + L.CART_H, places=9)
        self.assertEqual(scored(START_PH, R_TRAY, L_TRAY), 0.0)
        self.assertEqual(why(START_PH, R_TRAY, L_TRAY)["zero_reason"], "phantom_not_along_tube_axis")

    def test_centre_of_mass(self):
        com = L.PH_COM_X
        self.assertAlmostEqual(com, 0.01606, delta=0.0002)  # 16 mm toward the hip end
        shaft_c, shaft_s = L.PH_BOXES[1]
        lo, hi = shaft_c[0] - shaft_s[0] / 2, shaft_c[0] + shaft_s[0] / 2
        self.assertGreaterEqual(com - L.FINGER_HALF_WIDTH, lo + 0.010)  # the 27 mm fingers on the 36 mm shaft ...
        self.assertLessEqual(com + L.FINGER_HALF_WIDTH, hi - 0.010)  # ... at least 10 mm from each step

    def test_shape_extents(self):
        xs = [c[0] - s[0] / 2 for c, s in L.PH_BOXES] + [L.PH_HEAD[0][0] + L.PH_HEAD[1]]
        self.assertAlmostEqual(min(xs), -L.PH_LEN / 2)
        self.assertAlmostEqual(max(xs), L.PH_LEN / 2)
        heights = [s[2] for _, s in L.PH_BOXES]
        self.assertEqual(heights, sorted(heights))  # thickest at the hip end
        self.assertEqual((heights[0], heights[-1], L.PH_HEAD[2]), (0.024, 0.048, 0.048))

    def test_correct_pose_measures(self):
        self.assertTrue(flat_on_table(CORRECT))
        self.assertAlmostEqual(align_err_deg(CORRECT), 0.0)
        self.assertTrue(hip_toward_cathode(CORRECT))
        self.assertAlmostEqual(field_margin(CORRECT), 0.015, places=9)  # 15 mm to each field end
        np.testing.assert_allclose(centre_offset(CORRECT), [0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(marker_clearance(R_OK, CORRECT), 0.018, places=9)

    def test_reversed_measures(self):
        rev = ph(yaw=180.0)
        self.assertFalse(hip_toward_cathode(rev))
        self.assertAlmostEqual(align_err_deg(rev), 0.0)
        self.assertAlmostEqual(field_margin(rev), 0.015, places=9)

    def test_head_counts_in_the_outline(self):
        # a marker 6 mm beyond the round hip end along the axis: clear by 6 mm of the head's circle
        self.assertAlmostEqual(marker_clearance(mk_marker(0.390 + 0.006 + 0.012, 0.0), CORRECT), 0.006, places=6)
        # beside the head, 4 mm off its side: too close
        self.assertAlmostEqual(marker_clearance(mk_marker(0.370, 0.020 + 0.004 + 0.012), CORRECT), 0.004, places=6)

    def test_polygons(self):
        sq = hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
        self.assertEqual(len(sq), 4)
        far = sq + np.array([3.0, 0.0])
        self.assertAlmostEqual(poly_distance(sq, far), 2.0)
        bar_x = hull([(-2, -0.1), (2, -0.1), (2, 0.1), (-2, 0.1)])
        bar_y = hull([(-0.1, -2), (0.1, -2), (0.1, 2), (-0.1, 2)])
        self.assertEqual(poly_distance(bar_x, bar_y), 0.0)  # a plus sign: no corner inside the other, still overlapping
        seg = np.array([[0.5, 2.0], [0.5, 3.0]])
        self.assertAlmostEqual(poly_distance(sq, seg), 1.0)

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        holding = arm((L.CROSS[0] + L.PH_COM_X, 0.0, FIELD_TOP_W + 0.024), 40 / 93.9)
        self.assertLess(zone_gap(holding, lambda p: phantom_point_distance(p, CORRECT)), 0.0)
        self.assertGreater(zone_gap(PARK_R, lambda p: phantom_point_distance(p, CORRECT)), 0.05)


class Tiers(unittest.TestCase):
    def test_correct_scores_one(self):
        tr, r = settled()
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["marker_ok"])

    def test_every_tier_is_reachable_and_each_state_has_one_score(self):
        cases = {
            1.0: (CORRECT, R_OK),
            0.75: (CORRECT, R_TRAY),  # no marker
            0.5: (ph(cy=0.030), R_TRAY),  # 30 mm off the cross across the table
            0.25: (ph(yaw=180.0), R_OK),  # hip toward the anode
            0.0: (ph(yaw=90.0), R_OK),  # crosswise
        }
        for want, (p, r) in cases.items():
            self.assertEqual(scored(p, r), want)

    def test_centring_edges(self):
        other_side = mk_marker(0.270, -0.050)  # the marker on the side the phantom moves away from
        self.assertEqual(scored(ph(cy=0.0149), other_side), 1.0)
        self.assertEqual(scored(ph(cy=-0.0149)), 1.0)
        self.assertEqual(scored(ph(cy=0.0151), other_side), 0.5)
        self.assertEqual(scored(ph(cx=0.300 + 0.010, cy=0.0115)), 0.5)  # 15.2 mm diagonally
        self.assertEqual(scored(ph(cy=0.045)), 0.5)  # the farthest that is still wholly in the field
        self.assertEqual(scored(ph(cy=0.0451)), 0.0)

    def test_field_edges_along_the_axis(self):
        self.assertEqual(scored(ph(cx=0.300 + 0.0149)), 1.0)  # 0.1 mm inside the far end
        self.assertEqual(scored(ph(cx=0.300 - 0.0149)), 1.0)
        self.assertEqual(scored(ph(cx=0.300 + 0.0151)), 0.0)
        self.assertEqual(why(ph(cx=0.300 + 0.0151))["zero_reason"], "phantom_not_wholly_in_field")
        self.assertEqual(scored(ph(cx=0.300 - 0.030)), 0.0)  # the spec's '30 mm off along the axis' leaves the field
        # reversed, the round head is at the near end: its circle, not just its centre, must be inside the field
        self.assertEqual(scored(ph(cx=0.300 - 0.0149, yaw=180.0)), 0.25)
        self.assertEqual(scored(ph(cx=0.300 - 0.0151, yaw=180.0)), 0.0)
        self.assertEqual(scored(ph(cy=0.045, yaw=180.0)), 0.25)
        self.assertEqual(scored(ph(cy=-0.0451, yaw=180.0)), 0.0)

    def test_alignment_edges(self):
        self.assertEqual(scored(ph(yaw=9.9)), 1.0)
        self.assertEqual(scored(ph(yaw=-9.9)), 1.0)
        self.assertEqual(scored(ph(yaw=10.1)), 0.0)
        self.assertEqual(scored(ph(yaw=180.0 + 9.9)), 0.25)
        self.assertEqual(scored(ph(yaw=180.0 - 10.1)), 0.0)

    def test_flat_edges(self):
        self.assertEqual(scored(ph(lift=0.0029)), 1.0)
        self.assertEqual(scored(ph(lift=0.0031)), 0.0)
        self.assertEqual(scored(ph(lift=-0.0029)), 1.0)  # sunk into the paint by the solver
        self.assertEqual(scored(ph(roll=4.9)), 1.0)
        self.assertEqual(scored(ph(roll=5.1)), 0.0)
        self.assertEqual(scored(ph(pitch=-4.9)), 1.0)
        self.assertEqual(scored(ph(pitch=5.1)), 0.0)
        self.assertEqual(why(ph(roll=5.1))["zero_reason"], "phantom_not_flat_on_table")

    def test_upside_down_side_and_end(self):
        self.assertEqual(scored(ph(roll=180.0)), 0.0)  # PA, anterior face down
        self.assertEqual(scored(ph(roll=90.0)), 0.0)  # on its side
        self.assertEqual(scored(ph(pitch=-90.0)), 0.0)  # standing on its knee end
        self.assertAlmostEqual(phantom_tilt_deg(ph(roll=180.0)), 180.0, places=6)

    def test_hip_toward_anode_scores_quarter_whatever_the_centring(self):
        self.assertEqual(scored(ph(yaw=180.0), R_OK), 0.25)  # spec shortcut 1
        self.assertEqual(scored(ph(yaw=180.0, cy=0.030), R_OK), 0.25)
        self.assertEqual(scored(ph(yaw=180.0), R_TRAY), 0.25)

    def test_marker_conditions(self):
        on_bone = mk_marker(0.300, 0.0, bottom=FIELD_TOP_W + 0.036)
        cases = {
            "r_marker_not_flat_on_table": on_bone,  # spec shortcut 2: on the bone
            "r_marker_not_in_field": mk_marker(0.270, 0.0655),
            "r_marker_on_or_near_the_bone": mk_marker(0.270, 0.020 + 0.0049 + 0.012),
        }
        for reason, r in cases.items():
            d = why(CORRECT, r)
            self.assertEqual((d["tier"], d["marker_reason"]), (0.75, reason), reason)
        self.assertEqual(scored(CORRECT, mk_marker(0.270, 0.020 + 0.0051 + 0.012)), 1.0)
        self.assertEqual(scored(CORRECT, mk_marker(0.270, 0.0649)), 1.0)  # centre just inside: half off the field
        self.assertEqual(scored(CORRECT, mk_marker(0.270, 0.050, tilt=9.9)), 1.0)
        self.assertEqual(scored(CORRECT, mk_marker(0.270, 0.050, tilt=10.1)), 0.75)
        self.assertEqual(scored(CORRECT, mk_marker(0.270, 0.050, bottom=FIELD_TOP_W + 0.0031)), 0.75)
        self.assertEqual(scored(CORRECT, mk_marker(0.270, -0.050)), 1.0)  # either side of the bone

    def test_l_marker(self):
        lm_in = mk_marker(0.270, -0.050)
        self.assertEqual(why(CORRECT, R_TRAY, lm_in)["marker_reason"], "l_marker_in_field")
        self.assertEqual(scored(CORRECT, R_TRAY, lm_in), 0.75)  # spec shortcut 2: the L marker used instead
        d = why(CORRECT, R_OK, lm_in)
        self.assertEqual((d["tier"], d["marker_reason"]), (0.75, "l_marker_in_field"))  # both markers in the field
        self.assertEqual(scored(CORRECT, R_OK, mk_marker(0.150, 0.0, bottom=L.TOP)), 1.0)  # L dropped, outside

    def test_marker_held_or_moving(self):
        holding = dict(FAR, left=arm((0.270, 0.050, FIELD_TOP_W + 0.017), 24 / 93.9))
        d = why(CORRECT, R_OK, L_TRAY, holding)
        self.assertEqual((d["tier"], d["marker_reason"], d["released"]), (0.75, "r_marker_held", True))  # shortcut 5
        tr = Tracker(HZ)
        frames = [(CORRECT, mk_marker(0.270, 0.050 - 0.0002 * i), L_TRAY) for i in range(FULL)]
        r = feed(tr, frames)
        self.assertEqual((r.score, r.details["marker_reason"]), (0.75, "r_marker_moving"))

    def test_score_is_idempotent(self):
        tr, r = settled()
        for _ in range(3):
            self.assertEqual(tr.result(CORRECT, R_OK, L_TRAY, FAR).score, 1.0)
        self.assertEqual(tr.steps, FULL)


class ZeroConditions(unittest.TestCase):
    def test_phantom_held(self):
        holding = dict(FAR, right=arm((L.CROSS[0] + L.PH_COM_X, 0.0, FIELD_TOP_W + 0.024), 40 / 93.9))
        d = why(CORRECT, R_OK, L_TRAY, holding)
        self.assertEqual((d["tier"], d["zero_reason"]), (0.0, "phantom_held"))  # spec shortcut 5
        # an open gripper hovering 4 mm over the hip end is still 'on' it; 6 mm up it is clear
        top = FIELD_TOP_W + 0.048
        r_open = zone_radius(arm((0, 0, 0), 1.0))
        near = dict(FAR, right=arm((0.37, 0.0, top + 0.013 + r_open + 0.004)))
        clear = dict(FAR, right=arm((0.37, 0.0, top + 0.013 + r_open + 0.006)))
        self.assertEqual(scored(CORRECT, R_OK, L_TRAY, near), 0.0)
        self.assertEqual(scored(CORRECT, R_OK, L_TRAY, clear), 1.0)

    def test_left_gripper_counts_too(self):
        on = dict(FAR, left=arm((0.30, 0.0, FIELD_TOP_W + 0.030), 0.5))
        self.assertEqual(why(CORRECT, R_OK, L_TRAY, on)["zero_reason"], "phantom_held")

    def test_still_moving(self):
        tr = Tracker(HZ)
        frames = [(ph(cy=-0.0005 * (FULL - i)), R_OK, L_TRAY) for i in range(FULL)]
        r = feed(tr, frames)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "phantom_moving"))

    def test_still_path_edges(self):
        for step, want in ((0.00014, 1.0), (0.00016, 0.0)):  # 10 steps per window: 1.4 / 1.6 mm
            tr = Tracker(HZ)
            frames = [(ph(cx=0.300 + step * (i % 2)), R_OK, L_TRAY) for i in range(FULL)]
            self.assertEqual(feed(tr, frames).score, want, step)

    def test_turning_counts_as_moving(self):
        tr = Tracker(HZ)
        frames = [(ph(yaw=-2.5 + 2.5 * i / (FULL - 1)), R_OK, L_TRAY) for i in range(FULL)]
        r = feed(tr, frames)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "phantom_moving"))
        tr = Tracker(HZ)
        frames = [(ph(yaw=-1.5 + 1.5 * i / (FULL - 1)), R_OK, L_TRAY) for i in range(FULL)]
        self.assertEqual(feed(tr, frames).score, 1.0)

    def test_needs_one_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [(CORRECT, R_OK, L_TRAY)] * (FULL - 1))
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "phantom_moving"))
        r = feed(tr, [(CORRECT, R_OK, L_TRAY)])
        self.assertEqual(r.score, 1.0)

    def test_dropped_on_the_floor_latches(self):
        tr = Tracker(HZ)
        floor = ph(cx=0.300, cy=-0.150, yaw=90.0, base=L.TOP)
        feed(tr, [(floor, R_TRAY, L_TRAY)] * 3)
        self.assertEqual(tr.dropped_at, 1)
        r = feed(tr, [(CORRECT, R_OK, L_TRAY)] * (FULL + 40))  # picked up and placed perfectly: spec shortcut 4
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "phantom_dropped_on_floor"))
        self.assertFalse(r.success)
        tr.reset()
        self.assertEqual(feed(tr, [(CORRECT, R_OK, L_TRAY)] * FULL).score, 1.0)

    def test_floor_threshold(self):
        self.assertTrue(on_floor(ph(base=L.TOP + 0.0049)))
        self.assertFalse(on_floor(ph(base=L.TOP + 0.0051)))
        tipped = ph(cx=0.300, cy=0.075, roll=30.0, base=L.TOP)  # slid off the table's edge, one side on the floor
        self.assertTrue(on_floor(tipped))

    def test_crosswise_on_the_table(self):
        self.assertEqual(scored(ph(yaw=90.0)), 0.0)  # spec shortcut 4: it cannot fit the field
        self.assertEqual(scored(ph(yaw=45.0)), 0.0)

    def test_propped_on_a_marker_or_off_the_table(self):
        propped = ph(pitch=-8.0, lift=0.0)  # hip end raised on a marker under it
        self.assertEqual(scored(propped), 0.0)
        self.assertEqual(scored(ph(lift=0.016)), 0.0)  # lying level on two markers
        self.assertEqual(scored(ph(base=L.TOP + L.STRIP_H)), 0.0)  # on the strip's top
        self.assertEqual(scored(ph(cx=0.27, cy=-0.30, yaw=0.0, base=L.TOP + L.CART_H)), 0.0)  # along x on the cart


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled()
        n_hist, n = len(tr.history), tr.steps
        bad = [
            raw_ph((math.nan, 0.0, 0.8), quat()),
            raw_ph((0.3, 0.0, 0.8), (0.0, 0.0, 0.0, 0.0)),
            raw_ph((0.3, 0.0, math.inf), quat()),
            SimpleNamespace(quat_xyzw=quat(), pos=np.zeros(3), lin_vel=np.zeros(3)),  # no origin
        ]
        for b in bad:
            r = tr.update(b, R_OK, L_TRAY, FAR)
            self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "invalid_state"))
        nan_marker = mk_marker(0.27, 0.05)
        nan_marker.pos = np.array([0.27, math.nan, 0.77])
        self.assertFalse(valid_marker(nan_marker))
        self.assertEqual(tr.update(CORRECT, nan_marker, L_TRAY, FAR).score, 0.0)
        self.assertEqual(tr.update(CORRECT, R_OK, SimpleNamespace(pos=None), FAR).score, 0.0)
        self.assertEqual(len(tr.history), n_hist)
        self.assertEqual(tr.bad_steps, len(bad) + 2)
        self.assertEqual(tr.steps, n + len(bad) + 2)
        self.assertEqual(tr.update(CORRECT, R_OK, L_TRAY, FAR).score, 1.0)  # picks up where it was

    def test_unnormalised_quaternion(self):
        q = quat() * 3.0
        p = raw_ph(CORRECT.origin, q)
        self.assertEqual(scored(p), 1.0)

    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr, r = settled(arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(tr.arms_unavailable, FULL)
        self.assertIsNone(r.details["zone_gap_mm"])
        bad_arm = {"left": PARK_L, "right": arm((math.nan, 0, 0))}
        tr, r = settled(arms=bad_arm)
        self.assertEqual(tr.arms_unavailable, FULL)

    def test_unreadable_arms_cannot_skip_stillness(self):
        tr = Tracker(HZ)
        frames = [(ph(cy=-0.0005 * (FULL - i)), R_OK, L_TRAY) for i in range(FULL)]
        self.assertEqual(feed(tr, frames, None).score, 0.0)

    def test_random_states_do_not_crash(self):
        rng = random.Random(7)
        for _ in range(300):
            p = ph(cx=rng.uniform(0.0, 0.6), cy=rng.uniform(-0.5, 0.5), yaw=rng.uniform(-180, 180),
                   pitch=rng.uniform(-20, 20), roll=rng.uniform(-20, 20), lift=rng.uniform(-0.002, 0.05),
                   base=rng.choice([L.TOP, FIELD_TOP_W, L.TOP + L.CART_H]))
            rm = mk_marker(rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.3), yaw=rng.uniform(-90, 90),
                           tilt=rng.uniform(0, 40), bottom=rng.choice([FIELD_TOP_W, L.TOP, FIELD_TOP_W + 0.03]))
            arms = rng.choice([FAR, None, {"left": arm((rng.uniform(0, 0.5), rng.uniform(-0.3, 0.3),
                                                       L.TOP + rng.uniform(0, 0.2)), rng.random()), "right": PARK_R}])
            s = scored(p, rm, L_TRAY, arms)
            self.assertIn(s, TIERS)

    def test_teleport_back_to_start_resets(self):
        tr = Tracker(HZ)
        floor = ph(cx=0.300, cy=-0.150, yaw=90.0, base=L.TOP)
        feed(tr, [(floor, R_TRAY, L_TRAY)] * 5)
        self.assertIsNotNone(tr.dropped_at)
        feed(tr, [(START_PH, R_TRAY, L_TRAY)])  # the simulator put it back: a new trial
        self.assertIsNone(tr.dropped_at)
        self.assertEqual(tr.steps, 1)

    def test_jittered_start_resets_too(self):
        tr = Tracker(HZ)
        feed(tr, [(CORRECT, R_OK, L_TRAY)] * 5)
        jit = ph(L.PH_START[0] + 0.03, L.PH_START[1] - 0.02, 90.0, base=L.TOP + L.CART_H)
        feed(tr, [(jit, R_TRAY, L_TRAY)])
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker(HZ)
        feed(tr, [(START_PH, R_TRAY, L_TRAY)] * 5)
        self.assertEqual(tr.steps, 5)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr = Tracker(HZ)
        feed(tr, [(START_PH, R_TRAY, L_TRAY)] * 3 + [(CORRECT, R_OK, L_TRAY)])
        self.assertEqual(tr.steps, 4)

    def test_explicit_reset(self):
        tr, _ = settled()
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.arms_unavailable, tr.dropped_at), (0, 0, 0, None))


class Exploits(unittest.TestCase):
    """Unintended strategies earn nothing beyond what their end state deserves."""

    def test_marker_laid_on_the_bone(self):
        self.assertEqual(scored(CORRECT, mk_marker(0.300, 0.0, bottom=FIELD_TOP_W + 0.036)), 0.75)

    def test_marker_tucked_under_the_bone(self):
        self.assertEqual(scored(ph(lift=0.016), mk_marker(0.300, 0.0)), 0.0)

    def test_phantom_stood_on_its_hip_end_in_the_field(self):
        self.assertEqual(scored(ph(pitch=90.0)), 0.0)

    def test_both_markers_in_the_field_as_a_hedge(self):
        self.assertEqual(scored(CORRECT, R_OK, mk_marker(0.270, -0.050)), 0.75)

    def test_phantom_left_on_the_table_edge_half_out(self):
        self.assertEqual(scored(ph(cy=0.060)), 0.0)

    def test_held_in_place_by_the_gripper(self):
        holding = dict(FAR, right=arm((L.CROSS[0] + L.PH_COM_X, 0.0, FIELD_TOP_W + 0.024), 40 / 93.9))
        self.assertEqual(scored(CORRECT, R_OK, L_TRAY, holding), 0.0)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate

    def sim(self, p, r=R_OK, lm=L_TRAY):
        """Synthetic states turned into what the simulator would report (mechkit's AssemblyState for the phantom)."""
        st = self.mk.state_from_poses(self.task.PHANTOM, (*p.origin, *p.quat_xyzw), source="sim")
        return {"femur_phantom": st, "r_marker": r, "l_marker": lm}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def _bounds(self, o):
        if isinstance(o, self.mk.Assembly):
            return o.world_bounds()
        return np.asarray(o.pos) - np.asarray(o.size) / 2, np.asarray(o.pos) + np.asarray(o.size) / 2

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in t.OBJECTS if o.mass is not None),
                         ["femur_phantom", "l_marker", "r_marker"])
        for o in t.OBJECTS:
            if o.mass is not None:
                self.assertLessEqual(o.mass, 0.30)
            lo, hi = self._bounds(o)
            # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        self.assertAlmostEqual(t.R_MARKER.pos[2] - L.MARKER_SIZE[2] / 2, L.TOP + L.TRAY_H)

    def test_paint_tiles_the_field_flat(self):
        pieces = self.task.FIELD + self.task.CROSS
        area = sum(o.size[0] * o.size[1] for o in pieces)
        fx, fy = L.FIELD_X[1] - L.FIELD_X[0], L.FIELD_Y[1] - L.FIELD_Y[0]
        self.assertAlmostEqual(area, fx * fy, places=12)
        for o in pieces:
            self.assertAlmostEqual(o.pos[2] + o.size[2] / 2, FIELD_TOP_W, places=12)  # all flush
        for i, a in enumerate(pieces):
            alo, ahi = self._bounds(a)
            for b in pieces[i + 1:]:
                blo, bhi = self._bounds(b)
                ov = np.minimum(ahi, bhi) - np.maximum(alo, blo)
                self.assertFalse(ov[0] > 1e-9 and ov[1] > 1e-9, (a.name, b.name))
        # the cross is centred on (0.300, 0.000), 60 mm each way
        cx = self.task.CROSS[0]
        self.assertEqual((round(cx.pos[0], 6), round(cx.pos[1], 6), round(cx.size[0], 6)), (0.3, 0.0, 0.06))

    def test_tube_plates(self):
        t = self.task
        by = {o.name: o for o in t.OBJECTS}
        self.assertGreater(by["cathode_plate"].pos[0], by["anode_plate"].pos[0])  # "-" at the far (+x) end
        self.assertEqual(L.CATHODE_DIR, 1.0)
        self.assertLess(by["tube_housing"].pos[1], L.TABLE_Y[0])  # along the table's -y edge

    def test_phantom_geometry_matches_the_scorer(self):
        sp = self.task.PHANTOM
        lo, hi = sp.local_bounds()
        np.testing.assert_allclose(lo, [-0.090, -0.020, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.090, 0.020, 0.048], atol=1e-12)
        self.assertAlmostEqual(float(sp.center_of_mass()[0]), L.PH_COM_X, places=9)
        self.assertLess(abs(float(sp.center_of_mass()[1])), 1e-12)
        boxes = [(tuple(s.center), tuple(s.size)) for s in sp.shapes if s.kind == "box"]
        for c, s in L.PH_BOXES:
            self.assertIn((tuple(map(float, c)), tuple(map(float, s))), boxes)
        cyl = [s for s in sp.shapes if s.kind == "cylinder"]
        self.assertEqual(len(cyl), 1)
        self.assertEqual((tuple(cyl[0].center), cyl[0].radius, cyl[0].height), L.PH_HEAD)
        vol = sum(s.volume() for s in sp.shapes)
        self.assertAlmostEqual(sp.mass / vol, 790, delta=15)  # kg/m3: a foam-cored plastic phantom
        rest = sp.rest_state()
        self.assertTrue(Tracker.at_start(rest))
        self.assertGreater(rest.high[1], L.PH_START[1] + 0.089)  # the hip end toward +y at the start
        st = self.sim(CORRECT)["femur_phantom"]
        np.testing.assert_allclose(st.origin, CORRECT.origin, atol=1e-12)
        self.assertAlmostEqual(st.low[2], FIELD_TOP_W, places=9)
        self.assertAlmostEqual(st.high[0], 0.390, places=9)

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.PHANTOM.offline_box()[0]), quat_xyzw=quat(90.0),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.PHANTOM.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.PH_START[0], L.PH_START[1], L.TOP + L.CART_H], atol=1e-12)
        self.assertTrue(Tracker.at_start(st))

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        start = self.sim(START_PH, R_TRAY, L_TRAY)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        good = self.sim(CORRECT)
        for _ in range(FULL - 1):
            self.feed_arms(FAR)
            t.success(good)
        self.feed_arms(FAR)
        self.assertTrue(t.success(good))
        self.assertEqual(t.score(good), 1.0)
        self.assertEqual(t.score(good), 1.0)  # score() does not advance it
        holding = dict(FAR, right=arm((L.CROSS[0] + L.PH_COM_X, 0.0, FIELD_TOP_W + 0.024), 40 / 93.9))
        self.feed_arms(holding)
        self.assertEqual(t.score(good), 0.0)  # read with the arms of the step at hand
        self.feed_arms(FAR)
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(FAR)
        t.success(good)
        t.success(good)
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(good))
        self.assertEqual(tr.steps, 1)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.rig.offline_reset()
                self.feed_arms(FAR)
                self.task.success(self.sim(CORRECT))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "tier", "zero_reason", "tilt_deg", "lowest_mm", "flat",
                          "align_err_deg", "heading_deg", "hip_to_cathode", "field_margin_mm", "centre_off_mm",
                          "centre_dist_mm", "released", "zone_gap_mm", "still", "path_mm", "turn_deg",
                          "marker_ok", "marker_reason", "r_released", "r_still", "r_tilt_deg", "r_flat",
                          "r_in_field", "r_clear_mm", "l_in_field", "dropped_at", "arms_unavailable", "ph_origin",
                          "ph_quat", "ph_yaw_deg", "ph_off_x_mm", "ph_off_y_mm", "r_pos", "l_pos", "right_tcp",
                          "left_opening", "source"):
                    self.assertIn(k, row)
                self.assertAlmostEqual(row["ph_off_x_mm"], 0.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(CORRECT)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, side)  # every close and open takes 1.5 s
            by = {w.label: w for w in wps}
            self.assertGreaterEqual(by["slowly onto the light field"].duration, 1.5)
            self.assertEqual(by["let go of it"].pos, by["slowly onto the light field"].pos)
        right = {w.label: w for w in plan["right"]}
        grip, carry = right["grip the phantom"], right["carry it over the table, hip end turning to the cathode"]
        # the phantom turns with the gripper: from heading +90 (hip +y) to +90 + (place - pick) = -1 deg (hip +x)
        end_heading = math.degrees(L.PH_START_YAW + carry.yaw - grip.yaw)
        self.assertAlmostEqual(end_heading, -1.0, places=9)
        self.assertTrue(all(w.yaw == t.YAW_PLACE for w in plan["right"][7:15]))
        # gripped over its centre of mass, 8 mm narrower than its 40 mm, pad tips 11 mm above the cart
        self.assertAlmostEqual(grip.pos[0], L.PH_START[0])
        self.assertAlmostEqual(grip.pos[1], L.PH_START[1] + L.PH_COM_X)
        self.assertAlmostEqual(-2 * grip.finger, L.PH_W - 0.008)
        self.assertAlmostEqual(grip.pos[2] - 0.013 - (L.TOP + L.CART_H), 0.011)
        setw = right["slowly onto the light field"]
        self.assertAlmostEqual(setw.pos[0] - L.PH_COM_X, L.CROSS[0] + t.PH_TRIM[0])  # its centre over the cross
        self.assertAlmostEqual(setw.pos[2] - t.GRIP_UP - FIELD_TOP_W, t.DROP)
        self.assertLessEqual(abs(t.PH_TRIM[1]), 0.005)
        # carried with its underside well over the strip's top
        self.assertGreater(carry.pos[2] - t.GRIP_UP - (L.TOP + L.STRIP_H), 0.05)
        left = {w.label: w for w in plan["left"]}
        mgrip, mset = left["grip the R marker"], left["slowly onto the light field"]
        self.assertAlmostEqual(-2 * mgrip.finger, L.MARKER_SIZE[0] - 0.008)
        self.assertEqual((mgrip.pos[0], mgrip.pos[1]), L.R_START)
        self.assertAlmostEqual(mgrip.pos[2] - 0.013 - (L.TOP + L.TRAY_H), 0.004)
        # fingers close along x at the tray: the open pads (40 mm, 8 mm thick, 27 mm wide) stay clear of the L marker
        self.assertAlmostEqual(abs(mgrip.yaw), math.pi / 2)
        self.assertGreater((L.L_START[1] - L.MARKER_SIZE[1] / 2) - (L.R_START[1] + L.FINGER_HALF_WIDTH), 0.010)
        # the marker's place: inside the field, clear of the phantom laid at its trimmed place
        m = mk_marker(mset.pos[0], mset.pos[1])
        laid = ph(cx=L.CROSS[0] + t.PH_TRIM[0], cy=L.CROSS[1] + t.PH_TRIM[1], yaw=-1.0)
        self.assertGreater(marker_clearance(m, laid), 0.010)
        self.assertLess(m.pos[1] + L.MARKER_SIZE[1] / 2, L.FIELD_Y[1])
        # the left arm leaves the tray only after the right gripper is back up over the phantom
        t_right_up = t._right_up_at()
        t_left_go = sum(w.duration for w in plan["left"][:plan["left"].index(left["wait for the phantom to be let go"])
                                                         + 1])
        self.assertGreaterEqual(t_left_go, t_right_up)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_both_turns_are_open_to_the_right_arm(self):
        """The wrong answer (hip toward the anode) is the same grip at the pick taken at yaw -90 instead of +90, laid at
        the same yaw 0: both are inside the arm's reach and joint 6's range, so the turn the arm can make does not
        give the answer away."""
        off = HERE.parent.parent / "tools" / "offline"
        if not (off / "yamkin.py").is_file():
            self.skipTest("needs tools/offline/yamkin.py")
        sys.path.insert(0, str(off))
        import yamkin as K
        base = np.asarray(K.BASES["station"]["right"], dtype=float)
        t = self.task
        px, py, pz = t.PH_PICK
        for yaw_pick, place_x in ((math.pi / 2, L.CROSS[0] + L.PH_COM_X), (-math.pi / 2, L.CROSS[0] - L.PH_COM_X)):
            for pos, yaw in (((px, py, pz), yaw_pick), ((px, py, t.PH_HOVER), yaw_pick),
                             ((place_x, 0.0, t.PH_PLACE[2]), 0.0), ((place_x, 0.0, t.PH_HOVER_PLACE), 0.0)):
                ok, q, _, _ = K.reachable(pos, yaw, base)
                self.assertTrue(ok, (pos, yaw))
                self.assertGreater(float(np.min(K.joint_margins(q))), math.radians(5.0), (pos, yaw))

    def test_instruction(self):
        ins = self.task.INSTRUCTION
        for s in ("AP right femur", "tube housing", "- and +", "centred on the black cross", "light field",
                  "red R, blue L", "without covering the bone", "let go of everything"):
            self.assertIn(s, ins)
        for s in ("hip", "cathode", "anode", "thick", "radiation", "exposure"):  # the rule is not given away
            self.assertNotIn(s, ins.lower())
        self.assertLessEqual(len(ins), 490)
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], ins)
        for k in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes"):
            low = spec[k].lower()
            self.assertNotIn("radiation", low)
            self.assertNotIn("exposure", low)
            self.assertNotIn(chr(0x2014), spec[k])
            self.assertNotIn(chr(0x2013), spec[k])


if __name__ == "__main__":
    unittest.main()
