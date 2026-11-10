"""Offline tests for scoring.py and task.py: synthetic specimen states and gripper snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

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
    anvil_gaps,
    arms_valid,
    ends,
    heading_err_deg,
    notch_angle_deg,
    notch_offset,
    point_distance,
    pose_tier,
    supports,
    tilt_deg,
    valid,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
SUP_W = L.TOP + L.SUP_TOP


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


def raw(origin, q, vel=(0.0, 0.0, 0.0)):
    """A specimen state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o.copy(),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.06, high=o + 0.06)


def spec(off=0.0, gap=0.0, under=0.0, yaw=90.0, pitch=0.0, roll=0.0):
    """The bar with the centre of its notch's mouth ``off`` m along y from the midplane, ``gap`` m in front of the
    anvils' faces (x) and its mid-height ``under`` m above where a bar lying on the supports has it. yaw 90: along y,
    notch toward +x (away from the striker); pitch and roll turn it about the notch's mouth."""
    r = _rot(quat(yaw, pitch, roll))
    notch_w = np.array([L.FACE_X - gap, L.MID_Y + off, SUP_W + under + L.SP_H / 2])
    return raw(notch_w - r @ np.array(L.NOTCH_LOCAL), quat(yaw, pitch, roll))


def backwards(off=0.0, gap=0.0):
    """Laid notch toward the striker (yaw -90): its back face ``gap`` in front of the anvils, the notch ``off``."""
    r = _rot(quat(-90.0))
    notch_w = np.array([L.FACE_X - gap - L.SP_W, L.MID_Y + off, SUP_W + L.SP_H / 2])
    return raw(notch_w - r @ np.array(L.NOTCH_LOCAL), quat(-90.0))


def skewed(theta_deg, pivot_y=L.SP_LEN / 2):
    """The seated bar turned theta about a vertical axis through its notched face's end at y = pivot_y: it touches
    one anvil only (at that end), the other end standing off."""
    s = spec()
    p = np.array([L.FACE_X, pivot_y, 0.0])
    t = math.radians(theta_deg)
    rz = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0.0, 0.0, 1.0]])
    o = np.asarray(s.origin)
    o2 = p + rz @ (o - p)
    o2[2] = o[2]
    return raw(o2, quat(90.0 + theta_deg))


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


READY_LEFT = arm((0.20, 0.305, L.TOP + 0.25))
PARKED_RIGHT = arm((0.22, -0.20, L.TOP + 0.15), 0.53)
FAR = {"left": READY_LEFT, "right": PARKED_RIGHT}
# the right gripper closed on the seated bar as the demo holds it (TCP 17 mm above its underside, 12 mm commanded)
HOLDING = {"left": READY_LEFT, "right": arm((L.FACE_X - L.SP_W / 2, 0.0, SUP_W + 0.017), 12.0 / 93.9)}

START = raw((L.SP_START[0], L.SP_START[1], L.TOP + L.SP_START[2]), quat(0.0))
SEATED = spec()


def feed(tr, states, arms=FAR):
    r = None
    for st in states:
        r = tr.update(st, arms)
    return r


def settled(st, arms=FAR, n=WINDOW + 1):
    """A tracker that has seen ``st`` at rest for ``n`` steps; returns (tracker, last result)."""
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n, arms)


def scored(st, arms=FAR):
    return settled(st, arms)[1].score


def glide(a, b, n):
    """n states moving the origin from state a's to state b's, b's orientation."""
    oa, ob = np.asarray(a.origin, float), np.asarray(b.origin, float)
    vel = (ob - oa) * HZ / n
    return [raw(oa + (ob - oa) * (i / n), b.quat_xyzw, vel) for i in range(1, n + 1)]


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(valid(START))
        self.assertTrue(Tracker.at_start(START))
        a, b = ends(START)
        np.testing.assert_allclose(a, [L.MAT_X - 0.055, L.MAT_Y, L.TOP + L.MAT_TOP + 0.010], atol=1e-12)
        np.testing.assert_allclose(b, [L.MAT_X + 0.055, L.MAT_Y, L.TOP + L.MAT_TOP + 0.010], atol=1e-12)
        self.assertAlmostEqual(heading_err_deg(START), 90.0)
        self.assertAlmostEqual(notch_angle_deg(START), 90.0)  # notch toward -y: sideways to the tester's axis
        self.assertEqual(pose_tier(START)[0], 0.0)
        self.assertFalse(Tracker.at_start(SEATED))

    def test_seated_measures(self):
        self.assertAlmostEqual(tilt_deg(SEATED), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(SEATED), 0.0, places=6)
        self.assertAlmostEqual(notch_angle_deg(SEATED), 0.0, places=5)
        self.assertAlmostEqual(notch_offset(SEATED), 0.0, places=12)
        gl, gr = anvil_gaps(SEATED)
        self.assertAlmostEqual(gl, 0.0, places=12)
        self.assertAlmostEqual(gr, 0.0, places=12)
        sup = supports(SEATED)
        self.assertTrue(sup["on_supports"])
        np.testing.assert_allclose(sup["onto_mm"], [15.0, 15.0], atol=1e-9)  # ends at |y| 55, inner edges at 40
        np.testing.assert_allclose(sup["underside_mm"], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(sup["end_x_mm"], [320.0, 320.0], atol=1e-9)

    def test_offset_and_gap_read_as_a_rule_and_feeler_would(self):
        s = spec(off=0.003, gap=0.001)
        self.assertAlmostEqual(notch_offset(s), 0.003, places=12)
        # rule readings from each anvil's inner edge to the notch: 37 and 43 mm; half the difference is 3 mm
        a = L.ANVIL_Y_IN - notch_offset(s)
        b = notch_offset(s) + L.ANVIL_Y_IN
        self.assertAlmostEqual(abs(a - b) / 2, 0.003, places=12)
        gl, gr = anvil_gaps(s)
        self.assertAlmostEqual(gl, 0.001, places=12)
        self.assertAlmostEqual(gr, 0.001, places=12)

    def test_upside_down_reads_the_same(self):
        flipped = spec(yaw=-90.0, roll=180.0)  # tick face down, notch still toward +x
        self.assertAlmostEqual(tilt_deg(flipped), 0.0, places=6)
        self.assertAlmostEqual(notch_angle_deg(flipped), 0.0, places=5)
        self.assertAlmostEqual(notch_offset(flipped), 0.0, places=12)
        np.testing.assert_allclose(anvil_gaps(flipped), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(supports(flipped)["underside_mm"], [0.0, 0.0], atol=1e-9)

    def test_backwards_reads_toward_the_striker(self):
        b = backwards()
        self.assertAlmostEqual(notch_angle_deg(b), 180.0, places=5)
        self.assertTrue(supports(b)["on_supports"])

    def test_skew_opens_one_anvil(self):
        s = skewed(-1.0)  # pivot on the left anvil: the right end stands off
        gl, gr = anvil_gaps(s)
        self.assertLess(abs(gl), 0.0003)
        self.assertAlmostEqual(gr, L.SP_LEN * math.tan(math.radians(1.0)), delta=0.0001)  # 1.92 mm at the end
        self.assertEqual(anvil_gaps(spec(yaw=0.0)), (math.inf, math.inf))  # across the tester: no face in front

    def test_every_scored_pose_rests_on_both_supports(self):
        # up to the 8 mm off-centre limit each end still reaches 7 mm onto its support, and a bar up to 10 mm in
        # front of the anvils still lies wholly on the 30 mm deep supports
        for off in (-0.008, 0.008):
            self.assertGreaterEqual(min(supports(spec(off=off))["onto_mm"]), 7.0 - 1e-9)
        s = spec(gap=0.010)
        self.assertTrue(supports(s)["on_supports"])
        self.assertGreaterEqual(L.FACE_X - 0.010 - L.SP_W, L.SUP_X0 - 1e-12)

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 1), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(arm((0, 0, 1), 0.0)), math.hypot(0.008, 0.0135))
        self.assertLess(zone_gap(HOLDING["right"], SEATED), 0.0)
        self.assertGreater(zone_gap(PARKED_RIGHT, SEATED), 0.05)
        self.assertEqual(point_distance(np.asarray(SEATED.origin) + np.array([0, 0, 0.005]), SEATED), 0.0)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(scored(START), 0.0)

    def test_seated_scores_one(self):
        tr, r = settled(SEATED)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["at_rest"])
        self.assertTrue(r.details["hands_off"])
        self.assertEqual(r.details["zero_reason"], "")

    def test_centring_edges(self):
        for off, want in ((0.002, 1.0), (-0.002, 1.0), (0.0021, 0.75), (-0.0021, 0.75), (0.005, 0.75),
                          (0.008, 0.75), (-0.008, 0.75), (0.0081, 0.0), (-0.0081, 0.0), (0.012, 0.0)):
            with self.subTest(off=off):
                self.assertEqual(scored(spec(off=off)), want)

    def test_anvil_gap_edges(self):
        for gap, want in ((0.00149, 1.0), (0.0016, 0.75), (0.003, 0.75), (0.008, 0.75), (-0.0005, 1.0)):
            with self.subTest(gap=gap):
                self.assertEqual(scored(spec(gap=gap)), want)

    def test_half_credit(self):
        self.assertEqual(scored(spec(off=0.005, gap=0.003)), 0.5)
        self.assertEqual(scored(spec(off=-0.0075, gap=0.009)), 0.5)

    def test_every_tier_is_reachable_and_each_pose_has_one_score(self):
        got = {}
        for off in np.arange(-0.010, 0.0101, 0.0005):
            for gap in np.arange(0.0, 0.0101, 0.0005):
                got.setdefault(scored(spec(off=off, gap=gap)), 0)
                got[scored(spec(off=off, gap=gap))] += 1
        self.assertEqual(set(got), {0.0, 0.5, 0.75, 1.0})

    def test_one_anvil_only_is_not_square(self):
        self.assertEqual(scored(skewed(-0.5)), 1.0)  # 0.96 mm at the far end: the feeler does not enter
        self.assertEqual(scored(skewed(-1.0)), 0.75)  # 1.92 mm at the far end
        self.assertEqual(scored(skewed(1.0, pivot_y=-L.SP_LEN / 2)), 0.75)  # the other way round

    def test_small_tilts_count(self):
        self.assertEqual(scored(spec(roll=4.5)), 1.0)
        self.assertEqual(scored(spec(roll=-4.5)), 1.0)
        self.assertEqual(scored(spec(pitch=1.0)), 1.0)  # ends 1 mm above and below the supports' tops

    def test_upside_down_is_the_same_bar(self):
        self.assertEqual(scored(spec(yaw=-90.0, roll=180.0)), 1.0)

    def test_underside_within_2_mm(self):
        self.assertEqual(scored(spec(under=0.0019)), 1.0)
        self.assertEqual(scored(spec(under=-0.0005)), 1.0)  # solver penetration

    def test_score_is_idempotent(self):
        tr, _ = settled(SEATED)
        n, h = tr.steps, len(tr.history)
        for _ in range(3):
            self.assertEqual(tr.result(SEATED, FAR).score, 1.0)
        self.assertEqual((tr.steps, len(tr.history)), (n, h))


class ZeroConditions(unittest.TestCase):
    def assertZero(self, st, reason, arms=FAR):
        tr, r = settled(st, arms)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], reason)

    def test_notch_toward_the_striker(self):
        for off, gap in ((0.0, 0.0), (0.001, 0.0), (0.0, 0.003)):
            self.assertZero(backwards(off, gap), "notch_toward_striker")

    def test_notch_cone_is_30_degrees(self):
        # lying flat along the anvils the notch can only point within about 11 degrees of +x or of -x: the 30 degree
        # cone takes every such pose the right way round and none the wrong way round
        for yaw in (80.5, 90.0, 99.5):
            for roll in (-4.5, 0.0, 4.5):
                self.assertLessEqual(notch_angle_deg(spec(yaw=yaw, roll=roll)), 11.5)
        b = backwards(gap=0.009)
        r = _rot(quat(-81.0, 0.0, 4.5))
        notch_w = np.array([L.FACE_X - 0.009 - L.SP_W, 0.0, SUP_W + L.SP_H / 2])
        skew = raw(notch_w - r @ np.array(L.NOTCH_LOCAL), quat(-81.0, 0.0, 4.5))  # backwards, turned 9, rolled 4.5
        self.assertGreater(notch_angle_deg(skew), 168.0)
        self.assertTrue(supports(skew)["on_supports"])
        self.assertZero(skew, "notch_toward_striker")
        self.assertZero(b, "notch_toward_striker")
        self.assertAlmostEqual(notch_angle_deg(spec(yaw=120.0)), 30.0, places=6)
        self.assertAlmostEqual(notch_angle_deg(spec(yaw=180.0)), 90.0, places=6)

    def test_off_centre(self):
        self.assertZero(spec(off=0.0085), "off_centre")
        self.assertZero(spec(off=-0.011), "off_centre")  # the left end still 4 mm onto its support

    def test_not_lying_flat(self):
        self.assertZero(spec(roll=5.5), "not_lying_flat")
        self.assertZero(spec(roll=90.0), "not_lying_flat")  # on its back face, notch up
        self.assertZero(spec(roll=-90.0), "not_lying_flat")  # notch down
        self.assertZero(spec(pitch=90.0), "not_lying_flat")  # standing on an end

    def test_not_along_the_anvils(self):
        self.assertZero(spec(yaw=79.0), "not_along_the_anvils")
        self.assertZero(spec(yaw=0.0), "not_along_the_anvils")

    def test_not_on_the_supports(self):
        self.assertZero(spec(under=0.0025), "not_on_the_supports")  # held or propped up
        self.assertZero(spec(under=-0.010), "not_on_the_supports")  # down on the base between the supports
        self.assertZero(spec(under=L.ANVIL_TOP - L.SUP_TOP, gap=-0.010), "not_on_the_supports")  # on the anvils
        self.assertZero(spec(gap=0.025), "not_on_the_supports")  # pulled back off the supports' front edge
        self.assertZero(spec(pitch=3.0), "not_on_the_supports")  # one end 2.9 mm up

    def test_one_end_short_of_its_support(self):
        sup = supports(spec(off=0.013))  # the right end only 2 mm past its support's inner edge
        self.assertFalse(sup["on_supports"])
        self.assertAlmostEqual(min(sup["onto_mm"]), 2.0, places=6)

    def test_held_or_touched(self):
        self.assertZero(SEATED, "gripper_on_or_near_it", HOLDING)
        # open, straddling it 10 mm above: still on it
        over = {"left": READY_LEFT, "right": arm((L.FACE_X - L.SP_W / 2, 0.0, SUP_W + 0.040), 0.53)}
        self.assertZero(SEATED, "gripper_on_or_near_it", over)

    def test_hands_off_is_20_mm(self):
        r53 = math.hypot(0.53 * 0.0939 / 2 + 0.008, 0.0135)
        top = SUP_W + L.SP_H + L.TICK_SIZE[2]
        for clear, want in ((0.019, 0.0), (0.021, 1.0)):
            tips_z = top + r53 + clear
            a = {"left": READY_LEFT, "right": arm((L.FACE_X - L.SP_W / 2, 0.0, tips_z + 0.013), 0.53)}
            with self.subTest(clear=clear):
                self.assertEqual(scored(SEATED, a), want)

    def test_left_gripper_counts_too(self):
        a = {"left": arm((L.FACE_X - L.SP_W / 2, 0.03, SUP_W + 0.05), 1.0), "right": PARKED_RIGHT}
        self.assertZero(SEATED, "gripper_on_or_near_it", a)

    def test_still_moving(self):
        tr = Tracker(HZ)
        r = feed(tr, glide(spec(gap=0.004), SEATED, 4))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "still_moving")


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [SEATED] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["moved_mm"])
        self.assertEqual(tr.update(SEATED, FAR).score, 1.0)

    def test_comes_to_rest(self):
        tr = Tracker(HZ)
        feed(tr, glide(spec(gap=0.010), SEATED, 2))  # slid 10 mm into the anvils, 5 mm a step
        for _ in range(WINDOW - 1):
            self.assertEqual(tr.update(SEATED, FAR).score, 0.0)
        self.assertEqual(tr.update(SEATED, FAR).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        self.assertEqual(feed(tr, [spec(gap=(0.0019 if i % 2 else 0.0)) for i in range(WINDOW + 5)]).score, 1.0)
        tr = Tracker(HZ)
        self.assertEqual(feed(tr, [spec(off=(0.0021 if i % 2 else 0.0)) for i in range(WINDOW + 5)]).score, 0.0)

    def test_partial_tiers_need_rest_too(self):
        tr = Tracker(HZ)
        r = feed(tr, glide(spec(off=0.005, gap=0.004), spec(off=0.005, gap=0.003), 3))
        self.assertEqual((r.score, r.details["tier"]), (0.0, 0.5))


class Exploits(unittest.TestCase):
    def test_backwards_bar_centred_against_the_anvils(self):
        # every measurement right except the notch side: the striker would hit the notch
        self.assertEqual(scored(backwards()), 0.0)

    def test_held_in_place_against_the_anvils(self):
        self.assertEqual(scored(SEATED, HOLDING), 0.0)

    def test_pushed_against_the_anvils_by_an_open_gripper(self):
        # an open gripper behind the bar, one pad pressing its back face: the finger zone reaches it
        push = {"left": READY_LEFT, "right": arm((L.FACE_X - L.SP_W - 0.040, 0.0, SUP_W + 0.030), 0.53)}
        self.assertEqual(scored(SEATED, push), 0.0)

    def test_on_its_side_against_the_anvils(self):
        self.assertEqual(scored(spec(roll=90.0)), 0.0)

    def test_laid_on_the_anvils_tops(self):
        self.assertEqual(scored(spec(gap=-0.010, under=L.ANVIL_TOP - L.SUP_TOP)), 0.0)

    def test_dropped_in_the_gap(self):
        self.assertEqual(scored(spec(under=-(L.SUP_TOP - L.BASE_H))), 0.0)

    def test_standing_on_an_end_between_the_anvils(self):
        self.assertEqual(scored(spec(pitch=90.0)), 0.0)

    def test_propped_across_a_support_and_the_base(self):
        # one end on the right support, the other down on the base: pitched about 6 degrees
        self.assertEqual(scored(spec(pitch=6.0, off=0.02)), 0.0)

    def test_arms_unreadable_cannot_be_used_to_skip_the_rest(self):
        # unreadable arms only waive the hands rule: a backwards bar still scores 0
        self.assertEqual(scored(backwards(), None), 0.0)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr, r = settled(SEATED, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)
        self.assertIsNone(r.details["zone_gap_mm"])

    def test_non_finite_arms_count_as_unavailable(self):
        bad = {"left": READY_LEFT, "right": arm((math.nan, 0.0, 1.0))}
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": READY_LEFT}))
        self.assertFalse(arms_valid(None))
        tr, r = settled(SEATED, bad)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)

    def test_hands_off_applies_when_the_arms_come_back(self):
        tr, _ = settled(SEATED, None)
        self.assertEqual(tr.update(SEATED, HOLDING).score, 0.0)
        self.assertEqual(tr.update(SEATED, FAR).score, 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(SEATED)
        n, hist = tr.steps, list(tr.history)
        bad = [
            raw((math.nan, 0.0, 0.8), quat(90.0)),
            raw((0.32, math.inf, 0.8), quat(90.0)),
            raw((0.32, 0.0, 0.8), (math.nan, 0, 0, 1)),
            raw((0.32, 0.0, 0.8), (0.0, 0.0, 0.0, 0.0)),
            SimpleNamespace(quat_xyzw=quat(90.0), pos=np.zeros(3), lin_vel=np.zeros(3)),  # no origin
            SimpleNamespace(origin=np.zeros(2), quat_xyzw=quat(90.0), pos=np.zeros(3)),  # malformed
            None,
        ]
        for b in bad:
            with self.subTest(b=b):
                self.assertFalse(valid(b))
                r = tr.update(b, FAR)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.steps, n + len(bad))
        self.assertEqual(len(tr.history), len(hist))
        for (a, b), (c, d) in zip(tr.history, hist):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
        self.assertEqual(tr.update(SEATED, FAR).score, 1.0)  # and it carries on

    def test_unnormalised_quaternion(self):
        s = spec()
        self.assertEqual(scored(raw(s.origin, 3.0 * np.asarray(s.quat_xyzw))), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(11)
        tr = Tracker(HZ)
        for _ in range(500):
            q = rng.normal(size=4)
            o = rng.uniform([-0.2, -0.5, 0.6], [0.8, 0.5, 1.2])
            a = {"left": arm(rng.uniform(-1, 1, 3), rng.uniform(-1, 2)), "right": arm(rng.uniform(-1, 1, 3))}
            r = tr.update(raw(o, q), a if rng.random() < 0.8 else None)
            self.assertIn(r.score, (0.0, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(SEATED)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START, FAR)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.history), 1)
        self.assertEqual(r.score, 0.0)

    def test_reset_after_a_full_trial_then_a_new_trial(self):
        tr, _ = settled(SEATED)
        tr.update(START, FAR)
        self.assertEqual(feed(tr, [SEATED] * WINDOW).score, 0.0)  # not yet 0.5 s of the new trial seen
        self.assertEqual(tr.update(SEATED, FAR).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(SEATED)
        tr.update(raw((L.SP_START[0] + 0.03, L.SP_START[1] - 0.03, L.TOP + L.SP_START[2]), quat(0.0)), FAR)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(SEATED)
        feed(tr, glide(SEATED, START, 40))
        self.assertEqual(tr.steps, WINDOW + 1 + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(SEATED)
        tr.update(spec(gap=0.005), FAR)
        self.assertEqual(tr.steps, WINDOW + 2)
        self.assertEqual(tr.update(spec(gap=0.005), FAR).score, 0.0)  # the jump is in the window: moving

    def test_explicit_reset(self):
        tr, _ = settled(SEATED, None)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.arms_unavailable), (0, 0, 0))
        self.assertIsNone(tr.prev_origin)


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

    def sim(self, st):
        """A synthetic state turned into the AssemblyState mechkit would report from the simulator."""
        return {"specimen": self.mk.state_from_poses(self.task.SPECIMEN, (*st.origin, *st.quat_xyzw), source="sim")}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], ["specimen"])
        self.assertAlmostEqual(t.SPECIMEN.mass, 0.118)
        self.assertLessEqual(t.SPECIMEN.mass, 0.30)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        al, ar = t.ANVILS
        self.assertAlmostEqual(al.pos[0] - al.size[0] / 2, L.FACE_X)
        self.assertAlmostEqual((al.pos[1] - al.size[1] / 2) - (ar.pos[1] + ar.size[1] / 2), 0.080)  # the span
        self.assertAlmostEqual(al.pos[2] + al.size[2] / 2, L.TOP + L.ANVIL_TOP)
        for s in t.SUPPORTS:
            self.assertAlmostEqual(s.pos[2] + s.size[2] / 2, SUP_W)
            self.assertAlmostEqual(s.pos[0] + s.size[0] / 2, L.FACE_X)  # flush with the anvils' faces
        self.assertLess(t.STRIKER_MARK.pos[0], L.SUP_X0)  # on the striker's side, in front of the supports
        self.assertAlmostEqual(t.STRIKER_MARK.pos[1], L.MID_Y)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                lo = np.asarray(o.pos) - np.asarray(o.size) / 2
                hi = np.asarray(o.pos) + np.asarray(o.size) / 2
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)

    def test_specimen_geometry_matches_the_scorer(self):
        sp = self.task.SPECIMEN
        lo, hi = sp.local_bounds()
        np.testing.assert_allclose(lo, [-0.055, -0.010, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.055, 0.010, 0.0205], atol=1e-12)
        np.testing.assert_allclose((lo + hi) / 2, L.BOX_CENTER_LOCAL, atol=1e-12)
        np.testing.assert_allclose((hi - lo) / 2, L.BOX_HALF, atol=1e-12)

        def solid_at(p):
            return any(np.all(np.abs(np.asarray(p) - np.asarray(s.center)) < np.asarray(s.size) / 2)
                       for s in sp.shapes if tuple(s.size) != L.TICK_SIZE)

        # the slot is on the -y face at mid-length, 4 x 4 mm, full height; solid either side of it and behind it
        self.assertFalse(solid_at((0.0, -0.008, 0.010)))
        self.assertFalse(solid_at((0.0019, -0.0061, 0.019)))
        self.assertTrue(solid_at((0.0021, -0.008, 0.010)))
        self.assertTrue(solid_at((0.0, -0.0059, 0.010)))
        self.assertTrue(solid_at((0.0, 0.009, 0.010)))
        vol = sum(float(np.prod(s.size)) for s in sp.shapes)
        self.assertAlmostEqual(vol * 2700.0, 0.118, delta=0.001)  # aluminium
        com = sp.center_of_mass()
        self.assertLess(abs(com[0]), 1e-6)  # under the grip at mid-length
        rest = sp.rest_state()
        self.assertTrue(Tracker.at_start(rest))
        self.assertAlmostEqual(notch_angle_deg(rest), 90.0, places=6)
        self.assertLess(rest.low[1], L.MAT_Y - 0.0099)  # the notched face toward -y
        st = self.sim(SEATED)["specimen"]
        self.assertAlmostEqual(notch_offset(st), 0.0, places=9)
        np.testing.assert_allclose(st.low[2], SUP_W, atol=1e-9)
        self.assertAlmostEqual(st.high[0], L.FACE_X, places=9)

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.SPECIMEN.offline_box()[0]), quat_xyzw=quat(0.0),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.SPECIMEN.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.SP_START[0], L.SP_START[1], L.TOP + L.SP_START[2]], atol=1e-12)
        self.assertTrue(Tracker.at_start(st))

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        seated = self.sim(SEATED)
        for _ in range(WINDOW):
            self.feed_arms(FAR)
            t.success(seated)
        self.feed_arms(FAR)
        self.assertTrue(t.success(seated))
        self.assertEqual(t.score(seated), 1.0)
        self.assertEqual(t.score(seated), 1.0)  # score() does not advance it
        self.feed_arms(HOLDING)
        self.assertEqual(t.score(seated), 0.0)  # read with the arms of the step at hand
        self.feed_arms(FAR)
        n = tr.steps
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(FAR)
        t.success(seated)
        self.assertEqual(tr.steps, 1)
        t.success(seated)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(seated))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.rig.offline_reset()
                self.feed_arms(FAR)
                self.task.success(self.sim(SEATED))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "tier", "zero_reason", "tilt_deg", "heading_err_deg",
                          "notch_angle_deg", "notch_dir", "offset_mm", "gap_left_mm", "gap_right_mm", "onto_mm",
                          "end_x_mm", "underside_mm", "on_supports", "at_rest", "moved_mm", "hands_off",
                          "zone_gap_mm", "arms_unavailable", "sp_origin", "sp_quat", "end_a", "end_b", "source",
                          "right_tcp", "openings", "notch_y_mm", "notch_yaw_deg"):
                    self.assertIn(k, row)
                self.assertAlmostEqual(row["offset_mm"], 0.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(SEATED)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"right"})
        wps = plan["right"]
        by = {w.label: w for w in wps}
        # every close and open takes 1.5 s; so does every final descent onto the mat or the supports
        for prev, w in zip(wps, wps[1:]):
            if prev.finger != w.finger:
                self.assertGreaterEqual(w.duration, 1.5)
        self.assertGreaterEqual(by["slowly down beside it"].duration, 1.5)
        self.assertGreaterEqual(by["slowly onto the supports, pressed to the anvils"].duration, 1.5)
        grip, carry = by["grip the specimen"], by["carry it over the supports, notch turning to the anvils"]
        self.assertEqual(grip.yaw, 0.0)
        self.assertAlmostEqual(carry.yaw, math.pi / 2 + t.YAW_TRIM)  # a quarter turn left: notch from -y to +x
        # every waypoint over the tester uses the same trimmed yaw (no turning while it goes down or is pressed)
        after = wps[wps.index(carry):]
        self.assertTrue(all(w.yaw == t.YAW_SET for w in after))
        r0, r1 = _rot(quat(0.0)), _rot(quat(90.0))
        n_after = r1 @ r0.T @ (r0 @ np.array(L.NORMAL_LOCAL))
        np.testing.assert_allclose(n_after, [1.0, 0.0, 0.0], atol=1e-12)
        # gripped at its middle (the notch), so the notch's y is the TCP's y
        self.assertAlmostEqual(grip.pos[0], L.SP_START[0])
        self.assertAlmostEqual(grip.pos[1], L.SP_START[1])
        side, press = by["slowly down beside the anvils"], by["against the anvils"]
        setw, letgo = by["slowly onto the supports, pressed to the anvils"], by["let go of it"]
        self.assertAlmostEqual(L.FACE_X - (side.pos[0] + L.SP_W / 2), t.CLEAR)  # 4 mm in front of the anvils
        self.assertAlmostEqual(press.pos[0] + L.SP_W / 2 - L.FACE_X, t.PRESS)  # aimed 2.5 mm past them
        # far enough past them that the modelled 0.9 mm sag toward the striker still ends in contact, with margin;
        # not so far that the push (about 6 mm/N of lateral compliance) gets large
        self.assertGreaterEqual(press.pos[0] + L.SP_W / 2 - L.FACE_X, 0.002)
        self.assertLessEqual(press.pos[0] + L.SP_W / 2 - L.FACE_X, 0.004)
        self.assertGreaterEqual(L.FACE_X - (side.pos[0] + L.SP_W / 2), 0.003)  # comes down clear of the anvils
        self.assertEqual(press.pos[2], side.pos[2])  # moved across, not down
        self.assertAlmostEqual(setw.pos[0], press.pos[0])
        self.assertLess(abs(press.pos[1] - L.MID_Y), 0.001)
        self.assertEqual(letgo.pos, setw.pos)
        # heights: pad tips (13 mm below the TCP) 4 mm above the bar's underside at the pick and when set
        self.assertAlmostEqual(grip.pos[2] - 0.013 - (L.TOP + L.MAT_TOP), 0.004)
        self.assertAlmostEqual(setw.pos[2] - 0.013 - (SUP_W + t.DROP), 0.004)
        self.assertGreater(setw.pos[2] - 0.013, L.TOP + L.BASE_H + 0.010)  # pad tips well above the base
        # moved across with the bar's underside 8 mm above the supports (about 4.5 mm with the modelled sag)
        self.assertAlmostEqual(side.pos[2] - t.GRIP_UP - SUP_W, 0.008)
        # carried with its underside over the anvils' tops
        self.assertGreater(carry.pos[2] - t.GRIP_UP - (L.TOP + L.ANVIL_TOP), 0.025)
        # over the tester the fingers close along x: the +x finger (27 mm wide) stays in the 80 mm gap between the
        # anvils, and the open pads clear the anvils' faces
        self.assertLess(abs(press.pos[1]) + 0.0135, L.ANVIL_Y_IN - 0.020)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", set()),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_place_yaw_trim(self):
        """The trim turns the place yaw clockwise by about the GPU's counter-clockwise grip bias. With a bias anywhere
        from +0.45 to +1.95 degrees the bar, pushed until its forward end meets its anvil, has the other end within
        the 1.5 mm feeler; the notch still faces the anvils."""
        t = self.task
        self.assertAlmostEqual(math.degrees(t.YAW_TRIM), -1.2)

        def seated_at(skew_deg):  # turned skew_deg about the end that meets its anvil first
            return skewed(skew_deg, pivot_y=-L.SP_LEN / 2 if skew_deg > 0 else L.SP_LEN / 2)

        for bias in (0.45, 0.8, 1.2, 1.54, 1.95):
            st = seated_at(math.degrees(t.YAW_SET) + bias - 90.0)
            gl, gr = anvil_gaps(st)
            self.assertLess(min(gl, gr), 0.0003)  # touching at its end: widest 15 x sin(skew) at the inner edge
            self.assertLessEqual(max(gl, gr), L.ANVIL_GAP, msg=f"bias {bias}")
            self.assertEqual(pose_tier(st)[0], 1.0)
        for bias in (0.4, 2.0):  # just outside: the far end stands off more than 1.5 mm
            self.assertEqual(pose_tier(seated_at(math.degrees(t.YAW_SET) + bias - 90.0))[0], 0.75)
        # without the bias (offline) the bar is laid 1.2 degrees turned: the far anvil's gap is 2.3 mm (0.75)
        st = seated_at(math.degrees(t.YAW_TRIM))
        self.assertAlmostEqual(max(anvil_gaps(st)) * 1000, 110 * math.sin(math.radians(1.2)), places=6)
        self.assertEqual(pose_tier(st)[0], 0.75)
        self.assertLess(notch_angle_deg(st), 2.0)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("two supports", "both anvils", "away from the striker", "red strip", "centred", "let go",
                  "move clear"):
            self.assertIn(s, ins)


if __name__ == "__main__":
    unittest.main()
