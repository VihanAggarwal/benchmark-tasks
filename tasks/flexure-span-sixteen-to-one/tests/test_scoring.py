"""Offline tests for scoring.py and task.py: synthetic specimen and support states and gripper snapshots, no
simulator.  python3 -m unittest discover -s tests -v"""

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
    all_valid,
    arms_valid,
    crossing,
    on_rollers,
    pose_tier,
    roller_centre,
    roller_heading_err_deg,
    sp_ends,
    sp_point_distance,
    sp_tilt_deg,
    span_and_offset,
    standing,
    sup_point_distance,
    sup_tilt_deg,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
X = L.BASE_X  # the rollers' and the specimen's x in a set-up fixture


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


def bar(centre, yaw=0.0, pitch=0.0, roll=0.0, vel=(0.0, 0.0, 0.0)):
    """A specimen state (only what the scorer reads). yaw 0: its length along y."""
    c = np.asarray(centre, dtype=float)
    return SimpleNamespace(pos=c, quat_xyzw=quat(yaw, pitch, roll), lin_vel=np.asarray(vel, dtype=float),
                           ang_vel=np.zeros(3), low=c - 0.1, high=c + 0.1)


def sup(y, x=X, z=L.BASE_TOP_W, yaw=0.0, pitch=0.0, roll=0.0):
    """A support block's state as mechkit reports it: origin at the middle of its underside. yaw 0: roller along x."""
    o = np.array([x, y, z], dtype=float)
    q = quat(yaw, pitch, roll)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=o + _rot(q) @ np.array(L.SUP_BOX_CENTER),
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3))


def setup(span=L.SPAN_NOMINAL, mid=0.0, off=0.0, skew=0.0, under=0.0, sp_x=X, lean=0.0, flip=False,
          sup_kw=({}, {})):
    """Both supports on the base, rollers ``span`` apart about y = ``mid``; the specimen flat across them, its middle
    ``off`` along y from the span's middle, turned ``skew`` degrees from y, its underside ``under`` above the rollers'
    tops, leaning ``lean`` degrees about its length (onto an edge). flip: turned end for end and upside down (the
    same bar)."""
    sl = sup(mid + span / 2, **sup_kw[0])
    sr = sup(mid - span / 2, **sup_kw[1])
    c = (sp_x, mid + off, L.ROLLER_TOP_W + L.SP_DEPTH / 2 + under)
    sp = bar(c, yaw=skew + (180.0 if flip else 0.0), pitch=lean, roll=(180.0 if flip else 0.0))
    return sp, (sl, sr)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


PARKED_LEFT = arm((0.20, 0.23, L.TOP + 0.16), 40.0 / 93.9)
PARKED_RIGHT = arm((0.20, -0.26, L.TOP + 0.16), 50.0 / 93.9)
FAR = {"left": PARKED_LEFT, "right": PARKED_RIGHT}
# the left gripper closed on the set bar as the demo holds it (TCP 16 mm above its underside, 12 mm commanded)
HOLDING = {"left": arm((X, 0.0, L.ROLLER_TOP_W + 0.016), 12.0 / 93.9), "right": PARKED_RIGHT}

START = (bar((L.SP_START[0], L.SP_START[1], L.TOP + L.SP_START[2])),
         (sup(L.SUP_START["left"][1]), sup(L.SUP_START["right"][1])))
SET = setup()


def feed(tr, scenes, arms=FAR):
    r = None
    for sp, sups in scenes:
        r = tr.update(sp, sups, arms)
    return r


def settled(scene, arms=FAR, n=WINDOW + 1):
    """A tracker that has seen ``scene`` at rest for ``n`` steps; returns (tracker, last result)."""
    tr = Tracker(HZ)
    return tr, feed(tr, [scene] * n, arms)


def scored(scene, arms=FAR):
    return settled(scene, arms)[1].score


def glide(a, b, n):
    """n scenes moving every part from scene a's place to scene b's (b's orientations)."""
    (spa, (la, ra)), (spb, (lb, rb)) = a, b
    out = []
    for i in range(1, n + 1):
        f = i / n
        sp = bar(np.asarray(spa.pos) + (np.asarray(spb.pos) - np.asarray(spa.pos)) * f)
        sp.quat_xyzw = spb.quat_xyzw
        sups = []
        for s0, s1 in ((la, lb), (ra, rb)):
            o = np.asarray(s0.origin) + (np.asarray(s1.origin) - np.asarray(s0.origin)) * f
            s = sup(o[1], x=o[0], z=o[2])
            s.quat_xyzw = s1.quat_xyzw
            sups.append(s)
        out.append((sp, tuple(sups)))
    return out


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        sp, sups = START
        self.assertTrue(all_valid(sp, sups))
        self.assertTrue(Tracker.at_start(sp))
        self.assertTrue(all(Tracker.sup_at_start(s) for s in sups))
        s, e = span_and_offset(sups)
        self.assertAlmostEqual(s, 0.064, places=12)  # set for a 4 mm bar
        self.assertAlmostEqual(e, 0.0, places=12)
        a, b = sp_ends(sp)
        np.testing.assert_allclose(a, [L.MAT_X, L.MAT_Y - 0.105, L.TOP + L.MAT_TOP + 0.005], atol=1e-12)
        np.testing.assert_allclose(b, [L.MAT_X, L.MAT_Y + 0.105, L.TOP + L.MAT_TOP + 0.005], atol=1e-12)
        tier, reason, _ = pose_tier(sp, sups)
        self.assertEqual((tier, reason), (0.0, "specimen_not_on_both_rollers"))
        self.assertFalse(Tracker.at_start(SET[0]))

    def test_set_up_measures(self):
        sp, sups = SET
        for s in sups:
            ok, why, d = standing(s)
            self.assertTrue(ok, why)
            self.assertAlmostEqual(d["height_mm"], 0.0, places=9)
            np.testing.assert_allclose(roller_centre(s)[2], L.BASE_TOP_W + L.ROLLER_Z, atol=1e-12)
        s, e = span_and_offset(sups)
        self.assertAlmostEqual(s, 0.160, places=12)
        self.assertAlmostEqual(e, 0.0, places=12)
        ok, why, d, raw = on_rollers(sp, sups)
        self.assertTrue(ok, why)
        self.assertAlmostEqual(raw[0], 0.0, places=12)
        self.assertAlmostEqual(raw[1], 0.0, places=9)
        np.testing.assert_allclose(d["in_from_end_mm"], [25.0, 25.0], atol=1e-9)  # (210 - 160) / 2 overhang
        np.testing.assert_allclose(d["underside_mm"], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(d["along_roller_mm"], [0.0, 0.0], atol=1e-9)

    def test_span_and_offset_read_as_two_rule_readings_would(self):
        sups = (sup(0.083), sup(-0.079))
        s, e = span_and_offset(sups)
        self.assertAlmostEqual(s, 0.162, places=12)
        # readings from the red line to each roller's axis: 83 and 79 mm; half the difference is 2 mm
        self.assertAlmostEqual(e, abs(83 - 79) / 2 / 1000, places=12)
        self.assertEqual(span_and_offset(sups[::-1]), (s, e))  # either order in

    def test_specimen_offset_is_half_the_difference_of_the_overhangs(self):
        sp, sups = setup(off=0.003)
        ok, _, d, raw = on_rollers(sp, sups)
        self.assertTrue(ok)
        # overhangs past each roller's axis: 25 + 3 = 28 mm on the left, 22 on the right
        np.testing.assert_allclose(d["in_from_end_mm"], [28.0, 22.0], atol=1e-9)  # at the left, right roller
        self.assertAlmostEqual(raw[0], (28 - 22) / 2 / 1000, places=12)

    def test_skew_reads_as_angle_from_square(self):
        sp, sups = setup(skew=2.0)
        _, _, d, raw = on_rollers(sp, sups)
        np.testing.assert_allclose(d["square_err_deg"], [2.0, 2.0], atol=1e-9)
        self.assertAlmostEqual(raw[0], 0.0, places=12)  # still centred: symmetric about the span's middle
        # a roller turned 2 degrees the same way leaves the bar square to that roller only
        sp, sups = setup(skew=2.0, sup_kw=({"yaw": 2.0}, {}))
        _, _, d, _ = on_rollers(sp, sups)
        np.testing.assert_allclose(d["square_err_deg"], [0.0, 2.0], atol=1e-9)

    def test_the_same_bar_and_block_either_way_round(self):
        sp, sups = setup(flip=True)
        self.assertAlmostEqual(sp_tilt_deg(sp), 0.0, places=6)
        ok, _, d, raw = on_rollers(sp, sups)
        self.assertTrue(ok)
        np.testing.assert_allclose(d["underside_mm"], [0.0, 0.0], atol=1e-9)
        sl, sr = sup(0.08, yaw=180.0), sup(-0.08, yaw=-180.0)
        self.assertAlmostEqual(roller_heading_err_deg(sl), 0.0, places=6)
        self.assertEqual(scored((SET[0], (sl, sr))), 1.0)
        self.assertAlmostEqual(sup_tilt_deg(sup(0.08, roll=180.0)), 180.0, places=6)

    def test_crossing_underside_and_along_roller(self):
        sp, sups = setup(under=0.0015, sp_x=X + 0.007)
        c = crossing(sp, sups[0])
        self.assertAlmostEqual(c["gap"], 0.0015, places=12)
        self.assertAlmostEqual(c["along"], 0.007, places=12)
        self.assertAlmostEqual(c["tau"], 0.080, places=12)
        self.assertIsNone(crossing(bar((X, 0.0, 0.8), yaw=90.0), sups[0]))  # parallel to the roller

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 1), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(arm((0, 0, 1), 0.0)), math.hypot(0.008, 0.0135))
        sp, sups = SET
        self.assertLess(zone_gap(HOLDING["left"], sp, sups), 0.0)
        self.assertGreater(zone_gap(PARKED_LEFT, sp, sups), 0.05)
        self.assertGreater(zone_gap(PARKED_RIGHT, sp, sups), 0.05)
        self.assertEqual(sp_point_distance(np.asarray(sp.pos), sp), 0.0)
        self.assertEqual(sup_point_distance(roller_centre(sups[0]), sups[0]), 0.0)
        self.assertAlmostEqual(sup_point_distance(roller_centre(sups[0]) + [0, 0, 0.010], sups[0]), 0.005)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(scored(START), 0.0)

    def test_set_up_scores_one(self):
        tr, r = settled(SET)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["at_rest"])
        self.assertTrue(r.details["hands_off"])
        self.assertEqual(r.details["zero_reason"], "")
        self.assertEqual((r.details["span_tier"], r.details["sp_tier"]), (0.5, 0.5))

    def test_span_edges(self):
        for err, want in ((0.0049, 1.0), (-0.0049, 1.0), (0.005, 1.0), (0.0051, 0.75), (-0.0051, 0.75),
                          (0.0099, 0.75), (-0.0099, 0.75), (0.0101, 0.0), (-0.0101, 0.0), (0.096, 0.0)):
            with self.subTest(err=err):
                self.assertEqual(scored(setup(span=L.SPAN_NOMINAL + err)), want)

    def test_span_centring_edges(self):
        for mid, want in ((0.0049, 1.0), (-0.0049, 1.0), (0.0051, 0.75), (-0.0051, 0.75), (0.0079, 0.75),
                          (-0.0079, 0.75), (0.0081, 0.0), (-0.0081, 0.0), (0.020, 0.0)):
            with self.subTest(mid=mid):
                self.assertEqual(scored(setup(mid=mid)), want)

    def test_specimen_centring_edges(self):
        for off, want in ((0.0049, 1.0), (-0.0049, 1.0), (0.0051, 0.75), (-0.0051, 0.75), (0.0099, 0.75),
                          (-0.0099, 0.75), (0.0101, 0.5), (-0.0101, 0.5), (0.019, 0.5), (-0.019, 0.5)):
            with self.subTest(off=off):
                self.assertEqual(scored(setup(off=off)), want)

    def test_square_edges(self):
        for skew, want in ((2.9, 1.0), (-2.9, 1.0), (3.1, 0.75), (-3.1, 0.75), (9.9, 0.75), (-9.9, 0.75),
                           (10.1, 0.0)):
            with self.subTest(skew=skew):
                self.assertEqual(scored(setup(skew=skew)), want)

    def test_partial_credit(self):
        cases = (
            (dict(off=0.015), 0.5, (0.5, 0.0)),  # span set, specimen resting 15 mm off centre
            (dict(span=0.167, off=0.007), 0.5, (0.25, 0.25)),  # both within the loose limits
            (dict(span=0.153, off=0.015), 0.25, (0.25, 0.0)),  # span in the standards' band, specimen far off
            (dict(span=0.167), 0.75, (0.25, 0.5)),  # 16.7 x the depth, specimen centred and square
            (dict(mid=0.006, off=0.0), 0.75, (0.25, 0.5)),  # span 6 mm off the line, specimen on the span
            (dict(off=0.004, skew=4.0), 0.75, (0.5, 0.25)),  # centred but 4 degrees out of square
        )
        for kw, want, tiers in cases:
            with self.subTest(kw=kw):
                tr, r = settled(setup(**kw))
                self.assertEqual(r.score, want)
                self.assertEqual((r.details["span_tier"], r.details["sp_tier"]), tiers)
                self.assertFalse(r.success)

    def test_every_tier_is_reachable_and_each_pose_has_one_score(self):
        got = {}
        for err in np.arange(-0.012, 0.0121, 0.001):
            for mid in (0.0, 0.006, 0.009):
                for off in (0.0, 0.007, 0.013, 0.021):
                    sc = scored(setup(span=L.SPAN_NOMINAL + err, mid=mid, off=off))
                    got[sc] = got.get(sc, 0) + 1
        self.assertEqual(set(got), {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_small_tilts_count(self):
        self.assertEqual(scored(setup(lean=4.5)), 1.0)  # leaning about its length, still on both rollers
        self.assertEqual(scored(setup(lean=-4.5)), 1.0)
        # both rollers 4.5 degrees off x, the bar turned with them: square to both
        self.assertEqual(scored(setup(skew=4.5, sup_kw=({"yaw": 4.5}, {"yaw": 4.5}))), 1.0)
        # rollers turned opposite ways (9 degrees out of parallel): the bar is square to neither within 3 degrees
        self.assertEqual(scored(setup(sup_kw=({"yaw": 4.5}, {"yaw": -4.5}))), 0.75)
        # a support leaning 2 degrees toward the midline: its roller moves 1 mm in, 0.03 mm down
        sp, (sl, sr) = SET
        lean = sup(0.08, roll=-2.0)
        self.assertTrue(standing(lean)[0])
        self.assertEqual(scored((sp, (lean, sr))), 1.0)

    def test_underside_within_2_mm(self):
        self.assertEqual(scored(setup(under=0.0019)), 1.0)
        self.assertEqual(scored(setup(under=-0.0005)), 1.0)  # solver penetration

    def test_along_the_rollers(self):
        self.assertEqual(scored(setup(sp_x=X + 0.0149)), 1.0)  # its full width still on both 50 mm rollers
        self.assertEqual(scored(setup(sp_x=X - 0.0149)), 1.0)

    def test_score_is_idempotent(self):
        tr, _ = settled(SET)
        n, h = tr.steps, len(tr.history)
        for _ in range(3):
            self.assertEqual(tr.result(*SET, FAR).score, 1.0)
        self.assertEqual((tr.steps, len(tr.history)), (n, h))


class ZeroConditions(unittest.TestCase):
    def assertZero(self, scene, reason, arms=FAR):
        tr, r = settled(scene, arms)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], reason)

    def test_support_not_standing_on_the_base(self):
        sp, (sl, sr) = SET
        for bad in (sup(0.08, roll=5.5), sup(0.08, pitch=-6.0), sup(0.08, roll=180.0), sup(0.08, roll=90.0),
                    sup(0.08, z=L.BASE_TOP_W + 0.0025), sup(0.08, z=L.TOP), sup(0.08, x=L.BASE_X1 + 0.002),
                    sup(L.BASE_HALF_Y + 0.003)):
            with self.subTest(bad=bad.origin):
                self.assertZero((sp, (bad, sr)), "support_not_standing_on_base")
                self.assertZero((sp, (sl, bad)), "support_not_standing_on_base")

    def test_roller_not_parallel_to_the_centre_line(self):
        sp, (sl, sr) = SET
        self.assertZero((sp, (sup(0.08, yaw=5.5), sr)), "roller_not_parallel_to_centre_line")
        self.assertZero((sp, (sl, sup(-0.08, yaw=90.0))), "roller_not_parallel_to_centre_line")

    def test_specimen_not_flat(self):
        self.assertZero(setup(lean=5.5), "specimen_not_flat")
        self.assertZero(setup(lean=90.0), "specimen_not_flat")  # on its edge: depth 20 mm, not 10
        sp, sups = SET
        self.assertZero((bar(sp.pos, roll=5.5), sups), "specimen_not_flat")  # tipped along its length
        self.assertZero((bar(sp.pos, roll=90.0), sups), "specimen_not_flat")  # standing on an end

    def test_specimen_not_across_the_rollers(self):
        self.assertZero(setup(skew=10.5), "specimen_not_across_the_rollers")
        self.assertZero(setup(skew=90.0), "specimen_not_across_the_rollers")  # along the rollers

    def test_specimen_not_on_both_rollers(self):
        self.assertZero(setup(under=0.0021), "specimen_not_on_both_rollers")  # held or propped up
        self.assertZero(setup(under=-0.035), "specimen_not_on_both_rollers")  # down on the base between them
        self.assertZero(setup(sp_x=X + 0.0151), "specimen_not_on_both_rollers")  # partly off a roller's end
        self.assertZero(setup(off=0.0201), "specimen_not_on_both_rollers")  # a roller 4.9 mm from its end
        self.assertZero(setup(off=0.030), "specimen_not_on_both_rollers")  # one end short of a roller
        self.assertZero(setup(span=0.064, off=0.110), "specimen_not_on_both_rollers")  # both rollers to one side

    def test_both_rollers_on_one_side_of_its_middle(self):
        # supports left 64 mm apart, the bar 60 mm off to one side: it crosses both rollers well in from its ends,
        # but its middle (and weight) is beyond them, so it would tip off: not resting on them
        sp, sups = setup(span=0.064, off=0.060)
        ok, why, d, raw = on_rollers(sp, sups)
        self.assertEqual((ok, why), (False, "specimen_not_on_both_rollers"))
        self.assertGreaterEqual(min(d["in_from_end_mm"]), 5.0)
        self.assertZero((sp, sups), "specimen_not_on_both_rollers")

    def test_a_roller_5_mm_in_from_the_end_still_counts(self):
        self.assertEqual(scored(setup(off=0.0199)), 0.5)  # resting; the span set; the bar 20 mm off centre

    def test_span_out_of_band_or_off_centre(self):
        self.assertZero(setup(span=0.1702), "span_out_of_band")
        self.assertZero(setup(span=0.1498), "span_out_of_band")
        self.assertZero(setup(mid=0.0085), "span_off_centre")
        self.assertZero(setup(span=0.112, mid=0.024), "span_out_of_band")  # one support moved, one left

    def test_held_or_touched(self):
        self.assertZero(SET, "gripper_on_or_near_a_part", HOLDING)
        over = {"left": arm((X, 0.0, L.ROLLER_TOP_W + 0.045), 0.43), "right": PARKED_RIGHT}  # open, just above
        self.assertZero(SET, "gripper_on_or_near_a_part", over)

    def test_hands_off_is_20_mm(self):
        r43 = math.hypot(0.43 * 0.0939 / 2 + 0.008, 0.0135)
        top = L.ROLLER_TOP_W + L.SP_DEPTH
        for clear, want in ((0.019, 0.0), (0.021, 1.0)):
            tips_z = top + r43 + clear
            a = {"left": arm((X, 0.0, tips_z + 0.013), 0.43), "right": PARKED_RIGHT}
            with self.subTest(clear=clear):
                self.assertEqual(scored(SET, a), want)

    def test_a_gripper_on_a_support_counts_too(self):
        on_block = {"left": PARKED_LEFT, "right": arm((X, -0.08, L.BASE_TOP_W + 0.019), 22.0 / 93.9)}
        self.assertZero(SET, "gripper_on_or_near_a_part", on_block)
        # open, low beside the right support's outer end: 9 mm from the support, 24 mm from the bar
        beside = {"left": PARKED_LEFT, "right": arm((L.BASE_X + 0.065, -0.08, L.BASE_TOP_W + 0.019), 0.43)}
        sp, sups = SET
        self.assertGreater(zone_gap(beside["right"], sp, ()), L.HANDS_OFF)
        self.assertZero(SET, "gripper_on_or_near_a_part", beside)

    def test_still_moving(self):
        tr = Tracker(HZ)
        r = feed(tr, glide(setup(off=0.004), SET, 4))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "still_moving")


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [SET] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(r.details["moved_mm"])
        self.assertEqual(tr.update(*SET, FAR).score, 1.0)

    def test_comes_to_rest(self):
        tr = Tracker(HZ)
        feed(tr, glide(setup(off=0.010), SET, 2))
        for _ in range(WINDOW - 1):
            self.assertEqual(tr.update(*SET, FAR).score, 0.0)
        self.assertEqual(tr.update(*SET, FAR).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        self.assertEqual(feed(tr, [setup(off=(0.0019 if i % 2 else 0.0)) for i in range(WINDOW + 5)]).score, 1.0)
        tr = Tracker(HZ)
        self.assertEqual(feed(tr, [setup(off=(0.0021 if i % 2 else 0.0)) for i in range(WINDOW + 5)]).score, 0.0)

    def test_a_moving_support_counts(self):
        sp, (sl, sr) = SET
        scenes = [(sp, (sup(0.08 + (0.0025 if i % 2 else 0.0)), sr)) for i in range(WINDOW + 5)]
        self.assertEqual(feed(Tracker(HZ), scenes).details["zero_reason"], "still_moving")

    def test_partial_tiers_need_rest_too(self):
        tr = Tracker(HZ)
        r = feed(tr, glide(setup(off=0.015), setup(off=0.0185), 3))
        self.assertEqual((r.score, r.details["tier"]), (0.0, 0.5))


class Exploits(unittest.TestCase):
    def test_supports_left_where_they_start(self):
        # the bar laid centred across the 64 mm span the fixture was left at: resting, but the span is wrong
        sp, sups = setup(span=0.064)
        self.assertTrue(on_rollers(sp, sups)[0])
        self.assertEqual(scored((sp, sups)), 0.0)

    def test_held_in_place_on_the_rollers(self):
        self.assertEqual(scored(SET, HOLDING), 0.0)

    def test_pressed_down_by_an_open_gripper(self):
        press = {"left": arm((X, 0.05, L.ROLLER_TOP_W + L.SP_DEPTH + 0.013), 0.43), "right": PARKED_RIGHT}
        self.assertEqual(scored(SET, press), 0.0)

    def test_on_its_edge_across_the_rollers(self):
        sp, sups = SET
        edge = bar((X, 0.0, L.ROLLER_TOP_W + L.SP_W / 2), pitch=90.0)
        self.assertEqual(scored((edge, sups)), 0.0)

    def test_laid_on_the_base_between_the_supports(self):
        sp, sups = SET
        self.assertEqual(scored((bar((X, 0.0, L.BASE_TOP_W + L.SP_DEPTH / 2)), sups)), 0.0)

    def test_propped_across_a_roller_and_the_base(self):
        # one end on the left roller, the other down on the base: pitched about 13 degrees
        sp, sups = SET
        a = math.degrees(math.atan2(L.ROLLER_TOP, 0.150))
        self.assertEqual(scored((bar((X, 0.03, L.BASE_TOP_W + 0.020), roll=a), sups)), 0.0)

    def test_supports_on_their_sides_or_turned_across(self):
        sp, _ = SET
        tipped = (sup(0.08, roll=90.0, z=L.BASE_TOP_W + 0.015), sup(-0.08, roll=-90.0, z=L.BASE_TOP_W + 0.015))
        self.assertEqual(scored((sp, tipped)), 0.0)
        across = (sup(0.08, yaw=90.0), sup(-0.08, yaw=90.0))  # rollers along y: the bar along them, not across
        self.assertEqual(scored((sp, across)), 0.0)

    def test_supports_set_on_the_table_beside_the_base(self):
        sp, _ = SET
        table = (sup(0.08, x=L.BASE_X0 - 0.030, z=L.TOP), sup(-0.08, x=L.BASE_X0 - 0.030, z=L.TOP))
        moved = bar((L.BASE_X0 - 0.030, 0.0, L.TOP + L.ROLLER_TOP + L.SP_DEPTH / 2))
        self.assertEqual(scored((moved, table)), 0.0)

    def test_one_support_stacked_on_the_other(self):
        sp, (sl, sr) = SET
        self.assertEqual(scored((sp, (sup(-0.08, z=L.BASE_TOP_W + L.ROLLER_TOP), sr))), 0.0)

    def test_arms_unreadable_cannot_be_used_to_skip_the_rest(self):
        # unreadable arms only waive the hands rule: the fixture left at 64 mm still scores 0
        self.assertEqual(scored(setup(span=0.064), None), 0.0)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr, r = settled(SET, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)
        self.assertIsNone(r.details["zone_gap_mm"])

    def test_non_finite_arms_count_as_unavailable(self):
        bad = {"left": PARKED_LEFT, "right": arm((math.nan, 0.0, 1.0))}
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": PARKED_LEFT}))
        self.assertFalse(arms_valid(None))
        self.assertFalse(arms_valid({"left": PARKED_LEFT, "right": SimpleNamespace(tcp=np.zeros(3))}))
        tr, r = settled(SET, bad)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)

    def test_hands_off_applies_when_the_arms_come_back(self):
        tr, _ = settled(SET, None)
        self.assertEqual(tr.update(*SET, HOLDING).score, 0.0)
        self.assertEqual(tr.update(*SET, FAR).score, 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(SET)
        n, hist = tr.steps, [tuple(np.array(p) for p in h) for h in tr.history]
        sp, (sl, sr) = SET
        bad_sp = [
            bar((math.nan, 0.0, 0.8)),
            bar((0.31, math.inf, 0.8)),
            SimpleNamespace(pos=np.array([0.31, 0.0, 0.8]), quat_xyzw=np.array([math.nan, 0, 0, 1])),
            SimpleNamespace(pos=np.array([0.31, 0.0, 0.8]), quat_xyzw=np.zeros(4)),
            SimpleNamespace(quat_xyzw=quat()),  # no pos
            SimpleNamespace(pos=np.zeros(2), quat_xyzw=quat()),  # malformed
            None,
        ]
        bad_sup = [
            SimpleNamespace(origin=np.array([math.nan, 0.08, 0.76]), quat_xyzw=quat(), pos=np.zeros(3)),
            SimpleNamespace(origin=np.array([0.31, 0.08, 0.76]), quat_xyzw=np.zeros(4), pos=np.zeros(3)),
            SimpleNamespace(quat_xyzw=quat(), pos=np.zeros(3)),  # no origin: not built as an assembly
            None,
        ]
        scenes = [(b, (sl, sr)) for b in bad_sp] + [(sp, (b, sr)) for b in bad_sup] + [(sp, (sl,)), (sp, None)]
        for scene in scenes:
            with self.subTest(scene=scene):
                self.assertFalse(all_valid(*scene))
                r = tr.update(*scene, FAR)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.steps, n + len(scenes))
        self.assertEqual(len(tr.history), len(hist))
        for now, before in zip(tr.history, hist):
            for a, b in zip(now, before):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(tr.update(*SET, FAR).score, 1.0)  # and it carries on

    def test_unnormalised_quaternions(self):
        sp, (sl, sr) = SET
        sp2 = bar(sp.pos)
        sp2.quat_xyzw = 3.0 * np.asarray(sp.quat_xyzw)
        sl2 = sup(0.08)
        sl2.quat_xyzw = 0.7 * np.asarray(sl.quat_xyzw)
        self.assertEqual(scored((sp2, (sl2, sr))), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(11)
        tr = Tracker(HZ)
        for _ in range(500):
            sp = bar(rng.uniform([-0.2, -0.5, 0.6], [0.8, 0.5, 1.2]))
            sp.quat_xyzw = rng.normal(size=4)
            sups = []
            for _ in range(2):
                s = sup(rng.uniform(-0.3, 0.3), x=rng.uniform(0.1, 0.5), z=rng.uniform(0.7, 0.9))
                s.quat_xyzw = rng.normal(size=4)
                sups.append(s)
            a = {"left": arm(rng.uniform(-1, 1, 3), rng.uniform(-1, 2)), "right": arm(rng.uniform(-1, 1, 3))}
            r = tr.update(sp, tuple(sups), a if rng.random() < 0.8 else None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(SET)
        self.assertEqual(r.score, 1.0)
        r = tr.update(*START, FAR)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.history), 1)
        self.assertEqual(r.score, 0.0)

    def test_a_support_teleported_back_alone_resets(self):
        # a trial that never moved the specimen: only the supports jump back to their marks
        sp0, sups0 = START
        tr, _ = settled((sp0, (sup(0.08), sup(-0.08))))
        tr.update(sp0, sups0, FAR)
        self.assertEqual(tr.steps, 1)

    def test_reset_after_a_full_trial_then_a_new_trial(self):
        tr, _ = settled(SET)
        tr.update(*START, FAR)
        self.assertEqual(feed(tr, [SET] * WINDOW).score, 0.0)  # not yet 0.5 s of the new trial seen
        self.assertEqual(tr.update(*SET, FAR).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(SET)
        tr.update(bar((L.SP_START[0] + 0.03, L.SP_START[1] - 0.03, L.TOP + L.SP_START[2])), START[1], FAR)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(SET)
        feed(tr, glide(SET, START, 40))
        self.assertEqual(tr.steps, WINDOW + 1 + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(SET)
        tr.update(*setup(off=0.0045, mid=0.0), FAR)
        far = setup()
        far[1][0].origin = far[1][0].origin + np.array([0.0, 0.09, 0.0])  # a support thrown 9 cm, not to a mark
        tr.update(*far, FAR)
        self.assertEqual(tr.steps, WINDOW + 3)

    def test_explicit_reset(self):
        tr, _ = settled(SET, None)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.arms_unavailable), (0, 0, 0))
        self.assertIsNone(tr.prev)


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

    def sim(self, scene):
        """A synthetic scene turned into the states the simulator would report (mechkit AssemblyStates for the
        supports)."""
        sp, (sl, sr) = scene
        t = self.task
        return {"specimen": sp,
                "support_left": self.mk.state_from_poses(t.SUPPORTS[0], (*sl.origin, *sl.quat_xyzw), source="sim"),
                "support_right": self.mk.state_from_poses(t.SUPPORTS[1], (*sr.origin, *sr.quat_xyzw), source="sim")}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], ["support_left", "support_right",
                                                                              "specimen"])
        for o in t.OBJECTS:
            if o.mass is not None:
                self.assertLessEqual(o.mass, 0.30)
        self.assertAlmostEqual(t.SPECIMEN.mass, 0.057)
        self.assertAlmostEqual(L.SP_W * L.SP_LEN * L.SP_DEPTH * 1360.0, 0.057, delta=0.001)  # PA66-GF30
        self.assertEqual(tuple(t.SPECIMEN.size), (0.020, 0.210, 0.010))
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        # the base and its markings
        self.assertAlmostEqual(t.BASE.pos[2] + t.BASE.size[2] / 2, L.BASE_TOP_W)
        self.assertAlmostEqual(t.CENTRE_LINE.pos[1], L.MID_Y)
        self.assertAlmostEqual(t.CENTRE_LINE.pos[2] - t.CENTRE_LINE.size[2] / 2, L.BASE_TOP_W)
        self.assertEqual(len(t.TICKS), 28)
        ys = sorted(round(k.pos[1] * 1000) for k in t.TICKS)
        self.assertEqual(ys, sorted([10 * i for i in range(1, 15)] + [-10 * i for i in range(1, 15)]))
        long_ = sorted(round(k.pos[1] * 1000) for k in t.TICKS if k.size[0] > 0.010)
        self.assertEqual(long_, [-100, -50, 50, 100])
        for k in t.TICKS:  # on the near edge, at least 5 mm in front of where the supports stand
            self.assertLess(k.pos[0] + k.size[0] / 2, L.BASE_X - L.BLOCK_SIZE[0] / 2 - 0.005)
            self.assertGreater(k.pos[0] - k.size[0] / 2, L.BASE_X0)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                lo = np.asarray(o.pos) - np.asarray(o.size) / 2
                hi = np.asarray(o.pos) + np.asarray(o.size) / 2
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)

    def test_support_geometry_matches_the_scorer(self):
        t = self.task
        for s, side in zip(t.SUPPORTS, ("left", "right")):
            lo, hi = s.local_bounds()
            np.testing.assert_allclose(lo, [-0.025, -0.015, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [0.025, 0.015, L.ROLLER_TOP], atol=1e-12)
            np.testing.assert_allclose((lo + hi) / 2, L.SUP_BOX_CENTER, atol=1e-12)
            np.testing.assert_allclose((hi - lo) / 2, L.SUP_HALF, atol=1e-12)
            roller = [sh for sh in s.shapes if sh.kind == "cylinder"]
            self.assertEqual(len(roller), 1)
            self.assertEqual(roller[0].axis, "X")
            self.assertAlmostEqual(roller[0].radius, L.ROLLER_R)
            self.assertAlmostEqual(roller[0].height, L.ROLLER_LEN)
            np.testing.assert_allclose(roller[0].center, [0.0, 0.0, L.ROLLER_Z], atol=1e-12)
            self.assertAlmostEqual(s.mass, L.SUP_MASS)
            rest = s.rest_state()
            self.assertTrue(Tracker.sup_at_start(rest))
            self.assertTrue(standing(rest)[0])
            np.testing.assert_allclose(roller_centre(rest), [L.BASE_X, L.SUP_START[side][1],
                                                             L.BASE_TOP_W + L.ROLLER_Z], atol=1e-12)
            np.testing.assert_allclose(rest.low[2], L.BASE_TOP_W, atol=1e-12)
            np.testing.assert_allclose(rest.high[2], L.ROLLER_TOP_W, atol=1e-12)
        sp0 = SimpleNamespace(pos=np.array(t.SPECIMEN.pos), quat_xyzw=quat(), lin_vel=np.zeros(3))
        self.assertTrue(Tracker.at_start(sp0))
        self.assertAlmostEqual(t.SPECIMEN.pos[2] - L.SP_DEPTH / 2, L.TOP + L.MAT_TOP)  # lying on the mat

    def test_offline_state_path(self):
        s = self.task.SUPPORTS[0]
        box = SimpleNamespace(pos=np.array(s.offline_box()[0]), quat_xyzw=quat(), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = s.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.BASE_X, L.SUP_START["left"][1], L.BASE_TOP_W], atol=1e-12)
        self.assertTrue(Tracker.sup_at_start(st))

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        good = self.sim(SET)
        for _ in range(WINDOW):
            self.feed_arms(FAR)
            t.success(good)
        self.feed_arms(FAR)
        self.assertTrue(t.success(good))
        self.assertEqual(t.score(good), 1.0)
        self.assertEqual(t.score(good), 1.0)  # score() does not advance it
        self.feed_arms(HOLDING)
        self.assertEqual(t.score(good), 0.0)  # read with the arms of the step at hand
        self.feed_arms(FAR)
        n = tr.steps
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(FAR)
        t.success(good)
        self.assertEqual(tr.steps, 1)
        t.success(good)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(good))
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
                self.task.success(self.sim(SET))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "tier", "zero_reason", "span_mm", "span_error_mm",
                          "centre_off_mm", "roller_y_mm", "roller_x_mm", "span_tier", "sp_tier", "sp_offset_mm",
                          "sp_tilt_deg", "sp_heading_err_deg", "square_err_deg", "along_roller_mm",
                          "in_from_end_mm", "underside_mm", "left_tilt_deg", "left_roller_heading_deg",
                          "left_height_mm", "right_inside_base", "at_rest", "moved_mm", "hands_off",
                          "zone_gap_mm", "arms_unavailable", "sp_pos", "sp_quat", "end_a", "end_b", "sources",
                          "support_left_origin", "roller_right", "left_tcp", "openings"):
                    self.assertIn(k, row)
                self.assertAlmostEqual(row["span_mm"], 160.0)
                self.assertEqual(row["sources"], ["sim", "sim"])
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(SET)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})

        def clock(wps):
            out, s = {}, 0.0
            for w in wps:
                s += w.duration
                out.setdefault(w.label, []).append((s - w.duration, s, w))
            return out

        cl, cr = clock(plan["left"]), clock(plan["right"])
        for wps in plan.values():
            # every close and open takes 1.5 s; so does every final descent onto the base, the mat or the rollers
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)
                self.assertGreater(w.duration, 0.0)
        for c in (cl, cr):
            self.assertGreaterEqual(c["slowly down round the block"][0][2].duration, 1.5)
            self.assertGreaterEqual(c["slowly set it on the base"][0][2].duration, 1.5)
        self.assertGreaterEqual(cl["slowly down round the bar"][0][2].duration, 1.5)
        self.assertGreaterEqual(cl["slowly lower it onto the rollers"][0][2].duration, 2.0)
        # supports: picked where they stand, set 80 mm (less the trim) out, gripped across the block (yaw 0)
        for c, side, sign in ((cl, "left", 1.0), (cr, "right", -1.0)):
            grip = c["grip the support"][0][2]
            setw = c["slowly set it on the base"][0][2]
            self.assertEqual(grip.yaw, 0.0)
            self.assertEqual((grip.pos[0], grip.pos[1]), L.SUP_START[side])
            self.assertAlmostEqual(setw.pos[1], sign * (0.080 - t.SUP_TRIM))
            self.assertAlmostEqual(grip.pos[2] - 0.013 - L.BASE_TOP_W, 0.006)  # pad tips 6 mm above the base
            self.assertAlmostEqual(setw.pos[2] - t.SUP_GRIP_UP - L.BASE_TOP_W, 0.001)
            lift = c["lift it off the base"][0][2]
            self.assertGreater(lift.pos[2] - t.SUP_GRIP_UP - L.BASE_TOP_W, 0.030)
        # the specimen: picked at its middle across its width (fingers along x), laid centred over the span
        grip = cl["grip the specimen across its width"][0][2]
        lower = cl["slowly lower it onto the rollers"][0][2]
        letgo = cl["let go of the specimen"][0][2]
        self.assertAlmostEqual(grip.yaw, -math.pi / 2)
        self.assertEqual((grip.pos[0], grip.pos[1]), (L.SP_START[0], L.SP_START[1]))
        self.assertAlmostEqual(grip.pos[2] - 0.013 - (L.TOP + L.MAT_TOP), 0.003)  # pad tips 3 mm above the mat
        self.assertAlmostEqual(lower.pos[0], L.BASE_X)
        self.assertLess(abs(lower.pos[1] - L.MID_Y), 0.001)
        self.assertAlmostEqual(lower.pos[2] - t.SP_GRIP_UP - L.ROLLER_TOP_W, 0.001)
        self.assertEqual(letgo.pos, lower.pos)
        self.assertLess(letgo.finger, t.SP_GRIP)  # it opens (the joint value is minus half the gap)
        carry = cl["carry it over the middle of the span"][0][2]
        self.assertGreater(carry.pos[2] - t.SP_GRIP_UP - L.ROLLER_TOP_W, 0.050)  # carried well over the rollers
        # the open fingers (along x, 40 or 50 mm) stay on the base's 90 mm and clear of the supports (along y)
        self.assertLess(-2 * t.SP_OPEN / 2 + 0.008, L.BASE_X1 - L.BASE_X)
        self.assertGreater(L.SPAN_NOMINAL / 2 - L.BLOCK_SIZE[1] / 2 - 0.010 - 0.0135, 0.040)
        # the clock: two grippers are never over the fixture at once
        left_up = cl["straight up"][0][1]  # the left gripper has risen from its support
        right_over = cr["over the right support"][0]
        self.assertGreaterEqual(right_over[0], left_up)
        right_up = cr["straight up"][0][1]
        carry_start = cl["carry it over the middle of the span"][0][0]
        self.assertGreaterEqual(carry_start, right_up)
        self.assertAlmostEqual(right_over[0], t.RIGHT_GO)
        self.assertAlmostEqual(carry_start, t.SP_GO)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 1.0 + 3.0) * HZ, 1200)  # plan, success hold and the policy's 3 s hold: in limit
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("10 mm thick", "16 times", "centred on the red centre line", "flat across the rollers",
                  "centred and square", "Let go of everything"):
            self.assertIn(s, ins)

    def test_task_json_agrees(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        self.assertIn("1,200 steps", spec["run_limit"])
        for s in ("0, 0.25, 0.5, 0.75", "160 mm", "5 mm", "10 mm", "3 degrees", "20 mm", "0.5 s"):
            self.assertIn(s, spec["scoring"])


if __name__ == "__main__":
    unittest.main()
