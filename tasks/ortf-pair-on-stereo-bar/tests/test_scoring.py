"""Offline tests for scoring.py and task.py: synthetic microphone states and gripper snapshots, no simulator.
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
    arms_valid,
    bottom_gap,
    capsule,
    centre,
    evaluate,
    heading_deg,
    over_plate,
    pair,
    point_distance,
    tier,
    tilt_deg,
    valid,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FULL = L.STILL_WINDOWS * WINDOW + 1  # 21 samples: 1 s of history
A, B = L.MICS


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
    """A mic state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o + np.array([0.0, 0.0, 0.012]),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.04, high=o + 0.04)


def mic(cap_xy, yaw, under=L.PLATE_TOP_W, pitch=0.0, roll=0.0, vel=(0.0, 0.0, 0.0)):
    """A mic whose capsule point is at ``cap_xy`` (its height that of a mic lying flat with its underside at
    ``under``), its grille end pointing ``yaw`` degrees; pitch and roll turn it about the capsule point."""
    q = quat(yaw, pitch, roll)
    cap = np.array([cap_xy[0], cap_xy[1], under + L.MIC_H / 2])
    return raw(cap - _rot(q) @ np.array(L.CAPSULE_LOCAL), q, vel)


def mic_at_centre(c_xy, yaw, under=L.PLATE_TOP_W, **kw):
    """A mic lying flat with its centre at ``c_xy``."""
    u = np.array(L.unit(yaw))
    return mic(np.asarray(c_xy) + (L.CAPSULE_LOCAL[0] - L.CENTER_LOCAL[0]) * u, yaw, under, **kw)


def ortf(sep=0.170, angle=110.0, mid=(0.320, 0.0), bis=0.0, crossed=False, under=L.PLATE_TOP_W):
    """Two mics: capsules ``sep`` apart either side of ``mid`` square to the bisector heading ``bis``, axes ``angle``
    apart about it (splayed outward; ``crossed`` swaps the headings so the capsules point across each other)."""
    n = np.array(L.unit(bis + 90.0))
    ca, cb = np.asarray(mid) + n * sep / 2, np.asarray(mid) - n * sep / 2
    ya, yb = bis + angle / 2, bis - angle / 2
    if crossed:
        ya, yb = yb, ya
    return {A: mic(ca, ya, under), B: mic(cb, yb, under)}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


PARKED = {"left": arm((0.20, 0.25, L.TOP + 0.15), 50.0 / 93.9), "right": arm((0.20, -0.25, L.TOP + 0.15), 50.0 / 93.9)}
GOOD = ortf()


def holding(mics, name=B, opening=16.0 / 93.9, side="right"):
    """The ``side`` gripper closed on mic ``name`` as the demo holds it (TCP 17 mm above its underside)."""
    c = centre(mics[name])
    under = float(c[2]) - L.MIC_H / 2
    arms = dict(PARKED)
    arms[side] = arm((c[0], c[1], under + 0.017), opening)
    return arms


def feed(tr, states, arms=PARKED):
    r = None
    for st in states:
        r = tr.update(st, arms)
    return r


def settled(st, arms=PARKED, n=FULL):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n, arms)


def scored(st, arms=PARKED):
    return settled(st, arms)[1].score


def start():
    return {n: mic_at_centre(L.MIC_START[n][:2], L.MIC_START[n][2], under=L.TOP + L.PAD_T) for n in L.MICS}


def moved(st, dx=0.0, dy=0.0, dyaw=0.0):
    """The same mic shifted and turned about its centre (flat)."""
    c = centre(st)
    return mic_at_centre((c[0] + dx, c[1] + dy), heading_deg(st) + dyaw, under=float(c[2]) - L.MIC_H / 2)


class Geometry(unittest.TestCase):
    def test_mic_points(self):
        s = mic((0.320, 0.085), 55.0)
        np.testing.assert_allclose(capsule(s)[:2], [0.320, 0.085], atol=1e-12)
        np.testing.assert_allclose(centre(s)[:2], L.TARGET_CENTRE[A], atol=1e-12)
        np.testing.assert_allclose(centre(s)[:2], [0.30423, 0.06247], atol=1e-4)
        self.assertAlmostEqual(heading_deg(s), 55.0, places=9)
        self.assertAlmostEqual(tilt_deg(s), 0.0, places=9)
        self.assertAlmostEqual(bottom_gap(s), 0.0, places=12)
        # the capsule point is 7.5 mm behind the front face, the tail point 62.5 mm behind it (the rear face)
        self.assertAlmostEqual(L.MIC_LEN / 2 - L.CAPSULE_LOCAL[0], 0.0075)
        self.assertAlmostEqual(L.CAPSULE_LOCAL[0] - L.TAIL_BACK, -L.MIC_LEN / 2)

    def test_demo_end_state_fits_the_plate(self):
        """All four corners of each body lie inside the plate, and the bodies' nearest corners are 54 mm apart."""
        for n in L.MICS:
            s = GOOD[n]
            from scoring import corners
            c = corners(s)
            self.assertTrue(np.all(c[:, 0] > L.PLATE_X0 + 0.010) and np.all(c[:, 0] < L.PLATE_X1 - 0.005))
            self.assertTrue(np.all(np.abs(c[:, 1]) < L.PLATE_Y1 - 0.040))
        from scoring import corners
        ca, cb = corners(GOOD[A])[:, :2], corners(GOOD[B])[:, :2]
        self.assertGreater(min(np.linalg.norm(p - q) for p in ca for q in cb), 0.050)

    def test_pair_measures(self):
        p = pair(GOOD[A], GOOD[B])
        self.assertAlmostEqual(p["separation_mm"], 170.0, places=6)
        self.assertAlmostEqual(p["included_deg"], 110.0, places=6)
        self.assertTrue(p["splay_outward"])
        self.assertAlmostEqual(p["facing_err_deg"], 0.0, places=6)
        self.assertAlmostEqual(p["mid_off_axis_mm"], 0.0, places=6)
        self.assertTrue(p["A"] and p["S"] and p["C"])
        q = pair(GOOD[B], GOOD[A])  # the same with the mics swapped
        for k in ("separation_mm", "included_deg", "splay", "facing_err_deg", "mid_off_axis_mm", "A", "S", "C"):
            self.assertEqual(p[k], q[k], k)

    def test_crossed_pair_splays_inward(self):
        p = pair(*ortf(crossed=True).values())
        self.assertAlmostEqual(p["included_deg"], 110.0, places=6)
        self.assertFalse(p["splay_outward"])
        self.assertEqual(p["why_not_a"], "crossed_not_splayed_outward")

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 0.0)), math.hypot(0.008, 0.0135))
        s = GOOD[A]
        self.assertEqual(point_distance(centre(s), s), 0.0)
        self.assertLess(zone_gap(holding(GOOD, A, side="left")["left"], s), 0.0)
        self.assertGreater(zone_gap(PARKED["left"], s), 0.08)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        st = start()
        self.assertTrue(Tracker.at_start(st))
        r = settled(st)[1]
        self.assertEqual(r.score, 0.0)
        self.assertFalse(over_plate(st[A]))

    def test_ortf_scores_one(self):
        r = settled(GOOD)[1]
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertIsNone(r.details["zero_reason"])
        self.assertTrue(r.details["gate"])

    def test_spacing_off_gives_033(self):
        self.assertEqual(scored(ortf(sep=0.130)), 0.33)
        self.assertEqual(scored(ortf(sep=0.200)), 0.33)
        self.assertEqual(scored(ortf(sep=0.300)), 0.33)  # 110 degrees but 300 mm: the angle alone

    def test_midpoint_off_axis_gives_067(self):
        self.assertEqual(scored(ortf(mid=(0.320, 0.025))), 0.67)
        self.assertEqual(scored(ortf(mid=(0.320, -0.025))), 0.67)

    def test_aim_off_gives_067(self):
        self.assertEqual(scored(ortf(bis=8.0)), 0.67)  # centred but aimed 8 degrees off the mark
        self.assertEqual(scored(ortf(bis=-25.0)), 0.67)  # 25 degrees off: still facing the choir (A)

    def test_tolerance_edges(self):
        for ang, want in ((102.0, 1.0), (118.0, 1.0), (101.8, 0.0), (118.2, 0.0)):
            self.assertEqual(scored(ortf(angle=ang)), want, ang)
        for sep, want in ((0.1601, 1.0), (0.1799, 1.0), (0.1598, 0.33), (0.1802, 0.33)):
            self.assertEqual(scored(ortf(sep=sep)), want, sep)
        for y, want in ((0.0099, 1.0), (-0.0099, 1.0), (0.0102, 0.67)):
            self.assertEqual(scored(ortf(mid=(0.320, y))), want, y)
        for b, want in ((4.9, 1.0), (-4.9, 1.0), (5.2, 0.67), (29.8, 0.67), (30.2, 0.0), (-30.2, 0.0)):
            self.assertEqual(scored(ortf(bis=b)), want, b)

    def test_every_tier_reachable_each_state_one_score(self):
        cases = {1.0: ortf(), 0.67: ortf(mid=(0.32, 0.025)), 0.33: ortf(sep=0.13), 0.0: ortf(angle=90.0)}
        for want, st in cases.items():
            self.assertEqual(scored(st), want)
        for a_, s_, c_ in [(a, s, c) for a in (0, 1) for s in (0, 1) for c in (0, 1)]:
            t = tier({"A": bool(a_), "S": bool(s_), "C": bool(c_)})
            want = 0.0 if not a_ else (1.0 if s_ and c_ else 0.67 if s_ else 0.33)
            self.assertEqual(t, want)

    def test_angle_and_aim_without_spacing_is_033(self):
        self.assertEqual(scored(ortf(sep=0.140)), 0.33)  # A and C, not S

    def test_swapped_mics_score_the_same(self):
        g = {A: GOOD[B], B: GOOD[A]}
        self.assertEqual(scored(g), 1.0)

    def test_roll_about_its_axis_does_not_matter(self):
        for roll in (90.0, 180.0, -90.0):  # a square section: any face down lies the same
            st = {n: mic(capsule(GOOD[n])[:2], heading_deg(GOOD[n]), roll=roll) for n in L.MICS}
            self.assertEqual(scored(st), 1.0, roll)

    def test_score_is_idempotent(self):
        tr, r = settled(GOOD)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(GOOD, PARKED).score, 1.0)
        self.assertEqual(tr.steps, n)


class ZeroConditions(unittest.TestCase):
    def test_spec_shortcut_parallel_spaced_pair(self):
        st = {A: mic((0.320, 0.085), 0.0), B: mic((0.320, -0.085), 0.0)}
        r = settled(st)[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "included_angle_not_110")

    def test_spec_shortcut_xy_pair(self):
        self.assertEqual(scored(ortf(sep=0.020, angle=90.0)), 0.0)

    def test_spec_shortcut_xy_coincident_overlapping_capsules_is_still_zero(self):
        # an XY pair with crossed capsules (the usual way): 90 degrees, crossed
        self.assertEqual(scored(ortf(sep=0.020, angle=90.0, crossed=True)), 0.0)

    def test_nos_and_din_pairs(self):
        self.assertEqual(scored(ortf(sep=0.300, angle=90.0)), 0.0)  # NOS
        self.assertEqual(scored(ortf(sep=0.200, angle=90.0)), 0.0)  # DIN

    def test_spec_shortcut_crossed_ortf(self):
        r = settled(ortf(crossed=True))[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "crossed_not_splayed_outward")

    def test_spec_shortcut_aimed_away(self):
        r = settled(ortf(bis=180.0))[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "not_facing_the_choir")
        self.assertEqual(scored(ortf(bis=90.0)), 0.0)  # aimed sideways

    def test_spec_shortcut_mic_still_gripped(self):
        r = settled(GOOD, holding(GOOD))[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], f"{B}:held_or_touched")
        self.assertEqual(scored(GOOD, holding(GOOD, A, opening=50 / 93.9, side="left")), 0.0)  # open, still on it

    def test_hands_off_is_10_mm(self):
        s = GOOD[B]
        top = float(max(c[2] for c in __import__("scoring").corners(s)))
        c = centre(s)
        r0 = zone_radius(arm((0, 0, 0), 0.5))
        for above, want in ((r0 + 0.0105, 1.0), (r0 + 0.0095, 0.0)):
            tips_z = top + above  # straight above the mic's top: the zone's nearest point is the pad tips
            arms = dict(PARKED)
            arms["right"] = arm((c[0], c[1], tips_z + 0.013), 0.5)
            self.assertEqual(scored(GOOD, arms), want, above)

    def test_spec_shortcut_mic_sliding(self):
        """Mic B gliding 1 mm per step along the plate: not still, 0."""
        tr = Tracker(HZ)
        for i in range(FULL + 5):
            st = {A: GOOD[A], B: moved(GOOD[B], dx=0.001 * (i - FULL - 4))}
            r = tr.update(st, PARKED)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], f"{B}:still_moving")

    def test_spec_shortcut_mic_laid_across_the_other(self):
        """Mic B laid on mic A as an X at 110 degrees, centres together: not resting on the plate, 0."""
        ca = centre(GOOD[A])
        st = {A: mic_at_centre(ca[:2], 55.0), B: mic_at_centre(ca[:2], -55.0, under=L.PLATE_TOP_W + L.MIC_H)}
        r = settled(st)[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], f"{B}:not_resting_on_the_plate")

    def test_not_lying_flat(self):
        st = dict(GOOD)
        st[B] = mic(capsule(GOOD[B])[:2], -55.0, pitch=11.0)
        r = settled(st)[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], f"{B}:not_lying_flat")
        st[B] = mic(capsule(GOOD[B])[:2], -55.0, pitch=90.0)  # standing on its grille
        self.assertEqual(scored(st), 0.0)

    def test_centre_off_the_plate(self):
        st = ortf(mid=(0.390, 0.0))  # the right geometry, but the bodies' centres beyond the plate's far edge
        self.assertGreater(centre(st[A])[0], L.PLATE_X1)
        r = settled(st)[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], f"{A}:not_over_the_plate")

    def test_on_the_table_in_front_of_the_plate(self):
        st = ortf(mid=(0.230, 0.0), under=L.TOP)
        self.assertEqual(scored(st), 0.0)

    def test_bottom_within_5_mm(self):
        for dz, want in ((0.0049, 1.0), (0.0051, 0.0)):
            self.assertEqual(scored(ortf(under=L.PLATE_TOP_W + dz)), want, dz)

    def test_one_mic_on_the_plate_only(self):
        st = {A: GOOD[A], B: start()[B]}
        self.assertEqual(scored(st), 0.0)


class Stillness(unittest.TestCase):
    def test_needs_a_full_second(self):
        tr = Tracker(HZ)
        r = feed(tr, [GOOD] * (FULL - 1))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], f"{A}:still_moving")
        r = tr.update(GOOD, PARKED)
        self.assertEqual(r.score, 1.0)

    def test_comes_to_rest(self):
        tr = Tracker(HZ)
        feed(tr, [{A: GOOD[A], B: moved(GOOD[B], dx=0.002 * (i - 30))} for i in range(30)])
        r = feed(tr, [GOOD] * (FULL - 1))  # the last slide step (2 mm) is still in the first window
        self.assertEqual(r.score, 0.0)
        r = feed(tr, [GOOD])
        self.assertEqual(r.score, 1.0)

    def test_path_limit_per_window(self):
        def run(step):
            tr = Tracker(HZ)
            r = None
            for i in range(FULL):
                r = tr.update({A: GOOD[A], B: moved(GOOD[B], dx=step * (i % 2))}, PARKED)
            return r
        self.assertEqual(run(0.00014).score, 1.0)  # 10 x 0.14 = 1.4 mm of path per window
        self.assertEqual(run(0.00016).score, 0.0)  # 1.6 mm

    def test_slow_turn_is_not_still(self):
        tr = Tracker(HZ)
        r = None
        for i in range(FULL):
            r = tr.update({A: GOOD[A], B: moved(GOOD[B], dyaw=0.11 * i)}, PARKED)  # 2.2 degrees over 1 s
        self.assertEqual(r.score, 0.0)
        tr = Tracker(HZ)
        for i in range(FULL):
            r = tr.update({A: GOOD[A], B: moved(GOOD[B], dyaw=0.09 * i)}, PARKED)  # 1.8 degrees
        self.assertEqual(r.score, 1.0)

    def test_heading_across_180_does_not_read_as_a_turn(self):
        st = {A: mic((0.32, 0.085), 179.99), B: mic((0.32, -0.085), -179.99)}
        tr = Tracker(HZ)
        for i in range(FULL):
            tr.update({A: moved(st[A], dyaw=0.02 * (-1) ** i), B: st[B]}, PARKED)
        self.assertLess(tr.motion(A)[1], 0.05)


class Exploits(unittest.TestCase):
    def test_pair_on_the_table_beside_the_plate(self):
        """The right geometry on the table, in front of the plate: 0."""
        self.assertEqual(scored(ortf(mid=(0.200, 0.0), under=L.TOP)), 0.0)

    def test_pair_held_in_the_air_over_the_plate(self):
        """Both mics held 3 mm over the plate in the right geometry by both grippers: 0."""
        st = ortf(under=L.PLATE_TOP_W + 0.003)
        arms = holding(st, B, side="right")
        arms["left"] = holding(st, A, side="left")["left"]
        self.assertEqual(scored(st, arms), 0.0)

    def test_pressed_by_an_open_gripper(self):
        """An open gripper resting its pads on mic B (tips at its top face): 0."""
        s = GOOD[B]
        c = centre(s)
        arms = dict(PARKED)
        arms["right"] = arm((c[0], c[1], L.PLATE_TOP_W + L.MIC_H + 0.013), 1.0)
        self.assertEqual(scored(GOOD, arms), 0.0)

    def test_mics_standing_on_their_tails(self):
        st = {n: mic(capsule(GOOD[n])[:2], heading_deg(GOOD[n]), pitch=-90.0) for n in L.MICS}
        self.assertEqual(scored(st), 0.0)

    def test_mics_left_in_their_case_pads(self):
        self.assertEqual(scored(start()), 0.0)

    def test_unreadable_arms_cannot_skip_the_rest(self):
        self.assertEqual(scored(ortf(crossed=True), None), 0.0)
        self.assertEqual(scored(ortf(under=L.TOP), None), 0.0)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr, r = settled(GOOD, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(tr.arms_unavailable, FULL)
        self.assertIsNone(r.details["zone_gap_mm"])

    def test_non_finite_arms_count_as_unavailable(self):
        bad = dict(PARKED)
        bad["left"] = arm((math.nan, 0.0, 1.0))
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": PARKED["left"]}))
        self.assertFalse(arms_valid([]))
        tr, r = settled(GOOD, bad)
        self.assertEqual(tr.arms_unavailable, FULL)

    def test_left_gripper_counts_too(self):
        self.assertEqual(scored(GOOD, holding(GOOD, A, side="left")), 0.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(GOOD)
        hist = [list(h) for h in tr.history.values()]
        good_b = GOOD[B]
        for bad in (
            raw((math.nan, 0, 0), quat()),
            raw((0.3, 0, 0.76), (0, 0, 0, 0)),
            raw((0.3, 0, 0.76), (math.inf, 0, 0, 1)),
            SimpleNamespace(quat_xyzw=quat(), pos=np.zeros(3), lin_vel=np.zeros(3)),  # no origin
            SimpleNamespace(origin=np.zeros(3), quat_xyzw=quat(), pos=np.zeros(3), lin_vel=np.array([math.nan] * 3)),
            raw((0.3, 0, 0.76), (0, 0, 1)),  # a 3-element quaternion
            None,
        ):
            self.assertFalse(valid(bad))
            r = tr.update({A: GOOD[A], B: bad}, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        r = tr.update({A: GOOD[A]}, PARKED)  # a missing mic
        self.assertEqual(r.score, 0.0)
        self.assertEqual([list(h) for h in tr.history.values()], hist)
        self.assertEqual(tr.bad_steps, 8)
        self.assertEqual(tr.update({A: GOOD[A], B: good_b}, PARKED).score, 1.0)

    def test_unnormalised_quaternion(self):
        st = {n: raw(GOOD[n].origin, 3.0 * GOOD[n].quat_xyzw) for n in L.MICS}
        self.assertEqual(scored(st), 1.0)

    def test_random_states_do_not_crash_and_score_a_listed_value(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            st = {}
            for n in L.MICS:
                q = rng.normal(size=4)
                o = (rng.uniform(0.1, 0.5), rng.uniform(-0.4, 0.4), L.TOP + rng.uniform(-0.01, 0.06))
                st[n] = raw(o, q)
            r = Tracker(HZ).result(st, PARKED)
            self.assertIn(r.score, (0.0, 0.33, 0.67, 1.0))
            json.dumps(r.details)

    def test_random_flat_pairs_on_the_plate(self):
        rng = np.random.default_rng(11)
        seen = set()
        for _ in range(600):
            st = ortf(sep=rng.uniform(0.14, 0.20), angle=rng.uniform(95, 125), bis=rng.uniform(-12, 12),
                      mid=(0.320, rng.uniform(-0.02, 0.02)), crossed=bool(rng.random() < 0.2))
            if not all(over_plate(s) for s in st.values()):
                continue
            sc = evaluate(st, {n: True for n in L.MICS}, {n: True for n in L.MICS})[0]
            self.assertIn(sc, (0.0, 0.33, 0.67, 1.0))
            seen.add(sc)
        self.assertEqual(seen, {0.0, 0.33, 0.67, 1.0})


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(GOOD)
        tr.first_at[1.0] = 5
        r = tr.update(start(), PARKED)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(tr.first_at, {})
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(tr.history[A]), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(GOOD)
        st = {n: moved(s, dx=0.02, dy=-0.02) for n, s in start().items()}
        tr.update(st, PARKED)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(GOOD)
        n = tr.steps
        a0, a1 = centre(GOOD[A]), centre(start()[A])
        for i in range(1, 41):
            f = i / 40
            c = a0 + (a1 - a0) * f
            st = {A: raw(start()[A].origin + (c - a1), start()[A].quat_xyzw, (a1 - a0) * HZ / 40), B: GOOD[B]}
            tr.update(st, PARKED)
        self.assertEqual(tr.steps, n + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(GOOD)
        n = tr.steps
        tr.update({A: moved(GOOD[A], dy=-0.2), B: GOOD[B]}, PARKED)
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(GOOD)
        tr.reset()
        self.assertEqual((tr.steps, tr.bad_steps, tr.arms_unavailable), (0, 0, 0))
        self.assertEqual(tr.update(GOOD, PARKED).score, 0.0)  # needs a second of history again


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
        cls.specs = {m.name: m for m in cls.task.MICS}

    def sim(self, st):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*s.origin, *s.quat_xyzw), source="sim")
                for n, s in st.items()}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in t.OBJECTS if o.mass is not None), [A, B])
        for m in t.MICS:
            self.assertAlmostEqual(m.mass, 0.080)
            self.assertLessEqual(m.mass, 0.30)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        p = t.PLATE
        self.assertAlmostEqual(p.pos[2] + p.size[2] / 2, L.PLATE_TOP_W)
        self.assertAlmostEqual(p.size[0], 0.080)
        self.assertAlmostEqual(p.size[1], 0.280)
        self.assertAlmostEqual(t.MARK.pos[0], 0.485)
        self.assertAlmostEqual(t.MARK.pos[1], 0.0)
        self.assertAlmostEqual(t.MARK.pos[2] - t.MARK.height / 2, L.TOP + L.RISER_H)
        for f in t.FIGURES:  # the singers leave the mark clear
            self.assertGreater(abs(f.pos[1]) - f.radius, L.MARK_RADIUS + 0.020)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            elif hasattr(o, "size"):
                ext = np.array([abs(math.cos(o.yaw)) * o.size[0] / 2 + abs(math.sin(o.yaw)) * o.size[1] / 2,
                                abs(math.sin(o.yaw)) * o.size[0] / 2 + abs(math.cos(o.yaw)) * o.size[1] / 2, 0])
                lo, hi = np.asarray(o.pos) - ext, np.asarray(o.pos) + ext
            else:
                lo = np.asarray(o.pos) - o.radius
                hi = np.asarray(o.pos) + o.radius
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)

    def test_mic_geometry_matches_the_scorer(self):
        for n, spec in self.specs.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-0.035, -0.012, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [0.035, 0.012, 0.024], atol=1e-12)
            np.testing.assert_allclose(spec.center_of_mass(), L.CENTER_LOCAL, atol=1e-9)  # the grip is at the centre
            grille = [s for s in spec.shapes if s.color is not None and tuple(s.color) == L.GRILLE_SILVER]
            self.assertEqual(len(grille), 1)
            self.assertAlmostEqual(grille[0].center[0] - grille[0].size[0] / 2, 0.020)  # the front 15 mm
            rest = spec.rest_state()
            self.assertAlmostEqual(heading_deg(rest), L.MIC_START[n][2], places=6)
            np.testing.assert_allclose(centre(rest)[:2], L.MIC_START[n][:2], atol=1e-12)
            self.assertAlmostEqual(float(rest.low[2]), L.TOP + L.PAD_T, places=9)
        self.assertTrue(Tracker.at_start({n: s.rest_state() for n, s in self.specs.items()}))
        st = self.sim(GOOD)
        self.assertAlmostEqual(pair(st[A], st[B])["separation_mm"], 170.0, places=6)

    def test_start_footprints(self):
        """Mic A's footprint at start spans x 0.213-0.287, y 0.297-0.343 (mic B the mirror), inside its case pad."""
        lo, hi = self.specs[A].world_bounds()
        np.testing.assert_allclose([lo[0], hi[0], lo[1], hi[1]], [0.2130, 0.2870, 0.2968, 0.3432], atol=5e-4)
        lo_b, hi_b = self.specs[B].world_bounds()
        np.testing.assert_allclose([lo_b[1], hi_b[1]], [-hi[1], -lo[1]], atol=1e-12)
        for n in L.MICS:  # every corner inside the pad
            from scoring import corners
            x, y, yaw = L.MIC_START[n]
            r = np.array([[math.cos(math.radians(yaw)), math.sin(math.radians(yaw))],
                          [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))]])
            loc = (corners(self.specs[n].rest_state())[:, :2] - np.array([x, y])) @ r.T
            self.assertTrue(np.all(np.abs(loc[:, 0]) < L.PAD_SIZE[0] / 2) and np.all(np.abs(loc[:, 1]) < L.PAD_SIZE[1] / 2))

    def test_offline_state_path(self):
        spec = self.specs[A]
        box = SimpleNamespace(pos=np.array(spec.offline_box()[0]), quat_xyzw=quat(L.MIC_START[A][2]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(centre(st)[:2], L.MIC_START[A][:2], atol=1e-12)
        self.assertAlmostEqual(heading_deg(st), L.MIC_START[A][2], places=6)

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(PARKED)
        s0 = self.sim(start())
        self.assertFalse(t.success(s0))
        self.assertEqual(t.score(s0), 0.0)
        good = self.sim(GOOD)
        for _ in range(FULL - 1):
            self.feed_arms(PARKED)
            t.success(good)
        self.feed_arms(PARKED)
        self.assertTrue(t.success(good))
        self.assertEqual(t.score(good), 1.0)
        self.assertEqual(t.score(good), 1.0)  # score() does not advance it
        self.feed_arms(holding(GOOD))
        self.assertEqual(t.score(good), 0.0)  # read with the arms of the step at hand
        self.feed_arms(PARKED)
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(PARKED)
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
                self.feed_arms(PARKED)
                self.task.success(self.sim(GOOD))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "gate", "mics", "pair", "motion",
                          "zone_gap_mm", "arms_unavailable", "bad_steps", "first_step", f"{A}_pose", f"{B}_pose",
                          "left_tcp", "right_tcp", "openings", "episode", "mechkit_episode"):
                    self.assertIn(k, row)
                for k in ("separation_mm", "included_deg", "splay_outward", "bisector_deg", "facing_err_deg",
                          "mid_off_axis_mm", "A", "S", "C"):
                    self.assertIn(k, row["pair"])
                for k in ("centre_mm", "yaw_deg", "tilt_deg", "bottom_mm", "over_plate", "still", "hands_off"):
                    self.assertIn(k, row["mics"][A])
                self.assertAlmostEqual(row["pair"]["separation_mm"], 170.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(GOOD)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def _end_state(self, lag=0.0, bias=0.0):
        """Where the demo leaves the mics if each lands ``lag`` m short of its place point along the carry and
        ``bias`` degrees counter-clockwise of its commanded yaw."""
        t = self.task
        out = {}
        for side, n in t.ARM_OF.items():
            px, py = t.place_point(side)
            sx, sy = t.pick_point(side)
            d = math.hypot(px - sx, py - sy)
            c = (px - lag * (px - sx) / d, py - lag * (py - sy) / d)
            out[n] = mic_at_centre(c, math.degrees(t.yaws(side)[1]) + bias)
        return out

    def test_demo_end_state_scores_one_with_and_without_the_gpu_lag(self):
        offline = self._end_state()
        p = pair(offline[A], offline[B])
        self.assertAlmostEqual(p["separation_mm"], 162.2, delta=0.3)
        self.assertAlmostEqual(p["included_deg"], 110.0, places=6)
        self.assertAlmostEqual(p["facing_err_deg"], 1.0, delta=0.2)
        for lag in (0.0, 0.002, 0.004, 0.007, 0.008):
            for bias in (0.0, 0.6, 1.2, 1.7):
                self.assertEqual(scored(self._end_state(lag, bias)), 1.0, (lag, bias))
        at4 = pair(*self._end_state(0.004, 1.2).values())
        self.assertAlmostEqual(at4["separation_mm"], 170.0, delta=0.5)

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, wps in plan.items():
            by = {w.label: w for w in wps}
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every close and open takes 1.5 s
            self.assertGreaterEqual(by["slowly down round the mic"].duration, 1.5)
            self.assertGreaterEqual(by["slowly lay the mic on the plate"].duration, 1.5)
            self.assertGreaterEqual(by["settle"].duration, 0.5)
            grip, lay = by["grip the mic across its sides"], by["slowly lay the mic on the plate"]
            n = t.ARM_OF[side]
            self.assertAlmostEqual(math.degrees(grip.yaw), L.MIC_START[n][2])
            self.assertAlmostEqual(math.degrees(lay.yaw), L.TARGET_YAW_DEG[n] - 1.0)
            turn = L.TARGET_YAW_DEG[n] - L.MIC_START[n][2]
            self.assertAlmostEqual(abs(turn), 75.0)  # +75 (left) and -75 (right), each laid 1 degree clockwise
            self.assertAlmostEqual(math.degrees(lay.yaw - grip.yaw), turn - 1.0)
            self.assertEqual((grip.pos[0], grip.pos[1]), L.MIC_START[n][:2])
            # pad tips 4 mm above the mic's underside at the pick, and when it is laid 1 mm over the plate
            self.assertAlmostEqual(grip.pos[2] - 0.013 - (L.TOP + L.PAD_T), 0.004)
            self.assertAlmostEqual(lay.pos[2] - 0.013 - (L.PLATE_TOP_W + t.DROP), 0.004)
            self.assertAlmostEqual(t.GRIP, -0.008)  # 16 mm commanded across 24 mm
            # the place point leads the target by 4 mm along the carry
            tx, ty = L.TARGET_CENTRE[n]
            self.assertAlmostEqual(math.hypot(lay.pos[0] - tx, lay.pos[1] - ty), 0.004, places=9)
            self.assertLess(abs(lay.pos[1]), abs(ty))  # toward the midline
        # the right arm's carry starts only once the left gripper has let go and gone up
        tl = 0.0
        for w in plan["left"]:
            tl += w.duration
            if w.label == "straight up":
                left_up = tl
        tr_ = 0.0
        for w in plan["right"]:
            if w.label == "carry it over the bar plate, turning it":
                right_carry = tr_
            tr_ += w.duration
        self.assertGreaterEqual(right_carry, left_up - 1e-9)
        self.assertLessEqual((max(sum(w.duration for w in p) for p in plan.values()) + 3.05) * HZ, 600)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_poses_reachable_with_joint_margins(self):
        sys.path.insert(0, str(_stub_dir().parent))
        import yamkin as K
        t = self.task
        bases = {"left": (0.0, 0.305, 0.75), "right": (0.0, -0.305, 0.75)}
        for side, wps in t.demo().items():
            for w in wps:
                ok, q, _, _ = K.reachable(w.pos, w.yaw, bases[side])
                self.assertTrue(ok, (side, w.label))
                m = np.degrees(K.joint_margins(q))
                self.assertGreater(m.min(), 15.0, (side, w.label, m))
                self.assertGreater(m[5], 20.0, (side, w.label))  # joint 6 (the yaw)

    def test_instruction(self):
        ins = self.task.INSTRUCTION
        for s in ("ORTF pair", "silver grille", "capsule", "flat", "bar plate", "white mark", "let go of both"):
            self.assertIn(s, ins)
        for s in ("17", "110", "170", "55", "degree", "splay"):  # the technique is named; its numbers are not
            self.assertNotIn(s, ins)
        data = json.loads((HERE / "task.json").read_text())
        self.assertEqual(data["instruction"], ins)
        for s in ("17", "110"):
            self.assertNotIn(s, data["title"])


if __name__ == "__main__":
    unittest.main()
