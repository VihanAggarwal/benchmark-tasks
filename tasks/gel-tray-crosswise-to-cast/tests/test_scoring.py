"""Offline tests for scoring.py, offline_nest.py and task.py: synthetic tray and comb states and gripper snapshots, no
simulator. python3 -m unittest discover -s tests -v"""

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
    comb_in_tray,
    comb_slot,
    comb_zone_gap,
    cross_err_deg,
    gasket_gaps,
    point_distance,
    tray_check,
    tray_corners,
    tray_tilt_deg,
    tray_zone_gap,
    valid,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
PLAT_W = L.TOP + L.PLAT_TOP


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
    """A part's state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o.copy(),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.05, high=o + 0.05)


def tray(dx=0.0, dy=0.0, up=0.0, yaw=0.0, pitch=0.0, roll=0.0, x=None):
    """The tray with the middle of its underside at (tank centre + dx, dy) and ``up`` above the platform's top. yaw 0:
    crosswise (length along x); 90: lengthwise. pitch and roll turn it about the middle of its underside."""
    cx = L.TANK_X + dx if x is None else x
    return raw((cx, L.TANK_Y + dy, PLAT_W + up), quat(yaw, pitch, roll))


def comb(t, slot="end_minus", dx=0.0, dy=0.0, dz=0.0, yaw=0.0, pitch=0.0, roll=0.0, z_local=None):
    """The comb in tray state ``t``'s frame: the middle of its bar's underside at the slot's centre + dx along the tray,
    dy across it and the notch bottom + dz above the tray's underside (or ``z_local``), its bar along the tray's width
    turned ``yaw`` from it; pitch and roll tilt it about that point."""
    rt = _rot(t.quat_xyzw)
    zl = L.NOTCH_BOTTOM + dz if z_local is None else z_local
    local = np.array([L.SLOTS[slot] + dx, dy, zl])
    o = np.asarray(t.origin) + rt @ local
    rc = rt @ _rot(quat(yaw, pitch, roll))
    from scoring import rotation  # noqa: F401  (same convention)
    q = _quat_of(rc)
    return raw(o, q)


def _quat_of(r):
    """(x, y, z, w) of a rotation matrix."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        return np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
    if i == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        return np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    return np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s])


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


LEFT_PARK = arm((0.22, 0.22, L.TOP + 0.15))
RIGHT_PARK = arm((0.22, -0.20, L.TOP + 0.15))
FAR = {"left": LEFT_PARK, "right": RIGHT_PARK}
# the left gripper closed on the tray as the demo sets it down (TCP 18 mm above its underside, on the 64 mm walls)
HOLD_TRAY = {"left": arm((L.TANK_X, 0.0, PLAT_W + 0.018), 0.064 / L.FULL_GAP), "right": RIGHT_PARK}
# the right gripper closed on the comb's bar ends as the demo seats it (TCP 21 mm above the bar's underside)
HOLD_COMB_IN_SLOT = {"left": LEFT_PARK,
                     "right": arm((L.TANK_X + L.SLOTS["end_minus"], 0.0, PLAT_W + L.NOTCH_BOTTOM + 0.021),
                                  0.072 / L.FULL_GAP)}

START_TRAY = raw((L.TRAY_START[0], L.TRAY_START[1], L.TOP), quat(90.0))
START_COMB = raw((L.COMB_START[0], L.COMB_START[1], L.TOP + L.COMB_START[2]), quat(0.0))
SET = tray()
SEATED_END = comb(SET, "end_minus")


def feed(tr, pairs, arms=FAR):
    r = None
    for t, c in pairs:
        r = tr.update(t, c, arms)
    return r


def settled(t, c, arms=FAR, n=WINDOW + 1):
    """A tracker that has seen (t, c) at rest for ``n`` steps; returns (tracker, last result)."""
    tr = Tracker(HZ)
    return tr, feed(tr, [(t, c)] * n, arms)


def scored(t, c, arms=FAR):
    return settled(t, c, arms)[1].score


def reason(t, c, arms=FAR):
    return settled(t, c, arms)[1].details


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(valid(START_TRAY) and valid(START_COMB))
        self.assertTrue(Tracker.at_start(START_TRAY))
        self.assertFalse(Tracker.at_start(SET))
        self.assertAlmostEqual(cross_err_deg(START_TRAY), 90.0)  # lengthwise on the bench
        self.assertEqual(scored(START_TRAY, START_COMB), 0.0)

    def test_set_tray_measures(self):
        self.assertAlmostEqual(tray_tilt_deg(SET), 0.0, places=6)
        self.assertAlmostEqual(cross_err_deg(SET), 0.0, places=6)
        near, far = gasket_gaps(SET)
        self.assertAlmostEqual(near, 0.005, places=12)  # 84 mm in the 94 mm gap: 5 mm a side when centred
        self.assertAlmostEqual(far, 0.005, places=12)
        for c in tray_corners(SET):
            self.assertAlmostEqual(c[2], PLAT_W, places=12)
        ok, why, _ = tray_check(SET)
        self.assertTrue(ok, why)
        self.assertAlmostEqual(cross_err_deg(tray(yaw=180.0)), 0.0, places=6)  # either way round
        self.assertTrue(tray_check(tray(yaw=180.0))[0])

    def test_gaskets_read_as_a_rule_would(self):
        near, far = gasket_gaps(tray(dx=0.002))
        self.assertAlmostEqual(near, 0.007, places=12)
        self.assertAlmostEqual(far, 0.003, places=12)
        self.assertAlmostEqual(near + far, L.SIDE_IN[1] - L.SIDE_IN[0] - L.TRAY_LEN, places=12)  # the 10 mm of play
        self.assertAlmostEqual(near + far, L.GASKET_MAX, places=12)

    def test_a_tray_between_the_walls_is_crosswise_or_lengthwise(self):
        # turned theta, the 84 x 64 mm tray spans 84 cos + 64 sin along x: between walls 94 mm apart it can be turned
        # at most about 10.2 degrees from crosswise (so every tray down between them passes the 12 degree limit), and
        # at least 64 from it when lengthwise
        def span(th):
            return L.TRAY_LEN * math.cos(math.radians(th)) + L.TRAY_W * math.sin(math.radians(th))
        gap = L.SIDE_IN[1] - L.SIDE_IN[0]
        fits = [th / 10 for th in range(0, 901) if span(th / 10) <= gap]
        self.assertLess(max(t for t in fits if t < 45), L.CROSS_DEG)
        self.assertGreater(min(t for t in fits if t > 45), 60.0)
        self.assertLess(L.TRAY_W, gap)  # a lengthwise tray fits on the platform too: the 0 tier is reachable

    def test_seated_comb_measures(self):
        w = comb_in_tray(SEATED_END, SET)
        np.testing.assert_allclose(w["seat_x"], [-0.025, -0.025], atol=1e-12)
        np.testing.assert_allclose(w["seat_z"], [L.NOTCH_BOTTOM] * 2, atol=1e-12)
        self.assertAlmostEqual(w["mid_y"], 0.0, places=12)
        self.assertAlmostEqual(w["heading_deg"], 0.0, places=6)
        self.assertAlmostEqual(w["tilt_deg"], 0.0, places=5)
        self.assertEqual(comb_slot(SEATED_END, SET)[0], "end_minus")
        self.assertEqual(comb_slot(comb(SET, "end_plus"), SET)[0], "end_plus")
        self.assertEqual(comb_slot(comb(SET, "middle"), SET)[0], "middle")
        # turned 180 the comb is the same comb
        self.assertEqual(comb_slot(comb(SET, "end_minus", yaw=180.0), SET)[0], "end_minus")

    def test_seated_comb_clears_the_tray(self):
        # teeth 2 mm above the tray's floor, between the side walls with 5 mm a side; bar 5 mm a side in the notch
        teeth_bottom = L.NOTCH_BOTTOM - L.COMB_TEETH[2]
        self.assertAlmostEqual(teeth_bottom - L.TRAY_FLOOR, 0.002, places=12)
        inner = L.TRAY_W - 2 * L.TRAY_WALL
        self.assertAlmostEqual((inner - L.COMB_TEETH[1]) / 2, 0.005, places=12)
        self.assertAlmostEqual((L.NOTCH_W - L.COMB_BAR[0]) / 2, 0.005, places=12)
        # a bar physically down in a notch (5 mm of play a side, turned at most about 9 degrees) reads inside the
        # 6 mm slot tolerance and the 10 degree heading limit; neighbouring slots are 25 mm apart
        self.assertGreater(L.SLOT_X_TOL, (L.NOTCH_W - L.COMB_BAR[0]) / 2)
        self.assertLess(2 * L.SLOT_X_TOL, L.SLOTS["end_plus"] - L.SLOTS["middle"])
        self.assertGreater(L.COMB_BAR[1], L.TRAY_W)  # the bar spans both walls, 4 mm past each

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 1), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        self.assertLess(tray_zone_gap(HOLD_TRAY["left"], SET), 0.0)
        self.assertLess(comb_zone_gap(HOLD_COMB_IN_SLOT["right"], SEATED_END), 0.0)
        self.assertLess(tray_zone_gap(HOLD_COMB_IN_SLOT["right"], SET), L.HANDS_OFF)  # holding the comb in it
        self.assertGreater(tray_zone_gap(LEFT_PARK, SET), 0.05)
        self.assertGreater(comb_zone_gap(RIGHT_PARK, SEATED_END), 0.05)
        self.assertEqual(point_distance(np.asarray(SET.origin) + [0, 0, 0.005], SET, L.TRAY_BOX_CENTER,
                                        L.TRAY_BOX_HALF), 0.0)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(scored(START_TRAY, START_COMB), 0.0)

    def test_end_slots_score_one(self):
        for slot in L.END_SLOTS:
            tr, r = settled(SET, comb(SET, slot))
            self.assertEqual(r.score, 1.0, slot)
            self.assertTrue(r.success)
            self.assertEqual(r.details["comb_slot"], slot)
        self.assertEqual(scored(tray(yaw=180.0), comb(tray(yaw=180.0), "end_plus")), 1.0)

    def test_middle_slot_scores_three_quarters(self):
        r = settled(SET, comb(SET, "middle"))[1]
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_tray_set_comb_not_seated_is_half(self):
        cases = {
            "comb in its stand": START_COMB,
            "comb on the wall tops across an end slot": comb(SET, "end_minus", z_local=L.TRAY_H),
            "comb on the wall tops between slots": comb(SET, "end_minus", dx=0.011, z_local=L.TRAY_H),
            "comb lying in the tray": comb(SET, "middle", z_local=L.TRAY_FLOOR + L.COMB_BAR[0] / 2, roll=90.0),
            "comb on the bench": raw((0.20, -0.15, L.TOP + L.COMB_TEETH[2]), quat(30.0)),
            "comb along the tray": comb(SET, "middle", yaw=90.0, z_local=L.TRAY_FLOOR + L.COMB_TEETH[2]),
        }
        for name, c in cases.items():
            with self.subTest(name):
                self.assertEqual(scored(SET, c), 0.5)

    def test_comb_held_away_from_the_tray_is_half(self):
        held = raw((0.20, -0.24, L.TOP + 0.11), quat(0.0))
        arms = {"left": LEFT_PARK, "right": arm((0.20, -0.24, L.TOP + 0.11 + 0.021), 0.072 / L.FULL_GAP)}
        self.assertEqual(scored(SET, held, arms), 0.5)

    def test_every_tier_is_reachable_and_each_pose_has_one_score(self):
        got = {scored(START_TRAY, START_COMB), scored(SET, START_COMB), scored(SET, comb(SET, "middle")),
               scored(SET, SEATED_END)}
        self.assertEqual(got, {0.0, 0.5, 0.75, 1.0})

    def test_seat_tolerances(self):
        self.assertEqual(scored(SET, comb(SET, dz=0.0014)), 1.0)
        self.assertEqual(scored(SET, comb(SET, dz=-0.0014)), 1.0)
        self.assertEqual(scored(SET, comb(SET, dz=0.0016)), 0.5)
        self.assertEqual(scored(SET, comb(SET, dx=0.0059)), 1.0)
        self.assertEqual(scored(SET, comb(SET, dx=-0.0059)), 1.0)
        self.assertEqual(scored(SET, comb(SET, dx=0.0061)), 0.5)
        self.assertEqual(scored(SET, comb(SET, "middle", dx=-0.0061)), 0.5)
        self.assertEqual(scored(SET, comb(SET, yaw=9.0)), 1.0)
        self.assertEqual(scored(SET, comb(SET, dy=0.0059)), 1.0)
        self.assertEqual(scored(SET, comb(SET, dy=0.0061)), 0.5)
        self.assertEqual(scored(SET, comb(SET, yaw=4.0)), 1.0)
        self.assertEqual(scored(SET, comb(SET, yaw=11.0)), 0.5)
        # rolled along its length the bar's two seats part in height (29.5 mm out each side): 2 deg 1.0 mm, 4 deg 2.1
        self.assertEqual(scored(SET, comb(SET, roll=2.0)), 1.0)
        self.assertEqual(scored(SET, comb(SET, roll=4.0)), 0.5)
        self.assertEqual(scored(SET, comb(SET, pitch=4.0)), 1.0)
        self.assertEqual(scored(SET, comb(SET, pitch=6.0)), 0.5)

    def test_tray_tolerances(self):
        for kw, want in (({"dx": 0.0049}, 1.0), ({"dx": -0.0049}, 1.0), ({"up": 0.0019}, 1.0), ({"up": 0.0021}, 0.0),
                         ({"yaw": 5.0}, 1.0), ({"yaw": 10.0}, 1.0), ({"pitch": 2.5}, 1.0), ({"pitch": 3.5}, 0.0),
                         ({"roll": 3.5}, 0.0), ({"dy": 0.020}, 1.0), ({"dy": -0.0225}, 1.0), ({"dy": 0.0235}, 0.0)):
            with self.subTest(kw=kw):
                t = tray(**kw)
                self.assertEqual(scored(t, comb(t)), want)

    def test_score_is_idempotent(self):
        tr, r = settled(SET, SEATED_END)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(SET, SEATED_END, FAR).score, 1.0)
        self.assertEqual(tr.steps, n)


class ZeroConditions(unittest.TestCase):
    def test_lengthwise_in_the_tank(self):
        t = tray(yaw=90.0)
        d = reason(t, comb(t, "end_minus"))
        self.assertEqual(scored(t, comb(t, "end_minus")), 0.0)
        self.assertEqual(d["zero_reason"], "tray_lengthwise")

    def test_not_crosswise(self):
        t = tray(yaw=14.0)
        self.assertEqual(scored(t, comb(t)), 0.0)
        self.assertEqual(reason(t, comb(t))["zero_reason"], "tray_not_crosswise")

    def test_upside_down(self):
        t = tray(roll=180.0, up=L.TRAY_H)
        self.assertEqual(scored(t, START_COMB), 0.0)
        self.assertEqual(reason(t, START_COMB)["zero_reason"], "tray_upside_down")

    def test_tilted_or_propped(self):
        propped = tray(pitch=-math.degrees(math.atan2(L.TANK_H - L.PLAT_TOP, L.TRAY_LEN)))  # one end on a wall top
        for t in (tray(pitch=4.0), propped):
            self.assertEqual(scored(t, START_COMB), 0.0)

    def test_held_above_the_platform(self):
        self.assertEqual(scored(tray(up=0.003), START_COMB), 0.0)
        self.assertEqual(reason(tray(up=0.003), START_COMB)["zero_reason"], "tray_not_on_platform")

    def test_on_the_bench(self):
        self.assertEqual(scored(START_TRAY, START_COMB), 0.0)
        bench = raw((0.20, 0.20, L.TOP), quat(0.0))  # crosswise but on the table
        self.assertEqual(scored(bench, comb(bench)), 0.0)

    def test_not_between_the_side_walls(self):
        # at the platform's height but past a wall (only a scorer check: the walls stop it)
        t = tray(dx=0.0055)
        self.assertEqual(scored(t, START_COMB), 0.0)
        self.assertEqual(reason(t, START_COMB)["zero_reason"], "tray_not_between_side_walls")
        stand = raw((L.STAND_X, L.STAND_Y, L.TOP + L.STAND_PLATE[2] - 0.002), quat(0.0))  # on the comb stand
        self.assertEqual(scored(stand, START_COMB), 0.0)

    def test_off_the_platform(self):
        # every corner of its underside must be over the platform (110 mm long, the tray 64 mm wide): its centre within
        # 23 mm of y 0 when square; a tray hanging past the platform's end is not on it, even if level
        for dy in (0.060, 0.030, -0.024):
            with self.subTest(dy=dy):
                t = tray(dy=dy)
                self.assertEqual(scored(t, comb(t)), 0.0)
                self.assertEqual(reason(t, comb(t))["zero_reason"], "tray_off_platform")
        self.assertAlmostEqual(reason(tray(dy=0.030), START_COMB)["tray_overhang_mm"], 7.0)

    def test_tray_held_or_touched(self):
        self.assertEqual(scored(SET, SEATED_END, HOLD_TRAY), 0.0)
        self.assertEqual(reason(SET, SEATED_END, HOLD_TRAY)["zero_reason"], "gripper_on_or_near_tray")

    def test_comb_held_in_its_slot(self):
        self.assertEqual(scored(SET, SEATED_END, HOLD_COMB_IN_SLOT), 0.0)

    def test_hands_off_is_20_mm(self):
        # a gripper fully open straight above the tray's far end (the comb is in the near end slot): its finger zone
        # reaches 57 mm below the pad tips, so it is 20 mm clear of the tray with the tips 77 mm above the tray's top
        r = zone_radius(arm((0, 0, 0), 1.0))
        top = PLAT_W + L.TRAY_H
        x = L.TANK_X + 0.030
        c = comb(SET, "end_minus")
        above = {"left": arm((x, 0.0, top + r + 0.021 + 0.013)), "right": RIGHT_PARK}
        near = {"left": arm((x, 0.0, top + r + 0.019 + 0.013)), "right": RIGHT_PARK}
        self.assertAlmostEqual(tray_zone_gap(above["left"], SET), 0.021, places=9)
        self.assertAlmostEqual(tray_zone_gap(near["left"], SET), 0.019, places=9)
        self.assertGreater(comb_zone_gap(near["left"], c), L.HANDS_OFF)
        self.assertEqual(scored(SET, c, above), 1.0)
        self.assertEqual(scored(SET, c, near), 0.0)
        self.assertEqual(reason(SET, c, near)["zero_reason"], "gripper_on_or_near_tray")

    def test_still_moving(self):
        tr = Tracker(HZ)
        states = [(tray(dx=0.0004 * (i - 5)), comb(tray(dx=0.0004 * (i - 5)))) for i in range(WINDOW + 1)]
        self.assertEqual(feed(tr, states).score, 0.0)  # 4 mm in 0.5 s
        self.assertEqual(tr.last.details["zero_reason"], "tray_still_moving")

    def test_comb_still_moving_is_half(self):
        tr = Tracker(HZ)
        states = [(SET, comb(SET, dz=0.0012 * (-1) ** i)) for i in range(WINDOW + 1)]  # 2.4 mm up and down
        r = feed(tr, states)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["comb_reason"], "comb_still_moving")


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        self.assertEqual(feed(tr, [(SET, SEATED_END)] * WINDOW).score, 0.0)
        self.assertEqual(tr.update(SET, SEATED_END, FAR).score, 1.0)

    def test_comes_to_rest(self):
        tr = Tracker(HZ)
        moving = [(tray(dx=0.020 * (1 - i / 5)), comb(tray(dx=0.020 * (1 - i / 5)))) for i in range(6)]
        feed(tr, moving)
        self.assertEqual(feed(tr, [(SET, SEATED_END)] * (WINDOW - 1)).score, 0.0)
        self.assertEqual(feed(tr, [(SET, SEATED_END)] * 6).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        states = [(tray(dx=0.0008 * (-1) ** i), comb(tray(dx=0.0008 * (-1) ** i))) for i in range(WINDOW + 1)]
        self.assertEqual(feed(tr, states).score, 1.0)


class Exploits(unittest.TestCase):
    def test_lengthwise_tray_with_the_comb_in_an_end_slot(self):
        t = tray(yaw=90.0)
        self.assertEqual(scored(t, comb(t, "end_plus")), 0.0)

    def test_tray_held_down_between_the_walls(self):
        self.assertEqual(scored(SET, SEATED_END, HOLD_TRAY), 0.0)

    def test_comb_held_in_the_end_slot(self):
        self.assertEqual(scored(SET, SEATED_END, HOLD_COMB_IN_SLOT), 0.0)

    def test_comb_laid_on_the_wall_tops_over_the_end_slot(self):
        self.assertEqual(scored(SET, comb(SET, "end_minus", z_local=L.TRAY_H)), 0.5)

    def test_comb_seated_while_the_tray_is_on_the_bench(self):
        bench = raw((0.20, 0.20, L.TOP), quat(0.0))
        self.assertEqual(scored(bench, comb(bench, "end_minus")), 0.0)
        self.assertEqual(scored(START_TRAY, comb(START_TRAY, "end_minus")), 0.0)

    def test_tray_upside_down_crosswise_with_the_comb_on_it(self):
        t = tray(roll=180.0, up=L.TRAY_H)
        self.assertEqual(scored(t, raw(np.asarray(t.origin) + [0, 0, 0.001], quat(0.0))), 0.0)

    def test_comb_pushed_down_onto_the_floor(self):
        self.assertEqual(scored(SET, comb(SET, "end_minus", z_local=L.TRAY_FLOOR + L.COMB_TEETH[2])), 0.5)

    def test_arms_unreadable_cannot_be_used_to_score_a_lengthwise_tray(self):
        t = tray(yaw=90.0)
        self.assertEqual(scored(t, comb(t), None), 0.0)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr, r = settled(SET, SEATED_END, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)
        self.assertIsNone(r.details["zone_gap_mm"])

    def test_non_finite_arms_count_as_unavailable(self):
        bad = {"left": arm((math.nan, 0, 1)), "right": RIGHT_PARK}
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": LEFT_PARK}))
        self.assertFalse(arms_valid({"left": LEFT_PARK, "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3),
                                                                                   opening=math.inf)}))
        tr, r = settled(SET, SEATED_END, bad)
        self.assertEqual(r.details["arms_unavailable"], WINDOW + 1)

    def test_hands_off_applies_when_the_arms_come_back(self):
        tr, _ = settled(SET, SEATED_END, None)
        self.assertEqual(tr.update(SET, SEATED_END, HOLD_TRAY).score, 0.0)
        self.assertEqual(tr.update(SET, SEATED_END, FAR).score, 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(SET, SEATED_END)
        n = tr.steps
        th, ch = [list(map(np.copy, s)) for s in tr.tray_hist], [list(map(np.copy, s)) for s in tr.comb_hist]
        bad = [
            raw((math.nan, 0.0, 0.8), quat(0.0)),
            raw((0.32, math.inf, 0.8), quat(0.0)),
            raw((0.32, 0.0, 0.8), (math.nan, 0, 0, 1)),
            raw((0.32, 0.0, 0.8), (0.0, 0.0, 0.0, 0.0)),
            SimpleNamespace(quat_xyzw=quat(0.0), pos=np.zeros(3), lin_vel=np.zeros(3)),  # no origin
            SimpleNamespace(origin=np.zeros(2), quat_xyzw=quat(0.0), pos=np.zeros(3)),  # malformed
            None,
        ]
        for b in bad:
            for pair in ((b, SEATED_END), (SET, b)):
                with self.subTest(b=b, which=pair[0] is b):
                    self.assertFalse(valid(b))
                    r = tr.update(*pair, FAR)
                    self.assertEqual(r.score, 0.0)
                    self.assertFalse(r.success)
                    self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.steps, n + 2 * len(bad))
        for got, want in ((tr.tray_hist, th), (tr.comb_hist, ch)):
            self.assertEqual(len(got), len(want))
            for a, b in zip(got, want):
                for p, q in zip(a, b):
                    np.testing.assert_array_equal(p, q)
        self.assertEqual(tr.update(SET, SEATED_END, FAR).score, 1.0)  # and it carries on

    def test_unnormalised_quaternion(self):
        t = raw(SET.origin, 3.0 * np.asarray(SET.quat_xyzw))
        c = raw(SEATED_END.origin, 0.6 * np.asarray(SEATED_END.quat_xyzw))
        self.assertEqual(scored(t, c), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker(HZ)
        for _ in range(500):
            o1 = rng.uniform([-0.2, -0.5, 0.6], [0.8, 0.5, 1.2])
            o2 = rng.uniform([-0.2, -0.5, 0.6], [0.8, 0.5, 1.2])
            a = {"left": arm(rng.uniform(-1, 1, 3), rng.uniform(-1, 2)), "right": arm(rng.uniform(-1, 1, 3))}
            r = tr.update(raw(o1, rng.normal(size=4)), raw(o2, rng.normal(size=4)), a if rng.random() < 0.8 else None)
            self.assertIn(r.score, (0.0, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(SET, SEATED_END)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START_TRAY, START_COMB, FAR)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.tray_hist), 1)
        self.assertEqual(r.score, 0.0)

    def test_reset_then_a_new_trial(self):
        tr, _ = settled(SET, SEATED_END)
        tr.update(START_TRAY, START_COMB, FAR)
        self.assertEqual(feed(tr, [(SET, SEATED_END)] * WINDOW).score, 0.0)  # not yet 0.5 s of the new trial seen
        self.assertEqual(tr.update(SET, SEATED_END, FAR).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(SET, SEATED_END)
        tr.update(raw((L.TRAY_START[0] + 0.03, L.TRAY_START[1] - 0.03, L.TOP), quat(90.0)), START_COMB, FAR)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(SET, SEATED_END)
        a, b = np.asarray(SET.origin), np.asarray(START_TRAY.origin)
        feed(tr, [(raw(a + (b - a) * i / 40, quat(90.0)), START_COMB) for i in range(1, 41)])
        self.assertEqual(tr.steps, WINDOW + 1 + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(SET, SEATED_END)
        t = raw((0.20, 0.0, L.TOP), quat(0.0))
        tr.update(t, comb(t), FAR)
        self.assertEqual(tr.steps, WINDOW + 2)

    def test_explicit_reset(self):
        tr, _ = settled(SET, SEATED_END, None)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.tray_hist), len(tr.comb_hist), tr.arms_unavailable), (0, 0, 0, 0))
        self.assertIsNone(tr.prev_origin)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, offline_nest,
    the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        import offline_nest
        import rigstate
        cls.mk, cls.rig, cls.nest = mechkit, rigstate, offline_nest

    def sim(self, t, c, source="sim"):
        """Synthetic states turned into the AssemblyStates mechkit would report."""
        return {"gel_tray": self.mk.state_from_poses(self.task.TRAY, (*t.origin, *t.quat_xyzw), source=source),
                "comb": self.mk.state_from_poses(self.task.COMB, (*c.origin, *c.quat_xyzw), source=source)}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], ["gel_tray", "comb"])
        self.assertAlmostEqual(t.TRAY.mass, 0.038)
        self.assertAlmostEqual(t.COMB.mass, 0.020)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        by = {o.name: o for o in t.OBJECTS}
        near, far = by["tank_side_near"], by["tank_side_far"]
        self.assertAlmostEqual((far.pos[0] - far.size[0] / 2) - (near.pos[0] + near.size[0] / 2), 0.094)
        self.assertAlmostEqual(near.pos[2] + near.size[2] / 2, L.TOP + L.TANK_H)
        p = by["casting_platform"]
        self.assertAlmostEqual(p.pos[2] + p.size[2] / 2, PLAT_W)
        self.assertAlmostEqual(p.size[0], 0.094)
        a, b = by["comb_stand_near"], by["comb_stand_far"]
        self.assertAlmostEqual((b.pos[0] - b.size[0] / 2) - (a.pos[0] + a.size[0] / 2), L.STAND_GAP)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                lo = np.asarray(o.pos) - np.asarray(o.size) / 2
                hi = np.asarray(o.pos) + np.asarray(o.size) / 2
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)

    def test_tray_geometry_matches_the_scorer(self):
        tr = self.task.TRAY
        lo, hi = tr.local_bounds()
        np.testing.assert_allclose(lo, [-0.042, -0.032, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.042, 0.032, 0.018], atol=1e-12)

        def solid_at(p, gaskets=True):
            return any(np.all(np.abs(np.asarray(p) - np.asarray(s.center)) < np.asarray(s.size) / 2)
                       for s in tr.shapes if gaskets or tuple(s.size) != L.GASKET_SIZE)

        for name, sx in L.SLOTS.items():  # each slot: a notch 22 mm wide and 5 mm deep in both walls' tops
            for sy in (-1, 1):
                y = sy * L.WALL_Y
                self.assertFalse(solid_at((sx, y, 0.0155)), name)
                self.assertFalse(solid_at((sx + 0.0109, y, 0.0131)), name)
                self.assertFalse(solid_at((sx - 0.0109, y, 0.0131)), name)
                self.assertTrue(solid_at((sx + 0.0111, y, 0.0155)), name)
                self.assertTrue(solid_at((sx - 0.0111, y, 0.0155)), name)
                self.assertTrue(solid_at((sx, y, 0.0129)), name)
        self.assertFalse(solid_at((0.0, 0.0, 0.010)))  # open inside
        self.assertTrue(solid_at((0.0, 0.0, 0.002)))  # the floor
        self.assertTrue(solid_at((0.041, 0.0, 0.010)))  # a gasket on each open end
        self.assertTrue(solid_at((-0.041, 0.031, 0.002)))
        self.assertFalse(solid_at((0.041, 0.0, 0.0145)))  # 14 mm tall
        gaskets = [s for s in tr.shapes if tuple(s.size) == L.GASKET_SIZE]
        self.assertEqual(len(gaskets), 2)
        self.assertTrue(all(s.color == L.GASKET_BLACK for s in gaskets))
        acrylic = sum(float(np.prod(s.size)) for s in tr.shapes if tuple(s.size) != L.GASKET_SIZE)
        rubber = sum(float(np.prod(s.size)) for s in gaskets)
        self.assertAlmostEqual(acrylic * 1190 + rubber * 1200, L.TRAY_MASS, delta=0.002)
        com = tr.center_of_mass()
        self.assertLess(math.hypot(com[0], com[1]), 1e-6)  # under the grip at its middle
        self.assertTrue(Tracker.at_start(tr.rest_state()))

    def test_comb_geometry_matches_the_scorer(self):
        c = self.task.COMB
        lo, hi = c.local_bounds()
        np.testing.assert_allclose(lo, [-0.006, -0.036, -0.007], atol=1e-12)
        np.testing.assert_allclose(hi, [0.006, 0.036, 0.018], atol=1e-12)
        np.testing.assert_allclose((lo + hi) / 2, L.COMB_BOX_CENTER, atol=1e-12)
        np.testing.assert_allclose((hi - lo) / 2, L.COMB_BOX_HALF, atol=1e-12)
        vol = sum(float(np.prod(s.size)) for s in c.shapes)
        self.assertAlmostEqual(vol * 1190, L.COMB_MASS, delta=0.001)
        # in its stand: the bar on both plates' tops, the teeth in the gap with 1.5 mm a side, clear of the table
        st = c.rest_state()
        self.assertAlmostEqual(st.origin[2], L.TOP + L.STAND_PLATE[2])
        self.assertGreater(st.low[2] - L.TOP, 0.010)
        inner = L.STAND_PLATE_X[0] + L.STAND_PLATE[0] / 2, L.STAND_PLATE_X[1] - L.STAND_PLATE[0] / 2
        self.assertAlmostEqual(inner[0] - (L.STAND_X - L.COMB_TEETH[0] / 2), -0.0015)
        self.assertGreater((L.STAND_X - L.COMB_BAR[0] / 2), L.STAND_PLATE_X[0] - L.STAND_PLATE[0] / 2)
        self.assertLess((L.STAND_X - L.COMB_BAR[0] / 2), inner[0] - 0.002)  # the bar rests 3 mm onto each plate

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.TRAY.offline_box()[0]), quat_xyzw=quat(90.0),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.TRAY.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.TRAY_START[0], L.TRAY_START[1], L.TOP], atol=1e-12)
        self.assertTrue(Tracker.at_start(st))

    def test_offline_nest(self):
        t = SET
        on_box = PLAT_W + L.TRAY_H + L.COMB_TEETH[2]  # the comb's origin when the checker sets its box on the tray's

        def proxy(dx=0.0, dy=0.0, yaw=0.0, slot="end_minus"):
            c = comb(t, slot, dx=dx, dy=dy, yaw=yaw)
            return raw((c.origin[0], c.origin[1], on_box), c.quat_xyzw)

        cases = (("end_minus", 0.0, "end_minus", 1.0), ("middle", 0.0, "middle", 0.75),
                 ("end_plus", 0.0049, "end_plus", 1.0), ("end_minus", 0.0055, "wall_tops", 0.5),
                 ("middle", -0.0125, "wall_tops", 0.5))
        for slot, dx, on, want in cases:
            with self.subTest(slot=slot, dx=dx):
                s = self.sim(t, proxy(dx=dx, slot=slot), source="offline")
                fixed, info = self.nest.correct(s["gel_tray"], s["comb"], self.task.COMB)
                self.assertEqual(info["comb_on"], on)
                self.assertEqual(scored(s["gel_tray"], fixed), want)
        # not from the offline checker, a comb nowhere near the tray's top, or turned along the tray: left alone
        s = self.sim(t, proxy())
        self.assertIs(self.nest.correct(s["gel_tray"], s["comb"], self.task.COMB)[0], s["comb"])
        s = self.sim(t, START_COMB, source="offline")
        self.assertIsNone(self.nest.correct(s["gel_tray"], s["comb"], self.task.COMB)[1]["comb_on"])
        s = self.sim(t, proxy(yaw=90.0), source="offline")
        self.assertIsNone(self.nest.correct(s["gel_tray"], s["comb"], self.task.COMB)[1]["comb_on"])
        s = self.sim(t, proxy(dy=0.008), source="offline")
        self.assertIsNone(self.nest.correct(s["gel_tray"], s["comb"], self.task.COMB)[1]["comb_on"])

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        start = self.sim(START_TRAY, START_COMB)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        done = self.sim(SET, SEATED_END)
        for _ in range(WINDOW):
            self.feed_arms(FAR)
            t.success(done)
        self.feed_arms(FAR)
        self.assertTrue(t.success(done))
        self.assertEqual(t.score(done), 1.0)
        self.assertEqual(t.score(done), 1.0)  # score() does not advance it
        self.feed_arms(HOLD_COMB_IN_SLOT)
        self.assertEqual(t.score(done), 0.0)  # read with the arms of the step at hand
        self.feed_arms(FAR)
        n = tr.steps
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(FAR)
        t.success(done)
        self.assertEqual(tr.steps, 1)
        t.success(done)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(done))
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
                self.task.success(self.sim(SET, SEATED_END))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "tray_set", "tray_pose_ok", "tray_at_rest",
                          "tray_moved_mm", "comb_slot", "comb_slot_pose", "comb_reason", "comb_at_rest",
                          "comb_moved_mm", "clear_of_tray", "clear_of_comb", "zone_gap_mm", "arms_unavailable",
                          "tray_tilt_deg", "cross_err_deg", "corner_gap_mm", "gasket_gap_mm", "tray_centre_mm",
                          "tray_overhang_mm",
                          "comb_seat_x_mm", "comb_seat_z_mm", "comb_mid_y_mm", "comb_heading_deg", "comb_tilt_deg",
                          "tray_origin", "tray_quat", "comb_origin", "comb_quat", "source", "right_tcp", "openings"):
                    self.assertIn(k, row)
                np.testing.assert_allclose(row["gasket_gap_mm"], [5.0, 5.0])
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(SET, SEATED_END)))  # no crash
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
            # every close and open takes 1.5 s; so does every final descent onto a part or into place
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))
                if w.label.startswith("slowly"):
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))
        lb = {w.label: w for w in plan["left"]}
        rb = {w.label: w for w in plan["right"]}
        grip, carry = lb["grip the tray"], lb["carry it over the tank, turning it crosswise"]
        place, letgo = lb["slowly down between the side walls onto the platform"], lb["let go of the tray"]
        # picked lengthwise with the fingers along x, set down crosswise: a quarter turn of the grip turns the tray
        self.assertAlmostEqual(grip.yaw, -math.pi / 2)
        self.assertAlmostEqual(carry.yaw, 0.0)
        r0, r1 = _rot(quat(90.0)), _rot(quat(0.0))
        np.testing.assert_allclose(abs((r1 @ _rot(quat(90.0)).T @ r0)[:, 0] @ [1, 0, 0]), 1.0, atol=1e-12)
        self.assertEqual((grip.pos[0], grip.pos[1]), L.TRAY_START)
        self.assertEqual((place.pos[0], place.pos[1]), (L.TANK_X, L.TANK_Y))
        self.assertEqual(letgo.pos, place.pos)
        # pad tips 5 mm above the tray's underside at the pick and when set down; let go 0.5 mm above the platform
        self.assertAlmostEqual(grip.pos[2] - 0.013 - L.TOP, 0.005)
        self.assertAlmostEqual(place.pos[2] - t.TRAY_GRIP_UP - PLAT_W, 0.0005)
        self.assertGreater(place.pos[2] - 0.013 - PLAT_W, 0.005)  # pad tips above the platform
        self.assertGreater(lb["down to just above the walls"].pos[2] - t.TRAY_GRIP_UP, L.TOP + L.TANK_H)
        self.assertGreater(carry.pos[2] - t.TRAY_GRIP_UP, L.TOP + L.TANK_H + 0.03)
        # 56 mm commanded across the 64 mm tray; the fingers (27 mm wide) stay 33 mm clear of the tank's side walls
        self.assertAlmostEqual(-2 * t.TRAY_GRIP, L.TRAY_W - 0.008)
        self.assertGreater(min(place.pos[0] - 0.0135 - L.SIDE_IN[0], L.SIDE_IN[1] - place.pos[0] - 0.0135), 0.033)
        # the comb: picked by its bar's ends, never turned, seated in the slot nearer the arms 0.5 mm up
        cgrip, cset = rb["grip the comb"], rb["slowly down into the notches"]
        self.assertEqual(cgrip.yaw, 0.0)
        self.assertEqual(cset.yaw, 0.0)
        self.assertAlmostEqual(-2 * t.COMB_GRIP, L.COMB_BAR[1] - 0.008)
        self.assertAlmostEqual(cset.pos[0], L.TANK_X - 0.025)
        self.assertAlmostEqual(cset.pos[2] - t.COMB_GRIP_UP - (PLAT_W + L.NOTCH_BOTTOM), 0.0005)
        self.assertGreater(cset.pos[2] - 0.013, PLAT_W + L.TRAY_H)  # pad tips above the tray's walls
        self.assertAlmostEqual(cset.pos[0] - 0.0135 - L.SIDE_IN[0], 0.0085)  # fingers 8.5 mm from the near side wall
        self.assertEqual(rb["let go of the comb"].pos, cset.pos)
        entry = rb["down to just above the tray"]
        self.assertGreater(entry.pos[2] - t.COMB_GRIP_UP - L.COMB_TEETH[2], PLAT_W + L.TRAY_H + 0.012)

        # the right arm comes over the tank only after the left arm has let go and gone up
        def end_time(wps, label):
            s = 0.0
            for w in wps:
                s += w.duration
                if w.label == label:
                    return s
            raise KeyError(label)
        start_over = end_time(plan["right"], "wait for the tray and the left arm")
        self.assertGreater(start_over, end_time(plan["left"], "straight up"))
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess(total * 20, 1200 * 0.6)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("crosswise", "gaskets", "side walls", "not toward the electrodes", "comb", "nearest one end",
                  "notches", "Let go of both"):
            self.assertIn(s, ins)


if __name__ == "__main__":
    unittest.main()
