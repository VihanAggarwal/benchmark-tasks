"""Offline tests for scoring.py: synthetic object and arm states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import itertools
import math
import pathlib
import sys
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
    credit,
    footprint,
    on_chest,
    outline_gap,
    pads_gap,
    released,
    tilt_deg,
    valid,
    zone_gap,
)

HZ = 20.0
DT = 1.0 / HZ
SX, SY = L.TARGETS["sternal"]
AX, AY = L.TARGETS["apical"]
CT = L.CHEST_TOP


def quat(yaw_deg=0.0, tilt_deg_=0.0):
    """Turned yaw about the vertical after tilting tilt about the body's x axis (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg_) / 2
    (x1, y1, z1, w1) = (0.0, 0.0, math.sin(a), math.cos(a))
    (x2, y2, z2, w2) = (math.sin(b), 0.0, 0.0, math.cos(b))
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pad(x, y, low_z, yaw=0.0, tilt=0.0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """A pad state: centre (x, y), lowest point at low_z, as the simulator reports it (AABB of the turned box)."""
    q = quat(yaw, tilt)
    half = np.array(L.PAD_SIZE) / 2
    cs = np.array([rot(q) @ (half * np.array(s)) for s in itertools.product((-1, 1), repeat=3)])
    c = np.array([x, y, low_z - cs[:, 2].min()])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c + cs.min(axis=0), high=c + cs.max(axis=0))


def start_pad(name):
    x, y, yaw = L.PAD_START[name]
    return pad(x, y, L.TOP, yaw=math.degrees(yaw))


def on(x, y, yaw=0.0, **kw):
    """A pad lying flat on the chest at (x, y)."""
    return pad(x, y, CT, yaw=yaw, **kw)


def arm(tcp, opening=1.0, tips=None):
    tcp = np.array(tcp, float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0, 0, 0.013]) if tips is None else np.array(tips, float),
                           opening=opening)


AWAY = {"left": arm((0.24, 0.24, L.TOP + 0.16)), "right": arm((0.24, -0.24, L.TOP + 0.16))}
READY_ARMS = {"left": arm((0.20, 0.305, L.TOP + 0.25)), "right": arm((0.20, -0.305, L.TOP + 0.25))}


def holding(p, opening=None):
    """A gripper closed on pad p across its 70 mm width (TCP 18 mm above its underside)."""
    c = np.array(p.pos, float)
    tcp = (c[0], c[1], float(p.low[2]) + 0.018)
    return arm(tcp, (L.PAD_SIZE[1] - 0.008) / L.FULL_GAP if opening is None else opening)


def pads(left, right):
    return {"pad_left": left, "pad_right": right}


START = pads(start_pad("pad_left"), start_pad("pad_right"))
DONE = pads(on(SX, SY, yaw=90.0), on(AX, AY))


def score(p, arms=AWAY):
    return Tracker(HZ).result(p, arms)


def run(seq, t=None):
    """Feed (pads, arms) pairs to a tracker (a fresh one unless given); return the last result and the tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for p, a in seq:
        r = t.update(p, a)
    return r, t


def glide(a, b, n, low=(None, None), yaw=0.0):
    """n pad states moving from (x, y, low_z) a to b in equal steps, with the matching velocity."""
    a, b = np.array(a, float), np.array(b, float)
    v = (b - a) / (n * DT)
    return [pad(*((1 - i / n) * a + (i / n) * b), yaw=yaw, vel=v) for i in range(1, n + 1)]


def at(dist, target, bearing_deg=0.0, yaw=0.0):
    """A pad lying on the chest with its centre `dist` from a target, along a bearing."""
    tx, ty = L.TARGETS[target]
    b = math.radians(bearing_deg)
    return on(tx + dist * math.cos(b), ty + dist * math.sin(b), yaw=yaw)


class Geometry(unittest.TestCase):
    def test_tilt_either_face(self):
        self.assertAlmostEqual(tilt_deg(quat(30.0, 7.0)), 7.0, places=4)
        self.assertAlmostEqual(tilt_deg(quat(0.0, 180.0)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(quat(0.0, 170.0)), 10.0, places=4)

    def test_lying_on_the_chest(self):
        self.assertTrue(on_chest(on(SX, SY)))
        self.assertTrue(on_chest(pad(SX, SY, CT + 0.0049)))
        self.assertFalse(on_chest(pad(SX, SY, CT + 0.0052)))  # held above it
        self.assertTrue(on_chest(pad(SX, SY, CT - 0.004)))
        self.assertTrue(on_chest(pad(SX, SY, CT + L.MARK_T, tilt=1.0)))  # one edge on a 1 mm landmark line
        self.assertTrue(on_chest(on(SX, SY, tilt=9.9)))
        self.assertFalse(on_chest(on(SX, SY, tilt=10.5)))
        self.assertFalse(on_chest(on(SX, SY, tilt=90.0)))  # standing on its long edge
        self.assertTrue(on_chest(on(L.TORSO_X1 - 0.001, 0.0)))  # centre just over the shoulder line
        self.assertFalse(on_chest(on(L.TORSO_X1 + 0.002, 0.0)))
        self.assertFalse(on_chest(on(AX, -L.TORSO_HALF_Y - 0.002)))  # centre past the left edge
        self.assertFalse(on_chest(pad(AX, -0.25, L.TOP)))  # on the table beside the torso
        self.assertAlmostEqual(bottom_gap(on(SX, SY)), 0.0, places=9)

    def test_outlines(self):
        self.assertEqual(footprint(on(SX, SY)).shape, (4, 2))
        self.assertEqual(footprint(on(SX, SY, yaw=30.0, tilt=8.0)).shape, (6, 2))
        a, b = on(0.30, 0.10), on(0.30, 0.10 - 0.070 - 0.015)  # side by side along y, 15 mm apart
        self.assertAlmostEqual(outline_gap(footprint(a), footprint(b)), 0.015, places=6)
        c = on(0.30 + 0.110 + 0.008, 0.10)  # end to end along x, 8 mm apart
        self.assertAlmostEqual(outline_gap(footprint(a), footprint(c)), 0.008, places=6)
        self.assertEqual(outline_gap(footprint(a), footprint(on(0.32, 0.12))), 0.0)  # overlapping
        self.assertEqual(outline_gap(footprint(a), footprint(pad(0.30, 0.10, CT + 0.018))), 0.0)  # stacked
        reach45 = (0.055 + 0.035) * math.sqrt(0.5)  # a pad turned 45 deg reaches 63.6 mm along x (a corner)
        d = on(0.30 + 0.055 + 0.020 + reach45, 0.10, yaw=45.0)  # that corner 20 mm from a's end face
        self.assertAlmostEqual(outline_gap(footprint(a), footprint(d)), 0.020, places=6)
        self.assertGreater(pads_gap(DONE), 0.10)
        self.assertGreater(pads_gap(START), 0.40)

    def test_credit_rings(self):
        self.assertEqual(credit(0.0), 0.5)
        self.assertEqual(credit(0.025), 0.5)
        self.assertEqual(credit(0.0251), 0.25)
        self.assertEqual(credit(0.040), 0.25)
        self.assertEqual(credit(0.0401), 0.0)

    def test_targets_are_far_apart(self):
        d = math.hypot(SX - AX, SY - AY)
        self.assertGreater(d, 2 * L.FAR + 0.1)  # one pad can earn credit at one target only

    def test_finger_zone(self):
        p = on(SX, SY, yaw=90.0)
        top = CT + L.PAD_SIZE[2]
        r_open = math.hypot(L.FULL_GAP / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH)
        above = arm((SX, SY, top + 0.013 + r_open + 0.012))  # open, tips 12 mm beyond the zone's reach
        self.assertAlmostEqual(zone_gap(above, p), 0.012, places=6)
        self.assertTrue(released({"left": above, "right": AWAY["right"]}, p))
        lower = arm((SX, SY, top + 0.013 + r_open + 0.008))
        self.assertFalse(released({"left": lower, "right": AWAY["right"]}, p))
        self.assertLess(zone_gap(holding(p), p), 0.0)
        self.assertTrue(released(AWAY, p))
        self.assertTrue(released(READY_ARMS, on(AX, AY)))

    def test_tape_measure_form_of_the_clearance(self):
        """task.json: for an upright gripper, the tip point at least 67 mm from the pad fully open, 26 mm closed."""
        p = on(SX, SY, yaw=90.0)
        top = CT + L.PAD_SIZE[2]
        for opening, need in ((1.0, 0.067), (0.0, 0.026)):
            # the pad lies across (yaw 90): 70 mm along x, 110 mm along y. Tips above it, beside its end, and out
            # from its top edge at 45 degrees.
            for tips in ((SX, SY, top + need), (SX, SY + 0.055 + need, CT + 0.009),
                         (SX + 0.035 + need * math.sqrt(0.5), SY, top + need * math.sqrt(0.5))):
                a = arm(np.array(tips) + (0, 0, 0.013), opening, tips=tips)
                self.assertTrue(released({"left": a, "right": AWAY["right"]}, p), (opening, tips))
            close = (SX, SY, top + need - 0.002)
            a = arm(np.array(close) + (0, 0, 0.013), opening, tips=close)
            self.assertFalse(released({"left": a, "right": AWAY["right"]}, p), opening)

    def test_apical_pad_against_the_chest_edge_is_10mm_off(self):
        """README: a pad at the apical level with its outer edge on the chest's left edge, either way round."""
        along = on(AX, -L.TORSO_HALF_Y + L.PAD_SIZE[1] / 2)  # long side along x
        across = on(AX, -L.TORSO_HALF_Y + L.PAD_SIZE[0] / 2, yaw=90.0)  # long side along y
        for p in (along, across):
            self.assertAlmostEqual(math.hypot(p.pos[0] - AX, p.pos[1] - AY), 0.010, places=6)
            self.assertEqual(score(pads(on(SX, SY, yaw=90.0), p)).score, 1.0)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = score(START, READY_ARMS)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_both_placed_scores_one(self):
        r = score(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual((r.details["sternal_credit"], r.details["apical_credit"]), (0.5, 0.5))

    def test_within_25mm_any_direction_scores_one(self):
        for b in range(0, 360, 45):
            r = score(pads(at(0.0245, "sternal", b, yaw=90.0), at(0.0245, "apical", b + 90)))
            self.assertEqual(r.score, 1.0, b)

    def test_either_pad_either_target_any_heading_either_face(self):
        self.assertEqual(score(pads(on(AX, AY), on(SX, SY, yaw=90.0))).score, 1.0)  # swapped
        self.assertEqual(score(pads(on(SX, SY, yaw=17.0), on(AX, AY, yaw=-63.0))).score, 1.0)
        self.assertEqual(score(pads(on(SX, SY, tilt=180.0), on(AX, AY, tilt=180.0))).score, 1.0)

    def test_three_quarters(self):
        r = score(pads(at(0.010, "sternal", yaw=90.0), at(0.032, "apical", 180.0)))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        r = score(pads(at(0.035, "sternal", 90.0), at(0.0, "apical")))
        self.assertEqual(r.score, 0.75)

    def test_half(self):
        self.assertEqual(score(pads(on(SX, SY, yaw=90.0), start_pad("pad_right"))).score, 0.5)  # one pad only
        self.assertEqual(score(pads(start_pad("pad_left"), on(AX, AY))).score, 0.5)
        self.assertEqual(score(pads(at(0.030, "sternal", 0.0), at(0.038, "apical", 270.0))).score, 0.5)

    def test_quarter(self):
        self.assertEqual(score(pads(at(0.036, "sternal", 200.0), start_pad("pad_right"))).score, 0.25)
        self.assertEqual(score(pads(at(0.060, "sternal"), at(0.039, "apical", 45.0))).score, 0.25)

    def test_zero_off_target(self):
        self.assertEqual(score(pads(at(0.045, "sternal"), at(0.050, "apical", 90.0))).score, 0.0)
        self.assertEqual(score(pads(on(0.30, 0.05), on(0.25, -0.05))).score, 0.0)  # both on the chest, wrong places

    def test_every_tier_is_reachable(self):
        dists = {0.5: 0.010, 0.25: 0.032, 0.0: 0.080}
        seen = set()
        for cs, ca in itertools.product(dists, repeat=2):
            r = score(pads(at(dists[cs], "sternal", 0.0, yaw=90.0), at(dists[ca], "apical", 270.0)))
            self.assertEqual(r.score, cs + ca)
            seen.add(r.score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_success_needs_stillness(self):
        moving = pads(on(SX, SY, yaw=90.0, vel=(0.02, 0, 0)), on(AX, AY))
        r = score(moving)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r = score(pads(on(SX, SY, yaw=90.0), on(AX, AY, spin=(0, 0, 1.0))))
        self.assertFalse(r.success)

    def test_score_is_idempotent(self):
        t = Tracker(HZ)
        run([(DONE, AWAY)] * 5, t)
        a, b = t.result(DONE, AWAY), t.result(DONE, AWAY)
        self.assertEqual((a.score, a.details), (b.score, b.details))
        self.assertEqual(t.steps, 5)
        self.assertEqual(t.first_full_step, 1)


class ZeroConditions(unittest.TestCase):
    def test_pads_closer_than_10mm_score_zero(self):
        a = on(SX, SY, yaw=90.0)  # across: x 0.365 .. 0.435
        b = on(SX - 0.070 - 0.008, SY, yaw=90.0)  # 8 mm behind it, toward the arms
        self.assertAlmostEqual(pads_gap(pads(a, b)), 0.008, places=6)
        self.assertEqual(score(pads(a, b)).score, 0.0)
        c = on(SX - 0.070 - 0.012, SY, yaw=90.0)  # 12 mm: not touching; the first pad's 0.5 stands
        self.assertEqual(score(pads(a, c)).score, 0.5)

    def test_touching_or_overlapping_scores_zero(self):
        self.assertEqual(score(pads(on(SX, SY, yaw=90.0), on(SX - 0.070, SY, yaw=90.0))).score, 0.0)
        self.assertEqual(score(pads(on(AX, AY), on(AX + 0.050, AY + 0.020))).score, 0.0)

    def test_stacked_scores_zero(self):
        stacked = pads(on(SX, SY, yaw=90.0), pad(SX + 0.003, SY, CT + L.PAD_SIZE[2], yaw=90.0))
        self.assertEqual(score(stacked).score, 0.0)

    def test_held_pad_does_not_count(self):
        p = on(SX, SY, yaw=90.0)
        r = score(pads(p, on(AX, AY)), {"left": holding(p), "right": AWAY["right"]})
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["pad_left_released"])
        self.assertFalse(r.success)
        q = on(AX, AY)
        both = {"left": holding(p), "right": holding(q)}
        self.assertEqual(score(pads(p, q), both).score, 0.0)

    def test_open_gripper_resting_on_or_beside_a_pad_does_not_count(self):
        q = on(AX, AY)
        over = arm((AX, AY, CT + L.PAD_SIZE[2] + 0.013))  # open, tips on the pad's top
        self.assertEqual(score(pads(on(SX, SY, yaw=90.0), q), {"left": AWAY["left"], "right": over}).score, 0.5)
        beside = arm((AX, AY - 0.045, CT + 0.020), opening=0.0)  # closed, fingers against its side
        self.assertEqual(score(pads(on(SX, SY, yaw=90.0), q), {"left": AWAY["left"], "right": beside}).score, 0.5)

    def test_held_above_the_chest_does_not_count(self):
        hover = pad(SX, SY, CT + 0.015, yaw=90.0)
        self.assertEqual(score(pads(hover, on(AX, AY)), {"left": holding(hover), "right": AWAY["right"]}).score,
                         0.5)
        self.assertEqual(score(pads(hover, on(AX, AY))).score, 0.5)  # even with the arms far away (not lying)

    def test_on_edge_or_tilted_scores_nothing_there(self):
        self.assertEqual(score(pads(on(SX, SY, yaw=90.0, tilt=90.0), on(AX, AY))).score, 0.5)
        self.assertEqual(score(pads(on(SX, SY, yaw=90.0), on(AX, AY, tilt=25.0))).score, 0.5)

    def test_off_the_chest_scores_nothing(self):
        self.assertEqual(score(pads(pad(SX, 0.25, L.TOP, yaw=90.0), pad(AX, -0.215, L.TOP))).score, 0.0)


class Exploits(unittest.TestCase):
    def test_mirror_image_placement_scores_zero(self):
        # The classic error: the pads on the arms' right and left instead of the manikin's (its upper LEFT chest
        # and its RIGHT side).
        mirrored = pads(on(SX, -SY, yaw=90.0), on(AX, -AY))
        r = score(mirrored)
        self.assertEqual(r.score, 0.0)

    def test_both_pads_at_one_position(self):
        # Side by side at the sternal position, 12 mm apart: only one can be within the rings.
        r = score(pads(on(SX, SY, yaw=90.0), on(SX - 0.082, SY, yaw=90.0)))
        self.assertEqual(r.score, 0.5)
        r = score(pads(on(SX, SY, yaw=90.0), pad(SX, SY, CT + L.PAD_SIZE[2], yaw=90.0)))  # stacked
        self.assertEqual(r.score, 0.0)

    def test_holding_both_in_place_to_the_end_scores_zero(self):
        p, q = on(SX, SY, yaw=90.0), on(AX, AY)
        seq = [(pads(p, q), {"left": holding(p), "right": holding(q)})] * 40
        r, t = run(seq)
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(t.first_full_step)

    def test_too_high_and_forward_apical_pad(self):
        # The apical pad put on the front of the chest under the nipple, not on the side: 60 mm off.
        r = score(pads(on(SX, SY, yaw=90.0), on(0.300, -0.085)))
        self.assertEqual(r.score, 0.5)
        # The sternal pad over the breastbone: 80 mm off.
        r = score(pads(on(SX, 0.0, yaw=90.0), on(AX, AY)))
        self.assertEqual(r.score, 0.5)

    def test_one_pad_bridging_both_rings_is_impossible(self):
        mid = on((SX + AX) / 2, (SY + AY) / 2, yaw=60.0)
        r = score(pads(mid, start_pad("pad_right")))
        self.assertEqual(r.score, 0.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([SX, SY, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]),
                              high=np.zeros(3))
        d = DONE["pad_left"]
        zero_q = SimpleNamespace(pos=d.pos, quat_xyzw=np.zeros(4), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                 low=d.low, high=d.high)
        for bad in (nan, inf, zero_q, None):
            self.assertFalse(valid(bad))
            t = Tracker(HZ)
            run([(DONE, AWAY)] * 3, t)
            prev = t.prev
            r = t.update(pads(bad, DONE["pad_right"]), AWAY)
            self.assertEqual((r.score, r.success), (0.0, False))
            self.assertEqual(t.result(pads(bad, DONE["pad_right"]), AWAY).score, 0.0)
            self.assertEqual(t.bad_steps, 1)
            self.assertIs(t.prev, prev)
            r, _ = run([(DONE, AWAY)] * 3, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)
            self.assertEqual(t.steps, 7)

    def test_missing_pad_scores_zero(self):
        r = Tracker(HZ).update({"pad_left": DONE["pad_left"]}, AWAY)
        self.assertEqual(r.score, 0.0)

    def test_unreadable_arms_are_taken_as_released_and_counted(self):
        bad_arm = {"left": arm((np.nan, 0, 0)), "right": AWAY["right"]}
        for arms in (None, {}, {"left": AWAY["left"]}, bad_arm, {"left": arm((0, 0, 1), opening=np.nan),
                                                                  "right": AWAY["right"]}):
            self.assertFalse(arms_valid(arms))
            r, t = run([(DONE, arms)] * 4)
            self.assertEqual(r.score, 1.0)
            self.assertEqual(t.arms_unavailable, 4)
            self.assertIsNone(r.details["pad_left_released"])
        self.assertTrue(arms_valid(AWAY))

    def test_states_anywhere_do_not_crash(self):
        odd = [pad(5.0, -3.0, -1.0), pad(SX, SY, 0.0, tilt=120.0, yaw=33.0), pad(-0.2, 0.6, L.TOP, tilt=90.0),
               pad(AX, AY, CT - 0.05, tilt=89.9), pad(SX, SY, 2.0, vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        for a, b in zip(odd + odd[::-1], odd[::-1] + odd):
            r = t.update(pads(a, b), AWAY)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
        far_arms = {"left": arm((9, 9, 9), opening=0.0), "right": arm((-9, 0, 0), tips=(-9, 0, 0))}
        self.assertIn(t.update(DONE, far_arms).score, (1.0,))


class Reset(unittest.TestCase):
    def _trial(self):
        lifted = glide((L.PAD_START["pad_left"][0], L.PAD_START["pad_left"][1], L.TOP), (SX, SY, CT + 0.03), 40,
                       yaw=90.0)
        seq = [(START, READY_ARMS)] * 5 + [(pads(p, START["pad_right"]), AWAY) for p in lifted]
        return seq + [(pads(on(SX, SY, yaw=90.0), on(AX, AY)), AWAY)] * 10

    def test_teleport_back_to_start_resets(self):
        r, t = run(self._trial())
        self.assertEqual(r.score, 1.0)
        self.assertIsNotNone(t.first_full_step)
        r = t.update(START, READY_ARMS)  # both pads jump back in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertIsNone(t.first_full_step)
        self.assertEqual(r.score, 0.0)
        r, _ = run(self._trial()[1:], t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_a_seeded_reset_near_the_marks_resets(self):
        _, t = run(self._trial())
        jit = pads(pad(0.23, 0.24, L.TOP, yaw=90.0), pad(0.17, -0.30, L.TOP))
        t.update(jit, READY_ARMS)
        self.assertEqual(t.steps, 1)

    def test_carrying_them_back_slowly_does_not_reset(self):
        seq = self._trial()
        back = glide((SX, SY, CT + 0.03), (L.PAD_START["pad_left"][0], L.PAD_START["pad_left"][1], L.TOP), 40,
                     yaw=90.0)
        seq += [(pads(p, on(AX, AY)), AWAY) for p in back]
        seq += [(pads(start_pad("pad_left"), p), AWAY)
                for p in glide((AX, AY, CT), (L.PAD_START["pad_right"][0], L.PAD_START["pad_right"][1], L.TOP), 40)]
        r, t = run(seq + [(START, AWAY)] * 5)
        self.assertEqual(r.score, 0.0)
        self.assertGreater(t.steps, 100)  # the same episode
        self.assertIsNotNone(t.first_full_step)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, t = run(self._trial())
        n = t.steps
        t.update(pads(pad(0.10, 0.45, L.TOP), START["pad_right"]), AWAY)  # knocked far, not onto its mark
        self.assertEqual(t.steps, n + 1)


if __name__ == "__main__":
    unittest.main()
