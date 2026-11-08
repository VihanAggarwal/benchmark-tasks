"""Offline tests for scoring.py and the hooks in task.py: synthetic wheelchair states, no simulator.
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
    WINDOW, Tracker, alpha_deg, angle_credit, footprint, gap, heading_rad, lowest, tilt_deg, valid,
)

TOP = L.TOP
TIERS = {0.0, 0.25, 0.5, 0.75, 1.0}


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


def chair(cx, cy, heading, lift=0.0, pitch=0.0, roll=0.0, vel=(0, 0, 0)):
    """A chair state as mechkit reports it: frame origin (footprint centre) at (cx, cy), turned `heading` degrees
    and tilted by `pitch`/`roll`, resting with its lowest point `lift` above the floor (a tilted chair rests on an
    edge); low/high from its 95 x 63 x 85 mm box."""
    q = quat(heading, pitch, roll)
    local = np.array([(su * L.FOOT_U, sv * L.FOOT_V, z) for su in (-1, 1) for sv in (-1, 1) for z in (0.0, L.CHAIR_H)])
    pts = local @ rot(q).T
    o = np.array([cx, cy, TOP + lift - pts[:, 2].min()])
    pts = pts + o
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=lo, high=hi)


# The demo's nearest corner (the footprint's lowest y): 18 mm past the strong-side line, 22 mm short of Y_BESIDE
DEMO_MIN_Y = L.TARGET_CENTER[1] + min(dy for _, dy in L.corner_offsets(L.TARGET_ALPHA_DEG).values())


def placed(alpha=40.0, g=0.010, cy=None, min_y=None, **kw):
    """A chair beside the bed at angle alpha (degrees) to it with gap g; its centre at y cy, or placed so that its
    footprint's lowest y is min_y (default: the demo's, so at 40 degrees its centre is the demo's)."""
    off = L.corner_offsets(alpha)
    cx = L.BED_SIDE_X - g - max(dx for dx, _ in off.values())
    if cy is None:
        cy = (DEMO_MIN_Y if min_y is None else min_y) - min(dy for _, dy in off.values())
    return chair(cx, cy, 270.0 - alpha, **kw)


START = chair(L.CHAIR_START[0], L.CHAIR_START[1], 135.0)
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
    def test_angle_definition(self):
        for heading, want in ((270.0, 0.0), (225.0, 45.0), (230.0, 40.0), (180.0, 90.0), (90.0, 180.0),
                              (300.0, -30.0), (0.0, -90.0)):
            got = alpha_deg(chair(0.3, 0.2, heading))
            self.assertAlmostEqual(math.remainder(got - want, 360.0), 0.0, places=6, msg=heading)
        self.assertAlmostEqual(math.degrees(L.heading_of(40.0)), 230.0)

    def test_corner_offsets_match_the_footprint(self):
        for a in (0.0, 17.0, 40.0, 63.0, 90.0):
            s = placed(a)
            off = L.corner_offsets(a)
            fp = footprint(s)
            for i, name in enumerate(("front_left", "front_right", "rear_left", "rear_right")):
                np.testing.assert_allclose(fp[i], s.origin[:2] + np.array(off[name]), atol=1e-12)

    def test_gap_and_height(self):
        self.assertAlmostEqual(gap(TARGET), 0.010, places=9)
        self.assertAlmostEqual(gap(placed(65.0, 0.037)), 0.037, places=9)
        self.assertAlmostEqual(lowest(TARGET), 0.0, places=12)
        self.assertAlmostEqual(lowest(chair(0.3, 0.2, 230.0, lift=0.004)), 0.004, places=12)
        self.assertAlmostEqual(tilt_deg(chair(0.3, 0.2, 230.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(chair(0.3, 0.2, 230.0, roll=180.0)), 180.0, places=6)

    def test_start_layout_scores_zero(self):
        self.assertTrue(Tracker.at_start(START))
        r, _ = settled(START)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "weak_side_or_across_her")
        self.assertLess(footprint(START)[:, 1].max(), L.BED_Y0)  # past the foot of the bed
        self.assertTrue(np.all(np.abs(footprint(START)[:, 1]) < 0.45))  # clear of the arms' start-pose shadow

    def test_demo_target(self):
        r, _ = settled(TARGET)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        d = r.details
        self.assertAlmostEqual(d["alpha_deg"], 40.0)
        self.assertAlmostEqual(d["gap_mm"], 10.0)
        self.assertGreater(d["min_corner_y_mm"], 1000 * L.Y_STRONG + 15.0)  # 18 mm on her strong side of the line
        self.assertLess(d["min_corner_y_mm"], 1000 * L.Y_BESIDE - 20.0)  # 22 mm inside the 40 mm limit
        self.assertAlmostEqual(TARGET.origin[1], L.TARGET_CENTER[1], places=12)
        rear_left = TARGET.origin[1] + L.corner_offsets(40.0)["rear_left"][1]
        self.assertLess(rear_left, L.BED_Y1)  # its bed-side corner is beside the bed, not past its head end
        self.assertLess(TARGET.origin[1], L.HEAD_END_Y - 0.030)

    def test_the_handover_scores_zero(self):
        c = (L.HANDOVER_GRIP[0] + L.GRIP_U, L.HANDOVER_GRIP[1])  # facing the arms, heading 180: grip behind centre
        self.assertEqual(reason(chair(c[0], c[1], 180.0)), "weak_side_or_across_her")


class Tiers(unittest.TestCase):
    def test_worked_examples_in_task_json(self):
        cases = ((40.0, 0.010, 1.0), (20.0, 0.010, 0.75), (5.0, 0.010, 0.5), (40.0, 0.050, 0.5),
                 (20.0, 0.050, 0.25), (5.0, 0.050, 0.0), (85.0, 0.010, 0.5), (60.0, 0.010, 0.75))
        for a, g, want in cases:
            self.assertEqual(scored(placed(a, g)), want, (a, g))

    def test_angle_band_edges(self):
        for a, want in ((30.0, 0.5), (29.9, 0.25), (50.0, 0.5), (50.1, 0.25), (15.0, 0.25), (14.9, 0.0),
                        (65.0, 0.25), (65.1, 0.0), (0.0, 0.0), (90.0, 0.0)):
            self.assertEqual(angle_credit(a), want, a)
        self.assertEqual(scored(placed(30.05, 0.010)), 1.0)
        self.assertEqual(scored(placed(49.95, 0.010)), 1.0)
        self.assertEqual(scored(placed(29.5, 0.010)), 0.75)
        self.assertEqual(scored(placed(50.5, 0.010)), 0.75)
        self.assertEqual(scored(placed(14.5, 0.010)), 0.5)
        self.assertEqual(scored(placed(65.5, 0.010)), 0.5)

    def test_gap_edges(self):
        self.assertEqual(scored(placed(40.0, 0.0199)), 1.0)
        self.assertEqual(scored(placed(40.0, 0.0201)), 0.5)
        self.assertEqual(scored(placed(40.0, 0.0)), 1.0)  # touching the bed's side
        self.assertEqual(scored(placed(40.0, 0.0799)), 0.5)
        self.assertEqual(scored(placed(40.0, 0.0801)), 0.0)
        self.assertEqual(reason(placed(40.0, 0.0801)), "too_far_from_the_bed")

    def test_parallel_and_square_are_allowed_but_earn_no_angle(self):
        self.assertEqual(scored(placed(0.0, 0.010)), 0.5)
        self.assertEqual(scored(placed(90.0, 0.010)), 0.5)
        self.assertEqual(scored(placed(90.0, 0.050)), 0.0)

    def test_only_a_full_score_is_success(self):
        for a, g, want in ((20.0, 0.010, 0.75), (40.0, 0.050, 0.5), (20.0, 0.050, 0.25)):
            r, _ = settled(placed(a, g))
            self.assertEqual(r.score, want)
            self.assertFalse(r.success, (a, g))
        self.assertTrue(settled(TARGET)[0].success)

    def test_every_tier_is_reachable(self):
        seen = {scored(placed(a, g)) for a in (5.0, 20.0, 40.0, 60.0, 80.0) for g in (0.005, 0.040)}
        seen.add(scored(START))
        self.assertEqual(seen, TIERS)

    def test_score_is_read_from_the_state_at_hand(self):
        r, t = settled(TARGET)
        self.assertEqual(t.result(TARGET).score, 1.0)
        self.assertEqual(t.result(TARGET).score, 1.0)  # result() does not advance the tracker
        self.assertEqual(t.steps, WINDOW + 1)

    def test_random_states_map_to_a_tier(self):
        rng = random.Random(7)
        for _ in range(400):
            s = chair(rng.uniform(0.1, 0.5), rng.uniform(-0.3, 0.35), rng.uniform(-180, 180),
                      lift=rng.choice((0.0, 0.0, 0.002, 0.01)), pitch=rng.choice((0.0, 0.0, 5.0, 30.0)))
            self.assertIn(scored(s), TIERS)


class ZeroConditions(unittest.TestCase):
    def test_mirror_image_on_her_weak_side(self):
        a = 40.0
        off = L.corner_offsets(a)
        cx = L.BED_SIDE_X - 0.010 - max(dx for dx, _ in off.values())
        mirror = chair(cx, 2 * L.Y_P - L.TARGET_CENTER[1], 270.0 - (180.0 - a))  # front toward the head end, 40 out
        self.assertEqual(scored(mirror), 0.0)
        for h in range(0, 360, 15):  # her weak side at any angle
            self.assertEqual(scored(chair(0.330, 0.000, float(h))), 0.0, h)

    def test_strong_side_line_edge(self):
        self.assertEqual(scored(placed(40.0, 0.010, min_y=L.Y_STRONG + 0.0001)), 1.0)
        s = placed(40.0, 0.010, min_y=L.Y_STRONG - 0.0001)
        self.assertEqual(scored(s), 0.0)
        self.assertEqual(reason(s), "weak_side_or_across_her")

    def test_in_front_of_her_feet_or_across_her(self):
        self.assertEqual(scored(chair(0.300, 0.100, 230.0)), 0.0)  # straddling her centre line, in front of her
        self.assertEqual(scored(chair(0.300, 0.150, 270.0)), 0.0)  # parallel, its front reaching past her right hip

    def test_too_far_along_the_bed_from_her(self):
        # Its nearest corner at most 40 mm along the bed past her strong-side line, at every angle
        for a, want in ((0.0, 0.5), (16.0, 0.75), (40.0, 1.0), (64.0, 0.75), (90.0, 0.5)):
            self.assertEqual(scored(placed(a, 0.010, min_y=L.Y_BESIDE - 0.0001)), want, a)
            s = placed(a, 0.010, min_y=L.Y_BESIDE + 0.0001)
            self.assertEqual(reason(s), "too_far_along_the_bed_from_her", a)
            self.assertEqual(scored(s), 0.0, a)

    def test_the_40_mm_limit_keeps_it_short_of_the_head_end(self):
        # For any angle 0-90 the centre is at most 57 mm past the nearest corner: short of the bed's head end, so no
        # separate head-end rule is needed (README)
        for a in np.linspace(0.0, 90.0, 181):
            s = placed(float(a), 0.010, min_y=L.Y_BESIDE)
            self.assertLessEqual(s.origin[1], L.Y_BESIDE + L.TURN_RADIUS + 1e-12)
            self.assertLess(s.origin[1], L.HEAD_END_Y - 0.012)

    def test_past_the_bed_line(self):
        # A corner more than 3 mm into the bed's side (a broken state: the chair inside the bed) scores 0
        s = placed(40.0, -0.0031)
        self.assertEqual(reason(s), "past_the_bed_line")
        self.assertEqual(scored(placed(40.0, -0.0029)), 1.0)  # contact settling against the bed's side

    def test_turned_in_or_facing_the_head_end(self):
        self.assertEqual(reason(placed(-0.5, 0.010)), "turned_in_to_the_bed")
        self.assertEqual(reason(placed(-30.0, 0.010)), "turned_in_to_the_bed")
        self.assertEqual(reason(placed(90.5, 0.010)), "facing_the_head_end")
        self.assertEqual(reason(placed(140.0, 0.010)), "facing_the_head_end")
        self.assertEqual(scored(placed(180.0, 0.010)), 0.0)  # parallel but backwards

    def test_tipped(self):
        self.assertEqual(scored(placed(40.0, 0.010, pitch=9.9)), 1.0)
        self.assertEqual(reason(placed(40.0, 0.010, pitch=10.1)), "tipped")
        self.assertEqual(reason(placed(40.0, 0.010, roll=180.0)), "tipped")  # upside down
        self.assertEqual(reason(placed(40.0, 0.010, pitch=-90.0)), "tipped")  # on its back

    def test_off_the_floor(self):
        self.assertEqual(scored(placed(40.0, 0.010, lift=0.0029)), 1.0)
        self.assertEqual(reason(placed(40.0, 0.010, lift=0.0031)), "off_the_floor")
        self.assertEqual(reason(placed(40.0, 0.010, lift=L.BED_H)), "off_the_floor")  # resting on the bed's level
        self.assertEqual(reason(placed(40.0, 0.010, lift=0.030)), "off_the_floor")  # carried 30 mm up
        self.assertEqual(scored(placed(40.0, 0.010, lift=-0.0029)), 1.0)  # contact settling into the floor
        self.assertEqual(reason(placed(40.0, 0.010, lift=-0.0031)), "off_the_floor")  # sunk: a broken state

    def test_moving(self):
        t = Tracker()
        off = L.corner_offsets(40.0)
        cx = L.BED_SIDE_X - 0.010 - max(dx for dx, _ in off.values())
        for i in range(30):  # sliding along the bed at 20 mm/s: 1 mm a step, 10 mm over 0.5 s
            r = t.update(chair(cx, 0.180 + 0.001 * i, 230.0))
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
        off = L.corner_offsets(40.0)
        cx = L.BED_SIDE_X - 0.010 - max(dx for dx, _ in off.values())
        for i in range(40):  # 0.4 mm a step: 4 mm over 0.5 s
            r = t.update(chair(cx, 0.180 + 0.0004 * i, 230.0))
        self.assertTrue(r.details["still"])
        self.assertEqual(r.score, 1.0)

    def test_jitter_counts_in_the_path(self):
        t = Tracker()
        for i in range(30):  # 0.6 mm back and forth every step: 6 mm of path over 0.5 s, though it goes nowhere
            s = placed(40.0, 0.010, cy=L.TARGET_CENTER[1] + (0.0006 if i % 2 else 0.0))
            r = t.update(s)
        self.assertFalse(r.details["still"])

    def test_a_turn_in_place_counts_as_moving(self):
        t = Tracker()
        off = L.corner_offsets(40.0)
        cx = L.BED_SIDE_X - 0.020 - max(dx for dx, _ in off.values())
        for i in range(30):  # 1 degree a step about its centre: 1 mm a step at its corners
            r = t.update(chair(cx, L.TARGET_CENTER[1], 230.0 + i * 1.0 * (1 if i < 15 else -1)))
        self.assertFalse(r.details["still"])


class Exploits(unittest.TestCase):
    def test_held_still_just_above_the_floor_counts_as_set_down(self):
        # The scorer sees only the chair (task.json fidelity): held still 2 mm up at the place it counts as set down.
        self.assertEqual(scored(placed(40.0, 0.010, lift=0.002)), 1.0)

    def test_tucked_past_the_head_end(self):
        # Close to the bed's side line and at a good angle, but past the head of the bed, far from her: 0.
        self.assertEqual(scored(placed(40.0, 0.005, cy=0.280)), 0.0)
        self.assertEqual(scored(placed(40.0, -0.020, cy=0.300)), 0.0)

    def test_parked_up_by_the_head_end_out_of_her_reach(self):
        # Close to the bed at 40 degrees, but up by the pillow (centre y 0.235): its nearest corner 48 mm past her
        # strong-side line, out of her reach for a pivot: 0
        s = placed(40.0, 0.010, cy=0.235)
        self.assertEqual(reason(s), "too_far_along_the_bed_from_her")
        self.assertEqual(scored(s), 0.0)
        self.assertEqual(scored(placed(40.0, 0.010, cy=0.215)), 1.0)  # 10 mm further along than the demo: fine

    def test_backwards_beside_the_bed(self):
        self.assertEqual(scored(placed(220.0, 0.010)), 0.0)  # 40 deg out but its front toward the head end

    def test_parked_out_in_the_room(self):
        self.assertEqual(scored(placed(40.0, 0.200)), 0.0)

    def test_set_down_across_her_feet_or_against_her_shins(self):
        self.assertEqual(scored(chair(0.340, 0.110, 180.0)), 0.0)
        self.assertEqual(scored(chair(0.340, 0.110, 270.0)), 0.0)

    def test_on_its_side_at_the_place(self):
        self.assertEqual(scored(placed(40.0, 0.010, roll=90.0)), 0.0)

    def test_beyond_the_bed_or_on_it(self):
        self.assertEqual(reason(chair(0.540, 0.200, 230.0)), "past_the_bed_line")  # behind the bed, on the floor
        self.assertEqual(scored(chair(0.440, 0.200, 230.0, lift=L.BED_H)), 0.0)  # standing on the mattress

    def test_parked_close_but_half_across_her_strong_side_line(self):
        # Close to the bed at 40 degrees but slid 25 mm toward her: its front corner reaches past her right arm.
        self.assertEqual(reason(placed(40.0, 0.010, cy=L.TARGET_CENTER[1] - 0.025)), "weak_side_or_across_her")


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
        inf.origin = np.array([np.inf, 0.2, TOP])
        bad.append(inf)
        for s in bad:
            self.assertFalse(valid(s))
            res = t.update(s)
            self.assertEqual(res.score, 0.0)
            self.assertEqual(res.details["zero_reason"], "invalid_state")
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

    def test_front_pointing_straight_up(self):
        s = placed(40.0, 0.010, pitch=-90.0)
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
        t.update(chair(L.CHAIR_START[0] + 0.03, L.CHAIR_START[1] - 0.03, 135.0))
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a, b = np.array(L.CHAIR_START), np.array([0.25, -0.15])
        for i in range(20):
            p = b + (a - b) * i / 19
            t.update(chair(p[0], p[1], 135.0))
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
        return {"wheelchair": self.mk.state_from_poses(self.task.WHEELCHAIR, (*s.origin, *s.quat_xyzw),
                                                       source="sim")}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], ["wheelchair"])
        self.assertLessEqual(t.WHEELCHAIR.mass, 0.30)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        bed = t.BED
        self.assertAlmostEqual(bed.pos[0] - bed.size[0] / 2, L.BED_SIDE_X)
        self.assertAlmostEqual(bed.pos[1] + bed.size[1] / 2, L.HEAD_END_Y)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            c, size, yaw = o.offline_box() if hasattr(o, "offline_box") else (o.pos, getattr(o, "size", None), 0.0)
            if size is None:
                size = (2 * o.radius,) * 3
            half = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            self.assertLess(abs(c[1]) + half, 0.45, o.name)
        # the strong-side line is just past her right arm, and her slippers are on her side of it
        arm = [b for b in L.RESIDENT_BOXES if b[0] == "right_arm"][0]
        self.assertLessEqual(arm[2][1], L.Y_STRONG)
        self.assertTrue(all(b[2][1] <= L.Y_STRONG for b in L.RESIDENT_BOXES))

    def test_chair_geometry_matches_the_scorer(self):
        spec = self.task.WHEELCHAIR
        lo, hi = spec.local_bounds()
        np.testing.assert_allclose(lo, [-L.FOOT_U, -L.FOOT_V, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.FOOT_U, L.FOOT_V, L.CHAIR_H], atol=1e-12)
        shapes = [(c, s) for _, c, s, _ in L.CHAIR_SHAPES]
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                self.assertLess(_boxes_overlap(shapes[i], shapes[j]), 1e-12, (L.CHAIR_SHAPES[i][0],
                                                                              L.CHAIR_SHAPES[j][0]))
        com = spec.center_of_mass()
        self.assertLess(abs(com[0] - L.GRIP_U), 0.008)  # the grip is within 8 mm of the centre of mass
        self.assertAlmostEqual(com[1], 0.0, places=12)
        rest = spec.rest_state()
        np.testing.assert_allclose(rest.origin, [L.CHAIR_START[0], L.CHAIR_START[1], TOP], atol=1e-12)
        self.assertTrue(Tracker.at_start(rest))
        sim = self.sim(TARGET)["wheelchair"]
        self.assertAlmostEqual(float(sim.low[2]), TOP, places=9)
        self.assertEqual(scored(sim), 1.0)
        # the grip spans the wheel plates: 63 mm outside, the pads (20 mm wide) wholly on their 60 mm length
        wheel = [s for s in L.CHAIR_SHAPES if s[0] == "wheel_left"][0]
        self.assertAlmostEqual(wheel[1][1] + wheel[2][1] / 2, L.FOOT_V)
        self.assertLessEqual(wheel[1][0] - wheel[2][0] / 2, L.GRIP_U - 0.010)
        self.assertGreaterEqual(wheel[1][0] + wheel[2][0] / 2, L.GRIP_U + 0.010)

    def test_offline_state_path(self):
        spec = self.task.WHEELCHAIR
        c, _, yaw = spec.offline_box()
        box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.CHAIR_START[0], L.CHAIR_START[1], TOP], atol=1e-12)
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
                for k in ("step", "score", "success", "zero_reason", "set_down", "upright", "on_floor", "still",
                          "closeness", "angle_credit", "alpha_deg", "gap_mm", "min_corner_y_mm", "centre_mm",
                          "tilt_deg", "lowest_mm", "path_mm", "chair_origin", "chair_quat", "source"):
                    self.assertIn(k, row)
                self.assertEqual(row["gap_mm"], 10.0)
                self.assertEqual(row["zero_reason"], "moving")  # one step seen: not yet still
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(TARGET)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    @staticmethod
    def _timeline(wps):
        t, out = 0.0, []
        for w in wps:
            out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes 1.5 s, standing still
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertEqual(tuple(prev.pos), tuple(w.pos), (arm, w.label))
                if w.pos[2] < prev.pos[2] and w.pos[2] <= t.PLACE_Z + 1e-9:  # slow last descents
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.03 / 1.5 + 1e-9, (arm, w.label))
        # the grips are where the chair is: across its wheels at its start, at the handover and at the place
        sg = (L.CHAIR_START[0] + L.GRIP_U * math.cos(L.CHAIR_START_HEADING),
              L.CHAIR_START[1] + L.GRIP_U * math.sin(L.CHAIR_START_HEADING))
        np.testing.assert_allclose(t.START_GRIP, sg, atol=1e-12)
        h = L.heading_of(L.TARGET_ALPHA_DEG)
        np.testing.assert_allclose(t.PLACE_GRIP, (L.TARGET_CENTER[0] + L.GRIP_U * math.cos(h),
                                                  L.TARGET_CENTER[1] + L.GRIP_U * math.sin(h)), atol=1e-12)
        self.assertAlmostEqual(math.degrees(t.YAW_START), -45.0)
        self.assertAlmostEqual(math.degrees(t.YAW_PLACE), 50.0)
        # the right arm is parked before the left arm leaves its park for the midline
        right_end = sum(w.duration for w in plan["right"])
        left = self._timeline(plan["left"])
        go = [s for s, e, w in left if w.label == "over the chair on the midline"][0]
        self.assertGreaterEqual(go + 1e-9, right_end)

    def test_carry_path_clears_her_and_the_bed(self):
        """Replays the left arm's carry (positions and yaw interpolated as ArmPlan does, the chair rigid in the
        grip) and checks the lifted chair's footprint against her slippers and shins and the bed's side."""
        t = self.task
        wps = t._left()
        grab = [i for i, w in enumerate(wps) if w.label == "lift 30 mm"][0]
        put = [i for i, w in enumerate(wps) if w.label == "lower it slowly"][0]
        feet = [b for b in L.RESIDENT_BOXES if b[0] in ("slippers", "lower_legs")]
        worst_feet, worst_bed = 1.0, 1.0
        for a, b in zip(wps[grab:put], wps[grab + 1:put + 1]):
            for s in np.linspace(0.0, 1.0, 41):
                k = 3 * s * s - 2 * s * s * s
                p = (1 - k) * np.array(a.pos) + k * np.array(b.pos)
                yaw = a.yaw + k * math.remainder(b.yaw - a.yaw, 2 * math.pi)
                heading = yaw + math.pi
                cx, cy = p[0] - L.GRIP_U * math.cos(heading), p[1] - L.GRIP_U * math.sin(heading)
                fp = footprint(chair(cx, cy, math.degrees(heading)))
                worst_bed = min(worst_bed, L.BED_SIDE_X - fp[:, 0].max())
                for _, xs, ys, _, _ in feet:
                    for x, y in fp:
                        dx = max(xs[0] - x, 0.0, x - xs[1])
                        dy = max(ys[0] - y, 0.0, y - ys[1])
                        worst_feet = min(worst_feet, math.hypot(dx, dy))
        self.assertGreater(worst_feet, 0.040)  # README: at least 45 mm from her slippers and shins
        self.assertGreater(worst_bed, 0.0099)  # ... and at least 10 mm from the bed's side (the place itself)

    def test_one_arm_switches(self):
        t = self.task
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule_and_matches_everywhere(self):
        ins = self.task.INSTRUCTION
        for s in ("left side is weak", "next to her on her strong side", "at most 40 mm along the bed past her",
                  "within 20 mm", "1:10", "toward the foot of the bed", "30 to 50 degrees", "stand-pivot transfer"):
            self.assertIn(s, ins)
        self.assertNotIn("right side", ins)  # the model works out which side that is
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
        self.assertIn("2,400 steps", spec["run_limit"])
        self.assertIn("demo", spec["scores"])


if __name__ == "__main__":
    unittest.main()
