"""Offline tests for scoring.py (and the hooks in task.py): synthetic slider traces and synthetic arm snapshots fed
through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import math
import os
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from offline_press import kinematic_joints  # noqa: E402
from scoring import (  # noqa: E402
    Breath, Compression, Tracker, analyse, bag_press_distance, cap_box, clearance, cycle_rate, group_cycles,
    hand_at_mark, is_good, plate_box, press_distance, pressing_bag, read_joints,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # the finger tips below the tool point
REST_D = L.CHEST_REST * 1000  # mm: where each slider rests under its own weight
REST_B = L.BAG_REST * 1000
AT_MARK = np.array([L.MARK_X, L.MARK_Y, L.TOP + 0.100])  # the right tool point waiting over the mark
OVER_BAG = np.array([L.BAG_X, L.BAG_Y, L.TOP + 0.140])
OFF_CHEST = np.array([L.MARK_X, L.MARK_Y, L.TOP + 0.150])
OFF_BAG = np.array([L.BAG_X, L.BAG_Y, L.TOP + 0.170])
READY_L, READY_R = np.array(L.READY_TCP["left"]), np.array(L.READY_TCP["right"])


def smooth(s):
    return s * s * (3 - 2 * s)


def arm(tcp, opening=0.0):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


def arms_at(left=OVER_BAG, right=AT_MARK, lo=0.0, ro=0.0):
    return {"left": arm(left, lo), "right": arm(right, ro)}


def joints(d_mm, b_mm):
    return {"chest": d_mm / 1000.0, "bag": b_mm / 1000.0}


class Film:
    """A trial, one control step per frame: chest depth d and cap depth b (mm) and both tool points."""

    def __init__(self, left=OVER_BAG, right=AT_MARK):
        self.frames = []
        self.d, self.b = REST_D, REST_B
        self.left, self.right = np.array(left, float), np.array(right, float)
        self.lo = self.ro = 0.0

    def frame(self, d=None, b=None, left=None, right=None):
        self.frames.append((self.d if d is None else d, self.b if b is None else b,
                            self.left if left is None else np.array(left, float),
                            self.right if right is None else np.array(right, float), self.lo, self.ro))
        return self

    def hold(self, n, **kw):
        for _ in range(n):
            self.frame(**kw)
        return self

    def compress(self, depth=55.0, recoil=None, down=5, up=6, offset=(0.0, 0.0), b=None, side="right", lift=0.0):
        """One compression by the ``side`` gripper (the right one, over the mark, by default; moved by ``offset``):
        down in ``down`` steps, back up to ``recoil`` in ``up`` steps. Its tool point rides 6 mm + 13 mm above the
        plate (21 mm above the mark at a 55 mm peak), ``lift`` higher."""
        recoil = REST_D if recoil is None else recoil
        start = self.d
        xy = (self.right if side == "right" else self.left)[:2] + np.array(offset)
        for i in range(1, down + 1):
            self.d = start + (depth - start) * smooth(i / down)
            self.frame(**{side: (*xy, AT_MARK[2] + lift - (self.d - REST_D) / 1000)}, b=b)
        for i in range(1, up + 1):
            self.d = depth + (recoil - depth) * smooth(i / up)
            self.frame(**{side: (*xy, AT_MARK[2] + lift - (self.d - REST_D) / 1000)}, b=b)
        return self

    def cycle(self, n=30, depths=None, **kw):
        for i in range(n):
            self.compress(depth=55.0 if depths is None else depths[i], **kw)
        return self

    def breath(self, depth=35.0, rise=6, hold=14, fall=8, d=None, right=None, by="left"):
        """The cap pressed to ``depth`` and let back up: about 1.1 s above 15 mm with the default steps. Pressed by
        the ``by`` gripper's closed finger tips (its tool point rides 13 mm above the cap's top, no higher than it
        waits over the bag), or with ``by=None`` by something else (no tool point moves with it)."""
        b0 = self.b

        def step():
            if by is None:
                return self.frame(d=d, right=right)
            rider = (L.BAG_X, L.BAG_Y, min(OVER_BAG[2], L.TOP + L.CAP_TOP - self.b / 1000 + TIPS))
            return self.frame(d=d, **({"right": rider} if by == "right" else {"left": rider, "right": right}))

        for i in range(1, rise + 1):
            self.b = b0 + (depth - b0) * smooth(i / rise)
            step()
        for _ in range(hold):
            step()
        for i in range(1, fall + 1):
            self.b = depth + (REST_B - depth) * smooth(i / fall)
            step()
        return self

    def breaths(self, n=2, gap=6, **kw):
        for k in range(n):
            self.breath(**kw)
            if k < n - 1:
                self.hold(gap)
        return self

    def hands_off(self, n=30):
        for i in range(1, 9):
            s = smooth(i / 8)
            self.frame(left=OVER_BAG + (OFF_BAG - OVER_BAG) * s, right=AT_MARK + (OFF_CHEST - AT_MARK) * s)
        self.left, self.right = OFF_BAG.copy(), OFF_CHEST.copy()
        return self.hold(n)


def procedure(counts=(30, 30), breaths=(2, 2), pause_wait=4, depths=None, final=True, **kw):
    """The intended trial: a short wait, then for each cycle its compressions and the breaths after it; then both
    grippers up off the manikin and a still hold."""
    f = Film().hold(10)
    for k, (n, nb) in enumerate(zip(counts, breaths)):
        f.cycle(n, depths=None if depths is None else depths[k], **kw)
        f.hold(pause_wait)
        if nb:
            f.breaths(nb)
        f.hold(pause_wait)
    if final:
        f.hands_off()
    return f


def play(film, tracker=None, arms=True):
    t = tracker or Tracker(HZ)
    r = None
    for d, b, left, right, lo, ro in film.frames:
        r = t.update(joints(d, b), arms_at(left, right, lo, ro) if arms else None)
    return r, t


class Geometry(unittest.TestCase):
    def test_read_joints(self):
        self.assertEqual(read_joints({"chest": 0.055, "bag": 0.0}), (55.0, 0.0))
        for bad in ({"chest": math.nan, "bag": 0.0}, {"chest": 0.0}, None, {"chest": "x", "bag": 0.0},
                    {"chest": 0.0, "bag": math.inf}):
            self.assertIsNone(read_joints(bad))

    def test_rest_sag(self):
        self.assertAlmostEqual(REST_D, 1.96, places=2)  # 15 g on 75 N/m
        self.assertAlmostEqual(REST_B, 3.68, places=2)  # 15 g on 40 N/m
        self.assertLess(REST_D, L.RECOIL_MAX_MM)  # a free chest is fully recoiled
        self.assertLess(REST_B, L.BREATH_ON_MM)

    def test_stand_in_at_rest_and_pressed(self):
        rest = kinematic_joints(arms_at(READY_L, READY_R))
        self.assertAlmostEqual(rest["chest"], L.CHEST_REST)
        self.assertAlmostEqual(rest["bag"], L.BAG_REST)
        bottom = kinematic_joints(arms_at(right=(L.MARK_X, L.MARK_Y, L.TOP + 0.031)))
        self.assertTrue(53.0 < bottom["chest"] * 1000 < 58.0)  # the demo's stroke: about 55 mm
        breath = kinematic_joints(arms_at(left=(L.BAG_X, L.BAG_Y, L.TOP + 0.085)))
        self.assertTrue(33.0 < breath["bag"] * 1000 < 38.0)
        # the give: the slider goes less far than the tips (two springs in series)
        tips_below = L.MARK_TOP - (0.031 - TIPS)
        self.assertLess(bottom["chest"], tips_below)
        deep = kinematic_joints(arms_at(right=(L.MARK_X, L.MARK_Y, L.TOP + 0.0)))
        self.assertAlmostEqual(deep["chest"], L.CHEST_TRAVEL)  # the stop
        beside = kinematic_joints(arms_at(right=(L.MARK_X, L.MARK_Y - 0.070, L.TOP + 0.031)))
        self.assertAlmostEqual(beside["chest"], L.CHEST_REST)  # off the plate: nothing pressed
        self.assertEqual(kinematic_joints(None), {"chest": L.CHEST_REST, "bag": L.BAG_REST})

    def test_hand_at_mark(self):
        self.assertTrue(hand_at_mark(arms_at()))
        self.assertTrue(hand_at_mark(arms_at(right=AT_MARK + [0.024, 0.0, 0.0])))
        self.assertFalse(hand_at_mark(arms_at(right=AT_MARK + [0.026, 0.0, 0.0])))
        self.assertTrue(hand_at_mark(arms_at(right=(L.MARK_X, L.MARK_Y, L.TOP + L.MARK_TOP + 0.059))))
        self.assertFalse(hand_at_mark(arms_at(right=(L.MARK_X, L.MARK_Y, L.TOP + L.MARK_TOP + 0.061))))
        self.assertTrue(hand_at_mark(arms_at(left=AT_MARK, right=READY_R)))  # either gripper will do

    def test_pressing_bag(self):
        cap_top = L.TOP + L.CAP_TOP - 0.035  # the cap's top with the cap 35 mm down
        on = (L.BAG_X, L.BAG_Y, cap_top + TIPS)  # closed finger tips on it
        self.assertTrue(pressing_bag(arms_at(left=on), 35.0))
        self.assertAlmostEqual(bag_press_distance(arms_at(left=on), 35.0), 0.0)
        self.assertTrue(pressing_bag(arms_at(left=(L.BAG_X + 0.059, L.BAG_Y, cap_top + 0.039)), 35.0))
        self.assertFalse(pressing_bag(arms_at(left=(L.BAG_X + 0.061, L.BAG_Y, cap_top + 0.039)), 35.0))
        self.assertFalse(pressing_bag(arms_at(left=(L.BAG_X, L.BAG_Y, cap_top + 0.041)), 35.0))
        self.assertTrue(pressing_bag(arms_at(left=READY_L, right=on), 35.0))  # either gripper will do
        self.assertFalse(pressing_bag(arms_at(), 35.0))  # waiting over the bag (140 mm): too high
        self.assertFalse(pressing_bag(arms_at(left=READY_L), 35.0))
        # one gripper cannot press the cap and wait over the mark (39 cm apart)
        self.assertGreater(math.hypot(L.BAG_X - L.MARK_X, L.BAG_Y - L.MARK_Y), L.BAG_TOL + L.MARK_TOL + 0.25)

    def test_press_distance(self):
        mark_top = L.TOP + L.MARK_TOP - 0.055  # the mark with the chest 55 mm down
        low = (L.MARK_X + 0.010, L.MARK_Y, mark_top + 0.039)
        self.assertAlmostEqual(press_distance(arms_at(READY_L, low), 55.0), 0.010)
        self.assertAlmostEqual(press_distance(arms_at(low, READY_R), 55.0), 0.010)  # either gripper will do
        self.assertEqual(press_distance(arms_at(READY_L, (L.MARK_X, L.MARK_Y, mark_top + 0.041)), 55.0), math.inf)
        self.assertEqual(press_distance(arms_at(OVER_BAG, AT_MARK), 55.0), math.inf)  # waiting over it: too high
        self.assertLess(press_distance(arms_at(OVER_BAG, AT_MARK), REST_D), 1e-9)  # ... fine with the chest up

    def test_boxes_and_clearance(self):
        lo, hi = plate_box(0.0)
        self.assertAlmostEqual(hi[2], L.TOP + L.MARK_TOP)
        self.assertAlmostEqual(hi[0] - lo[0], L.PLATE_SIZE[0])
        lo2, hi2 = plate_box(55.0)
        self.assertAlmostEqual(hi[2] - hi2[2], 0.055)
        clo, chi = cap_box(0.0)
        self.assertAlmostEqual(chi[2], L.TOP + L.CAP_TOP)
        self.assertAlmostEqual(chi[2] - clo[2], L.CAP_SIZE[2])
        over_cap = 0.170 - TIPS - (L.CAP_TOP - REST_B / 1000)  # the left finger tips over the cap's top
        over_plate = 0.150 - TIPS - (L.MARK_TOP - REST_D / 1000)  # the right finger tips over the mark
        self.assertAlmostEqual(clearance(arms_at(OFF_BAG, OFF_CHEST), REST_D, REST_B), min(over_cap, over_plate))
        self.assertLess(clearance(arms_at(), REST_D, REST_B), L.CLEAR)  # waiting just over the mark: not clear

    def test_cycles_and_rate(self):
        cs = [Compression(0, 0, 55, t, 2, 0.0) for t in (1.0, 1.55, 2.10, 5.0, 5.6)]
        cyc = group_cycles(cs)
        self.assertEqual([len(c) for c in cyc], [3, 2])
        self.assertAlmostEqual(cycle_rate(cyc[0]), 60 * 2 / 1.10)
        self.assertIsNone(cycle_rate([cs[0]]))
        self.assertTrue(is_good(cs[0], None))
        self.assertTrue(is_good(cs[0], 0.43) and is_good(cs[0], 0.70))
        self.assertFalse(is_good(cs[0], 0.40) or is_good(cs[0], 0.75))
        self.assertFalse(is_good(Compression(0, 0, 49.9, 1, 2, 0.0), None))
        self.assertFalse(is_good(Compression(0, 0, 60.1, 1, 2, 0.0), None))
        self.assertFalse(is_good(Compression(0, 0, 55, 1, 5.1, 0.0), None))
        self.assertFalse(is_good(Compression(0, 0, 55, 1, 2, 0.026), None))
        self.assertTrue(is_good(Compression(0, 0, 55, 1, 2, None), None))  # arms unreadable at the peak

    def test_breath_placement(self):
        cs = [Compression(0, 0, 55, t, 2, 0.0) for t in (1.0, 1.55, 2.1, 6.0, 6.55)]
        ok = dict(depth=35, chest_max=2)
        brs = [Breath(0.2, 1.0, **ok), Breath(1.6, 2.05, **ok), Breath(2.5, 3.5, **ok), Breath(4.0, 5.0, **ok),
               Breath(7.0, 8.0, **ok), Breath(8.2, 9.0, chest_max=12, depth=35)]
        a = analyse(cs, brs)
        self.assertEqual(a.breaths_before, 1)
        self.assertEqual(a.breaths_during, 1)
        self.assertEqual([len(b) for b in a.pause_breaths], [2, 1])  # the last one: chest at 12 mm, not counted
        self.assertAlmostEqual(a.pauses[0], 6.0 - 2.1)
        self.assertAlmostEqual(a.pauses[1], 8.0 - 6.55)  # to the end of its only counted breath


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(len(t.compressions), 60)
        self.assertEqual(r.details["n_good"], 60)
        self.assertEqual([c["n"] for c in r.details["cycles"]], [30, 30])
        self.assertEqual([c["breaths_after"] for c in r.details["cycles"]], [2, 2])
        self.assertTrue(r.details["last_breath"]["by_gripper"])
        self.assertAlmostEqual(r.details["cycles"][0]["rate_per_min"], 109.1, places=1)
        self.assertAlmostEqual(r.details["cycles"][0]["pause_after_s"], 4.05, places=2)
        self.assertLess(r.details["cycles"][1]["pause_after_s"], L.FULL_PAUSE_S)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f)
        d, b, left, right, lo, ro = f.frames[-1]
        for _ in range(3):
            self.assertEqual(t.result(joints(d, b), arms_at(left, right)).score, 1.0)
        self.assertEqual(len(t.compressions), 60)
        self.assertEqual(t.steps, len(f.frames))

    def test_28_and_32_compressions_per_cycle(self):
        for n in (28, 32):
            r, _ = play(procedure(counts=(n, n)))
            self.assertEqual(r.score, 1.0, n)
        for n in (27, 33):
            r, _ = play(procedure(counts=(n, 30)))
            self.assertEqual(r.score, 0.75, n)

    def test_rate_outside_100_to_120_scores_075(self):
        # 0.65 s a compression (92 a minute): every interval within 0.43-0.70 s, so all good, but the rate is low
        r, _ = play(procedure(down=6, up=7))
        self.assertAlmostEqual(r.details["cycles"][0]["rate_per_min"], 92.3, places=1)
        self.assertEqual(r.details["n_good"], 60)
        self.assertEqual(r.score, 0.75)

    def test_too_fast_compressions_are_not_good(self):
        r, _ = play(procedure(down=4, up=4))  # 0.40 s apart: 150 a minute
        self.assertEqual(r.details["n_good"], 2)  # only the first of each cycle
        self.assertEqual(r.score, 0.0)

    def test_breath_counts(self):
        cases = {(2, 1): 0.75, (1, 2): 0.75, (3, 2): 0.75, (2, 3): 0.75, (1, 1): 0.5, (0, 0): 0.25}
        for breaths, want in cases.items():
            r, _ = play(procedure(breaths=breaths))
            self.assertEqual(r.score, want, breaths)
        # no breaths after the first cycle and a short wait: the two cycles run together into one of 60
        r, _ = play(procedure(breaths=(0, 2)))
        self.assertEqual([c["n"] for c in r.details["cycles"]], [60])
        self.assertEqual(r.score, 0.25)
        r, _ = play(procedure(breaths=(0, 2), pause_wait=20))  # 2.55 s between them: two cycles
        self.assertEqual([c["n"] for c in r.details["cycles"]], [30, 30])
        self.assertEqual(r.score, 0.5)

    def test_pause_over_10_s(self):
        f = Film().hold(10).cycle(30).hold(4).breaths(2).hold(160).cycle(30).hold(4).breaths(2).hands_off()
        r, _ = play(f)
        self.assertGreater(r.details["cycles"][0]["pause_after_s"], 10.0)
        self.assertEqual(r.score, 0.5)
        f = Film().hold(10).cycle(30).hold(4).breaths(2).hold(260).cycle(30).hold(4).breaths(2).hands_off()
        r, _ = play(f)
        self.assertGreater(r.details["cycles"][0]["pause_after_s"], 15.0)
        self.assertEqual(r.score, 0.25)

    def test_slow_final_breaths_count_against_the_last_pause(self):
        f = Film().hold(10).cycle(30).hold(4).breaths(2).hold(4).cycle(30).hold(4).breaths(2, gap=150).hands_off()
        r, _ = play(f)
        self.assertGreater(r.details["cycles"][1]["pause_after_s"], 10.0)
        self.assertEqual(r.score, 0.5)

    def test_one_cycle_with_breaths_scores_half(self):
        r, _ = play(procedure(counts=(30,), breaths=(2,)))
        self.assertEqual(r.score, 0.5)

    def test_hands_only_cpr_scores_quarter(self):
        r, _ = play(procedure(breaths=(0, 0)))
        self.assertEqual(r.score, 0.25)
        r, _ = play(procedure(counts=(20,), breaths=(0,)))
        self.assertEqual(r.score, 0.25)

    def test_too_few_good_compressions_scores_zero(self):
        r, _ = play(procedure(counts=(19,), breaths=(2,)))
        self.assertEqual(r.score, 0.0)

    def test_depth_band(self):
        for depth, want in ((50.5, 1.0), (59.5, 1.0), (48.0, 0.0), (62.0, 0.0), (L.CHEST_TRAVEL * 1000, 0.0)):
            r, _ = play(procedure(depths=([depth] * 30, [depth] * 30)))
            self.assertEqual(r.score, want, depth)

    def test_ninety_percent_good(self):
        bad6 = [55.0] * 27 + [45.0] * 3  # 6 of 60 shallow: 90 % good
        r, _ = play(procedure(depths=(bad6, bad6)))
        self.assertEqual(r.details["n_good"], 54)
        self.assertEqual(r.score, 1.0)
        bad7 = [55.0] * 26 + [45.0] * 4  # 7 of 60 shallow (one compression after a shallow one is still good)
        r, _ = play(procedure(depths=(bad6, bad7)))
        self.assertEqual(r.details["n_good"], 53)
        self.assertEqual(r.score, 0.75)
        bad = [55.0] * 20 + [45.0] * 10
        r, _ = play(procedure(depths=(bad, bad)))  # 66 % good
        self.assertEqual(r.score, 0.5)

    def test_incomplete_recoil_is_not_good(self):
        r, _ = play(procedure(recoil=7.0))
        self.assertEqual(r.details["n_good"], 1)  # only the very first: the chest never came back past 7 mm after
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(recoil=4.5))
        self.assertEqual(r.score, 1.0)

    def test_pressing_gripper_height_at_the_peak(self):
        r, _ = play(procedure(lift=0.018))  # tool point 39 mm above the mark at each peak
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(lift=0.020))  # 41 mm: not the gripper pressing it
        self.assertEqual(r.details["n_good"], 0)
        self.assertFalse(r.details["last_compression"]["pressed_at_mark"])
        self.assertIsNone(r.details["last_compression"]["mark_mm"])
        self.assertEqual(r.score, 0.0)

    def test_compressions_off_the_mark_are_not_good(self):
        r, _ = play(procedure(offset=(0.0, 0.030)))
        self.assertEqual(r.details["n_good"], 0)
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(offset=(0.015, 0.015)))  # 21 mm off: within 25 mm
        self.assertEqual(r.score, 1.0)

    def test_three_cycles_is_not_one(self):
        f = procedure(final=False)
        f.cycle(30).hold(4).breaths(2).hands_off()
        r, _ = play(f)
        self.assertEqual(len(r.details["cycles"]), 3)
        self.assertEqual(r.score, 0.75)

    def test_breaths_before_the_first_cycle_are_not_one(self):
        f = Film().hold(10).breaths(2).hold(4).cycle(30).hold(4).breaths(2).hold(4).cycle(30).hold(4).breaths(2)
        r, _ = play(f.hands_off())
        self.assertEqual(r.details["breaths_before_first_cycle"], 2)
        self.assertEqual(r.score, 0.75)

    def test_a_breath_during_a_cycle_is_not_one(self):
        f = Film().hold(10).cycle(15).hold(2).breath(hold=10).cycle(15).hold(4).breaths(2).hold(4)
        f.cycle(30).hold(4).breaths(2).hands_off()
        r, _ = play(f)
        self.assertEqual(r.details["breaths_during_cycles"], 1)
        self.assertEqual([c["n"] for c in r.details["cycles"]], [30, 30])
        self.assertEqual(r.score, 0.75)

    def test_breath_length(self):
        for hold, counted in ((14, True), (2, False), (4, True), (40, False), (8, True)):  # over 15 mm: hold + 8
            r, _ = play(procedure(final=False).breath(hold=hold))
            self.assertEqual(r.details["last_breath"]["counted"], counted, hold)
        r, _ = play(Film().hold(5).breath(depth=22.0))  # never past 25 mm: no breath at all
        self.assertEqual(r.details["breaths"], 0)

    def test_compression_thresholds(self):
        f = Film().hold(10)
        for depth in (22.0, 19.0, 22.0):  # a press past 20 mm is a compression (not a good one); 19 mm is none
            f.compress(depth=depth, down=6, up=6).hold(4)
        r, t = play(f)
        self.assertEqual(r.details["n_compressions"], 2)
        self.assertEqual([round(c.depth) for c in t.compressions], [22, 22])
        f = Film().hold(10).compress(depth=55.0, recoil=12.0).compress(depth=55.0)  # never back below 10 mm
        r, t = play(f.hold(10))
        self.assertEqual(r.details["n_compressions"], 1)

    def test_hands_off_threshold(self):
        f = procedure(final=False)
        f.hands_off(n=5)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        cap_top = L.TOP + L.CAP_TOP - REST_B / 1000
        for gap, want in ((0.025, 0.75), (0.032, 1.0)):
            near = (L.BAG_X, L.BAG_Y, cap_top + gap + TIPS)  # the left finger tips this far over the cap
            r = t.update(joints(REST_D, REST_B), arms_at(near, OFF_CHEST))
            self.assertEqual(r.score, want, gap)
            self.assertAlmostEqual(r.details["clear_mm"], gap * 1000, places=1)

    def test_hands_still_on_at_the_end(self):
        r, _ = play(procedure(final=False).hold(30))
        self.assertFalse(r.details["hands_off"])
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_success_needs_stillness_and_nothing_in_progress(self):
        f = procedure()
        r, t = play(f)
        self.assertTrue(r.success)
        r = t.update(joints(REST_D + 0.5, REST_B), arms_at(OFF_BAG, OFF_CHEST))  # the plate twitches 10 mm/s
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r = t.update(joints(REST_D, REST_B + 16.0), arms_at(OFF_BAG, OFF_CHEST))  # a breath begins
        self.assertFalse(r.success)


class Breaches(unittest.TestCase):
    def test_chest_pressed_during_a_breath_scores_zero(self):
        f = procedure(final=False)
        f.breath(d=25.0).hold(4).hands_off()
        r, t = play(f)
        self.assertIn("chest_pressed_during_breath", t.zero)
        self.assertEqual(r.score, 0.0)

    def test_chest_slightly_pressed_during_a_breath_does_not_count_it(self):
        f = Film().hold(10).cycle(30).hold(4).breath().hold(6).breath(d=12.0).hold(4).cycle(30).hold(4).breaths(2)
        r, t = play(f.hands_off())
        self.assertFalse(t.zero)
        self.assertEqual([c["breaths_after"] for c in r.details["cycles"]], [1, 2])
        self.assertEqual(r.score, 0.75)

    def test_breath_without_a_gripper_over_the_mark_does_not_count(self):
        away = OFF_CHEST + [0.0, 0.0, 0.05]
        f = Film().hold(10).cycle(30).hold(4).breaths(2, right=away).hold(4).cycle(30).hold(4)
        f.breaths(2, right=away).hands_off()
        r, _ = play(f)
        self.assertEqual(r.details["breaths_counted"], 0)
        self.assertEqual(r.score, 0.25)

    def test_zero_is_never_undone(self):
        f = Film().hold(5).breath(d=30.0)
        _, t = play(f)
        r, _ = play(procedure(), tracker=t)
        self.assertEqual(r.score, 0.0)


class Exploits(unittest.TestCase):
    def test_one_arm_does_both_jobs(self):
        # the right arm compresses, then stretches over to the bag itself: no gripper waits over the mark
        f = Film(left=READY_L).hold(10)
        for _ in range(2):
            f.cycle(30).hold(4)
            f.breaths(2, by="right").hold(4)
        r, _ = play(f.hands_off())
        self.assertEqual(r.details["breaths_counted"], 0)
        self.assertFalse(r.details["last_breath"]["hand_at_mark"])
        self.assertTrue(r.details["last_breath"]["by_gripper"])
        self.assertEqual(r.score, 0.25)

    def test_one_arm_leans_a_link_on_the_bag_while_its_gripper_waits_over_the_mark(self):
        # the right arm compresses, then (were it able to) pushes the cap down with its forearm while its gripper
        # stays over the mark; the left arm idles at ready: no gripper presses the cap, so no breath counts
        f = Film(left=READY_L).hold(10)
        for _ in range(2):
            f.cycle(30).hold(4).breaths(2, by=None).hold(4)
        r, _ = play(f.hands_off())
        self.assertEqual(r.details["breaths"], 4)
        self.assertTrue(r.details["last_breath"]["hand_at_mark"])
        self.assertFalse(r.details["last_breath"]["by_gripper"])
        self.assertEqual(r.details["breaths_counted"], 0)
        self.assertEqual(r.score, 0.25)
        # the same with the cap pressed by the gripper (two arms): the intended trial
        r, _ = play(procedure())
        self.assertEqual(r.score, 1.0)

    def test_leaning_on_the_chest_is_one_long_compression(self):
        f = Film().hold(10)
        for _ in range(30):
            f.compress(depth=55.0, recoil=25.0)
        f.compress(depth=55.0)
        r, _ = play(f.hold(4).breaths(2).hold(20))
        self.assertEqual(r.details["n_compressions"], 1)
        self.assertEqual(r.score, 0.0)

    def test_holding_the_bag_down_while_compressing(self):
        f = Film().hold(10).breath(hold=0, fall=0, rise=6)  # the cap pressed and held
        f.cycle(30, b=35.0)
        r, t = play(f)
        self.assertIn("chest_pressed_during_breath", t.zero)
        self.assertEqual(r.score, 0.0)

    def test_pressing_the_chest_edge_from_the_left(self):
        # the left arm's open fingers can reach the plate's edge, but its tool point stays 70 mm from the mark
        f = Film(left=(L.MARK_X, L.MARK_Y + 0.070, L.TOP + 0.10), right=READY_R).hold(10)
        f.right = np.array([L.MARK_X, L.MARK_Y + 0.070, L.TOP + 0.10])  # compress() moves 'right'; same point
        f.cycle(30).hold(4).cycle(30)
        r, _ = play(f.hold(10))
        self.assertEqual(r.details["n_good"], 0)
        self.assertEqual(r.score, 0.0)

    def test_hovering_over_the_mark_while_the_other_presses_elsewhere(self):
        # the right gripper only waits over the mark, the left presses the plate 50 mm from it (were it able to):
        # the gripper over the mark is too high to be the one pressing, so no compression is good
        edge = (L.MARK_X, L.MARK_Y + 0.050, L.TOP + 0.100)
        f = Film(left=edge, right=AT_MARK).hold(10)
        for _ in range(2):
            f.cycle(30, side="left").hold(4).breaths(2).hold(4)
        r, _ = play(f.hands_off())
        self.assertEqual(r.details["breaths_counted"], 4)
        self.assertEqual(r.details["n_good"], 0)
        self.assertEqual(r.score, 0.0)

    def test_fast_shallow_taps(self):
        f = Film().hold(10)
        for _ in range(80):
            f.compress(depth=25.0, down=2, up=2)
        r, _ = play(f.hold(10))
        self.assertEqual(r.details["n_good"], 0)
        self.assertEqual(r.score, 0.0)


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_taken_as_met_and_counted(self):
        f = procedure()
        r, t = play(f, arms=False)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(f.frames))
        self.assertIsNone(r.details["opening"])
        self.assertIsNone(r.details["clear_mm"])

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 0.0), "right": arm(AT_MARK)}
        t.update(joints(REST_D, REST_B), bad)
        t.update(joints(REST_D, REST_B), {"left": arm(OVER_BAG)})  # one arm missing
        bad_open = {"left": arm(OVER_BAG), "right": rigstate.Arm(AT_MARK, AT_MARK, float("nan"))}
        t.update(joints(REST_D, REST_B), bad_open)
        self.assertEqual(t.arms_unavailable, 3)

    def test_offline_feed_reaches_the_scorer(self):
        rigstate.offline_feed({"left": {"tcp": OVER_BAG, "tips": OVER_BAG - [0, 0, TIPS], "opening": 0.25},
                               "right": {"tcp": AT_MARK, "tips": AT_MARK - [0, 0, TIPS], "opening": 0.75}})
        a = rigstate.arms()
        self.assertAlmostEqual(a["left"].opening, 0.25)
        t = Tracker(HZ)
        r = t.update(kinematic_joints(a), a)
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["hand_at_mark"])
        self.assertAlmostEqual(r.details["chest_mm"], REST_D, places=1)


class Robustness(unittest.TestCase):
    def test_blow_up_readings_score_zero_and_leave_the_record_alone(self):
        f = procedure()
        _, t = play(f)
        n, steps = len(t.compressions), t.steps
        for bad in ({"chest": math.nan, "bag": 0.0}, {"chest": 0.0, "bag": math.inf}, {"chest": 0.0}, None,
                    {"chest": None, "bag": None}):
            r = t.update(bad, arms_at(OFF_BAG, OFF_CHEST))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(len(t.compressions), n)
        self.assertEqual(t.bad_steps, 5)
        self.assertEqual(t.steps, steps + 5)
        r = t.update(joints(REST_D, REST_B), arms_at(OFF_BAG, OFF_CHEST))
        self.assertEqual(r.score, 1.0)  # the record survived

    def test_readings_anywhere_do_not_crash(self):
        rng = np.random.default_rng(0)
        t = Tracker(HZ)
        for _ in range(500):
            d, b = rng.uniform(-20, 120), rng.uniform(-20, 120)
            r = t.update(joints(d, b), arms_at(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def finished(self):
        _, t = play(procedure())
        return t

    def test_jump_to_ready_resets(self):
        t = self.finished()
        r = t.update(joints(0.0, 0.0), arms_at(READY_L, READY_R, 1.0, 1.0))
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.compressions, [])
        self.assertEqual(r.score, 0.0)

    def test_slow_return_to_ready_does_not_reset(self):
        t = self.finished()
        for i in range(1, 41):
            s = smooth(i / 40)
            t.update(joints(REST_D, REST_B), arms_at(OFF_BAG + (READY_L - OFF_BAG) * s,
                                                     OFF_CHEST + (READY_R - OFF_CHEST) * s, 1.0, 1.0))
        self.assertEqual(len(t.compressions), 60)

    def test_a_jump_elsewhere_or_closed_is_not_a_reset(self):
        t = self.finished()
        t.update(joints(REST_D, REST_B), arms_at(READY_L + [0, 0, -0.2], READY_R, 1.0, 1.0))
        self.assertEqual(len(t.compressions), 60)
        t = self.finished()
        t.update(joints(REST_D, REST_B), arms_at(READY_L, READY_R, 0.0, 0.0))  # at ready but closed
        self.assertEqual(len(t.compressions), 60)
        t = self.finished()
        t.update(joints(30.0, REST_B), arms_at(READY_L, READY_R, 1.0, 1.0))  # chest still pressed
        self.assertEqual(len(t.compressions), 60)

    def test_rigstate_reset_hook_resets_a_tracker(self):
        t = self.finished()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual(t.steps, 0)
            self.assertEqual(t.compressions, [])
            self.assertFalse(t.zero)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: the manikin, its hooks, its reset hooks, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def manikin(self, d_mm=REST_D, b_mm=REST_B, source="sim", origin=None):
        spec = self.task.MANIKIN
        pose = (*(spec.pos if origin is None else origin), 0.0, 0.0, 0.0, 1.0)
        return self.mk.state_from_poses(spec, pose, joints={"chest": d_mm / 1000, "bag": b_mm / 1000},
                                        source=source)

    def feed(self, left=OVER_BAG, right=AT_MARK, lo=0.0, ro=0.0):
        rigstate.offline_feed({"left": {"tcp": left, "tips": np.asarray(left) - [0, 0, TIPS], "opening": lo},
                               "right": {"tcp": right, "tips": np.asarray(right) - [0, 0, TIPS], "opening": ro}})

    def test_objects(self):
        objs = self.task.OBJECTS
        self.assertEqual([o.name for o in objs], ["manikin"])
        m = objs[0]
        self.assertTrue(m.fixed_base)
        self.assertEqual([j.name for j in m.joints], ["chest", "bag"])
        self.assertEqual([j.kind for j in m.joints], ["prismatic", "prismatic"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        lo, hi = m.world_bounds()
        self.assertAlmostEqual(lo[2], L.TOP)
        self.assertAlmostEqual(hi[2], L.TOP + L.CAP_TOP)
        # the chest plate's top is level with the deck, the mark 1 mm above; the plate sits in the opening
        plo, phi = self.manikin(d_mm=0.0).link_bounds("chest")
        self.assertAlmostEqual(phi[2], L.TOP + L.MARK_TOP)
        self.assertAlmostEqual(phi[0] - plo[0], L.PLATE_SIZE[0])
        self.assertLess(L.PLATE_SIZE[0], L.OPENING)
        # at its stops neither slider reaches the table
        self.assertGreater(L.PLATE_TOP - L.PLATE_SIZE[2] - L.CHEST_TRAVEL, 0.0)
        self.assertGreater(L.CAP_TOP - L.CAP_SIZE[2] - L.BAG_TRAVEL, 0.0)
        self.assertGreater(L.CAP_TOP - L.CAP_PLATE - L.BAG_TRAVEL, L.POST_SIZE[2])  # nor the cap's plate the post

    def test_hooks_read_the_joints_in_the_simulator(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed()
        self.assertFalse(self.task.success({"manikin": self.manikin(d_mm=30.0)}))
        self.assertAlmostEqual(tr.last.details["chest_mm"], 30.0)
        self.assertTrue(tr.last.details["in_compression"])
        self.assertEqual(self.task.score({"manikin": self.manikin(d_mm=30.0)}), 0.0)

    def test_offline_states_use_the_stand_in(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(right=(L.MARK_X, L.MARK_Y, L.TOP + 0.031))
        self.task.success({"manikin": self.manikin(d_mm=0.0, b_mm=0.0, source="offline")})
        self.assertTrue(53.0 < tr.last.details["chest_mm"] < 58.0)

    def test_broken_manikin_state_scores_zero(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed()
        bad = self.manikin(origin=(math.nan, 0.0, 0.0))
        self.assertFalse(self.task.success({"manikin": bad}))
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(self.task.score({"manikin": bad}), 0.0)

    def test_reset_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed()
        self.task.success({"manikin": self.manikin(d_mm=30.0)})
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success({"manikin": self.manikin()}))  # arms None right after a reset
        self.assertEqual(tr.arms_unavailable, 1)
        self.feed()
        self.task.success({"manikin": self.manikin()})
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's reset counter: the second, independent signal
        self.task.success({"manikin": self.manikin()})
        self.assertEqual(tr.steps, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed()
            self.assertFalse(self.task.success({"manikin": self.manikin()}))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_timing(self):
        task = self.task
        plan = task.demo()
        self.assertEqual(set(plan), {"left", "right"})

        def times(wps):
            out, t = [], 0.0
            for w in wps:
                t += w.duration
                out.append((t, w))
            return out

        right, left = times(plan["right"]), times(plan["left"])
        downs = [t for t, w in right if w.label.endswith(": down")]
        self.assertEqual(len(downs), 60)
        gaps = np.diff(downs)
        self.assertTrue(np.allclose(gaps[gaps < 1.0], 0.55))
        self.assertEqual(int(np.sum(gaps > 2.0)), 1)
        # breaths start only 0.3 s or more after a cycle's last up-stroke and end 0.3 s or more before the next
        # cycle's first down-stroke (room for the arms' lag: the 0 rule needs the chest and the cap pressed at once)
        presses = [(t - w.duration, t) for t, w in left if "press the bag" in w.label]
        ups = [t for t, w in left if "let it come back up" in w.label]
        self.assertEqual(len(presses), 4)
        self.assertGreaterEqual(presses[0][0], task.CYCLE_ENDS[0] + 0.3)
        self.assertLessEqual(ups[1], task.CYCLE_STARTS[1] - 0.3)
        self.assertGreaterEqual(presses[2][0], task.CYCLE_ENDS[1] + 0.3)
        # pause 1 (cycle 1's last peak to cycle 2's first) well under 10 s
        self.assertLess(task.CYCLE_STARTS[1] + task.DOWN_S - (task.CYCLE_ENDS[0] - task.UP_S), 5.0)
        # the right arm waits over the mark, not pressing, until the last breath is done
        waits = [(t - w.duration, t, w) for t, w in right if w.label.startswith("wait over the mark")]
        self.assertEqual(len(waits), 2)
        self.assertGreaterEqual(waits[1][1], ups[-1] - 1e-9)
        for _, _, w in waits:
            self.assertAlmostEqual(w.pos[2], task.STROKE_TOP)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(task.demo()), want)
            finally:
                os.environ.pop(key, None)


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo played through our offline checker's kinematics (the presses from offline_press.py)."""

    def test_demo_predicts_one_and_each_arm_alone_less(self):
        import json
        import subprocess
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("TASK_TRACE", None)
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--one-arm", "--json", "-"],
                              capture_output=True, text=True, env=env, timeout=1200)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        rep = json.loads(proc.stdout)
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 1400)
        self.assertEqual(rep["one_arm"], {"left_only": 0.0, "right_only": 0.25})
        for arm in ("left", "right"):
            self.assertLess(rep["playback"]["arms"][arm]["worst_ik_mm"], 0.5)


if __name__ == "__main__":
    unittest.main()
