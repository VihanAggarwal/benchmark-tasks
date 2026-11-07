"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import itertools
import json
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
    Tracker, evaluate, far_row, fill, lowest, off_table, planted, rotation, tilt_deg, valid,
)

MM = 0.001
NAME_INDEX = {n: i for i, n in enumerate(L.NAMES)}
TARGET_YS = [y for _, y in L.TARGETS]  # +0.20, +0.10, 0.00, -0.10, -0.20


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A plug's state: its world bounding box worked out from the turned cylinder."""
    c = np.array(center, float)
    u = rotation(quat)[:, 2]
    ext = np.array([L.PLUG_H / 2 * abs(u[i]) + L.PLUG_R * np.sqrt(max(0.0, 1 - u[i] ** 2)) for i in range(3)])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def plug(x, y, bottom=L.BED_TOP, tilt=0.0, axis="x", **kw):
    """A plug at (x, y) with its lowest point at ``bottom``, tilted ``tilt`` degrees about the world ``axis``."""
    q = about(axis, tilt)
    zz = abs(rotation(q)[2, 2])
    ext_z = L.PLUG_H / 2 * zz + L.PLUG_R * np.sqrt(max(0.0, 1 - zz * zz))
    return state((x, y, bottom + ext_z), q, **kw)


def at_place(k, dx=0.0, dy=0.0, **kw):
    """A plug planted on target ``k`` (0..4: y +0.20 .. -0.20), off it by (dx, dy)."""
    return plug(L.NEAR_X + dx, TARGET_YS[k] + dy, **kw)


def start():
    """The starting layout (order of L.NAMES): the far row on the bed, three plugs on each tray."""
    return [plug(L.START[n][0], L.START[n][1], bottom=L.START[n][2] - L.PLUG_H / 2) for n in L.NAMES]


S0 = start()


def bed_with(places=None, spare=None, **over):
    """From the start: supply plugs planted on the given places (``places`` maps a supply plug's name to a target
    index; default: the demo's, l1 +0.20, l2 +0.10, l3 0.00, r1 -0.10, r2 -0.20, r3 left on its tray). ``spare``
    replaces plug_r3's state. ``over``: name -> state replaces that plug."""
    places = {"plug_l1": 0, "plug_l2": 1, "plug_l3": 2, "plug_r1": 3, "plug_r2": 4} if places is None else places
    out = list(S0)
    for name, k in places.items():
        out[NAME_INDEX[name]] = at_place(k)
    if spare is not None:
        out[NAME_INDEX["plug_r3"]] = spare
    for n, s in over.items():
        out[NAME_INDEX[n]] = s
    return out


DONE = bed_with()


def score_of(plugs):
    return evaluate(plugs)[0]


def glide(a, b, n):
    """n frames moving every plug of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None, hold=30):
    """From the start, every plug moved gradually to the five-place layout (or ``final``), then held there."""
    final = DONE if final is None else final
    return [S0] * 5 + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


class Geometry(unittest.TestCase):
    def test_lowest_point_and_tilt(self):
        s = plug(0.25, 0.0)
        self.assertAlmostEqual(lowest(s), L.BED_TOP)
        self.assertAlmostEqual(tilt_deg(s.quat_xyzw), 0.0)
        lying = plug(0.25, 0.0, tilt=90.0)
        self.assertAlmostEqual(lowest(lying), L.BED_TOP)
        self.assertAlmostEqual(tilt_deg(lying.quat_xyzw), 90.0, places=5)
        upside_down = plug(0.25, 0.0, tilt=180.0, axis="y")
        self.assertAlmostEqual(tilt_deg(upside_down.quat_xyzw), 0.0, places=4)  # a plain cylinder: either end up
        self.assertAlmostEqual(lowest(upside_down), L.BED_TOP)
        self.assertAlmostEqual(tilt_deg(plug(0.25, 0.0, tilt=172.0, axis="x").quat_xyzw), 8.0, places=4)
        self.assertAlmostEqual(tilt_deg(plug(0.25, 0.0, tilt=7.0, axis="y").quat_xyzw), 7.0, places=5)
        # The lowest point from the pose agrees with the bounding box the simulator reports.
        for tilt in (0.0, 5.0, 33.0, 90.0, 140.0):
            p = plug(0.25, 0.1, tilt=tilt)
            self.assertAlmostEqual(lowest(p), p.low[2], places=9)

    def test_planted(self):
        self.assertTrue(planted(at_place(0)))
        self.assertTrue(planted(at_place(0, bottom=L.BED_TOP + 0.0029)))  # held 2.9 mm up: counts (fidelity)
        self.assertFalse(planted(at_place(0, bottom=L.BED_TOP + 0.0035)))
        self.assertTrue(planted(at_place(0, tilt=9.9)))
        self.assertFalse(planted(at_place(0, tilt=10.5)))
        self.assertFalse(planted(at_place(0, tilt=90.0)))  # lying on the bed
        self.assertTrue(planted(at_place(0, tilt=180.0, axis="y")))  # the other end up: the same plain cylinder
        self.assertFalse(planted(at_place(0, tilt=169.0, axis="y")))  # 11 degrees off vertical
        self.assertFalse(planted(at_place(0, bottom=L.BED_TOP + L.PLUG_H)))  # standing on another plug
        self.assertFalse(planted(plug(0.17, 0.0, bottom=L.TOP)))  # on the table in front of the bed
        self.assertFalse(planted(plug(0.25, 0.32, bottom=L.BED_TOP)))  # centre past the bed's end (at that height)
        self.assertFalse(planted(S0[NAME_INDEX["plug_l1"]]))  # on its tray

    def test_fill_tolerance(self):
        for dx, dy, want in ((0.0149, 0.0, 0.2), (0.0151, 0.0, 0.0), (0.0, -0.0149, 0.2), (0.0, 0.0151, 0.0),
                             (0.0105, 0.0105, 0.2), (0.0107, 0.0107, 0.0)):
            with self.subTest(dx=dx, dy=dy):
                self.assertEqual(score_of(bed_with({"plug_l1": 0}, plug_l1=at_place(0, dx=dx, dy=dy))), want)

    def test_far_row_spots(self):
        """Each far-row spot is kept by a plug planted on it (upright, bottom within 3 mm of the bed, centre within
        10 mm in plan), whichever plug it is."""
        i = NAME_INDEX["far_3"]
        x0, y0, _ = L.START["far_3"]

        def kept(s):
            f = list(S0)
            f[i] = s
            return far_row(f)[2]

        self.assertEqual(far_row(S0), list(L.FAR_NAMES))
        self.assertEqual(kept(plug(x0 + 0.0099, y0)), "far_3")
        self.assertIsNone(kept(plug(x0 + 0.0101, y0)))
        self.assertIsNone(kept(plug(x0, y0 - 0.0101)))
        self.assertEqual(kept(plug(x0 + 0.007, y0 + 0.007)), "far_3")  # 9.9 mm on the diagonal
        self.assertIsNone(kept(plug(x0 + 0.0072, y0 + 0.0072)))  # 10.2 mm
        self.assertEqual(kept(plug(x0, y0, bottom=L.BED_TOP + 0.0029)), "far_3")
        self.assertIsNone(kept(plug(x0, y0, bottom=L.BED_TOP + 0.0035)))  # lifted
        self.assertEqual(kept(plug(x0, y0, tilt=9.5)), "far_3")
        self.assertIsNone(kept(plug(x0, y0, tilt=10.5)))
        self.assertIsNone(kept(plug(x0, y0, tilt=90.0)))  # knocked over where it stood
        f = list(S0)  # far_3 taken away and a tray plug stood on its spot: the spot is kept, by plug_r3
        f[i] = at_place(2)
        f[NAME_INDEX["plug_r3"]] = plug(x0 + 0.004, y0 - 0.003)
        self.assertEqual(far_row(f)[2], "plug_r3")

    def test_off_table(self):
        self.assertFalse(off_table(plug(0.5, 0.4, bottom=L.TOP)))
        self.assertTrue(off_table(plug(0.5, 0.4, bottom=L.TOP - 0.011)))
        self.assertTrue(off_table(plug(0.8, 0.0, bottom=0.02)))

    def test_valid(self):
        self.assertTrue(valid(S0[0]))
        nan = float("nan")
        self.assertFalse(valid(plug(nan, 0.0)))
        self.assertFalse(valid(state((0.3, 0, 0.8), quat=(0, 0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        self.assertFalse(valid(state((0.3, 0, 0.8), vel=(0, float("inf"), 0))))
        self.assertFalse(valid(None))

    def test_target_arithmetic(self):
        """The five places are level with the middle of each gap in the far row and mirrored across the tape: the
        far row 50 mm beyond it, the near row 50 mm on this side."""
        self.assertAlmostEqual(L.NEAR_X, 0.250)
        self.assertAlmostEqual(L.FAR_X - L.BED_X, L.BED_X - L.NEAR_X)
        self.assertAlmostEqual(L.FAR_X - L.BED_X, 0.050)
        self.assertTrue(np.allclose(TARGET_YS, [0.20, 0.10, 0.0, -0.10, -0.20]))
        far = sorted(y for _, y in L.FAR)
        self.assertTrue(np.allclose(np.diff(far), 0.10))
        for y in TARGET_YS:  # each place is half-way between two far plants, 50 mm from each
            d = sorted(abs(y - f) for f in far)
            self.assertAlmostEqual(d[0], 0.05)
            self.assertAlmostEqual(d[1], 0.05)
        # The wrong layouts are all 50 mm off: in line with the far plants, on the tape, mirrored twice as far.
        self.assertGreater(0.05, 3 * L.FILL_TOL)
        # Six supply plugs for five places; the places sit inside the bed and away from its ends.
        self.assertEqual(len(L.SUPPLY), len(L.TARGETS) + 1)
        for x, y in L.TARGETS:
            self.assertLess(abs(y) + L.PLUG_R, L.BED_HALF_LEN)
            self.assertLess(abs(x - L.BED_X) + L.PLUG_R, L.BED_W / 2)
        # No plug can be within 15 mm of two places.
        self.assertGreater(min(np.diff(sorted(TARGET_YS))), 2 * L.FILL_TOL)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["filled"], 0)
        self.assertEqual(d["far_spots_empty"], [])
        self.assertEqual(d["far_row_kept_by"], list(L.FAR_NAMES))
        self.assertEqual(d["planted"], [])
        r, _ = run([S0] * 20)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_all_five_scores_one(self):
        score, d = evaluate(DONE)
        self.assertEqual(score, 1.0)
        self.assertEqual(d["filled_by"], ["plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"])
        self.assertEqual(d["nearest_mm"], [0.0] * 5)
        r, _ = run(build())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_each_count_scores_a_fifth(self):
        names = ["plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"]
        for k in range(6):
            for chosen in itertools.combinations(range(5), k):
                with self.subTest(places=chosen):
                    places = {names[j]: j for j in chosen}
                    self.assertAlmostEqual(score_of(bed_with(places)), 0.2 * k)

    def test_every_tier_turns_up(self):
        seen = {score_of(bed_with(dict(list({"plug_l1": 0, "plug_l2": 1, "plug_l3": 2, "plug_r1": 3,
                                             "plug_r2": 4}.items())[:k]))) for k in range(6)}
        self.assertEqual(seen, {0.0, 0.2, 0.4, 0.6, 0.8, 1.0})

    def test_any_plug_may_fill_any_place(self):
        swapped = bed_with({"plug_r3": 0, "plug_r2": 1, "plug_r1": 2, "plug_l3": 3, "plug_l2": 4})
        self.assertEqual(score_of(swapped), 1.0)  # plug_l1 left on its tray as the spare

    def test_spare_anywhere_harmless(self):
        for spare in (plug(0.40, 0.45, bottom=L.TOP),  # on the table
                      plug(0.25, 0.27),  # standing on the bed beyond the end place
                      plug(0.30, 0.20, bottom=L.BED_TOP),  # standing on the drip tape
                      at_place(2, dy=0.03),  # next to the middle place, touching its plug
                      at_place(1, tilt=90.0),  # lying on the bed
                      at_place(3, bottom=L.BED_TOP + L.PLUG_H)):  # standing on a planted plug
            with self.subTest():
                self.assertEqual(score_of(bed_with(spare=spare)), 1.0)

    def test_two_plugs_at_one_place_count_once(self):
        both = bed_with({"plug_l1": 0, "plug_l2": 1}, plug_l3=at_place(0, dy=0.0149 * 0.99))
        # plug_l3 13 mm along from plug_l1's place cannot stand there (they would overlap); scored as found: 0.4.
        self.assertAlmostEqual(score_of(both), 0.4)

    def test_near_row_in_line_with_the_far_plants_scores_zero(self):
        """Not staggered: the near row set level with the far plants (y +-5, +-15 cm), 50 mm off every place."""
        names = ["plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"]
        units = list(S0)
        for n, y in zip(names, (0.15, 0.05, -0.05, -0.15, 0.25), strict=True):
            units[NAME_INDEX[n]] = plug(L.NEAR_X, y)
        self.assertEqual(score_of(units), 0.0)

    def test_near_row_on_the_drip_tape_scores_zero(self):
        units = list(S0)
        for n, y in zip(("plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"), TARGET_YS, strict=True):
            units[NAME_INDEX[n]] = plug(L.BED_X, y)
        score, d = evaluate(units)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["nearest_mm"], [50.0] * 5)

    def test_staggered_but_mirrored_wrong_scores_zero(self):
        """Level with the gaps but 100 mm in front of the tape (x 20.0 cm), or 30 mm in front of it (x 27.0 cm)."""
        for x in (0.200, 0.270):
            units = list(S0)
            for n, y in zip(("plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"), TARGET_YS, strict=True):
                units[NAME_INDEX[n]] = plug(x, y)
            with self.subTest(x=x):
                self.assertEqual(score_of(units), 0.0)
        units = list(S0)  # 14 mm in front of the right line still counts: inside the 15 mm
        for n, y in zip(("plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"), TARGET_YS, strict=True):
            units[NAME_INDEX[n]] = plug(L.NEAR_X - 0.014, y)
        self.assertEqual(score_of(units), 1.0)

    def test_far_row_disturbed_scores_zero(self):
        for name in L.FAR_NAMES:
            x0, y0, _ = L.START[name]
            for bad in (plug(x0 + 0.012, y0), plug(x0, y0 + 0.02), plug(x0, y0, tilt=15.0),
                        plug(x0, y0, tilt=90.0), plug(x0, y0, bottom=L.BED_TOP + 0.05),
                        plug(x0 + 0.08, y0, bottom=L.TOP)):
                with self.subTest(name=name):
                    score, d = evaluate(bed_with(**{name: bad}))
                    self.assertEqual(score, 0.0)
                    self.assertEqual(d["far_spots_empty"], [name])
        # Nudged 9 mm and leaning 8 degrees: not disturbed.
        self.assertEqual(score_of(bed_with(far_2=plug(L.FAR_X + 0.009, 0.15, tilt=8.0))), 1.0)

    def test_a_plug_off_the_table_scores_zero(self):
        score, d = evaluate(bed_with(spare=plug(0.3, 0.7, bottom=0.05)))
        self.assertEqual(score, 0.0)
        self.assertEqual(d["off_table"], ["plug_r3"])
        self.assertEqual(score_of(bed_with(spare=plug(0.3, 0.6, bottom=L.TOP - 0.009))), 1.0)  # not yet past 10 mm

    def test_success_needs_stillness(self):
        frames = build(hold=0)
        moving = [[state(s.pos + np.array([0, 0.0012 * (k % 2), 0]), s.quat_xyzw) for s in DONE] for k in range(20)]
        r, _ = run(frames + moving)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)  # every plug shuttling 1.2 mm a step: 11 mm of path in 0.5 s
        r, _ = run(frames + [DONE] * 9)
        self.assertFalse(r.success)  # 0.5 s of stillness not yet seen
        r, _ = run(frames + [DONE] * 11)
        self.assertTrue(r.success)
        r, _ = run([DONE] * 5)  # under 0.5 s of history: not still yet
        self.assertFalse(r.success)

    def test_success_needs_all_five(self):
        """Four places filled and everything still for 2 s: 0.8, never success (the trial runs on)."""
        four = bed_with({"plug_l1": 0, "plug_l2": 1, "plug_l3": 2, "plug_r1": 3})
        r, _ = run(build(final=four, hold=40))
        self.assertAlmostEqual(r.score, 0.8)
        self.assertTrue(r.details["all_still"])
        self.assertFalse(r.success)

    def test_score_is_the_state_at_hand(self):
        lifted = bed_with(plug_l3=at_place(2, bottom=L.BED_TOP + 0.03))
        r, t = run(build() + [lifted] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertAlmostEqual(r.score, 0.8)
        self.assertFalse(r.success)
        self.assertEqual(t.first_at[1.0], 25)  # reached on the last gliding frame


class Exploits(unittest.TestCase):
    def test_far_plants_used_as_supply(self):
        """Taking plugs from the far row instead of the trays leaves its spots empty: 0."""
        units = list(S0)
        for n, k in zip(L.FAR_NAMES[:5], range(5), strict=True):
            units[NAME_INDEX[n]] = at_place(k)
        score, d = evaluate(units)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["far_spots_empty"], list(L.FAR_NAMES[:5]))
        one = bed_with({"plug_l1": 0, "plug_l2": 1, "plug_l3": 2, "plug_r1": 3}, far_6=at_place(4))
        self.assertEqual(score_of(one), 0.0)

    def test_a_far_plant_swapped_for_a_tray_plug_is_the_same_bed(self):
        """The plugs are identical: a far plant moved to a place with a tray plug stood on its spot leaves the bed a
        photo cannot tell from the demo's, and scores the same (1.0); the score never asks which plug is which."""
        x0, y0, _ = L.START["far_6"]
        swap = bed_with({"plug_l1": 0, "plug_l2": 1, "plug_l3": 2, "plug_r1": 3}, far_6=at_place(4),
                        plug_r2=plug(x0, y0))
        score, d = evaluate(swap)
        self.assertEqual(score, 1.0)
        self.assertEqual(d["filled_by"][4], "far_6")
        self.assertEqual(d["far_row_kept_by"][5], "plug_r2")

    def test_moving_the_whole_far_row_to_suit(self):
        """Shoving the far row 50 mm along so the near row ends level with the far plants: 0."""
        units = bed_with()
        for n, y in L.FAR:
            units[NAME_INDEX[n]] = plug(L.FAR_X, y + 0.05)
        self.assertEqual(score_of(units), 0.0)

    def test_plugs_laid_down_or_stacked(self):
        """Plugs laid on their sides at the places, leaning 12 or 45 degrees, or stacked two high: nothing filled."""
        for bad in (dict(tilt=90.0), dict(tilt=45.0, axis="y"), dict(tilt=12.0)):
            units = list(S0)
            for n, k in zip(("plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"), range(5), strict=True):
                units[NAME_INDEX[n]] = at_place(k, **bad)
            with self.subTest(bad=bad):
                self.assertEqual(score_of(units), 0.0)
        stacked = bed_with({"plug_l1": 0}, plug_l2=at_place(0, bottom=L.BED_TOP + L.PLUG_H))
        self.assertAlmostEqual(score_of(stacked), 0.2)  # only the bottom one counts

    def test_plugs_held_above_their_places(self):
        units = list(S0)
        for n, k in zip(("plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"), range(5), strict=True):
            units[NAME_INDEX[n]] = at_place(k, bottom=L.BED_TOP + 0.005)
        self.assertEqual(score_of(units), 0.0)

    def test_near_row_on_the_table_in_front_of_the_bed(self):
        units = list(S0)
        for n, y in zip(("plug_l1", "plug_l2", "plug_l3", "plug_r1", "plug_r2"), TARGET_YS, strict=True):
            units[NAME_INDEX[n]] = plug(0.17, y, bottom=L.TOP)
        self.assertEqual(score_of(units), 0.0)

    def test_sweeping_the_spare_off_the_table(self):
        self.assertEqual(score_of(bed_with(spare=plug(0.7, -0.4, bottom=0.0))), 0.0)

    def test_one_arm_alone(self):
        """Gripping from above, an arm reaches only its own tray: three plugs, at most 0.6. The demo's right arm
        alone plants two (0.4), its left arm alone three (0.6)."""
        left_only = bed_with({"plug_l1": 0, "plug_l2": 1, "plug_l3": 2})
        self.assertAlmostEqual(score_of(left_only), 0.6)
        right_only = bed_with({"plug_r1": 3, "plug_r2": 4})
        self.assertAlmostEqual(score_of(right_only), 0.4)
        right_best = bed_with({"plug_r1": 3, "plug_r2": 4, "plug_r3": 2})
        self.assertAlmostEqual(score_of(right_best), 0.6)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for f in build():
            t.update(f)
        self.assertEqual(t.best, 1.0)
        r = t.update(S0)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        for f in build():
            r = t.update(f)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_putting_things_back_gradually_does_not_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 40) + [S0] * 10
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(t.steps, len(frames))

    def test_a_jump_elsewhere_does_not_reset(self):
        scattered = [plug(0.45, -0.25 + 0.045 * i, bottom=L.TOP) for i in range(12)]
        r, t = run(build() + [scattered] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.0)

    def test_a_plug_falling_back_onto_its_mark_is_not_a_reset(self):
        frames = build()
        frames += glide(frames[-1], S0, 30)
        i = NAME_INDEX["plug_l1"]
        x0, y0, z0 = L.START["plug_l1"]
        falling = plug(x0, y0, bottom=L.PAD_TOP + 0.10, vel=(0, 0, -1.4))
        f1 = list(S0)
        f1[i] = falling
        _, t = run(frames + [f1, S0])
        self.assertEqual(t.best, 1.0)
        resting = plug(x0, y0, bottom=L.PAD_TOP + 0.10)  # the same 10 cm jump by a plug that was still: a teleport
        f2 = list(S0)
        f2[i] = resting
        _, t = run(frames + [f2, S0])
        self.assertEqual(t.best, 0.0)

    def test_marks_need_every_plug_seated_on_its_mark(self):
        self.assertTrue(Tracker.at_start(S0))
        i = NAME_INDEX["plug_r2"]
        x0, y0, _ = L.START["plug_r2"]
        for bad in (plug(x0, y0, bottom=L.PAD_TOP, tilt=3.0), plug(x0, y0, bottom=L.PAD_TOP + 0.01),
                    plug(x0, y0, bottom=L.BED_TOP), plug(x0, y0, bottom=L.PAD_TOP, vel=(0, 0, 0.05)),
                    plug(x0 + 0.05, y0, bottom=L.PAD_TOP)):
            f = list(S0)
            f[i] = bad
            with self.subTest():
                self.assertFalse(Tracker.at_start(f))


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the six scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        good = DONE
        return [
            [plug(nan, 0.0), *good[1:]],
            [good[0], state((0.3, 0, L.TOP + 0.05), quat=(nan, 0, 0, nan)), *good[2:]],
            [*good[:11], plug(0.2, -0.36, bottom=L.PAD_TOP, vel=(0, inf, 0))],
            [*good[:11], plug(0.2, -0.36, bottom=L.PAD_TOP, spin=(nan, 0, 0))],
            [state((0.3, 0.0, L.TOP + 0.05), quat=(0, 0, 0, 0)), *good[1:]],
            [SimpleNamespace(pos=np.zeros(3)), *good[1:]],
            [None, *good[1:]],
            good[:11],  # a plug missing from the observation
        ]

    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        for frame in self.odd():
            with self.subTest():
                t = Tracker()
                for f in build():
                    t.update(f)
                prev, first, best, hist = t.prev, dict(t.first_at), t.best, list(t.history)
                r = t.update(frame)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.bad_steps, 1)
                self.assertIs(t.prev, prev)
                self.assertEqual(t.first_at, first)
                self.assertEqual(t.best, best)
                self.assertEqual(len(t.history), len(hist))
                json.dumps(r.details, allow_nan=False)
                r = t.update(DONE)
                self.assertEqual(r.score, 1.0)
                self.assertTrue(r.success)

    def test_far_and_strange_states(self):
        rng = np.random.default_rng(5)
        for _ in range(80):
            plugs = [state(rng.uniform(-2, 2, 3), tuple(rng.normal(size=4))) for _ in range(12)]
            r = Tracker().update(plugs)
            self.assertIn(round(r.score, 6), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
            json.dumps(r.details, allow_nan=False)

    def test_random_states_near_the_layout(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(600):
            plugs = []
            for i, s in enumerate(DONE):
                x, y, _ = s.pos
                on_bed = abs(x - L.BED_X) < L.BED_W / 2 and abs(y) < L.BED_HALF_LEN
                sigma = 0.004 if i < 6 else 0.012
                plugs.append(plug(x + rng.normal(0, sigma), y + rng.normal(0, sigma),
                                  bottom=(L.BED_TOP if on_bed else L.PAD_TOP) + abs(rng.normal(0, 0.0012)),
                                  tilt=abs(rng.normal(0, 3.0)), axis="x" if rng.random() < 0.5 else "y"))
            score, d = evaluate(plugs)
            seen.add(score)
            self.assertIn(score, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
            json.dumps(d, allow_nan=False)
            if score == 1.0:
                self.assertTrue(all(v is not None and v <= 15.05 for v in d["nearest_mm"]))
                self.assertEqual(d["far_spots_empty"], [])
        self.assertTrue({0.0, 0.6, 0.8, 1.0} <= seen)

    def test_fill_is_one_to_one(self):
        by, nearest = fill(DONE)
        self.assertEqual(len(set(b for b in by if b)), 5)
        self.assertEqual(len(nearest), 5)
        by, nearest = fill(S0)
        self.assertEqual(by, [None] * 5)
        self.assertEqual(nearest, [None] * 5)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, plugs):
        return dict(zip(self.task.PLUG_NAMES, plugs, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.PLUG_NAMES, L.NAMES)
        for spec, s in zip(t.PLUGS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((spec.radius, spec.height, spec.mass), (L.PLUG_R, L.PLUG_H, L.PLUG_MASS))
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 3 + 2 + 12)
        # The bed's three boxes meet with one flush top, side by side from x 190 to 410 mm.
        pieces = sorted(t.BED, key=lambda b: b.pos[0])
        edges = [(b.pos[0] - b.size[0] / 2, b.pos[0] + b.size[0] / 2) for b in pieces]
        self.assertAlmostEqual(edges[0][0], L.BED_X - L.BED_W / 2)
        self.assertAlmostEqual(edges[-1][1], L.BED_X + L.BED_W / 2)
        for (_, a1), (b0, _) in zip(edges, edges[1:]):
            self.assertAlmostEqual(a1, b0)
        for b in pieces:
            self.assertAlmostEqual(b.pos[2] + b.size[2] / 2, L.BED_TOP)
            self.assertAlmostEqual(b.size[1], 2 * L.BED_HALF_LEN)
        self.assertAlmostEqual(pieces[1].pos[0], L.BED_X)
        self.assertEqual(pieces[1].name, "drip_tape")
        # The supply plugs stand on their trays, the far row on the bed.
        for n, x, y in L.SUPPLY:
            self.assertLess(abs(x - L.PAD_X) + L.PLUG_R, L.PAD_SIZE[0] / 2)
            self.assertLess(abs(abs(y) - L.PAD_Y) + L.PLUG_R, L.PAD_SIZE[1] / 2 + 1e-9)
        # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
        for o in t.OBJECTS:
            half_y = o.size[1] / 2 if hasattr(o, "size") else o.radius
            self.assertLess(abs(o.pos[1]) + half_y, 0.45, o.name)

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                for _ in range(11):
                    ok = self.task.success(self.objects(DONE))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 12)
        for key in ("step", "score", "success", "filled", "filled_by", "nearest_mm", "planted", "far_row_kept_by",
                    "far_spots_empty", "far_max_move_mm", "far_max_tilt_deg", "off_table", "all_still", "bad_steps", "first_step",
                    "far_1", "plug_l1", "plug_r3"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["filled"], 5)
        self.assertEqual(rows[-1]["plug_l1"]["pos_mm"], [250.0, 200.0, 795.0])
        self.assertEqual(rows[-1]["plug_l1"]["path_mm"], 0.0)
        self.assertTrue(rows[-1]["plug_l1"]["planted"])
        self.assertEqual(rows[-1]["far_1"]["moved_mm"], 0.0)
        self.assertIsNone(rows[0]["plug_l1"]["path_mm"])

    def test_score_hook_reads_the_last_step_not_the_best(self):
        """score() is the state at hand: a plug lifted off its place after 1.0 was reached leaves 0.8."""
        self.task._TRACKER.reset()
        for f in build():
            self.task.success(self.objects(f))
        self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
        lifted = bed_with(plug_l3=at_place(2, bottom=L.BED_TOP + 0.03))
        self.assertFalse(self.task.success(self.objects(lifted)))
        self.assertAlmostEqual(self.task.score(self.objects(lifted)), 0.8)
        self.assertEqual(self.task._TRACKER.best, 1.0)

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(self.objects(S0)))
        finally:
            os.environ.pop("TASK_TRACE", None)
        self.assertFalse((HERE / "no_such_dir").exists())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                self.assertFalse(self.task.success(self.objects([plug(float("nan"), 0.0), *S0[1:]])))
                self.assertFalse(self.task.success({}))  # no plugs in the observation at all
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertFalse(rows[0]["valid_state"])
        self.assertFalse(rows[1]["valid_state"])

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_fills_every_place(self):
        t = self.task
        final = list(S0)
        for arm, steps in t.PLAN.items():
            for name, y in steps:
                self.assertTrue(name.startswith(f"plug_{arm[0]}"))  # each arm plants from its own tray
                final[NAME_INDEX[name]] = plug(L.NEAR_X, y)
        self.assertEqual(score_of(final), 1.0)
        self.assertEqual(sorted(y for steps in t.PLAN.values() for _, y in steps), sorted(TARGET_YS))

    @staticmethod
    def spells(wps):
        """(start, end, y, x) of each waypoint, on the plan's clock."""
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w.pos[1], w.pos[0]))
            t += w.duration
        return out

    def test_grippers_stay_apart(self):
        """Whenever both grippers are out over the bed at once, they are at least 0.20 m apart along y."""
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, ya, xa in left:
            for b0, b1, yb, xb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400)  # inside the run limit with the policy's 3 s hold

    def test_carried_plugs_clear_the_others(self):
        t = self.task
        self.assertGreaterEqual(t.HOVER_Z - t.GRIP_UP - (L.BED_TOP + L.PLUG_H), 0.025)  # over standing tops
        self.assertGreaterEqual(t.PICK_Z - 0.013 - L.PAD_TOP, 0.012)  # pad tips over the tray's top
        self.assertGreaterEqual(t.PLACE_Z - 0.013 - L.BED_TOP, 0.012)  # ... and over the bed's
        self.assertGreater(-2 * t.OPEN, 2 * L.PLUG_R + 0.050)  # 32 mm either side of a plug when open
        self.assertAlmostEqual(-2 * t.GRIP, 2 * L.PLUG_R - 0.008)  # the grip is commanded 8 mm narrower
        self.assertLess(t.GRIP_UP - 0.013, L.PLUG_H)  # the pads close on the plug, not over it
        self.assertLess(L.PLUG_H - t.GRIP_UP, 0.038)  # the plug's top stays clear of the gripper's palm
        # Open pads (gap plus two 8 mm pads) clear a planted neighbour 100 mm along y by more than 30 mm.
        self.assertGreater(0.100 - L.PLUG_R - (-t.OPEN + 0.008), 0.030)
        # On a tray the fingers close along y; the neighbours stand 60 mm along x, clear of the 20 mm wide pads.
        self.assertEqual(t.YAW, 0.0)
        self.assertGreater(L.SUPPLY_X[1] - L.SUPPLY_X[0] - L.PLUG_R - 0.010, 0.030)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_success_hold_outlasts_every_release(self):
        """A plug counts as planted while still gripped (within 3 mm), so the success hold must be longer than the
        settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        plan = self.task.demo()
        n = 0
        for arm, wps in plan.items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly stand it"):
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertTrue(settle.label == "settle" and let_go.label.startswith("let go"), arm)
                    self.assertLessEqual(settle.duration + let_go.duration + 0.5, self.task.SUCCESS_HOLD_S)
                    self.assertGreaterEqual(let_go.duration, 1.5)
                    n += 1
        self.assertEqual(n, 5)


if __name__ == "__main__":
    unittest.main()
