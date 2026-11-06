"""Offline tests for scoring.py (and the rigstate hooks and demo in task.py): synthetic object states and synthetic
arm snapshots, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import re
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, bottom_face_z, face_centre, in_band, nearest_disc, rim_gap, rim_top_z, riser_standing, rotation,
    supports, tilt_deg, top_face_z, valid,
)

TIPS = 0.013  # pad tips below the TCP
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
H = {**{n: v[0] for n, v in L.POTS.items()}, **{n: v[0] for n, v in L.RISERS.items()}}
R = {**{n: L.POT_R for n in L.POTS}, **{n: L.RISER_R for n in L.RISERS}}
START_XY = {**{n: (v[2], v[3]) for n, v in L.POTS.items()}, **{n: (v[2], v[3]) for n, v in L.RISERS.items()}}
DL = L.DISCS["disc_left"]
DR = L.DISCS["disc_right"]
DISC, FLOOR = L.DISC_TOP, L.FLOOR_TOP


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def cyl(name, x, y, bottom=0.0, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like cylinder ``name`` whose bottom face's centre is ``bottom`` above the table, centre over
    (x, y)."""
    h, r = H[name], R[name]
    az = abs(rotation(quat)[2, 2])
    c = np.array([x, y, L.TOP + bottom + h / 2 * az], float)
    a = rotation(quat)[:, 2]
    ext = np.array([r * math.sqrt(max(0.0, 1 - a[i] ** 2)) + h / 2 * abs(a[i]) for i in range(3)])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


START = {n: cyl(n, *START_XY[n]) for n in L.LOOSE}


def scene(**moved):
    """The starting layout with some objects replaced (name=state)."""
    out = dict(START)
    out.update(moved)
    return out


def a_on(riser, disc=DL, dx=0.0, dy=0.0, pot="pot_a", quat=Q0, base=DISC):
    """``riser`` standing on a disc (or the floor with base=FLOOR) and ``pot`` on top of it."""
    rs = cyl(riser, disc[0], disc[1], base)
    return {riser: rs, pot: cyl(pot, disc[0] + dx, disc[1] + dy, base + H[riser], quat)}


def tilted_riser(name, disc, deg, edge=DISC, axis="y"):
    """``name`` tilted ``deg`` degrees with the lowest point of its bottom rim resting at ``edge`` (a disc, or the
    floor), as a prop under its other edge would leave it: its bottom face's centre 30 mm x sin(tilt) higher."""
    return cyl(name, disc[0], disc[1], edge + L.RISER_R * math.sin(math.radians(deg)), quat=about(axis, deg))


def pot_flat_on(riser_state, riser, pot, step=0.0, quat=None):
    """``pot`` sitting on ``riser_state``'s top face (same tilt unless ``quat``), ``step`` higher than it."""
    top = face_centre(riser_state, H[riser], upper=True)
    q = riser_state.quat_xyzw if quat is None else quat
    return cyl(pot, top[0], top[1], top_face_z(riser_state, H[riser]) + step, quat=q)


NOMINAL = scene(**a_on("riser_l30", DL, pot="pot_a"), **a_on("riser_r20", DR, pot="pot_b"))
ONLY_A = scene(**a_on("riser_l30", DL, pot="pot_a"))
ONLY_B = scene(**a_on("riser_r20", DR, pot="pot_b"))


def arm(tcp, opening=1.0):
    tcp = np.array(tcp, float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


PARKED = {"left": arm((0.200, 0.250, L.TOP + 0.14)), "right": arm((0.200, -0.250, L.TOP + 0.14))}
READY = {"left": arm((0.323, 0.305, L.TOP + 0.26)), "right": arm((0.323, -0.305, L.TOP + 0.26))}
POT_HELD = (2 * L.POT_R - 0.008) / L.FULL_GAP


def holding_a(objects):
    """The left gripper closed on pot A where it is (TCP 21 mm above its bottom)."""
    p = objects["pot_a"]
    return {"left": arm((p.pos[0], p.pos[1], L.TOP + bottom_face_z(p, H["pot_a"]) + 0.021), POT_HELD),
            "right": PARKED["right"]}


def run(segments, hz=20.0):
    """Feed (objects, arms, steps) segments to a fresh tracker; return the last result and the tracker."""
    t = Tracker(hz)
    r = None
    for objs, arms, n in segments:
        for _ in range(n):
            r = t.update(objs, arms)
    return r, t


def end(objs, arms=PARKED, n=20):
    """Result for a still end state held for n control steps after a normal start."""
    r, _ = run([(START, READY, 5), (objs, arms, n)])
    return r


class Geometry(unittest.TestCase):
    def test_reference_rim_gaps(self):
        """The rim-gap table in task.json and the README, riser on a disc."""
        want = {"pot_a": {None: 45, "riser_l20": 25, "riser_l30": 15, "riser_l42": 3},
                "pot_b": {None: 34, "riser_r20": 14, "riser_r30": 4, "riser_r42": -8}}
        for pot, rows in want.items():
            for riser, mm in rows.items():
                base = DISC + (H[riser] if riser else 0.0)
                s = cyl(pot, *DL, base)
                self.assertAlmostEqual(rim_gap(s, H[pot]) * 1000, mm, places=6, msg=(pot, riser))
                if riser:  # a riser on the bare floor: 1 mm lower
                    s = cyl(pot, *DL, FLOOR + H[riser])
                    self.assertAlmostEqual(rim_gap(s, H[pot]) * 1000, mm + 1, places=6, msg=(pot, riser))

    def test_no_stack_of_two_risers_is_in_band(self):
        heights = sorted(L.RISERS[n][0] for n in L.RISERS)
        for i, a in enumerate(heights):
            for b in heights[i + 1:]:
                for pot in L.POTS:
                    for base in (DISC, FLOOR):
                        g = rim_gap(cyl(pot, *DL, base + a + b), H[pot])
                        self.assertFalse(L.GAP_MIN <= g <= L.GAP_MAX, msg=(pot, a, b))
                        self.assertLessEqual(g, 0.006 + 1e-9)

    def test_tilt_folds_and_raises_the_rim(self):
        up = cyl("pot_a", *DL, 0.036)
        flipped = cyl("pot_a", *DL, 0.036, quat=about("x", 180))
        self.assertAlmostEqual(tilt_deg(flipped), 0.0, places=6)
        self.assertAlmostEqual(rim_gap(flipped, H["pot_a"]), rim_gap(up, H["pot_a"]), places=9)
        t5 = cyl("pot_a", *DL, 0.036, quat=about("y", 5))
        self.assertAlmostEqual(tilt_deg(t5), 5.0, places=6)
        self.assertAlmostEqual(tilt_deg(cyl("pot_a", *DL, 0.036, quat=about("y", 175))), 5.0, places=6)
        raise_mm = (rim_top_z(t5, H["pot_a"]) - top_face_z(t5, H["pot_a"])) * 1000
        self.assertAlmostEqual(raise_mm, 35 * math.sin(math.radians(5)), places=6)  # about 3 mm
        side = cyl("riser_l30", *DL, 0.006, quat=about("x", 90))
        self.assertAlmostEqual(tilt_deg(side), 90.0, places=6)

    def test_faces(self):
        p = cyl("pot_b", *DR, 0.026)
        self.assertAlmostEqual(bottom_face_z(p, H["pot_b"]), 0.026, places=9)
        self.assertAlmostEqual(top_face_z(p, H["pot_b"]), 0.072, places=9)

    def test_supports_and_standing(self):
        r = cyl("riser_l30", *DL, DISC)
        self.assertTrue(riser_standing(r, H["riser_l30"]))
        self.assertTrue(riser_standing(cyl("riser_l30", *DL, FLOOR), H["riser_l30"]))
        self.assertFalse(riser_standing(cyl("riser_l30", 0.20, 0.0, 0.0), H["riser_l30"]))  # on the table
        self.assertFalse(riser_standing(cyl("riser_l30", *DL, DISC + 0.020), H["riser_l30"]))  # on a 20 mm riser
        ok, step, off = supports(cyl("pot_a", *DL, 0.036), H["pot_a"], r, H["riser_l30"])
        self.assertTrue(ok)
        self.assertAlmostEqual(step, 0.0, places=9)
        self.assertAlmostEqual(off, 0.0, places=9)

    def test_offset_is_between_the_touching_faces(self):
        """Seen from above, the offset is taken between the pot's bottom face and the riser's top face, whatever
        the tilt: a riser tilted 5 degrees under a centred pot has its top face's centre about 1.3 mm off its own
        centre."""
        from scoring import face_centre
        rs = cyl("riser_l30", *DL, DISC, quat=about("y", 5))
        top = face_centre(rs, H["riser_l30"], upper=True)
        self.assertAlmostEqual(top[0] - rs.pos[0], 0.015 * math.sin(math.radians(5)), places=9)
        pot = cyl("pot_a", top[0], top[1], top_face_z(rs, H["riser_l30"]))
        ok, step, off = supports(pot, H["pot_a"], rs, H["riser_l30"])
        self.assertTrue(ok)
        self.assertAlmostEqual(off, 0.0, places=9)
        flipped = cyl("pot_a", top[0], top[1], top_face_z(rs, H["riser_l30"]), quat=about("x", 180))
        self.assertAlmostEqual(supports(flipped, H["pot_a"], rs, H["riser_l30"])[2], 0.0, places=9)

    def test_nearest_disc(self):
        self.assertEqual(nearest_disc(cyl("pot_a", DR[0] + 0.01, DR[1], 0.03))[0], "disc_right")
        self.assertAlmostEqual(nearest_disc(cyl("pot_a", DL[0] + 0.012, DL[1] - 0.016, 0.03))[1], 0.020, places=9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([(START, READY, 30)])
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        risers_set = scene(**{"riser_l30": cyl("riser_l30", *DL, DISC), "riser_r20": cyl("riser_r20", *DR, DISC)})
        r, t = run([(START, READY, 10), (risers_set, READY, 40), (NOMINAL, holding_a(NOMINAL), 20),
                    (NOMINAL, PARKED, 20)])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["pot_a_riser"], "riser_l30")
        self.assertEqual(r.details["pot_b_riser"], "riser_r20")
        self.assertAlmostEqual(r.details["pot_a_rim_gap_mm"], 15.0)
        self.assertAlmostEqual(r.details["pot_b_rim_gap_mm"], 14.0)
        self.assertEqual(t.arms_unavailable, 0)

    def test_each_arm_alone_scores_half(self):
        self.assertEqual(end(ONLY_A).score, 0.5)
        self.assertEqual(end(ONLY_B).score, 0.5)

    def test_one_staged_one_wrong_scores_half(self):
        wrong_b = a_on("riser_r30", DR, pot="pot_b")  # rim 4 mm down: set, not staged
        r = end(scene(**a_on("riser_l30", DL), **wrong_b))
        self.assertEqual(r.score, 0.5)
        self.assertEqual((r.details["staged"], r.details["set"]), (1, 2))

    def test_wrong_riser_scores_quarter(self):
        for riser in ("riser_l20", "riser_l42"):  # pot A 25 mm down, or 3 mm
            self.assertEqual(end(scene(**a_on(riser, DL))).score, 0.25, msg=riser)
        for riser in ("riser_r30", "riser_r42"):  # pot B 4 mm down, or 8 mm above the rim
            self.assertEqual(end(scene(**a_on(riser, DR, pot="pot_b"))).score, 0.25, msg=riser)
        both_wrong = scene(**a_on("riser_l42", DL), **a_on("riser_r42", DR, pot="pot_b"))
        self.assertEqual(end(both_wrong).score, 0.25)

    def test_every_tier_is_reachable(self):
        got = {end(START).score, end(scene(**a_on("riser_l20", DL))).score, end(ONLY_A).score, end(NOMINAL).score}
        self.assertEqual(got, {0.0, 0.25, 0.5, 1.0})

    def test_pots_may_swap_discs(self):
        swapped = scene(**a_on("riser_l30", DR, pot="pot_a"), **a_on("riser_r20", DL, pot="pot_b"))
        self.assertEqual(end(swapped).score, 1.0)  # one on each disc, as the instruction says

    def test_any_riser_of_the_right_height_counts(self):
        other_30 = a_on("riser_r30", DL)  # pot A on the right set's 30 mm riser
        self.assertEqual(end(scene(**other_30)).score, 0.5)

    def test_riser_on_the_bare_floor_counts(self):
        r = end(scene(**a_on("riser_l30", (DL[0], DL[1] + 0.05), dy=-0.03, base=FLOOR)))
        # riser 50 mm off the disc's centre on the bare floor, the pot 20 mm off the disc's centre over it
        self.assertEqual(r.score, 0.5)
        self.assertAlmostEqual(r.details["pot_a_rim_gap_mm"], 16.0)


class ZeroConditions(unittest.TestCase):
    def test_pot_on_a_disc_without_a_riser(self):
        self.assertEqual(end(scene(pot_a=cyl("pot_a", *DL, DISC))).score, 0.0)
        self.assertEqual(end(scene(pot_b=cyl("pot_b", DR[0], DR[1] + 0.05, FLOOR))).score, 0.0)  # bare floor

    def test_pot_on_two_stacked_risers(self):
        stack = {"riser_l20": cyl("riser_l20", *DL, DISC), "riser_r20": cyl("riser_r20", *DL, DISC + 0.020),
                 "pot_a": cyl("pot_a", *DL, DISC + 0.040)}  # rim 5 mm down, but on a stack
        r = end(scene(**stack))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["pot_a_near_riser"], "riser_r20")

    def test_riser_on_its_side(self):
        side = cyl("riser_l42", *DL, DISC, quat=about("x", 90))  # lying: its top 60 mm up
        self.assertEqual(end(scene(riser_l42=side, pot_a=cyl("pot_a", *DL, DISC + 0.060))).score, 0.0)

    def test_riser_on_the_table_outside_the_planter(self):
        x, y = 0.20, 0.0
        objs = scene(riser_l30=cyl("riser_l30", x, y, 0.0), pot_a=cyl("pot_a", x, y, 0.030))
        self.assertEqual(end(objs).score, 0.0)

    def test_pot_tipped(self):
        self.assertEqual(end(scene(**a_on("riser_l30", DL, quat=about("x", 3.1)))).score, 0.0)
        self.assertEqual(end(scene(**a_on("riser_l30", DL, quat=about("x", 30)))).score, 0.0)

    def test_pot_off_its_disc(self):
        self.assertEqual(end(scene(**a_on("riser_l30", (DL[0] + 0.021, DL[1])))).score, 0.0)

    def test_pot_left_on_the_table_or_off_it(self):
        self.assertEqual(end(START).score, 0.0)
        self.assertEqual(end(scene(pot_a=cyl("pot_a", 0.6, 0.0, -0.4))).score, 0.0)

    def test_pot_still_held(self):
        self.assertEqual(end(ONLY_A, holding_a(ONLY_A)).score, 0.0)

    def test_pot_still_moving(self):
        t = Tracker()
        for _ in range(20):
            t.update(START, READY)
        rs = cyl("riser_l30", *DL, DISC)
        for i in range(12):  # 0.4 mm per step: 4 mm in 0.5 s, sliding at 8 mm/s
            r = t.update(scene(riser_l30=rs, pot_a=cyl("pot_a", DL[0] + 0.0004 * i, DL[1], 0.036)), PARKED)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["pot_a_still"])
        for _ in range(11):
            r = t.update(scene(riser_l30=rs, pot_a=cyl("pot_a", DL[0] + 0.0044, DL[1], 0.036)), PARKED)
        self.assertEqual(r.score, 0.5)  # stopped: still again after 0.5 s


class Boundaries(unittest.TestCase):
    """A value exactly on a bound the text states as inclusive lands in the tier the text gives it."""

    def test_pot_tilt_3_degrees(self):
        self.assertEqual(end(scene(**a_on("riser_l30", DL, quat=about("x", 3.0)))).score, 0.5)
        self.assertEqual(end(scene(**a_on("riser_l30", DL, quat=about("y", -3.0)))).score, 0.5)
        self.assertEqual(end(scene(**a_on("riser_l30", DL, quat=about("x", 177.0)))).score, 0.5)
        self.assertEqual(end(scene(**a_on("riser_l30", DL, quat=about("x", 3.1)))).score, 0.0)

    def test_disc_offset_20_mm(self):
        rs = cyl("riser_l30", *DL, DISC)
        for dx, dy, want in ((0.020, 0.0, 0.5), (-0.012, 0.016, 0.5), (0.0201, 0.0, 0.0)):
            objs = scene(riser_l30=rs, pot_a=cyl("pot_a", DL[0] + dx, DL[1] + dy, DISC + 0.030))
            self.assertEqual(end(objs).score, want, msg=(dx, dy))

    def test_riser_step_1_5_mm(self):
        rs = cyl("riser_l30", *DL, DISC)
        for step, want in ((0.0015, 0.5), (-0.0015, 0.5), (0.0016, 0.0), (-0.0016, 0.0)):
            objs = scene(riser_l30=rs, pot_a=cyl("pot_a", *DL, DISC + 0.030 + step))
            self.assertEqual(end(objs).score, want, msg=step)

    def test_riser_axis_offset_35_mm(self):
        rs = cyl("riser_l30", DL[0] + 0.015, DL[1], DISC)
        for dx, want in ((-0.020, 0.5), (-0.0201, 0.0)):  # pot 20 mm the other way: 35 mm apart (a synthetic state)
            pot = cyl("pot_a", DL[0] + dx, DL[1], DISC + 0.030)
            r = end(scene(riser_l30=rs, pot_a=pot))
            if dx == -0.020:
                self.assertAlmostEqual(r.details["pot_a_riser_axis_off_mm"], 35.0)
                self.assertEqual(r.score, want)
            else:
                self.assertEqual(r.score, 0.0)  # 35.1 mm apart, and 20.1 mm off the disc

    def test_riser_axis_offset_alone(self):
        rs = cyl("riser_l30", DL[0] + 0.0351, DL[1], DISC)
        self.assertEqual(end(scene(riser_l30=rs, pot_a=cyl("pot_a", *DL, DISC + 0.030))).score, 0.0)
        rs = cyl("riser_l30", DL[0] + 0.035, DL[1], DISC)
        self.assertEqual(end(scene(riser_l30=rs, pot_a=cyl("pot_a", *DL, DISC + 0.030))).score, 0.5)

    def test_riser_bottom_4_to_7_5_mm(self):
        for low, want in ((0.004, 0.25), (0.0075, 0.25), (0.0039, 0.0), (0.0076, 0.0)):
            # the 20 mm riser under pot A: rim 25 mm down at low 6 mm; shift it with low: always out of band
            objs = scene(riser_l20=cyl("riser_l20", *DL, low), pot_a=cyl("pot_a", *DL, low + 0.020))
            self.assertEqual(end(objs).score, want, msg=low)
        for low, want in ((0.004, 0.5), (0.0075, 0.5), (0.0039, 0.0), (0.0076, 0.0)):  # the 30 mm riser: 17 / 13.5
            objs = scene(riser_l30=cyl("riser_l30", *DL, low), pot_a=cyl("pot_a", *DL, low + 0.030))
            self.assertEqual(end(objs).score, want, msg=low)

    def test_riser_tilt_2_degrees(self):
        for deg, want in ((2.0, 0.5), (2.1, 0.0)):
            rs = cyl("riser_l30", *DL, DISC, quat=about("y", deg))
            pot = cyl("pot_a", *DL, top_face_z(rs, H["riser_l30"]))
            self.assertEqual(end(scene(riser_l30=rs, pot_a=pot)).score, want, msg=deg)

    def test_rim_gap_band_ends(self):
        """Both ends of the 10 to 20 mm band are inclusive."""
        for gap, want in ((0.010, True), (0.020, True), (0.015, True), (0.0099, False), (0.0201, False),
                          (-0.008, False)):
            self.assertEqual(in_band(gap), want, msg=gap)

    def test_rim_gap_10_mm_in_a_scene(self):
        """Exactly 10 mm with pot B on the 20 mm riser, every tolerance used: riser bottom 7.5 mm up, pot 1.5 mm
        above it and tilted until 46 cos(t) + 35 sin(t) = 47 mm. (No set pot can end 18.6 to 20.2 mm down, so the
        band's upper end is only checked on the gap itself, above.)"""
        f = lambda t: 0.046 * math.cos(t) + 0.035 * math.sin(t) - 0.047  # noqa: E731
        lo, hi = 0.0, math.radians(3.0)
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        t10 = math.degrees(hi)
        for deg, want in ((t10, 0.5), (t10 + 0.05, 0.25)):
            objs = scene(riser_r20=cyl("riser_r20", *DR, 0.0075),
                         pot_b=cyl("pot_b", *DR, 0.0075 + 0.020 + 0.0015, quat=about("x", deg)))
            r = end(objs)
            self.assertLessEqual(r.details["pot_b_tilt_deg"], 3.0)
            self.assertEqual(r.score, want, msg=deg)
        self.assertAlmostEqual(rim_gap(cyl("pot_b", *DR, 0.029, quat=about("x", t10)), H["pot_b"]), 0.010, places=9)

    def test_let_go_60_mm(self):
        c = ONLY_A["pot_a"].pos
        for d, want in ((0.060, 0.5), (0.0599, 0.0)):
            arms = {"left": arm((c[0], c[1], c[2] + d), 1.0), "right": PARKED["right"]}
            self.assertEqual(end(ONLY_A, arms).score, want, msg=d)
        arms = {"left": PARKED["left"], "right": arm((c[0] + 0.05, c[1], c[2]), 1.0)}  # the other gripper counts too
        self.assertEqual(end(ONLY_A, arms).score, 0.0)

    def test_still_path_is_strict(self):
        rs = cyl("riser_l30", *DL, DISC)
        for per_step, want in ((0.00029, 0.5), (0.0003, 0.0)):  # 10 steps: 2.9 mm, or exactly 3 mm (not shorter)
            t = Tracker()
            for i in range(11):
                r = t.update(scene(riser_l30=rs, pot_a=cyl("pot_a", DL[0] + per_step * i, DL[1], 0.036)), PARKED)
            self.assertEqual(r.score, want, msg=per_step)


class Stillness(unittest.TestCase):
    def test_velocity_chatter_is_ignored(self):
        """A resting pot reports 10-45 mm/s of chatter: stillness is read from its path, never its velocity."""
        t = Tracker()
        rs = cyl("riser_l30", *DL, DISC)
        for i in range(30):
            v = (0.045 * (-1) ** i, 0.03, -0.02)
            r = t.update(scene(riser_l30=rs, pot_a=cyl("pot_a", *DL, 0.036, vel=v, spin=(0.4, 0, 0))), PARKED)
        self.assertTrue(r.details["pot_a_still"])
        self.assertEqual(r.score, 0.5)

    def test_needs_half_a_second_of_history(self):
        t = Tracker()
        for k in range(1, 13):
            r = t.update(NOMINAL, PARKED)
            self.assertEqual(r.score, 1.0 if k >= 11 else 0.0, msg=k)

    def test_landing_needs_half_a_second(self):
        t = Tracker()
        above = scene(**a_on("riser_l30", DL), **a_on("riser_r20", DR, pot="pot_b"))
        above["pot_a"] = cyl("pot_a", *DL, 0.039)  # held 3 mm above its riser
        for _ in range(20):
            t.update(above, PARKED)
        scores = [t.update(NOMINAL, PARKED).score for _ in range(12)]
        self.assertEqual(scores[:10], [0.5] * 10)  # pot A dropped 3 mm: not still for 0.5 s
        self.assertEqual(scores[10:], [1.0, 1.0])


class Exploits(unittest.TestCase):
    def test_holding_the_pot_at_the_right_height_instead_of_placing(self):
        floating = scene(**{"riser_l30": cyl("riser_l30", *DL, DISC), "pot_a": cyl("pot_a", *DL, 0.036)})
        self.assertEqual(end(floating, holding_a(floating)).score, 0.0)
        hover = scene(pot_a=cyl("pot_a", *DL, 0.036))  # held at the staged height with no riser at all
        self.assertEqual(end(hover, holding_a(hover)).score, 0.0)

    def test_pushing_a_pot_over_the_wall_onto_the_floor(self):
        self.assertEqual(end(scene(pot_a=cyl("pot_a", DL[0], DL[1] + 0.01, DISC))).score, 0.0)

    def test_knocked_over_riser_with_the_pot_on_it(self):
        side = cyl("riser_l30", *DL, DISC, quat=about("x", 90))  # its curved side up 60 mm: pot rim -9 mm
        self.assertEqual(end(scene(riser_l30=side, pot_a=cyl("pot_a", *DL, DISC + 0.060))).score, 0.0)

    def test_the_wrong_pot_on_a_riser_scores_on_its_merits(self):
        b_on_30 = a_on("riser_l30", DL, pot="pot_b")  # pot B 4 mm down: set but not staged
        self.assertEqual(end(scene(**b_on_30)).score, 0.25)

    def test_tilting_cannot_bring_the_tall_riser_into_band(self):
        for deg in (0.0, 3.0, 5.0, 20.0):
            objs = scene(**a_on("riser_l42", DL, quat=about("x", deg)))
            self.assertLess(end(objs).score, 0.5, msg=deg)

    def test_gripper_hovering_just_inside_60_mm(self):
        c = ONLY_A["pot_a"].pos
        hover = {"left": arm((c[0], c[1], c[2] + 0.055), 1.0), "right": PARKED["right"]}
        self.assertEqual(end(ONLY_A, hover).score, 0.0)

    def test_pot_on_a_riser_standing_on_the_other_pot(self):
        objs = scene(**{"pot_b": cyl("pot_b", *DL, DISC), "riser_l20": cyl("riser_l20", *DL, DISC + 0.046),
                        "pot_a": cyl("pot_a", *DL, DISC + 0.066)})
        self.assertEqual(end(objs).score, 0.0)

    def test_open_gripper_left_hovering_is_not_moved_away(self):
        """Let go but not moved away: an open gripper with its pad tips 20 mm over a staged pot's top (5 mm over the
        planter's rim for pot A) is closer than 60 mm to the pot's centre: that pot is not set. The instruction asks
        for both grippers to be moved away; 30 mm over the top is far enough, and so is the demo's 14 cm."""
        for pot, side, gone, want_mm in (("pot_a", "left", 0.5, 50.5), ("pot_b", "right", 0.5, 56.0)):
            c = NOMINAL[pot].pos
            top = c[2] + H[pot] / 2
            for over, want in ((0.020, gone), (0.030, 1.0)):
                arms = dict(PARKED)
                arms[side] = arm((c[0], c[1], top + over + TIPS), 1.0)
                r = end(NOMINAL, arms)
                self.assertEqual(r.score, want, msg=(pot, over))
                if over == 0.020:
                    self.assertAlmostEqual(r.details[f"{pot}_tcp_mm"], want_mm)
        up = {"left": arm((*DL, L.TOP + 0.140), 1.0), "right": arm((*DR, L.TOP + 0.140), 1.0)}  # the demo's lift
        r = end(NOMINAL, up)
        self.assertEqual(r.score, 1.0)
        self.assertGreater(min(r.details["pot_a_tcp_mm"], r.details["pot_b_tcp_mm"]), 85.0)

    def test_no_hold_reaches_60_mm(self):
        """Pinching only pot B's top 6 mm puts the TCP 30 mm from its centre; closed fingers pressing on its top,
        36 mm. Neither counts as let go."""
        c = ONLY_B["pot_b"].pos
        for tcp_up, mm in ((H["pot_b"] / 2 - 0.006 + TIPS, 30.0), (H["pot_b"] / 2 + TIPS, 36.0)):
            arms = {"left": PARKED["left"], "right": arm((c[0], c[1], c[2] + tcp_up), POT_HELD)}
            r = end(ONLY_B, arms)
            self.assertEqual(r.score, 0.0)
            self.assertAlmostEqual(r.details["pot_b_tcp_mm"], mm)

    def test_propped_tilted_riser(self):
        """A riser tilted on a prop under one edge (its low edge on the disc), the pot flat on its top: at 4 to 5
        degrees the riser is not upright, so nothing is set (with a 5 degree limit the 20 mm riser put pot A in the
        band and the right risers fell out of it). At the 2 degree limit the wrong riser stays out of the band and
        the right ones in it; a riser straddling the disc's edge (0.95 degrees) changes almost nothing."""
        for pot, riser, disc, deg in (("pot_a", "riser_l20", DL, 5.0), ("pot_a", "riser_l30", DL, 5.0),
                                      ("pot_b", "riser_r20", DR, 4.0)):
            rs = tilted_riser(riser, disc, deg)
            self.assertEqual(end(scene(**{riser: rs, pot: pot_flat_on(rs, riser, pot)})).score, 0.0,
                             msg=(pot, riser, deg))
        for pot, riser, disc, want, gap in (("pot_a", "riser_l20", DL, 0.25, 22.76),
                                            ("pot_a", "riser_l30", DL, 0.5, 12.8),
                                            ("pot_b", "riser_r20", DR, 0.5, 11.8)):
            rs = tilted_riser(riser, disc, 2.0)
            r = end(scene(**{riser: rs, pot: pot_flat_on(rs, riser, pot)}))
            self.assertEqual(r.score, want, msg=(pot, riser))
            self.assertAlmostEqual(r.details[f"{pot}_rim_gap_mm"], gap, delta=0.05)
        straddle = math.degrees(math.atan(L.DISC_T / (2 * L.RISER_R)))  # one edge on the disc, one on the floor
        for pot, riser, disc, gap in (("pot_a", "riser_l30", DL, 14.93), ("pot_b", "riser_r20", DR, 13.93)):
            rs = tilted_riser(riser, disc, straddle, edge=FLOOR)
            r = end(scene(**{riser: rs, pot: pot_flat_on(rs, riser, pot)}))
            self.assertEqual(r.score, 0.5, msg=pot)
            self.assertAlmostEqual(r.details[f"{pot}_rim_gap_mm"], gap, delta=0.01)

    def test_tolerances_cannot_stage_a_wrong_riser(self):
        """Every corner of what the scorer allows (riser bottom 4 or 7.5 mm up, riser tilt 0 or 2 degrees, pot 1.5 mm
        below, level with or above the riser's top, pot tilt 0, 1.5 or 3 degrees): no wrong riser gives a staged
        pot, while pot A on the 30 mm riser is staged at every corner (10.2 to 18.5 mm)."""
        span = {}
        for pot, riser in (("pot_a", "riser_l20"), ("pot_a", "riser_l30"), ("pot_a", "riser_l42"),
                           ("pot_b", "riser_r20"), ("pot_b", "riser_r30"), ("pot_b", "riser_r42")):
            disc = DL if pot == "pot_a" else DR
            for low in (L.RISER_LOW_MIN, L.RISER_LOW_MAX):
                for rdeg, rax in ((0.0, "y"), (L.RISER_UPRIGHT_DEG, "y"), (L.RISER_UPRIGHT_DEG, "x")):
                    rs = cyl(riser, *disc, low, quat=about(rax, rdeg))
                    for step in (-L.STEP_TOL, 0.0, L.STEP_TOL):
                        for pdeg, pax in ((0.0, "y"), (1.5, "x"), (L.POT_UPRIGHT_DEG, "y"), (L.POT_UPRIGHT_DEG, "x")):
                            p = pot_flat_on(rs, riser, pot, step=step, quat=about(pax, pdeg))
                            r = end(scene(**{riser: rs, pot: p}), n=11)
                            self.assertTrue(r.details[f"{pot}_set"], msg=(pot, riser, low, rdeg, step, pdeg))
                            span.setdefault((pot, riser), []).append(r.details[f"{pot}_rim_gap_mm"])
                            if riser in ("riser_l20", "riser_l42", "riser_r30", "riser_r42"):
                                self.assertFalse(r.details[f"{pot}_staged"], msg=(pot, riser, low, rdeg, step, pdeg))
                            if riser == "riser_l30":
                                self.assertTrue(r.details[f"{pot}_staged"], msg=(low, rdeg, step, pdeg))
        self.assertAlmostEqual(min(span[("pot_a", "riser_l20")]), 20.22, delta=0.01)  # the closest a wrong riser gets
        self.assertAlmostEqual(max(span[("pot_b", "riser_r30")]), 7.52, delta=0.01)
        self.assertAlmostEqual(max(span[("pot_a", "riser_l42")]), 6.53, delta=0.01)
        self.assertAlmostEqual(min(span[("pot_a", "riser_l30")]), 10.22, delta=0.01)
        self.assertAlmostEqual(max(span[("pot_a", "riser_l30")]), 18.52, delta=0.01)

    def test_old_tolerance_corners_now_score_their_merits(self):
        """Two corners the earlier, looser limits let through: pot B on the 30 mm riser with the riser's bottom 3 mm
        up and tilted 5 degrees and the pot 3 mm low (10.1 mm: staged), and pot A on the 20 mm riser with the riser's
        bottom 9 mm up, the pot 3 mm high and tilted 5 degrees (16.1 mm: staged). Neither is set now; the nearest
        corners still allowed leave both on the wrong side of the band."""
        rs = cyl("riser_r30", *DR, 0.003, quat=about("y", 5.0))
        self.assertEqual(end(scene(riser_r30=rs, pot_b=pot_flat_on(rs, "riser_r30", "pot_b", -0.003, Q0))).score, 0.0)
        rs = cyl("riser_l20", *DL, 0.009)
        self.assertEqual(end(scene(riser_l20=rs, pot_a=pot_flat_on(rs, "riser_l20", "pot_a", 0.003,
                                                                   about("y", 5.0)))).score, 0.0)
        rs = cyl("riser_r30", *DR, L.RISER_LOW_MIN, quat=about("y", L.RISER_UPRIGHT_DEG))
        r = end(scene(riser_r30=rs, pot_b=pot_flat_on(rs, "riser_r30", "pot_b", -L.STEP_TOL, Q0)))
        self.assertEqual(r.score, 0.25)
        self.assertLess(r.details["pot_b_rim_gap_mm"], 8.1)
        rs = cyl("riser_l20", *DL, L.RISER_LOW_MAX)
        r = end(scene(riser_l20=rs, pot_a=pot_flat_on(rs, "riser_l20", "pot_a", L.STEP_TOL,
                                                      about("y", L.POT_UPRIGHT_DEG))))
        self.assertEqual(r.score, 0.25)
        self.assertGreater(r.details["pot_a_rim_gap_mm"], 20.2)

    def test_pot_propped_on_its_own_edge(self):
        """Pot A leaning on something with its bottom edge on the 20 mm riser: its bottom face's centre rises
        35 mm x sin(tilt). Up to about 2.45 degrees it is set but 22 mm down (0.25); more, and it is off the riser
        (step over 1.5 mm) and not set."""
        rs = cyl("riser_l20", *DL, DISC)
        for deg, want in ((2.4, 0.25), (2.9, 0.0)):
            lift = L.POT_R * math.sin(math.radians(deg))
            r = end(scene(riser_l20=rs, pot_a=pot_flat_on(rs, "riser_l20", "pot_a", lift, about("x", deg))))
            self.assertEqual(r.score, want, msg=deg)
            self.assertGreater(r.details["pot_a_rim_gap_mm"], 21.0)

    def test_two_same_height_risers_side_by_side(self):
        """The instruction says one riser; two of one height side by side under a pot give the same rim gap and
        count as one (no extra score, nothing gained)."""
        objs = scene(riser_l30=cyl("riser_l30", DL[0] - 0.030, DL[1], DISC),
                     riser_r30=cyl("riser_r30", DL[0] + 0.030, DL[1], DISC), pot_a=cyl("pot_a", *DL, DISC + 0.030))
        r = end(objs)
        self.assertEqual(r.score, 0.5)
        self.assertAlmostEqual(r.details["pot_a_rim_gap_mm"], 15.0)

    def test_unreadable_arms_are_counted(self):
        r, t = run([(NOMINAL, None, 20)])
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 20)
        self.assertFalse(r.details["arms_available"])
        bad = {"left": arm((np.nan, 0, 1)), "right": PARKED["right"]}
        _, t = run([(NOMINAL, bad, 12)])
        self.assertEqual(t.arms_unavailable, 12)


class Robustness(unittest.TestCase):
    def test_bad_states_score_zero_and_leave_the_tracker(self):
        t = Tracker()
        for _ in range(15):
            t.update(NOMINAL, PARKED)
        window = [list(map(tuple, t.window[p])) for p in L.POTS]
        steps_valid = t.prev
        nan = SimpleNamespace(**{**vars(NOMINAL["pot_a"]), "pos": np.array([np.nan, 0, 1])})
        inf = SimpleNamespace(**{**vars(NOMINAL["riser_r20"]), "lin_vel": np.array([np.inf, 0, 0])})
        zero = SimpleNamespace(**{**vars(NOMINAL["pot_b"]), "quat_xyzw": np.zeros(4)})
        no_attr = SimpleNamespace(pos=(0.27, 0.11, 0.8))
        missing = {k: v for k, v in NOMINAL.items() if k != "riser_l42"}
        cases = [scene(**{**NOMINAL, "pot_a": nan}), scene(**{**NOMINAL, "riser_r20": inf}),
                 scene(**{**NOMINAL, "pot_b": zero}), scene(**{**NOMINAL, "riser_l20": no_attr}), missing, None, {}]
        for bad in cases:
            r = t.update(bad, PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.bad_steps, len(cases))
        self.assertEqual([list(map(tuple, t.window[p])) for p in L.POTS], window)
        self.assertIs(t.prev, steps_valid)
        self.assertEqual(t.update(NOMINAL, PARKED).score, 1.0)  # picks up where it was

    def test_valid(self):
        self.assertTrue(valid(NOMINAL["pot_a"]))
        self.assertFalse(valid(SimpleNamespace(pos=(0, 0, 0))))
        self.assertFalse(valid(SimpleNamespace(**{**vars(NOMINAL["pot_a"]), "pos": np.zeros(2)})))

    def test_result_is_idempotent(self):
        t = Tracker()
        for _ in range(15):
            t.update(NOMINAL, PARKED)
        steps = t.steps
        a = [t.result(NOMINAL, PARKED) for _ in range(5)]
        self.assertEqual({r.score for r in a}, {1.0})
        self.assertEqual(t.steps, steps)
        self.assertEqual(a[0].details, a[-1].details)

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker()
        for _ in range(300):
            objs = {}
            for n in L.LOOSE:
                q = rng.normal(size=4)
                c = np.array([rng.uniform(0.0, 0.6), rng.uniform(-0.5, 0.5), L.TOP + rng.uniform(-0.1, 0.2)])
                objs[n] = SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=rng.normal(size=3), ang_vel=rng.normal(size=3),
                                          low=c - 0.03, high=c + 0.03)
            arms = {"left": arm(rng.uniform(0, 0.5, 3) + [0, 0, L.TOP], rng.uniform()),
                    "right": arm(rng.uniform(0, 0.5, 3) + [0, -0.5, L.TOP], rng.uniform())}
            r = t.update(objs, arms if rng.uniform() < 0.8 else None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(r.details)  # the trace row must serialise


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for _ in range(30):
            t.update(NOMINAL, None)
        self.assertEqual(t.arms_unavailable, 30)
        r = t.update(START, READY)  # everything jumps back to its place in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertEqual(len(t.window["pot_a"]), 1)
        self.assertEqual(r.score, 0.0)

    def test_jittered_teleport_resets(self):
        rng = np.random.default_rng(3)
        jit = {n: cyl(n, START_XY[n][0] + rng.uniform(-0.03, 0.03), START_XY[n][1] + rng.uniform(-0.03, 0.03),
                      0.0, quat=about("z", rng.uniform(-180, 180))) for n in L.LOOSE}
        t = Tracker()
        for _ in range(25):
            t.update(NOMINAL, PARKED)
        t.update(jit, READY)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = Tracker()
        a0, a1 = START["pot_a"].pos, NOMINAL["pot_a"].pos
        path = [scene(pot_a=cyl("pot_a", *(a0 + (a1 - a0) * k / 40)[:2], 0.0)) for k in range(41)]
        for objs in path + path[::-1]:
            t.update(objs, READY)
        self.assertEqual(t.steps, 82)

    def test_a_jump_elsewhere_does_not_reset(self):
        t = Tracker()
        for _ in range(10):
            t.update(START, READY)
        t.update(NOMINAL, READY)  # a jump, but not back onto the starting places
        self.assertEqual(t.steps, 11)
        t.update(scene(pot_a=cyl("pot_a", 0.30, 0.0, 0.0)), READY)  # back for all but one object
        self.assertEqual(t.steps, 12)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo, texts."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.spec = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()

    @staticmethod
    def feed(arms):
        rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in arms.items()})

    def test_objects_and_constants(self):
        T = self.task
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names[:7], ["planter_floor", "planter_wall_near", "planter_wall_far", "planter_wall_left",
                                     "planter_wall_right", "disc_left", "disc_right"])
        self.assertEqual(names[7:], list(L.LOOSE))
        mass = {o.name: o.mass for o in T.OBJECTS}
        self.assertTrue(all(mass[n] is None for n in names[:7]))
        self.assertEqual([mass[n] for n in L.LOOSE], [0.100, 0.130, 0.015, 0.022, 0.030, 0.022, 0.030, 0.015])
        self.assertEqual(T.SCENE, "station")
        self.assertEqual(T.SCORE_OVER, "end")
        self.assertEqual(T.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(self.spec["run_limit"], "3,600 steps (3 minutes at 20 Hz)")
        self.assertEqual(self.spec["category"], "Impact on jobs")
        self.assertIn("demo: 1.0", self.spec["scores"])
        self.assertNotIn("media", self.spec)
        # the planter as the setup states it
        self.assertAlmostEqual(L.OUTER_X, 0.176)
        self.assertAlmostEqual(L.OUTER_Y, 0.356)
        self.assertAlmostEqual(L.RIM, 0.086)
        for txt in ("17.6 cm along x, 35.6 cm along y and 8.6 cm tall", "x 19.0 to 35.0", "y -17.0 to +17.0",
                    "9.0 cm across and 1 mm thick", "3.5 cm tall, 100 g", "4.6 cm tall and 130 g",
                    "2.0 cm tall (15 g), 3.0 cm (22 g) and 4.2 cm (30 g)", "x 20.0, y +38.5", "x 20.0, y -38.5",
                    "2.0 cm at x 17.0, 3.0 cm at x 27.0, 4.2 cm at x 37.0",
                    "3.0 cm at x 17.0, 4.2 cm at x 27.0, 2.0 cm at x 37.0"):
            self.assertIn(txt, self.spec["setup"])

    def test_numbers_in_the_scoring_text(self):
        s = self.spec["scoring"]
        for txt in ("within 3 degrees of vertical", "within 20 mm of a disc", "within 1.5 mm", "within 35 mm",
                    "upright (within 2 degrees)", "4 to 7.5 mm above", "shorter than 3 mm over the last 0.5 s",
                    "at least 60 mm", "20.2 mm or more down", "two risers of one height side by side",
                    "rim gap is 10 to 20 mm", "Possible scores: 0, 0.25, 0.5, 1.0", "held for 1 s",
                    "25, 15 and 3 mm", "14, 4 and -8 mm", "45 mm with no riser", "34 mm with no riser"):
            self.assertIn(txt, s)
        self.assertEqual((L.POT_UPRIGHT_DEG, L.DISC_TOL, L.STEP_TOL, L.AXIS_TOL), (3.0, 0.020, 0.0015, 0.035))
        self.assertEqual((L.RISER_UPRIGHT_DEG, L.RISER_LOW_MIN, L.RISER_LOW_MAX), (2.0, 0.004, 0.0075))
        self.assertEqual((L.STILL_PATH, L.STILL_STEPS), (0.003, 10))
        self.assertEqual((L.RELEASE_DIST, L.GAP_MIN, L.GAP_MAX), (0.060, 0.010, 0.020))

    def test_instruction_is_identical_everywhere(self):
        self.assertEqual(self.task.INSTRUCTION, self.spec["instruction"])
        self.assertIn("let go and move both grippers away from the pots", self.task.INSTRUCTION)  # what (f) needs
        cmds = re.findall(r'--instruction "([^"]*)"', self.readme)
        self.assertEqual(len(cmds), 2)
        self.assertTrue(all(c == self.task.INSTRUCTION for c in cmds))

    def test_texts_have_no_dashes_or_identity(self):
        for text in (*[v for v in self.spec.values() if isinstance(v, str)], self.readme):
            self.assertNotIn("\u2014", text)
            self.assertNotIn("\u2013", text)
            self.assertNotRegex(text.lower(), r"t-70|vihan|aggarwal|claude|anthropic")
        for tool in ("check_task", "tools/", "board_check", "field_limits", "pipeline/"):  # no internal tooling
            self.assertNotIn(tool, self.readme)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(PARKED)
        for _ in range(11):
            self.task.success(NOMINAL)
        self.assertTrue(self.task.success(NOMINAL))
        self.assertEqual(self.task.score(NOMINAL), 1.0)
        self.assertEqual(self.task.score(NOMINAL), 1.0)  # score() does not feed the tracker
        self.feed(holding_a(NOMINAL))
        self.assertFalse(self.task.success(NOMINAL))
        self.assertEqual(self.task.score(NOMINAL), 0.5)
        rigstate.offline_reset()  # rigstate's reset hook: the tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(START))  # arms None after a reset
        self.assertEqual(tr.arms_unavailable, 1)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed(READY)
            self.assertFalse(self.task.success(START))
            self.assertFalse(self.task.success({}))  # a broken state is scored 0, not raised
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_timing_and_targets(self):
        T = self.task
        plan = T.demo()
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if w.finger != prev.finger and w.pos == prev.pos:  # every close and release: >= 1.5 s after a settle
                    self.assertGreaterEqual(w.duration, 1.5, msg=(side, w.label))
                    self.assertEqual(prev.label, "settle", msg=(side, w.label))
                    self.assertGreaterEqual(prev.duration, 0.5)
                if "slowly" in w.label:  # the last 30 mm of every descent over at least 1.5 s
                    self.assertAlmostEqual(prev.pos[2] - w.pos[2], T.ENTRY)
                    self.assertGreaterEqual(w.duration, 1.5)
            travel = {round(w.pos[2] - L.TOP, 4) for w in wps if w.label.startswith(("over", "carry", "lift", "park"))}
            self.assertTrue(all(0.10 <= z <= 0.14 for z in travel), msg=travel)
            self.assertEqual(len({w.yaw for w in wps}), 1)
        self.assertEqual(plan["left"][0].yaw, -math.pi / 2)
        self.assertEqual(plan["right"][0].yaw, math.pi / 2)
        self.assertAlmostEqual(T.RISER_GRIP, -0.026)
        self.assertAlmostEqual(T.POT_GRIP, -0.031)
        # each arm takes the riser that stages its pot, and lets it go 5 mm above the disc ...
        self.assertEqual((T.PLAN["left"]["riser"], T.PLAN["right"]["riser"]), ("riser_l30", "riser_r20"))
        self.assertAlmostEqual(T.RISER_PLACE_Z - L.TOP, 0.030)
        # ... and lets its pot go 3 mm above the riser's top (TCP 60 mm for pot A, 50 mm for pot B)
        self.assertAlmostEqual(T.pot_place_z("left") - L.TOP, 0.060)
        self.assertAlmostEqual(T.pot_place_z("right") - L.TOP, 0.050)
        # carried parts clear the 86 mm rim: riser bottom 111 mm, pot bottom 119 mm up
        self.assertAlmostEqual(T.RISER_CARRY_Z - T.RISER_PICK_Z, 0.111)
        self.assertAlmostEqual(T.POT_CARRY_Z - T.POT_PICK_Z, 0.119)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertAlmostEqual(total, 49.3)
        self.assertGreaterEqual(3600 / (total * 20), 3.0)  # the run limit is at least 3 times the demo
        # the right arm waits 10 s over its riser first: it enters the planter only once the left arm has left it
        self.assertEqual(plan["right"][1].label, "wait for the left arm to work in the planter")
        self.assertAlmostEqual(plan["right"][1].duration, T.RIGHT_LAG)
        self.assertEqual(len(plan["left"]) + 1, len(plan["right"]))

        def visits(wps):  # (start, end) of each stay over its disc: from arriving over it to leaving the planter
            t, out, start = 0.0, [], None
            for w in wps:
                t += w.duration
                over = math.hypot(w.pos[0] - L.DISC_X, abs(w.pos[1]) - L.DISC_Y) < 1e-9
                if over and start is None:
                    start = t
                if start is not None and w.label in ("up out of the planter", "straight up"):
                    out.append((start, t))
                    start = None
            return out

        left, right = visits(plan["left"]), visits(plan["right"])
        self.assertEqual(len(left), 2)
        for a0, a1 in left:
            for b0, b1 in right:
                self.assertTrue(b0 >= a1 or b1 <= a0, msg=((a0, a1), (b0, b1)))  # never in the planter together


if __name__ == "__main__":
    unittest.main()
