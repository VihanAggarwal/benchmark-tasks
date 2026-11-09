"""Offline tests for scoring.py (and the hooks and demo in task.py): synthetic part states, no simulator.
    python3 -m unittest discover -s tests -v"""

import importlib
import itertools
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
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

R1, C1, T1 = L.RESISTOR, L.CAPACITOR, L.TANTALUM
WINDOW = 11  # 0.5 s at 20 Hz plus the step at hand
MM = 0.001


# -- poses ------------------------------------------------------------------------------------------------------------
def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def quat_xyzw(r):
    """Unit quaternion (x, y, z, w) of a rotation matrix."""
    m = np.asarray(r, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return np.array(q)


def part_state(part, x, y, low_z, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, vel=(0.0, 0.0, 0.0)):
    """A part's state as mechkit reports it: the middle of its box at (x, y) seen from above, its lowest corner at
    world height low_z, its local x turned yaw_deg from world x (after a pitch about local y and a roll about local
    x)."""
    r = rz(math.radians(yaw_deg)) @ ry(math.radians(pitch_deg)) @ rx(math.radians(roll_deg))
    corners = S.box_corners(part) @ r.T
    mid = r @ np.array([0.0, 0.0, part.height / 2])
    origin = np.array([x - mid[0], y - mid[1], low_z - corners[:, 2].min()])
    world = corners + origin
    lo, hi = world.min(axis=0), world.max(axis=0)
    return SimpleNamespace(origin=origin, quat_xyzw=quat_xyzw(r), pos=(lo + hi) / 2, low=lo, high=hi,
                           lin_vel=np.array(vel, dtype=float), ang_vel=np.zeros(3), source="test")


def placed(part, along=0.0, across=0.0, gap=0.0, yaw_err=0.0, reverse=False, **kw):
    """On its own footprint: centre offset (along x, across y) from the footprint's centre, lowest point gap above
    the pads' top, turned yaw_err degrees from square; T1 band toward +x unless reverse."""
    fp = part.footprint
    base = 180.0 if reverse else 0.0
    return part_state(part, fp.x + along, fp.y + across, L.PAD_TOP_W + gap, base + yaw_err, **kw)


def on_fp(part, fp, along=0.0, across=0.0, yaw_deg=0.0):
    """On another footprint's pads (centred on it)."""
    return part_state(part, fp.x + along, fp.y + across, L.PAD_TOP_W, yaw_deg)


def at_start(part, dx=0.0, dy=0.0):
    """On its mat, long side along y (T1's band toward +y)."""
    return part_state(part, part.start[0] + dx, part.start[1] + dy, L.TOP + L.MAT_TOP, 90.0)


def scene(**over):
    """All three parts placed centred and square (T1 band on +), with any replaced."""
    out = {"resistor": placed(R1), "capacitor": placed(C1), "tantalum": placed(T1)}
    out.update(over)
    return out


def start_scene(**over):
    out = {p.name: at_start(p) for p in L.PARTS}
    out.update(over)
    return out


def settled(objects, n=WINDOW, tr=None):
    tr = tr or Tracker()
    res = None
    for _ in range(n):
        res = tr.update(objects)
    return tr, res


def sixths(objects):
    return settled(objects)[1].details["sixths"]


def credit(objects, name):
    return settled(objects)[1].details[name]["credit_sixths"]


def lim(part, frac):
    return L.lateral_limit(part, frac)


# -- geometry ---------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_limits_follow_ipc(self):
        for part, c3, c2 in ((R1, 8.25, 14.5), (C1, 7.0, 12.0), (T1, 10.5, 19.0)):
            self.assertAlmostEqual(lim(part, L.CLASS3_FRAC) / MM, c3, places=9)
            self.assertAlmostEqual(lim(part, L.CLASS2_FRAC) / MM, c2, places=9)
            self.assertAlmostEqual(L.axial_play(part) / MM, 6.0, places=9)

    def test_each_footprint_fits_only_its_part(self):
        """Pads 2 mm wider than the part per side and reaching 6 mm past its ends: true for its own part only."""
        for part in L.PARTS:
            for fp in (L.R_FP, L.C_FP, L.T_FP):
                fits = (abs((fp.pad_width - part.width) / 2 - 0.002) < 1e-9
                        and abs(fp.outer - part.length / 2 - 0.006) < 1e-9)
                self.assertEqual(fits, fp is part.footprint, (part.name, fp.key))

    def test_terminations_lie_on_the_pads_when_centred(self):
        for part in L.PARTS:
            fp = part.footprint
            inner_end = part.length / 2 - part.term
            self.assertGreaterEqual(inner_end, fp.inner - 1e-12)  # the termination starts on the pad
            self.assertLessEqual(part.length / 2, fp.outer)

    def test_footprints_on_the_board_and_apart(self):
        for fp in (L.R_FP, L.C_FP, L.T_FP):
            self.assertLessEqual(abs(fp.x - L.BOARD_X) + fp.outer, L.BOARD_SIZE[0] / 2)
            self.assertLessEqual(abs(fp.y - L.BOARD_Y) + fp.pad_width / 2, L.BOARD_SIZE[1] / 2)
        # a part at its Class 2 limit toward a neighbour stays more than 15 mm clear of the neighbour's pads
        for a, b in ((R1, C1), (C1, T1), (C1, R1), (T1, C1)):
            gap = abs(a.footprint.y - b.footprint.y) - lim(a, 0.5) - a.width / 2 - b.footprint.pad_width / 2
            self.assertGreater(gap, 0.015, (a.name, b.name))
        # the + mark on the board, beside the + pad
        self.assertLess(abs(L.PLUS_MARK_Y) + L.PLUS_MARK_SIZE[1] / 2, L.BOARD_SIZE[1] / 2)
        self.assertAlmostEqual(L.PLUS_MARK_X, L.T_FP.x + L.PLUS_PAD_SIGN * L.T_FP.pad_offset)

    def test_side_and_end_overhang_measures(self):
        st = placed(R1, across=0.010)
        self.assertAlmostEqual(S.side_overhangs(R1, st)[0], 0.008, places=9)  # 12.5 + 10 - 14.5
        self.assertAlmostEqual(S.side_overhangs(R1, st)[1], 0.008, places=9)
        self.assertEqual(S.side_overhangs(R1, placed(R1, across=0.0015)), (0.0, 0.0))
        self.assertAlmostEqual(S.end_overhang(R1, placed(R1, along=0.004)), -0.002, places=9)
        self.assertAlmostEqual(S.end_overhang(R1, placed(R1, along=-0.008)), 0.002, places=9)
        # turned: the corners of each termination are what overhang (the worse one)
        st = placed(R1, yaw_err=8.0)
        want = 0.025 * math.sin(math.radians(8)) + 0.0125 * math.cos(math.radians(8)) - 0.0145
        self.assertAlmostEqual(max(S.side_overhangs(R1, st)), want, places=9)
        # side overhang is read at the terminations only: the body's middle does not count
        st = placed(R1, yaw_err=8.0)
        body = S.in_footprint(R1, S.to_world(st, S.box_corners(R1)))
        self.assertAlmostEqual(np.abs(body[:, 1]).max() - 0.0145, want, places=9)

    def test_angles(self):
        self.assertAlmostEqual(S.tilt_deg(R1, placed(R1, roll_deg=180.0)), 0.0, places=6)  # either face up
        self.assertAlmostEqual(S.tilt_deg(T1, placed(T1, roll_deg=180.0)), 180.0, places=6)  # band face down
        self.assertAlmostEqual(S.tilt_deg(C1, placed(C1, roll_deg=4.0)), 4.0, places=6)
        self.assertAlmostEqual(S.axis_err_deg(placed(R1, yaw_err=-7.0)), 7.0, places=6)
        self.assertAlmostEqual(S.axis_err_deg(placed(R1, reverse=True, yaw_err=3.0)), 3.0, places=6)
        self.assertAlmostEqual(S.axis_err_deg(at_start(R1)), 90.0, places=6)
        self.assertEqual(S.axis_err_deg(placed(R1, pitch_deg=90.0)), 90.0)  # on an end
        self.assertEqual(S.plus_end_sign(placed(T1)), 1.0)
        self.assertEqual(S.plus_end_sign(placed(T1, reverse=True)), -1.0)

    def test_heights(self):
        self.assertAlmostEqual(S.lowest_gap(T1, placed(T1)), 0.0, places=9)
        self.assertAlmostEqual(S.lowest_gap(R1, placed(R1, gap=0.0015)), 0.0015, places=9)
        self.assertFalse(S.off_table(R1, at_start(R1)))
        self.assertFalse(S.off_table(R1, part_state(R1, 0.3, 0.7, L.TOP - 0.049)))
        self.assertTrue(S.off_table(R1, part_state(R1, 0.3, 0.7, L.TOP - 0.051)))

    def test_start_layout(self):
        st = start_scene()
        self.assertTrue(Tracker.at_start(st))
        for p in L.PARTS:
            self.assertEqual(S.assess(p, st[p.name])[0], 0)
        _, res = settled(st)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)
        self.assertEqual(res.details["zero_reasons"], [])


# -- tiers -------------------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_all_class_3_scores_one(self):
        tr, res = settled(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        for p in L.PARTS:
            self.assertEqual(res.details[p.name]["tier"], "class3")

    def test_class_3_edge_per_part(self):
        for part in L.PARTS:
            for sign in (1, -1):
                c3, c2 = lim(part, 0.25), lim(part, 0.5)
                self.assertEqual(credit(scene(**{part.name: placed(part, across=sign * (c3 - 0.0002))}),
                                        part.name), 2, part.name)
                self.assertEqual(credit(scene(**{part.name: placed(part, across=sign * (c3 + 0.0002))}),
                                        part.name), 1, part.name)
                self.assertEqual(credit(scene(**{part.name: placed(part, across=sign * (c2 - 0.0002))}),
                                        part.name), 1, part.name)
                self.assertEqual(credit(scene(**{part.name: placed(part, across=sign * (c2 + 0.0002))}),
                                        part.name), 0, part.name)

    def test_class_2_and_none_scores(self):
        _, res = settled(scene(resistor=placed(R1, across=0.012)))
        self.assertAlmostEqual(res.score, 5 / 6)
        self.assertFalse(res.success)
        self.assertEqual(res.details["resistor"]["tier"], "class2")
        _, res = settled(scene(resistor=placed(R1, across=0.016)))
        self.assertAlmostEqual(res.score, 4 / 6)
        self.assertEqual(res.details["resistor"]["zero_reason"], "side_overhang_over_half_its_width")

    def test_every_total_is_reachable(self):
        """0, 1/6, ..., 1: each part at Class 3 (2), Class 2 (1) or neither (0), every combination."""
        poses = {}
        for p in L.PARTS:
            poses[p.name] = {2: placed(p), 1: placed(p, across=(lim(p, 0.25) + lim(p, 0.5)) / 2),
                             0: placed(p, across=lim(p, 0.5) + 0.003)}
        seen = set()
        for combo in itertools.product((0, 1, 2), repeat=3):
            objs = {p.name: poses[p.name][c] for p, c in zip(L.PARTS, combo)}
            _, res = settled(objs)
            self.assertEqual(res.details["sixths"], sum(combo), combo)
            self.assertAlmostEqual(res.score, sum(combo) / 6)
            self.assertEqual(res.success, sum(combo) == 6)
            seen.add(res.details["sixths"])
        self.assertEqual(seen, set(range(7)))

    def test_along_the_axis_6_mm_of_play(self):
        for part in L.PARTS:
            for sign in (1, -1):
                self.assertEqual(credit(scene(**{part.name: placed(part, along=sign * 0.0059)}), part.name), 2)
                objs = scene(**{part.name: placed(part, along=sign * 0.0061)})
                self.assertEqual(credit(objs, part.name), 0)
                self.assertEqual(settled(objs)[1].details[part.name]["zero_reason"], "end_overhang")

    def test_end_overhang_is_zero_even_when_side_is_class_3(self):
        st = placed(R1, along=0.0062, across=0.002)
        self.assertEqual(S.assess(R1, st)[0], 0)

    def test_turn_tolerance_is_10_degrees(self):
        # C1 is short: at 9.9 degrees its termination corners are still inside Class 3
        self.assertEqual(credit(scene(capacitor=placed(C1, yaw_err=9.9)), "capacitor"), 2)
        self.assertEqual(credit(scene(capacitor=placed(C1, yaw_err=10.1)), "capacitor"), 0)
        self.assertEqual(credit(scene(capacitor=placed(C1, yaw_err=-10.1)), "capacitor"), 0)
        # turned and off to the side, the corners decide the class: 5 mm off is Class 3 square, Class 2 at 8 deg
        self.assertEqual(credit(scene(resistor=placed(R1, across=0.005)), "resistor"), 2)
        self.assertEqual(credit(scene(resistor=placed(R1, yaw_err=8.0, across=0.005)), "resistor"), 1)
        # turned the other way the - end's termination is the worse one, and it decides
        st = placed(R1, yaw_err=-8.0, across=0.005)
        plus, minus = S.side_overhangs(R1, st)
        self.assertLess(plus, 0.25 * R1.width - 0.005)  # the + end: only its inner corner, 0.5 mm
        self.assertGreater(minus, 0.25 * R1.width)
        self.assertEqual(credit(scene(resistor=st), "resistor"), 1)

    def test_tilt_tolerance_is_5_degrees(self):
        for part in L.PARTS:
            ok = placed(part, roll_deg=4.9)
            bad = placed(part, roll_deg=5.1)
            self.assertEqual(credit(scene(**{part.name: ok}), part.name), 2)
            self.assertEqual(credit(scene(**{part.name: bad}), part.name), 0)

    def test_height_tolerance_is_2_mm(self):
        self.assertEqual(credit(scene(tantalum=placed(T1, gap=0.0019)), "tantalum"), 2)
        self.assertEqual(credit(scene(tantalum=placed(T1, gap=0.0021)), "tantalum"), 0)
        self.assertEqual(credit(scene(resistor=placed(R1, gap=-0.0019)), "resistor"), 2)

    def test_resistor_and_ceramic_either_face_up_and_either_way_round(self):
        for part in (R1, C1):
            for st in (placed(part, roll_deg=180.0), placed(part, reverse=True),
                       placed(part, reverse=True, roll_deg=180.0)):
                self.assertEqual(credit(scene(**{part.name: st}), part.name), 2)

    def test_score_is_idempotent(self):
        tr, res = settled(scene())
        a = tr.result(scene()).score
        b = tr.result(scene()).score
        self.assertEqual((a, b, tr.steps), (1.0, 1.0, WINDOW))


# -- whole-trial zeros and per-part zeros -------------------------------------------------------------------------------
class ZeroConditions(unittest.TestCase):
    def test_reversed_tantalum_zeroes_the_trial(self):
        _, res = settled(scene(tantalum=placed(T1, reverse=True)))
        self.assertEqual(res.score, 0.0)
        self.assertIn("tantalum_reversed_on_its_pads", res.details["zero_reasons"])
        self.assertEqual(res.details["tantalum"]["zero_reason"], "band_on_the_minus_pad")
        # anywhere over its footprint, within 10 degrees, still counts as reversed
        for kw in ({"across": 0.018}, {"along": 0.030}, {"yaw_err": 9.0}, {"gap": 0.0018}):
            self.assertEqual(settled(scene(tantalum=placed(T1, reverse=True, **kw)))[1].score, 0.0, kw)

    def test_reversed_tantalum_needs_it_on_its_pads(self):
        """Band toward -x but not lying on its own pads: only T1 earns nothing; the other two still count."""
        for st in (placed(T1, reverse=True, across=0.0195), placed(T1, reverse=True, along=0.036),
                   placed(T1, reverse=True, yaw_err=11.0), placed(T1, reverse=True, gap=0.003),
                   placed(T1, reverse=True, roll_deg=180.0)):
            _, res = settled(scene(tantalum=st))
            self.assertAlmostEqual(res.score, 4 / 6)
            self.assertEqual(res.details["zero_reasons"], [])

    def test_reversed_counts_even_while_moving(self):
        objs = scene(tantalum=placed(T1, reverse=True, vel=(0.1, 0, 0)))
        self.assertEqual(Tracker().update(objs).score, 0.0)
        self.assertIn("tantalum_reversed_on_its_pads", Tracker().update(objs).details["zero_reasons"])

    def test_upside_down_tantalum_scores_zero_for_itself(self):
        _, res = settled(scene(tantalum=placed(T1, roll_deg=180.0)))
        self.assertAlmostEqual(res.score, 4 / 6)
        self.assertEqual(res.details["tantalum"]["zero_reason"], "upside_down")

    def test_any_part_off_the_table_zeroes_the_trial(self):
        for part in L.PARTS:
            gone = part_state(part, 0.30, 0.70, L.TOP - 0.30)
            _, res = settled(scene(**{part.name: gone}))
            self.assertEqual(res.score, 0.0, part.name)
            self.assertEqual(res.details["zero_reasons"], [f"{part.name}_off_the_table"])
        # on the floor beside the table but only 4.9 cm down (a ledge): not off the table
        _, res = settled(scene(resistor=part_state(R1, 0.30, 0.70, L.TOP - 0.049)))
        self.assertAlmostEqual(res.score, 4 / 6)

    def test_left_on_the_mat_or_the_table(self):
        self.assertAlmostEqual(settled(scene(capacitor=at_start(C1)))[1].score, 4 / 6)
        self.assertAlmostEqual(settled(scene(capacitor=part_state(C1, 0.40, 0.20, L.TOP)))[1].score, 4 / 6)

    def test_on_the_board_beside_its_pads(self):
        st = part_state(R1, R1.footprint.x, R1.footprint.y + 0.040, L.TOP + L.BOARD_TOP)
        self.assertEqual(S.assess(R1, st)[0], 0)

    def test_on_its_side_or_on_an_end(self):
        for st in (placed(R1, roll_deg=90.0), placed(R1, pitch_deg=90.0), placed(T1, roll_deg=90.0)):
            name = "tantalum" if st.origin[1] < -0.03 else "resistor"
            self.assertEqual(credit(scene(**{name: st}), name), 0)
        for st in (placed(C1, pitch_deg=90.0), placed(C1, pitch_deg=-90.0), placed(C1, roll_deg=45.0)):
            self.assertEqual(credit(scene(capacitor=st), "capacitor"), 0)  # on an end, or on an edge

    def test_ceramic_counts_on_any_long_face(self):
        """C1 is square in section (20 x 20 mm) with its end caps round all four long faces: rolled onto any of
        them it is the same part, shadow and height alike, so it scores the same; R1 (25 x 20) and T1 do not."""
        self.assertTrue(C1.square_section)
        self.assertFalse(R1.square_section)
        self.assertFalse(T1.square_section)
        for roll in (90.0, -90.0, 270.0):
            for kw in ({}, {"reverse": True}, {"across": 0.0069}, {"across": -0.0071}, {"along": 0.0059}):
                st = placed(C1, roll_deg=roll, **kw)
                ref = placed(C1, **kw)
                self.assertAlmostEqual(S.tilt_deg(C1, st), 0.0, places=6)
                self.assertEqual(S.assess(C1, st)[0], S.assess(C1, ref)[0])
                self.assertEqual(credit(scene(capacitor=st), "capacitor"), credit(scene(capacitor=ref), "capacitor"))
        # the same 5 degree tolerance about the nearest long face
        self.assertEqual(credit(scene(capacitor=placed(C1, roll_deg=85.1)), "capacitor"), 2)
        self.assertEqual(credit(scene(capacitor=placed(C1, roll_deg=84.9)), "capacitor"), 0)
        self.assertEqual(credit(scene(capacitor=placed(C1, roll_deg=94.9)), "capacitor"), 2)
        self.assertEqual(credit(scene(capacitor=placed(C1, roll_deg=95.1)), "capacitor"), 0)
        self.assertEqual(credit(scene(capacitor=placed(C1, roll_deg=90.0, pitch_deg=5.1)), "capacitor"), 0)
        self.assertAlmostEqual(S.tilt_deg(C1, placed(C1, pitch_deg=90.0)), 90.0, places=6)
        # a resistor on its side stays 0 (its shadow is 20 mm wide, not 25); the full score needs nothing else
        self.assertEqual(credit(scene(resistor=placed(R1, roll_deg=90.0)), "resistor"), 0)
        self.assertEqual(settled(scene(capacitor=placed(C1, roll_deg=90.0)))[1].score, 1.0)

    def test_across_its_footprint(self):
        for part in L.PARTS:
            self.assertEqual(credit(scene(**{part.name: placed(part, yaw_err=90.0)}), part.name), 0)


# -- at rest -----------------------------------------------------------------------------------------------------------
class Rest(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr, res = settled(scene(), n=WINDOW - 1)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["resistor"]["zero_reason"], "still_moving")
        self.assertIsNone(res.details["resistor"]["moved_mm"])
        self.assertEqual(tr.update(scene()).score, 1.0)

    def test_moving_into_place_scores_zero_until_at_rest(self):
        tr = Tracker()
        objs = None
        for i in range(20):  # T1 slides 1 mm a step onto its centre
            objs = scene(tantalum=placed(T1, across=(19 - i) * MM))
            res = tr.update(objs)
        self.assertEqual(res.details["tantalum"]["credit_sixths"], 0)
        self.assertAlmostEqual(res.score, 4 / 6)
        for _ in range(WINDOW - 1):
            res = tr.update(objs)
        self.assertEqual(res.score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker()
        for i in range(30):
            res = tr.update(scene(resistor=placed(R1, across=0.0009 * (-1) ** i)))
        self.assertEqual(res.score, 1.0)

    def test_a_slow_creep_over_2_mm_in_half_a_second_is_moving(self):
        tr = Tracker()
        for i in range(30):
            res = tr.update(scene(capacitor=placed(C1, along=-0.003 + 0.00025 * i)))
        self.assertEqual(res.details["capacitor"]["zero_reason"], "still_moving")

    def test_knocked_and_back_within_half_a_second_is_moving(self):
        tr, _ = settled(scene())
        tr.update(scene(capacitor=placed(C1, across=0.005)))  # bumped 5 mm, back on the next step
        for _ in range(5):  # the bump in the middle of the half-second window
            res = tr.update(scene())
        self.assertFalse(res.details["capacitor"]["at_rest"])
        for _ in range(WINDOW - 6):
            res = tr.update(scene())
        self.assertFalse(res.details["capacitor"]["at_rest"])
        self.assertAlmostEqual(res.score, 4 / 6)
        res = tr.update(scene())  # 0.5 s after the bump
        self.assertEqual(res.score, 1.0)

    def test_a_turn_in_place_is_moving(self):
        tr = Tracker()
        for i in range(30):  # 0.5 deg a step about its middle: its ends move 2.2 mm in 0.5 s
            res = tr.update(scene(resistor=placed(R1, yaw_err=-7.5 + 0.5 * i)))
        self.assertFalse(res.details["resistor"]["at_rest"])

    def test_partial_tiers_need_rest_too(self):
        tr = Tracker()
        for i in range(20):  # 0.25 mm a step: 2.5 mm in 0.5 s, inside the Class 2 band throughout
            res = tr.update(scene(resistor=placed(R1, across=0.0085 + 0.00025 * i)))
        self.assertEqual(res.details["resistor"]["credit_sixths"], 0)
        self.assertEqual(res.details["resistor"]["tier"], "class2")


# -- exploits ----------------------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_parts_on_each_others_pads(self):
        """R1 fits T1's pads with no overhang, and T1 fits R1's with 2.5 mm a side: neither earns anything."""
        objs = {"resistor": on_fp(R1, L.T_FP), "tantalum": on_fp(T1, L.R_FP), "capacitor": placed(C1)}
        _, res = settled(objs)
        self.assertAlmostEqual(res.score, 2 / 6)
        self.assertEqual(res.details["zero_reasons"], [])
        objs = {"resistor": on_fp(R1, L.C_FP), "capacitor": on_fp(C1, L.R_FP), "tantalum": placed(T1)}
        self.assertAlmostEqual(settled(objs)[1].score, 2 / 6)
        # T1 band-first on R1's pads is not 'reversed on its own pads'
        objs = {"resistor": at_start(R1), "capacitor": at_start(C1), "tantalum": on_fp(T1, L.R_FP, yaw_deg=180.0)}
        self.assertEqual(settled(objs)[1].details["zero_reasons"], [])

    def test_stacked_on_another_part(self):
        top = part_state(C1, R1.footprint.x, R1.footprint.y, L.PAD_TOP_W + R1.height)
        objs = scene(capacitor=top)
        _, res = settled(objs)
        self.assertAlmostEqual(res.score, 4 / 6)
        self.assertEqual(res.details["capacitor"]["zero_reason"], "not_on_the_pads_height")
        # T1 lying on R1 and C1 over its own pads would be 20 mm up: nothing
        self.assertEqual(S.assess(T1, placed(T1, gap=0.020))[0], 0)

    def test_held_above_the_pads(self):
        for part in L.PARTS:
            self.assertEqual(credit(scene(**{part.name: placed(part, gap=0.004)}), part.name), 0)

    def test_leaning_on_a_neighbour(self):
        self.assertEqual(S.assess(R1, placed(R1, roll_deg=12.0, across=0.006))[0], 0)

    def test_reversed_tantalum_dragged_off_to_dodge_the_zero(self):
        """Moving a reversed T1 just off its footprint avoids the trial zero but earns T1 nothing: never more than
        what the other two parts earn."""
        st = placed(T1, reverse=True, across=-(T1.footprint.pad_width / 2 + 0.0005))
        _, res = settled(scene(tantalum=st))
        self.assertAlmostEqual(res.score, 4 / 6)

    def test_pushed_along_the_board_is_scored_by_where_it_ends(self):
        tr = Tracker()
        for i in range(45):  # pushed 2 mm a step from 60 mm off to the side onto its pads, then left
            off = max(0.0, 0.060 - 0.002 * i)
            res = tr.update(scene(resistor=placed(R1, across=off, gap=-0.001 if off > 0.0145 + 0.0125 else 0.0)))
        self.assertEqual(res.score, 1.0)

    def test_no_pose_beats_the_rules(self):
        """A coarse sweep of R1's poses around its footprint: every pose that earns credit meets the rules."""
        for along, across, yaw, tilt, gap in itertools.product((-0.007, -0.003, 0.0, 0.005), (0.0, 0.007, 0.0148),
                                                              (0.0, 6.0, 12.0), (0.0, 6.0), (0.0, 0.003)):
            st = placed(R1, along=along, across=across, yaw_err=yaw, roll_deg=tilt, gap=gap)
            c = S.assess(R1, st)[0]
            if c:
                self.assertLessEqual(S.end_overhang(R1, st), 0.0)
                self.assertLessEqual(max(S.side_overhangs(R1, st)), 0.5 * R1.width + 1e-12)
                self.assertLessEqual(S.tilt_deg(R1, st), 5.0)
                self.assertLessEqual(S.axis_err_deg(st), 10.0)
                self.assertLessEqual(abs(S.lowest_gap(R1, st)), 0.002)


# -- robustness --------------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def bad_states(self):
        good = placed(R1)
        out = []
        for key, val in (("origin", np.array([np.nan, 0.0, 0.0])), ("origin", np.array([np.inf, 0.0, 0.0])),
                         ("quat_xyzw", np.zeros(4)), ("quat_xyzw", np.array([np.nan, 0, 0, 1.0])),
                         ("pos", np.array([0.0, np.nan, 0.0])), ("origin", np.zeros(2))):
            st = SimpleNamespace(**vars(good))
            setattr(st, key, val)
            out.append(st)
        no_origin = SimpleNamespace(**{k: v for k, v in vars(good).items() if k != "origin"})
        out.append(no_origin)
        out.append(None)
        return out

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, res = settled(scene())
        self.assertEqual(res.score, 1.0)
        hist = {k: list(v) for k, v in tr.history.items()}
        for bad in self.bad_states():
            r = tr.update(scene(resistor=bad))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
            self.assertEqual(tr.result(scene(resistor=bad)).score, 0.0)
        for k, v in tr.history.items():
            self.assertEqual(len(v), len(hist[k]))
            for a, b in zip(v, hist[k]):
                np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(tr.update(scene()).score, 1.0)  # and it carries on

    def test_missing_or_malformed_objects(self):
        tr = Tracker()
        for objs in ({}, {"resistor": placed(R1)}, None, [], "objects", {"resistor": 1, "capacitor": 2,
                                                                        "tantalum": 3}):
            self.assertEqual(tr.update(objs).score, 0.0)
        self.assertEqual(sum(len(v) for v in tr.history.values()), 0)

    def test_unnormalised_quaternion(self):
        objs = scene()
        for st in objs.values():
            st.quat_xyzw = st.quat_xyzw * 3.0
        self.assertEqual(settled(objs)[1].score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker()
        for _ in range(300):
            objs = {}
            for p in L.PARTS:
                st = part_state(p, rng.uniform(0.1, 0.5), rng.uniform(-0.4, 0.4), L.TOP + rng.uniform(-0.1, 0.1),
                                rng.uniform(-180, 180), rng.uniform(-180, 180), rng.uniform(-180, 180))
                objs[p.name] = st
            res = tr.update(objs)
            self.assertIn(res.details["sixths"], range(7))
            self.assertTrue(0.0 <= res.score <= 1.0)
            json.dumps(res.details)


# -- reset -------------------------------------------------------------------------------------------------------------
class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(scene())
        tr.update(start_scene())
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.history["resistor"]), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(scene())
        tr.update(start_scene(resistor=at_start(R1, 0.02, -0.02), tantalum=at_start(T1, -0.03, 0.01)))
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker()
        objs = start_scene(resistor=at_start(R1, 0.0, -0.06))
        for _ in range(5):
            tr.update(objs)
        for i in range(1, 13):  # carried back 5 mm a step onto its mark
            tr.update(start_scene(resistor=at_start(R1, 0.0, -0.06 + 0.005 * i)))
        self.assertEqual(tr.steps, 17)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(scene())
        tr.update(scene(resistor=placed(R1, across=-0.10)))
        self.assertEqual(tr.steps, WINDOW + 1)

    def test_explicit_reset(self):
        tr, _ = settled(scene())
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(sum(len(v) for v in tr.history.values()), 0)
        self.assertIsNone(tr.prev_origin)
        self.assertEqual(tr.result(scene()).score, 0.0)  # no history yet: not at rest

    def test_reset_after_a_full_trial_then_a_new_trial(self):
        tr, _ = settled(scene())
        tr.update(start_scene())
        _, res = settled(scene(tantalum=placed(T1, across=0.012)), tr=tr)
        self.assertAlmostEqual(res.score, 5 / 6)


# -- task.py against the offline yam_arena stub -------------------------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit
        cls.specs = {"resistor": cls.task.RESISTOR, "capacitor": cls.task.CAPACITOR, "tantalum": cls.task.TANTALUM}

    def sim(self, objs):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*st.origin, *st.quat_xyzw), source="sim")
                for n, st in objs.items()}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), ["capacitor", "resistor", "tantalum"])
        for o in loose:
            self.assertLessEqual(o.mass, 0.30)
            self.assertAlmostEqual(o.mass, L.BY_NAME[o.name].mass)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 2.5)
        pads = {o.name: o for o in t.OBJECTS if "pad" in o.name}
        self.assertEqual(len(pads), 6)
        for fp, near, far in ((L.R_FP, "r_pad_near", "r_pad_far"), (L.C_FP, "c_pad_near", "c_pad_far"),
                              (L.T_FP, "t_pad_minus", "t_pad_plus")):
            self.assertAlmostEqual(pads[far].pos[0] - pads[near].pos[0], 2 * fp.pad_offset)
            self.assertAlmostEqual(pads[far].pos[0], fp.x + fp.pad_offset)
            self.assertAlmostEqual(pads[near].pos[1], fp.y)
            self.assertEqual(tuple(pads[near].size), (fp.pad_len, fp.pad_width, L.PAD_T))
            self.assertAlmostEqual(pads[near].pos[2] + L.PAD_T / 2, L.PAD_TOP_W)
        # the + pad is the far one, and the + mark is beside it
        self.assertAlmostEqual(pads["t_pad_plus"].pos[0], t.PLUS_MARK.pos[0])
        self.assertGreater(pads["t_pad_plus"].pos[0], L.T_FP.x)
        # the mats are clear of the arms' start-pose shadow in the setup screenshot (|y| > 0.45 with x < 0.20)
        for m in t.MATS:
            self.assertLess(abs(m.pos[1]) + m.size[1] / 2, 0.45)

    def test_scorer_geometry_matches_mechkit(self):
        for name, spec in self.specs.items():
            p = L.BY_NAME[name]
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-p.length / 2, -p.width / 2, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [p.length / 2, p.width / 2, p.height], atol=1e-12)
            com = spec.center_of_mass()
            self.assertLess(abs(com[0]) + abs(com[1]), 0.0006)  # centred: no end is heavier (the band is 1 mm)
            st = self.sim(scene())[name]
            self.assertAlmostEqual(S.lowest_gap(p, st), 0.0, places=9)
            np.testing.assert_allclose(st.low[2], L.PAD_TOP_W, atol=1e-9)
            self.assertTrue(Tracker.at_start({n: s.rest_state() for n, s in self.specs.items()}))
        # the band is at the tantalum's local +x end, on its top; the plates are under its ends
        shapes = self.task.TANTALUM.shapes
        band = [s for s in shapes if tuple(s.color or ()) == L.TA_BAND][0]
        self.assertGreater(band.center[0], 0.0)
        self.assertAlmostEqual(band.center[2] + band.size[2] / 2, T1.height)
        plates = [s for s in shapes if abs(s.size[2] - L.TA_PLATE_T) < 1e-12 and s is not band]
        self.assertEqual(sorted(round(s.center[0], 6) for s in plates), [-0.025, 0.025])
        for s in plates:
            self.assertAlmostEqual(s.center[2] - s.size[2] / 2, 0.0)
        # the resistor's and ceramic's end caps are the scorer's terminations
        for spec, p in ((self.task.RESISTOR, R1), (self.task.CAPACITOR, C1)):
            caps = sorted(s.center[0] for s in spec.shapes if abs(s.size[0] - p.term) < 1e-12)
            self.assertEqual(len(caps), 2)
            self.assertAlmostEqual(caps[1], p.length / 2 - p.term / 2)

    def test_every_part_bridges_its_pads_when_placed(self):
        """Centred and anywhere in the scored play, each part's underside lies on both of its pads (it rests on
        them, not on the board between)."""
        for p in L.PARTS:
            fp = p.footprint
            for along in (-L.axial_play(p), 0.0, L.axial_play(p)):
                ends = (along - p.length / 2, along + p.length / 2)
                self.assertLess(ends[0], -fp.inner)
                self.assertGreater(ends[1], fp.inner)

    def test_offline_state_path(self):
        for name, spec in self.specs.items():
            centre, size, yaw = spec.offline_box()
            box = SimpleNamespace(pos=np.array(centre), quat_xyzw=quat_xyzw(rz(yaw)), lin_vel=np.zeros(3),
                                  ang_vel=np.zeros(3))
            st = spec.offline_state(box)
            self.assertEqual(st.source, "offline")
            p = L.BY_NAME[name]
            np.testing.assert_allclose(st.origin, [p.start[0], p.start[1], L.TOP + L.MAT_TOP], atol=1e-12)

    def test_hooks_and_mechkit_reset(self):
        t = self.task
        tr = t._TRACKER
        self.mk.offline_reset()
        start = self.sim(start_scene())
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        done = self.sim(scene())
        for _ in range(WINDOW - 1):
            t.success(done)
        self.assertTrue(t.success(done))
        self.assertEqual(t.score(done), 1.0)
        self.assertEqual(t.score(done), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(t.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        self.assertEqual(t.score(self.sim(scene(tantalum=placed(T1, reverse=True)))), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(scene(resistor=placed(R1, across=0.003))))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "sixths", "zero_reasons", "valid_state",
                          "resistor", "capacitor", "tantalum", "resistor_origin", "tantalum_yaw_deg",
                          "capacitor_offset_mm", "tantalum_source"):
                    self.assertIn(k, row)
                for k in ("credit_sixths", "tier", "zero_reason", "at_rest", "moved_mm", "along_mm", "across_mm",
                          "tilt_deg", "axis_err_deg", "low_gap_mm", "end_overhang_mm", "side_overhang_mm",
                          "side_pct_w"):
                    self.assertIn(k, row["resistor"])
                self.assertIn("band_toward_plus", row["tantalum"])
                self.assertAlmostEqual(row["resistor_offset_mm"][1], 3.0)
                self.assertEqual(row["tantalum_source"], "sim")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(scene())))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if abs(prev.finger - w.finger) > 1e-9 and np.allclose(prev.pos, w.pos):
                    self.assertGreaterEqual(w.duration, 1.5, w.label)  # every close and open in place: 1.5 s
                if prev.pos[2] > w.pos[2] + 1e-9 and w.pos[2] < L.TOP + 0.03:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)  # final descents: slow, the last 30 mm
                    self.assertLessEqual(prev.pos[2] - w.pos[2], t.APPROACH + 1e-9)
        # grips 8 mm narrower than each part's width, releases 8 mm clear each side
        for p in L.PARTS:
            self.assertAlmostEqual(-2 * t.grip(p), p.width - 0.008)
            self.assertAlmostEqual(-2 * t.release(p), p.width + 0.016)
            self.assertGreaterEqual(p.width, 0.016)
            self.assertLessEqual(p.width, 0.080)
            self.assertLessEqual(p.mass, 0.30)
        # pad tips 7-9 mm above each part's underside at the pick and the place
        for p in L.PARTS:
            self.assertAlmostEqual(t.pick_point(p)[2] - 0.013 - (L.TOP + L.MAT_TOP), t.GRIP_UP[p.name] - 0.013)
            self.assertGreaterEqual(t.GRIP_UP[p.name] - 0.013, 0.007)
            self.assertLess(t.GRIP_UP[p.name] - 0.013, p.height - 0.010)  # pads hold at least 10 mm of it
            self.assertAlmostEqual(t.place_point(p)[2] - t.GRIP_UP[p.name], L.PAD_TOP_W + t.DROP)
            self.assertLessEqual(t.DROP, L.FLAT_TOL / 2)
        # the quarter turns: picked at -90 (left) / +90 (right), set down at 0. T1's band goes from +y to +x
        right = plan["right"]
        grip_t = [w for w in right if w.label.startswith("grip the tantalum")][0]
        place_t = [w for w in right if w.label.startswith("slowly lower the tantalum")][0]
        self.assertAlmostEqual(grip_t.yaw, math.pi / 2)
        self.assertAlmostEqual(place_t.yaw, 0.0)
        band = rz(T1.start_yaw + (place_t.yaw - grip_t.yaw)) @ np.array([1.0, 0.0, 0.0])
        self.assertGreater(band[0] * L.PLUS_PAD_SIGN, 0.99)
        for w in plan["left"]:
            if w.label.startswith("grip"):
                self.assertAlmostEqual(w.yaw, -math.pi / 2)
        # the place points are the footprints' centres (no trims yet)
        for p in L.PARTS:
            q = t.place_point(p)
            self.assertAlmostEqual(q[0], p.footprint.x)
            self.assertAlmostEqual(q[1], p.footprint.y)
        # the arms take turns at the board: the right arm is back up before the left arm comes over it
        def times(wps):
            out, s = [], 0.0
            for w in wps:
                s += w.duration
                out.append((s, w))
            return out
        right_up = [s for s, w in times(right) if w.label == "straight up"][0]
        left_over = [s - w.duration for s, w in times(plan["left"]) if w.label.startswith("carry the resistor")][0]
        self.assertLess(right_up, left_over)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_task_json_matches_the_code(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], self.task.INSTRUCTION)
        self.assertEqual(d["category"], "Impact on jobs")
        for s in ("8.25 and 14.5 mm (R1), 7 and 12 mm (C1) or 10.5 and 19 mm (T1)", "6.25 mm for R1, 5 mm for C1, "
                  "8.5 mm for T1", "12.5, 10 and 17 mm", "up to 6 mm off along the axis", "within 2 mm of the pads",
                  "5 degrees or less", "within 10 degrees of x", "more than 2 mm in the last 0.5 s",
                  "more than 5 cm below the table top", "held at rest for 2.5 s", "on any of them",
                  "on its side (R1, T1) or an end", "band face up with its lowest point within 2 mm of the pads' top"):
            self.assertIn(s, d["scoring"])
        for s in ("150 mm (x) by 220 mm (y), 4 mm thick", "pads 18 mm along x by 29 mm", "pads 16 x 24 mm",
                  "pads 18 x 38 mm", "centred at x 32.6 cm, y -9.5 cm", "50 mm long, 25 mm wide, 20 mm tall, 30 g",
                  "26 mm long, 20 mm wide, 20 mm tall, 15 g", "58 mm long, 34 mm wide, 24 mm tall, 50 g",
                  "x 20.0 cm, y +21.5 cm", "x 20.0 cm, y +26.5 cm", "x 20.0 cm, y -24.0 cm"):
            self.assertIn(s, d["setup"])
        self.assertAlmostEqual(L.PLUS_MARK_Y, -0.095)
        self.assertIn("1,800 steps", d["run_limit"])


if __name__ == "__main__":
    unittest.main()
