"""Offline tests for scoring.py and task.py: synthetic item states and gripper snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import itertools
import json
import math
import os
import pathlib
import random
import re
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
    extent,
    footprint,
    lowest,
    overlap,
    point_distance,
    root_err_deg,
    square_err_deg,
    tilt_deg,
    upright_err_deg,
    valid,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FULL = WINDOW * L.STILL_WINDOWS + 1  # 21 observations: 1 s of history
SCORES = {0.0, 0.4, 0.6, 0.8, 1.0}
LX, LY = L.LABEL_SIZE[0] / 2, L.LABEL_SIZE[1] / 2
SX, SY = L.SLIP_SIZE[0] / 2, L.SLIP_SIZE[1] / 2
BX, BY = L.BOARD_SIZE[0] / 2, L.BOARD_SIZE[1] / 2


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


def raw(origin, q, vel=(0.0, 0.0, 0.0)):
    """An item's state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o.copy(),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.05, high=o + 0.05)


def at(cx, cy, yaw=0.0, up=L.SHEET_T, pitch=0.0, roll=0.0, vel=(0.0, 0.0, 0.0)):
    """An item lying with its bottom face's centre at (cx, cy), ``up`` above the table top (0.5 mm: on the sheet)."""
    return raw((cx, cy, L.TOP + up), quat(yaw, pitch, roll), vel)


def label_at(right_in=0.007, foot_in=0.007, yaw=0.0, **kw):
    """The label square (yaw 0: long side along y) with its right edge ``right_in`` and its bottom edge ``foot_in``
    inside the sheet's right edge and foot."""
    return at(L.SHEET_X0 + foot_in + LX, L.SHEET_Y0 + right_in + LY, yaw, **kw)


def slip_above(gap=0.006, align=0.0, label=None, yaw=0.0, **kw):
    """The slip square, its bottom edge ``gap`` above the label's top edge, its right edge ``align`` left of the
    label's right edge."""
    lab = label if label is not None else label_at()
    x0, x1, y0, y1 = extent(lab, "label")
    return at(x1 + gap + SX, y0 + align + SY, yaw, **kw)


def slip_left(gap=0.005, align=0.0, label=None, **kw):
    """The slip square, its right edge ``gap`` left of the label's left edge, its bottom edge ``align`` above the
    label's bottom edge."""
    lab = label if label is not None else label_at()
    x0, x1, y0, y1 = extent(lab, "label")
    return at(x0 + align + SX, y1 + gap + SY, **kw)


def spec_at(cx=0.355, cy=0.045, yaw=180.0, **kw):
    return at(cx, cy, yaw, **kw)


def start_state(name):
    x, y, yaw = L.START[name]
    return raw((x, y, L.TOP), quat(yaw))


START = {n: start_state(n) for n in L.ITEMS}
GOOD = {"label": label_at(), "slip": slip_above(), "specimen": spec_at()}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


FAR = {"left": arm((0.20, 0.30, L.TOP + 0.16), 86 / 93.9), "right": arm((0.20, -0.30, L.TOP + 0.16), 81 / 93.9)}


def holding(name, st, side="right", opening=None):
    """``side``'s gripper closed on the item where it lies, as the demo holds it (TCP 16 mm above its underside)."""
    o = np.asarray(st.origin, dtype=float)
    w = {"label": 0.059, "slip": 0.012, "specimen": 0.062}[name]
    arms = dict(FAR)
    arms[side] = arm((o[0], o[1], o[2] + 0.016), w / 93.9 if opening is None else opening)
    return arms


def feed(tr, states, arms=FAR):
    r = None
    for st in states:
        r = tr.update(st, arms)
    return r


def settled(items, arms=FAR, n=FULL):
    """A tracker that has seen ``items`` at rest for ``n`` steps; returns (tracker, last result)."""
    tr = Tracker(HZ)
    return tr, feed(tr, [items] * n, arms)


def scored(items, arms=FAR):
    return settled(items, arms)[1].score


def with_(**kw):
    d = dict(GOOD)
    d.update(kw)
    return d


class Geometry(unittest.TestCase):
    def test_sheet_is_the_standard_sheet_at_1_to_1_5(self):
        self.assertAlmostEqual(L.SHEET_LEN * 1.5, 16.5 * 0.0254, places=6)
        self.assertAlmostEqual(L.SHEET_W * 1.5, 11.5 * 0.0254, delta=0.0002)
        self.assertAlmostEqual(L.SHEET_X1, 0.4844)
        self.assertAlmostEqual(L.SHEET_Y1 - L.SHEET_Y0, 0.1947)
        # 'a few mm from the edges': 2-12 mm here is 3-18 mm on the full-size sheet
        self.assertAlmostEqual(L.EDGE_IN_MIN * 1.5, 0.003)
        self.assertAlmostEqual(L.EDGE_IN_MAX * 1.5, 0.018)
        # the barcode is in the top right corner, on the sheet
        self.assertGreater(L.BARCODE_X0, (L.SHEET_X0 + L.SHEET_X1) / 2)
        self.assertLess(L.BARCODE_Y1, 0.0)
        self.assertTrue(L.SHEET_X0 < L.BARCODE_X0 < L.BARCODE_X1 < L.SHEET_X1)
        self.assertTrue(L.SHEET_Y0 < L.BARCODE_Y0 < L.BARCODE_Y1 < L.SHEET_Y1)

    def test_start_layout(self):
        for n, st in START.items():
            self.assertTrue(valid(st), n)
            self.assertAlmostEqual(lowest(st, n), 0.0, places=9)
            x0, x1, y0, y1 = extent(st, n)
            off = x1 < L.SHEET_X0 or x0 > L.SHEET_X1 or y1 < L.SHEET_Y0 or y0 > L.SHEET_Y1
            self.assertTrue(off, f"{n} starts off the sheet")
        self.assertTrue(Tracker.at_start(START))
        self.assertEqual(scored(START), 0.0)
        np.testing.assert_allclose(extent(START["specimen"], "specimen"), (0.265, 0.335, 0.150, 0.350), atol=1e-9)
        self.assertLess(root_err_deg(START["specimen"]), 91.0)  # roots toward +y at the start: 90 degrees off

    def test_targets(self):
        lab, slip, sp = GOOD["label"], GOOD["slip"], GOOD["specimen"]
        np.testing.assert_allclose(lab.origin[:2], L.LABEL_TARGET, atol=1e-9)
        np.testing.assert_allclose(slip.origin[:2], L.SLIP_TARGET, atol=1e-9)
        np.testing.assert_allclose(sp.origin[:2], L.SPEC_TARGET[:2], atol=1e-9)
        np.testing.assert_allclose(extent(lab, "label"), (0.212, 0.259, -0.09035, -0.02335), atol=1e-9)
        np.testing.assert_allclose(extent(slip, "slip"), (0.265, 0.285, -0.09035, -0.03035), atol=1e-9)
        np.testing.assert_allclose(extent(sp, "specimen"), (0.255, 0.455, 0.010, 0.080), atol=1e-9)
        x0, x1, y0, y1 = extent(sp, "specimen")
        self.assertGreater(min(x0 - L.SHEET_X0, L.SHEET_X1 - x1, y0 - L.SHEET_Y0, L.SHEET_Y1 - y1), 0.017)
        self.assertGreater(y0 - extent(lab, "label")[3], 0.030)  # 33 mm clear of the label
        self.assertGreater(y0 - L.BARCODE_Y1, 0.045)

    def test_footprint_turns_with_the_item(self):
        st = label_at(yaw=90.0)
        x0, x1, y0, y1 = extent(st, "label")
        self.assertAlmostEqual(x1 - x0, 0.067, places=9)
        self.assertAlmostEqual(y1 - y0, 0.047, places=9)
        st = at(0.3, 0.0, 30.0)
        f = footprint(st, "label")
        self.assertAlmostEqual(np.linalg.norm(f[0] - f[1]), 0.047, places=9)
        self.assertAlmostEqual(np.linalg.norm(f[1] - f[2]), 0.067, places=9)

    def test_overlap(self):
        a = footprint(at(0.3, 0.0), "label")
        self.assertTrue(overlap(a, footprint(at(0.3 + 0.046, 0.0), "label")))
        self.assertFalse(overlap(a, footprint(at(0.3 + 0.047, 0.0), "label")))  # sharing an edge only
        self.assertFalse(overlap(a, footprint(at(0.3 + 0.048, 0.0), "label")))
        # turned 45 degrees: its corner reaches 0.5 * (47 + 67) / sqrt(2) = 40.3 mm
        self.assertTrue(overlap(a, footprint(at(0.3 + 0.0235 + 0.040, 0.0, 45.0), "label")))
        self.assertFalse(overlap(a, footprint(at(0.3 + 0.0235 + 0.041, 0.0, 45.0), "label")))

    def test_angles(self):
        self.assertAlmostEqual(tilt_deg(at(0.3, 0.0)), 0.0)
        self.assertAlmostEqual(tilt_deg(at(0.3, 0.0, pitch=7.0)), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(at(0.3, 0.0, roll=180.0)), 180.0, places=6)
        for yaw, err in ((0.0, 0.0), (180.0, 0.0), (4.0, 4.0), (-4.0, 4.0), (176.0, 4.0), (90.0, 90.0), (45.0, 45.0)):
            self.assertAlmostEqual(square_err_deg(at(0.3, 0.0, yaw)), err, places=6)
        for yaw, err in ((180.0, 0.0), (165.0, 15.0), (-170.0, 10.0), (0.0, 180.0), (90.0, 90.0)):
            self.assertAlmostEqual(root_err_deg(at(0.3, 0.0, yaw)), err, places=6)
        for yaw, err in ((0.0, 0.0), (5.0, 5.0), (-5.0, 5.0), (180.0, 180.0), (175.0, 175.0), (90.0, 90.0)):
            self.assertAlmostEqual(upright_err_deg(at(0.3, 0.0, yaw)), err, places=6)

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        st = GOOD["label"]
        o = np.asarray(st.origin)
        for opening in (0.0, 1.0):  # the zone is flat at the pad tips: tips 100 mm over the printed top is 100 mm
            up = arm((o[0], o[1], o[2] + 0.012 + 0.0005 + 0.013 + 0.100), opening)
            self.assertAlmostEqual(zone_gap(up, st, "label"), 0.100, places=6)
        self.assertAlmostEqual(zone_gap(holding("label", st)["right"], st, "label"), 0.0, places=9)
        # beside the item, at its height: the pad's outer corner (zone radius) counts
        side = arm((o[0], o[1] - LY - 0.030 - zone_radius(arm((0, 0, 0), 1.0)), o[2] + 0.020), 1.0)
        self.assertAlmostEqual(zone_gap(side, st, "label"), 0.030, places=6)
        # tilted gripper, corner case: the zone gap agrees with a brute-force sample of the cylinder
        tilt = SimpleNamespace(tcp=np.array([0.29, 0.01, L.TOP + 0.10]), tips=np.array([0.30, 0.0, L.TOP + 0.088]),
                               opening=0.5)
        u = (tilt.tcp - tilt.tips) / np.linalg.norm(tilt.tcp - tilt.tips)
        e1 = np.cross(u, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        r = zone_radius(tilt)
        pts = [tilt.tips + a * u + rho * (math.cos(th) * e1 + math.sin(th) * e2)
               for a in np.linspace(0.0, L.FINGER_LEN, 12) for rho in np.linspace(0.0, r, 10)
               for th in np.linspace(0.0, 2 * math.pi, 48, endpoint=False)]
        brute = min(point_distance(q, st, "label") for q in pts)
        self.assertAlmostEqual(zone_gap(tilt, st, "label"), brute, delta=0.0005)


class Tiers(unittest.TestCase):
    def test_target_layout_scores_one(self):
        tr, r = settled(GOOD)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["correct"], ["label", "slip", "specimen"])

    def test_slip_left_of_the_label_also_scores(self):
        """The Florida Museum guide allows annotation slips above or to the left of the label."""
        st = with_(slip=slip_left())
        self.assertEqual(scored(st), 1.0)
        tr, r = settled(st)
        self.assertTrue(r.details["slip"]["left"])
        self.assertFalse(r.details["slip"]["above"])
        self.assertFalse(overlap(footprint(st["slip"], "slip"), footprint(st["specimen"], "specimen")))

    def test_every_combination(self):
        away = {"label": START["label"], "slip": START["slip"], "specimen": START["specimen"]}
        for lab, slp, spc in itertools.product((True, False), repeat=3):
            st = {"label": GOOD["label"] if lab else away["label"], "slip": GOOD["slip"] if slp else away["slip"],
                  "specimen": GOOD["specimen"] if spc else away["specimen"]}
            want = 0.4 * lab + 0.2 * (slp and lab) + 0.4 * spc
            self.assertAlmostEqual(scored(st), want, msg=f"label {lab} slip {slp} specimen {spc}")

    def test_slip_alone_scores_nothing(self):
        st = {"label": START["label"], "slip": slip_above(), "specimen": START["specimen"]}
        tr, r = settled(st)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["slip"]["zero_reason"], "label_not_correct")

    def test_label_band_edges(self):
        for right_in, foot_in, want in ((0.002, 0.007, 1.0), (0.012, 0.007, 1.0), (0.007, 0.002, 1.0),
                                        (0.007, 0.012, 1.0), (0.0019, 0.007, 0.4), (0.0121, 0.007, 0.4),
                                        (0.007, 0.0019, 0.4), (0.007, 0.0121, 0.4)):
            lab = label_at(right_in, foot_in)
            st = {"label": lab, "slip": slip_above(label=lab), "specimen": GOOD["specimen"]}
            self.assertAlmostEqual(scored(st), want, msg=f"right {right_in} foot {foot_in}")

    def test_label_squareness(self):
        for yaw, want in ((5.0, 1.0), (-5.0, 1.0), (5.1, 0.4), (-5.1, 0.4), (90.0, 0.4), (180.0, 0.4),
                          (175.0, 0.4), (-175.0, 0.4)):
            lab = label_at(0.007 + 0.002 * abs(math.sin(math.radians(yaw))), 0.009, yaw)
            st = {"label": lab, "slip": slip_above(gap=0.007, label=lab), "specimen": GOOD["specimen"]}
            self.assertAlmostEqual(scored(st), want, msg=f"yaw {yaw}")

    def test_label_upside_down_is_wrong(self):
        # square to the sheet but turned end for end: its print reads upside down from the foot, as no curator lays it
        lab = label_at(yaw=180.0)
        tr, r = settled({"label": lab, "slip": slip_above(label=lab), "specimen": GOOD["specimen"]})
        self.assertAlmostEqual(r.score, 0.4)
        self.assertEqual(r.details["label"]["zero_reason"], "upside_down")
        self.assertAlmostEqual(r.details["label"]["upright_err_deg"], 180.0, places=4)

    def test_slip_gap_and_alignment_edges(self):
        for gap, align, want in ((0.0, 0.0, 1.0), (0.010, 0.0, 1.0), (0.0101, 0.0, 0.8), (0.006, 0.010, 1.0),
                                 (0.006, 0.0101, 0.8), (-0.0005, 0.0, 0.8)):
            st = with_(slip=slip_above(gap, align))
            self.assertAlmostEqual(scored(st), want, msg=f"gap {gap} align {align}")
        lab = label_at(0.012, 0.007)  # the label 12 mm in, so a slip 10 mm further right is still on the sheet
        for align, want in ((-0.010, 1.0), (-0.0101, 0.8)):
            st = with_(label=lab, slip=slip_above(0.006, align, label=lab))
            self.assertAlmostEqual(scored(st), want, msg=f"align {align}")
        for gap, align, want in ((0.0, 0.0, 1.0), (0.010, 0.0, 1.0), (0.0101, 0.0, 0.8), (0.005, 0.010, 1.0),
                                 (0.005, 0.0101, 0.8)):
            st = with_(slip=slip_left(gap, align))
            self.assertAlmostEqual(scored(st), want, msg=f"left gap {gap} align {align}")

    def test_slip_squareness(self):
        self.assertEqual(scored(with_(slip=slip_above(gap=0.007, yaw=4.9))), 1.0)
        tr, r = settled(with_(slip=slip_above(gap=0.007, yaw=180.0)))  # square but upside down
        self.assertAlmostEqual(r.score, 0.8)
        self.assertEqual(r.details["slip"]["zero_reason"], "upside_down")
        self.assertEqual(scored(with_(slip=slip_above(gap=0.007, yaw=-4.9))), 1.0)
        self.assertAlmostEqual(scored(with_(slip=slip_above(gap=0.007, yaw=5.2))), 0.8)
        self.assertAlmostEqual(scored(with_(slip=slip_above(gap=0.03, yaw=90.0))), 0.8)

    def test_specimen_margins(self):
        sheet = {"foot": (L.SHEET_X0 + 0.002 + BX, 0.045), "head": (L.SHEET_X1 - 0.002 - BX, 0.045),
                 "left": (0.355, L.SHEET_Y1 - 0.002 - BY)}
        for edge, (cx, cy) in sheet.items():
            self.assertEqual(scored(with_(specimen=spec_at(cx, cy))), 1.0, edge)
        for edge, (cx, cy) in (("foot", (L.SHEET_X0 + 0.0019 + BX, 0.045)),
                               ("head", (L.SHEET_X1 - 0.0019 - BX, 0.045)),
                               ("left", (0.355, L.SHEET_Y1 - 0.0019 - BY))):
            tr, r = settled(with_(specimen=spec_at(cx, cy)))
            self.assertAlmostEqual(r.score, 0.6, msg=edge)
            self.assertEqual(r.details["specimen"]["zero_reason"], "not_on_sheet")

    def test_specimen_orientation(self):
        for yaw, want in ((180.0, 1.0), (165.0, 1.0), (195.0, 1.0), (164.9, 0.6), (195.1, 0.6), (0.0, 0.6),
                          (90.0, 0.6)):
            sp = spec_at(0.355, 0.045 if abs(yaw) not in (90.0,) else 0.0, yaw)
            if abs(yaw - 180.0) < 20.0:
                # turned 15 deg the 200 x 70 mm board reaches 59.7 mm either side of its centre in y: at y 30 mm it
                # stays 7.7 mm inside the left edge and clear of the label, slip and barcode
                sp = spec_at(0.360, 0.030, yaw)
            self.assertAlmostEqual(scored(with_(specimen=sp)), want, msg=f"yaw {yaw}")

    def test_possible_scores(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(300):
            st = {}
            for n in L.ITEMS:
                if rng.random() < 0.3:
                    st[n] = START[n]
                else:
                    g = np.asarray(GOOD[n].origin)
                    st[n] = at(g[0] + rng.uniform(-0.02, 0.02), g[1] + rng.uniform(-0.02, 0.02),
                               (180.0 if n == "specimen" else 0.0) + rng.uniform(-8, 8))
            s = scored(st)
            self.assertIn(round(s, 6), SCORES)
            seen.add(round(s, 6))
        self.assertEqual(seen, SCORES)  # every tier occurs; 0.2 never does

    def test_score_is_idempotent(self):
        tr, r = settled(GOOD)
        a = tr.result(GOOD, FAR).score
        b = tr.result(GOOD, FAR).score
        self.assertEqual((a, b, tr.steps), (1.0, 1.0, FULL))


class ZeroConditions(unittest.TestCase):
    def reason(self, items, name, arms=FAR):
        return settled(items, arms)[1].details[name]["zero_reason"]

    def test_label_face_down(self):
        st = with_(label=label_at(roll=180.0, up=L.SHEET_T + 0.012))
        self.assertAlmostEqual(scored(st), 0.4)
        self.assertEqual(self.reason(st, "label"), "face_down")

    def test_label_tilted(self):
        st = with_(label=label_at(pitch=6.0, up=L.SHEET_T + 0.0025))
        self.assertAlmostEqual(scored(st), 0.4)
        self.assertEqual(self.reason(st, "label"), "not_flat")
        self.assertEqual(scored(with_(label=label_at(pitch=4.0, up=L.SHEET_T + 0.0017))), 1.0)

    def test_label_not_on_the_table(self):
        for up in (0.0035, 0.012, -0.004):
            st = with_(label=label_at(up=up))
            self.assertAlmostEqual(scored(st), 0.4, msg=up)
            self.assertEqual(self.reason(st, "label"), "not_on_table")
        self.assertEqual(scored(with_(label=label_at(up=0.0029))), 1.0)

    def test_label_held(self):
        st = GOOD
        self.assertAlmostEqual(scored(st, holding("label", st["label"])), 0.4)
        self.assertEqual(self.reason(st, "label", holding("label", st["label"])), "not_let_go")

    def test_hands_off_is_20_mm(self):
        st = GOOD
        o = np.asarray(st["label"].origin)
        top = o[2] + 0.0125
        for opening in (0.0, 1.0):
            for h, want in ((0.020, 1.0), (0.021, 1.0), (0.019, 0.4)):
                a = arm((o[0], o[1], top + 0.013 + h), opening)  # its pad tips h right above the label's top
                arms = {"left": FAR["left"], "right": a}
                self.assertAlmostEqual(scored(st, arms), want, msg=f"{h} {opening}")
        for h, want in ((0.020, 1.0), (0.019, 0.4)):  # beside it: from the pad's outer corner
            a = arm((o[0], o[1] - LY - h - zone_radius(arm((0, 0, 0), 1.0)), o[2] + 0.020), 1.0)
            self.assertAlmostEqual(scored(st, {"left": FAR["left"], "right": a}), want, msg=f"side {h}")

    def test_left_gripper_counts_too(self):
        self.assertAlmostEqual(scored(GOOD, holding("specimen", GOOD["specimen"], side="left")), 0.6)

    def test_not_still(self):
        tr = Tracker(HZ)
        lab = GOOD["label"]
        o = np.asarray(lab.origin)
        states = [with_(label=raw(o + np.array([0.0002 * i, 0.0, 0.0]), lab.quat_xyzw)) for i in range(FULL)]
        r = feed(tr, states)
        self.assertAlmostEqual(r.score, 0.4)  # 2 mm in each 0.5 s window: still moving (the slip needs the label)
        self.assertEqual(r.details["label"]["zero_reason"], "not_still")
        self.assertEqual(r.details["slip"]["zero_reason"], "label_not_correct")
        tr = Tracker(HZ)
        slp = GOOD["slip"]
        o = np.asarray(slp.origin)
        r = feed(tr, [with_(slip=raw(o + np.array([0.0002 * i, 0.0, 0.0]), slp.quat_xyzw)) for i in range(FULL)])
        self.assertAlmostEqual(r.score, 0.8)  # only the slip still moving
        self.assertEqual(r.details["slip"]["zero_reason"], "not_still")

    def test_slip_overlapping_the_label(self):
        st = with_(slip=slip_above(gap=-0.002))
        self.assertAlmostEqual(scored(st), 0.8)
        self.assertEqual(self.reason(st, "slip"), "overlaps_label")

    def test_slip_off_the_sheet(self):
        lab = label_at(0.002, 0.007)
        st = with_(label=lab, slip=slip_above(gap=0.005, align=-0.009, label=lab))  # 7 mm over the right edge
        self.assertAlmostEqual(scored(st), 0.8)
        self.assertEqual(self.reason(st, "slip"), "off_sheet")

    def test_slip_beside_but_not_next_to(self):
        lab = GOOD["label"]
        x0, x1, y0, y1 = extent(lab, "label")
        st = with_(slip=at(x1 + 0.03 + SX, y0 + SY))  # 30 mm above it
        self.assertAlmostEqual(scored(st), 0.8)
        self.assertEqual(self.reason(st, "slip"), "not_next_to_label")
        st = with_(slip=at(x0 + 0.015 + SX, y1 + 0.005 + SY))  # left of it but 15 mm higher (clear of the plant)
        self.assertAlmostEqual(scored(st), 0.8)
        self.assertEqual(self.reason(st, "slip"), "not_next_to_label")

    def test_specimen_covering(self):
        lab = GOOD["label"]
        ly1 = extent(lab, "label")[3]
        st = with_(specimen=spec_at(0.355, ly1 - 0.005 + BY))  # over the label by 5 mm
        self.assertAlmostEqual(scored(st), 0.6)
        self.assertEqual(self.reason(st, "specimen"), "covers_label")
        sy1 = extent(GOOD["slip"], "slip")[3]
        st = with_(specimen=spec_at(0.380, sy1 - 0.003 + BY))  # over the slip only
        self.assertEqual(self.reason(st, "specimen"), "covers_slip")
        self.assertAlmostEqual(scored(st), 0.6)
        st = with_(label=label_at(), slip=slip_left(), specimen=spec_at(0.380, -0.045 + BY))  # 5 mm over the barcode
        self.assertEqual(self.reason(st, "specimen"), "covers_barcode")
        self.assertAlmostEqual(scored(st), 0.6)

    def test_specimen_face_down_or_on_edge(self):
        st = with_(specimen=spec_at(roll=180.0, up=L.SHEET_T + 0.010))
        self.assertAlmostEqual(scored(st), 0.6)
        self.assertEqual(self.reason(st, "specimen"), "face_down")
        st = with_(specimen=spec_at(roll=90.0, up=L.SHEET_T + BY))
        self.assertAlmostEqual(scored(st), 0.6)


class ShortcutChecks(unittest.TestCase):
    """The spec's shortcut checks, one by one."""

    def test_1_label_bottom_centre(self):
        """Label at the middle of the foot, slip correctly above it, plant correct (laid further up and left so it
        clears them): only the plant counts."""
        lab = at(L.SHEET_X0 + 0.007 + LX, 0.0)
        sp = spec_at(0.380, 0.060)  # footprint x 280-480, y 25-95 mm
        st = {"label": lab, "slip": slip_above(label=lab, align=-0.005), "specimen": sp}
        tr, r = settled(st)
        self.assertAlmostEqual(r.score, 0.4)
        self.assertEqual(r.details["label"]["zero_reason"], "not_at_right_edge")
        self.assertEqual(r.details["slip"]["zero_reason"], "label_not_correct")
        self.assertIn("specimen", r.details["correct"])

    def test_wrong_corners(self):
        for name, lab in (("bottom left", at(L.SHEET_X0 + 0.007 + LX, L.SHEET_Y1 - 0.007 - LY)),
                          ("top right", at(L.SHEET_X1 - 0.007 - LX - 0.025, L.SHEET_Y0 + 0.007 + LY)),
                          ("top left", at(L.SHEET_X1 - 0.007 - LX, L.SHEET_Y1 - 0.007 - LY))):
            st = {"label": lab, "slip": START["slip"], "specimen": START["specimen"]}
            self.assertEqual(scored(st), 0.0, name)

    def test_2_slip_overlapping_or_left_at_start(self):
        self.assertAlmostEqual(scored(with_(slip=slip_above(gap=-0.004))), 0.8)
        self.assertAlmostEqual(scored(with_(slip=START["slip"])), 0.8)

    def test_3_slip_left_of_label(self):
        self.assertEqual(scored(with_(slip=slip_left())), 1.0)

    def test_4_roots_toward_the_head(self):
        st = with_(specimen=spec_at(yaw=0.0))
        self.assertAlmostEqual(scored(st), 0.6)
        self.assertEqual(settled(st)[1].details["specimen"]["zero_reason"], "roots_toward_head")

    def test_5_plant_over_the_label_by_5_mm(self):
        ly1 = extent(GOOD["label"], "label")[3]
        self.assertAlmostEqual(scored(with_(specimen=spec_at(0.355, ly1 - 0.005 + BY))), 0.6)

    def test_6_label_in_the_corner_on_the_specimen(self):
        sp = spec_at(L.SHEET_X0 + 0.010 + BX, L.SHEET_Y0 + 0.004 + BY)
        lab = label_at(up=L.SHEET_T + 0.010)  # in the corner, resting on the board
        st = {"label": lab, "slip": slip_above(label=lab, up=L.SHEET_T), "specimen": sp}
        tr, r = settled(st)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["label"]["zero_reason"], "not_on_table")
        self.assertEqual(r.details["specimen"]["zero_reason"], "covers_label_and_slip")

    def test_nothing_moved(self):
        self.assertEqual(scored(START), 0.0)


class Exploits(unittest.TestCase):
    def test_label_and_slip_swapped(self):
        """The slip in the label's corner and the label above it: only the plant counts."""
        slip = at(L.SHEET_X0 + 0.007 + SX, L.SHEET_Y0 + 0.007 + SY)
        x1 = extent(slip, "slip")[1]
        lab = at(x1 + 0.005 + LX, L.SHEET_Y0 + 0.007 + LY)
        self.assertAlmostEqual(scored({"label": lab, "slip": slip, "specimen": GOOD["specimen"]}), 0.4)

    def test_slip_stacked_on_the_label(self):
        lab = GOOD["label"]
        slip = at(lab.origin[0], lab.origin[1] - 0.003, up=L.SHEET_T + 0.012)
        tr, r = settled(with_(slip=slip))
        self.assertAlmostEqual(r.score, 0.8)  # no more than leaving the slip out
        self.assertEqual(r.details["slip"]["zero_reason"], "not_on_table")

    def test_everything_held_in_place(self):
        st = GOOD
        arms = holding("specimen", st["specimen"], "left")
        self.assertAlmostEqual(scored(st, arms), 0.6)  # the plant held in place: label and slip only
        arms = {"left": holding("specimen", st["specimen"], "left")["left"], "right": holding("slip", st["slip"])["right"]}
        # slip and plant held; the gripper on the slip is also within 20 mm of the label 6 mm below it: nothing counts
        self.assertAlmostEqual(scored(st, arms), 0.0)
        arms = holding("label", st["label"])
        self.assertAlmostEqual(scored(st, arms), 0.4)  # label held: neither label nor slip counts

    def test_pushed_by_an_open_gripper(self):
        st = GOOD
        o = np.asarray(st["label"].origin)
        arms = dict(FAR)
        arms["right"] = arm((o[0], o[1] - LY - 0.020, o[2] + 0.016), 1.0)  # an open pad against its end
        self.assertAlmostEqual(scored(st, arms), 0.4)

    def test_plant_on_top_of_the_paperwork(self):
        sp = spec_at(0.300, -0.050, yaw=180.0, up=L.SHEET_T + 0.012)  # laid over label and slip
        tr, r = settled(with_(specimen=sp))
        self.assertAlmostEqual(r.score, 0.6)
        self.assertEqual(r.details["specimen"]["zero_reason"], "not_on_table")

    def test_board_on_its_edge_reaching_the_sheet(self):
        sp = spec_at(0.355, 0.045, yaw=180.0, roll=90.0, up=L.SHEET_T + BY)
        self.assertAlmostEqual(scored(with_(specimen=sp)), 0.6)

    def test_items_off_the_table(self):
        st = with_(label=at(0.30, -0.20, up=-0.30), specimen=at(0.30, 0.20, 180.0, up=-0.30))
        self.assertEqual(scored(st), 0.0)

    def test_arms_unreadable_cannot_be_used_to_skip_the_rest(self):
        st = with_(label=label_at(0.015, 0.007))
        tr = Tracker(HZ)
        r = feed(tr, [st] * FULL, None)
        self.assertAlmostEqual(r.score, 0.4)
        self.assertEqual(r.details["arms_unavailable"], FULL)


class Stillness(unittest.TestCase):
    def test_needs_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [GOOD] * (FULL - 1))
        self.assertEqual(r.score, 0.0)
        r = feed(tr, [GOOD])
        self.assertEqual(r.score, 1.0)

    def test_both_windows_must_be_still(self):
        lab = GOOD["label"]
        o = np.asarray(lab.origin)
        moving = [with_(label=raw(o + np.array([0.0, -0.0003 * (WINDOW - i), 0.0]), lab.quat_xyzw))
                  for i in range(WINDOW)]
        tr = Tracker(HZ)
        feed(tr, moving + [GOOD] * WINDOW)  # 3 mm in the older window, still in the newer
        r = feed(tr, [GOOD])
        self.assertAlmostEqual(r.score, 0.4)  # the label is not yet still, so neither it nor the slip counts
        r = feed(tr, [GOOD] * WINDOW)
        self.assertEqual(r.score, 1.0)

    def test_small_jitter_is_still_but_chatter_is_not(self):
        lab = GOOD["label"]
        o = np.asarray(lab.origin)
        for step, want in ((0.00007, 1.0), (0.00016, 0.4)):  # path per 0.5 s: 1.4 mm, 3.2 mm
            st = [with_(label=raw(o + np.array([step * (i % 2), 0.0, 0.0]), lab.quat_xyzw)) for i in range(FULL)]
            self.assertAlmostEqual(feed(Tracker(HZ), st).score, want, msg=step)

    def test_turning_in_place_is_not_still(self):
        lab = GOOD["label"]
        o = np.asarray(lab.origin)
        # its top corners are 40.9 mm from its centre: 1.5 deg per 0.5 s moves them 1.07 mm (still), 3 deg 2.14 mm
        st = [with_(label=raw(o, quat(0.15 * i))) for i in range(FULL)]
        self.assertAlmostEqual(feed(Tracker(HZ), st).score, 1.0)
        st = [with_(label=raw(o, quat(0.30 * i))) for i in range(FULL)]
        self.assertAlmostEqual(feed(Tracker(HZ), st).score, 0.4)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr = Tracker(HZ)
        r = feed(tr, [GOOD] * FULL, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["arms_unavailable"], FULL)
        self.assertIsNone(r.details["zone_gap_mm"])

    def test_non_finite_arms_count_as_unavailable(self):
        bad = {"left": arm((math.nan, 0, 0)), "right": FAR["right"]}
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": FAR["left"]}))
        self.assertFalse(arms_valid({"left": FAR["left"], "right": arm((0, 0, 0), math.inf)}))
        tr = Tracker(HZ)
        feed(tr, [GOOD] * FULL, bad)
        self.assertEqual(tr.arms_unavailable, FULL)

    def test_hands_off_applies_when_the_arms_come_back(self):
        tr = Tracker(HZ)
        feed(tr, [GOOD] * FULL, None)
        self.assertAlmostEqual(tr.update(GOOD, holding("label", GOOD["label"])).score, 0.4)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        lab = GOOD["label"]
        bad_states = [
            raw((math.nan, 0, 0), lab.quat_xyzw), raw((math.inf, 0, 0), lab.quat_xyzw), raw(lab.origin, (0, 0, 0, 0)),
            raw(lab.origin, (math.nan, 0, 0, 1)), SimpleNamespace(pos=lab.origin, quat_xyzw=lab.quat_xyzw),
            None, raw(lab.origin[:2], lab.quat_xyzw),
        ]
        for b in bad_states:
            tr, _ = settled(GOOD)
            before = {n: len(h) for n, h in tr.history.items()}
            r = tr.update(with_(label=b), FAR)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
            self.assertEqual(r.details["bad_items"], ["label"])
            self.assertEqual({n: len(h) for n, h in tr.history.items()}, before)
            self.assertEqual(tr.bad_steps, 1)
            self.assertEqual(tr.update(GOOD, FAR).score, 1.0)  # picks up where it was
        tr = Tracker(HZ)
        r = tr.update({"label": GOOD["label"], "slip": GOOD["slip"]}, FAR)  # an item missing
        self.assertEqual(r.score, 0.0)

    def test_unnormalised_quaternion(self):
        st = with_(label=raw(GOOD["label"].origin, 3.0 * quat(0.0)))
        self.assertEqual(scored(st), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(3)
        tr = Tracker(HZ)
        for _ in range(400):
            st = {n: raw((rng.uniform(-1, 1), rng.uniform(-1, 1), L.TOP + rng.uniform(-0.5, 0.5)),
                         [rng.uniform(-1, 1) for _ in range(4)], [rng.uniform(-2, 2) for _ in range(3)])
                  for n in L.ITEMS}
            arms = {s: arm([rng.uniform(-1, 1) for _ in range(3)], rng.uniform(0, 1)) for s in ("left", "right")}
            r = tr.update(st, arms)
            self.assertIn(round(r.score, 6), SCORES)
            json.dumps(r.details, default=str)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(GOOD)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START, FAR)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.first_at, {})

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(GOOD)
        jit = {n: raw(np.asarray(s.origin) + np.array([0.02, -0.02, 0.0]), quat(L.START[n][2] + 5.0))
               for n, s in START.items()}
        tr.update(jit, FAR)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(GOOD)
        n0 = tr.steps
        for k in range(1, 41):
            st = {}
            for n in L.ITEMS:
                a, b = np.asarray(GOOD[n].origin), np.asarray(START[n].origin)
                st[n] = raw(a + (b - a) * k / 40, START[n].quat_xyzw, (b - a) * HZ / 40)
            tr.update(st, FAR)
        self.assertEqual(tr.steps, n0 + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(GOOD)
        n0 = tr.steps
        tr.update(with_(label=at(0.40, -0.25)), FAR)
        self.assertEqual(tr.steps, n0 + 1)

    def test_explicit_reset(self):
        tr, _ = settled(GOOD)
        tr.reset()
        self.assertEqual((tr.steps, tr.bad_steps, tr.arms_unavailable, tr.history), (0, 0, 0, {}))
        self.assertEqual(feed(tr, [GOOD] * (FULL - 1)).score, 0.0)


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
        offline = _stub_dir().parent
        if str(offline) not in sys.path:
            sys.path.insert(0, str(offline))
        cls.task = importlib.import_module("task")
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate

    def sim(self, items):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.task.ITEMS[n], (*st.origin, *st.quat_xyzw), source="sim")
                for n, st in items.items()}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = {o.name: o.mass for o in t.OBJECTS if o.mass is not None}
        self.assertEqual(loose, {"specimen": 0.060, "label": 0.020, "slip": 0.008})
        self.assertTrue(all(m <= 0.30 for m in loose.values()))
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S, t.SCENE), ("end", 1.0, "station"))
        s = t.SHEET
        self.assertAlmostEqual(s.pos[0] - s.size[0] / 2, L.SHEET_X0)
        self.assertAlmostEqual(s.pos[0] + s.size[0] / 2, L.SHEET_X1)
        self.assertAlmostEqual(s.pos[1] - s.size[1] / 2, L.SHEET_Y0)
        self.assertAlmostEqual(s.pos[2] + s.size[2] / 2, L.TOP + L.SHEET_T)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                lo = np.asarray(o.pos) - np.asarray(o.size) / 2
                hi = np.asarray(o.pos) + np.asarray(o.size) / 2
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.36, o.name)
            self.assertGreaterEqual(lo[0], 0.20, o.name)

    def test_item_geometry_matches_the_scorer(self):
        for n, spec in self.task.ITEMS.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose((lo + hi) / 2, L.box_center_local(n), atol=1e-12, err_msg=n)
            np.testing.assert_allclose((hi - lo) / 2, L.box_half(n), atol=1e-12, err_msg=n)
            body = spec.shapes[0]
            np.testing.assert_allclose(body.size, L.BODY_SIZE[n], atol=1e-12)
            for s in spec.shapes[1:]:  # every print lies on the body's top, inside its outline
                c, z = np.asarray(s.center), np.asarray(s.size)
                self.assertTrue(np.all(np.abs(c[:2]) + z[:2] / 2 <= np.asarray(L.BODY_SIZE[n][:2]) / 2 + 1e-12), n)
                self.assertAlmostEqual(c[2] - z[2] / 2, L.BODY_SIZE[n][2], places=12)
            com = spec.center_of_mass()
            self.assertLess(math.hypot(com[0], com[1]), 0.002, n)  # under the grip at its middle
            rest = spec.rest_state()
            self.assertTrue(valid(rest))
            np.testing.assert_allclose(rest.origin[:2], L.START[n][:2], atol=1e-12)
        sp = self.task.SPECIMEN
        roots = [s for s in sp.shapes if tuple(s.color or ()) == L.ROOT_BROWN]
        flowers = [s for s in sp.shapes if tuple(s.color or ()) == L.FLOWER_YELLOW]
        self.assertTrue(roots and flowers)
        self.assertTrue(all(s.center[0] > 0.06 for s in roots))  # roots at local +x
        self.assertTrue(all(s.center[0] < -0.06 for s in flowers))
        vol = float(np.prod(L.BOARD_SIZE))
        self.assertAlmostEqual(L.SPEC_MASS / vol / 1000, 0.43, delta=0.01)  # basswood, g/cm3
        self.assertAlmostEqual(L.LABEL_MASS / float(np.prod(L.LABEL_SIZE)) / 1000, 0.53, delta=0.01)  # pine
        self.assertAlmostEqual(L.SLIP_MASS / float(np.prod(L.SLIP_SIZE)) / 1000, 0.55, delta=0.02)

    def test_offline_state_path(self):
        for n, spec in self.task.ITEMS.items():
            box = SimpleNamespace(pos=np.array(spec.offline_box()[0]), quat_xyzw=quat(L.START[n][2]),
                                  lin_vel=np.zeros(3), ang_vel=np.zeros(3))
            st = spec.offline_state(box)
            self.assertEqual(st.source, "offline")
            np.testing.assert_allclose(st.origin, [L.START[n][0], L.START[n][1], L.TOP], atol=1e-12)

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        good = self.sim(GOOD)
        for _ in range(FULL - 1):
            self.feed_arms(FAR)
            t.success(good)
        self.feed_arms(FAR)
        self.assertTrue(t.success(good))
        self.assertEqual(t.score(good), 1.0)
        self.assertEqual(t.score(good), 1.0)  # score() does not advance it
        self.feed_arms(holding("label", GOOD["label"]))
        self.assertAlmostEqual(t.score(good), 0.4)  # read with the arms of the step at hand
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
                self.task.success(self.sim(GOOD))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "label", "slip", "specimen", "correct", "path_mm",
                          "zone_gap_mm", "arms_unavailable", "bad_steps", "label_pose", "slip_pose",
                          "specimen_pose", "left_tcp", "right_tcp", "openings", "episode", "mechkit_episode"):
                    self.assertIn(k, row)
                for k in ("right_edge_in_mm", "foot_in_mm", "square_err_deg", "tilt_deg", "lowest_mm", "still",
                          "let_go", "zero_reason", "extent_mm"):
                    self.assertIn(k, row["label"])
                for k in ("gap_above_mm", "align_above_mm", "gap_left_mm", "align_left_mm", "overlaps_label",
                          "on_sheet"):
                    self.assertIn(k, row["slip"])
                for k in ("root_err_deg", "margin_mm", "covers", "heading_deg"):
                    self.assertIn(k, row["specimen"])
                self.assertAlmostEqual(row["label"]["right_edge_in_mm"], 7.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(GOOD)))  # no crash
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
            by = {w.label: w for w in wps}
            for prev, w in zip(wps, wps[1:]):
                if w.label.startswith(("grip", "let go")):  # every close and open on a part takes 1.5 s, unmoving
                    self.assertNotEqual(prev.finger, w.finger)
                    self.assertGreaterEqual(w.duration, 1.5)
                    self.assertEqual(tuple(prev.pos), tuple(w.pos))
                    self.assertGreaterEqual(prev.duration, 0.3 if "settle" in prev.label else 1.5)
                if w.label.startswith("slowly"):
                    self.assertGreaterEqual(w.duration, 1.5)
                    self.assertAlmostEqual(prev.pos[2] - w.pos[2], t.APPROACH)
            self.assertTrue(all(-0.047 <= w.finger <= 0.0 for w in wps))
            del by
        # grasp widths 16-80 mm, each commanded 8 mm narrower; open pads clear the part by 7-8 mm a side
        for name, w in (("slip", 0.020), ("label", 0.067), ("specimen", 0.070)):
            op, gr = t.FINGERS[name]
            self.assertTrue(0.016 <= w <= 0.080)
            self.assertAlmostEqual(-2 * gr, w - 0.008)
            self.assertGreaterEqual(-2 * op - w, 0.014 - 1e-9)
            self.assertLessEqual(-2 * op, 0.0939 - 0.005)
        # heights: pad tips 3 mm above the item's underside at the pick, and at the place (1 mm above the sheet)
        self.assertAlmostEqual(t.PICK_Z - 0.013 - L.TOP, 0.003)
        self.assertAlmostEqual(t.PLACE_Z - 0.013 - (L.TOP + L.SHEET_T + t.DROP), 0.003)
        # each grip's yaw within 90 degrees of the item's bearing from its arm's base (joint 6 within reach)
        bases = {"left": (0.0, 0.305), "right": (0.0, -0.305)}
        for side, wps in plan.items():
            bx, by_ = bases[side]
            for w in wps:
                if w.label.startswith(("grip", "let go")):
                    bearing = math.atan2(w.pos[1] - by_, w.pos[0] - bx)
                    d = (w.yaw - bearing + math.pi) % (2 * math.pi) - math.pi
                    self.assertLess(abs(d), math.radians(90.0), (side, w.label))
        # the place points: the targets, LAND_AHEAD past along the carry
        for name in ("slip", "label", "specimen"):
            px, py = t.place_point(name)
            tx, ty = t.TARGET[name]
            self.assertAlmostEqual(math.hypot(px - tx, py - ty), t.LAND_AHEAD, places=9)
            sx, sy, _ = L.START[name]
            self.assertLess(math.hypot(px - sx, py - sy), math.hypot(tx - sx, ty - sy) + t.LAND_AHEAD + 1e-9)
            self.assertGreater(math.hypot(px - sx, py - sy), math.hypot(tx - sx, ty - sy))
        # the item turns with the gripper: the board from yaw 90 to 180 - 1.15 (roots to the foot), the tiles to
        # -1.15, all within their windows
        for name, start_yaw, want in (("specimen", 90.0, 180.0), ("label", 0.0, 0.0), ("slip", 0.0, 0.0)):
            pick, place = t.YAW[name]
            end = start_yaw + math.degrees(place - pick)
            self.assertAlmostEqual((end - want + 180) % 360 - 180, -1.15, places=6)
        # the right arm carries the slip over the sheet only after the left gripper is up from the specimen
        left, right = plan["left"], plan["right"]
        self.assertLess(t._until(left, "straight up"), t._until(right, "hold the slip until the left arm is up"))
        # the label is laid after the slip, fingers along y (yaw ~0): neither pad comes down between them
        self.assertLess(t._until(right, "let go of the slip"), t._until(right, "let go of the label"))
        self.assertLess(abs(t.YAW["label"][1]), math.radians(2.0))
        lx, ly = t.place_point("label")
        slip_x0 = L.SLIP_TARGET[0] - L.SLIP_SIZE[0] / 2
        self.assertGreater(slip_x0 - (lx + 0.0135), 0.010)  # the 27 mm wide fingers stay 10 mm short of the slip
        # demo end: the policy's steps fit the run limit
        end = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess(end * HZ + 2 * HZ, 1200)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_withholds_the_answer(self):
        ins = self.task.INSTRUCTION.lower()
        for s in ("herbarium", "foot", "nearest the arms", "barcode", "pressed specimen", "collection label",
                  "determination slip", "let go", "1:1.5"):
            self.assertIn(s, ins)
        words = set(re.findall(r"[a-z]+", ins))
        for s in ("right", "corner", "above", "left", "roots", "root", "down"):
            self.assertNotIn(s, words)  # ('upright' is not 'right')
        for s in ("bottom right", "lower right", "left of"):
            self.assertNotIn(s, ins)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)

    def test_task_json(self):
        p = HERE / "task.json"
        if not p.exists():
            self.skipTest("no task.json yet")
        spec = json.loads(p.read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(spec[k]), n, k)
        text = json.dumps(spec, ensure_ascii=False).lower() + (HERE / "README.md").read_text().lower()
        for bad in ("\u2014", "\u2013", "generated with"):
            self.assertNotIn(bad, text)
        self.assertFalse(_ident_hits(text))
        self.assertEqual(spec["category"], "Impact on jobs")
        # the numbers task.json states are layout.py's
        setup, scoring = spec["setup"], spec["scoring"]
        for n in L.ITEMS:
            x, y, _ = L.START[n]
            self.assertIn(f"centred at x {x * 100:.1f} cm, y {y * 100:+.1f} cm", setup, n)
        for s_ in ("x 20.5-48.44 cm", "y -9.735 to +9.735 cm", "x 45.9-47.9 cm, y -9.0 to -4.0 cm",
                   "200 x 70 x 10 mm, 60 g", "47 x 67 x 12 mm, 20 g", "20 x 60 x 12 mm, 8 g"):
            self.assertIn(s_, setup)
        for s_ in ("2-12 mm inside", "0-10 mm above", "0-10 mm left", "within 10 mm", "at least 2 mm inside",
                   "within 15 degrees", "within 5 degrees", "within 3 mm", "1.5 mm", "at least 20 mm", "1,200 steps"):
            self.assertIn(s_, scoring)
        self.assertEqual((L.EDGE_IN_MIN, L.EDGE_IN_MAX, L.SLIP_GAP_MAX, L.SLIP_ALIGN, L.SPEC_MARGIN, L.ROOT_DEG,
                          L.SQUARE_DEG, L.FLAT_DEG, L.ON_TABLE, L.STILL_PATH, L.HANDS_OFF),
                         (0.002, 0.012, 0.010, 0.010, 0.002, 15.0, 5.0, 5.0, 0.003, 0.0015, 0.020))
        self.assertTrue(spec["scores"].startswith("TBD after the GPU runs"))



def _ident_hits(text):
    """Team-identifying words found in text. They are kept as sha256 prefixes so this file names nobody."""
    import hashlib
    import re as _re
    banned = {"54622c861585ec4f", "7a5e09c6c81cf86f", "a4cde7cecd46cd77", "b1fc1d0d7b2f62fa", "c70eca6b0f88f44d", "c857d09db23e6822", "d807637b43a9a01c", "da3a1991ce3a0d58", "db6017dfdfe82073", "f0556d0df0fd581f"}
    found = []
    for word in set(_re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())):
        for w in {word, *word.split("-")}:
            if hashlib.sha256(w.encode()).hexdigest()[:16] in banned:
                found.append(w)
    return found


if __name__ == "__main__":
    unittest.main()
