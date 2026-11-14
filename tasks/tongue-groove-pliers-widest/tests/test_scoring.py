"""Offline tests for layout.py, scoring.py, offline_pliers.py and task.py: synthetic pliers states (mechkit
AssemblyStates from the task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v"""

import hashlib
import json
import math
import os
import pathlib
import re
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline stand-in

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_pliers as OP  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
from yam_arena import demo as D  # noqa: E402

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 20 steps
SPEC = T.PLIERS_SPEC
AWAY = {a: rigstate.Arm(np.array([0.28, s * 0.25, L.TOP + 0.12]), np.array([0.28, s * 0.25, L.TOP + 0.107]), 0.64)
        for a, s in (("left", 1.0), ("right", -1.0))}
BASES = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}
G5 = L.groove_centre(5)


# -- states --------------------------------------------------------------------------------------------------------


def pliers(q=0.0, sigma=L.SIGMA0_DEG, origin=None, dz=0.0, r=None, spread_tilt=None):
    """A pliers state: slide ``q`` m, spread ``sigma`` deg, the base frame at ``origin`` (default the start) raised
    ``dz``, turned by ``r`` (default the start heading). ``spread_tilt`` (axis, deg) tilts the rivet half alone about
    its own frame's origin."""
    o = np.array(L.ORIGIN if origin is None else origin, dtype=float)
    o[2] += dz
    rot = mk.yaw_quat(L.SLOT_YAW) if r is None else mk.matrix_to_quat(r)
    joints = {S.SLIP: q, S.SPREAD: L.hinge_of(sigma)}
    st = mk.state_from_poses(SPEC, (*o, *rot), joints=joints, source="offline")
    if spread_tilt is not None:
        links = {k: np.array(v, dtype=float) for k, v in st.links.items()}
        rs = mk.quat_to_matrix(links[S.SPREAD][3:7])
        extra = mk.axis_angle_matrix(np.asarray(spread_tilt[0], dtype=float), math.radians(spread_tilt[1]))
        links[S.SPREAD][3:7] = mk.matrix_to_quat(extra @ rs)
        st = mk.state_from_poses(SPEC, (*o, *rot), joints=joints, link_poses=links, source="offline")
    return {S.PLIERS: st}


def finished(q=G5, sigma=50.0, **kw):
    return pliers(q, sigma, **kw)


def arm_on(point_xy, h_tips=0.003, opening=0.064):
    p = np.array([point_xy[0], point_xy[1], L.TOP + h_tips])
    return rigstate.Arm(p + np.array([0.0, 0.0, 0.013]), p, opening)


def with_arm(side, arm):
    arms = dict(AWAY)
    arms[side] = arm
    return arms


def run(seq, tracker=None):
    tr = tracker or S.Tracker()
    for states, arms in seq:
        tr.update(states, arms)
    return tr


def hold(states, arms, n):
    return [(states, arms)] * n


def slid_then(final, arms_final=AWAY, end_steps=2 * WIN + 2):
    """The joint opened at 80 deg and carried 10 mm along (the 0.2 milestone), then ``final`` for 2 s."""
    return [(pliers(0.010, 80.0), AWAY)] + hold(final, arms_final, end_steps)


# -- the pliers' geometry (the design's numbers) ---------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_masses_recompute(self):
        m = L.masses_g()
        # the slotted handle's full-height part runs from 58 to 184.5 mm along its line: 126.5 mm of a 10.4 x 8 mm
        # steel core (7.85 g/cm3) in a 13 x 12 mm PVC dip (1.25 g/cm3)
        self.assertAlmostEqual(L.HANDLE_U[1] - L.HANDLE_U[0], 126.5, places=9)
        self.assertAlmostEqual(m["slotted handle"], L.handle_mass(126.5), places=9)
        core, outer = 126.5 * 10.4 * 8.0 / 1000.0, 126.5 * 13.0 * 12.0 / 1000.0
        self.assertAlmostEqual(m["slotted handle"], core * 7.85 + (outer - core) * 1.25, places=9)  # 94.13 g
        # the head plate: four 5 mm bars round the 32.4 x 8.4 mm slot in a 60 x 26 mm plate (7.8 behind it, 19.8 ahead
        # of it, 8.8 a side): 6439.2 mm3 of steel, 50.5 g
        self.assertAlmostEqual(5.0 * (7.8 * 26.0 + 19.8 * 26.0 + 2 * 32.4 * 8.8) / 1000.0 * 7.85, 50.548, delta=0.001)
        self.assertAlmostEqual(m["head plate"], 50.5, delta=0.1)
        self.assertAlmostEqual(m["slotted half"], 207.4, delta=0.1)
        self.assertAlmostEqual(m["rivet half"], 178.7, delta=0.1)
        self.assertAlmostEqual(m["slotted half"], L.SLOTTED_MASS * 1000, delta=0.1)
        self.assertAlmostEqual(m["rivet"], L.RIVET_MASS * 1000, delta=0.05)
        self.assertAlmostEqual(m["rivet half"], L.RIVET_HALF_MASS * 1000, delta=0.1)
        # the rivet: an 8 mm pin 10 mm long under a 10 mm head 2 mm high, steel: 3.95 + 1.23 g
        self.assertEqual((L.PIN_D, L.PIN_H, L.CAP_D, L.CAP_H), (8.0, 10.0, 10.0, 2.0))
        self.assertAlmostEqual(m["rivet"], math.pi * (4.0 ** 2 * 10.0 + 5.0 ** 2 * 2.0) / 1000.0 * 7.85, places=9)
        self.assertAlmostEqual(m["rivet"], 5.18, delta=0.01)
        self.assertAlmostEqual(m["total"], 391.2, delta=0.1)
        self.assertLess(abs(m["total"] / (L.CAT_MASS * 1000) - 1.0), 0.01)  # the catalogue's 394.6 g: 0.9 % under

    def test_catalogue_sizes(self):
        self.assertTrue(241.0 <= L.overall_length_mm() <= 246.0, L.overall_length_mm())  # 241.3 in the catalogue
        # along the slotted handle (slot-frame heading 212 deg): its end 184.5 mm back from the rivet, and the outer
        # jaw's far corner (x' 40, y' 44) 40 cos 32 + 44 sin 32 = 57.24 mm ahead of it
        ahead = 40.0 * math.cos(math.radians(32.0)) + 44.0 * math.sin(math.radians(32.0))
        self.assertAlmostEqual(L.overall_length_mm(), 184.5 + ahead, delta=0.01)  # 241.74
        self.assertAlmostEqual(L.overall_length_mm(), 241.7, delta=0.05)
        self.assertAlmostEqual(L.handle_ends_across_mm(), 40.0, delta=1.0)  # 1.57 in handle span
        self.assertAlmostEqual(sum(L.JOINT_STACK), 12.0, places=6)  # 0.47 in joint
        self.assertEqual(L.GROOVES, L.POSITIONS)
        self.assertLessEqual(L.jaw_gap_mm(5), L.CAT_CAPACITY_MM)

    def test_jaw_gaps_at_the_closed_stop(self):
        """The outer jaw's face is the plane x' = 34.5 - 11 / 2 = 29, square to the slot; the inner jaw, closed in
        groove 1, spans x' 23.5 -+ 5.5 = 18 to 29, so its face meets it; each groove moves it 6 mm back along -x'."""
        self.assertAlmostEqual(L.OUTER_JAW[0][0] - L.OUTER_JAW[1][1] / 2, 29.0, places=9)
        self.assertAlmostEqual(L.INNER_JAW_CLOSED[0][0] + L.INNER_JAW_CLOSED[1][1] / 2, 29.0, places=9)
        gaps = [round(L.jaw_gap_mm(n), 6) for n in range(1, 6)]
        self.assertEqual(gaps, [0.0, 6.0, 12.0, 18.0, 24.0])

    def test_slot_runs_from_groove_one_toward_the_handles(self):
        """The real tool: groove 1 (narrowest) at the slot's end nearest the jaws, the rivet sliding back toward the
        handles to widen. The slide axis is the slot frame's -x' (world heading -32 + 180 = 148 deg, toward the robot
        and its left)."""
        self.assertEqual(L.SLIDE_AXIS, "-X")
        self.assertEqual(SPEC.link(S.SLIP).axis, "-X")
        d = L.slot_dir()
        self.assertAlmostEqual(math.degrees(math.atan2(d[1], d[0])) % 360.0, 148.0, places=6)
        self.assertLess(d[0], 0.0)  # toward the robot
        self.assertGreater(d[1], 0.0)  # and its left
        # the slot in the head plate holds the rivet from groove 1 (x' 0) to groove 5 (x' -24) with its radius spare
        self.assertAlmostEqual(L.SLOT_X[1], L.SLOT_W / 2, places=9)
        self.assertAlmostEqual(L.SLOT_X[0], -L.SLIDE_TRAVEL / L.MM - L.SLOT_W / 2, places=9)
        # the rivet as mechkit places it, groove 1 and groove 5 (world)
        for n in (1, 5):
            st = pliers(L.groove_centre(n))[S.PLIERS]
            want = np.array(L.B1) + L.groove_centre(n) * d
            self.assertTrue(np.allclose(st.links[S.SLIP][:2], want, atol=1e-9), n)
        # groove 5 is nearer the handle ends and the robot, farther from the jaws
        j = np.array(L.OUTER_JAW[0][:2]) * L.MM
        ends = (L.rot2(L.SLOTTED_TURN_DEG) @ np.array([L.HANDLE_U[1], 0.0])) * L.MM
        r1, r5 = L.rivet_xy(L.groove_centre(1)), L.rivet_xy(L.groove_centre(5))
        self.assertLess(np.linalg.norm(r1 - j), np.linalg.norm(r5 - j))
        self.assertGreater(np.linalg.norm(r1 - ends), np.linalg.norm(r5 - ends))
        self.assertLess(L.to_world_xy(r5 * 1.0)[0], L.B1[0])  # toward the robot (smaller world x)

    def test_jaw_gap_grows_as_the_rivet_moves_toward_the_handles(self):
        gaps = [L.jaw_gap_mm(n) for n in range(1, 6)]
        self.assertTrue(all(b > a + 5.5 for a, b in zip(gaps, gaps[1:])), gaps)
        # the inner jaw moves with the rivet, along the slide direction, by exactly the slide
        a = [s for s in L.rivet_half_shapes(L.CLOSED_DEG, 0.0) if s[0] == "jaw"][0][1]
        b = [s for s in L.rivet_half_shapes(L.CLOSED_DEG, L.SLIDE_TRAVEL) if s[0] == "jaw"][0][1]
        self.assertTrue(np.allclose(np.subtract(b[:2], a[:2]), L.SLIDE_DIR * L.SLIDE_TRAVEL / L.MM))

    def test_outer_jaw_belongs_to_the_slotted_half(self):
        names = [s[0] for s in L.slotted_shapes()]
        self.assertIn("jaw", names)
        self.assertEqual([s for s in L.slotted_shapes() if s[0] == "jaw"][0][1:], L.OUTER_JAW)
        self.assertEqual(len(SPEC.shapes), len(L.slotted_shapes()))  # the root body (the slotted half) carries it
        # in every groove the outer jaw is farther along x' (toward the jaw tips) than the inner one
        for n in range(1, 6):
            inner = [s for s in L.rivet_half_shapes(L.CLOSED_DEG, L.groove_centre(n)) if s[0] == "jaw"][0][1]
            self.assertGreater(L.OUTER_JAW[0][0], inner[0], n)

    def test_jaws_open_as_the_handles_open(self):
        """Opening the handles turns the inner jaw away from the outer one (the jaw faces never meet)."""
        for n in (1, 5):
            prev = -1.0
            for sigma in (L.CLOSED_DEG, 10.0, 30.0, 60.0, 85.0):
                inner = [s for s in L.rivet_half_shapes(sigma, L.groove_centre(n)) if s[0] == "jaw"][0][1:]
                gap = min(float(np.linalg.norm(p - q)) for p in L._box_corners_2d(*inner)
                          for q in L._box_corners_2d(*L.OUTER_JAW))
                self.assertGreater(gap, prev, (n, sigma))
                prev = gap

    def test_no_interpenetration_at_any_spread_and_slide(self):
        """No box of one half meets a box of the other (same heights, overlapping plan) from 0 to 85 deg and 0 to
        24 mm; only the rib and the ridges (the tongue in its groove) are left out."""
        sigmas = list(np.arange(L.CLOSED_DEG, L.STOP_DEG, 2.5)) + [L.STOP_DEG]
        for sigma in sigmas:
            for q in np.linspace(0.0, L.SLIDE_TRAVEL, 13):
                self.assertEqual(L.interpenetrations(float(sigma), float(q)), [], (sigma, q))
        # the check itself sees an overlap: the inner jaw pushed 1 mm into the outer one
        oj, ij = L.OUTER_JAW, [s for s in L.rivet_half_shapes(L.CLOSED_DEG) if s[0] == "jaw"][0][1:]
        n = np.array([math.cos(math.radians(oj[2] - 90)), math.sin(math.radians(oj[2] - 90))])
        moved = ((ij[0][0] + n[0] * 1.1, ij[0][1] + n[1] * 1.1, ij[0][2]), ij[1], ij[2])
        self.assertTrue(L._boxes_meet(oj, moved))
        self.assertFalse(L._boxes_meet(oj, ij))
        # and the heights: the lap plate (z 6 to 10) over the head plate (z 0 to 5) does not count
        self.assertFalse(L._boxes_meet(((0, 0, 2.5), (10, 10, 5), 0), ((0, 0, 8), (10, 10, 4), 0)))

    def test_rivet_clears_the_slotted_half(self):
        """The rivet's 8 mm pin (z 0 to 10) and its 10 mm head (z 10 to 12) meet no box of the slotted half anywhere
        along the slot: the pin runs in the 8.4 mm slot and passes the ridge chords' ends with 0.15 mm or more to
        spare. ``interpenetrations`` counts the rivet too."""
        boxes = [(name, (c, s, yaw)) for name, c, s, yaw in L.slotted_shapes()]
        self.assertEqual(sum(name.startswith("ridge") for name, _ in boxes), 15)
        for q in np.linspace(0.0, L.SLIDE_TRAVEL, 481):  # every 0.05 mm
            axis = L.rivet_xy(float(q)) / L.MM
            for name, box in boxes:
                self.assertFalse(L._post_meets_box(axis, L.PIN_D / 2 + 0.15, 0.0, L.PIN_H, box), (q, name))
                self.assertFalse(L._post_meets_box(axis, L.CAP_D / 2, L.PIN_H, L.PIN_H + L.CAP_H, box), (q, name))
        # the check itself sees a post in a box: by a face, by a corner, and only at the box's own height
        box = ((5.0, 0.0, 5.5), (4.0, 4.0, 1.0), 0.0)  # its near face 3 mm from the axis
        self.assertTrue(L._post_meets_box((0.0, 0.0), 4.0, 0.0, 10.0, box))
        self.assertFalse(L._post_meets_box((0.0, 0.0), 3.0, 0.0, 10.0, box))
        self.assertFalse(L._post_meets_box((0.0, 0.0), 4.0, 6.5, 10.0, box))  # the post starts above the box
        corner = ((5.0, 5.0, 5.5), (4.0, 4.0, 1.0), 0.0)  # its near corner 4.24 mm from the axis
        self.assertFalse(L._post_meets_box((0.0, 0.0), 4.0, 0.0, 10.0, corner))
        self.assertTrue(L._post_meets_box((0.0, 0.0), 4.5, 0.0, 10.0, corner))
        turned = ((5.0, 0.0, 5.5), (4.0, 4.0, 1.0), 45.0)  # the same box turned 45 deg: a corner 2.17 mm away
        self.assertTrue(L._post_meets_box((0.0, 0.0), 2.5, 0.0, 10.0, turned))
        self.assertFalse(L._post_meets_box((0.0, 0.0), 2.0, 0.0, 10.0, turned))
        # and interpenetrations reports it: a ridge chord put across the slot at the rivet's place
        real = L.ridge_chords
        try:
            L.ridge_chords = lambda: [((0.0, 0.0, 5.5), (6.0, 1.5, 1.0), 90.0)]
            self.assertIn(("ridge0", "pin"), L.interpenetrations(L.SIGMA0_DEG, 0.0))
        finally:
            L.ridge_chords = real
        self.assertEqual(L.interpenetrations(L.SIGMA0_DEG, 0.0), [])

    def test_ridges(self):
        chords = L.ridge_chords()
        self.assertEqual(len(chords), 15)
        total = sum(c[1][0] for c in chords)
        self.assertAlmostEqual(total, 98.1, delta=0.1)
        # the arcs sit on the head plate, ahead of their centres (on the jaw side)
        for c, size, _ in chords:
            self.assertTrue(L.HEAD_X[0] < c[0] < L.HEAD_X[1] and abs(c[1]) < L.HEAD_W / 2)

    def test_release_angle_matches_the_built_rib_and_ridges(self):
        """The rib's across-slot band overlaps the ridges' band below 72 deg and clears it from 72 deg on."""
        for s in (L.CLOSED_DEG, 30.0, 50.0, 60.0, 70.0, 71.8):
            self.assertGreater(L.rib_ridge_overlap_mm(s), 0.0, s)
        for s in (72.1, 75.0, 80.0, 85.0):
            self.assertLess(L.rib_ridge_overlap_mm(s), 0.0, s)
        self.assertAlmostEqual(L.rib_ridge_overlap_mm(72.0), 0.0, delta=0.05)

    def test_grooves_and_bands(self):
        self.assertEqual([L.groove_of(L.groove_centre(n)) for n in range(1, 6)], [1, 2, 3, 4, 5])
        self.assertEqual(L.groove_of(0.0184), 4)
        self.assertIsNone(L.groove_of(0.0196))
        self.assertEqual(L.nearest_groove_target(0.0205), 0.018)
        self.assertEqual(L.nearest_groove_target(0.0295), 0.024)
        # from exactly half way between two grooves, the one nearer the jaws (never round-half-even)
        for q, want in ((0.003, 0.0), (0.009, 0.006), (0.015, 0.012), (0.021, 0.018)):
            self.assertEqual(L.nearest_groove_target(q), want, q)
            self.assertEqual(L.nearest_groove_target(q + 1e-7), round(want + L.PITCH, 9), q)
        self.assertEqual(L.nearest_groove_target(-0.002), 0.0)

    def test_start_pose_and_grips(self):
        self.assertEqual(L.B1, (0.380, 0.000))
        self.assertAlmostEqual(L.SLOTTED_HEADING_DEG, 180.0, places=9)  # -32 + 212: straight back at the robot
        # the slotted handle: 170 mm straight back from the rivet, its centre line 18 mm to the left of the rivet axis
        # (the slotted frame's -y is world +y when the frame points back along -x)
        self.assertTrue(np.allclose(L.slotted_grip(), (0.380 - 0.170, 0.018), atol=1e-12))
        # the rivet half's handle at heading 180 + sigma, 162 mm out, its centre line 9 mm to its frame's +y
        for sigma in (30.0, 80.0):
            h = math.radians(180.0 + sigma)
            want = (0.380 + 0.162 * math.cos(h) - 0.009 * math.sin(h), 0.162 * math.sin(h) + 0.009 * math.cos(h))
            self.assertTrue(np.allclose(L.rivet_grip(sigma), want, atol=1e-12), sigma)
        self.assertTrue(np.allclose(L.rivet_grip(30.0), (0.2442, -0.0888), atol=1e-4))
        self.assertTrue(np.allclose(L.rivet_grip(80.0), (0.3607, -0.1611), atol=1e-4))
        self.assertTrue(np.allclose(L.rivet_grip(80.0, 0.024) - L.rivet_grip(80.0), 0.024 * L.slot_dir(), atol=1e-12))
        self.assertAlmostEqual(L.rivet_grip_yaw_deg(30.0), 30.0)
        self.assertAlmostEqual(L.slotted_grip_yaw_deg(), 0.0)
        # the slotted handle is the left one, the rivet half's the right one
        self.assertGreater(L.slotted_grip()[1], 0.0)
        self.assertLess(L.rivet_grip(30.0)[1], 0.0)
        # the grips sit on the handles, 14.5 mm (slotted) and 12 mm (rivet half) from their ends, more than the
        # offline hold's 10 mm end margin
        self.assertAlmostEqual(L.HANDLE_U[1] - L.GRIP_U_SLOTTED / L.MM, 14.5)
        self.assertAlmostEqual(L.RIVET_HANDLE_R[1] - L.GRIP_U / L.MM, 12.0)
        self.assertTrue(L.HANDLE_U[0] * L.MM + OP.HOLD_END_MARGIN <= L.GRIP_U_SLOTTED
                        <= L.HANDLE_U[1] * L.MM - OP.HOLD_END_MARGIN)
        self.assertTrue(L.RIVET_HANDLE_R[0] * L.MM + OP.HOLD_END_MARGIN <= L.GRIP_U
                        <= L.RIVET_HANDLE_R[1] * L.MM - OP.HOLD_END_MARGIN)
        # and on the handles' centre lines, as the boxes are built
        for name, shapes, u, off in (("slotted", L.slotted_shapes(), L.GRIP_U_SLOTTED, -L.SLOTTED_OFFSET),
                                     ("rivet", L.rivet_half_shapes(L.SIGMA0_DEG), L.GRIP_U, L.RIVET_OFFSET)):
            c, size, yaw = [s[1:] for s in shapes if s[0] == "handle"][0]
            g = L.to_slot_xy(L.slotted_grip() if name == "slotted" else L.rivet_grip(L.SIGMA0_DEG)) / L.MM
            local = L.rot2(-yaw) @ (g - np.array(c[:2]))
            self.assertLess(abs(local[1]), 1e-6, name)  # on the centre line
            self.assertLess(abs(local[0]), size[0] / 2 - OP.HOLD_END_MARGIN / L.MM, name)
            self.assertAlmostEqual(off / L.MM, {"slotted": -18.0, "rivet": 9.0}[name], places=9)

    def test_the_real_tools_layout(self):
        """Facts of the real tool's plan view (a flat photograph of the maker's pliers), checked so that a
        flipped head cannot pass: the slot leans 32 deg off the handle line with its handle end toward the slotted
        half's own handle, the jaws stand ahead of the rivet at 58 deg to the handle line, the closed tool is 40 mm
        across the handles, about 54 mm wide and 241.7 mm long, and the rivet half lies on top."""
        def deg(v):
            return math.degrees(math.atan2(v[1], v[0]))

        # world: the rivet's way from groove 1 to groove 5, as mechkit places the rivet
        p1 = np.array(pliers(0.0)[S.PLIERS].links[S.SLIP][:2])
        p5 = np.array(pliers(L.SLIDE_TRAVEL)[S.PLIERS].links[S.SLIP][:2])
        d = (p5 - p1) / np.linalg.norm(p5 - p1)
        # the slotted handle's centre line, from its box: direction u toward the handle end, w from the rivet square
        # toward the line
        c, size, yaw = [s[1:] for s in L.slotted_shapes() if s[0] == "handle"][0]
        u = np.array([math.cos(math.radians(yaw + L.SLOT_YAW_DEG)), math.sin(math.radians(yaw + L.SLOT_YAW_DEG))])
        rel = L.to_world_xy(np.array(c[:2]) * L.MM) - p1
        self.assertGreater(rel @ u, 0.1)  # the handle lies back along u
        w = rel - (rel @ u) * u
        self.assertAlmostEqual(float(np.linalg.norm(w)), 0.018, places=9)  # the centre line 18 mm off the rivet
        w = w / np.linalg.norm(w)
        self.assertAlmostEqual(deg(u) % 360.0, 180.0, places=9)  # straight back at the robot
        self.assertAlmostEqual(float(d @ w), math.sin(math.radians(32.0)), places=9)  # toward its own handle
        self.assertAlmostEqual(float(d @ u), math.cos(math.radians(32.0)), places=9)  # and back toward the handle end
        lean = math.degrees(math.acos(float(d @ u)))
        self.assertTrue(28.0 < lean < 35.0, lean)
        # closed in groove 1 (slot frame, mm): every jaw corner ahead of the rivet along the handle-forward axis -u
        fwd = -(L.rot2(L.SLOTTED_TURN_DEG) @ np.array([1.0, 0.0]))
        jaws = [s[1:] for s in L.slotted_shapes() + L.rivet_half_shapes(L.CLOSED_DEG, 0.0) if s[0] == "jaw"]
        self.assertEqual(len(jaws), 2)
        for jc, js, jy in jaws:
            for corner in L._box_corners_2d(jc, js, jy):
                self.assertGreater(float(corner @ fwd), 10.0, corner)
            # each jaw face (the box's long side) stands 58 deg to the handle line (45 to 60 on real pliers)
            long_axis = jy if js[0] >= js[1] else jy + 90.0
            between = abs((long_axis - L.SLOTTED_TURN_DEG + 90.0) % 180.0 - 90.0)
            self.assertAlmostEqual(between, 58.0, places=9)
            self.assertTrue(45.0 <= between <= 60.0)
        # the closed tool in groove 1: length along the handles and width square to them, every box corner
        pts = np.array([p for _, bc, bs, by in L.slotted_shapes() + L.rivet_half_shapes(L.CLOSED_DEG, 0.0)
                        for p in L._box_corners_2d(bc, bs, by)])
        side = L.rot2(L.SLOTTED_TURN_DEG) @ np.array([0.0, 1.0])
        length, width = np.ptp(pts @ fwd), np.ptp(pts @ side)
        self.assertTrue(241.0 <= length <= 242.0, length)
        self.assertAlmostEqual(length, L.overall_length_mm(), places=9)
        # the width is the head plate's own (60 x 26 mm, 32 deg off the handle line): 60 sin 32 + 26 cos 32 = 53.84
        self.assertAlmostEqual(width, 60.0 * math.sin(math.radians(32.0)) + 26.0 * math.cos(math.radians(32.0)),
                               places=9)
        # the handles' outer edges: 18 + 6.5 on one side, 9 + 6.5 on the other, 40 mm across closed in groove 1
        # (the maker's 39.9 mm handle span); in groove 5 the rivet half has come 24 sin 32 = 12.7 mm nearer
        handles = np.array([p for nm, bc, bs, by in L.slotted_shapes() + L.rivet_half_shapes(L.CLOSED_DEG, 0.0)
                            if nm == "handle" for p in L._box_corners_2d(bc, bs, by)])
        across = float(np.ptp(handles @ side))
        self.assertAlmostEqual(across, 18.0 + 6.5 + 9.0 + 6.5, places=9)
        self.assertLessEqual(abs(across - L.CAT_HANDLE_SPAN_MM), 0.5)
        self.assertAlmostEqual(L.handle_ends_across_mm(), across, places=9)
        self.assertAlmostEqual(L.handle_ends_across_mm(L.CLOSED_DEG, L.SLIDE_TRAVEL),
                               40.0 - 24.0 * math.sin(math.radians(32.0)), places=9)  # 27.3
        # opening the handles turns the inner jaw away from the outer jaw's face (x' = 29, the inner jaw on x' < 29)
        for n in (1, 5):
            prev = math.inf
            for sigma in np.arange(L.CLOSED_DEG, L.STOP_DEG + 0.1, 5.0):
                ij = [s[1:] for s in L.rivet_half_shapes(float(sigma), L.groove_centre(n)) if s[0] == "jaw"][0]
                front = max(float(p[0]) for p in L._box_corners_2d(*ij))
                self.assertLess(front, prev, (n, sigma))
                self.assertLessEqual(front, 29.0 + 1e-9, (n, sigma))
                prev = front
        # sliding toward the handles widens the jaw gap by exactly the slide
        for q in np.linspace(0.0, L.SLIDE_TRAVEL, 9):
            ij = [s[1:] for s in L.rivet_half_shapes(L.CLOSED_DEG, float(q)) if s[0] == "jaw"][0]
            front = max(float(p[0]) for p in L._box_corners_2d(*ij))
            self.assertAlmostEqual(29.0 - front, q / L.MM, places=9)
        for n in range(1, 6):
            self.assertAlmostEqual(L.jaw_gap_mm(n) - L.jaw_gap_mm(1), L.groove_centre(n) / L.MM, places=9)
        # the rivet half lies on top: its lap plate (z 6 to 10) over the slotted half's head plate (z 0 to 5), and
        # over it in plan at every spread and slide
        lap = [s[1:] for s in L.rivet_half_shapes(L.CLOSED_DEG, 0.0) if s[0] == "lap_plate"][0]
        heads = [s[1:] for s in L.slotted_shapes() if s[0].startswith("head_")]
        self.assertEqual(len(heads), 4)
        for hc, hs, _ in heads:
            self.assertGreaterEqual(lap[0][2] - lap[1][2] / 2, hc[2] + hs[2] / 2)
        for sigma in (L.CLOSED_DEG, L.SIGMA0_DEG, L.STOP_DEG):
            for q in (0.0, L.SLIDE_TRAVEL):
                lc, ls, ly = [s[1:] for s in L.rivet_half_shapes(sigma, q) if s[0] == "lap_plate"][0]
                flat = ((lc[0], lc[1], 2.5), (ls[0], ls[1], 5.0), ly)  # dropped to the head plate's height
                self.assertTrue(any(L._boxes_meet(flat, h) for h in heads), (sigma, q))

    def test_each_half_is_one_piece(self):
        """Every box of a half touches another box of that half, and all of them hang together from the handle: no
        jaw, neck, bridge, arm or ridge floats (touching counts: a shared face or edge, in plan and in height)."""
        def touch(a, b):
            (ca, sa, ya), (cb, sb, yb) = a, b
            if min(ca[2] + sa[2] / 2, cb[2] + sb[2] / 2) - max(ca[2] - sa[2] / 2, cb[2] - sb[2] / 2) < -1e-6:
                return False
            pa, pb = np.array(L._box_corners_2d(ca, sa, ya)), np.array(L._box_corners_2d(cb, sb, yb))
            for yaw in (ya, ya + 90.0, yb, yb + 90.0):  # separating axes of two rectangles
                axis = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
                ra, rb = pa @ axis, pb @ axis
                if min(ra.max(), rb.max()) - max(ra.min(), rb.min()) < -1e-6:
                    return False
            return True

        for sigma in (L.CLOSED_DEG, L.SIGMA0_DEG, L.STOP_DEG):
            for half, shapes in (("slotted", L.slotted_shapes()), ("rivet", L.rivet_half_shapes(sigma))):
                names, boxes = [s[0] for s in shapes], [s[1:] for s in shapes]
                reached, stack = {names.index("handle")}, [names.index("handle")]
                while stack:
                    k = stack.pop()
                    for j in range(len(boxes)):
                        if j not in reached and touch(boxes[k], boxes[j]):
                            reached.add(j)
                            stack.append(j)
                self.assertEqual(sorted(names[j] for j in set(range(len(boxes))) - reached), [], (half, sigma))
        # the rivet's pin (8 mm) and cap (10 mm) sit inside the lap plate's outline (rivet-half frame, rivet at 0, mm)
        (cx, cy, _), (sx, sy, _), yaw = L.LAP_PLATE
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        u, v = -cx * c - cy * s, cx * s - cy * c  # the rivet's axis in the plate's own axes
        self.assertLessEqual(abs(u) + L.CAP_D / 2, sx / 2)
        self.assertLessEqual(abs(v) + L.CAP_D / 2, sy / 2)
        self.assertLess(L.PIN_D, L.CAP_D)

    def test_lies_flat_on_the_bench_at_rest(self):
        st = pliers()[S.PLIERS]
        for body in S.BODIES:
            lo, _ = st.link_bounds(body)
            self.assertAlmostEqual(float(lo[2]), L.TOP, places=9)
        self.assertTrue(np.allclose(st.links[S.SLIP][:3], L.ORIGIN))


class Kit(unittest.TestCase):
    def test_kit_files_are_unchanged_copies(self):
        kit = HERE.parents[1] / "tools" / "kit"
        if not (kit / "mechkit.py").is_file():
            self.skipTest("the kit folder is not next to this task")
        for name in ("mechkit.py", "rigstate.py"):
            a = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            b = hashlib.sha256((kit / name).read_bytes()).hexdigest()
            self.assertEqual(a, b, name)

    def test_joints(self):
        slip, spread = SPEC.link(S.SLIP), SPEC.link(S.SPREAD)
        self.assertEqual((slip.kind, slip.axis, slip.lower, slip.upper), ("prismatic", "-X", 0.0, L.SLIDE_TRAVEL))
        self.assertEqual((slip.spring, slip.damping, slip.friction), (10000.0, 60.0, 1.5))
        self.assertEqual((spread.kind, spread.axis, spread.parent), ("revolute", "Z", S.SLIP))
        self.assertAlmostEqual(L.sigma_deg(spread.lower), 0.0)  # the closed stop: handles parallel
        self.assertAlmostEqual(L.sigma_deg(spread.upper), 85.0)
        self.assertLess(max(abs(spread.lower), abs(spread.upper)), 2 * math.pi)
        self.assertEqual((spread.friction, spread.damping), (0.02, 0.002))
        self.assertAlmostEqual(SPEC.mass + slip.mass + spread.mass, L.TOTAL_MASS, places=9)

    def test_lock_rule(self):
        self.assertEqual(T.lock_target(0.0, 0.0, 30.0), ("locked", 0.0))
        self.assertEqual(T.lock_target(0.0204, 0.0, 71.9), ("locked", 0.018))
        self.assertEqual(T.lock_target(0.024, 0.0, 69.0), ("locked", 0.024))
        mode, tgt = T.lock_target(0.010, 0.0097, 72.0)  # moved 0.3 mm over the last step: led by 0.9 x 0.3 mm
        self.assertEqual(mode, "free")
        self.assertAlmostEqual(tgt, 0.01027, places=12)
        self.assertEqual(L.FOLLOW_GAIN, 0.9)
        self.assertAlmostEqual(T.lock_target(0.010, None, 80.0)[1], 0.010)  # no last step: at the rivet
        self.assertAlmostEqual(T.lock_target(0.010, 0.005, 80.0)[1], 0.011)  # 4.5 mm of lead clipped to 1 mm
        self.assertAlmostEqual(T.lock_target(0.0239, 0.0230, 80.0)[1], 0.024)  # 24.71 mm clipped to the slot
        self.assertAlmostEqual(T.lock_target(0.0003, 0.0010, 80.0)[1], 0.0)  # -0.33 mm clipped to 0
        self.assertAlmostEqual(T.lock_target(0.012, 0.0125, 80.0)[1], 0.01155, places=12)  # backward too

    def test_lock_rule_sets_the_slider_target_every_step(self):
        hub = mk._hub()
        T._groove_lock_reset()
        T._groove_lock(pliers(0.0204, 71.0))  # locked from the start: the tongue stays in groove 1
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.0)
        T._groove_lock(pliers(0.012, 80.0))  # the first free step after a lock leads by nothing, whatever came before
        self.assertEqual(T.LOCK["mode"], "free")
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.012)
        T._groove_lock(pliers(0.0124, 80.0))  # moved 0.4 mm: led by 0.9 x 0.4 mm
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.01276)
        T._groove_lock({S.PLIERS: None})  # unreadable: the last target stays, and the next step leads by nothing
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.01276)
        self.assertIsNone(T.LOCK["q_prev"])
        T._groove_lock(pliers(0.0130, 80.0))
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.0130)
        T._groove_lock(pliers(0.0204, 71.0))  # closes under 72 deg 20.4 mm along: drops into the nearest groove (4)
        self.assertEqual(T.LOCK["mode"], "locked")
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.018)
        T._groove_lock(pliers(0.0236, 71.0))  # still locked: nearest would be groove 5, but it keeps groove 4
        self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.018)
        T._groove_lock_reset()
        self.assertEqual(hub.targets[S.PLIERS][S.SLIP], 0.0)
        self.assertEqual(T.LOCK["mode"], "locked")
        self.assertIsNone(T.LOCK["q_prev"])

    def test_a_lock_pull_is_not_led_on(self):
        """Closing under 72 deg between grooves pulls the rivet into the nearest groove. The pull moves the rivet
        toward the handles, which re-opens the handles a little (the gripper holds the handle), so the lock lets go
        for a step or two. That first free step must not lead the pull on: the rivet ends in the groove that was
        nearest when the handles closed, never one further (before this rule, a close between grooves could run
        on past the nearest groove)."""
        hub = mk._hub()
        T._groove_lock_reset()
        try:
            for q, sigma in ((0.0152, 80.0), (0.0152, 73.0)):
                T._groove_lock(pliers(q, sigma))
            self.assertEqual((T.LOCK["mode"], T.LOCK["target"]), ("free", 0.0152))
            T._groove_lock(pliers(0.0152, 71.9))  # closes under 72 deg: groove 4 (18 mm) is the nearest
            self.assertEqual((T.LOCK["mode"], T.LOCK["target"]), ("locked", 0.018))
            T._groove_lock(pliers(0.0179, 72.3))  # the pull moved it 2.7 mm and re-opened the handles: free again
            self.assertEqual(T.LOCK["mode"], "free")
            self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.0179)  # held where it is, not led to 18.9 mm
            T._groove_lock(pliers(0.0179, 72.1))  # a second free step: the rivet did not move, nothing to lead
            self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.0179)
            T._groove_lock(pliers(0.0179, 71.8))  # the gripper closes on: locked again, the same groove
            self.assertEqual((T.LOCK["mode"], T.LOCK["target"]), ("locked", 0.018))
        finally:
            T._groove_lock_reset()

        # the same in a closed loop with the real rule: the rivet on the slider's spring and damper (240 Hz, 1.5 N of
        # friction), the gripper closing at 5 deg/s, each mm the rivet moves toward the handles re-opening the
        # spread by c deg (swept from 0.20 to 0.40). From anywhere between two grooves it ends in the nearer one.
        def close_from(q0, c, mass):
            dt, sub = 1 / 240, 12
            q, v, target = q0, 0.0, q0
            T._groove_lock_reset()
            T._groove_lock(pliers(q0, 80.0))
            for i in range(int(round(4.0 / dt))):
                sigma = 72.6 - 5.0 * i * dt + c * (q - q0) * 1000.0
                if i % sub == 0:
                    T._groove_lock(pliers(q, sigma))
                    target = hub.targets[S.PLIERS][S.SLIP]
                a = mass / dt + L.LOCK_DAMPING + L.LOCK_STIFFNESS * dt
                b = mass * v / dt + L.LOCK_STIFFNESS * (target - q)
                if abs(v) < 1e-6:
                    push = L.LOCK_STIFFNESS * (target - q)
                    v = 0.0 if abs(push) <= L.SLIDE_FRICTION else (b - math.copysign(L.SLIDE_FRICTION, push)) / a
                else:
                    vn = (b - math.copysign(L.SLIDE_FRICTION, v)) / a
                    v = 0.0 if vn * v < 0 else vn
                q = min(max(q + v * dt, 0.0), L.SLIDE_TRAVEL)
            return q, T.LOCK["mode"], T.LOCK["target"]

        try:
            for q0 in (0.0031, 0.0089, 0.0149, 0.0151, 0.0152, 0.0155, 0.0190, 0.0209, 0.0211):
                want = L.nearest_groove_target(q0)
                for c in (0.20, 0.27, 0.33, 0.40):
                    for mass in (0.18, 0.5, 1.0):  # the rivet and rivet half; more with the hand that holds them
                        q, mode, target = close_from(q0, c, mass)
                        self.assertEqual((mode, target), ("locked", want), (q0, c, mass))
                        self.assertAlmostEqual(q, want, delta=0.0003, msg=(q0, c, mass))
        finally:
            T._groove_lock_reset()

    def test_lock_keeps_its_groove(self):
        """The real ``_groove_lock`` (mechkit's offline hub takes its targets): once locked in groove 5, a slide
        reading 3.1 mm away (20.9 mm, nearer groove 4) while the spread stays under 72 deg leaves the target at
        24 mm. ``lock_target`` alone would move it to groove 4's 18 mm: the stored groove is what holds."""
        hub = mk._hub()
        T._groove_lock_reset()
        try:
            T._groove_lock(pliers(0.0230, 80.0))
            T._groove_lock(pliers(0.0236, 80.0))
            self.assertEqual(T.LOCK["mode"], "free")
            T._groove_lock(pliers(0.0238, 70.0))  # closed under 72 deg: drops into groove 5
            self.assertEqual(T.LOCK["mode"], "locked")
            self.assertEqual(T.LOCK["target"], 0.024)
            switches = T.LOCK["switches"]
            self.assertEqual(T.lock_target(0.0209, 0.0238, 60.0), ("locked", 0.018))
            for q, sigma in ((0.0209, 60.0), (0.0209, 71.9), (0.0209, 30.0)):
                T._groove_lock(pliers(q, sigma))
                self.assertEqual(T.LOCK["mode"], "locked", (q, sigma))
                self.assertEqual(T.LOCK["target"], 0.024, (q, sigma))
                self.assertAlmostEqual(hub.targets[S.PLIERS][S.SLIP], 0.024)
            self.assertEqual(T.LOCK["switches"], switches)
        finally:
            T._groove_lock_reset()

    def test_scored_over_the_end_and_held_one_second(self):
        self.assertEqual((T.SUCCESS_HOLD_S, T.SCORE_OVER), (1.0, "end"))

    def test_an_unpushed_rivet_stops(self):
        """The 1-D follow rule with no force on the rivet: the stiff spring puts it at the target each control step,
        so its travel per step is FOLLOW_GAIN times the last one. With 0.9 a rivet left moving at 0.5 mm per step goes
        0.5 x 0.9 / (1 - 0.9) = 4.5 mm further (under 5 mm) and stops; with 1.0 it keeps its speed and coasts to the
        slot's end (as an earlier build with 1.0 did on the GPU)."""
        def coast(gain):
            old = L.FOLLOW_GAIN
            L.FOLLOW_GAIN = gain
            try:
                q_prev, q = 0.0100, 0.0105
                for _ in range(400):
                    q_prev, q = q, T.lock_target(q, q_prev, 80.0)[1]
                return q
            finally:
                L.FOLLOW_GAIN = old
        self.assertEqual(L.FOLLOW_GAIN, 0.9)
        rest = coast(L.FOLLOW_GAIN)
        self.assertAlmostEqual(rest - 0.0105, 0.0005 * 0.9 / (1 - 0.9), delta=1e-6)
        self.assertLess(rest - 0.0105, 0.005)
        self.assertEqual(L.groove_of(rest), None)  # 15 mm: between grooves 3 and 4, short of the stop
        self.assertAlmostEqual(coast(1.0), L.SLIDE_TRAVEL, places=12)

    def test_free_mode_lets_a_push_reach_the_stop(self):
        """A 1-D check of the free slide (no PhysX): rivet, rivet half and hand as one mass on the slider's implicit
        spring and damper, its target written once per 50 ms control step by the lock rule, 240 Hz physics, Coulomb
        friction, both arms as one spring (damped at 0.7 of critical) to the demo's command (48 mm over 4 s, 2 s
        dwell, eased back 21 mm, each move eased in and out as the scripted policy does). The friction is the
        slide's setting (1.5 N) and half again; the arms' stiffness is set so that they reach that force with the
        command 15 to 22 mm ahead of the rivet, softer arms than the GPU showed (13 Nov: the open joint slid with
        the command about 12 mm ahead, see task.py's note above SLIDE_CMD). With the rule's lead the rivet reaches
        the 24 mm stop within the dwell and stays there in every case; with the target at the rivet (no lead) the
        stiff spring stops it short (a followed target creeps at force / spring per control step)."""
        def ease(x):
            x = min(1.0, max(0.0, x))
            return 3 * x * x - 2 * x ** 3

        def slide(rule, mass, arm_k, fr):
            dt, sub = 1 / 240, 12
            c_arm = 2 * 0.7 * math.sqrt(arm_k * mass)
            k, c = L.LOCK_STIFFNESS, L.LOCK_DAMPING
            q = v = target = 0.0
            q_prev, reached = None, None
            for i in range(int(round((T.SLIDE_S + T.DWELL_S + T.EASE_S + 0.5) / dt))):
                t = i * dt
                if i % sub == 0:
                    target = rule(q, q_prev)
                    q_prev = q
                if t < T.SLIDE_S:
                    cmd = T.SLIDE_CMD * ease(t / T.SLIDE_S)
                elif t < T.SLIDE_S + T.DWELL_S:
                    cmd = T.SLIDE_CMD
                else:
                    cmd = T.SLIDE_CMD - T.EASE_BACK * ease((t - T.SLIDE_S - T.DWELL_S) / T.EASE_S)
                a = mass / dt + c_arm + c + (arm_k + k) * dt
                b = mass * v / dt + arm_k * (cmd - q) + k * (target - q)
                if abs(v) < 1e-6:
                    push = arm_k * (cmd - q) + k * (target - q)
                    v = 0.0 if abs(push) <= fr else (b - math.copysign(fr, push)) / a
                else:
                    vn = (b - math.copysign(fr, v)) / a
                    v = 0.0 if vn * v < 0 else vn
                q += v * dt
                if q >= L.SLIDE_TRAVEL:
                    q, v = L.SLIDE_TRAVEL, min(v, 0.0)
                    reached = t if reached is None else reached
                if q <= 0.0:
                    q, v = 0.0, max(v, 0.0)
            return q, reached

        def lead(q, q_prev):
            return T.lock_target(q, q_prev, 80.0)[1]

        def at_rivet(q, q_prev):
            return T.lock_target(q, None, 80.0)[1]

        self.assertEqual((T.SLIDE_CMD, T.EASE_BACK, T.SLIDE_S, T.DWELL_S, T.EASE_S), (0.048, 0.021, 4.0, 2.0, 1.5))
        for fr in (1.5, 2.25):  # N: the slide's friction setting, and half again
            for give in (0.015, 0.016, 0.019, 0.022):  # m: the lead at which the arms reach it (friction / stiffness)
                for mass in (0.2, 0.5, 1.0, 2.0):  # the hand's share of the moving mass is not known
                    q, reached = slide(lead, mass, fr / give, fr)
                    self.assertAlmostEqual(q, L.SLIDE_TRAVEL, delta=0.0002, msg=(fr, give, mass))
                    self.assertIsNotNone(reached, (fr, give, mass))
                    self.assertLess(reached, T.SLIDE_S + T.DWELL_S, (fr, give, mass))
        self.assertLess(slide(at_rivet, 0.5, 1.5 / 0.016, 1.5)[0], 0.019)
        k_fit = 1.5 / 0.016  # 93.75 N/m: arms that reach 1.5 N with the command 16 mm ahead (the GPU showed 12 mm)
        for mass in (0.2, 0.5, 1.0, 2.0):  # half again the friction on those arms: a 24 mm lead, the whole over-run
            q, reached = slide(lead, mass, k_fit, 2.25)
            self.assertAlmostEqual(q, L.SLIDE_TRAVEL, delta=0.0002, msg=mass)
            self.assertIsNotNone(reached, mass)
            self.assertLess(reached, T.SLIDE_S + T.DWELL_S, mass)
        # half the stiffness is past what 48 mm can do: (48 - 24) mm x 46.9 N/m = 1.1 N, under the 1.5 N friction
        self.assertLess((T.SLIDE_CMD - L.SLIDE_TRAVEL) * k_fit / 2, 1.5)
        self.assertLess(slide(lead, 0.2, k_fit / 2, 1.5)[0], 0.018)

    def test_lock_holds_against_any_arm_push(self):
        """Locked, the rivet leaves its groove's band (1.5 mm) only under 15 N past the slide's friction (10,000 N/m),
        three times the push test's 5 N; half a pitch (3 mm) takes 30 N. Forced travel of a locked joint earns
        nothing in any case: the 0.2 latch needs the joint open (test_milestone_needs_the_joint_open)."""
        self.assertGreaterEqual(L.LOCK_STIFFNESS * L.GROOVE_BAND, 15.0)
        self.assertGreaterEqual(L.LOCK_STIFFNESS * L.LATCH_SLIDE, 30.0)

    def test_one_hand_pull_drags_the_loose_tool(self):
        """The open joint's friction (the 1.5 N setting) is more than the slotted half's bench friction
        (0.4 x 0.2074 kg x 9.81 = 0.81 N) and about the whole tool's (1.54 N): one hand pulling the joint drags the
        loose tool with it. The text says the pull drags the tool, not that one hand can never slide the joint."""
        self.assertGreater(L.SLIDE_FRICTION, L.FRICTION * L.SLOTTED_MASS * 9.81)
        self.assertAlmostEqual(L.FRICTION * L.SLOTTED_MASS * 9.81, 0.81, delta=0.01)
        self.assertAlmostEqual(L.FRICTION * L.TOTAL_MASS * 9.81, 1.54, delta=0.01)


# -- the scorer ----------------------------------------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def test_nothing_scores_zero(self):
        self.assertEqual(run(hold(pliers(), AWAY, 3 * WIN)).last.score, 0.0)

    def test_full_score(self):
        tr = run(slid_then(finished()))
        self.assertEqual(tr.last.score, 1.0)
        self.assertTrue(tr.last.success)

    def test_middle_grooves(self):
        for n in (2, 3, 4):
            self.assertEqual(run(slid_then(finished(L.groove_centre(n)))).last.score, 0.5, n)

    def test_groove_band(self):
        self.assertEqual(run(slid_then(finished(G5 - 0.0014))).last.score, 1.0)
        self.assertEqual(run(slid_then(finished(G5 - 0.0016))).last.score, 0.2)
        self.assertEqual(run(slid_then(finished(0.018 + 0.0016))).last.score, 0.2)  # 19.6 mm: no groove

    def test_lock_edge(self):
        self.assertEqual(run(slid_then(finished(sigma=71.9))).last.score, 1.0)
        self.assertEqual(run(slid_then(finished(sigma=72.1))).last.score, 0.2)  # handles left open: joint free
        self.assertEqual(run(slid_then(finished(sigma=80.0))).last.score, 0.2)

    def test_latched_milestone(self):
        seq = [(pliers(0.0029, 80.0), AWAY)] + hold(pliers(0.0, 50.0), AWAY, 2 * WIN)
        self.assertEqual(run(seq).last.score, 0.0)
        seq = [(pliers(0.0031, 80.0), AWAY)] + hold(pliers(0.0, 50.0), AWAY, 2 * WIN)
        self.assertEqual(run(seq).last.score, 0.2)  # slid, then back to groove 1

    def test_milestone_needs_the_joint_open(self):
        """The 0.2 is for a joint opened past the release angle and slid half a pitch. A rivet forced 3 mm or more
        along a joint that is not open (30 N or more against the lock spring) has released nothing: 0."""
        back = hold(pliers(0.0, 50.0), AWAY, 2 * WIN)
        for q in (0.0031, 0.0045, 0.006, 0.012, G5):
            tr = run(hold(pliers(q, 50.0), AWAY, 5) + back)
            self.assertFalse(tr.slid, q)
            self.assertEqual(tr.last.score, 0.0, q)
        self.assertEqual(run(hold(pliers(0.0035, 71.99), AWAY, 5) + back).last.score, 0.0)
        self.assertEqual(run([(pliers(0.0035, 72.0), AWAY)] + back).last.score, 0.2)  # 72 deg exactly is open
        # opened and closed again in groove 1, and forced only afterwards: still nothing
        seq = hold(pliers(0.0, 80.0), AWAY, 3) + hold(pliers(0.0, 60.0), AWAY, 3) + hold(pliers(0.0035, 60.0), AWAY, 3)
        self.assertEqual(run(seq + back).last.score, 0.0)
        # the travel read on the first step under 72 deg was made while the joint was still free: it counts
        tr = run([(pliers(0.0025, 73.0), AWAY), (pliers(0.0035, 71.5), AWAY)] + back)
        self.assertEqual((tr.slid, tr.slid_at, tr.last.score), (True, 2, 0.2))
        # ... one step later it does not (the lock has taken hold)
        tr = run([(pliers(0.0025, 73.0), AWAY), (pliers(0.0025, 71.5), AWAY), (pliers(0.0035, 71.5), AWAY)] + back)
        self.assertEqual((tr.slid, tr.last.score), (False, 0.0))
        # the lock rule keeps its free mode across an unreadable step, so the open reading carries over it
        for gap in (1, 2, 5):
            seq = [(pliers(0.0025, 73.0), AWAY)] + [({}, AWAY)] * gap + [(pliers(0.0035, 71.5), AWAY)]
            tr = run(seq + back)
            self.assertEqual((tr.slid, tr.last.score), (True, 0.2), gap)
        # a new trial does not inherit it: reset, or the fallback (a jump back to the start pose)
        tr = run([(pliers(0.0025, 73.0), AWAY)])
        tr.reset()
        self.assertFalse(run([(pliers(0.0035, 71.5), AWAY)] + back, tr).slid)
        tr = run([(pliers(0.010, 80.0), AWAY), (pliers(0.0, 30.0), AWAY), (pliers(0.0035, 71.5), AWAY)] + back)
        self.assertEqual((tr.resets, tr.slid), (1, False))
        # the edges together: 3 mm less the tolerance does not count, 72 deg flickering with 71.99 does
        self.assertFalse(run([(pliers(L.LATCH_SLIDE - 2 * L.EPS, 80.0), AWAY)] + back).slid)
        self.assertTrue(run([(pliers(0.0031, 71.99), AWAY), (pliers(0.0031, 72.0), AWAY)] + back).slid)
        self.assertTrue(run([(pliers(0.0, 72.0), AWAY), (pliers(0.0031, 71.99), AWAY)] + back).slid)

    def test_a_joint_locked_past_groove_1_has_its_milestone(self):
        """The lock rule and the scorer agree: the rule picks groove 2 or higher only from more than 3 mm along,
        read on its first locked step after a free one, and that reading sets the 0.2. So a tool locked in groove 2
        to 5 but still held or propped never scores under 0.2. Driven through the real rule and the real tracker,
        with unreadable steps in between."""
        self.assertEqual(L.nearest_groove_target(0.003), 0.0)
        self.assertEqual(L.nearest_groove_target(0.0030001), L.PITCH)
        hub = mk._hub()
        held = with_arm("right", arm_on(L.rivet_grip(50.0, G5)))
        try:
            for q_free, q_close, blind in ((0.0025, 0.0035, 0), (0.0025, 0.0035, 1), (0.0029, 0.0091, 2),
                                           (0.0, 0.0201, 0), (0.0029, 0.0031, 3), (0.0029, 0.003, 0)):
                T._groove_lock_reset()
                tr = S.Tracker()
                steps = [pliers(q_free, 73.0)] + [{}] * blind + [pliers(q_close, 71.5)]
                for st in steps:
                    T._groove_lock(st)
                    tr.update(st, AWAY)
                seat = hub.targets[S.PLIERS][S.SLIP]  # the groove the rule dropped the tongue into
                self.assertEqual((T.LOCK["mode"], seat), ("locked", L.nearest_groove_target(q_close)))
                for _ in range(2 * WIN):  # the spring has pulled the rivet to that groove; the gripper still holds
                    st = pliers(seat, 60.0)
                    T._groove_lock(st)
                    tr.update(st, held)
                want = 0.2 if q_close >= L.LATCH_SLIDE - L.EPS else 0.0
                self.assertEqual(tr.last.score, want, (q_free, q_close, blind))
                if seat > 0.0:
                    self.assertTrue(tr.slid, (q_free, q_close, blind))
        finally:
            T._groove_lock_reset()

    def test_held_or_touched(self):
        g = L.rivet_grip(50.0, G5)
        held = with_arm("right", arm_on(g))
        self.assertEqual(run(slid_then(finished(), held)).last.score, 0.2)
        resting = with_arm("left", arm_on(L.slotted_grip(), opening=0.64))  # open, a pad on the handle
        self.assertEqual(run(slid_then(finished(), resting)).last.score, 0.2)

    def test_release_distance(self):
        g = L.slotted_grip()
        radius = rigstate.zone_radius(0.64)
        for gap, want in ((0.024, 0.2), (0.026, 1.0)):
            h = 0.012 + gap + radius  # straight above the slotted handle's top
            arms = with_arm("left", arm_on(g, h_tips=h - 0.0005, opening=0.64))
            self.assertEqual(run(slid_then(finished(), arms)).last.score, want, gap)

    def test_release_must_hold_for_the_whole_last_second(self):
        g = L.slotted_grip()
        near = with_arm("left", arm_on(g, opening=0.64))
        seq = slid_then(finished()) + hold(finished(), near, 1) + hold(finished(), AWAY, WIN - 1)
        self.assertEqual(run(seq).last.score, 0.2)
        self.assertEqual(run(seq + [(finished(), AWAY)]).last.score, 1.0)

    def test_hovering_or_lifted(self):
        self.assertEqual(run(slid_then(finished(dz=0.0019))).last.score, 1.0)
        self.assertEqual(run(slid_then(finished(dz=0.0021))).last.score, 0.2)
        self.assertEqual(run(slid_then(finished(dz=0.05))).last.score, 0.2)

    def test_flat_thresholds(self):
        ok = {"base": 0.0, S.SPREAD: 0.0}
        self.assertTrue(S.is_flat({"base": 2.9, S.SPREAD: 0.0}, ok))
        self.assertFalse(S.is_flat({"base": 0.0, S.SPREAD: 3.1}, ok))
        self.assertTrue(S.is_flat({"base": 0.0, S.SPREAD: 0.0}, {"base": 0.0019, S.SPREAD: -0.0019}))
        self.assertFalse(S.is_flat({"base": 0.0, S.SPREAD: 0.0}, {"base": 0.0, S.SPREAD: 0.0021}))

    def test_tilted_or_propped(self):
        """The whole tool tipped 3 deg onto one edge (lowest corner on the bench), or the rivet half alone propped
        4 deg, handle end up: not flat."""
        r = mk.axis_angle_matrix(np.array([1.0, 0.0, 0.0]), math.radians(3.0)) @ mk.quat_to_matrix(
            mk.yaw_quat(L.SLOT_YAW))
        low = float(finished(r=r)[S.PLIERS].low[2])
        self.assertEqual(run(slid_then(finished(r=r, dz=L.TOP - low))).last.score, 0.2)
        st = finished(spread_tilt=((0.0, 1.0, 0.0), -4.0))
        self.assertEqual(run(slid_then(st)).last.score, 0.2)

    def test_still(self):
        for dx, want in ((0.0009, 1.0), (0.0011, 0.2)):
            seq = slid_then(finished())
            moving = [(finished(origin=(L.ORIGIN[0] + dx * k / WIN, L.ORIGIN[1], L.TOP)), AWAY) for k in range(WIN + 1)]
            self.assertEqual(run(seq + moving).last.score, want, dx)
        for deg, want in ((0.9, 1.0), (1.1, 0.2)):
            seq = slid_then(finished())
            turning = [(finished(sigma=50.0 + deg * k / WIN), AWAY) for k in range(WIN + 1)]
            self.assertEqual(run(seq + turning).last.score, want, deg)
        # the joint creeps along the slot, inside groove 5's band throughout: the rivet and the rivet half that
        # rides on it move, the slotted half stays put
        for dq, want in ((0.0009, 1.0), (0.0011, 0.2)):
            seq = slid_then(finished(G5))
            creeping = [(finished(G5 - dq * k / WIN), AWAY) for k in range(WIN + 1)]
            self.assertEqual(L.groove_of(G5 - dq), 5)
            self.assertEqual(run(seq + creeping).last.score, want, dq)

    def test_still_slotted_half_turning_about_the_rivet(self):
        """Within the last second the slotted half alone turns about the rivet (in groove 5) by 1.1 deg: the rivet
        and the rivet half stay exactly where they are (the hinge reading turns back by as much), and the slotted
        half's origin, 24 mm from the rivet, moves only 24 x 1.1 deg = 0.46 mm. Not still, so neither 1.0 nor 0.5;
        the 0.9 deg twin is still."""
        def turned(q, sigma, deg):
            yaw = L.SLOT_YAW + math.radians(deg)
            rivet = np.array(L.ORIGIN[:2]) + L.rot2(math.degrees(L.SLOT_YAW)) @ L.rivet_xy(q)
            origin = rivet - L.rot2(math.degrees(yaw)) @ L.rivet_xy(q)
            r = mk.quat_to_matrix(mk.yaw_quat(yaw))
            return pliers(q, sigma - deg, origin=(origin[0], origin[1], L.TOP), r=r)

        for q, sigma, settled in ((G5, 50.0, 1.0), (L.groove_centre(3), 50.0, 0.5)):
            start = turned(q, sigma, 0.0)[S.PLIERS]
            for deg, want in ((0.9, settled), (1.1, 0.2)):
                seq = slid_then(turned(q, sigma, 0.0))
                moving = [(turned(q, sigma, deg * k / WIN), AWAY) for k in range(WIN + 1)]
                end = moving[-1][0][S.PLIERS]
                for body in (S.SLIP, S.SPREAD):  # the rivet's frame turns with the slot it rides in; it stays put
                    (r0, p0), (r1, p1) = start.link_pose(body), end.link_pose(body)
                    self.assertTrue(np.allclose(p1, p0, atol=1e-12), body)
                self.assertLess(S.turn_deg(start.link_pose(S.SPREAD)[0], end.link_pose(S.SPREAD)[0]), 1e-4)
                self.assertAlmostEqual(S.turn_deg(start.link_pose("base")[0], end.link_pose("base")[0]), deg, places=6)
                tr = run(seq + moving)
                m = tr.last.details["motion"]
                self.assertLess(m["move_mm"], 1.0)
                self.assertAlmostEqual(m["turn_deg"], deg, places=3)
                self.assertEqual(tr.last.score, want, (q, deg))
                self.assertEqual(tr.last.details["still_1s"], deg < 1.0)

    def test_flat_either_face_up(self):
        """Both halves turned over together, 180 deg about the handles' axis (world -x), and lowered so their low
        points are on the bench: lying flat the other way up counts as flat (1.0). 4 deg more tilt about that axis,
        lowest corner on the bench, does not."""
        base = mk.quat_to_matrix(mk.yaw_quat(L.SLOT_YAW))
        axis = np.array([-1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(L.rot2(L.SLOTTED_HEADING_DEG) @ np.array([1.0, 0.0]), axis[:2], atol=1e-12))
        for extra, want in ((0.0, 1.0), (4.0, 0.2), (-4.0, 0.2)):
            r = mk.axis_angle_matrix(axis, math.radians(180.0 + extra)) @ base
            low = float(finished(r=r)[S.PLIERS].low[2])
            st = finished(r=r, dz=L.TOP - low)
            for body in S.HALVES:
                rb = st[S.PLIERS].link_pose(body)[0]
                self.assertLess(rb[2, 2], 0.0, body)  # upside down
                self.assertAlmostEqual(S.tilt_deg(rb), abs(extra), places=6)
            tr = run(slid_then(st))
            if extra == 0.0:
                for body in S.HALVES:
                    self.assertLess(abs(tr.last.details["low_mm"][body]), 0.001, body)  # on the bench
            self.assertEqual(tr.last.details["flat"], extra == 0.0, extra)
            self.assertEqual(tr.last.score, want, extra)

    def test_off_the_bench_at_the_end_scores_zero(self):
        """The whole tool tilted handle ends down over the bench edge (about world y; the handles point back along
        -x), every body origin (all near the rivet axis) above -5 cm: the lowest corner decides (the slotted handle's
        end 4.9 or 5.1 cm below, about 16 deg of tilt for its 184.5 mm)."""
        for depth, want in ((0.049, 0.2), (0.051, 0.0)):
            base = mk.quat_to_matrix(mk.yaw_quat(L.SLOT_YAW))
            found = None
            for deg in np.linspace(0.0, 40.0, 4001):
                r = mk.axis_angle_matrix(np.array([0.0, 1.0, 0.0]), -math.radians(deg)) @ base
                st = finished(r=r)[S.PLIERS]
                if L.TOP - float(st.low[2]) >= depth:
                    found = deg
                    break
            self.assertIsNotNone(found, depth)
            self.assertLess(abs(L.TOP - float(st.low[2]) - depth), 0.0005, depth)
            self.assertAlmostEqual(float(st.link_bounds("base")[0][2]), float(st.low[2]), places=9)  # the slotted half
            origins = [float(st.links[b][2]) if b != "base" else float(st.origin[2]) for b in S.BODIES]
            self.assertTrue(all(z > L.TOP - 0.05 for z in origins))
            tr = run(slid_then(finished(r=r)))
            self.assertEqual(tr.last.score, want, depth)

    def test_knocked_off_then_put_back(self):
        """Off the bench is judged at the end: a tool that fell and was laid back flat is scored where it ends."""
        seq = slid_then(finished())
        seq.insert(2, (finished(dz=-0.2), AWAY))
        self.assertEqual(run(seq).last.score, 1.0)

    def test_ends_on_the_floor(self):
        """Slid, then on the floor at the end: 0, not 0.2."""
        seq = slid_then(finished(dz=-0.3))
        self.assertEqual(run(seq).last.score, 0.0)
        self.assertEqual(run(slid_then(finished(dz=-0.04))).last.score, 0.2)  # 4 cm low: not flat, still slid

    def test_dragging_the_whole_tool(self):
        seq = [(pliers(0.0, 30.0, origin=(L.ORIGIN[0] + 0.002 * k, L.ORIGIN[1], L.TOP)), AWAY) for k in range(30)]
        self.assertEqual(run(seq + hold(pliers(0.0, 30.0, origin=(0.44, -0.02, L.TOP)), AWAY, 2 * WIN)).last.score,
                         0.0)

    def test_lifted_adjusted_in_the_air_and_laid_flat(self):
        seq = [(pliers(0.0, 80.0, dz=0.10), AWAY), (pliers(0.024, 80.0, dz=0.10), AWAY)]
        self.assertEqual(run(seq + hold(finished(), AWAY, 2 * WIN + 2)).last.score, 1.0)

    def test_every_listed_score_is_reachable(self):
        seen = {run(slid_then(finished())).last.score, run(slid_then(finished(0.012))).last.score,
                run(slid_then(finished(sigma=80.0))).last.score, run(hold(pliers(), AWAY, 50)).last.score}
        self.assertEqual(seen, {0.0, 0.2, 0.5, 1.0})


class FailClosed(unittest.TestCase):
    def test_unreadable_arms_never_released(self):
        nan = rigstate.Arm(np.array([np.nan, 0, 1.0]), np.array([0.3, 0, 1.0]), 0.5)
        short = rigstate.Arm(np.array([0.3, 0.0]), np.array([0.3, 0, 1.0]), 0.5)
        for arms in (None, {}, {"left": AWAY["left"]}, {"left": AWAY["left"], "right": nan},
                     {"left": AWAY["left"], "right": short}, "garbage",
                     {"left": AWAY["left"], "right": rigstate.Arm(AWAY["right"].tcp, AWAY["right"].tips, np.nan)}):
            self.assertEqual(run(slid_then(finished(), arms)).last.score, 0.2, repr(arms)[:60])

    def test_distance_that_raises_is_not_released(self):
        st = finished()[S.PLIERS]

        class Broken:
            def __getattr__(self, name):
                if name == "distance":
                    raise RuntimeError("no distance")
                return getattr(st, name)

        tr = run(slid_then({S.PLIERS: Broken()}))
        self.assertEqual(tr.last.score, 0.2)

    def test_missing_or_broken_state(self):
        tr = run(slid_then(finished()))
        self.assertEqual(tr.last.score, 1.0)
        tr.update({}, AWAY)
        self.assertEqual(tr.last.score, 0.0)
        self.assertFalse(tr.last.details["valid_state"])
        bad = mk.state_from_poses(SPEC, (np.nan, 0, 0, 0, 0, 0, 1), joints={S.SLIP: G5, S.SPREAD: 0.3})
        tr.update({S.PLIERS: bad}, AWAY)
        self.assertEqual(tr.last.score, 0.0)
        tr.update(finished(), AWAY)  # the windows start again: not still or released yet
        self.assertEqual(tr.last.score, 0.2)
        run(hold(finished(), AWAY, WIN), tr)
        self.assertEqual(tr.last.score, 1.0)

    def test_no_joint_readings(self):
        st = finished()[S.PLIERS]

        class NoJoints:
            joints = {}

            def __getattr__(self, name):
                return getattr(st, name)

        self.assertEqual(run(hold({S.PLIERS: NoJoints()}, AWAY, 50)).last.score, 0.0)


class Resets(unittest.TestCase):
    def test_reset_clears_the_latches(self):
        tr = run(slid_then(finished(dz=-0.2)))
        self.assertTrue(tr.slid and tr.off_bench)
        tr.reset()
        self.assertFalse(tr.slid or tr.off_bench)
        run(hold(pliers(), AWAY, 50), tr)
        self.assertEqual(tr.last.score, 0.0)

    def test_teleport_back_to_the_start_is_a_new_trial(self):
        tr = run(slid_then(finished()))
        self.assertEqual(tr.last.score, 1.0)
        run(hold(pliers(), AWAY, 30), tr)
        self.assertEqual(tr.resets, 1)
        self.assertFalse(tr.slid)
        self.assertEqual(tr.last.score, 0.0)

    def test_a_natural_return_to_the_start_is_not_a_reset(self):
        """Slid 3.5 mm at 80 deg, then eased back and closed to 30 deg in groove 1 step by step: no jump."""
        seq = [(pliers(0.0035 - 0.0005 * k, 80.0 - 7.0 * k), AWAY) for k in range(8)]
        tr = run(seq + hold(pliers(0.0, 30.0), AWAY, 2 * WIN))
        self.assertEqual(tr.resets, 0)
        self.assertEqual(tr.last.score, 0.2)

    def test_task_hooks_reset_on_both_signals(self):
        T._TRACKER.reset()
        T._TRACKER.slid = True
        T._new_episode()
        self.assertFalse(T._TRACKER.slid)
        T.OFFLINE.q = 0.01
        T._groove_lock_reset()
        self.assertEqual(T.OFFLINE.q, 0.0)


# -- the offline kinematics and the demo ----------------------------------------------------------------------------


def arms_at(t, ap):
    arms = {}
    for a in ("left", "right"):
        if a in ap:
            pos, rot, finger, _ = ap[a].target(t)
            tips = np.asarray(pos) + 0.013 * np.asarray(rot)[:, 2]
            arms[a] = rigstate.Arm(np.asarray(pos), tips, float(np.clip(finger / rigstate.FINGER_OPEN, 0, 1)))
        else:
            arms[a] = AWAY[a]
    return arms


def play(p, steps=None):
    """Play waypoint plans through the stub's ArmPlan targets (no IK) into the offline kinematics and a tracker:
    (final score, first step of a 1 s run of 1.0, kinematics, tracker)."""
    ap = {a: D.ArmPlan(list(w), BASES[a]) for a, w in p.items()}
    end = max((x.duration for x in ap.values()), default=0.0)
    n = steps or int(math.floor((end + 3.0) * HZ)) + 1
    kin, tr = OP.OfflinePliers(), S.Tracker()
    ok_run, success_at = 0, None
    for k in range(n):
        arms = arms_at((k + 1) / HZ, ap)
        kin.step(arms)
        o, yaw = kin.pose()
        st = mk.state_from_poses(SPEC, (*o, *mk.yaw_quat(yaw)), joints=kin.joints(), source="offline")
        tr.update({S.PLIERS: st}, arms)
        ok_run = ok_run + 1 if tr.last.score == 1.0 else 0
        if success_at is None and ok_run >= WIN:
            success_at = k + 1
    return tr.last.score, success_at, kin, tr


def plans(variant="", idle=""):
    old = {k: os.environ.get(k) for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
    try:
        os.environ["DEMO_VARIANT"], os.environ["DEMO_IDLE_ARM"] = variant, idle
        os.environ.pop("DEMO_HOLD_ONLY", None)
        return T.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_limit_steps():
    meta = json.loads((HERE / "task.json").read_text())
    return int(re.match(r"([\d,]+) steps", meta["run_limit"]).group(1).replace(",", ""))


class Demo(unittest.TestCase):
    def test_demo_reaches_groove_five(self):
        score, at, kin, tr = play(plans())
        self.assertEqual(score, 1.0)
        self.assertIsNotNone(at)
        self.assertLess(at, run_limit_steps())
        self.assertAlmostEqual(kin.q, G5, delta=1e-9)
        # the close is commanded about a centre 27 mm along the slot (48 - 21), the rivet stops at 24: in the slot
        # frame (mm) the hand ends at (-27, 0) + R(212 + 50) (162, 9) and its bearing from the real rivet (-24, 0)
        # gives the spread: about 1.05 deg under the commanded 50, since the handle points nearly across the slot
        a = math.radians(212.0 + 50.0)
        hand = np.array([-27.0 + 162.0 * math.cos(a) - 9.0 * math.sin(a), 162.0 * math.sin(a) + 9.0 * math.cos(a)])
        rel = hand - np.array([-24.0, 0.0])
        want = math.degrees(math.atan2(rel[1], rel[0]) - math.atan2(9.0, 162.0)) - 212.0
        want = (want + 180.0) % 360.0 - 180.0
        self.assertAlmostEqual(want, 48.95, delta=0.01)
        self.assertAlmostEqual(kin.sigma, want, delta=0.01)
        self.assertLess(kin.sigma, T.END_SIGMA)
        self.assertEqual(kin.towed, 0.0)
        self.assertTrue(tr.slid)

    def test_demo_locks_with_the_right_hand_alone(self):
        """The left has let go and moved aside before the right starts to close. At the end of the close (commanded
        50 deg about a centre 3 mm past the rivet) the spread about the real rivet is under 72 with 15 deg or more to
        spare, the rivet is in groove 5 and the loose slotted half has not been dragged."""
        p = plans()
        right, times = T._right_plan()
        _, _, kin, _ = play(p, steps=int(round((times["slid"] + T.LEFT_AWAY_S) * HZ)))
        self.assertEqual(kin.hold, {"slotted": None, "rivet": "right"})
        self.assertGreater(kin.sigma, L.RELEASE_DEG)  # still open: the close has not started
        self.assertEqual(kin.q, G5)
        _, _, kin, _ = play(p, steps=int(round(times["closed"] * HZ)))
        self.assertLess(kin.sigma, L.RELEASE_DEG - 15.0)
        self.assertEqual(kin.q, G5)
        self.assertEqual(kin.hold, {"slotted": None, "rivet": "right"})
        self.assertEqual(kin.towed, 0.0)

    def test_counterfactuals(self):
        # Offline kinematics only (offline_pliers.py), whose one-hand rule is a simplification (see its docstring).
        self.assertEqual(play(plans(idle="left"))[0], 0.0)  # right arm only, offline: the tool drags
        self.assertEqual(play(plans(idle="right"))[0], 0.0)  # left arm only: nothing moves the rivet half
        score, _, kin, _ = play(plans("naive"))
        self.assertEqual(score, 0.0)
        self.assertEqual(kin.max_q, 0.0)
        self.assertGreater(kin.towed, 0.01)
        score, _, kin, _ = play(plans("no_slide"))
        self.assertEqual(score, 0.0)  # opened and closed again without the slide: still groove 1
        self.assertEqual(kin.max_q, 0.0)
        score, _, kin, _ = play(plans("early_close"))
        # 24 mm commanded against the offline model's 17 mm of arm give: the rivet stops at 24 - 17 = 7 mm (easing
        # back 6 mm stays inside the give) and drops into the nearest groove, groove 2 at 6 mm
        self.assertEqual(score, 0.5)
        self.assertAlmostEqual(kin.max_q, 0.024 - OP.ARM_GIVE, delta=1e-9)
        self.assertEqual(L.nearest_groove_target(0.007), L.groove_centre(2))
        self.assertEqual(L.groove_of(kin.q), 2)
        score, _, kin, tr = play(plans("slide_open"))
        self.assertEqual(score, 0.2)  # slid to groove 5, but let go with the handles still open: not locked
        self.assertEqual(L.groove_of(kin.q), 5)
        self.assertGreater(kin.sigma, L.RELEASE_DEG)
        self.assertEqual(kin.hold, {"slotted": None, "rivet": None})
        self.assertTrue(tr.last.details["released_1s"])
        self.assertEqual(play({})[0], 0.0)

    def test_one_arm_drags_the_tool(self):
        """Offline kinematics only (offline_pliers.py's one-hand rule is a simplification; see its docstring).
        With the slotted half loose the rivet does not slide offline, so the 48 mm slide
        command, made at 80 deg, pulls the rivet half's grip in toward the rivet: in the slot frame (mm) the hand goes
        to (-48, 0) + R(212 + 80) (162, 9), 148.3 mm from the rivet instead of 162.25, and the 13.9 mm past the
        10 mm of give the model allows (TOW_GIVE) drags the whole tool 3.9 mm."""
        _, _, kin, _ = play(plans(idle="left"))
        self.assertEqual(kin.max_q, 0.0)
        a = math.radians(212.0 + 80.0)
        hand = np.array([-48.0 + 162.0 * math.cos(a) - 9.0 * math.sin(a), 162.0 * math.sin(a) + 9.0 * math.cos(a)])
        pulled_in = math.hypot(162.0, 9.0) - float(np.linalg.norm(hand))
        self.assertAlmostEqual(pulled_in, 13.92, delta=0.01)
        self.assertAlmostEqual(kin.towed, (pulled_in - OP.TOW_GIVE / L.MM) * L.MM, delta=0.0001)
        self.assertGreater(kin.towed, 0.003)

    def test_hold_only(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY", None)

    def test_finger_commands(self):
        """Finger targets are minus half the commanded gap: 3 mm commanded on the 13 mm handle (5 mm of squeeze a
        side) and 60 mm open (23.5 mm clear a side); every waypoint of every plan uses one of the two."""
        self.assertEqual((T.GRIP, T.OPEN), (-0.0015, -0.030))
        self.assertAlmostEqual((L.HANDLE_SECTION[0] * L.MM + 2 * T.GRIP) / 2, 0.005, places=12)
        self.assertAlmostEqual((-2 * T.OPEN - L.HANDLE_SECTION[0] * L.MM) / 2, 0.0235, places=12)
        self.assertLess(-2 * T.GRIP, OP.HANDLE_W)  # closes below the handle's width: the offline hold starts
        self.assertGreater(-2 * T.OPEN, OP.RELEASE_GAP)  # and opens past the offline release gap
        for variant in ("",) + T.VARIANTS:
            for side, plan in plans(variant).items():
                closes = [w for w in plan if w.label.startswith("close on the handle")]
                self.assertEqual(len(closes), 1, (variant, side))
                self.assertEqual(closes[0].finger, T.GRIP, (variant, side))
                for w in plan:
                    self.assertIn(w.finger, (T.GRIP, T.OPEN), (variant, side, w.label))
        for side, plan in plans().items():  # the demo's labels state the commanded gap
            self.assertEqual(sum(f"(commanded {-2 * T.GRIP / L.MM:.0f} mm)" in w.label for w in plan), 1, side)
            self.assertEqual(sum("(commanded 3 mm)" in w.label for w in plan), 1, side)
        # after letting go the left gripper moves aside, 95 mm to its left of the slotted grip at its lift height
        self.assertTrue(np.allclose(np.subtract(T.LEFT_PARK, L.slotted_grip()), (0.0, 0.095), atol=1e-12))
        left = plans()["left"]
        aside = [i for i, w in enumerate(left) if w.label.startswith("aside")]
        self.assertEqual(len(aside), 1)
        self.assertTrue(np.allclose(left[aside[0]].pos, T._at(T.LEFT_PARK, T.H_LEFT_LIFT), atol=1e-12))
        self.assertEqual(left[aside[0]].finger, T.OPEN)
        self.assertTrue(all(w.finger == T.OPEN for w in left[aside[0]:]))  # it never closes again

    def test_structure(self):
        """Opens and closes take 1.5 s and happen standing still; the last waypoints after the release hold open;
        both plans fit the run limit with room for the scripted policy's 3 s hold."""
        p = plans()
        for side, plan in p.items():
            prev = None
            for w in plan:
                if prev is not None and w.finger != prev.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))
                    self.assertEqual(tuple(w.pos), tuple(prev.pos), (side, w.label))
                prev = w
            self.assertEqual(plan[-1].finger, T.OPEN)
            self.assertLess(sum(w.duration for w in plan) * HZ + 3 * HZ, run_limit_steps())
        right = p["right"]
        k = max(i for i, w in enumerate(right) if w.finger == T.GRIP)
        self.assertEqual(right[k + 2].finger, T.OPEN)  # a settle after the release before rising
        self.assertEqual(tuple(right[k + 2].pos), tuple(right[k + 1].pos))

    def test_slide_command_covers_the_arms_give(self):
        """Commanded 48 mm for 24 mm of travel: 24 mm over the stop against the 17 mm of arm give the offline model
        allows (offline_pliers.ARM_GIVE, more than the 12 mm the GPU showed), 7 mm to spare; and past the command
        at which the rivet met the slot's end on the GPU by what task.py's note above SLIDE_CMD says (read from that
        note, so the two never disagree). The 17 mm is a design allowance, not a measurement: 2.2 N on the pushing
        arm (the 1.5 setting and the rivet half's 0.4 x 0.1787 kg x 9.81 on the bench) and 0.69 N on the holding
        arm (1.5 less the slotted half's 0.81 N), at an assumed 5.7 mm/N of give each, come to 16.5 mm. Easing
        back 21 mm leaves the hand 3 mm past the rivet at the stop."""
        self.assertEqual((T.SLIDE_CMD, T.EASE_BACK, OP.ARM_GIVE), (0.048, 0.021, 0.017))
        self.assertAlmostEqual(T.SLIDE_CMD - L.SLIDE_TRAVEL - OP.ARM_GIVE, 0.007, places=9)
        self.assertGreater(T.SLIDE_CMD - L.SLIDE_TRAVEL, OP.ARM_GIVE + 0.006)
        note = " ".join(ln.strip().lstrip("#").strip() for ln in (HERE / "task.py").read_text().splitlines()
                        if ln.strip().startswith("#"))
        m = re.search(r"the rivet met the slot's end with the command at (\d+) mm\. (\d+) mm commanded leaves (\d+) mm "
                      r"over that", note)
        self.assertIsNotNone(m)
        met, cmd, over = (int(g) * L.MM for g in m.groups())
        self.assertAlmostEqual(cmd, T.SLIDE_CMD, places=12)
        self.assertAlmostEqual(T.SLIDE_CMD - met, over, places=12)
        self.assertGreaterEqual(over, 0.010 - 1e-9)
        self.assertTrue(0.016 <= OP.ARM_GIVE <= 0.016 + 0.0015)
        push = L.SLIDE_FRICTION + L.FRICTION * L.RIVET_HALF_MASS * 9.81
        hold = L.SLIDE_FRICTION - L.FRICTION * L.SLOTTED_MASS * 9.81
        self.assertAlmostEqual(push, 2.20, delta=0.01)
        self.assertAlmostEqual(hold, 0.69, delta=0.01)
        self.assertGreaterEqual(OP.ARM_GIVE, (push + hold) * 0.0057)
        self.assertLess(OP.ARM_GIVE - (push + hold) * 0.0057, 0.0015)
        self.assertAlmostEqual(T.SLIDE_CMD - T.EASE_BACK - L.SLIDE_TRAVEL, 0.003, places=9)

    @staticmethod
    def _two_hand_clearance(sigma, q=0.0, seeds=None):
        """The arms' capsule clearance (m) with both grippers at their commanded grips, down at the handles and
        closed: the right on the rivet half at spread ``sigma`` with the rivet at ``q``, the left on the slotted one."""
        try:
            import yamkin as K
        except ImportError:
            raise unittest.SkipTest("the arm kinematics module is not next to this task")
        seeds = {} if seeds is None else seeds
        caps = {}
        for a, xy, yaw in (("right", L.rivet_grip(sigma, q), T._yaw_r(sigma)), ("left", L.slotted_grip(), T.YAW_L)):
            j, pe, _ = K.solve(np.asarray(T._at(xy, T.H_GRIP)), K.matrix_to_quat_xyzw(K.grip_rotation(yaw)),
                               seeds.get(a, K.READY), base_pos=BASES[a])
            assert pe < 1e-3, (a, sigma, q, pe)  # reachable
            seeds[a] = j
            caps[a] = K.arm().capsules(j, BASES[a], T.GRIP)
        return K.capsule_clearance(caps["left"], caps["right"])[0]

    def test_second_hand_only_once_the_wrists_clear(self):
        """With both arms down on their grips the arms' capsules are 15 mm apart only once the right has opened the
        handles past about 59 deg (at the 30 deg start they overlap: the grips are 112 mm apart). The left comes down
        only after the right has opened to 80 deg, past that spread, holds through the whole slide, and is open, up
        and aside before the right starts to close (a two-handed close would bring the capsules together again)."""
        seeds = {}
        self.assertLess(self._two_hand_clearance(L.SIGMA0_DEG, 0.0, seeds), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(L.slotted_grip() - L.rivet_grip(L.SIGMA0_DEG))), 0.112, delta=0.001)
        lo, hi = L.SIGMA0_DEG, T.OPEN_TO
        self.assertGreater(self._two_hand_clearance(hi, 0.0, seeds), 0.015)
        while hi - lo > 0.05:  # the clearance grows with the spread over 30 to 80 deg
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self._two_hand_clearance(mid, 0.0, seeds) < 0.015 else (lo, mid)
        clear_from = hi
        self.assertAlmostEqual(clear_from, 59.0, delta=1.0)
        for sigma in np.arange(clear_from, T.OPEN_TO + 0.1, 2.5):
            self.assertGreaterEqual(self._two_hand_clearance(float(sigma), 0.0, seeds), 0.015 - 1e-4, sigma)
        self.assertLess(clear_from, T.OPEN_TO - 15.0)
        # a two-handed close from 80 to 50 about the eased centre (27 mm) would meet the left arm
        self.assertLess(self._two_hand_clearance(T.END_SIGMA, T.SLIDE_CMD - T.EASE_BACK, seeds), 0.0)
        right, times = T._right_plan()
        left = T._left_plan(times["opened"], times["slid"])
        t, down_at, let_go_at, aside_at = 0.0, None, None, None
        for w in left:
            if w.label.startswith("slowly down") and down_at is None:
                down_at = t
            if w.label.startswith("let go"):
                let_go_at = t
            t += w.duration
            if w.label.startswith("aside"):
                aside_at = t
        self.assertGreaterEqual(down_at, times["opened"])
        self.assertAlmostEqual(let_go_at, times["slid"], places=9)
        self.assertAlmostEqual(aside_at, times["slid"] + T.LEFT_AWAY_S, places=9)
        t, close_from = 0.0, None
        for w in right:
            if w.label.startswith("swing the handles in alone") and close_from is None:
                close_from = t
            t += w.duration
        self.assertAlmostEqual(close_from, aside_at, places=9)
        # whenever the left is over the slotted handle and lower than its approach height, the right holds the
        # handles at OPEN_TO, past clear_from (the close starts only once the left is up and aside)
        ap = {a: D.ArmPlan(list(w), BASES[a]) for a, w in (("left", left), ("right", right))}
        low = 0
        for k in range(int(ap["left"].duration * HZ)):
            tt = (k + 1) / HZ
            pos = ap["left"].target(tt)[0]
            if pos[2] < L.TOP + T.H_LEFT_OVER - 1e-6 and np.linalg.norm(pos[:2] - L.slotted_grip()) < 0.03:
                low += 1
                self.assertTrue(times["opened"] <= tt <= close_from, tt)
                self.assertGreater(T.OPEN_TO, clear_from)
        self.assertGreater(low, 0)

    def _clearances(self):
        if not hasattr(Demo, "_clear"):
            Demo._clear = self._measure()
        return Demo._clear

    def test_arm_clearances_along_the_demo(self):
        """Every link but the fingers stays 10 mm or more from the pliers (as the offline kinematics move them), the
        arms' collision capsules never overlap, and no joint comes within 10 deg of a limit, all through the demo."""
        worst_arms, worst_pliers, worst_margin = self._clearances()
        self.assertGreater(worst_arms, 0.003)
        self.assertGreater(worst_pliers, 0.010)
        self.assertGreater(worst_margin, 10.0)

    def test_arm_capsules_keep_15_mm_apart(self):
        """15 mm or more between the arms' capsules all through the demo. The slide brings the right gripper toward
        the left one (about 20 mm between the capsules at the 48 mm command), and a two-handed close would make them
        overlap, which is why the left lets go and moves aside before the right closes."""
        self.assertGreater(self._clearances()[0], 0.015)

    def _measure(self):
        try:
            import yamkin as K
        except ImportError:
            raise unittest.SkipTest("the arm kinematics module is not next to this task")
        p = plans()
        ap = {a: D.ArmPlan(list(w), BASES[a]) for a, w in p.items()}
        kin, seeds = OP.OfflinePliers(), {}
        worst_arms, worst_pliers, worst_margin = math.inf, math.inf, math.inf
        end = max(x.duration for x in ap.values())
        for k in range(int(end * HZ)):
            t = (k + 1) / HZ
            arms = arms_at(t, ap)
            kin.step(arms)
            if k % 5 or t < 1.0:
                continue
            caps = {}
            for a in ap:
                pos, rot, finger, _ = ap[a].target(t)
                q, pe, _ = K.solve(np.asarray(pos), K.matrix_to_quat_xyzw(rot), seeds.get(a, K.READY),
                                   base_pos=BASES[a])
                self.assertLess(pe, 1e-3, (a, t))
                seeds[a] = q
                worst_margin = min(worst_margin, float(np.degrees(K.joint_margins(q)).min()))
                caps[a] = K.arm().capsules(q, BASES[a], finger)
            worst_arms = min(worst_arms, K.capsule_clearance(caps["left"], caps["right"])[0])
            o, yaw = kin.pose()
            st = mk.state_from_poses(SPEC, (*o, *mk.yaw_quat(yaw)), joints=kin.joints())
            for a in caps:
                for link, a0, a1, r in caps[a]:
                    if link.startswith("tip"):
                        continue  # the fingers are meant to be at the handles
                    pts = a0 + np.outer(np.linspace(0.0, 1.0, 30), a1 - a0)
                    worst_pliers = min(worst_pliers, float(st.distance(pts).min()) - r)
        return worst_arms, worst_pliers, worst_margin


class OfflineHooks(unittest.TestCase):
    def test_offline_box_is_the_head_plate_and_round_trips(self):
        c, size, yaw = SPEC.offline_box()
        self.assertAlmostEqual(yaw, L.SLOT_YAW)
        self.assertTrue(np.allclose(size, (0.060, 0.026, 0.012)))  # the head plate: x' -36 to 24 mm, 26 mm wide
        from yam_arena.task_file import ObjectState
        st = SPEC.offline_state(ObjectState(pos=np.array(c), quat_xyzw=mk.yaw_quat(yaw), lin_vel=np.zeros(3),
                                            ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3)))
        self.assertTrue(np.allclose(st.origin, L.ORIGIN, atol=1e-9))

    def test_success_hook_offline_starts_at_zero(self):
        T._TRACKER.reset()
        T.OFFLINE.reset()
        rigstate.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": 0.64} for a in AWAY})
        st = SPEC.rest_state()
        self.assertFalse(T.success({S.PLIERS: st}))
        self.assertEqual(T.score({S.PLIERS: st}), 0.0)


# -- the text ------------------------------------------------------------------------------------------------------


class Text(unittest.TestCase):
    META = json.loads((HERE / "task.json").read_text())
    README = (HERE / "README.md").read_text()

    def test_fields_and_limits(self):
        board = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "run_limit": 60,
                 "fidelity": 490, "reset_notes": 580}
        budget = {"instruction": 300, "context": 550, "setup": 2000, "scoring": 1400, "fidelity": 490,
                  "reset_notes": 350}  # the text budget's hard ceilings, stricter than the board's
        for limits in (board, budget):
            for k, n in limits.items():
                self.assertLessEqual(len(self.META[k]), n, k)
        self.assertEqual(self.META["category"], "Impact on jobs")
        self.assertIn("hand tool adjustment", self.META["keywords"])
        for text in ("1,621,800", "510,600", "2025", "BLS", "home toolbox"):  # the Handbook's job counts, home use
            self.assertIn(text, self.META["context"], text)
        self.assertNotIn("daily", self.META["context"])  # the BLS pages do not say how often
        self.assertNotIn("TK-", self.META["context"])  # no other board task is named in the fields
        self.assertEqual(self.META["instruction"], T.INSTRUCTION)
        self.assertRegex(self.META["scores"], r"^Dream-1 run: ([0-9.]+|@@dream@@); demo: (1\.0|@@demo@@)$")

    def test_no_dashes_no_barred_words(self):
        files = [HERE / "task.json", HERE / "README.md", HERE / "task.py", HERE / "layout.py", HERE / "scoring.py",
                 HERE / "offline_pliers.py", pathlib.Path(__file__)]
        for f in files:
            text = f.read_text()
            self.assertNotIn(chr(0x2014), text, f.name)
            self.assertNotIn(chr(0x2013), text, f.name)
            self.assertTrue(text.isascii(), f.name)
        prose = json.dumps({k: v for k, v in self.META.items() if k != "scores"}) + self.README
        barred = (r"\bscaled?\b", r"\bmodel of\b", r"\banalogue\b", r"\bmock\b", r"\btrainer\b", r"stand-in",
                  r"\bstands? for\b", r"\brepresents\b", r"\bTBD\b", r"to be measured", r"\bbought\b", r"\bbuy",
                  r"measured later", r"\bO\d+\b", r"\bN[0-7]\b", r"\bG\d\b", r"<[^>]*part number[^>]*>")
        for pat in barred:
            self.assertIsNone(re.search(pat, prose, flags=re.IGNORECASE if pat.islower() else 0), pat)

    def test_numbers_agree_with_the_layout(self):
        setup, scoring, context, fid, reset = (self.META[k] for k in ("setup", "scoring", "context", "fidelity",
                                                                      "reset_notes"))
        m = L.masses_g()
        self.assertEqual(f"{round(m['total'])} g", "391 g")
        self.assertIn("9.5 in long, 391 g", setup)
        self.assertNotIn("244 mm", setup)
        self.assertEqual((L.GROOVES, round(L.PITCH / L.MM), round(L.SLIDE_TRAVEL / L.MM)), (5, 6, 24))
        self.assertIn("five arc grooves, 6 mm apart (24 mm of travel)", setup)
        self.assertIn("24 mm", context)
        # the start pose: the rivet at (0.380, 0.000), the slotted handle straight back (heading 180) with its centre
        # line 18 mm to the left of the rivet, the rivet half's handle 30 deg to its right (heading 210, toward -y)
        self.assertEqual((L.B1, L.SIGMA0_DEG, L.SLOTTED_HEADING_DEG), ((0.380, 0.000), 30.0, 180.0))
        self.assertAlmostEqual(L.slotted_grip()[1] - L.B1[1], 0.018, places=12)
        h = L.SLOTTED_HEADING_DEG + L.SIGMA0_DEG
        self.assertLess(math.sin(math.radians(h)), 0.0)  # to the robot's right
        self.assertAlmostEqual(L.CAP_D, 10.0)
        for text in ("(0.380, 0.000)", "straight back", "18 mm left of the rivet", "30 degrees to the right",
                     "30 degrees apart", "10 mm round steel cap", "rivet half, which lies on top"):
            self.assertIn(text, setup, text)
        for text in ("(0.380, 0.000)", "straight back", "30 degrees to its right", "groove 1", "rivet half on top"):
            self.assertIn(text, reset, text)
        # the slot: from the rivet back toward the robot and its left (heading 148), 32 deg off the handle line
        d = L.slot_dir()
        self.assertTrue(d[0] < 0.0 < d[1])
        self.assertAlmostEqual(abs(L.SLOTTED_HEADING_DEG - L.SLIDE_HEADING_DEG), 32.0, places=9)
        self.assertIn("The slot runs from the rivet back toward the robot and its left, 32 degrees off the slotted "
                      "handle's line", setup)
        # the jaws point away and to the left: the outer jaw's long axis, from its root to its tip (away from the
        # rivet), at world heading -32 + 90 = 58 deg
        oc, osz, oyaw = L.OUTER_JAW
        tip = np.array([math.cos(math.radians(oyaw)), math.sin(math.radians(oyaw))])
        self.assertGreater(float(np.array(oc[:2]) @ tip), 0.0)
        jaw_heading = oyaw + L.SLOT_YAW_DEG
        self.assertAlmostEqual(jaw_heading, 58.0, places=9)
        self.assertTrue(math.cos(math.radians(jaw_heading)) > 0.0 and math.sin(math.radians(jaw_heading)) > 0.0)
        self.assertIn("the jaws point away and to the left", setup)
        # the README's tool: the maker's figures and ours
        self.assertEqual((L.CAT_LENGTH_MM, L.CAT_JOINT_MM, L.CAT_HANDLE_SPAN_MM, L.CAT_MASS, L.POSITIONS),
                         (241.3, 11.9, 39.9, 0.3946, 5))
        self.assertIn("Maker's page: 241.3 mm long, 0.47 in joint, 1.57 in (39.9 mm) handle span, 0.87 lb (394.6 g); "
                      "retailers list 5 positions", " ".join(self.README.split()))
        ours = (f"Ours: {round(L.overall_length_mm())} mm long, {round(m['total'])} g, a "
                f"{round(sum(L.JOINT_STACK))} mm joint, handles {round(L.handle_ends_across_mm())} mm across their "
                f"outer edges when closed in groove 1")
        self.assertEqual(ours, "Ours: 242 mm long, 391 g, a 12 mm joint, handles 40 mm across their outer edges when "
                               "closed in groove 1")
        self.assertIn(ours, " ".join(self.README.split()))
        lean = round(abs(L.SLOTTED_HEADING_DEG - L.SLIDE_HEADING_DEG))
        jaw = round(abs((oyaw - L.SLOTTED_TURN_DEG + 90.0) % 180.0 - 90.0))
        self.assertIn(f"The slot leans {lean} degrees off the handle line and the jaws stand ahead of the rivet, "
                      f"{jaw} degrees to that line", " ".join(self.README.split()))
        self.assertEqual((lean, jaw), (32, 58))
        self.assertIn("rivet at (0.380, 0.000) m", " ".join(self.README.split()))
        # the jaw gaps, closed: they touch in groove 1, then 6, 12, 18 and 24 mm
        gaps = [L.jaw_gap_mm(n) for n in range(1, 6)]
        self.assertTrue(np.allclose(gaps, [6.0 * (n - 1) for n in range(1, 6)], atol=1e-9), gaps)
        listed = ", ".join(f"{g:.0f}" for g in gaps[1:4]) + f" and {gaps[4]:.0f} mm apart in grooves 2 to 5"
        self.assertEqual(listed, "6, 12, 18 and 24 mm apart in grooves 2 to 5")
        self.assertIn("the jaws touch in groove 1 and stand " + listed, setup)  # said once, in setup
        # the scoring text reads the groove from the tongue, not from the jaw gap (real jaws meet at the tips first)
        self.assertIn("(real trial: the tongue seated in the nth groove from the jaw end)", scoring)
        self.assertNotIn("mm apart", scoring)
        self.assertEqual((L.RELEASE_DEG, L.STOP_DEG, L.CLOSED_DEG), (72.0, 85.0, 0.0))
        for text in ("72 degrees", "stop at 85"):
            self.assertIn(text, setup, text)
            self.assertIn(text, context, text)
        self.assertEqual((L.GROOVE_BAND, L.FLAT_TILT_DEG, L.FLAT_LOW, L.LET_GO, L.STILL_WINDOW_S, L.STILL_MOVE,
                          L.STILL_TURN_DEG, L.OFF_BENCH, L.LATCH_SLIDE),
                         (0.0015, 3.0, 0.002, 0.025, 1.0, 0.001, 1.0, 0.050, 0.003))
        for text in ("1.5 mm", "72 degrees", "3 degrees", "2 mm", "25 mm", "last 1 s", "1 mm", "1 degree", "5 cm",
                     "3 mm", "groove 5", "groove 2, 3 or 4", "Doing nothing scores 0", "at the end"):
            self.assertIn(text, scoring, text)
        self.assertNotIn("On a real bench", scoring)
        self.assertNotIn("ever 5 cm", scoring)
        # locked, defined once: a 5 N push moves the rivet under 1 mm (the 10,000 N/m spring: 0.5 mm), the slotted
        # half held (loose, the whole tool would slide: 1.54 N of bench friction)
        self.assertEqual(scoring.count("Locked:"), 1)
        self.assertIn("Locked: with the slotted half held, a 5 N push on the rivet half along the slot moves the rivet "
                      "under 1 mm", scoring)
        self.assertLess(5.0 / L.LOCK_STIFFNESS, 0.001)
        self.assertNotIn("per 10 N", scoring)  # the simulated spring is not a property of the real tool
        # the 0.2 latch needs the joint open (scoring.py SLID)
        self.assertIn("0.2: at some time the rivet slid at least 3 mm out of groove 1 with the handles open "
                      "(72 degrees or more).", scoring)
        self.assertEqual(round(L.SLIDE_TRAVEL / L.MM), 24)
        for text in ("0 to 24 mm", "0 to 85 degrees", "20 Hz", "do not collide", "joint friction set to 1.5 N",
                     "nearest groove", "handle spread", "our estimates", "in the groove it dropped into",
                     "slide force"):
            self.assertIn(text, fid, text)
        self.assertNotIn("nearly free", fid)
        self.assertIn("(handles 0 to 85 degrees)", " ".join(self.README.split()))
        self.assertIn("joint friction set to 1.5 N and a light drag", " ".join(self.README.split()))
        self.assertEqual((L.CLOSED_DEG, L.STOP_DEG, round(L.SLIDE_TRAVEL / L.MM)), (0.0, 85.0, 24))
        self.assertEqual(L.CONTROL_HZ, 20.0)
        # the open slide's force, stated only in fidelity (and the README's simplifications): the joint friction
        # setting (no GPU run measured a force)
        self.assertEqual(L.SLIDE_FRICTION, 1.5)
        for k in ("instruction", "context", "setup", "scoring", "reset_notes"):
            self.assertNotIn("1.5 N", self.META[k], k)
        self.assertEqual(run_limit_steps(), 1200)

    def test_directions_are_true(self):
        """Groove 1 (narrowest) is the slot's end nearest the jaws, groove 5 (widest) the end nearest the handles;
        no text says otherwise."""
        texts = json.dumps(self.META) + self.README
        self.assertIn("Groove 1, at the slot's end nearest the jaws, is the narrowest", self.META["setup"])
        self.assertIn("groove 5, at its end nearest the handles, the widest", self.META["setup"])
        self.assertIn("slot's end nearest the jaws (groove 1, the narrowest)", self.META["reset_notes"])
        self.assertIn("slot's end nearest the jaws (groove 1, the narrowest)", self.README)
        for stale in ("nearest the handles (groove 1" + ")", "toward" + " the jaws",
                      "end of the slot nearest the handles"):
            self.assertNotIn(stale, texts, stale)
        self.assertGreater(L.jaw_gap_mm(5), L.jaw_gap_mm(1))

    def test_scoring_text_lists_exactly_the_tiers(self):
        """Every score the scorer can give appears in the scoring text as a tier, and no other."""
        scoring = self.META["scoring"]
        levels = {S.tier(ob, lk, g, True, True, True, sl) for ob in (False, True) for lk in (False, True)
                  for g in (None, 1, 2, 3, 4, 5) for sl in (False, True)}
        self.assertEqual(levels, {0.0, 0.2, 0.5, 1.0})
        for v in ("1.0:", "0.5:", "0.2:", "0:"):
            self.assertIn(v, scoring, v)
        self.assertEqual(sorted(set(re.findall(r"\b(\d\.\d):", scoring))), ["0.2", "0.5", "1.0"])
        self.assertEqual(S.tier(False, True, 5, True, True, True, True), 1.0)
        self.assertEqual(S.tier(False, True, 3, True, True, True, True), 0.5)
        self.assertEqual(S.tier(False, False, 5, True, True, True, True), 0.2)
        self.assertEqual(S.tier(True, True, 5, True, True, True, True), 0.0)

    def test_readme(self):
        self.assertLessEqual(len(self.README), 8000)
        flat = " ".join(self.README.split())
        for text in ("Third-party assets: none (the photograph was measured, not copied)", "CC BY-SA 3.0",
                     "proportions only", "a 5 N push gauge", "hold the slotted handle",
                     "slides against joint friction set to 1.5 N", "the slide force",
                     "| Locked | slotted half held, a 5 N push on the rivet half along the slot moves the rivet "
                     "under 1 mm",
                     "Open loop", "`tests/`", "python3 -m unittest discover -s tests",
                     "DEMO_IDLE_ARM", "DEMO_VARIANT", "DEMO_HOLD_ONLY", "TASK_TRACE", "US 7,017,458",
                     "## Results in simulation", "steadies the slotted half", "## The tool"):
            self.assertIn(text, self.README, text)
        for variant in T.VARIANTS:
            self.assertIn(f"DEMO_VARIANT={variant}", self.README, variant)
        for text in ("two arms are needed", "needs two", "check_task", "tools/", "in this simulation",
                     "offline prediction", "two-handed", "step 843", "57.8 degrees", "1.5 in capacity",
                     "Maker's catalogue", "nearly free", "sized to", "jaw gap", "1 mm per 10 N"):
            self.assertNotIn(text, flat, text)
        self.assertEqual(flat.count("1.5 N"), 1)  # once, in the simplifications
        self.assertEqual([f"{L.jaw_gap_mm(n):.0f}" for n in range(2, 6)], ["6", "12", "18", "24"])

    def test_results_table(self):
        """Each run's score cell is its token until the GPU runs of this build fill it, then a score the scorer can
        give; either state passes, a row never needs both. Filled cells agree with task.json's scores line."""
        section = self.README.split("## Results in simulation", 1)[1].split("\n## ", 1)[0]
        rows = {"nothing": "Nothing (`DEMO_HOLD_ONLY=1`", "demo": "Scripted demo",
                "right_only": "Right arm alone (`DEMO_IDLE_ARM=left`)",
                "left_only": "Left arm alone (`DEMO_IDLE_ARM=right`)",
                "dream": "Dream-1", "naive": "(`DEMO_VARIANT=naive`)", "no_slide": "(`DEMO_VARIANT=no_slide`)",
                "early_close": "(`DEMO_VARIANT=early_close`)", "slide_open": "(`DEMO_VARIANT=slide_open`)"}
        lines = [ln for ln in section.splitlines() if ln.startswith("|") and not ln.startswith("|---")][1:]
        self.assertEqual(len(lines), len(rows))
        cells = {}
        for token, head in rows.items():
            mine = [ln for ln in lines if head in ln.split("|")[1]]
            self.assertEqual(len(mine), 1, token)
            cell = mine[0].rstrip().rstrip("|").split("|")[-1].strip()
            self.assertRegex(cell, rf"^(@@{token}@@|0|0\.0|0\.2|0\.5|1\.0)$", token)
            self.assertLessEqual(self.README.count(f"@@{token}@@"), 1, token)
            cells[token] = cell
        if not cells["demo"].startswith("@@"):
            self.assertEqual(cells["demo"], "1.0")  # only a demo that scores 1.0 is submitted
        if not cells["nothing"].startswith("@@"):
            self.assertEqual(float(cells["nothing"]), 0.0)  # doing nothing scores 0
        m = re.match(r"^Dream-1 run: ([0-9.]+|@@dream@@); demo: (1\.0|@@demo@@)$", self.META["scores"])
        self.assertIsNotNone(m)
        for got, token in ((m.group(1), "dream"), (m.group(2), "demo")):
            if not got.startswith("@@") and not cells[token].startswith("@@"):
                self.assertEqual(float(got), float(cells[token]), token)
        # the runs as described: the run limit, the counterfactuals' parameters (task.py's plans)
        self.assertIn("run limit 1,200 steps (60 s)", section)
        self.assertEqual(run_limit_steps(), 1200)
        naive_push = float(np.linalg.norm(T._naive_plan()[5].pos[:2] - np.asarray(T._at(L.rivet_grip(30.0), 0))[:2]))
        self.assertAlmostEqual(naive_push, 0.045, places=9)
        self.assertIn("pushes it 45 mm along the slot", section)
        early = T.plans("early_close")["right"]
        carried = [w for w in early if w.label.startswith("carry the rivet")][0]
        eased = [w for w in early if w.label.startswith("ease the push")][0]
        self.assertIn("commanded 24 mm", carried.label)
        self.assertIn("back 6 mm", eased.label)
        self.assertIn(f"commanded 24 mm, not {T.SLIDE_CMD / L.MM:.0f}, and eased back 6 mm, not "
                      f"{T.EASE_BACK / L.MM:.0f}", section)
        self.assertIn(f"opened to {T.OPEN_TO:.0f} degrees and closed again", section)
        no_slide = T.plans("no_slide")["right"]
        self.assertIn("commanded 0 mm", [w for w in no_slide if w.label.startswith("carry")][0].label)
        self.assertIn("still past 72 degrees", section)

    def test_demo_table_follows_the_plan(self):
        """The README's demo table: every time and figure recomputed from task.py's plans."""
        flat = " ".join(self.README.split())
        right, times = T._right_plan()
        left = T._left_plan(times["opened"], times["slid"])

        def starts(plan, prefix):
            t = 0.0
            for w in plan:
                if w.label.startswith(prefix):
                    return t
                t += w.duration
            raise AssertionError(prefix)

        grasped = starts(right, "swing the handles open")
        self.assertAlmostEqual(grasped, 6.5, places=9)  # 2.5 over, 1.5 down, 0.5, 1.5 close, 0.5
        self.assertAlmostEqual(times["opened"], 6.5 + (T.OPEN_TO - L.SIGMA0_DEG) / T.OPEN_SPEED, places=9)  # 11.5
        slide_from = starts(right, "carry the rivet")
        self.assertAlmostEqual(slide_from, times["opened"] + T.LEFT_DOWN_S, places=9)  # 18.0
        self.assertAlmostEqual(starts(left, "settle the grip") + 0.5, slide_from, places=9)  # the left has closed
        self.assertAlmostEqual(times["slid"], slide_from + T.SLIDE_S + T.DWELL_S + T.EASE_S + 0.5, places=9)  # 26.0
        close_from = starts(right, "swing the handles in alone")
        self.assertAlmostEqual(close_from, times["slid"] + T.LEFT_AWAY_S, places=9)  # 31.0
        self.assertAlmostEqual(starts(left, "home"), close_from, places=9)
        self.assertAlmostEqual(times["closed"], close_from + (T.OPEN_TO - T.END_SIGMA) / T.CLOSE_SPEED, places=9)
        end = sum(w.duration for w in right)
        self.assertAlmostEqual(end, times["closed"] + 6.5, places=9)  # 43.5
        self.assertLessEqual(sum(w.duration for w in left), times["closed"])
        got = [round(v, 1) for v in (grasped, times["opened"], slide_from, times["slid"], close_from,
                                      times["closed"], end)]
        self.assertEqual(got, [6.5, 11.5, 18.0, 26.0, 31.0, 37.0, 43.5])
        rows = [f"| 0 to {got[0]} |", f"| {got[0]} to {got[1]} |", f"| {got[1]} to {got[2]} |",
                f"| {got[2]} to {got[3]} |", f"| {got[3]} to {got[4]} |", f"| {got[4]} to {got[5]} |",
                f"| {got[5]} to {got[6]} |"]
        for row in rows:
            self.assertEqual(self.README.count(row), 1, row)
        self.assertIn(f"swing the handles open about the rivet, {L.SIGMA0_DEG:.0f} to {T.OPEN_TO:.0f} degrees", flat)
        self.assertIn(f"hold at {T.OPEN_TO:.0f} degrees", flat)
        self.assertIn(f"{T.SLIDE_CMD / L.MM:.0f} mm commanded for {L.SLIDE_TRAVEL / L.MM:.0f} mm of travel", flat)
        self.assertIn(f"dwell {T.DWELL_S:.0f} s, ease back {T.EASE_BACK / L.MM:.0f} mm", flat)
        self.assertIn(f"swing the handles in to {T.END_SIGMA:.0f} degrees: the tongue drops into groove 5 at "
                      f"{L.RELEASE_DEG:.0f}", flat)
        self.assertIn(f"the right {L.GRIP_U / L.MM:.0f} mm out from the rivet on the rivet half's handle, the left "
                      f"{L.GRIP_U_SLOTTED / L.MM:.0f} mm along the slotted one", flat)
        self.assertIn(f"from a grip {L.GRIP_U / L.MM:.0f} mm out", flat)
        self.assertEqual((L.GRIP_U, L.GRIP_U_SLOTTED), (0.162, 0.170))


if __name__ == "__main__":
    unittest.main()
