"""Offline tests for scoring.py and task.py: synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

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
from scoring import (  # noqa: E402
    Tracker,
    beam_angles,
    beam_gap,
    bearing,
    on_pier,
    overhang,
    packer_size,
    pier_face,
    rotation,
    thickness_tilt_deg,
    valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ)) + 1
TOP = L.TOP
D = L.TOP + L.DATUM


# -- building states --------------------------------------------------------------------------------------------


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def mat_to_quat(r):
    """Rotation matrix to a unit quaternion (x, y, z, w)."""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def box(size, x, y, low_z, yaw=0.0, rx=0.0, ry=0.0, centre_z=None):
    """A box's state as the simulator reports it: centre (x, y), lowest point at low_z (or centre at centre_z),
    turned R = Rz(yaw) Rx(rx) Ry(ry) (degrees), with the world-aligned bounding box of its eight corners."""
    r = _rz(math.radians(yaw)) @ _rx(math.radians(rx)) @ _ry(math.radians(ry))
    half = np.asarray(size, dtype=float) / 2
    corners = np.array([r @ (half * np.array(sg)) for sg in itertools.product((-1, 1), repeat=3)])
    cz = centre_z if centre_z is not None else low_z - corners[:, 2].min()
    c = np.array([x, y, cz])
    return SimpleNamespace(pos=c, quat_xyzw=mat_to_quat(r), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=c + corners.min(axis=0), high=c + corners.max(axis=0))


def packer(t, side, x, y, low_z, **kw):
    return box(packer_size(L.packer_name(t, side)), x, y, low_z, **kw)


def beam(x, y, low_z, **kw):
    return box(L.BEAM_SIZE, x, y, low_z, **kw)


def start():
    """The starting layout: every packer flat on its mark, the beam flat on the midline."""
    out = {}
    for side in L.SIDES:
        for t in L.PACKERS:
            out[L.packer_name(t, side)] = packer(t, side, L.SET_X[t], L.SIGN[side] * L.SET_Y, TOP)
    out["beam"] = beam(L.BEAM_X, L.BEAM_Y, TOP)
    return out


def on(side, dx=0.0, dy=0.0):
    """(x, y) over a pier's centre, offset."""
    return L.PIER_X + dx, L.SIGN[side] * L.PIER_Y + dy


def pack_left(objs, z_err=0.0):
    """Yellow 20 on the left pier (top 40 mm) and blue 30 on it (70 mm)."""
    x, y = on("left")
    objs["packer_20_left"] = packer(20, "left", x, y, TOP + 0.020)
    objs["packer_30_left"] = packer(30, "left", x, y, TOP + 0.040 + z_err)
    return objs


def pack_right(objs):
    """Green 40 on the right pier (top 70 mm)."""
    x, y = on("right")
    objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.030)
    return objs


def packed():
    return pack_right(pack_left(start()))


def seated(**kw):
    objs = packed()
    objs["beam"] = beam(L.PIER_X, 0.0, D, **kw)
    return objs


def beam_line(point, s_point, theta, yaw=0.0):
    """A beam turned R = Rz(yaw) Rx(theta) (degrees: theta tilts its length, positive = its +y end up) whose
    underside's centre line passes through the world point `point` at `s_point` along its length from its middle."""
    r = _rz(math.radians(yaw)) @ _rx(math.radians(theta))
    u = np.asarray(point, dtype=float) - s_point * r[:, 1]
    c = u + r[:, 2] * L.BEAM_SIZE[2] / 2
    return box(L.BEAM_SIZE, c[0], c[1], None, yaw=yaw, rx=theta, centre_z=c[2])


def tilted_beam(y1, z1, y2, z2, x=L.PIER_X, yc=0.0):
    """A beam lying across two contacts along y: its underside's centre line passes through (y1, z1) and (y2, z2)
    (heights above the table); its middle at y yc."""
    if y2 < y1:
        y1, z1, y2, z2 = y2, z2, y1, z1
    th = math.atan2(z2 - z1, y2 - y1)
    return beam_line((x, y1, TOP + z1), (y1 - yc) / math.cos(th), math.degrees(th))


def run(objs_seq, t=None, n=1):
    """Feed each state dict n times to a tracker (a fresh one unless given); return the last result and the
    tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for objs in objs_seq:
        for _ in range(n):
            r = t.update(objs)
    return r, t


def settle(objs, t=None):
    """The same state long enough to be still (a full 0.5 s window and then some)."""
    return run([objs], t, n=WINDOW + 2)


def glide(a, b, name, n):
    """n state dicts in which object `name` moves in equal steps from its pose in a to its pose in b (centre only)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        objs = dict(a)
        sa, sb = a[name], b[name]
        p = (1 - f) * sa.pos + f * sb.pos
        dlo, dhi = sb.low - sb.pos, sb.high - sb.pos
        objs[name] = SimpleNamespace(pos=p, quat_xyzw=sb.quat_xyzw, lin_vel=(sb.pos - sa.pos) * HZ / n,
                                     ang_vel=np.zeros(3), low=p + dlo, high=p + dhi)
        out.append(objs)
    return out


# -- tests ------------------------------------------------------------------------------------------------------


class Arithmetic(unittest.TestCase):
    def test_one_correct_packing_each_side(self):
        """From one set {20, 30, 40}: only 20 + 30 makes the left's 50 mm and only 40 the right's 40 mm."""
        sets = [c for k in (1, 2, 3) for c in itertools.combinations((20, 30, 40), k)]
        left = [c for c in sets if sum(c) == round((L.DATUM - L.PIER_H["left"]) * 1000)]
        right = [c for c in sets if sum(c) == round((L.DATUM - L.PIER_H["right"]) * 1000)]
        self.assertEqual(left, [(20, 30)])
        self.assertEqual(right, [(40,)])
        self.assertEqual(L.SOLUTION, {"left": (20, 30), "right": (40,)})

    def test_beam_cannot_span_the_datum_posts(self):
        inner = 2 * (L.POST_Y - L.POST_SIDE / 2)
        self.assertGreater(inner, L.BEAM_LEN + 0.010)

    def test_centred_beam_bears_50_mm_each_end(self):
        objs = seated()
        for side in L.SIDES:
            from scoring import face
            f = face(objs[L.packer_name(30 if side == "left" else 40, side)], packer_size("packer_40_left"), True)
            over, crosses = bearing(objs["beam"], f)
            self.assertAlmostEqual(over, 0.050, places=4)
            self.assertTrue(crosses)


class Geometry(unittest.TestCase):
    def test_tilt_either_face_down(self):
        self.assertAlmostEqual(thickness_tilt_deg(packer(20, "left", 0.3, 0.3, TOP)), 0.0, places=6)
        self.assertAlmostEqual(thickness_tilt_deg(packer(20, "left", 0.3, 0.3, TOP, rx=180)), 0.0, places=6)
        self.assertAlmostEqual(thickness_tilt_deg(packer(20, "left", 0.3, 0.3, TOP, rx=90)), 90.0, places=4)
        self.assertAlmostEqual(thickness_tilt_deg(packer(20, "left", 0.3, 0.3, TOP, ry=10)), 10.0, places=4)

    def test_on_pier_needs_centre_over_it_and_off_the_table(self):
        x, y = on("left")
        self.assertTrue(on_pier(packer(20, "left", x, y, TOP + 0.020), "left"))
        self.assertTrue(on_pier(packer(20, "left", x + 0.044, y, TOP + 0.020), "left"))
        self.assertFalse(on_pier(packer(20, "left", x + 0.046, y, TOP + 0.020), "left"))
        self.assertFalse(on_pier(packer(20, "left", x + 0.040, y, TOP), "left"))  # on the table beside it
        self.assertFalse(on_pier(packer(20, "left", x, y, TOP + 0.020), "right"))

    def test_overhang_uses_the_turned_outline(self):
        x, y = on("right")
        self.assertAlmostEqual(overhang(packer(40, "right", x, y, TOP + 0.03), packer_size("packer_40_right"),
                                        "right"), 0.0)
        self.assertAlmostEqual(overhang(packer(40, "right", x + 0.010, y, TOP + 0.03),
                                        packer_size("packer_40_right"), "right"), 0.005, places=6)
        turned = overhang(packer(40, "right", x, y, TOP + 0.03, yaw=45), packer_size("packer_40_right"), "right")
        self.assertAlmostEqual(turned, 0.025 * math.sqrt(2) - 0.030, places=5)  # 5.4 mm: just over 5

    def test_beam_angles(self):
        a = beam_angles(beam(0.29, 0.0, D))
        self.assertAlmostEqual(a["level_deg"], 0.0, places=5)
        self.assertAlmostEqual(a["face_deg"], 0.0, places=5)
        self.assertAlmostEqual(a["heading_deg"], 0.0, places=5)
        self.assertAlmostEqual(beam_angles(beam(0.29, 0.0, D, rx=2.0))["level_deg"], 2.0, places=4)
        self.assertAlmostEqual(beam_angles(beam(0.29, 0.0, D, ry=4.0))["face_deg"], 4.0, places=4)
        self.assertAlmostEqual(beam_angles(beam(0.29, 0.0, D, ry=90.0))["face_deg"], 0.0, places=4)  # on edge
        self.assertAlmostEqual(beam_angles(beam(0.29, 0.0, D, yaw=12.0))["heading_deg"], 12.0, places=4)
        self.assertAlmostEqual(beam_angles(beam(0.29, 0.0, D, yaw=180.0))["heading_deg"], 0.0, places=4)

    def test_beam_gap_finds_the_contact_edge(self):
        """Tilted 7.8 deg across one packed and one bare pier, the beam touches the bare pier at an edge: the gap
        there is about 0 although its underside is centimetres above the middle of the pier."""
        b = tilted_beam(-0.155, 0.070, 0.205, 0.020)
        g = beam_gap(b, pier_face("left"))
        self.assertIsNotNone(g)
        self.assertLess(abs(g), 0.0015)
        self.assertIsNone(beam_gap(beam(0.15, 0.0, TOP), pier_face("left")))  # not over it at all

    def test_valid(self):
        self.assertTrue(valid(beam(0.29, 0.0, D)))
        bad = beam(0.29, 0.0, D)
        bad.pos = np.array([np.nan, 0.0, 0.8])
        self.assertFalse(valid(bad))
        zq = beam(0.29, 0.0, D)
        zq.quat_xyzw = np.zeros(4)
        self.assertFalse(valid(zq))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))
        self.assertTrue(np.allclose(rotation((0, 0, 0, 2.0)), np.eye(3)))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = settle(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertFalse(r.details["left_packed"] or r.details["right_packed"])

    def test_left_packed_only(self):
        r, _ = settle(pack_left(start()))
        self.assertEqual(r.score, 0.25)
        self.assertTrue(r.details["left_packed"])
        self.assertEqual(r.details["left_level_mm"], 70.0)
        self.assertEqual(set(r.details["left_counted"]), {"packer_20_left", "packer_30_left"})

    def test_right_packed_only(self):
        r, _ = settle(pack_right(start()))
        self.assertEqual(r.score, 0.25)
        self.assertTrue(r.details["right_packed"])

    def test_both_packed_beam_on_table(self):
        r, _ = settle(packed())
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)

    def test_seated_scores_one_and_succeeds(self):
        r, _ = settle(seated())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["beam_seated"])
        self.assertEqual(r.details["left_bearing_mm"], 50.0)

    def test_seated_off_centre_within_bearing(self):
        self.assertEqual(settle(seated() | {"beam": beam(L.PIER_X, 0.0245, D)})[0].score, 1.0)  # 30.5 mm right end
        r, _ = settle(seated() | {"beam": beam(L.PIER_X, 0.0255, D)})  # 29.5 mm over the right packer
        self.assertEqual(r.score, 0.5)
        self.assertLess(r.details["right_bearing_mm"], 30.0)

    def test_beam_on_edge_is_a_long_face(self):
        objs = seated()
        objs["beam"] = beam(L.PIER_X, 0.0, D, ry=90.0)
        self.assertEqual(settle(objs)[0].score, 1.0)

    def test_beam_heading_and_roll_limits(self):
        self.assertEqual(settle(seated(yaw=8.0))[0].score, 1.0)
        self.assertEqual(settle(seated(yaw=12.0))[0].score, 0.5)
        self.assertEqual(settle(seated(ry=4.0))[0].score, 1.0)
        self.assertEqual(settle(seated(ry=7.0))[0].score, 0.5)

    def test_centre_line_must_cross_each_packer(self):
        objs = packed()
        objs["beam"] = beam(L.PIER_X + 0.030, 0.0, D)  # centre line 5 mm past the packers' far faces
        r, _ = settle(objs)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["left_centre_line_over"])
        objs["beam"] = beam(L.PIER_X + 0.020, 0.0, D)
        self.assertEqual(settle(objs)[0].score, 1.0)

    def test_beam_held_above_the_packing(self):
        objs = packed()
        objs["beam"] = beam(L.PIER_X, 0.0, D + 0.010)
        r, _ = settle(objs)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["beam_across_both"])

    def test_rest_tolerance(self):
        self.assertEqual(settle(pack_left(start(), z_err=0.0019))[0].score, 0.25)
        self.assertEqual(settle(pack_left(start(), z_err=-0.0019))[0].score, 0.25)
        r, _ = settle(pack_left(start(), z_err=0.0025))  # blue 2.5 mm above the yellow: held, not resting on it
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["left_rows"]["packer_30_left"]["rests_on"], None)

    def test_datum_band(self):
        """Packed means the packing's top within 3 mm of the datum (moved here instead of the packing)."""
        old = L.DATUM
        try:
            L.DATUM = 0.0729
            self.assertEqual(settle(pack_left(start()))[0].score, 0.25)
            L.DATUM = 0.0671
            self.assertEqual(settle(pack_left(start()))[0].score, 0.25)
            L.DATUM = 0.0732
            self.assertEqual(settle(pack_left(start()))[0].score, 0.0)
            L.DATUM = 0.0668
            self.assertEqual(settle(pack_left(start()))[0].score, 0.0)
        finally:
            L.DATUM = old

    def test_every_tier_is_reachable_and_nothing_else(self):
        """Score from a broad mix of end states: only 0, 0.25, 0.5 and 1.0 ever appear, and all four do."""
        seen = set()
        rng = np.random.default_rng(7)
        for _ in range(300):
            objs = start()
            if rng.random() < 0.7:
                pack_left(objs, z_err=float(rng.choice([0.0, 0.0, 0.0025, 0.01])))
            if rng.random() < 0.7:
                pack_right(objs)
            k = rng.integers(0, 4)
            if k == 1:
                objs["beam"] = beam(L.PIER_X + rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), D,
                                    yaw=rng.uniform(-15, 15))
            elif k == 2:
                objs["beam"] = tilted_beam(-0.155, 0.070, 0.205, 0.020)
            elif k == 3:
                objs["beam"] = beam(rng.uniform(0.1, 0.45), rng.uniform(-0.1, 0.1), TOP + rng.uniform(0, 0.2),
                                    yaw=rng.uniform(0, 180), rx=rng.uniform(-30, 30), ry=rng.uniform(-30, 30))
            s = settle(objs)[0].score
            self.assertIn(s, (0.0, 0.25, 0.5, 1.0))
            seen.add(s)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})


class ZeroConditions(unittest.TestCase):
    def test_beam_across_packed_and_bare_pier(self):
        """The defect: the beam laid across one packed pier and one bare one (7.8 deg): 0, though the right pier
        alone would score 0.25."""
        objs = pack_right(start())
        objs["beam"] = tilted_beam(-0.155, 0.070, 0.205, 0.020)
        r, _ = settle(objs)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["zero_beam_tilted_across"])
        self.assertLess(abs(r.details["left_beam_gap_mm"]), 1.0)
        self.assertGreater(r.details["beam_level_deg"], 7.5)
        self.assertTrue(r.details["right_packed"])

    def test_beam_across_unequal_packing(self):
        """Left packed to 70 mm, right packed to 80 mm (30 + 20 + 30): the beam rests 1.6 deg out of level: 0."""
        objs = pack_left(start())
        x, y = on("right")
        objs["packer_20_right"] = packer(20, "right", x, y, TOP + 0.030)
        objs["packer_30_right"] = packer(30, "right", x, y, TOP + 0.050)
        objs["beam"] = tilted_beam(0.205, 0.070, -0.155, 0.080)  # on the left's outer edge, the right's inner
        r, _ = settle(objs)
        self.assertTrue(r.details["beam_across_both"])
        self.assertAlmostEqual(r.details["beam_level_deg"], 1.59, places=2)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["zero_beam_tilted_across"])

    def test_level_threshold(self):
        """The same 1.59 deg beam is not zeroed with the level limit moved to 1.7 deg (the left pier then scores
        0.25; the right, at 80 mm, is not packed): the zero turns on exactly that threshold."""
        objs = pack_left(start())
        x, y = on("right")
        objs["packer_20_right"] = packer(20, "right", x, y, TOP + 0.030)
        objs["packer_30_right"] = packer(30, "right", x, y, TOP + 0.050)
        objs["beam"] = tilted_beam(0.205, 0.070, -0.155, 0.080)
        old = L.BEAM_LEVEL_DEG
        try:
            L.BEAM_LEVEL_DEG = 1.7
            r, _ = settle(objs)
            self.assertFalse(r.details["zero_beam_tilted_across"])
            self.assertEqual(r.score, 0.25)
        finally:
            L.BEAM_LEVEL_DEG = old

    def test_ten_mm_step_at_its_flattest_is_zeroed(self):
        """The likeliest bad packing: the green 40 mm packer alone on the 20 mm pier (60 mm) and the right pier packed.
        With the right packer 9 mm outboard and the left one 20 mm outboard the beam's contacts are 378 mm apart and
        it tilts only 1.47 deg: still over the 1.25 deg limit, so 0 wherever the packers sit (at 1.5 deg it was
        0.25 here and 0 with them centred)."""
        objs = start()
        x, y = on("right", dy=-0.009)
        objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.030)
        x, y = on("left", dy=0.020)
        objs["packer_40_left"] = packer(40, "left", x, y, TOP + 0.020)
        objs["beam"] = tilted_beam(-0.164, 0.070, 0.225, 0.060, yc=0.0151)
        r, _ = settle(objs)
        self.assertTrue(r.details["right_packed"])
        self.assertTrue(r.details["beam_across_both"])
        self.assertAlmostEqual(r.details["beam_level_deg"], 1.47, places=2)
        self.assertTrue(r.details["zero_beam_tilted_across"])
        self.assertEqual(r.score, 0.0)
        self.assertGreater(math.degrees(math.atan2(0.010, L.BEAM_LEN)), L.BEAM_LEVEL_DEG)  # 1.36: any 10 mm step

    def test_good_packing_at_its_limits_is_seated(self):
        """Both piers packed at opposite ends of their bands (left 72.9 mm: each packer 1.9 / 1.0 mm proud; right
        68.1 mm: the green 1.9 mm into the pier), both packers 9 mm inboard and the beam shifted 33 mm so that only
        31 mm of it bears on the right packer: its contacts 323 mm apart, tilted 0.85 deg: seated, 1.0. With its
        end just over the right packer's inner edge (contacts 293 mm apart, 0.94 deg) it is not seated but not
        zeroed either: 0.5. Two packed piers differ by at most 6 mm: at most 1.19 deg over 290 mm."""
        objs = start()
        x, y = on("left", dy=-0.009)
        objs["packer_20_left"] = packer(20, "left", x, y, TOP + 0.0219)
        objs["packer_30_left"] = packer(30, "left", x, y, TOP + 0.0429)
        x, y = on("right", dy=0.009)
        objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.0281)
        th = math.atan2(0.0729 - 0.0681, 0.146 + 0.177)
        objs["beam"] = tilted_beam(0.146, 0.0729, -0.177, 0.0681, yc=-0.177 + 0.21 * math.cos(th))
        r, _ = settle(objs)
        self.assertTrue(r.details["left_packed"] and r.details["right_packed"])
        self.assertAlmostEqual(r.details["beam_level_deg"], 0.85, places=2)
        self.assertGreaterEqual(r.details["right_bearing_mm"], 30.0)
        self.assertEqual(r.score, 1.0)
        th = math.atan2(0.0729 - 0.0681, 0.146 + 0.147)
        objs["beam"] = tilted_beam(0.146, 0.0729, -0.147, 0.0681, yc=-0.147 + 0.21 * math.cos(th))
        r, _ = settle(objs)
        self.assertTrue(r.details["beam_across_both"])
        self.assertFalse(r.details["zero_beam_tilted_across"])
        self.assertEqual(r.score, 0.5)
        self.assertLess(math.degrees(math.atan2(0.006, 0.290)), L.BEAM_LEVEL_DEG)  # 1.19

    def test_packer_on_edge_on_a_pier(self):
        """EXPLOIT: any packer stood on its 50 x t face on the 20 mm pier reaches exactly 70 mm. 0."""
        objs = pack_right(start())
        x, y = on("left")
        objs["packer_20_left"] = packer(20, "left", x, y, TOP + 0.020, rx=90.0)
        r, _ = settle(objs)
        self.assertAlmostEqual(r.details["left_level_mm"], 70.0, places=3)
        self.assertTrue(r.details["zero_on_edge"])
        self.assertEqual(r.score, 0.0)
        objs["beam"] = beam(L.PIER_X, 0.0, D)  # even with the beam level across it
        self.assertEqual(settle(objs)[0].score, 0.0)

    def test_packer_on_edge_on_the_right_pier(self):
        objs = pack_left(start())
        x, y = on("right")
        objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.030, ry=90.0)
        self.assertEqual(settle(objs)[0].score, 0.0)

    def test_packer_on_edge_off_the_piers_is_ignored(self):
        objs = packed()
        objs["packer_40_left"] = packer(40, "left", 0.28, 0.30, TOP, rx=90.0)
        self.assertEqual(settle(objs)[0].score, 0.5)


class PackerRules(unittest.TestCase):
    def test_overhang_limit(self):
        objs = pack_right(start())
        x, y = on("right")
        objs["packer_40_right"] = packer(40, "right", x + 0.0095, y, TOP + 0.030)
        self.assertEqual(settle(objs)[0].score, 0.25)
        objs["packer_40_right"] = packer(40, "right", x + 0.0130, y, TOP + 0.030)  # 8 mm over the edge
        r, _ = settle(objs)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["right_packed"])

    def test_leaning_packer_does_not_count(self):
        """Lying flat means within 3 deg: leaning 4 deg on an edge its underside's middle is still within the 2 mm
        rest band (1.7 mm up), so only the flatness rule rejects it; at 5 deg both do."""
        objs = start()
        x, y = on("right")
        objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.030, rx=2.5)
        self.assertEqual(settle(objs)[0].score, 0.25)
        objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.030, rx=4.0)
        r, _ = settle(objs)
        self.assertEqual(r.details["right_rows"]["packer_40_right"]["rests_on"], "pier")
        self.assertEqual(r.score, 0.0)
        objs["packer_40_right"] = packer(40, "right", x, y, TOP + 0.030, rx=5.0)
        self.assertEqual(settle(objs)[0].score, 0.0)

    def test_stack_needs_upper_over_lower(self):
        objs = pack_left(start())
        x, y = on("left")
        objs["packer_30_left"] = packer(30, "left", x + 0.009, y - 0.009, TOP + 0.040)
        self.assertEqual(settle(objs)[0].score, 0.25)

    def test_wrong_packing_is_not_packed(self):
        objs = start()
        x, y = on("left")
        objs["packer_40_left"] = packer(40, "left", x, y, TOP + 0.020)  # 60 mm
        r, _ = settle(objs)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["left_level_mm"], 60.0)
        objs["packer_20_left"] = packer(20, "left", x, y, TOP + 0.060)  # 80 mm
        self.assertEqual(settle(objs)[0].details["left_level_mm"], 80.0)
        self.assertEqual(settle(objs)[0].score, 0.0)

    def test_two_yellows_on_the_right_is_valid_packing(self):
        """20 + 20 (one from each set) also makes 40: the right pier counts as packed (the left cannot then be)."""
        objs = start()
        x, y = on("right")
        objs["packer_20_right"] = packer(20, "right", x, y, TOP + 0.030)
        objs["packer_20_left"] = packer(20, "left", x, y, TOP + 0.050)
        self.assertEqual(settle(objs)[0].score, 0.25)

    def test_every_packer_on_the_pier_must_count(self):
        """EXPLOIT: a correct stack with a stray packer perched on it, overhanging 15 mm (the stack's top is then
        not the datum): not packed."""
        objs = pack_left(start())
        x, y = on("left")
        objs["packer_40_left"] = packer(40, "left", x + 0.020, y, TOP + 0.070)
        r, _ = settle(objs)
        self.assertFalse(r.details["left_packed"])
        self.assertEqual(r.score, 0.0)

    def test_packer_on_the_beam_spoils_that_pier(self):
        objs = seated()
        x, y = on("left")
        objs["packer_40_left"] = packer(40, "left", x, y, D + 0.030)  # lying on the beam over the left bearing
        r, _ = settle(objs)
        self.assertFalse(r.details["left_packed"])
        self.assertEqual(r.score, 0.25)

    def test_packer_on_a_datum_post_is_ignored(self):
        objs = seated()
        objs["packer_20_right"] = packer(20, "right", L.POST_X, -L.POST_Y, TOP + L.DATUM)
        self.assertEqual(settle(objs)[0].score, 1.0)

    def test_held_packer_over_the_pier(self):
        """A packer still held 10 mm above the stack, or carried across it: that pier is not packed."""
        objs = pack_left(start())
        x, y = on("left")
        objs["packer_30_left"] = packer(30, "left", x, y, TOP + 0.050)
        self.assertEqual(settle(objs)[0].score, 0.0)


class Stillness(unittest.TestCase):
    def test_needs_a_full_still_window(self):
        objs = seated()
        r, t = run([objs], n=WINDOW - 1)
        self.assertLess(r.score, 1.0)
        r, _ = run([objs], t, n=1)
        self.assertEqual(r.score, 1.0)

    def test_sliding_beam_is_not_seated(self):
        a = seated()
        b = seated() | {"beam": beam(L.PIER_X, 0.012, D)}
        _, t = settle(a)
        r, _ = run(glide(a, b, "beam", 15), t)  # 12 mm in 0.75 s: 4.8 mm per 0.6 s window
        self.assertEqual(r.score, 0.5)
        r, _ = settle(b, t)
        self.assertEqual(r.score, 1.0)

    def test_creeping_packer_does_not_count(self):
        a = pack_right(start())
        x, y = on("right")
        b = dict(a)
        b["packer_40_right"] = packer(40, "right", x + 0.008, y, TOP + 0.030)
        _, t = settle(a)
        r, _ = run(glide(a, b, "packer_40_right", 10), t)
        self.assertEqual(r.score, 0.0)

    def test_jitter_under_a_millimetre_is_still(self):
        rng = np.random.default_rng(3)
        t = Tracker(HZ)
        r = None
        for _ in range(40):
            objs = seated()
            jb = rng.uniform(-0.0006, 0.0006, 3)
            b = objs["beam"]
            objs["beam"] = SimpleNamespace(pos=b.pos + jb, quat_xyzw=b.quat_xyzw, lin_vel=rng.normal(0, 0.05, 3),
                                           ang_vel=rng.normal(0, 1.0, 3), low=b.low + jb, high=b.high + jb)
            r = t.update(objs)
        self.assertEqual(r.score, 1.0)  # reported velocities chatter (gripped): positions decide


class RobustnessAndReset(unittest.TestCase):
    def test_nan_state_scores_zero_and_keeps_the_record(self):
        _, t = settle(seated())
        bad = seated()
        bad["beam"].pos = np.array([np.nan, 0.0, 0.8])
        r = t.update(bad)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(len(t.hist["beam"]), WINDOW)
        r = t.update(seated())
        self.assertEqual(r.score, 1.0)  # the window was kept: still seated at once

    def test_inf_and_zero_quaternion_and_missing_object(self):
        t = Tracker(HZ)
        bad = seated()
        bad["packer_20_left"].low = np.array([0.0, np.inf, 0.0])
        self.assertEqual(t.update(bad).score, 0.0)
        bad = seated()
        bad["packer_40_right"].quat_xyzw = np.zeros(4)
        self.assertEqual(t.update(bad).score, 0.0)
        bad = seated()
        del bad["beam"]
        self.assertEqual(t.update(bad).score, 0.0)
        self.assertEqual(t.result(bad).score, 0.0)

    def test_teleport_back_to_start_is_a_new_episode(self):
        _, t = settle(seated())
        r = t.update(start())
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(t.hist["beam"]), 1)  # the window began again
        self.assertTrue(Tracker.at_start(start()))
        self.assertFalse(Tracker.at_start(seated()))

    def test_jittered_reset_is_recognised(self):
        objs = start()
        for n in L.LOOSE:
            s = objs[n]
            objs[n] = SimpleNamespace(pos=s.pos + [0.02, -0.02, 0], quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel,
                                      ang_vel=s.ang_vel, low=s.low + [0.02, -0.02, 0], high=s.high + [0.02, -0.02, 0])
        self.assertTrue(Tracker.at_start(objs))

    def test_gradual_return_is_not_a_reset(self):
        a = start()
        b = start() | {"beam": beam(L.BEAM_X, 0.06, TOP)}
        _, t = settle(b)
        n = t.steps
        run(glide(b, a, "beam", 10), t)
        self.assertEqual(t.steps, n + 10)
        self.assertEqual(len(t.hist["beam"]), WINDOW)

    def test_reset_method(self):
        _, t = settle(seated())
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(seated()).score, 0.0)  # nothing is still yet: every object needs its window


class Exploits(unittest.TestCase):
    def test_beam_level_on_two_equally_wrong_stacks(self):
        """Both piers built to 80 mm (20 + 40 left, 30 + 20 + 30 right) and the beam level on them: neither pier is
        packed, so 0 although the beam is level."""
        objs = start()
        x, y = on("left")
        objs["packer_20_left"] = packer(20, "left", x, y, TOP + 0.020)
        objs["packer_40_left"] = packer(40, "left", x, y, TOP + 0.040)
        x, y = on("right")
        objs["packer_30_right"] = packer(30, "right", x, y, TOP + 0.030)
        objs["packer_20_right"] = packer(20, "right", x, y, TOP + 0.060)
        objs["beam"] = beam(L.PIER_X, 0.0, TOP + 0.080)
        r, _ = settle(objs)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["zero_beam_tilted_across"])

    def test_beam_laid_on_the_piers_without_packing(self):
        """The beam set straight on the bare piers (20 and 30 mm): out of level, 0."""
        objs = start()
        objs["beam"] = tilted_beam(0.205, 0.020, -0.150, 0.030)  # left pier's outer edge, right pier's inner
        r, _ = settle(objs)
        self.assertTrue(r.details["beam_across_both"])
        self.assertEqual(r.score, 0.0)

    def test_packer_on_top_of_the_beam_is_not_a_support(self):
        """The beam bears on the right packing with its left end held 15 mm up (tilted 2.4 deg) and a packer lying on
        top of that end, over the left pier: the beam does not rest on the left pier (what lies on it is no support),
        so it is not across both piers; the right pier scores 0.25 and the left is spoilt by the stray packer."""
        objs = pack_right(start())
        objs["beam"] = tilted_beam(-0.155, 0.070, 0.205, 0.085)
        x, y = on("left")
        th = math.atan2(0.015, 0.36)
        top_at = TOP + 0.070 + (y + 0.155) * math.tan(th) + L.BEAM_SIZE[2] / math.cos(th)
        objs["packer_20_left"] = packer(20, "left", x, y, top_at + 0.001, rx=math.degrees(th))
        r, _ = settle(objs)
        self.assertFalse(r.details["left_beam_rests"])
        self.assertFalse(r.details["beam_across_both"])
        self.assertEqual(r.score, 0.25)

    def test_beam_resting_on_one_packed_pier_and_the_table(self):
        """One end on the right packing, the other end on the table: not across both piers; the right pier still
        scores 0.25 and the beam nothing."""
        objs = pack_right(start())
        # turned 10 deg so that its left end lands on the table in front of the left pier; it bears on the right
        # packer's inner edge and its left end's underside edge touches the table: tilted 11 deg.
        th = math.degrees(math.asin(-0.070 / (0.21 + 0.15)))
        objs["beam"] = beam_line((L.PIER_X, -0.155, TOP + 0.070), -0.15 / math.cos(math.radians(th)), th, yaw=10.0)
        r, _ = settle(objs)
        self.assertTrue(r.details["right_beam_rests"])
        self.assertFalse(r.details["left_beam_rests"])
        self.assertLess(abs(objs["beam"].low[2] - TOP), 0.001)
        self.assertEqual(r.score, 0.25)


class TaskHooks(unittest.TestCase):
    """task.py against the offline yam_arena stub (tools/offline/yam_stub), when it is there."""

    @classmethod
    def setUpClass(cls):
        stub = HERE.parent.parent / "tools" / "offline" / "yam_stub"
        if not stub.is_dir():
            raise unittest.SkipTest("tools/offline/yam_stub not found (the task runs against yam_arena itself)")
        for p in (str(stub), str(stub.parent)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def load(self, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            sys.modules.pop("task", None)
            return importlib.import_module("task")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_objects_and_hooks(self):
        task = self.load()
        names = [o.name for o in task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        free = sorted(o.name for o in task.OBJECTS if o.mass is not None)
        self.assertEqual(free, sorted(L.LOOSE))
        self.assertTrue(all(o.mass <= 0.3 for o in task.OBJECTS if o.mass is not None))
        cap = next(o for o in task.OBJECTS if o.name == "datum_cap_left")
        self.assertAlmostEqual(cap.pos[2] + cap.size[2] / 2, TOP + L.DATUM)
        for side in L.SIDES:
            p = next(o for o in task.OBJECTS if o.name == f"pier_{side}")
            self.assertAlmostEqual(p.pos[2] + p.size[2] / 2, TOP + L.PIER_H[side])
        objs = seated()
        for _ in range(WINDOW + 2):
            ok = task.success(objs)
        self.assertTrue(ok)
        self.assertEqual(task.score(objs), 1.0)

    def test_trace_rows(self):
        task = self.load()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                task.success(start())
                task.success(seated())
            finally:
                os.environ.pop("TASK_TRACE")
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        for key in ("step", "score", "success", "left_level_mm", "right_level_mm", "beam_level_deg",
                    "left_bearing_mm", "zero_on_edge", "zero_beam_tilted_across", "beam_pos", "packer_20_left_pos"):
            self.assertIn(key, rows[1])
        os.environ["TASK_TRACE"] = "/nonexistent-dir/x/trace.jsonl"
        try:
            task.success(start())  # a bad trace path never stops a trial
        finally:
            os.environ.pop("TASK_TRACE")

    def test_demo_switches_and_timings(self):
        task = self.load()
        plan = task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(task.demo()), {"right"})
        finally:
            os.environ.pop("DEMO_IDLE_ARM")
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(task.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if (prev.finger - task.OPEN) * (w.finger - task.OPEN) == 0 and prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, f"{arm} {w.label}: opens/closes in under 1.5 s")
                if "slowly" in w.label:
                    self.assertGreaterEqual(w.duration, 1.0)
        right = plan["right"]
        grip_t = sum(w.duration for w in right[:next(i for i, w in enumerate(right) if "grip the beam" in w.label)])
        left_end = sum(w.duration for w in plan["left"])
        self.assertGreaterEqual(grip_t, left_end)  # the beam is gripped only after the left arm has parked
        self.assertLess(sum(w.duration for w in right) * 20 + task.SUCCESS_HOLD_S * 20, 2400)

    def test_beam_grip_pads_clear_the_packing(self):
        """The beam is gripped at its middle: the open pads (94 mm apart, 20 mm wide) stay over the midline, far
        from the packers at |y| >= 155 mm."""
        task = self.load()
        self.assertLess(0.010 + 0.0, L.PIER_Y - L.PACKER_HALF)
        self.assertAlmostEqual(task.BEAM_GRIP, -0.016)
        self.assertAlmostEqual(task.PACKER_GRIP, -0.021)


if __name__ == "__main__":
    unittest.main()
