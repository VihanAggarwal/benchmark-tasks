"""Offline tests for scoring.py and task.py: synthetic pin and bracket states, no simulator.
python3 -m unittest discover -s tests -v"""

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
    BRACKET,
    NAMES,
    PLATE_TOP_W,
    Tracker,
    bracket_corners,
    bracket_holes,
    bracket_pose,
    head_turn_err_deg,
    in_hole,
    off_table,
    pin_axis,
    pin_pose,
    rotation,
    socket_of,
    valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
R1, R2 = L.ROUND
D = L.DIAMOND
SA, SB = L.SOCKETS["A"], L.SOCKETS["B"]


# -- quaternions (x, y, z, w) --------------------------------------------------------------------------------------


def about(axis, deg):
    """A turn of ``deg`` degrees about a world axis ('x', 'y', 'z' or a 3-vector)."""
    v = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}.get(axis, axis)
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    h = math.radians(deg) / 2
    return np.array([*(math.sin(h) * v), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def pose_quat(yaw=0.0, tilt=0.0, tilt_axis="x"):
    """Turned ``yaw`` degrees about the vertical, then tilted ``tilt`` degrees about a world axis."""
    return qmul(about(tilt_axis, tilt), about("z", yaw))


# -- synthetic states ----------------------------------------------------------------------------------------------


def _rim(r, z, n=24):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def _box_corners(center, size):
    c, s = np.asarray(center, float), np.asarray(size, float) / 2
    return [tuple(c + s * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]


ROUND_PTS = np.array(_rim(L.PIN_R, -L.PIN_H / 2) + _rim(L.PIN_R, L.PIN_H / 2))  # about the centre
DIAMOND_PTS = np.array(_rim(L.PIN_R, 0.0) + _rim(L.PIN_R, L.SHANK_H) + _box_corners(L.HEAD_CENTER, L.HEAD_SIZE))
BRACKET_PTS = np.array([p for c, s in L.BRACKET_SHAPES for p in _box_corners(c, s)])


def _world(points, q, origin):
    return np.asarray(points) @ rotation(q).T + np.asarray(origin, dtype=float)


def round_pin(x, y, floor=0.0, yaw=0.0, tilt=0.0, tilt_axis="x", flip=False, base=L.TOP):
    """A plain cylinder (yam_arena reports its centre) with its lower end's centre at (x, y), ``floor`` m above
    ``base`` while upright; tilted about that point. ``flip``: the same pin the other end down."""
    q = pose_quat(yaw, tilt, tilt_axis)
    if flip:
        q = qmul(q, about("x", 180.0))
    bottom = np.array([x, y, base + floor])
    a = rotation(q)[:, 2]
    a = a if a[2] >= 0 else -a
    c = bottom + a * (L.PIN_H / 2)
    pts = _world(ROUND_PTS, q, c)
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pts.min(axis=0),
                           high=pts.max(axis=0))


def diamond(x, y, floor=0.0, yaw=0.0, tilt=0.0, tilt_axis="x", base=L.TOP):
    """The diamond pin as mechkit reports it: base frame at the centre of the shank's bottom face, (x, y), ``floor``
    m above ``base``; ``yaw`` the direction of the head's long side (0: along x, flats facing +-y)."""
    q = pose_quat(yaw, tilt, tilt_axis)
    o = np.array([x, y, base + floor])
    pts = _world(DIAMOND_PTS, q, o)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=lo, high=hi)


def bracket(x=L.FIX_X, y=L.FIX_Y + L.BR_AIM_Y, z=PLATE_TOP_W, yaw=0.0, tilt=0.0, tilt_axis="x"):
    """The bracket as mechkit reports it: base frame at the centre of its underside at (x, y, z)."""
    q = pose_quat(yaw, tilt, tilt_axis)
    o = np.array([x, y, z])
    pts = _world(BRACKET_PTS, q, o)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=lo, high=hi)


def in_socket(k, name, **kw):
    """A pin standing on the socket's floor, centred in socket ``k``."""
    sx, sy = L.SOCKETS[k]
    return diamond(sx, sy, **kw) if name == D else round_pin(sx, sy, **kw)


def on_tray(name):
    if name == D:
        return diamond(*L.DIAMOND_START, base=L.TOP + L.TRAY_TOP, yaw=math.degrees(L.DIAMOND_START_YAW))
    return round_pin(*L.ROUND_START[name], base=L.TOP + L.TRAY_TOP)


def scene(**parts):
    """The starting layout with some parts replaced."""
    s = {n: on_tray(n) for n in L.PIN_NAMES}
    s[BRACKET] = bracket(*L.BR_START, z=L.TOP)
    s.update(parts)
    return s


START = scene()
LOCATED = scene(diamond_pin=in_socket("A", D), round_pin_1=in_socket("B", R1))
SEATED = scene(diamond_pin=in_socket("A", D), round_pin_1=in_socket("B", R1), bracket=bracket())
MIRROR = scene(diamond_pin=in_socket("B", D), round_pin_2=in_socket("A", R2),
               bracket=bracket(y=L.FIX_Y - L.BR_AIM_Y))


def feed(tr, states):
    r = None
    for st in states:
        r = tr.update(st)
    return r


def settled(st, n=WINDOW + 1):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n)


def scored(st):
    return settled(st)[1].score


def glide(a, b, n):
    """n scenes moving every part's state from a to b (positions interpolated, b's orientations)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        s = {}
        for name in NAMES:
            sa, sb = a[name], b[name]
            kw = dict(quat_xyzw=sb.quat_xyzw, lin_vel=np.zeros(3), ang_vel=np.zeros(3))
            d = np.asarray(sb.pos) - np.asarray(sa.pos)
            kw.update(pos=np.asarray(sa.pos) + f * d, low=np.asarray(sb.low) - (1 - f) * d,
                      high=np.asarray(sb.high) - (1 - f) * d)
            if hasattr(sb, "origin"):
                kw["origin"] = np.asarray(sa.origin) + f * (np.asarray(sb.origin) - np.asarray(sa.origin))
            s[name] = SimpleNamespace(**kw)
        out.append(s)
    return out


# -- the bracket's way down, from the parts' shapes (a 2D check at the bracket's level, independent of scoring.py) --


def _section_inside_hole(pin_kind, axis_xy, head_yaw_deg, hole_c):
    """Does a pin's cross-section at the bracket's level (15-25 mm: the round pin's circle, the diamond's head
    bar) lie inside a 64 x 64 mm hole centred at ``hole_c`` (bracket square to the table)?"""
    hx, hy = hole_c
    px, py = axis_xy
    if pin_kind == "round":
        return abs(px - hx) + L.PIN_R <= L.HOLE_HALF + 1e-12 and abs(py - hy) + L.PIN_R <= L.HOLE_HALF + 1e-12
    c, s = math.cos(math.radians(head_yaw_deg)), math.sin(math.radians(head_yaw_deg))
    ex = abs(c) * L.HEAD_HALF_LONG + abs(s) * L.HEAD_HALF_FLATS
    ey = abs(s) * L.HEAD_HALF_LONG + abs(c) * L.HEAD_HALF_FLATS
    return abs(px - hx) + ex <= L.HOLE_HALF + 1e-12 and abs(py - hy) + ey <= L.HOLE_HALF + 1e-12


def can_go_down(bracket_xy, pins):
    """``pins``: [(kind, (x, y), head_yaw_deg)]. The bracket (square, holes along y) comes down flat onto the
    plate only if each pin's section lies inside one of its holes and no two pins share a hole."""
    bx, by = bracket_xy
    holes = [(bx, by + L.HOLE_Y), (bx, by - L.HOLE_Y)]
    used = set()
    for kind, xy, yaw in pins:
        ok = [i for i, h in enumerate(holes) if i not in used and _section_inside_hole(kind, xy, yaw, h)]
        if not ok:
            return False
        used.add(ok[0])
    return True


def window_mm(pins, x=L.FIX_X):
    """The bracket centre's y range (mm from the fixture's centre) over which it can come down, or None."""
    ys = [y for y in np.arange(-40.0, 40.0001, 0.05) if can_go_down((x, L.FIX_Y + y / 1000), pins)]
    return (round(min(ys), 2), round(max(ys), 2)) if ys else None


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(all(valid(n, START[n]) for n in NAMES))
        self.assertTrue(Tracker.at_start(START))
        self.assertAlmostEqual(head_turn_err_deg(START[D]), 90.0, places=6)  # starts along the line: wrong
        for n in L.PIN_NAMES:
            self.assertIsNone(pin_pose(n, START[n])["socket"])
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(settled(START)[1].details["zero_reason"], "no_pin_fitted")

    def test_sockets_and_fixture(self):
        self.assertAlmostEqual(SA[1] - SB[1], L.PIN_SPACING)
        self.assertAlmostEqual(L.SOCKET - 2 * L.PIN_R, 0.007)  # 3.5 mm of play either side
        self.assertEqual(socket_of((SA[0], SA[1] + 0.027)), "A")
        self.assertEqual(socket_of((SB[0] - 0.027, SB[1])), "B")
        self.assertIsNone(socket_of((SA[0], SA[1] + 0.028)))
        self.assertIsNone(socket_of((L.FIX_X, L.FIX_Y)))
        # the plate's five boxes leave exactly the two 55 x 55 mm openings, centred on the sockets
        mid, el, er = (L.FIXTURE_PARTS[k] for k in ("fixture_middle", "fixture_end_left", "fixture_end_right"))
        self.assertAlmostEqual(el[2] - mid[3], L.SOCKET)
        self.assertAlmostEqual(mid[2] - er[3], L.SOCKET)
        self.assertAlmostEqual((el[2] + mid[3]) / 2, SA[1])
        self.assertAlmostEqual(L.FIXTURE_PARTS["fixture_far"][0] - L.FIXTURE_PARTS["fixture_near"][1], L.SOCKET)

    def test_bracket_holes_from_its_shapes(self):
        """The five plate boxes leave two 64 x 64 mm through-holes centred at local y +-62 mm (124 mm apart)."""
        plate = [(np.array(c), np.array(s)) for c, s in L.BRACKET_SHAPES if abs(c[2] - L.BR_T / 2) < 1e-12]
        self.assertEqual(len(plate), 5)

        def solid(px, py):
            return any(abs(px - c[0]) <= s[0] / 2 and abs(py - c[1]) <= s[1] / 2 for c, s in plate)

        for hy in (L.HOLE_Y, -L.HOLE_Y):
            self.assertFalse(solid(0.0, hy))
            for dx, dy in ((0.0315, 0.0), (0.0, 0.0315), (-0.0315, 0.0), (0.0, -0.0315), (0.031, 0.031)):
                self.assertFalse(solid(dx, hy + dy), (dx, dy))
            for dx, dy in ((0.0325, 0.0), (0.0, 0.0325), (-0.0325, 0.0), (0.0, -0.0325)):
                self.assertTrue(solid(dx, hy + dy), (dx, dy))
        self.assertAlmostEqual(L.HOLE_SPACING - L.PIN_SPACING, 0.024)
        self.assertTrue(solid(0.0, 0.0) and solid(0.043, 0.105) and not solid(0.045, 0.0))

    def test_volumes_and_masses(self):
        br = sum(s[0] * s[1] * s[2] for _, s in L.BRACKET_SHAPES)
        self.assertAlmostEqual(br * 1.41e3, L.BR_MASS, delta=0.002)  # acetal, 1.41 g/cm3
        solid_pin = math.pi * L.PIN_R ** 2 * L.PIN_H * 7.85e3
        self.assertAlmostEqual(solid_pin, 0.78, delta=0.005)  # task.json: a solid steel pin would weigh 780 g
        for m in (L.PIN_MASS, L.DIAMOND_MASS, L.BR_MASS):
            self.assertLessEqual(m, 0.30)

    def test_window_for_the_located_set(self):
        """Round pin in B and the diamond in A, flats facing B: the bracket comes down with its centre 4 to 16 mm
        to the left (task.json); the mirror set, 16 to 4 mm to the right. The demo aims at the middle."""
        located = [("round", SB, 0.0), ("diamond", SA, 0.0)]
        self.assertEqual(window_mm(located), (4.0, 16.0))
        mirror = [("round", SA, 0.0), ("diamond", SB, 0.0)]
        self.assertEqual(window_mm(mirror), (-16.0, -4.0))
        self.assertAlmostEqual(L.WINDOW_LO, 0.004)
        self.assertAlmostEqual(L.WINDOW_HI, 0.016)
        self.assertAlmostEqual(L.BR_AIM_Y, 0.010)
        # across the line each pin leaves 8 mm either way
        self.assertTrue(can_go_down((L.FIX_X + 0.0079, L.FIX_Y + 0.010), located))
        self.assertFalse(can_go_down((L.FIX_X + 0.0081, L.FIX_Y + 0.010), located))

    def test_wrong_sets_cannot_come_down(self):
        """Two round pins, or the diamond turned a quarter turn (its head's long side along the line), leave no
        position from which the bracket comes down over both: it rests on the pin heads."""
        self.assertIsNone(window_mm([("round", SB, 0.0), ("round", SA, 0.0)]))
        self.assertIsNone(window_mm([("round", SB, 0.0), ("diamond", SA, 90.0)]))
        self.assertIsNone(window_mm([("round", SA, 0.0), ("diamond", SB, 90.0)]))
        # each pin alone allows a band; the two bands miss each other by 8 mm
        self.assertEqual(window_mm([("round", SB, 0.0)]), (4.0, 20.0))
        self.assertEqual(window_mm([("round", SA, 0.0)]), (-20.0, -4.0))
        self.assertAlmostEqual(L.BLOCKED_HI, -0.004)

    def test_play_in_the_sockets(self):
        """With each pin anywhere in its socket's 3.5 mm of play: a located set always leaves at least 5 mm of
        window, and a wrong set always blocks the bracket (by at least 1 mm)."""
        p = (L.SOCKET - 2 * L.PIN_R) / 2
        widths, gaps = [], []
        for d1 in (-p, 0.0, p):
            for d2 in (-p, 0.0, p):
                w = window_mm([("round", (SB[0], SB[1] + d1), 0.0), ("diamond", (SA[0], SA[1] + d2), 0.0)])
                self.assertIsNotNone(w)
                widths.append(w[1] - w[0])
                for wrong in (("round", 0.0), ("diamond", 90.0)):
                    pins = [("round", (SB[0], SB[1] + d1), 0.0), (wrong[0], (SA[0], SA[1] + d2), wrong[1])]
                    self.assertIsNone(window_mm(pins))
                a = window_mm([("round", (SB[0], SB[1] + d1), 0.0)])
                b = window_mm([("round", (SA[0], SA[1] + d2), 0.0)])
                gaps.append(a[0] - b[1])
        self.assertAlmostEqual(min(widths), 5.0, places=1)
        self.assertAlmostEqual(min(gaps), 1.0, places=1)

    def test_diamond_turned_up_to_15_degrees_still_lets_it_down(self):
        """'Turned right' allows 15 degrees: the bracket still comes down there; at 35 degrees it no longer
        does (between them a seated bracket scores 0.25, as the scoring text says)."""
        self.assertIsNotNone(window_mm([("round", SB, 0.0), ("diamond", SA, 15.0)]))
        self.assertIsNone(window_mm([("round", SB, 0.0), ("diamond", SA, 35.0)]))

    def test_turn_error_and_play_together(self):
        """README: with the pins centred, a 15-degree turn brings the window's upper edge down from 16 to 9.9 mm;
        with both pins pushed the wrong way in their play (each toward the other) that window closes, while a
        square diamond still leaves 5 mm."""
        w = window_mm([("round", SB, 0.0), ("diamond", SA, 15.0)])
        self.assertAlmostEqual(w[0], 4.0, places=1)
        self.assertAlmostEqual(w[1], 9.9, delta=0.06)
        p = (L.SOCKET - 2 * L.PIN_R) / 2
        inward = [("round", (SB[0], SB[1] + p), 0.0), ("diamond", (SA[0], SA[1] - p), 15.0)]
        self.assertIsNone(window_mm(inward))
        w0 = window_mm([inward[0], ("diamond", inward[1][1], 0.0)])
        self.assertAlmostEqual(w0[1] - w0[0], 5.0, places=1)

    def test_pin_axis_and_turn(self):
        b, t, tilt = pin_axis(R1, round_pin(0.3, 0.1, floor=0.001))
        np.testing.assert_allclose(b, [0.3, 0.1, L.TOP + 0.001], atol=1e-12)
        np.testing.assert_allclose(t, [0.3, 0.1, L.TOP + 0.001 + L.PIN_H], atol=1e-12)
        self.assertAlmostEqual(tilt, 0.0)
        b2, _, tilt2 = pin_axis(R1, round_pin(0.3, 0.1, floor=0.001, flip=True))
        np.testing.assert_allclose(b2, b, atol=1e-12)
        self.assertAlmostEqual(tilt2, 0.0, places=6)  # a round pin either end down
        _, _, td = pin_axis(D, diamond(0.3, 0.1, tilt=180.0))
        self.assertAlmostEqual(td, 180.0, places=6)  # the diamond's head down
        for yaw, err in ((0.0, 0.0), (180.0, 0.0), (90.0, 90.0), (-90.0, 90.0), (15.0, 15.0), (-30.0, 30.0),
                         (165.0, 15.0)):
            self.assertAlmostEqual(head_turn_err_deg(diamond(*SA, yaw=yaw)), err, places=6)
        self.assertAlmostEqual(head_turn_err_deg(diamond(*SA, tilt=90.0, tilt_axis="y")), 90.0)  # long side up

    def test_bracket_measures(self):
        br = bracket()
        np.testing.assert_allclose(bracket_holes(br)[:, 1], [L.FIX_Y + 0.010 + 0.062, L.FIX_Y + 0.010 - 0.062])
        self.assertTrue(bracket_pose(br)["flat"])
        self.assertTrue(all(abs(c[2] - PLATE_TOP_W) < 1e-12 for c in bracket_corners(br)))
        self.assertEqual(in_hole(br, *pin_axis(D, in_socket("A", D))[:2])[0], 0)
        self.assertEqual(in_hole(br, *pin_axis(R1, in_socket("B", R1))[:2])[0], 1)
        u, v = in_hole(br, *pin_axis(R1, in_socket("B", R1))[:2])[1]
        self.assertAlmostEqual(u, 0.0)
        self.assertAlmostEqual(v, -L.PIN_SPACING / 2 - (0.010 - L.HOLE_Y))  # 2 mm toward the outer edge
        self.assertEqual(in_hole(br, np.array([0.3, 0.0, L.TOP]), np.array([0.4, 0.0, L.TOP]))[0], None)  # parallel


class Tiers(unittest.TestCase):
    def test_every_tier_is_reachable(self):
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(scored(scene(round_pin_1=in_socket("B", R1))), 0.25)
        self.assertEqual(scored(LOCATED), 0.5)
        self.assertEqual(scored(SEATED), 1.0)

    def test_only_one_is_success(self):
        for st, want in ((START, False), (LOCATED, False), (scene(round_pin_1=in_socket("B", R1)), False),
                         (SEATED, True)):
            r = settled(st)[1]
            self.assertEqual(r.success, want)
            self.assertEqual(r.success, r.score == 1.0)

    def test_mirror_set_scores_one(self):
        self.assertEqual(scored(MIRROR), 1.0)
        self.assertEqual(settled(MIRROR)[1].details["located_round_pin"], R2)

    def test_either_round_pin_and_either_hole(self):
        st = scene(diamond_pin=in_socket("A", D), round_pin_2=in_socket("B", R2), bracket=bracket())
        self.assertEqual(scored(st), 1.0)
        # the bracket turned end for end: the holes swap, still seated
        st = scene(diamond_pin=in_socket("A", D), round_pin_1=in_socket("B", R1), bracket=bracket(yaw=180.0))
        self.assertEqual(scored(st), 1.0)

    def test_diamond_turned_either_way_round(self):
        for yaw in (180.0, 14.0, -14.0, 194.0):
            st = scene(diamond_pin=in_socket("A", D, yaw=yaw), round_pin_1=in_socket("B", R1), bracket=bracket())
            self.assertEqual(scored(st), 1.0, yaw)

    def test_one_pin_only_scores_a_quarter(self):
        for st in (scene(round_pin_1=in_socket("B", R1)), scene(round_pin_2=in_socket("A", R2)),
                   scene(diamond_pin=in_socket("A", D)), scene(diamond_pin=in_socket("B", D, yaw=90.0))):
            self.assertEqual(scored(st), 0.25)

    def test_two_round_pins_score_a_quarter(self):
        st = scene(round_pin_1=in_socket("B", R1), round_pin_2=in_socket("A", R2))
        self.assertEqual(scored(st), 0.25)
        self.assertFalse(settled(st)[1].details["located_set"])

    def test_round_pin_must_be_in_the_other_socket(self):
        """Both read in socket A (not possible with solid parts; e.g. a pin pushed through the other in a
        blow-up): no located set."""
        st = scene(diamond_pin=in_socket("A", D), round_pin_1=round_pin(SA[0] + 0.020, SA[1]))
        r = settled(st)[1]
        self.assertEqual(r.details["fitted"], {R1: "A", R2: None, D: "A"})
        self.assertEqual(r.score, 0.25)

    def test_diamond_not_turned_right_scores_a_quarter(self):
        for yaw in (90.0, 16.0, -16.0, 45.0, 270.0):
            st = scene(diamond_pin=in_socket("A", D, yaw=yaw), round_pin_1=in_socket("B", R1))
            self.assertEqual(scored(st), 0.25, yaw)

    def test_turn_tolerance_edge(self):
        for yaw, want in ((14.9, 0.5), (15.1, 0.25)):
            st = scene(diamond_pin=in_socket("A", D, yaw=yaw), round_pin_1=in_socket("B", R1))
            self.assertEqual(scored(st), want, yaw)

    def test_bracket_seated_over_a_wrong_set_stays_at_a_quarter(self):
        """In the simulation the bracket cannot come down over a wrong set; if it somehow did, it still scores
        0.25. The same over a single pin, and over the diamond turned 20 degrees (where it can come down)."""
        for pins in ({"round_pin_1": in_socket("B", R1), "round_pin_2": in_socket("A", R2)},
                     {"round_pin_1": in_socket("B", R1), "diamond_pin": in_socket("A", D, yaw=90.0)},
                     {"round_pin_1": in_socket("B", R1), "diamond_pin": in_socket("A", D, yaw=20.0)},
                     {"round_pin_1": in_socket("B", R1)},
                     {"diamond_pin": in_socket("A", D)}):
            st = scene(bracket=bracket(), **pins)
            r = settled(st)[1]
            self.assertEqual(r.score, 0.25)
            self.assertTrue(r.details["bracket_seated"])

    def test_located_set_with_the_bracket_anywhere_else_is_a_half(self):
        cases = {
            "on the table at its start": START[BRACKET],
            "held 5 mm up over the pins": bracket(z=PLATE_TOP_W + 0.005),
            "resting on the pin heads": bracket(y=L.FIX_Y - 0.010, z=L.TOP + L.PIN_H),
            "flat on the plate, tilted 3 degrees": bracket(tilt=3.0),
            "a corner 2.5 mm up": bracket(tilt=1.35, tilt_axis="x"),
            "boss down": bracket(z=PLATE_TOP_W + 0.035, tilt=180.0),
            "turned across the fixture, over one pin": bracket(x=L.FIX_X, y=SA[1] - L.HOLE_Y, yaw=90.0),
            "beside the fixture on the table": bracket(x=0.45, y=0.0, z=L.TOP),
        }
        for what, br in cases.items():
            with self.subTest(what):
                self.assertEqual(scored(dict(LOCATED, bracket=br)), 0.5)

    def test_turned_across_counts_only_one_pin(self):
        br = bracket(x=L.FIX_X, y=SA[1] - L.HOLE_Y, yaw=90.0)  # the +y hole now sits over socket A
        r = settled(dict(LOCATED, bracket=br))[1]
        self.assertTrue(r.details["bracket_flat"])
        self.assertFalse(r.details["bracket_over_fitted_pins"])
        self.assertEqual(r.details["pin_hole"][R1], None)

    def test_seat_tolerances(self):
        for kw, want in (({"z": PLATE_TOP_W + 0.0019}, 1.0), ({"z": PLATE_TOP_W + 0.0021}, 0.5),
                         ({"z": PLATE_TOP_W - 0.0019}, 1.0), ({"tilt": 1.0, "tilt_axis": "y"}, 1.0),
                         ({"tilt": 2.1, "tilt_axis": "y"}, 0.5)):
            with self.subTest(kw=kw):
                self.assertEqual(scored(dict(SEATED, bracket=bracket(**kw))), want)

    def test_anywhere_in_the_window_counts(self):
        for dy in (0.0045, 0.010, 0.0155):
            for dx in (-0.007, 0.0, 0.007):
                self.assertEqual(scored(dict(SEATED, bracket=bracket(x=L.FIX_X + dx, y=L.FIX_Y + dy))), 1.0)

    def test_spare_round_pin_anywhere_on_the_table(self):
        for spare in (round_pin(0.40, -0.10), round_pin(0.15, -0.20, floor=L.PIN_R, tilt=90.0, tilt_axis="x"),
                      on_tray(R2)):
            self.assertEqual(scored(dict(SEATED, round_pin_2=spare)), 1.0)

    def test_score_is_idempotent(self):
        tr, r = settled(SEATED)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(SEATED).score, 1.0)
        self.assertEqual(tr.steps, n)


class Fitted(unittest.TestCase):
    def test_upright_within_5_degrees(self):
        for tilt, want in ((4.9, 0.5), (5.1, 0.25)):
            st = scene(diamond_pin=in_socket("A", D, tilt=tilt), round_pin_1=in_socket("B", R1))
            self.assertEqual(scored(st), want, tilt)
            st = scene(diamond_pin=in_socket("A", D), round_pin_1=in_socket("B", R1, tilt=tilt, tilt_axis="y"))
            self.assertEqual(scored(st), want, tilt)

    def test_bottom_within_3_mm_of_the_floor(self):
        for floor, want in ((0.0029, 0.5), (0.0031, 0.25), (-0.0029, 0.5)):
            st = scene(diamond_pin=in_socket("A", D), round_pin_1=in_socket("B", R1, floor=floor))
            self.assertEqual(scored(st), want, floor)

    def test_bottom_inside_the_opening(self):
        for dy, want in ((0.0035, 0.5), (0.0274, 0.5), (0.0280, 0.25)):
            st = scene(diamond_pin=in_socket("A", D), round_pin_1=round_pin(SB[0], SB[1] - dy))
            self.assertEqual(scored(st), want, dy)

    def test_round_pin_either_end_down(self):
        st = scene(diamond_pin=in_socket("A", D), round_pin_1=in_socket("B", R1, flip=True), bracket=bracket())
        self.assertEqual(scored(st), 1.0)

    def test_diamond_head_down_is_not_fitted(self):
        st = scene(diamond_pin=diamond(*SA, floor=L.PIN_H, tilt=180.0), round_pin_1=in_socket("B", R1))
        r = settled(st)[1]
        self.assertEqual(r.score, 0.25)
        self.assertIsNone(r.details["fitted"][D])

    def test_pins_not_in_a_socket(self):
        cases = {
            "on the plate's top over a socket": round_pin(*SB, floor=L.FIX_TOP),
            "on the plate's top between the sockets": round_pin(L.FIX_X, L.FIX_Y, floor=L.FIX_TOP),
            "on the table in front of the fixture": round_pin(0.20, -0.05),
            "lying in the socket": round_pin(SB[0], SB[1], floor=L.PIN_R, tilt=90.0, tilt_axis="x"),
            "on its tray": on_tray(R1),
        }
        for what, pin in cases.items():
            with self.subTest(what):
                self.assertEqual(scored(scene(round_pin_1=pin)), 0.0)
                self.assertEqual(scored(scene(round_pin_1=pin, diamond_pin=in_socket("A", D))), 0.25)


class ZeroConditions(unittest.TestCase):
    def test_no_pin_fitted(self):
        for st in (START, scene(bracket=bracket()), scene(round_pin_1=round_pin(*SB, floor=L.FIX_TOP))):
            r = settled(st)[1]
            self.assertEqual(r.score, 0.0)
            self.assertEqual(r.details["zero_reason"], "no_pin_fitted")

    def test_anything_off_the_table(self):
        drops = {
            R2: round_pin(0.40, -0.70, base=0.0),
            R1: round_pin(0.95, 0.0),
            D: diamond(0.20, 0.30, base=L.TOP - 0.05),
            BRACKET: bracket(x=0.30, y=0.0, z=L.TOP - 0.02),
        }
        for name, st in drops.items():
            with self.subTest(name):
                tr, r = settled(dict(SEATED, **{name: st}))
                self.assertEqual(r.score, 0.0)
                self.assertEqual(r.details["zero_reason"], "off_the_table")
                self.assertIn(name, r.details["off_table"])

    def test_off_table_edges(self):
        self.assertFalse(off_table(R1, round_pin(0.30, 0.0, base=L.TOP - 0.0095)))
        self.assertTrue(off_table(R1, round_pin(0.30, 0.0, base=L.TOP - 0.0105)))
        self.assertTrue(off_table(R1, round_pin(0.30, L.TABLE_Y1 + 0.001)))
        self.assertTrue(off_table(R1, round_pin(L.TABLE_X0 - 0.001, 0.0)))


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [SEATED] * WINDOW)
        self.assertEqual(r.score, 0.0)  # nothing is at rest yet, so nothing is fitted
        self.assertEqual(tr.update(SEATED).score, 1.0)

    def test_moving_bracket_scores_a_half_until_at_rest(self):
        tr = Tracker(HZ)
        above = bracket(z=PLATE_TOP_W + 0.030)
        feed(tr, [dict(SEATED, bracket=above)] * (WINDOW + 1))
        seq = glide(dict(SEATED, bracket=above), SEATED, 6)
        for s in seq:
            self.assertLessEqual(tr.update(s).score, 0.5)
        scores = [tr.update(SEATED).score for _ in range(WINDOW + 1)]
        self.assertEqual(scores[0], 0.5)
        self.assertEqual(scores[-1], 1.0)
        self.assertEqual(scores.index(1.0), WINDOW - 1)  # 0.5 s after the last move of more than 2 mm

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        states = [dict(SEATED, bracket=bracket(y=L.FIX_Y + 0.010 + (0.0008 if i % 2 else -0.0008)))
                  for i in range(30)]
        self.assertEqual(feed(tr, states).score, 1.0)

    def test_moving_pin_is_not_fitted(self):
        tr, _ = settled(SEATED)
        r = tr.update(dict(SEATED, round_pin_1=round_pin(SB[0], SB[1] + 0.0025)))
        self.assertEqual(r.score, 0.25)  # 2.5 mm from where it was a step ago: moving, so not fitted
        self.assertIsNone(r.details["fitted"][R1])
        self.assertEqual(r.details["socket"][R1], "B")
        moved = round_pin(SB[0], SB[1] + 0.0025)
        r = feed(tr, [dict(SEATED, round_pin_1=moved)] * (WINDOW - 1))
        self.assertEqual(r.score, 0.25)  # still within 0.5 s of the move
        self.assertEqual(tr.update(dict(SEATED, round_pin_1=moved)).score, 1.0)

    def test_a_turn_of_the_bracket_counts_as_moving(self):
        tr, _ = settled(SEATED)
        r = tr.update(dict(SEATED, bracket=bracket(yaw=2.5)))  # holes move 2.7 mm
        self.assertEqual(r.score, 0.5)


class Exploits(unittest.TestCase):
    def test_bracket_held_still_just_above_the_pins(self):
        st = dict(LOCATED, bracket=bracket(z=L.TOP + L.PIN_H + 0.001))
        self.assertEqual(scored(st), 0.5)

    def test_bracket_flat_on_the_plate_beside_the_pins_does_not_count(self):
        """Laid flat at the plate's height with its holes off the pins (only possible if the pins are not
        under it): not seated."""
        st = dict(LOCATED, bracket=bracket(x=L.FIX_X + 0.080))
        self.assertEqual(scored(st), 0.5)

    def test_pins_set_on_the_plate_next_to_the_sockets(self):
        st = scene(diamond_pin=diamond(*SA, floor=L.FIX_TOP), round_pin_1=round_pin(*SB, floor=L.FIX_TOP),
                   bracket=bracket(z=PLATE_TOP_W + 0.040))
        self.assertEqual(scored(st), 0.0)

    def test_pins_tilted_into_the_bracket(self):
        """Pins leaned outward in their sockets so the bracket can come down over two round pins: they are no
        longer upright, so nothing is fitted."""
        st = scene(round_pin_1=in_socket("B", R1, tilt=-12.0), round_pin_2=in_socket("A", R2, tilt=12.0),
                   bracket=bracket())
        self.assertEqual(scored(st), 0.0)

    def test_diamond_laid_on_its_side_in_a_socket(self):
        st = scene(diamond_pin=diamond(SA[0], SA[1], floor=L.PIN_R, tilt=90.0, tilt_axis="y"),
                   round_pin_1=in_socket("B", R1), bracket=bracket())
        self.assertEqual(scored(st), 0.25)

    def test_bracket_teleport_shaped_shortcut_is_still_scored_by_pose(self):
        tr, _ = settled(LOCATED)
        r = feed(tr, [SEATED] * (WINDOW + 1))
        self.assertEqual(r.score, 1.0)  # a pose is a pose; the simulator decides whether it can be reached


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(SEATED)
        n = tr.steps
        hist = {k: [tuple(map(tuple, v)) for v in h] for k, h in tr.history.items()}
        bad_br = bracket()
        bad = [
            dict(SEATED, bracket=SimpleNamespace(**{**vars(bad_br), "origin": np.array([math.nan, 0.0, 0.8])})),
            dict(SEATED, round_pin_1=SimpleNamespace(**{**vars(SEATED[R1]), "pos": np.array([0.3, math.inf, 0.8])})),
            dict(SEATED, diamond_pin=SimpleNamespace(**{**vars(SEATED[D]), "quat_xyzw": np.zeros(4)})),
            dict(SEATED, diamond_pin=SimpleNamespace(**{**vars(SEATED[D]),
                                                       "quat_xyzw": np.array([math.nan, 0, 0, 1.0])})),
            dict(SEATED, bracket=SimpleNamespace(**{k: v for k, v in vars(bad_br).items() if k != "origin"})),
            dict(SEATED, round_pin_2=SimpleNamespace(pos=np.zeros(2), quat_xyzw=np.array([0, 0, 0, 1.0]),
                                                     low=np.zeros(3))),
            dict(SEATED, round_pin_1=None),
            {k: v for k, v in SEATED.items() if k != BRACKET},
        ]
        for b in bad:
            r = tr.update(b)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(tr.steps, n + len(bad))
        self.assertEqual({k: [tuple(map(tuple, v)) for v in h] for k, h in tr.history.items()}, hist)
        self.assertEqual(tr.update(SEATED).score, 1.0)  # and it carries on

    def test_valid(self):
        self.assertTrue(valid(R1, SEATED[R1]))
        self.assertTrue(valid(D, SEATED[D]))
        self.assertFalse(valid(D, SimpleNamespace(**{k: v for k, v in vars(SEATED[D]).items() if k != "origin"})))
        self.assertTrue(valid(R1, SimpleNamespace(**{k: v for k, v in vars(SEATED[R1]).items()})))
        self.assertFalse(valid(BRACKET, None))

    def test_unnormalised_quaternions(self):
        st = {k: SimpleNamespace(**{**vars(v), "quat_xyzw": 3.0 * np.asarray(v.quat_xyzw)}) for k, v in SEATED.items()}
        self.assertEqual(scored(st), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(11)
        tr = Tracker(HZ)
        for _ in range(300):
            st = {}
            for n in NAMES:
                q = rng.normal(size=4)
                o = rng.uniform([-0.3, -0.8, 0.0], [1.0, 0.8, 1.3])
                pts = {R1: ROUND_PTS, R2: ROUND_PTS, D: DIAMOND_PTS, BRACKET: BRACKET_PTS}[n]
                w = _world(pts, q, o)
                st[n] = SimpleNamespace(origin=o, pos=o if n in L.ROUND else (w.min(0) + w.max(0)) / 2,
                                        quat_xyzw=q, low=w.min(0), high=w.max(0), lin_vel=np.zeros(3),
                                        ang_vel=np.zeros(3))
            r = tr.update(st)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(SEATED)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertTrue(all(len(h) == 1 for h in tr.history.values()))
        self.assertEqual(r.score, 0.0)

    def test_reset_then_a_new_trial(self):
        tr, _ = settled(SEATED)
        tr.update(START)
        self.assertEqual(feed(tr, [SEATED] * WINDOW).score, 0.0)  # not yet 0.5 s of the new trial
        self.assertEqual(tr.update(SEATED).score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(SEATED)
        jit = scene(bracket=bracket(L.BR_START[0] + 0.03, L.BR_START[1] - 0.03, z=L.TOP),
                    round_pin_1=round_pin(L.ROUND_START[R1][0] - 0.03, L.ROUND_START[R1][1], base=L.TOP + L.TRAY_TOP))
        self.assertTrue(Tracker.at_start(jit))
        tr.update(jit)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(SEATED)
        feed(tr, glide(SEATED, START, 40))
        self.assertEqual(tr.steps, WINDOW + 1 + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(SEATED)
        tr.update(LOCATED)  # the bracket jumps back to its start, but the pins are in the sockets
        self.assertEqual(tr.steps, WINDOW + 2)

    def test_start_needs_every_part_on_its_mark(self):
        self.assertFalse(Tracker.at_start(scene(diamond_pin=diamond(*L.DIAMOND_START, base=L.TOP + L.TRAY_TOP))))
        self.assertFalse(Tracker.at_start(scene(round_pin_2=round_pin(0.22, -0.20, base=L.TOP + L.TRAY_TOP))))
        self.assertFalse(Tracker.at_start(scene(bracket=bracket(*L.BR_START, z=L.TOP + 0.01))))

    def test_explicit_reset(self):
        tr, _ = settled(SEATED)
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertTrue(all(len(h) == 0 for h in tr.history.values()))
        self.assertEqual(tr.prev_points, {})


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the
    demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim(self, st):
        """A synthetic scene with the two compounds turned into the AssemblyStates mechkit would report."""
        out = dict(st)
        for name, spec in ((D, self.task.DIAMOND_PIN), (BRACKET, self.task.BRACKET_PART)):
            s = st[name]
            out[name] = self.mk.state_from_poses(spec, (*s.origin, *s.quat_xyzw), source="sim")
        return out

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = {o.name: o.mass for o in t.OBJECTS if o.mass is not None}
        self.assertEqual(set(loose), {R1, R2, D, BRACKET})
        self.assertTrue(all(0 < m <= 0.30 for m in loose.values()))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 2.5)
        self.assertEqual(len(t.FIXTURE), 5)
        for o in t.FIXTURE:
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)
            self.assertAlmostEqual(o.size[2], L.FIX_H)
        for o in t.ROUND_PINS:
            self.assertAlmostEqual(o.radius, L.PIN_R)
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP + L.TRAY_TOP)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.20 m)
        for o in t.OBJECTS:
            self.assertLess(abs(o.pos[1]), 0.40, o.name)

    def test_scorer_geometry_matches_mechkit(self):
        lo, hi = self.task.DIAMOND_PIN.local_bounds()
        np.testing.assert_allclose(lo, [-L.PIN_R, -L.PIN_R, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.PIN_R, L.PIN_R, L.PIN_H], atol=1e-12)
        head = [s for s in self.task.DIAMOND_PIN.shapes if s.kind == "box"][0]
        self.assertEqual(tuple(head.size), L.HEAD_SIZE)
        self.assertAlmostEqual(head.center[2] - head.size[2] / 2, L.SHANK_H)  # the head starts at the plate's top
        self.assertAlmostEqual(L.SHANK_H, L.FIX_H)
        lo, hi = self.task.BRACKET_PART.local_bounds()
        np.testing.assert_allclose(lo, [-L.BR_LEN_X / 2, -L.BR_LEN_Y / 2, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.BR_LEN_X / 2, L.BR_LEN_Y / 2, L.BR_T + L.BOSS_SIZE[2]], atol=1e-12)
        # spawn poses are the starting marks the scorer expects
        rest = scene(diamond_pin=self.task.DIAMOND_PIN.rest_state(), bracket=self.task.BRACKET_PART.rest_state())
        self.assertTrue(Tracker.at_start(rest))
        # mechkit's states read as the synthetic ones
        st = self.sim(SEATED)
        np.testing.assert_allclose(st[BRACKET].low[2], PLATE_TOP_W, atol=1e-12)
        np.testing.assert_allclose(st[D].low[2], L.TOP, atol=1e-12)
        self.assertEqual(scored(st), 1.0)
        np.testing.assert_allclose(st[BRACKET].low, SEATED[BRACKET].low, atol=1e-12)
        np.testing.assert_allclose(st[D].high, SEATED[D].high, atol=1e-12)

    def test_the_diamond_stands_on_its_shank(self):
        com = self.task.DIAMOND_PIN.center_of_mass()
        self.assertAlmostEqual(com[0], 0.0)
        self.assertAlmostEqual(com[1], 0.0)
        self.assertLess(com[2], 0.020)  # 17.5 mm up on a 48 mm base: stable standing
        com = self.task.BRACKET_PART.center_of_mass()
        np.testing.assert_allclose(com[:2], [0.0, 0.0], atol=1e-12)  # balanced about the boss it is held by

    def test_offline_state_path(self):
        for spec, mark in ((self.task.DIAMOND_PIN, (*L.DIAMOND_START, L.TOP + L.TRAY_TOP)),
                           (self.task.BRACKET_PART, (*L.BR_START, L.TOP))):
            c, _, yaw = spec.offline_box()
            box = SimpleNamespace(pos=np.array(c), quat_xyzw=about("z", math.degrees(yaw)), lin_vel=np.zeros(3),
                                  ang_vel=np.zeros(3))
            st = spec.offline_state(box)
            self.assertEqual(st.source, "offline")
            np.testing.assert_allclose(st.origin, mark, atol=1e-12)

    def test_hooks_and_mechkit_reset(self):
        t = self.task
        tr = t._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        seated = self.sim(SEATED)
        for _ in range(WINDOW):
            t.success(seated)
        self.assertTrue(t.success(seated))
        self.assertEqual(t.score(seated), 1.0)
        self.assertEqual(t.score(seated), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(t.success(seated))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        self.assertEqual(t.score(self.sim(LOCATED)), 0.0)  # 1 step of history: nothing at rest yet

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                for _ in range(WINDOW + 1):
                    self.task.success(self.sim(SEATED))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "tier", "zero_reason", "fitted", "located_set",
                          "diamond_turn_err_deg", "bracket_seated", "bracket_flat", "bracket_over_fitted_pins",
                          "bracket_tilt_deg", "bracket_corner_mm", "bracket_x_mm", "bracket_y_mm", "pin_hole",
                          "pin_hole_offset_mm", "socket", "tilt_deg", "floor_mm", "centre_off_mm", "moved_mm",
                          "off_table", "source", "diamond_origin", "diamond_quat", "bracket_origin", "bracket_quat",
                          "bracket_holes", "pin_bottom"):
                    self.assertIn(k, row)
                self.assertEqual(row["score"], 1.0)
                self.assertAlmostEqual(row["bracket_y_mm"], 10.0)
                self.assertEqual(row["fitted"], {R1: "B", R2: None, D: "A"})
                self.assertEqual(row["source"], "sim")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertTrue(self.task.success(self.sim(SEATED)))  # no crash
                bad = dict(self.sim(SEATED), bracket=None)
                os.environ["TASK_TRACE"] = str(path)
                self.assertFalse(self.task.success(bad))
                self.assertTrue(json.loads(path.read_text().splitlines()[-1])["state_error"])
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        picks = {t.DIAMOND_PICK[2], t.ROUND_PICK[2], t.BR_PICK[2]}
        places = {t._pin_place("A", t.TRIM["diamond"])[2], t._pin_place("B", t.TRIM["round"])[2], t.BR_PLACE[2]}
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes 1.5 s, in place
                    self.assertGreaterEqual(w.duration, 1.5)
                    self.assertEqual(prev.pos, w.pos)
                if prev.pos[2] > w.pos[2] and any(abs(w.pos[2] - z) < 1e-9 for z in picks | places):
                    self.assertGreaterEqual(w.duration, 1.5)  # final descents are slow
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.0245 + 1e-9)
        left, right = plan["left"], plan["right"]
        # the diamond: picked with its head's long side along y (yaw 0), set in A turned a quarter turn
        grip = [w for w in left if w.label == "grip the diamond pin"][0]
        into = [w for w in left if w.label == "slowly lower it into socket A"][0]
        self.assertEqual(grip.yaw, 0.0)
        self.assertAlmostEqual(into.yaw, -math.pi / 2)
        self.assertAlmostEqual(math.degrees(L.DIAMOND_START_YAW + (into.yaw - grip.yaw)) % 180.0, 0.0)  # along x
        # pins let go 2 mm above the socket floor, the pad tips 10 mm over the plate's top (clear of it)
        for wps, label, k in ((left, "let go in socket A", "A"), (right, "let go in socket B", "B")):
            w = [w for w in wps if w.label == label][0]
            self.assertLess(math.hypot(w.pos[0] - L.SOCKETS[k][0], w.pos[1] - L.SOCKETS[k][1]), 0.0015)
            self.assertAlmostEqual(w.pos[2] - t.PIN_GRIP_UP - L.TOP, t.DROP)
            self.assertLessEqual(t.DROP, 0.0025)  # let go inside the socket, under the 3 mm floor tolerance
            self.assertGreater(w.pos[2] - t.PAD_TIP - (L.TOP + L.FIX_TOP), 0.010)
        # the bracket aimed at the middle of its 12 mm window, its underside 1 mm over the plate
        lo, hi = t.BR_PLACE[1] - L.FIX_Y, t.BR_PLACE[1] - L.FIX_Y
        self.assertTrue(L.WINDOW_LO + 0.004 <= lo and hi <= L.WINDOW_HI - 0.004)
        self.assertAlmostEqual(t.BR_PLACE[2] - t.BR_GRIP_UP - PLATE_TOP_W, t.BR_DROP)
        # carried with its underside over the pin heads
        carry = [w for w in left if w.label.startswith("carry it over the fixture")][0]
        self.assertGreater(carry.pos[2] - t.BR_GRIP_UP - (L.TOP + L.PIN_H), 0.025)
        # the arms take turns at the fixture: the right arm comes over socket B only after the left arm has left A
        # and gone back up; the left arm brings the bracket only after the right arm has left B
        def times(wps):
            out, acc = {}, 0.0
            for w in wps:
                acc += w.duration
                out.setdefault(w.label, acc)
            return out
        tl, trt = times(left), times(right)
        self.assertGreaterEqual(trt["carry it over socket B"] - 2.0, tl["straight up"])
        self.assertGreaterEqual(tl["carry it over the fixture, holes over the pins"] - 2.5, trt["aside"] - 1.5)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_grips(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.PIN_GRIP, 2 * L.PIN_R - 0.008)  # 8 mm narrower than the pins and head
        self.assertAlmostEqual(-2 * t.BOSS_GRIP, L.BOSS_SIZE[0] - 0.008)
        self.assertGreater(-2 * t.OPEN, L.BR_LEN_X)  # the open fingers come down outside the bracket's plate
        # pad tips 25 mm above a pin's bottom: on the diamond, 10 mm up its head
        self.assertAlmostEqual(t.PIN_GRIP_UP - t.PAD_TIP - L.SHANK_H, 0.010)
        # pad tips 4 mm above the bracket's top face at the pick
        self.assertAlmostEqual(t.BR_GRIP_UP - t.PAD_TIP - L.BR_T, 0.004)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("round pin", "diamond pin", "flat sides face the round pin", "two holes", "over both pins"):
            self.assertIn(s, ins)
        data = json.loads((HERE / "task.json").read_text())
        self.assertEqual(data["instruction"], ins)

    def test_task_json_numbers_match_the_layout(self):
        data = json.loads((HERE / "task.json").read_text())
        setup, scoring = data["setup"], data["scoring"]
        for s in ("55 x 55 mm", "48 mm across and 55 mm tall", "100 g each", "80 g", "190 g",
                  "48 mm long, 8 mm wide and 40 mm tall", "15 mm tall", "88 mm (x) by 212 mm (y) and 10 mm thick",
                  "64 x 64 mm", "124 mm apart", "30 mm (x) by 40 mm (y) and 25 mm tall", "3.5 mm of play",
                  "8 mm of clearance", "28 mm along the line", "4 to 16 mm", "x 22.0 cm, y -23.5 cm",
                  "x 22.0 cm, y -29.5 cm", "x 23.5 cm, y +25.0 cm", "x 13.5 cm, y +11.0 cm", "x 30.0 cm"):
            self.assertIn(s, setup)
        for s in ("5 degrees", "3 mm", "15 degrees", "2 degrees", "2 mm", "10 mm", "0.5 s", "2.5 s"):
            self.assertIn(s, scoring)
        self.assertEqual(data["run_limit"], "1,200 steps (60 s at 20 Hz)")


if __name__ == "__main__":
    unittest.main()
