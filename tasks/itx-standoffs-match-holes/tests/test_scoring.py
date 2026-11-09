"""Offline tests for scoring.py and task.py: synthetic standoff and board states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import itertools
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
    BOARD,
    NAMES,
    TRAY_TOP_W,
    Tracker,
    board_holes,
    board_yaw_deg,
    is_stray,
    nearest_hole,
    off_table,
    outside_outline,
    outside_tray,
    over_tray,
    rotation,
    standoff_axis,
    standoff_pose,
    under_board,
    valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
S1, S2, S3, S4, S5 = L.STANDOFFS
DISC_W = L.TOP + L.DISC_TOP  # a standoff's bottom standing on a hole's disc (world z)
DISH = (L.DISH_X, L.DISH_Y)
DISH_SPOTS = [(L.DISH_X - 0.025, L.DISH_Y), (L.DISH_X + 0.025, L.DISH_Y), (L.DISH_X, L.DISH_Y + 0.025)]


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


def _rim(r, z, n=24, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def _box_corners(center, size):
    c, s = np.asarray(center, float), np.asarray(size, float) / 2
    return [tuple(c + s * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]


STANDOFF_PTS = np.array(_rim(L.STANDOFF_R, -L.STANDOFF_H / 2) + _rim(L.STANDOFF_R, L.STANDOFF_H / 2))
BOARD_PTS = np.array(
    _box_corners((0.0, 0.0, L.BOARD_T / 2), (L.BOARD_LEN, L.BOARD_LEN, L.BOARD_T))
    + _box_corners(L.COOLER_CENTER, L.COOLER_SIZE)
    + _box_corners(L.IO_CENTER, L.IO_SIZE)
    + [p for hx, hy in L.BOARD_HOLES.values() for p in _rim(L.PAD_R, L.BOARD_T + L.PAD_H, cx=hx, cy=hy)]
)


def _world(points, q, origin):
    return np.asarray(points) @ rotation(q).T + np.asarray(origin, dtype=float)


def standoff(x, y, floor=0.0, yaw=0.0, tilt=0.0, tilt_axis="x", flip=False, base=DISC_W):
    """A plain cylinder (yam_arena reports its centre) with its lower end's centre at (x, y), ``floor`` m above
    ``base`` while upright; tilted about that point. ``flip``: the same standoff the other end down."""
    q = pose_quat(yaw, tilt, tilt_axis)
    if flip:
        q = qmul(q, about("x", 180.0))
    bottom = np.array([x, y, base + floor])
    a = rotation(q)[:, 2]
    a = a if a[2] >= 0 else -a
    c = bottom + a * (L.STANDOFF_H / 2)
    pts = _world(STANDOFF_PTS, q, c)
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pts.min(axis=0),
                           high=pts.max(axis=0))


def lying(x, y, axis_yaw=0.0):
    """A standoff lying on its side on the tray, its centre at (x, y)."""
    q = qmul(about("z", axis_yaw), about("y", 90.0))
    c = np.array([x, y, TRAY_TOP_W + L.STANDOFF_R])
    pts = _world(STANDOFF_PTS, q, c)
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pts.min(axis=0),
                           high=pts.max(axis=0))


def on(k, **kw):
    """A standoff standing centred on hole ``k``."""
    return standoff(*L.HOLES[k], **kw)


def board(x=L.MOUNT_X, y=L.MOUNT_Y, z=L.TOP + L.MOUNT_Z, yaw=0.0, tilt=0.0, tilt_axis="x"):
    """The board as mechkit reports it: base frame at the centre of its underside at (x, y, z); ``yaw`` the
    direction of its I/O edge (0: toward +x, the rear)."""
    q = pose_quat(yaw, tilt, tilt_axis)
    o = np.array([x, y, z])
    pts = _world(BOARD_PTS, q, o)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=lo, high=hi)


def board_on_mat():
    return board(*L.BOARD_START, z=L.TOP + L.MAT_TOP)


def scene(**parts):
    """The starting layout with some parts replaced."""
    s = {n: on(L.STANDOFF_START[n]) for n in L.STANDOFFS}
    s[BOARD] = board_on_mat()
    s.update(parts)
    return s


def in_dish(i=0):
    return standoff(*DISH_SPOTS[i], base=L.TOP + L.DISH_SIZE[2])


START = scene()
READY = scene(standoff_3=on("H3"), standoff_4=on("H4"))  # all four holes filled, board still on its mat
FINAL = scene(standoff_3=on("H3"), standoff_4=on("H4"), board=board())


def settled(state, n=WINDOW + 1, tracker=None):
    """Feed the same state ``n`` times (at rest for 0.5 s); return the tracker and its last result."""
    tr = tracker or Tracker(HZ)
    res = None
    for _ in range(n):
        res = tr.update(state)
    return tr, res


def scored(state):
    return settled(state)[1].score


def details(state):
    return settled(state)[1].details


def best_shift(yaw_deg):
    """The board turned ``yaw_deg`` and shifted (1 mm grid, +-30 mm) so that its holes come nearest the four
    mini-ITX holes: (largest hole-to-nearest-hole distance, shift)."""
    r = rotation(about("z", yaw_deg))[:2, :2]
    pts = [r @ np.array(h) for h in L.BOARD_HOLES.values()]
    targets = [np.array(v) - np.array([L.MOUNT_X, L.MOUNT_Y]) for v in L.ITX_HOLES.values()]
    best = (1e9, None)
    for dx in np.arange(-0.030, 0.0305, 0.001):
        for dy in np.arange(-0.030, 0.0305, 0.001):
            s = np.array([dx, dy])
            err = max(min(np.linalg.norm(p + s - t) for t in targets) for p in pts)
            if err < best[0]:
                best = (err, s)
    return best


# -- geometry --------------------------------------------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        self.assertTrue(Tracker.at_start(START))
        tr, res = settled(START)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["zero_reason"], "stray_standoff")
        self.assertEqual(res.details["strays"], [S3, S4])  # the ones on A1 and A2
        self.assertEqual(res.details["on_hole"], {"H1": S1, "H2": S2, "H3": None, "H4": None})

    def test_holes_and_outline(self):
        x0, x1, y0, y1 = L.OUTLINE
        self.assertAlmostEqual(x1 - x0, L.BOARD_LEN)
        self.assertAlmostEqual(y1 - y0, L.BOARD_LEN)
        for k in ("H1", "H2", "H3", "H4", "A1", "A2"):  # inside the outline
            self.assertEqual(outside_outline(L.HOLES[k]), 0.0, k)
        self.assertGreater(outside_outline(L.HOLES["A3"]), L.STRAY_MARGIN + 0.02)  # 31 mm outside
        self.assertGreater(outside_outline(L.HOLES["A3"]) - L.DISC_R, L.STANDOFF_R)  # a standoff on it clears the board
        self.assertEqual(L.SPARE_HOLES, ("A3",))
        # the board's outline lies well inside the tray: a standoff under the board is always in the case
        tx0, tx1, ty0, ty1 = L.TRAY_OUTLINE
        self.assertTrue(tx0 + 0.04 < x0 and x1 < tx1 - 0.04 and ty0 + 0.05 < y0 and y1 < ty1 - 0.05)
        for k in L.HOLES:
            self.assertEqual(outside_tray(L.HOLES[k]), 0.0, k)
        # the mat and the dish are off the tray, by more than a standoff's radius
        for (cx, cy), (sx, sy, _) in (((L.MAT_X, L.MAT_Y), L.MAT_SIZE), ((L.DISH_X, L.DISH_Y), L.DISH_SIZE)):
            for px in (cx - sx / 2, cx + sx / 2):
                for py in (cy - sy / 2, cy + sy / 2):
                    self.assertGreater(outside_tray((px, py)), L.STRAY_MARGIN)
        # every disc lies on the tray, and no two discs touch
        cx0, cx1 = L.CASE_X - L.CASE_SIZE[0] / 2, L.CASE_X + L.CASE_SIZE[0] / 2
        cy0, cy1 = L.CASE_Y - L.CASE_SIZE[1] / 2, L.CASE_Y + L.CASE_SIZE[1] / 2
        for k, (x, y) in L.HOLES.items():
            self.assertTrue(cx0 + L.DISC_R <= x <= cx1 - L.DISC_R and cy0 + L.DISC_R <= y <= cy1 - L.DISC_R, k)
        for (a, pa), (b, pb) in itertools.combinations(L.HOLES.items(), 2):
            self.assertGreaterEqual(math.dist(pa, pb), 2 * L.DISC_R + 0.0019, (a, b))  # 2 mm apart at least
        # a standoff's base is wholly on its hole's disc exactly when it is within the 5 mm tolerance
        self.assertAlmostEqual(L.DISC_R - L.STANDOFF_R, L.ON_HOLE_TOL)

    def test_board_holes_over_the_itx_holes(self):
        holes = board_holes(board())
        for k, (x, y) in L.ITX_HOLES.items():
            np.testing.assert_allclose(holes[k], [x, y, L.TOP + L.MOUNT_Z], atol=1e-12)
        self.assertAlmostEqual(board_yaw_deg(board()), 0.0)
        self.assertAlmostEqual(abs(board_yaw_deg(board(yaw=180))), 180.0)

    def test_pattern_is_not_a_rectangle(self):
        """The fourth corner of the rectangle through H1, H2 and H3 is A2: guessing a rectangle leaves A2's standoff
        under the board and H4 empty."""
        h1, h2, h3 = (np.array(L.ITX_HOLES[k]) for k in ("H1", "H2", "H3"))
        np.testing.assert_allclose(h3 + (h2 - h1), L.ATX_HOLES["A2"], atol=1e-12)
        self.assertAlmostEqual(math.dist(L.ITX_HOLES["H4"], L.ATX_HOLES["A2"]), 0.030)

    def test_pattern_fits_only_one_way_round(self):
        """Turned a quarter or half turn, the board's holes cannot all come within 7 mm of the four holes, wherever
        it is shifted: only the I/O-to-the-rear orientation mounts."""
        self.assertLess(best_shift(0.0)[0], 1e-9)
        for yaw in (90.0, 180.0, 270.0):
            err, _ = best_shift(yaw)
            self.assertGreater(err, 0.0149, yaw)  # one hole at least 15 mm off (21 for a quarter turn)

    def test_board_holes_inside_the_board(self):
        for k, (hx, hy) in L.BOARD_HOLES.items():
            self.assertLessEqual(max(abs(hx), abs(hy)) + L.PAD_R, L.BOARD_HALF, k)
        # the cooler and the I/O block clear the pads
        for hx, hy in L.BOARD_HOLES.values():
            for c, s in ((L.COOLER_CENTER, L.COOLER_SIZE), (L.IO_CENTER, L.IO_SIZE)):
                dx = max(abs(hx - c[0]) - s[0] / 2, 0.0)
                dy = max(abs(hy - c[1]) - s[1] / 2, 0.0)
                self.assertGreater(math.hypot(dx, dy), L.PAD_R)

    def test_masses(self):
        solid_brass = 8.5e3 * math.pi * L.STANDOFF_R ** 2 * L.STANDOFF_H
        self.assertLess(L.STANDOFF_MASS, solid_brass)
        self.assertLessEqual(L.BOARD_MASS, 0.30)

    def test_standoff_measures(self):
        st = on("H3")
        bottom, top, tilt = standoff_axis(st)
        np.testing.assert_allclose(bottom, [*L.ITX_HOLES["H3"], DISC_W], atol=1e-12)
        np.testing.assert_allclose(top, [*L.ITX_HOLES["H3"], L.TOP + L.STANDOFF_TOP], atol=1e-12)
        self.assertAlmostEqual(tilt, 0.0)
        p = standoff_pose(st)
        self.assertEqual(p["itx_hole"], "H3")
        self.assertAlmostEqual(p["floor"], L.DISC_H)
        self.assertEqual(nearest_hole(L.HOLES["A1"]), ("A1", 0.0))


# -- tiers -------------------------------------------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def test_every_tier_is_reachable(self):
        cases = {
            0.0: START,
            0.2: scene(standoff_1=in_dish(2), standoff_3=in_dish(0), standoff_4=in_dish(1)),  # only H2 left
            0.4: scene(standoff_3=in_dish(0), standoff_4=in_dish(1)),  # A1 and A2 to the dish
            0.6: scene(standoff_3=on("H3"), standoff_4=in_dish(1)),
            0.8: READY,
            1.0: FINAL,
        }
        for want, st in cases.items():
            self.assertAlmostEqual(scored(st), want, msg=str(want))

    def test_the_other_way_to_0_6(self):
        self.assertAlmostEqual(scored(scene(standoff_4=on("H4"), standoff_3=in_dish(0))), 0.6)
        self.assertAlmostEqual(scored(scene(standoff_3=on("H4"), standoff_4=in_dish(0))), 0.6)  # A1's onto H4

    def test_only_one_is_success(self):
        for st in (START, READY, scene(standoff_3=on("H3"), standoff_4=in_dish(1))):
            self.assertFalse(settled(st)[1].success)
        self.assertTrue(settled(FINAL)[1].success)

    def test_each_hole_counts_the_same(self):
        empty = {n: in_dish(0) if i == 0 else standoff(0.15, 0.20 - 0.03 * i, base=L.TOP) for i, n in
                 enumerate(L.STANDOFFS)}  # on the table, off the tray
        self.assertEqual(scored(scene(**empty)), 0.0)
        self.assertEqual(details(scene(**empty))["zero_reason"], "no_hole_filled")
        for k in L.ITX:
            self.assertAlmostEqual(scored(scene(**dict(empty, standoff_1=on(k)))), 0.2, msg=k)

    def test_any_standoff_on_any_hole(self):
        for perm in itertools.permutations(L.STANDOFFS[:4]):
            st = scene(**{n: on(k) for n, k in zip(perm, L.ITX)}, board=board())
            self.assertEqual(scored(st), 1.0)

    def test_a3_may_stay_or_go(self):
        for spot in (on("A3"), on("A3", flip=True), in_dish(0), standoff(0.15, 0.05, base=L.TOP),
                     lying(0.15, -0.05)):  # standing on its hole, in the dish, on the table off the tray
            self.assertEqual(scored(dict(FINAL, standoff_5=spot)), 1.0)

    def test_a3_knocked_over_or_off_its_hole_is_loose(self):
        x, y = L.HOLES["A3"]
        for spot in (lying(x, y), on("A3", tilt=10.1), standoff(x + 0.0051, y), standoff(x, y, floor=0.004)):
            st = dict(FINAL, standoff_5=spot)
            self.assertEqual(scored(st), 0.0)
            self.assertEqual(details(st)["strays"], [S5])
        self.assertEqual(scored(dict(FINAL, standoff_5=standoff(x + 0.0049, y))), 1.0)

    def test_a3_moved_onto_a_hole(self):
        st = scene(standoff_5=on("H3"), standoff_3=on("H4"), standoff_4=in_dish(0), board=board())
        self.assertEqual(scored(st), 1.0)

    def test_mounted_needs_all_four(self):
        st = dict(FINAL, standoff_4=in_dish(0))  # H4 empty, the board held flat in place
        self.assertAlmostEqual(scored(st), 0.6)
        self.assertFalse(details(st)["mounted"])

    def test_board_on_the_standoffs_as_they_start_is_zero(self):
        st = scene(board=board())
        self.assertEqual(scored(st), 0.0)
        self.assertEqual(details(st)["zero_reason"], "stray_standoff")

    def test_board_mount_tolerances(self):
        tol = L.BOARD_HOLE_TOL
        for d, want in ((tol - 0.0001, 1.0), (tol + 0.0001, 0.8)):
            self.assertAlmostEqual(scored(dict(READY, board=board(x=L.MOUNT_X + d))), want, msg=d)
            self.assertAlmostEqual(scored(dict(READY, board=board(y=L.MOUNT_Y - d))), want, msg=d)
        for t, want in ((2.9, 1.0), (3.1, 0.8)):  # level within 3 degrees, about the board's hole line
            st = dict(READY, board=board(tilt=t, tilt_axis="y"))
            self.assertAlmostEqual(scored(st), want if t < 3.0 else 0.8, msg=t)
        for dz, want in ((0.0019, 1.0), (0.0021, 0.8), (-0.0019, 1.0)):
            self.assertAlmostEqual(scored(dict(READY, board=board(z=L.TOP + L.MOUNT_Z + dz))), want, msg=dz)

    def test_tilted_board_detail(self):
        d = details(dict(READY, board=board(tilt=2.0, tilt_axis="y")))
        self.assertAlmostEqual(d["board_tilt_deg"], 2.0, places=3)
        self.assertTrue(d["mounted"])

    def test_board_turned_round_does_not_mount(self):
        for yaw in (90.0, 180.0, 270.0):
            err, s = best_shift(yaw)
            st = dict(READY, board=board(x=L.MOUNT_X + s[0], y=L.MOUNT_Y + s[1], yaw=yaw))
            self.assertAlmostEqual(scored(st), 0.8, msg=yaw)
            self.assertFalse(details(st)["mounted"])

    def test_small_turn_still_mounts(self):
        self.assertEqual(scored(dict(READY, board=board(yaw=5.0))), 1.0)
        self.assertAlmostEqual(scored(dict(READY, board=board(yaw=10.0))), 0.8)  # a hole 7.5 mm+ off

    def test_board_elsewhere_or_upside_down(self):
        for b in (board_on_mat(), board(x=0.30, y=0.0, z=L.TOP + L.TRAY_TOP), board(tilt=180.0),
                  board(x=0.20, y=0.20, z=L.TOP + L.DISH_SIZE[2])):
            self.assertAlmostEqual(scored(dict(READY, board=b)), 0.8)

    def test_score_is_idempotent(self):
        tr, res = settled(FINAL)
        self.assertEqual(tr.result(FINAL).score, 1.0)
        self.assertEqual(tr.result(FINAL).score, 1.0)
        self.assertEqual(tr.steps, WINDOW + 1)


# -- standing on a hole ----------------------------------------------------------------------------------------------


class OnHole(unittest.TestCase):
    def test_upright_within_10_degrees(self):
        for t, want in ((9.9, 1.0), (10.1, 0.0)):  # tilted past 10 degrees it is a stray under the board
            st = dict(FINAL, standoff_3=on("H3", tilt=t))
            self.assertAlmostEqual(scored(dict(st, board=board(z=L.TOP + L.MOUNT_Z))), want if t < 10 else 0.0,
                                   msg=t)

    def test_bottom_within_3_mm_of_the_tray(self):
        self.assertTrue(standoff_pose(standoff(*L.HOLES["H3"], base=TRAY_TOP_W, floor=0.0029))["itx_hole"])
        self.assertIsNone(standoff_pose(standoff(*L.HOLES["H3"], base=TRAY_TOP_W, floor=0.0031))["itx_hole"])
        self.assertTrue(standoff_pose(standoff(*L.HOLES["H3"], base=TRAY_TOP_W, floor=-0.0029))["itx_hole"])

    def test_within_5_mm_of_the_hole(self):
        x, y = L.HOLES["H3"]
        for d, want in ((0.0049, 0.8), (0.0051, 0.0)):  # 5.1 mm off: not on it, and under the board: a stray
            st = dict(READY, standoff_3=standoff(x + d * 0.6, y - d * 0.8))
            self.assertAlmostEqual(scored(st), want, msg=d)
        d = details(dict(READY, standoff_3=standoff(x + 0.0051, y)))
        self.assertEqual(d["strays"], [S3])

    def test_either_end_down(self):
        self.assertEqual(scored(dict(FINAL, standoff_3=on("H3", flip=True))), 1.0)

    def test_on_a_hole_without_its_disc_height(self):
        """A standoff knocked half off its disc stands on the tray, 1 mm lower: still on the hole if within 5 mm."""
        x, y = L.HOLES["H4"]
        self.assertEqual(scored(dict(FINAL, standoff_4=standoff(x + 0.004, y, base=TRAY_TOP_W))), 1.0)

    def test_standing_on_an_atx_hole_is_not_credit(self):
        p = standoff_pose(on("A1"))
        self.assertEqual(p["nearest"], "A1")
        self.assertIsNone(p["itx_hole"])


# -- strays ---------------------------------------------------------------------------------------------------------


class Strays(unittest.TestCase):
    def test_margin_is_one_radius_outside_the_tray(self):
        x0, x1, y0, y1 = L.TRAY_OUTLINE
        for y, want in ((y1 + L.STRAY_MARGIN - 0.0001, 0.0), (y1 + L.STRAY_MARGIN + 0.0001, 1.0)):
            st = dict(FINAL, standoff_5=standoff(0.30, y, base=L.TOP))  # on the table beside the tray
            self.assertEqual(scored(st), want, msg=y)
        for y, want in ((y0 - L.STRAY_MARGIN + 0.0001, 0.0), (y0 - L.STRAY_MARGIN - 0.0001, 1.0)):
            self.assertEqual(scored(dict(FINAL, standoff_5=standoff(0.30, y, base=L.TOP))), want, msg=y)
        for x, want in ((x0 - L.STRAY_MARGIN + 0.0001, 0.0), (x0 - L.STRAY_MARGIN - 0.0001, 1.0)):
            self.assertEqual(scored(dict(FINAL, standoff_5=standoff(x, 0.0, base=L.TOP))), want, msg=x)
        # off a corner the distance is diagonal
        c = x0 - 0.0062, y1 + 0.0062  # 8.8 mm from the corner
        self.assertEqual(scored(dict(FINAL, standoff_5=standoff(*c, base=L.TOP))), 0.0)
        c = x0 - 0.0066, y1 + 0.0066  # 9.3 mm
        self.assertEqual(scored(dict(FINAL, standoff_5=standoff(*c, base=L.TOP))), 1.0)

    def test_loose_on_the_tray_clear_of_the_board_is_a_stray(self):
        """A standoff left loose in the case scores 0 even where the board does not cover it: upright off any hole,
        lying down, or standing on the bare tray between holes, beside the board's outline."""
        x0, x1, y0, y1 = L.OUTLINE
        for spot in (standoff(0.230, 0.0, base=TRAY_TOP_W), standoff(0.300, 0.075, base=TRAY_TOP_W),
                     standoff(0.370, -0.080, base=TRAY_TOP_W), lying(0.300, -0.075), lying(0.230, 0.080, 30.0),
                     lying(L.TRAY_OUTLINE[0] + 0.002, 0.0)):
            self.assertFalse(under_board(spot))
            self.assertTrue(over_tray(spot))
            st = dict(FINAL, standoff_5=spot)
            self.assertEqual(scored(st), 0.0)
            self.assertEqual(details(st)["strays"], [S5])
            self.assertEqual(details(st)["zero_reason"], "stray_standoff")

    def test_stray_needs_the_case(self):
        """Off the tray (the dish, the mat, the table) a standoff is neither credited nor a stray, upright or
        lying."""
        for spot in (in_dish(1), lying(L.DISH_X, L.DISH_Y + 0.01),
                     standoff(L.MAT_X + 0.03, L.MAT_Y, base=L.TOP + L.MAT_TOP), standoff(0.45, 0.0, base=L.TOP)):
            p = standoff_pose(spot)
            self.assertFalse(is_stray(p, spot))
            self.assertEqual(scored(dict(FINAL, standoff_5=spot)), 1.0)

    def test_any_stray_zeroes_any_progress(self):
        for spot in ("A1", "A2"):
            st = dict(FINAL, standoff_5=on(spot))  # A3's standoff moved under the board
            self.assertEqual(scored(st), 0.0, spot)
        self.assertEqual(scored(scene(standoff_3=on("H3"))), 0.0)  # A2's still there

    def test_lying_or_stacked_or_on_top(self):
        for spot in (lying(0.300, 0.0), lying(0.31, 0.01, axis_yaw=45.0),
                     standoff(*L.HOLES["H1"], base=L.TOP + L.STANDOFF_TOP),  # stacked on H1's
                     standoff(0.300, 0.010, base=L.TOP + L.MOUNT_Z + L.BOARD_T)):  # left standing on the board
            st = dict(FINAL, standoff_5=spot)
            self.assertEqual(scored(st), 0.0)
            self.assertEqual(details(st)["strays"], [S5])

    def test_a_stray_while_moving_is_still_a_stray(self):
        tr, _ = settled(START)
        x, y = L.HOLES["A2"]
        for i in range(5):  # being slid across the tray under the board
            res = tr.update(dict(READY, standoff_5=standoff(x + 0.003 * i, y, base=TRAY_TOP_W)))
        self.assertEqual(res.score, 0.0)


# -- zero conditions ------------------------------------------------------------------------------------------------


class ZeroConditions(unittest.TestCase):
    def test_anything_off_the_table(self):
        for n in L.STANDOFFS:
            for spot in (standoff(0.30, -0.70, base=L.TOP), standoff(0.30, 0.66, base=L.TOP - 0.3)):
                st = dict(FINAL, **{n: spot})
                self.assertEqual(scored(st), 0.0, n)
                self.assertEqual(details(st)["zero_reason"], "off_the_table")
        st = dict(READY, board=board(x=0.30, y=0.70, z=L.TOP - 0.40))
        self.assertEqual(scored(st), 0.0)
        self.assertEqual(details(st)["off_table"], [BOARD])

    def test_off_table_edges(self):
        self.assertFalse(off_table(standoff(0.30, L.TABLE_Y1 - 0.001, base=L.TOP)))
        self.assertTrue(off_table(standoff(0.30, L.TABLE_Y1 + 0.001, base=L.TOP)))
        self.assertFalse(off_table(standoff(0.30, 0.50, base=L.TOP - L.OFF_TABLE_DROP + 0.001)))
        self.assertTrue(off_table(standoff(0.30, 0.50, base=L.TOP - L.OFF_TABLE_DROP - 0.001)))
        self.assertTrue(off_table(standoff(L.TABLE_X1 + 0.01, 0.0, base=L.TOP)))

    def test_strays_thrown_off_the_table(self):
        st = scene(standoff_3=standoff(0.30, 0.80, base=L.TOP - 0.7), standoff_4=standoff(0.3, -0.8, base=0.0))
        self.assertEqual(scored(st), 0.0)


# -- stillness -----------------------------------------------------------------------------------------------------


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        for _ in range(WINDOW):
            res = tr.update(FINAL)
            self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.update(FINAL).score, 1.0)

    def test_moving_standoff_is_not_credited(self):
        tr, _ = settled(READY)
        x, y = L.HOLES["H3"]
        for i in range(WINDOW + 1):  # shaken 4 mm to and fro, never more than 2 mm from the hole
            res = tr.update(dict(READY, standoff_3=standoff(x + 0.0004 * (i % 2) * 10 - 0.002, y)))
        self.assertAlmostEqual(res.score, 0.6)
        self.assertEqual(res.details["strays"], [])  # on its hole while moving: not a stray

    def test_moving_board_is_not_mounted(self):
        tr, _ = settled(FINAL)
        for i in range(WINDOW + 1):
            res = tr.update(dict(FINAL, board=board(x=L.MOUNT_X + 0.003 * (i % 2))))
        self.assertAlmostEqual(res.score, 0.8)

    def test_jitter_within_2_mm_is_rest(self):
        tr = Tracker(HZ)
        for i in range(3 * WINDOW):
            d = 0.0009 * (1 if i % 2 else -1)
            res = tr.update(dict(FINAL, board=board(x=L.MOUNT_X + d), standoff_4=standoff(L.HOLES["H4"][0] + d,
                                                                                          L.HOLES["H4"][1])))
        self.assertEqual(res.score, 1.0)

    def test_a_turn_of_the_board_counts_as_moving(self):
        tr, _ = settled(FINAL)
        for i in range(WINDOW + 1):
            res = tr.update(dict(FINAL, board=board(yaw=4.0 * (i % 2))))  # holes swing about 3.5 mm
        self.assertAlmostEqual(res.score, 0.8)

    def test_a_swing_about_one_hole_counts_as_moving(self):
        """Turned 3 degrees to and fro about its rear-left hole (which stays put), the board's other holes move up
        to 4 mm: not at rest."""
        r = rotation(about("z", 3.0))
        h1 = np.array([*L.ITX_HOLES["H1"], 0.0])
        o = np.array([L.MOUNT_X, L.MOUNT_Y, 0.0])
        o2 = h1 + r @ (o - h1)
        tr, _ = settled(FINAL)
        for i in range(WINDOW + 1):
            b = board(x=o2[0], y=o2[1], yaw=3.0) if i % 2 else board()
            res = tr.update(dict(FINAL, board=b))
            np.testing.assert_allclose(board_holes(b)["H1"][:2], L.ITX_HOLES["H1"], atol=1e-9)
        self.assertGreater(res.details["moved_mm"][BOARD], 2.0)
        self.assertAlmostEqual(res.score, 0.8)


# -- exploits --------------------------------------------------------------------------------------------------------


class Exploits(unittest.TestCase):
    def test_board_held_still_just_above_the_standoffs(self):
        self.assertEqual(scored(dict(READY, board=board(z=L.TOP + L.MOUNT_Z + 0.0015))), 1.0)  # documented
        self.assertAlmostEqual(scored(dict(READY, board=board(z=L.TOP + L.MOUNT_Z + 0.010))), 0.8)

    def test_board_laid_over_the_strays(self):
        st = scene(standoff_5=on("H3"), board=board())  # H3 filled from A3, A1 and A2 left under the board
        self.assertEqual(scored(st), 0.0)

    def test_board_on_a_copy_of_the_pattern_elsewhere(self):
        """Four standoffs stood in the board's pattern off the tray, the board laid on them: no hole has a standoff,
        so nothing scores."""
        dx, dy = -0.10, 0.22
        st = {n: standoff(x + dx, y + dy, base=L.TOP) for n, (x, y) in zip(L.STANDOFFS, L.ITX_HOLES.values())}
        st[S5] = on("A3")
        st[BOARD] = board(x=L.MOUNT_X + dx, y=L.MOUNT_Y + dy, z=L.TOP + L.STANDOFF_H)
        self.assertEqual(scored(st), 0.0)

    def test_all_standoffs_removed_and_board_on_the_tray(self):
        st = {n: in_dish(i % 3) if i < 3 else standoff(0.20, 0.30 - 0.03 * i, base=L.TOP)
              for i, n in enumerate(L.STANDOFFS)}
        st[BOARD] = board(z=TRAY_TOP_W)
        self.assertEqual(scored(st), 0.0)

    def test_standoffs_pushed_off_their_holes_by_the_board(self):
        x, y = L.HOLES["H4"]
        st = dict(FINAL, standoff_4=standoff(x - 0.008, y))  # shoved 8 mm: under the board, not on its hole
        self.assertEqual(scored(st), 0.0)

    def test_spare_knocked_aside_instead_of_removed(self):
        """A3's standoff moved onto H4, A1's onto H3, A2's knocked over or slid off its hole onto bare tray clear of
        the board's outline (not taken out), the board mounted: the loose standoff is a stray, 0."""
        x, y = L.HOLES["A2"]
        for spot in (lying(x, y - 0.030), lying(x - 0.025, y, axis_yaw=90.0), standoff(x - 0.025, y, base=TRAY_TOP_W),
                     standoff(x, y - 0.025, base=TRAY_TOP_W)):
            self.assertFalse(under_board(spot))
            st = scene(standoff_5=on("H4"), standoff_3=on("H3"), standoff_4=spot, board=board())
            self.assertEqual(scored(st), 0.0)
            self.assertEqual(details(st)["strays"], [S4])
        st = scene(standoff_5=on("H4"), standoff_3=on("H3"), standoff_4=in_dish(0), board=board())
        self.assertEqual(scored(st), 1.0)  # the same, with A2's taken to the dish

    def test_standoff_lying_across_its_hole(self):
        st = dict(READY, standoff_3=lying(*L.HOLES["H3"]))
        self.assertEqual(scored(st), 0.0)

    def test_two_on_one_hole_cannot_both_count(self):
        x, y = L.HOLES["H3"]
        st = dict(READY, standoff_5=standoff(x + 0.0045, y), standoff_3=standoff(x - 0.0045, y))  # overlapping
        res = settled(st)[1]
        self.assertEqual(sum(v is not None for v in res.details["on_hole"].values()), 4)
        self.assertAlmostEqual(res.score, 0.8)


# -- robustness ------------------------------------------------------------------------------------------------------


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(FINAL)
        before = (tr.steps, {n: [p.copy() for p in tr.history[n][-1]] for n in NAMES})
        bads = []
        for n in (S3, BOARD):
            b = FINAL[n]
            bads.append({n: SimpleNamespace(**dict(vars(b), pos=np.array([np.nan, 0.0, 0.0])))})
            bads.append({n: SimpleNamespace(**dict(vars(b), quat_xyzw=np.zeros(4)))})
            bads.append({n: SimpleNamespace(**dict(vars(b), low=np.array([0.0, np.inf, 0.0])))})
            bads.append({n: None})
        bads.append({BOARD: SimpleNamespace(**{k: v for k, v in vars(FINAL[BOARD]).items() if k != "origin"})})
        bads.append({BOARD: SimpleNamespace(**dict(vars(FINAL[BOARD]), origin=np.array([0.3, 0.0])))})
        for bad in bads:
            st = dict(FINAL, **bad)
            res = tr.update(st)
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
            self.assertEqual(res.details["zero_reason"], "invalid_state")
            self.assertEqual(tr.result(st).score, 0.0)
        self.assertEqual(tr.steps, before[0] + len(bads))
        for n in NAMES:  # the record of the trial is as it was
            for a, b in zip(tr.history[n][-1], before[1][n]):
                np.testing.assert_allclose(a, b)
        self.assertEqual(tr.update(FINAL).score, 1.0)  # and the trial goes on

    def test_missing_object(self):
        st = {k: v for k, v in FINAL.items() if k != S2}
        self.assertEqual(scored(st), 0.0)

    def test_valid(self):
        self.assertTrue(valid(S1, FINAL[S1]))
        self.assertTrue(valid(BOARD, FINAL[BOARD]))
        self.assertFalse(valid(BOARD, FINAL[S1]))  # a plain cylinder has no origin
        self.assertFalse(valid(S1, object()))
        self.assertFalse(valid(S1, SimpleNamespace(pos="abc", quat_xyzw=[0, 0, 0, 1], low=[0, 0, 0])))

    def test_unnormalised_quaternions(self):
        st = dict(FINAL, board=SimpleNamespace(**dict(vars(FINAL[BOARD]), quat_xyzw=FINAL[BOARD].quat_xyzw * 3.0)),
                  standoff_3=SimpleNamespace(**dict(vars(FINAL[S3]), quat_xyzw=FINAL[S3].quat_xyzw * 0.7)))
        self.assertEqual(scored(st), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rnd = random.Random(7)
        tr = Tracker(HZ)
        for _ in range(300):
            st = {n: standoff(rnd.uniform(-0.2, 0.9), rnd.uniform(-0.7, 0.7), base=L.TOP + rnd.uniform(-0.2, 0.2),
                              yaw=rnd.uniform(0, 360), tilt=rnd.uniform(0, 180), tilt_axis=(rnd.random(), rnd.random(),
                                                                                             rnd.random() + 0.01))
                  for n in L.STANDOFFS}
            st[BOARD] = board(rnd.uniform(-0.2, 0.9), rnd.uniform(-0.7, 0.7), L.TOP + rnd.uniform(-0.2, 0.2),
                              yaw=rnd.uniform(0, 360), tilt=rnd.uniform(0, 180))
            res = tr.update(st)
            self.assertIn(res.score, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
            json.dumps(res.details)


# -- resets -----------------------------------------------------------------------------------------------------------


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(FINAL)
        res = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(res.score, 0.0)
        res = tr.update(FINAL)  # 1 step of history in the new trial: nothing at rest yet
        self.assertEqual(res.score, 0.0)

    def test_reset_then_a_new_trial(self):
        tr, _ = settled(FINAL)
        tr.update(START)
        _, res = settled(FINAL, tracker=tr)
        self.assertEqual(res.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(FINAL)
        jit = scene(**{n: standoff(L.HOLES[L.STANDOFF_START[n]][0] + 0.02, L.HOLES[L.STANDOFF_START[n]][1] - 0.02)
                       for n in L.STANDOFFS}, board=board(L.BOARD_START[0] - 0.02, L.BOARD_START[1] + 0.025,
                                                          z=L.TOP + L.MAT_TOP))
        self.assertTrue(Tracker.at_start(jit))
        tr.update(jit)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(READY)
        n = tr.steps
        tr.update(START)  # standoffs back by 35 mm in one step, the board not moved: no jump over 8 cm
        self.assertEqual(tr.steps, n + 1)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(FINAL)
        n = tr.steps
        tr.update(dict(FINAL, board=board(x=0.20, y=0.25, z=L.TOP)))
        self.assertEqual(tr.steps, n + 1)

    def test_start_needs_every_part_on_its_mark(self):
        self.assertFalse(Tracker.at_start(dict(START, standoff_3=in_dish(0))))
        self.assertFalse(Tracker.at_start(dict(START, board=board())))
        self.assertFalse(Tracker.at_start(dict(START, standoff_1=on("H1", tilt=5.0))))

    def test_explicit_reset(self):
        tr, _ = settled(FINAL)
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
        """A synthetic scene with the board turned into the AssemblyState mechkit would report."""
        out = dict(st)
        s = st[BOARD]
        out[BOARD] = self.mk.state_from_poses(self.task.BOARD_PART, (*s.origin, *s.quat_xyzw), source="sim")
        return out

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = {o.name: o.mass for o in t.OBJECTS if o.mass is not None}
        self.assertEqual(set(loose), {*L.STANDOFFS, BOARD})
        self.assertTrue(all(0 < m <= 0.30 for m in loose.values()))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 3.0)
        self.assertEqual(len(t.HOLE_DISCS), 7)
        for o in t.HOLE_DISCS:
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP + L.TRAY_TOP)
            self.assertAlmostEqual(o.radius, L.DISC_R)
        for o in t.STANDOFF_PARTS:
            self.assertAlmostEqual(o.radius, L.STANDOFF_R)
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.TOP + L.DISC_TOP)
        self.assertAlmostEqual(t.CASE_TRAY.pos[2] - t.CASE_TRAY.size[2] / 2, L.TOP)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.20 m)
        for o in t.OBJECTS:
            lo_y = abs(o.pos[1]) + (o.size[1] / 2 if hasattr(o, "size") else o.radius)
            self.assertLess(lo_y, 0.40, o.name)

    def test_scorer_geometry_matches_mechkit(self):
        spec = self.task.BOARD_PART
        lo, hi = spec.local_bounds()
        np.testing.assert_allclose(lo, [-L.BOARD_HALF, -L.BOARD_HALF, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.BOARD_HALF, L.BOARD_HALF, L.BOARD_T + L.COOLER_SIZE[2]], atol=1e-12)
        pads = [s for s in spec.shapes if s.kind == "cylinder"]
        self.assertEqual(sorted((s.center[0], s.center[1]) for s in pads), sorted(L.BOARD_HOLES.values()))
        rest = scene(board=spec.rest_state())
        self.assertTrue(Tracker.at_start(rest))
        st = self.sim(FINAL)
        np.testing.assert_allclose(st[BOARD].low[2], L.TOP + L.MOUNT_Z, atol=1e-12)
        np.testing.assert_allclose(st[BOARD].low, FINAL[BOARD].low, atol=1e-12)
        np.testing.assert_allclose(st[BOARD].high, FINAL[BOARD].high, atol=1e-12)
        self.assertEqual(scored(st), 1.0)

    def test_the_board_is_balanced_on_its_cooler_and_standoffs(self):
        com = self.task.BOARD_PART.center_of_mass()
        self.assertLess(abs(com[0]), 0.006)  # the I/O block pulls it 4 mm toward the rear: 0.006 N m on the grip
        self.assertAlmostEqual(com[1], 0.0)
        hx = [h[0] for h in L.BOARD_HOLES.values()]
        self.assertTrue(min(hx) + 0.02 < com[0] < max(hx) - 0.02)  # well inside the four standoffs

    def test_offline_state_path(self):
        spec = self.task.BOARD_PART
        c, _, yaw = spec.offline_box()
        box = SimpleNamespace(pos=np.array(c), quat_xyzw=about("z", math.degrees(yaw)), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, (*L.BOARD_START, L.TOP + L.MAT_TOP), atol=1e-12)

    def test_hooks_and_mechkit_reset(self):
        t = self.task
        tr = t._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        final = self.sim(FINAL)
        for _ in range(WINDOW):
            t.success(final)
        self.assertTrue(t.success(final))
        self.assertEqual(t.score(final), 1.0)
        self.assertEqual(t.score(final), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(t.success(final))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                for _ in range(WINDOW + 1):
                    self.task.success(self.sim(FINAL))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "tier", "zero_reason", "strays", "on_hole",
                          "holes_filled", "mounted", "board_over_standoffs", "board_tilt_deg", "board_yaw_deg",
                          "board_x_mm", "board_y_mm", "board_z_mm", "board_hole_offset_mm", "board_hole_gap_mm",
                          "standoff_hole", "standoff_offset_mm", "standoff_tilt_deg", "standoff_floor_mm",
                          "standoff_seated", "standoff_over_tray", "standoff_under_board", "moved_mm", "off_table",
                          "source", "board_origin", "board_quat",
                          "board_holes", "standoff_bottom"):
                    self.assertIn(k, row)
                self.assertEqual(row["score"], 1.0)
                self.assertEqual(row["on_hole"], {"H1": S1, "H2": S2, "H3": S3, "H4": S4})
                self.assertEqual(row["standoff_hole"][S5], "A3")
                self.assertEqual(row["standoff_seated"], {S1: "H1", S2: "H2", S3: "H3", S4: "H4", S5: "A3"})
                self.assertTrue(all(row["standoff_over_tray"].values()))
                self.assertEqual(row["source"], "sim")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertTrue(self.task.success(self.sim(FINAL)))  # no crash
                bad = dict(self.sim(FINAL), board=None)
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
        ends = {t.SO_PICK_Z, t.SO_PLACE_Z, t.BOARD_PICK[2], t.BOARD_PLACE[2]}
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                closing = w.finger > prev.finger  # less negative: narrower
                holding = prev.finger > t.SO_OPEN + 1e-9
                if prev.finger != w.finger and (closing or holding):  # every grip and release: 1.5 s, in place
                    self.assertGreaterEqual(w.duration, 1.5)
                    self.assertEqual(prev.pos, w.pos)
                if prev.pos[2] > w.pos[2] and any(abs(w.pos[2] - z) < 1e-9 for z in ends):
                    self.assertGreaterEqual(w.duration, 1.5)  # final descents are slow
                    self.assertLessEqual((prev.pos[2] - w.pos[2]) / w.duration, 0.0167 + 1e-9)
                self.assertEqual(w.yaw, 0.0)
        left, right = plan["left"], plan["right"]

        def at(wps, label):
            return [w for w in wps if w.label == label][0]
        # standoffs let go 1.5 mm above their discs, over the hole centres
        for wps, dst in ((left, "H3"), (right, "H4")):
            w = at(wps, f"let go on {dst}")
            self.assertLess(math.dist(w.pos[:2], L.HOLES[dst]), 1e-9)
            self.assertAlmostEqual(w.pos[2] - t.SO_GRIP_UP - (L.TOP + L.DISC_TOP), t.SO_DROP)
            self.assertLessEqual(t.SO_DROP, 0.002)  # let go just above the disc, well under the 3 mm floor tolerance
            self.assertGreater(w.pos[2] - t.PAD_TIP - (L.TOP + L.TRAY_TOP), 0.005)  # pad tips clear of the tray
        self.assertEqual(at(left, "grip the standoff on A1").pos[:2], L.HOLES["A1"])
        self.assertEqual(at(right, "grip the standoff on A2").pos[:2], L.HOLES["A2"])
        # the board: let go with its underside 1 mm over the standoffs, within 2 mm of the mount centre
        w = at(right, "let go of the board")
        self.assertLess(math.dist(w.pos[:2], (L.MOUNT_X, L.MOUNT_Y)), 0.002)
        self.assertAlmostEqual(w.pos[2] - t.BOARD_GRIP_UP - (L.TOP + L.MOUNT_Z), t.BOARD_DROP)
        self.assertLess(t.BOARD_DROP, L.BOARD_SEAT_TOL)
        carry = at(right, "carry it over the tray, holes over the standoffs")
        self.assertGreater(carry.pos[2] - t.BOARD_GRIP_UP - (L.TOP + L.STANDOFF_TOP), 0.04)

        # the arms take turns over the tray
        def times(wps):
            out, acc = {}, 0.0
            for w in wps:
                acc += w.duration
                out.setdefault(w.label, acc)
            return out
        tl, trt = times(left), times(right)
        self.assertGreaterEqual(tl["over the standoff on A1"] - 1.5, trt["straight up from H4"])  # leaves after
        self.assertGreaterEqual(tl["over the standoff on A1"], trt["over the board's cooler"])  # right is at the mat
        self.assertGreaterEqual(trt["carry it over the tray, holes over the standoffs"] - 2.5,
                                tl["straight up from H3"])
        self.assertGreaterEqual(trt["carry it over the tray, holes over the standoffs"], tl["aside again"])
        self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ, 1800)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_grips(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.STANDOFF_GRIP, 2 * L.STANDOFF_R - 0.008)  # 8 mm narrower than the part
        self.assertAlmostEqual(-2 * t.COOLER_GRIP, L.COOLER_SIZE[1] - 0.008)
        self.assertGreater(-2 * t.SO_OPEN, 2 * L.STANDOFF_R + 0.025)  # 16 mm clearance each side
        # the open standoff fingers (6 mm thick) stay 20 mm or more from every other standoff's edge in y
        gap = -t.SO_OPEN + 0.006
        for a, b in (("A1", "H1"), ("A2", "H3"), ("H4", "A1"), ("H3", "H4")):
            (ax, ay), (bx, by) = L.HOLES[a], L.HOLES[b]
            if abs(ax - bx) < 0.0135 + L.STANDOFF_R:  # within the pads' width along x
                self.assertGreater(abs(ay - by) - gap - L.STANDOFF_R, 0.02)
        # pad tips 6 mm above a standoff's bottom, 8 mm above the board's top at the cooler
        self.assertAlmostEqual(t.SO_GRIP_UP - t.PAD_TIP, 0.006)
        self.assertAlmostEqual(t.BOARD_GRIP_UP - t.PAD_TIP - L.BOARD_T, 0.008)
        # the fingers close along y on the cooler: the I/O block (along the +x edge) is beside them, not between
        self.assertGreater(L.IO_CENTER[0] - L.IO_SIZE[0] / 2, 0.0135 + 0.01)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("mini-ITX", "a standoff under each of the board's four holes", "none anywhere else under the board",
                  "none loose on the tray", "a hole clear of the board", "parts dish", "each hole over its standoff",
                  "I/O block at the rear"):
            self.assertIn(s, ins)
        data = json.loads((HERE / "task.json").read_text())
        self.assertEqual(data["instruction"], ins)

    def test_task_json_numbers_match_the_layout(self):
        data = json.loads((HERE / "task.json").read_text())
        setup, scoring = data["setup"], data["scoring"]
        for s in ("180 mm along x, 200 mm along y and 3 mm thick", "28 mm across", "1 mm proud",
                  "H1 at x 33.5 cm, y +3.5 cm", "H2 at x 33.5, y -3.5", "H3 at x 26.5, y +3.5", "H4 at x 29.5, y -3.5",
                  "A1 at x 30.0, y +3.5", "A2 at x 26.5, y -3.5", "A3 at x 33.5, y -7.5", "18 mm across and 20 mm tall",
                  "20 g each", "150 g", "88 x 88 mm and 3 mm thick", "40 x 40 mm and 30 mm tall",
                  "12 mm deep, 48 mm long and 15 mm tall", "14 mm across", "rear left 35 mm toward the rear and 35 mm",
                  "front right 5 mm toward the front", "30 mm nearer the rear than the front-left one", "110 x 110 x 5 mm",
                  "x 18.0 cm, y -20.0 cm", "x 25.6 to 34.4 cm, y -4.4 to +4.4 cm", "80 x 80 x 3 mm",
                  "x 19.0 cm, y +21.0 cm", "x 30.0 cm, y 0"):
            self.assertIn(s, setup)
        for s in ("10 degrees", "3 mm", "5 mm", "28 mm disc", "2 mm", "0.5 s", "88 x 88 mm", "9 mm", "3 degrees",
                  "7 mm", "10 mm", "0.2", "0, 0.2, 0.4, 0.6, 0.8, 1.0", "3 s", "over the tray",
                  "does not stand on H1, H2, H3, H4 or A3", "loose on the tray or on the board, upright or lying"):
            self.assertIn(s, scoring)
        self.assertEqual(data["run_limit"], "1,800 steps (90 s at 20 Hz)")
        reset = data["reset_notes"]
        for k in ("H1", "H2", "A1", "A2", "A3"):
            x, y = L.HOLES[k]
            self.assertIn(f"{k} (x {x * 100:.1f} cm, y {y * 100:+.1f} cm)", reset)


if __name__ == "__main__":
    unittest.main()
