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
import rigstate  # noqa: E402
import trace_trim  # noqa: E402
from scoring import (  # noqa: E402
    OBJECTS,
    Tracker,
    held,
    inside_selection,
    lowest,
    over_mat,
    pitch_at,
    seated_geometry,
    tilt_deg,
    valid,
    yaw_off_deg,
)

HZ = 20.0
DT = 1.0 / HZ
SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
POSSIBLE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# -- poses ----------------------------------------------------------------------------------------------------------
def _qmul(a, b):
    (x1, y1, z1, w1), (x2, y2, z2, w2) = a, b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def _qaxis(axis, deg):
    h = math.radians(deg) / 2
    a = np.asarray(axis, dtype=float)
    return np.array([*(a * math.sin(h)), math.cos(h)])


def quat(yaw=0.0, tilt_x=0.0, tilt_y=0.0):
    """Tilted about the body's x axis, then its y axis, then turned yaw about the vertical (x, y, z, w)."""
    q = _qmul(_qaxis((0, 0, 1), yaw), _qaxis((0, 1, 0), tilt_y))
    return _qmul(q, _qaxis((1, 0, 0), tilt_x))


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def obj(size, x, y, low, yaw=0.0, tilt_x=0.0, tilt_y=0.0, vel=(0, 0, 0)):
    """An object state with its centre over (x, y) and its lowest point at height ``low``, as the simulator reports it
    (pos the centre, low/high the world-aligned bounding box of the turned box)."""
    q = quat(yaw, tilt_x, tilt_y)
    corners = (SIGNS * (np.asarray(size) / 2)) @ _rot(q).T
    c = np.array([x, y, low - corners[:, 2].min()])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c + corners.min(axis=0), high=c + corners.max(axis=0))


def box(x, y, low, **kw):
    return obj(L.BOX_SIZE, x, y, low, **kw)


def div(x, y, low, **kw):
    return obj(L.DIV_SIZE, x, y, low, **kw)


DIV_START = div(L.DIV_X, L.DIV_Y, L.SHELF)
DIV_ON_MAT = div(L.MAT_X, L.MAT_Y, L.MAT_TOP)
DIV_ON_TABLE = div(0.285, -0.120, L.TOP)  # dropped on the table off the mat, out of the selection
BOX_START = [box(x, L.BOX_Y, L.TOP) for x in L.BOX_XS]
SEATED = [box(x, L.LANE_Y, L.SHELF) for x in L.PITCH_X]
AWAY = [box(0.30 + 0.05 * k, -0.30, L.TOP) for k in range(4)]  # set down on the table to the right, out of the way


def scene(divider=DIV_START, boxes=None, **over):
    """The five scored objects; ``boxes`` a list of four states (default: on their supply marks)."""
    out = {"divider": divider}
    for n, s in zip(L.BOX_NAMES, boxes if boxes is not None else BOX_START):
        out[n] = s
    out.update(over)
    return out


def arm(tips, opening=1.0, tcp=None):
    tips = np.asarray(tips, dtype=float)
    return SimpleNamespace(tcp=tips + np.array([0.0, 0.0, 0.013]) if tcp is None else np.asarray(tcp, float),
                           tips=tips, opening=float(opening))


READY = {"left": arm((0.32, 0.305, L.TOP + 0.25)), "right": arm((0.32, -0.305, L.TOP + 0.25))}


def arms(left=None, right=None):
    return {"left": left or READY["left"], "right": right or READY["right"]}


def run(frames, t=None, arms_=READY, n=None):
    """Feed one scene (repeated n times) or a list of scenes; return the last result and the tracker."""
    t = Tracker(HZ) if t is None else t
    seq = frames if isinstance(frames, list) else [frames] * (n if n is not None else 12)
    r = None
    for f in seq:
        r = t.update(f, arms_)
    return r, t


def settled(objs, arms_=READY):
    r, _ = run(objs, arms_=arms_)
    return r


def final(**over):
    return scene(DIV_ON_MAT, SEATED, **over)


def with_box(k, state, base=None):
    boxes = list(base if base is not None else SEATED)
    boxes[k] = state
    return boxes


# -- tests ----------------------------------------------------------------------------------------------------------
class Layout(unittest.TestCase):
    def test_pitches_ribs_and_slot(self):
        for a, b in L.PITCHES:
            self.assertAlmostEqual(b - a, 0.054, places=9)
        self.assertEqual([round(x, 4) for x in L.PITCH_X], [0.195, 0.257, 0.319, 0.381])
        self.assertAlmostEqual(L.RIB_XS[0] - L.RIB_T / 2, L.TRAY_X0, places=9)  # first rib flush with the front
        self.assertAlmostEqual(L.TRAY_X1 - (L.RIB_XS[-1] + L.RIB_T / 2), 0.004, places=9)  # 4 mm behind the last
        self.assertAlmostEqual(L.SLOT_Y1 - L.SLOT_Y0, 0.024, places=9)
        self.assertAlmostEqual(L.DIV_Y - L.DIV_SIZE[1] / 2 - L.SLOT_Y0, 0.002, places=9)  # 2 mm to each rib
        self.assertAlmostEqual(L.LANE_Y, 0.140, places=9)
        self.assertAlmostEqual(2 * L.BOX_SIZE[0], 0.060, places=9)  # two boxes need more than one pitch's 54 mm
        self.assertGreater(2 * L.BOX_SIZE[0], L.PITCHES[0][1] - L.PITCHES[0][0])
        self.assertAlmostEqual(L.BOX_XS[1] - L.BOX_XS[0] - L.BOX_SIZE[0], 0.050, places=9)  # 50 mm supply gaps

    def test_pitch_at(self):
        self.assertEqual(pitch_at(0.195), 0)
        self.assertEqual(pitch_at(0.168), 0)
        self.assertEqual(pitch_at(0.222), 0)
        self.assertIsNone(pitch_at(0.226))  # over rib 2
        self.assertEqual(pitch_at(0.381), 3)
        self.assertIsNone(pitch_at(0.418))  # behind the last rib
        self.assertIsNone(pitch_at(0.160))


class Geometry(unittest.TestCase):
    def test_inside_selection(self):
        self.assertTrue(inside_selection(DIV_START, L.DIV_SIZE))
        self.assertFalse(inside_selection(DIV_ON_MAT, L.DIV_SIZE))
        for s in BOX_START:
            self.assertFalse(inside_selection(s, L.BOX_SIZE))
        for s in SEATED:
            self.assertTrue(inside_selection(s, L.BOX_SIZE))
        # a box upright on the table against the tray's back face (touching it from outside) is not inside
        self.assertFalse(inside_selection(box(L.TRAY_X1 + 0.015, 0.14, L.TOP), L.BOX_SIZE))
        self.assertFalse(inside_selection(box(L.TRAY_X1 + 0.0145, 0.14, L.TOP), L.BOX_SIZE))  # 0.5 mm into it
        self.assertTrue(inside_selection(box(L.TRAY_X1 + 0.013, 0.14, L.TOP), L.BOX_SIZE))  # 2 mm over the edge
        self.assertFalse(inside_selection(box(L.TRAY_X0 - 0.015, 0.14, L.TOP), L.BOX_SIZE))  # against the front face
        self.assertTrue(inside_selection(box(L.TRAY_X0 - 0.013, 0.14, L.TOP), L.BOX_SIZE))
        # leaning back over the tray from the table: its top is over the shelf
        self.assertTrue(inside_selection(box(L.TRAY_X1 + 0.022, 0.14, L.TOP, tilt_y=-15.0), L.BOX_SIZE))
        # at any height: high above the shelf still counts
        self.assertTrue(inside_selection(box(0.30, 0.14, L.TOP + 0.30), L.BOX_SIZE))
        # beside the tray on the right, outside the right wall: not inside
        self.assertFalse(inside_selection(box(0.30, 0.040 - 0.071, L.TOP), L.BOX_SIZE))

    def test_divider_pulled_partway_out_is_inside(self):
        part_out = div(0.150, L.DIV_Y, L.TOP)  # centre in front of the delivery edge, outline over pitches 1-2
        self.assertLess(part_out.pos[0], L.TRAY_X0)
        self.assertTrue(inside_selection(part_out, L.DIV_SIZE))
        clear = div(L.TRAY_X0 - L.DIV_SIZE[0] / 2 - 0.002, L.DIV_Y, L.TOP)
        self.assertFalse(inside_selection(clear, L.DIV_SIZE))

    def test_angles(self):
        self.assertAlmostEqual(tilt_deg(quat(30.0, 6.0)), 6.0, places=4)
        self.assertAlmostEqual(tilt_deg(quat(0.0, 180.0)), 0.0, places=4)  # upside down: the same box
        self.assertAlmostEqual(tilt_deg(quat(0.0, 0.0, 90.0)), 90.0, places=4)
        self.assertAlmostEqual(yaw_off_deg(quat(7.0)), 7.0, places=4)
        self.assertAlmostEqual(yaw_off_deg(quat(-7.0)), 7.0, places=4)
        self.assertAlmostEqual(yaw_off_deg(quat(180.0)), 0.0, places=4)  # turned end for end: the same
        self.assertAlmostEqual(yaw_off_deg(quat(90.0)), 90.0, places=4)
        self.assertAlmostEqual(lowest(SEATED[0], L.BOX_SIZE), L.SHELF, places=9)

    def test_held(self):
        b = SEATED[0]
        centre = np.asarray(b.pos)
        self.assertTrue(held(b, L.BOX_SIZE, arms(arm(centre, 0.2))))
        self.assertTrue(held(b, L.BOX_SIZE, arms(right=arm(centre, 0.2))))  # either gripper
        self.assertFalse(held(b, L.BOX_SIZE, arms(arm(centre, 0.6))))  # more than half open
        self.assertTrue(held(b, L.BOX_SIZE, arms(arm(centre, 0.046 / 0.0939))))  # a 46 mm release gap: held
        self.assertFalse(held(b, L.BOX_SIZE, arms(arm(centre, 0.048 / 0.0939))))  # 48 mm: not
        top = L.SHELF + L.BOX_SIZE[2]
        self.assertTrue(held(b, L.BOX_SIZE, arms(arm((b.pos[0], b.pos[1], top + 0.014), 0.2))))
        self.assertFalse(held(b, L.BOX_SIZE, arms(arm((b.pos[0], b.pos[1], top + 0.016), 0.2))))
        self.assertFalse(held(b, L.BOX_SIZE, None))  # unreadable arms hold nothing
        tilted = box(0.30, 0.14, L.TOP + 0.10, tilt_x=40.0)  # the grown outline turns with the object
        self.assertTrue(held(tilted, L.BOX_SIZE, arms(arm(tilted.pos + _rot(tilted.quat_xyzw)[:, 1] * 0.08, 0.2))))

    def test_seated_geometry(self):
        for k, s in enumerate(SEATED):
            self.assertEqual(seated_geometry(s), k)
        self.assertEqual(seated_geometry(box(0.183, 0.14, L.SHELF)), 0)  # 12 mm forward: still in pitch 1
        self.assertEqual(seated_geometry(box(0.195, 0.159, L.SHELF)), 0)  # 19 mm off the centreline
        self.assertIsNone(seated_geometry(box(0.195, 0.161, L.SHELF)))  # 21 mm off
        self.assertEqual(seated_geometry(box(0.195, 0.14, L.SHELF, yaw=9.0)), 0)
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF, yaw=11.0)))
        self.assertEqual(seated_geometry(box(0.195, 0.14, L.SHELF, tilt_x=7.5)), 0)
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF, tilt_x=8.5)))
        self.assertEqual(seated_geometry(box(0.195, 0.14, L.SHELF + 0.0029)), 0)
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF + 0.0035)))
        self.assertEqual(seated_geometry(box(0.195, 0.14, L.SHELF - 0.0009)), 0)  # a contact reading 0.9 mm in
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF - 0.0015)))
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.RIB_TOP)))  # riding a rib
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF, tilt_y=90.0)))  # lying on a side
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF, tilt_x=90.0)))  # standing on end
        self.assertIsNone(seated_geometry(box(0.195, 0.14, L.SHELF, yaw=90.0)))  # not turned across the lane

    def test_over_mat(self):
        self.assertTrue(over_mat(DIV_ON_MAT))
        self.assertTrue(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP, tilt_x=90.0)))  # lying on its 230 x 40 side
        self.assertTrue(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP, tilt_y=90.0)))  # standing on its end
        self.assertTrue(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP, yaw=20.0)))
        self.assertFalse(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP + 0.006)))  # on something 6 mm up
        self.assertTrue(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP + 0.0049)))
        self.assertTrue(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP - 0.0009)))  # a contact reading 0.9 mm in
        self.assertFalse(over_mat(div(L.MAT_X, L.MAT_Y, L.MAT_TOP - 0.0015)))
        self.assertFalse(over_mat(div(L.MAT_X, L.MAT_Y - 0.031, L.MAT_TOP)))  # centre 1 mm off the mat
        self.assertTrue(over_mat(div(L.MAT_X, L.MAT_Y - 0.029, L.MAT_TOP)))
        self.assertFalse(over_mat(div(L.MAT_X + 0.131, L.MAT_Y, L.MAT_TOP)))
        # tipped off the mat's edge with one edge on the bare table, its centre still over the mat: not on it
        self.assertFalse(over_mat(div(L.MAT_X, L.MAT_Y + 0.025, L.TOP, tilt_x=-14.0)))
        self.assertFalse(over_mat(DIV_ON_TABLE))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = settled(scene())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertIn("Z2 divider inside the selection", r.details["zeros"])

    def test_every_tier(self):
        self.assertEqual(settled(scene(DIV_ON_MAT)).score, 0.2)
        for n in range(1, 5):
            boxes = SEATED[:n] + BOX_START[n:]
            r = settled(scene(DIV_ON_MAT, boxes))
            self.assertEqual(r.score, [0.4, 0.6, 0.8, 1.0][n - 1])
            self.assertEqual(r.details["n"], n)
            self.assertEqual(r.success, n == 4)

    def test_divider_off_the_mat_loses_its_0_2(self):
        for n in range(0, 5):
            boxes = SEATED[:n] + BOX_START[n:]
            self.assertEqual(settled(scene(DIV_ON_TABLE, boxes)).score, [0.0, 0.2, 0.4, 0.6, 0.8][n])

    def test_the_run_must_start_at_the_front(self):
        back_three = BOX_START[:1] + SEATED[1:]  # pitches 2-4 seated, pitch 1 empty: the first vend drops nothing
        self.assertEqual(settled(scene(DIV_ON_MAT, back_three)).score, 0.2)
        self.assertEqual(settled(scene(DIV_ON_TABLE, back_three)).score, 0.0)
        gap_at_3 = SEATED[:2] + [BOX_START[2], SEATED[3]]  # pitches 1, 2 and 4
        r = settled(scene(DIV_ON_MAT, gap_at_3))
        self.assertEqual(r.score, 0.6)
        self.assertEqual(r.details["seated_pitches"], [1, 2, 4])
        boxes_elsewhere = SEATED[:1] + AWAY[1:]  # the others set down on the table to the right
        self.assertEqual(settled(scene(DIV_ON_MAT, boxes_elsewhere)).score, 0.4)

    def test_an_empty_pitch_in_front_of_a_filled_one(self):
        # The instruction: the front pitch must hold a box and no pitch may be left empty in front of a filled one.
        # A box behind an empty pitch neither counts nor zeroes; only the run from the front pitch is scored.
        for filled, want in (((1, 3), 0.4), ((1, 2, 4), 0.6), ((1, 3, 4), 0.4), ((2, 3, 4), 0.2), ((4,), 0.2),
                             ((3,), 0.2), ((1, 2, 3, 4), 1.0)):
            boxes = [SEATED[k] if k + 1 in filled else AWAY[k] for k in range(4)]
            r = settled(scene(DIV_ON_MAT, boxes))
            self.assertEqual(r.score, want, filled)
            self.assertEqual(r.details["seated_pitches"], sorted(filled))
            self.assertFalse(r.details["zeros"])

    def test_loose_seating_still_counts(self):
        loose = [box(0.183, 0.155, L.SHELF + 0.002, yaw=8.0), box(0.268, 0.125, L.SHELF, tilt_x=6.0),
                 box(0.319, 0.14, L.SHELF, yaw=-9.0, tilt_y=5.0), box(0.392, 0.14, L.SHELF + 0.001)]
        self.assertEqual(settled(scene(DIV_ON_MAT, loose)).score, 1.0)

    def test_either_way_up_or_end_for_end(self):
        flipped = [box(L.PITCH_X[0], L.LANE_Y, L.SHELF, tilt_x=180.0), box(L.PITCH_X[1], L.LANE_Y, L.SHELF, yaw=180.0),
                   SEATED[2], box(L.PITCH_X[3], L.LANE_Y, L.SHELF, yaw=179.0, tilt_y=180.0)]
        self.assertEqual(settled(scene(DIV_ON_MAT, flipped)).score, 1.0)

    def test_success_needs_the_full_score_at_rest(self):
        t = Tracker(HZ)
        r, _ = run(final(), t, n=10)
        self.assertFalse(r.success)  # not yet 0.5 s of stillness
        r, _ = run(final(), t, n=1)
        self.assertTrue(r.success)
        self.assertEqual(t.first_full_step, 11)
        r, _ = run(scene(DIV_ON_MAT, SEATED[:3] + BOX_START[3:]), Tracker(HZ))
        self.assertFalse(r.success)

    def test_scores_are_exact(self):
        for n in range(5):
            s = settled(scene(DIV_ON_MAT, SEATED[:n] + BOX_START[n:])).score
            self.assertIn(s, POSSIBLE)


class ZeroRules(unittest.TestCase):
    def check_zero(self, objs, rule="Z1"):
        r = settled(objs)
        self.assertEqual(r.score, 0.0, r.details)
        self.assertTrue(any(z.startswith(rule) for z in r.details["zeros"]), r.details["zeros"])
        return r

    def test_box_riding_a_rib(self):
        self.check_zero(final(box_4=box(0.350, 0.14, L.RIB_TOP)))  # on rib 4, between pitches 3 and 4
        self.check_zero(final(box_4=box(0.381, 0.14, L.RIB_TOP)))  # bridging the ribs either side of pitch 4

    def test_box_tilted_or_leaning(self):
        self.check_zero(final(box_2=box(0.257, 0.14, L.SHELF, tilt_y=12.0)))
        self.check_zero(final(box_2=box(0.257, 0.14, L.SHELF, tilt_x=10.0)))

    def test_box_lying_or_on_end(self):
        self.check_zero(final(box_1=box(0.195, 0.14, L.RIB_TOP, tilt_y=90.0)))  # on its 140 x 70 face: 70 along x
        self.check_zero(final(box_1=box(0.195, 0.14, L.SHELF, tilt_x=90.0)))  # standing on end, 140 tall

    def test_box_not_turned_across_the_lane(self):
        self.check_zero(final(box_3=box(0.30, 0.14, L.RIB_TOP, yaw=90.0)))  # 140 mm along x over three ribs

    def test_box_on_another_box(self):
        on_top = box(0.195, 0.14, L.SHELF + L.BOX_SIZE[2])
        self.check_zero(scene(DIV_ON_MAT, SEATED[:3] + [on_top]))

    def test_box_across_a_wall_top(self):
        on_wall = box(0.30, L.WALL_INNER_L + L.WALL_W / 2, L.WALL_TOP, yaw=90.0)  # lying along the left wall's top
        self.check_zero(final(box_4=on_wall))
        across = box(0.381, L.WALL_INNER_L + L.WALL_W / 2, L.WALL_TOP)  # bridging the wall top, half over the shelf
        self.check_zero(final(box_4=across))

    def test_box_off_the_lane(self):
        # Upright on the floor a box cannot be more than 20 mm off the centreline (the walls are 20 mm from its ends);
        # turned or tilted it can, and then it is not seated.
        self.check_zero(final(box_2=box(0.257, 0.140 + 0.0205, L.SHELF)))

    def test_box_on_the_back_floor_or_ahead_of_the_first_rib(self):
        self.check_zero(final(box_4=box(0.418, 0.14, L.SHELF, tilt_y=40.0)))  # tipped over the back edge
        self.check_zero(final(box_1=box(0.150, 0.14, L.TOP, tilt_y=-30.0)))  # leaning on the delivery edge

    def test_divider_left_in(self):
        self.check_zero(scene(DIV_START, SEATED), "Z2")

    def test_divider_held_over_the_tray_scores_zero(self):
        up = div(0.285, 0.14, L.TOP + 0.08)
        grip = arms(arm(np.asarray(up.pos) + [0, 0, 0.005], 0.2))
        r = settled(scene(up, SEATED), grip)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["divider"]["held"])

    def test_divider_across_the_walls_or_turned_on_the_mat(self):
        across = div(0.30, 0.14, L.WALL_TOP, yaw=90.0)  # lying across both wall tops
        self.check_zero(scene(across, SEATED), "Z2")
        turned = div(L.MAT_X, L.MAT_Y, L.MAT_TOP, yaw=90.0)  # on the mat but along y: reaches into the selection
        self.assertTrue(over_mat(turned))
        self.check_zero(scene(turned, SEATED), "Z2")

    def test_a_jam_zeroes_whatever_else_holds(self):
        r = self.check_zero(final(box_2=box(0.226, 0.14, L.RIB_TOP)))
        self.assertTrue(r.details["divider_on_mat"])


class HeldAndMoving(unittest.TestCase):
    def test_box_still_held_neither_counts_nor_zeroes(self):
        b4 = SEATED[3]
        grip = arms(arm(b4.pos, 0.3))
        self.assertEqual(settled(final(), grip).score, 0.8)
        riding = box(0.381, 0.14, L.RIB_TOP)  # held while riding the ribs: not a jam, not counted
        self.assertEqual(settled(final(box_4=riding), arms(arm(riding.pos, 0.3))).score, 0.8)
        self.assertEqual(settled(final(), arms(arm(b4.pos, 0.9))).score, 1.0)  # an open gripper holds nothing

    def test_parked_closed_gripper_gains_nothing(self):
        # Parking a closed gripper on a box riding a rib saves the score from Z1, but only to what leaving that box
        # off the tray gives.
        riding = box(0.257, 0.14, L.RIB_TOP)
        parked = settled(scene(DIV_ON_MAT, [SEATED[0], riding, SEATED[2], SEATED[3]]), arms(arm(riding.pos, 0.0)))
        off_tray = settled(scene(DIV_ON_MAT, [SEATED[0], AWAY[1], SEATED[2], SEATED[3]]))
        self.assertEqual(parked.score, off_tray.score)
        self.assertEqual(parked.score, 0.4)

    def test_divider_held_over_the_mat_is_not_on_it(self):
        g = arms(arm(np.asarray(DIV_ON_MAT.pos) + [0, 0, 0.01], 0.2))
        r = settled(final(), g)
        self.assertFalse(r.details["divider_on_mat"])
        self.assertEqual(r.score, 0.8)

    def test_unreadable_arms_hold_nothing(self):
        r, t = run(final(), arms_=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)
        self.assertFalse(r.details["arms_available"])

    def test_moving_box_neither_counts_nor_zeroes(self):
        t = Tracker(HZ)
        run(final(), t)
        slide = [final(box_4=box(0.381 + 0.001 * k, 0.14, L.RIB_TOP)) for k in range(1, 6)]  # 1 mm a step
        r, _ = run(slide, t)
        self.assertFalse(r.details["boxes"]["box_4"]["at_rest"])
        self.assertEqual(r.score, 0.8)
        r, _ = run(slide[-1:] * 12, t)  # and once it stops there, it is a jam
        self.assertEqual(r.score, 0.0)

    def test_at_rest_is_a_path_over_half_a_second(self):
        t = Tracker(HZ)
        r, _ = run(final(), t, n=10)  # 10 positions: only 9 steps of path, not yet 0.5 s
        self.assertFalse(r.details["boxes"]["box_1"]["at_rest"])
        self.assertEqual(r.score, 0.0)
        r, _ = run(final(), t, n=1)
        self.assertEqual(r.score, 1.0)
        # a box that came down 3 mm in the last 0.5 s is not at rest yet
        t = Tracker(HZ)
        run(final(box_1=box(0.195, 0.14, L.SHELF + 0.003, vel=(0, 0, -0.1))), t, n=12)
        r, _ = run(final(), t, n=10)
        self.assertFalse(r.details["boxes"]["box_1"]["at_rest"])
        r, _ = run(final(), t, n=1)
        self.assertTrue(r.details["boxes"]["box_1"]["at_rest"])

    def test_velocity_chatter_does_not_matter(self):
        # Reported velocities of 45 mm/s on still objects (contact chatter) change nothing: stillness is a path.
        noisy = {n: SimpleNamespace(**{**vars(s), "lin_vel": np.array([0.045, -0.03, 0.02])})
                 for n, s in final().items()}
        self.assertEqual(settled(noisy).score, 1.0)

    def test_position_jitter_under_the_path_limit_is_rest(self):
        rng = np.random.default_rng(1)
        seq = []
        for _ in range(40):
            boxes = [box(x + rng.uniform(-5e-5, 5e-5), L.LANE_Y + rng.uniform(-5e-5, 5e-5), L.SHELF)
                     for x in L.PITCH_X]
            seq.append(scene(DIV_ON_MAT, boxes))
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)


class Exploits(unittest.TestCase):
    def test_fifth_layer_on_top_of_a_seated_box(self):
        # Loading pitch 4 by laying box 4 flat across the tops of boxes 2 and 3 (70 mm above the floor): a jam.
        on_tops = box(0.288, 0.14, L.SHELF + L.BOX_SIZE[2], tilt_y=90.0)
        self.assertEqual(settled(scene(DIV_ON_MAT, SEATED[:3] + [on_tops])).score, 0.0)

    def test_divider_pushed_out_of_the_front_of_the_slot(self):
        part_out = div(0.150, L.DIV_Y, L.TOP + 0.012, tilt_y=-8.0)  # front end on the table, back end in the slot
        self.assertEqual(settled(scene(part_out, SEATED)).score, 0.0)
        out = div(0.285, -0.120, L.TOP)  # pulled all the way out and dropped on the table: boxes count, 1.0 is gone
        self.assertEqual(settled(scene(out, SEATED)).score, 0.8)

    def test_divider_parked_in_the_air(self):
        # Holding the divider up out of the slot while loading boxes (to dodge Z2 at a 'height'): Z2 at any height.
        up = div(0.285, 0.14, L.TOP + 0.20)
        self.assertEqual(settled(scene(up, SEATED), arms(arm(up.pos, 0.2))).score, 0.0)

    def test_two_boxes_in_one_pitch(self):
        squeezed = [SEATED[0], box(0.247, 0.14, L.SHELF), box(0.272, 0.14, L.RIB_TOP - 0.004, tilt_y=12.0),
                    SEATED[3]]
        self.assertEqual(settled(scene(DIV_ON_MAT, squeezed)).score, 0.0)

    def test_boxes_not_turned(self):
        flat = [box(x, 0.14, L.RIB_TOP, yaw=90.0) for x in (0.20, 0.36)]
        self.assertEqual(settled(scene(DIV_ON_MAT, flat + BOX_START[2:])).score, 0.0)

    def test_box_tipped_in_from_the_front_lies_down(self):
        tipped = box(0.195, 0.14, L.RIB_TOP, tilt_y=90.0)
        self.assertEqual(settled(scene(DIV_ON_MAT, [tipped] + BOX_START[1:])).score, 0.0)

    def test_hovering_a_box_earns_nothing(self):
        hover = box(0.195, 0.14, L.SHELF + 0.002)
        r = settled(scene(DIV_ON_MAT, [hover] + BOX_START[1:]), arms(arm(hover.pos, 0.3)))
        self.assertEqual(r.score, 0.2)

    def test_dropped_box_that_lands_seated_counts(self):
        fall = [scene(DIV_ON_MAT, [box(0.195, 0.14, L.SHELF + 0.05 - 0.5 * 9.81 * (k * DT) ** 2,
                                       vel=(0, 0, -9.81 * k * DT))] + BOX_START[1:]) for k in range(1, 3)]
        r, _ = run(fall + [scene(DIV_ON_MAT, SEATED[:1] + BOX_START[1:])] * 12)
        self.assertEqual(r.score, 0.4)

    def test_order_does_not_matter_for_the_end_state(self):
        t = Tracker(HZ)
        boxes = list(BOX_START)
        for k in (3, 2, 1, 0):  # loaded from the back, ending full
            boxes[k] = SEATED[k]
            run(scene(DIV_ON_MAT, boxes), t)
        self.assertEqual(t.last.score, 1.0)

    def test_divider_on_end_on_the_mat_counts(self):
        on_end = div(L.MAT_X, L.MAT_Y, L.MAT_TOP, tilt_y=90.0)
        self.assertEqual(settled(scene(on_end, SEATED)).score, 1.0)

    def test_box_left_leaning_on_the_tray_from_outside(self):
        lean = box(L.TRAY_X1 + 0.022, 0.14, L.TOP, tilt_y=-15.0)
        self.assertEqual(settled(scene(DIV_ON_MAT, SEATED[:3] + [lean])).score, 0.0)
        upright = box(L.TRAY_X1 + 0.0152, 0.14, L.TOP)  # upright on the table against the back face: not inside
        self.assertEqual(settled(scene(DIV_ON_MAT, SEATED[:3] + [upright])).score, 0.8)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        s0 = SEATED[0]
        inf = SimpleNamespace(pos=np.array([0.195, 0.14, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]),
                              high=np.zeros(3))
        zero_q = SimpleNamespace(pos=s0.pos, quat_xyzw=np.zeros(4), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                 low=s0.low, high=s0.high)
        missing = SimpleNamespace(pos=s0.pos, quat_xyzw=s0.quat_xyzw)
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            t = Tracker(HZ)
            run(final(), t)
            before = ({n: list(map(tuple, t.centres[n])) for n in OBJECTS}, t.steps)
            r = t.update(final(box_1=bad), READY)
            self.assertEqual((r.score, r.success), (0.0, False))
            self.assertEqual(t.result(final(box_1=bad), READY).score, 0.0)
            self.assertEqual({n: list(map(tuple, t.centres[n])) for n in OBJECTS}, before[0])
            self.assertEqual(t.bad_steps, 1)
            r = t.update(final(), READY)  # the next good state is scored as before
            self.assertEqual(r.score, 1.0)
        t = Tracker(HZ)
        r = t.update({"divider": DIV_ON_MAT}, READY)  # a missing object
        self.assertEqual(r.score, 0.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(0)
        t = Tracker(HZ)
        for _ in range(300):
            objs = {}
            for n in OBJECTS:
                size = L.DIV_SIZE if n == "divider" else L.BOX_SIZE
                objs[n] = obj(size, rng.uniform(-0.2, 0.7), rng.uniform(-0.6, 0.6), rng.uniform(-0.2, 1.2),
                              yaw=rng.uniform(-180, 180), tilt_x=rng.uniform(-180, 180), tilt_y=rng.uniform(-90, 90))
            g = arms(arm(rng.uniform(-0.2, 1.0, 3), rng.uniform(0, 1)), arm(rng.uniform(-0.2, 1.0, 3), 1.0))
            r = t.update(objs, g)
            self.assertIn(r.score, POSSIBLE)

    def test_score_is_idempotent(self):
        t = Tracker(HZ)
        run(final(), t)
        a, b = t.result(final(), READY), t.result(final(), READY)
        self.assertEqual((a.score, a.details), (b.score, b.details))
        self.assertEqual(t.steps, 12)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker(HZ)
        run(final(), t)
        self.assertEqual(t.last.score, 1.0)
        r = t.update(scene(), READY)  # everything jumps back onto its marks in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(t.centres["box_1"]), 1)
        self.assertIsNone(t.first_full_step)

    def test_jittered_start_resets_too(self):
        t = Tracker(HZ)
        run(final(), t)
        jit = scene(div(L.DIV_X + 0.02, L.DIV_Y, L.SHELF), [box(x - 0.02, L.BOX_Y + 0.02, L.TOP) for x in L.BOX_XS])
        t.update(jit, READY)
        self.assertEqual(t.steps, 1)

    def test_carrying_back_gradually_does_not_reset(self):
        t = Tracker(HZ)
        run(final(), t)
        steps = t.steps
        b = SEATED[0]
        path = [final(box_1=box(b.pos[0] + (L.BOX_XS[0] - b.pos[0]) * f, b.pos[1] + (L.BOX_Y - b.pos[1]) * f,
                                L.SHELF + 0.03))
                for f in np.linspace(0.05, 1.0, 30)]
        run(path, t)
        self.assertEqual(t.steps, steps + 30)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = Tracker(HZ)
        run(final(), t)
        t.update(final(box_1=box(0.0, -0.4, L.TOP)), READY)  # knocked far in one step, not onto its mark
        self.assertEqual(t.steps, 13)


# -- trace_trim.py: the GPU tuning aid -----------------------------------------------------------------------------------
def _trace(errors_mm, grip_mm=4.0, episode=1, tilt_closing=0.0):
    """Trace rows as task.py writes them, for one demo: each box carried with its centre grip_mm ahead of the tool
    point, then set down errors_mm[k] off its pitch's centre, the pads opening, and the box left there."""
    rows = []
    state = {n: {"held": False, "pos": [x, L.BOX_Y, L.TOP + 0.035], "low_mm": 0.0, "tilt_deg": 0.0, "pitch": None,
                 "jam": False} for n, x in zip(L.BOX_NAMES, L.BOX_XS)}

    def add(opening, tcp_x, div_held=False):
        rows.append({"episode": episode, "step": len(rows) + 1, "valid_state": True, "score": 1.0, "n": 4,
                     "divider_on_mat": True,
                     "divider": {"held": div_held, "tilt_deg": 0.7 if div_held else 0.0, "low_mm": 5.0},
                     "boxes": {n: dict(v) for n, v in state.items()}, "opening": {"left": opening, "right": 1.0},
                     "left_tcp": [tcp_x, 0.14, L.TOP + 0.1]})

    for _ in range(5):
        add(1.0, 0.2, div_held=True)
    for k, name in enumerate(L.BOX_NAMES):
        sx, px = L.BOX_XS[k], L.PITCH_X[k] + errors_mm[k] / 1000
        g = grip_mm / 1000
        for _ in range(6):  # the pads closing on the box at its supply mark
            state[name] = {**state[name], "held": True, "tilt_deg": tilt_closing}
            add(0.32, sx - g)
        for _ in range(10):  # carried
            state[name] = {**state[name], "pos": [px, L.LANE_Y, L.TOP + 0.125], "low_mm": 90.0, "tilt_deg": 0.0}
            add(0.32, px - g)
        for _ in range(12):  # settled at the place, still gripped
            state[name] = {**state[name], "pos": [px, L.LANE_Y, L.SHELF + 0.038], "low_mm": 23.0}
            add(0.32, px - g)
        for _ in range(5):  # the pads open past the box: let go
            state[name] = {**state[name], "pos": [px, L.LANE_Y, L.SHELF + 0.035], "low_mm": 20.0, "tilt_deg": 1.5}
            add(0.45, px - g)
        state[name] = {**state[name], "held": False, "pitch": k + 1}
        add(0.49, px - g)
    return rows


class TraceTrim(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)

    def write(self, rows_list):
        path = self.dir / f"trace_{len(list(self.dir.iterdir()))}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for rows in rows_list for r in rows) + "not json\n")
        return path

    def test_suggests_the_trim_that_cancels_the_median_place_error(self):
        rep = trace_trim.report(trace_trim.load(self.write([_trace((5.0, 6.0, 7.0, 6.0))])), trim=0.0)
        self.assertEqual(rep["suggested_trim_mm"], -6.0)
        self.assertFalse(rep["warnings"])
        for k, name in enumerate(L.BOX_NAMES):
            b = rep["boxes"][name]
            self.assertAlmostEqual(b["place_error_x_mm"], (5.0, 6.0, 7.0, 6.0)[k], places=6)
            self.assertAlmostEqual(b["grip_offset_x_mm"], 4.0, places=6)
            self.assertEqual(b["tilt_after_release_deg"], 1.5)
            self.assertEqual(b["end"]["pitch"], k + 1)
        self.assertEqual(rep["divider"], {"tilt_while_held_deg": 0.7, "end_low_mm": 5.0, "on_mat": True})
        # with a trim already set, the suggestion is the new total
        rep = trace_trim.report(trace_trim.load(self.write([_trace((1.0, 2.0, 1.0, 1.0))])), trim=-0.006)
        self.assertEqual(rep["suggested_trim_mm"], -7.0)

    def test_reads_the_last_episode_only(self):
        path = self.write([_trace((20.0, 20.0, 20.0, 20.0), episode=0), _trace((-3.0, -3.0, -3.0, -3.0), episode=1)])
        self.assertEqual(trace_trim.report(trace_trim.load(path))["suggested_trim_mm"], 3.0)

    def test_warnings(self):
        rep = trace_trim.report(trace_trim.load(self.write([_trace((0.0, 0.0, 13.0, 0.0), tilt_closing=9.0)])))
        self.assertEqual(rep["suggested_trim_mm"], 0.0)
        self.assertTrue(any("box_3: placed +13.0 mm" in w for w in rep["warnings"]), rep["warnings"])
        self.assertTrue(any("spread 13.0 mm" in w for w in rep["warnings"]), rep["warnings"])
        self.assertEqual(rep["boxes"]["box_1"]["tilt_before_lift_deg"], 9.0)  # tipped as the pads closed
        rows = trace_trim.load(self.write([_trace((0.0, 0.0, 0.0, 0.0))]))
        for r in rows:
            r["boxes"]["box_2"]["held"] = False
        self.assertIn("box_2 was never held", trace_trim.report(rows)["warnings"])
        self.assertEqual(trace_trim.load(self.write([[{"episode": 0, "valid_state": False}]])), [])


# -- task.py against the offline stand-in ------------------------------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def snap(left_tips=(0.32, 0.305, L.TOP + 0.25), opening=1.0):
    rigstate.offline_feed({"left": {"tcp": np.asarray(left_tips) + [0, 0, 0.013], "tips": left_tips,
                                    "opening": opening},
                           "right": {"tcp": (0.32, -0.305, L.TOP + 0.263), "tips": (0.32, -0.305, L.TOP + 0.25),
                                     "opening": 1.0}})


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hook, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, scored):
        out = {o.name: None for o in self.task.OBJECTS}
        out.update(scored)
        return out

    def test_objects_match_the_layout(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 19)  # tray, 2 walls, 10 ribs, divider, mat, 4 boxes
        loose = sorted(o.name for o in self.task.OBJECTS if o.mass is not None)
        self.assertEqual(loose, sorted(OBJECTS))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        d = next(o for o in self.task.OBJECTS if o.name == "divider")
        self.assertAlmostEqual(d.pos[2] - d.size[2] / 2, L.SHELF, places=9)

    def test_hooks_and_reset(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        for _ in range(12):
            snap()
            self.task.success(self.objects(final()))
        self.assertEqual(self.task.score(self.objects(final())), 1.0)
        self.assertTrue(self.task.success(self.objects(final())))
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects(final())))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects(scene())))
        self.assertEqual(self.task.score(self.objects(scene())), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        path = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(path)
            rigstate.offline_reset()
            snap()
            self.task.success(self.objects(final()))
            row = json.loads(path.read_text().splitlines()[-1])
            for key in ("score", "n", "divider_on_mat", "zeros", "seated_pitches", "boxes", "divider", "opening",
                        "arms_unavailable", "left_tips", "left_tcp"):
                self.assertIn(key, row)
            for key in ("pitch", "jam", "inside", "held", "at_rest", "path_mm", "low_mm", "tilt_deg", "yaw_off_deg",
                        "lane_off_mm", "pos"):
                self.assertIn(key, row["boxes"]["box_1"])
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(self.objects(final())))
        finally:
            path.unlink(missing_ok=True)
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", set()), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_numbers(self):
        t = self.task
        self.assertAlmostEqual(t.DIV_TCP - L.TOP, 0.058, places=9)
        self.assertAlmostEqual(t.MAT_TCP - L.TOP, 0.046, places=9)
        self.assertAlmostEqual(t.BOX_TCP - L.TOP, 0.038, places=9)
        self.assertAlmostEqual(t.BOX_HOVER - L.TOP, 0.128, places=9)
        self.assertAlmostEqual(t.PLACE_TCP - L.TOP, 0.061, places=9)
        self.assertAlmostEqual(-2 * t.BOX_RELEASE, 0.046, places=9)
        self.assertAlmostEqual(-2 * t.BOX_PRESET, 0.050, places=9)
        self.assertAlmostEqual(-2 * t.BOX_GRIP, 0.022, places=9)
        self.assertAlmostEqual(-2 * t.DIV_GRIP, 0.012, places=9)
        plan = t.demo()["left"]
        for a, b in zip(plan, plan[1:]):  # every close and open takes at least 1.5 s
            if abs(a.finger - b.finger) > 0.003:
                self.assertGreaterEqual(b.duration, 1.5 - 1e-9, b.label)
        # carried boxes clear the supply boxes, the walls and the placed boxes' tops
        self.assertGreater(t.BOX_HOVER - t.BOX_UP, L.TOP + L.BOX_SIZE[2])
        self.assertGreaterEqual(t.BOX_HOVER - t.BOX_UP + 1e-9, L.SHELF + L.BOX_SIZE[2])

    def test_divider_grip_has_no_rib_under_the_pads(self):
        # Gripped over the middle of pitch 2, not over rib 3: the pads' 20 mm along x is 17 mm from the rib faces.
        t = self.task
        self.assertAlmostEqual(t.DIV_GRIP_X, 0.257, places=9)
        self.assertAlmostEqual(t.DIV_GRIP_DX, -0.028, places=9)
        a, b = L.PITCHES[1]
        self.assertAlmostEqual(t.DIV_GRIP_X - t.PAD_HALF_WIDTH - a, 0.017, places=9)
        self.assertAlmostEqual(b - (t.DIV_GRIP_X + t.PAD_HALF_WIDTH), 0.017, places=9)
        for x in L.RIB_XS:
            self.assertGreaterEqual(abs(x - t.DIV_GRIP_X) - L.RIB_T / 2 - t.PAD_HALF_WIDTH, 0.017 - 1e-9)
        self.assertAlmostEqual(t.DIV_TCP - t.PAD_TIPS - L.SHELF, 0.025, places=9)  # pad tips 25 mm above the floor
        self.assertAlmostEqual(L.SHELF + L.DIV_SIZE[2] - (t.DIV_TCP - t.PAD_TIPS), 0.015, places=9)  # 15 mm down it
        self.assertLess(abs(t.DIV_GRIP_DX) + t.PAD_HALF_WIDTH, L.DIV_SIZE[0] / 2 - 0.05)  # well inside its length
        plan = t.demo()["left"]
        grip = next(w for w in plan if w.label == "grip the divider")
        self.assertAlmostEqual(grip.pos[0], t.DIV_GRIP_X, places=9)
        self.assertAlmostEqual(grip.pos[1], L.DIV_Y, places=9)
        drop = next(w for w in plan if w.label == "let go on the mat")
        self.assertAlmostEqual(drop.pos[0] - t.DIV_GRIP_DX, L.MAT_X, places=9)  # its centre over the mat's centre
        self.assertAlmostEqual(drop.pos[1], L.MAT_Y, places=9)

    def test_release_knobs(self):
        t = self.task
        self.assertAlmostEqual(t.PLACE_TCP - t.BOX_UP - L.SHELF, t.RELEASE_DROP, places=9)
        for drop in (t.RELEASE_DROP, 0.001, 0.002):  # the 3 mm used and the fallbacks the README names
            self.assertTrue(L.SEAT_LOW_MIN <= L.SHELF + drop <= L.SEAT_LOW_MAX, drop)
        self.assertGreaterEqual(t.RELEASE_S, 1.5)
        lets_go = [w for w in t.demo()["left"] if w.label.startswith("let go in pitch")]
        self.assertEqual(len(lets_go), 4)
        for w in lets_go:
            self.assertEqual(w.duration, t.RELEASE_S)
            self.assertAlmostEqual(w.pos[2], t.PLACE_TCP, places=9)
        self.assertEqual(trace_trim.current_trim(), t.PLACE_TRIM)  # the trace reader sees the trim task.py uses

    def test_texts_agree(self):
        meta = json.loads((HERE / "task.json").read_text())
        readme = (HERE / "README.md").read_text()
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        self.assertIn(self.task.INSTRUCTION, readme)
        for text in (json.dumps(meta), readme):
            self.assertNotIn(chr(0x2014), text)  # no em-dashes
        self.assertIn("Possible scores: 0, 0.2, 0.4, 0.6, 0.8, 1.0", meta["scoring"])
        # 'from the front' alone could mean the approach: the instruction states the front-pitch rule as an end state
        self.assertNotIn("from the front", self.task.INSTRUCTION)
        self.assertIn("The front pitch (nearest you) must hold a box, and no pitch may be left empty in front of a "
                      "filled one.", self.task.INSTRUCTION)
        for tk in ("TK-RMZJR4", "TK-BTYFXV", "TK-384FU2"):  # the nearest board tasks are named and told apart
            self.assertIn(tk, readme)


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo played through our offline checker's kinematics and carry/drop proxy."""

    def play(self, **env_extra):
        import subprocess
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **env_extra)
        env.pop("TASK_TRACE", None)
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--json", "-"],
                              capture_output=True, text=True, env=env, timeout=1200)
        return proc, json.loads(proc.stdout)

    def test_demo_predicts_one(self):
        proc, rep = self.play()
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertFalse(proxy["interference"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 2400)
        self.assertLess(rep["playback"]["arms"]["left"]["worst_ik_mm"], 0.5)
        self.assertGreaterEqual(min(rep["playback"]["arms"]["left"]["min_margin"]), 0.19)

    def test_left_arm_idle_scores_zero(self):
        _, rep = self.play(DEMO_IDLE_ARM="left")
        self.assertEqual(rep["playback"]["proxy"]["predicted_task_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
