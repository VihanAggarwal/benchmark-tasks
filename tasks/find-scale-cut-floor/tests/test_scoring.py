"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    SCORES, Tracker, axes_angle_deg, footprint, hands_clear, inside_cut, placement, polys_overlap, rests_on_find,
    rotation, scale_on_floor, trowel_stowed, turn_deg, valid,
)

HZ = 20.0
DT = 1.0 / HZ
D20 = math.degrees(L.FIND_YAW)
U = np.array([math.cos(L.FIND_YAW), math.sin(L.FIND_YAW)])  # the find's long axis
N = np.array([-U[1], U[0]])  # square to it, toward +y


def quat(yaw_deg=0.0, tilt_deg=0.0):
    """Turned yaw about the vertical after tilting about the body's x (long) axis (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qx = np.array([math.sin(b), 0.0, 0.0, math.cos(b)])
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qx
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def box(size, x, y, bottom, yaw_deg=0.0, tilt_deg=0.0, vel=(0, 0, 0)):
    """An ObjectState-like box: its centre at (x, y), lowest point ``bottom`` above the table top, turned and
    tilted, with its world bounding box."""
    q = quat(yaw_deg, tilt_deg)
    ext = np.abs(rotation(q)) @ (np.array(size, float) / 2)
    c = np.array([x, y, L.TOP + bottom + ext[2]])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c - ext, high=c + ext)


# Poses: (x, y, bottom above the table, yaw deg, tilt deg).
FIND0 = (L.FIND_X, L.FIND_Y, 0.0, D20, 0.0)
TROWEL0 = (L.TROWEL_X, L.TROWEL_Y, 0.0, D20, 0.0)
SCALE0 = (L.SCALE_X, L.SCALE_Y, L.KIT_FLOOR_H, 0.0, 0.0)
STOWED = (L.TRAY_X, L.TRAY_Y, L.TRAY_FLOOR_H, D20, 0.0)
ON_RIGHT_BANK = (0.26, -0.12, L.BANK_H, 0.0, 0.0)  # the trowel lifted onto the right bank (still in the shot)


def scale_pose(g=0.030, s=0.0, side=-1, a=0.0, bottom=0.0, tilt=0.0):
    """The scale's pose in the find frame: g (edge gap for a parallel scale), s along the find, on side -1 (-y, where
    the trowel starts) or +1, turned a degrees from parallel."""
    c = np.array([L.FIND_X, L.FIND_Y]) + side * (g + L.HALF_WIDTHS) * N + s * U
    return (float(c[0]), float(c[1]), bottom, D20 + a, tilt)


TIGHT = scale_pose()  # the demo's target: g 30 mm, s 0, parallel
OBJ = {"find": L.FIND_SIZE, "trowel": L.TROWEL_SIZE, "scale": L.SCALE_SIZE}

TIPS = 0.013
READY_L = (0.323, 0.305, L.TOP + 0.26)
READY_R = (0.323, -0.305, L.TOP + 0.26)
REST_L = (0.25, 0.25, L.TOP + 0.12)
REST_R = (0.20, -0.25, L.TOP + 0.12)
OPEN = 1.0


def snap(left_tcp=REST_L, left_open=OPEN, right_tcp=REST_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def obj(name, pose):
    return box(OBJ[name], *pose)


class Film:
    """A trial as frames: the three objects' poses and both arms (TCP, opening). ``go`` moves linearly to new values
    over n control steps."""

    def __init__(self, **start):
        self.cur = {"find": FIND0, "trowel": TROWEL0, "scale": SCALE0, "L": READY_L, "Lo": OPEN,
                    "R": READY_R, "Ro": OPEN}
        self.cur.update(start)
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a, b = dict(self.cur), {**self.cur, **to}
        for i in range(1, n + 1):
            f = i / n
            self.frames.append({k: (tuple((1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float))
                                    if isinstance(a[k], tuple) else (1 - f) * a[k] + f * b[k]) for k in a})
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)


def play(film, tracker=None, arms=True, episode=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None); ``episode`` is
    the rigstate episode number task.py passes (None: the scorer used on its own)."""
    t = tracker or Tracker(HZ)
    r = None
    for fr in film.frames:
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(obj("find", fr["find"]), obj("trowel", fr["trowel"]), obj("scale", fr["scale"]), a, episode)
    return r, t


def lifted(pose, h=0.10):
    return (pose[0], pose[1], h, pose[3], pose[4])


def procedure(trowel=STOWED, scale=TIGHT, arms_end=(REST_L, OPEN, REST_R, OPEN), settle=40, find_path=None):
    """The intended trial: the trowel lifted out of the cut and set in the tray (or at ``trowel``), the scale taken
    from the kit and laid at ``scale``, then the arms go to ``arms_end``; all still for ``settle`` steps."""
    f = Film().hold(10)
    f.go(10, trowel=lifted(TROWEL0)).go(30, trowel=lifted(trowel)).go(10, trowel=trowel)
    f.go(10, scale=lifted(SCALE0)).go(30, scale=lifted(scale)).go(15, scale=scale)
    if find_path:
        for pose, n in find_path:
            f.go(n, find=pose)
    lt, lo, rt, ro = arms_end
    f.go(20, L=lt, Lo=lo, R=rt, Ro=ro)
    return f.hold(settle)


def final(trowel=STOWED, scale=TIGHT, find=FIND0, arms=None):
    """The score of a trial that ends in this state (after a clean start)."""
    t = Tracker(HZ)
    play(Film().hold(5), t)
    a = arms if arms is not None else snap()
    for _ in range(15):
        r = t.update(obj("find", find), obj("trowel", trowel), obj("scale", scale), a)
    return r


class Geometry(unittest.TestCase):
    def test_layout_numbers_match_the_setup_text(self):
        find = obj("find", FIND0)
        corners = sorted(map(tuple, np.round(footprint(find, L.FIND_SIZE), 4)))
        self.assertEqual(corners, sorted([(0.3028, 0.0776), (0.3117, 0.0532), (0.2272, 0.0224), (0.2183, 0.0468)]))
        trowel = obj("trowel", TROWEL0)
        tc = sorted(map(tuple, footprint(trowel, L.TROWEL_SIZE)))
        want = sorted([(0.3132, 0.0345), (0.3269, -0.0031), (0.1954, -0.0509), (0.1817, -0.0134)])
        self.assertTrue(np.allclose(tc, want, atol=0.6e-4))
        d = np.array([L.TROWEL_X - L.FIND_X, L.TROWEL_Y - L.FIND_Y])
        self.assertAlmostEqual(-(d @ N) - 0.013 - 0.020, 0.018, delta=1e-4)  # 18 mm from the find's -y side
        self.assertAlmostEqual(L.TROWEL_YAW, L.FIND_YAW)  # parallel
        self.assertAlmostEqual(footprint(trowel, L.TROWEL_SIZE)[:, 0].min() - L.CUT[0], 0.012, delta=5e-4)
        self.assertGreaterEqual(footprint(trowel, L.TROWEL_SIZE)[:, 1].min() - L.CUT[2], 0.039)
        side = [np.array([L.FIND_X, L.FIND_Y]) + k * 0.045 * U + 0.013 * N for k in (-1, 1)]  # its +y long side
        gaps = np.array([L.CUT[3] - p[1] for p in side])
        self.assertAlmostEqual(gaps.min(), 0.0124, delta=2e-4)  # the find's +y side 12-43 mm from the left bank
        self.assertAlmostEqual(gaps.max(), 0.0432, delta=2e-4)
        self.assertTrue(np.allclose(L.TRAY_INNER, (0.16, 0.34, -0.425, -0.295)))
        self.assertTrue(np.allclose(L.CLEAR_BOX, (0.05, math.inf, -0.21, 0.21)))
        tight, rough, m = placement(obj("scale", TIGHT), find)
        self.assertTrue(tight and rough)
        self.assertAlmostEqual(m["g"], 0.030, places=6)
        self.assertAlmostEqual(m["s"], 0.0, places=6)
        self.assertTrue(np.allclose(TIGHT[:2], (0.2831, 0.0002), atol=1e-4))

    def test_angles(self):
        self.assertAlmostEqual(axes_angle_deg(math.radians(170), 0.0), 10.0, places=6)
        self.assertAlmostEqual(axes_angle_deg(math.radians(95), 0.0), 85.0, places=6)
        self.assertAlmostEqual(axes_angle_deg(math.radians(-8), math.radians(20)), 28.0, places=6)
        self.assertAlmostEqual(turn_deg(math.radians(183), math.radians(3)), 180.0, places=6)
        self.assertAlmostEqual(turn_deg(math.radians(-2), math.radians(2)), 4.0, places=6)
        # a scale turned end for end is still parallel
        _, rough, m = placement(obj("scale", scale_pose(a=180.0)), obj("find", FIND0))
        self.assertLess(m["a_deg"], 1e-6)
        self.assertTrue(rough)

    def test_on_the_floor(self):
        self.assertTrue(scale_on_floor(obj("scale", TIGHT)))
        self.assertTrue(scale_on_floor(obj("scale", scale_pose(bottom=0.0029))))
        self.assertFalse(scale_on_floor(obj("scale", scale_pose(bottom=0.0035))))
        self.assertTrue(scale_on_floor(obj("scale", scale_pose(tilt=4.5))))
        self.assertFalse(scale_on_floor(obj("scale", scale_pose(tilt=6.0))))
        self.assertTrue(scale_on_floor(obj("scale", scale_pose(tilt=180.0))))  # the other big face down
        self.assertFalse(scale_on_floor(obj("scale", scale_pose(tilt=90.0))))  # on edge
        self.assertFalse(scale_on_floor(obj("scale", SCALE0)))  # in the kit, 5 mm up
        self.assertFalse(scale_on_floor(obj("scale", (0.20, -0.20, 0.0, 0.0, 0.0))))  # table, outside the frame
        poly = footprint(obj("scale", (0.215, 0.0, 0.0, 0.0, 0.0)), L.SCALE_SIZE)  # its near end at x 0.165
        self.assertFalse(inside_cut(poly))
        poly = footprint(obj("scale", (0.2195, 0.0, 0.0, 0.0, 0.0)), L.SCALE_SIZE)  # 0.5 mm into the bank
        self.assertTrue(inside_cut(poly))

    def test_rests_on_the_find(self):
        find = obj("find", FIND0)
        on_top = obj("scale", (L.FIND_X, L.FIND_Y, L.FIND_SIZE[2], D20, 0.0))
        self.assertTrue(rests_on_find(on_top, L.SCALE_SIZE, find))
        leaning = obj("scale", (*(np.array([L.FIND_X, L.FIND_Y]) - 0.018 * N), 0.0, D20, 25.0))  # edge on it
        self.assertTrue(polys_overlap(footprint(leaning, L.SCALE_SIZE), footprint(find, L.FIND_SIZE)))
        self.assertTrue(rests_on_find(leaning, L.SCALE_SIZE, find))
        above = obj("scale", (L.FIND_X, L.FIND_Y, 0.030, D20, 0.0))  # held 14 mm over its top: not resting
        self.assertFalse(rests_on_find(above, L.SCALE_SIZE, find))
        self.assertFalse(rests_on_find(obj("scale", TIGHT), L.SCALE_SIZE, find))
        across = obj("trowel", (L.FIND_X, L.FIND_Y, L.FIND_SIZE[2], D20 + 90, 0.0))
        self.assertTrue(rests_on_find(across, L.TROWEL_SIZE, find))
        self.assertFalse(rests_on_find(obj("trowel", TROWEL0), L.TROWEL_SIZE, find))

    def test_stowed(self):
        self.assertTrue(trowel_stowed(obj("trowel", STOWED)))
        self.assertFalse(trowel_stowed(obj("trowel", TROWEL0)))
        on_rim = (L.TRAY_X, L.TRAY_Y, 0.030, 0.0, 0.0)  # across the walls' tops, 20 mm over the floor
        self.assertFalse(trowel_stowed(obj("trowel", on_rim)))
        self.assertFalse(trowel_stowed(obj("trowel", (L.TRAY_X, L.TRAY_Y, L.TRAY_FLOOR_H + 0.006, D20, 0.0))))
        leaning = (L.TRAY_X, L.TRAY_Y, L.TRAY_FLOOR_H, D20, 8.0)  # lowest point on the floor, centre inside
        self.assertTrue(trowel_stowed(obj("trowel", leaning)))
        self.assertFalse(trowel_stowed(obj("trowel", (0.35, L.TRAY_Y, L.TRAY_FLOOR_H, 90.0, 0.0))))  # centre out
        self.assertFalse(trowel_stowed(obj("trowel", (L.TRAY_X, -0.22, 0.0, 0.0, 0.0))))  # on the table beside it

    def test_hands_clear(self):
        self.assertTrue(hands_clear(snap()))
        self.assertTrue(hands_clear(snap(left_open=0.5)))  # pads 47 mm apart
        self.assertFalse(hands_clear(snap(left_open=0.45)))
        self.assertFalse(hands_clear(snap(right_open=0.2)))
        self.assertFalse(hands_clear(snap(left_tcp=(0.30, 0.17, L.TOP + 0.20))))  # high, but over the photo
        self.assertFalse(hands_clear(snap(left_tcp=(0.30, 0.205, L.TOP + 0.05))))  # pads could reach the bank
        self.assertTrue(hands_clear(snap(left_tcp=(0.30, 0.215, L.TOP + 0.05))))
        self.assertFalse(hands_clear(snap(right_tcp=(0.46, -0.10, L.TOP + 0.10))))
        self.assertFalse(hands_clear(snap(right_tcp=(0.50, -0.10, L.TOP + 0.10))))  # past the far side
        self.assertTrue(hands_clear(snap(right_tcp=(0.50, -0.215, L.TOP + 0.10))))
        self.assertFalse(hands_clear(snap(right_tcp=(0.06, -0.05, L.TOP + 0.10))))
        self.assertTrue(hands_clear(snap(right_tcp=(0.045, -0.05, L.TOP + 0.10))))
        self.assertTrue(hands_clear(snap(left_tcp=READY_L, right_tcp=READY_R)))  # the ready pose is clear
        self.assertFalse(hands_clear(None))
        self.assertFalse(hands_clear({"left": snap()["left"]}))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["tight"] and r.details["stowed"] and r.details["hands_clear"])
        self.assertFalse(r.details["disturbed"])

    def test_tight_band_edges_score_one(self):
        # (toward +u the far bank stops the scale at s about +17 mm on this side, so the far tolerances are -s)
        for pose in (scale_pose(g=0.0105), scale_pose(g=0.0395), scale_pose(s=0.015), scale_pose(s=-0.024),
                     scale_pose(a=4.5), scale_pose(a=-4.5), scale_pose(g=0.025, s=-0.02, a=4.0)):
            self.assertEqual(final(scale=pose).score, 1.0, pose)

    def test_rough_and_stowed_scores_075(self):
        for pose in (scale_pose(a=8.0), scale_pose(g=0.045), scale_pose(g=0.007), scale_pose(s=-0.035),
                     scale_pose(g=0.049, s=-0.044, a=11.0)):
            r = final(scale=pose)
            self.assertEqual(r.score, 0.75, pose)
            self.assertFalse(r.success)

    def test_g_45mm_scores_075_and_g_55mm_scores_quarter(self):
        self.assertEqual(final(scale=scale_pose(g=0.045)).score, 0.75)
        self.assertEqual(final(scale=scale_pose(g=0.055)).score, 0.25)

    def test_placed_but_trowel_not_stowed_scores_half(self):
        self.assertEqual(final(trowel=ON_RIGHT_BANK).score, 0.5)  # tight, trowel on a bank
        self.assertEqual(final(trowel=ON_RIGHT_BANK, scale=scale_pose(a=9.0)).score, 0.5)  # rough
        self.assertEqual(final(trowel=(0.20, -0.22, 0.0, 0.0, 0.0)).score, 0.5)  # on the table beside the tray
        self.assertEqual(final(trowel=(L.TRAY_X, L.TRAY_Y, 0.030, 0.0, 0.0)).score, 0.5)  # across the tray's rim

    def test_stowed_only_scores_quarter(self):
        self.assertEqual(final(scale=SCALE0).score, 0.25)  # the scale still in the kit
        self.assertEqual(final(scale=scale_pose(a=20.0)).score, 0.25)  # crooked
        self.assertEqual(final(scale=scale_pose(s=-0.050)).score, 0.25)  # off the find's middle
        self.assertEqual(final(scale=scale_pose(s=0.024)).score, 0.25)  # its far end up on the far bank
        self.assertEqual(final(scale=scale_pose(tilt=90.0)).score, 0.25)  # on edge
        self.assertEqual(final(scale=scale_pose(g=0.003)).score, 0.25)  # almost against it

    def test_nothing_placed_scores_zero(self):
        self.assertEqual(final(trowel=TROWEL0, scale=SCALE0).score, 0.0)
        self.assertEqual(final(trowel=ON_RIGHT_BANK, scale=scale_pose(g=0.060)).score, 0.0)

    def test_every_outcome_is_one_of_the_stated_scores(self):
        rng = np.random.default_rng(1)
        seen = set()
        for _ in range(400):
            sc = scale_pose(g=rng.uniform(-0.01, 0.07), s=rng.uniform(-0.06, 0.06), side=int(rng.choice((-1, 1))),
                            a=rng.uniform(-20, 20), bottom=float(rng.choice((0.0, 0.0, 0.018, 0.04))))
            tr = [STOWED, ON_RIGHT_BANK, TROWEL0][int(rng.integers(3))]
            r = final(trowel=tr, scale=sc)
            self.assertIn(r.score, SCORES)
            seen.add(r.score)
        self.assertEqual(seen, set(SCORES))

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        f.go(15, scale=scale_pose(s=0.015))  # still creeping along the find, 1 mm a step
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        self.assertFalse(r.details["still"])
        f.hold(6)
        r, _ = play(f)
        self.assertFalse(r.success)  # not yet 0.5 s within 3 mm (4 mm back in the window)
        f.hold(10)
        r, _ = play(f)
        self.assertTrue(r.success)

    def test_success_needs_hands_clear(self):
        r, _ = play(procedure(arms_end=((0.30, 0.10, L.TOP + 0.12), OPEN, REST_R, OPEN)))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_score_is_idempotent(self):
        f = procedure()
        r, t = play(f)
        fr = f.frames[-1]
        args = (obj("find", fr["find"]), obj("trowel", fr["trowel"]), obj("scale", fr["scale"]), snap())
        a, b = t.result(*args), t.result(*args)
        self.assertEqual((a.score, a.details), (b.score, b.details))
        self.assertEqual(t.steps, len(f.frames))


class FindDisturbed(unittest.TestCase):
    def test_moved_6mm_latches_zero_even_if_put_back(self):
        moved = (L.FIND_X + 0.006, L.FIND_Y, 0.0, D20, 0.0)
        r, t = play(procedure(find_path=[(moved, 10), (FIND0, 10)]))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["disturbed_why"], "moved")
        self.assertFalse(r.success)

    def test_nudged_4mm_is_tolerated(self):
        nudged = (L.FIND_X + 0.004 * U[0] - 0.0015 * N[0], L.FIND_Y + 0.004 * U[1] - 0.0015 * N[1], 0.0, D20, 0.0)
        r, _ = play(procedure(find_path=[(nudged, 10)]))
        self.assertEqual(r.score, 1.0)
        self.assertGreater(r.details["find_max_move_mm"], 4.0)

    def test_turned_4deg_latches_zero_2deg_does_not(self):
        r, _ = play(procedure(find_path=[((*FIND0[:3], D20 + 4.0, 0.0), 10)]))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["disturbed_why"], "turned")
        r, _ = play(procedure(find_path=[((*FIND0[:3], D20 - 2.0, 0.0), 10)]))
        self.assertEqual(r.score, 1.0)

    def test_tilted_4deg_latches_zero(self):
        r, _ = play(procedure(find_path=[((*FIND0[:4], 4.0), 5), (FIND0, 5)]))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["disturbed_why"], "tilted")

    def test_turned_over_latches_zero(self):
        r, _ = play(procedure(find_path=[((*FIND0[:4], 180.0), 1)]))
        self.assertEqual(r.score, 0.0)

    def test_lifted_4mm_and_put_back_latches_zero(self):
        up = (L.FIND_X, L.FIND_Y, 0.004, D20, 0.0)
        r, t = play(procedure(find_path=[(up, 10), (FIND0, 10)]))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["disturbed_why"], "lifted")
        self.assertAlmostEqual(r.details["find_max_lift_mm"], 4.0, places=3)
        up2 = (L.FIND_X, L.FIND_Y, 0.0025, D20, 0.0)  # 2.5 mm: a jostle, tolerated
        r, _ = play(procedure(find_path=[(up2, 10), (FIND0, 10)]))
        self.assertEqual(r.score, 1.0)

    def test_the_reference_is_the_first_control_step(self):
        settled = (L.FIND_X + 0.003, L.FIND_Y, 0.0, D20 + 1.0, 0.0)  # the find settles 3 mm off its mark at spawn
        f = procedure(find_path=None)
        for fr in f.frames:
            fr["find"] = settled
        f.cur["find"] = settled
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        f.go(10, find=(L.FIND_X + 0.003 + 0.0045, L.FIND_Y, 0.0, D20 + 1.0, 0.0)).hold(20)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)  # 4.5 mm from its reference, 7.5 mm from the mark
        f.go(10, find=(L.FIND_X + 0.003 + 0.0055, L.FIND_Y, 0.0, D20 + 1.0, 0.0)).hold(20)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_scale_on_the_find_scores_zero(self):
        on = (L.FIND_X, L.FIND_Y, L.FIND_SIZE[2], D20, 0.0)
        self.assertEqual(final(scale=on).score, 0.0)
        self.assertEqual(final(scale=on, trowel=ON_RIGHT_BANK).score, 0.0)

    def test_trowel_on_the_find_scores_zero(self):
        on = (L.FIND_X, L.FIND_Y, L.FIND_SIZE[2], D20 + 90.0, 0.0)
        self.assertEqual(final(trowel=on).score, 0.0)


class Exploits(unittest.TestCase):
    @staticmethod
    def search(side, trowel=True, slack=0.0):
        """Every scale pose on the cut floor on one side of the find (g 0-70 mm, s -60..+60 mm, a -12..+12 deg)
        inside the cut (``slack`` 0: what a real scale could take; L.CUT_SLACK: what the scorer allows) and
        overlapping neither the find nor (if present) the trowel. Returns the counts and the tight and rough poses."""
        find, tr = obj("find", FIND0), obj("trowel", TROWEL0)
        ff, tf = footprint(find, L.FIND_SIZE), footprint(tr, L.TROWEL_SIZE)
        feasible, tight, rough = 0, [], []
        for g in np.arange(0.0, 0.0701, 0.002):
            for s in np.arange(-0.060, 0.0601, 0.004):
                for a in range(-12, 13, 2):
                    pose = scale_pose(g=g, s=s, side=side, a=a)
                    sc = obj("scale", pose)
                    fp = footprint(sc, L.SCALE_SIZE)
                    if polys_overlap(fp, ff) or (trowel and polys_overlap(fp, tf)) or not inside_cut(fp, slack):
                        continue
                    feasible += 1
                    t, r, _ = placement(sc, find)
                    if t:
                        tight.append(pose)
                    if r:
                        rough.append((round(g * 1000), round(s * 1000), a, pose))
        return feasible, tight, rough

    def test_squeeze_beside_the_trowel_finds_nothing(self):
        # The trowel lies on the scale's strip, 18 mm from the find: the 20 mm scale cannot fit in, and beyond it
        # (g 58 mm or more) is past the 50 mm rough limit. No tight or rough pose exists on that side, with or
        # without the scorer's 2 mm cut slack.
        for slack in (0.0, L.CUT_SLACK):
            feasible, tight, rough = self.search(-1, trowel=True, slack=slack)
            self.assertGreater(feasible, 300)
            self.assertEqual((len(tight), len(rough)), (0, 0), slack)
        _, tight_free, _ = self.search(-1, trowel=False)  # with the trowel gone, plenty
        self.assertGreater(len(tight_free), 100)

    def test_cramped_far_side_placement_scores_half(self):
        # On the find's +y side the floor is 12-43 mm wide: cramped rough poses exist toward the near end, never a
        # tight one. A finer search (n 0.5 mm, s 1 mm, a 0.5 deg) finds, with the scorer's own 2 mm cut slack, about
        # 2,500 rough poses with g 5-16 mm, |s| 19-39 mm and a -12 to +2.5 deg; without the slack (what a real
        # scale could take) g 5-13.5 mm, |s| 24-37 mm, a -12 to -0.5 deg (turned toward the bank). This coarser grid
        # must stay inside those ranges. With the trowel left in the cut such a pose is 0.5.
        for slack, (g_max, s_min, s_max, a_max) in ((0.0, (13.5, 24, 37, -0.5)), (L.CUT_SLACK, (16, 19, 39, 2.5))):
            _, tight, rough = self.search(+1, trowel=True, slack=slack)
            self.assertEqual(len(tight), 0, slack)
            self.assertGreater(len(rough), 0, slack)
            gs = [g for g, _, _, _ in rough]
            ss = [s for _, s, _, _ in rough]
            self.assertGreaterEqual(min(gs), 5)
            self.assertLessEqual(max(gs), g_max)
            self.assertTrue(all(-s_max <= s <= -s_min for s in ss), (slack, min(ss), max(ss)))  # the near end
            self.assertTrue(all(-12 <= a <= a_max for _, _, a, _ in rough), slack)
        pose = rough[0][3]
        r, _ = play(procedure(trowel=TROWEL0, scale=pose))
        self.assertEqual(r.score, 0.5)

    def test_scale_beyond_the_trowel_earns_nothing(self):
        beyond = scale_pose(g=0.060)  # past the trowel, 58 mm+ from the find
        self.assertFalse(polys_overlap(footprint(obj("scale", beyond), L.SCALE_SIZE),
                                       footprint(obj("trowel", TROWEL0), L.TROWEL_SIZE)))
        r, _ = play(procedure(trowel=TROWEL0, scale=beyond))
        self.assertEqual(r.score, 0.0)

    def test_scale_held_at_floor_level_scores_zero(self):
        r, _ = play(procedure(arms_end=((*TIGHT[:2], L.TOP + 0.015), 0.21, REST_R, OPEN)))
        self.assertTrue(r.details["tight"] and r.details["stowed"])
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(arms_end=((*TIGHT[:2], L.TOP + 0.015), 0.6, REST_R, OPEN)))  # opened, but in the shot
        self.assertEqual(r.score, 0.0)

    def test_trowel_held_in_the_tray_scores_zero(self):
        grip = (L.TRAY_X - 0.045, L.TRAY_Y - 0.016, L.TOP + 0.025)  # outside the photo box, gripper closed
        r, _ = play(procedure(arms_end=(REST_L, OPEN, grip, 0.34)))
        self.assertTrue(r.details["stowed"])
        self.assertEqual(r.score, 0.0)

    def test_scale_on_the_trowel(self):
        on_trowel = (L.TROWEL_X, L.TROWEL_Y, L.TROWEL_SIZE[2], D20, 0.0)  # trowel left in the cut: no credit
        r, _ = play(procedure(trowel=TROWEL0, scale=on_trowel))
        self.assertFalse(r.details["scale_on_floor"])
        self.assertEqual(r.score, 0.0)
        in_tray = (L.TRAY_X, L.TRAY_Y, L.TRAY_FLOOR_H + L.TROWEL_SIZE[2], D20, 0.0)  # on the stowed trowel
        r, _ = play(procedure(scale=in_tray))
        self.assertEqual(r.score, 0.25)

    def test_scale_on_the_bank_beside_the_find_scores_quarter(self):
        # The ground surface above the cut: right distance, wrong level (it would read about 4 % large).
        bank = scale_pose(g=0.025, s=0.0, side=+1, bottom=L.BANK_H)
        _, _, m = placement(obj("scale", bank), obj("find", FIND0))
        self.assertLess(m["g"], 0.040)
        self.assertFalse(m["on_floor"])
        r, _ = play(procedure(scale=bank))
        self.assertEqual(r.score, 0.25)
        bank_near = (0.14, -0.01, L.BANK_H, D20, 0.0)
        r, _ = play(procedure(scale=bank_near))
        self.assertEqual(r.score, 0.25)

    def test_trowel_moved_to_a_bank_with_the_scale_tight_scores_half(self):
        r, _ = play(procedure(trowel=ON_RIGHT_BANK))
        self.assertEqual(r.score, 0.5)
        pushed = (0.20, -0.06, 0.0, 10.0, 0.0)  # pushed aside on the cut floor, out of the strip
        self.assertFalse(polys_overlap(footprint(obj("trowel", pushed), L.TROWEL_SIZE),
                                       footprint(obj("scale", TIGHT), L.SCALE_SIZE)))
        r, _ = play(procedure(trowel=pushed))
        self.assertEqual(r.score, 0.5)

    def test_trowel_left_on_the_tray_rim_is_not_stowed(self):
        r, _ = play(procedure(trowel=(L.TRAY_X, L.TRAY_Y, 0.030, 0.0, 0.0)))
        self.assertFalse(r.details["stowed"])
        self.assertEqual(r.score, 0.5)

    def test_shoving_the_find_to_make_room_scores_zero(self):
        shoved = (*(np.array([L.FIND_X, L.FIND_Y]) + 0.010 * N), 0.0, D20 + 5.0, 0.0)
        r, _ = play(procedure(trowel=TROWEL0, scale=scale_pose(g=0.012), find_path=[(shoved, 8)]))
        self.assertEqual(r.score, 0.0)

    def test_a_dropped_scale_that_lands_in_place_counts(self):
        f = procedure(settle=0, scale=lifted(TIGHT, 0.03))
        f.go(2, scale=TIGHT).hold(40)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_a_dropped_scale_that_bounces_into_the_find_scores_zero(self):
        f = procedure(settle=0, scale=lifted(TIGHT, 0.03))
        f.go(2, scale=TIGHT, find=(L.FIND_X - 0.007 * N[0], L.FIND_Y - 0.007 * N[1], 0.0, D20 + 2.0, 0.0)).hold(40)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_open_gripper_parked_just_off_the_frame_scores_zero(self):
        # A fully open gripper's pads reach about 56 mm from its TCP (half the 94 mm gap, the 8 mm pad, half its
        # 20 mm width): with only a 30 mm margin, a TCP at x 0.0799 hung its pads over the near bank in the shot.
        pad_reach = math.hypot(L.FULL_GAP / 2 + 0.008, 0.010)
        self.assertLess(pad_reach, L.CLEAR_MARGIN)
        near = (0.0799, -0.05, L.TOP + 0.020)
        r, _ = play(procedure(arms_end=(REST_L, OPEN, near, 0.5)))
        self.assertTrue(r.details["tight"] and r.details["stowed"])
        self.assertFalse(r.details["hands_clear"])
        self.assertEqual(r.score, 0.0)
        off = (L.FRAME[0] - L.CLEAR_MARGIN - 0.001, -0.05, L.TOP + 0.020)  # 61 mm off the frame: clear
        r, _ = play(procedure(arms_end=(REST_L, OPEN, off, 0.5)))
        self.assertEqual(r.score, 1.0)

    def test_tcp_past_the_far_side_with_the_arm_over_the_frame_scores_zero(self):
        # The left arm can reach about (0.48, 0.12) at table height, past the frame's far side: its upper arm then
        # passes over the far bank. The zone runs on past the far side, so that is not clear.
        past = (0.48, 0.12, L.TOP + 0.015)
        r, _ = play(procedure(arms_end=(past, OPEN, REST_R, OPEN)))
        self.assertFalse(r.details["hands_clear"])
        self.assertEqual(r.score, 0.0)

    def test_scale_left_standing_on_edge_or_tipped(self):
        self.assertEqual(final(scale=scale_pose(tilt=90.0)).score, 0.25)
        self.assertEqual(final(scale=scale_pose(tilt=7.0)).score, 0.25)


class Arms(unittest.TestCase):
    def test_unreadable_arms_are_not_clear_and_are_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])

    def test_scoring_resumes_when_the_arms_come_back(self):
        f = procedure()
        _, t = play(f, arms=False)
        fr = f.frames[-1]
        r = t.update(obj("find", fr["find"]), obj("trowel", fr["trowel"]), obj("scale", fr["scale"]), snap())
        self.assertEqual(r.score, 1.0)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(np.array(REST_R), np.array(REST_R), float("nan"))}
        r = t.update(obj("find", FIND0), obj("trowel", STOWED), obj("scale", TIGHT), bad)
        self.assertEqual((t.arms_unavailable, r.score), (1, 0.0))

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.3, 0.9), left_open=0.25, right_tcp=(0.1, -0.3, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        t = Tracker(HZ)
        r = t.update(obj("find", FIND0), obj("trowel", TROWEL0), obj("scale", SCALE0), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]),
                              high=np.zeros(3))
        zero_q = obj("scale", TIGHT)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=np.array(TIGHT[:3]))
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (t.ref[0].copy(), t.disturbed, t.max_move, [list(h) for h in t.hist.values()], t.prev,
                      t.steps)
            a = snap()
            good = (obj("find", FIND0), obj("trowel", STOWED), obj("scale", TIGHT))
            for k in range(3):
                args = list(good)
                args[k] = bad
                r = t.update(*args, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(*args, a).score, 0.0)
            self.assertTrue(np.allclose(t.ref[0], before[0]))
            self.assertEqual((t.disturbed, t.max_move), before[1:3])
            self.assertTrue(all(np.allclose(a, b) for h0, h1 in zip(before[3], t.hist.values())
                                for a, b in zip(h0, h1)))
            self.assertIs(t.prev, before[4])
            self.assertEqual(t.steps, before[5] + 3)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [obj("scale", (5.0, -3.0, -1.0, 0.0, 0.0)), obj("find", (0.3, 0.0, 0.0, 33.0, 120.0)),
               obj("trowel", (-0.2, 0.6, 0.0, 45.0, 90.0)), obj("scale", (*TIGHT[:3], D20, 89.9)),
               obj("find", (0.3, 0.0, 2.0, 0.0, 0.0))]
        t = Tracker(HZ)
        a = snap()
        for f, tr, s in zip(odd, odd[::-1], odd[1:] + odd[:1]):
            r = t.update(f, tr, s, a)
            self.assertIn(r.score, SCORES)
            self.assertFalse(r.success)


class Reset(unittest.TestCase):
    def disturbed(self):
        f = procedure(find_path=[((L.FIND_X + 0.010, L.FIND_Y, 0.0, D20, 0.0), 10)], settle=5)
        _, t = play(f)
        self.assertIsNotNone(t.disturbed)
        return t

    @staticmethod
    def disturb_restore_jump():
        """The find lifted 4 mm (latched 'lifted') and put back on its marks, the trowel untouched, the scale held
        25 mm over its place in the kit and dropped onto it (a 25 mm jump in one control step), then the whole task
        done properly."""
        f = Film().hold(5)
        f.go(10, find=(L.FIND_X, L.FIND_Y, 0.004, D20, 0.0)).go(10, find=FIND0)
        f.go(10, scale=lifted(SCALE0, L.KIT_FLOOR_H + 0.025)).go(1, scale=SCALE0)
        f.frames += procedure().frames[1:]
        return f

    def test_disturb_restore_and_jump_stays_disturbed_within_an_episode(self):
        # task.py passes rigstate's episode number: within one episode nothing clears the latch.
        r, t = play(self.disturb_restore_jump(), episode=3)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["disturbed_why"], "lifted")
        self.assertTrue(r.details["tight"] and r.details["stowed"] and r.details["hands_clear"])
        self.assertEqual(t.steps, len(self.disturb_restore_jump().frames))
        # The scorer used alone (no episode number) cannot tell this from a reset: its fallback starts over.
        r, _ = play(self.disturb_restore_jump())
        self.assertEqual(r.score, 1.0)

    def test_a_new_episode_number_starts_a_new_trial(self):
        f = procedure(find_path=[((L.FIND_X + 0.010, L.FIND_Y, 0.0, D20, 0.0), 10)], settle=5)
        r, t = play(f, episode=1)
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(), t, episode=1)  # the same episode, the task redone: still 0
        self.assertEqual(r.score, 0.0)
        r, _ = play(procedure(), t, episode=2)  # the next episode is scored on its own
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))

    def test_with_an_episode_number_a_teleport_is_not_a_reset(self):
        t = Tracker(HZ)
        play(procedure(find_path=[((L.FIND_X + 0.010, L.FIND_Y, 0.0, D20, 0.0), 10)], settle=5), t, episode=4)
        t.update(obj("find", FIND0), obj("trowel", TROWEL0), obj("scale", SCALE0), snap(), 4)
        self.assertIsNotNone(t.disturbed)

    def test_teleport_back_to_start_resets(self):
        # Without an episode number only (the scorer used on its own): everything teleported back onto its marks.
        t = self.disturbed()
        r = t.update(obj("find", FIND0), obj("trowel", TROWEL0), obj("scale", SCALE0), snap())
        self.assertIsNone(t.disturbed)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_gradual_return_does_not_reset(self):
        t = self.disturbed()
        f = Film(find=(L.FIND_X + 0.010, L.FIND_Y, 0.0, D20, 0.0), trowel=STOWED, scale=TIGHT, L=REST_L, R=REST_R)
        f.go(20, find=FIND0).go(10, trowel=lifted(STOWED)).go(40, trowel=lifted(TROWEL0)).go(10, trowel=TROWEL0)
        f.go(10, scale=lifted(TIGHT)).go(40, scale=lifted(SCALE0)).go(15, scale=SCALE0).hold(10)
        play(f, t)
        self.assertIsNotNone(t.disturbed)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.disturbed()
        t.update(obj("find", FIND0), obj("trowel", TROWEL0), obj("scale", (0.25, 0.20, 0.0, 0.0, 0.0)), snap())
        self.assertIsNotNone(t.disturbed)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.disturbed()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertIsNone(t.disturbed)
            self.assertIsNone(t.ref)
            self.assertEqual(t.steps, 0)
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
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo, and
    task.json agreeing with the code."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.meta = json.loads((HERE / "task.json").read_text())

    def objects(self, trowel=TROWEL0, scale=SCALE0, find=FIND0):
        return {"find": obj("find", find), "trowel": obj("trowel", trowel), "scale": obj("scale", scale)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = {o.name: o.mass for o in self.task.OBJECTS if o.mass is not None}
        self.assertEqual(loose, {"find": L.FIND_MASS, "trowel": L.TROWEL_MASS, "scale": L.SCALE_MASS})
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(self.task.SCENE, "station")

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        moved = (L.FIND_X + 0.008, L.FIND_Y, 0.0, D20, 0.0)
        self.assertFalse(self.task.success(self.objects()))
        self.assertFalse(self.task.success(self.objects(find=moved)))
        self.assertIsNotNone(tr.disturbed)
        self.assertEqual(self.task.score(self.objects(STOWED, TIGHT, moved)), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertIsNone(tr.disturbed)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertEqual(self.task.score(self.objects(STOWED, TIGHT)), 1.0)

    def test_disturb_restore_and_jump_through_the_task_hooks_scores_zero(self):
        # success() passes rigstate.episode(): the teleport-style return does not clear the latch within an episode.
        tr = self.task._TRACKER
        rigstate.offline_reset()
        for fr in Reset.disturb_restore_jump().frames:
            snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
            objs = self.objects(fr["trowel"], fr["scale"], fr["find"])
            self.assertFalse(self.task.success(objs))
        self.assertEqual(tr.disturbed[1], "lifted")
        self.assertEqual(self.task.score(objs), 0.0)
        rigstate.offline_reset()  # a real reset (rigstate's hook and a new episode number): scored afresh
        for fr in procedure().frames:
            snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"])
            objs = self.objects(fr["trowel"], fr["scale"], fr["find"])
            ok = self.task.success(objs)
        self.assertTrue(ok)
        self.assertEqual(self.task.score(objs), 1.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_points_and_timing(self):
        t = self.task
        self.assertTrue(np.allclose(t.TROWEL_GRIP, (0.2091, -0.0246), atol=1e-4))
        self.assertTrue(np.allclose(t.TROWEL_STOW, (0.2049, -0.3764), atol=1e-4))
        self.assertTrue(np.allclose(t.SCALE_PLACE, TIGHT[:2], atol=1e-9))
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        ends = {a: np.cumsum([w.duration for w in plan[a]]) for a in plan}
        self.assertAlmostEqual(ends["left"][-1], 25.3, places=6)
        self.assertAlmostEqual(ends["right"][-1], 17.6, places=6)
        # the left arm crosses toward the cut only once the right arm has the trowel over the tray
        i_cross = next(i for i, w in enumerate(plan["left"]) if "over the scale's place" in w.label)
        i_tray = next(i for i, w in enumerate(plan["right"]) if "over the tool tray" in w.label)
        self.assertGreaterEqual(ends["left"][i_cross - 1], ends["right"][i_tray] - 1e-9)
        for side, rest in (("left", t.LEFT_REST), ("right", t.RIGHT_REST)):
            last = plan[side][-1]
            self.assertEqual(last.finger, t.OPEN)
            self.assertTrue(hands_clear(snap(left_tcp=rest, right_tcp=t.RIGHT_REST if side == "left" else REST_R)))
        for w in plan["left"] + plan["right"]:
            self.assertTrue(-0.04695 - 1e-9 <= w.finger <= 0.0)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_task_json_agrees_with_the_code(self):
        m = self.meta
        self.assertEqual(m["instruction"], self.task.INSTRUCTION)
        self.assertEqual(m["category"], "Impact on jobs")
        self.assertIn("1,200 steps", m["run_limit"])
        self.assertIn("Possible scores: 0, 0.25, 0.5, 0.75, 1.0", m["scoring"])
        for text in ("x 26.5 cm, y +5.0 cm", "x 25.43 cm, y -0.82 cm", "x 25.0 cm, y -36.0 cm",
                     "x 25.0 cm, y +33.0 cm", "90 x 26 x 16 mm", "140 x 40 x 18 mm", "100 x 20 x 15 mm"):
            self.assertIn(text, m["setup"])
        for text in ("5 mm", "3 deg", "21 mm", "10 to 40 mm", "25 mm", "5 to 50 mm", "45 mm", "12 deg",
                     "at x 5 cm or more with y -21 to +21 cm", "0.5 s", "minus 23 mm",
                     "the trial video shows any lift, tilt or turn of the find"):
            self.assertIn(text, m["scoring"])
        self.assertIn("centred alongside it", m["instruction"])
        for field in ("setup", "fidelity"):
            self.assertIn("both 100 x 20 mm faces", m[field])
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{self.task.INSTRUCTION}"'), 2)
        self.assertNotIn("1 cm black bands on its top", readme + m["setup"])


if __name__ == "__main__":
    unittest.main()
