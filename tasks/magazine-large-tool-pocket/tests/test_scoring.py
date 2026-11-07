"""Offline tests for scoring.py and task.py: synthetic tool states, no simulator.
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
from scoring import Tracker, axis, lowest, over_disc, pocket_of, pose, shadow_gap, valid  # noqa: E402

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
PLATE = L.TOP + L.DISC_TOP  # the magazine plate's top (world z)
TRAY = L.TOP + L.TRAY_TOP


def quat(tilt_deg=0.0, yaw_deg=0.0):
    """(x, y, z, w): tilted tilt about the body's x axis, then turned yaw about the vertical."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    (x1, y1, z1, w1) = (0.0, 0.0, math.sin(a), math.cos(a))
    (x2, y2, z2, w2) = (math.sin(b), 0.0, 0.0, math.cos(b))
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def small(bottom, tilt=0.0, yaw=0.0):
    """A small tool whose (lower) bottom face is centred at ``bottom``, as the simulator reports a cylinder."""
    q = quat(tilt, yaw)
    c = np.asarray(bottom, dtype=float) + rot(q)[:, 2] * (L.TOOL_H / 2)
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3))


def small_lying(x, y, z_low, yaw=0.0):
    """A small tool on its side, its lowest line at z_low."""
    q = quat(90.0, yaw)
    return SimpleNamespace(pos=np.array([x, y, z_low + L.TOOL_R]), quat_xyzw=q, lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3))


def face(origin, tilt=0.0, yaw=0.0):
    """The face mill with its shank's bottom face centred at ``origin`` (mechkit's base frame)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=quat(tilt, yaw), pos=o + np.array([0.0, 0.0, L.FM_H / 2]),
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3))


def in_pocket(k, dx=0.0, dy=0.0, dz=0.0):
    px, py = L.POCKETS[k]
    return (px + dx, py + dy, PLATE + dz)


START = {
    **{n: small(in_pocket(k)) for n, k in L.LOADED.items()},
    **{n: small((*L.NEW_START[n], TRAY)) for n in L.NEW_SMALL},
    L.FACE: face((*L.FM_START, TRAY)),
}


def scene(**over):
    s = dict(START)
    s.update(over)
    return s


FULL = scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(3)), new_tool_b=small(in_pocket(4)))


def feed(tr, states):
    r = None
    for st in states:
        r = tr.update(st)
    return r


def settled(objs, n=WINDOW + 1):
    tr = Tracker(HZ)
    return tr, feed(tr, [objs] * n)


def scored(objs):
    return settled(objs)[1].score


def details(objs):
    return settled(objs)[1].details


class Geometry(unittest.TestCase):
    def test_pockets_on_the_ring(self):
        self.assertEqual(sorted(L.POCKETS), list(range(1, 9)))
        for k, (x, y) in L.POCKETS.items():
            self.assertAlmostEqual(math.hypot(x - L.MAG_X, y - L.MAG_Y), L.RING_R)
            self.assertAlmostEqual(math.degrees(math.atan2(y - L.MAG_Y, x - L.MAG_X)) % 360, 45.0 * (k - 1))
        self.assertGreater(L.POCKETS[1][0], L.MAG_X)  # P1 straight ahead
        self.assertGreater(L.POCKETS[3][1], 0.0)  # P3 on the robot's left
        self.assertLess(L.POCKETS[7][1], 0.0)  # P7 on its right
        self.assertAlmostEqual(L.PITCH, 0.0689, places=4)
        self.assertGreater(L.TAB_X, L.POCKETS[1][0] + L.POCKET_OUT / 2)  # the tab is beyond P1's far wall
        self.assertLess(L.TAB_X + L.TAB_SIZE[0] / 2, L.MAG_X + L.DISC_R)  # and on the plate

    def test_neighbours_wrap_round(self):
        self.assertEqual(L.neighbours(1), (8, 2))
        self.assertEqual(L.neighbours(8), (7, 1))
        self.assertEqual(L.neighbours(7), (6, 8))
        for k in range(1, 9):
            for j in L.neighbours(k):
                self.assertIn(k, L.neighbours(j))

    def test_the_rule_leaves_one_pocket_for_the_face_mill(self):
        loaded = set(L.LOADED.values())
        self.assertEqual(loaded, {1, 2, 5})
        ok = [k for k in range(1, 9) if k not in loaded and not loaded & set(L.neighbours(k))]
        self.assertEqual(ok, [L.FACE_POCKET])
        reserved = {L.FACE_POCKET, *L.neighbours(L.FACE_POCKET)}
        free = [k for k in range(1, 9) if k not in loaded | reserved]
        self.assertEqual(tuple(free), L.SMALL_POCKETS)

    def test_sockets_fit_on_the_plate_and_clear_each_other(self):
        h = L.POCKET_OUT / 2
        for k, (x, y) in L.POCKETS.items():
            for sx in (-1, 1):
                for sy in (-1, 1):
                    self.assertLess(math.hypot(x + sx * h - L.MAG_X, y + sy * h - L.MAG_Y), L.DISC_R)
            for j in L.neighbours(k):
                xj, yj = L.POCKETS[j]
                gap = max(abs(x - xj), abs(y - yj)) - L.POCKET_OUT  # the walls' outlines, axis-aligned squares
                self.assertGreater(gap, 0.003)

    def test_tools_fit_their_sockets(self):
        self.assertAlmostEqual(L.POCKET_IN / 2 - L.TOOL_R, 0.006)  # 6 mm of play either side
        self.assertAlmostEqual(L.POCKET_IN / 2 - L.SHANK_R, 0.006)
        self.assertGreater(2 * L.BODY_R, L.POCKET_IN)  # the cutter body is wider than a socket: it stays above
        # seated, the cutter body's underside is 15 mm above the socket walls
        self.assertAlmostEqual(L.SHANK_H - L.WALL_H, 0.015)
        # the cutter body (76 mm) would clear a seated neighbour's tool (40 mm) by 11 mm: the rule is the manual's
        self.assertGreater(L.PITCH - L.BODY_R - L.TOOL_R, 0.010)
        self.assertLessEqual(2 * L.BODY_R, 0.080)  # within the gripper's 80 mm
        self.assertLessEqual(max(L.TOOL_MASS, L.FM_MASS), 0.30)

    def test_start_marks(self):
        for n in L.NEW_SMALL:
            x, y = L.NEW_START[n]
            self.assertLessEqual(abs(x - L.LEFT_TRAY_X), L.LEFT_TRAY_SIZE[0] / 2 - L.TOOL_R)
            self.assertLessEqual(abs(y - L.LEFT_TRAY_Y), L.LEFT_TRAY_SIZE[1] / 2 - L.TOOL_R)
        self.assertAlmostEqual(L.FM_START[0], L.RIGHT_TRAY_X)
        self.assertLess(L.BODY_R, L.RIGHT_TRAY_SIZE[0] / 2)
        for n, st in START.items():
            self.assertTrue(valid(n, st))
            self.assertEqual(over_disc(n, st), n in L.LOADED)
        self.assertTrue(Tracker.at_start(START))
        self.assertFalse(Tracker.at_start(FULL))
        # nothing in the arms' start-pose shadow (x under 0.2 m and |y| over 0.45 m) or past the table's front
        for x, y in [*L.NEW_START.values(), L.FM_START]:
            self.assertLess(abs(y), 0.30)


class Pose(unittest.TestCase):
    def test_small_tool_either_end_down(self):
        up = small(in_pocket(3))
        b, t, tilt = axis("new_tool_a", up)
        np.testing.assert_allclose(b, in_pocket(3), atol=1e-12)
        self.assertAlmostEqual(tilt, 0.0, places=6)
        flipped = SimpleNamespace(pos=up.pos, quat_xyzw=quat(180.0))
        b2, _, tilt2 = axis("new_tool_a", flipped)
        np.testing.assert_allclose(b2, in_pocket(3), atol=1e-9)
        self.assertAlmostEqual(tilt2, 0.0, places=6)
        self.assertTrue(pose("new_tool_a", flipped)["down_in_pocket"])

    def test_face_mill_upside_down(self):
        fm = face((L.MAG_X, L.MAG_Y, PLATE + L.FM_H), tilt=180.0)
        _, top, tilt = axis(L.FACE, fm)
        self.assertAlmostEqual(tilt, 180.0, places=6)
        self.assertAlmostEqual(top[2], PLATE, places=9)
        self.assertAlmostEqual(lowest(L.FACE, fm), PLATE, places=9)

    def test_lowest_point(self):
        self.assertAlmostEqual(lowest("new_tool_a", small_lying(0.3, 0.0, PLATE)), PLATE, places=9)
        lean = small(in_pocket(3, dz=0.0), tilt=20.0)
        self.assertAlmostEqual(lowest("new_tool_a", lean), PLATE - L.TOOL_R * math.sin(math.radians(20.0)), places=9)
        self.assertAlmostEqual(lowest(L.FACE, face(in_pocket(7))), PLATE, places=12)
        # tilted 60 deg, the cutter body's rim is the lowest point
        fm = face((0.3, -0.09, PLATE + 0.05), tilt=60.0)
        s60, c60 = math.sin(math.radians(60.0)), math.cos(math.radians(60.0))
        self.assertAlmostEqual(lowest(L.FACE, fm), PLATE + 0.05 + min(0.0 - L.SHANK_R * s60,
                                                                      L.SHANK_H * c60 - L.BODY_R * s60,
                                                                      L.FM_H * c60 - L.BODY_R * s60), places=9)

    def test_pocket_of(self):
        for k, (x, y) in L.POCKETS.items():
            self.assertEqual(pocket_of((x, y)), k)
            self.assertEqual(pocket_of((x + 0.025, y - 0.025)), k)
            self.assertIsNone(pocket_of((x + 0.027, y)))
        self.assertIsNone(pocket_of((L.MAG_X, L.MAG_Y)))

    def test_over_disc(self):
        self.assertTrue(over_disc(L.FACE, face(in_pocket(7))))
        self.assertTrue(over_disc("new_tool_a", small((L.MAG_X + L.DISC_R + 0.018, 0.0, L.TOP))))
        self.assertFalse(over_disc("new_tool_a", small((L.MAG_X + L.DISC_R + 0.021, 0.0, L.TOP))))
        self.assertTrue(over_disc(L.FACE, face((L.MAG_X + L.DISC_R + 0.036, 0.0, L.TOP))))  # the 76 mm body overhangs
        self.assertFalse(over_disc(L.FACE, face((L.MAG_X + L.DISC_R + 0.040, 0.0, L.TOP))))
        # lying on the table, pointing at the plate: its end face 10 mm short of the rim is not over the plate
        rim = L.MAG_X + L.DISC_R
        self.assertFalse(over_disc("new_tool_a", small_lying(rim + 0.010 + L.TOOL_H / 2, 0.0, L.TOP, yaw=90.0)))
        self.assertTrue(over_disc("new_tool_a", small_lying(rim - 0.003 + L.TOOL_H / 2, 0.0, L.TOP, yaw=90.0)))
        # lying on the table along the rim (axis along y), its side overhanging by 2.7 mm against the plate's edge
        self.assertTrue(over_disc("new_tool_a", small_lying(rim + 0.0173, 0.0, L.TOP)))
        self.assertFalse(over_disc("new_tool_a", small_lying(rim + 0.022, 0.0, L.TOP)))
        # the face mill on its side on the table, shank toward the plate, shank end 5 mm short of the rim
        fm = face((rim + 0.005, 0.0, L.TOP + L.BODY_R), tilt=90.0, yaw=90.0)
        b, t, _ = axis(L.FACE, fm)
        self.assertGreater(t[0], b[0])  # axis pointing away from the plate
        self.assertFalse(over_disc(L.FACE, fm))

    def test_shadow_gap_matches_geometry(self):
        c = (L.MAG_X, L.MAG_Y)
        up = shadow_gap((0.5, 0.0, 0.0), (0.5, 0.0, 0.07), 0.02, c)  # upright: 200 mm off, radius 20
        self.assertAlmostEqual(up, 0.180, places=5)
        lying = shadow_gap((0.5, 0.0, 0.02), (0.57, 0.0, 0.02), 0.02, c)  # lying radially: the end face is 200 mm off
        self.assertAlmostEqual(lying, 0.200, places=5)
        across = shadow_gap((0.5, -0.035, 0.02), (0.5, 0.035, 0.02), 0.02, c)  # lying across: its side 180 mm off
        self.assertAlmostEqual(across, 0.180, places=5)
        self.assertLess(shadow_gap((0.29, 0.0, 0.0), (0.31, 0.0, 0.07), 0.02, c), 0.0)  # over the centre


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(details(START)["zero_reason"], "")

    def test_full_score(self):
        tr, r = settled(FULL)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["face_mill_pocket"], 7)
        self.assertEqual(sorted(r.details["small_tools_counted"]), ["new_tool_a", "new_tool_b"])
        swapped = scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(4)), new_tool_b=small(in_pocket(3)))
        self.assertEqual(scored(swapped), 1.0)

    def test_every_tier(self):
        cases = {
            0.75: scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(3))),
            0.5: scene(face_mill=face(in_pocket(7))),
            0.25: scene(new_tool_b=small(in_pocket(4))),
        }
        for want, s in cases.items():
            with self.subTest(want=want):
                r = settled(s)[1]
                self.assertEqual(r.score, want)
                self.assertFalse(r.success)
        self.assertEqual(scored(scene(new_tool_a=small(in_pocket(3)), new_tool_b=small(in_pocket(4)))), 0.5)
        self.assertEqual(scored(scene(face_mill=face(in_pocket(7)), new_tool_b=small(in_pocket(4)))), 0.75)

    def test_seated_anywhere_in_the_play(self):
        s = scene(face_mill=face(in_pocket(7, dx=-0.006, dy=0.006)), new_tool_a=small(in_pocket(3, dx=0.006)),
                  new_tool_b=small(in_pocket(4, dy=-0.006, dz=0.0025), tilt=4.0))
        self.assertEqual(scored(s), 1.0)

    def test_a_tool_not_down_is_not_seated(self):
        high = scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(3)),
                     new_tool_b=small(in_pocket(4, dz=0.004)))  # 4 mm up: resting on something, or still held
        r = settled(high)[1]
        self.assertEqual(r.score, 0.0)
        self.assertIn("tool_loose_on_the_magazine", r.details["zero_reason"])

    def test_off_magazine_tools_earn_nothing_but_do_not_zero(self):
        floor = small((0.3, 0.6, 0.0))  # fell off the table
        self.assertEqual(scored(scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(3)),
                                      new_tool_b=floor)), 0.75)
        table = small_lying(0.15, 0.17, L.TOP)  # dropped on the table beside the magazine
        self.assertFalse(over_disc("new_tool_a", table))
        self.assertEqual(scored(scene(face_mill=face(in_pocket(7)), new_tool_a=table)), 0.5)
        held = small(in_pocket(4, dz=L.ON_MAGAZINE_H + 0.002))  # held up over P4, its bottom 77 mm above the plate
        self.assertEqual(scored(scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(3)),
                                      new_tool_b=held)), 0.75)
        fm_on_tray_side = face((L.FM_START[0], L.FM_START[1], TRAY), tilt=90.0)  # knocked over on its tray
        self.assertEqual(scored(scene(face_mill=fm_on_tray_side, new_tool_a=small(in_pocket(3)))), 0.25)


class Zero(unittest.TestCase):
    def assertZero(self, s, reason):
        r = settled(s)[1]
        self.assertEqual(r.score, 0.0)
        self.assertIn(reason, r.details["zero_reason"])
        return r

    def test_face_mill_next_to_a_tool(self):
        for k in (3, 4, 6, 8):
            with self.subTest(pocket=k):
                others = {"new_tool_a": small(in_pocket(7))} if k in (6, 8) else {}
                r = self.assertZero(scene(face_mill=face(in_pocket(k)), **others), "face_mill_next_to_a_tool")
                self.assertTrue(r.details["face_mill_neighbours_with_tools"])

    def test_small_tool_next_to_the_face_mill(self):
        for k in (6, 8):
            with self.subTest(pocket=k):
                s = scene(face_mill=face(in_pocket(7)), new_tool_a=small(in_pocket(k)), new_tool_b=small(in_pocket(3)))
                r = self.assertZero(s, "face_mill_next_to_a_tool")
                self.assertEqual(r.details["face_mill_neighbours_with_tools"], [k])
                self.assertEqual(r.details["tier"], 0.25)  # the face mill earns nothing next to a tool

    def test_small_tool_in_the_face_mills_place_earns_nothing(self):
        for k in (6, 7, 8):
            with self.subTest(pocket=k):
                s = scene(new_tool_a=small(in_pocket(k)), new_tool_b=small(in_pocket(3)))
                r = settled(s)[1]
                self.assertEqual(r.score, 0.25)
                self.assertEqual(r.details["zero_reason"], "")

    def test_a_loaded_tool_moved(self):
        self.assertZero(scene(tool_p1=small(in_pocket(8)), face_mill=face(in_pocket(2))),
                        "loaded_pocket_without_a_loaded_tool")
        self.assertZero(scene(tool_p2=small((0.20, 0.13, L.TOP)), face_mill=face(in_pocket(7)),
                              new_tool_a=small(in_pocket(3)), new_tool_b=small(in_pocket(4))),
                        "loaded_pocket_without_a_loaded_tool")
        self.assertZero(scene(tool_p5=small_lying(L.MAG_X, L.MAG_Y, PLATE)), "loaded_pocket_without_a_loaded_tool")
        self.assertZero(FULL | {"tool_p1": small(in_pocket(1, dz=0.010))}, "loaded_pocket_without_a_loaded_tool")
        # lifted out and put back in its own pocket: fine
        self.assertEqual(scored(FULL | {"tool_p1": small(in_pocket(1, dx=0.004))}), 1.0)
        r = settled(scene(tool_p2=small((0.20, 0.13, L.TOP))))[1]
        self.assertEqual(r.details["loaded_pockets_missing"], [2])
        # a blue tool where a loaded one stood does not stand in for it
        self.assertZero(scene(tool_p1=small((0.20, 0.13, L.TOP)), new_tool_a=small(in_pocket(1))),
                        "loaded_pocket_without_a_loaded_tool")

    def test_loaded_tools_swapped_look_the_same(self):
        # the three loaded tools look alike: a swap among P1, P2 and P5 cannot be seen in a photo and is not scored
        swap = FULL | {"tool_p1": small(in_pocket(2)), "tool_p2": small(in_pocket(5)), "tool_p5": small(in_pocket(1))}
        r = settled(swap)[1]
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["loaded_pockets_missing"], [])

    def test_loose_on_the_magazine(self):
        loose = {
            "lying on the plate": small_lying(L.MAG_X, L.MAG_Y, PLATE),
            "lying across the walls of P6": small_lying(L.POCKETS[6][0], L.POCKETS[6][1], PLATE + L.WALL_H),
            "standing on the plate between pockets": small((L.MAG_X, L.MAG_Y, PLATE)),
            "standing on a loaded tool": small(in_pocket(1, dz=L.TOOL_H)),
            "leaning in P8": small(in_pocket(8, dz=L.TOOL_R * math.sin(math.radians(20.0))), tilt=20.0),
            "tilted 6 degrees in P4": small(in_pocket(4, dz=0.002), tilt=6.0),
            "held low over P4": small(in_pocket(4, dz=0.030)),
            "standing on the table, leaning on the plate's rim": small(
                (L.MAG_X + L.DISC_R + 0.024, 0.0, L.TOP + L.TOOL_R * math.sin(math.radians(25.0))),
                tilt=25.0, yaw=-90.0),
        }
        for what, st in loose.items():
            with self.subTest(what=what):
                self.assertZero(FULL | {"new_tool_b": st}, "tool_loose_on_the_magazine")

    def test_face_mill_loose(self):
        cases = {
            "upside down in the middle": face((L.MAG_X, L.MAG_Y, PLATE + L.FM_H), tilt=180.0),
            "on its side on the plate": face((L.MAG_X, L.MAG_Y + 0.035, PLATE + L.BODY_R), tilt=90.0),
            "shank on P7's wall tops": face(in_pocket(7, dx=0.030, dz=L.WALL_H)),
            "held 20 mm up over P7": face(in_pocket(7, dz=0.020)),
        }
        for what, fm in cases.items():
            with self.subTest(what=what):
                self.assertZero(FULL | {L.FACE: fm}, "tool_loose_on_the_magazine")

    def test_still_moving(self):
        tr, _ = settled(FULL)
        r = tr.update(FULL | {"new_tool_b": small(in_pocket(4, dx=0.003))})  # a 3 mm shift inside the window
        self.assertEqual(r.score, 0.0)
        self.assertIn("tool_loose_on_the_magazine", r.details["zero_reason"])
        r = feed(tr, [FULL | {"new_tool_b": small(in_pocket(4, dx=0.003))}] * WINDOW)
        self.assertEqual(r.score, 1.0)  # at rest again after 0.5 s

    def test_small_jitter_is_rest(self):
        tr = Tracker(HZ)
        rng = np.random.default_rng(3)
        r = None
        for _ in range(3 * WINDOW):
            j = rng.uniform(-0.0006, 0.0006, size=2)
            r = tr.update(FULL | {"new_tool_a": small(in_pocket(3, dx=j[0], dy=j[1]))})
        self.assertEqual(r.score, 1.0)

    def test_needs_half_a_second(self):
        tr = Tracker(HZ)
        r = feed(tr, [FULL] * WINDOW)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.update(FULL).score, 1.0)


class Robustness(unittest.TestCase):
    def test_invalid_states_score_zero_and_change_nothing(self):
        tr, _ = settled(FULL)
        n, hist = tr.steps, {k: list(v) for k, v in tr.history.items()}
        bad = [
            FULL | {"new_tool_a": SimpleNamespace(pos=np.array([np.nan, 0, 0]), quat_xyzw=quat())},
            FULL | {"new_tool_a": SimpleNamespace(pos=np.array([0.3, 0.09, 0.8]), quat_xyzw=np.zeros(4))},
            FULL | {L.FACE: SimpleNamespace(pos=np.array([0.3, -0.09, 0.8]), quat_xyzw=quat())},  # no origin
            FULL | {L.FACE: SimpleNamespace(origin=np.array([0.3, np.inf, 0.8]), quat_xyzw=quat())},
            {k: v for k, v in FULL.items() if k != "tool_p2"},
        ]
        for s in bad:
            r = tr.update(s)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.steps, n + len(bad))
        for k, v in hist.items():
            self.assertEqual(len(tr.history[k]), len(v))
            for (a, b), (c, d) in zip(tr.history[k], v):
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, d)
        self.assertEqual(tr.update(FULL).score, 1.0)  # and it carries on

    def test_unnormalised_quaternion(self):
        fm = face(in_pocket(7))
        fm.quat_xyzw = 3.0 * fm.quat_xyzw
        self.assertEqual(scored(FULL | {L.FACE: fm}), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(11)
        tr = Tracker(HZ)
        names = list(L.TOOL_NAMES)
        for _ in range(300):
            s = dict(FULL)
            for n in rng.choice(names, size=2, replace=False):
                p = rng.uniform([0.0, -0.4, 0.6], [0.6, 0.4, 1.0])
                q = rng.normal(size=4)
                s[n] = (SimpleNamespace(origin=p, quat_xyzw=q, pos=p) if n == L.FACE
                        else SimpleNamespace(pos=p, quat_xyzw=q))
            r = tr.update(s)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_result_is_idempotent(self):
        tr, r = settled(FULL)
        n = tr.steps
        self.assertEqual(tr.result(FULL).score, r.score)
        self.assertEqual(tr.steps, n)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(FULL)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(all(len(h) == 1 for h in tr.history.values()))

    def test_jittered_teleport_resets(self):
        tr, _ = settled(FULL)
        jit = scene(new_tool_a=small((L.NEW_START["new_tool_a"][0] + 0.03, L.NEW_START["new_tool_a"][1] - 0.03, TRAY)),
                    face_mill=face((L.FM_START[0] - 0.03, L.FM_START[1] + 0.03, TRAY)))
        tr.update(jit)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(FULL)
        a0, a1 = np.array(in_pocket(3)), np.array((*L.NEW_START["new_tool_a"], TRAY))
        for i in range(1, 41):
            tr.update(FULL | {"new_tool_a": small(a0 + (a1 - a0) * i / 40)})
        self.assertEqual(tr.steps, WINDOW + 1 + 40)

    def test_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(FULL)
        r = tr.update(FULL | {"new_tool_a": small(in_pocket(8))})
        self.assertEqual(tr.steps, WINDOW + 2)
        self.assertEqual(r.score, 0.0)  # moved in the window

    def test_explicit_reset(self):
        tr, _ = settled(FULL)
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(tr.prev_bottom, {})


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

    def sim(self, objs):
        """Synthetic states with the face mill turned into the AssemblyState mechkit would report."""
        out = dict(objs)
        fm = objs[L.FACE]
        out[L.FACE] = self.mk.state_from_poses(self.task.FACE_MILL, (*fm.origin, *fm.quat_xyzw), source="sim")
        return out

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = sorted(o.name for o in t.OBJECTS if o.mass is not None)
        self.assertEqual(loose, sorted(L.TOOL_NAMES))
        self.assertTrue(all(o.mass <= 0.30 for o in t.OBJECTS if o.mass is not None))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 2.5)
        self.assertEqual(len(t.WALLS), 32)
        by = {o.name: o for o in t.OBJECTS}
        for k, (px, py) in L.POCKETS.items():
            f, b = by[f"p{k}_wall_front"], by[f"p{k}_wall_back"]
            lt, rt = by[f"p{k}_wall_left"], by[f"p{k}_wall_right"]
            self.assertAlmostEqual((f.pos[0] - f.size[0] / 2) - (b.pos[0] + b.size[0] / 2), L.POCKET_IN)
            self.assertAlmostEqual((lt.pos[1] - lt.size[1] / 2) - (rt.pos[1] + rt.size[1] / 2), L.POCKET_IN)
            self.assertAlmostEqual((f.pos[0] + b.pos[0]) / 2, px)
            self.assertAlmostEqual((lt.pos[1] + rt.pos[1]) / 2, py)
            for w in (f, b, lt, rt):
                self.assertAlmostEqual(w.pos[2] - w.size[2] / 2, PLATE)
                self.assertAlmostEqual(w.pos[2] + w.size[2] / 2, L.TOP + L.WALL_TOP)
        for n, k in L.LOADED.items():
            self.assertAlmostEqual(by[n].pos[2] - L.TOOL_H / 2, PLATE)
            self.assertEqual(pocket_of(by[n].pos), k)
        self.assertAlmostEqual(t.PLATE.pos[2] + t.PLATE.height / 2, PLATE)

    def test_face_mill_compound(self):
        fm = self.task.FACE_MILL
        self.assertEqual(fm.kind, "rigid")
        self.assertAlmostEqual(fm.mass, L.FM_MASS)
        lo, hi = fm.local_bounds()
        np.testing.assert_allclose(lo, [-L.BODY_R, -L.BODY_R, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [L.BODY_R, L.BODY_R, L.FM_H], atol=1e-12)
        com = fm.center_of_mass()
        self.assertAlmostEqual(com[2], 0.0456, places=4)
        # held by the cutter body with the pad tips 4 mm above its underside: the centre of mass is between the pads
        self.assertGreater(com[2], L.SHANK_H + 0.004 - 0.002)
        # standing on its shank it tips only past 23 degrees
        self.assertGreater(math.degrees(math.atan2(L.SHANK_R, com[2])), 20.0)
        rest = fm.rest_state()
        self.assertTrue(Tracker.at_start(scene(face_mill=rest)))
        st = self.sim(FULL)[L.FACE]
        self.assertAlmostEqual(st.low[2], PLATE, places=9)
        self.assertEqual(settled(self.sim(FULL))[1].score, 1.0)

    def test_offline_state_path(self):
        fm = self.task.FACE_MILL
        centre, _, _ = fm.offline_box()
        box = SimpleNamespace(pos=np.array(centre), quat_xyzw=quat(), lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = fm.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [*L.FM_START, TRAY], atol=1e-12)
        self.assertTrue(Tracker.at_start(scene(face_mill=st)))

    def test_hooks_and_mechkit_reset(self):
        t = self.task
        tr = t._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        full = self.sim(FULL)
        for _ in range(WINDOW):
            t.success(full)
        self.assertTrue(t.success(full))
        self.assertEqual(t.score(full), 1.0)
        self.assertEqual(t.score(full), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(t.success(full))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(FULL))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "tier", "zero_reason", "face_mill_pocket",
                          "face_mill_neighbours_with_tools", "small_tools_counted", "loaded_pockets_missing",
                          "loose_on_magazine", "pockets", "seated", "tilt_deg", "floor_mm", "centre_off_mm", "low_mm",
                          "moved_mm", "source", "face_mill_origin", "face_mill_quat", "tool_pos"):
                    self.assertIn(k, row)
                self.assertEqual(row["pockets"]["face_mill"], 7)
                self.assertEqual(row["source"], "sim")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.task.success(self.sim(FULL))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def _times(self, wps):
        out, t = [], 0.0
        for w in wps:
            t += w.duration
            out.append((t - w.duration, t, w))
        return out

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, wps in plan.items():
            lows = [z for z in (w.pos[2] for w in wps)]
            grip_place = {round(z, 9) for z in lows if z <= min(lows) + 0.02}
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes 1.5 s
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                if prev.pos[2] > w.pos[2] and round(w.pos[2], 9) in grip_place:  # final descents: slow, 30 mm
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertLessEqual(prev.pos[2] - w.pos[2], t.APPROACH + 1e-9)
        # grips 8 mm narrower than the part
        self.assertAlmostEqual(-2 * t.TOOL_GRIP, 2 * L.TOOL_R - 0.008)
        self.assertAlmostEqual(-2 * t.FM_GRIP, 2 * L.BODY_R - 0.008)
        left, right = plan["left"], plan["right"]
        lets = {w.label: w for w in left + right if w.label.startswith("let go")}
        self.assertEqual(set(lets), {"let go in P3", "let go in P4", "let go in P7"})
        for lab, k, up in (("let go in P3", 3, t.TOOL_GRIP_UP), ("let go in P4", 4, t.TOOL_GRIP_UP),
                           ("let go in P7", 7, t.FM_GRIP_UP)):
            w = lets[lab]
            np.testing.assert_allclose(w.pos[:2], L.POCKETS[k], atol=1e-12)
            self.assertAlmostEqual(w.pos[2] - up - PLATE, t.DROP)
        self.assertTrue(0.0 < t.DROP <= 0.003)  # let go just above the plate: a 2 mm drop, no fall
        # pad tips 36 mm above a small tool's bottom; on the face mill 4 mm above the cutter body's underside
        grips = {w.label: w for w in left + right if w.label.startswith("grip")}
        self.assertAlmostEqual(grips["grip the small tool"].pos[2] - t.PAD_TIP - TRAY, 0.036)
        self.assertAlmostEqual(grips["grip the face mill"].pos[2] - t.PAD_TIP - TRAY, L.SHANK_H + 0.004)
        self.assertLess(0.036 + 0.025 + t.PAD_TIP, L.TOOL_H + 0.013)  # the pads' back end is near the tool's top
        # carried clear of the loaded tools' tops and the socket walls
        self.assertGreaterEqual(t.HOVER_L - t.TOOL_GRIP_UP - (PLATE + L.TOOL_H), 0.030)
        self.assertGreaterEqual(t.HOVER_R - t.FM_GRIP_UP - (L.TOP + L.WALL_TOP), 0.050)

    def test_pads_open_away_from_the_neighbours(self):
        """At P3 and P4 the fingers close along the radius, so the fully open pads (8 mm thick, 20 mm wide, their
        inner faces 47 mm from the axis) stay clear of the loaded tool next door; at P7 the neighbours are empty."""
        t = self.task
        half_gap = -t.OPEN
        for k, yaw in ((3, t.YAW_P3), (4, t.YAW_P4)):
            close = np.array([-math.sin(yaw), math.cos(yaw)])
            radial = np.array(L.POCKETS[k]) - np.array([L.MAG_X, L.MAG_Y])
            self.assertAlmostEqual(abs(float(close @ radial)) / np.linalg.norm(radial), 1.0, places=9)
            side = np.array([-close[1], close[0]])
            for sign in (-1, 1):
                pc = np.array(L.POCKETS[k]) + sign * close * (half_gap + 0.004)
                for j in L.neighbours(k):
                    d = np.array(L.POCKETS[j]) - pc
                    along, across = abs(float(d @ close)), abs(float(d @ side))
                    gap = math.hypot(max(along - 0.004, 0.0), max(across - 0.010, 0.0)) - L.TOOL_R
                    self.assertGreater(gap, 0.020, (k, j, sign))
        self.assertNotIn(L.FACE_POCKET, L.LOADED.values())
        for j in L.neighbours(L.FACE_POCKET):
            self.assertNotIn(j, L.LOADED.values())

    def test_face_mill_clears_the_walls_on_the_way_down(self):
        """The offline checker draws the face mill as its 76 x 76 x 70 mm bounding box and so reports it running
        into P7's walls (and P6's and P8's corners); its real shapes do not: the 40 mm shank goes into the 52 mm
        opening and the 76 mm cutter stays above the 25 mm walls. Checked with the shapes, along the demo's whole
        descent into P7 to the let-go height (the bottom 2 mm above the plate) and down to the plate."""
        t = self.task
        walls = [w for w in t.WALLS]
        px, py = L.POCKETS[L.FACE_POCKET]
        place = [w for w in t.demo()["right"] if w.label == "let go in P7"][0]
        lowest_bottom = place.pos[2] - t.FM_GRIP_UP
        self.assertAlmostEqual(lowest_bottom - PLATE, t.DROP)
        worst = math.inf
        for bottom in np.linspace(PLATE, t.HOVER_R - t.FM_GRIP_UP, 60):
            for w in walls:
                wx0, wx1 = w.pos[0] - w.size[0] / 2, w.pos[0] + w.size[0] / 2
                wy0, wy1 = w.pos[1] - w.size[1] / 2, w.pos[1] + w.size[1] / 2
                wz0, wz1 = w.pos[2] - w.size[2] / 2, w.pos[2] + w.size[2] / 2
                dx, dy = max(wx0 - px, 0.0, px - wx1), max(wy0 - py, 0.0, py - wy1)
                horiz = math.hypot(dx, dy)
                for z0, z1, r in ((bottom, bottom + L.SHANK_H, L.SHANK_R), (bottom + L.SHANK_H, bottom + L.FM_H,
                                                                              L.BODY_R)):
                    if z0 < wz1 and z1 > wz0:  # the same heights: they must be apart seen from above
                        worst = min(worst, horiz - r)
                    else:
                        worst = min(worst, math.hypot(max(horiz - r, 0.0), max(z0 - wz1, wz0 - z1)))
        self.assertGreater(worst, 0.005)

    def test_arms_take_turns_near_the_midline(self):
        t = self.task
        left, right = self._times(t.demo()["left"]), self._times(t.demo()["right"])
        p3_done = [e for s, e, w in left if w.label == "straight up"][0]
        p4_start = [s for s, e, w in left if w.label == "carry it over P4"][0]
        p7_start = [e for s, e, w in right if w.label == "carry it over P7"][0]
        p7_done = [e for s, e, w in right if w.label == "straight up"][0]
        self.assertGreaterEqual(p7_start, p3_done)  # the right arm is over P7 only after the left has left P3
        self.assertLessEqual(p7_done, p4_start)  # and is up out of P7 before the left comes in to P4
        self.assertGreater(t.RIGHT_WAIT, 0.0)

    def test_idle_switches(self):
        t = self.task
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("empty pockets", "face mill is a large tool", "both pockets next to it must stay empty",
                  "Leave the tools already in the magazine where they are", "let go"):
            self.assertIn(s, ins)
        with open(HERE / "task.json") as f:
            self.assertEqual(json.load(f)["instruction"], ins)


if __name__ == "__main__":
    unittest.main()
