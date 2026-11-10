"""Offline tests for scoring.py and task.py: synthetic trap states, no simulator.
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
from scoring import (  # noqa: E402
    NAMES,
    Tracker,
    corners,
    end_gap,
    pedal_end_mid,
    pedal_heading,
    rotation,
    tilt_deg,
    trap_facts,
    valid,
    wall_angle_deg,
)

HZ = L.CONTROL_HZ
WINDOW = int(round(L.REST_WINDOW_S * HZ)) + 1  # 11 samples: 0.5 s of stillness
PATCHES = tuple(L.PATCHES)
AIM = 0.004  # the demo sets each pedal end 4 mm from the wall


def quat(yaw_deg=0.0, tilt_deg_=0.0, tilt_axis="x"):
    """Turned yaw about the vertical after tilting about the body's x (or y) axis, as (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg_) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qt = np.array([math.sin(b), 0.0, 0.0, math.cos(b)]) if tilt_axis == "x" else \
        np.array([0.0, math.sin(b), 0.0, math.cos(b)])
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qt
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def trap(x, y, yaw=0.0, low=None, tilt=0.0, tilt_axis="x", vel=(0, 0, 0), spin=(0, 0, 0)):
    """A trap's state as mechkit reports it: its frame's origin at (x, y) with the base's lowest corner at ``low``
    (default: on the floor), ``yaw`` the heading of its plain end (degrees), and the bounding box of every shape."""
    q = quat(yaw, tilt, tilt_axis)
    r = rotation(q)
    base = np.array([(sx * L.TRAP_HALF_LEN, sy * L.TRAP_W / 2, sz * L.TRAP_BASE_H)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    oz = (L.FLOOR_TOP if low is None else low) - float((base @ r.T)[:, 2].min())
    origin = np.array([x, y, oz])
    pts = []
    for c, s, _ in L.TRAP_SHAPES:
        h = np.array(s) / 2
        for sx, sy, sz in itertools.product((-1, 1), repeat=3):
            pts.append(origin + r @ (np.array(c) + h * np.array([sx, sy, sz])))
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=origin, pos=(lo + hi) / 2, quat_xyzw=q, lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=lo, high=hi)


def shifted(s, dx=0.0, dy=0.0, dz=0.0):
    d = np.array([dx, dy, dz])
    return SimpleNamespace(origin=s.origin + d, pos=s.pos + d, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel,
                           ang_vel=s.ang_vel, low=s.low + d, high=s.high + d)


def set_trap(patch, gap=AIM, along=0.0, turn=0.0, reverse=False, **kw):
    """A trap at this patch: pedal end toward its wall (or away, ``reverse``), turned ``turn`` degrees off square,
    its nearest point ``gap`` from the wall's face and its pedal end's midpoint ``along`` from the patch's centre."""
    px, wall = L.PATCHES[patch]
    k = L.WALLS[wall]
    heading = -90.0 * k + turn + (180.0 if reverse else 0.0)  # the plain end points away from the wall
    s = trap(px, k * (L.FACE_Y - 0.06), heading, **kw)
    s = shifted(s, dy=k * (end_gap(s, wall) - gap))
    return shifted(s, dx=px + along - float(pedal_end_mid(s)[0]))


def start_trap(name):
    x, y, yaw = L.TRAP_START[name]
    return trap(x, y, math.degrees(yaw))


START = {n: start_trap(n) for n in NAMES}
DONE = {n: set_trap(p) for n, p in zip(NAMES, PATCHES)}


def scene(**over):
    """The starting layout with some traps replaced (by name)."""
    out = dict(START)
    out.update(over)
    return out


def run(frames, t=None):
    """Feed the frames to a tracker (a fresh one unless given); return the last result and the tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for f in frames:
        r = t.update(f)
    return r, t


def score_of(sc):
    """The score of a scene held still for 0.5 s."""
    return run([sc] * WINDOW)[0].score


def glide(a, b, n):
    """n frames moving every trap from scene a to scene b in equal steps (the orientation of b)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        frame = {}
        for k in NAMES:
            sa, sb = a[k], b[k]
            frame[k] = SimpleNamespace(origin=(1 - f) * sa.origin + f * sb.origin, pos=(1 - f) * sa.pos + f * sb.pos,
                                       quat_xyzw=sb.quat_xyzw, lin_vel=(sb.pos - sa.pos) * HZ / n,
                                       ang_vel=np.zeros(3), low=(1 - f) * sa.low + f * sb.low,
                                       high=(1 - f) * sa.high + f * sb.high)
        out.append(frame)
    return out


class Geometry(unittest.TestCase):
    def test_pedal_end_and_heading(self):
        s = start_trap("trap_1")  # pedal end toward the arms
        self.assertAlmostEqual(float(pedal_end_mid(s)[0]), 0.300 - L.TRAP_HALF_LEN, places=9)
        self.assertTrue(np.allclose(pedal_heading(rotation(s.quat_xyzw)), [-1.0, 0.0]))
        s = start_trap("trap_2")  # pedal end away from them
        self.assertAlmostEqual(float(pedal_end_mid(s)[0]), 0.420 + L.TRAP_HALF_LEN, places=9)
        self.assertTrue(np.allclose(pedal_heading(rotation(s.quat_xyzw)), [1.0, 0.0]))
        self.assertIsNone(pedal_heading(rotation(quat(0.0, 90.0, "y"))))  # standing on an end: no heading

    def test_wall_angles(self):
        r = rotation(quat(-90.0))  # plain end toward -y: pedal toward +y, the left wall
        self.assertAlmostEqual(wall_angle_deg(r, "left"), 0.0, places=6)
        self.assertAlmostEqual(wall_angle_deg(r, "right"), 180.0, places=6)
        self.assertAlmostEqual(wall_angle_deg(rotation(quat(0.0)), "left"), 90.0, places=6)  # parallel to the walls
        self.assertAlmostEqual(wall_angle_deg(rotation(quat(-75.0)), "left"), 15.0, places=6)
        self.assertAlmostEqual(wall_angle_deg(rotation(quat(-105.0)), "left"), 15.0, places=6)
        self.assertEqual(wall_angle_deg(rotation(quat(0.0, 90.0, "y")), "left"), 90.0)

    def test_tilt(self):
        self.assertAlmostEqual(tilt_deg(rotation(quat(30.0, 8.0))), 8.0, places=5)
        self.assertAlmostEqual(tilt_deg(rotation(quat(0.0, 180.0))), 180.0, places=5)  # upside down is not flat
        self.assertAlmostEqual(tilt_deg(rotation(quat(0.0, 90.0, "y"))), 90.0, places=5)

    def test_end_gap_and_along(self):
        for p in PATCHES:
            s = set_trap(p, gap=0.0057, along=-0.011, turn=7.0)
            wall = L.PATCHES[p][1]
            self.assertAlmostEqual(end_gap(s, wall), 0.0057, places=9)
            self.assertAlmostEqual(float(pedal_end_mid(s)[0]) - L.PATCHES[p][0], -0.011, places=9)
            f = trap_facts(s)
            self.assertEqual(f["faces"], wall)
            self.assertAlmostEqual(f["angles"][wall], 7.0, places=6)
            self.assertTrue(f["on_floor"] and f["base_down"])
        # The base's corners: a trap lying flat has four at the floor and four 10 mm up.
        c = corners(START["trap_1"])
        self.assertTrue(np.allclose(sorted(c[:, 2]), [L.FLOOR_TOP] * 4 + [L.FLOOR_TOP + L.TRAP_BASE_H] * 4))

    def test_demo_aim_is_the_set_pose(self):
        """A trap set as the demo sets it (origin 4 mm + half its length from the wall) has its pedal end 4 mm off."""
        s = trap(0.340, L.FACE_Y - AIM - L.TRAP_HALF_LEN, -90.0)
        self.assertAlmostEqual(end_gap(s, "left"), AIM, places=9)

    def test_valid(self):
        good = START["trap_1"]
        self.assertTrue(valid(good))
        for key, val in (("origin", np.array([np.nan, 0.0, 0.76])), ("quat_xyzw", np.zeros(4)),
                         ("pos", np.array([np.inf, 0.0, 0.0])), ("lin_vel", np.array([0.0, np.nan, 0.0])),
                         ("origin", np.zeros(2)), ("quat_xyzw", np.zeros(3) + 0.5)):
            bad = SimpleNamespace(**vars(good))
            setattr(bad, key, val)
            self.assertFalse(valid(bad), key)
        no_origin = SimpleNamespace(**{k: v for k, v in vars(good).items() if k != "origin"})
        self.assertFalse(valid(no_origin))  # built as a plain box, not a compound
        self.assertFalse(valid(None))
        self.assertFalse(valid(SimpleNamespace(origin="x", pos=good.pos, quat_xyzw=good.quat_xyzw,
                                               lin_vel=good.lin_vel, ang_vel=good.ang_vel, low=good.low,
                                               high=good.high)))

    def test_layout_arithmetic(self):
        for p in PATCHES:
            x, y = L.patch_xy(p)
            self.assertAlmostEqual(L.FACE_Y - abs(y), L.PATCH_FROM_WALL)
            self.assertAlmostEqual(L.FACE_Y - abs(y) - L.PATCH_SIZE[1] / 2, 0.005)  # 5-15 mm from the face
            self.assertLessEqual(L.FLOOR_X0 + 0.03, x - L.TRAP_W / 2)  # a set trap stands well inside the floor
            self.assertLessEqual(x + L.TRAP_W / 2, L.FLOOR_X1 - 0.01)
        # Patches on one wall are far apart: no trap can be within 25 mm (along the wall) of two of them, and two traps
        # set square at a wall's two patches leave at least 40 mm of floor between them.
        for a, b in itertools.combinations(PATCHES, 2):
            if L.PATCHES[a][1] == L.PATCHES[b][1]:
                self.assertGreater(abs(L.PATCHES[a][0] - L.PATCHES[b][0]), 2 * L.ALONG_MAX)
                self.assertGreaterEqual(abs(L.PATCHES[a][0] - L.PATCHES[b][0]) - L.TRAP_W, 0.040)
        # The traps' shapes all lie inside the base's outline seen from above, and the bail clears the pedal.
        for c, s, _ in L.TRAP_SHAPES:
            self.assertLessEqual(abs(c[0]) + s[0] / 2, L.TRAP_HALF_LEN + 1e-12)
            self.assertLessEqual(abs(c[1]) + s[1] / 2, L.TRAP_W / 2 + 1e-12)
        self.assertLess(L.BAIL_BAR_CENTER[0] + L.BAIL_BAR_SIZE[0] / 2, L.PEDAL_CENTER[0] - L.PEDAL_SIZE[0] / 2)
        self.assertAlmostEqual(L.TRAP_H, 0.017)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = run([START] * WINDOW)[0]
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["zero"], "no trap set at any patch")

    def test_the_demo_end_scores_one(self):
        r = run([DONE] * WINDOW)[0]
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(all(row["set"] for row in r.details["patches"].values()))

    def test_every_number_of_patches(self):
        """0.25 per patch with a trap set at it, whichever patches: all 16 combinations; the other traps left where
        they started."""
        for k in range(5):
            for chosen in itertools.combinations(range(4), k):
                sc = dict(START)
                for i in chosen:
                    sc[NAMES[i]] = DONE[NAMES[i]]
                self.assertEqual(score_of(sc), 0.25 * k, chosen)

    def test_every_outcome_in_the_text(self):
        # three right and one 12 mm off the wall
        self.assertEqual(score_of(dict(DONE, trap_4=set_trap("patch_right_far", gap=0.012))), 0.75)
        # one wall right, the other wall's traps laid parallel to it at their patches
        par3 = trap(L.PATCHES["patch_right_near"][0], -(L.FACE_Y - 0.004 - L.TRAP_W / 2), 180.0)
        par4 = trap(L.PATCHES["patch_right_far"][0], -(L.FACE_Y - 0.004 - L.TRAP_W / 2), 0.0)
        self.assertEqual(score_of(dict(DONE, trap_3=par3, trap_4=par4)), 0.5)
        # one wall right, the other wall's pedals turned out into the room
        self.assertEqual(score_of(dict(DONE, trap_3=set_trap("patch_right_near", reverse=True),
                                       trap_4=set_trap("patch_right_far", reverse=True))), 0.5)
        # two traps at one patch (side by side) and the other two left in the middle
        two = scene(trap_1=set_trap("patch_left_near", along=-0.023), trap_2=set_trap("patch_left_near", along=0.023))
        self.assertEqual(score_of(two), 0.25)
        # one patch only
        self.assertEqual(score_of(scene(trap_2=set_trap("patch_left_far"))), 0.25)
        # none
        self.assertEqual(score_of(START), 0.0)

    def test_traps_are_interchangeable(self):
        for perm in itertools.permutations(PATCHES):
            sc = {n: set_trap(p) for n, p in zip(NAMES, perm)}
            self.assertEqual(score_of(sc), 1.0, perm)

    def test_square_threshold(self):
        for turn in (14.9, -14.9):
            self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", turn=turn))), 1.0, turn)
        for turn in (15.1, -15.1, 30.0, 90.0):
            self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", turn=turn))), 0.75, turn)

    def test_gap_threshold(self):
        for gap in (0.0, 0.0079, -0.0029):  # touching, nearly 8 mm, pressed into contact
            self.assertEqual(score_of(dict(DONE, trap_2=set_trap("patch_left_far", gap=gap))), 1.0, gap)
        for gap in (0.0081, 0.020, -0.0031):
            self.assertEqual(score_of(dict(DONE, trap_2=set_trap("patch_left_far", gap=gap))), 0.75, gap)

    def test_a_turned_trap_is_measured_at_its_nearest_corner(self):
        """Turned 14 degrees, its nearest corner 7 mm off: set (its end's midpoint is 12.4 mm off). Turned 14
        degrees with the end's midpoint 7 mm off: the corner is 1.6 mm off, also set. Turned with the nearest corner
        9 mm off: not set."""
        s = set_trap("patch_right_near", gap=0.007, turn=14.0)
        self.assertGreater(L.FACE_Y - abs(float(pedal_end_mid(s)[1])), 0.012)
        self.assertEqual(score_of(dict(DONE, trap_3=s)), 1.0)
        self.assertEqual(score_of(dict(DONE, trap_3=set_trap("patch_right_near", gap=0.009, turn=14.0))), 0.75)

    def test_along_threshold(self):
        for along in (0.0249, -0.0249):
            self.assertEqual(score_of(dict(DONE, trap_3=set_trap("patch_right_near", along=along))), 1.0, along)
        for along in (0.0251, -0.0251, 0.060):
            self.assertEqual(score_of(dict(DONE, trap_3=set_trap("patch_right_near", along=along))), 0.75, along)

    def test_tilt_and_floor_thresholds(self):
        ok = set_trap("patch_left_near", tilt=9.9, tilt_axis="y")  # its pedal end lifted
        self.assertEqual(score_of(dict(DONE, trap_1=ok)), 1.0)
        bad = set_trap("patch_left_near", tilt=10.1, tilt_axis="y")
        self.assertEqual(score_of(dict(DONE, trap_1=bad)), 0.75)
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.0029))), 1.0)
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.0031))), 0.75)
        # resting on the 0.5 mm droppings patch at its pedal end: still on the floor
        rest = set_trap("patch_left_near", tilt=-0.33, tilt_axis="y", low=L.FLOOR_TOP)
        self.assertEqual(score_of(dict(DONE, trap_1=rest)), 1.0)


class ZeroConditions(unittest.TestCase):
    def test_upside_down_at_the_patch(self):
        s = set_trap("patch_left_near", tilt=180.0)  # flipped about its length: pedal end still toward the wall
        self.assertEqual(score_of(dict(START, trap_1=s)), 0.0)

    def test_on_its_side_or_on_an_end(self):
        side = set_trap("patch_left_near", tilt=90.0)
        self.assertEqual(score_of(dict(START, trap_1=side)), 0.0)
        end = trap(0.340, L.FACE_Y - 0.004 - L.TRAP_W / 2, 0.0, tilt=90.0, tilt_axis="y")
        self.assertEqual(score_of(dict(START, trap_1=end)), 0.0)

    def test_held_up(self):
        self.assertEqual(score_of(scene(trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.004))), 0.0)
        self.assertEqual(score_of(scene(trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.030))), 0.0)

    def test_sunk_into_the_floor(self):
        """A trap reported more than 3 mm below the floor's top (a simulator blow-up, or fallen past the floor's
        edge at the patch) is not lying on the floor."""
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP - 0.0029))), 1.0)
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP - 0.0031))), 0.75)
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.TOP - 0.1))), 0.75)

    def test_on_another_trap(self):
        under = set_trap("patch_left_near")
        over = set_trap("patch_left_near", low=L.FLOOR_TOP + L.TRAP_H)
        self.assertEqual(score_of(scene(trap_1=over, trap_2=start_trap("trap_2"))), 0.0)
        self.assertEqual(score_of(scene(trap_1=under, trap_2=over)), 0.25)  # the one beneath still counts

    def test_moving(self):
        a = set_trap("patch_left_near")
        frames = [dict(DONE, trap_1=shifted(a, dx=0.001 * i)) for i in range(WINDOW + 5)]  # 20 mm/s
        r = run(frames)[0]
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["traps"]["trap_1"]["at_rest"])

    def test_a_missing_or_broken_trap_scores_zero(self):
        missing = {n: DONE[n] for n in NAMES[:3]}
        r = run([missing] * WINDOW)[0]
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_pedal_out_or_parallel_at_every_patch(self):
        out = {n: set_trap(p, reverse=True, gap=0.004) for n, p in zip(NAMES, PATCHES)}
        self.assertEqual(score_of(out), 0.0)
        par = {}
        for n, p in zip(NAMES, PATCHES):
            px, wall = L.PATCHES[p]
            par[n] = trap(px, L.WALLS[wall] * (L.FACE_Y - 0.002 - L.TRAP_W / 2), 0.0)
        self.assertEqual(score_of(par), 0.0)


class Stillness(unittest.TestCase):
    def test_needs_half_a_second(self):
        self.assertEqual(run([DONE] * (WINDOW - 1))[0].score, 0.0)
        self.assertEqual(run([DONE] * WINDOW)[0].score, 1.0)

    def test_small_jitter_is_still(self):
        frames = [dict(DONE, trap_1=shifted(DONE["trap_1"], dx=0.0009 * (-1) ** i)) for i in range(WINDOW * 3)]
        self.assertEqual(run(frames)[0].score, 1.0)

    def test_slow_creep_is_not(self):
        frames = [dict(DONE, trap_2=shifted(DONE["trap_2"], dy=-0.00025 * i)) for i in range(WINDOW * 2)]
        r = run(frames)[0]
        self.assertFalse(r.details["traps"]["trap_2"]["at_rest"])
        self.assertEqual(r.score, 0.75)

    def test_a_turn_on_the_spot_is_motion(self):
        """Turning 3 degrees about its own origin moves its pedal end 2.6 mm: not still."""
        x, y = DONE["trap_1"].origin[:2]
        frames = [DONE] * WINDOW
        for i in range(1, 4):
            frames.append(dict(DONE, trap_1=trap(x, y, -90.0 + i)))
        r = run(frames)[0]
        self.assertFalse(r.details["traps"]["trap_1"]["at_rest"])
        self.assertEqual(r.score, 0.75)


class BrokenStatesAndResets(unittest.TestCase):
    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        r, t = run([DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)
        hist = {n: [p.copy() for p in t.hist[n]] for n in NAMES}
        for bad_value in (("origin", np.array([np.nan, 0.0, 0.76])), ("quat_xyzw", np.zeros(4)),
                          ("pos", np.array([np.inf, 0, 0])), ("origin", None)):
            bad = SimpleNamespace(**vars(DONE["trap_2"]))
            setattr(bad, *bad_value)
            r = t.update(dict(DONE, trap_2=bad))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.update({}).score, 0.0)
        self.assertEqual(t.bad_steps, 5)
        for n in NAMES:
            self.assertEqual(len(hist[n]), len(t.hist[n]))
            self.assertTrue(all(np.allclose(a, b) for a, b in zip(hist[n], t.hist[n])))
        self.assertEqual(t.update(DONE).score, 1.0)  # the record carried on: still 1.0 at once

    def test_teleport_back_to_start_resets(self):
        r, t = run([DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)
        steps = t.steps
        t.update(START)
        self.assertEqual(t.steps, 1)
        self.assertLess(t.steps, steps)
        self.assertEqual(t.update(DONE).score, 0.0)  # a fresh window: not still yet

    def test_jittered_reset_counts(self):
        r, t = run([DONE] * WINDOW)
        jit = {n: shifted(START[n], dx=0.02, dy=-0.025) for n in NAMES}
        t.update(jit)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        frames = [DONE] * WINDOW + glide(DONE, START, 40)
        r, t = run(frames)
        self.assertEqual(t.steps, WINDOW + 40)
        self.assertEqual(r.score, 0.0)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        r, t = run([DONE] * WINDOW)
        moved = {n: shifted(DONE[n], dx=-0.05) for n in NAMES}
        t.update(moved)
        self.assertEqual(t.steps, WINDOW + 1)

    def test_explicit_reset(self):
        r, t = run([DONE] * WINDOW)
        t.reset()
        self.assertEqual(t.steps, 0)
        self.assertEqual(t.update(DONE).score, 0.0)


class Exploits(unittest.TestCase):
    def test_pushing_traps_flat_against_the_wall_lengthwise(self):
        """The common mistake: every trap pushed against the skirting along it, at its patch, pedal toward the
        patch: 0."""
        sc = {}
        for n, p in zip(NAMES, PATCHES):
            px, wall = L.PATCHES[p]
            k = L.WALLS[wall]
            sc[n] = trap(px + L.TRAP_HALF_LEN - 0.010, k * (L.FACE_Y - L.TRAP_W / 2), 0.0)  # pedal end on the patch
        self.assertEqual(score_of(sc), 0.0)

    def test_plain_end_at_the_wall(self):
        """Square and touching the wall, but the plain end toward it (pedal out in the room): 0."""
        sc = {n: set_trap(p, reverse=True, gap=0.0) for n, p in zip(NAMES, PATCHES)}
        self.assertEqual(score_of(sc), 0.0)

    def test_standing_a_trap_on_its_pedal_end_against_the_wall(self):
        px, _ = L.PATCHES["patch_left_near"]
        # pitched 90 degrees about y with the plain end up: the pedal end on the floor at the wall
        s = trap(px, L.FACE_Y - 0.002 - L.TRAP_BASE_H, -90.0, tilt=-90.0, tilt_axis="y")
        self.assertEqual(score_of(dict(START, trap_1=s)), 0.0)

    def test_leaning_against_the_skirting(self):
        s = set_trap("patch_left_near", tilt=25.0, tilt_axis="y", low=L.FLOOR_TOP)
        self.assertEqual(score_of(dict(START, trap_1=s)), 0.0)

    def test_two_traps_stacked_at_one_patch_count_once(self):
        under = set_trap("patch_left_far")
        over = set_trap("patch_left_far", low=L.FLOOR_TOP + L.TRAP_H)
        self.assertEqual(score_of(scene(trap_1=under, trap_2=over)), 0.25)

    def test_a_trap_off_the_table_changes_nothing(self):
        gone = trap(0.30, 0.13, 0.0, low=0.30)
        self.assertEqual(score_of(dict(DONE, trap_4=gone)), 0.75)

    def test_held_still_just_above(self):
        """The scorer cannot see the grippers: a trap held still within 3 mm of the floor counts as lying there
        (fidelity says so); 3.5 mm up it does not."""
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.002))), 1.0)
        self.assertEqual(score_of(dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.0035))), 0.75)

    def test_pedal_end_at_the_wrong_wall(self):
        """A trap square to the left wall at the left patch's x, but its pedal end toward the right wall: 0 there,
        and far from every right patch."""
        s = set_trap("patch_left_near", reverse=True, gap=0.0)
        self.assertEqual(score_of(dict(START, trap_1=s)), 0.0)


class OneArm(unittest.TestCase):
    def test_each_arm_alone_with_the_demo_moves(self):
        left = scene(trap_1=DONE["trap_1"], trap_2=DONE["trap_2"])
        right = scene(trap_3=DONE["trap_3"], trap_4=DONE["trap_4"])
        self.assertEqual(score_of(left), 0.5)
        self.assertEqual(score_of(right), 0.5)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir() and (d / "yamkin.py").is_file():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline tools (tools/offline: yam_stub, yamkin)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        d = _offline_dir()
        sys.path.insert(0, str(d / "yam_stub"))
        sys.path.insert(0, str(d))
        cls.task = importlib.import_module("task")

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, ["passage_floor", "skirting_left", "skirting_right", *PATCHES, *NAMES])
        self.assertEqual(t.SCORE_OVER, "end")
        for o in (t.FLOOR, *t.SKIRTINGS, *t.PATCHES):
            self.assertIsNone(o.mass)  # fixed
        self.assertAlmostEqual(t.FLOOR.pos[2] + t.FLOOR.size[2] / 2, L.FLOOR_TOP)
        for o in t.SKIRTINGS:
            self.assertAlmostEqual(abs(o.pos[1]) - o.size[1] / 2, L.FACE_Y)  # the face at y +-0.250
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)
        for o in t.PATCHES:
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.FLOOR_TOP)  # lying on the floor
        for o in t.TRAPS:
            self.assertEqual(o.mass, L.TRAP_MASS)
            self.assertLessEqual(o.mass, 0.3)
            x, y, yaw = L.TRAP_START[o.name]
            self.assertTrue(np.allclose(o.pos, (x, y, L.FLOOR_TOP)))  # the frame's origin, on the floor
            self.assertAlmostEqual(o.yaw, yaw)
            self.assertEqual(len(o.shapes), len(L.TRAP_SHAPES))
            st = o.rest_state()
            self.assertTrue(valid(st))
            self.assertTrue(np.allclose(st.origin, START[o.name].origin))
            self.assertTrue(np.allclose(st.low, START[o.name].low) and np.allclose(st.high, START[o.name].high))
            # PhysX spreads the mass over the colliders: the centre of mass is within 2 mm of the grip (mid-length).
            self.assertLess(abs(o.center_of_mass()[0]), 0.002)
        for o in t.OBJECTS:
            # Nothing in the strip the arms' start pose hides in the setup screenshot (|y| > 0.45 m with x < 0.2 m).
            self.assertLess(abs(o.pos[1]) + 0.06, 0.45, o.name)

    def test_rest_state_scores_zero(self):
        states = {o.name: o.rest_state() for o in self.task.TRAPS}
        self.assertEqual(score_of(states), 0.0)
        self.assertTrue(Tracker.at_start(states))

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(START))
                self.assertEqual(self.task.score(START), 0.0)
                self.task._TRACKER.reset()
                ok = False
                for _ in range(WINDOW):
                    ok = self.task.success(DONE)
                self.assertTrue(ok)
                self.assertEqual(self.task.score(DONE), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1 + WINDOW)
        last = rows[-1]
        for key in ("step", "score", "success", "valid_state", "patches_set", "zero", "patches", "traps", "raw",
                    "mechkit_episode"):
            self.assertIn(key, last)
        row = last["traps"]["trap_1"]
        for key in ("set_at", "placed", "at_rest", "base_down", "tilt_deg", "on_floor", "low_mm",
                    "pedal_heading_deg", "faces", "square_err_deg", "end_gap_mm", "nearest_patch", "along_mm",
                    "origin_mm"):
            self.assertIn(key, row)
        self.assertEqual(row["set_at"], "patch_left_near")
        self.assertEqual(row["end_gap_mm"], 4.0)
        self.assertEqual(row["along_mm"], 0.0)
        self.assertEqual(row["pedal_heading_deg"], 90.0)
        self.assertEqual(last["traps"]["trap_3"]["faces"], "right")
        self.assertEqual(last["patches"]["patch_right_far"]["traps"], ["trap_4"])

    def test_score_hook_reads_the_last_step_not_the_best(self):
        self.task._TRACKER.reset()
        for _ in range(WINDOW):
            self.task.success(DONE)
        self.assertEqual(self.task.score(DONE), 1.0)
        lifted = dict(DONE, trap_1=set_trap("patch_left_near", low=L.FLOOR_TOP + 0.030))
        self.assertFalse(self.task.success(lifted))
        self.assertEqual(self.task.score(lifted), 0.75)

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(START))
        finally:
            os.environ.pop("TASK_TRACE", None)
        self.assertFalse((HERE / "no_such_dir").exists())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                bad = SimpleNamespace(**vars(START["trap_2"]))
                bad.origin = np.array([float("nan"), 0.0, 0.76])
                self.assertFalse(self.task.success(dict(START, trap_2=bad)))
                self.assertFalse(self.task.success({}))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertFalse(rows[0]["valid_state"])
        self.assertFalse(rows[1]["valid_state"])
        self.assertEqual(rows[1]["raw"]["trap_1"], "unreadable")

    def test_a_new_mechkit_episode_resets_the_tracker(self):
        import mechkit as mk
        t = self.task
        t._TRACKER.reset()
        for _ in range(WINDOW):
            t.success(DONE)
        self.assertGreater(t._TRACKER.steps, 1)
        t._EPISODE["mechkit"] = mk.episode() - 1  # as if mechkit had seen a reset since the last step
        t.success(DONE)
        self.assertEqual(t._TRACKER.steps, 1)

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_places_score_one(self):
        """Each trap carried from its start, turned by the plan's quarter turn and set where the plan sets it down
        scores 1.0 for the four; with one arm's moves only, 0.5."""
        t = self.task
        sc = dict(START)
        for side, pairs in t.PLAN.items():
            wps = t.demo()[side]
            for trap_name, patch in pairs:
                x0, y0, yaw0 = L.TRAP_START[trap_name]
                dx, dy = t.place_point(trap_name, patch)
                turn = t.turn(trap_name, patch)
                self.assertAlmostEqual(abs(turn), math.pi / 2)
                self.assertIn((dx, dy), [w.pos[:2] for w in wps])
                self.assertIn(turn, [w.yaw for w in wps if w.pos[:2] == (dx, dy)])
                sc[trap_name] = trap(dx, dy, math.degrees(yaw0 + turn))
        r = run([sc] * WINDOW)[0]
        self.assertEqual(r.score, 1.0)
        for n, row in r.details["traps"].items():
            self.assertEqual(row["end_gap_mm"], 4.0, n)
            self.assertEqual(row["along_mm"], 0.0, n)

    def test_each_arm_turns_its_traps_opposite_ways(self):
        """The two traps on a side start with their pedals pointing opposite ways, so they need opposite turns; each
        arm's turns are the ones its wrist reaches with room (the left arm's +90 and the right arm's -90 only at the
        far patches, about 16 degrees from joint 6's limit)."""
        t = self.task
        self.assertEqual([round(t.turn(n, p), 6) for n, p in t.PLAN["left"]], [round(-math.pi / 2, 6),
                                                                                round(math.pi / 2, 6)])
        self.assertEqual([round(t.turn(n, p), 6) for n, p in t.PLAN["right"]], [round(math.pi / 2, 6),
                                                                                 round(-math.pi / 2, 6)])

    def test_both_arms_stay_on_their_own_side(self):
        """The left gripper never comes within 0.12 m of the midline, nor the right; the two run on one clock."""
        plan = self.task.demo()
        self.assertGreaterEqual(min(w.pos[1] for w in plan["left"]), 0.12)
        self.assertLessEqual(max(w.pos[1] for w in plan["right"]), -0.12)
        self.assertEqual([w.duration for w in plan["left"]], [w.duration for w in plan["right"]])
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * HZ, 1200)  # inside the run limit with the policy's 3 s hold

    def test_grips_and_clearances(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.GRIP, L.TRAP_W - 0.008)  # commanded 8 mm narrower
        self.assertGreaterEqual(-2 * t.OPEN - L.TRAP_W, 0.016)  # at least 8 mm each side when open
        self.assertLessEqual(-2 * t.OPEN, 0.094)
        tips = t.GRIP_Z - t.PAD - L.FLOOR_TOP
        self.assertGreaterEqual(tips, 0.003 - 1e-9)  # pad tips above the floor (2.5 mm over a patch)
        self.assertGreaterEqual(tips - L.PATCH_SIZE[2], 0.0025 - 1e-9)
        self.assertLessEqual(L.TRAP_BASE_H - tips, L.TRAP_BASE_H)  # the pads hold the base's sides ...
        self.assertGreaterEqual(L.TRAP_BASE_H - tips, 0.007 - 1e-9)  # ... over 7 mm of them
        # The pads close on the base's sides, clear of the steel parts on top (narrower than the base).
        self.assertGreater(L.TRAP_W / 2 - L.SPRING_SIZE[1] / 2, 0.004)
        self.assertGreater(L.TRAP_W / 2 - (L.BAIL_SIDE_CENTERS[0][1] + L.BAIL_SIDE_SIZE[1] / 2), 0.002)
        # A carried trap's underside passes well over the other traps and the skirting's top.
        under = t.HOVER - (t.GRIP_Z - L.FLOOR_TOP)
        self.assertGreater(under - (L.FLOOR_TOP + L.TRAP_H), 0.04)
        self.assertGreaterEqual(under - (L.TOP + L.SKIRT_H), 0.005)  # 5 mm or more over the skirting's top
        # Set traps clear the traps still waiting (their outlines seen from above do not meet).
        for side, pairs in t.PLAN.items():
            (first, p1), (second, _) = pairs
            dx, dy = t.place_point(first, p1)
            sx, sy, _ = L.TRAP_START[second]
            gap_x = (sx - L.TRAP_HALF_LEN) - (dx + L.TRAP_W / 2)
            self.assertGreater(gap_x, 0.005, side)

    def test_carried_trap_never_reaches_the_skirting(self):
        """Turning while it is carried, the trap sweeps round the tool point; sampled along the carry (straight line,
        one-axis turn on the same clock), no corner comes within 3 mm of a skirting's face."""
        t = self.task
        for side, pairs in t.PLAN.items():
            k = t.SIDE[side]
            for trap_name, patch in pairs:
                x0, y0, yaw0 = L.TRAP_START[trap_name]
                dx, dy = t.place_point(trap_name, patch)
                turn = t.turn(trap_name, patch)
                for s in np.linspace(0.0, 1.0, 101):
                    c = trap((1 - s) * x0 + s * dx, (1 - s) * y0 + s * dy, math.degrees(yaw0 + s * turn))
                    self.assertGreater(L.FACE_Y - float(np.max(k * corners(c)[:, 1])), 0.003, (trap_name, s))

    def test_slow_final_descents_and_long_opens(self):
        for side, wps in self.task.demo().items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly"):
                    self.assertGreaterEqual(w.duration, 1.5, (side, i))
                    self.assertLessEqual(abs(wps[i - 1].pos[2] - w.pos[2]), 0.030 + 1e-9)
                if w.label.startswith(("grip", "let go")):
                    self.assertGreaterEqual(w.duration, 1.5, (side, i))
                    self.assertEqual(wps[i - 1].pos, w.pos)  # opened and closed in place

    def test_success_hold_outlasts_every_release(self):
        """A trap counts as set while still gripped (within 3 mm), so the success hold must be longer than the settle
        and the release after each set-down plus the stillness window: the trial cannot end with the fingers closed
        on a trap."""
        n = 0
        for side, wps in self.task.demo().items():
            for i, w in enumerate(wps):
                if w.label == "slowly set the trap down":
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertEqual((settle.label, let_go.label), ("settle", "let go of the trap"))
                    self.assertLessEqual(settle.duration + let_go.duration + L.REST_WINDOW_S,
                                         self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 4)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_task_json_numbers_match_the_layout(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["setup"] + " " + spec["scoring"]
        for needle in ("98 mm", "45 mm", "8 mm", "15 degrees", "25 mm", "10 degrees", "3 mm", "0.5 s", "2 mm",
                       "x 34.0 cm", "x 43.0 cm", "x 33.0 cm", "x 44.0 cm", "y +25.0", "25 g"):
            self.assertIn(needle, text, needle)
        self.assertIn("1,200", spec["run_limit"])

    def test_cross_reach(self):
        """Straight down at the height that grips a trap, neither arm reaches any corner of a trap set at the other
        wall's patches, in any pose the rule credits (turned up to 15 degrees, up to 8 mm off the wall, up to 25 mm
        along it): the README's 'One arm alone'."""
        sys.path.insert(0, str(_offline_dir()))
        import yamkin as K
        top, base = K.TABLE_TOP["station"], K.BASES["station"]["left"]
        z = top + 0.018
        worst = -1.0
        for patch in ("patch_right_near", "patch_right_far"):
            for turn in (-15.0, 0.0, 15.0):
                for along in (-0.025, 0.0, 0.025):
                    s = set_trap(patch, gap=L.END_GAP_MAX, along=along, turn=turn)
                    c = corners(s)
                    p = c[np.argmax(c[:, 1])]  # the corner nearest the left arm
                    ok, q, _, _ = K.reachable((p[0], p[1], z), 0.0, base)
                    if ok:
                        ok = K.pose_clear(q, base, top)["ok"]
                    self.assertFalse(ok, (patch, turn, along, p))
                    worst = max(worst, float(p[1]))
        self.assertLess(worst, -0.13)  # the nearest any such corner comes to the midline


if __name__ == "__main__":
    unittest.main()
