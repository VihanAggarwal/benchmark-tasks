"""Offline tests for scoring.py: synthetic object states, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
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
    POSTS,
    Tracker,
    beam_distance,
    facing_deg,
    guard_gap,
    over_floor,
    pose_check,
    tilt_deg,
    too_close,
    valid,
)

HZ = 20.0
EM, RX = POSTS  # emitter (starts on the left), receiver (starts on the right)
TX, TY = L.TARGET_X, L.TARGET_Y
HALF_D = L.POST_HALF[0]  # 18 mm: half the depth, which runs along y once the post is turned


def quat(yaw_deg=0.0, tilt_deg_=0.0, tilt_axis="x"):
    """Turned yaw about the vertical after tilting tilt about the body's x (or y) axis (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg_) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qt = np.array([math.sin(b), 0.0, 0.0, math.cos(b)]) if tilt_axis == "x" else \
        np.array([0.0, math.sin(b), 0.0, math.cos(b)])
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qt
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def post(x, y, low_z=L.FLOOR_TOP, yaw=None, tilt=0.0, tilt_axis="x"):
    """A post's state as the simulator reports it: centre (x, y), lowest point at low_z (AABB of the turned box).
    yaw defaults to the placed quarter turn (-90 on the left, +90 on the right: broad faces across the opening)."""
    if yaw is None:
        yaw = -90.0 if y >= 0 else 90.0
    q = quat(yaw, tilt, tilt_axis)
    half = np.array(L.POST_HALF)
    corners = np.array([rot(q) @ (half * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c = np.array([x, y, low_z - corners[:, 2].min()])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=c + corners.min(axis=0), high=c + corners.max(axis=0))


def at_d(d_mm, side="left", gap_mm=5.0, **kw):
    """A post standing turned against a guard with its beam d_mm from the die line and gap_mm to the guard."""
    sgn = 1.0 if side == "left" else -1.0
    return post(L.DIE_X - d_mm / 1000, sgn * (L.GUARD_INNER_Y - gap_mm / 1000 - HALF_D), **kw)


MAT_TOP = L.TOP + L.MAT_SIZE[2]
START_L = post(*L.POST_START["left"], low_z=MAT_TOP, yaw=0.0)
START_R = post(*L.POST_START["right"], low_z=MAT_TOP, yaw=0.0)
PLACED_L = post(TX, TY)
PLACED_R = post(TX, -TY)
# unturned (broad faces toward x, the 56 mm width along y), outer face 5 mm from the guard
UNTURNED_L = at_d(178, "left", gap_mm=5.0 + (L.POST_HALF[1] - HALF_D) * 1000, yaw=0.0)
UNTURNED_R = at_d(178, "right", gap_mm=5.0 + (L.POST_HALF[1] - HALF_D) * 1000, yaw=0.0)


def pair(a, b):
    return {EM: a, RX: b}


def run(seq, tr=None):
    """Feed (emitter, receiver) pairs to a tracker (a fresh one unless given); return the last result and it."""
    tr = Tracker(HZ) if tr is None else tr
    r = None
    for a, b in seq:
        r = tr.update(pair(a, b))
    return r, tr


def settled(a, b, n=12, tr=None):
    """The pair held still for n steps (0.55 s at 20 Hz: enough history to be at rest)."""
    return run([(a, b)] * n, tr)


def glide(a_from, a_to, n):
    """n post states moving from one pose to another (positions only; yaw kept)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        out.append(SimpleNamespace(pos=(1 - f) * a_from.pos + f * a_to.pos, quat_xyzw=a_to.quat_xyzw,
                                   lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                   low=(1 - f) * a_from.low + f * a_to.low, high=(1 - f) * a_from.high + f * a_to.high))
    return out


def score_of(a, b):
    return settled(a, b)[0].score


class Geometry(unittest.TestCase):
    def test_safety_distance_is_168_mm(self):
        self.assertAlmostEqual(L.S_MM, 168.0)
        self.assertAlmostEqual(L.S_MIN, 0.168)
        self.assertAlmostEqual(L.S_MAX, 0.188)
        self.assertAlmostEqual(L.TARGET_X, 0.262)
        self.assertAlmostEqual(L.TARGET_Y, 0.117)

    def test_start_layout(self):
        for s in (START_L, START_R):
            self.assertAlmostEqual(tilt_deg(s), 0.0)
            self.assertFalse(over_floor(s))
            self.assertFalse(too_close(s))
            self.assertFalse(pose_check(s)["pose_ok"])
        self.assertTrue(Tracker.at_start(EM, START_L))
        self.assertTrue(Tracker.at_start(RX, START_R))
        self.assertFalse(Tracker.at_start(EM, START_R))

    def test_placed_measures(self):
        f = pose_check(PLACED_L)
        self.assertTrue(f["pose_ok"])
        self.assertAlmostEqual(f["d_mm"], 178.0)
        self.assertAlmostEqual(f["gap_mm"], 5.0)
        self.assertAlmostEqual(f["bottom_mm"], 0.0)
        self.assertEqual(f["side"], "left")
        self.assertEqual(pose_check(PLACED_R)["side"], "right")
        self.assertAlmostEqual(pose_check(PLACED_R)["gap_mm"], 5.0)

    def test_gap_uses_the_real_outline(self):
        # unturned, the 56 mm width runs along y: the same centre leaves a 10 mm smaller gap
        self.assertAlmostEqual(guard_gap(post(TX, TY, yaw=0.0)) * 1000, 5.0 - 10.0)
        # turned 45 degrees the corner reaches out farther
        self.assertLess(guard_gap(post(TX, TY, yaw=45.0)), guard_gap(PLACED_L))
        # a tilt toward the guard swings the top out (turned -90 deg, the body's y axis runs along x)
        self.assertLess(guard_gap(post(TX, TY, tilt=4.0, tilt_axis="y")), guard_gap(PLACED_L))
        self.assertAlmostEqual(guard_gap(UNTURNED_L), 0.005)

    def test_beam_distance(self):
        self.assertAlmostEqual(beam_distance(at_d(168)) * 1000, 168.0)
        self.assertAlmostEqual(beam_distance(at_d(-10)) * 1000, -10.0)

    def test_tilt_either_end_down(self):
        self.assertAlmostEqual(tilt_deg(post(TX, TY, tilt=7.0)), 7.0, places=6)
        self.assertAlmostEqual(tilt_deg(post(TX, TY, tilt=180.0)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(post(TX, TY, tilt=90.0)), 90.0, places=6)

    def test_facing_angle(self):
        self.assertAlmostEqual(facing_deg(PLACED_L, PLACED_R), 0.0, places=6)
        self.assertAlmostEqual(facing_deg(post(TX, TY, yaw=-87.0), PLACED_R), 3.0, places=6)
        self.assertAlmostEqual(facing_deg(post(TX, TY, yaw=90.0), PLACED_R), 0.0, places=6)  # the other broad face
        self.assertAlmostEqual(facing_deg(post(TX, TY, yaw=0.0), PLACED_R), 90.0, places=6)
        # offset along x: the line between the centres turns
        a, b = at_d(168), at_d(188, "right")
        self.assertAlmostEqual(facing_deg(a, b), math.degrees(math.atan2(0.020, 2 * TY)), places=6)

    def test_too_close_region(self):
        self.assertFalse(too_close(at_d(168)))
        self.assertTrue(too_close(at_d(167.9)))
        self.assertTrue(too_close(post(0.30, 0.0)))  # middle of the opening
        self.assertTrue(too_close(post(0.48, 0.0, low_z=L.TOP + L.PRESS_H)))  # on the press
        self.assertTrue(too_close(post(0.30, 0.149, low_z=L.TOP + L.GUARD_H)))  # on a guard's top
        self.assertFalse(too_close(post(0.30, 0.151 + HALF_D, low_z=L.TOP)))  # outside the guard, on the table


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = settled(START_L, START_R)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_both_placed_facing_scores_one(self):
        r, _ = settled(PLACED_L, PLACED_R)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["placed_sides"], ["left", "right"])

    def test_every_tier_is_reachable(self):
        self.assertEqual(score_of(START_L, START_R), 0.0)
        self.assertEqual(score_of(PLACED_L, START_R), 0.25)  # one post placed
        self.assertEqual(score_of(START_L, PLACED_R), 0.25)
        self.assertEqual(score_of(UNTURNED_L, UNTURNED_R), 0.5)  # both placed, neither turned to face
        self.assertEqual(score_of(PLACED_L, UNTURNED_R), 0.75)  # both placed, one facing
        self.assertEqual(score_of(UNTURNED_L, PLACED_R), 0.75)
        self.assertEqual(score_of(PLACED_L, PLACED_R), 1.0)

    def test_facing_needs_both_placed(self):
        # one post placed and facing, the other standing in the opening but 20 mm from its guard
        self.assertEqual(score_of(PLACED_L, at_d(178, "right", gap_mm=20.0)), 0.25)

    def test_only_full_score_is_success(self):
        for a, b in ((PLACED_L, START_R), (UNTURNED_L, UNTURNED_R), (PLACED_L, UNTURNED_R)):
            self.assertFalse(settled(a, b)[0].success)

    def test_window_edges(self):
        self.assertEqual(score_of(at_d(168), at_d(168, "right")), 1.0)
        self.assertEqual(score_of(at_d(188), at_d(188, "right")), 1.0)
        self.assertEqual(score_of(at_d(168), at_d(188, "right")), 1.0)  # 4.9 deg between the lens axes and beam
        self.assertEqual(score_of(at_d(188.5), PLACED_R), 0.25)  # too far: that post does not count
        self.assertEqual(score_of(at_d(200), at_d(200, "right")), 0.0)

    def test_gap_edges(self):
        self.assertEqual(score_of(at_d(178, gap_mm=10.0), PLACED_R), 1.0)
        self.assertEqual(score_of(at_d(178, gap_mm=10.5), PLACED_R), 0.25)
        self.assertEqual(score_of(at_d(178, gap_mm=0.0), PLACED_R), 1.0)
        self.assertEqual(score_of(at_d(178, gap_mm=-2.0), PLACED_R), 1.0)  # contact penetration
        self.assertEqual(score_of(at_d(178, gap_mm=-3.0), PLACED_R), 0.25)

    def test_tilt_edges(self):
        self.assertEqual(score_of(post(TX, TY, tilt=4.9, tilt_axis="y"), PLACED_R), 1.0)
        self.assertEqual(score_of(post(TX, TY, tilt=5.5, tilt_axis="y"), PLACED_R), 0.25)

    def test_floor_edges(self):
        self.assertEqual(score_of(post(TX, TY, low_z=L.FLOOR_TOP + 0.0029), PLACED_R), 1.0)
        self.assertEqual(score_of(post(TX, TY, low_z=L.FLOOR_TOP + 0.0035), PLACED_R), 0.25)  # held in the air

    def test_facing_edges(self):
        self.assertEqual(score_of(post(TX, TY, yaw=-85.1), PLACED_R), 1.0)
        self.assertEqual(score_of(post(TX, TY, yaw=-84.5), PLACED_R), 0.75)
        self.assertEqual(score_of(post(TX, TY, yaw=-95.5), PLACED_R), 0.75)
        self.assertEqual(score_of(post(TX, TY, yaw=-84.5), post(TX, -TY, yaw=95.5)), 0.5)

    def test_score_is_idempotent(self):
        r, tr = settled(PLACED_L, PLACED_R)
        for _ in range(3):
            self.assertEqual(tr.result(pair(PLACED_L, PLACED_R)).score, 1.0)
        self.assertEqual(tr.steps, 12)

    def test_posts_may_swap_sides(self):
        # the emitter on the right and the receiver on the left: the posts are identical units here
        self.assertEqual(score_of(PLACED_R, PLACED_L), 1.0)


class ZeroConditions(unittest.TestCase):
    def test_a_post_closer_than_s_zeroes_everything(self):
        r, _ = settled(at_d(167.0), PLACED_R)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "closer_than_S")

    def test_closer_than_s_anywhere_between_the_guards(self):
        for bad in (post(0.30, 0.0), post(0.40, 0.05, yaw=30.0), post(0.43, -0.10, low_z=L.FLOOR_TOP + 0.05),
                    post(0.48, 0.0, low_z=L.TOP + L.PRESS_H), post(0.35, 0.0, low_z=L.FLOOR_TOP, tilt=90.0)):
            self.assertEqual(score_of(PLACED_L, bad), 0.0)
            self.assertEqual(score_of(bad, PLACED_R), 0.0)

    def test_closer_than_s_while_still_moving(self):
        seq = [(PLACED_L, PLACED_R)] * 12 + [(PLACED_L, s) for s in glide(PLACED_R, post(0.30, -TY), 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "closer_than_S")


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        r, _ = settled(PLACED_L, PLACED_R, n=10)
        self.assertEqual(r.score, 0.0)
        r, _ = settled(PLACED_L, PLACED_R, n=11)
        self.assertEqual(r.score, 1.0)

    def test_moving_post_is_not_placed(self):
        seq = [(PLACED_L, PLACED_R)] * 12 + [(s, PLACED_R) for s in glide(post(TX, TY + 0.0, low_z=L.FLOOR_TOP + 0.03),
                                                                     PLACED_L, 6)]
        r, _ = run(seq)
        self.assertEqual(r.score, 0.25)  # the left post landed this step: not yet at rest
        r, _ = settled(PLACED_L, PLACED_R, n=11, tr=_)
        self.assertEqual(r.score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        rng = np.random.default_rng(1)
        seq = []
        for _ in range(20):
            j = rng.uniform(-0.0006, 0.0006, size=2)
            seq.append((post(TX + j[0], TY + j[1]), PLACED_R))
        self.assertEqual(run(seq)[0].score, 1.0)

    def test_a_turn_in_place_counts_as_moving(self):
        seq = [(post(TX, TY, yaw=-90.0 + 0.8 * i), PLACED_R) for i in range(12)]  # 8.8 deg in 0.55 s
        r, _ = run(seq)
        self.assertFalse(r.details[EM]["at_rest"])
        self.assertEqual(r.score, 0.25)


class Exploits(unittest.TestCase):
    def test_post_lying_on_its_side_at_the_place(self):
        lying = post(TX, TY - 0.03, low_z=L.FLOOR_TOP, tilt=90.0)
        self.assertEqual(score_of(lying, PLACED_R), 0.25)

    def test_held_in_the_air_over_the_place(self):
        self.assertEqual(score_of(post(TX, TY, low_z=L.FLOOR_TOP + 0.02), post(TX, -TY, low_z=L.FLOOR_TOP + 0.02)),
                         0.0)

    def test_both_posts_against_the_same_guard(self):
        # each guard counts once (the two would overlap here; the scorer is tested on the poses alone)
        a = at_d(178, "left", gap_mm=5.0)
        self.assertEqual(score_of(a, at_d(170, "left", gap_mm=5.0)), 0.25)
        self.assertEqual(score_of(at_d(178, "right"), at_d(186, "right")), 0.25)

    def test_outside_the_guard_on_the_table(self):
        outside = post(TX, L.GUARD_OUTER_Y + HALF_D + 0.002, low_z=L.TOP)
        self.assertFalse(pose_check(outside)["pose_ok"])
        self.assertEqual(score_of(outside, PLACED_R), 0.25)

    def test_on_top_of_the_guard(self):
        on_top = post(TX, L.GUARD_INNER_Y + 0.005, low_z=L.TOP + L.GUARD_H)
        self.assertEqual(score_of(on_top, PLACED_R), 0.25)

    def test_post_wedged_diagonally(self):
        diag = post(TX, L.GUARD_INNER_Y - 0.034, yaw=-45.0)
        self.assertTrue(pose_check(diag)["pose_ok"])
        self.assertEqual(score_of(diag, PLACED_R), 0.75)  # both placed; its lens is 45 deg off: no facing credit

    def test_stacked_on_the_other_post(self):
        stacked = post(TX, -TY, low_z=L.FLOOR_TOP + L.POST_SIZE[2])
        self.assertEqual(score_of(stacked, PLACED_R), 0.25)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        _, tr = settled(PLACED_L, PLACED_R)
        hist = [dict(h) for h in tr.history]
        bad = []
        for field in ("pos", "quat_xyzw", "low", "high"):
            s = post(TX, TY)
            v = np.array(getattr(s, field), dtype=float)
            v[0] = np.nan
            setattr(s, field, v)
            bad.append(s)
        zq = post(TX, TY)
        zq.quat_xyzw = np.zeros(4)
        inf = post(TX, TY)
        inf.pos = np.array([np.inf, 0.0, 0.0])
        bad += [zq, inf, SimpleNamespace(pos=np.zeros(3))]
        for s in bad:
            r = tr.update(pair(s, PLACED_R))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(len(tr.history), len(hist))
        self.assertEqual(tr.update(pair(PLACED_L, PLACED_R)).score, 1.0)

    def test_missing_post(self):
        tr = Tracker(HZ)
        self.assertEqual(tr.update({EM: PLACED_L}).score, 0.0)
        self.assertEqual(tr.update({EM: PLACED_L, RX: None}).score, 0.0)
        self.assertEqual(tr.update(None).score, 0.0)
        self.assertEqual(len(tr.history), 0)

    def test_unnormalised_quaternion(self):
        a = post(TX, TY)
        a.quat_xyzw = a.quat_xyzw * 3.0
        self.assertEqual(score_of(a, PLACED_R), 1.0)
        self.assertTrue(valid(a))

    def test_random_states_score_a_listed_tier(self):
        rnd = random.Random(7)
        tr = Tracker(HZ)
        for _ in range(400):
            def rnd_post():
                return post(rnd.uniform(0.1, 0.55), rnd.uniform(-0.3, 0.3), low_z=L.TOP + rnd.uniform(0, 0.1),
                            yaw=rnd.uniform(-180, 180), tilt=rnd.choice([0.0, rnd.uniform(0, 90)]),
                            tilt_axis=rnd.choice("xy"))
            r = tr.update(pair(rnd_post(), rnd_post()))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        _, tr = settled(PLACED_L, PLACED_R, n=30)
        tr.update(pair(START_L, START_R))
        self.assertEqual((tr.steps, len(tr.history)), (1, 1))

    def test_one_post_teleported_back_resets(self):
        _, tr = settled(PLACED_L, START_R, n=30)
        tr.update(pair(START_L, START_R))
        self.assertEqual(tr.steps, 1)

    def test_jittered_start_resets(self):
        _, tr = settled(PLACED_L, PLACED_R, n=30)
        tr.update(pair(post(0.18, 0.20, low_z=MAT_TOP, yaw=0.0), post(0.15, -0.23, low_z=MAT_TOP, yaw=0.0)))
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        seq = [(PLACED_L, PLACED_R)] * 12 + list(zip(glide(PLACED_L, START_L, 40), glide(PLACED_R, START_R, 40)))
        _, tr = run(seq)
        self.assertEqual(tr.steps, 52)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, tr = settled(PLACED_L, PLACED_R, n=30)
        tr.update(pair(post(0.30, 0.0), PLACED_R))
        self.assertEqual(tr.steps, 31)

    def test_explicit_reset(self):
        _, tr = settled(PLACED_L, PLACED_R)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history)), (0, 0))
        self.assertIsNone(tr.prev)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.tf = importlib.import_module("yam_arena.task_file")

    def setUp(self):
        self.task._TRACKER.reset()

    def obs(self, a, b):
        return {n: self.tf.ObjectState(s.pos, s.quat_xyzw, s.lin_vel, s.ang_vel, s.low, s.high)
                for n, s in pair(a, b).items()}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], list(POSTS))
        for o in loose:
            self.assertLessEqual(o.mass, 0.3)
            self.assertEqual(tuple(o.size), L.POST_SIZE)
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, MAT_TOP)
        guard = self.task.GUARD_LEFT
        self.assertAlmostEqual(guard.pos[1] - guard.size[1] / 2, L.GUARD_INNER_Y)
        tape = self.task.DIE_LINE
        self.assertAlmostEqual(tape.pos[0] - tape.size[0] / 2, L.DIE_X)

    def test_hooks(self):
        t = self.task
        self.assertFalse(t.success(self.obs(START_L, START_R)))
        self.assertEqual(t.score(self.obs(START_L, START_R)), 0.0)
        t._TRACKER.reset()
        ok = [t.success(self.obs(PLACED_L, PLACED_R)) for _ in range(12)]
        self.assertTrue(ok[-1])
        self.assertFalse(ok[0])
        self.assertEqual(t.score(self.obs(PLACED_L, PLACED_R)), 1.0)
        self.assertEqual(t.score({EM: None}), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                for _ in range(12):
                    self.task.success(self.obs(PLACED_L, PLACED_R))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "zero_reason", "placed_sides", EM, RX, f"{EM}_pos",
                          f"{RX}_quat"):
                    self.assertIn(k, row)
                for k in ("d_mm", "gap_mm", "tilt_deg", "bottom_mm", "at_rest", "moved_mm", "placed", "facing_deg",
                          "side", "too_close", "upright", "on_floor", "at_guard", "in_window"):
                    self.assertIn(k, row[EM])
                self.assertAlmostEqual(row[EM]["d_mm"], 178.0)
                self.assertTrue(row["success"])
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.task.success(self.obs(PLACED_L, PLACED_R))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for arm, sgn in (("left", 1.0), ("right", -1.0)):
            wps = plan[arm]
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:  # every close and open takes at least 1.5 s
                    self.assertGreaterEqual(w.duration, 1.5)
                if prev.pos[2] > w.pos[2] and abs(w.pos[2] - t.PLACE_Z) < 1e-9:  # the last 8 mm over 2 s
                    self.assertGreaterEqual(w.duration, 2.0)
                    self.assertLessEqual(prev.pos[2] - w.pos[2], 0.0081)
            grip = [w for w in wps if w.label.startswith("grip it")][0]
            place = [w for w in wps if w.label == "let go"][0]
            self.assertEqual(grip.yaw, 0.0)  # fingers along y at the pick, across the 56 mm width
            self.assertAlmostEqual(grip.finger, -(0.056 - 0.008) / 2)
            self.assertAlmostEqual(place.yaw, -sgn * math.pi / 2)  # along x at the place: no finger by the guard
            self.assertAlmostEqual(L.DIE_X - place.pos[0], 0.178)  # the middle of the window
            self.assertAlmostEqual(L.GUARD_INNER_Y - (sgn * place.pos[1] + HALF_D), 0.005)  # 5 mm from the guard
            self.assertGreaterEqual(L.GUARD_INNER_Y - (sgn * place.pos[1] + 0.0135), 0.009)  # pads clear of it
            self.assertAlmostEqual(place.pos[2] - t.HOLD_UP - L.FLOOR_TOP, 0.002)  # let go 2 mm up
            # the turned post (half 28 mm along x) stays in front of the guard's end while it comes inside
            via = [w for w in wps if w.label.startswith("round the guard")][0]
            self.assertLess(via.pos[0] + L.POST_HALF[1], L.OPEN_X0 - 0.02)
            self.assertLessEqual(sgn * via.pos[1] + HALF_D, L.GUARD_INNER_Y - 0.010)
            # carried with its bottom 25 mm clear
            self.assertAlmostEqual(t.CARRY_Z - t.HOLD_UP - MAT_TOP, 0.025)
        # the right arm starts later, so the two wrists do not pass the guards' front ends together
        self.assertAlmostEqual(plan["right"][0].duration - plan["left"][0].duration, t.RIGHT_LAG)
        self.assertGreaterEqual(t.RIGHT_LAG, 1.0)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("against each side guard", "facing each other", "die line", "2000 mm/s", "8 x (d - 14 mm)",
                  "T = 0.06 s", "d = 20 mm", "20 mm farther"):
            self.assertIn(s, ins)


if __name__ == "__main__":
    unittest.main()
