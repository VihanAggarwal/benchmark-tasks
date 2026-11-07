"""Offline tests for scoring.py, offline_push.py and task.py: synthetic board states and synthetic arm snapshots fed
through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, axis_err_deg, board_info, evaluate, finger_zone, lowest, outline, rotation, segment_outline_distance,
    span_at_y, tilt_deg, valid, wall_distances, zone_gap, zone_radius,
)

HZ = 20.0
TOP = L.TOP
H = L.BOARD_HALF[2]
W = L.BOARD_SIZE[1]
TIPS = 0.013  # pad tips below the TCP
X_A, X_B, X_C = L.FAR_X - 0.015 - W / 2, L.FAR_X - 0.015 - 1.5 * W, L.FAR_X - 0.015 - 2.5 * W  # far gap 15 mm
L_PARK = np.array([0.22, 0.30, TOP + 0.15])
R_PARK = np.array([0.22, -0.30, TOP + 0.15])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def state(center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like board: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ np.array(L.BOARD_HALF, float)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def board(x, y=0.0, turn=0.0, lift=0.0, tilt_x=0.0, flip=False, along_x=False):
    """A board lying flat on the table centred at (x, y), its length across the bay (along y; ``along_x``: along x),
    turned ``turn`` degrees more about the vertical, raised ``lift``, tilted ``tilt_x`` degrees about its own length
    (resting on one long edge), turned upside down with ``flip``."""
    q = about("z", (0.0 if along_x else 90.0) + turn)
    if flip:
        q = qmul(q, about("x", 180.0))
    if tilt_x:
        q = qmul(q, about("x", tilt_x))
    s = state((x, y, 0.0), q)
    z = TOP + lift - (s.low[2] - s.pos[2])  # its lowest point at TOP + lift
    return state((x, y, z), q)


def floor(xa=X_A, xb=X_B, xc=X_C, ya=0.0, yb=0.0, yc=0.0, **kw):
    return {"board_a": board(xa, ya, **kw), "board_b": board(xb, yb, **kw), "board_c": board(xc, yc, **kw)}


def start():
    return {n: board(x, y, along_x=True) for n, (x, y) in L.BOARD_START.items()}


def snap(left_tcp=L_PARK, left_open=0.0, right_tcp=R_PARK, right_open=0.0):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def score_of(boards, arms="park"):
    if arms == "park":
        arms = snap()
    return evaluate(boards, arms)[0]


def settle(tracker, boards, n=12, arms="park"):
    """Feed the same state n steps (still)."""
    res = None
    for _ in range(n):
        res = tracker.update(boards, snap() if arms == "park" else arms)
    return res


class Geometry(unittest.TestCase):
    def test_bay(self):
        self.assertAlmostEqual(2 * L.BAY_HALF, 0.330)
        self.assertAlmostEqual(L.FAR_X - L.BAY_X0, 0.200)
        # the side walls end where the far wall's inside face is: the walls meet without overlapping
        self.assertAlmostEqual(L.SIDE_WALL_X + L.SIDE_WALL_LEN / 2, L.FAR_X)
        self.assertAlmostEqual(L.FAR_WALL_LEN / 2, L.BAY_HALF + L.WALL_T)
        # a 300 mm board centred in the bay: 15 mm each end, in the window only within 5 mm of centred
        self.assertAlmostEqual((2 * L.BAY_HALF - L.BOARD_SIZE[0]) / 2, 0.015)

    def test_start_layout_outside_the_bay(self):
        for n, s in start().items():
            i = board_info(s)
            self.assertFalse(i["laid"], n)
            self.assertFalse(i["pinned"], n)
            self.assertGreaterEqual(min(i["walls"].values()), 0.010, n)
        # the two left boards 25 mm apart (room for the open fingers), the inner ones 20 mm outside the walls
        self.assertAlmostEqual(L.BOARD_START["board_b"][1] - L.BOARD_START["board_a"][1] - W, 0.025)
        self.assertAlmostEqual(L.BOARD_START["board_a"][1] - W / 2 - (L.BAY_HALF + L.WALL_T), 0.025)

    def test_outline_tilt_axis(self):
        s = board(0.3, 0.0)
        poly = outline(s)
        self.assertAlmostEqual(poly[:, 0].max() - poly[:, 0].min(), W, places=9)
        self.assertAlmostEqual(poly[:, 1].max() - poly[:, 1].min(), L.BOARD_SIZE[0], places=9)
        self.assertAlmostEqual(tilt_deg(s), 0.0, places=6)
        self.assertAlmostEqual(axis_err_deg(s), 0.0, places=6)
        self.assertAlmostEqual(axis_err_deg(board(0.3, turn=2.5)), 2.5, places=6)
        self.assertAlmostEqual(axis_err_deg(board(0.3, turn=180)), 0.0, places=6)
        self.assertAlmostEqual(axis_err_deg(board(0.25, along_x=True)), 90.0, places=6)
        self.assertAlmostEqual(tilt_deg(board(0.3, flip=True)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(board(0.3, tilt_x=90)), 90.0, places=6)
        self.assertAlmostEqual(lowest(board(0.3, lift=0.004)), 0.004, places=9)

    def test_span_and_segment_distance(self):
        sq = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        self.assertEqual(span_at_y(sq, 0.5), (0.0, 1.0))
        self.assertIsNone(span_at_y(sq, 1.5))
        self.assertAlmostEqual(segment_outline_distance(((2.0, -1.0), (2.0, 2.0)), sq), 1.0)
        self.assertEqual(segment_outline_distance(((0.5, -1.0), (0.5, 2.0)), sq), 0.0)  # across it
        self.assertEqual(segment_outline_distance(((1.0, -1.0), (1.0, 2.0)), sq), 0.0)  # along an edge
        self.assertAlmostEqual(segment_outline_distance(((2.0, 2.0), (3.0, 3.0)), sq), math.sqrt(2))

    def test_wall_distances(self):
        d = wall_distances(board(X_A))
        self.assertAlmostEqual(d["far_wall"], 0.015, places=9)
        self.assertAlmostEqual(d["left_wall"], 0.015, places=9)
        self.assertAlmostEqual(d["right_wall"], 0.015, places=9)
        d = wall_distances(board(X_A, 0.004))
        self.assertAlmostEqual(d["left_wall"], 0.011, places=9)
        self.assertAlmostEqual(d["right_wall"], 0.019, places=9)

    def test_finger_zone(self):
        a = SimpleNamespace(tcp=np.array([0.3, 0.0, TOP + 0.1]), tips=np.array([0.3, 0.0, TOP + 0.087]), opening=1.0)
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))
        a.opening = 0.0
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FINGER_THICK, L.FINGER_HALF_WIDTH))
        pts, r = finger_zone(a)
        self.assertAlmostEqual(float(pts[:, 2].min()), TOP + 0.087)
        self.assertAlmostEqual(float(pts[:, 2].max()), TOP + 0.087 + L.FINGER_LEN)
        # over the floor: tips 87 mm up, boards 16 mm tall, closed radius 22.4 mm
        self.assertAlmostEqual(zone_gap(a, floor()), 0.087 - 0.016 - zone_radius(a), places=9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(start()), 0.0)

    def test_full_floor_scores_one(self):
        score, d = evaluate(floor(), snap())
        self.assertEqual(score, 1.0)
        self.assertEqual(d["n_laid"], 3)
        self.assertAlmostEqual(d["far_gap_mm"], 15.0)
        self.assertEqual(d["joints_mm"], [0.0, 0.0])

    def test_one_per_laid_board(self):
        out = start()
        self.assertEqual(score_of(out), 0.0)
        out["board_a"] = board(X_A)
        self.assertEqual(score_of(out), 0.25)
        out["board_b"] = board(X_B)
        self.assertEqual(score_of(out), 0.5)
        out["board_c"] = board(X_C)
        self.assertEqual(score_of(out), 1.0)

    def test_any_board_in_any_place(self):
        self.assertEqual(score_of({"board_a": board(X_C), "board_b": board(X_A), "board_c": board(X_B)}), 1.0)

    def test_open_joint_scores_075(self):
        self.assertEqual(score_of(floor(xb=X_B - 0.003, xc=X_C - 0.003)), 1.0)  # 3.0 mm joint: tight
        self.assertEqual(score_of(floor(xb=X_B - 0.0032, xc=X_C - 0.0032)), 0.75)
        self.assertEqual(score_of(floor(xc=X_C - 0.010)), 0.75)  # b tight, c 10 mm short

    def test_joint_is_read_at_both_ends(self):
        # c 2.5 mm short and turned 1 degree about its centre: 2.5 +- 2.4 mm on the two measuring lines
        b = floor()
        b["board_c"] = board(X_C - 0.0025, turn=1.0)
        _, d = evaluate(b, snap())
        self.assertGreater(d["joints_mm"][1], 3.0)
        self.assertEqual(score_of(b), 0.75)
        b["board_c"] = board(X_C - 0.0025, turn=0.15)  # 2.5 +- 0.37 mm: both 3 mm or less
        self.assertEqual(score_of(b), 1.0)

    def test_far_gap_window(self):
        for gap, want in ((0.010, 1.0), (0.020, 1.0), (0.0098, 0.75), (0.0202, 0.75), (0.035, 0.75)):
            d = 0.015 - gap
            self.assertEqual(score_of(floor(X_A + d, X_B + d, X_C + d)), want, gap)

    def test_floor_slid_to_the_open_edge(self):
        # all three tight, the floor's near edge at the bay's open edge: far gap 50 mm
        x0 = L.BAY_X0 + W / 2
        self.assertEqual(score_of(floor(x0 + 2 * W, x0 + W, x0)), 0.75)
        # past the open edge: c is no longer inside the bay
        self.assertEqual(score_of(floor(x0 + 2 * W - 0.002, x0 + W - 0.002, x0 - 0.002)), 0.5)

    def test_side_gap_window(self):
        for y, want in ((0.005, 1.0), (-0.005, 1.0), (0.0052, 0.5), (-0.0052, 0.5)):  # c not laid: 0.5
            self.assertEqual(score_of(floor(yc=y)), want, y)
        self.assertEqual(score_of(floor(ya=0.006, yb=0.006, yc=0.006)), 0.0)  # none laid (9 and 21 mm)

    def test_flat_and_on_the_floor(self):
        # c not laid leaves two laid boards: 0.5
        for kw, want in (({"lift": 0.002}, 1.0), ({"lift": 0.0022}, 0.5), ({"tilt_x": 3.0}, 1.0),
                         ({"tilt_x": 3.3}, 0.5), ({"turn": 3.0}, None), ({"turn": 3.3}, 0.5)):
            b = floor()
            x = X_C - (0.006 if "turn" in kw else 0.0)  # a turned board needs room
            b["board_c"] = board(x, **kw)
            if want is not None:
                self.assertEqual(score_of(b), want, kw)
            else:
                self.assertTrue(board_info(b["board_c"])["laid"], kw)

    def test_hands_off(self):
        b = floor()
        near = np.array([X_C - W / 2 - 0.030 - math.hypot(L.FINGER_THICK, L.FINGER_HALF_WIDTH), 0.0, TOP + 0.010])
        self.assertEqual(evaluate(b, snap(right_tcp=near + [0, 0, TIPS]))[0], 1.0)  # 30 mm clear
        self.assertEqual(evaluate(b, snap(right_tcp=near + [0.002, 0, TIPS]))[0], 0.75)  # 28 mm
        over = np.array([X_B, 0.0, TOP + 0.016 + 0.020 + TIPS])  # closed fingers 20 mm over b's top
        self.assertEqual(evaluate(b, snap(left_tcp=over))[0], 0.75)

    def test_arms_unavailable_taken_as_met(self):
        self.assertEqual(evaluate(floor(), None)[0], 1.0)
        bad = SimpleNamespace(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3), opening=0.0)
        self.assertEqual(evaluate(floor(), {"left": bad, "right": bad})[0], 1.0)

    def test_success_needs_stillness_and_holds(self):
        tr = Tracker(HZ)
        res = tr.update(floor(), snap())
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.success)  # 0.5 s of history not yet seen
        res = settle(tr, floor(), 10)
        self.assertTrue(res.success)
        # c creeping 0.3 mm per step: 3 mm in 0.5 s, not still
        for i in range(12):
            res = tr.update(floor(xc=X_C - 0.004 + 0.0003 * i), snap())
        self.assertFalse(res.success)
        res = settle(tr, floor(xc=X_C - 0.0007), 11)
        self.assertTrue(res.success)

    def test_score_is_idempotent(self):
        tr = Tracker(HZ)
        settle(tr, floor())
        a = tr.result(floor(), snap())
        b = tr.result(floor(), snap())
        self.assertEqual((a.score, a.success), (b.score, b.success))
        self.assertEqual(tr.steps, 12)


class ZeroConditions(unittest.TestCase):
    def test_a_board_against_the_far_wall(self):
        d = 0.015
        self.assertEqual(score_of(floor(X_A + d, X_B + d, X_C + d)), 0.0)  # touching
        for gap, want in ((0.003, 0.0), (0.0032, 0.75)):
            d = 0.015 - gap
            self.assertEqual(score_of(floor(X_A + d, X_B + d, X_C + d)), want, gap)

    def test_a_board_against_a_side_wall(self):
        self.assertEqual(score_of(floor(yb=0.012)), 0.0)  # 3 mm from the left wall
        self.assertEqual(score_of(floor(yb=-0.0118)), 0.5)  # 3.2 mm from the right wall: b not laid, a and c are
        self.assertEqual(score_of(floor(yb=0.015)), 0.0)  # touching

    def test_any_board_counts_laid_or_not(self):
        b = start()
        b["board_c"] = board(0.303, 0.0, turn=12)  # turned 12 degrees (not laid), a corner 1.3 mm from the far wall
        info = board_info(b["board_c"])
        self.assertTrue(info["pinned"])
        self.assertEqual(score_of(b), 0.0)

    def test_a_board_on_a_wall_top_or_leaning_on_one(self):
        b = floor()
        b["board_c"] = board(0.26, L.SIDE_WALL_Y, along_x=True, lift=L.WALL_H)  # lying along the left wall's top
        self.assertEqual(score_of(b), 0.0)
        b["board_c"] = board(L.FAR_WALL_X + 0.010, 0.0, lift=L.WALL_H)  # on the far wall's top, overhanging it
        self.assertEqual(score_of(b), 0.0)
        lean = state((0.2, 0.0, TOP + 0.030), qmul(about("z", 90), about("y", 20)))  # one end up near a wall
        b["board_c"] = lean
        self.assertTrue(board_info(lean)["pinned"] or not board_info(lean)["laid"])

    def test_held_above_the_walls_is_not_pinned(self):
        b = floor()
        self.assertAlmostEqual(L.WALL_ZONE_Z, L.WALL_H + 0.002)
        b["board_c"] = board(0.30, 0.10, along_x=True, lift=0.0425)  # carried over the left wall, 42.5 mm up
        self.assertFalse(board_info(b["board_c"])["pinned"])
        self.assertEqual(score_of(b), 0.5)
        b["board_c"] = board(0.30, 0.10, along_x=True, lift=0.042)  # at the walls' height + 2 mm: counts
        self.assertEqual(score_of(b), 0.0)

    def test_touching_a_wall_from_outside_the_bay_is_not_pinned(self):
        b = floor()
        b["board_c"] = board(0.26, L.BAY_HALF + L.WALL_T + W / 2, along_x=True)  # against the left wall's outside
        self.assertFalse(board_info(b["board_c"])["pinned"])
        self.assertEqual(score_of(b), 0.5)


class Exploits(unittest.TestCase):
    def test_a_board_left_on_top_of_the_floor(self):
        b = floor()
        b["board_c"] = board(X_B, lift=0.016)
        self.assertFalse(board_info(b["board_c"])["laid"])
        self.assertEqual(score_of(b), 0.5)

    def test_boards_on_edge(self):
        b = floor()
        b["board_c"] = board(X_C, tilt_x=90)
        self.assertEqual(score_of(b), 0.5)

    def test_two_boards_tight_far_gap_ok_third_missing(self):
        b = floor()
        b["board_c"] = start()["board_c"]
        self.assertEqual(score_of(b), 0.5)

    def test_the_floor_driven_into_the_far_wall_then_pulled_back_is_fine(self):
        # only the end state counts: a floor that touched the wall and was drawn back to 15 mm scores 1.0
        tr = Tracker(HZ)
        settle(tr, floor(X_A + 0.015, X_B + 0.015, X_C + 0.015), 5)
        self.assertEqual(tr.last.score, 0.0)
        self.assertTrue(settle(tr, floor(), 12).success)

    def test_upside_down_boards_score_the_same(self):
        self.assertEqual(score_of(floor(flip=True)), 1.0)
        self.assertEqual(score_of(floor(turn=180)), 1.0)


class Arms(unittest.TestCase):
    def test_unavailable_counted(self):
        tr = Tracker(HZ)
        tr.update(floor(), None)
        tr.update(floor(), snap())
        self.assertEqual(tr.arms_unavailable, 1)

    def test_offline_feed_reaches_the_scorer(self):
        arms = snap(left_tcp=[0.3, 0.0, TOP + 0.05], left_open=0.25)
        self.assertAlmostEqual(arms["left"].opening, 0.25)
        np.testing.assert_allclose(arms["left"].tips, [0.3, 0.0, TOP + 0.05 - TIPS])


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr = Tracker(HZ)
        settle(tr, floor(), 12)
        hist = {n: len(h) for n, h in tr.history.items()}
        for bad in (np.nan, np.inf):
            b = floor()
            b["board_b"] = SimpleNamespace(**{**vars(b["board_b"]), "pos": np.array([bad, 0.0, 0.0])})
            res = tr.update(b, snap())
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
        b = floor()
        b["board_a"] = SimpleNamespace(**{**vars(b["board_a"]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(tr.update(b, snap()).score, 0.0)
        b = floor()
        del b["board_c"]
        self.assertEqual(tr.update(b, snap()).score, 0.0)
        self.assertEqual(tr.update({"board_a": None}, snap()).score, 0.0)
        self.assertEqual(tr.bad_steps, 5)
        self.assertEqual({n: len(h) for n, h in tr.history.items()}, hist)
        self.assertTrue(tr.update(floor(), snap()).success)  # the record of the trial is intact
        self.assertFalse(valid(SimpleNamespace(pos=[0, 0, 0])))

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(7)
        tr = Tracker(HZ)
        seen = set()
        for _ in range(400):
            b = {}
            for n in L.BOARD_NAMES:
                q = np.array([rng.uniform(-1, 1) for _ in range(4)])
                q /= np.linalg.norm(q)
                if rng.random() < 0.5:
                    q = about("z", rng.uniform(-180, 180))
                b[n] = state((rng.uniform(0.1, 0.45), rng.uniform(-0.3, 0.3), TOP + rng.uniform(0.0, 0.06)), q)
            res = tr.update(b, snap() if rng.random() < 0.8 else None)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(res.score)
        self.assertIn(0.0, seen)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr = Tracker(HZ)
        settle(tr, floor(), 12)
        self.assertEqual(tr.steps, 12)
        tr.update(start(), snap())
        self.assertEqual(tr.steps, 1)
        self.assertIsNone(tr.first_full_step)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker(HZ)
        a0 = np.array(L.BOARD_START["board_a"])
        for k in range(30):  # board a carried back to its start 5 mm per step, the others there already
            p = a0 + np.array([0.0, -0.15]) * (1 - k / 29)
            b = start()
            b["board_a"] = board(p[0], p[1], along_x=True)
            tr.update(b, snap())
        self.assertEqual(tr.steps, 30)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr = Tracker(HZ)
        settle(tr, start(), 3)
        tr.update(floor(), snap())
        self.assertEqual(tr.steps, 4)

    def test_rigstate_reset_hook_runs_registered_callbacks(self):
        saved = dict(rigstate._RESET_CALLBACKS)
        hits = []
        try:
            rigstate.on_reset(lambda: hits.append(1))
            rigstate.offline_reset()
            self.assertEqual(hits, [1])
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


class PushModel(unittest.TestCase):
    """offline_push.py, the offline checker's stand-in for pushing (the simulator never uses it)."""

    def model(self, boards, compliance=L.PUSH_COMPLIANCE):
        from offline_push import BoardPush
        m = BoardPush(compliance=compliance)
        self.feed(m, boards)
        return m

    @staticmethod
    def feed(m, boards):
        return {n: m.board_state(n, boards[n]) for n in L.BOARD_NAMES}

    def push(self, m, boards, x_from, x_to, side="left", steps=60, opening=0.0, z=TOP + 0.018):
        out = None
        for x in np.linspace(x_from, x_to, steps):
            tcp = [x, 0.0, z]
            if side == "left":
                snap(left_tcp=tcp, left_open=opening)
            else:
                snap(right_tcp=tcp, right_open=opening)
            out = self.feed(m, boards)
        return out

    def test_push_closes_the_joint_and_stops_against_the_neighbour(self):
        b = floor(xb=X_B - 0.018, xc=0.15)  # c out of the way
        b["board_c"] = start()["board_c"]
        m = self.model(b)
        near = X_B - 0.018 - W / 2
        out = self.push(m, b, near - 0.016, X_B - W / 2 - L.PAD_THICK + 0.008)
        self.assertAlmostEqual(float(out["board_b"].pos[0]), X_B, places=9)  # tight against a
        self.assertAlmostEqual(float(out["board_a"].pos[0]), X_A, places=9)  # a not driven
        self.assertEqual(m.warnings, [])

    def test_too_little_overshoot_leaves_the_joint_open(self):
        b = floor(xb=X_B - 0.018)
        b["board_c"] = start()["board_c"]
        m = self.model(b)
        trail = L.PUSH_COMPLIANCE * L.FRICTION * L.BOARD_MASS * L.GRAVITY
        out = self.push(m, b, X_B - 0.018 - W / 2 - 0.016, X_B - W / 2 - L.PAD_THICK + trail - 0.002)
        self.assertAlmostEqual(float(out["board_b"].pos[0]), X_B - 0.002, places=9)

    def test_a_stiff_arm_drives_the_floor(self):
        b = floor(xb=X_B - 0.018)
        b["board_c"] = start()["board_c"]
        m = self.model(b, compliance=0.002)
        out = self.push(m, b, X_B - 0.018 - W / 2 - 0.016, X_B - W / 2 - L.PAD_THICK + 0.008)
        f2 = 2 * L.FRICTION * L.BOARD_MASS * L.GRAVITY
        moved = 0.008 - 0.002 * f2
        self.assertAlmostEqual(float(out["board_a"].pos[0]), X_A + moved, places=9)
        self.assertAlmostEqual(float(out["board_b"].pos[0]), X_B + moved, places=9)

    def test_the_far_wall_stops_the_floor(self):
        b = floor(xa=L.FAR_X - 0.002 - W / 2, xb=X_B - 0.018)
        b["board_c"] = start()["board_c"]
        m = self.model(b, compliance=0.0005)
        out = self.push(m, b, X_B - 0.018 - W / 2 - 0.016, X_B - W / 2 + 0.010)
        self.assertAlmostEqual(float(out["board_a"].high[0]), L.FAR_X, places=9)
        self.assertTrue(any(w["issue"] == "drove the floor into the far wall" for w in m.warnings))

    def test_open_fingers_do_not_push(self):
        b = floor(xb=X_B - 0.018)
        m = self.model(b)
        out = self.push(m, b, X_B - 0.018 - W / 2 - 0.016, X_B, opening=0.6)
        self.assertAlmostEqual(float(out["board_b"].pos[0]), X_B - 0.018, places=9)

    def test_coming_down_onto_a_board_is_recorded(self):
        b = floor(xb=X_B - 0.018)
        m = self.model(b)
        for z in np.linspace(TOP + 0.08, TOP + 0.018, 30):
            snap(left_tcp=[X_B - 0.030, 0.0, z])
            self.feed(m, b)
        self.assertTrue(any(w["issue"] == "came down onto the board" for w in m.warnings))

    def test_a_carried_board_follows_the_checker(self):
        b = floor(xb=X_B - 0.018)
        b["board_c"] = start()["board_c"]
        m = self.model(b)
        self.push(m, b, X_B - 0.018 - W / 2 - 0.016, X_B - W / 2 - L.PAD_THICK + 0.008)
        self.assertGreater(m.off["board_b"], 0.017)
        b["board_b"] = board(0.25, 0.1, lift=0.05)  # the checker picked it up
        out = self.feed(m, b)
        self.assertEqual(m.off["board_b"], 0.0)
        np.testing.assert_allclose(out["board_b"].pos, b["board_b"].pos)

    def test_no_arms_no_push(self):
        b = floor(xb=X_B - 0.018)
        m = self.model(b)
        rigstate.offline_reset()
        out = self.feed(m, b)
        self.assertAlmostEqual(float(out["board_b"].pos[0]), X_B - 0.018, places=9)


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, rigstate reset hook, the demo, the
    task.json text, and the offline checker on the whole folder."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")

    def test_objects(self):
        objs = self.task.OBJECTS
        names = [o.name for o in objs]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in objs if o.mass is not None], list(L.BOARD_NAMES))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        spec = {o.name: o for o in objs}
        for n, (x, y) in L.BOARD_START.items():
            np.testing.assert_allclose(spec[n].pos, (x, y, TOP + H))
            self.assertEqual(tuple(spec[n].size), L.BOARD_SIZE)
            self.assertAlmostEqual(spec[n].mass, L.BOARD_MASS)
            self.assertLessEqual(spec[n].mass, 0.3)
        # the walls' inside faces are the scorer's faces
        left, right, far = spec["wall_left"], spec["wall_right"], spec["wall_far"]
        self.assertAlmostEqual(left.pos[1] - left.size[1] / 2, L.BAY_HALF)
        self.assertAlmostEqual(right.pos[1] + right.size[1] / 2, -L.BAY_HALF)
        self.assertAlmostEqual(far.pos[0] - far.size[0] / 2, L.FAR_X)
        self.assertAlmostEqual(left.pos[0] - left.size[0] / 2, L.BAY_X0)
        self.assertAlmostEqual(left.pos[0] + left.size[0] / 2, L.FAR_X)
        for w in (left, right, far):
            self.assertIsNone(w.mass)
            self.assertAlmostEqual(w.size[2], L.WALL_H)
        # the arms' start pose covers the table near each base; nothing of the task lies there
        for n in L.BOARD_NAMES:
            self.assertLess(abs(spec[n].pos[1]) + W / 2, 0.45)

    def test_offline_specs_carry_the_push_hooks(self):
        spec = {o.name: o for o in self.task.OBJECTS}
        for n in L.BOARD_NAMES:
            self.assertTrue(callable(getattr(spec[n], "offline_state", None)))
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual([n for n in names if n.startswith("board")], list(L.BOARD_NAMES))

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(start()))
        self.assertEqual(self.task.score(start()), 0.0)
        self.task._PUSH.off["board_b"] = 0.01
        rigstate.offline_reset()  # rigstate's reset hook: the tracker and the push model start over
        self.assertEqual(tr.steps, 0)
        self.assertEqual(self.task._PUSH.off["board_b"], 0.0)
        self.assertFalse(self.task.success(floor()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        for _ in range(11):
            snap()
            ok = self.task.success(floor())
        self.assertTrue(ok)
        self.assertEqual(self.task.score(floor()), 1.0)
        self.assertFalse(self.task.success({"board_a": board(X_A)}))  # missing boards: no crash, 0
        self.assertEqual(self.task.score({"board_a": board(X_A)}), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        tmp = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(tmp)
            rigstate.offline_reset()
            snap()
            self.task.success(floor())
            row = json.loads(tmp.read_text().splitlines()[-1])
            for key in ("score", "n_laid", "far_gap_mm", "joints_mm", "hands_off", "left_zone_mm", "moved_mm",
                        "board_a", "board_a_pos", "left_tcp", "pinned", "arms_unavailable"):
                self.assertIn(key, row)
            self.assertIn("wall_mm", row["board_b"])
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(floor()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
            if tmp.exists():
                tmp.unlink()
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        t_of = {a: np.cumsum([w.duration for w in plan[a]]) for a in plan}

        def when(arm, start_of):
            i = next(i for i, w in enumerate(plan[arm]) if w.label.startswith(start_of))
            return t_of[arm][i] - plan[arm][i].duration, t_of[arm][i]

        # the right arm comes over the bay only once the left arm has parked
        self.assertGreaterEqual(when("right", "carry board c")[0], when("left", "park")[1] - 1e-9)
        # every open or close takes 1.5 s or more; every final set-down is slow
        for wps in plan.values():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
            for w in wps:
                if w.label.startswith("slowly down"):
                    self.assertGreaterEqual(w.duration, 2.0)
        # pushes: fingers closed, across the bay (yaw -90 left, +90 right), at the push height, along x only
        for arm, label, yaw in (("left", "push board b", -math.pi / 2), ("right", "push board c", math.pi / 2)):
            i = next(i for i, w in enumerate(plan[arm]) if w.label.startswith(label))
            w, prev = plan[arm][i], plan[arm][i - 1]
            self.assertEqual((w.finger, w.yaw), (t.CLOSED, yaw))
            self.assertAlmostEqual(w.pos[2], t.PUSH_Z)
            self.assertEqual((prev.pos[1], prev.pos[2]), (w.pos[1], w.pos[2]))
            self.assertGreater(w.pos[0], prev.pos[0])
        # the geometry the demo is built on
        self.assertAlmostEqual(t.X_A + W / 2, L.FAR_X - t.FAR_SET_GAP)
        self.assertAlmostEqual(t.B_PUSH_TO + t.FINGER_FRONT, t.X_B - W / 2 + t.OVERSHOOT)
        self.assertAlmostEqual(t.C_PUSH_TO + t.FINGER_FRONT, t.X_C - W / 2 + t.OVERSHOOT)
        self.assertAlmostEqual(t.B_PUSH_FROM + t.FINGER_FRONT, t.X_B_SET - W / 2 - t.APPROACH)
        # set down, the far finger opened to RELEASE stays clear of the board ahead (and of the far wall)
        reach = -t.RELEASE - W / 2 + L.PAD_THICK  # past the board's far face
        self.assertLess(reach, t.SET_GAP_B - 0.005)
        self.assertLess(reach, t.SET_GAP_C - 0.005)
        self.assertLess(reach, t.FAR_SET_GAP - 0.005)
        # the overshoot sits inside the window: one board's friction trail at most 10 mm/N, two boards' at 2 mm/N
        f = L.FRICTION * L.BOARD_MASS * L.GRAVITY
        self.assertGreater(t.OVERSHOOT + L.JOINT_MAX, 0.010 * f)
        self.assertLess((t.OVERSHOOT - 0.002 * 2 * f) + (t.OVERSHOOT - 0.002 * 3 * f), t.FAR_SET_GAP - L.GAP_MIN)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_task_json_matches_the_task(self):
        d = json.loads((HERE / "task.json").read_text())
        for k in ("title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "reset_notes",
                  "context", "keywords"):
            self.assertTrue(d.get(k), k)
        self.assertEqual(d["instruction"], self.task.INSTRUCTION)
        self.assertEqual(d["category"], "Impact on jobs")
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        s = d["scoring"]
        for phrase in ("10 to 20 mm", "3 mm", "3 degrees", "2 mm above the table", "42 mm", "y +14.0 cm",
                       "30 mm", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0"):
            self.assertIn(phrase, s)
        for x, y in L.BOARD_START.values():
            self.assertIn(f"x {x * 100:.1f} cm and y {y * 100:+.1f} cm", d["setup"])
            self.assertIn(f"x {x * 100:.1f} cm and y {y * 100:+.1f} cm", d["reset_notes"])
        self.assertIn("2,400 steps", d["run_limit"])

    def test_offline_checker_predicts_one_and_the_pushes_stay_clean(self):
        """tools/offline/check_task.py --strict --one-arm on this folder (no reach table, for time), and its
        playback's push model: no gripper came down onto a board or pushed one from in front. Run in-process with
        sys.modules and sys.path restored afterwards (the checker reloads the task's helper modules)."""
        saved_modules, saved_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(_offline_dir()))
        try:
            import check_task
            rep, _, code = check_task.check(HERE, strict=True, one_arm=True, reach=False)
            self.assertEqual(code, 0, rep.fails)
            self.assertEqual(rep.data["playback"]["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(rep.data["one_arm"], {"left_only": 0.5, "right_only": 0.25})
            task = check_task.load_task(HERE / "task.py", HERE)
            plans = check_task.get_plans(task, check_task.Report())
            solids = [check_task.G.solid_from_spec(o, task.table_top) for o in task.objects]
            out = check_task.Playback(task, plans, solids, hz=HZ, hold_s=3.0).run()
            self.assertEqual(out["proxy"]["predicted_task_score"], 1.0)
            push = task.module._PUSH
            self.assertEqual(push.warnings, [])
            self.assertEqual({e["board"] for e in push.log}, {"board_b", "board_c"})  # a is never driven
            last = task.module._TRACKER.last.details
            self.assertEqual(last["joints_mm"], [0.0, 0.0])
            self.assertAlmostEqual(last["far_gap_mm"], 1000 * task.module.FAR_SET_GAP, delta=0.05)
            check_task._forget_task_modules()
        finally:
            for name in [n for n in sys.modules if n not in saved_modules]:
                del sys.modules[name]
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path


if __name__ == "__main__":
    unittest.main()
