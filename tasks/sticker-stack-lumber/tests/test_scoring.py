"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import json
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
    BOARD_HALF, STICKER_HALF, Tracker, board_on_bearers, bearer_top, broad_face, crossing, evaluate,
    heading_err_deg, laid_board, laid_sticker, on_edge, off_table, rotation, sticker_on_boards, tilt_deg,
)

KT = L.STICKER_SIZE[2]


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def qmul(a, b):
    """The turn ``b`` followed by the turn ``a`` (quaternions xyzw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def state(center, size, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half  # the world-aligned bounding box of the turned box
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext, size=tuple(size))


def flat_piece(size, x, y, bottom, yaw=0.0, flip=False, **kw):
    """A piece lying flat with its underside at ``bottom``, turned ``yaw`` degrees (``flip``: other face up)."""
    q = about("z", yaw)
    if flip:
        q = qmul(q, about("x", 180))
    return state((x, y, bottom + size[2] / 2), size, q, **kw)


def board_at(x, y, bottom, yaw=0.0, **kw):
    return flat_piece(L.BOARD_SIZE, x, y, bottom, yaw, **kw)


def sticker_at(x, y, bottom, yaw=0.0, **kw):
    return flat_piece(L.STICKER_SIZE, x, y, bottom, yaw, **kw)


def start():
    boards = [board_at(L.BOARD_START_X, y, L.TOP) for _, y in L.BOARDS]
    stickers = [sticker_at(x, y, L.TOP, np.degrees(L.STICKER_START_YAW)) for _, x, y in L.STICKERS]  # along x
    return boards, stickers


def stack(l1_y=(L.LAYER_Y, -L.LAYER_Y), k_x=(L.BEARER_XS[1], L.BEARER_XS[0]), l2_y=None, k_y=(0.0, 0.0),
          k_yaw=(0.0, 0.0)):
    """The finished stack (order: left inner, left outer, right inner, right outer boards; left, right
    stickers), with every position adjustable."""
    l2_y = l1_y if l2_y is None else l2_y
    lower = [board_at(L.BOARD_X, y, L.LAYER1_BOTTOM) for y in l1_y]
    upper = [board_at(L.BOARD_X, y, L.LAYER2_BOTTOM) for y in l2_y]
    stickers = [sticker_at(x, y, L.STICKER_BOTTOM, yaw) for x, y, yaw in zip(k_x, k_y, k_yaw)]
    return [lower[0], upper[0], lower[1], upper[1]], stickers


B0, K0 = start()


def score_of(boards, stickers):
    return evaluate(boards, stickers)[0]


def run(frames, hz=20.0):
    """Feed (boards, stickers) frames to a fresh tracker; return the last result and the tracker."""
    t = Tracker(hz)
    r = None
    for boards, stickers in frames:
        r = t.update(boards, stickers)
    return r, t


def glide(a, b, n):
    """n frames moving every piece of frame a to its place in frame b in equal steps (orientations of b)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        frame = []
        for pa, pb in zip(a, b):
            frame.append([state((1 - f) * sa.pos + f * sb.pos, sb.size, sb.quat_xyzw) for sa, sb in zip(pa, pb)])
        out.append(tuple(frame))
    return out


def build(n=20):
    """From the start, every piece moved gradually into the finished stack, then held there."""
    full = stack()
    return [(B0, K0)] * 5 + glide((B0, K0), full, n) + [full] * 10


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_and_heading_either_way_round(self):
        self.assertAlmostEqual(tilt_deg(about("x", 6)), 6.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 180)), 0.0, places=4)  # the other broad face down
        self.assertAlmostEqual(tilt_deg(about("y", 174)), 6.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 90.0, places=4)  # on its edge
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(heading_err_deg(board_at(0.3, 0, L.TOP, 3), BOARD_HALF, "x"), 3.0, places=6)
        self.assertAlmostEqual(heading_err_deg(board_at(0.3, 0, L.TOP, 177), BOARD_HALF, "x"), 3.0, places=6)
        self.assertAlmostEqual(heading_err_deg(board_at(0.3, 0, L.TOP, 90), BOARD_HALF, "x"), 90.0, places=6)
        self.assertAlmostEqual(heading_err_deg(sticker_at(0.3, 0, L.TOP), STICKER_HALF, "y"), 0.0, places=6)
        self.assertAlmostEqual(heading_err_deg(sticker_at(0.3, 0, L.TOP, -184), STICKER_HALF, "y"), 4.0, places=6)
        self.assertAlmostEqual(heading_err_deg(sticker_at(0.3, 0, L.TOP, 90), STICKER_HALF, "y"), 90.0, places=6)

    def test_laid(self):
        self.assertTrue(laid_board(board_at(0.3, 0, L.TOP, 4.9)))
        self.assertFalse(laid_board(board_at(0.3, 0, L.TOP, 5.1)))
        self.assertTrue(laid_board(board_at(0.3, 0, L.TOP, flip=True)))
        tilted = state((0.3, 0, L.TOP + 0.02), L.BOARD_SIZE, about("x", 5.5))
        self.assertFalse(laid_board(tilted))
        self.assertTrue(laid_sticker(sticker_at(0.3, 0, L.TOP, 175.5)))
        self.assertFalse(laid_sticker(sticker_at(0.3, 0, L.TOP, 84)))  # along x, the boards' way

    def test_edge_and_off_table(self):
        on_side = state((0.46, 0.2, L.TOP + L.STICKER_SIZE[0] / 2), L.STICKER_SIZE, about("y", 90))
        self.assertTrue(on_edge(on_side))
        self.assertFalse(on_edge(state((0.3, 0, L.TOP + 0.03), L.BOARD_SIZE, about("x", 44))))
        self.assertTrue(off_table(board_at(0.3, 0.75, L.TOP - 0.3)))
        self.assertFalse(off_table(board_at(0.3, 0.0, L.TOP - 0.009)))

    def test_board_on_bearers_tolerances(self):
        nominal = board_at(L.BOARD_X, 0.05, L.LAYER1_BOTTOM)
        self.assertTrue(board_on_bearers(nominal))
        self.assertTrue(board_on_bearers(board_at(L.BOARD_X, 0.05, L.LAYER1_BOTTOM + 0.0029)))
        self.assertFalse(board_on_bearers(board_at(L.BOARD_X, 0.05, L.LAYER1_BOTTOM + 0.0031)))  # held above
        self.assertFalse(board_on_bearers(board_at(L.BOARD_X, 0.05, L.TOP)))  # on the table between them
        # along x: a board 35 mm toward the far bearer still covers 3/4 of the near bearer's width, 45 mm only 1/4
        self.assertTrue(board_on_bearers(board_at(L.BOARD_X + 0.035, 0.05, L.LAYER1_BOTTOM)))
        self.assertFalse(board_on_bearers(board_at(L.BOARD_X + 0.045, 0.05, L.LAYER1_BOTTOM)))
        # along y: past the bearers' ends (y 0.16) by more than half its width
        self.assertTrue(board_on_bearers(board_at(L.BOARD_X, 0.155, L.LAYER1_BOTTOM)))
        self.assertFalse(board_on_bearers(board_at(L.BOARD_X, 0.175, L.LAYER1_BOTTOM)))
        self.assertFalse(board_on_bearers(board_at(L.BOARD_X, 0.05, L.LAYER1_BOTTOM, 8)))  # turned 8 degrees

    def test_crossing_area_matches_sampling(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            b = board_at(L.BOARD_X + rng.uniform(-0.06, 0.06), rng.uniform(-0.17, 0.17), L.LAYER1_BOTTOM,
                         rng.uniform(-60, 60))
            top = bearer_top(0)
            area, gap = crossing(top, broad_face(b, BOARD_HALF, "bottom"))
            u, v = np.meshgrid(np.linspace(-0.5, 0.5, 301), np.linspace(-0.5, 0.5, 61))
            pts = np.stack([u.ravel() * L.BOARD_SIZE[0], v.ravel() * L.BOARD_SIZE[1], np.zeros(u.size)], 1)
            world = pts @ rotation(b.quat_xyzw).T + b.pos
            inside = (np.abs(world[:, 0] - L.BEARER_XS[0]) <= L.BEARER_SIZE[0] / 2) & \
                     (np.abs(world[:, 1] - L.BEARER_Y) <= L.BEARER_SIZE[1] / 2)
            self.assertAlmostEqual(area, inside.mean() * L.BOARD_SIZE[0] * L.BOARD_SIZE[1], delta=2.5e-5)
            if gap is not None:
                self.assertAlmostEqual(gap, 0.0, places=9)

    def test_sticker_on_tilted_or_lower_boards(self):
        lower = stack()[0][0::2]
        self.assertTrue(sticker_on_boards(sticker_at(0.36, 0, L.STICKER_BOTTOM), lower))
        self.assertFalse(sticker_on_boards(sticker_at(0.36, 0.12, L.STICKER_BOTTOM), lower))  # off the right board
        self.assertFalse(sticker_on_boards(sticker_at(0.36, 0, L.STICKER_BOTTOM + 0.004), lower))
        self.assertTrue(sticker_on_boards(sticker_at(0.36, 0, L.STICKER_BOTTOM, flip=True), lower))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([(B0, K0)] * 20)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_stack_scores_one(self):
        r, t = run(build())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["layer1_gap_mm"], 60.0)
        self.assertEqual(r.details["sticker_off_mm"], [0.0, 0.0])
        self.assertEqual(sorted(r.details["sticker_bearer"]), [0, 1])
        json.dumps(r.details, allow_nan=False)
        self.assertEqual(sorted(t.first_at), [0.25, 0.5, 0.75, 1.0])

    def test_any_board_anywhere_either_sticker_either_bearer_either_face_up(self):
        boards, stickers = stack(k_x=(L.BEARER_XS[0], L.BEARER_XS[1]))  # stickers swapped
        boards = [boards[1], boards[0], boards[3], boards[2]]  # outer boards below, inner above
        self.assertEqual(score_of(boards, stickers), 1.0)
        flipped = [board_at(b.pos[0], b.pos[1], b.low[2], 180, flip=True) for b in boards]
        stickers = [sticker_at(k.pos[0], k.pos[1], k.low[2], 180, flip=True) for k in stickers]
        self.assertEqual(score_of(flipped, stickers), 1.0)

    def test_sticker_offsets(self):
        bx = L.BEARER_XS
        cases = [(0.0099, 1.0), (0.0101, 0.75), (-0.020, 0.75), (0.0249, 0.75)]
        for off, want in cases:
            with self.subTest(off=off):
                self.assertEqual(score_of(*stack(k_x=(bx[1] + off, bx[0]))), want)
        # more than 25 mm off: the stickers do not count, so layer 2 does not either: layer 1 only
        self.assertEqual(score_of(*stack(k_x=(bx[1] - 0.0251, bx[0]))), 0.25)
        # a sticker over the gap between the bearers, or both over one bearer
        self.assertEqual(score_of(*stack(k_x=(0.28, bx[0]))), 0.25)
        self.assertEqual(score_of(*stack(k_x=(bx[0] + 0.009, bx[0] - 0.011))), 0.25)

    def test_layer2_offsets(self):
        y = L.LAYER_Y
        cases = [(0.0149, 1.0), (0.0151, 0.75), (-0.025, 0.75), (0.0299, 0.75), (0.0301, 0.5), (0.09, 0.5)]
        for off, want in cases:
            with self.subTest(off=off):
                self.assertEqual(score_of(*stack(l2_y=(y + off, -y))), want)
        # both a sticker and a top board loosely placed: 0.75 still
        self.assertEqual(score_of(*stack(k_x=(L.BEARER_XS[1] + 0.02, L.BEARER_XS[0]), l2_y=(y, -y - 0.02))), 0.75)

    def test_layer2_matched_to_the_nearer_lower_board(self):
        # the board that started on the right lies on the left: each top board is paired with the lower board
        # it is over, whatever the order of the pieces
        boards, stickers = stack()
        crossed = [boards[0], boards[3], boards[2], boards[1]]
        self.assertEqual(score_of(crossed, stickers), 1.0)

    def test_top_board_turned_or_on_one_sticker_is_not_layer2(self):
        boards, stickers = stack()
        turned = board_at(L.BOARD_X, -L.LAYER_Y, L.LAYER2_BOTTOM, 10)  # still on both stickers, but 10 deg off x
        self.assertEqual(score_of([*boards[:3], turned], stickers), 0.5)
        far_only = board_at(L.BOARD_X + 0.09, -L.LAYER_Y, L.LAYER2_BOTTOM)  # spans x 0.25-0.49: one sticker
        self.assertEqual(score_of([*boards[:3], far_only], stickers), 0.5)

    def test_layer1_and_stickers_only_scores_half(self):
        boards, stickers = stack()
        boards = [boards[0], B0[1], boards[2], B0[3]]  # the outer boards still where they started
        self.assertEqual(score_of(boards, stickers), 0.5)
        held = [boards[0], board_at(L.BOARD_X, L.LAYER_Y, L.LAYER2_BOTTOM + 0.005), boards[2], B0[3]]
        self.assertEqual(score_of(held, stickers), 0.5)  # a top board still held 5 mm above the stickers

    def test_layer1_only_scores_quarter(self):
        boards, _ = stack()
        lower_only = [boards[0], B0[1], boards[2], B0[3]]
        self.assertEqual(score_of(lower_only, K0), 0.25)
        _, stickers = stack(k_x=(L.BEARER_XS[1] + 0.015, L.BEARER_XS[0]))  # one sticker 15 mm off, no layer 2
        self.assertEqual(score_of(lower_only, stickers), 0.25)
        self.assertEqual(score_of(lower_only, [stack()[1][0], K0[1]]), 0.25)  # one sticker only

    def test_layer1_gap_limits(self):
        for half_gap, want in ((0.0351, 1.0), (0.0349, 0.0), (0.0649, 1.0), (0.0651, 0.0), (0.020, 0.0)):
            with self.subTest(centres_apart=2 * half_gap):
                self.assertEqual(score_of(*stack(l1_y=(half_gap, -half_gap))), want)
        self.assertEqual(score_of(*stack(l1_y=(0.12, 0.02))), 1.0)  # off to one side of the midline is fine

    def test_layer1_wrong_scores_zero(self):
        boards, stickers = stack()
        one = [boards[0], B0[1], B0[2], B0[3]]
        self.assertEqual(score_of(one, K0), 0.0)
        three = [boards[0], board_at(L.BOARD_X, 0.13, L.LAYER1_BOTTOM), boards[2], B0[3]]
        self.assertEqual(score_of(three, K0), 0.0)  # three boards on the bearers is not a two-board layer
        turned = [board_at(L.BOARD_X, 0.05, L.LAYER1_BOTTOM, 10), *boards[1:]]
        self.assertEqual(score_of(turned, stickers), 0.0)

    def test_a_piece_off_the_table_or_on_its_edge_scores_zero(self):
        boards, stickers = stack()
        fallen = [boards[0], board_at(0.3, 0.8, 0.0), boards[2], boards[3]]
        self.assertEqual(score_of(fallen, stickers), 0.0)
        self.assertEqual(evaluate(fallen, stickers)[1]["off_table"], [1])
        on_side = state((0.46, -0.2, L.TOP + L.STICKER_SIZE[0] / 2), L.STICKER_SIZE, about("y", 90))
        lower_only = [boards[0], B0[1], boards[2], B0[3]]
        self.assertEqual(score_of(lower_only, [stickers[0], on_side]), 0.0)
        board_up = state((L.BOARD_X, 0.3, L.TOP + L.BOARD_SIZE[1] / 2), L.BOARD_SIZE, about("x", 90))
        self.assertEqual(score_of([boards[0], board_up, boards[2], B0[3]], K0), 0.0)

    def test_success_needs_stillness(self):
        boards, stickers = stack()
        moving = [*boards[:3], board_at(L.BOARD_X, -L.LAYER_Y, L.LAYER2_BOTTOM, vel=(0, 0.02, 0))]
        r, _ = run(build() + [(moving, stickers)])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        spinning = [stickers[0], sticker_at(L.BEARER_XS[0], 0, L.STICKER_BOTTOM, spin=(0, 0, 0.8))]
        r, _ = run(build() + [(boards, spinning)])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r, _ = run(build() + [(boards, stickers)])
        self.assertTrue(r.success)

    def test_score_is_the_state_at_hand(self):
        # a finished stack knocked apart at the end scores what is left, not what was reached
        boards, stickers = stack()
        knocked = [boards[0], B0[1], boards[2], board_at(0.3, -0.4, L.TOP, 30)]
        r, t = run(build() + [(knocked, stickers)] * 5)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.best, 1.0)


class Exploits(unittest.TestCase):
    def test_stickers_straight_on_the_bearers_scores_zero(self):
        # stickers on the bearers, then the boards on the stickers: no board rests on the bearers
        stickers = [sticker_at(x, 0.0, L.BEARER_TOP) for x in (L.BEARER_XS[1], L.BEARER_XS[0])]
        mid = [board_at(L.BOARD_X, y, L.BEARER_TOP + KT) for y in (0.05, -0.05)]
        self.assertEqual(score_of([mid[0], B0[1], mid[1], B0[3]], stickers), 0.0)

    def test_two_layers_without_stickers_scores_quarter(self):
        lower = [board_at(L.BOARD_X, y, L.LAYER1_BOTTOM) for y in (0.05, -0.05)]
        upper = [board_at(L.BOARD_X, y, L.LAYER1_TOP) for y in (0.05, -0.05)]
        self.assertEqual(score_of([lower[0], upper[0], lower[1], upper[1]], K0), 0.25)

    def test_all_four_boards_in_one_layer_scores_zero(self):
        boards = [board_at(L.BOARD_X, y, L.LAYER1_BOTTOM) for y in (0.135, 0.045, -0.045, -0.135)]
        self.assertEqual(score_of(boards, K0), 0.0)

    def test_stickers_laid_along_the_boards_scores_quarter(self):
        boards, _ = stack()
        along = [sticker_at(L.BOARD_X, y, L.STICKER_BOTTOM, 90) for y in (0.05, -0.05)]
        self.assertFalse(laid_sticker(along[0]))
        self.assertEqual(score_of(boards, along), 0.25)

    def test_top_board_on_the_other_top_board_scores_half(self):
        boards, stickers = stack()
        piled = board_at(L.BOARD_X, L.LAYER_Y, L.LAYER2_TOP)
        self.assertEqual(score_of([boards[0], boards[1], boards[2], piled], stickers), 0.5)

    def test_both_top_boards_over_one_lower_board_scores_half(self):
        y = L.LAYER_Y
        self.assertEqual(score_of(*stack(l2_y=(y + 0.012, y - 0.012))), 0.5)

    def test_one_arm_side_alone_scores_quarter_at_most(self):
        # the left arm reaches only its own two boards and its sticker: layer 1 from both, its sticker on top
        boards = [board_at(L.BOARD_X, y, L.LAYER1_BOTTOM) for y in (0.05, -0.05)]
        one_side = [boards[0], boards[1], B0[2], B0[3]]
        self.assertEqual(score_of(one_side, [sticker_at(L.BEARER_XS[1], 0, L.STICKER_BOTTOM), K0[1]]), 0.25)
        # its sticker and its second board on top: no second sticker, so the top board rests on one only
        top = board_at(L.BOARD_X, 0.05, L.LAYER2_BOTTOM)
        self.assertEqual(score_of([boards[0], top, B0[2], boards[1]], [stack()[1][0], K0[1]]), 0.25)

    def test_board_across_the_bearers_turned_45_scores_zero(self):
        boards, stickers = stack()
        skew = board_at(L.BOARD_X, 0.05, L.LAYER1_BOTTOM, 45)
        self.assertEqual(score_of([skew, *boards[1:]], stickers), 0.0)

    def test_held_within_3mm_counts_as_set_down(self):
        # the scorer sees no gripper: a top board held still 2 mm above the stickers counts (fidelity notes)
        boards, stickers = stack()
        hovering = board_at(L.BOARD_X, -L.LAYER_Y, L.LAYER2_BOTTOM + 0.002)
        self.assertEqual(score_of([*boards[:3], hovering], stickers), 1.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for frame in build():
            t.update(*frame)
        self.assertEqual(t.best, 1.0)
        r = t.update(B0, K0)  # every piece jumps back to its mark in one step: a new episode
        self.assertEqual(t.best, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_back_to_jittered_marks_resets(self):
        t = Tracker()
        for frame in build():
            t.update(*frame)
        rng = np.random.default_rng(1)
        boards = [board_at(b.pos[0] + rng.uniform(-0.03, 0.03), b.pos[1] + rng.uniform(-0.03, 0.03), L.TOP)
                  for b in B0]
        stickers = [sticker_at(k.pos[0] + rng.uniform(-0.03, 0.03), k.pos[1] + rng.uniform(-0.03, 0.03), L.TOP, 90.5)
                    for k in K0]
        t.update(boards, stickers)
        self.assertEqual(t.best, 0.0)

    def test_putting_things_back_gradually_does_not_reset(self):
        frames = build()
        frames += glide(frames[-1], (B0, K0), 40) + [(B0, K0)] * 10
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)  # back at the start, but it got there gradually: the same episode
        self.assertEqual(t.steps, len(frames))

    def test_a_jump_elsewhere_does_not_reset(self):
        frames = build()
        scattered = ([board_at(0.3, y, L.TOP, 20) for y in (0.5, 0.4, -0.4, -0.5)],
                     [sticker_at(0.6, y, L.TOP) for y in (0.2, -0.2)])
        r, t = run(frames + [scattered] * 3)
        self.assertEqual(t.best, 1.0)
        self.assertEqual(r.score, 0.0)

    def test_a_board_falling_back_onto_its_mark_is_not_a_reset(self):
        # one board dropped from about 20 cm: 10 cm above its mark it falls at 1.4 m/s, so the last step's 10 cm
        # (more than RESET_JUMP plus a free fall from rest) fits its velocity; every other piece is already back
        frames = build()
        frames += glide(frames[-1], (B0, K0), 30)
        x0, y0 = B0[0].pos[:2]
        falling = board_at(x0, y0, L.TOP + 0.10, vel=(0, 0, -1.4))
        _, t = run(frames + [([falling, *B0[1:]], K0), (B0, K0)])
        self.assertEqual(t.best, 1.0)
        # the same 10 cm jump from a board that was still is a teleport: a new episode
        resting = board_at(x0, y0, L.TOP + 0.10)
        _, t = run(frames + [([resting, *B0[1:]], K0), (B0, K0)])
        self.assertEqual(t.best, 0.0)


class OddStates(unittest.TestCase):
    """Whatever the simulator reports, the scorer returns one of the five scores and JSON-clean details."""

    def odd(self):
        nan, inf = float("nan"), float("inf")
        boards, stickers = stack()
        return [
            ([board_at(nan, 0.05, L.LAYER1_BOTTOM), *boards[1:]], stickers),
            (boards, [stickers[0], state((0.2, 0, L.TOP + 0.04), L.STICKER_SIZE, quat=(nan, 0, 0, nan))]),
            ([*boards[:3], board_at(L.BOARD_X, -0.05, L.LAYER2_BOTTOM, vel=(0, inf, 0))], stickers),
            ([state((0.28, 0.05, L.TOP + 0.03), L.BOARD_SIZE, quat=(0, 0, 0, 0)), *boards[1:]], stickers),
            ([SimpleNamespace(pos=np.zeros(3)), *boards[1:]], stickers),  # attributes missing
        ]

    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        for frame in self.odd():
            with self.subTest():
                t = Tracker()
                for f in build():
                    t.update(*f)
                prev, first = t.prev, dict(t.first_at)
                r = t.update(*frame)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.success)
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.bad_steps, 1)
                self.assertIs(t.prev, prev)
                self.assertEqual(t.first_at, first)
                json.dumps(r.details, allow_nan=False)
                r = t.update(*stack())  # the next good step scores the stack again
                self.assertEqual(r.score, 1.0)

    def test_far_and_strange_states(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            boards = [state(rng.uniform(-2, 2, 3), L.BOARD_SIZE, tuple(rng.normal(size=4))) for _ in range(4)]
            stickers = [state(rng.uniform(-2, 2, 3), L.STICKER_SIZE, tuple(rng.normal(size=4))) for _ in range(2)]
            t = Tracker()
            r = t.update(boards, stickers)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details, allow_nan=False)


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

    def objects(self, boards, stickers):
        out = dict(zip(self.task.BOARD_NAMES, boards))
        out.update(zip(self.task.STICKER_NAMES, stickers))
        return out

    def test_objects_match_the_layout(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        for spec, b in zip(self.task.BOARDS, B0):
            self.assertTrue(np.allclose(spec.pos, b.pos))
        for spec, k in zip(self.task.STICKERS, K0):
            self.assertTrue(np.allclose(spec.pos, k.pos))
        fixed = [o.name for o in self.task.OBJECTS if o.mass is None]
        self.assertEqual(fixed, ["bearer_near", "bearer_far"])

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(B0, K0)))
                self.assertEqual(self.task.score(self.objects(B0, K0)), 0.0)
                self.assertTrue(self.task.success(self.objects(*stack())))
                self.assertEqual(self.task.score(self.objects(*stack())), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        for key in ("step", "score", "success", "layer1", "layer1_gap_mm", "sticker_off_mm", "layer2_off_mm",
                    "bad_steps", "first_step", "board_left_inner", "sticker_right"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["score"], 1.0)

    def test_trace_survives_a_bad_path(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(self.objects(B0, K0)))
        finally:
            os.environ.pop("TASK_TRACE", None)

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_takes_turns_at_the_stack(self):
        """No arm comes over the stack until the other has let go of its piece there and gone back up."""
        plan = self.task.demo()

        def times(wps, label_start):
            out, t = [], 0.0
            for w in wps:
                if w.label.startswith(label_start):
                    out.append((t, t + w.duration, w))
                t += w.duration
            return out

        carries = sorted([(s, "left") for s, _, _ in times(plan["left"], "carry")] +
                         [(s, "right") for s, _, _ in times(plan["right"], "carry")])
        ups = {arm: [e for _, e, w in times(plan[arm], "up")] for arm in plan}
        self.assertEqual([a for _, a in carries], ["left", "right", "left", "right", "left", "right"])
        for (s_prev, a_prev), (s, a) in zip(carries, carries[1:]):
            done = min(e for e in ups[a_prev] if e > s_prev)  # the previous arm's next 'up' (it let go before)
            self.assertGreaterEqual(s + 1e-9, done, f"{a} carries in at {s} before {a_prev} is up at {done}")
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400)  # inside the run limit with the policy's 3 s hold

    def test_instruction_is_the_same_everywhere(self):
        """task.py, task.json and both README commands give the model the same instruction."""
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_success_hold_outlasts_every_release(self):
        """A piece counts as resting while still gripped (within 3 mm), so the success hold must be longer than
        the settle and the release after each set-down: the trial cannot end with the fingers closed on it."""
        plan = self.task.demo()
        for arm, wps in plan.items():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly set"):
                    settle, let_go = wps[i + 1], wps[i + 2]
                    self.assertTrue(settle.label == "settle" and let_go.label.startswith("let go"), arm)
                    self.assertLessEqual(settle.duration + let_go.duration + 0.5, self.task.SUCCESS_HOLD_S)
                    self.assertGreaterEqual(let_go.duration, 1.5)  # the gripper's rate limit needs 1.5 s


if __name__ == "__main__":
    unittest.main()
