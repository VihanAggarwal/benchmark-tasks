"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic board states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
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
    Tracker, arms_valid, axis_err_deg, bearing_deg, bottom, centreline_offset, footprint, gap, plan_distance,
    sector, standing_pose, tcp_clearance, tilt_deg, valid,
)

HZ = 20.0
TC = np.array([L.TRAP_X, L.TRAP_Y])
HALF_LEN = L.BOARD_SIZE[0] / 2
GOOD_R = 0.105  # board centre 105 mm out: inner end face 40 mm out, gap 5 mm
POSSIBLE = {0.0, 0.25, 0.5, 0.75, 1.0}
PAD_THICK, PAD_HALF_WIDTH = 0.008, 0.010  # the finger pads: 8 x 20 x 38 mm boxes beside the gap


# -- synthetic states -------------------------------------------------------------------------------------------
def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def quat(yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0):
    """Turned yaw about the vertical after rolling about the board's long (x) axis and pitching about its y axis."""
    def axis(i, deg):
        h = math.radians(deg) / 2
        v = np.zeros(4)
        v[i], v[3] = math.sin(h), math.cos(h)
        return v
    return qmul(axis(2, yaw_deg), qmul(axis(0, roll_deg), axis(1, pitch_deg)))


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def board(x, y, yaw=0.0, roll=0.0, pitch=0.0, low=0.0, vel=(0, 0, 0)):
    """A board state: centre (x, y), lowest corner `low` above the table, as the simulator reports it."""
    q = quat(yaw, roll, pitch)
    half = np.array(L.BOARD_HALF)
    cs = np.array([rot(q) @ (half * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c = np.array([x, y, L.TOP + low - cs[:, 2].min()])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=c + cs.min(axis=0), high=c + cs.max(axis=0))


def on_line(k, r=GOOD_R, lateral=0.0, turn=0.0, **kw):
    """A board along line k (1..4): centre r from the trap's centre, moved `lateral` to the left of the line,
    turned `turn` deg from it."""
    b = math.radians(L.PEG_BEARINGS_DEG[k - 1])
    u, n = np.array([math.cos(b), math.sin(b)]), np.array([-math.sin(b), math.cos(b)])
    c = TC + r * u + lateral * n
    return board(c[0], c[1], yaw=L.PEG_BEARINGS_DEG[k - 1] + turn, **kw)


def pivot(k, turn, end=GOOD_R - HALF_LEN, flip=False, **kw):
    """A board along line k turned `turn` deg about the middle of its inner end face (`end` from the trap's
    centre), as when it is swung about the end at the rim; `flip` turns it end for end as well."""
    b = math.radians(L.PEG_BEARINGS_DEG[k - 1])
    a = b + math.radians(turn)
    c = TC + end * np.array([math.cos(b), math.sin(b)]) + HALF_LEN * np.array([math.cos(a), math.sin(a)])
    return board(c[0], c[1], yaw=math.degrees(a) + (180.0 if flip else 0.0), **kw)


def at_start(name):
    return board(*L.BOARD_STARTS[name])


START = {n: at_start(n) for n in L.BOARDS}
GOOD = {f"board_{k}": on_line(k) for k in range(1, 5)}


def layout(**changes):
    """The four good boards with some replaced."""
    out = dict(GOOD)
    out.update(changes)
    return out


def arm(x, y, z, opening=1.0):
    p = np.array([x, y, L.TOP + z])
    return rigstate.Arm(tcp=p, tips=p - [0, 0, 0.013], opening=opening)


PARKED = {"left": arm(0.22, 0.30, 0.16), "right": arm(0.22, -0.30, 0.16)}


def arms_with(side, a):
    out = dict(PARKED)
    out[side] = a
    return out


def run(frames, t=None, arms=PARKED):
    """Feed frames (dicts of board states) to a tracker (fresh unless given); return the last result and it."""
    t = Tracker(HZ) if t is None else t
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


def settled(frame, arms=PARKED, n=12, t=None):
    """The frame held still long enough for 'at rest' (0.5 s)."""
    return run([frame] * n, t, arms)


def glide(a, b, n):
    """n frames moving every board from frame a to frame b in a straight line (same orientation as b)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        fr = {}
        for name in L.BOARDS:
            p = (1 - f) * np.asarray(a[name].pos) + f * np.asarray(b[name].pos)
            s = b[name]
            fr[name] = SimpleNamespace(pos=p, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel,
                                       low=s.low - s.pos + p, high=s.high - s.pos + p)
        out.append(fr)
    return out


# -- geometry ---------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_layout_numbers(self):
        self.assertAlmostEqual(L.PEGS[0][1], 0.4114, places=4)
        self.assertAlmostEqual(L.PEGS[0][2], 0.1414, places=4)
        self.assertAlmostEqual(L.PEGS[2][1], 0.1286, places=4)
        self.assertAlmostEqual(L.PEGS[3][2], -0.1414, places=4)
        vol = L.BOARD_SIZE[0] * L.BOARD_SIZE[1] * L.BOARD_SIZE[2]
        self.assertAlmostEqual(L.BOARD_MASS / vol, 500.0, delta=1.0)  # pine
        # boards in a pair 60 mm apart face to face
        self.assertAlmostEqual(L.BOARD_STARTS["board_2"][1] - L.BOARD_STARTS["board_1"][1] - L.BOARD_SIZE[1], 0.060)

    def test_good_board_numbers(self):
        s = GOOD["board_1"]
        self.assertAlmostEqual(gap(s), 0.005, places=6)
        inner_corner = math.hypot(GOOD_R - HALF_LEN, L.BOARD_HALF[1])
        self.assertAlmostEqual(inner_corner, 0.0412, places=4)  # 6 mm outside the rim
        outer_end = GOOD_R + HALF_LEN
        self.assertAlmostEqual(L.PEG_DIST - L.PEG_RADIUS - outer_end, 0.024)  # 24 mm short of the peg
        a, b = footprint(GOOD["board_1"]), footprint(GOOD["board_2"])
        d = min(plan_distance(p, b) for p in a)
        self.assertAlmostEqual(d, 0.0424, delta=0.0005)  # adjacent inner ends 42 mm apart
        self.assertEqual(sector(s), 0)
        self.assertAlmostEqual(axis_err_deg(s, 0), 0.0, places=6)
        self.assertAlmostEqual(centreline_offset(s), 0.0, places=9)
        self.assertTrue(standing_pose(s))

    def test_footprint_is_the_whole_outline(self):
        self.assertEqual(footprint(GOOD["board_1"]).shape, (4, 2))  # standing: the 130 x 20 mm edge
        lean = footprint(board(0.30, 0.20, roll=8.0))  # leaning sideways 8 deg: a wider rectangle
        self.assertEqual(lean.shape, (4, 2))
        self.assertAlmostEqual(np.ptp(lean[:, 1]), 0.020 * math.cos(math.radians(8)) + 0.050 * math.sin(math.radians(8)))
        self.assertEqual(footprint(board(0.30, 0.20, roll=8.0, pitch=8.0)).shape, (6, 2))  # tipped two ways
        # lying flat on its 130 x 50 face: a 130 x 50 outline
        flat = footprint(board(0.30, 0.20, roll=90.0))
        self.assertAlmostEqual(np.ptp(flat[:, 0]), 0.130, places=6)
        self.assertAlmostEqual(np.ptp(flat[:, 1]), 0.050, places=6)

    def test_plan_distance(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        self.assertEqual(plan_distance((0.5, 0.5), sq), 0.0)
        self.assertEqual(plan_distance((1.0, 0.5), sq), 0.0)
        self.assertAlmostEqual(plan_distance((2.0, 0.5), sq), 1.0)
        self.assertAlmostEqual(plan_distance((2.0, 2.0), sq), math.sqrt(2))

    def test_gap_is_to_the_nearest_point_not_the_end_face_centre(self):
        # moved 8 mm sideways with its end face 34.5 mm out: the face's middle is 35.4 mm from the centre, but the
        # face itself passes 34.5 mm from it: the outline is 0.5 mm inside the rim
        s = on_line(1, r=0.0345 + HALF_LEN, lateral=0.008)
        self.assertAlmostEqual(gap(s), -0.0005, places=6)
        # turned 8 deg about its middle: the nearer inner corner is what counts
        t = on_line(1, turn=8.0)
        self.assertLess(gap(t), gap(GOOD["board_1"]))
        # a board over the trap's centre: -35 mm
        self.assertAlmostEqual(gap(board(L.TRAP_X, L.TRAP_Y, yaw=30.0)), -L.TRAP_RADIUS)

    def test_sectors(self):
        for deg, want in ((1, 0), (44.9, 0), (89.9, 0), (90.1, 1), (179.9, 1), (180.1, 2), (269.9, 2), (270.1, 3),
                          (359.9, 3)):
            c = TC + 0.1 * np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
            s = board(c[0], c[1], yaw=deg)
            self.assertEqual(sector(s), want, deg)
            self.assertAlmostEqual(bearing_deg(s), deg % 360, places=6)
        self.assertEqual(sector(on_line(3)), 2)
        self.assertEqual(sector(on_line(4)), 3)

    def test_axis_either_end_and_collinear_lines(self):
        s = on_line(3)  # along 225 deg
        self.assertAlmostEqual(axis_err_deg(s, 2), 0.0, places=6)
        self.assertAlmostEqual(axis_err_deg(s, 0), 0.0, places=6)  # the same line as line 1: hence the sectors
        self.assertAlmostEqual(axis_err_deg(on_line(1, turn=180.0), 0), 0.0, places=6)  # turned end for end
        self.assertAlmostEqual(axis_err_deg(on_line(1, turn=-7.0), 0), 7.0, places=6)
        self.assertAlmostEqual(axis_err_deg(on_line(1, turn=90.0), 0), 90.0, places=6)  # tangential

    def test_tilt_and_bottom(self):
        self.assertAlmostEqual(tilt_deg(on_line(1, roll=9.0)), 9.0, places=4)
        self.assertAlmostEqual(tilt_deg(on_line(1, roll=180.0)), 0.0, places=4)  # the other long edge down
        self.assertAlmostEqual(tilt_deg(board(0.3, 0.2, roll=90.0)), 90.0, places=4)  # flat
        self.assertAlmostEqual(tilt_deg(board(0.3, 0.2, pitch=90.0)), 90.0, places=4)  # on its end
        self.assertAlmostEqual(bottom(on_line(1, low=0.002)), 0.002, places=9)
        self.assertTrue(standing_pose(on_line(1, roll=180.0)))
        self.assertFalse(standing_pose(board(0.3, 0.2, pitch=90.0)))

    def test_tcp_clearance(self):
        s = GOOD["board_1"]
        foot = footprint(s)
        self.assertEqual(tcp_clearance(arms_with("left", arm(*s.pos[:2], 0.035)), s), 0.0)
        self.assertTrue(math.isinf(tcp_clearance(arms_with("left", arm(*s.pos[:2], 0.095)), s)))  # high: ignored
        side = np.array([-math.sin(math.pi / 4), math.cos(math.pi / 4)])  # across the board, to its left
        p = s.pos[:2] + side * (L.BOARD_HALF[1] + 0.025)
        self.assertAlmostEqual(tcp_clearance(arms_with("left", arm(*p, 0.05)), s), 0.025, places=6)
        self.assertAlmostEqual(plan_distance(p, foot), 0.025, places=6)


# -- the score tiers --------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = settled(START)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["lines_fenced"], [])

    def test_all_four_scores_one(self):
        r, _ = settled(GOOD)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["lines_fenced"], [1, 2, 3, 4])
        for b in r.details["boards"].values():
            self.assertAlmostEqual(b["gap_mm"], 5.0, places=3)

    def test_one_line_quarter(self):
        r, _ = settled(dict(START, board_1=GOOD["board_1"]))
        self.assertEqual(r.score, 0.25)

    def test_one_arms_two_boards_score_half(self):
        r, _ = settled(dict(START, board_1=GOOD["board_1"], board_2=GOOD["board_2"]))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)

    def test_lines_3_and_4_do_not_also_fence_the_collinear_lines(self):
        r, _ = settled(dict(START, board_3=GOOD["board_3"], board_4=GOOD["board_4"]))
        self.assertEqual(r.details["lines_fenced"], [3, 4])
        self.assertEqual(r.score, 0.5)

    def test_three_good_and_one_turned_12_deg(self):
        r, _ = settled(layout(board_4=on_line(4, turn=12.0)))
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["lines_fenced"], [1, 2, 3])

    def test_gap_15_mm_scores_zero(self):
        r, _ = settled({f"board_{k}": on_line(k, r=GOOD_R + 0.010) for k in range(1, 5)})
        self.assertEqual(r.score, 0.0)

    def test_one_end_3_mm_over_the_rim_scores_zero(self):
        r, _ = settled(layout(board_2=on_line(2, r=GOOD_R - 0.008)))  # gap -3 mm
        self.assertAlmostEqual(r.details["boards"]["board_2"]["gap_mm"], -3.0, places=3)
        self.assertTrue(r.details["zero_rule"])
        self.assertEqual(r.score, 0.0)

    def test_tangential_boards_score_zero(self):
        frame = {f"board_{k}": on_line(k, r=0.050, turn=90.0) for k in range(1, 5)}  # round the rim, 5 mm gap
        r, _ = settled(frame)
        for b in r.details["boards"].values():
            self.assertAlmostEqual(b["gap_mm"], 5.0, places=3)
        self.assertEqual(r.score, 0.0)

    def test_board_lying_flat_on_its_line_is_not_fenced(self):
        flat = on_line(1, roll=90.0)
        r, _ = settled(layout(board_1=flat))
        self.assertGreaterEqual(r.details["boards"]["board_1"]["gap_mm"], -1.0)
        self.assertEqual(r.details["lines_fenced"], [2, 3, 4])
        self.assertEqual(r.score, 0.75)

    def test_board_across_the_trap_scores_zero(self):
        r, _ = settled(layout(board_1=board(L.TRAP_X, L.TRAP_Y + 0.05, yaw=90.0)))  # through the trap, flat on it
        self.assertTrue(r.details["zero_rule"])
        self.assertEqual(r.score, 0.0)

    def test_two_boards_in_one_sector_count_once(self):
        # side by side on line 1, 9.8 mm either side of it (a 0.4 mm overlap no simulator would allow: the count
        # is what is tested)
        two = dict(START, board_1=on_line(1, lateral=-0.0098), board_2=on_line(1, lateral=0.0098))
        r, _ = settled(two)
        self.assertTrue(r.details["boards"]["board_1"]["fenced"] and r.details["boards"]["board_2"]["fenced"])
        self.assertEqual(r.details["lines_fenced"], [1])
        self.assertEqual(r.score, 0.25)

    def test_edge_tolerances(self):
        cases = [
            (pivot(1, 7.9), True), (pivot(1, 8.1), False),  # swung about the inner end: the 8 deg limit
            (pivot(1, -7.9), True), (pivot(1, 7.9, flip=True), True), (pivot(1, -8.1, flip=True), False),
            (on_line(1, turn=5.4), True), (on_line(1, turn=5.6), False),  # turned about its middle: the 10 mm
            (on_line(1, lateral=0.0099), True), (on_line(1, lateral=0.0101), False),
            (on_line(1, r=GOOD_R + 0.0049), True), (on_line(1, r=GOOD_R + 0.0051), False),  # gap 9.9 / 10.1
            (on_line(1, r=GOOD_R - 0.0059), True),  # gap -0.9 mm: touching the disc's edge
            (on_line(1, roll=9.9), True), (on_line(1, roll=10.1), False),
            (on_line(1, low=0.0029), True), (on_line(1, low=0.0031), False),
            (on_line(1, low=-0.0029), True),
        ]
        for s, want in cases:
            r, _ = settled(layout(board_1=s))
            got = r.details["boards"]["board_1"]["fenced"]
            self.assertEqual(got, want, r.details["boards"]["board_1"])
            self.assertEqual(r.score, 1.0 if want else 0.75)

    def test_zero_rule_edge(self):
        r, _ = settled(layout(board_1=on_line(1, r=GOOD_R - 0.0061)))  # gap -1.1 mm
        self.assertEqual(r.score, 0.0)
        r, _ = settled(layout(board_1=on_line(1, r=GOOD_R - 0.0059)))  # gap -0.9 mm
        self.assertEqual(r.score, 1.0)

    def test_score_is_idempotent(self):
        r, t = settled(GOOD)
        self.assertEqual(t.result(GOOD, PARKED).score, r.score)
        self.assertEqual(t.result(GOOD, PARKED).score, 1.0)


# -- standing still -------------------------------------------------------------------------------------------
class AtRest(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        r, t = run([GOOD] * 10)
        self.assertEqual(r.score, 0.0)  # only 0.45 s recorded
        r = t.update(GOOD, PARKED)
        self.assertEqual(r.score, 1.0)  # the 11th step: the current one and the 10 before (0.5 s)

    def test_a_creeping_board_is_not_at_rest(self):
        frames = []
        for i in range(30):  # sliding 0.3 mm per step along its line, 6 mm/s
            frames.append(layout(board_1=on_line(1, r=GOOD_R + 0.0003 * i)))
        r, t = run(frames[:15])
        self.assertFalse(r.details["boards"]["board_1"]["at_rest"])
        self.assertEqual(r.score, 0.75)
        r, _ = settled(frames[14], t=t)
        self.assertEqual(r.score, 1.0)

    def test_jitter_under_2_mm_is_at_rest(self):
        rng = random.Random(3)
        frames = [layout(board_1=on_line(1, r=GOOD_R + rng.uniform(-0.0009, 0.0009),
                                         lateral=rng.uniform(-0.0009, 0.0009))) for _ in range(40)]
        r, _ = run(frames)
        self.assertTrue(r.details["boards"]["board_1"]["at_rest"])
        self.assertEqual(r.score, 1.0)

    def test_velocity_is_never_read(self):
        chatter = {n: on_line(k, vel=(0.04, -0.03, 0.02)) for k, n in enumerate(L.BOARDS, 1)}  # 45 mm/s reported
        r, _ = settled(chatter)
        self.assertEqual(r.score, 1.0)

    def test_board_that_falls_over_after_release(self):
        r, t = settled(GOOD)
        self.assertEqual(r.score, 1.0)
        r, _ = settled(layout(board_3=on_line(3, roll=90.0)), t=t)
        self.assertEqual(r.score, 0.75)


# -- 'then let go of every board and lift the grippers clear' ---------------------------------------------------
class Free(unittest.TestCase):
    def test_a_gripped_board_does_not_count(self):
        s = GOOD["board_1"]
        holding = arms_with("left", arm(*s.pos[:2], 0.035, opening=0.13))
        r, _ = settled(GOOD, arms=holding)
        self.assertFalse(r.details["boards"]["board_1"]["free"])
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_released_and_lifted_counts(self):
        s = GOOD["board_1"]
        _, t = settled(GOOD, arms=arms_with("left", arm(*s.pos[:2], 0.035, opening=0.13)))
        r = t.update(GOOD, arms_with("left", arm(*s.pos[:2], 0.089, opening=1.0)))
        self.assertEqual(r.score, 0.75)  # open but still under 90 mm, over the board
        r = t.update(GOOD, arms_with("left", arm(*s.pos[:2], 0.0905, opening=1.0)))
        self.assertEqual(r.score, 1.0)  # 90 mm up: clear

    def test_moved_60_mm_away_counts(self):
        s = GOOD["board_1"]
        side = np.array([-math.sin(math.pi / 4), math.cos(math.pi / 4)])
        for d, want in ((0.059, 0.75), (0.061, 1.0)):
            p = s.pos[:2] + side * (L.BOARD_HALF[1] + d)
            r, _ = settled(GOOD, arms=arms_with("left", arm(*p, 0.035)))
            self.assertEqual(r.score, want, d)

    def test_open_finger_resting_on_a_board_is_not_free(self):
        # the review's attack: fingers fully open (94 mm), TCP 35 mm up, one finger pressed on board 1's face
        s = GOOD["board_1"]
        side = np.array([-math.sin(math.pi / 4), math.cos(math.pi / 4)])  # across the board, to its left
        along = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
        half_gap = -rigstate.FINGER_OPEN  # 47 mm from the TCP to a pad's inner face
        for name, p in (
                ("inner pad face on the face", s.pos[:2] + side * (L.BOARD_HALF[1] + half_gap)),
                ("outer pad face on the face", s.pos[:2] + side * (L.BOARD_HALF[1] + half_gap + PAD_THICK)),
                ("pad corner on the outer end's corner",  # the pad's far corner touching the board's corner
                 s.pos[:2] + side * (L.BOARD_HALF[1] + half_gap + PAD_THICK) + along * (HALF_LEN + PAD_HALF_WIDTH))):
            r, _ = settled(GOOD, arms=arms_with("left", arm(*p, 0.035, opening=1.0)))
            self.assertFalse(r.details["boards"]["board_1"]["free"], name)
            self.assertLess(r.details["boards"]["board_1"]["tcp_clear_mm"], L.FREE_RADIUS * 1000, name)
            self.assertEqual(r.score, 0.75, name)

    def test_free_radius_covers_an_open_pad(self):
        # the farthest point of an open pad, in plan, from its TCP: 47 mm half gap + 8 mm pad across, 10 mm along
        reach = math.hypot(-rigstate.FINGER_OPEN + PAD_THICK, PAD_HALF_WIDTH)
        self.assertAlmostEqual(reach, 0.0559, places=4)
        self.assertGreater(L.FREE_RADIUS, reach + 0.003)

    def test_the_other_arm_counts_too(self):
        s = GOOD["board_1"]
        r, _ = settled(GOOD, arms=arms_with("right", arm(*s.pos[:2], 0.06)))
        self.assertEqual(r.score, 0.75)

    def test_unreadable_arms_count_as_free_and_are_counted(self):
        r, t = settled(GOOD, arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)
        self.assertFalse(r.details["arms_available"])
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0), "right": PARKED["right"]}
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": PARKED["left"]}))
        r, t = settled(GOOD, arms=bad)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 12)

    def test_offline_feed_reaches_the_scorer(self):
        rigstate.offline_reset()
        rigstate.offline_feed({"left": {"tcp": [*GOOD["board_1"].pos[:2], L.TOP + 0.04], "tips": [0, 0, 0],
                                        "opening": 0.1},
                               "right": {"tcp": [0.22, -0.30, L.TOP + 0.16], "tips": [0, 0, 0], "opening": 1.0}})
        r, _ = settled(GOOD, arms=rigstate.arms())
        self.assertEqual(r.score, 0.75)
        rigstate.offline_reset()


# -- the zero rule --------------------------------------------------------------------------------------------
class ZeroRule(unittest.TestCase):
    def test_held_board_over_the_opening_scores_zero(self):
        # three good boards and the fourth held 40 mm up over the trap's mouth at the end
        held = on_line(4, r=0.050, low=0.040)
        r, _ = settled(layout(board_4=held), arms=arms_with("right", arm(*held.pos[:2], 0.075, opening=0.13)))
        self.assertEqual(r.score, 0.0)

    def test_board_riding_on_the_disc_scores_zero(self):
        ride = on_line(1, r=GOOD_R - 0.015, low=L.TRAP_HEIGHT)  # its end up on the 3 mm disc
        r, _ = settled(layout(board_1=ride))
        self.assertTrue(r.details["zero_rule"])
        self.assertEqual(r.score, 0.0)

    def test_zero_rule_overrides_every_line(self):
        r, _ = settled(layout(board_1=on_line(1, r=GOOD_R - 0.0065)))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_lying_board_over_the_rim_also_zero(self):
        r, _ = settled(dict(START, board_2=on_line(2, r=0.070, roll=90.0)))  # flat, end 5 mm over the opening
        self.assertEqual(r.score, 0.0)

    def test_pulled_back_off_the_rim_recovers(self):
        r, t = settled(layout(board_1=on_line(1, r=GOOD_R - 0.010)))
        self.assertEqual(r.score, 0.0)  # scored at the step: no latch
        r, _ = settled(GOOD, t=t)
        self.assertEqual(r.score, 1.0)


# -- unintended strategies ------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_one_board_cannot_fence_two_collinear_lines(self):
        r, _ = settled(dict(START, board_3=GOOD["board_3"]))
        self.assertEqual(r.details["lines_fenced"], [3])  # not line 1, which runs on through the trap
        self.assertEqual(r.score, 0.25)

    def test_one_arm_lines_1_and_3(self):
        r, _ = settled(dict(START, board_1=GOOD["board_1"], board_2=GOOD["board_3"]))  # left arm, its two boards
        self.assertEqual(r.details["lines_fenced"], [1, 3])
        self.assertEqual(r.score, 0.5)

    def test_one_long_run_through_the_trap_scores_zero(self):
        # boards end to end along lines 1 and 3 through the trap: the joint is over the opening
        a = on_line(1, r=HALF_LEN - 0.002)
        b = on_line(3, r=HALF_LEN + 0.002)
        r, _ = settled(dict(START, board_1=a, board_3=b))
        self.assertEqual(r.score, 0.0)

    def test_board_on_its_end_on_the_line(self):
        r, _ = settled(layout(board_1=on_line(1, r=0.066, pitch=90.0)))  # 130 mm upright, 20 x 50 end down
        self.assertAlmostEqual(r.details["boards"]["board_1"]["gap_mm"], 6.0, places=3)
        self.assertFalse(r.details["boards"]["board_1"]["fenced"])
        self.assertEqual(r.score, 0.75)

    def test_board_upside_down_still_stands_on_a_long_edge(self):
        r, _ = settled(layout(board_1=on_line(1, roll=180.0)))
        self.assertEqual(r.score, 1.0)

    def test_board_in_a_sector_aimed_at_the_neighbouring_peg(self):
        # centre in sector 1 but laid along the 135 deg direction: neither line's
        s = board(*(TC + 0.105 * np.array([math.cos(math.radians(80)), math.sin(math.radians(80))])), yaw=135.0)
        r, _ = settled(layout(board_1=s))
        self.assertEqual(r.details["boards"]["board_1"]["line"], 1)
        self.assertFalse(r.details["boards"]["board_1"]["fenced"])

    def test_board_between_two_pegs_aimed_at_the_trap(self):
        c = TC + 0.105 * np.array([0.0, 1.0])
        s = board(c[0], c[1], yaw=90.0)
        r, _ = settled(layout(board_1=s))
        self.assertFalse(r.details["boards"]["board_1"]["fenced"])
        self.assertEqual(r.score, 0.75)

    def test_stacked_boards(self):
        top = on_line(1, low=L.BOARD_SIZE[2])  # standing on top of the board on line 1
        r, _ = settled(dict(START, board_1=GOOD["board_1"], board_2=top))
        self.assertFalse(r.details["boards"]["board_2"]["fenced"])
        self.assertEqual(r.score, 0.25)

    def test_board_beyond_the_peg_or_short_of_the_rim(self):
        r, _ = settled(layout(board_1=on_line(1, r=0.270)))  # outside the peg
        self.assertEqual(r.score, 0.75)
        r, _ = settled(layout(board_1=on_line(1, r=0.120)))  # gap 20 mm
        self.assertEqual(r.score, 0.75)

    def test_dropped_from_height_lands_tipped(self):
        r, _ = settled(layout(board_1=on_line(1, roll=35.0, low=0.0)))
        self.assertEqual(r.score, 0.75)

    def test_hovering_gripper_over_finished_fence(self):
        # every board placed but a gripper left hovering 60 mm up over one: 'lift the grippers clear'
        s = GOOD["board_2"]
        r, _ = settled(GOOD, arms=arms_with("left", arm(*s.pos[:2], 0.060)))
        self.assertEqual(r.score, 0.75)

    def test_pushed_into_place_is_scored_by_the_end_state(self):
        frames = glide(layout(board_1=on_line(1, r=0.140)), GOOD, 20)
        r, _ = run(frames + [GOOD] * 12)
        self.assertEqual(r.score, 1.0)

    def test_random_states_never_crash_and_score_a_tier(self):
        rng = random.Random(7)
        t = Tracker(HZ)
        for _ in range(400):
            fr = {n: board(rng.uniform(0.0, 0.6), rng.uniform(-0.6, 0.6), yaw=rng.uniform(-180, 180),
                           roll=rng.choice([0, 0, 5, 90, 180]), pitch=rng.choice([0, 0, 90]), low=rng.uniform(0, 0.1))
                  for n in L.BOARDS}
            r = t.update(fr, PARKED if rng.random() < 0.7 else None)
            self.assertIn(r.score, POSSIBLE)


# -- blow-ups and resets --------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        _, t = settled(GOOD)
        steps, hist, unavailable = t.steps, {n: list(h) for n, h in t.history.items()}, t.arms_unavailable
        nan = on_line(1)
        nan.pos = np.array([np.nan, 0.0, 0.0])
        zero_q = on_line(2)
        zero_q.quat_xyzw = np.zeros(4)
        inf = on_line(3)
        inf.quat_xyzw = np.array([0, 0, np.inf, 1.0])
        for bad in (layout(board_1=nan), layout(board_2=zero_q), layout(board_3=inf),
                    {n: GOOD[n] for n in L.BOARDS[:3]}, layout(board_4=SimpleNamespace(pos=[0, 0]))):
            r = t.update(bad, None)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.steps, steps + 5)
        self.assertEqual(t.arms_unavailable, unavailable)
        for n in L.BOARDS:
            self.assertEqual(len(t.history[n]), len(hist[n]))
        self.assertFalse(valid(nan))
        self.assertFalse(valid(zero_q))
        r = t.update(GOOD, PARKED)  # back to normal: the history carried on
        self.assertEqual(r.score, 1.0)

    def test_teleport_back_to_start_resets(self):
        _, t = settled(GOOD, arms=None)
        self.assertGreater(t.arms_unavailable, 0)
        t.update(START, PARKED)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertEqual(len(t.history["board_1"]), 1)

    def test_reset_with_jittered_starts(self):
        _, t = settled(GOOD)
        jit = {n: board(L.BOARD_STARTS[n][0] + 0.02, L.BOARD_STARTS[n][1] - 0.02, yaw=2.0) for n in L.BOARDS}
        t.update(jit, PARKED)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        r, t = settled(GOOD)
        steps = t.steps
        run(glide(GOOD, START, 40), t)
        self.assertEqual(t.steps, steps + 40)

    def test_jump_elsewhere_is_not_a_reset(self):
        _, t = settled(START)
        steps = t.steps
        t.update(GOOD, PARKED)
        self.assertEqual(t.steps, steps + 1)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        _, t = settled(GOOD, arms=None)
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertEqual(t.steps, 0)
            self.assertEqual(t.arms_unavailable, 0)
            self.assertEqual(len(t.history["board_1"]), 0)
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
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None], list(L.BOARDS))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertTrue(Tracker.at_start(START))
        for o in self.task.BOARDS:
            self.assertEqual(tuple(o.size), L.BOARD_SIZE)
            self.assertAlmostEqual(o.pos[2] - L.BOARD_SIZE[2] / 2, L.TOP)
        self.assertNotIn(chr(0x2014), self.task.INSTRUCTION)  # no dashes of that kind
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertFalse(self.task.success(START))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        rigstate.offline_feed({a: {"tcp": PARKED[a].tcp, "tips": PARKED[a].tips, "opening": 1.0}
                               for a in ("left", "right")})
        for _ in range(12):
            ok = self.task.success(GOOD)
        self.assertTrue(ok)
        self.assertEqual(self.task.score(GOOD), 1.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertEqual(self.task.score(START), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.assertFalse(self.task.success(START))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_targets_fence_every_line(self):
        frame = {}
        for k in range(1, 5):
            x, y = self.task.line_target(k)
            frame[f"board_{k}"] = board(x, y, yaw=math.degrees(self.task.LINE_YAW[k]))
        r, _ = settled(frame)
        self.assertEqual(r.score, 1.0)
        for b in r.details["boards"].values():
            self.assertAlmostEqual(b["gap_mm"], 5.0, places=3)

    def test_demo_grips_each_board_20_mm_out_from_its_centre(self):
        for k in range(1, 5):
            b = math.radians(L.PEG_BEARINGS_DEG[k - 1])
            out = np.array([math.cos(b), math.sin(b)])
            yaw = self.task.LINE_YAW[k]
            shift = self.task.grip_shift(k)
            self.assertAlmostEqual(abs(shift), 0.020)
            # picked at yaw 0 with the TCP `shift` along the board's x from its centre; turned by the line's yaw,
            # the board's centre is then 20 mm nearer the trap than the TCP, on the line
            tcp = np.array(self.task.tcp_target(k))
            centre = tcp + np.array([math.cos(yaw), math.sin(yaw)]) * -shift
            self.assertTrue(np.allclose(centre, self.task.line_target(k), atol=1e-9), k)
            self.assertAlmostEqual(float((tcp - TC) @ out), GOOD_R + 0.020, places=9)
            # the grip point is on the board and the 20 mm pads stay on it
            self.assertLessEqual(abs(shift) + PAD_HALF_WIDTH, HALF_LEN - 0.030)

    def test_demo_place_tcps_are_far_from_the_other_boards(self):
        # 'free' has a 60 mm radius: a placing TCP must not take another, finished board's line away
        placed = {k: board(*self.task.line_target(k), yaw=math.degrees(self.task.LINE_YAW[k])) for k in range(1, 5)}
        for k in range(1, 5):
            tcp = self.task.tcp_target(k)
            for j in range(1, 5):
                if j != k:
                    self.assertGreater(plan_distance(tcp, footprint(placed[j])), L.FREE_RADIUS + 0.050, (k, j))

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        span = {a: sum(w.duration for w in plan[a]) for a in plan}
        self.assertAlmostEqual(span["left"], 34.0)
        self.assertAlmostEqual(span["right"], 34.0)
        self.assertAlmostEqual(self.task.GRIP, -(L.BOARD_SIZE[1] - 0.008) / 2)  # 8 mm narrower than the board
        for a in plan:
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every close and open takes 1.5 s
                    self.assertEqual(prev.pos, w.pos)  # standing still
                    self.assertIn("settle", prev.label)  # after a settle
                if prev.pos[2] > w.pos[2] and w.pos[2] < L.TOP + 0.04:
                    self.assertGreaterEqual(w.duration, 1.3)  # slow final descents
                if "carry" in w.label:
                    self.assertTrue(L.TOP + 0.10 <= w.pos[2] <= L.TOP + 0.14)

        def releases(ws):
            t, out = 0.0, []
            for i, w in enumerate(ws):
                t += w.duration
                if i and w.finger == self.task.OPEN and ws[i - 1].finger == self.task.GRIP:
                    out.append(round(t, 3))
            return out

        # the two phases run together: both arms let go at the same moments
        self.assertEqual(releases(plan["left"]), [14.5, 30.5])
        self.assertEqual(releases(plan["left"]), releases(plan["right"]))
        self.assertEqual(plan["left"][-1].pos, (0.220, 0.300, L.TOP + 0.160))
        self.assertEqual(plan["right"][-1].pos, (0.220, -0.300, L.TOP + 0.160))
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
