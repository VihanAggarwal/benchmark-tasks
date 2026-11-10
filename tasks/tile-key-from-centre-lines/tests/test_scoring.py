"""Offline tests for scoring.py (and task.py's hooks and demo): synthetic tile states, no simulator.

    python3 -m unittest discover -s tests -v
"""

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
    Tracker,
    at_crossing,
    cuts,
    direction_good,
    evaluate,
    lowest,
    rotation,
    square_err_deg,
    tilt_deg,
    valid,
)

HZ = L.CONTROL_HZ
HALF = np.array(L.TILE_SIZE) / 2
N_STILL = 12  # more than the 0.5 s window (10 steps + the current one)
TL, TR = "tile_left", "tile_right"
LEFT, RIGHT = L.ROOMS["left"], L.ROOMS["right"]
J = L.MODULE / 2  # 0.031: a joint on the line puts the tile's centre this far from it


# -- building tile states ----------------------------------------------------------------------------------------
def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def quat_of(r):
    """(x, y, z, w) of a rotation matrix (independent of the code under test)."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def box(centre, r, vel=(0.0, 0.0, 0.0)):
    """A tile state as the simulator reports it: centre, quaternion, velocities, world bounding box."""
    c = np.asarray(centre, dtype=float)
    corners = np.array([c + r @ (HALF * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return SimpleNamespace(pos=c, quat_xyzw=quat_of(r), lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=corners.min(axis=0), high=corners.max(axis=0))


def tile(x, y, yaw=0.0, lift=0.0, roll=0.0):
    """A tile turned ``yaw`` degrees and rolled ``roll`` degrees about its own x axis, its lowest point ``lift``
    above the floor."""
    r = rz(yaw) @ rx(roll)
    drop = float((np.array([r @ (HALF * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1)
                            for k in (-1, 1)])[:, 2]).min())
    return box((x, y, L.TOP + lift - drop), r)


def at(room, u_fb, u_ss, **kw):
    """A tile with its centre u_fb from the room's front-to-back line (along y) and u_ss from its side-to-side line
    (along x)."""
    return tile(room.cx + u_ss, room.cy + u_fb, **kw)


def start_tile(name):
    x, y, yaw = L.TILE_START[name]
    return tile(x, y, yaw)


START = {TL: start_tile(TL), TR: start_tile(TR)}
# The answers: the left room a joint on its front-to-back line (340 mm across), centred on its side-to-side line
# (278 mm across); the right room centred on its front-to-back line (278 mm), a joint on its side-to-side line (340).
NOMINAL = {TL: at(LEFT, +J, 0.0), TR: at(RIGHT, 0.0, -J)}


def run(seq, t=None):
    """Feed a list of {tile: state} dicts to a tracker (a fresh one unless given); last result and tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for tiles in seq:
        r = t.update(tiles)
    return r, t


def held(tiles, n=N_STILL):
    return [tiles] * n


def score_of(tiles):
    return run(held(tiles))[0].score


def moved_by(s, d):
    """The same state shifted by the vector d (m)."""
    d = np.asarray(d, dtype=float)
    return SimpleNamespace(pos=s.pos + d, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel, ang_vel=s.ang_vel,
                           low=s.low + d, high=s.high + d)


# -- the set-out arithmetic ----------------------------------------------------------------------------------------
class Arithmetic(unittest.TestCase):
    def test_rooms_and_crossings(self):
        self.assertEqual((LEFT.depth, LEFT.width, LEFT.cx, LEFT.cy), (0.278, 0.340, 0.289, 0.240))
        self.assertEqual((RIGHT.depth, RIGHT.width, RIGHT.cx, RIGHT.cy), (0.340, 0.278, 0.290, -0.209))
        self.assertAlmostEqual(L.MODULE, 0.062)

    def test_worked_examples(self):
        for span, u, want in ((0.278, 0.0, 45.0), (0.278, J, 14.0), (0.340, 0.0, 14.0), (0.340, J, 45.0),
                              (0.340, -J, 45.0), (0.278, -J, 14.0)):
            a, b = cuts(span, u)
            self.assertAlmostEqual(a * 1000, want, places=6, msg=(span, u))
            self.assertAlmostEqual(b * 1000, want, places=6, msg=(span, u))

    def test_full_size_example_from_the_context(self):
        """At full size (300 mm tiles, 10 mm joints), a 1.70 m room centred on its line leaves 70 mm slivers; with
        a joint on the line, 225 mm cuts. The 1:5 cuts are a fifth of these."""
        def full(span, u):
            return [round((span / 2 + s * u - 160) % 310, 6) for s in (-1, 1)]
        self.assertEqual(full(1700, 0), [70.0, 70.0])
        self.assertEqual(full(1700, 155), [225.0, 225.0])
        self.assertEqual(full(1390, 0), [225.0, 225.0])
        self.assertEqual(full(1390, 155), [70.0, 70.0])
        self.assertAlmostEqual(cuts(0.340, 0.0)[0] * 5, 0.070)
        self.assertAlmostEqual(cuts(0.340, J)[0] * 5, 0.225)

    def test_cuts_add_up_to_28_or_90(self):
        rng = np.random.default_rng(3)
        for _ in range(500):
            span = float(rng.choice([0.278, 0.340]))
            u = float(rng.uniform(-0.045, 0.045))
            total = sum(cuts(span, u)) * 1000
            self.assertTrue(abs(total - 28) < 1e-6 or abs(total - 90) < 1e-6, (span, u, total))
            if direction_good(span, u):  # both cuts >= 30 mm only when they add up to 90 mm
                self.assertAlmostEqual(total, 90.0, places=6)

    def test_windows(self):
        """Across 278 mm the direction is right with the centre within 15 mm of the line; across 340 mm, with it
        16 mm or more from the line (up to 46; the crossing limit is 45). Sampled every 0.1 mm."""
        for k in range(-450, 451):
            u = k / 10000.0
            self.assertEqual(direction_good(0.278, u), abs(k) <= 150, u)
            self.assertEqual(direction_good(0.340, u), abs(k) >= 160, u)

    def test_the_answers_are_mid_window(self):
        """The demo's aims leave 15 mm either way in each direction (14 mm tested, on the safe side)."""
        for name, room, fb, ss in ((TL, LEFT, J, 0.0), (TR, RIGHT, 0.0, -J)):
            tx, ty = L.TARGET["left" if room is LEFT else "right"]
            self.assertAlmostEqual(tx, room.cx + ss)
            self.assertAlmostEqual(ty, room.cy + fb)
            for d in (-0.014, 0.0, 0.014):
                self.assertTrue(direction_good(room.width, fb + d), (name, d))
                self.assertTrue(direction_good(room.depth, ss + d), (name, d))
        self.assertFalse(direction_good(LEFT.width, J - 0.0155))
        self.assertFalse(direction_good(RIGHT.depth, -J + 0.0155))

    def test_tilt_square_and_lowest(self):
        self.assertAlmostEqual(tilt_deg(tile(0.3, 0.2)), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(tile(0.3, 0.2, roll=4.0)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(tile(0.3, 0.2, roll=180.0)), 0.0, places=4)  # either face up
        self.assertAlmostEqual(tilt_deg(tile(0.3, 0.2, roll=90.0)), 90.0, places=4)  # on an edge
        for yaw, want in ((4.0, 4.0), (-4.0, 4.0), (86.0, 4.0), (45.0, 45.0), (90.0, 0.0), (180.0, 0.0),
                          (20.0, 20.0), (-25.0, 25.0)):
            self.assertAlmostEqual(square_err_deg(tile(0.3, 0.2, yaw=yaw)), want, places=4, msg=yaw)
        self.assertAlmostEqual(lowest(tile(0.3, 0.2, lift=0.0015, roll=3.0)), 0.0015, places=9)

    def test_at_crossing(self):
        self.assertTrue(at_crossing(at(LEFT, 0.045, -0.045), LEFT))
        self.assertFalse(at_crossing(at(LEFT, 0.0455, 0.0), LEFT))
        self.assertFalse(at_crossing(at(LEFT, 0.0, -0.0455), LEFT))
        self.assertFalse(at_crossing(at(LEFT, 0.0, 0.0), RIGHT))
        for name in (TL, TR):
            self.assertFalse(any(at_crossing(START[name], r) for r in L.ROOMS.values()))

    def test_valid(self):
        self.assertTrue(valid(NOMINAL[TL]))
        s = SimpleNamespace(**vars(NOMINAL[TL]))
        s.quat_xyzw = np.zeros(4)
        self.assertFalse(valid(s))
        s = SimpleNamespace(**vars(NOMINAL[TL]))
        s.pos = np.array([0.3, np.nan, 0.75])
        self.assertFalse(valid(s))
        s = SimpleNamespace(**vars(NOMINAL[TL]))
        s.pos = np.array([0.3, 0.2])
        self.assertFalse(valid(s))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))


# -- the tiers -----------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(held(START))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        for key in ("left", "right"):
            self.assertEqual(r.details["rooms"][key]["zero_reason"], "no_tile_at_crossing")

    def test_both_rooms_right_scores_one(self):
        r, _ = run(held(NOMINAL))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["rooms_right"], ["left", "right"])
        left = r.details["rooms"]["left"]
        self.assertEqual(left["cuts_across_front_to_back_line_mm"], [45.0, 45.0])
        self.assertEqual(left["cuts_across_side_to_side_line_mm"], [45.0, 45.0])
        self.assertEqual((left["off_front_to_back_line_mm"], left["off_side_to_side_line_mm"]), (31.0, 0.0))

    def test_every_right_spot(self):
        """A joint on the line may be on either side of it: two right spots per room, four layouts."""
        for fb in (+J, -J):
            for ss in (+J, -J):
                tiles = {TL: at(LEFT, fb, 0.0), TR: at(RIGHT, 0.0, ss)}
                self.assertEqual(score_of(tiles), 1.0, (fb, ss))

    def test_either_tile_in_either_room(self):
        self.assertEqual(score_of({TL: NOMINAL[TR], TR: NOMINAL[TL]}), 1.0)

    def test_one_room_right_scores_half(self):
        for tiles in ({TL: NOMINAL[TL], TR: START[TR]}, {TL: START[TL], TR: NOMINAL[TR]}):
            r, _ = run(held(tiles))
            self.assertEqual(r.score, 0.5)
            self.assertFalse(r.success)  # only 1.0 ends the trial early

    def test_each_wrong_set_out_scores_zero_for_its_room(self):
        """Of the four ways to set a room out (centred or a joint, on each line), only one is right."""
        for key, room, other, right in (("left", LEFT, TR, (J, 0.0)), ("right", RIGHT, TL, (0.0, -J))):
            mine = TL if other == TR else TR
            for fb in (0.0, J):
                for ss in (0.0, -J):
                    tiles = {mine: at(room, fb, ss), other: NOMINAL[other]}
                    r, _ = run(held(tiles))
                    good = (abs(fb - right[0]) < 1e-9 and abs(ss - right[1]) < 1e-9)
                    self.assertEqual(r.score, 1.0 if good else 0.5, (key, fb, ss))
                    if not good:
                        self.assertEqual(r.details["rooms"][key]["zero_reason"], "set_out_wrong")

    def test_window_edges(self):
        cases = (
            ({TL: at(LEFT, 0.016, 0.015)}, 1.0),
            ({TL: at(LEFT, 0.045, -0.015)}, 1.0),
            ({TL: at(LEFT, -0.016, 0.0)}, 1.0),
            ({TL: at(LEFT, 0.0155, 0.0)}, 0.5),  # neither centred nor a joint: a 29.5 mm cut
            ({TL: at(LEFT, J, 0.0155)}, 0.5),
            ({TL: at(LEFT, 0.0455, 0.0)}, 0.5),  # past the crossing limit
        )
        for tiles, want in cases:
            tiles = {**tiles, TR: NOMINAL[TR]}
            self.assertEqual(score_of(tiles), want, tiles[TL].pos)

    def test_every_tier_is_reachable(self):
        seen = {score_of(START), score_of({TL: NOMINAL[TL], TR: START[TR]}), score_of(NOMINAL)}
        self.assertEqual(seen, {0.0, 0.5, 1.0})

    def test_still_needs_half_a_second_of_the_trial(self):
        r, t = run(held(NOMINAL, 10))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_still")
        r = t.update(NOMINAL)  # the 11th sample completes the 0.5 s window
        self.assertEqual(r.score, 1.0)

    def test_moving_tile_is_not_laid(self):
        seq = [{TL: moved_by(NOMINAL[TL], (0.0, 0.0003 * i, 0.0)), TR: NOMINAL[TR]} for i in range(N_STILL)]
        r, _ = run(seq)  # 3.3 mm in 0.5 s
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_still")
        seq = [{TL: moved_by(NOMINAL[TL], (0.0, 0.0005 * (-1) ** i, 0.0)), TR: NOMINAL[TR]} for i in range(N_STILL)]
        self.assertEqual(run(seq)[0].score, 1.0)  # 1 mm of jitter is still

    def test_tilt_limit(self):
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0, roll=4.9), TR: NOMINAL[TR]}), 1.0)
        r, _ = run(held({TL: at(LEFT, J, 0.0, roll=5.2), TR: NOMINAL[TR]}))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_flat")

    def test_resting_on_a_chalk_line(self):
        """Laid across a chalk line (0.5 mm proud), a tile sits about 1 degree off level: still laid."""
        tilt = math.degrees(math.atan2(L.LINE_T, L.TILE / 2))
        self.assertAlmostEqual(tilt, 0.95, places=2)
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0, roll=tilt), TR: at(RIGHT, 0.0, -J, roll=-tilt)}), 1.0)
        # resting on both lines at once (centred on both): lowest point 0.5 mm up, still on the floor
        self.assertAlmostEqual(lowest(at(LEFT, 0.0, 0.0, lift=L.LINE_T)), L.LINE_T, places=12)

    def test_square_limit(self):
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0, yaw=4.9), TR: at(RIGHT, 0.0, -J, yaw=94.9)}), 1.0)
        for yaw in (5.2, 10.0, 45.0, -20.0):
            r, _ = run(held({TL: at(LEFT, J, 0.0, yaw=yaw), TR: NOMINAL[TR]}))
            self.assertEqual(r.score, 0.5, yaw)
            self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_square", yaw)

    def test_on_the_floor_limit(self):
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0, lift=0.0019), TR: NOMINAL[TR]}), 1.0)
        r, _ = run(held({TL: at(LEFT, J, 0.0, lift=0.0025), TR: NOMINAL[TR]}))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_on_floor")

    def test_on_an_edge_or_upside_down(self):
        r, _ = run(held({TL: at(LEFT, J, 0.0, roll=90.0), TR: NOMINAL[TR]}))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_flat")
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0, roll=180.0), TR: NOMINAL[TR]}), 1.0)  # faces alike


# -- exploits we tried ---------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_both_tiles_centred_on_the_crossings(self):
        """Not reading either room: each is wrong in one direction, 0."""
        r, _ = run(held({TL: at(LEFT, 0.0, 0.0), TR: at(RIGHT, 0.0, 0.0)}))
        self.assertEqual(r.score, 0.0)
        self.assertEqual({k: v["zero_reason"] for k, v in r.details["rooms"].items()},
                         {"left": "set_out_wrong", "right": "set_out_wrong"})

    def test_a_joint_on_every_line(self):
        self.assertEqual(score_of({TL: at(LEFT, J, J), TR: at(RIGHT, J, -J)}), 0.0)

    def test_left_answer_copied_into_the_right_room(self):
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0), TR: at(RIGHT, J, 0.0)}), 0.5)

    def test_hedging_with_both_tiles_at_one_crossing(self):
        """Two tiles at one crossing, set out two ways (one of them right): 0 for that room, and the other room
        has none."""
        tiles = {TL: at(RIGHT, 0.0, -J), TR: at(RIGHT, J, J)}
        self.assertGreaterEqual(abs(tiles[TL].pos[0] - tiles[TR].pos[0]), L.TILE)  # side by side, not overlapping
        r, _ = run(held(tiles))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["rooms"]["right"]["zero_reason"], "two_tiles_at_crossing")

    def test_stacked_at_a_crossing(self):
        tiles = {TL: NOMINAL[TL], TR: at(LEFT, J, 0.0, lift=L.TILE_T)}
        self.assertEqual(score_of(tiles), 0.0)

    def test_the_same_grid_one_module_further_out(self):
        """The right grid with its first tile a whole module out has the same cuts but is not at the crossing."""
        far = at(LEFT, J + L.MODULE, 0.0)
        self.assertTrue(direction_good(LEFT.width, J + L.MODULE))
        self.assertEqual(score_of({TL: far, TR: NOMINAL[TR]}), 0.5)

    def test_right_offsets_but_turned_ten_degrees(self):
        self.assertEqual(score_of({TL: at(LEFT, J, 0.0, yaw=10.0), TR: NOMINAL[TR]}), 0.5)

    def test_tile_off_the_table_or_on_a_wall(self):
        """A tile fallen to the floor under the left room's right spot is not laid there; nor is one on a wall."""
        gone = box((LEFT.cx, LEFT.cy + J, 0.30), np.eye(3))
        r, _ = run(held({TL: gone, TR: NOMINAL[TR]}))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["rooms"]["left"]["zero_reason"], "not_on_floor")
        sunk = at(LEFT, J, 0.0, lift=-0.0025)
        self.assertEqual(score_of({TL: sunk, TR: NOMINAL[TR]}), 0.5)
        on_wall = tile(LEFT.x0 - L.WALL_T / 2, LEFT.cy, lift=L.WALL_H)
        self.assertEqual(score_of({TL: on_wall, TR: NOMINAL[TR]}), 0.5)


# -- robustness ----------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_nan_state_scores_zero_and_keeps_the_record(self):
        r, t = run(held(NOMINAL))
        self.assertEqual(r.score, 1.0)
        bad = SimpleNamespace(**vars(NOMINAL[TL]))
        bad.pos = np.array([np.nan, 0.0, 0.0])
        r = t.update({TL: bad, TR: NOMINAL[TR]})
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(r.details["bad_tiles"], [TL])
        self.assertEqual(t.bad_steps, 1)
        self.assertEqual(t.moved(TL), 0.0)  # the history was left alone
        json.dumps(r.details, allow_nan=False)
        r = t.update(NOMINAL)
        self.assertEqual(r.score, 1.0)

    def test_zero_quaternion_inf_and_missing_fields(self):
        for key, val in (("quat_xyzw", np.zeros(4)), ("lin_vel", np.array([np.inf, 0, 0])), ("high", None),
                         ("ang_vel", np.array([0.0, 0.0]))):
            s = SimpleNamespace(**vars(NOMINAL[TR]))
            setattr(s, key, val)
            r, _ = run(held({TL: NOMINAL[TL], TR: s}))
            self.assertEqual(r.score, 0.0, key)
            json.dumps(r.details, allow_nan=False)
        r, _ = run([{TL: NOMINAL[TL], TR: SimpleNamespace(pos=NOMINAL[TR].pos)}])
        self.assertEqual(r.score, 0.0)

    def test_reset_by_teleport_starts_a_new_trial(self):
        r, t = run(held(START) + held(NOMINAL))
        self.assertEqual(r.score, 1.0)
        self.assertIn("1.0", r.details["first_step"])
        r = t.update(START)  # the simulator puts both tiles back in one step
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.details["first_step"], {})
        self.assertIsNone(t.moved(TL))

    def test_reset_with_jitter_or_swapped_tiles_is_recognised(self):
        _, t = run(held(NOMINAL))
        jit = {TL: tile(L.TILE_START[TL][0] + 0.02, L.TILE_START[TL][1] - 0.02, 30.0),
               TR: tile(L.TILE_START[TR][0] - 0.02, L.TILE_START[TR][1] + 0.025, -10.0)}
        t.update(jit)
        self.assertEqual(t.steps, 1)
        _, t = run(held(NOMINAL))
        t.update({TL: START[TR], TR: START[TL]})
        self.assertEqual(t.steps, 1)

    def test_gradual_return_or_jump_elsewhere_is_not_a_reset(self):
        _, t = run(held(NOMINAL))
        n = t.steps
        for i in range(1, 41):  # carried back gradually: same trial
            f = i / 40
            t.update({k: moved_by(NOMINAL[k], f * (START[k].pos - NOMINAL[k].pos)) for k in NOMINAL})
        self.assertEqual(t.steps, n + 40)
        _, t = run(held(NOMINAL))
        n = t.steps
        t.update({TL: tile(0.40, 0.33, 30.0), TR: NOMINAL[TR]})  # a jump, but not onto the marks
        self.assertEqual(t.steps, n + 1)

    def test_result_is_idempotent(self):
        _, t = run(held(NOMINAL))
        a, b = t.result(NOMINAL), t.result(NOMINAL)
        self.assertEqual(a.score, b.score)
        self.assertEqual(json.dumps(a.details, sort_keys=True), json.dumps(b.details, sort_keys=True))
        self.assertEqual(t.steps, N_STILL)

    def test_evaluate_without_history_never_counts_a_tile_still(self):
        score, details = evaluate(NOMINAL, {})
        self.assertEqual(score, 0.0)
        self.assertEqual(details["rooms"]["left"]["zero_reason"], "not_still")

    def test_random_states_never_crash(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            tiles = {}
            for n in (TL, TR):
                q = rng.normal(size=4)
                tiles[n] = box(rng.uniform([0.1, -0.4, L.TOP], [0.45, 0.4, L.TOP + 0.08]), rotation(q))
            res, _ = run(held(tiles, 2))
            self.assertIn(res.score, (0.0, 0.5, 1.0))
            json.dumps(res.details, allow_nan=False)


# -- task.py against the offline stand-in for yam_arena ----------------------------------------------------------
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

    def test_objects_match_the_layout(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        loose = [o.name for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(loose, [TL, TR])
        self.assertEqual(len(self.task.WALLS), 8)
        self.assertEqual(len(self.task.LINES), 4)
        for spec in self.task.TILES:
            x, y, yaw = L.TILE_START[spec.name]
            self.assertEqual((spec.pos[0], spec.pos[1]), (x, y))
            self.assertAlmostEqual(spec.pos[2], L.TOP + L.TILE_T / 2)
            self.assertAlmostEqual(spec.yaw, math.radians(yaw))
            self.assertEqual(tuple(spec.size), L.TILE_SIZE)
        walls = {w.name: w for w in self.task.WALLS}
        for key, room in L.ROOMS.items():
            near, far = walls[f"wall_{key}_near"], walls[f"wall_{key}_far"]
            right, left = walls[f"wall_{key}_right"], walls[f"wall_{key}_left"]
            self.assertAlmostEqual(near.pos[0] + near.size[0] / 2, room.x0)
            self.assertAlmostEqual(far.pos[0] - far.size[0] / 2, room.x1)
            self.assertAlmostEqual(right.pos[1] + right.size[1] / 2, room.y0)
            self.assertAlmostEqual(left.pos[1] - left.size[1] / 2, room.y1)
        lines = {ln.name: ln for ln in self.task.LINES}
        self.assertAlmostEqual(lines["line_left_front_to_back"].pos[1], 0.240)
        self.assertAlmostEqual(lines["line_left_side_to_side"].pos[0], 0.289)
        self.assertAlmostEqual(lines["line_right_front_to_back"].pos[1], -0.209)
        self.assertAlmostEqual(lines["line_right_side_to_side"].pos[0], 0.290)
        self.assertAlmostEqual(lines["line_left_front_to_back"].size[0], LEFT.depth)
        self.assertAlmostEqual(lines["line_right_side_to_side"].size[1], RIGHT.width)

    def test_start_tiles_clear_of_walls_and_the_screenshot_shadow(self):
        for name, (x, y, yaw) in L.TILE_START.items():
            room = LEFT if y > 0 else RIGHT
            h = L.TILE / 2 * (abs(math.cos(math.radians(yaw))) + abs(math.sin(math.radians(yaw))))
            self.assertGreaterEqual(min(x - h - room.x0, room.x1 - x - h, y - h - room.y0, room.y1 - y - h), 0.016)
            self.assertFalse(abs(y) + h > 0.45 and x - h < 0.2)

    def test_start_tiles_sit_flat_off_the_chalk_lines(self):
        """At the start no tile overlaps a chalk line (so each lies flat on the table), and the demo's laid tile with a
        joint on a line overlaps that 4 mm line's edge by only 1 mm."""
        for name, (x, y, yaw) in L.TILE_START.items():
            room = LEFT if y > 0 else RIGHT
            h = L.TILE / 2 * (abs(math.cos(math.radians(yaw))) + abs(math.sin(math.radians(yaw))))
            self.assertGreater(abs(y - room.cy) - h, L.LINE_W / 2, name)
            self.assertGreater(abs(x - room.cx) - h, L.LINE_W / 2, name)
        self.assertAlmostEqual(L.LINE_W / 2 - (J - L.TILE / 2), 0.001)

    def test_open_pads_clear_the_walls(self):
        """With the fingers open (79 mm) and turned as the demo turns them, both pads (8 mm thick, 27 mm wide) stay at
        least 20 mm inside every wall at each pick and each place."""
        plan = self.task.demo()
        gap = -2 * self.task.OPEN
        self.assertAlmostEqual(gap, 0.079)
        for side, wps in plan.items():
            room = L.ROOMS[side]
            for w in wps:
                if not (w.label.startswith("slowly down round") or w.label.startswith("slowly lay")):
                    continue
                c = np.array([-math.sin(w.yaw), math.cos(w.yaw)])  # the fingers close along this
                p = np.array([math.cos(w.yaw), math.sin(w.yaw)])
                pts = [np.array(w.pos[:2]) + s * r * c + q * 0.0135 * p
                       for s in (-1, 1) for r in (gap / 2, gap / 2 + 0.008) for q in (-1, 1)]
                clear = min(min(x - room.x0, room.x1 - x, y - room.y0, room.y1 - y) for x, y in pts)
                self.assertGreaterEqual(clear, 0.020, (side, w.label, clear))

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(START))
                self.assertEqual(self.task.score(START), 0.0)
                for _ in range(N_STILL):
                    ok = self.task.success(NOMINAL)
                self.assertTrue(ok)
                self.assertEqual(self.task.score(NOMINAL), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1 + N_STILL)
        for key in ("step", "score", "success", "rooms", "rooms_right", "moved_mm", "bad_steps", "first_step",
                    "tile_left_pose", "tile_right_pose"):
            self.assertIn(key, rows[-1])
        for key in ("tile", "off_front_to_back_line_mm", "off_side_to_side_line_mm",
                    "cuts_across_front_to_back_line_mm", "cuts_across_side_to_side_line_mm", "front_to_back_good",
                    "side_to_side_good", "tilt_deg", "square_err_deg", "lowest_mm", "still", "zero_reason"):
            self.assertIn(key, rows[-1]["rooms"]["left"])
        self.assertEqual(rows[-1]["score"], 1.0)

    def test_trace_survives_a_bad_path(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(START))
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

    def test_demo_plan(self):
        """Each arm picks its own room's tile turned with it and lays it square on the right spot, 2 mm up; the
        plan fits the run limit with the policy's 3 s hold."""
        plan = self.task.demo()
        for side, name in (("left", TL), ("right", TR)):
            wps = plan[side]
            sx, sy, syaw = L.TILE_START[name]
            grip = [w for w in wps if w.label.startswith("grip the tile")][0]
            lay = [w for w in wps if w.label.startswith("slowly lay")][0]
            self.assertEqual((grip.pos[0], grip.pos[1]), (sx, sy))
            self.assertAlmostEqual(grip.yaw, math.radians(syaw))
            self.assertAlmostEqual(grip.pos[2] - self.task.GRIP_UP, L.TOP)
            self.assertAlmostEqual(grip.finger, -(L.TILE - 0.008) / 2)
            self.assertEqual((lay.pos[0], lay.pos[1]), L.TARGET[side])
            self.assertAlmostEqual(lay.pos[2] - self.task.GRIP_UP - L.TOP, 0.002)  # let go 2 mm above the floor
            self.assertAlmostEqual(self.task.GRIP_UP, 0.016)  # pad tips 3 mm above the tile's underside
            self.assertAlmostEqual(lay.yaw % (math.pi / 2), 0.0)
            room = L.ROOMS[side]
            u_fb, u_ss = lay.pos[1] - room.cy, lay.pos[0] - room.cx
            self.assertTrue(direction_good(room.width, u_fb) and direction_good(room.depth, u_ss))
            self.assertEqual(wps[-1].label, "out of the way")
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 1200)
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.0)

    def test_task_json_states_the_thresholds(self):
        """The scoring text gives the numbers the scorer uses (not the scores placeholder, which the GPU run fills)."""
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for phrase in ("within 45 mm of both", "within 5 degrees of level", "within 5 degrees of square",
                       "within 2 mm of the floor", "more than 2 mm in the last 0.5 s", "(D/2 - u - 32 mm)",
                       "(D/2 + u - 32 mm)", "modulo 62 mm", "at least 30 mm", "within 15 mm of the line",
                       "16 mm or more from the line", "cuts 45 and 45 mm", "14 and 14 mm", "exactly one tile",
                       "both tiles at its crossing", "held for 2 s", "Possible scores: 0, 0.5, 1.0"):
            self.assertIn(phrase, text)
        self.assertEqual((L.AT_CROSSING, L.ON_FLOOR, L.STILL_TOL, L.HALF_TILE), (0.045, 0.002, 0.002, 0.030))
        self.assertEqual((L.FLAT_DEG, L.SQUARE_DEG, L.STILL_WINDOW_S), (5.0, 5.0, 0.5))
        self.assertIn("1,200 steps", spec["run_limit"])
        setup = spec["setup"]
        for phrase in ("x 15.0 and 42.8 cm", "y +7.0 and +41.0 cm", "x 12.0 and 46.0 cm", "y -34.8 and -7.0 cm",
                       "y +24.0 cm", "x 28.9 cm", "y -20.9 cm", "x 29.0 cm", "x 20.5 cm, y +33.5 cm",
                       "x 19.5 cm, y -27.5 cm", "20 degrees anticlockwise", "25 degrees clockwise", "80 g",
                       "60 x 60 mm and 10 mm thick", "10 mm thick and 25 mm tall", "4 mm wide and 0.5 mm thick"):
            self.assertIn(phrase, setup)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(L.TILE_START, {TL: (0.205, 0.335, 20.0), TR: (0.195, -0.275, -25.0)})  # as the setup says
        self.assertEqual((L.TILE_SIZE, L.TILE_MASS, L.WALL_T, L.WALL_H, L.LINE_W, L.LINE_T),
                         ((0.060, 0.060, 0.010), 0.080, 0.010, 0.025, 0.004, 0.0005))

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


if __name__ == "__main__":
    unittest.main()
