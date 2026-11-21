"""Offline tests for layout.py, scoring.py and task.py: synthetic poses of the hammer and the two bits and synthetic arm
snapshots (rigstate.Arm); no simulator.

    PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v
"""

import json
import math
import pathlib
import sys
import types
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
import tiltgrip  # noqa: E402

AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 0.0)
        for a, s in (("left", 1.0), ("right", -1.0))}


def q_of(rot) -> tuple:
    return tuple(float(v) for v in T.mk.matrix_to_quat(rot))


def st(pos, rot=None, vel=(0.0, 0.0, 0.0)):
    rot = T.HAMMER_ROT if rot is None else rot
    return types.SimpleNamespace(origin=np.array(pos, dtype=float), quat_xyzw=np.array(q_of(rot)),
                                 lin_vel=np.array(vel))


def bit_at(depth: float, roll: float = 0.0):
    """A bit on the chuck's axis with its shank end ``depth`` past the nose, turned ``roll`` about the axis."""
    return st(T.hammer_pt([0.0, depth, L.AXIS_Z]), T.HAMMER_ROT @ tiltgrip.ry(roll))


OLD_HOME = bit_at(L.DEPTH)
OLD_DOWN = st(L.at(0.15, -0.30, L.SHANK_R))
NEW_START = st(T.NEW_ORIGIN)
NEW_HOME = bit_at(L.DEPTH, L.NEW_ROLL)  # turned so its slots sit on the keys


def states(old=OLD_HOME, new=NEW_START, moving=False, sleeve=0.0):
    v = (0.02, 0.0, 0.0) if moving else (0.0, 0.0, 0.0)
    h = st(T.HAMMER_ORIGIN, vel=v)
    h.joints = {"sleeve": sleeve, "keys": 0.0}
    return {S.HAMMER: h, S.OLD_BIT: old, S.NEW_BIT: new}


def hand_at(p, side="right", opening=0.0):
    p = np.asarray(p, dtype=float)
    arms = dict(AWAY)
    arms[side] = rigstate.Arm(p + [0, 0, 0.013], p, opening)
    return arms


def score(s, arms=AWAY):
    return S.Tracker().update(s, arms)


class TestScoring(unittest.TestCase):
    def test_untouched_start_is_zero(self):
        r = score(states())
        self.assertFalse(r.details["old_out"])
        self.assertEqual(r.score, 0.0)

    def test_old_bit_out_and_down_is_a_quarter(self):
        self.assertEqual(score(states(old=OLD_DOWN)).score, 0.25)

    def test_old_bit_tilted_but_still_in_the_chuck_is_not_out(self):
        tilted = st(T.hammer_pt([0.0, L.DEPTH - 0.001, L.AXIS_Z]), T.HAMMER_ROT @ tiltgrip.rx(math.radians(6.0)))
        self.assertFalse(score(states(old=tilted)).details["old_out"])

    def test_old_bit_still_held_is_not_released(self):
        self.assertEqual(score(states(old=OLD_DOWN), hand_at(OLD_DOWN.origin + [0, 0, 0.004])).score, 0.0)

    def test_new_bit_started_is_half(self):
        r = score(states(old=OLD_DOWN, new=bit_at(0.011, 0.0)))
        self.assertTrue(r.details["started"])
        self.assertEqual(r.score, 0.5)

    def test_new_bit_home_and_clear_is_full(self):
        r = score(states(old=OLD_DOWN, new=NEW_HOME))
        self.assertTrue(r.details["locked"])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_new_bit_home_with_the_sleeve_held_back_is_not_locked(self):
        r = score(states(old=OLD_DOWN, new=NEW_HOME, sleeve=0.008))
        self.assertFalse(r.details["locked"])
        self.assertEqual(r.score, 0.5)

    def test_new_bit_home_with_slots_off_the_keys_is_not_locked(self):
        self.assertFalse(score(states(old=OLD_DOWN, new=bit_at(L.DEPTH, 0.0))).details["locked"])

    def test_new_bit_home_with_a_hand_on_it_is_three_quarters(self):
        on = T.axis_pt(-0.05) + [0, 0, 0.005]
        self.assertEqual(score(states(old=OLD_DOWN, new=NEW_HOME), hand_at(on)).score, 0.75)

    def test_moving_is_not_clear(self):
        self.assertEqual(score(states(old=OLD_DOWN, new=NEW_HOME, moving=True)).score, 0.75)

    def test_new_bit_off_the_axis_is_not_in_the_bore(self):
        off = st(T.hammer_pt([0.006, 0.020, L.AXIS_Z]))
        self.assertFalse(score(states(old=OLD_DOWN, new=off)).details["started"])

    def test_hammer_off_the_bench_is_a_foul(self):
        s = states(old=OLD_DOWN, new=NEW_HOME)
        s[S.HAMMER] = st(T.HAMMER_ORIGIN - [0, 0, 0.3])
        s[S.HAMMER].joints = {"sleeve": 0.0, "keys": 0.0}
        r = score(s)
        self.assertIn("off the bench", r.details["fouls"])
        self.assertEqual(r.score, 0.0)

    def test_unreadable_state(self):
        self.assertFalse(S.Tracker().update({}, AWAY).details["valid_state"])


class TestRules(unittest.TestCase):
    def test_keys_stay_open_for_a_locked_or_aligned_shank(self):
        self.assertTrue(T.keys_open([S.Seat(True, L.DEPTH, 0.0)]))
        self.assertTrue(T.keys_open([S.Seat(True, L.KEY_DEPTH - 0.002, math.radians(3.0))]))

    def test_keys_close_on_a_misaligned_shank_and_an_empty_bore(self):
        self.assertFalse(T.keys_open([S.Seat(True, L.KEY_DEPTH - 0.002, math.radians(30.0))]))
        self.assertFalse(T.keys_open([]))

    def test_a_shank_past_the_keys_keeps_them_open(self):
        self.assertTrue(T.keys_open([S.Seat(True, L.KEY_DEPTH + 0.010, math.radians(30.0))]))

    def test_new_bit_lies_45_degrees_off_the_keys(self):
        h = S.Body(T.HAMMER_ORIGIN, T.HAMMER_ROT, np.zeros(3))
        b = S.Body(T.NEW_ORIGIN, T.HAMMER_ROT, np.zeros(3))
        self.assertAlmostEqual(math.degrees(S.seat(h, b, L.NEW_ROLL).roll), 45.0, places=3)

    def test_the_lock_snaps_the_slots_onto_the_keys(self):
        h = S.Body(T.HAMMER_ORIGIN, T.HAMMER_ROT, np.zeros(3))
        b = S.Body(T.hammer_pt(T.OLD_HOME), T.HAMMER_ROT @ tiltgrip.ry(-L.NEW_ROLL + math.radians(5.0)), np.zeros(3))
        pos, quat = T.snapped(h, b, L.NEW_ROLL)
        locked = S.Body(T.hammer_pt(pos), T.HAMMER_ROT @ T.mk.quat_to_matrix(quat), np.zeros(3))
        self.assertLess(S.seat(h, locked, L.NEW_ROLL).roll, 1e-6)


class TestLayout(unittest.TestCase):
    def test_the_shank_fits_the_bore_and_the_cap_centres_it(self):
        self.assertGreater(L.BORE_R, L.SHANK_R)
        self.assertGreater(L.CAP_EXIT, L.SHANK_R)
        self.assertLess(L.CAP_EXIT / math.cos(math.pi / 8), L.BORE_R)  # the cap's exit corners inside the bore

    def test_the_open_gate_is_clear_of_the_bore(self):
        self.assertGreater(L.KEY_TRAVEL - L.BORE_R, L.BORE_R)


class TestDemo(unittest.TestCase):
    def test_variants(self):
        for v in ("", *T.VARIANTS):
            self.assertTrue(T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_the_twist_turns_the_slots_onto_the_keys(self):
        self.assertAlmostEqual(T.TWIST, math.degrees(L.NEW_ROLL))


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
