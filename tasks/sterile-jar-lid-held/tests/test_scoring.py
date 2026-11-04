"""Offline tests for scoring.py: synthetic object states, no simulator.  python3 -m unittest discover -s tests -v"""

import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import Tracker, cube_in_jar, lid_seated, tilt_deg  # noqa: E402

LID_H = L.LID_SIZE[2]
HALF_CUBE = L.CUBE_EDGE / 2


def state(center, size, quat=(0, 0, 0, 1), vel=(0, 0, 0)):
    c, s = np.array(center, float), np.array(size, float) / 2
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.zeros(3), low=c - s, high=c + s)


def lid_at(x, y, bottom_z, **kw):
    return state((x, y, bottom_z + LID_H / 2), L.LID_SIZE, **kw)


def cube_at(x, y, bottom_z, **kw):
    return state((x, y, bottom_z + HALF_CUBE), (L.CUBE_EDGE,) * 3, **kw)


SEATED = lid_at(L.JAR_X, L.JAR_Y, L.RIM_Z)
IN_AIR = lid_at(L.JAR_X, L.JAR_Y + 0.20, L.TOP + 0.133)
ON_BENCH = lid_at(L.JAR_X + 0.12, L.JAR_Y + 0.15, L.TOP)
CUBE_HOME = cube_at(L.CUBE_X, L.CUBE_Y, L.TOP)
CUBE_IN = cube_at(L.JAR_X, L.JAR_Y, L.TOP)
CUBE_UP = cube_at(L.CUBE_X, L.CUBE_Y, L.TOP + 0.12)


def glide(a, b, n):
    """n states moving a's centre to b's in equal steps (the simulator moves things gradually)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        c = (1 - f) * a.pos + f * b.pos
        out.append(state(c, a.high - a.low))
    return out


def run(sequence, hz=20.0):
    """Feed (lid, cube, steps) segments to a fresh tracker; return the last result."""
    t = Tracker(hz)
    r = None
    for lid, cube, n in sequence:
        for _ in range(n):
            r = t.update(lid, cube)
    return r, t


class Geometry(unittest.TestCase):
    def test_seated_and_tolerances(self):
        self.assertTrue(lid_seated(SEATED))
        self.assertTrue(lid_seated(lid_at(L.JAR_X + 0.004, L.JAR_Y, L.RIM_Z)))
        self.assertFalse(lid_seated(lid_at(L.JAR_X + 0.006, L.JAR_Y, L.RIM_Z)))
        self.assertFalse(lid_seated(lid_at(L.JAR_X, L.JAR_Y, L.RIM_Z + 0.006)))
        tilted = (np.sin(np.radians(3)), 0, 0, np.cos(np.radians(3)))  # 6 degrees about x
        self.assertAlmostEqual(tilt_deg(tilted), 6.0, places=4)
        self.assertFalse(lid_seated(lid_at(L.JAR_X, L.JAR_Y, L.RIM_Z, quat=tilted)))

    def test_cube_in_jar(self):
        self.assertTrue(cube_in_jar(CUBE_IN))
        self.assertTrue(cube_in_jar(cube_at(L.JAR_X + 0.012, L.JAR_Y - 0.012, L.TOP)))
        self.assertFalse(cube_in_jar(CUBE_HOME))
        self.assertFalse(cube_in_jar(cube_at(L.JAR_X, L.JAR_Y, L.RIM_Z + LID_H)))  # on top of the closed lid
        self.assertFalse(cube_in_jar(cube_at(L.JAR_X + 0.06, L.JAR_Y, L.TOP)))  # beside the jar


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run([(SEATED, CUBE_HOME, 10)])
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = run([(SEATED, CUBE_HOME, 20), (IN_AIR, CUBE_UP, 100), (IN_AIR, CUBE_IN, 60), (SEATED, CUBE_IN, 30)])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertAlmostEqual(t.open_s, 8.0)

    def test_open_too_long_scores_075(self):
        r, t = run([(IN_AIR, CUBE_UP, 300), (IN_AIR, CUBE_IN, 150), (SEATED, CUBE_IN, 30)])  # 22.5 s open
        self.assertGreater(t.open_s, L.OPEN_LIMIT_S)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_cube_in_lid_still_in_air_scores_half(self):
        r, _ = run([(IN_AIR, CUBE_IN, 40)])
        self.assertEqual(r.score, 0.5)

    def test_lid_lifted_only_scores_quarter(self):
        t = Tracker()
        for lid in [SEATED] * 5 + glide(SEATED, IN_AIR, 30) + glide(IN_AIR, SEATED, 30) + [SEATED] * 40:
            r = t.update(lid, CUBE_HOME)
        self.assertEqual(r.score, 0.25)

    def test_setting_the_lid_down_scores_zero_whatever_follows(self):
        r, _ = run([(IN_AIR, CUBE_UP, 20), (ON_BENCH, CUBE_IN, 20), (SEATED, CUBE_IN, 40)])
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_lid_resting_on_the_cube_is_a_breach(self):
        on_cube = lid_at(L.CUBE_X, L.CUBE_Y, L.TOP + L.CUBE_EDGE)
        r, _ = run([(on_cube, CUBE_HOME, 5), (SEATED, CUBE_IN, 40)])
        self.assertEqual(r.score, 0.0)

    def test_lid_knocked_off_scores_zero(self):
        r, _ = run([(lid_at(L.JAR_X + 0.10, L.JAR_Y, L.TOP), CUBE_HOME, 30)])
        self.assertEqual(r.score, 0.0)

    def test_crooked_lid_on_rim_is_not_a_breach(self):
        crooked = lid_at(L.JAR_X + 0.015, L.JAR_Y, L.RIM_Z)
        r, _ = run([(IN_AIR, CUBE_IN, 40), (crooked, CUBE_IN, 40)])
        self.assertEqual(r.score, 0.5)

    def test_success_needs_stillness(self):
        moving = cube_at(L.JAR_X, L.JAR_Y, L.TOP, vel=(0, 0, -0.2))
        r, _ = run([(IN_AIR, CUBE_IN, 20), (SEATED, moving, 5)])
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        t = Tracker()
        for _ in range(30):
            t.update(ON_BENCH, CUBE_HOME)
        self.assertTrue(t.breach)
        r = t.update(SEATED, CUBE_HOME)  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(r.score, 0.0)

    def test_putting_things_back_slowly_does_not_reset(self):
        t = Tracker()
        for lid in [SEATED] + glide(SEATED, IN_AIR, 30) + glide(IN_AIR, SEATED, 30):
            t.update(lid, CUBE_HOME)
        self.assertTrue(t.lifted)  # back at the start, but it got there gradually: same episode


if __name__ == "__main__":
    unittest.main()
