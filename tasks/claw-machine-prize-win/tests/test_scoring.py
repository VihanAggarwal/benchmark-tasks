"""Offline tests for layout.py, scoring.py and task.py: the machine's program driven by synthetic joystick and button
readings, the scorer fed synthetic prize positions and arm snapshots (rigstate.Arm); no simulator.

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

AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
TILT = math.radians(15.0)
HZ = int(L.CONTROL_HZ)
PRIZES = [c.copy() for c in L.CAPSULES]


def steer(p, fb_s=0.0, lr_s=0.0, press=True):
    """Hold the stick forward ``fb_s`` s (negative: back), then right ``lr_s`` s (negative: left), then press."""
    for _ in range(int(abs(fb_s) * HZ)):
        p.step(math.copysign(TILT, fb_s), 0.0, 0.0, PRIZES)
    for _ in range(int(abs(lr_s) * HZ)):
        p.step(0.0, math.copysign(TILT, lr_s), 0.0, PRIZES)
    p.step(0.0, 0.0, 0.0, PRIZES)
    if press:
        p.step(0.0, 0.0, 0.005, PRIZES)
    events = []
    for _ in range(40 * HZ):
        events += p.step(0.0, 0.0, 0.0, PRIZES)
    return events


def to_red():
    d = L.CAPSULES[0] - L.HOME
    return d[0] / L.SPEED, -d[1] / L.SPEED


def prize_states(z=L.BASE_H + L.PRIZE / 2, first=None):
    out = {}
    for i, (n, c) in enumerate(zip(S.CAPSULES, L.CAPSULES)):
        p = first if (i == 0 and first is not None) else [c[0], c[1], z]
        out[n] = types.SimpleNamespace(pos=L.world(p))
    return out


class TestProgram(unittest.TestCase):
    def test_the_stick_drives_the_claw_at_its_speed(self):
        p = S.Program()
        for _ in range(HZ):
            p.step(TILT, 0.0, 0.0, PRIZES)
        self.assertAlmostEqual(p.pos[0] - L.HOME[0], L.SPEED, places=6)
        for _ in range(HZ):
            p.step(0.0, TILT, 0.0, PRIZES)
        self.assertAlmostEqual(p.pos[1] - L.HOME[1], -L.SPEED, places=6)

    def test_a_tilt_under_the_switch_does_nothing(self):
        p = S.Program()
        for _ in range(2 * HZ):
            p.step(math.radians(6.0), math.radians(6.0), 0.0, PRIZES)
        self.assertTrue(np.allclose(p.pos, L.HOME))
        self.assertIsNone(p.timer)

    def test_over_a_prize_the_claw_grabs_it_and_brings_it_back(self):
        p = S.Program()
        ev = steer(p, *to_red())
        self.assertIn("grab 0", ev)
        self.assertIn("release", ev)
        self.assertTrue(p.near)
        self.assertEqual(p.state, "done")

    def test_short_of_it_the_claw_closes_on_nothing(self):
        p = S.Program()
        fb, lr = to_red()
        ev = steer(p, fb - 0.8, lr)  # 40 mm short
        self.assertNotIn("grab 0", ev)
        self.assertFalse(p.near)
        self.assertTrue(p.dropped)

    def test_a_near_miss(self):
        p = S.Program()
        fb, lr = to_red()
        steer(p, fb - 0.36, lr)  # 18 mm short
        self.assertTrue(p.near)
        self.assertIsNone(p.grabbed)

    def test_the_timer_drops_the_claw(self):
        p = S.Program()
        fb, lr = to_red()
        steer(p, fb, lr, press=False)
        self.assertTrue(p.dropped)
        self.assertEqual(p.grabbed, 0)

    def test_one_credit_one_drop(self):
        p = S.Program()
        steer(p, 1.0, 0.0)
        before = p.pos.copy()
        for _ in range(HZ):
            p.step(TILT, 0.0, 0.005, PRIZES)
        self.assertTrue(np.allclose(p.pos, before))
        self.assertEqual(p.state, "done")


class TestScoring(unittest.TestCase):
    def tracker_with(self, program, states, arms=AWAY):
        t = S.Tracker()
        return t.update(states, arms, program)

    def test_a_prize_through_the_chute_is_full(self):
        p = S.Program()
        r = self.tracker_with(p, prize_states(first=[0.055, 0.075, 0.03]))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_near_miss_and_attempt_tiers(self):
        p = S.Program()
        p.near, p.dropped, p.moved = True, True, 0.15
        self.assertEqual(self.tracker_with(p, prize_states()).score, 0.5)
        p.near = False
        self.assertEqual(self.tracker_with(p, prize_states()).score, 0.25)
        p.moved = 0.02
        self.assertEqual(self.tracker_with(p, prize_states()).score, 0.0)

    def test_a_hand_in_the_cabinet_scores_nothing(self):
        inside = L.world([0.15, 0.0, 0.20])
        arms = {"left": rigstate.Arm(inside + [0, 0, 0.013], inside, 0.5), "right": AWAY["right"]}
        p = S.Program()
        r = self.tracker_with(p, prize_states(first=[0.055, 0.075, 0.03]), arms)
        self.assertTrue(r.details["hand"])
        self.assertEqual(r.score, 0.0)

    def test_unreadable_state(self):
        r = S.Tracker().update({}, AWAY, S.Program())
        self.assertFalse(r.details["valid_state"])


class TestLayout(unittest.TestCase):
    def test_the_prizes_clear_the_chute_and_each_other(self):
        for i, c in enumerate(L.CAPSULES):
            self.assertFalse(L.in_chute(c))
            for d in L.CAPSULES[i + 1:]:
                self.assertGreater(float(np.linalg.norm(c - d)), L.PRIZE + 0.01)

    def test_the_claw_reaches_every_prize_and_drops_above_them(self):
        for c in L.CAPSULES:
            self.assertTrue(L.CLAW_X[0] <= c[0] <= L.CLAW_X[1] and L.CLAW_Y[0] <= c[1] <= L.CLAW_Y[1])
        self.assertGreater(L.DROP_TO, L.BASE_H + L.PRIZE)

    def test_the_controls_are_outside_the_cabinet(self):
        self.assertFalse(L.in_cabinet(T.bat_centre()))
        self.assertFalse(L.in_cabinet(T.button_tcp()))


class TestDemo(unittest.TestCase):
    def test_variants(self):
        for v in ("", *T.VARIANTS):
            self.assertEqual(set(T.plans(v)), {"left", "right"})
        with self.assertRaises(ValueError):
            T.plans("nope")


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
