"""Offline tests for layout.py, scoring.py and task.py: synthetic poses of the pruner, the battery and the scabbard
and synthetic arm snapshots (rigstate.Arm); no simulator.

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

ORIGIN = np.array([L.PRUNER_AT[0], L.PRUNER_AT[1], L.TOP])
AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.30, L.TOP + 0.30]), np.array([0.10, s * 0.30, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
Q0 = (0.0, 0.0, 0.0, 1.0)


def yaw_q(deg):
    a = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(a), math.cos(a))


def st(pos, quat=Q0, vel=(0.0, 0.0, 0.0)):
    return types.SimpleNamespace(origin=np.array(pos, dtype=float), quat_xyzw=np.array(quat), lin_vel=np.array(vel))


BATTERY_HOME = ORIGIN + L.BAT_HOME
BATTERY_DOWN = L.at(0.20, 0.30, L.BAT_Z / 2)
GUARD_START = L.at(*L.GUARD_START, L.GUARD_H / 2)
GUARD_SEATED = ORIGIN + [0.0, L.GUARD_SEAT_Y, L.BAR_Z]


def states(battery=BATTERY_HOME, guard=GUARD_START, pruner=ORIGIN, pruner_q=Q0, moving=False):
    v = (0.02, 0.0, 0.0) if moving else (0.0, 0.0, 0.0)
    return {S.PRUNER: st(pruner, pruner_q, v), S.BATTERY: st(battery), S.GUARD: st(guard)}


def hand_at(p, side="left"):
    p = np.asarray(p, dtype=float)
    arms = dict(AWAY)
    arms[side] = rigstate.Arm(p + [0, 0, 0.013], p, 0.0)
    return arms


def run(seq):
    t = S.Tracker()
    r = None
    for s, arms in seq:
        r = t.update(s, arms)
    return r


class TestScoring(unittest.TestCase):
    def test_untouched_start_is_zero(self):
        self.assertEqual(run([(states(), AWAY)] * 5).score, 0.0)

    def test_battery_out_and_down_is_a_quarter(self):
        r = run([(states(battery=BATTERY_DOWN), AWAY)])
        self.assertTrue(r.details["released"])
        self.assertEqual(r.score, 0.25)

    def test_battery_only_part_way_out_is_not_out(self):
        r = run([(states(battery=BATTERY_HOME + [0.0, 0.040, 0.0]), AWAY)])
        self.assertFalse(r.details["battery_out"])
        self.assertEqual(r.score, 0.0)

    def test_open_fingers_near_the_trigger_are_a_foul(self):
        (x0, x1), (y0, y1), (z0, z1) = L.TRIGGER
        p = ORIGIN + L.handle_to_pruner([x0 - 0.030, (y0 + y1) / 2, (z0 + z1) / 2])  # TCP line clear, the pads not
        arms = dict(AWAY)
        arms["left"] = rigstate.Arm(p + [0, 0, 0.013], p, 1.0)
        self.assertIn("trigger", run([(states(), arms)]).details["fouls"])

    def test_battery_still_held_is_not_released(self):
        r = run([(states(battery=BATTERY_DOWN), hand_at(BATTERY_DOWN))])
        self.assertEqual(r.score, 0.0)

    def test_scabbard_part_way_on_is_half(self):
        g = ORIGIN + [0.0, L.BAR_Y[0] + 0.05 - L.GUARD_L / 2, L.BAR_Z]  # covers the outer 50 mm
        r = run([(states(battery=BATTERY_DOWN, guard=g), AWAY)])
        self.assertAlmostEqual(r.details["covered_mm"], 50.0, places=1)
        self.assertEqual(r.score, 0.5)

    def test_seated_and_clear_is_full(self):
        r = run([(states(battery=BATTERY_DOWN, guard=GUARD_SEATED), AWAY)])
        self.assertTrue(r.details["seated"])
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_seated_with_a_hand_on_it_is_three_quarters(self):
        r = run([(states(battery=BATTERY_DOWN, guard=GUARD_SEATED), hand_at(GUARD_SEATED + [0, 0, 0.012], "right"))])
        self.assertEqual(r.score, 0.75)

    def test_moving_is_not_clear(self):
        self.assertEqual(run([(states(battery=BATTERY_DOWN, guard=GUARD_SEATED, moving=True), AWAY)]).score, 0.75)

    def test_scabbard_without_the_battery_out_scores_nothing(self):
        self.assertEqual(run([(states(guard=GUARD_SEATED), AWAY)]).score, 0.0)

    def test_hand_at_the_bar_with_the_battery_in_is_a_foul(self):
        near_bar = ORIGIN + [0.0, -0.13, L.BAR_Z + 0.015]
        r = run([(states(), hand_at(near_bar)), (states(battery=BATTERY_DOWN, guard=GUARD_SEATED), AWAY)])
        self.assertIn("bar", r.details["fouls"])
        self.assertEqual(r.score, 0.0)

    def test_scabbard_fitted_before_the_battery_is_a_foul(self):
        r = run([(states(guard=GUARD_SEATED), AWAY), (states(battery=BATTERY_DOWN, guard=GUARD_SEATED), AWAY)])
        self.assertIn("order", r.details["fouls"])
        self.assertEqual(r.score, 0.0)

    def test_hand_at_a_bare_tooth_is_a_foul(self):
        tooth = ORIGIN + [L.BAR_X + 0.005, -0.12, L.BAR_Z]
        r = run([(states(battery=BATTERY_DOWN), hand_at(tooth)), (states(battery=BATTERY_DOWN, guard=GUARD_SEATED),
                                                                    AWAY)])
        self.assertIn("tooth", r.details["fouls"])
        self.assertEqual(r.score, 0.0)

    def test_hand_at_a_bare_nose_tooth_is_a_foul(self):
        nose = ORIGIN + [0.0, L.BAR_Y[0] - 0.006, L.BAR_Z]  # just off the nose's tip
        r = run([(states(battery=BATTERY_DOWN), hand_at(nose))])
        self.assertIn("tooth", r.details["fouls"])

    def test_trigger_after_the_battery_is_out_is_not_a_foul(self):
        (x0, x1), (y0, y1), (z0, z1) = L.TRIGGER
        trig = ORIGIN + L.handle_to_pruner([x0 - 0.005, (y0 + y1) / 2, (z0 + z1) / 2])
        self.assertNotIn("trigger", run([(states(battery=BATTERY_DOWN), hand_at(trig))]).details["fouls"])

    def test_covered_teeth_are_not_bare(self):
        tooth = ORIGIN + [L.BAR_X + 0.005, -0.12, L.BAR_Z]
        r = run([(states(battery=BATTERY_DOWN, guard=GUARD_SEATED), hand_at(tooth))])
        self.assertNotIn("tooth", r.details["fouls"])

    def test_trigger_is_a_foul(self):
        (x0, x1), (y0, y1), (z0, z1) = L.TRIGGER
        trig = ORIGIN + L.handle_to_pruner([x0 - 0.005, (y0 + y1) / 2, (z0 + z1) / 2])
        self.assertIn("trigger", run([(states(), hand_at(trig))]).details["fouls"])

    def test_scabbard_off_line_does_not_count(self):
        g = GUARD_SEATED + [0.010, 0.0, 0.0]
        r = run([(states(battery=BATTERY_DOWN, guard=g), AWAY)])
        self.assertFalse(r.details["on"])
        self.assertEqual(r.score, 0.25)

    def test_works_in_the_pruners_frame(self):
        q = yaw_q(20.0)
        rot = S.quat_matrix(q)
        moved = ORIGIN + [0.03, -0.02, 0.0]
        g = moved + rot @ np.array([0.0, L.GUARD_SEAT_Y, L.BAR_Z])
        s = {S.PRUNER: st(moved, q), S.BATTERY: st(BATTERY_DOWN), S.GUARD: st(g, q)}
        self.assertEqual(run([(s, AWAY)]).score, 1.0)

    def test_unreadable_state(self):
        self.assertFalse(S.Tracker().update({}, AWAY).details["valid_state"])


class TestLatch(unittest.TestCase):
    def test_a_momentary_press_at_home_is_caught_again(self):
        self.assertEqual(S.latch_step(True, True, 0.0, 0.0), "free")
        self.assertEqual(S.latch_step(False, False, 0.004, 0.004), "catch")

    def test_held_pressed_it_stays_free(self):
        self.assertEqual(S.latch_step(False, True, 0.030, 0.030), "")

    def test_drawn_past_the_hooks_it_stays_free(self):
        self.assertEqual(S.latch_step(False, False, 0.040, 0.040), "")

    def test_pushed_back_home_it_clicks_in(self):
        self.assertEqual(S.latch_step(False, False, 0.001, 0.070), "click")


class TestScabbardCatch(unittest.TestCase):
    """The catch (task._guard_home, scoring.catch_ready) clicks only after the scabbard is slid over the bar from
    its tip to within SEAT_TOL of the seat; a scabbard aligned anywhere else never triggers it."""

    def guard_at(self, y, x=0.0, z=None):
        return ORIGIN + [x, y, L.BAR_Z if z is None else z]

    def feed(self, ys, **kw):
        flag, clicks = False, []
        for y in ys:
            covered, gy = S.guard_on(S.body(st(ORIGIN)), S.body(st(self.guard_at(y, **kw))))
            ready, flag = S.catch_ready(covered, gy, flag)
            clicks.append(ready)
        return clicks

    def test_slid_on_from_the_tip_clicks_at_the_seat(self):
        seat = L.GUARD_SEAT_Y
        self.assertEqual(self.feed([seat - 0.050, seat - 0.020, seat - 0.006, seat - 0.002]),
                         [False, False, False, True])

    def test_aligned_beyond_the_housing_never_clicks(self):
        for y in (L.GUARD_SEAT_Y + 0.15, L.GUARD_SEAT_Y + 0.25, 0.10, 0.30):
            self.assertEqual(self.feed([y, y, y]), [False] * 3, y)

    def test_slid_past_the_seat_does_not_click(self):
        seat = L.GUARD_SEAT_Y
        self.assertEqual(self.feed([seat - 0.030, seat + 0.010, seat + 0.050]), [False] * 3)

    def test_appearing_at_the_seat_without_a_slide_does_not_click(self):
        self.assertEqual(self.feed([L.GUARD_SEAT_Y, L.GUARD_SEAT_Y]), [False, False])

    def test_off_the_bars_line_does_not_click(self):
        seat = L.GUARD_SEAT_Y
        self.assertEqual(self.feed([seat - 0.030, seat - 0.002], x=0.010), [False, False])
        self.assertEqual(self.feed([seat - 0.030, seat - 0.002], z=L.BAR_Z + 0.010), [False, False])

    def test_leaving_the_line_resets_the_slide(self):
        seat = L.GUARD_SEAT_Y
        flag = False
        for y, x in ((seat - 0.030, 0.0), (seat - 0.030, 0.020), (seat + 0.15, 0.0), (seat - 0.002, 0.0)):
            covered, gy = S.guard_on(S.body(st(ORIGIN)), S.body(st(self.guard_at(y, x=x))))
            ready, flag = S.catch_ready(covered, gy, flag)
            if x:
                self.assertFalse(flag)
        # the last reading is at the seat, but the scabbard was never slid on again after it left the line
        self.assertFalse(ready)

    def test_task_catch_uses_the_same_rule(self):
        seat = L.GUARD_SEAT_Y
        beyond = [T._guard_home(states(battery=BATTERY_DOWN, guard=self.guard_at(seat + 0.2))) for _ in range(3)]
        self.assertEqual(beyond, [False] * 3)
        slid = [T._guard_home(states(battery=BATTERY_DOWN, guard=self.guard_at(y)))
                for y in (seat - 0.050, seat - 0.020, seat - 0.001)]
        self.assertEqual(slid, [False, False, True])


class TestLayout(unittest.TestCase):
    def test_scabbard_clears_the_chain(self):
        self.assertGreater(L.GUARD_W - 2 * L.GUARD_SIDE, 2 * L.BAR_X + 0.004)
        self.assertGreater(L.GUARD_H - 2 * L.GUARD_PLATE, L.BAR_T + 0.004)

    def test_scabbard_covers_the_whole_bar_when_seated(self):
        self.assertGreaterEqual(L.GUARD_L, L.BAR_Y[1] - L.BAR_Y[0])

    def test_hooks_free_before_full_travel(self):
        self.assertLess(L.HOOK_FREE, L.HOOK_TRAVEL)


class TestDemo(unittest.TestCase):
    def test_variants(self):
        for v in ("", *T.VARIANTS):
            self.assertTrue(T.plans(v))
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_battery_pinch_presses_both_hooks(self):
        self.assertLess(-T.PINCH, L.BAT_X / 2 + L.HOOK_SIZE[0] - L.HOOK_FREE)


class TestMeta(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], T.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
