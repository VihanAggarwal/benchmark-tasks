"""Offline tests for scoring.py and task.py: synthetic tyre states and gripper snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import subprocess
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
    arms_valid,
    dot_angle_deg,
    heading_deg,
    point_distance,
    pose_seated,
    rule_error_deg,
    tilt_deg,
    valid,
    wrap180,
    zone_gap,
    zone_radius,
)

HZ = 20.0
FULL = 2 * L.STILL_WINDOW_STEPS + 1  # 21 steps: the 1 s of history 'still' needs
TT_W = L.TOP + L.TT_TOP


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def raw(origin, q, vel=(0.0, 0.0, 0.0)):
    """A tyre state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o + np.array([0, 0, 0.0113]),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.032, high=o + 0.032)


def on_wheel(wheel, yaw_deg, dx=0.0, dy=0.0, up=0.0, pitch=0.0, roll=0.0):
    """A tyre lying on ``wheel``'s turntable (its centre dx, dy off the axis, underside ``up`` above the turntable's
    top), turned so its red dot points at ``yaw_deg``."""
    x, y = L.wheel_axis(wheel)
    return raw((x + dx, y + dy, TT_W + up), quat(yaw_deg, pitch, roll))


def red_at(wheel, angle, **kw):
    return on_wheel(wheel, angle - L.RED_LOCAL_DEG, **kw)


def yellow_at(wheel, angle, **kw):
    return on_wheel(wheel, angle - L.YELLOW_LOCAL_DEG, **kw)


def at_start(name, dx=0.0, dy=0.0, dyaw=0.0):
    x, y, yaw = L.TYRES[name]
    return raw((x + dx, y + dy, L.TOP), quat(yaw + dyaw))


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


PARKED = {"left": arm((0.20, 0.24, L.TOP + 0.15), 0.81), "right": arm((0.20, -0.24, L.TOP + 0.15), 0.81)}
HOLD_TCP_Z = TT_W + 0.001 + 0.016  # as the demo holds a tyre at the seat


def holding(wheel, side):
    """``side``'s gripper closed on a tyre seated on ``wheel`` (fingers at the tyre's 64 mm), the other parked."""
    x, y = L.wheel_axis(wheel)
    arms = dict(PARKED)
    arms[side] = arm((x, y, HOLD_TCP_Z), L.GRIP_WIDTH / L.FULL_GAP)
    return arms


A, B = L.MARKED_WHEEL, L.PLAIN_WHEEL
RIGHT = {"tyre_a": red_at(A, L.MARK_ANGLE), "tyre_b": yellow_at(B, L.VALVE_ANGLE)}
START = {n: at_start(n) for n in L.TYRE_NAMES}


def feed(tr, states, arms=PARKED):
    r = None
    for st in states:
        r = tr.update(st, arms)
    return r


def settled(tyres, arms=PARKED, n=FULL):
    tr = Tracker(HZ)
    return tr, feed(tr, [tyres] * n, arms)


def scored(tyres, arms=PARKED):
    return settled(tyres, arms)[1].score


def both(ta, tb):
    return {"tyre_a": ta, "tyre_b": tb}


class Geometry(unittest.TestCase):
    def test_layout_numbers(self):
        self.assertAlmostEqual(L.required_yaw(A), 120.0)  # red dot to the white mark at 120
        self.assertAlmostEqual(L.required_yaw(B), -100.0)  # yellow (local +100) to the valve at 0
        self.assertAlmostEqual(L.TYRE_R_IN - L.WHEEL_RADIUS, 0.005)
        self.assertLess(L.VALVE_R + L.VALVE_RADIUS, L.TYRE_R_IN - 0.009)  # the valve passes through the bore
        self.assertLess(L.TURNTABLE_RADIUS, L.TYRE_R_OUT - 0.004)
        self.assertGreater(L.WHEEL_TOP, L.TT_TOP + L.TYRE_H)
        self.assertAlmostEqual(L.DOT_R * math.radians(L.ANGLE_TOL_DEG) * 1000, 4.8, places=1)
        self.assertAlmostEqual(L.TREAD_DIP * 1000, 0.69, places=2)
        self.assertAlmostEqual(L.GRIP_WIDTH * 1000, 63.8, places=1)

    def test_start_layout(self):
        for n, (x, y, yaw) in L.TYRES.items():
            t = START[n]
            self.assertTrue(valid(t))
            self.assertTrue(Tracker.at_start(n, t))
            self.assertAlmostEqual(heading_deg(t), yaw, places=9)
            self.assertAlmostEqual(dot_angle_deg(t, "red"), yaw, places=9)
            self.assertAlmostEqual(wrap180(dot_angle_deg(t, "yellow") - yaw), 100.0, places=9)
        # neither start turn is within 10 degrees of an answer on either wheel (60 degrees off)
        self.assertAlmostEqual(rule_error_deg(START["tyre_a"], A), 60.0, places=6)
        self.assertAlmostEqual(rule_error_deg(START["tyre_b"], B), 60.0, places=6)
        for n in L.TYRE_NAMES:
            for w in L.WHEEL_NAMES:
                self.assertGreater(rule_error_deg(START[n], w), L.ANGLE_TOL_DEG + 20.0)

    def test_dot_angle_is_about_the_tyres_own_centre(self):
        t = red_at(A, 120.0, dx=0.004, dy=-0.003)
        self.assertAlmostEqual(dot_angle_deg(t, "red"), 120.0, places=9)
        self.assertAlmostEqual(rule_error_deg(t, A), 0.0, places=9)

    def test_tilt_and_upside_down(self):
        self.assertAlmostEqual(tilt_deg(red_at(A, 120.0, pitch=4.0)), 4.0, places=6)
        self.assertAlmostEqual(tilt_deg(red_at(A, 120.0, roll=180.0)), 180.0, places=6)

    def test_ring_distance(self):
        t = red_at(A, 120.0)
        x, y = L.wheel_axis(A)
        self.assertEqual(point_distance((x + 0.0275, y, TT_W + 0.01), t), 0.0)  # inside the rubber
        self.assertAlmostEqual(point_distance((x, y, TT_W + 0.01), t), L.TYRE_R_IN)  # on the axis: the bore
        self.assertAlmostEqual(point_distance((x + 0.040, y, TT_W + 0.01), t), 0.040 - L.TYRE_R_OUT)
        self.assertAlmostEqual(point_distance((x + 0.0275, y, TT_W + L.TYRE_TOP_H + 0.02), t), 0.02)

    def test_finger_zone(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135))
        t = red_at(A, 120.0)
        self.assertLess(zone_gap(holding(A, "left")["left"], t), 0.0)
        self.assertGreater(zone_gap(PARKED["left"], t), 0.05)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(scored(START), 0.0)

    def test_both_right_scores_one(self):
        tr, r = settled(RIGHT)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["n_correct"], 2)

    def test_tyres_swapped_between_the_wheels_also_one(self):
        self.assertEqual(scored(both(yellow_at(B, 0.0), red_at(A, 120.0))), 1.0)

    def test_one_wheel_right(self):
        self.assertEqual(scored(both(red_at(A, 120.0), yellow_at(B, 40.0))), 0.5)
        self.assertEqual(scored(both(red_at(A, 160.0), yellow_at(B, 0.0))), 0.5)
        r = settled(both(red_at(A, 120.0), at_start("tyre_b")))[1]  # the other tyre left on the table
        self.assertEqual((r.score, r.success), (0.5, False))

    def test_seated_but_no_wheel_right(self):
        self.assertEqual(scored(both(red_at(A, 0.0), yellow_at(B, 120.0))), 0.25)
        self.assertEqual(scored(both(red_at(A, 30.0), at_start("tyre_b"))), 0.25)

    def test_no_tyre_seated(self):
        self.assertEqual(scored(both(at_start("tyre_a", dx=0.03), at_start("tyre_b", dy=0.02))), 0.0)

    def test_every_tier_reachable_each_state_one_score(self):
        seen = {scored(s) for s in (RIGHT, both(red_at(A, 120.0), yellow_at(B, 50.0)),
                                    both(red_at(A, 0.0), yellow_at(B, 90.0)), START)}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})

    def test_angle_window_edges(self):
        for d in (-9.9, 9.9):
            self.assertEqual(scored(both(red_at(A, 120.0 + d), yellow_at(B, -d))), 1.0)
        for d in (-10.1, 10.1):
            self.assertEqual(scored(both(red_at(A, 120.0 + d), yellow_at(B, 0.0))), 0.5)
            self.assertEqual(scored(both(red_at(A, 120.0), yellow_at(B, d))), 0.5)

    def test_angle_wraps_round(self):
        self.assertEqual(scored(both(red_at(A, 120.0 - 360.0), yellow_at(B, 355.0))), 1.0)

    def test_seat_height_edges(self):
        self.assertEqual(scored(both(red_at(A, 120.0, up=0.0029), yellow_at(B, 0.0, up=-0.0029))), 1.0)
        self.assertEqual(scored(both(red_at(A, 120.0, up=0.0031), yellow_at(B, 0.0))), 0.5)
        self.assertEqual(scored(both(red_at(A, 120.0, up=-0.0031), yellow_at(B, 0.0))), 0.5)  # sunk below the top

    def test_let_go_margin(self):
        """An open gripper straight above a correctly turned tyre: its finger zone 9 mm from the ring is not let go,
        11 mm is."""
        x, y = L.wheel_axis(A)
        for gap, want in ((0.009, 0.5), (0.011, 1.0)):
            tips_z = TT_W + L.TYRE_TOP_H + gap
            arms = dict(PARKED)
            arms["left"] = arm((x + 0.0275, y, tips_z + 0.013), 0.81)
            self.assertAlmostEqual(zone_gap(arms["left"], RIGHT["tyre_a"]) + zone_radius(arms["left"]), gap)
            arms["left"] = arm((x + 0.0275, y, tips_z + 0.013 + zone_radius(arms["left"])), 0.81)
            self.assertAlmostEqual(zone_gap(arms["left"], RIGHT["tyre_a"]), gap)
            self.assertEqual(scored(RIGHT, arms), want)

    def test_tilt_edges(self):
        self.assertEqual(scored(both(red_at(A, 120.0, pitch=4.9), yellow_at(B, 0.0, roll=-4.9))), 1.0)
        self.assertEqual(scored(both(red_at(A, 120.0, pitch=5.1), yellow_at(B, 0.0))), 0.5)

    def test_axis_offset(self):
        self.assertEqual(scored(both(red_at(A, 120.0, dx=0.005), yellow_at(B, 0.0, dy=-0.005))), 1.0)
        self.assertEqual(scored(both(red_at(A, 120.0, dx=0.0065), yellow_at(B, 0.0))), 0.5)

    def test_score_is_idempotent(self):
        tr, r = settled(RIGHT)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(RIGHT, PARKED).score, 1.0)
        self.assertEqual(tr.steps, n)


class Shortcuts(unittest.TestCase):
    """The spec's shortcut checks and further unintended strategies."""

    def test_yellow_to_the_valve_on_both_wheels(self):  # the rule without its precedence
        self.assertEqual(scored(both(yellow_at(A, 0.0), yellow_at(B, 0.0))), 0.5)

    def test_red_to_the_valve_on_both_wheels(self):
        self.assertEqual(scored(both(red_at(A, 0.0), red_at(B, 0.0))), 0.25)

    def test_yellow_to_the_mark_red_to_the_valve(self):  # the colours crossed
        self.assertEqual(scored(both(yellow_at(A, 120.0), red_at(B, 0.0))), 0.25)

    def test_mounted_at_their_start_turns(self):
        self.assertEqual(scored(both(on_wheel(A, 60.0), on_wheel(B, -40.0))), 0.25)

    def test_right_but_one_still_held(self):
        self.assertEqual(scored(RIGHT, holding(B, "right")), 0.5)
        self.assertEqual(scored(RIGHT, holding(A, "left")), 0.5)
        self.assertEqual(scored(RIGHT), 1.0)

    def test_both_held_in_place(self):
        arms = {"left": holding(A, "left")["left"], "right": holding(B, "right")["right"]}
        self.assertEqual(scored(RIGHT, arms), 0.0)

    def test_open_gripper_resting_on_a_tyre(self):
        x, y = L.wheel_axis(A)
        arms = dict(PARKED)
        arms["left"] = arm((x + 0.0275, y, TT_W + L.TYRE_TOP_H + 0.014), 0.81)  # pad tips 1 mm over its sidewall
        self.assertEqual(scored(RIGHT, arms), 0.5)

    def test_dropped_beside_the_wheels(self):
        sa = raw((0.34, 0.15 + 0.06, L.TOP), quat(120.0))
        sb = raw((0.34 - 0.06, -0.15, L.TOP), quat(-100.0))
        self.assertEqual(scored(both(sa, sb)), 0.0)

    def test_stacked_on_a_wheel_top(self):
        x, y = L.wheel_axis(A)
        top = raw((x, y, L.TOP + L.WHEEL_TOP), quat(120.0))  # sitting on the wheel's top face, not round it
        self.assertEqual(scored(both(top, yellow_at(B, 0.0))), 0.5)

    def test_two_tyres_on_one_wheel(self):
        lower = red_at(A, 120.0)
        upper = red_at(A, 120.0, up=L.TYRE_H)  # lying on top of the first: its underside 22 mm up
        self.assertEqual(scored(both(lower, upper)), 0.5)
        r = settled(both(lower, red_at(A, 120.0, up=0.001)))[1]  # a state no simulator gives: both down on A
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.details["correct"][A])

    def test_upside_down_with_the_dots_underneath(self):
        x, y = L.wheel_axis(A)
        flipped = raw((x, y, TT_W + L.TYRE_H), quat(120.0, 0.0, 180.0))  # its local origin now on top
        self.assertEqual(scored(both(flipped, yellow_at(B, 0.0))), 0.5)

    def test_leaning_on_the_wheel(self):
        self.assertEqual(scored(both(red_at(A, 120.0, up=0.004, pitch=12.0), yellow_at(B, 0.0))), 0.5)

    def test_kept_spinning_on_the_turntable(self):
        tr = Tracker(HZ)
        r = None
        for i in range(60):  # turned 3 degrees a step through the window: never still
            r = tr.update(both(red_at(A, 100.0 + 3.0 * (i % 15)), yellow_at(B, 0.0)), PARKED)
        self.assertEqual(r.score, 0.5)

    def test_random_turns_rarely_score(self):
        rng = np.random.default_rng(1)
        hits = 0
        n = 4000
        for yaw in rng.uniform(-180.0, 180.0, n):
            hits += rule_error_deg(on_wheel(A, yaw), A) <= L.ANGLE_TOL_DEG
        self.assertAlmostEqual(hits / n, 20.0 / 360.0, delta=0.012)


class Stillness(unittest.TestCase):
    def test_needs_one_second(self):
        tr = Tracker(HZ)
        r = feed(tr, [RIGHT] * (FULL - 1))
        self.assertEqual(r.score, 0.0)
        r = tr.update(RIGHT, PARKED)
        self.assertEqual(r.score, 1.0)

    def test_moving_tyre_is_not_seated(self):
        tr = Tracker(HZ)
        states = [both(red_at(A, 120.0, dx=0.0002 * (i % 2)), yellow_at(B, 0.0)) for i in range(FULL)]
        r = feed(tr, states)  # 0.2 mm back and forth each step: 2 mm of path per 0.5 s
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["tyres"]["tyre_a"]["why_not_seated"], "still_moving")

    def test_small_jitter_is_still(self):
        tr = Tracker(HZ)
        states = [both(red_at(A, 120.0, dx=0.00005 * (i % 2)), yellow_at(B, 0.0)) for i in range(FULL)]
        self.assertEqual(feed(tr, states).score, 1.0)

    def test_slow_turn_over_two_degrees_is_not_still(self):
        tr = Tracker(HZ)
        states = [both(red_at(A, 118.0 + 0.11 * i), yellow_at(B, 0.0)) for i in range(FULL)]
        self.assertEqual(feed(tr, states).score, 0.5)

    def test_comes_to_rest(self):
        tr = Tracker(HZ)
        moving = [both(red_at(A, 120.0, up=0.03 - 0.0015 * i), yellow_at(B, 0.0)) for i in range(20)]
        feed(tr, moving)
        r = feed(tr, [RIGHT] * FULL)
        self.assertEqual(r.score, 1.0)


class Robustness(unittest.TestCase):
    def test_arms_unreadable_means_not_let_go(self):
        tr, r = settled(RIGHT, None)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.arms_unavailable, FULL)
        self.assertEqual(r.details["tyres"]["tyre_a"]["why_not_seated"], "arms_unreadable")
        bad = {"left": arm((np.nan, 0, 0)), "right": PARKED["right"]}
        self.assertFalse(arms_valid(bad))
        self.assertEqual(scored(RIGHT, bad), 0.0)
        self.assertFalse(arms_valid({"left": PARKED["left"]}))
        self.assertEqual(tr.update(RIGHT, PARKED).score, 1.0)  # the arms come back: judged again

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(RIGHT)
        n_hist = [list(h) for h in tr.history.values()]
        nan = raw((np.nan, 0.0, 0.0), quat(0.0))
        inf = raw((0.3, 0.1, np.inf), quat(0.0))
        zero_q = raw(RIGHT["tyre_a"].origin, (0.0, 0.0, 0.0, 0.0))
        no_origin = SimpleNamespace(quat_xyzw=quat(0.0), pos=np.zeros(3))
        for bad in (nan, inf, zero_q, no_origin, None):
            r = tr.update(both(bad, RIGHT["tyre_b"]), PARKED)
            self.assertEqual(r.score, 0.0)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(tr.update({"tyre_a": RIGHT["tyre_a"]}, PARKED).score, 0.0)  # a tyre missing
        self.assertEqual(tr.update(None, PARKED).score, 0.0)
        self.assertEqual([list(h) for h in tr.history.values()], n_hist)
        self.assertEqual(tr.update(RIGHT, PARKED).score, 1.0)

    def test_unnormalised_quaternion(self):
        s = both(raw(RIGHT["tyre_a"].origin, 3.0 * quat(120.0)), RIGHT["tyre_b"])
        self.assertEqual(scored(s), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker(HZ)
        for _ in range(300):
            s = {n: raw(rng.uniform([-0.2, -0.7, 0.0], [1.0, 0.7, 1.2]), rng.normal(size=4)) for n in L.TYRE_NAMES}
            r = tr.update(s, PARKED)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class Resets(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(RIGHT)
        r = tr.update(START, PARKED)
        self.assertEqual((tr.steps, r.score), (1, 0.0))

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(RIGHT)
        tr.update(both(at_start("tyre_a", 0.02, -0.02), at_start("tyre_b", -0.025, 0.01)), PARKED)
        self.assertEqual(tr.steps, 1)

    def test_one_tyre_put_back_resets(self):
        tr, _ = settled(RIGHT)
        tr.update(both(at_start("tyre_a"), RIGHT["tyre_b"]), PARKED)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(RIGHT)
        a0, a1 = np.asarray(RIGHT["tyre_a"].origin), np.asarray(START["tyre_a"].origin)
        for i in range(1, 21):
            tr.update(both(raw(a0 + (a1 - a0) * i / 20, quat(60.0)), RIGHT["tyre_b"]), PARKED)
        self.assertEqual(tr.steps, FULL + 20)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(RIGHT)
        tr.update(both(raw((0.1, 0.4, L.TOP), quat(60.0)), RIGHT["tyre_b"]), PARKED)
        self.assertEqual(tr.steps, FULL + 1)

    def test_explicit_reset(self):
        tr, _ = settled(RIGHT, None)
        tr.reset()
        self.assertEqual((tr.steps, tr.arms_unavailable), (0, 0))
        self.assertTrue(all(len(h) == 0 for h in tr.history.values()))
        self.assertTrue(all(v is None for v in tr.prev_origin.values()))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate
        cls.spec = {t.name: t for t in cls.task.TYRES}

    def sim(self, tyres):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.spec[n], (*t.origin, *t.quat_xyzw), source="sim")
                for n, t in tyres.items()}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in t.OBJECTS if o.mass is not None), ["tyre_a", "tyre_b"])
        for tyre in t.TYRES:
            self.assertAlmostEqual(tyre.mass, 0.040)
            self.assertLessEqual(tyre.mass, 0.30)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        self.assertEqual([m.name for m in t.MARKS], ["low_point_mark_a"])
        for w, tt, v in zip(t.WHEELS, t.TURNTABLES, t.VALVES):
            self.assertAlmostEqual(w.pos[2] - w.height / 2, tt.pos[2] + tt.height / 2)  # on its turntable
            self.assertAlmostEqual(v.pos[2] - v.height / 2, w.pos[2] + w.height / 2)  # on the wheel's top
            self.assertAlmostEqual(math.hypot(v.pos[0] - w.pos[0], v.pos[1] - w.pos[1]), L.VALVE_R)
            self.assertAlmostEqual(math.degrees(math.atan2(v.pos[1] - w.pos[1], v.pos[0] - w.pos[0])), L.VALVE_ANGLE)
        m, wa = t.MARKS[0], t.WHEELS[0]
        self.assertAlmostEqual(math.degrees(math.atan2(m.pos[1] - wa.pos[1], m.pos[0] - wa.pos[0])), L.MARK_ANGLE)
        self.assertAlmostEqual(math.hypot(m.pos[0] - wa.pos[0], m.pos[1] - wa.pos[1]), L.MARK_R)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                r = getattr(o, "radius", None)
                half = np.array([r, r, o.height / 2]) if r is not None else np.asarray(o.size) / 2
                lo, hi = np.asarray(o.pos) - half, np.asarray(o.pos) + half
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)
            self.assertGreater(lo[0], 0.15, o.name)

    def test_tyre_geometry_matches_the_scorer(self):
        sp = self.spec["tyre_a"]
        lo, hi = sp.local_bounds()
        np.testing.assert_allclose(lo[2], 0.0, atol=1e-12)
        np.testing.assert_allclose(hi[2], L.TYRE_TOP_H, atol=1e-12)
        self.assertLessEqual(hi[0], L.TYRE_R_OUT + 1e-12)
        ribs = [s for s in sp.shapes if s.radius == L.RIB_RADIUS]
        dots = [s for s in sp.shapes if s.radius == L.DOT_RADIUS]
        self.assertEqual((len(ribs), len(dots)), (36, 2))

        def solid(p):  # in the ring's ribs (2-D, seen from above)
            return any(math.hypot(p[0] - s.center[0], p[1] - s.center[1]) < s.radius for s in ribs)

        # the bore: nothing within 23 mm of the axis; the ring is closed (no gap a finger could enter)
        for a in np.linspace(0.0, 360.0, 721):
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            self.assertFalse(solid((0.0229 * c, 0.0229 * s)))
            self.assertTrue(solid((0.0275 * c, 0.0275 * s)))
            self.assertTrue(solid((0.0312 * c, 0.0312 * s)))  # the tread's trough is only 0.7 mm deep
        red = [s for s in dots if s.color == L.DOT_RED][0]
        yellow = [s for s in dots if s.color == L.DOT_YELLOW][0]
        self.assertAlmostEqual(math.degrees(math.atan2(red.center[1], red.center[0])), L.RED_LOCAL_DEG)
        self.assertAlmostEqual(math.degrees(math.atan2(yellow.center[1], yellow.center[0])), L.YELLOW_LOCAL_DEG)
        for d in dots:
            self.assertAlmostEqual(math.hypot(d.center[0], d.center[1]), L.DOT_R)
            self.assertTrue(solid(d.center[:2]))  # painted on the sidewall
        com = sp.center_of_mass()
        self.assertLess(math.hypot(com[0], com[1]), 0.0003)  # the dots barely move it off the axis
        for n in L.TYRE_NAMES:
            rest = self.spec[n].rest_state()
            self.assertTrue(Tracker.at_start(n, rest))
            self.assertAlmostEqual(dot_angle_deg(rest, "red"), L.TYRES[n][2], places=6)

    def test_tyre_clears_the_wheel_and_valve_when_lowered(self):
        """Seen from above, the ring's bore (23 mm) clears the wheel (18 mm) by 5 mm, and with the demo's place trim
        by 2.5 mm; the valve post and the mark are inside the wheel's outline or the bore."""
        trim = math.hypot(*self.task.PLACE_TRIM)
        self.assertGreaterEqual(L.TYRE_R_IN - trim - L.WHEEL_RADIUS, 0.0025 - 1e-9)
        self.assertLess(L.VALVE_R + L.VALVE_RADIUS, L.WHEEL_RADIUS)  # the post stands inside the wheel's outline
        self.assertLessEqual(L.VALVE_R + L.VALVE_RADIUS + trim, L.TYRE_R_IN - 0.0075 + 1e-9)  # 7.5 mm from the bore

    def test_offline_state_path(self):
        tyre = self.spec["tyre_a"]
        c, size, yaw = tyre.offline_box()
        box = SimpleNamespace(pos=np.array(c), quat_xyzw=quat(math.degrees(yaw)), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = tyre.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [0.24, 0.15, L.TOP], atol=1e-9)
        self.assertTrue(Tracker.at_start("tyre_a", st))

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(PARKED)
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        right = self.sim(RIGHT)
        for _ in range(FULL - 1):
            self.feed_arms(PARKED)
            t.success(right)
        self.feed_arms(PARKED)
        self.assertTrue(t.success(right))
        self.assertEqual(t.score(right), 1.0)
        self.assertEqual(t.score(right), 1.0)  # score() does not advance it
        self.feed_arms(holding(A, "left"))
        self.assertEqual(t.score(right), 0.5)  # read with the arms of the step at hand
        self.feed_arms(PARKED)
        n = tr.steps
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(PARKED)
        t.success(right)
        self.assertEqual(tr.steps, 1)
        t.success(right)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(right))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)
        self.assertFalse(t.success({}))  # no tyres in the dict: scored 0, no crash

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.rig.offline_reset()
                self.feed_arms(PARKED)
                self.task.success(self.sim(RIGHT))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "n_seated", "n_correct", "correct", "on_wheel", "tyres",
                          "arms_unavailable", "tyre_a_origin", "tyre_a_yaw_deg", "tyre_a_red_deg", "tyre_b_yellow_deg",
                          "tyre_a_source", "left_tcp", "right_tips", "openings"):
                    self.assertIn(k, row)
                for k in ("wheel", "axis_offset_mm", "bottom_mm", "tilt_deg", "heading_deg", "red_deg",
                          "yellow_deg", "seated_on", "why_not_seated", "let_go", "at_rest", "zone_gap_mm",
                          "rule_dot", "rule_target_deg", "rule_error_deg", "path_mm", "yaw_range_deg"):
                    self.assertIn(k, row["tyres"]["tyre_a"])
                self.assertAlmostEqual(row["tyres"]["tyre_a"]["rule_error_deg"], 0.0)
                self.assertEqual(row["tyres"]["tyre_b"]["rule_dot"], "yellow")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(RIGHT)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, name in (("left", "tyre_a"), ("right", "tyre_b")):
            wps = plan[side]
            by = {w.label: w for w in wps}
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every close and open takes 1.5 s
            self.assertGreaterEqual(by["slowly down beside it"].duration, 1.5)
            self.assertGreaterEqual(by["slowly down over the wheel to 1 mm over the turntable"].duration, 2.0)
            grip = by["grip it across its outside"]
            # the closing axis (yaw + 90 degrees) runs across the tyre's local 0/180 or 90/270 line: between ribs
            rel = (math.degrees(grip.yaw) + 90.0 - L.TYRES[name][2]) % 90.0
            self.assertAlmostEqual(min(rel, 90.0 - rel), 0.0, places=9)
            self.assertAlmostEqual(grip.pos[0], L.TYRES[name][0])
            self.assertAlmostEqual(grip.pos[1], L.TYRES[name][1])
            # the tyre turns with the grip: its red dot ends at the rule's answer for its wheel, 1 degree clockwise
            letgo = by["let go"]
            end_yaw = L.TYRES[name][2] + math.degrees(letgo.yaw - grip.yaw)
            wheel = t.WHEEL_OF[name]
            self.assertAlmostEqual(end_yaw, L.required_yaw(wheel) + t.YAW_TRIM_DEG, places=9)
            self.assertLessEqual(abs(math.degrees(letgo.yaw - grip.yaw)), 90.0)
            # heights: pad tips 3 mm above the table at the pick; the tyre's underside 1 mm over the turntable at the
            # release; carried over the valve stem's top; the slow descent starts 1 cm over it
            self.assertAlmostEqual(grip.pos[2] - 0.013 - L.TOP, 0.003)
            self.assertAlmostEqual(letgo.pos[2] - t.GRIP_UP - (L.TOP + L.TT_TOP), 0.001)
            carry = [w for w in wps if w.label.startswith("carry it over")][0]
            self.assertGreater(carry.pos[2] - t.GRIP_UP - (L.TOP + L.VALVE_TOP), 0.040)
            entry = by["down: its underside 1 cm over the valve stem"]
            self.assertAlmostEqual(entry.pos[2] - t.GRIP_UP - (L.TOP + L.VALVE_TOP), 0.010)
            wx, wy = L.wheel_axis(wheel)
            self.assertAlmostEqual(letgo.pos[0] - wx, t.PLACE_TRIM[0])
            self.assertAlmostEqual(letgo.pos[1] - wy, t.PLACE_TRIM[1])
            up = by["straight up"]
            self.assertEqual(up.pos[:2], letgo.pos[:2])
            # straight up far enough that the open fingers' zone is clear of the tyre by more than the 10 mm rule
            open_arm = arm(up.pos, -t.OPEN * 2 / L.FULL_GAP)
            self.assertGreater(zone_gap(open_arm, red_at(wheel, 120.0)), L.HANDS_OFF + 0.01)
        # the fingers open to 76 mm (6 mm each side of the tyre) and are commanded 8 mm narrower than it
        self.assertAlmostEqual(-2 * t.OPEN, 0.076)
        self.assertAlmostEqual(-2 * t.GRIP, L.TYRE_OD - 0.008)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_yaw_trim_covers_the_gpu_bias(self):
        """With the GPU's counter-clockwise turn of a gripped part anywhere from 0 to +3 degrees, the dots still land
        within 2 degrees of their marks (10 allowed)."""
        t = self.task
        self.assertAlmostEqual(t.YAW_TRIM_DEG, -1.0)
        for name in L.TYRE_NAMES:
            wheel = t.WHEEL_OF[name]
            for bias in (0.0, 0.6, 1.7, 3.0):
                yaw = L.required_yaw(wheel) + t.YAW_TRIM_DEG + bias
                self.assertLessEqual(rule_error_deg(on_wheel(wheel, yaw), wheel), 2.0 + 1e-9)

    def test_instruction(self):
        ins = self.task.INSTRUCTION
        for s in ("each wheel", "red and yellow dots", "valve stem", "white dot", "low-point mark", "let go"):
            self.assertIn(s, ins)
        # it names the marks but does not say which dot goes where (that is what the task tests)
        for s in ("red dot to", "yellow dot to", "red to", "yellow to", "align the red", "align the yellow"):
            self.assertNotIn(s, ins.lower())
        js = json.loads((HERE / "task.json").read_text())
        self.assertEqual(js["instruction"], ins)

    def test_texts_are_clean(self):
        bad = [chr(0x2014), chr(0x2013)]  # em and en dashes
        ids = ["co-authored"]
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            p = HERE / f
            if not p.exists():
                continue
            text = p.read_text()
            for ch in bad:
                self.assertNotIn(ch, text, f)
            if f in ("task.json", "README.md"):
                low = text.lower()
                for w in ids:
                    self.assertNotIn(w, low, f)
                self.assertFalse(_ident_hits(low), f)

    def test_offline_checker_predicts_one_and_half_with_one_arm(self):
        """The repository's offline checker (kinematics and a carry/drop proxy, no physics) plays the demo."""
        checker = _offline_dir() / "check_task.py"
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "r.json"
            proc = subprocess.run([sys.executable, str(checker), str(HERE), "--one-arm", "--json", str(out)],
                                  capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            rep = json.loads(out.read_text())
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertEqual(rep["one_arm"], {"left_only": 0.5, "right_only": 0.5})
        moved = proxy["moved_objects"]
        self.assertAlmostEqual(math.degrees(moved["tyre_a"]["yaw"]), 119.0, delta=0.1)  # red dot at the mark - 1
        self.assertAlmostEqual(math.degrees(moved["tyre_b"]["yaw"]), -101.0, delta=0.1)  # yellow at the valve - 1
        self.assertLessEqual(rep["playback"]["plan_steps"], 1200)


def _ident_hits(text):
    """Team-identifying words found in text. They are kept as sha256 prefixes so this file names nobody."""
    import hashlib
    import re as _re
    banned = {"54622c861585ec4f", "7a5e09c6c81cf86f", "a4cde7cecd46cd77", "b1fc1d0d7b2f62fa", "c70eca6b0f88f44d", "c857d09db23e6822", "d807637b43a9a01c", "da3a1991ce3a0d58", "db6017dfdfe82073", "f0556d0df0fd581f"}
    found = []
    for word in set(_re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())):
        for w in {word, *word.split("-")}:
            if hashlib.sha256(w.encode()).hexdigest()[:16] in banned:
                found.append(w)
    return found


if __name__ == "__main__":
    unittest.main()
