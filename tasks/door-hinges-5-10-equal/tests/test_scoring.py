"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

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
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, along_err_deg, bottom_above_edge, corners, evaluate, heading_deg, held, judge, lowest, reach,
    resting_on_edge, rotation, tier, tilt_deg, valid,
)

MM = 0.001
A, B, C = L.LEAF_NAMES


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    c = np.array(center, float)
    s = SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                        ang_vel=np.array(spin, float), low=c, high=c)
    k = corners(s)
    s.low, s.high = k.min(axis=0), k.max(axis=0)
    return s


def leaf(x, y, bottom=L.EDGE_TOP, yaw=0.0, tilt=0.0, axis="x", **kw):
    """A leaf centred at (x, y), its lowest point at ``bottom``, turned ``yaw`` degrees about the vertical and then
    tilted ``tilt`` degrees about a world axis."""
    q = qmul(about(axis, tilt), about("z", yaw))
    s = state((x, y, 0.0), q)
    return state((x, y, bottom - lowest(s)), q, **kw)


def on_edge(y, x=L.DOOR_X, **kw):
    return leaf(x, y, **kw)


def tray(name):
    x, y, yaw = L.LEAF_START[name]
    return leaf(x, y, bottom=L.TRAY_TOP, yaw=yaw)


S0 = {n: tray(n) for n in L.LEAF_NAMES}


def scene(**placed):
    """The three leaves: those given by name, the rest on the tray where they started."""
    out = dict(S0)
    out.update(placed)
    return out


DONE = scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL), C: on_edge(L.BOTTOM_Y)})


def run(frames, arms=None, hz=20.0):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


def held_for(final, n=25, arms=None):
    """From the start, the leaves jump (not to their marks) to ``final`` and stay there n steps."""
    return run([S0] + [final] * n, arms)


def score_after(final, n=25, arms=None):
    return held_for(final, n, arms)[0].score


ON = 0.0225 / L.FULL_GAP  # rigstate's opening with the fingers closed on a 22.5 mm leaf
DEMO_OPEN = 0.044 / L.FULL_GAP


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


PARKED = snap(left=(0.220, 0.250, L.TOP + 0.107), right=(0.220, -0.250, L.TOP + 0.107))


class Geometry(unittest.TestCase):
    def test_rule_numbers(self):
        self.assertAlmostEqual(L.DOOR_LEN, 0.400)  # 80 in at 1 in = 5 mm
        self.assertAlmostEqual(L.LEAF_LEN, 0.0225)  # 4-1/2 in
        self.assertAlmostEqual(L.HEAD_Y, 0.200)
        self.assertAlmostEqual(L.FLOOR_Y, -0.20375)  # 3/4 in undercut
        self.assertAlmostEqual(L.TOP_Y + L.LEAF_LEN / 2, L.HEAD_Y - 0.025)  # top of the top leaf 5 in down
        self.assertAlmostEqual(L.BOTTOM_Y - L.LEAF_LEN / 2, L.FLOOR_Y + 0.050)  # bottom of the bottom leaf 10 in up
        self.assertAlmostEqual(L.MID_NOMINAL, 0.010625)
        self.assertAlmostEqual(L.WINDOW, 1 * L.IN)

    def test_tilt_heading_and_lowest(self):
        s = on_edge(0.0)
        self.assertAlmostEqual(tilt_deg(s), 0.0)
        self.assertAlmostEqual(heading_deg(s), 0.0)
        self.assertAlmostEqual(lowest(s), L.EDGE_TOP)
        self.assertAlmostEqual(along_err_deg(on_edge(0.0, yaw=180.0)), 0.0, places=6)  # either way round
        self.assertAlmostEqual(along_err_deg(on_edge(0.0, yaw=-3.0)), 3.0, places=6)
        self.assertAlmostEqual(along_err_deg(on_edge(0.0, yaw=90.0)), 90.0, places=6)
        self.assertAlmostEqual(along_err_deg(on_edge(0.0, yaw=177.0)), 3.0, places=6)
        self.assertAlmostEqual(tilt_deg(on_edge(0.0, tilt=180.0)), 0.0, places=6)  # either face up
        t = on_edge(0.0, tilt=10.0, axis="y")
        self.assertAlmostEqual(tilt_deg(t), 10.0, places=6)
        self.assertAlmostEqual(lowest(t), L.EDGE_TOP)
        self.assertEqual(heading_deg(on_edge(0.0, tilt=90.0, axis="x")), 90.0)  # standing on an end

    def test_resting_on_edge_bands(self):
        y = L.TOP_Y
        self.assertTrue(resting_on_edge(on_edge(y)))
        self.assertTrue(resting_on_edge(on_edge(y, x=L.DOOR_X + 0.0079)))
        self.assertFalse(resting_on_edge(on_edge(y, x=L.DOOR_X + 0.0085)))
        self.assertTrue(resting_on_edge(on_edge(y, x=L.DOOR_X - 0.0079)))
        self.assertFalse(resting_on_edge(on_edge(y, x=L.DOOR_X - 0.0085)))
        self.assertTrue(resting_on_edge(on_edge(y, bottom=L.EDGE_TOP + 0.0029)))
        self.assertFalse(resting_on_edge(on_edge(y, bottom=L.EDGE_TOP + 0.0032)))
        self.assertFalse(resting_on_edge(on_edge(y, bottom=L.EDGE_TOP - 0.0032)))
        self.assertTrue(resting_on_edge(on_edge(y, tilt=4.9, axis="y")))
        self.assertFalse(resting_on_edge(on_edge(y, tilt=5.2, axis="y")))
        self.assertTrue(resting_on_edge(on_edge(y, yaw=4.9)))
        self.assertFalse(resting_on_edge(on_edge(y, yaw=5.2)))
        self.assertFalse(resting_on_edge(on_edge(y, yaw=90.0)))  # crosswise
        self.assertTrue(resting_on_edge(on_edge(L.ON_DOOR)))
        self.assertFalse(resting_on_edge(on_edge(L.ON_DOOR + 0.001)))  # past the door's top end
        self.assertFalse(resting_on_edge(tray(A)))
        # A leaf inside the across window is wholly on the 36 mm edge, even turned and tilted to the limits.
        self.assertGreaterEqual(L.DOOR_THICK / 2 - L.ACROSS_TOL - L.LEAF_W / 2, -1e-12)
        worst = 0.0
        for dx in (-L.ACROSS_TOL, L.ACROSS_TOL):
            for yaw in (-4.9, 4.9):
                for axis in ("x", "y"):
                    for tilt in (-4.9, 4.9):
                        s = on_edge(y, x=L.DOOR_X + dx, yaw=yaw, tilt=tilt, axis=axis)
                        self.assertTrue(resting_on_edge(s))
                        worst = max(worst, float(np.abs(corners(s)[:, 0] - L.DOOR_X).max()))
        self.assertLess(worst, L.DOOR_THICK / 2)

    def test_leaf_proportions_are_plain_to_see(self):
        # Lengthwise and flat must be visible in a photo: each side at least 1.4 times the next.
        self.assertGreaterEqual(L.LEAF_LEN / L.LEAF_W, 1.4)
        self.assertGreaterEqual(L.LEAF_W / L.LEAF_H, 1.4)
        self.assertAlmostEqual(L.LEAF_MASS, 8500 * L.LEAF_LEN * L.LEAF_W * L.LEAF_H, delta=0.001)  # solid brass

    def test_valid(self):
        self.assertTrue(valid(on_edge(0.0)))
        for bad in (state((np.nan, 0, 0.8)), state((np.inf, 0, 0.8)), SimpleNamespace(pos=(0, 0, 0))):
            self.assertFalse(valid(bad))
        z = on_edge(0.0)
        z.quat_xyzw = np.zeros(4)
        self.assertFalse(valid(z))


class Tiers(unittest.TestCase):
    def test_demo_layout_scores_one(self):
        r, t = held_for(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["tier"], "all_three_right_and_still_1s")
        self.assertEqual(r.details["top_leaf"], A)
        self.assertEqual(r.details["bottom_leaf"], C)
        self.assertEqual(r.details["middle_leaf"], B)

    def test_any_leaf_any_location(self):
        swapped = scene(**{C: on_edge(L.TOP_Y), A: on_edge(L.MID_NOMINAL), B: on_edge(L.BOTTOM_Y)})
        self.assertEqual(score_after(swapped), 1.0)

    def test_middle_off_centre_is_075(self):
        off = scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL + 0.010), C: on_edge(L.BOTTOM_Y)})
        self.assertEqual(score_after(off), 0.75)
        for d, want in ((0.0049, 1.0), (-0.0049, 1.0), (0.0052, 0.75), (-0.0052, 0.75)):
            s = scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL + d), C: on_edge(L.BOTTOM_Y)})
            self.assertEqual(score_after(s), want, d)

    def test_end_windows(self):
        for d, want in ((0.0049, 1.0), (-0.0049, 1.0), (0.0052, 0.5), (-0.0052, 0.5)):
            # top off by d: the middle stays centred between the leaves as hung
            top, bot = L.TOP_Y + d, L.BOTTOM_Y
            s = scene(**{A: on_edge(top), B: on_edge((top + bot) / 2), C: on_edge(bot)})
            self.assertEqual(score_after(s), want, ("top", d))
            top, bot = L.TOP_Y, L.BOTTOM_Y + d
            s = scene(**{A: on_edge(top), B: on_edge((top + bot) / 2), C: on_edge(bot)})
            self.assertEqual(score_after(s), want, ("bottom", d))

    def test_middle_follows_the_hinges_as_hung(self):
        """Top and bottom both 4 mm high: the middle is judged against their midpoint, 4 mm above nominal."""
        top, bot = L.TOP_Y + 0.004, L.BOTTOM_Y + 0.004
        s = scene(**{A: on_edge(top), B: on_edge(L.MID_NOMINAL + 0.004), C: on_edge(bot)})
        self.assertEqual(score_after(s), 1.0)
        top, bot = L.TOP_Y + 0.0045, L.BOTTOM_Y - 0.0045  # midpoint unchanged
        s = scene(**{A: on_edge(top), B: on_edge(L.MID_NOMINAL - 0.0045), C: on_edge(bot)})
        self.assertEqual(score_after(s), 1.0)
        top, bot = L.TOP_Y + 0.0045, L.BOTTOM_Y + 0.0045  # midpoint +4.5: a middle at -0.8 mm is 5.3 off
        s = scene(**{A: on_edge(top), B: on_edge(L.MID_NOMINAL - 0.0008), C: on_edge(bot)})
        self.assertEqual(score_after(s), 0.75)

    def test_not_yet_still_for_1s_is_075(self):
        r, _ = held_for(DONE, n=12)  # released and settled for 0.5 s, not yet 1 s
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["tier"], "all_three_right_not_still_1s")
        r, _ = held_for(DONE, n=21)
        self.assertEqual(r.score, 1.0)

    def test_two_leaves_on_the_edge(self):
        both_ends = scene(**{A: on_edge(L.TOP_Y), C: on_edge(L.BOTTOM_Y)})  # third left on the tray
        self.assertEqual(score_after(both_ends), 0.5)
        left_arm = scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL)})  # what the left arm alone hangs
        self.assertEqual(score_after(left_arm), 0.5)
        mid_bottom = scene(**{B: on_edge(L.MID_NOMINAL), C: on_edge(L.BOTTOM_Y)})
        self.assertEqual(score_after(mid_bottom), 0.5)
        one_wrong = scene(**{A: on_edge(L.TOP_Y), C: on_edge(L.BOTTOM_Y + 0.020)})
        self.assertEqual(score_after(one_wrong), 0.25)

    def test_one_location(self):
        for y in (L.TOP_Y, L.BOTTOM_Y, L.MID_NOMINAL):
            self.assertEqual(score_after(scene(**{A: on_edge(y)})), 0.25, y)
        self.assertEqual(score_after(scene(**{C: on_edge(L.BOTTOM_Y)})), 0.25)  # the right arm alone

    def test_nothing_right_is_zero(self):
        self.assertEqual(score_after(S0, n=30), 0.0)
        wrong = scene(**{A: on_edge(0.120), B: on_edge(-0.040), C: on_edge(-0.180)})
        self.assertEqual(score_after(wrong), 0.0)
        self.assertEqual(score_after(scene(**{A: on_edge(0.080)})), 0.0)

    def test_an_end_wrong_caps_at_05(self):
        """Top wrong, bottom right, the middle centred between them: k = 2 with an end wrong, 0.5."""
        top, bot = L.TOP_Y - 0.020, L.BOTTOM_Y
        s = scene(**{A: on_edge(top), B: on_edge((top + bot) / 2), C: on_edge(bot)})
        self.assertEqual(score_after(s), 0.5)

    def test_every_tier_and_only_listed_values(self):
        cases = {
            1.0: DONE,
            0.75: scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL + 0.010), C: on_edge(L.BOTTOM_Y)}),
            0.5: scene(**{A: on_edge(L.TOP_Y), C: on_edge(L.BOTTOM_Y)}),
            0.25: scene(**{A: on_edge(0.100), B: on_edge(0.0), C: on_edge(-0.100)}),
            0.0: S0,
        }
        for want, s in cases.items():
            self.assertEqual(score_after(s), want)

    def test_tier_function_is_a_partition(self):
        """Every combination of leaf count, k and end results maps to one score."""
        for n in range(4):
            for top_ok in (False, True):
                for bot_ok in (False, True):
                    for mid_ok in (False, True):
                        for still in (False, True):
                            k = int(top_ok) + int(bot_ok) + int(mid_ok)
                            if k > n:
                                continue
                            d = {"n_on_edge": n, "k": k, "top_ok": top_ok, "bottom_ok": bot_ok}
                            sc, _ = tier(d, still)
                            if n == 3 and k == 3 and still:
                                want = 1.0
                            elif n == 3 and top_ok and bot_ok:
                                want = 0.75
                            else:
                                want = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.5}[k]
                            self.assertEqual(sc, want, d)

    def test_judge_with_fewer_than_three_uses_nominal_windows(self):
        d = judge({A: L.TOP_Y, B: L.MID_NOMINAL + 0.0049})
        self.assertEqual((d["k"], d["top_ok"], d["middle_ok"], d["bottom_ok"]), (2, True, True, False))
        d = judge({A: L.TOP_Y, B: L.MID_NOMINAL + 0.006})
        self.assertEqual(d["k"], 1)


class ZeroConditions(unittest.TestCase):
    """A leaf that is not on the edge does not count; here the other two stay right, so 0.5 drops to 0.25 or
    1.0 to 0.5."""

    def check(self, bad_mid, want=0.5):
        s = scene(**{A: on_edge(L.TOP_Y), B: bad_mid, C: on_edge(L.BOTTOM_Y)})
        self.assertEqual(score_after(s), want)

    def test_each_condition(self):
        y = L.MID_NOMINAL
        self.check(on_edge(y, tilt=8.0, axis="y"))  # tilted
        self.check(on_edge(y, tilt=90.0, axis="y"))  # on its side
        self.check(on_edge(y, tilt=90.0, axis="x"))  # on an end
        self.check(on_edge(y, yaw=90.0))  # crosswise
        self.check(on_edge(y, yaw=7.0))  # turned
        self.check(on_edge(y, bottom=L.EDGE_TOP + 0.006))  # held up
        self.check(on_edge(y, x=L.DOOR_X + 0.012))  # off centre, overhanging the face
        self.check(leaf(L.DOOR_X - 0.050, y, bottom=L.TOP))  # on the table beside the door
        self.check(tray(B))  # left on the tray

    def test_moving_leaf_does_not_count(self):
        frames = [S0]
        for i in range(25):
            frames.append(scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL + 0.0004 * i), C: on_edge(L.BOTTOM_Y)}))
        r, _ = run(frames)
        self.assertEqual(r.score, 0.5)  # 4 mm in the last 0.5 s: not settled
        frames = [S0]
        for i in range(25):
            frames.append(scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL, yaw=0.2 * i), C: on_edge(L.BOTTOM_Y)}))
        r, _ = run(frames)
        self.assertEqual(r.score, 0.5)  # turning 2 deg in the last 0.5 s
        frames = [S0]
        for i in range(25):
            frames.append(scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.MID_NOMINAL + 0.00015 * i), C: on_edge(L.BOTTOM_Y)}))
        r, _ = run(frames)
        self.assertEqual(r.score, 1.0)  # a 1.5 mm creep in each 0.5 s is under the 2 mm limit: settled
        self.assertLess(r.details["leaves"][B]["spread_mm"], 2.0)


class LetGo(unittest.TestCase):
    def test_reach_values(self):
        self.assertAlmostEqual(L.release_reach(1.0), 0.0792, places=3)
        self.assertAlmostEqual(L.release_reach(DEMO_OPEN), 0.0565, places=3)

    def test_held_rules(self):
        s = on_edge(L.MID_NOMINAL)
        top = L.EDGE_TOP + L.LEAF_H
        self.assertTrue(held(s, snap(left=(L.DOOR_X, L.MID_NOMINAL, L.EDGE_TOP + 0.0045), left_open=ON)))
        self.assertTrue(held(s, snap(left=(L.DOOR_X, L.MID_NOMINAL, top + 0.009), left_open=1.0)))
        self.assertFalse(held(s, snap(left=(L.DOOR_X, L.MID_NOMINAL, top + 0.011), left_open=1.0)))
        self.assertFalse(held(s, snap(left=(L.DOOR_X, L.MID_NOMINAL + 0.060, L.EDGE_TOP + 0.004),
                                      left_open=DEMO_OPEN)))
        self.assertTrue(held(s, snap(left=(L.DOOR_X, L.MID_NOMINAL + 0.050, L.EDGE_TOP + 0.004),
                                     left_open=DEMO_OPEN)))
        self.assertFalse(held(s, None))
        self.assertFalse(held(s, PARKED))
        broken = {"left": SimpleNamespace(tips=(np.nan, 0, 0), opening=1.0)}
        self.assertFalse(held(s, broken))

    def test_still_gripped_at_the_end_does_not_count(self):
        grip = snap(left=(L.DOOR_X, L.MID_NOMINAL, L.EDGE_TOP + 0.0045), left_open=ON,
                    right=(0.220, -0.250, L.TOP + 0.107))
        self.assertEqual(score_after(DONE, arms=grip), 0.5)  # two leaves on the edge, k = 2
        self.assertEqual(score_after(DONE, arms=PARKED), 1.0)

    def test_release_must_last_half_a_second(self):
        grip = snap(left=(L.DOOR_X, L.MID_NOMINAL, L.EDGE_TOP + 0.0045), left_open=DEMO_OPEN,
                    right=(0.220, -0.250, L.TOP + 0.107))
        t = Tracker()
        t.update(S0, grip)
        for _ in range(25):
            r = t.update(DONE, grip)
        self.assertEqual(r.score, 0.5)
        for i in range(10):
            r = t.update(DONE, PARKED)
            self.assertEqual(r.score, 0.5, i)  # let go less than 0.5 s ago
        r = t.update(DONE, PARKED)
        self.assertEqual(r.score, 1.0)

    def test_unreadable_arms_are_skipped_and_counted(self):
        r, t = held_for(DONE, arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, 26)
        r, t = held_for(DONE, arms=PARKED)
        self.assertEqual(t.arms_unavailable, 0)


class Broken(unittest.TestCase):
    def test_nan_state_scores_zero_and_keeps_the_tracker(self):
        r, t = held_for(DONE)
        self.assertEqual(r.score, 1.0)
        hist = {n: len(h) for n, h in t.history.items()}
        for bad in (state((np.nan, 0, 0.8)), state((0, np.inf, 0.8))):
            r = t.update(scene(**{A: bad, B: DONE[B], C: DONE[C]}))
            self.assertEqual(r.score, 0.0)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        z = on_edge(L.TOP_Y)
        z.quat_xyzw = np.zeros(4)
        self.assertEqual(t.update(scene(**{A: z, B: DONE[B], C: DONE[C]})).score, 0.0)
        missing = SimpleNamespace(pos=np.zeros(3))
        self.assertEqual(t.update(scene(**{A: missing, B: DONE[B], C: DONE[C]})).score, 0.0)
        self.assertEqual(t.bad_steps, 4)
        self.assertEqual({n: len(h) for n, h in t.history.items()}, hist)
        self.assertEqual(t.update(DONE).score, 1.0)  # back to a good state: the history was kept


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        r, t = held_for(DONE)
        self.assertEqual(r.score, 1.0)
        r = t.update(S0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(len(t.history[A]), 1)

    def test_gradual_return_is_not_a_reset(self):
        _, t = held_for(DONE)
        for i in range(1, 31):
            f = i / 30
            t.update({n: leaf((1 - f) * DONE[n].pos[0] + f * S0[n].pos[0], (1 - f) * DONE[n].pos[1] + f * S0[n].pos[1],
                              bottom=L.TRAY_TOP) for n in L.LEAF_NAMES})
        self.assertGreater(t.steps, 30)

    def test_jump_elsewhere_is_not_a_reset(self):
        _, t = held_for(DONE)
        t.update(scene(**{A: on_edge(0.0)}))
        self.assertGreater(t.steps, 20)

    def test_jittered_and_swapped_reset_counts(self):
        _, t = held_for(DONE)
        jit = {A: leaf(L.TRAY_X + 0.02, 0.050 - 0.02, bottom=L.TRAY_TOP),
               B: leaf(L.TRAY_X - 0.01, 0.130 + 0.02, bottom=L.TRAY_TOP), C: tray(C)}
        t.update(jit)
        self.assertEqual(t.steps, 1)

    def test_rigstate_reset_hook_resets_the_task_tracker(self):
        task = _task()
        if task is None:
            self.skipTest("needs the offline yam_arena stub")
        task._TRACKER.reset()
        for _ in range(5):
            task._TRACKER.update(DONE)
        rigstate.offline_reset()
        self.assertEqual(task._TRACKER.steps, 0)


class Exploits(unittest.TestCase):
    def test_evenly_spaced_quarters(self):
        s = scene(**{A: on_edge(0.100), B: on_edge(0.0), C: on_edge(-0.100)})
        r, _ = held_for(s)
        self.assertEqual(r.score, 0.25)
        self.assertEqual((r.details["top_ok"], r.details["middle_ok"], r.details["bottom_ok"]), (False, True, False))

    def test_symmetric_margins_copied_from_the_top(self):
        s = scene(**{A: on_edge(L.TOP_Y), B: on_edge(0.0), C: on_edge(-L.TOP_Y)})
        r, _ = held_for(s)
        self.assertEqual(r.score, 0.5)
        self.assertEqual((r.details["top_ok"], r.details["middle_ok"], r.details["bottom_ok"]), (True, True, False))

    def test_measured_from_the_door_ends_not_the_lines(self):
        """Bottom 10 in from the door's own bottom end (ignoring the undercut): 3.75 mm high, still inside the
        +-5 mm window; 5 in from the floor (both ends 5 in): wrong."""
        top, bot = L.TOP_Y, L.DOOR_BOTTOM_END + 0.050 + L.LEAF_LEN / 2
        s = scene(**{A: on_edge(top), B: on_edge((top + bot) / 2), C: on_edge(bot)})
        self.assertEqual(score_after(s), 1.0)  # within 1 in: the window is +-1 in
        bot = L.FLOOR_Y + 0.025 + L.LEAF_LEN / 2
        s = scene(**{A: on_edge(top), B: on_edge((top + bot) / 2), C: on_edge(bot)})
        self.assertEqual(score_after(s), 0.5)

    def test_crosswise_leaf_at_the_right_place_counts_nothing(self):
        self.assertEqual(score_after(scene(**{A: on_edge(L.TOP_Y, yaw=90.0)})), 0.0)

    def test_held_on_the_spot_earns_nothing_for_it(self):
        grip = snap(left=(L.DOOR_X, L.TOP_Y, L.EDGE_TOP + 0.0045), left_open=ON,
                    right=(L.DOOR_X, L.BOTTOM_Y, L.EDGE_TOP + 0.0045), right_open=ON)
        self.assertEqual(score_after(DONE, arms=grip), 0.25)  # only the middle, against the nominal window

    def test_stacked_leaves_do_not_count(self):
        """Two leaves stacked at the top location: the upper one rests on a leaf, not on the edge."""
        s = scene(**{A: on_edge(L.TOP_Y), B: on_edge(L.TOP_Y, bottom=L.EDGE_TOP + L.LEAF_H), C: on_edge(L.BOTTOM_Y)})
        self.assertEqual(score_after(s), 0.5)

    def test_leaf_off_the_door_end_or_beside_the_door(self):
        s = scene(**{A: leaf(L.DOOR_X, L.HEAD_Y + 0.015, bottom=L.TOP), B: on_edge(L.MID_NOMINAL),
                     C: on_edge(L.BOTTOM_Y)})
        self.assertEqual(score_after(s), 0.5)

    def test_random_states_score_only_listed_values(self):
        rng = np.random.default_rng(11)
        seen = set()
        for _ in range(300):
            placed = {}
            for n in L.LEAF_NAMES:
                if rng.random() < 0.8:
                    centre = rng.choice([L.TOP_Y, L.MID_NOMINAL, L.BOTTOM_Y, rng.uniform(-0.19, 0.19)])
                    placed[n] = on_edge(float(centre + rng.normal(0, 0.005)), x=L.DOOR_X + rng.normal(0, 0.004),
                                        yaw=rng.normal(0, 3))
            leaves = scene(**placed)
            sc, d = evaluate(leaves, {n: True for n in L.LEAF_NAMES}, {n: True for n in L.LEAF_NAMES})
            self.assertIn(sc, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertLessEqual(d["k"], d["n_on_edge"])
            seen.add(sc)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _kin_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yamkin.py").is_file():
            return d
    return None


def _task():
    if _stub_dir() is None:
        return None
    sys.path.insert(0, str(_stub_dir()))
    return importlib.import_module("task")


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        cls.task = _task()

    def objects(self, leaves):
        return dict(leaves)

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.LEAF_NAMES, L.LEAF_NAMES)
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.LEAF_NAMES))
        for spec in t.LEAVES:
            s = S0[spec.name]
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual(tuple(spec.size), L.LEAF_SIZE)
            self.assertLessEqual(spec.mass, 0.3)
        d = t.DOOR
        self.assertAlmostEqual(d.pos[2] + d.size[2] / 2, L.EDGE_TOP)
        self.assertAlmostEqual(d.pos[1] + d.size[1] / 2, L.HEAD_Y)  # the head line's door-side edge
        self.assertAlmostEqual(t.HEAD_LINE.pos[1] - t.HEAD_LINE.size[1] / 2, L.HEAD_Y)
        self.assertAlmostEqual(t.FLOOR_LINE.pos[1] + t.FLOOR_LINE.size[1] / 2, L.FLOOR_Y)
        self.assertAlmostEqual(d.pos[1] - d.size[1] / 2 - L.FLOOR_Y, 0.00375)  # the 3/4 in undercut
        for line in (t.HEAD_LINE, t.FLOOR_LINE):  # visible past both faces of the door
            self.assertGreater(line.size[0] / 2 - L.DOOR_THICK / 2, 0.015)
        # Nothing where the arms' start pose hides the table in the setup screenshot (|y| > 0.45 m and x < 0.2 m).
        for o in t.OBJECTS:
            self.assertLess(abs(o.pos[1]) + o.size[1] / 2, 0.45, o.name)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as dname:
            path = os.path.join(dname, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(S0))
                self.assertEqual(self.task.score(S0), 0.0)
                oks = [self.task.success(DONE) for _ in range(21)]
                self.assertTrue(oks[-1])
                self.assertFalse(any(oks[:19]))
                self.assertEqual(self.task.score(DONE), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 22)
        last = rows[-1]
        for key in ("episode", "step", "score", "success", "tier", "k", "on_edge", "top_ok", "middle_ok",
                    "bottom_ok", "y_m_mm", "top_err_mm", "bottom_err_mm", "middle_err_mm", "all_still_1s", "leaves",
                    "bad_steps", "arms_unavailable", "first_step", "hinge_leaf_a_pose"):
            self.assertIn(key, last)
        self.assertEqual(last["top_err_mm"], 0.0)
        self.assertIn("rigstate_error", last)
        pose = last["hinge_leaf_b_pose"]
        for key in ("pos_mm", "tilt_deg", "along_err_deg", "off_centreline_mm", "bottom_above_edge_mm",
                    "resting_on_edge", "held"):
            self.assertIn(key, pose)
        self.assertTrue(pose["resting_on_edge"])

    def test_hooks_read_the_arms(self):
        def feed(tips, op):
            rigstate.offline_feed({"left": {"tcp": np.add(tips, (0, 0, 0.013)), "tips": tips, "opening": op},
                                   "right": {"tcp": PARKED["right"].tcp, "tips": PARKED["right"].tips,
                                             "opening": 1.0}})
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        grip = (L.DOOR_X, L.MID_NOMINAL, L.EDGE_TOP + 0.0045)
        try:
            for _ in range(25):
                feed(grip, ON)
                self.assertFalse(self.task.success(DONE))
            self.assertEqual(self.task.score(DONE), 0.5)
            for _ in range(11):
                feed(tuple(PARKED["left"].tips), 1.0)
                ok = self.task.success(DONE)
            self.assertTrue(ok)  # 0.5 s after the left gripper rose clear (the 2 s hold is the simulator's)
            self.assertEqual(self.task.score(DONE), 1.0)
        finally:
            rigstate.offline_reset()

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(S0))
        finally:
            os.environ.pop("TASK_TRACE", None)
        with tempfile.TemporaryDirectory() as dname:
            path = os.path.join(dname, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                self.assertFalse(self.task.success(scene(**{A: state((np.nan, 0, 0.8))})))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                row = json.loads(f.readline())
        self.assertFalse(row["valid_state"])
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_aims_score_one_landing_on_aim_or_4mm_short(self):
        t = self.task
        for short in (0.0, t.SHORT):
            placed = {}
            for loc, name in t.LEAF_FOR.items():
                sx, sy, _ = L.LEAF_START[name]
                px, py = t.place_point(loc)
                tx, ty = t.TARGET[loc]
                dd = math.hypot(tx - sx, ty - sy)
                placed[name] = on_edge(py - short * (ty - sy) / dd, x=px - short * (tx - sx) / dd, yaw=-1.15 + 1.15)
            self.assertEqual(score_after(scene(**placed)), 1.0, short)
        # The aim points are inside every window with at least 3.8 mm to spare across and 3.5 mm along.
        for loc in t.TARGET:
            px, py = t.place_point(loc)
            self.assertLess(abs(px - L.DOOR_X), L.ACROSS_TOL - 0.0039)
            self.assertLess(abs(py - t.TARGET[loc][1]), L.WINDOW - 0.0035)

    def test_one_arm_plans(self):
        """Left arm alone: top and middle hung (0.5). Right arm alone: bottom hung (0.25)."""
        t = self.task
        left = scene(**{t.LEAF_FOR[loc]: on_edge(t.place_point(loc)[1], x=t.place_point(loc)[0])
                        for loc in ("top", "middle")})
        right = scene(**{t.LEAF_FOR["bottom"]: on_edge(t.place_point("bottom")[1], x=t.place_point("bottom")[0])})
        self.assertEqual(score_after(left), 0.5)
        self.assertEqual(score_after(right), 0.25)

    @staticmethod
    def spells(wps):
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w.pos[0], w.pos[1]))
            t += w.duration
        return out

    def test_grippers_stay_apart_and_the_demo_fits(self):
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, _, ya in left:
            for b0, b1, _, yb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + self.task.SUCCESS_HOLD_S + 2.0) * L.CONTROL_HZ, 1200)
        # The right arm has parked before the left arm sets off for the middle leaf.
        right_end = sum(w.duration for w in plan["right"])
        start_mid = next(s for s, w in zip(left, plan["left"]) if w.label == "over hinge_leaf_b")[0]
        self.assertLessEqual(right_end, start_mid + 2.0)

    def test_grip_and_heights(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.GRIP, L.LEAF_LEN - 0.008)  # commanded 8 mm narrower
        self.assertGreaterEqual(-2 * t.GRIP, 0.008)
        self.assertGreaterEqual(L.LEAF_LEN, 0.016)
        self.assertLessEqual(L.LEAF_LEN, 0.080)
        self.assertGreater(-2 * t.OPEN - L.LEAF_LEN, 0.020)  # 10 mm off each end, open
        self.assertGreaterEqual(t.PICK_Z - t.PAD - L.TRAY_TOP, 0.003)  # pad tips over the tray
        self.assertGreaterEqual(t.PLACE_Z - t.PAD - L.EDGE_TOP, 0.004)  # pad tips over the edge at release
        self.assertLess(t.GRIP_UP - t.PAD, L.LEAF_H)  # the pads close on the leaf, not over it
        self.assertGreaterEqual(t.HOVER - t.GRIP_UP - L.EDGE_TOP, 0.020)  # a carried leaf clears the edge
        # Open fingers at a pick (44 mm gap, 8 mm pads) clear the neighbouring leaf on the tray.
        ys = sorted(v[1] for v in L.LEAF_START.values())
        gap = min(b - a for a, b in zip(ys, ys[1:]))
        self.assertGreater(gap - (-t.OPEN + 0.008) - L.LEAF_LEN / 2, 0.020)

    def test_success_hold_outlasts_every_release(self):
        plan = self.task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly lay it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    still_from = L.STILL_WINDOW_S * 2  # 1 s still after the descent ends
                    self.assertGreater(still_from + self.task.SUCCESS_HOLD_S, sum(a.duration for a in after))
                    for a in after:
                        self.assertGreaterEqual(a.duration, 0.5)
                    n += 1
        self.assertEqual(n, 3)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        # The instruction names the standard but does not give the answer (no offsets).
        for s in ("5 in", "10 in", "25 mm", "50 mm", "centred between"):
            self.assertNotIn(s, self.task.INSTRUCTION)

    def test_task_json_states_the_numbers(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("5 mm", "8 mm", "3 mm", "5 degrees", "2 mm", "1 degree", "0.5 s", "1 s", "10 mm above its top",
                  "1.0", "0.75", "0.5", "0.25", "+163.75 mm", "-142.5 mm", "+10.625 mm", "2 s", "1,200 steps", "79 mm", "56.5 mm",
                  "no finger has touched it"):
            self.assertIn(s, text, s)
        setup = spec["setup"]
        for s in ("36 mm", "400 mm", "80 mm", "22.5", "16 mm", "10 mm", "31 g", "brass", "x 20.0 cm",
                  "y +13.0", "+5.0", "-13.0", "x 30.0 cm", "3.75 mm", "60 x 320 x 3 mm"):
            self.assertIn(s, setup, s)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(spec["run_limit"], "1,200 steps (1 minute at 20 Hz)")
        for k in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes"):
            for bad in (chr(0x2014), chr(0x2013)):
                self.assertNotIn(bad, spec[k])


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the far end's hinge location."""

    def test_far_locations_out_of_reach(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        z = L.EDGE_TOP + 0.0015 + 0.016  # the TCP where a leaf is let go
        for arm, y in (("left", L.BOTTOM_Y), ("right", L.TOP_Y)):
            base = K.BASES["station"][arm]
            for yaw in np.linspace(-math.pi, math.pi, 17)[:-1]:
                for zz in (z, z + 0.020, z + 0.050):
                    self.assertFalse(K.reachable((L.DOOR_X, y, zz), float(yaw), base)[0], (arm, y, yaw, zz))
        for arm, ys in (("left", (L.TOP_Y, L.MID_NOMINAL)), ("right", (L.BOTTOM_Y, L.MID_NOMINAL))):
            base = K.BASES["station"][arm]
            for y in ys:
                self.assertTrue(K.reachable((L.DOOR_X, y, z), 0.0, base)[0], (arm, y))


if __name__ == "__main__":
    unittest.main()
