"""Offline tests for scoring.py (and task.py's hooks): synthetic stick states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import itertools
import json
import math
import os
import pathlib
import random
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
    LOCAL_CORNERS, Tracker, bottom_gap, centre_xy, evaluate, held, highest, lowest, off_itf, off_net_line,
    on_net_line, position_score, reach, rotation, standing, stick_score, tilt_deg, valid,
)

MM = 0.001
Y = L.STICK_Y  # the ITF point, |y| 0.3353
SINGLES = L.SINGLES_LINE_Y  # a stick centred on the singles sideline: |y| 0.2728
MID = 0.320  # 'the middle of the alley' (the alley's centre is 0.3200)
DOUBLES = 0.366  # against the doubles sideline
BESIDE_POST = 0.395  # next to the net post


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(math.sin(h) * np.array(v, float)) + (math.cos(h),)


def stick(x, y, bottom=L.COURT_TOP, tilt=0.0, axis="x", yaw=0.0, vel=(0, 0, 0)):
    """A stick whose centre on the court (its base frame's origin) is at (x, y), turned ``yaw`` degrees and then
    tilted ``tilt`` degrees about a world axis, its lowest point at ``bottom``."""
    q = quat_mul(about(axis, tilt), about("z", yaw))
    r = rotation(q)
    w = LOCAL_CORNERS @ r.T
    origin = np.array([x, y, bottom - w[:, 2].min()])
    pts = w + origin
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=origin, quat_xyzw=np.array(q, float), pos=(lo + hi) / 2,
                           lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=lo, high=hi)


def at(ya, yb, xa=L.NET_X, xb=L.NET_X):
    """Stick A at (xa, ya), stick B at (xb, yb), both upright on the court."""
    return [stick(xa, ya), stick(xb, yb)]


S0 = [stick(*L.START[n]) for n in L.NAMES]
DONE = at(Y, -Y)


def score_of(sticks, arms=None):
    return evaluate(sticks, arms)[0]


def glide(a, b, n):
    """n frames moving every stick of frame a to its place in frame b in equal steps (orientations of b)."""
    out = []
    for i in range(1, n + 1):
        f = []
        for sa, sb in zip(a, b, strict=True):
            o = (1 - i / n) * sa.origin + i / n * sb.origin
            s = SimpleNamespace(**vars(sb))
            s.origin = o
            s.pos = sb.pos + (o - sb.origin)
            f.append(s)
        out.append(f)
    return out


def build(final=None, n=20, hold=25):
    """From the start, both sticks moved gradually to ``final`` (default the ITF points), then held there."""
    final = DONE if final is None else final
    return [S0] + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0, arms=None):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


ON = L.POST_SIZE[0] / L.FULL_GAP  # rigstate's opening with the fingers closed on the 20 mm post


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    """A rigstate.arms() snapshot: each given arm's pad-tip point (x, y, z) and opening (0 closed .. 1 open)."""
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


# Both arms parked where the demo leaves them: over their own side of the near court, pad tips 147 mm up.
PARKED = snap(left=(0.200, 0.250, L.TOP + 0.147), right=(0.200, -0.250, L.TOP + 0.147))
GRIP_TIPS_Z = L.COURT_TOP + 0.050 - 0.013  # the demo's pad tips while it holds a stick standing on the court


class Geometry(unittest.TestCase):
    def test_scale_arithmetic(self):
        """ITF Rule 1 at 1:15: singles 8.23 m and doubles 10.97 m to the lines' outside edges, sticks and posts
        0.914 m outside the singles and doubles courts, 1.07 m net height."""
        self.assertAlmostEqual(L.SINGLES_HALF, 4.115 / 15, places=6)
        self.assertAlmostEqual(L.DOUBLES_HALF, 5.485 / 15, places=6)
        self.assertAlmostEqual(L.STICK_Y, (4.115 + 0.914) / 15, places=6)
        self.assertAlmostEqual(L.NET_POST_Y, (5.485 + 0.914) / 15, places=6)
        self.assertAlmostEqual(round(L.STICK_Y, 4), 0.3353)
        self.assertAlmostEqual(round(L.NET_POST_Y, 4), 0.4266)
        self.assertAlmostEqual((L.STICK_Y - L.SINGLES_HALF) / (L.DOUBLES_HALF - L.SINGLES_HALF), 0.914 / 1.37)
        self.assertAlmostEqual(L.STICK_H, 0.071)
        self.assertAlmostEqual(L.NET_HEIGHT, 1.07 / 15)
        self.assertAlmostEqual(L.STICK_MASS / L.STICK_VOLUME, 1052.6, places=0)  # ABS
        # the lines' outside edges are the measured edges
        self.assertAlmostEqual(L.SINGLES_LINE_Y + L.LINE_W / 2, L.SINGLES_HALF)
        self.assertAlmostEqual(L.DOUBLES_LINE_Y + L.LINE_W / 2, L.DOUBLES_HALF)

    def test_wrong_guesses_are_outside_the_window(self):
        """Every position a non-specialist picks is more than 8 mm from the ITF point; the nearest (mid-alley) is
        15 mm off."""
        for guess in (SINGLES, L.SINGLES_HALF, MID, (L.SINGLES_HALF + L.DOUBLES_HALF) / 2, DOUBLES, L.DOUBLES_HALF,
                      BESIDE_POST):
            self.assertGreater(abs(guess - Y), L.ITF_TOL + 0.006, guess)
        self.assertAlmostEqual(abs(MID - Y), 0.0153, places=4)

    def test_lowest_point_and_tilt(self):
        s = stick(0.35, Y)
        self.assertAlmostEqual(lowest(s), L.COURT_TOP)
        self.assertAlmostEqual(highest(s), L.COURT_TOP + L.STICK_H)
        self.assertAlmostEqual(tilt_deg(s), 0.0)
        self.assertTrue(standing(s))
        for axis in ("x", "y"):
            for deg in (5.0, 9.9, 10.1, 30.0, 90.0, 180.0):
                s = stick(0.35, Y, tilt=deg, axis=axis)
                self.assertAlmostEqual(tilt_deg(s), deg, places=6)
                self.assertAlmostEqual(lowest(s), L.COURT_TOP)
                self.assertEqual(standing(s), deg <= 10.0, (axis, deg))
        # its yaw does not matter (a square post)
        self.assertTrue(standing(stick(0.35, Y, yaw=37.0)))
        self.assertEqual(score_of([stick(0.35, Y, yaw=37.0), stick(0.35, -Y, yaw=-80.0)]), 1.0)

    def test_seat_tolerance(self):
        self.assertTrue(standing(stick(0.35, Y, bottom=L.COURT_TOP + 0.0029)))
        self.assertFalse(standing(stick(0.35, Y, bottom=L.COURT_TOP + 0.0031)))
        self.assertTrue(standing(stick(0.35, Y, bottom=L.COURT_TOP - 0.0029)))
        self.assertFalse(standing(stick(0.35, Y, bottom=L.COURT_TOP - 0.0031)))
        self.assertAlmostEqual(bottom_gap(stick(0.35, Y, bottom=L.COURT_TOP + 0.002)), 0.002)

    def test_the_centre_is_read_at_the_foot(self):
        """A tilted stick's centre on the court is where its axis meets it (the foot's middle), not its top."""
        s = stick(0.35, Y, tilt=9.0, axis="y")
        self.assertLess(abs(centre_xy(s)[0] - 0.35), 0.001)
        top = s.origin + rotation(s.quat_xyzw) @ np.array([0.0, 0.0, L.STICK_H])
        self.assertGreater(abs(float(top[0]) - 0.35), 0.010)

    def test_valid(self):
        self.assertTrue(valid(S0[0]))
        for bad in (dict(origin=np.array([np.nan, 0, 0])), dict(quat_xyzw=np.zeros(4)),
                    dict(pos=np.array([0.3, np.inf, 0.8])), dict(origin=None), dict(origin=np.zeros(2))):
            s = SimpleNamespace(**{**vars(S0[0]), **bad})
            self.assertFalse(valid(s), bad)
        s = SimpleNamespace(**{k: v for k, v in vars(S0[0]).items() if k != "origin"})
        self.assertFalse(valid(s))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(S0), 0.0)
        r, _ = run([S0] * 30)
        self.assertEqual((r.score, r.success), (0.0, False))

    def test_both_at_the_itf_points_scores_one(self):
        self.assertEqual(score_of(DONE), 1.0)
        r, _ = run(build())
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_itf_window_edges(self):
        for d, want in ((0.0079, 1.0), (-0.0079, 1.0), (0.0081, 0.75), (-0.0081, 0.75)):
            self.assertEqual(score_of(at(Y + d, -Y)), want, d)
            self.assertEqual(score_of(at(Y, -(Y + d))), want, d)

    def test_alley_band_edges(self):
        for a, want in ((0.2701, 0.75), (0.2699, 0.5), (0.3699, 0.75), (0.3701, 0.5)):
            self.assertEqual(score_of(at(a, -Y)), want, a)
            self.assertEqual(score_of(at(Y, -a)), want, a)

    def test_net_line_edges(self):
        for dx, want in ((0.0059, 1.0), (-0.0059, 1.0), (0.0061, 0.5), (-0.0061, 0.5)):
            self.assertEqual(score_of(at(Y, -Y, xa=L.NET_X + dx)), want, dx)
            self.assertEqual(score_of(at(Y, -Y, xb=L.NET_X + dx)), want, dx)

    def test_upright_edges(self):
        for deg, want in ((9.9, 1.0), (10.1, 0.5)):
            self.assertEqual(score_of([stick(L.NET_X, Y, tilt=deg, axis="x"), DONE[1]]), want, deg)
            self.assertEqual(score_of([DONE[0], stick(L.NET_X, -Y, tilt=deg, axis="y")]), want, deg)

    def test_every_outcome_listed_in_task_json(self):
        cases = {
            "both at their ITF points": (DONE, 1.0),
            "one ITF, the other mid-alley": (at(Y, -MID), 0.75),
            "one ITF, the other on its singles sideline": (at(SINGLES, -Y), 0.75),
            "one ITF, the other against its doubles sideline": (at(Y, -DOUBLES), 0.75),
            "both on the singles sidelines": (at(SINGLES, -SINGLES), 0.5),
            "both mid-alley": (at(MID, -MID), 0.5),
            "both against the doubles sidelines": (at(DOUBLES, -DOUBLES), 0.5),
            "one ITF, the other left at the start": ([DONE[0], S0[1]], 0.5),
            "one ITF, the other tipped over": ([DONE[0], stick(L.NET_X, -Y, tilt=90.0)], 0.5),
            "one ITF, the other upside down": ([DONE[0], stick(L.NET_X, -Y, tilt=180.0)], 0.5),
            "one ITF, the other off the net line": (at(Y, -Y, xb=0.30), 0.5),
            "one ITF, the other beside the net post": (at(Y, -BESIDE_POST), 0.5),
            "both in one alley, one at its ITF point": (at(Y, MID), 0.5),
            "one in its alley off the ITF point, the other not set": ([stick(L.NET_X, MID), S0[1]], 0.25),
            "both at the start": (S0, 0.0),
            "both beside the posts": (at(BESIDE_POST, -BESIDE_POST), 0.0),
            "both off the net line at the ITF offset": (at(Y, -Y, xa=0.30, xb=0.40), 0.0),
            "both tipped": ([stick(L.NET_X, Y, tilt=90.0), stick(L.NET_X, -Y, tilt=90.0, axis="y")], 0.0),
        }
        for label, (sticks, want) in cases.items():
            self.assertEqual(score_of(sticks), want, label)
        # still gripped: needs the arms
        grip = snap(left=(L.NET_X, -Y, GRIP_TIPS_Z), left_open=ON, right=(L.NET_X, Y, GRIP_TIPS_Z), right_open=ON)
        self.assertEqual(score_of(DONE, grip), 0.0)
        self.assertEqual(score_of(DONE, snap(left=(L.NET_X, -Y, GRIP_TIPS_Z), left_open=ON)), 0.5)

    def test_every_score_turns_up(self):
        seen = set()
        spots = (Y, MID, SINGLES, DOUBLES, BESIDE_POST, 0.0, 0.15)
        for a, b in itertools.product(spots, spots):
            s = score_of(at(a, -b))
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_either_stick_may_go_either_side(self):
        self.assertEqual(score_of(at(-Y, Y)), 1.0)
        self.assertEqual(score_of(at(-MID, Y)), 0.75)

    def test_position_score(self):
        self.assertEqual(position_score(stick(L.NET_X, Y)), 0.5)
        self.assertEqual(position_score(stick(L.NET_X, -MID)), 0.25)
        self.assertEqual(position_score(stick(L.NET_X, 0.2)), 0.0)
        self.assertAlmostEqual(off_itf(stick(L.NET_X, -(Y + 0.003))), 0.003)
        self.assertAlmostEqual(off_net_line(stick(L.NET_X + 0.002, Y)), 0.002)
        self.assertTrue(on_net_line(stick(L.NET_X + 0.006, Y)))

    def test_success_needs_one_second_of_stillness(self):
        frames = [S0] + glide(S0, DONE, 20)
        t = Tracker()
        results = [t.update(f) for f in frames + [DONE] * 25]
        first = next(i for i, r in enumerate(results) if r.score == 1.0)
        self.assertEqual(first, len(frames) - 1 + 20)  # 20 steps (1 s) after the sticks arrive
        self.assertTrue(results[first].success)

    def test_a_rocking_stick_is_not_still(self):
        """A stick rocking 1 mm back and forth (its centre's path 2 mm in a 0.5 s window) is not set; once it has
        stopped for 1 s it is."""
        rock = [[stick(L.NET_X, Y + (0.0005 if k % 2 else -0.0005)), DONE[1]] for k in range(30)]
        r, t = run([S0] + rock)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["sticks"]["stick_a"]["zero_reason"], "not_still_1s")
        r, _ = run([S0] + rock + [DONE] * 21)
        self.assertEqual(r.score, 1.0)
        # moving in only the older of the two windows still fails
        r, _ = run([S0] + rock + [DONE] * 12)
        self.assertEqual(r.score, 0.5)

    def test_slow_creep_under_the_limit_counts(self):
        frames = [[stick(L.NET_X, Y + k * 0.00012), DONE[1]] for k in range(25)]  # 1.2 mm per 0.5 s
        r, _ = run([S0] + frames)
        self.assertEqual(r.score, 1.0)
        frames = [[stick(L.NET_X, Y + k * 0.00018), DONE[1]] for k in range(25)]  # 1.8 mm per 0.5 s: moving
        r, _ = run([S0] + frames)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["sticks"]["stick_a"]["zero_reason"], "not_still_1s")

    def test_success_only_at_full_score(self):
        for final, want in ((at(Y, -MID), 0.75), (at(MID, -MID), 0.5), (at(MID, -0.2), 0.25)):
            r, _ = run(build(final))
            self.assertEqual((r.score, r.success), (want, False))

    def test_score_is_the_state_at_hand(self):
        r, t = run(build())
        self.assertEqual(r.score, 1.0)
        r2 = t.result(S0)
        self.assertEqual(r2.score, 0.0)
        self.assertEqual(t.result(DONE).score, 1.0)  # idempotent


class ZeroConditions(unittest.TestCase):
    def test_tipped_lying_upside_down(self):
        for s in (stick(L.NET_X, Y, tilt=12.0), stick(L.NET_X, Y, tilt=90.0), stick(L.NET_X, Y, tilt=180.0)):
            self.assertEqual(stick_score(s)[0], 0.0)
            self.assertEqual(stick_score(s)[1], "not_upright")

    def test_raised_or_sunk(self):
        for bottom in (L.COURT_TOP + L.FOOT_SIZE[2], L.COURT_TOP + L.NET_POST_H, L.COURT_TOP + 0.05,
                       L.TOP - 0.10):
            s = stick(L.NET_X, Y, bottom=bottom)
            self.assertEqual(stick_score(s), (0.0, "not_on_the_court"), bottom)

    def test_off_the_net_line(self):
        self.assertEqual(stick_score(stick(0.30, Y)), (0.0, "off_the_net_line"))
        self.assertEqual(stick_score(stick(0.40, Y)), (0.0, "off_the_net_line"))

    def test_outside_the_alley(self):
        for a in (0.0, 0.15, 0.26, 0.38, BESIDE_POST):
            self.assertEqual(stick_score(stick(L.NET_X, a))[0], 0.0, a)

    def test_on_the_centre_line_counts_for_neither_side(self):
        sc, d = evaluate(at(0.0, -Y))
        self.assertEqual(sc, 0.5)
        self.assertIsNone(d["sticks"]["stick_a"]["side"])

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        frames = build()
        t = Tracker()
        for f in frames:
            t.update(f)
        hist = len(t.history)
        broken = SimpleNamespace(**{**vars(DONE[0]), "origin": np.array([np.nan, 0.0, 0.0])})
        for bad in ([broken, DONE[1]], [SimpleNamespace(**{**vars(DONE[0]), "quat_xyzw": np.zeros(4)}), DONE[1]],
                    [SimpleNamespace(**{**vars(DONE[0]), "pos": np.array([np.inf, 0, 0])}), DONE[1]],
                    [DONE[0]], [SimpleNamespace(pos=np.zeros(3)), DONE[1]]):
            r = t.update(bad)
            self.assertEqual((r.score, r.success, r.details["valid_state"]), (0.0, False, False))
            self.assertEqual(len(t.history), hist)
        self.assertEqual(t.bad_steps, 5)
        self.assertEqual(t.update(DONE).score, 1.0)  # picks up where it was

    def test_before_one_second_nothing_is_still(self):
        r, _ = run([DONE] * 20)
        self.assertEqual(r.score, 0.0)
        r, _ = run([DONE] * 21)
        self.assertEqual(r.score, 1.0)


class LetGo(unittest.TestCase):
    """A stick counts only once let go: no gripper's pad-tip point lower than 10 mm above its top within that
    gripper's finger reach of its axis (94 mm fully open, 62 mm closed on the post)."""

    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0), 0.0936, places=4)
        self.assertAlmostEqual(L.release_reach(ON), 0.0616, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=float("nan"))), L.release_reach(1.0))
        self.assertAlmostEqual(reach(SimpleNamespace()), L.release_reach(1.0))

    def test_parked_arms_let_both_go(self):
        score, d = evaluate(DONE, PARKED)
        self.assertEqual(score, 1.0)
        self.assertFalse(any(v["held"] for v in d["sticks"].values()))

    def test_ready_pose_lets_both_go(self):
        """The idle arm's ready pose (pad tips about 0.25 m over the table, 4 cm from the ITF point seen from above)
        never holds a stick."""
        ready = snap(left=(0.323, 0.305, 0.995), right=(0.323, -0.305, 0.995))
        self.assertEqual(score_of(DONE, ready), 1.0)

    def test_a_stick_still_gripped_does_not_count(self):
        grip = snap(left=(L.NET_X, Y, GRIP_TIPS_Z), left_open=ON, right=PARKED["right"].tips)
        self.assertTrue(held(DONE[0], grip))
        score, d = evaluate(DONE, grip)
        self.assertEqual((score, d["sticks"]["stick_a"]["zero_reason"]), (0.5, "still_held"))
        r, _ = run(build(), arms=grip)
        self.assertEqual((r.score, r.success), (0.5, False))

    def test_an_open_gripper_round_the_post_still_holds_it(self):
        self.assertTrue(held(DONE[0], snap(left=(L.NET_X, Y, GRIP_TIPS_Z))))

    def test_a_finger_propping_it_is_holding_it(self):
        low = L.COURT_TOP + 0.040
        self.assertTrue(held(DONE[0], snap(left=(L.NET_X + 0.093, Y, low))))
        self.assertFalse(held(DONE[0], snap(left=(L.NET_X + 0.095, Y, low))))
        self.assertTrue(held(DONE[0], snap(left=(L.NET_X, Y - 0.061, low), left_open=ON)))
        self.assertFalse(held(DONE[0], snap(left=(L.NET_X, Y - 0.063, low), left_open=ON)))

    def test_pad_tips_10_mm_over_the_top_have_let_go(self):
        top = highest(DONE[0])
        self.assertTrue(held(DONE[0], snap(left=(L.NET_X, Y, top + 0.0099))))
        self.assertFalse(held(DONE[0], snap(left=(L.NET_X, Y, top + 0.0101))))
        self.assertEqual(score_of(DONE, snap(left=(L.NET_X, Y, top + 0.011))), 1.0)

    def test_a_tilted_stick_is_measured_along_its_axis(self):
        s = stick(L.NET_X, Y, tilt=9.0, axis="x")  # its top leans toward -y
        u = rotation(s.quat_xyzw)[:, 2]
        z = L.COURT_TOP + 0.040
        p = s.origin + u * ((z - s.origin[2]) / u[2])
        self.assertTrue(held(s, snap(left=(p[0] + 0.093, p[1], z))))
        self.assertFalse(held(s, snap(left=(p[0] + 0.095, p[1], z))))

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        r, t = run(build(), arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(build()))

    def test_one_unreadable_arm_never_hides_the_other(self):
        grip = snap(left=None, right=(L.NET_X, -Y, GRIP_TIPS_Z), right_open=ON)
        self.assertEqual(score_of(DONE, grip), 0.5)
        broken = snap(left=(np.nan, 0, 0), right=(L.NET_X, -Y, GRIP_TIPS_Z), right_open=ON)
        self.assertEqual(score_of(DONE, broken), 0.5)
        r, t = run(build(), arms=grip)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.arms_unavailable, len(build()))


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        t = Tracker()
        for f in build():
            t.update(f)
        self.assertEqual(t.best, 1.0)
        r = t.update(S0)
        self.assertEqual((t.steps, t.best, len(t.history)), (1, 0.0, 1))
        self.assertEqual(r.score, 0.0)

    def test_gradual_return_is_not_a_reset(self):
        t = Tracker()
        for f in build() + glide(DONE, S0, 30):
            t.update(f)
        self.assertEqual(t.best, 1.0)
        self.assertGreater(t.steps, 50)

    def test_jump_elsewhere_is_not_a_reset(self):
        t = Tracker()
        for f in build():
            t.update(f)
        t.update(at(0.10, -0.10, xa=0.25, xb=0.25))
        self.assertEqual(t.best, 1.0)

    def test_jittered_reset_counts(self):
        t = Tracker()
        for f in build():
            t.update(f)
        t.update([stick(0.26, 0.13), stick(0.22, -0.17)])
        self.assertEqual(t.steps, 1)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        if _stub_dir() is None:
            self.skipTest("needs the offline yam_arena stub")
        sys.path.insert(0, str(_stub_dir()))
        task = importlib.import_module("task")
        task._TRACKER.reset()
        for f in build():
            task._TRACKER.update(f)
        self.assertEqual(task._TRACKER.best, 1.0)
        rigstate.offline_reset()
        self.assertEqual((task._TRACKER.steps, task._TRACKER.best), (0, 0.0))


class Exploits(unittest.TestCase):
    """The spec's shortcut checks and the other unintended strategies we tried earn no more than they should."""

    def test_both_on_the_singles_sidelines(self):
        self.assertEqual(score_of(at(SINGLES, -SINGLES)), 0.5)
        self.assertEqual(score_of(at(L.SINGLES_HALF, -L.SINGLES_HALF)), 0.5)

    def test_both_mid_alley_or_against_the_doubles_sidelines(self):
        self.assertEqual(score_of(at(MID, -MID)), 0.5)
        self.assertEqual(score_of(at(DOUBLES, -DOUBLES)), 0.5)
        self.assertEqual(score_of(at(L.DOUBLES_LINE_Y, -L.DOUBLES_LINE_Y)), 0.5)

    def test_both_sticks_in_one_alley(self):
        self.assertEqual(score_of(at(Y, MID)), 0.5)
        self.assertEqual(score_of(at(Y, Y + 0.005)), 0.5)
        self.assertEqual(score_of(at(-Y, -Y + 0.004)), 0.5)

    def test_one_right_the_other_held_or_tipped(self):
        grip = snap(right=(L.NET_X, -Y, GRIP_TIPS_Z), right_open=ON, left=PARKED["left"].tips)
        self.assertEqual(score_of(DONE, grip), 0.5)
        self.assertEqual(score_of([DONE[0], stick(L.NET_X, -Y, tilt=90.0)]), 0.5)

    def test_left_at_the_start_or_beside_the_posts(self):
        self.assertEqual(score_of(S0), 0.0)
        self.assertEqual(score_of(at(BESIDE_POST, -BESIDE_POST)), 0.0)
        self.assertEqual(score_of(at(L.NET_POST_Y - 0.03, -(L.NET_POST_Y - 0.03))), 0.0)

    def test_stacked_or_propped_on_a_post_or_the_other_stick(self):
        on_foot = stick(L.NET_X, Y, bottom=L.COURT_TOP + L.FOOT_SIZE[2])
        on_post = stick(L.NET_X, Y, bottom=L.COURT_TOP + L.NET_POST_H)
        self.assertEqual(score_of([on_foot, DONE[1]]), 0.5)
        self.assertEqual(score_of([on_post, DONE[1]]), 0.5)
        leaning = stick(L.NET_X, Y, tilt=25.0)
        self.assertEqual(score_of([leaning, DONE[1]]), 0.5)

    def test_holding_both_sticks_on_their_points_earns_nothing(self):
        grip = snap(left=(L.NET_X, Y, GRIP_TIPS_Z), left_open=ON, right=(L.NET_X, -Y, GRIP_TIPS_Z), right_open=ON)
        r, _ = run(build(), arms=grip)
        self.assertEqual((r.score, r.success), (0.0, False))

    def test_hedging_two_sticks_across_the_window_in_one_alley(self):
        """Two sticks either side of the ITF point in one alley: at most 0.5, as for one stick."""
        self.assertEqual(score_of(at(Y - 0.007, Y + 0.007)), 0.5)
        self.assertEqual(score_of(at(Y - 0.012, Y + 0.012)), 0.25)

    def test_random_states_score_only_listed_values(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(400):
            ss = []
            for _k in range(2):
                ss.append(stick(rng.uniform(0.25, 0.45), rng.uniform(-0.45, 0.45),
                                bottom=L.COURT_TOP + rng.choice((0.0, 0.0, 0.002, 0.008)),
                                tilt=rng.choice((0.0, 0.0, 5.0, 15.0, 90.0)), yaw=rng.uniform(-90, 90)))
            s = score_of(ss)
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertGreaterEqual(len(seen), 3)


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


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim(self, sticks):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        return {spec.name: self.mk.state_from_poses(spec, (*s.origin, *s.quat_xyzw), source="sim")
                for spec, s in zip(self.task.STICKS, sticks, strict=True)}

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual((t.SCENE, t.SCORE_OVER), ("station", "end"))
        self.assertEqual(t.STICK_NAMES, L.NAMES)
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 9 + 10 + 2 + 2)
        for spec in t.STICKS:
            self.assertIsInstance(spec, self.mk.Assembly)
            self.assertEqual(spec.mass, L.STICK_MASS)
            lo, hi = spec.local_bounds({}) if callable(getattr(spec, "local_bounds", None)) else (None, None)
            if lo is not None:
                self.assertTrue(np.allclose(hi - lo, (0.040, 0.040, 0.071)))
        # every mat's top is the court surface; the lines sit on it; the posts stand on it
        for m in t.MATS:
            self.assertAlmostEqual(m.pos[2] + m.size[2] / 2, L.COURT_TOP)
        for ln in t.LINES:
            self.assertAlmostEqual(ln.pos[2] - ln.size[2] / 2, L.COURT_TOP)
            self.assertTrue(ln.pos[0] + ln.size[0] / 2 <= L.NET_X0 + 1e-9 or ln.pos[0] - ln.size[0] / 2 >= L.NET_X1 - 1e-9)
        # the mats tile the court with no gaps or overlaps: their areas add up to the whole rectangle
        area = sum(m.size[0] * m.size[1] for m in t.MATS)
        self.assertAlmostEqual(area, (L.COURT_X[1] - L.COURT_X[0]) * 2 * L.SURROUND_Y, places=9)
        # the lines' outside edges at the ITF widths
        ys = sorted(round(abs(ln.pos[1]) + L.LINE_W / 2, 5) for ln in t.LINES if abs(ln.pos[1]) > 0.1)
        self.assertEqual(set(ys), {round(L.SINGLES_HALF, 5), round(L.DOUBLES_HALF, 5)})
        for p in t.NET_POSTS:
            self.assertAlmostEqual(abs(p.pos[1]), L.NET_POST_Y)
            self.assertAlmostEqual(p.pos[2] - p.height / 2, L.COURT_TOP)
        # nothing where the arms' start pose hides the table in the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            half_y = o.size[1] / 2 if hasattr(o, "size") and o.size is not None else getattr(o, "radius", 0.02)
            self.assertLessEqual(abs(o.pos[1]) + half_y, 0.45 + 1e-9, o.name)
        for name in L.NAMES:
            self.assertGreaterEqual(L.START[name][0] - 0.02, 0.20)

    def test_starts_clear_of_every_line(self):
        for name in L.NAMES:
            x, y = L.START[name]
            for ly in (0.0, L.SINGLES_LINE_Y, L.DOUBLES_LINE_Y):
                self.assertGreater(abs(abs(y) - ly) - L.FOOT_SIZE[1] / 2 - L.LINE_W / 2, 0.10)
            self.assertGreater(L.NET_X0 - (x + L.FOOT_SIZE[0] / 2), 0.08)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.sim(S0)))
                self.assertEqual(self.task.score(self.sim(S0)), 0.0)
                for _ in range(20):
                    ok = self.task.success(self.sim(DONE))
                self.assertFalse(ok)  # the teleport from the start is not 1 s of stillness yet
                ok = self.task.success(self.sim(DONE))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.sim(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 22)
        last = rows[-1]
        for key in ("episode", "mechkit_episode", "step", "score", "success", "plus_side", "minus_side", "sticks",
                    "bad_steps", "first_step", "arms_unavailable", "stick_a_origin", "stick_b_quat"):
            self.assertIn(key, last)
        a = last["sticks"]["stick_a"]
        for key in ("score", "zero_reason", "side", "x_mm", "y_mm", "off_net_mm", "off_itf_mm", "tilt_deg",
                    "bottom_mm", "held", "still", "path_mm"):
            self.assertIn(key, a)
        self.assertEqual(a["off_itf_mm"], 0.0)
        self.assertEqual(a["path_mm"], [0.0, 0.0])
        self.assertIn("rigstate_error", last)
        self.assertEqual(rows[0]["sticks"]["stick_a"]["zero_reason"], "off_the_net_line")

    def test_hooks_read_the_arms(self):
        def feed(left_tips, left_open):
            rigstate.offline_feed({"left": {"tcp": np.add(left_tips, (0, 0, 0.013)), "tips": left_tips,
                                            "opening": left_open},
                                   "right": {"tcp": PARKED["right"].tcp, "tips": PARKED["right"].tips,
                                             "opening": 1.0}})
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                for _ in range(25):
                    feed((L.NET_X, Y, GRIP_TIPS_Z), ON)
                    self.assertFalse(self.task.success(self.sim(DONE)))
                self.assertEqual(self.task.score(self.sim(DONE)), 0.5)
                feed(tuple(PARKED["left"].tips), 1.0)
                self.assertTrue(self.task.success(self.sim(DONE)))
                self.assertEqual(self.task.score(self.sim(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
                rigstate.offline_reset()
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(rows[-2]["sticks"]["stick_a"]["held"])
        self.assertEqual(rows[-2]["sticks"]["stick_a"]["zero_reason"], "still_held")
        self.assertAlmostEqual(rows[-2]["left_opening"], ON)
        self.assertFalse(rows[-1]["sticks"]["stick_a"]["held"])
        self.assertEqual(rows[-1]["arms_readable"], ["left", "right"])

    def test_mechkit_reset_counter_starts_a_new_trial(self):
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        for _ in range(22):
            self.task.success(self.sim(DONE))
        self.assertEqual(self.task._TRACKER.best, 1.0)
        self.mk.offline_reset()
        self.task.success(self.sim(DONE))
        self.assertEqual((self.task._TRACKER.steps, self.task._TRACKER.best), (1, 0.0))

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(self.sim(S0)))
        finally:
            os.environ.pop("TASK_TRACE", None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                objs = self.sim(S0)
                objs["stick_a"] = SimpleNamespace(pos=np.array([np.nan, 0, 0.8]))
                self.assertFalse(self.task.success(objs))
                self.assertEqual(self.task.score(objs), 0.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                row = json.loads(f.readline())
        self.assertFalse(row["valid_state"])
        self.assertTrue(row["stick_a_state_error"])

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_sets_both_sticks_and_each_arm_works_its_own_side(self):
        t = self.task
        final = list(S0)
        for k, arm in enumerate(("left", "right")):
            sign = 1.0 if arm == "left" else -1.0
            self.assertGreater(sign * L.START[L.NAMES[k]][1], 0.0)
            px, py = t.place_point(arm)
            self.assertGreater(sign * py, 0.0)
            # offline (no lag) 2.0 mm past the net line and 3.4 mm outside the ITF point: inside both windows ...
            self.assertAlmostEqual(px - L.NET_X, 0.0020, places=4)
            self.assertAlmostEqual(abs(py) - Y, 0.0034, places=4)
            # ... and a landing 4 mm short along the carry, or 7 mm short, still inside them
            sx, sy = L.START[L.NAMES[k]]
            d = math.hypot(px - sx, py - sy)
            for short in (0.004, 0.007, 0.0):
                qx, qy = px - short * (px - sx) / d, py - short * (py - sy) / d
                self.assertEqual(score_of(at(qy, -Y, xa=qx) if arm == "left" else at(Y, qy, xb=qx)), 1.0)
            final[k] = stick(px, py)
            # one arm alone: its own side set, the other stick where it started
            part = list(S0)
            part[k] = stick(px, py)
            self.assertEqual(score_of(part), 0.5)
        self.assertEqual(score_of(final), 1.0)

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
        self.assertLess((total + self.task.SUCCESS_HOLD_S + 3.0) * L.CONTROL_HZ, 1200)

    def test_grip_geometry(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.GRIP, L.POST_SIZE[0] - 0.008)  # commanded 8 mm narrower
        self.assertGreaterEqual(-2 * t.OPEN - L.POST_SIZE[0], 0.030)  # 15 mm or more either side of the post
        self.assertGreater(-2 * t.OPEN, L.FOOT_SIZE[0] + 0.001)  # wider than the foot (the offline proxy's box)
        tips = t.GRIP_UP - 0.013
        self.assertGreater(tips - L.FOOT_SIZE[2], 0.025)  # the pad tips well above the foot
        self.assertLess(tips, L.STICK_H - 0.025)  # the pads hold the post's upper part
        self.assertGreaterEqual(t.HOVER_Z - t.GRIP_UP - L.COURT_TOP, 0.060)  # carried well clear of the court
        self.assertGreater(t.UP_Z - 0.013 - (L.COURT_TOP + L.STICK_H), L.RELEASE_ABOVE + 0.03)
        for w in t.demo()["left"]:
            if w.finger in (t.OPEN, t.GRIP) and w.label in ("grip stick_a", "let go of it"):
                self.assertGreaterEqual(w.duration, 1.5)

    def test_instruction_is_the_same_everywhere_and_withholds_the_answer(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        low = self.task.INSTRUCTION.lower()
        for leak in ("0.914", "3 ft", "3 feet", "three feet", "61 mm", "6.1 cm", "two thirds", "33.5", "0.335",
                     "outside the singles"):
            self.assertNotIn(leak, low)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)

    def test_task_json_states_the_thresholds(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("10 mm above", "94 mm", "62 mm", "1.5 mm", "0.5 s", "10 degrees", "3 mm", "6 mm", "x 35.0 cm",
                  "8 mm", "33.53 cm", "27.0 to 37.0 cm", "0.914 m", "2 s", "0, 0.25, 0.5, 0.75 or 1.0"):
            self.assertIn(s, text)
        setup = spec["setup"]
        for s in ("x 24.0 cm, y +15.0 cm", "x 24.0 cm, y -15.0 cm", "40 g", "20 x 20 mm", "63 mm",
                  "40 x 40 x 8 mm", "71 mm", "27.43", "36.57", "42.66", "x 34.5 to 35.5 cm", "25 mm across", "ABS",
                  "vinyl", "wooden"):
            self.assertIn(s, setup)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(spec["run_limit"], "1,200 steps (1 minute at 20 Hz)")
        self.assertEqual(spec["scores"], "TBD after the GPU runs (offline kinematic check predicts demo: 1.0)")

    def test_no_team_identifying_text_and_no_dashes(self):
        for p in [HERE / "task.json", HERE / "README.md", HERE / "task.py", HERE / "layout.py", HERE / "scoring.py",
                  pathlib.Path(__file__)]:
            text = p.read_text()
            low = text.lower()
            if p.name != "test_scoring.py":
                self.assertFalse(_ident_hits(low), p.name)
            self.assertNotIn(chr(0x2014), text, p.name)
            self.assertNotIn(chr(0x2013), text, p.name)

    def test_success_hold_outlasts_every_release(self):
        """With the arms unreadable a stick would count 1 s after its slow descent ends, while still gripped: the
        success hold plus that 1 s must outlast the settle and the release after it."""
        plan = self.task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly stand it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    self.assertLess(sum(a.duration for a in after) + 0.5,
                                    self.task.SUCCESS_HOLD_S + L.STILL_WINDOW_S * L.STILL_WINDOWS)
                    n += 1
        self.assertEqual(n, 2)


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the other side's stick or ITF point."""

    def test_other_sides_stick_and_point_are_out_of_reach(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        z = L.COURT_TOP + 0.050
        for arm, sign in (("left", -1.0), ("right", 1.0)):
            base = K.BASES["station"][arm]
            other = L.NAMES[1] if arm == "left" else L.NAMES[0]
            for x, y in (L.START[other], (L.NET_X, sign * Y)):
                for yaw in np.linspace(-math.pi, math.pi, 13):
                    self.assertFalse(K.reachable((x, y, z), float(yaw), base)[0], (arm, x, y, yaw))
        for arm, sign in (("left", 1.0), ("right", -1.0)):
            base = K.BASES["station"][arm]
            own = L.NAMES[0] if arm == "left" else L.NAMES[1]
            for x, y in (L.START[own], (L.NET_X, sign * Y)):
                self.assertTrue(K.reachable((x, y, z), 0.0, base)[0], (arm, x, y))



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
