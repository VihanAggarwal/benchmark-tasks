"""Offline tests for scoring.py (and task.py's hooks): synthetic dish states, no simulator.

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
    Tracker, evaluate, extremes, fish_directions, footprint, held, overlap_depth, overlaps, point_to_footprint,
    quadrant_inset, rotation, tilt_deg, tray_overhang, valid,
)

MM = 0.001
TRAY_TOP_W = L.TOP + L.TRAY_TOP
BOARD_TOP_W = L.TOP + L.BOARD_TOP
DEG = math.pi / 180


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(math.sin(h) * np.array(v, float)) + (math.cos(h),)


def _local_box(name):
    if name == "fish":
        return np.array([-L.PLATE_L / 2, -L.PLATE_W / 2, 0.0]), np.array([L.PLATE_L / 2, L.PLATE_W / 2, L.FISH_TOP])
    r, h = L.BOWL[name]
    return np.array([-r, -r, 0.0]), np.array([r, r, h])


def dish(name, x, y, bottom=TRAY_TOP_W, yaw=0.0, tilt=0.0, axis="x", vel=(0, 0, 0)):
    """A dish's state: turned ``yaw`` degrees about the vertical, then tilted ``tilt`` degrees about a world axis,
    with its lowest point at ``bottom`` and the middle of its base over (x, y) (before the tilt moves it)."""
    q = qmul(about(axis, tilt), about("z", yaw))
    probe = SimpleNamespace(origin=np.array([x, y, 0.0]), quat_xyzw=np.array(q, float))
    low, _ = extremes(name, probe)
    origin = np.array([x, y, bottom - low])
    lo, hi = _local_box(name)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    world = corners @ rotation(q).T + origin
    return SimpleNamespace(origin=origin, quat_xyzw=np.array(q, float), pos=(world.min(0) + world.max(0)) / 2,
                           lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=world.min(0), high=world.max(0))


def at_start(name):
    x, y, yaw = L.START[name]
    return dish(name, x, y, BOARD_TOP_W, math.degrees(yaw))


S0 = {n: at_start(n) for n in L.DISHES}
# The demo's places: each dish 55 mm inside the centre line across and 75 mm from the one front to back.
TARGET = {"rice": (0.385, -0.075), "soup": (0.385, 0.075), "side": (0.275, -0.075), "fish": (0.275, 0.075)}
FISH_YAW = -90.0  # the head to -y (the guest's left), the belly to +x (toward the guest)


def laid(**over):
    """Every dish on its target (the fish at yaw -90); ``over``: name -> state replaces that dish."""
    out = {n: dish(n, *TARGET[n], yaw=FISH_YAW if n == "fish" else 0.0) for n in L.DISHES}
    out.update(over)
    return out


DONE = laid()


def score_of(dishes, arms=None, tipped=(), still=True):
    paths = {n: 0.0 for n in L.DISHES} if still else None
    return evaluate(dishes, arms, tipped, paths)[0]


def run(frames, hz=20.0, arms=None):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


def glide(a, b, n):
    """n frames moving every dish of frame a to its place in frame b in equal steps (orientations of b)."""
    out = []
    for i in range(1, n + 1):
        f = {}
        for k in L.DISHES:
            s = b[k]
            d = (1 - i / n) * (a[k].origin - s.origin)
            f[k] = SimpleNamespace(origin=s.origin + d, quat_xyzw=s.quat_xyzw, pos=s.pos + d, lin_vel=s.lin_vel,
                                   ang_vel=s.ang_vel, low=s.low + d, high=s.high + d)
        out.append(f)
    return out


def build(final=None, n=20, hold=30):
    final = DONE if final is None else final
    return [S0] + glide(S0, final, n) + [final] * hold


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    """A rigstate.arms() snapshot: each given arm's pad-tip point (x, y, z) and opening (0 closed .. 1 open)."""
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


# Both arms parked where the demo leaves them, pad tips 137 mm up.
PARKED = snap(left=(0.200, 0.250, L.TOP + 0.137), right=(0.200, -0.250, L.TOP + 0.137))
ON_RICE = 0.066 / L.FULL_GAP  # rigstate's opening with the fingers closed on the rice bowl


class Geometry(unittest.TestCase):
    def test_tilt_and_extremes(self):
        s = dish("rice", 0.3, 0.0)
        self.assertAlmostEqual(tilt_deg(s), 0.0)
        self.assertAlmostEqual(extremes("rice", s)[0], TRAY_TOP_W)
        self.assertAlmostEqual(extremes("rice", s)[1], TRAY_TOP_W + L.RICE_H)
        t = dish("soup", 0.3, 0.0, tilt=30.0)
        self.assertAlmostEqual(tilt_deg(t), 30.0, places=6)
        self.assertAlmostEqual(extremes("soup", t)[0], TRAY_TOP_W)
        self.assertAlmostEqual(extremes("soup", t)[1] - TRAY_TOP_W,
                               L.SOUP_H * math.cos(30 * DEG) + 2 * L.SOUP_R * math.sin(30 * DEG), places=9)
        u = dish("side", 0.3, 0.0, tilt=180.0)  # upside down
        self.assertAlmostEqual(tilt_deg(u), 180.0, places=6)
        self.assertAlmostEqual(extremes("side", u)[1] - extremes("side", u)[0], L.SIDE_H)
        p = dish("fish", 0.3, 0.0, yaw=37.0)
        self.assertAlmostEqual(extremes("fish", p)[1] - TRAY_TOP_W, L.FISH_TOP)

    def test_fish_directions(self):
        for yaw, want in ((-90.0, (0.0, 0.0)), (90.0, (180.0, 180.0)), (0.0, (90.0, 90.0)), (180.0, (90.0, 90.0)),
                          (-75.0, (15.0, 15.0)), (-110.0, (20.0, 20.0))):
            h, b = fish_directions(dish("fish", 0.3, 0.1, yaw=yaw))
            self.assertAlmostEqual(h, want[0], places=6, msg=yaw)
            self.assertAlmostEqual(b, want[1], places=6, msg=yaw)
        # Upside down (flipped about x), the head still points -y but the belly points -x: it fails, and is tipped.
        h, b = fish_directions(dish("fish", 0.3, 0.1, yaw=-90.0, tilt=180.0, axis="y"))
        self.assertAlmostEqual(h, 0.0, places=6)
        self.assertAlmostEqual(b, 180.0, places=6)

    def test_footprints(self):
        a, b = footprint("rice", dish("rice", 0.3, 0.0)), footprint("side", dish("side", 0.3, 0.0605))
        self.assertAlmostEqual(overlap_depth(a, b), 0.0)  # just touching: 33 + 27.5 mm apart
        c = footprint("side", dish("side", 0.3, 0.0585))
        self.assertAlmostEqual(overlap_depth(a, c), 0.002)
        f = footprint("fish", dish("fish", 0.3, 0.0, yaw=-90.0))  # 60 mm along x, 130 mm along y
        self.assertAlmostEqual(point_to_footprint((0.3 + 0.030 + 0.01, 0.0), f), 0.01)
        self.assertAlmostEqual(point_to_footprint((0.3, 0.065 + 0.02), f), 0.02)
        self.assertEqual(point_to_footprint((0.31, 0.05), f), 0.0)
        bowl = footprint("soup", dish("soup", 0.3 + 0.030 + 0.032 + 0.004, 0.0))  # 4 mm off the plate's side
        self.assertAlmostEqual(overlap_depth(bowl, f), -0.004)
        inside = footprint("soup", dish("soup", 0.3, 0.02))  # centre on the plate
        self.assertGreater(overlap_depth(inside, f), 0.03)
        self.assertAlmostEqual(overlap_depth(f, inside), overlap_depth(inside, f))

    def test_tray_overhang_and_inset(self):
        self.assertAlmostEqual(tray_overhang(footprint("rice", dish("rice", L.TRAY_X1 - L.RICE_R + 0.004, -0.07))),
                               0.004)
        f = footprint("fish", dish("fish", 0.275, L.TRAY_Y1 - 0.065 + 0.003, yaw=-90.0))
        self.assertAlmostEqual(tray_overhang(f), 0.003)
        self.assertAlmostEqual(quadrant_inset("rice", dish("rice", 0.385, -0.075)), 0.055)
        self.assertAlmostEqual(quadrant_inset("fish", dish("fish", 0.275, 0.075, yaw=-90)), 0.055)
        self.assertLess(quadrant_inset("rice", dish("rice", 0.275, -0.075)), 0)  # the back row

    def test_valid(self):
        self.assertTrue(valid(DONE["rice"]))
        s = DONE["rice"]
        for bad in (dict(origin=np.array([np.nan, 0, 0])), dict(quat_xyzw=np.zeros(4)), dict(origin=None),
                    dict(pos=np.array([np.inf, 0, 0])), dict(quat_xyzw=np.array([0, 0, 1.0]))):
            self.assertFalse(valid(SimpleNamespace(**{**vars(s), **bad})), bad)
        self.assertFalse(valid(SimpleNamespace(quat_xyzw=s.quat_xyzw, pos=s.pos)))  # a plain box: no origin

    def test_layout_arithmetic(self):
        # The demo's targets: 55 mm inside both centre lines' nearer one, every footprint on the tray with room,
        # nothing touching.
        for n, (x, y) in TARGET.items():
            s = DONE[n]
            self.assertGreaterEqual(quadrant_inset(n, s), 0.055 - 1e-9, n)
            self.assertLessEqual(tray_overhang(footprint(n, s)), -0.010 + 1e-9, n)
        self.assertEqual(overlaps(DONE), {n: [] for n in L.DISHES})
        self.assertAlmostEqual(L.MID_X, 0.330)
        # The starts: on the serving boards, off the tray, apart, out of the strips the arms hide in the setup
        # screenshot (|y| > 0.45 m with x < 0.2 m, and the hanging grippers over x 0.41 to 0.57 m, |y| 0.30 to
        # 0.58 m).
        for n, s in S0.items():
            f = footprint(n, s)
            self.assertGreater(tray_overhang(f), 0.05, n)
            lo, hi = s.low, s.high
            self.assertTrue(L.BOARD_X0 <= lo[0] and hi[0] <= L.BOARD_X1, n)
            self.assertTrue(L.BOARD_Y_IN <= min(abs(lo[1]), abs(hi[1])) and max(abs(lo[1]), abs(hi[1]))
                            <= L.BOARD_Y_OUT, n)
            self.assertLess(hi[0], 0.40, n)
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, n)
        gaps = [-overlap_depth(footprint(a, S0[a]), footprint(b, S0[b])) for a, b in (("rice", "side"),
                                                                                    ("soup", "fish"))]
        self.assertGreater(min(gaps), 0.025)
        # The fish lies within its plate and the gripped edges are clear of it.
        self.assertLessEqual(L.FISH_BODY_SIZE[1] / 2, L.PLATE_W / 2 - 0.012)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(S0), 0.0)
        r, _ = run([S0] * 30)
        self.assertEqual(r.score, 0.0)

    def test_full_layout_scores_one(self):
        self.assertEqual(score_of(DONE), 1.0)
        self.assertEqual(score_of(DONE, PARKED), 1.0)
        r, _ = run(build())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_fish_head_to_the_guests_right_or_lengthwise(self):
        for yaw in (90.0, 0.0, 180.0, -60.0, -120.0):
            self.assertEqual(score_of(laid(fish=dish("fish", 0.275, 0.075, yaw=yaw))), 0.75, yaw)
        for yaw in (-76.0, -104.0, -90.0):
            self.assertEqual(score_of(laid(fish=dish("fish", 0.275, 0.075, yaw=yaw))), 1.0, yaw)

    def test_rice_and_side_dish_swapped_front_to_back(self):
        d = laid(rice=dish("rice", 0.275, -0.075), side=dish("side", 0.385, -0.075))
        self.assertEqual(score_of(d), 0.5)

    def test_rice_and_soup_swapped_left_to_right(self):
        d = laid(rice=dish("rice", 0.385, 0.075), soup=dish("soup", 0.385, -0.075))
        self.assertEqual(score_of(d), 0.5)

    def test_each_tier(self):
        self.assertEqual(score_of(laid(rice=S0["rice"])), 0.75)
        self.assertEqual(score_of(laid(rice=S0["rice"], side=S0["side"])), 0.5)  # left arm alone
        self.assertEqual(score_of(laid(soup=S0["soup"], fish=S0["fish"])), 0.5)  # right arm alone
        self.assertEqual(score_of({**S0, "fish": DONE["fish"]}), 0.25)
        self.assertEqual(score_of(S0), 0.0)
        for n in L.DISHES:
            self.assertEqual(score_of(laid(**{n: S0[n]})), 0.75, n)
            self.assertEqual(score_of({**S0, n: DONE[n]}), 0.25, n)

    def test_every_outcome_listed_in_task_json(self):
        """The outcomes task.json's scoring lists, with their scores."""
        robot_view = {"rice": dish("rice", 0.275, 0.075), "soup": dish("soup", 0.275, -0.075),
                      "fish": dish("fish", 0.385, -0.075, yaw=90.0), "side": dish("side", 0.385, 0.075)}
        cases = [
            (DONE, 1.0),
            (laid(fish=dish("fish", 0.275, 0.075, yaw=90.0)), 0.75),
            (laid(fish=dish("fish", 0.275, 0.075, yaw=0.0)), 0.75),
            (laid(rice=dish("rice", 0.275, -0.075), side=dish("side", 0.385, -0.075)), 0.5),
            (laid(rice=dish("rice", 0.300, -0.075)), 0.5),  # pushed into the side dish: both 0
            (laid(soup=S0["soup"], fish=S0["fish"]), 0.5),
            ({**S0, "fish": DONE["fish"]}, 0.25),
            (robot_view, 0.0),
        ]
        for d, want in cases:
            self.assertEqual(score_of(d), want)
        for n in L.DISHES:  # every dish of the robot's-view layout is in the opposite quadrant
            self.assertLess(quadrant_inset(n, robot_view[n]), 0)
        # The soup tipped on the way and set upright in its place: 0.75 (tracker test below).

    def test_score_is_the_state_at_hand(self):
        frames = build() + [laid(rice=S0["rice"])] * 25
        r, _ = run(frames)
        self.assertEqual(r.score, 0.75)

    def test_success_needs_a_second_of_stillness(self):
        t = Tracker()
        for f in [S0] + glide(S0, DONE, 10):
            t.update(f)
        r = t.update(DONE)
        self.assertEqual(r.score, 0.0)  # every dish moved more than 3 mm over the last second
        for _ in range(25):
            r = t.update(DONE)
        self.assertEqual(r.score, 1.0)

    def test_every_score_turns_up(self):
        seen = {score_of(laid(**{n: S0[n] for n in L.DISHES[:k]})) for k in range(5)}
        self.assertLessEqual({0.0, 0.25, 0.5, 0.75}, seen)  # 1.0 needs all four right: see the tier tests


class ZeroConditions(unittest.TestCase):
    def test_quadrant_inset_band(self):
        x = L.MID_X + L.INSET
        self.assertEqual(score_of(laid(rice=dish("rice", x + 0.0005, -0.075))), 1.0)
        self.assertEqual(score_of(laid(rice=dish("rice", x - 0.0005, -0.075))), 0.75)
        self.assertEqual(score_of(laid(rice=dish("rice", L.MID_X, -0.115))), 0.75)  # astride the line
        self.assertEqual(score_of(laid(soup=dish("soup", 0.385, 0.0095))), 0.75)
        self.assertEqual(score_of(laid(soup=dish("soup", 0.385, 0.0105))), 1.0)

    def test_overhang_band(self):
        r = L.RICE_R
        self.assertEqual(score_of(laid(rice=dish("rice", L.TRAY_X1 - r + 0.0045, -0.075))), 1.0)
        self.assertEqual(score_of(laid(rice=dish("rice", L.TRAY_X1 - r + 0.0055, -0.075))), 0.75)
        y_edge = L.TRAY_Y1 - L.PLATE_L / 2
        self.assertEqual(score_of(laid(fish=dish("fish", 0.275, y_edge + 0.0045, yaw=-90))), 1.0)
        self.assertEqual(score_of(laid(fish=dish("fish", 0.275, y_edge + 0.0055, yaw=-90))), 0.75)

    def test_off_the_tray_or_held_up(self):
        self.assertEqual(score_of(laid(side=dish("side", 0.275, -0.075, bottom=L.TOP))), 0.75)  # on the table
        self.assertEqual(score_of(laid(side=dish("side", 0.275, -0.075, bottom=TRAY_TOP_W + 0.0035))), 0.75)
        self.assertEqual(score_of(laid(side=dish("side", 0.275, -0.075, bottom=TRAY_TOP_W + 0.0025))), 1.0)

    def test_upright_band(self):
        self.assertEqual(score_of(laid(soup=dish("soup", 0.385, 0.075, tilt=9.5))), 1.0)
        self.assertEqual(score_of(laid(soup=dish("soup", 0.385, 0.075, tilt=10.5))), 0.75)
        self.assertEqual(score_of(laid(fish=dish("fish", 0.275, 0.075, yaw=-90, tilt=10.5, axis="x"))), 0.75)

    def test_fish_direction_band(self):
        self.assertEqual(score_of(laid(fish=dish("fish", 0.275, 0.075, yaw=-90 + 14.5))), 1.0)
        self.assertEqual(score_of(laid(fish=dish("fish", 0.275, 0.075, yaw=-90 - 15.5))), 0.75)

    def test_not_still(self):
        self.assertEqual(score_of(DONE, still=False), 0.0)
        paths = {n: 0.0 for n in L.DISHES}
        self.assertEqual(evaluate(DONE, None, (), {**paths, "rice": 0.0031})[0], 0.75)
        self.assertEqual(evaluate(DONE, None, (), {**paths, "rice": 0.0029})[0], 1.0)

    def test_overlap_zeroes_both(self):
        d = laid(rice=dish("rice", 0.330, -0.075))  # rice pushed back over the side dish's footprint
        self.assertEqual(overlaps(d)["rice"], ["side"])
        self.assertEqual(score_of(d), 0.5)
        # Touching within 1 mm is no overlap; 2 mm is.
        x_touch = 0.275 + L.SIDE_R + L.RICE_R
        self.assertEqual(overlaps(laid(rice=dish("rice", x_touch - 0.0009, -0.075)))["rice"], [])
        self.assertEqual(overlaps(laid(rice=dish("rice", x_touch - 0.002, -0.075)))["rice"], ["side"])

    def test_tip_latch(self):
        tipped = dish("soup", 0.385, 0.075, tilt=50.0)
        frames = build() + [laid(soup=tipped)] * 3 + [DONE] * 30
        r, t = run(frames)
        self.assertEqual(r.score, 0.75)
        self.assertIn("soup", t.tipped)
        self.assertEqual(r.details["dishes"]["soup"]["zero_reason"], "tipped")
        r2, t2 = run(build() + [laid(soup=dish("soup", 0.385, 0.075, tilt=44.0))] * 3 + [DONE] * 30)
        self.assertEqual(r2.score, 1.0)  # 44 degrees does not latch
        t.reset()
        self.assertEqual(t.tipped, {})

    def test_soup_tipped_on_the_way_and_set_upright(self):
        frames = [S0] + glide(S0, laid(soup=dish("soup", 0.30, 0.20, tilt=60.0, bottom=L.TOP + 0.05)), 10) \
            + glide(laid(soup=dish("soup", 0.30, 0.20, tilt=60.0, bottom=L.TOP + 0.05)), DONE, 10) + [DONE] * 30
        r, _ = run(frames)
        self.assertEqual(r.score, 0.75)

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        frames = build()
        r, t = run(frames)
        steps, hist = t.steps, len(t.history)
        bad = dict(DONE)
        bad["fish"] = SimpleNamespace(**{**vars(DONE["fish"]), "origin": np.array([np.nan, 0.0, 0.0])})
        r = t.update(bad)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(r.details["bad_dishes"], ["fish"])
        self.assertEqual(len(t.history), hist)
        self.assertEqual(t.bad_steps, 1)
        missing = {k: v for k, v in DONE.items() if k != "rice"}
        self.assertEqual(t.update(missing).score, 0.0)
        zero_q = dict(DONE)
        zero_q["side"] = SimpleNamespace(**{**vars(DONE["side"]), "quat_xyzw": np.zeros(4)})
        self.assertEqual(t.update(zero_q).score, 0.0)
        self.assertEqual(t.update(DONE).score, 1.0)  # the history was kept
        self.assertEqual(t.steps, steps + 4)

    def test_missing_velocity_is_tolerated(self):
        d = {n: SimpleNamespace(**{k: v for k, v in vars(s).items() if k != "lin_vel"}) for n, s in DONE.items()}
        r, _ = run([d] * 25)
        self.assertEqual(r.score, 1.0)


class LetGo(unittest.TestCase):
    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0), 0.0654, places=4)
        self.assertAlmostEqual(L.release_reach(ON_RICE), 0.0524, places=4)

    def test_a_dish_still_gripped_does_not_count(self):
        x, y = TARGET["rice"]
        grip = snap(right=(x, y, TRAY_TOP_W + 0.013), right_open=ON_RICE, left=PARKED["left"].tips)
        self.assertTrue(held("rice", DONE["rice"], grip))
        self.assertEqual(score_of(DONE, grip), 0.75)

    def test_an_open_gripper_round_a_dish_still_holds_it(self):
        x, y = TARGET["fish"]
        arms = snap(left=(x, y, TRAY_TOP_W + 0.004), left_open=1.0, right=PARKED["right"].tips)
        self.assertEqual(score_of(DONE, arms), 0.75)

    def test_a_finger_beside_a_dish_is_holding_it(self):
        x, y = TARGET["soup"]
        side = L.SOUP_R + L.release_reach(1.0) - 0.002
        arms = snap(left=(x + side, y, TRAY_TOP_W + 0.02), right=PARKED["right"].tips)
        self.assertTrue(held("soup", DONE["soup"], arms))
        arms = snap(left=(x + side + 0.004, y, TRAY_TOP_W + 0.02), right=PARKED["right"].tips)
        self.assertFalse(held("soup", DONE["soup"], arms))

    def test_pad_tips_10_mm_over_the_top_have_let_go(self):
        x, y = TARGET["side"]
        top = TRAY_TOP_W + L.SIDE_H
        self.assertTrue(held("side", DONE["side"], snap(right=(x, y, top + 0.0095))))
        self.assertFalse(held("side", DONE["side"], snap(right=(x, y, top + 0.0105))))

    def test_success_needs_every_dish_let_go(self):
        x, y = TARGET["soup"]
        grip = snap(left=(x, y, TRAY_TOP_W + 0.014), left_open=0.064 / L.FULL_GAP, right=PARKED["right"].tips)
        r, _ = run(build(), arms=grip)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        self.assertEqual(r.details["dishes"]["soup"]["zero_reason"], "held")

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        r, t = run(build(), arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(build()))

    def test_one_unreadable_arm_never_hides_the_other(self):
        x, y = TARGET["rice"]
        arms = snap(right=(x, y, TRAY_TOP_W + 0.013), right_open=ON_RICE)  # left missing
        r, t = run(build(), arms=arms)
        self.assertEqual(r.score, 0.75)
        self.assertEqual(t.arms_unavailable, len(build()))
        arms["right"] = SimpleNamespace(tcp=None, tips=np.array([np.nan, 0, 0]), opening=1.0)
        self.assertEqual(score_of(DONE, arms), 1.0)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        calls = []
        rigstate.on_reset(lambda: calls.append(1))
        rigstate.offline_reset()
        self.assertTrue(calls)


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        tipped = dish("soup", 0.385, 0.075, tilt=50.0)
        r, t = run(build() + [laid(soup=tipped)] * 2 + [DONE] * 5)
        self.assertIn("soup", t.tipped)
        r = t.update(S0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.tipped, {})
        for f in glide(S0, DONE, 20) + [DONE] * 25:
            r = t.update(f)
        self.assertEqual(r.score, 1.0)

    def test_gradual_return_is_not_a_reset(self):
        r, t = run(build() + glide(DONE, S0, 20) + [S0] * 3)
        self.assertGreater(t.steps, 50)

    def test_jump_elsewhere_is_not_a_reset(self):
        r, t = run(build() + [laid(rice=dish("rice", 0.2, -0.45))])
        self.assertGreater(t.steps, 50)

    def test_jittered_reset_counts(self):
        jit = {n: dish(n, L.START[n][0] + 0.02, L.START[n][1] - 0.02, BOARD_TOP_W, math.degrees(L.START[n][2]))
               for n in L.DISHES}
        r, t = run(build() + [jit])
        self.assertEqual(t.steps, 1)


class Exploits(unittest.TestCase):
    def test_laid_from_the_robots_own_side(self):
        """The shortcut the task tests: the layout read from the robot's side. Every dish lands in the opposite
        quadrant and the fish points the wrong way: 0."""
        robot_view = {"rice": dish("rice", 0.275, 0.075), "soup": dish("soup", 0.275, -0.075),
                      "fish": dish("fish", 0.385, -0.075, yaw=90.0), "side": dish("side", 0.385, 0.075)}
        r, _ = run(build(robot_view))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(all(d["zero_reason"] in ("wrong_quadrant", "fish_turned_wrong")
                            for d in r.details["dishes"].values()))

    def test_mirror_images_of_the_layout(self):
        # Left and right swapped (rice and side on +y, soup and fish on -y, the fish head still to -y): 0.
        mirror = {"rice": dish("rice", 0.385, 0.075), "soup": dish("soup", 0.385, -0.075),
                  "side": dish("side", 0.275, 0.075), "fish": dish("fish", 0.275, -0.075, yaw=-90.0)}
        self.assertEqual(score_of(mirror), 0.0)
        # Front and back swapped: 0.
        flipped = {"rice": dish("rice", 0.275, -0.075), "soup": dish("soup", 0.275, 0.075),
                   "side": dish("side", 0.385, -0.075), "fish": dish("fish", 0.385, 0.075, yaw=-90.0)}
        self.assertEqual(score_of(flipped), 0.0)

    def test_every_dish_at_the_tray_centre_hedging(self):
        hedge = {"rice": dish("rice", 0.335, -0.035), "soup": dish("soup", 0.335, 0.035),
                 "side": dish("side", 0.300, -0.010), "fish": dish("fish", 0.300, 0.030, yaw=-90.0)}
        self.assertEqual(score_of(hedge), 0.0)

    def test_a_bowl_set_on_the_fish_plate(self):
        on_plate = laid(side=dish("side", 0.275, 0.045, bottom=TRAY_TOP_W + L.FISH_TOP))
        self.assertEqual(score_of(on_plate), 0.5)  # both the side dish and the fish 0

    def test_bowls_stacked(self):
        stacked = laid(side=dish("side", 0.385, -0.075, bottom=TRAY_TOP_W + L.RICE_H))
        self.assertEqual(score_of(stacked), 0.5)

    def test_holding_dishes_in_place_earns_nothing_for_them(self):
        (rx, ry), (sx, sy) = TARGET["rice"], TARGET["soup"]
        arms = snap(right=(rx, ry, TRAY_TOP_W + 0.012), right_open=ON_RICE,
                    left=(sx, sy, TRAY_TOP_W + 0.014), left_open=0.064 / L.FULL_GAP)
        self.assertEqual(score_of(DONE, arms), 0.5)

    def test_plate_stood_on_its_edge_latches(self):
        edge = dish("fish", 0.275, 0.075, yaw=-90.0, tilt=90.0, axis="y")
        r, t = run(build() + [laid(fish=edge)] * 2 + [DONE] * 25)
        self.assertEqual(r.score, 0.75)

    def test_random_states_score_only_listed_values(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(400):
            d = {}
            for n in L.DISHES:
                u = rng.random()
                if u < 0.2:
                    d[n] = S0[n]
                elif u < 0.6:  # near its own place, sometimes turned or tilted
                    tx, ty = TARGET[n]
                    d[n] = dish(n, tx + rng.uniform(-0.03, 0.03), ty + rng.uniform(-0.03, 0.03),
                                yaw=(FISH_YAW if n == "fish" else 0.0) + rng.uniform(-30, 30),
                                tilt=rng.choice([0.0, 5.0, 15.0]), axis="x")
                else:
                    d[n] = dish(n, rng.uniform(0.20, 0.46), rng.uniform(-0.17, 0.17), yaw=rng.uniform(-180, 180),
                                tilt=rng.choice([0.0, 5.0, 15.0]), axis="x")
            s = score_of(d)
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertLessEqual({0.0, 0.25, 0.5, 0.75}, seen)  # 1.0 needs all four right: see the tier tests


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


def _mech_state(spec, s):
    """An AssemblyState for a task object from a synthetic dish state (the offline checker's path)."""
    return spec.offline_state(SimpleNamespace(pos=s.pos, quat_xyzw=s.quat_xyzw, lin_vel=s.lin_vel,
                                              ang_vel=s.ang_vel, low=s.low, high=s.high))


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, dishes):
        return {self.task.OBJECT_OF[n]: _mech_state(self.task.DISH_SPECS[n], s) for n, s in dishes.items()}

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        loose = {o.name: o.mass for o in t.OBJECTS if o.mass is not None}
        self.assertEqual(loose, {"rice_bowl": L.RICE_MASS, "side_dish_bowl": L.SIDE_MASS, "soup_bowl": L.SOUP_MASS,
                                 "fish_plate": L.FISH_MASS})
        self.assertTrue(all(m <= 0.3 for m in loose.values()))
        for n, spec in t.DISH_SPECS.items():
            st = spec.rest_state()
            self.assertTrue(np.allclose(st.origin, S0[n].origin, atol=1e-9), n)  # the base frame = the dish's base
            self.assertTrue(np.allclose(st.low, S0[n].low, atol=1e-9), n)
            self.assertTrue(np.allclose(st.high, S0[n].high, atol=1e-9), n)
        self.assertAlmostEqual(t.TRAY.pos[2] + t.TRAY.size[2] / 2, TRAY_TOP_W)
        for b in t.BOARDS:
            self.assertAlmostEqual(b.pos[2] + b.size[2] / 2, BOARD_TOP_W)
        self.assertGreater(L.CUSHION_X0, L.TRAY_X1)  # the guest sits beyond the tray's far edge
        for o in t.OBJECTS:
            half_y = o.size[1] / 2 if hasattr(o, "size") and o.size is not None else 0.0
            self.assertLess(abs(o.pos[1]) + half_y, 0.45, o.name)

    def test_offline_state_reads_the_pose(self):
        for n in L.DISHES:
            st = _mech_state(self.task.DISH_SPECS[n], DONE[n])
            self.assertTrue(np.allclose(st.origin, DONE[n].origin, atol=1e-9), n)
        self.assertEqual(score_of({n: _mech_state(self.task.DISH_SPECS[n], DONE[n]) for n in L.DISHES}), 1.0)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                for _ in range(21):
                    ok = self.task.success(self.objects(DONE))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 22)
        for key in ("episode", "mechkit_episode", "step", "score", "success", "dishes", "bad_steps", "first_step",
                    "arms_unavailable", "tipped_at_step", "rigstate_error"):
            self.assertIn(key, rows[-1])
        fish = rows[-1]["dishes"]["fish"]
        for key in ("centre_mm", "quadrant", "inset_mm", "overhang_mm", "rest_mm", "tilt_deg", "held", "path_mm",
                    "still", "overlaps", "head_err_deg", "belly_err_deg", "head_heading_deg", "points",
                    "zero_reason"):
            self.assertIn(key, fish)
        self.assertEqual(fish["quadrant"], "back right")
        self.assertEqual(rows[-1]["dishes"]["rice"]["quadrant"], "front left")
        self.assertAlmostEqual(fish["head_heading_deg"], -90.0)
        self.assertEqual(rows[0]["dishes"]["rice"]["zero_reason"], "wrong_quadrant")

    def test_hooks_read_the_arms(self):
        x, y = TARGET["soup"]

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
                for _ in range(22):
                    feed((x, y, TRAY_TOP_W + 0.014), 0.064 / L.FULL_GAP)
                    self.assertFalse(self.task.success(self.objects(DONE)))
                self.assertEqual(self.task.score(self.objects(DONE)), 0.75)
                feed(tuple(PARKED["left"].tips), 1.0)
                self.assertTrue(self.task.success(self.objects(DONE)))
            finally:
                os.environ.pop("TASK_TRACE", None)
                rigstate.offline_reset()
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(rows[-2]["dishes"]["soup"]["held"])
        self.assertFalse(rows[-1]["dishes"]["soup"]["held"])
        self.assertEqual(rows[-1]["arms_readable"], ["left", "right"])
        self.assertIn("left_tips", rows[-1])

    def test_trace_survives_a_bad_path_and_a_broken_state(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(self.objects(S0)))
        finally:
            os.environ.pop("TASK_TRACE", None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.task._TRACKER.reset()
                objs = self.objects(S0)
                objs["fish_plate"] = SimpleNamespace(pos=np.array([np.nan, 0, 0]), quat_xyzw=np.zeros(4))
                self.assertFalse(self.task.success(objs))
                del objs["fish_plate"]
                self.assertFalse(self.task.success(objs))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertFalse(rows[0]["valid_state"])
        self.assertEqual(rows[1]["bad_dishes"], ["fish"])

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_lays_the_meal_and_each_arm_works_its_own_side(self):
        t = self.task
        final = dict(S0)
        for arm, names in t.PLAN.items():
            sign = 1.0 if arm == "left" else -1.0
            for n in names:
                self.assertGreater(sign * L.START[n][1], 0.0)  # each arm fetches only its own side's dishes
                px, py = t.place_point(n, arm)
                self.assertAlmostEqual(math.hypot(px - TARGET[n][0], py - TARGET[n][1]), 0.004)  # led 4 mm
                final[n] = dish(n, px, py, yaw=math.degrees(t.PLACE_YAW[n]) if n == "fish" else 0.0)
        self.assertEqual(score_of(final), 1.0)
        self.assertEqual(t.TARGET, TARGET)
        for idle in ("left", "right"):
            part = dict(final)
            for n in t.PLAN[idle]:
                part[n] = S0[n]
            self.assertEqual(score_of(part), 0.5)
        # The fish is gripped across the plate's width at its start yaw and laid at yaw -90 (1 degree trim).
        self.assertAlmostEqual(t.PICK_YAW["fish"], L.START["fish"][2])
        self.assertAlmostEqual(math.degrees(t.PLACE_YAW["fish"]), -91.0)
        self.assertAlmostEqual(-2 * t.GRIP["fish"], L.PLATE_W - 0.008)

    @staticmethod
    def spells(wps):
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_arms_take_turns_at_the_tray(self):
        """Only one gripper is over the tray (within 3 cm of its outline) at any time, and the demo fits the run
        limit with room."""
        plan = self.task.demo()

        def over_tray(w):
            x, y = w.pos[0], w.pos[1]
            return L.TRAY_X0 - 0.03 <= x <= L.TRAY_X1 + 0.03 and L.TRAY_Y0 - 0.03 <= y <= L.TRAY_Y1 + 0.03

        def busy(wps):  # intervals in which the arm is moving to, at or from a point over the tray
            out, prev = [], None
            for t0, t1, w in self.spells(wps):
                if over_tray(w) or (prev is not None and over_tray(prev)):
                    out.append((t0, t1))
                prev = w
            return out
        for a0, a1 in busy(plan["left"]):
            for b0, b1 in busy(plan["right"]):
                self.assertFalse(a0 < b1 - 1e-9 and b0 < a1 - 1e-9, (a0, a1, b0, b1))
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 2400 / 2)

    def test_carried_dishes_clear_everything(self):
        t = self.task
        tallest = max(TRAY_TOP_W + L.SOUP_H, BOARD_TOP_W + L.SOUP_H, TRAY_TOP_W + L.FISH_TOP)
        for n in L.DISHES:
            self.assertGreaterEqual(t.HOVER_Z - t.GRIP_UP[n] - tallest, 0.0119, n)
            self.assertGreaterEqual(t.GRIP_UP[n] - 0.013, 0.003, n)  # pad tips above the dish's underside
            self.assertLess(t.GRIP_UP[n] - 0.013, (L.PLATE_T if n == "fish" else L.BOWL[n][1]) - 0.008, n)
        for n in ("rice", "soup", "side"):
            self.assertAlmostEqual(-2 * t.GRIP[n], 2 * L.BOWL[n][0] - 0.008)
            self.assertGreater(L.FULL_GAP - 2 * L.BOWL[n][0], 0.025)  # open: 14 mm or more a side
        # Straight up after letting go: pad tips 10 mm or more over every dish's top (it counts as let go).
        for n in L.DISHES:
            top = TRAY_TOP_W + (L.FISH_TOP if n == "fish" else L.BOWL[n][1])
            self.assertGreater(t.UP_Z - 0.013 - top, L.RELEASE_ABOVE + 0.015)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        # The instruction names no quadrant, side or direction: that is what the task tests.
        for word in ("left", "right", "front", "back", "head", "belly", "-y", "+x"):
            self.assertNotIn(word, spec["instruction"].lower())

    def test_task_json_states_the_thresholds(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("10 mm from both centre lines", "5 mm", "3 mm of the tray top", "10 degrees", "10 mm above",
                  "65 mm", "52 mm", "3 mm in all over the last 1 s", "15 degrees", "1 mm", "45 degrees", "0.25",
                  "2.5 s", "x 33.0 cm, y 0"):
            self.assertIn(s, text)
        setup = spec["setup"]
        for s in ("x 22.0 to 44.0 cm, y -15.0 to +15.0 cm", "x 45.5 to 57.5 cm, y -10.0 to +10.0 cm",
                  "x 33.0 cm, y -30.0 cm", "x 21.5 cm, y -30.0 cm", "x 20.5 cm, y +30.0 cm", "x 33.0 cm, y +30.0 cm",
                  "66 mm across and 44 mm tall", "55 mm across and 34 mm tall", "64 mm across and 45 mm tall",
                  "130 by 60 mm and 12 mm thick", "80 g", "50 g", "70 g", "120 g"):
            self.assertIn(s, setup)
        self.assertEqual(spec["run_limit"], "2,400 steps (2 minutes at 20 Hz)")
        self.assertEqual(spec["scores"], "TBD after the GPU runs (offline kinematic check predicts demo: 1.0)")

    def test_success_hold_outlasts_every_release(self):
        plan = self.task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly set it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    self.assertLess(sum(a.duration for a in after), self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 4)

    def test_text_has_no_long_dashes(self):
        dashes = (chr(0x2013), chr(0x2014))  # en and em dash
        for name in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / name).read_text()
            for ch in dashes:
                self.assertNotIn(ch, text, name)


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the other side's dishes where they start."""

    def test_other_sides_dishes_are_out_of_reach(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        for arm, names in (("left", ("rice", "side")), ("right", ("soup", "fish"))):
            base = K.BASES["station"][arm]
            for n in names:
                x, y, _ = L.START[n]
                for yaw in np.linspace(-math.pi, math.pi, 13):
                    self.assertFalse(K.reachable((x, y, BOARD_TOP_W + 0.020), float(yaw), base)[0], (arm, n, yaw))
        for arm, names in (("left", ("soup", "fish")), ("right", ("rice", "side"))):
            base = K.BASES["station"][arm]
            for n in names:
                x, y, _ = L.START[n]
                self.assertTrue(K.reachable((x, y, BOARD_TOP_W + 0.020), 0.0, base)[0], (arm, n))


if __name__ == "__main__":
    unittest.main()
