"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import math
import os
import pathlib
import random
import re
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
    Tracker, bottom_gap, evaluate, flat_on_bed, held, highest, lowest, nearest_spot, parts, reach, rotation,
    tilt_deg, valid,
)

MM = 0.001
IDX = {n: i for i, n in enumerate(L.NAMES)}


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A disc's state: its world bounding box worked out from the turned cylinder."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = np.array([L.PUCK_H / 2 * abs(a[k]) + L.PUCK_R * math.sqrt(max(0.0, 1 - a[k] ** 2)) for k in range(3)])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def disc(x, y, bottom=L.BED_TOP, tilt=0.0, axis="x", **kw):
    """A disc centred at (x, y) with its lowest point at ``bottom``, tilted ``tilt`` degrees about ``axis``."""
    q = about(axis, tilt)
    az = abs(rotation(q)[2, 2])
    half = L.PUCK_H / 2 * az + L.PUCK_R * math.sqrt(max(0.0, 1 - az * az))
    return state((x, y, bottom + half), q, **kw)


def start():
    return [disc(*L.START[n], bottom=L.start_bottom(n)) for n in L.NAMES]


S0 = start()


def table(**over):
    """The start, with some discs replaced: name -> (x, y) on the bed, or a ready-made state."""
    out = list(S0)
    for name, v in over.items():
        out[IDX[name]] = v if isinstance(v, SimpleNamespace) else disc(*v)
    return out


SPOT = L.SPOTS
DONE = table(black=SPOT["blue"], yellow=SPOT["yellow"])


def score_of(ps):
    return evaluate(ps)[0]


def run(frames, hz=20.0, arms=None):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


def glide(a, b, n):
    """n frames moving every disc of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(final, n=20, hold=30):
    return [S0] + glide(S0, final, n) + [final] * hold


def tracked(final, **kw):
    return run(build(final), **kw)[0]


ON = 0.030 / L.FULL_GAP  # rigstate's opening with the fingers closed on a 30 mm disc


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    """A rigstate.arms() snapshot: each given arm's pad-tip point (x, y, z) and opening (0 closed .. 1 open)."""
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


PARKED = snap(left=(0.200, 0.250, L.TOP + 0.137), right=(0.200, -0.250, L.TOP + 0.137))


class Layout(unittest.TestCase):
    def test_spots_are_the_wpbsa_table_at_one_sixth(self):
        self.assertAlmostEqual(L.PLAY_L * 6, 3.569)
        self.assertAlmostEqual(L.PLAY_W * 6, 1.778)
        self.assertAlmostEqual((L.BAULK_Y - L.BED_Y0) * 6, 0.737)
        self.assertAlmostEqual(L.D_R * 6, 0.292)
        self.assertAlmostEqual((L.BED_Y1 - SPOT["black"][1]) * 6, 0.324)
        self.assertAlmostEqual(SPOT["pink"][1] - SPOT["blue"][1], L.BED_Y1 - SPOT["pink"][1])
        self.assertEqual(SPOT["blue"], (L.CX, L.CY))
        # Viewed from the baulk end (-y), looking toward +y, the right hand is +x: the yellow is the +x corner.
        facing, up = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        right = np.cross(facing, up)
        self.assertGreater(float(right @ np.array([SPOT["yellow"][0] - L.CX, 0.0, 0.0])), 0.0)
        self.assertLess(SPOT["green"][0], L.CX)
        self.assertEqual(sorted(L.VALUE.values()), [2, 3, 4, 5, 6, 7])

    def test_the_rule_gives_the_blue_spot_for_the_black(self):
        """Rule 7(e) walked on the start layout: the black's own spot and the pink are occupied, the blue is the
        highest free spot; the yellow's own spot is free."""
        def occupied(spot):
            return any(math.hypot(L.START[n][0] - SPOT[spot][0], L.START[n][1] - SPOT[spot][1]) < L.TOUCH
                       for n in L.TABLE)
        self.assertEqual({s for s in SPOT if occupied(s)}, {"black", "pink", "green"})
        free = [s for s in sorted(SPOT, key=lambda k: -L.VALUE[k]) if not occupied(s)]
        self.assertEqual(free[0], L.BLACK_FULL)
        self.assertFalse(occupied("yellow"))

    def test_spacing(self):
        """Spots at least 48.7 mm apart (each placement maps to one spot); free spots well clear of every disc;
        table discs well apart; the colours' trays only at their own ends."""
        names = list(SPOT)
        gaps = [math.dist(SPOT[a], SPOT[b]) for i, a in enumerate(names) for b in names[i + 1:]]
        self.assertAlmostEqual(min(gaps), L.D_R)
        self.assertGreater(min(gaps), 2 * L.SPOT_TOL + 0.030)
        for spot, want in (("blue", 0.086), ("yellow", 0.0775), ("brown", 0.0486)):
            near = min(math.dist(SPOT[spot], L.START[n]) for n in L.TABLE)
            self.assertGreater(near, want, spot)
        tbl = [L.START[n] for n in L.TABLE]
        self.assertGreater(min(math.dist(a, b) for i, a in enumerate(tbl) for b in tbl[i + 1:]), 0.077)
        for n in L.NAMES:  # every disc wholly inside the bed or on its tray; nothing where the arms' start pose hides
            x, y = L.START[n]
            if n in L.TABLE:
                self.assertTrue(L.BED_X0 + L.PUCK_R < x < L.BED_X1 - L.PUCK_R and
                                L.BED_Y0 + L.PUCK_R < y < L.BED_Y1 - L.PUCK_R, n)
            self.assertLess(abs(y) + L.PUCK_R, 0.45)
            self.assertGreater(x - L.PUCK_R, 0.20)

    def test_d_strips_follow_the_arc(self):
        strips = L.d_strips()
        self.assertEqual(len(strips), 12)
        for x, y, length, _ in strips:
            r = math.hypot(x - L.CX, y - L.BAULK_Y)
            self.assertLess(abs(r - L.D_R), 0.0005)
            self.assertLess(y, L.BAULK_Y)  # on the baulk side of the line


class Geometry(unittest.TestCase):
    def test_lowest_highest_tilt(self):
        s = disc(0.3, 0.0)
        self.assertAlmostEqual(lowest(s), L.BED_TOP)
        self.assertAlmostEqual(highest(s), L.BED_TOP + L.PUCK_H)
        self.assertAlmostEqual(tilt_deg(disc(0.3, 0, tilt=180).quat_xyzw), 0.0)  # either face up
        self.assertAlmostEqual(tilt_deg(disc(0.3, 0, tilt=7, axis="y").quat_xyzw), 7.0)
        on_edge = disc(0.3, 0.0, tilt=90.0)
        self.assertAlmostEqual(lowest(on_edge), L.BED_TOP)
        self.assertAlmostEqual(on_edge.pos[2] - L.BED_TOP, L.PUCK_R)

    def test_flat_on_bed(self):
        self.assertTrue(flat_on_bed(disc(0.3, 0.0)))
        self.assertTrue(flat_on_bed(disc(0.3, 0.0, tilt=9.9)))
        self.assertFalse(flat_on_bed(disc(0.3, 0.0, tilt=10.0)))
        self.assertTrue(flat_on_bed(disc(0.3, 0.0, bottom=L.BED_TOP + 3 * MM)))
        self.assertFalse(flat_on_bed(disc(0.3, 0.0, bottom=L.BED_TOP + 3.5 * MM)))
        self.assertFalse(flat_on_bed(disc(0.3, 0.0, bottom=L.BED_TOP + L.PUCK_H)))  # stacked
        self.assertFalse(flat_on_bed(disc(0.3, 0.0, bottom=L.TOP)))  # fell under the bed's level
        self.assertFalse(flat_on_bed(disc(L.BED_X1 + MM, 0.0)))
        self.assertFalse(flat_on_bed(S0[IDX["black"]]))  # on its tray

    def test_nearest_spot(self):
        self.assertEqual(nearest_spot(disc(*SPOT["blue"]))[0], "blue")
        self.assertEqual(nearest_spot(disc(SPOT["yellow"][0] + 0.005, SPOT["yellow"][1]))[0], "yellow")

    def test_valid(self):
        self.assertTrue(valid(disc(0.3, 0.0)))
        for bad in (state((np.nan, 0, 0.77)), state((0.3, 0, 0.77), vel=(np.inf, 0, 0)),
                    state((0.3, 0, 0.77), quat=(0, 0, 0, 0)), SimpleNamespace(pos=(0.3, 0, 0.77))):
            self.assertFalse(valid(bad))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        self.assertEqual(score_of(S0), 0.0)
        r, _ = run([S0] * 30)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_both_right_scores_one(self):
        self.assertEqual(score_of(DONE), 1.0)
        r = tracked(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_black_on_the_brown_spot_and_yellow_right(self):
        self.assertEqual(tracked(table(black=SPOT["brown"], yellow=SPOT["yellow"])).score, 0.75)

    def test_yellow_right_black_elsewhere(self):
        bx, by = SPOT["black"]
        for black in (L.START["black"], (bx, by + 0.031 + MM), (0.25, 0.20), (0.45, -0.05)):
            ps = table(yellow=SPOT["yellow"])
            if black != L.START["black"]:
                ps[IDX["black"]] = disc(*black)
            self.assertEqual(tracked(ps).score, 0.5, black)

    def test_black_right_yellow_elsewhere(self):
        gx, gy = SPOT["green"]
        for yellow in (SPOT["brown"], (gx + 0.025, gy), (0.30, 0.20)):
            self.assertEqual(tracked(table(black=SPOT["blue"], yellow=yellow)).score, 0.5, yellow)
        self.assertEqual(tracked(table(black=SPOT["blue"])).score, 0.5)  # the yellow left on its tray

    def test_black_on_another_free_spot_only(self):
        self.assertEqual(tracked(table(black=SPOT["brown"])).score, 0.25)
        self.assertEqual(tracked(table(black=SPOT["yellow"])).score, 0.25)
        # The black on the yellow spot: the yellow can then count nowhere (touching it, or on another spot).
        self.assertEqual(tracked(table(black=SPOT["yellow"], yellow=SPOT["blue"])).score, 0.25)
        yx, yy = SPOT["yellow"]
        self.assertEqual(tracked(table(black=SPOT["yellow"], yellow=(yx + 0.025, yy))).score, 0.0)

    def test_spot_window_edges(self):
        sx, sy = SPOT["blue"]
        for d, want in ((0.0, 1.0), (7.9 * MM, 1.0), (8.0 * MM, 1.0), (8.2 * MM, 0.5)):
            for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                ps = table(black=(sx + d * math.cos(ang), sy + d * math.sin(ang)), yellow=SPOT["yellow"])
                self.assertEqual(score_of(ps), want, (d, ang))
        yx, yy = SPOT["yellow"]
        self.assertEqual(score_of(table(black=SPOT["blue"], yellow=(yx, yy + 8.2 * MM))), 0.5)

    def test_parts_table(self):
        self.assertEqual(parts("blue", "yellow"), (0.5, 0.5))
        for s in ("brown", "yellow", "green", "pink", "black"):
            self.assertEqual(parts(s, None), (0.25, 0.0))
        self.assertEqual(parts(None, "blue"), (0.0, 0.0))
        self.assertEqual(parts(None, "green"), (0.0, 0.0))

    def test_occupied_spots_can_never_count(self):
        """A colour within 8 mm of the green, pink or black spot is within 16 mm of the ball there: touching."""
        for spot in ("green", "pink", "black"):
            sx, sy = SPOT[spot]
            for ang in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                p = (sx + 0.008 * math.cos(ang), sy + 0.008 * math.sin(ang))
                d = evaluate(table(black=p))[1]
                self.assertIsNone(d["black"]["on_spot"], (spot, ang))

    def test_every_outcome_listed_in_task_json(self):
        gx, gy = SPOT["green"]
        bx, by = SPOT["black"]
        cases = [
            (table(black=SPOT["blue"], yellow=SPOT["yellow"]), 1.0),
            (table(black=SPOT["brown"], yellow=SPOT["yellow"]), 0.75),
            (table(yellow=SPOT["yellow"]), 0.5),
            (table(black=(bx, by + 0.032), yellow=SPOT["yellow"]), 0.5),
            (table(black=(0.25, -0.02), yellow=SPOT["yellow"]), 0.5),
            (table(black=SPOT["blue"], yellow=SPOT["brown"]), 0.5),
            (table(black=SPOT["blue"], yellow=(gx + 0.030, gy)), 0.5),
            (table(black=SPOT["blue"]), 0.5),
            (table(black=SPOT["brown"]), 0.25),
            (table(black=SPOT["yellow"], yellow=SPOT["blue"]), 0.25),
            (S0, 0.0),
        ]
        for ps, want in cases:
            self.assertEqual(tracked(ps).score, want)

    def test_every_score_turns_up_and_nothing_else(self):
        seen = set()
        rng = random.Random(7)
        targets = [*SPOT.values(), L.START["black"], L.START["yellow"], (0.25, 0.0), (0.45, -0.25)]
        for _ in range(1500):
            b = rng.choice(targets)
            y = rng.choice(targets)
            jb = (b[0] + rng.uniform(-0.007, 0.007), b[1] + rng.uniform(-0.007, 0.007))
            jy = (y[0] + rng.uniform(-0.007, 0.007), y[1] + rng.uniform(-0.007, 0.007))
            ps = list(S0)
            on_tray = {L.START["black"], L.START["yellow"]}
            ps[0] = disc(*jb, bottom=L.TRAY_TOP if b in on_tray else L.BED_TOP)
            ps[1] = disc(*jy, bottom=L.TRAY_TOP if y in on_tray else L.BED_TOP)
            s = score_of(ps)
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_score_is_the_state_at_hand(self):
        frames = build(DONE) + glide(DONE, table(yellow=SPOT["yellow"]), 10) + [table(yellow=SPOT["yellow"])] * 25
        r, _ = run(frames)
        self.assertEqual(r.score, 0.5)  # the black taken back off its spot (onto its tray): 0.5 at the end


class Stillness(unittest.TestCase):
    def test_a_colour_counts_only_after_one_second_still(self):
        frames = build(DONE, n=10, hold=0)
        t = Tracker()
        for f in frames:
            t.update(f)
        scores = []
        for _ in range(25):
            scores.append(t.update(DONE).score)
        self.assertEqual(scores.index(1.0), 19)  # 20 still steps -> the last 1 s windows clear

    def test_creeping_colour_is_not_still(self):
        t = Tracker()
        sx, sy = SPOT["blue"]
        r = None
        for k in range(60):  # 2 mm per 0.5 s: too much for one window
            r = t.update(table(black=(sx + k * 0.0002 - 0.006, sy), yellow=SPOT["yellow"]))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["black"]["still"])

    def test_jitter_under_the_limit_is_still(self):
        t = Tracker()
        sx, sy = SPOT["blue"]
        r = None
        for k in range(60):  # 0.05 mm back and forth each step: 1.0 mm per 0.5 s window
            r = t.update(table(black=(sx + (0.00005 if k % 2 else 0.0), sy), yellow=SPOT["yellow"]))
        self.assertEqual(r.score, 1.0)

    def test_windows_hold_the_right_steps(self):
        t = Tracker()
        self.assertEqual(t.window, 10)
        self.assertEqual(t.history.maxlen, 21)


class ZeroConditions(unittest.TestCase):
    def test_a_table_disc_moved_over_8_mm_latches_zero(self):
        for name in L.TABLE:
            x, y = L.START[name]
            for dx, dy in ((0.0081, 0.0), (0.0, -0.0081), (0.006, 0.006)):
                moved = table(black=SPOT["blue"], yellow=SPOT["yellow"])
                moved[IDX[name]] = disc(x + dx, y + dy, bottom=L.start_bottom(name))
                self.assertEqual(tracked(moved).score, 0.0, (name, dx, dy))

    def test_a_table_disc_moved_under_8_mm_is_fine(self):
        for name in L.TABLE:
            x, y = L.START[name]
            moved = list(DONE)
            moved[IDX[name]] = disc(x + 0.0075, y, bottom=L.start_bottom(name))
            r = tracked(moved)
            self.assertEqual(r.score, 1.0, name)

    def test_the_latch_holds_after_it_is_put_back(self):
        x, y = L.START["red"]
        knocked = table(red=(x + 0.012, y))
        frames = [S0, knocked, knocked] + build(DONE)[1:]
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["latched"])
        self.assertEqual(t.latched_by, "red")
        self.assertEqual(t.latched_at, 2)

    def test_lifted_and_put_back_latches(self):
        lifted = list(S0)
        x, y = L.START["blue"]
        lifted[IDX["blue"]] = disc(x, y, bottom=L.BED_TOP + 0.009)
        r, _ = run([S0, lifted] + build(DONE)[1:])
        self.assertEqual(r.score, 0.0)

    def test_score_alone_sees_a_disturbance(self):
        """score() without update(): the state at hand is still checked against the starting marks."""
        t = Tracker()
        moved = table(black=SPOT["blue"], yellow=SPOT["yellow"], brown=(0.43, -0.10))
        self.assertEqual(t.result(moved).score, 0.0)

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        frames = build(DONE)
        t = Tracker()
        for f in frames:
            t.update(f)
        self.assertEqual(t.last.score, 1.0)
        for bad in (state((np.nan, 0.0, 0.77)), state((0.3, 0.0, 0.77), quat=(0, 0, 0, 0)),
                    state((0.3, 0.0, 0.77), vel=(0, np.inf, 0)), SimpleNamespace(pos=(0.3, 0.0, 0.77))):
            ps = list(DONE)
            ps[IDX["green"]] = bad
            r = t.update(ps)
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(t.bad_steps, 4)
        self.assertFalse(t.latched)
        self.assertEqual(t.update(DONE).score, 1.0)  # the tracker kept its history
        self.assertEqual(t.update(DONE[:-1]).score, 0.0)  # a missing disc: no score, no crash

    def test_colour_off_the_bed_or_tilted_or_stacked(self):
        sx, sy = SPOT["blue"]
        for black in (disc(sx, sy, tilt=12.0), disc(sx, sy, bottom=L.BED_TOP + 0.005),
                      disc(sx, sy, bottom=L.BED_TOP + L.PUCK_H)):
            self.assertEqual(tracked(table(black=black, yellow=SPOT["yellow"])).score, 0.5)


class Release(unittest.TestCase):
    def test_reach_numbers(self):
        self.assertAlmostEqual(reach(SimpleNamespace(opening=1.0)), 0.0804, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=ON)), 0.0520, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=0.0)), 0.0424, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=float("nan"))), 0.0804, places=4)

    def test_parked_arms_release_both(self):
        self.assertEqual(evaluate(DONE, PARKED)[0], 1.0)
        self.assertEqual(tracked(DONE, arms=PARKED).score, 1.0)

    def test_a_colour_still_gripped_does_not_count(self):
        bx, by = SPOT["blue"]
        grip = snap(left=(bx, by, L.BED_TOP + 0.004), right=PARKED["right"].tips, left_open=ON)
        r = tracked(DONE, arms=grip)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)
        self.assertTrue(r.details["black"]["held"])

    def test_open_fingers_round_it_still_hold_it(self):
        yx, yy = SPOT["yellow"]
        arms = snap(left=PARKED["left"].tips, right=(yx + 0.03, yy, L.BED_TOP + 0.005), right_open=0.6)
        self.assertEqual(evaluate(DONE, arms)[0], 0.5)

    def test_pad_tips_10_mm_over_the_top_have_let_go(self):
        bx, by = SPOT["blue"]
        top = L.BED_TOP + L.PUCK_H
        self.assertEqual(evaluate(DONE, snap(left=(bx, by, top + 0.0101), right=PARKED["right"].tips))[0], 1.0)
        self.assertEqual(evaluate(DONE, snap(left=(bx, by, top + 0.0099), right=PARKED["right"].tips))[0], 0.5)
        far = snap(left=(bx + 0.081, by, top - 0.01), right=PARKED["right"].tips)
        self.assertEqual(evaluate(DONE, far)[0], 1.0)  # beside it, out of reach even fully open

    def test_unreadable_arms_are_skipped_and_counted(self):
        r, t = run(build(DONE), arms=None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(build(DONE)))
        bx, by = SPOT["blue"]
        only_left = snap(left=(bx, by, L.BED_TOP + 0.004), left_open=ON)  # the right arm unreadable
        r, t = run(build(DONE), arms=only_left)
        self.assertEqual(r.score, 0.5)  # the readable arm is still checked
        self.assertEqual(t.arms_unavailable_by_side, {"left": 0, "right": len(build(DONE))})

    def test_success_hold_outlasts_settle_and_release(self):
        sys.path.insert(0, str(_stub_dir()))
        task = importlib.import_module("task")
        plan = task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly set it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    # counted (if the arms were unreadable) 1 s after the set-down; open 2 s after it
                    self.assertLess(sum(a.duration for a in after), 1.0 + task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 2)


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        x, y = L.START["red"]
        frames = [S0, table(red=(x + 0.02, y))] + build(DONE)[1:] + [S0]
        r, t = run(frames)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.latched)  # the latch is cleared with the new episode
        self.assertEqual(t.steps, 1)
        for f in build(DONE)[1:]:
            r = t.update(f)
        self.assertEqual(r.score, 1.0)

    def test_gradual_return_is_not_a_reset(self):
        frames = build(DONE) + glide(DONE, S0, 20)
        _, t = run(frames)
        self.assertGreater(t.steps, 40)

    def test_jump_elsewhere_is_not_a_reset(self):
        frames = build(DONE) + [table(black=(0.25, 0.25), yellow=(0.25, -0.25))]
        _, t = run(frames)
        self.assertGreater(t.steps, 40)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        sys.path.insert(0, str(_stub_dir()))
        task = importlib.import_module("task")
        task._TRACKER.reset()
        task.success(dict(zip(task.PUCK_NAMES, S0, strict=True)))
        task.success(dict(zip(task.PUCK_NAMES, table(red=(0.33, 0.07)), strict=True)))
        self.assertTrue(task._TRACKER.latched)
        rigstate.offline_reset()
        self.assertFalse(task._TRACKER.latched)
        self.assertEqual(task._TRACKER.steps, 0)


class Exploits(unittest.TestCase):
    def test_black_just_in_front_of_its_own_occupied_spot(self):
        """Spec shortcut 1: the black toward the top cushion from its own (occupied) spot, the yellow right: 0.5."""
        bx, by = SPOT["black"]
        for gap in (0.0315, 0.040, 0.0465):
            self.assertEqual(tracked(table(black=(bx, by + gap), yellow=SPOT["yellow"])).score, 0.5, gap)

    def test_black_on_brown_yellow_on_tray_and_yellow_on_the_wrong_corner(self):
        """Spec shortcut 2: the black on the brown spot, the yellow on its tray: 0.25. The yellow forced onto the
        D's left corner pushes the green off its spot: latched, 0."""
        self.assertEqual(tracked(table(black=SPOT["brown"])).score, 0.25)
        gx, gy = SPOT["green"]
        pushed = table(black=SPOT["brown"], yellow=SPOT["green"], green=(gx - 0.031, gy))
        self.assertEqual(tracked(pushed).score, 0.0)

    def test_pushing_the_red_off_the_black_spot(self):
        """Spec shortcut 4: the red pushed off the black spot and the black put there: latched, 0."""
        bx, by = SPOT["black"]
        ps = table(black=SPOT["black"], yellow=SPOT["yellow"], red_on_black=(bx + 0.035, by))
        self.assertEqual(tracked(ps).score, 0.0)
        ps = table(black=SPOT["blue"], yellow=SPOT["yellow"], red_on_pink=(L.CX + 0.04, SPOT["pink"][1]))
        self.assertEqual(tracked(ps).score, 0.0)  # freeing the pink spot costs everything as well

    def test_yellow_still_in_the_gripper(self):
        """Spec shortcut 5: both right but the yellow still gripped: 0.5; both right and released: 1.0."""
        yx, yy = SPOT["yellow"]
        grip = snap(left=PARKED["left"].tips, right=(yx, yy, L.BED_TOP + 0.004), right_open=ON)
        self.assertEqual(tracked(DONE, arms=grip).score, 0.5)
        self.assertEqual(tracked(DONE, arms=PARKED).score, 1.0)

    def test_stacked_on_the_red_on_its_own_spot(self):
        bx, by = SPOT["black"]
        ps = table(black=disc(bx, by, bottom=L.MARK_TOP + L.PUCK_H), yellow=SPOT["yellow"])
        self.assertEqual(tracked(ps).score, 0.5)

    def test_colours_swapped(self):
        self.assertEqual(tracked(table(black=SPOT["yellow"], yellow=SPOT["blue"])).score, 0.25)

    def test_both_colours_pressed_against_neighbours(self):
        """The yellow touching the brown, or the black touching a red: neither counts (nor is the table moved)."""
        bx, by = L.START["brown"]
        self.assertEqual(tracked(table(black=SPOT["blue"], yellow=(bx, by - 0.030))).score, 0.5)

    def test_disc_on_its_edge_on_the_spot(self):
        sx, sy = SPOT["blue"]
        self.assertEqual(tracked(table(black=disc(sx, sy, tilt=90.0), yellow=SPOT["yellow"])).score, 0.5)


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

    def objects(self, pucks):
        return dict(zip(self.task.PUCK_NAMES, pucks, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.PUCK_NAMES, L.NAMES)
        for spec, s in zip(t.PUCKS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((spec.radius, spec.height, spec.mass), (L.PUCK_R, L.PUCK_H, L.PUCK_MASS))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 1 + 4 + 1 + 12 + 6 + 2 + 9)
        self.assertEqual({o.color for o in t.MARKS}, {L.WHITE})  # every marking alike, the spots unlabelled
        self.assertEqual({(o.radius, o.height) for o in t.MARKS if o.name.startswith("spot_")},
                         {(L.SPOT_R, L.MARK_T)})
        for o in t.MARKS:
            self.assertAlmostEqual(o.pos[2] + L.MARK_T / 2, L.MARK_TOP)
        self.assertAlmostEqual(t.BED.pos[2] + t.BED.size[2] / 2, L.BED_TOP)
        for r in t.RAILS:
            self.assertAlmostEqual(r.pos[2] + r.size[2] / 2, L.BED_TOP + 0.010)
        self.assertEqual(len({o.color for o in t.PUCKS if o.name.startswith("red")}), 1)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                ok = False
                for f in build(DONE)[1:]:
                    ok = self.task.success(self.objects(f))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), len(build(DONE)))
        last = rows[-1]
        for key in ("step", "score", "success", "black_part", "yellow_part", "latched", "latched_at", "latched_by",
                    "disturbed_now", "max_disp_mm", "black", "yellow", "pos_mm", "path_mm", "first_step",
                    "bad_steps", "arms_unavailable", "rigstate_error"):
            self.assertIn(key, last)
        self.assertEqual(last["black"]["on_spot"], "blue")
        self.assertEqual(last["yellow"]["on_spot"], "yellow")
        self.assertEqual(last["path_mm"]["black"], [0.0, 0.0])
        self.assertIsNone(rows[0]["path_mm"])
        self.assertEqual(last["first_step"]["1.0"], 1 + 20 + 20)  # in place at step 21, still for 1 s by step 41

    def test_hooks_read_the_arms(self):
        bx, by = SPOT["blue"]
        rigstate.offline_reset()
        self.task._TRACKER.reset()

        def feed(tips, op):
            rigstate.offline_feed({"left": {"tcp": np.add(tips, (0, 0, 0.013)), "tips": tips, "opening": op},
                                   "right": {"tcp": PARKED["right"].tcp, "tips": PARKED["right"].tips,
                                             "opening": 1.0}})
        try:
            for f in build(DONE):
                feed((bx, by, L.BED_TOP + 0.004), ON)
                self.assertFalse(self.task.success(self.objects(f)))
            self.assertEqual(self.task.score(self.objects(DONE)), 0.5)
            feed(tuple(PARKED["left"].tips), 1.0)
            self.assertTrue(self.task.success(self.objects(DONE)))
        finally:
            rigstate.offline_reset()

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
                self.assertFalse(self.task.success(self.objects([state((np.nan, 0, 0.77)), *S0[1:]])))
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                row = json.loads(f.readline())
        self.assertFalse(row["valid_state"])

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_finds_the_spots_by_the_rule(self):
        t = self.task
        self.assertEqual(t.free_spot("black"), "blue")
        self.assertEqual(t.free_spot("yellow"), "yellow")
        # Move the reds off the black and pink spots: the black would go home; block the yellow: the black's walk.
        c = {n: L.START[n] for n in L.TABLE}
        self.assertEqual(t.free_spot("black", {**c, "red_on_black": (0.25, 0.25)}), "black")
        self.assertEqual(t.free_spot("black", {**c, "red_on_pink": (0.25, 0.25)}), "pink")
        self.assertEqual(t.free_spot("yellow", {**c, "brown": SPOT["yellow"]}), "blue")
        # Each place point is within 8 mm (4 mm) of its spot, led along the carry.
        for colour, spot in (("black", "blue"), ("yellow", "yellow")):
            self.assertAlmostEqual(math.dist(t.place_point(colour), SPOT[spot]), t.LEAD)

    def test_demo_places_every_colour_and_each_arm_its_own(self):
        plan = self.task.demo()
        for arm, colour in (("left", "black"), ("right", "yellow")):
            wps = plan[arm]
            grip = [w for w in wps if w.label == f"grip the {colour}"][0]
            self.assertAlmostEqual(math.dist(grip.pos[:2], L.START[colour]), 0.0)
            let_go = [w for w in wps if w.label == "let go of it"][0]
            self.assertAlmostEqual(math.dist(let_go.pos[:2], self.task.place_point(colour)), 0.0)
            self.assertAlmostEqual(let_go.pos[2] - self.task.GRIP_UP - L.BED_TOP, self.task.DROP)
            for w in wps:
                self.assertGreaterEqual(w.duration, 0.3)
                if w.finger != wps[wps.index(w) - 1].finger:
                    self.assertGreaterEqual(w.duration, 1.5)  # every open and close at least 1.5 s

    def test_arms_take_turns_over_the_table(self):
        """The left arm starts its carry only after the right arm has let go of the yellow and risen."""
        plan = self.task.demo()

        def time_of(wps, label):
            t = 0.0
            for w in wps:
                t += w.duration
                if w.label == label:
                    return t
            raise KeyError(label)
        risen = time_of(plan["right"], "straight up, clear of it")
        carry_start = time_of(plan["left"], "carry it over its spot") - 2.5
        self.assertGreaterEqual(carry_start, risen - 1.0)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))

    def test_instruction_withholds_the_answer(self):
        text = self.task.INSTRUCTION.lower()
        for word in ("blue", "highest", "occupied", "corner", "right-hand", "value", "spot is"):
            self.assertNotIn(word, text)

    def test_task_json_states_the_thresholds(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("8 mm", "31 mm", "10 degrees", "3 mm", "1.5 mm", "0.5 s", "10 mm above", "80 mm", "52 mm",
                  "2 s", "0.25", "0.75", "Rule 7(e)", "Rule 18"):
            self.assertIn(s, text)
        setup = spec["setup"]
        for s in ("30 mm across and 20 mm tall, 25 g", "(24.0, +37.0)", "(24.0, -37.0)", "(35.0, +24.34)",
                  "(35.0, +14.87)", "(30.0, +7.0)", "(40.0, +9.0)", "(30.0, -7.0)", "(30.13, -17.46)",
                  "(42.0, -10.0)", "(39.87, -17.46)", "y -17.46", "48.7 mm", "296.3 mm", "594.8 mm",
                  "60 x 60 x 10 mm"):
            self.assertIn(s, setup)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(spec["run_limit"], "1,200 steps (1 minute at 20 Hz)")
        for k in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes"):
            self.assertIsNone(re.search("[" + chr(0x2013) + chr(0x2014) + "]", spec[k]), k)


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the other end's tray; each reaches its own tray
    and its colour's spot."""

    def test_trays_out_of_reach_of_the_other_arm(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        z = L.TRAY_TOP + 0.017
        for arm, colour in (("left", "yellow"), ("right", "black")):
            base = K.BASES["station"][arm]
            x, y = L.START[colour]
            for yaw in np.linspace(-math.pi, math.pi, 13):
                self.assertFalse(K.reachable((x, y, z), float(yaw), base)[0], (arm, colour, yaw))
        for arm, colour, spot in (("left", "black", "blue"), ("right", "yellow", "yellow")):
            base = K.BASES["station"][arm]
            self.assertTrue(K.reachable((*L.START[colour], z), 0.0, base)[0])
            self.assertTrue(K.reachable((*SPOT[spot], L.BED_TOP + 0.018), 0.0, base)[0])


if __name__ == "__main__":
    unittest.main()
