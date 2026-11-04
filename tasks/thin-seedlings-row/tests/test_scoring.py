"""Offline tests for scoring.py and the hooks in task.py: synthetic seedling states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import itertools
import json
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, moved, on_bed, one_wrong_choice, place, rotation, spacing_ok, tilt_deg, tray_of, valid,
)

HZ = 20.0
DT = 1.0 / HZ
R = L.SEEDLING_RADIUS
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
N = L.NAMES
S = {i + 1: n for i, n in enumerate(N)}  # S[1] == "seedling_01" ...


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def cyl(name, center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like upright-or-not cylinder: centre, orientation, velocities, world bounding box."""
    c, q = np.array(center, float), np.array(quat, float)
    a = rotation(q)[:, 2]
    h = L.HEIGHT[name]
    ext = R * np.sqrt(np.clip(1 - a * a, 0, 1)) + h / 2 * np.abs(a)
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c - ext, high=c + ext)


def start(name):
    return np.array(L.start_pos(name))


def in_tray(name, side, k=0, lying=False):
    """Where a seedling rests in a tray: standing on its floor at one of three spots, or lying."""
    tx, ty = L.TRAYS[side]
    dx, dy = ((0.0, 0.025), (0.025, -0.025), (-0.02, -0.025))[k % 3]
    if lying:
        return np.array([tx + dx, ty, L.TOP + R]), about("x", 90)
    return np.array([tx + dx, ty + dy, L.TOP + L.HEIGHT[name] / 2]), Q0


def side_of(name):
    return "left" if L.START_Y[name] > 0 else "right"


class Film:
    """A trial as a list of frames {name: (centre, quat)}; ``go`` moves the named seedlings linearly over n control
    steps (a quaternion switches halfway); velocities come from the frame-to-frame motion."""

    def __init__(self):
        self.cur = {n: (start(n), Q0) for n in N}
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        for k, v in to.items():
            b[k] = (np.array(v[0], float), np.array(v[1], float)) if isinstance(v, tuple) else (np.array(v, float), a[k][1])
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in N:
                (p0, q0), (p1, q1) = a[k], b[k]
                fr[k] = ((1 - f) * p0 + f * p1, q1 if f >= 0.5 else q0)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def pull(self, name, to, quat=Q0, carry=0.15):
        """Lift a seedling straight up, carry it over ``to`` and set it down there."""
        p = self.cur[name][0]
        top = L.TOP + carry
        self.go(20, **{name: [p[0], p[1], top]})
        self.go(30, **{name: [to[0], to[1], top]})
        self.go(4, **{name: (to, quat)})
        return self

    def drop(self, name, side, k, lying=False):
        p, q = in_tray(name, side, k, lying)
        return self.pull(name, p, q)


def states(frame, prev=None):
    out = {}
    for n in N:
        p, q = frame[n]
        v = (p - prev[n][0]) / DT if prev is not None else np.zeros(3)
        out[n] = cyl(n, p, q, v)
    return out


def play(film, tracker=None, until=None):
    t = tracker or Tracker(HZ)
    r, prev = None, None
    for fr in film.frames[:until] if until else film.frames:
        r = t.update(states(fr, prev))
        prev = fr
    return r, t


ORDER_L = (S[5], S[3], S[1])
ORDER_R = (S[9], S[8], S[7])


def thin(film=None, left=True, right=True, settle=30, lying=False):
    """The intended trial: each arm pulls its three seedlings into its tray."""
    f = film or Film().hold(5)
    for k in range(3):
        if left:
            f.drop(ORDER_L[k], "left", k, lying)
        if right:
            f.drop(ORDER_R[k], "right", k, lying)
    return f.hold(settle)


class Rule(unittest.TestCase):
    def test_keep_set_is_the_greedy_rule(self):
        self.assertEqual(L.KEEP, {S[2], S[4], S[6], S[10]})
        self.assertEqual(L.PULL, {S[1], S[3], S[5], S[7], S[8], S[9]})
        self.assertEqual(len(set(L.HEIGHT.values())), 10)  # no ties: 'from the tallest down' is unambiguous

    def test_keep_set_is_the_only_spaced_set_where_each_pulled_one_has_a_taller_keeper_nearby(self):
        def ok(keep):
            return spacing_ok(keep) and all(
                any(L.HEIGHT[k] > L.HEIGHT[p] and abs(L.START_Y[k] - L.START_Y[p]) < L.SPACING for k in keep)
                for p in N if p not in keep)
        sets = [frozenset(c) for r in range(1, 11) for c in itertools.combinations(N, r) if ok(frozenset(c))]
        self.assertEqual(sets, [L.KEEP])

    def test_one_wrong_choice_cases(self):
        swaps = sorted(one_wrong_choice(c) for c in map(frozenset, itertools.combinations(N, 4))
                       if one_wrong_choice(c))
        self.assertEqual(swaps, sorted([(S[2], S[1]), (S[6], S[7]), (S[10], S[8]), (S[10], S[9])]))
        # on this row every spaced set that differs from the keep set by one seedling is such a case: the stand-in
        # is always a shorter seedling within 10 cm of the keeper it replaces
        one_off = [c for c in map(frozenset, itertools.combinations(N, 4))
                   if len(c - L.KEEP) == 1 and spacing_ok(c)]
        self.assertEqual(len(one_off), 4)
        self.assertTrue(all(one_wrong_choice(c) for c in one_off))
        self.assertIsNone(one_wrong_choice(L.KEEP))
        self.assertIsNone(one_wrong_choice(L.KEEP - {S[6]} | {S[5]}))  # +20 is 70 mm from +90: spacing broken
        self.assertIsNone(one_wrong_choice(L.KEEP - {S[2]}))

    def test_every_other_seedling_breaks_the_spacing(self):
        self.assertFalse(spacing_ok(N[0::2]))
        self.assertFalse(spacing_ok(N[1::2]))
        self.assertTrue(spacing_ok(L.KEEP))
        # the closest pair of keepers is +90 / +195 mm, 105 mm apart
        self.assertAlmostEqual(min(abs(L.START_Y[a] - L.START_Y[b]) for a, b in itertools.combinations(L.KEEP, 2)),
                               0.105)

    def test_each_arm_reaches_only_its_own_half(self):
        # the right half's far seedlings are out of the left arm's straight-down reach and vice versa
        # (tools/offline/yamkin.py); here only the layout facts the README states are checked
        self.assertEqual({n for n in L.PULL if L.START_Y[n] > 0}, set(ORDER_L))
        self.assertEqual({n for n in L.PULL if L.START_Y[n] < 0}, set(ORDER_R))
        self.assertTrue(all(abs(L.START_Y[n]) + R <= L.BED_SIZE[1] / 2 - 0.01 + 1e-9 for n in N))

    def test_masses_follow_size(self):
        for n in N:
            self.assertAlmostEqual(L.MASS[n], 800 * math.pi * R * R * L.HEIGHT[n], places=4)
            self.assertTrue(0.010 <= L.MASS[n] <= 0.025)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_plain_cylinder(self):
        self.assertAlmostEqual(tilt_deg(about("x", 10)), 10.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 170)), 10.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 70)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 90.0, places=4)

    def test_bounding_box_of_a_lying_cylinder(self):
        s = cyl(S[1], (0.2, 0.0, L.TOP + R), about("x", 90))
        self.assertTrue(np.allclose(s.high - s.low, [2 * R, L.HEIGHT[S[1]], 2 * R]))

    def test_places(self):
        self.assertEqual(place(cyl(S[1], start(S[1]))), "bed")
        self.assertEqual(place(cyl(S[1], start(S[1]) + [0, 0, 0.004])), "bed")  # 4 mm up: still on the bed
        self.assertEqual(place(cyl(S[1], start(S[1]) + [0, 0, 0.02])), "elsewhere")  # held above it
        lying = cyl(S[6], (L.BED_X, -0.04, L.BED_TOP + R), about("y", 90))
        self.assertTrue(on_bed(lying))  # knocked over on the bed: still on it (and tilted)
        for side in ("left", "right"):
            for k in range(3):
                self.assertEqual(place(cyl(S[1], *in_tray(S[1], side, k))), f"tray_{side}")
            self.assertEqual(place(cyl(S[1], *in_tray(S[1], side, 0, lying=True))), f"tray_{side}")
        tx, ty = L.TRAYS["left"]
        # held up over the tray (lowest point above the rim): not in it yet
        self.assertIsNone(tray_of(cyl(S[1], (tx, ty, L.TRAY_TOP_Z + 0.001 + L.HEIGHT[S[1]] / 2))))
        # leaning against the inside of a wall, top over the rim: in it
        lean = cyl(S[1], (tx + 0.03, ty, L.TOP + 0.022), about("y", 50))
        self.assertEqual(tray_of(lean), "left")
        # lying across the rim, centre outside the floor: not in it
        self.assertIsNone(tray_of(cyl(S[1], (tx + 0.056, ty, L.TRAY_TOP_Z + R), about("x", 90))))
        # standing on another seedling lying in the tray: in it (its lowest point is below the rim)
        self.assertEqual(tray_of(cyl(S[1], (tx, ty, L.TOP + 2 * R + L.HEIGHT[S[1]] / 2))), "left")
        # standing on the table just outside a wall: not in it
        for dx, dy in ((L.TRAY_OUTER_HALF + 0.011, 0.0), (0.0, -(L.TRAY_OUTER_HALF + 0.011))):
            self.assertIsNone(tray_of(cyl(S[1], (tx + dx, ty + dy, L.TOP + L.HEIGHT[S[1]] / 2))))
        self.assertEqual(tray_of(cyl(S[1], (tx + L.TRAY_INNER_HALF - 0.011, ty, L.TOP + L.HEIGHT[S[1]] / 2))), "left")
        self.assertEqual(place(cyl(S[1], (0.30, 0.30, L.TOP + 0.035))), "elsewhere")  # on the table
        self.assertEqual(place(cyl(S[1], (0.30, 0.70, L.TOP - 0.3))), "off_table")

    def test_moved_counts_lifting(self):
        self.assertAlmostEqual(moved(S[2], cyl(S[2], start(S[2]) + [0, 0, 0.01])), 0.01)
        self.assertAlmostEqual(moved(S[2], cyl(S[2], start(S[2]) + [0.003, 0.004, 0])), 0.005)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().hold(30))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIsNone(r.details["zero_reason"])

    def test_full_thinning_scores_one_and_succeeds_once_still(self):
        r, t = play(thin())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertTrue(r.details["keep_set_left"])
        self.assertEqual(r.details["to_pull_in_trays"], 6)
        self.assertLess(r.details["max_keeper_move_mm"], 0.1)

    def test_lying_in_the_trays_is_fine(self):
        r, _ = play(thin(lying=True))
        self.assertEqual(r.score, 1.0)

    def test_success_waits_for_stillness(self):
        f = thin(settle=0)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)  # the last one was still moving on the last step
        r = t.update(states(f.frames[-1]))
        self.assertTrue(r.success)
        # a seedling still rolling in the tray: 1.0 but no success
        fr = f.frames[-1]
        st = states(fr)
        st[S[7]].lin_vel = np.array([0.05, 0.0, 0.0])
        st[S[7]].ang_vel = np.array([5.0, 0.0, 0.0])
        r = t.update(st)
        self.assertEqual((r.score, r.success), (1.0, False))
        self.assertEqual(r.details["moving"], [S[7]])

    def test_one_keeper_nudged(self):
        f = thin(settle=5)
        f.go(10, **{S[4]: start(S[4]) + [0.008, 0, 0]}).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["nudged"], [S[4]])
        # two keepers nudged: two faults
        f.go(10, **{S[10]: start(S[10]) + [0, 0.007, 0]}).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_one_wrong_choice(self):
        # keep +240 (70 mm) instead of its taller neighbour +195 (85 mm)
        f = Film().hold(5)
        for k, n in enumerate((S[5], S[3], S[2])):
            f.drop(n, "left", k)
        for k, n in enumerate(ORDER_R):
            f.drop(n, "right", k)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["wrong_choice"], [S[2], S[1]])
        self.assertFalse(r.success)

    def test_one_fault_with_a_pulled_one_on_the_table_is_half(self):
        for nudge in (True, False):
            f = Film().hold(5)
            left_pulls = ORDER_L if nudge else (S[5], S[3], S[2])  # else keep +240 instead of +195
            for k, n in enumerate(left_pulls):
                f.drop(n, "left", k)
            for k, n in enumerate(ORDER_R[:2]):
                f.drop(n, "right", k)
            f.pull(S[7], [0.35, -0.30, L.TOP + L.HEIGHT[S[7]] / 2])  # on the table, not in a tray
            if nudge:
                f.go(10, **{S[4]: start(S[4]) + [0.008, 0, 0]})
            r, _ = play(f.hold(30))
            self.assertEqual(r.score, 0.5)

    def test_wrong_choice_plus_a_nudge_is_half(self):
        f = Film().hold(5)
        for k, n in enumerate(ORDER_L):
            f.drop(n, "left", k)
        for k, n in enumerate((S[9], S[10], S[7])):  # keep -150 (60 mm) instead of -240 (80 mm)
            f.drop(n, "right", k)
        f.go(10, **{S[6]: start(S[6]) + [0.007, 0, 0]})
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["wrong_choice"], [S[10], S[8]])

    def test_pulled_ones_left_on_the_table_score_half(self):
        f = Film().hold(5)
        for k, n in enumerate(ORDER_L):
            f.drop(n, "left", k)
        for k, n in enumerate(ORDER_R[:2]):
            f.drop(n, "right", k)
        f.pull(S[7], [0.35, -0.30, L.TOP + L.HEIGHT[S[7]] / 2])  # set down on the table, not in a tray
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["rest_composted"])
        self.assertEqual(r.details["where"][S[7]], "elsewhere")

    def test_one_arm_alone_scores_a_quarter(self):
        for left, right in ((True, False), (False, True)):
            r, _ = play(thin(left=left, right=right))
            self.assertEqual(r.score, 0.25)
            self.assertFalse(r.details["spacing_ok"])

    def test_two_pulled_is_not_enough_for_a_quarter(self):
        f = Film().hold(5)
        f.drop(S[5], "left", 0).drop(S[9], "right", 0)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        f.drop(S[3], "left", 1)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.25)

    def test_quarter_needs_every_keeper_on_the_bed(self):
        f = Film().hold(5)
        for k, n in enumerate(ORDER_L):
            f.drop(n, "left", k)
        f.drop(S[2], "left", 0, lying=True)  # a keeper pulled too
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.0)


class Zero(unittest.TestCase):
    def test_off_the_table(self):
        f = thin(settle=5)
        f.go(20, **{S[1]: (np.array([0.10, 0.70, L.TOP - 0.4]), about("x", 90))}).hold(20)
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "off_table"))

    def test_keeper_knocked_over_on_the_bed(self):
        f = thin(settle=5)
        f.go(6, **{S[6]: (np.array([L.BED_X + 0.02, -0.04, L.BED_TOP + R]), about("y", 90))}).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "disturbed"))
        self.assertEqual(r.details["disturbed"], [S[6]])

    def test_keeper_shoved_more_than_15_mm(self):
        f = thin(settle=5)
        f.go(10, **{S[10]: start(S[10]) + [0.0, 0.018, 0]}).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_seedling_to_pull_knocked_over_and_left_on_the_bed(self):
        f = Film().hold(5)
        for k, n in enumerate(ORDER_L):
            f.drop(n, "left", k)
        for k, n in enumerate(ORDER_R[:2]):
            f.drop(n, "right", k)
        f.go(6, **{S[7]: (np.array([L.BED_X, -0.095, L.BED_TOP + R]), about("y", 90))}).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "disturbed"))

    def test_keeper_rocked_past_15_degrees_and_back(self):
        f = thin(settle=5)
        f.go(4, **{S[2]: (start(S[2]), about("x", 20))}).go(4, **{S[2]: (start(S[2]), Q0)}).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "disturbed"))
        self.assertGreaterEqual(r.details["peak_tilt_deg"][S[2]], 20.0)

    def test_keeper_rocked_10_degrees_and_back_is_fine(self):
        f = thin(settle=5)
        f.go(4, **{S[2]: (start(S[2]), about("x", 10))}).go(4, **{S[2]: (start(S[2]), Q0)}).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)

    def test_over_thinned(self):
        f = thin(settle=5)
        f.drop(S[2], "left", 0, lying=True).drop(S[4], "left", 1, lying=True)
        r, _ = play(f.hold(30))
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "over_thinned"))
        self.assertEqual(len(r.details["left_on_bed"]), 2)


class Exploits(unittest.TestCase):
    def test_pull_everything(self):
        f = Film().hold(5)
        for k, n in enumerate(N):
            f.drop(n, "left" if L.START_Y[n] > 0 else "right", k, lying=True)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.0)

    def test_keep_the_most_seedlings_at_spacing_ignoring_height(self):
        # five seedlings 105-145 mm apart (+240, +130, +20, -95, -240): spaced, but the weak ones kept
        keep = {S[1], S[3], S[5], S[7], S[10]}
        f = Film().hold(5)
        for k, n in enumerate(sorted(set(N) - keep)):
            f.drop(n, side_of(n), k, lying=True)
        r, _ = play(f.hold(30))
        self.assertTrue(spacing_ok(keep))
        self.assertEqual(r.score, 0.5)

    def test_pull_every_other_one(self):
        for keep in (set(N[0::2]), set(N[1::2])):
            f = Film().hold(5)
            for k, n in enumerate(sorted(set(N) - keep)):
                f.drop(n, side_of(n), k, lying=True)
            r, _ = play(f.hold(30))
            self.assertLessEqual(r.score, 0.25)

    def test_pull_a_keeper_and_replant_it(self):
        f = thin(settle=5)
        f.pull(S[4], start(S[4])).hold(30)  # lifted out and set back exactly on its spot
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "disturbed"))

    def test_replant_a_pulled_seedling_elsewhere_on_the_bed(self):
        f = thin(settle=5)
        # take one back out of the tray and stand it in the empty gap at y +0.14 (40 mm from +90 keeper)
        f.pull(S[3], [L.BED_X, 0.0, L.BED_TOP + L.HEIGHT[S[3]] / 2]).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_hold_the_last_one_over_the_tray(self):
        f = Film().hold(5)
        for k, n in enumerate(ORDER_L):
            f.drop(n, "left", k)
        for k, n in enumerate(ORDER_R[:2]):
            f.drop(n, "right", k)
        tx, ty = L.TRAYS["right"]
        f.pull(S[7], [tx, ty, L.TRAY_TOP_Z + 0.02 + L.HEIGHT[S[7]] / 2]).hold(30)  # still held up over it
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)

    def test_knock_the_extra_ones_off_the_bed_onto_the_table(self):
        f = Film().hold(5)
        for n in sorted(L.PULL):
            f.go(5, **{n: (np.array([L.BED_X - 0.06, L.START_Y[n], L.TOP + R]), about("y", 90))})
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.5)  # spaced, but nothing composted


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = cyl(S[1], start(S[1]))
        inf.pos = np.array([0.25, 0.24, np.inf])
        zero_q = cyl(S[1], start(S[1]))
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=start(S[1]))
        frames = thin().frames
        for bad in (nan, inf, zero_q, missing, None):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.peak_move), dict(t.peak_tilt), t.steps)
            good = states(frames[-26])
            for victim in (S[1], S[6]):
                st = dict(good)
                st[victim] = bad
                r = t.update(st)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(st).score, 0.0)
            st = dict(good)
            del st[S[6]]
            self.assertEqual(t.update(st).score, 0.0)
            self.assertEqual(t.update(None).score, 0.0)
            self.assertEqual((dict(t.peak_move), dict(t.peak_tilt)), before[:2])
            self.assertEqual(t.bad_steps, 4)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual((r.score, r.success), (1.0, True))

    def test_states_anywhere_do_not_crash(self):
        t = Tracker(HZ)
        rng = np.random.default_rng(3)
        for _ in range(60):
            st = {n: cyl(n, rng.uniform([-0.5, -1.0, 0.0], [1.0, 1.0, 1.5]), about("xyz"[rng.integers(3)],
                         rng.uniform(0, 360)), vel=rng.normal(size=3), spin=rng.normal(size=3)) for n in N}
            r = t.update(st)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_score_calls_do_not_change_the_record(self):
        f = thin(settle=5)
        _, t = play(f)
        st = states(f.frames[-1], f.frames[-2])
        before = (t.steps, dict(t.peak_move))
        for _ in range(5):
            t.result(st)
        self.assertEqual((t.steps, dict(t.peak_move)), before)


class Reset(unittest.TestCase):
    def disturbed(self):
        f = Film().hold(5).go(10, **{S[6]: start(S[6]) + [0.02, 0, 0]})
        thin(f, settle=5)
        _, t = play(f)
        self.assertEqual(t.result(states(f.frames[-1])).score, 0.0)
        return t, f

    def test_teleport_back_to_start_resets(self):
        t, _ = self.disturbed()
        r = t.update(states(Film().frames[0]))  # everything back on its spot in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(max(t.peak_move.values()), 0.0)
        f = thin()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_reset_right_after_a_blown_up_step(self):
        t, f = self.disturbed()
        bad = states(f.frames[-1])
        bad[S[3]] = SimpleNamespace(pos=np.full(3, np.nan))
        t.update(bad)
        t.update(states(Film().frames[0]))  # the simulator's reset comes next
        self.assertEqual(t.steps, 1)
        self.assertEqual(max(t.peak_move.values()), 0.0)

    def test_gradual_return_does_not_reset(self):
        t, f = self.disturbed()
        g = Film()
        g.cur = dict(f.cur)
        g.frames = [dict(f.cur)]
        for n in L.PULL:  # every pulled seedling carried back to its spot, one at a time
            g.pull(n, start(n))
        g.go(10, **{S[6]: start(S[6])}).hold(10)
        play(g, t)
        self.assertGreater(t.peak_move[S[6]], 0.015)  # back at the start, but gradually: same episode
        self.assertEqual(t.result(states(g.frames[-1])).score, 0.0)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t, f = self.disturbed()
        fr = dict(f.cur)
        fr[S[1]] = (np.array([0.40, 0.0, L.TOP + 0.035]), Q0)  # knocked far, not to the start
        steps = t.steps
        t.update(states(fr))
        self.assertEqual(t.steps, steps + 1)
        self.assertGreater(t.peak_move[S[1]], 0.1)
        self.assertGreater(t.peak_move[S[6]], 0.015)
        # every pulled seedling teleported back to its spot but the shoved keeper still off its spot: no reset
        fr = dict(f.cur)
        for n in L.PULL:
            fr[n] = (start(n), Q0)
        t.update(states(fr))
        self.assertEqual(t.steps, steps + 2)
        self.assertGreater(t.peak_move[S[1]], 0.1)

    def test_a_dropped_seedling_is_not_a_teleport(self):
        t, f = self.disturbed()
        fr = dict(f.cur)
        p = in_tray(S[1], "left", 0)[0] + [0, 0, 0.3]
        z, v = p[2], 0.0
        prev = None
        while True:
            v += 9.81 * DT
            z = max(in_tray(S[1], "left", 0)[0][2], z - v * DT)
            fr = dict(fr)
            fr[S[1]] = (np.array([p[0], p[1], z]), Q0)
            st = states(fr, prev)
            t.update(st)
            prev = fr
            if z <= in_tray(S[1], "left", 0)[0][2]:
                break
        self.assertGreater(t.peak_move[S[6]], 0.015)


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

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in N:
            o = next(o for o in self.task.OBJECTS if o.name == n)
            self.assertAlmostEqual(o.mass, L.MASS[n])
            self.assertAlmostEqual(o.height, L.HEIGHT[n])
            self.assertAlmostEqual(o.pos[2] - o.height / 2, L.BED_TOP)
        fixed = [o.name for o in self.task.OBJECTS if o.mass is None]
        self.assertEqual(len(fixed), 1 + 8)  # the bed and two trays of four walls
        self.assertEqual(self.task.SCORE_OVER, "end")

    def test_task_json_agrees(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        self.assertIn("1,600 steps", meta["run_limit"])
        readme = (HERE / "README.md").read_text()
        self.assertIn(self.task.INSTRUCTION, readme)

    def test_hooks(self):
        tr = self.task._TRACKER
        tr.reset()
        objs = {n: cyl(n, start(n)) for n in N}
        self.assertFalse(self.task.success(objs))
        self.assertEqual(self.task.score(objs), 0.0)
        self.assertEqual(tr.steps, 1)
        self.task.score(objs)
        self.assertEqual(tr.steps, 1)  # score() reads, it does not feed the tracker
        self.assertFalse(self.task.success({}))  # nothing readable: no crash
        self.assertEqual(self.task.score({}), 0.0)
        f = thin()
        prev = None
        for fr in f.frames:
            st = states(fr, prev)
            ok = self.task.success(st)
            prev = fr
        self.assertTrue(ok)
        self.assertEqual(self.task.score(st), 1.0)
        tr.reset()

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        path = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success({n: cyl(n, start(n)) for n in N}))  # unwritable: ignored
            os.environ["TASK_TRACE"] = str(path)
            self.task.success({n: cyl(n, start(n)) for n in N})
            row = json.loads(path.read_text().splitlines()[-1])
            for k in ("step", "score", "zero_reason", "left_on_bed", "spacing_ok", "wrong_choice", "where",
                      "peak_move_mm", "peak_tilt_deg", "max_keeper_move_mm", "to_pull_in_trays", "pos"):
                self.assertIn(k, row)
        finally:
            if path.exists():
                path.unlink()
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
            self.task._TRACKER.reset()

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, order in (("left", ORDER_L), ("right", ORDER_R)):
            labels = [w.label for w in plan[side]]
            self.assertEqual([lb[5:] for lb in labels if lb.startswith("over seedling")], list(order))
            self.assertEqual(sum(lb == "grip the stem" for lb in labels), 3)
            self.assertEqual(sum(lb == "let it drop into the tray" for lb in labels), 3)
        # the right arm reaches the row only after the left has pulled its first seedling (near the middle) up
        lw, rw = plan["left"], plan["right"]
        left_lifted = sum(w.duration for w in lw[:5])
        self.assertEqual(lw[4].label, "pull it straight up")
        right_over = sum(w.duration for w in rw[:3])
        self.assertEqual(rw[2].label, "over seedling_09")
        self.assertGreater(right_over, left_lifted)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_gripper_targets(self):
        self.assertAlmostEqual(-2 * self.task.GRIP, 0.012)  # 8 mm narrower than the 20 mm stem
        for n in L.PULL:
            gz = self.task.grip_z(n)
            self.assertGreaterEqual(gz - self.task.PAD_TIPS - L.BED_TOP, 0.010 - 1e-9)  # tips 10+ mm over the bed
            self.assertLessEqual(L.BED_TOP + L.HEIGHT[n] - gz, 0.038 + 1e-9)  # stem top <= 38 mm over the TCP
            bottom = self.task.CARRY_Z - (gz - L.BED_TOP)
            self.assertGreater(bottom, L.BED_TOP + max(L.HEIGHT.values()))  # carried over every seedling
            self.assertGreater(self.task.RELEASE_Z - (gz - L.BED_TOP), L.TRAY_TOP_Z + 0.015)  # let go over the rim
        self.assertGreater(self.task.RELEASE_Z - self.task.PAD_TIPS,
                           L.TOP + max(L.HEIGHT[n] for n in L.PULL if n not in (S[1],)) + 0.010)


if __name__ == "__main__":
    unittest.main()
