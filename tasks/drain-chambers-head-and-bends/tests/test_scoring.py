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
    Tracker, bottom_gap, evaluate, held, highest, lowest, nearest_point, reach, rotation, standing, tilt_deg, valid,
)

MM = 0.001
H, P, B1, B2 = L.H, L.P, L.B1, L.B2


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = np.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(np.sin(h) * np.array(v, float)) + (np.cos(h),)


def state(center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A chamber's state: its world bounding box worked out from the turned cylinder."""
    c = np.array(center, float)
    a = rotation(quat)[:, 2]
    ext = np.array([L.CHAMBER_H / 2 * abs(a[k]) + L.CHAMBER_RADIUS * math.sqrt(max(0.0, 1 - a[k] ** 2))
                    for k in range(3)])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def chamber(x, y, bottom=L.SHEET_TOP, tilt=0.0, axis="x", **kw):
    """A chamber centred at (x, y) with its lowest point at ``bottom``, tilted ``tilt`` degrees about ``axis``."""
    q = about(axis, tilt)
    az = abs(rotation(q)[2, 2])
    half = L.CHAMBER_H / 2 * az + L.CHAMBER_RADIUS * math.sqrt(max(0.0, 1 - az * az))
    return state((x, y, bottom + half), q, **kw)


def start():
    return [chamber(*L.START[n]) for n in L.NAMES]


S0 = start()


def at(*xys, **over):
    """Chambers at the given (x, y) in order; ``over``: name -> state replaces that chamber."""
    out = [chamber(x, y) for x, y in xys]
    for name, s in over.items():
        out[L.NAMES.index(name)] = s
    return out


DONE = at(H, P, B1, B2)


def score_of(cs, arms=None):
    return evaluate(cs, arms)[0]


def glide(a, b, n):
    """n frames moving every chamber of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state((1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw) for sa, sb in zip(a, b, strict=True)]
            for i in range(1, n + 1)]


def build(n=20, final=None, hold=30):
    """From the start, every chamber moved gradually to the four points (or ``final``), then held there."""
    final = DONE if final is None else final
    return [S0] + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0, arms=None):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


ON = 2 * L.CHAMBER_RADIUS / L.FULL_GAP  # rigstate's opening with the fingers closed on a 60 mm chamber


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    """A rigstate.arms() snapshot: each given arm's pad-tip point (x, y, z) and opening (0 closed .. 1 open)."""
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


# Both arms parked where the demo leaves them, pad tips 127 mm over the sheet.
PARKED = snap(left=(0.280, 0.230, L.SHEET_TOP + 0.127), right=(0.280, -0.290, L.SHEET_TOP + 0.127))


def along_route(s):
    """The point s m along the route's centre line from the head."""
    pts = [H, P, B1, B2, L.EXIST]
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if s <= d + 1e-12:
            f = s / d
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        s -= d
    return pts[-1]


class Geometry(unittest.TestCase):
    def test_lowest_point_and_tilt(self):
        c = chamber(*H)
        self.assertAlmostEqual(lowest(c), L.SHEET_TOP)
        self.assertAlmostEqual(highest(c), L.SHEET_TOP + L.CHAMBER_H)
        self.assertAlmostEqual(tilt_deg(c.quat_xyzw), 0.0)
        flipped = chamber(*H, tilt=180.0)
        self.assertAlmostEqual(tilt_deg(flipped.quat_xyzw), 0.0, places=6)  # either end up
        side = chamber(*H, tilt=90.0)
        self.assertAlmostEqual(tilt_deg(side.quat_xyzw), 90.0, places=6)
        self.assertAlmostEqual(lowest(side), L.SHEET_TOP)
        self.assertAlmostEqual(highest(side) - lowest(side), 2 * L.CHAMBER_RADIUS)

    def test_standing(self):
        self.assertTrue(standing(chamber(*H)))
        self.assertTrue(standing(chamber(*H, tilt=1.1, bottom=L.SHEET_TOP)))  # rim on the sheet, across a line
        self.assertTrue(standing(chamber(*H, bottom=L.LINE_TOP)))  # resting on a drawn line
        self.assertTrue(standing(chamber(*H, tilt=10.0)))
        self.assertFalse(standing(chamber(*H, tilt=10.5)))
        self.assertTrue(standing(chamber(*H, bottom=L.SHEET_TOP + 0.0029)))
        self.assertFalse(standing(chamber(*H, bottom=L.SHEET_TOP + 0.0031)))  # held up
        self.assertFalse(standing(chamber(*H, bottom=L.SHEET_TOP + L.CHAMBER_H)))  # stacked on another
        self.assertFalse(standing(chamber(*H, bottom=L.TOP - 0.2)))  # fallen off the table

    def test_valid(self):
        self.assertTrue(valid(chamber(*H)))
        self.assertFalse(valid(state((np.nan, 0, 0.77))))
        self.assertFalse(valid(state((0.3, 0, 0.77), quat=(0, 0, 0, 0))))
        self.assertFalse(valid(state((0.3, 0, 0.77), vel=(0, np.inf, 0))))
        self.assertFalse(valid(SimpleNamespace(pos=(0, 0, 0))))

    def test_nearest_point(self):
        self.assertEqual(nearest_point(chamber(*P))[0], "size_change")
        name, d = nearest_point(chamber(B2[0] + 0.010, B2[1]))
        self.assertEqual(name, "bend_2")
        self.assertAlmostEqual(d, 0.010)

    def test_scale_and_layout_arithmetic(self):
        self.assertAlmostEqual(L.ROUTE_LEN, 0.700)
        self.assertEqual([round(v * L.SCALE, 2) for v in L.LENGTHS.values()], [7.5, 8.0, 7.0, 12.5])
        for v in L.LENGTHS.values():  # Table 13: no length needs an extra access point for spacing
            self.assertLess(v * L.SCALE, L.TABLE13_FROM_START)
            self.assertLess(v * L.SCALE, L.TABLE13_CHAMBERS)
        self.assertAlmostEqual(L.COVER_TOL * L.SCALE, 0.75)
        self.assertAlmostEqual(2 * L.CHAMBER_RADIUS * L.SCALE, 3.0)
        pts = list(L.POINTS.values())
        gaps = [math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]]
        self.assertGreaterEqual(min(gaps), 0.140 - 1e-9)  # a chamber covers at most one point
        self.assertGreater(min(gaps), 2 * L.COVER_TOL + 2 * L.CHAMBER_RADIUS)

    def test_route_is_drawn_as_specified(self):
        """The strips abut without overlapping, the 100 mm line is 10 mm wide and runs head to P, the 150 mm line
        20 mm wide from P round both bends to the existing chamber's side; their centre lines pass through every
        point."""
        strips = {name: (x0, x1, y0, y1) for name, x0, x1, y0, y1 in L.ROUTE}
        x0, x1, y0, y1 = strips["pipe_100"]
        self.assertAlmostEqual(x1 - x0, L.W100)
        self.assertAlmostEqual((y0, y1), (H[1], P[1]))
        for name in ("pipe_150_a", "pipe_150_c"):
            x0, x1, _, _ = strips[name]
            self.assertAlmostEqual(x1 - x0, L.W150)
        _, _, y0, y1 = strips["pipe_150_b"]
        self.assertAlmostEqual(y1 - y0, L.W150)
        self.assertAlmostEqual(strips["pipe_150_c"][3], L.EXIST[1] - L.EXIST_RADIUS)
        rects = list(strips.values())
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                ox = min(a[1], b[1]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[2], b[2])
                self.assertLessEqual(min(ox, oy), 1e-9)  # touching at most
        for p in L.POINTS.values():
            self.assertTrue(any(r[0] - 1e-9 <= p[0] <= r[1] + 1e-9 and r[2] - 1e-9 <= p[1] <= r[3] + 1e-9
                                for r in rects), p)
        self.assertGreater(L.W150 - L.W100, 0.005)  # the width change reads in the setup screenshot

    def test_starts_are_off_the_route_and_apart(self):
        segs = [(H, P), (P, B1), (B1, B2), (B2, L.EXIST)]

        def seg_dist(p, a, b):
            a, b, p = np.array(a), np.array(b), np.array(p)
            t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
            return float(np.linalg.norm(p - (a + t * (b - a))))

        for name, xy in L.START.items():
            self.assertGreaterEqual(min(seg_dist(xy, a, b) for a, b in segs), 0.060 - 1e-9, name)
            self.assertGreaterEqual(math.dist(xy, L.EXIST), 0.120, name)
            self.assertGreater(nearest_point(chamber(*xy))[1], L.COVER_TOL + 0.05)
        xs = list(L.START.values())
        self.assertGreaterEqual(min(math.dist(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]), 0.095)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual(score, 0.0)
        self.assertEqual(d["n_covered"], 0)
        self.assertEqual(d["missing"], list(L.POINT_NAMES))

    def test_all_four_points_score_one(self):
        score, d = evaluate(DONE)
        self.assertEqual((score, d["n_covered"]), (1.0, 4))
        self.assertEqual(set(d["covered"]), set(L.POINT_NAMES))

    def test_each_count_scores_a_quarter(self):
        far = [L.START[n] for n in L.NAMES]
        pts = [H, P, B1, B2]
        for n in range(5):
            cs = at(*(pts[:n] + far[n:]))
            self.assertEqual(score_of(cs), 0.25 * n, n)

    def test_any_chamber_may_be_at_any_point(self):
        for perm in ((B2, B1, P, H), (P, H, B2, B1), (B1, B2, H, P)):
            self.assertEqual(score_of(at(*perm)), 1.0)

    def test_cover_radius_is_15_mm(self):
        for point in L.POINTS.values():
            for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                inside = (point[0] + 0.0149 * math.cos(ang), point[1] + 0.0149 * math.sin(ang))
                outside = (point[0] + 0.0151 * math.cos(ang), point[1] + 0.0151 * math.sin(ang))
                others = [p for p in L.POINTS.values() if p != point]
                self.assertEqual(score_of(at(inside, *others)), 1.0)
                self.assertEqual(score_of(at(outside, *others)), 0.75)

    def test_every_outcome_listed_in_task_json(self):
        mid_hp = ((H[0] + P[0]) / 2, (H[1] + P[1]) / 2)
        against = (L.EXIST[0], L.EXIST[1] - L.EXIST_RADIUS - L.CHAMBER_RADIUS)  # pushed up to the existing one
        far = [L.START[n] for n in L.NAMES]
        cases = {
            "H, P, B1, B2": (at(H, P, B1, B2), 1.0),
            "fourth on a straight": (at(H, mid_hp, B1, B2), 0.75),
            "fourth against the existing chamber": (at(H, against, B1, B2), 0.75),
            "on its side at B2": (at(H, P, B1, B2, chamber_4=chamber(*B2, tilt=90.0)), 0.75),
            "the two bends only": (at(B1, B2, far[2], far[3]), 0.5),
            "H and P only": (at(H, P, far[2], far[3]), 0.5),
            "even spacing": (at(*[along_route(k * 0.175) for k in range(4)]), 0.25),
            "nothing moved": (S0, 0.0),
        }
        for label, (cs, want) in cases.items():
            self.assertEqual(score_of(cs), want, label)
        # still gripped: 0.75
        gx, gy = P
        grip = snap(right=(gx, gy, L.SHEET_TOP + 0.011), right_open=ON, left=PARKED["left"].tips)
        self.assertEqual(score_of(DONE, grip), 0.75)

    def test_even_spacing_misses_by_25_40_75_mm(self):
        pts = [along_route(k * 0.175) for k in range(4)]
        self.assertEqual(pts[0], H)
        misses = [round(nearest_point(chamber(*p))[1] * 1000, 1) for p in pts[1:]]
        self.assertEqual(misses, [25.0, 40.0, 75.0])

    def test_every_score_turns_up(self):
        seen = set()
        far = [L.START[n] for n in L.NAMES]
        pts = [H, P, B1, B2]
        for n in range(5):
            seen.add(score_of(at(*(pts[:n] + far[n:]))))
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_one_needs_still_for_one_second(self):
        # all four in place but chamber_2 moving 3 mm in the second-to-last 0.5 s window: 0.75 until still for 1 s
        frames = build(hold=0)
        frames += [DONE] * 3
        wobble = list(DONE)
        wobble[1] = chamber(P[0] + 0.003, P[1])
        frames += [wobble] + [DONE] * 11  # moved 3 mm and back: path 6 mm, in the 0.5 s before the last 0.5 s
        r, t = run(frames)
        self.assertTrue(all(t.settled()))  # settled over the last 0.5 s ...
        self.assertFalse(t.still())  # ... but not over the 0.5 s before
        self.assertEqual((r.score, r.success), (0.75, False))
        r = None
        for _ in range(10):
            r = t.update(DONE)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_a_settling_chamber_is_not_counted(self):
        frames = build(hold=0)
        drift = [list(DONE) for _ in range(12)]
        for k, f in enumerate(drift):  # chamber_3 creeping 0.25 mm a step: 2.5 mm in 0.5 s
            f[2] = chamber(B1[0] + 0.00025 * k, B1[1])
        r, t = run(frames + drift)
        self.assertEqual(r.score, 0.75)
        self.assertIn("chamber_3", r.details["not_settled"])
        drift = [list(DONE) for _ in range(12)]
        for k, f in enumerate(drift):  # 0.15 mm a step: 1.5 mm in 0.5 s, settled
            f[2] = chamber(B1[0] + 0.00015 * k, B1[1])
        r, t = run(frames + [DONE] * 11 + drift)
        self.assertEqual(r.details["n_covered"], 4)

    def test_tracker_needs_half_a_second_of_history(self):
        t = Tracker()
        for k in range(10):
            r = t.update(DONE)
            self.assertEqual(r.score, 0.0, k)  # not yet settled: under 0.5 s seen
        r = t.update(DONE)
        self.assertEqual(r.score, 0.75)
        for _ in range(10):
            r = t.update(DONE)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_the_state_at_hand(self):
        r, t = run(build(hold=30) + glide(DONE, S0, 20) + [S0] * 15)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.best, 1.0)


class ZeroConditions(unittest.TestCase):
    def test_tipped_held_up_stacked_or_gone(self):
        for bad in (chamber(*H, tilt=12.0), chamber(*H, bottom=L.SHEET_TOP + 0.010),
                    chamber(*H, bottom=L.SHEET_TOP + L.CHAMBER_H), chamber(*H, bottom=L.TOP - 0.3)):
            cs = at(H, P, B1, B2, chamber_1=bad)
            self.assertEqual(score_of(cs), 0.75)
            only = at(*[L.START[n] for n in L.NAMES], chamber_1=bad)
            self.assertEqual(score_of(only), 0.0)

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        r, t = run(build(hold=25))
        self.assertTrue(r.success)
        hist = len(t.history)
        bad = list(DONE)
        bad[2] = state((np.nan, 0.0, 0.77))
        r = t.update(bad)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(r.details["bad_chambers"], ["chamber_3"])
        self.assertEqual(t.bad_steps, 1)
        self.assertEqual(len(t.history), hist)
        r = t.update(DONE)  # the next good step carries on
        self.assertTrue(r.success)
        for broken in (state((0.3, 0, 0.77), quat=(0, 0, 0, 0)), state((0.3, 0, 0.77), spin=(0, np.inf, 0))):
            r = t.update([broken, *DONE[1:]])
            self.assertEqual(r.score, 0.0)
        self.assertEqual(t.result(DONE[:3]).score, 0.0)  # a chamber missing
        self.assertEqual(t.update(DONE[:3]).score, 0.0)


class LetGo(unittest.TestCase):
    """A chamber counts only once let go: no gripper's pad-tip point lower than 10 mm above its top within that
    gripper's finger reach of its axis (95 mm fully open, 80 mm closed on a chamber)."""

    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0), 0.0954, places=4)
        self.assertAlmostEqual(L.release_reach(ON), 0.0797, places=4)
        self.assertAlmostEqual(L.release_reach(0.0), 0.0574, places=4)
        self.assertAlmostEqual(reach(SimpleNamespace(opening=float("nan"))), L.release_reach(1.0))
        self.assertAlmostEqual(reach(SimpleNamespace()), L.release_reach(1.0))

    def test_parked_arms_let_every_chamber_go(self):
        score, d = evaluate(DONE, PARKED)
        self.assertEqual((score, d["held"]), (1.0, []))

    def test_a_chamber_still_gripped_does_not_count(self):
        for name, (x, y), side in (("chamber_1", H, "right"), ("chamber_3", B1, "left")):
            grip = snap(**{side: (x, y, L.SHEET_TOP + 0.011), f"{side}_open": ON})
            score, d = evaluate(DONE, grip)
            self.assertEqual((score, d["held"]), (0.75, [name]))
            self.assertNotIn(name, d["counted"])

    def test_an_open_gripper_round_a_chamber_still_holds_it(self):
        self.assertTrue(held(chamber(*P), snap(right=(P[0], P[1], L.SHEET_TOP + 0.020))))

    def test_a_finger_propping_a_chamber_is_holding_it(self):
        c = chamber(*P)
        low = L.SHEET_TOP + 0.015
        self.assertTrue(held(c, snap(right=(P[0] + 0.095, P[1], low))))
        self.assertFalse(held(c, snap(right=(P[0] + 0.096, P[1], low))))
        self.assertTrue(held(c, snap(right=(P[0], P[1] + 0.057, low), right_open=0.0)))
        self.assertFalse(held(c, snap(right=(P[0], P[1] + 0.058, low), right_open=0.0)))

    def test_pad_tips_10_mm_over_the_top_have_let_go(self):
        c = chamber(*B2)
        top = highest(c)
        self.assertTrue(held(c, snap(left=(B2[0], B2[1], top + 0.0099))))
        self.assertFalse(held(c, snap(left=(B2[0], B2[1], top + 0.0101))))
        self.assertEqual(score_of(DONE, snap(left=(B2[0], B2[1], top + 0.011))), 1.0)

    def test_success_needs_every_chamber_let_go(self):
        r, _ = run(build(hold=30), arms=snap(left=(B2[0], B2[1], L.SHEET_TOP + 0.011), left_open=ON))
        self.assertEqual((r.score, r.success), (0.75, False))
        r, _ = run(build(hold=30), arms=PARKED)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_unreadable_arms_count_as_let_go_and_are_counted(self):
        frames = build(hold=25)
        r, t = run(frames, arms=None)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.arms_unavailable, len(frames))
        self.assertEqual(r.details["arms_unavailable_by_side"], {"left": len(frames), "right": len(frames)})

    def test_one_unreadable_arm_never_hides_the_other(self):
        right = SimpleNamespace(tips=np.array((H[0], H[1], L.SHEET_TOP + 0.011)), opening=ON)
        frames = build(hold=25)
        r, t = run(frames, arms={"left": SimpleNamespace(tips=np.full(3, np.nan), opening=1.0), "right": right})
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["held"], ["chamber_1"])
        self.assertEqual(r.details["arms_readable"], ["right"])
        self.assertEqual(t.arms_unavailable_by_side, {"left": len(frames), "right": 0})
        self.assertEqual(t.update(DONE, {"right": right}).score, 0.75)  # the left arm missing altogether

    def test_rigstate_reset_hook_resets_the_tracker(self):
        r, t = run(build(hold=25))
        self.assertEqual(t.best, 1.0)
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            rigstate.offline_reset()
            self.assertEqual((t.steps, t.best, t.first_at), (0, 0.0, {}))
            self.assertIsNone(rigstate.arms())
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        r, t = run(build(hold=25))
        self.assertEqual(t.best, 1.0)
        t.update(S0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(t.best, 0.0)
        self.assertEqual(t.first_at, {})
        self.assertEqual(len(t.history), 1)

    def test_gradual_return_is_not_a_reset(self):
        r, t = run(build(hold=25) + glide(DONE, S0, 30))
        self.assertEqual(t.best, 1.0)
        self.assertGreater(t.steps, 70)

    def test_jump_elsewhere_is_not_a_reset(self):
        r, t = run(build(hold=25))
        moved = at(*[(x + 0.10, y) for x, y in L.START.values()])
        t.update(moved)
        self.assertEqual(t.best, 1.0)

    def test_jittered_reset_counts(self):
        r, t = run(build(hold=25))
        jit = at(*[(x + 0.01, y - 0.01) for x, y in L.START.values()])
        t.update(jit)
        self.assertEqual((t.steps, t.best), (1, 0.0))


class Exploits(unittest.TestCase):
    """The spec's shortcut checks and other strategies that skip the rule earn only their own points."""

    def test_head_and_bends_with_the_fourth_on_a_straight(self):
        self.assertEqual(score_of(at(H, (0.370, -0.185), B1, B2)), 0.75)

    def test_spaced_evenly_along_the_route(self):
        self.assertEqual(score_of(at(*[along_route(k * 0.175) for k in range(4)])), 0.25)
        self.assertEqual(score_of(at(*[along_route(k * 0.7 / 3) for k in range(4)])), 0.25)  # ends included

    def test_both_ends_and_the_corners(self):
        """The head, the two bends and one hard against the existing chamber: the size change is missed."""
        against = (L.EXIST[0], L.EXIST[1] - L.EXIST_RADIUS - L.CHAMBER_RADIUS)
        self.assertEqual(score_of(at(H, B1, B2, against)), 0.75)

    def test_corners_and_size_change_without_the_head(self):
        self.assertEqual(score_of(at((0.370, -0.200), P, B1, B2)), 0.75)

    def test_holding_a_chamber_on_its_point_earns_nothing_for_it(self):
        grip = snap(right=(P[0], P[1], L.SHEET_TOP + 0.011), right_open=ON, left=PARKED["left"].tips)
        r, _ = run(build(hold=30), arms=grip)
        self.assertEqual((r.score, r.success), (0.75, False))

    def test_lying_down_on_the_points(self):
        self.assertEqual(score_of(at(*[(x, y) for x, y in (H, P, B1, B2)],
                                     **{n: chamber(*p, tilt=90.0) for n, p in zip(L.NAMES, (H, P, B1, B2))})), 0.0)
        self.assertEqual(score_of(at(H, P, B1, B2, chamber_4=chamber(*B2, tilt=90.0))), 0.75)

    def test_stacked_on_one_point_counts_once(self):
        cs = at(H, P, B1, B1, chamber_4=chamber(*B1, bottom=L.SHEET_TOP + L.CHAMBER_H))
        score, d = evaluate(cs)
        self.assertEqual((score, d["n_covered"]), (0.5 + 0.25, 3))
        self.assertIn("chamber_4", d["not_standing"])

    def test_two_chambers_cannot_share_a_point(self):
        """Only one 60 mm chamber can stand within 15 mm of a point; a second one beside it is 60 mm away."""
        cs = at(H, P, B1, (B1[0] - 0.060, B1[1]))
        self.assertEqual(score_of(cs), 0.75)

    def test_random_states_score_only_listed_values(self):
        rng = np.random.default_rng(11)
        seen = set()
        pts = list(L.POINTS.values())
        for _ in range(500):
            xy = []
            for _k in range(4):
                if rng.random() < 0.5:
                    p = pts[rng.integers(4)]
                    xy.append((p[0] + rng.normal(0, 0.012), p[1] + rng.normal(0, 0.012)))
                else:
                    xy.append((rng.uniform(L.SHEET[0], L.SHEET[1]), rng.uniform(L.SHEET[2], L.SHEET[3])))
            cs = [chamber(x, y, tilt=float(rng.choice([0.0, 0.0, 5.0, 15.0, 90.0]))) for x, y in xy]
            s = score_of(cs)
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertGreaterEqual(len(seen), 4)


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

    def objects(self, chambers):
        return dict(zip(self.task.CHAMBER_NAMES, chambers, strict=True))

    def feed(self, frames, arms=None):
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        ok = None
        for f in frames:
            if arms is not None:
                rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for s, a in arms.items()})
            ok = self.task.success(self.objects(f))
        return ok

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.CHAMBER_NAMES, L.NAMES)
        for spec, s in zip(t.CHAMBERS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertEqual((spec.radius, spec.height, spec.mass), (L.CHAMBER_RADIUS, L.CHAMBER_H, L.CHAMBER_MASS))
        loose = [o.name for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(loose, list(L.NAMES))
        self.assertEqual(len(t.OBJECTS), 1 + 4 + 1 + 2 + 4)
        self.assertAlmostEqual(t.SHEET.pos[2] + t.SHEET.size[2] / 2, L.SHEET_TOP)
        for r in t.ROUTE:
            self.assertAlmostEqual(r.pos[2] - r.size[2] / 2, L.SHEET_TOP)
            self.assertEqual(r.color, L.ROUTE_BLUE)
        self.assertAlmostEqual(t.EXISTING.pos[2] - L.EXIST_H / 2, L.SHEET_TOP)
        self.assertAlmostEqual(t.COVER.pos[2] - L.COVER_H / 2, L.SHEET_TOP + L.EXIST_H)
        # Every object on the sheet, nothing in the band the arms' start pose hides in the setup screenshot.
        for o in t.OBJECTS:
            hx = o.size[0] / 2 if hasattr(o, "size") else o.radius
            hy = o.size[1] / 2 if hasattr(o, "size") else o.radius
            self.assertLess(abs(o.pos[1]) + hy, 0.45, o.name)
            self.assertGreaterEqual(o.pos[0] - hx, L.SHEET[0] - 1e-9, o.name)
            self.assertLessEqual(o.pos[0] + hx, L.SHEET[1] + 1e-9, o.name)
            self.assertGreaterEqual(o.pos[1] - hy, L.SHEET[2] - 1e-9, o.name)
            self.assertLessEqual(o.pos[1] + hy, L.SHEET[3] + 1e-9, o.name)

    def test_hooks_and_trace(self):
        rigstate.offline_reset()  # no arms read (as offline): the let-go rule is skipped
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(self.objects(S0)))
                self.assertEqual(self.task.score(self.objects(S0)), 0.0)
                ok = None
                for _ in range(21):
                    ok = self.task.success(self.objects(DONE))
                self.assertTrue(ok)
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 22)
        for key in ("step", "score", "success", "n_covered", "covered", "missing", "counted", "held",
                    "not_standing", "not_settled", "all_still_1s", "nearest", "bad_steps", "first_step",
                    "chamber_1", "chamber_4", "rigstate_error"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["chamber_1"]["nearest"], "head")
        self.assertEqual(rows[-1]["chamber_1"]["off_mm"], 0.0)
        self.assertEqual(rows[-1]["chamber_2"]["path_mm"], 0.0)
        self.assertTrue(rows[-1]["chamber_4"]["standing"])
        self.assertIsNone(rows[0]["chamber_1"]["path_mm"])
        self.assertEqual(rows[0]["missing"], list(L.POINT_NAMES))
        self.assertFalse(rows[-1]["chamber_3"]["held"])

    def test_hooks_read_the_arms(self):
        """task.py passes rigstate's arms to the scorer: a chamber still gripped is not counted; once the gripper
        has risen clear it is, and the trace shows both arms."""
        left_grip = snap(left=(B2[0], B2[1], L.SHEET_TOP + 0.011), left_open=ON, right=PARKED["right"].tips)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.feed([DONE] * 25, left_grip))
                self.assertEqual(self.task.score(self.objects(DONE)), 0.75)
                rigstate.offline_feed({s: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening}
                                       for s, a in PARKED.items()})
                self.assertTrue(self.task.success(self.objects(DONE)))
                self.assertEqual(self.task.score(self.objects(DONE)), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
                rigstate.offline_reset()
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(rows[0]["chamber_4"]["held"])
        self.assertEqual(rows[0]["held"], ["chamber_4"])
        self.assertAlmostEqual(rows[0]["left_opening"], ON)
        self.assertEqual(rows[-1]["held"], [])
        self.assertEqual(rows[-1]["arms_readable"], ["left", "right"])
        self.assertEqual(rows[-1]["arms_unavailable"], 0)

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
                self.assertEqual(self.task.score(self.objects([state((np.nan, 0, 0.77)), *S0[1:]])), 0.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                row = json.loads(f.readline())
        self.assertFalse(row["valid_state"])
        self.assertIsNone(row["chamber_2"]["path_mm"])

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_covers_every_point_and_each_arm_works_its_own_side(self):
        t = self.task
        self.assertAlmostEqual(t.LAND_SHORT, 0.004)  # the GPU lesson: parts land about 4 mm short
        aimed, short = list(S0), list(S0)
        for arm, steps in t.PLAN.items():
            sign = 1.0 if arm == "left" else -1.0
            for name, point in steps:
                self.assertGreater(sign * L.START[name][1], 0.0)  # each arm fetches only its own side's chambers
                x, y = t.place_point(name, point, arm)
                px, py = L.POINTS[point]
                self.assertAlmostEqual(math.hypot(x - px, y - py), t.LAND_SHORT)  # aimed 4 mm on, along the carry
                aimed[L.NAMES.index(name)] = chamber(x, y)
                short[L.NAMES.index(name)] = chamber(px, py)  # landed the expected 4 mm short: on the point
        self.assertEqual({p for _, p in t.PLAN["left"]} | {p for _, p in t.PLAN["right"]}, set(L.POINT_NAMES))
        self.assertEqual(score_of(aimed), 1.0)
        self.assertEqual(score_of(short), 1.0)
        for idle in ("left", "right"):  # one arm's chambers left where they start: two points, 0.5
            part = list(short)
            for name, _ in t.PLAN[idle]:
                part[L.NAMES.index(name)] = S0[L.NAMES.index(name)]
            self.assertEqual(score_of(part), 0.5)

    @staticmethod
    def spells(wps):
        """(start, end, x, y) of each waypoint, on the plan's clock."""
        out, tt = [], 0.0
        for w in wps:
            out.append((tt, tt + w.duration, w.pos[0], w.pos[1]))
            tt += w.duration
        return out

    def test_grippers_stay_apart(self):
        """The two grippers' waypoints are at least 0.20 m apart whenever they overlap in time, the left arm
        comes down at B2 only after the right arm has left P, and the demo fits the run limit."""
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, xa, ya in left:
            for b0, b1, xb, yb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(math.hypot(xa - xb, ya - yb), 0.20)
        right_leaves_p = max(e for (s, e, x, y), w in zip(right, plan["right"], strict=True)
                             if w.label == "straight up, clear of it")
        left_at_b2 = min(s for (s, e, x, y), w in zip(left, plan["left"], strict=True)
                         if w.label == "down to just above the plan" and y < 0.1 and x < 0.3)
        self.assertGreaterEqual(left_at_b2, right_leaves_p)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 1200)

    def test_gripper_numbers(self):
        t = self.task
        self.assertAlmostEqual(-2 * t.GRIP, 2 * L.CHAMBER_RADIUS - 0.008)  # commanded 8 mm narrower
        self.assertGreaterEqual(-2 * t.OPEN - 2 * L.CHAMBER_RADIUS, 0.020)  # 10 mm either side when open
        self.assertLessEqual(-2 * t.OPEN, L.FULL_GAP)
        self.assertGreaterEqual(t.PICK_Z - 0.013 - L.SHEET_TOP, 0.010)  # pad tips 11 mm over the sheet
        self.assertLess(t.GRIP_UP - 0.013, L.CHAMBER_H)
        bottom = t.HOVER_Z - t.GRIP_UP
        self.assertGreaterEqual(bottom - (L.SHEET_TOP + L.CHAMBER_H), 0.020)  # over a standing chamber
        self.assertGreaterEqual(bottom - (L.SHEET_TOP + L.EXIST_H + L.COVER_H), 0.019)  # over the existing one
        for wps in t.demo().values():  # every open/close takes 1.5 s; each is preceded by a settle
            for i, w in enumerate(wps):
                if i and w.finger != wps[i - 1].finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
                    self.assertEqual(wps[i - 1].label, "settle", w.label)

    def test_open_fingers_clear_neighbours(self):
        """At each pick the open fingers (yaw 0: they close along y, 10 mm thick, 20 mm wide) stay 15 mm or more
        from every chamber standing at that moment: the right arm picks chamber_2 with chamber_1 already at H; the
        left arm picks chamber_4 with chamber_3 at B1 and chamber_2 at P (it waits for the right arm)."""
        half = -self.task.OPEN + 0.010  # the fingers' outer faces from the axis, along y
        standing_at_pick = {
            "chamber_1": [L.START[n] for n in ("chamber_2", "chamber_3", "chamber_4")],
            "chamber_3": [L.START[n] for n in ("chamber_1", "chamber_2", "chamber_4")],
            "chamber_2": [H, L.START["chamber_4"], B1],
            "chamber_4": [H, B1, P],
        }
        for name, others in standing_at_pick.items():
            x, y = L.START[name]
            for sy in (-1, 1):
                for dx in (-0.010, 0.010):
                    for o in others:
                        gap = math.dist((x + dx, y + sy * half), o) - L.CHAMBER_RADIUS
                        self.assertGreater(gap, 0.015, (name, o))
            self.assertGreater(math.dist((x, y), L.EXIST) - L.EXIST_RADIUS - half, 0.015)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        for words in ("head of the run", "100 mm pipe", "150 mm pipe", "existing chamber at the boundary",
                      "Approved Document H", "upright", "let go"):  # the plan's features and the act
            self.assertIn(words, self.task.INSTRUCTION)
        for leak in ("bend", "corner", "size change", "change of pipe size", "change of size", "turn"):
            self.assertNotIn(leak, self.task.INSTRUCTION)  # where the access points go is not given away

    def test_task_json_states_the_thresholds(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("15 mm", "10 degrees", "3 mm", "2 mm", "0.5 s", "1 s", "2 s", "10 mm above its top", "95 mm",
                  "80 mm", "0.25 x n", "0.75", "x 37.0, y -26.0 cm", "x 37.0, y -11.0 cm", "x 37.0, y +5.0 cm",
                  "x 23.0, y +5.0 cm", "25, 40 and 75 mm"):
            self.assertIn(s, text)
        setup = spec["setup"]
        for s in ("x 17.0 to 42.0 cm, y -31.0 to +35.0 cm", "60 mm across and 40 mm tall, 100 g",
                  "x 23.0, y -25.0 cm", "x 31.0, y -19.0 cm", "x 34.0, y +13.0 cm", "x 35.0, y +26.0 cm",
                  "70 mm across and 40 mm tall centred at x 23.0, y +30.0 cm", "10 mm wide", "20 mm",
                  "150, 160, 140 and 250 mm"):
            self.assertIn(s, setup)
        self.assertEqual(spec["category"], "Impact on jobs")
        self.assertEqual(spec["run_limit"], "1,200 steps (1 minute at 20 Hz)")
        self.assertEqual(spec["scores"], "TBD after the GPU runs (offline kinematic check predicts demo: 1.0)")
        for text in (json.dumps(spec), (HERE / "README.md").read_text()):
            self.assertNotIn(chr(0x2014), text)
            self.assertNotIn(chr(0x2013), text)

    def test_success_hold_outlasts_every_release(self):
        """Should the arms be unreadable, a chamber would count while still gripped once settled, so the success
        hold must be longer than the release after the last settle: the trial cannot end with fingers closed."""
        plan = self.task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly stand it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    self.assertLess(after[1].duration, self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 4)


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Gripping straight down at any heading, neither arm reaches the other side's starting chambers."""

    def test_other_sides_chambers_are_out_of_reach(self):
        sys.path.insert(0, str(_kin_dir()))
        import yamkin as K
        z = L.SHEET_TOP + 0.024
        for arm, names in (("left", ("chamber_1", "chamber_2")), ("right", ("chamber_3", "chamber_4"))):
            base = K.BASES["station"][arm]
            for name in names:
                x, y = L.START[name]
                for yaw in np.linspace(-math.pi, math.pi, 13):
                    self.assertFalse(K.reachable((x, y, z), float(yaw), base)[0], (arm, name, yaw))
        for arm, names in (("left", ("chamber_3", "chamber_4")), ("right", ("chamber_1", "chamber_2"))):
            base = K.BASES["station"][arm]
            for name in names:
                x, y = L.START[name]
                self.assertTrue(K.reachable((x, y, z), 0.0, base)[0], (arm, name))
        # The left arm cannot reach H or P either (so a lone left arm sets at most B1 and B2).
        base = K.BASES["station"]["left"]
        for x, y in (H, P):
            for yaw in np.linspace(-math.pi, math.pi, 13):
                self.assertFalse(K.reachable((x, y, z), float(yaw), base)[0])


if __name__ == "__main__":
    unittest.main()
