"""Offline tests for scoring.py (and task.py's hooks): synthetic object states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
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
    Tracker, bottom_gap, edge_clearances, evaluate, footprint, held, highest, lowest, rotation, tilt_deg, valid,
    view,
)

MM = 0.001
W1, W2 = L.WHITE_UNITS


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


def state(name, center, quat=(0, 0, 0, 1), vel=(0, 0, 0), spin=(0, 0, 0)):
    """A unit's state, its world bounding box worked out from its shape and pose."""
    c = np.array(center, float)
    r = rotation(quat)
    if L.KIND[name] == "cube":
        ext = np.abs(r).sum(axis=1) * L.SIDE / 2
    else:
        a = r[:, 2]
        ext = np.array([L.MAST_H / 2 * abs(a[k]) + L.MAST_RADIUS * math.sqrt(max(0.0, 1 - a[k] ** 2))
                        for k in range(3)])
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def unit(name, x, y, bottom=L.DECK_TOP, tilt=0.0, axis="x", yaw=0.0, **kw):
    """A unit with its footprint centre at (x, y) (upright) and its lowest point at ``bottom``, tilted ``tilt``
    degrees about ``axis`` after a turn of ``yaw`` degrees about z."""
    q = qmul(about(axis, tilt), about("z", yaw))
    probe = state(name, (x, y, 0.0), q)
    return state(name, (x, y, bottom - lowest(probe, L.KIND[name])), q, **kw)


def layout(**at):
    """Every unit at its start, except those given: name -> (x, y) on the deck, or a ready state."""
    out = []
    for n in L.NAMES:
        v = at.get(n)
        if v is None:
            out.append(unit(n, *L.START[n], bottom=L.TOP))
        elif isinstance(v, SimpleNamespace):
            out.append(v)
        else:
            out.append(unit(n, *v))
    return out


S0 = layout()
F_AT, A_AT = (L.CL_X, 0.280), (L.CL_X, -0.180)
RED_AT, GREEN_AT = (0.300, -0.080), (0.400, -0.080)
DONE = layout(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=RED_AT, sidelight_green=GREEN_AT)


def score_of(us, arms=None):
    return evaluate(us, arms)[0]


def glide(a, b, n):
    """n frames moving every unit of frame a to its place in frame b in equal steps (orientations of b)."""
    return [[state(name, (1 - i / n) * sa.pos + i / n * sb.pos, sb.quat_xyzw)
             for name, sa, sb in zip(L.NAMES, a, b, strict=True)] for i in range(1, n + 1)]


def build(n=20, final=None, hold=30):
    final = DONE if final is None else final
    return [S0] + glide(S0, final, n) + [final] * hold


def run(frames, hz=20.0, arms=None):
    t = Tracker(hz)
    r = None
    for f in frames:
        r = t.update(f, arms)
    return r, t


ON = 0.020 / L.FULL_GAP  # rigstate's opening with the fingers closed on a 20 mm unit


def snap(left=None, right=None, left_open=1.0, right_open=1.0):
    out = {}
    for side, tips, op in (("left", left, left_open), ("right", right, right_open)):
        if tips is not None:
            t = np.array(tips, float)
            out[side] = SimpleNamespace(tcp=t + (0.0, 0.0, 0.013), tips=t, opening=op)
    return out


PARKED = snap(left=(0.200, 0.250, L.TOP + 0.107), right=(0.200, -0.250, L.TOP + 0.107))
GRIP_TIPS_Z = L.DECK_TOP + 0.005  # the demo's pad tips while it sets a unit down


class Geometry(unittest.TestCase):
    def test_scale_and_rule_arithmetic(self):
        self.assertAlmostEqual(L.SHIP_L, 0.800, places=9)
        self.assertAlmostEqual(L.BEAM, 0.130, places=9)
        self.assertAlmostEqual(L.F_MIN_Y, 0.200, places=9)  # L/4 = 20 m from the stem
        self.assertAlmostEqual(L.MIN_SEPARATION, 0.400, places=9)  # L/2 = 40 m, under the 100 m cap
        self.assertAlmostEqual(L.SIDE_BAND, 1.0 / 100, places=9)  # min(10 % of 13 m, 1 m) = 1 m
        self.assertAlmostEqual(L.PORT_X, 0.285, places=9)
        self.assertAlmostEqual(L.STBD_X, 0.415, places=9)
        self.assertAlmostEqual(L.SHOULDER_Y, 0.240, places=9)  # the last 16 m taper to the stem
        self.assertGreaterEqual(L.SIDE_BAND / 2, 0.005)  # 5 mm either side of the band's middle
        # the forward limit for a 20 mm unit on the centreline: the taper's edges are 10 mm from its axis (measured
        # square to each edge) at y 0.3734
        self.assertTrue(view(unit(W1, L.CL_X, 0.3730), W1).on_deck)
        self.assertFalse(view(unit(W1, L.CL_X, 0.3740), W1).on_deck)
        self.assertAlmostEqual(L.half_width(0.0), L.BEAM / 2)
        self.assertAlmostEqual(L.half_width(L.STEM_Y), 0.0)

    def test_starboard_is_plus_x(self):
        fwd, up = np.array([0, 1, 0]), np.array([0, 0, 1])
        self.assertTrue(np.allclose(np.cross(fwd, up), [1, 0, 0]))  # facing forward, the right hand is +x

    def test_bow_steps_follow_the_taper(self):
        for y0, y1, hw in L.bow_steps():
            self.assertLessEqual(abs(hw - L.half_width(y0)), 0.0023)
            self.assertLessEqual(abs(hw - L.half_width(y1)), 0.0023)

    def test_lowest_point_and_tilt(self):
        for n in L.NAMES:
            u = unit(n, 0.35, 0.0)
            self.assertAlmostEqual(lowest(u, L.KIND[n]), L.DECK_TOP)
            self.assertAlmostEqual(highest(u, L.KIND[n]), L.DECK_TOP + 0.020)
            self.assertAlmostEqual(tilt_deg(u, L.KIND[n]), 0.0)
            t = unit(n, 0.35, 0.0, tilt=8.0)
            self.assertAlmostEqual(tilt_deg(t, L.KIND[n]), 8.0, places=4)
            self.assertAlmostEqual(bottom_gap(t, L.KIND[n]), 0.0)
        # a cube resting on any face is upright; a cylinder on its side is not
        for axis in ("x", "y"):
            self.assertAlmostEqual(tilt_deg(unit(L.RED, 0.32, 0, tilt=90.0, axis=axis), "cube"), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(unit(W1, 0.35, 0, tilt=90.0), "cyl"), 90.0, places=4)
        self.assertAlmostEqual(tilt_deg(unit(W1, 0.35, 0, tilt=180.0), "cyl"), 0.0, places=4)  # either end up

    def test_footprint_and_edges(self):
        u = unit(L.RED, *RED_AT)
        pts, r, c = footprint(u, "cube")
        self.assertEqual((pts.shape, r), ((4, 2), 0.0))
        self.assertTrue(np.allclose(c, RED_AT))
        cl = edge_clearances(pts, r)
        self.assertAlmostEqual(min(cl[i] for i in L.PORT_EDGES), RED_AT[0] - 0.010 - L.PORT_X)
        v = view(unit(W1, *F_AT), W1)
        self.assertTrue(v.on_deck)
        self.assertFalse(view(unit(W1, L.CL_X, 0.380), W1).on_deck)  # too far forward for the taper
        self.assertFalse(view(unit(L.GREEN, L.STBD_X - 0.0095, 0.0), L.GREEN).on_deck)  # 0.5 mm overhang
        self.assertTrue(view(unit(L.GREEN, L.STBD_X - 0.0100, 0.0), L.GREEN).on_deck)  # flush with the side
        self.assertFalse(view(unit(W1, L.CL_X, L.TRANSOM_Y + 0.009), W1).on_deck)  # over the transom

    def test_valid(self):
        self.assertTrue(valid(DONE[0]))
        bad = state(W1, (np.nan, 0, 0.77))
        self.assertFalse(valid(bad))
        z = SimpleNamespace(**{**vars(DONE[0]), "quat_xyzw": np.zeros(4)})
        self.assertFalse(valid(z))
        self.assertFalse(valid(SimpleNamespace(pos=np.zeros(3))))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        score, d = evaluate(S0)
        self.assertEqual((score, d["F"], d["A"]), (0.0, None, None))

    def test_demo_layout_scores_one(self):
        score, d = evaluate(DONE, PARKED)
        self.assertEqual(score, 1.0)
        self.assertEqual((d["F"], d["A"]), (W1, W2))
        self.assertEqual(d["separation_mm"], 460.0)
        self.assertAlmostEqual(d["F_from_stem_mm"], 120.0)
        # the white units are interchangeable: whichever stands nearer the bow is F
        swapped = layout(masthead_1=A_AT, masthead_2=F_AT, sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        self.assertEqual(evaluate(swapped)[1]["F"], W2)
        self.assertEqual(score_of(swapped), 1.0)

    def test_spec_shortcuts(self):
        # 1. small-craft habit: sidelights at the bow ahead of F, no second masthead light
        bow = layout(masthead_1=F_AT, sidelight_red=(L.CL_X - 0.012, 0.320), sidelight_green=(L.CL_X + 0.012, 0.320))
        self.assertEqual(score_of(bow), 0.25)
        # 2. mastheads 30 m (0.300 m) apart, sidelights right
        self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=(L.CL_X, -0.020), sidelight_red=RED_AT,
                                         sidelight_green=GREEN_AT)), 0.75)
        # 3. one masthead light at midships, the other left on the table, sidelights abaft it
        self.assertEqual(score_of(layout(masthead_1=(L.CL_X, 0.0), sidelight_red=RED_AT,
                                         sidelight_green=GREEN_AT)), 0.5)
        # 4. colours swapped
        self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=GREEN_AT,
                                         sidelight_green=RED_AT)), 0.5)
        # 5. sidelights midway between the centreline and the side
        self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=(0.3175, -0.080),
                                         sidelight_green=(0.3825, -0.080))), 0.5)
        # 6. a white unit stacked on the other, or lying on its side: not standing
        stacked = layout(masthead_1=unit(W1, *F_AT, bottom=L.DECK_TOP + L.MAST_H), masthead_2=F_AT,
                         sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        self.assertEqual(score_of(stacked), 0.75)  # the lower one is F (correct), no A
        lying = layout(masthead_1=F_AT, masthead_2=unit(W2, *A_AT, tilt=90.0), sidelight_red=RED_AT,
                       sidelight_green=GREEN_AT)
        self.assertEqual(score_of(lying), 0.75)

    def test_every_outcome_listed_in_task_json(self):
        rest = dict(sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        cases = [
            (dict(masthead_1=F_AT, masthead_2=A_AT, **rest), 1.0),
            (dict(masthead_1=F_AT, sidelight_red=(L.CL_X - 0.012, 0.320), sidelight_green=(L.CL_X + 0.012, 0.320)),
             0.25),
            (dict(masthead_1=F_AT, masthead_2=(L.CL_X, -0.020), **rest), 0.75),
            (dict(masthead_1=(L.CL_X, L.STEM_Y - 0.225), masthead_2=(L.CL_X, L.STEM_Y - 0.225 - 0.410), **rest),
             0.75),
            (dict(masthead_1=(L.CL_X, 0.0), **rest), 0.5),
            (dict(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=GREEN_AT, sidelight_green=RED_AT), 0.5),
            (dict(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=(0.3175, -0.080),
                  sidelight_green=(0.3825, -0.080)), 0.5),
            (dict(masthead_1=(L.CL_X - 0.010, 0.280), masthead_2=A_AT, **rest), 0.0),
            ({}, 0.0),
        ]
        for at, want in cases:
            self.assertEqual(score_of(layout(**at)), want, at)

    def test_centreline_band(self):
        rest = dict(masthead_2=A_AT, sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        for dx, want in ((0.0059, 1.0), (-0.0059, 1.0), (0.0061, 0.0), (-0.0061, 0.0)):
            self.assertEqual(score_of(layout(masthead_1=(L.CL_X + dx, 0.280), **rest)), want, dx)
        for dx, want in ((0.0059, 1.0), (-0.0061, 0.75)):
            self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=(L.CL_X + dx, -0.180),
                                             sidelight_red=RED_AT, sidelight_green=GREEN_AT)), want, dx)

    def test_forward_quarter_band(self):
        rest = dict(sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        y = L.F_MIN_Y
        self.assertEqual(score_of(layout(masthead_1=(L.CL_X, y + 0.0002), masthead_2=(L.CL_X, -0.210), **rest)), 1.0)
        self.assertEqual(score_of(layout(masthead_1=(L.CL_X, y - 0.0002), masthead_2=(L.CL_X, -0.210), **rest)),
                         0.75)
        self.assertEqual(score_of(layout(masthead_1=(L.CL_X, 0.373), masthead_2=(L.CL_X, -0.210), **rest)), 1.0)

    def test_separation_band(self):
        rest = dict(sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        for gap, want in ((L.MIN_SEPARATION + 0.0002, 1.0), (L.MIN_SEPARATION - 0.0002, 0.75)):
            self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=(L.CL_X, 0.280 - gap), **rest)), want)

    def test_side_band(self):
        base = dict(masthead_1=F_AT, masthead_2=A_AT)
        h = L.SIDE / 2
        for inboard, want in ((0.0, 1.0), (0.0099, 1.0), (0.0101, 0.75), (-0.0005, 0.75)):
            red = (L.PORT_X + inboard + h, -0.080)
            self.assertEqual(score_of(layout(**base, sidelight_red=red, sidelight_green=GREEN_AT)), want, inboard)
            green = (L.STBD_X - inboard - h, -0.080)
            self.assertEqual(score_of(layout(**base, sidelight_red=RED_AT, sidelight_green=green)), want, inboard)

    def test_a_turned_cube_is_measured_by_its_footprint(self):
        """A cube turned 20 degrees: its corner, not its face, is the outboard point."""
        base = dict(masthead_1=F_AT, masthead_2=A_AT, sidelight_green=GREEN_AT)
        reach = L.SIDE / 2 * (math.cos(math.radians(20)) + math.sin(math.radians(20)))
        x_ok = L.PORT_X + reach + 0.003
        self.assertEqual(score_of(layout(**base, sidelight_red=unit(L.RED, x_ok, -0.08, yaw=20.0))), 1.0)
        self.assertEqual(score_of(layout(**base, sidelight_red=unit(L.RED, x_ok + 0.008, -0.08, yaw=20.0))), 0.75)
        # the demo's yaw error (+0.6..+1.7 deg) changes the outboard reach by under 0.3 mm
        e = L.SIDE / 2 * (math.cos(math.radians(1.7)) + math.sin(math.radians(1.7)) - 1)
        self.assertLess(e, 0.0003)

    def test_not_ahead_of_f(self):
        base = dict(masthead_1=F_AT, masthead_2=A_AT, sidelight_green=GREEN_AT)
        # just aft of F counts, 1 mm ahead of it does not (here the cube's forward corner is past the shoulder)
        r0 = layout(**base, sidelight_red=(L.PORT_X + 0.013, 0.235))
        self.assertEqual(score_of(r0), 1.0)
        r1 = layout(masthead_1=(L.CL_X, 0.234), masthead_2=(L.CL_X, -0.180), sidelight_green=GREEN_AT,
                    sidelight_red=(L.PORT_X + 0.013, 0.235))
        self.assertEqual(score_of(r1), 0.75)

    def test_gate(self):
        """A, red and green are judged only when F stands on the centreline, right or wrong."""
        rest = dict(masthead_2=A_AT, sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        off = layout(masthead_1=(L.CL_X + 0.012, 0.280), **rest)
        score, d = evaluate(off)
        self.assertEqual((score, d["gate"], d["F"]), (0.0, False, W1))
        # F on the centreline but too far aft: F earns nothing, the rest are judged
        aft = layout(masthead_1=(L.CL_X, 0.120), masthead_2=(L.CL_X, -0.290), sidelight_red=RED_AT,
                     sidelight_green=GREEN_AT)
        score, d = evaluate(aft)
        self.assertEqual((score, d["gate"], d["F_ok"]), (0.75, True, False))
        # a white unit off the deck is never F
        table = layout(masthead_1=(L.CL_X, 0.280), masthead_2=unit(W2, 0.26, 0.250, bottom=L.TOP),
                       sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        self.assertEqual(evaluate(table)[1]["F"], W1)

    def test_every_score_turns_up(self):
        seen = set()
        for f in (F_AT, (L.CL_X, 0.100)):
            for a in (A_AT, (L.CL_X, 0.0), None):
                for red in (RED_AT, GREEN_AT):
                    for green in (GREEN_AT, (0.37, -0.08)):
                        at = dict(masthead_1=f, sidelight_red=red, sidelight_green=green)
                        if a:
                            at["masthead_2"] = a
                        seen.add(score_of(layout(**at)))
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_success_needs_stillness(self):
        r, t = run(build(hold=30), arms=PARKED)
        self.assertEqual((r.score, r.success), (1.0, True))
        r, t = run(build(hold=19), arms=PARKED)  # 0.95 s at rest: not yet still for 1 s
        self.assertEqual((r.score, r.success), (0.0, False))
        r, t = run(build(hold=21), arms=PARKED)
        self.assertEqual(r.score, 1.0)
        # a sidelight creeping 3.2 mm per second (1.6 mm per 0.5 s) is not still
        frames = build(hold=10)
        for i in range(30):
            g = unit(L.GREEN, GREEN_AT[0], GREEN_AT[1] + 0.0032 * i / 20)
            frames.append(DONE[:3] + [g])
        self.assertEqual(run(frames)[0].score, 0.75)
        # ... one creeping 2.4 mm per second is
        frames = build(hold=10)
        for i in range(30):
            frames.append(DONE[:3] + [unit(L.GREEN, GREEN_AT[0], GREEN_AT[1] - 0.0024 * i / 20)])
        self.assertEqual(run(frames)[0].score, 1.0)

    def test_score_is_the_state_at_hand(self):
        r, t = run(build(hold=30))
        self.assertEqual(r.score, 1.0)
        later = layout(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=RED_AT)
        r = None
        for _ in range(30):
            r = t.update(later)
        self.assertEqual((r.score, t.best), (0.75, 1.0))


class ZeroConditions(unittest.TestCase):
    def test_nothing_moved_or_all_on_the_table(self):
        self.assertEqual(score_of(S0), 0.0)
        table = [unit(n, 0.25, 0.1 * k - 0.15, bottom=L.TOP) for k, n in enumerate(L.NAMES)]
        self.assertEqual(score_of(table), 0.0)

    def test_no_white_unit_on_the_centreline(self):
        at = dict(masthead_1=(0.335, 0.280), masthead_2=(0.365, -0.18), sidelight_red=RED_AT,
                  sidelight_green=GREEN_AT)
        self.assertEqual(score_of(layout(**at)), 0.0)

    def test_tilted_or_floating(self):
        rest = dict(masthead_2=A_AT, sidelight_red=RED_AT, sidelight_green=GREEN_AT)
        self.assertEqual(score_of(layout(masthead_1=unit(W1, *F_AT, tilt=9.5), **rest)), 1.0)
        self.assertEqual(score_of(layout(masthead_1=unit(W1, *F_AT, tilt=10.5), **rest)), 0.0)
        self.assertEqual(score_of(layout(masthead_1=unit(W1, *F_AT, bottom=L.DECK_TOP + 0.0029), **rest)), 1.0)
        self.assertEqual(score_of(layout(masthead_1=unit(W1, *F_AT, bottom=L.DECK_TOP + 0.0031), **rest)), 0.0)
        self.assertEqual(score_of(layout(masthead_1=unit(W1, *F_AT, bottom=L.DECK_TOP - 0.0031), **rest)), 0.0)

    def test_nan_scores_zero_and_keeps_the_tracker(self):
        r, t = run(build(hold=30))
        self.assertEqual(r.score, 1.0)
        n_hist = len(t.history)
        for bad in (state(W1, (np.nan, 0.19, 0.77)), state(W1, (np.inf, 0.19, 0.77)),
                    SimpleNamespace(**{**vars(DONE[0]), "quat_xyzw": np.zeros(4)}),
                    SimpleNamespace(**{**vars(DONE[0]), "lin_vel": np.array([np.nan, 0, 0])}),
                    SimpleNamespace(pos=np.array([0.35, 0.19, 0.77]))):
            r = t.update([bad] + DONE[1:])
            self.assertEqual((r.score, r.success, r.details["valid_state"]), (0.0, False, False))
            self.assertEqual(r.details["bad_units"], [W1])
        self.assertEqual(len(t.history), n_hist)
        self.assertEqual(t.bad_steps, 5)
        self.assertEqual(t.update(DONE).score, 1.0)  # back to a good state: the paths kept still
        self.assertEqual(t.update(DONE[:3]).score, 0.0)  # a unit missing


class LetGo(unittest.TestCase):
    def test_reach_numbers(self):
        self.assertAlmostEqual(L.release_reach(1.0, W1), 0.0754, places=4)
        self.assertAlmostEqual(L.release_reach(1.0, L.RED), 0.0795, places=4)
        self.assertAlmostEqual(L.release_reach(ON, W1), 0.0433, places=4)
        self.assertAlmostEqual(L.release_reach(ON, L.RED), 0.0474, places=4)

    def test_parked_arms_let_every_unit_go(self):
        self.assertEqual(score_of(DONE, PARKED), 1.0)
        self.assertEqual(score_of(DONE, {}), 1.0)

    def test_a_unit_still_gripped_does_not_count(self):
        grip = snap(left=(*F_AT, GRIP_TIPS_Z), left_open=ON, right=PARKED["right"].tips)
        self.assertTrue(held(DONE[0], W1, grip))
        score, d = evaluate(DONE, grip)
        self.assertEqual((score, d["F"]), (0.0, W2))  # A becomes the only standing white unit, aft: 0
        grip = snap(left=PARKED["left"].tips, right=(*GREEN_AT, GRIP_TIPS_Z), right_open=ON)
        self.assertEqual(score_of(DONE, grip), 0.75)
        # fully open round the red cube, the pad tips low: its fingers could touch red (green is 100 mm and A 112 mm
        # away, beyond the 79.5 / 75.4 mm reach)
        open_round = snap(right=(*RED_AT, GRIP_TIPS_Z))
        score, d = evaluate(DONE, open_round)
        self.assertEqual(score, 0.75)
        self.assertEqual([n for n in L.NAMES if d["units"][n]["held"]], [L.RED])
        # ... and one fully open low between the red cube and A, within reach of both: 0.5
        between = snap(right=(0.325, -0.130, GRIP_TIPS_Z))
        score, d = evaluate(DONE, between)
        self.assertEqual(score, 0.5)
        self.assertEqual([n for n in L.NAMES if d["units"][n]["held"]], [W2, L.RED])

    def test_reach_and_height_edges(self):
        x, y = RED_AT
        top = highest(DONE[2], "cube")
        self.assertTrue(held(DONE[2], L.RED, snap(right=(x, y, top + 0.0099))))
        self.assertFalse(held(DONE[2], L.RED, snap(right=(x, y, top + 0.0101))))
        low = L.DECK_TOP + 0.010
        self.assertTrue(held(DONE[2], L.RED, snap(right=(x + 0.0794, y, low))))
        self.assertFalse(held(DONE[2], L.RED, snap(right=(x + 0.0796, y, low))))
        self.assertTrue(held(DONE[2], L.RED, snap(right=(x + 0.0473, y, low), right_open=ON)))
        self.assertFalse(held(DONE[2], L.RED, snap(right=(x + 0.0475, y, low), right_open=ON)))
        # the demo's gripper rising away after the green cube (pad tips 27 mm over its top)
        self.assertEqual(score_of(DONE, snap(right=(*GREEN_AT, L.TOP + 0.057))), 1.0)

    def test_unreadable_arms_are_skipped_and_counted(self):
        frames = build(hold=25)
        r, t = run(frames, arms=None)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.arms_unavailable, len(frames))
        right = SimpleNamespace(tips=np.array((*GREEN_AT, GRIP_TIPS_Z)), opening=ON)
        r, t = run(frames, arms={"left": SimpleNamespace(tips=np.full(3, np.nan), opening=1.0), "right": right})
        self.assertEqual(r.score, 0.75)  # the readable right arm still holds green
        self.assertEqual(r.details["arms_readable"], ["right"])
        bad_open = {"right": SimpleNamespace(tips=np.array((*GREEN_AT, GRIP_TIPS_Z)), opening=float("nan"))}
        # an unreadable opening is taken as fully open (the widest reach, 79.5 mm): green counts as held, red
        # (100 mm away) does not
        self.assertEqual(score_of(DONE, bad_open), 0.75)
        wide = {"right": SimpleNamespace(tips=np.array((0.300, -0.150, GRIP_TIPS_Z)), opening=float("nan"))}
        self.assertEqual(score_of(DONE, wide), 0.5)  # red (70 mm) and A (58 mm) both within the widest reach

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
        self.assertEqual((t.steps, t.best, t.first_at, len(t.history)), (1, 0.0, {}, 1))
        self.assertEqual(t.update(S0).score, 0.0)

    def test_gradual_return_is_not_a_reset(self):
        r, t = run(build(hold=25) + glide(DONE, S0, 40))
        self.assertEqual(t.best, 1.0)
        self.assertGreater(t.steps, 40)

    def test_jump_elsewhere_is_not_a_reset(self):
        r, t = run(build(hold=25))
        t.update(layout(masthead_1=(L.CL_X, 0.0)))
        self.assertEqual(t.best, 1.0)

    def test_jittered_reset_counts(self):
        r, t = run(build(hold=25))
        t.update([unit(n, L.START[n][0] + 0.02, L.START[n][1] - 0.02, bottom=L.TOP) for n in L.NAMES])
        self.assertEqual(t.steps, 1)


class Exploits(unittest.TestCase):
    def test_cubes_as_masthead_lights(self):
        """The coloured cubes on the centreline and the white units at the sides: the white units are the
        mastheads by kind, so no masthead light stands on the centreline: 0."""
        at = dict(sidelight_red=F_AT, sidelight_green=A_AT, masthead_1=(0.300, -0.08), masthead_2=(0.400, -0.08))
        self.assertEqual(score_of(layout(**at)), 0.0)

    def test_both_white_units_together_forward(self):
        at = dict(masthead_1=(L.CL_X, 0.300), masthead_2=(L.CL_X, 0.270), sidelight_red=RED_AT,
                  sidelight_green=GREEN_AT)
        self.assertEqual(score_of(layout(**at)), 0.75)

    def test_a_unit_on_the_sternlight_or_overhanging(self):
        perched = unit(W2, L.STERN_X, L.STERN_Y, bottom=L.DECK_TOP + L.STERN_H)
        self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=perched, sidelight_red=RED_AT,
                                         sidelight_green=GREEN_AT)), 0.75)
        over = (L.PORT_X + 0.008, -0.08)  # 2 mm over the port side: right at the lamp, but not on the deck
        self.assertEqual(score_of(layout(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=over,
                                         sidelight_green=GREEN_AT)), 0.75)

    def test_sidelights_on_the_table_beside_the_hull(self):
        at = dict(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=unit(L.RED, L.PORT_X - 0.011, -0.08, bottom=L.TOP),
                  sidelight_green=unit(L.GREEN, L.STBD_X + 0.011, -0.08, bottom=L.TOP))
        self.assertEqual(score_of(layout(**at)), 0.5)

    def test_sidelight_on_the_wrong_half_near_its_own_colours_side_band(self):
        """A red cube at the starboard side is never 'at the port side' even though both sides are edges."""
        score, d = evaluate(layout(masthead_1=F_AT, masthead_2=A_AT, sidelight_red=GREEN_AT,
                                   sidelight_green=GREEN_AT[:1] + (-0.120,)))
        self.assertEqual((d["red_ok"], d["green_ok"]), (False, True))

    def test_random_states_score_only_listed_values(self):
        rng = random.Random(7)
        for _ in range(400):
            at = {}
            for n in L.NAMES:
                if rng.random() < 0.8:
                    at[n] = unit(n, rng.uniform(0.27, 0.43), rng.uniform(-0.40, 0.40),
                                 bottom=L.DECK_TOP + rng.choice((0.0, 0.0, 0.002, 0.005)),
                                 tilt=rng.choice((0.0, 0.0, 5.0, 15.0, 90.0)), yaw=rng.uniform(-45, 45))
            s, d = evaluate(layout(**at))
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            if not d["gate"]:
                self.assertFalse(d["A_ok"] or d["red_ok"] or d["green_ok"])


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if d.is_dir():
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

    def objects(self, units):
        return dict(zip(self.task.UNIT_NAMES, units, strict=True))

    def test_objects_match_the_layout(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual((t.SCENE, t.SCORE_OVER), ("station", "end"))
        self.assertEqual(t.UNIT_NAMES, L.NAMES)
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], list(L.NAMES))
        for spec, s in zip(t.UNITS, S0, strict=True):
            self.assertTrue(np.allclose(spec.pos, s.pos))
            self.assertLessEqual(spec.mass, 0.3)
        # the deck boxes: flush tops at the deck top, inside the outline, and the outline's area covered
        area = 0.0
        for b in t.DECK:
            self.assertAlmostEqual(b.pos[2] + b.size[2] / 2, L.DECK_TOP)
            self.assertAlmostEqual(b.pos[2] - b.size[2] / 2, L.TOP)
            x0, x1 = b.pos[0] - b.size[0] / 2, b.pos[0] + b.size[0] / 2
            y0, y1 = b.pos[1] - b.size[1] / 2, b.pos[1] + b.size[1] / 2
            self.assertGreaterEqual(x0, L.PORT_X - 1e-9)
            self.assertLessEqual(x1, L.STBD_X + 1e-9)
            self.assertGreaterEqual(y0, L.TRANSOM_Y - 1e-9)
            self.assertLessEqual(y1, L.STEM_Y + 1e-9)
            area += b.size[0] * b.size[1]
        true_area = L.BEAM * (L.SHOULDER_Y - L.TRANSOM_Y) + L.BEAM / 2 * (L.STEM_Y - L.SHOULDER_Y)
        self.assertAlmostEqual(area, true_area, delta=1e-7)  # the steps' widths are the taper's at their middles
        self.assertAlmostEqual(t.STERNLIGHT.pos[2] - L.STERN_H / 2, L.DECK_TOP)
        # Nothing where the arms' start pose hides the table in the setup screenshot (|y| > 0.45 m, x < 0.2 m).
        for o in t.OBJECTS:
            half_y = o.size[1] / 2 if hasattr(o, "size") else o.radius
            half_x = o.size[0] / 2 if hasattr(o, "size") else o.radius
            self.assertLess(abs(o.pos[1]) + half_y, 0.45, o.name)
            self.assertGreater(o.pos[0] - half_x, 0.2, o.name)

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
        for key in ("episode", "step", "score", "success", "F", "A", "gate", "F_ok", "A_ok", "red_ok", "green_ok",
                    "F_from_stem_mm", "separation_mm", "units", "paths_mm", "bad_steps", "first_step",
                    "arms_readable", "arms_unavailable"):
            self.assertIn(key, rows[-1])
        u = rows[-1]["units"][L.RED]
        for key in ("x_mm", "y_mm", "tilt_deg", "bottom_mm", "on_deck", "port_gap_mm", "held", "still", "stands",
                    "off_centreline_mm"):
            self.assertIn(key, u)
        self.assertEqual(u["port_gap_mm"], 5.0)
        self.assertEqual(rows[-1]["separation_mm"], 460.0)
        self.assertIsNone(rows[0]["paths_mm"])
        self.assertIn("rigstate_error", rows[-1])

    def test_hooks_read_the_arms(self):
        def feed(right_tips, right_open):
            rigstate.offline_feed({"left": {"tcp": PARKED["left"].tcp, "tips": PARKED["left"].tips, "opening": 1.0},
                                   "right": {"tcp": np.add(right_tips, (0, 0, 0.013)), "tips": right_tips,
                                             "opening": right_open}})
        rigstate.offline_reset()
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                for _ in range(25):
                    feed((*GREEN_AT, GRIP_TIPS_Z), ON)
                    self.assertFalse(self.task.success(self.objects(DONE)))
                self.assertEqual(self.task.score(self.objects(DONE)), 0.75)
                feed(tuple(PARKED["right"].tips), 1.0)
                self.assertTrue(self.task.success(self.objects(DONE)))
            finally:
                os.environ.pop("TASK_TRACE", None)
                rigstate.offline_reset()
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertTrue(rows[0]["units"][L.GREEN]["held"])
        self.assertAlmostEqual(rows[0]["right_opening"], ON)
        self.assertFalse(rows[-1]["units"][L.GREEN]["held"])
        self.assertEqual(rows[-1]["arms_readable"], ["left", "right"])

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
                self.assertFalse(self.task.success(self.objects([state(W1, (np.nan, 0, 0.77)), *S0[1:]])))
                self.assertEqual(self.task.score(self.objects([state(W1, (np.nan, 0, 0.77)), *S0[1:]])), 0.0)
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

    def test_demo_targets_score_one_with_margins(self):
        t = self.task
        final = list(S0)
        for arm, steps in t.PLAN.items():
            for name, target, _ in steps:
                final[L.NAMES.index(name)] = unit(name, *target)
        self.assertEqual(score_of(final), 1.0)
        self.assertGreaterEqual(t.F_AT[1] - L.F_MIN_Y, 0.050)  # 80 mm inside the forward quarter
        self.assertGreaterEqual(t.F_AT[1] - t.A_AT[1] - L.MIN_SEPARATION, 0.050)  # 60 mm over half a length
        self.assertAlmostEqual(t.RED_AT[0] - L.SIDE / 2 - L.PORT_X, L.SIDE_BAND / 2)  # mid band
        self.assertAlmostEqual(L.STBD_X - t.GREEN_AT[0] - L.SIDE / 2, L.SIDE_BAND / 2)
        # each unit lands 4 mm short of its place along y (the last 40 mm run along y only): still 1.0
        short = list(final)
        for arm, steps in t.PLAN.items():
            for name, target, along in steps:
                short[L.NAMES.index(name)] = unit(name, target[0], target[1] - along * 0.004)
        self.assertEqual(score_of(short), 1.0)
        # one arm's plan dropped: the scripted one-arm outcome
        for idle, want in (("left", 0.0), ("right", 0.25)):
            part = list(final)
            for name, _, _ in t.PLAN[idle]:
                part[L.NAMES.index(name)] = S0[L.NAMES.index(name)]
            self.assertEqual(score_of(part), want, idle)

    @staticmethod
    def spells(wps):
        out, t = [], 0.0
        for w in wps:
            out.append((t, t + w.duration, w.pos[0], w.pos[1]))
            t += w.duration
        return out

    def test_grippers_stay_apart_and_fit_the_run_limit(self):
        plan = self.task.demo()
        left, right = self.spells(plan["left"]), self.spells(plan["right"])
        for a0, a1, _, ya in left:
            for b0, b1, _, yb in right:
                if a0 < b1 and b0 < a1:
                    self.assertGreaterEqual(abs(ya - yb), 0.20)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        with open(HERE / "task.json") as f:
            limit = int(json.load(f)["run_limit"].split()[0].replace(",", ""))
        self.assertLess((total + 3.0) * L.CONTROL_HZ, limit * 0.6)

    def test_carried_units_clear_everything(self):
        t = self.task
        bottom = t.HOVER_Z - t.GRIP_UP
        self.assertGreaterEqual(bottom - (L.DECK_TOP + 0.020), 0.020)  # over a standing unit's top
        self.assertGreaterEqual(bottom - (L.DECK_TOP + L.STERN_H), 0.020)  # over the sternlight
        self.assertGreaterEqual(t.PICK_Z - 0.013 - L.TOP, 0.003)  # pad tips over the table
        self.assertGreaterEqual(t.PLACE_Z - 0.013 - L.DECK_TOP, 0.003)  # and over the deck
        self.assertAlmostEqual(-2 * t.GRIP, L.SIDE - 0.008)  # commanded 8 mm narrower
        self.assertAlmostEqual(t.PLACE_Z - t.GRIP_UP - L.DECK_TOP, t.DROP)
        # open fingers along y (yaw 0, 40 mm gap plus a 10 mm finger) clear a neighbour 60 mm away on the table
        ys = sorted(L.START[n][1] for n in L.NAMES)
        self.assertGreater(min(b - a for a, b in zip(ys, ys[1:])) - (-t.OPEN + 0.010) - L.SIDE / 2, 0.015)

    def test_instruction_is_the_same_everywhere(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        # the instruction withholds the answers: no distance, side or order
        for word in ("quarter", "half", "port", "starboard", "centreline", "mm", "abaft", "forward"):
            self.assertNotIn(word, self.task.INSTRUCTION.lower())

    def test_task_json_states_the_thresholds(self):
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for s in ("10 degrees", "3 mm", "10 mm above its top", "1.5 mm", "0.5 s", "6 mm", "200 mm", "400 mm",
                  "10 mm (1 m)", "2 s", "0, 0.25, 0.5, 0.75 or 1.0"):
            self.assertIn(s, text)
        setup = spec["setup"]
        for s in ("x 28.5 cm", "x 41.5 cm", "y -40.0 cm", "y +24.0 cm", "x 35.0 cm, y +40.0 cm",
                  "20 mm across and 20 mm tall, 17 g", "20 mm on each edge, 22 g", "x 23.0 cm, y +15.0 cm",
                  "y -9.0 cm", "y +3.0 cm", "y -3.0 cm", "y -38.7 cm", "12 mm across and 15 mm tall", "1:100"):
            self.assertIn(s, setup)
        self.assertEqual(round(L.MAST_MASS * 1000), 17)
        self.assertEqual(round(L.SIDE_MASS * 1000), 22)
        self.assertEqual(spec["category"], "Impact on jobs")
        for k in ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes"):
            self.assertNotIn(chr(0x2014), spec[k])
            self.assertNotIn(chr(0x2013), spec[k])

    def test_success_hold_outlasts_every_release(self):
        """Should the arms be unreadable, a unit set down counts 1 s (stillness) after its descent ends, gripped;
        with the 2 s hold the trial still cannot end before the 0.5 s settle and 1.5 s release are over."""
        still = L.STILL_WINDOW_S * L.STILL_WINDOWS
        plan = self.task.demo()
        n = 0
        for wps in plan.values():
            for i, w in enumerate(wps):
                if w.label.startswith("slowly stand it"):
                    after = wps[i + 1:i + 3]
                    self.assertEqual([a.label for a in after], ["settle", "let go of it"])
                    self.assertLess(sum(a.duration for a in after), still + self.task.SUCCESS_HOLD_S)
                    n += 1
        self.assertEqual(n, 4)


@unittest.skipIf(_kin_dir() is None, "needs the offline kinematics (tools/offline/yamkin.py)")
class OneArmReach(unittest.TestCase):
    """Straight down at any heading, the right arm cannot set a masthead light on the centreline in the forward
    quarter; the left arm can reach every place a full score needs if F goes at the very bow."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_kin_dir()))
        cls.K = importlib.import_module("yamkin")

    def ok(self, arm, x, y, z, yaws):
        base = self.K.BASES["station"][arm]
        return any(self.K.reachable((x, y, z), float(yw), base)[0] for yw in yaws)

    def test_right_arm_cannot_reach_the_forward_quarter(self):
        z = L.DECK_TOP + 0.001 + 0.017
        yaws = np.radians(np.arange(-180, 180, 30))
        for y in (L.F_MIN_Y, 0.240, 0.300, 0.370):
            self.assertFalse(self.ok("right", L.CL_X, y, z, yaws), y)
        self.assertTrue(self.ok("right", L.CL_X, 0.060, z, (0.0,)))

    def test_left_arm_alone_reaches_a_full_layout(self):
        z = L.DECK_TOP + 0.001 + 0.017
        for x, y in ((L.CL_X, 0.372), (L.CL_X, 0.372 - 0.400 - 0.003), (0.300, 0.100), (0.400, 0.100)):
            self.assertTrue(self.ok("left", x, y, z, (0.0, math.radians(-30))), (x, y))
        for n in L.NAMES:
            self.assertTrue(self.ok("left", *L.START[n], L.TOP + 0.017, (0.0, math.radians(-30))), n)
        self.assertFalse(self.ok("left", L.CL_X, -0.180, z, np.radians(np.arange(-180, 180, 30))))


if __name__ == "__main__":
    unittest.main()
