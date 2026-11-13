"""Offline tests for scoring.py, offline_clip.py and task.py: synthetic oximeter states (mechkit AssemblyStates from
the task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import re
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline stand-in

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_clip as OC  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

WIN = S.WINDOW  # 20 steps
SPEC = T.OXIMETER_SPEC
AWAY = {a: rigstate.Arm(np.array([0.16, s * 0.15, L.TOP + 0.20]), np.array([0.16, s * 0.15, L.TOP + 0.187]), 0.74)
        for a, s in (("left", 1.0), ("right", -1.0))}
P = S.P_WORLD
HOLD = L.HOLD_DEG
FULL = L.OPEN_LIMIT_DEG


# -- states --------------------------------------------------------------------------------------------------------


def _rot(axis, deg):
    a = math.radians(deg)
    return mk.axis_angle_matrix(np.asarray(axis, dtype=float), a)


def unit(origin, r=np.eye(3), q_deg=0.0):
    return mk.state_from_poses(SPEC, (*origin, *mk.matrix_to_quat(r)), joints={S.JAW: math.radians(q_deg)},
                               source="offline")


def rest():
    return unit(S.START_ORIGIN)


def dropped():
    """Lying shut on the table away from its start (not a reset)."""
    return unit(S.START_ORIGIN + np.array([0.040, 0.050, 0.0]))


def on_finger(depth, q_deg=HOLD, v=0.0, w=0.0, pitch=0.0, roll=0.0, yaw=0.0, shift=(0.0, 0.0, 0.0)):
    """The unit with its centre line along the finger (turned ``yaw`` about z, ``pitch`` about its n axis,
    ``roll`` about its own centre line), the fingertip ``depth`` mm past the lips, ``v`` mm across the shells and
    ``w`` mm along the pivot off the centre line."""
    a = math.radians(q_deg) / 2
    rc = _rot((0, 0, 1), yaw)
    rc = rc @ _rot((0, 1, 0), pitch) @ _rot((1, 0, 0), roll)
    m, n, z = rc[:, 0], rc[:, 1], rc[:, 2]
    s = L.lip_line(a) - depth
    pivot = P - (m * s + n * v + z * w) * L.MM + np.asarray(shift, dtype=float)
    r = rc @ S.rz(-a)
    origin = pivot - r @ np.array([L.PIVOT_X, 0.0, 0.0])
    return unit(origin, r, q_deg)


def in_air(q_deg=FULL, height=0.060):
    """Pinched open and lifted clear of the table, well short of the finger."""
    a = math.radians(q_deg) / 2
    r = S.rz(-a)
    return unit((L.START[0], 0.0, L.TOP + L.START[2] + height), r, q_deg)


def pinch_arms(state, gap=0.025, side="right"):
    """One gripper pinching the tail (its TCP 0.5 mm behind the back end), the other away."""
    tcp = state.to_world(np.array([-L.U0 - 0.0005, 0.0, 0.0005]))
    arm = rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), gap / L.FULL_GAP)
    out = dict(AWAY)
    out[side] = arm
    return out


def above_arms(state, zone_gap_m):
    """The right gripper (open to 70 mm) straight above the unit's centre at a height whose finger zone is
    ``zone_gap_m`` from the unit (bisected), the left one away."""
    c = np.asarray(state.pos, dtype=float)
    lo, hi = 0.0, 0.3
    for _ in range(80):
        mid = (lo + hi) / 2
        tips = c + np.array([0.0, 0.0, mid])
        arm = rigstate.Arm(tips + np.array([0.0, 0.0, 0.013]), tips, 0.070 / L.FULL_GAP)
        g = rigstate.zone_gap(arm, state.distance)
        if g < zone_gap_m:
            lo = mid
        else:
            hi = mid
    tips = c + np.array([0.0, 0.0, hi])
    out = dict(AWAY)
    out["right"] = rigstate.Arm(tips + np.array([0.0, 0.0, 0.013]), tips, 0.070 / L.FULL_GAP)
    return out


def feed(tr, states, arms=AWAY, n=1):
    """``states``: one oximeter state, or a list of them played in order (each ``n`` times)."""
    seq = states if isinstance(states, list) else [states]
    res = None
    for st in seq:
        for _ in range(n):
            res = tr.update({S.OXIMETER: st, S.MANIKIN: T.MANIKIN_SPEC.rest_state()}, arms)
    return res


def slide(tr, d0, d1, q_deg=FULL, steps=40, arms=None):
    """Carry the unit from depth d0 to d1 at a fixed pivot angle, pinched by the right gripper."""
    res = None
    for d in np.linspace(d0, d1, steps):
        st = on_finger(d, q_deg)
        res = feed(tr, st, pinch_arms(st) if arms is None else arms)
    return res


def clip_on(tr, depth=None, n=WIN + 2, **kw):
    """Let go: the cushions on the finger, both arms away, for ``n`` steps."""
    a = math.radians(HOLD) / 2
    d = L.depth_on_stop(a) if depth is None else depth
    return feed(tr, on_finger(d, HOLD, w=3.5, **kw), AWAY, n)


def demo_like(tr):
    """The demo's sequence: open in the air, slide at 28 deg to the stop, close on it, let go."""
    feed(tr, in_air(), pinch_arms(in_air()), 12)
    slide(tr, -10.0, L.depth_on_stop(math.radians(FULL) / 2))
    on_stop = on_finger(L.depth_on_stop(math.radians(FULL) / 2), FULL)
    feed(tr, on_stop, pinch_arms(on_stop), 12)
    for q in np.linspace(FULL, HOLD, 6):
        a = math.radians(q) / 2
        st = on_finger(L.depth_on_stop(a), q)
        feed(tr, st, pinch_arms(st))
    return clip_on(tr)


# -- geometry ---------------------------------------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_section_2_table(self):
        rows = {0.0: (6.0, 11.7, 32.0, 37.0, 0.0), 14.04: (15.0, 14.6, 40.8, 34.3, 3.0),
                16.3: (16.4, 15.0, 42.2, 33.8, 3.4), 22.0: (20.0, 16.1, 45.5, 32.5, 4.3),
                28.0: (23.7, 17.2, 49.0, 31.1, 5.2)}
        for q, (mouth, inner, front, pads, force) in rows.items():
            a = math.radians(q) / 2
            self.assertAlmostEqual(2 * L.half_gap_lips(a), mouth, delta=0.06, msg=q)
            self.assertAlmostEqual(2 * L.half_gap_inner(a), inner, delta=0.06, msg=q)
            self.assertAlmostEqual(L.front_width(a), front, delta=0.06, msg=q)
            self.assertAlmostEqual(L.pads_across(a), pads, delta=0.06, msg=q)
            self.assertAlmostEqual(L.pad_force(a), force, delta=0.06, msg=q)

    def test_fits_per_side(self):
        a = math.radians(FULL) / 2
        self.assertAlmostEqual(L.half_gap_lips(a) - 7.5, 4.4, delta=0.05)
        self.assertAlmostEqual(L.half_gap_inner(a) - 7.5, 1.1, delta=0.05)
        self.assertAlmostEqual((22.0 - 15.0) / 2, 3.5)  # between the side walls

    def test_tails_touch_at_the_open_limit(self):
        touch = 2 * math.degrees(math.atan(L.TAIL_GAP / 2 / (L.PIVOT_U)))
        self.assertAlmostEqual(touch, 28.07, delta=0.01)
        self.assertLessEqual(L.OPEN_LIMIT_DEG, touch)
        self.assertLess(L.OPEN_LIMIT, 2 * math.pi)
        self.assertAlmostEqual(T.JAW_LINK.upper, L.OPEN_LIMIT)
        self.assertEqual(T.JAW_LINK.lower, 0.0)

    def test_cushions_parallel_on_the_finger(self):
        """The five steps' pivot-side edges are collinear, 7.5 mm from the pivot axis: at 16.3 deg all five touch a
        15 mm finger at once; between them each step's face falls away from the finger."""
        a = math.radians(HOLD) / 2
        for (u0, _), face in zip(L.CUSHION_STEPS_U, L.CUSHION_FACES):
            p = (u0 - L.PIVOT_U_MM) * math.cos(a) + face * math.sin(a)
            self.assertAlmostEqual(L.cushion_half_gap(p, a), 7.5, delta=0.02, msg=u0)
            self.assertGreater(L.cushion_half_gap(p + 2.0, a), 7.5)

    def test_depth_on_the_stop(self):
        self.assertAlmostEqual(L.depth_on_stop(0.0), 31.0, delta=0.01)
        self.assertAlmostEqual(L.depth_on_stop(math.radians(HOLD) / 2), 30.0, delta=0.05)
        self.assertAlmostEqual(L.depth_on_stop(math.radians(FULL) / 2), 28.9, delta=0.05)

    def test_spring_and_forces(self):
        self.assertAlmostEqual(L.SPRING * L.OPEN_LIMIT, 0.0733, delta=0.0002)
        self.assertAlmostEqual(L.cushion_force(), 1.9, delta=0.05)
        self.assertAlmostEqual(L.UNIT_MASS * 9.81, 0.49, delta=0.01)
        self.assertGreater(0.8 * 2 * L.cushion_force() / (L.UNIT_MASS * 9.81), 6.0)

    def test_squeeze_floor_with_the_finger_drive(self):
        for gap, q in ((25.0, 28.0), (29.0, 28.0), (30.0, 24.6), (30.8, 22.0), (32.0, 17.8), (32.4, 16.3),
                       (33.0, 14.1), (34.5, 9.0)):
            self.assertAlmostEqual(math.degrees(L.finger_force_gap(gap)), q, delta=0.25, msg=gap)

    def test_entry_by_pivot_angle(self):
        """A centred finger reaches the stop only with the jaws at 16.3 deg or more; under that the full 15 mm
        section jams at the lips or at the cushions' inner end."""
        for q, depth in ((0.0, 0.0), (10.0, 2.0), (12.1, 4.0), (14.0, 4.0)):
            a = math.radians(q) / 2
            self.assertAlmostEqual(L.lip_line(a) - L.deepest_s(a), depth, delta=0.1, msg=q)
        for q in (16.3, 17.8, 22.0, 28.0):
            a = math.radians(q) / 2
            self.assertAlmostEqual(L.deepest_s(a), L.stop_line(a), delta=0.06, msg=q)

    def test_the_unit_as_built(self):
        st = rest()
        self.assertAlmostEqual(float(st.low[2]), L.TOP, places=6)  # on the table, not in it
        size = np.asarray(st.high) - np.asarray(st.low)
        np.testing.assert_allclose(size, [0.057, 0.037, 0.031], atol=1e-6)  # 32 mm across the shells + 2 pads
        self.assertAlmostEqual(SPEC.mass + T.JAW_LINK.mass, 0.050, places=9)
        self.assertEqual(SPEC.contact_offset, 0.0005)
        self.assertEqual(T.MANIKIN_SPEC.contact_offset, 0.0005)
        self.assertIsNone(T.MANIKIN_SPEC.mass)
        # the lips 61.5 mm short of the fingertip
        self.assertAlmostEqual(S.read(st).depth, -61.5, delta=0.01)
        self.assertAlmostEqual(L.START[0] + (L.UNIT_LEN / 2), 0.2085, places=6)

    def test_the_hand_as_built(self):
        st = T.MANIKIN_SPEC.rest_state()
        self.assertAlmostEqual(float(st.low[0]), L.TIP_X, places=6)
        self.assertAlmostEqual(float(st.low[2]), L.TOP, places=6)
        np.testing.assert_allclose(np.asarray(st.high) - np.asarray(st.low), [0.347, 0.060, 0.085], atol=1e-6)
        # nothing but the index finger ahead of the knuckle
        for s in SPEC_HAND_SHAPES:
            if s.radius not in (0.0075, 0.0065, 0.005) or s.kind != "cylinder" or s.axis != "X":
                lo = s.center[0] + L.TIP_X - s.aabb_half(np.eye(3))[0]
                self.assertGreaterEqual(lo, L.KNUCKLE_X - 1e-9)
        # the finger: 15 mm across, its axis 75 mm up, tip at x 0.270
        d = st.distance(np.array([[0.300, 0.0075, L.TOP + 0.075], [0.300, 0.0, L.TOP + 0.0825],
                                  [0.2699, 0.0, L.TOP + 0.075]]))
        np.testing.assert_allclose(d, [0.0, 0.0, 0.0001], atol=1e-6)

    def test_reading_round_trip(self):
        for depth, q, v, w in ((5.0, FULL, 1.0, -2.0), (29.9, HOLD, 0.0, 3.5), (15.0, 22.0, -4.0, 4.0)):
            rd = S.read(on_finger(depth, q, v, w))
            self.assertAlmostEqual(rd.depth, depth, delta=1e-6)
            self.assertAlmostEqual(rd.v, v, delta=1e-6)
            self.assertAlmostEqual(rd.w, w, delta=1e-6)
            self.assertAlmostEqual(rd.q_deg, q, delta=1e-6)

    def test_released_unit_rests_on_the_finger(self):
        """The upper side wall rests on the finger 3.5 mm down, and nothing of the unit is inside the finger."""
        st = on_finger(L.depth_on_stop(math.radians(HOLD) / 2), HOLD, w=3.5)
        top = st.link_to_world("base", np.array([0.0, 0.0, 0.011]))
        self.assertAlmostEqual(top[2], L.TOP + L.FINGER_Z + 0.0075, delta=1e-6)
        self.assertGreater(S.read(st).low, 0.050)


SPEC_HAND_SHAPES = T.MANIKIN_SPEC.shapes


# -- the tiers ---------------------------------------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def test_doing_nothing_scores_zero(self):
        tr = S.Tracker()
        res = feed(tr, rest(), AWAY, 200)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_demo_sequence_scores_one(self):
        tr = S.Tracker()
        res = demo_like(tr)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.details["d_open_mm"], 29.1, delta=0.4)
        self.assertAlmostEqual(res.details["d_shut_mm"], 30.0, delta=0.05)

    def test_held_open_on_the_stop_then_pulled_off_is_a_quarter(self):
        tr = S.Tracker()
        d = L.depth_on_stop(math.radians(FULL) / 2)
        slide(tr, -5.0, d)
        st = on_finger(d, FULL)
        feed(tr, st, pinch_arms(st), 10)
        res = feed(tr, in_air(), pinch_arms(in_air()), 30)
        self.assertEqual(res.score, 0.25)  # pushed home while held earns only the first stage
        res = feed(tr, dropped(), AWAY, 30)
        self.assertEqual(res.score, 0.25)  # the record stands until a reset

    def test_held_open_round_the_fingertip_short_of_the_stop_is_a_quarter(self):
        tr = S.Tracker()
        slide(tr, -5.0, 20.0)
        st = on_finger(20.0, FULL)
        res = feed(tr, st, pinch_arms(st), 60)
        self.assertEqual(res.score, 0.25)

    def test_never_open_clipped_on_at_the_stop_is_three_quarters(self):
        tr = S.Tracker()
        for d in np.linspace(-3.0, 30.0, 40):
            feed(tr, on_finger(d, HOLD), AWAY)
        res = clip_on(tr)
        self.assertFalse(res.details["slid_on_open"])
        self.assertEqual(res.score, 0.75)

    def test_open_to_6_mm_then_shut_to_the_stop_is_three_quarters(self):
        tr = S.Tracker()
        slide(tr, -3.0, 6.0)
        for d in np.linspace(6.0, 30.0, 30):
            st = on_finger(d, HOLD)
            feed(tr, st, pinch_arms(st))
        res = clip_on(tr)
        self.assertEqual(res.score, 0.75)

    def test_shut_under_4_mm_then_open_to_the_stop_is_one(self):
        tr = S.Tracker()
        for d in np.linspace(-3.0, 3.9, 10):
            st = on_finger(d, 12.0)
            feed(tr, st, pinch_arms(st))
        slide(tr, 3.9, L.depth_on_stop(math.radians(FULL) / 2))
        for q in np.linspace(FULL, HOLD, 6):
            st = on_finger(L.depth_on_stop(math.radians(q) / 2), q)
            feed(tr, st, pinch_arms(st))
        self.assertEqual(clip_on(tr).score, 1.0)

    def test_a_failed_entry_then_a_clean_one_is_one(self):
        tr = S.Tracker()
        for d in np.linspace(-3.0, 12.0, 15):  # pushed in shut (in a real trial soft lips could give)
            feed(tr, on_finger(d, HOLD), AWAY)
        self.assertGreater(tr.d_shut, 11.0)
        feed(tr, on_finger(-2.0, 0.0), AWAY, 3)  # right out past the lips: a fresh try
        self.assertIsNone(tr.d_shut)
        self.assertEqual(demo_like(tr).score, 1.0)

    def test_contact_penetration_at_the_stop_still_scores_one(self):
        tr = S.Tracker()
        demo_like(tr)
        res = clip_on(tr, depth=30.5)
        self.assertEqual(res.score, 1.0)

    def test_past_the_stop_is_not_in_the_chamber(self):
        self.assertTrue(S.read(on_finger(32.5, HOLD)).in_chamber)
        self.assertFalse(S.read(on_finger(32.6, HOLD)).in_chamber)

    def test_every_documented_score_is_reachable_and_only_those(self):
        seen = set()
        for build in (lambda tr: feed(tr, rest(), AWAY, 30),
                      lambda tr: feed(tr, in_air(), pinch_arms(in_air()), 12),
                      lambda tr: (slide(tr, -5, 28.9), feed(tr, on_finger(28.9, FULL), pinch_arms(on_finger(28.9, FULL)), 12),
                                  feed(tr, on_finger(28.9, FULL), AWAY, WIN + 2)),
                      lambda tr: (feed(tr, in_air(), pinch_arms(in_air()), 2), slide(tr, -5, 15.0, steps=10),
                                  clip_on(tr, depth=15.0)),
                      demo_like):
            tr = S.Tracker()
            r = build(tr)
            seen.add(tr.last.score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_scores_are_only_the_listed_ones(self):
        rng = np.random.default_rng(3)
        tr = S.Tracker()
        for _ in range(400):
            d = float(rng.uniform(-10, 33))
            q = float(rng.uniform(0, 28))
            st = on_finger(d, q, v=float(rng.uniform(-6, 6)), w=float(rng.uniform(-6, 6)))
            arms = AWAY if rng.random() < 0.5 else pinch_arms(st)
            res = feed(tr, st, arms)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))


# -- near misses on every threshold ----------------------------------------------------------------------------


class NearMisses(unittest.TestCase):
    def _clipped(self, depth=None, n=WIN + 2, **kw):
        tr = S.Tracker()
        demo_like(tr)
        return clip_on(tr, depth=depth, n=n, **kw)

    def test_depth_28(self):
        self.assertEqual(self._clipped(depth=28.1).score, 1.0)
        self.assertEqual(self._clipped(depth=27.9).score, 0.75)

    def test_depth_10(self):
        tr = S.Tracker()
        self.assertEqual(clip_on(tr, depth=10.1).score, 0.75)
        tr = S.Tracker()
        self.assertEqual(clip_on(tr, depth=9.9).score, 0.0)

    def test_depth_4_counts_toward_the_slide_record(self):
        tr = S.Tracker()
        feed(tr, on_finger(3.9, 10.0))
        self.assertIsNone(tr.d_shut)
        feed(tr, on_finger(4.1, 10.0))
        self.assertAlmostEqual(tr.d_shut, 4.1, delta=1e-9)

    def test_hold_band(self):
        for q, ok in ((14.4, False), (14.6, True), (18.0, True), (18.2, False)):
            self.assertEqual(S.read(on_finger(29.5, q)).gripping, ok, msg=q)

    def test_jaws_open_line(self):
        self.assertFalse(S.read(in_air(21.9)).jaws_open)
        self.assertTrue(S.read(in_air(22.1)).jaws_open)
        tr = S.Tracker()
        self.assertEqual(feed(tr, in_air(21.9), pinch_arms(in_air(21.9)), 30).score, 0.0)

    def test_slide_margin(self):
        for extra, want in ((1.4, 1.0), (1.6, 0.75)):
            tr = S.Tracker()
            slide(tr, -3.0, 20.0)
            st = on_finger(20.0 + extra, HOLD)
            feed(tr, st, pinch_arms(st))
            # d_shut = 20 + extra, d_open = 20; closing alone adds 1.1 mm, so 1.4 passes and 1.6 does not
            self.assertAlmostEqual(tr.d_open, 20.0, delta=1e-9)
            res = feed(tr, on_finger(20.0 + extra, HOLD, w=3.5), AWAY, WIN + 2)
            self.assertEqual(res.score, 0.75)  # short of 28: 0.75 either way
            self.assertEqual(tr.slid_on_open(), want == 1.0, msg=extra)

    def test_centre_offsets(self):
        for kw in ({"v": 4.9}, {"v": -4.9}, {"w": 4.9}, {"w": -4.9}):
            self.assertTrue(S.read(on_finger(20.0, HOLD, **kw)).in_chamber, msg=kw)
        for kw in ({"v": 5.1}, {"v": -5.1}, {"w": 5.1}, {"w": -5.1}):
            self.assertFalse(S.read(on_finger(20.0, HOLD, **kw)).in_chamber, msg=kw)

    def test_pad_distance_30_mm(self):
        st = on_finger(29.9, HOLD, w=3.5)
        for g, ok in ((0.029, False), (0.031, True)):
            arms = above_arms(st, g)
            self.assertAlmostEqual(rigstate.hands_gap(arms, st.distance), g, delta=1e-6)
            tr = S.Tracker()
            demo_like(tr)
            res = feed(tr, st, arms, WIN + 2)
            self.assertEqual(res.score, 1.0 if ok else 0.25, msg=g)  # not let go: no better than the first stage

    def test_lowest_point_20_mm(self):
        for h, ok in ((0.019, False), (0.021, True)):
            st = in_air(FULL, height=0.0)
            st = unit(np.asarray(st.origin) + np.array([0.0, 0.0, h - (float(st.low[2]) - L.TOP)]), st.rotation(),
                      FULL)
            self.assertAlmostEqual(S.read(st).low, h, delta=1e-9)
            tr = S.Tracker()
            self.assertEqual(feed(tr, st, pinch_arms(st), 12).score, 0.25 if ok else 0.0, msg=h)

    def test_latches_need_10_steps(self):
        tr = S.Tracker()
        feed(tr, in_air(), pinch_arms(in_air()), 9)
        self.assertEqual(feed(tr, dropped(), AWAY).score, 0.0)
        tr = S.Tracker()
        feed(tr, in_air(), pinch_arms(in_air()), 10)
        self.assertEqual(feed(tr, dropped(), AWAY).score, 0.25)
        d = L.depth_on_stop(math.radians(FULL) / 2)
        st = on_finger(d, FULL)
        tr = S.Tracker()
        feed(tr, st, pinch_arms(st), 9)
        self.assertEqual(feed(tr, dropped(), AWAY, WIN + 2).score, 0.0)  # 9 steps: neither record
        self.assertFalse(tr.pushed_home)
        tr = S.Tracker()
        feed(tr, st, pinch_arms(st), 10)
        self.assertEqual(feed(tr, st, AWAY, WIN + 2).score, 0.5)  # 10 steps, then let go on the finger

    def test_alignment_20_deg(self):
        self.assertTrue(S.read(on_finger(20.0, HOLD, pitch=19.0)).over_finger)
        self.assertFalse(S.read(on_finger(20.0, HOLD, pitch=21.0)).over_finger)

    def test_nail_side_45_deg(self):
        self.assertTrue(S.read(on_finger(20.0, HOLD, roll=44.0)).over_finger)
        self.assertFalse(S.read(on_finger(20.0, HOLD, roll=46.0)).over_finger)

    def test_movement_2_mm(self):
        for dx, want in ((0.0019, 1.0), (0.0021, 0.5)):
            tr = S.Tracker()
            demo_like(tr)
            for k in range(WIN):
                feed(tr, on_finger(29.95, HOLD, w=3.5, shift=(0.0, dx * k / (WIN - 1), 0.0)), AWAY)
            self.assertEqual(tr.last.score, want, msg=dx)

    def test_turn_2_deg(self):
        for deg, want in ((1.9, 1.0), (2.1, 0.5)):
            tr = S.Tracker()
            demo_like(tr)
            for k in range(WIN):
                feed(tr, on_finger(29.95, HOLD, w=3.5, roll=deg * k / (WIN - 1)), AWAY)
            self.assertEqual(tr.last.score, want, msg=deg)

    def test_gripping_lost_on_one_step_of_the_last_second(self):
        tr = S.Tracker()
        demo_like(tr)
        feed(tr, on_finger(29.95, HOLD + 2.0, w=3.5), AWAY)
        res = feed(tr, on_finger(29.95, HOLD, w=3.5), AWAY, WIN - 1)
        self.assertEqual(res.score, 0.5)
        res = feed(tr, on_finger(29.95, HOLD, w=3.5), AWAY)
        self.assertEqual(res.score, 1.0)


# -- held, hovering and the exploit list ---------------------------------------------------------------------------


class Exploits(unittest.TestCase):
    def test_held_states(self):
        d_open = L.depth_on_stop(math.radians(FULL) / 2)
        # held open in the air: 0.25
        tr = S.Tracker()
        self.assertEqual(feed(tr, in_air(), pinch_arms(in_air()), 40).score, 0.25)
        # clipped on, gripper open but within 30 mm: 0.25 (not let go: only the held first stage counts)
        tr = S.Tracker()
        demo_like(tr)
        st = on_finger(29.95, HOLD, w=3.5)
        self.assertEqual(feed(tr, st, above_arms(st, 0.010), WIN + 2).score, 0.25)
        # gripper still on the tail, pinched at 16.3 on the stop (held at the clipped angle): 0.25
        tr = S.Tracker()
        demo_like(tr)
        self.assertEqual(feed(tr, st, pinch_arms(st), WIN + 2).score, 0.25)
        # clipped and let go short of the stop: 0.75; on the stop, never opened: 0.75 (tests above)
        tr = S.Tracker()
        slide(tr, -3.0, 15.0)
        self.assertEqual(clip_on(tr, depth=15.0).score, 0.75)
        # never pushed home, gripper hovering next to the clipped unit: 0.25 (it was opened in the air)
        tr = S.Tracker()
        feed(tr, in_air(), pinch_arms(in_air()), 12)
        slide(tr, -3.0, 15.0)
        st = on_finger(15.0, HOLD, w=3.5)
        self.assertEqual(feed(tr, st, above_arms(st, 0.0), WIN + 2).score, 0.25)
        self.assertTrue(d_open > 28.0)

    def test_push_home_hold_then_withdraw_is_not_half_credit(self):
        """Pushed fully on and held, then carried away with the grip still on the tail: the held stage earns at
        most the first tier (the open slide is in the air), never 0.5."""
        d = L.depth_on_stop(math.radians(FULL) / 2)
        for want in (0.25,):
            tr = S.Tracker()
            slide(tr, -5.0, d)
            st = on_finger(d, FULL)
            feed(tr, st, pinch_arms(st), 30)  # held home well over 0.5 s
            for h in np.linspace(0.0, 0.08, 20):  # withdrawn by the pinch, taking the unit with it
                up = unit(np.asarray(st.origin) + np.array([-h, 0.0, h]), st.rotation(), FULL)
                res = feed(tr, up, pinch_arms(up))
            self.assertTrue(tr.pushed_home)
            self.assertEqual(res.score, want)
            res = feed(tr, dropped(), AWAY, 2 * WIN)
            self.assertEqual(res.score, want)

    def test_grasp_round_the_body_pushed_at_the_fingertip(self):
        """Offline clip: a grasp ahead of the pivot holds the jaws shut; the lips butt on the fingertip."""
        self.assertEqual(play("body").last.score, 0.0)

    def test_grasp_over_the_pivot(self):
        tr = play("pivot")
        self.assertEqual(tr.last.score, 0.0)
        self.assertFalse(tr.opened_in_air)

    def test_lying_on_the_fist_or_across_the_finger(self):
        tr = S.Tracker()
        across = unit((0.300, 0.0, L.TOP + 0.0825 + 0.0155 + 0.001), S.rz(math.pi / 2))
        self.assertEqual(feed(tr, across, AWAY, 40).score, 0.0)
        on_fist = unit((0.39, -0.02, L.TOP + 0.085 + 0.0155), np.eye(3))
        self.assertEqual(feed(S.Tracker(), on_fist, AWAY, 40).score, 0.0)

    def test_pinched_open_on_the_table_without_lifting(self):
        a = math.radians(FULL) / 2
        st = unit(S.START_ORIGIN, S.rz(-a), FULL)
        tr = S.Tracker()
        self.assertEqual(feed(tr, st, pinch_arms(st), 60).score, 0.0)

    def test_show_open_relax_push_home(self):
        for let_go, want in ((True, 0.75), (False, 0.25)):  # held: only the first stage (the open slide is in the air)
            tr = S.Tracker()
            slide(tr, -3.0, 5.0)
            for d in np.linspace(5.0, 30.0, 30):
                st = on_finger(d, HOLD)
                feed(tr, st, pinch_arms(st))
            st = on_finger(30.0, HOLD, w=3.5)
            res = feed(tr, st, AWAY if let_go else pinch_arms(st), WIN + 2)
            self.assertEqual(res.score, want)

    def test_partly_open_tip_entry_then_open_is_one(self):
        tr = S.Tracker()
        for d in np.linspace(-2.0, 3.5, 6):
            st = on_finger(d, 14.0)
            feed(tr, st, pinch_arms(st))
        self.assertIsNone(tr.d_shut)
        slide(tr, 3.5, 28.9)
        for q in np.linspace(FULL, HOLD, 6):
            st = on_finger(L.depth_on_stop(math.radians(q) / 2), q)
            feed(tr, st, pinch_arms(st))
        self.assertEqual(clip_on(tr).score, 1.0)

    def test_wrong_side_or_crosswise(self):
        tr = S.Tracker()
        demo_like(tr)
        self.assertEqual(feed(tr, on_finger(29.9, HOLD, roll=180.0), AWAY, WIN + 2).score, 0.25)
        tr = S.Tracker()
        feed(tr, in_air(), pinch_arms(in_air()), 12)
        self.assertEqual(feed(tr, on_finger(29.9, HOLD, roll=180.0), AWAY, WIN + 2).score, 0.25)
        tr = S.Tracker()
        self.assertEqual(feed(tr, on_finger(20.0, HOLD, yaw=90.0), AWAY, WIN + 2).score, 0.0)

    def test_knocked_off_before_the_second_is_complete(self):
        tr2 = S.Tracker()
        feed(tr2, in_air(), pinch_arms(in_air()), 12)
        d = L.depth_on_stop(math.radians(FULL) / 2)
        slide(tr2, -5.0, d)
        st = on_finger(d, FULL)
        feed(tr2, st, pinch_arms(st), 12)
        feed(tr2, on_finger(29.95, HOLD, w=3.5), AWAY, WIN - 5)
        res = feed(tr2, dropped(), AWAY, 30)
        self.assertEqual(res.score, 0.25)  # off the finger: only the first stage is left

    def test_still_swinging(self):
        tr = S.Tracker()
        demo_like(tr)
        for k in range(WIN + 2):
            feed(tr, on_finger(29.95, HOLD, w=3.5, pitch=4.0 * math.sin(k)), AWAY)
        self.assertEqual(tr.last.score, 0.5)

    def test_unreadable_arms_are_never_let_go(self):
        st = on_finger(29.95, HOLD, w=3.5)
        nan = rigstate.Arm(np.array([np.nan, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 1.0)
        for bad in (None, {}, {"left": AWAY["left"]}, {"left": AWAY["left"], "right": nan},
                    {"left": AWAY["left"], "right": rigstate.Arm(AWAY["right"].tcp, AWAY["right"].tips, math.inf)}):
            tr = S.Tracker()
            demo_like(tr)
            res = feed(tr, st, bad, WIN + 2)
            self.assertEqual(res.score, 0.25, msg=repr(bad))
            self.assertFalse(res.details["released"])

    def test_unreadable_arms_on_one_step_of_the_last_second(self):
        tr = S.Tracker()
        demo_like(tr)
        st = on_finger(29.95, HOLD, w=3.5)
        feed(tr, st, None)
        self.assertEqual(feed(tr, st, AWAY, WIN - 1).score, 0.25)
        self.assertEqual(feed(tr, st, AWAY).score, 1.0)

    def test_a_distance_that_raises_is_not_let_go(self):
        class Broken:
            def __init__(self, s):
                self._s = s

            def __getattr__(self, k):
                if k == "distance":
                    raise RuntimeError("no distance")
                return getattr(self._s, k)

        tr = S.Tracker()
        demo_like(tr)
        res = feed(tr, Broken(on_finger(29.95, HOLD, w=3.5)), AWAY, WIN + 2)
        self.assertEqual(res.score, 0.25)

    def test_bad_states_keep_the_records_and_restart_the_windows(self):
        tr = S.Tracker()
        demo_like(tr)
        st = on_finger(29.95, HOLD, w=3.5)
        bad = unit(np.array([np.nan, 0.0, 0.8]))
        self.assertEqual(feed(tr, bad, AWAY).score, 0.25)
        self.assertEqual(feed(tr, None, AWAY).score, 0.25)
        self.assertEqual(feed(tr, st, AWAY, WIN - 1).score, 0.25)
        self.assertEqual(feed(tr, st, AWAY).score, 1.0)


# -- reset ---------------------------------------------------------------------------------------------------------


class Reset(unittest.TestCase):
    def test_records_survive_taking_it_off_and_a_reset_clears_them(self):
        tr = S.Tracker()
        demo_like(tr)
        self.assertEqual(feed(tr, in_air(), pinch_arms(in_air()), 5).score, 0.25)
        self.assertEqual(feed(tr, on_finger(-40.0, 0.0), AWAY, 5).score, 0.25)
        tr.reset()
        self.assertEqual(feed(tr, rest(), AWAY, 5).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        demo_like(tr)
        res = feed(tr, rest(), AWAY)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.resets, 1)

    def test_a_gradual_return_is_not_a_reset(self):
        tr = S.Tracker()
        feed(tr, in_air(), pinch_arms(in_air()), 12)
        for h in np.linspace(0.060, 0.0, 40):
            st = unit(S.START_ORIGIN + np.array([0.0, 0.0, h]))
            feed(tr, st, AWAY)
        self.assertEqual(tr.resets, 0)
        self.assertEqual(tr.last.score, 0.25)

    def test_task_resets_with_rigstate_and_mechkit(self):
        objs = {S.OXIMETER: rest(), S.MANIKIN: T.MANIKIN_SPEC.rest_state()}
        T._TRACKER.reset()
        T._TRACKER.pushed_home = True
        rigstate.offline_reset()  # runs the task's rigstate.on_reset hook
        self.assertFalse(T._TRACKER.pushed_home)
        T.success(dict(objs))
        T._TRACKER.opened_in_air = True
        mk.offline_reset()
        T.success(dict(objs))
        self.assertFalse(T._TRACKER.opened_in_air)


# -- the offline clip and the demo -----------------------------------------------------------------------------------


def _ready_tcp(side):
    import yamkin as K
    p, _ = K.tcp_pose(np.array(K.READY, dtype=float), np.asarray(K.BASES["station"][side], dtype=float))
    return np.asarray(p, dtype=float)


def _play_plan(wps, side, hz=20.0, hold_s=3.0):
    """Linear interpolation of the waypoints (TCP and finger), from the ready pose, as arm snapshots per step."""
    import yamkin as K
    pos0, fin0 = _ready_tcp(side), K.FINGER_OPEN
    segs, t = [], 0.0
    for w in wps:
        segs.append((t, t + w.duration, pos0, np.asarray(w.pos, dtype=float), fin0, float(w.finger)))
        t += w.duration
        pos0, fin0 = np.asarray(w.pos, dtype=float), float(w.finger)
    n = int(math.floor((t + hold_s) * hz)) + 1
    for k in range(n):
        tk = (k + 1) / hz
        pos, fin = pos0, fin0
        for t0, t1, p0, p1, f0, f1 in segs:
            if tk <= t1:
                u = (tk - t0) / (t1 - t0)
                pos, fin = p0 + (p1 - p0) * u, f0 + (f1 - f0) * u
                break
        arms = {}
        for a in ("left", "right"):
            if a == side:
                arms[a] = rigstate.Arm(pos, pos - np.array([0.0, 0.0, 0.013]), float(np.clip(fin / K.FINGER_OPEN, 0, 1)))
            else:
                r = _ready_tcp(a)
                arms[a] = rigstate.Arm(r, r - np.array([0.0, 0.0, 0.013]), 1.0)
        yield arms


def play(variant="", idle=None):
    """The demo (or a variant of it) through the offline clip and the scorer; returns the tracker."""
    old = {k: os.environ.get(k) for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM")}
    os.environ["DEMO_VARIANT"] = variant
    if idle:
        os.environ["DEMO_IDLE_ARM"] = idle
    else:
        os.environ.pop("DEMO_IDLE_ARM", None)
    try:
        plans = T.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    clip = OC.OfflineClip()
    tr = S.Tracker()
    ((side, wps),) = plans.items()
    for arms in _play_plan(wps, side):
        clip.step(arms)
        origin, yaw, q = clip.pose()
        st = mk.state_from_poses(SPEC, (*origin, *mk.yaw_quat(yaw)), joints={S.JAW: q}, source="offline")
        tr.update({S.OXIMETER: st}, arms)
    return tr


class OfflineClipAndDemo(unittest.TestCase):
    def test_the_demo_scores_one(self):
        tr = play("")
        self.assertEqual(tr.last.score, 1.0)
        self.assertAlmostEqual(tr.last.details["depth_mm"], 29.95, delta=0.1)
        self.assertTrue(tr.last.details["slid_on_open"])

    def test_the_left_arm_alone_scores_one(self):
        self.assertEqual(play("", idle="right").last.score, 1.0)

    def test_counterfactuals(self):
        for variant, want in (("light", 0.0), ("part", 0.75), ("just_open", 1.0), ("early_release", 0.75),
                              ("no_release", 0.25), ("hover", 0.25)):
            self.assertEqual(play(variant).last.score, want, msg=variant)

    def test_a_bad_variant_name_raises(self):
        os.environ["DEMO_VARIANT"] = "nonsense"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT", None)

    def test_hold_only_has_no_plan(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY", None)

    def test_squeeze_geometry_of_the_demo(self):
        """After the squeeze the pivot is 24.7 mm ahead of the TCP and the lips 59.9 mm; the fingertip passes the
        lips at TCP x 0.2101 and meets the stop at 0.2390; the slide ends 5.5 mm past that."""
        a = math.radians(FULL) / 2
        pivot_ahead = 10.5 + L.pad_lever(a)
        self.assertAlmostEqual(pivot_ahead, 24.7, delta=0.05)
        self.assertAlmostEqual(pivot_ahead + L.lip_line(a), 59.9, delta=0.05)
        self.assertAlmostEqual(L.TIP_X * 1000 - pivot_ahead - L.lip_line(a), 210.1, delta=0.1)
        stop_tcp = L.TIP_X * 1000 - pivot_ahead - L.stop_line(a)
        self.assertAlmostEqual(stop_tcp, 239.0, delta=0.1)
        self.assertAlmostEqual(T.SLIDE_END_X * 1000 - stop_tcp, 5.5, delta=0.1)
        self.assertAlmostEqual(L.pad_lever(a) - L.pad_lever(math.radians(HOLD) / 2), 1.7, delta=0.05)

    def test_pick_point_in_the_window(self):
        t = (T.PICK_X - (L.START[0] - L.U0)) * 1000.0
        self.assertAlmostEqual(t, -0.5, delta=1e-6)
        self.assertGreaterEqual(t, -10.5 + 9.9 - 1e-6)
        self.assertLessEqual(t, 9.4 - 9.9 + 1e-6)
        self.assertEqual(math.degrees(OC.q_max_for(t - 6.5)), FULL)
        self.assertAlmostEqual(math.degrees(OC.q_max_for(2.9)), 22.0, delta=1e-6)

    def test_open_fingers_clear_the_grip_pads(self):
        self.assertAlmostEqual(-2 * T.OPEN * 1000 / 2 - 18.5, 11.5, delta=1e-6)
        self.assertAlmostEqual(-2 * T.RELEASE * 1000 / 2 - L.pads_across(math.radians(HOLD) / 2) / 2, 18.1, delta=0.1)

    def test_demo_timing(self):
        wps = T.plan()
        total = sum(w.duration for w in wps)
        self.assertAlmostEqual(total, 23.5, delta=1e-9)
        # the last 3 cm of the descent take 1.5 s, and the gripper rests 0.5 s before the pinch and before the release
        down = next(i for i, w in enumerate(wps) if w.label.startswith("slowly down"))
        self.assertGreaterEqual(wps[down].duration, 1.5)
        self.assertAlmostEqual(wps[down - 1].pos[2] - wps[down].pos[2], 0.030, delta=1e-9)
        self.assertGreaterEqual(wps[down + 1].duration, 0.5)
        self.assertEqual(wps[down + 1].finger, wps[down].finger)
        self.assertLess(total * 20, 900)
        for w in wps:
            if w.finger != wps[0].finger and w.label.startswith(("pinch", "open")):
                self.assertGreaterEqual(w.duration, 1.5)
        slide_wp = next(w for w in wps if w.label.startswith("slide"))
        self.assertAlmostEqual((T.SLIDE_END_X - T.APPROACH_X) / slide_wp.duration, 0.010, delta=1e-9)

    def test_the_task_runs_the_offline_clip_only_without_a_simulator(self):
        objs = {S.OXIMETER: rest(), S.MANIKIN: T.MANIKIN_SPEC.rest_state()}
        self.assertTrue(T.no_simulator(objs))
        sim = dict(objs)
        sim[S.OXIMETER] = mk.state_from_poses(SPEC, (*S.START_ORIGIN, 0, 0, 0, 1), source="sim")
        self.assertFalse(T.no_simulator(sim))


# -- the text --------------------------------------------------------------------------------------------------------


class Text(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()

    def test_instruction_matches(self):
        self.assertEqual(self.meta["instruction"], T.INSTRUCTION)

    def test_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150,
                  "fidelity": 490, "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.meta[k]), n, msg=k)

    def test_no_dashes_and_ascii(self):
        for name in ("task.json", "README.md", "task.py", "scoring.py", "layout.py", "offline_clip.py",
                     "tests/test_scoring.py"):
            text = (HERE / name).read_text()
            self.assertNotIn(chr(0x2014), text, msg=name)
            self.assertNotIn(chr(0x2013), text, msg=name)
            self.assertTrue(all(ord(c) < 128 for c in text), msg=name)

    def test_banned_words(self):
        fields = " ".join(str(v) for k, v in self.meta.items() if k != "scores").lower()
        for word in (r"\bscale\b", r"\bscaled\b", r"model of", r"analogue", r"\bmock", r"trainer", r"stand-in",
                     r"stands for", r"represents", r"\btbd\b", r"<", r"measure later", r"to be measured",
                     r"\bbuy", r"\bbought", r"probe", r"monitor", r"alarm"):
            self.assertIsNone(re.search(word, fields), msg=word)
        self.assertIn("our estimates", self.meta["fidelity"])
        self.assertEqual(self.meta["scores"], "Dream-1 run: 0.0; demo: 1.0")

    def test_numbers_agree(self):
        sc, se, fi = self.meta["scoring"], self.meta["setup"], self.meta["fidelity"]
        for txt in ("22 degrees", "16.3 degrees", "1.8 degrees", "28 mm", "30 mm", "20 mm", "45.5 mm", "4 mm",
                    "32.5 mm", "5 mm", "0.5 s", "1 s"):
            self.assertIn(txt, sc, msg=txt)
        for txt in ("0.15 N m/rad", "57 mm", "50 g", "15 mm", "28 degrees", "61.5 mm", "10 x 20 x 2.5 mm"):
            self.assertIn(txt, se, msg=txt)
        self.assertIn("0.15 N m/rad", fi)
        self.assertIn("28 degrees", fi)
        self.assertEqual(self.meta["run_limit"], "900 steps (45 seconds at 20 Hz)")

    def test_readme_names_the_files_that_exist(self):
        for name in ("task.json", "task.py", "layout.py", "scoring.py", "offline_clip.py", "mechkit.py",
                     "rigstate.py", "tests/test_scoring.py"):
            self.assertTrue((HERE / name).is_file(), msg=name)
            self.assertIn(name, self.readme, msg=name)
        self.assertIn("Third-party assets: none", self.readme)

    def test_kit_files_are_the_kit(self):
        kit = None
        for p in HERE.parents:
            if (p / "tools" / "kit" / "mechkit.py").is_file():
                kit = p / "tools" / "kit"
                break
        if kit is None:
            self.skipTest("tools/kit not found (a submission copy)")
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual((HERE / name).read_bytes(), (kit / name).read_bytes(), msg=name)


if __name__ == "__main__":
    unittest.main()
