"""Offline tests for scoring.py, offline_saddle.py and task.py: synthetic states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import math
import os
import pathlib
import re
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


if _stub_dir() is not None and str(_stub_dir()) not in sys.path:
    sys.path.insert(1, str(_stub_dir()))  # mechkit (offline_saddle, task.py) imports yam_arena.task_file

import layout as L  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = L.CONTROL_HZ
SEAT_H = L.LINE_TOP  # the clamp's origin height when it is seated on the suction line
RIDE = L.DISPLAY_TOP - L.CEILING  # the cover's origin above a clamp's origin when its ceiling rests on the display
COVER_H = L.LINE_TOP + RIDE  # the cover's origin height when its ceiling rests on the seated clamp's display


def quat(yaw_deg=0.0, tilt_deg=0.0, axis="x"):
    """Turned yaw about the vertical after tilting about the body's x (or y) axis (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    qz = (0.0, 0.0, math.sin(a), math.cos(a))
    qt = (math.sin(b), 0.0, 0.0, math.cos(b)) if axis == "x" else (0.0, math.sin(b), 0.0, math.cos(b))
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qt
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def part(shapes, x, y, h, yaw=0.0, tilt=0.0, axis="x", source="sim"):
    """A part's state as mechkit reports it: frame origin (world), orientation, world bounding box."""
    o = np.array([x, y, L.TOP + h])
    q = quat(yaw, tilt, axis)
    lo, hi = S.part_bounds(shapes, o, q)
    return SimpleNamespace(origin=o, quat_xyzw=q, low=lo, high=hi, pos=(lo + hi) / 2, lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3), source=source)


def clamp(x=L.LINE_X, y=0.090, h=SEAT_H, **kw):
    return part(L.CLAMP_SHAPES, x, y, h, **kw)


def cover(x=L.LINE_X, y=0.090, h=COVER_H, **kw):
    return part(L.COVER_SHAPES, x, y, h, **kw)


CLAMP_START = clamp(*L.CLAMP_START)
COVER_START = cover(*L.COVER_START)


def arm(tcp):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=1.0)


READY = {"left": arm((0.3234, 0.305, L.TOP + 0.2606)), "right": arm((0.3234, -0.305, L.TOP + 0.2606))}


def arms_with_left(tcp):
    return {"left": arm(tcp), "right": READY["right"]}


def feed(seq, arms=READY, tracker=None, offline=False):
    """Feed (clamp, cover) pairs; returns the tracker and the last result."""
    t = tracker or Tracker()
    r = None
    for c, v in seq:
        r = t.update(c, v, arms, offline)
    return t, r


def hold(c, v, n=12):
    return [(c, v)] * n


def done(c, v, arms=READY, n=12):
    return feed(hold(c, v, n), arms)[1]


class Geometry(unittest.TestCase):
    def test_heading_folds_both_ways(self):
        for yaw, want in ((0, 0), (180, 0), (-180, 0), (15, 15), (165, 15), (90, 90), (-30, 30)):
            self.assertAlmostEqual(S.heading_err_deg(quat(yaw)), want, places=6)

    def test_tilt_is_not_folded(self):
        self.assertAlmostEqual(S.tilt_deg(quat(0, 4)), 4.0, places=6)
        self.assertAlmostEqual(S.tilt_deg(quat(30, 4, "y")), 4.0, places=6)
        self.assertAlmostEqual(S.tilt_deg(quat(0, 180)), 180.0, places=4)

    def test_bounds_match_mechkit(self):
        mk = importlib.import_module("mechkit")
        spec = mk.assembly("c", origin=(0.2, 0.1, L.TOP + 0.05), mass=0.06,
                           shapes=[mk.box(c, s) for c, s, _ in L.CLAMP_SHAPES])
        st = mk.state_from_poses(spec, (0.2, 0.1, L.TOP + 0.05, *quat(20, 7)))
        lo, hi = S.part_bounds(L.CLAMP_SHAPES, st.origin, st.quat_xyzw)
        np.testing.assert_allclose(lo, st.low, atol=1e-9)
        np.testing.assert_allclose(hi, st.high, atol=1e-9)

    def test_bare_copper_ends(self):
        half = L.BLOCK_S[1] / 2
        self.assertTrue(S.on_bare_copper(clamp(y=L.COUPLING_END_Y - half - 1e-6)))
        self.assertFalse(S.on_bare_copper(clamp(y=L.COUPLING_END_Y - half + 0.0005)))
        self.assertTrue(S.on_bare_copper(clamp(y=L.FOAM_START_Y + half + 1e-6)))
        self.assertFalse(S.on_bare_copper(clamp(y=L.FOAM_START_Y + half - 0.0005)))
        # turned, the block's corners reach further along the line
        self.assertFalse(S.on_bare_copper(clamp(y=L.COUPLING_END_Y - half, yaw=10)))

    def test_d_is_from_the_valve_face(self):
        self.assertAlmostEqual(S.d_along(clamp(y=0.090)), 0.100)
        self.assertAlmostEqual(S.d_along(clamp(y=L.VALVE_FACE_Y - 0.150)), 0.150)

    def test_ceiling_gap_and_outline(self):
        self.assertAlmostEqual(S.ceiling_gap(clamp(), cover()), 0.0, places=9)
        self.assertAlmostEqual(S.ceiling_gap(clamp(), cover(h=COVER_H + 0.002)), 0.002, places=9)
        self.assertTrue(S.block_in_cover(clamp(), cover()))
        edge = L.COVER_FOOT_Y - L.BLOCK_S[1] / 2  # 15 mm along the line
        self.assertTrue(S.block_in_cover(clamp(), cover(y=0.090 + edge - 1e-6)))
        self.assertFalse(S.block_in_cover(clamp(), cover(y=0.090 + edge + 0.0005)))
        side = L.COVER_FOOT_X - L.BLOCK_S[0] / 2  # 14 mm across it
        self.assertTrue(S.block_in_cover(clamp(), cover(x=L.LINE_X + side - 1e-6)))
        self.assertFalse(S.block_in_cover(clamp(), cover(x=L.LINE_X + side + 0.0005)))

    def test_physical_clearances(self):
        self.assertAlmostEqual(L.LEG_INNER - L.LINE_HALF, 0.007)  # the legs pass 7 mm either side of the copper
        self.assertAlmostEqual(L.CHANNEL_HALF - (L.LEG_X + L.LEG_S[0] / 2), 0.008)  # the walls 8 mm either side
        self.assertAlmostEqual(L.CHANNEL_HALF - L.BLOCK_S[0] / 2, 0.008)
        for t in (L.DISPLAY_S[2], L.WALL_T, L.LEG_S[0], L.ROOF_T, L.PAD_T + 0.001):
            self.assertGreaterEqual(t, 0.004 - 1e-12)  # the program's least wall (the pads are fixed plates)
        self.assertLessEqual(L.COVER_W - 0.008, 0.080)  # the cover's grip fits the fingers
        self.assertGreaterEqual(0.094 - L.COVER_W, 0.010)  # ... and the open fingers pass outside its walls

    def test_any_clamp_resting_astride_passes_x_and_heading(self):
        """The x and heading tolerances are looser than the physics: a clamp resting astride (legs clear of the
        copper's sides) always passes them, so only the end state decides."""
        play = L.LEG_INNER - L.LINE_HALF
        self.assertLess(play, L.SEAT_X_TOL)
        # the legs' inner corners reach the copper's side when L cos(t) - 20 mm sin(t) = 9.5 mm
        r, phi = math.hypot(L.LEG_INNER, L.BLOCK_S[1] / 2), math.atan2(L.BLOCK_S[1] / 2, L.LEG_INNER)
        t_max = math.degrees(math.acos(L.LINE_HALF / r) - phi)
        self.assertAlmostEqual(t_max, 18.0, delta=0.1)
        self.assertLess(t_max, L.SEAT_YAW_DEG)

    def test_a_turned_cover_cannot_rest_on_the_display(self):
        """The scorer leaves the cover's heading free: turned past about 31 degrees (centred), its walls' inner
        corners are over the copper, and their bottoms are below its top, so the walls land on the copper and the
        ceiling is 16 mm above the display: not insulated."""
        r, phi = math.hypot(L.CHANNEL_HALF, L.COVER_L / 2), math.atan2(L.COVER_L / 2, L.CHANNEL_HALF)
        t_max = math.degrees(math.acos(L.LINE_HALF / r) - phi)
        self.assertAlmostEqual(t_max, 31.4, delta=0.1)
        self.assertLess(COVER_H, L.LINE_TOP)  # the walls' bottoms reach below the copper's top
        on_copper = cover(h=L.LINE_TOP, yaw=45)
        self.assertAlmostEqual(S.ceiling_gap(clamp(), on_copper), L.CEILING - L.DISPLAY_TOP, places=9)
        self.assertEqual(done(clamp(), on_copper).score, 0.5)

    def test_point_box_distance(self):
        lo, hi = np.zeros(3), np.ones(3)
        self.assertEqual(S.point_box_distance((0.5, 0.5, 0.5), lo, hi), 0.0)
        self.assertAlmostEqual(S.point_box_distance((2.0, 0.5, 0.5), lo, hi), 1.0)
        self.assertAlmostEqual(S.point_box_distance((2.0, 2.0, 0.5), lo, hi), math.sqrt(2))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = done(CLAMP_START, COVER_START)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertTrue(Tracker.at_start(CLAMP_START, COVER_START))

    def test_seated_near_and_insulated_scores_one(self):
        r = done(clamp(), cover())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["d_mm"], 100.0)

    def test_at_rest_needs_half_a_second(self):
        t = Tracker()
        for i in range(L.STILL_STEPS):
            self.assertEqual(t.update(clamp(), cover(), READY).score, 0.0, i)
        self.assertEqual(t.update(clamp(), cover(), READY).score, 1.0)

    def test_seated_near_not_insulated_scores_half(self):
        self.assertEqual(done(clamp(), COVER_START).score, 0.5)

    def test_seated_far_scores_quarter_either_way(self):
        y = L.VALVE_FACE_Y - 0.200
        self.assertEqual(done(clamp(y=y), cover(y=y)).score, 0.25)
        self.assertEqual(done(clamp(y=y), COVER_START).score, 0.25)

    def test_d_boundary(self):
        y = L.VALVE_FACE_Y - L.NEAR_D + 1e-6
        self.assertEqual(done(clamp(y=y), cover(y=y)).score, 1.0)
        self.assertEqual(done(clamp(y=y - 0.0005), cover(y=y - 0.0005)).score, 0.25)

    def test_closest_seat_to_the_valve(self):
        y = L.COUPLING_END_Y - L.BLOCK_S[1] / 2 - 1e-6  # d 45 mm: the block's end at the coupling's end
        self.assertEqual(done(clamp(y=y), cover(y=y - 0.010)).score, 1.0)

    def test_seat_boundaries(self):
        cases = [
            (dict(x=L.LINE_X + 0.008 - 1e-6), 1.0), (dict(x=L.LINE_X - 0.0085), 0.0),
            (dict(h=L.LINE_TOP - 0.001 + 1e-6), 1.0), (dict(h=L.LINE_TOP - 0.0015), 0.0),
            (dict(h=L.LINE_TOP + 0.002 - 1e-6), 1.0), (dict(h=L.LINE_TOP + 0.0025), 0.0),
            (dict(tilt=3.99), 1.0), (dict(tilt=4.5), 0.0), (dict(tilt=-4.5, axis="y"), 0.0),
            (dict(yaw=19.99), 1.0), (dict(yaw=20.5), 0.0), (dict(yaw=180.0), 1.0), (dict(yaw=-160.01), 1.0),
        ]
        for kw, want in cases:
            c = clamp(**kw)
            # the cover resting on this clamp's display, so only the seat decides
            v = cover(x=c.origin[0], h=float(S.to_world(c, [0, 0, L.DISPLAY_TOP])[2]) - L.CEILING)
            self.assertEqual(done(c, v).score, want, kw)

    def test_insulation_boundaries(self):
        cases = [
            (dict(h=COVER_H - 0.001 + 1e-6), 1.0), (dict(h=COVER_H - 0.0015), 0.5),
            (dict(h=COVER_H + 0.003 - 1e-6), 1.0), (dict(h=COVER_H + 0.0035), 0.5),
            (dict(tilt=9.99, axis="y"), 1.0), (dict(tilt=11.0, axis="y"), 0.5),
            (dict(y=0.090 + 0.016), 0.5),  # the block's end sticks out of the cover
        ]
        for kw, want in cases:
            v = cover(**kw)
            if "tilt" in kw:  # tilted about its centre's ceiling point, as it would rest on the display
                c0 = S.to_world(v, [0, 0, L.CEILING])
                v = cover(h=COVER_H - (c0[2] - (COVER_H + L.CEILING)), **kw)
            self.assertEqual(done(clamp(), v).score, want, kw)

    def test_moving_parts_are_not_at_rest(self):
        moving = [(clamp(y=0.090 + 0.0004 * i), cover(y=0.090 + 0.0004 * i)) for i in range(12)]  # 8 mm/s
        self.assertEqual(feed(moving)[1].score, 0.0)
        drifting = [(clamp(), cover(y=0.090 + 0.0004 * i)) for i in range(12)]
        self.assertEqual(feed(drifting)[1].score, 0.5)
        creep = [(clamp(y=0.090 + 0.00025 * i), cover(y=0.090 + 0.00025 * i)) for i in range(12)]  # 2.5 mm in 0.5 s
        self.assertEqual(feed(creep)[1].score, 1.0)

    def test_grippers_must_be_clear(self):
        top = L.TOP + COVER_H + L.COVER_HEIGHT
        ok = arms_with_left((L.LINE_X, 0.090, top + L.CLEAR_ABOVE + 0.013 + 1e-6))  # tips 40 mm above the cover
        self.assertEqual(done(clamp(), cover(), ok).score, 1.0)
        near = arms_with_left((L.LINE_X, 0.090, top + L.CLEAR_ABOVE + 0.0125))  # tips 39.5 mm above it
        self.assertEqual(done(clamp(), cover(), near).score, 0.5)  # clear of the clamp, not of the cover
        low = arms_with_left((L.LINE_X, 0.090, L.TOP + SEAT_H + L.DISPLAY_TOP + L.CLEAR_ABOVE + 0.0125))
        self.assertEqual(done(clamp(), COVER_START, low).score, 0.0)  # tips 39.5 mm above the clamp
        edge = L.LINE_X + L.BLOCK_S[0] / 2
        beside = arms_with_left((edge + L.CLEAR_SIDE - 0.001, 0.090, L.TOP + SEAT_H + 0.013))
        self.assertEqual(done(clamp(), COVER_START, beside).score, 0.0)  # 64 mm beside the clamp, at its height
        away = arms_with_left((edge + L.CLEAR_SIDE + 0.001, 0.090, L.TOP + SEAT_H + 0.013))
        self.assertEqual(done(clamp(), COVER_START, away).score, 0.5)
        corner = arms_with_left((edge + 0.0458, 0.090 + L.BLOCK_S[1] / 2 + 0.0458, L.TOP + SEAT_H))
        self.assertEqual(done(clamp(), COVER_START, corner).score, 0.0)  # 64.8 mm off its corner, diagonally
        corner = arms_with_left((edge + 0.0465, 0.090 + L.BLOCK_S[1] / 2 + 0.0465, L.TOP + SEAT_H))
        self.assertEqual(done(clamp(), COVER_START, corner).score, 0.5)  # 65.8 mm off diagonally
        right = {"left": READY["left"], "right": arm((L.LINE_X, 0.090, L.TOP + SEAT_H + 0.030))}
        self.assertEqual(done(clamp(), COVER_START, right).score, 0.0)  # either gripper counts

    def test_open_fingers_cannot_touch_a_part_when_clear(self):
        """Whenever the rule says a straight-down gripper is clear and its pad tips are below 4 cm above the part, no
        point of either open finger (inner face 47 mm and outer face 57 mm from the tool point, 40 mm wide, at any
        yaw) is over the part's box. Some gripper that the rule calls not clear does have a finger over it."""
        rng = np.random.default_rng(7)
        across = np.linspace(L.FINGER_INNER_OPEN, L.FINGER_OUTER_OPEN, 6)
        wide = np.linspace(-0.020, 0.020, 9)
        offs = np.array([(r, w) for r in np.concatenate([across, -across]) for w in wide])
        touched_when_not_clear = 0
        for part_state in (clamp(), cover()):
            lo, hi = np.asarray(part_state.low), np.asarray(part_state.high)
            for _ in range(2000):
                tcp = np.array([*rng.uniform([0.10, -0.05], [0.40, 0.23]), rng.uniform(L.TOP, hi[2] + 0.04)])
                yaw = rng.uniform(0, math.pi)
                rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                q = tcp[:2] + offs @ rot.T
                over = bool(np.any(np.all((q >= lo[:2]) & (q <= hi[:2]), axis=1)))
                if S.gripper_margin(arm(tcp), part_state) >= 0:
                    self.assertFalse(over, (tcp, yaw))
                elif over:
                    touched_when_not_clear += 1
        self.assertGreater(touched_when_not_clear, 0)

    def test_unreadable_arms(self):
        t, r = feed(hold(clamp(), cover()), arms=None)
        self.assertEqual(r.score, 0.0)  # in the simulator: not clear
        self.assertEqual(t.arms_unavailable, 12)
        _, r = feed(hold(clamp(), cover()), arms=None, offline=True)
        self.assertEqual(r.score, 1.0)  # offline (the checker's first call): clear


class ZeroConditions(unittest.TestCase):
    def test_on_the_liquid_line(self):
        c = clamp(x=L.LIQ_X, h=L.LIQ_TOP)
        self.assertEqual(done(c, cover(x=L.LIQ_X, h=L.LIQ_TOP + RIDE)).score, 0.0)
        self.assertEqual(done(c, COVER_START).score, 0.0)

    def test_on_the_coupling(self):
        y = L.COUPLING_END_Y - L.BLOCK_S[1] / 2 + 0.010  # its end 10 mm onto the coupling, riding on it
        coupling_top = L.LINE_Z + L.COUPLING_W / 2
        self.assertEqual(done(clamp(y=y, h=coupling_top), cover(y=y, h=coupling_top + RIDE)).score, 0.0)
        self.assertEqual(done(clamp(y=y, h=coupling_top, tilt=4.3, axis="x"), COVER_START).score, 0.0)
        # even at the bare copper's height, a block that overlaps the coupling is not on bare copper
        self.assertEqual(done(clamp(y=y), cover(y=y - 0.010)).score, 0.0)

    def test_on_the_foam(self):
        foam_top = L.LINE_Z + L.FOAM_W / 2
        self.assertEqual(done(clamp(y=-0.115, h=foam_top + L.LEG_LEN), COVER_START).score, 0.0)
        edge = L.FOAM_START_Y + L.BLOCK_S[1] / 2 - 0.005  # its end 5 mm onto the foam
        self.assertEqual(done(clamp(y=edge), cover(y=edge)).score, 0.0)

    def test_on_the_valve_table_or_pad(self):
        valve_top = L.SVALVE_C[2] + L.SVALVE_S[2] / 2
        self.assertEqual(done(clamp(y=L.SVALVE_C[1], h=valve_top + L.LEG_LEN), COVER_START).score, 0.0)
        self.assertEqual(done(clamp(x=0.20, y=0.0, h=L.LEG_LEN), cover(x=0.20, y=0.0, h=L.LEG_LEN + RIDE)
                              ).score, 0.0)
        self.assertEqual(done(CLAMP_START, cover(*L.CLAMP_START[:2], h=L.CLAMP_START[2] + RIDE)).score, 0.0)

    def test_tilted_upside_down_or_on_its_side(self):
        self.assertEqual(done(clamp(tilt=20), COVER_START).score, 0.0)
        self.assertEqual(done(clamp(h=L.LINE_TOP + L.DISPLAY_TOP, tilt=180), COVER_START).score, 0.0)
        self.assertEqual(done(clamp(h=L.LINE_TOP + 0.0225, tilt=90, axis="y"), COVER_START).score, 0.0)

    def test_cover_alone_and_nothing_done(self):
        self.assertEqual(done(CLAMP_START, cover(h=L.LINE_TOP - L.CEILING)).score, 0.0)
        self.assertEqual(done(CLAMP_START, COVER_START).score, 0.0)

    def test_still_held(self):
        held = arms_with_left((L.LINE_X, 0.090, L.TOP + SEAT_H + 0.017))
        self.assertEqual(done(clamp(), COVER_START, held).score, 0.0)


class Reachable(unittest.TestCase):
    """Each listed score is reached by a plausible trial: parts carried over and set down, grippers then away."""

    def trial(self, c_end, v_end):
        seq = []
        for i in range(1, 21):  # carried in from their pads, then set down
            f = i / 20
            c = clamp(*(np.array(L.CLAMP_START) * (1 - f) + np.array([c_end.origin[0], c_end.origin[1],
                                                                        c_end.origin[2] - L.TOP + 0.03]) * f))
            seq.append((c, COVER_START))
        seq += hold(c_end, COVER_START, 5)
        if v_end is not None:
            seq += [(c_end, cover(*(np.array(L.COVER_START) * (1 - i / 20) + np.array(
                [v_end.origin[0], v_end.origin[1], v_end.origin[2] - L.TOP + 0.03]) * i / 20))) for i in range(1, 21)]
            seq += hold(c_end, v_end, 12)
        else:
            seq += hold(c_end, COVER_START, 12)
        t = Tracker()
        held = arms_with_left((0.2, 0.15, L.TOP + 0.06))
        for c, v in seq[:-12]:
            t.update(c, v, held)  # a gripper at work: never clear
        return feed(seq[-12:], READY, t)[1].score

    def test_every_tier(self):
        far = L.VALVE_FACE_Y - 0.200
        self.assertEqual(self.trial(clamp(), cover()), 1.0)
        self.assertEqual(self.trial(clamp(), None), 0.5)
        self.assertEqual(self.trial(clamp(y=far), cover(y=far)), 0.25)
        self.assertEqual(self.trial(clamp(x=L.LIQ_X, h=L.LIQ_TOP), None), 0.0)


class Robustness(unittest.TestCase):
    def test_broken_states_score_zero_and_leave_the_tracker_alone(self):
        t, _ = feed(hold(clamp(), cover(), 12))
        trail = [list(map(tuple, t.trail[k])) for k in ("clamp", "cover")]
        good = clamp()
        broken = [
            SimpleNamespace(**{**vars(good), "origin": np.array([np.nan, 0.0, 0.0])}),
            SimpleNamespace(**{**vars(good), "high": np.array([np.inf, 0.0, 0.0])}),
            SimpleNamespace(**{**vars(good), "quat_xyzw": np.zeros(4)}),
            SimpleNamespace(**{k: v for k, v in vars(good).items() if k != "origin"}),
            None,
        ]
        for b in broken:
            self.assertEqual(t.update(b, cover(), READY).score, 0.0)
            self.assertEqual(t.update(clamp(), b, READY).score, 0.0)
            self.assertEqual(t.result(b, cover(), READY).score, 0.0)
        self.assertEqual(t.bad_steps, 2 * len(broken))
        self.assertEqual([list(map(tuple, t.trail[k])) for k in ("clamp", "cover")], trail)
        self.assertEqual(t.update(clamp(), cover(), READY).score, 1.0)

    def test_broken_arms_are_unreadable(self):
        bad = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3)), "right": READY["right"]}
        self.assertEqual(done(clamp(), cover(), bad).score, 0.0)
        self.assertEqual(done(clamp(), cover(), {"left": READY["left"]}).score, 0.0)

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(3)
        t = Tracker()
        for _ in range(400):
            c = clamp(*rng.uniform([0.0, -0.3, -0.05], [0.6, 0.4, 0.3]), yaw=rng.uniform(-180, 180),
                      tilt=rng.uniform(-180, 180), axis=rng.choice(["x", "y"]))
            v = cover(*rng.uniform([0.0, -0.3, -0.05], [0.6, 0.4, 0.3]), yaw=rng.uniform(-180, 180),
                      tilt=rng.uniform(-180, 180))
            r = t.update(c, v, READY)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(r.details)

    def test_score_is_idempotent(self):
        t, r = feed(hold(clamp(), cover()))
        self.assertEqual([t.result(clamp(), cover(), READY).score for _ in range(3)], [1.0] * 3)
        self.assertEqual(t.steps, 12)


class Reset(unittest.TestCase):
    def worked(self):
        t, r = feed(hold(clamp(), cover()))
        self.assertEqual(r.score, 1.0)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.worked()
        t.update(CLAMP_START, COVER_START, READY)
        self.assertEqual(t.fallback_resets, 1)
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.trail["clamp"]), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.worked()
        c = clamp(L.CLAMP_START[0] + 0.03, L.CLAMP_START[1] - 0.02, L.CLAMP_START[2], yaw=8)
        v = cover(L.COVER_START[0] - 0.025, L.COVER_START[1] + 0.03, L.COVER_START[2], yaw=-5)
        t.update(c, v, READY)
        self.assertEqual(t.fallback_resets, 1)

    def test_only_one_part_moved_still_resets(self):
        t, _ = feed(hold(clamp(), COVER_START))
        t.update(CLAMP_START, COVER_START, READY)
        self.assertEqual(t.fallback_resets, 1)

    def test_carrying_back_gradually_is_not_a_reset(self):
        t = self.worked()
        a, b = np.array([L.LINE_X, 0.090, SEAT_H]), np.array(L.CLAMP_START)
        for i in range(1, 41):
            f = i / 40
            t.update(clamp(*((1 - f) * a + f * b)), COVER_START if i > 20 else cover(), READY)
        self.assertEqual(t.fallback_resets, 0)
        self.assertGreater(t.steps, 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.worked()
        t.update(clamp(y=-0.040), cover(), READY)  # 13 cm along the line in one step: not back to the start
        t.update(clamp(x=0.15, y=0.0, h=L.LEG_LEN), COVER_START, READY)
        self.assertEqual(t.fallback_resets, 0)


class Exploits(unittest.TestCase):
    def test_holding_the_cover_down_scores_zero(self):
        press = arms_with_left((L.LINE_X, 0.090, L.TOP + COVER_H + L.COVER_HEIGHT + 0.013))
        self.assertEqual(done(clamp(), cover(), press).score, 0.0)

    def test_clamp_on_the_liquid_line_under_the_cover_scores_zero(self):
        c = clamp(x=L.LIQ_X, y=0.090, h=L.LIQ_TOP)
        self.assertEqual(done(c, cover(x=L.LIQ_X, h=L.LIQ_TOP + RIDE)).score, 0.0)

    def test_clamp_across_the_line_on_its_legs_scores_zero(self):
        c = clamp(h=L.LINE_TOP + L.LEG_LEN, yaw=90)  # turned 90 deg: its legs rest on the copper's top
        self.assertEqual(done(c, cover(h=L.LINE_TOP + L.LEG_LEN + RIDE)).score, 0.0)

    def test_clamp_beside_the_line_under_the_cover_scores_zero(self):
        c = clamp(x=L.LINE_X + 0.012, h=L.LINE_TOP + L.LEG_LEN, tilt=3)  # a leg on the copper
        self.assertEqual(done(c, cover(x=L.LINE_X + 0.012, h=L.LINE_TOP + L.LEG_LEN + RIDE)).score, 0.0)

    def test_gaming_the_height_band_on_the_coupling_scores_zero(self):
        # the coupling's top is 3 mm above the bare copper: 1 mm over the band, and off the bare copper anyway
        c = clamp(y=L.COUPLING_C[1] - 0.005, h=L.LINE_Z + L.COUPLING_W / 2)
        self.assertEqual(done(c, COVER_START).score, 0.0)

    def test_cover_on_the_clamp_on_its_pad_scores_zero(self):
        v = cover(*L.CLAMP_START[:2], h=L.CLAMP_START[2] + RIDE)
        self.assertEqual(done(CLAMP_START, v).score, 0.0)

    def test_open_finger_resting_on_the_cover_roof_scores_zero(self):
        # review exploit: the tool point and pad-tip point 42 mm beside the cover's box (the old rule's 4 cm passed)
        # while the open left finger (yaw -90 deg, closing along x) lies 5 mm onto the cover's roof
        roof = L.TOP + COVER_H + L.COVER_HEIGHT
        tcp = (L.LINE_X + L.COVER_W / 2 + L.FINGER_INNER_OPEN - 0.005, 0.090, roof + 0.013)
        finger_x = (tcp[0] - L.FINGER_OUTER_OPEN, tcp[0] - L.FINGER_INNER_OPEN)
        self.assertLess(finger_x[1], L.LINE_X + L.COVER_W / 2)  # the finger is over the roof
        a = arms_with_left(tcp)
        self.assertGreaterEqual(S.point_box_distance(a["left"].tips, cover().low, cover().high), 0.040)
        self.assertEqual(done(clamp(), cover(), a).score, 0.0)

    def test_open_finger_resting_on_the_clamp_block_scores_zero(self):
        # the same with the cover untouched: a finger 5 mm onto the clamp's block top (the old rule gave 0.5)
        block_top = L.TOP + SEAT_H + L.BLOCK_S[2]
        tcp = (L.LINE_X + L.BLOCK_S[0] / 2 + L.FINGER_INNER_OPEN - 0.005, 0.090, block_top + 0.013)
        a = arms_with_left(tcp)
        self.assertGreaterEqual(S.point_box_distance(a["left"].tips, clamp().low, clamp().high), 0.040)
        self.assertEqual(done(clamp(), COVER_START, a).score, 0.0)

    def test_still_holding_the_cover_keeps_the_clamp_credit(self):
        # the run limit with the clamp seated and let go and the cover still in the gripper, overhead: 0.5, not 0
        held = cover(h=0.150)
        tcp = (L.LINE_X, 0.090, L.TOP + 0.150 + 0.030)
        self.assertEqual(done(clamp(), held, arms_with_left(tcp)).score, 0.5)
        far = L.VALVE_FACE_Y - 0.200
        tcp_far = (L.LINE_X, far, L.TOP + 0.150 + 0.030)
        self.assertEqual(done(clamp(y=far), cover(y=far, h=0.150), arms_with_left(tcp_far)).score, 0.25)
        # a gripper hovering 3 cm beside the untouched cover's pad: 0.5
        hover = (L.COVER_START[0] - L.COVER_W / 2 - 0.030, L.COVER_START[1], L.TOP + 0.060)
        self.assertEqual(done(clamp(), COVER_START, arms_with_left(hover)).score, 0.5)
        # but holding the cover just above the clamp, or down on it, is not clear of the clamp: 0
        low = cover(h=COVER_H + 0.010)
        self.assertEqual(done(clamp(), low, arms_with_left((L.LINE_X, 0.090, L.TOP + COVER_H + 0.040))).score, 0.0)

    def test_full_marks_need_both_parts_let_go(self):
        # everything in place, a gripper clear of the clamp but still 5 cm beside the cover, low: 0.5, not 1.0
        x = L.LINE_X + L.BLOCK_S[0] / 2 + L.CLEAR_SIDE + 0.001
        a = arms_with_left((x, 0.090, L.TOP + 0.090))  # pad tips 7 mm above the liquid line, below the cover's top
        self.assertLess(x - (L.LINE_X + L.COVER_W / 2), L.CLEAR_SIDE)
        self.assertEqual(done(clamp(), cover(), a).score, 0.5)

    def test_near_is_judged_at_the_clamp_middle(self):
        # d 15.2 cm: the block's near end is 13.2 cm from the valve, but its middle is over 15 cm: 0.25
        y = L.VALVE_FACE_Y - 0.152
        c = clamp(y=y)
        self.assertAlmostEqual(L.VALVE_FACE_Y - S.block_corner_ys(c)[1], 0.132)
        self.assertEqual(done(c, cover(y=y)).score, 0.25)

    def test_ceiling_through_the_display_onto_the_block_is_not_insulated(self):
        # the ceiling on the block's top beside the display (4 mm under its top) is outside the -1..+3 mm band
        self.assertEqual(done(clamp(), cover(h=COVER_H - L.DISPLAY_S[2])).score, 0.5)

    def test_cover_upside_down_or_tossed_scores_half(self):
        self.assertEqual(done(clamp(), cover(h=COVER_H + 0.05, tilt=180)).score, 0.5)
        self.assertEqual(done(clamp(), cover(x=0.15, y=-0.05, h=0.0)).score, 0.5)

    def test_knocking_the_clamp_off_after_covering_scores_zero(self):
        t, r = feed(hold(clamp(), cover()))
        self.assertEqual(r.score, 1.0)
        _, r = feed(hold(clamp(x=0.20, y=0.06, h=L.LEG_LEN), cover(h=L.LINE_TOP - L.CEILING)), tracker=t)
        self.assertEqual(r.score, 0.0)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class OfflineModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = importlib.import_module("task")
        cls.mk = importlib.import_module("mechkit")
        cls.saddle = importlib.import_module("offline_saddle")

    def box_state(self, spec, x, y, bottom_h):
        """What the checker hands the hooks: the part's box, dropped with its bottom at bottom_h."""
        lo, hi = spec.local_bounds()
        return self.mk.state_from_poses(spec, (x, y, L.TOP + bottom_h - lo[2], 0, 0, 0, 1), source="offline")

    def test_clamp_dropped_on_the_line_goes_astride(self):
        c = self.box_state(self.task.CLAMP, L.LINE_X, 0.090, L.LINE_TOP)
        z, on = self.saddle.clamp_rest(c)
        self.assertAlmostEqual(z, L.LINE_TOP)
        self.assertEqual(on, "suction_line")
        play = L.LEG_INNER - L.LINE_HALF  # 7 mm a side
        z, on = self.saddle.clamp_rest(self.box_state(self.task.CLAMP, L.LINE_X + play - 0.0005, 0.090, L.LINE_TOP))
        self.assertAlmostEqual(z, L.LINE_TOP)
        z, on = self.saddle.clamp_rest(self.box_state(self.task.CLAMP, L.LINE_X + play + 0.0005, 0.090, L.LINE_TOP))
        self.assertIsNone(z)  # a leg would land on the copper
        z, on = self.saddle.clamp_rest(self.box_state(self.task.CLAMP, L.LINE_X, 0.090, L.LINE_TOP + 0.004))
        self.assertIsNone(z)  # still held above it
        z, on = self.saddle.clamp_rest(self.box_state(self.task.CLAMP, L.LIQ_X, 0.090, L.LIQ_TOP))
        self.assertEqual(on, "liquid_line")

    def test_cover_dropped_on_the_line_rests_on_the_display(self):
        c = self.box_state(self.task.CLAMP, L.LINE_X, 0.090, L.LINE_TOP)
        v = self.box_state(self.task.COVER, L.LINE_X, 0.090, L.LINE_TOP)
        c2, v2, info = self.saddle.correct(c, v, self.task.SPECS)
        self.assertEqual(info, {"clamp_on": "suction_line", "cover_on": "clamp_display"})
        self.assertAlmostEqual(S.ceiling_gap(c2, v2), 0.0, places=9)
        self.assertEqual(done(c2, v2).score, 1.0)
        # no clamp under it: the ceiling on the copper
        _, v3, info = self.saddle.correct(self.box_state(self.task.CLAMP, *L.CLAMP_START), v, self.task.SPECS)
        self.assertEqual(info["cover_on"], "suction_line")
        self.assertAlmostEqual(float(v3.origin[2]) - L.TOP, L.LINE_TOP - L.CEILING)

    def test_simulator_states_are_left_alone(self):
        c, v = clamp(), cover()
        c2, v2, info = self.saddle.correct(c, v, self.task.SPECS)
        self.assertIs(c2, c)
        self.assertIs(v2, v)
        self.assertIsNone(info["clamp_on"])


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hook and demo."""

    @classmethod
    def setUpClass(cls):
        cls.task = importlib.import_module("task")
        cls.rigstate = importlib.import_module("rigstate")

    def objects(self, c=None, v=None):
        return {"clamp": c if c is not None else self.task.CLAMP.rest_state(),
                "cover": v if v is not None else self.task.COVER.rest_state()}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in self.task.OBJECTS if o.mass is not None), ["clamp", "cover"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertGreaterEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertTrue(Tracker.at_start(self.task.CLAMP.rest_state(), self.task.COVER.rest_state()))
        lo, hi = self.task.CLAMP.local_bounds()
        self.assertAlmostEqual(lo[2], -L.LEG_LEN)
        self.assertAlmostEqual(hi[2], L.DISPLAY_TOP)
        lo, hi = self.task.COVER.local_bounds()
        self.assertAlmostEqual(hi[2], L.COVER_HEIGHT)

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        self.rigstate.offline_reset()
        self.rigstate.offline_feed({s: {"tcp": READY[s].tcp, "tips": READY[s].tips, "opening": 1.0} for s in READY})
        for _ in range(L.STILL_STEPS):
            self.assertFalse(self.task.success(self.objects(clamp(), cover())))  # not yet 0.5 s at rest
        self.assertTrue(self.task.success(self.objects(clamp(), cover())))
        self.assertEqual(self.task.score(self.objects(clamp(), cover())), 1.0)
        self.assertEqual(self.task.score(self.objects(clamp(), cover())), 1.0)  # idempotent
        self.assertEqual(tr.steps, L.STILL_STEPS + 1)
        self.rigstate.offline_reset()  # rigstate's reset hook: the tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(self.task.score(self.objects()), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left"})
        wps = plan["left"]
        for prev, w in zip(wps, wps[1:]):
            if prev.finger != w.finger:
                self.assertGreaterEqual(w.duration, 1.5, w.label)
        slow = [w for w in wps if "slowly" in w.label]
        self.assertEqual(len(slow), 4)
        self.assertTrue(all(w.duration >= 1.5 for w in slow))
        self.assertAlmostEqual(t.CLAMP_PLACE_Z - t.CLAMP_TCP_UP - L.LINE_TOP, 0.002)
        self.assertAlmostEqual(t.COVER_PLACE_Z - t.COVER_TCP_UP + L.CEILING - (L.LINE_TOP + L.DISPLAY_TOP), 0.002)
        self.assertAlmostEqual(t.CLAMP_GRIP, -(0.040 - 0.008) / 2)
        self.assertAlmostEqual(t.COVER_GRIP, -(0.077 - 0.008) / 2)
        self.assertAlmostEqual(S.d_along(clamp(y=t.PLACE_XY[1])), 0.100)
        last = wps[-1]
        self.assertTrue(last.finger == t.OPEN)
        tips = last.pos[2] - L.TOP - t.PAD_TIPS
        self.assertGreaterEqual(tips - (L.LINE_TOP + L.DISPLAY_TOP + L.ROOF_T), L.CLEAR_ABOVE + 0.015)
        self.assertEqual((t.CLAMP_TRIM, t.COVER_TRIM), ((0.0, 0.0), (0.0, 0.0)))
        places = {w.label: w.pos for w in wps}
        self.assertEqual(places["let go of the clamp"][:2], t.PLACE_XY)
        self.assertEqual(places["let go of the cover"][:2], t.PLACE_XY)
        old = t.CLAMP_TRIM, t.COVER_TRIM
        t.CLAMP_TRIM, t.COVER_TRIM = (0.002, -0.003), (-0.001, 0.004)
        try:
            places = {w.label: w.pos for w in t.demo()["left"]}
        finally:
            t.CLAMP_TRIM, t.COVER_TRIM = old
        np.testing.assert_allclose(places["let go of the clamp"][:2], (L.LINE_X + 0.002, 0.087))
        np.testing.assert_allclose(places["let go of the cover"][:2], (L.LINE_X - 0.001, 0.094))
        self.assertGreaterEqual(sum(w.duration for w in wps), 15.0)
        for key, val, want in (("DEMO_IDLE_ARM", "left", set()), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_is_the_same_everywhere(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        quoted = re.findall(r'--instruction "([^"]*)"', readme)
        self.assertGreaterEqual(len(quoted), 2)
        self.assertTrue(all(q == self.task.INSTRUCTION for q in quoted))

    def test_task_json_scores_and_limit(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertIn("demo: 1.0", meta["scores"])
        self.assertNotIn("media", meta)
        steps = int(re.match(r"([\d,]+) steps", meta["run_limit"]).group(1).replace(",", ""))
        demo_steps = sum(w.duration for w in self.task.demo()["left"]) * HZ
        self.assertTrue(3 * demo_steps <= steps <= 6 * demo_steps)
        self.assertIn("Possible scores: 0, 0.25, 0.5, 1.0.", meta["scoring"])


if __name__ == "__main__":
    unittest.main()
