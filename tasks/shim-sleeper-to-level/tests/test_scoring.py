"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import json
import math
import os
import pathlib
import re
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

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
            sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"


def sleeper_at(slope, pivot=None, dx=0.0, dy=0.0, roll_extra=0.0, yaw=0.0):
    """The sleeper resting with its underside on the line through (pivot, DIP) with ``slope``."""
    if pivot is None:
        pivot = L.FAR_Y1 if slope <= 0 else L.FAR_Y0
    cy, cz = L.box_centre(slope, pivot)
    q = mk.quat_from_ypr(yaw, roll_extra, math.atan(slope) - L.ROLL0)
    return mk.state_from_poses(T.SLEEPER_PART, (L.SL_X + dx, cy + dy, L.TOP + cz, *q), source="test")


def sleeper_on_shim(d):
    a = L.shim_bear(d)
    if a <= 0:
        return sleeper_at(L.START_SLOPE, L.START_PIVOT)
    y_c = L.SH_Y + L.SH_W / 2 if a < L.DIP else L.SH_Y - L.SH_W / 2
    return sleeper_at(*L.rest_on(a, y_c))


def shim_at(d, dy=0.0, z=0.0, yaw=0.0, pitch=0.0):
    """The shim on the low pad with its tip ``d`` past the sleeper's front face (as found the tip is 2 mm short)."""
    x = L.SL_FRONT_X + d - L.SH_L / 2
    q = mk.quat_from_ypr(yaw, pitch, 0.0)
    return mk.state_from_poses(T.SHIM_PART, (x, L.SH_Y + dy, L.TOP + L.LOW_H + z, *q), source="test")


def level_on(sl, shift=0.0, lift=0.0, yaw=0.0):
    """The level riding on the sleeper ``sl`` as at the start (``shift`` along its length, ``lift`` up)."""
    r = sl.rotation()
    s0 = np.array(T.SL_ORIGIN)
    o = np.asarray(sl.origin) + r @ (np.array(T.LV_ORIGIN) - s0 + np.array([0.0, shift, lift]))
    q = mk.matrix_to_quat(r @ mk.axis_angle_matrix((0.0, 0.0, 1.0), yaw))
    return mk.state_from_poses(T.LEVEL_PART, (*o, *q), source="test")


def scene(d=T.D_TARGET, sl=None, sh=None, lv=None):
    sl = sl if sl is not None else sleeper_on_shim(d)
    return {S.SLEEPER: sl, S.SHIM: sh if sh is not None else shim_at(d),
            S.LEVEL: lv if lv is not None else level_on(sl)}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.SL_W, 0.0381, places=4)  # 1-1/2 in
        self.assertAlmostEqual(L.SL_H, 0.0889, places=4)  # 3-1/2 in
        self.assertAlmostEqual(L.PAD_W, 0.0889, places=4)
        self.assertAlmostEqual(L.SH_L, 0.2032, places=4)  # 8 in
        self.assertAlmostEqual(L.SH_W, 0.0381, places=4)  # 1-1/2 in
        self.assertAlmostEqual(L.SH_BUTT, 0.009525, places=6)  # 3/8 in
        self.assertAlmostEqual(L.SH_TIP, 0.0015875, places=6)  # 1/16 in
        self.assertAlmostEqual(L.LV_L, 0.229, places=3)  # 9 in
        self.assertAlmostEqual(L.LV_MASS, 0.091, places=3)  # 0.2 lb
        self.assertAlmostEqual(L.SL_MASS, 0.868, places=3)  # 420 kg/m3

    def test_parts_spawn_where_they_rest(self):
        """The sleeper's underside touches the low pad's outer edge and the far pad's inner edge; the level and the
        shim lie on their supports; nothing interpenetrates."""
        sl = S.read(start())
        loc_low = S.sleeper_local(sl, np.array([[L.SL_X, L.LOW_Y1, 0.0]]))
        self.assertTrue(math.isfinite(loc_low[0, 0]))
        for y, h in ((L.LOW_Y1, L.LOW_H), (L.FAR_Y1, L.FAR_H)):
            z = L.LOW_H + L.h_line(y, L.START_SLOPE, L.START_PIVOT)
            self.assertAlmostEqual(z, h, places=9)
        self.assertAlmostEqual(T.SHIM_PART.rest_state().low[2] - L.TOP, L.LOW_H, places=6)
        r = S.read(start())
        self.assertTrue(S.level_on(r))
        self.assertAlmostEqual(S.end_diff(r), L.end_diff(L.START_SLOPE), places=9)

    def test_start_is_out_of_level(self):
        self.assertAlmostEqual(L.end_diff(L.START_SLOPE), -0.00989, places=5)

    def test_tip_cannot_start_under_the_loaded_end(self):
        """As found the underside over the shim is 0.4 to 1.0 mm off the pad: less than the 1.6 mm tip."""
        for y in (L.SH_Y - L.SH_W / 2, L.SH_Y + L.SH_W / 2):
            h = L.h_line(y, L.START_SLOPE, L.START_PIVOT)
            self.assertLess(h, L.SH_TIP - 0.0005)
            self.assertGreater(h, 0.0)

    def test_taper_window(self):
        self.assertAlmostEqual(L.D_LEVEL, 0.1511, places=3)
        self.assertLessEqual(abs(L.diff_for_d(T.D_TARGET)), 0.0002)
        ok = [d for d in np.arange(0.0, 0.21, 0.0005) if abs(L.diff_for_d(d)) <= L.LEVEL_TOL]
        self.assertAlmostEqual(min(ok), 0.137, delta=0.001)
        self.assertAlmostEqual(max(ok), 0.1675, delta=0.001)
        near = [d for d in np.arange(0.0, 0.21, 0.0005) if abs(L.diff_for_d(d)) <= L.NEAR_TOL]
        self.assertAlmostEqual(min(near), 0.1225, delta=0.001)
        self.assertAlmostEqual(max(near), 0.184, delta=0.001)
        self.assertAlmostEqual(L.diff_for_d(L.SH_L - 0.001), 0.00308, delta=0.0001)  # fully home: 3.1 mm high
        # GPU: the tip 117.0 mm past the front face left the sleeper 2.39 mm low (flush on the taper)
        self.assertAlmostEqual(L.diff_for_d(0.117), -0.00239, delta=0.0002)

    def test_trap_sizes(self):
        """The wrong ways miss level by 2.9 mm or more end to end (beyond the 2 mm of the 0.5 tier)."""
        for d in T.VARIANT_D.values():
            self.assertGreater(abs(L.diff_for_d(d)), 0.0029)

    def test_lift_force_and_lift_margin(self):
        """Lifting the end needs 4.7 N (under the 8 N limit); 12 mm of real lift (GPU: 10.4 mm gave 0.1 mm over the
        front edge of the level shim, the arm trailing 9.6 mm) clears the pushed shim's front edge by 1 mm; the 30 mm
        commanded gives about 20 mm with that trail."""
        w = (L.SL_MASS + L.LV_MASS) * L.G
        f = w * (L.SL_Y - L.FAR_Y0) / (T.GRIP_Y - L.FAR_Y0)
        self.assertLess(f, 5.0)
        self.assertGreater(f, 4.0)
        lifted = L.rest_on(T.H_START_GRIP + 0.012, T.GRIP_Y)
        gap = min(L.h_line(L.SH_Y - L.SH_W / 2, *lifted), L.h_line(L.SH_Y + L.SH_W / 2, *lifted))
        self.assertGreater(gap - L.shim_t(T.D_TARGET), 0.001)
        real = T.LIFT - 0.0096
        lifted = L.rest_on(T.H_START_GRIP + real, T.GRIP_Y)
        gap = min(L.h_line(L.SH_Y - L.SH_W / 2, *lifted), L.h_line(L.SH_Y + L.SH_W / 2, *lifted))
        self.assertGreater(gap - L.shim_t(T.D_TARGET), 0.005)

    def test_shim_balance(self):
        """As found the shim's centre of mass (by volume, as PhysX spreads it) is 40 mm or more onto the pad."""
        com = T.SHIM_PART.total_center_of_mass()
        x = L.SH_X0 + com[0]
        self.assertGreater(x - L.PAD_X0, 0.040)

    def test_shim_shapes_follow_the_taper(self):
        """No shape pokes up through the sloped top; the underside is flat at 0."""
        st = T.SHIM_PART.rest_state()
        for x in np.linspace(-L.SH_L / 2 + 0.002, L.SH_L / 2 - 0.002, 41):
            top = T.shim_top(x)
            above = st.to_world(np.array([[x, 0.0, top + 0.0002]]))
            self.assertGreater(float(st.distance(above)[0]), 0.0, x)
            inside = st.to_world(np.array([[x, 0.0, top - 0.0004]]))
            self.assertEqual(float(st.distance(inside)[0]), 0.0, x)
        lo, hi = T.SHIM_PART.local_bounds()
        self.assertGreaterEqual(lo[2], -1e-6)
        self.assertAlmostEqual(hi[2], L.SH_BUTT, delta=0.0002)

    def test_bubble_lines(self):
        """1 mm over 610 mm puts the bubble 2.5 mm off centre: still between the lines (inner edges 6.5 mm out)."""
        pitch = math.asin(L.LEVEL_TOL / L.SL_L)
        off = S.bubble_offset(pitch)
        self.assertAlmostEqual(off, 0.00246, delta=0.0001)
        inner = L.MARK_C - L.MARK_W / 2
        self.assertAlmostEqual(inner, 0.0065)
        self.assertLessEqual(off + L.BUBBLE_L / 2, inner)
        self.assertGreater(S.bubble_offset(math.asin(L.NEAR_TOL / L.SL_L)) + L.BUBBLE_L / 2, inner)
        self.assertEqual(S.bubble_offset(-0.1), -L.BUBBLE_MAX)


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(scene())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_tier_table(self):
        self.assertEqual(S.tier(True, True, True, True, True, True, 0.0), 0.0)
        self.assertEqual(S.tier(False, False, True, True, True, True, 0.0), 0.0)
        self.assertEqual(S.tier(False, True, True, True, True, True, 0.0009), 1.0)
        self.assertEqual(S.tier(False, True, True, True, True, True, -0.0019), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, True, 0.0021), 0.25)
        self.assertEqual(S.tier(False, True, True, True, False, True, 0.0), 0.25)
        self.assertEqual(S.tier(False, True, True, True, True, False, 0.0), 0.25)
        self.assertEqual(S.tier(False, True, True, False, True, True, 0.0), 0.0)
        self.assertEqual(S.tier(False, True, False, True, True, True, 0.0), 0.0)

    def test_start_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_window_edges(self):
        self.assertEqual(run(scene(0.138)).score, 1.0)
        self.assertEqual(run(scene(0.166)).score, 1.0)
        self.assertEqual(run(scene(0.128)).score, 0.5)
        self.assertEqual(run(scene(0.178)).score, 0.5)
        self.assertEqual(run(scene(0.115)).score, 0.25)
        self.assertEqual(run(scene(0.190)).score, 0.25)

    def test_wrong_ways(self):
        for name, d in T.VARIANT_D.items():
            self.assertEqual(run(scene(d)).score, 0.25, name)

    def test_tip_only_under_the_front_edge(self):
        """The tip 20 mm past the front face: under, but not across the width: 0.25."""
        r = run(scene(0.020))
        self.assertTrue(r.details["shim_under"])
        self.assertFalse(r.details["full_bearing"])
        self.assertEqual(r.score, 0.25)

    def test_shim_pushed_through(self):
        """Pushed right through, the butt past the front face: no full bearing."""
        st = scene(L.SH_L + 0.010)
        r = run(st)
        self.assertFalse(r.details["full_bearing"])
        self.assertEqual(r.score, 0.25)

    def test_level_knocked_to_the_table(self):
        sl = sleeper_on_shim(T.D_TARGET)
        lv = mk.state_from_poses(T.LEVEL_PART, (0.25, 0.20, L.TOP + 0.0225, 0, 0, 0, 1), source="test")
        r = run(scene(sl=sl, lv=lv))
        self.assertFalse(r.details["level_on"])
        self.assertEqual(r.score, 0.0)

    def test_level_across_the_sleeper(self):
        sl = sleeper_on_shim(T.D_TARGET)
        r = run(scene(sl=sl, lv=level_on(sl, yaw=math.radians(30))))
        self.assertFalse(r.details["level_on"])
        self.assertEqual(r.score, 0.25)

    def test_level_moved_along_still_counts(self):
        sl = sleeper_on_shim(T.D_TARGET)
        self.assertEqual(run(scene(sl=sl, lv=level_on(sl, shift=0.15))).score, 1.0)

    def test_level_held_above(self):
        sl = sleeper_on_shim(T.D_TARGET)
        self.assertFalse(run(scene(sl=sl, lv=level_on(sl, lift=0.010))).details["level_on"])

    def test_level_sleeper_without_shim_scores_zero(self):
        """Held level (or propped) with the shim left out: 0 (no shim under)."""
        sl = sleeper_at(0.0)
        r = run(scene(sl=sl, sh=shim_at(-0.002)))
        self.assertFalse(r.details["shim_under"])
        self.assertEqual(r.score, 0.0)

    def test_shim_on_top_of_the_sleeper_does_not_count(self):
        sh = shim_at(0.151, z=L.SL_H + 0.006)
        self.assertFalse(run(scene(sh=sh)).details["shim_under"])

    def test_sleeper_slid_along_its_pads_is_fine(self):
        """Harmless repositioning: the sleeper 30 mm back along x and the shim with it still scores 1.0."""
        sl = sleeper_on_shim(T.D_TARGET)
        sl2 = mk.state_from_poses(T.SLEEPER_PART, (sl.origin[0] + 0.03, sl.origin[1], sl.origin[2], *sl.quat_xyzw),
                                  source="test")
        sh = shim_at(T.D_TARGET + 0.03)
        lv = level_on(sl2)
        self.assertEqual(run(scene(sl=sl2, sh=sh, lv=lv)).score, 1.0)

    def test_sleeper_slid_off_one_pad_along_its_length(self):
        """Exploit found by the review copy: slide the sleeper 150 mm along its length so its far end no longer spans
        the far pad. Its axis line still crosses both pads, but the sleeper itself does not: no credit."""
        sl = sleeper_on_shim(T.D_TARGET)
        sl2 = mk.state_from_poses(T.SLEEPER_PART, (sl.origin[0], sl.origin[1] + 0.15, sl.origin[2], *sl.quat_xyzw),
                                  source="test")
        r = run(scene(sl=sl2, lv=level_on(sl2)))
        self.assertFalse(r.details["on_pads"])
        self.assertEqual(r.score, 0.0)

    def test_shim_sideways_behind_the_sleeper(self):
        """Exploit found by the review copy: the shim laid on the low pad behind the sleeper, turned 90 degrees.
        It never crosses under the sleeper, so it earns nothing."""
        sh = shim_at(T.D_TARGET + 0.08, yaw=math.radians(90))
        r = run(scene(sl=start()[S.SLEEPER], sh=sh, lv=level_on(start()[S.SLEEPER])))
        self.assertFalse(r.details["shim_under"])
        self.assertEqual(r.score, 0.0)

    def test_shim_yawed(self):
        """A shim turned 10 degrees with its tip still past the rear face: the reading is the sleeper's pose."""
        r = run(scene(sh=shim_at(T.D_TARGET, yaw=math.radians(10))))
        self.assertTrue(r.details["full_bearing"])
        self.assertEqual(r.score, 1.0)


class Latches(unittest.TestCase):
    def test_sleeper_off_the_pads(self):
        tr = S.Tracker()
        sl = mk.state_from_poses(T.SLEEPER_PART, (0.30, 0.25, L.TOP + 0.02, 0, 0, 0, 1), source="test")
        run(scene(sl=sl), tracker=tr)
        self.assertIsNotNone(tr.latch["F1"])
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_sleeper_on_its_face(self):
        tr = S.Tracker()
        run(scene(sl=sleeper_at(0.0, roll_extra=0.0, yaw=0.0)), tracker=tr)
        self.assertFalse(tr.latched)
        q = mk.quat_from_ypr(0.0, math.radians(60), L.ROLL0 * 0)
        sl = mk.state_from_poses(T.SLEEPER_PART, (0.30, -0.135, L.TOP + 0.12, *q), source="test")
        run(scene(sl=sl), tracker=tr, steps=1)
        self.assertIsNotNone(tr.latch["F1"])

    def test_level_fell_to_the_table(self):
        tr = S.Tracker()
        lv = mk.state_from_poses(T.LEVEL_PART, (0.45, 0.0, L.TOP + 0.0225, 0, 0, 0, 1), source="test")
        run(scene(lv=lv), tracker=tr)
        self.assertIsNotNone(tr.latch["F1"])

    def test_level_put_on_a_pad_is_not_a_fall(self):
        tr = S.Tracker()
        lv = mk.state_from_poses(T.LEVEL_PART, (0.20, L.FAR_Y, L.TOP + L.FAR_H + 0.0225, 0, 0, 0, 1), source="test")
        self.assertEqual(run(scene(lv=lv), tracker=tr).score, 0.25)
        self.assertFalse(tr.latched)

    def test_lifted_sleeper_is_not_a_fall(self):
        tr = S.Tracker()
        run(scene(sl=sleeper_at(0.03)), tracker=tr)
        self.assertFalse(tr.latched)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_sleeper(self):
        arms = {"left": arm((L.SL_X, T.GRIP_Y, L.TOP + T.LEVEL_TCP), 0.32), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=arms).score, 0.0)

    def test_hand_clear(self):
        arms = {"left": arm((0.22, 0.30, L.TOP + 0.195)), "right": arm((0.14, -0.20, L.TOP + 0.15))}
        self.assertEqual(run(scene(), arms=arms).score, 1.0)

    def test_parked_hands_are_clear(self):
        arms = {"left": arm((T.PARK_L[0], T.PARK_L[1], L.TOP + T.PARK_L[2])),
                "right": arm((T.PARK_R[0], T.PARK_R[1], L.TOP + T.PARK_R[2]))}
        self.assertEqual(run(scene(), arms=arms).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        near = {"left": arm((L.SL_X, T.GRIP_Y, L.TOP + T.LEVEL_TCP), 0.32), "right": FAR["right"]}
        run(scene(), arms=near, tracker=tr)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN).score, 1.0)

    def test_moving_sleeper(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            sl = sleeper_on_shim(T.D_TARGET)
            sl = mk.state_from_poses(T.SLEEPER_PART, (sl.origin[0], sl.origin[1], sl.origin[2] + 0.0005 * i,
                                                      *sl.quat_xyzw), source="test")
            res = tr.update(scene(sl=sl), FAR)
        self.assertEqual(res.score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update({S.SLEEPER: sleeper_on_shim(0.1)}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_state(self):
        bad = mk.state_from_poses(T.SHIM_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(sh=bad), FAR).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        lv = mk.state_from_poses(T.LEVEL_PART, (0.45, 0.0, L.TOP + 0.0225, 0, 0, 0, 1), source="test")
        run(scene(lv=lv), tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(300):
            sl = sleeper_at(rng.uniform(-0.03, 0.03), dx=rng.uniform(-0.05, 0.05), dy=rng.uniform(-0.05, 0.05),
                            yaw=rng.uniform(-0.3, 0.3), roll_extra=rng.uniform(-0.5, 0.5))
            sh = shim_at(rng.uniform(-0.05, 0.25), dy=rng.uniform(-0.05, 0.05), z=rng.uniform(-0.01, 0.1),
                         yaw=rng.uniform(-1, 1))
            lv = level_on(sl, shift=rng.uniform(-0.3, 0.3), lift=rng.uniform(-0.01, 0.05), yaw=rng.uniform(-2, 2))
            r = tr.update({S.SLEEPER: sl, S.SHIM: sh, S.LEVEL: lv}, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class BubbleRule(unittest.TestCase):
    def test_target_follows_the_level(self):
        self.assertAlmostEqual(T.bubble_target(T.LEVEL_PART.rest_state()), -L.BUBBLE_MAX)  # as found: at the low stop
        sl = sleeper_on_shim(T.D_TARGET)
        lv = level_on(sl)
        expect = L.VIAL_R * math.asin(L.diff_for_d(T.D_TARGET) / L.SL_L)
        self.assertAlmostEqual(T.bubble_target(lv), expect, places=5)
        hi = level_on(sleeper_on_shim(0.200))
        self.assertGreater(T.bubble_target(hi), 0.006)  # toward the high +y end, past the line

    def test_unreadable_level_gives_centre(self):
        self.assertEqual(T.bubble_target(None), 0.0)

    def test_bubble_link(self):
        j = T.LEVEL_PART.link("bubble")
        self.assertEqual(j.kind, "prismatic")
        self.assertAlmostEqual(j.upper, L.BUBBLE_MAX)
        self.assertAlmostEqual(j.lower, -L.BUBBLE_MAX)
        self.assertGreater(j.spring, 0.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["low_pad", "far_pad", "sleeper", "shim", "level"])
        self.assertIsNone(T.LOW_PAD.mass)
        self.assertIsNone(T.FAR_PAD.mass)
        self.assertAlmostEqual(T.SLEEPER_PART.mass, L.SL_MASS)
        self.assertAlmostEqual(T.SHIM_PART.mass, L.SH_MASS)
        self.assertAlmostEqual(T.LEVEL_PART.mass, L.LV_MASS)

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("lift", "push", "raise", "151", "mm", "step", "tap", "hold"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
                n_grips = sum(1 for x, y in zip(wps, wps[1:]) if abs(x.finger - y.finger) > 0.005)
                self.assertLessEqual(n_grips, 4)
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ + 80, run_limit)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "home", {"left", "right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def test_demo_timing(self):
        """The shim moves only while the end is up: the push starts after the lift ends and ends before the
        lowering starts."""
        plan = T.plans("")

        def span(wps, prefix):
            t = 0.0
            for w in wps:
                if w.label.startswith(prefix):
                    return t, t + w.duration
                t += w.duration
            raise KeyError(prefix)

        lift_end = span(plan["left"], "lift its low end")[1]
        push0, push1 = span(plan["right"], "push the shim")
        back1 = span(plan["right"], "back off")[1]
        lower0 = span(plan["left"], "lower it onto")[0]
        self.assertGreaterEqual(push0, lift_end)
        self.assertLessEqual(back1, lower0 + 1e-9)
        self.assertLessEqual(push1, lower0)
        self.assertAlmostEqual(max(sum(w.duration for w in wps) for wps in plan.values()), 24.6, places=3)

    def test_push_geometry(self):
        """The closed block comes down 12 mm short of the butt, pad tips 2.5 mm over the pad's top (7 mm up the butt
        face: more than 6 mm), and stops with the tip 151 mm past the front face."""
        self.assertAlmostEqual(L.SH_BUTT_X0 - (T.PUSH_START_X + T.PAD_FACE), 0.012)
        self.assertAlmostEqual(T.PUSH_TCP - T.PAD_TIP - L.LOW_H, 0.0025)
        self.assertGreater(L.SH_BUTT - (T.PUSH_TCP - T.PAD_TIP - L.LOW_H), 0.006)
        end_face = T.push_end_x(T.D_TARGET) + T.PAD_FACE - T.OVERTRAVEL
        self.assertAlmostEqual(end_face + L.SH_L - L.SL_FRONT_X, T.D_TARGET, places=9)
        self.assertLess(T.PUSH_START_X + T.PAD_FACE + 0.0135, L.PAD_X0)  # comes down in the air, before the pad
        # the closed block (27 mm wide) sits inside the 38 mm butt
        self.assertLess(rigstate.FINGER_HALF_WIDTH, L.SH_W / 2)

    def test_grip_geometry(self):
        self.assertAlmostEqual(-2 * T.GRIP_L, L.SL_W - 0.008)
        self.assertAlmostEqual(-T.OPEN_L - L.SL_W / 2, 0.012)
        self.assertLess(T.GRIP_Y + rigstate.FINGER_HALF_WIDTH, L.SL_Y1)  # the pads stay on the sleeper
        self.assertGreater(T.GRIP_Y - rigstate.FINGER_HALF_WIDTH, L.LOW_Y1 + 0.015)  # 15 mm clear of the low pad
        self.assertAlmostEqual(T.SL_TOP_AT_GRIP - (T.GRIP_TCP - T.PAD_TIP), T.GRIP_TIPS_DOWN)

    def test_hands_13_cm_apart_during_the_push(self):
        lx, ly = L.SL_X, T.GRIP_Y
        rx = T.push_end_x(T.D_TARGET)
        self.assertGreater(math.hypot(lx - rx, ly - L.SH_Y), 0.13)


class OfflineModel(unittest.TestCase):
    def _push(self, m, lifted_arm, x0, x1, n=60):
        for x in np.linspace(x0, x1, n):
            m.step({"left": lifted_arm, "right": arm((x, L.SH_Y, L.TOP + T.PUSH_TCP), 0.0)})

    def test_push_with_the_end_lifted(self):
        m = T.OfflineModel()
        for z in np.linspace(T.GRIP_TCP, T.GRIP_TCP + T.LIFT, 20):
            m.step({"left": arm((L.SL_X, T.GRIP_Y, L.TOP + z), 0.32), "right": FAR["right"]})
        up = arm((L.SL_X, T.GRIP_Y, L.TOP + T.GRIP_TCP + T.LIFT), 0.32)
        self._push(m, up, T.PUSH_START_X, T.push_end_x(T.D_TARGET))
        self.assertAlmostEqual(m.d(), T.D_TARGET + T.OVERTRAVEL, places=6)
        m.step({"left": FAR["left"], "right": FAR["right"]})
        r = S.read(m.states())
        self.assertAlmostEqual(S.end_diff(r), L.diff_for_d(m.d()), places=6)

    def test_push_without_the_lift_jams(self):
        m = T.OfflineModel()
        self._push(m, FAR["left"], T.PUSH_START_X, T.push_end_x(T.D_TARGET))
        self.assertLessEqual(m.d(), 0.0)

    def test_push_under_load_jams(self):
        """Once the sleeper rests on the shim, pushing it further does not move it (the model's jam; in the simulator
        it would drag the sleeper's end)."""
        m = T.OfflineModel()
        up = arm((L.SL_X, T.GRIP_Y, L.TOP + T.GRIP_TCP + T.LIFT), 0.32)
        for z in np.linspace(T.GRIP_TCP, T.GRIP_TCP + T.LIFT, 20):
            m.step({"left": arm((L.SL_X, T.GRIP_Y, L.TOP + z), 0.32), "right": FAR["right"]})
        self._push(m, up, T.PUSH_START_X, T.push_end_x(0.080))
        m.step(FAR)
        d0 = m.d()
        self._push(m, FAR["left"], T.push_end_x(0.080) - 0.005, T.push_end_x(0.120))
        self.assertAlmostEqual(m.d(), d0, delta=0.0002 / L.SH_SLOPE)

    def test_high_gripper_does_not_push(self):
        m = T.OfflineModel()
        for x in np.linspace(T.PUSH_START_X, 0.2, 30):
            m.step({"left": FAR["left"], "right": arm((x, L.SH_Y, L.TOP + 0.05), 0.0)})
        self.assertEqual(m.u, 0.0)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("88.9 mm across", "x 100 to 400 mm", "y -380 mm", "y -10 mm", "82.9 mm up", "6.0 mm lower",
                      "38.1 x 88.9 x 610 mm", "0.87 kg", "y +170 to -440 mm", "9.9 mm end to end",
                      "229 x 25 x 45 mm", "91 g", "8 in (203 mm)", "1-1/2 in (38.1 mm)", "3/8 in (9.5 mm)",
                      "1/16 in (1.6 mm)", "18 g", "2 mm short", "x 281 mm", "0.4 to 1.0 mm", "151 mm",
                      "137 to 167 mm", "122 to 184 mm", "3.1 mm high", "4.7 N", "13 mm apart", "8 mm long"):
            self.assertIn(token, setup)
        for token in ("610 mm", "5 mm", "15 degrees", "38 mm", "2 mm allowed", "3 mm above", "10 degrees", "30 mm",
                      "1 mm", "0.3 degrees", "20 mm", "30 degrees", "Possible scores: 0, 0.25, 0.5, 1.0",
                      "within 2 mm", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "900 steps (45 s at 20 Hz)")
        self.assertAlmostEqual(L.LOW_Y, -0.010)
        self.assertAlmostEqual(L.SL_FRONT_X, 0.281, places=3)
        scores = d.get("scores", "")  # before the GPU run: empty or TBD; after it: the real Dream-1 run and demo 1.0
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_rule_and_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("Declared rule", "bubble", "1.5 m", "est", "6 mm shorter", "eased edges"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Carpenters", "47-2031", "889,700", "BLS"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertLess(len(d["instruction"]), 300)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "rigstate.py", "mechkit.py",
                  "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("—|–|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
