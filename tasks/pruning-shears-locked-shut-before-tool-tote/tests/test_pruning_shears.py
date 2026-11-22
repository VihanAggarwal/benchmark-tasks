"""Offline unit tests for pruning-shears-locked-shut-before-tool-tote (unittest + pytest compatible)."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parent.parent
OFFLINE_DIR = REPO_ROOT / "tools" / "offline"
STUB_DIR = OFFLINE_DIR / "yam_stub"

for _p in (str(TASK_DIR), str(OFFLINE_DIR), str(STUB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import interlock as I  # noqa: E402
import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
import yamkin as K  # noqa: E402


def _arm_snap(tcp, opening=1.0, rot=None):
    tcp = np.asarray(tcp, dtype=float)
    if rot is None:
        r = mk.quat_to_matrix(K.DOWN_XYZW)
    else:
        arr = np.asarray(rot, dtype=float)
        r = arr if arr.shape == (3, 3) else mk.quat_to_matrix(arr)
    tips = tcp + r @ np.array([0.0, 0.0, K.PAD_TIP_PAST_TCP])
    return SimpleNamespace(tcp=tcp, tips=tips, opening=float(opening))


def _clear_arms():
    return {
        "left": _arm_snap((0.24, +0.25, L.TOP + 0.14), 1.0),
        "right": _arm_snap((0.24, -0.25, L.TOP + 0.14), 1.0),
    }


def _make_states(origin=L.SHEARS_ORIGIN, yaw=0.0, open_deg=L.THETA_OPEN_DEG, catch_m=0.0):
    pose = (*origin, *mk.yaw_quat(yaw))
    st = mk.state_from_poses(
        T.SHEARS_SPEC, pose, np.zeros(6),
        joints={S.JAW: L.q_jaw_of(open_deg), S.CATCH: catch_m},
        source="offline",
    )
    return {S.SHEARS: st}


def _sample_waypoints(wps, t: float, ready_pos: np.ndarray):
    prev_pos = np.asarray(ready_pos, dtype=float)
    prev_f = K.FINGER_OPEN
    prev_yaw = 0.0
    elapsed = 0.0
    for wp in wps:
        if t <= elapsed + wp.duration:
            u = K.smoothstep((t - elapsed) / max(wp.duration, 1e-9))
            pos = prev_pos + u * (np.asarray(wp.pos, dtype=float) - prev_pos)
            f = prev_f + u * (float(wp.finger) - prev_f)
            yaw = prev_yaw + u * (float(wp.yaw) - prev_yaw)
            return pos, f, yaw
        elapsed += wp.duration
        prev_pos = np.asarray(wp.pos, dtype=float)
        prev_f = float(wp.finger)
        prev_yaw = float(wp.yaw)
    return prev_pos, prev_f, prev_yaw


def _run_offline_plan(plan: dict, extra_hold_s: float = 3.0) -> tuple[float, bool, dict]:
    T._TRACKER.reset()
    T.INTERLOCK.reset()
    T.OFFLINE.reset()
    rigstate.offline_reset()

    dur = max((sum(w.duration for w in wps) for wps in plan.values()), default=0.0) + extra_hold_s
    steps = int(math.ceil(dur * L.CONTROL_HZ)) + 1

    ready_l = K.tcp_pose(K.READY, K.BASES["station"]["left"])
    ready_r = K.tcp_pose(K.READY, K.BASES["station"]["right"])

    for k in range(steps):
        t = k / L.CONTROL_HZ
        snaps = {}
        for side, ready in (("left", ready_l), ("right", ready_r)):
            if side in plan:
                pos, finger, yaw = _sample_waypoints(plan[side], t, ready[0])
                opening = float(np.clip(-2.0 * finger / rigstate.FULL_GAP, 0.0, 1.0))
                s = _arm_snap(pos, opening, K.grip_rotation(yaw))
            else:
                s = _arm_snap(ready[0], 1.0, ready[1])
            snaps[side] = {"tcp": s.tcp, "tips": s.tips, "opening": s.opening}
        rigstate.offline_feed(snaps)
        T.success({S.SHEARS: SimpleNamespace(source="offline")})

    return T._TRACKER.last.score, T._TRACKER.last.success, T._TRACKER.last.details


class TestPruningShears(unittest.TestCase):
    def test_fiskars_p26_and_tote_dimensions(self):
        self.assertAlmostEqual(L.LENGTH_BACK + L.LENGTH_FRONT, 0.208, places=6)
        self.assertAlmostEqual(L.BOLSTER_W_HALF * 2, 0.073, places=6)
        self.assertAlmostEqual(L.PIVOT_H, 0.022, places=6)
        self.assertAlmostEqual(L.BASE_MASS + L.UPPER_MASS + L.CATCH_MASS, 0.200, places=6)
        self.assertAlmostEqual(L.TOTE_IN_X[1] - L.TOTE_IN_X[0], 0.260, places=6)
        self.assertAlmostEqual(L.TOTE_IN_Y[1] - L.TOTE_IN_Y[0], 0.180, places=6)
        self.assertAlmostEqual(L.TOTE_RIM, 0.060, places=6)
        _, span_open, _ = L.handle_outer_span_at(L.GRIP_X_LOCAL, L.THETA_OPEN_DEG)
        _, span_closed, _ = L.handle_outer_span_at(L.GRIP_X_LOCAL, 0.0)
        self.assertGreaterEqual(span_open, 0.072)
        self.assertLessEqual(span_open, 0.082)
        self.assertLess(span_open, K.MAX_GAP - 0.012)
        self.assertAlmostEqual(span_closed, 0.038, places=4)

    def test_catch_tab_drawn_on_top_of_pivot_housing(self):
        # mechkit draws link shapes in the BASE frame: the tab must sit on the rail (z 22..32 mm), not in the boss.
        (shape,) = T.CATCH_SHAPES
        c = np.asarray(shape.center, dtype=float)
        self.assertAlmostEqual(c[0], L.CATCH_ANCHOR[0], places=6)
        self.assertAlmostEqual(c[1], L.CATCH_ANCHOR[1], places=6)
        self.assertAlmostEqual(c[2] - L.CATCH_SIZE[2] / 2, L.PIVOT_H, places=6)

    def test_interlock_requires_closed_handles_and_catch_travel(self):
        lock = I.Interlock()
        locked, jaw_t, catch_t = lock.step(L.q_jaw_of(22.0), 0.012)
        self.assertFalse(locked)
        self.assertEqual(jaw_t, 0.0)
        self.assertEqual(catch_t, 0.0)

        locked, jaw_t, catch_t = lock.step(L.q_jaw_of(2.0), 0.008)
        self.assertFalse(locked)
        self.assertEqual(jaw_t, 0.0)
        self.assertEqual(catch_t, 0.0)

        locked, jaw_t, catch_t = lock.step(L.q_jaw_of(3.0), 0.0095)
        self.assertTrue(locked)
        self.assertAlmostEqual(jaw_t, L.JAW_UPPER)
        self.assertAlmostEqual(catch_t, L.CATCH_TRAVEL)

        locked, _, _ = lock.step(L.q_jaw_of(0.5), 0.007)
        self.assertTrue(locked)

        locked, jaw_t, catch_t = lock.step(L.q_jaw_of(0.5), 0.005)
        self.assertFalse(locked)
        self.assertEqual(jaw_t, 0.0)
        self.assertEqual(catch_t, 0.0)

        before = lock.skipped
        lock.step(math.nan, 0.012)
        self.assertEqual(lock.skipped, before + 1)

    def test_scoring_tiers_progression(self):
        tr = S.Tracker()
        arms = _clear_arms()
        res = tr.update(_make_states(), arms, False)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

        res = tr.update(_make_states(open_deg=2.0, catch_m=0.0), arms, False)
        self.assertEqual(res.score, 0.25)

        res = tr.update(_make_states(open_deg=0.0, catch_m=0.012), arms, True)
        self.assertEqual(res.score, 0.50)

        tote_origin = (0.330, L.TOTE_CENTRE[1], L.TOP + L.TOTE_FLOOR)
        res = tr.update(_make_states(origin=tote_origin, open_deg=0.0, catch_m=0.012), arms, True)
        self.assertEqual(res.score, 0.75)
        self.assertFalse(res.success)

        for _ in range(L.WIN_STEPS):
            res = tr.update(_make_states(origin=tote_origin, open_deg=0.0, catch_m=0.012), arms, True)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_unlocked_above_tote_rim_is_not_hard_zero(self):
        tr = S.Tracker()
        arms = _clear_arms()
        above_tote = (0.330, L.TOTE_CENTRE[1], L.TOP + 0.080)
        res = tr.update(_make_states(origin=above_tote, open_deg=22.0, catch_m=0.0), arms, False)
        self.assertEqual(res.details["zero"], "")
        self.assertEqual(res.score, 0.0)

        in_tote_pos = (0.330, L.TOTE_CENTRE[1], L.TOP + L.TOTE_FLOOR)
        for _ in range(L.WIN_STEPS + 2):
            res = tr.update(_make_states(origin=in_tote_pos, open_deg=0.0, catch_m=0.012), arms, True)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_unlocked_below_tote_rim_latches_hard_zero(self):
        tr = S.Tracker()
        arms = _clear_arms()
        below_rim = (0.330, L.TOTE_CENTRE[1], L.TOP + 0.040)
        res = tr.update(_make_states(origin=below_rim, open_deg=15.0, catch_m=0.0), arms, False)
        self.assertEqual(res.score, 0.0)
        self.assertIn("tote rim", res.details["zero"])

        in_tote_pos = (0.330, L.TOTE_CENTRE[1], L.TOP + L.TOTE_FLOOR)
        for _ in range(L.WIN_STEPS + 2):
            res = tr.update(_make_states(origin=in_tote_pos, open_deg=0.0, catch_m=0.012), arms, True)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_open_blade_contact_latches_hard_zero_but_closed_does_not(self):
        tr = S.Tracker()
        c_w, _, _ = L.blade_edge_world_box(np.array(L.SHEARS_ORIGIN), np.eye(3), L.THETA_OPEN_DEG)
        arms_hit = {
            "left": _arm_snap((c_w[0], c_w[1], c_w[2] + K.PAD_TIP_PAST_TCP), 0.0),
            "right": _arm_snap((0.24, -0.25, L.TOP + 0.14), 1.0),
        }
        res = tr.update(_make_states(open_deg=L.THETA_OPEN_DEG), arms_hit, False)
        self.assertEqual(res.score, 0.0)
        self.assertIn("blade", res.details["zero"])

        tr2 = S.Tracker()
        c_closed, _, _ = L.blade_edge_world_box(np.array(L.SHEARS_ORIGIN), np.eye(3), 0.0)
        arms_near_closed = {
            "left": _arm_snap((c_closed[0], c_closed[1], c_closed[2] + K.PAD_TIP_PAST_TCP), 0.0),
            "right": _arm_snap((0.24, -0.25, L.TOP + 0.14), 1.0),
        }
        res2 = tr2.update(_make_states(open_deg=0.0, catch_m=0.012), arms_near_closed, True)
        self.assertEqual(res2.details["zero"], "")
        self.assertEqual(res2.score, 0.50)

    def test_unreadable_and_teleport_reset(self):
        tr = S.Tracker()
        res = tr.update({}, None, False)
        self.assertEqual(res.score, 0.0)
        self.assertIn("unreadable", res.details)

        in_tote_pos = (0.330, L.TOTE_CENTRE[1], L.TOP + L.TOTE_FLOOR)
        tr.update(_make_states(origin=in_tote_pos, open_deg=0.0, catch_m=0.012), _clear_arms(), True)
        before = tr.resets
        tr.update(_make_states(), _clear_arms(), False)
        self.assertEqual(tr.resets, before + 1)

    def test_offline_demo_scores_1(self):
        sc, ok, det = _run_offline_plan(T.plans())
        self.assertEqual(sc, 1.0)
        self.assertTrue(ok)
        self.assertEqual(det["zero"], "")

    def test_offline_hold_only_scores_0(self):
        sc, ok, _ = _run_offline_plan({})
        self.assertEqual(sc, 0.0)
        self.assertFalse(ok)

    def test_offline_one_arm_left_idle_scores_0(self):
        p = T.plans()
        p.pop("left", None)
        sc, ok, _ = _run_offline_plan(p)
        self.assertEqual(sc, 0.0)
        self.assertFalse(ok)

    def test_offline_one_arm_right_idle_scores_0(self):
        p = T.plans()
        p.pop("right", None)
        sc, ok, _ = _run_offline_plan(p)
        self.assertEqual(sc, 0.0)
        self.assertFalse(ok)

    def test_offline_counterfactual_variants(self):
        sc_naive, _, det_naive = _run_offline_plan(T.plans("naive_tote"))
        self.assertEqual(sc_naive, 0.0)
        self.assertIn("tote rim", det_naive["zero"])

        sc_open, _, _ = _run_offline_plan(T.plans("catch_open"))
        self.assertEqual(sc_open, 0.0)

        sc_blade, _, det_blade = _run_offline_plan(T.plans("blade_grab"))
        self.assertEqual(sc_blade, 0.0)
        self.assertIn("blade", det_blade["zero"])

        sc_no_tote, _, det_no_tote = _run_offline_plan(T.plans("no_tote"))
        self.assertEqual(sc_no_tote, 0.50)
        self.assertTrue(det_no_tote["locked"])
        self.assertFalse(det_no_tote["in_tote"])


if __name__ == "__main__":
    unittest.main()
