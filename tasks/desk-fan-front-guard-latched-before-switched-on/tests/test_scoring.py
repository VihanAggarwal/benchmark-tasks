"""Unit and integration tests for desk-fan-front-guard-latched-before-switched-on."""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
OFFLINE_DIR = REPO_ROOT / "tools" / "offline"
for _p in (OFFLINE_DIR, OFFLINE_DIR / "yam_stub"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import latchrule  # noqa: E402
import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_fan  # noqa: E402
import rigstate  # noqa: E402
import scoring  # noqa: E402
import task as T  # noqa: E402


def _make_arm(tcp, opening: float = 0.0) -> rigstate.Arm:
    tcp_arr = np.asarray(tcp, dtype=float)
    tips = tcp_arr - np.array([0.0, 0.0, getattr(rigstate, "TIP_OFFSET", 0.013)])
    return rigstate.Arm(tcp=tcp_arr, tips=tips, opening=opening)


def _clear_arms() -> dict[str, rigstate.Arm]:
    return {
        "left": _make_arm((0.24, 0.30, L.TOP + 0.15), opening=1.0),
        "right": _make_arm((0.24, -0.30, L.TOP + 0.15), opening=1.0),
    }


def _fan_state(open_deg: float, clip_deg: float, rocker_deg: float) -> dict:
    joints = {
        "guard": math.radians(L.GUARD_OPEN_DEG - open_deg),
        "clip": math.radians(clip_deg),
        "rocker": math.radians(rocker_deg),
    }
    return {
        scoring.FAN: mk.state_from_poses(
            T.FAN_SPEC,
            (L.FAN_ORIGIN[0], L.FAN_ORIGIN[1], L.TOP, 0.0, 0.0, 0.0, 1.0),
            np.zeros(6),
            joints=joints,
            source="test",
        )
    }


def _run_plan(plan: dict) -> tuple[float, bool, dict]:
    """Execute a scripted demo waypoint plan against OfflineFan + Tracker at 20 Hz."""
    off = offline_fan.OfflineFan()
    tr = scoring.Tracker()
    arms_pos = {
        "left": np.array([0.24, 0.30, L.TOP + 0.15]),
        "right": np.array([0.24, -0.30, L.TOP + 0.15]),
    }

    total_t = max((sum(w.duration for w in wps) for wps in plan.values()), default=2.0)
    n_steps = int(round(total_t * L.CONTROL_HZ)) + 1

    def sample(wps, t: float, default_pos):
        if not wps:
            return default_pos.copy(), 1.0
        elapsed = 0.0
        prev_pos = default_pos.copy()
        prev_f = -0.025
        for w in wps:
            if t <= elapsed + w.duration + 1e-9:
                alpha = 1.0 if w.duration <= 1e-9 else float(np.clip((t - elapsed) / w.duration, 0.0, 1.0))
                pos = (1.0 - alpha) * prev_pos + alpha * np.asarray(w.pos, dtype=float)
                f = (1.0 - alpha) * prev_f + alpha * float(w.finger)
                opening = float(np.clip(-f / 0.025, 0.0, 1.0))
                return pos, opening
            elapsed += w.duration
            prev_pos = np.asarray(w.pos, dtype=float)
            prev_f = float(w.finger)
        w = wps[-1]
        return np.asarray(w.pos, dtype=float), float(np.clip(-float(w.finger) / 0.025, 0.0, 1.0))

    res = scoring.Result()
    for step in range(n_steps):
        t = step * L.DT
        arms = {}
        for side in ("left", "right"):
            pos, opening = sample(plan.get(side, []), t, arms_pos[side])
            arms[side] = _make_arm(pos, opening=opening)
        off.step(arms)
        states = {
            scoring.FAN: mk.state_from_poses(
                T.FAN_SPEC,
                (L.FAN_ORIGIN[0], L.FAN_ORIGIN[1], L.TOP, 0.0, 0.0, 0.0, 1.0),
                np.zeros(6),
                joints=off.fan_joints(),
                source="offline",
            )
        }
        res = tr.update(states, arms, off.latch.latched)
        if res.success:
            return res.score, res.success, res.details
    return res.score, res.success, res.details


# -- 1. Layout & kit integrity --------------------------------------------------------------------------------------
def test_kit_files_byte_identical():
    for name in ("mechkit.py", "rigstate.py"):
        assert (TASK_DIR / name).read_bytes() == (REPO_ROOT / "tools" / "kit" / name).read_bytes()


def test_geometry_height_and_clearances():
    # Total fan height including open top clip stays <= 20 cm
    open_clip_top = L.clip_top_z(L.CLIP_LEN, 0.0)
    assert open_clip_top <= 0.200
    # Impeller cavity is strictly behind the guard hinge and outer face
    assert L.CAVITY_Y[1] < L.GUARD_HINGE_Y
    assert L.CAVITY_Y[1] < L.GUARD_SHUT_Y1 - 0.015
    # Clip pivot sits above the guard top and behind the front guard plane
    assert L.CLIP_PIVOT_Z > L.SHROUD_CZ + L.SHROUD_R + 0.003
    assert L.CLIP_PIVOT_Y < L.GUARD_HINGE_Y


def test_geometry_helpers():
    p_shut = (L.SHROUD_CX, L.GUARD_SHUT_Y1, L.SHROUD_CZ + L.SHROUD_R)
    p0 = L.guard_point(p_shut, 0.0)
    assert np.allclose(p0, p_shut)
    p22 = L.guard_point(p_shut, L.GUARD_OPEN_DEG)
    assert p22[1] > p_shut[1] + 0.050  # tilts outward toward +y
    assert L.rocker_tilt_deg(0.0) == pytest.approx(12.0)
    assert L.rocker_tilt_deg(20.0) == pytest.approx(-8.0)
    assert L.rocker_tilt_deg(24.0) == pytest.approx(-12.0)


# -- 2. Latch rules -------------------------------------------------------------------------------------------------
def test_clip_target_interlock():
    # Past centre but guard ajar -> springs back open (< 0)
    q_past = math.radians(L.CLIP_CENTRE_DEG + 5.0)
    assert latchrule.clip_target(q_past, open_deg=10.0) < 0.0
    # Past centre and guard seated <= 3 deg -> snaps closed (> travel)
    assert latchrule.clip_target(q_past, open_deg=2.0) > math.radians(L.CLIP_TRAVEL_DEG)
    # Before centre even if guard seated -> springs back open (< 0)
    q_before = math.radians(L.CLIP_CENTRE_DEG - 5.0)
    assert latchrule.clip_target(q_before, open_deg=0.0) < 0.0


def test_guard_and_rocker_targets():
    assert latchrule.guard_target(False) == 0.0
    assert latchrule.guard_target(True) > math.radians(L.GUARD_OPEN_DEG)
    assert latchrule.rocker_target(math.radians(5.0)) < 0.0
    assert latchrule.rocker_target(math.radians(15.0)) > math.radians(L.ROCKER_TRAVEL_DEG)


def test_latch_hysteresis_and_unreadable():
    la = latchrule.Latch()
    assert not la.update(22.0, 0.0)
    # Closed clip on ajar guard is not latched
    assert not la.update(10.0, 30.0)
    # Seated guard + clip >= CLIP_ON_DEG -> latched
    assert la.update(1.0, L.CLIP_ON_DEG)
    # Hysteresis band: stays latched above CLIP_OFF_DEG
    assert la.update(1.0, L.CLIP_CENTRE_DEG)
    # Drops below CLIP_OFF_DEG -> unlatched
    assert not la.update(1.0, L.CLIP_OFF_DEG)
    # NaN fails closed
    assert la.update(1.0, 30.0)
    assert not la.update(math.nan, 30.0)
    assert not la.update(1.0, None)


# -- 3. Scoring tiers & hard-0 safety gates -------------------------------------------------------------------------
def test_scoring_tiers_happy_path():
    tr = scoring.Tracker()
    arms = _clear_arms()
    # Start: ajar, off -> 0.0
    r = tr.update(_fan_state(22.0, 0.0, 0.0), arms, latched_guard=False)
    assert r.score == 0.0 and not r.success
    # Guard shut but unlatched, off -> 0.25
    r = tr.update(_fan_state(0.5, 0.0, 0.0), arms, latched_guard=False)
    assert r.score == 0.25
    # Guard shut and latched, off -> 0.5
    r = tr.update(_fan_state(0.5, 35.0, 0.0), arms, latched_guard=True)
    assert r.score == 0.5
    # Guard shut and latched, ON, hand close (not clear) -> 0.75
    near_arms = dict(arms)
    near_arms["right"] = _make_arm((L.ROCKER_XY[0], L.ROCKER_XY[1], L.TOP + 0.140), opening=0.0)
    r = tr.update(_fan_state(0.5, 35.0, 24.0), near_arms, latched_guard=True)
    assert r.score == 0.75 and not r.success
    # Hold clear for 1.0 s (20 steps) -> 1.0 and success
    for _ in range(L.WIN_STEPS - 1):
        r = tr.update(_fan_state(0.5, 35.0, 24.0), arms, latched_guard=True)
        assert r.score == 0.75
    r = tr.update(_fan_state(0.5, 35.0, 24.0), arms, latched_guard=True)
    assert r.score == 1.0 and r.success


def test_hard_zero_switch_on_unlatched():
    tr = scoring.Tracker()
    arms = _clear_arms()
    # Switch on while guard ajar -> latched 0
    r = tr.update(_fan_state(22.0, 0.0, 24.0), arms, latched_guard=False)
    assert r.score == 0.0 and "not latched" in r.details["zero"]
    # Even if later closed, latched, and clear -> stays 0.0 forever
    for _ in range(L.WIN_STEPS + 2):
        r = tr.update(_fan_state(0.0, 35.0, 24.0), arms, latched_guard=True)
        assert r.score == 0.0 and not r.success


def test_hard_zero_guard_opened_while_on():
    tr = scoring.Tracker()
    arms = _clear_arms()
    tr.update(_fan_state(0.0, 35.0, 0.0), arms, latched_guard=True)
    r = tr.update(_fan_state(0.0, 35.0, 24.0), arms, latched_guard=True)
    assert r.score == 0.75
    # Guard opens >= 5 deg while fan is ON -> latched 0
    r = tr.update(_fan_state(6.0, 0.0, 24.0), arms, latched_guard=False)
    assert r.score == 0.0 and "opened" in r.details["zero"]


def test_hard_zero_cavity_breach_while_open():
    tr = scoring.Tracker()
    arms = _clear_arms()
    # Pushing on the outer guard face (y = 0.055) while open is SAFE (outside cavity y <= 0.038)
    push_arms = dict(arms)
    push_arms["left"] = _make_arm((L.GUARD_PUSH_X, T.GUARD_HOLD_Y, L.TOP + T.GUARD_TIPS_Z + T.TIP), opening=0.0)
    r = tr.update(_fan_state(10.0, 0.0, 0.0), push_arms, latched_guard=False)
    assert r.details["zero"] == ""
    # Entering the impeller cavity (y = L.FY + 0.020) while open -> latched 0 immediately!
    breach_arms = dict(arms)
    breach_arms["left"] = _make_arm((L.SHROUD_CX, L.FY + 0.020, L.TOP + 0.100), opening=0.0)
    r = tr.update(_fan_state(15.0, 0.0, 0.0), breach_arms, latched_guard=False)
    assert r.score == 0.0 and "cavity" in r.details["zero"]


def test_unreadable_and_teleport_reset():
    tr = scoring.Tracker()
    arms = _clear_arms()
    r = tr.update({}, arms, latched_guard=False)
    assert r.score == 0.0 and "unreadable" in r.details
    # Progress to latched ON, then teleport back to start -> resets cleanly
    tr.update(_fan_state(0.0, 35.0, 24.0), arms, latched_guard=True)
    assert tr.resets == 0
    tr.update(_fan_state(22.0, 0.0, 0.0), arms, latched_guard=False)
    assert tr.resets == 1 and tr.zero == ""


# -- 4. Offline kinematics & scripted demo / counterfactuals --------------------------------------------------------
def test_demo_succeeds_offline():
    sc, ok, details = _run_plan(T.plans(""))
    assert sc == 1.0 and ok is True
    assert details["guard_latched"] is True and details["on"] is True and details["zero"] == ""


def test_hold_only_scores_zero():
    sc, ok, _ = _run_plan({})
    assert sc == 0.0 and not ok


def test_one_arm_runs_both_score_zero():
    plan_right_only = T.plans("")
    plan_right_only.pop("left")
    sc_r, ok_r, det_r = _run_plan(plan_right_only)
    assert sc_r == 0.0 and not ok_r and det_r["zero"] != ""

    plan_left_only = T.plans("")
    plan_left_only.pop("right")
    sc_l, ok_l, det_l = _run_plan(plan_left_only)
    assert sc_l == 0.0 and not ok_l and det_l["open_deg"] > 20.0


@pytest.mark.parametrize("variant,expected", [
    ("switch_first", 0.0),
    ("no_latch", 0.0),
    ("no_switch", 0.5),
    ("finger_in_cavity", 0.0),
])
def test_counterfactual_variants(variant: str, expected: float):
    sc, ok, _ = _run_plan(T.plans(variant))
    assert sc == expected
    assert ok is False
