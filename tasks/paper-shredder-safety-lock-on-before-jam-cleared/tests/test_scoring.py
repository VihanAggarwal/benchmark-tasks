"""Unit tests for paper-shredder-safety-lock-on-before-jam-cleared scoring, rules, and geometry."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parent.parent
sys.path.insert(0, str(TASK_DIR))
_YAM_STUB = REPO_ROOT / "tools" / "offline" / "yam_stub"
if (_YAM_STUB / "yam_arena").is_dir() and str(_YAM_STUB) not in sys.path:
    sys.path.insert(0, str(_YAM_STUB))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring  # noqa: E402
import task  # noqa: E402
from offline_sim import _quat_from_rot, _rot_x, _rot_z  # noqa: E402


def _arm(tcp, open_frac=1.0):
    tcp = np.asarray(tcp, dtype=float)
    tips = tcp - np.array([0.0, 0.0, 0.013])
    return rigstate.Arm(tcp=tcp, tips=tips, opening=float(open_frac))


def _clear_arms(open_frac=1.0):
    return {
        "left": _arm((0.24, 0.36, L.TOP + 0.165), open_frac),
        "right": _arm((0.24, -0.42, L.TOP + 0.165), open_frac),
    }


def _shredder_state(rocker_deg=12.0, lock_mm=0.0, xy=L.SHRED_XY, z=L.TOP):
    # joint angle = 12 - rocker_deg (0 = +12 ON, 24 = -12 OFF)
    rq = math.radians(L.ROCKER_TILT_DEG - rocker_deg)
    lq = lock_mm / 1000.0
    return mk.state_from_poses(
        task.SHREDDER_SPEC,
        (xy[0], xy[1], z, 0.0, 0.0, 0.0, 1.0),
        np.zeros(6),
        joints={scoring.ROCKER_JOINT: rq, scoring.LOCK_JOINT: lq},
        source="sim",
    )


def _mailer_state(pos=None, rot=None):
    if pos is None:
        pos = (L.MAILER_ORIGIN[0], L.MAILER_ORIGIN[1], L.TOP + L.MAILER_ORIGIN[2])
    if rot is None:
        rot = np.eye(3)
    qx, qy, qz, qw = _quat_from_rot(rot)
    return mk.state_from_poses(
        task.MAILER_SPEC,
        (pos[0], pos[1], pos[2], qx, qy, qz, qw),
        np.zeros(6),
        source="sim",
    )


def _flat_in_tray_mailer(dx=0.0, dy=0.0, dz=0.0, tilt_extra_deg=0.0):
    rot = _rot_x(math.radians(90.0 + tilt_extra_deg)) @ _rot_z(math.pi / 2.0)
    pos = (L.TRAY_XY[0] + dx, L.TRAY_XY[1] + dy, L.TOP + L.TRAY_FLOOR + L.MAILER_SIZE[0] / 2.0 + dz)
    return _mailer_state(pos=pos, rot=rot)


# -- bistable rules ------------------------------------------------------------------------------------------------
def test_rocker_target_bistable():
    assert math.degrees(scoring.rocker_target(math.radians(0.0))) == pytest.approx(-L.ROCKER_PRELOAD_DEG)
    assert math.degrees(scoring.rocker_target(math.radians(11.9))) == pytest.approx(-L.ROCKER_PRELOAD_DEG)
    assert math.degrees(scoring.rocker_target(math.radians(12.0))) == pytest.approx(
        L.ROCKER_TRAVEL_DEG + L.ROCKER_PRELOAD_DEG
    )
    assert math.degrees(scoring.rocker_target(math.nan)) == pytest.approx(-L.ROCKER_PRELOAD_DEG)


def test_lock_target_bistable():
    assert scoring.lock_target(0.0) == pytest.approx(-L.LOCK_PRELOAD)
    assert scoring.lock_target(0.0099) == pytest.approx(-L.LOCK_PRELOAD)
    assert scoring.lock_target(0.010) == pytest.approx(L.LOCK_TRAVEL + L.LOCK_PRELOAD)
    assert scoring.lock_target(math.nan) == pytest.approx(-L.LOCK_PRELOAD)


# -- scoring ladder ------------------------------------------------------------------------------------------------
def test_start_state_scores_zero():
    tr = scoring.Tracker()
    res = tr.update({scoring.SHREDDER: _shredder_state(12.0, 0.0), scoring.MAILER: _mailer_state()}, _clear_arms())
    assert res.score == 0.0
    assert not res.success
    assert not res.details["latched"]


def test_single_switch_not_isolated():
    tr = scoring.Tracker()
    # Power OFF only -> 0.0
    r1 = tr.update({scoring.SHREDDER: _shredder_state(-12.0, 0.0), scoring.MAILER: _mailer_state()}, _clear_arms())
    assert r1.score == 0.0
    assert r1.details["power_off"] and not r1.details["lock_on"]
    # Safety Lock LOCKED only -> 0.0
    r2 = tr.update({scoring.SHREDDER: _shredder_state(12.0, 20.0), scoring.MAILER: _mailer_state()}, _clear_arms())
    assert r2.score == 0.0
    assert not r2.details["power_off"] and r2.details["lock_on"]


def test_full_ladder_to_success():
    tr = scoring.Tracker()
    sh_iso = _shredder_state(-10.0, 18.0)

    # 0.25: isolated, mailer still in slot
    r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: _mailer_state()}, _clear_arms())
    assert r.score == 0.25

    # 0.50: isolated, mailer extracted above deck
    ml_ext = _mailer_state(pos=(L.MAILER_X, L.MAILER_Y, L.TOP + L.SHRED_DECK_Z + 0.080))
    r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_ext}, _clear_arms())
    assert r.score == 0.50

    # 0.75: isolated, mailer flat in tray, not yet 20 steps
    ml_tray = _flat_in_tray_mailer()
    r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, _clear_arms())
    assert r.score == 0.75
    assert not r.success

    # Hold for 19 more steps -> 1.00 on step 20 of window
    for _ in range(L.WIN_STEPS - 2):
        r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, _clear_arms())
        assert r.score == 0.75
    r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, _clear_arms())
    assert r.score == 1.0
    assert r.success


# -- hard 0 safety latches -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("rocker_deg,lock_mm", [(12.0, 0.0), (-12.0, 0.0), (12.0, 20.0), (-5.0, 20.0), (-12.0, 14.0)])
def test_latch_on_unsafe_mailer_move(rocker_deg, lock_mm):
    tr = scoring.Tracker()
    sh_unsafe = _shredder_state(rocker_deg, lock_mm)
    # Step 1: at rest
    tr.update({scoring.SHREDDER: sh_unsafe, scoring.MAILER: _mailer_state()}, _clear_arms())
    # Step 2: mailer lifted 1.5 cm while unsafe -> hard 0 latch!
    ml_moved = _mailer_state(pos=(L.MAILER_X, L.MAILER_Y, L.TOP + L.MAILER_ORIGIN[2] + 0.015))
    r = tr.update({scoring.SHREDDER: sh_unsafe, scoring.MAILER: ml_moved}, _clear_arms())
    assert r.score == 0.0
    assert r.details["latched"]

    # Even if later isolated and placed cleanly in tray for 25 steps, stays latched at 0.0!
    sh_iso = _shredder_state(-12.0, 20.0)
    ml_tray = _flat_in_tray_mailer()
    for _ in range(25):
        r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, _clear_arms())
        assert r.score == 0.0
        assert not r.success


@pytest.mark.parametrize("rocker_deg,lock_mm", [(12.0, 0.0), (-12.0, 5.0), (0.0, 20.0)])
def test_latch_on_unsafe_gripper_touch(rocker_deg, lock_mm):
    tr = scoring.Tracker()
    sh_unsafe = _shredder_state(rocker_deg, lock_mm)
    # Right arm pad tips directly touching the protruding mailer
    arms_touch = {
        "left": _arm((0.24, 0.36, L.TOP + 0.165), 1.0),
        "right": _arm((L.MAILER_X, L.MAILER_Y, L.TOP + 0.175), 0.1),
    }
    r = tr.update({scoring.SHREDDER: sh_unsafe, scoring.MAILER: _mailer_state()}, arms_touch)
    assert r.score == 0.0
    assert "touched mailer" in r.details["latched"]


def test_latch_on_off_table():
    tr = scoring.Tracker()
    sh_iso = _shredder_state(-12.0, 20.0)
    ml_drop = _mailer_state(pos=(0.25, -0.80, L.TOP - 0.05))
    r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_drop}, _clear_arms())
    assert r.score == 0.0
    assert "left the table" in r.details["latched"]


# -- tray, release, and fail-closed checks -------------------------------------------------------------------------
def test_tilted_or_outside_tray_not_in_tray():
    tr = scoring.Tracker()
    sh_iso = _shredder_state(-12.0, 20.0)
    # Tilted 20 deg resting on rim -> not flat in tray (stays 0.50 extracted)
    r_tilt = tr.update(
        {scoring.SHREDDER: sh_iso, scoring.MAILER: _flat_in_tray_mailer(dz=0.03, tilt_extra_deg=20.0)},
        _clear_arms(),
    )
    assert r_tilt.score == 0.50
    # Corner overhanging tray rim -> not in tray
    r_out = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: _flat_in_tray_mailer(dx=0.05)}, _clear_arms())
    assert r_out.score == 0.50


def test_release_requires_open_and_4cm_clear():
    tr = scoring.Tracker()
    sh_iso = _shredder_state(-12.0, 20.0)
    ml_tray = _flat_in_tray_mailer()

    # Closed grippers -> not released (capped at 0.75)
    for _ in range(25):
        r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, _clear_arms(open_frac=0.2))
    assert r.score == 0.75

    # Open gripper hovering 2 cm over mailer -> not released
    close_arms = {
        "left": _arm((0.24, 0.36, L.TOP + 0.165), 1.0),
        "right": _arm((L.TRAY_XY[0], L.TRAY_XY[1], L.TOP + 0.040), 1.0),
    }
    for _ in range(25):
        r = tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, close_arms)
    assert r.score == 0.75


def test_fail_closed_unreadable_states():
    tr = scoring.Tracker()
    sh_iso = _shredder_state(-12.0, 20.0)
    ml_tray = _flat_in_tray_mailer()
    assert tr.update({}, _clear_arms()).score == 0.0
    assert tr.update({scoring.SHREDDER: sh_iso}, _clear_arms()).score == 0.0
    assert tr.update({scoring.SHREDDER: sh_iso, scoring.MAILER: ml_tray}, None).score == 0.75  # never 1.0 without arms


def test_kit_files_byte_identical():
    for fname in ("mechkit.py", "rigstate.py"):
        assert (TASK_DIR / fname).read_bytes() == (REPO_ROOT / "tools" / "kit" / fname).read_bytes()
