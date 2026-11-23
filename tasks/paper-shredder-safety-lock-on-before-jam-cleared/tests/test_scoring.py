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


def _shredder_state(ctrl_mm=0.0, rocker_deg=12.0, xy=L.SHRED_XY, z=L.TOP):
    # rocker joint angle = 12 - rocker_deg (0 = +12 ON, 24 = -12 OFF); control knob 0 Auto-On, 10 O, 20 Reverse
    rq = math.radians(L.ROCKER_TILT_DEG - rocker_deg)
    return mk.state_from_poses(
        task.SHREDDER_SPEC,
        (xy[0], xy[1], z, 0.0, 0.0, 0.0, 1.0),
        np.zeros(6),
        joints={scoring.ROCKER_JOINT: rq, scoring.CTRL_JOINT: ctrl_mm / 1000.0},
        source="sim",
    )


def _outlet_state(plug_mm=0.0):
    return mk.state_from_poses(
        task.OUTLET_SPEC, (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0), np.zeros(6),
        joints={scoring.PLUG_JOINT: plug_mm / 1000.0}, source="sim",
    )


def _states(ctrl_mm=0.0, plug_mm=0.0, mailer=None, rocker_deg=12.0):
    out = {scoring.SHREDDER: _shredder_state(ctrl_mm, rocker_deg), scoring.OUTLET: _outlet_state(plug_mm)}
    out[scoring.MAILER] = _mailer_state() if mailer is None else mailer
    return out


SAFE = {"ctrl_mm": 10.0, "plug_mm": 35.0}


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


# -- switch rules --------------------------------------------------------------------------------------------------
def test_rocker_target_bistable():
    assert math.degrees(scoring.rocker_target(math.radians(0.0))) == pytest.approx(-L.ROCKER_PRELOAD_DEG)
    assert math.degrees(scoring.rocker_target(math.radians(11.9))) == pytest.approx(-L.ROCKER_PRELOAD_DEG)
    assert math.degrees(scoring.rocker_target(math.radians(12.0))) == pytest.approx(
        L.ROCKER_TRAVEL_DEG + L.ROCKER_PRELOAD_DEG
    )
    assert math.degrees(scoring.rocker_target(math.nan)) == pytest.approx(-L.ROCKER_PRELOAD_DEG)


def test_control_target_three_position():
    assert scoring.control_target(0.0) == pytest.approx(-L.CTRL_PRELOAD)
    assert scoring.control_target(0.0039) == pytest.approx(-L.CTRL_PRELOAD)
    assert scoring.control_target(0.004) == pytest.approx(L.CTRL_OFF_POS)
    assert scoring.control_target(0.0159) == pytest.approx(L.CTRL_OFF_POS)
    assert scoring.control_target(0.016) == pytest.approx(L.CTRL_TRAVEL + L.CTRL_PRELOAD)
    assert scoring.control_target(math.nan) == pytest.approx(-L.CTRL_PRELOAD)


def test_control_position_bands():
    assert scoring.control_position(0.0) == "auto"
    assert scoring.control_position(0.010) == "off"
    assert scoring.control_position(0.007) == "off" and scoring.control_position(0.013) == "off"
    assert scoring.control_position(0.005) == "between"
    assert scoring.control_position(0.020) == "reverse"
    assert scoring.control_position(math.nan) == "unreadable"


# -- scoring ladder ------------------------------------------------------------------------------------------------
def test_start_state_scores_zero():
    tr = scoring.Tracker()
    res = tr.update(_states(), _clear_arms())
    assert res.score == 0.0
    assert not res.success
    assert not res.details["latched"]
    assert not res.details["unplugged"] and res.details["control"] == "auto"


@pytest.mark.parametrize("ctrl_mm,plug_mm,rocker_deg", [
    (10.0, 0.0, 12.0),    # switch at O, still plugged in
    (10.0, 17.0, -12.0),  # switch at O, rocker OFF, plug only 17 mm out (blades still in)
    (0.0, 35.0, 12.0),    # unplugged, switch left at Auto-On
    (20.0, 35.0, 12.0),   # unplugged, switch at Reverse
    (0.0, 35.0, -12.0),   # unplugged, rocker OFF, switch at Auto-On: the rocker does not stand in for O
    (0.0, 0.0, -12.0),    # rocker OFF only
])
def test_partial_isolation_is_not_safe(ctrl_mm, plug_mm, rocker_deg):
    tr = scoring.Tracker()
    r = tr.update(_states(ctrl_mm, plug_mm, rocker_deg=rocker_deg), _clear_arms())
    assert r.score == 0.0
    assert not r.details["isolated"]


def test_full_ladder_to_success():
    tr = scoring.Tracker()

    # 0.25: safe, mailer still in slot (rocker left ON: not scored)
    r = tr.update(_states(**SAFE), _clear_arms())
    assert r.score == 0.25
    assert r.details["rocker_deg"] == pytest.approx(12.0)

    # 0.50: safe, mailer extracted above deck
    ml_ext = _mailer_state(pos=(L.MAILER_X, L.MAILER_Y, L.TOP + L.SHRED_DECK_Z + 0.080))
    r = tr.update(_states(**SAFE, mailer=ml_ext), _clear_arms())
    assert r.score == 0.50

    # 0.75: safe, mailer flat in tray, not yet 20 steps
    ml_tray = _flat_in_tray_mailer()
    r = tr.update(_states(**SAFE, mailer=ml_tray), _clear_arms())
    assert r.score == 0.75
    assert not r.success

    for _ in range(L.WIN_STEPS - 2):
        r = tr.update(_states(**SAFE, mailer=ml_tray), _clear_arms())
        assert r.score == 0.75
    r = tr.update(_states(**SAFE, mailer=ml_tray), _clear_arms())
    assert r.score == 1.0
    assert r.success


def _turned_shredder_and_seated_mailer(rot, ctrl_mm=10.0):
    """Shredder turned by ``rot`` about its base origin, with the mailer carried rigidly (still seated in the slot)."""
    rot = np.asarray(rot, dtype=float)
    o = np.array([L.SHRED_XY[0], L.SHRED_XY[1], L.TOP])
    qx, qy, qz, qw = _quat_from_rot(rot)
    rq = math.radians(L.ROCKER_TILT_DEG - 12.0)
    sh = mk.state_from_poses(
        task.SHREDDER_SPEC, (o[0], o[1], o[2], qx, qy, qz, qw), np.zeros(6),
        joints={scoring.ROCKER_JOINT: rq, scoring.CTRL_JOINT: ctrl_mm / 1000.0}, source="sim",
    )
    m0 = np.array([L.MAILER_ORIGIN[0], L.MAILER_ORIGIN[1], L.TOP + L.MAILER_ORIGIN[2]])
    ml = _mailer_state(pos=o + rot @ (m0 - o), rot=rot)
    return sh, ml


@pytest.mark.parametrize("rot", [
    _rot_z(math.radians(90.0)),
    _rot_z(math.radians(-35.0)),
    _rot_z(math.radians(180.0)),
    _rot_x(math.radians(25.0)),
])
def test_turned_shredder_with_mailer_seated_is_not_extracted(rot):
    sh, ml = _turned_shredder_and_seated_mailer(rot)
    corners = L.mailer_corners(np.asarray(ml.origin, dtype=float), np.asarray(ml.rotation(), dtype=float))
    # Still seated in the shredder's own frame
    assert scoring.mailer_in_slot(corners, sh.origin, sh.rotation())
    states = {scoring.SHREDDER: sh, scoring.OUTLET: _outlet_state(35.0), scoring.MAILER: ml}
    r = scoring.Tracker().update(states, _clear_arms())
    assert not r.details["extracted"]
    assert r.score == 0.25


def test_yaw_turn_would_fool_a_world_axis_corridor():
    """The regression: a 90 deg turn moves the seated mailer out of the world-axis corridor but not out of the slot."""
    sh, ml = _turned_shredder_and_seated_mailer(_rot_z(math.radians(90.0)))
    corners = L.mailer_corners(np.asarray(ml.origin, dtype=float), np.asarray(ml.rotation(), dtype=float))
    assert not scoring.mailer_in_slot(corners, sh.origin)  # identity frame: would have been 'extracted'
    assert scoring.mailer_in_slot(corners, sh.origin, sh.rotation())


def test_turned_shredder_with_mailer_lifted_out_is_extracted():
    rot = _rot_z(math.radians(90.0))
    sh, _ = _turned_shredder_and_seated_mailer(rot)
    o = np.array([L.SHRED_XY[0], L.SHRED_XY[1], L.TOP])
    m0 = np.array([L.MAILER_ORIGIN[0], L.MAILER_ORIGIN[1], L.TOP + L.MAILER_ORIGIN[2] + 0.080])
    ml = _mailer_state(pos=o + rot @ (m0 - o), rot=rot)
    states = {scoring.SHREDDER: sh, scoring.OUTLET: _outlet_state(35.0), scoring.MAILER: ml}
    r = scoring.Tracker().update(states, _clear_arms())
    assert r.details["extracted"]
    assert r.score == 0.50


def test_replugging_at_the_end_is_not_success():
    tr = scoring.Tracker()
    ml_tray = _flat_in_tray_mailer()
    for _ in range(L.WIN_STEPS + 2):
        r = tr.update(_states(**SAFE, mailer=ml_tray), _clear_arms())
    assert r.score == 1.0
    r = tr.update(_states(ctrl_mm=10.0, plug_mm=0.0, mailer=ml_tray), _clear_arms())
    assert r.score == 0.0


# -- hard 0 safety latches -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("ctrl_mm,plug_mm", [(0.0, 0.0), (10.0, 0.0), (0.0, 35.0), (20.0, 35.0), (10.0, 12.0)])
def test_latch_on_unsafe_mailer_move(ctrl_mm, plug_mm):
    tr = scoring.Tracker()
    tr.update(_states(ctrl_mm, plug_mm), _clear_arms())
    ml_moved = _mailer_state(pos=(L.MAILER_X, L.MAILER_Y, L.TOP + L.MAILER_ORIGIN[2] + 0.015))
    r = tr.update(_states(ctrl_mm, plug_mm, mailer=ml_moved), _clear_arms())
    assert r.score == 0.0
    assert r.details["latched"]

    # Even if later made safe and placed cleanly in tray for 25 steps, stays latched at 0.0
    ml_tray = _flat_in_tray_mailer()
    for _ in range(25):
        r = tr.update(_states(**SAFE, mailer=ml_tray), _clear_arms())
        assert r.score == 0.0
        assert not r.success


@pytest.mark.parametrize("ctrl_mm,plug_mm", [(0.0, 0.0), (10.0, 0.0), (0.0, 35.0)])
def test_latch_on_unsafe_gripper_touch(ctrl_mm, plug_mm):
    tr = scoring.Tracker()
    arms_touch = {
        "left": _arm((0.24, 0.36, L.TOP + 0.165), 1.0),
        "right": _arm((L.MAILER_X, L.MAILER_Y, L.TOP + 0.175), 0.1),
    }
    r = tr.update(_states(ctrl_mm, plug_mm), arms_touch)
    assert r.score == 0.0
    assert "touched mailer" in r.details["latched"]


def test_touch_after_isolation_is_allowed():
    tr = scoring.Tracker()
    arms_touch = {
        "left": _arm((0.24, 0.36, L.TOP + 0.165), 1.0),
        "right": _arm((L.MAILER_X, L.MAILER_Y, L.TOP + 0.175), 0.1),
    }
    r = tr.update(_states(**SAFE), arms_touch)
    assert r.score == 0.25
    assert not r.details["latched"]


def test_latch_on_off_table():
    tr = scoring.Tracker()
    ml_drop = _mailer_state(pos=(0.25, -0.80, L.TOP - 0.05))
    r = tr.update(_states(**SAFE, mailer=ml_drop), _clear_arms())
    assert r.score == 0.0
    assert "left the table" in r.details["latched"]


# -- tray, release, and fail-closed checks -------------------------------------------------------------------------
def test_tilted_or_outside_tray_not_in_tray():
    tr = scoring.Tracker()
    r_tilt = tr.update(_states(**SAFE, mailer=_flat_in_tray_mailer(dz=0.03, tilt_extra_deg=20.0)), _clear_arms())
    assert r_tilt.score == 0.50
    r_out = tr.update(_states(**SAFE, mailer=_flat_in_tray_mailer(dx=0.05)), _clear_arms())
    assert r_out.score == 0.50


def test_release_requires_open_and_4cm_clear():
    tr = scoring.Tracker()
    ml_tray = _flat_in_tray_mailer()
    for _ in range(25):
        r = tr.update(_states(**SAFE, mailer=ml_tray), _clear_arms(open_frac=0.2))
    assert r.score == 0.75

    close_arms = {
        "left": _arm((0.24, 0.36, L.TOP + 0.165), 1.0),
        "right": _arm((L.TRAY_XY[0], L.TRAY_XY[1], L.TOP + 0.040), 1.0),
    }
    for _ in range(25):
        r = tr.update(_states(**SAFE, mailer=ml_tray), close_arms)
    assert r.score == 0.75


def test_fail_closed_unreadable_states():
    tr = scoring.Tracker()
    ml_tray = _flat_in_tray_mailer()
    assert tr.update({}, _clear_arms()).score == 0.0
    st = _states(**SAFE, mailer=ml_tray)
    assert tr.update({scoring.SHREDDER: st[scoring.SHREDDER]}, _clear_arms()).score == 0.0
    # outlet missing: the plug counts as plugged in
    no_outlet = {k: v for k, v in st.items() if k != scoring.OUTLET}
    r = tr.update(no_outlet, _clear_arms())
    assert r.score == 0.0 and not r.details["unplugged"]
    assert tr.update(st, None).score == 0.75  # never 1.0 without arms


# -- offline model ---------------------------------------------------------------------------------------------------
def test_offline_plug_pull_and_switch_push():
    from offline_sim import OfflineSim

    sim = OfflineSim()
    pc = sim.plug_centre()
    tcp = np.array([pc[0], task.PLUG_TCP_Y, L.TOP + task.PLUG_TCP_Z])
    grip = (task.PLUG_GRIP / rigstate.FINGER_OPEN)
    for dy in np.linspace(0.0, task.PLUG_PULL, 30):
        sim.step({"left": _arm(tcp + np.array([0.0, dy, 0.0]), grip), "right": _arm((0.24, -0.42, L.TOP + 0.165))})
    assert sim.plug_out == pytest.approx(L.PLUG_UPPER)

    # Pinch-and-drag: pads across x onto the 24 mm knob, drag +21 mm (into Reverse offline: a rigid arm), back to
    # +6 mm, let go -> the detent centres the knob on O
    sim = OfflineSim()
    kx, ky = L.CTRL_XY
    z = L.TOP + task.KNOB_TIPS + 0.013
    left = _arm((0.24, 0.36, L.TOP + 0.165))
    pinch = L.CTRL_KNOB[0] / rigstate.FULL_GAP  # pads stopped on the knob's x faces
    sim.step({"left": left, "right": _arm((kx, ky, z), task.KNOB_OPEN / rigstate.FINGER_OPEN)})
    sim.step({"left": left, "right": _arm((kx, ky, z), pinch)})
    for dy in np.linspace(0.0, task.KNOB_DRAG, 20):
        sim.step({"left": left, "right": _arm((kx, ky + dy, z), pinch)})
    assert scoring.control_position(sim.ctrl_m) == "reverse"
    for dy in np.linspace(task.KNOB_DRAG, task.KNOB_BACK, 20):
        sim.step({"left": left, "right": _arm((kx, ky + dy, z), pinch)})
    sim.step({"left": left, "right": _arm((kx, ky + task.KNOB_BACK, z), task.KNOB_OPEN / rigstate.FINGER_OPEN)})
    for _ in range(10):
        sim.step(_clear_arms())
    assert sim.ctrl_m == pytest.approx(L.CTRL_OFF_POS)
    assert scoring.control_position(sim.ctrl_m) == "off"


def test_kit_files_byte_identical():
    for fname in ("mechkit.py", "rigstate.py"):
        assert (TASK_DIR / fname).read_bytes() == (REPO_ROOT / "tools" / "kit" / fname).read_bytes()
