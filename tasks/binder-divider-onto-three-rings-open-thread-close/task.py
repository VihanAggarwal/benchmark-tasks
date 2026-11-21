"""On the station table, real size: an Avery Durable View 1 in slant-ring binder (item 17011, black) lies OPEN and flat,
its spine along y, the three rings standing on the back cover (beyond the spine) with the movable halves shut. At the
rings' -y end the mechanism's ring-opening thumb lever (a bar over the plate's -x edge) shows. On the arms' side of the
spine, on a slanted desk sorter, leans a loose Avery plastic divider (US letter, 279 x 216 mm, three punched holes at
the standard 4.25 in pitch), its punched edge down.

Goal: the divider on all three rings (every hole round its ring), the rings shut, both hands away.

The rings work by contact only (mechkit.py, next to this file). The three movable halves and the thumb lever are one
rigid link ("gate") on a hinge along y at the halves' feet, with a spring to a two-notch detent (shut, open 0.80 rad):
pressing the lever's bar down turns the gate open, and past 0.55 rad it snaps to the open stop; pushing the half rings
back from under 0.25 rad snaps it shut. The divider is a loose rigid sheet with three 8 mm square holes. No rule and
no flag: the score reads poses only.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import offline_binder  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402  (pitched grips for the demo: wraps yam_arena.demo.grip_rotation)
from scoring import BINDER, DIVIDER  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # on top of the scorer's own 1 s window

INSTRUCTION = ("Open the binder's rings, put the loose plastic divider onto all three rings, close the rings with the "
               "divider on them, and take your hands off everything.")

# -- the parts --------------------------------------------------------------------------------------------------------
HOLDER = "sorter"
HOLDER_PART = mk.assembly(HOLDER, origin=P.HOLD_O, shapes=P.holder_shapes(), color=L.HOLDER_COLOR,
                          friction=L.HOLDER_FRICTION, tight_fit=True)
BINDER_PART = mk.assembly(BINDER, origin=P.BIN_O, shapes=P.binder_shapes(), joints=[P.gate_joint()], color=L.BLACK,
                          friction=L.BINDER_FRICTION, tight_fit=True)
DIVIDER_PART = mk.assembly(DIVIDER, origin=P.DIV_O, shapes=P.divider_shapes(), mass=L.SHEET_MASS, color=L.SHEET_COLOR,
                           friction=L.SHEET_FRICTION, tight_fit=True)
OBJECTS = [HOLDER_PART, BINDER_PART, DIVIDER_PART]
SPECS = {BINDER: BINDER_PART, DIVIDER: DIVIDER_PART, HOLDER: HOLDER_PART}

_OFFLINE = offline_binder.BinderModel(BINDER_PART, DIVIDER_PART)


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(DIVIDER), "source", "sim") != "sim"


# -- the detent: a rule that sets the gate's drive target (the joint's spring pulls toward it) ------------------------
@mk.on_step
def _detent(states) -> None:
    st = states.get(BINDER)
    if st is None or "gate" not in getattr(st, "joints", {}):
        return
    mk.set_targets(BINDER_PART, "gate", mk.detent_target(float(st.joints["gate"]), (0.0, L.OPEN),
                                                         L.DETENT_CAPTURE))


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Divider on all three rings, rings shut, divider flat, still and hands off on every step of the last 1 s. This
    runs first every control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states(S.PARTS)
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["divider_origin"] = parts[DIVIDER].origin
        row["divider_quat"] = parts[DIVIDER].quat_xyzw
        row["gate_vel"] = parts[BINDER].joint_vel.get("gate")
    except (AttributeError, KeyError, TypeError):
        pass
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       left_tip=arms["left"].tips, right_tip=arms["right"].tips,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(DIVIDER), "source", None)
    if offline:
        row["offline"] = {"mode": _OFFLINE.mode, "holds": sorted(_OFFLINE.holds), "swing": round(_OFFLINE.swing, 3)}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the divider left the table, or the rings are shut with the divider off them; 0.3 rings open, divider not
    on all three; 0.7 divider on all three rings but not (shut, flat, still, hands off) for the whole last second;
    1.0 all of it. The latest scored step's score."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator.
# Both hands pinch the divider's far edge with their pads closing along world x and the whole hand pitched OPEN rad
# about y (approach axis leaning toward +x): the pinch axis is then the divider's normal, and the approach axis lies in
# the divider's plane. The pads straddle the top 12 mm of the far edge, one each side, at u = -70 and +70 mm.
YAW_L, YAW_R = -math.pi / 2, math.pi / 2
PITCH = L.OPEN
TILT_L = tiltgrip.tilted(YAW_L, PITCH)
TILT_R = tiltgrip.tilted(YAW_R, PITCH)
A = tiltgrip.rotation(YAW_R, PITCH)[:, 2]  # the tilted hands' approach axis: -D
PAD_TIP = 0.013
F_CLOSED = 0.0
F_OPEN = -0.040  # pads 80 mm apart
D, N = L.D_VEC, L.N_VEC
YV = np.array([0.0, 1.0, 0.0])


def wp(d, tip, finger, label, arm):
    """A waypoint whose pad-tip point (table frame) is ``tip``."""
    if arm == "L":
        a, yaw = A, TILT_L
    elif arm == "R":
        a, yaw = A, TILT_R
    else:  # straight-down grips 'l' / 'r': yaw 0, the closing axis along y (the pad block 16 mm in y, 27 mm in x)
        a, yaw = np.array([0.0, 0.0, -1.0]), 0.0
    t = np.asarray(tip, float) - PAD_TIP * a
    return Waypoint(d, (float(t[0]), float(t[1]), float(L.TOP + t[2])), finger, label, yaw)


def park(d, xyz, label, arm):
    return Waypoint(d, (xyz[0], xyz[1], L.TOP + xyz[2]), F_OPEN, label, 0.0)


O0 = np.array(L.SHEET_O)
_open_pts = L.rotate_gate(L.half_points("movable", 80), L.OPEN)  # foot -> tip, (x, z)
TIP_OPEN = np.array([_open_pts[-1][0], 0.0, _open_pts[-1][1]])


# The gate angle the threading is aimed at. The detent's stop is 0.80 rad, but in the first GPU run the lever press
# (pad tip to 9 mm) left the gate at 0.849 rad and the weak spring (0.05 N m/rad) never pulled it back; the demo is
# unchanged up to the threading, so it is aimed at what the simulator showed. Against the model's 0.80 rad the holes
# are then at most 2 mm off the wires' centres (inside the clearance; a hole edge that does touch only nudges the gate
# toward the stop).
Q_AIM = 0.849


def wire_at(s: float) -> np.ndarray:
    """The point of the open movable half's wire centre line ``s`` m along it from its tip (x, 0, z), the gate at
    Q_AIM."""
    return L.wire_at(Q_AIM, s)


def origin_for(hole_row_pt) -> np.ndarray:
    """The sheet frame's origin that puts the hole row's middle at ``hole_row_pt``."""
    return np.asarray(hole_row_pt, float) - L.HOLE_V * D


S_ENGAGE = 0.012  # the holes end 12 mm along the wire from its tip, the hands still holding
O1 = O0 + 0.035 * D  # drawn 35 mm up the sorter's slope, clear of the lip
O2 = O1 + 0.040 * N  # lifted 40 mm off the backrest
O_B = origin_for(wire_at(0.0) + 0.060 * N)  # in front of the tips, out of the wires' reach
# The hole row goes on along the wire's own centre line: first along the tangent past the tip, then down the curve in
# 3 mm steps. (The first GPU run went in a straight line from 20 mm over the tips to the end point: that line cuts the
# curve, so the tips met the sheet 5 mm off the holes' centres along v, past the hole edge, even before the carry sag.)
TIP_AIM = wire_at(0.0)
TIP_TAN = (TIP_AIM - wire_at(0.002)) / np.linalg.norm(TIP_AIM - wire_at(0.002))  # the wire's direction at its tip
O_A = origin_for(TIP_AIM + 0.020 * TIP_TAN)
S_STEPS = (0.0, 0.003, 0.006, 0.009, 0.012)
O_E = origin_for(wire_at(S_ENGAGE))
HOVER_V = L.SHEET_V + 0.045


def pinch_tip(o, u, v=None):
    return np.asarray(o, float) + u * YV + (L.GRIP_V if v is None else v) * D


# Carry trim: the hands aim higher by the carry sag measured on the GPU (layout.CARRY_SAG_N, CARRY_SAG_Z: the divider
# rode 6 mm low and 1 mm near, so in the first run the hole row met the wire tips 3.5 mm off along v and the holes
# struck the tips). The N part turns with the hands; at the 0.80 rad tilt the two add up to (+1, 0, +6) mm.
TRIM_N = L.CARRY_SAG_N
TRIM_Z = L.CARRY_SAG_Z


def trim(n_vec=N, z=TRIM_Z) -> np.ndarray:
    return TRIM_N * np.asarray(n_vec, float) + np.array([0.0, 0.0, z])


# Closing: the holes ride on the movable half rings (12 mm along the wires), so moving the held divider the way those
# wire points go as the gate shuts (about 17 mm toward +x, its tilt kept) carries the half rings shut with it: the
# holes' edges drive the wires (about 20 mm of hole travel along v per rad of gate lag, so the gate lags at most
# 0.1 rad inside the 2 mm clearance). The divider is aimed CLOSE_OVER rad past shut so a lagging gate is driven home
# against its stop. Shut, the divider's holes are round the shut rings: nothing can lift it off. Let go, it leans back
# onto the sorter about the rings (its centre of mass is on the arm side of them), which presses the movable halves
# shut, never open. (Turning the divider upright with the gate is out of reach: the hands would be 0.24 m over the
# rings at x 0.43 m.)
CLOSE_STEPS = 6
CLOSE_OVER = 0.15


def closing_origin(q) -> np.ndarray:
    """The divider's frame origin with its hole row S_ENGAGE along the movable wires of a gate at ``q`` (tilt kept)."""
    return origin_for(L.wire_at(q, S_ENGAGE))


# The lever: the right hand, fingers shut, straight down on the bar's top (pad block 27 mm in x, the bar 28 mm).
LEVER_XY = (L.LEVER_PRESS_X, 0.5 * (L.LEVER_Y[0] + L.LEVER_Y[1]))
PRESS_TIP_Z = 0.0090
# A pad push (not used by the demo since the GPU run of 2026-11-21: once the divider is on the open rings it covers
# every ring from y -140 to +140 mm, so a pad coming down beside a ring lands on the divider; the demo shuts the rings
# by carrying the held divider instead, see sheet_moves). Kept for the offline model's push rule and its test: fingers
# shut, straight down beside ring 3 on the arm side of the movable half's wire, then +x until the wire is home.
PUSH_Z = 0.022
PUSH_Y = L.RING_Y[2]
_z0, _z1 = PUSH_Z, PUSH_Z + 0.022
_open_sel = (_open_pts[:, 1] >= _z0) & (_open_pts[:, 1] <= _z1)
_closed_pts = L.half_points("movable", 80)
_closed_sel = (_closed_pts[:, 1] >= _z0) & (_closed_pts[:, 1] <= _z1)
PUSH_X0 = float(_open_pts[_open_sel, 0].min()) - 0.002 - 0.003 - 0.0135  # pad centre: the face 3 mm clear of the wire
PUSH_X1 = float(_closed_pts[_closed_sel, 0].min()) - 0.002 - 0.0015 - 0.0135  # pad centre, wire home, 1.5 mm over


PARK_L = (0.200, 0.240, 0.200)
PARK_R = (0.200, -0.240, 0.200)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _hold(plan, until, label):
    last = plan[-1]
    gap = until - _duration(plan)
    if gap > 1e-9:
        plan.append(Waypoint(gap, last.pos, last.finger, label, last.yaw))


T_GO = 13.0  # both hands start for the divider now (the right has pressed the lever and stepped aside)


def right_moves(stop: str = "") -> list:
    x, y = LEVER_XY
    plan = [
        wp(3.0, (x, y, 0.070), F_CLOSED, "over the thumb lever, fingers shut", "r"),
        wp(2.0, (x, y, 0.035), F_CLOSED, "down to just over the lever's bar", "r"),
        wp(2.5, (x, y, PRESS_TIP_Z), F_CLOSED, "press the bar down: the gate turns open", "r"),
        wp(1.0, (x, y, PRESS_TIP_Z), F_CLOSED, "hold it open", "r"),
        wp(1.5, (x, y, 0.080), F_CLOSED, "up", "r"),
    ]
    if stop == "open":
        return plan + [park(2.0, PARK_R, "aside", "r")]
    hov = pinch_tip(O0, -L.GRIP_U, HOVER_V)
    plan.append(park(2.0, PARK_R, "aside", "r"))
    _hold(plan, T_GO, "wait")
    plan.append(wp(3.0, hov + np.array([0.0, -0.10, 0.0]), F_OPEN, "round to the side of the divider", "R"))
    plan.append(wp(3.0, hov, F_OPEN, "over the divider's far edge, pads 80 mm apart, hand pitched", "R"))
    plan += sheet_moves(-L.GRIP_U, "R", release=(stop != "hang"), thread=(stop != "hang"))
    plan.append(park(2.0, PARK_R, "aside", "r"))
    if stop != "hang":
        plan.append(Waypoint(2.0, plan[-1].pos, F_OPEN, "rest: the divider lies still on the shut rings", 0.0))
    return plan


def left_moves(stop: str = "") -> list:
    hov = pinch_tip(O0, L.GRIP_U, HOVER_V)
    plan = [park(2.0, PARK_L, "aside", "l")]
    if stop == "open":
        return plan
    _hold(plan, T_GO, "wait")
    plan.append(wp(3.0, hov + np.array([0.0, 0.10, 0.0]), F_OPEN, "round to the side of the divider", "L"))
    plan.append(wp(3.0, hov, F_OPEN, "over the divider's far edge, pads 80 mm apart, hand pitched", "L"))
    plan += sheet_moves(L.GRIP_U, "L", release=(stop != "hang"), thread=(stop != "hang"))
    plan.append(park(2.0, PARK_L, "aside", "l"))
    if stop != "hang":
        plan.append(Waypoint(2.0, plan[-1].pos, F_OPEN, "rest: the divider lies still on the shut rings", 0.0))
    return plan


def sheet_moves(u, arm, release=True, thread=True) -> list:
    """The shared carry: both hands on their own clock offset only by the hover waits above."""
    def at(o, v=None, t=None):
        return pinch_tip(o, u, v) + (np.zeros(3) if t is None else t)
    tr = trim()
    plan = [
        wp(2.5, at(O0, HOVER_V), F_OPEN, "above the pinch", arm),
        wp(2.5, at(O0), F_OPEN, "down along the sheet's plane: a pad each side of the far edge", arm),
        wp(0.5, at(O0), F_OPEN, "settle", arm),
        wp(1.5, at(O0), F_CLOSED, "pinch the divider's far edge", arm),
        wp(0.5, at(O0), F_CLOSED, "settle the pinch", arm),
        wp(2.0, at(O1), F_CLOSED, "draw the divider up the sorter's slope", arm),
        wp(2.0, at(O2), F_CLOSED, "lift it off the backrest", arm),
        wp(4.0, at(O_B, t=tr), F_CLOSED, "carry it over the rings, tilted, to in front of the open tips", arm),
    ]
    if not thread:
        return plan + [wp(1.5, at(O_B, t=tr), F_OPEN, "let go in the air", arm),
                       wp(1.5, at(O_B, HOVER_V, t=tr), F_OPEN, "up", arm)]
    plan += [
        wp(2.5, at(O_A, t=tr), F_CLOSED, "in toward the movable tips, holes in line with the wires", arm),
    ]
    plan += [wp(0.6, at(origin_for(wire_at(sd)), t=tr), F_CLOSED,
                "on along the wires' curve: the holes 12 mm onto the half rings", arm) for sd in S_STEPS]
    plan += [
        wp(0.5, at(O_E, t=tr), F_CLOSED, "settle", arm),
    ]
    if not release:
        return plan
    for q in np.linspace(Q_AIM, -CLOSE_OVER, CLOSE_STEPS + 1)[1:]:
        plan.append(wp(1.0, at(closing_origin(q), t=tr), F_CLOSED,
                       "carry the divider with the half rings as they shut: the holes drive the wires", arm))
    o_shut = closing_origin(-CLOSE_OVER)
    plan += [
        wp(0.5, at(o_shut, t=tr), F_CLOSED, "settle: rings shut, holes round them", arm),
        wp(1.0, at(o_shut, t=tr), F_OPEN, "open: the divider leans back onto the sorter about the shut rings", arm),
        wp(1.5, at(o_shut, L.SHEET_V + 0.090, t=tr), F_OPEN, "straight back along the approach axis", arm),
    ]
    return plan


# Wrong ways (counterfactuals; the offline model predicts them):
#   open   only the rings opened: 0.3
#   hang   the divider let go in the air in front of the tips (not threaded) and the rings left open: 0.3
VARIANTS = ("open", "hang")


def plans(variant: str = "") -> dict:
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    return {"left": left_moves(variant), "right": right_moves(variant)}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check). DEMO_VARIANT: see VARIANTS."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
