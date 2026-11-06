"""On the station table: a cake turntable fixed in place 27 cm in front of the arm bases, just right of the midline: a
grey base and, turning on it about a vertical axis, a white plate 340 mm across with a red index mark, four black
pegs 20 mm across near its rim to turn it by, and a dummy iced cake (a cream cylinder 120 mm across, 50 mm tall) at its
centre. In front of the left arm a cake smoother (a clear blade 100 x 6 x 70 mm with a black handle along its top
edge) hangs by its handle in a grey slotted stand. Smoothing the cake's side as decorators do: hold the smoother
still, upright against the side with its bottom edge just above the plate, while the other gripper turns the
turntable round under it; the score is the share of the side that the turning carries past the still blade.

The turntable is a mechkit assembly (mechkit.py, next to this file: a fixed-base articulation with one revolute joint)
and the smoother a mechkit compound; the scorer reads the turntable's angle and the smoother's pose from them, and
the arms (tool point, finger tips, opening) through rigstate.py, which wraps the embodiment's step and reset from this
file's import. Neither changes what the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena: the turntable and smoother become mechkit bodies)
import numpy as np  # noqa: E402
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Held for 1 s once 340 sectors of the side have passed the blade, the smoother is 10 mm or more from the cake, every
# gripper's fingers are 20 mm or more from every peg (so the turning gripper has let go) and the turntable is still.
# The tracker times the same 1 s (layout.FINISH_HOLD_S) so that the score at a run-limit cut agrees.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Smooth the side of the cake on the turntable. With one gripper, take the smoother from its stand by the handle "
    "and hold it still and upright, its blade against the cake's side, its bottom edge just above the turntable. "
    "With the other gripper, turn the turntable anticlockwise (seen from above) by its black pegs almost a full "
    "turn. Then move the smoother 1 cm from the cake and both grippers 2 cm from the pegs. Never touch the cake "
    "with a gripper or its top with the smoother."
)

# -- the turntable: one fixed assembly, its base frame on the table top at the axis ------------------------------
_PZ = L.BASE_H + L.PLATE_T / 2


def _peg(angle_deg: float):
    a = math.radians(angle_deg)
    return mk.cylinder((L.PEG_CIRCLE * math.cos(a), L.PEG_CIRCLE * math.sin(a), L.PLATE_TOP + L.PEG_H / 2),
                       L.PEG_R, L.PEG_H, color=L.BLACK)


PLATE = mk.hinge(
    "plate", "Z", anchor=(0.0, 0.0, L.BASE_H),
    shapes=[mk.cylinder((0.0, 0.0, _PZ), L.PLATE_R, L.PLATE_T, color=L.WHITE),
            mk.cylinder((0.0, 0.0, L.PLATE_TOP + L.CAKE_H / 2), L.CAKE_R, L.CAKE_H, color=L.CREAM),
            *[_peg(a) for a in L.PEG_ANGLES_DEG],
            mk.box((L.MARK_AT, 0.0, L.PLATE_TOP + L.MARK_SIZE[2] / 2), L.MARK_SIZE, color=L.RED)],
    lower=L.PLATE_LOWER, upper=L.PLATE_UPPER, mass=L.PLATE_MASS, damping=L.PLATE_DAMPING, color=L.WHITE,
)
TURNTABLE = mk.assembly("turntable", origin=(L.AXIS[0], L.AXIS[1], L.TOP),
                        shapes=[mk.cylinder((0.0, 0.0, L.BASE_H / 2), L.BASE_R, L.BASE_H)], joints=[PLATE],
                        color=L.GREY)
# -- the smoother: one loose compound, its base frame at the middle of the blade's bottom edge -------------------
SMOOTHER = mk.assembly(
    "smoother", origin=L.SMOOTHER_START, mass=L.SMOOTHER_MASS, color=L.CLEAR,
    shapes=[mk.box((0.0, 0.0, L.BLADE[2] / 2), L.BLADE, color=L.CLEAR),
            mk.box((0.0, 0.0, L.BLADE[2] + L.HANDLE[2] / 2), L.HANDLE, color=L.BLACK)],
)
# -- the stand: two fixed blocks along x, an 18 mm slot between them ---------------------------------------------
_SY = L.STAND_SLOT / 2 + L.STAND_BLOCK[1] / 2
STAND = [Box(f"stand_{side}", pos=(L.STAND_XY[0], L.STAND_XY[1] + sign * _SY, L.TOP + L.STAND_BLOCK[2] / 2),
             size=L.STAND_BLOCK, color=L.STAND_GREY) for side, sign in (("left", 1.0), ("right", -1.0))]
OBJECTS = [TURNTABLE, SMOOTHER, *STAND]

_TRACKER = S.Tracker()
_MODEL = offline_model.PlateModel()  # offline only (see offline_model.py)
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the smoother teleported back)."""
    _TRACKER.reset()
    _MODEL.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _turn(t, arms, advance: bool) -> tuple[float, float]:
    """The plate's angle and angular velocity (rad, rad/s; positive anticlockwise seen from above). In the
    simulator: mechkit's joint reading (NaN if the turntable's state is broken). Offline (tools/offline/check_task.py:
    no physics, the joint stays at 0) the scorer is fed offline_model's angle instead: where the grippers would have
    turned it by its pegs. The simulator never takes that branch (its states say source "sim")."""
    if getattr(t, "source", "sim") == "offline":
        if advance:
            _MODEL.update(arms)
        return _MODEL.theta, _MODEL.omega
    try:
        if not np.all(np.isfinite(np.asarray(t.origin, dtype=float))):
            raise ValueError("origin")
        return float(t.joints["plate"]), float(t.joint_vel["plate"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return math.nan, math.nan


def success(objects) -> bool:
    """340 or more of the side's 360 sectors passed the blade, then the smoother 10 mm or more from the cake, no
    gripper's fingers within 20 mm of a peg and the turntable still, nothing ever wrong; held 1 s. This runs first
    every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _MODEL.reset()
        _EPISODE["mechkit"] = mk.episode()
    t, sm = objects["turntable"], objects["smoother"]
    arms = rigstate.arms()
    turn = _turn(t, arms, advance=True)
    result = _TRACKER.update(turn, sm, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "source": getattr(t, "source", None)}
    try:
        row.update(smoother_origin=sm.origin, smoother_quat=sm.quat_xyzw)
    except AttributeError:
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a gripper's fingers ever came within 5 mm of the cake, or the smoother touched the cake's top or fell
    off the table. Else 1.0 for 340 or more of the side's 360 sectors passed the blade and the finish held 1 s
    (smoother 10 mm clear, fingers 20 mm from the pegs, turntable still); 0.75 for 270 or more (or 340 without the
    finish); 0.5 for 180 or more; 0.25 if the smoother was ever set for 1 s; 0 otherwise."""
    arms = rigstate.arms()
    return _TRACKER.result(_turn(objects["turntable"], arms, advance=False), objects["smoother"], arms).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN_50 = -0.025  # 50 mm: 15 mm clear of each side of a 20 mm peg
OPEN_60 = -0.030  # 60 mm: 14 mm clear of each side of the 32 mm handle
HANDLE_GRIP = -(L.HANDLE[1] - 0.008) / 2
PEG_GRIP = -(2 * L.PEG_R - 0.008) / 2
# Left arm (the smoother): its fingers close across the handle, along y at the stand (yaw 0: the handle lies along x)
# and along the cake's radius at the set place (yaw -30 deg: the blade tangent to the side at 60 deg); it turns the
# smoother on the way.
SX, SY = L.STAND_XY
PICK_Z = L.TOP + L.HANG + L.GRIP_TCP  # the finger tips 9 mm above the handle's underside, 9 mm above the stand
OUT_OF_STAND_Z = PICK_Z + L.BLADE[2] + 0.012  # the blade's bottom 12 mm above the stand's top
SET_Z = L.TOP + L.PLATE_TOP + L.SET_LIFT + L.GRIP_TCP
CARRY_Z = SET_Z + 0.030  # the blade's bottom 30 mm higher than when set, over the pegs' tops (and 4 cm or more
#                          from them sideways all the way)
_PHI = math.radians(L.SET_ANGLE_DEG)
_BLADE_MID = L.CAKE_R + L.SET_GAP + L.BLADE[1] / 2  # from the axis to the blade's mid-plane when set


def _radial(r: float) -> tuple[float, float]:
    return (L.AXIS[0] + r * math.cos(_PHI), L.AXIS[1] + r * math.sin(_PHI))


SET_XY = _radial(_BLADE_MID)
APPROACH_XY = _radial(_BLADE_MID + L.APPROACH_OUT)
YAW_STAND = 0.0
YAW_L = _PHI - math.pi / 2  # at the set place: the fingers close radially, the blade lies along the tangent
# Right arm (the turntable): it grips a peg with the fingers closing along the pegs' circle (yaw = the peg's angle
# + 180 deg) and carries it 86 deg round in 5 straight chords (1.7 mm off the arc at their middles), the yaw
# following the plate; then lets go, lifts its tips 10 mm over the pegs and goes back 90 deg round the circle to
# the next peg. Four strokes turn the plate 344 deg: about 350 sectors of the side pass the blade (340 are needed),
# 5.5 deg short of the joint's stop.
PEG_Z = L.TOP + L.PLATE_TOP + 0.007 + L.PAD_TIPS  # the finger tips 7 mm above the plate, the pads on the peg
PEG_UP_Z = L.TOP + L.PEG_TOP + 0.010 + L.PAD_TIPS
PEG_HOVER_Z = L.TOP + 0.090
STROKE_DEG, CHORDS, CHORD_S = 86.0, 5, 0.8
RETURN_CHORDS, RETURN_S = 3, 0.5
FIRST_PEG = 2  # the peg at 210 deg (-150 deg: back right of the axis) is the first one gripped
TURN_DELAY_S = 0.5  # the turning starts this long after the smoother is set


def _peg_xy(world_deg: float) -> tuple[float, float]:
    a = math.radians(world_deg)
    return (L.AXIS[0] + L.PEG_CIRCLE * math.cos(a), L.AXIS[1] + L.PEG_CIRCLE * math.sin(a))


def _peg_yaw(world_deg: float) -> float:
    return math.radians(world_deg + 180.0)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left_setup() -> list[Waypoint]:
    return [
        Waypoint(1.8, (SX, SY, L.TOP + 0.150), OPEN_60, "over the smoother's handle", YAW_STAND),
        Waypoint(0.8, (SX, SY, L.TOP + 0.120), OPEN_60, "down toward the handle", YAW_STAND),
        Waypoint(1.5, (SX, SY, PICK_Z), OPEN_60, "down beside the handle, slowly", YAW_STAND),
        Waypoint(0.5, (SX, SY, PICK_Z), OPEN_60, "settle", YAW_STAND),
        Waypoint(1.5, (SX, SY, PICK_Z), HANDLE_GRIP, "grip the handle", YAW_STAND),
        Waypoint(0.3, (SX, SY, PICK_Z), HANDLE_GRIP, "settle", YAW_STAND),
        Waypoint(1.8, (SX, SY, OUT_OF_STAND_Z), HANDLE_GRIP, "lift it straight up out of the stand", YAW_STAND),
        Waypoint(2.4, (*APPROACH_XY, CARRY_Z), HANDLE_GRIP, "carry it to 2 cm outside the cake, turning it", YAW_L),
        Waypoint(1.5, (*APPROACH_XY, SET_Z), HANDLE_GRIP, "down to just above the turntable, slowly", YAW_L),
        Waypoint(1.5, (*SET_XY, SET_Z), HANDLE_GRIP, "in to the cake's side", YAW_L),
        Waypoint(0.5, (*SET_XY, SET_Z), HANDLE_GRIP, "settle: set", YAW_L),
    ]


T_SET = _span(_left_setup())  # 14.1 s: the smoother is set
T_TURN = T_SET + TURN_DELAY_S


def _stroke(start_deg: float, k: int) -> list[Waypoint]:
    step = STROKE_DEG / CHORDS
    return [Waypoint(CHORD_S, (*_peg_xy(start_deg + i * step), PEG_Z), PEG_GRIP,
                     f"turn: peg {k} to {start_deg + i * step:.0f} deg", _peg_yaw(start_deg + i * step))
            for i in range(1, CHORDS + 1)]


def _regrasp(end_deg: float, next_deg: float) -> list[Waypoint]:
    yaw_end = _peg_yaw(end_deg)
    wps = [Waypoint(0.5, (*_peg_xy(end_deg), PEG_Z), PEG_GRIP, "settle", yaw_end),
           Waypoint(1.5, (*_peg_xy(end_deg), PEG_Z), OPEN_50, "let go of the peg", yaw_end),
           Waypoint(0.6, (*_peg_xy(end_deg), PEG_UP_Z), OPEN_50, "lift the tips over the pegs", yaw_end)]
    step = (next_deg - end_deg) / RETURN_CHORDS
    wps += [Waypoint(RETURN_S, (*_peg_xy(end_deg + i * step), PEG_UP_Z), OPEN_50, "back round to the next peg",
                     _peg_yaw(end_deg + i * step)) for i in range(1, RETURN_CHORDS + 1)]
    yaw = _peg_yaw(next_deg)
    return wps + [Waypoint(1.5, (*_peg_xy(next_deg), PEG_Z), OPEN_50, "down round the peg, slowly", yaw),
                  Waypoint(0.5, (*_peg_xy(next_deg), PEG_Z), OPEN_50, "settle", yaw),
                  Waypoint(1.5, (*_peg_xy(next_deg), PEG_Z), PEG_GRIP, "grip the peg", yaw),
                  Waypoint(0.3, (*_peg_xy(next_deg), PEG_Z), PEG_GRIP, "settle", yaw)]


def _strokes() -> list[tuple[int, float]]:
    """(peg index, its world angle when gripped) for each stroke: each starts at the peg that the last one left
    90 deg behind the one it carried (86 deg strokes: 4 deg further back each time)."""
    out, theta = [], 0.0
    k = FIRST_PEG
    for _ in range(4):
        out.append((k, (L.PEG_ANGLES_DEG[k] + theta + 180.0) % 360.0 - 180.0))
        theta += STROKE_DEG
        k = (k - 1) % len(L.PEG_ANGLES_DEG)
    return out


def _right() -> list[Waypoint]:
    """Right arm: grip the first peg and wait until the smoother is set; four strokes of 86 deg, letting go and
    going back round for the next peg between them; let go, up."""
    strokes = _strokes()
    k0, a0 = strokes[0]
    yaw0 = _peg_yaw(a0)
    plan = [
        Waypoint(2.5, (*_peg_xy(a0), PEG_HOVER_Z), OPEN_50, f"over peg {k0}", yaw0),
        Waypoint(0.6, (*_peg_xy(a0), PEG_Z + 0.025), OPEN_50, "down toward the peg", yaw0),
        Waypoint(1.5, (*_peg_xy(a0), PEG_Z), OPEN_50, "down round the peg, slowly", yaw0),
        Waypoint(0.5, (*_peg_xy(a0), PEG_Z), OPEN_50, "settle", yaw0),
        Waypoint(1.5, (*_peg_xy(a0), PEG_Z), PEG_GRIP, "grip the peg", yaw0),
        Waypoint(0.3, (*_peg_xy(a0), PEG_Z), PEG_GRIP, "settle", yaw0),
    ]
    plan.append(Waypoint(T_TURN - _span(plan), (*_peg_xy(a0), PEG_Z), PEG_GRIP, "wait until the smoother is set",
                         yaw0))
    for j, (k, a) in enumerate(strokes):
        plan += _stroke(a, k)
        if j + 1 < len(strokes):
            plan += _regrasp(a + STROKE_DEG, strokes[j + 1][1])
    end = strokes[-1][1] + STROKE_DEG
    return plan + [Waypoint(0.5, (*_peg_xy(end), PEG_Z), PEG_GRIP, "settle", _peg_yaw(end)),
                   Waypoint(1.5, (*_peg_xy(end), PEG_Z), OPEN_50, "let go of the peg", _peg_yaw(end)),
                   Waypoint(0.8, (*_peg_xy(end), PEG_HOVER_Z + 0.010), OPEN_50, "up, clear of the pegs",
                            _peg_yaw(end))]


def _turn_end() -> float:
    """When the right arm's last stroke ends (the turntable at its final angle)."""
    t = 0.0
    last = 0.0
    for w in _right():
        t += w.duration
        if w.label.startswith("turn:"):
            last = t
    return last


T_TURNED = _turn_end()


def _left() -> list[Waypoint]:
    """Left arm: take the smoother from its stand and set it against the cake's front left; hold it there, still,
    while the right arm turns the turntable; then back it out 2 cm and lift it, still in hand."""
    plan = _left_setup()
    hold = (*SET_XY, SET_Z)
    plan.append(Waypoint(T_TURNED + 0.3 - _span(plan), hold, HANDLE_GRIP, "hold it still: the cake turns", YAW_L))
    return plan + [Waypoint(1.5, (*APPROACH_XY, SET_Z), HANDLE_GRIP, "back out 2 cm from the cake", YAW_L),
                   Waypoint(1.2, (*APPROACH_XY, CARRY_Z), HANDLE_GRIP, "up, still in hand", YAW_L)]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
