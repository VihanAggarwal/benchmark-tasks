"""On the station table, a belt drive with its belts off: a blower fixed in place in front of the right arm, its
90 mm pulley's outer face toward the arms on the line x 23.0 cm, and, 290 mm to its left, a loose motor (a 150 g
light motor with a printed shell and a 64 mm pulley) standing on a fixed steel base plate, turned 3 degrees and with its pulley face
12 mm behind the line. A 480 mm aluminium straightedge stands on its edge on a felt strip in front of them. Lining
up the pulleys as HVAC mechanics do: lay the straightedge across both pulleys' outer faces, flat on the blower's,
and slide and turn the motor on its plate until its pulley's face is in the same plane; then let go of everything.
The score is the total misalignment M (the motor's turn plus its offset over the centre distance, in degrees), and it
is 0 for the whole trial if the motor ever leaves its plate (its lowest corner more than 2 mm above it) or its line.

The blower and the motor are mechkit compounds (mechkit.py, next to this file: the blower fixed, the motor one loose
rigid body with its shaft and pulley along its length); the scorer reads the motor's pose from mechkit and the arms
(TCP, pad tips, opening) through rigstate.py, which wraps the embodiment's step and reset from this file's import.
Neither changes what the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena: the blower and the motor become mechkit bodies)
import numpy as np  # noqa: E402
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Held for 1 s: the motor in line (M <= 0.5 deg), at rest, upright, on its plate and its line, with every TCP and
# pad-tip point 60 mm or more from the motor and the straightedge (more than a fully open finger reaches) on every step
# of the last 0.5 s, and the motor never off its plate or its line on any step of the trial, so the hold starts only
# once both grippers have let go and risen.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Line up the motor's pulley with the blower's pulley. Slide and turn the loose motor on its base plate, keeping "
    "it on the plate and the same distance from the blower, until its pulley's outer face is in the same plane as "
    "the blower pulley's outer face, so the straightedge laid across both faces touches each pulley at both rims. "
    "Then let go of the motor and the straightedge and move both grippers away."
)

# -- the blower: fixed, its frame on the table under its pulley's centre ----------------------------------------
BLOWER = mk.assembly(
    "blower", origin=(L.BLOWER_ORIGIN[0], L.BLOWER_ORIGIN[1], L.TOP + L.BLOWER_ORIGIN[2]), color=L.STEEL_DARK,
    shapes=[mk.box(L.BLOWER_HOUSING_C, L.BLOWER_HOUSING, color=L.STEEL_DARK),
            mk.cylinder(L.BLOWER_SHAFT_C, L.BLOWER_SHAFT_R, L.BLOWER_SHAFT_LEN, axis="X", color=L.STEEL),
            mk.cylinder(L.BLOWER_PULLEY_C, L.BLOWER_PULLEY_R, L.BLOWER_PULLEY_LEN, axis="X", color=L.CAST_IRON)],
)
# -- the motor: loose, its frame at the centre of its body's underside, its shaft out of local -x ---------------
MOTOR = mk.assembly(
    "motor", origin=(L.MOTOR_START[0], L.MOTOR_START[1], L.TOP + L.MOTOR_START[2]), mass=L.MOTOR_MASS,
    yaw=L.MOTOR_YAW0, color=L.MOTOR_BLUE,
    shapes=[mk.box(L.MOTOR_BODY_C, L.MOTOR_BODY, color=L.MOTOR_BLUE),
            mk.cylinder(L.MOTOR_SHAFT_C, L.MOTOR_SHAFT_R, L.MOTOR_SHAFT_LEN, axis="X", color=L.STEEL),
            mk.cylinder(L.MOTOR_PULLEY_C, L.MOTOR_PULLEY_R, L.MOTOR_PULLEY_LEN, axis="X", color=L.CAST_IRON)],
)
PLATE = Box("base_plate", pos=(L.PLATE_C[0], L.PLATE_C[1], L.TOP + L.PLATE_SIZE[2] / 2), size=L.PLATE_SIZE,
            color=L.GALVANISED)
FELT = Box("felt", pos=(L.FELT_C[0], L.FELT_C[1], L.TOP + L.FELT_SIZE[2] / 2), size=L.FELT_SIZE, color=L.FELT_GREEN)
STRAIGHTEDGE = Box("straightedge", pos=(L.BAR_START[0], L.BAR_START[1], L.TOP + L.BAR_START[2]), size=L.BAR_SIZE,
                   mass=L.BAR_MASS, color=L.ALUMINIUM)
OBJECTS = [BLOWER, PLATE, MOTOR, FELT, STRAIGHTEDGE]

_TRACKER = S.Tracker()
_MODEL = offline_model.MotorModel(MOTOR)  # offline only (see offline_model.py)
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots everything teleported back)."""
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


def _motor(objects, arms, advance: bool):
    """The motor's state. In the simulator: mechkit's reading. Offline (tools/offline/check_task.py: no physics,
    nothing pushes anything) the scorer is fed offline_model's pose instead: where the straightedge would have
    pushed it. The simulator never takes that branch (its states say source "sim")."""
    m = objects["motor"]
    if getattr(m, "source", "sim") == "offline":
        if advance or _MODEL.state is None:
            return _MODEL.update(m, objects["straightedge"], arms)
        return _MODEL.state
    return m


def success(objects) -> bool:
    """The motor's pulley face in line with the blower's (M = |turn| + atan(|offset| / 290 mm) <= 0.5 deg), the
    motor at rest, upright, on its plate and within 10 mm of its line, every TCP and pad-tip point 60 mm or more from
    the motor and the straightedge over the last 0.5 s, and the motor never off its plate or its line during the
    trial; held 1 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate read for
    this step) and the whole-trial plate and line rules."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _MODEL.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    motor, bar = _motor(objects, arms, advance=True), objects["straightedge"]
    result = _TRACKER.update(motor, bar, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "source": getattr(motor, "source", None), "push_mode": _MODEL.mode}
    try:
        row.update(motor_origin=motor.origin, motor_quat=motor.quat_xyzw, bar_pos=bar.pos, bar_quat=bar.quat_xyzw)
    except AttributeError:
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, opening={s: round(float(arms[s].opening), 3) for s in S.ARMS})
        if os.environ.get("TASK_TRACE"):
            t = _TRACKER.steps / L.CONTROL_HZ
            for side in S.ARMS:  # the scripted demo's command at this step (meaningful for the demo only)
                cmd = demo_target(side, t)
                if cmd is not None:
                    row[f"{side}_cmd"] = cmd
                    row[f"{side}_cmd_err_mm"] = round(float(np.linalg.norm(np.asarray(arms[side].tcp) - cmd)) * 1000,
                                                      1)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the motor is at rest (its pulley face's path under 2 mm and its turn under 0.5 deg over the last
    0.5 s), upright within 2 deg, on its plate (every bottom corner inside it, within 2 mm of its top), within 10 mm
    of its line (y 0.140), both grippers readable and 60 mm or more from the motor and the straightedge on every step
    of the last 0.5 s, and on every step of the trial the motor stayed on its plate (lowest bottom corner within 2 mm
    of its top, the underside's centre over it, tipped under 10 deg) and within 10 mm of its line. Then 1.0 for
    M <= 0.5 deg, 0.5 for M <= 1.0 deg, 0.25 for M <= 2.0 deg, else 0 (M = |turn| + atan(|offset| / 290 mm))."""
    arms = rigstate.arms()
    return _TRACKER.result(_motor(objects, arms, advance=False), objects["straightedge"], arms).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
MOTOR_GRIP = -(L.MOTOR_BODY[1] - 0.008) / 2  # 62 mm across the body's 70 mm: about 20 N per finger
BAR_GRIP = -(L.BAR_SIZE[0] - 0.008) / 2  # 12 mm across the straightedge's 20 mm
BAR_OPEN = -0.025  # 50 mm round the straightedge: 15 mm clear each side, and the right arm's open fingers stay
#                    33 mm from its own base (fully open, 13 mm) gripping 15 cm in front of it
TRAVEL_Z = L.TOP + 0.150
# Left arm, first: the motor, which never leaves its plate. It grips the body at its centre of mass's height (the TCP
# 60 mm up: the 22 mm pads on the body 47 to 69 mm up, their tips 35 mm above the plate's top; the finger carriages,
# 57 mm above the TCP, 25 mm above the body's 92 mm top), the fingers closing across the body (yaw = the motor's
# 3 deg). Then, at the SAME height (no lift: the arm holds the TCP where it gripped the motor resting on the plate;
# lifting it at all would need the command about 3 mm higher just to carry its weight, by the 3.1 mm sag the GPU
# measured with this 150 g motor hanging from this arm), it slides the motor along the plate toward the arms while
# turning it square, and eases off before letting go. The arm is soft in x with its wrist free (J K^-1 J^T of the joint
# gains: about 150 N/m), so the plate's sliding friction (1.0 x 1.47 N) leaves the motor about 11 mm behind the
# command, rocked about 2.6 deg onto its leading edge, which stays pressed on the plate (by a quasi-static model of the
# arm, the motor and the plate; 1.3 deg for arms twice as stiff, 3.3 deg at the static friction): the slide is
# commanded DRAG_LAG past PRE_X. The ease-off then moves the command back to PRE_X, so the arm's pull drops below the
# 1.77 N static friction (the motor stays where it stopped, about 2 mm nearer the arms than PRE_X, and settles flat)
# before the fingers open; on a stiffer arm the motor follows the command back to PRE_X. The motor ends with its pulley
# face PROUD or more in front of the line (a straightedge only pushes): about 10 mm for the expected lag (the previous
# demo's press, unchanged, lined the motor up from 8.7 mm proud on the GPU), 7 mm for arms 0.8 times as stiff, and
# about on the line for arms half as stiff. At the expected stop its corners are all on the plate.
MOTOR_GRIP_Z = L.TOP + 0.060
DRAG_Z = MOTOR_GRIP_Z  # the slide keeps the grip height: the motor is never lifted
PROUD = 0.008
DRAG_LAG = 0.013
PRE_X = L.ALIGNED_X - PROUD  # 0.310: where the motor ends at the least, its face 8 mm proud (x 0.222), square
DRAG_X = PRE_X - DRAG_LAG  # 0.297: the slide's command (the motor about 11 mm behind it: x 0.308, 10 mm proud)
# Both arms, then: the straightedge. Each grips it with the fingers closing across its 20 mm (yaw -pi/2 left, +pi/2
# right), the TCP 3 mm above its centre: the right arm 35 mm in from its right end (y -0.250), outside the blower
# pulley's outer rim (y -0.195), the left at y 0.000 between the pulleys, so the bar can lie flat on the blower pulley
# at both rims while it pushes the motor.
BAR_GRIP_Y = {"left": 0.000, "right": -0.250}
BAR_YAW = {"left": -math.pi / 2, "right": math.pi / 2}
BAR_X0 = L.BAR_START[0]
BAR_TCP_UP = 0.003  # the TCP above the bar's centre
BAR_GRIP_Z = L.TOP + L.BAR_START[2] + BAR_TCP_UP  # 0.025: pad tips 10 mm above the bar's bottom
PRESS_Z = L.TOP + 0.029  # the bar's bottom 6 mm above the table while pressing (its top 46 mm up)
START_X = 0.195  # the bar's face at x 0.205: 19 mm from the motor's (proud) face, 25 mm from the blower's
# The press: both grippers are driven together OVERDRIVE past the point where the bar's face is on the line (TCP x
# 0.220), at 5 mm/s. The bar meets the motor's face first and pushes it back; once it reaches the blower pulley's face
# the commands go on and the arms' give builds the press force. Each arm is a spring at the TCP: by J K^-1 J^T of the
# joint gains at the press, 184 N/m along x for the left arm and 137 N/m for the right with the wrists free, 780 and
# 2,060 N/m if the bar's contacts hold the grippers' pitch (180-440 N/m measured on the GPU in a comparable reaching
# posture). As two equal springs with free wrists the bar pivots on the blower pulley's outer rim and passes 0.418 of
# each arm's force to the motor, so each must push 4.2 N to start it (1.8 N): 141 N/m at 30 mm, 170 N/m at 25 mm. A
# quasi-static model with each arm's own x and yaw stiffness at its posture lines the motor up at 30 mm for joint
# gains from 0.5 to 8 times nominal, however the contacts hold the pitch (25 mm: from 0.6 times). Keep it EQUAL on
# both arms (+-3 mm): more on the right arm than on the left lifts the motor end of the bar off the motor.
OVERDRIVE = 0.030
PRESS_X = L.PLANE_X - L.BAR_HALF[0] + OVERDRIVE  # 0.250: the TCP 30 mm past where the bar sits on the line
PRESS_S = 11.0  # 55 mm at 5 mm/s
BACK_X = 0.190  # drawn back: the bar's face at x 0.200, 30 mm in front of the pulleys
SET_DOWN_Z = L.TOP + 0.023  # the bar's bottom on the table
UP_Z = L.TOP + 0.150


def _motor_pre_place() -> list[Waypoint]:
    yaw0 = L.MOTOR_YAW0
    x0, y = L.MOTOR_START[0], L.MOTOR_Y
    return [
        Waypoint(3.0, (x0, y, TRAVEL_Z), OPEN, "over the motor", yaw0),
        Waypoint(1.0, (x0, y, L.TOP + 0.090), OPEN, "down toward the motor", yaw0),
        Waypoint(1.5, (x0, y, MOTOR_GRIP_Z), OPEN, "down round its body, slowly", yaw0),
        Waypoint(0.6, (x0, y, MOTOR_GRIP_Z), OPEN, "settle", yaw0),
        Waypoint(1.5, (x0, y, MOTOR_GRIP_Z), MOTOR_GRIP, "grip the motor's body", yaw0),
        Waypoint(0.5, (x0, y, MOTOR_GRIP_Z), MOTOR_GRIP, "settle the grip", yaw0),
        Waypoint(4.5, (DRAG_X, y, DRAG_Z), MOTOR_GRIP, "slide it on the plate toward the line, turning it square", 0.0),
        Waypoint(1.0, (PRE_X, y, DRAG_Z), MOTOR_GRIP, "ease off the pull, still on the plate", 0.0),
        Waypoint(0.6, (PRE_X, y, DRAG_Z), MOTOR_GRIP, "settle", 0.0),
        Waypoint(1.5, (PRE_X, y, DRAG_Z), OPEN, "let go of the motor", 0.0),
        Waypoint(1.5, (PRE_X, y, TRAVEL_Z), OPEN, "up, clear of it", 0.0),
    ]


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _bar(side: str) -> list[Waypoint]:
    """From over the straightedge: down, grip, lift, to the start line, press, hold, draw back, set it down, let go,
    up (the same targets in x and z for both arms, on one clock)."""
    y, yaw = BAR_GRIP_Y[side], BAR_YAW[side]
    return [
        Waypoint(1.0, (BAR_X0, y, L.TOP + 0.055), BAR_OPEN, "down toward the straightedge", yaw),
        Waypoint(1.5, (BAR_X0, y, BAR_GRIP_Z), BAR_OPEN, "down round it, slowly", yaw),
        Waypoint(0.6, (BAR_X0, y, BAR_GRIP_Z), BAR_OPEN, "settle", yaw),
        Waypoint(1.5, (BAR_X0, y, BAR_GRIP_Z), BAR_GRIP, "grip the straightedge", yaw),
        Waypoint(0.5, (BAR_X0, y, BAR_GRIP_Z), BAR_GRIP, "settle the grip", yaw),
        Waypoint(1.5, (BAR_X0, y, PRESS_Z), BAR_GRIP, "lift it 4 mm off the felt", yaw),
        Waypoint(3.0, (START_X, y, PRESS_Z), BAR_GRIP, "carry it to 2 cm in front of the pulleys", yaw),
        Waypoint(0.5, (START_X, y, PRESS_Z), BAR_GRIP, "settle", yaw),
        Waypoint(PRESS_S, (PRESS_X, y, PRESS_Z), BAR_GRIP, "press: flat on both pulleys, pushing the motor", yaw),
        Waypoint(1.5, (PRESS_X, y, PRESS_Z), BAR_GRIP, "hold", yaw),
        Waypoint(3.0, (BACK_X, y, PRESS_Z), BAR_GRIP, "draw it back off the pulleys", yaw),
        Waypoint(1.0, (BACK_X, y, SET_DOWN_Z), BAR_GRIP, "set it down on the table", yaw),
        Waypoint(0.6, (BACK_X, y, SET_DOWN_Z), BAR_GRIP, "settle", yaw),
        Waypoint(1.5, (BACK_X, y, SET_DOWN_Z), BAR_OPEN, "let go of the straightedge", yaw),
        Waypoint(2.0, (BACK_X, y, UP_Z), BAR_OPEN, "straight up, clear of it", yaw),
    ]


def _over_bar(side: str, duration: float) -> Waypoint:
    return Waypoint(duration, (BAR_X0, BAR_GRIP_Y[side], TRAVEL_Z), BAR_OPEN, "over the straightedge", BAR_YAW[side])


def _left() -> list[Waypoint]:
    """Left arm: slide the motor along its plate (never lifting it) to square with its face 8 to 10 mm proud of the
    line, then grip the straightedge between the pulleys and press it with the right arm."""
    return _motor_pre_place() + [_over_bar("left", 3.0)] + _bar("left")


T_OVER_BAR = _span(_motor_pre_place()) + 3.0  # 20.2 s: the left arm is over the straightedge


def _right() -> list[Waypoint]:
    """Right arm: wait over the straightedge's right end while the left arm moves the motor, then grip the bar
    outside the blower pulley's outer rim and press it with the left arm."""
    first = _over_bar("right", 3.0)
    return [first, Waypoint(T_OVER_BAR - 3.0, first.pos, BAR_OPEN, "wait while the motor is moved", first.yaw)] + \
        _bar("right")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan


def demo_target(side: str, t: float):
    """The scripted demo's TCP target for one arm at time t (s; smoothstep between waypoints, as ArmPlan does), for
    the trace; None for an arm the demo leaves idle, before the end of the first waypoint (that segment starts at the
    ready pose) or past the end."""
    wps = demo().get(side)
    if not wps:
        return None
    end = 0.0
    for i, w in enumerate(wps):
        start, end = end, end + w.duration
        if t <= end:
            if i == 0:
                return None
            u = min(max((t - start) / w.duration, 0.0), 1.0) if w.duration > 0 else 1.0
            s = u * u * (3.0 - 2.0 * u)
            a, b = np.asarray(wps[i - 1].pos, dtype=float), np.asarray(w.pos, dtype=float)
            return a + s * (b - a)
    return None
