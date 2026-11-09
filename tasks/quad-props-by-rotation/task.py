"""On the station table: a black carbon quadcopter frame without propellers, fixed on the midline 30 cm in front of the
arm bases, its front away from the arms: a centre plate, four arms along the diagonals and a dark grey motor bell
(28 mm across) on the end of each, the motors 200 mm apart side to side and 160 mm front to back. A red arrow painted
on the table beside each motor shows which way it turns seen from above: front left and rear right clockwise (CW),
front right and rear left counter-clockwise (CCW). On a light grey foam mat in front of each arm lie three 126 mm
propellers, blades along x: grey hub, black blades, a raised white strip along each blade's leading edge. The left
mat holds one CW and two CCW propellers, the right mat one CCW and two CW. Rule: fit each motor with a propeller of
its own hand, upright on the motor's top: its leading edges must lead in the direction of the motor's arrow. Any
propeller of the wrong hand (or upside down) on a motor scores 0: that drone flips at take-off.

Each propeller is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of its hub, two blades
and two leading-edge strips, so the strips show and the scorer can read its pose.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the propellers are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. A propeller counts once its hub's bottom is within 3 mm of the motor's top
# and it is at rest, which happens as the last slow descent ends, with the gripper still closed on it. Settling and
# letting go take 0.5 + 1.5 s, so with 2.5 s the trial ends with the fingers open and the hands going up.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Fit a propeller on each of the drone's four motors, standing upright and centred on the motor's top (hub up, "
    "as they lie on the mats). The red arrow beside each motor shows which way that motor turns: use a propeller "
    "whose leading edges, the raised white strips along one side of each blade, lead in that direction. Two of the "
    "six propellers are spares."
)


def _arm(name: str) -> Box:
    """A frame arm: flat on the table along the motor's diagonal, from under the centre plate to under the motor."""
    ux, uy = L.outboard(name)
    r = (L.ARM_R0 + L.ARM_R1) / 2
    return Box(f"arm_{name}", pos=(L.CENTER[0] + r * ux, L.CENTER[1] + r * uy, L.TOP + L.ARM_H / 2),
               size=(L.ARM_LEN, L.ARM_W, L.ARM_H), yaw=math.atan2(uy, ux), color=L.CARBON)


def _motor(name: str) -> Cylinder:
    x, y = L.MOTOR_XY[name]
    return Cylinder(f"motor_{name}", pos=(x, y, L.TOP + L.MOTOR_Z0 + L.MOTOR_H / 2), radius=L.MOTOR_R,
                    height=L.MOTOR_H, color=L.MOTOR_GREY)


def _arrow(name: str) -> list[Box]:
    """The red turn arrow beside a motor: a shaft and two barbs, thin plates painted on the table."""
    return [Box(f"arrow_{name}_{part}", pos=(cx, cy, L.TOP + z0 + size[2] / 2), size=size, yaw=yaw,
                color=L.ARROW_RED)
            for part, (cx, cy), yaw, size, z0 in L.arrow_parts(name)]


def _mat(side: str, sign: float) -> Box:
    return Box(f"mat_{side}", pos=(L.MAT_X, sign * L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE,
               color=L.MAT_GREY)


def _prop(name: str):
    x, y = L.PROP_START[name]
    edge_a, edge_b = L.edge_centers(L.HAND[name])
    return mk.assembly(
        name,
        origin=(x, y, L.TOP + L.MAT_TOP),
        mass=L.PROP_MASS,
        color=L.BLADE_BLACK,
        shapes=[
            mk.cylinder(L.HUB_CENTER, L.HUB_R, L.HUB_H, color=L.HUB_GREY),
            mk.box(L.BLADE_CENTERS[0], L.BLADE_SIZE),
            mk.box(L.BLADE_CENTERS[1], L.BLADE_SIZE),
            mk.box(edge_a, L.EDGE_SIZE, color=L.EDGE_WHITE),
            mk.box(edge_b, L.EDGE_SIZE, color=L.EDGE_WHITE),
        ],
    )


MOTOR_NAMES = [m[0] for m in L.MOTORS]
PLATE = Box("centre_plate", pos=(L.CENTER[0], L.CENTER[1], L.TOP + L.PLATE_Z0 + L.PLATE_SIZE[2] / 2),
            size=L.PLATE_SIZE, color=L.CARBON)
FRAME = [_arm(m) for m in MOTOR_NAMES] + [PLATE] + [_motor(m) for m in MOTOR_NAMES]
ARROWS = [b for m in MOTOR_NAMES for b in _arrow(m)]
MATS = [_mat("left", 1.0), _mat("right", -1.0)]
PROPS = [_prop(n) for n in L.PROPS]
OBJECTS = [*FRAME, *ARROWS, *MATS, *PROPS]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _props(objects) -> dict:
    return {n: objects.get(n) for n in L.PROPS}


def success(objects) -> bool:
    """All four motors fitted: on each, a propeller of that motor's hand standing upright (within 8 degrees) with its
    hub centred on the motor (within 6 mm) and its hub's bottom on the motor's top (within 3 mm), at rest, and no
    propeller of the wrong hand or upside down on any motor; held 2.5 s. This runs first every control step, so it
    also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    props = _props(objects)
    result = _TRACKER.update(props)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["source"] = getattr(props[L.PROPS[0]], "source", None)
        row["origins"] = {n: props[n].origin for n in L.PROPS}
        row["quats"] = {n: props[n].quat_xyzw for n in L.PROPS}
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any propeller of the wrong hand, or any propeller upside down, stands on a motor (hub's axis within 6 mm
    of the motor's, hub's bottom (or top, upside down) within 3 mm of the motor's top, axis within 8 degrees of
    vertical). Otherwise 0.25 for each motor carrying a propeller of its own hand, standing upright on it and at rest.
    Read from the state at hand (the last step)."""
    return _TRACKER.result(_props(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the 22 mm hub.
PICK_OPEN = -0.023  # 46 mm: 12 mm clear of the hub each side; the open fingers' outer faces 13 mm from the next hub
PLACE_OPEN = -0.020  # let go to 40 mm only
GRIP = -(2 * L.HUB_R - 0.008) / 2  # 14 mm commanded across the 22 mm hub: about 20 N per finger
# Every grip closes the fingers along y (yaw 0) across the hub, with the blades lying along x between the fingers'
# planes: the blades are 18 mm wide, the fingers close on the hub at 11 mm either side of its axis. The tool point is
# 26 mm above the hub's bottom, so the pads (from 13 mm below the tool point to 9 mm above it) hold the hub from 13 mm
# up to its top (34 mm), 6 mm above the leading-edge strips' tops (7 mm). The blades stay along x all the way.
YAW = 0.0
GRIP_UP = 0.026
DROP = 0.002  # each propeller is let go with its hub's bottom 2 mm above the motor's top
APPROACH = 0.030  # each final descent starts this far above its end
TRAVEL = L.TOP + 0.120  # carried with the hub's bottom 94 mm up: 22 mm over a fitted propeller's hub (72 mm)
PARK = {"left": (0.200, 0.250, L.TOP + 0.160), "right": (0.200, -0.250, L.TOP + 0.160)}
# (propeller, motor) in order for each arm. The two arms run on one clock with the same timings: the left arm fits
# the front left motor while the right arm fits the rear right one, then the left the rear left while the right the
# front right, so the grippers are never at the same x near the midline (their tool points stay 256 mm apart).
PLAN = {
    "left": (("prop_1", "front_left"), ("prop_2", "rear_left")),
    "right": (("prop_5", "rear_right"), ("prop_4", "front_right")),
}


def _pick(prop: str, first: bool) -> list[Waypoint]:
    """Over the propeller's hub with the fingers at 46 mm, down, settle, grip, lift."""
    x, y = L.PROP_START[prop]
    z = L.TOP + L.MAT_TOP + GRIP_UP
    return [
        Waypoint(2.5 if first else 2.0, (x, y, TRAVEL), PICK_OPEN, f"over {prop}", YAW),
        Waypoint(1.0, (x, y, z + APPROACH), PICK_OPEN, f"down to just above {prop}'s hub", YAW),
        Waypoint(1.5, (x, y, z), PICK_OPEN, f"slowly down around {prop}'s hub", YAW),
        Waypoint(0.5, (x, y, z), PICK_OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, z), GRIP, f"grip {prop}'s hub", YAW),
        Waypoint(1.5, (x, y, TRAVEL), GRIP, f"lift {prop}", YAW),
    ]


def _place(prop: str, motor: str) -> list[Waypoint]:
    """Carry the propeller over the motor, lower it slowly onto the motor's top, let go, rise."""
    x, y = L.MOTOR_XY[motor]
    z = L.TOP + L.MOTOR_TOP + DROP + GRIP_UP
    return [
        Waypoint(2.5, (x, y, TRAVEL), GRIP, f"carry {prop} over the {motor} motor", YAW),
        Waypoint(0.5, (x, y, TRAVEL), GRIP, "settle over the motor", YAW),
        Waypoint(1.0, (x, y, z + APPROACH), GRIP, "down to just above the motor", YAW),
        Waypoint(2.0, (x, y, z), GRIP, f"slowly set {prop} on the {motor} motor", YAW),
        Waypoint(0.5, (x, y, z), GRIP, "settle", YAW),
        Waypoint(1.5, (x, y, z), PLACE_OPEN, f"let go of {prop}", YAW),
        Waypoint(1.0, (x, y, TRAVEL), PLACE_OPEN, "straight up", YAW),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for i, (prop, motor) in enumerate(PLAN[arm]):
        wps += _pick(prop, i == 0) + _place(prop, motor)
    wps.append(Waypoint(2.0, PARK[arm], PLACE_OPEN, "aside", YAW))
    return wps


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
