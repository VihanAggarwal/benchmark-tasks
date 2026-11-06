"""On the station table: an armless adult training torso (tan, 320 x 380 x 60 mm, fixed) lies on its back with its
head away from the arms, its chest top bare, with the sternal notch, breastbone, collarbones and nipples drawn on
it in dark brown; a fixed yellow-green AED unit sits between the arm bases; and two identical white AED training
pads (110 x 70 x 18 mm, 40 g) lie flat on the table beside the torso, one on each arm's side. Rule (the
anterolateral pad position of the resuscitation guidelines): one pad on the manikin's upper right chest just below
the collarbone, beside the breastbone; the other on its left side below the armpit; the pads must not touch; then
let go of them. The manikin's right is the arms' left.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

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
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Both pads still for 1 s, with both grippers' finger zones at least 10 mm clear of them.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Put the two AED training pads on the manikin's bare chest in the standard anterolateral position: one on "
    "the manikin's upper right chest just below the collarbone, beside the breastbone, and the other on the left "
    "side of its chest below the armpit. The pads must not touch. Then let go of them."
)

_mark_z = L.CHEST_TOP + L.MARK_T / 2
TORSO = Box("torso", pos=(L.TORSO_X, 0.0, L.TOP + L.TORSO_H / 2), size=L.TORSO_SIZE, color=L.TAN)
NECK = Box("neck", pos=(L.NECK_X, 0.0, L.TOP + L.NECK_SIZE[2] / 2), size=L.NECK_SIZE, color=L.TAN)
HEAD = Cylinder("head", pos=(L.HEAD_X, 0.0, L.TOP + L.HEAD_H / 2), radius=L.HEAD_RADIUS, height=L.HEAD_H,
                color=L.TAN)
LANDMARKS = [
    Box("sternal_notch", pos=(L.NOTCH_X, 0.0, _mark_z), size=L.NOTCH_SIZE, color=L.BROWN),
    Box("breastbone", pos=(L.STERNUM_X, 0.0, _mark_z), size=L.STERNUM_SIZE, color=L.BROWN),
    Box("collarbone_right", pos=(L.CLAVICLE_X, L.CLAVICLE_Y, _mark_z), size=L.CLAVICLE_SIZE, color=L.BROWN),
    Box("collarbone_left", pos=(L.CLAVICLE_X, -L.CLAVICLE_Y, _mark_z), size=L.CLAVICLE_SIZE, color=L.BROWN),
    Cylinder("nipple_right", pos=(L.NIPPLE_X, L.NIPPLE_Y, _mark_z), radius=L.NIPPLE_RADIUS, height=L.MARK_T,
             color=L.BROWN),
    Cylinder("nipple_left", pos=(L.NIPPLE_X, -L.NIPPLE_Y, _mark_z), radius=L.NIPPLE_RADIUS, height=L.MARK_T,
             color=L.BROWN),
]
AED = Box("aed_unit", pos=(L.AED_X, L.AED_Y, L.TOP + L.AED_SIZE[2] / 2), size=L.AED_SIZE, color=L.AED_GREEN)
PADS = [
    Box(name, pos=(x, y, L.TOP + L.PAD_SIZE[2] / 2), size=L.PAD_SIZE, mass=L.PAD_MASS, color=L.PAD_WHITE, yaw=yaw)
    for name, (x, y, yaw) in L.PAD_START.items()
]
OBJECTS = [TORSO, NECK, HEAD, *LANDMARKS, AED, *PADS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _pads(objects) -> dict:
    return {name: objects[name] for name in L.PADS}


def success(objects) -> bool:
    """A pad lying flat on the chest with its centre within 25 mm of each of the two anterolateral positions,
    both grippers clear of both pads, the pads at least 10 mm apart; held still 1 s. This runs first every control
    step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(_pads(objects), arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the two pads' outlines seen from above are less than 10 mm apart. Otherwise, for each of the sternal
    and apical positions, 0.5 if a pad that lies on the chest (centre over it, tilted at most 10 deg, lowest point
    within 5 mm of the chest top) with both grippers' finger zones at least 10 mm from it has its centre within
    25 mm of the position, 0.25 within 40 mm; the score is the sum (0, 0.25, 0.5, 0.75, 1.0)."""
    return _TRACKER.result(_pads(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PAD_GRIP = -(L.PAD_SIZE[1] - 0.008) / 2  # across the pad's 70 mm width: 62 mm commanded
# Both arms grip their pad across its 70 mm width: the left pad lies across (long side along y), so the left
# fingers close along x (yaw -90 deg, the sign that keeps joint 6 well inside its limits at the sternal position);
# the right pad lies along x and the right fingers close along y (yaw 0). Neither grip turns on the way.
YAW_L, YAW_R = -math.pi / 2, 0.0
GRIP_UP = 0.018  # the TCP this far above the pad's underside: finger tips 5 mm above it
PICK_Z = L.TOP + GRIP_UP
LIFT_Z = L.TOP + 0.110  # pad underside 92 mm above the table, 32 mm above the chest top
CARRY_Z = L.CHEST_TOP + 0.045  # over the target: pad underside 27 mm above the chest top
# Let go with the pad's underside 2 mm above the chest top: the 40 g pad drops flat, and the finger tips stay
# 7 mm above the chest (6 mm above the 1 mm landmark lines) as the fingers open.
PLACE_Z = L.CHEST_TOP + 0.002 + GRIP_UP
ENTRY_Z = PLACE_Z + 0.015  # the last 15 mm are lowered slowly
UP_Z = CARRY_Z  # straight up after letting go: finger tips 14 mm above the pad's top (the left arm's reach ends
#                 about 25 mm higher at the sternal position)
# Where the hands wait at the end: open, 16 cm up, over the table beside the torso (whose sides are at y +-0.19).
# Each open finger zone is then more than 30 mm from the pad it placed (10 mm needed; straight above the sternal
# position the left arm reaches only about 130 mm up, so it moves back toward its base as it rises).
REST = {"left": (0.240, 0.240, L.TOP + 0.160), "right": (0.240, -0.240, L.TOP + 0.160)}
# Payload sag: the 40 g pad loads the arm with 0.4 N; the compliance model (J K^-1 J^T, joints
# 80/80/80/10/10/10 N m/rad) puts the gripper about 1 mm low at the place poses, so no trim is applied. The pads
# are scored to 25 mm. On the GPU read pad_left_to_sternal_mm and pad_right_to_apical_mm in TASK_TRACE.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _plan(side: str, target: str, yaw: float) -> list[Waypoint]:
    """Pick the pad on this arm's side, carry it over its target, lower it slowly, let go, move clear."""
    px, py, _ = L.PAD_START[f"pad_{side}"]
    tx, ty = L.TARGETS[target]
    tx, ty = tx + PLACE_TRIM[side][0], ty + PLACE_TRIM[side][1]
    return [
        Waypoint(1.5, (px, py, LIFT_Z), OPEN, "over the pad", yaw),
        Waypoint(1.5, (px, py, PICK_Z), OPEN, "down beside the pad", yaw),
        Waypoint(0.5, (px, py, PICK_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (px, py, PICK_Z), PAD_GRIP, "grip the pad across its width", yaw),
        Waypoint(0.3, (px, py, PICK_Z), PAD_GRIP, "settle the grip", yaw),
        Waypoint(1.2, (px, py, LIFT_Z), PAD_GRIP, "lift", yaw),
        Waypoint(2.0, (tx, ty, CARRY_Z), PAD_GRIP, f"carry over the {target} position", yaw),
        Waypoint(1.0, (tx, ty, ENTRY_Z), PAD_GRIP, "down to 15 mm above the chest", yaw),
        Waypoint(1.5, (tx, ty, PLACE_Z), PAD_GRIP, "lower the last 15 mm slowly", yaw),
        Waypoint(0.3, (tx, ty, PLACE_Z), PAD_GRIP, "settle", yaw),
        Waypoint(1.5, (tx, ty, PLACE_Z), OPEN, "let go", yaw),
        Waypoint(0.3, (tx, ty, PLACE_Z), OPEN, "settle", yaw),
        Waypoint(1.0, (tx, ty, UP_Z), OPEN, "straight up", yaw),
        Waypoint(1.5, REST[side], OPEN, "clear of the pads", yaw),
    ]


def demo():
    """Both arms on one clock, each placing the pad on its own side. DEMO_HOLD_ONLY=1: no plan (both arms hold the
    ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left", "sternal", YAW_L), "right": _plan("right", "apical", YAW_R)}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
