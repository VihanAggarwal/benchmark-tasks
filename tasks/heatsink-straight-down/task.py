"""On the station table: a green circuit board (200 x 160 x 4 mm, fixed) centred on the midline 30 cm in front of
the arm bases, with a dark grey processor die (30 x 30 x 3 mm) at its centre and four short black retention posts
(8 mm across, 14 mm tall) at the corners of a 68 mm square around the die; and a loose aluminium heatsink block
(56 x 56 x 32 mm, 0.20 kg) on the table 22 cm in front, 22 cm left of the midline. Rule: the heatsink goes onto the die
straight down, slowly, and once it touches the die it must not slide or turn on it (that smears the paste).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the task needs only one; the demo uses the left arm and the right arm holds its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The arm sags about 5.6 mm under the load, so the heatsink rests on the die
# about 0.8 s before the slow descent ends (t 11.7 s) and success starts counting while the gripper still holds it.
# The pads clear the heatsink at t 12.9 s and are fully open at 14.0 s. With 2.5 s the trial ends at about 14.2 s
# (step 284) with the fingers fully open; even with 15 mm of sag it ends with them 78 mm apart, clear of the 56 mm
# heatsink (2.0 s would end with them just touching it). With no sag at all it ends at about step 311.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Place the aluminium heatsink on the processor inside the four posts. "
    "Lower it straight down and do not slide it once it touches."
)

BOARD = Box("board", pos=(L.BOARD_X, L.BOARD_Y, L.TOP + L.BOARD_SIZE[2] / 2), size=L.BOARD_SIZE, color=L.PCB_GREEN)
DIE = Box("processor", pos=(L.DIE_X, L.DIE_Y, L.BOARD_TOP + L.DIE_SIZE[2] / 2), size=L.DIE_SIZE, color=L.DIE_GREY)
POSTS = [
    Cylinder(name, pos=(x, y, L.BOARD_TOP + L.POST_HEIGHT / 2), radius=L.POST_RADIUS, height=L.POST_HEIGHT,
             color=L.POST_BLACK)
    for name, x, y in L.POSTS
]
HEATSINK = Box(
    "heatsink",
    pos=(L.HS_X, L.HS_Y, L.TOP + L.HS_SIZE[2] / 2),
    size=L.HS_SIZE,
    mass=L.HS_MASS,
    color=L.ALUMINIUM,
)
OBJECTS = [BOARD, DIE, *POSTS, HEATSINK]

_TRACKER = Tracker()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def success(objects) -> bool:
    """Heatsink flat on the die, centred within 2 mm and square within 3 degrees, slid no more than 1 mm while
    touching, set down at 0.10 m/s or less; held still 2.5 s. This runs first every control step, so it also feeds
    the tracker."""
    hs = objects["heatsink"]
    result = _TRACKER.update(hs)
    _trace({"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details,
            "heatsink_pos": hs.pos, "heatsink_vel": hs.lin_vel})
    return result.success


def score(objects) -> float:
    """0 if the heatsink ends off the board, tilted over 30 degrees or resting on a post. Else 1.0 flat on the die,
    centred within 2 mm, square within 3 degrees, smear <= 1 mm, set-down <= 0.10 m/s; 0.75 the same placement with
    smear up to 5 mm or a faster set-down; 0.5 flat on the die within 5 mm and 10 degrees; 0.25 on the die."""
    return _TRACKER.result(objects["heatsink"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
HS_GRIP = -(L.HS_SIZE[1] - 0.008) / 2  # yaw 0: the fingers close along y, across the 56 mm
GRIP_UP = 0.020  # the TCP this far above the heatsink's bottom: pad tips 7 mm above it
CARRY_Z = L.TOP + 0.12
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.DIE_TOP + 0.001 + GRIP_UP  # let go with the heatsink's bottom 1 mm above the die
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm are lowered slowly
# Payload sag: the arm's controller does not compensate the 0.20 kg heatsink's weight. The static estimate
# (J K^-1 J^T of the weight; joints 80/80/80/10/10/10 N m/rad) over the die is (-1.9, +1.9, -5.6) mm and a 0.9 deg
# tilt. The vertical part only makes the heatsink touch the die a few mm before the slow descent ends (the die
# then takes most of its weight; the sideways spring force is 0.19 of the vertical, well inside friction, so it
# does not slide). That figure assumes the heatsink rocks with the wrist as the load comes off: the die's 30 mm top
# can resist only about 0.03 N m, against wrist moments near 1 N m, so it does; were it held rigidly flat the ratio
# would be 1.7, past friction. Rocking flat from the 0.9 deg tilt moves its centre about 0.2-0.3 mm, which counts
# as smear (limit 1 mm). The sideways part would set it down 2.7 mm off centre. PLACE_TRIM moves the place point by
# half of it, so the heatsink lands within 2 mm of the centre either way: about 1.3 mm off if the arm sags as
# estimated, 1.4 mm off if it does not sag at all (the offline kinematic check). On the GPU, read centre_off_mm
# in TASK_TRACE and adjust (with the full trim the offline check, which has no sag, predicts 0.5).
PLACE_TRIM = (0.0010, -0.0010)


def _left() -> list[Waypoint]:
    """Left arm: pick the heatsink, carry it high over the die, stop, lower it slowly straight down, let go."""
    hx, hy = L.HS_X, L.HS_Y
    dx, dy = L.DIE_X + PLACE_TRIM[0], L.DIE_Y + PLACE_TRIM[1]
    return [
        Waypoint(1.5, (hx, hy, CARRY_Z), OPEN, "over the heatsink"),
        Waypoint(1.5, (hx, hy, PICK_Z), OPEN, "down to the heatsink"),
        Waypoint(0.5, (hx, hy, PICK_Z), OPEN, "settle"),
        Waypoint(1.5, (hx, hy, PICK_Z), HS_GRIP, "grip the heatsink"),
        Waypoint(1.2, (hx, hy, CARRY_Z), HS_GRIP, "lift"),
        Waypoint(1.8, (dx, dy, CARRY_Z), HS_GRIP, "carry over the die"),
        Waypoint(1.2, (dx, dy, ENTRY_Z), HS_GRIP, "down to 2 cm above"),
        Waypoint(0.8, (dx, dy, ENTRY_Z), HS_GRIP, "settle over the die"),
        Waypoint(2.5, (dx, dy, PLACE_Z), HS_GRIP, "lower the last 2 cm slowly"),
        Waypoint(1.5, (dx, dy, PLACE_Z), OPEN, "let go"),
        Waypoint(1.0, (dx, dy, CARRY_Z), OPEN, "straight up"),
    ]


def demo():
    """One arm: the right arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
