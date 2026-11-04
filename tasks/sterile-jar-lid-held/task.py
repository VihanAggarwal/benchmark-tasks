"""On the station table: a white square sterile jar (66 x 66 mm, 70 mm tall, open top, clamped in place) on the
midline 26 cm in front of the arm bases, closed by a loose blue lid (74 x 74 x 18 mm) resting on its rim; and a
yellow 24 mm sample cube on the table 20 cm to the right of the midline. Aseptic rule: once the lid is off the
jar it must stay in the air, never set down, and the jar must not stay open longer than needed.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
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
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Take the lid off the sterile jar and keep holding it in the air, put the yellow cube into the jar, "
    "then put the lid back on. Never set the lid down."
)

_h = L.JAR_HEIGHT / 2
_o, _w = L.JAR_OUTER / 2, L.JAR_WALL
_side_len = L.JAR_OUTER - 2 * L.JAR_WALL
JAR = [  # four fixed walls; the table top is the jar's floor
    Box("jar_wall_front", pos=(L.JAR_X + _o - _w / 2, L.JAR_Y, L.TOP + _h), size=(_w, L.JAR_OUTER, L.JAR_HEIGHT), color=L.WHITE),
    Box("jar_wall_back", pos=(L.JAR_X - _o + _w / 2, L.JAR_Y, L.TOP + _h), size=(_w, L.JAR_OUTER, L.JAR_HEIGHT), color=L.WHITE),
    Box("jar_wall_left", pos=(L.JAR_X, L.JAR_Y + _o - _w / 2, L.TOP + _h), size=(_side_len, _w, L.JAR_HEIGHT), color=L.WHITE),
    Box("jar_wall_right", pos=(L.JAR_X, L.JAR_Y - _o + _w / 2, L.TOP + _h), size=(_side_len, _w, L.JAR_HEIGHT), color=L.WHITE),
]
LID = Box("lid", pos=(L.JAR_X, L.JAR_Y, L.RIM_Z + L.LID_SIZE[2] / 2), size=L.LID_SIZE, mass=L.LID_MASS, color=L.BLUE)
CUBE = Box(
    "sample_cube",
    pos=(L.CUBE_X, L.CUBE_Y, L.TOP + L.CUBE_EDGE / 2),
    size=(L.CUBE_EDGE,) * 3,
    mass=L.CUBE_MASS,
    color=L.YELLOW,
)
OBJECTS = [*JAR, LID, CUBE]

_TRACKER = Tracker()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def success(objects) -> bool:
    """Cube in the jar, lid seated, the jar open 20 s or less in all, the lid never set down; held still 1 s.
    This runs first every control step, so it also feeds the tracker."""
    result = _TRACKER.update(objects["lid"], objects["sample_cube"])
    _trace({"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details,
            "lid_pos": objects["lid"].pos, "cube_pos": objects["sample_cube"].pos})
    return result.success


def score(objects) -> float:
    """0 if the lid, while off the jar, ever came within 30 mm of the table. Else 1.0 cube in, lid seated, open
    <= 20 s; 0.75 the same, open longer; 0.5 cube in, lid not seated; 0.25 lid lifted 30 mm clear; 0 otherwise."""
    return _TRACKER.result(objects["lid"], objects["sample_cube"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
LID_GRIP = -(L.LID_SIZE[1] - 0.008) / 2
CUBE_GRIP = -(L.CUBE_EDGE - 0.008) / 2
HOVER_L, HOVER_R = L.TOP + 0.15, L.TOP + 0.14
LID_TCP_Z = L.RIM_Z + 0.004 + 0.013  # pad tips 4 mm above the lid's bottom face (13 mm below the TCP)
CUBE_TCP_Z = L.TOP + 0.004 + 0.013
PARK = (L.JAR_X, L.JAR_Y + 0.20)  # where the left arm holds the lid while the jar is open: 20 cm left of it
DROP_Z = L.TOP + 0.105  # cube bottom 18 mm above the rim when it is let go


def _left() -> list[Waypoint]:
    """Left arm: lift the lid straight up, hold it in the air 20 cm to the left, wait, put it back."""
    jx, jy = L.JAR_X, L.JAR_Y
    return [
        Waypoint(1.5, (jx, jy, HOVER_L), OPEN, "over the lid"),
        Waypoint(1.5, (jx, jy, LID_TCP_Z), OPEN, "down to the lid"),
        Waypoint(1.5, (jx, jy, LID_TCP_Z), LID_GRIP, "grip the lid"),
        Waypoint(1.2, (jx, jy, HOVER_L), LID_GRIP, "lift it straight up"),
        Waypoint(1.5, (*PARK, HOVER_L), LID_GRIP, "hold it aside, in the air"),
        Waypoint(6.6, (*PARK, HOVER_L), LID_GRIP, "wait for the cube"),
        Waypoint(1.5, (jx, jy, HOVER_L), LID_GRIP, "back over the jar"),
        Waypoint(1.5, (jx, jy, LID_TCP_Z + 0.002), LID_GRIP, "lower onto the rim"),
        Waypoint(1.5, (jx, jy, LID_TCP_Z + 0.002), OPEN, "let go"),
        Waypoint(1.0, (jx, jy, HOVER_L), OPEN, "up"),
    ]


def _right() -> list[Waypoint]:
    """Right arm: pick the cube, wait until the lid is out of the way, drop the cube in, go back."""
    cx, cy = L.CUBE_X, L.CUBE_Y
    jx, jy = L.JAR_X, L.JAR_Y
    return [
        Waypoint(1.5, (cx, cy, HOVER_R), OPEN, "over the cube"),
        Waypoint(1.5, (cx, cy, CUBE_TCP_Z), OPEN, "down to the cube"),
        Waypoint(1.5, (cx, cy, CUBE_TCP_Z), CUBE_GRIP, "grip the cube"),
        Waypoint(1.2, (cx, cy, HOVER_R), CUBE_GRIP, "lift"),
        Waypoint(1.6, (cx, cy, HOVER_R), CUBE_GRIP, "wait for the lid to be clear"),
        Waypoint(1.5, (jx, jy, HOVER_R), CUBE_GRIP, "over the open jar"),
        Waypoint(1.0, (jx, jy, DROP_Z), CUBE_GRIP, "lower to just above the rim"),
        Waypoint(1.5, (jx, jy, DROP_Z), OPEN, "let the cube drop in"),
        Waypoint(1.0, (jx, jy, HOVER_R), OPEN, "up"),
        Waypoint(1.5, (cx, cy, HOVER_R), OPEN, "out of the way"),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
