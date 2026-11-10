"""On the station table, at real size: a passage floor (light grey vinyl, 340 x 500 mm) between two white skirting
boards (60 mm tall, along x at y +-25 cm, the wall above them cut away), with two dark brown droppings patches on the
floor along each skirting, 5-15 mm from its face. Four wooden mouse snap traps (98 x 45 mm, unarmed: the bail has
sprung and lies over the yellow trigger pedal) lie in the middle of the passage, long sides along x, two on each side
of the midline, the nearer two with their pedal ends toward the arms, the farther two away from them. Rule: set one
trap at each droppings patch the way pest controllers do: square to the wall, its pedal (trigger) end toward the
wall and 0-8 mm from it, at the patch.

The traps are loose mechkit compounds (mechkit.py, next to this file): each is one rigid body of six boxes (pine
base, yellow pedal, steel spring block and bail), so its pedal end can be seen and the scorer reads its frame
(origin and orientation). The scorer reads only the traps.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the traps are built as compound bodies)
from scoring import NAMES, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Held for 2.5 s, not the default 1 s. The scorer cannot see the grippers, so the last traps count as set while still
# gripped 2 mm above the floor, 0.5 s after they stop; with 2.5 s the trial cannot end before the fingers have opened
# (0.3 + 1.5 s after the stop), so the end state is the traps lying free.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Mice are running along both walls of this passage; the dark brown droppings mark their runs. Set the four "
    "snap traps (they are not armed) the way pest controllers do: one trap at each droppings patch, square to the "
    "wall, with its trigger end (the yellow pedal) toward the wall and no more than 8 mm from the skirting board."
)


def _fixed_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


FLOOR = _fixed_box("passage_floor", L.FLOOR_X0, L.FLOOR_X1, -L.FACE_Y, L.FACE_Y, 0.0, L.FLOOR_SIZE[2], L.VINYL_GREY)
SKIRTINGS = [
    _fixed_box(f"skirting_{wall}", L.FLOOR_X0, L.FLOOR_X1, *sorted((k * L.FACE_Y, k * (L.FACE_Y + L.SKIRT_THICK))),
               0.0, L.SKIRT_H, L.SKIRT_WHITE)
    for wall, k in L.WALLS.items()
]
PATCHES = [
    Box(name, pos=(*L.patch_xy(name), L.FLOOR_TOP + L.PATCH_SIZE[2] / 2), size=L.PATCH_SIZE, color=L.DROPPINGS_BROWN)
    for name in L.PATCHES
]
TRAPS = [
    mk.assembly(
        name,
        origin=(L.TRAP_START[name][0], L.TRAP_START[name][1], L.FLOOR_TOP),
        yaw=L.TRAP_START[name][2],
        mass=L.TRAP_MASS,
        color=L.PINE,
        shapes=[mk.box(c, s, color=col) for c, s, col in L.TRAP_SHAPES],
    )
    for name in NAMES
]
OBJECTS = [FLOOR, *SKIRTINGS, *PATCHES, *TRAPS]

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


def success(objects) -> bool:
    """A trap set at every droppings patch: each lying still, base down on the floor, its pedal end pointing within
    15 degrees of straight at the patch's wall, that end 0-8 mm from the skirting's face and its midpoint within 25 mm
    of the patch's centre along the wall; held 2.5 s. This runs first every control step, so it also feeds the
    tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    states = {n: objects.get(n) for n in NAMES}
    result = _TRACKER.update(states)
    row = {"mechkit_episode": mk.episode(), "step": _TRACKER.steps, "score": result.score,
           "success": result.success, **result.details}
    raw = {}
    for n, s in states.items():
        try:
            raw[n] = {"origin": s.origin, "quat": s.quat_xyzw, "vel": s.lin_vel, "source": getattr(s, "source", None)}
        except AttributeError:
            raw[n] = "unreadable"
    row["raw"] = raw
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each droppings patch with a trap set at it: lying still, base down (tilted at most 10 degrees, its
    bottom within 3 mm of the floor), its pedal end pointing within 15 degrees of straight at the patch's wall, that
    end's nearest point 0-8 mm from the skirting's face and its midpoint within 25 mm of the patch's centre, measured
    along the wall. 0 if no patch has a trap set at it. Read from the state at hand (the last step)."""
    return _TRACKER.result({n: objects.get(n) for n in NAMES}).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Both arms at once, each on its own side of the passage: each sets the two traps on its side at the two patches on
# its own wall. A trap is gripped at its middle across its 45 mm base (yaw 0: the fingers close along y, the base
# lying along x), lifted, turned a quarter turn (whichever way brings its pedal end round to the wall) while it is
# carried, and set down square to the wall with its pedal end aimed END_AIM from the skirting's face.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.035  # 70 mm: 12.5 mm each side of the base
GRIP = -(L.TRAP_W - 0.008) / 2  # 37 mm commanded across the 45 mm base
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_Z = L.FLOOR_TOP + 0.016  # pad tips 3 mm above the floor: 7 mm of the base's 10 mm sides
DROP = 0.002  # traps are let go this far above the floor
HOVER = L.TOP + 0.085  # travel height: a carried trap's underside 67 mm above the floor, 7 mm over the skirting's
# top (so a carry that overshoots toward the wall passes over it) and 50 mm over the other traps
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
END_AIM = 0.004  # the pedal end set this far from the skirting's face (the band is 0-8 mm)
REST = (0.250, 0.200, L.TOP + 0.150)  # where each arm waits at the end, clear of the traps
SIDE = {"left": 1.0, "right": -1.0}
# Which trap goes to which patch, per arm, in order.
PLAN = {
    "left": (("trap_1", "patch_left_near"), ("trap_2", "patch_left_far")),
    "right": (("trap_3", "patch_right_near"), ("trap_4", "patch_right_far")),
}
# Trim (x, y) of each trap's place point, for the GPU run: if a trap lands off its aim (in TASK_TRACE end_gap_mm
# away from 4 or along_mm away from 0), move its place point by minus the offset.
PLACE_TRIM = {n: (0.0, 0.0) for n in NAMES}


def place_point(trap: str, patch: str) -> tuple[float, float]:
    """Where the trap's middle (and the tool point) goes: its pedal end's midpoint level with the patch's centre
    along the wall and END_AIM from the wall's face."""
    px, wall = L.PATCHES[patch]
    k = L.WALLS[wall]
    tx, ty = PLACE_TRIM[trap]
    return px + tx, k * (L.FACE_Y - END_AIM - L.TRAP_HALF_LEN) + ty


def turn(trap: str, patch: str) -> float:
    """The quarter turn (rad) that brings this trap's pedal end round from where it starts to point at the patch's
    wall: its plain end (+x of its frame) must point away from the wall."""
    wall = L.PATCHES[patch][1]
    want = math.atan2(-L.WALLS[wall], 0.0)  # heading of the plain end: -y for the left wall, +y for the right
    return math.remainder(want - L.TRAP_START[trap][2], 2 * math.pi)


def _set_trap(trap: str, patch: str, first: bool) -> list[Waypoint]:
    """Over the trap, down (slowly for the last 30 mm), grip, lift, carry to the patch turning it square to the
    wall, down (slowly), let go, up."""
    sx, sy, _ = L.TRAP_START[trap]
    dx, dy = place_point(trap, patch)
    yaw = turn(trap, patch)
    place_z = GRIP_Z + DROP
    return [
        Waypoint(2.0 if first else 1.5, (sx, sy, HOVER), OPEN, "over the trap", 0.0),
        Waypoint(1.0, (sx, sy, GRIP_Z + APPROACH), OPEN, "down to just above the trap", 0.0),
        Waypoint(1.5, (sx, sy, GRIP_Z), OPEN, "slowly down round the trap", 0.0),
        Waypoint(0.5, (sx, sy, GRIP_Z), OPEN, "settle", 0.0),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip the trap", 0.0),
        Waypoint(0.3, (sx, sy, GRIP_Z), GRIP, "settle the grip", 0.0),
        Waypoint(1.2, (sx, sy, HOVER), GRIP, "lift the trap", 0.0),
        Waypoint(2.0, (dx, dy, HOVER), GRIP, "carry it to the patch, turning it square to the wall", yaw),
        Waypoint(0.5, (dx, dy, HOVER), GRIP, "settle over the patch", yaw),
        Waypoint(1.0, (dx, dy, place_z + APPROACH), GRIP, "down to just above the floor", yaw),
        Waypoint(1.5, (dx, dy, place_z), GRIP, "slowly set the trap down", yaw),
        Waypoint(0.3, (dx, dy, place_z), GRIP, "settle", yaw),
        Waypoint(1.5, (dx, dy, place_z), OPEN, "let go of the trap", yaw),
        Waypoint(1.0, (dx, dy, HOVER), OPEN, "straight up", yaw),
    ]


def _arm(side: str) -> list[Waypoint]:
    """One arm, on its own side: its two traps to the two patches on its wall, then up and out of the way."""
    plan: list[Waypoint] = []
    for i, (trap, patch) in enumerate(PLAN[side]):
        plan += _set_trap(trap, patch, i == 0)
    plan.append(Waypoint(1.5, (REST[0], SIDE[side] * REST[1], REST[2]), OPEN, "out of the way", 0.0))
    return plan


def demo():
    """Both arms on one clock, each on its own side of the midline: the two tool points never come closer than
    0.26 m along y (traps start at y +-0.13 and go out to +-0.197). DEMO_HOLD_ONLY=1: no plan (both arms hold the
    ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
