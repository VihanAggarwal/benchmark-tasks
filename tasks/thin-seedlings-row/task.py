"""On the station table: a brown seed bed (520 x 60 x 10 mm, fixed) running across the table 25 cm in front of the
arm bases, with ten green seedlings standing in a row on it (upright cylinders 20 mm across, 45 to 90 mm tall,
40 to 70 mm apart); and two black compost trays (100 x 100 mm inside, 40 mm walls, fixed), one beside each arm's
base. Rule: thin the row to one seedling per 10 cm. Going from the tallest down, keep a seedling if it is at least
10 cm from every seedling kept; pull out all the others and put them in a compost tray; do not disturb the ones
kept. The rule keeps four (y +195, +90, -40 and -240 mm) and pulls six, three on each side of the table.

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
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # every seedling still for 1 s

INSTRUCTION = (
    "Thin the row of seedlings. Going from the tallest seedling down, keep a seedling if it is at least 10 cm "
    "(centre to centre) from every seedling you are keeping; pull out all the others and put them in the black "
    "compost trays. Don't disturb the seedlings you keep."
)


def _walls(prefix: str, x: float, y: float, inner: float, wall: float, height: float, color) -> list[Box]:
    """Four fixed walls around an ``inner`` x ``inner`` opening centred at (x, y); the table is the floor."""
    z = L.TOP + height / 2
    o = inner / 2 + wall / 2
    return [
        Box(f"{prefix}_wall_front", pos=(x + o, y, z), size=(wall, inner + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_back", pos=(x - o, y, z), size=(wall, inner + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_left", pos=(x, y + o, z), size=(inner, wall, height), color=color),
        Box(f"{prefix}_wall_right", pos=(x, y - o, z), size=(inner, wall, height), color=color),
    ]


BED = Box("seed_bed", pos=(L.BED_X, L.BED_Y, L.TOP + L.BED_SIZE[2] / 2), size=L.BED_SIZE, color=L.SOIL)
SEEDLINGS = [
    Cylinder(n, pos=L.start_pos(n), radius=L.SEEDLING_RADIUS, height=L.HEIGHT[n], mass=L.MASS[n], color=L.LEAF)
    for n in L.NAMES
]
TRAYS = [w for side, (x, y) in L.TRAYS.items()
         for w in _walls(f"tray_{side}", x, y, L.TRAY_INNER, L.TRAY_WALL, L.TRAY_HEIGHT, L.TRAY_BLACK)]
OBJECTS = [BED, *SEEDLINGS, *TRAYS]

_TRACKER = Tracker()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _seedlings(objects) -> dict:
    return {n: objects.get(n) for n in L.NAMES} if hasattr(objects, "get") else {}


def success(objects) -> bool:
    """The bed holds exactly the four seedlings the rule keeps, none ever disturbed, and the six others are in the
    compost trays; every seedling still for 1 s. This runs first every control step, so it also feeds the tracker."""
    states = _seedlings(objects)
    result = _TRACKER.update(states)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    if result.details.get("valid_state"):
        row["pos"] = {n: states[n].pos for n in L.NAMES}
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a seedling is off the table, a seedling left on the bed was ever tilted over 15 degrees or moved over
    15 mm from its spot, or fewer than three are left on the bed. Else 1.0 the bed holds exactly the rule's four
    keepers, each within 5 mm of its spot, and the six others are in a tray; 0.75 the same with one keeper moved
    5-15 mm, or with one keeper swapped for a shorter neighbour (10 cm spacing kept); 0.5 the seedlings left are at
    least 10 cm apart; 0.25 at least three of the six to pull are in a tray and all four keepers are on the bed;
    0 otherwise."""
    return _TRACKER.result(_seedlings(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(2 * L.SEEDLING_RADIUS - 0.008) / 2  # 12 mm commanded on a 20 mm stem
# At the row both grippers close along x, across the row (yaw -pi/2 left, +pi/2 right, the signs that reach the
# midline): the open fingers come down in front of and behind the bed, never into the 20-50 mm gaps between
# neighbours. Over the trays the yaw is turned 45 degrees toward the arm's own side (the row's yaw would run joint6
# into its limit out there); a gripped cylinder turns with it and stays the same.
YAW_ROW = {"left": -math.pi / 2, "right": math.pi / 2}
YAW_TRAY = {"left": -math.pi / 4, "right": math.pi / 4}
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.010  # the finger tips close at least 10 mm above the bed's top ...
TOP_ABOVE_TCP = 0.038  # ... and high enough that the stem's top stays at most 38 mm above the TCP, clear of the
#                        gripper's palm (the 65 and 70 mm seedlings are gripped 4 and 9 mm higher)


def grip_z(name: str) -> float:
    """TCP height for gripping a seedling's stem."""
    return L.BED_TOP + max(GRIP_UP + PAD_TIPS, L.HEIGHT[name] - TOP_ABOVE_TCP)


# Over the row the TCP travels at CARRY_Z: a carried seedling's bottom is then 108-117 mm up, over every seedling's
# top (100 mm at most), and on the way to a tray it passes at least 16 mm over any seedling it comes within 25 mm of
# (horizontally); the open finger tips travel 127 mm up.
CARRY_Z = L.TOP + 0.14
OVER_TRAY_Z = L.TOP + 0.12  # over the trays the TCP comes down to here (more wrist margin out by the bases)
RELEASE_Z = L.TOP + 0.09  # let go with the seedling's bottom 58-67 mm up (18-27 mm over the 40 mm rim) and the finger
#                            tips 77 mm up, 12 mm over the tallest seedling already standing in that tray (65 mm)
# The order: each arm thins its own half, the left from the middle out, the right from its end in. The only pair of
# targets less than 25 cm apart that the two arms work on at the same stage is the left's first (y +20 mm) and the
# right's first (y -190 mm), 21 cm apart; so the right arm first goes up over its tray and waits there until the left
# gripper has pulled its first seedling up and is carrying it away.
ORDER = {"left": ("seedling_05", "seedling_03", "seedling_01"),  # y +20, +130, +240 mm
         "right": ("seedling_09", "seedling_08", "seedling_07")}  # y -190, -150, -95 mm
# Three drop spots per tray, 45-56 mm apart and 25-35 mm from the tray's centre, one per seedling.
_SPOTS = ((0.0, 0.025), (0.025, -0.025), (-0.020, -0.025))
WAIT_Z = OVER_TRAY_Z
RIGHT_WAIT_UNTIL = 6.5  # s: the right arm leaves its waiting place over its tray at this time


def spots(side: str) -> list[tuple[float, float]]:
    tx, ty = L.TRAYS[side]
    s = 1.0 if side == "left" else -1.0
    return [(tx + dx, ty + s * dy) for dx, dy in _SPOTS]


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _pull(name: str, spot, side: str, first: bool) -> list[Waypoint]:
    """Pull one seedling straight up and drop it in the tray."""
    x, y = L.BED_X, L.START_Y[name]
    sx, sy = spot
    yr, yt = YAW_ROW[side], YAW_TRAY[side]
    gz = grip_z(name)
    return [
        Waypoint(2.0 if first else 1.8, (x, y, CARRY_Z), OPEN, f"over {name}", yr),
        Waypoint(1.5, (x, y, gz), OPEN, "down, fingers in front of and behind it", yr),
        Waypoint(0.5, (x, y, gz), OPEN, "settle", yr),
        Waypoint(1.5, (x, y, gz), GRIP, "grip the stem", yr),
        Waypoint(1.5, (x, y, CARRY_Z), GRIP, "pull it straight up", yr),
        Waypoint(2.0, (sx, sy, OVER_TRAY_Z), GRIP, "carry it over the compost tray", yt),
        Waypoint(0.8, (sx, sy, RELEASE_Z), GRIP, "lower it toward the tray", yt),
        Waypoint(0.4, (sx, sy, RELEASE_Z), GRIP, "settle", yt),
        Waypoint(1.5, (sx, sy, RELEASE_Z), OPEN, "let it drop into the tray", yt),
        Waypoint(0.8, (sx, sy, OVER_TRAY_Z), OPEN, "up", yt),
    ]


def _plan(side: str) -> list[Waypoint]:
    out: list[Waypoint] = []
    if side == "right":
        tx, ty = L.TRAYS["right"]
        out += [Waypoint(2.0, (tx, ty, WAIT_Z), OPEN, "up over the compost tray", YAW_TRAY["right"]),
                Waypoint(RIGHT_WAIT_UNTIL - 2.0, (tx, ty, WAIT_Z), OPEN, "wait while the left gripper works near the "
                         "middle", YAW_TRAY["right"])]
    for i, (name, spot) in enumerate(zip(ORDER[side], spots(side))):
        out += _pull(name, spot, side, first=(i == 0 and side == "left"))
    return out


def demo():
    """Both arms on one clock, in parallel, each thinning its own half of the row. DEMO_HOLD_ONLY=1: no plan (both
    arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm
    check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
