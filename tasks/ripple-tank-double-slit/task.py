"""On the station table: a small ripple tank centred on the midline (a pale acrylic floor 260 x 200 mm inside grey
PVC walls 15 mm tall, its near end 15 cm in front of the arm bases), with a fixed white dipper bar lying across it
18 cm out and a grey foam beach along its far wall; and three loose black aluminium barrier bars standing on the
table outside it, two 70 mm long (one in front of each arm) and one 30 mm long (in front of the left arm). Set-up
for the two-slit demonstration: the three bars in one straight line across the tank, 120 mm beyond the dipper and
parallel to it, closed against both side walls, leaving two slits about 15 mm wide with centres about 45 mm apart,
centred on the tank. The score is what a technician checks with a rule before switching the dipper on: the line,
both slits' widths, their spacing and centring, and the gaps at the walls.

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
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. The last bar counts as standing (lowest point within 2 mm of the floor) while the
# gripper still holds it, from the end of its slow descent; letting go takes 1.5 s more and rising 1 s. With 2.5 s
# the trial cannot end while the fingers are still closed on it.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Set up the ripple tank for the two-slit demonstration: stand the three black barrier bars in one straight line "
    "across the tank, parallel to the white dipper and 120 mm beyond it (centre line to centre line), closed against "
    "both side walls, so that they leave two slits about 15 mm wide, 45 mm apart centre to centre, centred on the "
    "tank."
)

_WALL_Z = L.TOP + L.WALL_H / 2
_ON_FLOOR = L.TOP + L.FLOOR_TOP
FLOOR = Box("tank_floor", pos=(L.TANK_X, 0.0, L.TOP + L.FLOOR_T / 2),
            size=(L.TANK_X1 - L.TANK_X0, 2 * L.TANK_HALF, L.FLOOR_T), color=L.ACRYLIC_PALE)
WALLS = [
    Box("tank_wall_left", pos=(L.TANK_X, L.SIDE_WALL_Y, _WALL_Z), size=(L.SIDE_WALL_LEN, L.WALL_T, L.WALL_H),
        color=L.PVC_GREY),
    Box("tank_wall_right", pos=(L.TANK_X, -L.SIDE_WALL_Y, _WALL_Z), size=(L.SIDE_WALL_LEN, L.WALL_T, L.WALL_H),
        color=L.PVC_GREY),
    Box("tank_wall_near", pos=(L.END_WALL_X[0], 0.0, _WALL_Z), size=(L.WALL_T, L.END_WALL_LEN, L.WALL_H),
        color=L.PVC_GREY),
    Box("tank_wall_far", pos=(L.END_WALL_X[1], 0.0, _WALL_Z), size=(L.WALL_T, L.END_WALL_LEN, L.WALL_H),
        color=L.PVC_GREY),
]
DIPPER = Box("dipper", pos=(L.DIPPER_X, 0.0, _ON_FLOOR + L.DIPPER_SIZE[2] / 2), size=L.DIPPER_SIZE,
             color=L.DIPPER_WHITE)
BEACH = Box("beach", pos=(L.BEACH_X, 0.0, _ON_FLOOR + L.BEACH_SIZE[2] / 2), size=L.BEACH_SIZE, color=L.FOAM_GREY)
BARS = [
    Box(name, pos=(x, y, L.TOP + L.BAR_SIZE[name][2] / 2), size=L.BAR_SIZE[name], mass=L.BAR_MASS[name],
        color=L.BAR_BLACK)
    for name, (x, y) in L.BAR_START.items()
]
OBJECTS = [FLOOR, *WALLS, DIPPER, BEACH, *BARS]

_TRACKER = Tracker()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace
    that cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def _bars(objects) -> dict:
    return {n: objects.get(n) for n in L.BAR_NAMES}


def success(objects) -> bool:
    """The three bars standing on the floor in line 110-130 mm beyond the dipper and parallel to it, both slits
    10-20 mm, their centres 40-50 mm apart with the midpoint within 5 mm of the centre line, both wall gaps 5 mm or
    less; held still 2.5 s. This runs first every control step, so it also feeds the tracker."""
    bars = _bars(objects)
    result = _TRACKER.update(bars)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    for n, s in bars.items():
        if s is not None:
            row[f"{n}_pos"], row[f"{n}_quat"] = getattr(s, "pos", None), getattr(s, "quat_xyzw", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the three bars stand on the tank floor in one line 110-130 mm beyond the dipper, parallel to it.
    Then 0.25, plus 0.25 each for: both slits 10-20 mm wide; slit centres 40-50 mm apart and their midpoint within
    5 mm of the tank's centre line; both wall gaps 5 mm or less."""
    return _TRACKER.result(_bars(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part. Every grip
# is across a bar's 20 mm thickness, with the fingers closing along x (yaw -90 deg left, +90 deg right), so the
# pads sit in front of and behind the bar, never in a slit or at a wall.
T = L.LONG_SIZE[0]  # 20 mm
PICK_OPEN = -(T + 0.016) / 2  # 36 mm: 8 mm each side
GRIP = -(T - 0.008) / 2  # 12 mm commanded
RELEASE = PICK_OPEN
YAW_L, YAW_R = -math.pi / 2, math.pi / 2
GRIP_UP = 0.020  # the TCP this far above the bar's bottom: pad tips 7 mm above it
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + L.FLOOR_TOP + 0.001 + GRIP_UP  # let go with the bar's bottom 1 mm above the tank floor
SLIDE_Z = L.TOP + L.FLOOR_TOP + 0.004 + GRIP_UP  # a long bar slides to its wall 4 mm above the floor (8 mm below
#                                                  the wall's top), so the wall's face, not its top, stops it
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm down are slow
CARRY_Z = L.TOP + 0.100  # a carried bar's bottom 80 mm up: over the walls (15 mm) and the bars in place (28 mm)
HOLD_Z = L.TOP + 0.120  # where an arm holds its next bar, over its own side, while the other works in the tank
L_PARK = (0.220, 0.300, L.TOP + 0.150)  # where each arm ends, over its own side, clear of the tank
R_PARK = (0.220, -0.300, L.TOP + 0.150)
# Placement trim (x, y) per bar: where the command goes relative to layout.PLACE_Y. 0 offline. On the GPU read each
# bar's y_mm and wall_gaps_mm in TASK_TRACE after it is let go. A long bar is driven 1.5 mm past its wall's face, so
# it should end against the wall (gap 0): if a wall gap is over 2 mm the arm fell short, so move that bar's trim
# toward its wall by the gap (the wall stops a generous trim). If the short bar lands more than 3 mm off the centre
# line (5 mm still scores 1.0 with both long bars against their walls), trim it back by its offset.
TRIM = {"bar_a": (0.0, 0.0), "bar_b": (0.0, 0.0), "bar_c": (0.0, 0.0)}


def _at(name: str, where: dict) -> tuple[float, float]:
    dx, dy = TRIM[name]
    return L.LINE_X + dx, where[name] + dy


def _approach(name: str, yaw: float) -> list[Waypoint]:
    x, y = L.BAR_START[name]
    return [Waypoint(2.0, (x, y, CARRY_Z), PICK_OPEN, f"over {name}", yaw)]


def _grip(name: str, yaw: float, lift_z: float) -> list[Waypoint]:
    """From over the bar: down round it, grip it across its 20 mm thickness, lift it."""
    x, y = L.BAR_START[name]
    return [
        Waypoint(1.5, (x, y, PICK_Z), PICK_OPEN, f"down round {name}", yaw),
        Waypoint(0.5, (x, y, PICK_Z), PICK_OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", yaw),
        Waypoint(1.2, (x, y, lift_z), GRIP, f"lift {name}", yaw),
    ]


def _lower(name: str, yaw: float, x: float, y: float) -> list[Waypoint]:
    """Carry the gripped bar over (x, y) on the line and bring it down to 2 cm above the tank floor."""
    return [
        Waypoint(2.0, (x, y, CARRY_Z), GRIP, f"carry {name} over its place on the line", yaw),
        Waypoint(1.2, (x, y, ENTRY_Z), GRIP, "down to 2 cm above the floor", yaw),
        Waypoint(0.5, (x, y, ENTRY_Z), GRIP, "settle", yaw),
    ]


def _let_go(name: str, yaw: float, x: float, y: float) -> list[Waypoint]:
    return [
        Waypoint(1.5, (x, y, PLACE_Z), RELEASE, f"let go of {name}", yaw),
        Waypoint(1.0, (x, y, CARRY_Z), RELEASE, "straight up, out of the tank", yaw),
    ]


def _place_long(name: str, yaw: float) -> list[Waypoint]:
    """A long bar: set it down 6 mm in from its wall, slide it out with its bottom below the wall's top toward a point
    1.5 mm past the wall's face (the wall stops it), lower it onto the floor, let go, rise."""
    x, y_set = _at(name, L.SET_Y)
    _, y_end = _at(name, L.PLACE_Y)
    return _lower(name, yaw, x, y_set) + [
        Waypoint(2.0, (x, y_set, SLIDE_Z), GRIP, f"slowly down: {name} to 4 mm above the floor", yaw),
        Waypoint(1.5, (x, y_end, SLIDE_Z), GRIP, f"slide {name} out against its wall", yaw),
        Waypoint(0.3, (x, y_end, SLIDE_Z), GRIP, "settle", yaw),
        Waypoint(1.0, (x, y_end, PLACE_Z), GRIP, f"slowly down: {name} onto the floor", yaw),
    ] + _let_go(name, yaw, x, y_end)


def _place_short(name: str, yaw: float) -> list[Waypoint]:
    """The short bar: straight down onto the floor in the middle of the line, let go, rise."""
    x, y = _at(name, L.PLACE_Y)
    return _lower(name, yaw, x, y) + [
        Waypoint(2.0, (x, y, PLACE_Z), GRIP, f"slowly down: {name} onto the floor", yaw),
    ] + _let_go(name, yaw, x, y)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _wait(wps: list[Waypoint], until: float, label: str) -> list[Waypoint]:
    """Hold the last pose of ``wps`` until ``until`` seconds (at least 0.5 s)."""
    last = wps[-1]
    return [Waypoint(max(until - _span(wps), 0.5), last.pos, last.finger, label, last.yaw)]


def _schedule() -> tuple[list[Waypoint], list[Waypoint]]:
    """Both arms on one clock. The left lays bar a against the left wall; the right waits over bar b, picks it so
    that its lift ends as the left rises out of the tank, and lays it against the right wall while the left picks
    bar c; the left waits with bar c until the right has risen out of the tank, lays it in the middle, parks."""
    left = _approach("bar_a", YAW_L) + _grip("bar_a", YAW_L, CARRY_Z) + _place_long("bar_a", YAW_L)
    a_done = _span(left)  # the left gripper is up out of the tank
    right = _approach("bar_b", YAW_R)
    grip_b = _grip("bar_b", YAW_R, HOLD_Z)
    right += _wait(right, a_done - _span(grip_b), "wait over bar b") + grip_b + _place_long("bar_b", YAW_R)
    b_done = _span(right)  # the right gripper is up out of the tank
    right += [Waypoint(2.0, R_PARK, RELEASE, "park over its own side", YAW_R)]
    left += _approach("bar_c", YAW_L)
    grip_c = _grip("bar_c", YAW_L, HOLD_Z)
    left += _wait(left, b_done - _span(grip_c), "wait over bar c") + grip_c
    left += _wait(left, b_done, "hold bar c until the right arm is out of the tank") + _place_short("bar_c", YAW_L)
    left += [Waypoint(2.0, L_PARK, RELEASE, "park over its own side", YAW_L)]
    return left, right


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    left, right = _schedule()
    plan = {"left": left, "right": right}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
