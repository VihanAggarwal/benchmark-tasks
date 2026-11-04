"""On the station table: a grey ESD mat (220 x 200 mm, fixed) on the midline 30 cm in front of the arm bases, with
a white foam cup standing on it; a gold ground stud (a 20 mm post, fixed) behind the mat on the midline; a silver
shielding box (140 x 140 mm inside, 25 mm walls, fixed) 41 cm out in front of each arm, each holding a green circuit
board in its carrier (120 x 60 x 20 mm); and a yellow parking square in the back right corner of the table. Both
boards go onto the mat. A gripper may touch a board only while it is grounded: it becomes grounded by closing on the
stud, and touching the foam cup (an insulator) charges it again. Boards rest only on the mat or in their boxes,
and the cup must be at least 30 cm from every board that is out of its box, so it has to be moved out of the way
first.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does.

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
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Both boards still on the mat for 1 s with both grippers let go (each open, or its finger zone > 10 mm from both).
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Move both green circuit boards from their silver shielding boxes onto the grey ESD mat. A gripper may touch a "
    "board only after closing on the gold ground stud, and must close on it again after it touches the white foam "
    "cup. Before taking a board out, move the cup onto the yellow square at the back right: it must stay at least "
    "30 cm from every board that is out of its box. Boards may rest only in their boxes or on the mat; set them "
    "down flat and apart, and do not knock the cup over."
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


MAT = Box("esd_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_GREY)
STUD = Cylinder("ground_stud", pos=(L.STUD_X, L.STUD_Y, L.TOP + L.STUD_HEIGHT / 2), radius=L.STUD_RADIUS,
                height=L.STUD_HEIGHT, color=L.GOLD)
PARK = Box("parking_square", pos=(L.PARK_X, L.PARK_Y, L.TOP + L.PARK_SIZE[2] / 2), size=L.PARK_SIZE, color=L.YELLOW)
BOXES = [w for side in ("left", "right")
         for w in _walls(f"{side}_shield", L.BOX_X, L.BOX_Y[side], L.BOX_INNER, L.BOX_WALL, L.BOX_HEIGHT, L.SILVER)]
BOARDS = [
    Box(L.BOARD_NAMES[side], pos=(L.BOX_X, L.BOX_Y[side], L.TOP + L.BOARD_SIZE[2] / 2), size=L.BOARD_SIZE,
        mass=L.BOARD_MASS, color=L.GREEN)
    for side in ("left", "right")
]
CUP = Cylinder("foam_cup", pos=(L.CUP_X, L.CUP_Y, L.MAT_TOP + L.CUP_HEIGHT / 2), radius=L.CUP_RADIUS,
               height=L.CUP_HEIGHT, mass=L.CUP_MASS, color=L.WHITE)
OBJECTS = [MAT, STUD, PARK, *BOXES, *BOARDS, CUP]

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


def _states(objects):
    return objects[L.BOARD_NAMES["left"]], objects[L.BOARD_NAMES["right"]], objects["foam_cup"]


def success(objects) -> bool:
    """Both boards flat and fully on the mat, at least 20 mm apart, both grippers let go, the cup at least 30 cm
    from every board out of its box at every step, and no breach at any time; held still 1 s. This runs first every
    control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    left, right, cup = _states(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(left, right, cup, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "left_board_pos": left.pos, "right_board_pos": right.pos, "cup_pos": cup.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, a gripper's fingers came within 5 mm of a board while it was not grounded (grounded:
    closed on the stud since it last touched the cup), a board touched the bare bench, tilted over 30 degrees or
    left the table, or the cup tilted over 30 degrees or left the table. Else, with 'kept' meaning the cup was at
    least 30 cm from every board out of its box at every step: 1.0 both boards flat and fully on the mat, at least
    20 mm apart, both grippers let go, kept; 0.75 both on the mat within 15 mm of its edge, kept; 0.5 one board on
    the mat, kept, or both on the mat but not kept; 0.25 no board on the mat, both grippers grounded and the cup at
    least 30 cm from both boards, kept; 0 otherwise."""
    left, right, cup = _states(objects)
    return _TRACKER.result(left, right, cup, rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
BOARD_GRIP = -(L.BOARD_SIZE[1] - 0.008) / 2  # across the board's 60 mm (along y: yaw 0)
CUP_GRIP = -(2 * L.CUP_RADIUS - 0.008) / 2  # across the cup's 50 mm
STUD_PINCH = -(2 * L.STUD_RADIUS - 0.004) / 2  # 4 mm narrower than the 20 mm stud: about 10 N per finger on it
YAW = 0.0  # every grip closes the fingers along y (the boards lie long side along x)
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.005  # boards are gripped with the finger tips 5 mm above their bottom: 15 mm of the 20 mm side held
BOARD_TCP_Z = L.TOP + GRIP_UP + PAD_TIPS  # in the box: the open fingers come down 15 mm inside its side walls
PLACE_TCP_Z = L.MAT_TOP + 0.002 + GRIP_UP + PAD_TIPS  # let go with the board's bottom 2 mm above the mat
ENTRY_Z = PLACE_TCP_Z + 0.020  # the last 20 mm are lowered slowly
HOVER = L.TOP + 0.10
CARRY_Z = L.TOP + 0.09  # a carried board's bottom 72 mm above the table: 47 mm over the boxes' walls
CUP_UP = 0.025  # the cup is gripped with the finger tips 25 mm above its bottom (the pads hold 25-47 mm up)
CUP_TCP_Z = L.MAT_TOP + CUP_UP + PAD_TIPS
PARK_TCP_Z = L.PARK_TOP + 0.002 + CUP_UP + PAD_TIPS  # let go with the cup's bottom 2 mm above the square
CUP_HOVER = L.TOP + 0.13  # the open fingers' tips 44 mm over the cup's top; a carried cup's bottom 92 mm up
STUD_TCP_Z = L.STUD_TOP - 0.015 + PAD_TIPS  # the finger tips 15 mm below the stud's top: 15 mm of it between the pads
STUD_HOVER = L.STUD_TOP + 0.06
MAT_SPOT = {"left": (L.MAT_X, 0.05), "right": (L.MAT_X, -0.05)}  # the boards' outlines 40 mm apart on the mat
LEFT_ASIDE = (0.30, 0.25, HOVER)  # where the left arm waits once its board is down, out of the right arm's way
LEFT_STAGE = (0.25, 0.25, L.TOP + 0.15)  # where the left arm waits at the start, while the right arm takes the cup
LEFT_START_S = 6.0  # ... until then: the two arms are never near the midline (stud, cup) together
STUD_HANDOVER_S = 1.0  # the right arm sets off for the stud this long after the left arm is up off it


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _ground(label: str) -> list[Waypoint]:
    """Close the gripper on the ground stud and let go of it again (8.4 s)."""
    sx, sy = L.STUD_X, L.STUD_Y
    return [
        Waypoint(3.0, (sx, sy, STUD_HOVER), OPEN, f"over the ground stud ({label})", YAW),
        Waypoint(1.2, (sx, sy, STUD_TCP_Z), OPEN, "down round the stud", YAW),
        Waypoint(0.3, (sx, sy, STUD_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.2, (sx, sy, STUD_TCP_Z), STUD_PINCH, "close on the stud: grounded", YAW),
        Waypoint(0.5, (sx, sy, STUD_TCP_Z), STUD_PINCH, "hold", YAW),
        Waypoint(1.2, (sx, sy, STUD_TCP_Z), OPEN, "open", YAW),
        Waypoint(1.0, (sx, sy, STUD_HOVER), OPEN, "up off the stud", YAW),
    ]


def _pick(side: str) -> list[Waypoint]:
    """Down into the box beside the board and grip it across its 60 mm (5.5 s)."""
    bx, by = L.BOX_X, L.BOX_Y[side]
    return [
        Waypoint(2.0, (bx, by, HOVER), OPEN, f"over the {side} shielding box", YAW),
        Waypoint(1.5, (bx, by, BOARD_TCP_Z), OPEN, "down into the box beside the board", YAW),
        Waypoint(0.5, (bx, by, BOARD_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (bx, by, BOARD_TCP_Z), BOARD_GRIP, "grip the board", YAW),
    ]


def _lift(side: str) -> list[Waypoint]:
    bx, by = L.BOX_X, L.BOX_Y[side]
    return [Waypoint(1.2, (bx, by, CARRY_Z), BOARD_GRIP, "lift it out of the box", YAW)]


def _place(side: str) -> list[Waypoint]:
    mx, my = MAT_SPOT[side]
    return [
        Waypoint(2.0, (mx, my, CARRY_Z), BOARD_GRIP, "carry it over the mat", YAW),
        Waypoint(1.0, (mx, my, ENTRY_Z), BOARD_GRIP, "down to 2 cm above", YAW),
        Waypoint(0.5, (mx, my, ENTRY_Z), BOARD_GRIP, "settle", YAW),
        Waypoint(1.5, (mx, my, PLACE_TCP_Z), BOARD_GRIP, "lower the last 2 cm slowly", YAW),
        Waypoint(0.5, (mx, my, PLACE_TCP_Z), BOARD_GRIP, "settle", YAW),
        Waypoint(1.5, (mx, my, PLACE_TCP_Z), OPEN, "let go", YAW),
        Waypoint(1.0, (mx, my, HOVER), OPEN, "up", YAW),
    ]


def _move_cup() -> list[Waypoint]:
    """Right arm: take the foam cup off the mat and stand it on the yellow square in the back right corner."""
    cx, cy = L.CUP_X, L.CUP_Y
    px, py = L.PARK_X, L.PARK_Y
    return [
        Waypoint(2.0, (cx, cy, CUP_HOVER), OPEN, "over the foam cup", YAW),
        Waypoint(1.5, (cx, cy, CUP_TCP_Z), OPEN, "down beside it", YAW),
        Waypoint(0.5, (cx, cy, CUP_TCP_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (cx, cy, CUP_TCP_Z), CUP_GRIP, "grip the cup", YAW),
        Waypoint(1.2, (cx, cy, CUP_HOVER), CUP_GRIP, "lift it off the mat", YAW),
        Waypoint(3.5, (px, py, CUP_HOVER), CUP_GRIP, "carry it over the yellow square", YAW),
        Waypoint(1.5, (px, py, PARK_TCP_Z), CUP_GRIP, "lower it onto the square", YAW),
        Waypoint(0.5, (px, py, PARK_TCP_Z), CUP_GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, PARK_TCP_Z), OPEN, "let go of the cup", YAW),
        Waypoint(1.0, (px, py, CUP_HOVER), OPEN, "up", YAW),
    ]


# The two arms' clock. Only one arm is near the midline (the cup on the mat, the stud) at a time.
CUP_PARKED_S = _span(_move_cup()[:9])  # the right gripper has opened over the square: the cup stands there
LEFT_OFF_STUD_S = LEFT_START_S + _span(_ground("left"))  # the left gripper is grounded and up off the stud
RIGHT_TO_STUD_S = max(_span(_move_cup()), LEFT_OFF_STUD_S + STUD_HANDOVER_S)  # the right arm sets off for the stud
RIGHT_GROUNDED_S = RIGHT_TO_STUD_S + _span(_ground("right"))  # ... and is up off it again, grounded


def _left() -> list[Waypoint]:
    """Left arm: wait while the right arm takes the cup off the mat, ground, take its board out of the box once the
    cup is parked, hold it over the box until the right arm is off the stud, set it on the mat's left half, let
    go, move aside."""
    stage = [Waypoint(2.0, LEFT_STAGE, OPEN, "up beside the box, out of the right arm's way", YAW),
             Waypoint(LEFT_START_S - 2.0, LEFT_STAGE, OPEN, "wait while the right arm takes the cup", YAW)]
    ground, pick = _ground("left"), _pick("left")
    wait_cup = max(0.0, CUP_PARKED_S - _span(stage + ground + pick))
    lift = _lift("left")
    hold = max(0.0, RIGHT_GROUNDED_S - _span(stage + ground + pick + lift) - wait_cup)
    bx, by = L.BOX_X, L.BOX_Y["left"]
    plan = stage + ground + pick
    if wait_cup > 0:
        plan.append(Waypoint(wait_cup, (bx, by, BOARD_TCP_Z), BOARD_GRIP, "wait until the cup is parked", YAW))
    plan += lift
    if hold > 0:
        plan.append(Waypoint(hold, (bx, by, CARRY_Z), BOARD_GRIP, "hold it over the box until the right arm is "
                                                                   "off the stud", YAW))
    plan += _place("left")
    plan.append(Waypoint(1.5, LEFT_ASIDE, OPEN, "aside, out of the right arm's way", YAW))
    return plan


LEFT_ASIDE_S = _span(_left())  # the left arm is aside: the right arm may come over the mat


def _right() -> list[Waypoint]:
    """Right arm: park the cup, wait until the left arm is off the stud, ground (after touching the cup), take its
    board out of the box, hold it until the left arm is aside, set it on the mat's right half, let go, go up."""
    plan = _move_cup()
    wait = RIGHT_TO_STUD_S - _span(plan)
    if wait > 0:
        px, py = L.PARK_X, L.PARK_Y
        plan.append(Waypoint(wait, (px, py, CUP_HOVER), OPEN, "wait until the left arm is off the stud", YAW))
    plan += _ground("right") + _pick("right") + _lift("right")
    wait = LEFT_ASIDE_S - _span(plan)
    if wait > 0:
        bx, by = L.BOX_X, L.BOX_Y["right"]
        plan.append(Waypoint(wait, (bx, by, CARRY_Z), BOARD_GRIP, "hold it over the box until the left arm is "
                                                                   "aside", YAW))
    return plan + _place("right")


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
