"""On the station table: a white backing board (180 x 130 x 5 mm, fixed) on the midline 30 cm in front of the arm
bases, its long side along x; a black riser (a glued mat-board stack, 70 x 30 x 10 mm, 15 g) lying flat on the
table 36 cm to the right, its 70 mm length along y; and a cream print dry-mounted on 10 mm board (110 x 70 x 10 mm,
22 g) lying face up on the table 36 cm to the left, its long side along x. Float mount: the riser goes down on the
board first and the print is laid face up on it, centred on the board with its long side along the board's; the
riser must end at least 10 mm in from every edge of the print, so it cannot be seen. Laid as it lies (across the
print), the 70 mm riser is as long as the print is wide and shows: it must be turned lengthwise.

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
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Float-mount the print on the white backing board. First lay the black riser flat on the board, then lay the "
    "print face up on top of the riser, centred on the board with its long side along the board's long side. The "
    "riser must end at least 1 cm in from every edge of the print, so that it cannot be seen. Then let go and move "
    "both grippers away."
)

BOARD = Box("backing_board", pos=(L.BOARD_X, L.BOARD_Y, L.TOP + L.BOARD_SIZE[2] / 2), size=L.BOARD_SIZE,
            color=L.BOARD_WHITE)
RISER = Box(
    "riser",
    pos=(L.RISER_X, L.RISER_Y, L.TOP + L.RISER_SIZE[2] / 2),
    size=L.RISER_SIZE,
    mass=L.RISER_MASS,
    color=L.RISER_BLACK,
    yaw=L.RISER_YAW,
)
PRINT = Box(
    "print",
    pos=(L.PRINT_X, L.PRINT_Y, L.TOP + L.PRINT_SIZE[2] / 2),
    size=L.PRINT_SIZE,
    mass=L.PRINT_MASS,
    color=L.PRINT_CREAM,
)
OBJECTS = [BOARD, RISER, PRINT]

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


def success(objects) -> bool:
    """Riser set flat on the board, print floating face up on it, inset >= 10 mm, print within 6 mm of the board's
    centre and square within 3 degrees, both grippers' finger pads 30 mm or more from both parts; held 1 s. This
    runs first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    riser, print_ = objects["riser"], objects["print"]
    arms = rigstate.arms()
    result = _TRACKER.update(riser, print_, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "riser_pos": riser.pos, "riser_quat": riser.quat_xyzw, "print_pos": print_.pos,
           "print_quat": print_.quat_xyzw, "riser_vel": riser.lin_vel, "print_vel": print_.lin_vel}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a part is off the table or the riser is not set (flat on the board, wholly on it, at rest). Else 0.25
    if the print is not floating (face up, at rest, underside 13-17 mm up, over the riser's centre); 1.0 inset
    >= 10 mm, off centre <= 6 mm, turned <= 3 deg, grippers clear; 0.75 inset >= 10 mm, off centre <= 12 mm,
    turned <= 8 deg; 0.5 otherwise."""
    return _TRACKER.result(objects["riser"], objects["print"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
RISER_GRIP = -(L.RISER_SIZE[0] - 0.008) / 2  # across the riser's 30 mm: -0.011
PRINT_GRIP = -(L.PRINT_SIZE[1] - 0.008) / 2  # across the print's 70 mm: -0.031
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.002  # both parts are gripped with the pad tips 2 mm above their bottom: 8 mm of the 10 mm side held
PLACE_GAP = 0.001  # each part is let go with its underside 1 mm above its support
PICK_Z = L.TOP + GRIP_UP + PAD_TIPS  # TCP 15 mm up: pad tips 2 mm above the table
RISER_PLACE_Z = L.BOARD_TOP + PLACE_GAP + GRIP_UP + PAD_TIPS  # TCP 21 mm up: riser 1 mm above the board
PRINT_PLACE_Z = L.RISER_TOP_ON_BOARD + PLACE_GAP + GRIP_UP + PAD_TIPS  # TCP 31 mm up: print 1 mm above the riser
ENTRY = 0.020  # the last 20 mm of each place are lowered slowly
CARRY_Z = L.TOP + 0.10
PARK_Z = L.TOP + 0.12
YAW_PICK_R = math.pi / 2  # right arm at the riser: fingers close along x, across its 30 mm width
YAW_PLACE_R = 0.0  # ... turned a quarter turn on the way, so the riser's 70 mm runs along x; fingers along y
YAW_L = 0.0  # left arm: fingers close along y, across the print's 70 mm, the whole time
PARK_R = (0.240, -0.300)
PARK_L = (0.240, 0.300)
WAIT_L = (0.220, 0.240)  # the left arm holds the print here while the right arm works at the board


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _right() -> list[Waypoint]:
    """Right arm: pick the riser across its width, carry it to the board turning it a quarter turn so it lies
    lengthwise along x, lower it slowly onto the board's centre, let go, go up and park out of the way. The first
    move (2.8 s, the gripper turned to yaw +pi/2) and the carry (3.5 s, turned back to yaw 0) are long enough to keep
    joint 6, which also makes up for joint 1's swing, under 1.2 rad/s."""
    rx, ry = L.RISER_X, L.RISER_Y
    bx, by = L.BOARD_X, L.BOARD_Y
    a, b = YAW_PICK_R, YAW_PLACE_R
    return [
        Waypoint(2.8, (rx, ry, CARRY_Z), OPEN, "over the riser", a),
        Waypoint(2.0, (rx, ry, PICK_Z), OPEN, "down beside it", a),
        Waypoint(0.5, (rx, ry, PICK_Z), OPEN, "settle", a),
        Waypoint(1.5, (rx, ry, PICK_Z), RISER_GRIP, "grip the riser across its width", a),
        Waypoint(1.5, (rx, ry, CARRY_Z), RISER_GRIP, "lift", a),
        Waypoint(3.5, (bx, by, CARRY_Z), RISER_GRIP, "carry it over the board, turning it lengthwise", b),
        Waypoint(1.2, (bx, by, RISER_PLACE_Z + ENTRY), RISER_GRIP, "down to 2 cm above the board", b),
        Waypoint(0.5, (bx, by, RISER_PLACE_Z + ENTRY), RISER_GRIP, "settle", b),
        Waypoint(2.0, (bx, by, RISER_PLACE_Z), RISER_GRIP, "lower the last 2 cm slowly", b),
        Waypoint(0.5, (bx, by, RISER_PLACE_Z), RISER_GRIP, "settle", b),
        Waypoint(1.5, (bx, by, RISER_PLACE_Z), OPEN, "let go", b),
        Waypoint(1.0, (bx, by, CARRY_Z), OPEN, "straight up", b),
        Waypoint(1.5, (*PARK_R, PARK_Z), OPEN, "park out of the way", b),
    ]


RIGHT_PARKED_S = _span(_right())  # 20.0 s: the right arm is parked 30 cm right of the board's centre


def _left_pick() -> list[Waypoint]:
    px, py = L.PRINT_X, L.PRINT_Y
    return [
        Waypoint(2.0, (px, py, CARRY_Z), OPEN, "over the print", YAW_L),
        Waypoint(2.0, (px, py, PICK_Z), OPEN, "down beside it", YAW_L),
        Waypoint(0.5, (px, py, PICK_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (px, py, PICK_Z), PRINT_GRIP, "grip the print across its width", YAW_L),
        Waypoint(1.5, (px, py, CARRY_Z), PRINT_GRIP, "lift", YAW_L),
        Waypoint(2.0, (*WAIT_L, CARRY_Z), PRINT_GRIP, "to the wait point", YAW_L),
    ]


def _left() -> list[Waypoint]:
    """Left arm: pick the print, hold it at the wait point until the right arm has parked, then carry it over the
    board, lower it slowly onto the riser, let go, go up and park out of the way."""
    bx, by = L.BOARD_X, L.BOARD_Y
    pick = _left_pick()
    wait = RIGHT_PARKED_S - _span(pick)
    return pick + [
        Waypoint(wait, (*WAIT_L, CARRY_Z), PRINT_GRIP, "wait until the right arm has parked", YAW_L),
        Waypoint(2.0, (bx, by, CARRY_Z), PRINT_GRIP, "carry it over the board", YAW_L),
        Waypoint(1.2, (bx, by, PRINT_PLACE_Z + ENTRY), PRINT_GRIP, "down to 2 cm above the riser", YAW_L),
        Waypoint(0.5, (bx, by, PRINT_PLACE_Z + ENTRY), PRINT_GRIP, "settle", YAW_L),
        Waypoint(2.0, (bx, by, PRINT_PLACE_Z), PRINT_GRIP, "lower the last 2 cm slowly", YAW_L),
        Waypoint(0.5, (bx, by, PRINT_PLACE_Z), PRINT_GRIP, "settle", YAW_L),
        Waypoint(1.5, (bx, by, PRINT_PLACE_Z), OPEN, "let go", YAW_L),
        Waypoint(1.0, (bx, by, CARRY_Z), OPEN, "straight up", YAW_L),
        Waypoint(2.0, (*PARK_L, PARK_Z), OPEN, "park out of the way", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
