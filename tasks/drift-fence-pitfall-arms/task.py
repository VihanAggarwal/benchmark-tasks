"""On the station table: a black pitfall-trap disc (70 mm across, 3 mm tall, fixed) on the midline 27 cm in front of
the arm bases, standing in for the mouth of a bucket sunk flush with the ground; four fixed orange pegs (12 mm
across, 60 mm tall) 200 mm from its centre at bearings 45, 135, 225 and 315 deg, each marking the far end of one
drift-fence arm; and four loose pine boards (130 x 20 x 50 mm, 65 g) standing on edge, long side along x, two
beside each arm's base (y +-0.37 and +-0.45). Each board must end standing on its long narrow edge along the line
from the trap's centre to one peg, its inner end 0-10 mm from the trap's rim and not over the opening, with no
gripper holding it.

The scorer reads the arms' TCPs through rigstate.py (copied unchanged from tools/kit/rigstate.py), which wraps the
embodiment's step and reset from this file's import; it changes nothing the embodiment does.

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
# All four lines fenced, nothing over the rim, every board standing, at rest and free, held for 1 s.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Set up the drift fence around the black pitfall trap: for each of the four orange pegs, stand one wooden "
    "board on its long narrow edge along the line from the trap's centre to that peg, with the board's inner end "
    "right up against the trap's rim but not over its opening. Then let go of every board and lift the grippers "
    "clear."
)

TRAP = Cylinder("trap", pos=(L.TRAP_X, L.TRAP_Y, L.TOP + L.TRAP_HEIGHT / 2), radius=L.TRAP_RADIUS,
                height=L.TRAP_HEIGHT, color=L.TRAP_BLACK)
PEGS = [
    Cylinder(name, pos=(x, y, L.TOP + L.PEG_HEIGHT / 2), radius=L.PEG_RADIUS, height=L.PEG_HEIGHT,
             color=L.PEG_ORANGE)
    for name, x, y in L.PEGS
]
BOARDS = [
    Box(name, pos=(*L.BOARD_STARTS[name], L.TOP + L.BOARD_SIZE[2] / 2), size=L.BOARD_SIZE, mass=L.BOARD_MASS,
        color=L.PINE)
    for name in L.BOARDS
]
OBJECTS = [TRAP, *PEGS, *BOARDS]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the boards teleported back)."""
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


def _boards(objects) -> dict:
    return {n: objects[n] for n in L.BOARDS if n in objects}


def success(objects) -> bool:
    """All four lines fenced (a standing, still, free board in each line's sector, along its line within 8 deg,
    the trap's centre within 10 mm of its centreline, its gap -1 to +10 mm) and no board's outline more than 1 mm
    inside the rim; held 1 s. This runs first every control step, so it also feeds the tracker (with the arms
    rigstate read for this step)."""
    boards = _boards(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(boards, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "board_pos": {n: s.pos for n, s in boards.items()},
           "board_quat": {n: s.quat_xyzw for n, s in boards.items()}}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any board's outline is more than 1 mm inside the trap's rim. Otherwise 0.25 for each line fenced by a
    standing, still, free board in its sector: 0, 0.25, 0.5, 0.75 or 1.0."""
    return _TRACKER.result(_boards(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.BOARD_SIZE[1] - 0.008) / 2  # -0.006: 12 mm commanded across the 20 mm thickness (yaw 0: along y)
CARRY_Z = L.TOP + 0.120  # board bottom 85 mm up while carried: over the 60 mm pegs and the 50 mm boards
PICK_Z = L.TOP + 0.035  # TCP 35 mm up: pad tips 22 mm up the 50 mm board
PLACE_Z = L.TOP + 0.036  # let go with the board's bottom 1 mm above the table
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm are lowered slowly
PARK_Z = L.TOP + 0.160
PARK = {"left": (0.220, +0.300), "right": (0.220, -0.300)}
# Each board's centre goes 105 mm from the trap's centre along its line: its inner end face 40 mm out (gap 5 mm,
# inner corners 41.2 mm out, 6.2 mm clear of the rim) and its outer end 170 mm out (24 mm short of the peg's
# surface). On the GPU read each board's gap_mm and offset_mm in TASK_TRACE: aim for a gap of 5-6 mm; if a
# released board settles outside 0-10 mm or more than 10 mm off its line, change PLACE_RADIUS (and the yaw if the
# board turns) from the measured trace.
PLACE_RADIUS = 0.105
# The grip yaw that lays a board (picked at yaw 0, long side along x) along each line: +45 deg for the 45 and 225
# deg lines, -45 deg for the 135 and 315 deg lines (a board's two ends are alike).
LINE_YAW = {1: math.radians(45.0), 2: math.radians(-45.0), 3: math.radians(45.0), 4: math.radians(-45.0)}
# Each board is gripped 20 mm from its centre towards the end that will face the peg, so the TCP stands 125 mm out
# along the line. With centred grips (TCP 105 mm out) the two arms' wrists came within 3.7 cm of each other (link4
# capsules, the left arm on line 2 and the right on line 4) whether the phases ran together or staggered; 20 mm
# out spreads them to 7 cm or more. The board's centre hangs 20 mm off the grip axis (65 g: 0.013 N m).
GRIP_OFFSET = 0.020


def line_target(k: int) -> tuple[float, float]:
    """Where a board's centre goes for line k (1..4)."""
    b = math.radians(L.PEG_BEARINGS_DEG[k - 1])
    return L.TRAP_X + PLACE_RADIUS * math.cos(b), L.TRAP_Y + PLACE_RADIUS * math.sin(b)


def grip_shift(k: int) -> float:
    """Where along the board (its own x, as it stands at its start) the TCP grips it for line k: +20 mm or -20 mm,
    the end that the turn to LINE_YAW[k] points away from the trap."""
    out = math.cos(LINE_YAW[k] - math.radians(L.PEG_BEARINGS_DEG[k - 1]))
    return GRIP_OFFSET if out > 0 else -GRIP_OFFSET


def tcp_target(k: int) -> tuple[float, float]:
    """Where the TCP goes to set a board on line k: GRIP_OFFSET further out along the line than its centre."""
    b = math.radians(L.PEG_BEARINGS_DEG[k - 1])
    r = PLACE_RADIUS + GRIP_OFFSET
    return L.TRAP_X + r * math.cos(b), L.TRAP_Y + r * math.sin(b)


def _set_board(board: str, k: int) -> list[Waypoint]:
    """One board, 16 s: pick it at yaw 0 across its thickness, 20 mm from its centre, lift it over the pegs, carry
    it over line k while turning to the line's yaw, lower it slowly, settle, open, go straight up."""
    bx, by = L.BOARD_STARTS[board]
    bx += grip_shift(k)
    tx, ty = tcp_target(k)
    yaw = LINE_YAW[k]
    return [
        Waypoint(2.0, (bx, by, CARRY_Z), OPEN, f"over {board}", 0.0),
        Waypoint(1.5, (bx, by, PICK_Z), OPEN, f"down round {board}", 0.0),
        Waypoint(0.5, (bx, by, PICK_Z), OPEN, "settle", 0.0),
        Waypoint(1.5, (bx, by, PICK_Z), GRIP, f"grip {board} across its thickness", 0.0),
        Waypoint(1.5, (bx, by, CARRY_Z), GRIP, "lift it clear of the pegs and boards", 0.0),
        Waypoint(2.5, (tx, ty, CARRY_Z), GRIP, f"carry it over line {k}, turning it along the line", yaw),
        Waypoint(0.5, (tx, ty, CARRY_Z), GRIP, "settle", yaw),
        Waypoint(1.2, (tx, ty, ENTRY_Z), GRIP, "down to 2 cm above the table", yaw),
        Waypoint(1.3, (tx, ty, PLACE_Z), GRIP, "lower the last 2 cm slowly", yaw),
        Waypoint(0.5, (tx, ty, PLACE_Z), GRIP, "settle on the table", yaw),
        Waypoint(1.5, (tx, ty, PLACE_Z), OPEN, "let go", yaw),
        Waypoint(1.5, (tx, ty, CARRY_Z), OPEN, "straight up, clear of the board", yaw),
    ]


def _park(side: str) -> list[Waypoint]:
    return [Waypoint(2.0, (*PARK[side], PARK_Z), OPEN, "park, clear of the fence", 0.0)]


def _left() -> list[Waypoint]:
    """Left arm: the inner board (board_1) to line 1, then the outer board (board_2) to line 2, then park."""
    return _set_board("board_1", 1) + _set_board("board_2", 2) + _park("left")


def _right() -> list[Waypoint]:
    """Right arm, at the same time: the inner board (board_3) to line 3, then board_4 to line 4, then park."""
    return _set_board("board_3", 3) + _set_board("board_4", 4) + _park("right")


def demo():
    """Both arms on one clock, in two simultaneous phases (staggering them brings the arms closer). DEMO_HOLD_ONLY=1:
    no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is
    dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
