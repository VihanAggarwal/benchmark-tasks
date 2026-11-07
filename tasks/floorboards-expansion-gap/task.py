"""On the station table: a bay walled on three sides by grey plywood walls 40 mm tall (side walls along x with their
inside faces 330 mm apart, a far wall along y 36 cm in front of the arm bases), open toward the arms; its floor is
the table. Three loose pine floorboards stained dark brown (300 x 50 x 16 mm, 0.11 kg) lie flat on the table outside
it, lengthwise along x: two in front of the left arm, one in front of the right arm. Rule (floor laying): the boards
go side by side across the bay, tight against each other, with a 10-20 mm expansion gap left open at both side walls
and at the far wall. A board pinned against a wall (touching it or within 3 mm) scores 0: a floor laid tight to a
wall buckles when the wood swells.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does. offline_push.py is used only by our
offline checker (no simulator), which cannot push; the simulator never calls it.

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
import offline_push  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s: every board laid, the joints tight, the gaps open, both grippers' finger zones 30 mm clear.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Lay the three brown floorboards flat in the walled bay, side by side across it, starting along the far wall. "
    "Leave a gap of 10 to 20 mm between the boards and each of the three walls, and push the boards tight against "
    "each other."
)

_WALL_Z = L.TOP + L.WALL_H / 2
WALLS = [
    Box("wall_left", pos=(L.SIDE_WALL_X, L.SIDE_WALL_Y, _WALL_Z), size=(L.SIDE_WALL_LEN, L.WALL_T, L.WALL_H),
        color=L.PLYWOOD_GREY),
    Box("wall_right", pos=(L.SIDE_WALL_X, -L.SIDE_WALL_Y, _WALL_Z), size=(L.SIDE_WALL_LEN, L.WALL_T, L.WALL_H),
        color=L.PLYWOOD_GREY),
    Box("wall_far", pos=(L.FAR_WALL_X, 0.0, _WALL_Z), size=(L.WALL_T, L.FAR_WALL_LEN, L.WALL_H),
        color=L.PLYWOOD_GREY),
]
BOARDS = [
    Box(name, pos=(x, y, L.TOP + L.BOARD_HALF[2]), size=L.BOARD_SIZE, mass=L.BOARD_MASS, color=L.WALNUT_STAIN)
    for name, (x, y) in L.BOARD_START.items()
]
OBJECTS = [*WALLS, *BOARDS]  # the boards in L.BOARD_NAMES order (offline_push steps on the first one)

# Offline checker only (tools/offline/check_task.py): it carries the boards but cannot push them, so their offline
# states come from a small push model. The simulator never calls these hooks.
_PUSH = offline_push.BoardPush()
_PUSH.attach(BOARDS)

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    _PUSH.reset()


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
    return {n: objects.get(n) for n in L.BOARD_NAMES}


def success(objects) -> bool:
    """All three boards laid (flat across the bay, 10-20 mm from both side walls), the gap to the far wall 10-20 mm,
    both joints 3 mm or less, no board within 3 mm of a wall, both grippers' finger zones 30 mm or more from every
    board; held still 1 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate
    read for this step)."""
    boards = _boards(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(boards, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in boards.items():
        if s is not None:
            row[f"{n}_pos"], row[f"{n}_quat"] = getattr(s, "pos", None), getattr(s, "quat_xyzw", None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_opening=arms["left"].opening,
                   right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any board at the walls' height comes within 3 mm of a wall's inside face (or lies on a wall). Else 1.0
    three boards laid, far gap 10-20 mm, both joints <= 3 mm, both grippers 30 mm clear; otherwise 0.25 per laid
    board (a board is laid when flat on the floor, across the bay within 3 degrees, inside it, with both end gaps
    10-20 mm)."""
    return _TRACKER.result(_boards(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
W = L.BOARD_SIZE[1]  # 50 mm: every grip is across a board's width
OPEN = -0.04695
PICK_OPEN = -(W + 0.010) / 2  # 60 mm: 5 mm each side; the pads keep 12 mm from the next board and the left wall
GRIP = -(W - 0.008) / 2  # 42 mm commanded
RELEASE = -(W + 0.006) / 2  # 56 mm: 3 mm each side, so the far pad (8 mm) ends 11 mm past the board's face
CLOSED = 0.0  # pushing: the fingers closed together, a block 16 mm deep along x, 20 mm wide along y
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # laying across the bay: the fingers close along x
GRIP_Z = L.TOP + 0.016  # TCP 16 mm up: pad tips 3 mm above the table, on the board's faces 3-16 mm up
PLACE_Z = GRIP_Z + 0.001  # let go with the board's underside 1 mm above the floor
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm down are slow
CARRY_Z = L.TOP + 0.085  # a carried board's underside 69 mm up: 29 mm over the walls
HOVER_Z = L.TOP + 0.100
HOLD_Z = L.TOP + 0.120  # the right arm holds its board here, on its own side, until the left arm has parked
CLEAR_Z = L.TOP + 0.060  # after letting go: pad tips 47 mm up, 31 mm over the boards, to close the fingers
PUSH_Z = L.TOP + 0.018  # pushing: pad tips 5 mm up, on the board's face 5-16 mm up
FINGER_FRONT = L.PAD_THICK  # the closed fingers' face is this far ahead of the TCP along the push
APPROACH = 0.008  # the closed fingers come down this far behind the board's near face

# Where the boards go: a along the far wall, set 17 mm from it; b and c each set down a little short of the board
# before it, far enough for the far finger to come down and open between them, then pushed tight.
FAR_SET_GAP = 0.017
X_A = L.FAR_X - FAR_SET_GAP - W / 2  # 0.318
X_B = X_A - W  # 0.268: b tight against a
X_C = X_B - W  # 0.218: c tight against b
SET_GAP_B = 0.018  # b is set down with its far face 18 mm from a (the far finger 7 mm from a once open)
SET_GAP_C = 0.022  # c 22 mm from where b should be (11 mm clear; 4 mm from b even if b had not moved)
X_B_SET = X_B - SET_GAP_B  # 0.250
X_C_SET = X_C - SET_GAP_C  # 0.196
# The arms trail a push. Without payload compensation each joint is a spring (80/80/80/10/10/10 N m/rad); the
# compliance model (J K^-1 J^T at the fingers, at these poses) puts a pushing gripper about 6 mm behind its command
# per newton along x (5.8-6.4), and a board slides on about 1.1 N (0.11 kg, friction 1.0; 1.3 N to start it), so a
# gripper pushing one board trails about 6.5 mm and pushing two about 13 mm. Each push is commanded OVERSHOOT past the
# point where the board meets its neighbour: enough for the board to reach it (the push stops when the spring's pull
# drops to the board's friction), not enough to drive the neighbour too (twice the friction). The window is wide:
# the joint closes to 3 mm or less for any compliance up to about 10 mm/N, and the far gap stays 10 mm or more (a
# stiff arm drives the floor on a few mm: board a is set 17 mm from the far wall) for any compliance down to about
# 2 mm/N. On the GPU read joints_mm, far_gap_mm and each board's x_mm in TASK_TRACE at the end of each push and tune
# OVERSHOOT if the joint stays open (larger) or board a moves (smaller).
OVERSHOOT = 0.008
B_PUSH_FROM = X_B_SET - W / 2 - APPROACH - FINGER_FRONT  # 0.209: TCP x where the fingers come down behind b
B_PUSH_TO = X_B - W / 2 - FINGER_FRONT + OVERSHOOT  # 0.243
C_PUSH_FROM = X_C_SET - W / 2 - APPROACH - FINGER_FRONT  # 0.155
C_PUSH_TO = X_C - W / 2 - FINGER_FRONT + OVERSHOOT  # 0.193
BACK_OFF = 0.020  # after each push the fingers back away from the board before rising
PUSH_S = 3.0  # each push: 34-38 mm in 3 s (peak about 19 mm/s)
L_PARK = (0.220, 0.300, L.TOP + 0.150)  # where each arm ends, over its own side, clear of the floor
R_PARK = (0.220, -0.300, L.TOP + 0.150)


def _pick(x: float, y: float, name: str) -> list[Waypoint]:
    """Grip a board lying along x at its middle, across its width (yaw 0: the fingers close along y), and lift it."""
    return [
        Waypoint(2.0, (x, y, HOVER_Z), PICK_OPEN, f"over {name}", 0.0),
        Waypoint(1.5, (x, y, GRIP_Z), PICK_OPEN, f"down round {name}", 0.0),
        Waypoint(0.5, (x, y, GRIP_Z), PICK_OPEN, "settle", 0.0),
        Waypoint(1.5, (x, y, GRIP_Z), GRIP, f"grip {name}", 0.0),
    ]


def _lay(x: float, yaw: float, name: str) -> list[Waypoint]:
    """Carry a gripped board over the bay turning it across (to y), lower it slowly onto the floor at x, let go."""
    return [
        Waypoint(3.0, (x, 0.0, CARRY_Z), GRIP, f"carry {name} over the bay, turning it across", yaw),
        Waypoint(1.2, (x, 0.0, ENTRY_Z), GRIP, "down to 2 cm above the floor", yaw),
        Waypoint(0.5, (x, 0.0, ENTRY_Z), GRIP, "settle", yaw),
        Waypoint(2.0, (x, 0.0, PLACE_Z), GRIP, f"slowly down: {name} onto the floor", yaw),
        Waypoint(1.5, (x, 0.0, PLACE_Z), RELEASE, f"let go of {name}", yaw),
        Waypoint(1.0, (x, 0.0, CLEAR_Z), RELEASE, "straight up, out of the joint", yaw),
    ]


def _push(x_set: float, x_from: float, x_to: float, yaw: float, name: str, neighbour: str) -> list[Waypoint]:
    """Close the fingers in the air, come down behind the board's near face, push it tight against its neighbour,
    back off, rise."""
    return [
        Waypoint(1.5, (x_set, 0.0, CLEAR_Z), CLOSED, "close the fingers for pushing", yaw),
        Waypoint(1.0, (x_from, 0.0, CLEAR_Z), CLOSED, f"behind {name}", yaw),
        Waypoint(1.2, (x_from, 0.0, PUSH_Z), CLOSED, f"down behind {name}", yaw),
        Waypoint(0.5, (x_from, 0.0, PUSH_Z), CLOSED, "settle", yaw),
        Waypoint(PUSH_S, (x_to, 0.0, PUSH_Z), CLOSED, f"push {name} tight against {neighbour}", yaw),
        Waypoint(1.0, (x_to, 0.0, PUSH_Z), CLOSED, "settle", yaw),
        Waypoint(1.0, (x_to - BACK_OFF, 0.0, PUSH_Z), CLOSED, f"back away from {name}", yaw),
        Waypoint(1.2, (x_to - BACK_OFF, 0.0, HOVER_Z), CLOSED, "up", yaw),
    ]


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left() -> list[Waypoint]:
    """Left arm: lay board a along the far wall, lay board b just short of it and push b tight against a, park."""
    ax, ay = L.BOARD_START["board_a"]
    bx, by = L.BOARD_START["board_b"]
    return (
        _pick(ax, ay, "board a")
        + [Waypoint(1.2, (ax, ay, CARRY_Z), GRIP, "lift board a", 0.0)]
        + _lay(X_A, YAW_L, "board a")
        + _pick(bx, by, "board b")
        + [Waypoint(1.2, (bx, by, CARRY_Z), GRIP, "lift board b", 0.0)]
        + _lay(X_B_SET, YAW_L, "board b")
        + _push(X_B_SET, B_PUSH_FROM, B_PUSH_TO, YAW_L, "board b", "board a")
        + [Waypoint(2.0, L_PARK, CLOSED, "park over its own side", YAW_L)]
    )


def _right() -> list[Waypoint]:
    """Right arm: pick board c while the left arm pushes b, hold it on its own side until the left arm has parked,
    then lay it just short of b, push it tight against b, park."""
    cx, cy = L.BOARD_START["board_c"]
    left_done = _span(_left())
    pick = _pick(cx, cy, "board c")
    lift = Waypoint(1.2, (cx, cy, HOLD_Z), GRIP, "lift board c", 0.0)
    # the lift ends 1 s before the left arm has parked: the grip starts while the left arm is pushing b
    wait = left_done - 1.0 - (_span(pick) + lift.duration)
    first = pick[0]
    out = [first, Waypoint(max(wait, 0.5), first.pos, first.finger, "wait over board c", 0.0)] + pick[1:] + [lift]
    out.append(Waypoint(max(left_done - _span(out), 0.5), (cx, cy, HOLD_Z), GRIP,
                        "hold board c until the left arm has parked", 0.0))
    return (
        out
        + _lay(X_C_SET, YAW_R, "board c")
        + _push(X_C_SET, C_PUSH_FROM, C_PUSH_TO, YAW_R, "board c", "board b")
        + [Waypoint(2.0, R_PARK, CLOSED, "park over its own side", YAW_R)]
    )


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
