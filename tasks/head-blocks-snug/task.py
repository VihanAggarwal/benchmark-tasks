"""On the station table: a yellow spine board (480 x 300 x 15 mm, fixed) along x, head end toward the arms, with a tan
casualty trainer lying on its back on it: torso and neck fixed to the board, and the head a separate weighted
cylinder (120 mm across, 100 mm tall, 0.30 kg) resting free on the board at x 20 cm, 10 mm short of the neck, so any
push moves it. Two orange foam head blocks (70 x 50 x 80 mm, 40 g) stand on the table beside the board, one in each
arm's column (x 20 cm, y +-28 cm). Rule: in spinal motion restriction the blocks go snug against both sides of the
head without moving it: each block upright on the board beside the head, its 70 mm side along the board, gap 5 mm
or less; the head never pushed more than 8 mm, lifted or tipped; then the blocks are let go.

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
from scoring import HEAD, NAMES, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # the 1.0 conditions (both blocks snug and let go) held with everything still for 1 s

INSTRUCTION = (
    "The training casualty lies on a spine board and its head must not move. Stand one orange head block upright "
    "on the board at each side of the head, with its long side snug against the head, without pushing the head, "
    "then let go of both blocks."
)

BOARD = Box("spine_board", pos=((L.BOARD_X0 + L.BOARD_X1) / 2, 0.0, L.TOP + L.BOARD_T / 2),
            size=(L.BOARD_X1 - L.BOARD_X0, 2 * L.BOARD_HALF_W, L.BOARD_T), color=L.YELLOW)
TORSO = Box("torso", pos=((L.TORSO_X0 + L.TORSO_X1) / 2, 0.0, L.BOARD_TOP + L.TORSO_H / 2),
            size=(L.TORSO_X1 - L.TORSO_X0, 2 * L.TORSO_HALF_W, L.TORSO_H), color=L.TAN)
NECK = Box("neck", pos=((L.NECK_X0 + L.NECK_X1) / 2, 0.0, L.BOARD_TOP + L.NECK_H / 2),
           size=(L.NECK_X1 - L.NECK_X0, 2 * L.NECK_HALF_W, L.NECK_H), color=L.TAN)
HEAD_PART = Cylinder(HEAD, pos=(L.HEAD_X, L.HEAD_Y, L.HEAD_Z), radius=L.HEAD_R, height=L.HEAD_H,
                     mass=L.HEAD_MASS, color=L.TAN)
BLOCKS = [
    Box(name, pos=(x, y, L.TOP + L.BLOCK_SIZE[2] / 2), size=L.BLOCK_SIZE, mass=L.BLOCK_MASS, color=L.ORANGE)
    for name, (x, y) in L.BLOCKS.items()
]
OBJECTS = [BOARD, TORSO, NECK, HEAD_PART, *BLOCKS]

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


def _blocks(objects) -> dict:
    return {n: objects[n] for n in NAMES}


def success(objects) -> bool:
    """A block snug (gap 5 mm or less) against each side of the head, upright on the board, let go; the head never
    moved more than 8 mm, lifted or tipped; everything still for 1 s. This runs first every control step, so it also
    feeds the tracker (with the arms rigstate read for this step)."""
    blocks, head = _blocks(objects), objects[HEAD]
    arms = rigstate.arms()
    result = _TRACKER.update(blocks, head, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "pos": {n: s.pos for n, s in [*blocks.items(), (HEAD, head)]}}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the head was ever pushed more than 8 mm from its start, lifted more than 10 mm or tipped over 10 deg, or
    if a block reaches more than 5 mm inside the head's outline seen from above. Else, for each side of the head, its
    best block standing there (upright within 10 deg, 70 mm side within 15 deg of the board's axis, on the board,
    centre within 25 mm of the head's in x, let go): 0.5 if the gap is 5 mm or less, 0.25 if 15 mm or less; the score
    is the sum of the two sides."""
    return _TRACKER.result(_blocks(objects), objects[HEAD], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.BLOCK_SIZE[0] - 0.008) / 2  # across the block's 70 mm length: 62 mm commanded
YAW = {"left": -math.pi / 2, "right": math.pi / 2}  # the fingers close along x, on the block's front and back faces
TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = L.BLOCK_SIZE[2] - 0.025 + TIPS  # TCP 68 mm above the block's bottom: pad tips 25 mm below its top
# Each block is gripped 10 mm off its centre, away from the head: no finger is ever between the block and the head
# (they close on its front and back faces), and the fingers stay at least 12 mm from the head (yamkin's capsules).
GRIP_OFF = 0.010
HOVER_Z = L.TOP + 0.160
PICK_Z = L.TOP + GRIP_UP  # the block stands on the table
ENTRY_Z = PICK_Z + 0.030  # the last 30 mm down onto a block are taken slowly
CARRY_Z = L.BOARD_TOP + 0.008 + GRIP_UP  # carried with its bottom 8 mm above the board's top
PLACE_Z = L.BOARD_TOP + 0.001 + GRIP_UP  # set down with its bottom aimed 1 mm above the board (the arm sags a little)
# Each block is set down with its inner face 2 mm from the head (its centre 87 mm from the head's axis). It comes in
# low to 10 mm further out, then closes the last 10 mm slowly, then is lowered 7 mm straight down: an overshoot of a
# few mm touches the head and moves it a little (8 mm allowed, gap 0 is snug), a shortfall of up to 3 mm is still
# snug (5 mm).
PLACE_GAP = 0.002
PLACE_Y = L.HEAD_R + PLACE_GAP + L.BLOCK_HALF[1]
APPROACH = 0.010
OUT_Y = 0.250  # where the left gripper goes, high, once its block is down
X = L.BLOCK_X  # both blocks start, travel and end centred at x 0.20 (the head's centre)
RIGHT_WAIT = 11.5  # the right arm waits over its block until the left gripper is up and moving away


def _pick(sign: int, yaw: float) -> list[Waypoint]:
    """Down round the block on this side, grip it across its 70 mm length, lift it to carry height."""
    y = sign * (L.BLOCK_Y + GRIP_OFF)
    return [
        Waypoint(1.5, (X, y, ENTRY_Z), OPEN, "down to just above the block", yaw),
        Waypoint(1.5, (X, y, PICK_Z), OPEN, "slowly down round the block", yaw),
        Waypoint(0.5, (X, y, PICK_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (X, y, PICK_Z), GRIP, "grip the block across its length", yaw),
        Waypoint(0.5, (X, y, PICK_Z), GRIP, "settle the grip", yaw),
        Waypoint(1.0, (X, y, CARRY_Z), GRIP, "lift it clear of the board's height", yaw),
    ]


def _place(sign: int, yaw: float) -> list[Waypoint]:
    """Carry the block in low beside the head, close the last 10 mm slowly, set it down, let go, straight up."""
    y_out = sign * (PLACE_Y + APPROACH + GRIP_OFF)
    y_in = sign * (PLACE_Y + GRIP_OFF)
    return [
        Waypoint(3.0, (X, y_out, CARRY_Z), GRIP, "carry it in low, 12 mm from the head", yaw),
        Waypoint(0.5, (X, y_out, CARRY_Z), GRIP, "settle", yaw),
        Waypoint(2.0, (X, y_in, CARRY_Z), GRIP, "close the last 10 mm slowly", yaw),
        Waypoint(0.5, (X, y_in, CARRY_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (X, y_in, PLACE_Z), GRIP, "set it down on the board", yaw),
        Waypoint(0.5, (X, y_in, PLACE_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (X, y_in, PLACE_Z), OPEN, "let go", yaw),
        Waypoint(0.5, (X, y_in, PLACE_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (X, y_in, HOVER_Z), OPEN, "straight up, clear of the block and the head", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: its block against the left side of the head, then up and out of the way (21.5 s)."""
    yaw = YAW["left"]
    return [Waypoint(2.0, (X, L.BLOCK_Y + GRIP_OFF, HOVER_Z), OPEN, "over the block", yaw), *_pick(1, yaw),
            *_place(1, yaw), Waypoint(1.5, (X, OUT_Y, HOVER_Z), OPEN, "out of the way", yaw)]


def _right() -> list[Waypoint]:
    """Right arm: waits over its block while the left works, then sets it against the right side of the head
    (31.5 s)."""
    yaw = YAW["right"]
    y0 = -(L.BLOCK_Y + GRIP_OFF)
    return [Waypoint(2.0, (X, y0, HOVER_Z), OPEN, "over the block", yaw),
            Waypoint(RIGHT_WAIT, (X, y0, HOVER_Z), OPEN, "wait for the left arm", yaw),
            *_pick(-1, yaw), *_place(-1, yaw)]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
