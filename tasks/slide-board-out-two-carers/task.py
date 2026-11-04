"""On the station table: a white bed deck (a fixed platform 200 x 440 x 30 mm) 28 cm in front of the arm bases,
a loose light-blue slide board (40 x 320 x 20 mm, 0.10 kg) lying flat on it along y, and a loose skin-pink block
standing in for the patient's torso (70 x 60 x 40 mm, 0.20 kg) lying on the board near its left end. After a lateral
transfer the board is withdrawn from under the patient: one carer steadies the patient while the other pulls the
board, because pulling with the patient's weight on it drags the patient. The scorer watches the patient at every
control step: moved more than 15 mm, raised more than 30 mm, tilted more than 10 degrees or let down faster than
0.15 m/s scores 0, whatever follows. The job needs both arms by physics (the board drags the block, friction 1.2)
and by rule, so one gripper holds the patient a few millimetres up while the other pulls the board out.

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
# Success must hold this long with both objects still and both grippers let go. Longer than letting go of the
# patient and lifting the hand clear (1.5 s + 1.0 s in the demo), so the trial cannot end while the pads are still
# beside the patient (the scorer first sees 'let go' about 1.05 s into the release, at an opening of 0.9).
SUCCESS_HOLD_S = 3.0
RUN_LIMIT_STEPS = 1200  # 60 s at 20 Hz

INSTRUCTION = (
    "Pull the blue slide board out from under the pink patient block and leave it lying on the white bed deck, "
    "with the patient block resting flat on the deck where it is. Steady the block with the other gripper while "
    "you pull: it must not be dragged, pushed or carried more than 15 mm, lifted more than 3 cm, tipped, or "
    "dropped. Let go of both at the end."
)

DECK = Box(
    "bed_deck",
    pos=(L.DECK_X, L.DECK_Y, L.TOP + L.DECK_SIZE[2] / 2),
    size=L.DECK_SIZE,
    color=L.WHITE,
)
BOARD = Box(
    "slide_board",
    pos=(L.BOARD_X, L.BOARD_Y, L.DECK_TOP + L.BOARD_SIZE[2] / 2),
    size=L.BOARD_SIZE,
    mass=L.BOARD_MASS,
    color=L.LIGHT_BLUE,
)
PATIENT = Box(
    "patient",
    pos=(L.PATIENT_X, L.PATIENT_Y, L.PATIENT_BOTTOM_Z + L.PATIENT_SIZE[2] / 2),
    size=L.PATIENT_SIZE,
    mass=L.PATIENT_MASS,
    color=L.SKIN_PINK,
)
OBJECTS = [DECK, BOARD, PATIENT]

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
    """Board fully out from under the patient and resting flat on the deck, the patient resting flat on the deck
    within 15 mm of where it lay, both grippers let go, no patient rule broken at any time; held still 2 s. This
    runs first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    patient, board = objects["patient"], objects["slide_board"]
    arms = rigstate.arms()
    result = _TRACKER.update(patient, board, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "patient_pos": patient.pos, "board_pos": board.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, the patient's centre went more than 15 mm from its start, its underside more than 30 mm
    above the board's top, it tilted more than 10 degrees, it came down faster than 0.15 m/s, or it or the board
    left the table; or if its centre ends off the deck. Else 1.0 board fully out (outlines 10 mm apart) and flat on
    the deck, patient flat on the deck, both grippers let go; 0.75 board fully out and patient flat on the deck,
    but a gripper still holding or the board not flat on the deck; 0.5 no more than half of the board's starting
    footprint under the patient still under it; 0.25 the board pulled at least 20 mm along its length; 0
    otherwise."""
    return _TRACKER.result(objects["patient"], objects["slide_board"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PATIENT_GRIP = -(L.PATIENT_SIZE[0] - 0.008) / 2  # 62 mm across the patient's 70 mm x-length
BOARD_GRIP = -(L.BOARD_SIZE[0] - 0.008) / 2  # 32 mm across the board's 40 mm width
# Both grips close the fingers along x (yaw -pi/2 left, +pi/2 right, the signs that reach the midline). Both grip
# points lie inboard of their arm's base (the right arm's between y -0.12 and -0.25, its base at -0.305), where
# those signs keep joint6 clear of its limits.
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
PAD = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004  # the pad tips this far above the held part's underside
APPROACH = 0.030  # each final descent starts this far above the grip height and takes 1.5 s
HOVER = L.TOP + 0.12
# The patient is gripped across its 70 mm x-length, the open pads (inner faces 47 mm from its centre, 8 mm thick)
# coming down outside the 40 mm board's edges (20 mm from the centre), their tips 4 mm above the patient's
# underside: 24 mm above the deck, 36 of its 40 mm held.
PATIENT_TCP_Z = L.PATIENT_BOTTOM_Z + GRIP_UP + PAD
# The raise. The arm's controller does not carry the payload's weight: the shoulder and elbow sag about 5.6 mm
# (and about 1 degree of tilt) under 0.20 kg at this reach (static estimate from the joint stiffness, as in the
# program's other tasks), so a commanded 18 mm raise lifts the patient about 12 mm clear of the board: enough for
# the board to slide out under it, and under the 30 mm limit for any sag from 0. It stays clear of the board for a
# sag up to about 16 mm. On the GPU, read max_raised_mm in TASK_TRACE: keep it 6-18 mm.
RAISE = 0.018
HELD_TCP_Z = PATIENT_TCP_Z + RAISE
# Setting the patient down on the deck, 20 mm below where it lay on the board: first to 10 mm above the deck, then
# the last 10 mm over 1.5 s (under 7 mm/s), aiming its underside 1 mm above the deck top. With the sag it already
# rests on the deck before the slow descent ends (the arm still carrying part of its weight), with no drop; with
# no sag at all it settles 1 mm, which reaches 0.14 m/s at most, under the 0.15 m/s limit.
SET_DROP = 0.001
SET_TCP_Z = L.DECK_TOP + SET_DROP + GRIP_UP + PAD
ENTRY_TCP_Z = L.DECK_TOP + 0.010 + GRIP_UP + PAD
# The board is gripped near its right end across its 40 mm width, the pads (27 mm wide along y) wholly on the board
# 20 mm from its end, their tips 4 mm above the deck (16 of its 20 mm held).
BOARD_TCP_Z = L.DECK_TOP + GRIP_UP + PAD
GRIP_Y0 = L.BOARD_Y - L.BOARD_SIZE[1] / 2 + 0.020  # y -0.12
PULL = 0.130  # along -y: the board then spans y -0.27..+0.05, its centre 7 cm inside the deck's right edge, its
#               right end 10 mm over the edge, its left end 20 mm clear of the patient's -y face (at y +0.07): twice
#               the 10 mm that 'fully out' needs. The pull is resisted only by the board's friction on the deck
#               (about 1.2 N), far under what the pads hold (about 48 N), so the board does not slip in the grip.
GRIP_Y1 = GRIP_Y0 - PULL  # y -0.25: 5.5 cm inboard of the right arm's base, the TCPs 35 cm apart
HOVER_R = L.TOP + 0.10


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left_raise() -> list[Waypoint]:
    px, py = L.PATIENT_X, L.PATIENT_Y
    return [
        Waypoint(1.5, (px, py, HOVER), OPEN, "over the patient", YAW_L),
        Waypoint(1.0, (px, py, PATIENT_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", YAW_L),
        Waypoint(1.5, (px, py, PATIENT_TCP_Z), OPEN, "slowly down beside its x faces, outside the board", YAW_L),
        Waypoint(0.5, (px, py, PATIENT_TCP_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (px, py, PATIENT_TCP_Z), PATIENT_GRIP, "grip the patient", YAW_L),
        Waypoint(0.5, (px, py, PATIENT_TCP_Z), PATIENT_GRIP, "settle the grip", YAW_L),
        Waypoint(1.5, (px, py, HELD_TCP_Z), PATIENT_GRIP, "raise it about 1 cm, straight up", YAW_L),
    ]


PATIENT_UP_S = _span(_left_raise())  # the patient is clear of the board


def _right() -> list[Waypoint]:
    """Right arm: grip the board's right end, wait until the patient is up, pull the board 13 cm along y out from
    under it at deck height, settle, let go and go up."""
    bx = L.BOARD_X
    pick = [
        Waypoint(1.5, (bx, GRIP_Y0, HOVER_R), OPEN, "over the board's right end", YAW_R),
        Waypoint(1.0, (bx, GRIP_Y0, BOARD_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", YAW_R),
        Waypoint(1.5, (bx, GRIP_Y0, BOARD_TCP_Z), OPEN, "slowly down beside its long sides", YAW_R),
        Waypoint(0.5, (bx, GRIP_Y0, BOARD_TCP_Z), OPEN, "settle", YAW_R),
        Waypoint(1.5, (bx, GRIP_Y0, BOARD_TCP_Z), BOARD_GRIP, "grip the board", YAW_R),
        Waypoint(0.5, (bx, GRIP_Y0, BOARD_TCP_Z), BOARD_GRIP, "settle the grip", YAW_R),
    ]
    wait = PATIENT_UP_S + 0.5 - _span(pick)
    return pick + [
        Waypoint(wait, (bx, GRIP_Y0, BOARD_TCP_Z), BOARD_GRIP, "wait for the patient to be up", YAW_R),
        Waypoint(4.0, (bx, GRIP_Y1, BOARD_TCP_Z), BOARD_GRIP, "pull the board out from under the patient", YAW_R),
        Waypoint(0.5, (bx, GRIP_Y1, BOARD_TCP_Z), BOARD_GRIP, "settle", YAW_R),
        Waypoint(1.5, (bx, GRIP_Y1, BOARD_TCP_Z), OPEN, "let go of the board", YAW_R),
        Waypoint(1.0, (bx, GRIP_Y1, HOVER_R), OPEN, "up", YAW_R),
    ]


BOARD_OUT_S = _span(_right()[:9])  # the pull is complete and has settled


def _left() -> list[Waypoint]:
    """Left arm: grip the patient across its x-length, raise it about 1 cm and hold it where it is until the board
    is out, lower it slowly onto the deck, let go, go up."""
    px, py = L.PATIENT_X, L.PATIENT_Y
    up = _left_raise()
    hold = BOARD_OUT_S + 0.5 - _span(up)
    return up + [
        Waypoint(hold, (px, py, HELD_TCP_Z), PATIENT_GRIP, "hold it steady while the board comes out", YAW_L),
        Waypoint(1.5, (px, py, ENTRY_TCP_Z), PATIENT_GRIP, "lower it to 1 cm above the deck", YAW_L),
        Waypoint(0.5, (px, py, ENTRY_TCP_Z), PATIENT_GRIP, "settle", YAW_L),
        Waypoint(1.5, (px, py, SET_TCP_Z), PATIENT_GRIP, "lower the last 1 cm slowly onto the deck", YAW_L),
        Waypoint(0.5, (px, py, SET_TCP_Z), PATIENT_GRIP, "settle", YAW_L),
        Waypoint(1.5, (px, py, SET_TCP_Z), OPEN, "let go of the patient", YAW_L),
        Waypoint(1.0, (px, py, HOVER), OPEN, "up", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
