"""On the station table: a fixed white bed deck (200 x 300 x 30 mm) between the arms and a little to the right, with
a thin pale green mark on its left half where the patient goes; a fixed dark grey trolley deck (200 x 160 x 30 mm)
at the left, 40 mm from the bed, out of the right arm's straight-down reach, with a loose skin-pink 'patient' block (70 x 60 x
40 mm, 0.20 kg) lying on it; and a loose light grey rail (a 200 x 30 x 40 mm bar, 0.10 kg) lying on the table at the
right, out of the left arm's straight-down reach. Fall-prevention rule: once the patient is on the bed, a gripper must keep a hand
on them on every control step until the rail is resting along the bed's right edge; the patient is never dropped
and never tips. The scorer checks it at every control step, not only at the end.

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
# Success must hold this long with the patient and the rail still. Longer than the demo's release (1.5 s) plus
# rise (1.0 s), so the trial cannot end while a gripper still holds the patient or the rail; and the 1.0 conditions
# need both grippers let go (each 0.9 open, or its finger zone more than 10 mm from both items), so the hold starts
# counting only once the last gripper has opened past 0.9.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Move the pink patient block from the dark grey trolley onto the green mark on the white bed, and put the light "
    "grey rail up along the bed's right edge, lying along the bed. Once the patient is on the bed, keep a gripper on "
    "them until the rail is up: never leave the patient on the bed with the rail down and no gripper on them. Lift "
    "the patient, do not slide them off the trolley; lower them gently and do not drop or tip them. Let go of both "
    "at the end."
)

BED = Box("bed_deck", pos=(L.BED_X, L.BED_Y, L.TOP + L.BED_SIZE[2] / 2), size=L.BED_SIZE, color=L.BED_WHITE)
MARK = Box("bed_mark", pos=(L.MARK_X, L.MARK_Y, L.DECK_TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE,
           color=L.MARK_GREEN)
TROLLEY = Box("trolley_deck", pos=(L.TROLLEY_X, L.TROLLEY_Y, L.TOP + L.TROLLEY_SIZE[2] / 2), size=L.TROLLEY_SIZE,
              color=L.TROLLEY_GREY)
PATIENT = Box(
    "patient",
    pos=(L.PATIENT_X, L.PATIENT_Y, L.TROLLEY_TOP + L.PATIENT_SIZE[2] / 2),
    size=L.PATIENT_SIZE,
    mass=L.PATIENT_MASS,
    color=L.SKIN_PINK,
)
RAIL = Box(
    "rail",
    pos=(L.RAIL_X, L.RAIL_Y, L.TOP + L.RAIL_SIZE[2] / 2),
    size=L.RAIL_SIZE,
    mass=L.RAIL_MASS,
    color=L.RAIL_GREY,
)
OBJECTS = [BED, MARK, TROLLEY, PATIENT, RAIL]

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
    """Patient flat on the bed mark within 15 mm, the rail up along the bed's right edge, both grippers let go, the
    patient never on the bed unattended with the rail down, never dropped, never tipped, never slid off the trolley;
    held still 3 s. This runs
    first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    patient, rail = objects["patient"], objects["rail"]
    arms = rigstate.arms()
    result = _TRACKER.update(patient, rail, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "patient_pos": patient.pos, "rail_pos": rail.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if on any step the patient was on the bed (any part of their outline over the deck, lowest point within
    10 mm of its top) with the rail not up and no gripper's hand on them (a gripper under 0.9 open, its finger zone
    overlapping them, its finger tips no higher than 10 mm above their top), or the patient ever descended faster than 0.15 m/s, tilted more than 15 degrees, or slid more than 15 mm
    while within 10 mm of the trolley's top or the table (pushed, not lifted); or if the patient or the rail ends
    off the table, or the patient ends on the table off both platforms. Else 1.0 rail up, patient flat
    on the mark within 15 mm, both grippers let go; 0.75 rail up, patient flat on the bed within 30 mm of the mark
    but not all of that; 0.5 rail up and the patient resting on the bed anywhere (or tilted 5 to 15 degrees), or
    rail not up and the patient flat on the bed within 30 mm with a hand on them; 0.25 exactly one of {rail up,
    patient resting on the bed with a hand on them}; 0 otherwise."""
    return _TRACKER.result(objects["patient"], objects["rail"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PATIENT_GRIP = -(L.PATIENT_SIZE[0] - 0.008) / 2  # 62 mm commanded across the patient's 70 mm (along x)
RAIL_GRIP = -(L.RAIL_SIZE[1] - 0.008) / 2  # 22 mm commanded across the rail's 30 mm (along y)
# The left grips the patient with the fingers closing along x (yaw -pi/2, the sign that reaches the midline): the
# open fingers (94 mm apart, 110 mm over the pads) straddle it along x over the 200 mm trolley and, at the set-down,
# over the 200 mm deck. The right grips the rail at its centre with the fingers closing along y (yaw 0): the open
# fingers span 110 mm along y about the rail, 35 mm clear of the deck's edge at the pick, and over the deck and the
# table's edge beside it at the set-down.
YAW_L = -math.pi / 2
YAW_R = 0.0
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = 0.004  # both parts are gripped with the finger tips 4 mm above their bottom: 36 of the patient's 40 mm,
#                  36 of the rail's 40 mm
PATIENT_TCP_Z = L.TROLLEY_TOP + GRIP_UP + PAD_TIPS  # TOP + 0.047: tips 4 mm above the trolley's top
RAIL_TCP_Z = L.TOP + GRIP_UP + PAD_TIPS  # TOP + 0.017: tips 4 mm above the table
APPROACH = 0.030  # each final descent to the grip starts this far above it and takes 1.5 s
HOVER_L = L.TOP + 0.120  # the left's carrying height: the patient's bottom 103 mm up as commanded, 73 mm over the
#                          platforms' tops (the arm sags about 6 mm under the 0.20 kg patient at this reach)
HOVER_R = L.TOP + 0.100  # the right's carrying height: the rail's bottom 83 mm up, 53 mm over the deck's top
# Set-down: let go with the patient's bottom 2 mm above the deck's top, 1 mm above the mark (the mark is 1 mm thick
# on the deck). With the sag the patient already rests on the mark, the arm still carrying part of its weight, when
# the fingers open: no drop; without sag it falls 1 mm, a step-to-step descent of 0.02 m/s, well under the 0.15 m/s
# limit. Either way its lowest point is within the 3 mm 'resting' band while held, so the hold is 'safe but
# unfinished' (0.5) until the rail is up.
PATIENT_SET_TCP_Z = L.DECK_TOP + 0.002 + GRIP_UP + PAD_TIPS
RAIL_SET_TCP_Z = L.DECK_TOP + 0.002 + GRIP_UP + PAD_TIPS  # the rail's bottom 2 mm above the deck's top
ENTRY = 0.020  # the last 20 mm of each set-down are lowered slowly (1.5 s)
UP_L = L.TOP + 0.120  # where the left goes after letting go
UP_R = L.TOP + 0.120  # where the right goes after letting go: straight up first (the opening fingers, their tips
#                       4 mm above the deck, straddle the 40 mm rail: a sideways move would knock it), then back out
SETDOWN_MARGIN_S = 0.5  # the right brings the rail in this long after the patient's set-down is complete
RAIL_MARGIN_S = 1.5  # the left keeps its hand on the patient this long after the rail's set-down is complete and
#                      settled, before it starts to open: the right's 1.5 s release, so the rail has been let go
#                      (0.5 s after it is up, plus the arm's tracking lag) before the patient is


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left_setdown() -> list[Waypoint]:
    """Left arm: take the patient off the trolley, carry them over the bed mark, lower them onto it."""
    px, py = L.PATIENT_X, L.PATIENT_Y
    mx, my = L.MARK_X, L.MARK_Y
    return [
        Waypoint(1.5, (px, py, HOVER_L), OPEN, "over the patient", YAW_L),
        Waypoint(1.0, (px, py, PATIENT_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", YAW_L),
        Waypoint(1.5, (px, py, PATIENT_TCP_Z), OPEN, "slowly down beside the patient", YAW_L),
        Waypoint(0.5, (px, py, PATIENT_TCP_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (px, py, PATIENT_TCP_Z), PATIENT_GRIP, "grip the patient", YAW_L),
        Waypoint(0.5, (px, py, PATIENT_TCP_Z), PATIENT_GRIP, "settle the grip", YAW_L),
        Waypoint(1.5, (px, py, HOVER_L), PATIENT_GRIP, "lift straight up", YAW_L),
        Waypoint(2.0, (mx, my, HOVER_L), PATIENT_GRIP, "carry over the bed mark", YAW_L),
        Waypoint(1.0, (mx, my, PATIENT_SET_TCP_Z + ENTRY), PATIENT_GRIP, "down to 2 cm above the set-down", YAW_L),
        Waypoint(0.5, (mx, my, PATIENT_SET_TCP_Z + ENTRY), PATIENT_GRIP, "settle", YAW_L),
        Waypoint(1.5, (mx, my, PATIENT_SET_TCP_Z), PATIENT_GRIP, "lower the last 2 cm slowly onto the mark", YAW_L),
        Waypoint(0.5, (mx, my, PATIENT_SET_TCP_Z), PATIENT_GRIP, "settle", YAW_L),
    ]


PATIENT_DOWN_S = _span(_left_setdown())  # the patient's set-down is complete and settled


def _right() -> list[Waypoint]:
    """Right arm: pick the rail up off the table, hold it in the air until the patient is set down on the bed, then
    carry it to the bed's right edge, lower it onto the deck, let go, go up and back out."""
    rx, ry = L.RAIL_X, L.RAIL_Y
    ex, ey = L.RAIL_PLACE_X, L.RAIL_PLACE_Y
    pick = [
        Waypoint(1.5, (rx, ry, HOVER_R), OPEN, "over the rail", YAW_R),
        Waypoint(1.0, (rx, ry, RAIL_TCP_Z + APPROACH), OPEN, "down to 3 cm above the grip", YAW_R),
        Waypoint(1.5, (rx, ry, RAIL_TCP_Z), OPEN, "slowly down beside the rail", YAW_R),
        Waypoint(0.5, (rx, ry, RAIL_TCP_Z), OPEN, "settle", YAW_R),
        Waypoint(1.5, (rx, ry, RAIL_TCP_Z), RAIL_GRIP, "grip the rail at its centre", YAW_R),
        Waypoint(0.5, (rx, ry, RAIL_TCP_Z), RAIL_GRIP, "settle the grip", YAW_R),
        Waypoint(1.2, (rx, ry, HOVER_R), RAIL_GRIP, "lift straight up", YAW_R),
    ]
    wait = PATIENT_DOWN_S + SETDOWN_MARGIN_S - _span(pick)
    return pick + [
        Waypoint(wait, (rx, ry, HOVER_R), RAIL_GRIP, "wait in the air until the patient is on the bed", YAW_R),
        Waypoint(1.5, (ex, ey, HOVER_R), RAIL_GRIP, "carry it in over the bed's right edge", YAW_R),
        Waypoint(1.0, (ex, ey, RAIL_SET_TCP_Z + ENTRY), RAIL_GRIP, "down to 2 cm above the set-down", YAW_R),
        Waypoint(0.5, (ex, ey, RAIL_SET_TCP_Z + ENTRY), RAIL_GRIP, "settle", YAW_R),
        Waypoint(1.5, (ex, ey, RAIL_SET_TCP_Z), RAIL_GRIP, "lower the last 2 cm slowly onto the deck", YAW_R),
        Waypoint(0.5, (ex, ey, RAIL_SET_TCP_Z), RAIL_GRIP, "settle", YAW_R),
        Waypoint(1.5, (ex, ey, RAIL_SET_TCP_Z), OPEN, "let go", YAW_R),
        Waypoint(1.0, (ex, ey, UP_R), OPEN, "straight up, clear of the rail", YAW_R),
        Waypoint(1.0, (rx, ry, UP_R), OPEN, "back out over the table", YAW_R),
    ]


RAIL_UP_S = _span(_right()[:13])  # the rail's set-down is complete and settled: it rests along the deck's edge


def _left() -> list[Waypoint]:
    """Left arm: set the patient on the bed mark, keep the fingers closed on them until the rail is up plus a
    margin, then let go and go up."""
    mx, my = L.MARK_X, L.MARK_Y
    setdown = _left_setdown()
    hold = RAIL_UP_S + RAIL_MARGIN_S - _span(setdown)
    return setdown + [
        Waypoint(hold, (mx, my, PATIENT_SET_TCP_Z), PATIENT_GRIP, "keep a hand on the patient until the rail is up",
                 YAW_L),
        Waypoint(1.5, (mx, my, PATIENT_SET_TCP_Z), OPEN, "let go", YAW_L),
        Waypoint(1.0, (mx, my, UP_L), OPEN, "up", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
