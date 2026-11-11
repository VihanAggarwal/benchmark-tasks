"""On the station table: a 1:15 model of one bay of beam formwork. Two girders run along x 20 cm either side of the
midline, each an open trough 50 mm wide between dark plywood side forms 25 mm tall, with a grey concrete column at
each end (column centres 14 and 44 cm in front of the arm bases: a 300 mm span). A cross beam's trough, 20 mm wide,
runs along y between the two girders and frames into each at x 23.5-25.5 cm. A loose yellow plywood stop-end
(40 x 25 x 60 mm, 0.03 kg) stands on the table beside each girder, turned 30 degrees. Rule (ACI 318, construction
joints): stand one stop-end across each girder, upright and square, wholly inside the middle third of its span
(x 24.0-34.0 cm) and at least twice the cross beam's width (40 mm) clear of the cross beam's face; then let go.
Only x 29.5-34.0 cm meets both; the mid-span spot (x 29.0 cm) meets only the first.

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
# Still for 1 s: both stop-ends standing in the band, let go (finger zones 20 mm clear), neither moving.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Today's pour stops in this bay. Stand one yellow stop-end upright across each of the two girder forms, square "
    "to it and on its floor, with all of it inside the middle third of the girder's span (between the column "
    "centres) and at least twice the cross beam's width (40 mm) clear of the cross beam's side. Then let go of both."
)

COLUMNS = [
    Box(name, pos=(x, y, L.TOP + L.COL_SIZE[2] / 2), size=L.COL_SIZE, color=L.CONCRETE)
    for name, x, y in L.COLUMNS
]
FORMS = [
    Box(name, pos=(x, y, L.TOP + L.FORM_H / 2), size=(sx, sy, L.FORM_H), color=L.FORM_PLY)
    for name, x, y, sx, sy in L.FORMS
]
STOP_ENDS = [
    Box(name, pos=(x, y, L.TOP + L.STOP_HALF[2]), size=L.STOP_SIZE, mass=L.STOP_MASS, color=L.STOP_YELLOW, yaw=yaw)
    for name, (x, y, yaw) in L.STOP_START.items()
]
OBJECTS = [*COLUMNS, *FORMS, *STOP_ENDS]

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


def _stops(objects) -> dict:
    return {n: objects.get(n) for n in L.STOP_NAMES}


def success(objects) -> bool:
    """Each girder has a stop-end standing across it (upright within 10 deg, on the floor within 5 mm, square
    within 10 deg, inside the trough, both grippers' finger zones 20 mm or more from it) wholly in x 0.295-0.340,
    and no other stop-end over it; both still for 1 s. This runs first every control step, so it also feeds the
    tracker (with the arms rigstate read for this step)."""
    stops = _stops(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(stops, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in stops.items():
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
    """Per girder, by the worst stop-end whose centre is over its trough (0 if none): 0.5 standing across it wholly
    in x 0.295-0.340 (middle third and 40 mm clear of the cross beam's face); 0.25 standing across it wholly in the
    middle third x 0.240-0.340 but nearer the cross beam; 0 otherwise. The sum: 0, 0.25, 0.5, 0.75 or 1.0."""
    return _TRACKER.result(_stops(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
T = L.STOP_SIZE[0]  # 25 mm: every grip is across a stop-end's thickness
OPEN_GAP = T + 0.020  # 45 mm open: 10 mm each side of the stop-end
PICK_OPEN = -OPEN_GAP / 2
GRIP = -(T - 0.008) / 2  # 17 mm commanded
RELEASE = PICK_OPEN
# Yaw: at yaw 0 the fingers close along y; the left arm turns to -90 deg and the right to +90 deg to close along x.
# Each stop-end starts with its thickness turned 30 degrees (left +30, right -30), so the grips start 30 deg short.
YAW_L, YAW_R = -math.pi / 2, math.pi / 2
YAW_L_PICK = YAW_L + L.STOP_START["stop_end_left"][2]  # -60 deg
YAW_R_PICK = YAW_R + L.STOP_START["stop_end_right"][2]  # +60 deg
GRIP_Z = L.TOP + 0.052  # TCP 52 mm up: pad tips 39 mm up, 14 mm over the forms, on the top 21 mm of the stop-end
PLACE_Z = GRIP_Z + 0.0005  # let go with the stop-end's bottom 0.5 mm above the floor
# The slow part of the set-down starts with the stop-end's bottom 10 mm above the forms' tops, so it enters the
# trough (5 mm clear of each side form) slowly and after a pause, not at the end of the faster drop from the carry.
ENTRY_Z = GRIP_Z + L.FORM_H + 0.010  # TCP 87 mm up: the stop-end's bottom 35 mm up
CARRY_Z = L.TOP + 0.130  # a carried stop-end's bottom 78 mm up: 53 mm over the forms, 38 mm over the columns
CLEAR_Z = L.TOP + 0.150  # after letting go: pad tips 77 mm over the stop-end's top
# Where each stop-end goes: the middle of the band x 0.295-0.340 that it must lie wholly in (its 25 mm thickness
# along x): centre x 0.3175, 10 mm from either limit; across the trough's centre line.
PLACE_X = (L.BAND_LO + L.MID_HI) / 2  # 0.3175
L_PARK = (0.220, 0.320, L.TOP + 0.150)  # where each arm ends, over its own side, clear of the girders
R_PARK = (0.220, -0.320, L.TOP + 0.150)


def _arm(name: str, start: tuple, yaw_pick: float, yaw_place: float, sgn: float, park: tuple) -> list[Waypoint]:
    """Pick a stop-end standing on the table at its top, lift it over the forms, carry it over its girder turning it
    square, lower it to 1 cm above the forms, then slowly into the trough and onto its floor, let go, rise, park."""
    sx, sy = start[0], start[1]
    px, py = PLACE_X, sgn * L.GIRDER_Y
    return [
        Waypoint(2.0, (sx, sy, CARRY_Z), PICK_OPEN, f"over {name}", yaw_pick),
        Waypoint(1.5, (sx, sy, GRIP_Z), PICK_OPEN, f"down round {name}", yaw_pick),
        Waypoint(0.5, (sx, sy, GRIP_Z), PICK_OPEN, "settle", yaw_pick),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, f"grip {name}", yaw_pick),
        Waypoint(1.2, (sx, sy, CARRY_Z), GRIP, f"lift {name} over the forms", yaw_pick),
        Waypoint(2.5, (px, py, CARRY_Z), GRIP, f"carry {name} over the girder, turning it square", yaw_place),
        Waypoint(1.2, (px, py, ENTRY_Z), GRIP, "down to 1 cm above the forms", yaw_place),
        Waypoint(0.8, (px, py, ENTRY_Z), GRIP, "settle", yaw_place),
        Waypoint(3.0, (px, py, PLACE_Z), GRIP, f"slowly down: {name} into the trough, onto its floor", yaw_place),
        Waypoint(1.5, (px, py, PLACE_Z), RELEASE, f"let go of {name}", yaw_place),
        Waypoint(1.5, (px, py, CLEAR_Z), RELEASE, "straight up", yaw_place),
        Waypoint(1.5, park, RELEASE, "park over its own side", yaw_place),
    ]


def _left() -> list[Waypoint]:
    return _arm("the left stop-end", L.STOP_START["stop_end_left"], YAW_L_PICK, YAW_L, 1.0, L_PARK)


def _right() -> list[Waypoint]:
    return _arm("the right stop-end", L.STOP_START["stop_end_right"], YAW_R_PICK, YAW_R, -1.0, R_PARK)


def demo():
    """Both arms at once, each on its own girder. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
