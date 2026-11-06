"""On the station table: a streaky amber glass sheet (180 mm along x by 300 mm along y, 3 mm thick, taped down, so
fixed) centred on the midline 29 cm in front of the arm bases, built from 12 strips 25 mm wide in alternating dark
and light amber, so its streaks run front to back, across its short side; and three loose white acrylic pattern
templates on the table: A1 (70 x 40 x 12 mm) and A2 (the same) to the left of the sheet, A1 with its long side
across the streaks and A2 with it already along them, and B (60 x 40 x 12 mm) to the right, turned 45 degrees.
Rule (a stained-glass cutting layout): every template laid flat on the sheet with its long side (its grain arrow)
along the streaks, at least 8 mm in from every sheet edge and 5 mm from the others, lined up along one long edge so a
strip at least 80 mm wide stays clear along the other; grippers let go and their fingers at least 30 mm away.

The scorer reads both grippers' finger-tip points, axes and openings through rigstate.py, which wraps the
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
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# All three templates laid and the clear strip at least 80 mm, held for 1 s, ends the trial. 'Laid' already needs
# both grippers' finger zones 30 mm or more from the template, so success cannot start while a gripper still holds
# one or an open finger still rests on one.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Lay the three white templates flat on the amber glass sheet, ready to trace. Turn each one so its long side "
    "runs along the sheet's streaks. Keep every template at least 8 mm in from the sheet's edges and at least 5 mm "
    "from the others, and line them up along one long edge of the sheet so that a strip at least 8 cm wide stays "
    "clear along the other long edge. Then let go and move the grippers at least 3 cm away from the templates."
)

STRIPS = [
    Box(f"glass_strip_{k:02d}", pos=(L.SHEET_X, y, L.TOP + L.SHEET_THICK / 2), size=L.STRIP_SIZE,
        color=L.DARK_AMBER if k % 2 == 0 else L.LIGHT_AMBER)
    for k, y in enumerate(L.STRIP_Y)
]
TEMPLATES = [
    Box(name, pos=(x, y, L.TOP + size[2] / 2), size=size, mass=mass, color=L.WHITE, yaw=yaw)
    for name, (size, mass, x, y, yaw) in L.TEMPLATES.items()
]
OBJECTS = [*STRIPS, *TEMPLATES]

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


def _templates(objects) -> dict:
    return {n: objects[n] for n in L.NAMES if n in objects}


def success(objects) -> bool:
    """All three templates laid (flat on the sheet, at rest, let go, 8 mm inside every edge, 5 mm apart, long side
    within 5 degrees of the streaks) and a clear strip of at least 80 mm along one long edge; held 1 s. This runs first
    every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    templates = _templates(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(templates, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in templates.items():
        row[f"{n}_pos"] = s.pos
        row[f"{n}_quat"] = s.quat_xyzw
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.2 for each laid template (0, 0.2, 0.4 or 0.6); 1.0 when all three are laid and the clear strip along one long
    edge is at least 80 mm. Read on the last step (SCORE_OVER 'end')."""
    return _TRACKER.result(_templates(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
A_GRIP = -(L.A_SIZE[0] - 0.008) / 2  # -0.031: 62 mm across A's 70 mm length
B_GRIP = -(L.B_SIZE[0] - 0.008) / 2  # -0.026: 52 mm across B's 60 mm length
A_RELEASE = -0.040  # 80 mm: 5 mm clear of each end of a 70 mm template
B_RELEASE = -0.036  # 72 mm: 6 mm clear of each end of B
# Every grip closes across the template's LENGTH, so the pads sit at its two ends (never in the gaps between
# templates, which run along y). Heights are the TCP's above the table top; the pad tips are 13 mm below the TCP.
PICK_H = 0.017  # pad tips 4 mm above the table: an 8 mm grip band on the 12 mm block
PLACE_H = 0.021  # the template's underside 1 mm above the sheet's top (3 mm) when let go; open pads 5 mm above it
ENTRY_H = PLACE_H + 0.015  # the last 15 mm are lowered slowly
CARRY_H = 0.100  # a carried template's underside 83 mm above the table
PARK_H = 0.120
# Yaw 0 closes the fingers along y; the left arm uses -pi/2 and the right +pi/2 to close along x at the midline.
YAW_A1 = 0.0  # across A1's length, which starts along y
YAW_X_LEFT = -math.pi / 2
YAW_B = math.pi / 4  # along B's long side, which starts at -45 degrees
YAW_X_RIGHT = math.pi / 2
LEFT_PARK = (0.240, 0.300)
RIGHT_PARK = (0.240, -0.300)


def _z(h: float) -> float:
    return L.TOP + h


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _place(x: float, y: float, yaw: float, grip: float, release: float, what: str) -> list[Waypoint]:
    """From carry height over (x, y): down, the last 15 mm slowly, settle, let go, straight up."""
    return [
        Waypoint(1.5, (x, y, _z(ENTRY_H)), grip, f"down over {what}", yaw),
        Waypoint(1.0, (x, y, _z(PLACE_H)), grip, f"lower {what} the last 15 mm slowly", yaw),
        Waypoint(0.5, (x, y, _z(PLACE_H)), grip, "settle", yaw),
        Waypoint(1.5, (x, y, _z(PLACE_H)), release, f"let go of {what}", yaw),
        Waypoint(1.5, (x, y, _z(CARRY_H)), release, "straight up", yaw),
    ]


def _pick(x: float, y: float, yaw: float, grip: float, what: str, hover_s: float) -> list[Waypoint]:
    """To carry height over (x, y), down with the fingers open, settle, close, lift."""
    return [
        Waypoint(hover_s, (x, y, _z(CARRY_H)), OPEN, f"over {what}", yaw),
        Waypoint(1.5, (x, y, _z(PICK_H)), OPEN, f"down to {what}", yaw),
        Waypoint(0.5, (x, y, _z(PICK_H)), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, _z(PICK_H)), grip, f"grip {what} by its ends", yaw),
        Waypoint(1.5, (x, y, _z(CARRY_H)), grip, "lift", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: A1 (turned 90 degrees in the air), then A2 (already with the streaks: moved, not turned), then park."""
    a1 = L.TEMPLATES["template_a1"]
    a2 = L.TEMPLATES["template_a2"]
    x = L.ROW_X
    return [
        *_pick(a1[2], a1[3], YAW_A1, A_GRIP, "template A1", 2.5),
        Waypoint(3.0, (x, L.PLACE_Y["template_a1"], _z(CARRY_H)), A_GRIP,
                 "carry A1 over the sheet, turning it 90 degrees", YAW_X_LEFT),
        *_place(x, L.PLACE_Y["template_a1"], YAW_X_LEFT, A_GRIP, A_RELEASE, "A1"),
        *_pick(a2[2], a2[3], YAW_X_LEFT, A_GRIP, "template A2", 3.0),
        Waypoint(3.0, (x, L.PLACE_Y["template_a2"], _z(CARRY_H)), A_GRIP, "carry A2 over the sheet, not turning it",
                 YAW_X_LEFT),
        *_place(x, L.PLACE_Y["template_a2"], YAW_X_LEFT, A_GRIP, A_RELEASE, "A2"),
        Waypoint(3.0, (*LEFT_PARK, _z(PARK_H)), OPEN, "park, clear of the sheet", 0.0),
    ]


LEFT_DONE_S = _span(_left())  # the left arm is parked at y +0.30: the right arm may come over the sheet
RIGHT_LIFTED_S = 25.5  # the right arm has B lifted by then and holds it up until the left arm is parked


def _right() -> list[Waypoint]:
    """Right arm: over B, wait, pick B, hold it up until the left arm has parked, carry it to the row turning it 45
    degrees, lay it beside A2, park."""
    b = L.TEMPLATES["template_b"]
    pick = _pick(b[2], b[3], YAW_B, B_GRIP, "template B", 2.5)
    hover, rest = pick[0], pick[1:]
    wait = RIGHT_LIFTED_S - _span(pick)
    early = [hover, Waypoint(wait, hover.pos, OPEN, "wait over B", YAW_B), *rest]
    hold = LEFT_DONE_S - _span(early)
    x, y = L.ROW_X, L.PLACE_Y["template_b"]
    return [
        *early,
        Waypoint(hold, (b[2], b[3], _z(CARRY_H)), B_GRIP, "hold B up until the left arm has parked", YAW_B),
        Waypoint(3.0, (x, y, _z(CARRY_H)), B_GRIP, "carry B over the sheet, turning it 45 degrees", YAW_X_RIGHT),
        *_place(x, y, YAW_X_RIGHT, B_GRIP, B_RELEASE, "B"),
        Waypoint(3.0, (*RIGHT_PARK, _z(PARK_H)), OPEN, "park, clear of the sheet", 0.0),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
