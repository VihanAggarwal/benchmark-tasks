"""On the station table: a 1:25 model of a ground slab ready to pour. An L-shaped slab form of dark plywood boards
(15 mm tall) on a thin stone sub-base: a long leg 120 x 460 mm (x 26-38 cm, y -24 to +22 cm) and a return leg
120 x 100 mm (x 14-26 cm, y +12 to +22 cm) toward the arms on its left end; the inside corner is at (26, +12) cm.
Four loose dark grey joint strips (18 mm wide, 12 mm thick) lie flat on the table beside it, along x: a 112 mm and the
83 mm one on the left, two 112 mm ones on the right. Rule (contraction joints): lay the strips so that every panel
they mark out is a rectangle no longer than 132 mm (3.3 m) and no more than 1.5 times as long as it is wide; then let
go. That forces one strip out of the inside corner across the long leg (y +12 cm), the short strip between the legs
from it to the left form (x 26 cm), and two across the rest of the long leg (near y 0 and -12 cm): three 120 mm
squares and two 120 x 100 mm panels.

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
# Still for 1 s: five good panels, every strip let go (finger zones 20 mm clear), no strip moving.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Lay the four grey joint strips flat on the sub-base inside the L-shaped slab form, each running square to the "
    "forms with both ends against a form or another strip, so that the joints divide the slab into panels that are "
    "all rectangles no longer than 132 mm and no more than 1.5 times as long as they are wide. Then let go of them."
)

FORMS = [
    Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.FORM_H / 2), size=(x1 - x0, y1 - y0, L.FORM_H),
        color=L.FORM_PLY)
    for name, x0, x1, y0, y1 in L.FORMS
]
SUB_BASE = [
    Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.SUBBASE_H / 2), size=(x1 - x0, y1 - y0, L.SUBBASE_H),
        color=L.STONE)
    for name, x0, x1, y0, y1 in L.SUBBASES
]
STRIPS = [
    Box(name, pos=(x, y, L.TOP + L.STRIP_T / 2), size=(length, L.STRIP_W, L.STRIP_T), mass=mass,
        color=L.STRIP_GREY)
    for name, (length, mass, x, y) in L.STRIPS.items()
]
OBJECTS = [*FORMS, *SUB_BASE, *STRIPS]

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


def _strips(objects) -> dict:
    return {n: objects.get(n) for n in L.STRIP_NAMES}


def success(objects) -> bool:
    """Five good panels (every panel the counted strips mark out a rectangle, its longer side <= 132 mm and <= 1.5
    times its shorter side), every strip that counts let go; no strip's end moved more than 2 mm in 0.5 s; held 1 s.
    This runs first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    strips = _strips(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(strips, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for n, s in strips.items():
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
    """0.2 per good panel (a rectangle, longer side <= 132 mm and <= 1.5 x its shorter side) that the counted strips'
    centre lines and the forms' inside faces mark out; at most 1.0. A strip counts when it lies flat on the sub-base,
    square to x or y within 5 deg, inside the slab, let go, each end within 8 mm of a form or another counted strip."""
    return _TRACKER.result(_strips(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
PICK_OPEN = -0.020  # 40 mm open: 11 mm each side of an 18 mm strip
GRIP = -(L.STRIP_W - 0.008) / 2  # 10 mm commanded across the strip's 18 mm width
RELEASE = PICK_OPEN
# Yaw 0 closes the fingers along y (across a strip lying along x); the left arm turns to -90 deg to close them along
# x (across a strip lying along y), the sign that keeps its joint 6 inside its limits.
YAW_ALONG_X, YAW_ALONG_Y = 0.0, -math.pi / 2
GRIP_UP = 0.016  # the TCP this far above a strip's underside: pad tips 3 mm above it, on its lower 3-12 mm
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + L.SUBBASE_H + 0.0005 + GRIP_UP  # let go with the strip's underside 0.5 mm above the sub-base
ENTRY_Z = PLACE_Z + 0.020  # the last 20 mm down are slow
CARRY_Z = L.TOP + 0.090  # a carried strip's underside 74 mm up, 59 mm over the forms
CLEAR_Z = L.TOP + 0.100  # after letting go: pad tips 74 mm over the strip's top (the 40 mm finger zone 38 mm clear)
L_PARK = (0.200, 0.300, L.TOP + 0.150)
R_PARK = (0.200, -0.300, L.TOP + 0.150)

# Where the strips go: their centres. The cross strips sit mid-way across the long leg (4 mm from each form face).
MID_X = (L.LONG[0] + L.LONG[1]) / 2  # 0.320
J1 = (MID_X, L.CORNER[1])  # out of the inside corner, along x on y 0.120
J2_GAP = (L.FACES_Y[0][0] - (L.CORNER[1] + L.STRIP_W / 2) - L.SHORT_LEN) / 2  # 4 mm at each end
J2 = (L.CORNER[0], L.CORNER[1] + L.STRIP_W / 2 + J2_GAP + L.SHORT_LEN / 2)  # (0.260, 0.1745): between the legs
J3 = (MID_X, 0.000)  # across the long leg: three 120 mm panels from J1 to the right end
J4 = (MID_X, -0.120)
PICK_S = 6.7  # one pick (over, down, settle, grip, lift)
PLACE_S = 8.7  # one place (carry, down, settle, slow down, let go, up)


def _start(name: str) -> tuple[float, float]:
    return L.STRIPS[name][2], L.STRIPS[name][3]


def _pick(name: str, label: str, yaw: float) -> list[Waypoint]:
    """Come down round a strip lying along x on the table and close across its 18 mm width; lift it over the forms."""
    sx, sy = _start(name)
    return [
        Waypoint(2.0, (sx, sy, CARRY_Z), PICK_OPEN, f"over {label}", yaw),
        Waypoint(1.5, (sx, sy, PICK_Z), PICK_OPEN, f"down round {label}", yaw),
        Waypoint(0.5, (sx, sy, PICK_Z), PICK_OPEN, "settle", yaw),
        Waypoint(1.5, (sx, sy, PICK_Z), GRIP, f"grip {label}", yaw),
        Waypoint(1.2, (sx, sy, CARRY_Z), GRIP, f"lift {label} over the forms", yaw),
    ]


def _place(at: tuple[float, float], label: str, joint: str, yaw: float) -> list[Waypoint]:
    """Carry a strip over its line, lower it slowly onto the sub-base, let go, rise straight up."""
    px, py = at
    return [
        Waypoint(2.5, (px, py, CARRY_Z), GRIP, f"carry {label} over {joint}", yaw),
        Waypoint(1.0, (px, py, ENTRY_Z), GRIP, "down to 2 cm above the sub-base", yaw),
        Waypoint(0.5, (px, py, ENTRY_Z), GRIP, "settle", yaw),
        Waypoint(2.0, (px, py, PLACE_Z), GRIP, f"slowly down: {label} onto the sub-base", yaw),
        Waypoint(1.5, (px, py, PLACE_Z), RELEASE, f"let go of {label}", yaw),
        Waypoint(1.2, (px, py, CLEAR_Z), RELEASE, "straight up", yaw),
    ]


def _left() -> list[Waypoint]:
    """Left arm: its long strip out of the inside corner (J1), while the right arm lays J4; then it picks the short
    strip and holds it high while the right arm lays J3 near the midline; then lays it between the legs (J2), turned
    along y, butting J1 and the left form; parks."""
    sx, sy = _start("strip_left_short")
    wait = PLACE_S  # the right arm picks its second strip with the left's, then places it: it is up from J3 after this
    return (
        _pick("strip_left_long", "the left long strip", YAW_ALONG_X)
        + _place(J1, "the left long strip", "the line out of the inside corner", YAW_ALONG_X)
        + _pick("strip_left_short", "the short strip", YAW_ALONG_X)
        + [Waypoint(wait, (sx, sy, CARRY_Z), GRIP, "hold it while the right arm lays its second strip",
                    YAW_ALONG_X)]
        + _place(J2, "the short strip", "the line between the legs, turning it along y", YAW_ALONG_Y)
        + [Waypoint(1.5, L_PARK, RELEASE, "park over its own side", YAW_ALONG_Y)]
    )


def _right() -> list[Waypoint]:
    """Right arm: its inner strip across the long leg 120 mm from the right end (J4), while the left arm lays J1;
    then its outer strip across the long leg on the midline (J3); parks."""
    return (
        _pick("strip_right_inner", "the right inner strip", YAW_ALONG_X)
        + _place(J4, "the right inner strip", "the line 120 mm from the right end", YAW_ALONG_X)
        + _pick("strip_right_outer", "the right outer strip", YAW_ALONG_X)
        + _place(J3, "the right outer strip", "the line on the midline", YAW_ALONG_X)
        + [Waypoint(1.5, R_PARK, RELEASE, "park over its own side", YAW_ALONG_X)]
    )


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
