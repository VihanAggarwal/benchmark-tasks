"""On the station table, at 1:100: the deck of an 80 m power-driven cargo ship (a fixed grey plate 800 x 130 mm,
10 mm thick) lying along y in front of both arms, its pointed bow toward +y (the left arm) and its square stern
toward -y, a flush white centreline along its length and a white sternlight fitted at the stern. In front of it on
the table: two white masthead light units (upright cylinders 20 mm across, 20 mm tall) and a red and a green
sidelight unit (20 mm cubes). Rule (COLREG 1972 Rules 21 and 23, Annex I section 3, IMO MSC.1/Circ.1260/Rev.1):
two masthead lights over the centreline, the forward one not more than a quarter length (200 mm) from the stem,
the after one at least half a length (400 mm) abaft it; the sidelights not in front of the forward masthead
light, red at the port side (-x, near the arms) and green at the starboard side (+x), each with its outboard face
at most 1 m (10 mm) inboard of the side. Each correct light earns 0.25. The scorer reads the arms (pad-tip
points and openings) through rigstate.py only to check that each unit was let go.

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
# Every light must already be let go and still for 1 s to count. Should the arms ever be unreadable, the last unit
# set down would count 1 s after its slow descent ends while still gripped; settling and opening take 0.5 + 1.5 s,
# so with a 2 s hold on top the trial still cannot end while the fingers are closed on a unit.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "This is a 1:100 deck model of a new 80 m power-driven cargo ship. Its bow (the pointed end) points toward the "
    "left arm and its stern toward the right arm; the white sternlight is already fitted. Stand the two white "
    "masthead light units and the red and green sidelight units upright on the deck where COLREG 1972, the "
    "international rules of the road at sea, requires them for this ship, then let go. Only where each light "
    "stands on the deck is judged, not heights, screens or arcs."
)


def _box(name: str, x0: float, x1: float, y0: float, y1: float, color) -> Box:
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.DECK_H / 2), size=(x1 - x0, y1 - y0, L.DECK_H),
               color=color)


_LINE_X0, _LINE_X1 = L.CL_X - L.LINE_W / 2, L.CL_X + L.LINE_W / 2
DECK = [
    _box("deck_port", L.PORT_X, _LINE_X0, L.TRANSOM_Y, L.SHOULDER_Y, L.DECK_GREY),
    _box("deck_starboard", _LINE_X1, L.STBD_X, L.TRANSOM_Y, L.SHOULDER_Y, L.DECK_GREY),
    _box("centreline", _LINE_X0, _LINE_X1, L.TRANSOM_Y, L.STEM_Y, L.LINE_WHITE),
]
for _k, (_y0, _y1, _hw) in enumerate(L.bow_steps()):
    DECK.append(_box(f"bow_port_{_k}", L.CL_X - _hw, _LINE_X0, _y0, _y1, L.DECK_GREY))
    DECK.append(_box(f"bow_starboard_{_k}", _LINE_X1, L.CL_X + _hw, _y0, _y1, L.DECK_GREY))
STERNLIGHT = Cylinder("sternlight", pos=(L.STERN_X, L.STERN_Y, L.DECK_TOP + L.STERN_H / 2), radius=L.STERN_RADIUS,
                      height=L.STERN_H, color=L.UNIT_WHITE)


def _unit(name: str):
    x, y = L.START[name]
    if L.KIND[name] == "cyl":
        return Cylinder(name, pos=(x, y, L.TOP + L.MAST_H / 2), radius=L.MAST_RADIUS, height=L.MAST_H,
                        mass=L.MAST_MASS, color=L.UNIT_WHITE)
    return Box(name, pos=(x, y, L.TOP + L.SIDE / 2), size=(L.SIDE, L.SIDE, L.SIDE), mass=L.SIDE_MASS,
               color=L.LAMP_RED if name == L.RED else L.LAMP_GREEN)


UNITS = [_unit(n) for n in L.NAMES]
OBJECTS = [*DECK, STERNLIGHT, *UNITS]
UNIT_NAMES = tuple(u.name for u in UNITS)

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the units teleporting back)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace
    that cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """1.0 now: F over the centreline within a quarter length of the stem, A over the centreline at least half a
    length abaft F, red and green at their sides not ahead of F, every unit standing (upright, on the deck, let go,
    still for 1 s); held SUCCESS_HOLD_S = 2 s. This runs first every control step, so it also feeds the tracker
    (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    units = [objects[n] for n in UNIT_NAMES]
    result = _TRACKER.update(units, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if isinstance(arms, dict):
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each correct light: F (the standing white unit nearer the bow) within 6 mm of the centreline and at
    most 200 mm from the stem; and, only when F is within 6 mm of the centreline, A within 6 mm of it at least
    400 mm abaft F, red at the port side and green at the starboard side (outboard face at most 10 mm inboard,
    centre not ahead of F). Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in UNIT_NAMES], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.020  # 40 mm: 10 mm either side of a 20 mm unit; neighbours on the table stand 60 mm apart
GRIP = -(L.SIDE - 0.008) / 2  # 12 mm commanded across a 20 mm unit (both kinds are 20 mm across)
PICK_YAW = 0.0  # fingers close along y: square on a cube's faces, never outboard of the deck's sides
PLACE_YAW = math.radians(-1.15)  # gripped parts end about 0.6 to 1.7 degrees CCW of the command
GRIP_UP = 0.017  # the TCP this far above a unit's bottom: the pad tips 4 mm above it
DROP = 0.001  # each unit is let go with its bottom 1 mm above the deck
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.DECK_TOP + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a unit, or onto the deck, are taken slowly
HOVER_Z = L.TOP + 0.070  # travel: a carried unit's bottom 53 mm up, 23 mm over a standing unit's top
APPROACH = 0.040  # each unit's last 40 mm of travel runs along y only, so arm lag lands it short in y, not in x
# The left arm parks back toward its own base, high, so the right arm fetching the green cube from y +0.030 keeps
# its links 5.5 cm or more from it (parked lower at x 0.20, y +0.25 it came within 4.5 cm).
PARK = {"left": (0.250, 0.300, L.TOP + 0.150), "right": (0.200, -0.250, L.TOP + 0.120)}
# Targets (unit centres). F: on the centreline 120 mm (12 m) from the stem; A: on the centreline 460 mm (46 m)
# abaft F; the sidelights abreast 360 mm abaft F, where a cargo ship's bridge wings stand, each with its outboard
# face 5 mm (0.5 m) inboard of its side, the middle of the 10 mm band.
F_AT = (L.CL_X, 0.280)
A_AT = (L.CL_X, -0.180)
RED_AT = (L.PORT_X + L.SIDE_BAND / 2 + L.SIDE / 2, -0.080)  # x 0.300
GREEN_AT = (L.STBD_X - L.SIDE_BAND / 2 - L.SIDE / 2, -0.080)  # x 0.400
# (unit, target, direction of the last 40 mm along y). The left arm (+y) sets F and parks; the right arm (-y) sets
# A, then red, then green (from y +0.030, after the left arm has parked at y +0.250).
PLAN = {
    "left": (("masthead_1", F_AT, +1.0),),
    "right": (("masthead_2", A_AT, -1.0), (L.RED, RED_AT, -1.0), (L.GREEN, GREEN_AT, -1.0)),
}
# Per-arm trim of the place points (x, y), for the GPU run: if one arm's units land off their targets by the same
# amount, read the units' x_mm / y_mm in TASK_TRACE and move that arm's place points by minus the offset.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _move(name: str, target, along: float, arm: str, first: bool) -> list[Waypoint]:
    """Over a unit, slowly down round it, grip it, lift it, carry it to 40 mm short of its place (in y), along y over
    its place, settle, set it down slowly on the deck, let go, straight up."""
    x, y = L.START[name]
    tx, ty = PLACE_TRIM[arm]
    px, py = target[0] + tx, target[1] + ty
    ay = py - along * APPROACH
    return [
        Waypoint(2.5 if first else 1.8, (x, y, HOVER_Z), OPEN, f"over {name}", PICK_YAW),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", PICK_YAW),
        Waypoint(1.2, (x, y, PICK_Z), OPEN, "slowly down round it", PICK_YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", PICK_YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", PICK_YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", PICK_YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", PICK_YAW),
        Waypoint(2.0, (px, ay, HOVER_Z), GRIP, "carry it to 40 mm short of its place", PLACE_YAW),
        Waypoint(1.2, (px, py, HOVER_Z), GRIP, "along y over its place", PLACE_YAW),
        Waypoint(0.5, (px, py, HOVER_Z), GRIP, "settle over its place", PLACE_YAW),
        Waypoint(1.0, (px, py, PLACE_Z + SLOW), GRIP, "down to just above the deck", PLACE_YAW),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly stand it on the deck", PLACE_YAW),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", PLACE_YAW),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", PLACE_YAW),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", PLACE_YAW),
        Waypoint(1.0, (px, py, HOVER_Z), OPEN, "straight up, clear of it", PLACE_YAW),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, (name, target, along) in enumerate(PLAN[arm]):
        wps += _move(name, target, along, arm, k == 0)
    return wps + [Waypoint(1.5, PARK[arm], OPEN, "park, clear of the deck", PICK_YAW)]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
