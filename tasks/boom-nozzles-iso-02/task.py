"""On the station table: a bench section of a crop sprayer's boom (fixed aluminium bar, 44 x 680 x 25 mm) along y,
32 cm in front of the arm bases, with six black nozzle bodies on it (square sockets, 32 mm inside, 8 mm tall),
125 mm apart; two white parts trays and two grey spares pads nearer the arms, one of each on each side. Thirteen
loose nozzles (acetal cylinders 22 mm across, 32 mm tall, 17 g) in ISO 10625 colours: at the start the bodies hold,
from the left, yellow, blue, nothing, lilac, yellow, blue, and each pad holds two yellows and two other sizes.
Task: every body holds one 02 (yellow) nozzle standing upright; the two fitted 02s stay; wrong sizes come off into
a tray; nothing is left lying anywhere else.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s: the last nozzle counts as fitted once it stands on the boom's top and has
# moved under 2 mm in 0.5 s, which is while the gripper still holds it; 2.5 s from then is past the end of the
# 1.5 s opening and 0.3 s settle, so the demo's trial ends with the fingers open and lifting off the nozzle.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "The job card calls for size 02 flat-fan nozzles. Make each of the six black nozzle bodies on the sprayer "
    "boom hold one 02 nozzle standing upright in it, telling the sizes apart by the ISO colour code. The 02s "
    "already fitted can stay. Put every nozzle of another size that you take off the boom into one of the two "
    "white parts trays, and fit 02s from the spares on the grey pads. Leave no nozzle anywhere else."
)

BOOM = Box("boom", pos=(L.BOOM_X, L.BOOM_Y, L.TOP + L.BOOM_SIZE[2] / 2), size=L.BOOM_SIZE, color=L.BOOM_GREY)
WALLS = [Box(name, pos=(x, y, L.BOOM_TOP + L.BODY_WALL_H / 2), size=(sx, sy, L.BODY_WALL_H), color=L.BODY_BLACK)
         for b in L.BODY_NAMES for name, x, y, sx, sy in L.body_walls(b)]
TRAYS = [Box(name, pos=(x, y, z), size=(sx, sy, sz), color=L.TRAY_WHITE)
         for side in L.TRAYS for name, x, y, z, sx, sy, sz in L.tray_parts(side)]
PADS = [Box(f"spares_pad_{side}", pos=(L.PAD_X, y, L.TOP + L.PAD_SIZE[2] / 2), size=L.PAD_SIZE, color=L.PAD_GREY)
        for side, y in L.PADS.items()]
NOZZLES = [Cylinder(name, pos=(x, y, L.start_bottom(name) + L.NOZZLE_H / 2), radius=L.NOZZLE_R,
                    height=L.NOZZLE_H, mass=L.NOZZLE_MASS, color=L.RGB[colour])
           for name, colour, x, y, _ in L.NOZZLES]
OBJECTS = [BOOM, *WALLS, *TRAYS, *PADS, *NOZZLES]

_TRACKER = Tracker()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _nozzles(objects) -> dict:
    return {n: objects[n] for n in L.NOZZLE_NAMES if n in objects}


def success(objects) -> bool:
    """Every body holds one fitted 02 nozzle (upright on the boom's top, still), no wrong size in a body, every
    other nozzle in a tray or on a pad (blue and lilac not on a pad); held for 2.5 s. This runs first every
    control step, so it also feeds the tracker."""
    states = _nozzles(objects)
    result = _TRACKER.update(states)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    for n, s in states.items():
        row[f"{n}_pos"] = getattr(s, "pos", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any nozzle is off the table, any body holds two nozzles, or L3 or R2 does not hold a fitted 02. Else
    0.25 for each of L2, L1, R1 and R3 holding a fitted 02, at most 0.5 while any body holds another size and at
    most 0.75 while any nozzle is outside the bodies, trays and pads (or a blue or lilac one is on a pad)."""
    return _TRACKER.result(_nozzles(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
APPROACH = -0.021  # 42 mm apart: 10 mm each side of a nozzle, clear of the neighbours on the pads (55-60 mm)
GRIP = -(2 * L.NOZZLE_R - 0.008) / 2  # 14 mm commanded across the 22 mm nozzle
YAW = 0.0  # every grip closes the fingers along y: along the boom the next body is 125 mm away
GRIP_UP = 0.028  # the TCP this far above a nozzle's bottom: pad tips 15 mm up, 7 mm above a body's walls
CARRY_Z = L.TOP + 0.13  # a carried nozzle's bottom 102 mm above the table, 45 mm over the fitted nozzles' tops
PARK_Z = L.TOP + 0.15
ENTRY = 0.020  # the last 20 mm down are taken slowly
TRAY_SPOTS = {  # where each removed nozzle is stood in its tray (tray centres x 0.200, y +-0.110)
    "blue_L2": (L.TRAY_X, L.TRAYS["left"]),
    "blue_R3": (L.TRAY_X + 0.020, L.TRAYS["right"]),
    "lilac_R1": (L.TRAY_X - 0.020, L.TRAYS["right"]),
}
# The right arm comes over R3's blue at 9.2 s, as the left arm arrives over its tray with L2's blue; from then on
# each right move runs 7.2 s (about half a move) behind the matching left one.
RIGHT_DELAY = 8.2
LEFT_PARK = (0.20, 0.27, PARK_Z)
RIGHT_PARK = (0.20, -0.27, PARK_Z)


def _pick(name: str, approach: float = 2.0) -> list[Waypoint]:
    """Over a nozzle, slowly down round it, grip it on its upper half, lift it straight up."""
    x, y = L.STARTS[name]
    z = L.start_bottom(name) + GRIP_UP
    return [
        Waypoint(approach, (x, y, CARRY_Z), APPROACH, f"over {name}", YAW),
        Waypoint(1.2, (x, y, z + ENTRY), APPROACH, "down to 2 cm above", YAW),
        Waypoint(1.0, (x, y, z), APPROACH, "slowly down round the nozzle", YAW),
        Waypoint(0.5, (x, y, z), APPROACH, "settle", YAW),
        Waypoint(1.5, (x, y, z), GRIP, f"grip {name}", YAW),
        Waypoint(1.2, (x, y, CARRY_Z), GRIP, "lift it straight up", YAW),
    ]


def _place(x: float, y: float, bottom: float, label: str) -> list[Waypoint]:
    """Carry the nozzle high over (x, y), stop, lower the last 2 cm slowly with its bottom ending 1-3 mm above
    ``bottom``, let go, go straight up."""
    z = bottom + GRIP_UP
    return [
        Waypoint(1.8, (x, y, CARRY_Z), GRIP, f"carry over {label}", YAW),
        Waypoint(1.2, (x, y, z + ENTRY), GRIP, "down to 2 cm above", YAW),
        Waypoint(0.5, (x, y, z + ENTRY), GRIP, "settle", YAW),
        Waypoint(1.5, (x, y, z), GRIP, "lower the last 2 cm slowly", YAW),
        Waypoint(1.5, (x, y, z), APPROACH, "let go", YAW),
        Waypoint(0.3, (x, y, z), APPROACH, "settle", YAW),
        Waypoint(1.0, (x, y, CARRY_Z), APPROACH, "straight up", YAW),
    ]


def _to_body(name: str, body: str, approach: float = 2.0) -> list[Waypoint]:
    return _pick(name, approach) + _place(L.BOOM_X, L.BODIES[body], L.BOOM_TOP + 0.001, f"body {body}")


def _to_tray(name: str, approach: float = 2.0) -> list[Waypoint]:
    x, y = TRAY_SPOTS[name]
    return _pick(name, approach) + _place(x, y, L.TRAY_FLOOR_TOP + 0.003, "the parts tray")


def _duration(plan: list[Waypoint]) -> float:
    return sum(w.duration for w in plan)


def _plans() -> dict[str, list[Waypoint]]:
    """Both arms on one clock, each working its own half of the boom. Every move takes the same 15.2 s and the
    right arm runs 7.2 s (about half a move) behind the left, so while one gripper is low near the middle (a tray,
    L1 or R1) the other is low only on its own outer side (its pad, L2/L3 or R2/R3) or is up:
    left:  L2's blue to the left tray; a yellow to L2; a yellow to L1; park over the left pad;
    right: (wait) R3's blue to the right tray; R1's lilac to the right tray; a yellow to R3; a yellow to R1."""
    left = (_to_tray("blue_L2") + _to_body("yellow_left_b", "L2") + _to_body("yellow_left_a", "L1")
            + [Waypoint(2.0, LEFT_PARK, APPROACH, "park over the left pad", YAW)])
    wait = (L.BOOM_X, L.BODIES["R3"], PARK_Z)
    right = ([Waypoint(2.0, wait, APPROACH, "over R3, waiting", YAW),
              Waypoint(RIGHT_DELAY - 2.0, wait, APPROACH, "wait for the left arm's first pick", YAW)]
             + _to_tray("blue_R3", approach=1.0) + _to_tray("lilac_R1") + _to_body("yellow_right_b", "R3")
             + _to_body("yellow_right_a", "R1")
             + [Waypoint(2.0, RIGHT_PARK, APPROACH, "park over the right pad", YAW)])
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = _plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
