"""On the station table: a charcoal-grey trough planter (fixed, 176 x 356 mm outside, 86 mm tall, open on top) on the
midline 27 cm in front of the arm bases, with two grey discs fixed on its floor at y +-11 cm, one place for each
plant. Two terracotta grow pots (70 mm across; pot A 35 mm tall on the left, pot B 46 mm tall on the right) stand on
the table outside the planter, and beside each a row of three white foam risers (60 mm across, 20, 30 and 42 mm
tall). Double potting: for each pot, stand a riser on a disc and the pot on the riser so that the pot's rim ends 10
to 20 mm below the planter's rim. Only one riser height does that for each pot (30 mm for pot A, 20 mm for pot B).

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
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # both pots staged (each set: still and let go) for 1 s

INSTRUCTION = (
    "Double-pot the two potted plants in the grey trough planter. For each plant, stand one of its white risers on "
    "a grey disc on the planter floor, then set the plant's pot on top of that riser so the pot's rim ends 10 to 20 "
    "mm below the planter's rim. The risers differ in height: choose the one that gives that gap. Leave both pots "
    "standing upright, one on each disc, then let go and move both grippers away from the pots."
)

# -- the scene ----------------------------------------------------------------------------------------------------
_WALL_Z = L.TOP + L.FLOOR_T + L.WALL_H / 2
FLOOR = Box("planter_floor", pos=(L.PLANTER_X, L.PLANTER_Y, L.TOP + L.FLOOR_T / 2),
            size=(L.OUTER_X, L.OUTER_Y, L.FLOOR_T), color=L.CHARCOAL)
WALLS = [
    Box("planter_wall_near", pos=(L.INNER_X0 - L.WALL / 2, L.PLANTER_Y, _WALL_Z), size=(L.WALL, L.OUTER_Y, L.WALL_H),
        color=L.CHARCOAL),
    Box("planter_wall_far", pos=(L.INNER_X1 + L.WALL / 2, L.PLANTER_Y, _WALL_Z), size=(L.WALL, L.OUTER_Y, L.WALL_H),
        color=L.CHARCOAL),
    Box("planter_wall_left", pos=(L.PLANTER_X, L.INNER_HALF_Y + L.WALL / 2, _WALL_Z),
        size=(L.INNER_X1 - L.INNER_X0, L.WALL, L.WALL_H), color=L.CHARCOAL),
    Box("planter_wall_right", pos=(L.PLANTER_X, -L.INNER_HALF_Y - L.WALL / 2, _WALL_Z),
        size=(L.INNER_X1 - L.INNER_X0, L.WALL, L.WALL_H), color=L.CHARCOAL),
]
DISCS = [Cylinder(name, pos=(x, y, L.TOP + L.FLOOR_TOP + L.DISC_T / 2), radius=L.DISC_R, height=L.DISC_T,
                  color=L.DISC_GREY) for name, (x, y) in L.DISCS.items()]
POTS = [Cylinder(name, pos=(x, y, L.TOP + h / 2), radius=L.POT_R, height=h, mass=m, color=L.TERRACOTTA)
        for name, (h, m, x, y) in L.POTS.items()]
RISERS = [Cylinder(name, pos=(x, y, L.TOP + h / 2), radius=L.RISER_R, height=h, mass=m, color=L.WHITE)
          for name, (h, m, x, y) in L.RISERS.items()]
OBJECTS = [FLOOR, *WALLS, *DISCS, *POTS, *RISERS]

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
    """Both pots staged: each upright within 3 degrees, its centre within 20 mm of a disc's, its bottom on a riser
    that stands upright on the planter floor or a disc, still over the last 0.5 s, both TCPs at least 60 mm from it,
    and its rim 10 to 20 mm below the planter's; held 1 s. This runs first every control step, so it also feeds the
    tracker (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for name in L.LOOSE:
        s = objects.get(name) if hasattr(objects, "get") else None
        if s is not None:
            row[f"{name}_pos"] = getattr(s, "pos", None)
            row[f"{name}_quat"] = getattr(s, "quat_xyzw", None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_opening=arms["left"].opening, right_opening=arms["right"].opening)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """A pot is set when it is upright (3 degrees), its centre within 20 mm of a disc's centre, its bottom face's
    centre within 1.5 mm (in height) and 35 mm (seen from above) of a riser's top face's centre, that riser upright
    (2 degrees) with its bottom face's centre 4 to 7.5 mm up (on the floor or a disc), the pot still (path under 3 mm
    over 0.5 s) and both TCPs at least 60 mm from its centre; staged when, set, its rim is 10 to 20 mm below the
    planter's rim. 1.0 both pots staged; 0.5 exactly one; 0.25 none staged but at least one set; 0 otherwise."""
    return _TRACKER.result(objects, rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; grips are 8 mm narrower than the part.
APPROACH = -0.040  # 80 mm: down beside a riser, 10 mm clear of it each side and 20 mm clear of its neighbours
RISER_GRIP = -(2 * L.RISER_R - 0.008) / 2  # across the riser's 60 mm: -0.026 (52 mm)
RISER_LET_GO = -0.038  # 76 mm: opened only so far inside the planter
POT_OPEN = -0.043  # 86 mm: 8 mm clear of a 70 mm pot each side
POT_GRIP = -(2 * L.POT_R - 0.008) / 2  # across the pot's 70 mm: -0.031 (62 mm)
YAW_L = -math.pi / 2  # both arms close their fingers along x, toward the planter's 160 mm inside length
YAW_R = math.pi / 2
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
GRIP_UP = {"riser": 0.006, "pot": 0.008}  # tips this far above the part's bottom while gripping it
RISER_PICK_Z = L.TOP + GRIP_UP["riser"] + PAD_TIPS  # TCP 19 mm up
POT_PICK_Z = L.TOP + GRIP_UP["pot"] + PAD_TIPS  # TCP 21 mm up
RISER_DROP = 0.005  # a riser is let go with its bottom 5 mm above the disc
RISER_PLACE_Z = L.TOP + L.DISC_TOP + RISER_DROP + GRIP_UP["riser"] + PAD_TIPS  # TCP 30 mm up
POT_DROP = 0.003  # a pot is let go with its bottom 3 mm above its riser's top
ENTRY = 0.030  # the last 30 mm of every descent are slow
RISER_CARRY_Z = L.TOP + 0.130  # a carried riser's bottom 111 mm up: 25 mm over the 86 mm rim
POT_CARRY_Z = L.TOP + 0.140  # a carried pot's bottom 119 mm up: 33 mm over the rim
PARK = (0.200, 0.250)  # each arm parks here (y mirrored), 14 cm up
RIGHT_LAG = 10.0  # the right arm waits this long over its riser, so only one arm at a time works in the planter

# Which riser each arm uses: the one that puts its pot's rim 10-20 mm below the planter's rim.
PLAN = {
    "left": {"yaw": YAW_L, "riser": "riser_l30", "pot": "pot_a", "disc": "disc_left", "sign": 1.0},
    "right": {"yaw": YAW_R, "riser": "riser_r20", "pot": "pot_b", "disc": "disc_right", "sign": -1.0},
}


def pot_place_z(arm: str) -> float:
    """TCP height at which this arm lets go of its pot: the pot's bottom 3 mm above its riser's top."""
    riser_top = L.DISC_TOP + L.RISERS[PLAN[arm]["riser"]][0]
    return L.TOP + riser_top + POT_DROP + GRIP_UP["pot"] + PAD_TIPS


def _arm(side: str) -> list[Waypoint]:
    """One arm: carry its riser into the planter and let it down onto its disc, then fetch its pot and lower it
    onto the riser, let go, go up and park. Both arms run this at once, each on its own half; the right arm first
    waits RIGHT_LAG (10 s) over its riser, so each of its visits to the planter starts after the left arm has left
    it (the arm-to-arm clearance is then 20 cm; with no wait both wrists are low over the discs together, 3.8 cm
    apart, and a wait of 2 to 8 s still leaves that 3.8 to 4.0 cm)."""
    p = PLAN[side]
    yaw, sign = p["yaw"], p["sign"]
    _, _, rx, ry = L.RISERS[p["riser"]]
    _, _, px, py = L.POTS[p["pot"]]
    sx, sy = L.DISCS[p["disc"]]
    place = pot_place_z(side)
    hi = L.TOP + 0.140
    lag = [Waypoint(RIGHT_LAG, (rx, ry, RISER_CARRY_Z), APPROACH, "wait for the left arm to work in the planter", yaw)]
    return [
        Waypoint(2.0, (rx, ry, RISER_CARRY_Z), APPROACH, "over the riser", yaw),
        *(lag if side == "right" else []),
        Waypoint(1.0, (rx, ry, RISER_PICK_Z + ENTRY), APPROACH, "down to 3 cm above the grip", yaw),
        Waypoint(1.5, (rx, ry, RISER_PICK_Z), APPROACH, "down slowly beside the riser", yaw),
        Waypoint(0.6, (rx, ry, RISER_PICK_Z), APPROACH, "settle", yaw),
        Waypoint(1.5, (rx, ry, RISER_PICK_Z), RISER_GRIP, "grip the riser", yaw),
        Waypoint(1.5, (rx, ry, RISER_CARRY_Z), RISER_GRIP, "lift the riser", yaw),
        Waypoint(2.5, (sx, sy, RISER_CARRY_Z), RISER_GRIP, "carry it over the disc", yaw),
        Waypoint(1.2, (sx, sy, RISER_PLACE_Z + ENTRY), RISER_GRIP, "down into the planter", yaw),
        Waypoint(1.5, (sx, sy, RISER_PLACE_Z), RISER_GRIP, "lower the last 3 cm slowly", yaw),
        Waypoint(0.6, (sx, sy, RISER_PLACE_Z), RISER_GRIP, "settle", yaw),
        Waypoint(1.5, (sx, sy, RISER_PLACE_Z), RISER_LET_GO, "let the riser down onto the disc", yaw),
        Waypoint(0.5, (sx, sy, RISER_PLACE_Z), RISER_LET_GO, "settle", yaw),
        Waypoint(1.5, (sx, sy, hi), RISER_LET_GO, "up out of the planter", yaw),
        Waypoint(2.5, (px, py, hi), POT_OPEN, "over the pot", yaw),
        Waypoint(1.5, (px, py, POT_PICK_Z + ENTRY), POT_OPEN, "down to 3 cm above the grip", yaw),
        Waypoint(1.5, (px, py, POT_PICK_Z), POT_OPEN, "down slowly beside the pot", yaw),
        Waypoint(0.6, (px, py, POT_PICK_Z), POT_OPEN, "settle", yaw),
        Waypoint(1.5, (px, py, POT_PICK_Z), POT_GRIP, "grip the pot", yaw),
        Waypoint(2.0, (px, py, POT_CARRY_Z), POT_GRIP, "lift the pot", yaw),
        Waypoint(2.5, (sx, sy, POT_CARRY_Z), POT_GRIP, "carry it over the riser", yaw),
        Waypoint(1.5, (sx, sy, place + ENTRY), POT_GRIP, "down into the planter", yaw),
        Waypoint(2.0, (sx, sy, place), POT_GRIP, "lower the last 3 cm slowly", yaw),
        Waypoint(0.8, (sx, sy, place), POT_GRIP, "settle", yaw),
        Waypoint(1.5, (sx, sy, place), POT_OPEN, "let go of the pot", yaw),
        Waypoint(0.5, (sx, sy, place), POT_OPEN, "settle", yaw),
        Waypoint(1.5, (sx, sy, hi), POT_OPEN, "straight up", yaw),
        Waypoint(2.0, (PARK[0], sign * PARK[1], hi), POT_OPEN, "park", yaw),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
