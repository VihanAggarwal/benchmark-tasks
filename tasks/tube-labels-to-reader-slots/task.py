"""At a clinical or research laboratory's analyser: a dark grey five-position analyser sample rack, its positions
24 x 24 mm at 28 mm pitch, held in place on the midline 30 cm in front of the arm bases. In the rack's wall nearest
the arms, beside each position, is an 18 mm reader slot: the analyser's barcode reader reads the tube in that position
through it. In front of each arm is a light grey bench tube rack with two cells, and in them stand four capped blood
collection tubes (16 x 100 mm, lavender closures, white barcode labels), every label facing sideways (two to the
left, two to the right), a quarter turn away from the reader. Rule: rack the four tubes, one per position, each
standing with its label facing out through its position's slot (within 30 degrees), and let go; a tube whose label
is turned away is a no-read and earns nothing. Neighbouring tubes in the rack stand 12 mm apart, so a tube set down
next to one already standing has to be held with the fingers across the rack (front and back), and the grip chosen
at the pick must leave the label facing the slot after the hand's quarter turn.

Each tube is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of the tube's body, its
closure and its label, so the label can be seen and the scorer can read which way it faces. rigstate.py (next to
this file) gives the scorer both grippers' positions, for the let-go rule.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the tubes are built as compound bodies)
import rigstate  # noqa: E402  (both grippers' state for the let-go rule; fed by the offline checker too)
from scoring import Tracker, points  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# A tube counts only once both grippers are at least 20 mm clear of it (the let-go rule), so success cannot start
# while a gripper still holds the last tube; the default 1 s hold is enough.
SUCCESS_HOLD_S = 1.0

rigstate.install()  # wraps the embodiment's step and reset (a no-op offline, where the checker feeds the arms)

INSTRUCTION = (
    "Load the four blood tubes from the two bench racks into the analyser's sample rack, one tube per position. Stand "
    "each tube upright with its white barcode label facing out through its position's reader slot, in the rack wall "
    "nearest you, and let go of it. A label turned away from its slot cannot be read."
)


def _span(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


def _analyser_rack() -> list[Box]:
    """Fixed walls: the far wall, the end walls (6 mm) and four 4 mm dividers, and the near wall with an 18 mm
    reader slot centred on each position (a 6 mm rail under the slots and six posts between and beside them)."""
    h, t, half, d = L.RACK_H, L.WALL_T, L.RACK_HALF_Y, L.DIV_T
    c = L.RACK_GREY
    walls = [
        _span("rack_far_wall", L.FAR_X0, L.FAR_X1, -half, half, 0.0, h, c),
        _span("rack_end_left", L.NEAR_X1, L.FAR_X0, half - t, half, 0.0, h, c),
        _span("rack_end_right", L.NEAR_X1, L.FAR_X0, -half, -half + t, 0.0, h, c),
        _span("rack_slot_rail", L.NEAR_X0, L.NEAR_X1, -half, half, 0.0, L.SLOT_Z0, c),
    ]
    for k in range(1, L.N_POS):  # dividers between P1|P2 .. P4|P5
        yd = (L.POS_Y[k - 1] + L.POS_Y[k]) / 2
        walls.append(_span(f"rack_divider_{k}{k + 1}", L.NEAR_X1, L.FAR_X0, yd - d / 2, yd + d / 2, 0.0, h, c))
    edges = [half] + [e for py in L.POS_Y for e in (py + L.SLOT_W / 2, py - L.SLOT_W / 2)] + [-half]
    for i in range(0, len(edges), 2):  # the near wall's posts between the slots' edges
        y_hi, y_lo = edges[i], edges[i + 1]
        walls.append(_span(f"rack_post_{i // 2}", L.NEAR_X0, L.NEAR_X1, y_lo, y_hi, L.SLOT_Z0, h, c))
    return walls


def _holding_rack(side: str, yc: float) -> list[Box]:
    """Two 30 x 30 mm cells along x, 20 mm walls: two long side walls and three cross walls."""
    h, t, half = L.HOLD_H, L.HOLD_WALL, L.HOLD_IN / 2
    c = L.HOLD_GREY
    x0, x1 = L.HOLD_X0, L.HOLD_X1
    (ca, cb) = L.HOLD_X
    return [
        _span(f"hold_{side}_side_a", x0, x1, yc + half, yc + half + t, 0.0, h, c),
        _span(f"hold_{side}_side_b", x0, x1, yc - half - t, yc - half, 0.0, h, c),
        _span(f"hold_{side}_front", x0, ca - half, yc - half, yc + half, 0.0, h, c),
        _span(f"hold_{side}_middle", ca + half, cb - half, yc - half, yc + half, 0.0, h, c),
        _span(f"hold_{side}_back", cb + half, x1, yc - half, yc + half, 0.0, h, c),
    ]


def _tube(name: str):
    x, y = L.TUBE_START[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP),
        yaw=L.TUBE_START_YAW[name],
        mass=L.TUBE_MASS,
        color=L.BLOOD,
        shapes=[
            mk.cylinder(L.BODY_CENTER, L.TUBE_R, L.BODY_H),
            mk.cylinder(L.CAP_CENTER, L.CAP_R, L.CAP_H, color=L.CAP_LAVENDER),
            mk.box(L.LABEL_CENTER, L.LABEL_SIZE, color=L.LABEL_WHITE),
        ],
    )


RACK = _analyser_rack()
HOLDING = _holding_rack("left", +L.HOLD_Y) + _holding_rack("right", -L.HOLD_Y)
TUBES = [_tube(n) for n in L.TUBES]
OBJECTS = [*RACK, *HOLDING, *TUBES]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _tubes(objects) -> dict:
    return {n: objects.get(n) for n in L.TUBES}


def success(objects) -> bool:
    """All four tubes readable: each standing upright (within 5 degrees) on the floor of its own analyser rack
    position, its label within 30 degrees of facing out through that position's reader slot, every tube at rest and
    both grippers at least 20 mm clear of it; held 1 s. This runs first every control step, so it also feeds the
    tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    tubes = _tubes(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(tubes, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["source"] = getattr(tubes[L.TUBES[0]], "source", None)
        row["origins"] = {n: tubes[n].origin for n in L.TUBES}
        row["quats"] = {n: tubes[n].quat_xyzw for n in L.TUBES}
        row["label_faces"] = {n: points(tubes[n])[2] for n in L.TUBES}
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.25 for each readable tube: standing upright (within 5 degrees) with its bottom's centre inside an analyser
    rack position and within 5 mm of the table, alone in that position, at rest, both grippers at least 20 mm clear
    of it, its label facing within 30 degrees of straight out through the position's reader slot (-x). Read from the
    state at hand (the last step)."""
    return _TRACKER.result(_tubes(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the 16 mm tube.
PICK_OPEN = -0.022  # 44 mm: 14 mm clear of the tube on each side, and the pads stay clear of the next cell's tube
PLACE_OPEN = -0.018  # let go to 36 mm only: the pads 10 mm off the tube. Next to a standing tube the fingers are
#                      across the rack (along x), so the pads open toward the front and back walls, below them
#                      nothing (their tips stay 19 mm over the walls), and sideways they reach 10 mm from the axis,
#                      10 mm short of the neighbour 28 mm away
GRIP = -(2 * L.TUBE_R - 0.008) / 2  # 8 mm commanded across the 16 mm tube: about 20 N per finger
# Each tube is gripped with the tool point 70 mm above its bottom: the pads (22 mm long, from 9 mm above the tool point
# to their tips 13 mm below it) hold the bare tube between 57 and 79 mm up, 7 mm above the label's top (50 mm) and
# 3 mm below the cap (82 mm), so the grip never touches the label, whichever way the fingers close.
GRIP_UP = 0.070
DROP = 0.002  # each tube is let go with its bottom 2 mm above the rack's floor (the table)
PICK_APPROACH = 0.030  # the final descent to a grip starts this far above it (pad tips 87 mm up, over the cap)
PLACE_APPROACH = 0.045  # ... and to a place: the tube's bottom 47 mm up, still over the rack's 40 mm walls
TRAVEL = L.TOP + 0.130  # carried with the tube's bottom 60 mm up: 20 mm over the analyser rack's walls
# The quarter turn is made with the wrist while the tube is carried: the grip's yaw at the pick and at the place differ
# by exactly +-90 degrees. Yaw 0 closes the fingers along y. Near the midline the left arm needs negative yaws and the
# right arm positive ones (joint 6 has +-120 degrees), so a tube whose label starts on the arm's outer side (tube_1,
# tube_3) is picked with the grip turned 120 degrees and set down at 30 degrees, and one whose label starts toward the
# midline (tube_2, tube_4) is picked at 0 and set down at 90 (fingers along x, opening toward the walls in front and
# behind, not toward the next tube). Positions are 28 mm apart: a tube going in beside one already standing must be
# held across the rack (a grip along the row would need pads in the 12 mm between neighbouring tubes), which fixes
# the grip's yaw at the place, and therefore at the pick too (a quarter turn from it, the right way round).
# (tube, rack position, yaw at the pick, yaw at the place); each arm racks the inner position first, so a tube is
# never carried past one already standing in the rack, and the 30-degree release (fingers opening partly along y)
# happens with both neighbouring positions still empty; the outer tube goes in beside the inner one, fingers across.
PLAN = {
    "left": (("tube_1", 2, math.radians(-120.0), math.radians(-30.0)),
             ("tube_2", 1, 0.0, math.radians(-90.0))),
    "right": (("tube_3", 4, math.radians(120.0), math.radians(30.0)),
              ("tube_4", 5, 0.0, math.radians(90.0))),
}
ORDER = (("left", 0), ("right", 0), ("left", 1), ("right", 1))  # whose turn it is at the analyser rack
RACK_GAP = 1.0  # an arm sets off toward the rack this long after the other has risen from it (and is moving away)
PARK = {"left": (0.230, 0.200, L.TOP + 0.150), "right": (0.230, -0.200, L.TOP + 0.150)}


def _pick(tube: str, yaw: float) -> list[Waypoint]:
    """Over the tube with the fingers at 44 mm, down, settle, grip, lift it out of its cell."""
    x, y = L.TUBE_START[tube]
    z = L.TOP + GRIP_UP
    return [
        Waypoint(2.0, (x, y, TRAVEL), PICK_OPEN, f"over {tube}", yaw),
        Waypoint(1.0, (x, y, z + PICK_APPROACH), PICK_OPEN, f"down to just above the grip on {tube}", yaw),
        Waypoint(1.5, (x, y, z), PICK_OPEN, f"slowly down around {tube}", yaw),
        Waypoint(0.5, (x, y, z), PICK_OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, z), GRIP, f"grip {tube}", yaw),
        Waypoint(1.5, (x, y, TRAVEL), GRIP, f"lift {tube} out of its cell", yaw),
    ]


def _place(tube: str, k: int, yaw: float) -> list[Waypoint]:
    """Carry the tube over rack position k turning it a quarter turn, lower it slowly, let go, rise."""
    x, y = L.RACK_X, L.POS_Y[k - 1]
    z = L.TOP + GRIP_UP + DROP
    return [
        Waypoint(2.5, (x, y, TRAVEL), GRIP, f"carry {tube} over P{k}, turning its label toward the slot", yaw),
        Waypoint(0.5, (x, y, TRAVEL), GRIP, "settle over the rack", yaw),
        Waypoint(1.0, (x, y, z + PLACE_APPROACH), GRIP, f"down until {tube} is just over P{k}'s walls", yaw),
        Waypoint(2.5, (x, y, z), GRIP, f"slowly stand {tube} in P{k}", yaw),
        Waypoint(0.5, (x, y, z), GRIP, "settle", yaw),
        Waypoint(1.5, (x, y, z), PLACE_OPEN, f"let go of {tube}", yaw),
        Waypoint(1.0, (x, y, TRAVEL), PLACE_OPEN, "straight up", yaw),
    ]


def _duration(wps) -> float:
    return sum(w.duration for w in wps)


def schedule() -> dict:
    """When each arm sets off from over its bench rack toward the analyser rack with each of its tubes (s from the
    start): as soon as the tube is picked, but not before the other arm has risen from the rack and RACK_GAP has
    passed, so only one gripper is ever over the rack."""
    pick = {(a, i): _duration(_pick(t, yp)) for a in PLAN for i, (t, _, yp, _) in enumerate(PLAN[a])}
    place = {(a, i): _duration(_place(t, k, yq)) for a in PLAN for i, (t, k, _, yq) in enumerate(PLAN[a])}
    free_at = {a: 0.0 for a in PLAN}  # when each arm is free to go for its next tube
    rack_clear = None  # when the arm last at the rack had risen from it
    out = {}
    for a, i in ORDER:
        go = free_at[a] + pick[(a, i)]
        if rack_clear is not None:
            go = max(go, rack_clear + RACK_GAP)
        out[(a, i)] = go
        rack_clear = free_at[a] = go + place[(a, i)]
    return out


def _arm(arm: str) -> list[Waypoint]:
    """One arm's two tubes: pick, wait (holding the tube over its cell) until its turn at the rack, place."""
    when = schedule()
    wps: list[Waypoint] = []
    t = 0.0
    for i, (tube, k, yaw_pick, yaw_place) in enumerate(PLAN[arm]):
        wps += _pick(tube, yaw_pick)
        t += _duration(_pick(tube, yaw_pick))
        go = when[(arm, i)]
        if go > t + 1e-9:
            wps.append(Waypoint(round(go - t, 3), wps[-1].pos, GRIP, "wait for the rack to be clear", yaw_pick))
            t = go
        wps += _place(tube, k, yaw_place)
        t += _duration(_place(tube, k, yaw_place))
    wps.append(Waypoint(2.0, PARK[arm], PLACE_OPEN, "aside", wps[-1].yaw))
    return wps


def demo():
    """Both arms on one clock, alternating at the analyser rack. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
