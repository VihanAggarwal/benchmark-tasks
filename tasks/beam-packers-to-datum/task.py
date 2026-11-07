"""On the station table: two fixed grey concrete piers (60 x 60 mm), one 29 cm in front of each arm and 18 cm either
side of the midline, the left one 20 mm tall and the right one 30 mm tall. Behind each pier stands a fixed grey
datum post (20 x 20 mm) with a red top band; its top, 70 mm above the table, is the level both bearings must reach.
In front of each arm lies a row of three loose beech packers (50 x 50 mm, colour-coded by thickness: yellow 20 mm,
blue 30 mm, green 40 mm), and a loose spruce beam (420 x 40 x 30 mm, 0.227 kg) lies flat on the midline 15 cm in
front of the arms, its length across the table. Rule: pack each pier up to the datum with packers lying flat and
fully on it (the left needs 20 + 30, the right 40), then seat the beam level across both; a beam left across bad
packing (out of level) or a packer stood on its edge on a pier scores 0.

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
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 3 s, not the default 1 s. The arm sags a few mm under the 0.227 kg beam (no payload compensation), so
# the beam comes to rest on the packing up to about 1 s before the slow descent ends and success starts counting
# 0.5 s later (everything must have been still for 0.5 s) while the gripper still holds it. With 3 s the trial ends
# after the fingers are fully open (1.5 s opening after a 0.3 s settle).
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Pack each grey pier with the wooden packers (yellow 20 mm, blue 30 mm, green 40 mm thick) until its top is "
    "level with the top of the red-capped datum post behind it, every packer lying flat and fully on the pier. "
    "Then lay the beam across both packed piers."
)


def _piers() -> list[Box]:
    out = []
    for side in L.SIDES:
        h = L.PIER_H[side]
        out.append(Box(f"pier_{side}", pos=(L.PIER_X, L.SIGN[side] * L.PIER_Y, L.TOP + h / 2),
                       size=(L.PIER_SIDE, L.PIER_SIDE, h), color=L.CONCRETE))
    return out


def _posts() -> list[Box]:
    out = []
    for side in L.SIDES:
        y = L.SIGN[side] * L.POST_Y
        out.append(Box(f"datum_post_{side}", pos=(L.POST_X, y, L.TOP + L.POST_BODY_H / 2),
                       size=(L.POST_SIDE, L.POST_SIDE, L.POST_BODY_H), color=L.POST_GREY))
        out.append(Box(f"datum_cap_{side}", pos=(L.POST_X, y, L.TOP + L.POST_BODY_H + L.CAP_H / 2),
                       size=(L.POST_SIDE, L.POST_SIDE, L.CAP_H), color=L.CAP_RED))
    return out


def _packers() -> list[Box]:
    out = []
    for side in L.SIDES:
        for t, (thick, mass, colour) in L.PACKERS.items():
            out.append(Box(L.packer_name(t, side), pos=(L.SET_X[t], L.SIGN[side] * L.SET_Y, L.TOP + thick / 2),
                           size=(L.PACKER_SIDE, L.PACKER_SIDE, thick), mass=mass, color=L.PACKER_COLOURS[colour]))
    return out


BEAM = Box("beam", pos=(L.BEAM_X, L.BEAM_Y, L.TOP + L.BEAM_SIZE[2] / 2), size=L.BEAM_SIZE, mass=L.BEAM_MASS,
           color=L.SPRUCE)
OBJECTS = [*_piers(), *_posts(), *_packers(), BEAM]

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


def success(objects) -> bool:
    """Both piers packed to the datum (every packer on them flat, fully on the pier, resting and still; packing top
    within 3 mm of 70 mm) and the beam seated level across them (on a long face, level within 1.25 deg, resting on
    each top packer with its centre line crossing it and 30 mm of bearing, still). This runs first every control
    step, so it also feeds the tracker."""
    result = _TRACKER.update(objects)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    if "beam" in objects:
        row["beam_pos"] = objects["beam"].pos
        row["beam_quat"] = objects["beam"].quat_xyzw
    for n in L.PACKER_NAMES:
        if n in objects:
            row[f"{n}_pos"] = objects[n].pos
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a packer on a pier stands on an edge or end, or if the beam rests across both piers out of level by more
    than 1.25 deg. Else 0.25 for each pier packed to the datum, plus 0.5 if both are packed and the beam is seated
    level across them: 0, 0.25, 0.5 or 1.0."""
    return _TRACKER.result(objects).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
PACKER_GRIP = -(L.PACKER_SIDE - 0.008) / 2  # yaw 0: fingers close along y, across the 50 mm (42 mm commanded)
BEAM_GRIP = -(L.BEAM_SIZE[0] - 0.008) / 2  # yaw pi/2: fingers close along x, across the beam's 40 mm (32 mm)
BEAM_YAW = math.pi / 2  # the right arm's sign that keeps joint6 inside its limits
# TCP height above a packer's bottom when gripping it: pad tips (13 mm past the TCP) 4, 7 and 9 mm above it.
GRIP_UP = {20: 0.017, 30: 0.020, 40: 0.022}
BEAM_GRIP_UP = 0.017  # pad tips 4 mm above the beam's underside
CARRY_Z = L.TOP + 0.12  # a carried packer's bottom 98-103 mm up: clear of the piers (30 mm) and the posts (70 mm)
ENTRY = 0.015  # the last 15 mm down onto a support are lowered slowly
PLACE_GAP = 0.001  # let go with the part's underside 1 mm above its support
PARK = (0.220, L.SET_Y, L.TOP + 0.16)  # the left arm waits over its own packer row while the beam goes on
BEAM_CARRY_Z = L.TOP + 0.110  # the carried beam's underside 93 mm up: 23 mm over the 70 mm packing
BEAM_HOVER_Z = L.TOP + 0.120
# While the left arm packs its pier the right arm waits here, not over the beam: over the beam's middle its forearm
# comes within 6 cm of the left arm's upper arm (capsule model); here 13 cm.
RIGHT_WAIT = (0.200, -0.120, L.TOP + 0.140)


def _move(x, y, z, finger, label, dur, yaw=0.0) -> Waypoint:
    return Waypoint(dur, (x, y, z), finger, label, yaw)


def _pack(t: int, side: str, support_h: float, first_move: float) -> list[Waypoint]:
    """Pick the t mm packer of this side's set and lay it flat on its pier, its underside on a support support_h
    above the table (the pier top, or the packer already there)."""
    px, py = L.SET_X[t], L.SIGN[side] * L.SET_Y
    qx, qy = L.PIER_X, L.SIGN[side] * L.PIER_Y
    pick = L.TOP + GRIP_UP[t]
    place = L.TOP + support_h + PLACE_GAP + GRIP_UP[t]
    name = f"{L.PACKERS[t][2]} {t} mm packer"
    return [
        _move(px, py, CARRY_Z, OPEN, f"over the {name}", first_move),
        _move(px, py, pick + 0.020, OPEN, "down to just above it", 1.0),
        _move(px, py, pick, OPEN, "slowly down round it", 1.0),
        _move(px, py, pick, OPEN, "settle", 0.3),
        _move(px, py, pick, PACKER_GRIP, f"grip the {name}", 1.5),
        _move(px, py, pick, PACKER_GRIP, "settle the grip", 0.3),
        _move(px, py, CARRY_Z, PACKER_GRIP, "lift", 1.0),
        _move(qx, qy, CARRY_Z, PACKER_GRIP, "carry over the pier", 1.5),
        _move(qx, qy, place + ENTRY, PACKER_GRIP, "down to 15 mm above its support", 1.0),
        _move(qx, qy, place + ENTRY, PACKER_GRIP, "settle", 0.3),
        _move(qx, qy, place, PACKER_GRIP, "lower the last 15 mm slowly", 1.5),
        _move(qx, qy, place, OPEN, "let go", 1.5),
        _move(qx, qy, CARRY_Z, OPEN, "straight up", 1.0),
    ]


def _duration(wps: list[Waypoint]) -> float:
    return sum(w.duration for w in wps)


def _left() -> list[Waypoint]:
    """Left arm: the yellow 20 mm packer onto the 20 mm pier, the blue 30 mm on top of it (70 mm), then out of the
    way over its own packer row."""
    plan = _pack(20, "left", L.PIER_H["left"], 2.0)
    plan += _pack(30, "left", L.PIER_H["left"] + L.PACKERS[20][0], 1.5)
    plan.append(_move(*PARK, OPEN, "out of the beam's way", 1.5))
    return plan


# The right arm grips the beam only once the left arm has parked: gripping the beam's middle with the wrist turned
# 90 deg, the right forearm's motor housing reaches about 65 mm left of the midline, 4 cm from the left fingers
# while those are still up over the left pier (capsule model); with the left arm parked the arms keep 9 cm apart.
LEFT_CLEAR_S = _duration(_left())


def _right() -> list[Waypoint]:
    """Right arm: the green 40 mm packer onto the 30 mm pier (70 mm); then wait clear of the left arm, come over the
    beam's middle as the left arm parks, grip the beam, lift it, carry it over both piers, lower it slowly onto the
    packing, let go, go up."""
    plan = _pack(40, "right", L.PIER_H["right"], 2.0)
    bx, by = L.BEAM_X, L.BEAM_Y
    pick = L.TOP + BEAM_GRIP_UP
    place = L.TOP + L.DATUM + 0.0005 + BEAM_GRIP_UP  # underside aimed 0.5 mm above the packing
    plan.append(_move(*RIGHT_WAIT, OPEN, "wait clear of the left arm", 1.5))
    over = _move(bx, by, BEAM_HOVER_Z, OPEN, "over the beam's middle", 1.5, BEAM_YAW)
    grip = [
        _move(bx, by, pick + 0.020, OPEN, "down to just above the beam", 1.2, BEAM_YAW),
        _move(bx, by, pick, OPEN, "slowly down round it", 1.0, BEAM_YAW),
        _move(bx, by, pick, OPEN, "settle", 0.3, BEAM_YAW),
        _move(bx, by, pick, BEAM_GRIP, "grip the beam across its width", 1.5, BEAM_YAW),
        _move(bx, by, pick, BEAM_GRIP, "settle the grip", 0.3, BEAM_YAW),
    ]
    wait = LEFT_CLEAR_S - _duration(plan) - over.duration  # over the beam as the left arm arrives at its park
    plan.append(_move(*RIGHT_WAIT, OPEN, "wait for the left pier to be packed", max(wait, 0.1)))
    plan.append(over)
    plan += grip
    plan += [
        _move(bx, by, BEAM_CARRY_Z, BEAM_GRIP, "lift the beam", 1.2, BEAM_YAW),
        _move(L.PIER_X, 0.0, BEAM_CARRY_Z, BEAM_GRIP, "carry it over both piers", 1.5, BEAM_YAW),
        _move(L.PIER_X, 0.0, place + 0.010, BEAM_GRIP, "down to 10 mm above the packing", 0.8, BEAM_YAW),
        _move(L.PIER_X, 0.0, place + 0.010, BEAM_GRIP, "settle", 0.5, BEAM_YAW),
        _move(L.PIER_X, 0.0, place, BEAM_GRIP, "lower the last 10 mm slowly", 2.0, BEAM_YAW),
        _move(L.PIER_X, 0.0, place, BEAM_GRIP, "settle", 0.3, BEAM_YAW),
        _move(L.PIER_X, 0.0, place, OPEN, "let go", 1.5, BEAM_YAW),
        _move(L.PIER_X, 0.0, BEAM_HOVER_Z, OPEN, "straight up", 1.0, BEAM_YAW),
    ]
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
