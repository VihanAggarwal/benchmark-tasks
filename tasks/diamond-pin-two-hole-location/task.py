"""On the station table: a grey steel fixture plate (100 x 250 x 15 mm, fixed) on the midline 30 cm in front of the
arm bases, with two 55 mm square sockets through it on a line along y, 100 mm apart (A on the robot's left, B on its
right). On a black tray on the right stand two round locating pins (48 mm across, 55 mm tall); on a black tray on the
left stands the diamond pin: a 48 mm round shank under a black head ground to a 48 x 8 mm bar, its long side along
the line between the sockets. On the table in front of the fixture lies a blue acetal bracket (88 x 212 x 10 mm) with
two 64 mm square holes 124 mm apart and a grip boss on top. Rule (two-hole location): one round pin in one socket and
the diamond pin in the other, turned so that its flats face the round pin, then the bracket flat on the fixture over
both pins. The holes are 24 mm farther apart than the sockets, so the bracket can come down over a round pin and the
diamond only with the diamond's flats facing the round pin; over two round pins, or over the diamond turned a quarter
turn, it rests on the pin heads.

The diamond pin and the bracket are loose mechkit compounds (mechkit.py, next to this file): each is one rigid body
made of several shapes, so the diamond's flats and the bracket's holes are real, and the scorer can read their pose
(base frame and orientation).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the diamond pin and bracket are compounds)
from scoring import BRACKET, Tracker, bracket_holes, pin_axis  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The arm sags a few mm under the 0.19 kg bracket, so the bracket comes to rest
# on the plate while the slow descent is still finishing, and success starts counting (after the 0.5 s needed to see
# it at rest) with the gripper still closed on the boss. Settling and letting go take 0.8 + 1.5 s, so with 2.5 s the
# trial ends with the fingers open (they clear the 30 mm boss within 0.2 s of starting to open).
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Locate the blue bracket on the fixture by its two holes: stand one round pin in one socket and the diamond pin "
    "in the other, turned so its two flat sides face the round pin, then lay the bracket flat on the fixture over "
    "both pins."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


FIXTURE = [_span_box(name, x0, x1, y0, y1, 0.0, L.FIX_H, L.FIXTURE_GREY)
           for name, (x0, x1, y0, y1) in L.FIXTURE_PARTS.items()]
LEFT_TRAY = Box("left_tray", pos=(L.LEFT_TRAY_X, L.LEFT_TRAY_Y, L.TOP + L.TRAY_H / 2), size=L.LEFT_TRAY_SIZE,
                color=L.TRAY_BLACK)
RIGHT_TRAY = Box("right_tray", pos=(L.RIGHT_TRAY_X, L.RIGHT_TRAY_Y, L.TOP + L.TRAY_H / 2), size=L.RIGHT_TRAY_SIZE,
                 color=L.TRAY_BLACK)
ROUND_PINS = [
    Cylinder(name, pos=(*L.ROUND_START[name], L.TOP + L.TRAY_TOP + L.PIN_H / 2), radius=L.PIN_R, height=L.PIN_H,
             mass=L.PIN_MASS, color=L.PIN_STEEL)
    for name in L.ROUND
]
DIAMOND_PIN = mk.assembly(
    L.DIAMOND,
    origin=(L.DIAMOND_START[0], L.DIAMOND_START[1], L.TOP + L.TRAY_TOP),
    yaw=L.DIAMOND_START_YAW,
    mass=L.DIAMOND_MASS,
    color=L.PIN_STEEL,
    shapes=[
        mk.cylinder(L.SHANK_CENTER, L.PIN_R, L.SHANK_H),
        mk.box(L.HEAD_CENTER, L.HEAD_SIZE, color=L.HEAD_BLACK),
    ],
)
BRACKET_PART = mk.assembly(
    BRACKET,
    origin=(L.BR_START[0], L.BR_START[1], L.TOP),
    mass=L.BR_MASS,
    color=L.BRACKET_BLUE,
    shapes=[mk.box(c, s) for c, s in L.BRACKET_SHAPES],
)
OBJECTS = [*FIXTURE, LEFT_TRAY, RIGHT_TRAY, *ROUND_PINS, DIAMOND_PIN, BRACKET_PART]

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


def _parts(objects) -> dict:
    return {n: objects.get(n) for n in (*L.PIN_NAMES, BRACKET)}


def success(objects) -> bool:
    """A round pin fitted in one socket and the diamond pin fitted in the other with its flats facing the round pin
    (head within 15 degrees of square to the line between the sockets), and the bracket lying flat on the plate over
    both pins, everything at rest; held 2.5 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = _parts(objects)
    result = _TRACKER.update(parts)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        dia, br = parts[L.DIAMOND], parts[BRACKET]
        row.update(source=getattr(br, "source", None), diamond_origin=dia.origin, diamond_quat=dia.quat_xyzw,
                   bracket_origin=br.origin, bracket_quat=br.quat_xyzw, bracket_holes=bracket_holes(br),
                   pin_bottom={n: pin_axis(n, parts[n])[0] for n in L.PIN_NAMES})
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any pin or the bracket is off the table, or no pin is fitted (upright within 5 degrees, its bottom
    within 3 mm of the socket floor, at rest). 0.25 with a pin fitted but no located set (one round pin and the
    diamond turned right, in different sockets). 0.5 with a located set and the bracket not seated. 1.0 with a
    located set and the bracket flat on the plate (level within 2 degrees, every underside corner within 2 mm of the
    plate's top) over both pins, at rest. Read from the state at hand (the last step)."""
    return _TRACKER.result(_parts(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm
PIN_GRIP = -(2 * L.PIN_R - 0.008) / 2  # 40 mm commanded across a round pin, or across the diamond's 48 mm head
BOSS_GRIP = -(L.BOSS_SIZE[0] - 0.008) / 2  # 22 mm commanded across the bracket's 30 mm boss
PAD_TIP = 0.013  # the pad tips are 13 mm below the TCP in a straight-down grip
# A pin is held with the pad tips 25 mm above its bottom: 30 mm of the pads on it (on the diamond, on its head's two
# 8 mm ends, which start 15 mm up). The bracket is held by its boss with the pad tips 4 mm above the plate's top.
PIN_GRIP_UP = 0.025 + PAD_TIP  # TCP height above a pin's bottom
BR_GRIP_UP = L.BR_T + 0.004 + PAD_TIP  # TCP height above the bracket's underside
DROP = 0.002  # a pin is let go with its bottom 2 mm above the socket floor
BR_DROP = 0.001  # the bracket with its underside 1 mm above the plate (the arm sags a few mm under both)
APPROACH = 0.030  # each final descent at a pick starts this far above the grip height
PIN_ENTRY = 0.035  # ... and at a socket this far above the place height: the pin's bottom 22 mm over the plate
BR_ENTRY = L.PIN_H + 0.010 - (L.FIX_TOP + BR_DROP)  # the bracket's underside 10 mm over the pin heads: 49 mm to go
HOVER_PIN = L.TOP + 0.12  # carried with a pin's bottom 82 mm above the table
HOVER_BR = L.TOP + 0.11  # carried with the bracket's underside 83 mm above the table, 28 mm over the pin heads
# Grip yaws: 0 closes the fingers along y. The diamond is picked at yaw 0 (its head's long side lies along y) and
# turned a quarter turn in the hand on the way to socket A, so it is set down with its long side along x: its flats
# face socket B. Near the midline the left arm works at yaw -pi/2 and the right arm at +pi/2 (fingers along x).
YAW_DIAMOND_PICK, YAW_LEFT, YAW_RIGHT = 0.0, -math.pi / 2, math.pi / 2
# Trim of each place point (x, y), for the GPU run. On the GPU, parts set down near x 0.30 by the left arm landed
# about (-2.2, +1.8) mm off their aim under 0.15 kg (and the right arm's (-4.4, -4.2) mm under 0.28 kg): each arm's
# sag toward its base. The trims below take off about half of that, scaled to each part's mass, as the heatsink task
# did. The sockets leave 3.5 mm of play either side of a pin and the bracket's window is 12 mm wide (aim in its
# middle): if a part lands far off its aim (centre_off_mm, bracket_y_mm in TASK_TRACE), move its point by minus the
# offset.
TRIM = {"diamond": (0.0007, -0.0006), "round": (0.0007, 0.0006), "bracket": (0.0012, -0.0010)}
# Waits on one clock: the two sockets are only 0.10 m apart, so the arms take turns at the fixture. The right arm
# holds its round pin up over its tray until the left arm has set the diamond in A and gone; the left arm holds the
# bracket up over its start until the right arm has set the round pin in B and moved aside.
RIGHT_WAIT = 9.0
LEFT_WAIT = 1.5
PARK_L = (0.20, 0.16, L.TOP + 0.15)
PARK_R = (0.18, -0.25, L.TOP + 0.15)


def _pin_place(k: str, trim) -> tuple[float, float, float]:
    sx, sy = L.SOCKETS[k]
    return (sx + trim[0], sy + trim[1], L.TOP + DROP + PIN_GRIP_UP)


def _pin_moves(pick, place, yaw_pick, yaw_place, what, where, wait=0.0) -> list[Waypoint]:
    """Pick a pin off its tray and stand it in a socket: come down beside it slowly, grip it, lift it, (wait), carry
    it over the socket (turning the hand to ``yaw_place``), lower it slowly into the socket, let go, go straight up."""
    px, py, pz = pick
    qx, qy, qz = place
    plan = [
        Waypoint(1.5, (px, py, HOVER_PIN), OPEN, f"over the {what}", yaw_pick),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, f"down to just above the {what}", yaw_pick),
        Waypoint(1.5, (px, py, pz), OPEN, f"slowly down beside the {what}", yaw_pick),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", yaw_pick),
        Waypoint(1.5, (px, py, pz), PIN_GRIP, f"grip the {what}", yaw_pick),
        Waypoint(1.5, (px, py, HOVER_PIN), PIN_GRIP, f"lift the {what}", yaw_pick),
    ]
    if wait:
        plan.append(Waypoint(wait, (px, py, HOVER_PIN), PIN_GRIP, "wait for the other arm to clear the fixture",
                             yaw_pick))
    plan += [
        Waypoint(2.0, (qx, qy, HOVER_PIN), PIN_GRIP, f"carry it over {where}", yaw_place),
        Waypoint(0.5, (qx, qy, HOVER_PIN), PIN_GRIP, f"settle over {where}", yaw_place),
        Waypoint(1.0, (qx, qy, qz + PIN_ENTRY), PIN_GRIP, f"down to just above {where}", yaw_place),
        Waypoint(2.0, (qx, qy, qz), PIN_GRIP, f"slowly lower it into {where}", yaw_place),
        Waypoint(0.5, (qx, qy, qz), PIN_GRIP, "settle", yaw_place),
        Waypoint(1.5, (qx, qy, qz), OPEN, f"let go in {where}", yaw_place),
        Waypoint(1.0, (qx, qy, HOVER_PIN), OPEN, "straight up", yaw_place),
    ]
    return plan


BR_PICK = (L.BR_START[0], L.BR_START[1], L.TOP + BR_GRIP_UP)
BR_PLACE = (L.FIX_X + TRIM["bracket"][0], L.FIX_Y + L.BR_AIM_Y + TRIM["bracket"][1],
            L.TOP + L.FIX_TOP + BR_DROP + BR_GRIP_UP)
DIAMOND_PICK = (L.DIAMOND_START[0], L.DIAMOND_START[1], L.TOP + L.TRAY_TOP + PIN_GRIP_UP)
ROUND_PICK = (*L.ROUND_START["round_pin_1"], L.TOP + L.TRAY_TOP + PIN_GRIP_UP)


def _left() -> list[Waypoint]:
    """Left arm: the diamond pin into socket A, turned a quarter turn so its flats face B; then the bracket by its
    boss, held up until the right arm is clear, lowered slowly over both pins, let go; then aside."""
    plan = _pin_moves(DIAMOND_PICK, _pin_place("A", TRIM["diamond"]), YAW_DIAMOND_PICK, YAW_LEFT,
                      "diamond pin", "socket A")
    px, py, pz = BR_PICK
    qx, qy, qz = BR_PLACE
    plan += [
        Waypoint(2.0, (px, py, HOVER_BR), OPEN, "over the bracket's boss", YAW_LEFT),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, "down to just above the boss", YAW_LEFT),
        Waypoint(1.5, (px, py, pz), OPEN, "slowly down beside the boss", YAW_LEFT),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", YAW_LEFT),
        Waypoint(1.5, (px, py, pz), BOSS_GRIP, "grip the boss", YAW_LEFT),
        Waypoint(1.5, (px, py, HOVER_BR), BOSS_GRIP, "lift the bracket", YAW_LEFT),
        Waypoint(LEFT_WAIT, (px, py, HOVER_BR), BOSS_GRIP, "wait for the right arm to clear the fixture", YAW_LEFT),
        Waypoint(2.5, (qx, qy, HOVER_BR), BOSS_GRIP, "carry it over the fixture, holes over the pins", YAW_LEFT),
        Waypoint(0.8, (qx, qy, HOVER_BR), BOSS_GRIP, "settle over the pins", YAW_LEFT),
        Waypoint(1.0, (qx, qy, qz + BR_ENTRY), BOSS_GRIP, "down to just above the pin heads", YAW_LEFT),
        Waypoint(3.0, (qx, qy, qz), BOSS_GRIP, "slowly lower it over both pins onto the plate", YAW_LEFT),
        Waypoint(0.8, (qx, qy, qz), BOSS_GRIP, "settle", YAW_LEFT),
        Waypoint(1.5, (qx, qy, qz), OPEN, "let go of the bracket", YAW_LEFT),
        Waypoint(1.0, (qx, qy, HOVER_BR), OPEN, "straight up", YAW_LEFT),
        Waypoint(1.5, PARK_L, OPEN, "aside", YAW_LEFT),
    ]
    return plan


def _right() -> list[Waypoint]:
    """Right arm: the nearer round pin into socket B once the left arm has left A; then aside."""
    plan = _pin_moves(ROUND_PICK, _pin_place("B", TRIM["round"]), YAW_RIGHT, YAW_RIGHT, "round pin", "socket B",
                      wait=RIGHT_WAIT)
    plan.append(Waypoint(1.5, PARK_R, OPEN, "aside", YAW_RIGHT))
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
