"""On the station table: a green 'machine' (a loose painted-steel box, 70 mm along x, 90 mm along y, 60 mm tall,
0.30 kg) standing on the midline 30 cm in front of the arm bases, and a loose hardwood skid (a plank 60 mm wide,
440 mm long, 12 mm thick, 0.12 kg) lying flat on the table to its right, long side along y, its left end 5 mm short
of the machine. The machine must be raised only as far as needed (no more than 30 mm), held where it stands while
the skid slides under it, and set down on the skid. Dragging it more than 30 mm from its place scores 0.

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
# Success must hold this long. Longer than letting go of the machine and lifting the hand clear (1.5 s + 1.0 s in
# the demo), so the episode cannot end while the machine still hangs in the closed gripper just above the skid.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Raise the green machine no more than 3 cm, slide the wooden skid under it, and set the machine down on the "
    "skid. Don't drag the machine out of place."
)

MACHINE = Box(
    "machine",
    pos=(L.MACHINE_X, L.MACHINE_Y, L.TOP + L.MACHINE_SIZE[2] / 2),
    size=L.MACHINE_SIZE,
    mass=L.MACHINE_MASS,
    color=L.GREEN,
)
SKID = Box(
    "skid",
    pos=(L.SKID_X, L.SKID_Y, L.TOP + L.SKID_SIZE[2] / 2),
    size=L.SKID_SIZE,
    mass=L.SKID_MASS,
    color=L.WOOD,
)
OBJECTS = [MACHINE, SKID]

_TRACKER = Tracker()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def success(objects) -> bool:
    """Machine on the skid, never more than 10 mm from its start horizontally nor raised more than 30 mm, both
    objects still (SUCCESS_HOLD_S = 3 s). This runs first every control step, so it also feeds the tracker."""
    result = _TRACKER.update(objects["machine"], objects["skid"])
    _trace({"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details,
            "machine_pos": objects["machine"].pos, "skid_pos": objects["skid"].pos})
    return result.success


def score(objects) -> float:
    """0 if the machine ever tilted more than 30 deg or drifted more than 30 mm, or ends off the table. Else 1.0 on
    the skid, drift <= 10 mm and raised <= 30 mm; 0.75 on the skid, raised <= 50 mm; 0.5 on the skid, raised
    higher; 0.25 the skid under the machine but the machine not resting on it; 0 otherwise."""
    return _TRACKER.result(objects["machine"], objects["skid"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
MACHINE_GRIP = -(L.MACHINE_SIZE[0] - 0.008) / 2  # across the machine's 70 mm x-length
SKID_GRIP = -(L.SKID_SIZE[0] - 0.008) / 2  # across the skid's 60 mm width
# Both grips close the fingers along x. The left arm needs yaw -pi/2 to reach the midline. The right arm works on
# its own outer side (the skid's right end is 16.5 cm outboard of its base), where yaw +pi/2 runs joint6 to within
# 1 deg of its limit; the same parallel grip turned 180 deg (yaw -pi/2) keeps joint6 >= 10 deg from its limits.
YAW_L = -math.pi / 2
YAW_R = -math.pi / 2
PAD = 0.013  # the pad tips are 13 mm below the TCP
MACHINE_TCP_Z = L.TOP + 0.015 + PAD  # pad tips 15 mm above the table, on the machine's x faces
# The arm sags under a held load (joints 1-3: 80 N m/rad, no gravity compensation for the payload): 8.4 mm down and
# 3.5 mm toward the left base with the 0.30 kg machine at this reach (J K^-1 J^T of its weight, all six joints;
# the wrist joints carry none of it), and the gripper tilts about 1.4 deg, which puts one corner of the underside
# about 1.4 mm above its centre. So the commanded raise is 28 mm, for a machine about 20 mm up in the simulator
# (lowest corner 6 mm over the skid's 12 mm top as it slides under). It stays under 30 mm for any sag from 0 and
# clears the skid by 2 mm for a sag up to 12 mm. On the GPU, read max_lift_mm in TASK_TRACE: keep it 16-28 mm.
RAISE = 0.028
HELD_TCP_Z = MACHINE_TCP_Z + RAISE
SET_GAP = 0.006  # commanded height of the machine's underside over the skid's top when let go: with the sag, the
#                  machine already rests on the skid (the arm still carrying about 70 % of its weight), no drop
SET_TCP_Z = L.SKID_TOP + SET_GAP + (MACHINE_TCP_Z - L.TOP)
SKID_TCP_Z = L.TOP + 0.004 + PAD  # pad tips 4 mm above the table, on the skid's long sides (8 mm of the 12 mm face)
APPROACH = 0.030  # each final descent starts this far above the grip height and takes 1.5 s
GRIP_Y0 = L.SKID_Y - L.SKID_SIZE[1] / 2 + 0.02  # 2 cm from the skid's right end: y -0.47
SKID_END_Y = -0.05  # where the skid's centre ends: its 440 mm cover y -0.27..+0.17, the machine's 90 mm and 12 cm more
SLIDE = SKID_END_Y - L.SKID_Y  # 0.22 m
GRIP_Y1 = GRIP_Y0 + SLIDE  # y -0.25: the TCPs 25 cm apart at the end of the slide
EXIT_Y = GRIP_Y1 - 0.10  # the right arm backs off outward once it has let go
HOVER_L, HOVER_R = L.TOP + 0.12, L.TOP + 0.10


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _left_raise() -> list[Waypoint]:
    mx, my = L.MACHINE_X, L.MACHINE_Y
    return [
        Waypoint(2.0, (mx, my, HOVER_L), OPEN, "over the machine", YAW_L),
        Waypoint(1.0, (mx, my, MACHINE_TCP_Z + APPROACH), OPEN, "down to just above the grip", YAW_L),
        Waypoint(1.5, (mx, my, MACHINE_TCP_Z), OPEN, "slowly down beside its x faces", YAW_L),
        Waypoint(0.5, (mx, my, MACHINE_TCP_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (mx, my, MACHINE_TCP_Z), MACHINE_GRIP, "grip the machine", YAW_L),
        Waypoint(0.5, (mx, my, MACHINE_TCP_Z), MACHINE_GRIP, "settle the grip", YAW_L),
        Waypoint(2.0, (mx, my, HELD_TCP_Z), MACHINE_GRIP, "raise it about 2 cm, straight up", YAW_L),
    ]


def _right() -> list[Waypoint]:
    """Right arm: grip the skid's right end, wait for the machine to be up, slide the skid 22 cm along y under
    it at table height, let go and back off up and outward before the machine comes down."""
    sx = L.SKID_X
    pick = [
        Waypoint(2.0, (sx, GRIP_Y0, HOVER_R), OPEN, "over the skid's right end", YAW_R),
        Waypoint(1.0, (sx, GRIP_Y0, SKID_TCP_Z + APPROACH), OPEN, "down to just above the skid", YAW_R),
        Waypoint(1.5, (sx, GRIP_Y0, SKID_TCP_Z), OPEN, "slowly down beside its sides", YAW_R),
        Waypoint(0.5, (sx, GRIP_Y0, SKID_TCP_Z), OPEN, "settle", YAW_R),
        Waypoint(1.5, (sx, GRIP_Y0, SKID_TCP_Z), SKID_GRIP, "grip the skid", YAW_R),
    ]
    wait = _span(_left_raise()) + 0.5 - _span(pick)
    return pick + [
        Waypoint(wait, (sx, GRIP_Y0, SKID_TCP_Z), SKID_GRIP, "wait for the machine to be up", YAW_R),
        Waypoint(4.0, (sx, GRIP_Y1, SKID_TCP_Z), SKID_GRIP, "slide it under the raised machine", YAW_R),
        Waypoint(0.5, (sx, GRIP_Y1, SKID_TCP_Z), SKID_GRIP, "settle", YAW_R),
        Waypoint(1.5, (sx, GRIP_Y1, SKID_TCP_Z), OPEN, "let go of the skid", YAW_R),
        Waypoint(1.5, (sx, EXIT_Y, HOVER_R), OPEN, "up and outward, out of the left arm's way", YAW_R),
    ]


def _left() -> list[Waypoint]:
    """Left arm: grip the machine across its x-length, raise it about 2 cm and hold it there until the skid is
    under it and the right arm is clear, lower it to just above the skid, let go, go up."""
    mx, my = L.MACHINE_X, L.MACHINE_Y
    up = _left_raise()
    hold = _span(_right()) - _span(up)
    return up + [
        Waypoint(hold, (mx, my, HELD_TCP_Z), MACHINE_GRIP, "hold it up while the skid slides under", YAW_L),
        Waypoint(1.5, (mx, my, SET_TCP_Z), MACHINE_GRIP, "lower it to just above the skid", YAW_L),
        Waypoint(0.5, (mx, my, SET_TCP_Z), MACHINE_GRIP, "settle", YAW_L),
        Waypoint(1.5, (mx, my, SET_TCP_Z), OPEN, "set it down on the skid", YAW_L),
        Waypoint(1.0, (mx, my, HOVER_L), OPEN, "up", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
