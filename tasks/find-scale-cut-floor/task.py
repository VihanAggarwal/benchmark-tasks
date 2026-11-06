"""On the station table: a small excavation. Four light-brown 40 mm blocks (the ground surface) make a square frame,
outer 300 x 300 mm, centred 26 cm in front of the arm bases on the midline, round a sunken cut 180 x 180 mm whose
floor is the table top. On the cut floor lie a cream find (90 x 26 x 16 mm, 35 g, long axis at 20 deg) and, parallel
to it and 18 mm from its -y side, a grey trowel (140 x 40 x 18 mm, 90 g) on the strip where the scale belongs. An
orange tool tray stands 36 cm right of the midline (only the right arm reaches it) and a dark-grey photo-kit box
33 cm left (only the left arm reaches it) with a white 10 cm scale bar (100 x 20 x 15 mm, 25 g) in it.

Rule: the find is photographed where it lies. The trowel goes in the tray, the scale lies flat on the cut floor
beside the find, centred alongside it, parallel and 1 to 4 cm from it, the find is never moved, and both hands end
clear of the shot.

The scorer reads the arms (TCP, gripper opening) and the episode number through rigstate.py, which wraps the
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
# The 1.0 state with the hands clear and the find, scale and trowel each still (within 3 mm over the last 0.5 s),
# held for 1 s.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Get the find ready for its record photograph. Put the trowel back in the orange tool tray, then lay the 10 cm "
    "scale flat on the floor of the cut beside the find, centred alongside it, parallel to it and 1 to 4 cm from it. "
    "Leave the find exactly where it lies. When you are done, let go and move both arms back, clear of the cut and "
    "out of the photo."
)

BANKS = [Box(name, pos=(x, y, L.TOP + L.BANK_H / 2), size=(sx, sy, L.BANK_H), color=L.BANK_BROWN)
         for name, x, y, sx, sy in L.BANKS]
FIND = Box("find", pos=(L.FIND_X, L.FIND_Y, L.TOP + L.FIND_SIZE[2] / 2), size=L.FIND_SIZE, mass=L.FIND_MASS,
           color=L.FIND_CREAM, yaw=L.FIND_YAW)
TROWEL = Box("trowel", pos=(L.TROWEL_X, L.TROWEL_Y, L.TOP + L.TROWEL_SIZE[2] / 2), size=L.TROWEL_SIZE,
             mass=L.TROWEL_MASS, color=L.TROWEL_GREY, yaw=L.TROWEL_YAW)


def _open_box(prefix: str, x: float, y: float, outer, floor_h: float, wall_t: float, wall_h: float, color):
    """An open tray of fixed boxes: a floor and four walls standing on it (long walls along x)."""
    ox, oy = outer
    wz = L.TOP + floor_h + wall_h / 2
    return [
        Box(f"{prefix}_floor", pos=(x, y, L.TOP + floor_h / 2), size=(ox, oy, floor_h), color=color),
        Box(f"{prefix}_wall_left", pos=(x, y + oy / 2 - wall_t / 2, wz), size=(ox, wall_t, wall_h), color=color),
        Box(f"{prefix}_wall_right", pos=(x, y - oy / 2 + wall_t / 2, wz), size=(ox, wall_t, wall_h), color=color),
        Box(f"{prefix}_wall_far", pos=(x + ox / 2 - wall_t / 2, y, wz), size=(wall_t, oy - 2 * wall_t, wall_h),
            color=color),
        Box(f"{prefix}_wall_near", pos=(x - ox / 2 + wall_t / 2, y, wz), size=(wall_t, oy - 2 * wall_t, wall_h),
            color=color),
    ]


TRAY = _open_box("tool_tray", L.TRAY_X, L.TRAY_Y, L.TRAY_OUTER, L.TRAY_FLOOR_H, L.TRAY_WALL_T, L.TRAY_WALL_H,
                 L.TRAY_ORANGE)
KIT = _open_box("kit_box", L.KIT_X, L.KIT_Y, L.KIT_OUTER, L.KIT_FLOOR_H, L.KIT_WALL_T, L.KIT_WALL_H, L.KIT_GREY)
SCALE = Box("scale", pos=(L.SCALE_X, L.SCALE_Y, L.KIT_FLOOR_Z + L.SCALE_SIZE[2] / 2), size=L.SCALE_SIZE,
            mass=L.SCALE_MASS, color=L.SCALE_WHITE, yaw=L.SCALE_YAW)
OBJECTS = [*BANKS, FIND, TROWEL, *TRAY, *KIT, SCALE]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the tracker also starts over when rigstate.episode()
    changes, and at no other time)."""
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
    """The trowel stowed and the scale tight (on the cut floor, within 5 deg of parallel, g 10-40 mm, s <= 25 mm), the
    find never disturbed, nothing resting on it, both hands clear; the find, scale and trowel each within 3 mm over
    the last 0.5 s; held 1 s. This runs first every control step, so it also feeds the tracker (with the arms
    rigstate read for this step, and rigstate's episode number: nothing clears the record within one episode)."""
    find, trowel, scale = objects["find"], objects["trowel"], objects["scale"]
    arms = rigstate.arms()
    result = _TRACKER.update(find, trowel, scale, arms, rigstate.episode())
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "find_pos": find.pos, "find_quat": find.quat_xyzw, "trowel_pos": trowel.pos,
           "trowel_quat": trowel.quat_xyzw, "scale_pos": scale.pos, "scale_quat": scale.quat_xyzw}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the find was ever disturbed (moved over 5 mm, turned or tilted over 3 deg, lifted over 3 mm), something
    rests on it, or the hands are not clear (a gripper under half open, or a TCP over the frame, within 60 mm of it
    or beyond its far side). Else 1.0 tight and stowed; 0.75 rough and stowed; 0.5 tight or rough, trowel not
    stowed; 0.25 stowed only."""
    return _TRACKER.result(objects["find"], objects["trowel"], objects["scale"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695


def gap(mm: float) -> float:
    """Finger joint target for a commanded pad gap of ``mm`` millimetres."""
    return -mm / 2000.0


YAW = L.TROWEL_YAW  # both the trowel and the placed scale lie at 20 deg: yaw 20 closes the fingers across them
UP = L.TOP + 0.12  # the right arm's travel height
CROSS = L.TOP + 0.10  # the left arm crosses the midline at 0.10 m (joint4 margin 0.34 rad; 0.25 at 0.12 m)
_U = (math.cos(L.TROWEL_YAW), math.sin(L.TROWEL_YAW))
HANDLE_OFF = 0.048  # the trowel is gripped at its handle end, 48 mm from its centre toward -x
TROWEL_GRIP = (L.TROWEL_X - HANDLE_OFF * _U[0], L.TROWEL_Y - HANDLE_OFF * _U[1])  # (0.2091, -0.0246)
TROWEL_STOW = (L.TRAY_X - HANDLE_OFF * _U[0], L.TRAY_Y - HANDLE_OFF * _U[1])  # its centre to the tray's centre
TROWEL_PICK_Z = L.TOP + 0.015  # pad tips 2 mm above the table
TROWEL_DROP_Z = L.TOP + 0.025  # its bottom on the tray floor, pad tips 2 mm above it
SCALE_PICK_Z = L.TOP + 0.020  # pad tips 7 mm above the table, 2 mm above the kit floor
# The scale's place: g 30 mm, s 0, parallel to the find (its centre 53 mm from the find's centre line, toward -y).
_FN = (-math.sin(L.FIND_YAW), math.cos(L.FIND_YAW))
SCALE_OFF = L.HALF_WIDTHS + 0.030
SCALE_PLACE = (L.FIND_X - SCALE_OFF * _FN[0], L.FIND_Y - SCALE_OFF * _FN[1])  # (0.2831, 0.0002)
SCALE_PLACE_Z = L.TOP + 0.015  # its bottom on the cut floor
SCALE_ENTRY_Z = L.TOP + 0.040  # the last 25 mm are lowered slowly
LEFT_WAIT = (0.28, 0.20)  # the left arm holds the scale here until the trowel is over the tray
LEFT_REST = (0.25, 0.25, L.TOP + 0.12)
RIGHT_REST = (0.20, -0.25, L.TOP + 0.12)


def _right() -> list[Waypoint]:
    """Right arm: take the trowel by its handle end, carry it to the tool tray, lay it on the tray floor, move back."""
    gx, gy = TROWEL_GRIP
    tx, ty = TROWEL_STOW
    return [
        Waypoint(2.5, (gx, gy, UP), OPEN, "over the trowel's handle end", YAW),
        Waypoint(1.5, (gx, gy, UP), gap(52), "fingers to 52 mm", YAW),
        Waypoint(1.5, (gx, gy, TROWEL_PICK_Z), gap(52), "down round the handle", YAW),
        Waypoint(0.5, (gx, gy, TROWEL_PICK_Z), gap(52), "settle", YAW),
        Waypoint(1.5, (gx, gy, TROWEL_PICK_Z), gap(32), "grip the trowel (32 mm on 40)", YAW),
        Waypoint(0.3, (gx, gy, TROWEL_PICK_Z), gap(32), "settle the grip", YAW),
        Waypoint(1.5, (gx, gy, UP), gap(32), "lift it out of the cut", YAW),
        Waypoint(2.5, (tx, ty, UP), gap(32), "carry it over the tool tray", YAW),
        Waypoint(1.5, (tx, ty, TROWEL_DROP_Z), gap(32), "lower it onto the tray floor", YAW),
        Waypoint(0.3, (tx, ty, TROWEL_DROP_Z), gap(32), "settle", YAW),
        Waypoint(1.5, (tx, ty, TROWEL_DROP_Z), gap(52), "let go (52 mm)", YAW),
        Waypoint(1.0, (tx, ty, UP), gap(52), "straight up", YAW),
        Waypoint(1.5, RIGHT_REST, OPEN, "move back, out of the photo, fully open", 0.0),
    ]


def _left() -> list[Waypoint]:
    """Left arm: take the scale from the kit box, wait beside the cut until the trowel is gone, lay the scale on the
    cut floor beside the find, let go, move back."""
    kx, ky = L.SCALE_X, L.SCALE_Y
    sx, sy = SCALE_PLACE
    return [
        Waypoint(2.5, (kx, ky, UP), gap(34), "over the scale, fingers to 34 mm", 0.0),
        Waypoint(1.5, (kx, ky, SCALE_PICK_Z), gap(34), "down round the scale", 0.0),
        Waypoint(0.5, (kx, ky, SCALE_PICK_Z), gap(34), "settle", 0.0),
        Waypoint(1.5, (kx, ky, SCALE_PICK_Z), gap(12), "grip the scale (12 mm on 20)", 0.0),
        Waypoint(0.3, (kx, ky, SCALE_PICK_Z), gap(12), "settle the grip", 0.0),
        Waypoint(1.5, (kx, ky, CROSS), gap(12), "lift it out of the kit", 0.0),
        Waypoint(2.0, (*LEFT_WAIT, CROSS), gap(12), "beside the cut", 0.0),
        Waypoint(2.2, (*LEFT_WAIT, CROSS), gap(12), "wait until the trowel is over the tray", 0.0),
        Waypoint(2.5, (sx, sy, CROSS), gap(12), "over the scale's place, turned parallel to the find", YAW),
        Waypoint(1.5, (sx, sy, SCALE_ENTRY_Z), gap(12), "down to 25 mm above the cut floor", YAW),
        Waypoint(2.0, (sx, sy, SCALE_PLACE_Z), gap(12), "slowly onto the cut floor", YAW),
        Waypoint(0.5, (sx, sy, SCALE_PLACE_Z), gap(12), "settle", YAW),
        Waypoint(1.5, (sx, sy, SCALE_PLACE_Z), gap(34), "let go (34 mm: the find-side pad stays 15 mm off)", YAW),
        Waypoint(0.3, (sx, sy, SCALE_PLACE_Z), gap(34), "settle", YAW),
        Waypoint(1.5, (sx, sy, CROSS), gap(34), "straight up", YAW),
        Waypoint(1.5, (sx, sy, CROSS), OPEN, "open fully, clear of everything", YAW),
        Waypoint(2.0, LEFT_REST, OPEN, "move back, out of the photo", 0.0),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
