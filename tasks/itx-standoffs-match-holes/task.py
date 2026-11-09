"""On the station table: a light grey steel case tray (180 x 200 x 3 mm, fixed) on the midline 30 cm in front of the
arm bases, with seven threaded holes drawn as dark discs 28 mm across. Brass standoffs (18 mm across, 20 mm tall)
stand loose on five of them, as a case ships for a larger board: on the mini-ITX board's rear holes H1 and H2, on A1
and A2 (inside the outline the mini-ITX board covers, where it has no hole) and on A3 (outside that outline). The
mini-ITX board (88 x 88 mm, about half scale, with a low CPU cooler at its centre and its I/O block along one edge) lies
on a pink anti-static mat on the right, I/O block to the rear. A black parts dish is on the left. Rule: before the
board goes in, a standoff under each of its four holes (H1-H4), none anywhere else under it and none loose on the
tray (a spare may stay standing on A3, clear of the board, or go to the dish); then the board laid on them, each
hole over its own standoff (which puts the I/O block at the rear: the hole pattern is not a rectangle).

The board is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of the board, the cooler,
the I/O block and four hole pads, so the scorer can read its pose (base frame and orientation).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the board is a compound)
from scoring import BOARD, Tracker, board_holes, standoff_axis  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 3 s, not the default 1 s. The arm sags a few mm under the 0.15 kg board, so the board comes to rest on
# the standoffs while the slow descent is still finishing, and success starts counting (after the 0.5 s needed to see
# it at rest) with the gripper still closed on the cooler. Settling and letting go take 0.8 + 1.5 s, so with 3 s the
# trial ends with the fingers open (they leave the 40 mm cooler about 0.2 s after they start to open).
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Get the case tray ready for the mini-ITX board and mount it: a standoff under each of the board's four holes, "
    "none anywhere else under the board and none loose on the tray (leave a spare on a hole clear of the board or "
    "put it in the parts dish), then lay the board on them with each hole over its standoff and its I/O block at "
    "the rear."
)

CASE_TRAY = Box("case_tray", pos=(L.CASE_X, L.CASE_Y, L.TOP + L.TRAY_TOP / 2), size=L.CASE_SIZE, color=L.TRAY_GREY)
HOLE_DISCS = [
    Cylinder(f"hole_{k.lower()}", pos=(x, y, L.TOP + L.TRAY_TOP + L.DISC_H / 2), radius=L.DISC_R, height=L.DISC_H,
             color=L.HOLE_DARK)
    for k, (x, y) in L.HOLES.items()
]
STANDOFF_PARTS = [
    Cylinder(name, pos=(*L.HOLES[L.STANDOFF_START[name]], L.TOP + L.DISC_TOP + L.STANDOFF_H / 2),
             radius=L.STANDOFF_R, height=L.STANDOFF_H, mass=L.STANDOFF_MASS, color=L.BRASS)
    for name in L.STANDOFFS
]
MAT = Box("antistatic_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_TOP / 2), size=L.MAT_SIZE, color=L.MAT_PINK)
DISH = Box("parts_dish", pos=(L.DISH_X, L.DISH_Y, L.TOP + L.DISH_SIZE[2] / 2), size=L.DISH_SIZE, color=L.DISH_BLACK)
BOARD_PART = mk.assembly(
    BOARD,
    origin=(L.BOARD_START[0], L.BOARD_START[1], L.TOP + L.MAT_TOP),
    mass=L.BOARD_MASS,
    color=L.PCB_GREEN,
    shapes=[
        mk.box((0.0, 0.0, L.BOARD_T / 2), (L.BOARD_LEN, L.BOARD_LEN, L.BOARD_T)),
        mk.box(L.COOLER_CENTER, L.COOLER_SIZE, color=L.COOLER_BLACK),
        mk.box(L.IO_CENTER, L.IO_SIZE, color=L.IO_SILVER),
        *[mk.cylinder((hx, hy, L.BOARD_T + L.PAD_H / 2), L.PAD_R, L.PAD_H, color=L.PAD_GOLD)
          for hx, hy in L.BOARD_HOLES.values()],
    ],
)
OBJECTS = [CASE_TRAY, *HOLE_DISCS, MAT, DISH, *STANDOFF_PARTS, BOARD_PART]

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
    return {n: objects.get(n) for n in (*L.STANDOFFS, BOARD)}


def success(objects) -> bool:
    """A standoff standing on each of the board's four holes, none elsewhere in the case but on A3, and the board flat
    on the four with each hole over its own standoff, everything at rest; held 3 s. This runs first every control
    step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = _parts(objects)
    result = _TRACKER.update(parts)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        br = parts[BOARD]
        row.update(source=getattr(br, "source", None), board_origin=br.origin, board_quat=br.quat_xyzw,
                   board_holes=board_holes(br), standoff_bottom={n: standoff_axis(parts[n])[0] for n in L.STANDOFFS})
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any standoff or the board is off the table, or any standoff is left in the case (centre over the tray or
    less than 9 mm outside its edge) without standing on one of the board's four holes or on A3, the spare hole
    clear of the board: on A1 or A2 under the board, or loose on the tray or the board. Otherwise 0.2 for
    each of the four holes with a standoff standing on it (upright within 10 degrees, bottom within 3 mm of the tray
    and within 5 mm of the hole's centre, at rest), and 0.2 more when the board is mounted: level within 3 degrees,
    each hole within 7 mm of its own standoff's axis with its underside within 2 mm of that standoff's top, at rest.
    Read from the state at hand (the last step)."""
    return _TRACKER.result(_parts(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm: over the board's cooler
SO_OPEN = -0.025  # 50 mm: at the standoffs, so the open fingers stay well clear of the neighbouring standoffs
STANDOFF_GRIP = -(2 * L.STANDOFF_R - 0.008) / 2  # 10 mm commanded across an 18 mm standoff
COOLER_GRIP = -(L.COOLER_SIZE[1] - 0.008) / 2  # 32 mm commanded across the 40 mm cooler
PAD_TIP = 0.013  # the pad tips are 13 mm below the TCP in a straight-down grip
# A standoff is held with the pad tips 6 mm above its bottom (14 mm of the pads on it); the board by its cooler with
# the pad tips 8 mm above the board's top face (22 mm of the pads on the cooler).
SO_GRIP_UP = 0.006 + PAD_TIP  # TCP height above a standoff's bottom
BOARD_GRIP_UP = L.BOARD_T + 0.008 + PAD_TIP  # TCP height above the board's underside
SO_DROP = 0.0015  # a standoff is let go with its bottom 1.5 mm above the hole's disc
BOARD_DROP = 0.001  # the board with its underside 1 mm above the standoffs' tops (the arm sags a few mm under it)
APPROACH = 0.025  # each final descent starts this far above its end
HOVER_SO = L.TOP + 0.08  # standoffs carried with their bottoms 61 mm above the table
HOVER_BOARD = L.TOP + 0.10  # the board carried with its underside 76 mm above the table, 52 mm over the standoffs
YAW = 0.0  # every grip closes the fingers along y: across the cooler, clear of the I/O block at the board's +x edge
# Trim of the board's place point (x, y), for the GPU run. On the GPU the right arm set a 0.28 kg part down near
# x 0.30 about (-4.4, -4.2) mm off its aim (its sag toward its base); the trim takes off about half of that, scaled to
# the board's 0.15 kg. Each hole has 7 mm of tolerance: if the board lands off its aim (board_x_mm, board_y_mm and
# board_hole_offset_mm in TASK_TRACE), move the point by minus the offset. The 20 g standoffs need no trim.
BOARD_TRIM = (0.0012, 0.0011)
# Waits on one clock: the standoffs to move are only 70 mm apart across the midline, so the arms take turns. The right
# arm moves A2 to H4 and goes to the board; the left arm comes over A1 only once the right arm has left; the right
# arm brings the board over the tray only once the left arm has set A1 on H3 and gone back up.
LEFT_WAIT = 13.5
RIGHT_WAIT = 7.5
PARK_L = (0.20, 0.24, L.TOP + 0.12)


def _hole_point(k: str, z: float) -> tuple[float, float, float]:
    x, y = L.HOLES[k]
    return (x, y, z)


SO_PICK_Z = L.TOP + L.DISC_TOP + SO_GRIP_UP
SO_PLACE_Z = L.TOP + L.DISC_TOP + SO_DROP + SO_GRIP_UP
BOARD_PICK = (L.BOARD_START[0], L.BOARD_START[1], L.TOP + L.MAT_TOP + BOARD_GRIP_UP)
BOARD_PLACE = (L.MOUNT_X + BOARD_TRIM[0], L.MOUNT_Y + BOARD_TRIM[1],
               L.TOP + L.MOUNT_Z + BOARD_DROP + BOARD_GRIP_UP)


def _move_standoff(src: str, dst: str) -> list[Waypoint]:
    """Pick the standoff off hole ``src`` and stand it on hole ``dst``: come down beside it slowly, grip it, lift it,
    carry it over the other hole, lower it slowly, let go, go straight up."""
    (px, py, pz), (qx, qy, qz) = _hole_point(src, SO_PICK_Z), _hole_point(dst, SO_PLACE_Z)
    return [
        Waypoint(1.5, (px, py, HOVER_SO), SO_OPEN, f"over the standoff on {src}", YAW),
        Waypoint(1.0, (px, py, pz + APPROACH), SO_OPEN, f"down to just above the standoff on {src}", YAW),
        Waypoint(1.5, (px, py, pz), SO_OPEN, f"slowly down beside the standoff on {src}", YAW),
        Waypoint(0.5, (px, py, pz), SO_OPEN, "settle", YAW),
        Waypoint(1.5, (px, py, pz), STANDOFF_GRIP, f"grip the standoff on {src}", YAW),
        Waypoint(1.2, (px, py, HOVER_SO), STANDOFF_GRIP, f"lift it off {src}", YAW),
        Waypoint(1.2, (qx, qy, HOVER_SO), STANDOFF_GRIP, f"carry it over {dst}", YAW),
        Waypoint(0.5, (qx, qy, HOVER_SO), STANDOFF_GRIP, f"settle over {dst}", YAW),
        Waypoint(1.0, (qx, qy, qz + APPROACH), STANDOFF_GRIP, f"down to just above {dst}", YAW),
        Waypoint(1.5, (qx, qy, qz), STANDOFF_GRIP, f"slowly stand it on {dst}", YAW),
        Waypoint(0.5, (qx, qy, qz), STANDOFF_GRIP, "settle", YAW),
        Waypoint(1.5, (qx, qy, qz), SO_OPEN, f"let go on {dst}", YAW),
        Waypoint(1.0, (qx, qy, HOVER_SO), SO_OPEN, f"straight up from {dst}", YAW),
    ]


def _left() -> list[Waypoint]:
    """Left arm: aside until the right arm has left the tray, then the standoff on A1 to H3 (both inside the board's
    outline on its left), then aside."""
    plan = [
        Waypoint(2.0, PARK_L, SO_OPEN, "aside", YAW),
        Waypoint(LEFT_WAIT, PARK_L, SO_OPEN, "wait for the right arm to leave the tray", YAW),
    ]
    plan += _move_standoff("A1", "H3")
    plan.append(Waypoint(1.5, PARK_L, SO_OPEN, "aside again", YAW))
    return plan


def _right() -> list[Waypoint]:
    """Right arm: the standoff on A2 to H4; then the board from its mat by the cooler, held up over the mat until the
    left arm is done, carried over the tray, lowered slowly onto the four standoffs, let go; then up."""
    plan = _move_standoff("A2", "H4")
    px, py, pz = BOARD_PICK
    qx, qy, qz = BOARD_PLACE
    plan += [
        Waypoint(2.0, (px, py, HOVER_BOARD), OPEN, "over the board's cooler", YAW),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, "down to just above the cooler", YAW),
        Waypoint(1.5, (px, py, pz), OPEN, "slowly down beside the cooler", YAW),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", YAW),
        Waypoint(1.5, (px, py, pz), COOLER_GRIP, "grip the cooler", YAW),
        Waypoint(1.5, (px, py, HOVER_BOARD), COOLER_GRIP, "lift the board", YAW),
        Waypoint(RIGHT_WAIT, (px, py, HOVER_BOARD), COOLER_GRIP, "wait for the left arm to leave the tray", YAW),
        Waypoint(2.5, (qx, qy, HOVER_BOARD), COOLER_GRIP, "carry it over the tray, holes over the standoffs", YAW),
        Waypoint(0.8, (qx, qy, HOVER_BOARD), COOLER_GRIP, "settle over the standoffs", YAW),
        Waypoint(1.0, (qx, qy, qz + APPROACH), COOLER_GRIP, "down to just above the standoffs", YAW),
        Waypoint(2.5, (qx, qy, qz), COOLER_GRIP, "slowly lay it on the four standoffs", YAW),
        Waypoint(0.8, (qx, qy, qz), COOLER_GRIP, "settle", YAW),
        Waypoint(1.5, (qx, qy, qz), OPEN, "let go of the board", YAW),
        Waypoint(1.0, (qx, qy, HOVER_BOARD), OPEN, "straight up", YAW),
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
