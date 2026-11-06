"""On the station table: a bench-scale lathe mock with its spindle axis along y, 25 cm in front of the arm bases. On
the left (+y) a grey chuck block holds a 50 mm square aluminium workpiece whose face, toward the midline, carries a
black square marking the bore's mouth. On the midline, facing it, stands a dark steel tool block (50 x 80 x 40 mm)
with two walls on top forming a 26 mm channel along y. On a black mat on the right lies the boring bar: a 16 mm
square steel bar, 150 mm long, with a black head and a gold insert on top at its tip, lying along x with the
tip forward. Rule: the bar goes into the channel the right way round (tip toward the chuck, insert up) with its tip
sticking out past the block's chuck-side face by 45-64 mm: the 40 mm bore depth plus 5 mm, and at most four times
the 16 mm shank; it must not touch the part.

The bar is a loose mechkit compound (mechkit.py, next to this file): one rigid body
made of three boxes, so its tip end is visible and the scorer can read its pose (base frame and orientation).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the task needs only one; the demo uses the right arm and the left arm holds its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the bar is built as a compound body)
from scoring import Tracker, ends  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The arm sags about 7 mm under the 0.30 kg bar (J K^-1 J^T of its weight at
# the place pose; joints 80/80/80/10/10/10 N m/rad), so the bar comes to rest on the channel floor while the slow
# descent is still finishing and success starts counting (after the 0.5 s needed to see it at rest) with the gripper
# still closed. Settling, letting go and lifting take 0.5 + 1.5 + 1.0 s, so with 2.5 s the trial ends with the
# fingers open and the hand going up. Offline (no sag) the bar is let go 2 mm above the floor.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Set the boring bar in the tool block to bore the 40 mm deep hole: lay it in the block's channel, gold insert up "
    "and its tip toward the chuck, sticking out past the block's face far enough to reach the full depth with 5 mm "
    "to spare (at least 45 mm) but no more than four times its 16 mm shank (64 mm). Keep the tip off the part."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


TOOL_BLOCK = _span_box("tool_block", L.BLOCK_X0, L.BLOCK_X1, L.BLOCK_Y0, L.FACE_Y, 0.0, L.FLOOR_Z, L.BLOCK_STEEL)
WALL_NEAR = _span_box("channel_wall_near", L.BLOCK_X0, L.CHANNEL_X0, L.BLOCK_Y0, L.FACE_Y, L.FLOOR_Z, L.WALL_TOP,
                      L.BLOCK_STEEL)
WALL_FAR = _span_box("channel_wall_far", L.CHANNEL_X1, L.BLOCK_X1, L.BLOCK_Y0, L.FACE_Y, L.FLOOR_Z, L.WALL_TOP,
                     L.BLOCK_STEEL)
CHUCK = Box("chuck", pos=(L.CHUCK_CENTER[0], L.CHUCK_CENTER[1], L.TOP + L.CHUCK_CENTER[2]), size=L.CHUCK_SIZE,
            color=L.CHUCK_GREY)
WORKPIECE = Box("workpiece", pos=(L.WORK_CENTER[0], L.WORK_CENTER[1], L.TOP + L.WORK_CENTER[2]), size=L.WORK_SIZE,
                color=L.ALUMINIUM)
BORE_MARK = Box("bore_mark", pos=(L.MARK_CENTER[0], L.MARK_CENTER[1], L.TOP + L.MARK_CENTER[2]), size=L.MARK_SIZE,
                color=L.MARK_BLACK)
TOOL_MAT = Box("tool_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_BLACK)
BAR = mk.assembly(
    "boring_bar",
    origin=(L.BAR_START[0], L.BAR_START[1], L.TOP + L.BAR_START[2]),
    yaw=L.BAR_START_YAW,
    mass=L.BAR_MASS,
    color=L.STEEL,
    shapes=[
        mk.box(L.SHANK_CENTER, L.SHANK_SIZE),
        mk.box(L.HEAD_CENTER, L.HEAD_SIZE, color=L.HEAD_BLACK),
        mk.box(L.INSERT_CENTER, L.INSERT_SIZE, color=L.GOLD),
    ],
)
OBJECTS = [TOOL_BLOCK, WALL_NEAR, WALL_FAR, CHUCK, WORKPIECE, BORE_MARK, TOOL_MAT, BAR]

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


def success(objects) -> bool:
    """The bar lies in the channel, insert up, tip toward the chuck, sticking out 45-64 mm, and has been at rest
    (neither end moving more than 2 mm in 0.5 s); held 2.5 s. This runs first every control step, so it also feeds
    the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    bar = objects["boring_bar"]
    result = _TRACKER.update(bar)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        tip, back, _ = ends(bar)
        row.update(source=getattr(bar, "source", None), bar_origin=bar.origin, bar_quat=bar.quat_xyzw,
                   bar_vel=bar.lin_vel, tip=tip, back=back)
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the bar ends at rest lying in the channel: insert up (tilted 5 degrees or less), along it (within 10
    degrees), at least 20 mm of it over the block, its centre line within 6 mm of the channel's centre and its
    underside within 3 mm of the floor at both ends of the block; 0 also if the tip points away from the chuck or
    sticks out more than 72 mm (within 3 mm of the part). Otherwise by the stick-out s: 1.0 for 45 <= s <= 64 mm;
    0.5 for 38 <= s < 45 or 64 < s <= 72 mm; 0.25 below 38 mm. Read from the state at hand (the last step)."""
    return _TRACKER.result(objects["boring_bar"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
BAR_OPEN = -0.025  # 50 mm: 17 mm either side of the 16 mm bar
BAR_GRIP = -(L.BAR_W - 0.008) / 2  # 8 mm commanded across the bar's 16 mm: about 20 N per finger
# Picked where it lies along x with the fingers closing along y (yaw 0), turned a quarter turn in the hand while it
# is carried, so the tip swings from +x to +y (toward the chuck), and set down along y with the fingers closing along
# x: the right arm's yaw +pi/2 (the other sign runs joint6 into its limit near the midline, and would also turn the
# tip the wrong way).
YAW_PICK, YAW_PLACE = 0.0, math.pi / 2
# The bar is held 35 mm back from its tip, on the shank just behind the black head: set with 55 mm sticking out,
# that grip is 20 mm past the block's face, so the fingers (about 18 mm wide at the pad tips) come down 11 mm clear
# of it and the bar's whole length behind them can drop into the channel. The bar's centre of mass is 40 mm behind
# the grip: 0.12 N m about the fingers' closing axis, under half the estimated 0.28 N m that the pads' friction holds
# (20 N per pad, friction about 1.2).
GRIP_FROM_TIP = 0.035
S_CMD = 0.055  # the stick-out aimed at: 10 mm inside the band's lower end (45 mm), 9 mm inside its upper end
GRIP_UP = 0.017  # the TCP 17 mm above the bar's underside: pad tips 4 mm above it (and the mat), 12 mm held
DROP = 0.002  # the bar is let go with its underside 2 mm above the channel floor (the arm sags about 7 mm under it)
APPROACH = 0.030  # each final descent starts this far above the grip height
HOVER = L.TOP + 0.10  # carried with the bar's underside 83 mm up: 31 mm over the channel walls
PICK = (L.BAR_START[0] + L.BAR_LEN / 2 - GRIP_FROM_TIP, L.BAR_START[1], L.TOP + L.MAT_TOP + GRIP_UP)
# Trim of the place point (x, y), for the GPU run: if the bar lands off its aim (s_mm in TASK_TRACE away from 55, or
# centre_off_mm away from 0), move the place point by minus the offset. The compliance model expects about
# (-1.3, +0.5) mm at the place pose, well inside the channel's 5 mm of play and the band's 9 mm.
PLACE_TRIM = (0.0, 0.0)
PLACE = (L.AXIS_X + PLACE_TRIM[0], L.FACE_Y + S_CMD - GRIP_FROM_TIP + PLACE_TRIM[1],
         L.TOP + L.FLOOR_Z + DROP + GRIP_UP)
PARK = (0.22, -0.20, L.TOP + 0.15)


def _right() -> list[Waypoint]:
    """Right arm: grip the bar near its tip, lift it, carry it over the channel while turning it a quarter turn,
    lower it slowly into the channel, let go, go up and aside."""
    px, py, pz = PICK
    qx, qy, qz = PLACE
    return [
        Waypoint(2.0, (px, py, HOVER), BAR_OPEN, "over the bar, 35 mm back from its tip", YAW_PICK),
        Waypoint(1.0, (px, py, pz + APPROACH), BAR_OPEN, "down to just above the bar", YAW_PICK),
        Waypoint(1.5, (px, py, pz), BAR_OPEN, "slowly down beside the bar", YAW_PICK),
        Waypoint(0.5, (px, py, pz), BAR_OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (px, py, pz), BAR_GRIP, "grip the bar", YAW_PICK),
        Waypoint(1.5, (px, py, HOVER), BAR_GRIP, "lift the bar", YAW_PICK),
        Waypoint(2.5, (qx, qy, HOVER), BAR_GRIP, "carry it over the channel, turning the tip toward the chuck",
                 YAW_PLACE),
        Waypoint(0.8, (qx, qy, HOVER), BAR_GRIP, "settle over the channel", YAW_PLACE),
        Waypoint(1.0, (qx, qy, qz + APPROACH), BAR_GRIP, "down to 3 cm above", YAW_PLACE),
        Waypoint(2.0, (qx, qy, qz), BAR_GRIP, "slowly lay the bar in the channel", YAW_PLACE),
        Waypoint(0.5, (qx, qy, qz), BAR_GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (qx, qy, qz), BAR_OPEN, "let go of the bar", YAW_PLACE),
        Waypoint(1.0, (qx, qy, HOVER), BAR_OPEN, "straight up", YAW_PLACE),
        Waypoint(1.5, PARK, BAR_OPEN, "aside", YAW_PLACE),
    ]


def demo():
    """One arm: the left arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
