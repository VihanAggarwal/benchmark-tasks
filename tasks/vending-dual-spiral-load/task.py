"""On the station table: a vending-machine spiral tray with two neighbouring spirals, fixed in front of the left arm
(a light-grey steel plate 260 x 200 x 20 mm at x 0.16-0.42, y 0.04-0.24, two 30 mm side walls along x, and each
spiral's coil modelled as five black 12 mm steel ribs across the shelf, 62 mm apart, lined up across both spirals
with a 24 mm slot between the spirals). A loose white HDPE lane divider (230 x 20 x 40 mm, 175 g) stands in the
slot. A grey rubber parts mat (260 x 60 x 5 mm, fixed) lies on the midline, and four wide product cartons
(140 x 30 x 70 mm, 90 g) stand on the table to the left of the tray, 140 mm side along y. Make the two spirals one
dual-spiral selection: divider out onto the mat, then one carton standing across both spirals in each of the four
pitches, the front pitch filled, nothing riding a rib.

The scorer reads the arms (TCP, pad tips, gripper opening) through rigstate.py, which wraps the embodiment's step
and reset from this file's import; it changes nothing the embodiment does. A box or the divider still held by a
gripper at the end neither counts nor zeroes.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the task needs only one; the demo uses the left arm and the right arm holds its ready pose.
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
from scoring import OBJECTS as SCORED, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Make the two spirals one wide selection: lift the divider out from between them and set it on the grey mat. "
    "Then load the four wide boxes into the selection: one box in each of the four pitches, standing upright on "
    "the shelf across both spirals, between the coil loops and not on them. The front pitch (nearest you) must "
    "hold a box, and no pitch may be left empty in front of a filled one."
)

TRAY = Box("spiral_tray", pos=(*L.TRAY_CENTRE, L.TOP + L.TRAY_H / 2), size=L.TRAY_SIZE, color=L.STEEL_GREY)
WALLS = [
    Box(name, pos=(L.TRAY_CENTRE[0], y, L.SHELF + L.WALL_H / 2), size=L.WALL_SIZE, color=L.STEEL_GREY)
    for name, y in L.WALLS
]
RIBS = [
    Box(f"rib_{side}{k + 1}", pos=(x, (y0 + y1) / 2, L.SHELF + L.RIB_H / 2), size=(L.RIB_T, y1 - y0, L.RIB_H),
        color=L.COIL_BLACK)
    for side, (y0, y1) in L.RIB_SPANS.items()
    for k, x in enumerate(L.RIB_XS)
]
DIVIDER = Box("divider", pos=(L.DIV_X, L.DIV_Y, L.SHELF + L.DIV_SIZE[2] / 2), size=L.DIV_SIZE, mass=L.DIV_MASS,
              color=L.HDPE_WHITE)
MAT = Box("parts_mat", pos=(L.MAT_X, L.MAT_Y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_GREY)
BOXES = [
    Box(name, pos=(x, L.BOX_Y, L.TOP + L.BOX_SIZE[2] / 2), size=L.BOX_SIZE, mass=L.BOX_MASS, color=L.CARTON_RED)
    for name, x in zip(L.BOX_NAMES, L.BOX_XS)
]
OBJECTS = [TRAY, *WALLS, *RIBS, DIVIDER, MAT, *BOXES]

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


def _scored(objects) -> dict:
    return {n: objects[n] for n in SCORED}


def success(objects) -> bool:
    """Divider free and at rest on the mat, and pitches 1-4 each holding a seated box, nothing jammed and no part of
    the divider in the selection; held for 1 s. This runs first every control step, so it also feeds the tracker
    (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(_scored(objects), arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any box is free, at rest and inside the selection but not seated (a jam), or any part of the divider is
    inside the selection. Else 0.2 for the divider free and at rest on the mat, plus 0.2 for each pitch seated in an
    unbroken run from the front pitch: 0, 0.2, 0.4, 0.6, 0.8 or 1.0."""
    return _TRACKER.result(_scored(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
DIV_GRIP = -(L.DIV_SIZE[1] - 0.008) / 2  # yaw 0: the fingers close along y, across the divider's 20 mm
BOX_PRESET = -0.050 / 2  # a 50 mm gap on the way down to a box: 10 mm each side of its 30 mm
BOX_GRIP = -(L.BOX_SIZE[0] - 0.008) / 2  # yaw -90 deg: the fingers close along x, across a box's 30 mm
BOX_RELEASE = -0.046 / 2  # let a box go with a 46 mm gap: 8 mm each side, clear of the neighbouring boxes
YAW_DIV = 0.0
YAW_BOX = -math.pi / 2  # the left arm reaches the midline side with yaw -pi/2 (joint6 stays inside its limit)
PAD_TIPS = 0.013  # the pad tips are 13 mm past the TCP
PAD_HALF_WIDTH = 0.010  # each pad is 20 mm wide across the closing direction (along x at yaw 0)

# The divider: gripped over the middle of pitch 2 (x 0.257, 28 mm nearer the arms than its centre), where no rib
# lies under the pads (x 0.247-0.267, 17 mm from the rib faces at x 0.230 and 0.284), with the pad tips 45 mm above
# the table (15 mm below its top, 25 mm above the pitch's floor). Off centre by 28 mm on a 230 mm, 175 g bar: a
# torque of 0.05 N m about the closing axis.
DIV_GRIP_X = L.PITCH_X[1]  # 0.257
DIV_GRIP_DX = DIV_GRIP_X - L.DIV_X  # -0.028: the TCP's x minus the divider's centre's x, kept through the carry
DIV_TCP = L.TOP + 0.045 + PAD_TIPS  # 58 mm
DIV_HOVER = L.TOP + 0.120  # its bottom then 82 mm above the table, clear of the 50 mm walls
DIV_UP = DIV_TCP - L.SHELF  # the TCP is 38 mm above the divider's bottom
MAT_TCP = L.MAT_TOP + 0.003 + DIV_UP  # 46 mm: let go with its bottom 3 mm above the mat

# The boxes: gripped at mid-height, the pad tips 25 mm above the box's bottom.
BOX_UP = 0.025 + PAD_TIPS  # the TCP is 38 mm above the box's bottom
BOX_TCP = L.TOP + BOX_UP  # 38 mm
# Carried at 12.8 cm: the box's bottom 90 mm up, above the 70 mm supply boxes and the 50 mm walls, level with the
# tops of placed boxes (90 mm), which a carried box never passes over (it moves along x at y 0.34, then along y).
BOX_HOVER = L.TOP + 0.128
BOX_ENTRY = L.TOP + 0.090  # the open pads 7 mm above a supply box's top: the last 52 mm are lowered slowly
# GPU tuning knobs (see README 'Things to check on the GPU'; trace_trim.py reads a demo run's TASK_TRACE and prints
# what to set). RELEASE_DROP: the box's bottom above the shelf floor when the pads open; RELEASE_S: how long they
# take to open. If a box tips on release, open more slowly (RELEASE_S 2.0) or let go lower (RELEASE_DROP 0.001 to
# 0.002; let go 1 to 3 mm above the floor, a box that does not drop at all already has its bottom inside the
# 19-23 mm seated band).
RELEASE_DROP = 0.003
RELEASE_S = 1.5
PLACE_TCP = L.SHELF + RELEASE_DROP + BOX_UP  # 61 mm: let go with the box's bottom 3 mm above the shelf floor
PLACE_ENTRY = PLACE_TCP + 0.025  # the carried box's bottom 16 mm above the rib tops: the last 25 mm slowly
# PLACE_TRIM moves the TCP's x over every pitch. Before the scored GPU demo, run the demo once, read where each box
# stood in its pitch just before the pads opened (trace_trim.py prints it) and set PLACE_TRIM to cancel the median
# error (a pitch has 12 mm of play either side of a centred 30 mm box; pads have been seen about 8 mm off the
# commanded point in other tasks).
PLACE_TRIM = 0.0


def _divider() -> list[Waypoint]:
    """Lift the divider straight up out of its slot, gripped over pitch 2, carry it to the mat and set it down
    there."""
    dx, dy = DIV_GRIP_X, L.DIV_Y
    mx, my = L.MAT_X + DIV_GRIP_DX, L.MAT_Y  # the divider's centre comes down over the mat's centre
    y = YAW_DIV
    return [
        Waypoint(2.0, (dx, dy, DIV_HOVER), OPEN, "over the divider", y),
        Waypoint(1.0, (dx, dy, DIV_TCP + 0.020), OPEN, "down to 2 cm above the grip", y),
        Waypoint(1.5, (dx, dy, DIV_TCP), OPEN, "the last 2 cm slowly", y),
        Waypoint(0.6, (dx, dy, DIV_TCP), OPEN, "settle", y),
        Waypoint(1.5, (dx, dy, DIV_TCP), DIV_GRIP, "grip the divider", y),
        Waypoint(1.5, (dx, dy, DIV_HOVER), DIV_GRIP, "lift it straight out of the slot", y),
        Waypoint(2.0, (mx, my, DIV_HOVER), DIV_GRIP, "carry it over the mat", y),
        Waypoint(1.0, (mx, my, MAT_TCP + 0.020), DIV_GRIP, "down to 2 cm above", y),
        Waypoint(1.0, (mx, my, MAT_TCP), DIV_GRIP, "the last 2 cm slowly", y),
        Waypoint(0.6, (mx, my, MAT_TCP), DIV_GRIP, "settle", y),
        Waypoint(1.5, (mx, my, MAT_TCP), OPEN, "let go on the mat", y),
        Waypoint(1.0, (mx, my, DIV_HOVER), OPEN, "up", y),
    ]


def _box(k: int, first: bool) -> list[Waypoint]:
    """Box k (0..3) from the supply line into pitch k: front pitch first. The fingers close along x. On the way to
    the box the fingers go to a 50 mm gap (from fully open after the divider, from the 46 mm release after a box),
    so the open pads come down 10 mm clear of the box's sides and 30 mm clear of the next box in the line."""
    sx, sy = L.BOX_XS[k], L.BOX_Y
    px, py = L.PITCH_X[k] + PLACE_TRIM, L.LANE_Y
    y = YAW_BOX
    name = L.BOX_NAMES[k]
    return [
        Waypoint(2.5 if first else 2.0, (sx, sy, BOX_HOVER), BOX_PRESET, f"over {name}, fingers to 50 mm", y),
        Waypoint(0.5, (sx, sy, BOX_HOVER), BOX_PRESET, "settle", y),
        Waypoint(1.0, (sx, sy, BOX_ENTRY), BOX_PRESET, "down to just above it", y),
        Waypoint(1.5, (sx, sy, BOX_TCP), BOX_PRESET, "down beside it slowly", y),
        Waypoint(0.6, (sx, sy, BOX_TCP), BOX_PRESET, "settle", y),
        Waypoint(1.5, (sx, sy, BOX_TCP), BOX_GRIP, f"grip {name}", y),
        Waypoint(1.5, (sx, sy, BOX_HOVER), BOX_GRIP, "lift", y),
        Waypoint(1.0, (px, sy, BOX_HOVER), BOX_GRIP, "along x, level with the pitch", y),
        Waypoint(2.5, (px, py, BOX_HOVER), BOX_GRIP, f"along y over pitch {k + 1}", y),
        Waypoint(0.6, (px, py, BOX_HOVER), BOX_GRIP, "settle", y),
        Waypoint(1.0, (px, py, PLACE_ENTRY), BOX_GRIP, "down to just above the ribs", y),
        Waypoint(1.0, (px, py, PLACE_TCP), BOX_GRIP, "the last 25 mm slowly", y),
        Waypoint(0.6, (px, py, PLACE_TCP), BOX_GRIP, "settle", y),
        Waypoint(RELEASE_S, (px, py, PLACE_TCP), BOX_RELEASE, f"let go in pitch {k + 1}", y),
        Waypoint(1.2, (px, py, BOX_HOVER), BOX_RELEASE, "straight up", y),
    ]


def _left() -> list[Waypoint]:
    plan = _divider()
    for k in range(len(L.BOX_NAMES)):
        plan += _box(k, first=(k == 0))
    plan.append(Waypoint(1.5, plan[-1].pos, OPEN, "open fully", YAW_BOX))
    return plan


def demo():
    """Left arm only: the right arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose,
    a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
