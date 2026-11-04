"""On the station table: a pink wound mark (a thin 70 x 70 mm pad, fixed) on the midline 30 cm in front of the arm
bases, with a used, stained dressing (a 50 x 50 x 16 mm pad) lying on it; a blue sterile tray (140 x 70 mm inside,
15 mm walls, fixed) 22 cm to the left of the midline holding a white clean pad of the same size; and a yellow waste
bin (90 x 90 mm inside, 40 mm walls, fixed) 24 cm to the right. Clean hand / dirty hand rule: only the right gripper
may handle the used dressing and it must keep 30 mm away from the clean pad; only the left gripper may handle the
clean pad and it must never touch the used dressing; the clean pad touches nothing but its tray, the wound mark and
the left gripper, and comes over the wound mark only once the used dressing is in the bin.

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
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with both grippers let go (each open, opening >= 0.9, or its finger zone > 10 mm from its item).
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Use only the right gripper for the stained used dressing and only the left gripper for the white clean pad, and "
    "keep the right gripper away from the clean pad. Put the used dressing in the yellow waste bin, then put the clean "
    "pad on the pink wound mark. Apart from the left gripper, the clean pad may touch only its blue tray and the "
    "wound mark, and it must never pass over or under the used dressing."
)


def _walls(prefix: str, x: float, y: float, inner_x: float, inner_y: float, wall: float, height: float,
           color) -> list[Box]:
    """Four fixed walls around an ``inner_x`` x ``inner_y`` opening centred at (x, y); the table is the floor."""
    z = L.TOP + height / 2
    ox, oy = inner_x / 2 + wall / 2, inner_y / 2 + wall / 2
    return [
        Box(f"{prefix}_wall_front", pos=(x + ox, y, z), size=(wall, inner_y + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_back", pos=(x - ox, y, z), size=(wall, inner_y + 2 * wall, height), color=color),
        Box(f"{prefix}_wall_left", pos=(x, y + oy, z), size=(inner_x, wall, height), color=color),
        Box(f"{prefix}_wall_right", pos=(x, y - oy, z), size=(inner_x, wall, height), color=color),
    ]


WOUND = Box("wound_mark", pos=(L.MARK_X, L.MARK_Y, L.TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE, color=L.PINK)
DRESSING = Box(
    "used_dressing",
    pos=(L.DRESSING_X, L.DRESSING_Y, L.MARK_TOP + L.DRESSING_SIZE[2] / 2),
    size=L.DRESSING_SIZE,
    mass=L.DRESSING_MASS,
    color=L.STAINED,
)
TRAY = _walls("tray", L.TRAY_X, L.TRAY_Y, L.TRAY_INNER_X, L.TRAY_INNER_Y, L.TRAY_WALL, L.TRAY_HEIGHT, L.TRAY_BLUE)
PAD = Box(
    "clean_pad",
    pos=(L.TRAY_X, L.TRAY_Y, L.TOP + L.PAD_SIZE[2] / 2),
    size=L.PAD_SIZE,
    mass=L.PAD_MASS,
    color=L.WHITE,
)
BIN = _walls("bin", L.BIN_X, L.BIN_Y, L.BIN_INNER, L.BIN_INNER, L.BIN_WALL, L.BIN_HEIGHT, L.BIN_YELLOW)
OBJECTS = [WOUND, DRESSING, *TRAY, PAD, *BIN]

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
    """Used dressing in the bin, clean pad flat on the wound mark within 10 mm of its centre, both grippers let go,
    no clean/dirty breach at any time; held still 1 s. This runs first every control step, so it also feeds the
    tracker (with the arms rigstate read for this step)."""
    dressing, pad = objects["used_dressing"], objects["clean_pad"]
    arms = rigstate.arms()
    result = _TRACKER.update(dressing, pad, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details, "dressing_pos": dressing.pos, "pad_pos": pad.pos}
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any moment, the left gripper came within 5 mm of the used dressing, the right gripper came within
    30 mm of the clean pad, the clean pad touched the used dressing (or passed over or under it), the bare table or
    the bin, or came over the wound mark before the dressing was in the bin; or if the used dressing ends off the
    table. Else 1.0 dressing in the bin, pad flat on the mark within 10 mm, both grippers let go (each open, or its
    finger zone more than 10 mm from its item); 0.75 dressing in the bin, pad flat on the mark within 25 mm; 0.5
    dressing in the bin; 0.25 dressing off the wound mark but not in the bin; 0 otherwise."""
    return _TRACKER.result(objects["used_dressing"], objects["clean_pad"], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.PAD_SIZE[0] - 0.008) / 2  # across the 50 mm (both pads are 50 x 50 mm)
# Both grips close the fingers along x (yaw -pi/2 left, +pi/2 right, the signs that reach the midline), so the open
# fingers point along x, not toward the other arm, and fit the tray's 140 mm length with 15 mm to each end wall.
YAW_L = -math.pi / 2
YAW_R = math.pi / 2
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
HOVER = L.TOP + 0.10
CARRY_Z = L.TOP + 0.10  # a carried pad's bottom 83 mm above the table: clear of the bin's 40 mm and tray's 15 mm walls
GRIP_UP = 0.004  # both pads are gripped with the finger tips 4 mm above their bottom: 12 mm of the 16 mm side held
DRESSING_TCP_Z = L.MARK_TOP + GRIP_UP + PAD_TIPS  # tips 4 mm above the wound mark's top as they close over it
PAD_TCP_Z = L.TOP + GRIP_UP + PAD_TIPS  # inside the tray: the open fingers come down 15 mm inside its end walls
PLACE_TCP_Z = L.MARK_TOP + 0.002 + GRIP_UP + PAD_TIPS  # let go with the clean pad's bottom 2 mm above the mark
ENTRY_Z = PLACE_TCP_Z + 0.020  # the last 20 mm are lowered slowly
# Into the bin: the dressing's bottom 12 mm above the rim when it is let go, the opening fingers 16 mm over the rim.
BIN_TCP_Z = L.BIN_TOP_Z + 0.012 + GRIP_UP + PAD_TIPS
UP_R = L.TOP + 0.13  # the right arm waits up here over the bin, 25 cm from the wound mark


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _right() -> list[Waypoint]:
    """Right arm (dirty hand): lift the used dressing off the wound, carry it to the bin, let it drop in, stay up
    over the bin, well away from the clean pad."""
    mx, my = L.DRESSING_X, L.DRESSING_Y
    bx, by = L.BIN_X, L.BIN_Y
    return [
        Waypoint(1.5, (mx, my, HOVER), OPEN, "over the used dressing", YAW_R),
        Waypoint(1.5, (mx, my, DRESSING_TCP_Z), OPEN, "down beside it", YAW_R),
        Waypoint(0.5, (mx, my, DRESSING_TCP_Z), OPEN, "settle", YAW_R),
        Waypoint(1.5, (mx, my, DRESSING_TCP_Z), GRIP, "grip the used dressing", YAW_R),
        Waypoint(1.2, (mx, my, CARRY_Z), GRIP, "lift it off the wound", YAW_R),
        Waypoint(1.5, (bx, by, CARRY_Z), GRIP, "carry it over the bin", YAW_R),
        Waypoint(1.0, (bx, by, BIN_TCP_Z), GRIP, "lower it to just above the rim", YAW_R),
        Waypoint(0.5, (bx, by, BIN_TCP_Z), GRIP, "settle", YAW_R),
        Waypoint(1.5, (bx, by, BIN_TCP_Z), OPEN, "let it drop into the bin", YAW_R),
        Waypoint(1.0, (bx, by, UP_R), OPEN, "up, out of the way", YAW_R),
    ]


DRESSING_BINNED_S = _span(_right()[:9])  # the right gripper is fully open over the bin: the dressing has dropped in


def _left_pick() -> list[Waypoint]:
    px, py = L.TRAY_X, L.TRAY_Y
    return [
        Waypoint(1.5, (px, py, HOVER), OPEN, "over the clean pad", YAW_L),
        Waypoint(1.5, (px, py, PAD_TCP_Z), OPEN, "down into the tray beside it", YAW_L),
        Waypoint(0.5, (px, py, PAD_TCP_Z), OPEN, "settle", YAW_L),
        Waypoint(1.5, (px, py, PAD_TCP_Z), GRIP, "grip the clean pad", YAW_L),
        Waypoint(1.2, (px, py, CARRY_Z), GRIP, "lift it out of the tray", YAW_L),
    ]


def _left() -> list[Waypoint]:
    """Left arm (clean hand): take the clean pad out of the tray and hold it up over the tray until the right arm
    has let the used dressing drop into the bin, then set the pad on the wound mark, let go, go up."""
    px, py = L.TRAY_X, L.TRAY_Y
    mx, my = L.MARK_X, L.MARK_Y
    pick = _left_pick()
    wait = DRESSING_BINNED_S - _span(pick)
    return pick + [
        Waypoint(wait, (px, py, CARRY_Z), GRIP, "wait until the used dressing is in the bin", YAW_L),
        Waypoint(1.8, (mx, my, CARRY_Z), GRIP, "carry it over the wound mark", YAW_L),
        Waypoint(1.0, (mx, my, ENTRY_Z), GRIP, "down to 2 cm above", YAW_L),
        Waypoint(0.5, (mx, my, ENTRY_Z), GRIP, "settle", YAW_L),
        Waypoint(1.5, (mx, my, PLACE_TCP_Z), GRIP, "lower the last 2 cm slowly", YAW_L),
        Waypoint(0.5, (mx, my, PLACE_TCP_Z), GRIP, "settle", YAW_L),
        Waypoint(1.5, (mx, my, PLACE_TCP_Z), OPEN, "let go", YAW_L),
        Waypoint(1.0, (mx, my, HOVER), OPEN, "up", YAW_L),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
