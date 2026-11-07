"""On the station table: a bench-scale CNC tool magazine, a fixed grey round plate (270 mm across) centred 30 cm in
front of the arm bases on the midline, with eight square sockets (52 mm inside, 25 mm deep) on a 90 mm radius
circle every 45 degrees, numbered P1 (straight ahead, red tab) to P8 counter-clockwise seen from above. Three steel
tools (40 mm across, 70 mm tall) already stand in P1, P2 and P5. On a tray on the left stand two new small tools
with blue holders; on a tray on the right stands the face mill, a 40 mm blue shank under a 76 mm black cutter body.
Rule: the face mill is a large tool, so both pockets next to it must stay empty; the loaded tools stay where they
are. With P1, P2 and P5 loaded, only P7 has two empty neighbours, so the small tools go in P3 and P4.

The face mill is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of two cylinders,
so its wide cutter is visible and the scorer can read its pose (base frame and orientation).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the face mill is built as a compound body)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The arm sags a few mm under a 0.15 kg tool, so the last tool comes to rest
# on the plate while the slow descent is still finishing, and success starts counting (after the 0.5 s needed to
# see every tool at rest) with the gripper still closed. Settling, letting go and lifting take 0.5 + 1.5 + 1.0 s, so
# with 2.5 s the trial ends with the fingers open and the hand going up.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Load the three new tools from the trays into empty pockets of the tool magazine. The face mill is a large "
    "tool: both pockets next to it must stay empty, or it will hit its neighbours at the next tool change. Leave "
    "the tools already in the magazine where they are, and let go of the tools."
)

PLATE = Cylinder("magazine_plate", pos=(L.MAG_X, L.MAG_Y, L.TOP + L.DISC_H / 2), radius=L.DISC_R, height=L.DISC_H,
                 color=L.PLATE_GREY)


def _pocket_walls(k: int) -> list[Box]:
    """Pocket Pk's four fixed walls on the plate: a 52 x 52 mm square opening, sides parallel to x and y."""
    px, py = L.POCKETS[k]
    z = L.TOP + L.DISC_TOP + L.WALL_H / 2
    d = L.POCKET_IN / 2 + L.WALL_T / 2
    return [
        Box(f"p{k}_wall_front", pos=(px + d, py, z), size=(L.WALL_T, L.POCKET_OUT, L.WALL_H), color=L.WALL_DARK),
        Box(f"p{k}_wall_back", pos=(px - d, py, z), size=(L.WALL_T, L.POCKET_OUT, L.WALL_H), color=L.WALL_DARK),
        Box(f"p{k}_wall_left", pos=(px, py + d, z), size=(L.POCKET_IN, L.WALL_T, L.WALL_H), color=L.WALL_DARK),
        Box(f"p{k}_wall_right", pos=(px, py - d, z), size=(L.POCKET_IN, L.WALL_T, L.WALL_H), color=L.WALL_DARK),
    ]


WALLS = [w for k in range(1, L.N_POCKETS + 1) for w in _pocket_walls(k)]
TAB = Box("p1_tab", pos=(L.TAB_X, L.TAB_Y, L.TOP + L.DISC_TOP + L.TAB_SIZE[2] / 2), size=L.TAB_SIZE, color=L.TAB_RED)
LEFT_TRAY = Box("left_tray", pos=(L.LEFT_TRAY_X, L.LEFT_TRAY_Y, L.TOP + L.TRAY_H / 2), size=L.LEFT_TRAY_SIZE,
                color=L.TRAY_GREY)
RIGHT_TRAY = Box("right_tray", pos=(L.RIGHT_TRAY_X, L.RIGHT_TRAY_Y, L.TOP + L.TRAY_H / 2), size=L.RIGHT_TRAY_SIZE,
                 color=L.TRAY_GREY)
LOADED_TOOLS = [
    Cylinder(name, pos=(*L.POCKETS[k], L.TOP + L.DISC_TOP + L.TOOL_H / 2), radius=L.TOOL_R, height=L.TOOL_H,
             mass=L.TOOL_MASS, color=L.STEEL)
    for name, k in L.LOADED.items()
]
NEW_TOOLS = [
    Cylinder(name, pos=(*L.NEW_START[name], L.TOP + L.TRAY_TOP + L.TOOL_H / 2), radius=L.TOOL_R, height=L.TOOL_H,
             mass=L.TOOL_MASS, color=L.NEW_BLUE)
    for name in L.NEW_SMALL
]
FACE_MILL = mk.assembly(
    L.FACE,
    origin=(L.FM_START[0], L.FM_START[1], L.TOP + L.TRAY_TOP),
    mass=L.FM_MASS,
    color=L.NEW_BLUE,
    shapes=[
        mk.cylinder(L.SHANK_CENTER, L.SHANK_R, L.SHANK_H),
        mk.cylinder(L.BODY_CENTER, L.BODY_R, L.BODY_H, color=L.CUTTER_BLACK),
    ],
)
OBJECTS = [PLATE, *WALLS, TAB, LEFT_TRAY, RIGHT_TRAY, *LOADED_TOOLS, *NEW_TOOLS, FACE_MILL]

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


def _tools(objects) -> dict:
    return {n: objects.get(n) for n in L.TOOL_NAMES}


def success(objects) -> bool:
    """The face mill seated in P7 with P6 and P8 empty, the two small tools seated in P3 and P4, the three loaded
    tools still seated in P1, P2 and P5, nothing else on the magazine, every tool at rest; held 2.5 s. This runs
    first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    tools = _tools(objects)
    result = _TRACKER.update(tools)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        fm = tools[L.FACE]
        row.update(source=getattr(fm, "source", None), face_mill_origin=fm.origin, face_mill_quat=fm.quat_xyzw,
                   tool_pos={n: tools[n].pos for n in L.TOOL_NAMES if n != L.FACE})
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if P1, P2 or P5 does not hold one of the three loaded tools, seated, a tool lies loose on the magazine (on
    its side, on the walls, on another tool, still moving), or the face mill is seated next to a pocket holding a
    tool. Otherwise
    0.5 for the face mill seated with both neighbours empty plus 0.25 for each new small tool seated in P3 or P4.
    Read from the state at hand (the last step)."""
    return _TRACKER.result(_tools(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm
TOOL_GRIP = -(2 * L.TOOL_R - 0.008) / 2  # 32 mm commanded across a small tool's 40 mm
FM_GRIP = -(2 * L.BODY_R - 0.008) / 2  # 68 mm commanded across the face mill's 76 mm cutter body
PAD_TIP = 0.013  # the pad tips are 13 mm below the TCP in a straight-down grip
# A small tool is held with the pad tips 36 mm above its bottom (34 mm of the pads on it, the top 4 mm below the
# pads' back end); the face mill by its cutter body, pad tips 4 mm above the body's underside (26 mm on it), its
# centre of mass 46 mm above the shank's bottom, 2 mm above the pad tips, so it hangs level.
TOOL_GRIP_UP = 0.036 + PAD_TIP  # TCP height above the small tool's bottom
FM_GRIP_UP = L.SHANK_H + 0.004 + PAD_TIP  # TCP height above the face mill's shank bottom
DROP = 0.002  # each tool is let go with its bottom 2 mm above the plate (the arm sags a few mm under it)
APPROACH = 0.030  # each final descent starts this far above the grip or place height
# Carried with the tool's bottom 115 mm above the table: 35 mm over the loaded tools' tops (80 mm) and 80 mm over
# the socket walls; the face mill's shank bottom 87 mm up (its body's underside 127 mm).
HOVER_L = L.TOP + 0.115 + TOOL_GRIP_UP
HOVER_R = L.TOP + 0.087 + FM_GRIP_UP
# The grip yaw: 0 closes the fingers along y. At P3 and P4 the fingers close and open along the line from the
# magazine's centre through the pocket (radially), so the open pads (55 mm out from the axis) swing toward the empty
# middle of the ring and past the plate's rim, never toward the loaded tools next door. At P7 they close along x
# (around the ring): its neighbours P6 and P8 are empty, and this keeps the right arm's open pads 11 cm from the left
# arm's working at P3 (closing along y there, the pads of the two arms would be 7 cm apart).
YAW_P3, YAW_P4, YAW_P7 = 0.0, math.pi / 4, math.pi / 2
# Trim of each place point (x, y), for the GPU run: if a tool lands off its pocket's centre (centre_off_mm in
# TASK_TRACE beyond about 3 mm), move that place point by minus the offset. The sockets give 6 mm of play either way.
TRIM = {3: (0.0, 0.0), 4: (0.0, 0.0), 7: (0.0, 0.0)}
# The right arm waits over the face mill this long before it comes down for it, so that it is over P7 only while
# the left arm is away at its tray: both reaching in at x 0.30 together would bring the two elbows within a few mm.
RIGHT_WAIT = 8.0
PARK_L = (0.20, 0.16, L.TOP + 0.17)
PARK_R = (0.18, -0.22, L.TOP + 0.17)


def _place(k: int, grip_up: float) -> tuple[float, float, float]:
    px, py = L.POCKETS[k]
    tx, ty = TRIM[k]
    return (px + tx, py + ty, L.TOP + L.DISC_TOP + DROP + grip_up)


def _pick_place(pick, place, yaw_pick, yaw_place, grip, hover, what, where) -> list[Waypoint]:
    """One pick and place: come down beside the part slowly, grip it, lift it, carry it over the pocket, lower it
    slowly into the socket, let go, go straight up."""
    px, py, pz = pick
    qx, qy, qz = place
    return [
        Waypoint(2.0, (px, py, hover), OPEN, f"over the {what}", yaw_pick),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, f"down to just above the {what}", yaw_pick),
        Waypoint(1.5, (px, py, pz), OPEN, f"slowly down beside the {what}", yaw_pick),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", yaw_pick),
        Waypoint(1.5, (px, py, pz), grip, f"grip the {what}", yaw_pick),
        Waypoint(1.5, (px, py, hover), grip, f"lift the {what}", yaw_pick),
        Waypoint(2.0, (qx, qy, hover), grip, f"carry it over {where}", yaw_place),
        Waypoint(0.8, (qx, qy, hover), grip, f"settle over {where}", yaw_place),
        Waypoint(1.0, (qx, qy, qz + APPROACH), grip, "down to 3 cm above", yaw_place),
        Waypoint(2.0, (qx, qy, qz), grip, f"slowly lower it into {where}", yaw_place),
        Waypoint(0.5, (qx, qy, qz), grip, "settle", yaw_place),
        Waypoint(1.5, (qx, qy, qz), OPEN, f"let go in {where}", yaw_place),
        Waypoint(1.0, (qx, qy, hover), OPEN, "straight up", yaw_place),
    ]


def _left() -> list[Waypoint]:
    """Left arm: the nearer small tool into P3, then the other into P4, then aside."""
    a, b = L.NEW_START["new_tool_a"], L.NEW_START["new_tool_b"]
    pick_z = L.TOP + L.TRAY_TOP + TOOL_GRIP_UP
    plan = _pick_place((*a, pick_z), _place(3, TOOL_GRIP_UP), YAW_P3, YAW_P3, TOOL_GRIP, HOVER_L,
                       "small tool", "P3")
    plan += _pick_place((*b, pick_z), _place(4, TOOL_GRIP_UP), YAW_P4, YAW_P4, TOOL_GRIP, HOVER_L,
                        "second small tool", "P4")
    plan.append(Waypoint(1.5, PARK_L, OPEN, "aside", YAW_P4))
    return plan


def _right() -> list[Waypoint]:
    """Right arm: the face mill into P7 (the only pocket with both neighbours empty), then aside."""
    pick = (L.FM_START[0], L.FM_START[1], L.TOP + L.TRAY_TOP + FM_GRIP_UP)
    plan = _pick_place(pick, _place(L.FACE_POCKET, FM_GRIP_UP), YAW_P7, YAW_P7, FM_GRIP, HOVER_R, "face mill", "P7")
    if RIGHT_WAIT:
        plan.insert(1, Waypoint(RIGHT_WAIT, plan[0].pos, OPEN, "wait for the left arm to clear P3", YAW_P7))
    plan.append(Waypoint(1.5, PARK_R, OPEN, "aside", YAW_P7))
    return plan


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
