"""A 1:10 stage on the station table, set for a seated acoustic duo: a dark grey stage deck (330 x 680 x 3 mm, fixed)
in front of the arm bases, the audience on the arms' side. On it stand two singers' wooden stools and, 40 mm in front
of each (toward the audience), her mic stand (all fixed), 30 cm apart: the left singer's with a black cardioid mic, the
right singer's with a silver supercardioid, each mic's ball head straight above its stand's pole at seated mouth
height and pointing upstage (+x) at her stool. Two floor wedges (stage monitors, 45 x 60 x 30 mm, 80 g, a black
stepped cabinet with a grey grille on its front) wait in the wings on the table, one each side, each grille facing
its own wing. Rule: one wedge in each mic's null (the cardioid's directly behind it, 180 degrees off its front; the
supercardioid's about 126 degrees off its front, either side), 80-150 mm from the stand, its grille aimed at that
singer's stool.

Each wedge is a loose mechkit compound (mechkit.py, next to this file): one rigid body of four boxes, so its grille
is visible and the scorer can read which way it faces (base frame and orientation).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the wedges are built as compound bodies)
from scoring import WEDGES, Tracker, wedge_ends  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder, Sphere  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 3 s, not the default 1 s. The arm sags a little under the 80 g wedge, so each wedge comes to rest on the
# deck while the slow descent is still finishing and success starts counting (after the 0.5 s needed to see it at
# rest) with the grippers still closed. Finishing the descent, settling and letting go take about 0.5 + 0.5 + 1.5 s,
# so with 3 s the trial ends with both grippers open and the hands going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Sound check on a 1:10 stage: set one floor wedge for each seated singer in the null of her mic, so it will not "
    "feed back. The left singer's black cardioid mic: directly behind it, 180 degrees off the mic's front. The "
    "right singer's silver supercardioid: about 126 degrees off its front, either side. Stand each wedge base down "
    "80 to 150 mm from its mic stand, its grey grille aimed at that singer's stool."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


DECK = _span_box("stage_deck", L.DECK_X0, L.DECK_X1, L.DECK_Y0, L.DECK_Y1, 0.0, L.DECK_T, L.DECK_GREY)
STANDS = []
for _s, (_x, _y, _pattern, _null) in L.SINGERS.items():
    _handle_col, _head_col = L.MIC_COLOURS[_s]
    _z0 = L.DECK_T
    STANDS += [
        Cylinder(f"stand_base_{_s}", pos=(_x, _y, L.TOP + _z0 + L.STAND_BASE_H / 2), radius=L.STAND_BASE_R,
                 height=L.STAND_BASE_H, color=L.STAND_BLACK),
        Cylinder(f"stand_pole_{_s}", pos=(_x, _y, L.TOP + _z0 + L.STAND_BASE_H + L.POLE_H / 2), radius=L.POLE_R,
                 height=L.POLE_H, color=L.STAND_BLACK),
        Box(f"mic_clip_{_s}", pos=(_x, _y, L.TOP + L.POLE_TOP + L.CLIP_SIZE[2] / 2), size=L.CLIP_SIZE,
            color=L.STAND_BLACK),
        Sphere(f"mic_head_{_s}", pos=(_x, _y, L.TOP + L.HEAD_Z), radius=L.HEAD_R, color=_head_col),
        _span_box(f"mic_handle_{_s}", _x - L.HEAD_R - L.HANDLE_LEN, _x - L.HEAD_R, _y - L.HANDLE_W / 2,
                  _y + L.HANDLE_W / 2, L.HEAD_Z - L.HANDLE_W / 2, L.HEAD_Z + L.HANDLE_W / 2, _handle_col),
    ]
STOOLS = [
    Cylinder(f"stool_{_s}", pos=(_px, _py, L.TOP + L.DECK_T + L.STOOL_H / 2), radius=L.STOOL_R, height=L.STOOL_H,
             color=L.STOOL_WOOD)
    for _s, (_px, _py) in L.STOOLS.items()
]

_SHAPE_COLOURS = {"cabinet": L.CABINET_BLACK, "grille": L.GRILLE_GREY}


def _wedge(name: str):
    x, y, heading = L.WEDGES[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP),
        yaw=heading,
        mass=L.WEDGE_MASS,
        color=L.CABINET_BLACK,
        shapes=[mk.box(center, size, color=_SHAPE_COLOURS[key]) for _, center, size, key in L.WEDGE_SHAPES],
    )


WEDGE_LEFT = _wedge("wedge_left")
WEDGE_RIGHT = _wedge("wedge_right")
OBJECTS = [DECK, *STANDS, *STOOLS, WEDGE_LEFT, WEDGE_RIGHT]

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


def _states(objects) -> dict:
    return {n: objects[n] for n in WEDGES}


def success(objects) -> bool:
    """A wedge in each mic's null, 80-150 mm from the stand, within 15 degrees of the null, its grille aimed within
    20 degrees of the singer's stool, standing on the deck at rest; held 3 s. This runs first every control step,
    so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    states = _states(objects)
    result = _TRACKER.update(states)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        for n in WEDGES:
            front, back = wedge_ends(states[n])
            row[f"{n}_origin"] = states[n].origin
            row[f"{n}_quat"] = states[n].quat_xyzw
            row[f"{n}_source"] = getattr(states[n], "source", None)
            row[f"{n}_front"] = front
            row[f"{n}_back"] = back
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """Left singer + right singer. A wedge is placed when it stands base down (within 10 degrees) on the deck, its
    lowest point within 3 mm of the deck's top, wholly over the deck, at rest; it serves the singer whose stand is
    nearer. Each singer scores from her best placed wedge: 0.5 at 80-150 mm
    from her stand, within 15 degrees of her mic's null and aimed within 20 degrees of her stool; 0.25 at 60-200
    mm, within 30 and 45 degrees; else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result(_states(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN_F = -0.04695  # fully open, 94 mm: 17 mm either side of the 60 mm wide wedge
GRIP_F = -(L.WIDTH - 0.008) / 2  # 52 mm commanded across its 60 mm width
# Each wedge is gripped across its width (the fingers close along its local y, so the gripper's yaw equals the
# grille's heading), 4 mm behind the middle of its footprint, over its centre of mass: the pads then hold the tall
# rear block's sides and the slab's. Turning the gripper turns the wedge by the same angle.
GRIP_X = -0.004
GRIP_UP = 0.020  # the TCP 20 mm above a wedge's base: pad tips 7 mm up
DROP = 0.001  # wedges are let go with their base 1 mm above the deck (the arm sags a little under them)
APPROACH = 0.030  # each final descent starts this far above its end
CARRY = L.TOP + 0.10  # carried with the base 80 mm up
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.DECK_TOP_W + DROP + GRIP_UP


DEMO_R = 0.110  # the demo sets each wedge 110 mm from its stand: mid-band (80-150), and 10 mm more room for the
#                wrist under the left mic's handle than at 100 mm (offline capsule proxy: 29 mm left, 37 mm right)


def target(singer: str) -> tuple[float, float, float]:
    """Where the demo puts that singer's wedge: (footprint middle x, y, grille heading). DEMO_R from her stand, right
    in the null (the cardioid's straight behind; the supercardioid's on the outer side, 126 degrees round toward
    -y), aimed straight at her stool."""
    sx, sy, _, null = L.SINGERS[singer]
    side = -1.0 if sy < 0 else 1.0  # the outer side: away from the midline
    a = math.radians(null) * side
    x, y = sx + DEMO_R * math.cos(a), sy + DEMO_R * math.sin(a)
    px, py = L.STOOLS[singer]
    return x, y, math.atan2(py - y, px - x)


# Trims of the place points (x, y), for the GPU run: if a wedge lands off its target, read its origin in TASK_TRACE
# (wedge_left_origin, wedge_right_origin) against target() and move the place point by minus the offset. The full
# band is -30/+40 mm in distance and +-15 degrees round the stand (about +-29 mm at 110 mm).
TRIM_LEFT = (0.0, 0.0)
TRIM_RIGHT = (0.0, 0.0)
YAW_LEFT_PICK = L.WEDGES["wedge_left"][2]  # +90 degrees: the left wedge's grille faces +y at the start
YAW_RIGHT_PICK = L.WEDGES["wedge_right"][2]  # -90 degrees
YAW_LEFT_PLACE = target("left")[2]  # 0: grille upstage, at the left stool
YAW_RIGHT_PLACE = target("right")[2]  # about +40.4 degrees
ASIDE_LEFT = (0.20, 0.30, L.TOP + 0.15)
ASIDE_RIGHT = (0.20, -0.30, L.TOP + 0.15)


def grip_point(x: float, y: float, heading: float) -> tuple[float, float]:
    """The TCP's x, y for a wedge whose footprint middle is at (x, y) with its grille heading ``heading``."""
    return x + GRIP_X * math.cos(heading), y + GRIP_X * math.sin(heading)


def _pick(xy, yaw: float) -> list[Waypoint]:
    """From anywhere: over the wedge's grip point, down, slowly down beside it, settle, grip, lift (7.2 s)."""
    x, y = xy
    return [
        Waypoint(1.5, (x, y, CARRY), OPEN_F, "over the wedge", yaw),
        Waypoint(1.0, (x, y, PICK_Z + APPROACH), OPEN_F, "down to just above the wedge", yaw),
        Waypoint(1.5, (x, y, PICK_Z), OPEN_F, "slowly down beside the wedge", yaw),
        Waypoint(0.5, (x, y, PICK_Z), OPEN_F, "settle", yaw),
        Waypoint(1.5, (x, y, PICK_Z), GRIP_F, "grip the wedge", yaw),
        Waypoint(1.2, (x, y, CARRY), GRIP_F, "lift the wedge", yaw),
    ]


def _place(start_xy, xy, yaw: float, start_yaw: float) -> list[Waypoint]:
    """From over its pick at ``start_xy``: carry over the place point, turning from ``start_yaw`` to ``yaw`` (in two
    halves, so no waypoint turns the wrist more than a quarter turn), settle, down, slowly down, settle, let go,
    straight up (9.3 s)."""
    x, y = xy
    mid = ((start_xy[0] + x) / 2, (start_xy[1] + y) / 2, CARRY)
    return [
        Waypoint(1.25, mid, GRIP_F, "carry the wedge toward the stage, turning it", (start_yaw + yaw) / 2),
        Waypoint(1.25, (x, y, CARRY), GRIP_F, "carry the wedge over its place, turning it", yaw),
        Waypoint(0.8, (x, y, CARRY), GRIP_F, "settle over it", yaw),
        Waypoint(1.0, (x, y, PLACE_Z + APPROACH), GRIP_F, "down to 3 cm above the deck", yaw),
        Waypoint(2.0, (x, y, PLACE_Z), GRIP_F, "slowly set the wedge on the deck", yaw),
        Waypoint(0.5, (x, y, PLACE_Z), GRIP_F, "settle", yaw),
        Waypoint(1.5, (x, y, PLACE_Z), OPEN_F, "let go of the wedge", yaw),
        Waypoint(1.0, (x, y, CARRY), OPEN_F, "straight up", yaw),
    ]


def _arm(name: str, singer: str, trim, aside) -> list[Waypoint]:
    sx, sy, sh = L.WEDGES[name]
    tx, ty, th = target(singer)
    pick = grip_point(sx, sy, sh)
    place = grip_point(tx + trim[0], ty + trim[1], th)
    plan = _pick(pick, sh)
    plan += _place(pick, place, th, sh)
    plan += [Waypoint(1.0, aside, OPEN_F, "aside", th)]
    return plan


def _left() -> list[Waypoint]:
    """Left arm: the left wedge into the cardioid's null, straight behind the left mic, turned from facing the wing
    (+y) to facing upstage at the left stool (yaw +90 to 0 degrees)."""
    return _arm("wedge_left", "left", TRIM_LEFT, ASIDE_LEFT)


def _right() -> list[Waypoint]:
    """Right arm: the right wedge into the supercardioid's outer null, 126 degrees off the right mic's front, turned
    from facing the wing (-y) to facing the right stool (yaw -90 to about +40.4 degrees)."""
    return _arm("wedge_right", "right", TRIM_RIGHT, ASIDE_RIGHT)


def demo():
    """Both arms at once. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
