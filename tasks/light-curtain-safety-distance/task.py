"""On the station table: a bench-scale power press mock (a grey-green block) at the far end, 45-53 cm in front of the
arm bases, with a yellow die line taped on the floor along its front. In front of it, a 280 mm wide opening between
two dark grey side guards (10 mm thick, 100 mm tall) running from the press back to x 20 cm, floored with a light
grey plate. Outside the guards, on a blue mat each side at x 16 cm, y +-22 cm, stands a light-curtain post: a black
36 x 56 x 100 mm box (the emitter on the left, the receiver on the right), broad faces toward the press. Rule
(ISO 13855): stand one post against each guard, lens (broad) faces toward each other across the opening, with the
beam through their centres at least S = 2000 mm/s x 0.06 s + 8 x (20 - 14) mm = 168 mm from the die line's near
edge and at most 188 mm. A post closer than 168 mm scores 0.

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
from scoring import POSTS, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 2.5 s, not the default 1 s. The arm sags about 2.8 mm under the 0.15 kg post (J K^-1 J^T of its weight
# at the place pose; joints 80/80/80/10/10/10 N m/rad), so each post comes to rest on the floor before the slow
# descent ends, is seen at rest 0.5 s later, and success starts counting with the gripper still closed. Settling,
# letting go and lifting take 0.5 + 1.5 + 1.0 s, so with 2.5 s the trial ends with the fingers open and the hands
# going up. Offline (no sag) each post is let go 2 mm above the floor.
SUCCESS_HOLD_S = 2.5

INSTRUCTION = (
    "Stand the light curtain's two posts in the opening in front of the press, one against each side guard, with "
    "their lens faces (the broad faces) facing each other across the opening. The beam runs through the posts' "
    "centres: put it at the minimum safety distance from the yellow die line's near edge, S = 2000 mm/s x T + "
    "8 x (d - 14 mm), with stopping time T = 0.06 s and resolution d = 20 mm, or up to 20 mm farther."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


PRESS = _span_box("press", L.PRESS_X0, L.PRESS_X1, -L.PRESS_HALF_Y, L.PRESS_HALF_Y, 0.0, L.PRESS_H, L.PRESS_GREEN)
FLOOR = _span_box("opening_floor", L.OPEN_X0, L.PRESS_X0, -L.GUARD_INNER_Y, L.GUARD_INNER_Y, 0.0, L.FLOOR_T,
                  L.FLOOR_GREY)
DIE_LINE = _span_box("die_line", L.DIE_X, L.DIE_X + L.DIE_TAPE_W, -L.GUARD_INNER_Y, L.GUARD_INNER_Y, L.FLOOR_T,
                     L.FLOOR_T + L.DIE_TAPE_T, L.TAPE_YELLOW)
GUARD_LEFT = _span_box("guard_left", L.OPEN_X0, L.PRESS_X0, L.GUARD_INNER_Y, L.GUARD_OUTER_Y, 0.0, L.GUARD_H,
                       L.GUARD_GREY)
GUARD_RIGHT = _span_box("guard_right", L.OPEN_X0, L.PRESS_X0, -L.GUARD_OUTER_Y, -L.GUARD_INNER_Y, 0.0, L.GUARD_H,
                        L.GUARD_GREY)
MATS = [
    Box(f"mat_{s}", pos=(x, y, L.TOP + L.MAT_SIZE[2] / 2), size=L.MAT_SIZE, color=L.MAT_BLUE)
    for s, (x, y) in L.POST_START.items()
]
EMITTER, RECEIVER = (
    Box(name, pos=(x, y, L.TOP + L.MAT_SIZE[2] + L.POST_HALF[2]), size=L.POST_SIZE, mass=L.POST_MASS,
        color=L.POST_BLACK, yaw=L.POST_START_YAW)
    for name, (x, y) in zip(POSTS, (L.POST_START["left"], L.POST_START["right"]))
)
OBJECTS = [PRESS, FLOOR, DIE_LINE, GUARD_LEFT, GUARD_RIGHT, *MATS, EMITTER, RECEIVER]

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


def _posts(objects) -> dict:
    return {n: objects.get(n) for n in POSTS}


def success(objects) -> bool:
    """One post standing at rest against each guard (upright, on the floor, outer face within 10 mm of the guard),
    beam 168-188 mm from the die line, each lens axis within 5 degrees of the line between the posts, neither post
    closer than 168 mm; held 2.5 s. This runs first every control step, so it also feeds the tracker."""
    posts = _posts(objects)
    result = _TRACKER.update(posts)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    for n, s in posts.items():
        if s is not None:
            row[f"{n}_pos"] = s.pos
            row[f"{n}_quat"] = s.quat_xyzw
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if either post's centre is between the guards less than 168 mm from the die line. Else 0.25 for each guard
    with a post placed against it (upright, at rest on the floor, outer face within 10 mm of the guard, beam
    168-188 mm from the die line); with both placed, 0.25 more for each post whose lens axis is within 5 degrees of
    the line between the two posts. 0, 0.25, 0.5, 0.75 or 1.0."""
    return _TRACKER.result(_posts(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.POST_SIZE[1] - 0.008) / 2  # across the post's 56 mm width (its narrow side faces): 48 mm commanded
GRIP_DOWN = 0.020  # the TCP this far below the post's top: the pads grip its top 33 mm (tips 33 mm below its top)
POST_TOP0 = L.TOP + L.MAT_SIZE[2] + L.POST_SIZE[2]  # the post's top on its mat
PICK_Z = POST_TOP0 - GRIP_DOWN
HOVER_Z = L.TOP + 0.150  # the open pad tips 34 mm above the posts' tops
LIFT = 0.025  # carried with its bottom 25 mm above the mat (the floor plate is 1 mm lower)
CARRY_Z = PICK_Z + LIFT
HOLD_UP = L.POST_SIZE[2] - GRIP_DOWN  # 0.080: TCP height above the post's bottom
ENTRY_Z = L.FLOOR_TOP + 0.010 + HOLD_UP  # the last 8 mm are lowered slowly, from 10 mm above the floor
PLACE_Z = L.FLOOR_TOP + 0.002 + HOLD_UP  # let go with the bottom 2 mm above the floor (the arm sags about 2.8 mm)
# Each post is turned a quarter turn on the way, so its broad faces (the lens) face across the opening: the grip
# closes along y at the pick (yaw 0, across the 56 mm width) and along x at the place (yaw -pi/2 left, +pi/2 right,
# the signs that keep joint6 inside its limits), so neither finger goes between the post and its guard.
YAW_PLACE = {"left": -math.pi / 2, "right": math.pi / 2}
VIA_X = 0.150  # round the guards' front ends (x 0.200): the turned post spans x 0.122-0.178 here
LANE_Y = L.TARGET_Y - 0.005  # 0.112: forward along the guard with 10 mm between the post and it
# Payload sag at the place pose (static estimate as above): (-1.2, +-0.9, -2.8) mm, the sideways part toward each
# arm's own guard. It moves the beam 1.2 mm farther from the die line (window +-10 mm) and the gap to the guard from
# 5 to about 4 mm (limit 10 mm), so no trim is needed. On the GPU read d_mm and gap_mm in TASK_TRACE.
# The right arm starts 1.5 s later. As mirror images on one clock, both wrists (link4, turned a quarter turn) would pass
# the guards' front ends at x 0.15 together, 15 mm apart (capsule proxy); staggered, the arms stay 48 mm or more apart
# (closest with both hands over their places) and the grippers 0.23 m.
RIGHT_LAG = 1.5


def _plan(arm: str, lag: float = 0.0) -> list[Waypoint]:
    """One arm: pick its own post off its mat, lift it, turn it a quarter turn, carry it round the front end of its
    guard and forward along the guard, set it down slowly at the beam's place, let go, go up."""
    sgn = 1.0 if arm == "left" else -1.0
    x0, y0 = L.POST_START[arm]
    yaw = YAW_PLACE[arm]
    tx, ty, lane = L.TARGET_X, sgn * L.TARGET_Y, sgn * LANE_Y
    return [
        Waypoint(2.0 + lag, (x0, y0, HOVER_Z), OPEN, "over the post", 0.0),
        Waypoint(1.5, (x0, y0, PICK_Z), OPEN, "down round its top", 0.0),
        Waypoint(0.5, (x0, y0, PICK_Z), OPEN, "settle", 0.0),
        Waypoint(1.5, (x0, y0, PICK_Z), GRIP, "grip it across its width", 0.0),
        Waypoint(0.5, (x0, y0, PICK_Z), GRIP, "settle the grip", 0.0),
        Waypoint(1.0, (x0, y0, CARRY_Z), GRIP, "lift it off the mat", 0.0),
        Waypoint(1.5, (x0, y0, CARRY_Z), GRIP, "turn it a quarter turn: lens faces across", yaw),
        Waypoint(1.5, (VIA_X, lane, CARRY_Z), GRIP, "round the guard's front end", yaw),
        Waypoint(2.0, (tx, lane, CARRY_Z), GRIP, "forward into the opening along the guard", yaw),
        Waypoint(1.0, (tx, ty, ENTRY_Z), GRIP, "over its place, 10 mm up", yaw),
        Waypoint(0.5, (tx, ty, ENTRY_Z), GRIP, "settle", yaw),
        Waypoint(2.0, (tx, ty, PLACE_Z), GRIP, "lower it slowly onto the floor", yaw),
        Waypoint(0.5, (tx, ty, PLACE_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (tx, ty, PLACE_Z), OPEN, "let go", yaw),
        Waypoint(1.0, (tx, ty, HOVER_Z), OPEN, "straight up", yaw),
    ]


def demo():
    """Both arms, mirror images, the right one 1.5 s behind the left. DEMO_HOLD_ONLY=1: no plan
    (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the
    one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right", RIGHT_LAG)}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
