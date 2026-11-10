"""On the station table, at 1:10 scale: the front of a classroom. At the far end stands the front wall (pale green,
190 mm tall) with two matt white projection screens side by side on it, each 160 mm wide (1.6 m) and 100 mm tall,
their centre lines at y +120 and -120 mm, with a black centre mark on the wall below each. In front of the wall stands
the projector table (classroom tables pushed together: beech laminate, 200 x 400 mm, 75 mm tall) and on it two
portable projectors (34 mm across, 26 mm deep, 14 mm tall with a lens 5 mm proud of the front; 50 g each, loose): a
black one with a 1.5:1 throw ratio on the left and a white one with a 1.0:1 throw ratio on the right, turned sideways
with their lenses toward the middle. Rule (AV set-up of a fixed-lens projector): throw distance (lens to screen) =
throw ratio x picture width, so to fill a 160 mm screen the black projector's lens stands 240 mm from it and the white
one's 160 mm; each projector square to the wall and its lens on its screen's centre line.

Each projector is a mechkit compound (mechkit.py, next to this file: one loose rigid body of two boxes, body and lens),
so its lens can be seen and the scorer can read which way it points. The scorer also reads the arms (TCP, finger-tip
point, opening) through rigstate.py, which wraps the embodiment's step and reset from this file's import, so that a
projector counts only once both grippers have let go of it. Neither changes anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the projectors are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, both projectors set up, still and let go) must hold 3 s, not the default 1 s. A projector counts only
# once both grippers have let go of it (at least 90 % open, or their fingers more than 10 mm away), which in the demo
# happens near the end of 'let go'; the 3 s hold then takes the trial past 'straight up', so it ends with the fingers
# open and clear of the projectors.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Set up the two projectors for the two screens on the front wall, one projector per screen. Each screen is "
    "160 mm wide (1.6 m at this 1:10 scale). The black projector has a 1.5:1 throw ratio and the white one 1.0:1 "
    "(throw distance, from the front of the lens to the screen = throw ratio x picture width). Stand each projector "
    "upright on the table facing its screen, square to the wall, its lens on the screen's centre line, at the "
    "distance where its picture just fills the screen."
)


def _span_box(name, xs, ys, zs, colour) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour])


FRONT_WALL = _span_box("front_wall", L.WALL_X, L.WALL_Y, (0.0, L.WALL_H), "wall")
SCREEN_BOXES = [
    _span_box(s, L.SCREEN_X, (L.SCREEN_Y[s] - L.SCREEN_W / 2, L.SCREEN_Y[s] + L.SCREEN_W / 2), L.SCREEN_Z, "screen")
    for s in L.SCREENS
]
MARKS = [
    _span_box(f"centre_mark_{s.split('_')[1]}", L.SCREEN_X, (L.SCREEN_Y[s] - L.MARK_W / 2, L.SCREEN_Y[s] + L.MARK_W / 2),
              L.MARK_Z, "mark")
    for s in L.SCREENS
]
PROJECTOR_TABLE = _span_box("projector_table", L.TABLE_X, L.TABLE_Y, (0.0, L.TABLE_H), "table")
ROOM = [FRONT_WALL, *SCREEN_BOXES, *MARKS, PROJECTOR_TABLE]


def _projector(name: str):
    x, y, heading = L.START[name]
    colours = {"body": L.BODY_COLOUR[name], "lens": L.LENS_COLOUR[name]}
    return mk.assembly(
        name,
        origin=(x, y, L.TABLE_TOP_W),
        yaw=heading,
        mass=L.PROJ_MASS,
        color=L.COLOURS[colours["body"]],
        shapes=[mk.box(c, s, color=L.COLOURS[colours[k]]) for _, c, s, k in L.SHAPES],
    )


PROJECTOR_BLACK = _projector("projector_black")
PROJECTOR_WHITE = _projector("projector_white")
OBJECTS = [*ROOM, PROJECTOR_BLACK, PROJECTOR_WHITE]

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
    """Both screens filled (score 1.0): a projector standing still on the table for each screen, let go by both
    grippers, square to the wall within 5 degrees, its lens within 6 mm of the screen's centre line and its picture
    within 5 % of 160 mm wide; held 3 s. This runs first every control step, so it also feeds the tracker (with the
    arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    black, white = objects.get("projector_black"), objects.get("projector_white")
    arms = rigstate.arms()
    result = _TRACKER.update(black, white, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                       right_tips=arms["right"].tips)
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    try:
        row.update(source=getattr(black, "source", None),
                   origins={"projector_black": black.origin, "projector_white": white.origin},
                   quats={"projector_black": black.quat_xyzw, "projector_white": white.quat_xyzw},
                   low_z={"projector_black": black.low[2], "projector_white": white.low[2]},
                   vels={"projector_black": black.lin_vel, "projector_white": white.lin_vel})
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """Per screen, the best projector set up for it: 0.5 if it stands still on the table (tilted 5 degrees or less,
    lowest point within 3 mm of the table top) let go by both grippers, its lens toward the wall square within 5
    degrees, the lens's centre within 6 mm of the screen's centre line and its picture (throw distance / its throw
    ratio) within 5 % of 160 mm; 0.25 if it stands still on the table, let go, with its lens within 10 degrees of
    square and 12 mm of the centre line and its picture within 15 %; else 0. The sum of the two screens. Read from the
    state at hand (the last step)."""
    return _TRACKER.result(objects.get("projector_black"), objects.get("projector_white"), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Both arms at once, each with its own projector: the left arm sets the black projector (1.5:1) 240 mm from the left
# screen, the right arm the white one (1.0:1) 160 mm from the right screen. Each grip is across the projector's 34 mm
# body at its middle (the frame origin, 0.8 mm behind the centre of mass), and the quarter turn is made with the wrist
# while the projector is carried: its heading changes with the grip's yaw. Yaw 0 closes the fingers along world y, so
# a projector whose lens points at the wall (+x) is held at yaw 0 and one turned sideways (lens toward -y or +y) at
# yaw -90 (left arm) or +90 (right arm) degrees. Turning the wrist the other way, to -180 / +180 degrees, would point
# the lens backwards, at the arms (and run joint 6 past its limit with this grip).
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm: 30 mm clear of the body on each side
GRIP = -(L.BODY_W - 0.008) / 2  # 26 mm commanded on the 34 mm body
GRIP_U = 0.0  # the grip's centre, forward of the projector's origin (m, projector frame)
# The TCP 17 mm above the table: the pad tips 4 mm above it, the pads on the body from 4 mm up to its top (14 mm).
GRIP_Z = L.TABLE_TOP_W + 0.017
CARRY_Z = GRIP_Z + 0.020  # the projector lifted 20 mm
PLACE_Z = GRIP_Z + 0.002  # let go with the projector's base 2 mm above the table
APPROACH_Z = GRIP_Z + 0.030  # the last 30 mm down to the grip are slow
ABOVE_Z = L.TOP + 0.150  # pad tips 137 mm up: 48 mm over the projectors' tops
PARK = {"left": (0.200, 0.300, L.TOP + 0.150), "right": (0.200, -0.300, L.TOP + 0.150)}
SIDE = {"left": "projector_black", "right": "projector_white"}


def grip_point(origin_xy, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a projector whose origin is at ``origin_xy`` facing ``heading``."""
    return (origin_xy[0] + GRIP_U * math.cos(heading), origin_xy[1] + GRIP_U * math.sin(heading))


def grip_yaw(heading: float) -> float:
    """The grip's yaw for a projector facing ``heading``: the fingers close across its body (along its left-right)."""
    return math.remainder(heading, 2 * math.pi)


def _arm(arm: str) -> list[Waypoint]:
    """One arm: over its projector, down slowly, grip across the body, lift 20 mm, carry it to its place turning it a
    quarter turn so the lens faces the wall, lower it slowly, let go, straight up, park."""
    name = SIDE[arm]
    x0, y0, h0 = L.START[name]
    x1, y1, h1 = L.TARGET[name]
    sx, sy = grip_point((x0, y0), h0)
    px, py = grip_point((x1, y1), h1)
    ys, yp = grip_yaw(h0), grip_yaw(h1)
    return [
        Waypoint(3.0, (sx, sy, ABOVE_Z), OPEN, f"over {name}", ys),
        Waypoint(1.5, (sx, sy, APPROACH_Z), OPEN, "down beside its body", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), OPEN, "down the last 30 mm slowly", ys),
        Waypoint(0.5, (sx, sy, GRIP_Z), OPEN, "settle", ys),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip across the body", ys),
        Waypoint(0.5, (sx, sy, GRIP_Z), GRIP, "settle the grip", ys),
        Waypoint(1.5, (sx, sy, CARRY_Z), GRIP, "lift 20 mm", ys),
        Waypoint(3.5, (px, py, CARRY_Z), GRIP, "carry it to its throw distance, turning the lens to the wall", yp),
        Waypoint(0.5, (px, py, CARRY_Z), GRIP, "settle over the place", yp),
        Waypoint(2.0, (px, py, PLACE_Z), GRIP, "lower it slowly", yp),
        Waypoint(0.6, (px, py, PLACE_Z), GRIP, "settle", yp),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go", yp),
        Waypoint(1.5, (px, py, ABOVE_Z), OPEN, "straight up", yp),
        Waypoint(2.0, PARK[arm], OPEN, "out of the way"),
    ]


def demo():
    """Both arms on one clock, each setting up its own projector. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
