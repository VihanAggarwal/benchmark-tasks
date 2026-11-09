"""On the station table, at 1:8 scale: a green armchair (fixed) across the far side, facing the arms; Mr Okafor, a
rigid seated manikin (fixed), in it, his feet on the floor, his forearms on the chair's arms; and his walking frame
(a U-shaped aluminium frame, 70 mm across, 56 mm deep, 106 mm tall, 60 g, loose) standing side-on to his left
front (the robot's right), out of his reach. Rule (care aides setting up a sit-to-stand): the frame stands on its
four feet directly in front of him, square to his chair with its open side toward him, centred on him, its back
legs 10-50 mm (8-40 cm at scale) in front of his toes.

The walking frame is a mechkit compound (mechkit.py, next to this file: one loose rigid body of seventeen boxes); the
scorer reads its pose from it. It changes nothing else the embodiment does.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: the demo uses the right arm only; DEMO_IDLE_ARM=right leaves it at its ready pose (nothing moves).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the frame is built as one compound body)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Sphere  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, the frame still on its four feet) must hold 3 s, not the default 1 s. The frame's feet reach the
# floor at the end of the slow descent, while the gripper still holds it; settling and opening take 0.6 + 1.5 s, so
# the trial ends with the fingers open, going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Mr Okafor is about to stand up from his armchair and walk with his walking frame. Leave the frame ready for "
    "him: standing on its four feet directly in front of him and centred on him, square to his chair, with its open "
    "side toward him and its back legs 10 to 50 mm (8 to 40 cm at this 1:8 scale) in front of his toes."
)


def _span_box(name, xs, ys, zs, colour) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour])


ARMCHAIR = [_span_box(name, xs, ys, zs, "upholstery") for name, xs, ys, zs in L.CHAIR_BOXES]
RESIDENT = [_span_box(name, xs, ys, zs, colour) for name, xs, ys, zs, colour in L.RESIDENT_BOXES]
HEAD = Sphere("head", pos=(L.HEAD_CENTER[0], L.HEAD_CENTER[1], L.TOP + L.HEAD_CENTER[2]), radius=L.HEAD_RADIUS,
              color=L.COLOURS["skin"])
WALKING_FRAME = mk.assembly(
    "walking_frame",
    origin=(L.FRAME_START[0], L.FRAME_START[1], L.TOP),
    yaw=L.FRAME_START_HEADING,
    mass=L.FRAME_MASS,
    color=L.COLOURS["alu"],
    shapes=[mk.box(c, s, color=L.COLOURS[k]) for _, c, s, k in L.FRAME_SHAPES],
)
OBJECTS = [*ARMCHAIR, *RESIDENT, HEAD, WALKING_FRAME]

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
    """The frame still on its four feet in front of him: square within 15 deg, centred within 20 mm, its nearest
    legs 10-50 mm from his toes; held 3 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    frame = objects["walking_frame"]
    result = _TRACKER.update(frame)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(frame, "source", None), frame_origin=frame.origin, frame_quat=frame.quat_xyzw,
                   frame_low_z=frame.low[2], frame_vel=frame.lin_vel)
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the frame stands still on its four feet (each foot's bottom within 3 mm of the floor, still over
    0.5 s), turned no more than 30 deg from square (open side toward him), its centre within 50 mm of his centre
    line and its nearest leg 0-100 mm in front of his toes. Then 1.0 if all three are in their full bands (15 deg,
    20 mm, 10-50 mm); 0.5 if one is only in its loose band; 0.25 if two are; 0 if all three are."""
    return _TRACKER.result(objects["walking_frame"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# One arm: the frame starts on the right arm's side and its place is on the midline, both within the right arm's
# reach. The right arm grips it across its two handgrips (70 mm outside), at its centre of mass along its depth,
# lifts it 20 mm, carries it in front of him turning it square, and sets it down. The left arm holds its ready pose.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(2 * L.FOOT_V - 0.008) / 2  # 62 mm on the 70 mm handgrip span
GRIP_Z = L.TOP + 0.107  # the TCP 107 mm up: the pad tips 94 mm up, 4 mm below the handgrips (98-106 mm)
LIFT = 0.020
CARRY_Z = GRIP_Z + LIFT  # the frame's feet 20 mm up (carrying higher over the place costs joint-4 margin)
PLACE_Z = GRIP_Z + 0.002  # let go with its feet 2 mm above the floor
ABOVE_Z = L.TOP + 0.150  # pad tips 137 mm up, 31 mm over the frame's top (at the start only)
RISE_Z = L.TOP + 0.135  # after letting go: pad tips 122 mm up, 16 mm over its top (higher costs joint-4 margin)
PARK_R = (0.200, -0.300, L.TOP + 0.150)


def _grip_point(centre, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a frame whose footprint centre is at ``centre`` facing ``heading``."""
    return (centre[0] + L.GRIP_U * math.cos(heading), centre[1] + L.GRIP_U * math.sin(heading))


START_GRIP = _grip_point(L.FRAME_START, L.FRAME_START_HEADING)  # (0.300, -0.172)
YAW_START = math.pi / 2  # the fingers close along x, across the side-on frame (the right arm's yaw for the midline)
PLACE_GRIP = _grip_point(L.TARGET_CENTER, L.TARGET_HEADING)  # (0.290, 0.000)
YAW_PLACE = 0.0  # the fingers close along y: turning the grip -90 deg turns the frame from -90 to 180 deg


def _right() -> list[Waypoint]:
    """Right arm: pick the frame up across its handgrips, carry it in front of him turning it square, set it down
    with its back legs 30 mm from his toes, let go and park."""
    sx, sy = START_GRIP
    px, py = PLACE_GRIP
    return [
        Waypoint(2.5, (sx, sy, ABOVE_Z), OPEN, "over the frame's handgrips", YAW_START),
        Waypoint(2.0, (sx, sy, GRIP_Z + LIFT), OPEN, "down to just over the handgrips", YAW_START),
        Waypoint(1.5, (sx, sy, GRIP_Z), OPEN, "down the last 20 mm slowly", YAW_START),
        Waypoint(0.5, (sx, sy, GRIP_Z), OPEN, "settle", YAW_START),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip across the handgrips", YAW_START),
        Waypoint(1.5, (sx, sy, CARRY_Z), GRIP, "lift 20 mm", YAW_START),
        Waypoint(3.5, (px, py, CARRY_Z), GRIP, "carry it in front of him, turning it square", YAW_PLACE),
        Waypoint(0.5, (px, py, CARRY_Z), GRIP, "steady", YAW_PLACE),
        Waypoint(2.0, (px, py, PLACE_Z), GRIP, "lower it slowly", YAW_PLACE),
        Waypoint(0.6, (px, py, PLACE_Z), GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go", YAW_PLACE),
        Waypoint(1.5, (px, py, RISE_Z), OPEN, "straight up", YAW_PLACE),
        Waypoint(2.0, PARK_R, OPEN, "out of the way"),
    ]


def demo():
    """One arm: the left arm holds its ready pose. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a
    settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
