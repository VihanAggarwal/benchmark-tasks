"""On the station table, at 1:10 scale: a white bed (90 x 240 x 48 mm, fixed) across the far side, its head end to the
robot's left, with a pillow; Mrs Hale, a rigid seated manikin (fixed), on its near edge facing the arms, her left
(weak) forearm in a blue sling; and her wheelchair (95 x 63 mm footprint, 85 mm tall, 120 g, loose) parked past the
foot of the bed on her weak side, angled toward the arms. Rule (nursing assistants' stand-pivot transfer set-up):
the chair goes beside the bed next to her on her strong side (her right: the robot's left; its nearest corner at most
40 mm along the bed past her), within 20 mm of the bed's side (0.2 m at scale), its front toward the foot of the bed
and angled 30-50 degrees out from the bed's side.

The wheelchair is a mechkit compound (mechkit.py, next to this file: one loose rigid body of twelve boxes); the
scorer reads its pose from it. It changes nothing else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the wheelchair is built as one compound body)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Sphere  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, the chair still on the floor) must hold 3 s, not the default 1 s. The chair reaches the floor at the
# end of the slow descent, while the gripper still holds it (sooner on the GPU, where the arm sags a few mm under the
# chair); settling, opening and rising take 0.6 + 1.5 + 1.5 s, so the trial ends with the fingers open, going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Mrs Hale is sitting on the edge of her bed, about to move to her wheelchair; her left side is weak after a "
    "stroke. Set the wheelchair up for a stand-pivot transfer: beside the bed next to her on her strong side (its "
    "nearest corner at most 40 mm along the bed past her), within 20 mm of the bed's side (0.4 m and 0.2 m at "
    "this 1:10 scale), its front toward the foot of the bed, angled 30 to 50 degrees out from the bed's side, so "
    "she can stand, turn on her strong leg and sit down."
)


def _span_box(name, xs, ys, zs, colour) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour])


BED = _span_box("bed", (L.BED_X0, L.BED_X1), (L.BED_Y0, L.BED_Y1), (0.0, L.BED_H), "bed")
PILLOW = _span_box("pillow", L.PILLOW[0:2], L.PILLOW[2:4], L.PILLOW[4:6], "pillow")
RESIDENT = [_span_box(name, xs, ys, zs, colour) for name, xs, ys, zs, colour in L.RESIDENT_BOXES]
HEAD = Sphere("head", pos=(L.HEAD_CENTER[0], L.HEAD_CENTER[1], L.TOP + L.HEAD_CENTER[2]), radius=L.HEAD_RADIUS,
              color=L.COLOURS["skin"])
WHEELCHAIR = mk.assembly(
    "wheelchair",
    origin=(L.CHAIR_START[0], L.CHAIR_START[1], L.TOP),
    yaw=L.CHAIR_START_HEADING,
    mass=L.CHAIR_MASS,
    color=L.COLOURS["vinyl"],
    shapes=[mk.box(c, s, color=L.COLOURS[k]) for _, c, s, k in L.CHAIR_SHAPES],
)
OBJECTS = [BED, PILLOW, *RESIDENT, HEAD, WHEELCHAIR]

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
    """The chair set down (upright, on the floor, still) beside the bed next to her on her strong side, within 20 mm
    of the bed and 30-50 degrees out from it; held 3 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    chair = objects["wheelchair"]
    result = _TRACKER.update(chair)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(chair, "source", None), chair_origin=chair.origin, chair_quat=chair.quat_xyzw,
                   chair_low_z=chair.low[2], chair_vel=chair.lin_vel)
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the chair is set down (tilted 10 deg or less, lowest point within 3 mm of the floor, still over
    0.5 s) with every footprint corner at y >= 0.130 (her strong side) and the nearest at y <= 0.170 (beside her),
    no corner more than 3 mm past the bed's side line, its angle to the bed 0-90 deg and its gap to the bed 80 mm or
    less.
    Then 0.5 for a gap of 20 mm or less, plus 0.5 for an angle of 30-50 deg (0.25 for 15-30 or 50-65)."""
    return _TRACKER.result(objects["wheelchair"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# A relay, because of reach: the chair starts 0.5 m from the left arm's base line, and the place beside the bed is
# 0.5 m from the right arm's. The right arm carries the chair to the midline and sets it down facing the arms; the
# left arm picks it up there and sets it beside the bed. Both grips are across the two wheel plates (63 mm outside),
# at the middle of their length.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(2 * L.FOOT_V - 0.008) / 2  # 55 mm on the 63 mm wheel span
LET_GO = -0.0375  # 75 mm: 6 mm clear of each wheel plate, the far pad 17 mm short of the bed's side face
GRIP_Z = L.TOP + 0.048  # the TCP 48 mm up: the pad tips 35 mm up, on the wheel plates from 35 to 60 mm
CARRY_Z = L.TOP + 0.078  # the chair lifted 30 mm: its lowest point 30 mm up
PLACE_Z = L.TOP + 0.050  # let go with the chair's lowest point 2 mm above the floor
ABOVE_Z = L.TOP + 0.140  # pad tips 127 mm up, clear of the 85 mm backrest
PARK_L = (0.200, 0.300, L.TOP + 0.150)
PARK_R = (0.200, -0.300, L.TOP + 0.150)


def _grip_point(centre, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a chair whose footprint centre is at ``centre`` facing ``heading``."""
    return (centre[0] + L.GRIP_U * math.cos(heading), centre[1] + L.GRIP_U * math.sin(heading))


def _yaw(heading: float) -> float:
    """The grip's yaw for a chair facing ``heading``: the fingers close across the chair (along its left-right)."""
    return math.remainder(heading - math.pi, 2 * math.pi)


START_GRIP = _grip_point(L.CHAIR_START, L.CHAIR_START_HEADING)  # (0.3071, -0.2071)
YAW_START = _yaw(L.CHAIR_START_HEADING)  # -45 deg
HANDOVER_GRIP = L.HANDOVER_GRIP  # (0.2575, 0): the chair's centre at (0.2475, 0), facing the arms (heading 180 deg)
YAW_HANDOVER = 0.0
PLACE_HEADING = L.heading_of(L.TARGET_ALPHA_DEG)  # 230 deg: 40 deg out from the bed's side
PLACE_GRIP = _grip_point(L.TARGET_CENTER, PLACE_HEADING)  # (0.3418, 0.2127)
YAW_PLACE = _yaw(PLACE_HEADING)  # +50 deg
# Before the last straight move in: the chair already turned to 40 deg, 32 mm short of the place in x, its footprint
# 73 mm from her slippers and 42 mm from the bed's side (tests/test_scoring.py checks the whole carry path).
TURN_GRIP = (0.310, 0.215)
ALONG_GRIP = (HANDOVER_GRIP[0], 0.170)  # carried along y first, facing the arms: 68 mm in front of her slippers


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _right() -> list[Waypoint]:
    """Right arm: pick the chair up across its wheels, carry it to the midline turning it to face the arms, set it
    down there, let go and park."""
    sx, sy = START_GRIP
    hx, hy = HANDOVER_GRIP
    return [
        Waypoint(3.0, (sx, sy, ABOVE_Z), OPEN, "over the chair's wheels", YAW_START),
        Waypoint(1.5, (sx, sy, CARRY_Z), OPEN, "down beside the wheels", YAW_START),
        Waypoint(1.5, (sx, sy, GRIP_Z), OPEN, "down the last 30 mm slowly", YAW_START),
        Waypoint(0.5, (sx, sy, GRIP_Z), OPEN, "settle", YAW_START),
        Waypoint(1.5, (sx, sy, GRIP_Z), GRIP, "grip across the wheels", YAW_START),
        Waypoint(1.5, (sx, sy, CARRY_Z), GRIP, "lift 30 mm", YAW_START),
        Waypoint(3.0, (hx, hy, CARRY_Z), GRIP, "carry it to the midline, turning it to face the arms", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, PLACE_Z), GRIP, "lower it slowly", YAW_HANDOVER),
        Waypoint(0.5, (hx, hy, PLACE_Z), GRIP, "settle", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, PLACE_Z), OPEN, "let go", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, ABOVE_Z), OPEN, "straight up", YAW_HANDOVER),
        Waypoint(2.5, PARK_R, OPEN, "out of the way"),
    ]


def _left() -> list[Waypoint]:
    """Left arm: wait parked until the right arm has left the midline, pick the chair up there, carry it along y in
    front of her, turn it out 40 degrees, bring it in to 10 mm from the bed, set it down, let go and go up."""
    hx, hy = HANDOVER_GRIP
    ax, ay = ALONG_GRIP
    tx, ty = TURN_GRIP
    px, py = PLACE_GRIP
    park = [Waypoint(2.0, PARK_L, OPEN, "park")]
    wait = _span(_right()) - _span(park)  # the right arm is parked at 20.0 s
    return park + [
        Waypoint(wait, PARK_L, OPEN, "wait for the right arm to leave the midline"),
        Waypoint(2.5, (hx, hy, ABOVE_Z), OPEN, "over the chair on the midline", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, CARRY_Z), OPEN, "down beside the wheels", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, GRIP_Z), OPEN, "down the last 30 mm slowly", YAW_HANDOVER),
        Waypoint(0.5, (hx, hy, GRIP_Z), OPEN, "settle", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, GRIP_Z), GRIP, "grip across the wheels", YAW_HANDOVER),
        Waypoint(1.5, (hx, hy, CARRY_Z), GRIP, "lift 30 mm", YAW_HANDOVER),
        Waypoint(2.5, (ax, ay, CARRY_Z), GRIP, "carry it along, in front of her feet", YAW_HANDOVER),
        Waypoint(2.5, (tx, ty, CARRY_Z), GRIP, "turn it out 40 degrees from the bed's side", YAW_PLACE),
        Waypoint(1.5, (px, py, CARRY_Z), GRIP, "in toward the bed", YAW_PLACE),
        Waypoint(2.0, (px, py, PLACE_Z), GRIP, "lower it slowly", YAW_PLACE),
        Waypoint(0.6, (px, py, PLACE_Z), GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (px, py, PLACE_Z), LET_GO, "let go", YAW_PLACE),
        Waypoint(1.5, (px, py, ABOVE_Z), LET_GO, "straight up", YAW_PLACE),
        Waypoint(2.0, PARK_L, LET_GO, "out of the way"),
    ]


def demo():
    """Both arms on one clock: a relay. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
