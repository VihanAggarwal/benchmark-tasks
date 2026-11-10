"""On the station table (a self-service salad bar's counter, 1:4): five fixed 1/3-size food pans in a row along y
(lettuce, tomato, cucumber, sweetcorn, croutons; 81 mm deep, 44 mm wide, food 25 mm above the counter), their
customer-side edge 34 cm in front of the arm bases, and a fixed stainless tray slide on the customers' side, 94 mm
past the pans. A loose portable food shield (one rigid body: a 320 x 80 mm acrylic top glass 90 mm up on two end
panels, a front lip under the glass's customer-side edge, a stainless end cap on the glass over each panel; 0.25 kg)
stands on the staff side, pulled back for refilling and left 30 mm toward the left end of the bar. Rule: put it back
over the bar, standing on both end panels, square, spanning the row, with its lip 45 to 75 mm past the pans'
customer-side edge at both ends of the row (0.75; 25 to 90 mm: 0.25), plus 0.25 if centred on the row within 15 mm.
Resting on a pan, the food or the tray slide scores 0.

The shield is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of six boxes, so it
stands on its two end panels astride the pans and the scorer can read its pose (base frame and orientation).

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the shield is built as one compound body)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 3 s, not the default 1 s. The arms sag a few mm under the shield, so its panels reach the counter before
# the slow descent ends and success starts counting 0.5 s later (the time needed to see it at rest) with both
# grippers still closed. Settling and opening take 0.5 + 1.5 s, so with 3 s the trial ends with the fingers open.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Put the portable food shield back in service over the salad bar: standing on its two end panels, centred "
    "along the row of pans, with its front lip (the customers' side, by the tray slide) 18 to 30 cm (45 to 75 mm "
    "at this 1:4 scale) beyond the customers' edge of the food pans. Do not rest it on the pans."
)


def _box(name: str, x: float, y: float, z0: float, size, color) -> Box:
    """A fixed box by its centre (x, y), its bottom height above the counter and its size."""
    return Box(name, pos=(x, y, L.TOP + z0 + size[2] / 2), size=size, color=color)


PANS = []
for _food, _y, _key in L.PANS:
    PANS.append(_box("pan_" + _food, L.PAN_X, _y, 0.0, (L.PAN_DEPTH, L.PAN_W, L.PAN_BODY_H), L.PAN_STEEL))
    PANS.append(_box("food_" + _food, L.PAN_X, _y, L.PAN_BODY_H,
                     (L.PAN_DEPTH - 2 * L.FOOD_INSET, L.PAN_W - 2 * L.FOOD_INSET, L.FOOD_H), L.colour(_key)))
TRAY_SLIDE = _box("tray_slide", L.SLIDE_X, 0.0, 0.0, L.SLIDE_SIZE, L.STEEL)

SHIELD = mk.assembly(
    "food_shield",
    origin=(L.START[0], L.START[1], L.TOP),
    yaw=L.START_YAW,
    mass=L.SHIELD_MASS,
    color=L.ACRYLIC,
    shapes=[mk.box(c, s, color=L.colour(k)) for c, s, k in L.SHIELD_BOXES],
)
OBJECTS = [*PANS, TRAY_SLIDE, SHIELD]

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
    """The shield standing on both end panels, square, spanning the row, at rest, its lip 45-75 mm past the pans'
    customer-side edge at both ends of the row and its centre within 15 mm of the row's; held 3 s. This runs first
    every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    s = objects["food_shield"]
    result = _TRACKER.update(s)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["shield_state"] = {"source": getattr(s, "source", None), "origin": getattr(s, "origin", None),
                               "quat": s.quat_xyzw, "vel": s.lin_vel}
    except AttributeError:
        row["shield_state"] = "unreadable"
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless the shield stands on the counter on both end panels (tilt <= 5 deg, every panel corner within 3 mm of
    the counter), square (lip within 10 deg of toward the customers), spanning the row and at rest. Then 0.75 if the
    lip overhangs the pans' customer-side edge by 45-75 mm at both ends of the row, 0.25 if by 25-90 mm; plus 0.25
    when that earned something and the centre is within 15 mm of the row's. Read from the state at hand."""
    return _TRACKER.result(objects["food_shield"]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Both arms grip the shield at once, each by the end cap on its own side, with the fingers closing along x across
# the cap's 30 mm depth (the left arm at yaw -pi/2, the right at +pi/2), lift it 30 mm, carry it forward together
# (and 30 mm to the right, centring it), lower it slowly onto the counter and let go together. The two grips are
# 296 mm apart along the bar.
OPEN = -0.04695  # fully open (94 mm)
CAP_GRIP = -(L.CAP_SIZE[0] - 0.008) / 2  # 22 mm commanded across the cap's 30 mm depth
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
SIDE = {"left": 1.0, "right": -1.0}
GRIP_UP = L.CAP_Z0 + 0.005 + 0.013  # TCP 112 mm above the panels' feet: pad tips 5 mm above the glass
LIFT = 0.030  # the panels' feet 30 mm up while carried: over the pans' 25 mm (they pass beside the row anyway)
DROP = 0.002  # the shield is let go with its panels' feet 2 mm above the counter
APPROACH = 0.020  # the descent onto the caps starts this far above the grip height
SET_APPROACH = 0.015  # the last 15 mm of the set-down are lowered slowly
HOVER_Z = L.TOP + 0.160
CARRY_Z = L.TOP + GRIP_UP + LIFT
# Where the shield is set: its lip 60 mm past the pans' edge (15 mm inside either end of the band), centred on the
# row. Trim (x, y), for the GPU run: if the shield lands off its aim (overhang_left_mm / overhang_right_mm in
# TASK_TRACE away from 60, or centre_mm[1] away from 0), move the place point by minus the offset.
PLACE = (L.PAN_X1 + L.OVERHANG_AIM - L.LIP_FACE_X, 0.0)  # (0.360, 0.0): the shield's centre
PLACE_TRIM = (0.0, 0.0)
PARK = (0.200, 0.250)  # each arm goes back here (mirrored for the right), clear of the shield


def _arm(side: str) -> list[Waypoint]:
    """One arm: over its end cap, grip it, lift (with the other arm), carry forward, lower, let go, go up and park."""
    k, yaw = SIDE[side], YAW[side]
    sx, sy = L.START[0], L.START[1] + k * L.CAP_Y
    px, py = PLACE[0] + PLACE_TRIM[0], PLACE[1] + PLACE_TRIM[1] + k * L.CAP_Y
    gz = L.TOP + GRIP_UP
    pz = L.TOP + DROP + GRIP_UP
    return [
        Waypoint(2.0, (sx, sy, HOVER_Z), OPEN, "over the end cap", yaw),
        Waypoint(1.0, (sx, sy, gz + APPROACH), OPEN, "down to just above the cap", yaw),
        Waypoint(1.5, (sx, sy, gz), OPEN, "slowly down round the cap", yaw),
        Waypoint(0.5, (sx, sy, gz), OPEN, "settle", yaw),
        Waypoint(1.5, (sx, sy, gz), CAP_GRIP, "grip the end cap", yaw),
        Waypoint(0.3, (sx, sy, gz), CAP_GRIP, "settle the grip", yaw),
        Waypoint(1.5, (sx, sy, CARRY_Z), CAP_GRIP, "lift the shield 30 mm, both arms together", yaw),
        Waypoint(5.0, (px, py, CARRY_Z), CAP_GRIP, "carry it forward over the pans, centring it", yaw),
        Waypoint(0.8, (px, py, CARRY_Z), CAP_GRIP, "settle over the place", yaw),
        Waypoint(1.0, (px, py, pz + SET_APPROACH), CAP_GRIP, "down until the feet are 17 mm up", yaw),
        Waypoint(2.0, (px, py, pz), CAP_GRIP, "slowly set it on the counter", yaw),
        Waypoint(0.5, (px, py, pz), CAP_GRIP, "settle", yaw),
        Waypoint(1.5, (px, py, pz), OPEN, "let go of the cap", yaw),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", yaw),
        Waypoint(1.0, (px, py, CARRY_Z), OPEN, "straight up", yaw),
        Waypoint(1.5, (PARK[0], k * PARK[1], HOVER_Z), OPEN, "back to the side", yaw),
    ]


def demo():
    """Both arms on one clock, mirrored: each grips the end cap on its own side and they carry the shield together.
    DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's
    plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
