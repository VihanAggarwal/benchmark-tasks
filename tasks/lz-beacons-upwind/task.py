"""On the station table, at 1:110: a green grass field (335 x 360 mm, a fixed 1 mm mat) in front of both arms, a
dark grey road along its left edge and a light grey parking strip along its right edge (fixed 1 mm mats), a red
fire engine parked on the strip, and a windsock beyond the road whose sleeve streams toward -y (the wind blows
from +y, the left). Five loose beacons (upright cylinders 28 mm across, 28 mm tall, 40 g): two orange and the blue
on the road, two orange on the strip. Rule: mark a helicopter landing zone on the grass, an orange beacon at each
corner of a zone at least 100 x 100 ft (277 mm) with square corners, and the blue beacon 1 to 6 m (10 to 55 mm)
outside the middle of the zone's upwind side. Any orange beacon not standing on the grass, or still held, scores 0.
Gripping from above, each arm reaches only its own side's beacons. The scorer reads the arms (pad-tip points and
openings) through rigstate.py, which wraps the embodiment's step and reset from this file's import, only to check
that each beacon was let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker, bottom_gap, held, on_grass_outline, placed, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. A beacon counts as placed only once let go (no pad-tip point lower than 10 mm above
# its top within the finger reach). Should the arms ever be unreadable, the blue beacon, the last one set down, would
# count from the end of its slow descent while still gripped; settling and letting go take 0.5 + 1.5 s more, so with
# 3 s the trial still cannot end while the fingers are closed on a beacon.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "A medical helicopter is coming to land on this field (1:110 scale). Mark its landing zone on the grass: an "
    "orange beacon at each corner of a zone at least 100 by 100 ft (277 mm) with square corners. Then stand the "
    "blue beacon just outside the middle of the zone's upwind side, 1 to 6 m (10 to 55 mm) beyond it. The windsock "
    "streams downwind. Every beacon must stand upright on the grass on its own."
)


def _mat(name: str, rect, color) -> Box:
    x0, x1, y0, y1 = rect
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + L.MAT_H / 2), size=(x1 - x0, y1 - y0, L.MAT_H),
               color=color)


GRASS = _mat("grass_field", L.GRASS, L.GRASS_GREEN)
ROAD = _mat("road", L.ROAD, L.ASPHALT)
STRIP = _mat("parking_strip", L.STRIP, L.CONCRETE)
ENGINE = Box("fire_engine", pos=(L.ENGINE_X, L.ENGINE_Y, L.MAT_TOP + L.ENGINE_SIZE[2] / 2), size=L.ENGINE_SIZE,
             color=L.ENGINE_RED)
MAST = Cylinder("windsock_mast", pos=(L.MAST_X, L.MAST_Y, L.TOP + L.MAST_H / 2), radius=L.MAST_RADIUS,
                height=L.MAST_H, color=L.WHITE)
SLEEVE = [  # five bands from the mouth at the mast to the tip: orange, white, orange, white, orange
    Box(f"windsock_{i + 1}", pos=(L.MAST_X, L.SLEEVE_Y0 - (i + 0.5) * L.SLEEVE_LEN, L.TOP + L.SLEEVE_Z),
        size=(side, L.SLEEVE_LEN, side), color=L.WHITE if i % 2 else L.SOCK_ORANGE)
    for i, side in enumerate(L.SLEEVE_SIDES)
]


def _beacon(name: str) -> Cylinder:
    x, y = L.START[name]
    return Cylinder(name, pos=(x, y, L.MAT_TOP + L.BEACON_H / 2), radius=L.BEACON_RADIUS, height=L.BEACON_H,
                    mass=L.BEACON_MASS, color=L.BEACON_BLUE if name == L.BLUE else L.BEACON_ORANGE)


BEACONS = [_beacon(n) for n in L.NAMES]
OBJECTS = [GRASS, ROAD, STRIP, ENGINE, MAST, *SLEEVE, *BEACONS]
BEACON_NAMES = tuple(b.name for b in BEACONS)

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the beacons teleporting back)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided. A trace
    that cannot be written is skipped (it must never break a run)."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def _pose(s, path, arms) -> dict:
    """A beacon's centre (mm), tilt (deg), bottom above the grass top (mm), whether its centre is inside the grass
    outline, its path over the last 0.5 s (mm), whether a gripper still holds it and whether the scorer counts it
    placed, for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "bottom_mm": round(bottom_gap(s) * 1000, 1),
                "in_outline": on_grass_outline(s),
                "path_mm": None if path is None else round(path * 1000, 2),
                "held": held(s, arms),
                "placed": placed(s, arms)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """1.0 now: the four orange beacons placed on the grass at the corners of a zone with every side at least
    277 mm and every corner 80-100 degrees, the blue placed 10-55 mm outside the middle third of the upwind side,
    every beacon let go; every beacon still (its centre moved under 5 mm over the last 0.5 s); held
    SUCCESS_HOLD_S = 3 s. This runs first every control step, so it also feeds the tracker (with the arms rigstate
    read for this step)."""
    arms = rigstate.arms()
    beacons = [objects[n] for n in BEACON_NAMES]
    result = _TRACKER.update(beacons, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if isinstance(arms, dict):  # each arm on its own: a missing one is traced as None, never a crash
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    paths = _TRACKER.paths() if result.details.get("valid_state") else None
    row.update({n: _pose(objects[n], None if paths is None else paths[i], arms) for i, n in enumerate(BEACON_NAMES)})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any orange beacon is not placed (upright within 10 degrees, lowest point within 3 mm of the grass top,
    centre inside the grass outline, let go: no gripper's pad-tip point lower than 10 mm above its top within that
    gripper's finger reach, 79 mm fully open, 50 mm closed on a beacon). Else square + wind marker: square 0.5 (every
    side >= 277 mm, corners 80-100 degrees) or 0.25 (>= 249 mm, 70-110 degrees), else 0 in all; marker 0.5 (blue
    placed, 10-55 mm outside the upwind side's line, in its middle third) or 0.25 (placed, outside that side up to
    100 mm, anywhere along it). Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in BEACON_NAMES], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 33 mm either side of a beacon; its neighbours on the road stand 60 mm away
GRIP = -(2 * L.BEACON_RADIUS - 0.008) / 2  # 20 mm commanded across the 28 mm beacon
YAW = 0.0  # both arms: the fingers close along y (a round beacon can be gripped at any heading)
GRIP_UP = 0.020  # the TCP this far above the beacon's bottom: the pad tips 7 mm above it, the pads on 7-28 mm
DROP = 0.001  # each beacon is let go with its bottom 1 mm above the grass
PICK_Z = L.MAT_TOP + GRIP_UP
PLACE_Z = L.MAT_TOP + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a beacon, or onto the grass, are taken slowly
HOVER_Z = L.TOP + 0.080  # travel height: a carried beacon's bottom 60 mm up, 31 mm over a standing beacon's top
PARK = (0.200, 0.250, L.TOP + 0.120)  # each arm ends over its own side's emptied road or strip (y mirrored)
# The zone the demo lays out: 292 x 292 mm (15 mm over the 277 mm minimum), square to the field, its upwind (left)
# side at y +146 mm, 44 mm inside the grass's left edge, so the blue beacon goes 30 mm outside its middle, at
# y +176 mm (its centre 14 mm inside the grass).
ZONE_X = (0.148, 0.440)
ZONE_Y = 0.146
BLUE_AT = ((ZONE_X[0] + ZONE_X[1]) / 2, ZONE_Y + 0.030)
# Where each arm takes which beacon, in order (both arms on one clock, mirror images for the first two):
#   1. its nearer orange beacon to the zone's near corner on its side;
#   2. its other orange beacon to the zone's far corner on its side;
#   3. left only: the blue beacon outside the middle of the upwind (left) side. The right arm parks.
# Then each arm parks over its own side's emptied road or strip, clear of the zone for the top camera.
# The two grippers are never closer than 0.29 m in y.
PLAN = {
    "left": (("orange_1", (ZONE_X[0], ZONE_Y)), ("orange_2", (ZONE_X[1], ZONE_Y)), ("blue", BLUE_AT)),
    "right": (("orange_3", (ZONE_X[0], -ZONE_Y)), ("orange_4", (ZONE_X[1], -ZONE_Y))),
}
# Per-arm trim of the place points (x, y), for the GPU run: if one arm's beacons land off their targets by the
# same amount, read the beacons' pos_mm in TASK_TRACE and move that arm's place points by minus the offset. A 40 g
# beacon barely loads the arm. Offline there is nothing to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _move(name: str, target, arm: str, first: bool) -> list[Waypoint]:
    """Over a beacon, slowly down round it, grip it, lift it, carry it over its place, set it down slowly on the
    grass, let go, straight up."""
    x, y = L.START[name]
    tx, ty = PLACE_TRIM[arm]
    px, py = target[0] + tx, target[1] + ty
    return [
        Waypoint(2.5 if first else 1.8, (x, y, HOVER_Z), OPEN, f"over {name}", YAW),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.2, (x, y, PICK_Z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        Waypoint(2.0, (px, py, HOVER_Z), GRIP, "carry it over its place", YAW),
        Waypoint(1.0, (px, py, PLACE_Z + SLOW), GRIP, "down to just above the grass", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly stand it on the grass", YAW),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, HOVER_Z), OPEN, "straight up, clear of it", YAW),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, (name, target) in enumerate(PLAN[arm]):
        wps += _move(name, target, arm, k == 0)
    sign = 1.0 if arm == "left" else -1.0
    return wps + [Waypoint(1.5, (PARK[0], sign * PARK[1], PARK[2]), OPEN, "park, clear of the zone", YAW)]


def demo():
    """Both arms on one clock, each fetching its own side's beacons. DEMO_HOLD_ONLY=1: no plan (both arms hold the
    ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
