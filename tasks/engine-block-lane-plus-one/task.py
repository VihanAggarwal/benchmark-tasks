"""On the station table, at 1:80 scale: a one-way, two-lane highway running across the table (traffic from the robot's
left to its right, toward -y), with a grass verge on the near side, a near shoulder, two lanes, a far shoulder and a
grass median; white edge line, dashed white lane line, yellow far edge line and four lane arrows. A crashed blue car
stands askew in the near lane (lane 1). A red fire engine (125 x 31 x 40 mm, 80 g, loose) waits upstream in the far
lane, heading downstream, its silver pump panel on its left side; a white ambulance (88 x 30 x 35 mm, 50 g, loose)
waits downstream on the near shoulder, facing upstream. Rule (traffic incident management): the engine parked
upstream of the car, angled 30 to 60 degrees to the lanes across both lanes ("lane + 1"), its pump-panel side facing
downstream (toward the car), its nearest point 50 to 125 mm (4 to 10 m) short of the car; the ambulance parked
downstream, straight in either lane, rear doors toward the car, 50 to 125 mm beyond it; neither touching the car.

Each vehicle is a mechkit compound (mechkit.py, next to this file: one loose rigid body of five or six boxes), so its
front and its pump-panel side can be seen and the scorer can read which way it faces. It changes nothing else the
embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the vehicles are built as compound bodies)
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, both vehicles parked still) must hold 3 s, not the default 1 s. A vehicle counts as parked once its
# lowest point is within 3 mm of the road, which happens at the end of the slow descent while the grippers still hold
# the vehicles (sooner on the GPU, where the arms sag a few mm under the load); settling, letting go and rising take
# 0.6 + 1.5 + 1.5 s, so the trial ends with the fingers open, going up.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "A crashed car is stopped in the near lane of this one-way, two-lane highway (1:80 scale; traffic comes from your "
    "left, as the arrows show). Park the fire engine upstream of the car, angled 30 to 60 degrees to the lanes so it "
    "blocks both lanes, its silver pump-panel side facing the car, 50 to 125 mm (4 to 10 m) short of it. Park the "
    "ambulance downstream, straight in either lane, rear doors toward the car, 50 to 125 mm beyond it. Neither may "
    "touch the car or the other vehicle."
)

_MARK_Z = L.TOP + L.ROAD_TOP + L.MARK_T / 2


def _span_box(name, xs, ys, zs, colour, **kw) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    (x0, x1), (y0, y1), (z0, z1) = xs, ys, zs
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=L.COLOURS[colour], **kw)


def _arrow(name: str, xc: float, yc: float) -> list[Box]:
    """A lane arrow pointing downstream (-y): a shaft along y and two head strokes meeting at the tip."""
    tip = yc - L.ARROW_TIP
    shaft = Box(f"{name}_shaft", pos=(xc, yc, _MARK_Z), size=(L.ARROW_W, 2 * L.ARROW_TIP, L.MARK_T),
                color=L.COLOURS["white"])
    heads = [
        Box(f"{name}_head_{side}", pos=(xc + sgn * L.HEAD_DX, tip + L.HEAD_DY, _MARK_Z),
            size=(L.ARROW_W, L.HEAD_LEN, L.MARK_T), color=L.COLOURS["white"], yaw=-sgn * math.pi / 4)
        for side, sgn in (("a", -1.0), ("b", 1.0))
    ]
    return [shaft, *heads]


VERGE = _span_box("verge", L.VERGE_X, L.ROAD_Y, (0.0, L.SHEET_H), "grass")
MEDIAN = _span_box("median", L.MEDIAN_X, L.ROAD_Y, (0.0, L.SHEET_H), "grass")
# The road: strips 1 mm thick laid side by side, the painted lines among them flush with the asphalt.
ROAD = [_span_box(name, xs, ys, (0.0, L.SHEET_H), colour) for name, xs, ys, colour in L.road_strips()]
ARROWS = [b for name, x, y in L.ARROWS for b in _arrow(name, x, y)]
CAR = Box("crashed_car", pos=(*L.CAR_XY, L.TOP + L.CAR_LOW_Z + L.CAR_SIZE[2] / 2), size=L.CAR_SIZE,
          color=L.COLOURS["car"], yaw=L.CAR_YAW)
CAR_CABIN = Box("crashed_car_cabin",
                pos=(*L.CAR_XY, L.TOP + L.CAR_LOW_Z + L.CAR_SIZE[2] + L.CAR_CABIN_SIZE[2] / 2),
                size=L.CAR_CABIN_SIZE, color=L.COLOURS["car_glass"], yaw=L.CAR_YAW)
ROADWAY = [VERGE, *ROAD, MEDIAN, *ARROWS, CAR, CAR_CABIN]


def _vehicle(name: str, shapes, mass: float, colour: str):
    x, y, heading = L.START[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP + L.ROAD_TOP),
        yaw=heading,
        mass=mass,
        color=L.COLOURS[colour],
        shapes=[mk.box(c, s, color=L.COLOURS[k]) for _, c, s, k in shapes],
    )


FIRE_ENGINE = _vehicle("fire_engine", L.ENGINE_SHAPES, L.ENGINE_MASS, "engine_red")
AMBULANCE = _vehicle("ambulance", L.AMB_SHAPES, L.AMB_MASS, "amb_white")
OBJECTS = [*ROADWAY, FIRE_ENGINE, AMBULANCE]

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
    """Both vehicles set (score 1.0): the engine parked upstream, angled 30-60 degrees across both lanes, pump panel
    downstream, 50-125 mm short of the car; the ambulance parked downstream, straight in a lane, rear doors to the car,
    50-125 mm beyond it; neither within 2 mm of the car or of each other; held 3 s. This runs first every control
    step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    engine, amb = objects.get("fire_engine"), objects.get("ambulance")
    result = _TRACKER.update(engine, amb)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(source=getattr(engine, "source", None),
                   origins={"fire_engine": engine.origin, "ambulance": amb.origin},
                   quats={"fire_engine": engine.quat_xyzw, "ambulance": amb.quat_xyzw},
                   low_z={"fire_engine": engine.low[2], "ambulance": amb.low[2]},
                   vels={"fire_engine": engine.lin_vel, "ambulance": amb.lin_vel})
    except (AttributeError, IndexError, TypeError, ValueError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if either vehicle is on its side or roof (tilted over 45 degrees), or within 2 mm (seen from above) of the
    car or of the other vehicle. Otherwise the engine earns 0.5 if parked (tilted 10 degrees or less, lowest point
    within 3 mm of the road, centre over the road, still over 0.5 s) upstream of the car, its long axis 30-60 degrees
    to the lanes, its outline spanning both lanes, its pump-panel side facing downstream and its nearest point 50-125
    mm short of the car; 0.25 if parked upstream within 250 mm and covering 28 mm or more of the crash lane but not
    all of that; else 0. The ambulance earns 0.5 if parked downstream with its centre in a lane, its front within 15
    degrees of downstream and its nearest point 50-125 mm beyond the car; 0.25 if parked downstream within 250 mm
    with its centre in a lane but not all of that; else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result(objects.get("fire_engine"), objects.get("ambulance")).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Both arms at once, each with the vehicle on its own side: the left arm takes the fire engine from the far lane,
# carries it 19 cm downstream and turns it 45 degrees clockwise (seen from above) into the angled block; the right
# arm takes the ambulance from the near shoulder, carries it 18 cm upstream into lane 1 and turns it half a turn so
# its rear doors face the car. Each grip is across the vehicle's width (the fingers close along its left-right) and
# the turn is made with the wrist while the vehicle is carried: the vehicle's heading changes with the grip's yaw.
# Yaw 0 closes the fingers along world y, so a vehicle heading h is held at yaw h (or h + 180 degrees: the parallel
# gripper is symmetric). The yaws keep joint 6 well inside its +-120 degrees: the left arm holds the engine at +90
# (joint 6 about -82 deg) and turns to +45 (-78 deg); the right arm holds the ambulance at -90 (+69 deg) and turns
# through 0 to +90 (-61 deg), in two quarter turns so each segment's interpolation is unambiguous.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # 94 mm
GRIP = {"fire_engine": -(0.031 - 0.008) / 2, "ambulance": -(0.030 - 0.008) / 2}  # 23 and 22 mm commanded
# Where along the vehicle the grip is (m forward of its origin): the engine is held 10 mm behind its middle, so the
# 27 mm pads (u -23.5 .. +3.5) stay 11.5 mm clear of the pump panel (u 15 .. 30); the ambulance at its middle.
GRIP_U = {"fire_engine": -0.010, "ambulance": 0.0}
# The TCP 25 mm (engine) and 24 mm (ambulance) above the vehicle's base: the pads (from 25 mm above the TCP to their
# tips 13 mm below it) on the body from 12 or 11 mm up to its roof (40 / 35 mm), and the gripper's body (from 35 mm
# above the TCP) 17 or 21 mm over the light bar (43 / 38 mm).
GRIP_H = {"fire_engine": 0.025, "ambulance": 0.024}
LIFT = 0.030  # the vehicle lifted 30 mm to carry it
APPROACH = 0.030  # the last 30 mm down to the grip are slow
PLACE_UP = 0.002  # let go with the vehicle's base 2 mm above the road
ABOVE_Z = L.TOP + 0.090  # pad tips 77 mm up, 34 mm over the engine's light bar
PARK = {"left": (0.200, 0.300, L.TOP + 0.150), "right": (0.200, -0.300, L.TOP + 0.150)}
SIDE = {"left": "fire_engine", "right": "ambulance"}
YAWS = {  # (at the grip, half-way through the carry, at the place)
    "fire_engine": (math.pi / 2, 3 * math.pi / 8, math.pi / 4),
    "ambulance": (-math.pi / 2, 0.0, math.pi / 2),
}


def grip_point(name: str, origin_xy, heading: float) -> tuple[float, float]:
    """Where the grip is (world x, y) for a vehicle whose origin is at ``origin_xy`` heading ``heading``."""
    u = GRIP_U[name]
    return (origin_xy[0] + u * math.cos(heading), origin_xy[1] + u * math.sin(heading))


def _arm(arm: str) -> list[Waypoint]:
    """One arm: over its vehicle, down slowly, grip across its width, lift 30 mm, carry it to its place turning it
    (two segments), lower it slowly, let go, straight up, park."""
    name = SIDE[arm]
    x0, y0, h0 = L.START[name]
    x1, y1, h1 = L.TARGET[name]
    sx, sy = grip_point(name, (x0, y0), h0)
    px, py = grip_point(name, (x1, y1), h1)
    mx, my = (sx + px) / 2, (sy + py) / 2
    ys, ym, yp = YAWS[name]
    grip_z = L.TOP + L.ROAD_TOP + GRIP_H[name]
    carry_z, place_z, approach_z = grip_z + LIFT, grip_z + PLACE_UP, grip_z + APPROACH
    grip = GRIP[name]
    return [
        Waypoint(3.0, (sx, sy, ABOVE_Z), OPEN, f"over the {name.replace('_', ' ')}", ys),
        Waypoint(1.5, (sx, sy, approach_z), OPEN, "down beside its body", ys),
        Waypoint(1.5, (sx, sy, grip_z), OPEN, "down the last 30 mm slowly", ys),
        Waypoint(0.5, (sx, sy, grip_z), OPEN, "settle", ys),
        Waypoint(1.5, (sx, sy, grip_z), grip, "grip across the body", ys),
        Waypoint(1.5, (sx, sy, carry_z), grip, "lift 30 mm", ys),
        Waypoint(2.0, (mx, my, carry_z), grip, "carry it, turning it", ym),
        Waypoint(2.0, (px, py, carry_z), grip, "carry it to its place, finishing the turn", yp),
        Waypoint(0.5, (px, py, carry_z), grip, "settle over the place", yp),
        Waypoint(2.0, (px, py, place_z), grip, "lower it slowly", yp),
        Waypoint(0.6, (px, py, place_z), grip, "settle", yp),
        Waypoint(1.5, (px, py, place_z), OPEN, "let go", yp),
        Waypoint(1.5, (px, py, ABOVE_Z), OPEN, "straight up", yp),
        Waypoint(2.0, PARK[arm], OPEN, "out of the way"),
    ]


def demo():
    """Both arms on one clock, each setting its own vehicle. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _arm("left"), "right": _arm("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
