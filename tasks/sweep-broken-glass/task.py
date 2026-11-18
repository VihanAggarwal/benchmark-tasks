"""On the station table, real size: a drinking glass broke into eight pieces in front of the robot. An OXO Good Grips
compact dustpan (186 x 137 x 40 mm) lies with its lip toward the pieces, and its brush stands on its bristles to the
right. Objects (mechkit.py next to this file): the dustpan and the brush (loose compound rigid bodies) and the eight
pieces (loose boxes of 3 mm clear acrylic, the stand-ins used in real trials). Nothing is moved or latched by code.

Goal: sweep every piece into the dustpan with the brush, never touching a piece with a gripper, and leave the dustpan
flat on the table (the brush put down, both grippers let go). See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=one_stroke | no_hold | keep_brush (see demo() below).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built)
import rigstate  # noqa: E402
from scoring import BRUSH, NAMES, PAN, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("A glass broke on the table. Sweep all the pieces into the dustpan with the brush without touching "
               "any of them, then put the brush down and leave the dustpan flat on the table.")
M = L.MM


# -- the objects ----------------------------------------------------------------------------------------------------
def _pan_shapes():
    c = L.PAN_COLOUR
    slope = math.atan2((L.PAN_FLOOR - L.LIP_EDGE) * M, L.LIP_LEN * M)
    ramp_t = 0.5 * M
    ramp_mid_top = (L.PAN_FLOOR + L.LIP_EDGE) / 2 * M
    shapes = [
        mk.box(((L.LIP_LEN + L.PAN_D) / 2 * M, 0.0, L.PAN_FLOOR / 2 * M), ((L.PAN_D - L.LIP_LEN) * M, L.PAN_W * M,
                                                                           L.PAN_FLOOR * M), c),
        mk.box((L.LIP_LEN / 2 * M, 0.0, ramp_mid_top - ramp_t / 2), (L.LIP_LEN * M / math.cos(slope), L.PAN_W * M,
                                                                     ramp_t), c, ypr=(0.0, -slope, 0.0)),
        mk.box(((L.PAN_D - L.PAN_WALL / 2) * M, 0.0, L.PAN_H / 2 * M), (L.PAN_WALL * M, L.PAN_W * M, L.PAN_H * M), c),
    ]
    side_x0 = 30.0  # mm: the side walls start behind the lip (they taper away to nothing at the lip on the real pan)
    for s in (-1.0, 1.0):
        shapes.append(mk.box(((side_x0 + L.PAN_D) / 2 * M, s * (L.PAN_W - L.PAN_WALL) / 2 * M, L.SIDE_H / 2 * M),
                             ((L.PAN_D - side_x0) * M, L.PAN_WALL * M, L.SIDE_H * M), c))
    return shapes


def _brush_shapes():
    z_back = (L.BRISTLE_H + L.BACK_H / 2) * M
    return [
        mk.box((0.0, 0.0, L.BRISTLE_H / 2 * M), (L.HEAD_D * M, L.HEAD_W * M, L.BRISTLE_H * M), L.BRISTLE_COLOUR),
        mk.box((0.0, 0.0, z_back), (L.HEAD_D * M, L.HEAD_W * M, L.BACK_H * M), L.BRUSH_COLOUR),
        mk.box((0.0, -(L.HEAD_W + L.HANDLE_L) / 2 * M, z_back), (L.HANDLE_W * M, L.HANDLE_L * M, L.HANDLE_T * M),
               L.BRUSH_COLOUR),
    ]


PAN_SPEC = mk.assembly(PAN, (L.PAN_LIP_X, L.PAN_Y, L.TOP), _pan_shapes(), mass=L.M_PAN, yaw=math.pi,
                       color=L.PAN_COLOUR, friction=L.PAN_FRICTION)  # pan frame x points back toward the robot
BRUSH_SPEC = mk.assembly(BRUSH, (L.BRUSH_START[0], L.BRUSH_START[1], L.TOP), _brush_shapes(), mass=L.M_BRUSH,
                         color=L.BRUSH_COLOUR, friction=L.BRUSH_FRICTION)
SHARD_SPECS = [Box(name, pos=(x, y, L.TOP + L.SHARD_T / 2 * M), color=L.SHARD_COLOUR, mass=L.shard_mass(sz),
                   yaw=math.radians(yaw), size=(sz[0] * M, sz[1] * M, L.SHARD_T * M))
               for name, (x, y), sz, yaw in L.SHARDS]
OBJECTS = [PAN_SPEC, BRUSH_SPEC, *SHARD_SPECS]


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _r(v, n: int = 5) -> list:
    return [round(float(x), n) for x in np.asarray(v, dtype=float).ravel()]


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """All eight pieces in the dustpan, the pan flat on the table, both grippers clear and everything still for the
    scorer's 20 steps; held 1 s. Runs every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    try:
        row["pieces"] = {n: _r(objects[n].pos, 4) for n in NAMES}
        row["pan"] = {"origin": _r(objects[PAN].origin), "quat_xyzw": _r(objects[PAN].quat_xyzw)}
        row["brush"] = {"origin": _r(objects[BRUSH].origin)}
    except Exception as exc:  # noqa: BLE001
        row["state_error"] = str(exc)[:200]
    if arms is not None:
        try:
            row.update(left_tcp=_r(arms["left"].tcp), right_tcp=_r(arms["right"].tcp),
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 all eight in, pan flat, let go and still; 0.5 six or more (or all, not yet left); 0.25 four or more; 0
    otherwise or if a gripper ever touched a piece or one fell off the table."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop. LEFT pinches the dustpan's back wall from above (pads across the 3 mm wall, their tips 12 mm below its
#   top) at y +50 mm and holds the pan where it lies through the sweeping, then lets go and lifts away.
# RIGHT takes the brush by its handle (pads across it, straight down), lifts it, sets the bristles down behind the
#   pieces, 160 mm from the lip, and sweeps toward the robot at 40 mm/s; its leading face stops 2 mm short of the
#   lip's edge, then it rises 3 mm while it crosses the lip (as bristles bend over a lip) and pushes on onto the
#   pan's floor 35 mm past the lip's edge. Twice. Then it stands the brush back on its bristles where it started and
#   lets go.
OPEN, CLOSED = -0.04695, 0.0
RELEASE = -0.012  # let go by opening the pads only to a 24 mm gap, so the open fingers stay out over nothing but the
#                   wall or the handle (fully open, a finger would stand 47 mm out, over the pieces in the pan)
SQUEEZE = 0.003
HANDLE_GRIP = -(L.HANDLE_W * M / 2 - SQUEEZE)  # finger target: 3 mm inside the 16 mm handle per finger
GRIP_DY = -(L.HEAD_W + L.HANDLE_L) / 2 * M  # the handle's middle, from the head's middle
HANDLE_Z = (L.BRISTLE_H + L.BACK_H / 2) * M  # the handle's centre above the bristle tips
BACK_WALL_X = L.PAN_LIP_X - (L.PAN_D - L.PAN_WALL / 2) * M
PAN_GRIP_Y = 0.050
SWEEP_START_X = L.PAN_LIP_X + 0.160
SWEEP_SPEED = 0.040  # m/s
STROKES = 2
VARIANTS = ("one_stroke", "no_hold", "keep_brush")


def brush_tcp(head_xy, lift: float) -> np.ndarray:
    """The right TCP holding the handle when the bristle footprint's middle is at ``head_xy`` and ``lift`` m above
    the table: the pads either side of the handle, their tips 2 mm below it."""
    tips = L.TOP + lift + HANDLE_Z - L.HANDLE_T / 2 * M - 0.002
    return np.array([head_xy[0], head_xy[1] + GRIP_DY, tips + 0.013])


def pan_tcp(dz: float = 0.0) -> np.ndarray:
    tips = L.TOP + (L.PAN_H - 12.0) * M
    return np.array([BACK_WALL_X, L.PAN_Y + PAN_GRIP_Y, tips + 0.013 + dz])


def _w(duration, tcp, finger, label, yaw) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in tcp), float(finger), label, float(yaw))


YAW_X = math.pi / 2  # pads closing along world x (the right arm)
YAW_XL = -math.pi / 2  # the same for the left arm, inside its wrist's range


def right_plan(strokes: int = STROKES, put_down: bool = True) -> list[Waypoint]:
    s = L.BRUSH_START
    up = np.array([0.0, 0.0, 0.05])
    lip = L.PAN_LIP_X
    plan = [
        _w(2.5, brush_tcp(s, 0.0) + up, OPEN, "over the brush's handle", YAW_X),
        _w(1.2, brush_tcp(s, 0.0), OPEN, "down: the pads either side of the handle", YAW_X),
        _w(0.6, brush_tcp(s, 0.0), HANDLE_GRIP, "close on the handle", YAW_X),
        _w(1.0, brush_tcp(s, 0.04), HANDLE_GRIP, "lift the brush", YAW_X),
    ]
    for k in range(strokes):
        start = (SWEEP_START_X, L.PAN_Y)
        plan += [
            _w(2.0, brush_tcp(start, 0.04), HANDLE_GRIP, f"stroke {k + 1}: behind the pieces", YAW_X),
            _w(1.0, brush_tcp(start, 0.001), HANDLE_GRIP, f"stroke {k + 1}: bristles down on the table", YAW_X),
            _w((SWEEP_START_X - (lip + 0.016)) / SWEEP_SPEED, brush_tcp((lip + 0.016, L.PAN_Y), 0.001), HANDLE_GRIP,
               f"stroke {k + 1}: sweep toward the pan", YAW_X),
            _w(0.8, brush_tcp((lip - 0.012, L.PAN_Y), 0.004), HANDLE_GRIP, f"stroke {k + 1}: over the lip", YAW_X),
            _w(0.6, brush_tcp((lip - 0.035, L.PAN_Y), 0.0045), HANDLE_GRIP, f"stroke {k + 1}: onto the pan's floor",
               YAW_X),
            _w(0.8, brush_tcp((lip - 0.035, L.PAN_Y), 0.04), HANDLE_GRIP, f"stroke {k + 1}: lift", YAW_X),
        ]
    if put_down:
        plan += [
            _w(2.0, brush_tcp(s, 0.04), HANDLE_GRIP, "back over its spot", YAW_X),
            _w(1.0, brush_tcp(s, 0.002), HANDLE_GRIP, "stand it on its bristles", YAW_X),
            _w(0.6, brush_tcp(s, 0.002), RELEASE, "let go", YAW_X),
            _w(1.0, brush_tcp(s, 0.002) + up, RELEASE, "up", YAW_X),
            _w(1.0, brush_tcp(s, 0.002) + up + np.array([-0.05, -0.03, 0.05]), OPEN, "clear", YAW_X),
        ]
    else:
        plan.append(_w(4.0, brush_tcp((lip - 0.035, L.PAN_Y), 0.04), HANDLE_GRIP, "keep holding the brush", YAW_X))
    return plan


def left_plan(hold_until: float) -> list[Waypoint]:
    up = np.array([0.0, 0.0, 0.05])
    t0 = 2.0 + 1.0 + 0.6
    return [
        _w(2.0, pan_tcp() + up, OPEN, "over the dustpan's back wall", YAW_XL),
        _w(1.0, pan_tcp(), OPEN, "down: the pads either side of the wall", YAW_XL),
        _w(0.6, pan_tcp(), CLOSED, "pinch the back wall", YAW_XL),
        _w(max(0.5, hold_until - t0), pan_tcp(), CLOSED, "hold the dustpan where it lies", YAW_XL),
        _w(0.6, pan_tcp(), RELEASE, "let go", YAW_XL),
        _w(1.0, pan_tcp() + up, RELEASE, "up", YAW_XL),
        _w(1.0, pan_tcp() + up + np.array([-0.03, 0.06, 0.05]), OPEN, "clear", YAW_XL),
        _w(2.0, pan_tcp() + up + np.array([-0.03, 0.06, 0.05]), OPEN, "hold", YAW_XL),
    ]


def _time_of(plan, label_start: str) -> float:
    t = 0.0
    for w in plan:
        if w.label.startswith(label_start):
            return t
        t += w.duration
    return t


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = right_plan(1 if variant == "one_stroke" else STROKES, put_down=variant != "keep_brush")
    swept = _time_of(right, "back over its spot") if variant != "keep_brush" else sum(w.duration for w in right)
    if variant == "no_hold":  # the left arm stays at its ready pose
        return {"right": right}
    return {"left": left_plan(swept + 0.5), "right": right}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      one_stroke  a single sweep
      no_hold     the dustpan is not held while the brush sweeps
      keep_brush  everything swept in, but the brush is never put down (0.5 at most)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
