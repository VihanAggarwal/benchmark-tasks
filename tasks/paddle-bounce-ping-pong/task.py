"""On the station table, real size: a table tennis paddle lying red face up on a foam block in front of the right arm,
its handle toward the robot, and a ping-pong ball in a small stand in front of the left arm.

Objects (mechkit.py next to this file): the paddle (one loose rigid body: a round blade 154 mm across and 6 mm thick
with a 100 mm handle; its surface springs the ball back with restitution 0.85), the ball (a loose sphere, 40 mm,
2.7 g), the foam block and the stand (fixed). Nothing is moved or latched by code.

Goal: pick up the paddle, drop the ball onto it from no more than 150 mm, and keep the ball bouncing on the red face,
its bottom 100 to 250 mm above the face each time, ten times in a row. The robot has to hold the paddle level and meet each falling ball
with a small upward tap in rhythm: a still paddle lets the bounces die away, and a tilted or mistimed one sends the
ball off. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=still | high_drop (see demo() below).
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
import tiltgrip  # noqa: E402  (importing it lets a waypoint carry a tilted grip)
from scoring import BALL, PADDLE, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Sphere  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Pick up the paddle, drop the ping-pong ball onto it from no more than 15 cm above the face, and keep "
               "it bouncing on the red face ten times in a row, each bounce 10 to 25 cm high.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _paddle():
    s = [mk.cylinder((0.0, 0.0, 0.0), L.BLADE_R, L.BLADE_T, "Z", L.RED),
         mk.box((L.HANDLE_X, 0.0, 0.0), (L.HANDLE_L, L.HANDLE_W, L.HANDLE_T), L.WOOD)]
    return mk.assembly(PADDLE, _f(L.PADDLE_START), s, mass=L.M_PADDLE, yaw=L.PADDLE_YAW, color=L.RED,
                       friction=L.PADDLE_FRICTION, restitution=L.RESTITUTION)


def _stand():
    c = (float(L.STAND[0]), float(L.STAND[1]), L.TOP + L.STAND_H / 2)
    return mk.assembly("stand", (0.0, 0.0, 0.0), mk.open_box(c, (L.STAND_OUTER, L.STAND_OUTER, L.STAND_H), L.STAND_WALL),
                       color=L.STAND_COLOUR)


PADDLE_SPEC = _paddle()
BALL_SPEC = Sphere(BALL, pos=_f(L.BALL_START), color=L.BALL_COLOUR, mass=L.M_BALL, radius=L.BALL_R)
BLOCK_SPEC = Box("foam_block", pos=(float(L.BLOCK[0]), float(L.BLOCK[1]), L.TOP + L.BLOCK_SIZE[2] / 2),
                 size=_f(L.BLOCK_SIZE), color=L.FOAM)
OBJECTS = [PADDLE_SPEC, BALL_SPEC, BLOCK_SPEC, _stand()]


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _r(v, n: int = 4) -> list:
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
    """Ten bounces in a row; held 1 s. Runs every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row.update(ball=_r(objects[BALL].pos), paddle=_r(objects[PADDLE].origin),
                   paddle_q=_r(objects[PADDLE].quat_xyzw, 3))
    except Exception as exc:  # noqa: BLE001
        row["objects_error"] = str(exc)[:200]
    if arms is not None:
        try:
            row.update(left_tcp=_r(arms["left"].tcp), right_tcp=_r(arms["right"].tcp),
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 ten bounces in a row; 0.5 six; 0.25 four; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, nothing read from the simulator. The right hand takes the paddle's handle in a straight-down grip, the
# pads either side of it, and lifts the paddle level to the bouncing spot. The left hand takes the ball from its stand,
# carries it over the blade's centre and lets it go 120 mm above the face. From then on the right hand moves the paddle
# straight up and down, AMP each way at one cycle per bounce (PERIOD), the bottom of each stroke timed (PHASE, from the
# simulator) so each falling ball meets the face on its way up: each tap gives back what the bounce loses.
OPEN = -0.04695
HANDLE_GRIP = -(L.HANDLE_T / 2 - 0.004)  # the pads on the handle's top and bottom flats
HANDLE_OPEN = -0.025  # coming in along the handle, the lower finger clear under it
BALL_GRIP = -(L.BALL_R - 0.005)
GRIP_ABOVE = 0.008  # the TCP above the gripped part's centre, the pads straddling it
HOLD = np.array([0.360, 0.000, L.TOP + 0.145])  # the blade's centre while bouncing (over its block, 42 mm up)
SIDE_AZIMUTH = math.radians(-15.0)  # the right hand comes in along the handle from the robot's right, turned 15 deg
DROP_PITCH = math.radians(-60.0)  # carrying and dropping, the left hand leans in from behind the blade (pitched about y)
DROP = 0.120  # the ball's bottom above the face when let go
RELEASE_AT = 9.0  # s: the left hand starts to open
# The rhythm, tuned on the simulator (open loop; the simulator repeats it exactly):
PERIOD = 0.35  # s: one tap per bounce; a flight this long rises about 150 mm
AMP = 0.012  # m: the stroke, each way from HOLD
PHASE = 0.035  # s from the release to the first stroke's bottom
TRIM = (0.0, math.radians(0.63))  # the face tipped back about world y and x against the held paddle's lean
CYCLES = 18
VARIANTS = ("still", "high_drop")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    yaw = yaw if isinstance(yaw, tiltgrip.TiltedYaw) else float(yaw)  # a tilted grip keeps its rotation
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, yaw)


def _side_rotation() -> np.ndarray:
    """The right hand's grip: pointing horizontally along the handle toward the blade (turned SIDE_AZIMUTH toward
    +x), the pads closing vertically on the handle's flats. A pinch resists the blade's weight in the pads' plane;
    a straight-down grip across the handle let it droop."""
    return mk.axis_angle_matrix((0, 0, 1), SIDE_AZIMUTH) @ tiltgrip.rotation(0.0, math.pi / 2)


SWING = math.radians(1.3)  # the face tipped against the wrist's flex, which leans it toward +x at the bottom of each
# stroke (accelerating up) and toward -x at the top: tipped -SWING at the bottom and +SWING at the top


def trimmed(swing: float = 0.0) -> float:
    """The right hand's grip, the face tipped back by the trim and ``swing`` (rad) about world y."""
    r = mk.axis_angle_matrix((1, 0, 0), TRIM[1]) @ mk.axis_angle_matrix((0, 1, 0), -(TRIM[0] + swing))
    return tiltgrip.TiltedYaw(0.0, r @ _side_rotation())


ACROSS = tiltgrip.TiltedYaw(0.0, _side_rotation())
TRIMMED = trimmed()


def handle_tcp(blade_centre, back: float = 0.0) -> np.ndarray:
    """The TCP gripping the handle's middle (``back`` m further back along the approach), the blade's centre at
    ``blade_centre`` and the face level."""
    a = _side_rotation()[:, 2]
    return np.asarray(blade_centre, dtype=float) + np.array([0.0, L.HANDLE_X, 0.0]) - (GRIP_ABOVE + back) * a


def ball_tcp(centre) -> np.ndarray:
    return np.asarray(centre, dtype=float) + np.array([0.0, 0.0, GRIP_ABOVE])


def drop_centre(drop: float = DROP) -> np.ndarray:
    """The ball's centre when let go, ``drop`` (its bottom) above the face at HOLD."""
    return HOLD + np.array([0.0, 0.0, L.BLADE_T / 2 + drop + L.BALL_R])


def right_plan(bounce: bool = True) -> list[Waypoint]:
    start = L.PADDLE_REST
    up = np.array([0.0, 0.0, 0.06])
    w = [
        _w(2.5, handle_tcp(start, 0.08) + up, HANDLE_OPEN, "beside the handle's end, level with it", ACROSS),
        _w(1.0, handle_tcp(start, 0.08), HANDLE_OPEN, "down to the handle's height", ACROSS),
        _w(1.2, handle_tcp(start), HANDLE_OPEN, "in along the handle, a pad above it and one below", ACROSS),
        _w(0.8, handle_tcp(start), HANDLE_GRIP, "close on the handle's flats", ACROSS),
        _w(1.5, handle_tcp(HOLD), HANDLE_GRIP, "lift the paddle level off its block", TRIMMED),
    ]
    t = sum(x.duration for x in w)
    first = RELEASE_AT + PHASE
    if not bounce:
        return w + [_w(first - t + CYCLES * PERIOD + 3.0, handle_tcp(HOLD), HANDLE_GRIP, "hold the paddle still", TRIMMED)]
    lo, hi = handle_tcp(HOLD - [0, 0, AMP]), handle_tcp(HOLD + [0, 0, AMP])
    w += [_w(first - t - PERIOD / 2, handle_tcp(HOLD), HANDLE_GRIP, "hold, waiting for the ball", TRIMMED),
          _w(PERIOD / 2, lo, HANDLE_GRIP, "down for the first tap", trimmed(SWING))]
    for i in range(CYCLES):
        w += [_w(PERIOD / 2, hi, HANDLE_GRIP, f"tap {i + 1}: up", trimmed(-SWING)),
              _w(PERIOD / 2, lo, HANDLE_GRIP, f"tap {i + 1}: down", trimmed(SWING))]
    w += [_w(PERIOD / 2, handle_tcp(HOLD), HANDLE_GRIP, "stop", TRIMMED), _w(3.0, handle_tcp(HOLD), HANDLE_GRIP, "hold", TRIMMED)]
    return w


def _pitched(pitch: float = DROP_PITCH) -> np.ndarray:
    return mk.axis_angle_matrix((0, 1, 0), pitch) @ tiltgrip.rotation(0.0, 0.0)


def left_tilt(pitch: float = DROP_PITCH) -> float:
    return tiltgrip.TiltedYaw(0.0, _pitched(pitch))


def tilted_tcp(centre, pitch: float = DROP_PITCH) -> np.ndarray:
    """The TCP holding the ball at ``centre`` with the hand leaning in from behind it: GRIP_ABOVE back along the
    hand's approach."""
    return np.asarray(centre, dtype=float) - GRIP_ABOVE * _pitched(pitch)[:, 2]


def left_plan(drop: float = DROP) -> list[Waypoint]:
    b = L.BALL_START
    up = np.array([0.0, 0.0, 0.06])
    over = drop_centre(drop)
    lift = b + np.array([0.0, 0.0, 0.16])
    tilt = left_tilt()
    away = tilted_tcp(over) + np.array([-0.08, 0.16, 0.0])
    w = [
        _w(2.5, ball_tcp(b) + up, OPEN, "over the ball"),
        _w(1.2, ball_tcp(b), OPEN, "down: the pads either side of the ball"),
        _w(0.8, ball_tcp(b), BALL_GRIP, "close on the ball"),
        _w(1.0, ball_tcp(lift), BALL_GRIP, "lift it out of the stand"),
        _w(1.5, tilted_tcp(lift), BALL_GRIP, "lean the hand in from behind", tilt),
        _w(1.5, tilted_tcp(over + [-0.02, 0.0, 0.02]), BALL_GRIP, "over the blade's centre", tilt),
    ]
    t = sum(x.duration for x in w)
    w += [
        _w(RELEASE_AT - t, tilted_tcp(over), BALL_GRIP, "to the drop height, still", tilt),
        _w(0.12, tilted_tcp(over), OPEN, "let go: the ball drops onto the face", tilt),
        _w(0.45, away, OPEN, "straight away back and to the left, clear of the bounce", tilt),
        _w(CYCLES * PERIOD + 2.0, away, OPEN, "hold", tilt),
    ]
    return w


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "still":  # the paddle held still: the bounces die away
        return {"right": right_plan(bounce=False), "left": left_plan()}
    if variant == "high_drop":  # the ball let go 250 mm above the face, the paddle held still
        return {"right": right_plan(bounce=False), "left": left_plan(0.25)}
    return {"right": right_plan(), "left": left_plan()}


def demo():
    """The right arm holds and bounces the paddle; the left drops the ball. DEMO_HOLD_ONLY=1: no plan (a settling
    check). DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      still      the paddle held still under the drop: the bounces die away
      high_drop  let go 250 mm above a still paddle: the passive bounces do not count"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
