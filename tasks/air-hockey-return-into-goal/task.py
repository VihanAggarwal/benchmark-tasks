"""On the station table, real size: a tabletop air hockey game (a 60 x 32 cm playfield with rails, a goal slot at each
end) running along x, the robot's goal at the near end. The yellow mallet stands in front of the robot's goal; the
puck waits at the far end, 8 cm left of centre, in front of a practice kicker that sends it down the table.

Objects (mechkit.py next to this file): the game (one fixed mechanism: the kicker slides on a driven slider), the puck
and the mallet (loose bodies sliding on the nearly frictionless playfield, the air cushion's stand-in; the mallet's rim
gives the puck back 75% of its speed, other contacts are dead). One rule drives the kicker: 7 s into the trial it sends the puck toward the robot at
about 0.55 m/s, then draws back. Nothing else is moved by code.

Goal: hold the mallet by its handle and hit the puck coming down the table into the far goal. The puck comes down the
left of the table, so a block sends it back into the end rail: the hit has to be aimed. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=block | late (see demo() below).
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
from scoring import MALLET, PUCK, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Play air hockey: hold the mallet by its handle and hit the puck that slides toward you so that it goes "
               "into the far goal. Keep the mallet on your half and your hands off the puck.")

GAME, KICKER = "air_hockey", "kicker"


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _game():
    x0, x1, w, s, rh, rt, g = L.X0, L.X1, L.HALF_W, L.SURFACE, L.RAIL_H, L.RAIL_T, L.GOAL_HALF
    cx, length = (x0 + x1) / 2, x1 - x0
    shapes = [mk.box((cx, 0.0, s / 2), (length, 2 * w, s), L.BODY),
              mk.box((cx, 0.0, s - 0.001), (length - 0.004, 2 * w - 0.004, 0.002), L.FIELD),
              mk.box((cx, w + rt / 2, s + rh / 2), (length + 2 * rt, rt, rh), L.RAIL),
              mk.box((cx, -w - rt / 2, s + rh / 2), (length + 2 * rt, rt, rh), L.RAIL)]
    for xe, sign in ((x0, -1), (x1, 1)):
        side = w + rt - g  # from the goal slot to the side rail's outer face
        for sy in (-1, 1):  # the end rails either side of the goal slot
            shapes.append(mk.box((xe + sign * rt / 2, sy * (g + side / 2), s + rh / 2), (rt, side, rh), L.RAIL))
        shapes.append(mk.box((xe + sign * 0.030, 0.0, s - 0.020), (0.040, 2 * g + 0.02, 0.006), L.GOAL_BOX))  # goal box
        shapes.append(mk.box((xe + sign * 0.055, 0.0, (s + rh) / 2), (0.010, 2 * g + 0.04, s + rh), L.GOAL_BOX))  # backstop
    kicker = mk.slider(KICKER, "-X", [mk.box((L.KICKER_X + 0.010, L.KICKER_Y, s + 0.008), (0.020, 0.050, 0.014),
                                             (0.30, 0.30, 0.32))],
                       upper=L.KICK_TRAVEL + 0.005, mass=0.05, spring=5000.0, damping=60.0)
    return mk.assembly(GAME, (0.0, 0.0, L.TOP), shapes, joints=[kicker], color=L.BODY, friction=L.SURFACE_FRICTION)


def _puck():
    s = [mk.cylinder((0.0, 0.0, 0.0), L.PUCK_R, L.PUCK_H, "Z", L.PUCK_COLOUR)]
    return mk.assembly(PUCK, _f(L.at(*L.PUCK_START, L.PUCK_H / 2 + 0.001)), s, mass=L.M_PUCK, color=L.PUCK_COLOUR,
                       friction=L.SURFACE_FRICTION)


def _mallet():
    s = [mk.cylinder((0.0, 0.0, 0.0), L.MALLET_R, L.MALLET_H, "Z", L.MALLET_COLOUR),
         mk.cylinder((0.0, 0.0, L.MALLET_H / 2 + L.HANDLE_H / 2), L.HANDLE_R, L.HANDLE_H, "Z", L.MALLET_COLOUR)]
    return mk.assembly(MALLET, _f(L.at(*L.MALLET_START, L.MALLET_H / 2 + 0.001)), s, mass=L.M_MALLET,
                       color=L.MALLET_COLOUR, friction=(0.15, 0.10), restitution=L.RESTITUTION)


GAME_SPEC, PUCK_SPEC, MALLET_SPEC = _game(), _puck(), _mallet()
OBJECTS = [GAME_SPEC, PUCK_SPEC, MALLET_SPEC]


# -- the kicker (the one rule) --------------------------------------------------------------------------------------
_CLOCK = {"step": 0, "episode": None}


@mk.on_step
def _kicker(states) -> None:
    """7 s in, the kicker's face moves 40 mm toward the robot at 0.55 m/s (sending the puck), waits, draws back."""
    if _CLOCK["episode"] != mk.episode():
        _CLOCK.update(step=0, episode=mk.episode())
    _CLOCK["step"] += 1
    t = _CLOCK["step"] * L.DT - L.KICK_AT
    stroke = L.KICK_TRAVEL / L.KICK_SPEED
    if t <= 0:
        target = 0.0
    elif t <= stroke:
        target = L.KICK_SPEED * t
    elif t <= stroke + 0.5:
        target = L.KICK_TRAVEL
    else:
        target = max(0.0, L.KICK_TRAVEL - 0.05 * (t - stroke - 0.5))
    mk.set_targets(GAME_SPEC, KICKER, target)


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """The puck hit into the far goal with the mallet, hands off it, own goal not conceded; held 1 s. Runs every
    control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if arms is not None:
        try:
            row["left_tcp"] = [round(float(v), 4) for v in arms["left"].tcp]
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 into the far goal; 0.5 returned past the halfway line; 0.25 hit with the mallet; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, the left arm only; nothing read from the simulator. A straight-down grip on the mallet's handle, the pads
# either side of it. The hand slides the mallet (on the table) to a ready spot behind where the puck will be, then at
# STRIKE_AT drives it STROKE m along AIM (from the puck toward the far goal, turned a little against the puck's own
# motion) in STROKE_S s, meeting the puck mid-stroke. The timing and the aim were worked out on the simulator.
OPEN = -0.04695
HANDLE_GRIP = -(L.HANDLE_R - 0.004)
GRIP_ABOVE = 0.008
CONTACT = np.array([0.360, L.PUCK_START[1]])  # where the puck is met
AIM_DEG = float(os.environ.get("HOCKEY_AIM", -8.0))  # the stroke's direction from +x (toward the far goal, right)
STRIKE_AT = float(os.environ.get("HOCKEY_STRIKE", 7.42))
STROKE = 0.10
STROKE_S = 0.20
VARIANTS = ("block", "late", "idle")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


PRESS = 0.002  # the hand holds the mallet 2 mm down onto the table, so it doesn't tip up as it strikes


def handle_tcp(xy) -> np.ndarray:
    return L.at(xy[0], xy[1], L.MALLET_H + L.HANDLE_H / 2 + GRIP_ABOVE - PRESS)


def aim(deg: float = AIM_DEG) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def left_plan(strike_at: float = STRIKE_AT, block: bool = False) -> list[Waypoint]:
    d = aim()
    meet = CONTACT - d * (L.PUCK_R + L.MALLET_R)  # the mallet's centre when it meets the puck
    ready = meet - d * STROKE / 2
    up = np.array([0.0, 0.0, 0.05])
    w = [_w(2.5, handle_tcp(L.MALLET_START) + up, OPEN, "over the mallet's handle"),
         _w(1.0, handle_tcp(L.MALLET_START), OPEN, "down: the pads either side of it"),
         _w(0.6, handle_tcp(L.MALLET_START), HANDLE_GRIP, "grip the handle"),
         _w(1.5, handle_tcp(ready if not block else meet - d * 0.01), HANDLE_GRIP,
            "slide the mallet to the ready spot, on the table")]
    t = sum(x.duration for x in w)
    if block:  # hold the mallet in the puck's way: it bounces straight back
        w.append(_w(8.0, handle_tcp(meet - d * 0.01), HANDLE_GRIP, "hold it in the puck's way"))
    else:
        w += [_w(strike_at - t, handle_tcp(ready), HANDLE_GRIP, "wait for the puck"),
              _w(STROKE_S, handle_tcp(ready + d * STROKE), HANDLE_GRIP, "strike: drive the mallet through the puck"),
              _w(6.0, handle_tcp(ready + d * STROKE), HANDLE_GRIP, "hold")]
    return w


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "block":
        return {"left": left_plan(block=True)}
    if variant == "idle":  # the hand waits over the mallet and never touches it
        return {"left": [_w(2.5, handle_tcp(L.MALLET_START) + [0.0, 0.0, 0.08], OPEN, "over the mallet"),
                         _w(16.0, handle_tcp(L.MALLET_START) + [0.0, 0.0, 0.08], OPEN, "wait")]}
    if variant == "late":  # the same strike 0.3 s late: the puck has gone past
        return {"left": left_plan(strike_at=STRIKE_AT + 0.3)}
    return {"left": left_plan()}


def demo():
    """The left arm plays; the right stays parked. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      block  the mallet held still in the puck's way: it bounces back into the end rail
      late   the strike 0.3 s late
      idle   the hand waits over the mallet: the puck passes it and stops against the near end rail"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
