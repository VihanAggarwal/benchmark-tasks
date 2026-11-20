"""On the station table, real size: a desktop claw machine clamped in front of the robot, its control deck toward the
robot with an 8-way bat-top joystick on the right and a round drop button on the left; three boxed prizes in its pit
and the prize chute in the pit's front-left corner. One credit is loaded.

Objects (mechkit.py next to this file): the machine (one fixed mechanism: the joystick tilts on two sprung hinges, the
button slides down on a spring, and the claw rides a gantry: two sliders and a drop slider) and the three prizes
(loose 50 mm boxes). One rule stands in for the machine's controller (scoring.Program): the stick's switches drive the
claw at 50 mm/s, the button starts the drop cycle, and the claw holds a capsule if it closes within 12 mm of its centre
(a mechkit weld that carries it under the claw, released over the chute). Nothing else is moved by code.

Goal: steer the claw over a capsule with the joystick, press the button, and win the capsule into the chute, without
reaching into the cabinet. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=short | no_drop (see demo() below).
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
from scoring import (BUTTON, CAPSULES, CARRIAGE_X, CARRIAGE_Y, CLAW, MACHINE, STICK_FB, STICK_LR, Program,  # noqa: E402
                     Tracker, capsule_positions, controls)
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Play the claw machine: steer the claw over a prize with the joystick, press the button to drop it, "
               "and win a prize into the chute. You have one credit; don't reach into the machine.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _machine():
    d, w, b, t = L.CAB_D, L.CAB_W, L.BASE_H, L.WALL_T
    x0, x1, y0, y1 = L.CHUTE
    s = [*mk.plate_with_holes((d / 2, 0.0, b - t / 2), (d, w), [((x0 + x1) / 2 - d / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)],
                              t, L.FLOOR_COLOUR),
         mk.box((t / 2, 0.0, (b - t) / 2), (t, w, b - t), L.CAB_COLOUR),
         mk.box((d - t / 2, 0.0, (b - t) / 2), (t, w, b - t), L.CAB_COLOUR),
         mk.box((d / 2, w / 2 - t / 2, (b - t) / 2), (d - 2 * t, t, b - t), L.CAB_COLOUR),
         mk.box((d / 2, -w / 2 + t / 2, (b - t) / 2), (d - 2 * t, t, b - t), L.CAB_COLOUR),
         # the pit's low clear walls, the corner posts and the gantry housing
         mk.box((t / 2, 0.0, b + L.LIP_H / 2), (t, w, L.LIP_H), L.POST_COLOUR),
         mk.box((d - t / 2, 0.0, b + L.LIP_H / 2), (t, w, L.LIP_H), L.POST_COLOUR),
         mk.box((d / 2, w / 2 - t / 2, b + L.LIP_H / 2), (d - 2 * t, t, L.LIP_H), L.POST_COLOUR),
         mk.box((d / 2, -w / 2 + t / 2, b + L.LIP_H / 2), (d - 2 * t, t, L.LIP_H), L.POST_COLOUR),
         mk.box((d / 2, 0.0, L.TOP_Z + L.HOUSING_H / 2), (d, w, L.HOUSING_H), L.CAB_COLOUR),
         # the chute's mouth, marked
         mk.box(((x0 + x1) / 2, y1 + 0.002, b + 0.010), (x1 - x0, 0.004, 0.020), L.CAB_COLOUR),
         mk.box((x1 + 0.002, (y0 + y1) / 2, b + 0.010), (0.004, y1 - y0, 0.020), L.CAB_COLOUR),
         # the control deck
         mk.box((-L.DECK_D / 2, 0.0, L.DECK_H / 2), (L.DECK_D, w, L.DECK_H), L.DECK_COLOUR)]
    for px in (L.POST / 2, d - L.POST / 2):
        for py in (w / 2 - L.POST / 2, -w / 2 + L.POST / 2):
            s.append(mk.cylinder((px, py, (b + L.TOP_Z) / 2), L.POST / 2, L.TOP_Z - b, "Z", L.POST_COLOUR))
    sx, sy = float(L.STICK[0]), float(L.STICK[1])
    pivot = (sx, sy, L.DECK_H)
    bat_z = L.DECK_H + L.SHAFT_L + L.BAT_L / 2
    joints = [
        mk.hinge(STICK_FB, "Y", pivot, [mk.box((sx, sy, L.DECK_H + 0.004), (0.020, 0.020, 0.008), L.DECK_COLOUR)],
                 upper=L.STICK_LIMIT, lower=-L.STICK_LIMIT, mass=0.01, spring=L.STICK_SPRING, damping=0.02),
        mk.hinge(STICK_LR, "X", pivot, [mk.cylinder((sx, sy, L.DECK_H + L.SHAFT_L / 2), 0.004, L.SHAFT_L, "Z", L.DARK),
                                        mk.cylinder((sx, sy, bat_z), L.BAT_R, L.BAT_L, "Z", L.BAT_COLOUR)],
                 upper=L.STICK_LIMIT, lower=-L.STICK_LIMIT, mass=0.02, spring=L.STICK_SPRING, damping=0.02,
                 parent=STICK_FB),
        mk.slider(BUTTON, "-Z", [mk.cylinder((float(L.BUTTON[0]), float(L.BUTTON[1]), L.DECK_H + 0.005), L.BUTTON_R,
                                             0.010, "Z", L.BUTTON_COLOUR)],
                  upper=L.BUTTON_TRAVEL, mass=0.01, spring=L.BUTTON_SPRING, damping=2.0),
        mk.slider(CARRIAGE_X, "X", [mk.box((float(L.HOME[0]), 0.0, L.TOP_Z - 0.008), (0.016, w - 2 * L.POST - 0.004, 0.010),
                                           L.POST_COLOUR)],
                  upper=L.CLAW_X[1] - L.HOME[0], lower=L.CLAW_X[0] - L.HOME[0], mass=0.05, spring=3000.0, damping=40.0),
        mk.slider(CARRIAGE_Y, "Y", [mk.box((float(L.HOME[0]), float(L.HOME[1]), L.TOP_Z - 0.025), (0.030, 0.030, 0.020),
                                           L.DECK_COLOUR)],
                  upper=L.CLAW_Y[1] - L.HOME[1], lower=L.CLAW_Y[0] - L.HOME[1], mass=0.03, spring=3000.0, damping=40.0,
                  parent=CARRIAGE_X),
        mk.slider(CLAW, "-Z", _claw_shapes(), upper=L.HUB_Z - L.DROP_TO, mass=0.03, spring=3000.0, damping=40.0,
                  parent=CARRIAGE_Y),
    ]
    return mk.assembly(MACHINE, (float(L.ORIGIN[0]), float(L.ORIGIN[1]), L.TOP), s, joints=joints, color=L.CAB_COLOUR)


def _claw_shapes():
    hx, hy, hz = float(L.HOME[0]), float(L.HOME[1]), L.HUB_Z
    s = [mk.cylinder((hx, hy, hz + 0.010), 0.015, 0.020, "Z", L.CLAW_COLOUR),
         mk.cylinder((hx, hy, hz + 0.022), 0.002, 0.004, "Z", L.DARK)]
    for k in range(3):  # three prongs round the hub
        a = 2 * math.pi * k / 3
        s.append(mk.box((hx + 0.017 * math.cos(a), hy + 0.017 * math.sin(a), hz + 0.008), (0.004, 0.004, 0.016),
                        L.CLAW_COLOUR))
    return s


MACHINE_SPEC = _machine()
CAPSULE_SPECS = [Box(n, pos=_f(L.world([c[0], c[1], L.BASE_H + L.PRIZE / 2 + 0.002])), color=col, mass=L.M_CAPSULE,
                     size=(L.PRIZE, L.PRIZE, L.PRIZE))
                  for n, c, col in zip(CAPSULES, L.CAPSULES, L.CAPSULE_COLOURS)]
OBJECTS = [MACHINE_SPEC, *CAPSULE_SPECS]
WELDS = [mk.weld(n, None, name=f"claw_holds_{n}") for n in CAPSULES]  # a held capsule's pose is set under the claw


def held_pose() -> np.ndarray:
    """Where a held capsule's centre is (world): just under the claw's hub."""
    return L.world([_PROG.pos[0], _PROG.pos[1], _PROG.z - L.CAPSULE_R - 0.004])


# -- the machine's controller (the one rule) ------------------------------------------------------------------------
_PROG = Program()
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


def _check_episode() -> None:
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _PROG.reset()
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()


@mk.on_step
def _controller(states) -> None:
    _check_episode()
    c, caps = controls(states), capsule_positions(states)
    if c is None or caps is None:
        return
    for ev in _PROG.step(*c, [p[:2] for p in caps]):
        if ev.startswith("grab"):
            WELDS[int(ev.split()[1])].fire(states, rel_pose=(_f(held_pose()), (0.0, 0.0, 0.0, 1.0)))
        elif ev == "release":
            for w in WELDS:
                w.release()
    for w in WELDS:
        if w.active:  # it rides with the claw
            w.rel = (held_pose(), np.eye(3))
    for joint, v in _PROG.targets().items():
        mk.set_targets(MACHINE_SPEC, joint, v)


# -- scoring plumbing -----------------------------------------------------------------------------------------------
@rigstate.on_reset
def _new_episode() -> None:
    _PROG.reset()
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
    """A capsule won through the chute, no hand in the cabinet; held 1 s. Runs every control step, so it also feeds
    the tracker."""
    _check_episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms, _PROG)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    c = controls(objects)
    if c is not None:
        row.update(fb_deg=round(math.degrees(c[0]), 1), lr_deg=round(math.degrees(c[1]), 1), button_mm=round(c[2] * 1000, 2))
    caps = capsule_positions(objects)
    if caps is not None:
        row["capsules_mm"] = [[round(v * 1000, 1) for v in p] for p in caps]
    if arms is not None:
        try:
            row.update(left_tcp=_r(arms["left"].tcp), right_tcp=_r(arms["right"].tcp))
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 a capsule won; 0.5 the claw closed within 25 mm of one; 0.25 a real attempt; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, nothing read from the simulator. The right hand cages the joystick's bat top between its open pads (40 mm
# apart, the bat 24 mm across) and pushes it with one pad, so the stick tilts freely: pads before and behind it, the
# hand moves forward and the rear pad tilts the stick forward (the claw travels away from the robot) for PUSH_X
# seconds, then back; the hand turned a quarter turn, pads either side, it pushes the stick right for PUSH_Y seconds and
# back. The durations put the claw over the red prize (worked out from the claw's 50 mm/s and tuned on the simulator).
# Then the left hand presses the drop button; the machine does the rest.
OPEN = -0.04695
CLOSED = 0.0
CAGE = -0.020  # pads 40 mm apart round the 24 mm bat
TILT = math.radians(16.0)  # past the 8 deg switch
PUSH = 0.020 - L.BAT_R + (L.SHAFT_L + L.BAT_L / 2) * math.sin(TILT)  # the gap to the pad, then the bat's travel
PUSH_X = 1.55  # s held forward (the switch is also closed for about 0.75 s of the push in and out)
PUSH_Y = 2.10  # s held right (and about 0.4 s of the push in and out)
PUSH_SIDE = PUSH + 0.012  # the arm gives more sideways: pushed further to tilt the stick past its switch
GRIP_ABOVE = 0.008
VARIANTS = ("short", "no_drop")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


def bat_centre() -> np.ndarray:
    return L.world([L.STICK[0], L.STICK[1], L.DECK_H + L.SHAFT_L + L.BAT_L / 2])


def bat_tcp(shift=(0.0, 0.0)) -> np.ndarray:
    return bat_centre() + np.array([shift[0], shift[1], GRIP_ABOVE])


def right_plan(push_x: float = PUSH_X, push_y: float = PUSH_Y) -> list[Waypoint]:
    up = np.array([0.0, 0.0, 0.05])
    q = math.pi / 2  # pads closing along x: one before and one behind the bat
    return [
        _w(2.5, bat_tcp() + up, OPEN, "over the joystick's bat top", q),
        _w(1.0, bat_tcp(), CAGE, "down: the pads before and behind it, not touching", q),
        _w(0.6, bat_tcp((PUSH, 0.0)), CAGE, "push it forward with the rear pad: the claw travels away", q),
        _w(push_x, bat_tcp((PUSH, 0.0)), CAGE, "hold it forward", q),
        _w(0.6, bat_tcp(), CAGE, "back: it springs to centre, the claw stops", q),
        _w(0.5, bat_tcp() + [0, 0, 0.03], CAGE, "up", q),
        _w(0.8, bat_tcp() + [0, 0, 0.03], CAGE, "turn the hand a quarter turn"),
        _w(0.6, bat_tcp(), CAGE, "down: the pads either side of it"),
        _w(0.6, bat_tcp((0.0, -PUSH_SIDE)), CAGE, "push it right with the left pad: the claw travels right"),
        _w(push_y, bat_tcp((0.0, -PUSH_SIDE)), CAGE, "hold it right"),
        _w(0.6, bat_tcp(), CAGE, "back to centre"),
        _w(1.0, bat_tcp() + up, OPEN, "up and away"),
        _w(26.0, bat_tcp() + up + [-0.05, -0.05, 0.0], OPEN, "wait while the machine plays"),
    ]


def button_tcp(down: float = 0.0) -> np.ndarray:
    return L.world([L.BUTTON[0], L.BUTTON[1], L.DECK_H + 0.010 + 0.013 - down])


def left_plan(press: bool = True) -> list[Waypoint]:
    t_ready = sum(w.duration for w in right_plan()[:11]) + 0.3
    w = [_w(3.0, button_tcp(-0.04), CLOSED, "over the drop button, fingers closed"),
         _w(t_ready - 3.0, button_tcp(-0.01), CLOSED, "just above it, waiting for the claw to stop")]
    if press:
        w += [_w(0.4, button_tcp(L.BUTTON_TRAVEL + 0.002), CLOSED, "press: the claw drops"),
              _w(0.4, button_tcp(L.BUTTON_TRAVEL + 0.002), CLOSED, "hold"),
              _w(0.4, button_tcp(-0.02), CLOSED, "let it up")]
    w.append(_w(26.0, button_tcp(-0.06), CLOSED, "wait while the machine plays"))
    return w


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "short":  # stops about 40 mm short of the red prize along x: the claw closes on nothing
        return {"right": right_plan(push_x=PUSH_X - 1.2), "left": left_plan()}
    if variant == "no_drop":  # steers, never presses: the timer runs out and the claw drops by itself
        return {"right": right_plan(), "left": left_plan(press=False)}
    return {"right": right_plan(), "left": left_plan()}


def demo():
    """Right hand on the joystick, left on the button. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short    held forward 1.2 s less: the claw closes about 40 mm short of the prize
      no_drop  never presses the button: the 30 s timer drops the claw where it stopped"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
