"""On the station table, real size: an EastPoint Sports Tabletop Arcade Pinball (483 x 330 mm) with its front
toward the robot, held to the table. Its playfield rises 6 deg away from the robot between the cabinet's side panels:
a plunger lane along the right wall with a curved guide at its top, inlane guides down each side, and two flippers at
the bottom either side of the drain. A spring plunger's knob sticks out of the front, and a flipper button sits on each
front corner of the cabinet's top. A 19 mm steel ball rests in the plunger lane against the plunger.

Objects (mechkit.py next to this file): the machine (a fixed mechanism: the plunger slides on a spring; each button
slides down on a spring; each flipper turns on a hinge) and the ball (a loose sphere). The machine has no electrics:
each button works its flipper through a lever, and one rule stands in for those levers: a flipper's drive target is the
angle its button's travel gives (rest with the button up, raised at 6 mm). Nothing else is moved or latched by code.

Goal: launch the ball with the plunger, then flip it back up the playfield into the top lanes with a flipper, without
a hand on the playfield. The robot has to pull the plunger back far enough (and not too far), and press the left
flipper button at the moment the ball rolls onto the left flipper. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=no_flip | early_flip (see demo() below).
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
from scoring import BALL, LEFT_BUTTON, LEFT_FLIPPER, MACHINE, PLUNGER, RIGHT_BUTTON, RIGHT_FLIPPER, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Sphere  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Play the pinball machine: launch the ball with the plunger, then flip it back up into the top lanes "
               "with a flipper.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _pf_rot(yaw: float = 0.0) -> np.ndarray:
    """A shape lying in the playfield's plane, turned ``yaw`` within it."""
    return mk.axis_angle_matrix((0, 1, 0), -L.SLOPE) @ mk.axis_angle_matrix((0, 0, 1), yaw)


def pf_box(u, v, h, size, colour, yaw: float = 0.0):
    """A box whose centre is at playfield (u, v), ``h`` above it, its sides along (u, v, normal) turned ``yaw``."""
    return mk.box(tuple(float(c) for c in L.pf_point(u, v, h)), tuple(float(c) for c in size), colour,
                  quat=tuple(float(c) for c in mk.matrix_to_quat(_pf_rot(yaw))))


def pf_wall(a, b, colour=L.WALL_COLOUR, t=0.006):
    """A wall standing on the playfield from (u, v) = a to b."""
    (u0, v0), (u1, v1) = a, b
    length = math.hypot(u1 - u0, v1 - v0)
    yaw = math.atan2(v1 - v0, u1 - u0)
    return pf_box((u0 + u1) / 2, (v0 + v1) / 2, L.WALL_H / 2, (length + t, t, L.WALL_H), colour, yaw)


def _arc(centre, r, a0, a1, n=4):
    out = []
    for k in range(n):
        t0, t1 = a0 + (a1 - a0) * k / n, a0 + (a1 - a0) * (k + 1) / n
        p0 = (centre[0] + r * math.cos(t0), centre[1] + r * math.sin(t0))
        p1 = (centre[0] + r * math.cos(t1), centre[1] + r * math.sin(t1))
        out.append(pf_wall(p0, p1))
    return out


def _flipper(name, side):
    """A flipper on a vertical hinge at its pivot (cabinet frame), drawn at rest."""
    s = 1.0 if side == "left" else -1.0
    pu, pv = L.FLIPPER_U, s * L.FLIPPER_V
    pivot = L.pf_point(pu, pv, 0.0)
    rest = math.atan2(-s * math.cos(-L.FLIPPER_REST), -math.sin(-L.FLIPPER_REST))  # tip toward the centre, down
    d = np.array([math.cos(rest), math.sin(rest), 0.0])
    z = pivot[2] + 0.005 + L.FLIPPER_H / 2
    shapes = [mk.box(tuple(float(c) for c in pivot[:2] + d[:2] * L.FLIPPER_L / 2) + (float(z),),
                     (L.FLIPPER_L, L.FLIPPER_W, L.FLIPPER_H),
                     L.FLIPPER_COLOUR, ypr=(rest, 0.0, 0.0)),
              mk.cylinder((float(pivot[0]), float(pivot[1]), float(z)), L.FLIPPER_W / 2 + 0.002, L.FLIPPER_H,
                          color=L.RUBBER)]
    swing = L.FLIPPER_UP - L.FLIPPER_REST
    return mk.hinge(name, "Z" if side == "left" else "-Z", (float(pivot[0]), float(pivot[1]), float(z)), shapes,
                    upper=swing,
                    mass=0.02, spring=L.FLIPPER_SPRING, damping=0.002, color=L.FLIPPER_COLOUR)


def _button(name, side):
    s = 1.0 if side == "left" else -1.0
    x, y = L.BUTTON_X, s * L.BUTTON_Y
    shape = mk.cylinder((x, y, L.DECK_H + 0.006), L.BUTTON_R, 0.012, color=(0.95, 0.75, 0.10))
    return mk.slider(name, "-Z", [shape], upper=L.BUTTON_TRAVEL, anchor=(x, y, L.DECK_H + 0.012), mass=0.01,
                     spring=L.BUTTON_SPRING, damping=1.0)


def _plunger():
    tip = L.pf_point(L.PLUNGER_REST_U, L.LANE_V, L.BALL_R)
    z = float(tip[2])
    rod_len = float(tip[0]) + 0.004
    out = L.PLUNGER_SHAFT_OUT
    shapes = [mk.cylinder(((rod_len - out) / 2, L.LANE_V, z), 0.004, rod_len + out, axis="X", color=(0.8, 0.8, 0.8)),
              mk.cylinder((-out - L.PLUNGER_KNOB_L / 2, L.LANE_V, z), L.PLUNGER_KNOB_R, L.PLUNGER_KNOB_L, axis="X",
                          color=(0.95, 0.10, 0.10))]
    return mk.slider(PLUNGER, "-X", shapes, upper=L.PLUNGER_TRAVEL, anchor=(0.0, L.LANE_V, z), mass=0.012,
                     spring=L.PLUNGER_SPRING, damping=0.05)


def plunger_knob_z() -> float:
    return float(L.pf_point(L.PLUNGER_REST_U, L.LANE_V, L.BALL_R)[2])


def _machine():
    W, Lc, t = L.CAB_W, L.CAB_L, L.WALL_T
    s = [
        mk.box((Lc / 2, 0.0, L.BASE_H / 2), (Lc, W, L.BASE_H), L.CAB_COLOUR),  # plinth
        mk.box((Lc / 2, W / 2 - t / 2, L.SIDE_H / 2), (Lc, t, L.SIDE_H), L.CAB_COLOUR),  # side panels
        mk.box((Lc / 2, -W / 2 + t / 2, L.SIDE_H / 2), (Lc, t, L.SIDE_H), L.CAB_COLOUR),
        mk.box((Lc - t / 2, 0.0, L.SIDE_H / 2), (t, W, L.SIDE_H), L.CAB_COLOUR),  # back panel
        mk.box((L.DECK_D / 2, 0.0, L.DECK_H / 2), (L.DECK_D, W, L.DECK_H), L.CAB_COLOUR),  # front deck
    ]
    # the playfield: a 6 mm board lying at 6 deg, and the wedge of the cabinet under it
    s.append(pf_box(L.PF_LEN / 2, 0.0, -0.003, (L.PF_LEN, L.PF_W, 0.006), L.FIELD))
    for k in range(4):
        u = L.PF_LEN * (k + 0.5) / 4
        top = L.pf_point(u, 0.0, -0.006)[2]
        s.append(mk.box((float(L.pf_point(u, 0, 0)[0]), 0.0, float((L.BASE_H + top) / 2)),
                        (L.PF_LEN / 4, W - 2 * t, float(max(0.002, top - L.BASE_H))), L.CAB_COLOUR))
    hw = L.PF_W / 2
    lane_in = L.LANE_V + L.LANE_W / 2 + 0.003
    s.append(pf_wall((L.LANE_WALL_U[0] + 0.02, lane_in), (L.LANE_WALL_U[1], lane_in)))  # plunger lane's inner wall
    r = 0.085
    s += _arc((L.PF_LEN - r, -hw + r), r, -math.pi / 2, 0.0)  # top right: turns the launched ball across the top
    s += _arc((L.PF_LEN - r, hw - r), r, 0.0, math.pi / 2)  # top left
    s.append(pf_wall((0.170, hw), (L.FLIPPER_U + 0.008, L.FLIPPER_V + 0.010)))  # left inlane guide
    s.append(pf_wall((0.170, lane_in), (L.FLIPPER_U + 0.008, -L.FLIPPER_V - 0.010)))  # right inlane guide
    for v in (-0.055, 0.055):  # top lanes' dividers
        s.append(pf_wall((L.TOP_U, v), (L.TOP_U + 0.04, v), colour=(0.95, 0.95, 0.95)))
    s.append(pf_box(L.TOP_U, 0.0, 0.0002, (0.003, L.PF_W - 0.06, 0.0004), (0.95, 0.95, 0.95)))  # the top lanes' line
    joints = [_plunger(), _button(LEFT_BUTTON, "left"), _button(RIGHT_BUTTON, "right"),
              _flipper(LEFT_FLIPPER, "left"), _flipper(RIGHT_FLIPPER, "right")]
    return mk.assembly(MACHINE, (L.CAB_FRONT_X, L.CAB_CY, L.TOP), s, joints=joints, color=L.CAB_COLOUR,
                       friction=(0.25, 0.2))


MACHINE_SPEC = _machine()
BALL_SPEC = Sphere(BALL, pos=tuple(float(c) for c in L.ball_start()), color=L.BALL_COLOUR, mass=L.M_BALL, radius=L.BALL_R)
OBJECTS = [MACHINE_SPEC, BALL_SPEC]


# -- the flippers' linkages (the one rule) ------------------------------------------------------------------------------
@mk.on_step
def _flipper_linkages(states) -> None:
    """The machine has no coils: each button pushes a lever that turns its flipper. The rule stands in for that lever:
    the flipper is driven toward the angle its button's travel gives (rest at the button up, fully raised at 6 mm),
    so it swings only as fast and as far as the button is pressed."""
    try:
        j = states[MACHINE].joints
    except Exception:  # noqa: BLE001
        return
    swing = L.FLIPPER_UP - L.FLIPPER_REST
    for button, flipper in ((LEFT_BUTTON, LEFT_FLIPPER), (RIGHT_BUTTON, RIGHT_FLIPPER)):
        frac = min(1.0, max(0.0, float(j[button]) / L.BUTTON_TRAVEL))
        mk.set_targets(MACHINE_SPEC, flipper, swing * frac)


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
    """A flipper shot that sent the ball into the top lanes, no hand on the playfield; held 1 s. Runs every control
    step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        j = objects[MACHINE].joints
        row["joints"] = {k: round(float(v), 4) for k, v in j.items()}
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
    """1.0 a flipper shot into the top lanes; 0.5 a flipper shot; 0.25 launched; 0 otherwise."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, the right arm only, nothing read from the simulator. A straight-down grip on the plunger's knob, the pads
# either side of it; draw it back PULL toward the robot (the arm gives a little against the spring), then open: the
# spring drives the rod into the ball and launches it up the lane to the top, and it rolls back down the right side,
# where the inlane guide sends it across the right flipper. The same hand, closed, moves over the right flipper button
# and presses it down at the moment the ball reaches the flipper (timed from the launch on the simulator), holds it
# 0.6 s, and lets it up: the flipper strikes the ball back up the playfield into the top lanes.
OPEN = -0.04695
KNOB_GRIP = -(L.PLUNGER_KNOB_R - 0.004)
PULL = 0.042  # m: how far the hand draws the plunger back (of its 45 mm; the arm gives a little against the spring)
LAUNCH_AT = 6.5  # s: when the hand starts to open on the plunger
PRESS_AT = float(os.environ.get("PINBALL_PRESS_AT", 9.20))  # s: when the button is 70% down (tuned on the simulator: the ball's arrival)
PRESS_S = float(os.environ.get("PINBALL_PRESS_S", 0.10))  # a quick, full press: the lever flips the flipper
VARIANTS = ("no_flip", "early_flip")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


def knob_tcp(pull: float = 0.0) -> np.ndarray:
    x = L.CAB_FRONT_X - L.PLUNGER_SHAFT_OUT - L.PLUNGER_KNOB_L / 2 - pull
    return np.array([x, L.CAB_CY + L.LANE_V, L.TOP + plunger_knob_z() + 0.008])


def button_tcp(side: str = "right", down: float = 0.0) -> np.ndarray:
    """The closed hand's TCP with its pad tips on a flipper button's top, pressed ``down``."""
    s = 1.0 if side == "left" else -1.0
    return np.array([L.CAB_FRONT_X + L.BUTTON_X, L.CAB_CY + s * L.BUTTON_Y, L.TOP + L.DECK_H + 0.012 + 0.013 - down])


def right_plan(pull: float = PULL, press_at: float | None = PRESS_AT) -> list[Waypoint]:
    up = np.array([0.0, 0.0, 0.05])
    w = [
        _w(2.5, knob_tcp() + up, OPEN, "over the plunger's knob"),
        _w(1.2, knob_tcp(), OPEN, "down: the pads either side of the knob"),
        _w(0.8, knob_tcp(), KNOB_GRIP, "close on the knob"),
        _w(1.2, knob_tcp(pull), KNOB_GRIP, "draw the plunger back"),
    ]
    w.append(_w(max(0.2, LAUNCH_AT - sum(x.duration for x in w)), knob_tcp(pull), KNOB_GRIP, "hold it drawn"))
    w += [
        _w(0.4, knob_tcp(pull), OPEN, "let go: the plunger launches the ball"),
        _w(0.6, knob_tcp(pull) + up, OPEN, "up and clear of the knob"),
        _w(1.0, button_tcp(down=-0.03), 0.0, "over the right flipper button, the hand closing"),
    ]
    t = sum(x.duration for x in w)
    if press_at is None:
        return w + [_w(8.0, button_tcp(down=-0.03), 0.0, "wait, never pressing")]
    start = press_at - 0.7 * PRESS_S  # the button is 70% of the way down (the flipper most of the way up) at press_at
    w += [
        _w(max(0.2, start - t), button_tcp(down=-0.002), 0.0, "just above the button, waiting for the ball"),
        _w(PRESS_S, button_tcp(down=L.BUTTON_TRAVEL + 0.002), 0.0, "press: the flipper strikes"),
        _w(0.6, button_tcp(down=L.BUTTON_TRAVEL + 0.002), 0.0, "hold the flipper up"),
        _w(0.3, button_tcp(down=-0.002), 0.0, "let the button up"),
        _w(1.0, button_tcp(down=-0.05), OPEN, "lift away"),
        _w(5.0, button_tcp(down=-0.05), OPEN, "hold"),
    ]
    return w


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "no_flip":  # launched, the flipper never pressed: the ball drains
        return {"right": right_plan(press_at=None)}
    if variant == "early_flip":  # the flipper pressed 0.6 s too early: up and down again before the ball arrives
        return {"right": right_plan(press_at=PRESS_AT - 0.9)}
    return {"right": right_plan()}


def demo():
    """The right arm launches and flips (the left stays parked). DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_flip     launched, never flipped: the ball drains (0.25)
      early_flip  the flipper pressed and let up 0.9 s early: the ball drains (0.25)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
