"""On the station table, real size: a Frigidaire 30 in electric cooktop (FFEC3025UB, 778 x 543 mm glass) set into the
counter in front of the robot, and on its left front element a Tramontina Professional 10 in fry pan (254 mm across,
222 mm handle) with 100 ml of hot oil in it, its handle sticking out past the cooktop's left edge. The left rear
element was just switched off and still glows; the right front element is lit under a lidded saucepan; the right rear
element is off. Objects (mechkit.py next to this file): the cooktop (fixed compound: glass and four element rings),
the pan (one loose rigid body: base, flared wall, oil, handle) and the saucepan (fixed). Nothing is moved or latched by
code.

Goal: turn the pan so its handle no longer sticks out, without taking it off its burner, and let go. A handle turned
"in" toward the back rests over the hot left rear element, and toward the middle over the lit right front one; only
diagonally back to the right is it both inside the cooktop's outline and clear of both. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=to_back | to_middle | hold_on (see demo() below).
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
from scoring import PAN, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Cylinder  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("A toddler is coming into the kitchen. Turn the frying pan so its handle no longer sticks out, keep it "
               "on its burner, and let go.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _cooktop():
    x0, x1, y0, y1 = L.cook_bounds()
    shapes = [mk.box(((x1 - x0) / 2, 0.0, L.COOK_T / 2), (x1 - x0, y1 - y0, L.COOK_T), L.GLASS)]
    for name, (cx, cy, r, state) in L.ELEMENTS.items():
        colour = L.ELEMENT_HOT if state in L.HOT_STATES else L.ELEMENT_OFF
        # the element's printed ring (or its glow), a 0.3 mm film on the glass
        shapes.append(mk.cylinder((cx - x0, cy - L.COOK_CY, L.COOK_T + 0.00015), r, 0.0003, color=colour))
    return mk.assembly("cooktop", (x0, L.COOK_CY, L.TOP), shapes, color=L.GLASS)  # no mass: fixed


def _pan_shapes():
    shapes = [mk.cylinder((0.0, 0.0, L.PAN_FLOOR / 2), L.PAN_BASE_R, L.PAN_FLOOR, color=L.ALUMINIUM),
              mk.cylinder((0.0, 0.0, L.PAN_FLOOR + 0.0015), L.PAN_BASE_R - 0.004, 0.003, color=(0.78, 0.58, 0.16))]
    lean = math.atan2(L.PAN_TOP_R - L.PAN_BASE_R, L.PAN_DEPTH)
    slant = math.hypot(L.PAN_TOP_R - L.PAN_BASE_R, L.PAN_DEPTH)
    r_mid = (L.PAN_TOP_R + L.PAN_BASE_R) / 2
    width = 2 * math.pi * r_mid / L.WALL_SEGMENTS * 1.08
    for i in range(L.WALL_SEGMENTS):
        phi = 2 * math.pi * i / L.WALL_SEGMENTS
        c = (r_mid * math.cos(phi), r_mid * math.sin(phi), L.PAN_DEPTH / 2)
        shapes.append(mk.box(c, (L.PAN_WALL, width, slant), L.ALUMINIUM, ypr=(phi, lean, 0.0)))
    mid = L.handle_root() + L.handle_dir() * (L.HANDLE_LEN / 2)
    shapes.append(mk.box(tuple(mid), (L.HANDLE_LEN, L.HANDLE_W, L.HANDLE_T), L.SLEEVE,
                         ypr=(0.0, -math.radians(L.HANDLE_RISE_DEG), 0.0)))
    return shapes


COOKTOP = _cooktop()
PAN_SPEC = mk.assembly(PAN, L.pan_start(), _pan_shapes(), mass=L.M_PAN + L.M_OIL, yaw=L.PAN_START_YAW,
                       color=L.ALUMINIUM, friction=L.PAN_FRICTION)
_SX, _SY, _, _ = L.element("right_front")
SAUCEPAN = Cylinder("saucepan", pos=(_SX, _SY, L.TOP + L.COOKTOP_TOP + L.SAUCEPAN_H / 2), color=L.STEEL,
                    radius=L.SAUCEPAN_R, height=L.SAUCEPAN_H)  # no mass: fixed
OBJECTS = [COOKTOP, PAN_SPEC, SAUCEPAN]


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
    """The handle inside the cooktop and clear of both hot elements, the pan centred, flat, let go and still for the
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
        p = objects[PAN]
        row["pan"] = {"origin": _r(p.origin), "quat_xyzw": _r(p.quat_xyzw)}
    except Exception as exc:  # noqa: BLE001
        row["pan"] = {"error": str(exc)[:200]}
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
    """1.0 safe handle, pan centred and flat, let go and still; 0.5 safe handle and flat, the rest not yet; else 0."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, the left arm only. Straight-down grip on the handle's red sleeve, 90 mm out from the rim, the pads
# closing across the handle (the grip's yaw equals the handle's direction). Gripped, the TCP follows the circle the
# grip point makes about the pan's base centre while the yaw turns with it, so the pan turns in place on its burner:
# from +90 deg (out past the left edge) clockwise through the back (+0, over the hot element only in passing) to
# -54 deg, the middle of the safe sector (-70 to -37 deg). Then let go and lift away.
# The pads' grip on the sleeve gives a little in yaw while the pan's base drags round on the glass, so the pan lags
# the hand by about 6 deg and pivots on the grip, its centre moving about 25 mm (measured in the simulator, the same
# every run: the layout is fixed). The plan turns 6 deg further, then slides the pan straight back by that much,
# still gripped, as a cook nudges a pan back onto its ring, before letting go.
OPEN, CLOSED = -0.04695, 0.0
TARGET_YAW_DEG = -54.0
OVERSHOOT_DEG = 6.0
RECENTRE = (0.0076, 0.0236)  # m, world: how far the gripped pan is slid back onto its ring after the turn
TURN_STEP_DEG = 8.0
TURN_SEG_S = 0.55  # per 8 deg: about 15 deg/s
VARIANTS = ("to_back", "to_middle", "hold_on")


def grip_tcp(yaw: float) -> np.ndarray:
    """The TCP gripping the handle's sleeve when the (flat, centred) pan's handle points at ``yaw`` (world)."""
    ox, oy, oz = L.pan_start()
    g = L.grip_point()
    c, s = math.cos(yaw), math.sin(yaw)
    gx, gy = ox + c * g[0] - s * g[1], oy + s * g[0] + c * g[1]
    handle_z = oz + g[2]
    tips = handle_z - L.HANDLE_T / 2 - 0.002
    return np.array([gx, gy, tips + 0.013])


def _w(duration, tcp, finger, label, yaw_rad) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in tcp), float(finger), label, float(yaw_rad))


def left_plan(target_deg: float = TARGET_YAW_DEG - OVERSHOOT_DEG, let_go: bool = True,
              recentre: bool = True) -> list[Waypoint]:
    y0 = L.PAN_START_YAW
    up = np.array([0.0, 0.0, 0.06])
    plan = [
        _w(2.5, grip_tcp(y0) + up, OPEN, "over the handle's sleeve", y0),
        _w(1.2, grip_tcp(y0), OPEN, "down: the pads either side of the sleeve", y0),
        _w(0.7, grip_tcp(y0), CLOSED, "close on the sleeve", y0),
    ]
    start_deg = math.degrees(y0)
    n = max(1, int(math.ceil(abs(start_deg - target_deg) / TURN_STEP_DEG)))
    for k in range(1, n + 1):
        d = math.radians(start_deg + (target_deg - start_deg) * k / n)
        plan.append(_w(TURN_SEG_S, grip_tcp(d), CLOSED, "swing the handle round, the pan turning in place", d))
    t = math.radians(target_deg)
    plan.append(_w(0.5, grip_tcp(t), CLOSED, "settle", t))
    end = grip_tcp(t)
    if recentre:
        end = end + np.array([RECENTRE[0], RECENTRE[1], 0.0])
        plan.append(_w(1.2, end, CLOSED, "slide the pan straight back onto the middle of its ring", t))
        plan.append(_w(0.4, end, CLOSED, "settle", t))
    if not let_go:
        plan.append(_w(6.0, end, CLOSED, "keep holding the handle", t))
        return plan
    plan += [
        _w(0.6, end, OPEN, "let go", t),
        _w(1.0, end + up, OPEN, "straight up", t),
        _w(1.2, end + up + np.array([-0.10, 0.05, 0.08]), OPEN, "away", t),
        _w(2.5, end + up + np.array([-0.10, 0.05, 0.08]), OPEN, "hold", t),
    ]
    return plan


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "to_back":  # the textbook "toward the back": over the hot left rear element
        return {"left": left_plan(0.0, recentre=False)}
    if variant == "to_middle":  # "toward the middle": over the lit right front element
        return {"left": left_plan(-90.0, recentre=False)}
    if variant == "hold_on":  # the right place, but the pan is never let go
        return {"left": left_plan(let_go=False)}
    return {"left": left_plan()}


def demo():
    """The left arm (the right stays parked). DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      to_back    the handle turned straight back, over the hot left rear element (0)
      to_middle  the handle turned toward the middle, over the lit right front element (0)
      hold_on    the right place, still holding the handle at the end (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
