"""On the bench, real size: a Victorinox Fibrox 8-inch chef's knife (33.5 cm overall, 20.0 cm blade, 180 g) and a
Mercer Culinary M33183P slide-on polypropylene blade guard (20.5 cm long, 45 g) lie flat on a wooden cutting board
(19 x 42 x 1.5 cm), both fully inside the board's edges. Directly behind the board sits an open kitchen utensil
drawer (18 x 38 x 5 cm interior).

Objects (mechkit.py next to this file):
- cutting_board (fixed)
- utensil_drawer (fixed open box)
- knife (loose compound rigid body: handle + tapered blade)
- guard (loose U-channel sleeve; a weld latch engages once slid from the blade tip to within 1.0 cm of the heel)

Goal: put the chef's knife away in the open kitchen utensil drawer. Safely storing a chef's knife in a utensil drawer
requires sliding the blade guard fully onto the blade first, never placing an unsheathed blade into the drawer and
never letting a gripper contact the exposed cutting edge. See scoring.py for tiers and hard-0 rules.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=bare_to_drawer | guard_only (counterfactuals, never the submitted demo).
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
from offline_sim import OfflineKnifeSim  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import BOARD, DRAWER, GUARD, KNIFE, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the chef's knife away in the open kitchen utensil drawer."


# -- objects ------------------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _cutting_board():
    bx, by, bh = L.BOARD_SIZE
    shapes = [mk.box((0.0, 0.0, bh / 2), (bx, by, bh), L.BOARD_COLOR)]
    return mk.assembly(BOARD, (float(L.BOARD_CENTER[0]), float(L.BOARD_CENTER[1]), L.TOP),
                       shapes, mass=None, color=L.BOARD_COLOR, friction=(0.6, 0.5))


def _utensil_drawer():
    ox, oy, oh = L.DRAWER_OUTER
    shapes = list(mk.open_box((0.0, 0.0, oh / 2), (ox, oy, oh), wall=L.DRAWER_WALL, floor=L.DRAWER_FLOOR,
                              color=L.DRAWER_COLOR))
    # Subtle contrasting trim strip on the top rim of the front wall
    shapes.append(mk.box((-ox / 2 + L.DRAWER_WALL / 2, 0.0, oh - 0.003),
                         (L.DRAWER_WALL + 0.0005, oy, 0.006), L.DRAWER_FRONT_COLOR))
    return mk.assembly(DRAWER, (float(L.DRAWER_CENTER[0]), float(L.DRAWER_CENTER[1]), L.TOP),
                       shapes, mass=None, color=L.DRAWER_COLOR, friction=(0.6, 0.5))


def _knife():
    hx0, hx1 = L.HANDLE_X
    hy0, hy1 = L.HANDLE_Y
    shapes = [
        # Fibrox TPE handle
        mk.box(((hx0 + hx1) / 2, (hy0 + hy1) / 2, L.HANDLE_H / 2),
               (hx1 - hx0, hy1 - hy0, L.HANDLE_H), L.HANDLE_COLOR),
    ]
    # Tapered 20.0 cm stainless-steel chef's blade (drop-point profile centered along BLADE_CX = -8.5 mm)
    shapes += _blade_shapes()
    return mk.assembly(KNIFE, (float(L.KNIFE_START[0]), float(L.KNIFE_START[1]), L.BOARD_TOP),
                       shapes, mass=L.M_KNIFE, color=L.HANDLE_COLOR,
                       friction=L.KNIFE_FRICTION, contact_offset=0.002,
                       linear_damping=L.KNIFE_LIN_DAMPING, angular_damping=L.KNIFE_ANG_DAMPING)


def _blade_shapes() -> list:
    """Stepped blade segments joined by short lead-in tapers (L.BLADE_TAPER_LEN) at every width step, on both
    sides, so a sleeve sliding on from the tip rides over a gradual taper like a real blade instead of meeting a
    square shoulder (GPU job 20261122T041814: the guard mouth stalled exactly at the 34 -> 42 mm and 42 -> 45 mm
    steps). Every taper lies inside the stepped outline that scoring.py checks for the cutting edge."""
    segs = sorted(L.BLADE_SEGMENTS)  # tip (-y) to heel
    r, tc, cx = L.BLADE_TAPER_LEN, L.BLADE_TAPER_T, L.BLADE_CX
    out = []
    for i, (y_lo, y_hi, w) in enumerate(segs):
        lo = y_lo + (r if i > 0 else 0.0)  # the narrower tip-side core runs r into this segment
        hi = y_hi + (r if i < len(segs) - 1 else 0.0)
        out.append(mk.box((cx, (lo + hi) / 2, L.BLADE_Z), (w, hi - lo, L.BLADE_T), L.BLADE_COLOR))
    for (_, y_s, w_n), (_, _, w_w) in zip(segs[:-1], segs[1:]):
        dlt = (w_w - w_n) / 2
        ln = math.hypot(dlt, r)
        for s in (-1.0, 1.0):  # -1: cutting-edge side, +1: spine side
            p0 = np.array([cx + s * w_n / 2, y_s])
            p1 = np.array([cx + s * w_w / 2, y_s + r])
            d = (p1 - p0) / ln
            outward = np.array([s * r, -dlt]) / ln
            c = (p0 + p1) / 2 - outward * tc / 2
            th = math.atan2(-d[0], d[1])
            out.append(mk.box((float(c[0]), float(c[1]), L.BLADE_Z), (tc, ln, L.BLADE_T), L.BLADE_COLOR,
                              quat=(0.0, 0.0, math.sin(th / 2), math.cos(th / 2))))
    return out


def _retention_leaf():
    """Sprung friction-fit leaf hinged just under the guard's top plate near the +y mouth, angled down toward the
    closed end so the entering blade cams it up; the spring then presses it onto the blade's top face."""
    py, pz, a = L.LEAF_PIVOT_Y, L.LEAF_PIVOT_Z, L.LEAF_TILT
    d = np.array([0.0, -math.cos(a), -math.sin(a)])  # pivot -> tip
    n = np.array([0.0, math.sin(a), -math.cos(a)])  # downward face normal
    center = np.array([0.0, py, pz]) + d * L.LEAF_LEN / 2 + n * L.LEAF_T / 2
    shapes = [
        mk.box(tuple(float(v) for v in center), (L.LEAF_W, L.LEAF_LEN, L.LEAF_T), L.GUARD_ACCENT_COLOR,
               quat=(math.sin(a / 2), 0.0, 0.0, math.cos(a / 2))),
    ]
    return mk.hinge("retention_leaf", "X", (0.0, py, pz), shapes, lower=L.LEAF_LOWER, upper=L.LEAF_UPPER,
                    mass=L.LEAF_MASS, spring=L.LEAF_SPRING, damping=L.LEAF_DAMPING)


def _guard():
    w, l, h, p, sd = L.GUARD_W, L.GUARD_L, L.GUARD_H, L.GUARD_PLATE, L.GUARD_SIDE
    shapes = [
        # Top and bottom polypropylene plates
        mk.box((0.0, 0.0, h / 2 - p / 2), (w, l, p), L.GUARD_COLOR),
        mk.box((0.0, 0.0, -h / 2 + p / 2), (w, l, p), L.GUARD_COLOR),
        # Left and right side walls
        mk.box((w / 2 - sd / 2, 0.0, 0.0), (sd, l, h - 2 * p), L.GUARD_COLOR),
        mk.box((-w / 2 + sd / 2, 0.0, 0.0), (sd, l, h - 2 * p), L.GUARD_COLOR),
        # Closed tip end (-y); open mouth at +y
        mk.box((0.0, -l / 2 + p / 2, 0.0), (w, p, h - 2 * p), L.GUARD_COLOR),
        # Subtle visual accent band near closed end
        mk.box((0.0, -l / 2 + 0.025, h / 2 + 0.0003), (w - 0.006, 0.018, 0.0006), L.GUARD_ACCENT_COLOR),
    ]
    return mk.assembly(GUARD, (float(L.GUARD_START[0]), float(L.GUARD_START[1]), L.BOARD_TOP + h / 2 + 0.0002),
                       shapes, joints=[_retention_leaf()], mass=L.M_GUARD, color=L.GUARD_COLOR,
                       friction=L.GUARD_FRICTION, contact_offset=0.002)


BOARD_SPEC = _cutting_board()
DRAWER_SPEC = _utensil_drawer()
KNIFE_SPEC = _knife()
GUARD_SPEC = _guard()
OBJECTS = [BOARD_SPEC, DRAWER_SPEC, KNIFE_SPEC, GUARD_SPEC]


# -- guard heel friction-fit catch --------------------------------------------------------------------------------------
IDENTITY = (0.0, 0.0, 0.0, 1.0)
_SLIDE = {"episode": None, "on_from_tip": False}


def _guard_home(states) -> bool:
    """Engages once the guard has slid onto the blade from the tip side and the robot has pushed it to within
    CATCH_TOL (1 mm) of the heel: the catch locks it there and never pulls it home from further out."""
    if _SLIDE["episode"] != mk.episode():
        _SLIDE.update(episode=mk.episode(), on_from_tip=False)
    k, g = S.body(states[KNIFE]), S.body(states[GUARD])
    if k is None or g is None:
        return False
    covered, gy = S.guard_on(k, g)
    ready, _SLIDE["on_from_tip"] = S.catch_ready(covered, gy, _SLIDE["on_from_tip"])
    return ready


GUARD_CATCH = mk.weld(GUARD_SPEC, KNIFE_SPEC,
                      ((L.GUARD_SEAT_X, L.GUARD_SEAT_Y, L.GUARD_SEAT_Z), IDENTITY),
                      when=_guard_home, name="guard_seated")

# Once the catch has locked, the guard's pose is rewritten every step, so the sprung leaf's push on the blade has no
# reaction on the guard and only shoves the knife (GPU job 20261122T044420: knife crept ~6 mm in the drawer). The
# leaf is then held a little up off the blade at the angle it had when the catch locked (a drive target, like the
# latched plunger in tasks/_mechkit2-smoke); every target returns to 0 at reset.
_LEAF_HOLD = {"episode": None, "target": None}


def leaf_hold_target(q: float) -> float:
    """Drive target for the leaf once the guard is seated: LEAF_HOLD_LIFT rad up from ``q`` (negative is up, toward
    the top plate), clamped to the hinge limits."""
    if not math.isfinite(q):
        q = 0.0
    return float(min(L.LEAF_UPPER, max(L.LEAF_LOWER, q - L.LEAF_HOLD_LIFT)))


@mk.on_step
def _leaf_rule(states) -> None:
    if _LEAF_HOLD["episode"] != mk.episode():
        _LEAF_HOLD.update(episode=mk.episode(), target=None)
    if not GUARD_CATCH.active:
        return
    if _LEAF_HOLD["target"] is None:
        g = states.get(GUARD) if hasattr(states, "get") else None
        q = float(getattr(g, "joints", {}).get("retention_leaf", 0.0)) if g is not None else 0.0
        _LEAF_HOLD["target"] = leaf_hold_target(q)
    mk.set_targets(GUARD_SPEC, "retention_leaf", _LEAF_HOLD["target"])


# -- scoring plumbing ---------------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_OFFLINE = OfflineKnifeSim()  # only stepped when mk.env() is None (no simulator)
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """Guard seated on blade, sheathed knife flat in utensil drawer, grippers clear >= 4 cm, still <= 2 mm for 1 s."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    states = _OFFLINE.step(arms) if mk.env() is None else objects
    result = _TRACKER.update(states, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "guard_caught": GUARD_CATCH.active if mk.env() is not None else _OFFLINE.welded, **result.details}
    if arms is not None:
        try:
            row["tcp"] = {s: [round(float(v), 4) for v in arms[s].tcp] for s in ("left", "right")}
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 sheathed in drawer + clear + still; 0.75 sheathed in drawer; 0.50 seated on board; 0.25 >= 10 cm on blade."""
    return _TRACKER.last.score


# -- scripted demo ------------------------------------------------------------------------------------------------------
# Open loop, nothing read from the simulator.
# Straight-down grips with fingers closing along x: left arm yaw -pi/2, right arm yaw +pi/2.
#   1. Left hand grips the knife handle firmly on the cutting board.
#   2. Right hand picks up the blade guard near its closed end, aligns its open mouth just beyond the blade tip,
#      and slides it smoothly onto the blade until seated at the heel.
#   3. Both hands lift the sheathed knife together over the drawer rim, set it flat on the drawer floor,
#      open grippers, lift clear >= 4 cm, and hold still.
OPEN = -0.04695
HANDLE_PINCH = -0.003  # -0.003 m (11 mm per side on 28 mm Fibrox handle for rock-solid ~55 N hold)
GUARD_PINCH = -(L.GUARD_W / 2 - 0.007)  # -0.025 m (7 mm past guard sides for slip-free hold)
YAW_LEFT = -math.pi / 2
YAW_RIGHT = math.pi / 2

KNIFE_ORIGIN = np.array([float(L.KNIFE_START[0]), float(L.KNIFE_START[1]), L.BOARD_TOP])
HANDLE_GRIP = KNIFE_ORIGIN + np.array([0.0, 0.080, L.HANDLE_H / 2 + 0.005])  # (0.258, +0.1125, TOP + 0.031)

GUARD_GRIP_LOCAL_Y = -0.055  # 5.5 cm toward closed end from guard center (more margin from cutting edge at heel)
GUARD_START_CENTER = np.array([float(L.GUARD_START[0]), float(L.GUARD_START[1]), L.BOARD_TOP + L.GUARD_H / 2 + 0.0002])
GUARD_GRIP = np.array([float(L.GUARD_START[0]), float(L.GUARD_START[1]) + GUARD_GRIP_LOCAL_Y, L.BOARD_TOP + 0.015])
GUARD_TO_TCP = GUARD_GRIP - GUARD_START_CENTER

# GPU open-loop aim compensation: centers held guard on blade line during +y slide
AIM_DX = float(os.environ.get("KNIFE_AIM_DX", 0.0343))
# GPU run 20261122T024858: the held guard drifted +8 mm in x (relative to the knife) over the first 100 mm of the
# slide while the hand x stayed constant; counter-steer by ramping the aim x back over the slide.
SLIDE_DX_END = float(os.environ.get("KNIFE_SLIDE_DX_END", -0.008))
AIM_DZ = float(os.environ.get("KNIFE_AIM_DZ", 0.0012))

BLADE_X_WORLD = float(KNIFE_ORIGIN[0] + L.GUARD_SEAT_X)  # 0.2495
BLADE_Z_WORLD = float(KNIFE_ORIGIN[2] + L.GUARD_SEAT_Z)  # BOARD_TOP + 0.011
SEAT_Y_WORLD = float(KNIFE_ORIGIN[1] + L.GUARD_SEAT_Y)  # -0.070
SLIDE_START_Y = float(KNIFE_ORIGIN[1] + L.BLADE_Y[0] - 0.010 - L.GUARD_L / 2)  # open mouth 1 cm right of tip (-0.280)
# Press past the seat: the handle face stops the guard's mouth at the heel, so the hand target overshoots it and the
# arms' compliance holds the guard against the stop. GPU job 20261122T051555 (old 10 mm catch): at the end of the old
# 0.6 s, 4 mm press the right hand still lagged its target by ~31 mm, the push had dragged the knife and the left
# hand ~15 mm toward +y, and the guard sat ~39 mm short of the heel; the old catch only closed it during the lift. With
# a 1 mm catch the robot must reach the stop itself, so the press goes 12 mm past the seat (est.) and is held 2.0 s.
SEAT_LEAD = 0.003  # end of the slide: 3 mm past the seat
SEAT_PUSH = 0.012  # press target: 12 mm past the seat, against the handle-face stop (est.)

# Carry delta from cutting board to open utensil drawer
CARRY_DX = 0.125  # moves knife bolster x from 0.258 to 0.383 (well inside [0.320, 0.500] and comfortable arm reach)
CARRY_LIFT_Z = 0.070  # lifts handle bottom to TOP + 0.085 (2.7 cm over 5.8 cm drawer rim); TCP at TOP + 0.101/0.104
DRAWER_DROP_DZ = (L.DRAWER_FLOOR_Z + 0.004) - L.BOARD_TOP  # -0.003 m relative to board-level grip (gentle 4 mm drop)

UP_LOCAL = np.array([0.0, 0.0, 0.055])
PARK_LEFT = L.at(0.20, 0.24, 0.125)
PARK_RIGHT = L.at(0.20, -0.24, 0.125)
VARIANTS = ("bare_to_drawer", "guard_only")


def _w(duration: float, pos, finger: float, label: str, yaw: float) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


def guard_tcp(center_y_world: float) -> np.ndarray:
    """Right TCP holding the blade guard with its center on the blade line at center_y_world."""
    return np.array([BLADE_X_WORLD + AIM_DX, center_y_world, BLADE_Z_WORLD + AIM_DZ]) + GUARD_TO_TCP


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")

    if variant == "bare_to_drawer":
        # Counterfactual: put the bare knife straight into the utensil drawer (triggers hard-0 violation)
        carry_up = HANDLE_GRIP + np.array([0.0, 0.0, CARRY_LIFT_Z])
        over_drawer = carry_up + np.array([CARRY_DX, 0.0, 0.0])
        in_drawer = HANDLE_GRIP + np.array([CARRY_DX, 0.0, DRAWER_DROP_DZ])
        left = [
            _w(1.5, HANDLE_GRIP + UP_LOCAL, OPEN, "over knife handle", YAW_LEFT),
            _w(1.0, HANDLE_GRIP, OPEN, "down onto handle", YAW_LEFT),
            _w(0.6, HANDLE_GRIP, HANDLE_PINCH, "grip handle", YAW_LEFT),
            _w(1.0, carry_up, HANDLE_PINCH, "lift bare knife", YAW_LEFT),
            _w(1.5, over_drawer, HANDLE_PINCH, "carry bare knife over drawer", YAW_LEFT),
            _w(1.0, in_drawer, HANDLE_PINCH, "set bare knife in drawer", YAW_LEFT),
            _w(0.6, in_drawer, OPEN, "release", YAW_LEFT),
            _w(1.0, over_drawer, OPEN, "lift clear", YAW_LEFT),
        ]
        return {"left": left, "right": []}

    # Phase 1: Left hand grips knife handle; right hand picks guard, aligns at blade tip, and slides until seated
    g_start = guard_tcp(SLIDE_START_Y)
    g_seat = guard_tcp(SEAT_Y_WORLD + SEAT_LEAD) + np.array([SLIDE_DX_END, 0.0, 0.0])  # +3 mm y lead for arm lag
    g_press = guard_tcp(SEAT_Y_WORLD + SEAT_PUSH) + np.array([SLIDE_DX_END, 0.0, 0.0])  # against the heel stop
    g_mid = guard_tcp(0.5 * (SLIDE_START_Y + SEAT_Y_WORLD)) + np.array([0.5 * SLIDE_DX_END, 0.0, 0.0])

    right = [
        _w(1.5, GUARD_GRIP + UP_LOCAL, OPEN, "over blade guard", YAW_RIGHT),
        _w(1.0, GUARD_GRIP, OPEN, "down onto guard sides", YAW_RIGHT),
        _w(0.3, GUARD_GRIP, OPEN, "settle over guard before pinch", YAW_RIGHT),
        _w(0.7, GUARD_GRIP, GUARD_PINCH, "pinch blade guard", YAW_RIGHT),
        _w(0.9, GUARD_GRIP + np.array([0.0, 0.0, 0.040]), GUARD_PINCH, "lift guard", YAW_RIGHT),
        _w(1.8, g_start + np.array([0.0, 0.0, 0.025]), GUARD_PINCH, "carry guard to blade tip", YAW_RIGHT),
        _w(1.0, g_start, GUARD_PINCH, "lower guard onto blade line", YAW_RIGHT),
        _w(1.8, g_mid, GUARD_PINCH, "slide guard onto blade (first half)", YAW_RIGHT),
        _w(2.6, g_seat, GUARD_PINCH, "slide guard onto blade until seated", YAW_RIGHT),
        _w(1.2, g_press, GUARD_PINCH, "press guard home against the heel", YAW_RIGHT),
        _w(0.8, g_press, GUARD_PINCH, "hold guard pressed against the heel", YAW_RIGHT),
    ]
    t_seated = sum(w.duration for w in right)  # 13.6 s

    left = [
        _w(1.5, HANDLE_GRIP + UP_LOCAL, OPEN, "over knife handle", YAW_LEFT),
        _w(1.0, HANDLE_GRIP, OPEN, "down onto knife handle", YAW_LEFT),
        _w(0.3, HANDLE_GRIP, OPEN, "settle over handle before grip", YAW_LEFT),
        _w(0.7, HANDLE_GRIP, HANDLE_PINCH, "grip handle firmly", YAW_LEFT),
        _w(t_seated - 3.5, HANDLE_GRIP, HANDLE_PINCH, "hold knife steady during sheathing", YAW_LEFT),
    ]

    if variant == "guard_only":
        left += [
            _w(0.6, HANDLE_GRIP, OPEN, "release handle", YAW_LEFT),
            _w(1.0, HANDLE_GRIP + np.array([0.0, 0.0, 0.090]), OPEN, "lift clear", YAW_LEFT),
            _w(1.2, PARK_LEFT, OPEN, "park clear", YAW_LEFT),
        ]
        right += [
            _w(0.6, g_press, OPEN, "release guard", YAW_RIGHT),
            _w(1.0, g_press + np.array([0.0, 0.0, 0.090]), OPEN, "lift clear", YAW_RIGHT),
            _w(1.2, PARK_RIGHT, OPEN, "park clear", YAW_RIGHT),
        ]
        return {"left": left, "right": right}

    # Phase 2: Synchronized bimanual carry of the sheathed knife into the open utensil drawer
    l_up = HANDLE_GRIP + np.array([0.0, 0.0, CARRY_LIFT_Z])
    r_up = g_press + np.array([0.0, 0.0, CARRY_LIFT_Z])  # no back-off along y: never pull the guard off the heel
    l_over = l_up + np.array([CARRY_DX, 0.0, 0.0])
    r_over = r_up + np.array([CARRY_DX, 0.0, 0.0])
    l_down = HANDLE_GRIP + np.array([CARRY_DX, 0.0, DRAWER_DROP_DZ])
    r_down = g_press + np.array([CARRY_DX, 0.0, DRAWER_DROP_DZ])
    l_clear = np.array([l_down[0], l_down[1], L.TOP + 0.125])
    r_clear = np.array([r_down[0], r_down[1], L.TOP + 0.125])

    left += [
        _w(1.0, l_up, HANDLE_PINCH, "lift sheathed knife (left)", YAW_LEFT),
        _w(1.6, l_over, HANDLE_PINCH, "carry over utensil drawer (left)", YAW_LEFT),
        _w(1.2, l_down, HANDLE_PINCH, "set flat in drawer (left)", YAW_LEFT),
        _w(0.6, l_down, OPEN, "open gripper (left)", YAW_LEFT),
        _w(1.0, l_clear, OPEN, "retract clear above drawer (left)", YAW_LEFT),
        _w(1.2, PARK_LEFT, OPEN, "park clear (left)", YAW_LEFT),
        _w(1.0, PARK_LEFT, OPEN, "hold parked (left)", YAW_LEFT),
    ]
    right += [
        _w(1.0, r_up, GUARD_PINCH, "lift sheathed knife (right)", YAW_RIGHT),
        _w(1.6, r_over, GUARD_PINCH, "carry over utensil drawer (right)", YAW_RIGHT),
        _w(1.2, r_down, GUARD_PINCH, "set flat in drawer (right)", YAW_RIGHT),
        _w(0.6, r_down, OPEN, "open gripper (right)", YAW_RIGHT),
        _w(1.0, r_clear, OPEN, "retract clear above drawer (right)", YAW_RIGHT),
        _w(1.2, PARK_RIGHT, OPEN, "park clear (right)", YAW_RIGHT),
        _w(1.0, PARK_RIGHT, OPEN, "hold parked (right)", YAW_RIGHT),
    ]
    return {"left": left, "right": right}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check; that arm holds its ready pose).
    DEMO_VARIANT (counterfactuals, never the submitted demo):
      bare_to_drawer  places bare knife into drawer without guard (hard 0)
      guard_only      sheaths knife on cutting board without placing in drawer (0.50)
    """
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
