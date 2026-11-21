"""On the bench, real size: a STIHL GTA 26 cordless garden pruner lying on its left side, its guide bar flat and bare
(pointing to the robot's right), its AS 2 battery still in the end of its handle (pointing left); its scabbard lies on
the bench to the right of the bar, open end toward it.

Objects (mechkit.py next to this file): the pruner (a loose body), the battery (a loose body with its two locking
hooks' buttons on sprung slides) and the scabbard (a loose sleeve). Two latches are rules, both welds (mechkit.Weld:
the part's pose is rewritten on the pruner every control step while latched): the battery is held in its handle
while both locking hooks are pressed in at least 1.8 mm (let go within 8 mm of home, they catch it again; pushed
back home, it clicks in); the scabbard is held on the bar once it is
pushed home within 4 mm of the bar's root (its catch). Nothing else is moved by code.

Goal: get the pruner ready to go in the van: the battery out (and put down), then the scabbard on over the bare chain,
as the maker's manual says for transport. Neither is in the instruction. See scoring.py for the tiers.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=guard_first | battery_only (see demo() below).
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
import scoring as S  # noqa: E402
from scoring import BATTERY, GUARD, PRUNER, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "We're packing up. Get the pruning saw ready to go in the van."

HOOKS = ("hook_left", "hook_right")  # the buttons on the battery's +x and -x faces


# -- the objects ----------------------------------------------------------------------------------------------------
def _f(v) -> tuple:
    return tuple(float(c) for c in v)


def _span(lo, hi):
    return (lo + hi) / 2, hi - lo


def _pruner():
    (x0, x1), (y0, y1), (z0, z1) = L.HOUSING
    r = 0.015  # the housing's top edges are rounded
    shapes = [mk.box((0.0, _span(y0, y1)[0], (z1 - r) / 2), (x1 - x0, y1 - y0, z1 - r), L.BODY),
              mk.box((0.0, _span(y0, y1)[0], z1 - r / 2), (x1 - x0 - 2 * r, y1 - y0, r), L.BODY)]
    shapes += [mk.cylinder((sx * (x1 - r), _span(y0, y1)[0], z1 - r), r, y1 - y0, "Y", L.BODY) for sx in (-1, 1)]
    shapes.append(mk.box((0.0, y0 + 0.004, (z0 + z1) / 2), (x1 - x0 - 0.010, 0.008, z1 - z0 - 0.010), L.DARK))
    # the control handle: a tube round the battery with a stop at its housing end, angled down (toward -x) as a
    # pistol grip; built straight in the handle frame and turned into place
    ix, iz0, iz1 = L.BAT_X / 2 + L.HANDLE_CLEAR, L.BAT_HOME_H[2] - L.BAT_Z / 2 - L.HANDLE_CLEAR, \
        L.BAT_HOME_H[2] + L.BAT_Z / 2 + L.HANDLE_CLEAR
    hy, hl, w = _span(*L.HANDLE_Y)[0], L.HANDLE_Y[1] - L.HANDLE_Y[0], L.HANDLE_WALL
    straight = [((sx * (ix + w / 2), hy, (iz0 - w + iz1 + w) / 2), (w, hl, iz1 - iz0 + 2 * w), L.DARK) for sx in (-1, 1)]
    straight += [((0.0, hy, iz0 - w / 2), (2 * (ix + w), hl, w), L.DARK),
                 ((0.0, hy, iz1 + w / 2), (2 * (ix + w), hl, w), L.DARK),
                 ((0.0, (L.HANDLE_Y[0] + L.BATTERY_STOP_Y) / 2, (iz0 + iz1) / 2),
                  (2 * ix, L.BATTERY_STOP_Y - L.HANDLE_Y[0], iz1 - iz0), L.DARK)]
    (tx0, tx1), (ty0, ty1), (tz0, tz1) = L.TRIGGER
    straight.append(((_span(tx0, tx1)[0], _span(ty0, ty1)[0], _span(tz0, tz1)[0]), (tx1 - tx0, ty1 - ty0, tz1 - tz0),
                     L.BODY))
    shapes += [mk.box(_f(L.handle_to_pruner(c)), size, col, ypr=(L.HANDLE_YAW, 0.0, 0.0)) for c, size, col in straight]
    shapes += bar_and_chain()
    return mk.assembly(PRUNER, (L.PRUNER_AT[0], L.PRUNER_AT[1], L.TOP), shapes, mass=L.M_PRUNER, color=L.BODY,
                       joints=[hood()], friction=L.PRUNER_FRICTION, contact_offset=0.002)


def bar_and_chain():
    """The guide bar (a 1.1 mm steel plate with a rounded nose) and its 1/4" chain: a dark band of drive links round
    its edge and a cutter every 6.35 mm, alternately above and below the plate, reaching BAR_X from the bar's line."""
    y_tip, y_root = L.BAR_Y
    nose_y = L.NOSE_Y
    out = [mk.box((0.0, (nose_y + y_root) / 2, L.BAR_Z), (2 * L.PLATE_X, y_root - nose_y, L.PLATE_T), L.CHAIN),
           mk.cylinder((0.0, nose_y, L.BAR_Z), L.PLATE_X, L.PLATE_T, "Z", L.CHAIN)]
    band_x, band_w = L.PLATE_X + 0.0005, 0.003
    for sx in (-1, 1):
        out.append(mk.box((sx * band_x, (nose_y + y_root) / 2, L.BAR_Z), (band_w, y_root - nose_y, 0.0035), DRIVE))
    tw, ta, th = L.TOOTH
    tx = L.BAR_X - ta / 2
    k = 0
    for sx in (-1, 1):
        y = y_root - 0.004
        while y > nose_y:
            dz = (L.BAR_T / 2 - th / 2) * (1 if k % 2 else -1)
            out.append(mk.box((sx * tx, y, L.BAR_Z + dz), (ta, tw, th), CUTTER))
            y -= L.PITCH
            k += 1
    for i in range(7):  # round the nose
        a = math.pi * (i + 0.5) / 7
        rr = L.BAR_X - ta / 2
        dz = (L.BAR_T / 2 - th / 2) * (1 if i % 2 else -1)
        out.append(mk.box((rr * math.cos(a), nose_y - rr * math.sin(a), L.BAR_Z + dz), (ta, tw, th), CUTTER,
                          ypr=(a, 0.0, 0.0)))
    return out


def hood():
    """The retractable chain guard over the chain's upper run (+x): an arm from its pivot at the housing, then a
    plate beside the teeth, its free end turned outward so whatever slides on along the bar rides it open."""
    px, py = L.HOOD_PIVOT
    x_in = L.BAR_X + 0.0025  # its inner face, 2.5 mm off the cutters
    y0, y1 = L.HOOD_Y
    plate_x = x_in + L.HOOD_T / 2
    lead = 0.012
    dx, dy = plate_x - px, y0 - py  # the arm, from the pivot to the plate's start
    arm = math.hypot(dx, dy)
    shapes = [mk.box((px + dx / 2, py + dy / 2, L.BAR_Z), (L.HOOD_T, arm + 0.002, L.HOOD_H), L.DARK,
                     ypr=(math.atan2(-dx, dy) - math.pi, 0.0, 0.0)),
              mk.box((plate_x, (y0 + y1) / 2, L.BAR_Z), (L.HOOD_T, abs(y1 - y0), L.HOOD_H), L.DARK),
              mk.box((plate_x + lead / 2 * math.sin(L.HOOD_LEAD), y1 - lead / 2 * math.cos(L.HOOD_LEAD), L.BAR_Z),
                     (L.HOOD_T, lead, L.HOOD_H), L.DARK, ypr=(L.HOOD_LEAD, 0.0, 0.0))]
    return mk.hinge("hood", "Z", (px, py, L.BAR_Z), shapes, upper=L.HOOD_OPEN, mass=0.03, spring=L.HOOD_SPRING,
                    damping=0.002)


DRIVE = (0.25, 0.25, 0.27)
CUTTER = (0.80, 0.80, 0.82)


def _battery():
    hx = L.BAT_X / 2 + L.HOOK_SIZE[0] / 2
    hooks = [mk.slider(name, axis, [mk.box((sx * hx, L.HOOK_Y, 0.0), L.HOOK_SIZE, L.BODY)], upper=L.HOOK_TRAVEL,
                       mass=0.005, spring=L.HOOK_SPRING, damping=2.0)
             for name, axis, sx in (("hook_left", "-X", 1), ("hook_right", "X", -1))]
    shapes = [mk.box((0.0, 0.0, 0.0), (L.BAT_X, L.BAT_L, L.BAT_Z), L.BATTERY_COLOUR)]
    home = np.array([L.PRUNER_AT[0], L.PRUNER_AT[1], L.TOP]) + L.BAT_HOME
    return mk.assembly(BATTERY, _f(home), shapes, joints=hooks, mass=L.M_BATTERY, color=L.BATTERY_COLOUR,
                       yaw=L.HANDLE_YAW, friction=(0.6, 0.5), contact_offset=0.002)


def _guard():
    w, l, h, p, sd = L.GUARD_W, L.GUARD_L, L.GUARD_H, L.GUARD_PLATE, L.GUARD_SIDE
    shapes = [mk.box((0.0, 0.0, sz * (h / 2 - p / 2)), (w, l, p), L.GUARD_COLOUR) for sz in (-1, 1)]
    shapes += [mk.box((sx * (w / 2 - sd / 2), 0.0, 0.0), (sd, l, h - 2 * p), L.GUARD_COLOUR) for sx in (-1, 1)]
    shapes.append(mk.box((0.0, -l / 2 + p / 2, 0.0), (w, p, h - 2 * p), L.GUARD_COLOUR))  # the closed end
    return mk.assembly(GUARD, _f(L.at(*L.GUARD_START, h / 2 + 0.0005)), shapes, mass=L.M_GUARD,
                       color=L.GUARD_COLOUR, friction=(0.4, 0.3), contact_offset=0.002)


PRUNER_SPEC, BATTERY_SPEC, GUARD_SPEC = _pruner(), _battery(), _guard()
OBJECTS = [PRUNER_SPEC, BATTERY_SPEC, GUARD_SPEC]


# -- the two latches (rules) ----------------------------------------------------------------------------------------
IDENTITY = (0.0, 0.0, 0.0, 1.0)
BATTERY_LATCH = mk.weld(BATTERY_SPEC, PRUNER_SPEC, (_f(L.BAT_HOME), _f(mk.yaw_quat(L.HANDLE_YAW))),
                        name="battery_latched")


_SLIDE = {"episode": None, "on_from_tip": False}


def _guard_home(states) -> bool:
    """The scabbard's catch (scoring.catch_ready): it clicks only once the scabbard has been seen on the bar's line,
    over the bar from its tip side and short of the seat, and has then been slid along the bar to within 4 mm of
    the seat (either side). A scabbard aligned anywhere else (past the housing, off the bar) never triggers it."""
    if _SLIDE["episode"] != mk.episode():
        _SLIDE.update(episode=mk.episode(), on_from_tip=False)
    p, g = S.body(states[PRUNER]), S.body(states[GUARD])
    if p is None or g is None:
        return False
    covered, gy = S.guard_on(p, g)
    ready, _SLIDE["on_from_tip"] = S.catch_ready(covered, gy, _SLIDE["on_from_tip"])
    return ready


GUARD_CATCH = mk.weld(GUARD_SPEC, PRUNER_SPEC, ((0.0, L.GUARD_SEAT_Y, L.BAR_Z), IDENTITY), when=_guard_home,
                      name="scabbard_caught")
_LATCH = {"episode": None, "max_disp": 0.0}


def hooks_pressed(states) -> bool:
    try:
        return all(float(states[BATTERY].joints[h]) >= L.HOOK_FREE for h in HOOKS)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


@mk.on_step
def _battery_latch(states) -> None:
    """The locking hooks (scoring.latch_step): in from the start; free while both are pressed; caught again where it
    is if they are let go before the battery is drawn 8 mm; clicked back home if pushed in with them let go."""
    if _LATCH["episode"] != mk.episode():
        _LATCH.update(episode=mk.episode(), max_disp=0.0)
        BATTERY_LATCH.fire(states)
        return
    p, b = S.body(states[PRUNER]), S.body(states[BATTERY])
    if p is None or b is None:
        return
    disp = float(np.linalg.norm(p.local(b.origin) - L.BAT_HOME))
    if not BATTERY_LATCH.active:
        _LATCH["max_disp"] = max(_LATCH["max_disp"], disp)
    act = S.latch_step(BATTERY_LATCH.active, hooks_pressed(states), disp, _LATCH["max_disp"])
    if act == "free":
        BATTERY_LATCH.release()
        _LATCH["max_disp"] = disp
    elif act == "click":
        BATTERY_LATCH.fire(states)
    elif act == "catch":
        BATTERY_LATCH.fire(states, rel_pose=(_f(p.local(b.origin)), _f(mk.matrix_to_quat(p.rot.T @ b.rot))))


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
    """Battery out and put down, scabbard seated over the chain, hands clear, nothing touched that shouldn't be; held
    1 s. Runs every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "battery_latched": BATTERY_LATCH.active, "scabbard_caught": GUARD_CATCH.active, **result.details}
    try:
        row["hood_deg"] = round(math.degrees(float(objects[PRUNER].joints["hood"])), 1)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        row["hooks_mm"] = [round(float(objects[BATTERY].joints[h]) * 1000, 2) for h in HOOKS]
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    if arms is not None:
        try:
            row["tcp"] = {s: [round(float(v), 4) for v in arms[s].tcp] for s in ("left", "right")}
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 battery out, scabbard seated, hands clear; 0.75 seated; 0.5 scabbard on the bar; 0.25 battery out."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, nothing read from the simulator. Straight-down grips; yaw 90 degrees puts the pads either side along x.
#   1. The right hand pins the housing to the bench with its closed fingertips; the left pinches the battery across its
#      two locking hooks (the pinch presses both), draws it out of the handle and puts it down on the bench.
#   2. The left hand pins the housing; the right picks the scabbard up by its sides near its closed end, lifts it to
#      the bar's height on the bar's line, and slides it on over the chain until its catch clicks.
OPEN = -0.04695
PINCH = -(L.BAT_X / 2 - 0.004)  # closes 4 mm past the battery's sides: both hooks pressed home
GUARD_PINCH = -(L.GUARD_W / 2 - 0.004)
CLOSED = 0.0
YAW_X = math.pi / 2  # pads either side along x
TIP = 0.013  # the pad tips below the TCP
ORIGIN = np.array([L.PRUNER_AT[0], L.PRUNER_AT[1], L.TOP])
BATTERY_DOWN = L.at(0.200, 0.300)  # where the battery is put down
PULL = 0.075  # draws the battery this far: out of the 60 mm in the handle, with room
VARIANTS = ("guard_first", "battery_only")


def _w(duration, pos, finger, label, yaw=0.0) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


def pruner_pt(x, y, z) -> np.ndarray:
    return ORIGIN + np.array([x, y, z])


def pin_tcp(y: float, top: float) -> np.ndarray:
    """Closed fingertips pressed 20 mm into the pruner's top at ``y`` (pruner frame): the arm's stiffness holds it down."""
    return pruner_pt(0.0, y, top + TIP - 0.020)


BATTERY_PINCH = ORIGIN + L.handle_to_pruner(L.BAT_HOME_H + [0.0, L.HOOK_Y, 0.004])
YAW_BATTERY = YAW_X + L.HANDLE_YAW - math.pi  # pads either side across the battery (the same pads, half a turn round,
# keeps the wrist inside its limits)
GUARD_GRIP = L.at(L.GUARD_START[0], L.GUARD_START[1] - 0.040, 0.017)  # 40 mm from its centre toward the closed end
GUARD_TO_TCP = GUARD_GRIP - L.at(L.GUARD_START[0], L.GUARD_START[1], L.GUARD_H / 2 + 0.0005)
BAR_LINE_Z = L.TOP + L.BAR_Z


def guard_tcp(centre_y_world: float) -> np.ndarray:
    """The TCP holding the scabbard with its centre on the bar's line at ``centre_y_world``."""
    return np.array([ORIGIN[0] + AIM_DX, centre_y_world, BAR_LINE_Z + AIM_DZ]) + GUARD_TO_TCP


SEAT_Y_WORLD = ORIGIN[1] + L.GUARD_SEAT_Y
AIM_DX = float(os.environ.get("PRUNER_AIM_DX", 0.003))  # the held scabbard sits about 3 mm short in x and 1 mm low
AIM_DZ = float(os.environ.get("PRUNER_AIM_DZ", 0.001))  # (worked out on the simulator)
UP = np.array([0.0, 0.0, 0.06])
PARK = {"left": L.at(0.16, 0.26, 0.10), "right": L.at(0.16, -0.26, 0.10)}  # out of the top camera's view of the bench


def battery_steps(t0: float = 0.0) -> tuple[list, list]:
    """(left, right) waypoints: the battery out and down, the right hand pinning the housing meanwhile."""
    pin = pin_tcp(-0.020, L.HOUSING[2][1])
    pull = BATTERY_PINCH + L.HANDLE_AXIS * PULL
    left = [_w(2.5 + t0, BATTERY_PINCH + UP, OPEN, "over the battery's end", YAW_BATTERY),
            _w(1.2, BATTERY_PINCH, OPEN, "down: the pads either side of it", YAW_BATTERY),
            _w(0.8, BATTERY_PINCH, PINCH, "pinch: both locking hooks pressed", YAW_BATTERY),
            _w(1.5, pull, PINCH, "draw it straight out of the handle", YAW_BATTERY),
            _w(1.0, pull + [0.0, 0.0, 0.03], PINCH, "lift it clear", YAW_BATTERY),
            _w(1.5, BATTERY_DOWN + [0.0, 0.0, 0.03 + L.BAT_Z / 2 + 0.004], PINCH, "carry it to the bench spot", YAW_BATTERY),
            _w(0.8, BATTERY_DOWN + [0.0, 0.0, L.BAT_Z / 2 + 0.004], PINCH, "set it down", YAW_BATTERY),
            _w(0.6, BATTERY_DOWN + [0.0, 0.0, L.BAT_Z / 2 + 0.004], OPEN, "let go", YAW_BATTERY),
            _w(0.8, BATTERY_DOWN + [0.0, 0.0, 0.07], OPEN, "up", YAW_BATTERY),
            _w(0.8, BATTERY_DOWN + [0.0, -0.12, 0.07], OPEN, "aside, clear of it", YAW_BATTERY)]
    right = [_w(2.5 + t0, pin + UP, CLOSED, "over the housing, fingers closed"),
             _w(1.2, pin, CLOSED, "press the housing down onto the bench"),
             _w(3.5, pin, CLOSED, "hold it while the battery comes out"),
             _w(0.8, pin + UP, CLOSED, "lift off")]
    return left, right


def guard_steps(t_left: float, t_right: float) -> tuple[list, list]:
    """(left, right) waypoints: the left hand pins the housing, the right fits the scabbard."""
    pin = pin_tcp(0.010, L.HOUSING[2][1])
    start_y = ORIGIN[1] + L.BAR_Y[0] - 0.010 - L.GUARD_L / 2  # open end 10 mm short of the bar's tip
    right = [_w(2.0, GUARD_GRIP + UP, OPEN, "over the scabbard's closed end", YAW_X),
             _w(1.2, GUARD_GRIP, OPEN, "down: the pads either side of it", YAW_X),
             _w(0.8, GUARD_GRIP, GUARD_PINCH, "pinch its sides", YAW_X),
             _w(1.0, GUARD_GRIP + [0.0, 0.0, 0.04], GUARD_PINCH, "lift it", YAW_X),
             _w(1.5, guard_tcp(start_y) + [0.0, 0.0, 0.02], GUARD_PINCH, "carry it to the bar's line, above", YAW_X),
             _w(1.0, guard_tcp(start_y), GUARD_PINCH, "down onto the bar's line, open end at the tip", YAW_X),
             _w(3.0, guard_tcp(SEAT_Y_WORLD + 0.002), GUARD_PINCH, "slide it on over the chain until it clicks",
                YAW_X),
             _w(0.6, guard_tcp(SEAT_Y_WORLD + 0.002), OPEN, "let go", YAW_X),
             _w(1.0, guard_tcp(SEAT_Y_WORLD) + [0.0, 0.0, 0.07], OPEN, "up", YAW_X),
             _w(1.5, PARK["right"], OPEN, "out to the side, clear of the view", YAW_X)]
    t_guard = t_right + sum(w.duration for w in right[:5])  # the scabbard reaches the bar's line
    left = [_w(max(0.5, t_guard - t_left - 1.2), pin + UP, CLOSED, "over the housing, fingers closed"),
            _w(1.2, pin, CLOSED, "press the housing down onto the bench"),
            _w(sum(w.duration for w in right[5:8]), pin, CLOSED, "hold it while the scabbard goes on"),
            _w(1.0, pin + [0.0, 0.0, 0.07], CLOSED, "up"),
            _w(1.5, PARK["left"], CLOSED, "out to the side, clear of the view")]
    return left, right


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "guard_first":  # the scabbard straight on, the battery left in
        left, right = guard_steps(0.0, 0.0)
        return {"left": left, "right": right}
    bl, br = battery_steps()
    if variant == "battery_only":
        return {"left": bl + [_w(1.5, PARK["left"], CLOSED, "out to the side")],
                "right": br + [_w(1.5, PARK["right"], CLOSED, "out to the side")]}
    gl, gr = guard_steps(sum(w.duration for w in bl), sum(w.duration for w in br))
    return {"left": bl + gl, "right": br + gr}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      guard_first   the scabbard fitted with the battery still in
      battery_only  the battery out, the scabbard left off"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
