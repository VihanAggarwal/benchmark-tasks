"""On the station table, a table-top dental surgery: a fixed light blue patient chair (a 80 x 300 x 40 mm body and a
60 x 50 x 40 mm headrest at its -y end) with a red mouth mark painted on the headrest at (0.260, 0.050). A patient
lying in it would have the top of the head toward -y and the feet toward +y. Three loose pieces of equipment: the
dentist's stool (blue cylinder, 50 mm across, 40 mm tall, 0.12 kg) and the assistant's mobile cart (grey box,
60 x 45 x 45 mm, 0.20 kg) on the right, and the assistant's stool (green cylinder, 50 mm across, 60 mm tall,
0.15 kg) beyond the foot of the chair on the left. Rule: the dentist is left-handed and works four-handed, so the
clock zones round the mouth mirror: the dentist's stool on the patient's left (12:30 to 3:30), the assistant's
stool on the patient's right (8:15 to 9:45), the cart on the assistant's side (8:15 to 11:45), each 80 to 200 mm
from the mouth, standing on the floor and let go, and nothing within 200 mm in the transfer zone (5 to 8). The
scorer reads the arms (pad-tip points and openings) through rigstate.py only to check that each object was let go.

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
import rigstate  # noqa: E402
from scoring import Tracker, bottom_gap, clock, held, in_transfer, standing, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0, which already needs every object released over the last 0.5 s and still for 1 s) must hold this
# long. Should the arms ever be unreadable, the cart, the last one set down, would look released and still from the
# end of its slow descent; the settle and the opening take 0.5 + 1.5 s more, so with 2 s plus the 1 s of stillness
# the trial still cannot end while the fingers are closed on it.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "Set up this dental surgery for the next patient. The patient will lie in the chair with their head on the "
    "headrest and mouth at the painted mark. Today's dentist is left-handed and works four-handed with a "
    "chairside assistant. Put the dentist's stool (blue), the assistant's stool (green, the taller one) and the "
    "assistant's mobile cart (grey) where they go around the chair for this dentist and assistant, each standing "
    "upright on the floor, not on the chair, then let go of them."
)


def _block(name: str, rect, h: float, color) -> Box:
    x0, x1, y0, y1 = rect
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + h / 2), size=(x1 - x0, y1 - y0, h), color=color)


CHAIR = _block("patient_chair", L.CHAIR, L.CHAIR_H, L.CHAIR_BLUE)
HEADREST = _block("headrest", L.HEADREST, L.CHAIR_H, L.CHAIR_BLUE)
MOUTH = Cylinder("mouth_mark", pos=(L.MOUTH[0], L.MOUTH[1], L.TOP + L.CHAIR_H + L.MARK_H / 2),
                 radius=L.MOUTH_RADIUS, height=L.MARK_H, color=L.MOUTH_RED)


def _stool(name: str, h: float, color) -> Cylinder:
    x, y, _ = L.START[name]
    return Cylinder(name, pos=(x, y, L.TOP + h / 2), radius=L.STOOL_RADIUS, height=h, mass=L.MASS[name], color=color)


D_STOOL = _stool(L.D_STOOL, L.D_STOOL_H, L.STOOL_BLUE)
A_STOOL = _stool(L.A_STOOL, L.A_STOOL_H, L.STOOL_GREEN)
_cx, _cy, _cyaw = L.START[L.CART]
CART = Box(L.CART, pos=(_cx, _cy, L.TOP + L.CART_SIZE[2] / 2), size=L.CART_SIZE, mass=L.MASS[L.CART],
           color=L.CART_GREY, yaw=math.radians(_cyaw))
OBJECTS = [CHAIR, HEADREST, MOUTH, D_STOOL, A_STOOL, CART]
EQUIPMENT = L.NAMES

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the objects teleporting back)."""
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


def _pose(name: str, s, path, arms) -> dict:
    """An object's centre (mm), clock position from the mouth mark (h:mm, degrees, mm), tilt (deg), bottom above the
    floor (mm), whether it is standing, in the transfer zone or held now, and its path over the last 0.5 s (mm)."""
    try:
        phi, r = clock(s)
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos], "clock": L.hours(phi),
                "phi_deg": round(phi, 1), "r_mm": round(r * 1000, 1), "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "bottom_mm": round(bottom_gap(name, s) * 1000, 1), "standing": standing(name, s),
                "in_transfer": in_transfer(s), "held_now": held(name, s, arms),
                "path_mm": None if path is None else round(path * 1000, 2)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """1.0 now: the dentist's stool 12:30 to 3:30, the assistant's stool 8:15 to 9:45 and the cart 8:15 to 11:45,
    each standing 80 to 200 mm from the mouth mark, all released and still for 1 s, nothing in the transfer zone;
    held SUCCESS_HOLD_S = 2 s. This runs first every control step, so it also feeds the tracker (with the arms
    rigstate read for this step) and writes the trace."""
    arms = rigstate.arms()
    objs = [objects[n] for n in EQUIPMENT]
    result = _TRACKER.update(objs, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    if isinstance(arms, dict):
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    paths = _TRACKER.paths(1) if result.details.get("valid_state") else None
    row.update({n: _pose(n, objects[n], None if paths is None else paths[0][i], arms)
                for i, n in enumerate(EQUIPMENT)})
    _trace(row)
    return result.success


def score(objects) -> float:
    """The decision list on the state at hand (the last step): 0 if anything is held or moving or in the transfer
    zone; 1.0 all three in their left-handed windows and still for 1 s; 0.75 both stools; 0.5 the dentist's stool
    only; 0.25 the right-handed layout; else 0."""
    return _TRACKER.result([objects[n] for n in EQUIPMENT], rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 22 mm either side of a stool, 24 mm either side of the cart
GRIP = {L.D_STOOL: -(2 * L.STOOL_RADIUS - 0.008) / 2,  # 42 mm across the 50 mm stools
        L.A_STOOL: -(2 * L.STOOL_RADIUS - 0.008) / 2,
        L.CART: -(L.CART_SIZE[1] - 0.008) / 2}  # 37 mm across the cart's 45 mm width (along y)
YAW = 0.0  # every grip closes the fingers along world y
GRIP_UP = {L.D_STOOL: 0.024, L.A_STOOL: 0.035, L.CART: 0.026}  # TCP above the object's bottom (pad tips 13 mm lower)
DROP = 0.002  # each object is let go with its bottom 2 mm above the floor
SLOW = 0.020  # the last 20 mm down onto an object, or onto the floor, are taken slowly
TRAVEL_Z = {"left": L.TOP + 0.090, "right": L.TOP + 0.120}  # TCP travel height; the left arm stays low at x 0.36
#   (joint 4 keeps 16.5 deg there at 0.09 m, 11.4 at 0.11 m; the carried stool's bottom rides 55 mm up, over no part)
PARK = {"left": (0.400, 0.300, L.TOP + 0.120), "right": (0.250, -0.300, L.TOP + 0.150)}  # emptied start spots
# Arms trail their command by a few mm under load and parts land about 4 mm short of it: each set-down point is
# commanded LEAD farther from that arm's base than the object's target (all well inside the windows either way).
LEAD = 0.004
PLAN = {"left": (L.A_STOOL,), "right": (L.CART, L.D_STOOL)}
# The left arm hovers over the assistant's stool for WAIT_FIRST s before picking it, so it carries it only once the
# right arm has set the cart and risen clear of it (16.0 s; the left carries from 16.5 s): the two arms never work
# beside the chair at once (two set-downs side by side brought their links within 1 cm in the offline check).
WAIT_FIRST = {"left": 8.3, "right": 0.0}
BASE_Y = {"left": 0.305, "right": -0.305}
# Per-arm trim of the place points (x, y), for the GPU run: if one arm's objects land off their targets by the same
# amount, read pos_mm in TASK_TRACE and move that arm's place points by minus the offset. Offline nothing to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def target(name: str) -> tuple[float, float]:
    """Where the object should end up: its clock position from layout.DEMO_AT."""
    return L.clock_xy(*L.DEMO_AT[name])


def place_point(name: str, arm: str) -> tuple[float, float]:
    """The commanded set-down point: the target moved LEAD farther from the arm's base, plus PLACE_TRIM."""
    tx, ty = target(name)
    dx, dy = tx, ty - BASE_Y[arm]
    n = math.hypot(dx, dy)
    ox, oy = PLACE_TRIM[arm]
    return tx + LEAD * dx / n + ox, ty + LEAD * dy / n + oy


def _move(name: str, arm: str, first: bool) -> list[Waypoint]:
    """Over an object (waiting there WAIT_FIRST s before the first one), slowly down round it, grip it, lift it,
    carry it over its place, set it down slowly on the floor, let go, straight up."""
    x, y, _ = L.START[name]
    px, py = place_point(name, arm)
    pick_z = L.TOP + GRIP_UP[name]
    place_z = pick_z + DROP
    hz = TRAVEL_Z[arm]
    grip = GRIP[name]
    wait = WAIT_FIRST[arm] if first else 0.0
    return [
        Waypoint(2.5 if first else 1.8, (x, y, hz), OPEN, f"over the {name}", YAW),
        *([Waypoint(wait, (x, y, hz), OPEN, "wait for the other arm", YAW)] if wait > 0 else []),
        Waypoint(1.0, (x, y, pick_z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.2, (x, y, pick_z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, pick_z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, pick_z), grip, f"grip the {name}", YAW),
        Waypoint(0.3, (x, y, pick_z), grip, "settle the grip", YAW),
        Waypoint(1.2, (x, y, hz), grip, "lift it straight up", YAW),
        Waypoint(2.0, (px, py, hz), grip, "carry it over its place", YAW),
        Waypoint(1.0, (px, py, place_z + SLOW), grip, "down to just above the floor", YAW),
        Waypoint(1.5, (px, py, place_z), grip, "slowly set it on the floor", YAW),
        Waypoint(0.5, (px, py, place_z), grip, "settle", YAW),
        Waypoint(1.5, (px, py, place_z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, place_z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, hz), OPEN, "straight up, clear of it", YAW),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for k, name in enumerate(PLAN[arm]):
        wps += _move(name, arm, k == 0)
    return wps + [Waypoint(1.5, PARK[arm], OPEN, "park, clear of the chair", YAW)]


def demo():
    """Both arms on one clock: the right sets the cart at 11 o'clock and rises clear (by 16.0 s); the left hovers over
    the assistant's stool (to 10.8 s), grips it, carries it from 16.5 s, sets it at 9 o'clock and parks (25.8 s);
    the right meanwhile fetches the dentist's stool, carries it from 23.5 s, sets it at 1:30 and parks (32.8 s).
    DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's
    plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
