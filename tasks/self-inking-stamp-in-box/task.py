"""On the station table, real size: a Trodat Printy 4912 self-inking stamp (black, 66 x 30 mm, 58 mm tall, die 47 x 18
mm reading RECEIVED) standing on its spot to the robot's right, its labelled front toward the robot, and a delivery
note (an A4 sheet on a 3 mm card backing) lying in front of the robot with a 30 x 60 mm 'GOODS RECEIVED' box printed
near its lower right corner. Objects (mechkit.py next to this file): the stamp (a loose mechanism: its base, and the
handle on a sprung slider down over it), the sheet (a loose compound with the box and text lines printed on it) and
the imprint (a thin red print, parked out of view until the stamp has printed).

Goal: stamp RECEIVED inside the box, square to it and the right way up, then put the stamp back where it stood. A
self-inking stamp prints only when its handle is pushed all the way down against its spring with the base flat on the
paper. See scoring.py for the tiers.

What is a rule, not physics: the die's flip and the ink. When the handle bottoms out with the base flat on the sheet
(scoring.py PRINT), the print is recorded where the die's footprint lies; once the stamp has been lifted 15 mm clear,
the red imprint is placed there on the sheet and held on it (a mechkit weld, rewritten every control step).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Variant: DEMO_VARIANT=half_press | skewed | not_back (see demo() below).
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
from scoring import HANDLE, IMPRINT, SHEET, STAMP, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Stamp RECEIVED inside the box marked on the delivery note, the right way up, then put the stamp back "
               "where it stood.")


# -- the objects ----------------------------------------------------------------------------------------------------
def _stamp():
    bx, by, bz = L.BASE
    base = [mk.box((0.0, 0.0, bz / 2), (bx, by, bz), L.BLACK)]
    hx, hy = L.HANDLE
    z0 = L.HANDLE_REST_Z
    skirt_h = L.HANDLE_TOP - L.CAP_T - z0
    w = L.SKIRT_WALL
    handle = [
        mk.box((0.0, 0.0, L.HANDLE_TOP - L.CAP_T / 2), (hx, hy, L.CAP_T), L.BLACK),
        mk.box((-(hx - w) / 2, 0.0, z0 + skirt_h / 2), (w, hy, skirt_h), L.BLACK),
        mk.box(((hx - w) / 2, 0.0, z0 + skirt_h / 2), (w, hy, skirt_h), L.BLACK),
        mk.box((0.0, -(hy - w) / 2, z0 + skirt_h / 2), (hx - 2 * w, w, skirt_h), L.BLACK),
        mk.box((0.0, (hy - w) / 2, z0 + skirt_h / 2), (hx - 2 * w, w, skirt_h), L.BLACK),
        # the label on the front (-x) face and the imprint preview on the cap, as on the real stamp
        mk.box((-hx / 2 - 0.0002, 0.0, z0 + skirt_h / 2), (0.0004, 0.040, 0.012), L.LABEL),
        mk.box((0.0, 0.0, L.HANDLE_TOP + 0.0002), (0.014, 0.040, 0.0004), L.INK),
    ]
    joint = mk.slider(HANDLE, "-Z", handle, upper=L.TRAVEL, anchor=(0.0, 0.0, L.HANDLE_TOP), mass=L.M_HANDLE,
                      spring=L.SPRING, damping=L.SPRING_DAMPING)
    return mk.assembly(STAMP, L.stamp_start(), base, mass=L.M_BASE, joints=[joint], color=L.BLACK,
                       friction=L.STAMP_FRICTION)


def _sheet():
    sx, sy, sz = L.SHEET
    z = sz + L.FILM / 2
    shapes = [mk.box((0.0, 0.0, sz / 2), (sx, sy, sz), L.PAPER)]
    bx, by = L.BOX_CENTRE
    ix, iy = L.BOX
    t = L.BOX_LINE
    for dx in (-1, 1):  # the box's two long sides and two ends, drawn just outside its inside
        shapes.append(mk.box((bx + dx * (ix + t) / 2, by, z), (t, iy + 2 * t, L.FILM), PRINT_BLACK))
    for dy in (-1, 1):
        shapes.append(mk.box((bx, by + dy * (iy + t) / 2, z), (ix, t, L.FILM), PRINT_BLACK))
    shapes.append(mk.box((bx - ix / 2 - 0.008, by, z), (0.004, 0.044, L.FILM), PRINT_BLACK))  # 'GOODS RECEIVED'
    # the delivery note's other printing: a heading and lines of text
    shapes.append(mk.box((0.120, 0.030, z), (0.010, 0.090, L.FILM), PRINT_BLACK))
    for i in range(8):
        shapes.append(mk.box((0.090 - 0.016 * i, 0.010, z), (0.003, 0.150 - 0.012 * (i % 3), L.FILM),
                             L.PRINT_GREY))
    for i in range(3):
        shapes.append(mk.box((-0.040 - 0.014 * i, 0.055, z), (0.003, 0.060, L.FILM), L.PRINT_GREY))
    return mk.assembly(SHEET, L.sheet_start(), shapes, mass=L.M_SHEET, color=L.PAPER, friction=L.SHEET_FRICTION)


def _imprint():
    """The red RECEIVED imprint: a 0.3 mm print, its border and its eight letters."""
    a, b = L.IMPRINT[1], L.IMPRINT[0]  # along x, along y
    t, f = 0.0012, 0.0003
    shapes = [mk.box((-(a - t) / 2, 0.0, f / 2), (t, b, f), L.INK), mk.box(((a - t) / 2, 0.0, f / 2), (t, b, f), L.INK),
              mk.box((0.0, -(b - t) / 2, f / 2), (a, t, f), L.INK), mk.box((0.0, (b - t) / 2, f / 2), (a, t, f), L.INK)]
    for i in range(8):  # the letters, reading left to right for the robot (+y to -y)
        shapes.append(mk.box((0.0, 0.0175 - 0.005 * i, f / 2), (0.009, 0.0032, f), L.INK))
    return mk.assembly(IMPRINT, (L.IMPRINT_PARK[0], L.IMPRINT_PARK[1], 0.0), shapes, mass=L.M_IMPRINT, color=L.INK,
                       contact_offset=0.0003)


PRINT_BLACK = (0.10, 0.10, 0.10)
STAMP_SPEC = _stamp()
SHEET_SPEC = _sheet()
IMPRINT_SPEC = _imprint()
OBJECTS = [STAMP_SPEC, SHEET_SPEC, IMPRINT_SPEC]


# -- scoring plumbing -----------------------------------------------------------------------------------------------
_TRACKER = Tracker()
_EPISODE = {"mechkit": None}
SHOW_LIFT = 0.015  # m: the stamp lifted this far off the sheet before the imprint is shown
SHOW_HOVER = 0.0006  # m: the imprint drawn just above the sheet's top (no contact with it)
_SHOW = mk.weld(IMPRINT_SPEC, SHEET_SPEC, name="imprint_on_sheet")


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()


@mk.on_step
def _show_imprint(states) -> None:
    """Once the stamp has printed and been lifted clear, place the imprint where it printed (sim only)."""
    if _SHOW.active or not _TRACKER.prints:
        return
    try:
        from scoring import read, sole_gap

        r = read(states)
        if r is None or sole_gap(r) < SHOW_LIFT:
            return
    except Exception:  # noqa: BLE001 - cosmetic only
        return
    p = _TRACKER.prints[0]
    yaw = math.radians(p.yaw_deg) / 2
    _SHOW.fire(states, rel_pose=((float(p.centre[0]), float(p.centre[1]), L.SHEET[2] + SHOW_HOVER),
                                 (0.0, 0.0, math.sin(yaw), math.cos(yaw))))


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
    """One print inside the box, square and the right way up, the sheet held still, the stamp back on its spot and
    let go; held 1 s. Runs every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    arms = rigstate.arms()
    result = _TRACKER.update(objects, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        row["stamp"] = {"origin": _r(objects[STAMP].origin), "quat_xyzw": _r(objects[STAMP].quat_xyzw),
                        "travel": round(float(objects[STAMP].joints[HANDLE]), 5)}
        row["sheet"] = _r(objects[SHEET].origin)
        row["imprint_shown"] = _SHOW.active
    except Exception as exc:  # noqa: BLE001
        row["objects_error"] = str(exc)[:200]
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
    """1.0 one square print in the box and the stamp back; 0.75 not back; 0.5 half in; 0.25 on the sheet; else 0."""
    return _TRACKER.last.score


# -- the scripted demo ----------------------------------------------------------------------------------------------
# Open loop, the right arm only (the left stays parked; the press is straight down, so the sheet does not need
# holding). A straight-down grip on the handle's cap, the pads closing on its two long sides 8 mm below the top. Lifted,
# carried over the box and set down on it square, the labelled front toward the robot; then the hand pushes the
# handle down its full 22 mm travel, aiming 25 mm past it (the arm gives under the spring's 12 N), holds it 0.8 s,
# lets it spring back up, lifts the stamp away, and stands it back on its spot.
OPEN = -0.04695
GRIP = -(L.HANDLE[0] / 2 - 0.004)  # 4 mm squeeze on the handle's 30 mm width
YAW = math.pi / 2  # pads closing along world x, across the stamp
TIPS_Z = 0.050  # the pad tips this far above the base's sole (8 mm below the stamp's top: on the cap), at rest
OVERDRIVE = 0.025  # the hand aims this far past the bottom of the travel (the arm gives under load); the pad tips
# still stay 3 mm above the paper
CARRY = 0.060  # m: the sole this far above the table while carrying
VARIANTS = ("half_press", "skewed", "not_back")


def tcp(sole_xy, sole_z: float, pressed: float = 0.0) -> np.ndarray:
    """The TCP holding the handle when the base's sole is at (sole_xy, sole_z) and the handle is down ``pressed``."""
    return np.array([sole_xy[0], sole_xy[1], sole_z + TIPS_Z + 0.013 - pressed])


def _w(duration, pos, finger, label, yaw=YAW) -> Waypoint:
    return Waypoint(float(duration), tuple(float(v) for v in pos), float(finger), label, float(yaw))


def plan(press: float = L.TRAVEL + OVERDRIVE, turn: float = 0.0, back: bool = True) -> list[Waypoint]:
    home = L.STAMP_SPOT
    box = L.box_world()
    sheet_top = L.TOP + L.SHEET[2]
    yaw = YAW + turn
    w = [
        _w(2.5, tcp(home, L.TOP) + [0.0, 0.0, 0.06], OPEN, "over the stamp"),
        _w(1.2, tcp(home, L.TOP), OPEN, "down: the pads either side of the handle"),
        _w(0.8, tcp(home, L.TOP), GRIP, "close on the handle"),
        _w(1.0, tcp(home, L.TOP + CARRY), GRIP, "lift the stamp"),
        _w(2.0, tcp(box, sheet_top + CARRY), GRIP, "over the box", yaw),
        _w(1.2, tcp(box, sheet_top + 0.004), GRIP, "down to just above the box, square to it", yaw),
        _w(0.8, tcp(box, sheet_top), GRIP, "set the base down on the paper", yaw),
        _w(1.2, tcp(box, sheet_top, press), GRIP, "push the handle down", yaw),
        _w(0.8, tcp(box, sheet_top, press), GRIP, "hold it down", yaw),
        _w(1.0, tcp(box, sheet_top), GRIP, "let the handle spring back up", yaw),
        _w(1.0, tcp(box, sheet_top + CARRY), GRIP, "lift the stamp off the paper", yaw),
    ]
    if not back:
        return w + [_w(4.0, tcp(box, sheet_top + CARRY), GRIP, "keep holding it", yaw)]
    return w + [
        _w(2.0, tcp(home, L.TOP + CARRY), GRIP, "back over its spot"),
        _w(1.2, tcp(home, L.TOP + 0.002), GRIP, "stand it down"),
        _w(0.8, tcp(home, L.TOP + 0.002), OPEN, "let go"),
        _w(1.0, tcp(home, L.TOP) + [0.0, 0.0, 0.07], OPEN, "straight up"),
        _w(1.5, (0.15, -0.25, L.TOP + 0.20), OPEN, "away", 0.0),
        _w(2.5, (0.15, -0.25, L.TOP + 0.20), OPEN, "hold", 0.0),
    ]


def plans(variant: str = "") -> dict:
    if variant not in ("", *VARIANTS):
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    if variant == "half_press":  # pushed only half way down: no print
        return {"right": plan(press=L.TRAVEL / 2)}
    if variant == "skewed":  # set down 12 deg off square: its corners cross the box's lines
        return {"right": plan(turn=math.radians(12.0))}
    if variant == "not_back":  # a good print, the stamp still in hand
        return {"right": plan(back=False)}
    return {"right": plan()}


def demo():
    """The right arm (the left stays parked). DEMO_HOLD_ONLY=1: no plan (a settling check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      half_press   the handle pushed only half way: no print (0)
      skewed       printed 12 deg off square, partly over the box's lines (0.5)
      not_back     a good print, the stamp still held at the end (0.75)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    return plans(os.environ.get("DEMO_VARIANT", ""))
