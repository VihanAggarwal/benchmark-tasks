"""A 1:5 steam-table section on the station table: four full-size pans (106 x 65 mm, rims 30 mm up, all fixed) in a row
across the table 33 cm in front of the arm bases, long sides along x, hanging in a grey stainless deck flush with their
rims. Pan 1 (left, outermost, chilli) holds a ladle with a black handle and pan 3 (just right of the midline,
sweetcorn) a slotted spoon with a red handle: each stands in the food near the pan's front wall, its stem rising 26 mm
past the rim and its handle running out over the front rim toward the arms (fixed). Pans 2 (rice) and 4 (green beans)
have no utensil. Four loose lids (light grey PVC plates 106 x 65 x 3 mm with a black handle bar on top, 46-48 g) lie
crosswise on the counters beside the deck, two each side: on each counter one notched lid (a 26 x 25 mm notch in the
middle of one short end) and one solid lid. Rule: a notched lid on each pan with a utensil, turned so the handle comes
out through the notch (notch to the front) and the lid lies flat; a solid lid on each pan without one. A solid lid, or
a notched lid turned the wrong way, stands propped open on the handle.

Each lid is a loose mechkit compound (mechkit.py, next to this file): one rigid body of boxes, so the notch is real
and the scorer reads the lid's base frame (the middle of its plate's underside) and orientation.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the lids are built as compound bodies)
from scoring import LIDS, UTENSIL_PARTS, Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Still for 3 s, not the default 1 s. The last lid comes to rest on its pan while the slow descent is still finishing
# and success starts counting (after the 0.5 s needed to see it at rest) with the gripper still closed on its bar.
# Settling and letting go take 0.5 + 1.5 s, so with 3 s the trial ends with the gripper open and rising.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Cover the four pans on the steam table. A pan with a serving utensil in it gets a notched lid, turned so the "
    "utensil's handle comes out through the notch and the lid sits flat; a pan without a utensil gets a solid lid. "
    "Leave the utensils where they are."
)


def _span_box(name, x0, x1, y0, y1, z0, z1, color) -> Box:
    """A fixed box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


DECK = [_span_box(n, x0, x1, y0, y1, 0.0, L.PAN_H, L.DECK_STEEL) for n, x0, x1, y0, y1 in L.DECK_STRIPS]

PANS = []
for _p, (_y, _) in L.PANS.items():
    _x0, _x1 = L.PAN_X - L.PAN_L / 2, L.PAN_X + L.PAN_L / 2
    _y0, _y1 = _y - L.PAN_W / 2, _y + L.PAN_W / 2
    _w = L.WALL
    PANS += [
        _span_box(f"{_p}_front", _x0, _x0 + _w, _y0, _y1, 0.0, L.PAN_H, L.PAN_STEEL),
        _span_box(f"{_p}_back", _x1 - _w, _x1, _y0, _y1, 0.0, L.PAN_H, L.PAN_STEEL),
        _span_box(f"{_p}_left", _x0 + _w, _x1 - _w, _y1 - _w, _y1, 0.0, L.PAN_H, L.PAN_STEEL),
        _span_box(f"{_p}_right", _x0 + _w, _x1 - _w, _y0, _y0 + _w, 0.0, L.PAN_H, L.PAN_STEEL),
        _span_box(f"{_p}_food", _x0 + _w, _x1 - _w, _y0 + _w, _y1 - _w, 0.0, L.FOOD_TOP, L.FOOD_COLOURS[_p]),
    ]

UTENSILS = []
for _p, (_prefix, _colour) in L.UTENSILS.items():
    _y = L.PANS[_p][0]
    _hw = L.UTENSIL_W / 2
    UTENSILS += [
        _span_box(f"{_prefix}_bowl", L.BOWL_X0, L.BOWL_X1, _y - L.BOWL_W / 2, _y + L.BOWL_W / 2, L.BOWL_Z0, L.BOWL_Z1,
                  L.UTENSIL_STEEL),
        _span_box(f"{_prefix}_stem", L.STEM_X0, L.STEM_X1, _y - _hw, _y + _hw, L.STEM_Z0, L.STEM_Z1, _colour),
        _span_box(f"{_prefix}_handle", L.HANDLE_X0, L.HANDLE_X1, _y - _hw, _y + _hw, L.HANDLE_Z0, L.HANDLE_Z1,
                  _colour),
    ]

_SHAPE_COLOURS = {"plate": L.LID_STEEL, "bar": L.BAR_BLACK}


def _lid(name: str):
    kind, x, y, yaw = L.LIDS[name]
    return mk.assembly(
        name,
        origin=(x, y, L.TOP),
        yaw=yaw,
        mass=L.LID_MASS[kind],
        color=L.LID_STEEL,
        shapes=[mk.box(center, size, color=_SHAPE_COLOURS[key]) for _, center, size, key in L.LID_SHAPES[kind]],
    )


LID_OBJECTS = [_lid(n) for n in LIDS]
OBJECTS = [*DECK, *PANS, *UTENSILS, *LID_OBJECTS]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _states(objects) -> dict:
    return {n: objects[n] for n in (*LIDS, *UTENSIL_PARTS) if n in objects}


def success(objects) -> bool:
    """The right lid seated on every pan (a notched lid with its notch to the front on pans 1 and 3, a solid lid on
    pans 2 and 4): flat, every corner within 2 mm of the rim top, centred within 8 mm, square within 6 degrees, at rest;
    the utensils where they were set; held 3 s. This runs first every control step, so it also feeds the tracker."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets: a new trial (a teleport is also spotted)
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    states = _states(objects)
    result = _TRACKER.update(states)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    try:
        for n in LIDS:
            row[f"{n}_origin"] = states[n].origin
            row[f"{n}_quat"] = states[n].quat_xyzw
            row[f"{n}_source"] = getattr(states[n], "source", None)
    except (AttributeError, TypeError, ValueError, KeyError):
        row["state_error"] = True
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a utensil was moved more than 5 mm. Otherwise 0.25 for each pan with the right lid seated on it: on pans 1
    and 3 (a utensil) a notched lid with its notch to the front; on pans 2 and 4 a solid lid. Seated: tilted 3 degrees
    or less, every corner of the plate's underside within 2 mm of the rim top, its middle within 8 mm of the pan's
    centre, its long side within 6 degrees of the pan's, at rest. Read from the state at hand (the last step)."""
    return _TRACKER.result(_states(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN_F = -0.04695  # fully open, 94 mm
GRIP_F = -(L.BAR[1] - 0.008) / 2  # 12 mm commanded across the 20 mm handle bar
# Let go by opening to 70 mm only: the pads stay 25 mm clear of the bar and 9 mm short of a neighbouring lid's edge
# (pans are 85 mm apart), and it is wider than the plate (65 mm), which the offline checker needs to see a release.
RELEASE_F = -0.035
# Each lid is gripped across its handle bar (the fingers close along the lid's local y, so the gripper's yaw equals the
# heading of the lid's long side), at the bar's middle. Turning the gripper turns the lid by the same angle.
GRIP_UP = 0.021  # the TCP 21 mm above the lid's underside: pad tips 8 mm up, 5 mm above the plate, on 9 mm of the bar
DROP = 0.0005  # lids are let go with the plate's underside 0.5 mm above the rim
APPROACH = 0.045  # each final descent starts this far above its end: the plate 11.5 mm above the handles' tops
CARRY = L.TOP + 0.125  # carried with the plate's underside 104 mm up, 40 mm over the handles
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.RIM_TOP_W + DROP + GRIP_UP
# Trims of the place points (x, y), for the GPU run: if a lid lands off its pan's centre, read its origin in TASK_TRACE
# (<lid>_origin) against the pan's centre and move the place point by minus the offset (tolerance 8 mm; the notch
# leaves 8 mm either side of a handle).
TRIMS = {"pan1": (0.0, 0.0), "pan2": (0.0, 0.0), "pan3": (0.0, 0.0), "pan4": (0.0, 0.0)}
ASIDE_LEFT = (0.20, 0.25, L.TOP + 0.15)
ASIDE_RIGHT = (0.20, -0.25, L.TOP + 0.15)
HOLD_RIGHT = (0.25, -0.20)  # the right arm waits here with the notched lid while the left arm covers pan 2
WAIT_RIGHT = 8.8  # s: until the left arm has let go of its lid on pan 2 and risen (t = 34.0 s)


def _pick(name: str) -> list[Waypoint]:
    """From anywhere: over the lid's middle, down, slowly down beside its bar, settle, grip, lift (7.2 s)."""
    _, x, y, yaw = L.LIDS[name]
    return [
        Waypoint(1.5, (x, y, CARRY), OPEN_F, f"over {name}", yaw),
        Waypoint(1.0, (x, y, PICK_Z + 0.030), OPEN_F, "down to just above the lid", yaw),
        Waypoint(1.5, (x, y, PICK_Z), OPEN_F, "slowly down beside the bar", yaw),
        Waypoint(0.5, (x, y, PICK_Z), OPEN_F, "settle", yaw),
        Waypoint(1.5, (x, y, PICK_Z), GRIP_F, "grip the bar", yaw),
        Waypoint(1.2, (x, y, CARRY), GRIP_F, "lift the lid", yaw),
    ]


def _place(start_xy, pan: str, start_yaw: float, yaw: float = 0.0) -> list[Waypoint]:
    """From over ``start_xy``: carry over the pan, turning from ``start_yaw`` to ``yaw`` (in two halves, so no waypoint
    turns the wrist more than 45 degrees), settle, down, slowly down around the handle, settle, let go, straight up
    (9.8 s)."""
    x, y = L.PAN_X + TRIMS[pan][0], L.PANS[pan][0] + TRIMS[pan][1]
    mid = ((start_xy[0] + x) / 2, (start_xy[1] + y) / 2, CARRY)
    return [
        Waypoint(1.25, mid, GRIP_F, f"carry the lid toward {pan}, turning it", (start_yaw + yaw) / 2),
        Waypoint(1.25, (x, y, CARRY), GRIP_F, f"over {pan}, turned", yaw),
        Waypoint(0.8, (x, y, CARRY), GRIP_F, "settle over the pan", yaw),
        Waypoint(1.0, (x, y, PLACE_Z + APPROACH), GRIP_F, "down to 45 mm above the rim", yaw),
        Waypoint(2.5, (x, y, PLACE_Z), GRIP_F, "slowly set the lid on the rim", yaw),
        Waypoint(0.5, (x, y, PLACE_Z), GRIP_F, "settle", yaw),
        Waypoint(1.5, (x, y, PLACE_Z), RELEASE_F, "let go of the lid", yaw),
        Waypoint(1.0, (x, y, CARRY), RELEASE_F, "straight up", yaw),
    ]


def _cover(name: str, pan: str, start_xy=None) -> list[Waypoint]:
    _, x, y, yaw = L.LIDS[name]
    return _pick(name) + _place((x, y) if start_xy is None else start_xy, pan, yaw)


def _left() -> list[Waypoint]:
    """Left arm: the notched lid onto pan 1 (the ladle), turned +90 degrees so its notch comes from +y to the front
    (-x), round the ladle's handle; then the solid lid onto pan 2, turned to lie along the pan; then aside (35.5 s)."""
    plan = _cover("lid_notched_left", "pan1")
    plan += _cover("lid_solid_left", "pan2")
    plan += [Waypoint(1.5, ASIDE_LEFT, OPEN_F, "aside", 0.0)]
    return plan


def _right() -> list[Waypoint]:
    """Right arm: the solid lid onto pan 4; then the notched lid, held aside until the left arm has covered pan 2,
    onto pan 3 (the slotted spoon), turned -90 degrees so its notch comes from -y to the front; then aside (45.3 s)."""
    plan = _cover("lid_solid_right", "pan4")
    _, _, _, yaw = L.LIDS["lid_notched_right"]
    plan += _pick("lid_notched_right")
    plan += [
        Waypoint(1.0, (*HOLD_RIGHT, CARRY), GRIP_F, "hold the notched lid aside", yaw),
        Waypoint(WAIT_RIGHT, (*HOLD_RIGHT, CARRY), GRIP_F, "wait for the left arm to leave pan 2", yaw),
    ]
    plan += _place(HOLD_RIGHT, "pan3", yaw)
    plan += [Waypoint(1.5, ASIDE_RIGHT, OPEN_F, "aside", 0.0)]
    return plan


def demo():
    """Both arms at once. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
