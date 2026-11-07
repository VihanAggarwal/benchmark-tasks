"""On the station table: a row-crop planter at 1:4. A fixed dark-grey steel toolbar (700 mm long along y, 40 mm
wide, 30 mm tall) runs across in front of both arms, 30 cm out, with a red centre mark 10 mm wide at y 0 and a
fixed hitch tongue on the midline in front of it; two fixed light-grey spare-unit stands (90 x 90 x 10 mm) stand on
the table in front of it, one each side. Six loose green row units (blocks 70 mm fore and aft, 36 mm wide, 45 mm
tall, 80 g) stand square on the toolbar at six rows 20 in (127 mm) apart, symmetric about the mark. Rule: reset to
four rows at 30 in (190.5 mm), symmetric about the mark and so, with an even row count, none on it: units at
+-95.25 and +-285.75 mm from the mark; the two spares onto the stands. Each side of the mark is scored on its own
(0.5 each). Gripping from above, neither arm reaches the far side's middle or outer unit, its outer target or its
stand.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
from scoring import Tracker, bottom_gap, on_toolbar, set_on_bar, tilt_deg, yaw_err_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. The last two units count as set (bottom within 3 mm of the bar's top) while the
# grippers still hold them, from the end of their slow descent; settling and letting go take 0.5 + 1.5 s more. With
# 3 s the trial cannot end while the fingers are still closed on a unit.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Reset this planter from six rows at 20 inches to four rows at 30 inches. Move the green row units along the "
    "toolbar so that four of them stand upright on it, square to it and centred across its width, two on each side "
    "of the red centre mark, symmetric about the mark and 30 inches (190.5 mm at this 1:4 scale) apart centre to "
    "centre, with no unit on the mark. Stand the two spare units upright, one on each grey stand."
)

_HALF = L.BAR_HALF_LEN - L.MARK_HALF  # each grey half of the toolbar: 345 mm long
BAR_HALVES = [
    Box(name, pos=(L.BAR_X, sign * (L.MARK_HALF + _HALF / 2), L.TOP + L.BAR_H / 2), size=(L.BAR_W, _HALF, L.BAR_H),
        color=L.STEEL)
    for name, sign in (("toolbar_left", 1.0), ("toolbar_right", -1.0))
]
MARK = Box("centre_mark", pos=(L.BAR_X, 0.0, L.TOP + L.BAR_H / 2), size=(L.BAR_W, 2 * L.MARK_HALF, L.BAR_H),
           color=L.RED)
HITCH = Box("hitch_tongue", pos=(L.HITCH_X, 0.0, L.TOP + L.HITCH_SIZE[2] / 2), size=L.HITCH_SIZE, color=L.STEEL)
STANDS = [
    Box(name, pos=(L.STAND_X, y, L.TOP + L.STAND_SIZE[2] / 2), size=L.STAND_SIZE, color=L.STAND_GREY)
    for name, y in L.STANDS
]
UNITS = [
    Box(name, pos=(L.BAR_X, y, L.BAR_TOP + L.UNIT_HALF[2]), size=L.UNIT_SIZE, mass=L.UNIT_MASS, color=L.GREEN)
    for name, y in L.UNITS
]
OBJECTS = [*BAR_HALVES, MARK, HITCH, *STANDS, *UNITS]
UNIT_NAMES = tuple(u.name for u in UNITS)

_TRACKER = Tracker()


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


def _pose(s, path) -> dict:
    """A unit's centre (mm), distance from the mark (mm), tilt and squareness (deg), bottom above the toolbar's top
    (mm), its path over the last 0.5 s (mm) and what the scorer makes of it, for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "from_mark_mm": round(abs(float(s.pos[1])) * 1000, 1),
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "square_err_deg": round(yaw_err_deg(s.quat_xyzw), 2),
                "bottom_mm": round(bottom_gap(s) * 1000, 1),
                "path_mm": None if path is None else round(path * 1000, 2),
                "on_toolbar": on_toolbar(s), "set": set_on_bar(s)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """Both sides done: on each side of the mark exactly two units on the toolbar, set square, at 95.25 and 285.75
    mm from the mark within 10 mm, and a unit on that side's stand; none off the table or on the mark; every unit
    still (its centre moved under 5 mm over the last 0.5 s), held SUCCESS_HOLD_S = 3 s. This runs first every
    control step, so it also feeds the tracker."""
    units = [objects[n] for n in UNIT_NAMES]
    result = _TRACKER.update(units)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    paths = _TRACKER.paths() if result.details.get("valid_state") else None
    row.update({n: _pose(objects[n], None if paths is None else paths[i]) for i, n in enumerate(UNIT_NAMES)})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a unit is off the table or a unit on the toolbar covers the red mark. Else the sum of the two sides,
    each 0.5 (exactly two units on that side of the toolbar, set square, within 10 mm of 95.25 and 285.75 mm from
    the mark, and a unit on that side's stand), 0.25 (the same two within 25 mm, and either both within 10 mm or
    a unit on the stand) or 0. Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in UNIT_NAMES]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 29 mm either side of a unit; its neighbours stand 91 mm away at the start
GRIP = -(L.UNIT_SIZE[1] - 0.008) / 2  # 28 mm commanded across the unit's 36 mm width: about 20 N per finger
YAW = 0.0  # both arms: the fingers close along y (across the unit, along the bar), the long side stays along x
GRIP_UP = 0.028  # the TCP this far above the unit's bottom: the pad tips 15 mm above it (over the bar's top)
DROP = 0.001  # each unit is let go with its bottom 1 mm above what it is set on
PICK_Z = L.BAR_TOP + GRIP_UP
PLACE_Z = L.BAR_TOP + DROP + GRIP_UP
STAND_Z = L.STAND_TOP + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a unit, or onto a stand, are taken slowly
LIFT = 0.030  # a unit moved along the bar is carried with its bottom 31 mm above the bar's top
SHIFT_Z = PLACE_Z + LIFT
HOVER_Z = L.TOP + 0.140  # travel height: a carried unit's bottom 112 mm up, 37 mm over the standing units' tops
# Where each arm takes which unit, in order (both arms on one clock, mirror images of each other):
#   1. its middle unit (y +-190.5 mm) onto its own stand;
#   2. left: its outer unit in from 317.5 to 285.75 mm; right: its inner unit out from 63.5 to 95.25 mm;
#   3. left: its inner unit out; right: its outer unit in.
# In step 2 and 3 the grippers are at least 0.38 m apart along the bar.
PLAN = {
    "left": (("unit_l2", "stand"), ("unit_l3", L.TARGETS[1]), ("unit_l1", L.TARGETS[0])),
    "right": (("unit_r2", "stand"), ("unit_r1", -L.TARGETS[0]), ("unit_r3", -L.TARGETS[1])),
}
# Per-arm trim of the place point on the bar (x, y), for the GPU run: if one arm's units land off their targets by
# the same amount (the pads off the commanded point), read the units' pos_mm in TASK_TRACE and move that arm's place
# point by minus the offset. An 80 g unit held at its middle makes the arm sag about 1 mm. Offline there is nothing
# to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _pick(name: str, move: float) -> list[Waypoint]:
    """Over a unit on the toolbar, slowly down round it, grip it across its width."""
    x, y = L.START[name]
    return [
        Waypoint(move, (x, y, HOVER_Z), OPEN, f"over {name}", YAW),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.2, (x, y, PICK_Z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", YAW),
    ]


def _release(x: float, y: float, z: float) -> list[Waypoint]:
    return [
        Waypoint(0.5, (x, y, z), GRIP, "settle", YAW),
        Waypoint(1.5, (x, y, z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (x, y, z), OPEN, "settle", YAW),
        Waypoint(1.0, (x, y, HOVER_Z), OPEN, "straight up, clear of it", YAW),
    ]


def _to_stand(name: str, arm: str) -> list[Waypoint]:
    """A spare: lift it straight up, carry it over its side's stand, set it down slowly, let go."""
    x, y = L.START[name]
    sy = L.STAND_Y if arm == "left" else -L.STAND_Y
    return _pick(name, 2.0) + [
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        Waypoint(1.8, (L.STAND_X, sy, HOVER_Z), GRIP, "carry it over the stand", YAW),
        Waypoint(1.0, (L.STAND_X, sy, STAND_Z + SLOW), GRIP, "down to just above the stand", YAW),
        Waypoint(1.2, (L.STAND_X, sy, STAND_Z), GRIP, "slowly set it on the stand", YAW),
    ] + _release(L.STAND_X, sy, STAND_Z)


def _along_bar(name: str, arm: str, target_y: float) -> list[Waypoint]:
    """A row: lift it 31 mm off the bar, move it along the bar to its new place, set it down slowly, let go."""
    x, y = L.START[name]
    tx, ty = PLACE_TRIM[arm]
    px, py = L.BAR_X + tx, target_y + ty
    return _pick(name, 1.8) + [
        Waypoint(1.0, (x, y, SHIFT_Z), GRIP, "lift it off the bar", YAW),
        Waypoint(1.2, (px, py, SHIFT_Z), GRIP, "move it along the bar to its new place", YAW),
        Waypoint(1.8, (px, py, PLACE_Z), GRIP, "slowly set it square on the bar", YAW),
    ] + _release(px, py, PLACE_Z)


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for name, where in PLAN[arm]:
        wps += _to_stand(name, arm) if where == "stand" else _along_bar(name, arm, where)
    return wps


def demo():
    """Both arms on one clock, each resetting its own side. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
