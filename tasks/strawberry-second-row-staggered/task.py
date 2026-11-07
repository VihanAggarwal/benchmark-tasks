"""On the station table: a strawberry bed at 1:3. A fixed raised bed of black plastic mulch (220 mm across, 600 mm
long, 25 mm tall) runs across the table 30 cm in front of the arm bases, with the drip tape as a blue strip along its
centre line; six green strawberry plugs (upright cylinders 30 mm across, 45 mm tall, 25 g) stand on it as the far
row the transplanter set, 50 mm beyond the tape and 100 mm apart. Two fixed light-grey plug trays on the table, one
beside each arm's base, hold three more plugs each. Rule: plant the near row by hand, one plug level with the middle
of each of the far row's five gaps (staggered), as far on the near side of the tape as the far row is on the other
side: x 25.0 cm, y +20, +10, 0, -10 and -20 cm (not marked). 0.2 per place filled within 15 mm; a far-row spot left
without a plug standing upright within 10 mm of it scores 0. Gripping from above, each arm reaches only its own tray.

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
from scoring import Tracker, bottom_gap, planted, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. The last plug counts as planted (bottom within 3 mm of the bed's top) while the
# gripper still holds it, from the end of its slow descent; settling and letting go take 0.5 + 1.5 s more. With 3 s
# the trial cannot end while the fingers are still closed on a plug.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "The transplanter set the far row of strawberry plugs on this bed; plant the near row by hand. Stand one plug "
    "from the grey trays upright on the bed level with the middle of each gap in the far row, so the two rows are "
    "staggered, as far on the near side of the blue drip tape as the far row is on the other side. Do not disturb "
    "the far row."
)

# The bed: three fixed boxes with one flush top (near half, drip tape, far half).
_HALF_W = (L.BED_W - L.TAPE_W) / 2  # 102 mm each side of the tape
_BED_Z = L.TOP + L.BED_H / 2
BED = [
    Box("bed_near", pos=(L.BED_X - L.TAPE_W / 2 - _HALF_W / 2, 0.0, _BED_Z),
        size=(_HALF_W, 2 * L.BED_HALF_LEN, L.BED_H), color=L.MULCH),
    Box("drip_tape", pos=(L.BED_X, 0.0, _BED_Z), size=(L.TAPE_W, 2 * L.BED_HALF_LEN, L.BED_H), color=L.TAPE_BLUE),
    Box("bed_far", pos=(L.BED_X + L.TAPE_W / 2 + _HALF_W / 2, 0.0, _BED_Z),
        size=(_HALF_W, 2 * L.BED_HALF_LEN, L.BED_H), color=L.MULCH),
]
TRAYS = [
    Box(name, pos=(L.PAD_X, y, L.TOP + L.PAD_SIZE[2] / 2), size=L.PAD_SIZE, color=L.TRAY_GREY)
    for name, y in L.PADS
]
PLUGS = [
    Cylinder(name, pos=L.START[name], radius=L.PLUG_R, height=L.PLUG_H, mass=L.PLUG_MASS, color=L.LEAF)
    for name in L.NAMES
]
OBJECTS = [*BED, *TRAYS, *PLUGS]
PLUG_NAMES = tuple(p.name for p in PLUGS)

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


def _pose(i: int, s, path) -> dict:
    """A plug's centre (mm), tilt from vertical (deg), bottom above the bed's top (mm), its path over the last 0.5 s
    (mm), how far it is from where it started (mm) and whether it counts as planted, for the trace."""
    try:
        start = L.START[L.NAMES[i]]
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "bottom_mm": round(bottom_gap(s) * 1000, 1),
                "moved_mm": round(sum((float(a) - b) ** 2 for a, b in zip(s.pos, start)) ** 0.5 * 1000, 1),
                "path_mm": None if path is None else round(path * 1000, 2),
                "planted": planted(s)}
    except (AttributeError, TypeError, ValueError, IndexError):
        return {"pos_mm": None}


def _plugs(objects) -> list:
    return [objects.get(n) for n in PLUG_NAMES] if hasattr(objects, "get") else []


def success(objects) -> bool:
    """All five places filled (a plug planted upright within 15 mm of each), a plug still standing upright on each
    far-row spot (within 10 mm) and no plug off the table; every plug still (its centre moved under 5 mm over the
    last 0.5 s), held SUCCESS_HOLD_S = 3 s. This runs first every control step, so it also feeds the tracker."""
    plugs = _plugs(objects)
    result = _TRACKER.update(plugs)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    paths = _TRACKER.paths() if result.details.get("valid_state") else None
    if result.details.get("valid_state"):
        row.update({n: _pose(i, plugs[i], None if paths is None else paths[i]) for i, n in enumerate(PLUG_NAMES)})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if any of the six far-row spots is left without a plug planted upright on it (centre within 10 mm), or any
    plug is off the table. Else 0.2 for each of the five places (x 25.0 cm; y +20, +10, 0, -10, -20 cm) with a plug
    planted upright on the bed within 15 mm of it. Read from the state at hand (the last step)."""
    return _TRACKER.result(_plugs(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 32 mm either side of a plug
GRIP = -(2 * L.PLUG_R - 0.008) / 2  # 22 mm commanded across the plug's 30 mm
YAW = 0.0  # both arms: the fingers close along y; neighbours on a tray stand along x, planted ones 100 mm along y
GRIP_UP = 0.030  # the TCP this far above the plug's bottom: the pad tips 17 mm above it, the plug's top 15 mm above
DROP = 0.001  # each plug is let go with its bottom 1 mm above the bed's top
PICK_Z = L.PAD_TOP + GRIP_UP
PLACE_Z = L.BED_TOP + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a plug, or onto the bed, are taken slowly
HOVER_Z = L.TOP + 0.130  # travel height: a carried plug's bottom 100 mm up, 30 mm over the standing plugs' tops
# Which arm plants which place, in order (both arms on one clock). Each arm takes the plugs on its own tray, nearest
# its base first. The grippers are at least 0.30 m apart along y whenever both are over the bed.
#   1. left: plug_l1 to y +0.20;  right: plug_r1 to y -0.10
#   2. left: plug_l2 to y +0.10;  right: plug_r2 to y -0.20
#   3. left: plug_l3 to y  0.00;  right: back up over its tray (plug_r3 is the spare)
PLAN = {
    "left": (("plug_l1", 0.20), ("plug_l2", 0.10), ("plug_l3", 0.00)),
    "right": (("plug_r1", -0.10), ("plug_r2", -0.20)),
}
# Per-arm trim of the place point on the bed (x, y), for the GPU run: if one arm's plugs land off their places by
# the same amount (the pads off the commanded point), read the plugs' pos_mm in TASK_TRACE and move that arm's place
# point by minus the offset. A 25 g plug makes the arm sag well under 1 mm. Offline there is nothing to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _plant(name: str, arm: str, target_y: float, first: bool) -> list[Waypoint]:
    """Over a plug on the tray, slowly down round it, grip it, lift it straight up, carry it over its place, set it
    down slowly upright on the bed, let go, straight up."""
    x, y, _ = L.START[name]
    tx, ty = PLACE_TRIM[arm]
    px, py = L.NEAR_X + tx, target_y + ty
    return [
        Waypoint(2.0 if first else 1.8, (x, y, HOVER_Z), OPEN, f"over {name}", YAW),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.2, (x, y, PICK_Z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        Waypoint(1.8, (px, py, HOVER_Z), GRIP, "carry it over its place", YAW),
        Waypoint(1.0, (px, py, PLACE_Z + SLOW), GRIP, "down to just above the bed", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly stand it on the bed", YAW),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, HOVER_Z), OPEN, "straight up, clear of it", YAW),
    ]


def _plan(arm: str) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for i, (name, target_y) in enumerate(PLAN[arm]):
        wps += _plant(name, arm, target_y, first=(i == 0))
    if arm == "right":  # back over its own tray, well clear of the left gripper planting the middle place
        wps.append(Waypoint(2.0, (L.PAD_X, -L.PAD_Y, HOVER_Z), OPEN, "back up over its tray", YAW))
    return wps


def demo():
    """Both arms on one clock, each planting from its own tray. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready
    pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
