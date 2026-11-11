"""On the station table, a steel door (a fixed grey box 36 mm thick, 400 mm long, 80 mm tall) stands on edge along
y in front of both arms, hinge edge up, its top end at +y. A red HEAD line on the table at the door's top end stands
for the frame's head rabbet; a blue FLOOR line 3.75 mm past its bottom end for the finished floor. Lengths along the
door are to scale, 1 in = 5 mm (an 80 in door). Three loose brass hinge leaves (blocks 22.5 x 16 x 10 mm, 31 g, a
4-1/2 in hinge) lie on a parts tray in front of the door. Rule ('5, 10 and equal', the NAAMM/HMMA hinge locations
for hollow metal doors): the top hinge's top 5 in (25 mm) below the head rabbet, the bottom hinge's bottom 10 in
(50 mm) above the finished floor, the third hinge centred between them; each leaf flat on the edge, lengthwise and
centred across it, let go. The scorer reads the arms (pad-tip points and openings) through rigstate.py, which wraps
the embodiment's step and reset from this file's import, only to check that each leaf was let go.

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
from scoring import (  # noqa: E402
    Tracker, along_err_deg, bottom_above_edge, held, off_centreline, resting_on_edge, tilt_deg,
)
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success must hold 2 s. A leaf counts only once let go for 0.5 s (rigstate). Should the arms ever be unreadable,
# the middle leaf, the last one hung, would count from 1 s after its slow descent ends, while still gripped 1.5 mm
# above the edge; the settle and the opening take 0.6 + 1.5 s, so with a 2 s hold (1 + 2 s > 2.1 s) the trial
# still cannot end before the fingers are open.
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "This steel door stands on edge, hinge edge up. The red HEAD line is the frame's head rabbet, where the top "
    "of the door hangs; the blue FLOOR line is the finished floor below its bottom end. Lengths along the door are "
    "to scale: 1 in = 5 mm. Set the three hinge leaves on the hinge edge at the NAAMM/HMMA hinge locations for a "
    "hollow metal door, each lying flat, lengthwise along the edge and centred across it, then let go of them."
)

DOOR = Box("door", pos=(L.DOOR_X, 0.0, L.TOP + L.DOOR_H / 2), size=L.DOOR_SIZE, color=L.STEEL_GREY)
HEAD_LINE = Box("head_line", pos=(L.DOOR_X, L.HEAD_Y + L.LINE_W / 2, L.TOP + L.LINE_T / 2),
                size=(L.LINE_LEN, L.LINE_W, L.LINE_T), color=L.HEAD_RED)
FLOOR_LINE = Box("floor_line", pos=(L.DOOR_X, L.FLOOR_Y - L.LINE_W / 2, L.TOP + L.LINE_T / 2),
                 size=(L.LINE_LEN, L.LINE_W, L.LINE_T), color=L.FLOOR_BLUE)
TRAY = Box("parts_tray", pos=(L.TRAY_X, 0.0, L.TOP + L.TRAY_SIZE[2] / 2), size=L.TRAY_SIZE, color=L.TRAY_GREY)
LEAVES = [
    Box(name, pos=(x, y, L.TRAY_TOP + L.LEAF_H / 2), size=L.LEAF_SIZE, mass=L.LEAF_MASS, color=L.BRASS,
        yaw=math.radians(yaw))
    for name, (x, y, yaw) in L.LEAF_START.items()
]
OBJECTS = [DOOR, HEAD_LINE, FLOOR_LINE, TRAY, *LEAVES]
LEAF_NAMES = tuple(b.name for b in LEAVES)

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots the leaves teleporting back)."""
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


def _pose(s, arms) -> dict:
    """A leaf's centre (mm), tilt and turn from lengthwise (deg), its centre off the edge's centreline and its
    lowest point above the edge top (mm), whether it rests on the edge and whether a gripper holds it now."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s), 2),
                "along_err_deg": round(along_err_deg(s), 2),
                "off_centreline_mm": round(off_centreline(s) * 1000, 1),
                "bottom_above_edge_mm": round(bottom_above_edge(s) * 1000, 1),
                "resting_on_edge": resting_on_edge(s),
                "held": held(s, arms)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def _leaves(objects) -> dict:
    return {n: objects[n] for n in LEAF_NAMES}


def success(objects) -> bool:
    """1.0 now (three leaves on the edge, every location right, all three still for 1 s), held SUCCESS_HOLD_S.
    This runs first every control step, so it also feeds the tracker (with the arms rigstate read for this
    step)."""
    arms = rigstate.arms()
    leaves = _leaves(objects)
    result = _TRACKER.update(leaves, arms)
    row = {"episode": rigstate.episode(), "score": result.score, "success": result.success, **result.details}
    if isinstance(arms, dict):
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    if result.details.get("valid_state"):
        row.update({f"{n}_pose": _pose(objects[n], arms) for n in LEAF_NAMES})
    _trace(row)
    return result.success


def score(objects) -> float:
    """1.0 three leaves on the edge, all three locations right, still 1 s; 0.75 three on the edge with the top and
    bottom right; 0.5 two locations right; 0.25 one; else 0. Read from the state at hand (the last step)."""
    return _TRACKER.result(_leaves(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.022  # 44 mm: 10.75 mm off each end of a 22.5 mm leaf
GRIP = -(L.LEAF_LEN - 0.008) / 2  # 14.5 mm commanded along the leaf's length
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_UP = 0.003 + PAD  # the TCP over the leaf's underside: pad tips 3 mm above it, the pads on 7 of its 10 mm
DROP = 0.0015  # each leaf is let go with its underside this far above the edge
APPROACH = 0.020  # each final descent starts this far above its end point and takes 1.5 s
HOVER = L.TOP + 0.120  # travel: a carried leaf's underside 24 mm over the edge (joint 4 keeps 13 deg of margin
#                        over the middle location)
PICK_Z = L.TRAY_TOP + GRIP_UP
PLACE_Z = L.EDGE_TOP + DROP + GRIP_UP
# Yaw 0 closes the fingers along y: across the leaf's length, so the fingers sit off its two ends and, at the door,
# over the edge (never straddling the door's faces). Each leaf is laid at -1.15 deg: gripped parts end 0.6 to 1.7
# deg anticlockwise of the commanded yaw, so it lands within about 0.6 deg of lengthwise.
YAW_PICK = 0.0
YAW_PLACE = math.radians(-1.15)
# Parts land about 4 mm short of the commanded point along the travel direction: each place point is moved this far
# on, along the line from the leaf's start to its target (the 8 mm across and 5 mm along windows hold either way).
SHORT = 0.004
# Trim (x, y) of each place point, for the GPU run: if a leaf lands off its aim (TASK_TRACE: a leaf's
# off_centreline_mm away from 0, or top_err_mm / bottom_err_mm / middle_err_mm away from 0), move its place point by
# minus the offset.
PLACE_TRIM = {"top": (0.0, 0.0), "middle": (0.0, 0.0), "bottom": (0.0, 0.0)}
TARGET = {"top": (L.DOOR_X, L.TOP_Y), "middle": (L.DOOR_X, L.MID_NOMINAL), "bottom": (L.DOOR_X, L.BOTTOM_Y)}
LEAF_FOR = {"top": "hinge_leaf_a", "middle": "hinge_leaf_b", "bottom": "hinge_leaf_c"}
# Where each arm waits at the end: 12 cm up beyond the door's ends, clear of the door and both lines for the top
# camera.
PARK = {"left": (0.220, 0.250, L.TOP + 0.120), "right": (0.220, -0.250, L.TOP + 0.120)}


def place_point(location: str) -> tuple[float, float]:
    sx, sy, _ = L.LEAF_START[LEAF_FOR[location]]
    tx, ty = TARGET[location]
    d = math.hypot(tx - sx, ty - sy)
    ex, ey = PLACE_TRIM[location]
    return tx + SHORT * (tx - sx) / d + ex, ty + SHORT * (ty - sy) / d + ey


def _hang(location: str, first: bool) -> list[Waypoint]:
    """Over the leaf, slowly down round it, grip it along its length, lift, carry it over its location, slowly
    down onto the edge, settle, let go, straight up."""
    name = LEAF_FOR[location]
    sx, sy, _ = L.LEAF_START[name]
    px, py = place_point(location)
    return [
        Waypoint(2.5 if first else 2.0, (sx, sy, HOVER), OPEN, f"over {name}", YAW_PICK),
        Waypoint(1.0, (sx, sy, PICK_Z + APPROACH), OPEN, "down to just above it", YAW_PICK),
        Waypoint(1.5, (sx, sy, PICK_Z), OPEN, "slowly down round it", YAW_PICK),
        Waypoint(0.5, (sx, sy, PICK_Z), OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (sx, sy, PICK_Z), GRIP, f"grip {name} along its length", YAW_PICK),
        Waypoint(0.3, (sx, sy, PICK_Z), GRIP, "settle the grip", YAW_PICK),
        Waypoint(1.2, (sx, sy, HOVER), GRIP, "lift it", YAW_PICK),
        Waypoint(2.0, (px, py, HOVER), GRIP, f"carry it over the {location} hinge location", YAW_PLACE),
        Waypoint(0.5, (px, py, HOVER), GRIP, "settle over the edge", YAW_PLACE),
        Waypoint(1.0, (px, py, PLACE_Z + APPROACH), GRIP, "down to just above the edge", YAW_PLACE),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly lay it on the edge", YAW_PLACE),
        Waypoint(0.6, (px, py, PLACE_Z), GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", YAW_PLACE),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", YAW_PLACE),
        Waypoint(1.0, (px, py, HOVER), OPEN, "straight up, clear of it", YAW_PLACE),
    ]


def _plan(arm: str) -> list[Waypoint]:
    """Left arm: leaf A to the top location, then leaf B to the middle. Right arm, at the same time as the left
    arm's first leaf: leaf C to the bottom location, then it parks before the left arm comes to the middle."""
    if arm == "left":
        wps = _hang("top", True) + _hang("middle", False)
    else:
        wps = _hang("bottom", True)
    return wps + [Waypoint(1.5, PARK[arm], OPEN, "park, clear of the door", YAW_PLACE)]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
