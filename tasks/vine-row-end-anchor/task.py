"""On the station table: two short vineyard trellis rows at about 1:12, side by side and parallel, running forward
and to the right at 15.1 degrees to x. Each row is a fixed tall end post (25 x 25 mm; 100 mm tall on the left row,
150 mm on the right) nearest the arms, a fixed line post 20 mm lower (80 / 130 mm) 130.5 mm beyond it along the
row, and a fixed trellis wire between them. Two loose concrete-grey anchor blocks (30 x 30 x 40 mm, 80 g) stand on the table, one in front of
each arm. Rule: each row's brace anchor goes on the row's line extended past its end post, as far from the end
post's centre as that end post is tall (so the brace wire would run at about 45 degrees): 100 mm out on the left
row, 150 mm on the right. Nothing is marked. Each row is scored on its own (0.5 each). Gripping from above, neither
arm reaches the other arm's block.

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
from scoring import Tracker, bottom_gap, is_set, offset_reach, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold this long. A block counts as set (lowest point within 3 mm of the table) while the gripper still
# holds it, from the end of its slow descent; settling and letting go take 0.5 + 1.5 s more. With 3 s the trial
# cannot end while the fingers are still closed on a block.
SUCCESS_HOLD_S = 3.0

INSTRUCTION = (
    "Set the brace anchor for each of the two vine rows. Each row has a tall end post nearest you and a shorter "
    "line post beyond it, and the row's line runs through the centres of its two posts. Stand one grey anchor block "
    "upright on each row's line, past its end post on the side away from the line post, with the block's centre as "
    "far from the end post's centre as that end post is tall. The two end posts are different heights."
)

def _row_objects(row: str, end, h: float, line) -> tuple[Box, Box, Box]:
    """A row's fixed end post, line post and the trellis wire between them, all turned along the row."""
    (ex, ey), (lx, ly) = end, line
    end_post = Box(f"end_post_{row}", pos=(ex, ey, L.TOP + h / 2), size=(L.POST_SIDE, L.POST_SIDE, h),
                   color=L.WOOD, yaw=L.ROW_YAW)
    lh = L.LINE_POST_H[row]
    line_post = Box(f"line_post_{row}", pos=(lx, ly, L.TOP + lh / 2),
                    size=(L.POST_SIDE, L.POST_SIDE, lh), color=L.WOOD, yaw=L.ROW_YAW)
    wire = Box(f"wire_{row}", pos=((ex + lx) / 2, (ey + ly) / 2, L.TOP + L.WIRE_Z),
               size=(L.WIRE_LEN, L.WIRE_SECTION, L.WIRE_SECTION), color=L.WIRE, yaw=L.ROW_YAW)
    return end_post, line_post, wire


_ROW_PARTS = [_row_objects(*r) for r in L.ROWS]
POSTS = [p for end_post, line_post, _ in _ROW_PARTS for p in (end_post, line_post)]
WIRES = [wire for *_, wire in _ROW_PARTS]
ANCHORS = [
    Box(name, pos=(x, y, L.TOP + L.ANCHOR_HALF[2]), size=L.ANCHOR_SIZE, mass=L.ANCHOR_MASS, color=L.CONCRETE)
    for name, (x, y) in L.ANCHORS
]
OBJECTS = [*POSTS, *WIRES, *ANCHORS]
ANCHOR_NAMES = tuple(a.name for a in ANCHORS)

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
    """A block's centre (mm), tilt (deg), bottom above the table (mm), its path over the last 0.5 s (mm), whether
    it is set, and its offset from and reach along each row's line (mm), for the trace."""
    try:
        out = {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
               "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
               "bottom_mm": round(bottom_gap(s) * 1000, 1),
               "path_mm": None if path is None else round(path * 1000, 2),
               "set": is_set(s)}
        for r in L.ROW_NAMES:
            off, reach = offset_reach(s, r)
            out[f"{r}_offset_mm"] = round(off * 1000, 1)
            out[f"{r}_reach_mm"] = round(reach * 1000, 1)
        return out
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """Both rows done: for each row a set block (upright on the table) within 10 mm of the row's line and with its
    centre within 10 mm of the end post's height beyond the end post's centre; neither block off the table; both
    still (centre moved under 5 mm over the last 0.5 s), held SUCCESS_HOLD_S = 3 s. This runs first every control
    step, so it also feeds the tracker."""
    blocks = [objects[n] for n in ANCHOR_NAMES]
    result = _TRACKER.update(blocks)
    row = {"step": _TRACKER.steps, "score": result.score, "success": result.success, **result.details}
    paths = _TRACKER.paths() if result.details.get("valid_state") else None
    row.update({n: _pose(objects[n], None if paths is None else paths[i]) for i, n in enumerate(ANCHOR_NAMES)})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a block is off the table. Else the sum of the two rows, each 0.5 (a set block within 10 mm of the
    row's line and its distance beyond the end post within 10 mm of that post's height), 0.25 (within 20 mm of the
    line and 15 mm of the height) or 0; a block counts for one row. Read from the state at hand (the last step)."""
    return _TRACKER.result([objects[n] for n in ANCHOR_NAMES]).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695  # fully open, 94 mm: 32 mm either side of a block
GRIP = -(L.ANCHOR_SIZE[1] - 0.008) / 2  # 22 mm commanded across the block's 30 mm: about 20 N per finger
YAW = 0.0  # both arms: the fingers close along y; the block is square, so its heading does not matter
GRIP_UP = 0.022  # the TCP this far above the block's bottom: the pad tips 9 mm above the table
DROP = 0.001  # each block is let go with its bottom 1 mm above the table
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + DROP + GRIP_UP
SLOW = 0.020  # the last 20 mm down onto a block, or onto the table, are taken slowly
HOVER_Z = L.TOP + 0.120  # travel height: a carried block's bottom 98 mm up
# Each arm sets the anchor of the row on its own side, both at once: the left arm its block to the left row's
# anchor point, the right arm its block to the right row's. The two place points are 0.367 m apart in y.
PLAN = {"left": ("anchor_left", "left"), "right": ("anchor_right", "right")}
# Per-arm trim of the place point (x, y), for the GPU run: if a block lands off its anchor point by a steady amount
# (the arm trailing or sagging under the 80 g block), read left/right offset_mm and reach_mm in TASK_TRACE (or the
# block's pos_mm) and move that arm's place point by minus the miss. Offline there is nothing to trim.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def _plan(arm: str) -> list[Waypoint]:
    """Pick the arm's own block from above, carry it at travel height over its row's anchor point, set it down
    slowly, let go, go straight up."""
    name, row = PLAN[arm]
    x, y = L.START[name]
    tx, ty = L.TARGETS[row]
    dx, dy = PLACE_TRIM[arm]
    px, py = tx + dx, ty + dy
    return [
        Waypoint(2.0, (x, y, HOVER_Z), OPEN, f"over {name}", YAW),
        Waypoint(1.0, (x, y, PICK_Z + SLOW), OPEN, "down to just above it", YAW),
        Waypoint(1.2, (x, y, PICK_Z), OPEN, "slowly down round it", YAW),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", YAW),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {name}", YAW),
        Waypoint(0.3, (x, y, PICK_Z), GRIP, "settle the grip", YAW),
        Waypoint(1.2, (x, y, HOVER_Z), GRIP, "lift it straight up", YAW),
        Waypoint(1.8, (px, py, HOVER_Z), GRIP, f"carry it over the {row} row's anchor point", YAW),
        Waypoint(1.0, (px, py, PLACE_Z + SLOW), GRIP, "down to just above the table", YAW),
        Waypoint(1.2, (px, py, PLACE_Z), GRIP, "slowly set it down on the row's line", YAW),
        Waypoint(0.5, (px, py, PLACE_Z), GRIP, "settle", YAW),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of it", YAW),
        Waypoint(0.3, (px, py, PLACE_Z), OPEN, "settle", YAW),
        Waypoint(1.0, (px, py, HOVER_Z), OPEN, "straight up, clear of it", YAW),
    ]


def demo():
    """Both arms on one clock, each setting the anchor of the row on its own side. DEMO_HOLD_ONLY=1: no plan (both
    arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm
    check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _plan("left"), "right": _plan("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
