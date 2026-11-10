"""On the station table, a 1:5 model of two rooms' floors, one in front of each arm, each walled with white 10 mm
walls 25 mm tall and with its two centre lines chalked wall to wall in blue (4 mm wide, 0.5 mm proud). The left
room is 278 mm front to back and 340 mm side to side (inside x 15.0-42.8 cm, y 7.0-41.0 cm); the right room is 340 mm
front to back and 278 mm side to side (x 12.0-46.0 cm, y -34.8 to -7.0 cm). A loose terracotta floor tile (60 x 60 x
10 mm, 80 g) lies in each room, away from the crossing and turned off square. Rule (tile setters' set-out from the
centre lines, 300 mm tiles and 10 mm joints at 1:5): in each room, lay the tile flat at the crossing, square to the
walls, and for each line either centred on it or with a 2 mm joint on it, whichever leaves both border cuts across
that line at least half a tile (30 mm). 0.5 per room with both directions right.

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
from scoring import Tracker, square_err_deg, tilt_deg  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
# Success must hold 2 s, not the default 1 s. A tile held 2 mm above the floor already counts as on it (the 2 mm
# tolerance), so with 1 s the trial could end while the fingers are still opening; with 2 s it ends after the
# fingers are fully open (let go 0.3 s after the slow descent, open over 1.5 s).
SUCCESS_HOLD_S = 2.0

INSTRUCTION = (
    "Each room has its two centre lines chalked in blue. In each room, lay the room's loose tile flat at the "
    "crossing of the centre lines as the first tile of the floor, square to the walls, and for each line either "
    "centred on it or with a joint on it, whichever leaves the cut tiles at both walls across that line at least "
    "half a tile wide. Tiles are 60 mm with 2 mm joints. The left room is 278 mm front to back and 340 mm wide; the "
    "right room is 340 mm front to back and 278 mm wide."
)


def _walls(key: str) -> list[Box]:
    """A room's four fixed walls: the near and far walls span its full outside width, the side walls fit between
    them."""
    r = L.ROOMS[key]
    t, h = L.WALL_T, L.WALL_H
    z = L.TOP + h / 2
    across = r.width + 2 * t
    return [
        Box(f"wall_{key}_near", pos=(r.x0 - t / 2, r.cy, z), size=(t, across, h), color=L.WALL_WHITE),
        Box(f"wall_{key}_far", pos=(r.x1 + t / 2, r.cy, z), size=(t, across, h), color=L.WALL_WHITE),
        Box(f"wall_{key}_right", pos=(r.cx, r.y0 - t / 2, z), size=(r.depth, t, h), color=L.WALL_WHITE),
        Box(f"wall_{key}_left", pos=(r.cx, r.y1 + t / 2, z), size=(r.depth, t, h), color=L.WALL_WHITE),
    ]


def _lines(key: str) -> list[Box]:
    """A room's two chalked centre lines, wall to wall: front to back (along x) and side to side (along y)."""
    r = L.ROOMS[key]
    z = L.TOP + L.LINE_T / 2
    return [
        Box(f"line_{key}_front_to_back", pos=(r.cx, r.cy, z), size=(r.depth, L.LINE_W, L.LINE_T),
            color=L.CHALK_BLUE),
        Box(f"line_{key}_side_to_side", pos=(r.cx, r.cy, z), size=(L.LINE_W, r.width, L.LINE_T),
            color=L.CHALK_BLUE),
    ]


WALLS = [*_walls("left"), *_walls("right")]
LINES = [*_lines("left"), *_lines("right")]
TILES = [
    Box(name, pos=(x, y, L.TOP + L.TILE_T / 2), size=L.TILE_SIZE, mass=L.TILE_MASS, color=L.TERRACOTTA,
        yaw=math.radians(yaw))
    for name, (x, y, yaw) in L.TILE_START.items()
]
OBJECTS = [*WALLS, *LINES, *TILES]
TILE_NAMES = tuple(t.name for t in TILES)

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


def _tiles(objects) -> dict:
    return {n: objects[n] for n in TILE_NAMES}


def _pose(s) -> dict:
    """A tile's centre (mm), tilt and squareness (deg) and speed (mm/s), for the trace."""
    try:
        return {"pos_mm": [round(float(v) * 1000, 1) for v in s.pos],
                "tilt_deg": round(tilt_deg(s), 2),
                "square_err_deg": round(square_err_deg(s), 2),
                "speed_mm_s": round(float(sum(float(v) ** 2 for v in s.lin_vel) ** 0.5) * 1000, 1)}
    except (AttributeError, TypeError, ValueError):
        return {"pos_mm": None}


def success(objects) -> bool:
    """In each room exactly one tile at the crossing (centre within 45 mm of both lines), flat (5 deg), square
    (5 deg), on the floor (lowest point within 2 mm of it), still (corners within 2 mm over 0.5 s), with both
    border cuts across each line >= 30 mm; held 2 s. This runs first every control step, so it also feeds the
    tracker."""
    tiles = _tiles(objects)
    result = _TRACKER.update(tiles)
    row = {"score": result.score, "success": result.success, **result.details}
    row.update({f"{n}_pose": _pose(objects[n]) for n in TILE_NAMES})
    _trace(row)
    return result.success


def score(objects) -> float:
    """0.5 for each room whose only tile at the crossing is laid flat, square, on the floor and still, with both
    border cuts across each centre line at least half a tile (30 mm); else 0 for that room. Read from the state at
    hand (the last step)."""
    return _TRACKER.result(_tiles(objects)).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.0395  # 79 mm: 9.5 mm either side of a 60 mm tile
GRIP = -(L.TILE - 0.008) / 2  # 52 mm commanded across the tile
PAD = 0.013  # the pad tips are this far below the tool point
GRIP_UP = 0.003 + PAD  # the TCP over the tile's underside: pad tips 3 mm above it, holding 7 of its 10 mm
DROP = 0.002  # each tile is let go with its underside this far above the floor (1.5 mm over a chalk line)
APPROACH = 0.030  # each final descent starts this far above its end point and takes 1.5 s
HOVER = L.TOP + 0.085  # carried with the tile's underside 69 mm up, 44 mm over the walls
PICK_Z = L.TOP + GRIP_UP
PLACE_Z = L.TOP + DROP + GRIP_UP
# Grip headings (rad). Yaw 0 closes the fingers along y. Each tile is gripped across two opposite edges, the
# fingers turned with it, and turned square on the way: the left arm grips its tile (turned +20 deg) at yaw +20,
# the right arm its tile (turned -25 deg) at yaw -25, and both lay them at yaw 0 (joint 6 is then near the middle of
# its range; the right arm at yaw +65 -> +90 would come within 5 deg of its limit). Open, each gripper's pads stay
# at least 24 mm from every wall.
YAW = {"left": (math.radians(20.0), 0.0), "right": (math.radians(-25.0), 0.0)}
START_OF = {"left": "tile_left", "right": "tile_right"}
# Where each arm waits at the end: 15 cm up, back over its room's near outer corner, clear of the laid tile, so the
# top camera sees both tiles.
REST = {"left": (0.200, 0.400, L.TOP + 0.150), "right": (0.200, -0.400, L.TOP + 0.150)}
# Trim (x, y) of each place point, for the GPU run: if a tile lands off its aim (in TASK_TRACE, a room's
# off_front_to_back_line_mm away from +31 (left room) or 0 (right room), or off_side_to_side_line_mm away from 0
# (left room) or -31 (right room)), move its place point by minus the offset. The windows reach 15 mm either way
# (14 mm outward from a joint line, where the 45 mm crossing limit ends them).
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}


def place_point(side: str) -> tuple[float, float]:
    tx, ty = L.TARGET[side]
    dx, dy = PLACE_TRIM[side]
    return tx + dx, ty + dy


def _lay(side: str) -> list[Waypoint]:
    """Over the tile, down (slowly for the last 30 mm), grip it across two edges, lift, carry it over the crossing
    turning it square to the walls, down (slowly), let go, up, out of the way."""
    sx, sy, _ = L.TILE_START[START_OF[side]]
    px, py = place_point(side)
    yaw_pick, yaw_place = YAW[side]
    return [
        Waypoint(2.0, (sx, sy, HOVER), OPEN, "over the tile", yaw_pick),
        Waypoint(1.0, (sx, sy, PICK_Z + APPROACH), OPEN, "down to just above the tile", yaw_pick),
        Waypoint(1.5, (sx, sy, PICK_Z), OPEN, "slowly down round the tile", yaw_pick),
        Waypoint(0.5, (sx, sy, PICK_Z), OPEN, "settle", yaw_pick),
        Waypoint(1.5, (sx, sy, PICK_Z), GRIP, "grip the tile across two edges", yaw_pick),
        Waypoint(0.3, (sx, sy, PICK_Z), GRIP, "settle the grip", yaw_pick),
        Waypoint(1.2, (sx, sy, HOVER), GRIP, "lift the tile", yaw_pick),
        Waypoint(2.0, (px, py, HOVER), GRIP, "carry it over the crossing, turning it square to the walls",
                 yaw_place),
        Waypoint(0.5, (px, py, HOVER), GRIP, "settle over the crossing", yaw_place),
        Waypoint(1.0, (px, py, PLACE_Z + APPROACH), GRIP, "down to just above the floor", yaw_place),
        Waypoint(1.5, (px, py, PLACE_Z), GRIP, "slowly lay the tile down", yaw_place),
        Waypoint(0.3, (px, py, PLACE_Z), GRIP, "settle", yaw_place),
        Waypoint(1.5, (px, py, PLACE_Z), OPEN, "let go of the tile", yaw_place),
        Waypoint(1.0, (px, py, HOVER), OPEN, "straight up", yaw_place),
        Waypoint(1.5, REST[side], OPEN, "out of the way", yaw_place),
    ]


def demo():
    """Both arms at once, each in its own room: the tool points stay more than 40 cm apart along y. The left room
    gets its tile centred on its side-to-side line with a joint on its front-to-back line; the right room the other
    way round. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right:
    that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _lay("left"), "right": _lay("right")}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
