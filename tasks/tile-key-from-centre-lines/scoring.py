"""The scorer, as plain functions of the two tiles' states. NumPy only; no simulator needed.

The rule is about the finished set-out, read from the state at the last control step. In each room the first tile
goes at the crossing of the two chalked centre lines, flat and square to the walls, and for each line either centred
on it or with a joint on it, whichever leaves both border cuts across that line at least half a tile (30 mm). The
scorer extends the laid tile's grid of 60 mm tiles and 2 mm joints to the walls and measures the cuts (see
``layout``'s docstring for the arithmetic). A room scores 0.5 only when both of its directions are right: one
tile per room, two decisions per tile. Either tile may go in either room.

A small tracker is fed once per control step by ``task.success()``: it keeps each tile's corners over the last
0.5 s (a tile is 'still' when none of its four top corners moved more than 2 mm in that time; a gripped tile's
reported velocity chatters, so its path is used, not its velocity) and recognises a new episode when the simulator
puts both tiles back on their starting marks in one step.

    Tracker.update(tiles) -> Result(score, success, details)

``tiles`` is a dict {name: state} of the two tiles, each anything with ``pos`` (bounding-box centre, which for a
box is its centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as
``yam_arena.task_file.ObjectState`` has. The tiles are plain boxes (``layout.TILE_SIZE``), so their corners follow
from ``pos`` and ``quat_xyzw``; either 60 x 60 mm face may be up. A state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

HALF = np.array(L.TILE_SIZE, dtype=float) / 2
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
_TOP = np.array([(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)], dtype=float)  # one face's four corners
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
ROOM_SCORE = 0.5


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the
    world. (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the right sizes, and the quaternion not degenerate (a simulator blow-up gives NaN,
    inf or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    sizes_ok = [p.size for p in parts] == [3, 4, 3, 3, 3, 3]
    return bool(sizes_ok and np.linalg.norm(parts[1]) > 0.5)


# -- one tile -------------------------------------------------------------------------------------------------
def tilt_deg(s) -> float:
    """Angle between the vertical and the axis through the tile's two 60 x 60 mm faces (degrees, 0..90): the
    tile is the same either face up, so whichever face is nearer to facing down counts."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def square_err_deg(s) -> float:
    """How far the tile's edges are turned from the walls (x and y), seen from above (0..45 degrees; a square
    tile has four ways round). A tile standing on an edge with its edges' plan view degenerate reads 45."""
    r = rotation(s.quat_xyzw)
    best = 45.0
    for col in (0, 1):  # either in-face edge direction, whichever is nearer horizontal
        hx, hy = float(r[0, col]), float(r[1, col])
        if math.hypot(hx, hy) > 0.5:
            a = math.degrees(math.atan2(hy, hx)) % 90.0
            best = min(best, a, 90.0 - a)
    return float(best)


def corners(s) -> np.ndarray:
    """The tile's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * HALF) @ rotation(s.quat_xyzw).T


def marks(s) -> np.ndarray:
    """Four corners of one 60 x 60 mm face (world, 4 x 3): what the stillness check follows."""
    return np.asarray(s.pos, dtype=float) + (_TOP * HALF) @ rotation(s.quat_xyzw).T


def lowest(s) -> float:
    """Height of the tile's lowest point above the table top (m)."""
    return float(corners(s)[:, 2].min() - L.TOP)


def offsets(s, room: L.Room) -> tuple[float, float]:
    """The tile centre's offsets from the room's two centre lines (m, signed): from the front-to-back line
    (along y; + to the left) and from the side-to-side line (along x; + away from the arms)."""
    return float(s.pos[1]) - room.cy, float(s.pos[0]) - room.cx


def at_crossing(s, room: L.Room) -> bool:
    """The tile's centre within 45 mm of both of the room's centre lines."""
    u_fb, u_ss = offsets(s, room)
    return bool(abs(u_fb) <= L.AT_CROSSING and abs(u_ss) <= L.AT_CROSSING)


def cuts(span: float, u: float) -> tuple[float, float]:
    """The two border cuts (m) across a centre line with ``span`` of room across it, for a tile whose centre is
    ``u`` from the line: the grid of 60 mm tiles and 2 mm joints extended from the tile to each wall leaves, at the
    wall on the + side and on the - side, the gap after the last whole tile minus one joint."""
    lead = L.HALF_TILE + L.JOINT  # 32 mm: the tile's half plus the joint before the next piece
    plus = (span / 2 - u - lead) % L.MODULE
    minus = (span / 2 + u - lead) % L.MODULE
    return float(plus), float(minus)


def direction_good(span: float, u: float) -> bool:
    """Both border cuts across this line at least half a tile (30 mm)."""
    return bool(min(cuts(span, u)) >= L.HALF_TILE - L.CUT_EPS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(float(v)) else None


def judge_room(room: L.Room, tiles: dict, still: dict[str, bool]) -> tuple[float, dict]:
    """One room's score (0 or 0.5) and why."""
    here = sorted(n for n, s in tiles.items() if at_crossing(s, room))
    d: dict = {"tiles_at_crossing": here}
    if not here:
        return 0.0, {**d, "zero_reason": "no_tile_at_crossing"}
    if len(here) > 1:
        return 0.0, {**d, "zero_reason": "two_tiles_at_crossing"}
    name = here[0]
    s = tiles[name]
    u_fb, u_ss = offsets(s, room)
    fb_cuts, ss_cuts = cuts(room.width, u_fb), cuts(room.depth, u_ss)
    d.update({
        "tile": name,
        "off_front_to_back_line_mm": _mm(u_fb),
        "off_side_to_side_line_mm": _mm(u_ss),
        "cuts_across_front_to_back_line_mm": [_mm(c) for c in fb_cuts],
        "cuts_across_side_to_side_line_mm": [_mm(c) for c in ss_cuts],
        "front_to_back_good": direction_good(room.width, u_fb),
        "side_to_side_good": direction_good(room.depth, u_ss),
        "tilt_deg": round(tilt_deg(s), 2),
        "square_err_deg": round(square_err_deg(s), 2),
        "lowest_mm": _mm(lowest(s)),
        "still": bool(still.get(name, False)),
    })
    if tilt_deg(s) > L.FLAT_DEG:
        return 0.0, {**d, "zero_reason": "not_flat"}
    if square_err_deg(s) > L.SQUARE_DEG:
        return 0.0, {**d, "zero_reason": "not_square"}
    if abs(lowest(s)) > L.ON_FLOOR:  # held up, resting on something, or fallen below the table top
        return 0.0, {**d, "zero_reason": "not_on_floor"}
    if not d["still"]:
        return 0.0, {**d, "zero_reason": "not_still"}
    if not (d["front_to_back_good"] and d["side_to_side_good"]):
        return 0.0, {**d, "zero_reason": "set_out_wrong"}
    return ROOM_SCORE, {**d, "zero_reason": None}


def evaluate(tiles: dict, still: dict[str, bool]) -> tuple[float, dict]:
    """The score of a state (both tiles valid) and the details for the trace."""
    rooms = {}
    total = 0.0
    for key, room in L.ROOMS.items():
        sc, det = judge_room(room, tiles, still)
        rooms[key] = {"score": sc, **det}
        total += sc
    return float(total), {"rooms": rooms, "rooms_right": [k for k, v in rooms.items() if v["score"] > 0]}


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: both tiles' corners over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf, a zero quaternion or a missing field in them
        self.history: dict[str, deque] = {}  # tile -> its face corners at this step and the 10 before it
        self.prev = None  # {tile: (centre, velocity)} at the previous valid control step
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(tiles: dict) -> bool:
        """The starting layout: both tiles flat on the floor on their marks, still (either tile on either mark)."""
        free = dict(L.TILE_START)
        for s in tiles.values():
            hit = None
            for mark, (sx, sy, syaw) in free.items():
                near = math.hypot(float(s.pos[0]) - sx, float(s.pos[1]) - sy) < L.START_TOL
                if near:
                    hit = mark
                    break
            if hit is None:
                return False
            free.pop(hit)
            slow = float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) < 0.05
            if not (abs(lowest(s)) < 0.003 and tilt_deg(s) < 2.0 and slow):
                return False
        return True

    def _teleported(self, now: dict) -> bool:
        """Some tile moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for n, (p0, v0) in self.prev.items() if n in now
        )

    def moved(self, name: str) -> float | None:
        """The farthest any of the tile's four face corners has been, over the last 0.5 s, from where it is now
        (m); None until 0.5 s of this trial has been seen."""
        h = self.history.get(name)
        if h is None or len(h) < self.window + 1:
            return None
        now = h[-1]
        return float(max(np.linalg.norm(p - now, axis=1).max() for p in h))

    def update(self, tiles: dict) -> Result:
        self.steps += 1
        if not all(valid(s) for s in tiles.values()):  # a blow-up: score 0 for this step, keep everything else
            self.bad_steps += 1
            self.last = self.result(tiles)
            return self.last
        now = {n: (np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for n, s in tiles.items()}
        if self.prev is not None and self._teleported(now) and self.at_start(tiles):
            bad = self.bad_steps
            self.reset()  # the simulator put both tiles back on their marks: a new episode
            self.steps, self.bad_steps = 1, bad
        self.prev = now
        for n, s in tiles.items():
            self.history.setdefault(n, deque(maxlen=self.window + 1)).append(marks(s))
        self.last = self.result(tiles)
        if self.last.score > 0 and self.last.score not in self.first_at:
            self.first_at[self.last.score] = self.steps
        return self.last

    def result(self, tiles: dict) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"step": self.steps, "bad_steps": self.bad_steps,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        bad = sorted(n for n, s in tiles.items() if not valid(s))
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_tiles": bad, "zero_reason": "invalid_state",
                                       **trial})
        moved = {n: self.moved(n) for n in tiles}
        still = {n: m is not None and m <= L.STILL_TOL for n, m in moved.items()}
        score, details = evaluate(tiles, still)
        details["moved_mm"] = {n: None if m is None else round(m * 1000, 2) for n, m in moved.items()}
        return Result(score, bool(score == 1.0), {"valid_state": True, "zero_reason": None, **details, **trial})
