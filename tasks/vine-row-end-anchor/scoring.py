"""The scorer, as plain functions of the two anchor blocks' states. NumPy only; no simulator needed.

The rule is about the finished end assemblies, read from the state at the last control step: for each row, an
anchor block set upright on the table on the row's line (the line through its two posts' centres), beyond the end
post on the side away from the line post, with its centre as far from the end post's centre as that end post is
tall. Each row is scored on its own. The posts are fixed, so the rows' lines come from ``layout``. A small tracker is
still fed once per control step by ``task.success()``: it keeps each block's recent path (for 'still', which success
needs), what the trace needs (when each score was first reached, how many observations were broken), and recognises
a new episode when the simulator puts both blocks back on their starting spots in one step.

    Tracker.update(anchors) -> Result(score, success, details)

``anchors`` (2, in the order of ``layout.NAMES``) are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. Each is a
plain 30 x 30 x 40 mm box whose 40 mm height is the body's z axis; its top and bottom are alike, so it may stand
either way up, and it is square, so its heading does not matter. Its lowest point is worked out from ``pos`` and
``quat_xyzw``. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and
leaves the tracker as it was.

The blocks are identical: either may serve either row, and they are matched to the rows the way that scores more
(each counts for at most one row).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9


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
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN, inf or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the block's 40 mm axis and the vertical (degrees, 0..90; either end up)."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def lowest(s) -> float:
    """Height (world z) of the block's lowest corner, from its pose."""
    r = rotation(s.quat_xyzw)
    return float(s.pos[2]) - float(sum(abs(r[2, i]) * L.ANCHOR_HALF[i] for i in range(3)))


def bottom_gap(s) -> float:
    """Height of the block's lowest point above the table top (m; negative = below it)."""
    return lowest(s) - L.TOP


def off_table(s) -> bool:
    """The block's lowest point is more than 10 mm below the table top: it fell off the table."""
    return bool(lowest(s) < L.TOP - L.OFF_TABLE_DROP)


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def is_set(s) -> bool:
    """Set: standing on the table on one of its 30 x 30 mm faces, its lowest point within 3 mm of the table top
    and its 40 mm axis tilted at most 10 degrees (not lying on a side, not on the other block, not held above)."""
    return bool(abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS and tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG + _EPS)


def offset_reach(s, row: str) -> tuple[float, float]:
    """For a row: (offset, reach) of the block's centre. Offset is its distance from the row's line (through the
    centres of the row's two posts); reach is how far it lies beyond the end post's centre along that line, away
    from the line post (negative beside or behind the end post)."""
    (ex, ey), (ux, uy), _ = L.row_axis(row)
    dx, dy = float(s.pos[0]) - ex, float(s.pos[1]) - ey
    return abs(ux * dy - uy * dx), ux * dx + uy * dy


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def row_score(s, row: str) -> tuple[float, dict]:
    """What one block earns for one row: 0.5 set with offset <= 10 mm and reach within 10 mm of H; 0.25 set with
    offset <= 20 mm and reach within 15 mm of H; 0 otherwise."""
    _, _, h = L.row_axis(row)
    off, reach = offset_reach(s, row)
    err = abs(reach - h)
    ok = is_set(s)
    d = {"set": ok, "offset_mm": _mm(off), "reach_mm": _mm(reach), "reach_err_mm": _mm(err)}
    if ok and off <= L.TIGHT_OFFSET + _EPS and err <= L.TIGHT_REACH + _EPS:
        return 0.5, d
    if ok and off <= L.LOOSE_OFFSET + _EPS and err <= L.LOOSE_REACH + _EPS:
        return 0.25, d
    return 0.0, d


def evaluate(anchors) -> tuple[float, dict]:
    """The score of the two end assemblies as they stand, and what was found (JSON-clean)."""
    blocks = list(anchors)
    names = list(L.NAMES[:len(blocks)])
    dropped = [names[i] for i, s in enumerate(blocks) if off_table(s)]
    table = [[row_score(s, r) for r in L.ROW_NAMES] for s in blocks]
    # Match blocks to rows one to one, the way that scores more (ties: block order, anchor_left to the left row).
    best, match = -1.0, None
    for perm in itertools.permutations(range(len(blocks)), min(len(blocks), len(L.ROW_NAMES))):
        total = sum(table[i][k][0] for k, i in enumerate(perm))
        if total > best + _EPS:
            best, match = total, perm
    rows = {}
    for k, r in enumerate(L.ROW_NAMES):
        i = match[k] if match is not None and k < len(match) else None
        if i is None:
            rows[r] = {"score": 0.0, "anchor": None}
        else:
            sc, d = table[i][k]
            rows[r] = {"score": sc, "anchor": names[i], **d}
    d = {"off_table": dropped, "left_row": rows["left"]["score"], "right_row": rows["right"]["score"],
         "left": rows["left"], "right": rows["right"]}
    if dropped:
        return 0.0, d
    return float(sum(rows[r]["score"] for r in L.ROW_NAMES)), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What success and the trace need about this trial (the score itself depends only on the state at hand)."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of both blocks at the previous valid control step
        self.history: deque = deque(maxlen=self.window + 1)  # both blocks' centres at the last valid steps

    @staticmethod
    def at_start(anchors) -> bool:
        """The starting layout: both blocks still and upright on the table, each on its own starting spot."""
        for s, name in zip(anchors, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            on_spot = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(bottom_gap(s)) < 0.003
            if not (on_spot and seated and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some block moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def paths(self) -> list[float] | None:
        """Each block's centre path (m) over the last 0.5 s, or None before 0.5 s of valid steps have been seen."""
        if len(self.history) < self.window + 1:
            return None
        h = list(self.history)
        return [float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(h, h[1:]))) for i in range(len(h[0]))]

    def update(self, anchors) -> Result:
        blocks = list(anchors)
        self.steps += 1
        if not all(valid(s) for s in blocks):  # a blow-up: score 0 for this step, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(blocks)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in blocks]
        if self.prev is not None and self._teleported(now) and self.at_start(blocks):
            self.reset()  # the simulator put both blocks back on their spots: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        self.last = self.result(blocks)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, anchors) -> Result:
        blocks = list(anchors)
        bad = [L.NAMES[i] for i, s in enumerate(blocks) if not valid(s)]
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_anchors": bad, **trial})
        score, details = evaluate(blocks)
        paths = self.paths()
        still = paths is not None and all(p < L.STILL_PATH for p in paths)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})

