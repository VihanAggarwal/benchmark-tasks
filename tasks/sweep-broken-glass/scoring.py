"""The scorer, as plain functions of the pieces', the dustpan's and the arms' states. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
eight pieces of glass (plain boxes), the dustpan (a loose mechkit compound) and both arms through rigstate.

    IN_PAN      a piece's centre inside the dustpan's walls (seen in the pan's own frame: between the side walls and
                between the lip's edge and the back wall) and at most 12 mm above the pan's floor top
    PAN_FLAT    the dustpan within 5 deg of level with its lip's edge within 4 mm of the table top
    CLEAR       rigstate.released with the dustpan's and the brush's own distances: both grippers' finger zones 30 mm
                or more from each; unreadable arms are NOT clear
    STILL       over the last 20 steps no piece and no corner of the dustpan moved more than 2 mm
    FAILED      latched for the rest of the trial: a gripper touched a piece of glass (a finger zone reached it,
                rigstate.hands_gap <= 0; unreadable arms on a step count as touching), or a piece fell off the table

    n = pieces IN_PAN.  1.0: all 8 in, PAN_FLAT and CLEAR on each of the last 20 steps, and STILL (success).
    0.5: 6 or 7 in, or all 8 but not yet set down, let go and still.  0.25: 4 or 5 in.  0: fewer, or FAILED.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PAN, BRUSH = "dustpan", "brush"
NAMES = tuple(s[0] for s in L.SHARDS)
N_SHARDS = len(NAMES)
PAN_CORNERS = np.array([(0.0, -L.PAN_W / 2, 0.0), (0.0, L.PAN_W / 2, 0.0), (L.PAN_D, -L.PAN_W / 2, 0.0),
                        (L.PAN_D, L.PAN_W / 2, 0.0)]) * L.MM


def tilt_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


def box_distance(state, size):
    """Distance (m) from world points to a box piece (its own pose and size)."""
    c = np.asarray(state.pos, dtype=float)
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    half = np.asarray(size, dtype=float) / 2

    def distance(points):
        local = (np.atleast_2d(np.asarray(points, dtype=float)) - c) @ r
        q = np.abs(local) - half
        return np.linalg.norm(np.maximum(q, 0.0), axis=1)

    return distance


SIZES = {name: (sz[0] * L.MM, sz[1] * L.MM, L.SHARD_T * L.MM) for name, _, sz, _ in L.SHARDS}


@dataclass(frozen=True)
class Reading:
    pieces: np.ndarray  # (8, 3) piece centres (world)
    pan_origin: np.ndarray
    pan_rot: np.ndarray
    track: np.ndarray  # pieces and the pan's four floor corners


def read(states) -> Reading | None:
    try:
        pieces = np.array([np.asarray(states[n].pos, dtype=float).ravel() for n in NAMES])
        pan = states[PAN]
        rot = np.asarray(pan.rotation(), dtype=float)
        origin = np.asarray(pan.origin, dtype=float).ravel()
        corners = np.asarray(pan.to_world(PAN_CORNERS), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if pieces.shape != (N_SHARDS, 3) or rot.shape != (3, 3):
        return None
    if not all(np.all(np.isfinite(a)) for a in (pieces, rot, origin, corners)):
        return None
    return Reading(pieces, origin, rot, np.vstack([pieces, corners]))


def in_pan(r: Reading) -> np.ndarray:
    """Per piece: inside the pan's walls and resting on (at most 12 mm above) its floor."""
    local = (r.pieces - r.pan_origin) @ r.pan_rot  # pan frame: x from the lip back, y across, z up
    inner_y = (L.PAN_W / 2 - L.PAN_WALL) * L.MM
    inside = ((local[:, 0] >= 0.0) & (local[:, 0] <= (L.PAN_D - L.PAN_WALL) * L.MM)
              & (np.abs(local[:, 1]) <= inner_y))
    above_floor = local[:, 2] - L.PAN_FLOOR * L.MM
    return inside & (above_floor >= -0.002) & (above_floor <= L.IN_PAN_Z)


def pan_flat(r: Reading) -> bool:
    return bool(tilt_deg(r.pan_rot) <= L.PAN_FLAT_DEG + L.EPS and abs(float(r.pan_origin[2]) - L.TOP)
                <= L.PAN_ON_TABLE + L.EPS)


def tier(n: int, finished: bool, failed: bool) -> float:
    if failed:
        return 0.0
    if n >= N_SHARDS and finished:
        return 1.0
    if n >= 6:
        return 0.5
    if n >= 4:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.failed: str | None = None
        self.prev: Reading | None = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.ok: deque = deque(maxlen=self.window)
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        start = np.array([[x, y] for _, (x, y), _, _ in L.SHARDS])
        return bool(np.all(np.linalg.norm(r.pieces[:, :2] - start, axis=1) <= L.RESET_START_TOL))

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        return cls.at_start(now) and float(np.linalg.norm(now.track - prev.track, axis=1).max()) >= L.RESET_JUMP

    def touch(self, states, arms) -> str | None:
        if not rigstate.readable(arms):
            return "arms unreadable (counted as touching)"
        for n in NAMES:
            gap = rigstate.hands_gap(arms, box_distance(states[n], SIZES[n]))
            if gap is None or not math.isfinite(gap) or gap <= L.TOUCH:
                return f"a gripper touched {n}"
        return None

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.ok.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        self.hist.append(r)
        if self.failed is None:
            if np.any(r.pieces[:, 2] < L.TOP - L.OFF_TABLE_Z):
                self.failed = "a piece fell off the table"
            else:
                self.failed = self.touch(states, arms)
        mask = in_pan(r)
        n = int(mask.sum())
        flat = pan_flat(r)
        try:
            clear = bool(rigstate.released(arms, states[PAN].distance, L.LET_GO)
                         and rigstate.released(arms, states[BRUSH].distance, L.LET_GO))
        except Exception:  # noqa: BLE001
            clear = False
        self.ok.append(bool(n == N_SHARDS and flat and clear))
        full = len(self.ok) == self.window and all(self.ok)
        m = self.motion()
        still = bool(m is not None and m <= L.STILL_MOVE + L.EPS)
        level = tier(n, full and still, self.failed is not None)
        details = {
            "valid_state": True, "score": level, "in_pan": n, "which": [NAMES[i] for i in np.flatnonzero(mask)],
            "pan_flat": flat, "pan_tilt_deg": round(tilt_deg(r.pan_rot), 2), "clear_now": clear, "ok_20": bool(full),
            "still_20": still, "motion_mm": None if m is None else _mm(m), "failed": self.failed,
            "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last

    def motion(self) -> float | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(x.track - h[0].track, axis=1).max()) for x in h[1:])
