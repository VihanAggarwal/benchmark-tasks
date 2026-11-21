"""The scorer, as plain functions of the binder's and the divider's states and the arms'. NumPy only; no simulator.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads two mechkit AssemblyStates: the binder (fixed base: covers, plate, the three fixed ring
halves; link ``gate``: the three movable halves and the thumb lever, on one hinge) and the divider (one loose body
with three square holes along its punched edge).

    OPEN       the gate 0.55 rad or more open (its detent catches it there)
    CLOSED     the gate 0.10 rad or less from shut (the movable tips against the fixed ones)
    THREADED   for each of the three rings, a stretch of its wire (either half) crosses the divider's mid-plane
               INSIDE that ring's hole; with the gate open the crossing must lie 8 mm or more along the wire from
               the wire's tip (hooked on the tip alone is not on the ring)
    RESTING    the divider's frame moving under 5 mm/s and the gate under 0.1 rad/s
    HANDS OFF  both grippers' finger zones 30 mm or more from every part of the divider
    LATCH      any part of the divider more than 3 mm below the table top, or past the table's edge, at any step

    0      a latch fired; or neither open nor threaded
    0.3    gate open, divider not on all three rings
    0.7    divider on all three rings (the holes round the wires), but not all of: gate closed, resting, hands off,
           on every step of the last 1 s
    1.0    divider on all three rings, gate closed, resting and hands off on every one of the last 20 steps

Every end state falls in exactly one tier. Nothing reads the lever or the order of moves: only the poses.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BINDER, DIVIDER = "binder", "divider"
PARTS = (BINDER, DIVIDER)
DEPTH_MIN = 0.008  # m along the wire from its tip, with the gate open
FLAT_DEG = 15.0
HOLE_TOL = 0.0005
N_PTS = 25


def to_sheet(div):
    """A function: world points (N, 3) -> sheet coordinates (u, v, w) of the divider's frame."""
    r, o = div.rotation(), np.asarray(div.origin, dtype=float)

    def f(p):
        q = (np.atleast_2d(np.asarray(p, dtype=float)) - o) @ r
        return q @ L.SHEET_R0
    return f


def wire_points(binder, ring: int, half: str) -> np.ndarray:
    """World points of one ring half's wire centre line, from its foot to its tip."""
    xz = L.half_points(half, N_PTS)
    base = np.column_stack([xz[:, 0] - L.XS, np.full(len(xz), L.RING_Y[ring]), xz[:, 1]])
    return binder.link_to_world("gate", base) if half == "movable" else binder.to_world(base)


def crossings(pts: np.ndarray, sheet) -> list:
    """(u, v, metres from the tip) for each place the polyline crosses the sheet's mid-plane (w = 0)."""
    s = sheet(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    out = []
    for i in range(len(pts) - 1):
        w0, w1 = s[i, 2], s[i + 1, 2]
        if (w0 < 0 <= w1) or (w1 < 0 <= w0):
            f = w0 / (w0 - w1)
            u = s[i, 0] + f * (s[i + 1, 0] - s[i, 0])
            v = s[i, 1] + f * (s[i + 1, 1] - s[i, 1])
            out.append((float(u), float(v), float(seg[i + 1:].sum() + (1 - f) * seg[i])))
    return out


def in_hole(ring: int, u: float, v: float) -> bool:
    h = L.HOLE_SIZE / 2 + HOLE_TOL
    return abs(u - L.HOLE_U[ring]) <= h and abs(v - L.HOLE_V) <= h


def ring_depth(binder, div, ring: int):
    """The greatest distance along the wire from its tip at which a half of ring ``ring`` crosses the hole, or None."""
    sheet = to_sheet(div)
    best = None
    for half in ("fixed", "movable"):
        for u, v, d in crossings(wire_points(binder, ring, half), sheet):
            if in_hole(ring, u, v) and (best is None or d > best):
                best = d
    return best


@dataclass(frozen=True)
class Reading:
    q: float
    depths: tuple  # per ring: None or metres from the tip
    flat_deg: float
    origin: np.ndarray
    low: np.ndarray
    high: np.ndarray
    div: object


def read(states) -> Reading | None:
    if not isinstance(states, dict):
        return None
    try:
        b, d = states.get(BINDER), states.get(DIVIDER)
        if b is None or d is None or b.spec is None or d.spec is None:
            return None
        q = float(b.joint("gate"))
        depths = tuple(ring_depth(b, d, i) for i in range(3))
        n = d.rotation() @ L.N_VEC
        flat = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(n[2]))))))
        o = np.asarray(d.origin, dtype=float).reshape(3)
        lo, hi = np.asarray(d.low, dtype=float), np.asarray(d.high, dtype=float)
        if not (math.isfinite(q) and math.isfinite(flat) and np.all(np.isfinite(o)) and np.all(np.isfinite(lo))
                and np.all(np.isfinite(hi))):
            return None
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    return Reading(q, depths, flat, o, lo, hi, d)


def threaded(r: Reading) -> bool:
    """All three holes round their rings (with the gate open, 8 mm or more along the wire from the tip)."""
    need = 0.0 if r.q <= L.CLOSED_MAX else DEPTH_MIN
    return all(d is not None and d >= need for d in r.depths)


def tier(is_open: bool, is_threaded: bool, window_ok: bool) -> float:
    if is_threaded and window_ok:
        return 1.0
    if is_threaded:
        return 0.7
    if is_open:
        return 0.3
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.latch = False
        self.prev: Reading | None = None
        self.window: deque = deque(maxlen=L.WINDOW)
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.window.append(False)
            self.prev = None
            self.last = Result(min(self.last.score, 0.7), False, {"bad_state": True, "step": self.steps})
            return self.last
        off_x = r.low[0] < L.TABLE_X[0] or r.high[0] > L.TABLE_X[1]
        off_y = r.low[1] < L.TABLE_Y[0] or r.high[1] > L.TABLE_Y[1]
        if r.low[2] - L.TOP < -L.BELOW_TABLE or off_x or off_y:
            self.latch = True
        is_open = r.q >= L.OPEN_MIN
        is_closed = r.q <= L.CLOSED_MAX
        thr = threaded(r)
        if self.prev is not None:
            speed = float(np.linalg.norm(r.origin - self.prev.origin)) * L.CONTROL_HZ
            qspeed = abs(r.q - self.prev.q) * L.CONTROL_HZ
        else:
            speed, qspeed = math.inf, math.inf
        resting = speed < L.STILL_SPEED and qspeed < 0.1
        gap = rigstate.hands_gap(arms, r.div.distance)
        hands_off = gap is not None and gap >= L.HANDS_OFF - 1e-9
        self.window.append(bool(thr and is_closed and resting and hands_off))
        window_ok = len(self.window) == L.WINDOW and all(self.window)
        score = 0.0 if self.latch else tier(is_open, thr, window_ok)
        details = {"step": self.steps, "gate": round(r.q, 3), "open": is_open, "closed": is_closed,
                   "depth_mm": [None if d is None else round(d * 1000, 1) for d in r.depths], "threaded": thr,
                   "flat_deg": round(r.flat_deg, 1), "speed_mm_s": None if not math.isfinite(speed) else
                   round(speed * 1000, 1), "resting": resting, "hands_gap_mm": None if gap is None or not
                   math.isfinite(gap) else round(gap * 1000, 1), "hands_off": hands_off, "latch": self.latch,
                   "lowest_mm": round(float(r.low[2] - L.TOP) * 1000, 1)}
        self.prev = r
        self.last = Result(score, score >= 1.0, details)
        return self.last
