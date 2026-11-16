"""The scorer, as plain functions of the six eggs', the carton's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the six eggs (base frame at mid-length, +z toward the blunt end) and of the carton (a
fixed assembly with one hinge link, "lid": 0 rad open at 110 degrees, 1.92 rad closed). Every reading is a pose, a
speed or a joint position; the arms are read only for the let-go check.

    CELL        an egg's centre within 12 mm of a cell's centre (top view), nearest cell, one egg to a cell
    SEATED      CELL, its centre within 6 mm of the seated height (the cell's cone holds it with the blunt end 28 mm
                over the rim), its long axis within 25 degrees of vertical, pointed end down
    CRACKED     latched: an egg ever moving at 0.8 m/s or more (a fall of about 33 mm, or a throw), or lost off the
                table (centre 10 mm under the table top)
    LID CLOSED  the lid turned to within 0.045 rad of closed (its 111 mm free edge within 5 mm of its closed place)
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the six eggs and the carton
                (lid included)
    STILL       over the last 1 s no egg moved more than 5 mm or turned more than 5 degrees and the lid turned no more than 5 degrees
                (motion a person sees on 30 fps video; no sub-millimetre reading)

    0      CRACKED, or not released and still, or no egg seated
    1.0    six eggs seated, lid closed
    0.75   six eggs seated, lid not closed
    0.5    four or five seated
    0.25   one to three seated
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

EGGS = L.EGGS
PARTS = EGGS + (L.CARTON_NAME,)
COS_DOWN = math.cos(math.radians(L.POINT_DOWN_DEG))


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # egg -> frame origin (mid-length)
    rot: dict  # egg -> rotation
    speed: dict  # egg -> speed (m/s)
    lid: float  # the lid's joint position (rad), 0 open .. LID_OPEN closed


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, speed = {}, {}, {}
    try:
        for name in EGGS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            v = np.asarray(s.lin_vel, dtype=float).ravel()
            if o.size != 3 or v.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))
                                                  and np.all(np.isfinite(v))):
                return None
            pos[name] = o - np.array([0.0, 0.0, L.TOP])
            rot[name] = r
            speed[name] = float(np.linalg.norm(v))
        c = states.get(L.CARTON_NAME)
        if c is None:
            return None
        q = float(c.joints["lid"])
        if not math.isfinite(q):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, speed, q)


# -- the readings ------------------------------------------------------------------------------------------------------
def cell_of(p) -> int | None:
    """The cell whose centre is nearest an egg's centre (top view), if within CELL_XY."""
    best, bi = math.inf, None
    for i, (cx, cy) in enumerate(L.CELLS):
        d = math.hypot(float(p[0]) - cx, float(p[1]) - cy)
        if d < best:
            best, bi = d, i
    return bi if best <= L.CELL_XY else None


def point_down(r: Reading, name: str) -> bool:
    """The long axis (+z of the egg's frame, pointed end to blunt end) within 25 degrees of vertical, blunt end up."""
    return bool(r.rot[name][2, 2] >= COS_DOWN)


def seated(r: Reading, name: str) -> int | None:
    """The cell an egg is seated in (centre over it, at the seated height, pointed end down), else None."""
    cell = cell_of(r.pos[name])
    if cell is None or not point_down(r, name):
        return None
    if abs(float(r.pos[name][2]) - L.EGG_Z_SEATED) > L.SEAT_TOL:
        return None
    return cell


def filled(r: Reading) -> list:
    """The cells holding a seated egg (each counted once), as a sorted list of cell indices."""
    return sorted({c for c in (seated(r, n) for n in EGGS) if c is not None})


def lid_closed(r: Reading) -> bool:
    return bool(r.lid >= L.LID_OPEN - L.LID_Q_TOL)


def cracked_now(r: Reading) -> bool:
    return any(r.speed[n] >= L.CRACK_SPEED or float(r.pos[n][2]) < L.FALL_Z for n in EGGS)


def tier(latched: bool, settled: bool, n_seated: int, lid: bool) -> float:
    if latched or not settled or n_seated <= 0:
        return 0.0
    if n_seated >= len(EGGS):
        return 1.0 if lid else 0.75
    if n_seated >= 4:
        return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The crack latch, the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"cracked": None, "cracked_egg": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["cracked"] is not None

    @staticmethod
    def starts() -> dict:
        return {n: np.array([x, y, L.EGG_START_Z]) for n, (x, y) in zip(EGGS, L.EGG_START)}

    def at_start(self, r: Reading) -> bool:
        return (all(float(np.linalg.norm(r.pos[n] - s)) <= L.START_TOL for n, s in self.starts().items())
                and r.lid < 0.02)  # a reset puts every egg back in the flat and the lid back open

    def teleported(self, prev: Reading, now: Reading) -> bool:
        """The eggs back in the flat and the lid open again, with a jump no trial makes in one step (a reset)."""
        if not self.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in EGGS)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        if self.latch["cracked"] is None and cracked_now(r):
            self.latch["cracked"] = self.steps
            self.latch["cracked_egg"] = next(n for n in EGGS
                                             if r.speed[n] >= L.CRACK_SPEED or float(r.pos[n][2]) < L.FALL_Z)
        try:
            parts = [states[n] for n in PARTS]

            def dist(points):
                """Distance to the nearest part. A part whose bounding box is 8 cm or more from a point is that far
                (the box's distance is a lower bound), so only parts near a finger zone get the exact shape sum."""
                pts = np.atleast_2d(np.asarray(points, dtype=float))
                best = np.full(len(pts), np.inf)
                for p in parts:
                    lo, hi = np.asarray(p.low, dtype=float), np.asarray(p.high, dtype=float)
                    box = np.linalg.norm(np.maximum(np.maximum(lo - pts, pts - hi), 0.0), axis=1)
                    d = np.where(box >= 0.08, box, p.distance(pts)) if np.any(box < 0.08) else box
                    best = np.minimum(best, d)
                return best
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, gap = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of any egg and the lid's turn (rad) from where they
        were at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = lid = 0.0
        for x in h[1:]:
            for n in EGGS:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
            lid = max(lid, abs(x.lid - first.lid))
        return {"move": move, "turn": turn, "lid": lid}

    def judge(self, r: Reading, gap) -> Result:
        cells = filled(r)
        lid = lid_closed(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS
                     and m["lid"] <= L.STILL_LID + L.EPS)
        level = tier(self.latched, released and still, len(cells), lid)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "n_seated": len(cells), "cells": cells, "lid_closed": lid, "lid_deg": round(math.degrees(r.lid), 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "egg_z_mm": [_mm(r.pos[n][2]) for n in EGGS],
            "egg_xy_mm": [[_mm(r.pos[n][0]), _mm(r.pos[n][1])] for n in EGGS],
            "egg_axis_up": [round(float(r.rot[n][2, 2]), 3) for n in EGGS],
            "egg_speed": [round(r.speed[n], 3) for n in EGGS],
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                              "lid_deg": round(math.degrees(m["lid"]), 3)},
        }
        return Result(level, bool(level == 1.0), details)
