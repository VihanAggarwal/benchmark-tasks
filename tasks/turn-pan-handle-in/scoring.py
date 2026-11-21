"""The scorer, as plain functions of the pan's state and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
frying pan (a loose mechkit compound: base, flared wall, oil and handle as one rigid body) and both arms through
rigstate. Plan views are world x, y.

    HANDLE_SAFE  every point of the handle (its centre line and both side edges, rim to end, 36 points) is inside the
                 cooktop's outline seen from above, the handle is turned toward the back (its end farther from the
                 front edge than where it leaves the pan), AND every point is at least 20 mm outside the ring of each
                 hot element (the left rear, just switched off and still hot, and the right front, lit under a
                 saucepan)
    CENTRED      the pan's base centre within 30 mm (in plan) of its element's centre
    FLAT         the pan within 3 deg of level with its base within 4 mm of the glass
    CLEAR        rigstate.released with the pan's own distance: both grippers' finger zones 30 mm or more from the
                 pan; unreadable arms, or a distance that cannot be computed, are NOT clear
    STILL        over the last 20 steps the base centre and the handle's end moved at most 2 mm
    FAILED       latched for the rest of the trial: at any moment the pan tilted past 10 deg (hot oil spilled), its
                 base rose more than 25 mm above the glass, or its base centre went more than 60 mm from its
                 element's centre (slid off its burner)

    1.0   HANDLE_SAFE, CENTRED, FLAT and CLEAR on each of the last 20 steps, and STILL          (success)
    0.5   HANDLE_SAFE and FLAT now, but not the rest (still held, off centre by 30 to 60 mm, or not yet still)
    0     anything else: the handle past an edge or over a hot element, and every FAILED trial
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PAN = "pan"
HANDLE_PTS = L.handle_outline()
TRACK = np.vstack([np.zeros(3), L.handle_points()[-1]])  # base centre, handle end (pan frame)


def tilt_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


@dataclass(frozen=True)
class Reading:
    origin: np.ndarray  # the pan's base centre (world)
    rot: np.ndarray
    handle_xy: np.ndarray  # (36, 2) the handle (centre line and side edges) seen from above
    track: np.ndarray  # (2, 3)


def read(pan) -> Reading | None:
    if pan is None:
        return None
    try:
        rot = np.asarray(pan.rotation(), dtype=float)
        origin = np.asarray(pan.origin, dtype=float).ravel()
        handle = np.asarray(pan.to_world(HANDLE_PTS), dtype=float)
        track = np.asarray(pan.to_world(TRACK), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if rot.shape != (3, 3) or origin.size != 3 or handle.shape != HANDLE_PTS.shape:
        return None
    if not all(np.all(np.isfinite(a)) for a in (rot, origin, handle, track)):
        return None
    return Reading(origin, rot, handle[:, :2], track)


def centre_offset(r: Reading) -> float:
    x, y, _, _ = L.element(L.PAN_ELEMENT)
    return float(math.hypot(r.origin[0] - x, r.origin[1] - y))


def handle_yaw_deg(r: Reading) -> float:
    d = r.rot @ np.array([1.0, 0.0, 0.0])
    return math.degrees(math.atan2(d[1], d[0]))


def failure(r: Reading) -> str | None:
    if tilt_deg(r.rot) > L.TILT_FAIL_DEG:
        return "pan tilted past 10 deg (hot oil spilled)"
    if float(r.origin[2]) - (L.TOP + L.COOKTOP_TOP) > L.LIFT_FAIL:
        return "pan lifted more than 25 mm"
    if centre_offset(r) > L.CENTRE_FAIL:
        return "pan slid more than 60 mm off its element"
    return None


def tier(handle_safe: bool, flat: bool, full: bool, still: bool, failed: bool) -> float:
    if failed or not (handle_safe and flat):
        return 0.0
    return 1.0 if full and still else 0.5


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
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.failed: str | None = None
        self.prev: Reading | None = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.ok: deque = deque(maxlen=self.window)
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        pan = states.get(PAN) if isinstance(states, dict) else None
        r = read(pan)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.ok.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        self.prev = r
        self.hist.append(r)
        if self.failed is None:
            self.failed = failure(r)
        inside, clear_hot, problem = L.safe_plan(r.handle_xy)
        handle_safe = inside and clear_hot
        off = centre_offset(r)
        centred = off <= L.CENTRE_OK + L.EPS
        flat = (tilt_deg(r.rot) <= L.FLAT_DEG + L.EPS
                and abs(float(r.origin[2]) - (L.TOP + L.COOKTOP_TOP)) <= L.FLAT_Z + L.EPS)
        try:
            dist = pan.distance
            clear = bool(rigstate.released(arms, dist, L.LET_GO))
        except Exception:  # noqa: BLE001 - no distance: not clear
            dist, clear = None, False
        self.ok.append(bool(handle_safe and centred and flat and clear))
        full = len(self.ok) == self.window and all(self.ok)
        m = self.motion()
        still = bool(m is not None and m <= L.STILL_MOVE + L.EPS)
        level = tier(handle_safe, flat, full, still, self.failed is not None)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        details = {
            "valid_state": True, "score": level, "handle_yaw_deg": round(handle_yaw_deg(r), 2),
            "handle_inside": inside, "handle_clear_of_hot": clear_hot, "problem": problem,
            "centre_mm": _mm(off), "tilt_deg": round(tilt_deg(r.rot), 2),
            "base_above_glass_mm": _mm(float(r.origin[2]) - (L.TOP + L.COOKTOP_TOP)), "flat": bool(flat),
            "clear_now": bool(clear), "ok_20": bool(full), "still_20": still,
            "motion_mm": None if m is None else _mm(m), "failed": self.failed, "zone_mm": _mm(gap),
            "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last

    def motion(self) -> float | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(x.track - h[0].track, axis=1).max()) for x in h[1:])
