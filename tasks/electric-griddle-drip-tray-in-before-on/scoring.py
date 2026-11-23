"""The scorer, as plain functions of the griddle's joints and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the griddle (a fixed mechkit articulation): the drip tray's link pose (a slider along +y; 0 = pulled out) and the
dial's link pose (a hinge about the vertical; clockwise seen from above positive, 0 = OFF).

    SEATED   the tray within 1 cm of its stop on the rails
    ON       the dial 15 degrees or more past OFF (the element is on; heat is not simulated)
    SET      the dial within 15 degrees of the 350 mark (90 degrees, est)
    UNSAFE   latched for the rest of the trial: on any step the dial ON while the tray is not SEATED

    0      UNSAFE at any step
    1.0    seated and set, both grippers 40 mm clear on every step of the last 1 s, tray and dial still over it
    0.75   seated and set
    0.5    seated, the dial OFF or on but not set
    0.25   the tray slid 10 cm or more along the rails (not seated), the dial OFF
    0      otherwise
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

GRIDDLE = "griddle"
TRAY_JOINT, DIAL_JOINT = "drip_tray", "dial"


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


def dial_angle_deg(link_rot: np.ndarray) -> float:
    """The dial's turn from OFF (deg, clockwise seen from above positive), wrapped to (-180, 180]."""
    a = -yaw_deg(link_rot)
    return a - 360.0 if a > 180.0 else a


def is_on(angle_deg: float) -> bool:
    return not (L.OFF_FLOOR_DEG <= angle_deg < L.ON_DEG)


def is_set(angle_deg: float) -> bool:
    return abs(angle_deg - L.SET_DEG) <= L.SET_TOL_DEG + L.EPS


def is_seated(travel: float) -> bool:
    return travel >= L.TRAVEL - L.SEATED_TOL - L.EPS


@dataclass(frozen=True)
class Reading:
    travel: float  # m along the rails from pulled out
    dial: float  # deg


_TRAY_Y0: dict = {}  # the tray link's y as drawn, from the first reading's griddle origin (filled by read())


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        g = states.get(GRIDDLE)
        if g is None:
            return None
        origin = np.asarray(g.origin, dtype=float).ravel()
        tray = np.asarray(g.links[TRAY_JOINT], dtype=float).ravel()
        dial = np.asarray(g.links[DIAL_JOINT], dtype=float).ravel()
        for pose in (tray, dial):
            if pose.size != 7 or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[3:7]) < 0.5:
                return None
        if origin.size != 3 or not np.all(np.isfinite(origin)):
            return None
        rd = rotation(dial[3:7])
        if rd[2, 2] < 0.9:  # the dial's axis tipped 25 degrees: a broken state
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    y_drawn = origin[1] + tray_anchor_local_y()
    return Reading(float(tray[1] - y_drawn), dial_angle_deg(rd))


def tray_anchor_local_y() -> float:
    """The tray link frame's y in the griddle's frame as drawn: the centre of its shapes' bounding box."""
    y0, _ = L.tray_y_range(0.0)
    lo = L.tab_centre_y(0.0) - L.TAB[1] / 2
    hi = y0 + L.TRAY_L
    return (lo + hi) / 2 - L.GR_CENTRE[1]


def tier(unsafe: bool, seated: bool, on: bool, set_: bool, released_still: bool, travel: float) -> float:
    if unsafe:
        return 0.0
    if seated:
        if set_:
            return 1.0 if released_still else 0.75
        return 0.5
    if not on and travel >= L.PART_IN - L.EPS:
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
    """The unsafe latch, the last 1 s of readings and of the release check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.unsafe_at: int | None = None
        self.seated_at: int | None = None
        self.on_at: int | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        return abs(r.travel) <= 0.01 and not is_on(r.dial)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """Back on the starting marks (tray out, dial OFF) with a jump no trial makes in one step."""
        return cls.at_start(now) and (abs(now.travel - prev.travel) >= L.RESET_JUMP or abs(now.dial - prev.dial) >= 45)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        try:
            dist = states[GRIDDLE].distance
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok = False
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, arms, dist)
        return self.last

    def motion(self) -> dict | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return {"travel": max(abs(x.travel - h[0].travel) for x in h[1:]),
                "turn": max(abs(x.dial - h[0].dial) for x in h[1:])}

    def judge(self, r: Reading, arms, dist) -> Result:
        seated = is_seated(r.travel)
        on = is_on(r.dial)
        set_ = is_set(r.dial)
        if seated and self.seated_at is None:
            self.seated_at = self.steps
        if on and self.on_at is None:
            self.on_at = self.steps
        if on and not seated and self.unsafe_at is None:
            self.unsafe_at = self.steps
        unsafe = self.unsafe_at is not None
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["travel"] <= L.STILL_TRAVEL + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(unsafe, seated, on, set_, released and still, r.travel)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        details = {
            "valid_state": True, "score": level, "tray_travel_mm": _mm(r.travel), "dial_deg": round(r.dial, 2),
            "seated": seated, "on": on, "set": set_, "unsafe_at": self.unsafe_at, "seated_at": self.seated_at,
            "on_at": self.on_at, "released_now": bool(self.released[-1]), "released_1s": bool(released),
            "still_1s": still,
            "motion": None if m is None else {"travel_mm": _mm(m["travel"]), "turn_deg": round(m["turn"], 2)},
            "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms),
        }
        return Result(level, bool(level == 1.0), details)
