"""The scorer, as plain functions of the tumbler's, the holder's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads the mechkit AssemblyStates of the tumbler (base frame at the centre of its underside,
z up the axis, mouth up) and of the ring holder (fixed, base frame at the ring's centre on the table top). Every
reading is a pose; the arms are read only for the let-go check.

    HUNG        the tumbler's base (centre of its underside) 32 to 48 mm over the table: it hangs in the ring by its
                taper (the seat is 40 mm; standing on the table is 0)
    TILT        the angle between the tumbler's axis and the vertical
    OFF-CENTRE  the distance from the tumbler's axis to the ring's centre, measured at the ring's top face height
    CAUGHT      the base at least 20 mm over the table, within 60 mm of the ring's axis (top view), mouth up (tilt
                under 60 degrees): the tumbler is on or in the ring, not on the table
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the tumbler and the holder
    STILL       over the last 1 s the tumbler moved under 1 mm and turned under 0.5 degrees
    F1 DROPPED  latched, an impact: the tumbler's base came down 20 mm or more within 3 control steps (0.15 s: a
                free fall of 20 mm takes 0.064 s), still coming down 6.5 mm or more (130 mm/s) in the last of them,
                then stopped dead (under 2 mm further down the next step) on the table or the ring (its lowest point
                within 5 mm of the table or below it; or within 71 mm of the ring's axis and no higher than 5 mm over
                the ring's top). A free fall onto the ring and a hand banging it down are both impacts; a fast hand
                that stops in free air is not.
    F2 HOLDER   latched: the holder moved more than 5 mm

    0      F1 or F2 ever fired, or not released and still (a held tumbler earns nothing)
    1.0    hung, tilt <= 6 degrees, off-centre <= 6 mm
    0.5    hung, tilt <= 15 degrees, off-centre <= 12 mm (otherwise)
    0.25   caught on or in the ring, otherwise
    0      anything else (on the table, upside down, lying, far from the ring)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

TUMBLER, HOLDER = "tumbler", "holder"
PARTS = (TUMBLER, HOLDER)
RING_XY = np.array([L.RING_X, L.RING_Y])
START = np.array([L.TUM_START[0], L.TUM_START[1], 0.0])
HOLDER_ORIGIN = np.array([L.RING_X, L.RING_Y, 0.0])


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
    pos: np.ndarray  # the tumbler's base frame origin: the centre of its underside
    rot: np.ndarray  # the tumbler's rotation
    holder: np.ndarray  # the holder's base frame origin


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        t, h = states.get(TUMBLER), states.get(HOLDER)
        if t is None or h is None:
            return None
        o = np.asarray(t.origin, dtype=float).ravel()
        r = rotation(t.quat_xyzw)
        ho = np.asarray(h.origin, dtype=float).ravel()
        if o.size != 3 or ho.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))
                                                and np.all(np.isfinite(ho))):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    shift = np.array([0.0, 0.0, L.TOP])
    return Reading(o - shift, r, ho - shift)


# -- the readings ------------------------------------------------------------------------------------------------------
def axis(r: Reading) -> np.ndarray:
    """The tumbler's axis, base toward the mouth."""
    return r.rot[:, 2]


def tilt_deg(r: Reading) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(axis(r)[2])))))


def base_height(r: Reading) -> float:
    return float(r.pos[2])


def horiz(r: Reading) -> float:
    """The base's distance from the ring's axis (top view)."""
    return float(np.linalg.norm(r.pos[:2] - RING_XY))


def off_centre(r: Reading) -> float:
    """Distance (m) from the tumbler's axis to the ring's centre line, taken at the ring's top face height; inf when the
    axis is not rising toward the mouth (lying or upside down)."""
    a = axis(r)
    if a[2] < 0.05:
        return math.inf
    s = (L.RING_H - float(r.pos[2])) / float(a[2])
    p = r.pos[:2] + s * a[:2]
    return float(np.linalg.norm(p - RING_XY))


def lowest(r: Reading) -> float:
    """The height (m over the table) of the tumbler's lowest point: the lower of its base circle (30 mm) and its rim
    circle (38 mm, 100 mm up its axis)."""
    a = axis(r)
    s = math.sqrt(max(0.0, 1.0 - float(a[2]) ** 2))
    base = float(r.pos[2]) - L.TUM_R_BASE * s
    rim = float(r.pos[2] + L.TUM_H * a[2]) - (L.TUM_R_BASE + L.TUM_TAN * L.TUM_H) * s
    return min(base, rim)


def supported(r: Reading) -> bool:
    """On the table (or off it, below it) or on or in the ring: where a fast fall that stops dead is an impact."""
    low = lowest(r)
    if low <= L.SUPPORT_TOL + L.EPS:
        return True
    return bool(horiz(r) <= L.ON_RING_NEAR + L.EPS and low <= L.RING_H + L.SUPPORT_TOL + L.EPS)


def impact(h: list) -> float:
    """The fall (m) that has just ended in an impact, judged on the latest reading h[-1]: the base came down over up
    to FALL_STEPS steps to h[-2], at least IMPACT_STEP in the last of them, then moved down under STOP_MOVE to h[-1],
    which is supported. 0 when there is no impact."""
    if len(h) < 3:
        return 0.0
    now, end = h[-1], h[-2]
    if end.pos[2] - now.pos[2] >= L.STOP_MOVE - L.EPS or h[-3].pos[2] - end.pos[2] < L.IMPACT_STEP - L.EPS:
        return 0.0
    if not supported(now):
        return 0.0
    return max(h[-2 - j].pos[2] - end.pos[2] for j in range(1, min(L.FALL_STEPS, len(h) - 2) + 1))


def hung(r: Reading) -> bool:
    return bool(abs(base_height(r) - L.H_SEAT) <= L.HUNG_TOL + L.EPS and tilt_deg(r) < L.CAUGHT_TILT_DEG)


def caught(r: Reading) -> bool:
    return bool(base_height(r) >= L.CAUGHT_BASE_MIN - L.EPS and horiz(r) <= L.CAUGHT_NEAR + L.EPS
                and tilt_deg(r) < L.CAUGHT_TILT_DEG)


def tier(latched: bool, settled: bool, r: Reading) -> float:
    if latched or not settled:
        return 0.0
    if hung(r):
        tilt, off = tilt_deg(r), off_centre(r)
        if tilt <= L.TILT_FULL_DEG + L.EPS and off <= L.OFF_FULL + L.EPS:
            return 1.0
        if tilt <= L.TILT_HALF_DEG + L.EPS and off <= L.OFF_HALF + L.EPS:
            return 0.5
    if caught(r):
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


class Tracker:
    """The latches (F1, F2), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "F2": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.max_drop = 0.0  # the largest drop of the base within FALL_STEPS control steps so far
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        return bool(float(np.linalg.norm(r.pos - START)) <= L.START_TOL and tilt_deg(r) < 3.0)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The tumbler put back on its starting mark with a jump no trial makes in one step (a reset)."""
        return bool(cls.at_start(now) and float(np.linalg.norm(now.pos - prev.pos)) >= L.RESET_JUMP)

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
        try:
            both = [states[TUMBLER].distance, states[HOLDER].distance]

            def dist(points):
                return np.min([d(points) for d in both], axis=0)
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
        # F1: an impact. The base came down 20 mm or more within 3 control steps and stopped dead on the table or the
        # ring (a free fall, or a hand that bangs it down). max_drop is the largest 3-step fall, for the trace.
        h = list(self.hist)
        drop = max((h[-1 - j].pos[2] - r.pos[2] for j in range(1, min(L.FALL_STEPS, len(h) - 1) + 1)), default=0.0)
        self.max_drop = max(self.max_drop, float(drop))
        if impact(h) >= L.FALL_DROP - L.EPS and self.latch["F1"] is None:
            self.latch["F1"] = self.steps
        if float(np.linalg.norm(r.holder - HOLDER_ORIGIN)) > L.HOLDER_MOVE and self.latch["F2"] is None:
            self.latch["F2"] = self.steps
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the tumbler from where it was at the window's
        start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.pos - first.pos)))
            turn = max(turn, turn_deg(first.rot, x.rot))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, r)
        off = off_centre(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "base_mm": _mm(base_height(r)), "tilt_deg": round(tilt_deg(r), 2), "off_centre_mm": _mm(off),
            "horiz_mm": _mm(horiz(r)), "hung": hung(r), "caught": caught(r),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap), "max_drop_mm": _mm(self.max_drop),
            "tumbler_mm": _mmv(r.pos),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
