"""The scorer, as plain functions of the cabinet's state and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the cabinet (``cabinet``: the back panel, side panels, both stubs, valve bodies and
clips as a fixed base, and the two levers as hinges ``cold`` and ``hot``, positive clockwise seen from above, 0 =
open along the pipe, pi/2 = the closing stop). The arms are read only for the let-go check.

    A         the cold lever's angle from open (degrees), 0 along the pipe, 90 against the stop across it
    F1 HOT    latched: the hot lever ever turned more than 10 degrees (the wrong line was touched)
    F2 MOVED  latched: the cabinet's frame moved more than 5 mm (a stub or valve torn from its wall)
    RELEASED  on every step of the last 1 s both finger zones are 30 mm or more from the cold lever and valve body
    STILL     over the last 1 s the cold lever's angle changed by under 2 degrees

    0      F1 or F2 ever fired, or not released and still (a held lever earns nothing), or A past 95 degrees
    1.0    A 88 to 90 degrees (against the stop)
    0.5    A from 45 up to 88
    0.25   A from 15 up to 45
    0      A under 15 degrees
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CABINET = "cabinet"
COLD, HOT = "cold", "hot"
ORIGIN0 = np.array([0.0, 0.0, L.TOP])  # the cabinet frame's origin at spawn (it is fixed)

# The valve body as the let-go check sees it: a box round the housing, its end sleeves, the stem and the stop plate.
_BODY_Z0, _BODY_Z1 = L.PIPE_Z - L.BODY_R, L.PLATE_Z + 0.00075
BODY_BOX = ((L.STEM_X, L.COLD_Y, L.TOP + (_BODY_Z0 + _BODY_Z1) / 2),
            (L.BODY_L, 2 * L.BODY_R, _BODY_Z1 - _BODY_Z0))


@dataclass(frozen=True)
class Reading:
    """One step: the two levers' angles (rad, clockwise from open) and the frame's origin (world)."""
    cold: float
    hot: float
    origin: np.ndarray


def read(states) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        s = states.get(CABINET)
        if s is None:
            return None
        cold, hot = float(s.joints[COLD]), float(s.joints[HOT])
        o = np.asarray(s.origin, dtype=float).ravel()
        if o.size != 3 or not (math.isfinite(cold) and math.isfinite(hot) and np.all(np.isfinite(o))):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(cold, hot, o)


def angle_deg(r: Reading) -> float:
    return math.degrees(r.cold)


def hot_deg(r: Reading) -> float:
    return math.degrees(r.hot)


def moved_mm(r: Reading) -> float:
    return float(np.linalg.norm(r.origin - ORIGIN0)) * 1000.0


def tier(latched: bool, settled: bool, a_deg: float) -> float:
    """The score for the cold lever at ``a_deg`` degrees: 0 when latched or not let go and still, or past the stop."""
    if latched or not settled or not (a_deg <= L.OVER_DEG + L.EPS):
        return 0.0
    if a_deg >= L.FULL_DEG - L.EPS:
        return 1.0
    if a_deg >= L.HALF_DEG - L.EPS:
        return 0.5
    if a_deg >= L.QUARTER_DEG - L.EPS:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _r(v, n=3):
    return None if v is None or not math.isfinite(v) else round(float(v), n)


class Tracker:
    """The latches (F1, F2), the last 1 s of the cold angle and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "F2": None}
        self.hist: deque = deque(maxlen=self.window + 1)  # the cold angle (degrees), the last 1 s
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Both levers back at open with a jump no trial makes in one step: a reset, not the arm."""
        return (math.degrees(prev.cold) - math.degrees(now.cold) >= L.RESET_DEG
                and math.degrees(now.cold) < 1.0 and math.degrees(now.hot) < 1.0)

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
        self.hist.append(angle_deg(r))
        self.prev = r
        if self.latch["F1"] is None and hot_deg(r) > L.HOT_DEG + L.EPS:
            self.latch["F1"] = self.steps
        if self.latch["F2"] is None and moved_mm(r) > L.MOVED_MM + L.EPS:
            self.latch["F2"] = self.steps
        ok, gap = False, None
        try:
            lever = states[CABINET].distance
            body = rigstate.boxes_distance([BODY_BOX])

            def dist(points):
                return np.minimum(lever(points, bodies=(COLD,)), body(points))

            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not released
            ok, gap = False, None
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def swing(self) -> float | None:
        """Over the last 1 s: the largest change of the cold angle (degrees) from the window's start; None before 1 s
        of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(abs(a - h[0]) for a in h[1:])

    def judge(self, r: Reading, gap) -> Result:
        a = angle_deg(r)
        released = len(self.released) == self.window and all(self.released)
        sw = self.swing()
        still = bool(sw is not None and sw < L.STILL_DEG - L.EPS)
        level = tier(self.latched, released and still, a)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "cold_deg": _r(a, 2), "hot_deg": _r(hot_deg(r), 2), "moved_mm": _r(moved_mm(r), 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "swing_1s_deg": None if sw is None else _r(sw, 3),
            "hands_gap_mm": None if gap is None else _r(gap * 1000.0, 2),
        }
        return Result(level, bool(level == 1.0), details)
