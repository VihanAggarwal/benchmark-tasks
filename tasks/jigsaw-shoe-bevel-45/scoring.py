"""The scorer, as plain functions of the saw's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the saw's mechkit AssemblyState: the body's pose, the joints ``slide`` (m), ``bevel`` (rad) and ``lever`` (rad), and
the distance from any point to the saw's shapes.

    UNLOCKED+OFF  the lever 10 deg or more out (not clamped) while the shoe is 6 mm or more forward, off the 0 degree
                  positive stop. Latched for the rest of the trial once seen.
    CLAMPED       the lever back within 10 deg of flush (the clamp rule holds the shoe: a 5 N push on its edge then
                  turns it 0.6 deg at most, against 10 deg or more unclamped)
    ANGLE         |bevel|: 45 +-2 deg ("45"); 28 deg or more ("30 or more": the 30 detent less 2 deg)
    HANDS OFF     on every step of the last 2 s both grippers' finger zones at least 50 mm from every shape of the saw
                  (unreadable arms: not off)
    STILL         over the last 2 s the bevel moved 0.5 deg or less, the lever 2 deg or less and the body 2 mm or less
    LATCHED ZERO  only clearly unsafe outcomes: the saw dropped or off the bench (any part 5 cm or more below the
                  bench top, the body 5 cm or more below where it lay, or its reference point, the shoe's pivot, past
                  the bench top's edge) or fallen over (the side that faced up turned 60 deg or more from vertical:
                  onto its handle, its shoe or an end). Sliding or turning the saw on the bench is not penalised
                  (revision 2026-11-15: the 30 mm and 15 deg latches are gone)

    1.0    HANDS OFF, STILL, ANGLE 45, CLAMPED            (success; the trial ends once it has held 1 s)
    0.75   HANDS OFF, STILL, ANGLE 45, not clamped
    0.5    HANDS OFF, STILL, ANGLE 28 deg or more, not 45
    0.25   UNLOCKED+OFF seen at some point, and none of the above
    0      otherwise, and for the rest of the trial after a LATCHED ZERO

Fail closed: a step whose saw state or arms cannot be read counts against every tier and restarts the 2 s window.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SAW = "saw"
JOINTS = ("slide", "bevel", "lever")


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    origin: np.ndarray
    rot: np.ndarray
    low_z: float
    slide_mm: float
    bevel_deg: float
    lever_deg: float


def read(states) -> Reading | None:
    """The saw's reading, or None when it is missing, malformed or not finite."""
    try:
        s = states[SAW]
        o = np.asarray(s.origin, dtype=float).ravel()
        r = np.asarray(s.rotation(), dtype=float)
        low = float(np.asarray(s.low, dtype=float).ravel()[2])
        q = {j: float(s.joints[j]) for j in JOINTS}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    if o.size != 3 or r.shape != (3, 3):
        return None
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and math.isfinite(low)
            and all(math.isfinite(v) for v in q.values())):
        return None
    return Reading(o, r, low, q["slide"] / L.MM, math.degrees(q["bevel"]), math.degrees(q["lever"]))


def clamped(r: Reading) -> bool:
    return r.lever_deg < L.LOCK_DEG - L.EPS


def off_stop_unlocked(r: Reading) -> bool:
    return (not clamped(r)) and r.slide_mm >= L.SLIDE_REL - L.EPS


def body_moved_mm(r: Reading) -> float:
    return float(np.hypot(r.origin[0] - L.ORIGIN_M[0], r.origin[1] - L.ORIGIN_M[1])) / L.MM


def body_tilt_deg(r: Reading) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r.rot[2, 2])))))


def body_drop_mm(r: Reading) -> float:
    return (L.ORIGIN_M[2] - float(r.origin[2])) / L.MM


def off_bench(r: Reading) -> bool:
    x, y = float(r.origin[0]), float(r.origin[1])
    return not (L.BENCH_X_M[0] <= x <= L.BENCH_X_M[1] and L.BENCH_Y_M[0] <= y <= L.BENCH_Y_M[1])


def wrecked(r: Reading) -> str:
    """Why the trial is latched at zero: only the saw fallen over, off the bench or dropped. Moving it is fine."""
    if body_tilt_deg(r) >= L.TIP_DEG - L.EPS:
        return "the saw fell over (turned 60 deg or more from lying on its side)"
    if off_bench(r):
        return "the saw was pushed off the bench"
    if r.low_z <= L.TOP - L.DROP_M + L.EPS:
        return "the saw was dropped (a part 5 cm or more below the bench top)"
    if body_drop_mm(r) >= L.BODY_DROP_M / L.MM - L.EPS:
        return "the saw was dropped (its body 5 cm or more below where it lay)"
    return ""


def tier(hands_off: bool, still: bool, band: str, is_clamped: bool, milestone: bool, latched: str) -> float:
    if latched:
        return 0.0
    if hands_off and still:
        if band == "45":
            return 1.0 if is_clamped else 0.75
        if band == "30+":
            return 0.5
    return 0.25 if milestone else 0.0


class Tracker:
    def __init__(self) -> None:
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.milestone = False
        self.latched = ""
        self.prev: Reading | None = None
        self.last = Result()

    @staticmethod
    def at_start(r: Reading) -> bool:
        return body_moved_mm(r) < 2.0 and abs(r.bevel_deg) < 0.5 and r.slide_mm < 0.5 and r.lever_deg < 1.0

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Put back at the start with a jump no trial makes in one step (a reset the hooks did not report)."""
        if not Tracker.at_start(now):
            return False
        jump = float(np.linalg.norm(now.origin - prev.origin)) >= L.RESET_JUMP_M
        return jump or abs(prev.bevel_deg) >= 10.0 or prev.slide_mm >= 3.0 or prev.lever_deg >= 20.0

    def _unreadable(self, why: str) -> Result:
        self.win.clear()
        self.prev = None
        self.last = Result(0.0, False, {"unreadable": why, "latched": self.latched, "milestone": self.milestone})
        if self.milestone and not self.latched:
            self.last.score = 0.25
        return self.last

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            return self._unreadable("saw state")
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        why = wrecked(r)
        if why and not self.latched:
            self.latched = why
        if off_stop_unlocked(r):
            self.milestone = True
        if not rigstate.readable(arms):
            self.win.clear()
            res = tier(False, False, "low", False, self.milestone, self.latched)
            self.last = Result(res, False, {"unreadable": "arms", "latched": self.latched,
                                            "milestone": self.milestone})
            return self.last
        try:
            dist = states[SAW].distance
            off = bool(rigstate.released(arms, dist, L.CLEAR_M))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not off
            off, gap = False, None
        self.win.append((off, r))
        full = len(self.win) == L.WIN_STEPS
        hands_off = full and all(w[0] for w in self.win)
        still = False
        motion = None
        if full:
            bev = [w[1].bevel_deg for w in self.win]
            lev = [w[1].lever_deg for w in self.win]
            o0 = self.win[0][1].origin
            body = max(float(np.linalg.norm(w[1].origin - o0)) for w in self.win) / L.MM
            motion = {"bevel_deg": round(max(bev) - min(bev), 3), "lever_deg": round(max(lev) - min(lev), 3),
                      "body_mm": round(body, 3)}
            still = (motion["bevel_deg"] <= L.STILL_BEVEL_DEG + L.EPS and motion["lever_deg"] <= L.STILL_LEVER_DEG
                     + L.EPS and motion["body_mm"] <= L.STILL_BODY_MM + L.EPS)
        band = L.bevel_band(r.bevel_deg)
        score = tier(hands_off, still, band, clamped(r), self.milestone, self.latched)
        details = {"bevel_deg": round(r.bevel_deg, 2), "slide_mm": round(r.slide_mm, 2),
                   "lever_deg": round(r.lever_deg, 2), "band": band, "clamped": clamped(r),
                   "milestone": self.milestone, "hands_off_now": off, "hands_off_2s": hands_off, "still_2s": still,
                   "motion": motion, "gap_mm": None if gap is None else round(gap / L.MM, 1),
                   "body_moved_mm": round(body_moved_mm(r), 2), "body_tilt_deg": round(body_tilt_deg(r), 2),
                   "body_drop_mm": round(body_drop_mm(r), 2),
                   "latched": self.latched}
        self.last = Result(score, score >= 1.0, details)
        return self.last
