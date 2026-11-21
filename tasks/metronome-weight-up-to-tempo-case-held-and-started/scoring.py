"""The scorer, as plain functions of the metronome's state and the arms' states. NumPy only; no simulator needed.

``Tracker.update(state, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads one mechkit AssemblyState: the metronome (base: the case; links ``rod``, the pendulum on
its hinge, angle 0 upright, and ``weight``, the sliding weight on the rod, slide 0 at the 160 mark).

    TICK      the rod passes upright (its angle changes sign between two steps); its time is interpolated between
              the steps. A real metronome ticks once per swing, at the same rate.
    TIMED     the last 8 ticks, if both grippers' finger zones were 30 mm or more from every part of the metronome on
              every step from the step before the first of them to now, no two of them 1.6 s or more apart,
              every swing between two of them reaching 15 deg or more, and the last one under 1.6 s ago
    TEMPO     60 / the mean interval of the timed ticks (beats per minute)
    SET       the weight's pointer (its top front edge) within 2.5 mm of the 100 mark's height, measured along the
              rod (the pointer is 7 mm over the weight's centre, and so is every mark over its centre height)
    LATCH     the case lifted 5 mm or more off the table, tilted 15 deg or more, or its footprint moved 30 mm or more
              on the table, at any step

    1.0    timed, tempo within 4 of 100 (96 to 104), both hands still clear at this step
    0.75   timed, tempo within 10 of 100 (90 to 110), hands clear
    0.5    not as above, but the weight SET
    0.25   timed at another tempo (hands clear), weight not set
    0      none of these

Every end state falls in exactly one tier. Nothing reads the order of moves or which hand did what: a weight dragged
up without the case held lifts the light case with it (the clip's 3 N against its 1.68 N) instead of sliding, so the
weight stays at 160.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import pendulum as P
import rigstate

METRONOME = "metronome"
PARTS = (METRONOME,)


@dataclass(frozen=True)
class Reading:
    theta: float
    slide: float
    origin: np.ndarray
    tilt: float
    state: object


def read(states) -> Reading | None:
    if not isinstance(states, dict):
        return None
    try:
        m = states.get(METRONOME)
        if m is None or m.spec is None:
            return None
        th = float(m.joint("rod"))
        s = float(m.joint("weight"))
        o = np.asarray(m.origin, dtype=float).reshape(3)
        tilt = float(m.tilt())
        if not (math.isfinite(th) and math.isfinite(s) and math.isfinite(tilt) and np.all(np.isfinite(o))):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(th, s, o, tilt, m)


def tier(timed: bool, tempo: float | None, hands_off: bool, set_ok: bool) -> float:
    if timed and hands_off and tempo is not None and abs(tempo - L.GOAL_BPM) <= L.TEMPO_TOL + 1e-9:
        score = 1.0
    elif timed and hands_off and tempo is not None and abs(tempo - L.GOAL_BPM) <= L.TEMPO_NEAR + 1e-9:
        score = 0.75
    elif set_ok:
        score = 0.5
    elif timed and hands_off:
        score = 0.25
    else:
        score = 0.0
    return score


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
        self.first: Reading | None = None
        self.prev: Reading | None = None
        self.prev_off = False
        self.peak = 0.0  # the largest |angle| since the last tick
        self.touch_t = -math.inf  # the time of the last step a finger zone was within 30 mm (or unreadable)
        self.fall_t = -math.inf  # the time of the last step the case was not standing on the table
        self.ticks: deque = deque(maxlen=L.TICKS)  # (time, hands clear, peak of the swing that ended at it)
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        now = self.steps * L.DT
        r = read(states)
        if r is None:  # an unreadable step clears the timed ticks and never scores above 0.5
            self.ticks.clear()
            self.touch_t = now
            self.fall_t = now
            self.prev = None
            self.last = Result(min(self.last.score, 0.5), False, {"bad_state": True, "step": self.steps})
            return self.last
        if self.first is None:
            self.first = r
        lift = float(r.origin[2] - L.TOP)
        moved = float(np.linalg.norm(r.origin[:2] - self.first.origin[:2]))
        standing = lift <= L.STAND_LIFT + 1e-9 and r.tilt < L.STAND_TILT
        if not standing:
            self.fall_t = now
        gap = rigstate.hands_gap(arms, r.state.distance)
        hands_off = gap is not None and math.isfinite(gap) and gap >= L.HANDS_OFF - 1e-9
        if not hands_off:
            self.touch_t = now
        self.peak = max(self.peak, abs(r.theta))
        if self.prev is not None and r.theta * self.prev.theta < 0.0:
            frac = self.prev.theta / (self.prev.theta - r.theta)
            t = (self.steps - 1 + frac) * L.DT
            self.ticks.append((t, hands_off and self.prev_off, self.peak))
            self.peak = abs(r.theta)
        timed, tempo = self._timed(now)
        weight_r = P.R_START + r.slide
        pointer, mark = weight_r + L.W_SIZE[2] / 2, P.R_GOAL + L.W_SIZE[2] / 2  # both measured from the pivot
        set_ok = abs(pointer - mark) <= L.SET_TOL_MM / 1000 + 1e-9
        score = tier(timed, tempo, hands_off, set_ok)
        details = {"step": self.steps, "angle_deg": round(math.degrees(r.theta), 2),
                   "weight_mm_over_pivot": round(weight_r * 1000, 2), "set": set_ok,
                   "ticks": len(self.ticks), "timed": timed, "tempo_bpm": None if tempo is None else round(tempo, 2),
                   "hands_gap_mm": None if gap is None or not math.isfinite(gap) else round(gap * 1000, 1),
                   "hands_off": hands_off, "lift_mm": round(lift * 1000, 2), "moved_mm": round(moved * 1000, 1),
                   "tilt_deg": round(math.degrees(r.tilt), 2), "standing": standing}
        self.prev = r
        self.prev_off = hands_off
        self.last = Result(score, score >= 1.0, details)
        return self.last

    def _timed(self, now: float) -> tuple[bool, float | None]:
        if len(self.ticks) < L.TICKS:
            return False, None
        ts = [t for t, _, _ in self.ticks]
        if now - ts[-1] >= L.TICK_GAP_MAX:
            return False, None
        if not all(off for _, off, _ in self.ticks) or self.touch_t >= ts[0] - L.DT:
            return False, None  # a hand near the metronome at any step of the timed swings
        if self.fall_t >= ts[0] - L.DT:
            return False, None  # held up off the table or lying over at some step of the timed swings
        gaps = np.diff(ts)
        if np.any(gaps >= L.TICK_GAP_MAX) or np.any(gaps <= 0):
            return False, None
        # the swings between timed ticks: each one's peak is recorded at the tick that ends it
        if any(p < L.AMP_MIN for _, _, p in list(self.ticks)[1:]):
            return False, None
        return True, float(60.0 / np.mean(gaps))
