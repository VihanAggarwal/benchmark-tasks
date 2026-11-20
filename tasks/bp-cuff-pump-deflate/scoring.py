"""The air in the cuff (a rule) and the scorer, as plain functions of the objects' states. NumPy only.

``Cuff`` is the rule that stands in for the air: fed the bulb's two joints every control step, it keeps the cuff's
pressure. A squeeze adds air in proportion to its stroke (the bulb's halves closing, past a 0.5 mm deadband), less as
the cuff fills; the valve lets air out at an orifice's rate, LEAK * f(theta) * sqrt(P), theta its opening from shut
(rad) and f linear up to 115 deg, steep past it (layout.valve_flow). Nothing else changes
the pressure. task.py shows it on the gauge's needle.

``Tracker.update(states, arms, pressure) -> Result(score, success, details)`` reads that pressure once per step:

    PEAK        the highest pressure reached
    DESCENT     from the first step the pressure is 2 mmHg below its peak (the peak at least 160; pumping higher
                starts it again) to its first fall through 60, in consecutive 1 s windows, plus one more window
                ending at the fall through 60 (so the last part-second is judged too); a window is ON RATE if the
                pressure falls 1.5 to 3.5 mmHg in it (2 to 3, with the logger's tolerance), GROSS if it falls less
                than 1 (a pause or a rise) or more than 5 (a dump)
    EMPTIED     below 20 mmHg within 10 s of passing 60

    1.0   peak 170 to 200, at least 90% of the windows on rate, none gross, emptied     (success, held 1 s)
    0.5   peak 160 to 220, at least 60% on rate, none falling more than 8 (no dump)
    0.25  peak 160 to 220
    0     anything else, including a peak over 220 (over-inflation hurts)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BULB, GAUGE = "bulb", "cuff"
SQUEEZE, VALVE, NEEDLE = "squeeze", "valve", "needle"


class Cuff:
    """The cuff's pressure (mmHg), stepped once per control step from the bulb's joints."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.p = 0.0
        self.ref: float | None = None  # the squeeze the next stroke is measured from

    @staticmethod
    def opening(valve_joint: float) -> float:
        """The valve's opening from shut (rad), from its joint (0 as it starts, a quarter turn open)."""
        return min(max(L.VALVE_START + float(valve_joint), 0.0), L.VALVE_MAX)

    @staticmethod
    def leak_rate(theta: float, p: float) -> float:
        return L.LEAK * L.valve_flow(theta) * math.sqrt(max(p, 0.0))

    def step(self, squeeze: float, valve_joint: float, dt: float = L.DT) -> float:
        s = float(squeeze)
        if self.ref is None:
            self.ref = s
        elif s > self.ref + L.PUMP_DEADBAND:  # the halves closing: air pushed in
            self.p += L.PUMP_PER_MM * (s - self.ref) * 1000.0 * max(0.0, 1.0 - self.p / L.PUMP_CEILING)
            self.ref = s
        elif s < self.ref - L.PUMP_DEADBAND:  # opening: the bulb refills from outside (its inlet valve)
            self.ref = s
        self.p = max(0.0, self.p - self.leak_rate(self.opening(valve_joint), self.p) * dt)
        return self.p


def bulb_joints(states) -> tuple[float, float] | None:
    try:
        j = states[BULB].joints
        s, v = float(j[SQUEEZE]), float(j[VALVE])
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    return (s, v) if math.isfinite(s) and math.isfinite(v) else None


def tier(peak: float, windows: list[float], emptied: bool) -> float:
    if peak > L.PEAK_LOOSE[1] or peak < L.PEAK_LOOSE[0]:
        return 0.0
    on = sum(L.RATE_ON[0] <= d <= L.RATE_ON[1] for d in windows)
    frac = on / len(windows) if windows else 0.0
    gross = any(d < L.RATE_GROSS[0] or d > L.RATE_GROSS[1] for d in windows)
    if L.PEAK_RANGE[0] <= peak <= L.PEAK_RANGE[1] and windows and frac >= 0.9 and not gross and emptied:
        return 1.0
    if windows and frac >= 0.6 and all(d <= 8.0 for d in windows):
        return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Only an episode reset (task.py: rigstate.on_reset and mechkit's episode counter) clears the history."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.peak = 0.0
        self.trace: list[float] = []  # the pressure every step
        self.start: int | None = None  # the step the pressure first fell 2 mmHg below its peak (at least 160)
        self.end: int | None = None  # ... and through 60
        self.emptied = False
        self.last = Result(0.0, False, {})

    def windows(self) -> list[float]:
        """The pressure's fall in each whole 1 s window of the descent."""
        if self.start is None:
            return []
        stop = self.end if self.end is not None else len(self.trace) - 1
        n = int(L.CONTROL_HZ)
        out = [self.trace[k] - self.trace[k + n] for k in range(self.start, stop - n + 1, n)]
        if self.end is not None and self.end - n >= self.start and (self.end - self.start) % n:
            out.append(self.trace[self.end - n] - self.trace[self.end])  # the last second before 60
        return out

    def update(self, states: dict, arms, pressure: float) -> Result:
        self.steps += 1
        if bulb_joints(states) is None or not math.isfinite(pressure):
            self.bad_steps += 1
            self.last = Result(self.last.score, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        p = float(pressure)
        prev = self.trace[-1] if self.trace else 0.0
        self.trace.append(p)
        if p > self.peak:  # over-inflation at any time counts; pumping higher starts the descent again
            self.peak = p
            if self.end is None:
                self.start = None
        if self.start is None:
            if self.peak >= L.DESCENT[0] and p <= self.peak - L.DESCENT_START_DROP:
                self.start = len(self.trace) - 1
        elif self.end is None:
            if prev >= L.DESCENT[1] > p:
                self.end = len(self.trace) - 1
        elif not self.emptied and p < L.EMPTY_BELOW and (len(self.trace) - 1 - self.end) <= L.EMPTY_WITHIN_S * L.CONTROL_HZ:
            self.emptied = True
        w = self.windows() if self.end is not None else []
        level = tier(self.peak, w, self.emptied) if self.end is not None else (
            0.25 if L.PEAK_LOOSE[0] <= self.peak <= L.PEAK_LOOSE[1] else 0.0)
        details = {
            "valid_state": True, "score": level, "p": round(p, 2), "peak": round(self.peak, 1),
            "descent": [self.start, self.end], "windows": [round(d, 2) for d in self.windows()][-60:],
            "emptied": self.emptied, "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
