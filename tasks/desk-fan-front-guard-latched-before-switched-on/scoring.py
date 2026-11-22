"""The scorer, as plain functions of the fan's and the arms' state. NumPy only.

``Tracker.update(states, arms, latched_guard)`` -> Result(score, success, details), one observation per control step.
It reads the fan's mechkit AssemblyState (joints ``guard``, ``clip``, ``rocker``) and the latch rule's state
(latchrule.py).

    OPEN DEG       22 deg minus the guard's joint (0 = flush closed, 22 = ajar)
    CLIP DEG       degrees(clip joint) (0 = open +12 deg, 24 = closed -12 deg)
    ROCKER TILT    12 deg minus rocker joint (+12 deg = OFF, -12 deg = ON)
    ON / OFF       rocker joint >= 20 deg (tilt <= -8 deg) reads ON; <= 8 deg (tilt >= +4 deg) reads OFF
                   (hysteresis in between); trial starts OFF
    SWITCH-ON      a step on which the reading turns from OFF to ON (every one is checked)
    SHUT           front guard within 2 deg of flush closed (open_deg <= 2 deg)
    LATCHED        top retaining clip closed (>= 15 deg, open again at <= 9 deg) and guard shut (<= 2 deg), from latchrule.Latch
    CAVITY BREACH  any finger zone of either gripper inside the impeller cavity box while the front guard is open
                   (not SHUT)
    LATCHED 0      SWITCH-ON with front guard not LATCHED (or unreadable arms), OR guard >= 5 deg open on a step
                   that reads ON, OR a CAVITY BREACH while the guard is open
    CLEAR          over the last 1.0 s every finger zone of both grippers at least 5 cm from the fan

    1.0   front guard shut and LATCHED, ON (gate passed), CLEAR for 1.0 s (success)
    0.75  front guard shut and LATCHED, ON (gate passed), not yet CLEAR for 1.0 s
    0.5   front guard shut and LATCHED, OFF
    0.25  front guard SHUT (<= 2 deg) but not latched, OFF
    0.0   anything else (guard ajar), and for the rest of the trial after a LATCHED 0

Fail closed: a step whose fan state cannot be read scores 0 and restarts the clear window; unreadable arms are never
CLEAR and fail a SWITCH-ON gate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import latchrule
import layout as L
import rigstate

FAN = "desk_fan"


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    open_deg: float
    clip_deg: float
    rocker_deg: float


def read(states) -> Reading | None:
    try:
        f = states[FAN]
        guard = float(f.joints["guard"])
        clip = float(f.joints["clip"])
        rocker = float(f.joints["rocker"])
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(guard) and math.isfinite(clip) and math.isfinite(rocker)):
        return None
    return Reading(L.GUARD_OPEN_DEG - math.degrees(guard), latchrule.clip_closed_deg(clip), math.degrees(rocker))


def shut(r: Reading) -> bool:
    return r.open_deg <= L.SHUT_DEG + L.EPS


CAVITY_ZONE = rigstate.boxes_distance([L.cavity_box()])


def fingers_in_cavity(arms) -> bool | None:
    """True if a finger zone enters the impeller cavity; None if arms are unreadable."""
    if rigstate.read_arms(arms) is None:
        return None
    return rigstate.keep_out_breached(arms, CAVITY_ZONE, margin=0.0)


def tier(is_shut: bool, latched_guard: bool, on: bool, clear: bool, zero: str) -> float:
    if zero:
        return 0.0
    if latched_guard:
        if on:
            return 1.0 if clear else 0.75
        return 0.5
    if is_shut and not on:
        return 0.25
    return 0.0


class Tracker:
    def __init__(self) -> None:
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.on = False
        self.switch_ons: list = []
        self.zero = ""
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.prev: Reading | None = None
        self.last = Result()

    @staticmethod
    def at_start(r: Reading) -> bool:
        return r.open_deg > L.GUARD_OPEN_DEG - 1.0 and r.clip_deg < 1.0 and r.rocker_deg < 1.0

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Put back at the start with a jump no trial makes in one step (a reset the hooks did not report)."""
        if not Tracker.at_start(now):
            return False
        return (prev.open_deg < L.GUARD_OPEN_DEG - 10.0 or prev.clip_deg > L.CLIP_CENTRE_DEG
                or prev.rocker_deg > L.ROCKER_CENTRE_DEG)

    def _zero(self, why: str) -> None:
        if not self.zero:
            self.zero = why

    def update(self, states, arms, latched_guard: bool) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.win.clear()
            self.prev = None
            self.last = Result(0.0, False, {"unreadable": "desk_fan", "zero": self.zero})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r

        was_on = self.on
        if r.rocker_deg >= L.ROCKER_ON_DEG - L.EPS:
            self.on = True
        elif r.rocker_deg <= L.ROCKER_OFF_DEG + L.EPS:
            self.on = False

        is_shut = shut(r)
        latched_now = bool(latched_guard) and is_shut
        in_cavity = fingers_in_cavity(arms)
        if not is_shut and in_cavity is True:
            self._zero("finger entered exposed impeller cavity while guard was open")

        gate = None
        if self.on and not was_on:
            arms_ok = in_cavity is not None
            gate = {"step": self.steps, "shut": is_shut, "latched": latched_now,
                    "fingers_in_cavity": in_cavity if arms_ok else True}
            self.switch_ons.append(gate)
            if not latched_now:
                self._zero("switched on with the front guard not latched")
            elif not arms_ok:
                self._zero("switched on with unreadable arms")
            elif in_cavity:
                self._zero("switched on with a finger in the impeller cavity")

        if self.on and r.open_deg >= L.OPEN_LIVE_DEG - L.EPS:
            self._zero("the front guard opened while the fan was on")

        try:
            dist = states[FAN].distance
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not clear
            gap = None
        clear_now = gap is not None and gap >= L.CLEAR_M - 1e-9
        self.win.append(clear_now)
        clear = len(self.win) == L.WIN_STEPS and all(self.win)

        score = tier(is_shut, latched_now, self.on, clear, self.zero)
        details = {
            "open_deg": round(r.open_deg, 2),
            "clip_deg": round(r.clip_deg, 2),
            "rocker_deg": round(r.rocker_deg, 2),
            "rocker_tilt_deg": round(L.rocker_tilt_deg(r.rocker_deg), 2),
            "on": self.on,
            "guard_shut": is_shut,
            "guard_latched": latched_now,
            "fingers_in_cavity": bool(in_cavity) if in_cavity is not None else None,
            "clear_1s": clear,
            "gap_mm": None if gap is None else round(gap * 1000.0, 1),
            "zero": self.zero,
            "switch_ons": len(self.switch_ons),
        }
        if gate is not None:
            details["gate"] = gate
        self.last = Result(score, score >= 1.0, details)
        return self.last
