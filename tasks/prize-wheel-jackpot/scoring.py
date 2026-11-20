"""The scorer, as plain functions of the wheel's state and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
wheel's turn (a spindle joint: unwrapped step by step) and both grippers' TCPs and pad tips.

    TOUCH      a gripper point within 40 mm in front of the wheel's face and within its radius plus 25 mm (the wheel
               or a peg); the coast is measured from the last step with a touch
    AT REST    turning slower than 0.02 rad/s for 1 s
    COAST      how far it turned, clockwise, from the last touch to where it came to rest
    SEGMENT    the segment at the pointer (12 o'clock) at rest

    1.0   at rest on JACKPOT after a coast of at least one full turn          (success, held 1 s)
    0.5   at rest on a segment next to JACKPOT after a full-turn coast
    0.25  at rest after a full-turn coast
    0     anything else (less than a full turn after the last touch: walked round, or stopped by hand)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

STAND, WHEEL = "prize_wheel", "wheel"


def turn_of(states) -> float | None:
    try:
        v = float(states[STAND].joints[WHEEL])
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    return v if math.isfinite(v) else None


def touching(p) -> bool:
    p = np.asarray(p, dtype=float)
    in_front = L.FACE_X - L.TOUCH_IN_FRONT <= p[0] <= L.CENTRE[0] + L.THICK
    radial = math.hypot(p[1] - L.CENTRE[1], p[2] - (L.TOP + L.CENTRE[2]))
    return bool(in_front and radial <= L.R + L.TOUCH_OUT)


def tier(at_rest: bool, coast: float, segment: int) -> float:
    if not at_rest or coast < L.FULL_TURN - L.EPS:
        return 0.0
    if segment == L.JACKPOT:
        return 1.0
    if segment in ((L.JACKPOT - 1) % L.SEGMENTS, (L.JACKPOT + 1) % L.SEGMENTS):
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
        self.raw: float | None = None
        self.turn = 0.0  # unwrapped, clockwise +
        self.touch_turn = 0.0  # the turn at the last touch
        self.touched_ever = False
        self.still = 0
        self.max_speed = 0.0
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        raw = turn_of(states) if isinstance(states, dict) else None
        if raw is None:
            self.bad_steps += 1
            self.last = Result(self.last.score, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        step = 0.0
        if self.raw is not None:
            step = (raw - self.raw + math.pi) % (2 * math.pi) - math.pi
        self.raw = raw
        self.turn += step
        speed = abs(step) * L.CONTROL_HZ
        self.max_speed = max(self.max_speed, speed)
        touch = False
        if rigstate.readable(arms):
            touch = any(touching(arms[s].tcp) or touching(arms[s].tips) for s in ("left", "right"))
        if touch:
            self.touch_turn, self.touched_ever = self.turn, True
        self.still = self.still + 1 if speed < L.REST_SPEED else 0
        at_rest = self.still >= L.REST_STEPS and not touch
        coast = self.turn - self.touch_turn if self.touched_ever else 0.0
        seg = L.segment_under_pointer(self.turn)
        level = tier(at_rest, coast, seg)
        details = {
            "valid_state": True, "score": level, "turn_deg": round(math.degrees(self.turn), 1),
            "speed": round(speed, 3), "touch": touch, "coast_deg": round(math.degrees(coast), 1), "at_rest": at_rest,
            "segment": seg, "max_speed": round(self.max_speed, 3), "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
