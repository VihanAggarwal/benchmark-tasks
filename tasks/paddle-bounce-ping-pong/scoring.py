"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
paddle (one loose rigid body: its pose), the ball (a loose sphere: its position and velocity) and both arms through
rigstate. The ball is read in the paddle's frame: its height above the red face (the ball's bottom) and its distance
from the blade's centre in the face's plane.

    TOUCH       between two control steps, the ball's velocity along the face's normal (relative to the paddle) turns from
                falling to rising, its own velocity along the normal jumps up by more than 0.5 m/s (an impulse:
                gravity alone takes 0.49 m/s off it), and its bottom is within 50 mm of the face at one of the two and
                not below the face at either (a ball under the blade is not on it)
    FACE TOUCH  a touch with the ball over the blade (centre within the blade's radius) and the paddle held (its
                blade at least 15 mm above its block); any other touch (the table, the blade's edge, the handle) ends
                the run
    HAND        a gripper's TCP or pad tips within 30 mm of the ball's surface: it ends the run
    HELD        within 10 mm: a hand has the ball; the next flight is a new drop
    DROP        the flight from the last step a hand had the ball to the first face touch: if the ball's bottom was
                more than 150 mm above the face at any step of it (the let-go point included), the bounces after it
                do not count until a hand has the ball again
    RUN         starts at the first face touch after a drop from a hand (not too high); it ends at a touch anywhere
                else, a hand at the ball, or a flight outside the height band below. After a touch anywhere else or a
                hand at the ball without holding it, only a new drop from a hand starts the next run
    BOUNCE      a flight between two face touches in a run, no hand near the ball, its bottom rising 100 to 250 mm
                above the face; a flight outside that band ends the run (that touch can start the next)

    1.0   10 bounces in a row (success)
    0.5   6 in a row
    0.25  4 in a row
    0     fewer (a still paddle under the highest allowed drop or flight never gets that far)
The best run in the trial counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PADDLE, BALL = "paddle", "ball"


@dataclass(frozen=True)
class Reading:
    gap: float  # the ball's bottom above the red face (m), along its normal
    radial: float  # the ball's centre from the blade's centre, in the face's plane (m)
    vel_n: float  # the ball's velocity along the face's normal, relative to the paddle (m/s)
    own_n: float  # the ball's own velocity along the face's normal (m/s)
    ball: np.ndarray  # world
    blade_z: float  # the blade's centre above the table (m)


def read(states) -> Reading | None:
    try:
        p, b = states[PADDLE], states[BALL]
        origin = np.asarray(p.origin, dtype=float).ravel()
        quat = np.asarray(p.quat_xyzw, dtype=float).ravel()
        pv = np.asarray(p.lin_vel, dtype=float).ravel()
        c = np.asarray(b.pos, dtype=float).ravel()
        v = np.asarray(b.lin_vel, dtype=float).ravel()
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if not all(a.size == n and np.all(np.isfinite(a)) for a, n in ((origin, 3), (quat, 4), (pv, 3), (c, 3), (v, 3))):
        return None
    if not np.linalg.norm(quat) > 0.5:
        return None
    o, rot = L.paddle_frame(origin, quat / np.linalg.norm(quat))
    q = L.in_paddle(c, o, rot)
    vel_n = float(rot[:, 2] @ (v - pv))
    return Reading(float(q[2] - L.BLADE_T / 2 - L.BALL_R), float(math.hypot(q[0], q[1])), vel_n,
                   float(rot[:, 2] @ v), c, float(o[2] - L.TOP))


def hand_near(ball, arms, clear: float = L.HAND_CLEAR) -> bool:
    """A gripper's TCP or pad tips within ``clear`` of the ball's surface."""
    if not rigstate.readable(arms):
        return False
    for side in ("left", "right"):
        for pt in (arms[side].tcp, arms[side].tips):
            if np.linalg.norm(np.asarray(pt, dtype=float) - ball) < L.BALL_R + clear:
                return True
    return False


def tier(best: int) -> float:
    for n, s in L.TIERS:
        if best >= n:
            return s
    return 0.0


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
        self.prev: Reading | None = None
        self.run = 0  # bounces in the current run
        self.best = 0
        self.bounces = 0  # every bounce counted, all runs
        self.in_flight = False  # the ball left the face at the last face touch, no hand since
        self.flight_peak = 0.0
        self.dropping = False  # let go by a hand, not yet touched the face
        self.drop_peak = 0.0
        self.high_drop = False  # the last drop was from too high: nothing counts until a hand has the ball again
        self.from_hand = False  # the ball came from a hand's drop (not too high) and has touched only the face since
        self.last_touch = ""
        self.last = Result(0.0, False, {})

    def _end_run(self, lost: bool = False) -> None:
        self.run = 0
        self.in_flight = False
        if lost:  # the ball touched something else, or a hand at it: only a new drop from a hand starts a run
            self.from_hand = False

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.last = Result(tier(self.best), False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if hand_near(r.ball, arms, L.HOLD_CLEAR):  # a hand has the ball: whatever follows is a new drop
            self._end_run()
            self.dropping, self.drop_peak, self.high_drop = True, r.gap, False
            self.last_touch = "hand"
        elif hand_near(r.ball, arms):  # a hand at the ball, not holding it: the run ends, the drop goes on
            self._end_run(lost=not self.dropping)
            if self.dropping:
                self.drop_peak = max(self.drop_peak, r.gap)
            self.last_touch = "hand"
        else:
            if self.dropping:
                self.drop_peak = max(self.drop_peak, r.gap)
            if self.in_flight:
                self.flight_peak = max(self.flight_peak, r.gap)
            p = self.prev
            if (p is not None and p.vel_n < 0.0 <= r.vel_n and r.own_n - p.own_n > L.KICK
                    and min(p.gap, r.gap) <= L.TOUCH_BAND):  # a touch, between the last step and this one
                on_top = min(p.gap, r.gap) >= -0.005
                over = max(p.radial, r.radial) <= L.BLADE_R
                held = min(p.blade_z, r.blade_z) >= L.LIFTED
                if over and held and on_top:
                    self.last_touch = "face"
                    if self.dropping:
                        self.high_drop = self.drop_peak > L.MAX_DROP
                        self.from_hand = not self.high_drop
                        self.dropping = False
                        self.run = 0
                    elif self.in_flight and self.from_hand and L.MIN_RISE <= self.flight_peak <= L.MAX_RISE:
                        self.run += 1
                        self.bounces += 1
                        self.best = max(self.best, self.run)
                    else:
                        self.run = 0
                    self.in_flight, self.flight_peak = True, 0.0
                else:
                    self.last_touch = "paddle down" if over and on_top else "off the face"
                    self._end_run(lost=True)
        self.prev = r
        level = tier(self.best)
        details = {
            "valid_state": True, "score": level, "run": self.run, "best": self.best, "bounces": self.bounces,
            "gap_mm": round(r.gap * 1000, 1), "radial_mm": round(r.radial * 1000, 1), "vel_n": round(r.vel_n, 3),
            "blade_mm": round(r.blade_z * 1000, 1), "high_drop": self.high_drop, "dropping": self.dropping,
            "from_hand": self.from_hand,
            "last_touch": self.last_touch, "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
