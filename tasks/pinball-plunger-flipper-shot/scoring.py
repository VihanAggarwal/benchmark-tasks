"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
pinball machine (a fixed mechkit mechanism: plunger, two buttons, two flippers), the ball (a loose sphere: its position
and velocity) and both arms through rigstate. Ball positions are in playfield coordinates (layout.to_pf): u up the
slope, v across it.

    LAUNCHED    latched: the ball has left the plunger lane for the playfield
    FLIPPED     latched: after the ball has left the plunger lane, the ball moving up the slope faster than 0.3 m/s
                while a flipper is raised more than half way, having been within that flipper's reach (its length
                plus the ball's radius plus 12 mm from its pivot) in the last 0.2 s, and within 0.3 s after that
                flipper fired (rose past half way): struck by it, not rolling up from the launch and not bouncing off
                a flipper held up
    TOP         latched: after a flipper shot, the ball reached the top lanes (330 mm up the playfield)
    DRAINED     latched: the ball fell below the flippers (within 30 mm of the playfield's bottom) after launch; nothing
                after that counts
    HAND        latched: a gripper's TCP or pad-tip point over the playfield, inside the cabinet's walls and below
                their top (a hand in the game); unreadable arms are not judged here

    1.0   TOP (a flipper shot into the top lanes)                            (success)
    0.5   FLIPPED, the ball did not reach the top lanes
    0.25  LAUNCHED, never flipped
    0     not launched, and every trial with HAND
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

MACHINE, BALL = "pinball", "ball"
PLUNGER, LEFT_BUTTON, RIGHT_BUTTON = "plunger", "left_button", "right_button"
LEFT_FLIPPER, RIGHT_FLIPPER = "left_flipper", "right_flipper"


@dataclass(frozen=True)
class Reading:
    u: float
    v: float
    h: float
    vel_u: float
    flipper_up: dict  # each flipper's raise, as a fraction of its swing (0 rest .. 1 raised)


def read(states) -> Reading | None:
    try:
        m, b = states[MACHINE], states[BALL]
        p = np.asarray(b.pos, dtype=float).ravel()
        vel = np.asarray(b.lin_vel, dtype=float).ravel()
        swing = L.FLIPPER_UP - L.FLIPPER_REST
        up = {LEFT_FLIPPER: float(m.joints[LEFT_FLIPPER]) / swing, RIGHT_FLIPPER: float(m.joints[RIGHT_FLIPPER]) / swing}
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if p.size != 3 or vel.size != 3 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(vel))
                                            and all(math.isfinite(x) for x in up.values())):
        return None
    u, v, h = L.to_pf(p)
    vel_u = float(vel[0] * math.cos(L.SLOPE) + vel[2] * math.sin(L.SLOPE))
    return Reading(u, v, h, vel_u, up)


def over_playfield(p) -> bool:
    u, v, h = L.to_pf(p)
    return bool(-0.005 <= u <= L.PF_LEN and abs(v) <= L.PF_W / 2 and h < L.WALL_H + 0.06)


def near_flipper(r: Reading, name: str) -> bool:
    """The ball within reach of that flipper: its length, the ball's radius and 12 mm from its pivot."""
    v = L.FLIPPER_V if name == LEFT_FLIPPER else -L.FLIPPER_V
    return math.hypot(r.u - L.FLIPPER_U, r.v - v) <= L.FLIPPER_L + L.BALL_R + 0.012


def in_lane(r: Reading) -> bool:
    return r.v < L.LANE_V + L.LANE_W / 2 + 0.003 + L.BALL_R and r.u < L.LANE_WALL_U[1]


def tier(launched: bool, flipped: bool, top: bool, failed: bool) -> float:
    if failed or not launched:
        return 0.0
    if top:
        return 1.0
    return 0.5 if flipped else 0.25


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
        self.failed: str | None = None
        self.launched = self.flipped = self.top = self.drained = False
        self.since_zone = {LEFT_FLIPPER: 10 ** 9, RIGHT_FLIPPER: 10 ** 9}  # steps since the ball was in each one's reach
        self.since_fire = {LEFT_FLIPPER: 10 ** 9, RIGHT_FLIPPER: 10 ** 9}  # steps since each flipper rose past half way
        self.was_up = {LEFT_FLIPPER: False, RIGHT_FLIPPER: False}
        self.last = Result(0.0, False, {})

    def _hand(self, arms) -> str | None:
        if not rigstate.readable(arms):
            return None
        for side in ("left", "right"):
            for p in (np.asarray(arms[side].tcp, dtype=float), np.asarray(arms[side].tips, dtype=float)):
                if over_playfield(p):
                    return f"the {side} gripper reached onto the playfield"
        return None

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.failed is None:
            self.failed = self._hand(arms)
        if not self.drained:
            if not self.launched and not in_lane(r):
                self.launched = True
            for name in (LEFT_FLIPPER, RIGHT_FLIPPER):
                up = r.flipper_up[name] > 0.5
                self.since_fire[name] = 0 if (up and not self.was_up[name]) else self.since_fire[name] + 1
                self.was_up[name] = up
                reach = self.launched and not in_lane(r) and near_flipper(r, name)
                self.since_zone[name] = 0 if reach else self.since_zone[name] + 1
                if (up and self.since_zone[name] <= L.FLIP_MEMORY and self.since_fire[name] <= L.FIRE_MEMORY
                        and r.vel_u > L.FLIP_SPEED):
                    self.flipped = True
            if self.flipped and r.u >= L.TOP_U:
                self.top = True
            if self.launched and r.u < L.DRAIN_U and not in_lane(r):
                self.drained = True
        level = tier(self.launched, self.flipped, self.top, self.failed is not None)
        details = {
            "valid_state": True, "score": level, "ball_u_mm": round(r.u * 1000, 1), "ball_v_mm": round(r.v * 1000, 1),
            "ball_h_mm": round(r.h * 1000, 1), "vel_u": round(r.vel_u, 3),
            "flipper_up": round(max(r.flipper_up.values()), 2),
            "launched": self.launched, "flipped": self.flipped, "top": self.top, "drained": self.drained,
            "failed": self.failed, "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
