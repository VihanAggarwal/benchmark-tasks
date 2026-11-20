"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
puck and the mallet (loose bodies: their centres and the puck's velocity) and both grippers' TCPs and pad tips.

    HIT       the puck's velocity along the table turns from toward the robot to away from it (at 0.05 m/s or more),
              between two control
              steps, with the mallet's rim within 30 mm of the puck's (at 20 Hz) and the mallet moving at 0.2 m/s or more over the 0.1 s up to it (a
              strike: a mallet held still in the puck's way is not a hit), the mallet held (a gripper's TCP within 20 mm
              of its axis at the handle's height, the fingers closed past half open) and the puck's edge at least 10 mm
              from the near end rail (not a rebound off the rail)
    RETURNED  after a hit, the puck crosses the halfway line moving away
    WON       after a hit, the puck goes into the far goal (past the far end within the goal's 90 mm)
    CONCEDED  the puck goes into the robot's own goal
    HAND      a gripper's TCP or pad tips within 15 mm of the puck's rim (and 30 mm of its height), at any time
    OVER      any part of the mallet past the halfway line, at any time (the game's rule: each player keeps to their half)

    1.0   WON                                  (success, held 1 s)
    0.5   RETURNED
    0.25  HIT
    0     anything else, and every trial with HAND, OVER or CONCEDED
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PUCK, MALLET = "puck", "mallet"


@dataclass(frozen=True)
class Reading:
    puck: np.ndarray  # centre (world)
    vel: np.ndarray  # the puck's velocity (world)
    mallet: np.ndarray  # the mallet disc's centre (world)


def read(states) -> Reading | None:
    try:
        p = np.asarray(states[PUCK].origin, dtype=float).ravel()
        v = np.asarray(states[PUCK].lin_vel, dtype=float).ravel()
        m = np.asarray(states[MALLET].origin, dtype=float).ravel()
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if not all(a.size == 3 and np.all(np.isfinite(a)) for a in (p, v, m)):
        return None
    return Reading(p, v, m)


def held(arms, mallet: np.ndarray) -> bool:
    """A gripper closed on the mallet's handle."""
    if not rigstate.readable(arms):
        return False
    for s in ("left", "right"):
        tcp = np.asarray(arms[s].tcp, dtype=float)
        dz = float(tcp[2] - mallet[2])
        if math.dist(tcp[:2], mallet[:2]) <= L.HELD_R and L.HELD_DZ[0] <= dz <= L.HELD_DZ[1] and \
                float(arms[s].opening) < L.HELD_OPENING:
            return True
    return False


def tier(won: bool, returned: bool, hit: bool, foul: bool) -> float:
    if foul:
        return 0.0
    if won:
        return 1.0
    if returned:
        return 0.5
    return 0.25 if hit else 0.0


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
        self.mallets: list[np.ndarray] = []  # the mallet's last three positions
        self.hit = self.returned = self.won = self.conceded = self.hand = self.over = False
        self.hit_step: int | None = None
        self.hit_swing: float | None = None
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.last = Result(self.last.score, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if rigstate.readable(arms):
            for s in ("left", "right"):
                for pt in (arms[s].tcp, arms[s].tips):
                    if math.dist(np.asarray(pt, dtype=float)[:2], r.puck[:2]) < L.PUCK_R + L.HAND_CLEAR and \
                            abs(float(np.asarray(pt, dtype=float)[2]) - r.puck[2]) < 0.03:
                        self.hand = True
        self.mallets = (self.mallets + [r.mallet])[-3:]
        if r.mallet[0] + L.MALLET_R > L.CENTRE_X:
            self.over = True
        p = self.prev
        if p is not None and p.vel[0] < 0.0 and r.vel[0] >= L.HIT_SPEED:
            gap = min(math.dist(p.puck[:2], p.mallet[:2]), math.dist(r.puck[:2], r.mallet[:2])) - L.PUCK_R - L.MALLET_R
            swing = math.dist(self.mallets[0][:2], r.mallet[:2]) / (L.DT * (len(self.mallets) - 1))
            off_rail = min(p.puck[0], r.puck[0]) - L.PUCK_R - L.X0 >= L.RAIL_CLEAR
            if gap <= L.HIT_GAP and swing >= L.STRIKE_SPEED and off_rail and held(arms, r.mallet):
                self.hit, self.hit_step, self.hit_swing = True, self.steps, swing
        if self.hit and p is not None and p.puck[0] < L.CENTRE_X <= r.puck[0] and r.vel[0] > 0:
            self.returned = True
        if self.hit and L.in_far_goal(r.puck):
            self.won = True
        if L.in_own_goal(r.puck):
            self.conceded = True
        self.prev = r
        level = tier(self.won, self.returned, self.hit, self.hand or self.over or self.conceded)
        details = {
            "valid_state": True, "score": level, "puck_mm": [round(v * 1000, 1) for v in r.puck[:2]],
            "puck_v": [round(v, 3) for v in r.vel[:2]], "mallet_mm": [round(v * 1000, 1) for v in r.mallet[:2]],
            "mallet_lift_mm": round((r.mallet[2] - L.MALLET_H / 2 - L.TOP - L.SURFACE) * 1000, 1),
            "hit": self.hit, "returned": self.returned, "won": self.won, "conceded": self.conceded, "hand": self.hand,
            "over": self.over, "hit_step": self.hit_step,
            "hit_swing": None if self.hit_swing is None else round(self.hit_swing, 3),
            "arms_readable": rigstate.readable(arms), "held": held(arms, r.mallet),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
