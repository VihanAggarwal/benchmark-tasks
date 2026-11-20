"""The scorer, as plain functions of the robot's and the protector's states and the arms. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads two mechkit AssemblyStates: ``robot`` (its slide joint ``pull``: how far the hose is
out of the retraction unit, m) and ``protector`` (its hinge joint ``lid``: 0 open, -LID_OPEN shut).

    PULL      q = the hose's pull, metres out of the unit; the mark is two links, 27.8 mm (TRE.40: 13.9 mm a link)
    SHUT      the lid within 2.5 deg of shut (a lid pushed down over a crest instead of a waist stops about 7 deg
              open, its rib on the crest)
    NEAR      |q - 27.8 mm| <= 7 mm (half a link)
    CLOSE     |q - 27.8 mm| <= 5 mm
    HANDS OFF both grippers' finger zones 40 mm or more from the hose and from the protector (unreadable arms: not
              off)
    STILL     the hose moving under 2 mm/s along its slide

    0      never NEAR
    0.25   NEAR on some step
    0.5    SHUT while NEAR on some step
    0.75   now SHUT and CLOSE
    1.0    SHUT, CLOSE, hands off and still on every one of the last 20 steps (1 s)

Every state falls in exactly one tier (the highest whose condition holds). Nothing reads which arm did what. The
pull left once the protector rests on the exit is the pretension: shut on the waist that meets the rib at 33.8 mm,
the hose runs back 6 mm (the rib's 2 mm play, the 4 mm gap to the exit) to 27.8 mm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

ROBOT, RING = "robot", "protector"
PARTS = (ROBOT, RING)


@dataclass(frozen=True)
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def pull(robot) -> float:
    try:
        return float(robot.joints["pull"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0.0


def lid_open(ring) -> float:
    """How far the lid is open from shut (rad, 0 shut)."""
    try:
        return float(ring.joints["lid"]) + L.LID_OPEN
    except (AttributeError, KeyError, TypeError, ValueError):
        return L.LID_OPEN


def parts_distance(robot, ring):
    """Distance from world points to the hose or the protector (its shell and lid)."""

    def dist(q):
        return np.minimum(robot.distance(q, bodies=("pull",)), ring.distance(q))

    return dist


class Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.ever_near = False
        self.ever_shut_near = False
        self.prev_q = None
        self.window = deque(maxlen=L.WINDOW)
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        robot, ring = states[ROBOT], states[RING]
        q = pull(robot)
        phi = lid_open(ring)
        err = abs(q - L.MARK)
        near, close, shut = err <= L.NEAR, err <= L.CLOSE, phi <= L.SHUT
        self.ever_near |= near
        self.ever_shut_near |= shut and near
        try:
            vel = abs(float(robot.joint_vel["pull"]))
        except (AttributeError, KeyError, TypeError, ValueError):
            vel = 0.0
        if self.prev_q is not None and getattr(robot, "source", "sim") != "sim":
            vel = abs(q - self.prev_q) * L.CONTROL_HZ
        self.prev_q = q
        still = vel < L.STILL
        try:
            off = bool(rigstate.readable(arms) and rigstate.released(arms, parts_distance(robot, ring), L.RELEASE))
        except Exception:  # noqa: BLE001  (fail closed: an unreadable step is not hands off)
            off = False
        self.window.append(shut and close and off and still)
        if len(self.window) == L.WINDOW and all(self.window):
            score = 1.0
        elif shut and close:
            score = 0.75
        elif self.ever_shut_near:
            score = 0.5
        elif self.ever_near:
            score = 0.25
        else:
            score = 0.0
        details = {"pull_mm": round(q * 1000, 1), "lid_deg": round(math.degrees(phi), 1), "near": near,
                   "close": close, "shut": shut, "hands_off": off, "still": still}
        self.last = Result(score, score >= 1.0, details)
        return self.last
