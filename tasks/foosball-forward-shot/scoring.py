"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
foosball table (a fixed mechkit mechanism: the robot's three-man forward rod slides along x and turns freely about it;
every other rod is held still), the ball (a loose sphere) and both arms through rigstate.

    GOAL        latched: the ball wholly past the right-hand goal line inside the goal mouth (layout.in_goal)
    ON_TARGET   latched: the ball reached the right-hand end wall within 100 mm of the goal mouth (a shot that missed)
    PROGRESS    the ball's furthest travel toward the right-hand goal, from where it started
    SPIN        latched: the forward rod turned more than 360 deg one way from where it started (against the rules)
    TOUCH       latched: a gripper's finger zone within 5 mm of the ball, or a TCP or pad-tip point over the playfield
                below the cabinet's top (a hand in the game, against the rules); unreadable arms are not judged here
    OWN_GOAL    latched: the ball past the left-hand goal line
    LET_GO      both grippers' finger zones 30 mm or more from the forward rod's handle on each of the last 20 steps

    1.0   GOAL and LET_GO                                                                  (success)
    0.75  GOAL, the handle still held
    0.5   ON_TARGET (no goal)
    0.25  PROGRESS of 150 mm or more (no goal, not on target)
    0     anything else, and every trial with SPIN, TOUCH or OWN_GOAL
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

TABLE, BALL = "foosball_table", "ball"
SLIDE, TURN = "rod_slide", "rod_turn"


@dataclass(frozen=True)
class Reading:
    ball: np.ndarray
    slide: float
    turn: float


def read(states) -> Reading | None:
    try:
        t, b = states[TABLE], states[BALL]
        r = Reading(np.asarray(b.pos, dtype=float).ravel(), float(t.joints[SLIDE]), float(t.joints[TURN]))
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if r.ball.size != 3 or not np.all(np.isfinite(r.ball)) or not (math.isfinite(r.slide) and math.isfinite(r.turn)):
        return None
    return r


def handle_points(slide: float) -> np.ndarray:
    """(n, 3) points along the forward rod's handle (world), with the rod slid ``slide`` along x."""
    y = L.rod_y(L.rod_index(L.SHOOTER))
    xs = np.linspace(L.NEAR_X - L.HANDLE_L, L.NEAR_X, 9) + slide
    return np.stack([xs, np.full_like(xs, y), np.full_like(xs, L.TOP + L.ROD_Z)], axis=1)


def over_field(p) -> bool:
    return bool(L.FIELD_X0 <= p[0] <= L.FIELD_X1 and L.FIELD_Y0 <= p[1] <= L.FIELD_Y1 and p[2] < L.TOP + L.CAB_H)


def on_target(p) -> bool:
    return bool(p[1] < L.goal_line_y() + L.BALL_R + 0.003
                and abs(p[0] - L.FIELD_CX) <= L.GOAL_W / 2 + L.NEAR_MISS)


def tier(goal: bool, target: bool, progress: float, let_go: bool, failed: bool) -> float:
    if failed:
        return 0.0
    if goal:
        return 1.0 if let_go else 0.75
    if target:
        return 0.5
    return 0.25 if progress >= L.PROGRESS - L.EPS else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """Only an episode reset (task.py: rigstate.on_reset and mechkit's episode counter) clears the history."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(control_hz)))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.failed: str | None = None
        self.goal = False
        self.target = False
        self.progress = 0.0
        self.turn0: float | None = None
        self.turn_prev: float | None = None
        self.turned = 0.0  # unwrapped net turn of the forward rod from its start (rad)
        self.ball0: np.ndarray | None = None
        self.ok: deque = deque(maxlen=self.window)
        self.last = Result(0.0, False, {})

    def _turn(self, q: float) -> None:
        if self.turn_prev is None:
            self.turn_prev = q
            return
        d = (q - self.turn_prev + math.pi) % (2 * math.pi) - math.pi  # the joint may be reported wrapped
        self.turned += d
        self.turn_prev = q

    def _touch(self, r: Reading, arms) -> str | None:
        if not rigstate.readable(arms):
            return None
        for side in ("left", "right"):
            for p in (np.asarray(arms[side].tcp, dtype=float), np.asarray(arms[side].tips, dtype=float)):
                if over_field(p):
                    return f"the {side} gripper reached into the playfield"
        gap = rigstate.hands_gap(arms, lambda pts: np.linalg.norm(np.asarray(pts) - r.ball, axis=1) - L.BALL_R)
        if gap is not None and gap < 0.005:
            return "a gripper touched the ball"
        return None

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.ok.clear()
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.ball0 is None:
            self.ball0 = r.ball.copy()
        self._turn(r.turn)
        if self.failed is None:
            if abs(self.turned) > L.SPIN_LIMIT:
                self.failed = "the forward rod was spun (more than a full turn)"
            elif r.ball[1] > L.FIELD_Y1 + L.BALL_R:
                self.failed = "own goal"
            else:
                self.failed = self._touch(r, arms)
        if L.in_goal(r.ball):
            self.goal = True
        if on_target(r.ball) and not self.goal:
            self.target = True
        self.progress = max(self.progress, float(self.ball0[1] - r.ball[1]))
        try:
            gap = rigstate.hands_gap(arms, handle_points(r.slide))
        except Exception:  # noqa: BLE001
            gap = None
        let_go_now = bool(rigstate.readable(arms) and gap is not None and gap >= 0.030)
        self.ok.append(let_go_now)
        let_go = len(self.ok) == self.window and all(self.ok)
        level = tier(self.goal, self.target, self.progress, let_go, self.failed is not None)
        details = {
            "valid_state": True, "score": level, "goal": self.goal, "on_target": self.target,
            "progress_mm": _mm(self.progress), "ball_mm": [round(float(v) * 1000, 1) for v in r.ball],
            "slide_mm": _mm(r.slide), "turned_deg": round(math.degrees(self.turned), 1), "handle_gap_mm": _mm(gap),
            "let_go_20": let_go, "failed": self.failed, "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
