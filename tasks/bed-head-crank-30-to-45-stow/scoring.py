"""The crank rule and the scorer, as plain functions of the bed's state and the arms' state. NumPy only.

Two pieces, both fed one observation per control step:

``Winder.update(bed)``: the worm drive. The crank's angle is integrated unwrapped (forward = raising, clamped at the
flat stop: net turns never go below 0), the deck's target angle is 5.0 degrees times the net turns, and the load the
crank meets is T_LOAD while it turns forward. In the simulator task.py runs this from a mechkit ``on_step`` rule
(before the physics) and drives the deck hinge to the target; offline it runs from ``success()``.
``crank_target`` turns the load into the crank drive's target.

``Tracker.update(states, arms, winder)`` -> Result(score, success, details), judged every step:

    0 for the rest of the trial once the deck's angle has exceeded 60 degrees (a patient would be folded and sheared);
    1.0   deck angle 30.0 to 45.0 degrees, the handle folded (fold angle 15 degrees or less), both grippers
          released the shaft, arm and knob (every finger zone 30 mm or more away) on every step of the last 1 s, and
          the deck's angle moved 0.5 degrees or less over that 1 s;
    0.5   deck angle 30.0 to 45.0 degrees, but not that whole top condition (handle left out, a gripper still on it,
          or the deck still moving);
    0.25  handle folded and the deck at 15.0 to 30.0 or 45.0 to 60.0 degrees;
    0     otherwise (doing nothing; a crank turned a little; a deck past 60 degrees at any time; a broken state).
    An unreadable arm or a distance that cannot be computed is NOT released (rigstate's fail-closed rule).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

TWO_PI = 2.0 * math.pi


def wrap_to_pi(a: float) -> float:
    return math.remainder(float(a), TWO_PI)


def _finite(*values) -> bool:
    try:
        return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)
    except (TypeError, ValueError):
        return False


def valid(s) -> bool:
    """A part state the scorer can use: present with a finite pose (a blow-up gives NaN)."""
    try:
        return _finite(s.origin, s.quat_xyzw, s.pos)
    except AttributeError:
        return False


# -- the worm drive -------------------------------------------------------------------------------------------------


class Winder:
    """Net crank turns and what they decide this step: the deck's target angle and the crank's load."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.th_prev: float | None = None
        self.th_un = 0.0  # net crank angle, rad, forward positive, never below 0 (the flat stop)
        self.target = 0.0  # the deck's target angle, rad
        self.drag = 0.0
        self.info: dict = {}
        self.steps = 0

    @property
    def turns(self) -> float:
        return self.th_un / TWO_PI

    def update(self, bed) -> dict:
        """One control step. ``bed`` carries the joints crank (and its velocity). A broken state changes nothing."""
        self.steps += 1
        try:
            th = float(bed.joints["crank"])
            w = float(bed.joint_vel.get("crank", 0.0))
            if not (_finite(th, w) and valid(bed)):
                raise ValueError("non-finite state")
        except (AttributeError, KeyError, TypeError, ValueError):
            self.drag = 0.0
            self.info = {"valid": False, "turns": round(self.turns, 3)}
            return self.info
        dth = 0.0 if self.th_prev is None else wrap_to_pi(th - self.th_prev)
        self.th_prev = th
        self.th_un = max(0.0, self.th_un + dth)
        self.target = min(L.DECK_LIMIT - 0.02, L.DECK_PER_TURN * self.turns)
        loaded = w > 0.0 and self.target < L.DECK_LIMIT - 0.02 - L.EPS
        self.drag = L.T_LOAD * min(1.0, w / L.W_SAT) if loaded else 0.0
        self.info = {"valid": True, "crank_rad": round(th, 4), "crank_w": round(w, 3), "turns": round(self.turns, 3),
                     "deck_target_deg": round(math.degrees(self.target), 3), "drag_nm": round(self.drag, 4)}
        return self.info


def crank_target(th: float, w: float, drag: float) -> float:
    """The crank drive's target (rad) for one control step: torque = CRANK_K (target - q) - CRANK_DAMP q'. Leading
    the crank by w DT / 2 makes the spring's torque average zero over the step at steady speed, the CRANK_DAMP w /
    CRANK_K term cancels the drive's damping on average, and -drag / CRANK_K leaves a mean resisting torque of
    ``drag``. With drag 0 the free crank feels only its bearing friction. Kept inside the joint's limits."""
    t = float(th) + float(w) * L.DT / 2 + (L.CRANK_DAMP * float(w) - float(drag)) / L.CRANK_K
    return max(-L.CRANK_LIMIT, min(L.CRANK_LIMIT, t))


def wrap_shift(q: float) -> float:
    """The whole turn (rad) that brings a hinge angle beyond +-pi back inside it, else 0: the rule rewrites the
    crank's and the knob's angles by it (a whole turn leaves every part in the same pose)."""
    q = float(q)
    if q > L.WRAP_AT:
        return -TWO_PI
    if q < -L.WRAP_AT:
        return TWO_PI
    return 0.0


# -- the scorer ---------------------------------------------------------------------------------------------------------


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _deg(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(math.degrees(float(v)), 3)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


def tier(angle: float, folded: bool, released_1s: bool, still: bool, over: bool) -> float:
    """The score for a deck angle (rad), the handle's state, the release and stillness checks, and the latched gate."""
    if over:
        return 0.0
    lo, hi = L.ANGLE_OK
    if lo - L.EPS <= angle <= hi + L.EPS:
        return 1.0 if (folded and released_1s and still) else 0.5
    nlo, nhi = L.ANGLE_NEAR
    if folded and nlo - L.EPS <= angle <= nhi + L.EPS:
        return 0.25
    return 0.0


class Tracker:
    """The last 1 s of the deck's angle (stillness) and of the release check, and the latched 60 degree gate."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0  # fallback resets seen (the deck fell back to flat), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.over = False  # the deck's angle has exceeded 60 degrees at some step of this trial
        self.peak = 0.0
        self.hist: deque = deque(maxlen=self.window + 1)  # deck angle per valid step
        self.released: deque = deque(maxlen=self.window)
        self.prev: float | None = None
        self.last = Result(0.0, False, {})

    def teleported(self, angle: float) -> bool:
        return self.prev is not None and self.prev - angle >= L.RESET_JUMP and angle <= L.RESET_JUMP / 4

    def update(self, states: dict, arms, winder: Winder) -> Result:
        """One control step: record the deck's angle and the release check, then judge."""
        self.steps += 1
        bed = states.get("bed")
        try:
            angle = float(bed.joints["deck"])
            fold = float(bed.joints["fold"])
            ok = valid(bed) and _finite(angle, fold)
        except (AttributeError, KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            self.bad_steps += 1  # a blow-up or a missing part: 0, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.teleported(angle):
            self.reset()
            winder.reset()  # a new trial the reset hooks missed: the net turns start again too
            self.steps = 1
            self.resets += 1
        self.hist.append(angle)
        self.prev = angle
        self.peak = max(self.peak, angle)
        if angle > L.ANGLE_GATE + L.EPS:
            self.over = True
        try:
            dist = lambda pts: bed.distance(pts, bodies=L.RELEASE_BODIES)  # noqa: E731
            dist(np.zeros((1, 3)))
        except Exception:  # noqa: BLE001 - no distance: rigstate's rule, not released
            dist = None
        self.released.append(dist is not None and rigstate.released(arms, dist, L.LET_GO))
        self.last = self.judge(bed, arms, angle, fold, dist)
        return self.last

    def spread(self) -> float | None:
        """The deck angle's range over the last 1 s (rad), or None before 1 s of valid states."""
        if len(self.hist) < self.window + 1:
            return None
        return max(self.hist) - min(self.hist)

    def judge(self, bed, arms, angle: float, fold: float, dist) -> Result:
        folded = fold <= L.FOLDED_MAX + L.EPS
        released_now = bool(self.released and self.released[-1])
        released_1s = len(self.released) == self.window and all(self.released)
        spread = self.spread()
        still = spread is not None and spread <= L.STILL_ANGLE + L.EPS
        score = tier(angle, folded, released_1s, still, self.over)
        gap = None
        try:
            gap = rigstate.hands_gap(arms, dist) if dist is not None else None
        except Exception:  # noqa: BLE001 - trace only
            gap = None
        details = {"valid_state": True, "deck_deg": _deg(angle), "peak_deg": _deg(self.peak), "fold_deg": _deg(fold),
                   "folded": bool(folded), "over_60": bool(self.over), "score": score, "released": released_now,
                   "released_1s": bool(released_1s), "spread_1s_deg": _deg(spread), "still_1s": bool(still),
                   "zone_mm": _mm(gap), "turns": round(float(angle) / L.DECK_PER_TURN, 3),
                   "arms_readable": rigstate.readable(arms)}
        return Result(score, bool(score == 1.0), details)
