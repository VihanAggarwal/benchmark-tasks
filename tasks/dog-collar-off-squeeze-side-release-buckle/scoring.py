"""The scorer, as plain functions of the collar's, the male half's, the manikin's and the arms' states. NumPy only;
no simulator needed.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads three mechkit AssemblyStates: the collar (base: the female half; links w1..w13, the
webbing), the male half (base; links prong_p, prong_n) and the dog manikin (one loose body).

    LATCH A   collar off the table: a part of the collar more than 3 mm below the table top, or past its edge, at any
              step
    LATCH B   manikin dragged: its nose or rump point 100 mm or more (3-D) from where it was at the first step
    NUDGED    its nose or rump point 20 mm or more from where it was, at any step: the score is capped at 0.75
    UNLATCHED the male 3 mm or more out of the female (its slide along the female's axis), part of it still inside
    APART     every part of the male 5 mm or more from the female's outer box (it is out of the mouth, tips clear)
    OFF DOG   every part of the collar (webbing, both halves, D-ring, tri-glide) 25 mm or more from the manikin
    RESTING   no part of the collar over 40 mm above the table top, and no part moving 5 mm/s or faster
    HANDS OFF both grippers' finger zones 30 mm or more from every part of the collar (unreadable arms: not off)

    0      a latch fired, or the male under 3 mm out (still buckled)
    0.25   unlatched at the last step, not apart
    0.5    apart at the last step, some part of the collar within 25 mm of the manikin
    0.75   apart and off the dog at the last step, but resting or hands off failed on a step of the last 1 s, or the
           dog was nudged 20 mm or more at some step
    1.0    apart, off the dog, resting and hands off on every one of the last 20 steps (1 s)

Every end state falls in exactly one tier. Nothing reads the tabs or the order of moves.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

COLLAR, MALE, DOG = "collar", "buckle_male", "dog"
PARTS = (COLLAR, MALE, DOG)
SPACING = 0.006  # surface sample spacing on every collar shape


def box_surface_points(center, size, rot=None, spacing=SPACING) -> np.ndarray:
    """Points on a box's surface (its frame: ``rot`` columns), no two neighbours more than ``spacing`` apart."""
    c = np.asarray(center, dtype=float)
    h = np.asarray(size, dtype=float) / 2
    r = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
    axes = [np.linspace(-h[k], h[k], max(2, int(math.ceil(2 * h[k] / spacing)) + 1)) for k in range(3)]
    pts = []
    for k in range(3):
        a, b = [j for j in range(3) if j != k]
        for side in (-h[k], h[k]):
            for u in axes[a]:
                for v in axes[b]:
                    p = np.zeros(3)
                    p[k], p[a], p[b] = side, u, v
                    pts.append(p)
    pts = np.unique(np.round(np.array(pts), 7), axis=0)
    return c + pts @ r.T


def body_points(spec, body: str) -> np.ndarray:
    """Surface samples of one body's shapes, in the base frame at rest (what link_to_world takes)."""
    out = [box_surface_points(s.center, s.size, s.rotation()) for s in spec.body_shapes(body)]
    return np.vstack(out)


_POINTS: dict = {}


def points_of(spec, body: str) -> np.ndarray:
    key = (spec.name, body)
    if key not in _POINTS:
        _POINTS[key] = body_points(spec, body)
    return _POINTS[key]


@dataclass(frozen=True)
class Reading:
    """One step, world frame (m)."""
    f_rot: np.ndarray
    f_origin: np.ndarray
    m_origin: np.ndarray
    slide: float  # the male out of the female along her axis (0 clicked)
    male_inside: bool
    male_gap: float  # the male's nearest point to the female's outer box
    collar_pts: np.ndarray  # every collar sample point
    origins: np.ndarray  # every body's frame origin (the speed check)
    dog_gap: float
    nose: np.ndarray
    rump: np.ndarray
    collar_state: object
    male_state: object


def _female_box_gap(pts_local: np.ndarray) -> np.ndarray:
    lo = np.array([-L.F_HALF_W, L.F_Y0, 0.0])
    hi = np.array([L.F_HALF_W, L.F_Y1, L.F_H])
    c, h = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(pts_local - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        col, male, dog = states.get(COLLAR), states.get(MALE), states.get(DOG)
        if col is None or male is None or dog is None or col.spec is None or male.spec is None or dog.spec is None:
            return None
        f_rot = col.rotation()
        f_o = np.asarray(col.origin, dtype=float).reshape(3)
        m_o = np.asarray(male.origin, dtype=float).reshape(3)
        cpts = [col.link_to_world(b, points_of(col.spec, b)) for b in col.spec.body_names]
        mpts = np.vstack([male.link_to_world(b, points_of(male.spec, b)) for b in male.spec.body_names])
        allpts = np.vstack(cpts + [mpts])
        origins = np.vstack([f_o, m_o] + [np.asarray(col.links[j.name][:3], dtype=float) for j in col.spec.joints])
        local = (mpts - f_o) @ f_rot
        gaps = _female_box_gap(local)
        slide = -float(((m_o - f_o) @ f_rot)[1])
        dog_gap = float(np.min(dog.distance(allpts)))
        nose = dog.to_world(np.array(L.NOSE_PT) - np.array([L.DOG_ORIGIN[0], L.DOG_ORIGIN[1], 0.0]))
        rump = dog.to_world(np.array(L.RUMP_PT) - np.array([L.DOG_ORIGIN[0], L.DOG_ORIGIN[1], 0.0]))
        nums = [f_rot, f_o, m_o, allpts, origins, nose, rump]
        if not all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in nums) or not (
                math.isfinite(slide) and math.isfinite(dog_gap)):
            return None
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    return Reading(f_rot, f_o, m_o, slide, bool(np.any(gaps <= 0.0)), float(np.min(gaps)), allpts, origins,
                   dog_gap, nose, rump, col, male)


def collar_distance(r: Reading):
    """Distance function (world points -> m) to every collar part: the finger zones' check."""
    def distance(pts):
        return np.minimum(r.collar_state.distance(pts), r.male_state.distance(pts))
    return distance


def tier(apart: bool, unlatched: bool, off_dog: bool, window_ok: bool) -> float:
    if apart and off_dog and window_ok:
        return 1.0
    if apart and off_dog:
        return 0.75
    if apart:
        return 0.5
    if unlatched:
        return 0.25
    return 0.0


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
        self.resets = getattr(self, "resets", -1) + 1
        self.latch_a = self.latch_b = False
        self.nudged = False
        self.first: Reading | None = None
        self.prev: Reading | None = None
        self.window: deque = deque(maxlen=L.WINDOW)
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.window.append(False)
            self.prev = None
            # an unreadable step restarts the 1 s window and never holds 1.0
            self.last = Result(min(self.last.score, 0.75), False, {"bad_state": True, "step": self.steps})
            return self.last
        if self.first is None:
            self.first = r
        z = r.collar_pts[:, 2] - L.TOP
        out_x = (r.collar_pts[:, 0] < L.TABLE_X[0]) | (r.collar_pts[:, 0] > L.TABLE_X[1])
        out_y = (r.collar_pts[:, 1] < L.TABLE_Y[0]) | (r.collar_pts[:, 1] > L.TABLE_Y[1])
        if float(z.min()) < -L.BELOW_TABLE or bool(np.any(out_x | out_y)):
            self.latch_a = True
        dog_moved = max(float(np.linalg.norm(r.nose - self.first.nose)), float(np.linalg.norm(r.rump - self.first.rump)))
        if dog_moved >= L.DOG_MOVED:
            self.latch_b = True
        if dog_moved >= L.DOG_NUDGED:
            self.nudged = True
        apart = r.male_gap >= L.APART_OUT and not r.male_inside
        unlatched = r.slide >= L.UNLATCH_OUT
        off_dog = r.dog_gap >= L.OFF_DOG
        high = float(z.max())
        if self.prev is not None:
            speed = float(np.max(np.linalg.norm(r.origins - self.prev.origins, axis=1))) * L.CONTROL_HZ
        else:
            speed = math.inf
        resting = high <= L.REST_HIGH and speed < L.STILL_SPEED
        gap = rigstate.hands_gap(arms, collar_distance(r))
        hands_off = gap is not None and gap >= L.HANDS_OFF - 1e-9
        self.window.append(bool(apart and off_dog and resting and hands_off))
        window_ok = len(self.window) == L.WINDOW and all(self.window)
        latched = self.latch_a or self.latch_b
        score = 0.0 if latched else tier(apart, unlatched, off_dog, window_ok)
        if self.nudged:
            score = min(score, 0.75)
        details = {"step": self.steps, "slide_mm": round(r.slide * 1000, 2), "male_gap_mm": round(r.male_gap * 1000, 2),
                   "apart": apart, "unlatched": unlatched, "off_dog": off_dog, "dog_gap_mm": round(r.dog_gap * 1000, 1),
                   "highest_mm": round(high * 1000, 1), "lowest_mm": round(float(z.min()) * 1000, 1),
                   "off_table": bool(np.any(out_x | out_y)), "speed_mm_s": None if not math.isfinite(speed) else
                   round(speed * 1000, 1), "resting": resting, "hands_gap_mm": None if gap is None or not
                   math.isfinite(gap) else round(gap * 1000, 1), "hands_off": hands_off,
                   "latch_table": self.latch_a, "latch_dog": self.latch_b, "dog_nudged": self.nudged,
                   "dog_moved_mm": round(dog_moved * 1000, 1)}
        self.prev = r
        self.last = Result(score, score >= 1.0, details)
        return self.last
