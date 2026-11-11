"""The scorer, as plain functions of the four chamber bases' states. NumPy only; no simulator needed.

The rule is about the finished run, read from the state at the last control step. Approved Document H asks for
access at the head of the run, at each bend and at the change of pipe size, so four points need a chamber: the head
H, the size change P and the bends B1 and B2 (``layout.POINTS``; not marked on the plan). A chamber covers a point
when it is counted (upright within 10 degrees, its lowest point within 3 mm of the plan sheet, let go, and settled:
its centre moved under 2 mm in all over the last 0.5 s) and its centre is within 15 mm of the point, seen from
above. The points are at least 140 mm apart, so a chamber covers at most one point. n = the number of points
covered (0 to 4). Score = 0.25 x n, except that n = 4 scores 1.0 only once every chamber has been settled in each
of the last two 0.5 s windows (still for 1 s), and 0.75 until then.

A small tracker is fed once per control step by ``task.success()``: it keeps each chamber's recent path (for
'settled' and 'still'), what the trace needs (when each score was first reached, how many observations were broken,
on how many steps an arm could not be read), and recognises a new episode when the simulator puts every chamber
back on its starting mark in one step.

    Tracker.update(chambers, arms) -> Result(score, success, details)

``chambers`` (4, in the order of ``layout.NAMES``) are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. Each is
an upright cylinder whose axis is the body's z axis; its two ends are alike, so it may stand either way up. Its
lowest point is worked out from ``pos`` and ``quat_xyzw``. A state with a non-finite number or a zero quaternion
(a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point, 13 mm past the
TCP, midway between the finger tips) and ``opening`` (0 closed .. 1 fully open), or ``None`` when the arms could not
be read. An arm that cannot be read is skipped by the let-go rule (the other is still checked) and the step is
counted in ``arms_unavailable``. Offline, and in these functions' defaults, ``arms=None``.

The chambers are identical: any of them may be at any point.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9
SIDES = ("left", "right")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the
    world. (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN, inf or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the chamber's axis and the vertical (degrees, 0..90; either end up)."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def _extent(s) -> float:
    """Half the chamber's vertical extent, from its pose: half its height along the axis's vertical part plus its
    radius along the rest."""
    az = abs(float(rotation(s.quat_xyzw)[2, 2]))
    side = math.sqrt(max(0.0, 1.0 - az * az))
    return L.CHAMBER_H / 2 * az + L.CHAMBER_RADIUS * side


def lowest(s) -> float:
    """Height (world z) of the chamber's lowest point."""
    return float(s.pos[2]) - _extent(s)


def highest(s) -> float:
    """Height (world z) of the chamber's highest point."""
    return float(s.pos[2]) + _extent(s)


def bottom_gap(s) -> float:
    """Height of the chamber's lowest point above the plan sheet's top (m; negative = below it)."""
    return lowest(s) - L.SHEET_TOP


def standing(s) -> bool:
    """Standing on the plan: tilted at most 10 degrees and its lowest point within 3 mm of the sheet top (a chamber
    resting on a drawn line, 0.5 mm proud, stands level on it or up to about 1.5 degrees off upright)."""
    return bool(tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG + _EPS and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS)


# -- let go: the arms (rigstate) ---------------------------------------------------------------------------------
def arm_tips(arms, side: str) -> np.ndarray | None:
    """One arm's pad-tip point, or None if it cannot be read (no snapshot, the arm missing, a non-finite value)."""
    try:
        tips = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None
    return tips if tips.size == 3 and np.all(np.isfinite(tips)) else None


def readable_sides(arms) -> tuple[str, ...]:
    """The arms whose pad-tip point can be read at this step (rigstate.arms() can be None)."""
    return tuple(side for side in SIDES if arm_tips(arms, side) is not None)


def opening(arm) -> float:
    """rigstate's opening, 0 (closed) .. 1 (fully open); an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def reach(arm) -> float:
    """How far from this gripper's pad-tip point (horizontally) a chamber's axis can be while a finger touches it,
    over every yaw, plus 5 mm: 95 mm fully open, 80 mm closed on a chamber (m)."""
    return L.release_reach(opening(arm))


def tip_offset(tips, s) -> tuple[float, float]:
    """For one pad-tip point: the horizontal distance to the chamber's axis (taken at the tip's height) and the
    tip's height above the chamber's highest point (m)."""
    p = np.asarray(tips, dtype=float).ravel()
    c = np.asarray(s.pos, dtype=float)
    u = rotation(s.quat_xyzw)[:, 2]
    u = u if u[2] >= 0.0 else -u
    on_axis = c + u * ((p[2] - c[2]) / u[2]) if u[2] > 0.1 else c
    return float(np.hypot(*(p[:2] - on_axis[:2]))), float(p[2] - highest(s))


def held(s, arms) -> bool:
    """Not let go: some gripper's pad-tip point lower than 10 mm above the chamber's top with the chamber's axis
    within that gripper's reach horizontally (a finger may be touching, pressing or propping it, whatever the
    gripper's yaw). Each arm is checked on its own: an unreadable arm is skipped, the other is still checked."""
    for side in readable_sides(arms):
        xy, above = tip_offset(arm_tips(arms, side), s)
        if xy < reach(arms[side]) and above < L.RELEASE_ABOVE:
            return True
    return False


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- the access points -------------------------------------------------------------------------------------------
def off_point(s, point) -> float:
    """Horizontal distance from the chamber's centre to a point (m)."""
    return float(math.hypot(float(s.pos[0]) - point[0], float(s.pos[1]) - point[1]))


def nearest_point(s) -> tuple[str, float]:
    """The required point nearest the chamber's centre, and how far it is (m)."""
    best = min(L.POINT_NAMES, key=lambda k: off_point(s, L.POINTS[k]))
    return best, off_point(s, L.POINTS[best])


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(chambers, arms=None, settled=None, still=None) -> tuple[float, dict]:
    """The score of the run as it stands, and what was found (JSON-clean).

    ``arms``: rigstate.arms() or None. ``settled``: one bool per chamber, its centre path under 2 mm over the last
    0.5 s (None: every chamber settled, as for a state judged on its own). ``still``: every chamber settled in each
    of the last two 0.5 s windows (None: True)."""
    cs = list(chambers)
    names = list(L.NAMES[:len(cs)])
    settled = [True] * len(cs) if settled is None else list(settled)
    still = True if still is None else bool(still)
    stand = [standing(s) for s in cs]
    hold = [held(s, arms) for s in cs]
    counted = [a and not b and c for a, b, c in zip(stand, hold, settled, strict=True)]
    near = [nearest_point(s) for s in cs]
    covered: dict[str, str] = {}
    for name, ok, (point, dist) in zip(names, counted, near, strict=True):
        if ok and dist <= L.COVER_TOL + _EPS and point not in covered:
            covered[point] = name
    n = len(covered)
    score = L.POINT_SCORE * n
    if n == len(L.POINTS) and not still:
        score = 0.75
    d = {"n_covered": n, "covered": covered, "missing": [k for k in L.POINT_NAMES if k not in covered],
         "counted": [names[i] for i, ok in enumerate(counted) if ok],
         "held": [names[i] for i, h in enumerate(hold) if h],
         "not_standing": [names[i] for i, ok in enumerate(stand) if not ok],
         "not_settled": [names[i] for i, ok in enumerate(settled) if not ok],
         "all_still_1s": still,
         "nearest": {names[i]: [p, _mm(dist)] for i, (p, dist) in enumerate(near)}}
    return float(score), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What 'settled', 'still', success and the trace need about this trial (the score depends on the state at hand
    and on each chamber's path over the last second)."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.arms_unavailable = 0  # steps on which rigstate could not read one arm or both
        self.arms_unavailable_by_side = {side: 0 for side in SIDES}  # the same, per arm
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every chamber at the previous valid control step
        # every chamber's centre at the last valid steps: two 0.5 s windows
        self.history: deque = deque(maxlen=L.STILL_WINDOWS_FULL * self.window + 1)

    @staticmethod
    def at_start(chambers) -> bool:
        """The starting layout: every chamber still and upright on the sheet, on its own mark."""
        for s, name in zip(chambers, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(bottom_gap(s)) < L.SEAT_TOL
            if not (on_mark and seated and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some chamber moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def window_paths(self, k: int = 0) -> list[float] | None:
        """Each chamber's centre path (m) over the k-th 0.5 s window back (0: the last 0.5 s, 1: the 0.5 s before
        it), or None before enough valid steps have been seen."""
        h = list(self.history)
        need = (k + 1) * self.window + 1
        if len(h) < need:
            return None
        seg = h[len(h) - need:len(h) - k * self.window]
        return [float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(seg, seg[1:]))) for i in range(len(seg[0]))]

    def paths(self) -> list[float] | None:
        """Each chamber's centre path over the last 0.5 s (m), or None before 0.5 s of valid steps."""
        return self.window_paths(0)

    def settled(self) -> list[bool]:
        """Per chamber: its centre path under 2 mm over the last 0.5 s (False before 0.5 s of valid steps)."""
        p = self.paths()
        return [False] * len(L.NAMES) if p is None else [v < L.STILL_PATH for v in p]

    def still(self) -> bool:
        """Every chamber settled in each of the last two 0.5 s windows: still for 1 s."""
        for k in range(L.STILL_WINDOWS_FULL):
            p = self.window_paths(k)
            if p is None or not all(v < L.STILL_PATH for v in p):
                return False
        return True

    def update(self, chambers, arms=None) -> Result:
        cs = list(chambers)
        self.steps += 1
        if len(cs) != len(L.NAMES) or not all(valid(s) for s in cs):  # a blow-up: score 0, keep the tracker
            self.bad_steps += 1
            self.last = self.result(cs, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in cs]
        if self.prev is not None and self._teleported(now) and self.at_start(cs):
            self.reset()  # the simulator put every chamber back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        readable = readable_sides(arms)
        for side in SIDES:
            if side not in readable:
                self.arms_unavailable_by_side[side] += 1
        if len(readable) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(cs, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, chambers, arms=None) -> Result:
        cs = list(chambers)
        bad = [L.NAMES[i] for i, s in enumerate(cs) if i < len(L.NAMES) and not valid(s)]
        readable = readable_sides(arms)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_readable": list(readable),
                 "reach_mm": {s: _mm(reach(arms[s])) for s in readable},
                 "arms_unavailable": self.arms_unavailable,
                 "arms_unavailable_by_side": dict(self.arms_unavailable_by_side)}
        if bad or len(cs) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_chambers": bad, **trial})
        score, details = evaluate(cs, arms, settled=self.settled(), still=self.still())
        success = bool(score == 1.0)
        return Result(score, success, {"valid_state": True, **details, **trial})
