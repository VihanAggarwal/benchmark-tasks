"""The scorer, as plain functions of the six row units' states. NumPy only; no simulator needed.

The rule is about the finished planter, read from the state at the last control step: four units set square on the
toolbar, two on each side of the red centre mark, at 95.25 and 285.75 mm from it (30 in rows at 1:4, symmetric, none
on the centre line), and the two spare units standing on their stands. Each side of the mark is scored on its own.
The toolbar and stands are fixed, so their faces come from ``layout``. A small tracker is still fed once per control
step by ``task.success()``: it keeps each unit's recent path (for 'still', which success needs), what the trace needs
(when each score was first reached, how many observations were broken), and recognises a new episode when the
simulator puts every unit back on its starting mark in one step.

    Tracker.update(units) -> Result(score, success, details)

``units`` (6, in the order of ``layout.NAMES``) are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. Each is a
plain box whose size is in ``layout``: its long side (70 mm) is the body's x axis, its height (45 mm) the body's z
axis. Its top and bottom are alike, so it may stand either way up. Its lowest point is worked out from ``pos`` and
``quat_xyzw``. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and
leaves the tracker as it was.

The units are identical: any unit may take any place. A unit belongs to the side of the mark its centre is on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9


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
    """Angle between the unit's height axis and the vertical (degrees, 0..90; either way up)."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def yaw_err_deg(quat_xyzw) -> float:
    """How far the unit's long side, seen from above, is turned from x (square to the bar): 0..90 degrees, either
    end forward."""
    r = rotation(quat_xyzw)
    if math.hypot(r[0, 0], r[1, 0]) < 1e-9:  # long side vertical: as far from square as it can be
        return 90.0
    a = math.degrees(math.atan2(r[1, 0], r[0, 0])) % 180.0
    return float(min(a, 180.0 - a))


def lowest(s) -> float:
    """Height (world z) of the unit's lowest corner, from its pose."""
    r = rotation(s.quat_xyzw)
    return float(s.pos[2]) - float(sum(abs(r[2, i]) * L.UNIT_HALF[i] for i in range(3)))


def bottom_gap(s) -> float:
    """Height of the unit's lowest point above the toolbar's top (m; negative = below it)."""
    return lowest(s) - L.BAR_TOP


def off_table(s) -> bool:
    """The unit's lowest point is more than 10 mm below the table top: it fell off the table."""
    return bool(lowest(s) < L.TOP - L.OFF_TABLE_DROP)


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- on the toolbar, set, on a stand -------------------------------------------------------------------------------
def over_bar(s) -> bool:
    """Its centre, seen from above, is over the toolbar's top (40 mm wide, y -350 .. +350 mm)."""
    return bool(abs(float(s.pos[0]) - L.BAR_X) <= L.BAR_W / 2 + _EPS and abs(float(s.pos[1])) <= L.BAR_HALF_LEN)


def on_toolbar(s) -> bool:
    """Resting on the toolbar: its centre over the toolbar's top and its lowest point within 3 mm of that top (in
    any attitude: set square, turned or tipped over)."""
    return bool(over_bar(s) and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS)


def set_on_bar(s) -> bool:
    """Set square on the bar: on the toolbar, tilted at most 5 degrees, its long side within 8 degrees of x and its
    centre within 12 mm of the bar's centre line."""
    return bool(on_toolbar(s)
                and tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG + _EPS
                and yaw_err_deg(s.quat_xyzw) <= L.SQUARE_DEG + _EPS
                and abs(float(s.pos[0]) - L.BAR_X) <= L.LINE_TOL + _EPS)


def covers_mark(s) -> bool:
    """On the toolbar with its centre within 23 mm of the red mark's centre (so it covers part of the mark): a row
    on the centre line."""
    return bool(on_toolbar(s) and abs(float(s.pos[1])) < L.MARK_COVER)


def on_stand(s, stand_y: float) -> bool:
    """Stands on the stand centred at y ``stand_y``: its centre over the stand's 90 x 90 mm top, its lowest point
    within 3 mm of that top and tilted at most 5 degrees."""
    over = (abs(float(s.pos[0]) - L.STAND_X) <= L.STAND_SIZE[0] / 2 + _EPS
            and abs(float(s.pos[1]) - stand_y) <= L.STAND_SIZE[1] / 2 + _EPS)
    return bool(over and abs(lowest(s) - L.STAND_TOP) <= L.SEAT_TOL + _EPS
                and tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG + _EPS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def side_score(units, names, sign: float, stand_y: float) -> tuple[float, dict]:
    """One side of the mark (sign +1 left, -1 right): 0.5 with exactly two units on that side of the toolbar, both
    set square, one within 10 mm of 95.25 mm from the mark and the other within 10 mm of 285.75 mm, and a unit
    standing on that side's stand; 0.25 with the same two units within 25 mm of those distances and one of the two
    0.5 conditions met (both within 10 mm with the stand empty, or the stand filled with a row 10-25 mm off); 0
    otherwise (so rows nudged part-way with the spare not stowed earn nothing)."""
    here = [i for i, s in enumerate(units) if on_toolbar(s) and sign * float(s.pos[1]) > 0.0]
    stand = any(on_stand(s, stand_y) for s in units)
    d: dict = {"on_toolbar": [names[i] for i in here], "n_on_toolbar": len(here), "stand_filled": stand,
               "all_set": None, "inner_off_mm": None, "outer_off_mm": None}
    if len(here) != 2:
        return 0.0, d
    inner, outer = sorted((units[i] for i in here), key=lambda s: abs(float(s.pos[1])))
    all_set = set_on_bar(inner) and set_on_bar(outer)
    off_in = abs(abs(float(inner.pos[1])) - L.TARGETS[0])
    off_out = abs(abs(float(outer.pos[1])) - L.TARGETS[1])
    d.update(all_set=all_set, inner_from_mark_mm=_mm(abs(float(inner.pos[1]))), inner_off_mm=_mm(off_in),
             outer_from_mark_mm=_mm(abs(float(outer.pos[1]))), outer_off_mm=_mm(off_out))
    if not all_set:
        return 0.0, d
    worst = max(off_in, off_out)
    tight = worst <= L.TIGHT + _EPS
    if tight and stand:
        return 0.5, d
    if worst <= L.LOOSE + _EPS and (tight or stand):
        return 0.25, d
    return 0.0, d


def evaluate(units) -> tuple[float, dict]:
    """The score of the planter as it stands, and what was found (JSON-clean)."""
    us = list(units)
    names = list(L.NAMES[:len(us)])
    dropped = [names[i] for i, s in enumerate(us) if off_table(s)]
    on_mark = [names[i] for i, s in enumerate(us) if covers_mark(s)]
    left, dl = side_score(us, names, +1.0, +L.STAND_Y)
    right, dr = side_score(us, names, -1.0, -L.STAND_Y)
    d = {"off_table": dropped, "covering_mark": on_mark, "left_side": left, "right_side": right,
         "left": dl, "right": dr}
    if dropped or on_mark:
        return 0.0, d
    return left + right, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What success and the trace need about this trial (the score itself depends only on the state at hand)."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every unit at the previous valid control step
        self.history: deque = deque(maxlen=self.window + 1)  # every unit's centre at the last valid steps

    @staticmethod
    def at_start(units) -> bool:
        """The starting layout: every unit still, upright and square on the toolbar's top, on its own mark."""
        for s, name in zip(units, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(bottom_gap(s)) < 0.003
            if not (on_mark and seated and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some unit moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def paths(self) -> list[float] | None:
        """Each unit's centre path (m) over the last 0.5 s, or None before 0.5 s of valid steps have been seen."""
        if len(self.history) < self.window + 1:
            return None
        h = list(self.history)
        return [float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(h, h[1:]))) for i in range(len(h[0]))]

    def update(self, units) -> Result:
        us = list(units)
        self.steps += 1
        if not all(valid(s) for s in us):  # a blow-up: score 0 for this step, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(us)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in us]
        if self.prev is not None and self._teleported(now) and self.at_start(us):
            self.reset()  # the simulator put every unit back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        self.last = self.result(us)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, units) -> Result:
        us = list(units)
        bad = [L.NAMES[i] for i, s in enumerate(us) if not valid(s)]
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_units": bad, **trial})
        score, details = evaluate(us)
        paths = self.paths()
        still = paths is not None and all(p < L.STILL_PATH for p in paths)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})
