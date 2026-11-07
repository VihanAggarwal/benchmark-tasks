"""The scorer, as plain functions of the twelve plugs' states. NumPy only; no simulator needed.

The rule is about the finished bed, read from the state at the last control step: one plug planted (standing
upright on the bed's top) within 15 mm of each of the five places the instruction defines (level with the middle of
each gap in the far row, mirrored across the drip tape), 0.2 each; and the far row as the transplanter set it (a
plug standing upright on each of its six spots, centre within 10 mm), or the score is 0. The twelve plugs are
identical, so the score asks only where plugs stand, never which plug it is: what a photo of the bed shows. The bed
is fixed, so its top comes from ``layout``. A small tracker is still fed once per control step by
``task.success()``: it keeps each plug's recent path (for 'still', which success needs), what the trace needs (when
each score was first reached, how many observations were broken), and recognises a new episode when the simulator
puts every plug back on its starting mark in one step.

    Tracker.update(plugs) -> Result(score, success, details)

``plugs`` (12, in the order of ``layout.NAMES``: the six far-row plugs, then the six supply plugs) are anything with
``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as
``yam_arena.task_file.ObjectState`` has. Each is a plain upright cylinder whose size is in ``layout``; its axis is the
body's z axis, and it is the same either end up. Its lowest point is worked out from ``pos`` and ``quat_xyzw``. A
state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the
tracker as it was.

The order of ``plugs`` matters only for the trace (names) and the settle diagnostics (how far the six plugs that
started in the far row have moved); the score does not depend on it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9
N_FAR = len(L.FAR_NAMES)
START = [np.array(L.START[n], dtype=float) for n in L.NAMES]


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
    """Angle between the plug's axis and the vertical (degrees, 0..90: standing on either end 0, lying on its side
    90). The plug is a plain cylinder, the same either end up, so which end is up is not asked."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def lowest(s) -> float:
    """Height (world z) of the plug's lowest point, from its pose: a cylinder of radius r and height h whose axis
    makes angle a with the vertical reaches h/2 |cos a| + r sin a below its centre."""
    zz = abs(float(rotation(s.quat_xyzw)[2, 2]))
    return float(s.pos[2]) - (L.PLUG_H / 2 * zz + L.PLUG_R * math.sqrt(max(0.0, 1.0 - zz * zz)))


def bottom_gap(s) -> float:
    """Height of the plug's lowest point above the bed's top (m; negative = below it)."""
    return lowest(s) - L.BED_TOP


def off_table(s) -> bool:
    """The plug's lowest point is more than 10 mm below the table top: it fell off the table."""
    return bool(lowest(s) < L.TOP - L.OFF_TABLE_DROP)


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- planted, filled, far row kept --------------------------------------------------------------------------------
def over_bed(s) -> bool:
    """Its centre, seen from above, is over the bed's top (x 190 .. 410 mm, y -300 .. +300 mm)."""
    return bool(abs(float(s.pos[0]) - L.BED_X) <= L.BED_W / 2 + _EPS
                and abs(float(s.pos[1])) <= L.BED_HALF_LEN + _EPS)


def planted(s) -> bool:
    """Standing on the bed: its centre over the bed's top, its lowest point within 3 mm of that top and its axis
    within 10 degrees of vertical."""
    return bool(over_bed(s) and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS
                and tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG + _EPS)


def plan_offset(s, target) -> float:
    """Horizontal distance from the plug's centre to a target point (m)."""
    return math.hypot(float(s.pos[0]) - target[0], float(s.pos[1]) - target[1])


def far_move(i: int, s) -> float:
    """How far the plug that started at far-row spot ``i`` is from it (m, centre to centre in 3D). A settle
    diagnostic only: the score asks whether each spot still has a plug standing on it, whichever plug."""
    return float(np.linalg.norm(np.asarray(s.pos, dtype=float) - START[i]))


def far_row(plugs) -> list[str | None]:
    """For each of the six far-row spots (in the order of ``layout.FAR``), the plug planted on it (standing upright
    on the bed, its centre within 10 mm of the spot, in plan), or None when the spot is left empty."""
    names = list(L.NAMES[:len(plugs)])
    out: list[str | None] = []
    for spot in L.FAR_SPOTS:
        on = [(plan_offset(s, spot), names[i]) for i, s in enumerate(plugs)
              if planted(s) and plan_offset(s, spot) <= L.FAR_SPOT_TOL + _EPS]
        out.append(min(on)[1] if on else None)
    return out


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def fill(plugs, keepers=None) -> tuple[list[str | None], list[float | None]]:
    """For each target (in the order of ``layout.TARGETS``), the plug that fills it (planted, centre within 15 mm of
    it; each plug fills at most one target, the nearest first) or None; and the plan distance from each target to the
    nearest planted plug (m, None when there is none). The plugs standing on far-row spots (``keepers``; worked out
    when not given), at least 111 mm from every target, are left out."""
    if keepers is None:
        keepers = {b for b in far_row(plugs) if b}
    names = list(L.NAMES[:len(plugs)])
    cands = [(i, s) for i, s in enumerate(plugs) if names[i] not in keepers and planted(s)]
    pairs = sorted((plan_offset(s, t), k, i) for k, t in enumerate(L.TARGETS) for i, s in cands)
    by: list[str | None] = [None] * len(L.TARGETS)
    used: set[int] = set()
    for d, k, i in pairs:
        if d <= L.FILL_TOL + _EPS and by[k] is None and i not in used:
            by[k] = names[i]
            used.add(i)
    nearest = [min((plan_offset(s, t) for _, s in cands), default=None) for t in L.TARGETS]
    return by, nearest


def evaluate(plugs) -> tuple[float, dict]:
    """The score of the bed as it stands, and what was found (JSON-clean)."""
    ps = list(plugs)
    names = list(L.NAMES[:len(ps)])
    dropped = [names[i] for i, s in enumerate(ps) if off_table(s)]
    kept_by = far_row(ps)
    empty = [L.FAR_NAMES[k] for k, b in enumerate(kept_by) if b is None]
    by, nearest = fill(ps, keepers={b for b in kept_by if b})
    n = sum(b is not None for b in by)
    d = {"filled": n, "filled_by": by, "nearest_mm": [_mm(v) for v in nearest],
         "planted": [names[i] for i, s in enumerate(ps) if planted(s) and names[i] not in kept_by],
         "far_row_kept_by": kept_by, "far_spots_empty": empty, "off_table": dropped,
         "far_max_move_mm": _mm(max(far_move(i, s) for i, s in enumerate(ps[:N_FAR]))),
         "far_max_tilt_deg": round(max(tilt_deg(s.quat_xyzw) for s in ps[:N_FAR]), 2)}
    if dropped or empty:
        return 0.0, d
    return round(L.POINTS_EACH * n, 6), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What success and the trace need about this trial (the score itself depends only on the state at hand)."""

    TIERS = (0.2, 0.4, 0.6, 0.8, 1.0)

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
        self.prev = None  # [(centre, velocity)] of every plug at the previous valid control step
        self.history: deque = deque(maxlen=self.window + 1)  # every plug's centre at the last valid steps

    @staticmethod
    def at_start(plugs) -> bool:
        """The starting layout: every plug still and standing upright on its own mark (far row on the bed, supply
        on the trays)."""
        for s, name in zip(plugs, L.NAMES, strict=True):
            x0, y0, z0 = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(lowest(s) - (z0 - L.PLUG_H / 2)) < 0.003
            if not (on_mark and seated and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some plug moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def paths(self) -> list[float] | None:
        """Each plug's centre path (m) over the last 0.5 s, or None before 0.5 s of valid steps have been seen."""
        if len(self.history) < self.window + 1:
            return None
        h = list(self.history)
        return [float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(h, h[1:]))) for i in range(len(h[0]))]

    def update(self, plugs) -> Result:
        ps = list(plugs)
        self.steps += 1
        if len(ps) != len(L.NAMES) or not all(valid(s) for s in ps):  # a blow-up: 0 this step, keep everything
            self.bad_steps += 1
            self.last = self.result(ps)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in ps]
        if self.prev is not None and self._teleported(now) and self.at_start(ps):
            self.reset()  # the simulator put every plug back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        self.last = self.result(ps)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, plugs) -> Result:
        ps = list(plugs)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if len(ps) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_plugs": ["missing"], **trial})
        bad = [L.NAMES[i] for i, s in enumerate(ps) if not valid(s)]
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_plugs": bad, **trial})
        score, details = evaluate(ps)
        paths = self.paths()
        still = paths is not None and all(p < L.STILL_PATH for p in paths)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})
