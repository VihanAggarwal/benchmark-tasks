"""The scorer, as plain functions of the six balusters' states. NumPy only; no simulator needed.

The rule is about the finished guard, read from the state at the last control step: the balusters standing on the
shoe rail between the two newel posts must leave no opening that the 51 mm test ball can pass (the end openings,
post to baluster, included), and the openings should be even. How many balusters that takes is part of the task:
four can never do it, five or six can. The rail and posts are fixed, so their faces come from ``layout``. A small
tracker is still fed once per control step by ``task.success()``: it keeps what the trace needs (when each score was
first reached, how many observations were broken) and recognises a new episode when the simulator puts every
baluster back on its starting mark in one step.

    Tracker.update(balusters) -> Result(score, success, details)

``balusters`` (6, in the order of ``layout.NAMES``) are anything with ``pos`` (centre), ``quat_xyzw``,
``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState``
has. Each is a plain cylinder whose radius and height are in ``layout``; its axis is the body's z axis. Its ends
are alike, so it may stand on either. Its lowest point and its outline seen from above are worked out from ``pos``
and ``quat_xyzw``. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that
step and leaves the tracker as it was.

Openings, in order along the rail from the left post: from the left post's inner face to the first standing
baluster (the distance from its centre to the face, less its radius), between each pair of neighbouring standing
balusters (the distance between their centres, less one diameter: the clear gap a ball must pass through), and from
the last one to the right post's inner face. With none standing there is one opening, the 356 mm between the posts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_RIM = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0.0, 2 * math.pi, 24, endpoint=False)])
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


def axis(s) -> np.ndarray:
    """The baluster's axis in the world (unit vector, either way along it)."""
    return rotation(s.quat_xyzw)[:, 2]


def tilt_deg(quat_xyzw) -> float:
    """Angle between the baluster's axis and the vertical (degrees, 0..90; either end may be down)."""
    az = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(az, 0.0, 1.0))))


def lowest(s) -> float:
    """Height (world z) of the baluster's lowest point: the lowest point of its lower end's rim."""
    az = min(abs(float(axis(s)[2])), 1.0)
    return float(s.pos[2]) - (L.BAL_H / 2) * az - L.BAL_R * math.sqrt(max(0.0, 1.0 - az * az))


def bottom_gap(s) -> float:
    """Height of the baluster's lowest point above the rail's top (m; negative = below it)."""
    return lowest(s) - L.RAIL_TOP


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def off_table(s) -> bool:
    """The baluster's lowest point is more than 10 mm below the table top: it fell off the table."""
    return bool(lowest(s) < L.TOP - L.OFF_TABLE_DROP)


# -- outlines -------------------------------------------------------------------------------------------------------
def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1])


def outline(s) -> np.ndarray:
    """The baluster's outline seen from above: the shadow of its two end rims, tilted and turned as it is."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    u, v, a = r[:, 0], r[:, 1], r[:, 2]
    pts = [c + e * (L.BAL_H / 2) * a + L.BAL_R * (cx * u + cy * v) for e in (-1, 1) for cx, cy in _RIM]
    return _hull(np.array(pts)[:, :2])


def _overlaps_rect(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Does a convex polygon (n, 2) overlap the axis-aligned rectangle [x0, x1] x [y0, y1]? (Separating axes;
    touching edges do not count.)"""
    if len(poly) < 3:
        return False
    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.hypot(*e) > 1e-12:
            axes.append(np.array([-e[1], e[0]]))
    for ax in axes:
        p, q = poly @ ax, rect @ ax
        if p.max() <= q.min() or q.max() <= p.min():
            return False
    return True


_T = L.DOWN_TOUCH
GUARD_RECTS = (  # the shoe rail's top and the two posts' tops seen from above, grown by 2 mm: (x0, x1, y0, y1)
    (L.RAIL_X - L.RAIL_W / 2 - _T, L.RAIL_X + L.RAIL_W / 2 + _T, -L.POST_FACE, L.POST_FACE),
    *((L.RAIL_X - L.POST_SIZE[0] / 2 - _T, L.RAIL_X + L.POST_SIZE[0] / 2 + _T,
       y - L.POST_SIZE[1] / 2 - _T, y + L.POST_SIZE[1] / 2 + _T) for _, y in L.POSTS),
)


# -- standing, down, openings ---------------------------------------------------------------------------------------
def stands(s) -> bool:
    """Stands on the rail: axis within 5 degrees of vertical, bottom within 2 mm of the rail's top, centre over
    the rail's top (within 20 mm of its centre line) and between the posts."""
    return bool(tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG
                and abs(bottom_gap(s)) <= L.SEAT_TOL
                and abs(float(s.pos[0]) - L.RAIL_X) <= L.OVER_RAIL
                and abs(float(s.pos[1])) < L.POST_FACE)


def down_on_guard(s) -> bool:
    """Lying on or leaning against the guard without standing: not standing, tilted more than 5 degrees, its
    outline seen from above overlapping (or within 2 mm of) the rail's top or a post's top, and its lowest point at
    most 20 mm above the rail's top (a baluster carried high over the guard, or lying on a post's top, is not
    down)."""
    if stands(s) or tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG or bottom_gap(s) > L.DOWN_CLEAR:
        return False
    poly = outline(s)
    return any(_overlaps_rect(poly, *r) for r in GUARD_RECTS)


def openings(standing) -> list[float]:
    """The openings along the rail, left post to right post (m), for the standing balusters' states."""
    order = sorted(standing, key=lambda s: -float(s.pos[1]))
    if not order:
        return [L.RAIL_LEN]
    out = [L.POST_FACE - float(order[0].pos[1]) - L.BAL_R]
    for a, b in zip(order, order[1:]):
        out.append(float(np.hypot(float(b.pos[0]) - float(a.pos[0]), float(b.pos[1]) - float(a.pos[1]))) - L.BAL_D)
    out.append(float(order[-1].pos[1]) + L.POST_FACE - L.BAL_R)
    return out


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(balusters) -> tuple[float, dict]:
    """The score of the guard as it stands, and what was found (JSON-clean)."""
    bals = list(balusters)
    names = list(L.NAMES[:len(bals)])
    up = [stands(s) for s in bals]
    down = [names[i] for i, s in enumerate(bals) if down_on_guard(s)]
    standing = [s for s, ok in zip(bals, up) if ok]
    ops = openings(standing)
    passing = sum(1 for g in ops if g >= L.BALL - _EPS)
    spread = max(ops) - min(ops)
    d: dict = {
        "standing": [names[i] for i, ok in enumerate(up) if ok],
        "n_standing": len(standing),
        "down_on_guard": down,
        "openings_mm": [_mm(g) for g in ops],
        "passing": passing,
        "spread_mm": _mm(spread),
        "widest_mm": _mm(max(ops)),
        "narrowest_mm": _mm(min(ops)),
    }
    if down:
        return 0.0, d
    if passing == 0:
        return (1.0 if spread <= L.SPREAD_MAX + _EPS else 0.5), d
    if passing == 1 and len(standing) >= L.MIN_FOR_QUARTER:
        return 0.25, d
    return 0.0, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the trace needs about this trial (the score itself depends only on the state at hand)."""

    TIERS = (0.25, 0.5, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every baluster at the previous valid control step

    @staticmethod
    def at_start(balusters) -> bool:
        """The starting layout: every baluster still and upright on the table, on its own starting mark."""
        for s, name in zip(balusters, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            on_table = abs(lowest(s) - L.TOP) < 0.003
            if not (on_mark and on_table and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some baluster moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def update(self, balusters) -> Result:
        bals = list(balusters)
        self.steps += 1
        if not all(valid(s) for s in bals):  # a blow-up: score 0 for this step, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(bals)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in bals]
        if self.prev is not None and self._teleported(now) and self.at_start(bals):
            self.reset()  # the simulator put every baluster back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.last = self.result(bals)
        for tier in self.TIERS:
            if self.last.score >= tier and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, balusters) -> Result:
        bals = list(balusters)
        bad = [L.NAMES[i] for i, s in enumerate(bals) if not valid(s)]
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_balusters": bad, **trial})
        score, details = evaluate(bals)
        # 'Still': every baluster under 1 cm/s and 0.5 rad/s, except one that has fallen off the table.
        still = all(is_resting(s) or off_table(s) for s in bals)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})
