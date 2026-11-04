"""The scorer, as plain functions of the heatsink's state. NumPy only; no simulator needed.

The rule is about how the heatsink arrives, not only where it ends: once it touches the processor it must not
slide across it (the paste smears), and it must come down onto it gently. So the scorer keeps a little state
(whether the heatsink has touched the die yet, how far it has slid while touching, how fast it came down each
time just before it came to touch) and is fed one observation per control step by ``task.success()``.

    Tracker.update(heatsink) -> Result(score, success, details)

``heatsink`` is anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low``
and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. A state with a non-finite
number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The heatsink is a plain block with two identical 56 x 56 mm faces, so either of them may face down: 'tilt' is
measured from whichever is nearer to facing down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a unit quaternion (x, y, z, w); its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(hs) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(hs, k), dtype=float).ravel()
                 for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through the heatsink's two 56 x 56 mm faces (degrees,
    0..90): the block is the same either way up, so whichever face is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def yaw_rad(quat_xyzw) -> float:
    """Heading of the body's x axis seen from above (rad)."""
    r = rotation(quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def yaw_err_deg(quat_xyzw) -> float:
    """How far the heatsink's sides are turned from the board's axes (0..45 degrees; a square has 4 ways round)."""
    a = math.degrees(yaw_rad(quat_xyzw)) % 90.0
    return float(min(a, 90.0 - a))


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def bottom_gap(hs) -> float:
    """Height of the heatsink's lowest point above the die's top (m; negative = below it)."""
    return float(hs.low[2] - L.DIE_TOP)


def _underside(hs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre of the heatsink's underside (the 56 x 56 mm face nearer to facing down), that face's outward
    normal reversed (pointing up into the heatsink), and the rotation matrix."""
    r = rotation(hs.quat_xyzw)
    up = r[:, 2] if r[2, 2] >= 0.0 else -r[:, 2]
    return np.asarray(hs.pos, dtype=float) - up * (L.HS_SIZE[2] / 2), up, r


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

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def footprint(hs) -> np.ndarray:
    """The heatsink's real outline seen from above, (n, 2): the shadow of its eight corners, turned and tilted as
    it is (for a flat heatsink, its 56 x 56 mm underside)."""
    r = rotation(hs.quat_xyzw)
    half = np.array(L.HS_SIZE) / 2
    corners = [np.asarray(hs.pos, dtype=float) + r @ (half * np.array([sx, sy, sz]))
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return _hull(np.array(corners)[:, :2])


def _overlaps_rect(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Does a convex polygon (n, 2) overlap the axis-aligned rectangle [x0, x1] x [y0, y1]? (Separating axes.)"""
    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.hypot(*e) > 1e-12:
            axes.append(np.array([-e[1], e[0]]))
    for a in axes:
        p, q = poly @ a, rect @ a
        if p.max() < q.min() or q.max() < p.min():
            return False
    return True


def over_die(hs) -> bool:
    """The heatsink's footprint overlaps the die's 30 x 30 mm top."""
    h = L.DIE_HALF
    return _overlaps_rect(footprint(hs), L.DIE_X - h, L.DIE_X + h, L.DIE_Y - h, L.DIE_Y + h)


def touching(hs) -> bool:
    """'Touching': the heatsink's bottom within 2 mm of the die's top while its footprint overlaps the die."""
    return bool(over_die(hs) and abs(bottom_gap(hs)) <= L.TOUCH_TOL)


def centre_off(hs) -> float:
    """Horizontal distance from the heatsink's centre to the die's centre (m)."""
    return float(np.hypot(hs.pos[0] - L.DIE_X, hs.pos[1] - L.DIE_Y))


def on_post(hs) -> bool:
    """Resting on a post: a post's top lies under the heatsink's footprint and the underside is at most 2 mm
    above that post's top there."""
    c, n, r = _underside(hs)
    if n[2] < 0.5:  # tilted past 60 degrees: the underside is not facing down (tipped, scored 0 anyway)
        return False
    for _, px, py in L.POSTS:
        z = c[2] - (n[0] * (px - c[0]) + n[1] * (py - c[1])) / n[2]  # the underside's plane above the post
        d = np.array([px, py, z]) - c
        du = max(abs(float(d @ r[:, 0])) - L.HS_HALF, 0.0)
        dv = max(abs(float(d @ r[:, 1])) - L.HS_HALF, 0.0)
        if math.hypot(du, dv) < L.POST_RADIUS and z <= L.POST_TOP + L.POST_TOL:
            return True
    return False


def off_board(hs) -> bool:
    """The heatsink's centre is not over the 200 x 160 mm board."""
    hx, hy = L.BOARD_SIZE[0] / 2, L.BOARD_SIZE[1] / 2
    return bool(abs(hs.pos[0] - L.BOARD_X) > hx or abs(hs.pos[1] - L.BOARD_Y) > hy)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.touched = False  # the heatsink has touched the die
        self.smear = 0.0  # m the heatsink has slid over the die while touching (centre travel + turn)
        self.setdown = 0.0  # m/s: fastest descent in the last 10 mm each time before it came to touch, and then
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (centre, yaw, bottom gap, touching) at the previous valid control step
        self.mark = None  # (centre xy, yaw) where smear was last counted, while touching

    @staticmethod
    def at_start(hs) -> bool:
        """The starting layout: the heatsink standing still, square and upright on the table on its mark."""
        on_mark = (np.hypot(hs.pos[0] - L.HS_X, hs.pos[1] - L.HS_Y) < L.START_TOL
                   and abs(hs.low[2] - L.TOP) < 0.003)
        square = tilt_deg(hs.quat_xyzw) < 1.0 and yaw_err_deg(hs.quat_xyzw) < 1.0
        return bool(on_mark and square and is_resting(hs))

    @staticmethod
    def _travel(a, b) -> float:
        """How far the heatsink moved over the die from pose a to pose b ((centre xy, yaw) each): the centre's
        horizontal travel plus the turn about the vertical counted at the die's corners."""
        turn = abs(math.remainder(b[1] - a[1], 2 * math.pi))
        return float(np.hypot(*(np.asarray(b[0]) - np.asarray(a[0])))) + L.SMEAR_TURN_RADIUS * turn

    def update(self, hs) -> Result:
        if not valid(hs):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(hs)
            return self.last
        c = np.array(hs.pos, dtype=float)
        if self.prev is not None and self.at_start(hs):
            if float(np.linalg.norm(c - self.prev[0])) > L.RESET_JUMP:  # the simulator put it back: a new episode
                self.reset()
        self.steps += 1
        yaw, gap, touch = yaw_rad(hs.quat_xyzw), bottom_gap(hs), touching(hs)
        was_touching = self.prev is not None and self.prev[3]
        now = (c[:2], yaw)

        # Set-down speed, every time it comes down onto the die (not only the first): its downward speed while
        # over the die in the last 10 mm above the touching band, and on the step it comes to touch also how far
        # it came down in that step (a fast drop can skip the window between two control steps).
        descent = max(-float(hs.lin_vel[2]), 0.0)
        if touch and not was_touching:
            if self.prev is not None:
                descent = max(descent, (self.prev[2] - gap) / self.dt)
            self.setdown = max(self.setdown, descent)
            self.touched = True
        elif not touch and over_die(hs) and L.TOUCH_TOL < gap <= L.TOUCH_TOL + L.SETDOWN_WINDOW:
            self.setdown = max(self.setdown, descent)

        # Smear, over every step that begins or ends touching, followed in steps of SMEAR_STEP.
        if touch:
            if was_touching and self.mark is not None:
                d = self._travel(self.mark, now)
                if d >= L.SMEAR_STEP:
                    self.smear += d
                    self.mark = now
            else:  # it comes to touch: all of that step's travel counts
                if self.prev is not None:
                    self.smear += self._travel((self.prev[0][:2], self.prev[1]), now)
                self.mark = now
        elif was_touching:  # it stops touching: what was left since the last count, and the step that left
            if self.mark is not None:
                self.smear += self._travel(self.mark, now)
            self.mark = None

        self.prev = (c, yaw, gap, touch)
        self.last = self.result(hs)
        return self.last

    def result(self, hs) -> Result:
        if not valid(hs):
            return Result(0.0, False, {"valid_state": False, "touched": self.touched,
                                       "smear_mm": round(self.smear * 1000, 2), "setdown_mps": round(self.setdown, 3)})
        tilt, yaw_err, off = tilt_deg(hs.quat_xyzw), yaw_err_deg(hs.quat_xyzw), centre_off(hs)
        touch, post, away = touching(hs), on_post(hs), off_board(hs)
        flat = tilt < L.FLAT_DEG
        exact = touch and flat and off <= L.CENTRE_TIGHT and yaw_err <= L.YAW_TIGHT_DEG
        if away or post or tilt > L.TIP_DEG:
            score = 0.0
        elif exact and self.smear <= L.SMEAR_TIGHT and self.setdown <= L.SETDOWN_MAX:
            score = 1.0
        elif exact and self.smear <= L.SMEAR_LOOSE:
            score = 0.75
        elif touch and flat and off <= L.CENTRE_LOOSE and yaw_err <= L.YAW_LOOSE_DEG:
            score = 0.5
        elif touch:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(hs))
        details = {
            "valid_state": True,
            "touching": touch,
            "touched": self.touched,
            "bottom_gap_mm": round(bottom_gap(hs) * 1000, 2),
            "centre_off_mm": round(off * 1000, 2),
            "yaw_err_deg": round(yaw_err, 2),
            "tilt_deg": round(tilt, 2),
            "smear_mm": round(self.smear * 1000, 2),
            "setdown_mps": round(self.setdown, 3),
            "on_post": post,
            "off_board": away,
        }
        return Result(score, success, details)
