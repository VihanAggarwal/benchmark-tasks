"""The scorer, as plain functions of the four flags' states and the arms' state. NumPy only; no simulator needed.

A set of sampling flags is checked at the end, after the arms withdraw: a top photo or a tape from each standing
flag's base to the stains. So the score is an end state (how many of the four sampling points have a planted flag
on them), plus two facts that need a short memory: whether each flag has been still for the last 0.5 s (its base
centre's path, never its velocity, which chatters while a part is gripped) and whether a gripper has let it go.
The tracker is fed one observation per control step by ``task.success()``.

    Tracker.update(flags, arms) -> Result(score, success, details)

``flags`` maps each flag's name to anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has.
``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the pad-tip point, 13 mm
past the TCP, midway between the finger tips) and ``opening`` (0 closed .. 1 fully open), or ``None`` when the arms
could not be read. Each arm is checked on its own: an arm that cannot be read (``None``, its key missing, or a
non-finite or malformed pad-tip point) is skipped by the let-go rule for that step, the other arm is still checked,
and the step is counted in ``arms_unavailable`` (and per arm in ``arms_unavailable_by_side``). rigstate does not
report a gripper's yaw, so the let-go rule takes the fingers' widest reach over every yaw at the gripper's current
opening (``layout.release_reach``). A step on which any flag's state has a non-finite number or a zero quaternion
(a simulator blow-up) scores 0 and leaves the tracker as it was.

A flag is a plain cylinder whose two ends are alike, so either end may be down: its 'base' is whichever end is
lower, and 'tilt' is measured from the axis pointing up out of that end.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
SIDES = ("left", "right")


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


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[4].size == 3 and parts[5].size == 3)


def arm_tips(arms, side: str):
    """One arm's pad-tip point as a finite 3-vector, or None when that arm cannot be read: ``arms`` is None or not a
    dict, its key is missing, or its point is non-finite or malformed. One arm's failure never hides the other."""
    if not isinstance(arms, dict):
        return None
    try:
        tips = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return tips if tips.size == 3 and np.all(np.isfinite(tips)) else None


def readable_sides(arms) -> tuple[str, ...]:
    """The arms whose pad-tip point can be read at this step (rigstate.arms() can be None)."""
    return tuple(side for side in SIDES if arm_tips(arms, side) is not None)


def axis(flag) -> np.ndarray:
    """The flag's axis (unit vector), pointing up out of whichever end is down."""
    a = rotation(flag.quat_xyzw)[:, 2]
    return a if a[2] >= 0.0 else -a


def tilt_deg(flag) -> float:
    """Angle between the flag's axis and the vertical (degrees, 0..90; either end may be down)."""
    return float(np.degrees(np.arccos(np.clip(axis(flag)[2], 0.0, 1.0))))


def base_centre(flag) -> np.ndarray:
    """Centre of the flag's lower end (m, world)."""
    return np.asarray(flag.pos, dtype=float) - axis(flag) * (L.FLAG_HEIGHT / 2)


def bottom_height(flag) -> float:
    """Height of the flag's lowest point above the table top (m)."""
    return float(flag.low[2] - L.TOP)


def upright(flag) -> bool:
    """(a) Its axis within 10 degrees of vertical."""
    return tilt_deg(flag) <= L.UPRIGHT_DEG


def on_ground(flag) -> bool:
    """(b) Its lowest point within 5 mm of the table top: standing on the table or on the sheets (1 to 3 mm
    thick), not on another flag (60 mm tall) and not held up in the air."""
    return abs(bottom_height(flag)) <= L.BOTTOM_TOL


def top_height(flag) -> float:
    """World height of the flag's highest point (m)."""
    return float(flag.high[2])


def tip_offset(arm, flag) -> tuple[float, float]:
    """For one arm: the horizontal distance from its pad-tip point to the flag's axis (taken at the tip's height),
    and the tip's height above the flag's top (m)."""
    p = np.asarray(arm.tips, dtype=float).ravel()
    b = base_centre(flag)
    u = axis(flag)
    on_axis = b + u * ((p[2] - b[2]) / u[2]) if u[2] > 0.1 else b  # where the axis is at the tip's height
    return float(np.hypot(*(p[:2] - on_axis[:2]))), float(p[2] - top_height(flag))


def opening(arm) -> float:
    """rigstate's opening, 0 (closed) .. 1 (fully open); an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def reach(arm) -> float:
    """(d) How far from this gripper's pad-tip point (horizontally) a flag's axis can be while a finger touches it,
    over every yaw, plus 5 mm: 78 mm fully open, 48 mm closed on a flag (m)."""
    return L.release_reach(opening(arm))


def held(flag, arms) -> bool:
    """(d) not met: a gripper's pad-tip point lower than 10 mm above the flag's top with the flag's axis within
    that gripper's reach horizontally (so a finger can be touching, pressing or propping the flag, whatever the
    gripper's yaw). Each arm is checked on its own: an unreadable arm is skipped (and counted by the tracker), and
    the other, readable arm is still checked, so a flag it grips is still held."""
    for side in readable_sides(arms):
        xy, above = tip_offset(arms[side], flag)
        if xy < reach(arms[side]) and above < L.RELEASE_ABOVE:
            return True
    return False


def path_length(history) -> float:
    """Path length of a sequence of positions (m)."""
    pts = list(history)
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(pts, pts[1:])))


def in_target(px: float, py: float, stain: str) -> bool:
    """HOT SPOT: a point inside the stain's core at least 5 mm in from all four edges (its 40 x 40 mm target)."""
    x0, x1, y0, y1 = L.target(stain)
    return bool(x0 <= px <= x1 and y0 <= py <= y1)


def on_sheet_inside(px: float, py: float) -> bool:
    """A point on the site sheet at least 15 mm inside its edges: a flag based there stands wholly on the sheet,
    its 12.5 mm radius with 2.5 mm to spare."""
    x0, x1, y0, y1 = L.rect_bounds(L.SHEET)
    m = L.SHEET_INSET
    return bool(x0 + m <= px <= x1 - m and y0 + m <= py <= y1 - m)


def is_background(px: float, py: float) -> bool:
    """BACKGROUND: a point on the sheet at least 15 mm inside its edges and at least 150 mm from every stain's
    outline (the nearest point of every halo lobe)."""
    return bool(on_sheet_inside(px, py) and L.nearest_stain_distance(px, py) >= L.BACKGROUND_MIN)


def item_at(px: float, py: float) -> str | None:
    """Which sampling point a planted flag's base centre at (px, py) meets: 'S1', 'S2', 'S3', 'background' or
    None. A core lies inside its halo, so no point meets two items."""
    for stain in L.STAIN_NAMES:
        if in_target(px, py, stain):
            return stain
    return "background" if is_background(px, py) else None


def items_met(placed: dict[str, str | None]) -> list[str]:
    """The sampling points with at least one planted flag on them (each counts once)."""
    return sorted({item for item in placed.values() if item is not None})


def at_start_mark(flag, mark) -> bool:
    """Standing upright on the bare table on its start mark."""
    _, x, y = mark
    return bool(np.hypot(flag.pos[0] - x, flag.pos[1] - y) < L.START_TOL and abs(bottom_height(flag)) < 0.003
                and tilt_deg(flag) < 5.0)


def _mm(v: float) -> float:
    return round(float(v) * 1000.0, 1)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: each flag's recent base-centre positions, and the steps the arms
    were unreadable."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.invalid_steps = 0  # steps with a non-finite flag state (scored 0, nothing recorded)
        self.arms_unavailable = 0  # steps on which rigstate could not read one arm or both
        self.arms_unavailable_by_side = {side: 0 for side in SIDES}  # the same, per arm
        self.history = {n: deque(maxlen=L.REST_WINDOW + 1) for n in L.FLAG_NAMES}
        self.prev = None  # each flag's centre at the previous valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(flags) -> bool:
        """The starting layout: every flag standing on the table on its own mark."""
        return all(at_start_mark(flags[name], mark) for name, mark in zip(L.FLAG_NAMES, L.FLAG_STARTS))

    def at_rest(self, name: str) -> bool:
        """(c) The base centre's path over the last 0.5 s (10 control steps) under 2 mm."""
        h = self.history[name]
        return len(h) == h.maxlen and path_length(h) < L.REST_PATH

    @staticmethod
    def _all_valid(flags) -> bool:
        try:
            return all(name in flags and valid(flags[name]) for name in L.FLAG_NAMES)
        except TypeError:
            return False

    def update(self, flags, arms=None) -> Result:
        if not self._all_valid(flags):  # a blow-up: 0 for this step, nothing recorded
            self.steps += 1
            self.invalid_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "invalid_steps": self.invalid_steps})
            return self.last
        now = {n: np.array(flags[n].pos, dtype=float) for n in L.FLAG_NAMES}
        if self.prev is not None and self.at_start(flags):
            if max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.FLAG_NAMES) > L.RESET_JUMP:
                self.reset()  # the simulator put the flags back on their marks: a new episode
        self.steps += 1
        for n in L.FLAG_NAMES:
            self.history[n].append(base_centre(flags[n]))
        self.prev = now
        readable = readable_sides(arms)
        for side in SIDES:
            if side not in readable:
                self.arms_unavailable_by_side[side] += 1
        if len(readable) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(flags, arms)
        return self.last

    def result(self, flags, arms=None) -> Result:
        if not self._all_valid(flags):
            return Result(0.0, False, {"valid_state": False, "invalid_steps": self.invalid_steps})
        readable = readable_sides(arms)
        placed, per_flag = {}, {}
        for name in L.FLAG_NAMES:
            f = flags[name]
            b = base_centre(f)
            up, ground, rest, grip = upright(f), on_ground(f), self.at_rest(name), held(f, arms)
            planted = bool(up and ground and rest and not grip)
            item = item_at(float(b[0]), float(b[1]))
            placed[name] = item if planted else None
            row = {
                "base_x_mm": _mm(b[0]), "base_y_mm": _mm(b[1]), "tilt_deg": round(tilt_deg(f), 2),
                "bottom_mm": _mm(bottom_height(f)), "path_mm": _mm(path_length(self.history[name])),
                "upright": up, "on_ground": ground, "at_rest": rest, "held": grip, "planted": planted,
                "spot": item, "stain_dist_mm": _mm(L.nearest_stain_distance(float(b[0]), float(b[1]))),
            }
            if readable:
                offs = {s: tip_offset(arms[s], f) for s in readable}
                row["tip_xy_mm"] = min(_mm(o[0]) for o in offs.values())
                row["tip_above_mm"] = min(_mm(o[1]) for o in offs.values())
            per_flag[name] = row
        met = items_met(placed)
        score = L.POINTS_PER_ITEM * len(met)
        success = bool(len(met) == len(L.STAIN_NAMES) + 1)
        details = {
            "valid_state": True,
            "items": met,
            "n_items": len(met),
            "flags": per_flag,
            "arms_available": len(readable) == len(SIDES),
            "arms_readable": list(readable),
            "reach_mm": {s: (_mm(reach(arms[s])) if s in readable else None) for s in SIDES} if readable else None,
            "arms_unavailable": self.arms_unavailable,
            "arms_unavailable_by_side": dict(self.arms_unavailable_by_side),
            "invalid_steps": self.invalid_steps,
        }
        return Result(score, success, details)
