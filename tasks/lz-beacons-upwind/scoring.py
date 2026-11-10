"""The scorer, as plain functions of the five beacons' states. NumPy only; no simulator needed.

The rule is about the finished landing zone, read from the state at the last control step: the four orange beacons
stand on the grass at the corners of a zone at least 100 ft (277 mm at 1:110) on every side with square corners
(0.5; 0.25 for at least 90 ft, 249 mm, and corners within 20 degrees of square), and the blue beacon stands on the
grass just outside the middle of the zone's upwind side (0.5; 0.25 anywhere outside that side, up to 100 mm out).
The windsock streams toward -y, so the wind comes from +y. A beacon counts only once it is let go: no gripper's
pad-tip point lower than 10 mm above its top within that gripper's finger reach of its axis. A small tracker is
still fed once per control step by ``task.success()``: it keeps each beacon's recent path (for 'still', which
success needs), what the trace needs (when each score was first reached, how many observations were broken, on how
many steps an arm could not be read), and recognises a new episode when the simulator puts every beacon back on its
starting mark in one step.

    Tracker.update(beacons, arms) -> Result(score, success, details)

``beacons`` (5, in the order of ``layout.NAMES``: four orange, then the blue) are anything with ``pos`` (centre),
``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as
``yam_arena.task_file.ObjectState`` has. Each is an upright cylinder whose axis is the body's z axis; its two ends
are alike, so it may stand either way up. Its lowest point is worked out from ``pos`` and ``quat_xyzw``. A state
with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker
as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point, 13 mm past the
TCP, midway between the finger tips) and ``opening`` (0 closed .. 1 fully open), or ``None`` when the arms could not
be read. An arm that cannot be read is skipped by the let-go rule (the other is still checked) and the step is
counted in ``arms_unavailable``. Offline, and in these functions' defaults, ``arms=None``.

The orange beacons are identical: any of them may be any corner.
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
    """Angle between the beacon's axis and the vertical (degrees, 0..90; either end up)."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def lowest(s) -> float:
    """Height (world z) of the beacon's lowest point, from its pose: half its height along the axis's vertical
    part plus its radius along the rest."""
    az = abs(float(rotation(s.quat_xyzw)[2, 2]))
    side = math.sqrt(max(0.0, 1.0 - az * az))
    return float(s.pos[2]) - L.BEACON_H / 2 * az - L.BEACON_RADIUS * side


def highest(s) -> float:
    """Height (world z) of the beacon's highest point, from its pose (the mirror of ``lowest``)."""
    az = abs(float(rotation(s.quat_xyzw)[2, 2]))
    side = math.sqrt(max(0.0, 1.0 - az * az))
    return float(s.pos[2]) + L.BEACON_H / 2 * az + L.BEACON_RADIUS * side


def bottom_gap(s) -> float:
    """Height of the beacon's lowest point above the grass top (m; negative = below it)."""
    return lowest(s) - L.MAT_TOP


def on_grass_outline(s) -> bool:
    """Its centre, seen from above, is inside the grass field's outline."""
    x0, x1, y0, y1 = L.GRASS
    x, y = float(s.pos[0]), float(s.pos[1])
    return bool(x0 - _EPS <= x <= x1 + _EPS and y0 - _EPS <= y <= y1 + _EPS)


def standing(s) -> bool:
    """Standing on the grass: tilted at most 10 degrees, its lowest point within 3 mm of the grass top and its
    centre inside the grass outline."""
    return bool(tilt_deg(s.quat_xyzw) <= L.UPRIGHT_DEG + _EPS
                and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS
                and on_grass_outline(s))


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
    """How far from this gripper's pad-tip point (horizontally) a beacon's axis can be while a finger touches it,
    over every yaw, plus 5 mm: 79 mm fully open, 50 mm closed on a beacon (m)."""
    return L.release_reach(opening(arm))


def tip_offset(tips, s) -> tuple[float, float]:
    """For one pad-tip point: the horizontal distance to the beacon's axis (taken at the tip's height) and the
    tip's height above the beacon's highest point (m)."""
    p = np.asarray(tips, dtype=float).ravel()
    c = np.asarray(s.pos, dtype=float)
    u = rotation(s.quat_xyzw)[:, 2]
    u = u if u[2] >= 0.0 else -u
    on_axis = c + u * ((p[2] - c[2]) / u[2]) if u[2] > 0.1 else c
    return float(np.hypot(*(p[:2] - on_axis[:2]))), float(p[2] - highest(s))


def held(s, arms) -> bool:
    """Not let go: some gripper's pad-tip point lower than 10 mm above the beacon's top with the beacon's axis within
    that gripper's reach horizontally (a finger may be touching, pressing or propping it, whatever the gripper's
    yaw). Each arm is checked on its own: an unreadable arm is skipped, the other is still checked."""
    for side in readable_sides(arms):
        xy, above = tip_offset(arm_tips(arms, side), s)
        if xy < reach(arms[side]) and above < L.RELEASE_ABOVE:
            return True
    return False


def placed(s, arms=None) -> bool:
    """Standing on the grass (``standing``) and let go (not ``held`` by either readable gripper)."""
    return bool(standing(s) and not held(s, arms))


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- the zone ------------------------------------------------------------------------------------------------------
def corners(points) -> np.ndarray:
    """The four orange centres (x, y) in order of angle round their centroid: counter-clockwise seen from above."""
    p = np.asarray(points, dtype=float)[:, :2]
    c = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]), kind="stable")
    return p[order]


def quad(points) -> dict:
    """Sides (m, side i runs from corner i to corner i + 1), corner angles (degrees, at corner i), whether the
    quadrilateral is convex, and its corners."""
    q = corners(points)
    n = len(q)
    sides = [float(np.linalg.norm(q[(i + 1) % n] - q[i])) for i in range(n)]
    angles, turns = [], []
    for i in range(n):
        a, b = q[i - 1] - q[i], q[(i + 1) % n] - q[i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            angles.append(0.0)
        else:
            angles.append(float(np.degrees(np.arccos(np.clip(float(a @ b) / (na * nb), -1.0, 1.0)))))
        e0, e1 = q[i] - q[i - 1], q[(i + 1) % n] - q[i]
        turns.append(float(e0[0] * e1[1] - e0[1] * e1[0]))
    convex = all(t > 1e-12 for t in turns)
    return {"corners": q, "sides": sides, "angles": angles, "convex": convex}


def square_score(z: dict) -> float:
    """0.5: every side at least 277 mm (100 ft) and every corner 80-100 degrees; 0.25: every side at least 249 mm
    (90 ft) and every corner 70-110 degrees; 0 otherwise (too small, not square, not a convex quadrilateral)."""
    if not z["convex"]:
        return 0.0
    sides, angles = z["sides"], z["angles"]
    if min(sides) >= L.SIDE_FULL - _EPS and all(L.ANGLE_FULL[0] - _EPS <= a <= L.ANGLE_FULL[1] + _EPS
                                                 for a in angles):
        return 0.5
    if min(sides) >= L.SIDE_PART - _EPS and all(L.ANGLE_PART[0] - _EPS <= a <= L.ANGLE_PART[1] + _EPS
                                                 for a in angles):
        return 0.25
    return 0.0


def upwind_sides(z: dict) -> tuple[list[int], list[np.ndarray]]:
    """The upwind side: the side whose outward direction (square to it, away from the zone) is nearest to the
    direction the wind comes from (+y). Returns the side indices that tie for nearest (normally one) and every
    side's outward unit normal."""
    q = z["corners"]
    wind = np.array(L.WIND_FROM, dtype=float)
    normals = []
    for i in range(len(q)):
        e = q[(i + 1) % len(q)] - q[i]
        ne = float(np.linalg.norm(e))
        normals.append(np.array([e[1], -e[0]]) / ne if ne > 1e-9 else np.zeros(2))  # right of a CCW edge: out
    dots = [float(n @ wind) for n in normals]
    best = max(dots)
    return [i for i, d in enumerate(dots) if d >= best - 1e-9], normals


def marker_on_side(z: dict, i: int, normal: np.ndarray, blue_xy) -> tuple[float, float, float]:
    """(d, t, score) of the blue beacon's centre against side i: d its distance outside the side's line (m;
    negative = inside), t its position along the side (0 at corner i, 1 at corner i + 1), score 0.5 / 0.25 / 0."""
    q = z["corners"]
    a, b = q[i], q[(i + 1) % len(q)]
    e = b - a
    p = np.asarray(blue_xy, dtype=float)[:2] - a
    d = float(p @ normal)
    t = float(p @ e) / float(e @ e)
    if L.MARK_NEAR - _EPS <= d <= L.MARK_FAR + _EPS and L.MARK_T[0] - _EPS <= t <= L.MARK_T[1] + _EPS:
        return d, t, 0.5
    if _EPS < d <= L.MARK_LOOSE + _EPS and -_EPS <= t <= 1.0 + _EPS:  # outside the line, not on it
        return d, t, 0.25
    return d, t, 0.0


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(beacons, arms=None) -> tuple[float, dict]:
    """The score of the zone as it stands, and what was found (JSON-clean). ``arms``: rigstate.arms() or None."""
    bs = list(beacons)
    names = list(L.NAMES[:len(bs)])
    oranges, blue = bs[:4], bs[4]
    not_placed = [names[i] for i, s in enumerate(oranges) if not placed(s, arms)]
    blue_placed = placed(blue, arms)
    d: dict = {"orange_not_placed": not_placed, "blue_placed": blue_placed,
               "held": [names[i] for i, s in enumerate(bs) if held(s, arms)], "square": 0.0, "marker": 0.0,
               "sides_mm": None, "angles_deg": None, "convex": None, "upwind_side": None,
               "upwind_normal_deg": None, "blue_out_mm": None, "blue_along": None}
    if not_placed:
        return 0.0, d
    z = quad([s.pos for s in oranges])
    sq = square_score(z)
    d.update(sides_mm=[_mm(v) for v in z["sides"]], angles_deg=[round(a, 1) for a in z["angles"]],
             convex=z["convex"], square=sq)
    if sq == 0.0:
        return 0.0, d
    up, normals = upwind_sides(z)
    best = None
    for i in up:
        out, along, m = marker_on_side(z, i, normals[i], blue.pos)
        if best is None or m > best[3]:
            best = (i, out, along, m)
    i, out, along, m = best
    if not blue_placed:
        m = 0.0
    off_wind = float(np.clip(normals[i] @ np.array(L.WIND_FROM, dtype=float), -1.0, 1.0))
    d.update(upwind_side=i, upwind_normal_deg=round(math.degrees(math.acos(off_wind)), 1),
             blue_out_mm=_mm(out), blue_along=round(along, 3), marker=m)
    return sq + m, d


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
        self.arms_unavailable = 0  # steps on which rigstate could not read one arm or both
        self.arms_unavailable_by_side = {side: 0 for side in SIDES}  # the same, per arm
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every beacon at the previous valid control step
        self.history: deque = deque(maxlen=self.window + 1)  # every beacon's centre at the last valid steps

    @staticmethod
    def at_start(beacons) -> bool:
        """The starting layout: every beacon still and upright on the road or the strip, on its own mark."""
        for s, name in zip(beacons, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(bottom_gap(s)) < L.SEAT_TOL
            if not (on_mark and seated and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some beacon moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def paths(self) -> list[float] | None:
        """Each beacon's centre path (m) over the last 0.5 s, or None before 0.5 s of valid steps have been seen."""
        if len(self.history) < self.window + 1:
            return None
        h = list(self.history)
        return [float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(h, h[1:]))) for i in range(len(h[0]))]

    def update(self, beacons, arms=None) -> Result:
        bs = list(beacons)
        self.steps += 1
        if not all(valid(s) for s in bs):  # a blow-up: score 0 for this step, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(bs, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in bs]
        if self.prev is not None and self._teleported(now) and self.at_start(bs):
            self.reset()  # the simulator put every beacon back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        readable = readable_sides(arms)
        for side in SIDES:
            if side not in readable:
                self.arms_unavailable_by_side[side] += 1
        if len(readable) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(bs, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, beacons, arms=None) -> Result:
        bs = list(beacons)
        bad = [L.NAMES[i] for i, s in enumerate(bs) if not valid(s)]
        readable = readable_sides(arms)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_readable": list(readable),
                 "reach_mm": {s: _mm(reach(arms[s])) for s in readable},
                 "arms_unavailable": self.arms_unavailable,
                 "arms_unavailable_by_side": dict(self.arms_unavailable_by_side)}
        if bad or len(bs) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_beacons": bad, **trial})
        score, details = evaluate(bs, arms)
        paths = self.paths()
        still = paths is not None and all(p < L.STILL_PATH for p in paths)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})
