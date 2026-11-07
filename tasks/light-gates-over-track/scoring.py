"""The scorer, as plain functions of the two light gates' states. NumPy only; no simulator needed.

A light gate is set when it stands on the bench straddling the air track: both feet flat on the bench, one on
each side of the track, the crossbar square to it, its centre in the timing section in front of the parked glider,
at rest. Nothing of either gate may be in the glider's path (the space it sweeps on its run): a gate resting on the
track, lying across it or standing too close to it would stop the run, and the trial scores 0. Two set gates also
earn credit for their spacing along the track. The scorer keeps a short history of where each gate's feet were (to
tell whether it is at rest) and is fed one observation per control step by ``task.success()``:

    Tracker.update({"gate_left": state, "gate_right": state}) -> Result(score, success, details)

A state is anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has, plus ``origin`` (the gate's base frame: the centre between its feet, on
their undersides), as mechkit's ``AssemblyState`` has; without ``origin`` it is worked out from ``pos``. A state
with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker
as it was.
"""

from __future__ import annotations

import math
from collections import deque
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


def valid(g) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(g, k), dtype=float).ravel()
                 for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")]
        origin = getattr(g, "origin", None)
        if origin is not None:
            parts.append(np.asarray(origin, dtype=float).ravel())
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def frame(g) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the gate's base frame in the world: ``origin`` if the state has it (mechkit), else
    the centre of its box less the box centre's offset from the base frame."""
    r = rotation(g.quat_xyzw)
    origin = getattr(g, "origin", None)
    if origin is not None:
        return r, np.asarray(origin, dtype=float)
    return r, np.asarray(g.pos, dtype=float) - r @ np.asarray(L.GATE_BBOX_CENTRE, dtype=float)


def to_world(r: np.ndarray, o: np.ndarray, local) -> np.ndarray:
    return np.asarray(local, dtype=float) @ r.T + o


def foot_corners(r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(8, 3): the four corners of each foot's underside."""
    hx, hy = L.FOOT_SIZE[0] / 2, L.FOOT_SIZE[1] / 2
    pts = [(sx * L.LEG_X + dx, dy, 0.0) for sx in (-1, 1) for dx in (-hx, hx) for dy in (-hy, hy)]
    return to_world(r, o, pts)


def foot_centres(r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(2, 3): the centre of each foot's underside (local x -0.040 first)."""
    return to_world(r, o, [(-L.LEG_X, 0.0, 0.0), (L.LEG_X, 0.0, 0.0)])


def heading_deg(r: np.ndarray) -> float:
    """Heading of the crossbar (local x) seen from above, degrees."""
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def square_err_deg(r: np.ndarray) -> float:
    """How far the crossbar is turned from straight across the track (world x), 0..90 degrees; the gate is the same
    either way round."""
    a = heading_deg(r) % 180.0
    return float(min(a, 180.0 - a))


def obb_hits_aabb(c, r, half, lo, hi) -> bool:
    """Does a turned box (centre c, axes the columns of r, half sizes half) touch or overlap the axis-aligned box
    [lo, hi]? Separating-axis test over the 15 candidate axes."""
    c, half, lo, hi = (np.asarray(v, dtype=float) for v in (c, half, lo, hi))
    a_c, a_h = (lo + hi) / 2, (hi - lo) / 2
    t = c - a_c
    eye = np.eye(3)
    axes = [eye[i] for i in range(3)] + [r[:, j] for j in range(3)]
    axes += [np.cross(eye[i], r[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        n = float(np.linalg.norm(ax))
        if n < 1e-9:
            continue
        ax = ax / n
        ra = float(np.sum(a_h * np.abs(ax)))
        rb = float(np.sum(half * np.abs(r.T @ ax)))
        if abs(float(t @ ax)) > ra + rb + 1e-12:
            return False
    return True


PATH_LO = np.array([L.TRACK_X - L.PATH_HALF_W, L.PATH_Y0, L.TOP - 0.010])
PATH_HI = np.array([L.TRACK_X + L.PATH_HALF_W, L.PATH_Y1, L.TOP + L.PATH_TOP])


def in_path(r: np.ndarray, o: np.ndarray) -> bool:
    """Any part of the gate (feet, legs, windows, crossbar, as turned and tilted) within 22 mm of the track's centre
    line across it, lower than 57 mm above the bench, between the buffers: the glider would hit it."""
    for centre, size in L.GATE_BOXES:
        c = to_world(r, o, centre)
        if obb_hits_aabb(c, r, np.asarray(size, dtype=float) / 2, PATH_LO, PATH_HI):
            return True
    return False


def path_gap(r: np.ndarray, o: np.ndarray) -> float:
    """Diagnostic (m): how far the gate's lower parts (feet, legs, windows) stay outside the path's 22 mm half-width,
    across the track, at their nearest corner; negative if inside it or across the centre line."""
    best = math.inf
    for centre, size in L.GATE_BOXES:
        if centre[2] > L.CROSSBAR_Z0:  # the crossbar passes over the path
            continue
        h = np.asarray(size, dtype=float) / 2
        corners = np.array([(sx * h[0], sy * h[1], sz * h[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        dx = to_world(r, o, corners + np.asarray(centre, dtype=float))[:, 0] - L.TRACK_X
        d = float(np.min(np.abs(dx))) if (np.all(dx > 0) or np.all(dx < 0)) else 0.0
        best = min(best, d - L.PATH_HALF_W)
    return best


def gate_facts(g) -> dict:
    """Everything about one gate's pose that the rule needs (rest is the tracker's)."""
    r, o = frame(g)
    corners = foot_corners(r, o)
    feet = foot_centres(r, o)
    heights = corners[:, 2] - L.TOP
    standing = bool(np.all(np.abs(heights) <= L.STAND_TOL))
    straddling = bool((feet[0, 0] - L.TRACK_X) * (feet[1, 0] - L.TRACK_X) < 0.0)
    square = square_err_deg(r)
    section = bool(L.SECTION_Y0 <= o[1] <= L.SECTION_Y1)
    return {
        "centre": (float(o[0]), float(o[1])),
        "standing": standing,
        "straddling": straddling,
        "square_err_deg": square,
        "in_section": section,
        "in_path": in_path(r, o),
        "foot_low_mm": float(heights.min() * 1000),
        "foot_high_mm": float(heights.max() * 1000),
        "off_centre_mm": float((o[0] - L.TRACK_X) * 1000),
        "path_gap_mm": float(path_gap(r, o) * 1000),
        "feet": feet,
    }


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: where each gate's feet were over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = int(round(L.REST_WINDOW_S * control_hz))  # 10 steps at 20 Hz: 11 samples span 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hist = {name: deque(maxlen=self.window + 1) for name in L.GATES}
        self.prev = None  # each gate's origin at the previous valid control step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, g) -> bool:
        """The starting layout: this gate standing on the bench on its starting mark."""
        r, o = frame(g)
        sx, sy = L.START[name]
        on_mark = math.hypot(o[0] - sx, o[1] - sy) < L.START_TOL
        return bool(on_mark and np.all(np.abs(foot_corners(r, o)[:, 2] - L.TOP) <= L.STAND_TOL))

    def at_rest(self, name: str) -> bool:
        """Neither foot of this gate has moved more than 2 mm in the last 0.5 s (from positions, not the reported
        velocity, which chatters while a part is gripped)."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return False
        now = h[-1]
        return bool(all(float(np.max(np.linalg.norm(p - now, axis=1))) <= L.REST_MOVE for p in h))

    def update(self, gates: dict) -> Result:
        if not all(valid(gates.get(n)) for n in L.GATES):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.last = self.result(gates)
            return self.last
        origins = {n: frame(gates[n])[1] for n in L.GATES}
        if self.prev is not None and all(self.at_start(n, gates[n]) for n in L.GATES):
            jump = max(float(np.linalg.norm(origins[n] - self.prev[n])) for n in L.GATES)
            if jump > L.RESET_JUMP:  # the simulator put the gates back: a new episode
                self.reset()
        self.steps += 1
        for n in L.GATES:
            r, o = frame(gates[n])
            self.hist[n].append(foot_centres(r, o))
        self.prev = origins
        self.last = self.result(gates)
        return self.last

    def result(self, gates: dict) -> Result:
        if not all(valid(gates.get(n)) for n in L.GATES):
            return Result(0.0, False, {"valid_state": False})
        facts = {n: gate_facts(gates[n]) for n in L.GATES}
        gate_sets = {}
        rows = {}
        for n, f in facts.items():
            rest = self.at_rest(n)
            is_set = (f["standing"] and f["straddling"] and f["square_err_deg"] <= L.SQUARE_DEG and f["in_section"]
                      and not f["in_path"] and rest)
            gate_sets[n] = is_set
            rows[n] = {
                "set": is_set,
                "at_rest": rest,
                "standing": f["standing"],
                "straddling": f["straddling"],
                "square_err_deg": round(f["square_err_deg"], 2),
                "in_section": f["in_section"],
                "in_path": f["in_path"],
                "centre_mm": [round(f["centre"][0] * 1000, 1), round(f["centre"][1] * 1000, 1)],
                "off_centre_mm": round(f["off_centre_mm"], 2),
                "path_gap_mm": round(f["path_gap_mm"], 2),
                "foot_low_mm": round(f["foot_low_mm"], 2),
                "foot_high_mm": round(f["foot_high_mm"], 2),
            }
        blocked = any(f["in_path"] for f in facts.values())
        n_set = sum(gate_sets.values())
        spacing = abs(facts["gate_left"]["centre"][1] - facts["gate_right"]["centre"][1])
        bonus = 0.0
        if n_set == 2:
            if L.SPACING_FULL[0] - 1e-9 <= spacing <= L.SPACING_FULL[1] + 1e-9:
                bonus = 0.5
            elif L.SPACING_PART[0] - 1e-9 <= spacing <= L.SPACING_PART[1] + 1e-9:
                bonus = 0.25
        score = 0.0 if blocked else 0.25 * n_set + bonus
        details = {
            "valid_state": True,
            "blocked": blocked,
            "n_set": n_set,
            "spacing_mm": round(spacing * 1000, 2),
            "gates": rows,
        }
        return Result(score, bool(score == 1.0), details)
