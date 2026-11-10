"""The scorer, as plain functions of the lens's and the screen's states. NumPy only; no simulator needed.

An element (the lens in its holder, or the screen) counts when it stands on the bench: upright, square across it
(the screen's white face toward the lamp), centred on it, at rest. With both counting, the image of the lamp's arrow
is sharp when the screen is within 10 mm of where the lens law puts the image for the lens's actual position:
v* = f u / (u - f), with u the arrow-to-lens distance along the bench, v the lens-to-screen distance and f = 100 mm.
A sharp image the same size as the arrow (v / u from 0.9 to 1.1: lens at 2f, screen 2f beyond it) scores 1.0; sharp
at another size 0.5; not sharp, but one counting element at its same-size mark 0.25. Either element tipped over or
fallen off the bench scores 0. The scorer keeps a short history of each element's position (to tell whether it is at
rest) and is fed one observation per control step by ``task.success()``:

    Tracker.update({"lens": state, "screen": state}) -> Result(score, success, details)

A state is anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has, plus ``origin`` (the element's base frame: the centre of its rider's
underside), as mechkit's ``AssemblyState`` has; without ``origin`` it is worked out from ``pos``. A state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.
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


def bbox_centre(name: str) -> np.ndarray:
    """Local centre of the box around every shape of the element (what ``pos`` is, at rest)."""
    pts = np.array([np.asarray(c) + s * np.asarray(h) / 2 for c, h in L.BOXES[name] for s in (-1, 1)])
    return (pts.min(axis=0) + pts.max(axis=0)) / 2


def frame(name: str, g) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the element's base frame in the world: ``origin`` if the state has it (mechkit), else
    the centre of its box less the box centre's offset from the base frame."""
    r = rotation(g.quat_xyzw)
    origin = getattr(g, "origin", None)
    if origin is not None:
        return r, np.asarray(origin, dtype=float)
    return r, np.asarray(g.pos, dtype=float) - r @ bbox_centre(name)


def to_world(r: np.ndarray, o: np.ndarray, local) -> np.ndarray:
    return np.asarray(local, dtype=float) @ r.T + o


def rider_corners(name: str, r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(4, 3): the corners of the rider's underside."""
    hx, hy = L.RIDERS[name][0] / 2, L.RIDERS[name][1] / 2
    return to_world(r, o, [(sx * hx, sy * hy, 0.0) for sx in (-1, 1) for sy in (-1, 1)])


def track_points(name: str, r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(5, 3): what the rest test follows: the rider's four bottom corners and the top of the element."""
    return np.vstack([rider_corners(name, r, o), to_world(r, o, [(0.0, 0.0, L.HEIGHTS[name])])])


def tilt_deg(r: np.ndarray) -> float:
    """How far the element leans from upright (its local z from the vertical), degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def heading_deg(r: np.ndarray) -> float:
    """Heading of the element's width (local x) seen from above, degrees (0 = square across the bench)."""
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def square_err_deg(name: str, r: np.ndarray) -> float:
    """How far the element is turned from square across the bench, seen from above. The lens is the same either way
    round (0..90 degrees); the screen's white face must face the lamp (+y), so turned end for end it is 180 off."""
    a = heading_deg(r)
    if name == "lens":
        a %= 180.0
        return float(min(a, 180.0 - a))
    return float(abs((a + 180.0) % 360.0 - 180.0))


def optic_point(name: str, r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """The lens's centre, or the point of the screen's white face on the optical axis (at the axis height)."""
    local = (0.0, 0.0, L.LENS_Z) if name == "lens" else (0.0, L.FACE_Y, L.AXIS_Z)
    return to_world(r, o, local)


def element_facts(name: str, g) -> dict:
    """Everything about one element's pose that the rule needs (rest is the tracker's)."""
    r, o = frame(name, g)
    heights = rider_corners(name, r, o)[:, 2] - L.BENCH_TOP
    over = bool(abs(o[0] - L.BENCH_X) <= L.BENCH_W / 2 and L.BENCH_Y0 <= o[1] <= L.BENCH_Y1)
    tilt = tilt_deg(r)
    point = optic_point(name, r, o)
    return {
        "centre": (float(o[0]), float(o[1]), float(o[2])),
        "tilt_deg": tilt,
        "square_err_deg": square_err_deg(name, r),
        "on_bench": bool(over and abs(float(heights.min())) <= L.STAND_TOL),  # resting on it (a lean is tilt's)
        "off_centre_mm": float((o[0] - L.BENCH_X) * 1000),
        "centred": bool(abs(o[0] - L.BENCH_X) <= L.CENTRE_TOL + 1e-12),
        "tipped": bool(tilt > L.TILT_ZERO_DEG),
        "off_bench": bool(o[2] < L.BENCH_TOP - L.OFF_BENCH_DROP),
        "from_object_mm": float((L.OBJ_Y - point[1]) * 1000),  # along the bench, from the arrow's plane
        "rider_low_mm": float(heights.min() * 1000),
        "rider_high_mm": float(heights.max() * 1000),
    }


def judge(lens_counts: bool, screen_counts: bool, u: float, d: float) -> tuple[float, dict]:
    """The score from the two elements' standing and their distances from the arrow along the bench (metres: u to
    the lens's centre, d to the screen's face), when neither is tipped or off the bench."""
    v = d - u  # the lens-to-screen distance (the screen beyond the lens: v > 0)
    v_star = L.image_distance(u) if u > L.F + 1e-9 else math.inf
    sharp = bool(lens_counts and screen_counts and u > L.F + 1e-9 and v > 0.0 and abs(v - v_star) <= L.SHARP_TOL + 1e-9)
    mag = v / u if u > 1e-9 else math.inf
    same = bool(L.SAME_SIZE[0] - 1e-9 <= mag <= L.SAME_SIZE[1] + 1e-9)
    lens_mark = bool(lens_counts and L.LENS_MARK[0] - 1e-9 <= u <= L.LENS_MARK[1] + 1e-9)
    screen_mark = bool(screen_counts and L.SCREEN_MARK[0] - 1e-9 <= d <= L.SCREEN_MARK[1] + 1e-9)
    if sharp:
        score = 1.0 if same else 0.5
    elif lens_mark or screen_mark:
        score = 0.25
    else:
        score = 0.0
    return score, {
        "u_mm": round(u * 1000, 2),
        "v_mm": round(v * 1000, 2),
        "v_star_mm": round(v_star * 1000, 2) if math.isfinite(v_star) else None,
        "focus_err_mm": round((v - v_star) * 1000, 2) if math.isfinite(v_star) else None,
        "magnification": round(mag, 4) if math.isfinite(mag) else None,
        "sharp": sharp,
        "same_size": bool(sharp and same),
        "lens_at_mark": lens_mark,
        "screen_at_mark": screen_mark,
    }


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: where each element's rider corners and top were over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = int(round(L.REST_WINDOW_S * control_hz))  # 10 steps at 20 Hz: 11 samples span 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hist = {name: deque(maxlen=self.window + 1) for name in L.ELEMENTS}
        self.prev = None  # each element's origin at the previous valid control step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, g) -> bool:
        """The starting layout: this element standing on the bench on its starting mark."""
        r, o = frame(name, g)
        sx, sy = L.START[name]
        on_mark = math.hypot(o[0] - sx, o[1] - sy) < L.START_TOL
        return bool(on_mark and np.all(np.abs(rider_corners(name, r, o)[:, 2] - L.BENCH_TOP) <= L.STAND_TOL))

    def at_rest(self, name: str) -> bool:
        """No corner of the rider and not the element's top has moved more than 2 mm in the last 0.5 s (from
        positions, not the reported velocity, which chatters while a part is gripped)."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return False
        now = h[-1]
        return bool(all(float(np.max(np.linalg.norm(p - now, axis=1))) <= L.REST_MOVE for p in h))

    def update(self, elements: dict) -> Result:
        if not all(valid(elements.get(n)) for n in L.ELEMENTS):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.last = self.result(elements)
            return self.last
        origins = {n: frame(n, elements[n])[1] for n in L.ELEMENTS}
        if self.prev is not None and all(self.at_start(n, elements[n]) for n in L.ELEMENTS):
            jump = max(float(np.linalg.norm(origins[n] - self.prev[n])) for n in L.ELEMENTS)
            if jump > L.RESET_JUMP:  # the simulator put the elements back: a new episode
                self.reset()
        self.steps += 1
        for n in L.ELEMENTS:
            r, o = frame(n, elements[n])
            self.hist[n].append(track_points(n, r, o))
        self.prev = origins
        self.last = self.result(elements)
        return self.last

    def result(self, elements: dict) -> Result:
        if not all(valid(elements.get(n)) for n in L.ELEMENTS):
            return Result(0.0, False, {"valid_state": False})
        facts = {n: element_facts(n, elements[n]) for n in L.ELEMENTS}
        counts, rows = {}, {}
        for n, f in facts.items():
            rest = self.at_rest(n)
            ok = (f["on_bench"] and f["tilt_deg"] <= L.TILT_COUNT_DEG + 1e-9
                  and f["square_err_deg"] <= L.SQUARE_DEG + 1e-9 and f["centred"] and rest)
            counts[n] = bool(ok)
            rows[n] = {
                "counts": bool(ok),
                "at_rest": rest,
                "on_bench": f["on_bench"],
                "tilt_deg": round(f["tilt_deg"], 2),
                "square_err_deg": round(f["square_err_deg"], 2),
                "centred": f["centred"],
                "tipped": f["tipped"],
                "off_bench": f["off_bench"],
                "centre_mm": [round(c * 1000, 1) for c in f["centre"]],
                "off_centre_mm": round(f["off_centre_mm"], 2),
                "from_object_mm": round(f["from_object_mm"], 2),
                "rider_low_mm": round(f["rider_low_mm"], 2),
                "rider_high_mm": round(f["rider_high_mm"], 2),
            }
        down = any(f["tipped"] or f["off_bench"] for f in facts.values())
        u = facts["lens"]["from_object_mm"] / 1000
        d = facts["screen"]["from_object_mm"] / 1000
        score, optics = judge(counts["lens"], counts["screen"], u, d)
        if down:
            score = 0.0
        details = {"valid_state": True, "down": down, **optics, "elements": rows}
        return Result(score, bool(score == 1.0), details)
