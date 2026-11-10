"""The scorer, as plain functions of the food shield's state. NumPy only; no simulator needed.

The shield is back in service when it stands on the counter on both end panels, square to the bar with its lip
toward the customers, spanning the whole row of pans, at rest, and its front lip overhangs the pans' customer-side
edge by the set amount (read with a tape at each end of the row); centring it on the row earns the last quarter.
The scorer keeps a short history of where the panels' feet were (to tell whether the shield is at rest) and is fed
one observation per control step by ``task.success()``:

    Tracker.update(shield) -> Result(score, success, details)

``shield`` is anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has, plus ``origin`` (the base frame: the centre of the footprint, on the panels'
undersides), as mechkit's ``AssemblyState`` has; without ``origin`` it is worked out from ``pos``. A state with a
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


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel()
                 for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")]
        origin = getattr(s, "origin", None)
        if origin is not None:
            parts.append(np.asarray(origin, dtype=float).ravel())
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def frame(s) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the shield's base frame in the world: ``origin`` if the state has it (mechkit), else
    the centre of its box less the box centre's offset from the base frame."""
    r = rotation(s.quat_xyzw)
    origin = getattr(s, "origin", None)
    if origin is not None:
        return r, np.asarray(origin, dtype=float)
    return r, np.asarray(s.pos, dtype=float) - r @ np.asarray(L.SHIELD_BBOX_CENTRE, dtype=float)


def to_world(r: np.ndarray, o: np.ndarray, local) -> np.ndarray:
    return np.asarray(local, dtype=float) @ r.T + o


def panel_corners(r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(8, 3): the four corners of each end panel's underside."""
    hx, hy = L.PANEL_SIZE[0] / 2, L.PANEL_SIZE[1] / 2
    pts = [(dx, sy * L.PANEL_Y + dy, 0.0) for sy in (1, -1) for dx in (-hx, hx) for dy in (-hy, hy)]
    return to_world(r, o, pts)


def panel_feet(r: np.ndarray, o: np.ndarray) -> np.ndarray:
    """(2, 3): the centre of each end panel's underside (the left one, local +y, first)."""
    return to_world(r, o, [(0.0, L.PANEL_Y, 0.0), (0.0, -L.PANEL_Y, 0.0)])


def tilt_deg(r: np.ndarray) -> float:
    """Angle between the shield's up axis and the vertical (degrees, 0..180)."""
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2]))))))


def turn_deg(r: np.ndarray) -> float:
    """Heading of the lip's direction (local +x) seen from above, from straight toward the customers (+x),
    -180..180 degrees: a shield turned lip to the staff side reads 180."""
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def overhang_at(r: np.ndarray, o: np.ndarray, y_world: float) -> float | None:
    """The lip's overhang past the pans' customer-side edge (m), read across the bar at ``y_world`` (an end of the
    row of pans): where the line along the lip's outer face, at its mid-height, crosses that y. None if the lip
    does not reach that far along the bar (the shield does not span that end of the row) or runs along x."""
    a = to_world(r, o, (L.LIP_FACE_X, 0.0, L.LIP_MID_Z))
    along = r[:, 1]  # the lip's direction along the bar
    if abs(along[1]) < 1e-6:
        return None
    t = (y_world - a[1]) / along[1]
    if abs(t) > L.LIP_HALF + L.EPS:
        return None
    return float(a[0] + t * along[0] - L.PAN_X1)


def shield_facts(s) -> dict:
    """Everything about the shield's pose that the rule needs (rest is the tracker's)."""
    r, o = frame(s)
    heights = panel_corners(r, o)[:, 2] - L.TOP
    tilt, turn = tilt_deg(r), turn_deg(r)
    standing = bool(np.all(np.abs(heights) <= L.STAND_TOL + L.EPS) and tilt <= L.TILT_MAX_DEG + L.EPS)
    readings = [overhang_at(r, o, y) for y in L.ROW_ENDS]
    spans = all(d is not None for d in readings)
    return {
        "centre": (float(o[0]), float(o[1])),
        "standing": standing,
        "tilt_deg": tilt,
        "turn_deg": turn,
        "square": abs(turn) <= L.TURN_MAX_DEG + L.EPS,
        "spans_row": spans,
        "overhang": readings,
        "panel_low_mm": float(heights.min() * 1000),
        "panel_high_mm": float(heights.max() * 1000),
        "centre_off_mm": float(o[1] * 1000),
    }


def overhang_credit(readings) -> float:
    """0.75 if both readings are 45-75 mm; 0.25 if both are 25-90 mm (not both in the first band); else 0."""
    if any(d is None for d in readings):
        return 0.0
    lo, hi = min(readings), max(readings)
    if L.OVERHANG_FULL[0] - L.EPS <= lo and hi <= L.OVERHANG_FULL[1] + L.EPS:
        return 0.75
    if L.OVERHANG_PART[0] - L.EPS <= lo and hi <= L.OVERHANG_PART[1] + L.EPS:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: where the shield's two panel feet were over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = int(round(L.REST_WINDOW_S * control_hz))  # 10 steps at 20 Hz: 11 samples span 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.prev = None  # the shield's origin at the previous valid control step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(s) -> bool:
        """The starting layout: the shield standing on the counter on its starting mark, square."""
        r, o = frame(s)
        on_mark = math.hypot(o[0] - L.START[0], o[1] - L.START[1]) < L.START_TOL
        flat = bool(np.all(np.abs(panel_corners(r, o)[:, 2] - L.TOP) <= L.STAND_TOL))
        return bool(on_mark and flat and abs(turn_deg(r)) < 5.0)

    def at_rest(self) -> bool:
        """Neither panel's foot has moved more than 3 mm in the last 0.5 s (from positions, not the reported
        velocity, which chatters while a part is gripped)."""
        h = self.hist
        if len(h) < self.window + 1:
            return False
        now = h[-1]
        return bool(all(float(np.max(np.linalg.norm(p - now, axis=1))) <= L.REST_MOVE + L.EPS for p in h))

    def update(self, s) -> Result:
        if not valid(s):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.last = self.result(s)
            return self.last
        r, o = frame(s)
        if self.prev is not None and self.at_start(s) and float(np.linalg.norm(o - self.prev)) > L.RESET_JUMP:
            self.reset()  # the simulator put the shield back: a new episode
        self.steps += 1
        self.hist.append(panel_feet(r, o))
        self.prev = o
        self.last = self.result(s)
        return self.last

    def result(self, s) -> Result:
        if not valid(s):
            return Result(0.0, False, {"valid_state": False})
        f = shield_facts(s)
        rest = self.at_rest()
        zero = not (f["standing"] and f["square"] and f["spans_row"] and rest)
        credit = 0.0 if zero else overhang_credit(f["overhang"])
        centred = abs(f["centre"][1]) <= L.CENTRE_TOL + L.EPS
        score = credit + (0.25 if credit > 0 and centred else 0.0)
        d = [None if v is None else round(v * 1000, 2) for v in f["overhang"]]
        details = {
            "valid_state": True,
            "at_rest": rest,
            "standing": f["standing"],
            "square": f["square"],
            "spans_row": f["spans_row"],
            "overhang_left_mm": d[0],
            "overhang_right_mm": d[1],
            "overhang_credit": credit,
            "centred": centred,
            "centre_mm": [round(f["centre"][0] * 1000, 1), round(f["centre"][1] * 1000, 1)],
            "turn_deg": round(f["turn_deg"], 2),
            "tilt_deg": round(f["tilt_deg"], 2),
            "panel_low_mm": round(f["panel_low_mm"], 2),
            "panel_high_mm": round(f["panel_high_mm"], 2),
        }
        return Result(score, bool(score == 1.0), details)
