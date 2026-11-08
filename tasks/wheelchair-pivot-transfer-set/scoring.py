"""The scorer, as plain functions of the wheelchair's state. NumPy only; no simulator needed.

The rule is about where the chair ends and which way it faces: beside the bed next to Mrs Hale on her strong (right,
+y) side (its nearest corner at most 40 mm along the bed past her strong-side line), close to the bed, its front
toward the foot of the bed and angled out 30-50 degrees. All of it is read from the
chair's pose, as a top photo with a protractor and a tape from the bed's side would read it. The only memory the
scorer keeps is the chair's last half second of positions, to tell whether it is still.

    Tracker.update(chair) -> Result(score, success, details)

``chair`` is anything with ``origin`` (the chair frame's origin: the middle of its footprint, on the floor),
``quat_xyzw``, ``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned
bounding box), as mechkit's ``AssemblyState`` has. A state with a non-finite number or a zero quaternion (a simulator
blow-up) scores 0 for that step and leaves the tracker as it was.

Angles: the chair's 'front' is its local +x (the footplates). Its angle to the bed, alpha, is measured from the
bed's side pointing to the foot of the bed (-y), positive turned out toward the room (-x): alpha = atan2(-f_x, -f_y)
for the front's horizontal direction f. 0 = parallel to the bed facing the foot; 90 = square to the bed facing the
room; negative = turned in toward the bed; over 90 = facing toward the head end.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 steps: the path is summed over the last WINDOW moves
ANGLE_EPS = 1e-6  # degrees: rounding slack at the 0 and 90 degree limits (exactly square to the bed is allowed)


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


def valid(chair) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(chair, k), dtype=float).ravel()
                 for k in ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(chair) -> float:
    """Angle between the chair's up axis and the vertical (degrees, 0..180: upside down is 180)."""
    zz = rotation(chair.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def heading_rad(chair) -> float:
    """Which way its front points seen from above (rad, counter-clockwise from +x); NaN if it points straight up or
    down."""
    f = rotation(chair.quat_xyzw)[:, 0]
    if math.hypot(f[0], f[1]) < 1e-9:
        return float("nan")
    return math.atan2(float(f[1]), float(f[0]))


def alpha_deg(chair) -> float:
    """The chair's angle to the bed (degrees, -180..180): 0 parallel facing the foot of the bed, + turned out."""
    h = heading_rad(chair)
    if not math.isfinite(h):
        return float("nan")
    return math.degrees(math.atan2(-math.cos(h), -math.sin(h)))


def footprint(chair) -> np.ndarray:
    """The four corners of its 95 x 63 mm footprint (the rectangle on the floor under its wheels and castors), (4, 2):
    front-left, front-right, rear-left, rear-right."""
    r = rotation(chair.quat_xyzw)
    o = np.asarray(chair.origin, dtype=float)
    local = np.array([(su * L.FOOT_U, sv * L.FOOT_V, 0.0) for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
    return (o + local @ r.T)[:, :2]


def centre(chair) -> np.ndarray:
    """The middle of its footprint, seen from above."""
    return np.asarray(chair.origin, dtype=float)[:2].copy()


def gap(chair) -> float:
    """The gap g: from the bed's side face to the footprint's corner nearest it, square to the bed (m; negative =
    past the line of the bed's side)."""
    return float(L.BED_SIDE_X - footprint(chair)[:, 0].max())


def lowest(chair) -> float:
    """Height of its lowest point above the floor (m)."""
    return float(np.asarray(chair.low, dtype=float)[2] - L.TOP)


def angle_credit(alpha: float) -> float:
    """0.5 for 30-50 degrees, 0.25 for 15-30 or 50-65, else 0."""
    lo, hi = L.ANGLE_FULL_DEG
    plo, phi = L.ANGLE_PART_DEG
    if lo <= alpha <= hi:
        return 0.5
    if plo <= alpha <= phi:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The chair's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=WINDOW + 1)  # (centre xy, heading) at the last valid control steps
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(chair) -> bool:
        """The starting layout: the chair upright on the floor on its mark, facing as it starts."""
        on_mark = float(np.hypot(*(centre(chair) - np.asarray(L.CHAIR_START)))) < L.START_TOL
        turn = abs(math.remainder(heading_rad(chair) - L.CHAIR_START_HEADING, 2 * math.pi))
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(chair) < 2.0
                    and abs(lowest(chair)) < L.FLOOR_TOL)

    def path(self) -> float:
        """How far the chair moved over the last 0.5 s: its centre's path plus its turns counted at the footprint's
        corners (m). Infinite until half a second has been seen."""
        h = list(self.history)
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (c0, a0), (c1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.hypot(*(c1 - c0))) + L.TURN_RADIUS * turn
        return total

    def update(self, chair) -> Result:
        self.steps += 1
        if not valid(chair):  # a blow-up: score 0 for this step, keep everything as it was
            self.last = self.result(chair)
            return self.last
        c = centre(chair)
        if self.history and self.at_start(chair):
            if float(np.hypot(*(c - self.history[-1][0]))) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()
                self.steps = 1
        self.history.append((c, heading_rad(chair)))
        self.last = self.result(chair)
        return self.last

    def result(self, chair) -> Result:
        if not valid(chair):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        tilt, low, path = tilt_deg(chair), lowest(chair), self.path()
        alpha, g = alpha_deg(chair), gap(chair)
        corners = footprint(chair)
        c = centre(chair)
        upright = tilt <= L.TILT_MAX_DEG
        on_floor = abs(low) <= L.FLOOR_TOL
        still = path < L.STILL_PATH
        min_y = float(corners[:, 1].min())
        if not upright:
            reason = "tipped"
        elif not on_floor:
            reason = "off_the_floor"
        elif not still:
            reason = "moving"
        elif min_y < L.Y_STRONG:
            reason = "weak_side_or_across_her"
        elif min_y > L.Y_BESIDE:
            reason = "too_far_along_the_bed_from_her"
        elif g < -L.BED_LINE_TOL:
            reason = "past_the_bed_line"
        elif not alpha >= L.ANGLE_MIN_DEG - ANGLE_EPS:
            reason = "turned_in_to_the_bed"
        elif alpha > L.ANGLE_MAX_DEG + ANGLE_EPS:
            reason = "facing_the_head_end"
        elif g > L.GAP_MAX:
            reason = "too_far_from_the_bed"
        else:
            reason = None
        if reason is None:
            closeness = 0.5 if g <= L.GAP_CLOSE else 0.0
            angle = angle_credit(alpha)
            score = closeness + angle
        else:
            closeness = angle = 0.0
            score = 0.0
        details = {
            "valid_state": True,
            "zero_reason": reason,
            "set_down": bool(upright and on_floor and still),
            "upright": upright,
            "on_floor": on_floor,
            "still": still,
            "closeness": closeness,
            "angle_credit": angle,
            "alpha_deg": round(alpha, 2) if math.isfinite(alpha) else None,
            "gap_mm": round(g * 1000, 1),
            "min_corner_y_mm": round(min_y * 1000, 1),
            "centre_mm": [round(float(v) * 1000, 1) for v in c],
            "tilt_deg": round(tilt, 2),
            "lowest_mm": round(low * 1000, 1),
            "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
        }
        return Result(score, bool(score == 1.0), details)
