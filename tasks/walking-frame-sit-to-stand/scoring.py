"""The scorer, as plain functions of the walking frame's state. NumPy only; no simulator needed.

The rule is about where the frame ends and which way it faces: standing on its four feet in front of Mr Okafor,
square to his chair with its open side toward him, centred on him, its nearest legs a short step from his toes. All
of it is read from the frame's pose, as a top photo with a protractor and a rule would read it. The only memory the
scorer keeps is the frame's last half second of positions, to tell whether it is still.

    Tracker.update(frame) -> Result(score, success, details)

``frame`` is anything with ``origin`` (the frame's origin: the middle of its footprint, on the floor),
``quat_xyzw``, ``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (world-aligned
bounding box), as mechkit's ``AssemblyState`` has. A state with a non-finite number or a zero quaternion (a simulator
blow-up) scores 0 for that step and leaves the tracker as it was.

Conditions (each with a full band and a wider loose band; outside its loose band the score is 0):
- square: the turn, the angle seen from above between the frame's forward direction (from its open side to its
  crossbar) and -x (toward the arms, away from him): full <= 15 degrees, loose <= 30. Side-on is 90, crossbar
  toward him 180.
- centred: the footprint centre's distance from his centre line (y 0): full <= 20 mm, loose <= 50 mm.
- toe gap d: from the line of his toes (x 0.350) to the frame's footprint corner nearest it, along x: full 10-50 mm,
  loose 0-100 mm. Negative = a leg past the toe line (around his feet).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 steps: the path is summed over the last WINDOW moves
FOOT_CORNERS = np.array([(su * L.LEG_U + cu * L.TUBE / 2, sv * L.LEG_V + cv * L.TUBE / 2, 0.0)
                         for su in (1, -1) for sv in (1, -1) for cu in (1, -1) for cv in (1, -1)])  # 4 feet x 4


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


def valid(frame) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(frame, k), dtype=float).ravel()
                 for k in ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(frame) -> float:
    """Angle between the frame's up axis and the vertical (degrees, 0..180: upside down is 180)."""
    zz = rotation(frame.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def heading_rad(frame) -> float:
    """Which way its forward direction (open side -> crossbar) points seen from above (rad, counter-clockwise from
    +x); NaN if it points straight up or down."""
    f = rotation(frame.quat_xyzw)[:, 0]
    if math.hypot(f[0], f[1]) < 1e-9:
        return float("nan")
    return math.atan2(float(f[1]), float(f[0]))


def turn_deg(frame) -> float:
    """How far it is turned from square (degrees, 0..180): 0 = forward direction at -x (open side toward him),
    90 = side-on, 180 = crossbar toward him. NaN if its forward direction points straight up or down."""
    h = heading_rad(frame)
    if not math.isfinite(h):
        return float("nan")
    return abs(math.degrees(math.remainder(h - math.pi, 2 * math.pi)))


def footprint(frame) -> np.ndarray:
    """The four corners of its 56 x 70 mm footprint (the outer corners of its four legs), (4, 2): front-left,
    front-right, back-left, back-right (front = its crossbar side)."""
    r = rotation(frame.quat_xyzw)
    o = np.asarray(frame.origin, dtype=float)
    local = np.array([(su * L.FOOT_U, sv * L.FOOT_V, 0.0) for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
    return (o + local @ r.T)[:, :2]


def centre(frame) -> np.ndarray:
    """The middle of its footprint, seen from above."""
    return np.asarray(frame.origin, dtype=float)[:2].copy()


def toe_gap(frame) -> float:
    """d: from the line of his toes to the footprint corner nearest it, along x (m; negative = past the line)."""
    return float(L.TOE_X - footprint(frame)[:, 0].max())


def off_centre(frame) -> float:
    """The footprint centre's distance from his centre line (m)."""
    return abs(float(centre(frame)[1]) - L.Y_C)


def feet_heights(frame) -> np.ndarray:
    """Height above the floor of the lowest point of each of its four feet (m), (4,)."""
    r = rotation(frame.quat_xyzw)
    o = np.asarray(frame.origin, dtype=float)
    z = (o + FOOT_CORNERS @ r.T)[:, 2] - L.TOP
    return z.reshape(4, 4).min(axis=1)


def band(value: float, full, loose):
    """'full', 'loose' or 'out' for a value against a full band and a loose band, each (lo, hi) inclusive."""
    if not math.isfinite(value):
        return "out"
    if full[0] <= value <= full[1]:
        return "full"
    if loose[0] <= value <= loose[1]:
        return "loose"
    return "out"


def bands(frame) -> dict:
    """Each condition's band: square, centred, toe gap."""
    return {
        "square": band(turn_deg(frame), (0.0, L.SQUARE_FULL_DEG), (0.0, L.SQUARE_LOOSE_DEG)),
        "centred": band(off_centre(frame), (0.0, L.CENTRE_FULL), (0.0, L.CENTRE_LOOSE)),
        "toe_gap": band(toe_gap(frame), L.DIST_FULL, L.DIST_LOOSE),
    }


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The frame's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=WINDOW + 1)  # (centre xy, heading) at the last valid control steps
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(frame) -> bool:
        """The starting layout: the frame standing on its mark, side-on as it starts."""
        on_mark = float(np.hypot(*(centre(frame) - np.asarray(L.FRAME_START)))) < L.START_TOL
        turn = abs(math.remainder(heading_rad(frame) - L.FRAME_START_HEADING, 2 * math.pi))
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(frame) < 2.0
                    and np.all(np.abs(feet_heights(frame)) < L.FLOOR_TOL))

    def path(self) -> float:
        """How far the frame moved over the last 0.5 s: its centre's path plus its turns counted at the footprint's
        corners (m). Infinite until half a second has been seen."""
        h = list(self.history)
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (c0, a0), (c1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.hypot(*(c1 - c0))) + L.TURN_RADIUS * turn
        return total

    def update(self, frame) -> Result:
        self.steps += 1
        if not valid(frame):  # a blow-up: score 0 for this step, keep everything as it was
            self.last = self.result(frame)
            return self.last
        c = centre(frame)
        if self.history and self.at_start(frame):
            if float(np.hypot(*(c - self.history[-1][0]))) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()
                self.steps = 1
        self.history.append((c, heading_rad(frame)))
        self.last = self.result(frame)
        return self.last

    def result(self, frame) -> Result:
        if not valid(frame):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        feet = feet_heights(frame)
        path = self.path()
        turn, off, d = turn_deg(frame), off_centre(frame), toe_gap(frame)
        b = bands(frame)
        on_feet = bool(np.all(np.abs(feet) <= L.FLOOR_TOL))
        still = path < L.STILL_PATH
        if not on_feet:
            reason = "not_on_its_four_feet"
        elif not still:
            reason = "moving"
        elif b["square"] == "out":
            reason = "not_square_side_on_or_crossbar_toward_him"
        elif b["centred"] == "out":
            reason = "off_to_the_side"
        elif d < L.DIST_LOOSE[0]:
            reason = "past_his_toes"
        elif b["toe_gap"] == "out":
            reason = "too_far_from_his_toes"
        else:
            reason = None
        n_loose = sum(v == "loose" for v in b.values())
        score = L.SCORE_BY_LOOSE[n_loose] if reason is None else 0.0
        details = {
            "valid_state": True,
            "zero_reason": reason,
            "set_down": bool(on_feet and still),
            "on_feet": on_feet,
            "still": still,
            "square": b["square"],
            "centred": b["centred"],
            "toe_gap": b["toe_gap"],
            "n_loose": n_loose if reason is None else None,
            "turn_deg": round(turn, 2) if math.isfinite(turn) else None,
            "off_centre_mm": round(off * 1000, 1),
            "toe_gap_mm": round(d * 1000, 1),
            "centre_mm": [round(float(v) * 1000, 1) for v in centre(frame)],
            "feet_mm": [round(float(v) * 1000, 1) for v in feet],
            "tilt_deg": round(tilt_deg(frame), 2),
            "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
        }
        return Result(score, bool(score == 1.0), details)
