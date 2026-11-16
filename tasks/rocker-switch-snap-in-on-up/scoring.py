"""The scorer, as plain functions of the switch's state and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyState of the switch (base frame at the centre of the flange's rear face, z along its axis toward the
rocker, x toward its printed I half, y along the rocker's long side). The panel is fixed, so its numbers are constants
(layout.py). The arms are read only for the let-go check.

    F1 FELL        latched: the switch's lowest point ever more than 10 mm below the table top (off the table)
    IN HOLE        the flange's centre within 2.5 mm (x) and 3 mm (y) of the cutout's centre (top view)
    GAP            the flange's rear face above the plate's top, at its highest corner
    SQUARE         the switch's axis within 3 degrees of the panel normal (the vertical)
    ORIENTATION    the I half within 20 degrees of the panel's top-edge direction (+x): "top"; within 20 degrees of
                   the opposite: "bottom"; otherwise "neither"
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from the switch
    STILL          over the last 1 s the switch moved under 0.5 mm and turned under 0.3 degrees

    0      F1 ever fired, or not released and still (a held switch earns nothing), or not in the hole
    1.0    in: gap <= 0.5 mm and square, I half toward the top edge
    0.5    in and square, turned half a turn (I half toward the bottom edge)
    0.25   in the hole, proud by up to 3 mm (or crooked up to 8 degrees, or turned neither way) otherwise
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SWITCH = "switch"
PARTS = (SWITCH,)
_UP_WORLD = np.array([0.0, 0.0, 1.0])
_TOP_DIR = np.array([1.0, 0.0, 0.0])  # the panel's top edge is the far edge (+x)


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b) / ((float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _box_pts(sx, sy, z0, z1):
    return [(ix * sx / 2, iy * sy / 2, z) for ix in (-1, 1) for iy in (-1, 1) for z in (z0, z1)]


# The switch's points in its own frame: the body's rear face (the lowest points), the flange (its rear face is z 0)
# and the rocker's top.
BODY_PTS = np.array(_box_pts(L.BODY_X - 2 * L.CHAMFER, L.BODY_Y - 2 * L.CHAMFER, -L.DEPTH, -L.DEPTH + L.CHAMFER)
                    + _box_pts(L.BODY_X, L.BODY_Y, -L.DEPTH + L.CHAMFER, 0.0))
FLANGE_REAR = np.array(_box_pts(L.FL_X, L.FL_Y, 0.0, 0.0)[::2])  # the four corners of the flange's rear face
FLANGE_PTS = np.array(_box_pts(L.FL_X, L.FL_Y, 0.0, L.FL_T))
ROCKER_PTS = np.array(_box_pts(L.RK_X, L.RK_Y, L.FL_T, L.TOP_OF_ROCKER))
SWITCH_PTS = np.vstack([BODY_PTS, FLANGE_PTS, ROCKER_PTS])


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: np.ndarray  # the switch frame's origin
    rot: np.ndarray
    pts: np.ndarray  # (N, 3) points of the switch
    rear: np.ndarray  # (4, 3) corners of the flange's rear face


def read(states) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        s = states.get(SWITCH)
        if s is None:
            return None
        o = np.asarray(s.origin, dtype=float).ravel()
        r = rotation(s.quat_xyzw)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
            return None
        o = o - np.array([0.0, 0.0, L.TOP])
        return Reading(o, r, o + SWITCH_PTS @ r.T, o + FLANGE_REAR @ r.T)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


# -- the readings ------------------------------------------------------------------------------------------------------
def tilt_deg(r: Reading) -> float:
    """The angle (deg) between the switch's axis and the panel normal (the vertical)."""
    return angle_deg(r.rot @ _UP_WORLD, _UP_WORLD)


def gap(r: Reading) -> float:
    """The flange's rear face above the plate's top (m), at its highest corner."""
    return float(np.max(r.rear[:, 2])) - L.PLATE_TOP


def orient(r: Reading) -> tuple[float, str]:
    """(angle of the I half from the panel's top-edge direction in degrees, "top" / "bottom" / "neither")."""
    a = angle_deg(r.rot @ np.array([1.0, 0.0, 0.0]), _TOP_DIR)
    if a <= L.ORIENT_DEG + L.EPS:
        return a, "top"
    if a >= 180.0 - L.ORIENT_DEG - L.EPS:
        return a, "bottom"
    return a, "neither"


def in_hole(r: Reading) -> bool:
    return bool(abs(r.pos[0] - L.PX) <= L.IN_HOLE_X + L.EPS and abs(r.pos[1] - L.PY) <= L.IN_HOLE_Y + L.EPS)


def fell(r: Reading) -> bool:
    return float(np.min(r.pts[:, 2])) < L.FALL_Z


def tier(latched: bool, settled: bool, hole: bool, g: float, tilt: float, ori: str) -> float:
    if latched or not settled or not hole:
        return 0.0
    if g <= L.SEAT_TOL + L.EPS and tilt <= L.SQUARE_DEG + L.EPS:
        if ori == "top":
            return 1.0
        if ori == "bottom":
            return 0.5
    if g <= L.PROUD_MAX + L.EPS and tilt <= L.CROOKED_DEG + L.EPS:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


class Tracker:
    """The latch (F1), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return float(np.linalg.norm(r.pos - np.array([L.SW_X0, L.SW_Y0, L.SW_Z0]))) <= L.START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The switch put back on its starting mark with a jump no trial makes in one step (a reset)."""
        return cls.at_start(now) and float(np.linalg.norm(now.pos - prev.pos)) >= L.RESET_JUMP

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        if self.latch["F1"] is None and fell(r):
            self.latch["F1"] = self.steps
        try:
            dist = states[SWITCH].distance
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, hands = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                hands = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, hands)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the switch from where it was at the window's
        start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.pos - first.pos)))
            turn = max(turn, turn_deg(first.rot, x.rot))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, hands) -> Result:
        g, tilt = gap(r), tilt_deg(r)
        ang, ori = orient(r)
        hole = in_hole(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, hole, g, tilt, ori)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "gap_mm": _mm(g), "tilt_deg": round(tilt, 2), "i_from_top_deg": round(ang, 1), "orientation": ori,
            "in_hole": hole, "switch_mm": _mmv(r.pos), "switch_yaw_deg": round(yaw_deg(r.rot), 1),
            "from_cutout_mm": _mmv([r.pos[0] - L.PX, r.pos[1] - L.PY]),
            "lowest_mm": _mm(float(np.min(r.pts[:, 2]))),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(hands),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
