"""The scorer, as plain functions of the picture's, the carton's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyState of the two loose parts: the picture frame (body origin at its bottom back edge, its shapes drawn
leaning 15 degrees, see layout.py) and the carton (origin at the middle of its bottom face).

    FALL        (latched for the whole trial) the picture's centre of mass drops faster than 0.3 m/s between two
                control steps (15 mm in 50 ms), or the picture leaves the table. A picture let go from leaning on
                the carton slaps down flat at about 1.0 m/s; set down by hand or tipped onto the wall it moves under
                0.1 m/s.
    CARTON OFF  (latched) the carton's centre outside the table's outline, or its lowest point 10 mm below the top
    CLEAR       the picture is 5 mm or more from the carton (no longer leaning on it)
    PLACED      the picture is clear, lies with its lowest point within 5 mm of the table, and EITHER leans 5 to 25
                degrees from vertical back toward the wall with its top back edge within 10 mm of the wall's face
                (and within the wall's width), OR lies flat (its face within 10 degrees of horizontal)
    SHELVED     the carton's four base corners are over the shelf's top and within 5 mm of it, upright within 10 deg
    RELEASED    for the last 1 s both grippers' finger zones are 30 mm or more from the picture (PLACED) or from the
                carton (SHELVED) (rigstate.released; unreadable arms: not released)
    STILL       over the last 1 s the picture's centre of mass, or the carton's origin, moved under 2 mm and turned
                under 1 degree

    0      FALL, or CARTON OFF, at any time; or nothing of the below
    1.0    the carton SHELVED, released and still; and the picture PLACED, released and still
    0.75   the carton SHELVED, released and still; the picture not placed (still held, or standing on its edge)
    0.5    the picture PLACED, released and still; the carton not shelved
    0.25   the picture CLEAR of the carton (held or standing), nothing else done
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

FRAME = "picture_frame"
CARTON = "carton"
PARTS = (FRAME, CARTON)


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not n > 0 or not math.isfinite(n):
        raise ValueError("bad quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one step (m, deg; z above the table top)."""
    frame_origin: np.ndarray
    frame_u: np.ndarray  # the picture's upright (U) frame in the world: columns x_U (glass to back), y_U, z_U
    com: np.ndarray  # the picture's centre of mass
    frame_low: float  # its lowest point
    lean: float  # deg from vertical of its height axis
    lean_x: float  # the height axis' x component: > 0 leaning back toward the wall
    face_tilt: float  # deg of its face from horizontal (0: lying flat)
    top_back: np.ndarray  # the middle of its top back edge
    carton_origin: np.ndarray
    carton_rot: np.ndarray
    carton_low: float
    carton_centre: np.ndarray
    frame_centre: np.ndarray


def read(states) -> Reading | None:
    """A Reading, or None when either state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        f, c = states.get(FRAME), states.get(CARTON)
        if f is None or c is None:
            return None
        fo = np.asarray(f.origin, dtype=float).ravel() - np.array([0.0, 0.0, L.TOP])
        co = np.asarray(c.origin, dtype=float).ravel() - np.array([0.0, 0.0, L.TOP])
        fr, cr = rotation(f.quat_xyzw), rotation(c.quat_xyzw)
        if fo.size != 3 or co.size != 3:
            return None
        clo = np.asarray(c.low, dtype=float).ravel()
        cpos = np.asarray(c.pos, dtype=float).ravel()
        fpos = np.asarray(f.pos, dtype=float).ravel()
        if clo.size != 3 or cpos.size != 3 or fpos.size != 3:
            return None
        if not all(np.all(np.isfinite(v)) for v in (fo, co, fr, cr, clo, cpos, fpos)):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    u = fr @ L.R0
    corners = fo + L.frame_corners_u() @ u.T
    com = fo + u @ L.COM_U
    up, face = u[:, 2], u[:, 0]
    lean = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    face_tilt = math.degrees(math.asin(max(-1.0, min(1.0, abs(float(face[2]))))))
    face_tilt = 90.0 - face_tilt
    top_back = fo + u @ np.array([0.0, 0.0, L.FR_H])
    return Reading(fo, u, com, float(corners[:, 2].min()), lean, float(up[0]), face_tilt, top_back, co, cr,
                   float(clo[2] - L.TOP), cpos[:2], fpos[:2])


def off_table(xy, low) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return not (L.TABLE_X0 <= x <= L.TABLE_X1 and L.TABLE_Y0 <= y <= L.TABLE_Y1) or low < -L.OFF_TABLE_DROP


def frame_carton_gap(states) -> float:
    """The smallest distance (m) between the picture and the carton (0 if touching), from both sides' samples."""
    f, c = states[FRAME], states[CARTON]
    fr = np.asarray(f.origin, dtype=float), rotation(f.quat_xyzw) @ L.R0
    cr = np.asarray(c.origin, dtype=float), rotation(c.quat_xyzw)
    fpts = fr[0] + L.frame_samples_u() @ fr[1].T
    cpts = cr[0] + L.carton_samples_local() @ cr[1].T
    return float(min(np.min(c.distance(fpts)), np.min(f.distance(cpts))))


def on_wall(r: Reading) -> bool:
    if not (L.LEAN_MIN - L.EPS <= r.lean <= L.LEAN_MAX + L.EPS and r.lean_x > 0.0):
        return False
    tb = r.top_back
    half = abs(r.frame_u[1, 1]) * L.FR_W / 2
    if not (L.WALL_Y[0] <= tb[1] - half and tb[1] + half <= L.WALL_Y[1]):
        return False
    return 0.0 <= L.WALL_FACE_X - tb[0] <= L.WALL_TOUCH + L.EPS


def lying_flat(r: Reading) -> bool:
    return r.face_tilt <= L.FLAT_DEG + L.EPS


def placed(r: Reading, gap: float) -> bool:
    if gap < L.CLEAR_OF_CARTON - L.EPS:
        return False
    if not (-0.002 <= r.frame_low <= L.ON_TABLE + L.EPS):
        return False
    return on_wall(r) or lying_flat(r)


def shelved(r: Reading) -> bool:
    pts = r.carton_origin + L.carton_corners_local() @ r.carton_rot.T
    if math.degrees(math.acos(max(-1.0, min(1.0, float(r.carton_rot[2, 2]))))) > L.SHELF_TILT + L.EPS:
        return False
    for p in pts:
        if not (L.SHELF_X[0] <= p[0] <= L.SHELF_X[1] and L.SHELF_YR[0] <= p[1] <= L.SHELF_YR[1]):
            return False
        if abs(p[2] - L.SHELF_TOP) > L.SHELF_Z_TOL + L.EPS:
            return False
    return True


def tier(fall: bool, carton_off: bool, carton_done: bool, frame_done: bool, clear: bool) -> float:
    if fall or carton_off:
        return 0.0
    if carton_done:
        return 1.0 if frame_done else 0.75
    if frame_done:
        return 0.5
    return 0.25 if clear else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The latched FALL and CARTON OFF, the last 1 s of readings (stillness) and of the release checks, and the
    whole-trial reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.hz = control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.fall = False
        self.fall_step = None
        self.fall_why = None
        self.carton_off = False
        self.max_down = 0.0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.free_frame: deque = deque(maxlen=self.window)
        self.free_carton: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (math.hypot(r.frame_origin[0] - L.PIVOT_X, r.frame_origin[1] - L.FRAME_Y) <= L.START_TOL
                and abs(r.lean - math.degrees(L.LEAN0)) <= 3.0
                and math.hypot(r.carton_origin[0] - L.CARTON_START[0], r.carton_origin[1] - L.CARTON_START[1])
                <= L.START_TOL and abs(r.carton_origin[2]) <= 0.005)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """Both parts put back on their start with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return (float(np.linalg.norm(now.com - prev.com)) >= L.RESET_JUMP
                or float(np.linalg.norm(now.carton_origin - prev.carton_origin)) >= L.RESET_JUMP)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, the 1 s windows start again (latches kept)
            self.hist.clear()
            self.free_frame.clear()
            self.free_carton.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "fall": self.fall})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        down = None
        if self.prev is not None:
            down = (self.prev.com[2] - r.com[2]) * self.hz
            self.max_down = max(self.max_down, down)
            if down > L.FALL_SPEED + L.EPS and not self.fall:
                self.fall, self.fall_step, self.fall_why = True, self.steps, f"dropped {down:.2f} m/s"
        if off_table(r.frame_centre, r.frame_low) and not self.fall:
            self.fall, self.fall_step, self.fall_why = True, self.steps, "off the table"
        if off_table(r.carton_centre, r.carton_low):
            self.carton_off = True
        self.hist.append(r)
        self.prev = r
        try:
            gap = frame_carton_gap(states)
        except Exception:  # noqa: BLE001 - no gap: treated as touching
            gap = -1.0
        self.free_frame.append(self._released(arms, states, FRAME))
        self.free_carton.append(self._released(arms, states, CARTON))
        self.last = self.judge(r, gap, arms, states, down)
        return self.last

    @staticmethod
    def _released(arms, states, name) -> bool:
        try:
            return bool(rigstate.released(arms, states[name].distance, L.LET_GO))
        except Exception:  # noqa: BLE001 - fail closed
            return False

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the picture (its centre of mass) and of the
        carton (its origin) from where they were at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        a = h[0]
        fm = max(float(np.linalg.norm(x.com - a.com)) for x in h[1:])
        ft = max(turn_deg(a.frame_u, x.frame_u) for x in h[1:])
        cm = max(float(np.linalg.norm(x.carton_origin - a.carton_origin)) for x in h[1:])
        ct = max(turn_deg(a.carton_rot, x.carton_rot) for x in h[1:])
        return {"frame_move": fm, "frame_turn": ft, "carton_move": cm, "carton_turn": ct}

    def judge(self, r: Reading, gap: float, arms, states, down) -> Result:
        m = self.motion()
        full = len(self.free_frame) == self.window
        frame_still = bool(m is not None and m["frame_move"] <= L.STILL_MOVE + L.EPS
                           and m["frame_turn"] <= L.STILL_TURN + L.EPS)
        carton_still = bool(m is not None and m["carton_move"] <= L.STILL_MOVE + L.EPS
                            and m["carton_turn"] <= L.STILL_TURN + L.EPS)
        frame_free = full and all(self.free_frame)
        carton_free = full and all(self.free_carton)
        is_placed = placed(r, gap)
        is_shelved = shelved(r)
        frame_done = is_placed and frame_free and frame_still
        carton_done = is_shelved and carton_free and carton_still
        clear = gap >= L.CLEAR_OF_CARTON - L.EPS
        level = tier(self.fall, self.carton_off, carton_done, frame_done, clear)
        details = {
            "valid_state": True, "score": level, "fall": self.fall, "fall_step": self.fall_step,
            "fall_why": self.fall_why, "carton_off": self.carton_off,
            "down_mps": None if down is None else round(down, 3), "max_down_mps": round(self.max_down, 3),
            "lean_deg": round(r.lean, 2), "lean_back": r.lean_x > 0, "face_tilt_deg": round(r.face_tilt, 2),
            "frame_low_mm": _mm(r.frame_low), "com_mm": [_mm(v) for v in r.com],
            "wall_gap_mm": _mm(L.WALL_FACE_X - r.top_back[0]), "gap_to_carton_mm": _mm(gap),
            "on_wall": on_wall(r), "flat": lying_flat(r), "placed": is_placed, "shelved": is_shelved,
            "carton_mm": [_mm(v) for v in r.carton_origin],
            "frame_released_1s": frame_free, "carton_released_1s": carton_free,
            "frame_still_1s": frame_still, "carton_still_1s": carton_still,
            "motion": None if m is None else {k: round(v * (1000.0 if "move" in k else 1.0), 3) for k, v in m.items()},
            "arms_readable": rigstate.readable(arms),
        }
        return Result(level, bool(level == 1.0), details)
