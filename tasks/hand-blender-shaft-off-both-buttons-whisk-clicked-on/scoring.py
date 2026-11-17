"""The scorer, as plain functions of the motor part's, the shaft's and the whisk's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyStates of the motor part (base frame at the centre of its coupling face, axis along +y into the body,
two button links ``btn_p`` / ``btn_n`` whose joint value is 0 out and negative in), the shaft and the whisk (base
frame at the centre of their shoulder face, stub toward +y). Every reading is a pose or a joint value; the arms are
read only for the let-go check. The hooks have no state of their own: a hook is behind the lip when its button is
out (within 0.4 mm of rest) while the attachment is home, because a hook that is not behind the lip rides on it.

    SHAFT OFF      not inserted, wherever it lies: inserted means the stub's tip more than 0.5 mm past the coupling
                   face with its axis within 30 mm of the socket's and 30 degrees of its direction
    SHAFT ON MAT   off, its centre over the mat (100 mm each way from the mat's centre) and its lowest point within
                   5 mm of the mat's top
    WHISK HOME     inserted with the shoulder face within 1 mm of the coupling face, the axis within 3 mm of the
                   socket's and 3 degrees square
    WHISK ON       home with BOTH hooks behind the lip (both buttons out)
    WHISK SEATED   home with a hook not behind the lip (a button held in, or the lip on a hook)
    F1 FELL        latched: any part's lowest point more than 30 mm below the table top
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from all three parts
    STILL          over the last 1 s every part moved under 5 mm and turned under 5 degrees

    0      F1 ever fired, or not released and still, or the shaft not off
    1.0    shaft on the mat, whisk on
    0.5    whisk on, the shaft elsewhere; or the shaft on the mat and the whisk seated
    0.25   the shaft off, otherwise (whisk not home, or seated with the shaft elsewhere)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

MOTOR, SHAFT, WHISK = "motor", "shaft", "whisk"
PARTS = (MOTOR, SHAFT, WHISK)
BTN_P, BTN_N = "btn_p", "btn_n"
SHAFT_MID_U = (L.STUB_LEN + L.SH_BELL_U0) / 2  # the shaft's centre along its axis (local y)


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
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # part -> base frame origin
    rot: dict  # part -> rotation
    low: dict  # part -> lowest point's height
    q: dict  # button joint values (m; 0 out, negative in)


def lowest_z(state) -> float:
    """The lowest point of every shape of a state (world z), exact for boxes and cylinders, turned ones included."""
    spec = state.spec
    low = math.inf
    for body in spec.body_names:
        r, p = state.link_pose(body)
        off = spec.body_offset(body)
        for s in spec.body_shapes(body):
            rw = r @ s.rotation()
            c = p + r @ (np.asarray(s.center, dtype=float) - off)
            if s.kind == "box":
                half = np.asarray(s.size, dtype=float) / 2
                z = c[2] - float(np.sum(np.abs(rw[2, :]) * half))
            else:
                k = "XYZ".index(s.axis)
                az = float(rw[2, k])
                z = c[2] - s.height / 2 * abs(az) - s.radius * math.sqrt(max(0.0, 1.0 - az * az))
            low = min(low, z)
    return float(low)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, low, q = {}, {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            lz = lowest_z(s)
            if not math.isfinite(lz):
                return None
            pos[name], rot[name], low[name] = o - np.array([0.0, 0.0, L.TOP]), r, lz - L.TOP
        for j in (BTN_P, BTN_N):
            v = float(states[MOTOR].joints[j])
            if not math.isfinite(v):
                return None
            q[j] = v
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, low, q)


# -- the readings ------------------------------------------------------------------------------------------------------
def fit(r: Reading, name: str) -> dict:
    """An attachment's pose in the motor part's frame: ``p`` its shoulder face along the socket's axis (0 at the
    coupling face, negative outside), ``lat`` its axis' offset from the socket's, ``ang`` the angle between the axes,
    ``tip`` how far the stub's end is past the coupling face."""
    rm = r.rot[MOTOR]
    rel = rm.T @ (r.pos[name] - r.pos[MOTOR])
    axis_a, axis_m = r.rot[name][:, 1], rm[:, 1]
    p = float(rel[1])
    # the attachment's axis line against the socket's axis line, at the shoulder (a tilted stub counts at its face)
    lat = math.hypot(float(rel[0]), float(rel[2]))
    ang = angle_deg(axis_a, axis_m)
    return {"p": p, "lat": lat, "ang": ang, "tip": p + L.STUB_LEN}


def inserted(f: dict) -> bool:
    return bool(f["lat"] <= L.INSERT_LAT and f["ang"] <= L.INSERT_DEG and f["tip"] > L.INSERT_MIN)


def home(f: dict) -> bool:
    return bool(inserted(f) and abs(f["p"]) <= L.HOME_TOL + L.EPS and f["lat"] <= L.HOME_LAT + L.EPS
                and f["ang"] <= L.HOME_DEG + L.EPS)


def hook_in(q: float) -> bool:
    """A hook is behind the lip when its button is out (within 0.4 mm of rest)."""
    return bool(q >= -L.HOOK_IN_MAX - L.EPS)


def hooks(r: Reading) -> tuple[bool, bool]:
    return hook_in(r.q[BTN_P]), hook_in(r.q[BTN_N])


def shaft_centre(r: Reading) -> np.ndarray:
    return r.pos[SHAFT] + r.rot[SHAFT][:, 1] * SHAFT_MID_U


def on_mat(r: Reading) -> bool:
    c = shaft_centre(r)
    over = abs(float(c[0]) - L.MAT_X) <= L.MAT_HALF + L.EPS and abs(float(c[1]) - L.MAT_Y) <= L.MAT_HALF + L.EPS
    return bool(over and abs(r.low[SHAFT] - L.MAT_T) <= L.LOW_TOL + L.EPS)


def fell(r: Reading) -> bool:
    return any(r.low[n] < -L.FALL_DROP for n in PARTS)


def tier(latched: bool, settled: bool, shaft_off: bool, shaft_mat: bool, whisk_home: bool, both_hooks: bool) -> float:
    if latched or not settled or not shaft_off:
        return 0.0
    on = whisk_home and both_hooks
    seated = whisk_home and not both_hooks
    if on and shaft_mat:
        return 1.0
    if on or (seated and shaft_mat):
        return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


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
        m0 = np.array([L.M_X, L.M_Y0, L.M_AXIS_Z])
        w0 = np.array([L.WK_X, L.WK_SHOULDER_Y, L.WK_AXIS_Z])
        return (float(np.linalg.norm(r.pos[MOTOR] - m0)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[WHISK] - w0)) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in PARTS)

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
            dists = [states[n].distance for n in PARTS]

            def dist(points):
                return np.min([d(points) for d in dists], axis=0)
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, gap = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of any part from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            for n in PARTS:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        fs, fw = fit(r, SHAFT), fit(r, WHISK)
        shaft_off = not inserted(fs)
        shaft_mat = shaft_off and on_mat(r)
        whisk_home = home(fw)
        hp, hn = hooks(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, shaft_off, shaft_mat, whisk_home, hp and hn)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "shaft_off": shaft_off, "shaft_on_mat": shaft_mat, "whisk_home": whisk_home,
            "hook_p_in": hp, "hook_n_in": hn, "whisk_on": bool(whisk_home and hp and hn),
            "btn_p_mm": _mm(r.q[BTN_P]), "btn_n_mm": _mm(r.q[BTN_N]),
            "shaft_p_mm": _mm(fs["p"]), "shaft_lat_mm": _mm(fs["lat"]), "shaft_ang_deg": round(fs["ang"], 2),
            "whisk_p_mm": _mm(fw["p"]), "whisk_lat_mm": _mm(fw["lat"]), "whisk_ang_deg": round(fw["ang"], 2),
            "shaft_low_mm": _mm(r.low[SHAFT]), "shaft_centre_mm": _mmv(shaft_centre(r)),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "motor_mm": _mmv(r.pos[MOTOR]), "motor_yaw_deg": round(yaw_deg(r.rot[MOTOR]), 2),
            "shaft_mm": _mmv(r.pos[SHAFT]), "whisk_mm": _mmv(r.pos[WHISK]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
