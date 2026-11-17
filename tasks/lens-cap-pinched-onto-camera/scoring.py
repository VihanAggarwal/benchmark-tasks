"""The scorer, as plain functions of the camera's and the cap's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyStates of the camera (base frame at the centre of the lens's front face, +y into the lens toward the
body, z up through the camera's top) and of the cap (base frame at the centre of its seat face, +z out of its front,
two latch links ``latch_p`` / ``latch_n`` whose joint value is 0 out and negative in). Every reading is a pose or a
joint value; the arms are read only for the let-go check. A latch has no state of its own beyond its slide: a latch is
behind the lip when it is out (within 0.4 mm of rest) while the cap is home, because a latch that is not behind the lip
is held in by it (it cannot be out with the cap home and the lip in its way).

    CAP HOME     the cap's seat face within 1 mm of the lens's front face along the lens's axis (and not more than
                 0.5 mm past it), its axis within 2 mm of the lens's and 3 degrees square
    CAP LATCHED  home with BOTH latches out (behind the lip)
    CAMERA DOWN  the camera's up axis within 10 degrees of vertical and its lowest point within 5 mm of the table top
    F1 FELL      latched: any part's lowest point more than 30 mm below the table top (off the table)
    RELEASED     on every step of the last 1 s both finger zones are 30 mm or more from the camera and the cap
    STILL        over the last 1 s each part moved under 5 mm and turned under 5 degrees

    0      F1 ever fired, or not released and still, or the cap not home
    1.0    cap latched and camera down
    0.5    cap latched, the camera not down (on its side, on its back)
    0.25   cap home but not latched (a latch held in), camera down
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CAMERA, CAP = "camera", "cap"
PARTS = (CAMERA, CAP)
LATCH_P, LATCH_N = "latch_p", "latch_n"
LATCHES = (LATCH_P, LATCH_N)


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
    q: dict  # latch joint values (m; 0 out, negative in)


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
        for j in LATCHES:
            v = float(states[CAP].joints[j])
            if not math.isfinite(v):
                return None
            q[j] = v
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, low, q)


# -- the readings ------------------------------------------------------------------------------------------------------
def fit(r: Reading) -> dict:
    """The cap's pose in the camera's frame: ``p`` its seat face's distance out from the lens's front face along the
    lens's axis (0 home, positive in front of the lens, negative past the face), ``lat`` its centre's offset from the
    lens's axis, ``ang`` the angle between the cap's axis (out of its front) and the lens's (out of the lens)."""
    rc = r.rot[CAMERA]
    rel = rc.T @ (r.pos[CAP] - r.pos[CAMERA])
    return {"p": -float(rel[1]), "lat": math.hypot(float(rel[0]), float(rel[2])),
            "ang": angle_deg(r.rot[CAP][:, 2], -rc[:, 1])}


def home(f: dict) -> bool:
    return bool(-L.HOME_IN - L.EPS <= f["p"] <= L.HOME_OUT + L.EPS and f["lat"] <= L.HOME_LAT + L.EPS
                and f["ang"] <= L.HOME_DEG + L.EPS)


def latch_out(q: float) -> bool:
    """A latch is behind the lip when it is out (within 0.4 mm of rest) with the cap home."""
    return bool(q >= -L.LATCH_IN_MAX - L.EPS)


def latches(r: Reading) -> tuple[bool, bool]:
    return latch_out(r.q[LATCH_P]), latch_out(r.q[LATCH_N])


def camera_down(r: Reading) -> bool:
    up = r.rot[CAMERA][:, 2]
    return bool(angle_deg(up, np.array([0.0, 0.0, 1.0])) <= L.UPRIGHT_DEG + L.EPS
                and r.low[CAMERA] <= L.ON_TABLE + L.EPS)


def fell(r: Reading) -> bool:
    return any(r.low[n] < -L.FALL_DROP for n in PARTS)


def tier(latched: bool, settled: bool, cap_home: bool, both_out: bool, cam_down: bool) -> float:
    if latched or not settled or not cap_home:
        return 0.0
    if both_out:
        return 1.0 if cam_down else 0.5
    return 0.25 if cam_down else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


CAM_START = np.array([L.CAM_X, L.CAM_Y0, L.CAM_AXIS_Z])
CAP_START = np.array(L.CAP_ORIGIN) - np.array([0.0, 0.0, L.TOP])


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
        return (float(np.linalg.norm(r.pos[CAMERA] - CAM_START)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[CAP] - CAP_START)) <= L.START_TOL)

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
        f = fit(r)
        cap_home = home(f)
        lp, ln = latches(r)
        down = camera_down(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, cap_home, lp and ln, down)
        up = r.rot[CAMERA][:, 2]
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "cap_home": cap_home, "latch_p_out": lp, "latch_n_out": ln, "cap_latched": bool(cap_home and lp and ln),
            "camera_down": down,
            "latch_p_mm": _mm(r.q[LATCH_P]), "latch_n_mm": _mm(r.q[LATCH_N]),
            "cap_p_mm": _mm(f["p"]), "cap_lat_mm": _mm(f["lat"]), "cap_ang_deg": round(f["ang"], 2),
            "camera_tilt_deg": round(angle_deg(up, np.array([0.0, 0.0, 1.0])), 2),
            "camera_low_mm": _mm(r.low[CAMERA]), "cap_low_mm": _mm(r.low[CAP]),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "camera_mm": _mmv(r.pos[CAMERA]), "camera_yaw_deg": round(yaw_deg(r.rot[CAMERA]), 2),
            "cap_mm": _mmv(r.pos[CAP]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
