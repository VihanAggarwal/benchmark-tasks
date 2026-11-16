"""The scorer, as plain functions of the caddy's, the forks' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the caddy (base frame at the middle of its footprint, at its base's underside, x toward
its back wall) and of the three forks (base frame at each fork's middle). Every reading is a pose; the arms are read
only for the let-go check. The rack's rail is fixed: its axis is the line x R_X, z RAIL_Z along y.

    F1 FELL             latched: the caddy's or a fork's middle more than 50 mm below the table top
    SEATED (per hook)   the rail's axis, taken where it passes the hook's centre line and read in the caddy's frame,
                        lies between the hook's two legs (the rail's surface at least -1 mm clear of each leg's inner
                        face), with the bridge's underside from 2 mm below to 3 mm above the rail's top (resting on
                        it); the hook's centre is at least 12 mm inside the rail's ends; and the caddy faces the rack
                        (its yaw within 30 degrees of the wall's normal) and stands upright within 30 degrees
    HUNG                both hooks seated, the caddy's yaw within 5 degrees of the wall's normal, its tilt within 5
    CUTLERY KEPT        all three forks' middles over the caddy's footprint, above its floor and below its rim
    RELEASED            on every step of the last 1 s both finger zones are 30 mm or more from the caddy and the forks
    STILL               over the last 1 s the caddy moved under 5 mm and turned under 5 degrees

    0      F1 ever fired, or not released and still (a held caddy earns nothing), or no hook seated
    1.0    hung, cutlery kept
    0.5    hung, a fork out of the caddy
    0.25   one hook seated, or both seated with the caddy crooked (yaw or tilt over 5 degrees)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CADDY = "caddy"
FORKS = L.FORK_NAMES
PARTS = (CADDY,) + FORKS
HOOK_SIDES = (-1, 1)


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


def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # part -> base frame origin
    rot: dict  # part -> rotation


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot = {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            pos[name], rot[name] = o - np.array([0.0, 0.0, L.TOP]), r
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot)


# -- the readings ------------------------------------------------------------------------------------------------------
def to_caddy(r: Reading, points) -> np.ndarray:
    """World points (N, 3) in the caddy's frame."""
    return (np.atleast_2d(np.asarray(points, dtype=float)) - r.pos[CADDY]) @ r.rot[CADDY]


def from_caddy(r: Reading, local) -> np.ndarray:
    return r.pos[CADDY] + np.atleast_2d(np.asarray(local, dtype=float)) @ r.rot[CADDY].T


def hook_reading(r: Reading, side: int) -> dict:
    """Hook ``side`` (+1 / -1: its y in the caddy's frame): where the rail's axis lies in the caddy's frame at the
    hook's centre line, and whether the hook is seated on it."""
    centre = from_caddy(r, (0.0, side * L.HK_Y, 0.0))[0]
    p = np.array([L.R_X, float(centre[1]), L.RAIL_Z])
    q = to_caddy(r, p)[0]
    qx, qz = float(q[0]), float(q[2])
    clear_near = qx - L.RAIL_R - L.NEAR_FACE  # the rail's surface from the near leg's inner face (+ clear)
    clear_rear = L.REAR_INNER - (qx + L.RAIL_R)  # and from the rear leg's
    bridge_gap = L.U - (qz + L.RAIL_R)  # the bridge's underside over the rail's top (+ above)
    along = abs(float(centre[1]) - L.R_Y)
    rc = r.rot[CADDY]
    facing = bool(abs(yaw_deg(rc)) <= L.SEAT_TURN_DEG and tilt_deg(rc) <= L.SEAT_TURN_DEG)  # back wall to the rack
    seated = bool(facing and clear_near >= -L.SEAT_X_TOL and clear_rear >= -L.SEAT_X_TOL
                  and -L.SEAT_Z_LO <= bridge_gap <= L.SEAT_Z_HI and along <= L.RAIL_L / 2 - L.END_MARGIN)
    return {"side": side, "qx": qx, "qz": qz, "clear_near": clear_near, "clear_rear": clear_rear,
            "bridge_gap": bridge_gap, "along": along, "seated": seated}


def forks_in(r: Reading) -> list:
    """Per fork: its middle over the caddy's footprint, above the floor and below the rim."""
    out = []
    for n in FORKS:
        lx, ly, lz = (float(v) for v in to_caddy(r, r.pos[n])[0])
        out.append(bool(abs(lx) <= L.C_D / 2 and abs(ly) <= L.C_W / 2 and L.FORK_Z_LO <= lz <= L.C_H))
    return out


def fell(r: Reading) -> bool:
    return any(float(r.pos[n][2]) < -L.FALL_DROP for n in PARTS)


def base_height(r: Reading) -> float:
    """The caddy's lowest point over the table top (m): the base's underside's lowest corner."""
    corners = np.array([(sx * L.C_D / 2, sy * L.C_W / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1)])
    return float(np.min(from_caddy(r, corners)[:, 2]))


def tier(latched: bool, settled: bool, n_seated: int, square: bool, forks_ok: bool) -> float:
    if latched or not settled or n_seated == 0:
        return 0.0
    if n_seated == 2 and square:
        return 1.0 if forks_ok else 0.5
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
        s0 = np.array([L.START_X, L.START_Y, L.START_Z])
        return float(np.linalg.norm(r.pos[CADDY] - s0)) <= L.START_TOL

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
        """Over the last 1 s: the caddy's largest move (m) and turn (deg) from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.pos[CADDY] - first.pos[CADDY])))
            turn = max(turn, turn_deg(first.rot[CADDY], x.rot[CADDY]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        hooks = {s: hook_reading(r, s) for s in HOOK_SIDES}
        n_seated = sum(1 for h in hooks.values() if h["seated"])
        rc = r.rot[CADDY]
        yaw, tilt = yaw_deg(rc), tilt_deg(rc)
        square = bool(abs(yaw) <= L.YAW_DEG + L.EPS and tilt <= L.TILT_DEG + L.EPS)
        ins = forks_in(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, n_seated, square, all(ins))
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "seated": {str(s): h["seated"] for s, h in hooks.items()}, "n_seated": n_seated,
            "rail_qx_mm": {str(s): _mm(h["qx"]) for s, h in hooks.items()},
            "clear_near_mm": {str(s): _mm(h["clear_near"]) for s, h in hooks.items()},
            "clear_rear_mm": {str(s): _mm(h["clear_rear"]) for s, h in hooks.items()},
            "bridge_gap_mm": {str(s): _mm(h["bridge_gap"]) for s, h in hooks.items()},
            "yaw_deg": round(yaw, 2), "tilt_deg": round(tilt, 2), "square": square,
            "forks_in": ins, "base_mm": _mm(base_height(r)),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "caddy_mm": _mmv(r.pos[CADDY]),
            "forks_mm": {n: _mmv(r.pos[n]) for n in FORKS},
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
