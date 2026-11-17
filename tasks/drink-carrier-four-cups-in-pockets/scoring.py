"""The scorer, as plain functions of the four cups', the carrier's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. The
cups are tapered compounds (mechkit AssemblyState: origin = the centre of the underside; a plain state with pos = the
centre is read too); the carrier is a mechkit AssemblyState
(``origin`` = its frame's origin, on the table at the carrier's centre; ``quat_xyzw`` its body's turn; the frame's z is
up and z = 0 is the table top). Every reading is a pose or a velocity; the arms are read only for the let-go check.

    SEATED      a cup's underside within 2 mm of a pocket's floor, its axis within 2 degrees of the carrier's
                vertical, its centre within 8 mm of the pocket's axis (all in the carrier's frame), and it is the
                only cup over that pocket's mouth
    SPILL       latched: any cup tilted more than 15 degrees from the vertical at any step
    DROPPED     latched: any cup falling faster than 0.886 m/s (a free fall of 40 mm or more)
    CARRIER     latched: the carrier's centre more than 30 mm from its mark, or tilted more than 5 degrees
    FELL        latched: a cup or the carrier more than 50 mm below the table top
    LYING       a cup tilted more than 45 degrees now
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from every cup and the carrier
    STILL       over the last 1 s every cup and the carrier moves under 5 mm and turns under 5 degrees

    0      a latch fired, a cup lying, or not released and still (a held cup earns nothing)
    n/4    otherwise, n = the number of pockets with a seated cup: 1.0, 0.75, 0.5, 0.25, or 0 for none
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CARRIER = "carrier"
CUPS = L.CUP_NAMES
PARTS = CUPS + (CARRIER,)
_Z = np.array([0.0, 0.0, 1.0])


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) or 1.0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _mm(v) -> float | None:
    return round(float(v) * 1000.0, 2) if v is not None and np.isfinite(v) else None


def _mmv(v) -> list:
    return [round(float(a) * 1000.0, 1) for a in v]


@dataclass
class Reading:
    """One step's poses, as plain arrays."""
    pos: dict          # cup name -> centre (3,); "carrier" -> frame origin (3,)
    rot: dict          # name -> rotation matrix
    vel: dict          # name -> linear velocity
    low: dict          # name -> lowest point z of its bounding box


def read(states) -> Reading | None:
    """The step's readings, or None when any part is missing or any number is not finite (a blow-up)."""
    pos, rot, vel, low = {}, {}, {}, {}
    try:
        for n in PARTS:
            s = states[n]
            q = np.asarray(s.quat_xyzw, dtype=float).reshape(4)
            if n == CARRIER:
                p = np.asarray(s.origin, dtype=float).reshape(3)
            elif getattr(s, "origin", None) is not None:  # a cup compound: its origin is its underside's centre
                o = np.asarray(s.origin, dtype=float).reshape(3)
                p = o + rotation(q) @ np.array([0.0, 0.0, L.CUP_H / 2]) if np.all(np.isfinite(q)) else o
            else:  # a plain state (a test double): pos is the cup's centre
                p = np.asarray(s.pos, dtype=float).reshape(3)
            v = np.asarray(s.lin_vel, dtype=float).reshape(3)
            w = np.asarray(s.ang_vel, dtype=float).reshape(3)
            lo = np.asarray(s.low, dtype=float).reshape(3)
            hi = np.asarray(s.high, dtype=float).reshape(3)
            for a in (p, q, v, w, lo, hi):
                if not np.all(np.isfinite(a)):
                    return None
            if float(np.linalg.norm(q)) < 1e-9:
                return None
            pos[n], rot[n], vel[n], low[n] = p, rotation(q), v, float(lo[2])
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return Reading(pos, rot, vel, low)


# -- geometry in the carrier's frame ---------------------------------------------------------------------------------
def to_carrier(r: Reading, p) -> np.ndarray:
    """A world point in the carrier's frame (origin on the table at its centre, z up when it lies flat)."""
    return r.rot[CARRIER].T @ (np.asarray(p, dtype=float) - r.pos[CARRIER])


def pocket_local(name: str) -> np.ndarray:
    x, y = L.POCKET_XY[name]
    return np.array([x - L.CAR_X, y - L.CAR_Y])


def cup_tilt_deg(r: Reading, cup: str) -> float:
    """The cup's axis from the world's vertical (degrees)."""
    return _angle_deg(r.rot[cup][:, 2], _Z)


def cup_rel_tilt_deg(r: Reading, cup: str) -> float:
    """The cup's axis from the carrier's vertical (degrees)."""
    return _angle_deg(r.rot[cup][:, 2], r.rot[CARRIER][:, 2])


def cup_underside(r: Reading, cup: str) -> float:
    """Height of the centre of the cup's underside in the carrier's frame (m)."""
    c = to_carrier(r, r.pos[cup])
    axis = r.rot[CARRIER].T @ r.rot[cup][:, 2]
    return float(c[2] - axis[2] * L.CUP_H / 2)


def pocket_offset(r: Reading, cup: str, pocket: str) -> float:
    """Horizontal distance (carrier frame) from the cup's centre to the pocket's axis."""
    c = to_carrier(r, r.pos[cup])
    return float(np.hypot(*(c[:2] - pocket_local(pocket))))


def seated_in(r: Reading, cup: str, pocket: str) -> bool:
    return bool(
        pocket_offset(r, cup, pocket) <= L.SEAT_XY_TOL + L.EPS
        and abs(cup_underside(r, cup) - L.FLOOR_TOP) <= L.SEAT_Z_TOL + L.EPS
        and cup_rel_tilt_deg(r, cup) <= L.SEAT_TILT_DEG + L.EPS
    )


def over_mouth(r: Reading, cup: str, pocket: str) -> bool:
    """The cup's centre is over the pocket's mouth (inside the 93 mm opening, seen from above)."""
    return pocket_offset(r, cup, pocket) <= L.POCKET_R_RIM


def pocket_credit(r: Reading) -> dict:
    """pocket -> the cup that earns it, or None: a seated cup that is the only cup over that pocket's mouth."""
    out = {}
    for pocket in L.POCKETS:
        over = [c for c in CUPS if over_mouth(r, c, pocket)]
        seated = [c for c in over if seated_in(r, c, pocket)]
        out[pocket] = seated[0] if len(over) == 1 and len(seated) == 1 else None
    return out


# -- distances for the let-go check ----------------------------------------------------------------------------------
def cup_distance(r: Reading, cup: str):
    """Distance to the cup's outline: the cone of its body (56 to 80 mm) and its 82 mm lid, as a stack of discs."""
    c, rot = r.pos[cup], r.rot[cup]
    parts = [(zc - L.CUP_H / 2, d / 2, h) for _, zc, d, h in L.cup_parts()]

    def distance(points):
        q = (np.atleast_2d(np.asarray(points, dtype=float)) - c) @ rot
        radial = np.hypot(q[:, 0], q[:, 1])
        best = np.full(len(q), np.inf)
        for zc, rad, h in parts:
            d = np.hypot(np.maximum(radial - rad, 0.0), np.maximum(np.abs(q[:, 2] - zc) - h / 2, 0.0))
            best = np.minimum(best, d)
        return best

    return distance


def carrier_distance(r: Reading):
    """The carrier as one box: its footprint, from the table to the pockets' mouths."""
    rot = r.rot[CARRIER]
    centre = r.pos[CARRIER] + rot @ np.array([0.0, 0.0, L.RIM_H / 2])
    size = (L.PLATE, L.PLATE, L.RIM_H)
    return rigstate.boxes_distance([(centre, size, rot)])


def all_distance(r: Reading):
    fns = [cup_distance(r, c) for c in CUPS] + [carrier_distance(r)]

    def distance(points):
        return np.min([f(points) for f in fns], axis=0)

    return distance


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The latches, the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"spill": None, "dropped": None, "carrier": None, "fell": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        for c in CUPS:
            x, y = L.CUP_START[c]
            if float(np.hypot(r.pos[c][0] - x, r.pos[c][1] - y)) > L.START_TOL:
                return False
        return bool(float(np.hypot(r.pos[CARRIER][0] - L.CAR_X, r.pos[CARRIER][1] - L.CAR_Y)) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """Every part back on its starting mark, with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in PARTS)

    def _latch(self, r: Reading) -> None:
        s = self.steps
        if self.latch["spill"] is None and any(cup_tilt_deg(r, c) > L.SPILL_DEG for c in CUPS):
            self.latch["spill"] = s
        if self.latch["dropped"] is None and any(float(r.vel[c][2]) < -L.DROP_SPEED for c in CUPS):
            self.latch["dropped"] = s
        if self.latch["carrier"] is None:
            moved = float(np.hypot(r.pos[CARRIER][0] - L.CAR_X, r.pos[CARRIER][1] - L.CAR_Y))
            tilt = _angle_deg(r.rot[CARRIER][:, 2], _Z)
            if moved > L.CAR_MOVE or tilt > L.CAR_TILT_DEG:
                self.latch["carrier"] = s
        if self.latch["fell"] is None and any(r.low[n] < L.TOP - L.OFF_TABLE for n in PARTS):
            self.latch["fell"] = s

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
        self._latch(r)
        ok, gap = False, None
        try:
            dist = all_distance(r)
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and axis turn (deg) of any part from where it was at the window's
        start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            for n in PARTS:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, _angle_deg(x.rot[n][:, 2], first.rot[n][:, 2]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        credit = pocket_credit(r)
        n = sum(1 for v in credit.values() if v is not None)
        lying = any(cup_tilt_deg(r, c) > L.LYING_DEG for c in CUPS)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        if self.latched or lying or not (released and still):
            score = 0.0
        else:
            score = n / 4.0
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch), "seated": n,
            "credit": {k: v for k, v in credit.items()}, "lying": lying,
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "cup_tilt_deg": {c: round(cup_tilt_deg(r, c), 2) for c in CUPS},
            "cup_underside_mm": {c: _mm(cup_underside(r, c)) for c in CUPS},
            "cup_off_mm": {c: _mm(min(pocket_offset(r, c, p) for p in L.POCKETS)) for c in CUPS},
            "carrier_move_mm": _mm(float(np.hypot(r.pos[CARRIER][0] - L.CAR_X, r.pos[CARRIER][1] - L.CAR_Y))),
            "carrier_tilt_deg": round(_angle_deg(r.rot[CARRIER][:, 2], _Z), 2),
            "cups_mm": {c: _mmv(r.pos[c]) for c in CUPS},
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(score, bool(score == 1.0), details)
