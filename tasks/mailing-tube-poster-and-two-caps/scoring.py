"""The scorer, as plain functions of the tube's, the roll's, the two caps' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the tube (base frame at its centre, local y along its axis), the roll (base frame at its
centre, local y along its axis) and the two caps (base frame at the centre of the flange's inner face, local y out of
the tube, plug at y < 0). Every reading is a pose; the arms are read only for the let-go check.

    F1 LOST          latched: a part's centre more than 30 mm below the table top, or off the table
    ROLL IN          both ends of the roll at least 15 mm inside the tube's two ends (along the tube's axis) and each
                     within 8 mm of the tube's axis
    CAP SEATED       at a tube end: the cap's axis within 5 degrees of the tube's and pointing out of that end, its
                     flange's inner face within 1.5 mm outside the end (or 2 mm in), the flange centre within 2.5 mm
                     of the tube's axis, and the plug's tip within 3 mm of the axis (the plug is in the bore)
    ENDS CAPPED      how many of the tube's two ends have a seated cap (0, 1 or 2)
    RELEASED         on every step of the last 1 s both finger zones are 30 mm or more from all four parts
    STILL            over the last 1 s every part moved under 1 mm and turned under 2 degrees

    0      F1 ever fired, or not released and still, or the roll not in
    1.0    roll in and both ends capped
    0.5    roll in and one end capped
    0.25   roll in, no end capped
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

TUBE, ROLL, CAP_A, CAP_B = L.PARTS
CAPS = (CAP_A, CAP_B)
PARTS = L.PARTS
_Y = np.array([0.0, 1.0, 0.0])


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
def axis(r: Reading, name: str) -> np.ndarray:
    """A part's own axis (its local y) in the world."""
    return r.rot[name] @ _Y


def tube_frame(r: Reading, point) -> tuple[float, float]:
    """A world point along the tube's axis from its centre (m) and its distance from the axis."""
    u = axis(r, TUBE)
    d = np.asarray(point, dtype=float) - r.pos[TUBE]
    s = float(d @ u)
    return s, float(np.linalg.norm(d - s * u))


def roll_ends(r: Reading) -> list:
    u = axis(r, ROLL)
    return [r.pos[ROLL] + sgn * u * (L.ROLL_L / 2) for sgn in (-1.0, 1.0)]


def roll_in(r: Reading) -> bool:
    """Both ends of the roll inside the tube: at least 15 mm in from each end, within 8 mm of the tube's axis."""
    for e in roll_ends(r):
        s, rad = tube_frame(r, e)
        if abs(s) > L.TUBE_END - L.ROLL_END_IN + L.EPS or rad > L.ROLL_RADIAL + L.EPS:
            return False
    return True


def cap_seated_at(r: Reading, cap: str, end: int) -> bool:
    """Whether ``cap`` is pressed on the tube's end ``end`` (+1: the end the tube's local +y points to, -1: the other)."""
    u = axis(r, TUBE)
    w = end * u  # out of that end
    v = axis(r, cap)  # out of the tube, if the cap is on
    if float(v @ w) < math.cos(math.radians(L.CAP_TILT_DEG)) - L.EPS:
        return False
    p = r.pos[cap]
    s, rad = tube_frame(r, p)
    gap = end * s - L.TUBE_END  # the flange's inner face outside the end: > 0
    if gap > L.CAP_GAP + L.EPS or gap < -L.CAP_PEN - L.EPS:
        return False
    if rad > L.CAP_RADIAL + L.EPS:
        return False
    tip = p - v * L.CAP_DEPTH
    _, tip_rad = tube_frame(r, tip)
    return bool(tip_rad <= L.CAP_NOSE_RADIAL + L.EPS)


def ends_capped(r: Reading) -> int:
    n = 0
    for end in (1, -1):
        if any(cap_seated_at(r, c, end) for c in CAPS):
            n += 1
    return n


def lost(r: Reading) -> bool:
    for n in PARTS:
        p = r.pos[n]
        if p[2] < L.FALL_Z or abs(p[1]) > L.TABLE_Y or not (L.TABLE_X0 - 0.0 <= p[0] <= L.TABLE_X1):
            return True
    return False


def tier(latched: bool, settled: bool, rolled: bool, capped: int) -> float:
    if latched or not settled or not rolled:
        return 0.0
    if capped >= 2:
        return 1.0
    if capped == 1:
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
        starts = {TUBE: L.tube_origin(), ROLL: L.roll_origin(), CAP_A: L.cap_origin("a"), CAP_B: L.cap_origin("b")}
        return all(float(np.linalg.norm(r.pos[n] - np.array(starts[n]))) <= L.START_TOL for n in PARTS)

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
        if self.latch["F1"] is None and lost(r):
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
        rolled = roll_in(r)
        capped = ends_capped(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, rolled, capped)
        ends = [tube_frame(r, e) for e in roll_ends(r)]
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "roll_in": rolled, "ends_capped": capped,
            "cap_a_seated": [cap_seated_at(r, CAP_A, e) for e in (1, -1)],
            "cap_b_seated": [cap_seated_at(r, CAP_B, e) for e in (1, -1)],
            "roll_end_s_mm": [_mm(s) for s, _ in ends], "roll_end_radial_mm": [_mm(d) for _, d in ends],
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "tube_mm": _mmv(r.pos[TUBE]), "roll_mm": _mmv(r.pos[ROLL]),
            "cap_a_mm": _mmv(r.pos[CAP_A]), "cap_b_mm": _mmv(r.pos[CAP_B]),
            "tube_yaw_deg": round(math.degrees(math.atan2(float(r.rot[TUBE][1, 0]), float(r.rot[TUBE][0, 0]))), 2),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        for c in CAPS:
            s, rad = tube_frame(r, r.pos[c])
            details[f"{c}_axial_mm"] = _mm(abs(s) - L.TUBE_END)
            details[f"{c}_radial_mm"] = _mm(rad)
        return Result(level, bool(level == 1.0), details)
