"""The scorer, as plain functions of the two totes' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the base tote (the one standing on the cart's -y bay) and the picked tote (the one that
starts in the +y bay), each with its frame at the centre of its foot's underside. Every reading is a pose; the arms
are read only for the let-go check.

    F1 FELL        latched: either tote's foot underside lower than 3 mm above the table (off the deck), or either
                   tote tilted more than 30 degrees (on its side)
    F2 DISTURBED   latched: the base tote more than 10 mm from where it stood, or turned or tilted more than 5 degrees
    SEATED         the picked tote's origin within 1 mm of the seated height above the base tote's (55 mm), its centre
                   within 4 mm of the base tote's in x and y, its long axis within 2 degrees of the base tote's, its
                   tilt under 1 degree
    NEAR           on the base tote, 52 to 67 mm above it, within 20 mm of its centre in x and y, within 4 degrees
                   of its long axis, tilt under 3 degrees (resting on the rim, or seated sloppily)
    ON             on the base tote: its centre within 60 mm in x and y, its origin 45 to 80 mm above the base
                   tote's, long axis within 20 degrees, tilt under 20 degrees
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from both totes
    STILL          over the last 1 s each tote moved under 5 mm and turned under 5 degrees (no visible motion)

    0      F1 or F2 ever fired, or not released and still (a held tote earns nothing)
    1.0    SEATED
    0.5    NEAR but not SEATED
    0.25   ON but not NEAR
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BASE, PICK = "base_tote", "pick_tote"
PARTS = (BASE, PICK)
_UP = np.array([0.0, 0.0, 1.0])
FOOT_PTS = np.array(L.FOOT_PTS)
RIM_PTS = np.array(L.RIM_PTS)
START = {BASE: np.array(L.BASE_ORIGIN), PICK: np.array(L.PICK_ORIGIN)}


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


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # part -> its frame origin (the foot underside's centre)
    rot: dict  # part -> rotation
    foot: dict  # part -> (4, 3) the foot underside's corners
    rim: dict  # part -> (4, 3) the rim's corners


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, foot, rim = {}, {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            o = o - np.array([0.0, 0.0, L.TOP])
            pos[name], rot[name] = o, r
            foot[name], rim[name] = o + FOOT_PTS @ r.T, o + RIM_PTS @ r.T
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, foot, rim)


# -- the readings ------------------------------------------------------------------------------------------------------
def relative(r: Reading) -> dict:
    """The picked tote in the base tote's frame: the offset (m) of its origin, its turn about the long axis's
    direction in the top view (degrees, in -90..90: a tote is the same turned 180), and its tilt (degrees)."""
    rb, rp = r.rot[BASE], r.rot[PICK]
    d = rb.T @ (r.pos[PICK] - r.pos[BASE])
    rr = rb.T @ rp
    yaw = math.degrees(math.atan2(float(rr[1, 0]), float(rr[0, 0])))
    yaw = (yaw + 90.0) % 180.0 - 90.0
    return {"dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]), "yaw": yaw, "tilt": tilt_deg(rr)}


def seated(rel: dict) -> bool:
    return bool(abs(rel["dz"] - L.SEAT_DZ) <= L.SEAT_TOL_Z + L.EPS
                and abs(rel["dx"]) <= L.SEAT_TOL_XY + L.EPS and abs(rel["dy"]) <= L.SEAT_TOL_XY + L.EPS
                and abs(rel["yaw"]) <= L.SEAT_TOL_YAW + L.EPS and rel["tilt"] <= L.SEAT_TOL_TILT + L.EPS)


def near(rel: dict) -> bool:
    return bool(on_base(rel) and L.NEAR_Z0 <= rel["dz"] <= L.NEAR_Z1
                and abs(rel["dx"]) <= L.NEAR_XY + L.EPS and abs(rel["dy"]) <= L.NEAR_XY + L.EPS
                and abs(rel["yaw"]) <= L.NEAR_YAW + L.EPS and rel["tilt"] <= L.NEAR_TILT + L.EPS)


def on_base(rel: dict) -> bool:
    return bool(abs(rel["dx"]) <= L.ON_XY + L.EPS and abs(rel["dy"]) <= L.ON_XY + L.EPS
                and L.ON_Z0 <= rel["dz"] <= L.ON_Z1 and rel["tilt"] <= L.ON_TILT + L.EPS
                and abs(rel["yaw"]) <= L.ON_YAW + L.EPS)


def fell(r: Reading) -> bool:
    for n in PARTS:
        if float(np.min(r.foot[n][:, 2])) < L.FALL_Z or tilt_deg(r.rot[n]) > L.TIP_DEG:
            return True
    return False


def base_disturbed(r: Reading) -> bool:
    moved = float(np.linalg.norm(r.pos[BASE][:2] - START[BASE][:2]))
    rb = r.rot[BASE]
    yaw = abs(math.degrees(math.atan2(float(rb[1, 0]), float(rb[0, 0]))))
    yaw = min(yaw, 180.0 - yaw)  # the tote is the same turned 180
    return bool(moved > L.MOVED_XY + L.EPS or yaw > L.MOVED_YAW + L.EPS or tilt_deg(rb) > L.MOVED_YAW + L.EPS)


def tier(latched: bool, settled: bool, rel: dict) -> float:
    if latched or not settled:
        return 0.0
    if seated(rel):
        return 1.0
    if near(rel):
        return 0.5
    if on_base(rel):
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
    """The latches (F1, F2), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "F2": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None or self.latch["F2"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return all(float(np.linalg.norm(r.pos[n] - START[n])) <= L.START_TOL for n in PARTS)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The totes put back on their starting marks with a jump no trial makes in one step (a reset)."""
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
        if self.latch["F2"] is None and base_disturbed(r):
            self.latch["F2"] = self.steps
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
        """Over the last 1 s: the largest move (m) and turn (deg) of either tote from where it was at the window's
        start; None before 1 s of valid readings."""
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
        rel = relative(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, rel)
        rb = r.rot[BASE]
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "seated": seated(rel), "near": near(rel), "on_base": on_base(rel),
            "dz_mm": _mm(rel["dz"]), "seat_error_mm": _mm(rel["dz"] - L.SEAT_DZ),
            "dx_mm": _mm(rel["dx"]), "dy_mm": _mm(rel["dy"]),
            "yaw_deg": round(rel["yaw"], 3), "tilt_deg": round(rel["tilt"], 3),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "base_mm": _mmv(r.pos[BASE]), "pick_mm": _mmv(r.pos[PICK]),
            "base_moved_mm": _mm(float(np.linalg.norm(r.pos[BASE][:2] - START[BASE][:2]))),
            "base_yaw_deg": round(yaw_deg(rb), 3), "base_tilt_deg": round(tilt_deg(rb), 3),
            "pick_low_mm": _mm(float(np.min(r.foot[PICK][:, 2]))),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
