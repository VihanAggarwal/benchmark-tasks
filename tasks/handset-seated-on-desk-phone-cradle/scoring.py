"""The scorer, as plain functions of the handset's and the phone's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step (from
``task.success()``). It reads the mechkit AssemblyStates of the handset (frame at the centre of its pad-bottom plane) and
the phone (frame at the centre of its underside, hook-switch plunger as joint ``hook``, metres pressed down) and, for the
let-go check only, the arms. Everything is measured in the PHONE's own frame, so a phone that was nudged is judged
fairly.

    PAD IN CUP   a pad's four bottom corners inside its own cup's rectangle (0.5 mm allowed) and 2 mm below to 6 mm
                 above the cup's floor (earpiece pad in the earpiece cup, mouthpiece pad in the mouthpiece cup)
    PAD RESTS    a pad's four bottom corners over the phone's top outline and 2 mm below to 14 mm above the cup floor
                 (down in a cup, or on the flat top, 10 mm higher)
    SEATED       both pads in their own cups with every corner 2 mm below to 3 mm above the floor; the handset within
                 3 degrees of level against the phone and within 6 degrees of its heading (plan view); the hook switch
                 pressed down 2 mm or more (the line is closed)
    F2 OFF       latched: the handset's or the phone's lowest point more than 10 mm below the desk top
    F3 SHOVED    latched: the phone's origin ever more than 8 mm from its mark, or tilted more than 3 degrees
    RELEASED     on every step of the last 1 s both finger zones are 30 mm or more from the handset and the phone
    STILL        over the last 1 s the handset and the phone each moved under 5 mm and turned under 5 degrees

    0      F2 or F3 ever fired; or not (released and still): a held handset earns nothing
    1.0    seated, released and still
    0.5    one or both pads in their own cups and both pads resting on the phone, but not seated
    0.25   both pads resting on the phone, neither in its own cup (lying across the top, or turned the wrong way round)
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

HANDSET, PHONE = "handset", "phone"
PARTS = (HANDSET, PHONE)
_MARK = {HANDSET: np.array(L.HANDSET_ORIGIN), PHONE: np.array(L.PHONE_ORIGIN)}
_PAD_CORNERS = {n: np.array(L.pad_corners(n)) for n in L.PADS}
_HALF_TOP = np.array([L.PHONE_SIZE[0] / 2, L.PHONE_SIZE[1] / 2])


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-9:
        return np.full((3, 3), np.nan)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _acos_deg(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


def tilt_deg(r: np.ndarray) -> float:
    """Angle between a body's z axis and the vertical (degrees)."""
    return _acos_deg(r[2, 2])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    return _acos_deg((float(np.trace(r0.T @ r1)) - 1.0) / 2.0)


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """One step (m): the parts' frame origins and rotations in the world, each one's lowest point, the hook's travel."""
    pos: dict
    rot: dict
    low: dict
    hook: float


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, low = {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            lo = float(np.asarray(s.low, dtype=float).ravel()[2])
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and math.isfinite(lo)):
                return None
            pos[name], rot[name], low[name] = o, r, lo
        hook = float(states[PHONE].joints[L.HOOK])
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    if not math.isfinite(hook):
        return None
    return Reading(pos, rot, low, hook)


# -- the rules ---------------------------------------------------------------------------------------------------------
def pad_in_phone_frame(r: Reading, name: str) -> np.ndarray:
    """The pad's four bottom corners in the phone's own frame (m)."""
    world = r.pos[HANDSET] + _PAD_CORNERS[name] @ r.rot[HANDSET].T
    return (world - r.pos[PHONE]) @ r.rot[PHONE]


def pad_reading(r: Reading, name: str) -> dict:
    """One pad against its own cup and against the phone's top (m): how far its worst corner is outside the cup's
    rectangle (negative: inside), the lowest and highest corner over the cup floor, and how far its worst corner is
    outside the phone's top outline (negative: over the phone)."""
    c = pad_in_phone_frame(r, name)
    (cx, cy), (sx, sy) = L.CUPS[name]
    over = float(max(np.max(np.abs(c[:, 0] - cx)) - sx / 2, np.max(np.abs(c[:, 1] - cy)) - sy / 2))
    dz = c[:, 2] - L.FLOOR_Z
    off = float(max(np.max(np.abs(c[:, 0])) - _HALF_TOP[0], np.max(np.abs(c[:, 1])) - _HALF_TOP[1]))
    return {"over": over, "dz_min": float(dz.min()), "dz_max": float(dz.max()), "off_top": off,
            "centre_off": float(math.hypot(c[:, 0].mean() - cx, c[:, 1].mean() - cy))}


def in_cup(p: dict, strict: bool = False) -> bool:
    high = L.SEAT_HIGH if strict else L.CUP_HIGH
    return bool(p["over"] <= L.SLACK + L.EPS and L.SEAT_LOW - L.EPS <= p["dz_min"] and p["dz_max"] <= high + L.EPS)


def rests(p: dict) -> bool:
    return bool(p["off_top"] <= L.EPS and L.SEAT_LOW - L.EPS <= p["dz_min"] and p["dz_max"] <= L.REST_HIGH + L.EPS)


def attitude(r: Reading) -> dict:
    """The handset against the phone: its turn out of level and its heading, both in degrees."""
    rel = r.rot[PHONE].T @ r.rot[HANDSET]
    return {"tilt": tilt_deg(rel), "yaw": yaw_deg(rel)}


def seated(ear: dict, mouth: dict, a: dict, hook: float) -> bool:
    return bool(
        in_cup(ear, strict=True) and in_cup(mouth, strict=True)
        and a["tilt"] <= L.SEAT_TILT_DEG + L.EPS and abs(a["yaw"]) <= L.SEAT_YAW_DEG + L.EPS
        and hook >= L.HOOK_DOWN - L.EPS
    )


def phone_drift(r: Reading) -> tuple[float, float]:
    """How far the phone's origin is from its mark (m, horizontal) and how far it is tilted (degrees)."""
    d = r.pos[PHONE] - _MARK[PHONE]
    return float(math.hypot(d[0], d[1])), tilt_deg(r.rot[PHONE])


def tier(latched: bool, settled: bool, is_seated: bool, ear_in: bool, mouth_in: bool, ear_rest: bool,
         mouth_rest: bool) -> float:
    if latched or not settled:
        return 0.0
    if is_seated:
        return 1.0
    if (ear_in or mouth_in) and ear_rest and mouth_rest:
        return 0.5
    if ear_rest and mouth_rest:
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


class Tracker:
    """The latches (F2, F3; how the handset is tilted on the way is not judged), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F2": None, "F3": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        return all(float(np.linalg.norm(r.pos[n] - _MARK[n])) <= L.START_TOL for n in PARTS)

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
        if self.latch["F2"] is None and any(r.low[n] < L.TOP - L.OFF_DROP for n in PARTS):
            self.latch["F2"] = self.steps
        move, p_tilt = phone_drift(r)
        if self.latch["F3"] is None and (move > L.PHONE_MOVE or p_tilt > L.PHONE_TILT_DEG):
            self.latch["F3"] = self.steps
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
        """Over the last 1 s: the largest move (m) and turn (deg) of either part from where it was at the window's
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
        ear, mouth = pad_reading(r, "ear"), pad_reading(r, "mouth")
        a = attitude(r)
        is_seated = seated(ear, mouth, a, r.hook)
        ear_in, mouth_in = in_cup(ear), in_cup(mouth)
        ear_rest, mouth_rest = rests(ear), rests(mouth)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, is_seated, ear_in, mouth_in, ear_rest, mouth_rest)
        p_move, p_tilt = phone_drift(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "seated": is_seated, "ear_in_cup": ear_in, "mouth_in_cup": mouth_in,
            "ear_rests": ear_rest, "mouth_rests": mouth_rest,
            "ear_over_cup_mm": _mm(ear["over"]), "mouth_over_cup_mm": _mm(mouth["over"]),
            "ear_dz_mm": [_mm(ear["dz_min"]), _mm(ear["dz_max"])],
            "mouth_dz_mm": [_mm(mouth["dz_min"]), _mm(mouth["dz_max"])],
            "ear_centre_off_mm": _mm(ear["centre_off"]), "mouth_centre_off_mm": _mm(mouth["centre_off"]),
            "handset_tilt_vs_phone_deg": round(a["tilt"], 2), "handset_yaw_vs_phone_deg": round(a["yaw"], 2),
            "hook_mm": _mm(r.hook), "line_closed": bool(r.hook >= L.HOOK_DOWN - L.EPS),
            "phone_move_mm": _mm(p_move), "phone_tilt_deg": round(p_tilt, 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "handset_mm": _mmv(r.pos[HANDSET]), "phone_mm": _mmv(r.pos[PHONE]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
