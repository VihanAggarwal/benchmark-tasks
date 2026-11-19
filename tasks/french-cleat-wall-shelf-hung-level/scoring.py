"""The scorer, as plain functions of the shelf's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyState of the shelf (unit frame: origin at the middle of its underside, x toward the wall, y left, z
up, no drawn turn). Every reading is a pose; the arms are read only for the let-go check. The wall is fixed.

    COVER       the unit cleat's two ends (local y +-150 mm) projected on the wall: the part of it over the wall
                cleat (y +-170 mm) is the ENGAGED length (0 to 300 mm), between the overlap's two ends A and B
    CONTACT     at A (or B): the unit cleat's bevel face coincides with the wall cleat's bevel plane within 1 mm at two
                probes 10 mm apart along the slope, and the two faces overlap by 4 mm or more along the slope
    BACK        the unit cleat's back face x at A and B, from the wall plane (- off the wall, + into it)
    UNDER       the shelf's underside height at A and B above the bench
    LEVEL       tilt of the unit's long axis from horizontal; PLUMB: tilt of its back face from vertical
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the shelf
    STILL       over the last 1 s the shelf moved under 1 mm and turned under 0.3 degrees

    0      not released and still (a held shelf earns nothing), or no CONTACT at A or B
    1.0    CONTACT at A and B, the whole unit cleat engaged (300 mm), BACK within -1.5 to +1 mm, UNDER 40 +-2 mm,
           LEVEL and PLUMB within 1 degree
    0.5    CONTACT at A and B, engaged 150 mm or more, BACK 8 mm off the wall or less, LEVEL and PLUMB within 3 degrees
    0.25   CONTACT at one of A, B only, or at both with the shelf outside the 0.5 limits (one-ended or poorly seated)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SHELF = "shelf"
PARTS = (SHELF,)

_XM = 0.5 * (L.UC_FRONT_LOCAL + L.UC_BACK_LOCAL)  # the middle of the unit cleat's bevel (local x)
_E_MID = np.array([_XM, 0.0, L.u_bevel_z(_XM)])
_PROBES = (L.CONTACT_X_A, L.CONTACT_X_B)
_WALL_S = sorted((L.slope_s(L.W_FACE_X0, L.bevel_z(L.W_FACE_X0)), L.slope_s(L.W_FACE_X1, L.bevel_z(L.W_FACE_X1))))


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


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """One step (m). ``pos`` is the unit frame's origin in the world frame with the table top at z 0."""
    pos: np.ndarray
    rot: np.ndarray


def read(states) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        s = states.get(SHELF)
        if s is None:
            return None
        o = np.asarray(s.origin, dtype=float).ravel()
        r = rotation(s.quat_xyzw)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
            return None
        return Reading(o - np.array([0.0, 0.0, L.TOP]), r)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def world(r: Reading, local) -> np.ndarray:
    return r.pos + r.rot @ np.asarray(local, dtype=float)


def level_deg(r: Reading) -> float:
    """Tilt of the shelf's long axis from horizontal (deg, + when its +y end is higher)."""
    return math.degrees(math.asin(max(-1.0, min(1.0, float(r.rot[2, 1])))))


def plumb_deg(r: Reading) -> float:
    """Tilt of the back face from vertical (deg, + when the wall side is higher)."""
    return math.degrees(math.asin(max(-1.0, min(1.0, float(r.rot[2, 0])))))


def cover(r: Reading) -> tuple[float, float, float]:
    """(lo, hi, engaged): the world y range (m) of the unit cleat that lies over the wall cleat, and its length."""
    ends = [world(r, (_E_MID[0], s * L.UCLEAT_L / 2, _E_MID[2]))[1] for s in (-1.0, 1.0)]
    lo = max(min(ends), -L.WCLEAT_L / 2)
    hi = min(max(ends), L.WCLEAT_L / 2)
    return lo, hi, max(0.0, hi - lo)


def at_y(r: Reading, y: float) -> dict | None:
    """The unit cleat where it crosses world y = ``y``: its bevel probes' distance to the wall's bevel plane, the
    bevel faces' overlap along the slope, the back face's x off the wall and the underside's height. None when the
    shelf's long axis is turned too far to say."""
    c = float(r.rot[1, 1])
    if c < 0.5:
        return None
    lam = (y - float(r.pos[1])) / c
    d, ss = [], []
    for lx in _PROBES:
        p = world(r, (lx, lam, L.u_bevel_z(lx)))
        d.append(float(L.plane_distance(p[0], p[2])))
        ss.append(float(L.slope_s(p[0], p[2])))
    ov = min(max(ss), _WALL_S[1]) - max(min(ss), _WALL_S[0])
    back = float(world(r, (L.UC_BACK_LOCAL, lam, 0.5 * L.UC_TOP))[0] - L.WALL_X)
    under = float(world(r, (0.0, lam, 0.0))[2])
    contact = bool(max(abs(d[0]), abs(d[1])) <= L.CONTACT_TOL + L.EPS and ov >= L.SLOPE_OVERLAP - L.EPS)
    return {"d0": d[0], "d1": d[1], "slope_overlap": ov, "back": back, "under": under, "contact": contact}


def judge_pose(r: Reading) -> dict:
    """Everything the tiers read from one pose."""
    lo, hi, engaged = cover(r)
    out = {"engaged": engaged, "level": level_deg(r), "plumb": plumb_deg(r), "a": None, "b": None,
           "contact_a": False, "contact_b": False}
    if engaged > 0.0:
        a, b = at_y(r, lo), at_y(r, hi)
        out.update(a=a, b=b, contact_a=bool(a and a["contact"]), contact_b=bool(b and b["contact"]))
    return out


def tier(j: dict, settled: bool) -> float:
    """The score of one pose; 0 unless released and still."""
    if not settled:
        return 0.0
    ca, cb = j["contact_a"], j["contact_b"]
    if not (ca or cb):
        return 0.0
    if ca and cb:
        a, b = j["a"], j["b"]
        tilt = max(abs(j["level"]), abs(j["plumb"]))
        backs = (a["back"], b["back"])
        if (j["engaged"] >= L.FULL_ENGAGE and tilt <= L.LEVEL_DEG + L.EPS
                and all(L.BACK_LO - L.EPS <= v <= L.BACK_HI + L.EPS for v in backs)
                and all(abs(v["under"] - L.SEAT) <= L.UNDER_TOL + L.EPS for v in (a, b))):
            return 1.0
        if (j["engaged"] >= L.HALF_ENGAGE and tilt <= L.HUNG_DEG + L.EPS
                and all(-L.BACK_HUNG - L.EPS <= v <= 0.003 for v in backs)):
            return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        return float(np.linalg.norm(r.pos - np.array([L.SH_START_X, 0.0, 0.0]))) <= L.START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The shelf put back on its mark with a jump no trial makes in one step (a reset)."""
        return cls.at_start(now) and float(np.linalg.norm(now.pos - prev.pos)) >= L.RESET_JUMP

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        ok, gap = False, None
        try:
            dist = states[SHELF].distance
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not released
            ok, gap = False, None
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the shelf from where it was at the window's
        start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = max(float(np.linalg.norm(x.pos - first.pos)) for x in h[1:])
        turn = max(turn_deg(first.rot, x.rot) for x in h[1:])
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        j = judge_pose(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(j, released and still)
        a, b = j["a"], j["b"]
        details = {
            "valid_state": True, "score": level, "pose_tier": tier(j, True),
            "engaged_mm": _mm(j["engaged"]), "contact_a": j["contact_a"], "contact_b": j["contact_b"],
            "level_deg": round(j["level"], 3), "plumb_deg": round(j["plumb"], 3),
            "yaw_deg": round(yaw_deg(r.rot), 3),
            "back_a_mm": _mm(a["back"]) if a else None, "back_b_mm": _mm(b["back"]) if b else None,
            "under_a_mm": _mm(a["under"]) if a else None, "under_b_mm": _mm(b["under"]) if b else None,
            "face_gap_a_mm": _mm(max(abs(a["d0"]), abs(a["d1"]))) if a else None,
            "face_gap_b_mm": _mm(max(abs(b["d0"]), abs(b["d1"]))) if b else None,
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "shelf_mm": [round(float(c) * 1000.0, 2) for c in r.pos],
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
