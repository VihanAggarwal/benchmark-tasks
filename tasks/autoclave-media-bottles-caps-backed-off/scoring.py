"""The scorer, as plain functions of the two bottles' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of each bottle (base frame at the centre of the base's underside; slider link "rise" = the
cap's lift off its seat, the thread's 6 mm a turn; hinge link "turn" = the cap's turn from tight, anticlockwise from
above = unscrewing). Every reading is a pose or a joint position; the arms are read only for the let-go check.

The cap's TURN from tight is what a person reads in a real trial: the rise of its skirt over the shoulder ring (a
caliper or feeler gauge, 6 mm a turn) and the number of times its black line passed the shoulder's on the top video.

    CAP CLASS   on its thread: turn from tight -8 to 540 degrees, rise within 2 mm of 6 mm a turn, axis within 8 mm
                of the bottle's; otherwise OFF (lifted off the thread). On its thread: TIGHT (sealed) turn under
                90 degrees, GOOD 90 to 360 degrees, LOOSE over 360 up to 540 degrees
    IN PAN      the base circle inside the pan's inner walls and the base within 3 mm of the pan's floor
    UPRIGHT     axis within 10 degrees of vertical
    SEALED      a bottle resting in the pan (its base centre inside the inner walls, below the rim) with a TIGHT cap
    FELL        latched: a base more than 50 mm below the table top
    TIPPED      latched: a bottle's axis more than 45 degrees off vertical (the water would run out)
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from both bottles and caps
    STILL       over the last 1 s every base moved under 5 mm and turned under 5 degrees, every cap turned under
                5 degrees and rose or sank under 5 mm

    0      FELL or TIPPED ever; any SEALED bottle (a sealed bottle loaded for the cycle)
    1.0    both bottles IN PAN and UPRIGHT, both caps GOOD, RELEASED and STILL
    0.75   both IN PAN and UPRIGHT, caps GOOD or LOOSE, but a cap LOOSE or not RELEASED and STILL
    0.5    both IN PAN and UPRIGHT, a cap OFF the thread (the other GOOD or LOOSE)
    0.25   one IN PAN and UPRIGHT with its cap GOOD or LOOSE, the other not in the pan at all
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PARTS = ("bottle_a", "bottle_b")
STARTS = {"bottle_a": L.A0, "bottle_b": L.B0}
TIGHT, GOOD, LOOSE, OFF = "tight", "good", "loose", "off"


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
    """One bottle at one step (m, rad). Positions in the world frame with the table top at z 0."""
    pos: np.ndarray  # base frame origin: the centre of the base's underside
    rot: np.ndarray  # base rotation
    turn: float  # the cap's turn from tight (rad, unscrewing positive)
    rise: float  # the cap's lift off its seat (m): its gap
    lateral: float  # the cap's axis off the bottle's axis (m)
    distance: object  # AssemblyState.distance (points -> metres to the nearest part), or None


def read(states, name: str) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        s = states.get(name)
        if s is None:
            return None
        o = np.asarray(s.origin, dtype=float).ravel()
        r = rotation(s.quat_xyzw)
        tu, ri = float(s.joints.get("turn")), float(s.joints.get("rise"))
        cap = np.asarray(s.link_pose("turn")[1], dtype=float).ravel()
        if o.size != 3 or cap.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))
                                                and np.all(np.isfinite(cap)) and math.isfinite(tu)
                                                and math.isfinite(ri)):
            return None
        local = np.asarray(s.to_local(cap), dtype=float).ravel()
        lateral = float(math.hypot(float(local[0]), float(local[1])))
        if not math.isfinite(lateral):
            return None
        return Reading(o - np.array([0.0, 0.0, L.TOP]), r, tu, ri, lateral, getattr(s, "distance", None))
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None


# -- the readings ------------------------------------------------------------------------------------------------------
def tilt_of(r: Reading) -> float:
    return tilt_deg(r.rot)


def fell(r: Reading) -> bool:
    return float(r.pos[2]) < -L.FALL_DROP


def tipped(r: Reading) -> bool:
    return tilt_of(r) > L.TIP_DEG + L.EPS


def upright(r: Reading) -> bool:
    return bool(tilt_of(r) <= L.STAND_TILT_DEG + L.EPS)


def in_pan(r: Reading) -> bool:
    """The base circle inside the pan's inner walls and the base within 3 mm of the floor (and not sunk into it)."""
    x, y, z = (float(v) for v in r.pos)
    return bool(L.PAN_X0 + L.BODY_R - L.EPS <= x <= L.PAN_X1 - L.BODY_R + L.EPS
                and L.PAN_Y0 + L.BODY_R - L.EPS <= y <= L.PAN_Y1 - L.BODY_R + L.EPS
                and L.PAN_FLOOR - 0.002 <= z <= L.PAN_FLOOR + L.FLOOR_TOL + L.EPS)


def in_pan_zone(r: Reading) -> bool:
    """Resting in the pan, upright or not: the base centre inside the inner walls and below the rim."""
    x, y, z = (float(v) for v in r.pos)
    return bool(L.PAN_X0 <= x <= L.PAN_X1 and L.PAN_Y0 <= y <= L.PAN_Y1 and z < L.PAN_H)


def turn_of(r: Reading) -> float:
    """The cap's turn from tight, in degrees (unscrewing positive)."""
    return math.degrees(r.turn)


def on_thread(r: Reading) -> bool:
    """Turned -8 to 540 degrees from tight, risen what that turn lifts it (within 2 mm), axis on the bottle's."""
    deg = turn_of(r)
    return bool(-math.degrees(L.TIGHTEN_RAD) - 1.0 <= deg <= L.RELEASE_DEG + L.EPS
                and r.lateral <= L.CAP_LATERAL + L.EPS
                and abs(r.rise - L.gap_for_turn(max(r.turn, 0.0))) <= L.THREAD_TOL + L.EPS)


def cap_class(r: Reading) -> str:
    if not on_thread(r):
        return OFF
    deg = turn_of(r)
    if deg < L.SEALED_DEG - L.EPS:
        return TIGHT
    if deg <= L.ONE_TURN_DEG + L.EPS:
        return GOOD
    return LOOSE


def rise_target(turn_rad) -> float:
    """The thread's lift (m) for a cap turned ``turn_rad`` from tight: PITCH a turn, none below tight, stopped at the
    thread's stop; 0 for a turn that cannot be read."""
    try:
        t = float(turn_rad)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(t):
        return 0.0
    return L.gap_for_turn(min(max(t, 0.0), L.STOP_RAD))


def tier(latched: bool, settled: bool, pan: list[tuple[bool, bool, str]], other_in_zone: bool = False) -> float:
    """``pan``: one (in_pan_and_upright, in_pan_zone, cap_class) per bottle. ``settled``: released and still."""
    if latched:
        return 0.0
    if any(zone and cls == TIGHT for _, zone, cls in pan):
        return 0.0  # a sealed bottle loaded for the cycle
    up = [(ok, cls) for ok, _, cls in pan if ok]
    if len(up) == len(pan) == 2:
        classes = [cls for _, cls in up]
        if OFF in classes:
            return 0.5
        if all(c == GOOD for c in classes) and settled:
            return 1.0
        return 0.75
    if len(up) == 1 and len(pan) == 2 and up[0][1] in (GOOD, LOOSE):
        return 0.25 if not any(zone for ok, zone, _ in pan if not ok) else 0.0
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
    """The latches (fell, tipped), the last 1 s of readings (stillness) and of the let-go check, and the reset
    fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"FELL": None, "TIPPED": None}
        self.max_gap = {n: 0.0 for n in PARTS}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: dict | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(name: str, r: Reading) -> bool:
        x, y = STARTS[name]
        return bool(math.hypot(float(r.pos[0] - x), float(r.pos[1] - y)) <= L.START_TOL
                    and abs(float(r.pos[2])) <= L.START_TOL and abs(r.turn) <= 0.05 and r.rise <= 0.002)

    def teleported(self, prev: dict, now: dict) -> bool:
        """A bottle back on its mark with the cap tight again by a jump no trial makes in one step (a reset): its
        base moved 50 mm or its cap's turn changed by more than a radian."""
        for n in PARTS:
            if self.at_start(n, now[n]) and (
                    float(np.linalg.norm(now[n].pos - prev[n].pos)) >= L.RESET_JUMP
                    or abs(now[n].turn - prev[n].turn) >= L.RESET_TURN):
                return True
        return False

    def update(self, states, arms) -> Result:
        self.steps += 1
        rs = {n: read(states, n) for n in PARTS}
        if any(v is None for v in rs.values()):
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, rs):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(rs)
        self.prev = rs
        for n, r in rs.items():
            if self.latch["FELL"] is None and fell(r):
                self.latch["FELL"] = self.steps
            if self.latch["TIPPED"] is None and tipped(r):
                self.latch["TIPPED"] = self.steps
            self.max_gap[n] = max(self.max_gap[n], r.rise)
        ok, gap = False, None
        try:
            dists = [rs[n].distance for n in PARTS]
            if all(d is not None for d in dists):
                def dist(q, _d=dists):
                    return np.min([np.asarray(f(q), dtype=float) for f in _d], axis=0)

                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not released
            ok, gap = False, None
        self.released.append(ok)
        self.last = self.judge(rs, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of any base, the largest change of any cap's turn
        (deg) and rise (m), from where each was at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        out = {"move": 0.0, "turn": 0.0, "cap_turn": 0.0, "cap_rise": 0.0}
        for x in h[1:]:
            for n in PARTS:
                out["move"] = max(out["move"], float(np.linalg.norm(x[n].pos - first[n].pos)))
                out["turn"] = max(out["turn"], turn_deg(first[n].rot, x[n].rot))
                out["cap_turn"] = max(out["cap_turn"], abs(math.degrees(x[n].turn - first[n].turn)))
                out["cap_rise"] = max(out["cap_rise"], abs(x[n].rise - first[n].rise))
        return out

    def judge(self, rs: dict, gap) -> Result:
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS
                     and m["cap_turn"] <= L.STILL_TURN_DEG + L.EPS and m["cap_rise"] <= L.STILL_MOVE + L.EPS)
        pan = [(in_pan(rs[n]) and upright(rs[n]), in_pan_zone(rs[n]), cap_class(rs[n])) for n in PARTS]
        level = tier(self.latched, released and still, pan)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "in_pan_upright": {n: p[0] for n, p in zip(PARTS, pan)},
            "in_pan_zone": {n: p[1] for n, p in zip(PARTS, pan)},
            "cap_class": {n: p[2] for n, p in zip(PARTS, pan)},
            "gap_mm": {n: _mm(rs[n].rise) for n in PARTS},
            "turn_deg": {n: round(math.degrees(rs[n].turn), 2) for n in PARTS},
            "max_gap_mm": {n: _mm(self.max_gap[n]) for n in PARTS},
            "cap_lateral_mm": {n: _mm(rs[n].lateral) for n in PARTS},
            "tilt_deg": {n: round(tilt_of(rs[n]), 2) for n in PARTS},
            "bottle_mm": {n: _mmv(rs[n].pos) for n in PARTS},
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                              "cap_turn_deg": round(m["cap_turn"], 3),
                                              "cap_rise_mm": _mm(m["cap_rise"])},
        }
        return Result(level, bool(level == 1.0), details)
