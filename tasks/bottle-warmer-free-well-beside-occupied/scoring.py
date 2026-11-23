"""The scorer, as plain functions of the two bottles', the warmer's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the neighbour's bottle and the loose bottle (base frame at the centre of the base's
underside, local z up the axis) and of the warmer (fixed; its two push buttons are its joints ``btn_a`` and ``btn_b``).
The arms are read only for the let-go check. The warmer's outline and the wells are design values.

Over the whole trial (each latched: nothing later undoes it):
    F1 FELL        0: either bottle's lowest point more than 50 mm below the counter's top
    F2 OUT         0: the neighbour's centre 12 mm off its well's axis, or its base 30 mm up (it left its well)
    F3 SPILL       0: either bottle's axis tilts more than 30 degrees
    D  DISTURBED   caps the score: the neighbour's centre was ever 3 mm or more off where it stood, its base ever 2 mm
                   up, its axis ever more than 3 degrees off vertical, or its own button ever pushed 1 mm
    PUSH           the free well's button pushed in 3 mm or more; a push counts again only after the button has come
                   back out to 1 mm or less (it springs back). Like the real well's switch, a push turns the well on
                   and the next push turns it off again
    DRY            caps the score: a push while the loose bottle was not seated (the well started dry)
End state (the last step):
    STARTED        the free well is on: an odd number of pushes
    SEATED         the loose bottle's centre within 6 mm of the free well's axis, its base within 2 mm of the well's
                   floor top, its axis within 3 degrees of vertical
    RESTING        not seated, on the warmer's top (lowest point within 3 mm of it, centre over its outline) or leaning
                   in the free well (centre within 36 mm of its axis, lowest point between its floor and the top)
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from both bottles and the warmer
    STILL          over the last 1 s each bottle moves under 5 mm and turns under 5 degrees (visible motion in a
                   30 fps video); a button is not read for stillness: its state is the latched push count

    0      F1, F2 or F3 ever fired, or the loose bottle neither seated nor resting on the warmer
    1.0    seated, started, released and still
    0.75   seated, started, not yet released and still
    0.5    seated, not started
    0.25   resting on the warmer or leaning in the free well, released and still
    D or DRY: the score becomes at most 0.5 if started, else at most 0.25
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

NEIGHBOUR, BOTTLE, WARMER = "neighbour", "bottle", "warmer"
BOTTLES = (NEIGHBOUR, BOTTLE)
PARTS = (NEIGHBOUR, BOTTLE, WARMER)
_PTS = np.array(L.bottle_pts_local())
START = {NEIGHBOUR: np.array([L.A_X0, L.A_Y0, L.FLOOR]), BOTTLE: np.array([L.BOT_X0, L.BOT_Y0, L.BOT_Z0])}


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
    pos: dict  # bottle -> base frame origin
    rot: dict
    pts: dict  # bottle -> (32, 3) rim points
    btn: dict  # "btn_a" / "btn_b" -> travel (m)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, pts = {}, {}, {}
    try:
        for name in BOTTLES:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            o = o - np.array([0.0, 0.0, L.TOP])
            pos[name], rot[name], pts[name] = o, r, o + _PTS @ r.T
        w = states.get(WARMER)
        if w is None:
            return None
        btn = {k: float(w.joints[k]) for k in ("btn_a", "btn_b")}
        if not all(math.isfinite(v) for v in btn.values()):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, pts, btn)


# -- the readings ------------------------------------------------------------------------------------------------------
def low(r: Reading, name) -> float:
    return float(np.min(r.pts[name][:, 2]))


def tilt(r: Reading, name) -> float:
    return tilt_deg(r.rot[name])


def xy_off(r: Reading, name, centre) -> float:
    return math.hypot(float(r.pos[name][0] - centre[0]), float(r.pos[name][1] - centre[1]))


def fell(r: Reading) -> bool:
    return any(low(r, n) < L.FALL_Z for n in BOTTLES)


def out_of_well(r: Reading) -> bool:
    return bool(xy_off(r, NEIGHBOUR, L.WELL_A) > L.NB_OUT_XY or low(r, NEIGHBOUR) > L.FLOOR + L.NB_OUT_LIFT)


def spilled(r: Reading) -> bool:
    return any(tilt(r, n) > L.SPILL_DEG for n in BOTTLES)


def disturbed(r: Reading) -> bool:
    """The neighbour is off where it stood on this step (the latch keeps it)."""
    return bool(math.hypot(*(r.pos[NEIGHBOUR][:2] - START[NEIGHBOUR][:2])) >= L.NB_MOVE - L.EPS
                or low(r, NEIGHBOUR) - L.FLOOR >= L.NB_LIFT - L.EPS
                or tilt(r, NEIGHBOUR) > L.UPRIGHT_DEG
                or r.btn["btn_a"] >= L.BTN_A_TOUCH - L.EPS)


def seated(r: Reading) -> bool:
    return bool(xy_off(r, BOTTLE, L.WELL_B) <= L.SEAT_XY + L.EPS
                and abs(low(r, BOTTLE) - L.FLOOR) <= L.SEAT_Z + L.EPS
                and tilt(r, BOTTLE) <= L.UPRIGHT_DEG + L.EPS)


def resting(r: Reading) -> bool:
    """Not seated but held up by the warmer: its base on the warmer's top (lowest point within 3 mm of it, centre over
    the outline), or leaning in the free well (centre within the well's 36 mm, lowest point between the floor and the
    top). Supported, not hovering: the tier also needs both hands off and everything still for 1 s."""
    x, y = float(r.pos[BOTTLE][0]), float(r.pos[BOTTLE][1])
    lo = low(r, BOTTLE)
    on_top = L.W_X0 <= x <= L.W_X1 and L.W_Y0 <= y <= L.W_Y1 and abs(lo - L.W_H) <= L.REST_Z + L.EPS
    in_well = xy_off(r, BOTTLE, L.WELL_B) <= L.WELL_IN + L.EPS and L.FLOOR - L.SEAT_Z <= lo <= L.W_H
    return bool(on_top or in_well)


def pressed(r: Reading) -> bool:
    return r.btn["btn_b"] >= L.PRESSED - L.EPS


def let_out(r: Reading) -> bool:
    return r.btn["btn_b"] <= L.REARM + L.EPS


def tier(latched: bool, dist: bool, is_seated: bool, is_resting: bool, started: bool, settled: bool) -> float:
    if latched:
        return 0.0
    if is_seated:
        level = (1.0 if settled else 0.75) if started else 0.5
    elif is_resting and settled:
        level = 0.25
    else:
        return 0.0
    if dist:
        level = min(level, 0.5 if started else 0.25)
    return level


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 1) for c in v]


class Tracker:
    """The latches (F1, F2, F3, D, DRY), the push count (STARTED), the last 1 s of readings (stillness) and of the let-go check, and the
    reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "F2": None, "F3": None}
        self.dist: dict = {"D": None}
        self.started: int | None = None  # the step of the latest push that left the well on (None: off)
        self.pushes = 0
        self.armed = True  # the button is out (1 mm or less): the next push counts
        self.dry: int | None = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        return all(float(np.linalg.norm(r.pos[n] - START[n])) <= L.START_TOL for n in BOTTLES)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The bottles put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in BOTTLES)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch), "disturbed": dict(self.dist)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        for key, fired in (("F1", fell), ("F2", out_of_well), ("F3", spilled)):
            if self.latch[key] is None and fired(r):
                self.latch[key] = self.steps
        if self.dist["D"] is None and disturbed(r):
            self.dist["D"] = self.steps
        if self.armed and pressed(r):
            self.armed = False
            self.pushes += 1
            if not seated(r) and self.dry is None:
                self.dry = self.steps
            self.started = self.steps if self.pushes % 2 == 1 else None
        elif not self.armed and let_out(r):
            self.armed = True
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
        """Over the last 1 s: the largest move (m) and turn (deg) of either bottle from where
        it was at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            for n in BOTTLES:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        is_seated, is_resting = seated(r), resting(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, self.dist["D"] is not None or self.dry is not None, is_seated, is_resting,
                     self.started is not None, released and still)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch), "disturbed": dict(self.dist),
            "started_step": self.started, "pushes": self.pushes, "dry_step": self.dry, "seated": is_seated, "resting": is_resting,
            "pressed_now": pressed(r), "btn_a_mm": _mm(r.btn["btn_a"]), "btn_b_mm": _mm(r.btn["btn_b"]),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "bottle_mm": _mmv(r.pos[BOTTLE]), "bottle_tilt_deg": round(tilt(r, BOTTLE), 2),
            "bottle_low_mm": _mm(low(r, BOTTLE)), "bottle_off_well_mm": _mm(xy_off(r, BOTTLE, L.WELL_B)),
            "neighbour_mm": _mmv(r.pos[NEIGHBOUR]), "neighbour_tilt_deg": round(tilt(r, NEIGHBOUR), 2),
            "neighbour_moved_mm": _mm(xy_off(r, NEIGHBOUR, L.WELL_A)),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
