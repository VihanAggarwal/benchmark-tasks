"""The scorer, as plain functions of the two plates', the draining board's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyStates of the two plates (each plate's body frame on its axis at mid-depth, drawn leaning 12 degrees
back as found) and of the draining board (a fixed base with one hinged link, ``rack``: the plate rack, 0 folded flat,
90 degrees upright). Every reading is a pose, a velocity or the rack's angle; the arms are read only for the let-go
check. Each plate is read as a cloud of points on both of its faces (layout.plate_points: the real dished shape).

    F1 FELL        latched: a plate's downward speed reached 0.9 m/s at a step (41 mm of free fall reaches it; every
                   free fall of 100 mm or more shows it at a 20 Hz step), or its lowest point went more than 20 mm
                   below the counter
    RACK UP        the plate rack at 85 degrees or more from flat (the lock holds it at 90)
    LOW            a plate's lowest point over the board's deck
    TILT           the angle of a plate's plane from vertical
    PLACED         RACK UP, the lowest point within 5 mm of the deck, every point of the plate within 10 mm of its
                   lowest point between the inner faces of one slot's two prongs (0.5 mm allowed), the plane within 25
                   degrees of vertical, the centre line 30 to 70 mm beyond the prong row (y 110 to 150 mm), so the
                   prongs cross the plate well inside its rim
    UPRIGHT        the plane within 40 degrees of vertical (standing or leaning on its rim, not lying down)
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from both plates and the rack
    STILL          over the last 1 s no point of either plate (its whole surface cloud) and no point of the rack
                   (its prong tips, 99 mm from the hinge, move most) moved 2 mm or more from where it was at the
                   window's start

    0      F1 ever fired, or not released and still (a held plate or a held rack earns nothing), or the rack not up
    1.0    both plates PLACED, in different slots
    0.5    one slot holds a placed plate (or both plates are placed in the same slot) and every plate is UPRIGHT
    0.25   no plate placed and every plate UPRIGHT (the rack raised and locked, nothing in it yet), or one slot
           holds a placed plate and the other plate lies down
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PLATES = L.PL_NAMES
BOARD = "drainer"
RACK = "rack"
_AXIS0 = np.asarray(L.AXIS0)
_REST = {n: np.array([L.LEAN_X, L.PL_Y[n], L.LEAN_Z]) for n in PLATES}  # each body origin at spawn (counter = 0)


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
    """One step (m). Positions in the world frame with the counter at z 0."""
    pos: dict  # plate -> body origin
    rot: dict  # plate -> rotation
    vz: dict  # plate -> vertical velocity of its body
    rack_deg: float  # the plate rack's angle from flat


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, vz = {}, {}, {}
    try:
        for name in PLATES:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            v = np.asarray(s.lin_vel, dtype=float).ravel()
            if o.size != 3 or v.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))
                                                  and np.all(np.isfinite(v))):
                return None
            pos[name], rot[name], vz[name] = o - np.array([0.0, 0.0, L.TOP]), r, float(v[2])
        b = states.get(BOARD)
        if b is None:
            return None
        q = float(b.joints[RACK])
        if not math.isfinite(q):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, vz, math.degrees(q))


# -- the readings ------------------------------------------------------------------------------------------------------
def normal(r: Reading, name: str) -> np.ndarray:
    return r.rot[name] @ _AXIS0


def tilt_deg(r: Reading, name: str) -> float:
    """The angle of the plate's plane from vertical (0: standing, 90: lying flat)."""
    nz = abs(float(normal(r, name)[2]))
    return math.degrees(math.asin(max(0.0, min(1.0, nz))))


def points(r: Reading, name: str) -> np.ndarray:
    return L.plate_points(r.pos[name], r.rot[name])


def bottom(r: Reading, name: str) -> dict:
    """``z`` the plate's lowest point over the counter, ``x0``/``x1`` the x extent of every point within 10 mm of it,
    ``y`` the plate's centre line (its body origin's y)."""
    p = points(r, name)
    z = float(p[:, 2].min())
    band = p[p[:, 2] <= z + L.BOTTOM_BAND]
    return {"z": z, "x0": float(band[:, 0].min()), "x1": float(band[:, 0].max()), "y": float(r.pos[name][1])}


def slot_of(b: dict) -> int | None:
    """The slot whose inner faces hold the plate's bottom (0.5 mm allowed), else None."""
    for k in range(L.N_SLOTS):
        if b["x0"] >= L.SLOT_LO[k] - L.SLOT_TOL and b["x1"] <= L.SLOT_HI[k] + L.SLOT_TOL:
            return k
    return None


def rack_up(r: Reading) -> bool:
    return r.rack_deg >= L.RACK_UP_DEG - L.EPS


def reading_of(r: Reading, name: str) -> dict:
    b = bottom(r, name)
    tilt = tilt_deg(r, name)
    low = b["z"] - L.DECK  # over the deck
    slot = slot_of(b)
    y = b["y"]
    placed = bool(rack_up(r) and slot is not None and abs(low) <= L.PLACED_LOW + L.EPS
                  and tilt <= L.PLACED_TILT_DEG + L.EPS and L.PLACED_Y0 - L.EPS <= y <= L.PLACED_Y1 + L.EPS)
    upright = bool(tilt <= L.UPRIGHT_DEG + L.EPS)
    return {"slot": slot, "placed": placed, "upright": upright, "tilt": tilt, "low": low, "y": y,
            "bottom_x": (b["x0"] + b["x1"]) / 2}


def fell(r: Reading) -> bool:
    for n in PLATES:
        if r.vz[n] <= -L.FALL_SPEED or bottom(r, n)["z"] < -L.OFF_TABLE:
            return True
    return False


def tier(latched: bool, settled: bool, up: bool, plates: dict) -> float:
    if latched or not settled or not up:
        return 0.0
    slots = {p["slot"] for p in plates.values() if p["placed"]}
    upright = all(p["upright"] for p in plates.values())
    if len(slots) == 2:
        return 1.0
    if len(slots) == 1:
        return 0.5 if upright else 0.25
    return 0.25 if upright else 0.0


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
        return all(float(np.linalg.norm(r.pos[n] - _REST[n])) <= L.START_TOL for n in PLATES) and r.rack_deg < 1.0

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The plates put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in PLATES)

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
            dists = [states[n].distance for n in PLATES]
            board = states[BOARD]

            def dist(pts):
                return np.min([d(pts) for d in dists] + [board.distance(pts, bodies=(RACK,))], axis=0)
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
        """Over the last 1 s: the largest move (m) of any point of either plate and of the rack's prong tips from where
        it was at the window's start (plus the plates' largest turn and the rack's, in degrees, for the trace); None
        before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        p0 = {n: L.plate_points(first.pos[n], first.rot[n]) for n in PLATES}
        move = turn = rack = 0.0
        for x in h[1:]:
            rack = max(rack, abs(x.rack_deg - first.rack_deg))
            for n in PLATES:
                pts = L.plate_points(x.pos[n], x.rot[n])
                move = max(move, float(np.linalg.norm(pts - p0[n], axis=1).max()))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
        rack_move = 2.0 * L.RACK_TIP_R * math.sin(math.radians(rack) / 2.0)  # chord swept by a prong tip
        return {"move": move, "turn": turn, "rack": rack, "rack_move": rack_move}

    def judge(self, r: Reading, gap) -> Result:
        plates = {n: reading_of(r, n) for n in PLATES}
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] < L.STILL_MOVE and m["rack_move"] < L.STILL_MOVE)
        up = rack_up(r)
        level = tier(self.latched, released and still, up, plates)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch), "rack_deg": round(r.rack_deg, 2),
            "rack_up": up, "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                              "rack_deg": round(m["rack"], 3), "rack_move_mm": _mm(m["rack_move"])},
        }
        for i, n in enumerate(PLATES, 1):
            p = plates[n]
            details.update({
                f"p{i}_slot": p["slot"], f"p{i}_placed": p["placed"], f"p{i}_upright": p["upright"],
                f"p{i}_tilt_deg": round(p["tilt"], 2), f"p{i}_low_mm": _mm(p["low"]), f"p{i}_y_mm": _mm(p["y"]),
                f"p{i}_bottom_x_mm": _mm(p["bottom_x"]), f"p{i}_vz_mps": round(r.vz[n], 3),
                f"p{i}_mm": _mmv(r.pos[n]),
            })
        return Result(level, bool(level == 1.0), details)
