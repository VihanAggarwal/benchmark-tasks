"""The scorer, as plain functions of the syringe's, the tray's, the forceps' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of each loose part: the syringe (base frame at the needle's tip, local z up its axis), the
tray (base frame at the centre of its underside) and the forceps. Every reading is a pose; the arms are read only for
the needle-contact gate and the let-go check.

    H1 SHARP IN THE WASTE   latched: any part of the syringe inside the bin's inner outline seen from above while
                            its lowest point is below the bin's rim (in the tray or loose)
    H2 OFF THE TABLE        latched: the syringe's centre outside the table's outline, or its lowest point more
                            than 10 mm below the table top
    H3 NEEDLE TOUCHED       latched: a gripper's finger region (within rigstate's finger-zone radius of the
                            gripper's axis, from the pad tips 38 mm back toward the TCP) comes within 2 mm of the
                            needle's collider. Fail closed: a step whose arms cannot be read counts as a touch.
    SYRINGE IN              the syringe's whole box (world-aligned, around every shape) inside the container: within
                            its inner walls (2 mm allowed into a wall or the floor: contact) and below the lid's
                            underside
    TRAY IN                 the tray's whole box inside the bin's inner outline (2 mm allowed into a wall) and below
                            its rim, its lowest point within 20 mm of the bin's floor
    FORCEPS IN              the forceps' whole box inside the bin's inner outline (the same 2 mm) and below its rim
    RELEASED                on every step of the last 1 s both finger zones are 25 mm or more from the tray, the
                            forceps and the syringe (unreadable arms: not released)
    STILL                   over the last 1 s the tray and the forceps moved under 3 mm and turned under 3 degrees

    0      H1, H2 or H3 ever fired
    1.0    syringe in, tray in, forceps in, released and still     (the trial ends once this has held 1 s)
    0.75   syringe in, tray in (the forceps elsewhere, or not yet let go and still)
    0.5    syringe in, the tray not in the bin (a safe stop: the sharp is secured, the tray still on the table)
    0      anything else: nothing done, or the syringe not wholly in the container (left on the table, hanging in
           the opening, still held), whatever happened to the tray. A used sharp left exposed earns nothing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SYRINGE, TRAY, FORCEPS = "syringe", "tray", "forceps"
PARTS = (SYRINGE, TRAY, FORCEPS)
NEEDLE_SAMPLES = 17  # points along the needle's axis, 1 mm apart


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
    pos: dict  # part -> base frame origin
    rot: dict  # part -> rotation
    low: dict  # part -> world-aligned box's low corner
    high: dict  # part -> its high corner
    centre: dict  # part -> box centre (x, y)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, low, high, centre = {}, {}, {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            lo = np.asarray(s.low, dtype=float).ravel()
            hi = np.asarray(s.high, dtype=float).ravel()
            if o.size != 3 or lo.size != 3 or hi.size != 3:
                return None
            if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and np.all(np.isfinite(lo))
                    and np.all(np.isfinite(hi))):
                return None
            off = np.array([0.0, 0.0, L.TOP])
            pos[name], rot[name], low[name], high[name] = o - off, r, lo - off, hi - off
            centre[name] = ((lo + hi) / 2)[:2]
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, low, high, centre)


# -- the readings ------------------------------------------------------------------------------------------------------
def needle_points(r: Reading) -> np.ndarray:
    """NEEDLE_SAMPLES world points (table top at z 0) along the bare needle, tip to hub."""
    o, rot = r.pos[SYRINGE], r.rot[SYRINGE]
    t = np.linspace(0.0, L.NEEDLE_L, NEEDLE_SAMPLES)
    return o + np.outer(t, rot[:, 2])


def finger_region_distance(points, tcp, tips, opening) -> float:
    """Smallest distance (m) from ``points`` to one gripper's finger region: the solid cylinder of radius
    rigstate.zone_radius(opening) about the gripper's axis, from the pad-tip point FINGER_LEN back toward the TCP."""
    tcp, tips = np.asarray(tcp, dtype=float).reshape(3), np.asarray(tips, dtype=float).reshape(3)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    w = np.atleast_2d(np.asarray(points, dtype=float)) - tips
    back = -(w @ u)  # how far back from the pad tips toward the TCP
    perp = np.linalg.norm(w + np.outer(back, u), axis=1)
    da = np.maximum(np.maximum(-back, back - L.FINGER_LEN), 0.0)
    dr = np.maximum(perp - rigstate.zone_radius(opening), 0.0)
    return float(np.min(np.hypot(da, dr)))


def needle_gap(r: Reading, arms) -> float | None:
    """The nearer finger region's distance to the needle's collider surface (m), or None when the arms are
    unreadable. The arms' positions are in the env frame (the table top at z TOP)."""
    read_ = rigstate.read_arms(arms)
    if read_ is None:
        return None
    pts = needle_points(r) + np.array([0.0, 0.0, L.TOP])
    return min(finger_region_distance(pts, tcp, tips, op) for tcp, tips, op in read_) - L.NEEDLE_R


def needle_touched(r: Reading, arms) -> bool:
    gap = needle_gap(r, arms)
    return gap is None or not math.isfinite(gap) or gap <= L.NEEDLE_TOUCH + L.EPS


def in_bin_outline(lo, hi) -> bool:
    """Any part of the box (lo, hi) inside the bin's inner outline seen from above."""
    x0, x1, y0, y1, _, _ = L.bin_inner()
    return lo[0] < x1 and hi[0] > x0 and lo[1] < y1 and hi[1] > y0


def sharp_in_waste(r: Reading) -> bool:
    lo, hi = r.low[SYRINGE], r.high[SYRINGE]
    return in_bin_outline(lo, hi) and lo[2] < L.BIN_H - L.EPS


def off_table(r: Reading, name: str = SYRINGE) -> bool:
    x, y = r.centre[name]
    if not (L.TABLE_X0 <= x <= L.TABLE_X1 and L.TABLE_Y0 <= y <= L.TABLE_Y1):
        return True
    return r.low[name][2] < -L.OFF_TABLE_DROP


def syringe_in(r: Reading) -> bool:
    x0, x1, y0, y1, zf, zl = L.container_inner()
    lo, hi = r.low[SYRINGE], r.high[SYRINGE]
    w = L.WALL_TOL
    return bool(lo[0] >= x0 - w and hi[0] <= x1 + w and lo[1] >= y0 - w and hi[1] <= y1 + w
                and lo[2] >= zf - 0.002 and hi[2] <= zl + L.EPS)


def wholly_in_bin(r: Reading, name: str) -> bool:
    x0, x1, y0, y1, _, rim = L.bin_inner()
    lo, hi = r.low[name], r.high[name]
    w = L.WALL_TOL
    return bool(lo[0] >= x0 - w and hi[0] <= x1 + w and lo[1] >= y0 - w and hi[1] <= y1 + w
                and hi[2] <= rim + L.EPS)


def tray_in(r: Reading) -> bool:
    return wholly_in_bin(r, TRAY) and r.low[TRAY][2] <= L.BIN_FLOOR + L.TRAY_FLOOR_TOL + L.EPS


def forceps_in(r: Reading) -> bool:
    return wholly_in_bin(r, FORCEPS)


def tier(latched: bool, s_in: bool, t_in: bool, f_in: bool, released: bool, still: bool) -> float:
    if latched:
        return 0.0
    if s_in and t_in:
        return 1.0 if (f_in and released and still) else 0.75
    if s_in:
        return 0.5
    return 0.0  # the syringe not secured in the container: no credit, even with the tray binned (revision 2026-11-15b)


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
    """The latches (H1 to H3), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"H1": None, "H2": None, "H3": None}  # the step each fired at
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.min_needle_gap: float | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        s = np.array([L.SYR_X, L.SYR_Y, L.SYR_TIP_Z])
        t = np.array([L.TRAY_X, L.TRAY_Y, 0.0])
        return (float(np.linalg.norm(r.pos[SYRINGE] - s)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[TRAY] - t)) <= L.START_TOL)

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
        gap = needle_gap(r, arms)
        if gap is not None and (self.min_needle_gap is None or gap < self.min_needle_gap):
            self.min_needle_gap = gap
        if self.latch["H1"] is None and sharp_in_waste(r):
            self.latch["H1"] = self.steps
        if self.latch["H2"] is None and off_table(r):
            self.latch["H2"] = self.steps
        if self.latch["H3"] is None and needle_touched(r, arms):
            self.latch["H3"] = self.steps
        try:
            dists = [states[n].distance for n in PARTS]

            def dist(points):
                return np.min([d(points) for d in dists], axis=0)
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok = False
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the tray and of the forceps from where they were
        at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            for n in (TRAY, FORCEPS):
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        s_in, t_in, f_in = syringe_in(r), tray_in(r), forceps_in(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, s_in, t_in, f_in, released, still)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "syringe_in": s_in, "tray_in": t_in, "forceps_in": f_in,
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "needle_gap_mm": _mm(gap), "min_needle_gap_mm": _mm(self.min_needle_gap),
            "syringe_mm": _mmv(r.pos[SYRINGE]), "syringe_tilt_deg": round(tilt_deg(r.rot[SYRINGE]), 2),
            "syringe_low_mm": _mmv(r.low[SYRINGE]), "syringe_high_mm": _mmv(r.high[SYRINGE]),
            "tray_mm": _mmv(r.pos[TRAY]), "tray_tilt_deg": round(tilt_deg(r.rot[TRAY]), 2),
            "tray_low_z_mm": _mm(r.low[TRAY][2]), "forceps_mm": _mmv(r.pos[FORCEPS]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 2)},
        }
        return Result(level, bool(level == 1.0), details)
