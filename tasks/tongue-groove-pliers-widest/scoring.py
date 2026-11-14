"""The scorer, as plain functions of the pliers' state and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
pliers' mechkit AssemblyState (the slider ``slip``: the rivet's travel along the slot, 0 in groove 1 at the slot's end
nearest the jaws, 0.024 m in groove 5 at its end nearest the handles; the hinge ``spread``: 0 as drawn at the 30 deg
start; the three bodies' poses and shapes) and both arms through rigstate.

    SPREAD      sigma = 30 deg + the hinge reading
    GROOVE n    the rivet's travel within 1.5 mm of groove n's centre, 6 (n - 1) mm
    LOCKED      sigma under 72 deg (the tongue is down in a groove)
    FLAT        each half (the slotted half, the rivet half): tilt under 3 deg and its lowest point within 2 mm of
                the bench top
    RELEASED    on every step of the last 1 s, both grippers' finger zones at least 25 mm from every shape of the
                pliers (rigstate.released with the pliers' own distance; unreadable arms or a distance that cannot be
                computed are NOT released)
    STILL       over the last 1 s neither half nor the rivet moved more than 1 mm from where it was at the window's
                start, and neither half turned more than 1 deg
    SLID        latched: the rivet's travel reached 3 mm with the joint open: sigma 72 deg or more on that step, or
                on the last readable step before it (the travel read on the first step under 72 deg was made
                while the joint was still free; like the lock rule, this carries over an unreadable step).
                Travel forced against the lock spring of a joint that is not open does not count
    OFF BENCH   now: the lowest point of any shape of any body 50 mm or more below the bench top (judged on the
                scored step, the last one: a tool that fell and was put back is judged where it ends)

    0      off bench at the end
    1.0    locked, groove 5, flat, released and still            (the trial ends once this has held 1 s)
    0.5    locked, groove 2, 3 or 4, flat, released and still
    0.2    slid at some step
    0      anything else (and any step whose pliers state is missing or not finite: the windows start again)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PLIERS = "pliers"
SLIP = "slip"
SPREAD = "spread"
BODIES = ("base", SLIP, SPREAD)
HALVES = ("base", SPREAD)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    """The angle (deg) of the rotation from orientation r0 to r1."""
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def tilt_deg(r: np.ndarray) -> float:
    """How far (deg) a body's own z axis leans from the vertical, either face up: pliers lying flat the other way up
    are flat too."""
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(r[2, 2]))))))


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one pliers state (m and deg)."""
    q: float  # slide travel, m
    sigma: float  # handle spread, deg
    pos: dict  # body -> its frame's origin (world)
    rot: dict  # body -> its rotation
    low: dict  # body -> its lowest point above the bench top (m; negative below it)


def read(s) -> Reading | None:
    """A Reading, or None when the state is missing, incomplete or not finite."""
    if s is None:
        return None
    try:
        joints = getattr(s, "joints", None) or {}
        q, qh = float(joints[SLIP]), float(joints[SPREAD])
        pos, rot, low = {}, {}, {}
        for body in BODIES:
            r, p = s.link_pose(body)
            lo, _ = s.link_bounds(body)
            r, p, lo = np.asarray(r, dtype=float), np.asarray(p, dtype=float).ravel(), np.asarray(lo, dtype=float)
            if r.shape != (3, 3) or p.size != 3 or lo.size != 3:
                return None
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(p)) and np.all(np.isfinite(lo))):
                return None
            pos[body], rot[body], low[body] = p, r, float(lo.ravel()[2] - L.TOP)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    if not (math.isfinite(q) and math.isfinite(qh)):
        return None
    return Reading(q, L.sigma_deg(qh), pos, rot, low)


def pliers_distance(state):
    """The pliers' exact distance function (every shape of the three bodies as they are now) for rigstate.released.
    A state without one raises here: Tracker.update counts it NOT released."""
    return state.distance


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def is_flat(tilts: dict, lows: dict) -> bool:
    """Each half's tilt (deg) under 3 and its lowest point (m above the bench top) within 2 mm of the bench top."""
    return all(tilts[b] < L.FLAT_TILT_DEG and abs(lows[b]) <= L.FLAT_LOW + L.EPS for b in HALVES)


def tier(off_bench: bool, locked: bool, groove: int | None, flat: bool, released: bool, still: bool,
         slid: bool) -> float:
    if off_bench:
        return 0.0
    if locked and flat and released and still:
        if groove == L.GROOVES:
            return 1.0
        if groove in (2, 3, 4):
            return 0.5
    if slid:
        return 0.2
    return 0.0


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The whole-trial latch (slid with the joint open), the off-bench reading of the current step and the last 1 s
    of readings (stillness) and of the release check."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0  # fallback resets seen (teleports back to the start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.slid = False
        self.slid_at: int | None = None
        self.open_prev = False  # the joint read open (72 deg or more) on the last readable step
        self.off_bench = False
        self.off_bench_at: int | None = None
        self.max_q = 0.0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    # -- reset fallback ----------------------------------------------------------------------------------------
    @staticmethod
    def at_start(r: Reading) -> bool:
        start = np.array(L.ORIGIN, dtype=float)
        if float(np.linalg.norm(r.pos["base"] - start)) > L.RESET_START_TOL:
            return False
        return abs(r.q) <= 0.0005 and abs(r.sigma - L.SIGMA0_DEG) <= 1.0

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The simulator put the pliers back on their start pose, groove 1 at 30 deg, with a jump no trial makes in
        one control step: a new episode."""
        if not cls.at_start(now):
            return False
        return (abs(now.q - prev.q) >= L.RESET_SLIDE_JUMP
                or float(np.linalg.norm(now.pos["base"] - prev.pos["base"])) >= L.RESET_JUMP
                or abs(now.sigma - prev.sigma) >= L.RESET_TURN_DEG
                or turn_deg(prev.rot["base"], now.rot["base"]) >= L.RESET_TURN_DEG)

    # -- per step ------------------------------------------------------------------------------------------------
    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        s = states.get(PLIERS) if isinstance(states, dict) else None
        r = read(s)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "slid": self.slid,
                                            "off_bench": self.off_bench})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.max_q = max(self.max_q, r.q)
        open_now = not L.locked(r.sigma)
        if not self.slid and r.q >= L.LATCH_SLIDE - L.EPS and (open_now or self.open_prev):
            self.slid, self.slid_at = True, self.steps  # only a joint opened past the release angle counts
        self.open_prev = open_now
        lowest = min(r.low.values())
        self.off_bench = lowest <= -L.OFF_BENCH + L.EPS  # the current step only: the score is the last step's
        if self.off_bench and self.off_bench_at is None:
            self.off_bench_at = self.steps  # trace only: the first step it was off the bench
        self.hist.append(r)
        self.prev = r
        try:
            dist = pliers_distance(s)
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        self.released.append(dist is not None and rigstate.released(arms, dist, L.LET_GO))
        self.last = self.judge(r, arms, dist)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest distance of each body's origin from where it was at the window's start
        (m) and the largest turn of each half (deg); None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = max(float(np.linalg.norm(x.pos[b] - first.pos[b])) for x in h[1:] for b in BODIES)
        turn = max(turn_deg(first.rot[b], x.rot[b]) for x in h[1:] for b in HALVES)
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, arms, dist) -> Result:
        locked = L.locked(r.sigma)
        groove = L.groove_of(r.q)
        tilts = {b: tilt_deg(r.rot[b]) for b in HALVES}
        flat = is_flat(tilts, r.low)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.off_bench, locked, groove, flat, released, still, self.slid)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        details = {
            "valid_state": True, "score": level, "slide_mm": _mm(r.q), "spread_deg": round(r.sigma, 3),
            "groove": groove, "locked": bool(locked), "flat": bool(flat),
            "tilt_deg": {b: round(t, 3) for b, t in tilts.items()},
            "low_mm": {b: _mm(v) for b, v in r.low.items()}, "released_now": bool(self.released[-1]),
            "released_1s": bool(released), "still_1s": still, "slid": self.slid, "slid_at": self.slid_at,
            "max_slide_mm": _mm(self.max_q), "off_bench": self.off_bench, "off_bench_at": self.off_bench_at,
            "rivet": [round(float(v), 4) for v in r.pos[SLIP]],
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
            "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms),
        }
        return Result(level, bool(level == 1.0), details)
