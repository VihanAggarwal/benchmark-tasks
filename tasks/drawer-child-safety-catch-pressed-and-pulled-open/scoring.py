"""The scorer, as plain functions of the unit's and the catch's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the unit (base frame at the carcass's front face, bottom, centre; joint "drawer", the
slider's travel in m, 0 shut; joint "latch", the latch arm's angle in rad, positive pressed) and of the fixed catch. Every reading is
a joint value or a link pose; the arms are read only for the let-go check.

    OPEN           the drawer's opening: the gap between the back of its front and the carcass's front edge (the
                   slider's travel minus the inset front's 15 mm)
    F1 FORCED      latched: the latch's hook more than 2 mm into the catch's lip in both x and z (the solver let the
                   hook through the catch: in a real trial the latch or the catch broke), or the hook jumped from
                   behind the lip to in front of it in one step with the arm not pressed
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from the unit (drawer, handle and
                   latch included) and the catch
    STILL          over the last 1 s the drawer moved under 5 mm

    0      F1 ever fired, or the drawer open less than 60 mm at the end
    1.0    open 100 mm or more, released and still
    0.5    open 60 to 100 mm, released and still
    0.25   open 60 mm or more, but not released or not still
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import parts as P
import rigstate

UNIT, CATCH = "unit", "catch"
PARTS = (UNIT, CATCH)
DRAWER, LATCH = "drawer", "latch"

_HOOK_C, _HOOK_S = P.hook_box()
_LIP_LO, _LIP_HI = (np.asarray(v, dtype=float) for v in P.lip_bounds())


def _corners(center, size) -> np.ndarray:
    c, h = np.asarray(center, dtype=float), np.asarray(size, dtype=float) / 2
    return np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


_HOOK_CORNERS = _corners(_HOOK_C, _HOOK_S)


@dataclass(frozen=True)
class Reading:
    """One step (m, rad)."""
    opening: float  # the back of the front past the carcass's front edge (travel - FRONT_BACK)
    angle: float
    hook_lo: np.ndarray  # the hook's box in the unit's base frame now (world-aligned in that frame)
    hook_hi: np.ndarray
    origin: np.ndarray  # the unit's base origin (world, table top at z 0)
    catch_origin: np.ndarray
    travel: float = 0.0  # the slider's joint value (0 shut)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        u, c = states.get(UNIT), states.get(CATCH)
        if u is None or c is None:
            return None
        o = float(u.joints[DRAWER])
        a = float(u.joints[LATCH])
        org = np.asarray(u.origin, dtype=float).ravel()
        corg = np.asarray(c.origin, dtype=float).ravel()
        if not (math.isfinite(o) and math.isfinite(a)) or org.size != 3 or corg.size != 3:
            return None
        if not (np.all(np.isfinite(org)) and np.all(np.isfinite(corg))):
            return None
        world = u.link_to_world(LATCH, _HOOK_CORNERS)
        local = u.to_local(world)
        if not np.all(np.isfinite(local)):
            return None
    except (AttributeError, TypeError, ValueError, KeyError, RuntimeError):
        return None
    top = np.array([0.0, 0.0, L.TOP])
    return Reading(o - L.FRONT_BACK, a, local.min(axis=0), local.max(axis=0), org - top, corg - top, o)


def overlap(r: Reading) -> np.ndarray:
    """How far (m) the hook's box reaches into the lip's box along x, y and z (0 where they are apart)."""
    return np.maximum(0.0, np.minimum(r.hook_hi, _LIP_HI) - np.maximum(r.hook_lo, _LIP_LO))


def forced(r: Reading, prev: Reading | None = None) -> bool:
    ov = overlap(r)
    if ov[1] > 0.0 and ov[0] > L.FORCE_OVERLAP and ov[2] > L.FORCE_OVERLAP:
        return True
    if prev is None:
        return False
    # tunnelling: the hook's front face behind the strike face one step, its back face in front of the lip the next,
    # with the hook standing more than 2 mm into the lip's height at both steps
    jumped = prev.hook_lo[0] >= _LIP_HI[0] - L.FORCE_OVERLAP and r.hook_hi[0] <= _LIP_LO[0]
    high = (prev.hook_hi[2] > _LIP_LO[2] + L.FORCE_OVERLAP and r.hook_hi[2] > _LIP_LO[2] + L.FORCE_OVERLAP)
    return bool(jumped and high)


def tier(latched: bool, opening: float, settled: bool) -> float:
    if latched or opening < L.OPEN_PAST - L.EPS:
        return 0.0
    if not settled:
        return 0.25
    if opening >= L.OPEN_FULL - L.EPS:
        return 1.0
    return 0.5


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


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
        self.max_opening = 0.0
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return abs(r.travel) <= L.START_TOL and abs(r.angle) <= 0.05

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The drawer put back shut with a jump no trial makes in one step (a reset)."""
        return cls.at_start(now) and prev.opening - now.opening >= L.RESET_JUMP

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
        if self.latch["F1"] is None and forced(r, self.prev):
            self.latch["F1"] = self.steps
        self.hist.append(r)
        self.prev = r
        self.max_opening = max(self.max_opening, r.opening)
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

    def motion(self) -> float | None:
        """Over the last 1 s: the drawer's largest move (m) from where it was at the window's start; None before 1 s
        of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(abs(x.opening - h[0].opening) for x in h[1:])

    def judge(self, r: Reading, gap) -> Result:
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m <= L.STILL_MOVE + L.EPS)
        level = tier(self.latched, r.opening, released and still)
        ov = overlap(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "opening_mm": _mm(r.opening), "max_opening_mm": _mm(self.max_opening),
            "latch_deg": round(math.degrees(r.angle), 2),
            "hook_front_mm": _mm(float(r.hook_lo[0])), "hook_top_mm": _mm(float(r.hook_hi[2])),
            "hook_in_lip_mm": [_mm(float(v)) for v in ov],
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "drawer_move_1s_mm": _mm(m), "hands_gap_mm": _mm(gap),
        }
        return Result(level, bool(level == 1.0), details)
