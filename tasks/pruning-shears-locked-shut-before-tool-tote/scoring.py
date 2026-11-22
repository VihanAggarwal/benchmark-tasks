"""The scorer, as plain functions of the shears' and the arms' state. NumPy only.

``Tracker.update(states, arms, interlock_locked)`` -> Result(score, success, details), one observation
per control step. Reads the shears' mechkit AssemblyState (joints ``jaw`` in rad, ``catch`` in m,
and root pose) and the arms' finger zones via rigstate.

Conditions:
  CLOSED          handle hinge opening <= 5.0 deg (open_deg <= 5.0)
  LOCKED          CLOSED AND thumb safety catch >= 9.0 mm (of 12.0 mm travel) AND interlock engaged
  IN TOTE         every sampled corner of the shears is horizontally inside the 26 x 18 cm tote
                  interior, resting on/above the 4 mm floor (within 3 mm), below the 6.0 cm rim,
                  and lying flat (tilt <= 10 deg)
  TOTE BREACH     any sampled corner of the shears enters below the actual tote rim (z < 6.0 cm
                  above the bench top AND horizontally inside the 26 x 18 cm tote interior) on a
                  step when NOT LOCKED -> LATCHED HARD ZERO for the rest of the trial
  BLADE CONTACT   while the handles are open (open_deg > 5.0 deg), any finger zone of either
                  gripper contacts (distance <= 0 mm) the exposed cutting edge of the bypass
                  blade -> LATCHED HARD ZERO for the rest of the trial
  OFF BENCH       any part of the shears 5 cm or more below the bench top -> 0
  CLEAR & STILL   over the last 1.0 s (20 steps), every finger zone of both grippers is >= 4.0 cm
                  clear of the shears (unreadable arms: not clear), the shears' origin moved
                  <= 2.0 mm, and the shears turned <= 2.0 deg

Score tiers (when not latched hard zero and not OFF BENCH):
  1.00  LOCKED, IN TOTE, CLEAR & STILL for 1.0 s (success = True)
  0.75  LOCKED and IN TOTE (not yet clear and still for 1.0 s)
  0.50  LOCKED (handles closed <= 5 deg and safety catch >= 9 mm), outside tote
  0.25  CLOSED (handles closed <= 5 deg), safety catch not engaged
  0.00  anything else, OFF BENCH, or latched hard zero
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SHEARS, TOTE = "pruning_shears", "garden_tool_tote"
JAW, CATCH = "jaw", "catch"


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    origin: np.ndarray
    rot: np.ndarray
    open_deg: float
    catch_m: float
    pts: np.ndarray
    tilt_deg: float


def _tilt_deg(rot: np.ndarray) -> float:
    cos_t = float(np.clip(rot[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(cos_t))


def _rot_diff_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    tr = float(np.trace(r0.T @ r1))
    cos_a = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def read(states) -> Reading | None:
    try:
        s = states[SHEARS]
        q_jaw = float(s.joints[JAW])
        q_catch = float(s.joints[CATCH])
        o = np.asarray(s.origin, dtype=float).ravel()
        r = np.asarray(s.rotation(), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    if (o.size != 3 or r.shape != (3, 3)
            or not (math.isfinite(q_jaw) and math.isfinite(q_catch)
                    and np.all(np.isfinite(o)) and np.all(np.isfinite(r)))):
        return None
    open_deg = L.open_deg_of(q_jaw)
    pts = L.shears_world_points(o, r, open_deg, q_catch)
    return Reading(o, r, open_deg, q_catch, pts, _tilt_deg(r))


def is_closed(r: Reading) -> bool:
    return r.open_deg <= L.CLOSED_MAX_DEG + L.EPS


def is_locked(r: Reading, interlock_locked: bool) -> bool:
    return bool(interlock_locked) and is_closed(r) and r.catch_m >= L.CATCH_ENGAGE_M - L.EPS


def in_tote(r: Reading) -> bool:
    return L.all_in_tote(r.pts) and r.tilt_deg <= L.TILT_MAX_DEG + L.EPS


def tote_breach(r: Reading, locked_now: bool) -> bool:
    """True when any part of UNLOCKED shears enters below the actual 6.0 cm tote rim inside the 26 x 18 cm tote."""
    if locked_now:
        return False
    return bool(np.any(L.point_in_tote_interior(r.pts)))


def blade_contact(r: Reading, arms) -> bool:
    """True when handles are open (> 5 deg) and a readable gripper finger zone contacts the exposed blade edge."""
    if is_closed(r):
        return False
    c_w, size, rot_w = L.blade_edge_world_box(r.origin, r.rot, r.open_deg)

    def blade_dist(points):
        return rigstate.box_distances(points, c_w, size, rot_w)

    gap = rigstate.hands_gap(arms, blade_dist)
    return gap is not None and gap <= 1e-6


def off_bench(r: Reading) -> bool:
    return float(np.min(r.pts[:, 2])) < L.TOP - L.OFF_BENCH_DROP


def shears_gap(states, r: Reading, arms) -> float | None:
    """Distance from the nearer gripper finger zone to the shears (m), or None if arms are unreadable."""
    try:
        dist = states[SHEARS].distance
        return rigstate.hands_gap(arms, dist)
    except Exception:  # noqa: BLE001 - fallback to corner cloud distance
        return rigstate.hands_gap(arms, r.pts)


def tier(closed: bool, locked: bool, tote: bool, clear_still: bool, dropped: bool, zero: str) -> float:
    if zero or dropped:
        return 0.0
    if locked and tote:
        return 1.0 if clear_still else 0.75
    if locked:
        return 0.50
    if closed:
        return 0.25
    return 0.0


class Tracker:
    def __init__(self) -> None:
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.zero = ""
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.prev: Reading | None = None
        self.last = Result()

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (float(np.linalg.norm(r.origin - np.array(L.SHEARS_ORIGIN))) < 0.005
                and r.open_deg >= L.THETA_OPEN_DEG - 1.5
                and r.catch_m <= 0.001)

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        if not Tracker.at_start(now):
            return False
        return (float(np.linalg.norm(now.origin - prev.origin)) >= L.RESET_JUMP_M
                or prev.open_deg < L.THETA_OPEN_DEG - 10.0
                or prev.catch_m > 0.005)

    def _zero(self, why: str) -> None:
        if not self.zero:
            self.zero = why

    def update(self, states, arms, interlock_locked: bool) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.win.clear()
            self.prev = None
            self.last = Result(0.0, False, {"unreadable": SHEARS, "zero": self.zero})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r

        closed_now = is_closed(r)
        locked_now = is_locked(r, interlock_locked)
        tote_now = in_tote(r)
        dropped_now = off_bench(r)

        if tote_breach(r, locked_now):
            self._zero("unlocked shears entered below tote rim")
        if blade_contact(r, arms):
            self._zero("gripper contacted exposed blade while open")

        gap = shears_gap(states, r, arms)
        clear_now = gap is not None and gap >= L.CLEAR_M - 1e-9
        self.win.append((clear_now, r.origin.copy(), r.rot.copy()))
        full = len(self.win) == L.WIN_STEPS
        clear_1s = full and all(w[0] for w in self.win)
        moved = None
        turned = None
        if full:
            o0, r0 = self.win[0][1], self.win[0][2]
            moved = max(float(np.linalg.norm(w[1] - o0)) for w in self.win)
            turned = max(_rot_diff_deg(r0, w[2]) for w in self.win)
        still_1s = (moved is not None and turned is not None
                    and moved <= L.STILL_POS_M + 1e-9 and turned <= L.STILL_ANG_DEG + 1e-9)
        clear_still = clear_1s and still_1s

        sc = tier(closed_now, locked_now, tote_now, clear_still, dropped_now, self.zero)
        details = {
            "open_deg": round(r.open_deg, 2),
            "catch_mm": round(r.catch_m * 1000.0, 2),
            "closed": closed_now,
            "locked": locked_now,
            "in_tote": tote_now,
            "tilt_deg": round(r.tilt_deg, 2),
            "off_bench": dropped_now,
            "clear_1s": clear_1s,
            "still_1s": still_1s,
            "gap_mm": None if gap is None else round(gap * 1000.0, 1),
            "moved_mm": None if moved is None else round(moved * 1000.0, 2),
            "turned_deg": None if turned is None else round(turned, 2),
            "zero": self.zero,
        }
        self.last = Result(sc, sc >= 1.0, details)
        return self.last
