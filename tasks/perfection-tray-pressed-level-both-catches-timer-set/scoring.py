"""The scorer, as plain functions of the game unit's and the catches' states and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the unit (joints "tray", "rock", "dial", "switch"; layout.py gives their meaning) and of
the fixed catches (joints "pawl_l", "pawl_r"), and the barbs' and teeth's boxes from the link poses. The arms are read
only for the let-go check.

    CAUGHT (a side)  the pawl's tooth reaches 2 mm or more in over that side's barb, its underside not below the barb's
                     top (by more than 0.5 mm) and at most 6 mm above it: the tray cannot rise past the tooth
    LATCHED          both sides caught
    FORCED           a barb more than 2 mm into a tooth in both y and z at any step (the solver let it through: in a
                     real trial a broken catch)
    SET              the dial shows 50 s or more
    RUNNING          over the last 1 s the switch was at START on every step and the dial ran down 2 deg or more
    RELEASED         on every step of the last 1 s both finger zones are 50 mm or more from the unit and the catches

    0      FORCED ever, or none of the below
    1.0    LATCHED, SET, RUNNING and RELEASED (the trial ends after this has held 1 s more)
    0.7    LATCHED and SET at the end
    0.5    LATCHED at the end
    0.25   one side was CAUGHT at some step
    0.1    the tray was pressed 10 mm or more down at its centre at some step
The untouched start (tray up, dial at 0, switch at START) scores exactly 0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import parts as P
import rigstate

UNIT, CATCHES = "game", "catches"
PARTS = (UNIT, CATCHES)
TRAY, ROCK, DIAL, SWITCH = "tray", "rock", "dial", "switch"
PAWL = {1: "pawl_l", -1: "pawl_r"}
SIDES = (1, -1)


def _corners(center, size) -> np.ndarray:
    c, h = np.asarray(center, dtype=float), np.asarray(size, dtype=float) / 2
    return np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


_BARB = {s: _corners(*P.barb_box(s)) for s in SIDES}
_TOOTH = {s: _corners(*P.tooth_box(s)) for s in SIDES}


@dataclass(frozen=True)
class Reading:
    """One step (m, rad); boxes in the unit's base frame, as (lo, hi)."""
    travel: float
    rock: float
    dial: float
    switch: float
    pawls: dict
    barbs: dict
    teeth: dict


def _box(state, link, corners, unit) -> tuple:
    world = state.link_to_world(link, corners)
    local = unit.to_local(world)
    if not np.all(np.isfinite(local)):
        raise ValueError("non-finite box")
    return local.min(axis=0), local.max(axis=0)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        u, c = states.get(UNIT), states.get(CATCHES)
        if u is None or c is None:
            return None
        j = {k: float(u.joints[k]) for k in (TRAY, ROCK, DIAL, SWITCH)}
        pawls = {s: float(c.joints[PAWL[s]]) for s in SIDES}
        if not all(math.isfinite(v) for v in list(j.values()) + list(pawls.values())):
            return None
        barbs = {s: _box(u, ROCK, _BARB[s], u) for s in SIDES}
        teeth = {s: _box(c, PAWL[s], _TOOTH[s], u) for s in SIDES}
    except (AttributeError, TypeError, ValueError, KeyError, RuntimeError):
        return None
    return Reading(j[TRAY], j[ROCK], j[DIAL], j[SWITCH], pawls, barbs, teeth)


def _y_overlap(a, b) -> float:
    return float(min(a[1][1], b[1][1]) - max(a[0][1], b[0][1]))


def caught(r: Reading, side: int) -> bool:
    barb, tooth = r.barbs[side], r.teeth[side]
    gap = float(tooth[0][2] - barb[1][2])  # the tooth's underside above the barb's top
    return bool(_y_overlap(barb, tooth) >= L.LATCH_OVERLAP - L.EPS and -0.0005 <= gap <= L.LATCH_DROP)


def forced(r: Reading) -> bool:
    for s in SIDES:
        barb, tooth = r.barbs[s], r.teeth[s]
        z = float(min(barb[1][2], tooth[1][2]) - max(barb[0][2], tooth[0][2]))
        x = float(min(barb[1][0], tooth[1][0]) - max(barb[0][0], tooth[0][0]))
        if x > 0.0 and _y_overlap(barb, tooth) > L.FORCE_OVERLAP and z > L.FORCE_OVERLAP:
            return True
    return False


def tier(forced_ever: bool, latched: bool, dial_set: bool, running: bool, released: bool, one_caught_ever: bool,
         pressed_ever: bool) -> float:
    if forced_ever:
        return 0.0
    if latched and dial_set and running and released:
        return 1.0
    if latched and dial_set:
        return 0.7
    if latched:
        return 0.5
    if one_caught_ever:
        return 0.25
    if pressed_ever:
        return 0.1
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The memories (forced, one side caught, pressed), the last 1 s of switch and dial readings and of the let-go
    check, and a reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.RUN_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.forced_at = None
        self.caught_at = None
        self.pressed_at = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        return abs(r.travel) <= 0.002 and abs(r.dial) <= 0.02 and abs(r.switch) <= 0.002

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Put back to the start with a jump no trial makes in one step (a reset): the dial back to 0 by 20 deg or
        more, or the tray up by 15 mm or more with the switch also jumping to START."""
        if not Tracker.at_start(now):
            return False
        return (prev.dial - now.dial >= math.radians(20.0)
                or (prev.travel - now.travel >= L.RESET_JUMP and prev.switch - now.switch >= 0.006))

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        if self.forced_at is None and forced(r):
            self.forced_at = self.steps
        sides = {s: caught(r, s) for s in SIDES}
        if self.caught_at is None and any(sides.values()):
            self.caught_at = self.steps
        if self.pressed_at is None and r.travel >= L.PRESSED - L.EPS:
            self.pressed_at = self.steps
        self.hist.append(r)
        self.prev = r
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
        self.last = self.judge(r, sides, gap)
        return self.last

    def running(self) -> bool:
        if len(self.hist) < self.window + 1:
            return False
        h = list(self.hist)
        at_start = all(x.switch <= L.SWITCH_START_TOL for x in h)
        return bool(at_start and h[0].dial - h[-1].dial >= L.RUN_MIN - L.EPS and h[-1].dial > 0.0)

    def judge(self, r: Reading, sides: dict, gap) -> Result:
        released = len(self.released) == self.window and all(self.released)
        latched = all(sides.values())
        secs = L.dial_seconds(r.dial)
        dial_set = secs >= L.DIAL_SET_S - 1e-6
        run = self.running()
        level = tier(self.forced_at is not None, latched, dial_set, run, released, self.caught_at is not None,
                     self.pressed_at is not None)
        details = {
            "valid_state": True, "score": level,
            "forced_at": self.forced_at, "caught_at": self.caught_at, "pressed_at": self.pressed_at,
            "travel_mm": _mm(r.travel), "rock_deg": round(math.degrees(r.rock), 2),
            "side_travel_mm": {"L": _mm(L.side_travel(r.travel, r.rock, 1)),
                               "R": _mm(L.side_travel(r.travel, r.rock, -1))},
            "caught": {"L": sides[1], "R": sides[-1]}, "latched": latched,
            "pawl_deg": {"L": round(math.degrees(r.pawls[1]), 2), "R": round(math.degrees(r.pawls[-1]), 2)},
            "dial_s": round(secs, 2), "switch_mm": _mm(r.switch), "running_1s": run,
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "hands_gap_mm": _mm(gap),
        }
        return Result(level, bool(level == 1.0), details)
