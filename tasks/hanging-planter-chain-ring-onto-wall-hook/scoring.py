"""The scorer, as plain functions of the planter's state (pot, bail, ring) and the arms'. NumPy only.

``Tracker.update(state, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyState of the planter: the pot's pose (the base body; origin at the centre of its bottom face) and the
ring link's pose (frame at the bail's apex, the ring's centre 22.5 mm above it). The arms are read only for the
let-go check. The wall panel and the bracket are fixed and known (layout.py).

    F1 FELL             latched: the pot's lowest point more than 20 mm below the table top (off the table)
    ON BAR              the bracket's bar passes through the ring: the ring's hole axis within 60 degrees of the
                        bar's, the bar's axis crossing the ring's plane within 20 mm of its centre (the 8 mm bar
                        in the 40 mm hole; hung, its axis is 16 mm above the centre), that crossing behind the lip (x 287 to 419 mm), and the ring's centre
                        at least 8 mm below the bar's axis (hanging from it; hung it is 16 mm below)
    CLEAR               the pot's lowest point at least 20 mm above the table top and its rim at least 3 mm short of
                        the wall panel: nothing but the ring touches it
    PLUMB               the pot's axis within 10 degrees of vertical
    RELEASED            on every step of the last 1 s both finger zones are 30 mm or more from the planter
    STILL               over the last 1 s the pot's origin and the ring's centre each moved under 5 mm and the pot
                        and the ring each turned under 5 degrees (measured from where each was at the second's start)

    0      F1 ever fired, or not released, or the ring not on the bar
    1.0    on the bar, clear, plumb, released and still
    0.5    on the bar, clear and released, but still swinging or leaning (not plumb or not still)
    0.25   on the bar and released, but the pot touches the table or the wall (not clear)
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PLANTER = "planter"
RING = "ring"
_X = np.array([1.0, 0.0, 0.0])
_RING_C = np.array([0.0, 0.0, L.RING_DZ])  # the ring's centre in its link frame (origin at the bail's apex)


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
    pot: np.ndarray  # the pot's origin (centre of its bottom face)
    pot_r: np.ndarray  # its rotation
    ring: np.ndarray  # the ring's centre
    ring_r: np.ndarray  # the ring's rotation


def read(states) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        s = states.get(PLANTER)
        if s is None:
            return None
        o = np.asarray(s.origin, dtype=float).ravel()
        r = rotation(s.quat_xyzw)
        rr, ro = s.link_pose(RING)
        rr, ro = np.asarray(rr, dtype=float), np.asarray(ro, dtype=float).ravel()
        if o.size != 3 or ro.size != 3 or rr.shape != (3, 3):
            return None
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and np.all(np.isfinite(rr))
                and np.all(np.isfinite(ro))):
            return None
        top = np.array([0.0, 0.0, L.TOP])
        return Reading(o - top, r, ro - top + rr @ _RING_C, rr)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


# -- the readings ------------------------------------------------------------------------------------------------------
def pot_low(r: Reading) -> float:
    """The pot's lowest point above the table top (m): its bottom face's lowest rim point, or the rim lip's."""
    rot = r.pot_r
    bottom = float(r.pot[2]) - L.POT_R * math.hypot(float(rot[2, 0]), float(rot[2, 1]))
    lip = float(r.pot[2] + rot[2, 2] * L.POT_H) - L.RIM_R * math.hypot(float(rot[2, 0]), float(rot[2, 1]))
    return min(bottom, lip)


def pot_max_x(r: Reading) -> float:
    """The pot's furthest point toward the wall (m): the rim lip's and the body's extremes."""
    rot = r.pot_r
    tip = float(r.pot[0] + rot[0, 2] * L.POT_H) + L.RIM_R * math.hypot(float(rot[0, 0]), float(rot[0, 1]))
    body = float(r.pot[0]) + L.POT_R * math.hypot(float(rot[0, 0]), float(rot[0, 1]))
    return max(tip, body)


def bar_crossing(r: Reading) -> dict:
    """Where the bar's axis meets the ring's plane: ``x`` along the bar, ``off`` from the ring's centre (m), and
    ``n_x`` the ring's hole axis along the bar (0 when the ring is edge-on to the bar)."""
    n = r.ring_r @ _X
    c = r.ring
    nx = float(n[0])
    if abs(nx) < 1e-9:
        return {"x": None, "off": None, "n_x": 0.0}
    x = float(c[0] - (n[1] * (L.HOOK_Y - c[1]) + n[2] * (L.HOOK_Z - c[2])) / nx)
    p = np.array([x, L.HOOK_Y, L.HOOK_Z])
    return {"x": x, "off": float(np.linalg.norm(p - c)), "n_x": abs(nx)}


def on_bar(r: Reading) -> bool:
    b = bar_crossing(r)
    if b["x"] is None:
        return False
    return bool(b["n_x"] >= L.N_MIN and b["off"] <= L.THROUGH_R + L.EPS
                and L.BAR_BEHIND_LIP - L.EPS <= b["x"] <= L.BAR_X1 - L.BAR_END_MARGIN + L.EPS
                and float(r.ring[2]) <= L.HOOK_Z - L.HUNG_BELOW + L.EPS)


def clear(r: Reading) -> bool:
    return bool(pot_low(r) >= L.CLEAR - L.EPS and pot_max_x(r) <= L.WALL_X0 - L.WALL_GAP + L.EPS)


def plumb(r: Reading) -> bool:
    return bool(tilt_deg(r.pot_r) <= L.PLUMB_DEG + L.EPS)


def fell(r: Reading) -> bool:
    return pot_low(r) < -L.FALL_DROP


def tier(latched: bool, released: bool, onbar: bool, is_clear: bool, is_plumb: bool, still: bool) -> float:
    if latched or not released or not onbar:
        return 0.0
    if not is_clear:
        return 0.25
    return 1.0 if (is_plumb and still) else 0.5


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
        return float(np.linalg.norm(r.pot - np.array(L.POT_START))) <= L.NEAR_START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The pot put back on its starting mark with a jump no trial makes in one step (a reset)."""
        return cls.at_start(now) and float(np.linalg.norm(now.pot - prev.pot)) >= L.RESET_JUMP

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
        ok, gap = False, None
        try:
            dist = states[PLANTER].distance
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance or unreadable arms: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) of the pot's origin and of the ring's centre and the largest turn
        (deg) of the pot and of the ring from where each was at the window's start; None before 1 s of readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        pot_move = ring_move = pot_turn = ring_turn = 0.0
        for x in h[1:]:
            pot_move = max(pot_move, float(np.linalg.norm(x.pot - first.pot)))
            ring_move = max(ring_move, float(np.linalg.norm(x.ring - first.ring)))
            pot_turn = max(pot_turn, turn_deg(first.pot_r, x.pot_r))
            ring_turn = max(ring_turn, turn_deg(first.ring_r, x.ring_r))
        return {"pot_move": pot_move, "ring_move": ring_move, "pot_turn": pot_turn, "ring_turn": ring_turn}

    def judge(self, r: Reading, gap) -> Result:
        onbar, is_clear, is_plumb = on_bar(r), clear(r), plumb(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and max(m["pot_move"], m["ring_move"]) <= L.STILL_MOVE + L.EPS
                     and max(m["pot_turn"], m["ring_turn"]) <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released, onbar, is_clear, is_plumb, still)
        b = bar_crossing(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "on_bar": onbar, "clear": is_clear, "plumb": is_plumb,
            "bar_x_mm": _mm(b["x"]), "bar_off_mm": _mm(b["off"]), "ring_n_x": round(b["n_x"], 3),
            "ring_below_bar_mm": _mm(L.HOOK_Z - float(r.ring[2])),
            "pot_low_mm": _mm(pot_low(r)), "pot_max_x_mm": _mm(pot_max_x(r)), "pot_tilt_deg": round(tilt_deg(r.pot_r), 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "pot_mm": _mmv(r.pot), "ring_mm": _mmv(r.ring),
            "motion": None if m is None else {"pot_move_mm": _mm(m["pot_move"]), "ring_move_mm": _mm(m["ring_move"]),
                                              "pot_turn_deg": round(m["pot_turn"], 3),
                                              "ring_turn_deg": round(m["ring_turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
