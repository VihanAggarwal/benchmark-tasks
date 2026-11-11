"""The scorer, as plain functions of the three hinge leaves' states (and the arms, for 'released'). NumPy only.

The rule is the hollow metal door trade's hinge layout, '5, 10 and equal' (NAAMM/HMMA), read from the state at
hand: the top hinge's top 5 in (25 mm at 1 in = 5 mm) below the HEAD line, the bottom hinge's bottom 10 in (50 mm)
above the FLOOR line, the third hinge centred between the two as hung. See ``layout`` for the numbers.

A leaf is 'on the edge' when it is released (no gripper has held it during the last 0.5 s), settled (its centre
moved at most 2 mm and its long axis turned at most 1 degree over the last 0.5 s), flat (5 degrees), lengthwise
(its long axis within 5 degrees of the door's length, either way round), its lowest point within 3 mm of the edge
top, its centre within 8 mm of the edge's centreline and wholly within the door's length.

Locations: with three leaves on the edge, the highest (largest y) is the top leaf, the lowest the bottom leaf and
the third the middle leaf; the top is right within 5 mm of y +0.16375, the bottom within 5 mm of y -0.14250, the
middle within 5 mm of the midpoint of the top and bottom leaves as placed. With fewer than three on the edge, each
leaf is judged against the nominal windows (the middle's at +0.010625); the windows are disjoint, so each leaf
counts for at most one location. k = the number of locations right (0 to 3).

Ordered tiers (first match wins): three on the edge, k = 3 and all three still for 1 s: 1.0; three on the edge with
the top and bottom both right: 0.75; k = 2: 0.5; k = 1: 0.25; otherwise 0.

    Tracker.update(leaves, arms) -> Result(score, success, details)

``leaves`` is a dict {name: state} of the three leaves, each anything with ``pos`` (centre), ``quat_xyzw``,
``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()`` ({"left": Arm, "right": Arm} with ``tips`` and ``opening``) or None; an arm that cannot be read
is skipped by the held rule (the other is still checked) and counted in ``arms_unavailable``. A state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it
was.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

HALF = np.array(L.LEAF_SIZE, dtype=float) / 2
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9
SIDES = ("left", "right")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the
    world. (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the right sizes, and the quaternion not degenerate."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool([p.size for p in parts] == [3, 4, 3, 3, 3, 3] and np.linalg.norm(parts[1]) > 0.5)


# -- one leaf ------------------------------------------------------------------------------------------------
def tilt_deg(s) -> float:
    """Angle between the vertical and the leaf's thickness axis (degrees, 0..90; either face up)."""
    zz = abs(float(rotation(s.quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def heading_deg(s) -> float:
    """The direction of the leaf's long (22.5 mm) axis seen from above, degrees from +y, folded to -90..90 (the
    leaf is the same either way round). A leaf standing on an end has no horizontal long axis: 90."""
    r = rotation(s.quat_xyzw)
    hx, hy = float(r[0, 1]), float(r[1, 1])
    if math.hypot(hx, hy) < 0.3:
        return 90.0
    a = math.degrees(math.atan2(-hx, hy))  # 0 along +y, + anticlockwise
    a = (a + 90.0) % 180.0 - 90.0
    return float(a)


def along_err_deg(s) -> float:
    """How far the leaf's long axis is turned from the door's length (0..90 degrees)."""
    return abs(heading_deg(s))


def corners(s) -> np.ndarray:
    return np.asarray(s.pos, dtype=float) + (_SIGNS * HALF) @ rotation(s.quat_xyzw).T


def lowest(s) -> float:
    return float(corners(s)[:, 2].min())


def highest(s) -> float:
    return float(corners(s)[:, 2].max())


def bottom_above_edge(s) -> float:
    """The leaf's lowest point above the hinge edge's top (m; negative = below it)."""
    return lowest(s) - L.EDGE_TOP


def off_centreline(s) -> float:
    """The leaf centre's distance from the edge's centreline, across the edge (m, signed: + away from the arms)."""
    return float(s.pos[0]) - L.DOOR_X


def resting_on_edge(s) -> bool:
    """Geometry only: flat, lengthwise, its lowest point within 3 mm of the edge top, centred across the edge
    within 8 mm, wholly within the door's length."""
    return bool(tilt_deg(s) <= L.FLAT_DEG + _EPS
                and along_err_deg(s) <= L.ALONG_DEG + _EPS
                and abs(bottom_above_edge(s)) <= L.SEAT_TOL + _EPS
                and abs(off_centreline(s)) <= L.ACROSS_TOL + _EPS
                and abs(float(s.pos[1])) <= L.ON_DOOR + _EPS)


# -- held: the arms (rigstate) ---------------------------------------------------------------------------------
def arm_tips(arms, side: str) -> np.ndarray | None:
    try:
        tips = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None
    return tips if tips.size == 3 and np.all(np.isfinite(tips)) else None


def readable_sides(arms) -> tuple[str, ...]:
    return tuple(side for side in SIDES if arm_tips(arms, side) is not None)


def opening(arm) -> float:
    """rigstate's opening, 0..1; an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def reach(arm) -> float:
    return L.release_reach(opening(arm))


def held(s, arms) -> bool:
    """Some readable gripper's pad-tip point lower than 10 mm above the leaf's top, with the leaf's centre within
    that gripper's finger reach horizontally (a finger may be touching, pressing or propping it)."""
    c = np.asarray(s.pos, dtype=float)
    top = highest(s)
    for side in readable_sides(arms):
        p = arm_tips(arms, side)
        if float(np.hypot(*(p[:2] - c[:2]))) < reach(arms[side]) and float(p[2]) - top < L.RELEASE_ABOVE:
            return True
    return False


# -- the layout ------------------------------------------------------------------------------------------------
def in_window(y: float, centre: float) -> bool:
    return bool(abs(float(y) - centre) <= L.WINDOW + _EPS)


def judge(on: dict) -> dict:
    """The locations, from the centres' y of the leaves on the edge ({name: y})."""
    d: dict = {"on_edge": sorted(on), "n_on_edge": len(on)}
    if len(on) == 3:
        order = sorted(on, key=lambda n: on[n])
        bot, mid, top = order
        y_m = (on[top] + on[bot]) / 2
        top_ok, bot_ok, mid_ok = in_window(on[top], L.TOP_Y), in_window(on[bot], L.BOTTOM_Y), in_window(on[mid], y_m)
        d.update({"top_leaf": top, "middle_leaf": mid, "bottom_leaf": bot, "y_m_mm": round(y_m * 1000, 2),
                  "top_err_mm": round((on[top] - L.TOP_Y) * 1000, 2),
                  "bottom_err_mm": round((on[bot] - L.BOTTOM_Y) * 1000, 2),
                  "middle_err_mm": round((on[mid] - y_m) * 1000, 2)})
    else:
        ys = list(on.values())
        top_ok = any(in_window(y, L.TOP_Y) for y in ys)
        bot_ok = any(in_window(y, L.BOTTOM_Y) for y in ys)
        mid_ok = any(in_window(y, L.MID_NOMINAL) for y in ys)
        d["y_m_mm"] = round(L.MID_NOMINAL * 1000, 2)
    d.update({"top_ok": bool(top_ok), "middle_ok": bool(mid_ok), "bottom_ok": bool(bot_ok),
              "k": int(top_ok) + int(mid_ok) + int(bot_ok)})
    return d


def tier(d: dict, all_still_1s: bool) -> tuple[float, str]:
    """The ordered decision list (first match wins)."""
    three = d["n_on_edge"] == 3
    if three and d["k"] == 3 and all_still_1s:
        return L.S_FULL, "all_three_right_and_still_1s"
    if three and d["top_ok"] and d["bottom_ok"]:
        return L.S_ENDS, "both_ends_right" if d["k"] < 3 else "all_three_right_not_still_1s"
    if d["k"] == 2:
        return L.S_TWO, "two_locations"
    if d["k"] == 1:
        return L.S_ONE, "one_location"
    return 0.0, "no_location"


def evaluate(leaves: dict, counted: dict, still_1s: dict) -> tuple[float, dict]:
    """The score of a state (every leaf valid), given which leaves are counted (released and settled) and which
    have been still for 1 s."""
    on = {n: float(s.pos[1]) for n, s in leaves.items() if counted.get(n, False) and resting_on_edge(s)}
    d = judge(on)
    all_still = bool(len(on) == 3 and all(still_1s.get(n, False) for n in on))
    sc, why = tier(d, all_still)
    d.update({"all_still_1s": all_still, "tier": why})
    return float(sc), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each leaf's centre and heading, and whether it was held, over the
    last 1 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.history: dict[str, deque] = {}  # leaf -> (centre, heading) over the last 2 windows + 1 steps
        self.held_hist: dict[str, deque] = {}  # leaf -> held flags over the last window + 1 steps
        self.prev = None
        self.first_at: dict[float, int] = {}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(leaves: dict) -> bool:
        """The starting layout: every leaf flat on the tray on a starting mark (any leaf on any mark), slow."""
        free = dict(L.LEAF_START)
        for s in leaves.values():
            hit = None
            for mark, (sx, sy, _) in free.items():
                if math.hypot(float(s.pos[0]) - sx, float(s.pos[1]) - sy) < L.START_TOL:
                    hit = mark
                    break
            if hit is None:
                return False
            free.pop(hit)
            slow = float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) < 0.05
            if not (abs(lowest(s) - L.TRAY_TOP) < 0.003 and tilt_deg(s) < 2.0 and slow):
                return False
        return True

    def _teleported(self, now: dict) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for n, (p0, v0) in self.prev.items() if n in now
        )

    @staticmethod
    def _spread(samples) -> tuple[float, float]:
        """Farthest any sample's centre is from the last one's (m), and the most any heading differs from it
        (degrees, mod 180)."""
        c_now, h_now = samples[-1]
        dist = max(float(np.linalg.norm(c - c_now)) for c, _ in samples)
        turn = max(abs((h - h_now + 90.0) % 180.0 - 90.0) for _, h in samples)
        return dist, turn

    def settled_windows(self, name: str) -> tuple[bool, bool, float | None, float | None]:
        """(settled over the last 0.5 s, settled over the 0.5 s before that, centre spread mm, turn deg)."""
        h = self.history.get(name)
        w = self.window
        if h is None or len(h) < w + 1:
            return False, False, None, None
        samples = list(h)
        d1, t1 = self._spread(samples[-(w + 1):])
        now_ok = d1 <= L.STILL_TOL + _EPS and t1 <= L.STILL_YAW_DEG + _EPS
        prev_ok = False
        if len(samples) >= 2 * w + 1:
            d0, t0 = self._spread(samples[-(2 * w + 1):-w])
            prev_ok = d0 <= L.STILL_TOL + _EPS and t0 <= L.STILL_YAW_DEG + _EPS
        return bool(now_ok), bool(prev_ok), d1, t1

    def released(self, name: str) -> bool:
        h = self.held_hist.get(name)
        return bool(h is not None and len(h) >= self.window + 1 and not any(h))

    def update(self, leaves: dict, arms=None) -> Result:
        self.steps += 1
        if not all(valid(s) for s in leaves.values()):
            self.bad_steps += 1
            self.last = self.result(leaves, arms)
            return self.last
        now = {n: (np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for n, s in leaves.items()}
        if self.prev is not None and self._teleported(now) and self.at_start(leaves):
            bad, unavailable = self.bad_steps, self.arms_unavailable
            self.reset()
            self.steps, self.bad_steps, self.arms_unavailable = 1, bad, unavailable
        self.prev = now
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        for n, s in leaves.items():
            self.history.setdefault(n, deque(maxlen=2 * self.window + 1)).append(
                (np.array(s.pos, dtype=float), heading_deg(s)))
            self.held_hist.setdefault(n, deque(maxlen=self.window + 1)).append(held(s, arms))
        self.last = self.result(leaves, arms)
        if self.last.score > 0 and self.last.score not in self.first_at:
            self.first_at[self.last.score] = self.steps
        return self.last

    def result(self, leaves: dict, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"step": self.steps, "bad_steps": self.bad_steps, "arms_unavailable": self.arms_unavailable,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        bad = sorted(n for n, s in leaves.items() if not valid(s))
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_leaves": bad, "zero_reason": "invalid_state",
                                       **trial})
        counted, still_1s, per = {}, {}, {}
        for n, s in leaves.items():
            now_ok, prev_ok, spread, turn = self.settled_windows(n)
            rel = self.released(n)
            held_now = held(s, arms)
            counted[n] = bool(rel and now_ok and not held_now)
            still_1s[n] = bool(now_ok and prev_ok)
            per[n] = {"released": rel, "held_now": held_now, "settled": now_ok, "still_1s": still_1s[n],
                      "spread_mm": None if spread is None else round(spread * 1000, 2),
                      "turn_deg": None if turn is None else round(turn, 2),
                      "resting_on_edge": resting_on_edge(s)}
        score, details = evaluate(leaves, counted, still_1s)
        details["leaves"] = per
        details["arms_readable"] = list(readable_sides(arms))
        return Result(score, bool(score == L.S_FULL), {"valid_state": True, "zero_reason": None, **details,
                                                       **trial})
