"""The scorer, as plain functions of the three pieces of equipment's states. NumPy only; no simulator needed.

The rule is about the finished set-up, read from the state at hand: each object's clock angle and distance seen from
above around the painted mouth mark (12 o'clock behind the head, -y; 3 o'clock the supine patient's left, -x).
'In a window' means standing (upright within 10 degrees, its lowest point within 5 mm of the floor), 80 to 200 mm
from the mouth, at a clock angle inside the window. Ordered decision list, first match wins:

1. Any object not released (a gripper's pad-tip point within finger reach and less than 10 mm above its top at any
   step of the last 0.5 s) or not settled (its centre's path over the last 0.5 s 2 mm or more), or any object's
   centre within 200 mm of the mouth in the left-handed transfer zone (5 to 8 o'clock): 0.
2. Dentist's stool 12:30 to 3:30, assistant's stool 8:15 to 9:45, cart 8:15 to 11:45, every object settled in each
   of the last two 0.5 s windows (still for 1 s): 1.0.
3. Both stools in their left-handed windows: 0.75.
4. Dentist's stool in its left-handed window, the assistant's not: 0.5.
5. The right-handed layout, dentist's stool 8:30 to 11:30 and assistant's stool 2:15 to 3:45: 0.25.
6. Otherwise: 0.

A small tracker is fed once per control step by ``task.success()``: it keeps each object's recent centres (for
'settled' and 'still') and whether a gripper held it (for 'released'), what the trace needs (the first step each
score was reached, broken observations, steps with an arm unreadable), and recognises a new episode when the
simulator puts every object back on its starting mark in one step.

    Tracker.update(objects, arms) -> Result(score, success, details)

``objects`` (3, in the order of ``layout.NAMES``) are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. A state with a non-finite number or a
zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was. ``arms`` is
``rigstate.arms()`` or None (offline): an arm that cannot be read is skipped by the release rule (the other is still
checked) and the step is counted in ``arms_unavailable``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9
SIDES = ("left", "right")
TIERS = (0.25, 0.5, 0.75, 1.0)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN, inf or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's vertical axis and the vertical (degrees, 0..90; either end up)."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def _half_height(name: str, quat_xyzw) -> float:
    """How far the object reaches below (and above) its centre, from its pose."""
    r = rotation(quat_xyzw)
    if name == L.CART:
        return float(sum(abs(r[2, i]) * L.CART_SIZE[i] / 2 for i in range(3)))
    az = abs(float(r[2, 2]))
    return L.HEIGHT[name] / 2 * az + L.STOOL_RADIUS * math.sqrt(max(0.0, 1.0 - az * az))


def lowest(name: str, s) -> float:
    return float(s.pos[2]) - _half_height(name, s.quat_xyzw)


def highest(name: str, s) -> float:
    return float(s.pos[2]) + _half_height(name, s.quat_xyzw)


def bottom_gap(name: str, s) -> float:
    """Height of the object's lowest point above the floor (m; negative = below it)."""
    return lowest(name, s) - L.TOP


def standing(name: str, s) -> bool:
    """Upright (its vertical axis within 10 degrees of vertical) with its lowest point within 5 mm of the floor."""
    return bool(tilt_deg(s.quat_xyzw) < L.UPRIGHT_DEG and abs(bottom_gap(name, s)) <= L.FLOOR_TOL + _EPS)


def clock(s) -> tuple[float, float]:
    """(clock angle in degrees 0..360, distance from the mouth mark in m) of the object's centre, seen from above."""
    return L.clock_of(float(s.pos[0]), float(s.pos[1]))


def in_sector(phi: float, r: float, window) -> bool:
    lo, hi = window
    return bool(L.R_IN - _EPS <= r <= L.R_OUT + _EPS and lo - _EPS <= phi <= hi + _EPS)


def in_window(name: str, s, window) -> bool:
    """Standing, 80 to 200 mm from the mouth mark, its clock angle inside ``window``."""
    phi, r = clock(s)
    return bool(standing(name, s) and in_sector(phi, r, window))


def in_transfer(s) -> bool:
    """Its centre within 200 mm of the mouth mark at 5 to 8 o'clock (whatever its height or tilt)."""
    phi, r = clock(s)
    lo, hi = L.TRANSFER
    return bool(r <= L.R_OUT + _EPS and lo - _EPS <= phi <= hi + _EPS)


# -- released: the arms (rigstate) ----------------------------------------------------------------------------------
def arm_tips(arms, side: str) -> np.ndarray | None:
    """One arm's pad-tip point, or None if it cannot be read (no snapshot, the arm missing, a non-finite value)."""
    try:
        tips = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None
    return tips if tips.size == 3 and np.all(np.isfinite(tips)) else None


def readable_sides(arms) -> tuple[str, ...]:
    return tuple(side for side in SIDES if arm_tips(arms, side) is not None)


def opening(arm) -> float:
    """rigstate's opening, 0 (closed) .. 1 (fully open); an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def reach(arm, name: str) -> float:
    return L.release_reach(opening(arm), name)


def held(name: str, s, arms) -> bool:
    """Some readable gripper's pad-tip point is within finger reach of the object's centre horizontally and lower
    than 10 mm above its top: a finger may be holding, pressing or propping it, whatever the gripper's yaw."""
    for side in readable_sides(arms):
        p = arm_tips(arms, side)
        xy = float(np.hypot(p[0] - float(s.pos[0]), p[1] - float(s.pos[1])))
        if xy < reach(arms[side], name) and p[2] - highest(name, s) < L.RELEASE_ABOVE:
            return True
    return False


def decide(state: dict) -> tuple[float, int]:
    """The decision list on precomputed facts: (score, the line that matched, 1..6)."""
    if not (state["released"] and state["settled"]) or state["transfer"]:
        return 0.0, 1
    lh_d, lh_a, lh_c = state["lh_dentist"], state["lh_assistant"], state["lh_cart"]
    if lh_d and lh_a and lh_c and state["still_1s"]:
        return 1.0, 2
    if lh_d and lh_a:
        return 0.75, 3
    if lh_d:
        return 0.5, 4
    if state["rh_dentist"] and state["rh_assistant"]:
        return 0.25, 5
    return 0.0, 6


def evaluate(objects, released, settled, still_1s) -> tuple[float, dict]:
    """The score of the set-up as it stands, given per-object 'released' and 'settled' (from the tracker) and whether
    every object has been still for 1 s; and what was found (JSON-clean)."""
    d, a, c = objects
    facts = {
        "released": all(released), "settled": all(settled), "still_1s": bool(still_1s),
        "transfer": any(in_transfer(s) for s in objects),
        "lh_dentist": in_window(L.D_STOOL, d, L.LH_DENTIST),
        "lh_assistant": in_window(L.A_STOOL, a, L.LH_ASSISTANT),
        "lh_cart": in_window(L.CART, c, L.LH_CART),
        "rh_dentist": in_window(L.D_STOOL, d, L.RH_DENTIST),
        "rh_assistant": in_window(L.A_STOOL, a, L.RH_ASSISTANT),
    }
    score, line = decide(facts)
    names = L.NAMES
    facts.update(line=line,
                 not_released=[n for n, ok in zip(names, released, strict=True) if not ok],
                 not_settled=[n for n, ok in zip(names, settled, strict=True) if not ok],
                 in_transfer_zone=[n for n, s in zip(names, objects, strict=True) if in_transfer(s)])
    return score, facts


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the release and stillness rules and the trace need about this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.arms_unavailable_by_side = {side: 0 for side in SIDES}
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every object at the previous valid control step
        n = self.window * L.STILL_WINDOWS_FOR_FULL + 1
        self.history: deque = deque(maxlen=n)  # every object's centre at the last valid steps
        self.held_history: deque = deque(maxlen=self.window + 1)  # every object's 'held' at the last valid steps

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every object still and upright on the floor, on its own mark."""
        for s, name in zip(objects, L.NAMES, strict=True):
            x0, y0, _ = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            resting = np.linalg.norm(s.lin_vel) < 0.01 and np.linalg.norm(s.ang_vel) < 0.5
            if not (on_mark and abs(bottom_gap(name, s)) < L.FLOOR_TOL and tilt_deg(s.quat_xyzw) < 1.0
                    and resting):
                return False
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def paths(self, windows: int = 1) -> list[list[float]] | None:
        """For each of the last ``windows`` 0.5 s windows (newest first), each object's centre path (m); None before
        that much valid history has been seen."""
        need = self.window * windows + 1
        if len(self.history) < need:
            return None
        h = list(self.history)[-need:]
        out = []
        for w in range(windows):
            seg = h[len(h) - 1 - (w + 1) * self.window: len(h) - w * self.window]
            out.append([float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(seg, seg[1:])))
                        for i in range(len(seg[0]))])
        return out

    def update(self, objects, arms=None) -> Result:
        objs = list(objects)
        self.steps += 1
        if len(objs) != len(L.NAMES) or not all(valid(s) for s in objs):  # a blow-up: 0, keep the tracker
            self.bad_steps += 1
            self.last = self.result(objs, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in objs]
        if self.prev is not None and self._teleported(now) and self.at_start(objs):
            self.reset()  # the simulator put every object back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        self.held_history.append([held(n, s, arms) for n, s in zip(L.NAMES, objs, strict=True)])
        readable = readable_sides(arms)
        for side in SIDES:
            if side not in readable:
                self.arms_unavailable_by_side[side] += 1
        if len(readable) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(objs, arms)
        for tier in TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, objects, arms=None) -> Result:
        """The score of the state at hand, using the tracker's history (``update`` has already added this state)."""
        objs = list(objects)
        readable = readable_sides(arms)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_readable": list(readable), "arms_unavailable": self.arms_unavailable,
                 "arms_unavailable_by_side": dict(self.arms_unavailable_by_side)}
        bad = [L.NAMES[i] for i, s in enumerate(objs[:len(L.NAMES)]) if not valid(s)]
        if bad or len(objs) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_objects": bad, **trial})
        n = len(L.NAMES)
        full = len(self.held_history) >= self.window + 1
        released = [full and not any(row[i] for row in self.held_history) for i in range(n)]
        p1 = self.paths(1)
        settled = [p1 is not None and p1[0][i] < L.STILL_PATH for i in range(n)]
        p2 = self.paths(L.STILL_WINDOWS_FOR_FULL)
        still_1s = p2 is not None and all(p < L.STILL_PATH for win in p2 for p in win)
        score, details = evaluate(objs, released, settled, still_1s)
        return Result(score, bool(score == 1.0), {"valid_state": True, "score_line": details.pop("line"),
                                                  **details, **trial})
