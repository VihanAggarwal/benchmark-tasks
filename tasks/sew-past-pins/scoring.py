"""The scorer, as plain functions of the work's feed, the three pins' states and the arms' state. NumPy only; no
simulator needed.

Most of the rule is about the whole trial, not only its end: no pin may ever reach the needle or come loose without
a gripper holding it, each pin's pull is judged at the moment it leaves the plate, and the work may move only while a
gripper holds its handle. So the scorer keeps a little state (what has happened, and when) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(feed, pins, arms) -> Result(score, success, details)

``feed`` is how far the work has slid along its rail (m, 0 at the start, positive toward the needle; mechkit's joint
reading). ``pins`` maps each pin's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the handle, pin-holding and
let-go rules are taken as met for that step and the step is counted in ``arms_unavailable``. A non-finite feed or a pin state with a
non-finite number, a zero quaternion or a missing field (a simulator blow-up) scores 0 for that step and leaves the
tracker as it was.

Terms (layout.py has the numbers):
- d: a pin's centre's y minus the needle line's y (positive before the needle; the feed takes it down).
- In the plate: the pin's centre over the plate (as the plate is now) and its lowest point less than 3 mm above the
  plate's top. Standing in its pocket a pin's lowest point is 6 mm below the plate's top; lying on the plate, on it.
- Holds a pin: a gripper less open than 0.30 (fingers under 28 mm apart) with its finger-tip point within 4 mm of
  the pin's box. Closed fingers that push a pin from the side keep that point 6 mm or more from it.
- Pulled: a pin leaves the plate (lifted out of its pocket or slid off the plate's side) while a gripper holds it.
  A pin that leaves the plate while no gripper holds it has been knocked loose: that scores 0 for the trial. Only a
  pin's first pull counts.
- Pulled in time: pulled with d between 30 and 70 mm, and never closer than 30 mm to the needle line while in the
  plate at any time in the trial (feeding the work back to move a late pin away, or putting a pin back and letting
  it ride on, does not make it timely).
- Strike: any part of a pin (its bounding box) within 6 mm of the needle line or across it (an upright pin's centre
  within 15 mm), or crossing it between two steps, while the pin is under the needle head: its centre within 72 mm
  of the seam line in x (between the posts' outer faces) and its lowest point below the cross bar's top (90 mm above
  the table). A pin riding in the work into the needle, or carried past it under the bar, both count.
- Holds the handle: a gripper less open than 0.6 (fingers under 56 mm apart) with its finger-tip point over the
  handle's footprint grown by 5 mm and below the handle's top (where the handle is now): its fingers are round it.
- In the dish: the pin's centre over the dish's 140 x 70 mm floor, its lowest point below the 15 mm rim, still
  (under 1 cm/s and 0.5 rad/s).
- Off the table: a pin's lowest point 10 mm or more below the table's top (it has fallen off).
- Let go: a gripper open 0.9 or more, or its finger-tip point more than 20 mm from the handle's box and every pin's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

ARMS = ("left", "right")
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_HANDLE_HALF = np.array(L.HANDLE, dtype=float) / 2


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def valid_feed(feed) -> bool:
    try:
        return math.isfinite(float(feed))
    except (TypeError, ValueError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def tilt_deg(quat_xyzw) -> float:
    """Angle between a pin's axis and the vertical (degrees, 0..90; a pin standing on either end is 0)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = x * x + y * y + z * z + w * w
    zz = 1.0 - 2.0 * (x * x + y * y) / (n if n > 0 else 1.0)  # R[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to an axis-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


# -- the work ---------------------------------------------------------------------------------------------------------
def handle_box(feed: float) -> tuple[np.ndarray, np.ndarray]:
    """The handle's box (low, high) where the feed has taken it."""
    c = np.array([L.X_SEAM, L.HANDLE_Y - feed, L.TOP + L.HANDLE_Z])
    return c - _HANDLE_HALF, c + _HANDLE_HALF


def over_plate(p, feed: float) -> bool:
    """A point's (x, y) over the plate's top as the feed has placed it."""
    return bool(abs(float(p[0]) - L.X_SEAM) <= L.PLATE_W / 2
                and L.PLATE_Y0 - feed <= float(p[1]) <= L.PLATE_Y1 - feed)


def in_work(pin, feed: float) -> bool:
    """In the plate: over it with its lowest point less than 3 mm above the plate's top."""
    return bool(over_plate(pin.pos, feed) and float(pin.low[2]) - (L.TOP + L.PLATE_TOP) < L.OUT_CLEAR)


def pin_d(pin) -> float:
    """The pin's centre's distance before the needle line along the seam (m; negative past it)."""
    return float(pin.pos[1]) - L.Y_NEEDLE


def needle_gap(pin) -> float:
    """From the needle line to the nearest part of the pin's bounding box, along y (0 when it straddles the line)."""
    lo, hi = float(pin.low[1]), float(pin.high[1])
    if lo <= L.Y_NEEDLE <= hi:
        return 0.0
    return min(abs(lo - L.Y_NEEDLE), abs(hi - L.Y_NEEDLE))


def under_head(pin) -> bool:
    """Centre between the needle head's posts (within 72 mm of the seam line in x), lowest point below the bar's top."""
    return bool(abs(float(pin.pos[0]) - L.X_SEAM) <= L.HEAD_HALF and float(pin.low[2]) < L.TOP + L.BAR_Z1)


def in_dish(pin) -> bool:
    """Centre over the dish's floor, lowest point below its rim, still."""
    return bool(abs(float(pin.pos[0]) - L.DISH_X) <= L.DISH_HALF[0]
                and abs(float(pin.pos[1]) - L.DISH_Y) <= L.DISH_HALF[1]
                and float(pin.low[2]) < L.TOP + L.DISH_RIM and is_resting(pin))


def off_table(pin) -> bool:
    """Fallen off: its lowest point 10 mm or more below the table's top (a pin held in the air past the table's edge
    has not)."""
    return bool(float(pin.low[2]) <= L.TOP - L.OFF_TABLE_DROP)


# -- the arms ---------------------------------------------------------------------------------------------------------
def holds_handle(arm, feed: float) -> bool:
    """Less open than 0.6, finger-tip point over the handle's footprint grown by 5 mm and below its top: the fingers
    are round the handle, not hovering over it."""
    lo, hi = handle_box(feed)
    tips = np.asarray(arm.tips, dtype=float)
    over = bool(np.all(tips[:2] >= lo[:2] - L.HOLD_MARGIN) and np.all(tips[:2] <= hi[:2] + L.HOLD_MARGIN))
    return bool(float(arm.opening) < L.HOLD_OPENING and over and tips[2] <= hi[2])


def holds_pin(arm, pin) -> bool:
    """Less open than 0.30, finger-tip point within 4 mm of the pin's box: the pin is between the fingers."""
    lo, hi = np.asarray(pin.low, dtype=float), np.asarray(pin.high, dtype=float)
    return bool(float(arm.opening) < L.PIN_HOLD_OPENING and point_box_distance(arm.tips, lo, hi) <= L.PIN_HOLD_GAP)


def let_go(arm, feed: float, pins: dict) -> bool:
    """Open 0.9 or more, or the finger-tip point more than 20 mm from the handle's box and every pin's box."""
    if float(arm.opening) >= L.RELEASE_OPENING:
        return True
    lo, hi = handle_box(feed)
    boxes = [(lo, hi)] + [(np.asarray(p.low, float), np.asarray(p.high, float)) for p in pins.values()]
    return all(point_box_distance(arm.tips, a, b) > L.LET_GO_GAP for a, b in boxes)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


ZERO_TEXT = {
    "strike": "a pin came within 6 mm of the needle line (or crossed it) under the needle head",
    "knocked_loose": "a pin left the plate with no gripper holding it",
    "moved_unheld": "the work moved more than 5 mm in all while no gripper held its handle",
    "off_table": "a pin fell off the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a non-finite state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.zeros: dict[str, dict] = {}  # zero condition -> when and what
        self.feed = 0.0
        self.feed_max = 0.0
        self.feed_speed = 0.0
        self.anchor = None  # the feed where travel was last counted
        self.unheld_travel = 0.0  # m the work moved while no gripper held its handle
        self.total_travel = 0.0
        self.held_now = False
        self.prev_held = False
        self.pulls: dict[str, dict] = {}  # pin -> where it was first pulled (left the plate held)
        self.min_d = {n: math.inf for n in L.PINS}  # closest each pin came to the needle line while in the plate
        self.min_gap = {n: math.inf for n in L.PINS}  # closest each pin's box came to the needle line under the head
        self.side: dict[str, int | None] = {n: None for n in L.PINS}  # its side of the needle line under the head
        self.prev_in_work: dict[str, bool] | None = None
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (feed, [(centre, velocity) of each pin]) at the previous valid control step

    @staticmethod
    def at_start(feed: float, pins: dict) -> bool:
        """The starting layout: the work at the start of its rail, every pin upright in its pocket, all still."""
        if not (valid_feed(feed) and abs(float(feed)) < 0.002):
            return False
        for name, start in L.PIN_START.items():
            s = pins.get(name)
            if s is None or not valid(s):
                return False
            if not (np.linalg.norm(np.asarray(s.pos, float) - start) < L.START_TOL and tilt_deg(s.quat_xyzw) < 2.0
                    and is_resting(s)):
                return False
        return True

    def _teleported(self, feed: float, now) -> bool:
        """The work or a pin moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        f0, pins0 = self.prev
        if abs(feed - f0) > L.RESET_JUMP:
            return True
        return any(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for (p, _), (p0, v0) in zip(now, pins0))

    def _zero(self, key: str, **what) -> None:
        self.zeros.setdefault(key, {"step": self.steps, **what})

    def _ordered(self, pins: dict) -> list:
        return [pins.get(n) for n in L.PINS]

    def update(self, feed, pins: dict, arms=None) -> Result:
        states = self._ordered(pins)
        if not (valid_feed(feed) and all(valid(s) for s in states)):  # a blow-up: score 0, keep the record
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "zeros": sorted(self.zeros)})
            return self.last
        feed = float(feed)
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in states]
        if self.prev is not None and self._teleported(feed, now) and self.at_start(feed, pins):
            self.reset()  # the simulator put the work and every pin back: a new episode (rigstate's and mechkit's
            #               reset hooks normally do this first; this is the fallback)
        self.feed_speed = 0.0 if self.prev is None else (feed - self.prev[0]) / self.dt
        self.prev = (feed, now)
        self.steps += 1
        self.feed = feed
        self.feed_max = max(self.feed_max, feed)

        have_arms = arms_valid(arms)
        if have_arms:
            self.held_now = any(holds_handle(arms[s], feed) for s in ARMS)
        else:
            self.arms_unavailable += 1
            self.held_now = True  # unreadable: taken as held, counted
        if self.anchor is None:
            self.anchor = feed
        elif abs(feed - self.anchor) >= L.TRAVEL_STEP:
            moved = abs(feed - self.anchor)
            self.anchor = feed
            self.total_travel += moved
            if not (self.held_now or self.prev_held):
                self.unheld_travel += moved
        self.prev_held = self.held_now
        if self.unheld_travel > L.UNHELD_TRAVEL:
            self._zero("moved_unheld", travel_mm=_mm(self.unheld_travel))

        in_now = {n: in_work(s, feed) for n, s in zip(L.PINS, states)}
        for n, s in zip(L.PINS, states):
            d = pin_d(s)
            if in_now[n]:
                self.min_d[n] = min(self.min_d[n], d)
            was = in_now[n] if self.prev_in_work is None else self.prev_in_work[n]
            if was and not in_now[n]:  # it has left the plate this step
                held = [a for a in ARMS if holds_pin(arms[a], s)] if have_arms else ["unreadable"]
                if not held:
                    self._zero("knocked_loose", pin=n, d_mm=_mm(d))
                elif n not in self.pulls:
                    self.pulls[n] = {"by": held[0], "step": self.steps, "d": d}
            if under_head(s):
                gap = needle_gap(s)
                self.min_gap[n] = min(self.min_gap[n], gap)
                side = 1 if d >= 0 else -1
                crossed = self.side[n] is not None and side != self.side[n]
                if gap < L.STRIKE_GAP or crossed:
                    self._zero("strike", pin=n, d_mm=_mm(d), gap_mm=_mm(gap), crossed=crossed)
                self.side[n] = side
            else:
                self.side[n] = None
            if off_table(s):
                self._zero("off_table", pin=n)
        self.prev_in_work = in_now
        self.last = self.result(feed, pins, arms)
        return self.last

    def timely(self, name: str) -> bool:
        """Pulled with d between 30 and 70 mm, and never closer than 30 mm to the needle line while in the plate at
        any time in the trial."""
        p = self.pulls.get(name)
        return bool(p and L.WINDOW[0] <= p["d"] <= L.WINDOW[1] and self.min_d[name] >= L.WINDOW[0])

    def result(self, feed, pins: dict, arms=None) -> Result:
        states = self._ordered(pins)
        if not (valid_feed(feed) and all(valid(s) for s in states)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "zeros": sorted(self.zeros)})
        feed = float(feed)
        have_arms = arms_valid(arms)
        released = all(let_go(arms[s], feed, pins) for s in ARMS) if have_arms else True  # unreadable: taken as met
        dish = {n: in_dish(s) for n, s in zip(L.PINS, states)}
        timely = {n: self.timely(n) for n in L.PINS}
        n_timely = sum(timely.values())
        started = self.feed_max >= L.FEED_START
        complete = self.feed_max >= L.FEED_DONE
        all_dish = all(dish.values())
        if self.zeros:
            score = 0.0
        elif complete and n_timely == 3 and all_dish and released:
            score = 1.0
        elif complete and n_timely == 3:
            score = 0.75
        elif n_timely >= 2:
            score = 0.5
        elif n_timely >= 1 and started:
            score = 0.25
        else:
            score = 0.0
        still = all(is_resting(s) for s in states) and abs(self.feed_speed) < L.FEED_STILL
        success = bool(score == 1.0 and still)
        holder = None
        if have_arms:
            holder = [s for s in ARMS if holds_handle(arms[s], feed)] or None
        details = {
            "valid_state": True,
            "zeros": sorted(self.zeros),
            "zero_at": self.zeros or None,
            "zero_reasons": [ZERO_TEXT[k] for k in sorted(self.zeros)],
            "feed_mm": _mm(feed),
            "feed_max_mm": _mm(self.feed_max),
            "seam_started": started,
            "seam_complete": complete,
            "handle_held_by": holder,
            "unheld_travel_mm": _mm(self.unheld_travel),
            "total_travel_mm": _mm(self.total_travel),
            "pin_d_mm": {n: _mm(pin_d(s)) for n, s in zip(L.PINS, states)},
            "in_work": sorted(n for n, s in zip(L.PINS, states) if in_work(s, feed)),
            "pulls": {n: {"by": p["by"], "step": p["step"], "d_mm": _mm(p["d"])} for n, p in self.pulls.items()},
            "min_d_mm": {n: _mm(v) if math.isfinite(v) else None for n, v in self.min_d.items()},
            "pulled_in_time": sorted(n for n, v in timely.items() if v),
            "min_needle_gap_mm": {n: _mm(v) if math.isfinite(v) else None for n, v in self.min_gap.items()},
            "in_dish": sorted(n for n, v in dish.items() if v),
            "pin_low_mm": {n: _mm(float(s.low[2]) - L.TOP) for n, s in zip(L.PINS, states)},
            "pin_tilt_deg": {n: round(tilt_deg(s.quat_xyzw), 1) for n, s in zip(L.PINS, states)},
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "still": still,
        }
        return Result(score, success, details)
