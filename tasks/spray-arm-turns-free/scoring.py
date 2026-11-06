"""The scorer, as plain functions of the four items' states, the spray arm's angle and the two grippers. NumPy only; no
simulator needed.

The task has an end state (the rack loaded so the arm can turn: every item standing on the rack floor, the tall jars
outside the arm's sweep, the grippers let go) and one rule about the trial: the arm was turned at least half a turn
while the load stood still. So the scorer keeps a little record and is fed one observation per control step by
``task.success()``:

    Tracker.update(items, arm_yaw, arms) -> Result(score, success, details)

``items`` maps each of ``layout.ITEMS`` to anything with ``pos`` (the centre) and ``quat_xyzw``, as
``yam_arena.task_file.ObjectState`` has (``lin_vel``, ``ang_vel``, ``low``, ``high`` are not used, but a non-finite
one marks a broken state). ``arm_yaw`` is the spray arm's angle (rad, anticlockwise seen from above, the direction of
its red end; wrapped or not: the tracker counts it on continuously by adding each step's wrapped change). ``arms`` is
``rigstate.arms()``: each gripper's tool point, finger-tip point and opening, or None when they cannot be read.

An item is LOADED when its centre, seen from above, is inside the rack's inner walls, its axis is within 10 degrees of
vertical (either end up: the bowls go in upside down, and a cylinder is the same either way) and its lowest point is
within 5 mm of the rack floor's top. A jar is CLEAR when its centre is at least 14.05 cm from the hub's axis and, at
every height the arm and its cap pass (8.5 to 10.9 cm above the table), its side is at least 11.55 cm from that axis
(5 mm or more outside the 11.05 cm circle the arm's ends sweep; the jar's cross-section there is taken as a circle of
radius 2.5 cm / cos(tilt), so this bites only for a jar leaning toward the hub). A gripper is ON an item when its
fingers come within 2 cm of the item's side, horizontally, with its pad-tip point (midway between the finger tips)
below the item's top; the fingers are taken as a disc round the pad-tip point out to a finger's outer corner
(hypot(half the gap + 10 mm, 13.5 mm): 16.8 mm closed, 58.5 mm fully open). A LOAD STRETCH runs while all four items
are loaded, both jars are clear, no gripper is on an item and no item's centre has moved more than 10 mm from where it
was when the stretch began; a stretch has a CLEAN HALF TURN once the arm's angle has been 180 degrees or more from its
angle at the stretch's start (either way; back and forth does not add up). Full marks need the clean half turn in the
stretch still running at the end: the load let go of and left as it stood while the arm turned (touching it, moving it
more than 10 mm afterwards, or taking it out of the rules, ends that stretch and its turn with it). Arms that cannot be
read on a step are taken from the last step they could be read (never read: no gripper is on an item).

Stillness (for success only) is judged by each item's path over the last 0.5 s, never by its reported velocity (a
resting object's velocity chatters in the simulator).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

ARMS = ("left", "right")
HUB = np.array(L.HUB, dtype=float)
TURN_NEED = math.radians(L.TURN_NEED_DEG)
_OPTIONAL = ("lin_vel", "ang_vel", "low", "high")


# -- states ---------------------------------------------------------------------------------------------------------
def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a unit quaternion (x, y, z, w); its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """A finite centre and a non-degenerate quaternion, and nothing else non-finite (a simulator blow-up)."""
    try:
        pos = np.asarray(s.pos, dtype=float).ravel()
        quat = np.asarray(s.quat_xyzw, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return False
    if pos.size != 3 or quat.size != 4 or not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return False
    if np.linalg.norm(quat) < 0.5:
        return False
    for k in _OPTIONAL:
        v = getattr(s, k, None)
        if v is None:
            continue
        try:
            if not np.all(np.isfinite(np.asarray(v, dtype=float))):
                return False
        except (TypeError, ValueError):
            return False
    return True


def items_valid(items) -> bool:
    try:
        return all(valid(items[n]) for n in L.ITEMS)
    except (KeyError, TypeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with a finite finger-tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            tips = np.asarray(arms[side].tips, dtype=float).ravel()
            if tips.size != 3 or not np.all(np.isfinite(tips)) or not math.isfinite(float(arms[side].opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- the items ------------------------------------------------------------------------------------------------------
def centre(s) -> np.ndarray:
    return np.asarray(s.pos, dtype=float).ravel()


def tilt_deg(s) -> float:
    """Angle between the vertical and the cylinder's axis (degrees, 0..90): either end may be up."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def _extent_z(name: str, s) -> float:
    """Half the cylinder's height along the vertical, as tilted: its lowest and highest points are this far from its
    centre."""
    r, h = L.SIZE[name]
    t = math.radians(tilt_deg(s))
    return 0.5 * h * math.cos(t) + r * math.sin(t)


def bottom(name: str, s) -> float:
    """The item's lowest point (world z)."""
    return float(centre(s)[2]) - _extent_z(name, s)


def top(name: str, s) -> float:
    """The item's highest point (world z)."""
    return float(centre(s)[2]) + _extent_z(name, s)


def inside_rack(p) -> bool:
    """A point's (x, y) inside the rack's inner walls."""
    return bool(L.INNER_X[0] <= p[0] <= L.INNER_X[1] and L.INNER_Y[0] <= p[1] <= L.INNER_Y[1])


def loaded(name: str, s) -> bool:
    """Standing on the rack floor inside its walls: centre inside the walls (seen from above), axis within 10 degrees
    of vertical, lowest point within 5 mm of the floor's top (so not lying, stacked, propped on a wall or held up)."""
    return bool(inside_rack(centre(s)) and tilt_deg(s) <= L.LOAD_TILT_DEG
                and abs(bottom(name, s) - L.FLOOR_TOP) <= L.LOAD_BOTTOM)


def hub_distance(s) -> float:
    """Horizontal distance from the item's centre to the hub's axis (m)."""
    c = centre(s)
    return float(np.hypot(c[0] - HUB[0], c[1] - HUB[1]))


def _segment_distance(p, a, b) -> float:
    """Distance from the 2D point ``p`` to the segment from ``a`` to ``b``."""
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-18 else float(np.clip((p - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def jar_side_at_arm(s) -> float:
    """The smallest horizontal distance from the hub's axis to the jar's side at the heights the arm and its cap pass
    (8.5 to 10.9 cm above the table), its cross-section there taken as a circle of radius r / cos(tilt) round its
    axis (m). -inf for a jar tilted 60 degrees or more (lying: never clear)."""
    c = centre(s)
    a = rotation(s.quat_xyzw)[:, 2]
    if abs(a[2]) < 0.5:
        return -math.inf
    p0, p1 = (c[:2] + a[:2] * ((L.TOP + z - c[2]) / a[2]) for z in L.CLEAR_Z)
    return _segment_distance(HUB, p0, p1) - L.JAR_R / abs(float(a[2]))


def jar_clear(s) -> bool:
    """A jar's centre at least 14.05 cm from the hub's axis and its side at least 11.55 cm from it at the arm's
    height: its side 5 mm or more outside the arm's sweep, even if it leans toward the hub."""
    return bool(hub_distance(s) >= L.CLEAR_R - 1e-9 and jar_side_at_arm(s) >= L.CLEAR_SIDE - 1e-9)


def finger_reach(arm) -> float:
    """From the pad-tip point (midway between the finger tips) to a finger's outer corner (m): the fingers taken as a
    disc that size round it, whichever way the gripper is turned."""
    gap = float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def held_by(arm, name: str, s) -> bool:
    """The gripper's fingers within 20 mm of the item's side (horizontally) with its pad-tip point below the item's
    top: still holding or touching it."""
    tips = np.asarray(arm.tips, dtype=float).ravel()
    c = centre(s)
    side_gap = float(np.hypot(tips[0] - c[0], tips[1] - c[1])) - L.SIZE[name][0] - finger_reach(arm)
    return bool(side_gap < L.HELD_MARGIN and float(tips[2]) < top(name, s))


def holds(arms, items) -> list[str]:
    """``side:item`` for every gripper still holding or touching an item."""
    if not arms_valid(arms):
        return []
    return [f"{side}:{n}" for side in ARMS for n in L.ITEMS if held_by(arms[side], n, items[n])]


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a non-finite state (ignored)
        self.arms_unavailable = 0  # control steps on which the arms could not be read
        self.theta: float | None = None  # the arm's angle, counted on continuously (rad)
        self.raw_prev: float | None = None
        # the load stretch now running: {"start": step, "ref": {item: centre}, "theta": angle at its start,
        # "turn": the largest turn within it (rad), "half_step": the step it reached a clean half turn, or None}
        self.stretch: dict | None = None
        self.stretches = 0  # how many load stretches have started
        self.max_drift = 0.0  # m: the furthest an item moved from its place within the current stretch
        self.best_turn = 0.0  # rad: the largest turn seen within any one stretch (diagnostics)
        self.first_half_step: int | None = None  # the first step any stretch reached a clean half turn (diagnostics)
        self.history: deque = deque(maxlen=L.STILL_STEPS + 1)  # the items' centres over the last 0.5 s
        self.prev: dict | None = None  # the items' centres at the previous valid step
        self.arms_seen = None  # the last readable arms
        self.last = Result(0.0, False, {})

    @property
    def turned(self) -> bool:
        """The load stretch now running has had a clean half turn."""
        return bool(self.stretch is not None and self.stretch["half_step"] is not None)

    @property
    def latch_step(self) -> int | None:
        return self.stretch["half_step"] if self.stretch is not None else None

    # -- per step ----------------------------------------------------------------------------------------------------
    def _teleported_home(self, centres: dict, raw: float) -> bool:
        """The simulator put everything back: some item jumped more than 20 mm in one step and now every item is
        within 5 mm of its start, the arm back along x with its red end forward (within 5 degrees)."""
        jump = max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in L.ITEMS)
        home = all(float(np.linalg.norm(centres[n] - np.asarray(L.START[n]))) <= L.START_TOL for n in L.ITEMS)
        arm_home = abs(math.degrees(math.remainder(raw, 2 * math.pi))) <= L.START_ARM_TOL_DEG
        return bool(jump > L.RESET_JUMP and home and arm_home)

    def still(self) -> tuple[bool, float]:
        """Every item's path over the last 0.5 s (10 control steps) under 5 mm; and the largest such path (m)."""
        if len(self.history) < L.STILL_STEPS + 1:
            return False, math.inf
        frames = list(self.history)
        worst = max(sum(float(np.linalg.norm(b[n] - a[n])) for a, b in zip(frames, frames[1:])) for n in L.ITEMS)
        return bool(worst < L.STILL_PATH), worst

    def update(self, items, arm_yaw, arms=None) -> Result:
        try:
            raw = float(arm_yaw)
        except (TypeError, ValueError):
            raw = math.nan
        if not (items_valid(items) and math.isfinite(raw)):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latched": self.turned})
            return self.last
        centres = {n: centre(items[n]).copy() for n in L.ITEMS}
        if self.prev is not None and self._teleported_home(centres, raw):
            self.reset()  # a new episode (rigstate's and mechkit's reset hooks normally come first)
        self.prev = centres
        self.steps += 1
        if self.raw_prev is None:
            self.theta = raw
        else:
            self.theta += math.remainder(raw - self.raw_prev, 2 * math.pi)
        self.raw_prev = raw
        if arms_valid(arms):
            self.arms_seen = arms
        else:
            self.arms_unavailable += 1
        self.history.append(centres)

        # a stretch runs only with the load as it stands, let go of: no gripper on an item (the last arms that could
        # be read), so a turn made while the last item is still in a gripper, or with a hand on the load, never counts
        ok = (all(loaded(n, items[n]) for n in L.ITEMS) and all(jar_clear(items[j]) for j in L.JARS)
              and not holds(self.arms_seen, items))
        if self.stretch is not None:
            drift = max(float(np.linalg.norm(centres[n] - self.stretch["ref"][n])) for n in L.ITEMS)
            if not ok or drift > L.STRETCH_MOVE:
                self.stretch = None  # the load moved (or is no longer all loaded and clear): the stretch ends
            else:
                self.max_drift = max(self.max_drift, drift)
        if self.stretch is None and ok:  # a new stretch can start on the same step
            self.stretch = {"start": self.steps, "ref": centres, "theta": self.theta, "turn": 0.0, "half_step": None}
            self.stretches += 1
            self.max_drift = 0.0
        if self.stretch is not None:
            st = self.stretch
            st["turn"] = max(st["turn"], abs(self.theta - st["theta"]))
            self.best_turn = max(self.best_turn, st["turn"])
            if st["turn"] >= TURN_NEED - 1e-9 and st["half_step"] is None:
                st["half_step"] = self.steps
                if self.first_half_step is None:
                    self.first_half_step = self.steps
        self.last = self.result(items, arms, raw)
        return self.last

    # -- the score -----------------------------------------------------------------------------------------------------
    def result(self, items, arms=None, arm_yaw=0.0) -> Result:
        try:
            raw = float(arm_yaw)
        except (TypeError, ValueError):
            raw = math.nan
        if not (items_valid(items) and math.isfinite(raw)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "latched": self.turned})
        have_arms = arms_valid(arms)
        seen = arms if have_arms else self.arms_seen  # unreadable: the last snapshot that could be read
        is_loaded = {n: loaded(n, items[n]) for n in L.ITEMS}
        clear = {j: jar_clear(items[j]) for j in L.JARS}
        held = holds(seen, items)
        if not all(is_loaded.values()) or held:
            score = 0.0
        elif not all(clear.values()):
            score = 0.25
        elif not self.turned:
            score = 0.5
        else:
            score = 1.0
        still, path = self.still()
        success = bool(score == 1.0 and still)
        st = self.stretch
        details = {
            "valid_state": True,
            "loaded": {n: is_loaded[n] for n in L.ITEMS},
            "jar_clear": clear,
            "jar_hub_mm": {j: _mm(hub_distance(items[j])) for j in L.JARS},
            "jar_side_mm": {j: _mm(jar_side_at_arm(items[j])) for j in L.JARS},
            "tilt_deg": {n: round(tilt_deg(items[n]), 2) for n in L.ITEMS},
            "bottom_mm": {n: _mm(bottom(n, items[n]) - L.FLOOR_TOP) for n in L.ITEMS},
            "held": held,
            "turn_deg": round(math.degrees(self.theta), 2) if self.theta is not None else None,
            "stretch_start": st["start"] if st else None,
            "stretch_turn_deg": round(math.degrees(abs(self.theta - st["theta"])), 2) if st else None,
            "stretch_best_deg": round(math.degrees(st["turn"]), 2) if st else None,
            "best_turn_deg": round(math.degrees(self.best_turn), 2),
            "stretches": self.stretches,
            "max_drift_mm": _mm(self.max_drift),
            "latched": self.turned,
            "latch_step": self.latch_step,
            "first_half_step": self.first_half_step,
            "still": still,
            "path_mm": _mm(path),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
