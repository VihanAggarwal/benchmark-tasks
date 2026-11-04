"""The scorer, as plain functions of the part's state and the arms' state. NumPy only; no simulator needed.

The rules are about the whole trial, not only its end. A two-hand control starts the press cycle only while both
run buttons are held at once, pressed within 0.5 s of each other (synchronous actuation) and held for the cycle's
2 s with the part seated in the nest, and on no step may a button be pressed while any gripper's fingers are in the
die space. So the scorer keeps a little state (when each button's current press began, when the current double
press began and whether it is synchronous and the part has stayed seated through it, whether a cycle has completed,
whether a breach has happened, and some counts for the trace) and is fed one observation per control step by
``task.success()``.

    Tracker.update(part, arms) -> Result(score, success, details)

``part`` is anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then no press is registered on that step (a running press
span, and so a running double press, is broken), no breach is recorded, 'let go' and 'released' are taken as met,
and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a zero quaternion (a
simulator blow-up) scores 0 for that step and leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn (and a
little more), so 'the fingers are in the die space' never misses a real entry. The zone reaches no lower than the
finger-tip point (the lowest point of a gripper pointing down), so a gripper passing over the nest with its tips
above the 20 cm top is outside it.

Each run button is scored by its own arm only: the left button by the left gripper, the right button by the right.
They stand 42 cm apart, each out of the other arm's reach, so one gripper can never be on both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
SIDES = ("left", "right")


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
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in SIDES:
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
    """Angle between the world's vertical and the nearest of the cube's three body axes (degrees, 0..54.7): a cube
    is the same whichever face is down, so this is 0 when any face is flat."""
    zz = np.abs(rotation(quat_xyzw)[2, :]).max()
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- boxes ---------------------------------------------------------------------------------------------------------
def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def zone_gap(arm, lo, hi) -> float:
    """How far a gripper's finger zone is from a world-aligned box (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_box_distance(p, lo, hi) for p in pts) - r


def _xy_gap(p) -> float:
    """Horizontal distance from a point to the die space's footprint (0 over it)."""
    dx = max(0.0, L.DIE_X0 - p[0], p[0] - L.DIE_X1)
    dy = max(0.0, L.DIE_Y0 - p[1], p[1] - L.DIE_Y1)
    return float(math.hypot(dx, dy))


def die_gap(arm) -> float:
    """How far a gripper's finger zone is from the die space (negative: it is inside). The zone reaches its radius
    sideways from each point of its axis, and no lower than the finger tips (the lowest point of a gripper pointing
    down), so a gripper passing over the nest with its tips above 20 cm is outside, whatever its opening: for each
    axis point, the larger of (horizontal distance to the footprint minus the radius) and (height above the space's
    top); the smallest of those over the axis."""
    pts, r = finger_zone(arm)
    return min(max(_xy_gap(p) - r, float(p[2]) - L.DIE_TOP_Z) for p in pts)


def in_die(arm) -> bool:
    """Any point of the gripper's finger zone is inside the die space's footprint below 20 cm above the table."""
    return die_gap(arm) < 0.0


def tips_to_button(arm, side: str) -> tuple[float, float]:
    """(horizontal distance from the finger-tip point to that side's button axis, its height above the button's
    top), metres."""
    t = np.asarray(arm.tips, dtype=float)
    return float(np.hypot(t[0] - L.BUTTON_X, t[1] - L.BUTTON_Y[side])), float(t[2] - L.BUTTON_TOP_Z)


def button_pressed(arm, side: str) -> bool:
    """That side's gripper is pressing its own run button: its finger-tip point within 16 mm of the button's axis
    horizontally and between 5 mm below and 10 mm above the button's top, with the fingers at most half open (47 mm
    apart)."""
    off, up = tips_to_button(arm, side)
    return bool(off <= L.PRESS_XY and -L.PRESS_Z_BELOW <= up <= L.PRESS_Z_ABOVE
                and float(arm.opening) <= L.PRESS_OPENING)


def button_released(arm, side: str) -> bool:
    """That side's gripper is clearly off its button: its finger-tip point more than 30 mm above the button's top,
    or more than 30 mm from its axis horizontally. (Between 'pressed' and 'released' there is a band that is
    neither.)"""
    off, up = tips_to_button(arm, side)
    return bool(up > L.RELEASE_Z or off > L.RELEASE_XY)


def let_go(arm, lo, hi) -> bool:
    """The gripper is not holding the item in the box (lo, hi): it is open (opening 0.9 or more), or its finger zone
    is more than 10 mm from the item. A gripper closed somewhere else is not holding it."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, lo, hi) > L.LET_GO_GAP)


def grippers_let_go(arms, part) -> bool:
    """Neither gripper holds the part."""
    lo, hi = _lo_hi(part)
    return let_go(arms["left"], lo, hi) and let_go(arms["right"], lo, hi)


# -- the part -------------------------------------------------------------------------------------------------------
def part_seated(part) -> bool:
    """The part sits in the nest: its centre within 4 mm of the nest's centre horizontally, its lowest point within
    3 mm of the table top, tilted less than 5 degrees. (The 36 mm floor between 8 mm rails leaves 3 mm each way, so a
    part inside the rails and flat is always seated; one up on a rail is tilted or too high.)"""
    off = float(np.hypot(part.pos[0] - L.NEST_X, part.pos[1] - L.NEST_Y))
    return bool(off <= L.SEAT_XY_TOL and abs(float(part.low[2]) - L.TOP) <= L.SEAT_Z_TOL
                and tilt_deg(part.quat_xyzw) < L.SEAT_TILT_DEG)


def part_in_tray(part) -> bool:
    """The part rests in the output tray: its centre over the tray's 80 x 140 mm floor and its lowest point within
    3 mm of the table top."""
    x, y = float(part.pos[0]), float(part.pos[1])
    over = abs(x - L.TRAY_X) <= L.TRAY_INNER_X / 2 and abs(y - L.TRAY_Y) <= L.TRAY_INNER_Y / 2
    return bool(over and abs(float(part.low[2]) - L.TOP) <= L.IN_TRAY_Z_TOL)


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


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
        self.sync_steps = int(round(L.SYNC_S * control_hz))
        self.cycle_steps = int(round(L.CYCLE_S * control_hz))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        # the buttons
        self.press_start: dict[str, int | None] = {"left": None, "right": None}  # step the current press began
        self.press_steps = {"left": 0, "right": 0}  # steps each button has been pressed in all (traced)
        # the double press (both pressed on the same step)
        self.double_start: int | None = None  # step the current double press began (None: not both pressed now)
        self.double_sync = False  # the two presses of the current double press began within SYNC_S of each other
        self.double_seated = False  # the part has been seated on every step of the current double press
        self.double_len = 0  # steps of the current double press so far
        self.double_presses = 0  # how many double presses began
        self.seated_presses = 0  # ... of which with the part seated in the nest (an attempt at a cycle)
        self.sync_presses = 0  # ... of which synchronous
        self.longest_double = 0  # the longest double press so far (steps, traced)
        self.double_gap_steps: int | None = None  # how far apart the two presses of the latest double press began
        # the cycle
        self.cycled = False  # a cycle has completed: synchronous double press held CYCLE_S with the part seated
        self.cycle_step: int | None = None  # the step on which it completed
        self.cycle_seated_steps = 0  # steps of the completed cycle (traced)
        # the breach
        self.breach_steps = 0  # control steps with a button pressed and a gripper's finger zone in the die space
        self.first_breach_step: int | None = None
        self.breach_arms: set[str] = set()  # which grippers were in the die space on a breach step
        self.in_die_steps = 0  # control steps with a gripper in the die space (any, button pressed or not)
        # bookkeeping
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (centre, velocity) of the part at the previous valid control step

    @property
    def breach(self) -> bool:
        return self.breach_steps > 0

    @property
    def attempted(self) -> bool:
        """A double press began with the part seated in the nest: an attempt at a cycle, whether or not it was
        synchronous, held long enough or kept the part seated. (A double press with the part elsewhere is an empty
        cycle and counts for nothing.)"""
        return self.seated_presses > 0

    @staticmethod
    def at_start(part) -> bool:
        """The starting layout: the part flat on the feed mark (within START_TOL, which covers a seeded reset's
        jitter) and still."""
        on_mark = (np.hypot(part.pos[0] - L.PLATE_X, part.pos[1] - L.PLATE_Y) < L.START_TOL
                   and abs(float(part.low[2]) - L.PLATE_TOP_Z) < L.START_Z_TOL and tilt_deg(part.quat_xyzw) < 1.0)
        return bool(on_mark and is_resting(part))

    def _teleported(self, now) -> bool:
        """The part moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        (p, _), (p0, v0) = now, self.prev
        return bool(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall)

    def _buttons(self, pressed: dict[str, bool], seated: bool) -> None:
        """Advance the press spans, the double press and the cycle by one step."""
        for side in SIDES:
            if pressed[side]:
                self.press_steps[side] += 1
                if self.press_start[side] is None:
                    self.press_start[side] = self.steps
            else:
                self.press_start[side] = None
        if pressed["left"] and pressed["right"]:
            if self.double_start is None:
                self.double_start = self.steps
                self.double_gap_steps = abs(self.press_start["left"] - self.press_start["right"])
                self.double_sync = self.double_gap_steps <= self.sync_steps
                self.double_seated = True
                self.double_len = 0
                self.double_presses += 1
                self.seated_presses += int(seated)
                self.sync_presses += int(self.double_sync)
            self.double_len += 1
            self.longest_double = max(self.longest_double, self.double_len)
            if not seated:
                self.double_seated = False
            if self.double_sync and self.double_seated and self.double_len >= self.cycle_steps and not self.cycled:
                self.cycled = True
                self.cycle_step = self.steps
                self.cycle_seated_steps = self.double_len
        else:
            self.double_start = None
            self.double_len = 0

    def update(self, part, arms=None) -> Result:
        if not valid(part):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(part, arms)
            return self.last
        now = (np.array(part.pos, dtype=float), np.array(part.lin_vel, dtype=float))
        if self.prev is not None and self._teleported(now) and self.at_start(part):
            self.reset()  # the simulator put the part back on its mark: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        seated = part_seated(part)

        if arms_valid(arms):
            pressed = {s: button_pressed(arms[s], s) for s in SIDES}
            inside = {s: in_die(arms[s]) for s in SIDES}
            anyone = any(inside.values())
            if anyone:
                self.in_die_steps += 1
            if anyone and any(pressed.values()):
                self.breach_steps += 1
                if self.first_breach_step is None:
                    self.first_breach_step = self.steps
                self.breach_arms |= {s for s, v in inside.items() if v}
            self._buttons(pressed, seated)
        else:  # unreadable: no press registered (a running double press is broken), no breach recorded
            self.arms_unavailable += 1
            self._buttons({"left": False, "right": False}, seated)
        self.last = self.result(part, arms, seated)
        return self.last

    def result(self, part, arms=None, seated: bool | None = None) -> Result:
        if not valid(part):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "breach": self.breach,
                                       "breach_steps": self.breach_steps, "cycled": self.cycled})
        seated = part_seated(part) if seated is None else seated
        have_arms = arms_valid(arms)
        if have_arms:
            pressed = {s: button_pressed(arms[s], s) for s in SIDES}
            released = {s: button_released(arms[s], s) for s in SIDES}
            let_go_ = grippers_let_go(arms, part)
            inside = {s: in_die(arms[s]) for s in SIDES}
            gaps = {s: die_gap(arms[s]) for s in SIDES}
            to_button = {s: tips_to_button(arms[s], s) for s in SIDES}
        else:  # unreadable: 'released' and 'let go' taken as met, counted
            pressed = {s: False for s in SIDES}
            released = {s: True for s in SIDES}
            let_go_, inside, gaps, to_button = True, None, None, None
        in_tray = part_in_tray(part)
        gone = off_table(part)
        if self.breach or gone:
            score = 0.0
        elif self.cycled and in_tray and let_go_ and all(released.values()):
            score = 1.0
        elif self.cycled:
            score = 0.75
        elif seated and self.attempted:
            score = 0.5
        elif seated:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(part))
        details = {
            "valid_state": True,
            "breach": self.breach,
            "breach_steps": self.breach_steps,
            "first_breach_step": self.first_breach_step,
            "breach_arms": sorted(self.breach_arms),
            "in_die": inside,
            "die_gap_mm": ({s: _mm(g) for s, g in gaps.items()} if gaps is not None else None),
            "pressed": pressed if have_arms else None,
            "released": released if have_arms else None,
            "tips_to_button_axis_mm": ({s: _mm(v[0]) for s, v in to_button.items()} if to_button else None),
            "tips_above_button_mm": ({s: _mm(v[1]) for s, v in to_button.items()} if to_button else None),
            "press_s": {s: round(self.press_steps[s] * self.dt, 2) for s in SIDES},
            "double_pressed_now": self.double_start is not None,
            "double_now_s": round(self.double_len * self.dt, 2),
            "double_now_sync": self.double_sync if self.double_start is not None else None,
            "double_now_seated": self.double_seated if self.double_start is not None else None,
            "double_gap_s": (round(self.double_gap_steps * self.dt, 2) if self.double_gap_steps is not None else None),
            "double_presses": self.double_presses,
            "seated_presses": self.seated_presses,
            "sync_presses": self.sync_presses,
            "longest_double_s": round(self.longest_double * self.dt, 2),
            "cycled": self.cycled,
            "cycle_step": self.cycle_step,
            "part_seated": seated,
            "part_in_tray": in_tray,
            "part_off_table": gone,
            "part_low_mm": _mm(float(part.low[2]) - L.TOP),
            "part_to_nest_mm": _mm(float(np.hypot(part.pos[0] - L.NEST_X, part.pos[1] - L.NEST_Y))),
            "part_tilt_deg": round(tilt_deg(part.quat_xyzw), 2),
            "grippers_let_go": let_go_ if have_arms else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in SIDES} if have_arms else None),
            "in_die_steps": self.in_die_steps,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "part_resting": is_resting(part),
        }
        return Result(score, success, details)
