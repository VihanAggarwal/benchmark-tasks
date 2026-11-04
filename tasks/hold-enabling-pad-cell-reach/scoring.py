"""The scorer, as plain functions of the dropped part's state and the arms' state. NumPy only; no simulator needed.

The rule is about the whole trial, not only its end: at every control step on which any gripper is inside the
cell's safeguarded volume, the enabling pad must be pressed on that same step, and at the end the pad must be
released again (an enabling device is never tied down). So the scorer keeps a little state (whether a breach has
happened, whether the part has been lifted while the pad was pressed, and some counts for the trace) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(part, arms) -> Result(score, success, details)

``part`` is anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then no breach is recorded for that step, the pad is taken
as pressed, 'let go' and 'released' as met, and the step is counted in ``arms_unavailable``. An object state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn (and a
little more), so 'the fingers are inside the cell' never misses a real entry. The zone reaches no lower than the
finger-tip point (the lowest point of a gripper pointing down), so a gripper passing over the cell with its tips
above the 30 cm top is outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
# The safeguarded volume: the cell's footprint from far below the table (the zone can never be there) up to 30 cm
# above the table top.
CELL_LO = np.array([L.CELL_X0, L.CELL_Y0, L.TOP - 1.0])
CELL_HI = np.array([L.CELL_X1, L.CELL_Y1, L.CELL_TOP_Z])


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
        for side in ("left", "right"):
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
    """Horizontal distance from a point to the cell's footprint (0 over it)."""
    dx = max(0.0, L.CELL_X0 - p[0], p[0] - L.CELL_X1)
    dy = max(0.0, L.CELL_Y0 - p[1], p[1] - L.CELL_Y1)
    return float(math.hypot(dx, dy))


def cell_gap(arm) -> float:
    """How far a gripper's finger zone is from the cell's safeguarded volume (negative: it is inside). The zone
    reaches its radius sideways from each point of its axis, and no lower than the finger tips (the lowest point
    of a gripper pointing down), so a gripper passing over the cell with its tips above 30 cm is outside, whatever
    its opening: for each axis point, the larger of (horizontal distance to the footprint minus the radius) and
    (height above the volume's top); the smallest of those over the axis."""
    pts, r = finger_zone(arm)
    return min(max(_xy_gap(p) - r, float(p[2]) - L.CELL_TOP_Z) for p in pts)


def in_cell(arm) -> bool:
    """Any point of the gripper's finger zone is inside the cell's footprint below 30 cm above the table."""
    return cell_gap(arm) < 0.0


def tips_to_pad(arm) -> tuple[float, float]:
    """(horizontal distance from the finger-tip point to the pad's axis, its height above the pad's top), metres."""
    t = np.asarray(arm.tips, dtype=float)
    return float(np.hypot(t[0] - L.PAD_X, t[1] - L.PAD_Y)), float(t[2] - L.PAD_TOP_Z)


def pad_pressed(arm) -> bool:
    """The gripper is pressing the enabling pad: its finger-tip point within 16 mm of the pad's axis horizontally and
    between 5 mm below and 10 mm above the pad's top, with the fingers at most half open (47 mm apart)."""
    off, up = tips_to_pad(arm)
    return bool(off <= L.PRESS_XY and -L.PRESS_Z_BELOW <= up <= L.PRESS_Z_ABOVE and float(arm.opening) <= L.PRESS_OPENING)


def pad_released(arm) -> bool:
    """The gripper is clearly off the pad: its finger-tip point more than 30 mm above the pad's top, or more than
    30 mm from its axis horizontally. (Between 'pressed' and 'released' there is a band that is neither.)"""
    off, up = tips_to_pad(arm)
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
def centre_in_cell(part) -> bool:
    """The part's centre is over the cell's floor (inside the footprint, at any height)."""
    x, y = float(part.pos[0]), float(part.pos[1])
    return bool(L.CELL_X0 <= x <= L.CELL_X1 and L.CELL_Y0 <= y <= L.CELL_Y1)


def part_lifted(part) -> bool:
    """The part's lowest point at least 20 mm above the table top."""
    return bool(float(part.low[2]) >= L.TOP + L.LIFT_CLEAR)


def part_in_tray(part) -> bool:
    """The part rests in the recovery tray: its centre over the tray's 80 x 140 mm floor and its lowest point within
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
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breach_steps = 0  # control steps with a gripper in the cell and the pad not pressed
        self.first_breach_step: int | None = None
        self.breach_arms: set[str] = set()  # which grippers were inside on a breach step
        self.lifted_pressed = False  # the part has been 20 mm off the table while the pad was pressed
        self.in_cell_steps = 0  # control steps with a gripper in the cell (any, pad pressed or not)
        self.entries = 0  # how many times a gripper went from outside the cell to inside (either gripper)
        self.inside_prev = False  # a gripper was inside on the previous readable step
        self.pressed_idle_before = 0  # steps with the pad pressed and no gripper in the cell before the first entry
        self.pressed_idle_after = 0  # ... and after the latest exit (not scored; traced)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (centre, velocity) of the part at the previous valid control step

    @property
    def breach(self) -> bool:
        return self.breach_steps > 0

    @staticmethod
    def at_start(part) -> bool:
        """The starting layout: the part flat on the cell floor on its mark (within START_TOL, which covers a seeded
        reset's jitter) and still."""
        on_mark = (np.hypot(part.pos[0] - L.PART_X, part.pos[1] - L.PART_Y) < L.START_TOL
                   and abs(float(part.low[2]) - L.TOP) < L.START_Z_TOL and tilt_deg(part.quat_xyzw) < 1.0)
        return bool(on_mark and is_resting(part))

    def _teleported(self, now) -> bool:
        """The part moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        (p, _), (p0, v0) = now, self.prev
        return bool(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall)

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

        if arms_valid(arms):
            inside = {s: in_cell(arms[s]) for s in ("left", "right")}
            pressed = pad_pressed(arms["left"])
            anyone = any(inside.values())
            if anyone and not pressed:
                self.breach_steps += 1
                if self.first_breach_step is None:
                    self.first_breach_step = self.steps
                self.breach_arms |= {s for s, v in inside.items() if v}
            if anyone:
                self.in_cell_steps += 1
                if not self.inside_prev:
                    self.entries += 1
                self.pressed_idle_after = 0  # 'after' counts from the latest exit
            elif pressed:  # the pad held down with nobody inside: not scored, but traced
                if self.entries == 0:
                    self.pressed_idle_before += 1
                else:
                    self.pressed_idle_after += 1
            self.inside_prev = anyone
        else:
            self.arms_unavailable += 1
            pressed = True  # unreadable: the arm rules are taken as kept this step
        if part_lifted(part) and pressed:
            self.lifted_pressed = True
        self.last = self.result(part, arms)
        return self.last

    def result(self, part, arms=None) -> Result:
        if not valid(part):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breach": self.breach, "breach_steps": self.breach_steps})
        have_arms = arms_valid(arms)
        if have_arms:
            pressed = pad_pressed(arms["left"])
            released = pad_released(arms["left"])
            let_go_ = grippers_let_go(arms, part)
            inside = {s: in_cell(arms[s]) for s in ("left", "right")}
            gaps = {s: cell_gap(arms[s]) for s in ("left", "right")}
            off, up = tips_to_pad(arms["left"])
        else:  # unreadable: taken as met, counted
            pressed, released, let_go_ = False, True, True
            inside, gaps, off, up = None, None, None, None
        in_tray = part_in_tray(part)
        in_cell_ = centre_in_cell(part)
        gone = off_table(part)
        if self.breach or gone:
            score = 0.0
        elif in_tray and released and let_go_:
            score = 1.0
        elif in_tray:
            score = 0.75
        elif not in_cell_:
            score = 0.5
        elif self.lifted_pressed:
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
            "in_cell": inside,
            "cell_gap_mm": ({s: _mm(g) for s, g in gaps.items()} if gaps is not None else None),
            "pad_pressed": pressed if have_arms else None,
            "pad_released": released if have_arms else None,
            "left_tips_to_pad_axis_mm": _mm(off),
            "left_tips_above_pad_mm": _mm(up),
            "part_in_tray": in_tray,
            "part_centre_in_cell": in_cell_,
            "part_lifted": part_lifted(part),
            "lifted_while_pressed": self.lifted_pressed,
            "part_off_table": gone,
            "part_low_mm": _mm(float(part.low[2]) - L.TOP),
            "part_tilt_deg": round(tilt_deg(part.quat_xyzw), 2),
            "grippers_let_go": let_go_ if have_arms else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "in_cell_steps": self.in_cell_steps,
            "entries": self.entries,
            "pressed_idle_before_s": round(self.pressed_idle_before * self.dt, 2),
            "pressed_idle_after_s": round(self.pressed_idle_after * self.dt, 2),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "part_resting": is_resting(part),
        }
        return Result(score, success, details)
