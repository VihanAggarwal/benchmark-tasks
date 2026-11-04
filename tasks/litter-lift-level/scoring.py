"""The scorer, as plain functions of the litter's and the casualty's states and the arms' state. NumPy only; no
simulator needed.

The rule is about the whole trial, not only its end: the litter's two ends must go up and come down together, the
litter must stay level while it is handled, it must never be dragged along the table or the trolley, both grippers
must hold it the whole time it is in the air, it must be lowered onto the trolley and not dropped, and the casualty
must stay on it throughout. So the scorer keeps a little state (the worst that has happened so far: the largest
difference between the ends' heights and the largest tilt while an end was up, how far the litter has slid while
low, how fast it came down onto the trolley, which rules were broken and when, and whether it was ever carried
level) and is fed one observation per control step by ``task.success()``.

    Tracker.update(litter, casualty, arms) -> Result(score, success, details)

``litter`` and ``casualty`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. Both
are boxes whose sizes are in ``layout``, so their corners and faces follow from ``pos`` and ``quat_xyzw``. ``arms``
is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips)
and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the arm rules ("both
grippers hold it in the air", "both grippers have let go") are taken as kept for that step and the step is counted
in ``arms_unavailable``. An object state with a non-finite number or a zero quaternion (a simulator blow-up) scores
0 for that step and leaves the tracker as it was.

Heights: the litter's 'end heights' are the heights of the bottom edge of each 40 x 20 mm end face (the mean of
that face's two lower corners); the 'end difference' is how much they differ. 'Tilt' of the litter is the angle of
its long axis from the horizontal; 'flat' (on the trolley) is the angle of its broad face from the horizontal,
which also covers a roll about the long axis. The surface under the litter is the trolley's top when the litter's
centre is over the trolley's 120 x 360 mm footprint, else the table top; 'clearance' is its lowest point's height
above that surface.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
LITTER_HALF = np.array(L.LITTER_SIZE, dtype=float) / 2
CASUALTY_HALF = np.array(L.CASUALTY_SIZE, dtype=float) / 2


# -- states -------------------------------------------------------------------------------------------------------
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
        return True
    except (KeyError, AttributeError, TypeError, ValueError):
        return False


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- orientation ---------------------------------------------------------------------------------------------------
def face_tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through a box's two broad faces (degrees, 0..90): either
    face may be down, so whichever is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def axis_tilt_deg(quat_xyzw) -> float:
    """Angle of the litter's long axis (its body y) from the horizontal (degrees, 0..90): one end higher than the
    other. A roll about the long axis is not counted here (see ``face_tilt_deg``)."""
    zy = rotation(quat_xyzw)[2, 1]
    return float(np.degrees(np.arcsin(np.clip(abs(zy), 0.0, 1.0))))


def heading_err_deg(quat_xyzw) -> float:
    """How far the litter's long axis, seen from above, is turned from the world's y axis (degrees, 0..90; either
    way round)."""
    r = rotation(quat_xyzw)
    return float(np.degrees(math.atan2(abs(r[0, 1]), abs(r[1, 1]))))


def _up(r: np.ndarray) -> np.ndarray:
    """The body's z axis, turned to point upward (a plain box is the same either way up)."""
    return r[:, 2] if r[2, 2] >= 0.0 else -r[:, 2]


def corners(s, half) -> np.ndarray:
    """The eight corners of a box with half sizes ``half`` (world, 8 x 3) from its centre and orientation."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * np.asarray(half, dtype=float)) @ rotation(s.quat_xyzw).T


# -- the litter ---------------------------------------------------------------------------------------------------
def lowest(litter) -> float:
    """Height (z) of the litter's lowest point."""
    return float(corners(litter, LITTER_HALF)[:, 2].min())


def end_heights(litter) -> tuple[float, float]:
    """Heights (z) of the bottom edge of the litter's left (+y) and right (-y) end faces: for each end, the mean of
    the two lower corners of its 40 x 20 mm end face. Whichever end is further to the left in the world is 'left'."""
    r, c = rotation(litter.quat_xyzw), np.asarray(litter.pos, dtype=float)
    ends = []
    for sy in (1.0, -1.0):
        face = np.array([c + r @ (np.array([sx, sy, sz]) * LITTER_HALF) for sx in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        z = np.sort(face[:, 2])
        ends.append((float(z[0] + z[1]) / 2.0, float(face[:, 1].mean())))
    ends.sort(key=lambda e: -e[1])  # the end with the greater world y first
    return ends[0][0], ends[1][0]


def end_diff(litter) -> float:
    a, b = end_heights(litter)
    return abs(a - b)


def over_trolley(p) -> bool:
    """Is a point (x, y, ...) over the trolley's 120 x 360 mm footprint?"""
    return bool(abs(float(p[0]) - L.TROLLEY_X) <= L.TROLLEY_HALF_X
                and abs(float(p[1]) - L.TROLLEY_Y) <= L.TROLLEY_HALF_Y)


def footprint_meets_trolley(litter) -> bool:
    """Any part of the litter's outline from above (its bounding box) is over the trolley."""
    lo, hi = np.asarray(litter.low, dtype=float), np.asarray(litter.high, dtype=float)
    return bool(lo[0] < L.TROLLEY_X + L.TROLLEY_HALF_X and hi[0] > L.TROLLEY_X - L.TROLLEY_HALF_X
                and lo[1] < L.TROLLEY_Y + L.TROLLEY_HALF_Y and hi[1] > L.TROLLEY_Y - L.TROLLEY_HALF_Y)


def surface_under(litter) -> float:
    """Height of the surface the litter is over: the trolley's top when its centre is over the trolley, else the
    table top."""
    return L.TROLLEY_TOP if over_trolley(litter.pos) else L.TOP


def clearance(litter) -> float:
    """Height of the litter's lowest point above the surface under it (m; negative: below it)."""
    return lowest(litter) - surface_under(litter)


def low(litter) -> bool:
    """Low enough to be dragging: its lowest point less than 10 mm above the table top, or, while any part of it is
    over the trolley, less than 10 mm above the trolley's top (so a litter shoved across the trolley's edge counts
    as low whichever surface its centre is over)."""
    z = lowest(litter)
    return bool(z - L.TOP < L.DRAG_BAND or (footprint_meets_trolley(litter) and z - L.TROLLEY_TOP < L.DRAG_BAND))


def resting_on_table(litter) -> bool:
    """The litter's lowest point within 3 mm of the table top (anywhere on the table)."""
    return bool(abs(lowest(litter) - L.TOP) <= L.REST_TOL)


def resting_on_trolley(litter) -> bool:
    """The litter's centre over the trolley and its lowest point within 3 mm of the trolley's top."""
    return bool(over_trolley(litter.pos) and abs(lowest(litter) - L.TROLLEY_TOP) <= L.REST_TOL)


def lifted(litter) -> bool:
    """Lifted clear: its lowest point at least 10 mm above the table top and not resting on the trolley."""
    return bool(lowest(litter) - L.TOP >= L.LIFT_CLEAR and not resting_on_trolley(litter))


def in_the_air(litter) -> bool:
    """Not low: at least 10 mm clear of the table top and, wherever any part of it is over the trolley, of the
    trolley's top as well. Then both grippers must hold it."""
    return not low(litter)


def end_up(litter) -> bool:
    """Either end's bottom edge at least 10 mm above the table top: the litter is being handled (lifted, lifted at
    one end, or on the trolley)."""
    return bool(max(end_heights(litter)) - L.TOP >= L.END_UP)


def centre_off(litter) -> tuple[float, float]:
    """The litter's centre from the trolley's centre, in x and in y (m, signed)."""
    return float(litter.pos[0]) - L.TROLLEY_X, float(litter.pos[1]) - L.TROLLEY_Y


def flat_on_trolley(litter, tol_x: float, tol_y: float) -> bool:
    """Rests flat on the trolley: on it (lowest point within 3 mm of its top, centre over it), its broad face
    tilted less than 3 degrees, its long axis along the trolley's within 10 degrees, and its centre within
    ``tol_x`` and ``tol_y`` of the trolley's centre."""
    dx, dy = centre_off(litter)
    return bool(resting_on_trolley(litter) and face_tilt_deg(litter.quat_xyzw) < L.FLAT_DEG
                and heading_err_deg(litter.quat_xyzw) < L.HEADING_DEG and abs(dx) <= tol_x and abs(dy) <= tol_y)


def on_trolley(litter) -> bool:
    """Rests on the trolley however placed: on it, its broad face tilted less than 10 degrees."""
    return bool(resting_on_trolley(litter) and face_tilt_deg(litter.quat_xyzw) < L.ON_TROLLEY_TILT)


def off_table(s, half) -> bool:
    """The body's lowest point more than 10 mm below the table top: it fell off the table."""
    return bool(corners(s, half)[:, 2].min() < L.TOP - L.OFF_TABLE_DROP)


# -- the casualty --------------------------------------------------------------------------------------------------
def casualty_gap(casualty, litter) -> float:
    """Height of the casualty's underside (at its centre) above the litter's top face (at its centre) (m)."""
    c_bottom = float(casualty.pos[2]) - _up(rotation(casualty.quat_xyzw))[2] * CASUALTY_HALF[2]
    l_top = float(litter.pos[2]) + _up(rotation(litter.quat_xyzw))[2] * LITTER_HALF[2]
    return c_bottom - l_top


def casualty_offset(casualty, litter) -> float:
    """Horizontal distance between the casualty's centre and the litter's centre (m)."""
    return float(np.hypot(casualty.pos[0] - litter.pos[0], casualty.pos[1] - litter.pos[1]))


def casualty_on(casualty, litter) -> bool:
    """On the litter: its underside within 6 mm of the litter's top, its centre within 30 mm of the litter's centre
    horizontally, and tilted less than 10 degrees."""
    return bool(abs(casualty_gap(casualty, litter)) <= L.CASUALTY_GAP
                and casualty_offset(casualty, litter) <= L.CASUALTY_OFF
                and face_tilt_deg(casualty.quat_xyzw) < L.CASUALTY_TILT_ON)


def casualty_tipped(casualty) -> bool:
    return face_tilt_deg(casualty.quat_xyzw) > L.CASUALTY_TIP_DEG


# -- the arms ---------------------------------------------------------------------------------------------------
def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


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


def let_go(arm, lo, hi) -> bool:
    """The gripper is not holding the item in the box (lo, hi): it is open (opening 0.9 or more), or its finger zone
    is more than 10 mm from the item. A gripper closed somewhere else is not holding it."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, lo, hi) > L.LET_GO_GAP)


def _box(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def grippers_holding(arms, litter) -> bool:
    """Both grippers hold the litter: each closed below 0.9 with its finger zone within 10 mm of the litter."""
    lo, hi = _box(litter)
    return not let_go(arms["left"], lo, hi) and not let_go(arms["right"], lo, hi)


def grippers_released(arms, litter) -> bool:
    """Both grippers have let go of the litter."""
    lo, hi = _box(litter)
    return let_go(arms["left"], lo, hi) and let_go(arms["right"], lo, hi)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


def _deg(v) -> float | None:
    return round(float(v), 2) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "dragged": "the litter slid more than 15 mm horizontally while within 10 mm of the surface under it",
    "casualty_off": "the casualty was not on the litter (underside more than 6 mm off its top, centre more than "
                    "30 mm off, or tilted 10 degrees or more) for 5 control steps in a row (0.25 s)",
    "casualty_tipped": "the casualty tilted more than 15 degrees",
    "litter_tipped": "the litter's long axis tilted more than 10 degrees",
    "lifted_at_one_end": "the ends' heights differed by more than 40 mm while an end was up",
    "one_gripper": "the litter was 10 mm or more clear of the surface under it while a gripper was not holding it",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.max_end_diff = 0.0  # m: the ends' heights' largest difference while an end was up
        self.max_tilt_up = 0.0  # degrees: the long axis's largest tilt while an end was up
        self.max_tilt = 0.0  # degrees: the long axis's largest tilt at any moment
        self.max_clearance = 0.0  # m: the lowest point's greatest height above the table top
        self.drag = 0.0  # m: the centre's horizontal travel while low (followed in 0.5 mm steps)
        self.drag_mark = None  # (x, y) where drag was last counted, while low
        self.setdown = 0.0  # m/s: the fastest descent in the last 10 mm above the trolley before it came to rest
        self.carried_level = False  # at some step: lowest point 30 mm up, level, ends within 20 mm, casualty on
        self.carried_level_step = None
        self.casualty_off_run = 0  # control steps in a row on which the casualty has not been on the litter
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the litter, of the casualty] at the previous valid control step
        self.prev_clearance = None  # the litter's clearance over the trolley at the previous valid step, or None

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(litter, casualty) -> bool:
        """The starting layout: the litter flat on the table on its mark along y, the casualty on it, both still
        (within START_TOL of the marks, which covers a seeded reset's jitter)."""
        on_mark = (np.hypot(litter.pos[0] - L.LITTER_X, litter.pos[1] - L.LITTER_Y) < L.START_TOL
                   and resting_on_table(litter) and face_tilt_deg(litter.quat_xyzw) < 1.0
                   and heading_err_deg(litter.quat_xyzw) < 2.0)
        return bool(on_mark and casualty_on(casualty, litter) and is_resting(litter) and is_resting(casualty))

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def update(self, litter, casualty, arms=None) -> Result:
        if not (valid(litter) and valid(casualty)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(litter, casualty, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (litter, casualty)]
        if self.prev is not None and self._teleported(now) and self.at_start(litter, casualty):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1

        low_z, clear = lowest(litter), clearance(litter)
        diff, tilt = end_diff(litter), axis_tilt_deg(litter.quat_xyzw)
        on_c = casualty_on(casualty, litter)
        self.max_clearance = max(self.max_clearance, low_z - L.TOP)
        self.max_tilt = max(self.max_tilt, tilt)

        # Dragging: the centre's horizontal travel while the litter is within 10 mm of the surface under it, followed
        # in 0.5 mm steps so that a resting body's contact jitter does not add up.
        xy = now[0][0][:2]
        if low(litter):
            if self.drag_mark is None:
                self.drag_mark = xy
            else:
                d = float(np.linalg.norm(xy - self.drag_mark))
                if d >= L.DRAG_STEP:
                    self.drag += d
                    self.drag_mark = xy
        else:
            self.drag_mark = None
        if self.drag > L.DRAG_MAX:
            self._flag("dragged")

        # Set-down speed onto the trolley: while the litter is over the trolley in the last 10 mm above it, how fast
        # it comes down, as its downward velocity and as how far it came down since the last step (a fast drop can
        # skip the band between two control steps). Within the 3 mm rest band only the step-to-step descent counts,
        # as a 20 Hz video would read it: the instantaneous velocity there is a contact transient (a litter let go
        # 2 mm up reaches 0.2 m/s for a few milliseconds before it lands, which no video frame would show).
        on_tr = resting_on_trolley(litter)
        if over_trolley(litter.pos):
            rate = 0.0 if self.prev_clearance is None else (self.prev_clearance - clear) / self.dt
            if on_tr:
                self.setdown = max(self.setdown, rate)
            elif clear < L.SETDOWN_WINDOW:
                self.setdown = max(self.setdown, rate, -float(litter.lin_vel[2]))
            self.prev_clearance = clear
        else:
            self.prev_clearance = None

        # The casualty, the litter's tilt and its ends, and the grippers while it is in the air. The casualty counts
        # as having left the litter once it has not been on it for CASUALTY_OFF_STEPS steps in a row: a one-step
        # contact hop as the litter is lifted or set down is not it leaving.
        self.casualty_off_run = 0 if on_c else self.casualty_off_run + 1
        if self.casualty_off_run >= L.CASUALTY_OFF_STEPS:
            self._flag("casualty_off")
        if casualty_tipped(casualty):
            self._flag("casualty_tipped")
        if tilt > L.TILT_MAX:
            self._flag("litter_tipped")
        if end_up(litter):
            self.max_end_diff = max(self.max_end_diff, diff)
            self.max_tilt_up = max(self.max_tilt_up, tilt)
            if diff > L.END_DIFF_MAX:
                self._flag("lifted_at_one_end")
        if have_arms and in_the_air(litter) and not grippers_holding(arms, litter):
            self._flag("one_gripper")
        if (not self.carried_level and low_z - L.TOP >= L.CARRY_CLEAR and tilt < L.LEVEL_DEG
                and diff <= L.END_DIFF_FULL and on_c):
            self.carried_level = True
            self.carried_level_step = self.steps

        self.last = self.result(litter, casualty, arms)
        return self.last

    @property
    def handled_evenly(self) -> bool:
        """The ends never differed by more than 20 mm and the litter never tilted more than 5 degrees while an end
        was up, and it was lowered onto the trolley at 0.10 m/s or less, not dropped."""
        return bool(self.max_end_diff <= L.END_DIFF_FULL and self.max_tilt_up <= L.TILT_FULL
                    and self.setdown <= L.SETDOWN_MAX)

    def result(self, litter, casualty, arms=None) -> Result:
        if not (valid(litter) and valid(casualty)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        released = grippers_released(arms, litter) if have_arms else True  # unreadable: taken as met, counted
        on_c = casualty_on(casualty, litter)
        gone = off_table(litter, LITTER_HALF) or off_table(casualty, CASUALTY_HALF)
        tight = flat_on_trolley(litter, L.CENTRE_X_FULL, L.CENTRE_Y_FULL)
        loose = flat_on_trolley(litter, L.CENTRE_LOOSE, L.CENTRE_LOOSE)
        on_tr = on_trolley(litter)
        if self.breach or gone:
            score = 0.0
        elif tight and on_c and released and self.handled_evenly:
            score = 1.0
        elif loose and on_c:
            score = 0.75
        elif on_tr and on_c:
            score = 0.5
        elif self.carried_level:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(litter) and is_resting(casualty))
        left_z, right_z = end_heights(litter)
        dx, dy = centre_off(litter)
        lo, hi = _box(litter)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "surface": "trolley" if over_trolley(litter.pos) else "table",
            "lowest_mm": _mm(lowest(litter) - L.TOP),
            "clearance_mm": _mm(clearance(litter)),
            "end_left_mm": _mm(left_z - L.TOP),
            "end_right_mm": _mm(right_z - L.TOP),
            "end_diff_mm": _mm(abs(left_z - right_z)),
            "tilt_deg": _deg(axis_tilt_deg(litter.quat_xyzw)),
            "face_tilt_deg": _deg(face_tilt_deg(litter.quat_xyzw)),
            "heading_err_deg": _deg(heading_err_deg(litter.quat_xyzw)),
            "resting_on_table": resting_on_table(litter),
            "resting_on_trolley": resting_on_trolley(litter),
            "lifted": lifted(litter),
            "in_the_air": in_the_air(litter),
            "end_up": end_up(litter),
            "low": low(litter),
            "centre_dx_mm": _mm(dx),
            "centre_dy_mm": _mm(dy),
            "flat_on_trolley": tight,
            "flat_on_trolley_loose": loose,
            "on_trolley": on_tr,
            "casualty_on": on_c,
            "casualty_gap_mm": _mm(casualty_gap(casualty, litter)),
            "casualty_off_mm": _mm(casualty_offset(casualty, litter)),
            "casualty_tilt_deg": _deg(face_tilt_deg(casualty.quat_xyzw)),
            "off_table": gone,
            "max_end_diff_mm": _mm(self.max_end_diff),
            "max_tilt_up_deg": _deg(self.max_tilt_up),
            "max_tilt_deg": _deg(self.max_tilt),
            "max_clearance_mm": _mm(self.max_clearance),
            "drag_mm": _mm(self.drag),
            "setdown_mps": round(self.setdown, 3) if math.isfinite(self.setdown) else None,
            "carried_level": self.carried_level,
            "carried_level_step": self.carried_level_step,
            "handled_evenly": self.handled_evenly,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "zone_gap_mm": ({s: _mm(zone_gap(arms[s], lo, hi)) for s in ("left", "right")} if have_arms else None),
            "grippers_holding": grippers_holding(arms, litter) if have_arms else None,
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "litter_resting": is_resting(litter),
            "casualty_resting": is_resting(casualty),
        }
        return Result(score, success, details)
