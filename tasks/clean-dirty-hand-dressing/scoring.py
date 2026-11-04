"""The scorer, as plain functions of the two pads' states and the arms' state. NumPy only; no simulator needed.

The rule is about the whole trial, not only its end: the right (dirty) gripper must never come near the clean pad,
the left (clean) gripper must never touch the used dressing, the clean pad must never meet the used dressing, touch
the bin or the bare table, and it may come over the wound mark only once the used dressing is in the bin. So the
scorer keeps a little state (which of those has happened, and when) and is fed one observation per control step by
``task.success()``.

    Tracker.update(dressing, pad, arms) -> Result(score, success, details)

``dressing`` and ``pad`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the arm rules are taken as kept
for that step and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn (and a
little more), so 'the fingers come within d of a box' never misses a real approach.

Both pads are plain boxes with two identical 50 x 50 mm faces, so either may face down: 'tilt' is measured from
whichever is nearer to facing down.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
BIN_LO = np.array([L.BIN_X - L.BIN_OUTER_HALF, L.BIN_Y - L.BIN_OUTER_HALF, L.TOP])
BIN_HI = np.array([L.BIN_X + L.BIN_OUTER_HALF, L.BIN_Y + L.BIN_OUTER_HALF, L.BIN_TOP_Z])
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box


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
    """Angle between the world's vertical and the axis through the pad's two 50 x 50 mm faces (degrees, 0..90):
    the pad is the same either way up, so whichever face is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- boxes and footprints -------------------------------------------------------------------------------------
def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def box_gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Distance between two world-aligned boxes (0 when they touch or overlap)."""
    d = np.maximum(0.0, np.maximum(np.asarray(lo_b) - np.asarray(hi_a), np.asarray(lo_a) - np.asarray(hi_b)))
    return float(np.linalg.norm(d))


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


def _overlap_1d(a0, a1, b0, b1) -> bool:
    return bool(a0 < b1 and b0 < a1)


def footprints_overlap(lo_a, hi_a, lo_b, hi_b) -> bool:
    """Do two boxes overlap seen from above (whatever their heights)?"""
    return _overlap_1d(lo_a[0], hi_a[0], lo_b[0], hi_b[0]) and _overlap_1d(lo_a[1], hi_a[1], lo_b[1], hi_b[1])


MARK_FOOT = (np.array([L.MARK_X - L.MARK_HALF, L.MARK_Y - L.MARK_HALF]),
             np.array([L.MARK_X + L.MARK_HALF, L.MARK_Y + L.MARK_HALF]))


def corners(s, size=L.PAD_SIZE) -> np.ndarray:
    """The eight corners of a box of ``size`` (world, 8 x 3) from its centre and orientation."""
    half = np.array(size, dtype=float) / 2
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def over_mark_xy(p) -> bool:
    return bool(abs(p[0] - L.MARK_X) <= L.MARK_HALF and abs(p[1] - L.MARK_Y) <= L.MARK_HALF)


def over_tray_floor_xy(p, slack: float = L.TRAY_FLOOR_SLACK) -> bool:
    return bool(abs(p[0] - L.TRAY_X) <= L.TRAY_INNER_X / 2 + slack and abs(p[1] - L.TRAY_Y) <= L.TRAY_INNER_Y / 2 + slack)


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


def grippers_open(arms) -> bool:
    return bool(all(float(arms[s].opening) >= L.GRIPPER_OPEN for s in ("left", "right")))


def let_go(arm, lo, hi) -> bool:
    """The gripper is not holding the item in the box (lo, hi): it is open (opening 0.9 or more), or its finger zone
    is more than 10 mm from the item. A gripper closed somewhere else is not holding it."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, lo, hi) > L.LET_GO_GAP)


def grippers_released(arms, dressing, pad) -> bool:
    """The left gripper has let go of the clean pad and the right gripper of the used dressing."""
    return let_go(arms["left"], *_lo_hi(pad)) and let_go(arms["right"], *_lo_hi(dressing))


# -- the objects -------------------------------------------------------------------------------------------------
def centre_off(pad) -> float:
    """Horizontal distance from the clean pad's centre to the wound mark's centre (m)."""
    return float(np.hypot(pad.pos[0] - L.MARK_X, pad.pos[1] - L.MARK_Y))


def pad_on_mark(pad, tol: float) -> bool:
    """Flat on the wound mark: tilted under 5 degrees, its lowest point within 3 mm of the mark's top, its centre
    within ``tol`` of the mark's centre."""
    return bool(tilt_deg(pad.quat_xyzw) < L.FLAT_DEG and abs(float(pad.low[2]) - L.MARK_TOP) <= L.ON_MARK_Z_TOL
                and centre_off(pad) <= tol)


def pad_over_mark(pad) -> bool:
    """Any part of the clean pad's outline from above (its bounding box) over the wound mark, at any height."""
    return footprints_overlap(*_lo_hi(pad), *MARK_FOOT)


def pad_in_tray(pad) -> bool:
    """The clean pad's centre over the tray's 140 x 70 mm floor, its lowest point within 5 mm of the table top."""
    return bool(over_tray_floor_xy(pad.pos, 0.0) and float(pad.low[2]) < L.TOP + 0.005)


def lowest_bare_corner(pad) -> float | None:
    """Height above the table top of the clean pad's lowest corner that is over neither the tray's floor nor the
    wound mark (None if every corner is over one of them)."""
    out = [float(c[2]) - L.TOP for c in corners(pad) if not (over_tray_floor_xy(c) or over_mark_xy(c))]
    return min(out) if out else None


def pad_on_bare_table(pad) -> bool:
    """A corner of the clean pad within 1 mm of the table top, or below it (off the table), where it is over neither
    the tray's floor nor the wound mark: the pad touches the bare table (or the floor)."""
    h = lowest_bare_corner(pad)
    return h is not None and h < L.BARE_TABLE_TOL


def pad_at_bin(pad) -> bool:
    """The clean pad's box within 3 mm of the bin's (inside it, on its walls, or against them)."""
    return box_gap(*_lo_hi(pad), BIN_LO, BIN_HI) < L.PAD_BIN_GAP


def pads_meet(dressing, pad) -> bool:
    """The clean pad and the used dressing touch (boxes within 3 mm), or one passes over the other (their outlines
    from above overlap, at any height)."""
    a, b = _lo_hi(dressing), _lo_hi(pad)
    return bool(box_gap(*a, *b) < L.PAD_DRESSING_GAP or footprints_overlap(*a, *b))


def dressing_in_bin(dressing) -> bool:
    """The used dressing's centre inside the bin: over its 90 x 90 mm floor and below its rim."""
    return bool(abs(dressing.pos[0] - L.BIN_X) <= L.BIN_INNER_HALF and abs(dressing.pos[1] - L.BIN_Y) <= L.BIN_INNER_HALF
                and float(dressing.pos[2]) < L.BIN_TOP_Z)


def dressing_off_wound(dressing) -> bool:
    """No part of the used dressing's outline from above is over the wound mark."""
    return not footprints_overlap(*_lo_hi(dressing), *MARK_FOOT)


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "left_touched_dressing": "the left (clean) gripper's fingers came within 5 mm of the used dressing",
    "right_near_pad": "the right (dirty) gripper's fingers came within 30 mm of the clean pad",
    "pad_met_dressing": "the clean pad touched the used dressing or passed over or under it",
    "pad_before_bin": "the clean pad came over the wound mark before the used dressing was in the bin",
    "pad_on_bare_table": "the clean pad touched the table away from its tray and the wound mark",
    "pad_at_bin": "the clean pad touched the bin",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.min_left_dressing = math.inf  # m: the left finger zone's closest approach to the used dressing
        self.min_right_pad = math.inf  # m: the right finger zone's closest approach to the clean pad
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the dressing, of the pad] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(dressing, pad) -> bool:
        """The starting layout: the used dressing flat on the wound mark, the clean pad flat in its tray, both
        still (within START_TOL of their marks, which covers a seeded reset's jitter)."""
        d_ok = (np.hypot(dressing.pos[0] - L.DRESSING_X, dressing.pos[1] - L.DRESSING_Y) < L.START_TOL
                and abs(float(dressing.low[2]) - L.MARK_TOP) < 0.003 and tilt_deg(dressing.quat_xyzw) < 1.0)
        p_ok = (np.hypot(pad.pos[0] - L.TRAY_X, pad.pos[1] - L.TRAY_Y) < L.START_TOL
                and abs(float(pad.low[2]) - L.TOP) < 0.003 and tilt_deg(pad.quat_xyzw) < 1.0)
        return bool(d_ok and p_ok and is_resting(dressing) and is_resting(pad))

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

    def update(self, dressing, pad, arms=None) -> Result:
        if not (valid(dressing) and valid(pad)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(dressing, pad, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (dressing, pad)]
        if self.prev is not None and self._teleported(now) and self.at_start(dressing, pad):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        if arms_valid(arms):
            left = zone_gap(arms["left"], *_lo_hi(dressing))
            right = zone_gap(arms["right"], *_lo_hi(pad))
            self.min_left_dressing = min(self.min_left_dressing, left)
            self.min_right_pad = min(self.min_right_pad, right)
            if left < L.TOUCH_GAP:
                self._flag("left_touched_dressing")
            if right < L.KEEP_OUT:
                self._flag("right_near_pad")
        else:
            self.arms_unavailable += 1
        if pads_meet(dressing, pad):
            self._flag("pad_met_dressing")
        if pad_over_mark(pad) and not dressing_in_bin(dressing):
            self._flag("pad_before_bin")
        if pad_on_bare_table(pad):
            self._flag("pad_on_bare_table")
        if pad_at_bin(pad):
            self._flag("pad_at_bin")
        self.last = self.result(dressing, pad, arms)
        return self.last

    def result(self, dressing, pad, arms=None) -> Result:
        if not (valid(dressing) and valid(pad)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        released = grippers_released(arms, dressing, pad) if have_arms else True  # unreadable: taken as met, counted
        in_bin = dressing_in_bin(dressing)
        gone = off_table(dressing)
        if self.breach or gone:
            score = 0.0
        elif in_bin and pad_on_mark(pad, L.CENTRE_TIGHT) and released:
            score = 1.0
        elif in_bin and pad_on_mark(pad, L.CENTRE_LOOSE):
            score = 0.75
        elif in_bin:
            score = 0.5
        elif dressing_off_wound(dressing):
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(dressing) and is_resting(pad))
        left = zone_gap(arms["left"], *_lo_hi(dressing)) if have_arms else None
        right = zone_gap(arms["right"], *_lo_hi(pad)) if have_arms else None
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "dressing_in_bin": in_bin,
            "dressing_off_wound": dressing_off_wound(dressing),
            "dressing_off_table": gone,
            "dressing_low_mm": _mm(float(dressing.low[2]) - L.TOP),
            "pad_on_mark": pad_on_mark(pad, L.CENTRE_LOOSE),
            "pad_over_mark": pad_over_mark(pad),
            "pad_in_tray": pad_in_tray(pad),
            "pad_centre_off_mm": _mm(centre_off(pad)),
            "pad_tilt_deg": round(tilt_deg(pad.quat_xyzw), 2),
            "pad_low_mm": _mm(float(pad.low[2]) - L.TOP),
            "pad_bare_corner_mm": _mm(lowest_bare_corner(pad)),
            "pad_dressing_gap_mm": _mm(box_gap(*_lo_hi(dressing), *_lo_hi(pad))),
            "left_to_dressing_mm": _mm(left),
            "right_to_pad_mm": _mm(right),
            "min_left_to_dressing_mm": _mm(self.min_left_dressing),
            "min_right_to_pad_mm": _mm(self.min_right_pad),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "grippers_open": grippers_open(arms) if have_arms else None,
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "dressing_resting": is_resting(dressing),
            "pad_resting": is_resting(pad),
        }
        return Result(score, success, details)
