"""The scorer, as plain functions of the probe's and the part's states and the arms' state. NumPy only; no simulator.

The rules are about the whole trial, not only its end. Before the right (entrant) gripper may reach into the tank,
the probe held by the left (attendant) gripper must test the tank's air at three heights in order, top, middle,
bottom, holding its tip still in each band for 2 s; while the right gripper is in the tank the probe must stay in
the tank's lower half, held by the left gripper; the left gripper never goes into the tank; the probe is never let
go inside the tank, dropped or knocked over; and the part leaves the tank only in the right gripper. So the scorer
keeps a little state (how far the test has got, which rule has been broken and when) and is fed one observation per
control step by ``task.success()``.

    Tracker.update(probe, part, arms) -> Result(score, success, details)

``probe`` and ``part`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the arm rules are taken as kept
for that step (neither gripper in the tank, the probe and the part held where a rule needs it) and the step is
counted in ``arms_unavailable``. An object state with a non-finite number or a zero quaternion (a simulator blow-up)
scores 0 for that step and leaves the tracker as it was.

The probe's 'tip' is the lower end of its axis (the sensor head), found from its centre and orientation, so a tilted
probe is measured where its end really is. rigstate gives no gripper heading, so a gripper's fingers are taken to be
anywhere round its axis: the 'finger zone' is every point within a radius of the 38 mm segment from the point between
the pad tips back up the gripper; the radius reaches a pad's outer corner (half the current gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). A gripper's 'lowest point' is the point between its pad tips,
lowered by that radius times the sine of the gripper's tilt from straight down (for a gripper pointing straight down,
the pad tips themselves).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
DWELL_STEPS = int(round(L.DWELL_S * L.CONTROL_HZ))  # 40 control steps


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


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


# -- the tank ---------------------------------------------------------------------------------------------------
def over_tank(p, margin: float = 0.0) -> bool:
    """Is the point (x, y) over the tank's outline (outer walls) grown by ``margin``?"""
    return bool(abs(p[0] - L.TANK_X) <= L.TANK_OUT_HX + margin and abs(p[1] - L.TANK_Y) <= L.TANK_OUT_HY + margin)


def in_opening(p) -> bool:
    """Is the point (x, y) over the tank's 200 x 360 mm opening (inside its walls)?"""
    return bool(abs(p[0] - L.TANK_X) <= L.TANK_IN_HX and abs(p[1] - L.TANK_Y) <= L.TANK_IN_HY)


def footprint_off_tank(s) -> bool:
    """No part of the object's outline from above (its bounding box) is over the tank's outline."""
    lo, hi = _lo_hi(s)
    return bool(hi[0] < L.TANK_X - L.TANK_OUT_HX or lo[0] > L.TANK_X + L.TANK_OUT_HX
                or hi[1] < L.TANK_Y - L.TANK_OUT_HY or lo[1] > L.TANK_Y + L.TANK_OUT_HY)


# -- the probe ----------------------------------------------------------------------------------------------------
def probe_axis(probe) -> np.ndarray:
    return rotation(probe.quat_xyzw)[:, 2]


def probe_tip(probe) -> np.ndarray:
    """The lower end of the probe's axis: where its sensor head is (for an upright probe, the centre of its base)."""
    u = probe_axis(probe)
    if u[2] < 0:
        u = -u
    return np.asarray(probe.pos, dtype=float) - u * (L.PROBE_LEN / 2)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the probe's axis and the vertical (degrees, 0..90; the cylinder is the same either way up)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def tip_height(probe) -> float:
    """Height of the probe's tip above the tank floor (the table top)."""
    return float(probe_tip(probe)[2]) - L.TOP


def probe_in_tank(probe) -> bool:
    """The probe's tip is below the rim and over the tank's outline: the probe is in the tank."""
    tip = probe_tip(probe)
    return bool(tip[2] < L.RIM_Z and over_tank(tip))


def band_of(probe) -> int | None:
    """Which test band (0 top, 1 middle, 2 bottom) the tip is in, if it is over the tank's opening; else None."""
    tip = probe_tip(probe)
    if not in_opening(tip):
        return None
    h = float(tip[2]) - L.TOP
    for i, (lo, hi) in enumerate(L.BANDS):
        if lo <= h <= hi:
            return i
    return None


def probe_low(probe) -> bool:
    """The tip is over the tank's opening and below half the tank's depth (40 mm above the floor)."""
    tip = probe_tip(probe)
    return bool(in_opening(tip) and float(tip[2]) < L.TOP + L.LOWER_HALF)


def standing(s) -> bool:
    """Upright on the table: tilted less than 10 degrees, its lowest point within 5 mm of the table top."""
    return bool(tilt_deg(s.quat_xyzw) < L.UPRIGHT_DEG and abs(float(s.low[2]) - L.TOP) <= L.REST_Z_TOL)


def probe_in_holder(probe) -> bool:
    """Standing in its holder: upright on the table, its centre within 12 mm of the holder's centre along x and y."""
    return bool(standing(probe) and abs(probe.pos[0] - L.PROBE_X) <= L.HOLDER_TOL
                and abs(probe.pos[1] - L.PROBE_Y) <= L.HOLDER_TOL)


# -- the part -----------------------------------------------------------------------------------------------------
def part_in_tank(part) -> bool:
    """The part's centre is over the tank's outline and its lowest point below the rim."""
    return bool(over_tank(part.pos) and float(part.low[2]) < L.RIM_Z)


def part_out(part) -> bool:
    """Out of the tank: its lowest point at least 10 mm above the rim, or its outline clear of the tank's."""
    return bool(float(part.low[2]) >= L.RIM_Z + L.OUT_CLEAR or footprint_off_tank(part))


def part_in_tray(part) -> bool:
    """The part's centre over the tray's 80 x 80 mm floor and its lowest point within 5 mm of the table top."""
    half = L.TRAY_INNER / 2
    return bool(abs(part.pos[0] - L.TRAY_X) <= half and abs(part.pos[1] - L.TRAY_Y) <= half
                and float(part.low[2]) <= L.TOP + L.REST_Z_TOL)


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


# -- the arms -------------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def approach_axis(arm) -> np.ndarray:
    """Unit vector from the TCP to the point between the pad tips (straight down for a vertical gripper)."""
    u = np.asarray(arm.tips, dtype=float) - np.asarray(arm.tcp, dtype=float)
    n = float(np.linalg.norm(u))
    return u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])


def finger_low_z(arm) -> float:
    """The fingers' lowest point: the pad tips' height, lowered by the zone radius times the sine of the gripper's
    tilt from straight down (a tilted gripper's pad corners reach below its tip point)."""
    u = approach_axis(arm)
    sin_tilt = math.sqrt(max(0.0, 1.0 - float(u[2]) ** 2)) if u[2] < 0 else 1.0
    return float(arm.tips[2]) - zone_radius(arm) * sin_tilt


def gripper_in_tank(arm) -> bool:
    """The gripper's fingers reach below the rim with the point between its pad tips within 10 mm of the tank's
    outline (seen from above)."""
    return bool(finger_low_z(arm) < L.RIM_Z and over_tank(arm.tips, L.IN_TANK_MARGIN))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    tips = np.asarray(arm.tips, dtype=float)
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, approach_axis(arm)), zone_radius(arm)


def zone_gap(arm, lo, hi) -> float:
    """How far a gripper's finger zone is from a world-aligned box (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_box_distance(p, lo, hi) for p in pts) - r


def holds(arm, s) -> bool:
    """The gripper holds the object: its fingers closed to 47 mm or less (opening <= 0.5) with the point between its
    pad tips inside the object's bounding box grown by 5 mm (so the object is between the fingers)."""
    lo, hi = _lo_hi(s)
    return bool(float(arm.opening) <= L.HOLD_OPENING and point_box_distance(arm.tips, lo, hi) <= L.HOLD_SLACK)


def let_go(arm, s) -> bool:
    """The gripper is not holding the object: open (opening 0.9 or more), or its finger zone more than 10 mm away."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, *_lo_hi(s)) > L.LET_GO_GAP)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "right_entered_unprotected": "the right gripper was in the tank without a completed top-middle-bottom test, or "
                                 "without the probe held low in the tank by the left gripper",
    "left_in_tank": "the left (attendant) gripper went into the tank",
    "probe_let_go_in_tank": "the probe was in the tank without the left gripper holding it",
    "probe_dropped": "the probe was out of the tank, held by neither gripper, with its lowest point more than 10 mm "
                     "above the table",
    "probe_toppled": "the probe tilted more than 45 degrees",
    "part_out_unheld": "the part came up out of the tank without the right gripper holding it",
    "off_table": "the probe or the part went off the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.stage = 0  # bands completed in the current test: 0..3 (3: the test is complete and still valid)
        self.window: deque = deque(maxlen=DWELL_STEPS)  # the tip's positions over the current run in the next band
        self.band_steps: list[int | None] = [None, None, None]  # control step each band was completed (last test)
        self.tests_completed = 0  # complete top-middle-bottom tests so far
        self.tests_voided = 0  # complete tests voided by taking the probe out of the tank or letting it go
        self.entered = False  # the right gripper has been in the tank
        self.first_entry_step: int | None = None
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.part_was_in: bool | None = None
        self.min_left_clear = math.inf  # m: the left fingers' lowest point above the rim, over the tank
        self.max_tip_in_entry = -math.inf  # m: the tip's highest point above the floor while the right was in
        self.arms_unavailable = 0
        self.bad_steps = 0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the probe, of the part] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @property
    def test_valid(self) -> bool:
        return self.stage == 3

    @staticmethod
    def at_start(probe, part) -> bool:
        """The starting layout: the probe upright in its holder, the part flat on its mark in the tank, both still."""
        part_ok = (np.hypot(part.pos[0] - L.PART_X, part.pos[1] - L.PART_Y) < L.PART_START_TOL
                   and abs(float(part.low[2]) - L.TOP) < 0.003)
        return bool(probe_in_holder(probe) and part_ok and is_resting(probe) and is_resting(part))

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

    def _advance_test(self, probe, left_holds: bool) -> None:
        """One step of the ordered test: a band counts once the tip has stayed in it, over the opening, with the
        probe held by the left gripper, for 40 consecutive steps inside a 10 mm box. A completed test stays valid
        while the probe stays in the tank, held by the left gripper; else a new full test is needed."""
        if self.stage == 3:
            if not (left_holds and probe_in_tank(probe)):
                self.stage = 0
                self.tests_voided += 1
                self.window.clear()
            return
        if left_holds and band_of(probe) == self.stage:
            self.window.append(probe_tip(probe))
            if len(self.window) == DWELL_STEPS:
                pts = np.array(self.window)
                if float(np.max(np.ptp(pts, axis=0))) <= L.DWELL_BOX:
                    self.band_steps[self.stage] = self.steps
                    self.stage += 1
                    self.window.clear()
                    if self.stage == 3:
                        self.tests_completed += 1
        else:
            self.window.clear()

    def update(self, probe, part, arms=None) -> Result:
        if not (valid(probe) and valid(part)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(probe, part, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (probe, part)]
        if self.prev is not None and self._teleported(now) and self.at_start(probe, part):
            self.reset()  # the simulator put everything back: a new episode (rigstate's on_reset hook normally
            #               does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        have = arms_valid(arms)
        if not have:
            self.arms_unavailable += 1
        left_holds = holds(arms["left"], probe) if have else True
        right_holds_probe = holds(arms["right"], probe) if have else False
        right_holds_part = holds(arms["right"], part) if have else True

        self._advance_test(probe, left_holds)

        if have:
            left, right = arms["left"], arms["right"]
            if over_tank(left.tips, L.IN_TANK_MARGIN):
                self.min_left_clear = min(self.min_left_clear, finger_low_z(left) - L.RIM_Z)
            if gripper_in_tank(left):
                self._flag("left_in_tank")
            if gripper_in_tank(right):
                if not self.entered:
                    self.entered, self.first_entry_step = True, self.steps
                self.max_tip_in_entry = max(self.max_tip_in_entry, tip_height(probe))
                if not (self.test_valid and left_holds and probe_low(probe)):
                    self._flag("right_entered_unprotected")

        inside = probe_in_tank(probe)
        if inside and not left_holds:
            self._flag("probe_let_go_in_tank")
        if (not inside and not (left_holds or right_holds_probe)
                and float(probe.low[2]) > L.TOP + L.DROP_HEIGHT):
            self._flag("probe_dropped")
        if tilt_deg(probe.quat_xyzw) > L.TOPPLE_DEG:
            self._flag("probe_toppled")
        if off_table(probe) or off_table(part):
            self._flag("off_table")

        part_in = part_in_tank(part)
        if self.part_was_in and not part_in and not right_holds_part:
            self._flag("part_out_unheld")
        self.part_was_in = part_in

        self.last = self.result(probe, part, arms)
        return self.last

    def result(self, probe, part, arms=None) -> Result:
        if not (valid(probe) and valid(part)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have = arms_valid(arms)
        released = (let_go(arms["left"], probe) and let_go(arms["right"], part)) if have else True
        left_holds = holds(arms["left"], probe) if have else True
        in_tray = part_in_tray(part)
        in_holder = probe_in_holder(probe)
        probe_out_safe = (not probe_in_tank(probe)) and (left_holds or (standing(probe) and footprint_off_tank(probe)))
        if self.breach:
            score = 0.0
        elif in_tray and in_holder and released:
            score = 1.0
        elif in_tray and probe_out_safe:
            score = 0.75
        elif part_out(part):
            score = 0.5
        elif self.tests_completed >= 1:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(probe) and is_resting(part))
        band = band_of(probe)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "test_stage": self.stage,
            "band_done_steps": list(self.band_steps),
            "tests_completed": self.tests_completed,
            "tests_voided": self.tests_voided,
            "tip_band": L.BAND_NAMES[band] if band is not None else None,
            "tip_height_mm": _mm(tip_height(probe)),
            "probe_tilt_deg": round(tilt_deg(probe.quat_xyzw), 2),
            "probe_in_tank": probe_in_tank(probe),
            "probe_low": probe_low(probe),
            "probe_in_holder": in_holder,
            "probe_out_safe": probe_out_safe,
            "left_holds_probe": left_holds if have else None,
            "right_entered": self.entered,
            "first_entry_step": self.first_entry_step,
            "max_tip_during_entry_mm": _mm(self.max_tip_in_entry) if self.entered else None,
            "min_left_above_rim_mm": _mm(self.min_left_clear),
            "part_in_tank": part_in_tank(part),
            "part_out": part_out(part),
            "part_in_tray": in_tray,
            "part_low_mm": _mm(float(part.low[2]) - L.TOP),
            "grippers_let_go": released if have else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have else None),
            "arms_available": have,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "probe_resting": is_resting(probe),
            "part_resting": is_resting(part),
        }
        return Result(score, success, details)
