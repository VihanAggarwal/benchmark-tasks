"""The scorer, as plain functions of the beam's state and the arms' state. NumPy only; no simulator needed.

The rule is about the whole move, not only where the beam ends: the beam starts out of square, must be squared
first, and must never be let rack (one end leading the other by more than 12 mm) on its way to the marks. So the
scorer keeps a little state (when the beam was first square and how far it had moved by then, the largest lead
before and after that, how far it moved without a gripper at each end, and whether it ever racked, popped up or
left the rails) and is fed one observation per control step by ``task.success()``.

    Tracker.update(beam, arms) -> Result(score, success, details)

``beam`` is anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has; the beam is a box of ``layout.BEAM_SIZE``, so its ends and corners follow
from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``,
``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could
not be read; then the hands-off and both-ends conditions are taken as met for that step and the step is counted in
``arms_unavailable``. A beam state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0
for that step and leaves the tracker as it was.

'Lead' is how much farther from the arms (along x) one end's centre is than the other's: 520 mm times the sine of
the beam's turn about the vertical while it lies flat. The beam is a plain box, the same either way up, so its
tilt is measured from whichever broad face is nearer to facing down.

The move must be made 'by both ends together', as the instruction says: a gripper is at an end when its finger-tip
point is within 10 mm of the beam and at least 130 mm from its middle along its length, toward that end; on steps
without a gripper at each end the centre's movement toward the marks is added up (signed, so resting jitter
cancels), and more than 20 mm in all is 0. Tilted 10 degrees either arm reaches the beam's middle, where a pull
turns nothing, so without this rule one arm could do the whole move.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the
8 mm pad thickness, and half the pad's 27 mm width across; 57 mm fully open).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_HALF = np.array(L.BEAM_HALF, dtype=float)
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


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- the beam --------------------------------------------------------------------------------------------------
def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through the beam's two broad (40 x 520 mm) faces (degrees,
    0..90): the beam is the same either way up, so whichever face is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def end_centres(beam) -> tuple[np.ndarray, np.ndarray]:
    """The centres of the beam's two end faces (world), left (larger y) first."""
    c = np.asarray(beam.pos, dtype=float)
    u = rotation(beam.quat_xyzw)[:, 1] * L.END_Y
    a, b = c + u, c - u
    return (a, b) if a[1] >= b[1] else (b, a)


def signed_lead(beam) -> float:
    """The right end's centre minus the left end's along x (m): positive while the right end lags farther from
    the arms, as at the start (+8 mm)."""
    left, right = end_centres(beam)
    return float(right[0] - left[0])


def lead(beam) -> float:
    """How much farther from the arms one end's centre is than the other's (m)."""
    return abs(signed_lead(beam))


def corners(beam) -> np.ndarray:
    """The beam's eight corners (world, 8 x 3)."""
    return np.asarray(beam.pos, dtype=float) + (_SIGNS * _HALF) @ rotation(beam.quat_xyzw).T


def rise(beam) -> float:
    """Height of the beam's lowest point above the table top (m)."""
    return float(corners(beam)[:, 2].min() - L.TOP)


def popped(beam) -> bool:
    """Popped up: the lowest point more than 5 mm above the table (the lips allow 4 mm), or tilted over 5 deg."""
    return bool(rise(beam) > L.POP_RISE or tilt_deg(beam.quat_xyzw) > L.POP_TILT_DEG)


def off_rails(beam) -> bool:
    """Out of the rails: an end's centre past either end of the rails (x 0.170 .. 0.500), or the beam's centre
    more than 10 mm off the midline sideways."""
    return bool(any(not (L.RAIL_X0 <= float(e[0]) <= L.RAIL_X1) for e in end_centres(beam))
                or abs(float(beam.pos[1])) > L.RAIL_SIDE_TOL)


def end_offsets(beam) -> tuple[float, float]:
    """How far each end's centre is from the marks' line x 0.220 along x (m; left, right; + = beyond it)."""
    left, right = end_centres(beam)
    return float(left[0] - L.TARGET_X), float(right[0] - L.TARGET_X)


def ends_within(beam, tol: float) -> bool:
    return bool(all(abs(d) <= tol for d in end_offsets(beam)))


def centre_off(beam) -> float:
    """How far the beam's centre is from the marks' line along x (m)."""
    return float(abs(float(beam.pos[0]) - L.TARGET_X))


def travel(beam) -> float:
    """Share of the move done: 0 at the start (centre x 0.444), 1 with the centre on the marks' line (x 0.220)."""
    return float((L.BEAM_X0 - float(beam.pos[0])) / L.TRAVEL_SPAN)


def point_beam_distance(p, beam) -> float:
    """Distance from a point to the beam (0 inside it), in the beam's own frame."""
    r = rotation(beam.quat_xyzw)
    d = r.T @ (np.asarray(p, dtype=float) - np.asarray(beam.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(d) - _HALF, 0.0)))


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


def zone_gap(arm, beam) -> float:
    """How far a gripper's finger zone is from the beam (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_beam_distance(p, beam) for p in pts) - r


def hands_off(arms, beam) -> bool:
    """Both grippers' finger zones at least 30 mm from the beam: neither can be holding or touching it."""
    return bool(all(zone_gap(arms[s], beam) >= L.HANDS_OFF for s in ("left", "right")))


def end_of(arm, beam) -> int:
    """Which end of the beam a gripper is at: +1 the left end, -1 the right end, 0 neither. At an end: its
    finger-tip point within 10 mm of the beam and at least 130 mm from the beam's middle along its length, toward
    that end (the demo grips 200 mm out)."""
    tips = np.asarray(arm.tips, dtype=float)
    if point_beam_distance(tips, beam) > L.AT_BEAM:
        return 0
    left, right = end_centres(beam)
    u = (left - right) / np.linalg.norm(left - right)
    s = float(np.dot(tips - np.asarray(beam.pos, dtype=float), u))
    return 1 if s >= L.END_ZONE else -1 if s <= -L.END_ZONE else 0


def ends_held(arms, beam) -> bool:
    """A gripper at each end of the beam (the instruction's 'by both ends together')."""
    return {end_of(arms[s], beam) for s in ("left", "right")} == {1, -1}


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "racked": "one end led the other by more than 12 mm",
    "popped": "the beam's lowest point rose more than 5 mm off the table or it tilted more than 5 degrees",
    "off_rails": "an end went past the rails' ends or the beam left the midline by more than 10 mm",
    "unheld": "the beam's centre came more than 20 mm toward the marks without a gripper at each end",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.squared_step: int | None = None  # the first step with the lead at or under 2 mm
        self.squared_travel = math.nan  # m the beam's centre had moved toward the marks by then
        self.max_lead = 0.0  # m, over the whole trial
        self.max_lead_after = 0.0  # m, from the squared step on
        self.min_zone = {"left": math.inf, "right": math.inf}  # m: each finger zone's closest approach to the beam
        self.unheld_travel = 0.0  # m the centre came toward the marks (net) on steps without a gripper at each end
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite beam state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (centre, velocity) of the beam at the previous valid control step

    @property
    def squared(self) -> bool:
        return self.squared_step is not None

    @property
    def squared_first(self) -> bool:
        """Squared before the beam's centre had moved 25 mm toward the marks (the lagging end read and moved
        first, not the beam dragged out of square and straightened at the end)."""
        return bool(self.squared and self.squared_travel <= L.SQUARE_BY)

    @staticmethod
    def at_start(beam) -> bool:
        """The starting layout: the beam flat on the table near its starting place, still."""
        near = math.hypot(float(beam.pos[0]) - L.BEAM_X0, float(beam.pos[1])) < L.START_TOL
        return bool(near and abs(rise(beam)) < 0.003 and tilt_deg(beam.quat_xyzw) < 1.0 and is_resting(beam))

    def _teleported(self, c) -> bool:
        """The beam moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        p0, v0 = self.prev
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def update(self, beam, arms=None) -> Result:
        if not valid(beam):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(beam, arms)
            return self.last
        c = np.array(beam.pos, dtype=float)
        if self.prev is not None and self._teleported(c) and self.at_start(beam):
            self.reset()  # the simulator put the beam back at the start: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        have_arms = arms_valid(arms)
        if self.prev is not None and have_arms and not ends_held(arms, beam):
            # signed, so resting jitter cancels; unreadable arms are taken as holding, as for hands-off
            self.unheld_travel += float(self.prev[0][0] - c[0])
        self.prev = (c, np.array(beam.lin_vel, dtype=float))
        self.steps += 1

        ld = lead(beam)
        self.max_lead = max(self.max_lead, ld)
        if self.squared_step is None and ld <= L.SQUARE_LEAD:
            self.squared_step = self.steps
            self.squared_travel = float(L.BEAM_X0 - c[0])
        if self.squared:
            self.max_lead_after = max(self.max_lead_after, ld)
        if ld > L.LEAD_MAX:
            self._flag("racked")
        if popped(beam):
            self._flag("popped")
        if off_rails(beam):
            self._flag("off_rails")
        if self.unheld_travel > L.UNHELD_MAX:
            self._flag("unheld")
        if have_arms:
            for side in ("left", "right"):
                self.min_zone[side] = min(self.min_zone[side], zone_gap(arms[side], beam))
        else:
            self.arms_unavailable += 1
        self.last = self.result(beam, arms)
        return self.last

    def result(self, beam, arms=None) -> Result:
        if not valid(beam):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches), "squared": self.squared})
        have_arms = arms_valid(arms)
        off = hands_off(arms, beam) if have_arms else True  # unreadable: taken as met, counted
        ld = lead(beam)
        broken = bool(self.breaches) or ld > L.LEAD_MAX or popped(beam) or off_rails(beam)
        if broken or not self.squared:
            score = 0.0
        elif (self.squared_first and ends_within(beam, L.END_TIGHT) and ld <= L.FINAL_LEAD
              and self.max_lead_after <= L.AFTER_SQUARE_TIGHT and off):
            score = 1.0
        elif self.squared_first and ends_within(beam, L.END_LOOSE) and self.max_lead_after <= L.AFTER_SQUARE_LOOSE:
            score = 0.75
        elif centre_off(beam) <= L.CENTRE_LOOSE:
            score = 0.5
        elif travel(beam) >= L.TRAVEL_PART:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(beam))
        left_off, right_off = end_offsets(beam)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "lead_mm": _mm(signed_lead(beam)),
            "max_lead_mm": _mm(self.max_lead),
            "squared": self.squared,
            "squared_step": self.squared_step,
            "squared_travel_mm": _mm(self.squared_travel) if self.squared else None,
            "max_lead_after_square_mm": _mm(self.max_lead_after) if self.squared else None,
            "left_end_off_mm": _mm(left_off),
            "right_end_off_mm": _mm(right_off),
            "centre_off_mm": _mm(centre_off(beam)),
            "travel": round(travel(beam), 3),
            "side_mm": _mm(float(beam.pos[1])),
            "rise_mm": _mm(rise(beam)),
            "tilt_deg": round(tilt_deg(beam.quat_xyzw), 2),
            "left_zone_mm": _mm(zone_gap(arms["left"], beam)) if have_arms else None,
            "right_zone_mm": _mm(zone_gap(arms["right"], beam)) if have_arms else None,
            "min_left_zone_mm": _mm(self.min_zone["left"]),
            "min_right_zone_mm": _mm(self.min_zone["right"]),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "hands_off": off if have_arms else None,
            "ends_held": ends_held(arms, beam) if have_arms else None,
            "unheld_travel_mm": _mm(self.unheld_travel),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "resting": is_resting(beam),
        }
        return Result(score, success, details)
