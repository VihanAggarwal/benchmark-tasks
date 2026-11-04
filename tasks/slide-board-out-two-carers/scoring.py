"""The scorer, as plain functions of the patient's and the board's states and the arms' state. NumPy only; no
simulator needed.

The rule is about the whole trial, not only its end: the patient (a block lying on the slide board) may never be
dragged, pushed or carried more than 15 mm from where it lies, lifted more than 30 mm, tilted more than 10 degrees
or let down faster than 0.15 m/s, and neither it nor the board may leave the table. So the scorer keeps a little
state (the worst of each of those so far, and when each rule was first broken) and is fed one observation per
control step by ``task.success()``.

    Tracker.update(patient, board, arms) -> Result(score, success, details)

``patient`` and ``board`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then 'both grippers let go' is taken
as met for that step and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a
zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

How far the board is out from under the patient is read from the two outlines seen from above (world-aligned
bounding boxes): the share of the board's starting footprint under the patient (40 x 60 mm) that is still under it.
The board's +y end starts 50 mm past the patient, so a straight pull along the board of 80 mm leaves half of that
footprint under, and one of 110 mm leaves none; 120 mm puts the outlines the 10 mm apart that 'fully out' needs.
'Moved' and 'pulled' are measured from where the patient and the board lay at the episode's first step.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_PH = L.PATIENT_SIZE[2] / 2  # half the patient's height
_BH = L.BOARD_SIZE[2] / 2  # half the board's thickness
START_UNDER_AREA = L.BOARD_SIZE[0] * L.PATIENT_SIZE[1]  # the board's footprint under the patient at the start (m2)
_EPS = 1e-6  # so that exactly half out, or exactly 20 mm of pull, counts


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
    """Angle between the body's z axis and the world's vertical (degrees, 0..180). The patient has a top and a
    bottom, so turning it over is a full tilt."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def board_tilt_deg(quat_xyzw) -> float:
    """The board's tilt, from whichever broad face is nearer to facing down (degrees, 0..90): a board turned over
    is still a flat board."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- boxes and footprints -------------------------------------------------------------------------------------
def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


def _overlap_len(a0, a1, b0, b1) -> float:
    return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))


def footprint_overlap(lo_a, hi_a, lo_b, hi_b) -> tuple[float, float]:
    """Overlap of two outlines seen from above, along x and along y (m, 0 when apart)."""
    return _overlap_len(lo_a[0], hi_a[0], lo_b[0], hi_b[0]), _overlap_len(lo_a[1], hi_a[1], lo_b[1], hi_b[1])


def footprint_gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Distance between two outlines seen from above (0 when they overlap or touch)."""
    d = np.maximum(0.0, np.maximum(np.asarray(lo_b[:2]) - np.asarray(hi_a[:2]),
                                   np.asarray(lo_a[:2]) - np.asarray(hi_b[:2])))
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


def let_go(arm, lo, hi) -> bool:
    """The gripper is not holding the item in the box (lo, hi): it is open (opening 0.9 or more), or its finger zone
    is more than 10 mm from the item. A gripper closed somewhere else is not holding it."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, lo, hi) > L.LET_GO_GAP)


def grippers_released(arms, patient, board) -> bool:
    """The left gripper has let go of the patient and the right gripper of the board."""
    return let_go(arms["left"], *_lo_hi(patient)) and let_go(arms["right"], *_lo_hi(board))


# -- the patient -------------------------------------------------------------------------------------------------
def moved(patient, start_xy=(L.PATIENT_X, L.PATIENT_Y)) -> float:
    """Horizontal distance of the patient's centre from where it lay (m)."""
    return float(np.hypot(patient.pos[0] - start_xy[0], patient.pos[1] - start_xy[1]))


def underside_z(patient) -> float:
    """Height of the centre of the patient's underside (z)."""
    return float(patient.pos[2] - rotation(patient.quat_xyzw)[2, 2] * _PH)


def raised(patient) -> float:
    """How far the centre of the patient's underside is above where it started, the board's top (m; negative once
    it rests on the deck, 20 mm lower)."""
    return underside_z(patient) - L.PATIENT_BOTTOM_Z


def patient_flat_on_deck(patient) -> bool:
    """Resting flat on the deck: tilted under 5 degrees, its lowest point within 3 mm of the deck's top, its centre
    over the deck."""
    return bool(tilt_deg(patient.quat_xyzw) < L.PATIENT_FLAT_DEG
                and abs(float(patient.low[2]) - L.DECK_TOP) <= L.ON_DECK_Z_TOL and over_deck(patient.pos))


def over_deck(p) -> bool:
    """A point (x, y) over the deck's footprint."""
    return bool(abs(float(p[0]) - L.DECK_X) <= L.DECK_HALF_X and abs(float(p[1]) - L.DECK_Y) <= L.DECK_HALF_Y)


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


# -- the board ----------------------------------------------------------------------------------------------------
def under_share(patient, board) -> float:
    """Share (0..1) of the board's starting footprint under the patient (40 x 60 mm) that is still under it: the
    overlap of the two outlines seen from above, over that starting area."""
    ox, oy = footprint_overlap(*_lo_hi(board), *_lo_hi(patient))
    return float(np.clip(ox * oy / START_UNDER_AREA, 0.0, 1.0))


def board_clear(patient, board) -> float:
    """Distance between the board's and the patient's outlines seen from above (m; 0 while the board is under)."""
    return footprint_gap(*_lo_hi(board), *_lo_hi(patient))


def board_fully_out(patient, board) -> bool:
    """No part of the board under the patient, and the two outlines at least 10 mm apart."""
    return board_clear(patient, board) >= L.OUT_CLEAR - _EPS


def board_on_deck(board) -> bool:
    """Resting flat on the deck: its lowest point within 3 mm of the deck's top, tilted under 3 degrees (either
    broad face up), its centre over the deck (its ends may hang over the deck's edge)."""
    return bool(abs(float(board.low[2]) - L.DECK_TOP) <= L.ON_DECK_Z_TOL
                and board_tilt_deg(board.quat_xyzw) < L.BOARD_FLAT_DEG and over_deck(board.pos))


def board_travel(board) -> float:
    """Horizontal distance of the board's centre from its starting mark (m)."""
    return float(np.hypot(board.pos[0] - L.BOARD_X, board.pos[1] - L.BOARD_Y))


def pulled(board, start_y=L.BOARD_Y) -> float:
    """How far the board's centre has gone along -y from where it lay (m): the way out from under the patient
    along the board's length (negative if pushed further under)."""
    return float(start_y - board.pos[1])


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "patient_moved": "the patient's centre went more than 15 mm from its start (dragged, pushed or carried)",
    "patient_raised": "the patient's underside went more than 30 mm above the board's top (lifted away)",
    "patient_tilted": "the patient tilted more than 10 degrees",
    "patient_dropped": "the patient came down faster than 0.15 m/s",
    "patient_off_table": "the patient left the table",
    "board_off_table": "the board left the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.max_moved = 0.0  # m: the patient's centre, horizontally from its start
        self.max_raised = -math.inf  # m: the centre of the patient's underside above the board's top
        self.max_tilt = 0.0  # degrees
        self.max_descent = 0.0  # m/s: the patient's fastest descent
        self.min_under_share = 1.0  # the least of the board's starting footprint under the patient so far
        self.start = None  # (patient centre xy, board centre y) at the episode's first valid step
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the patient, of the board] at the previous valid control step
        self.prev_underside = None  # (z of the centre of the patient's underside, step) at the last valid step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(patient, board) -> bool:
        """The starting layout: the patient flat on the board near its left end, the board flat on the deck, both
        still (within START_TOL of their marks, which covers a seeded reset's jitter)."""
        p_ok = (moved(patient) < L.START_TOL and abs(float(patient.low[2]) - L.PATIENT_BOTTOM_Z) < 0.003
                and tilt_deg(patient.quat_xyzw) < 1.0)
        b_ok = (board_travel(board) < L.START_TOL and abs(float(board.low[2]) - L.DECK_TOP) < 0.003
                and board_tilt_deg(board.quat_xyzw) < 1.0)
        return bool(p_ok and b_ok and is_resting(patient) and is_resting(board))

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

    def update(self, patient, board, arms=None) -> Result:
        if not (valid(patient) and valid(board)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(patient, board, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (patient, board)]
        if self.prev is not None and self._teleported(now) and self.at_start(patient, board):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.steps += 1
        if self.start is None:  # where things lay at the episode's first step: 'moved' and 'pulled' count from here
            self.start = (np.array(patient.pos[:2], dtype=float), float(board.pos[1]))

        # The patient's rules, at every step.
        m, r, t = moved(patient, self.start[0]), raised(patient), tilt_deg(patient.quat_xyzw)
        self.max_moved = max(self.max_moved, m)
        self.max_raised = max(self.max_raised, r)
        self.max_tilt = max(self.max_tilt, t)
        if m > L.MOVED_MAX:
            self._flag("patient_moved")
        if r > L.RAISE_MAX:
            self._flag("patient_raised")
        if t > L.TILT_MAX_DEG:
            self._flag("patient_tilted")
        # Its descent: its reported downward speed, and how far its underside came down in this step (a fall from
        # the board's 20 mm takes about 64 ms, so one 50 ms step may hold most of it while the reported speed
        # at the step's end, after the landing, is already 0).
        bottom = underside_z(patient)
        descent = max(-float(patient.lin_vel[2]), 0.0)
        if self.prev_underside is not None:
            z0, step0 = self.prev_underside
            descent = max(descent, (z0 - bottom) / (self.dt * max(self.steps - step0, 1)))
        self.max_descent = max(self.max_descent, descent)
        if descent > L.DESCENT_MAX:
            self._flag("patient_dropped")
        if off_table(patient):
            self._flag("patient_off_table")
        if off_table(board):
            self._flag("board_off_table")
        self.prev = now
        self.prev_underside = (bottom, self.steps)

        self.min_under_share = min(self.min_under_share, under_share(patient, board))
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(patient, board, arms)
        return self.last

    def result(self, patient, board, arms=None) -> Result:
        if not (valid(patient) and valid(board)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        released = grippers_released(arms, patient, board) if have_arms else True  # unreadable: taken as met
        p_start, b_start_y = self.start if self.start is not None else ((L.PATIENT_X, L.PATIENT_Y), L.BOARD_Y)
        share = under_share(patient, board)
        pull = pulled(board, b_start_y)
        out = board_fully_out(patient, board)
        b_rest = board_on_deck(board)
        p_flat = patient_flat_on_deck(patient)
        p_on_deck = over_deck(patient.pos)
        if self.breach or not p_on_deck:
            score = 0.0
        elif out and b_rest and p_flat and released:
            score = 1.0
        elif out and p_flat:
            score = 0.75
        elif share <= L.HALF_OUT_SHARE + _EPS:
            score = 0.5
        elif pull >= L.PULL_MIN - _EPS:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(patient) and is_resting(board))
        left = zone_gap(arms["left"], *_lo_hi(patient)) if have_arms else None
        right = zone_gap(arms["right"], *_lo_hi(board)) if have_arms else None
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "moved_mm": _mm(moved(patient, p_start)),
            "max_moved_mm": _mm(self.max_moved),
            "raised_mm": _mm(raised(patient)),
            "max_raised_mm": _mm(self.max_raised) if math.isfinite(self.max_raised) else None,
            "patient_tilt_deg": round(tilt_deg(patient.quat_xyzw), 2),
            "max_tilt_deg": round(self.max_tilt, 2),
            "max_descent_m_s": round(self.max_descent, 3),
            "patient_low_mm": _mm(float(patient.low[2]) - L.DECK_TOP),
            "patient_flat_on_deck": p_flat,
            "patient_over_deck": p_on_deck,
            "under_share": round(share, 3),
            "min_under_share": round(self.min_under_share, 3),
            "pulled_mm": _mm(pull),
            "board_clear_mm": _mm(board_clear(patient, board)),
            "board_fully_out": out,
            "board_on_deck": b_rest,
            "board_travel_mm": _mm(board_travel(board)),
            "board_low_mm": _mm(float(board.low[2]) - L.DECK_TOP),
            "board_tilt_deg": round(board_tilt_deg(board.quat_xyzw), 2),
            "left_to_patient_mm": _mm(left),
            "right_to_board_mm": _mm(right),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "patient_resting": is_resting(patient),
            "board_resting": is_resting(board),
        }
        return Result(score, success, details)
