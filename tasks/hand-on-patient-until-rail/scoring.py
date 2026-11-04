"""The scorer, as plain functions of the patient's and the rail's states and the arms' state. NumPy only; no
simulator needed.

The rule is about the whole trial, not only its end: on no control step may the patient be on the bed with the rail
down and no gripper's hand on them; the patient must never be dropped (descend faster than 0.15 m/s), never tilt
more than 15 degrees, and never be slid off the trolley instead of lifted (moved more than 15 mm while within 10 mm
of the trolley's top or the table). So the scorer keeps a little state (which of those has happened, and when, the
fastest descent and the drag so far, and some counts for the trace) and is fed one observation per control step by
``task.success()``.

    Tracker.update(patient, rail, arms) -> Result(score, success, details)

``patient`` and ``rail`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then 'a hand on the patient' is taken
as true for that step (the rule is taken as kept), 'both grippers let go' as met, and the step is counted in
``arms_unavailable``. An object state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0
for that step and leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within a radius of the 38 mm segment from the point between its finger tips back up the gripper; the
radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn (and a
little more). A gripper has a hand on the patient when it is less than 0.9 open, that zone overlaps the patient's
box, and its finger-tip point is no higher than 10 mm above the patient's top (closed round them, or pressing on
them; not hovering above or beside them). It has let go of an item when it is 0.9 open or more, or its zone is more
than 10 mm from the item (the looser test, so that a gripper closed near an item at the end still counts as holding
it; a closed gripper within 10 mm of the patient but not overlapping them is neither a hand on them nor let go).

Descent: the patient's 'descent rate' on a step is how far its centre came down since the previous step, per
second, and, while its lowest point is more than 3 mm above the surface under it (the deck, the trolley or the
table, by where its centre is), its downward velocity as well. Within the 3 mm rest band only the step-to-step
descent counts, as a 20 Hz video would read it: a block let go 2 mm up reaches 0.2 m/s for a few milliseconds
before it lands, which no video frame would show, while a block let go 20 mm up comes down 20 mm between two
frames (0.4 m/s) and is caught either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")


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
def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's z axis and the world's vertical (degrees, 0..180). Not folded: the patient's top
    must stay up (a patient turned over is 180 degrees)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def face_tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through a box's top and bottom faces (degrees, 0..90):
    either face may be down. For the rail, a plain bar that is the same either way up."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def heading_err_deg(quat_xyzw) -> float:
    """How far a box's long axis (its body x), seen from above, is turned from the world's x axis (degrees, 0..90;
    either way round)."""
    r = rotation(quat_xyzw)
    return float(np.degrees(math.atan2(abs(r[1, 0]), abs(r[0, 0]))))


# -- footprints and surfaces ---------------------------------------------------------------------------------------
def over_deck(p) -> bool:
    """Is a point (x, y, ...) over the bed deck's 200 x 300 mm footprint?"""
    return bool(L.DECK_X0 <= float(p[0]) <= L.DECK_X1 and L.DECK_Y0 <= float(p[1]) <= L.DECK_Y1)


def outline_meets_deck(s) -> bool:
    """Any part of the body's outline from above (its bounding box) is over the deck."""
    lo, hi = np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)
    return bool(lo[0] < L.DECK_X1 and hi[0] > L.DECK_X0 and lo[1] < L.DECK_Y1 and hi[1] > L.DECK_Y0)


def over_trolley(p) -> bool:
    """Is a point over the trolley's 200 x 160 mm footprint?"""
    return bool(L.TROLLEY_X0 <= float(p[0]) <= L.TROLLEY_X1 and L.TROLLEY_Y0 <= float(p[1]) <= L.TROLLEY_Y1)


def surface_under(s) -> float:
    """Height of the surface the body's centre is over: the deck's top, the trolley's top, else the table top."""
    if over_deck(s.pos):
        return L.DECK_TOP
    if over_trolley(s.pos):
        return L.TROLLEY_TOP
    return L.TOP


def clearance(s) -> float:
    """Height of the body's lowest point above the surface under it (m; negative: below it)."""
    return float(s.low[2]) - surface_under(s)


# -- the patient ---------------------------------------------------------------------------------------------------
def centre_off(patient) -> float:
    """Horizontal distance from the patient's centre to the bed mark's centre (m)."""
    return float(np.hypot(patient.pos[0] - L.MARK_X, patient.pos[1] - L.MARK_Y))


def on_bed(patient) -> bool:
    """The unattended rule's 'on the bed': any part of the patient's outline over the deck and its lowest point
    within 10 mm of the deck's top, above or below (resting on it, just above it, or half on it, bridging the gap
    to the trolley or hanging over an edge)."""
    return bool(outline_meets_deck(patient) and abs(float(patient.low[2]) - L.DECK_TOP) <= L.ON_BED_Z_TOL)


def resting_on_bed(patient) -> bool:
    """Resting on the bed, however placed: its centre over the deck, its lowest point within 3 mm of the deck's top
    (on the mark it is 1 mm up), tilted less than 15 degrees."""
    return bool(over_deck(patient.pos) and abs(float(patient.low[2]) - L.DECK_TOP) <= L.REST_Z_TOL
                and tilt_deg(patient.quat_xyzw) < L.TIP_DEG)


def flat_on_bed(patient, tol: float) -> bool:
    """Resting flat on the bed: resting on it, tilted less than 5 degrees, its centre within ``tol`` of the mark's
    centre horizontally."""
    return bool(resting_on_bed(patient) and tilt_deg(patient.quat_xyzw) < L.FLAT_DEG and centre_off(patient) <= tol)


def resting_on_trolley(patient) -> bool:
    """Its centre over the trolley and its lowest point within 3 mm of the trolley's top."""
    return bool(over_trolley(patient.pos) and abs(float(patient.low[2]) - L.TROLLEY_TOP) <= L.REST_Z_TOL)


def lifted_off_trolley(patient) -> bool:
    """Its lowest point at least 20 mm above the trolley's top (traced, not scored)."""
    return bool(float(patient.low[2]) >= L.TROLLEY_TOP + L.LIFT_CLEAR)


def on_table_off_platforms(patient) -> bool:
    """The patient's centre over neither platform and its lowest point within 10 mm of the table top, or below it:
    it ended on the table (in the gap, or beside the platforms)."""
    return bool(not over_deck(patient.pos) and not over_trolley(patient.pos)
                and float(patient.low[2]) < L.TOP + L.ON_TABLE_TOL)


def tipped(patient) -> bool:
    return tilt_deg(patient.quat_xyzw) > L.TIP_DEG


def low(patient) -> bool:
    """Low enough off the bed to be dragging: its centre not over the bed deck and its lowest point less than 10 mm
    above the surface under it (the trolley's top when its centre is over the trolley, else the table; over the
    40 mm gap between the platforms it is 30 mm up, not low). On the bed, sliding to position the patient is
    allowed, so it is never 'low' there."""
    return bool(not over_deck(patient.pos) and clearance(patient) < L.DRAG_BAND)


def off_table(s) -> bool:
    """The body's lowest point more than 10 mm below the table top: it fell off the table."""
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


# -- the rail ------------------------------------------------------------------------------------------------------
def rail_off(rail) -> float:
    """Horizontal distance from the rail's centre to its place at the deck's right edge (m)."""
    return float(np.hypot(rail.pos[0] - L.RAIL_PLACE_X, rail.pos[1] - L.RAIL_PLACE_Y))


def rail_up(rail) -> bool:
    """The rail is up: resting along the deck's right edge with its long side along x, its centre within 10 mm of
    its place horizontally, its long axis along x within 5 degrees, tilted less than 5 degrees, and its lowest point
    within 3 mm of the deck's top (resting on the deck, or held there just above it)."""
    return bool(rail_off(rail) <= L.RAIL_XY_TOL and heading_err_deg(rail.quat_xyzw) < L.RAIL_HEADING_DEG
                and face_tilt_deg(rail.quat_xyzw) < L.RAIL_TILT_DEG
                and abs(float(rail.low[2]) - L.DECK_TOP) <= L.RAIL_Z_TOL)


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


def _box(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def let_go(arm, lo, hi) -> bool:
    """The gripper is not holding the item in the box (lo, hi): it is open (opening 0.9 or more), or its finger zone
    is more than 10 mm from the item. A gripper closed somewhere else is not holding it."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, lo, hi) > L.HAND_GAP)


def hand_on(arm, lo, hi) -> bool:
    """The gripper has a hand on the item in the box (lo, hi): less than 0.9 open, its finger zone overlapping the
    item's box, and its finger-tip point no higher than 10 mm above the item's top (closed round it or pressing on
    it, not hovering above or beside it)."""
    return bool(float(arm.opening) < L.GRIPPER_OPEN and zone_gap(arm, lo, hi) < 0.0
                and float(np.asarray(arm.tips, dtype=float)[2]) <= float(hi[2]) + L.HAND_GAP)


def hands_on(arms, patient) -> dict[str, bool]:
    """Which grippers have a hand on the patient."""
    lo, hi = _box(patient)
    return {s: hand_on(arms[s], lo, hi) for s in ("left", "right")}


def held(arms, patient) -> bool:
    """Some gripper has a hand on the patient (either counts)."""
    return any(hands_on(arms, patient).values())


def grippers_released(arms, patient, rail) -> bool:
    """Every gripper has let go of both the patient and the rail."""
    p, r = _box(patient), _box(rail)
    return all(let_go(arms[s], *p) and let_go(arms[s], *r) for s in ("left", "right"))


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
    "unattended": "the patient was on the bed with the rail not up and no gripper's hand on them",
    "dropped": "the patient came down faster than 0.15 m/s",
    "tipped": "the patient tilted more than 15 degrees",
    "dragged": "the patient slid more than 15 mm horizontally while within 10 mm of the trolley's top or the table "
               "(pushed off the trolley instead of lifted)",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.unattended_steps = 0  # control steps on which the patient was unattended
        self.max_descent = 0.0  # m/s: the patient's fastest descent rate so far
        self.max_tilt = 0.0  # degrees: the patient's largest tilt so far
        self.drag = 0.0  # m: the patient's centre's horizontal travel while low (followed in 0.5 mm steps)
        self.drag_mark = None  # (x, y) where drag was last counted, while low
        self.lifted = False  # the patient has been 20 mm clear of the trolley's top at some step
        self.on_bed_steps = 0  # control steps with the patient on the bed (held or not, rail up or not)
        self.held_on_bed_steps = 0  # ... of which a gripper had a hand on them while the rail was not up
        self.rail_up_steps = 0  # control steps with the rail up
        self.first_rail_up_step: int | None = None
        self.first_on_bed_step: int | None = None
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the patient, of the rail] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(patient, rail) -> bool:
        """The starting layout: the patient flat on the trolley on its mark and the rail flat on the table on its
        mark along x, both still (within START_TOL of the marks, which covers a seeded reset's jitter)."""
        p_ok = (np.hypot(patient.pos[0] - L.PATIENT_X, patient.pos[1] - L.PATIENT_Y) < L.START_TOL
                and abs(float(patient.low[2]) - L.TROLLEY_TOP) < L.START_Z_TOL and tilt_deg(patient.quat_xyzw) < 1.0)
        r_ok = (np.hypot(rail.pos[0] - L.RAIL_X, rail.pos[1] - L.RAIL_Y) < L.START_TOL
                and abs(float(rail.low[2]) - L.TOP) < L.START_Z_TOL and face_tilt_deg(rail.quat_xyzw) < 1.0
                and heading_err_deg(rail.quat_xyzw) < 2.0)
        return bool(p_ok and r_ok and is_resting(patient) and is_resting(rail))

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

    def update(self, patient, rail, arms=None) -> Result:
        if not (valid(patient) and valid(rail)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(patient, rail, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (patient, rail)]
        if self.prev is not None and self._teleported(now) and self.at_start(patient, rail):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        prev = self.prev
        self.prev = now
        self.steps += 1
        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1

        # Descent: how far the centre came down since the last step, per second, and, above the 3 mm rest band,
        # the downward velocity too.
        rate = 0.0 if prev is None else (float(prev[0][0][2]) - float(now[0][0][2])) / self.dt
        if clearance(patient) > L.REST_BAND:
            rate = max(rate, -float(patient.lin_vel[2]))
        self.max_descent = max(self.max_descent, rate)
        if self.max_descent > L.DROP_SPEED:
            self._flag("dropped")

        tilt = tilt_deg(patient.quat_xyzw)
        self.max_tilt = max(self.max_tilt, tilt)
        if tilt > L.TIP_DEG:
            self._flag("tipped")
        if lifted_off_trolley(patient):
            self.lifted = True

        # Dragging: the centre's horizontal travel while the patient is low off the bed (within 10 mm of the
        # trolley's top or the table), followed in 0.5 mm steps so that a resting body's contact jitter does not
        # add up.
        xy = now[0][0][:2]
        if low(patient):
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

        # The hand-on rule, on this step: the patient on the bed, the rail not up, nobody's hand on them.
        up = rail_up(rail)
        bed = on_bed(patient)
        hand = held(arms, patient) if have_arms else True  # unreadable: taken as kept, counted above
        if up:
            self.rail_up_steps += 1
            if self.first_rail_up_step is None:
                self.first_rail_up_step = self.steps
        if bed:
            self.on_bed_steps += 1
            if self.first_on_bed_step is None:
                self.first_on_bed_step = self.steps
            if not up:
                if hand:
                    self.held_on_bed_steps += 1
                else:
                    self.unattended_steps += 1
                    self._flag("unattended")

        self.last = self.result(patient, rail, arms)
        return self.last

    def result(self, patient, rail, arms=None) -> Result:
        if not (valid(patient) and valid(rail)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        hands = hands_on(arms, patient) if have_arms else None
        hand = any(hands.values()) if have_arms else True  # unreadable: taken as met, counted
        released = grippers_released(arms, patient, rail) if have_arms else True
        up = rail_up(rail)
        resting = resting_on_bed(patient)
        tight = flat_on_bed(patient, L.CENTRE_TIGHT)
        loose = flat_on_bed(patient, L.CENTRE_LOOSE)
        gone = off_table(patient) or off_table(rail)
        on_table = on_table_off_platforms(patient)
        if self.breach or gone or on_table:
            score = 0.0
        elif up and tight and released:
            score = 1.0
        elif up and loose:
            score = 0.75
        elif up and resting:
            score = 0.5
        elif loose and hand:  # the rail is not up: safe but unfinished
            score = 0.5
        elif up or (resting and hand):  # exactly one of the two halves of the job
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(patient) and is_resting(rail))
        p, r = _box(patient), _box(rail)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "unattended_steps": self.unattended_steps,
            "rail_up": up,
            "rail_off_mm": _mm(rail_off(rail)),
            "rail_heading_err_deg": _deg(heading_err_deg(rail.quat_xyzw)),
            "rail_tilt_deg": _deg(face_tilt_deg(rail.quat_xyzw)),
            "rail_low_mm": _mm(float(rail.low[2]) - L.DECK_TOP),
            "on_bed": on_bed(patient),
            "resting_on_bed": resting,
            "flat_on_mark": tight,
            "flat_on_bed_loose": loose,
            "resting_on_trolley": resting_on_trolley(patient),
            "on_table_off_platforms": on_table,
            "off_table": gone,
            "surface": "deck" if over_deck(patient.pos) else ("trolley" if over_trolley(patient.pos) else "table"),
            "clearance_mm": _mm(clearance(patient)),
            "patient_low_mm": _mm(float(patient.low[2]) - L.DECK_TOP),
            "centre_off_mm": _mm(centre_off(patient)),
            "patient_tilt_deg": _deg(tilt_deg(patient.quat_xyzw)),
            "max_tilt_deg": _deg(self.max_tilt),
            "max_descent_mps": round(self.max_descent, 3) if math.isfinite(self.max_descent) else None,
            "low": low(patient),
            "drag_mm": _mm(self.drag),
            "lifted_off_trolley": self.lifted,
            "hands_on": hands,
            "held": hand if have_arms else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "zone_to_patient_mm": ({s: _mm(zone_gap(arms[s], *p)) for s in ("left", "right")} if have_arms else None),
            "zone_to_rail_mm": ({s: _mm(zone_gap(arms[s], *r)) for s in ("left", "right")} if have_arms else None),
            "grippers_let_go": released if have_arms else None,
            "on_bed_steps": self.on_bed_steps,
            "held_on_bed_s": round(self.held_on_bed_steps * self.dt, 2),
            "rail_up_steps": self.rail_up_steps,
            "first_rail_up_step": self.first_rail_up_step,
            "first_on_bed_step": self.first_on_bed_step,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "patient_resting": is_resting(patient),
            "rail_resting": is_resting(rail),
        }
        return Result(score, success, details)
