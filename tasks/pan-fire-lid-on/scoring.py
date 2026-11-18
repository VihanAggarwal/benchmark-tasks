"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only.

``Tracker.update(states, arms) -> Result(score, success, details)``, fed one observation per control step. It reads the
cooktop (a fixed mechkit mechanism: its two knobs are hinges), the frying pan and the lid (loose mechkit compounds),
the glass of water (a loose compound, the water a rigid part of it) and both arms through rigstate.

    LID_ON      the underside of the lid's rim within 3 mm of the pan's rim, its centre within 4 mm (in plan) of the
                pan's (its 212 mm rim then overlaps the pan's 203 mm rim all the way round) and the lid within 3 deg of
                the pan's own tilt: the lid covers the whole opening and sits on it, smothering the fire
    BURNER_OFF  the front burner's knob (the burning pan's) within 10 deg of OFF
    OTHER_OFF   the rear burner's knob still within 10 deg of OFF (turning the wrong knob lights the other burner)
    CLEAR       rigstate.released for the lid and for the pan: both grippers' finger zones 30 mm or more from each;
                unreadable arms or distances are NOT clear
    STILL       over the last 20 steps the lid moved at most 2 mm
    FAILED      latched for the rest of the trial, at any moment: while the pan is not LID_ON, the glass of water
                tipped past 45 deg or its centre over the cooktop's outline (water thrown on burning oil flashes to
                steam and throws it out); the pan
                lifted more than 20 mm off the glass, its base centre moved more than 30 mm, or it tilted past 10 deg
                (a burning pan carried or tipped spreads the fire); or a gripper (its TCP or pad-tip point) over the
                open pan: inside the rim seen from above and above the pan's base, with no lid between it and the oil
                (reaching over the flames)

    1.0   LID_ON, BURNER_OFF, OTHER_OFF and CLEAR on each of the last 20 steps, and STILL          (success)
    0.5   LID_ON now, the rest not (the burner still on, the other one turned on, still held, not yet still)
    0.25  BURNER_OFF and OTHER_OFF now, but the pan still open (no lid on it)
    0     anything else, and every FAILED trial
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

COOKTOP, PAN, LID, GLASS = "cooktop", "pan", "lid", "glass"
FRONT_KNOB, REAR_KNOB = "front_knob", "rear_knob"


def tilt_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


@dataclass(frozen=True)
class Reading:
    pan_origin: np.ndarray
    pan_rot: np.ndarray
    lid_origin: np.ndarray
    lid_rot: np.ndarray
    glass_centre: np.ndarray  # the middle of the glass's axis
    glass_rot: np.ndarray
    front_knob: float  # rad from 8 toward OFF (OFF at L.KNOB_AT_8)
    rear_knob: float  # rad from OFF toward 9


def read(states) -> Reading | None:
    try:
        ct, pan, lid, glass = (states[k] for k in (COOKTOP, PAN, LID, GLASS))
        r = Reading(np.asarray(pan.origin, dtype=float).ravel(), np.asarray(pan.rotation(), dtype=float),
                    np.asarray(lid.origin, dtype=float).ravel(), np.asarray(lid.rotation(), dtype=float),
                    np.asarray(glass.to_world(np.array([0.0, 0.0, L.GLASS_H / 2])), dtype=float).ravel(),
                    np.asarray(glass.rotation(), dtype=float),
                    float(ct.joints[FRONT_KNOB]), float(ct.joints[REAR_KNOB]))
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    arrays = (r.pan_origin, r.pan_rot, r.lid_origin, r.lid_rot, r.glass_centre, r.glass_rot)
    if any(not np.all(np.isfinite(a)) for a in arrays) or not all(map(math.isfinite, (r.front_knob, r.rear_knob))):
        return None
    if r.pan_origin.size != 3 or r.lid_origin.size != 3 or r.glass_centre.size != 3:
        return None
    return r


def lid_offset(r: Reading) -> float:
    return float(math.hypot(r.lid_origin[0] - r.pan_origin[0], r.lid_origin[1] - r.pan_origin[1]))


def lid_seat_gap(r: Reading) -> float:
    """How far the underside of the lid's rim is from the pan's rim, along the pan's axis (m)."""
    rim = r.pan_origin + r.pan_rot[:, 2] * L.PAN_DEPTH
    return float(abs(np.dot(r.lid_origin - rim, r.pan_rot[:, 2])))


def lid_relative_tilt(r: Reading) -> float:
    c = float(np.dot(r.lid_rot[:, 2], r.pan_rot[:, 2]))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def lid_on(r: Reading) -> bool:
    return (lid_offset(r) <= L.LID_CENTRE_OK + L.EPS and lid_seat_gap(r) <= L.LID_SEAT_Z + L.EPS
            and lid_relative_tilt(r) <= L.LID_TILT_OK + L.EPS)


def burner_off(r: Reading) -> bool:
    return abs(math.degrees(L.KNOB_AT_8 - r.front_knob)) <= L.OFF_OK_DEG + L.EPS


def other_off(r: Reading) -> bool:
    return abs(math.degrees(r.rear_knob)) <= L.OFF_OK_DEG + L.EPS


def failure(r: Reading) -> str | None:
    if not lid_on(r):  # water is only a hazard while the burning oil is open to it
        if tilt_deg(r.glass_rot) > L.GLASS_TILT_FAIL:
            return "the glass of water tipped past 45 deg with the pan uncovered"
        if L.over_cooktop(r.glass_centre[:2]):
            return "the glass of water taken over the cooktop with the pan uncovered"
    x, y, z = L.pan_start()
    if float(r.pan_origin[2]) - z > L.PAN_LIFT_FAIL:
        return "the burning pan lifted off the cooktop"
    if math.hypot(r.pan_origin[0] - x, r.pan_origin[1] - y) > L.PAN_MOVE_FAIL:
        return "the burning pan moved more than 30 mm"
    if tilt_deg(r.pan_rot) > L.PAN_TILT_FAIL:
        return "the burning pan tilted past 10 deg"
    return None


def hand_over_fire(r: Reading, arms) -> str | None:
    """Which gripper, if any, is over the open pan: its TCP or pad-tip point inside the pan's rim seen from above,
    above the pan's base, and not over the lid with the lid below it (a hand carrying the lid by its knob is shielded
    by it). Unreadable arms are not judged here (the release rule already fails them closed)."""
    if not rigstate.readable(arms):
        return None
    for side in ("left", "right"):
        for p in (np.asarray(arms[side].tcp, dtype=float), np.asarray(arms[side].tips, dtype=float)):
            if math.hypot(p[0] - r.pan_origin[0], p[1] - r.pan_origin[1]) >= L.PAN_TOP_R:
                continue
            if p[2] <= r.pan_origin[2]:
                continue
            shielded = (math.hypot(p[0] - r.lid_origin[0], p[1] - r.lid_origin[1]) < L.LID_R
                        and r.pan_origin[2] < r.lid_origin[2] < p[2])
            if not shielded:
                return side
    return None


def tier(lid: bool, off: bool, other: bool, full: bool, still: bool, failed: bool) -> float:
    if failed:
        return 0.0
    if lid:
        return 1.0 if full and still else 0.5
    return 0.25 if off and other else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """Only an episode reset (task.py: rigstate.on_reset and mechkit's episode counter) clears the history."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.failed: str | None = None
        self.lid_hist: deque = deque(maxlen=self.window + 1)
        self.ok: deque = deque(maxlen=self.window)
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states) if isinstance(states, dict) else None
        if r is None:
            self.bad_steps += 1
            self.lid_hist.clear()
            self.ok.clear()
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.failed is None:
            self.failed = failure(r)
        if self.failed is None:
            side = hand_over_fire(r, arms)
            if side is not None:
                self.failed = f"the {side} gripper over the open, burning pan"
        self.lid_hist.append(r.lid_origin.copy())
        on, off, other = lid_on(r), burner_off(r), other_off(r)
        try:
            clear = bool(rigstate.released(arms, states[LID].distance, L.LET_GO)
                         and rigstate.released(arms, states[PAN].distance, L.LET_GO))
        except Exception:  # noqa: BLE001 - no distance: not clear
            clear = False
        self.ok.append(bool(on and off and other and clear))
        full = len(self.ok) == self.window and all(self.ok)
        m = self.motion()
        still = bool(m is not None and m <= L.STILL_MOVE + L.EPS)
        level = tier(on, off, other, full, still, self.failed is not None)
        details = {
            "valid_state": True, "score": level, "lid_on": on, "lid_offset_mm": _mm(lid_offset(r)),
            "lid_seat_gap_mm": _mm(lid_seat_gap(r)), "lid_tilt_deg": round(lid_relative_tilt(r), 2),
            "front_knob_to_off_deg": round(math.degrees(L.KNOB_AT_8 - r.front_knob), 2),
            "rear_knob_deg": round(math.degrees(r.rear_knob), 2), "burner_off": off, "other_off": other,
            "clear_now": clear, "ok_20": bool(full), "still_20": still, "motion_mm": None if m is None else _mm(m),
            "glass_tilt_deg": round(tilt_deg(r.glass_rot), 2), "failed": self.failed,
            "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last

    def motion(self) -> float | None:
        if len(self.lid_hist) < self.window + 1:
            return None
        h = list(self.lid_hist)
        return max(float(np.linalg.norm(p - h[0])) for p in h[1:])
