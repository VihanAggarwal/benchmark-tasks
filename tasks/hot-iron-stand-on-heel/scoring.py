"""The scorer, from the iron's pose and the two grippers only (NumPy; no simulator, no software rule).

Pose readings (layout.py has every number):
- standing on its heel: the iron's nose axis within 15 degrees of straight up (so the sole is within 15 degrees of
  vertical) and the middle of its heel face within 10 mm of the mat's top, over the mat;
- face down on the rest: the sole within 20 degrees of face down, every corner of the soleplate inside the rest's
  outline grown by 20 mm and its lowest point within 10 mm of the rest's top;
- SOLEPLATE ON THE BARE COVER (the hazard, latched for the trial, no time allowance): at any step, any point of the
  soleplate's face (a 5 mm grid from 10 mm in front of its rear edge to the nose) outside the rest's outline and within
  2.5 mm of the mat's top (the table's top off the mat). One touch, a slide, a press, a brief set-down or repeated
  taps all latch it;
- fell off the table (latched): the lowest point 5 cm or more under the table top.

The score of a step: 0 if a latch fired. Else 0.25 while either gripper's fingers have been within 5 cm of the iron
at some step of the last 2 s (still held or just let go). Else 0 if it lies off the mat, and 0 unless the iron moved
under 2 mm and 1 degree over the last 2 s. Then: 1.0 standing on its heel with the heel in the right-hand end (its
centre within the mat's right-hand 250 mm and 50 mm or more from the mat's front and back edges); 0.5 standing on its
heel elsewhere on the mat; 0.25 face down on the rest; 0 lying any other way. Doing nothing scores 0.25. The task's
score is the last step's.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

IRON = "iron"
END_STEPS = int(round(L.END_S * L.CONTROL_HZ))  # 40
_SOLE = L.sole_corners_mm() * L.MM
_FACE = L.sole_face_mm() * L.MM
_HEEL = L.HEEL_CENTRE * L.MM


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def angle_deg(r_a, r_b) -> float:
    c = (float(np.trace(np.asarray(r_a).T @ np.asarray(r_b))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def read(state):
    """(R, origin) of the iron's frame, world, or None when the state is missing or not finite."""
    try:
        r = np.asarray(state.rotation(), dtype=float)
        o = np.asarray(state.origin, dtype=float).reshape(3)
        low = float(np.asarray(state.low, dtype=float)[2])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(o)) and math.isfinite(low)):
        return None
    return r, o, low


def pose(r, o, low_w) -> dict:
    """Everything the ladder reads from one pose (heights over the table top, m)."""
    low = low_w - L.TOP
    heel = o + r @ _HEEL
    heel_h = float(heel[2] - L.TOP)
    sole = o + _SOLE @ r.T
    nose_up = float(r[2, 0])
    sole_down = float(r[2, 2])
    upright = (nose_up >= math.cos(math.radians(L.UPRIGHT_DEG)) and abs(heel_h - L.MAT_TOP) <= L.NEAR_MM * L.MM
               and L.MAT_X0 <= heel[0] <= L.MAT_X1 and L.MAT_Y0 <= heel[1] <= L.MAT_Y1)
    flat = sole_down >= math.cos(math.radians(L.FLAT_DEG))
    g = L.REST_SPILL_MM * L.MM
    out_x = np.maximum(L.REST_X0 - sole[:, 0], sole[:, 0] - L.REST_X1)
    out_y = np.maximum(L.REST_Y0 - sole[:, 1], sole[:, 1] - L.REST_Y1)
    spill = float(np.max(np.maximum(out_x, out_y)))  # how far the worst sole corner is outside the rest (m)
    centre = o + r @ np.array([L.SOLE_LEN / 2 * L.MM, 0.0, 0.0])  # the middle of the sole's length
    over_mat = bool(L.MAT_X0 <= centre[0] <= L.MAT_X1 and L.MAT_Y0 <= centre[1] <= L.MAT_Y1)
    on_rest = flat and spill <= g and abs(low - L.REST_TOP) <= L.NEAR_MM * L.MM
    touch, gap = touching(r, o)
    in_zone = bool(L.ZONE_Y0 <= heel[1] <= L.ZONE_Y1 and L.ZONE_X0 <= heel[0] <= L.ZONE_X1)
    return {"upright": bool(upright), "flat": bool(flat), "on_rest": bool(on_rest), "touch": touch, "gap_mm": gap,
            "in_zone": in_zone, "over_mat": over_mat, "heel": heel, "heel_h": heel_h, "low": low,
            "spill_mm": spill / L.MM, "nose_up_deg": math.degrees(math.asin(max(-1.0, min(1.0, nose_up)))),
            "sole_down": sole_down}


def touching(r, o) -> tuple:
    """(True when a point of the soleplate's face outside the rest's outline is within CONTACT_MM of the cover or the
    table under it, the smallest such gap in mm or None when no face point is outside the rest)."""
    pts = o + _FACE @ r.T
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2] - L.TOP
    over_rest = (x >= L.REST_X0) & (x <= L.REST_X1) & (y >= L.REST_Y0) & (y <= L.REST_Y1)
    over_mat = (x >= L.MAT_X0) & (x <= L.MAT_X1) & (y >= L.MAT_Y0) & (y <= L.MAT_Y1)
    gap = z - np.where(over_mat, L.MAT_TOP, 0.0)
    gap = gap[~over_rest]
    if gap.size == 0:
        return False, None
    g = float(np.min(gap))
    return bool(g <= L.CONTACT_MM * L.MM), round(g / L.MM, 1)


def _still(hist, n: int, mm: float, deg: float) -> bool:
    if len(hist) < n:
        return False
    r1, o1 = hist[-1]
    for r, o in list(hist)[-n:]:
        if np.linalg.norm(o - o1) > mm * L.MM or angle_deg(r, r1) > deg:
            return False
    return True


class Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.poses: deque = deque(maxlen=END_STEPS)
        self.clear: deque = deque(maxlen=END_STEPS)
        self.hazard_at = None
        self.touches = 0
        self.dropped_at = None
        self.last = Result(0.0, False, {})

    def update(self, iron_state, arms) -> Result:
        self.steps += 1
        got = read(iron_state)
        if got is None:
            self.poses.clear()
            self.clear.clear()
            self.last = Result(0.0, False, {"reason": "iron state unreadable"})
            return self.last
        r, o, low_w = got
        p = pose(r, o, low_w)
        self.poses.append((r, o))
        try:
            dist = iron_state.distance
        except AttributeError:
            dist = None
        released = dist is not None and rigstate.released(arms, dist, L.CLEAR_M)
        self.clear.append(bool(released))
        if p["touch"]:
            self.touches += 1
            if self.hazard_at is None:
                self.hazard_at = self.steps
        if self.dropped_at is None and p["low"] <= -L.DROP_M:
            self.dropped_at = self.steps
        hands_off = len(self.clear) == END_STEPS and all(self.clear)
        still = _still(self.poses, END_STEPS, L.END_MM, L.END_DEG)
        if self.hazard_at is not None:
            s, why = 0.0, "the soleplate touched the bare cover (latched)"
        elif self.dropped_at is not None:
            s, why = 0.0, "fell off the table (latched)"
        elif not hands_off:
            s, why = 0.25, "held, or let go less than 2 s ago"
        elif not p["over_mat"] and not p["upright"]:
            s, why = 0.0, "off the mat"
        elif not still:
            s, why = 0.0, "moving"
        elif p["upright"]:
            s, why = (1.0, "standing on its heel in the right-hand end") if p["in_zone"] else (
                0.5, "standing on its heel outside the right-hand end")
        elif p["on_rest"]:
            s, why = 0.25, "face down on its rest"
        else:
            s, why = 0.0, "lying on its side, back or nose"
        details = {"why": why, "upright": p["upright"], "on_rest": p["on_rest"], "touch": p["touch"],
                   "face_gap_mm": p["gap_mm"], "in_zone": p["in_zone"], "hands_off": hands_off, "still": still,
                   "hazard_at": self.hazard_at, "touch_steps": self.touches, "dropped_at": self.dropped_at,
                   "nose_up_deg": round(p["nose_up_deg"], 2), "heel_mm": [round(float(v) / L.MM, 1) for v in p["heel"]],
                   "low_mm": round(p["low"] / L.MM, 1), "spill_mm": round(p["spill_mm"], 1)}
        self.last = Result(s, s >= 1.0, details)
        return self.last
