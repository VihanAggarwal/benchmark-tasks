"""The scorer, as plain functions of the four loose kerb units' states. NumPy only; no simulator needed.

A dropped kerb at a driveway entrance is laid the way kerb layers lay it: the two low (dropped) kerbs in the middle
two slots of the gap in the kerb line, and a transition kerb in each end slot rising to the full kerb beside it, every
unit bedded flat, its back in line with the full kerbs' backs and its battered face to the road. The two transitions
are mirror images: only one of them can rise toward the left full kerb with its face to the road, and only the other
toward the right one. Each slot holding the right unit laid the right way earns 0.25. The score is read from the
state at hand (the last step); the scorer keeps only a short history of where each unit was, to tell whether it is
still, and is fed one observation per control step by ``task.success()``:

    Tracker.update({"low_kerb_1": state, ..., "transition_right": state}) -> Result(score, success, details)

A unit's state is a mechkit ``AssemblyState`` (``origin``: its frame's origin, the centre of its underside;
``quat_xyzw``: its frame's orientation; and the usual ``pos``, ``lin_vel``, ``ang_vel``, ``low``, ``high``), or
anything with those attributes. The unit's shape (layout.py) gives every point of it from ``origin`` and
``quat_xyzw``. A missing unit, or a state with a non-finite number, a zero quaternion or no ``origin`` (a simulator
blow-up, or a unit not built as a compound), scores 0 for that step and leaves the tracker as it was. The two low
kerbs are alike; each transition is told by its shape (which end of it is high), not by where it lies.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

NAMES = L.UNITS
_KEYS = ("origin", "pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)], dtype=float)
# The solid slab every unit has under it (60 x 25 mm, 8 mm: the face strip's height at a unit's low end), in its own
# frame; its four bottom corners are the unit's lowest points, the two at -y its back corners.
SLAB = _SIGNS * np.array([L.UNIT_LEN / 2, L.UNIT_DEPTH / 2, L.LOW_H - L.LIP_DROP])  # (8, 3)
BACK_CORNERS = np.array([(-L.UNIT_LEN / 2, -L.UNIT_DEPTH / 2, 0.0), (L.UNIT_LEN / 2, -L.UNIT_DEPTH / 2, 0.0)])


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
    """Every number finite, the quaternion not degenerate and the frame's origin there (a simulator blow-up gives
    NaN or zeros; a unit that was not built as a compound has no origin)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 3 and parts[2].size == 4 and np.linalg.norm(parts[2]) > 0.5)


def frame(s) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the unit's frame in the world."""
    return rotation(s.quat_xyzw), np.asarray(s.origin, dtype=float)


def slab_corners(s) -> np.ndarray:
    """(8, 3): the corners of the unit's solid bottom slab in the world."""
    r, o = frame(s)
    return SLAB @ r.T + o


def back_corners(s) -> np.ndarray:
    """(2, 3): the two bottom corners of the unit's back face (the side away from its battered face), in the world."""
    r, o = frame(s)
    return BACK_CORNERS @ r.T + o


def tilt_deg(r: np.ndarray) -> float:
    """Angle between the vertical and the unit's up axis, 0..180 degrees: 180 is upside down."""
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2]))))))


def _horizontal(v: np.ndarray) -> np.ndarray | None:
    u = np.array([float(v[0]), float(v[1])])
    n = float(np.linalg.norm(u))
    return None if n < 1e-6 else u / n


def face_angle_deg(r: np.ndarray) -> float:
    """How far (0..180 degrees, seen from above) the unit's battered face points away from straight at the road
    (+x); 90 for a unit standing on its face or back (no direction)."""
    u = _horizontal(r[:, 1])
    if u is None:
        return 90.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(u[0]))))))


def high_end_y(name: str, r: np.ndarray) -> float:
    """For a transition: the y component (seen from above, unit length) of the direction from its low end to its
    high end; positive = it rises toward +y (the left full kerb). 0 for a low kerb (no high end)."""
    if L.KIND[name] != "transition":
        return 0.0
    u = _horizontal(L.HIGH_END[name] * r[:, 0])
    return 0.0 if u is None else float(u[1])


def heading_deg(r: np.ndarray) -> float | None:
    """The world angle (degrees) of the unit's own +x seen from above; None standing on an end."""
    u = _horizontal(r[:, 0])
    return None if u is None else float(math.degrees(math.atan2(u[1], u[0])))


def track_points(s) -> np.ndarray:
    """(2, 3): the unit's origin and the bottom of one end, whose movement says whether it is still (both catch a
    slide, the second also a turn on the spot)."""
    r, o = frame(s)
    return np.array([o, o + r @ np.array([L.UNIT_LEN / 2, 0.0, 0.0])])


def unit_facts(name: str, s) -> dict:
    """Everything about one unit's pose that the rule needs (stillness is the tracker's)."""
    r, o = frame(s)
    c = slab_corners(s)
    low = float(c[:, 2].min())
    tilt = tilt_deg(r)
    backs = back_corners(s)
    back_off = [float(p[0]) - L.BACK_X for p in backs]
    face = face_angle_deg(r)
    return {
        "origin": o,
        "tilt_deg": tilt,
        "flat": bool(tilt <= L.TILT_MAX_DEG + 1e-9),
        "low": low,
        "on_bed": bool(abs(low - L.TOP) <= L.BED_TOL + 1e-12),
        "back_off": back_off,
        "in_line": bool(max(abs(v) for v in back_off) <= L.LINE_TOL + 1e-12),
        "face_deg": face,
        "face_to_road": bool(face <= L.FACE_MAX_DEG + 1e-9),
        "high_y": high_end_y(name, r),
        "heading": heading_deg(r),
    }


def bedded(f: dict, still: bool) -> bool:
    """Is this unit (its facts) bedded on the kerb line: still, flat (tilted at most 5 degrees), its lowest point
    within 3 mm of the bed, both back corners within 5 mm of the line of the full kerbs' backs and its battered face
    pointing within 30 degrees of straight at the road?"""
    return bool(still and f["flat"] and f["on_bed"] and f["in_line"] and f["face_to_road"])


def in_slot(f: dict, slot: str) -> bool:
    """The unit's middle within 8 mm of the slot's centre, along the kerb line."""
    return bool(abs(float(f["origin"][1]) - L.SLOTS[slot]) <= L.ALONG_TOL + 1e-12)


def right_in(name: str, f: dict, slot: str, still: bool) -> bool:
    """Does this unit, bedded in this slot, make the slot right: a low kerb in S2 or S3; in S1 or S4 a transition
    rising toward the full kerb beside the slot?"""
    kind, toward = L.SLOT_WANTS[slot]
    if L.KIND[name] != kind or not (bedded(f, still) and in_slot(f, slot)):
        return False
    return kind == "low" or f["high_y"] * toward > 0.5


def nearest_slot(f: dict) -> str:
    y = float(f["origin"][1])
    return min(L.SLOTS, key=lambda k: abs(y - L.SLOTS[k]))


def _mm(v: float) -> float:
    return round(float(v) * 1000, 1)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: where each unit was over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = int(round(L.REST_WINDOW_S * control_hz))  # 10 steps at 20 Hz: 11 samples span 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev = None  # each unit's origin at the previous valid control step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects: dict) -> bool:
        """The starting layout: every unit lying flat on the bed on its starting mark."""
        for n in NAMES:
            s = objects[n]
            sx, sy, _ = L.UNIT_START[n]
            if math.hypot(float(s.origin[0]) - sx, float(s.origin[1]) - sy) >= L.START_TOL:
                return False
            f = unit_facts(n, s)
            if not (f["on_bed"] and f["flat"]):
                return False
        return True

    def at_rest(self, name: str) -> bool:
        """Neither tracked point of this unit has moved more than 2 mm in the last 0.5 s (from positions, not the
        reported velocity, which chatters while a part is gripped)."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return False
        now = h[-1]
        return bool(all(float(np.max(np.linalg.norm(p - now, axis=1))) <= L.REST_MOVE + 1e-12 for p in h))

    def update(self, objects: dict) -> Result:
        if not all(valid(objects.get(n)) for n in NAMES):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(objects)
            return self.last
        origins = {n: np.asarray(objects[n].origin, dtype=float) for n in NAMES}
        if self.prev is not None and self.at_start(objects):
            jump = max(float(np.linalg.norm(origins[n] - self.prev[n])) for n in NAMES)
            if jump > L.RESET_JUMP:  # the simulator put every unit back: a new episode
                self.reset()
        self.steps += 1
        for n in NAMES:
            self.hist[n].append(track_points(objects[n]))
        self.prev = origins
        self.last = self.result(objects)
        return self.last

    def result(self, objects: dict) -> Result:
        if not all(valid(objects.get(n)) for n in NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        facts = {n: unit_facts(n, objects[n]) for n in NAMES}
        rest = {n: self.at_rest(n) for n in NAMES}
        slot_rows = {}
        for slot in L.SLOTS:
            units = [n for n in NAMES if right_in(n, facts[n], slot, rest[n])]
            slot_rows[slot] = {"right": bool(units), "units": units, "wants": L.SLOT_WANTS[slot][0]}
        unit_rows = {}
        for n, f in facts.items():
            near = nearest_slot(f)
            rights = [k for k, row in slot_rows.items() if n in row["units"]]
            unit_rows[n] = {
                "kind": L.KIND[n],
                "right_in": rights[0] if rights else None,
                "bedded": bedded(f, rest[n]),
                "at_rest": rest[n],
                "flat": f["flat"],
                "tilt_deg": round(f["tilt_deg"], 2),
                "on_bed": f["on_bed"],
                "low_mm": round((f["low"] - L.TOP) * 1000, 2),
                "back_off_mm": [_mm(v) for v in f["back_off"]],
                "in_line": f["in_line"],
                "face_err_deg": round(f["face_deg"], 2),
                "face_to_road": f["face_to_road"],
                "rises_toward": None if L.KIND[n] == "low" else ("+y" if f["high_y"] > 0.5 else
                                                                  "-y" if f["high_y"] < -0.5 else "neither"),
                "heading_deg": None if f["heading"] is None else round(f["heading"], 1),
                "nearest_slot": near,
                "along_mm": _mm(float(f["origin"][1]) - L.SLOTS[near]),
                "origin_mm": [_mm(f["origin"][0]), _mm(f["origin"][1])],
            }
        n_right = sum(row["right"] for row in slot_rows.values())
        score = 0.25 * n_right
        details = {
            "valid_state": True,
            "slots_right": n_right,
            "zero": "" if n_right else "no slot holds the right unit laid the right way",
            "slots": slot_rows,
            "units": unit_rows,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0), details)
