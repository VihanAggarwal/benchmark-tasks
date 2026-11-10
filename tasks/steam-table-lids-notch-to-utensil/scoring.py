"""The scorer, as plain functions of the four lids' (and the two utensils') states. NumPy only; no simulator needed.

The score is read from where the lids lie, as a cook or an inspector checks a steam table (a top photo, and a ruler
at each corner of each lid): which lid is on which pan, does it lie flat on the rim, is it centred and square on the
pan, and on a pan with a utensil is it a notched lid turned so the handle comes out through the notch. The scorer
keeps only the last half second of each lid's corner positions (to tell a lid at rest from one still moving) and is
fed one observation per control step by ``task.success()``:

    Tracker.update(states) -> Result(score, success, details)

``states`` maps the four lid names to what mechkit reports (``AssemblyState``): ``origin`` (the base frame's origin,
the middle of the plate's outline on its underside) and ``quat_xyzw`` (that frame: local +x along the long side, the
notch at the local -x end, local +z up), with the ObjectState fields. It may also hold the utensils' six fixed parts
(plain ObjectStates; ``pos`` is each box's centre). A state with a non-finite number, a zero quaternion or a lid
without ``origin`` (a simulator blow-up, or a lid not built as an assembly) scores 0 for that step and leaves the
tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

LIDS = L.LID_NAMES
PANS = L.PAN_NAMES
UTENSIL_PARTS = tuple(f"{prefix}_{part}" for prefix, _ in L.UTENSILS.values() for part in L.UTENSIL_PARTS)
_CORNERS = np.array(L.PLATE_CORNERS)  # base frame


def utensil_spawn() -> dict[str, np.ndarray]:
    """Where each utensil part's box centre was set (world)."""
    out = {}
    for pan, (prefix, _) in L.UTENSILS.items():
        y = L.PANS[pan][0]
        boxes = {
            "bowl": (L.BOWL_X0, L.BOWL_X1, y - L.BOWL_W / 2, y + L.BOWL_W / 2, L.BOWL_Z0, L.BOWL_Z1),
            "stem": (L.STEM_X0, L.STEM_X1, y - L.UTENSIL_W / 2, y + L.UTENSIL_W / 2, L.STEM_Z0, L.STEM_Z1),
            "handle": (L.HANDLE_X0, L.HANDLE_X1, y - L.UTENSIL_W / 2, y + L.UTENSIL_W / 2, L.HANDLE_Z0, L.HANDLE_Z1),
        }
        for part, (x0, x1, y0, y1, z0, z1) in boxes.items():
            out[f"{prefix}_{part}"] = np.array([(x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2])
    return out


_UTENSIL_AT = utensil_spawn()


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _finite(obj, keys) -> list | None:
    try:
        parts = [np.asarray(getattr(obj, k), dtype=float).ravel() for k in keys]
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return None
    return parts


def valid_lid(s) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    parts = _finite(s, ("origin", "quat_xyzw", "pos"))
    return bool(parts and parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid(states) -> bool:
    """Every lid valid, and every utensil part that is reported has a finite position."""
    try:
        if not all(valid_lid(states[n]) for n in LIDS):
            return False
        for n in UTENSIL_PARTS:
            if n in states:
                parts = _finite(states[n], ("pos",))
                if not parts or parts[0].size != 3:
                    return False
        return True
    except (KeyError, TypeError):
        return False


# -- one lid ---------------------------------------------------------------------------------------------------------


def lid_corners(s) -> np.ndarray:
    """The plate's four outer corners on its underside, in the world (4, 3)."""
    return np.asarray(s.origin, dtype=float) + _CORNERS @ rotation(s.quat_xyzw).T


def tilt_deg(s) -> float:
    """How far the lid's up axis is from vertical (0 lying flat face up, 180 upside down)."""
    return float(np.degrees(math.acos(max(-1.0, min(1.0, float(rotation(s.quat_xyzw)[2, 2]))))))


def heading_deg(s) -> float | None:
    """The heading of the lid's long axis toward its plain end (local +x), seen from above (degrees, -180..180;
    0 = +x, away from the server). A notched lid's notch then points the opposite way: heading 0 is notch to the
    front (-x), toward the server. None when the long axis stands vertical."""
    a = rotation(s.quat_xyzw)[:, 0]
    if math.hypot(a[0], a[1]) < 1e-6:
        return None
    return float(math.degrees(math.atan2(a[1], a[0])))


def axis_err_deg(heading: float | None) -> float:
    """How far the long side is turned from the pans' long sides (along x), either way round (0..90)."""
    if heading is None:
        return 90.0
    a = abs(heading) % 180.0
    return float(min(a, 180.0 - a))


def notch_err_deg(heading: float | None) -> float:
    """How far a notched lid's notch is turned from facing the front (-x, the utensil's handle) (0..180)."""
    if heading is None:
        return 180.0
    return float(abs(math.remainder(heading, 360.0)))


def nearest_pan(s) -> tuple[str, float]:
    """The pan whose centre is nearest the middle of the lid's plate, seen from above, and that distance (m)."""
    o = np.asarray(s.origin, dtype=float)
    best = min(PANS, key=lambda p: math.hypot(o[0] - L.PAN_X, o[1] - L.PANS[p][0]))
    return best, float(math.hypot(o[0] - L.PAN_X, o[1] - L.PANS[best][0]))


def lid_pose(s, kind: str) -> dict:
    """What a top photo and a ruler at each corner tell about one lid (no rest test): the pan it is over, how far its
    middle is from that pan's centre, how far its long side is turned from the pan's, which way its notch faces, its
    tilt and how far its corners are from the rim's height. ``flat_on_pan`` is the seat test without the rest test:
    tilted 3 degrees or less, every corner within 2 mm of the rim top, centred within 8 mm, square within 6 degrees."""
    corners = lid_corners(s)
    pan, off = nearest_pan(s)
    head = heading_deg(s)
    tilt = tilt_deg(s)
    gap = float(np.max(np.abs(corners[:, 2] - L.RIM_TOP_W)))
    axis = axis_err_deg(head)
    flat = bool(tilt <= L.TILT_MAX_DEG and gap <= L.SEAT_Z_TOL and off <= L.CENTRE_TOL and axis <= L.YAW_TOL_DEG)
    out = {"kind": kind, "pan": pan, "centre_off_mm": round(off * 1000, 2),
           "heading_deg": None if head is None else round(head, 2), "axis_err_deg": round(axis, 2),
           "tilt_deg": round(tilt, 2), "corner_gap_mm": round(gap * 1000, 2), "flat_on_pan": flat}
    if kind == "notched":
        out["notch_err_deg"] = round(notch_err_deg(head), 2)
    return out


def right_lid(pan: str, pose: dict) -> bool:
    """Is this lid, seated on ``pan``, the right one for it? A pan with a utensil: a notched lid with its notch to the
    front (within 6 degrees), where the handle comes out. A pan without: a solid lid, either way round."""
    if L.PANS[pan][1]:
        return pose["kind"] == "notched" and pose["notch_err_deg"] <= L.YAW_TOL_DEG
    return pose["kind"] == "solid"


def utensil_moved(states, ref: dict | None = None) -> dict[str, float]:
    """How far each reported utensil part is from where it was at the start of the trial (m): ``ref`` holds where the
    tracker first saw each part; a part it has not seen is compared with where the layout sets it."""
    out = {}
    for n in UTENSIL_PARTS:
        if n in states:
            at = (ref or {}).get(n, _UTENSIL_AT[n])
            out[n] = float(np.linalg.norm(np.asarray(states[n].pos, dtype=float) - at))
    return out


# -- the tiers -------------------------------------------------------------------------------------------------------


def evaluate(states, rest: dict, ref: dict | None = None) -> tuple[float, str, dict]:
    """(score, why it is not higher, diagnostics) of a valid state, given which lids are at rest and where the utensils'
    parts were at the start of the trial (``ref``; see utensil_moved).

    0 if any part of either utensil is more than 5 mm from where it was set. Otherwise 0.25 for each pan with the
    right lid seated on it (flat, at the rim, centred, square and at rest): a notched lid with its notch to the front on
    pans 1 and 3 (the utensils), a solid lid on pans 2 and 4."""
    d: dict = {}
    moved = utensil_moved(states, ref)
    d["utensil_moved_mm"] = {n: round(v * 1000, 2) for n, v in moved.items()}
    for n in LIDS:
        p = lid_pose(states[n], L.LIDS[n][0])
        p["at_rest"] = bool(rest.get(n, False))
        p["seated"] = bool(p["flat_on_pan"] and p["at_rest"])
        d[n] = p
    if any(v > L.UTENSIL_MOVE_TOL for v in moved.values()):
        return 0.0, "utensil_moved", d
    total, why = 0.0, []
    for pan in PANS:
        on = [n for n in LIDS if d[n]["seated"] and d[n]["pan"] == pan]
        good = [n for n in on if right_lid(pan, d[n])]
        d[pan] = {"seated": on, "right": good, "credit": L.PAN_CREDIT if good else 0.0}
        if good:
            total += L.PAN_CREDIT
        elif not on:
            why.append(f"{pan}_not_covered")
        elif L.PANS[pan][1]:
            why.append(f"{pan}_utensil_without_a_notched_lid_to_the_front")
        else:
            why.append(f"{pan}_notched_lid_on_a_plain_pan")
    return round(total, 2), ",".join(why), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each lid's four plate corners over the last half second, and where
    each utensil part was when the trial began (so a fixed part reported with any constant offset never reads as
    moved)."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {n: deque(maxlen=self.window + 1) for n in LIDS}
        self.prev = None  # every lid's origin at the previous valid step
        self.utensil_ref: dict[str, np.ndarray] = {}  # each utensil part's position when first seen this trial
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: all four lids lying flat on the counters on their starting marks, as they start."""
        for n, (_, x, y, yaw) in L.LIDS.items():
            s = states[n]
            o = np.asarray(s.origin, dtype=float)
            if math.hypot(o[0] - x, o[1] - y) > L.START_TOL:
                return False
            if float(np.max(np.abs(lid_corners(s)[:, 2] - L.TOP))) > 0.003:
                return False
            a = rotation(s.quat_xyzw)[:, 0]
            if float(a[0] * math.cos(yaw) + a[1] * math.sin(yaw)) < math.cos(math.radians(5.0)):
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest any of the lid's four corners has been, over the last 0.5 s, from where it is now (m); None
        until 0.5 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        now = h[-1]
        return float(max(np.max(np.linalg.norm(c - now, axis=1)) for c in h))

    def update(self, states) -> Result:
        if not valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(states)
            return self.last
        now = {n: np.asarray(states[n].origin, dtype=float) for n in LIDS}
        if self.prev is not None and self.at_start(states):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in LIDS)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        for n in UTENSIL_PARTS:
            if n in states and n not in self.utensil_ref:
                self.utensil_ref[n] = np.asarray(states[n].pos, dtype=float).copy()
        for n in LIDS:
            self.history[n].append(lid_corners(states[n]))
        self.prev = now
        self.last = self.result(states)
        return self.last

    def result(self, states) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not valid(states):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        moved = {n: self.moved(n) for n in LIDS}
        rest = {n: (m is not None and m <= L.STILL_TOL) for n, m in moved.items()}
        score, reason, d = evaluate(states, rest, self.utensil_ref)
        details = {"valid_state": True, "zero_reason": reason,
                   "moved_mm": {n: (None if m is None else round(m * 1000, 2)) for n, m in moved.items()}, **d}
        return Result(score, bool(score == 1.0), details)
