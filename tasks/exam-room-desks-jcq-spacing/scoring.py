"""The scorer, as plain functions of the four desks' states. NumPy only; no simulator needed.

The rule is the seating rule exams officers and invigilators in the UK set rooms out by (the Joint Council for
Qualifications' Instructions for Conducting Examinations): every candidate facing the same way, and at least 1.25 m
between the centres of neighbouring desks in every direction; rooms usually meet the first part by facing the front
(the clock and the exam times), which this task asks for. All of it is read from the desks' poses, as a top photo with a tape and a protractor would read it. The only memory the
scorer keeps is each desk's last half second of positions, to tell whether it is still.

    Tracker.update(desks) -> Result(score, success, details)

``desks`` maps each name in ``layout.DESKS`` to anything with ``origin`` (the desk frame's origin: on the floor under
the middle of the desk top), ``quat_xyzw``, ``pos`` (bounding-box centre), ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as mechkit's ``AssemblyState`` has. A missing desk or a state with a non-finite
number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was. The four
desks are alike: names are labels only, and any desk may end anywhere.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 steps: the path is summed over the last WINDOW moves
FIELDS = ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")


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


def valid(d) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [np.asarray(getattr(d, k), dtype=float).ravel() for k in FIELDS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and parts[5].size == 3 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(d) -> float:
    """Angle between the desk's up axis and the vertical (degrees, 0..180: upside down is 180)."""
    zz = rotation(d.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def facing(d) -> tuple[float, float] | None:
    """The way the candidate faces (the desk frame's +u, from the chair to the desk) seen from above, as a unit 2-D
    vector; None if it points straight up or down."""
    a = rotation(d.quat_xyzw)[:, 0]
    n = math.hypot(float(a[0]), float(a[1]))
    if n < 1e-9:
        return None
    return float(a[0]) / n, float(a[1]) / n


def heading_rad(d) -> float:
    """Which way the candidate faces seen from above (rad, counter-clockwise from +x); NaN if straight up or down."""
    f = facing(d)
    return math.atan2(f[1], f[0]) if f is not None else float("nan")


def off_front_deg(d) -> float:
    """How far the facing direction is turned from the front (+x, square to the front wall), seen from above
    (degrees, 0..180: 180 is facing the back). NaN if it points straight up or down."""
    h = heading_rad(d)
    if not math.isfinite(h):
        return float("nan")
    return abs(math.degrees(math.atan2(math.sin(h), math.cos(h))))


def desk_top_centre(d) -> np.ndarray:
    """Where the middle of the desk top is (world x, y, z): the point the spacing is measured between."""
    return np.asarray(d.origin, dtype=float) + rotation(d.quat_xyzw) @ np.asarray(L.DESK_TOP_CENTRE, dtype=float)


def lowest(d) -> float:
    """Height of its lowest point above the floor (m)."""
    return float(np.asarray(d.low, dtype=float)[2] - (L.TOP + L.FLOOR))


def on_floor_area(d) -> bool:
    """Its desk-top centre over the grey floor, seen from above (edges count)."""
    x, y = (float(v) for v in desk_top_centre(d)[:2])
    return bool(L.ROOM_X[0] <= x <= L.ROOM_X[1] and L.ROOM_Y[0] <= y <= L.ROOM_Y[1])


def nearest(name: str, centres: dict) -> tuple[float, str | None]:
    """The top-view distance (m) from this desk's desk-top centre to the nearest other desk's, and which desk."""
    best, who = math.inf, None
    for other, c in centres.items():
        if other == name:
            continue
        dist = float(np.hypot(*(centres[name][:2] - c[:2])))
        if dist < best:
            best, who = dist, other
    return best, who


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each desk's last half second, for stillness, and a step counter. Everything else is read from the state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {name: deque(maxlen=WINDOW + 1) for name in L.DESKS}  # (origin, heading) per valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(name: str, d) -> bool:
        """The starting layout for this desk: upright on the floor on its mark, facing as it starts."""
        x, y, h = L.START[name]
        on_mark = float(np.hypot(d.origin[0] - x, d.origin[1] - y)) < L.START_TOL
        hd = heading_rad(d)
        turn = abs(math.remainder(hd - h, 2 * math.pi)) if math.isfinite(hd) else math.pi
        return bool(on_mark and turn < math.radians(10.0) and tilt_deg(d) < 2.0 and abs(lowest(d)) < L.FLOOR_TOL)

    def path(self, name: str) -> float:
        """How far the desk moved over the last 0.5 s: its origin's path plus its turns about the vertical counted at
        its outline's farthest corner (m). Infinite until half a second has been seen."""
        h = list(self.history[name])
        if len(h) < WINDOW + 1:
            return math.inf
        total = 0.0
        for (p0, a0), (p1, a1) in zip(h, h[1:]):
            turn = abs(math.remainder(a1 - a0, 2 * math.pi)) if math.isfinite(a0) and math.isfinite(a1) else math.pi
            total += float(np.linalg.norm(p1 - p0)) + L.turn_radius() * turn
        return total

    def update(self, desks: dict) -> Result:
        self.steps += 1
        ds = {n: (desks or {}).get(n) for n in L.DESKS}
        if not all(valid(d) for d in ds.values()):  # a blow-up: score 0 for this step, keep everything as it was
            self.last = self.result(desks)
            return self.last
        if all(self.history[n] for n in L.DESKS) and all(self.at_start(n, d) for n, d in ds.items()):
            jump = max(float(np.linalg.norm(np.asarray(d.origin, dtype=float) - self.history[n][-1][0]))
                       for n, d in ds.items())
            if jump > L.RESET_JUMP:  # the simulator put them back: a new episode
                self.reset()
                self.steps = 1
        for n, d in ds.items():
            self.history[n].append((np.asarray(d.origin, dtype=float).copy(), heading_rad(d)))
        self.last = self.result(desks)
        return self.last

    def _set(self, name: str, d) -> tuple[str | None, dict]:
        """None if the desk is set (upright, on the floor, its desk-top centre over the floor, still), else why not."""
        tilt, low, path = tilt_deg(d), lowest(d), self.path(name)
        c = desk_top_centre(d)
        out = {"tilt_deg": round(tilt, 2), "lowest_mm": round(low * 1000, 1),
               "path_mm": round(path * 1000, 2) if math.isfinite(path) else None,
               "origin_mm": [round(float(v) * 1000, 1) for v in np.asarray(d.origin, dtype=float)],
               "centre_mm": [round(float(v) * 1000, 1) for v in c[:2]]}
        if tilt > L.TILT_MAX_DEG:
            return "not_upright", out
        if abs(low) > L.FLOOR_TOL:
            return "not_on_the_floor", out
        if not on_floor_area(d):
            return "centre_off_the_floor", out
        if path >= L.STILL_PATH:
            return "moving", out
        return None, out

    def desk_result(self, name: str, d, centres: dict) -> dict:
        """One desk: set or not, which way it faces and how far that is off the front, its nearest neighbour and
        whether it is credited (set, facing the front within 15 degrees, at least 125 mm from every other desk)."""
        why, out = self._set(name, d)
        off = off_front_deg(d)
        dist, who = nearest(name, centres)
        hd = heading_rad(d)
        out.update(heading_deg=round(math.degrees(hd), 2) if math.isfinite(hd) else None,
                   off_front_deg=round(off, 2) if math.isfinite(off) else None,
                   faces_front=bool(math.isfinite(off) and off <= L.FACING_DEG + 1e-9),
                   nearest_mm=round(dist * 1000, 1) if math.isfinite(dist) else None, nearest_desk=who,
                   spaced=bool(dist >= L.SPACING_MIN - 1e-12), set=why is None)
        if why is None:
            if not out["faces_front"]:
                why = "not_facing_the_front"
            elif not out["spaced"]:
                why = "closer_than_125_mm_to_another_desk"
        out.update(credited=why is None, why=why)
        return out

    def result(self, desks) -> Result:
        ds = {n: (desks or {}).get(n) for n in L.DESKS}
        if not all(valid(d) for d in ds.values()):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        centres = {n: desk_top_centre(d) for n, d in ds.items()}  # every desk takes up its place, set or not
        per = {n: self.desk_result(n, d, centres) for n, d in ds.items()}
        credited = [n for n in L.DESKS if per[n]["credited"]]
        score = L.CREDIT * len(credited)
        details = {"valid_state": True, "zero_reason": None if credited else "no_desk_set_facing_and_spaced",
                   "credited": credited, **per}
        return Result(score, bool(score == 1.0), details)
