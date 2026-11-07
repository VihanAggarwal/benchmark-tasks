"""The scorer, as plain functions of the nozzles' states. NumPy only; no simulator needed.

The rule is about the end state: every one of the six nozzle bodies on the boom must hold one size 02 (yellow, by
ISO 10625) nozzle standing upright in it. The four bodies that need work earn 0.25 each; the two that start with
an 02 must still hold one; a wrong size left in any body caps the score at 0.5, a nozzle left anywhere but a body,
a parts tray or a spares pad (or a removed one put back among the spares) caps it at 0.75.

    Tracker.update(states) -> Result(score, success, details)

``states`` maps each nozzle's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. The tracker keeps only the last 0.5 s of each
nozzle's path, for 'still'. A state with a non-finite number, a zero quaternion or a missing nozzle (a simulator
blow-up) scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # control steps in the 0.5 s stillness window


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


def all_valid(states) -> bool:
    """Every nozzle present and valid."""
    try:
        return all(valid(states[n]) for n in L.NOZZLE_NAMES)
    except (KeyError, TypeError):
        return False


def axis(s) -> np.ndarray:
    """The nozzle's axis (unit vector, world frame)."""
    return rotation(s.quat_xyzw)[:, 2]


def tilt_deg(s) -> float:
    """Angle between the nozzle's axis and the vertical (degrees, 0..90; either end down)."""
    return float(np.degrees(np.arccos(np.clip(abs(axis(s)[2]), 0.0, 1.0))))


def lowest(s) -> float:
    """Height of the nozzle's lowest point (world z), from its pose: a cylinder of radius r and height h."""
    a = axis(s)
    return float(s.pos[2] - (L.NOZZLE_H / 2) * abs(a[2]) - L.NOZZLE_R * math.sqrt(max(0.0, 1.0 - a[2] ** 2)))


def size_of(name: str) -> str:
    """The nozzle's ISO 10625 size, from its colour."""
    return L.SIZE_OF[L.COLOUR[name]]


def place_of(s) -> tuple[str, str]:
    """Where a nozzle is, by its centre seen from above and its lowest point. Exactly one of:
    ('off_table', '') its centre below the table's top or outside the table;
    ('body', name) its centre over the body's 32 x 32 mm opening, lowest point less than 40 mm above the boom top;
    ('tray', side) its centre over the tray's 90 x 90 mm inside, lowest point less than 40 mm above the tray floor;
    ('pad', side) its centre over the 100 x 110 mm pad, lowest point less than 40 mm above the pad's top;
    ('elsewhere', '') anywhere else (on the table, lying on the boom, still held up in the air)."""
    x, y, z = (float(v) for v in s.pos)
    if z < L.TOP or not (L.TABLE_X[0] <= x <= L.TABLE_X[1] and L.TABLE_Y[0] <= y <= L.TABLE_Y[1]):
        return ("off_table", "")
    low = lowest(s)
    half = L.BODY_OPEN / 2
    for b, by in L.BODIES.items():
        if abs(x - L.BOOM_X) <= half and abs(y - by) <= half and low < L.BOOM_TOP + L.HOLD_ABOVE:
            return ("body", b)
    half = L.TRAY_IN / 2
    for side, ty in L.TRAYS.items():
        if abs(x - L.TRAY_X) <= half and abs(y - ty) <= half and low < L.TRAY_FLOOR_TOP + L.HOLD_ABOVE:
            return ("tray", side)
    hx, hy = L.PAD_SIZE[0] / 2, L.PAD_SIZE[1] / 2
    for side, py in L.PADS.items():
        if abs(x - L.PAD_X) <= hx and abs(y - py) <= hy and low < L.PAD_TOP + L.HOLD_ABOVE:
            return ("pad", side)
    return ("elsewhere", "")


def seated(s) -> bool:
    """Standing upright on the boom's top: axis within 10 degrees of the vertical, lowest point within 3 mm of the
    boom's top. (Stillness is the tracker's part of 'fitted'.)"""
    return bool(tilt_deg(s) <= L.FIT_TILT_DEG and abs(lowest(s) - L.BOOM_TOP) <= L.FIT_BOTTOM)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v: float) -> float:
    return round(float(v) * 1000, 1)


class Tracker:
    """What has happened so far in this trial: the last 0.5 s of each nozzle's path (for 'still')."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.episodes = 0  # new episodes spotted by a teleport back to the start (not cleared by reset)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.paths: dict[str, deque] = {n: deque(maxlen=_WINDOW + 1) for n in L.NOZZLE_NAMES}  # recent centres
        self.bad_steps = 0  # control steps with a non-finite or missing nozzle state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # {name: (centre, velocity)} at the previous valid control step

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: every nozzle upright near its own starting mark, at its starting height, slow."""
        for n, (x, y) in L.STARTS.items():
            s = states[n]
            if math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) >= L.START_TOL:
                return False
            if tilt_deg(s) >= 5.0 or abs(lowest(s) - L.start_bottom(n)) >= 0.012:
                return False
            if np.linalg.norm(np.asarray(s.lin_vel, dtype=float)) >= 0.05:
                return False
        return True

    def _teleported(self, states) -> bool:
        """Some nozzle moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        for n in L.NOZZLE_NAMES:
            p0, v0 = self.prev[n]
            c = np.asarray(states[n].pos, dtype=float)
            if float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall:
                return True
        return False

    def update(self, states) -> Result:
        if not all_valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(states)
            return self.last
        if self.prev is not None and self._teleported(states) and self.at_start(states):
            self.reset()  # the simulator put the nozzles back at the start: a new episode
            self.episodes += 1
        self.prev = {n: (np.array(states[n].pos, dtype=float), np.array(states[n].lin_vel, dtype=float))
                     for n in L.NOZZLE_NAMES}
        for n in L.NOZZLE_NAMES:
            self.paths[n].append(np.array(states[n].pos, dtype=float))
        self.steps += 1
        self.last = self.result(states)
        return self.last

    def path(self, name: str) -> float:
        """How far the nozzle's centre has moved (path length) over the last 0.5 s; infinite before 0.5 s of
        observations exist."""
        p = self.paths[name]
        if len(p) < _WINDOW + 1:
            return math.inf
        return float(np.linalg.norm(np.diff(np.array(p), axis=0), axis=1).sum())

    def still(self, name: str) -> bool:
        """Moved no more than 2 mm (path length) over the last 0.5 s."""
        return self.path(name) <= L.STILL_PATH

    def fitted(self, name: str, s) -> bool:
        """In a body (see place_of), standing upright on the boom's top and still."""
        return bool(place_of(s)[0] == "body" and seated(s) and self.still(name))

    def result(self, states) -> Result:
        if not all_valid(states):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        places = {n: place_of(states[n]) for n in L.NOZZLE_NAMES}
        held: dict[str, list[str]] = {b: [] for b in L.BODY_NAMES}
        for n, (kind, where) in places.items():
            if kind == "body":
                held[where].append(n)
        off_table = [n for n, p in places.items() if p[0] == "off_table"]
        doubled = [b for b in L.BODY_NAMES if len(held[b]) >= 2]
        good = {b: (len(held[b]) == 1 and size_of(held[b][0]) == L.TARGET_SIZE
                    and self.fitted(held[b][0], states[held[b][0]])) for b in L.BODY_NAMES}
        lost_keep = [b for b in L.KEEP if not good[b]]
        mixed = [b for b in L.BODY_NAMES if any(size_of(n) != L.TARGET_SIZE for n in held[b])]
        loose = [n for n, p in places.items() if p[0] == "elsewhere"]
        back_in_spares = [n for n in L.OFF_THE_BOOM if places[n][0] == "pad"]
        points = 0.25 * sum(good[b] for b in L.WORK)
        cap = 1.0
        if mixed:
            cap = min(cap, L.CAP_MIXED)
        if loose or back_in_spares:
            cap = min(cap, L.CAP_LOOSE)
        if off_table or doubled or lost_keep:
            score = 0.0
        else:
            score = min(points, cap)
        success = bool(score == 1.0)
        details = {
            "valid_state": True,
            "points": points,
            "cap": cap,
            "done": [b for b in L.WORK if good[b]],
            "holds": {b: held[b] for b in L.BODY_NAMES},
            "fitted_02": [b for b in L.BODY_NAMES if good[b]],
            "wrong_size_in": mixed,
            "two_in_one": doubled,
            "keep_lost": lost_keep,
            "off_table": off_table,
            "loose": loose,
            "removed_on_pad": back_in_spares,
            "places": {n: (f"{k}_{w}" if w else k) for n, (k, w) in places.items()},
            "tilt_deg": {n: round(tilt_deg(states[n]), 1) for n in L.NOZZLE_NAMES},
            "bottom_mm": {n: _mm(lowest(states[n]) - L.TOP) for n in L.NOZZLE_NAMES},
            "path_mm": {n: (_mm(self.path(n)) if math.isfinite(self.path(n)) else None) for n in L.NOZZLE_NAMES},
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
