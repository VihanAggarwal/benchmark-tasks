"""The scorer, as plain functions of the four snap traps' states. NumPy only; no simulator needed.

A trap is set at a droppings patch the way pest controllers set one: lying base down on the floor, square to the
wall (its long side pointing at the wall), with its pedal end (the trigger) toward the wall and nearly touching it,
at the patch. Each of the four patches earns 0.25 when a trap is set at it. The score is read from the state at
hand (the last step); the scorer keeps only a short history of where each trap was, to tell whether it is still,
and is fed one observation per control step by ``task.success()``:

    Tracker.update({"trap_1": state, ..., "trap_4": state}) -> Result(score, success, details)

A trap's state is a mechkit ``AssemblyState`` (``origin``: its frame's origin, the centre of the base's underside;
``quat_xyzw``: its frame's orientation; and the usual ``pos``, ``lin_vel``, ``ang_vel``, ``low``, ``high``), or
anything with those attributes. The trap's shape (layout.py) gives every point of it from ``origin`` and
``quat_xyzw``. A missing trap, or a state with a non-finite number, a zero quaternion or no ``origin`` (a simulator
blow-up, or a trap not built as a compound), scores 0 for that step and leaves the tracker as it was. The traps are
alike: names are labels only, and any trap may be set at any patch.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

NAMES = L.TRAPS
_KEYS = ("origin", "pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)], dtype=float)
BASE_CORNERS = _SIGNS * np.array([L.TRAP_HALF_LEN, L.TRAP_W / 2, L.TRAP_BASE_H])  # (8, 3), in the trap's frame


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
    NaN or zeros; a trap that was not built as a compound has no origin)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 3 and parts[2].size == 4 and np.linalg.norm(parts[2]) > 0.5)


def frame(s) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the trap's frame in the world."""
    return rotation(s.quat_xyzw), np.asarray(s.origin, dtype=float)


def corners(s) -> np.ndarray:
    """(8, 3): the corners of the trap's wooden base in the world (it carries everything else, inside its outline)."""
    r, o = frame(s)
    return BASE_CORNERS @ r.T + o


def pedal_end_mid(s) -> np.ndarray:
    """The midpoint of the base's end face at the pedal end, in the world."""
    r, o = frame(s)
    return o + r @ np.asarray(L.PEDAL_END, dtype=float)


def tilt_deg(r: np.ndarray) -> float:
    """Angle between the vertical and the base's up axis (its pedal side), 0..180 degrees: 180 is upside down."""
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2]))))))


def pedal_heading(r: np.ndarray) -> np.ndarray | None:
    """Unit vector, seen from above, from the trap's plain end toward its pedal end; None for a trap standing on
    an end (no heading)."""
    u = -np.array([r[0, 0], r[1, 0]], dtype=float)
    n = float(np.linalg.norm(u))
    return None if n < 1e-6 else u / n


def wall_angle_deg(r: np.ndarray, wall: str) -> float:
    """How far (0..180 degrees, seen from above) the pedal end points away from straight at this wall."""
    u = pedal_heading(r)
    if u is None:
        return 90.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(u[1]) * L.WALLS[wall])))))


def end_gap(s, wall: str) -> float:
    """From the wall's face to the trap's nearest point (m; negative: past the face)."""
    k = L.WALLS[wall]
    return float(L.FACE_Y - np.max(k * corners(s)[:, 1]))


def track_points(s) -> np.ndarray:
    """(2, 3): the trap's origin and its pedal end's midpoint, whose movement says whether it is still (both catch
    a slide, the second also a turn on the spot)."""
    return np.array([np.asarray(s.origin, dtype=float), pedal_end_mid(s)])


def trap_facts(s) -> dict:
    """Everything about one trap's pose that the rule needs (stillness is the tracker's)."""
    r, o = frame(s)
    c = corners(s)
    low = float(c[:, 2].min())
    tilt = tilt_deg(r)
    angles = {w: wall_angle_deg(r, w) for w in L.WALLS}
    faces = min(angles, key=angles.get)
    mid = pedal_end_mid(s)
    return {
        "origin": o,
        "tilt_deg": tilt,
        "base_down": bool(tilt <= L.TILT_MAX_DEG + 1e-9),
        "low": low,
        "on_floor": bool(abs(low - L.FLOOR_TOP) <= L.FLOOR_TOL + 1e-12),
        "angles": angles,
        "faces": faces,
        "gaps": {w: end_gap(s, w) for w in L.WALLS},
        "end_mid": mid,
    }


def set_at(f: dict, patch: str, still: bool) -> bool:
    """Is this trap (its facts) set at this patch: placed (still, base down, its bottom within 3 mm of the floor),
    its pedal end pointing within 15 degrees of straight at the patch's wall, that end's nearest point 0-8 mm from
    the wall's face (3 mm of contact overlap counts as touching) and its midpoint within 25 mm of the patch's
    centre, measured along the wall?"""
    px, wall = L.PATCHES[patch]
    return bool(still and f["base_down"] and f["on_floor"]
                and f["angles"][wall] <= L.SQUARE_MAX_DEG + 1e-9
                and L.END_GAP_MIN - 1e-12 <= f["gaps"][wall] <= L.END_GAP_MAX + 1e-12
                and abs(float(f["end_mid"][0]) - px) <= L.ALONG_MAX + 1e-12)


def nearest_patch(f: dict) -> str:
    """The patch, on the wall the trap's pedal end points nearer to, nearest its pedal end along that wall."""
    on_wall = [p for p, (_, w) in L.PATCHES.items() if w == f["faces"]]
    return min(on_wall, key=lambda p: abs(float(f["end_mid"][0]) - L.PATCHES[p][0]))


def _mm(v: float) -> float:
    return round(float(v) * 1000, 1)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: where each trap was over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = int(round(L.REST_WINDOW_S * control_hz))  # 10 steps at 20 Hz: 11 samples span 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev = None  # each trap's origin at the previous valid control step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects: dict) -> bool:
        """The starting layout: every trap lying base down on the floor on its starting mark."""
        for n in NAMES:
            s = objects[n]
            sx, sy, _ = L.TRAP_START[n]
            if math.hypot(float(s.origin[0]) - sx, float(s.origin[1]) - sy) >= L.START_TOL:
                return False
            f = trap_facts(s)
            if not (f["on_floor"] and f["base_down"]):
                return False
        return True

    def at_rest(self, name: str) -> bool:
        """Neither tracked point of this trap has moved more than 2 mm in the last 0.5 s (from positions, not the
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
            if jump > L.RESET_JUMP:  # the simulator put every trap back: a new episode
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
        facts = {n: trap_facts(objects[n]) for n in NAMES}
        rest = {n: self.at_rest(n) for n in NAMES}
        patch_rows = {}
        for p in L.PATCHES:
            traps = [n for n in NAMES if set_at(facts[n], p, rest[n])]
            patch_rows[p] = {"set": bool(traps), "traps": traps}
        trap_rows = {}
        for n, f in facts.items():
            near = nearest_patch(f)
            wall = f["faces"]
            sets = [p for p, row in patch_rows.items() if n in row["traps"]]
            u = pedal_heading(rotation(objects[n].quat_xyzw))
            trap_rows[n] = {
                "set_at": sets[0] if sets else None,
                "placed": bool(rest[n] and f["base_down"] and f["on_floor"]),
                "at_rest": rest[n],
                "base_down": f["base_down"],
                "tilt_deg": round(f["tilt_deg"], 2),
                "on_floor": f["on_floor"],
                "low_mm": round((f["low"] - L.FLOOR_TOP) * 1000, 2),
                "pedal_heading_deg": None if u is None else round(math.degrees(math.atan2(u[1], u[0])), 1),
                "faces": wall,
                "square_err_deg": round(f["angles"][wall], 2),
                "end_gap_mm": _mm(f["gaps"][wall]),
                "nearest_patch": near,
                "along_mm": _mm(float(f["end_mid"][0]) - L.PATCHES[near][0]),
                "origin_mm": [_mm(f["origin"][0]), _mm(f["origin"][1])],
            }
        n_set = sum(row["set"] for row in patch_rows.values())
        score = 0.25 * n_set
        details = {
            "valid_state": True,
            "patches_set": n_set,
            "zero": "" if n_set else "no trap set at any patch",
            "patches": patch_rows,
            "traps": trap_rows,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0), details)
