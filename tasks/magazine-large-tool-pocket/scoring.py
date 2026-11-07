"""The scorer, as plain functions of the six tools' states. NumPy only; no simulator needed.

The score is read from where the tools stand at the end, as a setter checks a magazine with a look from above and a
rule: which pockets hold a tool, is each tool down on the plate and upright in its socket, is anything left lying on
the magazine. The scorer keeps only each tool's last half second of positions (to tell a tool at rest from one still
moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(objects) -> Result(score, success, details)

``objects`` maps the tool names in ``layout.TOOL_NAMES`` to their states. The five small tools are plain cylinders:
``pos`` (their centre), ``quat_xyzw``. The face mill is a mechkit compound (``AssemblyState``): ``origin`` (the centre
of its shank's bottom face) and ``quat_xyzw`` (local +z up its axis). A step on which any tool's state has a
non-finite number, a zero quaternion or (for the face mill) no ``origin`` scores 0 and leaves the tracker as it was.

A small tool is the same shape either end down, so its 'bottom' is whichever end is lower; the face mill has a
bottom (its shank) and is upside down when its cutter body is underneath.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

DISC_TOP_W = L.TOP + L.DISC_TOP  # the magazine plate's top (world z)


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


def valid(name: str, st) -> bool:
    """Every number the scorer reads is finite and the quaternion is not degenerate; the face mill also needs its
    base frame's origin (it is missing if the face mill was not built as a compound)."""
    keys = ("origin", "quat_xyzw") if name == L.FACE else ("pos", "quat_xyzw")
    try:
        parts = [np.asarray(getattr(st, k), dtype=float).ravel() for k in keys]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def axis(name: str, st) -> tuple[np.ndarray, np.ndarray, float]:
    """(bottom, top, tilt): the centres of the tool's bottom and top faces in the world and its tilt from upright
    (degrees). A small tool's bottom is its lower end (tilt 0..90); the face mill's is its shank's (tilt 0..180:
    upside down is 180)."""
    a = rotation(st.quat_xyzw)[:, 2]
    if name == L.FACE:
        bottom = np.asarray(st.origin, dtype=float)
        return bottom, bottom + a * L.FM_H, float(np.degrees(np.arccos(np.clip(a[2], -1.0, 1.0))))
    if a[2] < 0.0:
        a = -a
    c = np.asarray(st.pos, dtype=float)
    return c - a * (L.TOOL_H / 2), c + a * (L.TOOL_H / 2), float(np.degrees(np.arccos(np.clip(a[2], -1.0, 1.0))))


def lowest(name: str, st) -> float:
    """World height of the tool's lowest point: the lowest point of the lowest rim of its outline."""
    a = rotation(st.quat_xyzw)[:, 2]
    side = math.sqrt(max(0.0, 1.0 - float(a[2]) ** 2))
    if name == L.FACE:
        o = np.asarray(st.origin, dtype=float)
        return float(min(o[2] + a[2] * s - r * side for s, r in L.FM_RIMS))
    bottom, _, _ = axis(name, st)
    return float(bottom[2] - L.TOOL_R * side)


_DIRS = np.stack([np.cos(np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)),
                  np.sin(np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False))], axis=1)  # every 0.5 degrees


def shadow_gap(p0, p1, r: float, c) -> float:
    """Seen from above, the distance from the point ``c`` to the shadow of a cylinder of radius ``r`` whose end faces
    are centred at ``p0`` and ``p1`` (3-D); negative when ``c`` is inside the shadow. The cylinder is the convex hull
    of its two end discs, so its shadow is the convex hull of their two shadows (ellipses); the distance is read from
    the support function, max over horizontal directions u of u.c - h(u), sampled every 0.5 degrees."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    a = p1 - p0
    n = float(np.linalg.norm(a))
    a = a / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    ua = _DIRS @ a[:2]
    h = np.maximum(_DIRS @ p0[:2], _DIRS @ p1[:2]) + r * np.sqrt(np.clip(1.0 - ua * ua, 0.0, 1.0))
    return float(np.max(_DIRS @ np.asarray(c, dtype=float)[:2] - h))


def over_disc(name: str, st) -> bool:
    """Some part of the tool is over the magazine plate seen from above: its shadow (end faces and sides) overlaps the
    plate's circle by more than 1 mm. The face mill is its two cylinders, shank and cutter body."""
    c = np.array([L.MAG_X, L.MAG_Y])
    bottom, top, _ = axis(name, st)
    if name == L.FACE:
        neck = bottom + (top - bottom) * (L.SHANK_H / L.FM_H)
        gap = min(shadow_gap(bottom, neck, L.SHANK_R, c), shadow_gap(neck, top, L.BODY_R, c))
    else:
        gap = shadow_gap(bottom, top, L.TOOL_R, c)
    return bool(gap < L.DISC_R - 0.001)


def pocket_of(point) -> int | None:
    """The pocket whose 52 mm square opening the point lies inside, seen from above (None if none)."""
    for k, (px, py) in L.POCKETS.items():
        if abs(point[0] - px) <= L.SEAT_HALF and abs(point[1] - py) <= L.SEAT_HALF:
            return k
    return None


def pose(name: str, st) -> dict:
    """Everything the score reads from one tool's pose (no history)."""
    bottom, top, tilt = axis(name, st)
    k = pocket_of(bottom)
    low = lowest(name, st)
    floor = float(bottom[2] - DISC_TOP_W)
    off = None if k is None else float(math.hypot(bottom[0] - L.POCKETS[k][0], bottom[1] - L.POCKETS[k][1]))
    return {
        "bottom": bottom,
        "top": top,
        "tilt_deg": tilt,
        "pocket": k,
        "floor": floor,  # the bottom face's centre above the plate's top (m)
        "centre_off": off,  # the bottom face's centre from its pocket's centre, horizontally (m)
        "low": low,
        "over_disc": over_disc(name, st),
        "down_in_pocket": bool(k is not None and tilt <= L.SEAT_TILT_DEG and abs(floor) <= L.SEAT_FLOOR_TOL),
    }


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: every tool's axis end points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: dict[str, deque] = {n: deque(maxlen=self.window + 1) for n in L.TOOL_NAMES}
        self.prev_bottom: dict[str, np.ndarray] = {}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout for the three new tools: each standing upright on its tray mark."""
        for name in (*L.NEW_SMALL, L.FACE):
            st = objects[name]
            bottom, _, tilt = axis(name, st)
            mark = L.FM_START if name == L.FACE else L.NEW_START[name]
            if not (math.hypot(bottom[0] - mark[0], bottom[1] - mark[1]) < L.START_TOL
                    and abs(bottom[2] - (L.TOP + L.TRAY_TOP)) < 0.003 and tilt < 2.0):
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest either end of the tool's axis has been, over the last 0.5 s, from where it is now (m); None
        until 0.5 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        b, t = h[-1]
        return float(max(max(np.linalg.norm(b0 - b), np.linalg.norm(t0 - t)) for b0, t0 in h))

    def update(self, objects) -> Result:
        if not all(valid(n, objects.get(n)) for n in L.TOOL_NAMES):  # a blow-up: score 0, keep everything
            self.steps += 1
            self.last = self.result(objects)
            return self.last
        ends = {n: axis(n, objects[n])[:2] for n in L.TOOL_NAMES}
        if self.prev_bottom and self.at_start(objects):
            jumped = any(float(np.linalg.norm(ends[n][0] - self.prev_bottom[n])) > L.RESET_JUMP
                         for n in (*L.NEW_SMALL, L.FACE))
            if jumped:  # the simulator put the new tools back on their marks: a new episode
                self.reset()
        self.steps += 1
        for n in L.TOOL_NAMES:
            self.history[n].append(ends[n])
            self.prev_bottom[n] = ends[n][0]
        self.last = self.result(objects)
        return self.last

    def result(self, objects) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        bad = [n for n in L.TOOL_NAMES if not valid(n, objects.get(n))]
        if bad:
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state", "invalid": bad})
        poses = {n: pose(n, objects[n]) for n in L.TOOL_NAMES}
        moved = {n: self.moved(n) for n in L.TOOL_NAMES}
        rest = {n: moved[n] is not None and moved[n] <= L.STILL_TOL for n in L.TOOL_NAMES}
        seated = {n: poses[n]["down_in_pocket"] and rest[n] for n in L.TOOL_NAMES}
        seated_in = {n: poses[n]["pocket"] if seated[n] else None for n in L.TOOL_NAMES}

        reasons = []
        # (a) P1, P2 and P5 must each still hold one of the three tools loaded there, seated. The three look alike,
        # so which of them stands in which of those pockets does not matter (a photo cannot tell a swap).
        loaded_seated = {seated_in[n] for n in L.LOADED}
        loaded_missing = [k for k in sorted(set(L.LOADED.values())) if k not in loaded_seated]
        if loaded_missing:
            reasons.append("loaded_pocket_without_a_loaded_tool")
        # (b) a tool on the magazine (over the plate, lowest point under 75 mm above it) that is not seated
        loose = [n for n in L.TOOL_NAMES
                 if poses[n]["over_disc"] and poses[n]["low"] < DISC_TOP_W + L.ON_MAGAZINE_H and not seated[n]]
        if loose:
            reasons.append("tool_loose_on_the_magazine")
        # (c) the face mill seated next to a pocket that holds a seated tool
        fk = seated_in[L.FACE]
        crowded = []
        if fk is not None:
            crowded = [k for k in L.neighbours(fk) if any(seated_in[n] == k for n in L.TOOL_NAMES if n != L.FACE)]
            if crowded:
                reasons.append("face_mill_next_to_a_tool")

        face_ok = fk is not None and not crowded
        small_ok = [n for n in L.NEW_SMALL if seated_in[n] in L.SMALL_POCKETS]
        tier = 0.5 * face_ok + 0.25 * len(small_ok)
        score = 0.0 if reasons else tier
        details = {
            "valid_state": True,
            "tier": tier,
            "zero_reason": ",".join(reasons),
            "face_mill_pocket": fk,
            "face_mill_neighbours_with_tools": crowded,
            "small_tools_counted": small_ok,
            "loaded_pockets_missing": loaded_missing,
            "loose_on_magazine": loose,
            "pockets": {n: poses[n]["pocket"] for n in L.TOOL_NAMES},
            "seated": seated_in,
            "tilt_deg": {n: round(poses[n]["tilt_deg"], 2) for n in L.TOOL_NAMES},
            "floor_mm": {n: round(poses[n]["floor"] * 1000, 2) for n in L.TOOL_NAMES},
            "centre_off_mm": {n: None if poses[n]["centre_off"] is None else round(poses[n]["centre_off"] * 1000, 2)
                              for n in L.TOOL_NAMES},
            "low_mm": {n: round((poses[n]["low"] - DISC_TOP_W) * 1000, 2) for n in L.TOOL_NAMES},
            "moved_mm": {n: None if moved[n] is None else round(moved[n] * 1000, 2) for n in L.TOOL_NAMES},
        }
        return Result(score, bool(score == 1.0), details)
