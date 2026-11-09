"""The scorer, as plain functions of the five standoffs' and the board's states. NumPy only; no simulator needed.

The score is read from the end state, as a builder checks a case tray before and after mounting a board: which of
the board's four holes has a standoff under it, whether any standoff is left in the case other than on those four
holes or on the spare hole A3 clear of the board (under the board where it has no hole, or loose on the tray), and
whether the board lies on its four standoffs with each hole over its own. The scorer keeps only each part's last
half second of positions (to tell a part at rest from one still moving) and is fed one observation per control step
by ``task.success()``:

    Tracker.update(objects) -> Result(score, success, details)

``objects`` maps the names in ``layout.STANDOFFS`` and ``"board"`` to their states. A standoff is a plain cylinder:
``pos`` (its centre), ``quat_xyzw`` and ``low`` (lowest corner of its world-aligned box). The board is a mechkit
compound (``AssemblyState``): ``origin`` (the centre of its underside), ``quat_xyzw``, ``pos`` and ``low``. A step on
which any part's state has a non-finite number, a zero quaternion or (for the board) no ``origin`` scores 0 and
leaves the tracker as it was.

A standoff is the same either end down, so its bottom is whichever end is lower.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

BOARD = L.BOARD
NAMES = (*L.STANDOFFS, BOARD)
TRAY_TOP_W = L.TOP + L.TRAY_TOP  # the tray's top (world z)


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
    """Every number the scorer reads is finite and the quaternion is not degenerate; the board also needs its base
    frame's origin (it is missing if it was not built as a compound)."""
    keys = ("origin", "quat_xyzw", "pos", "low") if name == BOARD else ("pos", "quat_xyzw", "low")
    try:
        parts = [np.asarray(getattr(st, k), dtype=float).ravel() for k in keys]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size == (4 if k == "quat_xyzw" else 3) and np.all(np.isfinite(p)) for k, p in zip(keys, parts)):
        return False
    return bool(np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(r: np.ndarray) -> float:
    """Angle between the vertical and the body's +z axis (0..180 degrees)."""
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))


def standoff_axis(st) -> tuple[np.ndarray, np.ndarray, float]:
    """(bottom, top, tilt): the centres of the standoff's lower and upper faces in the world and its tilt from
    upright (0..90 degrees; it is the same either end down)."""
    a = rotation(st.quat_xyzw)[:, 2]
    if a[2] < 0.0:
        a = -a
    c = np.asarray(st.pos, dtype=float)
    return (c - a * (L.STANDOFF_H / 2), c + a * (L.STANDOFF_H / 2),
            float(np.degrees(np.arccos(np.clip(a[2], -1.0, 1.0)))))


def nearest_hole(point, holes=L.HOLES) -> tuple[str, float]:
    """The hole whose centre is nearest the point seen from above, and that distance (m)."""
    name = min(holes, key=lambda k: math.hypot(point[0] - holes[k][0], point[1] - holes[k][1]))
    return name, float(math.hypot(point[0] - holes[name][0], point[1] - holes[name][1]))


def standoff_pose(st) -> dict:
    """Where a standoff stands: its tilt, the height of its bottom face's centre above the tray's top, the nearest
    of the tray's seven holes and the distance to it, the hole it stands on, if any (upright within 10 degrees, that
    centre within 3 mm of the tray's top and within 5 mm of the hole's centre), and that hole if it is a mini-ITX
    one."""
    bottom, top, tilt = standoff_axis(st)
    floor = float(bottom[2] - TRAY_TOP_W)
    near, off = nearest_hole(bottom)
    upright, on_tray = tilt <= L.UPRIGHT_DEG, abs(floor) <= L.FLOOR_TOL
    seated = near if (off <= L.ON_HOLE_TOL and upright and on_tray) else None
    itx = seated if seated in L.ITX_HOLES else None
    return {"bottom": bottom, "top": top, "tilt_deg": tilt, "floor": floor, "nearest": near, "offset": off,
            "upright": upright, "on_tray": on_tray, "seated": seated, "itx_hole": itx}


def _outside(point, rect) -> float:
    x0, x1, y0, y1 = rect
    dx = max(x0 - point[0], 0.0, point[0] - x1)
    dy = max(y0 - point[1], 0.0, point[1] - y1)
    return float(math.hypot(dx, dy))


def outside_outline(point) -> float:
    """How far the point lies outside the mounted board's 88 x 88 mm outline seen from above (m; 0 inside it)."""
    return _outside(point, L.OUTLINE)


def outside_tray(point) -> float:
    """How far the point lies outside the case tray's 180 x 200 mm outline seen from above (m; 0 inside it)."""
    return _outside(point, L.TRAY_OUTLINE)


def under_board(st) -> bool:
    """The standoff's centre, seen from above, is inside the mounted board's outline or less than 9 mm (its radius)
    outside it: some of it is under (or over) the board's place. (Diagnostic: the outline lies inside the tray, so
    such a standoff is also over the tray.)"""
    return outside_outline(np.asarray(st.pos, dtype=float)) < L.STRAY_MARGIN


def over_tray(st) -> bool:
    """The standoff's centre, seen from above, is over the case tray or less than 9 mm (its radius) outside its
    edge: it is in the case."""
    return outside_tray(np.asarray(st.pos, dtype=float)) < L.STRAY_MARGIN


def is_stray(pose: dict, st) -> bool:
    """A standoff in the case (over the tray) that does not stand on one of the board's four holes or on the spare
    hole A3 clear of the board: one left on A1 or A2 (under the board, where it has no hole), or one loose on the
    tray or on the board, upright or lying."""
    return over_tray(st) and pose["seated"] not in (*L.ITX, *L.SPARE_HOLES)


def board_frame(st) -> tuple[np.ndarray, np.ndarray]:
    """(R, origin) of the board's frame: origin at the centre of its underside, +x toward its I/O edge."""
    return rotation(st.quat_xyzw), np.asarray(st.origin, dtype=float)


def board_holes(st) -> dict[str, np.ndarray]:
    """The centres of the board's four holes on its underside in the world, keyed by the tray hole each belongs
    over."""
    r, o = board_frame(st)
    return {k: o + r @ np.array([hx, hy, 0.0]) for k, (hx, hy) in L.BOARD_HOLES.items()}


def board_yaw_deg(st) -> float:
    """Heading of the board's I/O direction (+x) seen from above, from the rear (+x): 0 when the I/O block is at the
    rear, +-180 when it faces the robot (degrees)."""
    a = rotation(st.quat_xyzw)[:, 0]
    return float(math.degrees(math.atan2(a[1], a[0])))


def off_table(st) -> bool:
    """The part is off the table: its centre outside the table's outline seen from above, or its lowest point more
    than 10 mm below the table top."""
    c = np.asarray(st.pos, dtype=float)
    outside = not (L.TABLE_X0 <= c[0] <= L.TABLE_X1 and L.TABLE_Y0 <= c[1] <= L.TABLE_Y1)
    return bool(outside or float(np.asarray(st.low, dtype=float)[2]) < L.TOP - L.OFF_TABLE_DROP)


def track_points(name: str, st) -> tuple[np.ndarray, np.ndarray]:
    """Two points that follow a part's every move: a standoff's bottom and top centres; the board's rear-left and
    front-right hole centres."""
    if name == BOARD:
        h = board_holes(st)
        return h["H1"], h["H4"]
    bottom, top, _ = standoff_axis(st)
    return bottom, top


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: every part's two tracked points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: dict[str, deque] = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev_points: dict[str, np.ndarray] = {}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every standoff upright on its starting hole and the board flat on its mat, I/O
        block to the rear."""
        for name in L.STANDOFFS:
            bottom, _, tilt = standoff_axis(objects[name])
            hx, hy = L.HOLES[L.STANDOFF_START[name]]
            if not (math.hypot(bottom[0] - hx, bottom[1] - hy) < L.START_TOL
                    and abs(bottom[2] - (L.TOP + L.DISC_TOP)) < 0.003 and tilt < 2.0):
                return False
        r, o = board_frame(objects[BOARD])
        return bool(math.hypot(o[0] - L.BOARD_START[0], o[1] - L.BOARD_START[1]) < L.START_TOL
                    and abs(o[2] - (L.TOP + L.MAT_TOP)) < 0.003 and tilt_deg(r) < 2.0)

    def moved(self, name: str) -> float | None:
        """The farthest either tracked point of the part has been, over the last 0.5 s, from where it is now (m);
        None until 0.5 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        a, b = h[-1]
        return float(max(max(np.linalg.norm(a0 - a), np.linalg.norm(b0 - b)) for a0, b0 in h))

    def update(self, objects) -> Result:
        if not all(valid(n, objects.get(n)) for n in NAMES):  # a blow-up: score 0, keep everything
            self.steps += 1
            self.last = self.result(objects)
            return self.last
        points = {n: track_points(n, objects[n]) for n in NAMES}
        if self.prev_points and self.at_start(objects):
            jumped = any(float(np.linalg.norm(points[n][0] - self.prev_points[n])) > L.RESET_JUMP for n in NAMES)
            if jumped:  # the simulator put the parts back on their marks: a new episode
                self.reset()
        self.steps += 1
        for n in NAMES:
            self.history[n].append(points[n])
            self.prev_points[n] = points[n][0]
        self.last = self.result(objects)
        return self.last

    def result(self, objects) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        bad = [n for n in NAMES if not valid(n, objects.get(n))]
        if bad:
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state", "invalid": bad})
        moved = {n: self.moved(n) for n in NAMES}
        rest = {n: moved[n] is not None and moved[n] <= L.STILL_TOL for n in NAMES}
        poses = {n: standoff_pose(objects[n]) for n in L.STANDOFFS}

        # Strays: in the case (over the tray) and not standing on one of the board's holes or on the spare hole A3.
        strays = [n for n in L.STANDOFFS if is_stray(poses[n], objects[n])]
        # A hole has a standoff when one stands on it, at rest.
        on_hole = {k: None for k in L.ITX}
        for n in L.STANDOFFS:
            k = poses[n]["itx_hole"]
            if k is not None and rest[n]:
                on_hole[k] = n
        filled = [k for k in L.ITX if on_hole[k] is not None]

        # Mounted: flat, at rest, every hole over the standoff on its own tray hole and resting on its top.
        br, bo = board_frame(objects[BOARD])
        b_tilt = tilt_deg(br)
        holes = board_holes(objects[BOARD])
        hole_off, hole_gap = {}, {}
        for k in L.ITX:
            n = on_hole[k]
            if n is None:
                hole_off[k], hole_gap[k] = None, None
                continue
            top = poses[n]["top"]
            hole_off[k] = float(math.hypot(holes[k][0] - top[0], holes[k][1] - top[1]))
            hole_gap[k] = float(holes[k][2] - top[2])
        over = all(hole_off[k] is not None and hole_off[k] <= L.BOARD_HOLE_TOL
                   and abs(hole_gap[k]) <= L.BOARD_SEAT_TOL for k in L.ITX)
        mounted = bool(len(filled) == 4 and b_tilt <= L.BOARD_FLAT_DEG and over and rest[BOARD])

        off = [n for n in NAMES if off_table(objects[n])]
        if off:
            score, reason = 0.0, "off_the_table"
        elif strays:
            score, reason = 0.0, "stray_standoff"
        else:
            score = round(L.PER_HOLE * len(filled) + (L.MOUNT_CREDIT if mounted else 0.0), 2)
            reason = "" if score > 0 else "no_hole_filled"
        details = {
            "valid_state": True,
            "tier": score,
            "zero_reason": reason,
            "strays": strays,
            "on_hole": on_hole,
            "holes_filled": len(filled),
            "mounted": mounted,
            "board_over_standoffs": over,
            "board_tilt_deg": round(b_tilt, 2),
            "board_yaw_deg": round(board_yaw_deg(objects[BOARD]), 2),
            "board_x_mm": round((bo[0] - L.MOUNT_X) * 1000, 2),
            "board_y_mm": round((bo[1] - L.MOUNT_Y) * 1000, 2),
            "board_z_mm": round((bo[2] - (L.TOP + L.MOUNT_Z)) * 1000, 2),
            "board_hole_offset_mm": {k: None if v is None else round(v * 1000, 2) for k, v in hole_off.items()},
            "board_hole_gap_mm": {k: None if v is None else round(v * 1000, 2) for k, v in hole_gap.items()},
            "standoff_hole": {n: poses[n]["nearest"] for n in L.STANDOFFS},
            "standoff_offset_mm": {n: round(poses[n]["offset"] * 1000, 2) for n in L.STANDOFFS},
            "standoff_tilt_deg": {n: round(poses[n]["tilt_deg"], 2) for n in L.STANDOFFS},
            "standoff_floor_mm": {n: round(poses[n]["floor"] * 1000, 2) for n in L.STANDOFFS},
            "standoff_seated": {n: poses[n]["seated"] for n in L.STANDOFFS},
            "standoff_over_tray": {n: over_tray(objects[n]) for n in L.STANDOFFS},
            "standoff_under_board": {n: under_board(objects[n]) for n in L.STANDOFFS},
            "moved_mm": {n: None if moved[n] is None else round(moved[n] * 1000, 2) for n in NAMES},
            "off_table": off,
        }
        return Result(score, bool(score == 1.0), details)
