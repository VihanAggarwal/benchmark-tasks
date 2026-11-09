"""The scorer, as plain functions of the three pins' and the bracket's states. NumPy only; no simulator needed.

The score is read from the end state, as a setter checks a two-pin fixture with a look from above and a rule: which
pin stands in which socket, which way the diamond pin's flats face, and whether the bracket lies flat on the fixture
over both pins. The scorer keeps only each part's last half second of positions (to tell a part at rest from one
still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(objects) -> Result(score, success, details)

``objects`` maps the names in ``layout.PIN_NAMES`` and ``"bracket"`` to their states. The two round pins are plain
cylinders: ``pos`` (their centre), ``quat_xyzw``, ``low`` (lowest corner of their world-aligned box). The diamond pin
and the bracket are mechkit compounds (``AssemblyState``): ``origin`` (the diamond's: the centre of its shank's bottom
face; the bracket's: the centre of its underside), ``quat_xyzw``, ``pos`` and ``low``. A step on which any part's
state has a non-finite number, a zero quaternion or (for a compound) no ``origin`` scores 0 and leaves the tracker as
it was.

A round pin is the same either end down, so its bottom is whichever end is lower; the diamond pin has a bottom (its
round shank) and is upside down when its head is underneath (tilt 180 degrees).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

BRACKET = "bracket"
NAMES = (*L.PIN_NAMES, BRACKET)
PLATE_TOP_W = L.TOP + L.FIX_TOP  # the fixture plate's top (world z)
MID_PLANE = L.BR_T / 2  # a pin is 'in a hole' where its axis crosses the bracket's mid-plane


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
    """Every number the scorer reads is finite and the quaternion is not degenerate; the compounds (diamond pin and
    bracket) also need their base frame's origin (it is missing if they were not built as compounds)."""
    keys = ("origin", "quat_xyzw", "pos", "low") if name in (L.DIAMOND, BRACKET) else ("pos", "quat_xyzw", "low")
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


def pin_axis(name: str, st) -> tuple[np.ndarray, np.ndarray, float]:
    """(bottom, top, tilt): the centres of the pin's bottom and top faces in the world and its tilt from upright
    (degrees). A round pin's bottom is its lower end (tilt 0..90); the diamond's is its shank's end (tilt 0..180:
    head down is 180)."""
    r = rotation(st.quat_xyzw)
    a = r[:, 2]
    if name == L.DIAMOND:
        bottom = np.asarray(st.origin, dtype=float)
        return bottom, bottom + a * L.PIN_H, tilt_deg(r)
    if a[2] < 0.0:
        a = -a
    c = np.asarray(st.pos, dtype=float)
    return c - a * (L.PIN_H / 2), c + a * (L.PIN_H / 2), float(np.degrees(np.arccos(np.clip(a[2], -1.0, 1.0))))


def socket_of(point) -> str | None:
    """The socket whose 55 x 55 mm opening the point lies inside, seen from above (None if none)."""
    for k, (sx, sy) in L.SOCKETS.items():
        if abs(point[0] - sx) <= L.SOCKET_HALF and abs(point[1] - sy) <= L.SOCKET_HALF:
            return k
    return None


def pin_pose(name: str, st) -> dict:
    """Where a pin stands: the socket its bottom face's centre is in (seen from above), its tilt, the height of that
    centre above the table, its distance from the socket's centre, and whether that makes it 'down in a socket'
    (upright within 5 degrees, bottom within 3 mm of the table, inside the opening)."""
    bottom, top, tilt = pin_axis(name, st)
    k = socket_of(bottom)
    floor = float(bottom[2] - L.TOP)
    off = None if k is None else float(math.hypot(bottom[0] - L.SOCKETS[k][0], bottom[1] - L.SOCKETS[k][1]))
    down = k is not None and tilt <= L.FIT_TILT_DEG and abs(floor) <= L.FIT_FLOOR_TOL
    return {"socket": k, "tilt_deg": tilt, "floor": floor, "centre_off": off, "down_in_socket": down,
            "bottom": bottom, "top": top}


def head_turn_err_deg(st) -> float:
    """How far the diamond's head is turned from square to the line between the sockets, seen from above (0..90
    degrees): 0 when its long side runs across the line, so that its two flats face the other socket; 90 when its
    long side runs along the line (a quarter turn wrong)."""
    a = rotation(st.quat_xyzw)[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-9:  # lying with its long side vertical: not square to anything
        return 90.0
    line = np.subtract(L.SOCKETS["A"], L.SOCKETS["B"])
    line = line / np.linalg.norm(line)
    along = math.degrees(math.acos(min(1.0, abs(a[0] * line[0] + a[1] * line[1]) / h)))
    return float(90.0 - along)


def bracket_frame(st) -> tuple[np.ndarray, np.ndarray]:
    """(R, origin) of the bracket's frame: origin at the centre of its underside, +y along its length."""
    return rotation(st.quat_xyzw), np.asarray(st.origin, dtype=float)


def bracket_corners(st) -> np.ndarray:
    """The four corners of the bracket's underside in the world, (4, 3)."""
    r, o = bracket_frame(st)
    return np.array([o + r @ np.array(c) for c in L.BR_CORNERS])


def bracket_holes(st) -> np.ndarray:
    """The centres of the bracket's two holes on its underside in the world, (2, 3)."""
    r, o = bracket_frame(st)
    return np.array([o + r @ np.array(c) for c in L.BR_HOLES])


def bracket_pose(st) -> dict:
    """Is the bracket lying flat at the plate's height: level within 2 degrees (its boss up) and every corner of its
    underside within 2 mm of the plate's top?"""
    r, o = bracket_frame(st)
    tilt = tilt_deg(r)
    gaps = [float(c[2] - PLATE_TOP_W) for c in bracket_corners(st)]
    flat = tilt <= L.SEAT_TILT_DEG and max(abs(g) for g in gaps) <= L.SEAT_TOL
    return {"tilt_deg": tilt, "corner_gaps": gaps, "flat": flat, "origin": o}


def in_hole(st_bracket, bottom, top) -> tuple[int | None, tuple[float, float] | None]:
    """Which of the bracket's holes a pin stands in: where the pin's axis crosses the bracket's mid-plane, inside a
    hole's 64 x 64 mm outline in the bracket's own frame. Returns (hole index 0 = +y end, 1 = -y end, or None; the
    crossing point's offset (u, v) from that hole's centre, m)."""
    r, o = bracket_frame(st_bracket)
    n = r[:, 2]
    d = np.asarray(top, dtype=float) - np.asarray(bottom, dtype=float)
    plane = o + n * MID_PLANE
    den = float(n @ d)
    if abs(den) < 1e-9:  # the pin's axis runs parallel to the bracket
        return None, None
    t = float(n @ (plane - bottom)) / den
    p = np.asarray(bottom, dtype=float) + t * d
    loc = r.T @ (p - o)
    for i, (hx, hy, _) in enumerate(L.BR_HOLES):
        du, dv = float(loc[0] - hx), float(loc[1] - hy)
        if abs(du) <= L.HOLE_HALF and abs(dv) <= L.HOLE_HALF:
            return i, (du, dv)
    return None, None


def off_table(name: str, st) -> bool:
    """The part is off the table: its centre outside the table's outline seen from above, or its lowest point more
    than 10 mm below the table top."""
    c = np.asarray(st.pos, dtype=float)
    outside = not (L.TABLE_X0 <= c[0] <= L.TABLE_X1 and L.TABLE_Y0 <= c[1] <= L.TABLE_Y1)
    return bool(outside or float(np.asarray(st.low, dtype=float)[2]) < L.TOP - L.OFF_TABLE_DROP)


def track_points(name: str, st) -> tuple[np.ndarray, np.ndarray]:
    """Two points that follow a part's every move: a pin's bottom and top centres, the bracket's two hole centres."""
    if name == BRACKET:
        h = bracket_holes(st)
        return h[0], h[1]
    bottom, top, _ = pin_axis(name, st)
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
        """The starting layout: every pin standing upright on its tray mark, the diamond's head along the line
        between the sockets, and the bracket lying flat on the table on its mark."""
        for name in L.PIN_NAMES:
            bottom, _, tilt = pin_axis(name, objects[name])
            mark = L.DIAMOND_START if name == L.DIAMOND else L.ROUND_START[name]
            if not (math.hypot(bottom[0] - mark[0], bottom[1] - mark[1]) < L.START_TOL
                    and abs(bottom[2] - (L.TOP + L.TRAY_TOP)) < 0.003 and tilt < 2.0):
                return False
        if head_turn_err_deg(objects[L.DIAMOND]) < 80.0:
            return False
        br = objects[BRACKET]
        r, o = bracket_frame(br)
        return bool(math.hypot(o[0] - L.BR_START[0], o[1] - L.BR_START[1]) < L.START_TOL
                    and abs(o[2] - L.TOP) < 0.003 and tilt_deg(r) < 2.0)

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
        pins = {n: pin_pose(n, objects[n]) for n in L.PIN_NAMES}
        fitted = {n: pins[n]["socket"] if pins[n]["down_in_socket"] and rest[n] else None for n in L.PIN_NAMES}
        dia = objects[L.DIAMOND]
        turn_err = head_turn_err_deg(dia)
        turned_right = turn_err <= L.TURN_TOL_DEG
        br = bracket_pose(objects[BRACKET])
        holes = {n: in_hole(objects[BRACKET], pins[n]["bottom"], pins[n]["top"]) for n in L.PIN_NAMES}

        # The located set: the diamond fitted, turned right, and a round pin fitted in the other socket.
        located_round = None
        if fitted[L.DIAMOND] is not None and turned_right:
            others = [n for n in L.ROUND if fitted[n] is not None and fitted[n] != fitted[L.DIAMOND]]
            located_round = others[0] if others else None
        located = located_round is not None
        # Seated: flat at the plate's height, at rest, with every fitted pin standing in a hole of its own.
        fitted_names = [n for n in L.PIN_NAMES if fitted[n] is not None]
        used = [holes[n][0] for n in fitted_names]
        over_pins = bool(fitted_names) and all(h is not None for h in used) and len(set(used)) == len(used)
        seated = br["flat"] and rest[BRACKET] and over_pins

        off = [n for n in NAMES if off_table(n, objects[n])]
        if off:
            tier, reason = 0.0, "off_the_table"
        elif not fitted_names:
            tier, reason = 0.0, "no_pin_fitted"
        elif not located:
            tier, reason = 0.25, ""
        elif not seated:
            tier, reason = 0.5, ""
        else:
            tier, reason = 1.0, ""
        o = br["origin"]
        details = {
            "valid_state": True,
            "tier": tier,
            "zero_reason": reason,
            "fitted": fitted,
            "located_set": located,
            "located_round_pin": located_round,
            "diamond_turn_err_deg": round(turn_err, 2),
            "diamond_turned_right": turned_right,
            "bracket_seated": seated,
            "bracket_flat": br["flat"],
            "bracket_over_fitted_pins": over_pins,
            "bracket_tilt_deg": round(br["tilt_deg"], 2),
            "bracket_corner_mm": [round(g * 1000, 2) for g in br["corner_gaps"]],
            "bracket_x_mm": round((o[0] - L.FIX_X) * 1000, 2),
            "bracket_y_mm": round((o[1] - L.FIX_Y) * 1000, 2),
            "pin_hole": {n: holes[n][0] for n in L.PIN_NAMES},
            "pin_hole_offset_mm": {n: None if holes[n][1] is None else [round(v * 1000, 2) for v in holes[n][1]]
                                   for n in L.PIN_NAMES},
            "socket": {n: pins[n]["socket"] for n in L.PIN_NAMES},
            "tilt_deg": {n: round(pins[n]["tilt_deg"], 2) for n in L.PIN_NAMES},
            "floor_mm": {n: round(pins[n]["floor"] * 1000, 2) for n in L.PIN_NAMES},
            "centre_off_mm": {n: None if pins[n]["centre_off"] is None else round(pins[n]["centre_off"] * 1000, 2)
                              for n in L.PIN_NAMES},
            "moved_mm": {n: None if moved[n] is None else round(moved[n] * 1000, 2) for n in NAMES},
            "off_table": off,
        }
        return Result(tier, bool(tier == 1.0), details)
