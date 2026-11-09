"""The scorer, as plain functions of the three parts' states. NumPy only; no simulator needed.

The score is read from where the parts lie at the end, as an inspector checks hand placement against IPC-A-610 with
a top photo and a ruler: is each part flat on its own pads and square to them, does an end stick out past a pad's
outer end (end overhang, never allowed), and how far does a termination stick out past the side of its pad (side
overhang: at most 25 % of the part's width for Class 3, 50 % for Classes 1 and 2). The tantalum capacitor is
polarised: its banded end must sit on the + pad. The scorer keeps only each part's last half second of positions
(to tell a part at rest from one still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(objects) -> Result(score, success, details)

``objects`` maps each part's name ("resistor", "capacitor", "tantalum") to its state as mechkit reports it
(``AssemblyState``): ``origin`` (its base frame's origin: the middle of its bottom face), ``quat_xyzw`` (that frame's
orientation: local +x along its length, toward the band on the tantalum; local +z up) and the ``ObjectState``
fields. A step on which any part's state has a non-finite number, a zero quaternion or no ``origin`` (a simulator
blow-up, or a part not built as an assembly) scores 0 and leaves the tracker as it was.

Every outline is the part's shadow seen from straight above (its 8 box corners, turned and tilted as it lies), so a
top photo measures the same thing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L


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


def valid(st) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(st, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def to_world(st, local: np.ndarray) -> np.ndarray:
    """Points (n, 3) in the part's base frame -> the world."""
    return np.asarray(local, dtype=float) @ rotation(st.quat_xyzw).T + np.asarray(st.origin, dtype=float)


def box_corners(part: L.Part) -> np.ndarray:
    """The 8 corners of the part's box in its own frame (origin at the middle of its bottom face)."""
    return np.array([(sx * part.length / 2, sy * part.width / 2, z)
                     for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, part.height)])


def termination_corners(part: L.Part, end: int) -> np.ndarray:
    """The 8 corners of one termination (end = +1 or -1: the last ``term`` of the part's length at that end)."""
    xs = (end * part.length / 2, end * (part.length / 2 - part.term))
    return np.array([(x, sy * part.width / 2, z) for x in xs for sy in (-1, 1) for z in (0.0, part.height)])


def in_footprint(part: L.Part, world: np.ndarray) -> np.ndarray:
    """World points (n, 3) -> (n, 2): along the part's own footprint's axis (x) and across it (y), from its centre."""
    fp = part.footprint
    return np.asarray(world, dtype=float)[:, :2] - np.array([fp.x, fp.y])


def tilt_deg(part: L.Part, st) -> float:
    """How far the part is tilted from lying on a face it may lie on (degrees). The tantalum must lie band face up:
    0..180 from its +z. The resistor is the same either broad face up: 0..90, from the nearer of its +-z. The ceramic
    capacitor is square in section with its terminations round all four long faces, so it is the same on any of
    them: from the nearest of its +-y and +-z (0..90; 90 when it stands on an end)."""
    r = rotation(st.quat_xyzw)
    if part.square_section:
        return float(np.degrees(np.arccos(np.clip(max(abs(r[2, 1]), abs(r[2, 2])), 0.0, 1.0))))
    a = float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))
    return a if part.polarised else min(a, 180.0 - a)


def axis_err_deg(st) -> float:
    """Angle between the part's long axis seen from above and the footprints' axis (x), either way round (0..90)."""
    a = rotation(st.quat_xyzw)[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-9:  # standing on an end
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(a[0]) / h))))


def plus_end_sign(st) -> float:
    """+1 if the part's local +x end (the tantalum's band) points toward world +x (the + pad), else -1."""
    return 1.0 if rotation(st.quat_xyzw)[0, 0] > 0.0 else -1.0


def lowest_gap(part: L.Part, st) -> float:
    """Height of the part's lowest corner above the pads' top (m; negative = below it)."""
    return float(to_world(st, box_corners(part))[:, 2].min() - L.PAD_TOP_W)


def centre_offset(part: L.Part, st) -> np.ndarray:
    """(along, across) offset of the part's centre (middle of its box) from its footprint's centre (m)."""
    c = to_world(st, np.array([[0.0, 0.0, part.height / 2]]))
    return in_footprint(part, c)[0]


def end_overhang(part: L.Part, st) -> float:
    """How far the part's outline reaches past the outer end of its footprint's pads, along the axis (m; 0 or less
    means no end overhang). Taken at whichever end sticks out more."""
    p = in_footprint(part, to_world(st, box_corners(part)))
    return float(np.abs(p[:, 0]).max() - part.footprint.outer)


def side_overhangs(part: L.Part, st) -> tuple[float, float]:
    """(at the + end, at the - end): how far each termination's outline reaches past the side of the pads, across
    the axis (m; 0 when it stays within the pad's width). The footprint's two pads are equally wide and in line."""
    out = []
    for end in (1, -1):
        p = in_footprint(part, to_world(st, termination_corners(part, end)))
        out.append(max(0.0, float(np.abs(p[:, 1]).max() - part.footprint.pad_width / 2)))
    return out[0], out[1]


def flat_on_pads(part: L.Part, st) -> bool:
    """Lying flat (up axis within 5 degrees of vertical; the tantalum band face up) with its lowest point within
    2 mm of the pads' top."""
    return bool(tilt_deg(part, st) <= L.TILT_MAX_DEG and abs(lowest_gap(part, st)) <= L.FLAT_TOL)


def over_own_footprint(part: L.Part, st) -> bool:
    """The part's centre is inside the rectangle its two pads span (outer end to outer end, pad width across)."""
    a, c = centre_offset(part, st)
    return bool(abs(a) <= part.footprint.outer and abs(c) <= part.footprint.pad_width / 2)


def reversed_on_pads(st) -> bool:
    """The tantalum on its own pads the wrong way round: flat, band face up, its axis within 10 degrees of the
    footprint's, its centre over the footprint, and its band at the - end."""
    t = L.TANTALUM
    return bool(flat_on_pads(t, st) and axis_err_deg(st) <= L.AXIS_MAX_DEG and over_own_footprint(t, st)
                and plus_end_sign(st) != L.PLUS_PAD_SIGN)


def off_table(part: L.Part, st) -> bool:
    """The part has fallen off the table: its lowest corner more than 5 cm below the table top."""
    return bool(to_world(st, box_corners(part))[:, 2].min() < L.TOP - L.OFF_TABLE_DROP)


def assess(part: L.Part, st) -> tuple[int, str, dict]:
    """(credit in sixths from the pose alone: 2 Class 3, 1 Classes 1-2, 0; why it is 0; diagnostics)."""
    tilt, axis, gap = tilt_deg(part, st), axis_err_deg(st), lowest_gap(part, st)
    along, across = centre_offset(part, st)
    end = end_overhang(part, st)
    side_p, side_m = side_overhangs(part, st)
    side = max(side_p, side_m)
    d = {"along_mm": round(along * 1000, 2), "across_mm": round(across * 1000, 2), "tilt_deg": round(tilt, 2),
         "axis_err_deg": round(axis, 2), "low_gap_mm": round(gap * 1000, 2), "end_overhang_mm": round(end * 1000, 2),
         "side_overhang_mm": [round(side_p * 1000, 2), round(side_m * 1000, 2)],
         "side_pct_w": round(100 * side / part.width, 1)}
    if part.polarised:
        d["band_toward_plus"] = plus_end_sign(st) == L.PLUS_PAD_SIGN
    if tilt > L.TILT_MAX_DEG:
        return 0, "upside_down" if part.polarised and tilt > 90.0 else "not_flat", d
    if abs(gap) > L.FLAT_TOL:
        return 0, "not_on_the_pads_height", d
    if axis > L.AXIS_MAX_DEG:
        return 0, "not_along_its_footprint", d
    if part.polarised and plus_end_sign(st) != L.PLUS_PAD_SIGN:
        return 0, "band_on_the_minus_pad", d
    if end > L.END_TOL:
        return 0, "end_overhang", d
    if side <= L.CLASS3_FRAC * part.width:
        return L.CLASS3_SIXTHS, "", d
    if side <= L.CLASS2_FRAC * part.width:
        return L.CLASS2_SIXTHS, "", d
    return 0, "side_overhang_over_half_its_width", d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each part's end points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: dict[str, deque] = {p.name: deque(maxlen=self.window + 1) for p in L.PARTS}
        self.prev_origin: dict[str, np.ndarray] | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def states(objects) -> dict | None:
        """The three parts' states, or None if any is missing or not valid."""
        try:
            out = {p.name: objects[p.name] for p in L.PARTS}
        except (KeyError, TypeError, IndexError):
            return None
        return out if all(valid(s) for s in out.values()) else None

    @staticmethod
    def at_start(states: dict) -> bool:
        """The starting layout: every part lying still and flat on its mat, on its mark, long side along y (the
        tantalum's band toward +y)."""
        for p in L.PARTS:
            st = states[p.name]
            o = np.asarray(st.origin, dtype=float)
            if math.hypot(o[0] - p.start[0], o[1] - p.start[1]) >= L.START_TOL:
                return False
            if abs(o[2] - (L.TOP + L.MAT_TOP)) > 0.003 or tilt_deg(p, st) > 2.0:
                return False
            a = rotation(st.quat_xyzw)[:, 0]
            if abs(a[1]) < math.cos(math.radians(5.0)) or (p.polarised and a[1] < 0.0):
                return False
            try:
                if float(np.linalg.norm(np.asarray(st.lin_vel, dtype=float))) >= 0.01:
                    return False
            except (AttributeError, TypeError, ValueError):
                pass
        return True

    @staticmethod
    def _ends(part: L.Part, st) -> tuple[np.ndarray, np.ndarray]:
        """The centres of the part's two end faces (world)."""
        h = part.height / 2
        e = to_world(st, np.array([[part.length / 2, 0.0, h], [-part.length / 2, 0.0, h]]))
        return e[0], e[1]

    def moved(self, name: str) -> float | None:
        """The farthest either end of the part has been, over the last 0.5 s, from where it is now (m); None until
        0.5 s of this trial has been seen."""
        hist = self.history[name]
        if len(hist) < self.window + 1:
            return None
        a, b = hist[-1]
        return float(max(max(np.linalg.norm(p - a), np.linalg.norm(q - b)) for p, q in hist))

    def update(self, objects) -> Result:
        states = self.states(objects)
        if states is None:  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(objects)
            return self.last
        origins = {n: np.asarray(s.origin, dtype=float) for n, s in states.items()}
        if self.prev_origin is not None and self.at_start(states):
            jump = max(float(np.linalg.norm(origins[n] - self.prev_origin[n])) for n in origins)
            if jump > L.RESET_JUMP:  # the simulator put the parts back: a new episode
                self.reset()
        self.steps += 1
        for p in L.PARTS:
            self.history[p.name].append(self._ends(p, states[p.name]))
        self.prev_origin = origins
        self.last = self.result(objects)
        return self.last

    def result(self, objects) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        states = self.states(objects)
        if states is None:
            return Result(0.0, False, {"valid_state": False, "zero_reasons": ["invalid_state"]})
        details: dict = {"valid_state": True}
        zero: list[str] = []
        total = 0
        for p in L.PARTS:
            st = states[p.name]
            sixths, reason, d = assess(p, st)
            moved = self.moved(p.name)
            at_rest = moved is not None and moved <= L.STILL_TOL
            credit = sixths if at_rest else 0
            if sixths and not at_rest:
                reason = "still_moving"
            if off_table(p, st):
                zero.append(f"{p.name}_off_the_table")
            total += credit
            details[p.name] = {"credit_sixths": credit, "tier": {2: "class3", 1: "class2", 0: "none"}[sixths],
                               "zero_reason": reason, "at_rest": at_rest,
                               "moved_mm": None if moved is None else round(moved * 1000, 2), **d}
        if reversed_on_pads(states[L.TANTALUM.name]):
            zero.append("tantalum_reversed_on_its_pads")
        sixths_total = 0 if zero else total
        details["zero_reasons"] = zero
        details["sixths"] = sixths_total
        return Result(sixths_total / 6.0, bool(sixths_total == 6), details)
