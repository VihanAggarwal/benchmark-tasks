"""The scorer, as plain functions of the two magnets' and two keepers' states. NumPy only; no simulator needed.

The score is read from where things lie, as a technician checks a put-away pair with a steel rule and a look from
above: are both magnets lying flat in the tray along its length, side by side; do their red (north) ends point
opposite ways (unlike poles beside each other); and is there a keeper across each end of the pair, touching (or
within 8 mm of) both magnets' end faces. The scorer keeps only the last half second of each object's positions (to
tell an object at rest from one still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(states) -> Result(score, success, details)

``states`` maps ``magnet_a``, ``magnet_b``, ``keeper_1`` and ``keeper_2`` to their states. A magnet's state is what
mechkit reports (``AssemblyState``): ``origin`` (its base frame's origin, the middle of its bottom face) and
``quat_xyzw`` (that frame: local +x toward the red end, local +z up), with the ObjectState fields. A keeper's state is
a plain ``ObjectState`` (``pos`` its centre, ``quat_xyzw``). A state with a non-finite number, a zero quaternion or a
magnet without ``origin`` (a simulator blow-up, or a magnet not built as an assembly) scores 0 for that step and
leaves the tracker as it was.

The two magnets are identical and so are the two keepers: neither is told apart from its twin, and either keeper may
close either end.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

MAGNETS = ("magnet_a", "magnet_b")
KEEPERS = ("keeper_1", "keeper_2")
NAMES = MAGNETS + KEEPERS
FLOOR_W = L.FLOOR_TOP_W

_MAG_CORNERS = np.array([(sx * L.HALF_LEN, sy * L.MAG_W / 2, z) for sx in (-1, 1) for sy in (-1, 1)
                         for z in (0.0, L.MAG_H)])
_KEEP_CORNERS = np.array([(sx * L.KEEP_SIZE[0] / 2, sy * L.KEEP_SIZE[1] / 2, sz * L.KEEP_SIZE[2] / 2)
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


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


def valid_magnet(m) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    parts = _finite(m, ("origin", "quat_xyzw", "pos"))
    return bool(parts and parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid_keeper(k) -> bool:
    parts = _finite(k, ("pos", "quat_xyzw"))
    return bool(parts and parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid(states) -> bool:
    try:
        return all(valid_magnet(states[n]) for n in MAGNETS) and all(valid_keeper(states[n]) for n in KEEPERS)
    except (KeyError, TypeError):
        return False


# -- magnets ---------------------------------------------------------------------------------------------------------


def magnet_corners(m) -> np.ndarray:
    """The magnet's eight corners in the world, (8, 3)."""
    r = rotation(m.quat_xyzw)
    return np.asarray(m.origin, dtype=float) + _MAG_CORNERS @ r.T


def magnet_centre(m) -> np.ndarray:
    r = rotation(m.quat_xyzw)
    return np.asarray(m.origin, dtype=float) + r @ np.array([0.0, 0.0, L.MAG_H / 2])


def magnet_ends(m) -> tuple[np.ndarray, np.ndarray]:
    """(red end face's centre, blue end face's centre) in the world."""
    r = rotation(m.quat_xyzw)
    o = np.asarray(m.origin, dtype=float)
    return o + r @ np.array([L.HALF_LEN, 0.0, L.MAG_H / 2]), o + r @ np.array([-L.HALF_LEN, 0.0, L.MAG_H / 2])


def level_deg(axis) -> float:
    """How far a body axis is from level (0..90 degrees)."""
    return float(np.degrees(math.asin(min(1.0, abs(float(axis[2]))))))


def heading_from_deg(axis, ref: int) -> float:
    """Angle, seen from above, between a body axis and the world's x (ref 0) or y (ref 1) axis, either way round
    (0..90 degrees); 90 if the axis is vertical."""
    h = math.hypot(axis[0], axis[1])
    if h < 1e-9:
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(float(axis[ref])) / h))))


def red_sign(m) -> int:
    """+1 if the red end points along the tray away from the arms (+x), -1 toward them, 0 exactly across."""
    a = rotation(m.quat_xyzw)[0, 0]
    return 1 if a > 1e-9 else (-1 if a < -1e-9 else 0)


def inside_rims(points_xy, slack: float = L.RIM_SLACK) -> bool:
    p = np.asarray(points_xy, dtype=float)
    return bool(np.all(p[:, 0] >= L.INNER_X0 - slack) and np.all(p[:, 0] <= L.INNER_X1 + slack)
                and np.all(np.abs(p[:, 1] - L.TRAY_Y) <= L.INNER_Y + slack))


def magnet_pose(m) -> dict:
    """What a rule and a look from above tell about one magnet (no rest test)."""
    c = magnet_corners(m)
    axis = rotation(m.quat_xyzw)[:, 0]
    low = float(c[:, 2].min() - FLOOR_W)
    centre = magnet_centre(m)
    head = heading_from_deg(axis, 0)
    lev = level_deg(axis)
    inside = inside_rims(c[:, :2])
    in_tray = bool(inside_rims(centre[None, :2], 0.0) and low < L.IN_TRAY_LOW_MAX)
    stored = bool(low <= L.STORE_LOW_TOL and lev <= L.STORE_LEVEL_DEG and head <= L.STORE_HEADING_DEG and inside)
    return {"low_mm": round(low * 1000, 2), "level_deg": round(lev, 2), "heading_deg": round(head, 2),
            "inside_rims": inside, "in_tray": in_tray, "lengthwise": bool(head <= L.LENGTHWISE_DEG),
            "red": red_sign(m), "stored_pose": stored, "centre": centre}


def pair_measure(ma, mb) -> tuple[float, float]:
    """(across, along): how far apart the two magnets' centres are across the tray (y) and along it (x), in m."""
    ca, cb = magnet_centre(ma), magnet_centre(mb)
    return float(abs(ca[1] - cb[1])), float(abs(ca[0] - cb[0]))


def end_face(m, side: int) -> tuple[np.ndarray, tuple[float, float]]:
    """The centre of the magnet's end face on that side of the tray (side +1: the end farther from the arms, -1:
    the nearer end), and that face's extent across the tray (y min, y max)."""
    red, blue = magnet_ends(m)
    use_red = side * red[0] >= side * blue[0]
    e = red if use_red else blue
    sx = 1.0 if use_red else -1.0
    r = rotation(m.quat_xyzw)
    o = np.asarray(m.origin, dtype=float)
    pts = [o + r @ np.array([sx * L.HALF_LEN, sy * L.MAG_W / 2, z]) for sy in (-1, 1) for z in (0.0, L.MAG_H)]
    ys = [float(p[1]) for p in pts]
    return e, (min(ys), max(ys))


# -- keepers ---------------------------------------------------------------------------------------------------------


def keeper_corners(k) -> np.ndarray:
    r = rotation(k.quat_xyzw)
    return np.asarray(k.pos, dtype=float) + _KEEP_CORNERS @ r.T


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def outline(k) -> np.ndarray:
    """The keeper's outline seen from above (the shadow of its corners), (n, 2)."""
    return _hull(keeper_corners(k)[:, :2])


def crossing(poly: np.ndarray, y: float) -> tuple[float, float] | None:
    """Where the line at this y (parallel to the tray's length) crosses a convex outline: (x min, x max), or None."""
    xs = []
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 - y) * (y1 - y) <= 0.0:
            if abs(y1 - y0) < 1e-12:
                xs += [float(x0), float(x1)]
            else:
                xs.append(float(x0 + (y - y0) * (x1 - x0) / (y1 - y0)))
    return (min(xs), max(xs)) if xs else None


def keeper_pose(k) -> dict:
    c = keeper_corners(k)
    axis = rotation(k.quat_xyzw)[:, 1]  # its 74 mm length
    low = float(c[:, 2].min() - FLOOR_W)
    lev, across = level_deg(axis), heading_from_deg(axis, 1)
    on_floor = bool(low <= L.KEEP_LOW_TOL and lev <= L.KEEP_LEVEL_DEG)
    return {"low_mm": round(low * 1000, 2), "level_deg": round(lev, 2), "across_err_deg": round(across, 2),
            "on_floor": on_floor, "across": bool(across <= L.KEEP_ACROSS_DEG)}


def keeper_at_end(k, mags, side: int) -> dict:
    """How the keeper sits at one end of the pair: for each magnet, the gap along the tray from that end face's
    centre to the keeper's near edge (on the line through the face's centre, parallel to the tray's length;
    None if the keeper does not cross that line) and how much of the face's width it spans."""
    poly = outline(k)
    ylo, yhi = float(poly[:, 1].min()), float(poly[:, 1].max())
    gaps, covers = [], []
    for m in mags:
        e, (fy0, fy1) = end_face(m, side)
        cr = crossing(poly, float(e[1]))
        if cr is None:
            gaps.append(None)
        else:
            gaps.append(cr[0] - float(e[0]) if side > 0 else float(e[0]) - cr[1])
        covers.append(max(0.0, min(yhi, fy1) - max(ylo, fy0)))
    ok_gaps = all(g is not None and L.KEEP_GAP_MIN <= g <= L.KEEP_GAP_MAX for g in gaps)
    ok_cover = all(c >= L.KEEP_COVER_MIN for c in covers)
    return {"gaps_mm": [None if g is None else round(g * 1000, 2) for g in gaps],
            "cover_mm": [round(c * 1000, 2) for c in covers], "ok": bool(ok_gaps and ok_cover)}


# -- the tiers -------------------------------------------------------------------------------------------------------


def evaluate(states, rest: dict) -> tuple[float, str, dict]:
    """(score, why it is not higher, diagnostics) of a valid state, given which objects are at rest.

    0: both magnets lie lengthwise in the tray with their red ends pointing the same way (like poles side by side);
    or no magnet is stored. 0.25: one magnet stored, or both stored but not side by side. 0.5: stored side by side
    (with unlike poles, since like poles score 0). 0.75: and one end closed by a keeper. 1.0: both ends closed."""
    mags = [states[n] for n in MAGNETS]
    mp = {n: magnet_pose(states[n]) for n in MAGNETS}
    d: dict = {}
    for n in MAGNETS:
        p = dict(mp[n])
        p["centre"] = [round(float(v), 4) for v in p.pop("centre")]
        p["stored"] = bool(p["stored_pose"] and rest.get(n, False))
        p["at_rest"] = bool(rest.get(n, False))
        d[n] = p
    across, along = pair_measure(*mags)
    d["pair_across_mm"], d["pair_along_mm"] = round(across * 1000, 2), round(along * 1000, 2)
    reds = [mp[n]["red"] for n in MAGNETS]
    like = bool(all(mp[n]["in_tray"] and mp[n]["lengthwise"] for n in MAGNETS) and reds[0] == reds[1] != 0)
    d["like_poles"] = bool(like)
    n_stored = sum(d[n]["stored"] for n in MAGNETS)
    paired = bool(n_stored == 2 and across <= L.PAIR_ACROSS_MAX and along <= L.PAIR_ALONG_MAX)
    d["n_stored"], d["paired"] = n_stored, paired
    ends = {"far": False, "near": False}
    for kn in KEEPERS:
        kp = keeper_pose(states[kn])
        kp["at_rest"] = bool(rest.get(kn, False))
        for side, label in ((1, "far"), (-1, "near")):
            at = keeper_at_end(states[kn], mags, side)
            kp[label] = at
            if paired and kp["on_floor"] and kp["across"] and kp["at_rest"] and at["ok"]:
                ends[label] = True
        d[kn] = kp
    d["far_end_closed"], d["near_end_closed"] = ends["far"], ends["near"]
    if like:
        return 0.0, "like_poles_side_by_side", d
    if n_stored == 0:
        return 0.0, "no_magnet_stored", d
    if n_stored == 1:
        return 0.25, "one_magnet_stored", d
    if not paired:
        return 0.25, "stored_not_side_by_side", d
    closed = int(ends["far"]) + int(ends["near"])
    return 0.5 + 0.25 * closed, ("" if closed == 2 else "end_not_closed"), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _track_points(name: str, s) -> tuple[np.ndarray, np.ndarray]:
    """Two points of the object whose travel tells whether it is at rest: a magnet's end-face centres, a keeper's
    two end-face centres."""
    if name in MAGNETS:
        return magnet_ends(s)
    r = rotation(s.quat_xyzw)
    c = np.asarray(s.pos, dtype=float)
    return c + r[:, 1] * L.KEEP_HALF_LEN, c - r[:, 1] * L.KEEP_HALF_LEN


def _anchor(name: str, s) -> np.ndarray:
    return np.asarray(s.origin if name in MAGNETS else s.pos, dtype=float)


STARTS = {
    "magnet_a": (L.MAG_A_START[0], L.MAG_A_START[1], L.MAG_A_YAW),
    "magnet_b": (L.MAG_B_START[0], L.MAG_B_START[1], L.MAG_B_YAW),
    "keeper_1": (L.KEEP_1_START[0], L.KEEP_1_START[1], 0.0),
    "keeper_2": (L.KEEP_2_START[0], L.KEEP_2_START[1], 0.0),
}


class Tracker:
    """What the scorer remembers within a trial: every object's two tracking points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev = None  # every object's anchor at the previous valid step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: every object still, on the table, on its starting mark, as it starts."""
        for n, (x, y, yaw) in STARTS.items():
            s = states[n]
            a = _anchor(n, s)
            if math.hypot(a[0] - x, a[1] - y) > L.START_TOL:
                return False
            corners = magnet_corners(s) if n in MAGNETS else keeper_corners(s)
            if abs(float(corners[:, 2].min()) - L.TOP) > 0.003:
                return False
            r = rotation(s.quat_xyzw)
            axis = r[:, 0] if n in MAGNETS else r[:, 1]
            want = np.array([math.cos(yaw), math.sin(yaw)]) if n in MAGNETS else np.array([0.0, 1.0])
            dot = float(axis[0] * want[0] + axis[1] * want[1])
            if (dot if n in MAGNETS else abs(dot)) < math.cos(math.radians(5.0)):
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest the object's tracking points have been, over the last 0.5 s, from where they are now (m);
        None until 0.5 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        p, q = h[-1]
        return float(max(max(np.linalg.norm(a - p), np.linalg.norm(b - q)) for a, b in h))

    def update(self, states) -> Result:
        if not valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(states)
            return self.last
        now = {n: _anchor(n, states[n]) for n in NAMES}
        if self.prev is not None and self.at_start(states):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in NAMES)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        for n in NAMES:
            self.history[n].append(_track_points(n, states[n]))
        self.prev = now
        self.last = self.result(states)
        return self.last

    def result(self, states) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not valid(states):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        moved = {n: self.moved(n) for n in NAMES}
        rest = {n: (m is not None and m <= L.STILL_TOL) for n, m in moved.items()}
        score, reason, d = evaluate(states, rest)
        details = {"valid_state": True, "zero_reason": reason,
                   "moved_mm": {n: (None if m is None else round(m * 1000, 2)) for n, m in moved.items()}, **d}
        return Result(score, bool(score == 1.0), details)
