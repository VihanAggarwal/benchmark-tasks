"""The scorer, as plain functions of the phantom's, the two markers' and the grippers' states. NumPy only; no
simulator needed.

The score is read from the end state, as a radiographer checks a set-up before taking an AP femur: is the phantom
lying flat on the table (anterior face up) along the tube's cathode-anode axis (x), with its thick hip end toward the
cathode ("-", +x), wholly inside the light field and centred on the central-ray cross, and is the R marker lying in
the light field clear of the bone, with the L marker out of the field. Every tier needs the phantom let go by both
grippers and still. Dropping the phantom on the floor (the bare station table) at any step scores 0 for the rest of
the trial. A small tracker is fed once per control step by ``task.success()``:

    Tracker.update(phantom, r_marker, l_marker, arms) -> Result(score, success, details)

``phantom`` is the phantom's state as mechkit reports it (``AssemblyState``): ``origin`` (its frame's origin: the
middle of its bottom face half-way along its length), ``quat_xyzw`` (local +x from the knee end to the hip end,
local +z up out of its anterior face) and the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. The markers
are plain ``ObjectState`` boxes (``pos`` their centre). A state with a non-finite number, a zero quaternion or (for
the phantom) no ``origin`` scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then 'released' is taken as
met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so a gripper's
fingers are taken to be anywhere round its axis: its 'finger zone' is every point within a radius of the 38 mm
segment from the point between its pad tips back up the gripper; the radius reaches a pad's outer corner (half the
current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across; 57 mm fully open).

Outlines are taken from above: each of the phantom's three box sections is the convex hull of its eight corners
seen from above, and its round head (an upright cylinder) is covered by the two circles of its end faces and the
band between them. The R marker's outline is the hull of its eight corners.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

FIELD_TOP_W = L.TOP + L.FIELD_TOP
FLOOR_W = L.TOP + L.FLOOR_Z
_EPS = 1e-9
SIDES = ("left", "right")


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


def _finite(s, keys) -> bool:
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in keys]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid_phantom(ph) -> bool:
    """Every number finite, the quaternion not degenerate, and the frame's origin present."""
    return _finite(ph, ("origin", "quat_xyzw", "pos", "lin_vel"))


def valid_marker(m) -> bool:
    return _finite(m, ("pos", "quat_xyzw", "lin_vel"))


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in SIDES:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- geometry ---------------------------------------------------------------------------------------------------------
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def box_corners(origin, r, center, size) -> np.ndarray:
    """The eight corners (8, 3) in the world of a box given in a body's frame."""
    local = np.asarray(center, dtype=float) + _SIGNS * (np.asarray(size, dtype=float) / 2)
    return np.asarray(origin, dtype=float) + local @ r.T


def heading_deg(r) -> float:
    """The body's local x seen from above, counter-clockwise from +x (degrees, -180..180)."""
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def wrap_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def phantom_tilt_deg(ph) -> float:
    """Angle between the phantom's local z (out of its anterior face) and straight up (0..180 degrees)."""
    return float(math.degrees(math.acos(float(np.clip(rotation(ph.quat_xyzw)[2, 2], -1.0, 1.0)))))


def align_err_deg(ph) -> float:
    """Angle between its length seen from above and the tube axis (x), either way round (0..90 degrees)."""
    r = rotation(ph.quat_xyzw)
    if math.hypot(r[0, 0], r[1, 0]) < 1e-9:  # standing on an end
        return 90.0
    h = abs(heading_deg(r))
    return float(min(h, 180.0 - h))


def hip_toward_cathode(ph) -> bool:
    """Its knee-to-hip direction points toward the cathode end (+x) seen from above."""
    return bool(rotation(ph.quat_xyzw)[0, 0] * L.CATHODE_DIR > 0.0)


def head_discs(ph) -> tuple[np.ndarray, np.ndarray]:
    """World centres of the head cylinder's bottom and top faces."""
    r = rotation(ph.quat_xyzw)
    c, _, h = L.PH_HEAD
    o = np.asarray(ph.origin, dtype=float)
    axis = r[:, 2]
    mid = o + r @ np.asarray(c, dtype=float)
    return mid - axis * h / 2, mid + axis * h / 2


def phantom_lowest(ph) -> float:
    """World z of the phantom's lowest point (box corners and the head's rim)."""
    r = rotation(ph.quat_xyzw)
    o = np.asarray(ph.origin, dtype=float)
    low = min(float(box_corners(o, r, c, s)[:, 2].min()) for c, s in L.PH_BOXES)
    _, rad, _ = L.PH_HEAD
    az = abs(float(r[2, 2]))
    rim = rad * math.sqrt(max(0.0, 1.0 - az * az))
    low = min(low, *(float(p[2]) - rim for p in head_discs(ph)))
    return low


def phantom_outline(ph) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray, float]]:
    """Seen from above: the three box sections as convex polygons (xy), and the head as (disc a, disc b, radius)."""
    r = rotation(ph.quat_xyzw)
    o = np.asarray(ph.origin, dtype=float)
    polys = [hull(box_corners(o, r, c, s)[:, :2]) for c, s in L.PH_BOXES]
    a, b = head_discs(ph)
    return polys, (a[:2], b[:2], L.PH_HEAD[1])


def field_margin(ph) -> float:
    """How far the phantom's outline is inside the light field at its tightest (m; negative: partly outside)."""
    x0, x1 = L.FIELD_X
    y0, y1 = L.FIELD_Y
    polys, (a, b, rad) = phantom_outline(ph)
    m = math.inf
    for p in polys:
        for x, y in p:
            m = min(m, x - x0, x1 - x, y - y0, y1 - y)
    for x, y in (a, b):
        m = min(m, x - x0 - rad, x1 - x - rad, y - y0 - rad, y1 - y - rad)
    return float(m)


def centre_offset(ph) -> np.ndarray:
    """(dx, dy) of the phantom's centre (half-way along its length) from the cross (m)."""
    c = np.asarray(ph.origin, dtype=float) + rotation(ph.quat_xyzw) @ np.asarray(L.PH_CENTRE_LOCAL, dtype=float)
    return c[:2] - np.asarray(L.CROSS, dtype=float)


def flat_on_table(ph) -> bool:
    return bool(phantom_tilt_deg(ph) <= L.FLAT_TILT_DEG + _EPS and abs(phantom_lowest(ph) - FIELD_TOP_W) <= L.SEAT_TOL)


def on_floor(ph) -> bool:
    """Some point of the phantom below 5 mm above the floor (below the tops of the cart, the table and the tray)."""
    return bool(phantom_lowest(ph) < FLOOR_W + L.FLOOR_TOUCH)


# -- the markers ------------------------------------------------------------------------------------------------------
def marker_corners(m) -> np.ndarray:
    return box_corners(m.pos, rotation(m.quat_xyzw), (0.0, 0.0, 0.0), L.MARKER_SIZE)


def marker_tilt_deg(m) -> float:
    """Angle between the marker's short (16 mm) axis and the vertical, either face up (0..90 degrees)."""
    return float(math.degrees(math.acos(float(np.clip(abs(rotation(m.quat_xyzw)[2, 2]), 0.0, 1.0)))))


def marker_flat(m) -> bool:
    """Lying flat on the table top: tilted at most 10 degrees, its lowest point within 3 mm of the field's top."""
    low = float(marker_corners(m)[:, 2].min())
    return bool(marker_tilt_deg(m) <= L.MARKER_TILT_DEG + _EPS and abs(low - FIELD_TOP_W) <= L.SEAT_TOL)


def centre_in_field(m) -> bool:
    x, y = float(m.pos[0]), float(m.pos[1])
    return bool(L.FIELD_X[0] <= x <= L.FIELD_X[1] and L.FIELD_Y[0] <= y <= L.FIELD_Y[1])


def marker_clearance(m, ph) -> float:
    """Gap between the marker's outline and the phantom's outline seen from above (m; 0 or less: they overlap)."""
    mp = hull(marker_corners(m)[:, :2])
    polys, (a, b, rad) = phantom_outline(ph)
    gaps = [poly_distance(mp, p) for p in polys]
    gaps.append(poly_distance(mp, np.array([a, b])) - rad)
    return float(min(gaps))


# -- 2-D helpers --------------------------------------------------------------------------------------------------------
def hull(points) -> np.ndarray:
    """Convex hull (counter-clockwise, no repeats) of 2-D points (Andrew's monotone chain)."""
    pts = sorted({(round(float(x), 12), round(float(y), 12)) for x, y in np.asarray(points, dtype=float)})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _points_to_edges(points, poly) -> float:
    """Smallest distance from any of ``points`` (N, 2) to any edge of ``poly`` (vectorised)."""
    e = _edges(poly)
    a = np.array([u for u, _ in e], dtype=float)
    ab = np.array([v for _, v in e], dtype=float) - a
    pts = np.asarray(points, dtype=float)
    ap = pts[:, None, :] - a[None, :, :]
    den = np.maximum(np.sum(ab * ab, axis=1), 1e-18)
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / den[None, :], 0.0, 1.0)
    return float(np.min(np.linalg.norm(ap - t[:, :, None] * ab[None, :, :], axis=2)))


def _edges(poly):
    n = len(poly)
    if n == 1:
        return [(poly[0], poly[0])]
    if n == 2:
        return [(poly[0], poly[1])]
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


def _overlap(p, q) -> bool:
    """Separating-axis test for two convex polygons (a segment or a point counts as a degenerate polygon)."""
    axes = []
    for poly in (p, q):
        for a, b in _edges(poly):
            e = b - a
            if float(np.hypot(*e)) > 1e-12:
                axes.append(np.array([-e[1], e[0]]))
    if not axes:
        return bool(np.linalg.norm(p[0] - q[0]) < 1e-12)
    for ax in axes:
        pp, qq = p @ ax, q @ ax
        if pp.max() < qq.min() - 1e-15 or qq.max() < pp.min() - 1e-15:
            return False
    return True


def poly_distance(p, q) -> float:
    """Distance between two convex polygons (0 if they overlap); exact for convex shapes."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if _overlap(p, q):
        return 0.0
    return min(_points_to_edges(p, q), _points_to_edges(q, p))


# -- the grippers -------------------------------------------------------------------------------------------------------
def _local(p, origin, r, center) -> np.ndarray:
    """World point(s) (3,) or (N, 3) in a body's frame, relative to a shape's centre."""
    return (np.atleast_2d(np.asarray(p, dtype=float)) - np.asarray(origin, dtype=float)) @ r - np.asarray(center,
                                                                                                         dtype=float)


def _point_box(p, origin, r, center, half) -> np.ndarray:
    q = _local(p, origin, r, center)
    return np.linalg.norm(np.maximum(np.abs(q) - np.asarray(half, dtype=float), 0.0), axis=1)


def _point_cylinder(p, origin, r, center, radius, height) -> np.ndarray:
    q = _local(p, origin, r, center)
    radial = np.maximum(np.hypot(q[:, 0], q[:, 1]) - radius, 0.0)
    axial = np.maximum(np.abs(q[:, 2]) - height / 2, 0.0)
    return np.hypot(radial, axial)


def phantom_point_distance(p, ph):
    """Distance from world point(s) to the phantom's solid (0 inside it): a float for one point, an array for (N, 3)."""
    r = rotation(ph.quat_xyzw)
    o = np.asarray(ph.origin, dtype=float)
    c, rad, h = L.PH_HEAD
    d = np.min([_point_box(p, o, r, bc, np.asarray(bs) / 2) for bc, bs in L.PH_BOXES]
               + [_point_cylinder(p, o, r, c, rad, h)], axis=0)
    return float(d[0]) if np.ndim(p) == 1 else d


def marker_point_distance(p, m):
    d = _point_box(p, m.pos, rotation(m.quat_xyzw), (0.0, 0.0, 0.0), np.asarray(L.MARKER_SIZE) / 2)
    return float(d[0]) if np.ndim(p) == 1 else d


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, dist, samples: int = 20) -> float:
    """How far a gripper's finger zone is from a solid whose point distance is ``dist(points)`` (negative:
    overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return float(np.min(dist(pts))) - zone_radius(arm)


def released(arms, dist) -> tuple[bool, list | None]:
    """(both grippers' finger zones at least 5 mm from the solid, the two gaps in mm); unreadable arms: met, None."""
    if not arms_valid(arms):
        return True, None
    gaps = [zone_gap(arms[s], dist) for s in SIDES]
    return bool(min(gaps) >= L.HANDS_OFF - _EPS), [round(g * 1000, 1) for g in gaps]


# -- the tracker --------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(float(v)) else None


class Tracker:
    """What the scorer remembers within a trial: the phantom's and the R marker's last second of positions and
    headings, whether the phantom has touched the floor, and how many steps had a broken state or unreadable arms."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.dropped_at: int | None = None  # the step the phantom first touched the floor (latched)
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.prev_origin = None
        self.history: deque = deque(maxlen=2 * self.window + 1)  # (phantom xyz, heading, R marker xyz, heading)
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(ph) -> bool:
        """The starting layout: the phantom lying still on the cart on its mark, along y, hip end toward +y."""
        o = np.asarray(ph.origin, dtype=float)
        on_mark = (math.hypot(o[0] - L.PH_START[0], o[1] - L.PH_START[1]) < L.START_TOL
                   and abs(o[2] - (L.TOP + L.CART_H)) < 0.003)
        r = rotation(ph.quat_xyzw)
        along = r[1, 0] > math.cos(math.radians(10.0)) and r[2, 2] > math.cos(math.radians(2.0))
        try:
            slow = float(np.linalg.norm(np.asarray(ph.lin_vel, dtype=float))) < 0.01
        except (AttributeError, TypeError, ValueError):
            slow = True
        return bool(on_mark and along and slow)

    def _still(self, idx_pos: int, idx_yaw: int) -> tuple[bool, float | None, float | None, float | None]:
        """(still, path in the older 0.5 s window, path in the newer one, largest heading change over 1 s)."""
        if len(self.history) < 2 * self.window + 1:
            return False, None, None, None
        h = list(self.history)
        pts = [np.asarray(e[idx_pos]) for e in h]
        steps = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
        older, newer = sum(steps[:self.window]), sum(steps[self.window:])
        now = h[-1][idx_yaw]
        turn = max(abs(wrap_deg(e[idx_yaw] - now)) for e in h)
        ok = older < L.STILL_PATH and newer < L.STILL_PATH and turn < L.STILL_YAW_DEG
        return bool(ok), older, newer, turn

    def update(self, ph, r_marker, l_marker, arms=None) -> Result:
        self.steps += 1
        if not (valid_phantom(ph) and valid_marker(r_marker) and valid_marker(l_marker)):
            self.bad_steps += 1  # a blow-up: score 0 for this step, keep everything else as it was
            self.last = self.result(ph, r_marker, l_marker, arms)
            return self.last
        o = np.asarray(ph.origin, dtype=float)
        if self.prev_origin is not None and self.at_start(ph):
            if float(np.linalg.norm(o - self.prev_origin)) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()
                self.steps = 1
        self.prev_origin = o
        if self.dropped_at is None and on_floor(ph):
            self.dropped_at = self.steps
        self.history.append((o.copy(), heading_deg(rotation(ph.quat_xyzw)), np.asarray(r_marker.pos, float).copy(),
                             heading_deg(rotation(r_marker.quat_xyzw))))
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(ph, r_marker, l_marker, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def marker_check(self, ph, rm, lm, arms) -> tuple[bool, str, dict]:
        """(marker OK, why not, diagnostics): the R marker released, still, flat on the table, its centre in the
        light field, its outline 5 mm or more clear of the phantom; the L marker's centre not in the light field."""
        rel, gaps = released(arms, lambda p: marker_point_distance(p, rm))
        still, older, newer, turn = self._still(2, 3)
        clear = marker_clearance(rm, ph)
        d = {"r_released": rel, "r_zone_gap_mm": gaps, "r_still": still, "r_path_mm": [_mm(older), _mm(newer)],
             "r_turn_deg": None if turn is None else round(turn, 2), "r_tilt_deg": round(marker_tilt_deg(rm), 2),
             "r_flat": marker_flat(rm), "r_in_field": centre_in_field(rm), "r_clear_mm": _mm(clear),
             "l_in_field": centre_in_field(lm)}
        if d["l_in_field"]:
            return False, "l_marker_in_field", d
        if not d["r_in_field"]:
            return False, "r_marker_not_in_field", d
        if not d["r_flat"]:
            return False, "r_marker_not_flat_on_table", d
        if clear < L.MARKER_CLEAR - _EPS:
            return False, "r_marker_on_or_near_the_bone", d
        if not rel:
            return False, "r_marker_held", d
        if not still:
            return False, "r_marker_moving", d
        return True, "", d

    def result(self, ph, rm, lm, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"steps": self.steps, "bad_steps": self.bad_steps, "arms_unavailable": self.arms_unavailable,
                 "dropped_at": self.dropped_at, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if not (valid_phantom(ph) and valid_marker(rm) and valid_marker(lm)):
            return Result(0.0, False, {"valid_state": False, "tier": 0.0, "zero_reason": "invalid_state", **trial})
        rel, gaps = released(arms, lambda p: phantom_point_distance(p, ph))
        still, older, newer, turn = self._still(0, 1)
        off = centre_offset(ph)
        d = {"valid_state": True, "tilt_deg": round(phantom_tilt_deg(ph), 2),
             "lowest_mm": _mm(phantom_lowest(ph) - FIELD_TOP_W), "flat": flat_on_table(ph),
             "align_err_deg": round(align_err_deg(ph), 2), "heading_deg": round(heading_deg(rotation(ph.quat_xyzw)), 2),
             "hip_to_cathode": hip_toward_cathode(ph), "field_margin_mm": _mm(field_margin(ph)),
             "centre_off_mm": [_mm(off[0]), _mm(off[1])], "centre_dist_mm": _mm(float(np.hypot(*off))),
             "released": rel, "zone_gap_mm": gaps, "still": still, "path_mm": [_mm(older), _mm(newer)],
             "turn_deg": None if turn is None else round(turn, 2), "on_floor_now": on_floor(ph), **trial}
        mok, mwhy, md = self.marker_check(ph, rm, lm, arms)
        d.update(md, marker_ok=mok, marker_reason=mwhy)

        def out(score, reason):
            d.update(tier=score, zero_reason=reason if score == 0.0 else "")
            return Result(score, bool(score == 1.0), d)

        if self.dropped_at is not None or d["on_floor_now"]:
            return out(0.0, "phantom_dropped_on_floor")
        if not rel:
            return out(0.0, "phantom_held")
        if not still:
            return out(0.0, "phantom_moving")
        if not d["flat"]:
            return out(0.0, "phantom_not_flat_on_table")
        if align_err_deg(ph) > L.ALIGN_DEG + _EPS:
            return out(0.0, "phantom_not_along_tube_axis")
        if field_margin(ph) < -_EPS:
            return out(0.0, "phantom_not_wholly_in_field")
        if not d["hip_to_cathode"]:
            return out(0.25, "")
        if float(np.hypot(*off)) > L.CENTRE_TOL + _EPS:
            return out(0.5, "")
        if not mok:
            return out(0.75, "")
        return out(1.0, "")
