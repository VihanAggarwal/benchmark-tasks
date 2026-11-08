"""The scorer, as plain functions of the three cots' and three pillows' states. NumPy only; no simulator needed.

The rest area is ready when the three cots lie flat on the rug, pointing forward as they started, spread so that
neighbouring cots have at least 114 mm of floor between them (3 ft at 1:8), and each cot has one pillow at one of its
ends, neighbours at opposite ends (head to toe). The score is read from the state at hand (the last step); the scorer
keeps only a short history of where each object was, to tell whether it is at rest, and is fed one observation per
control step by ``task.success()``:

    Tracker.update({"cot_left": state, ..., "pillow_right": state}) -> Result(score, success, details)

A state is anything with ``pos`` (the box's centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``
(world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. A missing object, or a state with a
non-finite number or a zero quaternion (a simulator blow-up), scores 0 for that step and leaves the tracker as it was.
The three cots are alike and so are the three pillows: names are labels only. Neighbours are read from the cots' order
across the room (by y), and any pillow may end on any cot.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

NAMES = (*L.COTS, *L.PILLOWS)
SIZES = {**{n: L.COT_SIZE for n in L.COTS}, **{n: L.PILLOW_SIZE for n in L.PILLOWS}}


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
        parts = [np.asarray(getattr(s, k), dtype=float).ravel()
                 for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def box_corners(s, size) -> np.ndarray:
    """(8, 3): the corners of a box of this size, centred at the state's pos and turned by its quaternion."""
    r = rotation(s.quat_xyzw)
    h = np.asarray(size, dtype=float) / 2
    local = np.array([(sx * h[0], sy * h[1], sz * h[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return local @ r.T + np.asarray(s.pos, dtype=float)


def tilt_deg(r: np.ndarray) -> float:
    """Angle between the vertical and the axis through the box's two broad faces (local z), either face up."""
    a = math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))
    return min(a, 180.0 - a)


def heading_err_deg(r: np.ndarray) -> float:
    """How far the long side (local x) is turned from x seen from above, 0..90 degrees, either way round. A box
    standing on its end has no heading: 90."""
    ux, uy = float(r[0, 0]), float(r[1, 0])
    if math.hypot(ux, uy) < 1e-6:
        return 90.0
    a = math.degrees(math.atan2(uy, ux)) % 180.0
    return float(min(a, 180.0 - a))


def long_axis_xy(r: np.ndarray) -> np.ndarray:
    """Unit vector of the long side seen from above, pointed toward +x (toward the far end)."""
    u = np.array([r[0, 0], r[1, 0]], dtype=float)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return np.array([1.0, 0.0])
    u /= n
    return u if u[0] > 0 or (u[0] == 0 and u[1] >= 0) else -u


def hull(points) -> np.ndarray:
    """Convex hull (counter-clockwise) of 2-D points, monotone chain."""
    pts = sorted({(round(float(p[0]), 12), round(float(p[1]), 12)) for p in points})
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


def _seg_point(p, a, b) -> float:
    ab = b - a
    t = 0.0 if float(ab @ ab) == 0 else max(0.0, min(1.0, float((p - a) @ ab) / float(ab @ ab)))
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(p, poly) -> bool:
    """Is the point inside (or on) a counter-clockwise convex polygon?"""
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            return False
    return True


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2, d3, d4 = orient(c, d, a), orient(c, d, b), orient(a, b, c), orient(a, b, d)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def poly_distance(pa: np.ndarray, pb: np.ndarray) -> float:
    """Smallest distance between two convex polygons seen from above; 0 if they touch or overlap."""
    if any(_inside(p, pb) for p in pa) or any(_inside(p, pa) for p in pb):
        return 0.0
    na, nb = len(pa), len(pb)
    for i in range(na):
        for j in range(nb):
            if _segments_cross(pa[i], pa[(i + 1) % na], pb[j], pb[(j + 1) % nb]):
                return 0.0
    best = math.inf
    for p in pa:
        for j in range(nb):
            best = min(best, _seg_point(p, pb[j], pb[(j + 1) % nb]))
    for p in pb:
        for i in range(na):
            best = min(best, _seg_point(p, pa[i], pa[(i + 1) % na]))
    return best


def outline(s, size) -> np.ndarray:
    """The box's outline seen from above: the convex hull of its eight corners."""
    return hull(box_corners(s, size)[:, :2])


def rug_margin(poly: np.ndarray) -> float:
    """How far the outline stays inside the rug's edges at its closest point (negative: over an edge)."""
    return float(min(np.min(poly[:, 0]) - L.RUG_X0, L.RUG_X1 - np.max(poly[:, 0]),
                     np.min(poly[:, 1]) - L.RUG_Y0, L.RUG_Y1 - np.max(poly[:, 1])))


def track_points(s, size) -> np.ndarray:
    """(2, 3): the box's centre and the centre of one end face (half its length out along local x), the points whose
    movement says whether it is at rest (both catch a slide, the second also a turn on the spot)."""
    r = rotation(s.quat_xyzw)
    c = np.asarray(s.pos, dtype=float)
    return np.array([c, c + r[:, 0] * size[0] / 2])


def cot_facts(s) -> dict:
    """Everything about one cot's pose that the rule needs (rest is the tracker's)."""
    r = rotation(s.quat_xyzw)
    poly = outline(s, L.COT_SIZE)
    low = float(box_corners(s, L.COT_SIZE)[:, 2].min())
    tilt = tilt_deg(r)
    heading = heading_err_deg(r)
    margin = rug_margin(poly)
    flat = bool(abs(low - L.RUG_TOP) <= L.FLAT_TOL + 1e-12 and tilt <= L.TILT_MAX_DEG + 1e-9)
    return {
        "centre": (float(s.pos[0]), float(s.pos[1])),
        "outline": poly,
        "axis": long_axis_xy(r),
        "top": top_face(s, r),
        "flat": flat,
        "tilt_deg": tilt,
        "heading_err_deg": heading,
        "pointing": bool(heading <= L.HEADING_MAX_DEG + 1e-9),
        "on_rug": bool(margin >= -1e-9),
        "rug_margin_mm": margin * 1000,
        "low_mm": (low - L.RUG_TOP) * 1000,
    }


def top_face(s, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centre, unit normal) of whichever broad face of the cot is the upper one."""
    n = r[:, 2] if r[2, 2] >= 0 else -r[:, 2]
    return np.asarray(s.pos, dtype=float) + n * L.COT_SIZE[2] / 2, n


def top_height(cot: dict, xy) -> float:
    """Height of the cot's upper face (its plane) above the point xy; for a cot on its side, its highest point."""
    c, n = cot["top"]
    if n[2] < 0.2:
        return float(c[2] + L.COT_SIZE[1] / 2)
    return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])


def pillow_on(p, cot: dict) -> tuple[bool, float, float]:
    """(on, along, low gap): is the pillow lying on this cot (its lowest point within 3 mm of the cot's top face
    under it, its centre over the cot's outline)? 'along' is its centre's offset along the cot's long side from the
    cot's centre, + toward the far end (m); 'low gap' its lowest point above the cot's top (m)."""
    low = float(box_corners(p, L.PILLOW_SIZE)[:, 2].min())
    gap = low - top_height(cot, p.pos[:2])
    rel = np.asarray(p.pos[:2], dtype=float) - np.asarray(cot["centre"], dtype=float)
    u = cot["axis"]
    v = np.array([-u[1], u[0]])
    along, across = float(rel @ u), float(rel @ v)
    over = abs(along) <= L.COT_SIZE[0] / 2 + 1e-12 and abs(across) <= L.COT_SIZE[1] / 2 + 1e-12
    return bool(over and abs(gap) <= L.ON_COT_TOL + 1e-12), along, gap


def end_marked(along: float) -> str:
    """'near' (the end nearer the robot, smaller x), 'far', or 'none' for a pillow centre this far along the cot."""
    reach = L.COT_HALF_LEN - L.END_BAND  # 32.5 mm from the cot's centre
    if along >= reach - 1e-12:
        return "far"
    if along <= -reach + 1e-12:
        return "near"
    return "none"


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: where each cot and pillow was over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = int(round(L.REST_WINDOW_S * control_hz))  # 10 steps at 20 Hz: 11 samples span 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hist = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev = None  # each object's centre at the previous valid control step
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(objects: dict) -> bool:
        """The starting layout: every cot lying flat on the rug and every pillow on its cot, each on its mark."""
        for n in NAMES:
            s = objects[n]
            sx, sy = L.START[n]
            if math.hypot(float(s.pos[0]) - sx, float(s.pos[1]) - sy) >= L.START_TOL:
                return False
            base = L.RUG_TOP if n in L.COTS else L.COT_TOP
            if abs(float(box_corners(s, SIZES[n])[:, 2].min()) - base) > L.FLAT_TOL:
                return False
        return True

    def at_rest(self, name: str) -> bool:
        """Neither tracked point of this object has moved more than 2 mm in the last 0.5 s (from positions, not the
        reported velocity, which chatters while a part is gripped)."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return False
        now = h[-1]
        return bool(all(float(np.max(np.linalg.norm(p - now, axis=1))) <= L.REST_MOVE + 1e-12 for p in h))

    def update(self, objects: dict) -> Result:
        if not all(valid(objects.get(n)) for n in NAMES):  # a blow-up: score 0 for this step, keep the record
            self.steps += 1
            self.last = self.result(objects)
            return self.last
        centres = {n: np.asarray(objects[n].pos, dtype=float) for n in NAMES}
        if self.prev is not None and self.at_start(objects):
            jump = max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in NAMES)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        for n in NAMES:
            self.hist[n].append(track_points(objects[n], SIZES[n]))
        self.prev = centres
        self.last = self.result(objects)
        return self.last

    def result(self, objects: dict) -> Result:
        if not all(valid(objects.get(n)) for n in NAMES):
            return Result(0.0, False, {"valid_state": False})
        cots = {n: cot_facts(objects[n]) for n in L.COTS}
        cot_rows, pillow_rows = {}, {}
        for n, f in cots.items():
            rest = self.at_rest(n)
            f["set_down"] = bool(f["flat"] and f["pointing"] and f["on_rug"] and rest)
            cot_rows[n] = {
                "set_down": f["set_down"],
                "at_rest": rest,
                "flat": f["flat"],
                "tilt_deg": round(f["tilt_deg"], 2),
                "heading_err_deg": round(f["heading_err_deg"], 2),
                "on_rug": f["on_rug"],
                "rug_margin_mm": round(f["rug_margin_mm"], 1),
                "low_mm": round(f["low_mm"], 2),
                "centre_mm": [round(f["centre"][0] * 1000, 1), round(f["centre"][1] * 1000, 1)],
                "pillows": [],
            }
        for pn in L.PILLOWS:
            p = objects[pn]
            rest = self.at_rest(pn)
            on, along, gap = None, None, None
            best = None
            for cn, f in cots.items():
                ok, a, g = pillow_on(p, f)
                if ok and (best is None or abs(g) < abs(best[2])):
                    best = (cn, a, g)
            if best is not None and rest:
                on, along, gap = best
                cot_rows[on]["pillows"].append(pn)
            pillow_rows[pn] = {
                "on_cot": on,
                "at_rest": rest,
                "end": end_marked(along) if on else "none",
                "along_mm": None if along is None else round(along * 1000, 1),
                "low_gap_mm": None if gap is None else round(gap * 1000, 2),
                "centre_mm": [round(float(p.pos[0]) * 1000, 1), round(float(p.pos[1]) * 1000, 1),
                              round((float(p.pos[2]) - L.TOP) * 1000, 1)],
            }
        reasons = []
        if not all(f["set_down"] for f in cots.values()):
            reasons.append("cot not set down")
        if any(r["on_cot"] is None for r in pillow_rows.values()):
            reasons.append("pillow not on a cot")
        if any(len(r["pillows"]) > 1 for r in cot_rows.values()):
            reasons.append("two pillows on one cot")
        order = sorted(L.COTS, key=lambda n: -cots[n]["centre"][1])  # across the room, +y (robot's left) first
        pairs = []
        for a, b in zip(order, order[1:]):
            gap = poly_distance(cots[a]["outline"], cots[b]["outline"])
            ends = []
            for c in (a, b):
                ps = cot_rows[c]["pillows"]
                ends.append(pillow_rows[ps[0]]["end"] if len(ps) == 1 else "none")
            opposite = bool(ends[0] != "none" and ends[1] != "none" and ends[0] != ends[1])
            pairs.append({"cots": [a, b], "gap_mm": round(gap * 1000, 1), "gap_ok": bool(gap >= L.GAP_MIN - 1e-9),
                          "ends": ends, "opposite": opposite})
        score = 0.0 if reasons else 0.25 * sum(p["gap_ok"] for p in pairs) + 0.25 * sum(p["opposite"] for p in pairs)
        details = {
            "valid_state": True,
            "zero": "; ".join(reasons),
            "order": order,
            "pairs": pairs,
            "cots": cot_rows,
            "pillows": pillow_rows,
        }
        return Result(score, bool(score == 1.0), details)
