"""The scorer, as plain functions of the three floorboards' states and the grippers' state. NumPy only; no simulator
needed.

The rule is about the floor as it is left: every board laid flat across the bay, the three tight together, and an
expansion gap of 10 to 20 mm kept open between the floor and each of the three walls. A board pinned against a wall
(touching it or within 3 mm) is the classic floor-laying defect: the floor buckles when the wood swells. So the score
is read from the end state; the scorer is fed one observation per control step by ``task.success()`` and keeps only
what it needs for that (each board's end points over the last 0.5 s, to tell that the floor is still, and counters
for the trace).

    Tracker.update(boards, arms) -> Result(score, success, details)

``boards`` maps each board's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; each board is a box of known size
(``layout.BOARD_SIZE``), so its corners follow from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then the hands-off condition is taken as met for that step
and the step is counted in ``arms_unavailable``. A step on which any board has a non-finite number or a zero
quaternion (a simulator blow-up), or is missing, scores 0 and leaves the tracker as it was.

A board is a plain box, the same either way up and either way round, so tilt is measured from whichever broad face
is nearer to facing down and its heading only by the line of its length.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a finger's outer corner (half the current finger gap plus
the finger's 10 mm thickness, and half its 40 mm width across; 60 mm fully open, 22 mm closed).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
HALF = np.array(L.BOARD_HALF, dtype=float)
ARMS = ("left", "right")


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
    if s is None:
        return False
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- one board ---------------------------------------------------------------------------------------------------
def corners(s) -> np.ndarray:
    """The board's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * HALF) @ rotation(s.quat_xyzw).T


def lowest(s) -> float:
    """Height of the board's lowest point above the table top (m)."""
    return float(corners(s)[:, 2].min() - L.TOP)


def tilt_deg(s) -> float:
    """Angle between the vertical and the axis through the board's two broad faces (degrees, 0..90): either broad
    face may be down."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def axis_err_deg(s) -> float:
    """How far the board's length, seen from above, is turned from y, the line across the bay (degrees, 0..90;
    either end may point either way). A board standing on its end has no length seen from above: 90."""
    u = rotation(s.quat_xyzw)[:, 0]
    n = math.hypot(float(u[0]), float(u[1]))
    if n < 1e-9:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(abs(float(u[1])) / n, 0.0, 1.0))))


def ends(s) -> np.ndarray:
    """The centres of the board's two end faces (2 x 3)."""
    c, u = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)[:, 0] * HALF[0]
    return np.array([c + u, c - u])


def _hull(points) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def outline(s) -> np.ndarray:
    """The board's outline seen from above (n x 2, counter-clockwise): the shadow of its eight corners, turned and
    tilted as it is."""
    return _hull(corners(s)[:, :2])


def _point_segment(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / max(float(np.dot(d, d)), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _segments_cross(a, b, c, d) -> bool:
    """Do the closed segments ab and cd meet?"""
    d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
    if d1 * d2 < 0.0 and d3 * d4 < 0.0:  # a proper crossing
        return True
    return min(  # touching: an end of one on the other
        _point_segment(a, c, d), _point_segment(b, c, d), _point_segment(c, a, b), _point_segment(d, a, b)
    ) <= 1e-12


def _inside(p, poly) -> bool:
    if len(poly) < 3:
        return False
    return all(_cross(poly[i], poly[(i + 1) % len(poly)], p) >= -1e-15 for i in range(len(poly)))


def segment_outline_distance(seg, poly) -> float:
    """Distance seen from above between a segment ((x0, y0), (x1, y1)) and a convex outline (0 if they meet)."""
    a, b = (np.asarray(v, dtype=float) for v in seg)
    if _inside(a, poly) or _inside(b, poly):
        return 0.0
    n = len(poly)
    best = math.inf
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n] if n > 1 else poly[i]
        if _segments_cross(a, b, p, q):
            return 0.0
        best = min(best, _point_segment(p, a, b), _point_segment(a, p, q), _point_segment(b, p, q))
    return float(best)


def wall_distances(s) -> dict[str, float]:
    """Distance seen from above from the board's outline to each wall's inside face (0: touching or across it)."""
    poly = outline(s)
    return {name: segment_outline_distance(seg, poly) for name, seg in L.FACES.items()}


def span_at_y(poly, y0: float) -> tuple[float, float] | None:
    """Where the line y = y0 crosses a convex outline: (smallest x, largest x), or None if it misses it."""
    xs = []
    n = len(poly)
    for i in range(n):
        (x0, ya), (x1, yb) = poly[i], poly[(i + 1) % n]
        if (ya - y0) * (yb - y0) <= 0.0:
            if abs(yb - ya) < 1e-15:
                xs += [float(x0), float(x1)]
            else:
                xs.append(float(x0 + (y0 - ya) * (x1 - x0) / (yb - ya)))
    if not xs:
        return None
    return min(xs), max(xs)


def off_table(s) -> bool:
    return lowest(s) < -L.OFF_TABLE


def board_info(s) -> dict:
    """Everything the rules ask of one board, at this step."""
    poly = outline(s)
    low, tilt, axis = lowest(s), tilt_deg(s), axis_err_deg(s)
    walls = {name: segment_outline_distance(seg, poly) for name, seg in L.FACES.items()}
    xmin, xmax = float(poly[:, 0].min()), float(poly[:, 0].max())
    ymin, ymax = float(poly[:, 1].min()), float(poly[:, 1].max())
    gap_left, gap_right = L.BAY_HALF - ymax, ymin + L.BAY_HALF
    pinned = bool(low <= L.WALL_ZONE_Z + L.EPS and min(walls.values()) <= L.WALL_ZERO + L.EPS)
    in_bay = bool(xmin >= L.BAY_X0 - L.EPS and xmax <= L.FAR_X + L.EPS)
    gaps_ok = all(L.GAP_MIN - L.EPS <= g <= L.GAP_MAX + L.EPS for g in (gap_left, gap_right))
    laid = bool(tilt <= L.FLAT_DEG + 1e-9 and low <= L.ON_FLOOR + L.EPS and axis <= L.AXIS_DEG + 1e-9 and in_bay
                and gaps_ok and not pinned)
    return {"poly": poly, "low": low, "tilt": tilt, "axis": axis, "walls": walls, "xmin": xmin, "xmax": xmax,
            "gap_left": gap_left, "gap_right": gap_right, "pinned": pinned, "in_bay": in_bay, "laid": laid}


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger's outer corner: half the gap plus the finger's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def _points_box_distance(pts, s) -> float:
    """Smallest distance from a set of points (n, 3) to the board (0 inside it), in the board's own frame."""
    d = (np.asarray(pts, dtype=float) - np.asarray(s.pos, dtype=float)) @ rotation(s.quat_xyzw)
    return float(np.linalg.norm(np.maximum(np.abs(d) - HALF, 0.0), axis=1).min())


def zone_gap(arm, boards: dict) -> float:
    """How far a gripper's finger zone is from the nearest board (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(_points_box_distance(pts, s) for s in boards.values()) - r


def hands_off(arms, boards: dict) -> bool:
    """Both grippers' finger zones at least 30 mm from every board: neither can be pushing, holding or touching one."""
    return bool(all(zone_gap(arms[side], boards) >= L.HANDS_OFF - L.EPS for side in ARMS))


# -- the floor ---------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


def joints(boards: dict, infos: dict, laid: list[str]) -> list[float]:
    """The joints between neighbouring laid boards, far to near: for each pair, the gap along x between the two
    outlines on each of the two measuring lines (y +-0.140), the wider one (inf if a line misses a board)."""
    order = sorted(laid, key=lambda n: -float(boards[n].pos[0]))
    out = []
    for far, near in zip(order, order[1:]):
        widest = -math.inf
        for y0 in L.JOINT_LINES_Y:
            a, b = span_at_y(infos[far]["poly"], y0), span_at_y(infos[near]["poly"], y0)
            widest = max(widest, math.inf if a is None or b is None else a[0] - b[1])
        out.append(widest)
    return out


def evaluate(boards: dict, arms=None) -> tuple[float, dict]:
    """The score of the floor as it stands, and what it was read from. ``boards`` must all be valid."""
    infos = {n: board_info(s) for n, s in boards.items()}
    pinned = [n for n, i in infos.items() if i["pinned"]]
    laid = [n for n in L.BOARD_NAMES if n in infos and infos[n]["laid"]]
    far_gap = L.FAR_X - max(infos[n]["xmax"] for n in laid) if laid else None
    js = joints(boards, infos, laid)
    have_arms = arms_valid(arms)
    zones = {side: zone_gap(arms[side], boards) for side in ARMS} if have_arms else None
    off = all(z >= L.HANDS_OFF - L.EPS for z in zones.values()) if have_arms else True  # unreadable: met
    tight = len(js) == 2 and all(j <= L.JOINT_MAX + L.EPS for j in js)
    far_ok = far_gap is not None and L.GAP_MIN - L.EPS <= far_gap <= L.GAP_MAX + L.EPS
    if pinned:
        score = 0.0
    elif len(laid) == 3 and far_ok and tight and off:
        score = 1.0
    else:
        score = 0.25 * len(laid)
    details = {
        "n_laid": len(laid),
        "laid": laid,
        "pinned": pinned,
        "far_gap_mm": _mm(far_gap),
        "joints_mm": [_mm(j) for j in js],
        "hands_off": bool(off),
        "arms_available": bool(have_arms),
        "left_zone_mm": _mm(zones["left"]) if have_arms else None,
        "right_zone_mm": _mm(zones["right"]) if have_arms else None,
    }
    for n, i in infos.items():
        details[n] = {
            "laid": i["laid"],
            "low_mm": _mm(i["low"]),
            "tilt_deg": round(i["tilt"], 2),
            "axis_err_deg": round(i["axis"], 2),
            "in_bay": i["in_bay"],
            "gap_left_mm": _mm(i["gap_left"]),
            "gap_right_mm": _mm(i["gap_right"]),
            "wall_mm": {w: _mm(d) for w, d in i["walls"].items()},
            "x_mm": _mm(float(boards[n].pos[0])),
            "y_mm": _mm(float(boards[n].pos[1])),
        }
    return score, details


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer keeps over a trial: each board's end points over the last 0.5 s, and counters."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # steps with a board missing or non-finite (scored 0, tracker untouched)
        self.arms_unavailable = 0  # steps on which rigstate could not read the arms
        self.history: dict[str, deque] = {}  # board -> end points at this step and the 10 before it
        self.prev: dict[str, np.ndarray] | None = None  # each board's centre at the previous valid step
        self.first_all_laid_step: int | None = None
        self.first_full_step: int | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(boards: dict) -> bool:
        """The starting layout: every board flat on the table on its mark (within 45 mm), lying along x."""
        for n, (x, y) in L.BOARD_START.items():
            s = boards.get(n)
            if s is None or not valid(s):
                return False
            on_mark = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            if not (on_mark and abs(lowest(s)) < 0.003 and tilt_deg(s) < 2.0 and axis_err_deg(s) > 80.0):
                return False
        return True

    def moved(self) -> dict[str, float | None]:
        """How far each board's ends moved at most over the last 0.5 s (None until 0.5 s has been seen)."""
        out = {}
        for n in L.BOARD_NAMES:
            h = self.history.get(n)
            if h is None or len(h) < self.window + 1:
                out[n] = None
                continue
            pts = np.array(h)  # (k, 2, 3)
            out[n] = float(np.linalg.norm(pts - pts[-1], axis=2).max())
        return out

    def update(self, boards: dict, arms=None) -> Result:
        if any(not valid(boards.get(n)) for n in L.BOARD_NAMES):
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        centres = {n: np.asarray(boards[n].pos, dtype=float) for n in L.BOARD_NAMES}
        if self.prev is not None and self.at_start(boards):
            jump = max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in L.BOARD_NAMES)
            if jump > L.RESET_JUMP:  # the simulator put the boards back: a new trial
                self.reset()
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        for n in L.BOARD_NAMES:
            self.history.setdefault(n, deque(maxlen=self.window + 1)).append(ends(boards[n]))
        self.prev = centres
        self.last = self.result(boards, arms)
        if self.first_all_laid_step is None and self.last.details.get("n_laid") == 3:
            self.first_all_laid_step = self.steps
        if self.first_full_step is None and self.last.score == 1.0:
            self.first_full_step = self.steps
        self.last.details.update(first_all_laid_step=self.first_all_laid_step, first_full_step=self.first_full_step)
        return self.last

    def result(self, boards: dict, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (it changes nothing)."""
        if any(not valid(boards.get(n)) for n in L.BOARD_NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        score, details = evaluate({n: boards[n] for n in L.BOARD_NAMES}, arms)
        moved = self.moved()
        still = all(m is not None and m <= L.STILL_TOL + L.EPS for m in moved.values())
        details.update(valid_state=True, still=still, moved_mm={n: _mm(m) for n, m in moved.items()},
                       steps=self.steps, bad_steps=self.bad_steps, arms_unavailable=self.arms_unavailable)
        return Result(score, bool(score == 1.0 and still), details)
