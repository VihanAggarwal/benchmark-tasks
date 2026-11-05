"""The scorer, as plain functions of the three bars' states. NumPy only; no simulator needed.

The rule is about the finished mat, read from the state at the last control step: three bars across both
bolsters, parallel, 100 mm apart centre to centre, every bar end at least 20 mm from the side forms (the concrete
cover). Any bar may go in any place. The forms and bolsters are fixed, so their faces come from ``layout``. A small
tracker is still fed once per control step by ``task.success()``: it keeps what the trace needs (when each score
was first reached, how many observations were broken) and recognises a new episode when the simulator puts every
bar back on its starting mark in one step.

    Tracker.update(bars) -> Result(score, success, details)

``bars`` (3) are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low``
and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. Each is a plain box whose
size is in ``layout`` (body x across, body y along its length, body z up as it lies in the mat). Its cross-section
is square, so it may lie on any of its four long faces and either way round. A state with a non-finite number or
a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

BAR_HALF = np.array(L.BAR_SIZE, dtype=float) / 2
BAR_W = L.BAR_SIZE[0]  # the bar's section (16 x 16 mm)
_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # a face's corners, in order round it
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the
    world. (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN, inf or zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the normal of whichever of the bar's four long faces is nearest to
    facing up (degrees, 0..90). The section is square, so a bar is the same on any long face: lying flat 0, one
    end raised by an angle a gives a, rolled onto an edge (arris) 45, standing on an end 90."""
    r = rotation(quat_xyzw)
    up = max(abs(r[2, 0]), abs(r[2, 2]))  # body x and z are the long faces' normals
    return float(np.degrees(np.arccos(np.clip(up, 0.0, 1.0))))


def heading_err_deg(s, axis: str = "y") -> float:
    """How far the bar's length, seen from above, is turned from the world's ``axis`` ('x' or 'y'), in degrees
    0..90 (a bar is the same either way round)."""
    long_axis = rotation(s.quat_xyzw)[:, 1]
    hx, hy = float(long_axis[0]), float(long_axis[1])
    if math.hypot(hx, hy) < 1e-9:  # standing on its end: no heading
        return 90.0
    ang = math.degrees(math.atan2(hy, hx)) % 180.0  # 0 = along x
    if axis == "y":
        ang = (ang - 90.0) % 180.0
    return float(min(ang, 180.0 - ang))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def corners(s) -> np.ndarray:
    """The bar's eight corners (world), (8, 3)."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    return np.array([c + r @ (BAR_HALF * np.array([i, j, k]))
                     for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])


def off_table(s) -> bool:
    """The bar's lowest point is more than 10 mm below the table top: it fell off the table."""
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def on_edge(s) -> bool:
    """Standing on an end or an edge (tilted more than 30 degrees)."""
    return tilt_deg(s.quat_xyzw) > L.EDGE_DEG


# -- outlines, faces and 'rests on' --------------------------------------------------------------------------------
def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


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

    lower, upper = half(pts), half(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1])


def outline(s) -> np.ndarray:
    """The bar's real outline seen from above (the shadow of its eight corners, turned and tilted as it is)."""
    return _hull(corners(s)[:, :2])


def _overlaps_rect(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Does a convex polygon (n, 2) overlap the axis-aligned rectangle [x0, x1] x [y0, y1]? (Separating axes.)"""
    if len(poly) < 3:
        return False
    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.hypot(*e) > 1e-12:
            axes.append(np.array([-e[1], e[0]]))
    for a in axes:
        p, q = poly @ a, rect @ a
        if p.max() <= q.min() or q.max() <= p.min():
            return False
    return True


def _centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-15:
        return p.mean(axis=0)
    cx = float(((x + np.roll(x, -1)) * cross).sum()) / (6 * a)
    cy = float(((y + np.roll(y, -1)) * cross).sum()) / (6 * a)
    return np.array([cx, cy])


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex polygon ``clip`` (both CCW)."""
    out = [np.asarray(v, dtype=float) for v in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        inp, out = out, []
        if not inp:
            return None
        prev = inp[-1]
        for cur in inp:
            sp, sc = side(prev), side(cur)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
            prev = cur
    return np.array(out) if len(out) >= 3 else None


class Face:
    """A flat rectangular face: its centre, its normal (pointing up), its four corners (world)."""

    def __init__(self, centre, normal, corners_) -> None:
        self.c = np.asarray(centre, dtype=float)
        self.n = np.asarray(normal, dtype=float)
        self.corners = np.asarray(corners_, dtype=float)

    def z_at(self, xy) -> float:
        """Height of the face's plane above the point ``xy``."""
        n, c = self.n, self.c
        if abs(n[2]) < 1e-6:
            return float(c[2])
        return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / n[2])

    def outline(self) -> np.ndarray:
        """The face's outline seen from above, counter-clockwise, (4, 2)."""
        return _ccw(self.corners[:, :2])


def underside(s) -> Face:
    """The bar's lower face: whichever of its four long faces is nearest to facing down."""
    r, c = rotation(s.quat_xyzw), np.asarray(s.pos, dtype=float)
    if abs(r[2, 0]) >= abs(r[2, 2]):  # the section axis nearer to vertical is the face normal; the other spans it
        ax, other = r[:, 0], r[:, 2]
    else:
        ax, other = r[:, 2], r[:, 0]
    up = ax if ax[2] >= 0.0 else -ax
    centre = c - up * BAR_HALF[0]  # the section is square: either section axis has half-size BAR_HALF[0]
    face = [centre + r[:, 1] * i * BAR_HALF[1] + other * j * BAR_HALF[0] for i, j in _SIGNS]
    return Face(centre, up, face)


def bolster_top(i: int) -> Face:
    """The top face of fixed bolster ``i`` (0: y +0.12, 1: y -0.12)."""
    by, a, b = L.BOLSTER_YS[i], L.BOLSTER_LEN / 2, L.BOLSTER_W / 2
    centre = (L.BOLSTER_X, by, L.BOLSTER_TOP)
    return Face(centre, (0.0, 0.0, 1.0), [(L.BOLSTER_X + i_ * a, by + j * b, L.BOLSTER_TOP) for i_, j in _SIGNS])


def crossing(lower: Face, upper: Face) -> tuple[float, float | None]:
    """Where ``upper``'s underside crosses ``lower``'s top, seen from above: (the area of the overlap, the height
    of ``upper``'s underside above ``lower``'s top at the overlap's centroid; None if they do not overlap)."""
    part = _clip(upper.outline(), lower.outline())
    if part is None:
        return 0.0, None
    area = _area(part)
    if not np.isfinite(area) or area <= 1e-12:
        return 0.0, None
    at = _centroid(part)
    return float(area), upper.z_at(at) - lower.z_at(at)


def rests_on_bolster(s, i: int) -> bool:
    """The bar rests on bolster ``i``: seen from above its underside overlaps the bolster's top by at least half of
    a square crossing of their widths (96 square mm), and is within 3 mm of it there (above or below it)."""
    area, gap = crossing(bolster_top(i), underside(s))
    return bool(gap is not None and area >= L.REST_COVER * BAR_W * L.BOLSTER_W and abs(gap) <= L.REST_Z_TOL)


def on_bolsters(s) -> bool:
    """Flat (tilted under 5 degrees) and resting on both bolsters."""
    return bool(tilt_deg(s.quat_xyzw) < L.FLAT_DEG and all(rests_on_bolster(s, i) for i in range(len(L.BOLSTER_YS))))


def laid(s) -> bool:
    """On both bolsters with its length within 3 degrees of y (either way round)."""
    return bool(on_bolsters(s) and heading_err_deg(s, "y") <= L.HEADING_DEG)


def end_covers(s) -> tuple[float, float]:
    """(left, right) end cover: how far the bar's outline stays inside the left side form's inner face (y +0.23)
    and the right one's (y -0.23). Negative if it reaches past a face."""
    ys = corners(s)[:, 1]
    return float(L.SLAB_Y - ys.max()), float(ys.min() + L.SLAB_Y)


def min_cover(s) -> float:
    return min(end_covers(s))


def across_form(s) -> bool:
    """Lying on or across a form board: its outline seen from above overlaps a board and its lowest point is no
    more than 20 mm above the board's top (a bar that would stick through the slab's edge)."""
    if float(corners(s)[:, 2].min()) > L.FORM_TOP + L.FORM_CLEAR:
        return False
    poly = outline(s)
    return any(_overlaps_rect(poly, x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2) for _, x, y, sx, sy in L.FORMS)


def x_at(s, y: float) -> float:
    """Where the bar's centre line, seen from above, crosses the line at ``y`` (its x there)."""
    a = rotation(s.quat_xyzw)[:, 1]
    if abs(a[1]) < 1e-9:
        return float("nan")
    return float(s.pos[0] + (y - float(s.pos[1])) * a[0] / a[1])


def spacing(a, b) -> tuple[float, float]:
    """Centre-to-centre spacing of two bars where they cross each bolster (left, right): the distance along x
    between their centre lines there, as a tape laid along the bolster reads it."""
    return tuple(abs(x_at(b, y) - x_at(a, y)) for y in L.BOLSTER_YS)  # type: ignore[return-value]


def _dev(gaps) -> float:
    """Largest departure of a set of spacings from 100 mm."""
    return float(max(abs(g - L.SPACING) for g in gaps))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(bars) -> tuple[float, dict]:
    """The score of the mat as it stands, and what was found (JSON-clean)."""
    bars = list(bars)
    fell = [i for i, s in enumerate(bars) if off_table(s)]
    edge = [i for i, s in enumerate(bars) if on_edge(s)]
    form = [i for i, s in enumerate(bars) if across_form(s)]
    seated = [on_bolsters(s) for s in bars]
    lay = [bool(seated[i] and heading_err_deg(s, "y") <= L.HEADING_DEG) for i, s in enumerate(bars)]
    cov = [min_cover(s) for s in bars]
    good = [bool(lay[i] and cov[i] >= L.COVER_FULL) for i in range(len(bars))]
    fair = [bool(lay[i] and cov[i] >= L.COVER_PART) for i in range(len(bars))]
    d: dict = {
        "off_table": fell, "across_form": form, "on_edge": edge,
        "max_tilt_deg": round(max(tilt_deg(s.quat_xyzw) for s in bars), 2),
        "on_bolsters": seated, "laid": lay, "good": good,
        "heading_err_deg": [round(heading_err_deg(s, "y"), 2) for s in bars],
        "cover_mm": [_mm(c) for c in cov],
        "mat_order": [], "gaps_mm": [], "spacing_dev_mm": None, "pair": None,
    }
    if fell or edge or form:
        return 0.0, d

    # The mat: the bars resting flat on both bolsters, in order along x (by centre).
    mat = sorted((i for i in range(len(bars)) if seated[i]), key=lambda i: float(bars[i].pos[0]))
    d["mat_order"] = mat
    pairs = list(zip(mat, mat[1:]))
    gaps = {p: spacing(bars[p[0]], bars[p[1]]) if lay[p[0]] and lay[p[1]] else None for p in pairs}
    d["gaps_mm"] = [[_mm(g) for g in gaps[p]] if gaps[p] is not None else None for p in pairs]

    # 1.0 / 0.75: all three bars laid with their ends covered, both spacings right at both bolsters.
    if len(mat) == len(bars) and all(fair) and all(gaps[p] is not None for p in pairs):
        dev = max(_dev(gaps[p]) for p in pairs)
        d["spacing_dev_mm"] = _mm(dev)
        if dev <= L.GAP_TIGHT and all(good):
            return 1.0, d
        if dev <= L.GAP_LOOSE:
            return 0.75, d

    # 0.5: two neighbouring bars in the mat (no other bar on the bolsters between them) both right, 100 +- 5 mm.
    for p in pairs:
        if good[p[0]] and good[p[1]] and gaps[p] is not None and _dev(gaps[p]) <= L.GAP_TIGHT:
            d["pair"] = list(p)
            return 0.5, d

    # 0.25: at least one bar laid across both bolsters with its ends covered.
    return (0.25 if any(good) else 0.0), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the trace needs about this trial (the score itself depends only on the state at hand)."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every bar at the previous valid control step

    @staticmethod
    def at_start(bars) -> bool:
        """The starting layout: every bar still, flat on its rack, along x, on its own starting mark."""
        for s, (_, y) in zip(bars, L.BARS, strict=True):
            on_mark = math.hypot(float(s.pos[0]) - L.BAR_START_X, float(s.pos[1]) - y) < L.START_TOL
            on_rack = abs(float(s.low[2]) - L.RACK_TOP) < 0.003
            square = tilt_deg(s.quat_xyzw) < 1.0 and heading_err_deg(s, "x") < 1.0
            if not (on_mark and on_rack and square and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some bar moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def update(self, bars) -> Result:
        bars = list(bars)
        self.steps += 1
        if not all(valid(s) for s in bars):  # a blow-up: score 0 for this step, keep everything as it was
            self.bad_steps += 1
            self.last = self.result(bars)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in bars]
        if self.prev is not None and self._teleported(now) and self.at_start(bars):
            self.reset()  # the simulator put every bar back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.last = self.result(bars)
        for tier in self.TIERS:
            if self.last.score >= tier and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, bars) -> Result:
        bars = list(bars)
        bad = [i for i, s in enumerate(bars) if not valid(s)]
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())}}
        if bad:
            return Result(0.0, False, {"valid_state": False, "bad_bars": bad, **trial})
        score, details = evaluate(bars)
        still = all(is_resting(s) for s in bars)
        success = bool(score == 1.0 and still)
        return Result(score, success, {"valid_state": True, "all_still": still, **details, **trial})
