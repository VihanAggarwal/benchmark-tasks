"""The scorer, as plain functions of the three barrier bars' states. NumPy only; no simulator needed.

The rule is about the barrier as it is left: the three bars standing on the tank floor in one straight line across
the tank, 110-130 mm beyond the dipper and parallel to it; then the four openings along that line (the gap at the
left wall, the two slits, the gap at the right wall) are what a technician checks with a rule before switching the
dipper on. So the score is read from the end state; the scorer is fed one observation per control step by
``task.success()`` and keeps only what it needs for that (each bar's end points over the last 0.5 s, to tell that
the barrier is still, and counters for the trace).

    Tracker.update(bars) -> Result(score, success, details)

``bars`` maps each bar's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low``
and ``high``, as ``yam_arena.task_file.ObjectState`` has; each bar is a box of known size (``layout.BAR_SIZE``), so
its corners follow from ``pos`` and ``quat_xyzw``. A step on which any bar has a non-finite number or a zero
quaternion (a simulator blow-up), or is missing, scores 0 and leaves the tracker's record of the bars as it was
(only its step counters move).

A bar 'stands' when its lowest point is within 2 mm of the tank floor's top, its 25 mm axis within 10 degrees of
vertical and its outline seen from above inside the tank's inside faces (3 mm allowed for contact with a wall, so a
bar propped on a wall's top does not stand). The three form the 'line' when all stand, each is within 5 degrees of
across the tank, their centres are within 6 mm of each other along x and each is 110-130 mm beyond the dipper.

A bar is a plain box whose two 20 mm x length faces are alike and whose two ends are alike, so it may stand either
way up and either end may point either way: tilt is measured from whichever of those faces is nearer to facing
down and its heading only by the line of its length.
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
HALF = {n: np.array(L.BAR_SIZE[n], dtype=float) / 2 for n in L.BAR_NAMES}


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


# -- one bar -----------------------------------------------------------------------------------------------------
def corners(s, name: str) -> np.ndarray:
    """The bar's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * HALF[name]) @ rotation(s.quat_xyzw).T


def lowest(s, name: str) -> float:
    """Height of the bar's lowest point above the table top (m)."""
    return float(corners(s, name)[:, 2].min() - L.TOP)


def tilt_deg(s) -> float:
    """Angle between the vertical and the bar's 25 mm axis (degrees, 0..90): either 20 mm x length face may be
    down."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def axis_err_deg(s) -> float:
    """How far the bar's length, seen from above, is turned from y, the line across the tank and along the dipper
    (degrees, 0..90; either end may point either way). A bar standing on its end has no length seen from above: 90."""
    u = rotation(s.quat_xyzw)[:, 1]
    n = math.hypot(float(u[0]), float(u[1]))
    if n < 1e-9:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(abs(float(u[1])) / n, 0.0, 1.0))))


def ends(s, name: str) -> np.ndarray:
    """The centres of the bar's two end faces (2 x 3)."""
    c, u = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)[:, 1] * HALF[name][1]
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


def outline(s, name: str) -> np.ndarray:
    """The bar's outline seen from above (n x 2, counter-clockwise): the shadow of its eight corners, turned and
    tilted as it is."""
    return _hull(corners(s, name)[:, :2])


def bar_info(s, name: str) -> dict:
    """Everything the rules ask of one bar, at this step."""
    poly = outline(s, name)
    low, tilt, axis = lowest(s, name), tilt_deg(s), axis_err_deg(s)
    xmin, xmax = float(poly[:, 0].min()), float(poly[:, 0].max())
    ymin, ymax = float(poly[:, 1].min()), float(poly[:, 1].max())
    inside = bool(xmin >= L.TANK_X0 - L.INSIDE_TOL - L.EPS and xmax <= L.TANK_X1 + L.INSIDE_TOL + L.EPS
                  and ymin >= -L.TANK_HALF - L.INSIDE_TOL - L.EPS and ymax <= L.TANK_HALF + L.INSIDE_TOL + L.EPS)
    on_floor = bool(abs(low - L.FLOOR_TOP) <= L.ON_FLOOR + L.EPS)
    stands = bool(on_floor and tilt <= L.TILT_DEG + 1e-9 and inside)
    parallel = bool(axis <= L.AXIS_DEG + 1e-9)
    beyond = float(s.pos[0]) - L.DIPPER_X  # how far its centre is beyond the dipper's centre line
    return {"poly": poly, "low": low, "tilt": tilt, "axis": axis, "xmin": xmin, "xmax": xmax, "ymin": ymin,
            "ymax": ymax, "inside": inside, "on_floor": on_floor, "stands": stands, "parallel": parallel,
            "beyond": beyond, "x": float(s.pos[0]), "y": float(s.pos[1])}


# -- the barrier -------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


def openings(infos: dict) -> dict:
    """The four openings along the line, left wall to right wall, from the bars' outlines seen from above: the
    left wall gap (left wall's inside face to the leftmost bar), slit 1, slit 2, the right wall gap. Each is a
    stretch of y (its two ends); its width is never below 0 (bars that touch, or whose outlines overlap along y,
    leave a slit of width 0 at the point where they meet)."""
    order = sorted(infos, key=lambda n: -infos[n]["y"])  # left (+y) to right, by centre
    left, middle, right = (infos[n] for n in order)
    stretches = [(left["ymax"], L.TANK_HALF), (middle["ymax"], left["ymin"]), (right["ymax"], middle["ymin"]),
                 (-L.TANK_HALF, right["ymin"])]
    out = []
    for lo, hi in stretches:
        out.append({"width": max(0.0, hi - lo), "centre": (lo + hi) / 2})
    return {"order": order, "left_gap": out[0], "slit_1": out[1], "slit_2": out[2], "right_gap": out[3]}


def evaluate(bars: dict) -> tuple[float, dict]:
    """The score of the barrier as it stands, and what it was read from. ``bars`` must all be valid."""
    infos = {n: bar_info(bars[n], n) for n in L.BAR_NAMES}
    xs = [i["x"] for i in infos.values()]
    spread = max(xs) - min(xs)
    standing = [n for n, i in infos.items() if i["stands"]]
    parallel = [n for n, i in infos.items() if i["parallel"]]
    at_distance = [n for n, i in infos.items()
                   if L.LINE_DIST_MIN - L.EPS <= i["beyond"] <= L.LINE_DIST_MAX + L.EPS]
    line = bool(len(standing) == 3 and len(parallel) == 3 and len(at_distance) == 3
                and spread <= L.INLINE_MAX + L.EPS)
    details = {
        "line": line,
        "standing": standing,
        "parallel": parallel,
        "at_distance": at_distance,
        "inline_spread_mm": _mm(spread),
    }
    for n, i in infos.items():
        details[n] = {
            "stands": i["stands"],
            "low_mm": _mm(i["low"]),
            "tilt_deg": round(i["tilt"], 2),
            "axis_err_deg": round(i["axis"], 2),
            "inside": i["inside"],
            "beyond_dipper_mm": _mm(i["beyond"]),
            "x_mm": _mm(i["x"]),
            "y_mm": _mm(i["y"]),
            "ymin_mm": _mm(i["ymin"]),
            "ymax_mm": _mm(i["ymax"]),
        }
    if not line:
        details.update(slits_ok=False, centres_ok=False, walls_ok=False)
        return 0.0, details
    op = openings(infos)
    s1, s2 = op["slit_1"], op["slit_2"]
    sep = s1["centre"] - s2["centre"]
    mid = (s1["centre"] + s2["centre"]) / 2
    gaps = (op["left_gap"]["width"], op["right_gap"]["width"])
    slits_ok = all(L.SLIT_MIN - L.EPS <= s["width"] <= L.SLIT_MAX + L.EPS for s in (s1, s2))
    centres_ok = bool(L.SEP_MIN - L.EPS <= sep <= L.SEP_MAX + L.EPS and abs(mid) <= L.MID_MAX + L.EPS)
    walls_ok = all(g <= L.WALL_GAP_MAX + L.EPS for g in gaps)
    score = 0.25 * (1 + int(slits_ok) + int(centres_ok) + int(walls_ok))
    details.update(
        order=op["order"],
        slits_mm=[_mm(s1["width"]), _mm(s2["width"])],
        slit_centres_mm=[_mm(s1["centre"]), _mm(s2["centre"])],
        slit_sep_mm=_mm(sep),
        slit_mid_mm=_mm(mid),
        wall_gaps_mm=[_mm(g) for g in gaps],
        slits_ok=slits_ok,
        centres_ok=centres_ok,
        walls_ok=walls_ok,
    )
    return score, details


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer keeps over a trial: each bar's end points over the last 0.5 s, and counters."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # steps with a bar missing or non-finite (scored 0, tracker untouched)
        self.history: dict[str, deque] = {}  # bar -> end points at this step and the 10 before it
        self.prev: dict[str, np.ndarray] | None = None  # each bar's centre at the previous valid step
        self.first_line_step: int | None = None
        self.first_full_step: int | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(bars: dict) -> bool:
        """The starting layout: every bar standing on the table on its mark (within 45 mm), its length along y."""
        for n, (x, y) in L.BAR_START.items():
            s = bars.get(n)
            if s is None or not valid(s):
                return False
            on_mark = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            if not (on_mark and abs(lowest(s, n)) < 0.003 and tilt_deg(s) < 2.0 and axis_err_deg(s) < 10.0):
                return False
        return True

    def moved(self) -> dict[str, float | None]:
        """How far each bar's ends moved at most over the last 0.5 s (None until 0.5 s has been seen)."""
        out = {}
        for n in L.BAR_NAMES:
            h = self.history.get(n)
            if h is None or len(h) < self.window + 1:
                out[n] = None
                continue
            pts = np.array(h)  # (k, 2, 3)
            out[n] = float(np.linalg.norm(pts - pts[-1], axis=2).max())
        return out

    def update(self, bars: dict) -> Result:
        if any(not valid(bars.get(n)) for n in L.BAR_NAMES):
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        centres = {n: np.asarray(bars[n].pos, dtype=float) for n in L.BAR_NAMES}
        if self.prev is not None and self.at_start(bars):
            jump = max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in L.BAR_NAMES)
            if jump > L.RESET_JUMP:  # the simulator put the bars back: a new trial
                self.reset()
        self.steps += 1
        for n in L.BAR_NAMES:
            self.history.setdefault(n, deque(maxlen=self.window + 1)).append(ends(bars[n], n))
        self.prev = centres
        self.last = self.result(bars)
        if self.first_line_step is None and self.last.details.get("line"):
            self.first_line_step = self.steps
        if self.first_full_step is None and self.last.score == 1.0:
            self.first_full_step = self.steps
        self.last.details.update(first_line_step=self.first_line_step, first_full_step=self.first_full_step)
        return self.last

    def result(self, bars: dict) -> Result:
        """The score of the state at hand, with the history fed so far (it changes nothing)."""
        if any(not valid(bars.get(n)) for n in L.BAR_NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        score, details = evaluate({n: bars[n] for n in L.BAR_NAMES})
        moved = self.moved()
        still = all(m is not None and m <= L.STILL_TOL + L.EPS for m in moved.values())
        details.update(valid_state=True, still=still, moved_mm={n: _mm(m) for n, m in moved.items()},
                       steps=self.steps, bad_steps=self.bad_steps)
        return Result(score, bool(score == 1.0 and still), details)
