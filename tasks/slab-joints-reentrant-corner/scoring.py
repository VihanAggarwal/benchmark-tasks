"""The scorer, as plain functions of the four joint strips' states and the grippers' state. NumPy only; no simulator
needed.

The rule is about the panels the joints mark out: every panel a rectangle (no inside corner), its longer side at most
132 mm and at most 1.5 times its shorter side. So the score is read from the end state: which strips count as joints,
the lines they draw across the slab, and the panels those lines and the forms' inside faces enclose. The scorer is fed
one observation per control step by ``task.success()`` and keeps only what it needs for that (each strip's end
points over the last 0.5 s, to tell that all are still, and counters for the trace).

    Tracker.update(strips, arms) -> Result(score, success, details)

``strips`` maps each strip's name to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low``
and ``high``, as ``yam_arena.task_file.ObjectState`` has; each is a box of known size (``layout.STRIPS``), so its
corners follow from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with
``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms
could not be read; then the let-go condition is taken as met for that step and the step is counted in
``arms_unavailable``. A step on which any strip has a non-finite number or a zero quaternion (a simulator blow-up),
or is missing, scores 0 and leaves the tracker as it was.

How the joints divide the slab (all exact, no raster):
1. A strip is a candidate when it lies flat on an 18 mm face on the sub-base, square to x or y within 5 degrees,
   inside the outline, let go. It is measured squared to its nearer axis about its centre. For the line it draws, a
   centre line within 6 mm of the inside corner's lines (y 0.120 along x, x 0.260 along y) is taken to lie on them;
   where it lies (its ends and outline, step 2) is measured unsnapped.
2. A strip counts when each end is within 8 mm of a form's inside face (slid straight along the strip) or of another
   counted strip's outline lying beyond it. Candidates that fail are dropped and the rest checked again, until none
   changes.
3. Each counted strip draws its centre line from the face it meets (or the centre line of the strip it meets, or the
   end of a strip it meets end to end) at one end to the same at the other. Those lines and the outline divide the
   slab into panels, found on the grid of every line's coordinates (a union-find over its cells).
4. A panel is good when it is a rectangle (its area fills its bounding box), its longer side is at most 132 mm and at
   most 1.5 times its shorter side. Score = 0.2 per good panel, at most 1.0.

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
ARMS = ("left", "right")


def half_size(name: str) -> np.ndarray:
    return np.array((L.STRIPS[name][0] / 2, L.STRIP_W / 2, L.STRIP_T / 2), dtype=float)


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


# -- one strip --------------------------------------------------------------------------------------------------------
def corners(s, name: str) -> np.ndarray:
    """The strip's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half_size(name)) @ rotation(s.quat_xyzw).T


def lowest(s, name: str) -> float:
    """Height of the strip's lowest point above the table top (m)."""
    return float(corners(s, name)[:, 2].min() - L.TOP)


def tilt_deg(s) -> float:
    """Angle between the vertical and the strip's 12 mm thickness (degrees, 0..90): either 18 mm face may be down.
    Lying flat 0; on a 12 mm edge or standing on end 90."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def axis_of(s) -> tuple[str | None, float]:
    """The axis ('x' or 'y') the strip's length is nearer to, seen from above, and how far it is turned from it
    (degrees, 0..45; either end may point either way). A strip standing on end has no line from above: (None, 90)."""
    u = rotation(s.quat_xyzw)[:, 0]
    n = math.hypot(float(u[0]), float(u[1]))
    if n < 1e-9:
        return None, 90.0
    a = math.degrees(math.atan2(float(u[1]), float(u[0]))) % 180.0  # 0..180, either way round
    to_x = min(a, 180.0 - a)
    to_y = abs(90.0 - a)
    return ("x", float(to_x)) if to_x <= to_y else ("y", float(to_y))


def in_outline(x: float, y: float, tol: float = 0.0) -> bool:
    """Is the point inside the slab outline (either leg), grown by ``tol``?"""
    return any(x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol for x0, x1, y0, y1 in L.LEGS)


def inside(s, name: str) -> bool:
    """The strip's outline seen from above (its corners, turned and tilted as it is) is inside the slab, within 2 mm."""
    return all(in_outline(float(p[0]), float(p[1]), L.INSIDE_TOL + L.EPS) for p in corners(s, name)[:, :2])


def snapped(axis: str, c: float) -> float:
    """A centre line within 6 mm of the inside corner's line for its axis is taken to lie on it."""
    line = L.CORNER[1] if axis == "x" else L.CORNER[0]
    return line if abs(c - line) <= L.SNAP + L.EPS else c


@dataclass
class Bar:
    """A candidate strip as the scorer measures it: squared to its axis about its centre, its centre line snapped."""

    name: str
    axis: str  # 'x' or 'y': the direction of its length
    a0: float  # its centre's coordinate along its length
    c: float  # its centre line's coordinate across (snapped)
    c_raw: float  # ... before snapping
    h: float  # half its length
    w: float = L.STRIP_W / 2

    def along(self) -> tuple[float, float]:
        return self.a0 - self.h, self.a0 + self.h

    def across_raw(self) -> tuple[float, float]:
        return self.c_raw - self.w, self.c_raw + self.w

    def ranges(self, axis: str) -> tuple[tuple[float, float], tuple[float, float]]:
        """(its real range along ``axis``, its real range across ``axis``): where it lies, unsnapped."""
        return (self.along(), self.across_raw()) if self.axis == axis else (self.across_raw(), self.along())


def bar_of(s, name: str) -> Bar:
    axis, _ = axis_of(s)
    axis = axis or "x"
    x, y = float(s.pos[0]), float(s.pos[1])
    a0, c = (x, y) if axis == "x" else (y, x)
    return Bar(name, axis, a0, snapped(axis, c), c, L.STRIPS[name][0] / 2)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def end_meets(b: Bar, d: int, others) -> tuple[float, str | None, float | None]:
    """How far the strip's end (d = +1: its + end, -1: its - end) is from what it butts against, what that is ('form'
    or the other strip's name), and where its centre line is then carried to. A form's inside face counts when the
    end, slid straight along the strip, meets it (the distance slid); the face is where the line goes. Another strip
    counts when it lies beyond the end (or at most 2 mm into it) and the distance from the end to its outline is
    measured; the line goes to that strip's centre line if it lies across this one, or to its end if they lie end to
    end. Where the strips lie is measured unsnapped. (inf, None, None) if nothing is beyond the end."""
    e = b.a0 + d * b.h
    lat0, lat1 = b.across_raw()
    best = (math.inf, None, None)
    for k, l0, l1 in (L.FACES_X if b.axis == "x" else L.FACES_Y):
        if _overlap(lat0, lat1, l0, l1) > L.EPS:
            g = d * (k - e)
            if g >= -L.END_PEN - L.EPS and g < best[0]:
                best = (g, "form", k)
    for o in others:
        if o.name == b.name:
            continue
        (o0, o1), (p0, p1) = o.ranges(b.axis)
        near = o0 if d > 0 else o1
        g = d * (near - e)
        if g < -L.END_PEN - L.EPS:
            continue
        dist = math.hypot(max(g, 0.0), max(0.0, -_overlap(lat0, lat1, p0, p1)))
        if dist < best[0]:
            best = (dist, o.name, o.c if o.axis != b.axis else near)
    return best


def counted(bars: list[Bar]) -> tuple[list[Bar], dict]:
    """Drop candidates until every one left has both ends within 8 mm of a form or of another one left. Returns the
    strips that count and, for every candidate, its two ends as last measured: {name: [(gap, meets, to), ...]}."""
    keep = list(bars)
    ends = {}
    while True:
        ends = {b.name: [end_meets(b, d, keep) for d in (-1, 1)] for b in keep}
        drop = [b for b in keep if any(g > L.END_TOL + L.EPS for g, _, _ in ends[b.name])]
        if not drop:
            break
        dropped = {b.name for b in drop}
        keep = [b for b in keep if b.name not in dropped]
    for b in bars:  # the dropped ones, measured against those that count, for the trace
        if b.name not in ends:
            ends[b.name] = [end_meets(b, d, keep) for d in (-1, 1)]
    return keep, ends


def joint_lines(keep: list[Bar], ends: dict) -> list[tuple[str, float, float, float]]:
    """Each counted strip's centre line carried across its end gaps: (axis, coordinate across, from, to)."""
    lines = []
    for b in keep:
        (_, _, t0), (_, _, t1) = ends[b.name]
        lo, hi = sorted((float(t0), float(t1)))
        lines.append((b.axis, b.c, lo, hi))
    return lines


# -- the panels ----------------------------------------------------------------------------------------------------
def _r(v: float) -> float:
    return round(float(v), 7)


def panels(lines) -> list[dict]:
    """The panels that the joint lines and the slab outline enclose: the cells of the grid through every line's
    coordinates and ends, joined where no line separates them. Each: bounding box, area, rectangle or not."""
    bx0, bx1, by0, by1 = L.BBOX
    xs = {_r(v) for leg in L.LEGS for v in leg[:2]}
    ys = {_r(v) for leg in L.LEGS for v in leg[2:]}
    for axis, c, lo, hi in lines:
        if axis == "x":  # a line along x at y = c
            ys.add(_r(c))
            xs.update((_r(lo), _r(hi)))
        else:
            xs.add(_r(c))
            ys.update((_r(lo), _r(hi)))
    xs = sorted(v for v in xs if bx0 - L.EPS <= v <= bx1 + L.EPS)
    ys = sorted(v for v in ys if by0 - L.EPS <= v <= by1 + L.EPS)
    nx, ny = len(xs) - 1, len(ys) - 1
    cell_in = {(i, j): in_outline((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)
               for i in range(nx) for j in range(ny)}
    along_x = [(_r(c), lo, hi) for axis, c, lo, hi in lines if axis == "x"]  # walls between rows
    along_y = [(_r(c), lo, hi) for axis, c, lo, hi in lines if axis == "y"]  # walls between columns
    tol = 1e-7

    def blocked(walls, at, a0, a1) -> bool:
        return any(abs(c - at) <= tol and lo <= a0 + tol and hi >= a1 - tol for c, lo, hi in walls)

    parent = {k: k for k, v in cell_in.items() if v}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for (i, j) in list(parent):
        if (i + 1, j) in parent and not blocked(along_y, xs[i + 1], ys[j], ys[j + 1]):
            parent[find((i + 1, j))] = find((i, j))
        if (i, j + 1) in parent and not blocked(along_x, ys[j + 1], xs[i], xs[i + 1]):
            parent[find((i, j + 1))] = find((i, j))
    groups: dict = {}
    for k in parent:
        groups.setdefault(find(k), []).append(k)
    out = []
    for cells in groups.values():
        x0 = min(xs[i] for i, _ in cells)
        x1 = max(xs[i + 1] for i, _ in cells)
        y0 = min(ys[j] for _, j in cells)
        y1 = max(ys[j + 1] for _, j in cells)
        area = sum((xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]) for i, j in cells)
        rect = abs((x1 - x0) * (y1 - y0) - area) <= 1e-9
        long_side, short_side = max(x1 - x0, y1 - y0), min(x1 - x0, y1 - y0)
        good = bool(rect and long_side <= L.MAX_SIDE + L.EPS and long_side <= L.MAX_ASPECT * short_side + L.EPS)
        out.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "area": area, "rect": rect, "long": long_side,
                    "short": short_side, "good": good})
    out.sort(key=lambda p: (p["y1"], p["x0"]), reverse=True)
    return out


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


def zone_gap(arm, s, name: str) -> float:
    """How far a gripper's finger zone is from the strip (negative: they overlap)."""
    pts, r = finger_zone(arm)
    d = (np.asarray(pts, dtype=float) - np.asarray(s.pos, dtype=float)) @ rotation(s.quat_xyzw)
    return float(np.linalg.norm(np.maximum(np.abs(d) - half_size(name), 0.0), axis=1).min()) - r


# -- the rules -----------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


def strip_info(s, name: str, arms=None) -> dict:
    """Everything the candidate test asks of one strip, at this step. ``arms`` unreadable: let go is taken as met."""
    tilt, low = tilt_deg(s), lowest(s, name)
    axis, turn = axis_of(s)
    have_arms = arms_valid(arms)
    zones = {side: zone_gap(arms[side], s, name) for side in ARMS} if have_arms else None
    let_go = all(z >= L.HANDS_OFF - L.EPS for z in zones.values()) if have_arms else True
    flat = tilt <= L.FLAT_DEG + 1e-9
    on_base = low <= L.SUBBASE_H + L.ON_BASE + L.EPS
    square = axis is not None and turn <= L.SQUARE_DEG + 1e-9
    within = inside(s, name)
    candidate = bool(flat and on_base and square and within and let_go)
    why = ("candidate" if candidate else "not flat" if not flat else "not on the sub-base" if not on_base
           else "not square to x or y" if not square else "outside the slab" if not within else "not let go")
    return {"candidate": candidate, "why": why, "tilt": tilt, "low": low, "axis": axis, "turn": turn,
            "inside": within, "let_go": bool(let_go), "zones": zones}


def evaluate(strips: dict, arms=None) -> tuple[float, dict]:
    """The score of the slab as it stands, and what it was read from. ``strips`` must all be valid."""
    infos = {n: strip_info(s, n, arms) for n, s in strips.items()}
    bars = [bar_of(strips[n], n) for n in strips if infos[n]["candidate"]]
    keep, ends = counted(bars)
    lines = joint_lines(keep, ends)
    cells = panels(lines)
    good = sum(1 for p in cells if p["good"])
    score = round(min(1.0, L.PER_PANEL * good), 6)
    kept = {b.name for b in keep}
    by_name = {b.name: b for b in bars}
    details = {
        "good_panels": good,
        "panels": [{"x_mm": [_mm(p["x0"]), _mm(p["x1"])], "y_mm": [_mm(p["y0"]), _mm(p["y1"])],
                    "rect": p["rect"], "good": p["good"]} for p in cells],
        "lines_mm": [[a, _mm(c), _mm(lo), _mm(hi)] for a, c, lo, hi in lines],
        "arms_available": arms_valid(arms),
    }
    for n, i in infos.items():
        b = by_name.get(n)
        row = {
            "counts": n in kept,
            "why": ("counts" if n in kept else "an end not within 8 mm of a form or a counted strip"
                    if i["candidate"] else i["why"]),
            "x_mm": _mm(float(strips[n].pos[0])),
            "y_mm": _mm(float(strips[n].pos[1])),
            "axis": i["axis"],
            "turn_deg": round(i["turn"], 2),
            "tilt_deg": round(i["tilt"], 2),
            "low_mm": _mm(i["low"]),
            "inside": i["inside"],
            "let_go": i["let_go"],
            "left_zone_mm": _mm(i["zones"]["left"]) if i["zones"] else None,
            "right_zone_mm": _mm(i["zones"]["right"]) if i["zones"] else None,
        }
        if b is not None:
            row.update(line_mm=_mm(b.c), line_raw_mm=_mm(b.c_raw),
                       end_gaps_mm=[_mm(g) if math.isfinite(g) else None for g, _, _ in ends[n]],
                       ends_meet=[m for _, m, _ in ends[n]])
        details[n] = row
    return score, details


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer keeps over a trial: each strip's end points over the last 0.5 s, and counters."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # steps with a strip missing or non-finite (scored 0, tracker untouched)
        self.arms_unavailable = 0  # steps on which rigstate could not read the arms
        self.history: dict[str, deque] = {}  # strip -> end points at this step and the 10 before it
        self.prev: dict[str, np.ndarray] | None = None  # each strip's centre at the previous valid step
        self.first_full_step: int | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(strips: dict) -> bool:
        """The starting layout: all four strips lying flat on the table on their marks (within 45 mm)."""
        for n, (_, _, x, y) in L.STRIPS.items():
            s = strips.get(n)
            if s is None or not valid(s):
                return False
            on_mark = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            if not (on_mark and abs(lowest(s, n)) < 0.003 and tilt_deg(s) < 2.0):
                return False
        return True

    @staticmethod
    def ends(s, name: str) -> np.ndarray:
        """The centres of the strip's two end faces (2 x 3)."""
        c, u = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)[:, 0] * (L.STRIPS[name][0] / 2)
        return np.array([c + u, c - u])

    def moved(self) -> dict[str, float | None]:
        """How far each strip's ends moved at most over the last 0.5 s (None until 0.5 s has been seen)."""
        out = {}
        for n in L.STRIP_NAMES:
            h = self.history.get(n)
            if h is None or len(h) < self.window + 1:
                out[n] = None
                continue
            pts = np.array(h)  # (k, 2, 3)
            out[n] = float(np.linalg.norm(pts - pts[-1], axis=2).max())
        return out

    def update(self, strips: dict, arms=None) -> Result:
        if any(not valid(strips.get(n)) for n in L.STRIP_NAMES):
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        centres = {n: np.asarray(strips[n].pos, dtype=float) for n in L.STRIP_NAMES}
        if self.prev is not None and self.at_start(strips):
            jump = max(float(np.linalg.norm(centres[n] - self.prev[n])) for n in L.STRIP_NAMES)
            if jump > L.RESET_JUMP:  # the simulator put the strips back: a new trial
                self.reset()
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        for n in L.STRIP_NAMES:
            self.history.setdefault(n, deque(maxlen=self.window + 1)).append(self.ends(strips[n], n))
        self.prev = centres
        self.last = self.result(strips, arms)
        if self.first_full_step is None and self.last.score == 1.0:
            self.first_full_step = self.steps
        self.last.details.update(first_full_step=self.first_full_step)
        return self.last

    def result(self, strips: dict, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (it changes nothing)."""
        if any(not valid(strips.get(n)) for n in L.STRIP_NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        score, details = evaluate({n: strips[n] for n in L.STRIP_NAMES}, arms)
        moved = self.moved()
        still = all(m is not None and m <= L.STILL_TOL + L.EPS for m in moved.values())
        details.update(valid_state=True, still=still, moved_mm={n: _mm(m) for n, m in moved.items()},
                       steps=self.steps, bad_steps=self.bad_steps, arms_unavailable=self.arms_unavailable)
        return Result(score, bool(score == 1.0 and still), details)
