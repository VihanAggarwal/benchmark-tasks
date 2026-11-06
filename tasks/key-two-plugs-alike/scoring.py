"""The scorer, as plain functions of the ten key pins' states and the arms' state. NumPy only; no simulator needed.

The score is the number of PINNED chambers over both plugs, k (0-8), over 8. A chamber is pinned when all of
these hold at that control step:

  (a) a key pin stands in it: the pin's centre, seen from above, is inside the chamber's 34 x 34 mm opening, and
      its axis is at most 8 degrees from vertical (either end up);
  (b) the centre of its top face is within 2.5 mm of the plug's top surface (the shear line), above or below;
  (c) it is still: its centre's path over the last 0.5 s is under 2 mm (positions, never velocities);
  (d) it is let go: neither gripper's TCP is within 50 mm of its top-face centre;
  (e) nothing else is over it: no part of any other pin, seen from above (its outline, at any height), lies over
      that chamber's opening (a second pin stacked on it, beside it in the chamber, or lying on the plug's top
      across it).

Only the pin of the size the code gives can be flush: sizes differ by 6 mm, and a lean of 8 degrees or less moves a
pin's top-face centre by at most 1.4 mm. A pin can stay leaning further (its foot caught against a chamber wall,
friction stopping its rim from sliding back), and then (a) rejects it. The scorer keeps a little state (each pin's
recent positions, for the path in (c)) and is fed one observation per control step by ``task.success()``:

    Tracker.update(pins, arms) -> Result(score, success, details)

``pins`` maps each pin's name (``layout.PINS``) to anything with ``pos`` (bounding-box centre, which for a
cylinder is its body centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with
``tcp``, or ``None`` when the arms could not be read; then no chamber counts as let go at that step (no credit
without evidence) and the step is counted in ``arms_unavailable``. A pin state with a non-finite number or a zero
quaternion (a simulator blow-up), or a missing pin, scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
CHAMBERS = tuple((side, i) for side in L.SIDES for i in range(4))  # (plug, chamber index 0-3)


def chamber_id(side: str, i: int) -> str:
    """'L1' .. 'L4' for the left plug's chambers, 'R1' .. 'R4' for the right plug's (chamber 1 nearest the bow)."""
    return f"{side[0].upper()}{i + 1}"


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
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def pins_valid(pins) -> bool:
    """All ten pins present with valid states."""
    try:
        return all(valid(pins[name]) for name in L.PINS)
    except (KeyError, TypeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            tcp = np.asarray(arms[side].tcp, dtype=float).ravel()
            if tcp.size != 3 or not np.all(np.isfinite(tcp)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def axis_up(s) -> np.ndarray:
    """The pin's axis as a unit vector, pointing up (a pin is the same either end up)."""
    u = rotation(s.quat_xyzw)[:, 2]
    return u if u[2] >= 0.0 else -u


def tilt_deg(s) -> float:
    """Angle between the pin's axis and the vertical (degrees, 0..90; either end up)."""
    return float(np.degrees(np.arccos(np.clip(axis_up(s)[2], 0.0, 1.0))))


def size_of(name: str) -> int:
    return L.PINS[name][1]


def top_centre(s, size: int) -> np.ndarray:
    """Centre of the pin's upper end face (world, m)."""
    return np.asarray(s.pos, dtype=float) + axis_up(s) * (L.PIN_LEN[size] / 2)


def top_offset(s, size: int) -> float:
    """Height of the pin's top-face centre above the shear line, the plug's top surface (m; negative: low)."""
    return float(top_centre(s, size)[2] - L.PLUG_TOP_Z)


def opening(side: str, i: int) -> tuple[float, float, float, float]:
    """The chamber's 34 x 34 mm opening seen from above: (x0, x1, y0, y1)."""
    cx, cy, h = L.CHAMBER_X[i], L.PLUG_Y[side], L.CHAMBER_HALF
    return cx - h, cx + h, cy - h, cy + h


def centre_in_opening(s, side: str, i: int) -> bool:
    """The pin's centre, seen from above, is inside the chamber's opening."""
    x0, x1, y0, y1 = opening(side, i)
    return bool(x0 <= float(s.pos[0]) <= x1 and y0 <= float(s.pos[1]) <= y1)


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


def shadow(s, size: int) -> np.ndarray:
    """The pin's outline seen from above, (n, 2): the convex hull of its two end circles, as it is (a standing
    pin's is its 24 mm circle; a lying pin's, a 24 mm wide band its length long)."""
    r = rotation(s.quat_xyzw)
    u, e1, e2 = r[:, 2], r[:, 0], r[:, 1]
    c = np.asarray(s.pos, dtype=float)
    h = L.PIN_LEN[size] / 2
    t = np.linspace(0.0, 2 * math.pi, L.SHADOW_SAMPLES, endpoint=False)
    ring = L.PIN_R * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))
    pts = np.vstack([c + h * u + ring, c - h * u + ring])
    return _hull(pts[:, :2])


def _overlaps_rect(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Does a convex polygon (n, 2) overlap the axis-aligned rectangle [x0, x1] x [y0, y1]? (Separating axes;
    shapes that only touch along an edge do not overlap.)"""
    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.hypot(*e) > 1e-12:
            axes.append(np.array([-e[1], e[0]]))
    for a in axes:
        p, q = poly @ a, rect @ a
        if p.max() <= q.min() + 1e-9 or q.max() <= p.min() + 1e-9:
            return False
    return True


def over_opening(s, size: int, side: str, i: int) -> bool:
    """Some part of the pin, seen from above, lies over the chamber's opening (at any height)."""
    x0, x1, y0, y1 = opening(side, i)
    c, reach = np.asarray(s.pos, dtype=float), L.PIN_LEN[size] / 2 + L.PIN_R  # no part is farther than this
    if c[0] + reach <= x0 or c[0] - reach >= x1 or c[1] + reach <= y0 or c[1] - reach >= y1:
        return False  # too far away to reach over it (the cheap test, before drawing the outline)
    return _overlaps_rect(shadow(s, size), x0, x1, y0, y1)


def tcp_distance(arms, point) -> float:
    """The nearer gripper's TCP to ``point`` (m); inf when the arms cannot be read."""
    if not arms_valid(arms):
        return math.inf
    p = np.asarray(point, dtype=float)
    return min(float(np.linalg.norm(np.asarray(arms[a].tcp, dtype=float) - p)) for a in ("left", "right"))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each pin's recent positions (for stillness), and the episode."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a missing or non-finite pin state (score 0, ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.best_k = 0  # most chambers pinned at once so far (a diagnostic; the score is read at the end)
        self.hist: dict[str, deque] = {name: deque(maxlen=self.window + 1) for name in L.PINS}
        self.prev: dict[str, np.ndarray] | None = None  # pin centres at the previous valid control step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(pins) -> bool:
        """The starting layout: every pin standing upright on the table on its kit mark."""
        for name, (mx, my) in L.PIN_START.items():
            s = pins[name]
            if (math.hypot(float(s.pos[0]) - mx, float(s.pos[1]) - my) >= L.START_TOL
                    or abs(float(s.low[2]) - L.TOP) >= 0.003 or tilt_deg(s) >= 2.0):
                return False
        return True

    def path(self, name: str) -> float:
        """How far the pin's centre has moved over the last 0.5 s (m); inf until 0.5 s has been seen."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return math.inf
        pts = np.array(h)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def drift(self, name: str) -> tuple[float, float]:
        """Trace-only diagnostics over the same 0.5 s window, never scored: the centre's net displacement (last
        position minus first) and its spread (largest distance from the window's mean position), in m; inf until
        0.5 s has been seen. A resting pin whose summed path is pushed over 2 mm by frame-to-frame noise shows
        both of these far under 2 mm (the GPU watch item for rule (c))."""
        h = self.hist[name]
        if len(h) < self.window + 1:
            return math.inf, math.inf
        pts = np.array(h)
        net = float(np.linalg.norm(pts[-1] - pts[0]))
        spread = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        return net, spread

    def update(self, pins, arms=None) -> Result:
        if not pins_valid(pins):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        now = {name: np.array(pins[name].pos, dtype=float) for name in L.PINS}
        if self.prev is not None and self.at_start(pins):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.PINS)
            if jump > L.RESET_JUMP:  # the simulator put the pins back in the kit: a new episode (rigstate's
                self.reset()  #       reset hook normally does this first; this is the fallback)
        self.prev = now
        for name, p in now.items():
            self.hist[name].append(p)
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(pins, arms)
        self.best_k = max(self.best_k, self.last.details.get("k", 0))
        return self.last

    def chamber(self, pins, arms, side: str, i: int) -> dict:
        """Whether one chamber is pinned, and every quantity that decides it."""
        inside = [n for n in L.PINS if centre_in_opening(pins[n], side, i)]
        row: dict = {"pins_in": inside, "pinned": False}
        if not inside:
            row["why"] = "empty"
            return row
        name = inside[0] if len(inside) == 1 else None
        if name is None:  # two pins with their centres in one chamber: each is 'something else over' the other
            row["why"] = "two pins"
            return row
        s, size = pins[name], size_of(name)
        tilt, off = tilt_deg(s), top_offset(s, size)
        path = self.path(name)
        net, spread = self.drift(name)
        tcp = tcp_distance(arms, top_centre(s, size))
        others = [n for n in L.PINS if n != name and over_opening(pins[n], size_of(n), side, i)]
        checks = {
            "upright": tilt <= L.TILT_MAX_DEG,
            "flush": abs(off) <= L.FLUSH_TOL,
            "still": path < L.STILL_PATH,
            "let_go": arms_valid(arms) and tcp >= L.LET_GO_R,  # unreadable arms: no credit without evidence
            "clear": not others,
        }
        row.update(pin=name, size=size, tilt_deg=round(tilt, 2), top_mm=_mm(off), path_mm=_mm(path),
                   net_mm=_mm(net), spread_mm=_mm(spread), tcp_mm=_mm(tcp), others_over=others, **checks)
        row["pinned"] = all(checks.values())
        if not row["pinned"]:
            row["why"] = ",".join(k for k, v in checks.items() if not v)
        return row

    def result(self, pins, arms=None) -> Result:
        if not pins_valid(pins):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        rows = {chamber_id(side, i): self.chamber(pins, arms, side, i) for side, i in CHAMBERS}
        k = sum(1 for r in rows.values() if r["pinned"])
        score = k / 8.0
        details = {
            "valid_state": True,
            "k": k,
            "pinned": [c for c, r in rows.items() if r["pinned"]],
            "chambers": rows,
            "arms_available": arms_valid(arms),
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "best_k": max(self.best_k, k),
        }
        return Result(score, bool(k == 8), details)
