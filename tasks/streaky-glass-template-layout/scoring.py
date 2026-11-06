"""The scorer, as plain functions of the three templates' states and the arms'. NumPy only; no simulator needed.

The score is a function of the end state: which templates lie laid on the glass sheet, and whether the three of them
leave a clear strip at least 80 mm wide along one long edge. The only memory the scorer keeps is each template's
recent positions (to judge 'at rest' by the path its centre travelled over the last 0.5 s, never by a velocity: a
gripped part's reported velocity chatters at 10-45 mm/s while it stays put), how many steps the arms could not be
read, and a reset when a new episode starts. It is fed one observation per control step by ``task.success()``.

    Tracker.update(templates, arms) -> Result(score, success, details)

``templates`` maps each template's name (``layout.NAMES``) to anything with ``pos`` (centre), ``quat_xyzw``,
``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; only ``pos`` and
``quat_xyzw`` are used for the rules (every corner, the outline seen from above and the lowest point come from them
and the template's size in ``layout``). ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with
``tcp``, ``tips`` (the finger-tip point midway between the two pad tips, 13 mm past the TCP) and ``opening`` (0
closed .. 1 open), or ``None`` when the arms could not be read; then no template counts as let go on that step (fail
closed) and the step is counted in ``arms_unavailable``. A template state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

A template is LAID when all of these hold (task.json 'scoring'):
  (a) flat: tilted less than 3 degrees (either face down; the faces are alike);
  (b) directly on the sheet: its lowest point within 2 mm of the sheet's top and its centre over the sheet (a
      template stacked on another is 12 mm too high);
  (c) at rest: its centre's path over the last 0.5 s (10 control steps) under 2 mm;
  (d) let go: both grippers' finger zones at least 30 mm from its box (3-D). A finger zone is a round cylinder about
      the gripper's axis, from the finger-tip point 38 mm back up, of radius hypot(half the pad gap + 10 mm, 20 mm)
      (60 mm fully open): it holds both fingers at any heading, so an open finger resting on a template is never
      'let go', and straight above a template it is judged by the finger tips' real height;
  (e) every corner of its outline seen from above at least 8 mm inside every sheet edge;
  (f) at least 5 mm, seen from above, from every other template lying low (its lowest point within 20 mm of the
      sheet's top: on the sheet, on another template or on the table); one carried higher is not counted;
  (g) its long side within 5 degrees of the streaks (the x axis), either way round.
Clear strip (all three laid): the larger of (lowest x of any template corner - 0.200) and (0.380 - highest x).
Score: 0.2 per laid template (0, 0.2, 0.4, 0.6); 1.0 when all three are laid and the clear strip is at least 80 mm.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

ARMS = ("left", "right")
SHORT = {"template_a1": "a1", "template_a2": "a2", "template_b": "b"}
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


# -- geometry ---------------------------------------------------------------------------------------------------
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
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP, finger-tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            for key in ("tcp", "tips"):
                v = np.asarray(getattr(arms[side], key), dtype=float).ravel()
                if v.size != 3 or not np.all(np.isfinite(v)):
                    return False
            if not math.isfinite(float(arms[side].opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def size_of(name: str) -> np.ndarray:
    return np.asarray(L.TEMPLATES[name][0], dtype=float)


def corners(s, size) -> np.ndarray:
    """The eight corners of a template's box (world, 8 x 3) from its centre, orientation and size."""
    half = np.asarray(size, dtype=float) / 2
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and the axis through the template's two large faces (degrees, 0..90): the faces are
    alike, so whichever is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def grain_deg(quat_xyzw) -> float:
    """Angle seen from above between the template's long side (body x) and the streaks (world x), either way round
    (degrees, 0..90). A long side pointing nearly straight up has no direction seen from above: 90."""
    v = rotation(quat_xyzw)[:2, 0]
    if float(np.hypot(*v)) < 1e-6:
        return 90.0
    a = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
    return float(min(a, 180.0 - a))


def bottom_gap(s, size) -> float:
    """Height of the template's lowest corner above the sheet's top (m; negative = below it)."""
    return float(corners(s, size)[:, 2].min() - L.SHEET_TOP)


def over_sheet(p) -> bool:
    """A point's x, y over the sheet's 180 x 300 mm outline."""
    return bool(L.SHEET_X0 <= p[0] <= L.SHEET_X1 and L.SHEET_Y0 <= p[1] <= L.SHEET_Y1)


def edge_inset(s, size) -> float:
    """How far the template's outline seen from above lies inside the sheet's nearest edge, at its worst corner
    (m; negative = past the edge)."""
    c = corners(s, size)[:, :2]
    return float(min((c[:, 0] - L.SHEET_X0).min(), (L.SHEET_X1 - c[:, 0]).min(),
                     (c[:, 1] - L.SHEET_Y0).min(), (L.SHEET_Y1 - c[:, 1]).min()))


def x_span(s, size) -> tuple[float, float]:
    """Lowest and highest x of the template's corners."""
    x = corners(s, size)[:, 0]
    return float(x.min()), float(x.max())


def _hull(points: np.ndarray) -> np.ndarray:
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


def outline(s, size) -> np.ndarray:
    """The template's outline seen from above, (n, 2): the shadow of its eight corners (for a flat template, its
    turned rectangle)."""
    return _hull(corners(s, size)[:, :2])


def _overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Do two convex polygons overlap (or touch)? Separating axes."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            if float(np.hypot(*e)) < 1e-12:
                continue
            a = np.array([-e[1], e[0]])
            pa, qa = p @ a, q @ a
            if pa.max() < qa.min() or qa.max() < pa.min():
                return False
    return True


def _point_segment(pt, a, b) -> float:
    ab = b - a
    n = float(ab @ ab)
    t = 0.0 if n < 1e-18 else float(np.clip((pt - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(pt - (a + t * ab)))


def polygon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Smallest distance between two convex polygons (n, 2) (0 when they overlap or touch)."""
    if _overlap(p, q):
        return 0.0
    best = math.inf
    for u, v in ((p, q), (q, p)):
        for pt in u:
            for i in range(len(v)):
                best = min(best, _point_segment(pt, v[i], v[(i + 1) % len(v)]))
    return float(best)


def points_box_distance(pts, s, size) -> np.ndarray:
    """3-D distance from each of n points (n, 3) to a template's (turned) box (0 inside it)."""
    d = (np.atleast_2d(np.asarray(pts, dtype=float)) - np.asarray(s.pos, dtype=float)) @ rotation(s.quat_xyzw)
    q = np.maximum(np.abs(d) - np.asarray(size, dtype=float) / 2, 0.0)
    return np.linalg.norm(q, axis=1)


def point_box_distance(pt, s, size) -> float:
    """3-D distance from a point to a template's (turned) box (0 inside it)."""
    return float(points_box_distance(pt, s, size)[0])


# -- the grippers -----------------------------------------------------------------------------------------------
def zone_radius(arm) -> float:
    """From the gripper's axis to a finger's outer corner: half the pad gap plus the finger's thickness, and half its
    width across (60.4 mm fully open, 22.4 mm closed)."""
    half_gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP / 2
    return float(math.hypot(half_gap + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


# Each disc of the zone: its rim at 72 points (5 degrees apart: within 0.1 mm of the true disc's distances), rings at
# three quarters, half and a quarter of the radius, and the centre (at most 15 mm apart), so a template partly under
# the disc (40 mm or more across) always has sample points over it.
_RING = np.array([(f * math.cos(a), f * math.sin(a)) for f in (1.0, 0.75, 0.5, 0.25)
                  for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)] + [(0.0, 0.0)])


def finger_zone(arm, samples: int = 20) -> np.ndarray:
    """Points filling the finger zone: discs about the gripper's axis (TCP to finger-tip point), from the finger-tip
    point back up 38 mm (2 mm apart)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    e1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    centres = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    disc = zone_radius(arm) * (np.outer(_RING[:, 0], e1) + np.outer(_RING[:, 1], e2))
    return (centres[:, None, :] + disc[None, :, :]).reshape(-1, 3)


def zones(arms) -> dict[str, np.ndarray] | None:
    """Both grippers' finger zones, or None if the arms are unreadable."""
    if not arms_valid(arms):
        return None
    return {a: finger_zone(arms[a]) for a in ARMS}


def zone_distance(arms, s, size, zone_points=None) -> float | None:
    """The nearer of the two grippers' finger zones to the template's box (m), or None if the arms are unreadable.
    ``zone_points`` may pass zones(arms) computed once for the step."""
    z = zone_points if zone_points is not None else zones(arms)
    if z is None:
        return None
    return float(min(np.min(points_box_distance(z[a], s, size)) for a in ARMS))


def lies_low(s, size) -> bool:
    """Counted by the gap rule (f): the template's lowest point within 20 mm of the sheet's top, above or below (on
    the sheet, on another template, or on the table beside it); one carried higher, or fallen to the floor, is not."""
    return abs(bottom_gap(s, size)) <= L.GAP_BAND + L.EPS


# -- the scorer -------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None else round(float(v) * 1000.0, 2)


class Tracker:
    """What the scorer remembers within a trial: each template's centre over the last 0.5 s."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: dict[str, deque] = {n: deque(maxlen=self.window + 1) for n in L.NAMES}
        self.prev: dict[str, np.ndarray] | None = None  # centres at the previous valid control step
        self.arms_unavailable = 0  # control steps on which the arms could not be read
        self.invalid_steps = 0  # control steps with an unreadable template state
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(templates) -> bool:
        """The starting layout: every template on its start mark (within 45 mm), flat on the table."""
        for name, (size, _, x, y, _) in L.TEMPLATES.items():
            s = templates[name]
            near = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            on_table = abs(float(corners(s, size)[:, 2].min()) - L.TOP) < 0.003
            if not (near and on_table and tilt_deg(s.quat_xyzw) < 2.0):
                return False
        return True

    def path(self, name: str) -> float | None:
        """The template centre's path over the last 0.5 s (m), or None before 0.5 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        pts = np.array(h)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def update(self, templates, arms=None) -> Result:
        if not all(n in templates and valid(templates[n]) for n in L.NAMES):  # a blow-up: 0, keep everything
            self.steps += 1
            self.invalid_steps += 1
            self.last = self.result(templates, arms)
            return self.last
        now = {n: np.asarray(templates[n].pos, dtype=float).copy() for n in L.NAMES}
        if self.prev is not None and self.at_start(templates):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.NAMES)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        for n in L.NAMES:
            self.history[n].append(now[n])
        self.prev = now
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(templates, arms)
        return self.last

    def assess(self, templates, arms) -> dict[str, dict]:
        """Every term of 'laid' for each template, from the current states and the stored paths."""
        out: dict[str, dict] = {}
        shapes = {n: outline(templates[n], size_of(n)) for n in L.NAMES}
        low = {n: lies_low(templates[n], size_of(n)) for n in L.NAMES}
        zone_points = zones(arms)
        for n in L.NAMES:
            s, size = templates[n], size_of(n)
            others = [polygon_distance(shapes[n], shapes[o]) for o in L.NAMES if o != n and low[o]]
            gap = min(others) if others else None  # None: no other template lies low
            zone = zone_distance(arms, s, size, zone_points)
            path = self.path(n)
            t = {
                "tilt_deg": tilt_deg(s.quat_xyzw),
                "bottom": bottom_gap(s, size),
                "over_sheet": over_sheet(s.pos),
                "path": path,
                "zone": zone,
                "edge": edge_inset(s, size),
                "gap": gap,
                "low": low[n],
                "grain_deg": grain_deg(s.quat_xyzw),
            }
            t["flat"] = t["tilt_deg"] < L.TILT_MAX_DEG
            t["on_sheet"] = abs(t["bottom"]) <= L.BOTTOM_TOL + L.EPS and t["over_sheet"]
            t["still"] = path is not None and path < L.STILL_PATH
            t["let_go"] = zone is not None and zone >= L.LETGO_DIST - L.EPS
            t["inside"] = t["edge"] >= L.EDGE_MIN - L.EPS
            t["spaced"] = gap is None or gap >= L.GAP_MIN - L.EPS
            t["with_grain"] = t["grain_deg"] <= L.GRAIN_MAX_DEG + L.EPS
            t["laid"] = all(t[k] for k in ("flat", "on_sheet", "still", "let_go", "inside", "spaced", "with_grain"))
            out[n] = t
        return out

    @staticmethod
    def clear_strip(templates) -> tuple[float, float, float]:
        """(clear strip, near-edge margin, far-edge margin): the margins from the near long edge (x 0.200) to the
        lowest template corner and from the highest template corner to the far long edge (x 0.380)."""
        spans = [x_span(templates[n], size_of(n)) for n in L.NAMES]
        near = min(a for a, _ in spans) - L.SHEET_X0
        far = L.SHEET_X1 - max(b for _, b in spans)
        return max(near, far), near, far

    def result(self, templates, arms=None) -> Result:
        if not all(n in templates and valid(templates[n]) for n in L.NAMES):
            return Result(0.0, False, {"valid_state": False, "steps": self.steps,
                                       "invalid_steps": self.invalid_steps})
        terms = self.assess(templates, arms)
        laid = [n for n in L.NAMES if terms[n]["laid"]]
        strip = near = far = None
        if len(laid) == len(L.NAMES):
            strip, near, far = self.clear_strip(templates)
        if strip is not None and strip >= L.STRIP_MIN - L.EPS:
            score = 1.0
        else:
            score = L.SCORE_PER_LAID[len(laid)]
        details = {
            "valid_state": True,
            "n_laid": len(laid),
            "strip_mm": _mm(strip),
            "near_margin_mm": _mm(near),
            "far_margin_mm": _mm(far),
            "arms_ok": arms_valid(arms),
            "arms_unavailable": self.arms_unavailable,
            "invalid_steps": self.invalid_steps,
        }
        for n in L.NAMES:
            t, k = terms[n], SHORT[n]
            details.update({
                f"{k}_laid": t["laid"],
                f"{k}_tilt_deg": round(t["tilt_deg"], 2),
                f"{k}_bottom_mm": _mm(t["bottom"]),
                f"{k}_over_sheet": t["over_sheet"],
                f"{k}_path_mm": _mm(t["path"]),
                f"{k}_zone_mm": _mm(t["zone"]),
                f"{k}_edge_mm": _mm(t["edge"]),
                f"{k}_gap_mm": _mm(t["gap"]),
                f"{k}_low": t["low"],
                f"{k}_grain_deg": round(t["grain_deg"], 2),
            })
        return Result(score, score == 1.0, details)
