"""The scorer, as plain functions of the six pins' states and the arms' state. NumPy only; no simulator needed.

The score is the end state an SMT line operator checks before the next board comes in on the conveyor: one support
pin standing upright in each quarter of the board's outline, at least 5 mm inside the outline and 3 mm clear of
every red area (parts on the board's underside), let go; and no pin touching a red area or at a conveyor rail,
which would crush a part or hit the edge clamp when the board is clamped down. The scorer keeps a little state
(the last 0.5 s of each pin's path, for 'still', and counts for the trace) and is fed one observation per control
step by ``task.success()``.

    Tracker.update(states, arms) -> Result(score, success, details)

``states`` maps each pin's name (``layout.PIN_NAMES``) to anything with ``pos`` (centre), ``quat_xyzw``,
``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has; every pin is a cylinder
20 mm across and 45 mm long, so its shape follows from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening``
(0 closed .. 1 open), or ``None`` when the arms could not be read; then every pin is taken as let go for that step
and the step is counted in ``arms_unavailable``. A state with a non-finite number or a zero quaternion (a simulator
blow-up), or a missing pin, scores 0 for that step and leaves the tracker as it was.

The pins are told apart by nothing but where they stand: any pin can support any quarter, and spare pins may stay
in the trays or anywhere else that breaks no rule. A pin's 'outline' is its shadow seen from above: the convex hull
of its two end circles (for an upright pin, a 20 mm circle). A plain cylinder is the same either way up, so 'tilt'
is measured from whichever end is nearer to facing down.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the
8 mm pad thickness, and half the pad's 27 mm width across; 57 mm fully open). A pin a finger zone reaches is still
held and does not count: the zone covers both pads at any heading, so a zone clear of a pin cannot be touching it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_WINDOW = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 control steps
_ANG = np.linspace(0.0, 2 * math.pi, L.HULL_SIDES, endpoint=False)
_CIRCLE = np.stack([np.cos(_ANG), np.sin(_ANG)], axis=1) / math.cos(math.pi / L.HULL_SIDES)  # circumscribed
RAIL_RECTS = {name: L.rail_rect(x) for name, x in L.RAILS}
BOARD_RECT = (L.BOARD_X0, L.BOARD_X1, L.BOARD_Y0, L.BOARD_Y1)


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


def all_valid(states) -> bool:
    """Every pin present and valid."""
    try:
        return all(valid(states[n]) for n in L.PIN_NAMES)
    except (KeyError, TypeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- a pin --------------------------------------------------------------------------------------------------------
def axis(s) -> np.ndarray:
    """The pin's long axis (unit, world)."""
    return rotation(s.quat_xyzw)[:, 2]


def tilt_deg(s) -> float:
    """Angle between the pin's axis and the vertical (degrees, 0..90; either end down)."""
    return float(np.degrees(np.arccos(np.clip(abs(axis(s)[2]), 0.0, 1.0))))


def lowest(s) -> float:
    """Height of the pin's lowest point (world z)."""
    a = axis(s)
    return float(s.pos[2] - (L.PIN_H / 2) * abs(a[2]) - L.PIN_R * math.sqrt(max(0.0, 1.0 - a[2] ** 2)))


def outline(s) -> np.ndarray:
    """The pin's shadow seen from above, (n, 2): the convex hull of its two end circles (48 points each, the
    polygon drawn round the circle, so it never cuts inside it)."""
    r = rotation(s.quat_xyzw)
    c = np.asarray(s.pos, dtype=float)
    pts = []
    for sign in (-1.0, 1.0):
        centre = c + r[:, 2] * sign * L.PIN_H / 2
        ring = centre + L.PIN_R * (_CIRCLE[:, :1] * r[:, 0] + _CIRCLE[:, 1:] * r[:, 1])
        pts.append(ring[:, :2])
    return _hull(np.vstack(pts))


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def side(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = side(pts), side(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1])


def _overlaps(poly: np.ndarray, rect) -> bool:
    """Does a convex polygon overlap (or touch) an axis-aligned rectangle (x0, x1, y0, y1)? (Separating axes.)"""
    x0, x1, y0, y1 = rect
    if poly[:, 0].max() < x0 or poly[:, 0].min() > x1 or poly[:, 1].max() < y0 or poly[:, 1].min() > y1:
        return False
    box = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    normals = normals[np.hypot(normals[:, 0], normals[:, 1]) > 1e-12]
    p, q = poly @ normals.T, box @ normals.T
    return bool(np.all(p.max(axis=0) >= q.min(axis=0) - 1e-12) and np.all(q.max(axis=0) >= p.min(axis=0) - 1e-12))


def gap_to_rect(poly: np.ndarray, rect) -> float:
    """Distance seen from above between a convex polygon and a rectangle (x0, x1, y0, y1); 0 when they overlap
    or touch."""
    if _overlaps(poly, rect):
        return 0.0
    x0, x1, y0, y1 = rect
    dx = np.maximum(np.maximum(x0 - poly[:, 0], poly[:, 0] - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - poly[:, 1], poly[:, 1] - y1), 0.0)
    best = float(np.hypot(dx, dy).min())  # the polygon's corners to the rectangle
    corners = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float)
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    ap = corners[:, None, :] - a[None, :, :]  # (4, n, 2)
    t = np.clip((ap * ab[None]).sum(axis=2) / np.maximum((ab * ab).sum(axis=1), 1e-18)[None], 0.0, 1.0)
    d = ap - t[..., None] * ab[None]
    return float(min(best, np.hypot(d[..., 0], d[..., 1]).min()))  # ... and the rectangle's corners to its sides


def inside_margin(poly: np.ndarray) -> float:
    """How far the outline stays inside the board's edges (the white outline's centre lines): the smallest
    distance from any point of it to an edge (m; negative when some of it is outside)."""
    x0, x1, y0, y1 = BOARD_RECT
    return float(min(np.minimum.reduce([poly[:, 0] - x0, x1 - poly[:, 0], poly[:, 1] - y0, y1 - poly[:, 1]])))


def red_gap(s, poly=None) -> tuple[float, str]:
    """The smallest distance seen from above from the pin's outline to a red area, and that area's name."""
    poly = outline(s) if poly is None else poly
    return min((gap_to_rect(poly, r[1:]), r[0]) for r in L.RED)


def rail_gap(s, poly=None) -> tuple[float, str]:
    """The smallest distance seen from above from the pin's outline to a rail's outline, and the rail's name."""
    poly = outline(s) if poly is None else poly
    return min((gap_to_rect(poly, rect), name) for name, rect in RAIL_RECTS.items())


def _touching(low: float, red: float) -> bool:
    return bool(low - L.PLATE_TOP <= L.TOUCH_LOW and red <= 0.0)


def _railing(low: float, rail: float) -> bool:
    return bool(low - L.RAIL_TOP <= L.RAIL_ABOVE and rail < L.RAIL_NEAR)


def touches_red(s, poly=None) -> bool:
    """Touching a red area: its outline seen from above overlaps (or touches) a red area while its lowest point is
    within 5 mm of the plate's top (standing, lying or held on or just over the board's underside parts)."""
    return _touching(lowest(s), red_gap(s, poly)[0])


def at_rail(s, poly=None) -> bool:
    """At a rail: its outline seen from above within 5 mm of a rail's outline while its lowest point is at most
    5 mm above the rail's top (on the rail, leaning on it, or standing beside it)."""
    return _railing(lowest(s), rail_gap(s, poly)[0])


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
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


def point_pin_distance(p, s) -> float:
    """Distance from a point (or from each of an (n, 3) array of points: the smallest) to the pin, 0 inside it."""
    r = rotation(s.quat_xyzw)
    d = (np.atleast_2d(np.asarray(p, dtype=float)) - np.asarray(s.pos, dtype=float)) @ r  # in the pin's frame
    radial = np.maximum(np.hypot(d[:, 0], d[:, 1]) - L.PIN_R, 0.0)
    along = np.maximum(np.abs(d[:, 2]) - L.PIN_H / 2, 0.0)
    return float(np.hypot(radial, along).min())


def zone_gap(arm, s) -> float:
    """How far a gripper's finger zone is from a pin (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return point_pin_distance(pts, s) - r


def released(arms, s) -> bool:
    """Neither gripper's finger zone reaches the pin (unreadable arms: taken as let go)."""
    if not arms_valid(arms):
        return True
    return bool(all(zone_gap(arms[a], s) >= L.HANDS_OFF for a in ("left", "right")))


# -- supporting a quarter -----------------------------------------------------------------------------------------
def pin_report(s, arms=None) -> dict:
    """Everything the score needs to know about one pin."""
    poly = outline(s)
    tilt, low = tilt_deg(s), lowest(s)
    margin = inside_margin(poly)
    red, red_name = red_gap(s, poly)
    rail, rail_name = rail_gap(s, poly)
    upright = tilt <= L.TILT_MAX_DEG
    on_plate = abs(low - L.PLATE_TOP) <= L.BOTTOM_TOL
    free = released(arms, s)
    quarter = L.quarter_of(float(s.pos[0]), float(s.pos[1]))
    supports = bool(quarter is not None and upright and on_plate and margin >= L.EDGE_IN
                    and red >= L.RED_CLEAR and free)
    return {
        "quarter": quarter,
        "supports": supports,
        "upright": upright,
        "on_plate": on_plate,
        "released": free,
        "touches_red": _touching(low, red),
        "at_rail": _railing(low, rail),
        "tilt_deg": round(tilt, 2),
        "bottom_mm": _mm(low - L.PLATE_TOP),
        "inside_mm": _mm(margin),
        "red_mm": _mm(red),
        "red_nearest": red_name,
        "rail_mm": _mm(rail),
        "rail_nearest": rail_name,
    }


def supports_quarter(s, arms=None) -> str | None:
    """The quarter this pin supports, or None: standing upright (axis within 10 degrees of the vertical) with its
    lowest point within 3 mm of the plate's top, its whole outline at least 5 mm inside the board's edges and at
    least 3 mm clear of every red area, let go; it supports the quarter holding its centre."""
    rep = pin_report(s, arms)
    return rep["quarter"] if rep["supports"] else None


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: the last 0.5 s of each pin's path, and counts for the trace."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.paths: dict[str, deque] = {n: deque(maxlen=_WINDOW + 1) for n in L.PIN_NAMES}  # recent centres
        self.speeds: dict[str, float] = {n: 0.0 for n in L.PIN_NAMES}
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite or missing pin state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # {name: (centre, velocity)} at the previous valid control step

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: every pin standing upright in its tray near its own starting mark, still."""
        for n, (x, y) in L.STARTS.items():
            s = states[n]
            if math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) >= L.START_TOL:
                return False
            if abs(lowest(s) - L.TRAY_TOP) >= 0.003 or tilt_deg(s) >= 1.0:
                return False
            if np.linalg.norm(np.asarray(s.lin_vel, dtype=float)) >= L.STILL_SPEED:
                return False
        return True

    def _teleported(self, states) -> bool:
        """Some pin moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        for n in L.PIN_NAMES:
            p0, v0 = self.prev[n]
            c = np.asarray(states[n].pos, dtype=float)
            if float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall:
                return True
        return False

    def update(self, states, arms=None) -> Result:
        if not all_valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(states, arms)
            return self.last
        if self.prev is not None and self._teleported(states) and self.at_start(states):
            self.reset()  # the simulator put the pins back at the start: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = {n: (np.array(states[n].pos, dtype=float), np.array(states[n].lin_vel, dtype=float))
                     for n in L.PIN_NAMES}
        for n in L.PIN_NAMES:
            self.paths[n].append(np.array(states[n].pos, dtype=float))
            self.speeds[n] = float(np.linalg.norm(np.asarray(states[n].lin_vel, dtype=float)))
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(states, arms)
        return self.last

    def still(self, name: str) -> bool:
        """Moved no more than 2 mm (path length) over the last 0.5 s and slower than 5 cm/s now."""
        path = self.paths[name]
        if len(path) < _WINDOW + 1:
            return False
        pts = np.array(path)
        travel = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        return bool(travel <= L.STILL_PATH and self.speeds[name] < L.STILL_SPEED)

    def result(self, states, arms=None) -> Result:
        if not all_valid(states):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        have_arms = arms_valid(arms)
        pins = {n: pin_report(states[n], arms) for n in L.PIN_NAMES}
        on_red = [n for n, p in pins.items() if p["touches_red"]]
        on_rail = [n for n, p in pins.items() if p["at_rail"]]
        support: dict[str, list[str]] = {q: [] for q in L.QUARTER_NAMES}
        for n, p in pins.items():
            if p["supports"]:
                support[p["quarter"]].append(n)
        supported = [q for q in L.QUARTER_NAMES if support[q]]
        score = 0.0 if (on_red or on_rail) else 0.25 * len(supported)
        still_q = {q: any(self.still(n) for n in support[q]) for q in L.QUARTER_NAMES}
        still = all(still_q.values())
        success = bool(score == 1.0 and still)
        details = {
            "valid_state": True,
            "supported": supported,
            "n_supported": len(supported),
            "supporters": {q: support[q] for q in L.QUARTER_NAMES},
            "touching_red": on_red,
            "at_rail": on_rail,
            "still": still,
            "pins": pins,
            "zones_mm": ({a: {n: _mm(zone_gap(arms[a], states[n])) for n in L.PIN_NAMES} for a in ("left", "right")}
                         if have_arms else None),
            "opening": ({a: round(float(arms[a].opening), 3) for a in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
