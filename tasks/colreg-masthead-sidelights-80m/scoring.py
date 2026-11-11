"""The scorer, as plain functions of the four light units' states. NumPy only; no simulator needed.

The rule (COLREG 1972 Rules 21 and 23, Annex I section 3, IMO MSC.1/Circ.1260/Rev.1), on a 1:100 deck model of an
80 m ship with its bow toward +y: two white masthead lights over the centreline, the forward one (F) not more than
a quarter length (200 mm) from the stem and the after one (A) at least half a length (400 mm) abaft it; the red
sidelight at the port side (-x) and the green at the starboard side (+x), each with its outboard face at most 1 m
(10 mm) inboard of the side and not in front of F. Each correct light earns 0.25. F is the standing white unit
nearer the bow; A, red and green are judged only when F stands over the centreline.

A unit counts only when it 'stands': upright (tilt at most 10 degrees), its lowest point within 3 mm of the deck
top, its whole footprint inside the deck outline, let go (no gripper's pad-tip point lower than 10 mm above its
top within that gripper's finger reach), and still (its centre's path under 1.5 mm in each of the last two 0.5 s
windows). A small tracker, fed once per control step by ``task.success()``, keeps each unit's recent path (for
'still'), what the trace needs, and recognises a new episode when the simulator puts every unit back on its mark.

    Tracker.update(units, arms) -> Result(score, success, details)

``units`` (4, in the order of ``layout.NAMES``: masthead_1, masthead_2, red, green) are anything with ``pos``,
``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. A state
with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()`` (``{"left": Arm, "right": Arm}`` with ``tips`` and ``opening``) or None when the arms
could not be read; an unreadable arm is skipped by the let-go rule (the other is still checked).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_EPS = 1e-9
SIDES = ("left", "right")
_CUBE_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes."""
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
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3)


def up_axis(s, kind: str) -> np.ndarray:
    """The unit's axis nearest the vertical, pointing up: a cylinder's own axis, or a cube's face normal."""
    r = rotation(s.quat_xyzw)
    k = 2 if kind == "cyl" else int(np.argmax(np.abs(r[2, :])))
    a = r[:, k]
    return a if a[2] >= 0.0 else -a


def tilt_deg(s, kind: str) -> float:
    """Angle between the unit's up axis and the vertical (degrees, 0..90; a cube on any face, a cylinder either end
    up)."""
    return float(np.degrees(np.arccos(np.clip(float(up_axis(s, kind)[2]), 0.0, 1.0))))


def _cube_points(s) -> np.ndarray:
    return np.asarray(s.pos, dtype=float) + (rotation(s.quat_xyzw) @ (_CUBE_CORNERS * (L.SIDE / 2)).T).T


def lowest(s, kind: str) -> float:
    """Height (world z) of the unit's lowest point, from its pose."""
    if kind == "cube":
        return float(_cube_points(s)[:, 2].min())
    az = abs(float(rotation(s.quat_xyzw)[2, 2]))
    return float(s.pos[2]) - L.MAST_H / 2 * az - L.MAST_RADIUS * math.sqrt(max(0.0, 1.0 - az * az))


def highest(s, kind: str) -> float:
    if kind == "cube":
        return float(_cube_points(s)[:, 2].max())
    az = abs(float(rotation(s.quat_xyzw)[2, 2]))
    return float(s.pos[2]) + L.MAST_H / 2 * az + L.MAST_RADIUS * math.sqrt(max(0.0, 1.0 - az * az))


def bottom_gap(s, kind: str) -> float:
    """Height of the unit's lowest point above the deck top (m; negative = below it)."""
    return lowest(s, kind) - L.DECK_TOP


def footprint(s, kind: str) -> tuple[np.ndarray, float, np.ndarray]:
    """What stands on the deck, seen from above: (points (n, 2), radius, centre (2,)). A cube: its four lowest
    corners (radius 0); a cylinder: its lower end's centre with its radius."""
    if kind == "cube":
        pts = _cube_points(s)
        low4 = pts[np.argsort(pts[:, 2], kind="stable")[:4], :2]
        return low4, 0.0, low4.mean(axis=0)
    c = np.asarray(s.pos, dtype=float)
    base = (c - up_axis(s, kind) * L.MAST_H / 2)[:2]
    return base[None, :], L.MAST_RADIUS, base


def edge_clearances(points, radius: float) -> list[float]:
    """For each edge of the deck outline, the footprint's least distance inside that edge's line (m; negative =
    over it). The outline is convex, so the footprint is on the deck iff every value is >= 0, and the least value
    over an edge set is the footprint's gap to that part of the boundary."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    out = []
    n = len(L.OUTLINE)
    for i in range(n):
        a = np.array(L.OUTLINE[i], dtype=float)
        b = np.array(L.OUTLINE[(i + 1) % n], dtype=float)
        e = b - a
        inward = np.array([-e[1], e[0]]) / float(np.linalg.norm(e))  # left of a CCW edge: inside
        out.append(float(((p - a) @ inward).min()) - radius)
    return out


@dataclass
class UnitView:
    """What the scorer found about one unit at this step (JSON-clean through as_dict)."""
    name: str
    x: float
    y: float
    tilt: float
    bottom: float
    on_deck: bool
    port_gap: float
    stbd_gap: float
    held: bool
    still: bool

    @property
    def stands(self) -> bool:
        return bool(self.tilt <= L.UPRIGHT_DEG + _EPS and abs(self.bottom) <= L.SEAT_TOL + _EPS and self.on_deck
                    and not self.held and self.still)

    def as_dict(self) -> dict:
        return {"x_mm": _mm(self.x), "y_mm": _mm(self.y), "tilt_deg": round(self.tilt, 2),
                "bottom_mm": _mm(self.bottom), "on_deck": self.on_deck, "port_gap_mm": _mm(self.port_gap),
                "stbd_gap_mm": _mm(self.stbd_gap), "held": self.held, "still": self.still, "stands": self.stands,
                "off_centreline_mm": _mm(self.x - L.CL_X)}


# -- let go: the arms (rigstate) ---------------------------------------------------------------------------------
def arm_tips(arms, side: str) -> np.ndarray | None:
    """One arm's pad-tip point, or None if it cannot be read."""
    try:
        tips = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, IndexError, AttributeError, TypeError, ValueError):
        return None
    return tips if tips.size == 3 and np.all(np.isfinite(tips)) else None


def readable_sides(arms) -> tuple[str, ...]:
    return tuple(side for side in SIDES if arm_tips(arms, side) is not None)


def opening(arm) -> float:
    """rigstate's opening, 0 (closed) .. 1 (fully open); an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def held(s, name: str, arms) -> bool:
    """Not let go: some readable gripper's pad-tip point lower than 10 mm above the unit's top with the unit's centre
    within that gripper's finger reach horizontally (whatever the gripper's yaw)."""
    kind = L.KIND[name]
    top = highest(s, kind)
    c = np.asarray(s.pos, dtype=float)
    for side in readable_sides(arms):
        p = arm_tips(arms, side)
        if float(np.hypot(*(p[:2] - c[:2]))) < L.release_reach(opening(arms[side]), name) and \
                float(p[2] - top) < L.RELEASE_ABOVE:
            return True
    return False


def view(s, name: str, arms=None, still: bool = True) -> UnitView:
    kind = L.KIND[name]
    pts, r, centre = footprint(s, kind)
    clear = edge_clearances(pts, r)
    return UnitView(name=name, x=float(centre[0]), y=float(centre[1]), tilt=tilt_deg(s, kind),
                    bottom=bottom_gap(s, kind), on_deck=bool(min(clear) >= -_EPS),
                    port_gap=min(clear[i] for i in L.PORT_EDGES), stbd_gap=min(clear[i] for i in L.STBD_EDGES),
                    held=held(s, name, arms), still=bool(still))


def on_centreline(u: UnitView) -> bool:
    return bool(abs(u.x - L.CL_X) <= L.CL_TOL + _EPS)


def sidelight_ok(u: UnitView, f: UnitView, port: bool) -> bool:
    """Standing on its own side's half of the deck, its footprint at most 10 mm (1 m) inboard of that side, and
    its centre not in front of (not nearer the bow than) F's."""
    if not u.stands:
        return False
    on_half = u.x < L.CL_X - _EPS if port else u.x > L.CL_X + _EPS
    gap = u.port_gap if port else u.stbd_gap
    return bool(on_half and gap <= L.SIDE_BAND + _EPS and u.y <= f.y + _EPS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def evaluate(units, arms=None, still=None) -> tuple[float, dict]:
    """The score of the lights as they stand, and what was found. ``still``: one bool per unit (default all)."""
    us = list(units)
    still = [True] * len(us) if still is None else list(still)
    views = {n: view(s, n, arms, st) for n, s, st in zip(L.NAMES, us, still, strict=True)}
    whites = sorted((views[n] for n in L.WHITE_UNITS if views[n].stands), key=lambda u: u.y, reverse=True)
    f = whites[0] if whites else None
    a = whites[1] if len(whites) > 1 else None
    d: dict = {"F": f.name if f else None, "A": a.name if a else None, "gate": False,
               "F_ok": False, "A_ok": False, "red_ok": False, "green_ok": False,
               "F_from_stem_mm": _mm(L.STEM_Y - f.y) if f else None,
               "separation_mm": _mm(f.y - a.y) if f and a else None,
               "units": {n: v.as_dict() for n, v in views.items()}}
    if f is None:
        return 0.0, d
    d["F_ok"] = bool(on_centreline(f) and f.y >= L.F_MIN_Y - _EPS)
    d["gate"] = on_centreline(f)
    if d["gate"]:
        d["A_ok"] = bool(a is not None and on_centreline(a) and f.y - a.y >= L.MIN_SEPARATION - _EPS)
        d["red_ok"] = sidelight_ok(views[L.RED], f, port=True)
        d["green_ok"] = sidelight_ok(views[L.GREEN], f, port=False)
    score = 0.25 * sum(bool(d[k]) for k in ("F_ok", "A_ok", "red_ok", "green_ok"))
    return score, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What 'still' and the trace need about this trial; the score depends only on the state at hand and the
    units' paths over the last second."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None
        self.history: deque = deque(maxlen=self.window * L.STILL_WINDOWS + 1)

    @staticmethod
    def at_start(units) -> bool:
        """The starting layout: every unit upright and at rest on the table on its own mark."""
        for s, name in zip(units, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            kind = L.KIND[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(lowest(s, kind) - L.TOP) < L.SEAT_TOL
            resting = np.linalg.norm(s.lin_vel) < 0.01 and np.linalg.norm(s.ang_vel) < 0.5
            if not (on_mark and seated and tilt_deg(s, kind) < 1.0 and resting):
                return False
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for (p, _), (p0, v0) in zip(now, self.prev, strict=True))

    def window_paths(self) -> list[list[float]] | None:
        """Each unit's centre path (m) in each of the last two 0.5 s windows (oldest first), or None before 1 s of
        valid steps has been seen."""
        if len(self.history) < self.window * L.STILL_WINDOWS + 1:
            return None
        h = list(self.history)
        out = []
        for i in range(len(h[0])):
            per = []
            for w in range(L.STILL_WINDOWS):
                seg = h[w * self.window:(w + 1) * self.window + 1]
                per.append(float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(seg, seg[1:]))))
            out.append(per)
        return out

    def still_flags(self) -> list[bool]:
        paths = self.window_paths()
        if paths is None:
            return [False] * len(L.NAMES)
        return [all(p < L.STILL_PATH for p in per) for per in paths]

    def update(self, units, arms=None) -> Result:
        us = list(units)
        self.steps += 1
        if len(us) != len(L.NAMES) or not all(valid(s) for s in us):
            self.bad_steps += 1
            self.last = self.result(us, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in us]
        if self.prev is not None and self._teleported(now) and self.at_start(us):
            self.reset()  # the simulator put every unit back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(us, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, units, arms=None) -> Result:
        us = list(units)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_readable": list(readable_sides(arms)), "arms_unavailable": self.arms_unavailable}
        if len(us) != len(L.NAMES) or not all(valid(s) for s in us):
            bad = [L.NAMES[i] for i, s in enumerate(us[:len(L.NAMES)]) if not valid(s)]
            return Result(0.0, False, {"valid_state": False, "bad_units": bad, **trial})
        paths = self.window_paths()
        score, details = evaluate(us, arms, self.still_flags())
        details["paths_mm"] = None if paths is None else {
            n: [round(p * 1000, 2) for p in per] for n, per in zip(L.NAMES, paths, strict=True)}
        return Result(score, bool(score == 1.0), {"valid_state": True, **details, **trial})
