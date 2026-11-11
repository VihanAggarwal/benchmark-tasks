"""The scorer, as plain functions of the two grab bars' states and the grippers' state. NumPy only; no simulator.

The rule is the 2010 ADA Standards' grab-bar layout for a water closet (604.5), read from where the bars lie at the
end, as an inspector checks a fit-out with a tape: is a bar along each wall, and does each wall's bar meet that
wall's rule (side wall, 604.5.1: at least 42 in long, rear end at most 12 in from the rear wall, front end at least
54 in from it; rear wall, 604.5.2: at least 36 in long, reaching at least 12 in past the toilet's centreline toward
the side wall and at least 24 in past it toward the open side). Each wall is judged by what the code asks of it,
whichever bar is there. A bar counts only once it is let go and has settled; a small tracker, fed once per control
step by ``task.success()``, keeps each bar's last second of centre, heading and let-go flags for that:

    Tracker.update(bars, arms) -> Result(score, success, details)

``bars`` (2, in the order of ``layout.NAMES``: the 48 in bar, then the 40 in bar) are anything with ``pos``
(centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState``
has. Each is a box whose length is its local x axis; its square section makes its four long faces alike. A state
with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker
as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the let-go condition is
taken as met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so a
gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is every point within a radius of the
38 mm segment from the point between its pad tips back up the gripper; the radius reaches a pad's outer corner (half
the current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across; 57 mm fully open).
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
WALLS = ("side", "rear")


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
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3)


# -- a bar's pose -------------------------------------------------------------------------------------------------
def heading_deg(s) -> float:
    """The bar's length seen from above, counter-clockwise from +x, either way round: 0 .. 180 degrees (0 along x,
    the side wall's direction; 90 along y, the rear wall's)."""
    a = rotation(s.quat_xyzw)[:, 0]
    if math.hypot(a[0], a[1]) < 1e-9:  # standing on an end
        return float("nan")
    return float(math.degrees(math.atan2(a[1], a[0])) % 180.0)


def heading_change_deg(h1: float, h0: float) -> float:
    """The smallest turn between two headings (degrees, 0..90; a bar's two ends are alike)."""
    if not (math.isfinite(h1) and math.isfinite(h0)):
        return 90.0
    d = abs(h1 - h0) % 180.0
    return min(d, 180.0 - d)


def flat_err_deg(s) -> float:
    """Angle between the vertical and the nearest of the bar's four long faces' normals (degrees, 0..90): 0 lying
    flat on a long face; tilted end-up, rolled onto an edge, or standing, it grows."""
    r = rotation(s.quat_xyzw)
    best = max(abs(float(r[2, 1])), abs(float(r[2, 2])))
    return float(math.degrees(math.acos(min(1.0, best))))


def lowest(s, name: str) -> float:
    """Height (world z) of the bar's lowest corner, from its pose."""
    r = rotation(s.quat_xyzw)
    half = (L.BAR_LEN[name] / 2, L.BAR_W / 2, L.BAR_W / 2)
    return float(s.pos[2]) - sum(abs(float(r[2, i])) * half[i] for i in range(3))


def ends(s, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The bar's two end points seen from above (x, y): its centre plus and minus half its length along its axis."""
    a = rotation(s.quat_xyzw)[:2, 0] * (L.BAR_LEN[name] / 2)
    c = np.asarray(s.pos, dtype=float)[:2]
    return c - a, c + a


def on_floor(s, name: str) -> bool:
    """Flat (a long face within 5 degrees of facing down) with its lowest corner within 3 mm of the floor."""
    return bool(flat_err_deg(s) <= L.FLAT_DEG + _EPS and abs(lowest(s, name) - L.TOP) <= L.ON_FLOOR + _EPS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def along(s, name: str) -> dict:
    """Which wall the bar lies along (pose only; let-go and settled are the tracker's), and that wall's code
    readings. 'Along the side wall': on the floor, its length within 5 degrees of x, its wall-side face 0 to 15 mm
    off the side kerb's face at its midpoint and both its ends beside the kerb (x 0.200 to 0.460). 'Along the rear
    wall': the same with y, the rear kerb, and its ends between y -0.320 and +0.010. A bar parallel to one wall
    cannot be along the other."""
    h = heading_deg(s)
    a, b = ends(s, name)
    c = np.asarray(s.pos, dtype=float)
    d: dict = {"wall": None, "heading_deg": None if not math.isfinite(h) else round(h, 2),
               "flat_err_deg": round(flat_err_deg(s), 2), "low_mm": _mm(lowest(s, name) - L.TOP),
               "ends_mm": [[_mm(v) for v in a], [_mm(v) for v in b]], "gap_mm": None, "compliant": False,
               "why_not": ""}
    if not on_floor(s, name):
        d["why_not"] = "not_flat_on_floor"
        return d
    if heading_change_deg(h, 0.0) <= L.PARALLEL_DEG + _EPS:  # along x: beside the side wall?
        gap = (float(c[1]) - L.SIDE_FACE_Y) - L.BAR_W / 2
        lo, hi = min(a[0], b[0]), max(a[0], b[0])
        beside = L.SIDE_SPAN[0] - L.SPAN_TOL - _EPS <= lo and hi <= L.SIDE_SPAN[1] + L.SPAN_TOL + _EPS
        d["gap_mm"] = _mm(gap)
        if not (L.WALL_GAP[0] - _EPS <= gap <= L.WALL_GAP[1] + _EPS and beside):
            d["why_not"] = "not_beside_a_wall"
            return d
        rear_off, extent = L.REAR_FACE_X - hi, L.REAR_FACE_X - lo
        ok_len = L.BAR_LEN[name] >= L.SIDE_LEN_MIN - _EPS
        ok_rear = rear_off <= L.SIDE_REAR_MAX + _EPS
        ok_ext = extent >= L.SIDE_EXTENT_MIN - _EPS
        d.update(wall="side", rear_end_from_rear_wall_mm=_mm(rear_off), front_end_from_rear_wall_mm=_mm(extent),
                 compliant=bool(ok_len and ok_rear and ok_ext),
                 why_not="" if ok_len and ok_rear and ok_ext else
                 ("under_42_in" if not ok_len else "rear_end_over_12_in" if not ok_rear else "under_54_in_out"))
        return d
    if heading_change_deg(h, 90.0) <= L.PARALLEL_DEG + _EPS:  # along y: beside the rear wall?
        gap = (L.REAR_FACE_X - float(c[0])) - L.BAR_W / 2
        lo, hi = min(a[1], b[1]), max(a[1], b[1])
        beside = L.REAR_SPAN[0] - L.SPAN_TOL - _EPS <= lo and hi <= L.REAR_SPAN[1] + L.SPAN_TOL + _EPS
        d["gap_mm"] = _mm(gap)
        if not (L.WALL_GAP[0] - _EPS <= gap <= L.WALL_GAP[1] + _EPS and beside):
            d["why_not"] = "not_beside_a_wall"
            return d
        wall_side, open_side = L.CL_Y - lo, hi - L.CL_Y
        ok_len = L.BAR_LEN[name] >= L.REAR_LEN_MIN - _EPS
        ok_wall = wall_side >= L.REAR_WALL_SIDE_MIN - _EPS
        ok_open = open_side >= L.REAR_OPEN_SIDE_MIN - _EPS
        d.update(wall="rear", wall_side_past_cl_mm=_mm(wall_side), open_side_past_cl_mm=_mm(open_side),
                 compliant=bool(ok_len and ok_wall and ok_open),
                 why_not="" if ok_len and ok_wall and ok_open else
                 ("under_36_in" if not ok_len else "wall_side_under_12_in" if not ok_wall else "open_side_under_24_in"))
        return d
    d["why_not"] = "not_parallel_to_a_wall"
    return d


# -- the grippers ---------------------------------------------------------------------------------------------------
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


def point_distance(p, s, name: str) -> float:
    """Distance from a point to the bar's box (0 inside it), in the bar's own frame."""
    q = rotation(s.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))
    half = np.array([L.BAR_LEN[name] / 2, L.BAR_W / 2, L.BAR_W / 2])
    return float(np.linalg.norm(np.maximum(np.abs(q) - half, 0.0)))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, s, name: str, samples: int = 20) -> float:
    """How far a gripper's finger zone is from the bar (negative: they overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return min(point_distance(p, s, name) for p in pts) - zone_radius(arm)


def hands_off(s, name: str, arms) -> tuple[bool, list | None]:
    """Both grippers' finger zones at least 10 mm from the bar at this step (taken as met when the arms cannot be
    read), and the two gaps (mm)."""
    if not arms_valid(arms):
        return True, None
    gaps = [zone_gap(arms[side], s, name) for side in SIDES]
    return bool(min(gaps) >= L.HANDS_OFF - _EPS), [round(g * 1000, 1) for g in gaps]


# -- the score ------------------------------------------------------------------------------------------------------
def decide(walls: dict) -> tuple[float, str]:
    """The ordered decision list (first match wins). ``walls``: for 'side' and 'rear', ``has`` (a counted bar lies
    along it) and ``ok`` (a counted bar along it meets its rule); and ``still_1s`` (every bar still for 1 s)."""
    side, rear = walls["side"], walls["rear"]
    if side["ok"] and rear["ok"] and walls["still_1s"]:
        return 1.0, "both_walls_compliant"
    if (side["ok"] and rear["has"]) or (rear["ok"] and side["has"]):
        return 0.75, "one_wall_compliant_other_wall_has_a_bar"
    if side["ok"] or rear["ok"]:
        return 0.5, "one_wall_compliant_other_wall_bare"
    if side["has"] or rear["has"]:
        return 0.25, "bar_on_a_wall_none_compliant"
    return 0.0, "no_bar_on_a_wall"


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each bar's centre, heading and let-go flag over the last second,
    and how many steps were broken or had the arms unreadable."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf or a zero quaternion in them
        self.arms_unavailable = 0
        self.first_at: dict[float, int] = {}  # each tier -> the first control step it was reached
        self.best = 0.0
        self.prev = None  # [(centre, velocity)] of each bar at the previous valid step
        # per valid step: [(centre, heading, hands_off)] for each bar; this step and the 2 windows before it
        self.history: deque = deque(maxlen=2 * self.window + 1)
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(bars) -> bool:
        """The starting layout: each bar flat on the floor on its own mark, still, turned as it starts."""
        for s, name in zip(bars, L.NAMES, strict=True):
            x0, y0, yaw0 = L.BAR_START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            turned = heading_change_deg(heading_deg(s), yaw0 % 180.0) < L.START_YAW_TOL_DEG
            slow = float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) < 0.01
            if not (on_mark and turned and on_floor(s, name) and slow):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some bar moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for (p, _), (p0, v0) in zip(now, self.prev, strict=True))

    def update(self, bars, arms=None) -> Result:
        bs = list(bars)
        self.steps += 1
        if len(bs) != len(L.NAMES) or not all(valid(s) for s in bs):  # a blow-up: 0, keep the record as it was
            self.bad_steps += 1
            self.last = self.result(bs, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in bs]
        if self.prev is not None and self._teleported(now) and self.at_start(bs):
            self.reset()  # the simulator put both bars back on their marks: a new episode
            self.steps = 1
        self.prev = now
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.history.append([(np.array(s.pos, dtype=float), heading_deg(s), hands_off(s, n, arms)[0])
                             for s, n in zip(bs, L.NAMES, strict=True)])
        self.last = self.result(bs, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def _window(self, i: int, k: int) -> list | None:
        """Bar i's history entries for window k (0: the last 0.5 s, 1: the 0.5 s before it), window + 1 entries each,
        or None if not seen yet."""
        h = list(self.history)
        end = len(h) - k * self.window
        start = end - self.window - 1
        if start < 0:
            return None
        return [e[i] for e in h[start:end]]

    def _settled(self, i: int, k: int) -> tuple[bool, float | None, float | None]:
        w = self._window(i, k)
        if w is None:
            return False, None, None
        path = float(sum(np.linalg.norm(b[0] - a[0]) for a, b in zip(w, w[1:])))
        turn = max(heading_change_deg(e[1], w[-1][1]) for e in w)
        return bool(path < L.STILL_PATH and turn < L.STILL_YAW_DEG), path, turn

    def _released(self, i: int) -> bool:
        w = self._window(i, 0)
        return w is not None and all(e[2] for e in w)

    def result(self, bars, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        bs = list(bars)
        trial = {"steps": self.steps, "bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_unavailable": self.arms_unavailable}
        if len(bs) != len(L.NAMES) or not all(valid(s) for s in bs):
            bad = [n for s, n in zip(bs, L.NAMES) if not valid(s)]
            return Result(0.0, False, {"valid_state": False, "bad_bars": bad, "tier_reason": "invalid_state",
                                       **trial})
        walls = {w: {"has": False, "ok": False} for w in WALLS}
        per_bar, still_all = {}, True
        for i, (s, name) in enumerate(zip(bs, L.NAMES, strict=True)):
            d = along(s, name)
            settled, path, turn = self._settled(i, 0)
            settled_before = self._settled(i, 1)[0]
            released = self._released(i)
            off_now, gaps = hands_off(s, name, arms)
            counted = bool(settled and released)
            still_all = still_all and counted and settled_before
            d.update(settled=settled, released=released, counted=counted, still_1s=counted and settled_before,
                     path_mm=None if path is None else round(path * 1000, 2),
                     turn_deg=None if turn is None else round(turn, 2), hands_off_now=off_now, zone_gap_mm=gaps,
                     centre_mm=[_mm(v) for v in s.pos])
            if counted and d["wall"] is not None:
                walls[d["wall"]]["has"] = True
                walls[d["wall"]]["ok"] = walls[d["wall"]]["ok"] or d["compliant"]
            per_bar[name] = d
        score, reason = decide({**walls, "still_1s": still_all})
        details = {"valid_state": True, "tier_reason": reason, "side_wall": walls["side"], "rear_wall": walls["rear"],
                   "all_still_1s": still_all, **{f"{n}": per_bar[n] for n in L.NAMES}, **trial}
        return Result(score, bool(score == 1.0), details)
