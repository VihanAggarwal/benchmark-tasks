"""The scorer, as plain functions of the four shoes' states and the grippers'. NumPy only; no simulator needed.

The rule: on a duo-servo drum brake the primary shoe (the shorter lining) goes on the seat toward the front of the
car and the secondary (the longer lining) on the rear seat; the left and right rear brakes are mirror images, so the
arrow on each plate decides which of its two seats takes the primary. The score is read from the state at the last
control step, by an ordered list (first match wins):

    1.0   both plates correct (fine), all four shoes still for 1 s
    0.75  both plates correct (coarse)
    0.5   exactly one plate correct (coarse)
    0.25  two or more shoes seated (coarse) anywhere, no plate correct
    0     fewer than two shoes seated

A shoe is seated in a seat (fine) when it is counted (let go and settled over the last 0.5 s), flat (its up axis
within 5 degrees of vertical, so not upside down), on the plate (its lowest point within 3 mm of the plate's top)
and its mid-arc point is within 8 mm and its yaw within 8 degrees of the seat's; coarse is 14 mm and 15 degrees. A
seat holds a shoe only when exactly one shoe is seated in it. A plate is correct (at a level) when its front seat
holds a primary and its rear seat a secondary, both seated at that level. A shoe turned concave side out is 180
degrees off: not seated. A held shoe is never counted.

A tracker is fed once per control step by ``task.success()``: it keeps each shoe's last second (mid-arc point, yaw,
whether a gripper is on it), which 'counted' and 'still for 1 s' need, what the trace needs (when each score was
first reached, broken observations, unreadable arms), and recognises a new episode when the simulator puts every
shoe back on its starting mark in one step.

    Tracker.update(shoes, arms) -> Result(score, success, details)

``shoes`` (4, in the order of ``layout.NAMES``) are anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low``, ``high`` and ``origin`` (the shoe's body-frame origin: the mid-arc point of its centre line, on its bottom
face), as mechkit's AssemblyState has. A state with a non-finite number, a zero quaternion or no origin (a simulator
blow-up) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point) and ``opening``
(0 closed .. 1 fully open), or ``None`` when the arms could not be read. An arm that cannot be read is skipped by the
let-go rule (the other is still checked) and the step is counted in ``arms_unavailable``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high", "origin")
_EPS = 1e-9
SIDES = ("left", "right")
PLATE_NAMES = tuple(L.PLATES)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the quaternion not degenerate and the body origin present (a blow-up gives NaN, inf or
    zeros)."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    sizes = [p.size for p in parts]
    return bool(sizes == [3, 4, 3, 3, 3, 3, 3] and np.linalg.norm(parts[1]) > 0.5)


def wrap_deg(a: float) -> float:
    """An angle in degrees brought into -180 .. 180."""
    return (float(a) + 180.0) % 360.0 - 180.0


def mid_arc(s) -> np.ndarray:
    """The shoe's mid-arc point (its body origin), world."""
    return np.asarray(s.origin, dtype=float)


def yaw_deg(s) -> float:
    """Heading of the shoe's mid-arc direction (body +x, the convex side) seen from above, degrees."""
    r = rotation(s.quat_xyzw)
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def tilt_deg(s) -> float:
    """Angle between the shoe's up axis (body z, through the lining on its top) and the vertical: 0 flat, lining up;
    180 upside down."""
    zz = float(rotation(s.quat_xyzw)[2, 2])
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _outline() -> np.ndarray:
    """Points round the shoe's outline in its body frame (x, y): inner and outer edges every 5 degrees along the
    centre line, out to its rounded ends."""
    pts = []
    for d in np.arange(-L.SHOE_SPAN_DEG / 2, L.SHOE_SPAN_DEG / 2 + 1e-9, 5.0):
        a = math.radians(d)
        for r in (L.R_MID - L.SHOE_HALF_W, L.R_MID + L.SHOE_HALF_W):
            pts.append((L.CURVE_C[0] + r * math.cos(a), L.CURVE_C[1] + r * math.sin(a)))
    return np.array(pts)


_OUTLINE = _outline()
_BOTTOM = np.column_stack([_OUTLINE, np.zeros(len(_OUTLINE))])
_TOP = np.column_stack([_OUTLINE, np.full(len(_OUTLINE), L.SHOE_TOP)])


def lowest(s) -> float:
    """Height (world z) of the shoe's lowest point, from its pose (the corners of its bottom face, or of its top
    face when it is turned over)."""
    r = rotation(s.quat_xyzw)
    o = mid_arc(s)
    return float(min((_BOTTOM @ r.T)[:, 2].min(), (_TOP @ r.T)[:, 2].min()) + o[2])


def highest(s) -> float:
    r = rotation(s.quat_xyzw)
    o = mid_arc(s)
    return float(max((_BOTTOM @ r.T)[:, 2].max(), (_TOP @ r.T)[:, 2].max()) + o[2])


def bottom_gap(s) -> float:
    """Height of the shoe's lowest point above the plates' top (m; negative = below it)."""
    return lowest(s) - (L.TOP + L.PLATE_TOP)


def footprint_distance(s, point) -> float:
    """Horizontal distance (m) from a world point to the shoe seen from above: to its centre line (a 46 mm arc
    ending 57.4 degrees either side of mid-arc) less its 10 mm half width; 0 over the shoe."""
    r = rotation(s.quat_xyzw)
    d = np.asarray(point, dtype=float).ravel()[:3] - mid_arc(s)
    local = d @ r  # into the body frame
    u, v = local[0] - L.CURVE_C[0], local[1] - L.CURVE_C[1]
    ang = math.degrees(math.atan2(v, u))
    lim = L.END_DISC_DEG
    if -lim <= ang <= lim:
        to_line = abs(math.hypot(u, v) - L.R_MID)
    else:
        a = math.radians(math.copysign(lim, ang))
        to_line = math.hypot(u - L.R_MID * math.cos(a), v - L.R_MID * math.sin(a))
    return max(0.0, to_line - L.SHOE_HALF_W)


# -- let go: the arms (rigstate) ---------------------------------------------------------------------------------
def arm_tips(arms, side: str) -> np.ndarray | None:
    """One arm's pad-tip point, or None if it cannot be read (no snapshot, the arm missing, a non-finite value)."""
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


def held(s, arms) -> bool:
    """Not let go: some gripper's pad-tip point lower than 10 mm above the shoe's top while the shoe's footprint is
    within that gripper's finger reach of it horizontally (a finger may be touching, pressing or propping it,
    whatever the gripper's yaw). Each arm is checked on its own: an unreadable arm is skipped."""
    top = highest(s)
    for side in readable_sides(arms):
        tips = arm_tips(arms, side)
        if tips[2] - top < L.RELEASE_ABOVE and footprint_distance(s, tips) < L.finger_reach(opening(arms[side])):
            return True
    return False


# -- seats ---------------------------------------------------------------------------------------------------------
def flat_on_plate(s) -> bool:
    """Lying flat, lining up (tilt at most 5 degrees) with its lowest point within 3 mm of the plates' top."""
    return bool(tilt_deg(s) <= L.TILT_MAX_DEG + _EPS and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS)


def seat_error(s, key) -> tuple[float, float]:
    """(distance of the mid-arc point from the seat's, m; yaw error, degrees, -180 .. 180) for one seat."""
    x, y, yaw = L.SEATS[key]
    o = mid_arc(s)
    return float(math.hypot(o[0] - x, o[1] - y)), wrap_deg(yaw_deg(s) - yaw)


def fits(s, key, level: str) -> bool:
    """The pose part of 'seated' at a level ('fine' or 'coarse'): flat on the plate, near the seat, turned to it."""
    pos_tol, yaw_tol = (L.FINE_POS, L.FINE_YAW_DEG) if level == "fine" else (L.COARSE_POS, L.COARSE_YAW_DEG)
    d, e = seat_error(s, key)
    return bool(flat_on_plate(s) and d <= pos_tol + _EPS and abs(e) <= yaw_tol + _EPS)


def nearest_seat(s) -> tuple[tuple, float, float]:
    """The seat whose point is nearest the shoe's mid-arc point, and the errors to it."""
    best = min(L.SEAT_KEYS, key=lambda k: seat_error(s, k)[0])
    return (best, *seat_error(s, best))


def evaluate(shoes, counted, still_1s) -> tuple[float, dict]:
    """The score of the brakes as they stand, and what was found (JSON-clean). ``counted[i]``: shoe i let go and
    settled over the last 0.5 s; ``still_1s[i]``: settled in each of the last two 0.5 s windows."""
    names = list(L.NAMES)
    seated = {lvl: {} for lvl in ("fine", "coarse")}  # level -> seat -> [shoe names seated there]
    for i, s in enumerate(shoes):
        if not counted[i]:
            continue
        for lvl in seated:
            for key in L.SEAT_KEYS:
                if fits(s, key, lvl):
                    seated[lvl].setdefault(key, []).append(names[i])

    def holds(lvl, key):
        there = seated[lvl].get(key, [])
        return there[0] if len(there) == 1 else None

    def correct(lvl, plate):
        f, r = holds(lvl, (plate, "front")), holds(lvl, (plate, "rear"))
        return bool(f in L.PRIMARIES and r in L.SECONDARIES)

    fine = {p: correct("fine", p) for p in PLATE_NAMES}
    coarse = {p: correct("coarse", p) for p in PLATE_NAMES}
    n_seated = len({n for there in seated["coarse"].values() for n in there})
    all_still = bool(all(still_1s))
    if all(fine.values()) and all_still:
        score, line = 1.0, 1
    elif all(coarse.values()):
        score, line = 0.75, 2
    elif sum(coarse.values()) == 1:
        score, line = 0.5, 3
    elif n_seated >= 2:
        score, line = 0.25, 4
    else:
        score, line = 0.0, 5
    d = {"line": line,
         "plate_fine": fine, "plate_coarse": coarse,
         "seats_fine": {f"{k[0]}:{k[1]}": v for k, v in sorted(seated["fine"].items())},
         "seats_coarse": {f"{k[0]}:{k[1]}": v for k, v in sorted(seated["coarse"].items())},
         "n_seated": n_seated, "counted": [names[i] for i, c in enumerate(counted) if c],
         "all_still_1s": all_still}
    return score, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and np.isfinite(v) else None


class Tracker:
    """What 'counted', 'still for 1 s', success and the trace need about this trial (the seating itself depends only
    on the state at hand)."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # observations with NaN, inf, a zero quaternion or no origin in them
        self.arms_unavailable = 0  # steps on which rigstate could not read one arm or both
        self.arms_unavailable_by_side = {side: 0 for side in SIDES}
        self.first_at: dict[float, int] = {}  # each score -> the first control step it was reached
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(mid-arc point, velocity)] of every shoe at the previous valid control step
        # every shoe's (mid-arc point, yaw in degrees, held) at the last 2 x 0.5 s + 1 valid steps
        self.history: deque = deque(maxlen=2 * self.window + 1)

    @staticmethod
    def at_start(shoes) -> bool:
        """The starting layout: every shoe still and flat on the bench, on its own mark."""
        for s, name in zip(shoes, L.NAMES, strict=True):
            _, x0, y0, yaw0 = L.SHOES[name]
            o = mid_arc(s)
            on_mark = math.hypot(o[0] - x0, o[1] - y0) < L.START_TOL
            on_bench = abs(lowest(s) - L.TOP) < L.SEAT_TOL
            resting = np.linalg.norm(s.lin_vel) < 0.01 and np.linalg.norm(s.ang_vel) < 0.5
            if not (on_mark and on_bench and tilt_deg(s) < 1.0 and resting):
                return False
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def _windows(self, i: int, k: int):
        """Shoe i's (path m, yaw range deg, ever held) over the k-th 0.5 s window back (k = 0: the last), or None
        before enough valid steps have been seen."""
        h = list(self.history)
        w = self.window
        end = len(h) - k * w
        start = end - w - 1
        if start < 0:
            return None
        seg = h[start:end]
        pts = [e[i][0] for e in seg]
        path = float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))
        yaws = np.unwrap(np.radians([e[i][1] for e in seg]))
        yaw_range = float(np.degrees(yaws.max() - yaws.min()))
        ever_held = any(e[i][2] for e in seg)
        return path, yaw_range, ever_held

    @staticmethod
    def _settled(win) -> bool:
        return win is not None and win[0] < L.STILL_PATH and win[1] < L.STILL_YAW_DEG

    def flags(self) -> tuple[list[bool], list[bool], list]:
        """Per shoe: counted (let go at every step and settled over the last 0.5 s), still for 1 s (settled in each
        of the last two 0.5 s windows), and the last window's numbers."""
        n = len(self.history[-1]) if self.history else len(L.NAMES)
        counted, still, last = [], [], []
        for i in range(n):
            w0, w1 = self._windows(i, 0), self._windows(i, 1)
            counted.append(bool(self._settled(w0) and not w0[2]))
            still.append(bool(self._settled(w0) and self._settled(w1)))
            last.append(w0)
        return counted, still, last

    def update(self, shoes, arms=None) -> Result:
        ss = list(shoes)
        self.steps += 1
        if len(ss) != len(L.NAMES) or not all(valid(s) for s in ss):  # a blow-up: score 0, keep the tracker
            self.bad_steps += 1
            self.last = self.result(ss, arms)
            return self.last
        now = [(mid_arc(s).copy(), np.array(s.lin_vel, dtype=float)) for s in ss]
        if self.prev is not None and self._teleported(now) and self.at_start(ss):
            self.reset()  # the simulator put every shoe back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([(mid_arc(s).copy(), yaw_deg(s), held(s, arms)) for s in ss])
        readable = readable_sides(arms)
        for side in SIDES:
            if side not in readable:
                self.arms_unavailable_by_side[side] += 1
        if len(readable) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(ss, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, shoes, arms=None) -> Result:
        """The score of the state at hand, with the last second the tracker has seen. A shoe held at this step is
        never counted, whatever the history says."""
        ss = list(shoes)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_readable": list(readable_sides(arms)), "arms_unavailable": self.arms_unavailable,
                 "arms_unavailable_by_side": dict(self.arms_unavailable_by_side)}
        bad = [L.NAMES[i] for i, s in enumerate(ss) if i < len(L.NAMES) and not valid(s)]
        if bad or len(ss) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_shoes": bad, **trial})
        if self.history:
            counted, still, last = self.flags()
        else:
            counted, still, last = [False] * len(ss), [False] * len(ss), [None] * len(ss)
        now_held = [held(s, arms) for s in ss]
        counted = [c and not h for c, h in zip(counted, now_held, strict=True)]
        score, details = evaluate(ss, counted, still)
        per = {}
        for i, s in enumerate(ss):
            key, dist, err = nearest_seat(s)
            w = last[i]
            per[L.NAMES[i]] = {
                "mid_mm": [round(float(v) * 1000, 1) for v in mid_arc(s)],
                "yaw_deg": round(yaw_deg(s), 2), "tilt_deg": round(tilt_deg(s), 2),
                "bottom_mm": _mm(bottom_gap(s)), "held": now_held[i], "counted": counted[i], "still_1s": still[i],
                "nearest_seat": f"{key[0]}:{key[1]}", "off_seat_mm": _mm(dist), "yaw_err_deg": round(err, 2),
                "path_mm": None if w is None else _mm(w[0]), "yaw_range_deg": None if w is None else round(w[1], 3)}
        success = bool(score == 1.0)
        return Result(score, success, {"valid_state": True, **details, "shoes": per, **trial})
