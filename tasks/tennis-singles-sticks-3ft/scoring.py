"""The scorer, as plain functions of the two singles sticks' states and the grippers' state. NumPy only; no simulator.

The rule is about where the sticks end up, as a court crew or an umpire checks them with a tape before a singles
match on a doubles court: is each stick standing upright on the net line, and is its centre 0.914 m (3 ft) outside
the singles court, two thirds of the way across its alley (|y| 0.3353 m at 1:15). Each side of the court (alley)
scores on its own and at most one stick counts per side:

    stick score: 0.5 at the ITF point (| |y| - 0.3353 | <= 8 mm), 0.25 elsewhere in its alley (0.270 <= |y| <= 0.370),
                 0 otherwise; and 0 unless the stick is 'set': let go, still for 1 s, upright, standing on the court
                 and on the net line (its centre within 6 mm of it along x).
    side score:  the best stick score among the sticks on that side (y > 0: the +y side, y < 0: the -y side).
    task score:  +y side + -y side: 0, 0.25, 0.5, 0.75 or 1.0.

A small tracker is fed once per control step by ``task.success()``: it keeps the middle of each stick's axis over the last 1 s
(for 'still'), counts broken observations and unreadable arms, and recognises a new episode when the simulator puts
both sticks back on their starting marks in one step.

    Tracker.update(sticks, arms) -> Result(score, success, details)

``sticks`` (2, in the order of ``layout.NAMES``) are each stick's state as mechkit reports it (``AssemblyState``):
``origin`` (its base frame's origin: the middle of the foot's underside, where the post's axis meets the court),
``quat_xyzw`` (that frame's orientation, local +z up the post) and the ``ObjectState`` fields ``pos``, ``lin_vel``,
``ang_vel``. A state with a non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up, or a stick
not built as an assembly) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point) and ``opening``
(0 closed .. 1 open), or ``None`` when the arms could not be read. An arm that cannot be read is skipped by the
let-go rule (the other is still checked) and the step is counted in ``arms_unavailable``.

The two sticks are identical: either may go on either side. A stick's position is its centre on the court (the
base frame's origin), read from above.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_EPS = 1e-9
SIDES = ("left", "right")


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world.
    (Callers check ``valid`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid(s) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def to_world(s, local) -> np.ndarray:
    return np.asarray(s.origin, dtype=float) + rotation(s.quat_xyzw) @ np.asarray(local, dtype=float)


def _corners(center, size) -> np.ndarray:
    c, h = np.asarray(center, dtype=float), np.asarray(size, dtype=float) / 2
    return np.array([c + h * np.array(sg) for sg in
                     ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1),
                      (1, 1, 1))])


LOCAL_CORNERS = np.vstack([_corners(L.FOOT_CENTER, L.FOOT_SIZE), _corners(L.POST_CENTER, L.POST_SIZE)])


def world_corners(s) -> np.ndarray:
    """The 16 corners of the foot and the post, in the world."""
    return LOCAL_CORNERS @ rotation(s.quat_xyzw).T + np.asarray(s.origin, dtype=float)


def lowest(s) -> float:
    return float(world_corners(s)[:, 2].min())


def highest(s) -> float:
    return float(world_corners(s)[:, 2].max())


def bottom_gap(s) -> float:
    """Height of the stick's lowest point above the court surface (m; negative = below it)."""
    return lowest(s) - L.COURT_TOP


def tilt_deg(s) -> float:
    """Angle between the post's axis (foot to top) and straight up (0..180 degrees: 180 upside down)."""
    return float(np.degrees(np.arccos(np.clip(rotation(s.quat_xyzw)[2, 2], -1.0, 1.0))))


def centre_xy(s) -> np.ndarray:
    """The stick's centre on the court, seen from above (the base frame's origin: the post's axis at the foot)."""
    return np.asarray(s.origin, dtype=float)[:2].copy()


def centre(s) -> np.ndarray:
    """The middle of the stick's axis, in the world (where its stillness is read)."""
    return to_world(s, L.CENTRE_LOCAL)


def standing(s) -> bool:
    """Upright (axis within 10 degrees of vertical, foot down) with its lowest point within 3 mm of the court."""
    return bool(tilt_deg(s) <= L.UPRIGHT_DEG + _EPS and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS)


def off_net_line(s) -> float:
    """Signed distance of the stick's centre from the net line along x (m; + beyond it, away from the arms)."""
    return float(centre_xy(s)[0] - L.NET_X)


def on_net_line(s) -> bool:
    return bool(abs(off_net_line(s)) <= L.NET_TOL + _EPS)


def off_itf(s) -> float:
    """Signed distance of the stick's centre from the ITF point across the court: |y| - 0.3353 (m; + outward)."""
    return float(abs(centre_xy(s)[1]) - L.STICK_Y)


def side_of(s) -> str | None:
    """Which side of the court the stick is on: 'plus' (y > 0), 'minus' (y < 0) or None (on the centre line)."""
    y = float(centre_xy(s)[1])
    return "plus" if y > 0.0 else ("minus" if y < 0.0 else None)


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


def reach(arm) -> float:
    """How far from this gripper's pad-tip point (horizontally) a stick's axis can be while a finger touches it, over
    every yaw, plus 5 mm: 94 mm fully open, 62 mm closed on the post (m)."""
    return L.release_reach(opening(arm))


def tip_offset(tips, s) -> tuple[float, float]:
    """For one pad-tip point: the horizontal distance to the stick's axis (taken at the tip's height) and the tip's
    height above the stick's highest point (m)."""
    p = np.asarray(tips, dtype=float).ravel()
    o = np.asarray(s.origin, dtype=float)
    u = rotation(s.quat_xyzw)[:, 2]
    u = u if u[2] >= 0.0 else -u
    on_axis = o + u * ((p[2] - o[2]) / u[2]) if u[2] > 0.1 else o
    return float(np.hypot(*(p[:2] - on_axis[:2]))), float(p[2] - highest(s))


def held(s, arms) -> bool:
    """Not let go: some gripper's pad-tip point lower than 10 mm above the stick's top with the stick's axis within
    that gripper's reach horizontally (a finger may be touching, pressing or propping it, whatever the gripper's yaw).
    Each arm is checked on its own: an unreadable arm is skipped, the other is still checked."""
    for side in readable_sides(arms):
        xy, above = tip_offset(arm_tips(arms, side), s)
        if xy < reach(arms[side]) and above < L.RELEASE_ABOVE:
            return True
    return False


# -- the score ---------------------------------------------------------------------------------------------------
def position_score(s) -> float:
    """0.5 at the ITF point, 0.25 elsewhere in its alley, 0 otherwise (the position alone, read across the court)."""
    a = abs(float(centre_xy(s)[1]))
    if abs(a - L.STICK_Y) <= L.ITF_TOL + _EPS:
        return 0.5
    if L.ALLEY[0] - _EPS <= a <= L.ALLEY[1] + _EPS:
        return 0.25
    return 0.0


def stick_score(s, arms=None, still: bool = True) -> tuple[float, str]:
    """(the stick's score, why it is 0 if it is). Set = let go, still, standing upright on the court, on the net
    line; then its position score."""
    if not standing(s):
        return 0.0, ("not_upright" if tilt_deg(s) > L.UPRIGHT_DEG else "not_on_the_court")
    if not on_net_line(s):
        return 0.0, "off_the_net_line"
    if held(s, arms):
        return 0.0, "still_held"
    if not still:
        return 0.0, "not_still_1s"
    p = position_score(s)
    return p, ("" if p > 0 else "outside_the_alley")


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(float(v)) else None


def evaluate(sticks, arms=None, still=None) -> tuple[float, dict]:
    """The score of the sticks as they stand, and what was found (JSON-clean). ``still``: one flag per stick (the
    tracker's), default all still. ``arms``: rigstate.arms() or None."""
    ss = list(sticks)
    still = [True] * len(ss) if still is None else list(still)
    sides = {"plus": 0.0, "minus": 0.0}
    per = {}
    for name, s, st in zip(L.NAMES, ss, still, strict=True):
        sc, why = stick_score(s, arms, st)
        side = side_of(s)
        if side is not None:
            sides[side] = max(sides[side], sc)
        xy = centre_xy(s)
        per[name] = {"score": sc, "zero_reason": why, "side": side, "x_mm": _mm(xy[0]), "y_mm": _mm(xy[1]),
                     "off_net_mm": _mm(off_net_line(s)), "off_itf_mm": _mm(off_itf(s)),
                     "tilt_deg": round(tilt_deg(s), 2), "bottom_mm": _mm(bottom_gap(s)), "held": held(s, arms),
                     "still": bool(st)}
    total = sides["plus"] + sides["minus"]
    return total, {"plus_side": sides["plus"], "minus_side": sides["minus"], "sticks": per}


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: the middle of each stick's axis over the last 1 s (for 'still'), and counters."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.prev = None  # [(origin, velocity)] of both sticks at the previous valid step
        self.history: deque = deque(maxlen=self.window * L.STILL_WINDOWS + 1)  # both centres, last 1 s
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(sticks) -> bool:
        """The starting layout: both sticks standing upright and slow on their own starting marks."""
        for s, name in zip(sticks, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            xy = centre_xy(s)
            on_mark = math.hypot(xy[0] - x0, xy[1] - y0) < L.START_TOL
            try:
                slow = float(np.linalg.norm(np.asarray(s.lin_vel, dtype=float))) < 0.01
            except (AttributeError, TypeError, ValueError):
                slow = True
            if not (on_mark and standing(s) and tilt_deg(s) < 2.0 and slow):
                return False
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for (p, _), (p0, v0) in zip(now, self.prev, strict=True))

    def window_paths(self) -> list[list[float]] | None:
        """For each stick, its axis midpoint's path (m) over each of the last two 0.5 s windows (older first), or None
        before 1 s of valid steps has been seen."""
        if len(self.history) < self.history.maxlen:
            return None
        h = list(self.history)
        out = []
        for i in range(len(h[0])):
            paths = []
            for w in range(L.STILL_WINDOWS):
                seg = h[w * self.window:(w + 1) * self.window + 1]
                paths.append(float(sum(np.linalg.norm(b[i] - a[i]) for a, b in zip(seg, seg[1:]))))
            out.append(paths)
        return out

    def still_flags(self) -> list[bool]:
        paths = self.window_paths()
        if paths is None:
            return [False] * len(L.NAMES)
        return [all(p < L.STILL_PATH for p in ps) for ps in paths]

    def update(self, sticks, arms=None) -> Result:
        ss = list(sticks)
        self.steps += 1
        if len(ss) != len(L.NAMES) or not all(valid(s) for s in ss):
            self.bad_steps += 1
            self.last = self.result(ss, arms)
            return self.last
        now = []
        for s in ss:
            try:
                v = np.asarray(s.lin_vel, dtype=float).ravel()
                v = v if v.size == 3 and np.all(np.isfinite(v)) else np.zeros(3)
            except (AttributeError, TypeError, ValueError):
                v = np.zeros(3)
            now.append((np.asarray(s.origin, dtype=float).copy(), v))
        if self.prev is not None and self._teleported(now) and self.at_start(ss):
            self.reset()  # the simulator put both sticks back on their marks: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([centre(s) for s in ss])
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(ss, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, sticks, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        ss = list(sticks)
        trial = {"steps": self.steps, "bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "arms_readable": list(readable_sides(arms)), "arms_unavailable": self.arms_unavailable}
        bad = [L.NAMES[i] for i, s in enumerate(ss[:len(L.NAMES)]) if not valid(s)]
        if bad or len(ss) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_sticks": bad, **trial})
        paths = self.window_paths()
        score, d = evaluate(ss, arms, self.still_flags())
        if paths is not None:
            for name, ps in zip(L.NAMES, paths, strict=True):
                d["sticks"][name]["path_mm"] = [round(p * 1000, 2) for p in ps]
        return Result(score, bool(score == 1.0), {"valid_state": True, **d, **trial})
