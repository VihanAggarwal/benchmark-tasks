"""The scorer, as plain functions of the nine pucks' states. NumPy only; no simulator needed.

The rule (WPBSA Official Rules, Section 3 Rule 7, and Section 2 Rule 18): a colour is re-spotted on its own spot
if that is free, otherwise on the highest-value spot available, and a spotted colour must not touch another ball.
In this layout the black's own spot and the pink spot hold reds, so the black belongs on the blue spot; the
yellow's own spot is free. Re-spotting must not move any ball already on the table.

    Tracker.update(pucks, arms) -> Result(score, success, details)

``pucks`` (9, in the order of ``layout.NAMES``: the black, the yellow, then the seven table pucks) are anything
with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has. Each is an upright cylinder (a disc) whose axis is the body's z axis; its
two faces are alike. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that
step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point, 13 mm past the
TCP, midway between the finger tips) and ``opening`` (0 closed .. 1 fully open), or ``None`` when the arms could not
be read. An arm that cannot be read is skipped by the release rule (the other is still checked) and the step is
counted in ``arms_unavailable``. Offline, and in these functions' defaults, ``arms=None``.

Scoring, decided in this order:
1. Latched: if any of the seven table pucks has at any step been more than 8 mm (3-D, centre) from where it stood
   at the start of the trial, the score is 0 for the rest of the trial, even if it is put back.
2. Otherwise score = black part + yellow part. A colour 'counts on' a spot when its centre is within 8 mm of the
   spot (seen from above), it lies flat on the bed (tilt under 10 degrees, lowest point within 3 mm of the bed
   top), no other puck's centre is within 31 mm of its centre (seen from above: not touching, not stacked), it is
   released (no gripper's pad-tip point lower than 10 mm above its top within that gripper's finger reach) and it
   is still (its centre's path under 1.5 mm in each of the last two 0.5 s windows).
   Black part: 0.5 counting on the blue spot, 0.25 counting on any other spot, else 0.
   Yellow part: 0.5 counting on the yellow spot, else 0.
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


def tilt_deg(quat_xyzw) -> float:
    """Angle between the puck's axis and the vertical (degrees, 0..90; either face up)."""
    zz = abs(float(rotation(quat_xyzw)[2, 2]))
    return float(np.degrees(np.arccos(np.clip(zz, 0.0, 1.0))))


def _half_extent_z(s) -> float:
    az = abs(float(rotation(s.quat_xyzw)[2, 2]))
    return L.PUCK_H / 2 * az + L.PUCK_R * math.sqrt(max(0.0, 1.0 - az * az))


def lowest(s) -> float:
    """Height (world z) of the puck's lowest point, from its pose."""
    return float(s.pos[2]) - _half_extent_z(s)


def highest(s) -> float:
    """Height (world z) of the puck's highest point, from its pose."""
    return float(s.pos[2]) + _half_extent_z(s)


def bottom_gap(s) -> float:
    """Height of the puck's lowest point above the bed top (m; negative = below it)."""
    return lowest(s) - L.BED_TOP


def over_bed(s) -> bool:
    """Its centre, seen from above, is over the playing bed."""
    x, y = float(s.pos[0]), float(s.pos[1])
    return bool(L.BED_X0 - _EPS <= x <= L.BED_X1 + _EPS and L.BED_Y0 - _EPS <= y <= L.BED_Y1 + _EPS)


def flat_on_bed(s) -> bool:
    """Flat (tilt under 10 degrees), its lowest point within 3 mm of the bed top, its centre over the bed."""
    return bool(tilt_deg(s.quat_xyzw) < L.FLAT_DEG and abs(bottom_gap(s)) <= L.SEAT_TOL + _EPS and over_bed(s))


def xy(s) -> np.ndarray:
    return np.asarray(s.pos, dtype=float)[:2]


def nearest_spot(s) -> tuple[str, float]:
    """The spot nearest the puck's centre (seen from above) and the distance to it (m)."""
    p = xy(s)
    name = min(L.SPOTS, key=lambda k: float(np.hypot(*(p - np.array(L.SPOTS[k])))))
    return name, float(np.hypot(*(p - np.array(L.SPOTS[name]))))


def nearest_other(pucks, i: int) -> tuple[str | None, float]:
    """The puck whose centre is nearest puck i's centre, seen from above, and that distance (m)."""
    best, dist = None, math.inf
    for j, s in enumerate(pucks):
        if j != i:
            d = float(np.hypot(*(xy(s) - xy(pucks[i]))))
            if d < dist:
                best, dist = L.NAMES[j], d
    return best, dist


# -- released: the arms (rigstate) -------------------------------------------------------------------------------
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
    """How far from this gripper's pad-tip point (horizontally) a puck's axis can be while a finger touches it, over
    every yaw, plus 5 mm: 80 mm fully open (m)."""
    return L.release_reach(opening(arm))


def held(s, arms) -> bool:
    """Not released: some gripper's pad-tip point lower than 10 mm above the puck's top with the puck's centre
    within that gripper's reach horizontally. Each arm is checked on its own; an unreadable arm is skipped."""
    for side in readable_sides(arms):
        t = arm_tips(arms, side)
        d = float(np.hypot(*(t[:2] - xy(s))))
        if d < reach(arms[side]) and float(t[2]) - highest(s) < L.RELEASE_ABOVE:
            return True
    return False


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- the score ---------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and np.isfinite(v) else None


def colour_on(pucks, i: int, arms=None, still: bool = True) -> tuple[str | None, dict]:
    """The spot colour i counts on (or None) and why, for the trace."""
    s = pucks[i]
    spot, off = nearest_spot(s)
    other, gap = nearest_other(pucks, i)
    d = {"nearest_spot": spot, "off_spot_mm": _mm(off), "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
         "bottom_mm": _mm(bottom_gap(s)), "nearest_puck": other, "nearest_puck_mm": _mm(gap),
         "held": held(s, arms), "still": bool(still)}
    ok = (off <= L.SPOT_TOL + _EPS and flat_on_bed(s) and gap > L.TOUCH + _EPS and not d["held"] and still)
    d["on_spot"] = spot if ok else None
    return d["on_spot"], d


def parts(black_spot: str | None, yellow_spot: str | None) -> tuple[float, float]:
    """The black part and the yellow part for the spots they count on (None: on no spot)."""
    black = 0.0 if black_spot is None else (0.5 if black_spot == L.BLACK_FULL else L.BLACK_PART)
    yellow = 0.5 if yellow_spot == "yellow" else 0.0
    return black, yellow


def evaluate(pucks, arms=None, still=None, latched: bool = False) -> tuple[float, dict]:
    """The score of the table as it stands, and what was found (JSON-clean). ``still``: {name: bool} for the two
    colours (None: both taken as still, a single photo); ``latched``: a table puck was disturbed earlier."""
    ps = list(pucks)
    still = {} if still is None else still
    bs, bd = colour_on(ps, 0, arms, still.get("black", True))
    ys, yd = colour_on(ps, 1, arms, still.get("yellow", True))
    black, yellow = parts(bs, ys)
    d = {"latched": bool(latched), "black_part": black, "yellow_part": yellow, "black": bd, "yellow": yd}
    if latched:
        return 0.0, d
    return black + yellow, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the score needs beyond the state at hand (the disturbance latch, the stillness windows) and what the
    trace needs (when each score was first reached, broken observations, unreadable arms). A new episode is
    recognised when rigstate sees the embodiment reset or the simulator puts every puck back on its mark."""

    TIERS = (0.25, 0.5, 0.75, 1.0)

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, round(L.STILL_WINDOW_S * control_hz))  # control steps in 0.5 s
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.arms_unavailable_by_side = {side: 0 for side in SIDES}
        self.first_at: dict[float, int] = {}
        self.best = 0.0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of every puck at the previous valid control step
        self.history: deque = deque(maxlen=L.STILL_WINDOWS * self.window + 1)
        self.ref: dict[str, np.ndarray] | None = None  # the table pucks' centres at the first valid step
        self.max_disp = {n: 0.0 for n in L.TABLE}
        self.latched = False
        self.latched_at: int | None = None
        self.latched_by: str | None = None

    @staticmethod
    def at_start(pucks) -> bool:
        """The starting layout: every puck still and flat on its own mark."""
        for s, name in zip(pucks, L.NAMES, strict=True):
            x0, y0 = L.START[name]
            on_mark = math.hypot(float(s.pos[0]) - x0, float(s.pos[1]) - y0) < L.START_TOL
            seated = abs(lowest(s) - L.start_bottom(name)) < L.SEAT_TOL
            if not (on_mark and seated and tilt_deg(s.quat_xyzw) < 2.0 and is_resting(s)):
                return False
        return True

    def _teleported(self, now) -> bool:
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev, strict=True)
        )

    def window_paths(self) -> list[list[float]] | None:
        """For each puck, its centre's path (m) in each of the last two 0.5 s windows (oldest first), or None
        before 1 s of valid steps has been seen."""
        if len(self.history) < self.history.maxlen:
            return None
        h = list(self.history)
        out = []
        for i in range(len(h[0])):
            steps = [float(np.linalg.norm(b[i] - a[i])) for a, b in zip(h, h[1:])]
            out.append([sum(steps[k * self.window:(k + 1) * self.window]) for k in range(L.STILL_WINDOWS)])
        return out

    def still(self) -> dict[str, bool]:
        paths = self.window_paths()
        return {n: paths is not None and all(p < L.STILL_PATH for p in paths[i]) for i, n in enumerate(L.NAMES)}

    def references(self) -> dict[str, np.ndarray]:
        """Where each table puck stood at the start: its centre at the first valid step of the episode (the layout's
        marks before any step)."""
        if self.ref is not None:
            return self.ref
        return {n: np.array([*L.START[n], L.start_bottom(n) + L.PUCK_H / 2]) for n in L.TABLE}

    def update(self, pucks, arms=None) -> Result:
        ps = list(pucks)
        self.steps += 1
        if len(ps) != len(L.NAMES) or not all(valid(s) for s in ps):  # a blow-up: 0 now, nothing changes
            self.bad_steps += 1
            self.last = self.result(ps, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in ps]
        if self.prev is not None and self._teleported(now) and self.at_start(ps):
            self.reset()  # the simulator put every puck back on its mark: a new episode
            self.steps = 1
        self.prev = now
        self.history.append([p for p, _ in now])
        if self.ref is None:
            self.ref = {n: now[L.NAMES.index(n)][0].copy() for n in L.TABLE}
        for n in L.TABLE:
            disp = float(np.linalg.norm(now[L.NAMES.index(n)][0] - self.ref[n]))
            self.max_disp[n] = max(self.max_disp[n], disp)
            if disp > L.DISTURB + _EPS and not self.latched:
                self.latched, self.latched_at, self.latched_by = True, self.steps, n
        readable = readable_sides(arms)
        for side in SIDES:
            if side not in readable:
                self.arms_unavailable_by_side[side] += 1
        if len(readable) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(ps, arms)
        for tier in self.TIERS:
            if self.last.score >= tier - _EPS and tier not in self.first_at:
                self.first_at[tier] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, pucks, arms=None) -> Result:
        ps = list(pucks)
        bad = [L.NAMES[i] for i, s in enumerate(ps) if i < len(L.NAMES) and not valid(s)]
        readable = readable_sides(arms)
        trial = {"bad_steps": self.bad_steps, "best_score": self.best,
                 "first_step": {str(k): v for k, v in sorted(self.first_at.items())},
                 "latched_at": self.latched_at, "latched_by": self.latched_by,
                 "max_disp_mm": {n: _mm(v) for n, v in self.max_disp.items()},
                 "arms_readable": list(readable),
                 "arms_unavailable": self.arms_unavailable,
                 "arms_unavailable_by_side": dict(self.arms_unavailable_by_side)}
        if bad or len(ps) != len(L.NAMES):
            return Result(0.0, False, {"valid_state": False, "bad_pucks": bad, **trial})
        # The latch also looks at the state at hand, so score() between updates cannot miss a disturbance.
        ref = self.references()
        disturbed_now = [n for n in L.TABLE
                         if float(np.linalg.norm(np.asarray(ps[L.NAMES.index(n)].pos, float) - ref[n]))
                         > L.DISTURB + _EPS]
        still = self.still()
        score, details = evaluate(ps, arms, still, self.latched or bool(disturbed_now))
        success = bool(score == 1.0)
        return Result(score, success, {"valid_state": True, "disturbed_now": disturbed_now, **details, **trial})
