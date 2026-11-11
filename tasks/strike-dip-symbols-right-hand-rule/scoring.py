"""The scorer, as plain functions of the three tokens' states and the two grippers. NumPy only; no simulator needed.

The rule is about the finished map, read from the state at hand (the last control step), as a geologist checks a
plotted map with a protractor or a top photo: at each of the three stations A, B and C, is a strike-and-dip symbol
centred on the station's circle, lying flat on the map and let go; is its bar on the station's recorded strike; and
does its tick point the right-hand-rule dip direction (the strike plus 90 degrees, clockwise). A station earns 1/3
for the full symbol, 1/6 for the bar on the strike with the tick the other way, 0 otherwise; the task score is the
sum. A small tracker is fed once per control step by ``task.success()``: it keeps each token's last second of
positions (for 'still', which every credit needs), what the trace needs, and recognises a new episode when the
simulator puts every token back on its starting mark in one step.

    Tracker.update(states, arms) -> Result(score, success, details)

``states`` maps ``token_1``, ``token_2`` and ``token_3`` to what mechkit reports (``AssemblyState``): ``origin`` (the
middle of the bar's footprint on its base plane) and ``quat_xyzw`` (its base frame: local +x the way the tick
points, the bar along local y, local +z up), with the ObjectState fields. A state with a non-finite number, a zero
quaternion or a token without ``origin`` (a simulator blow-up, or a token not built as an assembly) scores 0 for that
step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the pad-tip point, 13 mm past the
tool point) and ``opening`` (0 closed .. 1 open), or None. An arm that cannot be read is skipped (the let-go test is
then taken as met for it) and counted in the trace.

The three tokens are identical: any token may serve any station. At each station the token that counts is the one
nearest the station's centre among those that qualify there.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

NAMES = tuple(L.TOKENS)  # ("token_1", "token_2", "token_3")
STATIONS = tuple(L.STATIONS)  # ("A", "B", "C")
SIDES = ("left", "right")
_EPS = 1e-9

# the token's two boxes in its base frame: (centre, half extents)
_BOXES = ((np.array(L.BAR_CENTER), np.array(L.BAR) / 2), (np.array(L.TICK_CENTER), np.array(L.TICK) / 2))
_CORNERS = np.array([c + h * np.array([sx, sy, sz]) for c, h in _BOXES
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])  # (16, 3)
_TIP = np.array(L.TICK_TIP_LOCAL)


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w), normalised; its columns are the body's axes in the world.
    (Callers check ``valid_token`` first: a zero quaternion is not a rotation.)"""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def valid_token(t) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(t, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid(states) -> bool:
    try:
        return all(valid_token(states[n]) for n in NAMES)
    except (KeyError, TypeError):
        return False


# -- one token ---------------------------------------------------------------------------------------------------------


def corners(t) -> np.ndarray:
    """The corners of the token's two boxes (bar and tick) in the world, (16, 3)."""
    return np.asarray(t.origin, dtype=float) + _CORNERS @ rotation(t.quat_xyzw).T


def tick_tip(t) -> np.ndarray:
    """The middle of the tick's end face in the world (35 mm from the bar's centre along the tick)."""
    return np.asarray(t.origin, dtype=float) + rotation(t.quat_xyzw) @ _TIP


def tilt_deg(t) -> float:
    """How far the token's up axis is from vertical (0 lying flat, 90 on its side or end, 180 upside down)."""
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(rotation(t.quat_xyzw)[2, 2]))))))


def bottom_gap(t) -> float:
    """Height of the token's lowest point above the map's surface (m; negative = below it)."""
    return float(corners(t)[:, 2].min()) - L.MAP_TOP_W


def top_z(t) -> float:
    """World height of the token's highest point."""
    return float(corners(t)[:, 2].max())


def tick_yaw_deg(t) -> float | None:
    """The way the tick points, seen from above (degrees, -180..180, counter-clockwise from +x, north); None when
    the token stands on its end (local x vertical)."""
    a = rotation(t.quat_xyzw)[:, 0]
    if math.hypot(a[0], a[1]) < 1e-6:
        return None
    return math.degrees(math.atan2(float(a[1]), float(a[0])))


def bar_yaw_deg(t) -> float | None:
    """The bar's axis seen from above (degrees, -90..90: a line, not a direction): the tick's yaw plus 90."""
    k = tick_yaw_deg(t)
    if k is None:
        return None
    return math.degrees(math.remainder(math.radians(k + 90.0), math.pi))


def line_err_deg(a: float, b: float) -> float:
    """Angle between two lines (degrees, 0..90): 180 degrees apart is the same line."""
    return abs(math.degrees(math.remainder(math.radians(a - b), math.pi)))


def dir_err_deg(a: float, b: float) -> float:
    """Angle between two directions (degrees, 0..180)."""
    return abs(math.degrees(math.remainder(math.radians(a - b), 2 * math.pi)))


def lying(t) -> bool:
    """Lying flat on the map: tilted under 5 degrees and its lowest point within 3 mm of the map's surface (so a
    token resting on another token, held up, tipped or upside down does not lie on the map)."""
    return bool(tilt_deg(t) < L.FLAT_DEG and abs(bottom_gap(t)) <= L.ON_MAP_TOL + _EPS)


def nearest_station(t) -> tuple[str, float]:
    """The station whose centre is nearest the bar's centre, seen from above, and that distance (m)."""
    o = np.asarray(t.origin, dtype=float)
    d = {s: math.hypot(o[0] - x, o[1] - y) for s, (x, y) in L.STATIONS.items()}
    s = min(STATIONS, key=lambda k: d[k])
    return s, d[s]


def errors(t, station: str) -> tuple[float, float]:
    """(strike error, tick error) in degrees of the token for that station: the bar's axis against the recorded
    strike (mod 180, 0..90) and the tick's direction against the right-hand-rule dip direction (0..180)."""
    k = tick_yaw_deg(t)
    if k is None:
        return 90.0, 180.0
    return line_err_deg(k + 90.0, L.STRIKE_YAW_DEG[station]), dir_err_deg(k, L.DIP_YAW_DEG[station])


def symbol_sixths(t, station: str) -> int:
    """A qualifying token's credit at a station, in sixths of the task score: 2 (1/3) with the tick within 6 degrees
    of the dip direction (the bar is then within 6 degrees of the strike too); 1 (1/6) with the bar within 6 degrees
    of the strike but the tick the other way; 0 with the bar off the strike."""
    strike, tick = errors(t, station)
    if tick <= L.TICK_TOL_DEG + _EPS:
        return L.CREDIT_SIXTHS["symbol"]
    if strike <= L.STRIKE_TOL_DEG + _EPS:
        return L.CREDIT_SIXTHS["strike_only"]
    return 0


# -- let go: the arms (rigstate) -----------------------------------------------------------------------------------------


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


def outline_distance(t, point) -> float:
    """Horizontal distance (m) from a point to the token's outline (its bar and tick seen from above, in its own
    frame); 0 when the point is over the token."""
    local = (np.asarray(point, dtype=float).ravel() - np.asarray(t.origin, dtype=float)) @ rotation(t.quat_xyzw)
    best = math.inf
    for c, h in _BOXES:
        dx = max(abs(float(local[0] - c[0])) - h[0], 0.0)
        dy = max(abs(float(local[1] - c[1])) - h[1], 0.0)
        best = min(best, math.hypot(dx, dy))
    return best


def held(t, arms) -> bool:
    """Not let go: some readable gripper's pad-tip point lower than 10 mm above the token's top with the token's
    outline within that gripper's finger reach horizontally (a finger may be touching, pressing or propping it,
    whatever the gripper's yaw)."""
    for side in readable_sides(arms):
        tips = arm_tips(arms, side)
        if (outline_distance(t, tips) < L.release_reach(opening(arms[side]))
                and float(tips[2]) - top_z(t) < L.RELEASE_ABOVE):
            return True
    return False


# -- the map -----------------------------------------------------------------------------------------------------------


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


def token_pose(t, arms, still: bool) -> dict:
    """What a protractor, a rule and a look from the side tell about one token, and whether it qualifies at its
    nearest station (lying flat on the map, let go, still, its bar's centre within 8 mm of the station's centre)."""
    s, off = nearest_station(t)
    strike, tick = errors(t, s)
    k, b = tick_yaw_deg(t), bar_yaw_deg(t)
    lie, h = lying(t), held(t, arms)
    qualifies = bool(lie and not h and still and off <= L.CENTRE_TOL + _EPS)
    o = np.asarray(t.origin, dtype=float)
    return {"origin_mm": [round(float(v) * 1000, 1) for v in o], "tick_yaw_deg": None if k is None else round(k, 2),
            "bar_yaw_deg": None if b is None else round(b, 2), "tilt_deg": round(tilt_deg(t), 2),
            "bottom_mm": _mm(bottom_gap(t)), "lying": lie, "held": h, "still": bool(still), "station": s,
            "off_station_mm": _mm(off), "strike_err_deg": round(strike, 2), "tick_err_deg": round(tick, 2),
            "qualifies": qualifies}


def evaluate(states, arms=None, still: dict | None = None) -> tuple[float, dict]:
    """The score of the map as it stands, given which tokens are still, and what was found (JSON-clean).

    At each station the token that counts is the nearest to its centre among the tokens that qualify there; it
    earns 1/3 (bar on the strike and tick down-dip by the right-hand rule), 1/6 (bar on the strike, tick the other
    way) or 0. Score = the sum over A, B and C, in sixths: 0, 1/6, ..., 1.0."""
    still = still or {}
    d: dict = {n: token_pose(states[n], arms, bool(still.get(n, False))) for n in NAMES}
    sixths, why = 0, []
    for s in STATIONS:
        here = [n for n in NAMES if d[n]["qualifies"] and d[n]["station"] == s]
        best = min(here, key=lambda n: (nearest_station(states[n])[1], n)) if here else None
        got = symbol_sixths(states[best], s) if best else 0
        d[s] = {"token": best, "candidates": here, "sixths": got}
        sixths += got
        if best is None:
            why.append(f"{s}_no_symbol")
        elif got == 0:
            why.append(f"{s}_bar_off_the_strike")
        elif got < L.CREDIT_SIXTHS["symbol"]:
            why.append(f"{s}_tick_not_down_dip")
    d["sixths"] = sixths
    d["zero_reason"] = ",".join(why)
    return sixths / 6.0, d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each token's bar centre and tick tip over the last second, and
    diagnostics for the trace (the score itself depends only on the state at hand and that last second)."""

    TIERS = tuple(k / 6.0 for k in range(1, 7))

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.arms_unavailable = 0
        self.first_at: dict[str, int] = {}
        self.best = 0.0
        self.prev = None  # every token's origin at the previous valid step
        self.history: deque = deque(maxlen=self.window * L.STILL_WINDOWS + 1)
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: every token lying flat on the map on its starting mark, turned as it starts."""
        for n, (x, y, yaw) in L.TOKENS.items():
            t = states[n]
            o = np.asarray(t.origin, dtype=float)
            if math.hypot(o[0] - x, o[1] - y) > L.START_TOL or not lying(t):
                return False
            k = tick_yaw_deg(t)
            if k is None or dir_err_deg(k, yaw) > 5.0:
                return False
        return True

    def paths(self) -> dict | None:
        """Per token, the longer of its bar centre's and tick tip's paths (m) in each of the last two 0.5 s
        windows; None until 1 s of valid steps has been seen."""
        h = list(self.history)
        if len(h) < self.history.maxlen:
            return None
        out = {}
        for n in NAMES:
            per = []
            for w in range(L.STILL_WINDOWS):
                seg = h[w * self.window:(w + 1) * self.window + 1]
                per.append(max(sum(float(np.linalg.norm(b[n][i] - a[n][i])) for a, b in zip(seg, seg[1:]))
                               for i in (0, 1)))
            out[n] = per
        return out

    def update(self, states, arms=None) -> Result:
        if not valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(states, arms)
            return self.last
        now = {n: np.asarray(states[n].origin, dtype=float) for n in NAMES}
        if self.prev is not None and self.at_start(states):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in NAMES)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        self.history.append({n: (now[n], tick_tip(states[n])) for n in NAMES})
        self.prev = now
        self.last = self.result(states, arms)
        for k, tier in enumerate(self.TIERS, start=1):
            if self.last.details.get("sixths", 0) >= k and str(k) not in self.first_at:
                self.first_at[str(k)] = self.steps
        self.best = max(self.best, self.last.score)
        return self.last

    def result(self, states, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        trial = {"steps": self.steps, "bad_steps": self.bad_steps, "arms_readable": list(readable_sides(arms)),
                 "arms_unavailable": self.arms_unavailable, "first_step_sixths": dict(self.first_at),
                 "best_score": round(self.best, 4)}
        if not valid(states):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state", **trial})
        paths = self.paths()
        still = {n: bool(paths is not None and all(p < L.STILL_PATH for p in paths[n])) for n in NAMES}
        score, d = evaluate(states, arms, still)
        details = {"valid_state": True, **d, **trial,
                   "path_mm": None if paths is None else {n: [round(p * 1000, 2) for p in v] for n, v in paths.items()}}
        return Result(score, bool(d["sixths"] == 6), details)
