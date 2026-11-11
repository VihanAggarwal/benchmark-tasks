"""The scorer, as plain functions of the two flanges' states and the grippers' state. NumPy only; no simulator needed.

The score is read from where the flanges end, as a fitter checks a fit-up before tacking, with a rule, a square on
the painted centrelines and a look from the side (or a photo from above): is each flange on the stub it is bored for,
its neck resting on the stub end, centred on the stub's axis and level; and is its bolt-hole pattern two-holed, the
holes straddling the plant centrelines instead of sitting on them. Every credit also needs the flange let go and
still. The scorer keeps only the last second of each flange's positions (to tell a flange at rest from one still
moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(states, arms) -> Result(score, success, details)

``states`` maps ``flange_4`` and ``flange_8`` to what mechkit reports (``AssemblyState``): ``origin`` (its base
frame's origin, on the flange's axis at the bottom of its neck) and ``quat_xyzw`` (that frame: local +z up the axis,
face up; local +x through the centre of bolt hole 1), with the ObjectState fields. A state with a non-finite number,
a zero quaternion or a flange without ``origin`` (a simulator blow-up, or a flange not built as an assembly) scores
0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tips`` (the point between the pad tips) and
``opening`` (0 closed .. 1 open), or ``None``. An arm that cannot be read is skipped (taken as not holding anything)
and counted in ``arms_unavailable``; the other arm is still checked.

With s the number of seated flanges and t the number of those that are two-holed: score = 0.25 s + 0.25 t.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

FLANGES = tuple(L.FLANGES)  # ("flange_4", "flange_8")
NAMES = FLANGES
SIDES = ("left", "right")


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


def _finite(obj, keys) -> list | None:
    try:
        parts = [np.asarray(getattr(obj, k), dtype=float).ravel() for k in keys]
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return None
    return parts


def valid_flange(f) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    parts = _finite(f, ("origin", "quat_xyzw", "pos"))
    return bool(parts and parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def valid(states) -> bool:
    try:
        return all(valid_flange(states[n]) for n in FLANGES)
    except (KeyError, TypeError):
        return False


# -- one flange -------------------------------------------------------------------------------------------------------


def to_world(f, local) -> np.ndarray:
    return np.asarray(f.origin, dtype=float) + rotation(f.quat_xyzw) @ np.asarray(local, dtype=float)


def hole_point(f, name: str) -> np.ndarray:
    """The centre of bolt hole 1 on the flange's face, in the world."""
    return to_world(f, (L.FLANGES[name]["bc_r"], 0.0, L.FACE_Z))


def track_points(f, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(the neck's bottom centre, the centre of hole 1): two points whose paths show any slide or turn."""
    return np.asarray(f.origin, dtype=float).copy(), hole_point(f, name)


def tilt_deg(f) -> float:
    """How far the flange's axis is from vertical, face up (0 neck down, 90 on its rim, 180 face down)."""
    return float(np.degrees(math.acos(max(-1.0, min(1.0, float(rotation(f.quat_xyzw)[2, 2]))))))


def pattern_yaw_deg(f) -> float | None:
    """The hole pattern's heading seen from above: the direction from the axis to hole 1 (degrees, 0 = +x, plant
    north; counter-clockwise positive); None when the flange stands on its rim with hole 1 straight up or down."""
    a = rotation(f.quat_xyzw)[:, 0]
    if math.hypot(a[0], a[1]) < 1e-6:
        return None
    return math.degrees(math.atan2(float(a[1]), float(a[0])))


def straddle_off_deg(name: str, yaw_deg: float | None) -> float:
    """How far the hole pattern is from straddling the centrelines (degrees): 0 when the holes sit exactly half a
    pitch off the lines (45 degrees for 4 holes, 22.5 for 8), half the pitch when a hole is on a line. The pattern
    repeats every pitch, and the N-S and E-W lines are a quarter turn apart, so the angle is read modulo the pitch."""
    if yaw_deg is None:
        return L.two_hole_deg(name)
    p = L.pitch_deg(name)
    return abs((yaw_deg % p) - p / 2.0)


def centre_offset(f, stub: str) -> float:
    """Distance seen from above between the flange's axis (at the neck's bottom) and the stub's axis (m)."""
    sx, sy, _ = L.STUBS[stub]
    o = np.asarray(f.origin, dtype=float)
    return float(math.hypot(o[0] - sx, o[1] - sy))


def seat_gap(f) -> float:
    """The neck's bottom centre above the stub end's height (m; negative: below it)."""
    return float(np.asarray(f.origin, dtype=float)[2] - L.STUB_END_W)


def highest(f, name: str) -> float:
    """The flange's highest point (world z): the centre of its painted face plus the rim's rise when tilted."""
    r = rotation(f.quat_xyzw)
    c = to_world(f, (0.0, 0.0, L.TOP_Z))
    return float(c[2] + L.FLANGES[name]["disc_r"] * math.sqrt(max(0.0, 1.0 - float(r[2, 2]) ** 2)))


def nearest_stub(f) -> tuple[str, float]:
    o = np.asarray(f.origin, dtype=float)
    best = min(L.STUBS, key=lambda s: math.hypot(o[0] - L.STUBS[s][0], o[1] - L.STUBS[s][1]))
    return best, centre_offset(f, best)


def on_stub_pose(f, stub: str) -> bool:
    """Resting on that stub's end as a fitter sets it: axis within 6 mm of the stub's, neck's bottom within 3 mm of
    the stub end's height, axis within 5 degrees of vertical, face up (no stillness or release test)."""
    return bool(centre_offset(f, stub) <= L.CENTRE_TOL and abs(seat_gap(f)) <= L.SEAT_TOL
                and tilt_deg(f) <= L.TILT_MAX_DEG)


# -- the grippers -------------------------------------------------------------------------------------------------


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


def tip_offset(tips, f, name: str) -> tuple[float, float]:
    """For one pad-tip point: the horizontal distance to the flange's axis and the tip's height above the flange's
    highest point (m)."""
    p = np.asarray(tips, dtype=float).ravel()
    o = np.asarray(f.origin, dtype=float)
    return float(math.hypot(p[0] - o[0], p[1] - o[1])), float(p[2] - highest(f, name))


def held_by(f, name: str, arms) -> list[str]:
    """The grippers that may be holding, pressing or propping the flange: pad-tip point lower than 10 mm above its
    highest point while within a finger's reach of it horizontally, at any yaw."""
    out = []
    for side in readable_sides(arms):
        xy, above = tip_offset(arm_tips(arms, side), f, name)
        if xy < L.release_reach(opening(arms[side]), L.FLANGES[name]["disc_r"]) and above < L.RELEASE_ABOVE:
            out.append(side)
    return out


# -- the tiers -------------------------------------------------------------------------------------------------------


def flange_pose(f, name: str) -> dict:
    """What a rule, a square and a look from the side tell about one flange (no rest or release test)."""
    own = L.FLANGES[name]["stub"]
    near, near_off = nearest_stub(f)
    yaw = pattern_yaw_deg(f)
    off = straddle_off_deg(name, yaw)
    return {"own_stub": own, "nearest_stub": near, "nearest_offset_mm": round(near_off * 1000, 2),
            "centre_mm": round(centre_offset(f, own) * 1000, 2), "seat_gap_mm": round(seat_gap(f) * 1000, 2),
            "tilt_deg": round(tilt_deg(f), 2), "pattern_yaw_deg": None if yaw is None else round(yaw, 2),
            "straddle_off_deg": round(off, 2), "on_own_stub": on_stub_pose(f, own),
            "on_other_stub": any(on_stub_pose(f, s) for s in L.STUBS if s != own),
            "pattern_ok": bool(off <= L.TWO_HOLE_TOL_DEG[name])}


def evaluate(states, rest: dict, arms=None) -> tuple[float, str, dict]:
    """(score, why it is not higher, diagnostics) of a valid state, given which flanges are at rest.

    A flange is seated when it is on its own stub (``on_stub_pose``), still and let go; it is two-holed when it is
    seated and its hole pattern is within 8 (4 holes) or 7 (8 holes) degrees of straddling the centrelines.
    Score = 0.25 per seated flange + 0.25 per two-holed flange."""
    d: dict = {}
    s = t = 0
    why = []
    for n in FLANGES:
        p = flange_pose(states[n], n)
        p["held_by"] = held_by(states[n], n, arms)
        p["at_rest"] = bool(rest.get(n, False))
        p["seated"] = bool(p["on_own_stub"] and p["at_rest"] and not p["held_by"])
        p["two_holed"] = bool(p["seated"] and p["pattern_ok"])
        d[n] = p
        s += p["seated"]
        t += p["two_holed"]
        if p["on_own_stub"] and p["held_by"]:
            why.append(f"{n}_still_held")
        elif p["on_own_stub"] and not p["at_rest"]:
            why.append(f"{n}_still_moving")
        elif p["on_other_stub"]:
            why.append(f"{n}_on_the_wrong_stub")
        elif not p["seated"]:
            why.append(f"{n}_not_seated")
        elif not p["two_holed"]:
            why.append(f"{n}_holes_not_straddling")
    d["seated_count"], d["two_holed_count"] = s, t
    return round(L.SEATED_CREDIT * s + L.TWO_HOLED_CREDIT * t, 2), ",".join(why), d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: each flange's two tracked points over the last second, and how
    many arm readings were unavailable."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {n: deque(maxlen=self.window + 1) for n in NAMES}
        self.prev = None  # every flange's origin at the previous valid step
        self.arms_unavailable = 0
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(states) -> bool:
        """The starting layout: both flanges neck down on the table on their starting marks, holes on the lines."""
        for n, spec in L.FLANGES.items():
            f = states[n]
            o = np.asarray(f.origin, dtype=float)
            if math.hypot(o[0] - spec["start"][0], o[1] - spec["start"][1]) > L.START_TOL:
                return False
            if abs(float(o[2]) - L.TOP) > 0.003 or tilt_deg(f) > 2.0:
                return False
            if straddle_off_deg(n, pattern_yaw_deg(f)) < L.two_hole_deg(n) - 5.0:
                return False
        return True

    def moved(self, name: str) -> float | None:
        """The farthest the flange's two tracked points have been, over the last 1 s, from where they are now (m);
        None until 1 s of this trial has been seen."""
        h = self.history[name]
        if len(h) < self.window + 1:
            return None
        p, q = h[-1]
        return float(max(max(np.linalg.norm(a - p), np.linalg.norm(b - q)) for a, b in h))

    def update(self, states, arms=None) -> Result:
        if not valid(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(states, arms)
            return self.last
        now = {n: np.asarray(states[n].origin, dtype=float) for n in NAMES}
        if self.prev is not None and self.at_start(states):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in NAMES)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode
                self.reset()
        self.steps += 1
        for n in NAMES:
            self.history[n].append(track_points(states[n], n))
        self.prev = now
        if len(readable_sides(arms)) < len(SIDES):
            self.arms_unavailable += 1
        self.last = self.result(states, arms)
        return self.last

    def result(self, states, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not valid(states):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        moved = {n: self.moved(n) for n in NAMES}
        rest = {n: (m is not None and m <= L.STILL_TOL) for n, m in moved.items()}
        score, reason, d = evaluate(states, rest, arms)
        details = {"valid_state": True, "zero_reason": reason, "arms_unavailable": self.arms_unavailable,
                   "readable_arms": list(readable_sides(arms)),
                   "moved_mm": {n: (None if m is None else round(m * 1000, 2)) for n, m in moved.items()}, **d}
        return Result(score, bool(score == 1.0), details)
