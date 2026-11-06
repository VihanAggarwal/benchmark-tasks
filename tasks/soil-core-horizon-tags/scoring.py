"""The scorer, as plain functions of the four tags' states and the arms' state. NumPy only; no simulator needed.

A logged core is checked at the end: a ruler from each standing tag to the nearest colour change. So the score is
an end state (how many of the four boundaries have a placed tag at them), plus two facts that need a short memory:
whether each tag has been still for the last 0.5 s (its centre's path, never its velocity, which chatters while a
part is gripped) and whether a gripper has let it go. The tracker is fed one observation per control step by
``task.success()``.

    Tracker.update(tags, arms) -> Result(score, success, details)

``tags`` maps each tag's name to anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has.
``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the pad-tip point, 13 mm
past the TCP, midway between the finger tips) and ``opening`` (0 closed .. 1 fully open), or ``None`` when the arms
could not be read; then the release rule is taken as met for that step and the step is counted in
``arms_unavailable``. rigstate does not report a gripper's yaw, so the release rule takes the fingers' widest
reach over every yaw at the gripper's current opening (``layout.release_reach``). A step on which any tag's state
has a non-finite number or a zero quaternion (a simulator blow-up) scores 0 and leaves the tracker as it was.

A tag is a plain cylinder whose two ends are alike, so either end may be down: 'tilt' is measured from whichever
end is nearer to facing down.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L


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
        parts = [np.asarray(getattr(s, k), dtype=float).ravel()
                 for k in ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def arms_valid(arms) -> bool:
    """Both arms present with a finite pad-tip point (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            tips = np.asarray(arms[side].tips, dtype=float).ravel()
            if tips.size != 3 or not np.all(np.isfinite(tips)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def axis(tag) -> np.ndarray:
    """The tag's axis (unit vector), pointing up whichever end is down."""
    a = rotation(tag.quat_xyzw)[:, 2]
    return a if a[2] >= 0.0 else -a


def tilt_deg(tag) -> float:
    """Angle between the tag's axis and the vertical (degrees, 0..90; either end may be down)."""
    return float(np.degrees(np.arccos(np.clip(axis(tag)[2], 0.0, 1.0))))


def upright(tag) -> bool:
    """(a) Its axis within 10 degrees of vertical."""
    return tilt_deg(tag) <= L.UPRIGHT_DEG


def bottom_gap(tag) -> float:
    """Height of the tag's lowest point above the tray's top (m; negative = below it)."""
    return float(tag.low[2] - L.TRAY_TOP)


def on_tray(tag) -> bool:
    """(b) Its lowest point within 3 mm of the tray's top: standing on the tray, not on the core (30 mm higher),
    on another tag or on the table (6 mm lower)."""
    return abs(bottom_gap(tag)) <= L.ON_TRAY_TOL


def side_of_core(tag) -> str | None:
    """(d) 'near' or 'far' when its centre is outside the core and at most 45 mm from that long edge, else None."""
    x = float(tag.pos[0])
    if L.NEAR_BAND[0] <= x <= L.NEAR_BAND[1]:
        return "near"
    if L.FAR_BAND[0] <= x <= L.FAR_BAND[1]:
        return "far"
    return None


def nearest_boundary(tag) -> tuple[int, float]:
    """Index of the boundary nearest to the tag's centre along the core, and the signed offset to it (m, +y)."""
    y = float(tag.pos[1])
    offs = [y - by for by in L.BOUNDARY_Y]
    k = int(np.argmin(np.abs(offs)))
    return k, float(offs[k])


def tip_offset(arm, tag) -> tuple[float, float]:
    """For one arm: the horizontal distance from its pad-tip point to the tag's axis (taken at the tip's height),
    and the tip's height above the tag's top (m)."""
    p = np.asarray(arm.tips, dtype=float)
    c = np.asarray(tag.pos, dtype=float)
    u = axis(tag)
    on_axis = c + u * ((p[2] - c[2]) / u[2]) if u[2] > 0.1 else c  # where the axis is at the tip's height
    return float(np.hypot(*(p[:2] - on_axis[:2]))), float(p[2] - tag.high[2])


def opening(arm) -> float:
    """rigstate's opening, 0 (closed) .. 1 (fully open); an unreadable value is taken as fully open (widest reach)."""
    try:
        v = float(getattr(arm, "opening", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), 1.0) if math.isfinite(v) else 1.0


def reach(arm) -> float:
    """(e) How far from this gripper's pad-tip point (horizontally) a tag's axis can be while a finger touches it,
    over every yaw, plus 5 mm: 75 mm fully open, 43 mm closed on a tag (m)."""
    return L.release_reach(opening(arm))


def held(tag, arms) -> bool:
    """(e) Not let go and clear: a gripper's pad-tip point lower than 10 mm above the tag's top with the tag's axis
    within that gripper's reach horizontally (so a finger can be touching or propping the tag, whatever the
    gripper's yaw). Unreadable arms count as let go (and are counted by the tracker)."""
    if not arms_valid(arms):
        return False
    for side in ("left", "right"):
        xy, above = tip_offset(arms[side], tag)
        if xy < reach(arms[side]) and above < L.RELEASE_ABOVE:
            return True
    return False


def path_length(history) -> float:
    """Path length of a sequence of centre positions (m)."""
    pts = list(history)
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(pts, pts[1:])))


def at_start_mark(tag, mark) -> bool:
    """Standing on the table on its start mark."""
    _, x, y = mark
    return bool(np.hypot(tag.pos[0] - x, tag.pos[1] - y) < L.START_TOL and abs(tag.low[2] - L.TOP) < 0.003
                and tilt_deg(tag) < 5.0)


def tagged_boundaries(placed: dict[str, bool], tags: dict) -> list[str]:
    """Names of the boundaries with a placed tag's centre within 10 mm of them along the core (each counts once)."""
    out = set()
    for name, ok in placed.items():
        if ok:
            for bname, _, by in L.BOUNDARIES:
                if abs(float(tags[name].pos[1]) - by) <= L.TAG_TOL:
                    out.add(bname)
    return sorted(out)


def _mm(v: float) -> float:
    return round(float(v) * 1000.0, 1)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial: each tag's recent centre positions, and the steps the arms were
    unreadable."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.invalid_steps = 0  # steps with a non-finite tag state (scored 0, nothing recorded)
        self.arms_unavailable = 0  # steps on which rigstate could not read the arms
        self.history = {n: deque(maxlen=L.REST_WINDOW + 1) for n in L.TAG_NAMES}
        self.prev = None  # each tag's centre at the previous valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(tags) -> bool:
        """The starting layout: every tag standing on the table on its own mark."""
        return all(at_start_mark(tags[name], mark) for name, mark in zip(L.TAG_NAMES, L.TAG_STARTS))

    def at_rest(self, name: str) -> bool:
        """(c) The centre's path over the last 0.5 s (10 control steps) under 3 mm."""
        h = self.history[name]
        return len(h) == h.maxlen and path_length(h) < L.REST_PATH

    def update(self, tags, arms=None) -> Result:
        if not all(name in tags and valid(tags[name]) for name in L.TAG_NAMES):  # blow-up: 0, nothing recorded
            self.steps += 1
            self.invalid_steps += 1
            self.last = Result(0.0, False, {"valid_state": False, "invalid_steps": self.invalid_steps})
            return self.last
        now = {n: np.array(tags[n].pos, dtype=float) for n in L.TAG_NAMES}
        if self.prev is not None and self.at_start(tags):
            if max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.TAG_NAMES) > L.RESET_JUMP:
                self.reset()  # the simulator put the tags back on their marks: a new episode
        self.steps += 1
        for n in L.TAG_NAMES:
            self.history[n].append(now[n])
        self.prev = now
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(tags, arms)
        return self.last

    def result(self, tags, arms=None) -> Result:
        if not all(name in tags and valid(tags[name]) for name in L.TAG_NAMES):
            return Result(0.0, False, {"valid_state": False, "invalid_steps": self.invalid_steps})
        have_arms = arms_valid(arms)
        placed, per_tag = {}, {}
        for name in L.TAG_NAMES:
            t = tags[name]
            up, tray, side, rest = upright(t), on_tray(t), side_of_core(t), self.at_rest(name)
            grip = held(t, arms)
            placed[name] = bool(up and tray and side is not None and rest and not grip)
            k, off = nearest_boundary(t)
            row = {
                "x_mm": _mm(t.pos[0]), "y_mm": _mm(t.pos[1]), "tilt_deg": round(tilt_deg(t), 2),
                "bottom_gap_mm": _mm(bottom_gap(t)), "upright": up, "on_tray": tray, "side": side,
                "path_mm": _mm(path_length(self.history[name])), "at_rest": rest, "held": grip,
                "nearest": L.BOUNDARIES[k][0], "off_mm": _mm(off), "placed": placed[name],
            }
            if have_arms:
                row["tip_xy_mm"] = min(_mm(tip_offset(arms[s], t)[0]) for s in ("left", "right"))
            per_tag[name] = row
        tagged = tagged_boundaries(placed, tags)
        score = L.POINTS_PER_BOUNDARY * len(tagged)
        success = bool(len(tagged) == len(L.BOUNDARIES))
        details = {
            "valid_state": True,
            "tagged": tagged,
            "n_tagged": len(tagged),
            "tags": per_tag,
            "arms_available": have_arms,
            "reach_mm": {s: _mm(reach(arms[s])) for s in ("left", "right")} if have_arms else None,
            "arms_unavailable": self.arms_unavailable,
            "invalid_steps": self.invalid_steps,
        }
        return Result(score, success, details)
