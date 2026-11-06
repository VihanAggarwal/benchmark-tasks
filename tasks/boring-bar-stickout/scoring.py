"""The scorer, as plain functions of the boring bar's state. NumPy only; no simulator needed.

The score is read from where the bar lies at the end, as a setter checks it with a steel rule and a look from
above: is it lying in the tool block's channel with its insert up, does its tip point at the chuck, and how far does
the tip stick out past the block's chuck-side face (along the spindle axis, y). The scorer keeps only the bar's
last half second of positions (to tell a bar at rest from one still moving) and is fed one observation per control
step by ``task.success()``:

    Tracker.update(bar) -> Result(score, success, details)

``bar`` is the boring bar's state as mechkit reports it (``AssemblyState``): ``origin`` (its base frame's origin:
the middle of the bar's bottom face), ``quat_xyzw`` (that frame's orientation: local +x toward the tip, local +z up
through the insert), and the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``. A state with a non-finite
number, a zero quaternion or no ``origin`` (a simulator blow-up, or the bar not built as an assembly) scores 0 for
that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

FLOOR_W = L.TOP + L.FLOOR_Z  # the channel floor (world z)


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


def valid(bar) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(bar, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


def ends(bar) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tip, back, R): the centres of the bar's tip face and back face in the world, and its rotation."""
    r = rotation(bar.quat_xyzw)
    o = np.asarray(bar.origin, dtype=float)
    return o + r @ np.array(L.TIP_LOCAL), o + r @ np.array(L.BACK_LOCAL), r


def tilt_deg(bar) -> float:
    """Angle between the vertical and the bar's insert-up axis (0..180 degrees): 0 lying level with the insert on
    top; pitched or rolled shows as tilt; on its side 90; insert down 180."""
    return float(np.degrees(np.arccos(np.clip(rotation(bar.quat_xyzw)[2, 2], -1.0, 1.0))))


def heading_err_deg(bar) -> float:
    """Angle between the bar's axis seen from above and the spindle axis (y), either way round (0..90 degrees)."""
    a = rotation(bar.quat_xyzw)[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-9:  # standing on an end
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(a[1]) / h))))


def tip_toward_chuck(bar) -> bool:
    """The tip end is the end nearer the chuck (+y)."""
    return bool(rotation(bar.quat_xyzw)[1, 0] > 0.0)


def stickout(bar) -> float:
    """How far the tip face's centre is past the block's chuck-side face, along the spindle axis (m; negative =
    the tip is behind the face, inside the block)."""
    tip, _, _ = ends(bar)
    return float(tip[1] - L.FACE_Y)


def channel(bar) -> dict:
    """Where the bar's centre line runs over the block: the part of its length between the block's back face and
    its chuck-side face (along y), and at both ends of that part the centre line's sideways offset from the
    channel's centre and the bar's underside height above the channel floor."""
    tip, back, r = ends(bar)
    axis = tip - back
    out = {"over_block_mm": 0.0, "centre_off_mm": None, "underside_mm": None}
    if abs(axis[1]) < 1e-9:  # lying across y: nothing to measure along it
        return out
    y_lo = max(L.BLOCK_Y0, min(tip[1], back[1]))
    y_hi = min(L.FACE_Y, max(tip[1], back[1]))
    if y_hi <= y_lo:
        return out
    out["over_block_mm"] = (y_hi - y_lo) * 1000
    offs, under = [], []
    for y in (y_lo, y_hi):
        p = back + (y - back[1]) / axis[1] * axis  # on the centre line (mid-height of the bar)
        bottom = p - r[:, 2] * (L.BAR_W / 2)  # the underside straight below it, in the bar's own frame
        offs.append((p[0] - L.AXIS_X) * 1000)
        under.append((bottom[2] - FLOOR_W) * 1000)
    out["centre_off_mm"], out["underside_mm"] = offs, under
    return out


def pose_tier(bar) -> tuple[float, str, dict]:
    """(score from the pose alone, why it is 0 if it is, diagnostics). Every in-channel, right-way-round, insert-up
    pose maps to exactly one tier by its stick-out."""
    tilt, head, toward = tilt_deg(bar), heading_err_deg(bar), tip_toward_chuck(bar)
    s = stickout(bar)
    ch = channel(bar)
    d = {"s_mm": round(s * 1000, 2), "tilt_deg": round(tilt, 2), "heading_err_deg": round(head, 2),
         "tip_toward_chuck": toward, "over_block_mm": round(ch["over_block_mm"], 1),
         "centre_off_mm": None if ch["centre_off_mm"] is None else [round(v, 2) for v in ch["centre_off_mm"]],
         "underside_mm": None if ch["underside_mm"] is None else [round(v, 2) for v in ch["underside_mm"]]}
    if tilt > L.TILT_MAX_DEG:
        return 0.0, "tilted_or_on_its_side", d
    if head > L.HEADING_MAX_DEG:
        return 0.0, "not_along_the_channel", d
    if not toward:
        return 0.0, "tip_away_from_chuck", d
    if ch["over_block_mm"] < L.MIN_OVER_BLOCK * 1000:
        return 0.0, "not_over_the_block", d
    if max(abs(v) for v in ch["centre_off_mm"]) > L.CENTRE_TOL * 1000:
        return 0.0, "out_of_the_channel", d
    if max(abs(v) for v in ch["underside_mm"]) > L.FLOOR_TOL * 1000:
        return 0.0, "not_on_the_channel_floor", d
    if s > L.LONG_HI:
        return 0.0, "tip_at_the_part", d
    if L.BAND_LO <= s <= L.BAND_HI:
        return 1.0, "", d
    if L.SHORT_LO <= s < L.BAND_LO or L.BAND_HI < s <= L.LONG_HI:
        return 0.5, "", d
    return 0.25, "", d


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: the bar's end points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=self.window + 1)  # (tip, back) for this step and the 10 before it
        self.prev_origin = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(bar) -> bool:
        """The starting layout: the bar lying still on the mat on its mark, along x with the tip forward."""
        o = np.asarray(bar.origin, dtype=float)
        on_mark = (math.hypot(o[0] - L.BAR_START[0], o[1] - L.BAR_START[1]) < L.START_TOL
                   and abs(o[2] - (L.TOP + L.MAT_TOP)) < 0.003)
        a = rotation(bar.quat_xyzw)[:, 0]
        forward = a[0] > math.cos(math.radians(5.0))
        try:
            slow = float(np.linalg.norm(np.asarray(bar.lin_vel, dtype=float))) < 0.01
        except (AttributeError, TypeError, ValueError):
            slow = True
        return bool(on_mark and forward and tilt_deg(bar) < 2.0 and slow)

    def moved(self) -> float | None:
        """The farthest either end of the bar has been, over the last 0.5 s, from where it is now (m); None until
        0.5 s of this trial has been seen."""
        if len(self.history) < self.window + 1:
            return None
        tip, back = self.history[-1]
        return float(max(max(np.linalg.norm(t - tip), np.linalg.norm(b - back)) for t, b in self.history))

    def update(self, bar) -> Result:
        if not valid(bar):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(bar)
            return self.last
        o = np.asarray(bar.origin, dtype=float)
        if self.prev_origin is not None and self.at_start(bar):
            if float(np.linalg.norm(o - self.prev_origin)) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()
        self.steps += 1
        tip, back, _ = ends(bar)
        self.history.append((tip, back))
        self.prev_origin = o
        self.last = self.result(bar)
        return self.last

    def result(self, bar) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not valid(bar):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        tier, reason, d = pose_tier(bar)
        moved = self.moved()
        at_rest = moved is not None and moved <= L.STILL_TOL
        score = tier if at_rest else 0.0
        if tier > 0 and not at_rest:
            reason = "still_moving"
        success = bool(score == 1.0)
        details = {"valid_state": True, "tier": tier, "zero_reason": reason, "at_rest": at_rest,
                   "moved_mm": None if moved is None else round(moved * 1000, 2), **d}
        return Result(score, success, details)
