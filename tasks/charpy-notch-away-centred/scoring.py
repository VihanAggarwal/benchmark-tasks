"""The scorer, as plain functions of the specimen's state and the grippers' state. NumPy only; no simulator needed.

The score is read from where the specimen lies at the end, as a technician checks a loaded Charpy bar before the
pendulum is released: is it lying on both supports along the line between the anvils with its notch running
vertically, does its notch face away from the striker (toward the anvils), how far is the notch from the midplane
between the anvils (half the difference of two rule readings, one from each anvil's inner edge), and does a 1.5 mm
feeler gauge enter between the bar and either anvil. Every tier also needs the bar let go (both grippers clear of it)
and at rest. The scorer keeps only the bar's last half second of end positions (to tell a bar at rest from one still
moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(specimen, arms) -> Result(score, success, details)

``specimen`` is the bar's state as mechkit reports it (``AssemblyState``): ``origin`` (its base frame's origin: the
middle of its bottom face), ``quat_xyzw`` (that frame's orientation: local +x along the bar, local -y out of the
notched face, local +z up through the black tick on its top face) and the ``ObjectState`` fields ``pos``,
``lin_vel``, ``ang_vel``. A state with a non-finite number, a zero quaternion or no ``origin`` (a simulator blow-up,
or the bar not built as an assembly) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the pad
tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the hands-off condition
is taken as met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so
a gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is every point within a radius of the
38 mm segment from the point between its pad tips back up the gripper; the radius reaches a pad's outer corner (half
the current finger gap plus the 8 mm pad thickness, and half the fingers' 27 mm width across; 57 mm fully open).

The bar is the same either way up (its notch runs its full height), so 'lying flat' is measured from whichever of
its top and bottom faces is nearer to facing down, and the notch's direction is read from the notched face itself.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

SUP_TOP_W = L.TOP + L.SUP_TOP  # the supports' tops (world z)


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


def valid(sp) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(sp, k), dtype=float).ravel() for k in ("origin", "quat_xyzw", "pos")]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5)


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


def to_world(sp, local) -> np.ndarray:
    return np.asarray(sp.origin, dtype=float) + rotation(sp.quat_xyzw) @ np.asarray(local, dtype=float)


def ends(sp) -> tuple[np.ndarray, np.ndarray]:
    """The centres of the bar's two end faces (its local -x end, then its +x end), in the world."""
    return to_world(sp, L.END_LOCAL[0]), to_world(sp, L.END_LOCAL[1])


def tilt_deg(sp) -> float:
    """Angle between the vertical and the notch's 20 mm length (the bar's local z), either way up (0..90 degrees): 0
    lying flat with the notch running vertically; 90 lying on its notched or back face, or standing on an end."""
    return float(np.degrees(np.arccos(np.clip(abs(rotation(sp.quat_xyzw)[2, 2]), 0.0, 1.0))))


def heading_err_deg(sp) -> float:
    """Angle between the bar's length seen from above and the line between the anvils (y), either way round
    (0..90 degrees)."""
    a = rotation(sp.quat_xyzw)[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-9:  # standing on an end
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(a[1]) / h))))


def notch_normal(sp) -> np.ndarray:
    """The notched face's outward normal, in the world."""
    return rotation(sp.quat_xyzw) @ np.asarray(L.NORMAL_LOCAL, dtype=float)


def notch_angle_deg(sp) -> float:
    """Angle between the notched face's outward normal and +x (toward the anvils, away from the striker): 0 notch
    away from the striker, 180 notch toward it, 90 sideways, up or down."""
    return float(np.degrees(np.arccos(np.clip(notch_normal(sp)[0], -1.0, 1.0))))


def notch_offset(sp) -> float:
    """Signed distance along y from the centre of the notch's mouth to the midplane between the anvils (m; + toward
    the left anvil). On a real rig: half the difference of the two rule readings from each anvil's inner edge."""
    return float(to_world(sp, L.NOTCH_LOCAL)[1] - L.MID_Y)


def _underside(sp, point) -> np.ndarray:
    """The bar's underside straight below a mid-height point of it, in its own frame (whichever way up it is)."""
    r = rotation(sp.quat_xyzw)
    up = r[:, 2] if r[2, 2] >= 0.0 else -r[:, 2]
    return np.asarray(point, dtype=float) - up * (L.SP_H / 2)


def supports(sp) -> dict:
    """Where the bar's two ends are: for the end nearer the left anvil (+y) and the one nearer the right anvil, how far
    past its support's inner edge it reaches along y, its centre line's x, and its underside's height above the
    supports' tops; and whether both ends rest on their supports."""
    a, b = ends(sp)
    left, right = (a, b) if a[1] >= b[1] else (b, a)
    out = {"onto_mm": [], "end_x_mm": [], "underside_mm": [], "on_supports": True}
    for end, sign in ((left, 1.0), (right, -1.0)):
        onto = sign * end[1] - L.SUP_Y_IN
        under = float(_underside(sp, end)[2] - SUP_TOP_W)
        ok = (onto >= L.ONTO_SUPPORT and L.SUP_X0 <= end[0] <= L.SUP_X1 and abs(under) <= L.SUPPORT_TOL)
        out["onto_mm"].append(round(onto * 1000, 2))
        out["end_x_mm"].append(round(float(end[0]) * 1000, 2))
        out["underside_mm"].append(round(under * 1000, 2))
        out["on_supports"] = out["on_supports"] and bool(ok)
    return out


def anvil_gaps(sp) -> tuple[float, float]:
    """(left, right): the widest gap (m) between each anvil's face and the bar's notched face, along the stretch of
    the bar that lies in front of that anvil (where a feeler gauge would go in, at mid-height); inf when no part of
    the bar's notched face is in front of it. Negative: pressed into it (the solver's penetration)."""
    r = rotation(sp.quat_xyzw)
    p0 = to_world(sp, (0.0, -L.SP_W / 2, L.SP_H / 2))  # the notched face's line at mid-height: p0 + s * axis
    axis = r[:, 0]
    out = []
    for sign in (1.0, -1.0):
        lo, hi = sorted((sign * L.ANVIL_Y_IN, sign * L.ANVIL_Y_OUT))
        if abs(axis[1]) < 1e-9:
            out.append(math.inf)
            continue
        s1, s2 = sorted(((lo - p0[1]) / axis[1], (hi - p0[1]) / axis[1]))
        s1, s2 = max(s1, -L.SP_LEN / 2), min(s2, L.SP_LEN / 2)
        if s2 < s1:
            out.append(math.inf)
            continue
        out.append(float(max(L.FACE_X - (p0[0] + s * axis[0]) for s in (s1, s2))))
    return out[0], out[1]


def pose_tier(sp) -> tuple[float, str, dict]:
    """(score from the pose alone, why it is 0 if it is, diagnostics). Every pose lying on the supports, notch away
    and within 8 mm of the midplane maps to exactly one of 0.5, 0.75 and 1.0."""
    tilt, head, ang = tilt_deg(sp), heading_err_deg(sp), notch_angle_deg(sp)
    off = notch_offset(sp)
    sup = supports(sp)
    gl, gr = anvil_gaps(sp)
    n = notch_normal(sp)

    def mm(v):
        return round(v * 1000, 2) if math.isfinite(v) else None

    d = {"tilt_deg": round(tilt, 2), "heading_err_deg": round(head, 2), "notch_angle_deg": round(ang, 2),
         "notch_dir": [round(float(v), 3) for v in n], "offset_mm": round(off * 1000, 2),
         "gap_left_mm": mm(gl), "gap_right_mm": mm(gr), **sup}
    if tilt > L.TILT_MAX_DEG:
        return 0.0, "not_lying_flat", d
    if head > L.HEADING_MAX_DEG:
        return 0.0, "not_along_the_anvils", d
    if not sup["on_supports"]:
        return 0.0, "not_on_the_supports", d
    if ang > L.NOTCH_CONE_DEG:
        return 0.0, ("notch_toward_striker" if n[0] < 0.0 else "notch_sideways"), d
    if abs(off) > L.CENTRE_LOOSE:
        return 0.0, "off_centre", d
    tier = 0.5
    if abs(off) <= L.CENTRE_TIGHT:
        tier += 0.25
    if max(gl, gr) <= L.ANVIL_GAP:
        tier += 0.25
    return tier, "", d


# -- the grippers ---------------------------------------------------------------------------------------------------
def point_distance(p, sp) -> float:
    """Distance from a point to the bar's bounding box (0 inside it), in the bar's own frame."""
    q = rotation(sp.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(sp.origin, dtype=float))
    d = np.abs(q - np.asarray(L.BOX_CENTER_LOCAL)) - np.asarray(L.BOX_HALF)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def zone_radius(arm) -> float:
    """From the gripper's axis to a pad's outer corner: half the gap plus the pad's thickness, and half its width."""
    gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
    return float(math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def zone_gap(arm, sp, samples: int = 20) -> float:
    """How far a gripper's finger zone is from the bar (negative: they overlap)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    return min(point_distance(p, sp) for p in pts) - zone_radius(arm)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: the bar's end points over the last half second, and how many steps
    the arms could not be read."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=self.window + 1)  # (end a, end b) for this step and the 10 before it
        self.prev_origin = None
        self.arms_unavailable = 0
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(sp) -> bool:
        """The starting layout: the bar lying still on the mat on its mark, along x, notched face toward -y."""
        o = np.asarray(sp.origin, dtype=float)
        on_mark = (math.hypot(o[0] - L.SP_START[0], o[1] - L.SP_START[1]) < L.START_TOL
                   and abs(o[2] - (L.TOP + L.MAT_TOP)) < 0.003)
        facing = -notch_normal(sp)[1] > math.cos(math.radians(5.0))
        try:
            slow = float(np.linalg.norm(np.asarray(sp.lin_vel, dtype=float))) < 0.01
        except (AttributeError, TypeError, ValueError):
            slow = True
        return bool(on_mark and facing and rotation(sp.quat_xyzw)[2, 2] > math.cos(math.radians(2.0)) and slow)

    def moved(self) -> float | None:
        """The farthest either end of the bar has been, over the last 0.5 s, from where it is now (m); None until
        0.5 s of this trial has been seen."""
        if len(self.history) < self.window + 1:
            return None
        a, b = self.history[-1]
        return float(max(max(np.linalg.norm(p - a), np.linalg.norm(q - b)) for p, q in self.history))

    def update(self, sp, arms=None) -> Result:
        if not valid(sp):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(sp, arms)
            return self.last
        o = np.asarray(sp.origin, dtype=float)
        if self.prev_origin is not None and self.at_start(sp):
            if float(np.linalg.norm(o - self.prev_origin)) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()
        self.steps += 1
        self.history.append(ends(sp))
        self.prev_origin = o
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(sp, arms)
        return self.last

    def result(self, sp, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (idempotent: it changes nothing)."""
        if not valid(sp):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state"})
        tier, reason, d = pose_tier(sp)
        moved = self.moved()
        at_rest = moved is not None and moved <= L.STILL_TOL
        if arms_valid(arms):
            zl, zr = zone_gap(arms["left"], sp), zone_gap(arms["right"], sp)
            off = min(zl, zr) >= L.HANDS_OFF
            zones = [round(zl * 1000, 1), round(zr * 1000, 1)]
        else:  # not readable: taken as met, and counted
            off, zones = True, None
        score = tier if (at_rest and off) else 0.0
        if tier > 0 and not off:
            reason = "gripper_on_or_near_it"
        elif tier > 0 and not at_rest:
            reason = "still_moving"
        details = {"valid_state": True, "tier": tier, "zero_reason": reason, "at_rest": at_rest,
                   "moved_mm": None if moved is None else round(moved * 1000, 2), "hands_off": off,
                   "zone_gap_mm": zones, "arms_unavailable": self.arms_unavailable, **d}
        return Result(score, bool(score == 1.0), details)
