"""The scorer, as plain functions of the manikin's state and the arms'. NumPy only; no simulator needed.

The rule is about the whole move, not only where the manikin ends up: the neck must stay straight all the way, the
head and the trunk must be lifted and lowered together, the manikin must stay low, and nobody may hold it by the
neck. So the scorer keeps a little state (the worst neck angle so far, the first rule broken, whether both bodies
were ever up together) and is fed one observation per control step by ``task.success()``.

    Tracker.update(manikin, arms) -> Result(score, success, details)

``manikin`` is the manikin's state as ``mechkit.AssemblyState`` gives it: ``origin`` and ``quat_xyzw`` of the
trunk's frame (the base body), ``lin_vel`` and ``ang_vel`` of the trunk, ``joints["neck"]`` and
``joint_vel["neck"]`` (rad, rad/s; 0 = straight) and ``links["neck"]``, the frame of the link the neck joint
carries (the head with its neck bar: x, y, z, qx, qy, qz, qw; its origin is the hinge, its axes the trunk's when
the neck is straight; mechkit names a link after its joint). The bodies' shapes come from
``layout`` (the same numbers task.py declares), so every height and footprint here is of a body's own boxes, never
of the whole manikin's bounding box. ``arms`` is ``rigstate.arms()``: {"left": Arm, "right": Arm} with ``tcp``,
``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or None when the arms could not
be read (then the gripper rules are not applied for that step and the count is kept).

A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the
tracker as it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

BODIES = ("trunk", "head")
# Each body's boxes (centre, size) in the manikin's base frame with the neck straight.
BOXES = {
    "trunk": ((L.TRUNK_CENTER, L.TRUNK_SIZE),),
    "head": ((L.NECK_BAR_CENTER, L.NECK_BAR_SIZE), (L.HEAD_CENTER, L.HEAD_SIZE)),
}
OFFSET = {"trunk": np.zeros(3), "head": np.asarray(L.NECK_ANCHOR, dtype=float)}  # each body frame's origin at rest
_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


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


def _finite(*values) -> bool:
    try:
        arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    except (TypeError, ValueError):
        return False
    return all(a.size and np.all(np.isfinite(a)) for a in arrays)


def valid(m) -> bool:
    """Every number the scorer reads is finite and both quaternions are non-degenerate."""
    try:
        head = np.asarray(m.links["neck"], dtype=float).ravel()
        parts = (m.origin, m.quat_xyzw, m.lin_vel, m.ang_vel, m.joints["neck"], m.joint_vel["neck"], head)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if not _finite(*parts) or head.size != 7 or np.asarray(m.quat_xyzw, dtype=float).size != 4:
        return False
    return bool(np.linalg.norm(np.asarray(m.quat_xyzw, dtype=float)) > 0.5 and np.linalg.norm(head[3:]) > 0.5)


def arms_ok(arms) -> bool:
    """Both arms present with finite tool point, finger-tip point and opening."""
    if not arms:
        return False
    try:
        return all(_finite(arms[a].tcp, arms[a].tips, arms[a].opening) and np.asarray(arms[a].tcp).size == 3
                   and np.asarray(arms[a].tips).size == 3 for a in ("left", "right"))
    except (AttributeError, KeyError, TypeError):
        return False


def frame(m, body: str) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of a body's frame in the world: the trunk's is the manikin's base frame, the head's is
    the head link's (origin on the hinge)."""
    if body == "trunk":
        return rotation(m.quat_xyzw), np.asarray(m.origin, dtype=float)
    lp = np.asarray(m.links["neck"], dtype=float).ravel()
    return rotation(lp[3:7]), lp[:3]


def to_world(m, body: str, point) -> np.ndarray:
    """A point drawn in the base frame with the neck straight that rides on ``body``: where it is now."""
    r, p = frame(m, body)
    return r @ (np.asarray(point, dtype=float) - OFFSET[body]) + p


def corners(m, body: str) -> np.ndarray:
    """(8k, 3) world corners of a body's k boxes as they are now."""
    r, p = frame(m, body)
    out = []
    for c, s in BOXES[body]:
        local = np.asarray(c, dtype=float) - OFFSET[body] + _SIGNS * (np.asarray(s, dtype=float) / 2)
        out.append(local @ r.T + p)
    return np.vstack(out)


def low(m, body: str) -> float:
    """Height of a body's lowest point above the table top (m)."""
    return float(corners(m, body)[:, 2].min() - L.TOP)


def footprint(m, body: str) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1): the world-aligned rectangle around a body's corners seen from above."""
    c = corners(m, body)
    return float(c[:, 0].min()), float(c[:, 0].max()), float(c[:, 1].min()), float(c[:, 1].max())


def _inside(x: float, y: float, rect) -> bool:
    return rect[0] <= x <= rect[1] and rect[2] <= y <= rect[3]


def surface_under(m, body: str) -> tuple[float, str]:
    """Height above the table top of the surface under the middle of a body's footprint (stretcher, mat or table).
    The middle, not any part: a body hovering half over the stretcher's edge with its middle still over the table
    is that high above the table (it is lifted), and it can only rest on the stretcher with its middle over it."""
    x0, x1, y0, y1 = footprint(m, body)
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    if _inside(x, y, L.STRETCHER_RECT):
        return L.STRETCHER_TOP, "stretcher"
    if _inside(x, y, L.MAT_RECT):
        return L.MAT_TOP, "mat"
    return 0.0, "table"


def lift(m, body: str) -> float:
    """How far a body's lowest point is above the surface under its middle (m)."""
    return low(m, body) - surface_under(m, body)[0]


def neck_deg(m) -> float:
    """How far the neck is bent, either way (degrees)."""
    return abs(math.degrees(float(m.joints["neck"])))


def trunk_tilt_deg(m) -> float:
    """Angle between straight up and the trunk's back-to-front axis (its base frame's z) (degrees, 0..180): 0 lying
    on its back as at the start, 90 on its side, 180 face down. A casualty goes onto the stretcher on its back."""
    return float(np.degrees(np.arccos(np.clip(rotation(m.quat_xyzw)[2, 2], -1.0, 1.0))))


def trunk_axis_deg(m) -> float:
    """Angle between the trunk's long axis (hips to head), seen from above, and the stretcher's (+y) (degrees)."""
    v = rotation(m.quat_xyzw)[:, 1]
    return float(math.degrees(math.atan2(abs(v[0]), v[1])))


def head_centre(m) -> np.ndarray:
    return to_world(m, "head", L.HEAD_CENTER)


def neck_centre(m) -> np.ndarray:
    return to_world(m, "head", L.NECK_BAR_CENTER)


def head_off(m) -> float:
    """Horizontal distance from the head box's centre to the head zone's centre (m)."""
    c = head_centre(m)
    return float(math.hypot(c[0] - L.HEAD_TARGET[0], c[1] - L.HEAD_TARGET[1]))


def inside_stretcher(m, body: str, slack: float = 0.001) -> bool:
    """Every corner of a body, seen from above, lies over the stretcher's top (to within 1 mm)."""
    c = corners(m, body)
    x0, x1, y0, y1 = L.STRETCHER_RECT
    return bool(np.all(c[:, 0] >= x0 - slack) and np.all(c[:, 0] <= x1 + slack)
                and np.all(c[:, 1] >= y0 - slack) and np.all(c[:, 1] <= y1 + slack))


def on_stretcher(m) -> tuple[bool, dict]:
    """Both bodies over the stretcher's top and resting on it (lowest points within 3 mm below / 4 mm above it),
    the trunk flat on its back, the head box's centre within 25 mm of the head zone's. (Lying wholly on the 110 mm
    wide top, the 240 mm trunk is within about 9 deg of the stretcher's axis; the axis is reported, not tested.)"""
    gaps = {b: low(m, b) - L.STRETCHER_TOP for b in BODIES}
    inside = all(inside_stretcher(m, b) for b in BODIES)
    resting = all(-L.ON_LOW_BELOW <= g <= L.ON_LOW_ABOVE for g in gaps.values())
    axis, tilt, off = trunk_axis_deg(m), trunk_tilt_deg(m), head_off(m)
    ok = bool(inside and resting and tilt <= L.FLAT_DEG and off <= L.HEAD_TOL)
    return ok, {"inside": inside, "resting_on_it": resting, "axis_deg": round(axis, 2), "tilt_deg": round(tilt, 2),
                "head_off_mm": round(off * 1000, 1)}


def point_box_distance(point, r, centre, half) -> float:
    """Distance from a world point to a box (rotation ``r``, world ``centre``, half sizes); 0 inside."""
    d = (np.asarray(point, dtype=float) - centre) @ r
    return float(np.linalg.norm(np.maximum(np.abs(d) - half, 0.0)))


def distance_to_manikin(m, point) -> float:
    """Distance from a world point to the nearest of the manikin's boxes (m)."""
    best = math.inf
    for body in BODIES:
        r, _ = frame(m, body)
        for c, s in BOXES[body]:
            best = min(best, point_box_distance(point, r, to_world(m, body, c), np.asarray(s, dtype=float) / 2))
    return best


def neck_zone(m, tcp) -> bool:
    """A tool point is at the neck: within 25 mm of the neck bar's centre along the manikin's length, 50 mm across
    it and no more than 60 mm above it."""
    n = neck_centre(m)
    d = (np.asarray(tcp, dtype=float) - n) @ rotation(m.quat_xyzw)
    return bool(abs(d[1]) <= L.NECK_ZONE_ALONG and abs(d[0]) <= L.NECK_ZONE_ACROSS
                and float(tcp[2]) - float(n[2]) <= L.NECK_ZONE_ABOVE)


def gripper_clear(m, arm) -> bool:
    """A gripper has let go: open (opening >= 0.9) and its finger-tip point at least 30 mm from the manikin."""
    return bool(float(arm.opening) >= L.GRIP_OPEN and distance_to_manikin(m, arm.tips) >= L.CLEAR_DIST)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.worst_neck = 0.0  # deg: the largest neck bend so far (W)
        self.breach: str | None = None  # the first rule broken (score 0 for the rest of the trial)
        self.breach_step: int | None = None
        self.lifted_together = False  # both bodies were up >= 20 mm at once with W <= 10 deg up to then
        self.arms_unavailable = 0  # steps on which rigstate could not read the arms
        self.head_speed = 0.0  # m/s of the head box's centre over the last step
        self.in_zone: list[str] = []  # arms whose tool point is at the neck now
        self.prev = None  # (step, trunk origin, lowest points, head centre) at the previous valid step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(m) -> bool:
        """The starting layout: the manikin flat and still on its mark on the mat, neck straight."""
        o = np.asarray(m.origin, dtype=float)
        on_mark = (math.hypot(o[0] - L.MANIKIN_ORIGIN[0], o[1] - L.MANIKIN_ORIGIN[1]) < L.START_TOL
                   and abs(o[2] - L.MANIKIN_ORIGIN[2]) < 0.003)
        return bool(on_mark and trunk_tilt_deg(m) < 1.0 and neck_deg(m) < 1.0
                    and np.linalg.norm(np.asarray(m.lin_vel, dtype=float)) < L.STILL_SPEED)

    def _break(self, reason: str) -> None:
        if self.breach is None:
            self.breach, self.breach_step = reason, self.steps

    def update(self, m, arms=None) -> Result:
        if not valid(m):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.last = self.result(m, arms)
            return self.last
        origin = np.asarray(m.origin, dtype=float)
        if self.prev is not None and self.at_start(m):
            if float(np.linalg.norm(origin - self.prev[1])) > L.RESET_JUMP:  # put back by the simulator: new episode
                self.reset()
        self.steps += 1
        if not arms_ok(arms):
            arms = None
            self.arms_unavailable += 1
        theta = neck_deg(m)
        self.worst_neck = max(self.worst_neck, theta)
        lows = {b: low(m, b) for b in BODIES}
        lifts = {b: lows[b] - surface_under(m, b)[0] for b in BODIES}
        lifted = {b: lifts[b] > L.LIFTED for b in BODIES}
        head = head_centre(m)

        if self.worst_neck > L.NECK_ZERO_DEG:
            self._break("neck_bent")
        if max(lows.values()) > L.LIFT_CAP:
            self._break("lifted_too_high")
        if min(lows.values()) < -L.OFF_TABLE:
            self._break("off_the_table")
        for a, b in (("trunk", "head"), ("head", "trunk")):
            if lifts[a] > L.LIFT_ALONE and not lifted[b]:
                self._break(f"{a}_lifted_alone")
        if self.prev is not None:
            elapsed = (self.steps - self.prev[0]) * self.dt
            for b in BODIES:
                if (self.prev[2][b] - lows[b]) / elapsed > L.FALL_SPEED:
                    self._break(f"{b}_fell")
            self.head_speed = float(np.linalg.norm(head - self.prev[3])) / elapsed
        self.in_zone = []
        if arms is not None:
            self.in_zone = [a for a in ("left", "right") if neck_zone(m, arms[a].tcp)]
            if self.in_zone and any(lifted.values()):
                self._break("held_by_the_neck")
        if all(lifts[b] >= L.LIFT_TOGETHER for b in BODIES) and self.worst_neck <= L.NECK_TOGETHER_DEG:
            self.lifted_together = True
        self.prev = (self.steps, origin, lows, head)
        self.last = self.result(m, arms)
        return self.last

    def still(self, m) -> bool:
        return bool(np.linalg.norm(np.asarray(m.lin_vel, dtype=float)) < L.STILL_SPEED
                    and np.linalg.norm(np.asarray(m.ang_vel, dtype=float)) < L.STILL_SPIN
                    and abs(float(m.joint_vel["neck"])) < L.STILL_NECK and self.head_speed < L.STILL_SPEED)

    def result(self, m, arms=None) -> Result:
        if not valid(m):
            return Result(0.0, False, {"valid_state": False, "breach": self.breach,
                                       "worst_neck_deg": round(self.worst_neck, 2)})
        arms = arms if arms_ok(arms) else None
        theta = neck_deg(m)
        worst = max(self.worst_neck, theta)
        on, where = on_stretcher(m)
        clear = {a: (gripper_clear(m, arms[a]) if arms is not None else None) for a in ("left", "right")}
        let_go = all(v is not False for v in clear.values())  # arms unreadable: taken as let go
        if self.breach is not None or worst > L.NECK_ZERO_DEG:
            score = 0.0
        elif on and worst <= L.NECK_ONE_DEG and theta <= L.FINAL_ONE_DEG and let_go:
            score = 1.0
        elif on and worst <= L.NECK_075_DEG and theta <= L.FINAL_075_DEG:
            score = 0.75
        elif on and worst <= L.NECK_05_DEG:
            score = 0.5
        elif on or self.lifted_together:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and self.still(m))
        lows = {b: low(m, b) for b in BODIES}
        surf = {b: surface_under(m, b) for b in BODIES}
        details = {
            "valid_state": True,
            "neck_deg": round(theta, 2),
            "worst_neck_deg": round(worst, 2),
            "on_stretcher": on,
            **where,
            "low_trunk_mm": round(lows["trunk"] * 1000, 1),
            "low_head_mm": round(lows["head"] * 1000, 1),
            "lift_trunk_mm": round((lows["trunk"] - surf["trunk"][0]) * 1000, 1),
            "lift_head_mm": round((lows["head"] - surf["head"][0]) * 1000, 1),
            "under_trunk": surf["trunk"][1],
            "under_head": surf["head"][1],
            "lifted_together": self.lifted_together,
            "left_clear": clear["left"],
            "right_clear": clear["right"],
            "at_neck": list(self.in_zone),
            "head_speed_mps": round(self.head_speed, 4),
            "breach": self.breach,
            "breach_step": self.breach_step,
            "arms_unavailable": self.arms_unavailable,
        }
        return Result(score, success, details)
