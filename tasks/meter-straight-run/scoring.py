"""The scorer, as plain functions of the four loose parts' states and the grippers' state. NumPy only; no simulator.

The score is read from the end state, as a meter fitter checks a meter setting with a tape and a look from above:
are the meter and the three straights all lying in the channel, butted end to end from the elbow, is the meter's
arrow pointing toward the valve with its register up, and how much straight pipe is there before the meter (from
the elbow's face to the meter's inlet face) and after it (from its outlet face to the valve's face). Both grippers
must have let go and moved clear. The scorer keeps only the parts' last half second of end points (to tell a run at rest from one
still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(states, arms) -> Result(score, success, details)

``states`` maps each part's name to its state: the three straights (``pipe_250``, ``pipe_125``, ``pipe_75``) are
plain boxes with ``pos`` (centre), ``quat_xyzw`` and ``lin_vel``, as ``yam_arena.task_file.ObjectState`` has; the
meter is a mechkit compound (``AssemblyState``) whose ``origin`` is its body's centre and whose ``quat_xyzw`` turns
its frame (local +x along the flow arrow, +z up through the register). ``pos`` of the meter is the centre of the box
round its body and register, so it is not used. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}``
with ``tips`` (the point between the finger tips) and ``tcp`` (13 mm back up the gripper), or ``None`` when the arms
could not be read. A gripper has let go when its 'finger line', from ``tips`` 38 mm back up the gripper (through
``tcp``), is 65 mm or more from every part: an open finger reaches 60.4 mm out from that line, so no finger touches a
part, however the gripper is tilted. Each gripper is judged on its own: one that cannot be read (no snapshot, no such
arm, a non-finite point) has NOT let go on that step, so the score is 0 then, and the step is counted in
``arms_unavailable``. A part state with a non-finite
number, a zero quaternion or a missing field (a simulator blow-up) scores 0 for that step and leaves the tracker as
it was.

The straights are square tubes, the same end for end and on any of their four long faces: 'flat' is measured from
whichever long face is nearest to facing down, and their two ends are alike.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

STILL_STEPS = int(round(L.STILL_WINDOW_S * L.CONTROL_HZ))  # 10 control steps at 20 Hz


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


def _finite(v, n: int) -> np.ndarray | None:
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return a if a.size == n and np.all(np.isfinite(a)) else None


def centre_key(name: str) -> str:
    """The field that holds a part's centre: the meter's body centre is its frame's ``origin``."""
    return "origin" if name == "meter" else "pos"


def valid(name: str, s) -> bool:
    """Its centre and orientation present and finite, the quaternion not degenerate."""
    c = _finite(getattr(s, centre_key(name), None), 3)
    q = _finite(getattr(s, "quat_xyzw", None), 4)
    return bool(c is not None and q is not None and np.linalg.norm(q) > 0.5)


def pose(name: str, s) -> tuple[np.ndarray, np.ndarray]:
    """(centre, rotation) of a part: a straight's centre, the meter's body centre."""
    return np.asarray(getattr(s, centre_key(name)), dtype=float), rotation(s.quat_xyzw)


def outline(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(centre, half size) of a part's outline box in its own frame: a straight's whole tube; the meter's body with
    its register and arrow on top (25 + 15 mm tall)."""
    if name == "meter":
        lo, hi = -L.HALF_W, L.METER_TOP_LOCAL
        return np.array([0.0, 0.0, (lo + hi) / 2]), np.array([L.METER_LEN / 2, L.HALF_W, (hi - lo) / 2])
    return np.zeros(3), np.array([L.LENGTH[name] / 2, L.HALF_W, L.HALF_W])


_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def lowest(name: str, c, r) -> float:
    """Height of the part's lowest point above the table top (m), from its outline box."""
    oc, half = outline(name)
    corners = c + (oc + _SIGNS * half) @ r.T
    return float(corners[:, 2].min() - L.TOP)


def flat_deg(name: str, r) -> float:
    """(a) For a straight: the angle between the vertical and the normal of whichever long face is nearest to facing
    down (0..90). For the meter: the angle between the vertical and its register's up axis (0..180; upside down
    180, on its side 90)."""
    if name == "meter":
        return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))
    return float(np.degrees(np.arccos(np.clip(max(abs(r[2, 1]), abs(r[2, 2])), 0.0, 1.0))))


def heading_err_deg(r) -> float:
    """(b) The angle between the part's long axis seen from above and y, either way round (0..90)."""
    a = r[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-9:  # standing on an end
        return 90.0
    return float(np.degrees(math.acos(min(1.0, abs(a[1]) / h))))


def ends(name: str, c, r) -> tuple[float, float]:
    """(y of the end nearer the elbow, y of the end nearer the valve) of a part's two end-face centres."""
    d = r[:, 0] * (L.LENGTH[name] / 2)
    y1, y2 = float(c[1] + d[1]), float(c[1] - d[1])
    return max(y1, y2), min(y1, y2)


def laid(name: str, c, r) -> tuple[bool, str, dict]:
    """A part lies in the channel: (a) flat (register up for the meter) within 5 degrees, (b) along y within 5
    degrees, (c) its centre within 8 mm of x 0.300, (d) its lowest point within 3 mm of the table top."""
    flat, head = flat_deg(name, r), heading_err_deg(r)
    off, low = abs(float(c[0]) - L.RUN_X), lowest(name, c, r)
    d = {"flat_deg": round(flat, 2), "heading_deg": round(head, 2), "centre_off_mm": _mm(off),
         "bottom_mm": _mm(low)}
    if flat > L.FLAT_DEG:
        return False, ("meter_not_register_up" if name == "meter" else "not_flat"), d
    if head > L.HEADING_DEG:
        return False, "not_along_the_run", d
    if off > L.CENTRE_TOL:
        return False, "off_the_centre_line", d
    if abs(low) > L.BOTTOM_TOL:
        return False, "not_on_the_table", d
    return True, "", d


def point_box_distance(p, c, r, name: str) -> float:
    """Distance from a world point to a part's outline box, turned as the part is (0 inside it)."""
    oc, half = outline(name)
    local = (np.asarray(p, dtype=float) - c) @ r - oc
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


SIDES = ("left", "right")


def finger_line(arms, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    """One gripper's finger line: (its finger-tip point, the point 38 mm back up the gripper toward and past its
    TCP), or None when it cannot be read (rigstate.arms() can be None)."""
    try:
        tips, tcp = _finite(arms[side].tips, 3), _finite(arms[side].tcp, 3)
    except (KeyError, AttributeError, TypeError, IndexError):
        return None
    if tips is None or tcp is None:
        return None
    up = tcp - tips
    n = float(np.linalg.norm(up))
    up = up / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    return tips, tips + L.FINGER_LEN * up


def arms_valid(arms) -> bool:
    """Both grippers can be read."""
    return all(finger_line(arms, s) is not None for s in SIDES)


def segment_box_distance(p0, p1, c, r, name: str) -> float:
    """Distance from the segment p0-p1 to a part's outline box, turned as the part is (0 if they meet). The distance
    to a box is convex along a line, so a ternary search finds its smallest value (to well under a micrometre)."""
    oc, half = outline(name)
    a = [float(v) for v in (np.asarray(p0, dtype=float) - c) @ r - oc]  # p0 in the box's frame
    b = [float(v) for v in (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)) @ r]  # p0 -> p1 in it
    h = [float(v) for v in half]

    def f(t: float) -> float:
        return math.sqrt(sum(max(abs(a[i] + t * b[i]) - h[i], 0.0) ** 2 for i in range(3)))

    lo, hi = 0.0, 1.0
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if f(m1) <= f(m2):
            hi = m2
        else:
            lo = m1
    return min(f(0.0), f(1.0), f((lo + hi) / 2))


def tip_clearance(arms, poses: dict) -> dict[str, float | None]:
    """Each gripper's finger line's distance to the nearest part's outline box (m); None for a gripper that cannot be
    read."""
    out = {}
    for s in SIDES:
        seg = finger_line(arms, s)
        out[s] = None if seg is None else min(segment_box_distance(*seg, *poses[n], n) for n in L.PARTS)
    return out


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


def evaluate(poses: dict, arms) -> tuple[float, str, dict]:
    """(score, why it is not 1.0, diagnostics) of a layout. ``poses`` maps each part to (centre, rotation)."""
    d: dict = {}
    clear = tip_clearance(arms, poses)
    readable = all(v is not None for v in clear.values())
    released = bool(readable and min(clear.values()) >= L.RELEASE - L.EPS)
    d["arms_available"] = readable
    d["tips_clear_mm"] = {s: _mm(v) for s, v in clear.items()}
    d["released"] = released
    parts = {}
    for n in L.PARTS:
        ok, why, info = laid(n, *poses[n])
        hi, lo = ends(n, *poses[n])
        parts[n] = {"laid": ok, "why": why, "inlet_end_mm": _mm(hi), "outlet_end_mm": _mm(lo), **info}
    d["parts"] = parts
    order = sorted(L.PARTS, key=lambda n: -float(poses[n][0][1]))
    d["order"] = order
    hi = {n: ends(n, *poses[n])[0] for n in L.PARTS}
    lo = {n: ends(n, *poses[n])[1] for n in L.PARTS}
    joints = [L.ELBOW_FACE_Y - hi[order[0]]] + [lo[a] - hi[b] for a, b in zip(order, order[1:])]
    d["joints_mm"] = [_mm(j) for j in joints]
    d["valve_gap_mm"] = _mm(lo[order[-1]] - L.VALVE_FACE_Y)
    m_r = poses["meter"][1]
    downstream = bool(m_r[1, 0] < 0.0)
    u, dn = L.ELBOW_FACE_Y - hi["meter"], lo["meter"] - L.VALVE_FACE_Y
    d["arrow"] = "downstream" if downstream else "backwards"
    d["U_mm"], d["D_mm"] = _mm(u), _mm(dn)
    if not readable:
        return 0.0, "gripper_not_readable", d
    if not released:
        return 0.0, "gripper_not_released", d
    for n in L.PARTS:
        if not parts[n]["laid"]:
            return 0.0, f"{n}_{parts[n]['why']}", d
    if max(joints) > L.JOINT_MAX + L.EPS:
        return 0.0, "joint_open", d
    if min(joints) < L.JOINT_MIN - L.EPS:
        return 0.0, "joint_overlap", d
    return (*tier(u, dn, downstream), d)


def tier(u: float, dn: float, downstream: bool) -> tuple[float, str]:
    """The score of a closed run from the meter's direction and its straight runs: 0.25 backwards; 1.0 with the
    arrow toward the valve, U >= 245 mm and D >= 120 mm; 0.5 otherwise (a short U means the 250 is downstream, so
    D is long, and a short D means U is long: only one can be short)."""
    if not downstream:
        return 0.25, "meter_backwards"
    if u >= L.U_MIN - L.EPS and dn >= L.D_MIN - L.EPS:
        return 1.0, ""
    return 0.5, ("upstream_short" if u < L.U_MIN - L.EPS else "downstream_short")


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: every part's two end points over the last half second."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.arms_unavailable = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a non-finite part state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.history: deque = deque(maxlen=self.window + 1)  # (8, 3) end points for this step and the 10 before
        self.prev = None  # {name: (centre, velocity)} at the previous valid step
        self.last = Result(0.0, False, {})

    @staticmethod
    def valid_states(states) -> bool:
        try:
            return all(valid(n, states[n]) for n in L.PARTS)
        except (KeyError, TypeError):
            return False

    @staticmethod
    def poses(states) -> dict:
        return {n: pose(n, states[n]) for n in L.PARTS}

    @staticmethod
    def at_start(poses: dict) -> bool:
        """The starting layout: every part flat on the table near its starting place, lying along x (the meter's
        arrow forward)."""
        for n in L.PARTS:
            c, r = poses[n]
            sx, sy = L.METER_START if n == "meter" else L.PIPE_START[n][:2]
            a = r[:, 0]
            along_x = (a[0] if n == "meter" else abs(a[0])) > math.cos(math.radians(5.0))
            if not (math.hypot(c[0] - sx, c[1] - sy) < L.START_TOL and along_x and flat_deg(n, r) < 2.0
                    and abs(lowest(n, c, r)) < 0.003):
                return False
        return True

    @staticmethod
    def _velocity(name: str, s) -> np.ndarray:
        v = _finite(getattr(s, "lin_vel", None), 3)
        return np.zeros(3) if v is None else v

    def _teleported(self, now: dict) -> bool:
        """Some part moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for n, (p0, v0) in self.prev.items())

    def end_points(self, poses: dict) -> np.ndarray:
        pts = []
        for n in L.PARTS:
            c, r = poses[n]
            d = r[:, 0] * (L.LENGTH[n] / 2)
            pts += [c + d, c - d]
        return np.array(pts)

    def path(self) -> float | None:
        """The longest path any part's end point has travelled over the last 0.5 s (m); None until 0.5 s of this
        trial has been seen."""
        if len(self.history) < self.window + 1:
            return None
        pts = np.array(self.history)  # (11, 8, 3)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=2).sum(axis=0).max())

    def still(self) -> bool:
        p = self.path()
        return p is not None and p <= L.STILL_PATH + L.EPS

    def update(self, states, arms=None) -> Result:
        if not self.valid_states(states):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(states, arms)
            return self.last
        poses = self.poses(states)
        now = {n: (poses[n][0], self._velocity(n, states[n])) for n in L.PARTS}
        if self.prev is not None and self._teleported(now) and self.at_start(poses):
            self.reset()  # the simulator put every part back on its mark: a new episode (rigstate's on_reset hook
            #               normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.history.append(self.end_points(poses))
        self.last = self.result(states, arms)
        return self.last

    def result(self, states, arms=None) -> Result:
        if not self.valid_states(states):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        score, why, d = evaluate(self.poses(states), arms)
        path = self.path()
        still = path is not None and path <= L.STILL_PATH + L.EPS
        d.update(valid_state=True, reason=why, path_mm=_mm(path), still=still, arms_unavailable=self.arms_unavailable,
                 bad_steps=self.bad_steps)
        return Result(score, bool(score == 1.0 and still), d)
