"""The scorer, as plain functions of the two bindings' states and the arms' state. NumPy only; no simulator needed.

The task is judged on the end state: each binding flat on the board, its disc over an insert grid and centred on the
centre line, turned to its angle, still and let go, and the two discs the right distance apart and centred on the
reference cross. The only thing the scorer keeps from step to step is the last 0.5 s of each binding's pose, to judge
stillness by the path its centre travelled (never by the reported velocity, which chatters while a body rests). There
are no whole-trial latches. It is fed one observation per control step by ``task.success()``.

    Tracker.update(binding_nose, binding_tail, arms) -> Result(score, success, details)

A binding's state is ``mechkit.AssemblyState``: ``origin`` (its frame's origin, the centre of the baseplate's
underside, which is the disc's centre) and ``quat_xyzw`` (its orientation) are all the scorer reads. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point midway between each gripper's
finger tips) and ``opening`` (0 closed .. 1 open), or None when the arms could not be read: then no binding counts as
let go. 'Let go' is judged from each gripper's finger zone (layout.py): every point within a radius (half the finger
gap plus the pad's thickness, and half its width) of the 38 mm segment from the tip point back toward the TCP, so a
finger resting on any part of a binding, even with the gripper open, is not let go. A binding state with a non-finite
number, a zero quaternion or a missing field (a simulator blow-up) scores 0 for that step and leaves the tracker as
it was.

Which binding is which is decided at every step from where they are, not from where they started: the one whose
centre is nearer the nose (larger y) is the front binding (target +15 deg), the other the back binding (-15 deg).
The two are identical, so swapping them changes nothing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

NAMES = ("binding_nose", "binding_tail")
STARTS = {"binding_nose": L.NOSE_START, "binding_tail": L.TAIL_START}
ARMS = ("left", "right")


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


def wrap_deg(a: float) -> float:
    """An angle in degrees wrapped to (-180, 180]."""
    a = math.fmod(float(a) + 180.0, 360.0)
    if a <= 0.0:
        a += 360.0
    return a - 180.0


def valid(s) -> bool:
    """The binding's origin and orientation present, every number finite and the quaternion not degenerate."""
    try:
        o = np.asarray(getattr(s, "origin"), dtype=float).ravel()
        q = np.asarray(getattr(s, "quat_xyzw"), dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return False
    if o.size != 3 or q.size != 4 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(q))):
        return False
    return bool(np.linalg.norm(q) > 0.5)


@dataclass(frozen=True)
class Pose:
    origin: np.ndarray  # the binding's centre: the baseplate's underside centre (m)
    rot: np.ndarray  # its axes in the world (columns u, v, z)
    heading: float  # angle of u seen from above, degrees in (-180, 180], 0 = +x, positive toward the nose (+y)
    tilt: float  # angle between its z axis and the vertical, degrees 0..180 (not folded: upside down is 180)


def pose(s) -> Pose:
    r = rotation(s.quat_xyzw)
    heading = wrap_deg(math.degrees(math.atan2(r[1, 0], r[0, 0])))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))
    return Pose(np.asarray(s.origin, dtype=float).copy(), r, heading, tilt)


def read_arms(arms) -> list[tuple[np.ndarray, np.ndarray, float]] | None:
    """Both grippers as (tcp, tips, opening), or None when the arms are unreadable (None, a side missing, or a
    number missing or not finite)."""
    if not isinstance(arms, dict):
        return None
    out = []
    try:
        for side in ARMS:
            a = arms[side]
            tcp = np.asarray(a.tcp, dtype=float).ravel()
            tips = np.asarray(a.tips, dtype=float).ravel()
            opening = float(a.opening)
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return None
            if not math.isfinite(opening):
                return None
            out.append((tcp, tips, opening))
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return out


def arm_tips(arms) -> list[np.ndarray] | None:
    """Both grippers' finger-tip points, or None when the arms are unreadable."""
    read = read_arms(arms)
    return None if read is None else [tips for _, tips, _ in read]


def part_distances(points, b: Pose) -> np.ndarray:
    """Distance from each world point to the nearest part of a binding (each part a box in the binding's frame; the
    disc as its bounding box). 0 inside a part."""
    local = (np.atleast_2d(np.asarray(points, dtype=float)) - b.origin) @ b.rot  # rows: rot.T @ (p - origin)
    best = np.full(len(local), math.inf)
    for _, centre, size in L.PART_BOXES:
        q = np.abs(local - np.asarray(centre, dtype=float)) - np.asarray(size, dtype=float) / 2
        best = np.minimum(best, np.linalg.norm(np.maximum(q, 0.0), axis=1))
    return best


def part_distance(p, b: Pose) -> float:
    return float(part_distances(p, b)[0])


def zone_radius(opening: float) -> float:
    """From the gripper's axis to a finger pad's outer corner (m): half the gap plus the pad's thickness, and half
    its width across. 56.6 mm fully open, 45.1 mm at 70 mm, 15.7 mm closed."""
    gap = float(np.clip(opening, 0.0, 1.0)) * L.FULL_GAP
    return math.hypot(gap / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH)


def zone_gap(arm, b: Pose) -> float:
    """How far one gripper's finger zone is from the nearest part of the binding (m; negative: they overlap). The
    zone's axis runs from the tip point 38 mm back toward the TCP."""
    tcp, tips, opening = arm
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    pts = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, L.ZONE_SAMPLES), u)
    return float(part_distances(pts, b).min()) - zone_radius(opening)


def hands_gap(arms, b: Pose) -> float | None:
    """The nearer of the two grippers' finger zones to the binding (m), or None (arms unreadable)."""
    read = read_arms(arms)
    if read is None:
        return None
    return min(zone_gap(a, b) for a in read)


def tip_distance(arms, b: Pose) -> float | None:
    """The nearer of the two grippers' finger-tip points to any part of the binding (m), or None (arms unreadable).
    For the trace only: 'let go' uses the finger zones (hands_gap)."""
    tips = arm_tips(arms)
    if tips is None:
        return None
    return min(part_distance(p, b) for p in tips)


def over_grid(b: Pose) -> bool:
    """The disc's centre over one of the two insert grids: |y| 140 to 260 mm."""
    return bool(L.GRID_Y[0] - L.EPS <= abs(float(b.origin[1])) <= L.GRID_Y[1] + L.EPS)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def tier(n_set: int, w: float, c: float) -> float:
    """The score from how many bindings are set and, with both set, the stance width w and its centre c (m)."""
    if n_set <= 0:
        return 0.0
    if n_set == 1:
        return 0.25
    if L.W_TIGHT[0] - L.EPS <= w <= L.W_TIGHT[1] + L.EPS and abs(c) <= L.C_TIGHT + L.EPS:
        return 1.0
    if L.W_LOOSE[0] - L.EPS <= w <= L.W_LOOSE[1] + L.EPS and abs(c) <= L.C_LOOSE + L.EPS:
        return 0.75
    return 0.5


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


class Tracker:
    """The last 0.5 s of both bindings' poses (for stillness); everything else is read from the current state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.resets = 0  # fallback resets seen (teleports back to the start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)  # {name: (origin, heading)} per valid step
        self.prev = None
        self.last = Result(0.0, False, {})

    # -- reset fallback ----------------------------------------------------------------------------------------
    @staticmethod
    def at_start(sample) -> bool:
        """Both bindings on their start poses, within 5 mm and 2 deg."""
        for name, (origin, heading) in sample.items():
            x, y, a = STARTS[name]
            if float(np.linalg.norm(origin - np.array([x, y, L.BOARD_TOP]))) > L.RESET_START_TOL:
                return False
            if abs(wrap_deg(heading - a)) > L.RESET_START_TURN_DEG:
                return False
        return True

    @classmethod
    def teleported(cls, prev, now) -> bool:
        """The simulator put both bindings back on their start poses in one step: a new episode. At least one of
        them jumped 50 mm or turned 20 deg in that step (a gradual return, or a jump elsewhere, is not a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now[n][0] - prev[n][0])) > L.RESET_JUMP
                   or abs(wrap_deg(now[n][1] - prev[n][1])) > L.RESET_TURN_DEG for n in NAMES)

    # -- per step ------------------------------------------------------------------------------------------------
    def update(self, nose, tail, arms) -> Result:
        states = {"binding_nose": nose, "binding_tail": tail}
        if not all(valid(s) for s in states.values()):  # a blow-up: 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(nose, tail, arms)
            return self.last
        poses = {n: pose(s) for n, s in states.items()}
        sample = {n: (p.origin, p.heading) for n, p in poses.items()}
        if self.prev is not None and self.teleported(self.prev, sample):
            self.reset()
            self.resets += 1
        self.steps += 1
        self.hist.append(sample)
        self.prev = sample
        self.last = self.result(nose, tail, arms)
        return self.last

    def motion(self, name: str) -> tuple[float, float] | None:
        """(path of the centre in m, summed heading change in deg) over the last 0.5 s, or None before 0.5 s of
        valid states has been seen."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        path = sum(float(np.linalg.norm(b[name][0] - a[name][0])) for a, b in zip(h, h[1:]))
        turn = sum(abs(wrap_deg(b[name][1] - a[name][1])) for a, b in zip(h, h[1:]))
        return path, turn

    def result(self, nose, tail, arms) -> Result:
        states = {"binding_nose": nose, "binding_tail": tail}
        if not all(valid(s) for s in states.values()):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        poses = {n: pose(s) for n, s in states.items()}
        front = max(NAMES, key=lambda n: (float(poses[n].origin[1]), n == "binding_nose"))
        back = NAMES[1] if front == NAMES[0] else NAMES[0]
        targets = {front: L.FRONT_TARGET_DEG, back: L.BACK_TARGET_DEG}
        per, n_set = {}, 0
        for n in NAMES:
            b = poses[n]
            z_above = float(b.origin[2] - L.BOARD_TOP)
            x_off = float(b.origin[0] - L.LINE_X)
            m = self.motion(n)
            tip = tip_distance(arms, b)
            zone = hands_gap(arms, b)
            err = wrap_deg(b.heading - targets[n])
            flat = abs(z_above) <= L.FLAT_Z_TOL + L.EPS and b.tilt <= L.FLAT_TILT_DEG + L.EPS_DEG
            gridded = over_grid(b)
            centred = abs(x_off) <= L.CENTRE_TOL + L.EPS
            angled = abs(err) <= L.ANGLE_TOL_DEG + L.EPS_DEG
            still = m is not None and m[0] <= L.STILL_PATH + L.EPS and m[1] <= L.STILL_TURN_DEG + L.EPS_DEG
            let_go = zone is not None and zone >= L.LET_GO - L.EPS
            ok = bool(flat and gridded and centred and angled and still and let_go)
            n_set += ok
            per[n] = {
                "role": "front" if n == front else "back",
                "origin": [round(float(v), 4) for v in b.origin],
                "heading_deg": round(b.heading, 2),
                "angle_err_deg": round(err, 2),
                "tilt_deg": round(b.tilt, 2),
                "z_above_mm": _mm(z_above),
                "x_off_mm": _mm(x_off),
                "path_mm": None if m is None else _mm(m[0]),
                "turn_deg": None if m is None else round(m[1], 3),
                "tip_mm": _mm(tip),
                "zone_mm": _mm(zone),
                "flat": bool(flat), "over_grid": gridded, "centred": bool(centred), "angled": bool(angled),
                "still": bool(still), "let_go": bool(let_go), "set": ok,
            }
        w = float(poses[front].origin[1] - poses[back].origin[1])
        c = float(poses[front].origin[1] + poses[back].origin[1]) / 2
        score = tier(n_set, w, c)
        details = {"valid_state": True, "n_set": n_set, "W_mm": _mm(w), "C_mm": _mm(c), "tier": score,
                   "arms_readable": read_arms(arms) is not None, **per}
        return Result(score, bool(score == 1.0), details)
