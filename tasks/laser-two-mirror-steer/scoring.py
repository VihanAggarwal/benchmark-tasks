"""The scorer, as plain functions of the objects' states and the arms' state. NumPy only; no simulator needed.

The laser is always on. Every control step the scorer traces its beam through the scene: from the laser's
aperture, 40 mm above the table, along +x. The beam reflects off a mirror's 40 x 40 mm window, in the middle of
either 60 x 60 mm face (law of reflection, in 3-D, from the mirror's full orientation), stops on anything else it
meets (the block, the dump, the detector, the backstop, the laser, a mirror's 10 mm rim round the window or its
20 mm edge, the table top), and is *stray* if it leaves the table's outline without being stopped. Every object is
its true (oriented) box: the mirrors and the block from their states, the fixed devices from layout.py. The arms
are not obstacles.

The rules hold for the whole trial, so the scorer keeps a little state and is fed one observation per control
step by ``task.success()``:

    Tracker.update(objects, arms) -> Result(score, success, details)

``objects`` maps names to anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has; only the two
mirrors and the block are read. ``arms`` is ``rigstate.arms()`` (``{"left": Arm, "right": Arm}`` with ``tcp``,
``tips`` and ``opening``, 0 closed .. 1 open) or ``None`` when the arms could not be read. The scorer fails
closed: a step whose arms cannot be read (None, a side missing, a non-finite number) is a step on which the
grippers have NOT let go, so it can never be credited with any tier above 0; such steps are counted in
``arms_unavailable``. 'Let go' is judged from finger geometry only (the finger zone more than 20 mm from both mirrors
and the block, whatever the opening) and must hold on every control step of the last 0.5 s; every tier above 0 (each
credits a mirror on the beam's path) needs it, so a mirror still held earns nothing. A state with a
non-finite number or a zero quaternion (a simulator blow-up), or a missing object, scores 0 for that step, restarts
the let-go count and otherwise leaves the tracker as it was.

Breaches (any one, at any step, scores 0 for the trial): the beam is stray; a mirror moves more than 3 mm or turns
more than 1 degree while the beam is on it, measured from its place at the last control step before the beam came
onto it (its last place in the dark) to every step the beam is on it and the first step after the beam leaves it,
so a mirror swung into or out of the beam between two control steps counts, and a mirror moved behind the block
and then lit again starts afresh; a mirror whose box passed through the beam's current path between two control
steps (the volume it swept, tested exactly for each leg of the beam) counts as lit on that step, so a mirror swept
right across the live beam between two samples is charged like one standing in it; a mirror or the block falls off
the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
EPS = 1e-9
SWEEP_TURN_DEG = 2.0  # a mirror's motion between two control steps is tested against the beam in pieces of this turn


# -- states ---------------------------------------------------------------------------------------------------
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
        parts = [np.asarray(getattr(s, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    return bool(parts[0].size == 3 and parts[1].size == 4 and np.linalg.norm(parts[1]) > 0.5
                and parts[2].size == 3 and parts[3].size == 3 and parts[4].size == 3 and parts[5].size == 3)


def loose_states(objects) -> dict | None:
    """The two mirrors' and the block's states, or None if one is missing or not a valid state."""
    try:
        out = {name: objects[name] for name in L.LOOSE}
    except (KeyError, TypeError):
        return None
    return out if all(valid(s) for s in out.values()) else None


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


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def off_table(s) -> bool:
    return bool(float(np.asarray(s.low, dtype=float)[2]) < L.TOP - L.OFF_TABLE_DROP)


def turn_deg(r_a, r_b) -> float:
    """The angle of the rotation that takes orientation r_a to r_b (degrees, 0..180)."""
    c = (float(np.trace(np.asarray(r_a).T @ np.asarray(r_b))) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def mirror_yaw_deg(s) -> float:
    """Heading of a mirror's face normal (its local x) seen from above, folded to (-90, 90]: both faces reflect,
    so a mirror turned 180 degrees is the same mirror."""
    r = rotation(s.quat_xyzw)
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    while yaw <= -90.0:
        yaw += 180.0
    while yaw > 90.0:
        yaw -= 180.0
    return yaw


def normal_tilt_deg(s) -> float:
    """How far a mirror's face normal points up or down from horizontal (degrees): 0 for a mirror standing
    upright; a tilted mirror sends the beam up or down."""
    r = rotation(s.quat_xyzw)
    return float(np.degrees(np.arcsin(np.clip(abs(r[2, 0]), 0.0, 1.0))))


# -- the beam -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Solid:
    """An oriented box the beam can meet. ``kind``: 'mirror' (the window in the middle of each face normal to its
    local x reflects, the rest stops the beam), 'detector' (its local +x face is the sensitive face) or 'stop'."""

    name: str
    kind: str
    centre: np.ndarray
    rot: np.ndarray  # columns: the box's axes in the world
    half: np.ndarray


def _fixed(name: str, kind: str, x: float, y: float, size) -> Solid:
    return Solid(name, kind, np.array([x, y, L.TOP + size[2] / 2]), np.eye(3), np.array(size, dtype=float) / 2)


FIXED = (
    _fixed("laser", "stop", L.LASER_X, L.LASER_Y, L.LASER_SIZE),
    _fixed("beam_dump", "stop", L.DUMP_X, L.DUMP_Y, L.DUMP_SIZE),
    _fixed("detector", "detector", L.DET_X, L.DET_Y, L.DET_SIZE),
    _fixed("backstop", "stop", L.BACK_X, L.BACK_Y, L.BACK_SIZE),
)  # the 2 mm lines on the table are part of the table top for the beam


def loose_solid(name: str, s) -> Solid:
    return Solid(name, "mirror" if name in L.MIRRORS else "stop", np.asarray(s.pos, dtype=float),
                 rotation(s.quat_xyzw), np.array(L.SIZES[name], dtype=float) / 2)


def scene_solids(states: dict) -> list[Solid]:
    return [*FIXED, *(loose_solid(n, states[n]) for n in L.LOOSE)]


def in_window(p, box: Solid) -> bool:
    """Whether a point on a mirror's face is inside its window: at least MIRROR_FRAME inside the face's edges."""
    q = box.rot.T @ (np.asarray(p, dtype=float) - box.centre)
    lim = box.half - L.MIRROR_FRAME
    return bool(abs(q[1]) <= lim[1] + EPS and abs(q[2]) <= lim[2] + EPS)


def ray_box(o, d, box: Solid):
    """Where the ray o + t d (t > 0) first meets ``box``: (t, axis, sign) of the face it enters through (axis in
    the box's own frame, sign -1 or +1); (0.0, -1, 0.0) if o is inside the box; None if it misses."""
    ol = box.rot.T @ (np.asarray(o, dtype=float) - box.centre)
    dl = box.rot.T @ np.asarray(d, dtype=float)
    t0, t1, axis, sign = -math.inf, math.inf, -1, 0.0
    for i in range(3):
        h = box.half[i]
        if abs(dl[i]) < 1e-12:
            if abs(ol[i]) > h:
                return None
            continue
        a, b = (-h - ol[i]) / dl[i], (h - ol[i]) / dl[i]
        enter = -1.0 if a < b else 1.0  # moving toward +: enters through the -h face
        if a > b:
            a, b = b, a
        if a > t0:
            t0, axis, sign = a, i, enter
        t1 = min(t1, b)
        if t0 > t1:
            return None
    if t1 <= EPS:
        return None  # behind the ray
    if t0 <= EPS:
        return 0.0, -1, 0.0  # the ray starts inside the box
    return t0, axis, sign


def _exit_distance(o, d) -> float:
    """How far along d the ray leaves the table's outline seen from above (0 if it starts outside it)."""
    (x0, x1), (y0, y1) = L.TABLE_X, L.TABLE_Y
    if not (x0 <= o[0] <= x1 and y0 <= o[1] <= y1):
        return 0.0
    t = math.inf
    for k, (lo, hi) in ((0, (x0, x1)), (1, (y0, y1))):
        if d[k] > 1e-12:
            t = min(t, (hi - o[k]) / d[k])
        elif d[k] < -1e-12:
            t = min(t, (lo - o[k]) / d[k])
    return t


@dataclass
class Beam:
    points: list  # the aperture, every reflection point and the end point (world, m)
    reflections: list  # the mirrors' names, in the order the beam meets them
    end: str  # 'stray', 'table', 'trapped' or the name of the object that stops it
    lit: set  # the mirrors the beam meets (reflected, or stopped on a rim or an edge)
    on_detector: bool = False  # stopped on the detector's sensitive (+x) face
    det_dy: float | None = None  # where on that face: across (y), from its centre (m)
    det_dz: float | None = None  # and up (z), from beam height (m)

    @property
    def mirrors(self) -> set:
        return set(self.reflections)

    @property
    def det_off(self) -> float | None:
        return None if self.det_dy is None else float(math.hypot(self.det_dy, self.det_dz))

    @property
    def stray(self) -> bool:
        return self.end == "stray"


def trace_beam(solids) -> Beam:
    """Follow the beam from the laser's aperture until it is stopped, leaves the table, or has reflected more than
    MAX_BOUNCES times (trapped between the mirrors: it stays on the table)."""
    o = np.array([L.APERTURE_X, L.LASER_Y, L.BEAM_Z])
    d = np.array([1.0, 0.0, 0.0])
    beam = Beam([o.copy()], [], "trapped", set())
    last = "laser"  # the beam leaves the laser's front face
    for _ in range(L.MAX_BOUNCES + 1):
        best = None
        for s in solids:
            if s.name == last:  # a flat face cannot be met again straight after leaving it
                continue
            hit = ray_box(o, d, s)
            if hit is not None and (best is None or hit[0] < best[0][0]):
                best = (hit, s)
        t_obj = best[0][0] if best else math.inf
        t_tab = (L.TOP - o[2]) / d[2] if d[2] < -1e-12 else math.inf
        t_out = _exit_distance(o, d)
        t = min(t_obj, t_tab, t_out)
        if not math.isfinite(t):  # straight up, nothing above it
            beam.end = "stray"
            beam.points.append(o + 1.0 * d)
            return beam
        p = o + t * d
        if t_out <= t_obj and t_out <= t_tab:
            beam.end = "stray"
            beam.points.append(p)
            return beam
        if t_tab <= t_obj:
            beam.end = "table"
            beam.points.append(p)
            return beam
        (_, axis, sign), s = best
        if s.kind == "mirror":
            beam.lit.add(s.name)
            if axis == 0 and in_window(p, s):  # a 60 x 60 mm face, inside its 40 x 40 mm window: reflect
                n = sign * s.rot[:, 0]
                d = d - 2.0 * float(d @ n) * n
                d = d / np.linalg.norm(d)
                beam.reflections.append(s.name)
                beam.points.append(p)
                o, last = p, s.name
                continue
        beam.end = s.name
        beam.points.append(p)
        if s.kind == "detector" and axis == 0 and sign > 0:
            beam.on_detector = True
            beam.det_dy = float(p[1] - L.DET_Y)
            beam.det_dz = float(p[2] - L.BEAM_Z)
        return beam
    beam.points.append(o.copy())
    return beam  # trapped


# -- a mirror swept through the beam between two control steps ------------------------------------------------
def _clip(poly: list, a: float, b: float, c: float) -> list:
    """Clip a convex polygon in the (t, s) plane to the half-plane a + b t + c s <= 0 (Sutherland-Hodgman)."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        fp, fq = a + b * p[0] + c * p[1], a + b * q[0] + c * q[1]
        if fp <= 0.0:
            out.append(p)
        if (fp < 0.0 < fq) or (fq < 0.0 < fp):
            k = fp / (fp - fq)
            out.append((p[0] + k * (q[0] - p[0]), p[1] + k * (q[1] - p[1])))
    return out


def segment_meets_slide(p0, p1, c0, c1, rot, half) -> bool:
    """Whether the segment p0-p1 meets the volume a box (axes ``rot``, half sizes ``half``) sweeps while its centre
    slides from c0 to c1 without turning: is there t, s in [0, 1] with p0 + t (p1 - p0) inside the box centred at
    c0 + s (c1 - c0)? Exact: the conditions are linear in (t, s), so the unit square is clipped by the box's six
    faces and the segment meets the swept volume unless nothing is left (touching counts)."""
    rot = np.asarray(rot, dtype=float)
    a = rot.T @ (np.asarray(p0, dtype=float) - np.asarray(c0, dtype=float))
    b = rot.T @ (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    c = -(rot.T @ (np.asarray(c1, dtype=float) - np.asarray(c0, dtype=float)))
    poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for i in range(3):
        h = float(half[i]) + EPS
        for sg in (1.0, -1.0):  # sg (a + b t + c s) - h <= 0
            poly = _clip(poly, sg * a[i] - h, sg * b[i], sg * c[i])
            if not poly:
                return False
    return True


def _slerp(q0, q1, f: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(q0 @ q1)
    if dot < 0.0:  # the shorter way round
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + f * (q1 - q0)
    else:
        th = math.acos(min(1.0, dot))
        q = (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def swept_through_beam(name: str, pos0, quat0, pos1, quat1, beam: Beam) -> bool:
    """Whether a loose part's box, going from (pos0, quat0) at the previous control step to (pos1, quat1) at this
    one, passed through the beam's current path on the way: the step is cut into pieces of at most SWEEP_TURN_DEG of
    turn (the shorter way round), and each piece's slide is tested, exactly, at its start and end orientations
    against every leg of the beam."""
    half = np.array(L.SIZES[name], dtype=float) / 2
    turn = turn_deg(rotation(quat0), rotation(quat1))
    n = max(1, int(math.ceil(turn / SWEEP_TURN_DEG)))
    pos0, pos1 = np.asarray(pos0, dtype=float), np.asarray(pos1, dtype=float)
    legs = list(zip(beam.points[:-1], beam.points[1:]))
    prev_c, prev_r = pos0, rotation(quat0)
    for k in range(1, n + 1):
        f = k / n
        c, r = pos0 + f * (pos1 - pos0), rotation(_slerp(quat0, quat1, f))
        for p0, p1 in legs:
            if any(segment_meets_slide(p0, p1, prev_c, c, rr, half) for rr in (prev_r, r)):
                return True
        prev_c, prev_r = c, r
    return False


# -- the arms -------------------------------------------------------------------------------------------------
def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    dd = np.maximum(0.0, np.maximum(np.asarray(lo, dtype=float) - p, p - np.asarray(hi, dtype=float)))
    return float(np.linalg.norm(dd))


def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across (rigstate gives no heading, so the fingers may be turned any way about the axis)."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def _upright_gap(arm, r: float, lo, hi) -> float:
    """How far an upright cylinder that contains the finger zone is from a world-aligned box (0 if they overlap).
    For a gripper pointing straight down it is the finger zone itself: a cylinder of radius r about the axis, from
    the tips to FINGER_LEN up, so nothing below the tips counts. A tilted gripper's zone is covered by a wider and
    taller upright cylinder (its axis leans |u_xy| per metre; each end disc reaches r |u_xy| up and down)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    top = tips - L.FINGER_LEN * u  # the zone's far end, up the gripper
    lean = float(math.hypot(u[0], u[1]))
    centre = (tips[:2] + top[:2]) / 2
    radius = r + L.FINGER_LEN * lean / 2
    z0, z1 = min(tips[2], top[2]) - r * lean, max(tips[2], top[2]) + r * lean
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    across = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo[:2] - centre, centre - hi[:2])))) - radius
    up = max(float(lo[2]) - z1, z0 - float(hi[2]))
    return float(math.hypot(max(0.0, across), max(0.0, up)))


def zone_gap(arm, lo, hi) -> float:
    """How far a gripper's finger zone (a cylinder of radius zone_radius about its axis, from the tips to FINGER_LEN
    up) is from a world-aligned box, never more than the true distance: the larger of two lower bounds, a capsule
    about the axis (which reaches r past the tips) and an upright cylinder that contains the zone (exact for a
    gripper pointing straight down). 0 or negative: they overlap; NaN if either cannot be computed."""
    pts, r = finger_zone(arm)
    capsule = min(point_box_distance(p, lo, hi) for p in pts) - r
    upright = _upright_gap(arm, r, lo, hi)
    if not (math.isfinite(capsule) and math.isfinite(upright)):
        return math.nan
    return max(capsule, upright)


def let_go(arm, states: dict) -> bool:
    """The gripper holds nothing and touches nothing: its finger zone is more than 20 mm from both mirrors and the
    block, open or closed (an open hand resting on a part has not let go). A gap that cannot be computed (an error
    or a non-finite number) is not let go."""
    try:
        gaps = [zone_gap(arm, states[n].low, states[n].high) for n in L.LOOSE]
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return all(math.isfinite(g) and g > L.LET_GO_GAP for g in gaps)


def finger_clearance(arm, states: dict) -> float | None:
    """The smallest gap from a gripper's finger zone to the mirrors and the block (m; None if it cannot be computed),
    for the trace."""
    try:
        g = min(zone_gap(arm, states[n].low, states[n].high) for n in L.LOOSE)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return g if math.isfinite(g) else None


def grippers_released(arms, states: dict) -> bool:
    """Both grippers let go on this step; unreadable arms are never released (fail closed)."""
    if not arms_valid(arms):
        return False
    return let_go(arms["left"], states) and let_go(arms["right"], states)


# -- the tracker ----------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "beam_stray": "the beam left the table without being stopped",
    "lit_mirror_moved": "a mirror moved more than 3 mm or turned more than 1 degree while the beam was on it (from its "
                        "last place in the dark)",
    "object_off_table": "a mirror or the block fell off the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.release_need = max(1, int(round(L.RELEASE_HOLD_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.lit_ref: dict[str, tuple] = {}  # mirror -> (pos, rot, quat), its last pose before this lit spell
        self.last_pose: dict[str, tuple] = {}  # mirror -> (pos, rot, quat) at the previous valid step
        self.swept: list[str] = []  # mirrors whose box passed through the beam between the last two steps
        self.max_lit_move = 0.0  # m: the most a mirror moved in one lit spell (from its last place in the dark)
        self.max_lit_turn = 0.0  # degrees
        self.first_lit: dict[str, int] = {}  # mirror -> the first control step the beam met it
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms (never let go)
        self.release_streak = 0  # consecutive control steps, up to this one, with both grippers let go
        self.bad_steps = 0  # control steps with a missing or non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of the loose objects at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(states: dict) -> bool:
        """The starting layout: both mirrors and the block on their starting places (within START_TOL, which covers
        a seeded reset's jitter), upright on the table and still."""
        for name in L.LOOSE:
            s = states[name]
            x, y = L.STARTS[name]
            if np.hypot(s.pos[0] - x, s.pos[1] - y) >= L.START_TOL:
                return False
            if abs(float(s.low[2]) - L.TOP) >= 0.003 or rotation(s.quat_xyzw)[2, 2] < math.cos(math.radians(1.0)):
                return False
            if not is_resting(s):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    def update(self, objects, arms=None) -> Result:
        states = loose_states(objects)
        if states is None:  # a blow-up or a missing object: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.release_streak = 0  # nothing could be judged on this step: the let-go count starts again
            self.last = self.result(objects, arms)
            return self.last
        now = [(np.array(states[n].pos, dtype=float), np.array(states[n].lin_vel, dtype=float)) for n in L.LOOSE]
        if self.prev is not None and self._teleported(now) and self.at_start(states):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        beam = trace_beam(scene_solids(states))
        if beam.stray:
            self._flag("beam_stray")
        self.swept = []
        for m in L.MIRRORS:
            quat = np.array(states[m].quat_xyzw, dtype=float)
            pos, rot = np.array(states[m].pos, dtype=float), rotation(quat)
            lit = m in beam.lit
            if not lit and m in self.last_pose:
                # Not on the beam at either sampled step, but its box may have passed through the beam's current
                # path between them (0.5 m/s is 25 mm a step, more than a mount's 20 mm): that counts as lit.
                p0, _, q0 = self.last_pose[m]
                lit = swept_through_beam(m, p0, q0, pos, quat, beam)
                if lit:
                    self.swept.append(m)
            if lit:
                self.first_lit.setdefault(m, self.steps)
                # A new lit spell is measured from where the mirror stood at the last step in the dark, so a mirror
                # swung into the beam between two control steps has moved while lit.
                ref = self.lit_ref.setdefault(m, self.last_pose.get(m, (pos, rot, quat)))
            else:
                # Dark again: the first dark step still counts (a mirror swung out of the beam between two steps
                # moved while lit); the next lit spell starts afresh.
                ref = self.lit_ref.pop(m, None)
            self.last_pose[m] = (pos, rot, quat)
            if ref is None:
                continue
            move, turn = float(np.linalg.norm(pos - ref[0])), turn_deg(ref[1], rot)
            self.max_lit_move = max(self.max_lit_move, move)
            self.max_lit_turn = max(self.max_lit_turn, turn)
            if move > L.LIT_MOVE or turn > L.LIT_TURN_DEG:
                self._flag("lit_mirror_moved")
        if any(off_table(states[n]) for n in L.LOOSE):
            self._flag("object_off_table")
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.release_streak = self.release_streak + 1 if grippers_released(arms, states) else 0
        self.last = self._result(states, arms, beam)
        return self.last

    def result(self, objects, arms=None) -> Result:
        """The score of the current state given what has happened so far (does not change the tracker)."""
        states = loose_states(objects)
        if states is None:
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        return self._result(states, arms, trace_beam(scene_solids(states)))

    def _result(self, states: dict, arms, beam: Beam) -> Result:
        have_arms = arms_valid(arms)
        released_now = grippers_released(arms, states)  # False when the arms cannot be read (fail closed)
        # Every tier that credits a mirror on the beam's path needs both grippers let go on this step and on every
        # step of the last 0.5 s (the streak is counted by update(), which task.success() runs before task.score()
        # every control step): a mirror still held, or a hand still on a part, earns nothing.
        released = released_now and self.release_streak >= self.release_need
        both = beam.mirrors >= set(L.MIRRORS)
        gone = any(off_table(states[n]) for n in L.LOOSE)
        off = beam.det_off
        if self.breach or beam.stray or gone or not released:
            score = 0.0
        elif both and beam.on_detector and off <= L.CENTRE_TIGHT:
            score = 1.0
        elif both and beam.on_detector:
            score = 0.75
        elif both:
            score = 0.5
        elif beam.mirrors:
            score = 0.25
        else:
            score = 0.0
        still = all(is_resting(states[n]) for n in L.LOOSE)
        success = bool(score == 1.0 and still)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "beam_end": beam.end,
            "reflections": list(beam.reflections),
            "lit": sorted(beam.lit),
            "on_detector": beam.on_detector,
            "det_off_mm": _mm(off),
            "det_dy_mm": _mm(beam.det_dy),
            "det_dz_mm": _mm(beam.det_dz),
            "beam_points_mm": [[round(float(v) * 1000, 1) for v in p] for p in beam.points],
            "max_lit_move_mm": _mm(self.max_lit_move),
            "max_lit_turn_deg": round(self.max_lit_turn, 3),
            "first_lit_step": dict(self.first_lit),
            "swept_through_beam": list(self.swept),
            "mirror_left_yaw_deg": round(mirror_yaw_deg(states["mirror_left"]), 3),
            "mirror_right_yaw_deg": round(mirror_yaw_deg(states["mirror_right"]), 3),
            "mirror_left_tilt_deg": round(normal_tilt_deg(states["mirror_left"]), 3),
            "mirror_right_tilt_deg": round(normal_tilt_deg(states["mirror_right"]), 3),
            "off_table": gone,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "grippers_let_go": released_now,
            "finger_gap_mm": ({s: _mm(finger_clearance(arms[s], states)) for s in ("left", "right")}
                              if have_arms else None),
            "let_go_steps": self.release_streak,
            "grippers_released": released,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "still": still,
        }
        return Result(score, success, details)
