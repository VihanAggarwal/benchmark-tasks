"""The scorer, as plain functions of the three rings' states and the arms' state. NumPy only; no simulator needed.

One rule is about the whole trial, not only its end: neither a ring nor a gripper may ever touch the wax crown or
its sprue (soft wax dents at the lightest touch). So the scorer keeps a little state (whether and when the wax was
touched, how many steps the arms could not be read, each ring's last positions for stillness) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(rings, arms) -> Result(score, success, details)

``rings`` maps each ring's name to anything with ``pos`` (the centre of its world-aligned bounding box, which for a
square tube is its geometric centre at any orientation), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high``, as ``yam_arena.task_file.ObjectState`` (and mechkit's AssemblyState) has. ``arms`` is ``rigstate.arms()``
passed through ``with_turn`` (task.py does this with gripturn's link poses): ``{"left": Gripper, "right": Gripper}``
with ``tcp``, ``tips`` (the point between the finger tips), ``opening`` (0 closed .. 1 open, pads 93.9 mm apart) and
``rot`` (the gripper's frame, or None when its turn about its axis is not known), or ``None`` when the arms could not
be read: then the gripper part of the touch rule is not applied on that step, every ring counts as held on it (fail
closed), and the step is counted in ``arms_unavailable``; if that is more than 1% of the steps the trial scores 0
(the touch rule could not be judged). A ring state with a non-finite number or a zero quaternion (a simulator
blow-up), or a missing ring, scores 0 for that step and leaves the tracker as it was.

Touching the wax, at any moment, either of:
- (a) a ring wall (its real 4 mm wall box, posed as the ring is) within 0.5 mm of the crown's box or of the 4 x 4 mm
  square post standing for the round sprue: separating-axis test of each wall against each of them grown 0.5 mm;
- (b) a point of the crown's or the sprue's surface (sampled every 2 mm or finer) inside a finger. Each finger is a
  box 10 mm thick (outside the gap), 40 mm wide and 70 mm long (from the tips up to the finger carriage), posed as
  the gripper is and grown 1 mm on every side. On a step where the gripper's turn about its axis is not known (no
  link pose that agrees with rigstate's tool point and axis), the box is swept round the axis instead (any turn):
  from half the gap less 1 mm to sqrt((gap/2 + 11 mm)^2 + (21 mm)^2) from the axis, from 1 mm below the tips to
  71 mm above them. Either way the space between the fingers is open: a ring held across its outside, or open
  fingers, pass over the wax without touching it, as real fingers would.

The rings are square tubes open at both ends, so either end may be down: 'tilt' is folded (0..90 degrees) and the
'bottom' is whichever end is lower.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
ARMS = ("left", "right")
FORMER_XY = np.array([L.FORMER_X, L.FORMER_Y])
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
_SIGNS2 = np.array(list(itertools.product((-1, 1), repeat=2)), dtype=float)
TIERS = (0.0, 0.25, 0.5, 1.0)
EPS, EPS_DEG = 1e-9, 1e-9  # rounding allowance at the thresholds (m, degrees)


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


def rings_valid(rings) -> bool:
    try:
        return all(valid(rings[n]) for n in L.RING_NAMES)
    except (KeyError, TypeError, AttributeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float).ravel(), np.asarray(a.tips, dtype=float).ravel()
            if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
                return False
            if not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def tilt_deg(quat_xyzw) -> float:
    """Angle between the ring's axis and the vertical (degrees, 0..90): open at both ends, either end may be down."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


# -- a ring's geometry ------------------------------------------------------------------------------------------
def frame(s, h: float) -> tuple[np.ndarray, np.ndarray]:
    """(centre of the lower end's face, rotation whose third column points from the lower end to the upper one)."""
    r = rotation(s.quat_xyzw)
    if r[2, 2] < 0.0:  # upside down: the same tube, turned half a turn about its x axis
        r = r @ np.diag([1.0, -1.0, -1.0])
    return np.asarray(s.pos, dtype=float) - r[:, 2] * (h / 2), r


def to_ring(s, h: float, points) -> np.ndarray:
    """World point(s) in the ring's frame: origin at the centre of its lower end, z' up its axis."""
    o, r = frame(s, h)
    return (np.asarray(points, dtype=float) - o) @ r


def bottom_corners(s, h: float) -> np.ndarray:
    """The four outer corners of the ring's lower end (world, 4 x 3)."""
    o, r = frame(s, h)
    local = np.column_stack([_SIGNS2 * L.RING_OUTER_HALF, np.zeros(4)])
    return o + local @ r.T


def outer_corners(s, h: float) -> np.ndarray:
    half = np.array([L.RING_OUTER_HALF, L.RING_OUTER_HALF, h / 2])
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def top_z(s, h: float) -> float:
    """Height (world z) of the ring's highest point."""
    return float(outer_corners(s, h)[:, 2].max())


def walls(s, h: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The ring's four wall boxes as (centre, rotation, half sizes), posed as the ring is."""
    c, r = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)
    off = L.RING_OUTER_HALF - L.RING_WALL / 2  # 0.028
    hw = L.RING_WALL / 2
    local = (((off, 0.0, 0.0), (hw, L.RING_OUTER_HALF, h / 2)), ((-off, 0.0, 0.0), (hw, L.RING_OUTER_HALF, h / 2)),
             ((0.0, off, 0.0), (L.RING_INNER_HALF, hw, h / 2)), ((0.0, -off, 0.0), (L.RING_INNER_HALF, hw, h / 2)))
    return [(c + r @ np.array(p), r, np.array(hh)) for p, hh in local]


# -- the wax ----------------------------------------------------------------------------------------------------
CROWN_BOX = (np.array(L.CROWN_CENTRE), np.eye(3), np.array(L.CROWN_SIZE) / 2)
SPRUE_BOX = (np.array(L.SPRUE_CENTRE), np.eye(3), np.array(L.SPRUE_POST_HALF))
WAX_BOXES = {"crown": CROWN_BOX, "sprue": SPRUE_BOX}
CROWN_CORNERS = CROWN_BOX[0] + _SIGNS * CROWN_BOX[2]
CROWN_TOP_CORNERS = CROWN_CORNERS[CROWN_CORNERS[:, 2] > CROWN_BOX[0][2]]
WAX_CORNERS = np.vstack([CROWN_CORNERS, SPRUE_BOX[0] + _SIGNS * SPRUE_BOX[2]])  # 16 points


def _box_surface(centre, half, spacing: float) -> np.ndarray:
    axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / spacing - 1e-9)) + 1)) for h in half]
    pts = np.array(list(itertools.product(*axes)))
    on = np.any(np.isclose(np.abs(pts), np.asarray(half)), axis=1)
    return np.asarray(centre) + pts[on]


def _cylinder_surface(centre, radius: float, height: float, spacing: float) -> np.ndarray:
    n_a = max(8, int(math.ceil(2 * math.pi * radius / spacing)))
    n_z = max(2, int(math.ceil(height / spacing - 1e-9)) + 1)
    a = np.linspace(0.0, 2 * math.pi, n_a, endpoint=False)
    z = np.linspace(-height / 2, height / 2, n_z)
    ring = np.array([(radius * math.cos(t), radius * math.sin(t), zz) for zz in z for t in a])
    return np.asarray(centre) + np.vstack([ring, [(0.0, 0.0, -height / 2), (0.0, 0.0, height / 2)]])


WAX_POINTS = np.vstack([_box_surface(L.CROWN_CENTRE, np.array(L.CROWN_SIZE) / 2, L.WAX_SAMPLE),
                        _cylinder_surface(L.SPRUE_CENTRE, L.SPRUE_RADIUS, L.SPRUE_HEIGHT, L.WAX_SAMPLE)])


def sat_gap(a, b, grow: float = 0.0) -> tuple[float, float]:
    """Separating-axis gaps between two boxes (centre, rotation, half sizes): the largest separation of their
    projections over the 15 candidate axes (m), for ``b`` as it is and for ``b`` grown by ``grow`` on every side.
    Negative: they overlap; positive: a lower bound on their distance (exact when a face separates them)."""
    ca, ra, ha = a
    cb, rb, hb = b
    d = np.asarray(cb, dtype=float) - np.asarray(ca, dtype=float)
    cross = np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(9, 3)
    n = np.linalg.norm(cross, axis=1)
    axes = np.vstack([ra.T, rb.T, cross[n > 1e-9] / n[n > 1e-9, None]])
    pb = np.abs(axes @ rb)
    sep = np.abs(axes @ d) - np.abs(axes @ ra) @ ha - pb @ hb
    return float(sep.max()), float((sep - grow * pb.sum(axis=1)).max())


def _near_wax(s, h: float) -> bool:
    reach = math.sqrt(2 * L.RING_OUTER_HALF ** 2 + (h / 2) ** 2) + 0.02
    return float(np.linalg.norm(np.asarray(s.pos, dtype=float) - CROWN_BOX[0])) < reach


def wall_wax(s, h: float) -> tuple[float | None, bool]:
    """(the smallest separating-axis gap between any of the ring's walls and the crown or the sprue post, in m, or
    None when the ring is nowhere near the wax; whether a wall is within 0.5 mm of them: it meets them grown by
    0.5 mm)."""
    if not _near_wax(s, h):
        return None, False
    gaps = [sat_gap(w, box, L.WAX_GAP) for w in walls(s, h) for box in WAX_BOXES.values()]
    return min(g for g, _ in gaps), any(g <= 0.0 for _, g in gaps)


def wall_wax_gap(s, h: float) -> float | None:
    return wall_wax(s, h)[0]


def wall_touches(s, h: float) -> bool:
    return wall_wax(s, h)[1]


# -- the grippers -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gripper:
    """An arm as rigstate reports it, with its turn: ``rot`` is the gripper's frame (columns: the closing axis x, then
    y, then the approach axis z, which points from the tool point to the tips), or None when it is not known."""

    tcp: np.ndarray
    tips: np.ndarray
    opening: float
    rot: np.ndarray | None = None


def gripper_turn(arm, candidates) -> np.ndarray | None:
    """The gripper's frame from candidate (position, quaternion) readings of its ``gripper`` link, or None.

    A reading counts if, taken as (w, x, y, z) or as (x, y, z, w), it puts the tool point (0.134 m along the link's
    z) within 3 mm of the arm's tool point and the link's z within 3 degrees of the arm's tool-point-to-tips axis.
    All readings that count must agree on the closing axis (within 3 degrees, either sense), else None."""
    try:
        tcp = np.asarray(arm.tcp, dtype=float).ravel()
        tips = np.asarray(arm.tips, dtype=float).ravel()
        if tcp.size != 3 or tips.size != 3 or not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips))):
            return None
        axis = tips - tcp
        n = float(np.linalg.norm(axis))
        if n < 1e-4:
            return None
        axis = axis / n
        cos_tol = math.cos(math.radians(L.TURN_AXIS_DEG))
        found = []
        for pos, quat in candidates or ():
            p = np.asarray(pos, dtype=float).ravel()
            q = np.asarray(quat, dtype=float).ravel()
            if p.size != 3 or q.size != 4 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
                continue
            if not 0.9 < float(np.linalg.norm(q)) < 1.1:
                continue
            for xyzw in (q[[1, 2, 3, 0]], q):  # Isaac Lab's (w, x, y, z) first, then (x, y, z, w)
                r = rotation(xyzw)
                if (float(np.linalg.norm(p + r[:, 2] * L.TCP_OFFSET - tcp)) <= L.TURN_POS_TOL
                        and float(r[:, 2] @ axis) >= cos_tol):
                    found.append(r)
        if not found or any(abs(float(r[:, 0] @ found[0][:, 0])) < cos_tol for r in found[1:]):
            return None
        return found[0]
    except (TypeError, ValueError, AttributeError, IndexError):
        return None


def with_turn(arms, bodies):
    """rigstate's arms, each with its turn from ``bodies`` ({arm: [(position, quaternion), ...]}, gripturn's
    reading, or None); anything that is not a dict of arms passes through unchanged."""
    if not isinstance(arms, dict):
        return arms
    out = {}
    for side, a in arms.items():
        cands = bodies.get(side) if isinstance(bodies, dict) else None
        try:
            out[side] = Gripper(a.tcp, a.tips, a.opening, gripper_turn(a, cands))
        except AttributeError:
            out[side] = a  # not an arm: arms_valid() refuses it
    return out


def turn_of(arm) -> np.ndarray | None:
    """The gripper's frame if it is known (a finite 3 x 3 rotation), else None."""
    r = getattr(arm, "rot", None)
    if r is None:
        return None
    try:
        r = np.asarray(r, dtype=float)
    except (TypeError, ValueError):
        return None
    if r.shape != (3, 3) or not np.all(np.isfinite(r)) or not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
        return None
    return r


def finger_gap(arm) -> float:
    return float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP


def finger_boxes(arm, r) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The two fingers as boxes (centre, rotation, half sizes), posed as the gripper (frame ``r``) is and grown by
    1 mm on every side: each from half the gap to half the gap plus 10 mm along the closing axis, 40 mm wide, from
    the tips up 70 mm."""
    g = finger_gap(arm) / 2
    m = L.ZONE_MARGIN
    tips = np.asarray(arm.tips, dtype=float)
    half = np.array([L.FINGER_THICK / 2 + m, L.FINGER_HALF_WIDTH + m, L.FINGER_LEN / 2 + m])
    return [(tips + r[:, 0] * (k * (g + L.FINGER_THICK / 2)) - r[:, 2] * (L.FINGER_LEN / 2), r, half)
            for k in (1.0, -1.0)]


def box_margin(box, points) -> np.ndarray:
    """Signed distance of each point from the box (m; zero or negative: inside it)."""
    c, r, half = box
    q = np.abs((np.asarray(points, dtype=float) - c) @ r) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def zone_radii(arm) -> tuple[float, float]:
    """Inner and outer radius of the turn-free zone (the grown finger boxes swept round the axis) (m)."""
    g = finger_gap(arm) / 2
    m = L.ZONE_MARGIN
    return g - m, math.hypot(g + L.FINGER_THICK + m, L.FINGER_HALF_WIDTH + m)


def sweep_margin(arm, points=WAX_POINTS) -> float:
    """How far the closest of ``points`` is outside the turn-free zone (m; zero or negative: inside it)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    d = np.asarray(points, dtype=float) - tips
    s = d @ u  # along the axis, up from the tips
    r = np.linalg.norm(d - np.outer(s, u), axis=1)
    inner, outer = zone_radii(arm)
    lo, hi = -L.ZONE_MARGIN, L.FINGER_LEN + L.ZONE_MARGIN
    da = np.maximum(np.maximum(lo - s, s - hi), 0.0)
    dr = np.maximum(np.maximum(inner - r, r - outer), 0.0)
    outside = np.hypot(da, dr)
    depth = np.minimum.reduce([s - lo, hi - s, r - inner, outer - r])
    m = np.where((da > 0) | (dr > 0), outside, -depth)
    return float(m.min())


def zone_margin(arm, points=WAX_POINTS) -> float:
    """How far the closest of ``points`` is outside the gripper's fingers (m; zero or negative: inside a finger):
    the posed finger boxes when the gripper's turn is known, else the turn-free zone."""
    r = turn_of(arm)
    if r is None:
        return sweep_margin(arm, points)
    return float(min(box_margin(b, points).min() for b in finger_boxes(arm, r)))


def holds(arm, s, h: float) -> bool:
    """The gripper holds the ring: fingers under 88 mm apart (a grip across the corners stops at 84.9 mm), tip point
    below the ring's top and within 45 mm of its axis horizontally (the corners are at 42.4 mm)."""
    tips = np.asarray(arm.tips, dtype=float)
    c = np.asarray(s.pos, dtype=float)
    return bool(finger_gap(arm) < L.HOLD_GAP and tips[2] < top_z(s, h)
                and math.hypot(tips[0] - c[0], tips[1] - c[1]) <= L.HOLD_RADIUS)


# -- the end-state tests ----------------------------------------------------------------------------------------
def encloses(s, h: float) -> bool:
    """The ring is over the wax: tilted at most 10 degrees, every corner of the crown and of the sprue post inside
    its 52 x 52 mm opening (in its own frame), and the crown's top above its lower end."""
    if tilt_deg(s.quat_xyzw) > L.ENCLOSE_TILT_DEG + EPS_DEG:
        return False
    p = to_ring(s, h, WAX_CORNERS)
    if not np.all(np.abs(p[:, :2]) < L.RING_INNER_HALF):
        return False
    return bool(np.all(to_ring(s, h, CROWN_TOP_CORNERS)[:, 2] > 0.0))


def corner_heights(s, h: float) -> np.ndarray:
    """Heights above the table top of the four outer corners of the ring's lower end (m)."""
    return bottom_corners(s, h)[:, 2] - L.TOP


def on_base(s, h: float) -> bool:
    """Level on the former's base: all four lower outer corners 9 to 12 mm up (the base's top is 10 mm up, the
    step's 16 mm). Four corners within 3 mm of each other over the 60 mm sides means tilted under 2.9 degrees."""
    z = corner_heights(s, h)
    return bool(np.all(z >= L.SEAT_LOW - EPS) and np.all(z <= L.SEAT_HIGH + EPS))


def near_former(s) -> bool:
    """The ring's centre within 45 mm of the former's axis, seen from above."""
    return bool(np.hypot(*(np.asarray(s.pos, dtype=float)[:2] - FORMER_XY)) < L.SECOND_RING_RADIUS)


_N_EDGE = int(math.ceil(L.RING_OUTER / L.EDGE_SAMPLE - 1e-9))
_T_EDGE = np.linspace(-L.RING_OUTER_HALF, L.RING_OUTER_HALF, _N_EDGE + 1)
_EDGE_LOCAL = np.vstack([np.column_stack([_T_EDGE, np.full_like(_T_EDGE, k * L.RING_OUTER_HALF), 0 * _T_EDGE])
                         for k in (-1, 1)] +
                        [np.column_stack([np.full_like(_T_EDGE, k * L.RING_OUTER_HALF), _T_EDGE, 0 * _T_EDGE])
                         for k in (-1, 1)])


def lower_edge(s, h: float) -> np.ndarray:
    """Points of the outer edge of the ring's lower end, every 5 mm, corners included (world, n x 3)."""
    o, r = frame(s, h)
    return o + _EDGE_LOCAL @ r.T


def on_former(s, h: float) -> bool:
    """The ring is on the former: its centre within 45 mm of the former's axis, seen from above, or a point of its
    lower end's outer edge over the base's 88 x 88 mm outline, seen from above, and 5 mm or more above the table
    (resting or leaning on the base, on the step or on another ring there, or held over the base)."""
    if near_former(s):
        return True
    p = lower_edge(s, h)
    over = ((np.abs(p[:, 0] - L.FORMER_X) <= L.BASE_HALF + EPS) & (np.abs(p[:, 1] - L.FORMER_Y) <= L.BASE_HALF + EPS)
            & (p[:, 2] - L.TOP >= L.ON_FORMER_Z - EPS))
    return bool(over.any())


def axis_offset(s) -> float:
    """Horizontal distance from the ring's centre to the former's axis (m)."""
    return float(np.hypot(*(np.asarray(s.pos, dtype=float)[:2] - FORMER_XY)))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


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
        self.steps = 0  # control steps fed, blow-ups included
        self.good_steps = 0  # ... with valid ring states
        self.bad_steps = 0
        self.arms_unavailable = 0  # good steps on which the arms could not be read
        self.touched = False
        self.touch: dict | None = None  # the first touch: step and cause
        self.min_wall_gap = {n: math.inf for n in L.RING_NAMES}
        self.min_zone = {a: math.inf for a in ARMS}
        self.turn_unknown = {a: 0 for a in ARMS}  # steps with the arms read but this gripper's turn not known
        self.hist = {n: deque(maxlen=L.STILL_STEPS + 1) for n in L.RING_NAMES}
        self.prev = None  # {name: (centre, velocity)} at the previous good step
        self.last: Result = Result(0.0, False, {})

    # -- episode boundaries ------------------------------------------------------------------------------------
    @staticmethod
    def at_start(rings) -> bool:
        """Every ring upright on the table on its start spot (within 45 mm: a seeded reset's jitter)."""
        for n in L.RING_NAMES:
            s, (h, _, x, y) = rings[n], L.RINGS[n]
            low = float(np.min(corner_heights(s, h)))
            if not (math.hypot(s.pos[0] - x, s.pos[1] - y) < L.START_TOL and abs(low) < 0.003
                    and tilt_deg(s.quat_xyzw) < 1.0):
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some ring moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(float(np.linalg.norm(now[n][0] - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
                   for n, (p0, v0) in self.prev.items())

    def _latch(self, cause: str) -> None:
        if not self.touched:
            self.touched = True
            self.touch = {"step": self.steps, "cause": cause}

    # -- one control step --------------------------------------------------------------------------------------
    def update(self, rings, arms=None) -> Result:
        if not rings_valid(rings):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(rings, arms)
            return self.last
        now = {n: (np.array(rings[n].pos, dtype=float), np.array(rings[n].lin_vel, dtype=float))
               for n in L.RING_NAMES}
        if self.prev is not None and self._teleported(now) and self.at_start(rings):
            self.reset()  # everything put back on its start: a new episode (rigstate's and mechkit's reset hooks
            #               normally do this first; this is the fallback)
        self.steps += 1
        self.good_steps += 1
        self.prev = now
        for n in L.RING_NAMES:
            self.hist[n].append(now[n][0])
        if arms_valid(arms):
            for a in ARMS:
                if turn_of(arms[a]) is None:
                    self.turn_unknown[a] += 1  # the turn-free zone is used on this step
                m = zone_margin(arms[a])
                self.min_zone[a] = min(self.min_zone[a], m)
                if m <= 0.0:
                    self._latch(f"{a} gripper")
        else:
            self.arms_unavailable += 1
        for n in L.RING_NAMES:
            g, touch = wall_wax(rings[n], L.ring_height(n))
            if g is not None:
                self.min_wall_gap[n] = min(self.min_wall_gap[n], g)
            if touch:
                self._latch(n)
        self.last = self.result(rings, arms)
        return self.last

    def still(self, name: str) -> bool:
        """The ring's centre moved at most 2 mm in total over the last 10 control steps (0.5 s)."""
        h = self.hist[name]
        if len(h) < L.STILL_STEPS + 1:
            return False
        p = np.array(h)
        return bool(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() <= L.STILL_PATH)

    def blind_fraction(self) -> float:
        return self.arms_unavailable / self.good_steps if self.good_steps else 0.0

    def result(self, rings, arms=None) -> Result:
        if not rings_valid(rings):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "touched": self.touched})
        have_arms = arms_valid(arms)
        per: dict[str, dict] = {}
        for n in L.RING_NAMES:
            s, h = rings[n], L.ring_height(n)
            held = {a: holds(arms[a], s, h) for a in ARMS} if have_arms else None
            is_held = (any(held.values()) if held is not None else True)  # unreadable arms: held (fail closed)
            enc = encloses(s, h)
            level = on_base(s, h)
            still = self.still(n)
            z = corner_heights(s, h)
            gap = wall_wax_gap(s, h)
            per[n] = {"encloses": enc, "seated": bool(enc and level and still and not is_held), "held": is_held,
                      "still": still, "on_base": level, "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                      "low_corner_mm": _mm(float(z.min())), "high_corner_mm": _mm(float(z.max())),
                      "axis_offset_mm": _mm(axis_offset(s)), "wall_wax_gap_mm": _mm(gap),
                      "min_wall_wax_gap_mm": _mm(self.min_wall_gap[n]), "on_former": on_former(s, h),
                      "pos": [round(float(v), 4) for v in s.pos]}
        near = [n for n in L.RING_NAMES if per[n]["on_former"]]
        blind = self.blind_fraction()
        right = per[L.RIGHT_RING]
        wrong_seated = [n for n in L.RING_NAMES if n != L.RIGHT_RING and per[n]["seated"]]
        if self.touched:
            score, tier = 0.0, "wax_touched"
        elif len(near) >= 2:
            score, tier = 0.0, "two_rings_on_former"
        elif blind > L.BLIND_LIMIT:
            score, tier = 0.0, "arms_unreadable"
        elif right["seated"]:
            score, tier = 1.0, "right_ring_seated"
        elif right["encloses"]:
            score, tier = 0.5, "right_ring_over_wax"
        elif wrong_seated:
            score, tier = 0.25, "wrong_ring_seated"
        else:
            score, tier = 0.0, "none"
        details = {
            "valid_state": True,
            "tier": tier,
            "touched": self.touched,
            "touch": self.touch,
            "rings_on_former": near,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "arms_blind_fraction": round(blind, 4),
            "arms_turn_unknown": dict(self.turn_unknown),
            "bad_steps": self.bad_steps,
            "rings": per,
        }
        if have_arms:
            details["arms"] = {a: {"gap_mm": _mm(finger_gap(arms[a])), "zone_margin_mm": _mm(zone_margin(arms[a])),
                                   "min_zone_margin_mm": _mm(self.min_zone[a]),
                                   "turn_known": turn_of(arms[a]) is not None} for a in ARMS}
        return Result(score, bool(score == 1.0), details)
