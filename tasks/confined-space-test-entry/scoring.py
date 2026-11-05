"""The scorer, as plain functions of the loose objects' states and the arms' state. NumPy only; no simulator needed.

The rule is about the whole trial, not only its end: no gripper may go into the tank until both manway covers are
off and the air has been tested at three depths, top first, and while a gripper is inside, the gas probe must be
held in the tank. So the scorer keeps a little state (when the covers came off, which depths have been tested, when
the tool came out, which rule was broken and when) and is fed one observation per control step by
``task.success()``.

    Tracker.update(cover_left, cover_right, probe, tool, arms) -> Result(score, success, details)

Each object is anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then the arm rules are not applied for that step, 'both
grippers open and clear' is taken as met, and the step is counted in ``arms_unavailable``. An object state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The covers and the tool are plain boxes, so either of their two broad faces may be down; the probe is a plain
cylinder, so either end may be its lower end (the 'tip' is whichever end is lower): every tilt is folded.
"""

from __future__ import annotations

import collections
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
SIDES = ("left", "right")
_SIGNS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))  # the eight corners of a box, as fractions


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


def arms_valid(arms) -> bool:
    """Both arms present with finite TCP, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in SIDES:
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
    """Angle between the world's vertical and the body's z axis, folded to 0..90 degrees: the covers' and the
    tool's thin axis (either broad face down) and the probe's long axis (either end down)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def _low(s) -> float:
    return float(s.low[2])


# -- the tank -----------------------------------------------------------------------------------------------------
def over_tank(p, grow: float = 0.0) -> bool:
    """Is the point over the tank's footprint (grown by ``grow`` on every side), seen from above?"""
    return bool(L.TANK_X0 - grow <= p[0] <= L.TANK_X1 + grow and L.TANK_Y0 - grow <= p[1] <= L.TANK_Y1 + grow)


def manway_rect(side: str) -> tuple[float, float, float, float]:
    """x0, x1, y0, y1 of a manway opening."""
    hx, hy = L.MANWAY_HALF
    y = L.MANWAY_Y[side]
    return L.MANWAY_X - hx, L.MANWAY_X + hx, y - hy, y + hy


def in_column(p) -> bool:
    """Is the point inside the left manway's outline, seen from above (in the column under it)?"""
    x0, x1, y0, y1 = manway_rect("left")
    return bool(x0 <= p[0] <= x1 and y0 <= p[1] <= y1)


# -- the arms -----------------------------------------------------------------------------------------------------
def tips_of(arm) -> np.ndarray:
    return np.asarray(arm.tips, dtype=float)


def entered(arm) -> bool:
    """The gripper is in the tank: the point between its finger tips is below the deck's top, over the tank's
    footprint grown by 5 mm."""
    p = tips_of(arm)
    return bool(p[2] < L.DECK_Z and over_tank(p, L.ENTRY_GROW))


def finger_gap(arm) -> float:
    """How far apart the fingers are (m), from rigstate's opening (0 closed .. 1 fully open, 94 mm)."""
    return float(np.clip(float(arm.opening), 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_axis(arm) -> tuple[np.ndarray, np.ndarray]:
    """The point between the finger tips and the unit vector from there back up the gripper (toward the TCP)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), tips_of(arm)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    return tips, (u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0]))


def _perpendiculars(u) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to each other and to the unit vector u."""
    e1 = np.cross(u, np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(u, e1)


_AXIS_S = np.linspace(0.0, L.FINGER_LEN, 9)  # sample points along the zone's axis (from the tips up)
_RING = np.linspace(0.0, 2 * math.pi, 24, endpoint=False)  # and round the rims of its two flat ends
_COS, _SIN = np.cos(_RING), np.sin(_RING)


def _cylinder_gap(tips, u, r, q) -> np.ndarray:
    """Distances from the points q (n x 3) to the solid cylinder of radius r about the axis from ``tips`` along
    ``u`` for FINGER_LEN (negative inside it, by the radial depth)."""
    v = q - tips
    a = v @ u
    rho = np.linalg.norm(v - np.outer(a, u), axis=1)
    da = np.maximum(0.0, np.maximum(-a, a - L.FINGER_LEN))
    dr = rho - r
    return np.where((da == 0.0) & (dr <= 0.0), dr, np.hypot(da, np.maximum(0.0, dr)))


def finger_zone(arm) -> tuple[np.ndarray, float]:
    """Points on the zone's axis and on the rims of its two flat ends (tips and 38 mm up), and its radius."""
    tips, u = finger_axis(arm)
    r = zone_radius(arm)
    e1, e2 = _perpendiculars(u)
    circle = r * (np.outer(_COS, e1) + np.outer(_SIN, e2))
    return np.vstack([tips + np.outer(_AXIS_S, u), tips + circle, tips + L.FINGER_LEN * u + circle]), r


def zone_box_gap(arm, lo, hi) -> float:
    """How far a gripper's finger zone is from a world-aligned box (negative: they overlap). The zone is a
    flat-ended cylinder (the fingers end at their tips), so a gripper whose finger tips are 3 cm above a box is 3 cm
    from it. The smallest distance is searched over the box's points nearest to the zone's axis and rims, and the
    box's corners."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    pts, r = finger_zone(arm)
    tips, u = finger_axis(arm)
    cands = np.vstack([np.clip(pts, lo, hi), lo + _SIGNS * (hi - lo)])
    return float(np.min(_cylinder_gap(tips, u, r, cands)))


# The loose objects' real shapes, in the order of NAMES: a box (its full size) or a cylinder (radius, length).
SHAPES = (("box", L.COVER_SIZE), ("box", L.COVER_SIZE), ("cyl", (L.PROBE_RADIUS, L.PROBE_LEN)), ("box", L.TOOL_SIZE))


def _closest_on(s, shape, pts) -> np.ndarray:
    """The point of the object (its real, turned shape) nearest to each of the points (n x 3); a point inside it is
    its own nearest point."""
    kind, dims = shape
    c, r = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)
    local = (np.asarray(pts, dtype=float) - c) @ r  # each row: the point in the object's frame
    if kind == "box":
        h = np.asarray(dims, dtype=float) / 2
        q = np.clip(local, -h, h)
    else:
        rad, length = dims
        xy = local[:, :2]
        n = np.linalg.norm(xy, axis=1, keepdims=True)
        xy = np.where(n > rad, xy * (rad / np.maximum(n, 1e-12)), xy)
        q = np.column_stack([xy, np.clip(local[:, 2], -length / 2, length / 2)])
    return q @ r.T + c


def _outline(s, shape) -> np.ndarray:
    """A box's eight corners, or points round the rims of a cylinder's two ends, in the world."""
    kind, dims = shape
    c, r = np.asarray(s.pos, dtype=float), rotation(s.quat_xyzw)
    if kind == "box":
        local = (_SIGNS - 0.5) * np.asarray(dims, dtype=float)
    else:
        rad, length = dims
        ring = np.column_stack([rad * _COS, rad * _SIN, np.zeros_like(_COS)])
        local = np.vstack([ring + [0.0, 0.0, -length / 2], ring + [0.0, 0.0, length / 2]])
    return local @ r.T + c


def zone_object_gap(arm, s, shape) -> float:
    """How far a gripper's finger zone is from a loose object's real shape (a turned box, or the probe's cylinder);
    negative: they overlap. Searched as zone_box_gap is: the object's points nearest to the zone's axis and rims,
    and the object's corners (or rims)."""
    pts, r = finger_zone(arm)
    tips, u = finger_axis(arm)
    cands = np.vstack([_closest_on(s, shape, pts), _outline(s, shape)])
    return float(np.min(_cylinder_gap(tips, u, r, cands)))


# -- the probe ----------------------------------------------------------------------------------------------------
def probe_up(probe) -> np.ndarray:
    """The probe's axis as a unit vector pointing up (either end may be the lower one)."""
    a = rotation(probe.quat_xyzw)[:, 2]
    return a if a[2] >= 0.0 else -a


def probe_tip(probe) -> np.ndarray:
    """The sensor tip: the centre of the probe's lower end."""
    return np.asarray(probe.pos, dtype=float) - probe_up(probe) * (L.PROBE_LEN / 2)


def grip_point(probe) -> np.ndarray:
    """Where the probe is held: on its axis, 25 mm below its upper end."""
    return np.asarray(probe.pos, dtype=float) + probe_up(probe) * (L.PROBE_LEN / 2 - L.PROBE_GRIP_DOWN)


def tip_speed(probe) -> float:
    """Speed of the tip (m/s): the probe's velocity plus its spin about the centre, at the tip."""
    r = probe_tip(probe) - np.asarray(probe.pos, dtype=float)
    v = np.asarray(probe.lin_vel, dtype=float) + np.cross(np.asarray(probe.ang_vel, dtype=float), r)
    return float(np.linalg.norm(v))


def tip_band(probe) -> int | None:
    """Index of the test band (0 top, 1 middle, 2 bottom) that the tip's height is in, or None."""
    h = float(probe_tip(probe)[2]) - L.TOP
    for i, (_, lo, hi) in enumerate(L.BANDS):
        if lo <= h <= hi:
            return i
    return None


def holders(probe, arms) -> list[str]:
    """The grippers holding the probe: TCP within 20 mm of its grip point and fingers less than half open."""
    g = grip_point(probe)
    return [s for s in SIDES
            if float(np.linalg.norm(np.asarray(arms[s].tcp, dtype=float) - g)) <= L.HELD_DIST
            and float(arms[s].opening) < L.HELD_OPENING]


def tip_in_tank(probe) -> bool:
    """The probe's tip is below the deck's top, over the tank's footprint grown by 5 mm."""
    t = probe_tip(probe)
    return bool(t[2] < L.DECK_Z and over_tank(t, L.ENTRY_GROW))


def monitoring_position(probe) -> bool:
    """The tip is inside the tank under the left manway: in its column, no higher than the deck's underside (58 mm
    above the table)."""
    t = probe_tip(probe)
    return bool(in_column(t) and float(t[2]) <= L.TOP + L.MONITOR_TIP_H)


def monitored(probe, arms) -> bool:
    """Monitoring the space: the probe in position and held by a gripper."""
    return bool(monitoring_position(probe) and holders(probe, arms))


def tip_off_holder(probe) -> float:
    """Horizontal distance from the tip to the socket's centre (m)."""
    t = probe_tip(probe)
    return float(np.hypot(t[0] - L.HOLDER_X, t[1] - L.HOLDER_Y))


def probe_in_holder(probe) -> bool:
    """Back in its holder: upright within 10 degrees, its lowest point within 5 mm of the table top, its tip within
    8 mm of the socket's centre (inside the 24 mm socket)."""
    return bool(tilt_deg(probe.quat_xyzw) < L.PROBE_UPRIGHT_DEG and abs(_low(probe) - L.TOP) <= L.PROBE_BOTTOM_TOL
                and tip_off_holder(probe) <= L.PROBE_HOLDER_TOL)


# -- the covers ---------------------------------------------------------------------------------------------------
def cover_off(cover, side: str) -> bool:
    """The cover's real outline seen from above (the shadow of the turned or tilted plate) no longer overlaps its
    manway's opening. Both are convex, so they are apart exactly when some line separates them: tested along x, y
    and across each of the plate's three edges as seen from above (the edges of its shadow); touching is apart."""
    corners = _outline(cover, SHAPES[0])[:, :2]
    x0, x1, y0, y1 = manway_rect(side)
    hole = np.array([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])
    r = rotation(cover.quat_xyzw)
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for k in range(3):
        e = r[:2, k]
        n = float(np.linalg.norm(e))
        if n > 1e-9:
            axes.append(np.array([-e[1], e[0]]) / n)
    for a in axes:
        p, q = corners @ a, hole @ a
        if p.max() <= q.min() or q.max() <= p.min():
            return True
    return False


def cover_centre_off(cover, side: str) -> float:
    return float(np.hypot(cover.pos[0] - L.MANWAY_X, cover.pos[1] - L.MANWAY_Y[side]))


def covers_opening(cover, side: str) -> bool:
    """All four corners of the opening lie under the cover's real (turned) outline."""
    r = rotation(cover.quat_xyzw)
    ex, ey = r[:2, 0], r[:2, 1]  # the cover's long and short axes seen from above (it is flat when this matters)
    nx, ny = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    if nx < 1e-6 or ny < 1e-6:
        return False
    ex, ey = ex / nx, ey / ny
    c = np.asarray(cover.pos[:2], dtype=float)
    x0, x1, y0, y1 = manway_rect(side)
    for p in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        d = np.asarray(p) - c
        if abs(float(d @ ex)) > L.COVER_HALF[0] or abs(float(d @ ey)) > L.COVER_HALF[1]:
            return False
    return True


def cover_back(cover, side: str) -> bool:
    """Back on its manway: flat (tilted under 5 degrees), resting on the deck (lowest point within 3 mm of its
    top), centre within 10 mm of the opening's centre, and the whole opening under it."""
    return bool(tilt_deg(cover.quat_xyzw) < L.COVER_FLAT_DEG and abs(_low(cover) - L.DECK_Z) <= L.COVER_Z_TOL
                and cover_centre_off(cover, side) <= L.COVER_CENTRE_TOL and covers_opening(cover, side))


# -- the tool -----------------------------------------------------------------------------------------------------
def tool_out(tool) -> bool:
    """Lifted out of the tank: its lowest point more than 5 mm above the deck's top, its centre off the tank's
    footprint."""
    return bool(_low(tool) > L.DECK_Z + L.TOOL_OUT_RISE and not over_tank(tool.pos))


def tool_on_pad(tool) -> bool:
    """Lying flat on the pad: a 60 x 24 mm face down within 10 degrees, its lowest point within 3 mm of the pad's
    top, its centre over the pad."""
    hx, hy = L.PAD_SIZE[0] / 2, L.PAD_SIZE[1] / 2
    over = abs(float(tool.pos[0]) - L.PAD_X) <= hx and abs(float(tool.pos[1]) - L.PAD_Y) <= hy
    return bool(over and tilt_deg(tool.quat_xyzw) < L.TOOL_FLAT_DEG and abs(_low(tool) - L.PAD_TOP) <= L.TOOL_Z_TOL)


# -- every loose object -------------------------------------------------------------------------------------------
def falling(s) -> bool:
    return bool(float(s.lin_vel[2]) < -L.FALL_SPEED)


def tipped(s) -> bool:
    return bool(tilt_deg(s.quat_xyzw) > L.TIP_DEG)


def off_table(s) -> bool:
    return bool(_low(s) < L.TOP - L.OFF_TABLE_DROP)


def min_clearance(arms, objects, shapes=None) -> float:
    """The smallest distance from either gripper's finger zone to any of the objects (m): to each object's real
    shape when ``shapes`` (one per object, as in SHAPES) is given, else to its world-aligned bounding box."""
    if shapes is None:
        return min(zone_box_gap(arms[s], *_lo_hi(o)) for s in SIDES for o in objects)
    return min(zone_object_gap(arms[s], o, sh) for s in SIDES for o, sh in zip(objects, shapes))


def grippers_clear(arms, objects, gap: float | None = None, shapes=None) -> bool:
    """Both grippers open (opening 0.9 or more) and each finger zone at least 30 mm from every loose object
    (``gap``: that smallest distance, if already known)."""
    if any(float(arms[s].opening) < L.GRIPPER_OPEN for s in SIDES):
        return False
    return bool((min_clearance(arms, objects, shapes) if gap is None else gap) >= L.CLEAR_GAP)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "entered_before_test": "a gripper went into the tank before the air test was complete",
    "entered_cover_on": "a gripper went into the tank while a manway cover was not off",
    "entered_unmonitored": "a gripper was in the tank while the probe was not held in the left manway's column",
    "probe_let_go_in_tank": "the probe was not held while its tip was in the tank",
    "fell": "a cover, the probe or the tool fell (moved down faster than 0.25 m/s)",
    "tipped": "a cover, the probe or the tool tipped over (tilted more than 30 degrees)",
    "off_table": "a cover, the probe or the tool left the table",
}
NAMES = ("cover_left", "cover_right", "probe", "tool")


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.DWELL_WINDOW_S / self.dt)))  # control steps in the speed window
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.breach_objects: dict[str, str] = {}  # fell / tipped / off_table -> which object
        self.ventilated_step: int | None = None  # both covers off for the first time
        self.band_steps: list[int] = []  # the step each depth's test was completed, top first
        self.dwell_band: int | None = None  # the band the tip is dwelling in now ...
        self.dwell_n = 0  # ... for this many consecutive steps
        self.tip_hist: collections.deque = collections.deque(maxlen=self.window + 1)  # the tip's last positions
        self.tool_out_step: int | None = None
        self.first_entry: dict[str, int | None] = {s: None for s in SIDES}
        self.steps_inside = {s: 0 for s in SIDES}
        self.min_tips_over_tank = {s: math.inf for s in SIDES}  # lowest finger-tip point over the tank (m above top)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity)] of the four loose objects at the previous valid control step
        self._covers_off = False  # both covers off at the current step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @property
    def bands_done(self) -> int:
        return len(self.band_steps)

    @property
    def test_complete(self) -> bool:
        return self.bands_done == len(L.BANDS)

    @property
    def test_step(self) -> int | None:
        return self.band_steps[-1] if self.test_complete else None

    @staticmethod
    def at_start(cl, cr, probe, tool) -> bool:
        """The starting layout: both covers flat on their manways, the probe upright in its holder, the tool flat
        on the tank floor under the right manway, all still (within START_TOL of their marks, which covers a
        seeded reset's jitter)."""
        covers = all(
            cover_centre_off(c, side) < L.START_TOL and abs(_low(c) - L.DECK_Z) < 0.003 and tilt_deg(c.quat_xyzw) < 1.0
            for c, side in ((cl, "left"), (cr, "right")))
        p = (np.hypot(probe.pos[0] - L.HOLDER_X, probe.pos[1] - L.HOLDER_Y) < L.START_TOL
             and abs(_low(probe) - L.TOP) < 0.003 and tilt_deg(probe.quat_xyzw) < 1.0)
        t = (np.hypot(tool.pos[0] - L.TOOL_X, tool.pos[1] - L.TOOL_Y) < L.START_TOL
             and abs(_low(tool) - L.FLOOR_Z) < 0.003 and tilt_deg(tool.quat_xyzw) < 1.0)
        return bool(covers and p and t and all(is_resting(s) for s in (cl, cr, probe, tool)))

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall
        could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str, what: str | None = None) -> None:
        if key not in self.breaches:
            self.breaches[key] = self.steps
            if what is not None:
                self.breach_objects[key] = what

    @property
    def path_speed(self) -> float:
        """The tip's speed for the test (m/s): the length of the path it moved over the last 0.5 s, divided by
        0.5 s; infinite until 0.5 s of positions have been seen. A path, not a displacement, so bobbing up and down
        in a band is never 'slow'."""
        if len(self.tip_hist) <= self.window:
            return math.inf
        pts = list(self.tip_hist)
        path = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
        return path / (self.window * self.dt)

    def _dwell(self, probe) -> None:
        """Follow the test: both covers off, the tip in a band, in the column, slower than 5 mm/s (its path over
        the last 0.5 s), without a break; a band counts once that has lasted 2 s, and only if it is the next band
        down."""
        tip = probe_tip(probe)
        self.tip_hist.append(np.array(tip, dtype=float))
        band = tip_band(probe)
        ok = (self._covers_off and band is not None and in_column(tip)
              and self.path_speed < L.DWELL_SPEED)
        if not ok:
            self.dwell_band, self.dwell_n = None, 0
            return
        if band == self.dwell_band:
            self.dwell_n += 1
        else:
            self.dwell_band, self.dwell_n = band, 1
        if band == self.bands_done and self.dwell_n * self.dt >= L.DWELL_S - 1e-9:
            self.band_steps.append(self.steps)

    def _arm_rules(self, probe, arms) -> None:
        held = holders(probe, arms)
        watched = bool(monitoring_position(probe) and held)
        for s in SIDES:
            p = tips_of(arms[s])
            if over_tank(p, L.ENTRY_GROW):
                self.min_tips_over_tank[s] = min(self.min_tips_over_tank[s], float(p[2]) - L.TOP)
            if not entered(arms[s]):
                continue
            self.steps_inside[s] += 1
            if self.first_entry[s] is None:
                self.first_entry[s] = self.steps
            if not self.test_complete:
                self._flag("entered_before_test", s)
            if not self._covers_off:
                self._flag("entered_cover_on", s)
            if not watched:
                self._flag("entered_unmonitored", s)
        if tip_in_tank(probe) and not held:
            self._flag("probe_let_go_in_tank")

    def update(self, cl, cr, probe, tool, arms=None) -> Result:
        objs = (cl, cr, probe, tool)
        if not all(valid(s) for s in objs):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(cl, cr, probe, tool, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in objs]
        if self.prev is not None and self._teleported(now) and self.at_start(*objs):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        self._covers_off = cover_off(cl, "left") and cover_off(cr, "right")
        if self._covers_off and self.ventilated_step is None:
            self.ventilated_step = self.steps
        self._dwell(probe)
        if arms_valid(arms):
            self._arm_rules(probe, arms)
        else:
            self.arms_unavailable += 1
        if self.tool_out_step is None and tool_out(tool):
            self.tool_out_step = self.steps
        for name, s in zip(NAMES, objs):
            if falling(s):
                self._flag("fell", name)
            if tipped(s):
                self._flag("tipped", name)
            if off_table(s):
                self._flag("off_table", name)
        self.last = self.result(cl, cr, probe, tool, arms)
        return self.last

    def result(self, cl, cr, probe, tool, arms=None) -> Result:
        objs = (cl, cr, probe, tool)
        if not all(valid(s) for s in objs):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        on_pad = tool_on_pad(tool)
        # The grippers matter only for 1.0, which needs the tool on the pad: measured only then (it is slow).
        gap = min_clearance(arms, objs, SHAPES) if have_arms and on_pad else None
        clear = grippers_clear(arms, objs, gap) if have_arms and on_pad else True  # unreadable: taken as met
        in_holder = probe_in_holder(probe)
        back = {"left": cover_back(cl, "left"), "right": cover_back(cr, "right")}
        if self.breach:
            score = 0.0
        elif on_pad and in_holder and all(back.values()) and clear:
            score = 1.0
        elif on_pad:
            score = 0.75
        elif self.tool_out_step is not None:
            score = 0.5
        elif self.test_complete:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and all(is_resting(s) for s in objs))
        tip = probe_tip(probe)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "breach_objects": dict(self.breach_objects),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "covers_off": {"left": cover_off(cl, "left"), "right": cover_off(cr, "right")},
            "ventilated_step": self.ventilated_step,
            "bands_done": self.bands_done,
            "band_steps": list(self.band_steps),
            "test_complete_step": self.test_step,
            "dwell_band": L.BANDS[self.dwell_band][0] if self.dwell_band is not None else None,
            "dwell_s": round(self.dwell_n * self.dt, 2),
            "tip_mm": [_mm(v) for v in (tip[0], tip[1], float(tip[2]) - L.TOP)],
            "tip_speed_mm_s": round(tip_speed(probe) * 1000, 1),  # the simulator's instantaneous reading
            "tip_path_speed_mm_s": (round(self.path_speed * 1000, 1) if math.isfinite(self.path_speed) else None),
            "tip_in_column": in_column(tip),
            "probe_held_by": holders(probe, arms) if have_arms else None,
            "monitored": monitored(probe, arms) if have_arms else None,
            "entered": {s: entered(arms[s]) for s in SIDES} if have_arms else None,
            "first_entry_step": dict(self.first_entry),
            "steps_inside": dict(self.steps_inside),
            "min_tips_over_tank_mm": {s: _mm(v) for s, v in self.min_tips_over_tank.items()},
            "tool_out_step": self.tool_out_step,
            "tool_on_pad": on_pad,
            "tool_low_mm": _mm(_low(tool) - L.TOP),
            "tool_tilt_deg": round(tilt_deg(tool.quat_xyzw), 2),
            "probe_in_holder": in_holder,
            "probe_tilt_deg": round(tilt_deg(probe.quat_xyzw), 2),
            "probe_low_mm": _mm(_low(probe) - L.TOP),
            "probe_off_holder_mm": _mm(tip_off_holder(probe)),
            "covers_back": back,
            "cover_centre_off_mm": {"left": _mm(cover_centre_off(cl, "left")),
                                    "right": _mm(cover_centre_off(cr, "right"))},
            "cover_low_mm": {"left": _mm(_low(cl) - L.TOP), "right": _mm(_low(cr) - L.TOP)},
            "opening": ({s: round(float(arms[s].opening), 3) for s in SIDES} if have_arms else None),
            "grippers_clear": clear if gap is not None else None,
            "min_gripper_clearance_mm": _mm(gap),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "resting": {n: is_resting(s) for n, s in zip(NAMES, objs)},
        }
        return Result(score, success, details)
