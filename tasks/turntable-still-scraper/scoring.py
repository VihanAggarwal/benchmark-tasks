"""The scorer, as plain functions of the turntable's angle, the smoother's pose and the two grippers. NumPy only; no
simulator needed.

The rule is about how the side of the cake was smoothed, so the scorer keeps a record of the whole trial and is fed
one observation per control step by ``task.success()``:

    Tracker.update(turn, smoother, arms) -> Result(score, success, details)

``turn`` is (angle, angular velocity) of the turntable's plate (rad, rad/s; positive anticlockwise seen from above).
``smoother`` is anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (as
``yam_arena.task_file.ObjectState`` has) and, from mechkit, ``origin`` (the middle of the blade's bottom edge; worked
out from ``pos`` when missing). ``arms`` is ``rigstate.arms()``: each gripper's tool point, finger-tip point and
opening (0 closed .. 1 open), or None when they cannot be read.

The smoother is SET when a gripper holds it by the handle, it stands upright (within 5 deg), the blade's lowest
point is within -1..+5 mm of the plate's top, the blade (its real box) comes within -1..+3 mm of the cake's side
where they are nearest (the contact point) with its length within 45 deg of the side's tangent there (flat on the
side, or angled with an edge on it), and it is still: its centre moving slower than 1 cm/s and within 10 mm of where
it was when it was put in place, its heading turning slower than 3 deg/s and within 4 deg of its heading then. Once
out of either window it is not set again until it has been taken away (its centre 20 mm from where it was put, its
bottom 20 mm above the plate, or let go). The side is 360 one-degree sectors in the cake's own frame. While the
smoother stays set and the OTHER gripper's finger zone is within 5 mm of a peg (it is turning the turntable), every
control step the turntable's turn carries the side across both edges of the credited arc (3 deg either side of the
contact point): the sectors each edge crosses count as having passed the blade, once. Only the turntable's turn
counts, less any part of it that the blade followed (its centre moving round with the side): moving the smoother
round a still cake, dragging or twisting the cake round with it, or a plate coasting with no gripper at its pegs,
counts for nothing. Arms that cannot be read hold nothing and turn nothing.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
ARMS = ("left", "right")
AXIS = np.array(L.AXIS, dtype=float)
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
# The smoother's two boxes in its own frame (origin at the middle of the blade's bottom edge).
BLADE_C = np.array([0.0, 0.0, L.BLADE[2] / 2])
BLADE_H = np.array(L.BLADE, dtype=float) / 2
HANDLE_C = np.array([0.0, 0.0, L.BLADE[2] + L.HANDLE[2] / 2])
HANDLE_H = np.array(L.HANDLE, dtype=float) / 2
_BOX_CENTRE = np.array([0.0, 0.0, L.SMOOTHER_H / 2])  # the centre of the box around both (ObjectState.pos at rest)
Z_PLATE = L.TOP + L.PLATE_TOP
Z_CAKE_TOP = L.TOP + L.CAKE_TOP
Z_PEG_TOP = L.TOP + L.PEG_TOP


def _surface(centre, half, step: float = 0.005) -> np.ndarray:
    """Points on a box's surface (local frame), no more than ``step`` apart, corners included."""
    axes = [np.linspace(-h, h, max(2, int(math.ceil(2 * h / step)) + 1)) for h in half]
    pts = []
    for i in range(3):
        j, k = [a for a in range(3) if a != i]
        gj, gk = np.meshgrid(axes[j], axes[k], indexing="ij")
        for sign in (-1.0, 1.0):
            p = np.zeros((gj.size, 3))
            p[:, i] = sign * half[i]
            p[:, j] = gj.ravel()
            p[:, k] = gk.ravel()
            pts.append(p)
    return np.unique(np.round(np.vstack(pts) + centre, 7), axis=0)


SMOOTHER_POINTS = np.vstack([_surface(BLADE_C, BLADE_H), _surface(HANDLE_C, HANDLE_H)])
_CORNERS = np.vstack([BLADE_C + _SIGNS * BLADE_H, HANDLE_C + _SIGNS * HANDLE_H])
_BLADE_CORNERS = BLADE_C + _SIGNS * BLADE_H
_SIDE_Z = np.linspace(Z_PLATE, Z_CAKE_TOP, 11)  # heights at which the blade's gap to the side is measured


# -- states ---------------------------------------------------------------------------------------------------------
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
        origin = getattr(s, "origin", None)
        if origin is not None:
            parts.append(np.asarray(origin, dtype=float).ravel())
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    sizes = [p.size for p in parts]
    return bool(sizes[:6] == [3, 4, 3, 3, 3, 3] and (len(sizes) == 6 or sizes[6] == 3)
                and np.linalg.norm(parts[1]) > 0.5)


def turn_valid(turn) -> bool:
    try:
        a, w = (float(v) for v in turn)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(w)


def arms_valid(arms) -> bool:
    """Both arms present with finite tool point, tip point and opening (rigstate.arms() can be None)."""
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


def pose(s) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the smoother's frame in the world."""
    r = rotation(s.quat_xyzw)
    origin = getattr(s, "origin", None)
    if origin is None:
        origin = np.asarray(s.pos, dtype=float) - r @ _BOX_CENTRE
    return r, np.asarray(origin, dtype=float)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the smoother's up axis (blade bottom to handle) and the world's vertical (degrees). The blade
    is the same either face forward; upside down is 180."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def is_resting(s, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def wrap(a: float) -> float:
    """An angle (rad) folded into (-pi, pi]."""
    return float(-((-a + math.pi) % (2 * math.pi) - math.pi))


# -- the turntable --------------------------------------------------------------------------------------------------
def peg_centres(theta: float) -> np.ndarray:
    """The four pegs' centres (world x, y) with the plate turned ``theta``."""
    a = np.radians(np.asarray(L.PEG_ANGLES_DEG)) + float(theta)
    return AXIS + L.PEG_CIRCLE * np.stack([np.cos(a), np.sin(a)], axis=1)


def cylinder_distance(points, centre_xy, radius: float, z0: float, z1: float) -> np.ndarray:
    """Signed distance (m) from each point (N x 3) to an upright solid cylinder (negative inside)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    dr = np.hypot(p[:, 0] - centre_xy[0], p[:, 1] - centre_xy[1]) - radius
    dz = np.maximum(p[:, 2] - z1, z0 - p[:, 2])
    outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
    return np.where((dr <= 0) & (dz <= 0), np.maximum(dr, dz), outside)


def cake_distance(points) -> np.ndarray:
    return cylinder_distance(points, AXIS, L.CAKE_R, Z_PLATE, Z_CAKE_TOP)


# -- the smoother ---------------------------------------------------------------------------------------------------
def smoother_points(s) -> np.ndarray:
    r, o = pose(s)
    return SMOOTHER_POINTS @ r.T + o


def lowest(s) -> float:
    """The smoother's lowest point (world z)."""
    r, o = pose(s)
    return float(np.min((_CORNERS @ r.T + o)[:, 2]))


def blade_centre(s) -> np.ndarray:
    """The centre of the blade (world): what the smoother's stillness is measured on."""
    r, o = pose(s)
    return o + r @ BLADE_C


def blade_bottom(s) -> float:
    """The blade's lowest point above the plate's top (m)."""
    r, o = pose(s)
    return float(np.min((_BLADE_CORNERS @ r.T + o)[:, 2])) - Z_PLATE


def on_top(s) -> bool:
    """Some part of the smoother is over the cake's top (within its radius less 3 mm, seen from above) and less
    than 3 mm above it."""
    p = smoother_points(s)
    over = np.hypot(p[:, 0] - AXIS[0], p[:, 1] - AXIS[1]) <= L.CAKE_R - L.TOP_INSET
    return bool(np.any(over & (p[:, 2] < Z_CAKE_TOP + L.TOP_TOUCH)))


def smoother_cake_gap(s) -> float:
    """From the smoother to the cake (m; negative: in contact, pressed in)."""
    return float(np.min(cake_distance(smoother_points(s))))


@dataclass
class Contact:
    gap: float  # m: from the blade to the cake's side where they are nearest (negative: pressed in)
    along: float  # m: where that is along the blade, from its middle
    face_deg: float  # the blade's length to the side's tangent there (0: flat on it; 90: end on)
    point: np.ndarray  # the contact point: the side's point nearest the blade (world x, y)
    phi: float  # its angle about the turntable's axis (rad, world frame, anticlockwise from forward)


def blade_contact(s) -> Contact | None:
    """Where the cake's side is nearest the blade (its real box: a face, an edge or a corner), measured horizontally
    at the heights of the side that the blade spans. None if it spans none of them."""
    r, o = pose(s)
    axis_pts = np.column_stack([np.full(_SIDE_Z.size, AXIS[0]), np.full(_SIDE_Z.size, AXIS[1]), _SIDE_Z])
    local = (axis_pts - o) @ r  # the cake's axis in the smoother's frame
    spans = (local[:, 2] >= 0.0) & (local[:, 2] <= L.BLADE[2])
    if not np.any(spans):
        return None
    near = np.clip(local, BLADE_C - BLADE_H, BLADE_C + BLADE_H)  # the blade's point nearest each axis point
    d = (near @ r.T + o)[:, :2] - axis_pts[:, :2]
    dist = np.hypot(d[:, 0], d[:, 1])
    gaps = np.where(spans, dist - L.CAKE_R, np.inf)
    i = int(np.argmin(gaps))
    if dist[i] < 1e-6:
        return None
    toward = d[i] / dist[i]  # from the axis toward the blade, horizontally
    length = r[:2, 0]
    n = float(np.hypot(*length))
    tangent = np.array([-toward[1], toward[0]])
    face = 90.0 if n < 1e-6 else math.degrees(math.acos(min(1.0, abs(float(length @ tangent)) / n)))
    return Contact(float(gaps[i]), float(near[i, 0]), face, AXIS + L.CAKE_R * toward,
                   float(math.atan2(toward[1], toward[0])))


def heading(s) -> float:
    """The blade's heading: its length's direction seen from above (rad; the blade is the same turned half round)."""
    r = rotation(s.quat_xyzw)
    return float(math.atan2(r[1, 0], r[0, 0]))


def half_turn(a: float) -> float:
    """An angle difference (rad) folded into [-pi/2, pi/2) (headings of a symmetric blade)."""
    return float((a + math.pi / 2) % math.pi - math.pi / 2)


# -- the grippers ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger's outer corner: half the gap plus the finger's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


# Each disc of the zone: its rim at 72 points (distances to a peg or the cake within 0.1 mm of the true disc's), a
# ring at half the radius and the centre.
_RING = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)]
                 + [(0.5 * math.cos(a), 0.5 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
                 + [(0.0, 0.0)])


def finger_zone(arm, samples: int = 20) -> np.ndarray:
    """Points filling the finger zone: discs round the gripper's axis, from its finger-tip point back up 38 mm."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    e1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    centres = tips - np.outer(np.linspace(0.0, L.FINGER_LEN, samples), u)
    disc = zone_radius(arm) * (np.outer(_RING[:, 0], e1) + np.outer(_RING[:, 1], e2))
    return (centres[:, None, :] + disc[None, :, :]).reshape(-1, 3)


def zone_cake_gap(arm) -> float:
    """From a gripper's finger zone to the cake (m; negative: they overlap)."""
    return float(np.min(cake_distance(finger_zone(arm))))


def zone_peg_gap(arm, theta: float) -> float:
    """From a gripper's finger zone to the nearest peg (m)."""
    pts = finger_zone(arm)
    return float(min(np.min(cylinder_distance(pts, c, L.PEG_R, Z_PLATE, Z_PEG_TOP)) for c in peg_centres(theta)))


def handle_distance(s, point) -> float:
    """From a point to the smoother's handle (its box, in the smoother's frame; 0 inside)."""
    r, o = pose(s)
    local = r.T @ (np.asarray(point, dtype=float) - o) - HANDLE_C
    return float(np.linalg.norm(np.maximum(np.abs(local) - HANDLE_H, 0.0)))


def holder(arms, s) -> str | None:
    """Which gripper holds the smoother by its handle: at most half open with its finger-tip point within 10 mm of
    the handle (None: neither)."""
    for side in ARMS:
        a = arms[side]
        if float(a.opening) <= L.HOLD_OPENING and handle_distance(s, a.tips) <= L.HOLD_TOL:
            return side
    return None


# -- coverage -------------------------------------------------------------------------------------------------------
def mark(covered: np.ndarray, a_deg: float, b_deg: float) -> None:
    """Mark the one-degree sectors (cake frame) that the stretch from ``a_deg`` to ``b_deg`` touches."""
    lo, hi = min(a_deg, b_deg), max(a_deg, b_deg)
    if hi - lo >= L.N_SECTORS - 1:
        covered[:] = True
        return
    idx = np.arange(math.floor(lo), math.floor(hi) + 1) % L.N_SECTORS
    covered[idx] = True


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


ZERO_TEXT = {
    "left_fingers_at_cake": "the left gripper's fingers came within 5 mm of the cake",
    "right_fingers_at_cake": "the right gripper's fingers came within 5 mm of the cake",
    "smoother_on_top": "the smoother touched the cake's top",
    "smoother_off_table": "the smoother fell off the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.covered = np.zeros(L.N_SECTORS, dtype=bool)  # the side's sectors that have passed the blade (cake frame)
        self.zeros: dict[str, int] = {}  # zero condition -> the control step it first happened
        self.anchor = None  # the blade's centre (world) when the smoother was put in place, until it is taken away
        self.anchor_heading = None  # its heading then (rad)
        self.latched = False  # it left its window since it was put in place: not set until taken away
        self.was_set = False
        self.set_steps = 0  # consecutive control steps set
        self.set_1s = False
        self.set_count = 0  # control steps set in all
        self.turned_while_set = 0.0  # rad, total turn counted
        self.first_full_step = None
        self.max_drift = 0.0  # m: the blade's centre's furthest from its anchor while set
        self.finish_steps = 0  # consecutive control steps the finish has held
        self.min_finger_cake = {"left": math.inf, "right": math.inf}
        self.min_top_clear = math.inf
        self.arms_unavailable = 0
        self.bad_steps = 0
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # (smoother origin, its velocity, blade centre, turn angle, contact angle or None, heading)
        #                   at the previous step

    @property
    def zero(self) -> bool:
        return bool(self.zeros)

    @property
    def n_covered(self) -> int:
        return int(np.count_nonzero(self.covered))

    @staticmethod
    def at_start(s, theta: float) -> bool:
        """The starting layout: the smoother hanging still in its stand, the turntable at 0."""
        _, o = pose(s)
        return bool(np.linalg.norm(o - np.asarray(L.SMOOTHER_START)) < L.START_TOL and tilt_deg(s.quat_xyzw) < 1.0
                    and is_resting(s) and abs(theta) < L.START_TURN_TOL)

    def _teleported(self, origin) -> bool:
        p0, v0 = self.prev[0], self.prev[1]
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return float(np.linalg.norm(origin - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall

    def _flag(self, key: str) -> None:
        self.zeros.setdefault(key, self.steps)

    @staticmethod
    def finish_now(turn, smoother, arms) -> bool:
        """The finish, at this step: the smoother 10 mm or more from the cake, every finger zone 20 mm or more from
        every peg, the turntable still. Unreadable arms: not met."""
        if not arms_valid(arms):
            return False
        theta, omega = (float(v) for v in turn)
        return bool(smoother_cake_gap(smoother) >= L.FINISH_CAKE and abs(omega) < L.STILL_TURN
                    and min(zone_peg_gap(arms[side], theta) for side in ARMS) >= L.FINISH_PEG)

    def update(self, turn, smoother, arms=None) -> Result:
        if not (valid(smoother) and turn_valid(turn)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(turn, smoother, arms)
            return self.last
        theta, _ = (float(v) for v in turn)
        _, origin = pose(smoother)
        if self.prev is not None and self._teleported(origin) and self.at_start(smoother, theta):
            self.reset()  # the simulator put the smoother back in its stand and the turntable at 0: a new episode
            #               (rigstate's and mechkit's reset hooks normally do this first; this is the fallback)
        self.steps += 1
        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1

        contact = blade_contact(smoother)
        hold = holder(arms, smoother) if have_arms else None  # unreadable arms hold nothing (no credit)
        bottom = blade_bottom(smoother)
        placed = bool(
            hold is not None and contact is not None and tilt_deg(smoother.quat_xyzw) < L.TILT_MAX_DEG
            and L.BOTTOM_MIN <= bottom <= L.BOTTOM_MAX and L.GAP_MIN <= contact.gap <= L.GAP_MAX
            and contact.face_deg <= L.FACE_MAX_DEG
        )
        mid, head = blade_centre(smoother), heading(smoother)
        taken_away = False
        if self.anchor is not None and (float(np.linalg.norm(mid - self.anchor)) >= L.AWAY or bottom >= L.AWAY
                                        or hold is None):
            self.anchor, self.anchor_heading, self.latched, taken_away = None, None, False, True
        if placed and self.anchor is None and not taken_away:
            self.anchor, self.anchor_heading = mid.copy(), head
        is_set = False
        if placed and self.anchor is not None:
            drift = float(np.linalg.norm(mid - self.anchor))
            turned_head = abs(half_turn(head - self.anchor_heading))
            if drift > L.STILL_WINDOW or turned_head > math.radians(L.HEADING_WINDOW_DEG):
                self.latched = True  # out of its window: not set again until it is taken away
            if self.prev is not None:
                speed = float(np.linalg.norm(mid - self.prev[2])) / self.dt
                spin = abs(half_turn(head - self.prev[5])) / self.dt
            else:
                speed = float(np.linalg.norm(np.asarray(smoother.lin_vel, dtype=float)))
                spin = abs(float(np.asarray(smoother.ang_vel, dtype=float)[2]))
            is_set = not self.latched and speed < L.STILL_SPEED and spin < math.radians(L.STILL_TURN_DPS)
            if is_set:
                self.max_drift = max(self.max_drift, drift)
        driver = None  # the other gripper, turning the turntable by a peg
        if hold is not None:
            other = "right" if hold == "left" else "left"
            if zone_peg_gap(arms[other], theta) <= L.DRIVE_GAP:
                driver = other
        if is_set and self.was_set and driver is not None and self.prev is not None and self.prev[4] is not None:
            turned = wrap(theta - self.prev[3])
            phi0 = self.prev[4]
            tangent = np.array([-math.sin(phi0), math.cos(phi0), 0.0])
            followed = float(np.dot(mid - self.prev[2], tangent)) / L.CAKE_R  # the blade going round with the side
            d = turned - float(np.median([0.0, followed, turned]))  # the turn the blade did not follow
            if math.radians(L.TURN_NOISE_DEG) <= abs(d) and abs(turned) <= math.radians(L.TURN_GLITCH_DEG):
                for edge in (-L.ARC_DEG, L.ARC_DEG):
                    start = math.degrees(phi0 - self.prev[3]) + edge  # the edge's place on the side, cake frame
                    mark(self.covered, start, start - math.degrees(d))  # carried across it by the turn
                self.turned_while_set += abs(d)
                if self.first_full_step is None and self.n_covered >= L.FULL:
                    self.first_full_step = self.steps
        self.set_steps = self.set_steps + 1 if is_set else 0
        self.set_count += int(is_set)
        if self.set_steps * self.dt >= L.SET_HOLD_S - 1e-9:
            self.set_1s = True
        self.was_set = is_set

        if have_arms:
            for side in ARMS:
                g = zone_cake_gap(arms[side])
                self.min_finger_cake[side] = min(self.min_finger_cake[side], g)
                if g < L.FINGER_CAKE_MIN:
                    self._flag(f"{side}_fingers_at_cake")
        p = smoother_points(smoother)
        over = np.hypot(p[:, 0] - AXIS[0], p[:, 1] - AXIS[1]) <= L.CAKE_R - L.TOP_INSET  # as on_top() judges it
        if np.any(over):
            clear = float(np.min(p[over, 2])) - Z_CAKE_TOP
            self.min_top_clear = min(self.min_top_clear, clear)
            if clear < L.TOP_TOUCH:
                self._flag("smoother_on_top")
        if lowest(smoother) <= L.TOP - L.OFF_TABLE:
            self._flag("smoother_off_table")
        self.finish_steps = self.finish_steps + 1 if self.finish_now(turn, smoother, arms) else 0
        self.prev = (origin.copy(), np.asarray(smoother.lin_vel, dtype=float).copy(), mid.copy(), theta,
                     None if contact is None else contact.phi, head)
        self.last = self.result(turn, smoother, arms, placed=placed, is_set=is_set, contact=contact, driver=driver)
        return self.last

    def result(self, turn, smoother, arms=None, *, placed=None, is_set=None, contact=None, driver=None) -> Result:
        """The score as things stand (read at the last step). ``success`` is what task.success() reports: the 1.0
        conditions at this step (the simulator then needs them for SUCCESS_HOLD_S, the same 1 s that the finish's
        hold here counts)."""
        if not (valid(smoother) and turn_valid(turn)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "zeros": sorted(self.zeros), "covered": self.n_covered})
        theta, omega = (float(v) for v in turn)
        have_arms = arms_valid(arms)
        n = self.n_covered
        finish = self.finish_now(turn, smoother, arms)
        held_1s = finish and self.finish_steps * self.dt >= L.FINISH_HOLD_S - 1e-9
        if self.zero:
            score = 0.0
        elif n >= L.FULL and held_1s:
            score = 1.0
        elif n >= L.THREE_QUARTER:
            score = 0.75
        elif n >= L.HALF:
            score = 0.5
        elif self.set_1s:
            score = 0.25
        else:
            score = 0.0
        success = bool(not self.zero and n >= L.FULL and finish)
        if contact is None:
            contact = blade_contact(smoother)
        peg_gaps = {side: zone_peg_gap(arms[side], theta) for side in ARMS} if have_arms else None
        details = {
            "valid_state": True,
            "zeros": sorted(self.zeros),
            "first_zero_step": min(self.zeros.values()) if self.zeros else None,
            "covered": n,
            "first_full_step": self.first_full_step,
            "turn_deg": round(math.degrees(theta), 2),
            "turn_speed_dps": round(math.degrees(omega), 2),
            "turned_while_set_deg": round(math.degrees(self.turned_while_set), 1),
            "placed": placed,
            "set": is_set,
            "latched": self.latched,
            "driver": driver,
            "set_1s": self.set_1s,
            "set_count": self.set_count,
            "holder": (holder(arms, smoother) if have_arms else None),
            "gap_mm": _mm(contact.gap) if contact is not None else None,
            "along_mm": _mm(contact.along) if contact is not None else None,
            "face_deg": round(contact.face_deg, 2) if contact is not None else None,
            "contact_deg": round(math.degrees(contact.phi), 2) if contact is not None else None,
            "bottom_mm": _mm(blade_bottom(smoother)),
            "tilt_deg": round(tilt_deg(smoother.quat_xyzw), 2),
            "heading_deg": round(math.degrees(heading(smoother)), 2),
            "heading_err_deg": (round(math.degrees(abs(half_turn(heading(smoother) - self.anchor_heading))), 2)
                                if self.anchor_heading is not None else None),
            "drift_mm": _mm(float(np.linalg.norm(blade_centre(smoother) - self.anchor))) if self.anchor is not None
            else None,
            "max_drift_mm": _mm(self.max_drift),
            "smoother_cake_mm": _mm(smoother_cake_gap(smoother)),
            "min_top_clear_mm": _mm(self.min_top_clear),
            "min_finger_cake_mm": {k: _mm(v) for k, v in self.min_finger_cake.items()},
            "peg_clear_mm": {k: _mm(v) for k, v in peg_gaps.items()} if peg_gaps else None,
            "finish": finish,
            "finish_s": round(self.finish_steps * self.dt, 2),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
