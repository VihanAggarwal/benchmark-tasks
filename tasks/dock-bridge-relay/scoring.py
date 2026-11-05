"""The scorer, as plain functions of the cart's and the two runners' states and the arms' state. NumPy only; no
simulator needed.

The rule is about the whole trial, not only where the cart ends: the path must be built before it is used (no part
of the cart may come over the gap until both runners are seated: a cart pushed at an open gap drops into it, and one
held up over it by the grippers has not used the bridge), the cart is pushed, never lifted, and a runner must not
slide out from under the cart. So the scorer keeps a little state (which rules were ever broken and when, where each
runner was when the cart came onto it, the cart's lowest and highest heights so far) and is fed one observation per
control step by ``task.success()``.

    Tracker.update(cart, runner_a, runner_b, arms) -> Result(score, success, details)

``cart`` and the runners are anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and
``high``, as ``yam_arena.task_file.ObjectState`` has; each is a box of known size (``layout.CART_SIZE``,
``layout.RUNNER_SIZE``), so its corners follow from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``:
``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and ``opening`` (0 closed
.. 1 open), or ``None`` when the arms could not be read; then the hands-off condition is taken as met for that step
and the step is counted in ``arms_unavailable``. A step on which any of the three objects has a non-finite number
or a zero quaternion (a simulator blow-up), or is missing, scores 0 and leaves the tracker as it was.

The cart and the runners are plain boxes, the same either way up, so tilt is measured from whichever broad face is
nearer to facing down, and the cart's square footprint is 'square to the bed' at any multiple of 90 degrees.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a finger's outer corner (half the current finger gap plus
the finger's 10 mm thickness, and half its 40 mm width across near its base; 60 mm fully open, 22 mm closed).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
CART_HALF = np.array(L.CART_HALF, dtype=float)
RUNNER_HALF = np.array(L.RUNNER_HALF, dtype=float)
RUNNER_NAMES = ("runner_a", "runner_b")
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


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    if s is None:
        return False
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


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- boxes ------------------------------------------------------------------------------------------------------
def corners(s, half) -> np.ndarray:
    """The box's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * np.asarray(half, dtype=float)) @ rotation(s.quat_xyzw).T


def lowest(s, half) -> float:
    """Height of the box's lowest point above the table top (m)."""
    return float(corners(s, half)[:, 2].min() - L.TOP)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through the box's two broad faces (degrees, 0..90): a plain
    box is the same either way up, so whichever broad face is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def yaw_err_deg(quat_xyzw) -> float:
    """How far the cart's sides are turned from the bed's axes (0..45 degrees; its square outline has 4 ways
    round)."""
    r = rotation(quat_xyzw)
    a = math.degrees(math.atan2(r[1, 0], r[0, 0])) % 90.0
    return float(min(a, 90.0 - a))


def point_box_distance(p, s, half) -> float:
    """Distance from a point to the box (0 inside it), in the box's own frame."""
    r = rotation(s.quat_xyzw)
    d = r.T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(d) - np.asarray(half, dtype=float), 0.0)))


def _xy_box(s, half) -> tuple[float, float, float, float]:
    c = corners(s, half)
    return float(c[:, 0].min()), float(c[:, 0].max()), float(c[:, 1].min()), float(c[:, 1].max())


def _inside(pts, rect, tol: float = 0.0) -> bool:
    x0, x1, y0, y1 = rect
    return bool(np.all((pts[:, 0] >= x0 - tol) & (pts[:, 0] <= x1 + tol)
                       & (pts[:, 1] >= y0 - tol) & (pts[:, 1] <= y1 + tol)))


# -- the cart ---------------------------------------------------------------------------------------------------
def cart_low(cart) -> float:
    """Height of the cart's lowest point above the table top (m): 40 mm on the dock, 39.5 on the runners, 39 on
    the bed."""
    return lowest(cart, CART_HALF)


def cart_dropped(cart) -> bool:
    return cart_low(cart) < L.CART_DROP


def cart_lifted(cart) -> bool:
    return cart_low(cart) > L.CART_LIFT


def cart_tipped(cart) -> bool:
    return tilt_deg(cart.quat_xyzw) > L.CART_TIP_DEG


def on_bed(cart) -> bool:
    """The cart's whole outline over the truck bed's top, clear of its seats (x 0.230..0.410, y +0.085..+0.300),
    and its lowest point within 2 mm of the bed top."""
    c = corners(cart, CART_HALF)
    return bool(_inside(c[:, :2], (L.X0, L.X1, L.SEAT_Y, L.BED_Y_END))
                and abs(cart_low(cart) - L.BED_TOP) <= L.BED_Z_TOL)


def over_gap(cart) -> bool:
    """Some part of the cart's outline over the gap (more than 1 mm into y -0.040 .. +0.040)."""
    _, _, y0, y1 = _xy_box(cart, CART_HALF)
    return bool(y1 > -L.GAP_Y + L.OVER_GAP_TOL and y0 < L.GAP_Y - L.OVER_GAP_TOL)


def centre_off(cart) -> float:
    """Horizontal distance from the cart's centre to the target (m)."""
    return float(math.hypot(float(cart.pos[0]) - L.TARGET_X, float(cart.pos[1]) - L.TARGET_Y))


def past_handover(cart) -> bool:
    """The cart's centre at or past y +0.040: over the truck bed's side of the gap."""
    return bool(float(cart.pos[1]) >= L.HANDOVER_Y)


def rear_y(cart) -> float:
    """y of the cart's rear (most negative) side: where a push toward the bed bears on it."""
    return float(corners(cart, CART_HALF)[:, 1].min())


# -- the runners ------------------------------------------------------------------------------------------------
def runner_ends(runner) -> tuple[np.ndarray, np.ndarray]:
    """The four corners at each end of the runner (world, 4 x 3 each), the end with the lower y first."""
    c = corners(runner, RUNNER_HALF)
    u = rotation(runner.quat_xyzw)[:, 1]  # along its length
    s = (c - np.asarray(runner.pos, dtype=float)) @ u
    a, b = c[s < 0], c[s > 0]
    return (a, b) if a[:, 1].mean() <= b[:, 1].mean() else (b, a)


def end_centres(runner) -> np.ndarray:
    """Centres of the runner's two end faces (world, 2 x 3), the lower y first."""
    a, b = runner_ends(runner)
    return np.array([a.mean(axis=0), b.mean(axis=0)])


def seat_of(runner) -> str | None:
    """Which pair of seats the runner is seated in ('a' at x 0.260-0.310, 'b' at x 0.340-0.390), or None. Seated:
    lying flat (broad faces within 5 degrees of level), the corners of one end inside a dock seat and those of the
    other end inside the facing truck-bed seat, and its bottom within 2 mm of the seat floors at both ends."""
    if tilt_deg(runner.quat_xyzw) > L.SEAT_FLAT_DEG:
        return None
    dock_end, bed_end = runner_ends(runner)
    floor = L.TOP + L.SEAT_FLOOR
    if not all(abs(float(e[:, 2].min()) - floor) <= L.SEAT_Z_TOL for e in (dock_end, bed_end)):
        return None
    for key in L.SEATS:
        if (_inside(dock_end[:, :2], L.DOCK_SEAT[key], L.SEAT_XY_TOL)
                and _inside(bed_end[:, :2], L.BED_SEAT[key], L.SEAT_XY_TOL)):
            return key
    return None


def both_seated(runner_a, runner_b) -> bool:
    """Both runners seated, in different pairs of seats."""
    a, b = seat_of(runner_a), seat_of(runner_b)
    return bool(a is not None and b is not None and a != b)


def cart_on_runner(cart, runner) -> bool:
    """The cart resting on (or sliding over) the runner: their outlines overlap by 5 mm or more along x and along y
    and the cart's lowest point is within 3 mm of the runner's top."""
    cx0, cx1, cy0, cy1 = _xy_box(cart, CART_HALF)
    rx0, rx1, ry0, ry1 = _xy_box(runner, RUNNER_HALF)
    overlap = min(cx1, rx1) - max(cx0, rx0), min(cy1, ry1) - max(cy0, ry0)
    top = float(corners(runner, RUNNER_HALF)[:, 2].max()) - L.TOP
    return bool(min(overlap) >= L.ON_RUNNER_OVERLAP and abs(cart_low(cart) - top) <= L.ON_RUNNER_Z)


def off_table(s, half) -> bool:
    """The box's lowest point more than 10 mm below the table top: it fell off the table."""
    return lowest(s, half) < -L.OFF_TABLE


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger's outer corner: half the gap plus the finger's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def _points_box_distance(pts, s, half) -> float:
    """Smallest distance from a set of points (n, 3) to the box (0 inside it), in the box's own frame."""
    d = (np.asarray(pts, dtype=float) - np.asarray(s.pos, dtype=float)) @ rotation(s.quat_xyzw)
    return float(np.linalg.norm(np.maximum(np.abs(d) - np.asarray(half, dtype=float), 0.0), axis=1).min())


def zone_gap(arm, s, half) -> float:
    """How far a gripper's finger zone is from a box (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return _points_box_distance(pts, s, half) - r


def zone_gaps(arm, cart, runner_a, runner_b) -> float:
    """How far a gripper's finger zone is from the nearest of the cart and the two runners."""
    pts, r = finger_zone(arm)
    return min(_points_box_distance(pts, cart, CART_HALF), _points_box_distance(pts, runner_a, RUNNER_HALF),
               _points_box_distance(pts, runner_b, RUNNER_HALF)) - r


def hands_off(arms, cart, runner_a, runner_b) -> bool:
    """Both grippers' finger zones at least 30 mm from the cart and both runners: neither can be pushing, holding
    or touching any of them."""
    return bool(all(zone_gaps(arms[s], cart, runner_a, runner_b) >= L.HANDS_OFF for s in ARMS))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "dropped": "the cart's lowest point went below 35 mm above the table (into the gap, or off the dock)",
    "lifted": "the cart's lowest point rose above 45 mm (it was lifted, not pushed)",
    "tipped": "the cart tilted more than 10 degrees",
    "runner_slid": "a runner's end moved more than 10 mm while the cart was on it",
    "off_table": "the cart or a runner fell off the table",
    "open_gap": "part of the cart came over the gap while the two runners were not both seated",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.min_low = math.inf  # m: the cart's lowest point, its lowest and highest so far
        self.max_low = -math.inf
        self.max_tilt = 0.0  # deg
        self.ref: dict[str, np.ndarray | None] = {n: None for n in RUNNER_NAMES}  # end centres when the cart came on
        self.shift = {n: 0.0 for n in RUNNER_NAMES}  # m: how far each runner has moved since the cart came onto it
        self.max_shift = {n: 0.0 for n in RUNNER_NAMES}
        self.both_seated_step: int | None = None  # the first step both runners were seated
        self.handover_step: int | None = None  # the first step the cart's centre was past y +0.040
        self.min_zone = {s: math.inf for s in ARMS}  # m: each finger zone's closest approach to cart or runners
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite or missing object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # {name: (centre, velocity)} at the previous valid control step

    # -- reset detection ---------------------------------------------------------------------------------------
    @staticmethod
    def at_start(cart, runner_a, runner_b) -> bool:
        """The starting layout: the cart on the dock and both runners on the table near their starting places,
        flat and still."""
        cart_ok = (math.hypot(float(cart.pos[0]) - L.CART_X, float(cart.pos[1]) - L.CART_Y) < L.START_TOL
                   and abs(cart_low(cart) - L.DOCK_TOP) < 0.003)
        runners_ok = all(
            math.hypot(float(r.pos[0]) - L.RUNNERS[n][0], float(r.pos[1]) - L.RUNNERS[n][1]) < L.START_TOL
            and abs(lowest(r, RUNNER_HALF)) < 0.003
            for n, r in zip(RUNNER_NAMES, (runner_a, runner_b)))
        flat = all(tilt_deg(s.quat_xyzw) < 1.0 for s in (cart, runner_a, runner_b))
        still = all(is_resting(s) for s in (cart, runner_a, runner_b))
        return bool(cart_ok and runners_ok and flat and still)

    def _teleported(self, name: str, c) -> bool:
        """The object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        p0, v0 = self.prev[name]
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall

    def _flag(self, key: str) -> None:
        self.breaches.setdefault(key, self.steps)

    # -- per step ------------------------------------------------------------------------------------------------
    def update(self, cart, runner_a, runner_b, arms=None) -> Result:
        objs = {"cart": cart, "runner_a": runner_a, "runner_b": runner_b}
        if not all(valid(s) for s in objs.values()):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(cart, runner_a, runner_b, arms)
            return self.last
        now = {n: np.array(s.pos, dtype=float) for n, s in objs.items()}
        if (self.prev is not None and any(self._teleported(n, now[n]) for n in objs)
                and self.at_start(cart, runner_a, runner_b)):
            self.reset()  # the simulator put everything back: a new episode (rigstate's on_reset hook normally does
            #               this first; this is the fallback)
        self.prev = {n: (now[n], np.array(s.lin_vel, dtype=float)) for n, s in objs.items()}
        self.steps += 1

        low, tilt = cart_low(cart), tilt_deg(cart.quat_xyzw)
        self.min_low, self.max_low = min(self.min_low, low), max(self.max_low, low)
        self.max_tilt = max(self.max_tilt, tilt)
        if low < L.CART_DROP:
            self._flag("dropped")
        if low > L.CART_LIFT:
            self._flag("lifted")
        if tilt > L.CART_TIP_DEG:
            self._flag("tipped")
        if off_table(cart, CART_HALF) or any(off_table(r, RUNNER_HALF) for r in (runner_a, runner_b)):
            self._flag("off_table")
        for name, runner in (("runner_a", runner_a), ("runner_b", runner_b)):
            ends = end_centres(runner)
            if cart_on_runner(cart, runner):
                if self.ref[name] is None:
                    self.ref[name] = ends
                d = float(np.max(np.linalg.norm(ends - self.ref[name], axis=1)))
                self.shift[name] = d
                self.max_shift[name] = max(self.max_shift[name], d)
                if d > L.RUNNER_SHIFT_MAX:
                    self._flag("runner_slid")
            else:
                self.ref[name], self.shift[name] = None, 0.0
        seated_now = both_seated(runner_a, runner_b)
        if over_gap(cart) and not seated_now:
            self._flag("open_gap")
        if self.both_seated_step is None and seated_now:
            self.both_seated_step = self.steps
        if self.handover_step is None and past_handover(cart):
            self.handover_step = self.steps
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(cart, runner_a, runner_b, arms, _feed=True)
        return self.last

    def result(self, cart, runner_a, runner_b, arms=None, _feed: bool = False) -> Result:
        """The score as things stand (no change to the record; ``update`` passes ``_feed`` to record the finger
        zones' closest approach as well)."""
        if not all(valid(s) for s in (cart, runner_a, runner_b)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        zones = {s: zone_gaps(arms[s], cart, runner_a, runner_b) for s in ARMS} if have_arms else None
        if _feed and zones:
            for side in ARMS:
                self.min_zone[side] = min(self.min_zone[side], zones[side])
        off = all(z >= L.HANDS_OFF for z in zones.values()) if have_arms else True  # unreadable: taken as met
        low = cart_low(cart)
        seats = (seat_of(runner_a), seat_of(runner_b))
        seated = bool(seats[0] is not None and seats[1] is not None and seats[0] != seats[1])
        gap = over_gap(cart)
        broken = bool(self.breaches) or low < L.CART_DROP or low > L.CART_LIFT or cart_tipped(cart) or \
            off_table(cart, CART_HALF) or off_table(runner_a, RUNNER_HALF) or off_table(runner_b, RUNNER_HALF) or \
            (gap and not seated)
        bed, dist, yaw_err = on_bed(cart), centre_off(cart), yaw_err_deg(cart.quat_xyzw)
        if broken:
            score = 0.0
        elif seated and bed and dist <= L.CENTRE_TIGHT and yaw_err <= L.YAW_TIGHT_DEG and off:
            score = 1.0
        elif seated and bed and dist <= L.CENTRE_LOOSE:
            score = 0.75
        elif seated and past_handover(cart):
            score = 0.5
        elif seated:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and all(is_resting(s) for s in (cart, runner_a, runner_b)))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "seat_a": seats[0],
            "seat_b": seats[1],
            "both_seated": seated,
            "both_seated_step": self.both_seated_step,
            "cart_low_mm": _mm(low),
            "min_cart_low_mm": _mm(self.min_low),
            "max_cart_low_mm": _mm(self.max_low),
            "cart_tilt_deg": round(tilt_deg(cart.quat_xyzw), 2),
            "max_cart_tilt_deg": round(self.max_tilt, 2),
            "cart_yaw_err_deg": round(yaw_err, 2),
            "cart_x_off_mm": _mm(float(cart.pos[0]) - L.TARGET_X),
            "cart_y_mm": _mm(float(cart.pos[1])),
            "cart_rear_y_mm": _mm(rear_y(cart)),
            "centre_off_mm": _mm(dist),
            "on_bed": bed,
            "cart_over_gap": gap,
            "handover_step": self.handover_step,
            "cart_on_runner": [n for n, r in (("runner_a", runner_a), ("runner_b", runner_b))
                               if cart_on_runner(cart, r)],
            "runner_shift_mm": {n: _mm(v) for n, v in self.shift.items()},
            "max_runner_shift_mm": {n: _mm(v) for n, v in self.max_shift.items()},
            "runner_low_mm": {n: _mm(lowest(r, RUNNER_HALF)) for n, r in (("runner_a", runner_a),
                                                                          ("runner_b", runner_b))},
            "left_zone_mm": _mm(zones["left"]) if have_arms else None,
            "right_zone_mm": _mm(zones["right"]) if have_arms else None,
            "min_left_zone_mm": _mm(self.min_zone["left"]),
            "min_right_zone_mm": _mm(self.min_zone["right"]),
            "rear_minus_tcp_mm": ({s: _mm(rear_y(cart) - float(arms[s].tcp[1])) for s in ARMS}
                                  if have_arms else None),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "hands_off": off if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "resting": all(is_resting(s) for s in (cart, runner_a, runner_b)),
        }
        return Result(score, success, details)
