"""The scorer, as plain functions of the die's, the safety block's and the part's states and the arms' state. NumPy
only; no simulator needed.

The rule is about the whole trial, not only its end: no gripper may reach between the die's faces until the die is
blocked open (pulled open by its handle, the yellow block's bar seated square in its white seat, the spring pressing
the moving platen on it, no hand left at the handle post and none holding the platen), and the part and the block
must never fall. So the scorer keeps a little state
(what has happened, and for how long the die has been blocked) and is fed one observation per control step by
``task.success()``.

    Tracker.update(die, block, part, arms) -> Result(score, success, details)

``die`` and ``block`` are mechkit ``AssemblyState``s (``joints["platen"]``, ``joint_vel["platen"]``, ``origin``,
``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low``, ``high``); ``part`` is anything with ``pos``, ``quat_xyzw``,
``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp`` (tool point), ``tips`` (the point between the
finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then every rule about
the grippers is taken as kept for that step (nobody is in the pinch zone, nobody is near or holding the handle post,
both grippers have let go) and the step is counted in ``arms_unavailable``. A state with a non-finite number or a
zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The die is fixed, so its moving platen's parts are where they were drawn, moved by -j along x (j = the opening).
The block's bar and handle arm are the same height and lie on the same bottom face, so the block is the same either
broad face down: its tilt is measured from whichever is nearer to facing down.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
BED_TOP = L.TOP + L.BED_TOP_Z
STAND_TOP = L.TOP + L.STAND_TOP_Z
TRAY_FLOOR = L.TOP + L.TRAY_FLOOR_Z
PART_START_LOW = L.TOP + L.BED_SLAB_Z


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


def _finite(v, n: int | None = None) -> bool:
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    return bool(a.size and np.all(np.isfinite(a)) and (n is None or a.size == n))


def valid(s) -> bool:
    """Every number finite and the quaternion not degenerate (a simulator blow-up gives NaN or zeros)."""
    try:
        parts = [getattr(s, k) for k in _KEYS]
    except AttributeError:
        return False
    sizes = (3, 4, 3, 3, 3, 3)
    if not all(_finite(p, n) for p, n in zip(parts, sizes)):
        return False
    return bool(np.linalg.norm(np.asarray(parts[1], dtype=float)) > 0.5)


def valid_assembly(s, joint: str | None = None) -> bool:
    """``valid`` plus a finite base-frame origin and, if named, a finite joint position and speed."""
    if not valid(s) or not _finite(getattr(s, "origin", None), 3):
        return False
    if joint is None:
        return True
    try:
        return bool(math.isfinite(float(s.joints[joint])) and math.isfinite(float(s.joint_vel[joint])))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with a finite tool point, tip point and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            a = arms[side]
            if not (_finite(a.tcp, 3) and _finite(a.tips, 3) and math.isfinite(float(a.opening))):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


def tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and the block's (or part's) vertical axis, either way up (0..90 degrees)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo, dtype=float) - p, p - np.asarray(hi, dtype=float)))
    return float(np.linalg.norm(d))


def _box(spec) -> tuple[np.ndarray, np.ndarray]:
    """((x0, x1), (y0, y1), (z0, z1)) above the table top -> world (low, high)."""
    (x0, x1), (y0, y1), (z0, z1) = spec
    return np.array([x0, y0, L.TOP + z0]), np.array([x1, y1, L.TOP + z1])


def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


# -- the die -------------------------------------------------------------------------------------------------------
def opening(die) -> float:
    """j: how far the moving platen is pulled open (m, 0 shut)."""
    return float(die.joints["platen"])


def opening_speed(die) -> float:
    """dj/dt (m/s): positive opening, negative closing."""
    return float(die.joint_vel["platen"])


def moving_face_x(j: float) -> float:
    return L.MOVING_FACE_X0 - j


def post_box(j: float) -> tuple[np.ndarray, np.ndarray]:
    """The black handle post's world box with the die open j."""
    lo, hi = _box(L.POST)
    return lo - [j, 0.0, 0.0], hi - [j, 0.0, 0.0]


def holdable_parts(j: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """The moving platen's boxes a hand outside the pinch zone could hold it open by (face plate, bracket, post),
    with the die open j. The cheeks are left out: they stand inside the zone, beside the part."""
    return [(lo - [j, 0.0, 0.0], hi - [j, 0.0, 0.0])
            for lo, hi in (_box(p) for p in (L.FACE_PLATE, L.BRACKET, L.POST))]


def platen_box(j: float) -> tuple[np.ndarray, np.ndarray]:
    """The box round the whole moving platen (face plate, cheeks, bracket, post) with the die open j."""
    parts = [_box(p) for p in (L.FACE_PLATE, L.CHEEK_LEFT, L.CHEEK_RIGHT, L.BRACKET, L.POST)]
    lo = np.min([p[0] for p in parts], axis=0) - [j, 0.0, 0.0]
    hi = np.max([p[1] for p in parts], axis=0) - [j, 0.0, 0.0]
    return lo, hi


def in_pinch_zone(p, j: float) -> bool:
    """A point between the die's faces: from 10 mm in front of the moving face to 10 mm into the fixed platen, 10 mm
    beyond the faces' ends, from the table top up to 140 mm."""
    x, y, z = (float(v) for v in p)
    return bool(moving_face_x(j) - L.ZONE_MARGIN <= x <= L.ZONE_X1 and L.ZONE_Y0 <= y <= L.ZONE_Y1
                and L.TOP <= z <= L.TOP + L.ZONE_Z1)


def die_footprint_xy(p) -> bool:
    return bool(L.BED_X0 <= float(p[0]) <= L.BED_X1 and L.BED_Y0 <= float(p[1]) <= L.BED_Y1)


# -- the block -----------------------------------------------------------------------------------------------------
def _corners(origin, quat, center, size) -> np.ndarray:
    half = np.array(size, dtype=float) / 2
    local = np.asarray(center, dtype=float) + _SIGNS * half
    return np.asarray(origin, dtype=float) + local @ rotation(quat).T


def bar_corners(block) -> np.ndarray:
    """The eight corners of the block's bar (world, 8 x 3)."""
    return _corners(block.origin, block.quat_xyzw, L.BAR_CENTER, L.BAR_SIZE)


def block_corners(block) -> np.ndarray:
    """The corners of the bar and the handle arm (world, 16 x 3)."""
    return np.vstack([bar_corners(block), _corners(block.origin, block.quat_xyzw, L.ARM_CENTER, L.ARM_SIZE)])


def bar_yaw_deg(block) -> float:
    """How far the bar's long axis is turned from x, seen from above (0..90 degrees, either end first)."""
    ax = rotation(block.quat_xyzw)[:, 0]
    a = abs(math.degrees(math.atan2(float(ax[1]), float(ax[0]))))
    return float(min(a, 180.0 - a))


def bar_seat_offset(block) -> float:
    """How far (m) the bar's outline seen from above sticks out of the white seat (0: inside it)."""
    c = bar_corners(block)
    out = np.maximum.reduce([L.SEAT_X0 - c[:, 0], c[:, 0] - L.SEAT_X1, L.SEAT_Y0 - c[:, 1], c[:, 1] - L.SEAT_Y1,
                             np.zeros(len(c))])
    return float(out.max())


def block_seated(block, j: float) -> dict:
    """The mechanical half of 'blocked': the bar lying flat and square on the bed in its seat, the die open
    0.045-0.060 m and the moving face resting on the bar's near end. Returns each condition."""
    c = bar_corners(block)
    low = float(c[:, 2].min())
    front = float(c[:, 0].min())
    gap = front - moving_face_x(j)  # moving face to the bar's near end (m)
    out = {
        "bar_on_bed": abs(low - BED_TOP) <= L.BAR_ON_BED_TOL,
        "bar_in_seat": bar_seat_offset(block) <= L.SEAT_TOL,
        "bar_square": bar_yaw_deg(block) <= L.BAR_YAW_DEG and tilt_deg(block.quat_xyzw) <= L.BAR_TILT_DEG,
        "j_in_band": L.BLOCKED_J[0] <= j <= L.BLOCKED_J[1],
        "face_on_bar": -L.FACE_ON_BAR_TOL <= gap <= L.FACE_ON_BAR_TOL,
    }
    out["seated"] = all(out.values())
    out["face_gap_mm"] = round(gap * 1000, 1)
    return out


def block_on_stand(block) -> bool:
    """The bar's outline seen from above inside the stand's pad, the block lying flat on the pad (its lowest point
    within 3 mm of the pad's top, tilted under 10 degrees)."""
    c = bar_corners(block)
    hx, hy = L.STAND_SIZE[0] / 2, L.STAND_SIZE[1] / 2
    inside = bool(np.all(np.abs(c[:, 0] - L.STAND_X) <= hx) and np.all(np.abs(c[:, 1] - L.STAND_Y) <= hy))
    low = float(block_corners(block)[:, 2].min())
    return bool(inside and abs(low - STAND_TOP) <= L.STAND_Z_TOL and tilt_deg(block.quat_xyzw) < L.BAR_TILT_DEG)


# -- the part ------------------------------------------------------------------------------------------------------
def part_in_tray(part) -> bool:
    """The part's centre inside the tray's walls (over its floor) and its lowest point within 5 mm of the floor."""
    p = np.asarray(part.pos, dtype=float)
    inside = abs(p[0] - L.TRAY_X) <= L.TRAY_INNER[0] / 2 and abs(p[1] - L.TRAY_Y) <= L.TRAY_INNER[1] / 2
    return bool(inside and abs(float(part.low[2]) - TRAY_FLOOR) <= L.IN_TRAY_Z_TOL)


def part_lift(part) -> float:
    """How far (m) the part's lowest point is above where it started (the pocket's floor)."""
    return float(part.low[2]) - PART_START_LOW


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE)


# -- the arms ------------------------------------------------------------------------------------------------------
def _points(arm) -> list[np.ndarray]:
    return [np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)]


def near_post(arm, j: float) -> float:
    """Distance (m) from the gripper's tool point or finger-tip point, whichever is nearer, to the handle post."""
    lo, hi = post_box(j)
    return min(point_box_distance(p, lo, hi) for p in _points(arm))


def holding_post(arm, j: float) -> bool:
    """A gripper closed (opening under 0.9) with its tool point within 10 mm of the handle post: it holds the die."""
    lo, hi = post_box(j)
    return bool(float(arm.opening) < L.GRIP_OPEN
                and point_box_distance(np.asarray(arm.tcp, dtype=float), lo, hi) <= L.HELD_POST_DIST)


def holding_platen(arm, j: float) -> bool:
    """A gripper closed (opening under 0.9) with its tool point within 10 mm of the moving platen's face plate,
    bracket or post: it holds the die, wherever on them it took hold."""
    tcp = np.asarray(arm.tcp, dtype=float)
    return bool(float(arm.opening) < L.GRIP_OPEN
                and min(point_box_distance(tcp, lo, hi) for lo, hi in holdable_parts(j)) <= L.HELD_PLATEN_DIST)


def clearance(arm, boxes) -> float:
    """The smallest distance from the gripper's tool point or finger-tip point to any of ``boxes``."""
    return min(point_box_distance(p, lo, hi) for p in _points(arm) for lo, hi in boxes)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


ZERO_RULES = {
    "reached_in_unblocked": "a gripper's tool point was in the pinch zone while the die was not blocked",
    "part_fell": "the part moved down faster than 0.25 m/s",
    "block_fell": "the block moved down faster than 0.25 m/s",
    "part_off_table": "the part left the table",
    "block_off_table": "the block left the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.zeroes: dict[str, int] = {}  # zero rule -> the control step it first happened
        self.blocked_run = 0  # control steps in a row the die has been blocked
        self.blocked_steps = 0  # control steps blocked in all
        self.blocked_1s_step: int | None = None  # the step at which it had been blocked for 1 s in a row
        self.pulled_open = False  # since the die was last shut, a gripper has held the post with j >= 45 mm
        self.slammed_step: int | None = None
        self.removed_step: int | None = None  # the part lifted 30 mm and outside the die's footprint
        self.max_j = 0.0
        self.max_lift = 0.0
        self.fastest_drop = 0.0  # m/s: the part's or the block's fastest downward speed
        self.fastest_close = 0.0  # m/s: the platen's fastest closing speed while nobody held the post
        self.min_zone_entry_j = None  # j when a tool point was first in the pinch zone (blocked or not)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the part, of the block] at the previous valid control step

    @property
    def zeroed(self) -> bool:
        return bool(self.zeroes)

    @property
    def blocked_1s(self) -> bool:
        return self.blocked_1s_step is not None

    @staticmethod
    def at_start(die, block, part) -> bool:
        """The starting layout: the die shut, the part standing in its pocket, the block flat on its stand, all
        still (within START_TOL of their places, which covers a seeded reset's jitter)."""
        p_ok = (math.hypot(part.pos[0] - L.PART_X, part.pos[1] - L.PART_Y) < L.START_TOL
                and abs(float(part.low[2]) - PART_START_LOW) < 0.003 and tilt_deg(part.quat_xyzw) < 1.0)
        b_ok = (math.hypot(block.origin[0] - L.BLOCK_ORIGIN[0], block.origin[1] - L.BLOCK_ORIGIN[1]) < L.START_TOL
                and abs(float(block.low[2]) - STAND_TOP) < 0.003 and tilt_deg(block.quat_xyzw) < 1.0)
        return bool(p_ok and b_ok and opening(die) < 0.005 and is_resting(part) and is_resting(block))

    def _teleported(self, now) -> bool:
        """The part or the block moved, in one control step, RESET_JUMP further than its last velocity plus a free
        fall could have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(p - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall
            for (p, _), (p0, v0) in zip(now, self.prev)
        )

    def _flag(self, key: str) -> None:
        self.zeroes.setdefault(key, self.steps)

    @staticmethod
    def _valid(die, block, part) -> bool:
        return valid_assembly(die, "platen") and valid_assembly(block) and valid(part)

    def update(self, die, block, part, arms=None) -> Result:
        if not self._valid(die, block, part):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(die, block, part, arms)
            return self.last
        now = [(np.array(part.pos, dtype=float), np.array(part.lin_vel, dtype=float)),
               (np.array(block.origin, dtype=float), np.array(block.lin_vel, dtype=float))]
        if self.prev is not None and self._teleported(now) and self.at_start(die, block, part):
            self.reset()  # the simulator put everything back: a new episode (rigstate's and mechkit's reset hooks
            #               normally do this first; this is the fallback)
        self.prev = now
        self.steps += 1

        j, jv = opening(die), opening_speed(die)
        self.max_j = max(self.max_j, j)
        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1
        seated = block_seated(block, j)
        hand_at_post = have_arms and any(near_post(arms[s], j) < L.HAND_NEAR_POST for s in ("left", "right"))
        held = have_arms and any(holding_post(arms[s], j) for s in ("left", "right"))
        platen_held = have_arms and any(holding_platen(arms[s], j) for s in ("left", "right"))
        if j <= L.CLOSED_J:
            self.pulled_open = False
        elif j >= L.PULLED_OPEN_J and (held or not have_arms):  # unread arms: taken as pulled by its handle
            self.pulled_open = True
        blocked = bool(seated["seated"] and self.pulled_open and not hand_at_post and not platen_held)
        if blocked:
            self.blocked_run += 1
            self.blocked_steps += 1
            if self.blocked_1s_step is None and self.blocked_run * self.dt >= L.BLOCKED_HOLD_S - 1e-9:
                self.blocked_1s_step = self.steps
        else:
            self.blocked_run = 0
        if have_arms:
            inside = [s for s in ("left", "right") if in_pinch_zone(arms[s].tcp, j)]
            if inside and self.min_zone_entry_j is None:
                self.min_zone_entry_j = j
            if inside and not blocked:
                self._flag("reached_in_unblocked")
            if not held:  # with the arms unread nobody counts as letting go of the post
                self.fastest_close = max(self.fastest_close, -jv)
                if -jv > L.SLAM_SPEED and self.slammed_step is None:
                    self.slammed_step = self.steps
        for key, s in (("part", part), ("block", block)):
            drop = -float(s.lin_vel[2])
            self.fastest_drop = max(self.fastest_drop, drop)
            if drop > L.FALL_SPEED:
                self._flag(f"{key}_fell")
            if off_table(s):
                self._flag(f"{key}_off_table")
        lift = part_lift(part)
        self.max_lift = max(self.max_lift, lift)
        if self.removed_step is None and lift >= L.LIFT_MIN and not die_footprint_xy(part.pos):
            self.removed_step = self.steps
        self.last = self.result(die, block, part, arms, _seated=seated, _blocked=blocked, _held=held,
                                _hand=hand_at_post, _platen_held=platen_held)
        return self.last

    def result(self, die, block, part, arms=None, *, _seated=None, _blocked=None, _held=None, _hand=None,
               _platen_held=None) -> Result:
        if not self._valid(die, block, part):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "zero_rules": sorted(self.zeroes)})
        j, jv = opening(die), opening_speed(die)
        have_arms = arms_valid(arms)
        seated = _seated if _seated is not None else block_seated(block, j)
        if _held is None:
            _held = have_arms and any(holding_post(arms[s], j) for s in ("left", "right"))
        if _hand is None:
            _hand = have_arms and any(near_post(arms[s], j) < L.HAND_NEAR_POST for s in ("left", "right"))
        if _platen_held is None:
            _platen_held = have_arms and any(holding_platen(arms[s], j) for s in ("left", "right"))
        blocked = (_blocked if _blocked is not None
                   else bool(seated["seated"] and self.pulled_open and not _hand and not _platen_held))
        held_or_unknown = _held or not have_arms
        shut_on_something = L.CLOSED_ON_J[0] < j < L.CLOSED_ON_J[1] and not held_or_unknown
        in_tray = part_in_tray(part)
        on_stand = block_on_stand(block)
        closed = j <= L.CLOSED_J
        boxes = [_lo_hi(part), (np.min(block_corners(block), axis=0), np.max(block_corners(block), axis=0)),
                 platen_box(j)]
        if have_arms:
            gaps = {s: clearance(arms[s], boxes) for s in ("left", "right")}
            let_go = all(float(arms[s].opening) >= L.GRIP_OPEN and gaps[s] >= L.CLEAR_DIST for s in ("left", "right"))
        else:
            gaps, let_go = None, True  # unreadable: taken as met, counted in arms_unavailable
        slammed = self.slammed_step is not None
        if self.zeroed or shut_on_something:
            score = 0.0
        elif in_tray and self.blocked_1s and on_stand and closed and not slammed and let_go:
            score = 1.0
        elif in_tray and self.blocked_1s:
            score = 0.75
        elif self.blocked_1s and self.removed_step is not None:
            score = 0.5
        elif self.blocked_1s:
            score = 0.25
        else:
            score = 0.0
        still = (is_resting(part) and is_resting(block) and abs(jv) < L.STILL_JOINT)
        success = bool(score == 1.0 and still)
        c = bar_corners(block)
        details = {
            "valid_state": True,
            "zero_rules": sorted(self.zeroes),
            "first_zero_step": min(self.zeroes.values()) if self.zeroes else None,
            "shut_on_something": shut_on_something,
            "j_mm": _mm(j),
            "j_speed_mps": round(jv, 4),
            "max_j_mm": _mm(self.max_j),
            "blocked": blocked,
            **{k: v for k, v in seated.items() if k != "seated"},
            "block_seated": seated["seated"],
            "pulled_open": self.pulled_open,
            "hand_at_post": _hand,
            "post_held": _held,
            "platen_held": _platen_held,
            "blocked_run_s": round(self.blocked_run * self.dt, 2),
            "blocked_s_total": round(self.blocked_steps * self.dt, 2),
            "blocked_1s_step": self.blocked_1s_step,
            "zone_entry_j_mm": _mm(self.min_zone_entry_j),
            "slammed_step": self.slammed_step,
            "fastest_close_mps": round(self.fastest_close, 3),
            "fastest_drop_mps": round(self.fastest_drop, 3),
            "bar_low_mm": _mm(float(c[:, 2].min()) - L.TOP),
            "bar_front_x_mm": _mm(float(c[:, 0].min())),
            "bar_seat_out_mm": _mm(bar_seat_offset(block)),
            "bar_yaw_deg": round(bar_yaw_deg(block), 2),
            "block_tilt_deg": round(tilt_deg(block.quat_xyzw), 2),
            "block_on_stand": on_stand,
            "part_in_tray": in_tray,
            "part_lift_mm": _mm(part_lift(part)),
            "max_part_lift_mm": _mm(self.max_lift),
            "part_removed_step": self.removed_step,
            "die_closed": closed,
            "grippers_let_go": let_go if have_arms else None,
            "gripper_clear_mm": {s: _mm(g) for s, g in gaps.items()} if gaps else None,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "part_resting": is_resting(part),
            "block_resting": is_resting(block),
        }
        return Result(score, success, details)
