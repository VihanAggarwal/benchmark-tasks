"""The scorer, as plain functions of the two boards', the cup's and the arms' state. NumPy only; no simulator needed.

The rules are about the whole trial, not only its end: a gripper may touch a board only while it is grounded (it
becomes grounded by closing on the ground stud and stops being grounded when it touches the foam cup); no board
may touch the bare bench, be dropped or leave the table; the cup must not be knocked over or leave the table; and
the cup must be at least 30 cm from every board that is out of its shielding box. So the scorer keeps a little
state (each gripper's grounding, which rules have been broken, and when) and is fed one observation per control
step by ``task.success()``.

    Tracker.update(left_board, right_board, cup, arms) -> Result(score, success, details)

The boards and the cup are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``,
``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the finger tips) and
``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then the gripper rules are not
applied for that step (no touch is flagged, no grounding is seen) and the step is counted in ``arms_unavailable``.
An object state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and
leaves the tracker as it was.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is a solid round cylinder about the axis, from the point between its finger tips to 38 mm back up the gripper, with
flat ends; its radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the
8 mm pad thickness, and half the pad's 27 mm width across). That holds both finger pads whatever the gripper's turn
(and a little more), so 'the fingers touch a board' never misses a real touch, while a gripper above a board is
only as close to it as its finger tips are.

The boards are plain boxes with two identical 120 x 60 mm faces, so either may face down: a board's tilt is
measured from whichever is nearer to facing down. The cup has a top and a bottom: its tilt is its axis's angle
from upright (an upside-down cup is tilted 180 degrees).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

SIDES = ("left", "right")
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
MAT_LO = np.array([L.MAT_X - L.MAT_HALF_X, L.MAT_Y - L.MAT_HALF_Y])
MAT_HI = np.array([L.MAT_X + L.MAT_HALF_X, L.MAT_Y + L.MAT_HALF_Y])


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


def board_tilt_deg(quat_xyzw) -> float:
    """Angle between the world's vertical and the axis through the board's two 120 x 60 mm faces (degrees, 0..90):
    the board looks the same either way up, so whichever face is nearer to facing down counts."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def cup_tilt_deg(quat_xyzw) -> float:
    """Angle between the cup's axis and upright (degrees, 0..180): an upside-down cup is at 180."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def is_resting(s, max_speed: float = L.STILL_SPEED, max_spin: float = L.STILL_SPIN) -> bool:
    return bool(np.linalg.norm(s.lin_vel) < max_speed and np.linalg.norm(s.ang_vel) < max_spin)


# -- boxes and outlines -----------------------------------------------------------------------------------------
def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


def outline_gap(a, b) -> float:
    """Horizontal distance between two objects' outlines seen from above (their bounding boxes; 0 if they overlap)."""
    (lo_a, hi_a), (lo_b, hi_b) = _lo_hi(a), _lo_hi(b)
    d = np.maximum(0.0, np.maximum(lo_b[:2] - hi_a[:2], lo_a[:2] - hi_b[:2]))
    return float(np.linalg.norm(d))


def corners(s, size=L.BOARD_SIZE) -> np.ndarray:
    """The eight corners of a box of ``size`` (world, 8 x 3) from its centre and orientation."""
    half = np.array(size, dtype=float) / 2
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def over_mat_xy(p) -> bool:
    return bool(abs(p[0] - L.MAT_X) <= L.MAT_HALF_X and abs(p[1] - L.MAT_Y) <= L.MAT_HALF_Y)


def over_box_floor_xy(p, side: str, slack: float = L.BOX_FLOOR_SLACK) -> bool:
    return bool(abs(p[0] - L.BOX_X) <= L.BOX_INNER_HALF + slack and abs(p[1] - L.BOX_Y[side]) <= L.BOX_INNER_HALF + slack)


def over_any_box_floor_xy(p, slack: float = L.BOX_FLOOR_SLACK) -> bool:
    return any(over_box_floor_xy(p, side, slack) for side in SIDES)


# -- the arms ---------------------------------------------------------------------------------------------------
def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
    width across."""
    return float(math.hypot(finger_gap(arm) / 2 + L.FINGER_THICK, L.FINGER_HALF_WIDTH))


def finger_axis(arm) -> tuple[np.ndarray, np.ndarray]:
    """The point between the finger tips and the unit vector from there back up the gripper (toward the TCP)."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tcp - tips
    n = float(np.linalg.norm(u))
    return tips, (u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0]))


def _cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _perpendiculars(u) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to each other and to the unit vector u."""
    e1 = _cross(u, np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    return e1, _cross(u, e1)


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


def zone_point_gap(arm, q) -> float:
    """Distance from a point to a gripper's finger zone: the solid round cylinder about its axis from the finger
    tips to 38 mm back up the gripper, out to the zone radius (negative inside it, by the radial depth)."""
    tips, u = finger_axis(arm)
    return float(_cylinder_gap(tips, u, zone_radius(arm), np.asarray(q, dtype=float).reshape(1, 3))[0])


def finger_zone(arm) -> tuple[np.ndarray, float]:
    """Points on the zone's axis and on the rims of its two flat ends (tips and 38 mm up), and its radius."""
    tips, u = finger_axis(arm)
    r = zone_radius(arm)
    e1, e2 = _perpendiculars(u)
    circle = r * (np.outer(_COS, e1) + np.outer(_SIN, e2))
    return np.vstack([tips + np.outer(_AXIS_S, u), tips + circle, tips + L.FINGER_LEN * u + circle]), r


def corners_of(lo, hi) -> np.ndarray:
    """The eight corners of a world-aligned box."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return lo + (_SIGNS + 1) / 2 * (hi - lo)


def zone_gap(arm, s) -> float:
    """How far a gripper's finger zone is from an object's world-aligned bounding box (negative: they overlap).
    The zone is a flat-ended cylinder, so an open gripper 3 cm above a box is 3 cm from it (not touching). The
    smallest distance is searched over the box's points nearest to the zone's axis and rims, and its corners."""
    lo, hi = _lo_hi(s)
    pts, r = finger_zone(arm)
    tips, u = finger_axis(arm)
    cands = np.vstack([np.clip(pts, lo, hi), corners_of(lo, hi)])
    return float(np.min(_cylinder_gap(tips, u, r, cands)))


def on_stud(arm) -> bool:
    """The gripper is closed on the ground stud: its TCP within 15 mm (horizontally) of the stud's axis, the point
    between its finger tips at least 5 mm below the stud's top (and not below the table), and its opening 0.40 or
    less (fingers 37.6 mm apart or closer; closed on the 20 mm stud they stop at about 0.21). The stud is fixed, so
    fingers that closed that far with the stud between them are against it."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    near = math.hypot(tcp[0] - L.STUD_X, tcp[1] - L.STUD_Y) <= L.STUD_AXIS_TOL
    deep = L.TOP <= float(tips[2]) <= L.STUD_TOP - L.STUD_DEPTH
    return bool(near and deep and float(arm.opening) <= L.STUD_CLOSED)


def stud_distance(arm) -> float:
    """Horizontal distance from the gripper's TCP to the stud's axis (m)."""
    return float(math.hypot(float(arm.tcp[0]) - L.STUD_X, float(arm.tcp[1]) - L.STUD_Y))


def let_go(arm, boards) -> bool:
    """The gripper holds no board: it is open (opening 0.9 or more), or its finger zone is more than 10 mm from
    both boards. A gripper closed somewhere else holds nothing."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or all(zone_gap(arm, b) > L.LET_GO_GAP for b in boards))


def grippers_open(arms) -> bool:
    return bool(all(float(arms[s].opening) >= L.GRIPPER_OPEN for s in SIDES))


# -- the objects -------------------------------------------------------------------------------------------------
def in_box(board) -> bool:
    """Inside a shielding box: the board's centre over a box's 140 x 140 mm floor and its lowest point below the
    rim (the walls' top)."""
    return bool(over_any_box_floor_xy(board.pos, 0.0) and float(board.low[2]) < L.BOX_RIM_Z)


def overhang(board) -> float:
    """How far the board's outline seen from above reaches past the mat's edge (m; 0 if it is all over the mat)."""
    lo, hi = _lo_hi(board)
    return float(max(0.0, *(MAT_LO - lo[:2]), *(hi[:2] - MAT_HI)))


def on_mat(board, tol: float) -> bool:
    """Flat on the mat: tilted under 5 degrees, its lowest point within 3 mm of the mat's top, and its outline past
    the mat's edge by at most ``tol``."""
    return bool(board_tilt_deg(board.quat_xyzw) < L.FLAT_DEG and abs(float(board.low[2]) - L.MAT_TOP) <= L.ON_MAT_Z_TOL
                and overhang(board) <= tol + 1e-9)


def lowest_bare_corner(board) -> float | None:
    """Height above the table top of the board's lowest corner that is over neither the mat nor a box's floor
    (None if every corner is over one of them)."""
    out = [float(c[2]) - L.TOP for c in corners(board) if not (over_mat_xy(c) or over_any_box_floor_xy(c))]
    return min(out) if out else None


def on_park_square(board) -> bool:
    """Resting on the yellow parking square: the board's lowest point within 3 mm of the square's top while its
    outline overlaps the square. The square is bench (a place for the cup), not a place for boards."""
    lo, hi = _lo_hi(board)
    sq_lo = np.array([L.PARK_X - L.PARK_SIZE[0] / 2, L.PARK_Y - L.PARK_SIZE[1] / 2])
    sq_hi = np.array([L.PARK_X + L.PARK_SIZE[0] / 2, L.PARK_Y + L.PARK_SIZE[1] / 2])
    overlaps = bool(np.all(lo[:2] < sq_hi) and np.all(hi[:2] > sq_lo))
    return overlaps and abs(float(board.low[2]) - L.PARK_TOP) <= L.PARK_REST_TOL


def on_bare_bench(board) -> bool:
    """A corner of the board within 1 mm of the table top, or below it (off the table), where it is over neither the
    mat nor a box's floor, or the board resting on the parking square: the board touches the bare bench."""
    if on_park_square(board):
        return True
    h = lowest_bare_corner(board)
    return h is not None and h < L.BARE_BENCH_TOL


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "left_ungrounded_touch": "the left gripper's fingers came within 5 mm of a board while it was not grounded",
    "right_ungrounded_touch": "the right gripper's fingers came within 5 mm of a board while it was not grounded",
    "left_board_on_bench": "the left board touched the bare bench (or fell off the table)",
    "right_board_on_bench": "the right board touched the bare bench (or fell off the table)",
    "left_board_dropped": "the left board tilted more than 30 degrees",
    "right_board_dropped": "the right board tilted more than 30 degrees",
    "cup_knocked_over": "the cup tilted more than 30 degrees",
    "cup_off_table": "the cup left the table",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.grounded = {s: False for s in SIDES}  # is each gripper grounded now?
        self.first_ground_step: dict[str, int | None] = {s: None for s in SIDES}
        self.groundings = {s: 0 for s in SIDES}  # times each gripper closed on the stud (ungrounded before)
        self.cup_contacts = {s: 0 for s in SIDES}  # times a grounded gripper lost its grounding on the cup
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.cup_rule_step: int | None = None  # first step the cup was within 30 cm of a board out of its box
        self.first_out_step: dict[str, int | None] = {s: None for s in SIDES}  # first step each board was out
        self.min_cup_to_out_board = math.inf  # m: the closest the cup came to a board out of its box
        self.min_ungrounded_to_board = {s: math.inf for s in SIDES}  # m: closest ungrounded approach to a board
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of each object] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(boards, cup) -> bool:
        """The starting layout: each board flat in its box, the cup upright at its place on the mat, all still
        (within START_TOL of their places, which covers a seeded reset's jitter)."""
        ok = all(np.hypot(b.pos[0] - L.BOX_X, b.pos[1] - L.BOX_Y[side]) < L.START_TOL
                 and abs(float(b.low[2]) - L.TOP) < 0.003 and board_tilt_deg(b.quat_xyzw) < 1.0 and is_resting(b)
                 for side, b in zip(SIDES, boards))
        return bool(ok and np.hypot(cup.pos[0] - L.CUP_X, cup.pos[1] - L.CUP_Y) < L.START_TOL
                    and abs(float(cup.low[2]) - L.MAT_TOP) < 0.003 and cup_tilt_deg(cup.quat_xyzw) < 1.0
                    and is_resting(cup))

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

    def update(self, left_board, right_board, cup, arms=None) -> Result:
        boards = (left_board, right_board)
        if not all(valid(s) for s in (*boards, cup)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(left_board, right_board, cup, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (*boards, cup)]
        if self.prev is not None and self._teleported(now) and self.at_start(boards, cup):
            self.reset()  # the simulator put everything back in its place: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        if arms_valid(arms):
            for side in SIDES:
                a = arms[side]
                if on_stud(a):  # closing on the stud grounds the gripper
                    if not self.grounded[side]:
                        self.groundings[side] += 1
                    self.grounded[side] = True
                    if self.first_ground_step[side] is None:
                        self.first_ground_step[side] = self.steps
                elif zone_gap(a, cup) < L.TOUCH_GAP:  # touching the insulating cup charges it again
                    if self.grounded[side]:
                        self.cup_contacts[side] += 1
                    self.grounded[side] = False
                if not self.grounded[side]:
                    gap = min(zone_gap(a, b) for b in boards)
                    self.min_ungrounded_to_board[side] = min(self.min_ungrounded_to_board[side], gap)
                    if gap < L.TOUCH_GAP:
                        self._flag(f"{side}_ungrounded_touch")
        else:
            self.arms_unavailable += 1
        for side, b in zip(SIDES, boards):
            if not in_box(b):
                if self.first_out_step[side] is None:
                    self.first_out_step[side] = self.steps
                d = outline_gap(cup, b)
                self.min_cup_to_out_board = min(self.min_cup_to_out_board, d)
                if d < L.CUP_DISTANCE and self.cup_rule_step is None:
                    self.cup_rule_step = self.steps
            if on_bare_bench(b):
                self._flag(f"{side}_board_on_bench")
            if board_tilt_deg(b.quat_xyzw) > L.DROP_DEG:
                self._flag(f"{side}_board_dropped")
        if cup_tilt_deg(cup.quat_xyzw) > L.DROP_DEG:
            self._flag("cup_knocked_over")
        if off_table(cup):
            self._flag("cup_off_table")
        self.last = self.result(left_board, right_board, cup, arms)
        return self.last

    def result(self, left_board, right_board, cup, arms=None) -> Result:
        boards = (left_board, right_board)
        if not all(valid(s) for s in (*boards, cup)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        released = all(let_go(arms[s], boards) for s in SIDES) if have_arms else True  # unreadable: taken as met
        tight = [on_mat(b, L.OVERHANG_TIGHT) for b in boards]
        loose = [on_mat(b, L.OVERHANG_LOOSE) for b in boards]
        cup_ok = self.cup_rule_step is None
        apart = outline_gap(left_board, right_board)
        cup_now = [outline_gap(cup, b) for b in boards]
        ready = all(self.grounded.values()) and all(d >= L.CUP_DISTANCE for d in cup_now)
        if self.breach:
            score = 0.0
        elif cup_ok and all(tight) and apart >= L.BOARD_GAP and released:
            score = 1.0
        elif cup_ok and all(loose):
            score = 0.75
        elif (cup_ok and any(loose)) or all(loose):
            score = 0.5
        elif cup_ok and ready:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and all(is_resting(b) for b in boards))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "grounded": dict(self.grounded),
            "first_ground_step": dict(self.first_ground_step),
            "groundings": dict(self.groundings),
            "cup_contacts": dict(self.cup_contacts),
            "cup_rule_kept": cup_ok,
            "cup_rule_step": self.cup_rule_step,
            "first_out_step": dict(self.first_out_step),
            "in_box": {s: in_box(b) for s, b in zip(SIDES, boards)},
            "on_mat": {s: v for s, v in zip(SIDES, loose)},
            "fully_on_mat": {s: v for s, v in zip(SIDES, tight)},
            "overhang_mm": {s: _mm(overhang(b)) for s, b in zip(SIDES, boards)},
            "board_tilt_deg": {s: round(board_tilt_deg(b.quat_xyzw), 2) for s, b in zip(SIDES, boards)},
            "board_low_mm": {s: _mm(float(b.low[2]) - L.TOP) for s, b in zip(SIDES, boards)},
            "bare_corner_mm": {s: _mm(lowest_bare_corner(b)) for s, b in zip(SIDES, boards)},
            "boards_apart_mm": _mm(apart),
            "cup_to_board_mm": {s: _mm(d) for s, d in zip(SIDES, cup_now)},
            "min_cup_to_out_board_mm": _mm(self.min_cup_to_out_board),
            "cup_tilt_deg": round(cup_tilt_deg(cup.quat_xyzw), 2),
            "cup_low_mm": _mm(float(cup.low[2]) - L.TOP),
            "min_ungrounded_to_board_mm": {s: _mm(v) for s, v in self.min_ungrounded_to_board.items()},
            "stud_distance_mm": ({s: _mm(stud_distance(arms[s])) for s in SIDES} if have_arms else None),
            "opening": ({s: round(float(arms[s].opening), 3) for s in SIDES} if have_arms else None),
            "grippers_open": grippers_open(arms) if have_arms else None,
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "boards_resting": {s: is_resting(b) for s, b in zip(SIDES, boards)},
        }
        return Result(score, success, details)
