"""The scorer, as plain functions of the tower cooler's state and the arms' state. NumPy only; no simulator needed.

The score is read from where the cooler ends, as a builder checks a mount with a look from above and a feeler from
the side: is its base flat on the processor's lid, inside the retention frame, with its airflow toward the rear
exhaust, and has the robot let go of it. The scorer keeps only the cooler's last half second of positions (to tell
a cooler at rest from one still moving) and is fed one observation per control step by ``task.success()``:

    Tracker.update(cooler, arms) -> Result(score, success, details)

``cooler`` is the cooler's state as mechkit reports it (``AssemblyState``): ``origin`` (its base frame's origin: the
centre of the base's bottom face), ``quat_xyzw`` (that frame's orientation: local +x is the airflow, from the fan
through the fins, local +z up), and the ``ObjectState`` fields ``pos``, ``lin_vel``, ``ang_vel``, ``low``, ``high``
(world-aligned box round all its shapes). A state with a non-finite number, a zero quaternion or no ``origin`` (a
simulator blow-up, or the cooler not built as an assembly) scores 0 for that step and leaves the tracker as it was.

``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the
finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; 'let go' is then
taken as met for that step and the step is counted in ``arms_unavailable``. rigstate gives no gripper heading, so
a gripper's fingers are taken to be anywhere round its axis: its 'finger zone' is every point within a radius of
the 38 mm segment from the point between its finger tips back up the gripper; the radius reaches a pad's outer
corner (half the current finger gap plus the 8 mm pad thickness, and half the pad's 27 mm width across).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

CPU_TOP_W = L.TOP + L.CPU_TOP  # the processor lid's top (world z)
_KEYS = ("origin", "quat_xyzw", "pos", "lin_vel", "ang_vel", "low", "high")
_FOOT = np.array([(sx * L.BASE_HALF, sy * L.BASE_HALF, 0.0) for sx in (-1, 1) for sy in (-1, 1)])


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


def valid(c) -> bool:
    """Every number finite, the quaternion not degenerate, and the base frame's origin present."""
    try:
        parts = [np.asarray(getattr(c, k), dtype=float).ravel() for k in _KEYS]
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    sizes_ok = all(parts[i].size == 3 for i in (0, 2, 3, 4, 5, 6)) and parts[1].size == 4
    return bool(sizes_ok and np.linalg.norm(parts[1]) > 0.5)


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


# -- the cooler's pose ---------------------------------------------------------------------------------------------
def to_world(c, local) -> np.ndarray:
    """A point given in the cooler's base frame, in the world."""
    return np.asarray(c.origin, dtype=float) + rotation(c.quat_xyzw) @ np.asarray(local, dtype=float)


def foot_corners(c) -> np.ndarray:
    """The four bottom corners of the copper base in the world (4 x 3)."""
    return np.asarray(c.origin, dtype=float) + _FOOT @ rotation(c.quat_xyzw).T


def tilt_deg(c) -> float:
    """Angle between the vertical and the cooler's up axis (0..180 degrees): 0 standing on its base, 90 on its side,
    180 upside down."""
    return float(np.degrees(np.arccos(np.clip(rotation(c.quat_xyzw)[2, 2], -1.0, 1.0))))


def airflow_err_deg(c) -> float:
    """Angle, seen from above, between the cooler's airflow (fan to fins) and +x, toward the rear exhaust (0..180
    degrees; 180 when the airflow points straight up or down)."""
    a = rotation(c.quat_xyzw)[:, 0]
    h = math.hypot(a[0], a[1])
    if h < 1e-6:
        return 180.0
    return float(np.degrees(math.acos(max(-1.0, min(1.0, a[0] / h)))))


def base_offset(c) -> tuple[float, float]:
    """The base's bottom-face centre minus the processor's centre, along x and y (m)."""
    o = np.asarray(c.origin, dtype=float)
    return float(o[0] - L.CPU_X), float(o[1] - L.CPU_Y)


def base_off(c) -> float:
    """Horizontal distance from the base's bottom-face centre to the processor's centre (m)."""
    return float(math.hypot(*base_offset(c)))


def foot_gap(c) -> float:
    """Height of the base's bottom-face centre above the processor's lid (m; negative = below it)."""
    return float(c.origin[2] - CPU_TOP_W)


def frame_margin(c) -> float:
    """How far the base's four bottom corners are inside the retention frame's 44 x 44 mm opening, seen from above:
    the smallest margin to the opening's edges (m; negative when a corner is over or past the frame)."""
    k = foot_corners(c)
    m = np.minimum(L.FRAME_OPEN_HALF - np.abs(k[:, 0] - L.CPU_X), L.FRAME_OPEN_HALF - np.abs(k[:, 1] - L.CPU_Y))
    return float(m.min())


def right_way_round(c) -> bool:
    """The airflow (fan to fins) within 15 degrees of +x seen from above: the one way round the fin stack clears the
    memory."""
    return bool(airflow_err_deg(c) <= L.AIRFLOW_TOL_DEG)


def seated(c) -> bool:
    """Seated: the base's bottom-face centre within 2 mm of the processor's lid, tilted 2 degrees or less, its four
    bottom corners inside the frame's opening, and the airflow within 15 degrees of +x."""
    return bool(abs(foot_gap(c)) <= L.SEAT_Z_TOL and tilt_deg(c) <= L.SEAT_TILT_DEG and frame_margin(c) >= 0.0
                and right_way_round(c))


def over_board(c) -> bool:
    """The base's bottom-face centre is over the board's 160 x 170 mm outline, seen from above."""
    o = np.asarray(c.origin, dtype=float)
    return bool(L.BOARD_X0 <= o[0] <= L.BOARD_X1 and L.BOARD_Y0 <= o[1] <= L.BOARD_Y1)


# -- the arms ------------------------------------------------------------------------------------------------------
def _lo_hi(s) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


def finger_gap(arm) -> float:
    return float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP


def zone_radius(arm) -> float:
    """From the gripper's axis to a finger pad's outer corner: half the gap plus the pad's thickness, and half its
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


def zone_gap(arm, lo, hi) -> float:
    """How far a gripper's finger zone is from a world-aligned box (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_box_distance(p, lo, hi) for p in pts) - r


def let_go(arm, lo, hi) -> bool:
    """The gripper is not holding the cooler in the box (lo, hi): it is open (opening 0.9 or more, fingers 84.5 mm
    apart), or its finger zone is more than 10 mm from the cooler. A gripper closed somewhere else holds nothing."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, lo, hi) > L.LET_GO_GAP)


def released(arms, c) -> bool:
    """Both grippers have let go of the cooler."""
    lo, hi = _lo_hi(c)
    return bool(let_go(arms["left"], lo, hi) and let_go(arms["right"], lo, hi))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 2) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer remembers within a trial: the cooler's base centre and fin-stack top over the last half
    second, and how many steps had no readable arms or no valid cooler state."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history: deque = deque(maxlen=self.window + 1)  # (base centre, stack top) for this step and the 10 before
        self.prev_origin = None
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite cooler state (ignored)
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(c) -> bool:
        """The starting layout: the cooler standing still and upright on the foam pad, on its mark."""
        o = np.asarray(c.origin, dtype=float)
        on_mark = (math.hypot(o[0] - L.START[0], o[1] - L.START[1]) < L.START_TOL
                   and abs(o[2] - L.START[2]) < 0.003)
        try:
            slow = float(np.linalg.norm(np.asarray(c.lin_vel, dtype=float))) < 0.01
        except (AttributeError, TypeError, ValueError):
            slow = True
        return bool(on_mark and tilt_deg(c) < 2.0 and slow)

    def moved(self) -> float | None:
        """The farthest the base centre or the fin stack's top has been, over the last 0.5 s, from where it is now
        (m); None until 0.5 s of this trial has been seen."""
        if len(self.history) < self.window + 1:
            return None
        base, top = self.history[-1]
        return float(max(max(np.linalg.norm(b - base), np.linalg.norm(t - top)) for b, t in self.history))

    def update(self, c, arms=None) -> Result:
        if not valid(c):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(c, arms)
            return self.last
        o = np.asarray(c.origin, dtype=float)
        if self.prev_origin is not None and self.at_start(c):
            if float(np.linalg.norm(o - self.prev_origin)) > L.RESET_JUMP:  # the simulator put it back: new episode
                self.reset()  # (rigstate's reset hook normally does this first; this is the fallback)
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.history.append((o.copy(), to_world(c, L.TOP_LOCAL)))
        self.prev_origin = o
        self.last = self.result(c, arms)
        return self.last

    def result(self, c, arms=None) -> Result:
        """The score of the state at hand, with the history fed so far (it changes nothing)."""
        if not valid(c):
            return Result(0.0, False, {"valid_state": False, "zero_reason": "invalid_state",
                                       "bad_steps": self.bad_steps, "arms_unavailable": self.arms_unavailable})
        have_arms = arms_valid(arms)
        free = released(arms, c) if have_arms else True  # unreadable arms: taken as let go, and counted
        moved = self.moved()
        at_rest = moved is not None and moved <= L.STILL_TOL
        sit, tilt, off = seated(c), tilt_deg(c), base_off(c)
        right_way = right_way_round(c)
        reason = ""
        if not at_rest:
            score, reason = 0.0, "still_moving" if moved is not None else "not_yet_seen_0.5_s"
        elif sit:
            score = 1.0 if free else 0.5
        elif not free:
            score, reason = 0.0, "still_held"
        elif tilt > L.UPRIGHT_DEG:
            score, reason = 0.0, "tipped_over"
        elif off <= L.NEAR and right_way:  # the right way round, on or by the processor, not quite seated
            score = 0.5
        elif over_board(c):  # turned another way round (e.g. on the memory), or elsewhere on the board
            score = 0.25
        else:
            score, reason = 0.0, "not_on_the_board"
        dx, dy = base_offset(c)
        lo, hi = _lo_hi(c)
        details = {
            "valid_state": True,
            "zero_reason": reason,
            "seated": sit,
            "right_way": right_way,
            "let_go": free,
            "at_rest": at_rest,
            "moved_mm": _mm(moved),
            "foot_gap_mm": _mm(foot_gap(c)),
            "tilt_deg": round(tilt, 2),
            "airflow_err_deg": round(airflow_err_deg(c), 2),
            "frame_margin_mm": _mm(frame_margin(c)),
            "base_off_mm": _mm(off),
            "base_dx_mm": _mm(dx),
            "base_dy_mm": _mm(dy),
            "over_board": over_board(c),
            "lowest_mm": _mm(float(lo[2]) - L.TOP),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "left_zone_mm": _mm(zone_gap(arms["left"], lo, hi)) if have_arms else None,
            "right_zone_mm": _mm(zone_gap(arms["right"], lo, hi)) if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0), details)
