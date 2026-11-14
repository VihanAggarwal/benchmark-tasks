"""The scorer, as plain functions of the arm pair's and the effector's states and the arms' state. NumPy only, no
simulator import. Spec section 4 as the overrides V4 to V7 change it.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
``slider_arms`` (mechkit AssemblyState: the links ``arm_a`` and ``arm_b``), ``effector`` (a loose mechkit compound)
and both grippers through rigstate.

    WIDTH     W = D + 22 mm, D the distance between the two free sockets' mouth centres (spec 4.1)
    ON        for one of the effector's three stud pairs, each ball matched to a socket (the pairing with the smaller
              summed distance): both balls have e >= 0 (centre at or inboard of the mouth plane) and rho <= 1.5 mm
              (centre to the socket's axis LINE in 3D, V4), and |W - 84| <= 1.5 mm
    CLEAR     per gripper: rigstate.zone_gap >= 30 mm to the arm links (tubes, blocks, struts, stop screw) and to
              the effector (every shape) (V6); unreadable arms or a distance that cannot be computed: not clear
    FLAT      each of the six bottom corners of the hexagonal plate within 1.0 mm of the table top (V5)
    STILL     over the last 20 steps none of these eight points moved more than 1.0 mm: the two free socket mouth
              centres and the six effector ball centres (V7)

    1.0   on, both grippers clear and flat on every one of the last 20 steps, and still over them (success)
    0.5   on and at least one gripper clear
    0.25  on
    0     otherwise (and any step whose states are missing or not finite: the window starts again)

Nothing is latched; the score is the last step's (SCORE_OVER "end").
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SLIDER_ARMS = "slider_arms"
EFFECTOR = "effector"
REST_BAR = "rest_bar"


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one step's states (m; world frame)."""
    mouths: tuple  # (mouth centre, inboard normal) per free socket, arm_a then arm_b
    balls: tuple  # three pairs of ball centres
    corners: np.ndarray  # (6, 3) bottom corners of the plate
    origin: np.ndarray  # the effector's frame origin
    joints: dict  # arm_a, arm_b (rad)
    rotation: np.ndarray | None = None  # the effector's rotation (trace only)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, incomplete or not finite."""
    if not isinstance(states, dict):
        return None
    arms_s, eff = states.get(SLIDER_ARMS), states.get(EFFECTOR)
    if arms_s is None or eff is None:
        return None
    try:
        mouths = tuple(L.socket_frames(arms_s.link_to_world))
        r = np.asarray(eff.rotation(), dtype=float)
        o = np.asarray(eff.origin, dtype=float).ravel()
        balls = tuple((o + r @ L.m(a), o + r @ L.m(b)) for a, b in L.effector_balls_local())
        corners = o + L.m(L.plate_corners_local()) @ r.T
        joints = {j: float(arms_s.joints[j]) for j in L.ARMS_PAIR}
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    flat = [v for mo in mouths for v in mo] + [v for p in balls for v in p] + [corners, o]
    if not all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in flat):
        return None
    if not all(math.isfinite(v) for v in joints.values()):
        return None
    return Reading(mouths, balls, corners, o, joints, r)


# -- the conditions, each its own small function ---------------------------------------------------------------------
def width_mm(mouths) -> float:
    """W over the tubes' outer faces at the free end (mm)."""
    return float(np.linalg.norm(np.asarray(mouths[0][0]) - np.asarray(mouths[1][0]))) / L.MM + L.WIDTH_EXTRA


def pair_fit(mouths, pair) -> tuple[float, float, float, float]:
    """For one stud pair: (e_1, rho_1, e_2, rho_2) in mm, each ball matched to a socket, the pairing with the smaller
    summed distance."""
    b1, b2 = pair
    (ma, na), (mb, nb) = mouths
    straight = np.linalg.norm(b1 - ma) + np.linalg.norm(b2 - mb)
    crossed = np.linalg.norm(b1 - mb) + np.linalg.norm(b2 - ma)
    if crossed < straight:
        b1, b2 = b2, b1
    e1, r1 = L.ball_in_socket(b1, ma, na)
    e2, r2 = L.ball_in_socket(b2, mb, nb)
    return e1, r1, e2, r2


def pair_seated(fit) -> bool:
    e1, r1, e2, r2 = fit
    return e1 >= L.E_MIN - L.EPS and e2 >= L.E_MIN - L.EPS and r1 <= L.RHO_TOL + L.EPS and r2 <= L.RHO_TOL + L.EPS


def width_on(w: float) -> bool:
    return abs(w - L.WIDTH_ON) <= L.WIDTH_TOL + L.EPS


def on_pair(r: Reading) -> int | None:
    """The index (0 front, 1 and 2 the sides) of the stud pair the pair is on, or None."""
    if not width_on(width_mm(r.mouths)):
        return None
    for i, pair in enumerate(r.balls):
        if pair_seated(pair_fit(r.mouths, pair)):
            return i
    return None


def arm_pair_distance(state):
    """The arm links' distance function (tubes, blocks, struts, stop screw). The slider, its keeper pins, the rest
    bar and its pins are not parts for the clear rule."""
    def distance(points):
        return state.distance(points, bodies=list(L.ARMS_PAIR))
    return distance


def effector_distance(state):
    return state.distance


def gripper_gaps(arms, states) -> list[tuple[float, float]] | None:
    """Per gripper (left, right): (zone gap to the arm pair, zone gap to the effector) in m; None when unreadable."""
    read_arms = rigstate.read_arms(arms)
    if read_arms is None:
        return None
    try:
        d_arms = arm_pair_distance(states[SLIDER_ARMS])
        d_eff = effector_distance(states[EFFECTOR])
    except (KeyError, AttributeError, TypeError):
        return None
    return [(rigstate.zone_gap(a, d_arms), rigstate.zone_gap(a, d_eff)) for a in read_arms]


def is_clear(gaps: tuple[float, float]) -> bool:
    return all(math.isfinite(g) and g >= L.CLEAR - L.EPS for g in gaps)


def n_clear(gaps) -> int:
    """How many grippers are clear (0 when the arms cannot be read)."""
    return 0 if gaps is None else sum(1 for g in gaps if is_clear(g))


def is_flat(corners, top: float = L.TOP) -> bool:
    """Every bottom corner of the plate within 1.0 mm of the table top (V5)."""
    z = (np.asarray(corners, dtype=float)[:, 2] - top) / L.MM
    return bool(np.all(np.abs(z) <= L.FLAT_TOL + L.EPS))


def points_of(r: Reading) -> np.ndarray:
    """The eight points of the stillness rule: two mouth centres, six ball centres (V7)."""
    return np.array([r.mouths[0][0], r.mouths[1][0]] + [b for p in r.balls for b in p], dtype=float)


def max_move_mm(history) -> float | None:
    """Largest displacement (mm) of any of the eight points from where it was at the window's start."""
    if not history:
        return None
    first = points_of(history[0])
    return max(float(np.max(np.linalg.norm(points_of(x) - first, axis=1))) for x in history) / L.MM


def tier(on: bool, clear_count: int, full_window: bool) -> float:
    """The highest line that holds. ``full_window``: on, both clear and flat on each of the last 20 steps and still
    over them."""
    if not on:
        return 0.0
    if full_window and clear_count >= 2:
        return 1.0
    if clear_count >= 1:
        return 0.5
    return 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) / L.MM, 2)


def _table_mm(p) -> list[float]:
    """A world point as table-frame millimetres (z above the table top), for the trace."""
    q = np.asarray(p, dtype=float) - np.array([0.0, 0.0, L.TOP])
    return [round(float(v) / L.MM, 2) for v in q]


def calibration_details(r: Reading) -> dict:
    """Debug output only (no rule reads it): what the calibration jobs C1 to C7 read from the trace (spec 5.4)."""
    mouths = [_table_mm(mo) for mo, _ in r.mouths]
    yaw = None
    if r.rotation is not None:
        rot = np.asarray(r.rotation, dtype=float)
        yaw = round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 2)
    return {"mouths_mm": mouths, "centre_line_x_mm": round((mouths[0][0] + mouths[1][0]) / 2, 2),
            "effector_yaw_deg": yaw,
            "front_balls_mm": [_table_mm(b) for b in r.balls[0]]}


class Tracker:
    """The last 1 s of readings and of the per-step 1.0 conditions."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.full: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        start = np.array([*L.EFFECTOR_ORIGIN, 0.0]) + np.array([0.0, 0.0, L.TOP])
        return float(np.linalg.norm(r.origin - start)) <= L.RESET_START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The effector put back on its start pose with a jump no trial makes in one control step: a new episode."""
        return cls.at_start(now) and float(np.linalg.norm(now.origin - prev.origin)) >= L.RESET_JUMP

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.full.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        self.hist.append(r)
        w = width_mm(r.mouths)
        pair = on_pair(r)
        on = pair is not None
        gaps = gripper_gaps(arms, states)
        count = n_clear(gaps)
        flat = is_flat(r.corners)
        self.full.append(bool(on and count == 2 and flat))
        move = max_move_mm(list(self.hist)) if len(self.hist) == self.window + 1 else None
        still = move is not None and move <= L.STILL_MOVE + L.EPS
        full_window = len(self.full) == self.window and all(self.full) and still
        level = tier(on, count, full_window)
        fits = [pair_fit(r.mouths, p) for p in r.balls]
        z = (r.corners[:, 2] - L.TOP) / L.MM
        details = {
            "valid_state": True, "score": level, "width_mm": round(w, 2), "on": on, "on_pair": pair,
            "fit_mm": [[round(v, 2) for v in f] for f in fits],
            "joints_rad": {k: round(v, 5) for k, v in r.joints.items()},
            "zone_gap_mm": None if gaps is None else {s: {"arm_pair": _mm(g[0]), "effector": _mm(g[1])}
                                                       for s, g in zip(rigstate.ARMS, gaps)},
            "clear": None if gaps is None else {s: is_clear(g) for s, g in zip(rigstate.ARMS, gaps)},
            "n_clear": count, "flat": flat, "corner_z_mm": [round(float(v), 2) for v in z],
            "still_move_mm": None if move is None else round(move, 3), "still": still,
            "window_full": full_window, "effector_origin": [round(float(v), 4) for v in r.origin],
            "arms_readable": rigstate.readable(arms), **calibration_details(r),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
