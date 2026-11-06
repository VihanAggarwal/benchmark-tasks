"""The scorer, as plain functions of the four boards' states and the arms' state. NumPy only; no simulator needed.

A drift fence feeds a pitfall trap only if each fence arm runs straight at the trap and its end meets the rim:
a gap lets animals walk round the trap, and a fence end over the mouth blocks it. Four orange pegs mark the far
ends of four arms (lines 1 to 4, at bearings 45, 135, 225 and 315 deg about the trap's centre). Every control step
the scorer reads the boards' poses and, through rigstate.py, the arms, and decides which lines are fenced:

    Tracker.update(boards, arms) -> Result(score, success, details)

``boards`` maps "board_1" .. "board_4" to anything with ``pos`` (the box's centre) and ``quat_xyzw``, as
``yam_arena.task_file.ObjectState`` has. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with a
``tcp`` (env-frame metres), or ``None`` when the arms could not be read; then every board counts as free on that
step and the step is counted in ``arms_unavailable``. A board state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The tracker keeps no rule about the whole trial: only each board's recent centre positions (for 'at rest', which
is never read from velocity, since a gripped part's reported velocity chatters at 10-45 mm/s while it stays put)
and a count of steps without the arms. A new episode (rigstate's reset hook, or every board put back on its own
starting mark in one step) clears it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_SIGNS = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
TRAP_C = np.array([L.TRAP_X, L.TRAP_Y])


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
    """A finite centre and a non-degenerate quaternion (a simulator blow-up gives NaN or zeros)."""
    try:
        pos = np.asarray(s.pos, dtype=float).ravel()
        q = np.asarray(s.quat_xyzw, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(pos.size == 3 and q.size == 4 and np.all(np.isfinite(pos)) and np.all(np.isfinite(q))
                and np.linalg.norm(q) > 0.5)


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ("left", "right"):
            tcp = np.asarray(arms[side].tcp, dtype=float).ravel()
            if tcp.size != 3 or not np.all(np.isfinite(tcp)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- one board's geometry ---------------------------------------------------------------------------------------
def corners(s) -> np.ndarray:
    """The board's eight corners in the world (8 x 3), from its centre and orientation."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * np.array(L.BOARD_HALF)) @ rotation(s.quat_xyzw).T


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def footprint(s) -> np.ndarray:
    """The board's plan outline, (n, 2): the convex hull of its eight corners seen from above, standing or not."""
    return _hull(corners(s)[:, :2])


def plan_distance(p, poly: np.ndarray) -> float:
    """Distance in plan from point ``p`` (x, y) to a convex counter-clockwise polygon (0 on or inside it)."""
    p = np.asarray(p, dtype=float)[:2]
    n = len(poly)
    if n == 1:
        return float(np.linalg.norm(p - poly[0]))
    inside = n >= 3
    best = math.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        if (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) < 0:
            inside = False
        ee = float(e @ e)
        t = 0.0 if ee <= 0 else min(max(float((p - a) @ e) / ee, 0.0), 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * e))))
    return 0.0 if inside else best


def tilt_deg(s) -> float:
    """Angle between the vertical and the board's 50 mm axis (0..90 deg; either long edge may be down)."""
    zz = rotation(s.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def bottom(s) -> float:
    """Height of the board's lowest point above the table top (m)."""
    return float(corners(s)[:, 2].min() - L.TOP)


def standing_pose(s) -> bool:
    """On its long narrow edge: 50 mm axis within 10 deg of vertical, lowest point within 3 mm of the table top."""
    return bool(tilt_deg(s) <= L.STAND_TILT_DEG and abs(bottom(s)) <= L.STAND_LOW_TOL)


def gap(s) -> float:
    """Plan distance from the trap's centre to the nearest point of the board's footprint, minus the 35 mm rim
    radius (m). Negative: the outline is inside the rim (over the opening); -35 mm if it covers the centre."""
    return plan_distance(TRAP_C, footprint(s)) - L.TRAP_RADIUS


def bearing_deg(s) -> float:
    """Bearing of the board's centre from the trap's centre, anticlockwise from +x (0..360 deg)."""
    return float(math.degrees(math.atan2(s.pos[1] - L.TRAP_Y, s.pos[0] - L.TRAP_X)) % 360.0)


def sector(s) -> int:
    """0..3 for lines 1..4: the peg whose bearing (45, 135, 225, 315 deg) is nearest the bearing of the board's
    centre. The sectors are 90 deg wide; a centre exactly on a boundary goes to the sector anticlockwise of it."""
    return int(bearing_deg(s) // 90.0) % 4


def long_axis(s) -> np.ndarray | None:
    """The board's 130 mm axis seen from above, as a unit (x, y) vector (None if it points straight up)."""
    a = rotation(s.quat_xyzw)[:2, 0]
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-6 else None


def axis_err_deg(s, k: int) -> float:
    """How far the board's long axis is turned, in plan, from the direction from the trap's centre to peg k
    (0..90 deg; either end may face the trap)."""
    a = long_axis(s)
    if a is None:
        return 90.0
    d = math.degrees(math.atan2(a[1], a[0])) - L.PEG_BEARINGS_DEG[k]
    return float(abs((d + 90.0) % 180.0 - 90.0))


def centreline_offset(s) -> float:
    """Plan distance from the trap's centre to the board's long centreline (the line through its centre along its
    long axis), m."""
    a = long_axis(s)
    if a is None:
        return math.inf
    d = TRAP_C - np.asarray(s.pos, dtype=float)[:2]
    return float(abs(a[0] * d[1] - a[1] * d[0]))


def tcp_clearance(arms, s) -> float:
    """Nearest plan distance from a gripper TCP that is less than 90 mm above the table to the board's footprint
    (m; inf when neither TCP is that low). 'Free' needs more than 60 mm: with the fingers fully open (94 mm) a
    pad's outer corner is 56 mm from the TCP in plan, so no pad can touch a board whose footprint is 60 mm away."""
    poly = footprint(s)
    best = math.inf
    for side in ("left", "right"):
        tcp = np.asarray(arms[side].tcp, dtype=float)
        if tcp[2] - L.TOP < L.FREE_HEIGHT:
            best = min(best, plan_distance(tcp[:2], poly))
    return best


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The boards' recent positions in this trial (for 'at rest'), and the steps without the arms."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.history = {n: deque(maxlen=L.REST_STEPS + 1) for n in L.BOARDS}  # the current step and the 10 before
        self.prev: dict | None = None  # board centres at the previous valid step
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(boards) -> bool:
        """Every board on its own starting mark, standing, long side along x (the starting layout)."""
        for n in L.BOARDS:
            s = boards[n]
            x0, y0 = L.BOARD_STARTS[n]
            if math.hypot(s.pos[0] - x0, s.pos[1] - y0) > L.START_TOL or not standing_pose(s):
                return False
            a = long_axis(s)
            if a is None or abs(a[1]) > math.sin(math.radians(5.0)):
                return False
        return True

    def path(self, name: str, s) -> float | None:
        """The largest distance of the board's recorded centres over the last 0.5 s from its current centre (m);
        None until 0.5 s has been recorded."""
        h = self.history[name]
        if len(h) < L.REST_STEPS + 1:
            return None
        c = np.asarray(s.pos, dtype=float)
        return float(max(np.linalg.norm(p - c) for p in h))

    def update(self, boards, arms=None) -> Result:
        if not all(n in boards and valid(boards[n]) for n in L.BOARDS):  # a blow-up: 0 now, keep everything
            self.steps += 1
            self.last = self.result(boards, arms)
            return self.last
        now = {n: np.asarray(boards[n].pos, dtype=float).copy() for n in L.BOARDS}
        if self.prev is not None and self.at_start(boards):
            if max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.BOARDS) > L.RESET_JUMP:
                self.reset()  # the simulator put the boards back on their marks: a new episode
        self.steps += 1
        for n in L.BOARDS:
            self.history[n].append(now[n])
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.prev = now
        self.last = self.result(boards, arms)
        return self.last

    def board_report(self, name: str, s, arms) -> dict:
        """Everything the scorer decides about one board."""
        k = sector(s)
        path = self.path(name, s)
        at_rest = path is not None and path <= L.REST_TOL
        have_arms = arms_valid(arms)
        clear = tcp_clearance(arms, s) if have_arms else math.inf
        free = (clear > L.FREE_RADIUS) if have_arms else True
        stands = standing_pose(s)
        g, ax, off = gap(s), axis_err_deg(s, k), centreline_offset(s)
        fenced = bool(stands and at_rest and free and ax <= L.AXIS_TOL_DEG and off <= L.CENTRELINE_TOL
                      and L.GAP_MIN <= g <= L.GAP_MAX)
        return {
            "line": k + 1,
            "fenced": fenced,
            "standing": bool(stands and at_rest),
            "tilt_deg": round(tilt_deg(s), 2),
            "bottom_mm": round(bottom(s) * 1000, 2),
            "at_rest": at_rest,
            "path_mm": None if path is None else round(path * 1000, 2),
            "free": free,
            "tcp_clear_mm": None if not math.isfinite(clear) else round(clear * 1000, 1),
            "bearing_deg": round(bearing_deg(s), 2),
            "axis_err_deg": round(ax, 2),
            "offset_mm": None if not math.isfinite(off) else round(off * 1000, 2),
            "gap_mm": round(g * 1000, 2),
            "over_rim": bool(g < L.ZERO_GAP),
        }

    def result(self, boards, arms=None) -> Result:
        if not all(n in boards and valid(boards[n]) for n in L.BOARDS):
            return Result(0.0, False, {"valid_state": False, "arms_unavailable": self.arms_unavailable})
        reports = {n: self.board_report(n, boards[n], arms) for n in L.BOARDS}
        lines = [any(r["fenced"] and r["line"] == k + 1 for r in reports.values()) for k in range(4)]
        zero = any(r["over_rim"] for r in reports.values())
        score = 0.0 if zero else L.CREDIT_PER_LINE * sum(lines)
        details = {
            "valid_state": True,
            "lines_fenced": [k + 1 for k in range(4) if lines[k]],
            "zero_rule": zero,
            "arms_available": arms_valid(arms),
            "arms_unavailable": self.arms_unavailable,
            "boards": reports,
        }
        return Result(score, bool(score == 1.0), details)
