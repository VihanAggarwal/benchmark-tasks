"""The scorer, as plain functions of the head's and the two head blocks' states and the arms' state. NumPy only; no
simulator needed.

The score is the end state (where each head block stands against the head when the trial ends) plus one fact about
the whole trial: whether the head was ever pushed off its place, lifted or tipped. So the scorer is fed one
observation per control step by ``task.success()``; it remembers where the head started and how far it has ever
been from there, and keeps counters for the trace.

    Tracker.update(blocks, head, arms) -> Result(score, success, details)

``blocks`` maps each block's name (``layout.BLOCKS``) and ``head`` is the head; each is anything with ``pos``
(centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as ``yam_arena.task_file.ObjectState`` has.
A block is a box of its stated size and the head an upright cylinder, so their shapes follow from ``pos`` and
``quat_xyzw``. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point
between the finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then
every block is taken as let go for that step and the step is counted in ``arms_unavailable``. A state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Geometry. Seen from above, the head's outline is its axis (the segment between the centres of its two flat ends,
projected onto the table) widened by its 60 mm radius: for an upright head that is exactly its 120 mm circle, and
for a head tilted up to 10 degrees it is at most 0.9 mm wider than the true outline. A block's outline is the
convex hull of its eight corners seen from above. The 'gap' between a block and the head is the horizontal distance
between the two outlines (the head's axis to the block's outline, minus the radius): 0 or less where they touch,
and negative by how far the block reaches inside the head's outline. A block is on the side of the head its centre
is on (y greater or smaller than the head's centre).

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger
zone' is every point within a radius of the 38 mm segment from the point between its finger tips back up the
gripper; the radius is the distance from that axis to a pad's outer corner (half the current finger gap plus the
8 mm pad thickness, and half the pad's 27 mm width across; 57 mm fully open).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
NAMES = tuple(L.BLOCKS)
HEAD = "head"
SIDES = {1: "left", -1: "right"}
EPS = 1e-9  # thresholds are inclusive ("5 mm or less"); this keeps a value exactly on one from failing by rounding


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


def states_valid(blocks, head) -> bool:
    try:
        return valid(head) and all(valid(blocks[n]) for n in NAMES)
    except (KeyError, TypeError):
        return False


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


# -- plane geometry -----------------------------------------------------------------------------------------------
def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points (monotone chain), counter-clockwise."""
    p = sorted({(round(float(a), 12), round(float(b), 12)) for a, b in pts})
    if len(p) <= 2:
        return np.array(p, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _point_segment(p, a, b) -> float:
    """Distance from point p to segment a-b (2-D)."""
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    d = b - a
    n = float(d @ d)
    t = 0.0 if n < 1e-18 else min(max(float((p - a) @ d) / n, 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * d)))


def _segments_cross(a, b, c, d) -> bool:
    """Do segments a-b and c-d (2-D) properly cross?"""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return bool(o1 * o2 < 0 and o3 * o4 < 0)


def _inside(p, poly: np.ndarray) -> bool:
    """Is point p inside (or on) a counter-clockwise convex polygon?"""
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            return False
    return True


def segment_polygon_distance(a, b, poly: np.ndarray) -> float:
    """Distance between segment a-b and a convex polygon, seen from above (0 if they meet)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if _inside(a, poly) or _inside(b, poly):
        return 0.0
    n = len(poly)
    best = math.inf
    for i in range(n):
        c, d = poly[i], poly[(i + 1) % n]
        if _segments_cross(a, b, c, d):
            return 0.0
        best = min(best, _point_segment(a, c, d), _point_segment(b, c, d), _point_segment(c, a, b),
                   _point_segment(d, a, b))
    return float(best)


# -- one block ----------------------------------------------------------------------------------------------------
def corners(s) -> np.ndarray:
    """The block's eight corners (world, 8 x 3)."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * np.array(L.BLOCK_HALF)) @ rotation(s.quat_xyzw).T


def footprint(s) -> np.ndarray:
    """The block's outline seen from above: the convex hull of its corners' (x, y)."""
    return _hull(corners(s)[:, :2])


def tilt_deg(quat_xyzw) -> float:
    """How far a body's own z axis (a block's 80 mm axis, the head's axis) leans from vertical (degrees, 0..90;
    either way up)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def heading_deg(quat_xyzw) -> float:
    """The angle between the block's 70 mm side (its x axis) and the board's long axis (x), seen from above
    (degrees, 0..90; either end forward)."""
    u = rotation(quat_xyzw)[:, 0]
    return float(np.degrees(math.atan2(abs(u[1]), abs(u[0])))) if abs(u[0]) + abs(u[1]) > 1e-9 else 90.0


def rise(s) -> float:
    """Height of the block's lowest point above the board's top (m)."""
    return float(corners(s)[:, 2].min() - L.BOARD_TOP)


def point_box_distance(p, s) -> float:
    """Distance from a point to the block (0 inside it), in the block's own frame."""
    r = rotation(s.quat_xyzw)
    d = r.T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))
    return float(np.linalg.norm(np.maximum(np.abs(d) - np.array(L.BLOCK_HALF), 0.0)))


# -- the head -----------------------------------------------------------------------------------------------------
def head_axis(h) -> tuple[np.ndarray, np.ndarray]:
    """The centres of the head's two flat ends (world)."""
    c = np.asarray(h.pos, dtype=float)
    u = rotation(h.quat_xyzw)[:, 2] * (L.HEAD_H / 2)
    return c - u, c + u


def head_low(h) -> float:
    """Height of the head's lowest point above the board's top (m): its lower rim."""
    a, b = head_axis(h)
    zz = abs(float(rotation(h.quat_xyzw)[2, 2]))
    return float(min(a[2], b[2]) - L.HEAD_R * math.sqrt(max(0.0, 1.0 - zz * zz)) - L.BOARD_TOP)


def gap(s, h) -> float:
    """Horizontal gap between a block and the head, seen from above (m; 0 or less where they touch, negative by
    how far the block reaches inside the head's outline)."""
    a, b = head_axis(h)
    return segment_polygon_distance(a[:2], b[:2], footprint(s)) - L.HEAD_R


def side_of(s, h) -> int:
    """+1 if the block's centre is on the head's left (greater y), -1 on its right."""
    return 1 if float(s.pos[1]) >= float(h.pos[1]) else -1


# -- the arms -----------------------------------------------------------------------------------------------------
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


def zone_gap(arm, s) -> float:
    """How far a gripper's finger zone is from a block (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_box_distance(p, s) for p in pts) - r


def let_go(arm, s) -> bool:
    """A gripper has let go of a block when it is at least 90% open (fingers 84.5 mm apart) or its finger zone is
    more than 10 mm from the block."""
    return bool(float(arm.opening) >= L.GRIPPER_OPEN or zone_gap(arm, s) > L.LET_GO_GAP)


def released(arms, s) -> bool:
    """Neither gripper still holds the block (unreadable arms: taken as let go)."""
    if not arms_valid(arms):
        return True
    return bool(let_go(arms["left"], s) and let_go(arms["right"], s))


# -- scoring ------------------------------------------------------------------------------------------------------
def block_report(s, h, arms=None) -> dict:
    """Everything the score needs about one block, and its credit for its side: 0.5 if it stands beside the head
    with a gap of 5 mm or less, 0.25 if 15 mm or less, 0 otherwise."""
    g = gap(s, h)
    tilt, heading, up = tilt_deg(s.quat_xyzw), heading_deg(s.quat_xyzw), rise(s)
    dx = float(s.pos[0]) - float(h.pos[0])
    free = released(arms, s)
    beside = bool(free and tilt <= L.UPRIGHT_MAX_DEG + EPS and heading <= L.HEADING_MAX_DEG + EPS
                  and abs(up) <= L.ON_BOARD + EPS and abs(dx) <= L.CENTRED_X + EPS)
    snug = bool(beside and g <= L.SNUG_GAP + EPS)
    close = bool(beside and g <= L.CLOSE_GAP + EPS)
    credit = L.SNUG_CREDIT if snug else L.CLOSE_CREDIT if close else 0.0
    return {"side": side_of(s, h), "credit": credit, "snug": snug, "beside": beside, "let_go": free, "gap": g,
            "tilt_deg": tilt, "heading_deg": heading, "rise": up, "dx": dx, "over_head": bool(g < -L.OVER_HEAD)}


def head_report(h, start_xy=None) -> dict:
    """Where the head is now: how far its centre is from its start seen from above, how high its lowest point is
    above the board, how far its axis leans."""
    sx, sy = (L.HEAD_X, L.HEAD_Y) if start_xy is None else start_xy
    return {"shift": float(math.hypot(float(h.pos[0]) - sx, float(h.pos[1]) - sy)), "lift": head_low(h),
            "tilt_deg": tilt_deg(h.quat_xyzw)}


def disturbed(shift: float, lift: float, tilt: float) -> bool:
    """The head was pushed off its place (more than 8 mm), lifted (more than 10 mm) or tipped (over 10 degrees)."""
    return bool(shift > L.HEAD_SHIFT_MAX + EPS or lift > L.HEAD_LIFT_MAX + EPS or tilt > L.HEAD_TILT_MAX_DEG + EPS)


def evaluate(blocks, head, arms=None, worst=(0.0, 0.0, 0.0)) -> tuple[float, dict]:
    """The score of a state, given the worst (shift, lift, tilt) the head has had so far in the trial, and the
    reports. Score 0 if the head was ever disturbed or any block reaches more than 5 mm inside its outline; else
    the sum, over the head's two sides, of the best block's credit there."""
    reps = {n: block_report(blocks[n], head, arms) for n in NAMES}
    over = sorted(n for n, r in reps.items() if r["over_head"])
    hurt = disturbed(*worst)
    sides = {side: max([r["credit"] for r in reps.values() if r["side"] == side] or [0.0]) for side in (1, -1)}
    score = 0.0 if (hurt or over) else float(sides[1] + sides[-1])
    return score, {"reports": reps, "over_head": over, "head_disturbed": hurt, "left_side": sides[1],
                   "right_side": sides[-1]}


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


def _brief(rep: dict) -> dict:
    """One block's report, rounded for the trace."""
    return {
        "side": SIDES[rep["side"]], "credit": rep["credit"], "snug": rep["snug"], "beside": rep["beside"],
        "let_go": rep["let_go"], "gap_mm": _mm(rep["gap"]), "tilt_deg": round(rep["tilt_deg"], 2),
        "heading_deg": round(rep["heading_deg"], 2), "rise_mm": _mm(rep["rise"]), "dx_mm": _mm(rep["dx"]),
    }


def _objects(blocks, head) -> list:
    """(name, state) of the two blocks and the head."""
    return [*((n, blocks[n]) for n in NAMES), (HEAD, head)]


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Where the head started and the worst it has had since; counters for the trace."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a non-finite state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.head_start: tuple[float, float] | None = None  # the head's centre (x, y) at the trial's first step
        self.worst_shift = 0.0  # m: the farthest the head's centre has been from its start, seen from above
        self.worst_lift = 0.0  # m: the highest its lowest point has been above the board's top
        self.worst_tilt = 0.0  # degrees: the most its axis has leaned
        self.first_disturbed_step: int | None = None  # the first step the head was disturbed (diagnostic only)
        self.first_full_step: int | None = None  # the first step the 1.0 conditions held (diagnostic only)
        self.best = 0.0  # the best score seen (diagnostic only: the score is read at the last step)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # {name: (centre, velocity)} at the previous valid control step

    @staticmethod
    def at_start(blocks, head) -> bool:
        """The starting layout: each block standing on its mark on the table and the head on its mark on the board,
        upright and still."""
        for n in NAMES:
            s = blocks[n]
            x, y = L.BLOCKS[n]
            near = math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) < L.START_TOL
            on_table = abs(rise(s) + L.BOARD_T) < 0.003
            if not (near and on_table and tilt_deg(s.quat_xyzw) < 1.0 and is_resting(s)):
                return False
        near = math.hypot(float(head.pos[0]) - L.HEAD_X, float(head.pos[1]) - L.HEAD_Y) < L.START_TOL
        return bool(near and abs(head_low(head)) < 0.003 and tilt_deg(head.quat_xyzw) < 1.0 and is_resting(head))

    def _teleported(self, blocks, head) -> bool:
        """An object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * L.GRAVITY * self.dt ** 2
        for n, s in _objects(blocks, head):
            p0, v0 = self.prev[n]
            c = np.asarray(s.pos, dtype=float)
            if float(np.linalg.norm(c - p0)) > L.RESET_JUMP + float(np.linalg.norm(v0)) * self.dt + fall:
                return True
        return False

    def update(self, blocks, head, arms=None) -> Result:
        if not states_valid(blocks, head):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(blocks, head, arms)
            return self.last
        if self.prev is not None and self._teleported(blocks, head) and self.at_start(blocks, head):
            self.reset()  # the simulator put everything back: a new episode (rigstate's on_reset hook normally does
            #               this first; this is the fallback)
        self.prev = {n: (np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float))
                     for n, s in _objects(blocks, head)}
        self.steps += 1
        if self.head_start is None:
            self.head_start = (float(head.pos[0]), float(head.pos[1]))
        if not arms_valid(arms):
            self.arms_unavailable += 1
        hr = head_report(head, self.head_start)
        self.worst_shift = max(self.worst_shift, hr["shift"])
        self.worst_lift = max(self.worst_lift, hr["lift"])
        self.worst_tilt = max(self.worst_tilt, hr["tilt_deg"])
        self.last = self.result(blocks, head, arms)
        if self.last.details.get("head_disturbed") and self.first_disturbed_step is None:
            self.first_disturbed_step = self.steps
        if self.last.score == 1.0 and self.first_full_step is None:
            self.first_full_step = self.steps
        self.best = max(self.best, self.last.score)
        self.last.details.update(first_disturbed_step=self.first_disturbed_step,
                                 first_full_step=self.first_full_step, best_score=self.best)
        return self.last

    def result(self, blocks, head, arms=None) -> Result:
        if not states_valid(blocks, head):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        have_arms = arms_valid(arms)
        hr = head_report(head, self.head_start)
        worst = (max(self.worst_shift, hr["shift"]), max(self.worst_lift, hr["lift"]),
                 max(self.worst_tilt, hr["tilt_deg"]))
        score, ev = evaluate(blocks, head, arms, worst)
        resting = bool(is_resting(head) and all(is_resting(blocks[n]) for n in NAMES))
        success = bool(score == 1.0 and resting)
        details = {
            "valid_state": True,
            "left_side": ev["left_side"],
            "right_side": ev["right_side"],
            "head_disturbed": ev["head_disturbed"],
            "over_head": ev["over_head"],
            "resting": resting,
            "head_shift_mm": _mm(hr["shift"]),
            "head_lift_mm": _mm(hr["lift"]),
            "head_tilt_deg": round(hr["tilt_deg"], 2),
            "worst_shift_mm": _mm(worst[0]),
            "worst_lift_mm": _mm(worst[1]),
            "worst_tilt_deg": round(worst[2], 2),
            "blocks": {n: _brief(r) for n, r in ev["reports"].items()},
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
