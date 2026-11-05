"""The scorer, as plain functions of the crate's and the two fork bars' states and the arms' state. NumPy only; no
simulator needed.

The rules are about the whole trial, not only its end: the crate may be lifted only while both bars are at least
two-thirds of the way into their pockets, must never tilt more than 5 degrees, rise above 9.5 cm, fall or be touched
by a gripper. So the scorer keeps a little state (which rules were broken and when, the least insertion any lift was
made with, whether the crate was ever carried on both bars, whether both bars were ever fully in) and is fed one
observation per control step by ``task.success()``.

    Tracker.update(crate, bars, arms) -> Result(score, success, details)

``crate`` and each of ``bars`` (a pair: the bar that starts on the left, the one that starts on the right) are
anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high``, as
``yam_arena.task_file.ObjectState`` has. All three are plain boxes whose sizes are in ``layout``, so their corners
follow from ``pos`` and ``quat_xyzw``. ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``,
``tips`` (the point between the finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not
be read; then the gripper rules are taken as kept for that step and the step is counted in ``arms_unavailable``. An
object state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves
the tracker as it was.

The pockets are the gaps between fixed bearers, not part of the crate, so a bar's insertion is measured against the
crate wherever it is: the length of the bar's centre line that lies under the crate's footprint (crate frame), counted
while that stretch of the line is within 15 mm (across) of one of the crate's two pocket lines (y +-0.120 in its own
frame) and the bar is just under the crate (its centre below the crate's underside, its top no more than 20 mm below
it). A pocket's insertion is the largest of the two bars' there.

rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its 'finger zone'
is every point within 24 mm of the 38 mm segment from the point between its finger tips back up the gripper, at every
opening. 24 mm reaches a finger's outer corner when the fingers are closed on a bar (half the 24 mm gap plus the
finger's thickness, about 6 mm, and half the pad's 27 mm width across: 22.5 mm), whatever the gripper's turn, so such
fingers touching the crate are never missed; fingers opened wider spread along an unknown direction and are not
followed further out (an open gripper 45 mm in front of the crate, its fingers spread along y, is not touching it).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
CRATE_HALF = np.array(L.CRATE_SIZE, dtype=float) / 2
BAR_HALF = np.array(L.BAR_SIZE, dtype=float) / 2
CRATE_START = np.array([L.CRATE_X, L.CRATE_Y, L.TOP + L.FLOOR_TOP + L.CRATE_HALF[2]])
BAR_STARTS = (np.array([L.BAR_X, +L.POCKET_Y, L.TOP + L.BAR_HALF_H]),
              np.array([L.BAR_X, -L.POCKET_Y, L.TOP + L.BAR_HALF_H]))
SIDES = (+1, -1)  # the crate's two pocket lines: +y and -y in its own frame (left and right as it starts)


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
                and all(p.size == 3 for p in parts[2:]))


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


# -- boxes -----------------------------------------------------------------------------------------------------
def corners(s, half) -> np.ndarray:
    """The eight corners (world, 8 x 3) of a box with half sizes ``half`` from its centre and orientation."""
    return np.asarray(s.pos, dtype=float) + (_SIGNS * np.asarray(half, dtype=float)) @ rotation(s.quat_xyzw).T


def to_local(s, p) -> np.ndarray:
    """A world point in the box's own frame."""
    return rotation(s.quat_xyzw).T @ (np.asarray(p, dtype=float) - np.asarray(s.pos, dtype=float))


def point_box_distance(p, s, half) -> float:
    """Distance from a world point to a (turned) box (0 inside it)."""
    d = np.maximum(0.0, np.abs(to_local(s, p)) - np.asarray(half, dtype=float))
    return float(np.linalg.norm(d))


def _hull(points) -> np.ndarray:
    """Convex hull (counter-clockwise) of 2-D points: a box's outline seen from above."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def outline(s, half) -> np.ndarray:
    """The box's outline seen from above (convex polygon of its corners' x, y)."""
    return _hull(corners(s, half)[:, :2])


def outlines_overlap(a: np.ndarray, b: np.ndarray, tol: float = 0.001) -> bool:
    """Do two convex outlines overlap by more than ``tol`` (separating-axis test)?"""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            axis = np.array([-e[1], e[0]])
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() + tol or pb.max() < pa.min() + tol:
                return False
    return True


# -- the crate -------------------------------------------------------------------------------------------------
def tilt_deg(crate) -> float:
    """Angle between the vertical and the axis through the crate's top and bottom (degrees, 0..90): a plain box,
    so whichever of the two faces is nearer to facing down counts."""
    zz = rotation(crate.quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def yaw_off_deg(crate) -> float:
    """How far the crate's long sides are turned from their start about the vertical (degrees, 0..90; a plain box
    turned half a turn is the same box)."""
    x = rotation(crate.quat_xyzw)[:, 0]
    a = math.degrees(math.atan2(float(x[1]), float(x[0])))
    return abs((a + 90.0) % 180.0 - 90.0)


def lowest(crate) -> float:
    """Height of the crate's lowest corner above the table top."""
    return float(corners(crate, CRATE_HALF)[:, 2].min() - L.TOP)


def support_top(crate) -> float:
    """Height of the bearers the crate stands over: the rack's (0.072) once its centre is at or past x 0.400, else
    the floor bearers' (0.036)."""
    return L.RACK_TOP if float(crate.pos[0]) >= L.RACK_SUPPORT_X else L.FLOOR_TOP


def is_lifted(crate) -> bool:
    """The crate's lowest point more than 3 mm above the bearers it stands over."""
    return lowest(crate) > support_top(crate) + L.LIFT_TOL


def drop_speed(crate) -> float:
    """Downward speed of the crate's lowest corner right now, from its velocity and spin (m/s)."""
    cs = corners(crate, CRATE_HALF)
    r = cs[int(np.argmin(cs[:, 2]))] - np.asarray(crate.pos, dtype=float)
    v = np.asarray(crate.lin_vel, dtype=float) + np.cross(np.asarray(crate.ang_vel, dtype=float), r)
    return float(-v[2])


def on_rack(crate) -> bool:
    """Resting on the rack's bearers: its outline from above inside x 0.400..0.530 and within 0.165 of y 0, its
    lowest point within 3 mm of the bearers' top, tilted no more than 3 degrees, turned no more than 5 degrees."""
    xy = corners(crate, CRATE_HALF)[:, :2]
    return bool(
        xy[:, 0].min() >= L.ON_RACK_X0 and xy[:, 0].max() <= L.ON_RACK_X1
        and np.abs(xy[:, 1]).max() <= L.ON_RACK_HALF_Y
        and abs(lowest(crate) - L.RACK_TOP) <= L.ON_RACK_Z_TOL
        and tilt_deg(crate) <= L.ON_RACK_TILT_DEG
        and yaw_off_deg(crate) <= L.ON_RACK_YAW_DEG
    )


# -- the bars --------------------------------------------------------------------------------------------------
def bar_ends(bar) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of a bar's centre line (world)."""
    u = rotation(bar.quat_xyzw)[:, 0] * L.BAR_HALF_LEN
    c = np.asarray(bar.pos, dtype=float)
    return c - u, c + u


def bar_pitch_deg(bar) -> float:
    """How far a bar's long axis is from horizontal (degrees): a bar pitching in the fingers shows here."""
    u = rotation(bar.quat_xyzw)[:, 0]
    return float(np.degrees(np.arcsin(np.clip(abs(u[2]), 0.0, 1.0))))


def insertion(crate, bar) -> tuple[int | None, float]:
    """(pocket, length): which of the crate's pocket lines (+1 or -1, its own +y or -y) the bar is in, and how much
    of its centre line lies under the crate's footprint (m, its length seen from above in the crate's frame); (None,
    0.0) if it is in neither."""
    e0, e1 = bar_ends(bar)
    a, b = to_local(crate, e0), to_local(crate, e1)
    if a[0] > b[0]:
        a, b = b, a
    lo, hi = max(a[0], -CRATE_HALF[0]), min(b[0], CRATE_HALF[0])
    if hi - lo <= 1e-9:
        return None, 0.0
    span = b[0] - a[0]

    def at(x):
        return a + (b - a) * ((x - a[0]) / span if span > 1e-9 else 0.5)

    p0, p1 = at(lo), at(hi)
    z = 0.5 * (p0[2] + p1[2])  # the bar's centre line, half way along the stretch under the crate
    under = -CRATE_HALF[2] - L.BAR_HALF_H - L.UNDER_GAP <= z < -CRATE_HALF[2]
    if not under:
        return None, 0.0
    for side in SIDES:
        if abs(p0[1] - side * L.POCKET_Y) <= L.POCKET_BAND and abs(p1[1] - side * L.POCKET_Y) <= L.POCKET_BAND:
            return side, float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return None, 0.0


def pocket_insertions(crate, bars) -> dict[int, float]:
    """Each pocket's insertion (m): the larger of the two bars' in it."""
    out = {s: 0.0 for s in SIDES}
    for bar in bars:
        side, length = insertion(crate, bar)
        if side is not None:
            out[side] = max(out[side], length)
    return out


def bars_out(crate, bars) -> bool:
    """Both bars wholly outside the crate's outline seen from above (no overlap at all)."""
    c = outline(crate, CRATE_HALF)
    return not any(outlines_overlap(outline(b, BAR_HALF), c, tol=0.0) for b in bars)


# -- the arms --------------------------------------------------------------------------------------------------
def zone_radius(arm) -> float:
    """The finger zone's radius: 24 mm at every opening (a finger's outer corner when closed on a bar, whatever the
    gripper's turn). ``arm`` is accepted so the rule could follow the opening; it does not."""
    return float(L.ZONE_RADIUS)


def finger_zone(arm, samples: int = 20) -> tuple[np.ndarray, float]:
    """Points along the fingers' axis (from the point between the tips back up the gripper) and the zone's radius."""
    tcp, tips = np.asarray(arm.tcp, dtype=float), np.asarray(arm.tips, dtype=float)
    u = tips - tcp
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    s = np.linspace(0.0, L.FINGER_LEN, samples)
    return tips - np.outer(s, u), zone_radius(arm)


def zone_gap(arm, crate) -> float:
    """How far a gripper's finger zone is from the crate (negative: they overlap)."""
    pts, r = finger_zone(arm)
    return min(point_box_distance(p, crate, CRATE_HALF) for p in pts) - r


def clearance(arm, crate, bars) -> float:
    """The smallest distance from the gripper's tool point or finger-tip point to the crate or a bar."""
    return min(point_box_distance(p, s, h) for p in (arm.tcp, arm.tips)
               for s, h in ((crate, CRATE_HALF), (bars[0], BAR_HALF), (bars[1], BAR_HALF)))


def let_go(arms, crate, bars) -> bool:
    """Both grippers open (opening 0.9 or more) with their tool and finger-tip points at least 30 mm from the bars
    and the crate."""
    return all(float(arms[s].opening) >= L.GRIPPER_OPEN and clearance(arms[s], crate, bars) >= L.CLEAR_DIST
               for s in ("left", "right"))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "lifted_on_tips": "the crate was lifted while a bar was less than 73 mm into its pocket",
    "tilted": "the crate tilted more than 5 degrees",
    "too_high": "the crate's lowest point rose more than 95 mm above the table",
    "fell": "the crate's lowest point dropped faster than 0.25 m/s",
    "left_table": "the crate left the table",
    "gripper_touched": "a gripper's fingers touched the crate",
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.min_lift_insertion = math.inf  # m: the least insertion (either pocket) on any step the crate was lifted
        self.lift_steps = 0
        self.carried = False  # lifted 20 mm on both bars with its centre 50 mm toward the rack, at some step
        self.both_full = False  # both pockets 95 mm or more at the same step, at some step
        self.max_tilt = 0.0  # degrees
        self.max_lowest = -math.inf  # m above the table
        self.max_drop = 0.0  # m/s
        self.min_zone_gap = math.inf  # m: the closest either gripper's finger zone came to the crate
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.prev_lowest: float | None = None
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the crate and of each bar] at the previous valid control step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(crate, bars) -> bool:
        """The starting layout: the crate square on the floor bearers, each bar flat on the table in line with its
        pocket, all still (within START_TOL of their marks, which covers a seeded reset's jitter)."""
        c_ok = (np.hypot(*(np.asarray(crate.pos[:2], dtype=float) - CRATE_START[:2])) < L.START_TOL
                and abs(lowest(crate) - L.FLOOR_TOP) < 0.003 and tilt_deg(crate) < 1.0 and is_resting(crate))
        b_ok = all(
            np.hypot(*(np.asarray(b.pos[:2], dtype=float) - start[:2])) < L.START_TOL
            and abs(float(b.pos[2]) - start[2]) < 0.003 and bar_pitch_deg(b) < 1.0 and is_resting(b)
            for b, start in zip(bars, BAR_STARTS)
        )
        return bool(c_ok and b_ok)

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

    def update(self, crate, bars, arms=None) -> Result:
        bars = tuple(bars)
        if not (valid(crate) and all(valid(b) for b in bars)):  # a blow-up: score 0, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(crate, bars, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (crate, *bars)]
        if self.prev is not None and self._teleported(now) and self.at_start(crate, bars):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        low, tilt = lowest(crate), tilt_deg(crate)
        ins = pocket_insertions(crate, bars)
        least = min(ins.values())
        if arms_valid(arms):
            gap = min(zone_gap(arms[s], crate) for s in ("left", "right"))
            self.min_zone_gap = min(self.min_zone_gap, gap)
            if gap < L.TOUCH_GAP:
                self._flag("gripper_touched")
        else:
            self.arms_unavailable += 1
        if is_lifted(crate):
            self.lift_steps += 1
            self.min_lift_insertion = min(self.min_lift_insertion, least)
            if least < L.TWO_THIRDS_IN:
                self._flag("lifted_on_tips")
        if tilt > L.TILT_MAX_DEG:
            self._flag("tilted")
        if low > L.HEIGHT_MAX:
            self._flag("too_high")
        drop = drop_speed(crate)
        if self.prev_lowest is not None:
            drop = max(drop, (self.prev_lowest - low) / self.dt)
        self.prev_lowest = low
        self.max_drop = max(self.max_drop, drop)
        if drop > L.FALL_SPEED:
            self._flag("fell")
        if low < -L.OFF_TABLE_DROP:
            self._flag("left_table")
        if least >= L.FULL_IN:
            self.both_full = True
        if (least >= L.TWO_THIRDS_IN and low - L.FLOOR_TOP >= L.CARRY_LIFT
                and float(crate.pos[0]) - L.CRATE_X >= L.CARRY_ADVANCE):
            self.carried = True
        self.max_tilt = max(self.max_tilt, tilt)
        self.max_lowest = max(self.max_lowest, low)
        self.last = self.result(crate, bars, arms)
        return self.last

    @property
    def clean_lifts(self) -> bool:
        """Every lift so far was made with both bars fully in (95 mm or more)."""
        return self.min_lift_insertion >= L.FULL_IN

    def result(self, crate, bars, arms=None) -> Result:
        bars = tuple(bars)
        if not (valid(crate) and all(valid(b) for b in bars)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        placed = on_rack(crate)
        out = bars_out(crate, bars)
        released = let_go(arms, crate, bars) if have_arms else True  # unreadable: taken as met, counted
        if self.breach:
            score = 0.0
        elif placed and self.clean_lifts and out and released:
            score = 1.0
        elif placed:
            score = 0.75
        elif self.carried:
            score = 0.5
        elif self.both_full:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(crate) and all(is_resting(b) for b in bars))
        ins = pocket_insertions(crate, bars)
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "insertion_mm": {"pos_y": _mm(ins[+1]), "neg_y": _mm(ins[-1])},
            "lifted": is_lifted(crate),
            "lift_steps": self.lift_steps,
            "min_lift_insertion_mm": _mm(self.min_lift_insertion),
            "both_full_seen": self.both_full,
            "carried": self.carried,
            "on_rack": placed,
            "bars_out": out,
            "crate_low_mm": _mm(lowest(crate)),
            "support_mm": _mm(support_top(crate)),
            "crate_tilt_deg": round(tilt_deg(crate), 2),
            "crate_yaw_deg": round(yaw_off_deg(crate), 2),
            "max_tilt_deg": round(self.max_tilt, 2),
            "max_low_mm": _mm(self.max_lowest),
            "max_drop_mps": round(self.max_drop, 3),
            "bar_pitch_deg": [round(bar_pitch_deg(b), 2) for b in bars],
            "zone_gap_mm": ({s: _mm(zone_gap(arms[s], crate)) for s in ("left", "right")} if have_arms else None),
            "min_zone_gap_mm": _mm(self.min_zone_gap),
            "clearance_mm": ({s: _mm(clearance(arms[s], crate, bars)) for s in ("left", "right")}
                             if have_arms else None),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "grippers_let_go": released if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "crate_resting": is_resting(crate),
            "bars_resting": [is_resting(b) for b in bars],
        }
        return Result(score, success, details)
