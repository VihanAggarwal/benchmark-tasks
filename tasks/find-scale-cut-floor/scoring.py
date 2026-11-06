"""The scorer, as plain functions of the three loose objects' states and the arms'. NumPy only; no simulator needed.

Before a find is lifted it is photographed where it lies: tools come out of the shot, and a scale goes beside the
find, parallel to it and flat on the same surface (here the floor of a 40 mm cut, not the ground surface above it),
without the find being moved. So the scorer keeps a little state over the whole trial (the find's pose at the first
control step, whether it has been disturbed since, the objects' recent positions for stillness) and is fed one
observation per control step by ``task.success()``:

    Tracker.update(find, trowel, scale, arms, episode) -> Result(score, success, details)

``find``, ``trowel`` and ``scale`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms``
is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` and ``opening`` (0 closed .. 1 open),
or ``None`` when the arms could not be read; then the hands are not clear (no credit) and the step is counted in
``arms_unavailable``. An object state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0
for that step and leaves the tracker as it was. ``episode`` is ``rigstate.episode()``: the tracker starts a new trial
when it changes and at no other time, so nothing clears a latched 'disturbed' within one episode. Only with
``episode=None`` (the scorer used without rigstate) does it fall back to spotting a new trial from the objects: all
three back on their starting poses with one of them jumping more than 20 mm in one control step.

Terms (task.json 'scoring'). Find frame: u along the find's long axis, n square to it, seen from above.
- Disturbed (latched): at any step the find's centre is more than 5 mm (horizontally) from its reference, its
  heading has turned more than 3 deg, it has tilted more than 3 deg, or its bottom is more than 3 mm above the floor.
- Rests on the find: the scale's or the trowel's outline seen from above overlaps the find's and that object's
  bottom is less than 21 mm above the cut floor.
- On the floor: the scale's bottom within 3 mm of the cut floor, its 15 mm axis within 5 deg of vertical (either
  big face down: a real build bands both 100 x 20 mm faces), its whole outline seen from above inside the cut
  opening (2 mm slack for contact penetration).
- a: the angle seen from above between the scale's and the find's long axes, folded to 0-90 deg. g: the scale's
  centre's distance from the find's long centre line (along n) minus 23 mm. s: the distance from the find's centre
  to the scale's along u. Overlap: the two outlines overlap seen from above.
- Tight: on the floor, a <= 5 deg, 10 mm <= g <= 40 mm, s <= 25 mm, no overlap. Rough: on the floor, a <= 12 deg,
  5 mm <= g <= 50 mm, s <= 45 mm, no overlap (tight is also rough).
- Stowed: the trowel's centre inside the tray's inner outline, its bottom within 5 mm of the tray floor.
- Hands clear: both grippers open 0.5 or more (pads 47 mm or more apart) and neither TCP, seen from above, at
  x 0.05 or more with y -0.21..+0.21: over the frame, within 60 mm of it (a fully open gripper's pads reach at most
  about 56 mm from its TCP, so no pad is over the frame either) or beyond its far side (the arm would reach over
  the frame).
- Still (for success): each loose object's centre within 3 mm of every position it had over the last 0.5 s: a net
  displacement of positions, never a velocity (a gripped part's reported velocity chatters at 10-45 mm/s).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

ARMS = ("left", "right")
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box
SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


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
    sizes_ok = parts[0].size == 3 and parts[1].size == 4 and all(p.size == 3 for p in parts[2:])
    return bool(sizes_ok and np.linalg.norm(parts[1]) > 0.5)


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP and opening (rigstate.arms() can be None)."""
    if not isinstance(arms, dict):
        return False
    try:
        for side in ARMS:
            a = arms[side]
            tcp = np.asarray(a.tcp, dtype=float).ravel()
            if tcp.size != 3 or not np.all(np.isfinite(tcp)) or not math.isfinite(float(a.opening)):
                return False
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    return True


# -- angles ----------------------------------------------------------------------------------------------------------
def yaw_rad(quat_xyzw) -> float:
    """Heading of the body's long (x) axis seen from above (rad)."""
    r = rotation(quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def tilt_deg(quat_xyzw) -> float:
    """Angle between the body's z axis and the world's vertical (degrees, 0..180): a find turned over is tilted."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def flat_tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and the scale's 15 mm axis (degrees, 0..90): either 100 x 20 mm face may be down
    (the simulated scale is plain on both; a real build paints the bands on both, so either face up reads)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def axes_angle_deg(yaw_a: float, yaw_b: float) -> float:
    """Angle between two long axes seen from above, folded to 0..90 degrees (an axis has no direction)."""
    return float(abs(math.degrees(math.remainder(yaw_a - yaw_b, math.pi))))


def turn_deg(yaw_a: float, yaw_b: float) -> float:
    """How far a heading turned from yaw_b to yaw_a (degrees, 0..180)."""
    return float(abs(math.degrees(math.remainder(yaw_a - yaw_b, 2 * math.pi))))


# -- outlines seen from above -----------------------------------------------------------------------------------------
def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(list(reversed(pts)))
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def corners(s, size) -> np.ndarray:
    """The eight corners of a box of ``size`` (world, 8 x 3) from its centre and orientation."""
    half = np.asarray(size, dtype=float) / 2
    return np.asarray(s.pos, dtype=float) + (_SIGNS * half) @ rotation(s.quat_xyzw).T


def footprint(s, size) -> np.ndarray:
    """A box's real outline seen from above, (n, 2): the shadow of its eight corners, turned and tilted as it is."""
    return _hull(corners(s, size)[:, :2])


def polys_overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Do two convex polygons (n, 2) overlap (touching edges do not count)? Separating axes."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            if np.hypot(*e) < 1e-12:
                continue
            a = np.array([-e[1], e[0]])
            pa, qa = p @ a, q @ a
            if pa.max() <= qa.min() + 1e-12 or qa.max() <= pa.min() + 1e-12:
                return False
    return True


# -- the find and the scale --------------------------------------------------------------------------------------------
def find_frame(find) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The find's centre seen from above, its long axis u and the square direction n (n = u turned +90 deg)."""
    a = yaw_rad(find.quat_xyzw)
    u = np.array([math.cos(a), math.sin(a)])
    return np.asarray(find.pos, dtype=float)[:2], u, np.array([-u[1], u[0]])


def scale_vs_find(scale, find) -> dict:
    """a (deg, 0..90), g and s (m), the side of the find the scale is on (+1: +n, toward +y; -1: -n) and overlap."""
    c, u, n = find_frame(find)
    d = np.asarray(scale.pos, dtype=float)[:2] - c
    along_n = float(d @ n)
    return {
        "a_deg": axes_angle_deg(yaw_rad(scale.quat_xyzw), yaw_rad(find.quat_xyzw)),
        "g": abs(along_n) - L.HALF_WIDTHS,
        "s": abs(float(d @ u)),
        "side": 1 if along_n >= 0 else -1,
        "overlap": polys_overlap(footprint(scale, L.SCALE_SIZE), footprint(find, L.FIND_SIZE)),
    }


def inside_cut(poly: np.ndarray, slack: float = L.CUT_SLACK) -> bool:
    x0, x1, y0, y1 = L.CUT
    return bool(np.all(poly[:, 0] >= x0 - slack) and np.all(poly[:, 0] <= x1 + slack)
                and np.all(poly[:, 1] >= y0 - slack) and np.all(poly[:, 1] <= y1 + slack))


def scale_on_floor(scale) -> bool:
    """Flat on the cut floor: bottom within 3 mm of it, 15 mm axis within 5 deg of vertical, outline inside the cut."""
    return bool(abs(float(scale.low[2]) - L.TOP) <= L.FLOOR_TOL and flat_tilt_deg(scale.quat_xyzw) <= L.FLAT_DEG
                and inside_cut(footprint(scale, L.SCALE_SIZE)))


def placement(scale, find) -> tuple[bool, bool, dict]:
    """(tight, rough, metrics) of the scale against the find; tight is also rough."""
    m = scale_vs_find(scale, find)
    floor = scale_on_floor(scale)
    tight = (floor and not m["overlap"] and m["a_deg"] <= L.TIGHT_A_DEG
             and L.TIGHT_G[0] <= m["g"] <= L.TIGHT_G[1] and m["s"] <= L.TIGHT_S)
    rough = (floor and not m["overlap"] and m["a_deg"] <= L.ROUGH_A_DEG
             and L.ROUGH_G[0] <= m["g"] <= L.ROUGH_G[1] and m["s"] <= L.ROUGH_S)
    m["on_floor"] = floor
    return bool(tight), bool(rough), m


def rests_on_find(obj, size, find) -> bool:
    """The object's outline seen from above overlaps the find's and its bottom is under 21 mm above the cut floor."""
    low = float(obj.low[2]) - L.TOP
    return bool(low < L.REST_ON_FIND_H and polys_overlap(footprint(obj, size), footprint(find, L.FIND_SIZE)))


def trowel_stowed(trowel) -> bool:
    """The trowel's centre inside the tray's inner outline, its bottom within 5 mm of the tray floor."""
    x0, x1, y0, y1 = L.TRAY_INNER
    x, y = float(trowel.pos[0]), float(trowel.pos[1])
    return bool(x0 <= x <= x1 and y0 <= y <= y1 and abs(float(trowel.low[2]) - L.TRAY_FLOOR_Z) <= L.STOW_TOL)


def tcp_in_photo(tcp) -> bool:
    """The TCP, seen from above (at any height), at x 0.05 or more with y -0.21..+0.21: over the frame, within
    60 mm of it, or beyond its far side."""
    x0, x1, y0, y1 = L.CLEAR_BOX
    return bool(x0 <= float(tcp[0]) <= x1 and y0 <= float(tcp[1]) <= y1)


def hands_clear(arms) -> bool:
    """Both grippers open at least halfway and neither TCP over the frame, within 60 mm of it or beyond its far
    side. Unreadable arms are not clear."""
    if not arms_valid(arms):
        return False
    return all(float(arms[s].opening) >= L.HALF_OPEN and not tcp_in_photo(arms[s].tcp) for s in ARMS)


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


_START = {  # name: (x, y, bottom height above the table, yaw)
    "find": (L.FIND_X, L.FIND_Y, 0.0, L.FIND_YAW),
    "trowel": (L.TROWEL_X, L.TROWEL_Y, 0.0, L.TROWEL_YAW),
    "scale": (L.SCALE_X, L.SCALE_Y, L.KIT_FLOOR_H, L.SCALE_YAW),
}


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = int(round(L.STILL_WINDOW_S * control_hz)) + 1  # this step and the previous 0.5 s
        self.episode = None  # the last rigstate episode number seen (kept across reset())
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.ref = None  # the find's (centre xy, yaw) at the first valid control step
        self.disturbed = None  # (step, reason) the first time the find was disturbed
        self.max_move = 0.0  # m: the find's largest horizontal displacement from its reference
        self.max_turn = 0.0  # deg
        self.max_tilt = 0.0  # deg
        self.max_lift = 0.0  # m: its bottom's highest point above the cut floor
        self.arms_unavailable = 0
        self.bad_steps = 0
        self.hist = {n: deque(maxlen=self.window) for n in _START}
        self.prev = None  # {name: centre} at the previous valid control step
        self.last: Result = Result(0.0, False, {})

    @staticmethod
    def at_start(objs: dict) -> bool:
        """The starting layout: all three loose objects within 2 mm and 2 deg of their starting poses, flat."""
        for name, (x, y, h, yaw) in _START.items():
            s = objs[name]
            if (math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) > L.START_TOL
                    or abs(float(s.low[2]) - L.TOP - h) > L.START_TOL
                    or turn_deg(yaw_rad(s.quat_xyzw), yaw) > L.START_TURN_DEG
                    or tilt_deg(s.quat_xyzw) > L.START_TURN_DEG):
                return False
        return True

    def update(self, find, trowel, scale, arms=None, episode=None) -> Result:
        """One control step. ``episode`` (rigstate.episode()): a new trial starts when it changes, and only then.
        With ``episode=None`` a teleport of all three loose objects back onto their marks starts one instead."""
        if episode is not None:
            if self.episode is not None and episode != self.episode:
                self.reset()  # a new episode (rigstate's on_reset hook has normally done this already)
            self.episode = episode
        objs = {"find": find, "trowel": trowel, "scale": scale}
        if not all(valid(s) for s in objs.values()):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(find, trowel, scale, arms)
            return self.last
        now = {n: np.array(s.pos, dtype=float) for n, s in objs.items()}
        if episode is None and self.prev is not None and self.at_start(objs):
            if max(float(np.linalg.norm(now[n] - self.prev[n])) for n in now) > L.RESET_JUMP:
                self.reset()  # without an episode number only: everything teleported back onto its marks
        self.prev = now
        self.steps += 1
        if not arms_valid(arms):
            self.arms_unavailable += 1

        yaw = yaw_rad(find.quat_xyzw)
        if self.ref is None:
            self.ref = (now["find"][:2].copy(), yaw)
        move = float(np.hypot(*(now["find"][:2] - self.ref[0])))
        turn = turn_deg(yaw, self.ref[1])
        tilt = tilt_deg(find.quat_xyzw)
        lift = float(find.low[2]) - L.TOP
        self.max_move, self.max_turn = max(self.max_move, move), max(self.max_turn, turn)
        self.max_tilt, self.max_lift = max(self.max_tilt, tilt), max(self.max_lift, lift)
        if self.disturbed is None:
            for bad, why in ((move > L.FIND_MOVE, "moved"), (turn > L.FIND_TURN_DEG, "turned"),
                             (tilt > L.FIND_TILT_DEG, "tilted"), (lift > L.FIND_LIFT, "lifted")):
                if bad:
                    self.disturbed = (self.steps, why)
                    break
        for n, p in now.items():
            self.hist[n].append(p)
        self.last = self.result(find, trowel, scale, arms)
        return self.last

    def still(self) -> bool:
        """Each loose object's centre within 3 mm of every position it had over the last 0.5 s (a full window)."""
        for h in self.hist.values():
            if len(h) < self.window:
                return False
            p = h[-1]
            if max(float(np.linalg.norm(p - q)) for q in h) >= L.STILL_MOVE:
                return False
        return True

    def result(self, find, trowel, scale, arms=None) -> Result:
        if not all(valid(s) for s in (find, trowel, scale)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "disturbed": self.disturbed is not None})
        tight, rough, m = placement(scale, find)
        stowed = trowel_stowed(trowel)
        scale_on, trowel_on = rests_on_find(scale, L.SCALE_SIZE, find), rests_on_find(trowel, L.TROWEL_SIZE, find)
        clear = hands_clear(arms)
        if self.disturbed is not None or scale_on or trowel_on or not clear:
            score = 0.0
        elif tight and stowed:
            score = 1.0
        elif rough and stowed:
            score = 0.75
        elif rough:  # tight or rough (tight is also rough), trowel not stowed
            score = 0.5
        elif stowed:
            score = 0.25
        else:
            score = 0.0
        still = self.still()
        success = bool(score == 1.0 and still)
        have_arms = arms_valid(arms)
        if self.ref is not None:
            move = float(np.hypot(*(np.asarray(find.pos, dtype=float)[:2] - self.ref[0])))
            turn = turn_deg(yaw_rad(find.quat_xyzw), self.ref[1])
        else:
            move, turn = None, None
        details = {
            "valid_state": True,
            "disturbed": self.disturbed is not None,
            "disturbed_step": self.disturbed[0] if self.disturbed else None,
            "disturbed_why": self.disturbed[1] if self.disturbed else None,
            "find_move_mm": _mm(move),
            "find_turn_deg": round(turn, 2) if turn is not None else None,
            "find_tilt_deg": round(tilt_deg(find.quat_xyzw), 2),
            "find_lift_mm": _mm(float(find.low[2]) - L.TOP),
            "find_max_move_mm": _mm(self.max_move),
            "find_max_turn_deg": round(self.max_turn, 2),
            "find_max_tilt_deg": round(self.max_tilt, 2),
            "find_max_lift_mm": _mm(self.max_lift),
            "scale_on_floor": m["on_floor"],
            "scale_bottom_mm": _mm(float(scale.low[2]) - L.TOP),
            "scale_tilt_deg": round(flat_tilt_deg(scale.quat_xyzw), 2),
            "scale_yaw_deg": round(math.degrees(yaw_rad(scale.quat_xyzw)), 2),
            "scale_in_cut": inside_cut(footprint(scale, L.SCALE_SIZE)),
            "a_deg": round(m["a_deg"], 2),
            "g_mm": _mm(m["g"]),
            "s_mm": _mm(m["s"]),
            "scale_side": m["side"],
            "overlap": m["overlap"],
            "tight": tight,
            "rough": rough,
            "stowed": stowed,
            "trowel_bottom_over_tray_floor_mm": _mm(float(trowel.low[2]) - L.TRAY_FLOOR_Z),
            "trowel_tilt_deg": round(flat_tilt_deg(trowel.quat_xyzw), 2),
            "scale_rests_on_find": scale_on,
            "trowel_rests_on_find": trowel_on,
            "hands_clear": clear,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ARMS} if have_arms else None),
            "tcp_in_photo": ({s: tcp_in_photo(arms[s].tcp) for s in ARMS} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "still": still,
            "bad_steps": self.bad_steps,
        }
        return Result(score, success, details)
