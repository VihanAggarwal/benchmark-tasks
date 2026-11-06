"""The scorer, as plain functions of the divider's and the four boxes' states and the grippers'. NumPy only; no
simulator needed.

The score is read from the end state: where the divider and the boxes are at the last control step. The scorer
keeps only what that reading needs over time: each object's last 0.5 s of positions (for 'at rest', a path length,
never a velocity, because a gripped object's reported velocity chatters at 10-45 mm/s while it stays put), a count
of steps on which the grippers could not be read or a state was broken, and a new-episode check. It is fed one
observation per control step by ``task.success()``.

    Tracker.update(objects, arms) -> Result(score, success, details)

``objects`` maps "divider", "box_1" .. "box_4" to anything with ``pos`` (centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has. ``arms`` is
``rigstate.arms()``: {"left": Arm, "right": Arm} with ``tcp``, ``tips`` (the pad-tip point, 13 mm past the TCP) and
``opening`` (0 closed .. 1 open; the pad gap is 93.9 mm times it), or None when they could not be read. Then no
object counts as held on that step and the step is counted in ``arms_unavailable``. A state with a non-finite number
or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Terms (task.json 'scoring'):
- At rest: the object's centre moved less than 2 mm (path length) over the last 0.5 s (10 control steps).
- Held: a gripper less than half open (opening under 0.5: pads under 47 mm apart) has its pad-tip point inside the
  object's outline grown by 15 mm on every side. Free means not held.
- Tilt: the angle of a box's 70 mm axis (the divider's 40 mm axis) from vertical, either way up.
- Inside the selection: some part of the object's outline seen from above lies more than 1 mm inside the area over
  the shelf between the walls' inner faces (x 0.160-0.420, y 0.050-0.230), at any height.
- Seated in pitch k: a box free and at rest, its lowest point 19 to 23 mm above the table top (the shelf floor is
  at 20 mm; 1 mm below it allows for a resting contact), tilted at most 8 degrees, its centre inside pitch k's
  54 mm along x and within 20 mm of the lane centreline (y 0.140), its 140 mm side within 10 degrees of the y axis.
- n: the pitches seated in an unbroken run from pitch 1.
- Divider on the mat: free and at rest, its centre over the mat's top, its lowest point 4 to 10 mm above the table
  top (the mat's top is at 5 mm).
Score 0 if any box is free, at rest and inside the selection but not seated (a jam), or any part of the divider is
inside the selection; otherwise 0.2 for the divider on the mat plus 0.2 per pitch in the run.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

ARMS = ("left", "right")
OBJECTS = ("divider", *L.BOX_NAMES)
SIZES = {"divider": L.DIV_SIZE, **{n: L.BOX_SIZE for n in L.BOX_NAMES}}
# Starting marks: (x, y, lowest point) of each loose object.
STARTS = {"divider": (L.DIV_X, L.DIV_Y, L.SHELF), **{n: (x, L.BOX_Y, L.TOP) for n, x in zip(L.BOX_NAMES, L.BOX_XS)}}
GRAVITY = 9.81
_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)  # the eight corners of a unit box


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


def all_valid(objects) -> bool:
    try:
        return all(valid(objects[n]) for n in OBJECTS)
    except (KeyError, TypeError):
        return False


def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP, pad-tip point and opening (rigstate.arms() can be None)."""
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


# -- geometry of one object -----------------------------------------------------------------------------------------
def corners(s, size) -> np.ndarray:
    """The eight corners of the object's real (turned and tilted) box, (8, 3)."""
    r = rotation(s.quat_xyzw)
    return np.asarray(s.pos, dtype=float) + (_SIGNS * (np.asarray(size, dtype=float) / 2)) @ r.T


def lowest(s, size) -> float:
    """Height of the object's lowest point (world z, m)."""
    return float(corners(s, size)[:, 2].min())


def tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and the body's z axis (a box's 70 mm axis, the divider's 40 mm axis), either way
    up (degrees, 0..90)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def yaw_off_deg(quat_xyzw) -> float:
    """How far a box's 140 mm side (its body y axis), seen from above, is turned from the world's y axis, either way
    round (degrees, 0..90)."""
    r = rotation(quat_xyzw)
    a = math.degrees(math.atan2(r[0, 1], r[1, 1])) % 180.0
    return float(min(a, 180.0 - a))


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


def outline(s, size) -> np.ndarray:
    """The object's real outline seen from above, (n, 2): the shadow of its eight corners."""
    return _hull(corners(s, size)[:, :2])


def _overlaps_rect(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Does a convex polygon (n, 2) overlap the axis-aligned rectangle [x0, x1] x [y0, y1]? (Separating axes.)"""
    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        if np.hypot(*e) > 1e-12:
            axes.append(np.array([-e[1], e[0]]))
    for a in axes:
        p, q = poly @ a, rect @ a
        if p.max() < q.min() or q.max() < p.min():
            return False
    return True


def inside_selection(s, size) -> bool:
    """Some part of the outline seen from above lies more than 1 mm inside x 0.160-0.420, y 0.050-0.230 (over the
    shelf between the walls' inner faces), at any height."""
    d = L.SEL_INSET
    return _overlaps_rect(outline(s, size), L.SEL_X0 + d, L.SEL_X1 - d, L.SEL_Y0 + d, L.SEL_Y1 - d)


def tip_in_grown(tips, s, size, grow: float = L.HELD_GROW) -> bool:
    """Is the point inside the object's box grown by ``grow`` on every side?"""
    r = rotation(s.quat_xyzw)
    local = r.T @ (np.asarray(tips, dtype=float) - np.asarray(s.pos, dtype=float))
    return bool(np.all(np.abs(local) <= np.asarray(size, dtype=float) / 2 + grow))


def held(s, size, arms) -> bool:
    """Held: some gripper less than half open has its pad-tip point inside the outline grown by 15 mm. Unreadable
    arms hold nothing."""
    if not arms_valid(arms):
        return False
    return any(float(arms[a].opening) < L.HELD_OPENING and tip_in_grown(arms[a].tips, s, size) for a in ARMS)


def pitch_at(x: float) -> int | None:
    """The pitch (0..3 for pitches 1..4) whose 54 mm clear space along x holds x, or None."""
    for k, (a, b) in enumerate(L.PITCHES):
        if a <= x <= b:
            return k
    return None


def seat_checks(box) -> dict:
    """The geometric parts of 'seated' (the box being free and at rest is checked by the tracker)."""
    size = L.BOX_SIZE
    return {
        "low_ok": L.SEAT_LOW_MIN <= lowest(box, size) <= L.SEAT_LOW_MAX,
        "tilt_ok": tilt_deg(box.quat_xyzw) <= L.SEAT_TILT_DEG,
        "lane_ok": abs(float(box.pos[1]) - L.LANE_Y) <= L.SEAT_Y_TOL,
        "yaw_ok": yaw_off_deg(box.quat_xyzw) <= L.SEAT_YAW_DEG,
        "pitch": pitch_at(float(box.pos[0])),
    }


def seated_geometry(box) -> int | None:
    """The pitch a box stands in by geometry alone (0..3), or None."""
    c = seat_checks(box)
    if c["low_ok"] and c["tilt_ok"] and c["lane_ok"] and c["yaw_ok"]:
        return c["pitch"]
    return None


def over_mat(div) -> bool:
    """The divider's centre over the mat's top, and its lowest point 4 to 10 mm above the table top (on the mat's
    5 mm top, up to 5 mm above it)."""
    hx, hy = L.MAT_SIZE[0] / 2, L.MAT_SIZE[1] / 2
    x, y = float(div.pos[0]), float(div.pos[1])
    centred = (L.MAT_X - hx <= x <= L.MAT_X + hx) and (L.MAT_Y - hy <= y <= L.MAT_Y + hy)
    return bool(centred and L.MAT_LOW_MIN <= lowest(div, L.DIV_SIZE) <= L.MAT_LOW_MAX)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Each object's last 0.5 s of positions, and counts of unreadable steps."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_S * control_hz)))  # 10 steps at 20 Hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.centres = {n: deque(maxlen=self.window + 1) for n in OBJECTS}
        self.prev = None  # {name: (centre, velocity)} at the previous valid control step
        self.first_full_step = None  # the first step scored 1.0 (for the trace)
        self.last: Result = Result(0.0, False, {})

    # -- stillness ----------------------------------------------------------------------------------------------
    def path(self, name: str) -> float | None:
        """The object's centre's path over the last 0.5 s (m), or None before 0.5 s of positions have been seen."""
        pts = self.centres[name]
        if len(pts) < self.window + 1:
            return None
        a = np.array(pts)
        return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))

    def at_rest(self, name: str) -> bool:
        p = self.path(name)
        return p is not None and p < L.STILL_PATH

    # -- a new episode ------------------------------------------------------------------------------------------
    @staticmethod
    def at_start(objects) -> bool:
        """The starting layout: every loose object upright on its starting mark (within START_TOL)."""
        for n in OBJECTS:
            s = objects[n]
            x, y, z = STARTS[n]
            if math.hypot(float(s.pos[0]) - x, float(s.pos[1]) - y) > L.START_TOL:
                return False
            if abs(lowest(s, SIZES[n]) - z) > 0.003 or tilt_deg(s.quat_xyzw) > 1.0:
                return False
        return True

    def _teleported(self, now) -> bool:
        """Some object moved, in one control step, RESET_JUMP further than its last velocity plus a free fall could
        have carried it: the simulator put it there."""
        fall = 0.5 * GRAVITY * self.dt ** 2
        return any(
            float(np.linalg.norm(now[n][0] - self.prev[n][0]))
            > L.RESET_JUMP + float(np.linalg.norm(self.prev[n][1])) * self.dt + fall
            for n in OBJECTS
        )

    # -- per step -----------------------------------------------------------------------------------------------
    def update(self, objects, arms=None) -> Result:
        if not all_valid(objects):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(objects, arms)
            return self.last
        now = {n: (np.array(objects[n].pos, dtype=float), np.array(objects[n].lin_vel, dtype=float))
               for n in OBJECTS}
        if self.prev is not None and self._teleported(now) and self.at_start(objects):
            self.reset()  # everything is back on its marks: a new episode (rigstate's reset hook normally does
            #               this first; this is the fallback)
        self.steps += 1
        for n in OBJECTS:
            self.centres[n].append(now[n][0])
        self.prev = now
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(objects, arms)
        if self.last.score == 1.0 and self.first_full_step is None:
            self.first_full_step = self.steps
        return self.last

    def result(self, objects, arms=None) -> Result:
        if not all_valid(objects):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "arms_unavailable": self.arms_unavailable})
        have_arms = arms_valid(arms)
        boxes, seated, jams = {}, set(), []
        for n in L.BOX_NAMES:
            s = objects[n]
            rest, hold = self.at_rest(n), held(s, L.BOX_SIZE, arms)
            inside = inside_selection(s, L.BOX_SIZE)
            pitch = seated_geometry(s) if (rest and not hold) else None
            jam = bool(rest and not hold and inside and pitch is None)
            if pitch is not None:
                seated.add(pitch)
            if jam:
                jams.append(n)
            c = seat_checks(s)
            boxes[n] = {
                "pitch": None if pitch is None else pitch + 1,
                "jam": jam,
                "inside": inside,
                "held": hold,
                "at_rest": rest,
                "path_mm": _mm(self.path(n)),
                "low_mm": _mm(lowest(s, L.BOX_SIZE) - L.TOP),
                "tilt_deg": round(tilt_deg(s.quat_xyzw), 2),
                "yaw_off_deg": round(yaw_off_deg(s.quat_xyzw), 2),
                "lane_off_mm": _mm(float(s.pos[1]) - L.LANE_Y),
                "x_pitch": None if c["pitch"] is None else c["pitch"] + 1,
                "pos": [round(float(v), 4) for v in s.pos],
            }
        div = objects["divider"]
        d_rest, d_hold = self.at_rest("divider"), held(div, L.DIV_SIZE, arms)
        d_inside = inside_selection(div, L.DIV_SIZE)
        on_mat = bool(d_rest and not d_hold and over_mat(div))
        n_run = 0
        while n_run < len(L.PITCHES) and n_run in seated:
            n_run += 1
        zeros = []
        if jams:
            zeros.append("Z1 jammed box: " + ", ".join(jams))
        if d_inside:
            zeros.append("Z2 divider inside the selection")
        score = 0.0 if zeros else (2 * (int(on_mat) + n_run)) / 10
        success = bool(score == 1.0)
        details = {
            "valid_state": True,
            "n": n_run,
            "seated_pitches": sorted(k + 1 for k in seated),
            "divider_on_mat": on_mat,
            "zeros": zeros,
            "divider": {
                "inside": d_inside,
                "held": d_hold,
                "at_rest": d_rest,
                "over_mat": over_mat(div),
                "path_mm": _mm(self.path("divider")),
                "low_mm": _mm(lowest(div, L.DIV_SIZE) - L.TOP),
                "tilt_deg": round(tilt_deg(div.quat_xyzw), 2),
                "pos": [round(float(v), 4) for v in div.pos],
            },
            "boxes": boxes,
            "opening": ({a: round(float(arms[a].opening), 3) for a in ARMS} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "first_full_step": self.first_full_step,
        }
        return Result(score, success, details)
