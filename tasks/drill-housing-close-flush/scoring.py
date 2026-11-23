"""The scorer, as plain functions of the three loose objects' states and the arms' state. NumPy only; no simulator.

The score is read from the end state, the way an assembler checks a closed clamshell before it goes to the screw
station: the motor-gearbox unit lying in its cradle ribs on the parting plane with its spindle through the nose,
the trigger switch upright in its pocket with the trigger through its slot, and the right half closed down on the
left half all round (no gap at the parting line, its outline in line with the left half's). The ribs, stops,
pocket and slots are physical: a unit turned round, off its ribs or riding on a stop, or a switch turned round or
off its standoff, stands high, and the right half's own ribs then land on it and hold the halves apart. Closing the
halves over a part that is not seated scores 0.

A little history is kept by a tracker fed one observation per control step by ``task.success()``: each object's last
0.5 s of frame positions (for 'at rest') and when a new episode starts. There is no whole-trial rule.

    Tracker.update(objects, arms) -> Result(score, success, details)

``objects`` maps "motor", "switch" and "cover" to their states (mechkit AssemblyStates in the simulator: ``origin``
is the part's frame, ``quat_xyzw`` its orientation; anything without ``origin`` is taken as a box state whose
``pos`` is the centre of the part's bounding box). ``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}``
with ``tcp``, ``tips`` (the point between the pad tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the
arms could not be read; then every loose object counts as held on that step (fail closed: the step scores 0 and is
never a success) and the step is counted in ``arms_unavailable``. An object state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_HOUSING = np.array([L.OX, L.OY, L.TOP + L.PARTING])  # O in the world
_SCORE = {(False, False): 0.0, (True, False): L.MOTOR_SCORE, (False, True): L.SWITCH_SCORE,
          (True, True): round(L.MOTOR_SCORE + L.SWITCH_SCORE, 2)}


# -- the loose parts' shapes in their own frames: ("box", lo, hi) or ("cyl_x", x0, x1, radius) ---------------------
def _box(b):
    return ("box", np.array([b[0][0], b[1][0], b[2][0]]), np.array([b[0][1], b[1][1], b[2][1]]))


def _switch_boxes():
    sx, sy, sz = (v / 2 for v in L.SWITCH_BODY)
    (tx, ty, tz), (wx, wy, wz) = L.TRIGGER
    return [("box", np.array([-sx, -sy, -sz]), np.array([sx, sy, sz])),
            ("box", np.array([tx - wx / 2, ty - wy / 2, tz - wz / 2]), np.array([tx + wx / 2, ty + wy / 2, tz + wz / 2]))]


SHAPES = {
    "motor": [("cyl_x", x0, x1, r) for _, x0, x1, r in L.UNIT_PARTS],
    "switch": _switch_boxes(),
    "cover": [_box(b) for b in L.COVER.values()],
}


def _shape_bounds(sh):
    if sh[0] == "box":
        return sh[1], sh[2]
    _, x0, x1, r = sh
    return np.array([x0, -r, -r]), np.array([x1, r, r])


def local_bounds(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The part's bounding box in its own frame."""
    los, his = zip(*(_shape_bounds(sh) for sh in SHAPES[name]))
    return np.min(los, axis=0), np.max(his, axis=0)


LOCAL_CENTRE = {n: (lambda b: (b[0] + b[1]) / 2)(local_bounds(n)) for n in L.LOOSE}


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
        pos = np.asarray(s.pos, dtype=float).ravel()
        quat = np.asarray(s.quat_xyzw, dtype=float).ravel()
        origin = getattr(s, "origin", None)
        org = pos if origin is None else np.asarray(origin, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(pos.size == 3 and quat.size == 4 and org.size == 3 and np.all(np.isfinite(pos))
                and np.all(np.isfinite(quat)) and np.all(np.isfinite(org)) and np.linalg.norm(quat) > 0.5)


def frame(name: str, s) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of the part's own frame in the world."""
    r = rotation(s.quat_xyzw)
    origin = getattr(s, "origin", None)
    if origin is None:  # a plain box state: pos is the bounding box's centre
        return np.asarray(s.pos, dtype=float) - r @ LOCAL_CENTRE[name], r
    return np.asarray(origin, dtype=float), r


def surface_distance(name: str, s, point) -> float:
    """Distance from a world point to the part's own shapes (0 inside one)."""
    o, r = frame(name, s)
    p = (np.asarray(point, dtype=float) - o) @ r
    best = math.inf
    for sh in SHAPES[name]:
        if sh[0] == "box":
            d = float(np.linalg.norm(np.maximum(0.0, np.maximum(sh[1] - p, p - sh[2]))))
        else:
            _, x0, x1, rad = sh
            da = max(0.0, x0 - p[0], p[0] - x1)
            dr = max(0.0, math.hypot(p[1], p[2]) - rad)
            d = math.hypot(da, dr)
        best = min(best, d)
    return best


def in_outline(xy) -> bool:
    """A world point seen from above lies inside the housing's outline (barrel or handle)."""
    x, y = float(xy[0]) - L.OX, float(xy[1]) - L.OY
    return any(x0 <= x <= x1 and y0 <= y <= y1 for _, (x0, x1), (y0, y1) in L.OUTLINE)


def _deg(c: float) -> float:
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def motor_pose(s) -> dict:
    """Where the motor-gearbox unit lies against its seat: offsets of its frame from O (mm) and its axis's angle
    from the housing's +x (deg; 180 = turned round)."""
    o, r = frame("motor", s)
    d = o - _HOUSING
    return {"dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]), "axis_deg": _deg(float(r[0, 0]))}


def motor_in_seat(s) -> bool:
    p = motor_pose(s)
    return bool(abs(p["dx"]) <= L.MOTOR_AXIAL_TOL and abs(p["dy"]) <= L.MOTOR_SIDE_TOL
                and abs(p["dz"]) <= L.MOTOR_HEIGHT_TOL and p["axis_deg"] <= L.MOTOR_AXIS_DEG)


def switch_pose(s) -> dict:
    """The switch body against its seat: top-view offset (m), height offset (m), tilt and the trigger's heading
    away from +x (deg)."""
    o, r = frame("switch", s)
    seat = np.array(L.world(L.SWITCH_SEAT))
    d = o - seat
    heading = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
    return {"dxy": float(math.hypot(d[0], d[1])), "dz": float(d[2]), "tilt_deg": _deg(abs(float(r[2, 2]))),
            "yaw_deg": abs(heading)}


def switch_in_seat(s) -> bool:
    p = switch_pose(s)
    return bool(p["dxy"] <= L.SWITCH_XY_TOL and abs(p["dz"]) <= L.SWITCH_HEIGHT_TOL
                and p["tilt_deg"] <= L.SWITCH_TILT_DEG and p["yaw_deg"] <= L.SWITCH_YAW_DEG)


def cover_corners(s) -> np.ndarray:
    """The cover's six outline corners at its rim (world, 6 x 3)."""
    o, r = frame("cover", s)
    local = np.array([(x, y, 0.0) for x, y in L.CORNERS])
    return o + local @ r.T


def rim_gaps(s) -> np.ndarray:
    """Each outline corner's height above the parting plane (m; what a feeler gauge reads there)."""
    return cover_corners(s)[:, 2] - _HOUSING[2]


def corner_steps(s) -> np.ndarray:
    """Each outline corner's top-view distance from the left half's matching corner (m; a rule across the step)."""
    want = np.array([(L.OX + x, L.OY + y) for x, y in L.CORNERS])
    return np.linalg.norm(cover_corners(s)[:, :2] - want, axis=1)


def cover_shut(s) -> bool:
    """Down on the rim all round and in line: every corner within 1.5 mm of the parting plane and 4 mm of its
    match."""
    return bool(np.all(np.abs(rim_gaps(s)) <= L.RIM_GAP) and np.all(corner_steps(s) <= L.CORNER_STEP))


# -- the arms ---------------------------------------------------------------------------------------------------------
def arms_valid(arms) -> bool:
    """Both arms present with a finite TCP, tip point and opening (rigstate.arms() can be None)."""
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


def holders(arms, objects) -> dict[str, str | None]:
    """Which gripper holds each loose object (None: not held). 'Held': a gripper less than 90 % open has its
    pad-tip point within 15 mm of the object's surface and no farther from it than from any other loose object (a
    tie holds both). When the arms cannot be read, every object counts as held (``L.UNREADABLE``): it fails closed."""
    if not arms_valid(arms):
        return {n: L.UNREADABLE for n in objects}
    out: dict[str, str | None] = {n: None for n in objects}
    for side in ("left", "right"):
        a = arms[side]
        if float(a.opening) >= L.HELD_OPENING:
            continue
        d = {n: surface_distance(n, s, a.tips) for n, s in objects.items()}
        near = min(d.values())
        for n, v in d.items():
            if v <= L.HELD_REACH and v <= near + 1e-9 and out[n] is None:
                out[n] = side
    return out


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


# -- the trial --------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


START = {"motor": np.array(L.UNIT_START), "switch": np.array(L.SWITCH_START), "cover": np.array(L.COVER_START)}


class Tracker:
    """What the scorer needs from the past: each loose object's last 0.5 s of frame positions, and when a new
    episode starts."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.window = {n: deque(maxlen=L.STILL_STEPS + 1) for n in L.LOOSE}
        self.arms_unavailable = 0
        self.bad_steps = 0
        self.last: Result = Result(0.0, False, {})
        self.prev = None

    @staticmethod
    def at_start(objects) -> bool:
        """Every loose object back on its starting place (within START_TOL) and level."""
        for n in L.LOOSE:
            o, r = frame(n, objects[n])
            if np.linalg.norm(o[:2] - START[n][:2]) >= L.START_TOL or abs(o[2] - START[n][2]) > 0.004:
                return False
            if r[2, 2] < 0.9998:
                return False
        return True

    def path(self, name: str) -> float:
        """The frame origin's path over the last 0.5 s (m); infinite until the window has filled."""
        win = self.window[name]
        if len(win) < L.STILL_STEPS + 1:
            return math.inf
        pts = list(win)
        return float(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))

    def at_rest(self, name: str) -> bool:
        return bool(self.path(name) < L.STILL_PATH)

    def update(self, objects, arms=None) -> Result:
        if not all(valid(objects[n]) for n in L.LOOSE):
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(objects, arms)
            return self.last
        now = {n: frame(n, objects[n])[0] for n in L.LOOSE}
        if self.prev is not None and self.at_start(objects):
            jump = max(float(np.linalg.norm(now[n] - self.prev[n])) for n in L.LOOSE)
            if jump > L.RESET_JUMP:  # the simulator put everything back: a new episode (rigstate's hook normally
                self.reset()  # does this first; this is the fallback)
        self.prev = now
        self.steps += 1
        for n in L.LOOSE:
            self.window[n].append(now[n])
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.last = self.result(objects, arms)
        return self.last

    def result(self, objects, arms=None) -> Result:
        if not all(valid(objects[n]) for n in L.LOOSE):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
        loose = {n: objects[n] for n in L.LOOSE}
        held = holders(arms, loose)
        motor, switch, cover = loose["motor"], loose["switch"], loose["cover"]
        rest = {n: self.at_rest(n) for n in L.LOOSE}
        m_seated = bool(held["motor"] is None and rest["motor"] and motor_in_seat(motor))
        s_seated = bool(held["switch"] is None and rest["switch"] and switch_in_seat(switch))
        seated = {"motor": m_seated, "switch": s_seated}
        inside = {n: in_outline(frame(n, loose[n])[0]) for n in L.PARTS}
        unseated_inside = [n for n in L.PARTS if held[n] is None and inside[n] and not seated[n]]
        c_free = held["cover"] is None
        c_on = bool(c_free and in_outline(frame("cover", cover)[0]))  # its frame: on the barrel's axis
        shut = bool(c_free and rest["cover"] and cover_shut(cover))
        closed_over = bool(c_on and unseated_inside)
        if closed_over:
            score = 0.0
        elif m_seated and s_seated and shut:
            score = 1.0
        elif m_seated and s_seated and c_on and rest["cover"]:
            score = L.COVER_ON_SCORE
        else:
            score = _SCORE[(m_seated, s_seated)]
        have_arms = arms_valid(arms)
        mp, sp = motor_pose(motor), switch_pose(switch)
        gaps, steps = rim_gaps(cover), corner_steps(cover)
        details = {
            "valid_state": True,
            "motor_seated": m_seated, "switch_seated": s_seated, "cover_shut": shut, "cover_on_housing": c_on,
            "closed_over_unseated": closed_over, "unseated_inside": unseated_inside,
            "motor_held_by": held["motor"], "switch_held_by": held["switch"], "cover_held_by": held["cover"],
            "motor_at_rest": rest["motor"], "switch_at_rest": rest["switch"], "cover_at_rest": rest["cover"],
            "motor_path_mm": _mm(self.path("motor")), "switch_path_mm": _mm(self.path("switch")),
            "cover_path_mm": _mm(self.path("cover")),
            "motor_dx_mm": _mm(mp["dx"]), "motor_dy_mm": _mm(mp["dy"]), "motor_dz_mm": _mm(mp["dz"]),
            "motor_axis_deg": round(mp["axis_deg"], 2),
            "switch_dxy_mm": _mm(sp["dxy"]), "switch_dz_mm": _mm(sp["dz"]), "switch_tilt_deg": round(sp["tilt_deg"], 2),
            "switch_yaw_deg": round(sp["yaw_deg"], 2),
            "rim_gaps_mm": [round(float(v) * 1000, 2) for v in gaps],
            "corner_steps_mm": [round(float(v) * 1000, 2) for v in steps],
            "max_rim_gap_mm": round(float(np.max(np.abs(gaps))) * 1000, 2),
            "max_corner_step_mm": round(float(np.max(steps)) * 1000, 2),
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
        }
        return Result(score, bool(score == 1.0 and have_arms), details)
