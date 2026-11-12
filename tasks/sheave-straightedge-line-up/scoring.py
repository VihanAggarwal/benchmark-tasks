"""The scorer, as plain functions of the motor's state, the straightedge's state and the two grippers. NumPy only;
no simulator needed.

The task is judged on the end state (how far the motor's pulley face is out of line with the blower pulley's face)
and on two rules checked on EVERY control step of the trial: the motor stays on its plate and on its line. The tracker
keeps what a judgement 'at rest' and 'hands off' needs (the motor's and the grippers' last 0.5 s), the first breach of
each whole-trial rule (latched until the next reset) and some counters for the trace, and is fed one observation per
control step by ``task.success()``:

    Tracker.update(motor, straightedge, arms) -> Result(score, success, details)

``motor`` is anything with ``pos``, ``quat_xyzw``, ``lin_vel``, ``ang_vel``, ``low`` and ``high`` (as
``yam_arena.task_file.ObjectState`` has) and, from mechkit, ``origin`` (the centre of the motor body's underside;
worked out from ``pos`` when missing). ``straightedge`` is a plain ObjectState. ``arms`` is ``rigstate.arms()``:
each gripper's TCP, pad-tip point and opening (0 closed .. 1 open), or None when they cannot be read. Everything
fails closed: a step whose arms cannot be read is NOT hands off (and is counted in ``arms_unavailable``), so the
score is 0 until 0.5 s of readable, clear steps have passed; a step whose motor state has a non-finite number or a
zero quaternion (a simulator blow-up) breaks both whole-trial rules (0 for the rest of the trial; the rest window is
left alone); a bad straightedge state is not hands off for that step.

Definitions (layout.py holds the numbers):

- F, the centre of the motor pulley's outer face: the motor's frame origin + R (-0.088, 0, 0.040).
- e = F_x - 0.230 (m): + behind the blower pulley's face plane (pushed too far), - proud of it (toward the arms).
- theta, the motor's heading: the signed angle of its local x axis from world x, seen from above. It is never
  folded: a motor turned round 180 deg is 180 deg out.
- alpha = atan(|e| / 0.290 m): the parallel offset as an angle over the 290 mm centre distance.
- M = |theta| + alpha (degrees): angular plus parallel misalignment, added as the belt maker's guidance adds them.

Score 0 unless all of, at the end: at rest (F's path under 2 mm and the heading's summed turn under 0.5 deg over the
last 0.5 s); upright (the body's up axis within 2 deg of vertical); on its plate (all four bottom corners of the body
inside the plate's outline and within 2 mm of its top); on its line (origin y within 10 mm of 0.140); hands off (on
every step of the last 0.5 s both arms readable and every TCP and pad-tip point at least 60 mm from the motor's box
and from the straightedge's box: a fully open finger's outer face is 57 mm out from its TCP, so open fingers resting
on either fail it too). And all of, on every control step of the trial: stayed on its plate (the body's lowest bottom
corner within 2 mm of the plate's top, the centre of its underside over the plate's outline, its up axis within
10 deg of vertical: lifting it more than 2 mm off the plate, even for one step, breaks it); stayed on its line
(origin y within 10 mm of 0.140). Then 1.0 for M <= 0.5 deg, 0.5 for M <= 1.0, 0.25 for M <= 2.0, else 0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")
ARMS = ("left", "right")
FACE_LOCAL = np.array(L.FACE_LOCAL, dtype=float)
BOX_CENTRE = np.array(L.MOTOR_BOX_CENTRE, dtype=float)
_CORNERS_LOCAL = np.array([(sx * L.BODY_HALF[0], sy * L.BODY_HALF[1], 0.0) for sx in (-1, 1) for sy in (-1, 1)])
PLATE_TOP_W = L.TOP + L.PLATE_TOP  # world z of the plate's top
GATES = ("at_rest", "upright", "on_plate", "on_line", "hands_off", "stayed_on_plate", "stayed_on_line")
POSSIBLE_SCORES = (0.0, 0.25, 0.5, 1.0)


# -- states ---------------------------------------------------------------------------------------------------------
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
        origin = getattr(s, "origin", None)
        if origin is not None:
            parts.append(np.asarray(origin, dtype=float).ravel())
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(p.size and np.all(np.isfinite(p)) for p in parts):
        return False
    sizes = [p.size for p in parts]
    return bool(sizes[:6] == [3, 4, 3, 3, 3, 3] and (len(sizes) == 6 or sizes[6] == 3)
                and np.linalg.norm(parts[1]) > 0.5)


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


def wrap_deg(a: float) -> float:
    """An angle (degrees) folded into (-180, 180]."""
    return float(-((-a + 180.0) % 360.0 - 180.0))


# -- the motor ------------------------------------------------------------------------------------------------------
def motor_pose(m) -> tuple[np.ndarray, np.ndarray]:
    """(rotation, origin) of the motor's frame in the world (origin: the centre of the body's underside)."""
    r = rotation(m.quat_xyzw)
    origin = getattr(m, "origin", None)
    if origin is None:
        origin = np.asarray(m.pos, dtype=float) - r @ BOX_CENTRE
    return r, np.asarray(origin, dtype=float)


def face_centre(m) -> np.ndarray:
    """F: the centre of the motor pulley's outer face (world)."""
    r, o = motor_pose(m)
    return o + r @ FACE_LOCAL


def offset(m) -> float:
    """e (m): how far the motor's pulley face centre is behind (+) or proud of (-) the blower pulley's face plane."""
    return float(face_centre(m)[0] - L.PLANE_X)


def heading_deg(m) -> float:
    """theta (degrees): the motor's local x axis from world x seen from above, signed, never folded."""
    r = rotation(m.quat_xyzw)
    return float(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def alpha_deg(e: float) -> float:
    """The parallel offset as an angle over the centre distance (degrees)."""
    return float(math.degrees(math.atan(abs(e) / L.CENTRE_DIST)))


def misalignment(e: float, theta_deg: float) -> float:
    """M (degrees) = |theta| + alpha."""
    return float(abs(theta_deg) + alpha_deg(e))


def misalignment_deg(m) -> float:
    return misalignment(offset(m), heading_deg(m))


def tilt_deg(m) -> float:
    """Angle between the motor body's up axis and the world's vertical (degrees; 180 upside down)."""
    zz = rotation(m.quat_xyzw)[2, 2]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, zz)))))


def body_corners(m) -> np.ndarray:
    """The four bottom corners of the motor's body (world, 4 x 3)."""
    r, o = motor_pose(m)
    return _CORNERS_LOCAL @ r.T + o


def plate_margin(m) -> float:
    """How far (m) the body's bottom corner nearest the plate's edge is inside the outline (negative: outside)."""
    c = body_corners(m)
    return float(np.min(np.stack([c[:, 0] - L.PLATE_X0, L.PLATE_X1 - c[:, 0],
                                  c[:, 1] - L.PLATE_Y0, L.PLATE_Y1 - c[:, 1]])))


def corner_height(m) -> float:
    """The largest distance (m) of a bottom corner of the body from the plate's top, up or down."""
    return float(np.max(np.abs(body_corners(m)[:, 2] - PLATE_TOP_W)))


def on_plate(m) -> bool:
    """The end-state rule: every bottom corner inside the plate's outline and within 2 mm of its top."""
    return bool(plate_margin(m) >= 0.0 and corner_height(m) <= L.PLATE_Z_TOL)


def lowest_corner(m) -> float:
    """How far (m) the body's lowest bottom corner is above the plate's top (negative: below it)."""
    return float(np.min(body_corners(m)[:, 2]) - PLATE_TOP_W)


def over_plate(m) -> bool:
    """The centre of the body's underside (the motor's origin) over the plate's outline."""
    o = motor_pose(m)[1]
    return bool(L.PLATE_X0 <= o[0] <= L.PLATE_X1 and L.PLATE_Y0 <= o[1] <= L.PLATE_Y1)


def plate_breach(m) -> str | None:
    """The whole-trial plate rule for one step: None while the motor rests on its plate (its lowest bottom corner within
    2 mm of the plate's top, the centre of its underside over the plate, up axis within 10 deg of vertical), else why
    not."""
    low = lowest_corner(m)
    if low > L.PLATE_CONTACT_TOL:
        return "lifted off the plate"
    if low < -L.PLATE_CONTACT_TOL:
        return "below the plate's top (off an edge or sunk)"
    if not over_plate(m):
        return "off the plate's outline"
    if tilt_deg(m) > L.TIP_LIMIT_DEG:
        return "tipped up"
    return None


def line_breach(m) -> str | None:
    """The whole-trial line rule for one step: None while the motor's origin is within 10 mm of y 0.140."""
    return None if abs(y_offset(m)) <= L.Y_TOL else "off its line (centre distance changed)"


def y_offset(m) -> float:
    """The motor's origin from its line, y 0.140 (m)."""
    return float(motor_pose(m)[1][1] - L.MOTOR_Y)


def tier(m_deg: float) -> float:
    """The score for a total misalignment of ``m_deg`` degrees with every gate met."""
    if m_deg <= L.TIER_FULL_DEG:
        return 1.0
    if m_deg <= L.TIER_HALF_DEG:
        return 0.5
    if m_deg <= L.TIER_QUARTER_DEG:
        return 0.25
    return 0.0


# -- the straightedge -----------------------------------------------------------------------------------------------
def bar_axes(bar) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, unit normal of its broad face that faces +x, unit long axis) of the straightedge."""
    r = rotation(bar.quat_xyzw)
    n = r[:, 0] if r[0, 0] >= 0.0 else -r[:, 0]
    return np.asarray(bar.pos, dtype=float), n, r[:, 1]


def bar_face_x(bar, y: float) -> float | None:
    """Where the straightedge's broad face toward +x crosses the line through y, at the straightedge's mid-height
    (world x, m); None when that y is past its ends or it lies along x."""
    c, n, t = bar_axes(bar)
    if abs(t[1]) < 0.2:
        return None
    s = (y - c[1] - L.BAR_HALF[0] * n[1]) / t[1]
    if abs(s) > L.BAR_HALF[1]:
        return None
    return float(c[0] + L.BAR_HALF[0] * n[0] + s * t[0])


def bar_yaw_deg(bar) -> float:
    """The straightedge's long axis from world y seen from above (degrees, folded: the bar is the same either way
    round)."""
    r = rotation(bar.quat_xyzw)
    a = math.degrees(math.atan2(-r[0, 1], r[1, 1]))
    return float(-((-a + 90.0) % 180.0 - 90.0))


def bar_pitch_deg(bar) -> float:
    """How far the straightedge's top edge leans toward +x (degrees; negative: its bottom edge leads)."""
    r = rotation(bar.quat_xyzw)
    up = r[:, 2] if r[2, 2] >= 0.0 else -r[:, 2]
    return float(math.degrees(math.atan2(up[0], up[2])))


def bar_upright(bar, tol_deg: float) -> bool:
    """Standing on one of its 20 mm edges (its 40 mm axis within ``tol_deg`` of vertical)."""
    return bool(abs(rotation(bar.quat_xyzw)[2, 2]) >= math.cos(math.radians(tol_deg)))


# -- the arms ---------------------------------------------------------------------------------------------------------
def point_box_distance(p, low, high) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p, lo, hi = (np.asarray(v, dtype=float) for v in (p, low, high))
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def hand_clearance(arm, s) -> float:
    """The nearer of a gripper's TCP and pad-tip point to an object's world-aligned box (m)."""
    return min(point_box_distance(arm.tcp, s.low, s.high), point_box_distance(arm.tips, s.low, s.high))


def clearances(arms, motor, bar) -> dict:
    return {f"{side}_{name}": hand_clearance(arms[side], s)
            for side in ARMS for name, s in (("motor", motor), ("bar", bar))}


def hands_clear(arms, motor, bar) -> bool:
    """One step of 'hands off', failing closed: both arms readable, both objects' states valid and every TCP and pad-tip
    point at least 60 mm from the motor's and the straightedge's boxes. Unreadable arms are never hands off."""
    if not (arms_valid(arms) and valid(motor) and valid(bar)):
        return False
    try:
        return bool(all(v >= L.HANDS_OFF for v in clearances(arms, motor, bar).values()))
    except (AttributeError, TypeError, ValueError):  # a distance that cannot be computed is not clear
        return False


# -- the trial ------------------------------------------------------------------------------------------------------
def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


def _r(v, nd: int = 3) -> float | None:
    return round(float(v), nd) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """What the scorer has seen of this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.window: deque = deque(maxlen=L.STILL_STEPS + 1)  # (F, heading deg) for the last 0.5 s
        self.hands: deque = deque(maxlen=L.STILL_STEPS + 1)  # hands off on each of the last 0.5 s of steps
        self.plate_breach = None  # the first step that broke the whole-trial plate rule: {step, why, ...}, latched
        self.line_breach = None  # the first step that broke the whole-trial line rule, latched
        self.max_lowest = -math.inf  # the highest the lowest bottom corner rose above the plate (diagnostic)
        self.max_dy = 0.0  # the largest |y offset| seen (diagnostic)
        self.prev = None  # (motor origin, straightedge centre) at the previous valid step
        self.bad_steps = 0  # control steps with a non-finite state (ignored)
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.min_m = math.inf  # the smallest M seen (diagnostic)
        self.first_full_step = None  # the first step that scored 1.0
        self.last: Result = Result(0.0, False, {})

    # -- rest ----------------------------------------------------------------------------------------------------
    def motion(self) -> tuple[float, float]:
        """The face centre's path (m) and the heading's summed turn (degrees) over the last 0.5 s; infinite until
        the window is full."""
        if len(self.window) < L.STILL_STEPS + 1:
            return math.inf, math.inf
        pts = list(self.window)
        path = sum(float(np.linalg.norm(b[0] - a[0])) for a, b in zip(pts, pts[1:]))
        turn = sum(abs(wrap_deg(b[1] - a[1])) for a, b in zip(pts, pts[1:]))
        return path, turn

    def resting(self) -> bool:
        path, turn = self.motion()
        return bool(path < L.STILL_PATH and turn < L.STILL_TURN_DEG)

    # -- reset fallback ------------------------------------------------------------------------------------------
    @staticmethod
    def at_start(motor, bar) -> bool:
        """Every loose object back at its starting place: the motor upright on its plate within 5 mm of its start and
        turned within 2 deg of its starting 3 deg; the straightedge on its edge within 5 mm of its start, along y
        within 2 deg (the layout has no jitter; a solved motor, 12 mm from its start, never passes)."""
        _, o = motor_pose(motor)
        m_ok = (math.hypot(o[0] - L.MOTOR_START[0], o[1] - L.MOTOR_START[1]) <= L.START_TOL
                and abs(wrap_deg(heading_deg(motor) - L.MOTOR_YAW0_DEG)) <= L.START_YAW_TOL_DEG
                and tilt_deg(motor) <= L.START_YAW_TOL_DEG and abs(o[2] - PLATE_TOP_W) <= 0.005)
        c = np.asarray(bar.pos, dtype=float)
        b_ok = (math.hypot(c[0] - L.BAR_START[0], c[1] - L.BAR_START[1]) <= L.START_TOL
                and abs(bar_yaw_deg(bar)) <= L.START_YAW_TOL_DEG and bar_upright(bar, L.START_YAW_TOL_DEG))
        return bool(m_ok and b_ok)

    def _watch(self, motor) -> None:
        """The whole-trial rules for this step: the first breach of each is latched (until a reset)."""
        if not valid(motor):  # the motor's state cannot be read: neither rule can be verified (fail closed)
            why = "motor state unreadable"
            if self.plate_breach is None:
                self.plate_breach = {"step": self.steps, "why": why}
            if self.line_breach is None:
                self.line_breach = {"step": self.steps, "why": why}
            return
        low, dy = lowest_corner(motor), y_offset(motor)
        self.max_lowest = max(self.max_lowest, low)
        self.max_dy = max(self.max_dy, abs(dy))
        why = plate_breach(motor)
        if why is not None and self.plate_breach is None:
            self.plate_breach = {"step": self.steps, "why": why, "lowest_corner_mm": _mm(low),
                                 "tilt_deg": _r(tilt_deg(motor), 2)}
        why = line_breach(motor)
        if why is not None and self.line_breach is None:
            self.line_breach = {"step": self.steps, "why": why, "y_off_mm": _mm(dy)}

    def _teleported(self, o, c) -> bool:
        """An object moved RESET_JUMP or more in one control step (the simulator put it somewhere)."""
        po, pc = self.prev
        return bool(np.linalg.norm(o - po) >= L.RESET_JUMP or np.linalg.norm(c - pc) >= L.RESET_JUMP)

    # -- one step --------------------------------------------------------------------------------------------------
    def update(self, motor, bar, arms=None) -> Result:
        m_ok, b_ok = valid(motor), valid(bar)
        if m_ok and b_ok:
            _, o = motor_pose(motor)
            c = np.asarray(bar.pos, dtype=float)
            if self.prev is not None and self._teleported(o, c) and self.at_start(motor, bar):
                self.reset()  # every loose object jumped back to its start: a new episode (rigstate's and mechkit's
                #               reset hooks normally do this first; this is the fallback)
            self.prev = (o.copy(), c.copy())
        self.steps += 1
        self._watch(motor)  # the whole-trial rules, on every step (an unreadable motor breaks both)
        if not arms_valid(arms):
            self.arms_unavailable += 1
        self.hands.append(hands_clear(arms, motor, bar))
        if not (m_ok and b_ok):  # a blow-up: score 0 for this step; the rest window and prev stay as they were
            self.bad_steps += 1
            self.last = self.result(motor, bar, arms)
            return self.last
        self.window.append((face_centre(motor), heading_deg(motor)))
        self.last = self.result(motor, bar, arms)
        if self.last.details.get("m_deg") is not None:
            self.min_m = min(self.min_m, self.last.details["m_deg"])
        if self.last.score == 1.0 and self.first_full_step is None:
            self.first_full_step = self.steps
        return self.last

    def _breaches(self) -> dict:
        return {"stayed_on_plate": self.plate_breach is None, "stayed_on_line": self.line_breach is None,
                "plate_breach": self.plate_breach, "line_breach": self.line_breach}

    def result(self, motor, bar, arms=None) -> Result:
        if not (valid(motor) and valid(bar)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, **self._breaches()})
        e, theta = offset(motor), heading_deg(motor)
        m_deg = misalignment(e, theta)
        tilt, margin, height, dy = tilt_deg(motor), plate_margin(motor), corner_height(motor), y_offset(motor)
        path, turn = self.motion()
        have_arms = arms_valid(arms)
        clear = clearances(arms, motor, bar) if have_arms else {}
        hands_now = hands_clear(arms, motor, bar)  # this step (unreadable arms: not clear) ...
        hands_window = bool(len(self.hands) == self.hands.maxlen and all(self.hands))  # ... and the last 0.5 s
        gates = {
            "at_rest": bool(path < L.STILL_PATH and turn < L.STILL_TURN_DEG),
            "upright": tilt <= L.UPRIGHT_DEG,
            "on_plate": bool(margin >= 0.0 and height <= L.PLATE_Z_TOL),
            "on_line": abs(dy) <= L.Y_TOL,
            "hands_off": bool(hands_now and hands_window),
            "stayed_on_plate": self.plate_breach is None,
            "stayed_on_line": self.line_breach is None,
        }
        failed = [g for g in GATES if not gates[g]]
        score = 0.0 if failed else tier(m_deg)
        f = face_centre(motor)
        bar_m, bar_b = bar_face_x(bar, float(f[1])), bar_face_x(bar, L.BLOWER_Y)
        details = {
            "valid_state": True,
            "score_tier": tier(m_deg),
            "failed_gates": failed,
            "e_mm": _mm(e),
            "theta_deg": _r(theta),
            "alpha_deg": _r(alpha_deg(e)),
            "m_deg": _r(m_deg, 4),
            "min_m_deg": _r(min(self.min_m, m_deg), 4),
            "face_x_mm": _mm(f[0]),
            "face_y_mm": _mm(f[1]),
            "tilt_deg": _r(tilt, 2),
            "plate_margin_mm": _mm(margin),
            "corner_height_mm": _mm(height),
            "lowest_corner_mm": _mm(lowest_corner(motor)),
            "max_lowest_corner_mm": _mm(max(self.max_lowest, lowest_corner(motor))),
            "max_y_off_mm": _mm(max(self.max_dy, abs(dy))),
            "y_off_mm": _mm(dy),
            "path_mm": _mm(path),
            "turn_deg": _r(turn),
            "hands_off": gates["hands_off"],
            "hands_off_steps": int(sum(self.hands)),
            **self._breaches(),
            "clear_mm": {k: _mm(v) for k, v in clear.items()} if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "first_full_step": self.first_full_step,
            "bar_face_at_motor_mm": _mm(bar_m),
            "bar_face_at_blower_mm": _mm(bar_b),
            "bar_yaw_deg": _r(bar_yaw_deg(bar), 2),
            "bar_pitch_deg": _r(bar_pitch_deg(bar), 2),
            "bar_low_mm": _mm(float(np.asarray(bar.low, dtype=float)[2]) - L.TOP),
        }
        return Result(score, bool(score == 1.0), details)
