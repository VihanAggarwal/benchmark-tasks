"""The scorer, as plain functions of the carriage's and the plunger's states and the arms' state. NumPy only; no
simulator needed.

The rule is about the whole trial, not only its end: the carriage may move only while the enabling plunger is held
upright in its middle band, never faster than 5 cm/s, and it must stop on the red, then the blue, then the home mark. So
the scorer keeps a little state (how far the carriage has moved while not enabled, its fastest speed, which stops
it has made, in order) and is fed one observation per control step by ``task.success()``.

    Tracker.update(carriage, plunger, arms) -> Result(score, success, details)

``carriage`` and ``plunger`` are anything with ``pos`` (bounding-box centre), ``quat_xyzw``, ``lin_vel``,
``ang_vel``, ``low`` and ``high`` (world-aligned bounding box), as ``yam_arena.task_file.ObjectState`` has.
``arms`` is ``rigstate.arms()``: ``{"left": Arm, "right": Arm}`` with ``tcp``, ``tips`` (the point between the
finger tips) and ``opening`` (0 closed .. 1 open), or ``None`` when the arms could not be read; then 'held' is
taken as met (the plunger's height and tilt alone decide whether the step is enabled: an upright plunger cannot
stay up in its band unless something holds it, since it must tilt 5.4 degrees or more to jam across its sleeve)
and the step is counted in ``arms_unavailable``. An object state with a
non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it
was.

Distances along the stage are horizontal (x, y) distances of the carriage's centre. Travel is followed in steps
of 0.2 mm, as a ruler would read it: a move counts once the carriage is 0.2 mm from where travel was last counted,
so physics jitter of a resting carriage adds nothing. Speed is the carriage's horizontal displacement over the
last two control steps (0.1 s at 20 Hz), from its positions, as a 30 fps video would give it.

The carriage is a plain cube, so any face may be down: its tilt is measured from whichever body axis is nearest
to vertical. The plunger is the same either end up: its tilt is the angle of its axis from the vertical.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L

_KEYS = ("pos", "quat_xyzw", "lin_vel", "ang_vel", "low", "high")


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


def point_box_distance(p, lo, hi) -> float:
    """Distance from a point to a world-aligned box (0 inside it)."""
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo, dtype=float) - p, p - np.asarray(hi, dtype=float)))
    return float(np.linalg.norm(d))


def off_table(s) -> bool:
    return bool(float(s.low[2]) < L.TOP - L.OFF_TABLE_DROP)


# -- the carriage -------------------------------------------------------------------------------------------------
def carriage_tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and whichever of the cube's three axes is nearest to it (degrees, 0..54.7): any
    face down is the same cube."""
    col = np.abs(rotation(quat_xyzw)[2, :])
    return float(np.degrees(np.arccos(np.clip(col.max(), 0.0, 1.0))))


def carriage_lift(c) -> float:
    """How far the carriage's lowest point is above the plate's top (m; about 0 when it rests on the plate)."""
    return float(c.low[2]) - L.PLATE_TOP_Z


def off_target(c, xy) -> float:
    """Horizontal distance from the carriage's centre to a mark (m)."""
    return float(math.hypot(float(c.pos[0]) - xy[0], float(c.pos[1]) - xy[1]))


# -- the plunger --------------------------------------------------------------------------------------------------
def plunger_tilt_deg(quat_xyzw) -> float:
    """Angle between the vertical and the plunger's axis (degrees, 0..90; either end up)."""
    zz = rotation(quat_xyzw)[2, 2]
    return float(np.degrees(np.arccos(np.clip(abs(zz), 0.0, 1.0))))


def plunger_height(p) -> float:
    """h: the plunger's lowest point above the table top (m; 0 when it stands on the table)."""
    return float(p.low[2]) - L.TOP


def grip_point(p) -> np.ndarray:
    """On the plunger's axis, 20 mm below its upper end."""
    axis = rotation(p.quat_xyzw)[:, 2]
    if axis[2] < 0:
        axis = -axis
    return np.asarray(p.pos, dtype=float) + axis * (L.PLUNGER_HEIGHT / 2 - L.PLUNGER_GRIP_BELOW_TOP)


def holder(p, arms) -> str | None:
    """Which gripper holds the plunger: its TCP within 25 mm of the grip point and its opening under 0.5."""
    g = grip_point(p)
    for side in ("left", "right"):
        a = arms[side]
        if float(a.opening) < L.HELD_OPENING and float(np.linalg.norm(np.asarray(a.tcp, dtype=float) - g)) <= \
                L.HELD_RADIUS:
            return side
    return None


def in_band(p) -> bool:
    """The plunger lifted into its middle band: its lowest point 15 to 30 mm above the table top."""
    return bool(L.ENABLE_LO <= plunger_height(p) <= L.ENABLE_HI)


def upright(p) -> bool:
    """The plunger tilted under 4 degrees: it would need 5.4 or more to jam across its sleeve and stay up alone."""
    return bool(plunger_tilt_deg(p.quat_xyzw) < L.ENABLE_TILT_MAX)


def plunger_in_sleeve(p) -> bool:
    """The plunger's centre over the sleeve's 30 x 30 mm opening and its lowest point below the sleeve's top."""
    return bool(abs(float(p.pos[0]) - L.SLEEVE_X) <= L.SLEEVE_HALF
                and abs(float(p.pos[1]) - L.SLEEVE_Y) <= L.SLEEVE_HALF
                and plunger_height(p) <= L.SLEEVE_HEIGHT)


def plunger_down(p) -> bool:
    return bool(plunger_height(p) < L.PLUNGER_DOWN)


# -- the grippers at the end ---------------------------------------------------------------------------------------
def gripper_clearance(arm, objs) -> float:
    """The smallest distance from the gripper's tip point or TCP to any of the objects' boxes (m)."""
    return min(point_box_distance(pt, o.low, o.high) for pt in (arm.tips, arm.tcp) for o in objs)


def grippers_done(arms, carriage, plunger) -> bool:
    """Both grippers open (opening 0.9 or more) and at least 30 mm from the carriage and the plunger."""
    return bool(all(float(arms[s].opening) >= L.GRIPPER_OPEN and gripper_clearance(arms[s], (carriage, plunger))
                    >= L.AWAY for s in ("left", "right")))


def _mm(v) -> float | None:
    return round(float(v) * 1000, 1) if v is not None and math.isfinite(v) else None


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


BREACHES = {
    "moved_while_not_enabled": "the carriage travelled over 2 mm in all while the plunger was not held upright in "
                               "its band",
    "too_fast": "the carriage moved faster than 50 mm/s",
    "carriage_lifted": "the carriage's lowest point was more than 5 mm above the plate",
    "carriage_tipped": "the carriage tilted more than 10 degrees",
    "plunger_tipped": "the plunger tilted more than 15 degrees",
    "plunger_out_of_sleeve": "the plunger left its sleeve",
    "off_table": "the carriage or the plunger left the table",
}
ORDER = ("red", "blue", "home")  # the stops, in the order they must be made


class Tracker:
    """What has happened so far in this trial."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.breaches: dict[str, int] = {}  # breach -> the control step it first happened
        self.stage = 0  # stops made in order: 0 none, 1 red, 2 red then blue, 3 red, blue, then home
        self.stop_steps: dict[str, int | None] = {k: None for k in ORDER}  # step each stop in the order completed
        self.streak = {k: 0 for k in ORDER}  # consecutive steps stopped at each mark
        self.history: deque = deque(maxlen=L.SPEED_STEPS + 1)  # the carriage's recent (x, y)
        self.anchor = None  # where travel was last counted
        self.idle_travel = 0.0  # m, travel while not enabled
        self.enabled_travel = 0.0  # m, travel while enabled (whole trial)
        self.stretch_steps = 0  # the current continuous enabled stretch
        self.stretch_start = None  # the carriage's (x, y) when it began
        self.stretch_reach = 0.0  # m, the farthest the carriage's centre has been from there during it
        self.guided_at: int | None = None  # step the 0.25 was earned (20 mm from where a 3 s stretch began)
        self.enabled = False
        self.held_by: str | None = None
        self.enabled_steps = 0
        self.speed = 0.0
        self.max_speed = 0.0
        self.max_lift = -math.inf
        self.max_carriage_tilt = 0.0
        self.max_plunger_tilt = 0.0
        self.max_h = -math.inf
        self.arms_unavailable = 0  # control steps on which rigstate could not read the arms
        self.bad_steps = 0  # control steps with a non-finite object state (ignored)
        self.last: Result = Result(0.0, False, {})
        self.prev = None  # [(centre, velocity) of the carriage, of the plunger] at the previous valid step

    @property
    def breach(self) -> bool:
        return bool(self.breaches)

    @staticmethod
    def at_start(carriage, plunger) -> bool:
        """The starting layout: the carriage flat on the plate at home, the plunger down in its sleeve, both still."""
        c_ok = (off_target(carriage, L.HOME) < L.START_TOL and abs(carriage_lift(carriage)) < 0.002
                and carriage_tilt_deg(carriage.quat_xyzw) < 2.0)
        p_ok = (plunger_in_sleeve(plunger) and plunger_down(plunger) and plunger_tilt_deg(plunger.quat_xyzw) < 6.0)
        return bool(c_ok and p_ok and is_resting(carriage) and is_resting(plunger))

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

    def _stop_check(self, carriage) -> None:
        """Count consecutive enabled, slow steps within 3 mm of each mark; a 1 s run is a stop, and it advances
        the order if it is the next mark due. (The marks are at least 50 mm apart, so a stop at the next mark
        always begins after the previous stop was made.)"""
        for i, name in enumerate(ORDER):
            ok = (self.enabled and off_target(carriage, L.TARGETS[name]) <= L.STOP_TOL
                  and self.speed < L.STOP_SPEED)
            if not ok:
                self.streak[name] = 0
                continue
            self.streak[name] += 1
            if self.stage == i and self.streak[name] * self.dt >= L.STOP_S - 1e-9:
                self.stage = i + 1
                self.stop_steps[name] = self.steps

    def update(self, carriage, plunger, arms=None) -> Result:
        if not (valid(carriage) and valid(plunger)):  # a blow-up: score 0 for this step, keep everything as it was
            self.steps += 1
            self.bad_steps += 1
            self.last = self.result(carriage, plunger, arms)
            return self.last
        now = [(np.array(s.pos, dtype=float), np.array(s.lin_vel, dtype=float)) for s in (carriage, plunger)]
        if self.prev is not None and self._teleported(now) and self.at_start(carriage, plunger):
            self.reset()  # the simulator put everything back on its marks: a new episode (rigstate's on_reset
            #               hook normally does this first; this is the fallback)
        self.prev = now
        self.steps += 1

        have_arms = arms_valid(arms)
        if not have_arms:
            self.arms_unavailable += 1
        self.held_by = holder(plunger, arms) if have_arms else None
        held = self.held_by is not None or not have_arms
        self.enabled = bool(in_band(plunger) and upright(plunger) and held)
        self.enabled_steps += self.enabled

        xy = np.array(carriage.pos[:2], dtype=float)
        self.history.append(xy)
        n = len(self.history) - 1
        self.speed = float(np.linalg.norm(self.history[-1] - self.history[0])) / (n * self.dt) if n else 0.0
        self.max_speed = max(self.max_speed, self.speed)

        if self.anchor is None:
            self.anchor = xy
        d = float(np.linalg.norm(xy - self.anchor))
        if d >= L.RULER:
            if self.enabled:
                self.enabled_travel += d
            else:
                self.idle_travel += d
            self.anchor = xy
        if self.enabled:  # 0.25: how far from where it began one enabled stretch took the carriage (not a jiggle)
            if self.stretch_start is None:
                self.stretch_start = xy
            self.stretch_steps += 1
            self.stretch_reach = max(self.stretch_reach, float(np.linalg.norm(xy - self.stretch_start)))
            if (self.guided_at is None and self.stretch_steps * self.dt >= L.GUIDE_S - 1e-9
                    and self.stretch_reach >= L.GUIDE_MIN - 1e-9):
                self.guided_at = self.steps
        else:
            self.stretch_steps, self.stretch_start, self.stretch_reach = 0, None, 0.0

        lift = carriage_lift(carriage)
        ct, pt = carriage_tilt_deg(carriage.quat_xyzw), plunger_tilt_deg(plunger.quat_xyzw)
        self.max_lift = max(self.max_lift, lift)
        self.max_carriage_tilt = max(self.max_carriage_tilt, ct)
        self.max_plunger_tilt = max(self.max_plunger_tilt, pt)
        self.max_h = max(self.max_h, plunger_height(plunger))
        if self.idle_travel > L.IDLE_TRAVEL_MAX:
            self._flag("moved_while_not_enabled")
        if self.speed > L.SPEED_CAP:
            self._flag("too_fast")
        if lift > L.LIFT_MAX:
            self._flag("carriage_lifted")
        if ct > L.CARRIAGE_TILT_MAX:
            self._flag("carriage_tipped")
        if pt > L.PLUNGER_TILT_MAX:
            self._flag("plunger_tipped")
        if not plunger_in_sleeve(plunger):
            self._flag("plunger_out_of_sleeve")
        if off_table(carriage) or off_table(plunger):
            self._flag("off_table")

        self._stop_check(carriage)
        self.last = self.result(carriage, plunger, arms)
        return self.last

    def result(self, carriage, plunger, arms=None) -> Result:
        if not (valid(carriage) and valid(plunger)):
            return Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                       "breaches": sorted(self.breaches)})
        have_arms = arms_valid(arms)
        done = grippers_done(arms, carriage, plunger) if have_arms else True  # unreadable: taken as met, counted
        at_home = off_target(carriage, L.HOME) <= L.HOME_END_TOL
        down = plunger_down(plunger)
        if self.breach:
            score = 0.0
        elif self.stage >= 3 and at_home and down and done:
            score = 1.0
        elif self.stage >= 2:
            score = 0.75
        elif self.stage >= 1:
            score = 0.5
        elif self.guided_at is not None:
            score = 0.25
        else:
            score = 0.0
        success = bool(score == 1.0 and is_resting(carriage) and is_resting(plunger))
        details = {
            "valid_state": True,
            "breaches": sorted(self.breaches),
            "first_breach_step": min(self.breaches.values()) if self.breaches else None,
            "stage": self.stage,
            "stop_steps": dict(self.stop_steps),
            "enabled": self.enabled,
            "held_by": self.held_by,
            "enabled_steps": self.enabled_steps,
            "plunger_h_mm": _mm(plunger_height(plunger)),
            "max_plunger_h_mm": _mm(self.max_h),
            "speed_mm_s": _mm(self.speed),
            "max_speed_mm_s": _mm(self.max_speed),
            "idle_travel_mm": _mm(self.idle_travel),
            "enabled_travel_mm": _mm(self.enabled_travel),
            "stretch_s": round(self.stretch_steps * self.dt, 2),
            "stretch_reach_mm": _mm(self.stretch_reach),
            "guided_at": self.guided_at,
            "off_red_mm": _mm(off_target(carriage, L.RED)),
            "off_blue_mm": _mm(off_target(carriage, L.BLUE)),
            "off_home_mm": _mm(off_target(carriage, L.HOME)),
            "carriage_lift_mm": _mm(carriage_lift(carriage)),
            "max_lift_mm": _mm(self.max_lift),
            "carriage_tilt_deg": round(carriage_tilt_deg(carriage.quat_xyzw), 2),
            "max_carriage_tilt_deg": round(self.max_carriage_tilt, 2),
            "plunger_tilt_deg": round(plunger_tilt_deg(plunger.quat_xyzw), 2),
            "plunger_upright": upright(plunger),
            "max_plunger_tilt_deg": round(self.max_plunger_tilt, 2),
            "plunger_in_sleeve": plunger_in_sleeve(plunger),
            "plunger_down": down,
            "carriage_at_home": at_home,
            "opening": ({s: round(float(arms[s].opening), 3) for s in ("left", "right")} if have_arms else None),
            "gripper_clearance_mm": ({s: _mm(gripper_clearance(arms[s], (carriage, plunger)))
                                      for s in ("left", "right")} if have_arms else None),
            "grippers_done": done if have_arms else None,
            "arms_available": have_arms,
            "arms_unavailable": self.arms_unavailable,
            "bad_steps": self.bad_steps,
            "carriage_resting": is_resting(carriage),
            "plunger_resting": is_resting(plunger),
        }
        return Result(score, success, details)
