"""The line's one declared rule (the starter) and the scorer, plain functions of the parts' and arms' state. NumPy only.

THE RULE (``Line``, run once per control step before the physics): the line runs from the start of the trial. Pressing
the red STOP cap 1.5 mm or more (where the XB5AL42's NC contact opens) drops the starter out: the rollers coast to rest
in 0.5 s and stay stopped until the green START cap is pressed 2.6 mm or more with STOP not pressed; they then reach
speed in 0.5 s. Speed 30 ft/min (0.152 m/s) at the roller surface, toward the discharge end. task.py turns the
commanded speed into drive targets for the eight roller hinges.

THE SCORER (``Tracker.update(states, arms, line)``, one observation per control step):

    TURNING     the commanded surface speed or the fastest roller's measured surface speed is over 0.02 m/s (a roller
                state that cannot be read counts as turning)
    IN ZONE     a gripper's pad-tip point lower than 80 mm above the roller tops (139.1 mm above the bench) and over the
                bed (between the frame's inner faces, from the end plate to the infeed end), the bed's outline widened
                by half that gripper's finger gap plus 8 mm (so a finger anywhere over the bed counts)
    LATCH (0 for the rest of the trial)
                a step that is TURNING with a gripper IN ZONE, or TURNING with the arms unreadable; or the carton off
                the table (its centre outside the table, or its lowest corner 1 cm below the bench top)
    ON BED      the carton's centre over the bed's outline (top view)
    FREED       every corner of the carton at least 10 mm upstream of the guard roof's edge, or the carton off the bed
    STOPPED     not TURNING and the starter out
    IN TRAY     every corner inside the tray's inner outline, its underside from 3 mm below to 6 mm above the
                tray floor's top (0 to 9 mm above the bench), flat within 10 degrees
    RELEASED    every finger zone of both grippers at least 3 cm from the carton (unreadable arms: not released)
    STILL       the carton's centre moved 2 mm or less over the last 20 steps

    1.0   IN TRAY and RELEASED on each of the last 20 steps, and STILL
    0.75  off the bed (not ON BED) and RELEASED
    0.5   FREED and STOPPED while ON BED; or off the bed and still held
    0.25  STOPPED, the carton ON BED (still under the guard)
    0     anything else, or latched.
Fail closed: a step whose carton state cannot be read scores 0 and restarts the 20-step window.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CARTON, CONVEYOR, STATION, TRAY = "carton", "conveyor", "station", "tray"
ROLLERS = tuple(f"r{i}" for i in range(L.N_ROLLERS))


# -- the rule: the starter ------------------------------------------------------------------------------------------
class Line:
    """The motor starter of the line, stepped once per control step with the two caps' depths (m, pressed down)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.steps = 0
        self.running = True
        self.t_change = 0.0
        self.v_change = 0.0  # the commanded speed when the starter last changed state (0: the trial starts at rest)
        self.v = 0.0
        self.stops = 0
        self.starts = 0
        self.events: list = []

    def update(self, stop_depth, start_depth) -> float:
        """One control step: read the caps, change state, return the commanded surface speed (m/s, toward -y)."""
        self.steps += 1
        self.t = self.steps * L.DT
        stop = _finite(stop_depth) and float(stop_depth) >= L.STOP_AT - L.EPS
        start = _finite(start_depth) and float(start_depth) >= L.START_AT - L.EPS
        stop_unread = not _finite(stop_depth)
        if self.running and stop:
            self.running, self.t_change, self.v_change = False, self.t, self.v
            self.stops += 1
            self.events.append(("stop", self.steps))
        elif not self.running and start and not stop and not stop_unread:
            self.running, self.t_change, self.v_change = True, self.t, self.v
            self.starts += 1
            self.events.append(("start", self.steps))
        dt = self.t - self.t_change
        if self.running:
            self.v = min(L.SPEED, self.v_change + L.SPEED * dt / L.RAMP_S)
        else:
            self.v = max(0.0, self.v_change - L.SPEED * dt / L.COAST_S)
        return self.v

    def snapshot(self) -> dict:
        return {"running": self.running, "speed": self.v, "stops": self.stops, "starts": self.starts,
                "events": list(self.events[-4:])}


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def drive_target(q: float, omega: float) -> float:
    """A roller's drive target (rad) for one control step: torque = K (target - q) - D q'. Leading the roller by
    omega DT / 2 makes the spring's torque average zero over the step at steady speed, and D omega / K cancels the
    drive's damping, so the drive acts as a velocity servo of gain D about ``omega``. Kept inside the hinge's limits."""
    t = float(q) + float(omega) * (L.DT / 2 + L.ROLLER_D_GAIN / L.ROLLER_K)
    return max(-L.ROLLER_LIMIT + 0.01, min(L.ROLLER_LIMIT - 0.01, t))


def wrap_shift(q: float) -> float:
    """The whole turn (rad) that brings a roller angle beyond +-pi back inside it, else 0."""
    q = float(q)
    if q > L.WRAP_AT:
        return -2 * math.pi
    if q < -L.WRAP_AT:
        return 2 * math.pi
    return 0.0


# -- readings ----------------------------------------------------------------------------------------------------
def measured_speed(conveyor) -> float:
    """The fastest roller's surface speed (m/s) from the conveyor's joint velocities; inf when unreadable."""
    try:
        w = [abs(float(conveyor.joint_vel[r])) for r in ROLLERS]
    except (AttributeError, KeyError, TypeError, ValueError):
        return math.inf
    if not all(math.isfinite(v) for v in w):
        return math.inf
    return max(w) * L.ROLLER_R


def in_zone(arm) -> bool:
    """A gripper's pad-tip point under the zone's top and over the bed's outline widened by its finger reach."""
    tcp, tips, opening = arm
    r = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP / 2 + L.FINGER_REACH
    x, y, z = (float(v) for v in tips)
    return (z < L.TOP + L.ZONE_TOP and L.ZONE_X[0] - r <= x <= L.ZONE_X[1] + r
            and L.ZONE_Y[0] - r <= y <= L.ZONE_Y[1] + r)


def tilt_deg(rot) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(rot)[2, 2])))))


def on_bed(centre) -> bool:
    return L.X_IN[0] <= centre[0] <= L.X_IN[1] and L.Y_END <= centre[1] <= L.Y_IN


def in_tray(corners, rot) -> bool:
    xy_ok = bool(np.all(corners[:, 0] >= L.TRAY_IN_X[0]) and np.all(corners[:, 0] <= L.TRAY_IN_X[1])
                 and np.all(corners[:, 1] >= L.TRAY_IN_Y[0]) and np.all(corners[:, 1] <= L.TRAY_IN_Y[1]))
    under = float(corners[:, 2].min()) - (L.TOP + L.TRAY_FLOOR)
    return xy_ok and -0.003 <= under <= L.IN_TRAY_Z and tilt_deg(rot) <= L.FLAT_DEG


def freed(corners) -> bool:
    return bool(np.all(corners[:, 1] >= L.GUARD_EDGE_Y + L.FREE_CLEAR - 1e-9))


def off_table(centre, corners) -> bool:
    return (not (L.TABLE_X[0] <= centre[0] <= L.TABLE_X[1] and L.TABLE_Y[0] <= centre[1] <= L.TABLE_Y[1])
            or float(corners[:, 2].min()) < L.TOP - L.OFF_TABLE_DROP)


# -- the scorer --------------------------------------------------------------------------------------------------
@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.last = Result()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.latched = ""
        self.latch_step = None
        self.first_zone_step = None
        self.last = Result()

    def _latch(self, why: str) -> None:
        if not self.latched:
            self.latched, self.latch_step = why, self.steps

    def update(self, states, arms, line: dict) -> Result:
        self.steps += 1
        v_cmd = float(line.get("speed", math.inf)) if isinstance(line, dict) else math.inf
        if not math.isfinite(v_cmd):
            v_cmd = math.inf
        running = bool(line.get("running", True)) if isinstance(line, dict) else True
        v_meas = measured_speed(states.get(CONVEYOR)) if isinstance(states, dict) else math.inf
        turning = max(v_cmd, v_meas) > L.TURNING
        read = rigstate.read_arms(arms)
        zone = None if read is None else [in_zone(a) for a in read]
        if turning and (zone is None or any(zone)):
            self._latch("arms unreadable while the rollers turned" if zone is None
                        else "a gripper in the roller zone while the rollers turned")
        if zone is not None and any(zone) and self.first_zone_step is None:
            self.first_zone_step = self.steps
        details = {"v_cmd": round(v_cmd, 4) if math.isfinite(v_cmd) else None,
                   "v_meas": round(v_meas, 4) if math.isfinite(v_meas) else None, "running": running,
                   "in_zone": zone, "latched": self.latched, "latch_step": self.latch_step}
        try:
            c = states[CARTON]
            origin = np.asarray(c.origin, dtype=float)
            rot = np.asarray(c.rotation(), dtype=float)
            corners = L.carton_corners(origin, rot)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "carton"})
            return self.last
        if not (np.all(np.isfinite(corners)) and np.all(np.isfinite(rot))):
            self.win.clear()
            self.last = Result(0.0, False, {**details, "unreadable": "carton numbers"})
            return self.last
        centre = corners.mean(axis=0)
        if off_table(centre, corners):
            self._latch("the carton left the table")
        stopped = (not running) and not turning
        bed = on_bed(centre)
        free = (not bed) or freed(corners)
        tray = in_tray(corners, rot)
        rel = rigstate.released(arms, c.distance, L.RELEASE) if read is not None else False
        self.win.append((tray and rel, centre))
        full = still = False
        if len(self.win) == L.WIN_STEPS:
            full = all(w[0] for w in self.win)
            c0 = self.win[0][1]
            still = max(float(np.linalg.norm(w[1] - c0)) for w in self.win) <= L.STILL_MOVE + 1e-12
        if self.latched:
            score = 0.0
        elif full and still:
            score = 1.0
        elif not bed and rel:
            score = 0.75
        elif (free and stopped and bed) or (not bed):
            score = 0.5
        elif stopped and bed:
            score = 0.25
        else:
            score = 0.0
        details.update({"carton_mm": [round(float(v) * 1000, 1) for v in (centre - np.array([0, 0, L.TOP]))],
                        "lead_y_mm": round(float(corners[:, 1].min()) * 1000, 1), "tilt_deg": round(tilt_deg(rot), 1),
                        "on_bed": bed, "freed": free, "stopped": stopped, "in_tray": tray, "released": rel,
                        "still": still})
        self.last = Result(score, score >= 1.0, details)
        return self.last
