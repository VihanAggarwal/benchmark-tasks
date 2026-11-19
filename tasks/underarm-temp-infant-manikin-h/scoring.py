"""The scorer and the thermometer's declared rules, as plain functions of the parts' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), one observation per 50 ms control step. It
reads the doll's mechkit AssemblyState (its pose and the joint ``shoulder``, rad, 0 = the arm lying along the body),
the thermometer's (pose, the joint ``button``, m pressed in) and the mat's.

The device (``Unit``, the manual's rules, declared):
    SWITCH      the button in 0.3 mm or more, then released within 5 s, turns the unit on or off (held 5 s or more it
                opens the C/F setting instead and does not switch). The trial starts off.
    READY       2 s after switching on (self-test).
    READING     on and ready, CLOSED held unbroken for 11.0 s (a break restarts the count); the display then holds
                the result (Lo on a doll at room temperature) until the unit is off.
    AUTO-OFF    90 s after switching on if no reading is taken, 90 s after a reading.
The scorer:
    CLOSED      the unit on and ready, the metal tip (its front 8 mm) within 0.5 mm of the armpit sensor's disc (in
                contact: the engine's rest offset is 0), the thermometer's body within 0.5 mm of the doll's right arm
                (the arm resting on it), the shoulder at 22 deg or less; READING needs it on every step of the 11.0 s count
    SHOWN       after a reading, while on: the thermometer on the mat (lowest point within 4 mm of its top), the tip
                30 mm or more from the disc and no part within 10 mm of the doll's arm, the display facing within
                30 deg of straight up, every finger zone of both grippers 30 mm or more from it; held 2.0 s unbroken
    ARM BACK    the shoulder at 5 deg or less
    OFF         the unit switched off with its button after SHOWN, and off now
    HANDS OFF   on every step of the last 1 s every finger zone 30 mm or more from doll and thermometer
    STILL       over the last 1 s the thermometer, the doll and its hand each moved 5 mm or less
    ZERO        (latched) unsafe handling only: the doll off the mat (its back's midline point outside the mat top),
                lifted (that point more than 30 mm over the mat top) or rolled over (its up axis more than 45 deg from
                vertical); or the thermometer fell off the mat (a part 5 mm or more under the mat top). Moving or
                turning the doll on the mat is allowed (the sensor moves with it)

    1.0   a READING, SHOWN, ARM BACK, OFF, HANDS OFF and STILL (success)
    0.75  a READING, SHOWN and ARM BACK (not all of the rest)
    0.5   a READING
    0.25  CLOSED at some step
    0     otherwise, or ZERO
Fail closed: a step whose doll, thermometer or mat cannot be read scores 0 and restarts the windows; unreadable arms
are never hands off.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import parts as P
import rigstate

DOLL, THERMO, MAT = P.DOLL, P.THERMO, P.MAT


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


class Unit:
    """The thermometer's own behaviour (the V912 series manual), stepped once per control step."""

    def __init__(self, dt: float = L.DT) -> None:
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.on = False
        self.t_on = None
        self.pressed = False
        self.press_t = 0.0
        self.reading = False  # a reading taken in this on-session (the display holds it)
        self.reading_at = None
        self.closed_s = 0.0
        self.readings = 0
        self.events: list = []  # (time, "on" | "off", cause)
        self.setting_opened = False

    @property
    def ready(self) -> bool:
        return self.on and self.t_on is not None and self.t - self.t_on >= L.READY_S - L.EPS

    def _switch(self, cause: str) -> None:
        self.on = not self.on
        if self.on:
            self.t_on = self.t
            self.reading = False
            self.reading_at = None
        self.closed_s = 0.0
        self.events.append((round(self.t, 3), "on" if self.on else "off", cause))

    def update(self, travel: float, closed: bool) -> None:
        self.t += self.dt
        now = travel >= L.PRESS_M - L.EPS
        if now and not self.pressed:
            self.press_t = self.t
        if self.pressed and not now:
            if self.t - self.press_t < L.HOLD_SETTING_S - L.EPS:
                self._switch("button")
            else:
                self.setting_opened = True
        self.pressed = now
        if not self.on:
            return
        since = self.t - (self.reading_at if self.reading else self.t_on)
        if since >= L.AUTO_OFF_S - L.EPS:
            self._switch("auto")
            return
        if self.ready and not self.reading:
            if closed:
                self.closed_s += self.dt
                if self.closed_s >= L.PEAK_S - L.EPS:
                    self.reading = True
                    self.reading_at = self.t
                    self.readings += 1
            else:
                self.closed_s = 0.0

    def last_off_by_button(self) -> float | None:
        """When the unit was last switched off, if that was by its button and it is off now."""
        if self.on or not self.events or self.events[-1][2] != "button":
            return None
        return self.events[-1][0]

    def status(self) -> dict:
        return {"on": self.on, "ready": self.ready, "reading": self.reading, "closed_s": round(self.closed_s, 2),
                "readings": self.readings, "events": self.events[-4:], "setting": self.setting_opened}


@dataclass(frozen=True)
class Reading:
    doll: object
    th: object
    theta: float
    travel: float
    doll_origin: np.ndarray
    doll_rot: np.ndarray
    th_origin: np.ndarray
    hand: np.ndarray
    mat_top: float
    mat_xy: tuple


def read(states) -> Reading | None:
    try:
        doll, th, mat = states[DOLL], states[THERMO], states[MAT]
        theta = float(doll.joints[P.ARM_LINK])
        travel = float(th.joints["button"])
        o = np.asarray(doll.origin, dtype=float).ravel()
        rd = np.asarray(doll.rotation(), dtype=float)
        to = np.asarray(th.origin, dtype=float).ravel()
        mo = np.asarray(mat.origin, dtype=float).ravel()
        hand = np.asarray(doll.link_to_world(P.ARM_LINK, P.HAND_LOCAL), dtype=float)
        th.rotation()
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, RuntimeError):
        return None
    if not (math.isfinite(theta) and math.isfinite(travel) and np.all(np.isfinite(o)) and np.all(np.isfinite(rd))
            and np.all(np.isfinite(to)) and np.all(np.isfinite(mo)) and np.all(np.isfinite(hand))):
        return None
    mat_top = float(mo[2]) + L.MAT_H
    half = ((L.MAT_X[1] - L.MAT_X[0]) / 2, (L.MAT_Y[1] - L.MAT_Y[0]) / 2)
    return Reading(doll, th, theta, travel, o, rd, to, hand, mat_top,
                   ((mo[0] - half[0], mo[0] + half[0]), (mo[1] - half[1], mo[1] + half[1])))


def geometry(r: Reading) -> dict:
    pts = r.th.to_world(P.TH_POINTS)
    lowest = float(np.min(pts[:, 2]))
    centre = pts.mean(axis=0)
    up = float(np.degrees(math.acos(max(-1.0, min(1.0, float(r.th.rotation()[2, 2]))))))
    (x0, x1), (y0, y1) = r.mat_xy
    return {"tip_gap": P.tip_to_fsr(r.doll, r.th), "arm_gap": P.thermo_to_arm(r.doll, r.th), "lowest": lowest,
            "up_deg": up, "over_mat": bool(x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1)}


def contact(r: Reading, g: dict) -> bool:
    """The geometric half of CLOSED: tip on the disc, the body touching the arm, the arm down."""
    return (g["tip_gap"] <= L.TOUCH_M + L.EPS and g["arm_gap"] <= L.TOUCH_M + L.EPS
            and math.degrees(r.theta) <= L.CLOSED_MAX_DEG + L.EPS)


def tilt_deg(rot: np.ndarray) -> float:
    """How far the doll's up axis is from vertical (0: lying on its back)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


def unsafe_handling(r: Reading) -> str:
    """The doll handled unsafely: off the mat, lifted over 30 mm, or rolled over 45 deg ('' if none)."""
    (x0, x1), (y0, y1) = r.mat_xy
    if not (x0 <= r.doll_origin[0] <= x1 and y0 <= r.doll_origin[1] <= y1):
        return "the doll was taken off the mat"
    if float(r.doll_origin[2]) - r.mat_top > L.DOLL_LIFT_M + L.EPS:
        return "the doll was lifted off the mat"
    if tilt_deg(r.doll_rot) > L.DOLL_ROLL_DEG + L.EPS:
        return "the doll was rolled over"
    return ""


def turned_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def tier(zero: str, reading: bool, shown: bool, arm_back: bool, off: bool, hands_off: bool, still: bool,
         closed_ever: bool) -> float:
    if zero:
        return 0.0
    if reading and shown and arm_back:
        return 1.0 if (off and hands_off and still) else 0.75
    if reading:
        return 0.5
    return 0.25 if closed_ever else 0.0


class Tracker:
    def __init__(self) -> None:
        self.resets = 0
        self.unit = Unit()
        self.reset()

    def reset(self) -> None:
        self.unit.reset()
        self.steps = 0
        self.start = None  # (doll origin, doll rotation)
        self.closed_ever = False
        self.reading_ever = False
        self.shown_s = 0.0
        self.shown_at = None
        self.zero = ""
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.prev: Reading | None = None
        self.last = Result()

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (abs(r.theta) < math.radians(1.0) and r.travel < 0.0001
                and float(np.linalg.norm(r.th_origin - np.asarray(P.TH_ORIGIN))) < 0.005
                and float(np.linalg.norm(r.doll_origin - np.asarray(P.DOLL_ORIGIN))) < 0.005)

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Put back at the start with a jump no trial makes in one step (a reset the hooks did not report)."""
        return Tracker.at_start(now) and (float(np.linalg.norm(now.th_origin - prev.th_origin)) > 0.03
                                          or abs(now.theta - prev.theta) > math.radians(10.0))

    def _zero(self, why: str) -> None:
        if not self.zero:
            self.zero = why

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.win.clear()
            self.prev = None
            self.shown_s = 0.0
            self.unit.update(0.0, False)
            self.last = Result(0.0, False, {"unreadable": "doll, thermometer or mat", "zero": self.zero})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        if self.start is None:
            self.start = (r.doll_origin.copy(), r.doll_rot.copy())
        g = geometry(r)
        moved = float(np.linalg.norm(r.doll_origin - self.start[0]))
        turned = turned_deg(self.start[1], r.doll_rot)
        unsafe = unsafe_handling(r)
        if unsafe:
            self._zero(unsafe)
        if g["lowest"] < r.mat_top - L.OFF_MAT_M - L.EPS:
            self._zero("the thermometer fell off the mat")
        touching = contact(r, g)
        u = self.unit
        closed = touching and u.ready
        u.update(r.travel, touching)
        self.closed_ever = self.closed_ever or closed
        self.reading_ever = self.reading_ever or u.reading
        try:
            th_free = rigstate.released(arms, r.th.distance, L.RELEASE_M)
            doll_free = rigstate.released(arms, r.doll.distance, L.RELEASE_M)
        except Exception:  # noqa: BLE001 - a distance we cannot compute: not released (fail closed)
            th_free = doll_free = False
        on_mat = g["over_mat"] and g["lowest"] <= r.mat_top + L.ON_MAT_M + L.EPS
        shown_now = (u.on and u.reading and on_mat and g["tip_gap"] >= L.SHOWN_TIP_M - L.EPS
                     and g["arm_gap"] >= L.SHOWN_ARM_M - L.EPS and g["up_deg"] <= L.SHOWN_UP_DEG + L.EPS
                     and th_free)
        if self.shown_at is None:
            self.shown_s = self.shown_s + L.DT if shown_now else 0.0
            if self.shown_s >= L.SHOWN_S - L.EPS:
                self.shown_at = u.t
        off_t = u.last_off_by_button()
        off = self.shown_at is not None and off_t is not None and off_t > self.shown_at
        arm_back = math.degrees(r.theta) <= L.ARM_BACK_DEG + L.EPS
        self.win.append((th_free and doll_free, r.th_origin.copy(), r.doll_origin.copy(), r.hand.copy()))
        full = len(self.win) == L.WIN_STEPS
        hands_off = full and all(w[0] for w in self.win)
        move = None
        if full:
            move = max(max(float(np.linalg.norm(w[k] - self.win[0][k])) for w in self.win) for k in (1, 2, 3))
        still = move is not None and move <= L.STILL_M + L.EPS
        shown = self.shown_at is not None
        score = tier(self.zero, self.reading_ever, shown, arm_back, off, hands_off, still, self.closed_ever)
        details = {"shoulder_deg": round(math.degrees(r.theta), 2), "button_mm": round(r.travel * 1000, 3),
                   "tip_gap_mm": round(g["tip_gap"] * 1000, 2), "arm_gap_mm": round(g["arm_gap"] * 1000, 2),
                   "up_deg": round(g["up_deg"], 1), "on_mat": on_mat, "contact": touching, "closed": closed,
                   "closed_ever": self.closed_ever, "reading": self.reading_ever, "shown": shown,
                   "shown_s": round(self.shown_s, 2), "arm_back": arm_back, "off": off, "hands_off": hands_off,
                   "still": still, "moved_mm": None if move is None else round(move * 1000, 2),
                   "torso_moved_mm": round(moved * 1000, 2), "torso_turned_deg": round(turned, 2),
                   "doll_lift_mm": round((float(r.doll_origin[2]) - r.mat_top) * 1000, 2),
                   "doll_tilt_deg": round(tilt_deg(r.doll_rot), 2),
                   "zero": self.zero, "unit": u.status()}
        self.last = Result(score, score >= 1.0, details)
        return self.last
