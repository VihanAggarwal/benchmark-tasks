"""The scorer and the declared air rule, as plain functions of the bag and mask's and the arms' state. NumPy only;
no simulator needed.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads the bag and mask (a loose mechkit articulation, layout.py's mask frame: the cushion's 8
pads ``pad_0`` .. ``pad_7`` on sprung slides, the free swivel ``valve``, the free bending neck ``bag`` and the two
squeeze panels ``squeeze_a``/``squeeze_b`` on sprung slides) and the grippers (rigstate). The manikin, its chest and
the tray are fixed; the chest's rise is the air rule's output (``Lung``), which the simulator also shows by driving
the chest plate (task.py).

    SEALED      the mask's axis within 8 mm (top view) of the target point on the face, tilted 6 degrees or less, its
                cushion's lower face 0.5 mm or less over the face, every pad pressed 0.3 to 9.5 mm and the cushion's
                total spring force 2 to 10 N (over 10 N the airway is pressed shut)
    KINKED      the bag's neck bent more than 0.40 rad at the valve: no air goes through
    SQUEEZE     the two panels' travel added (m)
    RISE        the air rule (Lung): while sealed and not kinked, every further mm of squeeze raises the chest
                0.6 mm; easing 1 mm off the peak opens the valve and the lungs empty (time constant 0.3 s); unsealed
                or kinked, the chest falls back (0.2 s) and squeezing does nothing; capped at 12 mm (relief valve)
    INFLATION   the rise at 4 mm or more (dips that stay over 1 mm join), counted when it falls under 1 mm; good if
                it lasted 2.0 to 3.0 s (from the first step at 4 mm to the last, plus one step)
    OVER        latched: a rise over 9 mm
    TOO LONG    latched: an inflation at 4 mm or more for over 3.0 s
    DROPPED     latched: after the mask has been lifted 10 mm off the tray, any part of the bag and mask within 2 mm
                of the table
    EYES        latched: the bag and mask within 1 mm of an eye zone (a 7.5 mm ball round each eye's centre), or
                a gripper's finger zone within 2 mm of one
    LIFTED      latched: after the first good inflation, unsealed for more than 1.0 s at a time
    AT REST     on every step of the last 1 s: sealed, squeeze under 2 mm, every finger zone 5 mm or more from the
                bag and its panels, rise under 1 mm

    0      eyes, or no good inflation
    0.25   over or dropped
    0.5    too long
    1.0    exactly 5 inflations, all good, at rest, not lifted
    0.75   5 or more good inflations otherwise
    0.5    3 or 4 good inflations
    0.25   1 or 2 good inflations

Possible scores: 0, 0.25, 0.5, 0.75, 1.0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

HEAD, BODY, TRAY, BVM = L.HEAD, L.BODY, L.TRAY, L.BVM
PARTS = (HEAD, BODY, TRAY, BVM)
PADS = tuple(f"pad_{i}" for i in range(L.PAD_N))
SQUEEZE = ("squeeze_a", "squeeze_b")
BAG_BODIES = ("bag",) + SQUEEZE
TARGET = np.array([L.FACE_X, L.FACE_Y])
START = np.array([L.BVM_X0, L.BVM_Y0, L.TRAY_Z])
EYES = np.array(L.EYES) + np.array([0.0, 0.0, L.TOP])  # world
EPS = L.EPS


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def eye_distance(points) -> np.ndarray:
    """Distance (m) from world points to the nearer eye zone's surface (negative inside)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    d = np.linalg.norm(pts[:, None, :] - EYES[None, :, :], axis=2).min(axis=1)
    return d - L.EYE_R


@dataclass(frozen=True)
class Reading:
    """One step (m, deg). Positions with the table top at z 0."""
    origin: np.ndarray  # the mask frame's origin: the centre of the cushion's lower face
    rot: np.ndarray
    tilt_deg: float
    lateral: float  # top view, from the target point
    pads: tuple  # each pad's compression (m)
    squeeze: float  # both panels' travel added (m)
    neck: float  # the bag's neck (rad; positive: the bag's far end down)
    low: float  # the bag and mask's lowest point

    @property
    def force(self) -> float:
        """The cushion's total spring force (N)."""
        return L.PAD_K * float(sum(self.pads))

    @property
    def kinked(self) -> bool:
        return abs(self.neck) > L.NECK_KINK + EPS


def read(states) -> Reading | None:
    """A Reading, or None when the bag and mask's state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        b = states.get(BVM)
        if b is None:
            return None
        o = np.asarray(b.origin, dtype=float).ravel()
        rot = rotation(b.quat_xyzw)
        pads = tuple(float(b.joints[n]) for n in PADS)
        sq = float(b.joints[SQUEEZE[0]]) + float(b.joints[SQUEEZE[1]])
        neck = float(b.joints["bag"])
        low = float(np.asarray(b.low, dtype=float).ravel()[2])
        if o.size != 3:
            return None
        nums = (*pads, sq, neck, low)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(rot)) and all(math.isfinite(v) for v in nums)):
            return None
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    o = o - np.array([0.0, 0.0, L.TOP])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))
    lateral = float(math.hypot(o[0] - TARGET[0], o[1] - TARGET[1]))
    return Reading(o, rot, tilt, lateral, pads, sq, neck, low - L.TOP)


# -- the seal ----------------------------------------------------------------------------------------------------------
def seal_fault(r: Reading) -> str | None:
    """Why the mask is not sealed on the face, or None when it is."""
    if r.lateral > L.SEAL_LAT + EPS:
        return "off the target"
    if r.tilt_deg > L.SEAL_TILT_DEG + EPS:
        return "tilted"
    z = float(r.origin[2])
    if z > L.FACE_Z + L.SEAL_HIGH + EPS:
        return "not on the face"
    if z < L.FACE_Z - L.PAD_H - L.SEAL_HIGH - EPS:
        return "below the face"
    if min(r.pads) < L.PAD_MIN - EPS:
        return "a pad not pressed"
    if max(r.pads) > L.PAD_STOP + EPS:
        return "a pad at its stop"
    if r.force < L.SEAL_F[0] - EPS:
        return "pressed too lightly"
    if r.force > L.SEAL_F[1] + EPS:
        return "pressed too hard: airway shut"
    return None


def sealed(r: Reading) -> bool:
    return seal_fault(r) is None


# -- the air rule ------------------------------------------------------------------------------------------------------
class Lung:
    """The declared air rule (est). ``update(open_airway, squeeze, dt)`` -> the chest's rise (m).

    open_airway (sealed and not kinked): while the squeeze grows past its running peak, each further m of squeeze
    adds LUNG_GAIN m of rise (the bag's air goes into the lungs); easing EASE off the peak opens the valve: the lungs
    empty (TAU_OUT) and the bag refills, its squeeze tracking down; squeezing EASE past that trough again starts
    the next breath from there. Not open: no air goes in, the rise falls back (TAU_LEAK) and the reference follows
    the squeeze, so a squeeze held through the seal being made does not count. Capped at RISE_CAP."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rise = 0.0
        self.mode = "in"
        self.ref: float | None = None

    def update(self, open_airway: bool, squeeze: float, dt: float = L.DT) -> float:
        if not math.isfinite(squeeze):
            return self.rise
        if not open_airway:
            self.ref = squeeze
            self.mode = "in"
            self.rise *= math.exp(-dt / L.TAU_LEAK)
            return self.rise
        if self.ref is None:
            self.ref = squeeze
        if self.mode == "in":
            if squeeze > self.ref:
                self.rise = min(L.RISE_CAP, self.rise + L.LUNG_GAIN * (squeeze - self.ref))
                self.ref = squeeze
            elif squeeze < self.ref - L.EASE - EPS:
                self.mode = "out"
                self.ref = squeeze
        if self.mode == "out":
            self.rise *= math.exp(-dt / L.TAU_OUT)
            if squeeze < self.ref:
                self.ref = squeeze
            elif squeeze > self.ref + L.EASE + EPS:
                self.mode = "in"
                self.rise = min(L.RISE_CAP, self.rise + L.LUNG_GAIN * (squeeze - self.ref))
                self.ref = squeeze
        return self.rise


class Breaths:
    """Counts inflations from the rise, one step at a time."""

    def __init__(self, dt: float = L.DT) -> None:
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.start: int | None = None
        self.last: int | None = None
        self.good = 0
        self.total = 0
        self.durations: list = []
        self.too_long: int | None = None  # the step it latched
        self.first_good: int | None = None

    def length(self) -> float:
        """The inflation in progress so far (s); 0 when none."""
        return 0.0 if self.start is None else (self.last - self.start + 1) * self.dt

    def update(self, step: int, rise: float) -> None:
        if rise >= L.RISE_OK - EPS:
            if self.start is None:
                self.start = step
            self.last = step
            if self.too_long is None and self.length() > L.BREATH[1] + EPS:
                self.too_long = step
        elif rise < L.RISE_LOW and self.start is not None:
            d = self.length()
            self.durations.append(round(d, 3))
            self.total += 1
            if L.BREATH[0] - EPS <= d <= L.BREATH[1] + EPS:
                self.good += 1
                if self.first_good is None:
                    self.first_good = step
            self.start = self.last = None


def tier(eyes: bool, good: int, total: int, over: bool, dropped: bool, too_long: bool, at_rest: bool,
         lifted: bool) -> float:
    if eyes or good == 0:
        return 0.0
    if over or dropped:
        return 0.25
    if too_long:
        return 0.5
    if good >= L.N_BREATHS:
        return 1.0 if (good == L.N_BREATHS and total == L.N_BREATHS and at_rest and not lifted) else 0.75
    return 0.5 if good >= 3 else 0.25


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The air rule, the inflation count, the latches (over, too long, dropped, eyes, lifted), the last second of
    readings (at rest) and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"over": None, "dropped": None, "eyes": None, "lifted": None}
        self.why: dict = {}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})
        self.lung = Lung()
        self.breaths = Breaths(self.dt)
        self.raised: int | None = None  # the step the mask was first 10 mm off the tray
        self.unsealed_s = 0.0
        self.unread = 0
        self.max_rise = 0.0
        self.max_force = 0.0
        self.sealed_steps = 0

    @staticmethod
    def at_start(r: Reading) -> bool:
        return float(np.linalg.norm(r.origin - START)) <= L.START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The bag and mask put back on their marks with a jump no trial makes in one step (a reset)."""
        return cls.at_start(now) and float(np.linalg.norm(now.origin - prev.origin)) >= L.RESET_JUMP

    @staticmethod
    def _bag_gap(states):
        b = states[BVM]

        def gap(points):
            return np.asarray(b.distance(points, bodies=BAG_BODIES), dtype=float)

        return gap

    def _released(self, states, arms) -> bool:
        """Every finger zone 5 mm or more from the bag and its panels. Any failure: not released (fail closed)."""
        try:
            return bool(rigstate.released(arms, self._bag_gap(states), L.BAG_LET_GO))
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _eye_touch(states, arms) -> str | None:
        """The bag and mask or a gripper at an eye: which, else None. Unreadable arms are not counted here (they
        can never hold a 1.0)."""
        try:
            pts = np.asarray(EYES, dtype=float)
            d = np.asarray(states[BVM].distance(pts), dtype=float)
            if np.all(np.isfinite(d)) and float(d.min()) <= L.EYE_R + L.EYE_TOUCH + EPS:
                return "the bag and mask at an eye"
        except Exception:  # noqa: BLE001
            pass
        gap = rigstate.hands_gap(arms, eye_distance)
        if gap is not None and gap < L.EYE_HAND - EPS:
            return "a gripper at an eye"
        return None

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s window starts again
            self.hist.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            resets = self.resets
            self.reset()
            self.resets = resets + 1
            self.steps = 1
        step = self.steps
        fault = seal_fault(r)
        is_sealed = fault is None
        rise = self.lung.update(is_sealed and not r.kinked, r.squeeze, self.dt)
        self.breaths.update(step, rise)
        self.max_rise = max(self.max_rise, rise)
        if is_sealed:
            self.sealed_steps += 1
            self.max_force = max(self.max_force, r.force)
        if self.raised is None and r.origin[2] >= START[2] + L.DROP_LIFT - EPS:
            self.raised = step
        if self.latch["over"] is None and rise > L.RISE_OVER + EPS:
            self.latch["over"] = step
        if self.raised is not None and self.latch["dropped"] is None and r.low <= L.DROP_TABLE + EPS:
            self.latch["dropped"] = step
            self.why["dropped"] = "the bag and mask on the table"
        if self.latch["eyes"] is None:
            eye = self._eye_touch(states, arms)
            if eye is not None:
                self.latch["eyes"] = step
                self.why["eyes"] = eye
        if self.breaths.first_good is not None and not is_sealed:
            self.unsealed_s += self.dt
            if self.latch["lifted"] is None and self.unsealed_s > L.LIFT_OFF_S + EPS:
                self.latch["lifted"] = step
                self.why["lifted"] = fault
        else:
            self.unsealed_s = 0.0
        arms_ok = rigstate.readable(arms)
        if not arms_ok:
            self.unread += 1
            self.hist.clear()  # no 1.0 can be held through unreadable steps
        rel = arms_ok and self._released(states, arms)
        rest_now = bool(is_sealed and r.squeeze < L.REST_SQUEEZE and rel and rise < L.RISE_LOW)
        self.hist.append(rest_now)
        self.prev = r
        self.last = self.judge(r, fault, rise, rel)
        return self.last

    def at_rest(self) -> bool:
        return len(self.hist) >= self.window + 1 and all(self.hist)

    def judge(self, r: Reading, fault: str | None, rise: float, released_now: bool) -> Result:
        b = self.breaths
        rest = self.at_rest()
        score = tier(self.latch["eyes"] is not None, b.good, b.total + (1 if b.start is not None else 0),
                     self.latch["over"] is not None, self.latch["dropped"] is not None, b.too_long is not None,
                     rest, self.latch["lifted"] is not None)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch), "why": dict(self.why),
            "good": b.good, "inflations": b.total, "durations_s": list(b.durations), "too_long": b.too_long,
            "in_progress_s": round(b.length(), 3), "rise_mm": _mm(rise), "max_rise_mm": _mm(self.max_rise),
            "lung_mode": self.lung.mode, "sealed": fault is None, "seal_fault": fault, "kinked": r.kinked,
            "force_n": round(r.force, 3), "max_sealed_force_n": round(self.max_force, 3),
            "pads_mm": [_mm(v) for v in r.pads], "squeeze_mm": _mm(r.squeeze),
            "neck_rad": round(r.neck, 4), "tilt_deg": round(r.tilt_deg, 2), "lateral_mm": _mm(r.lateral),
            "origin_mm": [_mm(v) for v in r.origin], "low_mm": _mm(r.low), "raised_step": self.raised,
            "released_now": bool(released_now), "at_rest_1s": rest, "sealed_steps": self.sealed_steps,
            "arms_unreadable_steps": self.unread,
        }
        return Result(score, bool(score == 1.0), details)
