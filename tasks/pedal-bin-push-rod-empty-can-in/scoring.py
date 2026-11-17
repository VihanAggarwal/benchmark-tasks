"""The scorer, as plain functions of the bin's, the rod's, the can's and the arms' state. NumPy only; no simulator
needed.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads three mechkit AssemblyStates: the bin (base frame on the table under its axis; joints
``pedal`` and ``lid`` in rad), the push rod (base frame at the middle of its foot) and the can (base frame at its
centre). The arms are read only for the lid-touch check (rigstate's finger zones).

    F1 LATCH    latched zero: the bin tipped over 30 degrees or slid more than 100 mm (3-D) at any step
    IN BIN      the can's highest point 10 mm or more below the rim, its centre inside the 144 mm opening (top view)
                and its lowest point no lower than 4 mm under the bucket floor, at the last step
    SHUT        the lid within 5 degrees of closed on every step of the last second
    STILL       over the last second the bin and the can each moved under 5 mm and turned under 5 degrees
    BIN KEPT    the bin within 30 mm of where it started (3-D) and tilted under 5 degrees, at the last step
    PUSHED      at any step a finger zone is within 1 mm of the lid, or the lid is more than 10 degrees above the
                angle the rod holds it at and still rising (lifted by something other than the rod). A step whose
                arms cannot be read does not latch it but restarts the 1 s windows (so it cannot be held at 1.0)
    CAN HIT     at any step the can is within 1 mm of the lid

    0      F1 latched, or the can not in the bin
    0.25   in the bin, but the lid was pushed or lifted
    0.5    in the bin, lid never pushed, but not all of: shut and still, bin kept, can never hit the lid
    1.0    in the bin, never pushed, shut and still, bin kept, can never hit the lid

Possible scores: 0, 0.25, 0.5, 1.0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BIN, ROD, CAN = "pedal_bin", "push_rod", "can"
PARTS = (BIN, ROD, CAN)
_RING = 24  # points round each of the can's three circles (bottom edge, middle, top edge)
_PHI = np.linspace(0.0, 2.0 * math.pi, _RING, endpoint=False)
CAN_PTS = np.array([(L.CAN_R * math.cos(p), L.CAN_R * math.sin(p), z) for z in (-L.CAN_H / 2, 0.0, L.CAN_H / 2)
                    for p in _PHI] + [(0.0, 0.0, -L.CAN_H / 2), (0.0, 0.0, L.CAN_H / 2)])


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


@dataclass(frozen=True)
class Reading:
    """One step (m, rad). Positions in the world frame with the table top at z 0."""
    bin_pos: np.ndarray  # the bin's axis point on the table
    bin_rot: np.ndarray
    lid_deg: float  # the lid's opening
    pedal_deg: float  # the lever's turn (pedal down)
    rod_rise: float  # the rod's foot above its start, along the bin's axis
    can_pos: np.ndarray  # the can's centre
    can_rot: np.ndarray
    can_low: float
    can_high: float


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        b, rod, c = states.get(BIN), states.get(ROD), states.get(CAN)
        if b is None or rod is None or c is None:
            return None
        bo = np.asarray(b.origin, dtype=float).ravel()
        br = rotation(b.quat_xyzw)
        lid = float(b.joints["lid"])
        pedal = float(b.joints["pedal"])
        ro = np.asarray(rod.origin, dtype=float).ravel()
        co = np.asarray(c.origin, dtype=float).ravel()
        cr = rotation(c.quat_xyzw)
        low, high = float(np.asarray(c.low, dtype=float).ravel()[2]), float(np.asarray(c.high, dtype=float).ravel()[2])
        if bo.size != 3 or co.size != 3 or ro.size != 3:
            return None
        rise = float((br.T @ (ro - bo))[2]) - L.ROD_BOT
        if not (np.all(np.isfinite(bo)) and np.all(np.isfinite(br)) and np.all(np.isfinite(co))
                and np.all(np.isfinite(cr)) and math.isfinite(lid) and math.isfinite(pedal)
                and math.isfinite(rise) and math.isfinite(low) and math.isfinite(high)):
            return None
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    off = np.array([0.0, 0.0, L.TOP])
    return Reading(bo - off, br, math.degrees(lid), math.degrees(pedal), rise, co - off, cr, low - L.TOP,
                   high - L.TOP)


# -- the readings ------------------------------------------------------------------------------------------------------
BIN_START = np.array([L.BIN_X, L.BIN_Y, 0.0])
CAN_START = np.array([L.CAN_X, L.CAN_Y, L.CAN_H / 2])


def rim_z(r: Reading) -> float:
    return float(r.bin_pos[2]) + L.RIM


def plan_dist(r: Reading) -> float:
    """The can's centre from the bin's axis (top view)."""
    return float(np.hypot(r.can_pos[0] - r.bin_pos[0], r.can_pos[1] - r.bin_pos[1]))


def can_in_bin(r: Reading) -> bool:
    return bool(r.can_high <= rim_z(r) - L.IN_DEPTH + L.EPS and plan_dist(r) <= L.IN_R_SCORE + L.EPS
                and r.can_low >= float(r.bin_pos[2]) + L.FLOOR_TOP - L.FLOOR_TOL)


def bin_drift(r: Reading) -> float:
    return float(np.linalg.norm(r.bin_pos - BIN_START))


def bin_tilt(r: Reading) -> float:
    return tilt_deg(r.bin_rot)


def fell(r: Reading) -> bool:
    return bool(bin_tilt(r) > L.TIP_DEG or bin_drift(r) > L.SLID_DIST)


def bin_kept(r: Reading) -> bool:
    return bool(bin_drift(r) <= L.KEPT_DIST + L.EPS and bin_tilt(r) < L.KEPT_TILT)


def rod_lid_deg(r: Reading) -> float:
    """The lid angle (degrees) the rod's top holds the lid at."""
    return math.degrees(L.lid_from_rod(r.rod_rise))


def can_points(r: Reading) -> np.ndarray:
    """Points on the can's surface in the states' world frame (table top at L.TOP)."""
    return r.can_pos + np.array([0.0, 0.0, L.TOP]) + CAN_PTS @ r.can_rot.T


def tier(latched: bool, in_bin: bool, pushed: bool, settled: bool, kept: bool, clean: bool) -> float:
    if latched or not in_bin:
        return 0.0
    if pushed:
        return 0.25
    return 1.0 if (settled and kept and clean) else 0.5


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The latches (F1, pushed, can hit), the last second of readings (shut and still) and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "pushed": None, "can_hit": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})
        self.why: dict = {}
        self.lid_max = 0.0
        self.can_lid_min = math.inf
        self.unread = 0

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None

    @property
    def pushed(self) -> bool:
        return self.latch["pushed"] is not None

    @property
    def clean(self) -> bool:
        return self.latch["can_hit"] is None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (float(np.linalg.norm(r.bin_pos - BIN_START)) <= L.START_TOL
                and float(np.linalg.norm(r.can_pos - CAN_START)) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return (float(np.linalg.norm(now.can_pos - prev.can_pos)) >= L.RESET_JUMP
                or float(np.linalg.norm(now.bin_pos - prev.bin_pos)) >= L.RESET_JUMP)

    # -- per-step rules ------------------------------------------------------------------------------------------------
    @staticmethod
    def _lid_distance(states):
        b = states[BIN]
        return lambda pts: b.distance(pts, bodies=("lid",))

    def _fingers_touch_lid(self, states, arms) -> bool:
        """A finger zone within 1 mm of the lid. Unreadable arms or a distance that cannot be computed: touching
        (fail closed, as rigstate's rule)."""
        try:
            return bool(rigstate.keep_out_breached(arms, self._lid_distance(states), L.TOUCH))
        except Exception:  # noqa: BLE001
            return True

    def _can_lid_gap(self, states, r: Reading) -> float:
        try:
            d = np.asarray(states[BIN].distance(can_points(r), bodies=("lid",)), dtype=float)
            if d.size == 0 or not np.all(np.isfinite(d)):
                return -math.inf
            return float(d.min())
        except Exception:  # noqa: BLE001
            return -math.inf

    @staticmethod
    def _forced(r: Reading, prev: Reading | None) -> bool:
        """The lid more than 10 degrees above the angle the rod holds it at and still rising: lifted by something
        else (a hand, the can)."""
        if prev is None:
            return False
        return bool(r.lid_deg > rod_lid_deg(r) + L.FORCED_DEG and r.lid_deg > prev.lid_deg + 0.3)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        step = self.steps
        if self.latch["F1"] is None and fell(r):
            self.latch["F1"] = step
        arms_ok = rigstate.readable(arms)
        if not arms_ok:
            # Unreadable arms cannot clear the lid-touch check: the 1 s windows start again (no 1.0 can be held
            # through such steps), but one bad read does not latch "pushed" for the rest of the trial.
            self.unread += 1
            self.hist.clear()
        touch = arms_ok and self._fingers_touch_lid(states, arms)
        forced = self._forced(r, self.prev)
        if self.latch["pushed"] is None and (touch or forced):
            self.latch["pushed"] = step
            self.why["pushed"] = "finger zone within 1 mm of the lid" if touch else "lid lifted above the rod's angle"
        gap = self._can_lid_gap(states, r)
        self.can_lid_min = min(self.can_lid_min, gap)
        if self.latch["can_hit"] is None and gap <= L.CAN_TOUCH:
            self.latch["can_hit"] = step
        self.lid_max = max(self.lid_max, r.lid_deg)
        self.hist.append(r)
        self.prev = r
        self.last = self.judge(r, touch)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the bin and of the can from where each was at the
        window's start, and the highest lid angle; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        out = {"bin_move": 0.0, "bin_turn": 0.0, "can_move": 0.0, "can_turn": 0.0,
               "lid_max": max(x.lid_deg for x in h[1:])}
        for x in h[1:]:
            out["bin_move"] = max(out["bin_move"], float(np.linalg.norm(x.bin_pos - first.bin_pos)))
            out["bin_turn"] = max(out["bin_turn"], turn_deg(first.bin_rot, x.bin_rot))
            out["can_move"] = max(out["can_move"], float(np.linalg.norm(x.can_pos - first.can_pos)))
            out["can_turn"] = max(out["can_turn"], turn_deg(first.can_rot, x.can_rot))
        return out

    def judge(self, r: Reading, touch: bool) -> Result:
        in_bin = can_in_bin(r)
        kept = bin_kept(r)
        m = self.motion()
        shut = bool(m is not None and m["lid_max"] <= L.SHUT_DEG + L.EPS)
        still = bool(m is not None and max(m["bin_move"], m["can_move"]) <= L.STILL_MOVE + L.EPS
                     and max(m["bin_turn"], m["can_turn"]) <= L.STILL_TURN_DEG + L.EPS)
        score = tier(self.latched, in_bin, self.pushed, shut and still, kept, self.clean)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch), "why": dict(self.why),
            "in_bin": in_bin, "shut_1s": shut, "still_1s": still, "bin_kept": kept, "clean": self.clean,
            "pushed": self.pushed, "fingers_touch_lid_now": bool(touch), "arms_unreadable_steps": self.unread,
            "lid_deg": round(r.lid_deg, 2), "lid_max_deg": round(self.lid_max, 2),
            "pedal_deg": round(r.pedal_deg, 2), "rod_rise_mm": _mm(r.rod_rise),
            "rod_lid_deg": round(rod_lid_deg(r), 2), "can_lid_min_mm": _mm(self.can_lid_min),
            "bin_drift_mm": _mm(bin_drift(r)), "bin_tilt_deg": round(bin_tilt(r), 2),
            "can_mm": [_mm(v) for v in r.can_pos], "can_high_below_rim_mm": _mm(rim_z(r) - r.can_high),
            "can_plan_mm": _mm(plan_dist(r)),
            "motion": None if m is None else {k: (round(v, 3) if "turn" in k or k == "lid_max" else _mm(v))
                                              for k, v in m.items()},
        }
        return Result(score, bool(score == 1.0), details)
