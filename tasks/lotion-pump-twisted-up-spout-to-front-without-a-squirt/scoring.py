"""The scorer, as plain functions of the bottle's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the bottle (base frame at the centre of the base's underside; slider link "lift" = the
pump head's travel down from its free height, hinge link "twist" = the head's turn, anticlockwise from above, the
spout pointing -y at 0). Every reading is a pose or a joint position; the arms are read only for the let-go check.

    FELL        latched: the base more than 50 mm below its start height (off the vanity)
    TIPPED      latched: the bottle's axis more than 45 degrees off vertical
    UNLOCKED    the head turned 80 degrees or more at some moment (latched: the lugs cleared) and, now, the head
                18 mm or more above its locked height (lift <= 5 mm)
    STROKE      latched: once unlocked, the head more than 4 mm below the highest it has been (a stroke)
    SPOUT       the spout's bearing within 15 degrees of -x (toward the user), seen from above
    STANDING    axis within 8 degrees of vertical and base centre within 30 mm of its start (top view)
    TURNED      latched: the head turned 30 degrees or more toward unlocking at some moment
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from every part of the bottle
    STILL       over the last 1 s the base moved under 5 mm and turned under 5 degrees, the head turned under
                5 degrees and rose or sank under 5 mm

    0      FELL, TIPPED or STROKE ever, or not released and still (a held bottle earns nothing), or not STANDING
    1.0    UNLOCKED and SPOUT
    0.5    UNLOCKED, spout elsewhere
    0.25   not UNLOCKED, TURNED
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PART = "bottle"
_SPOUT0 = np.array([0.0, -1.0, 0.0])
LIFT_UNLOCKED = L.TRAVEL - L.UNLOCK_UP  # the lift reading at 18 mm up


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


@dataclass(frozen=True)
class Reading:
    """One step (m, rad). Positions in the world frame with the table top at z 0."""
    pos: np.ndarray  # base frame origin
    rot: np.ndarray  # base rotation
    twist: float  # the head's turn
    lift: float  # the head's travel down from free
    spout: np.ndarray  # the spout's direction in the world


def read(states) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        s = states.get(PART)
        if s is None:
            return None
        o = np.asarray(s.origin, dtype=float).ravel()
        r = rotation(s.quat_xyzw)
        tw, lf = float(s.joints.get("twist")), float(s.joints.get("lift"))
        rt = np.asarray(s.link_pose("twist")[0], dtype=float)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and math.isfinite(tw)
                               and math.isfinite(lf) and np.all(np.isfinite(rt))):
            return None
        return Reading(o - np.array([0.0, 0.0, L.TOP]), r, tw, lf, rt @ _SPOUT0)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


# -- the readings ------------------------------------------------------------------------------------------------------
def spout_bearing(r: Reading) -> float | None:
    """The spout's bearing seen from above (degrees, 0 = +x), None when it points up or down."""
    if math.hypot(float(r.spout[0]), float(r.spout[1])) < 1e-6:
        return None
    return math.degrees(math.atan2(float(r.spout[1]), float(r.spout[0])))


def spout_off_deg(r: Reading) -> float:
    """The angle (degrees) between the spout's bearing and -x (toward the user)."""
    b = spout_bearing(r)
    return 180.0 if b is None else L.off_front_deg(b)


def tilt_of(r: Reading) -> float:
    return tilt_deg(r.rot)


def fell(r: Reading) -> bool:
    return float(r.pos[2]) < L.BASE_Z - L.FALL_DROP


def tipped(r: Reading) -> bool:
    return tilt_of(r) > L.TIP_DEG + L.EPS


def standing(r: Reading) -> bool:
    return bool(tilt_of(r) <= L.STAND_TILT_DEG + L.EPS
                and math.hypot(float(r.pos[0] - L.BX), float(r.pos[1] - L.BY)) <= L.STAND_XY + L.EPS)


def head_up(r: Reading) -> bool:
    return r.lift <= LIFT_UNLOCKED + L.EPS


def tier(latched: bool, settled: bool, stand: bool, unlocked: bool, spout_ok: bool, turned: bool) -> float:
    if latched or not settled or not stand:
        return 0.0
    if unlocked:
        return 1.0 if spout_ok else 0.5
    return 0.25 if turned else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


class Tracker:
    """The latches (fell, tipped, stroke, lugs cleared, turned 30), the last 1 s of readings (stillness) and of the
    let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"FELL": None, "TIPPED": None, "STROKE": None, "LUGS": None, "TURNED": None}
        self.top_lift = None  # the smallest lift (the head's highest) since the lugs cleared
        self.max_twist = 0.0
        self.max_depth = 0.0  # the deepest the head was pushed below its highest, since the lugs cleared
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(self.latch[k] is not None for k in ("FELL", "TIPPED", "STROKE"))

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (math.hypot(float(r.pos[0] - L.BX), float(r.pos[1] - L.BY)) <= L.START_TOL
                and abs(float(r.pos[2]) - L.BASE_Z) <= L.START_TOL and abs(r.twist) <= 0.05
                and r.lift >= L.TRAVEL - 0.002)

    def teleported(self, prev: Reading, now: Reading) -> bool:
        """The bottle back on its mark with the head locked again by a jump no trial makes in one step (a reset): the
        base moved 50 mm, the head 15 mm further down, or the head's turn changed by more than a radian."""
        if not self.at_start(now):
            return False
        return bool(float(np.linalg.norm(now.pos - prev.pos)) >= L.RESET_JUMP
                    or now.lift - prev.lift >= L.RESET_LIFT or abs(now.twist - prev.twist) >= L.RESET_TWIST)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        if self.latch["FELL"] is None and fell(r):
            self.latch["FELL"] = self.steps
        if self.latch["TIPPED"] is None and tipped(r):
            self.latch["TIPPED"] = self.steps
        self.max_twist = max(self.max_twist, r.twist)
        if self.latch["TURNED"] is None and math.degrees(r.twist) >= L.TURN_PART_DEG - L.EPS:
            self.latch["TURNED"] = self.steps
        if self.latch["LUGS"] is None and r.twist >= L.LUG_RAD - L.EPS:
            self.latch["LUGS"] = self.steps
            self.top_lift = r.lift
        if self.latch["LUGS"] is not None:
            self.top_lift = r.lift if self.top_lift is None else min(self.top_lift, r.lift)
            self.max_depth = max(self.max_depth, r.lift - self.top_lift)
            if self.latch["STROKE"] is None and r.lift - self.top_lift > L.STROKE + L.EPS:
                self.latch["STROKE"] = self.steps
        try:
            dist = states[PART].distance
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, gap = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the base, and the largest change of the head's
        turn (deg) and travel (m), from where each was at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        out = {"move": 0.0, "turn": 0.0, "twist": 0.0, "lift": 0.0}
        for x in h[1:]:
            out["move"] = max(out["move"], float(np.linalg.norm(x.pos - first.pos)))
            out["turn"] = max(out["turn"], turn_deg(first.rot, x.rot))
            out["twist"] = max(out["twist"], abs(math.degrees(x.twist - first.twist)))
            out["lift"] = max(out["lift"], abs(x.lift - first.lift))
        return out

    def judge(self, r: Reading, gap) -> Result:
        stand = standing(r)
        unlocked = self.latch["LUGS"] is not None and head_up(r)
        off = spout_off_deg(r)
        spout_ok = off <= L.SPOUT_TOL_DEG + L.EPS
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS
                     and m["twist"] <= L.STILL_TURN_DEG + L.EPS and m["lift"] <= L.STILL_MOVE + L.EPS)
        level = tier(self.latched, released and still, stand, unlocked, spout_ok, self.latch["TURNED"] is not None)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "standing": stand, "unlocked": unlocked, "spout_ok": spout_ok, "turned30": self.latch["TURNED"] is not None,
            "twist_deg": round(math.degrees(r.twist), 2), "max_twist_deg": round(math.degrees(self.max_twist), 2),
            "lift_mm": _mm(r.lift), "head_up_mm": _mm(L.TRAVEL - r.lift), "stroke_depth_mm": _mm(self.max_depth),
            "spout_off_front_deg": round(spout_off_deg(r), 2), "tilt_deg": round(tilt_of(r), 2),
            "xy_off_mm": _mm(math.hypot(float(r.pos[0] - L.BX), float(r.pos[1] - L.BY))),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap), "bottle_mm": _mmv(r.pos), "bottle_yaw_deg": round(yaw_deg(r.rot), 2),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                              "twist_deg": round(m["twist"], 3), "lift_mm": _mm(m["lift"])},
        }
        return Result(level, bool(level == 1.0), details)
