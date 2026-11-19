"""The scorer and the bayonet's J-slot rule, as plain functions of the pendant's, the bulb's, the two stands' and the
arms' state. NumPy only; no simulator needed.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads mechkit AssemblyStates: the pendant (a fixed-base articulation: the rose, the flex's
joints, the holder on its twist hinge ``holder`` whose link frame is the centre of the mouth, the plungers
``plunger_a``/``plunger_b`` and the hooks' catches ``catch_a``/``catch_b``), the bulb (base frame at the centre of its
cap's end, z toward the cap), the ceiling stand and the ring stand (both fixed).

    LOCKED      at the last step: both catches 80% or more of their travel in (the J rule has the pins in the hooks),
                the bulb turned 13 to 17 degrees from the J's leg, its cap's end 1.5 to 2.5 mm over the plungers'
                rest (the pins on the hook floor at 2 mm), its axis within 5 degrees and 1.5 mm of the holder's
    RELEASED    on every step of the last 1 s both grippers' finger zones 2 mm or more from the bulb and the holder
                (its body, plungers and catches)
    HANGING     over the last 1 s the mouth moved under 10 mm, and at the last step it is within 30 mm (top view)
                and 15 mm (height) of where it hung at the start
    DROPPED     latched: after the bulb has left its stand (its cap's end 8 mm over its start), its lowest point
                within 2 mm of the table, or its globe (a 30 mm sphere round the globe's centre) within 1 mm of the
                ceiling stand or the ring stand
    STRAINED    latched: the mouth more than 15 mm below where it hung at the start
    WOUND       latched: the holder turned more than 90 degrees on its twist hinge

    0      dropped, or the bulb not locked
    0.25   locked, but strained
    0.5    locked, not strained, but wound
    1.0    locked, not strained or wound, released and hanging
    0.5    otherwise

Possible scores: 0, 0.25, 0.5, 1.0.

``Bayonet`` is the J-slot rule (the slot's shape, declared): the simulator's step rule (task.py) and the offline
stand-in both run it, and it drives the catches. It reads the bulb in the holder's frame: lateral offset, tilt, the
cap's end height over the mouth and the turn (0 at the J's leg, 15 degrees at the hook).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PENDANT, BULB, STAND, RING = L.PENDANT, L.BULB, L.STAND, L.RING_STAND
PARTS = (PENDANT, BULB, STAND, RING)
HOLDER_BODIES = ("holder", "plunger_a", "plunger_b", "catch_a", "catch_b")
CATCHES = ("catch_a", "catch_b")
MOUTH_START = np.array([L.PEND_X, L.PEND_Y, L.MOUTH_Z])
BULB_START = np.array([L.RING_X, L.RING_Y, L.BULB_START_Z])


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


def wrap_half(deg: float) -> float:
    """An angle folded to [-90, 90): the cap's two pins are alike, so the turn repeats every 180 degrees."""
    return (deg + 90.0) % 180.0 - 90.0


def bulb_in_holder(mouth, holder_rot, bulb_pos, bulb_rot) -> tuple[float, float, float, float]:
    """(lateral offset m, cap's end height over the mouth m, tilt deg, turn deg) of the bulb in the holder's frame.
    The turn is 0 with the pins at the J's leg and 15 at the hooks."""
    rh = np.asarray(holder_rot, dtype=float)
    rb = np.asarray(bulb_rot, dtype=float)
    d = rh.T @ (np.asarray(bulb_pos, dtype=float) - np.asarray(mouth, dtype=float))
    ax = rh.T @ rb[:, 2]
    xb = rh.T @ rb[:, 0]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(ax[2])))))
    phi = math.degrees(math.atan2(float(xb[1]), float(xb[0])))
    return float(math.hypot(d[0], d[1])), float(d[2]), tilt, wrap_half(phi - L.ENTRY_DEG)


class Bayonet:
    """The J-slot rule. ``update`` -> True while the pins are in the hooks (the catches go in under them).

    In the bore: the cap's axis within 2 mm and 6 degrees of the holder's and the pins above the mouth. Entering the
    bore, the pins must be at the leg (turn within 5 degrees), else the J has been missed (``ok`` false until the
    bulb comes out again). In the J: turned past 5 degrees with the cap's end under 3 mm (the pins still under the
    J's tooth), turned past -5 or past 20 degrees: forced (``ok`` false). In the hooks: from ``ok`` with the turn 13
    to 17 degrees and the cap's end 3 mm or more up (the pins over the lips); they stay in while the turn is 9 to 21
    degrees (the lips hold them) and the bulb is in the bore."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.inside = False
        self.ok = False
        self.hooked = False
        self.missed = 0  # how many times the J was missed or forced (diagnostics)

    @staticmethod
    def in_bore(lateral: float, h_end: float, tilt: float) -> bool:
        return bool(lateral <= L.INSIDE_LAT + L.EPS and tilt <= L.INSIDE_TILT_DEG
                    and h_end - L.PIN_FROM_END > 0.0)

    def update(self, lateral: float, h_end: float, tilt: float, turn: float) -> bool:
        vals = (lateral, h_end, tilt, turn)
        if not all(math.isfinite(v) for v in vals):
            return self.hooked  # an unreadable step changes nothing
        if not self.in_bore(lateral, h_end, tilt):
            self.inside = self.ok = self.hooked = False
            return False
        c = h_end - L.H_PL
        if not self.inside:
            self.inside = True
            self.ok = abs(turn) <= L.LEG_TOL_DEG
            if not self.ok:
                self.missed += 1
        if self.hooked and not (L.HOLD_LO_DEG <= turn <= L.HOLD_HI_DEG):
            self.hooked = False
        if self.ok and not self.hooked:
            if turn < -L.LEG_TOL_DEG or turn > L.OVER_DEG or (turn > L.TOOTH_DEG and c < L.C_HOOK - L.EPS):
                self.ok = False
                self.missed += 1
            elif L.HOOK_LO_DEG <= turn <= L.HOOK_HI_DEG and c >= L.C_HOOK - L.EPS:
                self.hooked = True
        return self.hooked

    def status(self) -> dict:
        return {"inside": self.inside, "ok": self.ok, "hooked": self.hooked, "missed": self.missed}


@dataclass(frozen=True)
class Reading:
    """One step (m, deg). Positions in the world frame with the table top at z 0."""
    mouth: np.ndarray  # the centre of the holder's mouth
    holder_rot: np.ndarray
    twist_deg: float  # the holder's twist hinge
    catches: tuple  # (catch_a, catch_b) travel in (m)
    plungers: tuple
    bulb_pos: np.ndarray  # the centre of the cap's end
    bulb_rot: np.ndarray
    bulb_low: float
    lateral: float
    h_end: float
    tilt_deg: float
    turn_deg: float

    @property
    def c(self) -> float:
        """How far the cap's end stands over the plungers' rest (m)."""
        return self.h_end - L.H_PL

    def globe_centre(self) -> np.ndarray:
        """World frame (states' frame, table top at L.TOP)."""
        return self.bulb_pos + np.array([0.0, 0.0, L.TOP]) + self.bulb_rot @ np.array([0.0, 0.0, L.GLOBE_C])


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        p, b = states.get(PENDANT), states.get(BULB)
        if p is None or b is None:
            return None
        rh, mouth = p.link_pose("holder")
        rh = np.asarray(rh, dtype=float)
        mouth = np.asarray(mouth, dtype=float).ravel()
        twist = math.degrees(float(p.joints["holder"]))
        catches = tuple(float(p.joints[n]) for n in CATCHES)
        plungers = (float(p.joints["plunger_a"]), float(p.joints["plunger_b"]))
        bo = np.asarray(b.origin, dtype=float).ravel()
        rb = rotation(b.quat_xyzw)
        low = float(np.asarray(b.low, dtype=float).ravel()[2])
        if mouth.size != 3 or bo.size != 3 or rh.shape != (3, 3):
            return None
        nums = (twist, low, *catches, *plungers)
        if not (np.all(np.isfinite(mouth)) and np.all(np.isfinite(rh)) and np.all(np.isfinite(bo))
                and np.all(np.isfinite(rb)) and all(math.isfinite(v) for v in nums)):
            return None
        lat, h_end, tilt, turn = bulb_in_holder(mouth, rh, bo, rb)
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    off = np.array([0.0, 0.0, L.TOP])
    return Reading(mouth - off, rh, twist, catches, plungers, bo - off, rb, low - L.TOP, lat, h_end, tilt, turn)


# -- the conditions --------------------------------------------------------------------------------------------------
def locked(r: Reading) -> bool:
    return bool(min(r.catches) >= L.LOCK_CATCH * L.CATCH_TRAVEL - L.EPS
                and L.LOCK_TURN[0] - L.EPS <= r.turn_deg <= L.LOCK_TURN[1] + L.EPS
                and L.LOCK_C[0] - L.EPS <= r.c <= L.LOCK_C[1] + L.EPS
                and r.tilt_deg <= L.LOCK_TILT_DEG + L.EPS and r.lateral <= L.LOCK_LAT + L.EPS)


def sag(r: Reading) -> float:
    """How far the mouth is below where it hung at the start (m; negative: above)."""
    return float(MOUTH_START[2] - r.mouth[2])


def strained(r: Reading) -> bool:
    return sag(r) > L.STRAIN_DZ + L.EPS


def wound(r: Reading) -> bool:
    return abs(r.twist_deg) > L.WOUND_DEG


def off_start(r: Reading) -> tuple[float, float]:
    """(top-view, height) distance of the mouth from where it hung at the start."""
    d = r.mouth - MOUTH_START
    return float(math.hypot(d[0], d[1])), float(abs(d[2]))


def tier(dropped: bool, is_locked: bool, is_strained: bool, is_wound: bool, released: bool, hanging: bool) -> float:
    if dropped or not is_locked:
        return 0.0
    if is_strained:
        return 0.25
    if is_wound:
        return 0.5
    return 1.0 if (released and hanging) else 0.5


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The latches (dropped, strained, wound), the last second of readings (released, hanging) and the reset
    fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"dropped": None, "strained": None, "wound": None}
        self.why: dict = {}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})
        self.lifted = None
        self.unread = 0
        self.max_sag = -math.inf
        self.max_twist = 0.0
        self.max_c = -math.inf

    @property
    def dropped(self) -> bool:
        return self.latch["dropped"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (float(np.linalg.norm(r.mouth - MOUTH_START)) <= L.START_TOL
                and float(np.linalg.norm(r.bulb_pos - BULB_START)) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return (float(np.linalg.norm(now.bulb_pos - prev.bulb_pos)) >= L.RESET_JUMP
                or float(np.linalg.norm(now.mouth - prev.mouth)) >= L.RESET_JUMP)

    # -- per-step rules --------------------------------------------------------------------------------------------
    @staticmethod
    def _gap_fn(states):
        """Distance from world points to the bulb and the holder's bodies."""
        p, b = states[PENDANT], states[BULB]

        def gap(points):
            return np.minimum(np.asarray(b.distance(points), dtype=float),
                              np.asarray(p.distance(points, bodies=HOLDER_BODIES), dtype=float))

        return gap

    def _released(self, states, arms) -> bool:
        """Both finger zones 2 mm or more from the bulb and the holder. Any failure: not released (fail closed)."""
        try:
            return bool(rigstate.released(arms, self._gap_fn(states), L.RELEASE_GAP))
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _globe_touch(states, r: Reading) -> str | None:
        """The globe within 1 mm of a stand: that stand's name, else None. A stand that cannot be read is skipped
        (both are fixed)."""
        c = r.globe_centre()
        for name in (STAND, RING):
            s = states.get(name)
            if s is None:
                continue
            try:
                d = float(np.min(np.asarray(s.distance(c), dtype=float)))
            except Exception:  # noqa: BLE001
                continue
            if math.isfinite(d) and d <= L.GLOBE_R + L.DROP_TOUCH + L.EPS:
                return name
        return None

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
        if self.lifted is None and r.bulb_pos[2] >= BULB_START[2] + L.LEFT_STAND - L.EPS:
            self.lifted = step
        if self.lifted is not None and self.latch["dropped"] is None:
            touch = self._globe_touch(states, r)
            if r.bulb_low <= L.DROP_TABLE + L.EPS:
                self.latch["dropped"] = step
                self.why["dropped"] = "bulb on the table"
            elif touch is not None:
                self.latch["dropped"] = step
                self.why["dropped"] = f"globe within 1 mm of the {touch}"
        if self.latch["strained"] is None and strained(r):
            self.latch["strained"] = step
        if self.latch["wound"] is None and wound(r):
            self.latch["wound"] = step
        self.max_sag = max(self.max_sag, sag(r))
        self.max_twist = max(self.max_twist, abs(r.twist_deg))
        self.max_c = max(self.max_c, r.c) if r.lateral <= L.INSIDE_LAT else self.max_c
        arms_ok = rigstate.readable(arms)
        if not arms_ok:
            self.unread += 1
            self.hist.clear()  # no 1.0 can be held through unreadable steps
        rel = arms_ok and self._released(states, arms)
        self.hist.append((r, rel))
        self.prev = r
        self.last = self.judge(r, rel)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the mouth's largest move from where it was at the window's start, and whether both
        hands were clear on every step; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0][0]
        move = max(float(np.linalg.norm(x.mouth - first.mouth)) for x, _ in h[1:])
        return {"mouth_move": move, "released": all(rel for _, rel in h)}

    def judge(self, r: Reading, released_now: bool) -> Result:
        is_locked = locked(r)
        m = self.motion()
        plan, height = off_start(r)
        released = bool(m is not None and m["released"])
        hanging = bool(m is not None and m["mouth_move"] < L.HANG_MOVE and plan <= L.HANG_PLAN + L.EPS
                       and height <= L.HANG_DZ + L.EPS)
        is_strained = self.latch["strained"] is not None
        is_wound = self.latch["wound"] is not None
        score = tier(self.dropped, is_locked, is_strained, is_wound, released, hanging)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch), "why": dict(self.why),
            "locked": is_locked, "released_1s": released, "released_now": bool(released_now), "hanging_1s": hanging,
            "strained": is_strained, "wound": is_wound, "lifted_step": self.lifted,
            "arms_unreadable_steps": self.unread,
            "catches_mm": [_mm(v) for v in r.catches], "plungers_mm": [_mm(v) for v in r.plungers],
            "turn_deg": round(r.turn_deg, 2), "c_mm": _mm(r.c), "lateral_mm": _mm(r.lateral),
            "tilt_deg": round(r.tilt_deg, 2), "twist_deg": round(r.twist_deg, 2),
            "max_twist_deg": round(self.max_twist, 2), "sag_mm": _mm(sag(r)), "max_sag_mm": _mm(self.max_sag),
            "max_c_mm": _mm(self.max_c) if math.isfinite(self.max_c) else None,
            "mouth_off_plan_mm": _mm(plan), "mouth_off_height_mm": _mm(height),
            "bulb_mm": [_mm(v) for v in r.bulb_pos], "bulb_low_mm": _mm(r.bulb_low),
            "motion": None if m is None else {"mouth_move_mm": _mm(m["mouth_move"]), "released": m["released"]},
        }
        return Result(score, bool(score == 1.0), details)
