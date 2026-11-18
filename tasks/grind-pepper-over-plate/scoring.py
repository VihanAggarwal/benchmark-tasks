"""The scorer, as plain functions of the mill's and the plate's states and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mill (a mechkit AssemblyState: the body, and the head on a free spindle about the mill's axis), the plate (an
ObjectState) and both arms through rigstate.

    HEAD        the head's angle relative to the body, unwrapped step by step (the simulator may report it modulo a
                turn; one control step turns it far less than half a turn). Grinding is clockwise seen from above
                the mill: the head turning NEGATIVE about the mill's own +z axis.
    OVER        the outlet (the centre of the mill's base) within 80 mm in plan of the plate's centre, at least 5 mm
                above the plate's top (held clear, not resting on the plate), and the mill within 30 deg of upright
    GROUND      the clockwise head rotation accumulated on steps that start AND end OVER (turning back counts for
                nothing and does not undo grinding); TURNS = whole turns of GROUND
    STOOD       the mill within 10 deg of upright, its base within 5 mm of the table top and its axis at least 160 mm
                from the plate's centre (on the table, off the plate)
    CLEAR       rigstate.released with the mill's own distance: each gripper's finger zone at least 30 mm from the
                mill; unreadable arms, or a distance that cannot be computed, are NOT clear
    STILL       over the last 20 steps the base centre and the knob's top moved at most 2 mm
    FAILED      latched for the rest of the trial: the mill tipped past 60 deg (dropped), its base more than 20 mm
                below the table top (off the table), or the plate pushed more than 30 mm or tipped past 10 deg

    score = 0 if FAILED, else 0.15 x min(TURNS, 5), plus 0.25 when TURNS >= 5 and STOOD and CLEAR held on each of
    the last 20 steps and STILL (then 1.0: success). Possible scores: 0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

MILL, HEAD, PLATE = "mill", "head", "plate"
KNOB_TOP = np.array([0.0, 0.0, L.HEIGHT * L.MM])  # mill frame
OUTLET = np.zeros(3)


def wrap(a: float) -> float:
    """An angle (rad) brought into [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def tilt_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one step (world, m, rad)."""
    origin: np.ndarray  # the mill's base centre (its outlet)
    rot: np.ndarray  # the mill's rotation
    head: float  # the head's joint reading as reported (maybe wrapped)
    plate: np.ndarray  # the plate's centre
    plate_rot: np.ndarray
    track: np.ndarray  # (2, 3): base centre and knob top, for stillness


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not n > 0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read(mill, plate) -> Reading | None:
    """A Reading, or None when a state is missing, incomplete or not finite."""
    if mill is None or plate is None:
        return None
    try:
        rot = np.asarray(mill.rotation(), dtype=float)
        origin = np.asarray(mill.origin, dtype=float).ravel()
        head = float(mill.joints[HEAD])
        knob = np.asarray(mill.to_world(KNOB_TOP), dtype=float).ravel()
        p = np.asarray(plate.pos, dtype=float).ravel()
        prot = _quat_to_matrix(plate.quat_xyzw)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    arrays = (rot, origin, knob, p, prot)
    if rot.shape != (3, 3) or origin.size != 3 or p.size != 3:
        return None
    if not all(np.all(np.isfinite(a)) for a in arrays) or not math.isfinite(head):
        return None
    return Reading(origin, rot, head, p, prot, np.vstack([origin[None, :], knob[None, :]]))


def over_plate(r: Reading) -> tuple[bool, float, float]:
    """(OVER, plan distance of the outlet from the plate's centre (m), outlet height over the plate's top (m))."""
    plan = float(math.hypot(r.origin[0] - r.plate[0], r.origin[1] - r.plate[1]))
    plate_top = float(r.plate[2]) + L.PLATE_H * L.MM / 2
    above = float(r.origin[2]) - plate_top
    ok = (plan <= L.TARGET_R + L.EPS and above >= L.CLEARANCE - L.EPS
          and tilt_deg(r.rot) <= L.GRIND_TILT_DEG + L.EPS)
    return ok, plan, above


def stood(r: Reading) -> bool:
    off_plate = math.hypot(r.origin[0] - r.plate[0], r.origin[1] - r.plate[1]) >= L.STAND_OFF_PLATE - L.EPS
    return bool(tilt_deg(r.rot) <= L.STAND_TILT_DEG + L.EPS and abs(float(r.origin[2]) - L.TOP) <= L.STAND_Z_TOL
                and off_plate)


def failure(r: Reading, plate_start: np.ndarray) -> str | None:
    if tilt_deg(r.rot) > L.DROP_TILT_DEG:
        return "mill tipped past 60 deg (dropped)"
    if float(r.origin[2]) < L.TOP - L.FALL_Z:
        return "mill off the table"
    if float(np.linalg.norm(r.plate[:2] - plate_start[:2])) > L.PLATE_MOVE:
        return "plate pushed more than 30 mm"
    if tilt_deg(r.plate_rot) > L.PLATE_TILT_DEG:
        return "plate tipped past 10 deg"
    return None


def tier(turns: int, failed: bool, finished: bool) -> float:
    if failed:
        return 0.0
    n = min(int(turns), L.TURNS_NEEDED)
    if n >= L.TURNS_NEEDED and finished:
        return 1.0
    return round(L.PER_TURN * n, 4)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The grinding tally, the failure latch and the last 20 steps' verdicts (stillness, the stand-back)."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0  # fallback resets seen (teleports back to the start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.ground = 0.0  # rad of clockwise head rotation while OVER
        self.turned = 0.0  # rad of head rotation anywhere, either way (trace only)
        self.head = None  # unwrapped head angle
        self.prev: Reading | None = None
        self.prev_over = False
        self.failed: str | None = None
        self.plate_start: np.ndarray | None = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.ok: deque = deque(maxlen=self.window)
        self.last = Result(0.0, False, {})

    # -- reset fallback ----------------------------------------------------------------------------------------
    @staticmethod
    def at_start(r: Reading) -> bool:
        start = np.array([*L.MILL_START, L.TOP])
        return (float(np.linalg.norm(r.origin - start)) <= L.RESET_START_TOL and tilt_deg(r.rot) <= 2.0
                and math.hypot(r.plate[0] - L.PLATE_CENTRE[0], r.plate[1] - L.PLATE_CENTRE[1]) <= L.RESET_START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The simulator put the scene back on its start with a jump no trial makes in one control step."""
        return cls.at_start(now) and (float(np.linalg.norm(now.origin - prev.origin)) >= L.RESET_JUMP
                                      or float(np.linalg.norm(now.plate - prev.plate)) >= L.RESET_JUMP)

    # -- per step ------------------------------------------------------------------------------------------------
    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        mill = states.get(MILL) if isinstance(states, dict) else None
        plate = states.get(PLATE) if isinstance(states, dict) else None
        r = read(mill, plate)
        if r is None:
            # A blow-up or a missing state: nothing is credited on this step, the 20-step windows start again and
            # the head's unwrapping restarts from the next valid reading (no turn is counted across the gap).
            self.bad_steps += 1
            self.hist.clear()
            self.ok.clear()
            self.prev = None
            self.prev_over = False
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        if self.plate_start is None:
            self.plate_start = r.plate.copy()

        over, plan, above = over_plate(r)
        if self.prev is None:
            self.head = r.head
        else:
            d = wrap(r.head - self.prev.head)
            self.head += d
            self.turned += abs(d)
            grind = L.GRIND_SENSE * d
            if grind > 0.0 and over and self.prev_over:
                self.ground += grind
        self.prev, self.prev_over = r, over
        self.hist.append(r)

        if self.failed is None:
            self.failed = failure(r, self.plate_start)
        try:
            dist = mill.distance
            clear = bool(rigstate.released(arms, dist, L.LET_GO))
        except Exception:  # noqa: BLE001 - no distance: not clear
            dist, clear = None, False
        is_stood = stood(r)
        self.ok.append(bool(is_stood and clear))
        full = len(self.ok) == self.window and all(self.ok)
        m = self.motion()
        still = bool(m is not None and m <= L.STILL_MOVE + L.EPS)
        turns = int(math.floor(L.turns(self.ground) + 1e-6))
        level = tier(turns, self.failed is not None, full and still)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        details = {
            "valid_state": True, "score": level, "turns": turns, "ground_deg": round(math.degrees(self.ground), 1),
            "head_deg": round(math.degrees(self.head), 1), "turned_deg": round(math.degrees(self.turned), 1),
            "over": bool(over), "outlet_plan_mm": _mm(plan), "outlet_above_mm": _mm(above),
            "tilt_deg": round(tilt_deg(r.rot), 2), "stood": is_stood, "clear_now": bool(clear), "ok_20": bool(full),
            "still_20": still, "motion_mm": None if m is None else _mm(m), "failed": self.failed,
            "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last

    def motion(self) -> float | None:
        """Over the last 20 steps: the largest distance (m) a tracked point is from where it was at the window's
        start; None before 20 steps of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(x.track - h[0].track, axis=1).max()) for x in h[1:])
