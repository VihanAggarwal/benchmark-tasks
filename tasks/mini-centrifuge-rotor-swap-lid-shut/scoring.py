"""The scorer, as plain functions of the parts' and the arms' state. NumPy only; no simulator needed.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads mechkit AssemblyStates: the centrifuge (base frame on the table under its shaft; joints
``lid`` (rad, 0 shut), ``tab`` (m, + pressed), ``plunger_p`` and ``plunger_n`` (m, 0 out, - pushed back)), the two
rotors (frame at the middle of the hub's underside), the two strips (frame at the middle of the underside, x along
the strip) and the rack (unused but for completeness). The arms are read only for the hands-off check (rigstate's
finger zones).

    LATCH       the scorer's own copy of the lid catch (layout.Latch), fed the lid angle and the tab's travel: shut
                and latched at the start; the tab pressed 2 mm frees it; it catches again once the lid has opened
                3 degrees and is back within 0.6 degrees of shut with the tab up. LATCHED at a step: caught and the
                lid within 1 degree of shut
    SEATED      the strip rotor's hub within 1 mm of the shaft's axis, its underside within 0.5 mm of its seat
                (no more than 1 mm under it), tilted under 3 degrees, and both snap noses back out within 0.2 mm
                (the click)
    ON SHAFT    the strip rotor's hub within 2.5 mm of the axis, its underside less than 10 mm over its seat,
                tilted under 8 degrees (it sits on the pin or on the noses)
    STRIP IN    a strip in a slot of the strip rotor: its middle within 2.5 mm of the slot's middle line and 4 mm
                along it, its underside within 2 mm of the slot floor, tilted under 10 degrees from the rotor and
                turned under 15 degrees from the slot
    BALANCED    both strips in, in the two slots of one opposite pair (19 mm or 31 mm each side of the hub)
    STORED      the tube rotor's lowest point within 4 mm of the table top (not under it by more than 2 mm),
                tilted under 10 degrees, its footprint 5 mm or more clear of the unit's
    HANDS OFF   over the last 1 s (20 steps), every finger zone 30 mm or more from the boxes round the unit's
                bodies, the rotors and the strips; a step whose arms cannot be read is not hands off

    SLID        the unit moved 15 mm or more (3-D) from its start at any step: recorded in the trace only; it does
                not change the score (a slid but upright, unplugged unit is harmless)

    0      at any step a rotor or strip 20 mm under the table top (off the table), or the unit tilted over 10
           degrees (tipped), or the lid more than 2 degrees off its stop while caught (forced), or the tab
           pressed again after a catch that followed an opening (a plugged-in unit would have started a run);
           at the last step: hands off failed, or LATCHED with the strip rotor not SEATED or the load not BALANCED
    1.0    SEATED, BALANCED, LATCHED, STORED
    0.75   SEATED, BALANCED, STORED, not latched
    0.5    SEATED, STORED, not latched, not balanced
    0.25   STORED, the strip rotor ON SHAFT but not seated, not latched
    0      otherwise

Possible scores: 0, 0.25, 0.5, 0.75, 1.0.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

UNIT, TUBE_ROTOR, STRIP_ROTOR, RACK = "centrifuge", "tube_rotor", "strip_rotor", "strip_rack"
STRIPS = ("strip_a", "strip_b")
PARTS = (UNIT, TUBE_ROTOR, STRIP_ROTOR, RACK, *STRIPS)
LOOSE = (TUBE_ROTOR, STRIP_ROTOR, *STRIPS)
UNIT_JOINTS = ("lid", "tab", "plunger_p", "plunger_n")


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


@dataclass(frozen=True)
class Pose:
    p: np.ndarray  # frame origin, world, table top at z 0
    r: np.ndarray
    low: float  # lowest point, above the table top
    lo: np.ndarray  # world-aligned box round it (world, table top at z 0)
    hi: np.ndarray


@dataclass(frozen=True)
class Reading:
    unit: Pose
    joints: dict
    parts: dict  # name -> Pose for the rotors and strips
    boxes: tuple  # ((center, size), ...) world, the simulator's z (for rigstate)
    unit_box: tuple  # the unit body's (lo, hi), table top at z 0


def _pose(s) -> Pose:
    o = np.asarray(s.origin, dtype=float).ravel()
    lo = np.asarray(s.low, dtype=float).ravel()
    hi = np.asarray(s.high, dtype=float).ravel()
    if o.size != 3 or lo.size != 3 or hi.size != 3:
        raise ValueError("bad pose")
    r = rotation(s.quat_xyzw)
    off = np.array([0.0, 0.0, L.TOP])
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("not finite")
    return Pose(o - off, r, float(lo[2] - L.TOP), lo - off, hi - off)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        u = states.get(UNIT)
        if u is None or any(states.get(n) is None for n in LOOSE):
            return None
        unit = _pose(u)
        joints = {j: float(u.joints[j]) for j in UNIT_JOINTS}
        if not all(math.isfinite(v) for v in joints.values()):
            return None
        parts = {n: _pose(states[n]) for n in LOOSE}
        boxes = []
        unit_box = None
        for body in ("base", "lid", "tab", "plunger_p", "plunger_n"):
            lo, hi = (np.asarray(v, dtype=float) for v in u.link_bounds(body))
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                return None
            boxes.append(((lo + hi) / 2, hi - lo))
            if body == "base":
                unit_box = (lo - np.array([0.0, 0.0, L.TOP]), hi - np.array([0.0, 0.0, L.TOP]))
        for n in LOOSE:
            s = states[n]
            lo, hi = np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)
            boxes.append(((lo + hi) / 2, hi - lo))
    except (AttributeError, TypeError, ValueError, KeyError, IndexError, RuntimeError):
        return None
    return Reading(unit, joints, parts, tuple(boxes), unit_box)


# -- the readings ------------------------------------------------------------------------------------------------------
UNIT_START = np.array([L.UX, L.UY, 0.0])


def local(r: Reading, part: str) -> tuple[np.ndarray, np.ndarray]:
    """A part's frame origin and rotation in the unit's frame."""
    q = r.parts[part]
    return r.unit.r.T @ (q.p - r.unit.p), r.unit.r.T @ q.r


def unit_moved(r: Reading) -> float:
    return float(np.linalg.norm(r.unit.p - UNIT_START))


def unit_tilt(r: Reading) -> float:
    return angle_deg(r.unit.r[:, 2], np.array([0.0, 0.0, 1.0]))


def rotor_fit(r: Reading, part: str = STRIP_ROTOR) -> tuple[float, float, float]:
    """(distance from the shaft's axis, rise of the hub's underside over its seat, tilt in degrees from the
    shaft), unit frame."""
    p, rot = local(r, part)
    return float(math.hypot(p[0], p[1])), float(p[2] - L.SEAT_Z), angle_deg(rot[:, 2], np.array([0.0, 0.0, 1.0]))


def clicked(r: Reading) -> bool:
    return bool(r.joints["plunger_p"] >= -L.PLUNGER_OUT - L.EPS and r.joints["plunger_n"] >= -L.PLUNGER_OUT - L.EPS)


def seated(r: Reading) -> bool:
    off, rise, tilt = rotor_fit(r)
    return bool(off <= L.ROTOR_OFF_AXIS + L.EPS and -0.001 <= rise <= L.CLICK_MM * L.MM + L.EPS
                and tilt < L.ROTOR_TILT_DEG and clicked(r))


def on_shaft(r: Reading) -> bool:
    off, rise, tilt = rotor_fit(r)
    return bool(off <= L.ON_SHAFT_AXIS + L.EPS and -0.001 <= rise < L.ON_SHAFT_RISE and tilt < L.ON_SHAFT_TILT_DEG)


def strip_slot(r: Reading, strip: str) -> int | None:
    """The slot (0..3) a strip sits in, or None."""
    rot = r.parts[STRIP_ROTOR]
    s = r.parts[strip]
    q = rot.r.T @ (s.p - rot.p)
    sr = rot.r.T @ s.r
    tilt = angle_deg(sr[:, 2], np.array([0.0, 0.0, 1.0]))
    turn = angle_deg(np.array([sr[0, 0], sr[1, 0], 0.0]), np.array([1.0, 0.0, 0.0]))
    turn = min(turn, 180.0 - turn)
    if not (abs(q[0]) <= L.STRIP_TOL_X + L.EPS and abs(q[2] - L.SEATED_Z) <= L.STRIP_TOL_Z + L.EPS
            and tilt < L.STRIP_TILT_DEG and turn < L.STRIP_TURN_DEG):
        return None
    for i, y in enumerate(L.SLOT_Y):
        if abs(q[1] - y) <= L.STRIP_TOL_XY + L.EPS:
            return i
    return None


def balanced(r: Reading) -> bool:
    slots = [strip_slot(r, s) for s in STRIPS]
    if any(v is None for v in slots):
        return False
    return tuple(sorted(slots)) in tuple(tuple(sorted(p)) for p in L.SLOT_PAIRS)


def stored(r: Reading) -> bool:
    t = r.parts[TUBE_ROTOR]
    tilt = angle_deg(t.r[:, 2], np.array([0.0, 0.0, 1.0]))
    lo, hi = r.unit_box
    gap = max(lo[0] - t.hi[0], t.lo[0] - hi[0], lo[1] - t.hi[1], t.lo[1] - hi[1])
    return bool(-0.002 <= t.low <= L.STORED_LOW + L.EPS and tilt < L.STORED_TILT_DEG and gap >= L.STORED_CLEAR - L.EPS)


def dropped(r: Reading) -> bool:
    return any(r.parts[n].low < L.DROP_Z for n in LOOSE)


def tier(zero: bool, seat: bool, bal: bool, latched: bool, store: bool, on: bool) -> float:
    if zero:
        return 0.0
    if latched:
        return 1.0 if (seat and bal and store) else 0.0
    if not store:
        return 0.0
    if seat:
        return 0.75 if bal else 0.5
    return 0.25 if on else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The latches (dropped, unit moved, lid forced, reopened), the scorer's copy of the lid catch, the last second
    of hands-off steps and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.HANDS_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"dropped": None, "unit_tipped": None, "lid_forced": None, "reopened": None}
        self.slid: int | None = None
        self.catch = L.Latch()
        self.hands: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})
        self.unread = 0
        self.lid_max = 0.0
        self.unit_max = 0.0

    @property
    def zero(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        t = r.parts[TUBE_ROTOR].p
        s = r.parts[STRIP_ROTOR].p
        tube0 = np.array([L.TUBE_ROTOR_START[0], L.TUBE_ROTOR_START[1], L.TUBE_ROTOR_START[2]])
        strip0 = np.array(L.STRIP_ROTOR_START)
        return bool(np.linalg.norm(r.unit.p - UNIT_START) <= L.START_TOL
                    and np.linalg.norm(t - tube0) <= L.START_TOL and np.linalg.norm(s - strip0) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.parts[n].p - prev.parts[n].p)) >= L.RESET_JUMP for n in LOOSE)

    def _hands_off(self, r: Reading, arms) -> bool:
        try:
            return bool(rigstate.released(arms, rigstate.boxes_distance(list(r.boxes)), clearance=L.HANDS_OFF))
        except Exception:  # noqa: BLE001 - a distance that cannot be computed is not hands off (fail closed)
            return False

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the hands-off window starts again
            self.hands.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "latch": dict(self.latch)})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        step = self.steps
        lid, tab = r.joints["lid"], r.joints["tab"]
        self.catch.step(lid, tab)
        self.lid_max = max(self.lid_max, math.degrees(lid))
        self.unit_max = max(self.unit_max, unit_moved(r))
        if self.latch["dropped"] is None and dropped(r):
            self.latch["dropped"] = step
        if self.latch["unit_tipped"] is None and unit_tilt(r) > L.UNIT_TILT_DEG:
            self.latch["unit_tipped"] = step
        if self.slid is None and unit_moved(r) >= L.UNIT_MOVED:
            self.slid = step
        if self.latch["lid_forced"] is None and self.catch.latched and math.degrees(lid) > L.FORCED_DEG:
            self.latch["lid_forced"] = step
        if self.latch["reopened"] is None and self.catch.reopened:
            self.latch["reopened"] = step
        if not rigstate.readable(arms):
            self.unread += 1
        off = rigstate.readable(arms) and self._hands_off(r, arms)
        self.hands.append(off)
        self.prev = r
        self.last = self.judge(r, off)
        return self.last

    def hands_off_1s(self) -> bool:
        return len(self.hands) == self.window and all(self.hands)

    def judge(self, r: Reading, off_now: bool) -> Result:
        seat = seated(r)
        bal = balanced(r)
        store = stored(r)
        on = on_shaft(r)
        lid_deg = math.degrees(r.joints["lid"])
        latched = bool(self.catch.latched and lid_deg <= 1.0)
        hands = self.hands_off_1s()
        score = tier(self.zero or not hands, seat, bal, latched, store, on)
        off, rise, tilt = rotor_fit(r)
        toff, trise, _ = rotor_fit(r, TUBE_ROTOR)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch), "slid_step": self.slid, "catch": self.catch.snapshot(),
            "latched": latched, "seated": seat, "balanced": bal, "stored": store, "on_shaft": on,
            "hands_off_1s": hands, "hands_off_now": bool(off_now), "arms_unreadable_steps": self.unread,
            "slots": [strip_slot(r, s) for s in STRIPS], "lid_deg": round(lid_deg, 2),
            "lid_max_deg": round(self.lid_max, 2), "tab_mm": _mm(r.joints["tab"]),
            "plungers_mm": [_mm(r.joints["plunger_p"]), _mm(r.joints["plunger_n"])],
            "strip_rotor_axis_mm": _mm(off), "strip_rotor_rise_mm": _mm(rise), "strip_rotor_tilt_deg": round(tilt, 2),
            "tube_rotor_axis_mm": _mm(toff), "tube_rotor_rise_mm": _mm(trise),
            "tube_rotor_low_mm": _mm(r.parts[TUBE_ROTOR].low), "unit_moved_mm": _mm(unit_moved(r)),
            "unit_moved_max_mm": _mm(self.unit_max), "unit_tilt_deg": round(unit_tilt(r), 2),
        }
        return Result(score, bool(score == 1.0), details)
