"""The scorer, as plain functions of the micrometer's joints, the gauge block's pose and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyStates of the micrometer (a fixed frame with four moving controls: the spindle's slide, the thimble, the
ratchet knob and the lock lever) and of the gauge block. Every reading is a joint or a pose; the arms are read only for
the let-go check.

    F1 FELL        latched: the block's centre more than 5 mm below the work plate's top (it left the plate)
    F2 SQUEEZED    latched: with the block between the faces, the thimble turned more than 10 degrees past the turn
                   at which the faces touch the block (the screw then closes on the block at about 1 N per micron;
                   the ratchet stops turning the thimble at touch, so it never does this)
    GAP            the distance between the two faces: 20.30 mm less the spindle's travel
    BETWEEN        the block's centre within 14.3 mm of the axis along its 35 mm side and 1.3 mm across its 9 mm side
                   (each 6.35 mm face wholly on the block's face), standing on the plate (within 1.5 mm), its measuring
                   faces within 5 degrees of square to the spindle
    TOUCH          between, and the gap 20.02 mm or less (the faces on the block)
    CLICKS         while touching, with the thimble seated (within 2 degrees of its touch turn) and the lock lever
                   off (under 34 degrees), the ratchet knob's slip on the thimble past where it stood when those
                   conditions last began, 15 degrees per click; the most it reached. Slip on a locked spindle, or
                   with the faces short of the block, earns no click
    LOCKED         the lock lever turned 80 degrees or more
    RELEASED       on every step of the last 1 s both finger zones are 30 mm or more from the block and the controls
    STILL          over the last 1 s the block moved under 0.5 mm and turned under 1 degree, the thimble, ratchet and
                   lever turned under 1 degree, the spindle moved under 5 microns

    0      F1 or F2 ever fired, or not released and still (a held block or a hand on a control earns nothing)
    1.0    between, gap within 0.03 mm of 20.000, 3 or more clicks, locked
    0.75   the same, not locked
    0.5    between, gap within 0.03 mm, 1 or 2 clicks
    0.25   between, gap within 0.20 mm, otherwise (no click yet, or the gap off by 0.03 to 0.20 mm)
    0      anything else (the faces not on a block between them, the gap more than 0.20 mm off)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

MIC, BLOCK = "micrometer", "block"
PARTS = (MIC, BLOCK)


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
    """One step (m, rad). ``block`` is the block's centre relative to the axis (x, y) and the plate's top (z)."""
    q: float  # the spindle's travel toward the anvil from as found
    theta: float  # thimble
    phi: float  # ratchet knob, relative to the thimble (rad): its slip once the thimble has stopped
    lever: float
    block: np.ndarray
    rot: np.ndarray  # the block's rotation


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        mic, blk = states.get(MIC), states.get(BLOCK)
        if mic is None or blk is None:
            return None
        j = mic.joints
        vals = [float(j[L.SPINDLE]), float(j[L.THIMBLE]), float(j[L.RATCHET]), float(j[L.LEVER])]
        o = np.asarray(blk.origin, dtype=float).ravel()
        r = rotation(blk.quat_xyzw)
        finite = all(math.isfinite(v) for v in vals) and np.all(np.isfinite(o)) and np.all(np.isfinite(r))
        if o.size != 3 or not finite:
            return None
        rel = o - np.array([L.X_A, L.Y_A, L.TOP + L.PL])
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(vals[0], vals[1], vals[2], vals[3], rel, r)


# -- the readings ----------------------------------------------------------------------------------------------------
def gap(r: Reading) -> float:
    return L.gap_of(r.q)


def fell(r: Reading) -> bool:
    return bool(float(r.block[2]) < L.FELL_DZ)


def block_between(block_pos, rot) -> bool:
    """The block's centre ``block_pos`` (relative to the axis and the plate's top) and rotation: each 6.35 mm face
    wholly on a measuring face, standing on the plate, square to the spindle."""
    p = np.asarray(block_pos, dtype=float)
    return bool(abs(p[0]) <= L.BETWEEN_DX + L.EPS and abs(p[1]) <= L.BETWEEN_DY + L.EPS
                and abs(p[2] - L.BL_Z / 2) <= L.BETWEEN_DZ + L.EPS
                and tilt_deg(rot) <= L.BETWEEN_TILT_DEG + 1e-6)


def between(r: Reading) -> bool:
    return block_between(r.block, r.rot)


def touching(r: Reading) -> bool:
    return bool(between(r) and gap(r) - L.T_BLOCK <= L.CONTACT_TOL + L.EPS)


def squeezed_now(r: Reading) -> bool:
    return bool(between(r) and r.theta - L.THETA_C > math.radians(L.SQUEEZE_DEG) + L.EPS)


def seated(r: Reading) -> bool:
    """The faces on the block with the lock off: the only state in which the ratchet's slip counts as clicks."""
    return bool(touching(r) and r.theta >= L.THETA_C - math.radians(L.CLICK_SEAT_DEG) - L.EPS
                and r.lever < L.CLICK_LEVER_MAX)


def locked(r: Reading) -> bool:
    return bool(r.lever >= L.LOCK_Q - L.EPS)


def clicks_of(slip_rad: float) -> int:
    return int(max(0.0, slip_rad) / math.radians(L.CLICK_DEG) + 1e-9)


def tier(clean: bool, btw: bool, gap_err: float, clicks: int, is_locked: bool) -> float:
    """``clean``: no latch, hands off and still."""
    if not clean or not btw:
        return 0.0
    if gap_err <= L.GAP_TOL + L.EPS:
        if clicks >= L.CLICKS_FULL:
            return 1.0 if is_locked else 0.75
        return 0.5 if clicks >= 1 else 0.25
    if gap_err <= L.GAP_NEAR + L.EPS:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 3)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


class Tracker:
    """The latches (F1, F2), the touch record (the ratchet's slip), the last 1 s of readings (stillness) and of the
    let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "F2": None}
        self.slip = 0.0  # the most the ratchet slipped (rad) past where it stood at first touch, faces on the block
        self.touch_step: int | None = None
        self.seated_prev = False
        self.phi0 = 0.0  # the knob's turn on the thimble when the faces last became seated with the lock off
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        b0 = np.array([L.BL_START_DX, 0.0, L.BL_Z / 2])
        return bool(float(np.linalg.norm(r.block - b0)) <= L.START_TOL and abs(r.theta) <= 0.05
                    and abs(r.phi) <= 0.05 and abs(r.lever) <= 0.05)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """Everything put back on its starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return bool(float(np.linalg.norm(now.block - prev.block)) >= L.RESET_JUMP or abs(prev.theta) > 0.3
                    or abs(prev.lever) > 0.3)

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
        if self.latch["F1"] is None and fell(r):
            self.latch["F1"] = self.steps
        if self.latch["F2"] is None and squeezed_now(r):
            self.latch["F2"] = self.steps
        if touching(r) and self.touch_step is None:
            self.touch_step = self.steps
        if seated(r):
            if not self.seated_prev:
                self.phi0 = r.phi
            self.slip = max(self.slip, r.phi - self.phi0)
        self.seated_prev = seated(r)
        try:
            blk, mic = states[BLOCK], states[MIC]

            def dist(points):
                return np.minimum(blk.distance(points), mic.distance(points, bodies=L.CONTROLS))
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, hands = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                hands = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, hands)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) of the block and of the spindle, and turn (deg) of the block or any
        control, from where it was at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = spindle = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.block - first.block)))
            spindle = max(spindle, abs(x.q - first.q))
            turn = max(turn, turn_deg(first.rot, x.rot))
            for a, b in ((x.theta, first.theta), (x.phi, first.phi), (x.lever, first.lever)):
                turn = max(turn, abs(math.degrees(a - b)))
        return {"move": move, "turn": turn, "spindle": spindle}

    def judge(self, r: Reading, hands) -> Result:
        btw, touch, lock = between(r), touching(r), locked(r)
        g = gap(r)
        gap_err = abs(g - L.T_BLOCK)
        n_clicks = clicks_of(self.slip)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + 1e-6
                     and m["spindle"] <= L.STILL_SPINDLE + L.EPS)
        level = tier(not self.latched and released and still, btw, gap_err, n_clicks, lock)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "gap_mm": _mm(g), "gap_err_mm": _mm(gap_err), "between": btw, "touching": touch, "seated": seated(r),
            "clicks": n_clicks,
            "slip_deg": round(math.degrees(self.slip), 2), "phi_deg": round(math.degrees(r.phi), 2),
            "locked": lock, "lever_deg": round(math.degrees(r.lever), 2),
            "theta_deg": round(math.degrees(r.theta), 2),
            "over_touch_deg": round(math.degrees(r.theta - L.THETA_C), 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(hands),
            "block_mm": _mmv(r.block), "block_tilt_deg": round(tilt_deg(r.rot), 2),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                              "spindle_um": round(m["spindle"] * 1e6, 2)},
        }
        return Result(level, bool(level == 1.0), details)
