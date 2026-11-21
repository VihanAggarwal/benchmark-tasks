"""The scorer, as plain functions of the casting's, the two clamps' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the casting (base frame at its centre) and of the two clamps (fixed bases; joints
``handle``, 0 = open on its stop, QH_DC = vertical, dead centre, QH_UPPER = on its over-centre stop 1.5 degrees past
it; ``arm``, carrying the red pad; ``link``). Every reading is a pose or a joint angle; the arms are read only for the
let-go check.

    SEATED     the casting rests flat on the plate (tilt under 1 degree, lowest point within 1.5 mm of the plate's
               top) and its far face is within 1.0 mm of S1's face (at both ends of S1) and its right face within
               1.0 mm of S2's face (at both ends of S2)
    SHUT       a handle over centre: 0.5 degree or more past vertical (dead centre); let go there, the pad's force
               holds it on its stop
    LOCKED     shut and its pad's bottom face touching the casting's top face (between 1.5 mm into it and 0.5 mm
               above it, the pad's centre at least 4 mm inside the top face): it then presses the casting with 20 N
    EMPTY      shut and not touching the casting: closed on nothing (reported; it is simply not LOCKED)
    CROOKED    reported only (trace): the first step at which a clamp was LOCKED while the casting was more than
               1.5 mm off a stop. It does not change the score: each locked pad presses the casting with 20 N, so a
               casting clamped off a stop is held there by friction (about 12 N with one pad, 24 N with both) and is
               simply not SEATED, unless the clamp is reopened and the casting reseated
    RELEASED   on every step of the last 1 s both finger zones are 30 mm or more from the casting and from both clamps
    STILL      over the last 1 s the casting moved at most 1 mm and turned at most 1 degree and neither handle
               turned over 2 degrees

    0      not released and still (a held casting earns nothing), or not SEATED (this includes a clamp locked on
           an unseated casting)
    1.0    SEATED and both clamps LOCKED
    0.5    SEATED and one clamp LOCKED
    0.25   SEATED and no clamp LOCKED
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CASTING, CLAMP1, CLAMP2 = "casting", "clamp1", "clamp2"
PARTS = (CASTING, CLAMP1, CLAMP2)
HANDLE, ARM, LINK = "handle", "arm", "link"
CLAMP_OF = {CLAMP1: L.C1, CLAMP2: L.C2}
# The pad's bottom-centre point as drawn (arm at qa = 0, open), in each clamp's base frame (origin on the table under
# the arm pivot): mechkit's link_to_world takes it to where the arm carries it.
PAD_DRAWN = {n: c.pad_bottom(0.0) - np.array([c.px, c.py, 0.0]) for n, c in CLAMP_OF.items()}
CORNERS = np.array([(sx * L.CS_L / 2, sy * L.CS_W / 2, sz * L.CS_H / 2) for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-1, 1)])
MID_Z = L.PLATE_H + L.STOP_H / 2  # the stops' faces are read at their mid height
S1_POINTS = [np.array([L.S1_X0, y, MID_Z]) for y in (-0.040, 0.040)]  # inside S1's 90 mm face
S2_POINTS = [np.array([x, L.S2_Y1, MID_Z]) for x in (0.285, 0.315)]  # inside S2's 40 mm face
FACE_INSET = 0.003


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


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: np.ndarray  # the casting's centre
    rot: np.ndarray  # its rotation
    pts: np.ndarray  # its 8 corners
    q: dict  # clamp -> handle joint angle (rad)
    pad: dict  # clamp -> its pad's bottom-centre point (on the arm)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        cs = states.get(CASTING)
        if cs is None:
            return None
        o = np.asarray(cs.origin, dtype=float).ravel()
        r = rotation(cs.quat_xyzw)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
            return None
        o = o - np.array([0.0, 0.0, L.TOP])
        q, pad = {}, {}
        for name in (CLAMP1, CLAMP2):
            s = states.get(name)
            if s is None:
                return None
            v = float(s.joints[HANDLE])
            p = np.asarray(s.link_to_world(ARM, PAD_DRAWN[name]), dtype=float).ravel() - np.array([0.0, 0.0, L.TOP])
            if not (math.isfinite(v) and p.size == 3 and np.all(np.isfinite(p))):
                return None
            q[name], pad[name] = v, p
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(o, r, o + CORNERS @ r.T, q, pad)


# -- the readings ------------------------------------------------------------------------------------------------------
def stop_gaps(r: Reading) -> tuple[list, list]:
    """Signed gaps (m; + a gap, - overlap) from the casting's far face to S1's face at S1's two ends, and from its
    right face to S2's face at S2's two ends. A point of the stop that does not lie against the face (outside its
    width or its height) gives +inf: no contact there."""
    c, rot = r.pos, r.rot
    ax, ay, az = rot[:, 0], rot[:, 1], rot[:, 2]
    g1, g2 = [], []
    for e in S1_POINTS:
        d = e - c
        ok = abs(float(ay @ d)) <= L.CS_W / 2 - FACE_INSET and abs(float(az @ d)) <= L.CS_H / 2
        g1.append(float(ax @ d) - L.CS_L / 2 if ok else math.inf)
    for e in S2_POINTS:
        d = e - c
        ok = abs(float(ax @ d)) <= L.CS_L / 2 - FACE_INSET and abs(float(az @ d)) <= L.CS_H / 2
        g2.append(float(-ay @ d) - L.CS_W / 2 if ok else math.inf)
    return g1, g2


def flat_low(r: Reading) -> float:
    """The casting's lowest point above the plate's top (m)."""
    return float(np.min(r.pts[:, 2])) - L.PLATE_H


def seated_within(r: Reading, tol: float, tilt_max: float = L.SEAT_TILT_DEG) -> bool:
    g1, g2 = stop_gaps(r)
    return bool(all(abs(g) <= tol + L.EPS for g in g1 + g2) and tilt_deg(r.rot) <= tilt_max
                and abs(flat_low(r)) <= L.SEAT_FLAT)


def seated(r: Reading) -> bool:
    return seated_within(r, L.SEAT_TOL)


def seated_loose(r: Reading) -> bool:
    """Within the latch's 1.5 mm of both stops (and flat): the casting is in its corner for a clamp to close."""
    return seated_within(r, L.LATCH_TOL, 3.0)


def shut(q: float) -> bool:
    """The handle over centre (past dead centre toward its stop)."""
    return q >= L.QH_SHUT - L.EPS


def pad_touching(r: Reading, name: str) -> bool:
    """The pad's bottom-face centre over the casting's top face (4 mm inside) and touching it."""
    loc = r.rot.T @ (r.pad[name] - r.pos)
    over = abs(loc[0]) <= L.CS_L / 2 - L.PAD_INSET and abs(loc[1]) <= L.CS_W / 2 - L.PAD_INSET
    gap = float(loc[2]) - L.CS_H / 2
    return bool(over and L.PAD_TOUCH_LOW <= gap <= L.PAD_TOUCH_HIGH)


def locked(r: Reading, name: str) -> bool:
    return bool(shut(r.q[name]) and pad_touching(r, name))


def empty(r: Reading, name: str) -> bool:
    """Shut with nothing under the pad."""
    return bool(shut(r.q[name]) and not pad_touching(r, name))


def tier(settled: bool, is_seated: bool, n_locked: int) -> float:
    if not settled or not is_seated:
        return 0.0
    return {2: 1.0, 1: 0.5, 0: 0.25}[n_locked]


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


def _mmv(v) -> list:
    return [round(float(c) * 1000.0, 2) for c in v]


class Tracker:
    """The CROOKED record (trace only), the last 1 s of readings (stillness) and of the let-go check, and the reset
    fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"CROOKED": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["CROOKED"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return float(np.linalg.norm(r.pos - np.array([L.START_X, L.START_Y, L.START_Z]))) <= L.START_TOL

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The casting put back on its staging block with a jump no trial makes in one step (a reset)."""
        return bool(cls.at_start(now) and float(np.linalg.norm(now.pos - prev.pos)) >= L.RESET_JUMP)

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
        if self.latch["CROOKED"] is None and any(locked(r, n) for n in (CLAMP1, CLAMP2)) and not seated_loose(r):
            self.latch["CROOKED"] = self.steps
        try:
            dists = [states[n].distance for n in PARTS]

            def dist(points):
                return np.min([d(points) for d in dists], axis=0)
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
        """Over the last 1 s: the casting's largest move (m) and turn (deg) from where it was at the window's start,
        and each handle's largest turn (deg); None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        handle = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.pos - first.pos)))
            turn = max(turn, turn_deg(first.rot, x.rot))
            for n in (CLAMP1, CLAMP2):
                handle = max(handle, abs(math.degrees(x.q[n] - first.q[n])))
        return {"move": move, "turn": turn, "handle": handle}

    def judge(self, r: Reading, gap) -> Result:
        g1, g2 = stop_gaps(r)
        is_seated = seated(r)
        lk = {n: locked(r, n) for n in (CLAMP1, CLAMP2)}
        em = {n: empty(r, n) for n in (CLAMP1, CLAMP2)}
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS
                     and m["handle"] <= L.STILL_HANDLE_DEG + L.EPS)
        level = tier(released and still, is_seated, sum(lk.values()))
        far = [g for g in g1 if math.isfinite(g)]
        right = [g for g in g2 if math.isfinite(g)]
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "seated": is_seated, "seated_3mm": seated_loose(r),
            "gap_s1_mm": _mm(max(far, key=abs)) if far else None,
            "gap_s2_mm": _mm(max(right, key=abs)) if right else None,
            "gap_s1_ends_mm": [_mm(g) for g in g1], "gap_s2_ends_mm": [_mm(g) for g in g2],
            "casting_tilt_deg": round(tilt_deg(r.rot), 3), "casting_yaw_deg": round(yaw_deg(r.rot), 3),
            "casting_low_mm": _mm(flat_low(r)),
            "casting_mm": _mmv(r.pos),
            "q1_deg": round(math.degrees(r.q[CLAMP1]), 2), "q2_deg": round(math.degrees(r.q[CLAMP2]), 2),
            "psi1_deg": round(math.degrees(r.q[CLAMP1] - L.QH_DC), 2),
            "psi2_deg": round(math.degrees(r.q[CLAMP2] - L.QH_DC), 2),
            "locked1": lk[CLAMP1], "locked2": lk[CLAMP2], "empty1": em[CLAMP1], "empty2": em[CLAMP2],
            "pad1_mm": _mmv(r.pad[CLAMP1]), "pad2_mm": _mmv(r.pad[CLAMP2]),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3),
                                                "handle_deg": round(m["handle"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
