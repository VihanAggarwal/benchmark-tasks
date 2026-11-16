"""The scorer, as plain functions of the rig's joints, the feeler leaf's pose and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyStates of the rig (the fixed clamp, dog, stand and tray, with two joints: ``sensor``, a slider along
x in metres toward the dog, and ``lever``, a hinge in radians, 0 open and LEVER_MAX shut) and of the leaf (base frame
at the centre of its tip's bottom edge, the blade along -y, its flat face toward x). Every reading is a joint value or a
pose; the arms are read only for the let-go check.

    GAP        g = dog face minus sensor face = 12.0 mm - the sensor's slide (the sensor's face is its slide link)
    CLOSED     the lever within 5 degrees of its stop
    GOOD       |g - 1.00 mm| <= 0.25 mm
    RANGE      0.50 mm <= g <= 1.62 mm (the maker's secured operating distance, 0.81 x Sn)
    POOR       0.30 mm <= g <= 4.0 mm
    LEAF OUT   every point of the leaf at least 30 mm (top view) from the gap's centre line
    IN TRAY    the leaf's every point over the tray's footprint, lowest point within 10 mm of the tray's floor
    RELEASED   on every step of the last 1 s both finger zones are 30 mm or more from the sensor, the lever and the leaf
    STILL      over the last 1 s: the slide moved under 0.2 mm, the lever under 1 degree, the leaf under 1 mm, 0.3 deg
    CRASH      latched: the lever has been shut with g under 0.30 mm (the first cycle drives the dog into the face)
    LOCKED ON THE LEAF  judged on the step the lever last came shut (into its last 5 degrees after being 20 degrees
               or more open): the leaf's blade stood upright and flat (10 degrees) between the face and the dog, over
               6 mm or more of the face's round, each of its faces within 0.25 mm of the face it meets, and g within
               0.25 mm of the leaf's 1.00 mm (the clamp was locked on the gauge; a right gap without that: at most 0.5)

    0      CRASH, or not released and still, or a state that cannot be read
    1.0    CLOSED, GOOD, LOCKED ON THE LEAF, LEAF OUT, IN TRAY
    0.75   CLOSED, GOOD, LOCKED ON THE LEAF, LEAF OUT, not in the tray
    0.5    CLOSED, GOOD, leaf still in the gap or not LOCKED ON THE LEAF; or CLOSED, RANGE (not GOOD), LEAF OUT
    0.25   CLOSED, POOR (otherwise); or lever open with GOOD
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

RIG, LEAF = "rig", "leaf"
PARTS = (RIG, LEAF)
SENSOR, LEVER = "sensor", "lever"
MOVERS = (SENSOR, LEVER)


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


def _corners(size, centre):
    c = np.asarray(centre, dtype=float)
    return np.array([c + np.array([sx, sy, sz]) * np.asarray(size) / 2
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


# The leaf's corners in its own frame (origin at the tip's bottom-edge centre, blade along -y, flat face toward x).
LEAF_PTS = np.vstack([_corners((L.LF_T, L.LF_L, L.LF_W), (0.0, -L.LF_L / 2, L.LF_W / 2)),
                      _corners((L.LF_H_X, L.LF_H_Y, L.LF_H_Z), (0.0, L.LF_H_YC, L.LF_H_Z / 2))])
# The blade sampled every 2 mm along its length (a corner test would miss a blade's middle over the gap)
LEAF_SAMPLES = np.array([(0.0, y, z) for y in np.linspace(-L.LF_L, 0.0, 102) for z in (0.0, L.LF_W)])
LEAF_ALL = np.vstack([LEAF_PTS, LEAF_SAMPLES])


@dataclass(frozen=True)
class Reading:
    """One step (m, rad). Positions in the world frame with the table top at z 0."""
    q_sensor: float
    q_lever: float
    leaf_pos: np.ndarray
    leaf_rot: np.ndarray
    leaf_pts: np.ndarray


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        rig, leaf = states.get(RIG), states.get(LEAF)
        if rig is None or leaf is None:
            return None
        qs, ql = float(rig.joints[SENSOR]), float(rig.joints[LEVER])
        o = np.asarray(leaf.origin, dtype=float).ravel()
        r = rotation(leaf.quat_xyzw)
        if o.size != 3 or not (math.isfinite(qs) and math.isfinite(ql) and np.all(np.isfinite(o))
                               and np.all(np.isfinite(r))):
            return None
        o = o - np.array([0.0, 0.0, L.TOP])
        return Reading(qs, ql, o, r, o + LEAF_ALL @ r.T)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


# -- the readings ------------------------------------------------------------------------------------------------------
def gap(r: Reading) -> float:
    return L.gap_of(r.q_sensor)


def lever_closed(r: Reading) -> bool:
    return bool(r.q_lever >= L.LEVER_MAX - L.LEVER_CLOSED_BAND)


def is_good(g: float) -> bool:
    return bool(abs(g - L.GAP_SET) <= L.GOOD_TOL + L.EPS)


def in_range(g: float) -> bool:
    return bool(L.RANGE_LO - L.EPS <= g <= L.RANGE_HI + L.EPS)


def is_poor(g: float) -> bool:
    return bool(L.POOR_LO - L.EPS <= g <= L.POOR_HI + L.EPS)


def leaf_out(r: Reading) -> bool:
    d = np.hypot(r.leaf_pts[:, 0] - L.GAP_X, r.leaf_pts[:, 1] - L.GAP_Y)
    return bool(float(np.min(d)) >= L.OUT_OF_GAP - L.EPS)


def leaf_in_tray(r: Reading) -> bool:
    p = r.leaf_pts
    hx, hy = L.TRAY_OUT[0] / 2, L.TRAY_OUT[1] / 2
    over = bool(np.all(np.abs(p[:, 0] - L.TRAY_X) <= hx) and np.all(np.abs(p[:, 1] - L.TRAY_Y) <= hy))
    return bool(over and float(np.min(p[:, 2])) <= L.TRAY_FLOOR + L.IN_TRAY_LOW)


def leaf_between_faces(r: Reading) -> bool:
    """A point of the blade over the sensor's face (within its 12 mm round, at the axis height's blade) and between the
    face and the dog: the leaf is in the gap."""
    p = r.leaf_pts
    face = L.S_FACE0 + r.q_sensor
    near = (np.abs(p[:, 1] - L.GAP_Y) <= L.S_R) & (p[:, 0] >= face - L.EPS) & (p[:, 0] <= L.X_DOG + L.EPS) & (
        np.abs(p[:, 2] - L.AXIS_Z) <= L.S_R)
    return bool(np.any(near))


# The blade's mid-plane sampled every 1 mm along its length and across its width (for the lock-on-the-leaf check)
BLADE_GRID = np.array([(0.0, y, z) for y in np.linspace(-L.LF_L, 0.0, 204) for z in np.linspace(0.0, L.LF_W, 14)])


def blade_on_both_faces(r: Reading) -> bool:
    """The blade stands upright and flat between the sensor's face and the dog, touching both: its flat faces within
    10 degrees of square to the sensor's axis and its width within 10 degrees of vertical; the part of its mid-plane
    over the face's 12 mm round spans 6 mm or more of the face across; and over that part each of its two flat faces
    is within 0.25 mm of the face it meets (the sensor's face on one side, the dog on the other)."""
    nx = abs(float(r.leaf_rot[0, 0]))  # the blade's face normal (its x axis) along the sensor's axis
    up = float(r.leaf_rot[2, 2])  # the blade's width (its z axis) along world z
    cos10 = math.cos(math.radians(L.UPRIGHT_DEG))
    if nx < cos10 or up < cos10:
        return False
    pts = r.leaf_pos + BLADE_GRID @ r.leaf_rot.T
    dy, dz = pts[:, 1] - L.GAP_Y, pts[:, 2] - L.AXIS_Z
    over = pts[np.hypot(dy, dz) <= L.S_R]
    if over.shape[0] == 0 or float(np.ptp(over[:, 1])) < L.FACE_COVER - L.EPS:
        return False
    half = L.LF_T / (2.0 * nx)
    face = L.S_FACE0 + r.q_sensor
    to_face = over[:, 0] - half - face  # the blade's near face to the sensor's face
    to_dog = L.X_DOG - (over[:, 0] + half)  # the blade's far face to the dog
    tol = L.FILL_TOL + L.EPS
    return bool(np.all(np.abs(to_face) <= tol) and np.all(np.abs(to_dog) <= tol))


def leaf_fills_gap(r: Reading) -> bool:
    """The leaf in the gap with the face on it: the gap within 0.25 mm of the leaf's 1.00 mm (a feeler that drags) and
    the blade upright, flat and on both faces."""
    return bool(leaf_between_faces(r) and abs(gap(r) - L.LF_T) <= L.FILL_TOL + L.EPS and blade_on_both_faces(r))


def lever_reopened(r: Reading) -> bool:
    return bool(r.q_lever <= L.LEVER_MAX - L.LEVER_REARM)


def crash(r: Reading) -> bool:
    return bool(lever_closed(r) and gap(r) < L.CRASH_GAP)


def tier(latched: bool, settled: bool, closed: bool, g: float, out: bool, tray: bool, witnessed: bool = True) -> float:
    if latched or not settled or not math.isfinite(g):
        return 0.0
    if closed:
        if is_good(g):
            if out and witnessed:
                return 1.0 if tray else 0.75
            return 0.5
        if in_range(g) and out:
            return 0.5
        if is_poor(g):
            return 0.25
        return 0.0
    return 0.25 if is_good(g) else 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 3)


class Tracker:
    """The latch (CRASH), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"CRASH": None, "LOCKED_ON_LEAF": None}
        self.armed = True  # the lever is open (or re-opened 20 degrees): its next shutting is judged
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["CRASH"] is not None

    @property
    def witnessed(self) -> bool:
        return self.latch["LOCKED_ON_LEAF"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (abs(r.q_sensor) <= 0.0005 and abs(r.q_lever) <= 0.05
                and float(np.linalg.norm(r.leaf_pos - np.array([L.LEAF_X0, L.LEAF_Y0, 0.0]))) <= 0.01)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return (float(np.linalg.norm(now.leaf_pos - prev.leaf_pos)) >= L.RESET_JUMP
                or abs(now.q_sensor - prev.q_sensor) >= 0.005 or abs(now.q_lever - prev.q_lever) >= 0.3)

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
            # A real trial reset reaches the tracker through the reset hooks (rigstate.on_reset, mechkit on_reset),
            # which clear everything. This jump check is only a fallback: it restarts the timed windows but never
            # clears a latch, so a robot cannot erase a crash by snapping the parts back to their start marks.
            self.hist.clear()
            self.released.clear()
            self.armed = True
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        if self.latch["CRASH"] is None and crash(r):
            self.latch["CRASH"] = self.steps
        if lever_closed(r) and self.armed:  # the lever has just come shut: was the clamp locked on the leaf?
            self.armed = False
            self.latch["LOCKED_ON_LEAF"] = self.steps if leaf_fills_gap(r) else None
        elif lever_reopened(r):
            self.armed = True
        ok, hands = False, None
        try:
            rig, leaf = states[RIG], states[LEAF]

            def dist(points):
                d_rig = rig.distance(points, bodies=MOVERS)
                return np.minimum(d_rig, leaf.distance(points))

            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            hands = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(r, hands)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move of the slide (m), the lever (deg), the leaf (m, deg) from where each was
        at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        slide = lever = move = turn = 0.0
        for x in h[1:]:
            slide = max(slide, abs(x.q_sensor - first.q_sensor))
            lever = max(lever, math.degrees(abs(x.q_lever - first.q_lever)))
            move = max(move, float(np.linalg.norm(x.leaf_pos - first.leaf_pos)))
            turn = max(turn, turn_deg(first.leaf_rot, x.leaf_rot))
        return {"slide": slide, "lever": lever, "move": move, "turn": turn}

    def judge(self, r: Reading, hands) -> Result:
        g = gap(r)
        closed, out, tray = lever_closed(r), leaf_out(r), leaf_in_tray(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["slide"] <= L.STILL_SLIDE + L.EPS and m["lever"] <= L.STILL_LEVER_DEG + L.EPS
                     and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, closed, g, out, tray, self.witnessed)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch), "locked_on_leaf": self.witnessed,
            "leaf_in_gap": leaf_between_faces(r),
            "gap_mm": _mm(g), "lever_deg": round(math.degrees(r.q_lever), 2), "lever_closed": closed,
            "good": is_good(g), "in_range": in_range(g), "leaf_out": out, "leaf_in_tray": tray,
            "leaf_mm": [round(float(c) * 1000.0, 2) for c in r.leaf_pos],
            "leaf_tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(r.leaf_rot[2, 2]))))), 1),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(hands),
            "motion": None if m is None else {"slide_mm": _mm(m["slide"]), "lever_deg": round(m["lever"], 3),
                                              "leaf_mm": _mm(m["move"]), "leaf_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
