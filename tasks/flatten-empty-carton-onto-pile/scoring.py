"""The scorer, as plain functions of the carton's, the stack's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the carton (base frame at the centre of its underneath panel A, three crease joints
"ab", "bc", "cd", positions in rad, 0 as drawn standing) and of each of the three flats of the stack (one box each).
Every reading is a pose or a joint position; the arms are read only for the let-go check.

    FLAT        every crease within 15 degrees of fully shut (interior angle 0) or fully open (180), the carton's
                thickness (highest point over lowest, all four panels) 12 mm or less, its footprint within
                540 mm (y) x 180 mm (x)
    ON STACK    the footprint's centre within 60 mm of the stack's centre (top view), its lowest point within 6 mm
                of the stack's top (a flat carton cannot also be tilted: 12 mm thick over 152 mm is under 3 degrees)
    ON, OFFSET  the same within 120 mm and 20 mm
    KEPT        no flat of the stack moved over 15 mm (top view) or 4 mm up or turned over 8 degrees from its start
    KNOCKED     latched: any flat moved over 40 mm or turned over 30 degrees
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the carton and every flat
    STILL       over the last 1 s nothing moved under 1 mm (any panel corner) or turned under 0.3 degrees (any panel
                or crease)

    0      KNOCKED ever, or not released and still (a carton held in a gripper earns nothing)
    1.0    flat, on the stack, stack kept
    0.75   flat, on the stack but offset or leaning, stack kept
    0.5    flat anywhere else, or on the stack with the stack not kept (moved under 40 mm)
    0.25   not flat, but no more than 110 mm tall (collapsed over half way)
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CARTON = "carton"
FLATS = tuple(f"flat_{i + 1}" for i in range(L.FLAT_N))
PARTS = (CARTON, *FLATS)


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


def _corners(centre, size):
    c = np.asarray(centre, dtype=float)
    return np.array([c + np.array([sx, sy, sz]) * np.asarray(size) / 2
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


PANEL_PTS = {b: _corners(L.PANELS[b][0], L.PANELS[b][1]) for b in L.BODIES}  # base frame as drawn
FLAT_START = [np.array([L.PILE_X, L.PILE_Y, L.flat_z(i)]) for i in range(L.FLAT_N)]
CARTON_START = np.array([L.CARTON_X, L.CARTON_Y, L.CARTON_ORIGIN_Z])


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pts: np.ndarray  # (32, 3) every panel's corners
    rotA: np.ndarray  # the underneath panel's rotation
    origin: np.ndarray  # its centre
    q: dict  # crease -> joint position (rad)
    body_rot: dict  # body -> rotation (turn checks)
    flat_pos: list
    flat_rot: list


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        c = states.get(CARTON)
        if c is None:
            return None
        pts, body_rot = [], {}
        for b in L.BODIES:
            r, p = c.link_pose(b)
            r, p = np.asarray(r, dtype=float), np.asarray(p, dtype=float).ravel()
            if p.size != 3 or not (np.all(np.isfinite(r)) and np.all(np.isfinite(p))):
                return None
            pts.append(p + (PANEL_PTS[b] - np.asarray(L.ANCHOR[b])) @ r.T)
            body_rot[b] = r
        q = {k: float(c.joints[k]) for k in L.CREASES}
        origin = np.asarray(c.origin, dtype=float).ravel()
        if origin.size != 3 or not all(math.isfinite(v) for v in q.values()):
            return None
        fp, fr = [], []
        for n in FLATS:
            s = states.get(n)
            if s is None:
                return None
            pos = np.asarray(s.pos, dtype=float).ravel()
            rot = rotation(s.quat_xyzw)
            if pos.size != 3 or not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))):
                return None
            fp.append(pos - np.array([0.0, 0.0, L.TOP]))
            fr.append(rot)
        pts = np.vstack(pts) - np.array([0.0, 0.0, L.TOP])
        origin = origin - np.array([0.0, 0.0, L.TOP])
        if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(origin))):
            return None
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pts, body_rot["base"], origin, q, body_rot, fp, fr)


# -- the readings -----------------------------------------------------------------------------------------------------
def interior(q: float) -> float:
    return L.interior_deg(q)


def crease_flat(q: float) -> bool:
    i = interior(q)
    return abs(i) <= L.FLAT_HINGE_DEG + 1e-9 or abs(i - 180.0) <= L.FLAT_HINGE_DEG + 1e-9


def thickness(r: Reading) -> float:
    return float(np.max(r.pts[:, 2]) - np.min(r.pts[:, 2]))


def footprint(r: Reading) -> tuple[float, float]:
    """(x extent, y extent) of the world-aligned box round every panel."""
    return (float(np.ptp(r.pts[:, 0])), float(np.ptp(r.pts[:, 1])))


def plan_centre(r: Reading) -> np.ndarray:
    lo, hi = np.min(r.pts[:, :2], axis=0), np.max(r.pts[:, :2], axis=0)
    return (lo + hi) / 2


def is_flat(r: Reading) -> bool:
    fx, fy = footprint(r)
    return bool(all(crease_flat(r.q[k]) for k in L.CREASES) and thickness(r) <= L.THICK_MAX + L.EPS
                and fx <= L.FOOT_X + L.EPS and fy <= L.FOOT_Y + L.EPS)


def on_stack(r: Reading, centre=L.ON_CENTRE, low=L.ON_LOW) -> bool:
    c = plan_centre(r)
    off = float(np.hypot(c[0] - L.PILE_X, c[1] - L.PILE_Y))
    lowest = float(np.min(r.pts[:, 2]))
    return bool(off <= centre + L.EPS and abs(lowest - L.PT) <= low + L.EPS)


def stack_moves(r: Reading) -> dict:
    """The largest move (top view), lift and turn (degrees) of any flat from where it started."""
    move = lift = turn = 0.0
    for i in range(L.FLAT_N):
        d = r.flat_pos[i] - FLAT_START[i]
        move = max(move, float(np.hypot(d[0], d[1])))
        lift = max(lift, abs(float(d[2])))
        turn = max(turn, turn_deg(np.eye(3), r.flat_rot[i]))
    return {"move": move, "lift": lift, "turn": turn}


def stack_kept(m: dict) -> bool:
    return bool(m["move"] <= L.KEEP_MOVE + L.EPS and m["lift"] <= L.KEEP_LIFT + L.EPS
                and m["turn"] <= L.KEEP_TURN_DEG + L.EPS)


def knocked(m: dict) -> bool:
    return bool(m["move"] > L.KNOCK_MOVE or m["turn"] > L.KNOCK_TURN_DEG)


def tier(latched: bool, settled: bool, flat: bool, on1: bool, on2: bool, kept: bool, height: float) -> float:
    if latched or not settled:
        return 0.0
    if flat:
        if on1 and kept:
            return 1.0
        if on2 and kept:
            return 0.75
        return 0.5
    if height <= L.THICK_HALF + L.EPS:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The latch (KNOCKED), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"KNOCKED": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["KNOCKED"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (float(np.linalg.norm(r.origin - CARTON_START)) <= L.START_TOL
                and all(abs(v) <= 0.02 for v in r.q.values())
                and stack_moves(r)["move"] <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return float(np.max(np.abs(now.pts - prev.pts))) >= L.RESET_JUMP

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
        m = stack_moves(r)
        if self.latch["KNOCKED"] is None and knocked(m):
            self.latch["KNOCKED"] = self.steps
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
        self.last = self.judge(r, m, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) of any panel corner or flat and turn (deg) of any panel, crease
        or flat from where it was at the window's start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.max(np.linalg.norm(x.pts - first.pts, axis=1))))
            for b in L.BODIES:
                turn = max(turn, turn_deg(first.body_rot[b], x.body_rot[b]))
            for k in L.CREASES:
                turn = max(turn, abs(math.degrees(x.q[k] - first.q[k])))
            for i in range(L.FLAT_N):
                move = max(move, float(np.linalg.norm(x.flat_pos[i] - first.flat_pos[i])))
                turn = max(turn, turn_deg(first.flat_rot[i], x.flat_rot[i]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, sm: dict, gap) -> Result:
        flat = is_flat(r)
        on1, on2 = on_stack(r), on_stack(r, L.ON2_CENTRE, L.ON2_LOW)
        kept = stack_kept(sm)
        height = thickness(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, flat, on1, on2, kept, height)
        fx, fy = footprint(r)
        c = plan_centre(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "flat": flat, "on_stack": on1, "on_stack_offset": on2, "stack_kept": kept,
            "interior_deg": {k: round(interior(r.q[k]), 1) for k in L.CREASES},
            "thickness_mm": _mm(height), "foot_x_mm": _mm(fx), "foot_y_mm": _mm(fy),
            "centre_off_mm": _mm(float(np.hypot(c[0] - L.PILE_X, c[1] - L.PILE_Y))),
            "lowest_over_stack_mm": _mm(float(np.min(r.pts[:, 2])) - L.PT), "tilt_deg": round(tilt_deg(r.rotA), 2),
            "stack_move_mm": _mm(sm["move"]), "stack_lift_mm": _mm(sm["lift"]), "stack_turn_deg": round(sm["turn"], 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "carton_mm": [round(float(v) * 1000.0, 2) for v in r.origin],
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
