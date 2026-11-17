"""The scorer, as plain functions of the strainer's, the bowl's and the arms' state. NumPy only; no simulator.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads the mechkit AssemblyStates of the strainer (base frame at the centre of the ring on the
plane its hook strap's and handle's undersides lie in; see layout.py) and of the bowl (base frame at the centre of its
foot's underside). Every reading is a pose; the arms are read only for the let-go check. All the geometry is taken
in the BOWL's frame, so a bowl that was nudged is judged where it stands (and scores 0 if, at the end, it is not
upright and within 30 mm of its mark).

    HOOK        the bead lies in the hook's gap: the hook's inner leg face is 100 mm or less from the bowl's axis (inside
                the wall) and its outer leg face 110 mm or more (outside the bead), top view; the strap's underside is
                within 6 mm above the bead's top and no more than 2 mm below it; and it is the RIGHT rim: the hook's
                middle lies within 45 degrees of straight right (-y) of the bowl's axis (top view)
    HANDLE      the handle's underside, where it crosses the far bead (within 5 mm of the bead's centre line), is within
                6 mm above the bead's top and no more than 2 mm below it
    RING IN     the ring's centre within 30 mm of the bowl's axis (top view): the whole ring inside the rim's inner edge
                with 5 mm to spare
    MESH CLEAR  the mesh's underside (lowest and highest point) is 15 mm or more above the stock and at least 20 mm
                below the rim's top
    LEVEL       the long axis within 10 degrees of horizontal, the strainer within 15 degrees of upright, and the long
                axis passing within 20 mm of the bowl's axis (top view): across the bowl, not along a chord
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the strainer and the bowl
    STILL       over the last 1 s each part moved under 5 mm and turned under 5 degrees (no visible motion)
    BOWL OK     in the FINAL state only: the bowl's foot within 30 mm of its mark (top view), on the mat (not lifted
                20 mm) and tilted 10 degrees or less. The bowl may be pushed or knocked on the way and put right:
                only where it ends counts
    LATCH       F2 either part's lowest point ever more than 20 mm below the table top (fallen off the counter)

    0      the latch fired, the bowl not OK at the end, or not released and still (a held strainer earns nothing)
    1.0    HOOK, HANDLE, RING IN, MESH CLEAR and LEVEL
    0.5    RING IN and MESH CLEAR, the long axis within 25 degrees of horizontal (it rests across the rims, not hooked,
           hooked off line, or hooked over another part of the rim)
    0      anything else (on the table, leaning on the pot, the mesh down in the stock: the failure the task is about)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

STRAINER, BOWL = "strainer", "bowl"
PARTS = (STRAINER, BOWL)

_PHI = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
# the mesh's underside: points round the floor's edge and its centre, in the strainer's frame
MESH_PTS = np.array([(L.RING_R * math.cos(p), L.RING_R * math.sin(p), -L.RING_DEPTH) for p in _PHI]
                    + [(0.0, 0.0, -L.RING_DEPTH)])
HANDLE_T = np.linspace(L.HD_X0 + 0.010, L.HD_X1 - 0.010, 61)  # along the handle's centre line, its underside
HANDLE_PTS = np.array([(t, 0.0, 0.0) for t in HANDLE_T])
# a few points on the strainer's body for the 'fell' check (its lowest point is the mesh floor's underside)
BODY_PTS = np.vstack([MESH_PTS, [(L.HK_END, 0.0, -L.HK_LEG_OUT), (L.HD_X1, 0.0, 0.0)]])


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
    pos: dict  # part -> base frame origin
    rot: dict  # part -> rotation
    low: dict  # part -> lowest point of its shapes (z, table top at 0)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, low = {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            lo = float(np.asarray(s.low, dtype=float).ravel()[2])
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and math.isfinite(lo)):
                return None
            pos[name], rot[name], low[name] = o - np.array([0.0, 0.0, L.TOP]), r, lo - L.TOP
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    return Reading(pos, rot, low)


# -- the strainer in the bowl's frame --------------------------------------------------------------------------------
def to_bowl(r: Reading, local) -> np.ndarray:
    """Strainer-frame point(s) (3,) or (N, 3) in the bowl's frame (x, y across, z up from the foot's underside)."""
    p = np.atleast_2d(np.asarray(local, dtype=float))
    world = r.pos[STRAINER] + p @ r.rot[STRAINER].T
    return (world - r.pos[BOWL]) @ r.rot[BOWL]


def radial(p: np.ndarray) -> np.ndarray:
    return np.hypot(p[:, 0], p[:, 1])


def long_axis(r: Reading) -> np.ndarray:
    """The strainer's long axis (toward its handle), in the world."""
    return r.rot[STRAINER] @ np.array([1.0, 0.0, 0.0])


def axis_slope_deg(r: Reading) -> float:
    """Angle (degrees, 0..90) of the long axis from horizontal."""
    return abs(math.degrees(math.asin(max(-1.0, min(1.0, float(long_axis(r)[2]))))))


def strainer_tilt_deg(r: Reading) -> float:
    return tilt_deg(r.rot[STRAINER])


def ring_offset(r: Reading) -> float:
    """Distance of the ring's centre from the bowl's axis (top view, m)."""
    c = to_bowl(r, (0.0, 0.0, 0.0))[0]
    return float(math.hypot(c[0], c[1]))


def axis_miss(r: Reading) -> float:
    """How far the bowl's axis is from the strainer's long axis (top view, m); inf when the axis stands upright."""
    c = to_bowl(r, (0.0, 0.0, 0.0))[0]
    d = (to_bowl(r, (1.0, 0.0, 0.0))[0] - c)[:2]
    n = float(np.hypot(*d))
    if n < 1e-6:
        return math.inf
    d = d / n
    return abs(float(c[0] * d[1] - c[1] * d[0]))


def mesh_height(r: Reading) -> tuple[float, float]:
    """Lowest and highest point of the mesh's underside above the bowl's foot (m)."""
    z = to_bowl(r, MESH_PTS)[:, 2]
    return float(z.min()), float(z.max())


def mesh_clear(r: Reading) -> bool:
    lo, hi = mesh_height(r)
    return bool(lo >= L.STOCK_TOP_B + L.MESH_ABOVE_STOCK - L.EPS and hi <= L.BEAD_TOP_B - L.MESH_BELOW_RIM + L.EPS)


def in_stock(r: Reading) -> bool:
    """The ring over the bowl's floor and its mesh within 15 mm of the stock's top (resting in it or on its floor)."""
    lo, _ = mesh_height(r)
    return bool(ring_offset(r) <= L.IN_BOWL_CENTRE + L.EPS and lo < L.STOCK_TOP_B + L.MESH_ABOVE_STOCK)


def hook_reading(r: Reading) -> dict:
    """Where the hook's two leg faces and the strap's underside are against the rim (bowl frame, m)."""
    q = to_bowl(r, [(L.HK_IN, 0.0, 0.0), (L.HK_OUT, 0.0, 0.0), ((L.HK_IN + L.HK_OUT) / 2, 0.0, 0.0)])
    # which side of the bowl the hook is on, in the world's directions (top view): degrees from straight right (-y)
    w = r.pos[STRAINER] + r.rot[STRAINER] @ np.array([(L.HK_IN + L.HK_OUT) / 2, 0.0, 0.0]) - r.pos[BOWL]
    side = abs(math.degrees(math.atan2(float(w[0]), -float(w[1]))))
    return {"r_in": float(radial(q)[0]), "r_out": float(radial(q)[1]), "gap": float(q[2, 2] - L.BEAD_TOP_B),
            "side_deg": side}


def hook_on(h: dict) -> bool:
    return bool(h["r_in"] <= L.R_RIM_IN + L.EPS and h["r_out"] >= L.BEAD_OUT - L.EPS
                and -L.HOOK_GAP_DOWN - L.EPS <= h["gap"] <= L.HOOK_GAP_UP + L.EPS
                and h.get("side_deg", 0.0) <= L.RIGHT_RIM_DEG + L.EPS)


def handle_reading(r: Reading) -> dict:
    """The handle's underside where it crosses the far bead: the point of its centre line nearest the bead's centre
    line (radius 105 mm), its distance from that line and its height over the bead's top."""
    q = to_bowl(r, HANDLE_PTS)
    miss = np.abs(radial(q) - L.RIM_R)
    i = int(np.argmin(miss))
    return {"miss": float(miss[i]), "gap": float(q[i, 2] - L.BEAD_TOP_B)}


def handle_on(h: dict) -> bool:
    return bool(h["miss"] <= L.HANDLE_OVER + L.EPS and -L.HOOK_GAP_DOWN - L.EPS <= h["gap"] <= L.HOOK_GAP_UP + L.EPS)


def level(r: Reading) -> bool:
    return bool(axis_slope_deg(r) <= L.LEVEL_DEG + L.EPS and strainer_tilt_deg(r) <= 15.0 + L.EPS
                and axis_miss(r) <= L.AXIS_THROUGH + L.EPS)


def bowl_moved(r: Reading) -> float:
    """How far the bowl's foot is from its mark (m), counting a lift."""
    d = r.pos[BOWL] - np.array([L.BOWL_X, L.BOWL_Y, L.MAT_SIZE[2]])
    return float(np.linalg.norm(d[:2])) if abs(float(d[2])) < L.FALL_DROP else math.inf


def bowl_tilt_deg(r: Reading) -> float:
    return tilt_deg(r.rot[BOWL])


def fell(r: Reading) -> bool:
    """A part off the counter: its lowest point more than 20 mm below the table top (the bowl or the strainer
    knocked off the edge or through it)."""
    return bool(min(r.low[STRAINER], r.low[BOWL]) < -L.FALL_DROP)


def bowl_ok(r: Reading) -> bool:
    """The bowl's final state: within 30 mm of its mark, on the mat and upright (10 degrees or less)."""
    return bool(bowl_moved(r) <= L.BOWL_MOVED + L.EPS and bowl_tilt_deg(r) <= L.BOWL_TILT_DEG + L.EPS)


def tier(latched: bool, settled: bool, hook: bool, handle: bool, ring_in: bool, clear: bool, lvl: bool,
         rests: bool) -> float:
    if latched or not settled:
        return 0.0
    if hook and handle and ring_in and clear and lvl:
        return 1.0
    if ring_in and clear and rests:
        return 0.5
    return 0.0


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
    """The latches, the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F2": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(v is not None for v in self.latch.values())

    @staticmethod
    def at_start(r: Reading) -> bool:
        s0 = np.array([L.ST_X, L.ST_Y, L.RING_DEPTH])
        b0 = np.array([L.BOWL_X, L.BOWL_Y, L.MAT_SIZE[2]])
        return (float(np.linalg.norm(r.pos[STRAINER] - s0)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[BOWL] - b0)) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The parts put back on their starting marks with a jump no trial makes in one step (a reset)."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now.pos[n] - prev.pos[n])) >= L.RESET_JUMP for n in PARTS)

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
        if self.latch["F2"] is None and fell(r):
            self.latch["F2"] = self.steps
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
        """Over the last 1 s: the largest move (m) and turn (deg) of any part from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            for n in PARTS:
                move = max(move, float(np.linalg.norm(x.pos[n] - first.pos[n])))
                turn = max(turn, turn_deg(first.rot[n], x.rot[n]))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        hk, hd = hook_reading(r), handle_reading(r)
        hook, handle = hook_on(hk), handle_on(hd)
        off = ring_offset(r)
        ring_in = bool(off <= L.RING_CENTRE_MAX + L.EPS)
        clear = mesh_clear(r)
        lvl = level(r)
        rests = bool(axis_slope_deg(r) <= L.RESTS_TILT_DEG + L.EPS)
        sitting = in_stock(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        bowl_good = bowl_ok(r)
        score = tier(self.latched or not bowl_good, released and still, hook, handle, ring_in, clear, lvl, rests)
        lo, hi = mesh_height(r)
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch),
            "hook_on": hook, "handle_on": handle, "ring_in": ring_in, "mesh_clear": clear, "level": lvl,
            "rests": rests, "in_stock": sitting, "bowl_ok": bowl_good,
            "hook_r_in_mm": _mm(hk["r_in"]), "hook_r_out_mm": _mm(hk["r_out"]), "hook_gap_mm": _mm(hk["gap"]), "hook_side_deg": round(hk["side_deg"], 2),
            "handle_miss_mm": _mm(hd["miss"]), "handle_gap_mm": _mm(hd["gap"]),
            "ring_offset_mm": _mm(off), "axis_miss_mm": _mm(axis_miss(r)),
            "mesh_low_mm": _mm(lo), "mesh_high_mm": _mm(hi),
            "slope_deg": round(axis_slope_deg(r), 2), "strainer_tilt_deg": round(strainer_tilt_deg(r), 2),
            "strainer_yaw_deg": round(yaw_deg(r.rot[STRAINER]), 2),
            "bowl_moved_mm": _mm(bowl_moved(r)), "bowl_tilt_deg": round(bowl_tilt_deg(r), 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "strainer_mm": _mmv(r.pos[STRAINER]), "bowl_mm": _mmv(r.pos[BOWL]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(score, bool(score == 1.0), details)
