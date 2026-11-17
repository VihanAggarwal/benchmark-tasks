"""The scorer, as plain functions of the pan's state and the arms' state. NumPy only; no simulator needed.

``Tracker.update(pan, arms)`` -> Result(score, success, details), fed one observation per control step. ``pan`` is the
mechkit AssemblyState of the quarter sheet pan (base frame at the centre of its underside, local x toward its leading
edge, z up). Every reading is the pan's pose; the arms are read only for the let-go check. The oven is fixed, so its
rails, walls and back wall are the constants in layout.py.

    ON RAILS    no corner of the pan's underside more than 3 mm below the lower rails' top or more than 12 mm above
                it, the pan's footprint passes over each rail's inner edge (an area of at least 2 mm x 50 mm
                over each rail), and no corner of it is inside a side wall or the back wall (2 mm allowed)
    IN          the pan's centre at least 30 mm inside the oven's front face
    LEVEL       the pan's up axis within 2 degrees of vertical
    HOME        no part of the pan in front of the oven's front face (the door could shut): with the cavity 3.7 mm
                deeper than the pan, its leading edge is then within 3.7 mm of the back wall
    NEAR        the pan's trailing edge no more than 40 mm out of the front face
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the pan
    STILL       over the last 1 s the pan moved under 5 mm and turned under 5 degrees (a real trial: no visible motion)
    FELL        latched: the pan's lowest point ever more than 20 mm below the table top (it fell off the table)

    0      FELL, or not released and still, or not on the rails, or not in
    1.0    on the rails, in, level, home
    0.5    on the rails, in, level, not home, near
    0.25   on the rails and in, otherwise (tilted, or more than 40 mm out)
    (the rules are tested in this order: the first that holds sets the score)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PAN = "pan"
_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # a face's corners, in order round it
UNDER_PTS = np.array([(sx * L.PAN_D / 2, sy * L.PAN_W / 2, 0.0) for sx, sy in _SIGNS])
TOP_PTS = np.array([(sx * L.PAN_D / 2, sy * L.PAN_W / 2, L.PAN_H) for sx, sy in _SIGNS])
PAN_PTS = np.vstack([UNDER_PTS, TOP_PTS])
RAIL_MIN_AREA = L.RAIL_OVERLAP * 0.05  # 2 mm of the pan's edge over a rail's inner edge, along 50 mm of it


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
    """Angle between the pan's up axis and the vertical (180 for a pan turned over)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: np.ndarray  # the base frame's origin: the centre of the pan's underside
    rot: np.ndarray
    pts: np.ndarray  # (8, 3) the pan's corners: underside 0..3, top 4..7


def read(pan) -> Reading | None:
    """A Reading, or None when the state is missing, malformed or not finite."""
    try:
        o = np.asarray(pan.origin, dtype=float).ravel()
        r = rotation(pan.quat_xyzw)
        if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
            return None
        o = o - np.array([0.0, 0.0, L.TOP])
        return Reading(o, r, o + PAN_PTS @ r.T)
    except (AttributeError, TypeError, ValueError):
        return None


# -- the readings ------------------------------------------------------------------------------------------------------
def underside(r: Reading) -> np.ndarray:
    return r.pts[:4]


def leading_gap(r: Reading) -> float:
    """From the pan's leading edge to the back wall (m; 0 or less: touching)."""
    return float(L.BACK_X - np.max(r.pts[:, 0]))


def trailing_out(r: Reading) -> float:
    """How far the pan's trailing edge is in front of the oven's front face (m; negative: inside)."""
    return float(L.OVEN_X - np.min(r.pts[:, 0]))


def fell(r: Reading) -> bool:
    return bool(np.min(r.pts[:, 2]) < -L.FALL_DROP)


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex polygon ``clip`` (both CCW)."""
    out = [np.asarray(v, dtype=float) for v in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        inp, out = out, []
        if not inp:
            return None
        prev = inp[-1]
        for cur in inp:
            sp, sc = side(prev), side(cur)
            if sc >= 0:
                if sp < 0:
                    out.append(prev + (cur - prev) * (sp / (sp - sc)))
                out.append(cur)
            elif sp >= 0:
                out.append(prev + (cur - prev) * (sp / (sp - sc)))
            prev = cur
    return np.array(out) if len(out) >= 3 else None


def rail_overlap_area(r: Reading, side: int) -> float:
    """The area (m2, top view) of the pan's underside footprint over the rail on one side (side +1: left, -1: right)."""
    foot = _ccw(underside(r)[:, :2])
    y0, y1 = (L.RAIL_Y, L.CW / 2) if side > 0 else (-L.CW / 2, -L.RAIL_Y)
    rail = np.array([(L.OVEN_X, y0), (L.BACK_X, y0), (L.BACK_X, y1), (L.OVEN_X, y1)])
    part = _clip(foot, _ccw(rail))
    if part is None:
        return 0.0
    a = _area(part)
    return float(a) if math.isfinite(a) else 0.0


def inside_walls(r: Reading) -> bool:
    """No corner of the pan's underside is inside a side wall or the back wall (2 mm allowed): a pose a rigid pan cannot
    take, so a simulator glitch is not scored as a pan on the rails."""
    for x, y, _ in underside(r):
        if x >= L.OVEN_X and (abs(y) > L.CW / 2 + 0.002 or x > L.BACK_X + 0.002):
            return False
    return True


def on_rails(r: Reading) -> bool:
    """The pan lies on the lower pair: its underside's corners within -3 mm to +12 mm of the lower rails' top, its
    footprint over both rails' inner edges, and inside the walls."""
    z = underside(r)[:, 2]
    if not (np.all(z >= L.RAIL1_TOP - L.RAIL_LOW) and np.all(z <= L.RAIL1_TOP + L.RAIL_HIGH)):
        return False
    if not inside_walls(r):
        return False
    return bool(all(rail_overlap_area(r, s) >= RAIL_MIN_AREA for s in (1, -1)))


def in_oven(r: Reading) -> bool:
    return bool(r.pos[0] >= L.OVEN_X + L.IN_CENTRE and r.pos[0] <= L.BACK_X)


def level(r: Reading) -> bool:
    return bool(tilt_deg(r.rot) <= L.LEVEL_DEG + 1e-9)


def home(r: Reading) -> bool:
    return bool(trailing_out(r) <= L.HOME_OUT + L.EPS)


def near(r: Reading) -> bool:
    return bool(trailing_out(r) <= L.NEAR_OUT + L.EPS)


def tier(latched: bool, settled: bool, rails: bool, inside: bool, lvl: bool, is_home: bool, is_near: bool) -> float:
    """The first rule that holds sets the score."""
    if latched or not settled or not rails or not inside:
        return 0.0
    if lvl and is_home:
        return 1.0
    if lvl and is_near:
        return 0.5
    return 0.25


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
    """The latch (FELL), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"FELL": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["FELL"] is not None

    @staticmethod
    def at_start(r: Reading) -> bool:
        p0 = np.array([L.PAN_CX, L.PAN_Y, 0.0])
        return bool(float(np.linalg.norm(r.pos - p0)) <= L.START_TOL and tilt_deg(r.rot) < 3.0)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The pan put back on its starting mark with a jump no trial makes in one step (a reset)."""
        return bool(cls.at_start(now) and float(np.linalg.norm(now.pos - prev.pos)) >= L.RESET_JUMP)

    def update(self, pan, arms) -> Result:
        self.steps += 1
        r = read(pan)
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
        ok, gap = False, None
        try:
            dist = pan.distance
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance or no arms: not released
            ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the largest move (m) and turn (deg) of the pan from where it was at the window's start;
        None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = 0.0
        for x in h[1:]:
            move = max(move, float(np.linalg.norm(x.pos - first.pos)))
            turn = max(turn, turn_deg(first.rot, x.rot))
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        rails, inside, lvl, is_home, is_near = on_rails(r), in_oven(r), level(r), home(r), near(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        score = tier(self.latched, released and still, rails, inside, lvl, is_home, is_near)
        z = underside(r)[:, 2]
        details = {
            "valid_state": True, "score": score, "latch": dict(self.latch),
            "on_rails": rails, "in_oven": inside, "level": lvl, "home": is_home, "near": is_near,
            "tilt_deg": round(tilt_deg(r.rot), 3), "yaw_deg": round(yaw_deg(r.rot), 2),
            "leading_gap_mm": _mm(leading_gap(r)), "trailing_out_mm": _mm(trailing_out(r)),
            "underside_min_mm": _mm(float(z.min()) - L.RAIL1_TOP),
            "underside_max_mm": _mm(float(z.max()) - L.RAIL1_TOP),
            "rail_overlap_mm2": [round(rail_overlap_area(r, s) * 1e6, 1) for s in (1, -1)],
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap), "pan_mm": _mmv(r.pos),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(score, bool(score == 1.0), details)
