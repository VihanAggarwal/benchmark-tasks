"""The scorer, as plain functions of the base's, the lid's and the product's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the base (base frame at the centre of its footprint on the table), the lid (base frame
at the centre of its skirt's lower rim, z up through the top panel) and the product (base frame at its centre). Every
reading is a pose; the arms are read only for the let-go check.

    F1 FELL             latched: the lid or the base more than 10 mm below the table top, or the product's centre
                        outside the base's inside (it left the base)
    SEATED              the lid's top outer face within 1.5 mm of 42 mm over the table (the base's 40 mm plus the
                        panel's 2 mm)
    SQUARE              the lid's yaw within 1.5 degrees of the base's (modulo 180) and its top face tilted under 1.0
    CENTRED             the lid's centre within 3.5 mm of the base's (top view)
    ENGAGED             the lid upright (tilt under 10 degrees), its centre within 8 mm of the base's and its skirt's
                        lowest point 10 mm or more below the base's rim
    ON BASE             the lid upright within 30 degrees, its centre within 25 mm of the base's and its lowest point
                        at least 4 mm over the table (carried by the base)
    BOX STILL           the base within 20 mm of where it began (top view), its footing within 3 mm of the table, tilted
                        under 2 degrees
    RELEASED            on every step of the last 1 s both finger zones are 30 mm or more from all three parts
    STILL               over the last 1 s every part moved under 1 mm and turned under 0.3 degrees

    0      F1 ever fired, or not released and still, or the box not still
    1.0    seated, square, centred
    0.5    engaged otherwise
    0.25   on the base otherwise
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BASE, LID, PRODUCT = L.PARTS
PARTS = L.PARTS


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


def _corners(size, centre=(0.0, 0.0, 0.0)):
    c = np.asarray(centre, dtype=float)
    return np.array([c + np.array([sx, sy, sz]) * np.asarray(size) / 2
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


# Corner points in each part's own frame. The lid's first four are its rim (z 0), the last four its top face.
BASE_PTS = _corners((L.BASE_LX, L.BASE_LY, L.BASE_H), (0.0, 0.0, L.BASE_H / 2))
LID_PTS = np.array([(sx * L.LID_OX / 2, sy * L.LID_OY / 2, z) for z in (0.0, L.LID_H)
                    for sx in (-1, 1) for sy in (-1, 1)])
PRODUCT_PTS = _corners((L.PR_LX, L.PR_LY, L.PR_H))
LOCAL = {BASE: BASE_PTS, LID: LID_PTS, PRODUCT: PRODUCT_PTS}


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # part -> base frame origin
    rot: dict  # part -> rotation
    pts: dict  # part -> (N, 3) corner points


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, pts = {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            o = o - np.array([0.0, 0.0, L.TOP])
            pos[name], rot[name], pts[name] = o, r, o + LOCAL[name] @ r.T
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, pts)


# -- the readings ------------------------------------------------------------------------------------------------------
def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


def yaw_gap_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The yaw difference of two rectangles (deg), modulo 180 (a box looks the same turned half a turn), 0 to 90."""
    d = (yaw_deg(a) - yaw_deg(b)) % 180.0
    return min(d, 180.0 - d)


def centre(r: Reading, name: str) -> np.ndarray:
    """The middle of a part's footprint column: its frame turned to the box's mid height (top view uses x, y)."""
    h = L.LID_H if name == LID else (L.BASE_H if name == BASE else 0.0)
    return r.pos[name] + r.rot[name] @ np.array([0.0, 0.0, h / 2])


def lid_top_z(r: Reading) -> float:
    """Height of the lid's top outer face centre over the table (m)."""
    return float((r.pos[LID] + r.rot[LID] @ np.array([0.0, 0.0, L.LID_H]))[2])


def lid_offset(r: Reading) -> float:
    """Top-view distance (m) between the lid's centre and the base's."""
    return float(np.linalg.norm(centre(r, LID)[:2] - centre(r, BASE)[:2]))


def lid_rim_low(r: Reading) -> float:
    """The lowest of the lid's four rim corners (m over the table)."""
    return float(np.min(r.pts[LID][:4, 2]))


def lid_low(r: Reading) -> float:
    return float(np.min(r.pts[LID][:, 2]))


def fell(r: Reading) -> bool:
    return bool(float(np.min(r.pts[LID][:, 2])) < L.FALL_Z or float(np.min(r.pts[BASE][:, 2])) < L.FALL_Z
                or not product_in_base(r))


def product_in_base(r: Reading) -> bool:
    """The product's centre inside the base's inside box (taken in the base's own frame), above the floor and below
    the rim."""
    loc = (r.pos[PRODUCT] - r.pos[BASE]) @ r.rot[BASE]
    return bool(abs(loc[0]) <= L.INNER_LX / 2 + L.PRODUCT_MARGIN and abs(loc[1]) <= L.INNER_LY / 2 + L.PRODUCT_MARGIN
                and 0.0 <= loc[2] <= L.BASE_H)


def seated(r: Reading) -> bool:
    return bool(abs(lid_top_z(r) - L.SEAT_TOP) <= L.SEAT_TOL + L.EPS)


def square(r: Reading) -> bool:
    return bool(yaw_gap_deg(r.rot[LID], r.rot[BASE]) <= L.SQUARE_YAW_DEG + L.EPS
                and tilt_deg(r.rot[LID]) <= L.SQUARE_TILT_DEG + L.EPS)


def centred(r: Reading) -> bool:
    return bool(lid_offset(r) <= L.CENTRE_TOL + L.EPS)


def engaged(r: Reading) -> bool:
    return bool(r.rot[LID][2, 2] > 0.0 and tilt_deg(r.rot[LID]) <= L.ENGAGED_TILT_DEG
                and lid_offset(r) <= L.ENGAGED_CENTRE
                and L.BASE_H - lid_rim_low(r) >= L.ENGAGED_OVERLAP - L.EPS)


def on_base(r: Reading) -> bool:
    return bool(r.rot[LID][2, 2] > 0.0 and tilt_deg(r.rot[LID]) <= L.ON_BASE_TILT_DEG
                and lid_offset(r) <= L.ON_BASE_CENTRE and lid_low(r) >= L.ON_BASE_LOW - L.EPS)


def box_still(r: Reading, start_xy) -> bool:
    """The base where it began (top view), on the table, upright."""
    moved = float(np.linalg.norm(r.pos[BASE][:2] - np.asarray(start_xy, dtype=float)))
    return bool(moved <= L.BASE_MOVE + L.EPS and abs(float(r.pos[BASE][2])) <= L.BASE_LIFT + L.EPS
                and tilt_deg(r.rot[BASE]) <= L.BASE_TILT_DEG + L.EPS)


def tier(latched: bool, settled: bool, still_box: bool, is_on: bool, is_engaged: bool, is_seated: bool,
         is_square: bool, is_centred: bool) -> float:
    if latched or not settled or not still_box or not is_on:
        return 0.0
    if is_engaged and is_seated and is_square and is_centred:
        return 1.0
    if is_engaged:
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


class Tracker:
    """The latch (F1), the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return self.latch["F1"] is not None

    @staticmethod
    def start_xy() -> np.ndarray:
        return np.array([L.BASE_X, L.BASE_Y])

    @staticmethod
    def at_start(r: Reading) -> bool:
        lid0 = np.array([L.LID_X, L.LID_Y, 0.0])
        base0 = np.array([L.BASE_X, L.BASE_Y, 0.0])
        return (float(np.linalg.norm(r.pos[LID] - lid0)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[BASE] - base0)) <= L.START_TOL)

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
        if self.latch["F1"] is None and fell(r):
            self.latch["F1"] = self.steps
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
        s_ok, q_ok, c_ok = seated(r), square(r), centred(r)
        e_ok, o_ok = engaged(r), on_base(r)
        b_ok = box_still(r, self.start_xy())
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, b_ok, o_ok, e_ok, s_ok, q_ok, c_ok)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "lid_top_mm": _mm(lid_top_z(r)), "lid_rim_low_mm": _mm(lid_rim_low(r)),
            "lid_offset_mm": _mm(lid_offset(r)), "lid_yaw_gap_deg": round(yaw_gap_deg(r.rot[LID], r.rot[BASE]), 3),
            "lid_tilt_deg": round(tilt_deg(r.rot[LID]), 3), "base_tilt_deg": round(tilt_deg(r.rot[BASE]), 3),
            "seated": s_ok, "square": q_ok, "centred": c_ok, "engaged": e_ok, "on_base": o_ok, "box_still": b_ok,
            "product_in_base": product_in_base(r),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "base_mm": _mmv(r.pos[BASE]), "lid_mm": _mmv(r.pos[LID]), "product_mm": _mmv(r.pos[PRODUCT]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
