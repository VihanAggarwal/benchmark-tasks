"""The scorer, as plain functions of the two doors' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads the mechkit AssemblyStates of the front door (loose, with the bar pull) and the rear door
(loose, hung at the start). Both have their base frame at the middle of the door's underside, x across the 22 mm
thickness (+x toward the cabinet's back), y along the 320 mm width, z up. Every reading is a pose; the arms are read
only for the let-go check.

    TOP IN GROOVE   all four corners of the front door's top edge between the front groove's walls (+-2 mm) and 6 mm
                    or more above its mouth (z 316 mm or higher), not above the roof (+2 mm)
    FOOT DOWN       every corner of the door's underside within 3 mm of the floor (z 37 to 43 mm): the foot
                    strips straddle the rail
    FOOT ON RAIL    (a reading only) the underside's highest corner 3 to 9 mm over the floor: resting on the rail top
    RAIL IN GROOVE  both ends of the foot groove's centre line within 5 mm of the rail's centre line (x 387 mm): the
                    rail lies inside the foot groove (its strips stop it at 4 mm); a foot down beside the rail is 14 mm
                    or more off
    PULL OUT        the door's outward normal (the pull's side) within 60 degrees of facing the robot (-x)
    FACE            the door's centre plane at mid-height within 7 mm of the front track's centre line
    TILT            the angle between the door's up axis and the vertical
    REAR KEPT       the rear door moved no more than 10 mm along y and 5 mm along x, tilted under 4 degrees
    RELEASED        on every step of the last 1 s both finger zones are 30 mm or more from the front door
    STILL           over the last 1 s each door moved under 5 mm and turned under 5 degrees (visible motion)
    F1 FELL         latched: the rear door's lowest point below z 30 mm, or tilted over 10 degrees, or moved out of
                    its track (its centre more than 20 mm in x); also not finite

    0      F1 ever fired, or not released and still (a held door earns nothing), or a state that is not finite
    1.0    top in the groove, foot down, rail in the foot groove, face within 7 mm, tilt under 3 degrees, inside the
           opening, pull out, rear door kept
    0.5    top in the groove, foot down, tilt under 8 degrees, pull out (not 1.0: tilted 3 to 8 degrees, the foot on
           the floor beside the rail, the centre plane more than 7 mm off the track, or an end outside the opening)
    0.25   top in the groove, otherwise (the foot up on the rail top, or out: leaning; the door turned round, pull
           inward); also anything above with the rear door moved
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

FRONT, REAR = "front_door", "rear_door"
PARTS = (FRONT, REAR)


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


_HT, _HW, _G = L.DOOR_T / 2, L.DOOR_W / 2, L.FOOT_GROOVE_W / 2
# the door's local points: the top edge's four corners, the underside's eight (two strips' bottom corners)
TOP_PTS = np.array([(sx * _HT, sy * _HW, L.DOOR_H) for sx in (-1, 1) for sy in (-1, 1)])
FOOT_PTS = np.array([(x, sy * _HW, 0.0) for x in (-_HT, -_G, _G, _HT) for sy in (-1, 1)])
MID_PT = np.array([0.0, 0.0, L.DOOR_H / 2])
GROOVE_PTS = np.array([(0.0, sy * _HW, 0.0) for sy in (-1, 1)])  # the foot groove's centre line at both ends
ALL_PTS = np.vstack([TOP_PTS, FOOT_PTS])


@dataclass(frozen=True)
class Reading:
    """One step (m). Positions in the world frame with the table top at z 0."""
    pos: dict  # door -> base frame origin (the middle of its underside)
    rot: dict  # door -> rotation
    top: dict  # door -> (4, 3) the top edge's corners
    foot: dict  # door -> (8, 3) the underside's corners
    mid: dict  # door -> mid-height point


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, top, foot, mid = {}, {}, {}, {}, {}
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
            pos[name], rot[name] = o, r
            top[name], foot[name], mid[name] = o + TOP_PTS @ r.T, o + FOOT_PTS @ r.T, o + r @ MID_PT
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, top, foot, mid)


# -- the readings ----------------------------------------------------------------------------------------------------
def top_in_groove(r: Reading, door: str = FRONT) -> bool:
    t = r.top[door]
    lo, hi = L.FRONT_GROOVE
    return bool(np.all(t[:, 0] >= lo - 0.002) and np.all(t[:, 0] <= hi + 0.002)
                and np.all(t[:, 2] >= L.TRACK_BOTTOM + L.IN_GROOVE - L.EPS) and np.all(t[:, 2] <= L.ROOF + 0.002)
                and np.all(np.abs(t[:, 1]) <= L.OPEN_HALF + 0.002))


def foot_height(r: Reading, door: str = FRONT) -> float:
    """The highest corner of the underside over the floor (m)."""
    return float(np.max(r.foot[door][:, 2])) - L.FLOOR_TOP


def foot_down(r: Reading, door: str = FRONT) -> bool:
    """Every corner of the underside within 3 mm of the floor's top (above or below)."""
    z = r.foot[door][:, 2] - L.FLOOR_TOP
    return bool(np.all(np.abs(z) <= L.FOOT_DOWN + L.EPS))


def foot_on_rail(r: Reading, door: str = FRONT) -> bool:
    h = foot_height(r, door)
    return bool(L.FOOT_DOWN < h <= L.FOOT_RAIL_TOP + L.EPS)


def face_offset(r: Reading, door: str = FRONT) -> float:
    """The door's centre plane at mid-height (11 mm behind its front face) from the front track's centre line."""
    return float(r.mid[door][0]) - L.X_F


def rail_offset(r: Reading, door: str = FRONT) -> float:
    """The larger x offset (m) of the foot groove's centre line, at either end of the door, from the rail's."""
    ends = r.pos[door] + GROOVE_PTS @ r.rot[door].T
    return float(np.max(np.abs(ends[:, 0] - L.X_F)))


def rail_in_groove(r: Reading, door: str = FRONT) -> bool:
    """The front rail inside the door's foot groove along its whole length (read with foot down)."""
    return bool(rail_offset(r, door) <= L.RAIL_IN + L.EPS)


def pull_out(r: Reading, door: str = FRONT) -> bool:
    """The pull's side faces the robot: the door's outward normal (base -x) within 60 degrees of world -x."""
    return bool(float(r.rot[door][0, 0]) >= L.FACING_COS)


def inside_opening(r: Reading, door: str = FRONT) -> bool:
    pts = np.vstack([r.top[door], r.foot[door]])
    return bool(np.all(np.abs(pts[:, 1]) <= L.OPEN_HALF + 0.002))


def rear_moved(r: Reading) -> dict:
    o = r.pos[REAR] - np.array([L.REAR_X, L.REAR_Y, L.FLOOR_TOP])
    return {"dx": float(o[0]), "dy": float(o[1]), "dz": float(o[2]), "tilt": tilt_deg(r.rot[REAR])}


def rear_kept(r: Reading) -> bool:
    m = rear_moved(r)
    return bool(abs(m["dy"]) <= L.REAR_Y_TOL + L.EPS and abs(m["dx"]) <= L.REAR_X_TOL + L.EPS
                and m["tilt"] < L.REAR_TILT_KEPT)


def rear_fell(r: Reading) -> bool:
    m = rear_moved(r)
    return bool(float(np.min(r.foot[REAR][:, 2])) < L.FALL_Z or m["tilt"] > L.REAR_TILT_FELL
                or abs(m["dx"]) > 0.020 or abs(m["dy"]) > 0.060)


def tier(latched: bool, settled: bool, r: Reading) -> float:
    if latched or not settled:
        return 0.0
    if not top_in_groove(r):
        return 0.0
    tilt = tilt_deg(r.rot[FRONT])
    down = foot_down(r)
    level = 0.25
    if down and tilt < L.TILT_PART and pull_out(r):
        level = 0.5
        if (tilt < L.TILT_FULL and rail_in_groove(r) and abs(face_offset(r)) <= L.FACE_TOL + L.EPS
                and inside_opening(r)):
            level = 1.0
    if level > 0.25 and not rear_kept(r):
        level = 0.25
    return level


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
    def at_start(r: Reading) -> bool:
        f0 = np.array([L.FRONT_START_X, L.FRONT_START_Y, 0.0])
        r0 = np.array([L.REAR_X, L.REAR_Y, L.FLOOR_TOP])
        return (float(np.linalg.norm(r.pos[FRONT] - f0)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[REAR] - r0)) <= L.START_TOL)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The doors put back on their starting marks with a jump no trial makes in one step (a reset)."""
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
        if self.latch["F1"] is None and rear_fell(r):
            self.latch["F1"] = self.steps
        try:
            dist = states[FRONT].distance  # the door being hung: the idle hand's ready pose is near the rear door
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
        """Over the last 1 s: the largest move (m) and turn (deg) of either door from where it was at the window's
        start; None before 1 s of valid readings."""
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
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] < L.STILL_MOVE and m["turn"] < L.STILL_TURN_DEG)
        level = tier(self.latched, released and still, r)
        rm = rear_moved(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "top_in_groove": top_in_groove(r), "foot_height_mm": _mm(foot_height(r)),
            "foot_down": foot_down(r), "foot_on_rail": foot_on_rail(r),
            "rail_in_groove": rail_in_groove(r), "rail_offset_mm": _mm(rail_offset(r)), "pull_out": pull_out(r),
            "face_offset_mm": _mm(face_offset(r)), "tilt_deg": round(tilt_deg(r.rot[FRONT]), 2),
            "top_min_z_mm": _mm(float(np.min(r.top[FRONT][:, 2]))), "inside_opening": inside_opening(r),
            "rear_kept": rear_kept(r), "rear_dy_mm": _mm(rm["dy"]), "rear_dx_mm": _mm(rm["dx"]),
            "rear_tilt_deg": round(rm["tilt"], 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "front_mm": _mmv(r.pos[FRONT]), "rear_mm": _mmv(r.pos[REAR]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
