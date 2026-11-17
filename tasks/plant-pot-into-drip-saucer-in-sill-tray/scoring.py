"""The scorer, as plain functions of the pot's, the saucer's and the neighbouring plant's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step (from
``task.success()``). It reads the mechkit AssemblyStates of the pot, the saucer and the neighbour (each one's base
frame sits at the centre of its underside, z up) and, for the let-go check only, the arms.

    SEATED      the pot's four foot corners, in the saucer's own frame, inside the well's 72 mm square (1 mm
                allowed) and between 2 mm below and 3 mm above the well's floor; the pot's axis within 5 degrees
                of the saucer's, its sides within 10 degrees of the saucer's (a square into a square), its axis
                within 8 mm of the saucer's axis
    IN THE BAY  the saucer's tilt under 3 degrees, its four base corners inside the bay's 102 mm square (1 mm
                allowed) and within 2 mm of the tray floor's top height (not riding a stop)
    DISTURBED   latched: the neighbour's base ever more than 3 mm from its mark, or tilted more than 3 degrees
    F1 TIPPED   latched: the pot's axis more than 30 degrees from vertical at any step
    F2 OFF      latched: any loose part's lowest point more than 10 mm below the table top (off the sill)
    F3 TOPPLED  latched: the neighbour's axis more than 30 degrees from vertical at any step
    RELEASED    on every step of the last 1 s both finger zones are 30 mm or more from the pot, the saucer and the
                neighbour
    STILL       over the last 1 s each of the three moved under 5 mm and turned under 5 degrees (visible motion)

    0      F1, F2 or F3 ever fired; or not (released and still): a held pot earns nothing
    1.0    seated, saucer in the bay, neighbour never disturbed
    0.5    seated, with the saucer's centre still inside the tray's edge, but the saucer out of the bay (shoved,
           tilted or ridden onto a stop) or the neighbour disturbed (moved or tilted, not toppled)
    0      seated, but the saucer's centre outside the tray's edge (taken off the sill tray)
    0.25   not seated: the pot upright (axis within 10 degrees of
           vertical), its foot's centre inside the tray's outline and at most 20 mm over the table top, and the
           neighbour never disturbed
    0      anything else (the pot still on the drainer or elsewhere on the sill)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

POT, SAUCER, NEIGHBOUR = "pot", "saucer", "neighbour"
PARTS = (POT, SAUCER, NEIGHBOUR)
_FEET = np.array(L.foot_corners())
_BASE = np.array(L.base_corners())
_MARK = {POT: np.array(L.POT_ORIGIN), SAUCER: np.array(L.SAU_ORIGIN), NEIGHBOUR: np.array(L.NB_ORIGIN)}


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n < 1e-9:
        return np.full((3, 3), np.nan)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _acos_deg(c: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


def tilt_deg(r: np.ndarray) -> float:
    """Angle between a body's z axis and the vertical (degrees)."""
    return _acos_deg(r[2, 2])


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    return _acos_deg((float(np.trace(r0.T @ r1)) - 1.0) / 2.0)


def yaw_deg(r: np.ndarray) -> float:
    return math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """One step (m): the parts' base-frame origins and rotations in the world, and each one's lowest point."""
    pos: dict
    rot: dict
    low: dict


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
            pos[name], rot[name], low[name] = o, r, lo
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    return Reading(pos, rot, low)


# -- the rules ---------------------------------------------------------------------------------------------------------
def seat_reading(r: Reading) -> dict:
    """The pot against the saucer, in the saucer's own frame (m, degrees)."""
    rp, op = r.rot[POT], r.pos[POT]
    rs, o_s = r.rot[SAUCER], r.pos[SAUCER]
    loc = (op + _FEET @ rp.T - o_s) @ rs
    dz = loc[:, 2] - L.SAU_FLOOR
    centre = (op - o_s) @ rs
    rel = rs.T @ rp
    return {
        "dz_min": float(dz.min()), "dz_max": float(dz.max()),
        "inside": bool(np.all(np.abs(loc[:, :2]) <= L.SAU_WELL / 2 + L.WELL_SLACK)),
        "centre": float(math.hypot(centre[0], centre[1])),
        "tilt": tilt_deg(rel), "yaw": L.yaw_fold_deg(yaw_deg(rel)),
    }


def seated(s: dict) -> bool:
    return bool(
        s["inside"] and L.SEAT_LOW - L.EPS <= s["dz_min"] and s["dz_max"] <= L.SEAT_HIGH + L.EPS
        and s["tilt"] <= L.SEAT_TILT_DEG + L.EPS and s["centre"] <= L.SEAT_CENTRE + L.EPS
        and abs(s["yaw"]) <= L.SEAT_YAW_DEG + L.EPS
    )


def bay_reading(r: Reading) -> dict:
    """The saucer against its bay (m, degrees): tilt, how far its worst base corner is outside the bay's square
    (negative: inside), and how far the base corners stand above the tray floor's top."""
    rs, o_s = r.rot[SAUCER], r.pos[SAUCER]
    corners = o_s + _BASE @ rs.T
    rel = corners[:, :2] - np.array(L.BAY_B)
    over = float(np.max(np.abs(rel)) - L.BAY / 2)
    z = corners[:, 2] - (L.TOP + L.TRAY_FLOOR)
    return {"tilt": tilt_deg(rs), "over": over, "z_min": float(z.min()), "z_max": float(z.max()),
            "offset": float(math.hypot(*(o_s[:2] - np.array(L.BAY_B))))}


def in_bay(b: dict) -> bool:
    return bool(b["tilt"] < L.SAU_FLAT_DEG and b["over"] <= L.BAY_SLACK + L.EPS
                and max(abs(b["z_min"]), abs(b["z_max"])) <= L.SAU_FLOOR_TOL + L.EPS)


def upright_in_tray(r: Reading) -> bool:
    """The pot standing (axis within 10 degrees of vertical) with its foot's centre over the tray's outline and no
    more than 20 mm over the table top (on the tray floor, a stop or the saucer, not perched on the other plant)."""
    o = r.pos[POT]
    over = abs(o[0] - L.TRAY_X) <= L.TRAY_OUTER[0] / 2 and abs(o[1] - L.TRAY_Y) <= L.TRAY_OUTER[1] / 2
    low = o[2] - L.TOP <= L.LOW_FOOT + L.EPS
    return bool(over and low and tilt_deg(r.rot[POT]) <= L.POT_UPRIGHT_DEG + L.EPS)


def saucer_on_tray(r: Reading) -> bool:
    """The saucer's centre inside the tray's raised edge (top view): still on the sill tray, maybe out of its bay."""
    o = r.pos[SAUCER]
    return bool(abs(o[0] - L.TRAY_X) <= L.TRAY_INNER[0] / 2 and abs(o[1] - L.TRAY_Y) <= L.TRAY_INNER[1] / 2)


def neighbour_drift(r: Reading) -> tuple[float, float]:
    """How far the neighbour's base has moved from its mark (m, in any direction, a lift included) and how far it is
    tilted (degrees)."""
    d = r.pos[NEIGHBOUR] - _MARK[NEIGHBOUR]
    return float(np.linalg.norm(d)), tilt_deg(r.rot[NEIGHBOUR])


def tier(latched: bool, settled: bool, is_seated: bool, bay_ok: bool, disturbed: bool, upright: bool,
         on_tray: bool = True) -> float:
    if latched or not settled:
        return 0.0
    if is_seated:
        if not on_tray:
            return 0.0  # the saucer (with the pot in it) taken off the tray
        return 1.0 if (bay_ok and not disturbed) else 0.5
    return 0.25 if (upright and not disturbed) else 0.0


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
    """The latches (F1 to F3, disturbed), the last 1 s of readings (stillness) and of the let-go check, and the reset
    fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch: dict = {"F1": None, "F2": None, "F3": None, "disturbed": None}
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @property
    def latched(self) -> bool:
        return any(self.latch[k] is not None for k in ("F1", "F2", "F3"))

    @staticmethod
    def at_start(r: Reading) -> bool:
        return all(float(np.linalg.norm(r.pos[n] - _MARK[n])) <= L.START_TOL for n in PARTS)

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
        if self.latch["F1"] is None and tilt_deg(r.rot[POT]) > L.TIP_DEG:
            self.latch["F1"] = self.steps
        if self.latch["F2"] is None and any(r.low[n] < L.TOP - L.OFF_DROP for n in PARTS):
            self.latch["F2"] = self.steps
        move, nb_tilt = neighbour_drift(r)
        if self.latch["F3"] is None and nb_tilt > L.TIP_DEG:
            self.latch["F3"] = self.steps
        if self.latch["disturbed"] is None and (move > L.NB_MOVE or nb_tilt > L.NB_TILT_DEG):
            self.latch["disturbed"] = self.steps
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
        s = seat_reading(r)
        b = bay_reading(r)
        is_seated, bay_ok = seated(s), in_bay(b)
        upright = upright_in_tray(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        disturbed = self.latch["disturbed"] is not None
        on_tray = saucer_on_tray(r)
        level = tier(self.latched, released and still, is_seated, bay_ok, disturbed, upright, on_tray)
        nb_move, nb_tilt = neighbour_drift(r)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "seated": is_seated, "in_bay": bay_ok, "saucer_on_tray": on_tray, "upright_in_tray": upright, "nb_disturbed": disturbed,
            "foot_dz_min_mm": _mm(s["dz_min"]), "foot_dz_max_mm": _mm(s["dz_max"]),
            "foot_inside_well": s["inside"], "pot_centre_off_mm": _mm(s["centre"]),
            "pot_tilt_vs_saucer_deg": round(s["tilt"], 2), "pot_yaw_vs_saucer_deg": round(s["yaw"], 2),
            "pot_tilt_deg": round(tilt_deg(r.rot[POT]), 2),
            "saucer_tilt_deg": round(b["tilt"], 2), "saucer_over_bay_mm": _mm(b["over"]),
            "saucer_z_range_mm": [_mm(b["z_min"]), _mm(b["z_max"])], "saucer_off_bay_centre_mm": _mm(b["offset"]),
            "nb_move_mm": _mm(nb_move), "nb_tilt_deg": round(nb_tilt, 2),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "pot_mm": _mmv(r.pos[POT]), "saucer_mm": _mmv(r.pos[SAUCER]), "neighbour_mm": _mmv(r.pos[NEIGHBOUR]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
