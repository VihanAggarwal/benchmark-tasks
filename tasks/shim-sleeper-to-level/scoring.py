"""The scorer, as plain functions of the sleeper's, the shim's, the level's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the sleeper (base frame at its box's centre; the box is drawn turned ROLL0 about x, the
slope it rests at as found), the shim (base frame at the centre of its underside, local x toward the thin tip) and
the level (base frame at its body's centre, drawn turned ROLL0 like the sleeper). Every reading is a pose; the arms
are read only for the let-go check.

    F1 FELL             latched: the sleeper's or the level's lowest point more than 20 mm below the low pad's top
                        (off the pads), or the sleeper turned more than 30 degrees off its edge (onto its face)
    DIFF                the sleeper's end-to-end height difference: 610 mm x the sine of its long axis's slope
    ON PADS             the sleeper's centre line, at each pad's centre line, over that pad's top (top view), and
                        its lowest point no lower than 5 mm below the low pad's top
    SHIM UNDER          the shim lying on the low pad (lowest point within 5 mm of its top, tilt under 15 degrees),
                        its centre within the sleeper's length, its thin tip past the sleeper's front face
    FULL BEARING        the tip past the sleeper's REAR face and the butt still short of its front face: the shim
                        crosses the sleeper's whole 38 mm width
    LEVEL ON            the level's underside centre over the sleeper's top face (2 mm allowed), within 3 mm above it,
                        its long axis within 10 degrees of the sleeper's
    RELEASED            on every step of the last 1 s both finger zones are 30 mm or more from all three parts
    STILL               over the last 1 s every part moved under 1 mm and turned under 0.3 degrees

    0      F1 ever fired, or not released and still (a held sleeper earns nothing)
    1.0    on pads, shim under with full bearing, level on, |DIFF| <= 1 mm
    0.5    the same with |DIFF| <= 2 mm
    0.25   on pads and shim under, otherwise (too far in or not far enough, only under the front edge, level off)
    0      anything else
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SLEEPER, SHIM, LEVEL = "sleeper", "shim", "level"
PARTS = (SLEEPER, SHIM, LEVEL)
_AXIS0 = np.array([0.0, math.cos(L.ROLL0), math.sin(L.ROLL0)])  # the sleeper's (and level's) long axis as drawn
_UP0 = np.array([0.0, -math.sin(L.ROLL0), math.cos(L.ROLL0)])  # their up axis as drawn
_ACROSS0 = np.array([1.0, 0.0, 0.0])


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


# The sleeper's and the level's corners in their own frames as drawn: a box turned ROLL0 about x.
_RX0 = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(L.ROLL0), -math.sin(L.ROLL0)],
                 [0.0, math.sin(L.ROLL0), math.cos(L.ROLL0)]])
SLEEPER_PTS = _corners((L.SL_W, L.SL_L, L.SL_H)) @ _RX0.T
LEVEL_PTS = _corners((L.LV_W, L.LV_L, L.LV_H)) @ _RX0.T
SHIM_PTS = np.array([(sx * L.SH_L / 2, sy * L.SH_W / 2, z) for sx in (-1, 1) for sy in (-1, 1)
                     for z in (0.0, L.SH_BUTT if sx < 0 else L.SH_TIP)])


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
        for name, local in ((SLEEPER, SLEEPER_PTS), (SHIM, SHIM_PTS), (LEVEL, LEVEL_PTS)):
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                return None
            o = o - np.array([0.0, 0.0, L.TOP])
            pos[name], rot[name], pts[name] = o, r, o + local @ r.T
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, pts)


# -- the readings ------------------------------------------------------------------------------------------------------
def axis(r: Reading, name=SLEEPER) -> np.ndarray:
    return r.rot[name] @ _AXIS0


def up(r: Reading, name=SLEEPER) -> np.ndarray:
    return r.rot[name] @ _UP0


def end_diff(r: Reading) -> float:
    """The sleeper's end-to-end height difference (m; + when its +y end is higher)."""
    a = axis(r)
    return float(L.SL_L * a[2] * (1.0 if a[1] >= 0 else -1.0))


def level_pitch(r: Reading) -> float:
    """The level's slope along its long axis (rad; + when its +y end is higher): what its vial reads."""
    a = axis(r, LEVEL)
    return math.asin(max(-1.0, min(1.0, float(a[2])))) * (1.0 if a[1] >= 0 else -1.0)


def bubble_offset(pitch: float) -> float:
    """Where the bubble sits from the vial's centre (m, + toward the level's +y end): VIAL_R x pitch, toward the high
    end, stopped by the vial's ends."""
    return float(np.clip(L.VIAL_R * pitch, -L.BUBBLE_MAX, L.BUBBLE_MAX))


def fell(r: Reading) -> bool:
    low = min(float(np.min(r.pts[SLEEPER][:, 2])), float(np.min(r.pts[LEVEL][:, 2])))
    off_edge = math.degrees(math.acos(max(-1.0, min(1.0, float(up(r)[2])))))
    return low < L.LOW_H - L.FALL_DROP or off_edge > L.TIP_DEG


def sleeper_local(r: Reading, pts) -> np.ndarray:
    """World points (N, 3) in the sleeper's frame: x across (+ toward its rear face), y along, z up, all as drawn
    (the drawn roll taken out)."""
    return ((np.atleast_2d(pts) - r.pos[SLEEPER]) @ r.rot[SLEEPER]) @ _RX0


def on_pads(r: Reading) -> bool:
    a = axis(r)
    c = r.pos[SLEEPER]
    ok = True
    for pad_y, y0, y1 in ((L.LOW_Y, L.LOW_Y0, L.LOW_Y1), (L.FAR_Y, L.FAR_Y0, L.FAR_Y1)):
        if abs(a[1]) < 1e-6:
            return False
        p = c + a * (pad_y - c[1]) / a[1]
        ok = ok and L.PAD_X0 <= p[0] <= L.PAD_X1 and y0 - L.EPS <= p[1] <= y1 + L.EPS
        # the sleeper itself, not the endless line of its axis, must span the pad's whole width
        for edge in (y0, y1):
            ok = ok and abs((edge - c[1]) / a[1]) <= L.SL_L / 2
    return bool(ok and float(np.min(r.pts[SLEEPER][:, 2])) >= L.LOW_H - L.ON_PAD_LOW)


def shim_reading(r: Reading) -> dict:
    """The shim's tip and butt across the sleeper (m, in its frame: 0 is its centre plane, its faces at -+19 mm),
    its centre along it, its lowest point over the low pad's top and its tilt."""
    rs, o = r.rot[SHIM], r.pos[SHIM]
    tip = o + rs @ np.array([L.SH_L / 2, 0.0, 0.0])
    butt = o + rs @ np.array([-L.SH_L / 2, 0.0, 0.0])
    loc = sleeper_local(r, np.array([tip, butt, o]))
    return {"tip_x": float(loc[0, 0]), "butt_x": float(loc[1, 0]), "along": float(loc[2, 1]),
            "low": float(np.min(r.pts[SHIM][:, 2])) - L.LOW_H, "tilt": tilt_deg(rs),
            "tip_past_front": float(loc[0, 0] + L.SL_W / 2)}


def shim_under(s: dict) -> bool:
    return bool(abs(s["low"]) <= L.SHIM_ON_PAD and s["tilt"] <= L.SHIM_FLAT_DEG
                and abs(s["along"]) <= L.SL_L / 2 and s["tip_x"] > -L.SL_W / 2
                and s["butt_x"] < L.SL_W / 2)  # it crosses into the sleeper's footprint, not lying beside or behind it


def full_bearing(s: dict) -> bool:
    return bool(s["tip_x"] >= L.SL_W / 2 and s["butt_x"] <= -L.SL_W / 2)


def level_on(r: Reading) -> bool:
    """The level lying along the sleeper's top: its underside's centre over the top face, within 3 mm above it."""
    under = r.pos[LEVEL] - up(r, LEVEL) * (L.LV_H / 2)
    loc = sleeper_local(r, under)[0]
    over = abs(loc[0]) <= L.SL_W / 2 + L.LV_ON_EDGE and abs(loc[1]) <= L.SL_L / 2
    gap = loc[2] - L.SL_H / 2
    c = abs(float(axis(r, LEVEL) @ axis(r)))
    align = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return bool(over and -0.002 <= gap <= L.LV_ON_GAP and align <= L.LV_ON_ALIGN_DEG)


def tier(latched: bool, settled: bool, pads: bool, under: bool, full: bool, lv_on: bool, diff: float) -> float:
    if latched or not settled or not pads or not under:
        return 0.0
    if full and lv_on:
        if abs(diff) <= L.LEVEL_TOL + L.EPS:
            return 1.0
        if abs(diff) <= L.NEAR_TOL + L.EPS:
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
        s0 = np.array([L.SL_X, L.SL_C0[0], L.SL_C0[1]])
        h0 = np.array([L.SH_X0, L.SH_Y, L.LOW_H])
        return (float(np.linalg.norm(r.pos[SLEEPER] - s0)) <= L.START_TOL
                and float(np.linalg.norm(r.pos[SHIM] - h0)) <= L.START_TOL)

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
        s = shim_reading(r)
        under, full, lv_on, pads = shim_under(s), full_bearing(s), level_on(r), on_pads(r)
        diff = end_diff(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(self.latched, released and still, pads, under, full, lv_on, diff)
        details = {
            "valid_state": True, "score": level, "latch": dict(self.latch),
            "diff_mm": _mm(diff), "on_pads": pads, "shim_under": under, "full_bearing": full, "level_on": lv_on,
            "shim_tip_past_front_mm": _mm(s["tip_past_front"]), "shim_butt_x_mm": _mm(s["butt_x"]),
            "shim_along_mm": _mm(s["along"]), "shim_low_mm": _mm(s["low"]), "shim_tilt_deg": round(s["tilt"], 2),
            "shim_yaw_deg": round(yaw_deg(r.rot[SHIM]), 2),
            "level_pitch_deg": round(math.degrees(level_pitch(r)), 3),
            "bubble_mm": _mm(bubble_offset(level_pitch(r))),
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": _mm(gap),
            "sleeper_mm": _mmv(r.pos[SLEEPER]), "sleeper_yaw_deg": round(yaw_deg(r.rot[SLEEPER]), 2),
            "sleeper_low_end_mm": _mm(float(np.min(r.pts[SLEEPER][:, 2])) - L.LOW_H),
            "shim_mm": _mmv(r.pos[SHIM]), "level_mm": _mmv(r.pos[LEVEL]),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)
