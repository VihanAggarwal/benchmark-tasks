"""The scorer and the flame rule, as plain functions of the parts' and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the candlestick, the snuffer and the vase (frames: layout.py).

    FLAME (the one declared rule) The taper is lit at the start. It goes out once the snuffer's bell has covered the
                wick for 1.5 s without a break: the bell mouth-down within 20 degrees, its mouth (rim plane) at or
                below the wick's tip and its axis within 8 mm of the wick (top view). Nothing else puts it out and
                nothing relights it. This is the simulator proxy only: a real trial reads the actual flame from the top
                and side videos (out from the first frame after which none is visible, through the last frame and for
                at least the last 3 s; seen again later = lit), and the bell is not scored there.
    MOVED       (latched) at any step the chamberstick's centre is 10 mm or more (top view) from where it stood at
                the first step, or 5 mm higher or lower, or it tips 10 degrees or more. The NCA: "Don't move a candle
                while the wax is liquid" and, after snuffing, "Don't touch or move the candle until the wax has
                completely cooled" (minutes, longer than a trial), so this counts at every step, lit or not.
    IN BOX      the vase's base centre inside the box's inner outline (top view) and its lowest point below the
                walls' top edge (104 mm).
    LIT ENTRY   (latched) the vase first came into the box while the flame was lit and within 12 in (305 mm, top
                view) of the box's outline.
    PACKED      in the box, standing on the packing sheet (base within 5 mm of it), upright within 10 degrees.
    RELEASED    on every step of the last 1 s both grippers' finger zones at least 3 cm from the vase.
    STILL       over the last 1 s the vase's base centre moved under 2 mm.
    OFF TABLE   the vase or the snuffer with its centre outside the table's outline, or 20 mm or more below its top.

    0      MOVED; or the flame is lit at the end; or the vase or snuffer off the table
    0.25   LIT ENTRY (the flame out at the end)
    0.5    the flame out, the vase not in the box
    0.75   the flame out before the vase came in, the vase in the box, but not all of packed, released and still
    1.0    the flame out before the vase came in, the vase packed, released and still     (held 1 s: success)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CANDLE, SNUFFER, VASE = "candlestick", "snuffer", "vase"
PARTS = (CANDLE, SNUFFER, VASE)


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(r: np.ndarray) -> float:
    """How far (deg) the body's z axis leans from straight up (180 for upside down)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


@dataclass(frozen=True)
class Reading:
    """One step (m and deg), world frame with the table top at z 0."""
    pos: dict  # part -> its base frame's origin
    rot: dict  # part -> rotation
    low: dict  # part -> lowest point above the table top
    centre: dict  # part -> bounding-box centre (x, y)


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    pos, rot, low, centre = {}, {}, {}, {}
    try:
        for name in PARTS:
            s = states.get(name)
            if s is None:
                return None
            o = np.asarray(s.origin, dtype=float).ravel()
            r = rotation(s.quat_xyzw)
            lo = np.asarray(s.low, dtype=float).ravel()
            c = np.asarray(s.pos, dtype=float).ravel()
            if o.size != 3 or lo.size != 3 or c.size != 3:
                return None
            if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r)) and np.all(np.isfinite(lo))
                    and np.all(np.isfinite(c))):
                return None
            pos[name] = o - np.array([0.0, 0.0, L.TOP])
            rot[name] = r
            low[name] = float(lo[2] - L.TOP)
            centre[name] = c[:2]
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    return Reading(pos, rot, low, centre)


def wick_tip(r: Reading) -> np.ndarray:
    """The wick's tip in the world (table top at 0), riding on the candlestick."""
    return r.pos[CANDLE] + r.rot[CANDLE] @ np.array([0.0, 0.0, L.WICK_TIP])


def cover(r: Reading) -> dict:
    """How the bell sits over the wick: axis offset (top view, m), mouth height over the wick tip (m), tilt (deg), and
    whether that counts as covering it."""
    tip = wick_tip(r)
    mouth = r.pos[SNUFFER]
    off = float(math.hypot(mouth[0] - tip[0], mouth[1] - tip[1]))
    dz = float(mouth[2] - tip[2])
    tilt = tilt_deg(r.rot[SNUFFER])
    ok = off <= L.COVER_XY + L.EPS and dz <= L.EPS and tilt <= L.COVER_TILT_DEG + L.EPS
    return {"off": off, "dz": dz, "tilt": tilt, "covers": bool(ok)}


def moved(r: Reading, ref: Reading) -> dict:
    d = r.pos[CANDLE] - ref.pos[CANDLE]
    xy, dz, tilt = float(math.hypot(d[0], d[1])), float(d[2]), tilt_deg(r.rot[CANDLE])
    return {"xy": xy, "dz": dz, "tilt": tilt,
            "moved": bool(xy >= L.MOVE_XY - L.EPS or abs(dz) >= L.MOVE_Z - L.EPS or tilt >= L.MOVE_TILT_DEG - L.EPS)}


def in_box(r: Reading) -> bool:
    p = r.pos[VASE]
    hx, hy = L.BOX_IN[0] / 2, L.BOX_IN[1] / 2
    inside = abs(p[0] - L.BOX_XY[0]) <= hx and abs(p[1] - L.BOX_XY[1]) <= hy
    return bool(inside and r.low[VASE] < L.BOX_TOP)


def packed(r: Reading) -> bool:
    return bool(in_box(r) and abs(r.pos[VASE][2] - L.PAPER_TOP) <= L.PACK_Z_TOL + L.EPS
                and tilt_deg(r.rot[VASE]) <= L.PACK_TILT_DEG + L.EPS)


def off_table(r: Reading) -> bool:
    for n in (SNUFFER, VASE):
        x, y = r.centre[n]
        if not (L.TABLE_X0 <= x <= L.TABLE_X1 and L.TABLE_Y0 <= y <= L.TABLE_Y1):
            return True
        if r.low[n] <= -L.OFF_TABLE_DROP:
            return True
    return False


def flame_near_box(r: Reading) -> bool:
    tip = wick_tip(r)
    return L.flame_box_distance((tip[0], tip[1])) < L.NEAR


def tier(moved_latch: bool, lit: bool, off: bool, lit_entry: bool, inside: bool, done: bool) -> float:
    if moved_latch or lit or off:
        return 0.0
    if lit_entry:
        return 0.25
    if not inside:
        return 0.5
    return 1.0 if done else 0.75


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The flame's state, the latches and the last 1 s windows. One per trial: reset() at every episode."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.hz = control_hz
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.out_steps = int(math.ceil(L.OUT_AFTER_S * control_hz - 1e-9))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.ref: Reading | None = None
        self.lit = True
        self.out_at: int | None = None
        self.covered = 0  # consecutive steps covered
        self.moved = False
        self.moved_at: int | None = None
        self.entered_at: int | None = None
        self.lit_entry = False
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a missing or broken state: 0 this step, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.covered = 0
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.ref is None:
            self.ref = r
        # the flame rule
        c = cover(r)
        if self.lit:
            self.covered = self.covered + 1 if c["covers"] else 0
            if self.covered >= self.out_steps:
                self.lit = False
                self.out_at = self.steps
        # the latches
        m = moved(r, self.ref)
        if m["moved"] and not self.moved:
            self.moved, self.moved_at = True, self.steps
        inside = in_box(r)
        if inside and self.entered_at is None:
            self.entered_at = self.steps
            self.lit_entry = bool(self.lit and flame_near_box(r))
        # the windows
        self.hist.append(r)
        self.prev = r
        try:
            dist = states[VASE].distance
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok = False
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, arms, dist, c, m, inside)
        return self.last

    def vase_motion(self) -> float | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(x.pos[VASE] - h[0].pos[VASE])) for x in h[1:])

    def judge(self, r: Reading, arms, dist, c: dict, m: dict, inside: bool) -> Result:
        off = off_table(r)
        pk = packed(r)
        released = len(self.released) == self.window and all(self.released)
        mv = self.vase_motion()
        still = bool(mv is not None and mv <= L.STILL_MOVE + L.EPS)
        level = tier(self.moved, self.lit, off, self.lit_entry, inside, pk and released and still)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        tip = wick_tip(r)
        details = {
            "valid_state": True, "score": level, "lit": self.lit, "out_at_step": self.out_at,
            "covered_steps": self.covered, "bell_off_mm": _mm(c["off"]), "bell_mouth_over_tip_mm": _mm(c["dz"]),
            "bell_tilt_deg": round(c["tilt"], 2), "moved": self.moved, "moved_at_step": self.moved_at,
            "candle_dxy_mm": _mm(m["xy"]), "candle_dz_mm": _mm(m["dz"]), "candle_tilt_deg": round(m["tilt"], 2),
            "wick_tip_mm": [_mm(v) for v in tip], "flame_box_mm": _mm(L.flame_box_distance((tip[0], tip[1]))),
            "vase_mm": [_mm(v) for v in r.pos[VASE]], "vase_tilt_deg": round(tilt_deg(r.rot[VASE]), 2),
            "in_box": inside, "entered_at_step": self.entered_at, "lit_entry": self.lit_entry, "packed": pk,
            "released_1s": bool(released), "still_1s": still, "vase_move_1s_mm": _mm(mv), "off_table": off,
            "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms),
            "snuffer_mm": [_mm(v) for v in r.pos[SNUFFER]],
        }
        return Result(level, bool(level == 1.0), details)
