"""The scorer, as plain functions of the sign's state and the arms'. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyState of the sign: leaf A is the base (its frame's origin is its foot line's middle, its z axis up
the leaf, its x axis along the ridge, its outer face toward -y); leaf B is the link ``leaf_b``. Every reading is an
end-state reading: where the feet, the hinge and the leaves are now. The arms are read only for the hands-off check.

    OPEN       the distance between the two foot lines (top view) 244 to 270 mm: the hinge on (or within 1.9
               degrees of) its moulded stop, which gives 264 mm
    LEAN       each leaf's angle from vertical                                       9 to 16 degrees
    PLACED     the nearer foot line 40 to 140 mm from the patch's near edge (along y), and neither foot pad over the
               patch (top view)
    BESIDE     the hinge's middle within the patch's 300 mm length (along x)
    SQUARE     the ridge line (the hinge axis) within 10 degrees of the patch's long edge (along x)
    STANDS     over the last 2 s no tracked point (both foot lines, the hinge) moves more than 3 mm, the spread
               changes under 2 mm and neither leaf turns more than 1 degree
    HANDS OFF  on every step of the last 2 s both grippers' finger zones are 50 mm or more from every part of the sign
    GROUNDED   both foot lines within 10 mm of the table top (the sign is on its feet, not held up or leaning on
               something)
    FALLEN     either leaf more than 45 degrees off vertical now

    0      not finite, FALLEN, not STANDS, not GROUNDED, or the spread under 70 mm (do nothing scores 0)
    1.0    OPEN, LEAN, PLACED, BESIDE, SQUARE, STANDS, HANDS OFF
    0.75   all of those but HANDS OFF (a gripper within 50 mm)
    0.5    STANDS, OPEN, ridge within 30 degrees, no foot pad over the patch, otherwise (a leaf off the lean window,
           the placement outside its band, not beside the patch, the ridge 10 to 30 degrees off)
    0.25   STANDS with the spread at least 70 mm otherwise (not open to the stop, the ridge more than 30 degrees off,
           or a foot pad over the patch)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SIGN = "sign"
LEAF_B = "leaf_b"
_UP = np.array([0.0, 0.0, 1.0])


def _tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def _fold_deg(a: float) -> float:
    """An angle of a line (deg) folded to 0..90: a ridge turned 180 is the same ridge."""
    a = abs(((a + 90.0) % 180.0) - 90.0)
    return a


@dataclass(frozen=True)
class Reading:
    """One step (m, world frame with the table top at z 0)."""
    q: float  # the hinge angle (rad)
    foot_a: np.ndarray  # leaf A's foot line, its middle
    foot_b: np.ndarray
    hinge: np.ndarray  # the hinge's middle, from leaf B's frame
    hinge_a: np.ndarray  # the hinge's middle, from leaf A's frame
    lean_a: float  # deg
    lean_b: float
    yaw: float  # the ridge's angle from x (deg, 0..90)
    low: np.ndarray
    high: np.ndarray
    pads: np.ndarray  # (16, 3) the corners of both foot pads


def read(states) -> Reading | None:
    """A Reading, or None when the sign's state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    s = states.get(SIGN)
    if s is None:
        return None
    try:
        off = np.array([0.0, 0.0, L.TOP])
        o = np.asarray(s.origin, dtype=float).ravel() - off
        ra = np.asarray(s.rotation(), dtype=float)
        rb, hb = s.link_pose(LEAF_B)
        rb = np.asarray(rb, dtype=float)
        hb = np.asarray(hb, dtype=float).ravel() - off
        q = float(s.joints[LEAF_B])
        foot_b = np.asarray(s.link_to_world(LEAF_B, (0.0, 0.0, 0.0)), dtype=float).ravel() - off
        low = np.asarray(s.low, dtype=float).ravel() - off
        high = np.asarray(s.high, dtype=float).ravel() - off
        corners = np.array([(sx * L.PAD_W / 2, sy * L.PAD_D / 2, sz * L.PAD_H / 2) for sx in (-1, 1)
                            for sy in (-1, 1) for sz in (-1, 1)])
        pads_a = o + (corners + np.array([0.0, -L.PAD_D / 2, L.PAD_H / 2])) @ ra.T
        pads_b = np.asarray(s.link_to_world(LEAF_B, corners + np.array([0.0, L.PAD_D / 2, L.PAD_H / 2])),
                            dtype=float) - off
        pads = np.vstack([pads_a, pads_b])
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    parts = (o, ra, rb, hb, np.array([q]), foot_b, low, high, pads)
    if o.size != 3 or foot_b.size != 3 or hb.size != 3 or low.size != 3 or high.size != 3:
        return None
    if not all(np.all(np.isfinite(p)) for p in parts):
        return None
    xa = ra[:, 0]
    yaw = _fold_deg(math.degrees(math.atan2(float(xa[1]), float(xa[0]))))
    return Reading(q=q, foot_a=o, foot_b=foot_b, hinge=hb, hinge_a=o + ra @ np.array([0.0, 0.0, L.H]),
                   lean_a=_tilt_deg(ra), lean_b=_tilt_deg(rb), yaw=yaw, low=low, high=high, pads=pads)


# -- the readings ------------------------------------------------------------------------------------------------------
def spread(r: Reading) -> float:
    return float(np.linalg.norm(r.foot_a[:2] - r.foot_b[:2]))


def near_gap(r: Reading) -> float:
    """From the foot line nearer the patch (the larger y) to the patch's near edge (m; negative when past it)."""
    return float(L.PATCH_Y0 - max(r.foot_a[1], r.foot_b[1]))


def beside(r: Reading) -> bool:
    """The hinge's middle within the patch's length along x: the sign stands at the spill, not off to one end."""
    return bool(abs(float(r.hinge[0]) - L.PATCH_XC) <= L.BESIDE_HALF + L.EPS)


def _box_over_rect(corners: np.ndarray) -> bool:
    """A box (its 8 world corners) overlapping the patch seen from above: separating-axis test of the box's top-view
    outline against the patch rectangle (axes x, y and the normals of the box's edges in the top view)."""
    pts = corners[:, :2]
    rect = np.array([(L.PATCH_X0, L.PATCH_Y0), (L.PATCH_X1, L.PATCH_Y0), (L.PATCH_X1, L.PATCH_Y1),
                     (L.PATCH_X0, L.PATCH_Y1)])
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i, j in ((0, 4), (0, 2), (0, 1)):  # the box's three edge directions (corner order: x, y, z bits)
        e = pts[j] - pts[i]
        if float(np.hypot(*e)) > 1e-9:
            axes.append(np.array([-e[1], e[0]]) / float(np.hypot(*e)))
    for a in axes:
        p, r = pts @ a, rect @ a
        if p.max() <= r.min() or r.max() <= p.min():
            return False
    return True


def over_patch(r: Reading) -> bool:
    """Either foot pad over the patch, seen from above, each pad tested on its own (the grips stand well above the
    floor and may overhang)."""
    return bool(_box_over_rect(r.pads[:8]) or _box_over_rect(r.pads[8:]))


def grounded(r: Reading) -> bool:
    return bool(abs(float(r.foot_a[2])) <= L.FOOT_Z + L.EPS and abs(float(r.foot_b[2])) <= L.FOOT_Z + L.EPS)


def fallen(r: Reading) -> bool:
    return bool(max(r.lean_a, r.lean_b) > L.GATE_DEG)


def lean_ok(r: Reading) -> bool:
    return bool(all(L.LEAN_LO_DEG <= v <= L.LEAN_HI_DEG for v in (r.lean_a, r.lean_b)))


def spread_ok(r: Reading) -> bool:
    return bool(L.SPREAD_LO - L.EPS <= spread(r) <= L.SPREAD_HI + L.EPS)


def placed(r: Reading) -> bool:
    return bool(L.GAP_LO - L.EPS <= near_gap(r) <= L.GAP_HI + L.EPS and not over_patch(r))


def square(r: Reading) -> bool:
    return bool(r.yaw <= L.YAW_OK_DEG + L.EPS)


def tier(settled: bool, r: Reading, released: bool) -> float:
    """The score for a reading, ``settled`` meaning STANDS held over the window."""
    if fallen(r) or not settled or not grounded(r) or spread(r) < L.OPENED_SPREAD:
        return 0.0
    full = spread_ok(r) and lean_ok(r) and placed(r) and beside(r) and square(r)
    if full:
        return 1.0 if released else 0.75
    if spread_ok(r) and r.yaw <= L.YAW_NEAR_DEG + L.EPS and not over_patch(r):
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
    """The last 2 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        return bool(float(np.linalg.norm(r.foot_a - np.array([L.SX, L.SY, 0.0]))) <= L.START_TOL
                    and abs(r.q) <= 0.02)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The sign put back on its starting mark with a jump no trial makes in one step (a reset)."""
        return bool(cls.at_start(now) and (float(np.linalg.norm(now.foot_a - prev.foot_a)) >= L.RESET_JUMP
                                           or abs(now.q - prev.q) >= 0.2))

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0 now, and the 2 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        try:
            dist = states[SIGN].distance
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
        """Over the last 2 s: the largest move (m) of a tracked point, the largest change of the spread (m) and the
        largest turn of a leaf (deg) from where they were at the window's start; None before 2 s of readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = dspread = turn = 0.0
        for x in h[1:]:
            for a, b in ((x.foot_a, first.foot_a), (x.foot_b, first.foot_b), (x.hinge, first.hinge)):
                move = max(move, float(np.linalg.norm(a - b)))
            dspread = max(dspread, abs(spread(x) - spread(first)))
            turn = max(turn, abs(x.lean_a - first.lean_a), abs(x.lean_b - first.lean_b))
        return {"move": move, "spread": dspread, "turn": turn}

    def judge(self, r: Reading, gap) -> Result:
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["spread"] <= L.STILL_SPREAD + L.EPS
                     and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(still, r, released)
        details = {
            "valid_state": True, "score": level,
            "spread_mm": _mm(spread(r)), "q_deg": round(math.degrees(r.q), 3),
            "lean_a_deg": round(r.lean_a, 2), "lean_b_deg": round(r.lean_b, 2),
            "near_gap_mm": _mm(near_gap(r)), "yaw_deg": round(r.yaw, 2),
            "spread_ok": spread_ok(r), "lean_ok": lean_ok(r), "placed": placed(r), "beside": beside(r),
            "square": square(r), "over_patch": over_patch(r), "fallen": fallen(r), "grounded": grounded(r),
            "released_now": bool(self.released[-1]), "released_2s": bool(released), "still_2s": still,
            "hands_gap_mm": _mm(gap),
            "foot_a_mm": _mmv(r.foot_a), "foot_b_mm": _mmv(r.foot_b), "hinge_mm": _mmv(r.hinge),
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "spread_mm": _mm(m["spread"]),
                                              "turn_deg": round(m["turn"], 3)},
        }
        return Result(level, bool(level == 1.0), details)

