"""The scorer, as plain functions of the chest's, the screwdriver's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads the
mechkit AssemblyState of the chest (base frame on the table at the middle of its front face, x back into the chest;
links "left" and "right", the two drawers) and of the screwdriver (origin on the handle's axis, local y along the tool).

    OPENING     how far a drawer is out of the chest (m): the left joint + 0.190 (it is drawn open), the right joint
    TIP         any foot's bottom 10 mm or more above the table (a 10 mm block slides under it). Latched for the trial
    DRAGGED     the chest's origin more than 30 mm from where it started, seen from above. Latched
    OFF TABLE   the chest or the screwdriver outside the table's outline or more than 10 mm below its top. Latched
    IN DRAWER   all 8 corners of the box round the screwdriver inside the right drawer's box, taken in the drawer's own
                frame: between its front panel and back wall, between its side walls, above its floor and below the
                front panel's top (91 mm over the table with the chest level)
    CLEAR       on every step of the last 1 s both grippers' finger zones 5 cm or more from the chest and the
                screwdriver (rigstate; unreadable arms: not clear)
    STILL       over the last 1 s the screwdriver moved under 2 mm and turned under 2 degrees

    0      TIP, DRAGGED or OFF TABLE at any time; or not CLEAR at the end; or nothing below
    0.25   the left drawer 95 mm or less open
    0.5    the left drawer 10 mm or less open (shut)
    0.75   the left drawer shut and the right drawer 150 mm or more open, or the screwdriver in it but not still
    1.0    the screwdriver IN DRAWER (right) and STILL                                  (held for 1 s ends the trial)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

CHEST = "tool_chest"
TOOL = "screwdriver"
PARTS = (CHEST, TOOL)


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


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one step. World frame, table top at z 0."""
    chest_o: np.ndarray
    chest_r: np.ndarray
    opening: dict  # drawer -> m out of the chest
    right_pose: tuple  # (rotation, origin) of the right drawer's frame (world, table top at 0)
    right_off: np.ndarray  # that frame's origin in the chest frame at rest
    tool_o: np.ndarray
    tool_r: np.ndarray
    chest_low: float
    chest_xy: np.ndarray
    tool_low: float
    tool_xy: np.ndarray


def _vec(v, n=3) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    if a.size != n or not np.all(np.isfinite(a)):
        raise ValueError("bad vector")
    return a


def read(states) -> Reading | None:
    """A Reading, or None when a state is missing, malformed or not finite."""
    if not isinstance(states, dict):
        return None
    try:
        c, t = states.get(CHEST), states.get(TOOL)
        if c is None or t is None:
            return None
        lift = np.array([0.0, 0.0, L.TOP])
        co = _vec(c.origin) - lift
        cr = rotation(_vec(c.quat_xyzw, 4))
        q = {n: float(c.joints[n]) for n in L.DRAWERS}
        if not all(math.isfinite(v) for v in q.values()):
            return None
        opening = {"left": q["left"] + L.TRAVEL, "right": q["right"]}
        rp = _vec(c.links["right"], 7)
        right_pose = (rotation(rp[3:7]), rp[:3] - lift)
        spec = getattr(c, "spec", None)
        off = np.asarray(spec.body_offset("right"), dtype=float) if spec is not None else np.zeros(3)
        to = _vec(t.origin) - lift
        tr = rotation(_vec(t.quat_xyzw, 4))
        if not (np.all(np.isfinite(cr)) and np.all(np.isfinite(tr))):
            return None
        return Reading(co, cr, opening, right_pose, off, to, tr, float(_vec(c.low)[2] - L.TOP), _vec(c.pos)[:2],
                       float(_vec(t.low)[2] - L.TOP), _vec(t.pos)[:2])
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None


def foot_lift(r: Reading) -> float:
    """The highest of the four feet's bottom corners above the table (m)."""
    pts = L.foot_corners() @ r.chest_r.T + r.chest_o
    return float(pts[:, 2].max())


def tool_in_right(r: Reading) -> bool:
    """All 8 corners of the screwdriver's box inside the right drawer's box (its own frame, at rest = shut)."""
    rot, org = r.right_pose
    world = L.tool_corners() @ r.tool_r.T + r.tool_o
    local = (world - org) @ rot + r.right_off  # chest frame at rest, the drawer shut
    m = L.IN_MARGIN
    cy = L.DRAWER_Y["right"]
    return bool(np.all(local[:, 0] >= L.INNER_X[0] - m) and np.all(local[:, 0] <= L.INNER_X[1] + m)
                and np.all(np.abs(local[:, 1] - cy) <= L.INNER_HALF_Y + m)
                and np.all(local[:, 2] >= L.FLOOR_Z[1] - m) and np.all(local[:, 2] <= L.FRONT_Z[1] + m))


def off_table(r: Reading) -> bool:
    for xy, low in ((r.chest_xy, r.chest_low), (r.tool_xy, r.tool_low)):
        if not (L.TABLE_X0 <= xy[0] <= L.TABLE_X1 and L.TABLE_Y0 <= xy[1] <= L.TABLE_Y1):
            return True
        if low < -L.OFF_TABLE_DROP:
            return True
    return False


def tier(dead: bool, clear: bool, left_open: float, right_open: float, inside: bool, still: bool) -> float:
    if dead or not clear:
        return 0.0
    if inside and still:
        return 1.0
    if left_open <= L.SHUT + L.EPS and (right_open >= L.RIGHT_OPEN - L.EPS or inside):
        return 0.75
    if left_open <= L.SHUT + L.EPS:
        return 0.5
    if left_open <= L.HALF + L.EPS:
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
    """The latches (TIP, dragged, off the table), the last 1 s of readings and of the clear check, and the reset
    fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)
        self.clear: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.start_xy: np.ndarray | None = None
        self.latched: dict = {"tip": None, "dragged": None, "off_table": None}
        self.max_lift = 0.0
        self.max_open = {n: 0.0 for n in L.DRAWERS}
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (abs(r.opening["left"] - L.START_OPEN["left"]) < 0.005 and r.opening["right"] < 0.005
                and float(np.linalg.norm(r.tool_o[:2] - np.array(L.TOOL_START))) < 0.03)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """Everything put back on its start with a jump no trial makes in one step."""
        if not cls.at_start(now):
            return False
        return (float(np.linalg.norm(now.tool_o - prev.tool_o)) >= L.RESET_JUMP
                or abs(now.opening["left"] - prev.opening["left"]) >= L.RESET_JUMP
                or abs(now.opening["right"] - prev.opening["right"]) >= L.RESET_JUMP)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0, and the 1 s windows start again
            self.hist.clear()
            self.clear.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        if self.start_xy is None:
            self.start_xy = r.chest_o[:2].copy()
        self.hist.append(r)
        self.prev = r
        lift = foot_lift(r)
        self.max_lift = max(self.max_lift, lift)
        for n in L.DRAWERS:
            self.max_open[n] = max(self.max_open[n], r.opening[n])
        if self.latched["tip"] is None and lift >= L.TIP_LIFT - L.EPS:
            self.latched["tip"] = self.steps
        if self.latched["dragged"] is None and float(np.linalg.norm(r.chest_o[:2] - self.start_xy)) > L.DRAG + L.EPS:
            self.latched["dragged"] = self.steps
        if self.latched["off_table"] is None and off_table(r):
            self.latched["off_table"] = self.steps
        try:
            dc, dt = states[CHEST].distance, states[TOOL].distance

            def dist(points):
                return np.minimum(dc(points), dt(points))
        except Exception:  # noqa: BLE001 - no distance: not clear
            dist = None
        ok = False
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.CLEAR))
            except Exception:  # noqa: BLE001
                ok = False
        self.clear.append(ok)
        self.last = self.judge(r, arms, dist, lift)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: the screwdriver's largest move (m) and turn (deg) from where it was at the window's
        start; None before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        corners = L.tool_corners()

        def pts(x):
            return corners @ x.tool_r.T + x.tool_o

        p0 = pts(first)
        move = max(float(np.max(np.linalg.norm(pts(x) - p0, axis=1))) for x in h[1:])
        turn = max(turn_deg(first.tool_r, x.tool_r) for x in h[1:])
        return {"move": move, "turn": turn}

    def judge(self, r: Reading, arms, dist, lift: float) -> Result:
        dead = any(v is not None for v in self.latched.values())
        clear = len(self.clear) == self.window and all(self.clear)
        inside = tool_in_right(r)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS)
        level = tier(dead, clear, r.opening["left"], r.opening["right"], inside, still)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(r.chest_r[2, 2])))))
        details = {
            "valid_state": True, "score": level, "latched": dict(self.latched),
            "left_open_mm": _mm(r.opening["left"]), "right_open_mm": _mm(r.opening["right"]),
            "max_open_mm": {n: _mm(v) for n, v in self.max_open.items()},
            "foot_lift_mm": _mm(lift), "max_foot_lift_mm": _mm(self.max_lift), "chest_tilt_deg": round(tilt, 3),
            "chest_xy_mm": [_mm(r.chest_o[0] - L.CHEST_X), _mm(r.chest_o[1])],
            "tool_mm": [_mm(v) for v in r.tool_o], "tool_in_right": inside, "still_1s": still,
            "motion": None if m is None else {"move_mm": _mm(m["move"]), "turn_deg": round(m["turn"], 3)},
            "clear_now": bool(self.clear[-1]), "clear_1s": bool(clear), "zone_mm": _mm(gap),
            "arms_readable": rigstate.readable(arms),
        }
        return Result(level, bool(level == 1.0), details)
