"""The scorer, as plain functions of the window's state and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the window's mechkit AssemblyState (``window``: the frame as a fixed base, the sash on its hinge, the stay's swivel
on the sash and the stay's bar on its lift pin) and both arms through rigstate.

    ON_PIN(k)   the stay's hole k is on pin A: the hole's centre within 2.0 mm of the pin's axis in plan, and the
                pin's tip reaches the bar's top face there or stands above it (4.25 mm of pin in the hole). Holes
                are numbered from the swivel: 1 (window shut) to 4 (the last hole: widest open).
    RELEASED    rigstate.released with the stay's own distance (bar and swivel): each gripper's finger zone at least
                30 mm away; unreadable arms, or a distance that cannot be computed, are NOT released
    STILL       over the last 20 steps the ball and the sash's free bottom corner moved at most 1.0 mm
    GLASS       a finger zone enters the pane's space (inside the sash's stiles and rails, through its thickness):
                in a real window the hand strikes the glass. Latched for the trial (readable arms only).

    1.0    ON_PIN(4) and RELEASED on each of the last 20 steps, and STILL                       (success)
    0.5    ON_PIN(3) and RELEASED                       (one setting wider than the start, let go)
    0.25   ON_PIN(4) or ON_PIN(3), not released or not yet still for 20 steps (the stay still held)
    0      anything else: still on hole 2 (the start), shut, the stay off the pin (the sash free to swing), any
           step whose window state is missing or not finite (the windows start again), or GLASS (latched)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

WINDOW = "window"
SASH, SWIVEL, STAY = "sash", "swivel", "stay"
BOARD = "board"

IN_HOLE_XY = 0.0020  # m: hole centre to the pin's axis in plan
PROUD = 0.0  # m: the pin's tip at or above the bar's top face
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.0010
RESET_START_DEG = 1.0
RESET_TURN_DEG = 4.0
EPS = 1e-9

HOLE_PTS = np.array([L.bar_point(s, L.WEB_BOT_H) for s in L.HOLE_S])  # base frame at rest
HOLE_TOP_PTS = np.array([L.bar_point(s, L.WEB_TOP_H) for s in L.HOLE_S])
BALL_PT = L.bar_point(L.TAIL_S, L.BALL_H)
PIN_TIP_Z = L.Z0 + L.PIN_TOP_H


def _sash_corner_rest() -> np.ndarray:
    """The sash's free bottom corner (outside face, far from the hinge), base frame at rest."""
    rot = L.rz(L.THETA_START)
    p = L.HINGE_XY + rot @ (np.array([L.XS + L.SASH_T, -L.SASH_W / 2]) - L.HINGE_XY)
    b = L.to_base_xy(p)
    return np.array([b[0], b[1], L.SASH_Z0])


SASH_CORNER = _sash_corner_rest()


def _pane_rest():
    """(centre, size, yaw) of the pane's space in the base frame at rest."""
    c, size = L.PANE_BOX_SHUT
    p = L.HINGE_XY + L.rz(L.THETA_START) @ (c[:2] - L.HINGE_XY)
    b = L.to_base_xy(p)
    return np.array([b[0], b[1], c[2] - L.TOP]), size, L.THETA_START - L.BASE_YAW


PANE_C, PANE_SIZE, PANE_YAW = _pane_rest()


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one window state (world, m)."""
    joints: tuple  # (sash, swivel, lift) rad
    holes: np.ndarray  # (4, 3) web underside at each hole's centre
    hole_tops: np.ndarray  # (4, 3) bar top face there
    ball: np.ndarray
    corner: np.ndarray  # the sash's free bottom corner
    pane: tuple  # (centre, size, rotation) of the pane's space, world

    @property
    def track(self) -> np.ndarray:
        return np.vstack([self.ball, self.corner])


def read(sw) -> Reading | None:
    """A Reading, or None when the state is missing, incomplete or not finite."""
    if sw is None:
        return None
    try:
        q = (float(sw.joints[SASH]), float(sw.joints[SWIVEL]), float(sw.joints[STAY]))
        holes = np.asarray(sw.link_to_world(STAY, HOLE_PTS), dtype=float)
        tops = np.asarray(sw.link_to_world(STAY, HOLE_TOP_PTS), dtype=float)
        ball = np.asarray(sw.link_to_world(STAY, BALL_PT), dtype=float).ravel()
        corner = np.asarray(sw.link_to_world(SASH, SASH_CORNER), dtype=float).ravel()
        r_s, _ = sw.link_pose(SASH)
        pane_c = np.asarray(sw.link_to_world(SASH, PANE_C), dtype=float).ravel()
        r_s = np.asarray(r_s, dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    arrays = (holes, tops, ball, corner, pane_c, r_s)
    if holes.shape != (4, 3) or tops.shape != (4, 3) or ball.size != 3 or r_s.shape != (3, 3):
        return None
    if not all(np.all(np.isfinite(a)) for a in arrays) or not all(math.isfinite(v) for v in q):
        return None
    pane = (pane_c, PANE_SIZE, r_s @ _rot_z(PANE_YAW))
    return Reading(q, holes, tops, ball, corner, pane)


def hole_on_pin(r: Reading, pin: str = "A") -> int | None:
    """The hole (1..4) that is on pin ``pin``, or None."""
    p = L.PINS[pin]
    for k in range(4):
        d = float(np.hypot(r.holes[k, 0] - p[0], r.holes[k, 1] - p[1]))
        if d <= IN_HOLE_XY + EPS and r.hole_tops[k, 2] <= PIN_TIP_Z - PROUD + EPS:
            return k + 1
    return None


def stay_distance(sw):
    def distance(points):
        return sw.distance(points, bodies=(STAY, SWIVEL))

    return distance


def pane_distance(r: Reading):
    c, size, rot = r.pane

    def distance(points):
        return rigstate.box_distances(points, c, size, rot)

    return distance


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def tier(hole: int | None, released: bool, full: bool, still: bool, glass: bool) -> float:
    if glass:
        return 0.0
    if hole == 4 and full and still:
        return 1.0
    if hole == 3 and released:
        return 0.5
    if hole in (3, 4):
        return 0.25
    return 0.0


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The last 20 steps' readings (stillness) and verdicts (the 1.0 conditions); the glass latch."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.glass = False
        self.glass_step = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.ok: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        lim = math.radians(RESET_START_DEG)
        return all(abs(v) <= lim for v in r.joints)

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The simulator put the window back on its start pose with a jump no trial makes in one control step."""
        if not cls.at_start(now):
            return False
        return any(math.degrees(abs(a - b)) >= RESET_TURN_DEG for a, b in zip(prev.joints, now.joints))

    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        sw = states.get(WINDOW) if isinstance(states, dict) else None
        r = read(sw)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.ok.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps, "glass": self.glass})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            resets = self.resets + 1
            self.reset()
            self.resets = resets
            self.steps = 1
        self.hist.append(r)
        self.prev = r
        try:
            dist = stay_distance(sw)
            released = bool(rigstate.released(arms, dist, LET_GO))
        except Exception:  # noqa: BLE001 - no distance: not released
            dist, released = None, False
        pane_gap = None
        if rigstate.readable(arms):
            try:
                pane_gap = rigstate.hands_gap(arms, pane_distance(r))
            except Exception:  # noqa: BLE001 - trace only; no latch on an unreadable distance
                pane_gap = None
            if pane_gap is not None and pane_gap < 0.0 and not self.glass:
                self.glass, self.glass_step = True, self.steps
        self.last = self.judge(r, arms, dist, released, pane_gap)
        return self.last

    def motion(self) -> float | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        return max(float(np.linalg.norm(x.track - h[0].track, axis=1).max()) for x in h[1:])

    def judge(self, r: Reading, arms, dist, released: bool, pane_gap) -> Result:
        hole = hole_on_pin(r, "A")
        shut = hole_on_pin(r, "B") is not None
        if shut:
            hole = None
        self.ok.append(bool(hole == 4 and released))
        full = len(self.ok) == self.window and all(self.ok)
        m = self.motion()
        still = bool(m is not None and m <= STILL_MOVE + EPS)
        level = tier(hole, released, full, still, self.glass)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        p = L.PINS["A"]
        near = [_mm(float(np.hypot(h[0] - p[0], h[1] - p[1]))) for h in r.holes]
        details = {
            "valid_state": True, "score": level, "hole": hole, "released": released, "ok_20": bool(full),
            "still_20": still, "glass": self.glass, "glass_step": self.glass_step,
            "sash_deg": round(math.degrees(r.joints[0] + L.THETA_START), 2),
            "swivel_deg": round(math.degrees(r.joints[1]), 2), "lift_deg": round(math.degrees(r.joints[2]), 2),
            "hole_to_pin_mm": near, "hole_top_mm": [_mm(t[2] - L.Z0) for t in r.hole_tops],
            "motion_mm": None if m is None else _mm(m), "zone_mm": _mm(gap), "pane_mm": _mm(pane_gap),
            "arms_readable": rigstate.readable(arms),
        }
        return Result(level, bool(level == 1.0), details)
