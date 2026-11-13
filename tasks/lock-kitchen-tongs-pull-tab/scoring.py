"""The scorer, as plain functions of the tongs' state and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the tongs' mechkit AssemblyState (the hinge ``free_arm``: 0 open as drawn, +0.480 rad shut; the slider ``tab``: 0 in,
0.012 m at its stop; the root body's pose; the world box round every shape) and both arms through rigstate.

    TIP GAP     2 x 176 mm x sin(theta / 2), theta = 0.480 - q_hinge     (84 mm open, 9 mm resting on the lock)
    TAB GAP     5 mm + q_tab                                             (5 mm in, 17 mm at the stop)
    SHUT        tip gap <= 14 mm
    SQUEEZED    shut on 20 consecutive steps (1 s) at some point of the trial: latched until the next trial
    ENGAGED     tab gap >= 16 mm (within 1 mm of the stop) and shut, now
    LET GO      on every step of the last 1 s, both grippers' finger zones at least 30 mm from every shape of the
                tongs (rigstate.released with the tongs' own distance; unreadable arms or a distance that cannot be
                computed are NOT let go)
    STILL       over the last 1 s: the root body's path under 2 mm and its summed turn under 2 deg, and the tip gap
                and the tab gap each within 1 mm (max - min)
    ON TOP      the tongs' lowest point within 1 mm of the worktop
    NOT FALLEN  the lowest point no more than 10 mm below the worktop

    1.0   engaged, let go, still and on top                         (the trial ends once this has held 1 s)
    0.5   engaged and not fallen, but not all of let go, still and on top
    0.25  not engaged, not fallen, and squeezed at some point
    0     anything else (and any step whose tongs state is missing or not finite)

No zero condition and no whole-trial latch other than the squeeze milestone: nothing here can hurt a person or the
tool. The squeeze flag is the only state kept across steps besides the 1 s windows.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

TONGS = "tongs"
HINGE = "free_arm"
TAB = "tab"


# -- geometry ------------------------------------------------------------------------------------------------------


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrap_deg(a: float) -> float:
    a = math.fmod(float(a) + 180.0, 360.0)
    if a <= 0.0:
        a += 360.0
    return a - 180.0


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    """The angle (deg) of the rotation from orientation r0 to r1."""
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def heading_deg(r: np.ndarray) -> float:
    return wrap_deg(math.degrees(math.atan2(r[1, 0], r[0, 0])))


@dataclass(frozen=True)
class Reading:
    """What the scorer reads from one tongs state (m and deg)."""
    origin: np.ndarray
    rot: np.ndarray
    q_hinge: float
    q_tab: float
    tip_gap: float
    tab_gap: float
    low_z: float  # the lowest point of any shape, above the worktop (m; negative below it)


def read(s) -> Reading | None:
    """A Reading, or None when the state is missing, incomplete or not finite."""
    if s is None:
        return None
    try:
        o = np.asarray(getattr(s, "origin", None) if getattr(s, "origin", None) is not None else s.pos,
                       dtype=float).ravel()
        q = np.asarray(s.quat_xyzw, dtype=float).ravel()
        joints = getattr(s, "joints", None) or {}
        qh, qt = float(joints[HINGE]), float(joints[TAB])
        low = np.asarray(s.low, dtype=float).ravel()
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if o.size != 3 or q.size != 4 or low.size != 3:
        return None
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(q)) and np.all(np.isfinite(low))
            and math.isfinite(qh) and math.isfinite(qt)) or np.linalg.norm(q) < 0.5:
        return None
    return Reading(o, rotation(q), qh, qt, L.tip_gap(L.theta_of(qh)), L.tab_gap(qt), float(low[2] - L.TOP))


def tongs_distance(state):
    """The tongs' exact distance function (every box of the three bodies as they are now) for rigstate.released. A
    state without one raises inside rigstate, which counts as NOT let go."""
    return state.distance


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def tier(engaged: bool, let_go: bool, still: bool, on_top: bool, not_fallen: bool, squeezed: bool) -> float:
    if engaged and let_go and still and on_top:
        return 1.0
    if engaged and not_fallen:
        return 0.5
    if not engaged and not_fallen and squeezed:
        return 0.25
    return 0.0


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000.0, 2)


class Tracker:
    """The squeeze milestone, the last 1 s of readings (stillness) and of the let-go check."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.squeeze_need = max(1, int(round(L.SQUEEZE_S * control_hz)))  # 20 steps
        self.resets = 0  # fallback resets seen (teleports back to the start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.shut_run = 0
        self.squeezed = False
        self.squeezed_at: int | None = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    # -- reset fallback ----------------------------------------------------------------------------------------
    @staticmethod
    def at_start(r: Reading) -> bool:
        start = np.array(L.ORIGIN, dtype=float)
        if float(np.linalg.norm(r.origin - start)) > L.RESET_START_TOL:
            return False
        if abs(wrap_deg(heading_deg(r.rot) - math.degrees(L.YAW))) > L.RESET_START_TURN_DEG:
            return False
        return abs(r.q_hinge) <= 0.02 and abs(r.q_tab) <= 0.0005

    @classmethod
    def teleported(cls, prev: Reading, now: Reading) -> bool:
        """The simulator put the tongs back on their start pose, open, tab in, with a jump no trial makes in one
        control step: a new episode."""
        if not cls.at_start(now):
            return False
        return (abs(now.q_tab - prev.q_tab) >= L.RESET_TAB_JUMP
                or float(np.linalg.norm(now.origin - prev.origin)) >= L.RESET_JUMP
                or turn_deg(prev.rot, now.rot) >= L.RESET_TURN_DEG)

    # -- per step ------------------------------------------------------------------------------------------------
    def update(self, states: dict, arms) -> Result:
        self.steps += 1
        s = states.get(TONGS) if isinstance(states, dict) else None
        r = read(s)
        if r is None:
            self.bad_steps += 1  # a blow-up or a missing state: 0, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.shut_run = 0
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "squeezed": self.squeezed})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        shut = r.tip_gap <= L.SHUT_TIP_GAP + L.EPS
        self.shut_run = self.shut_run + 1 if shut else 0
        if not self.squeezed and self.shut_run >= self.squeeze_need:
            self.squeezed = True
            self.squeezed_at = self.steps
        self.hist.append(r)
        self.prev = r
        try:
            dist = tongs_distance(s)
        except Exception:  # noqa: BLE001 - no distance: not let go
            dist = None
        self.released.append(dist is not None and rigstate.released(arms, dist, L.LET_GO))
        self.last = self.judge(r, arms, dist, shut)
        return self.last

    def motion(self) -> dict | None:
        """Over the last 1 s: root path (m), summed turn (deg), spread of the tip gap and of the tab gap (m); None
        before 1 s of valid readings."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        path = sum(float(np.linalg.norm(b.origin - a.origin)) for a, b in zip(h, h[1:]))
        turn = sum(turn_deg(a.rot, b.rot) for a, b in zip(h, h[1:]))
        tips = [x.tip_gap for x in h]
        tabs = [x.tab_gap for x in h]
        return {"path": path, "turn": turn, "tip_spread": max(tips) - min(tips), "tab_spread": max(tabs) - min(tabs)}

    def judge(self, r: Reading, arms, dist, shut: bool) -> Result:
        engaged = bool(r.tab_gap >= L.ENGAGED_TAB_GAP - L.EPS and shut)
        let_go = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["path"] < L.STILL_PATH and m["turn"] < L.STILL_TURN_DEG
                     and m["tip_spread"] <= L.STILL_GAP + L.EPS and m["tab_spread"] <= L.STILL_GAP + L.EPS)
        on_top = abs(r.low_z) <= L.ON_TOP_TOL + L.EPS
        not_fallen = r.low_z > -L.FALLEN_DEPTH
        level = tier(engaged, let_go, still, on_top, not_fallen, self.squeezed)
        gap = None
        if dist is not None:
            try:
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001 - trace only
                gap = None
        details = {
            "valid_state": True, "score": level, "hinge_from_shut_deg": round(math.degrees(L.theta_of(r.q_hinge)), 3),
            "tab_mm": _mm(r.q_tab), "tip_gap_mm": _mm(r.tip_gap), "tab_gap_mm": _mm(r.tab_gap), "shut": bool(shut),
            "squeezed": self.squeezed, "squeezed_at": self.squeezed_at, "engaged": engaged,
            "released": bool(self.released and self.released[-1]), "let_go_1s": bool(let_go), "still_1s": still,
            "on_top": bool(on_top), "not_fallen": bool(not_fallen), "low_mm": _mm(r.low_z),
            "origin": [round(float(v), 4) for v in r.origin], "heading_deg": round(heading_deg(r.rot), 2),
            "motion": None if m is None else {"path_mm": _mm(m["path"]), "turn_deg": round(m["turn"], 3),
                                               "tip_spread_mm": _mm(m["tip_spread"]),
                                               "tab_spread_mm": _mm(m["tab_spread"])},
            "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms),
        }
        return Result(level, bool(level == 1.0), details)
