"""The scorer, as plain functions of the box's, the supply's, the tool's and the arms' state. NumPy only.

``Tracker.update(states, arms, latched)`` -> Result(score, success, details), one observation per control step. It
reads the box's mechkit AssemblyState (its joint ``cover``, rad, 0 = open on its stop), the supply's (joint
``rocker``, rad, 0 = OFF), the tool's pose and the catch rule's state (catchrule.py).

    OPEN DEG    97 deg minus the cover's joint (0 = shut)
    ON / OFF    the rocker 2 deg past its centre toward ON reads ON, 2 deg before it toward OFF reads OFF (in
                between it keeps its last reading); the trial starts OFF
    SWITCH-ON   a step on which the reading turns from OFF to ON (every one is checked)
    TOOL IN     any part of the screwdriver inside the box's inner volume (to the rim)
    GATE        at SWITCH-ON: TOOL OUT, the cover LATCHED and shut, and no finger zone of either gripper inside the
                box's outline below 14 cm (unreadable arms fail it)
    LATCHED 0   the GATE failed at a SWITCH-ON, or the cover 5 deg or more open on a step that reads ON
    ON TABLE    the tool's centre over the table, no part of it 1 cm below the table top and none of it more than
                6 cm above it (lying on the table or in the tray, not held up)
    RELEASED    every finger zone of both grippers at least 3 cm from the tool (unreadable arms: not released)
    STILL       over the last 1 s the tool moved 2 mm or less
    CLEAR       over the last 1 s every finger zone of both grippers at least 5 cm from the box (cover included)

    1.0   TOOL OUT, ON TABLE, RELEASED, STILL, the cover LATCHED and shut, ON (the gate passed), CLEAR (success)
    0.75  as 1.0 but not CLEAR
    0.5   TOOL OUT, ON TABLE, RELEASED, STILL, the cover LATCHED and shut, OFF
    0.25  TOOL OUT, ON TABLE, RELEASED and STILL
    0     anything else (the tool in the box, off the table, held, moving)
          and for the rest of the trial after a LATCHED 0
Fail closed: a step whose box, supply or tool cannot be read scores 0 and restarts the window; unreadable arms are
never CLEAR and fail a GATE.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BOX, SUPPLY, TOOL, TRAY = "control_box", "test_supply", "screwdriver", "tool_tray"


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    open_deg: float
    rocker_deg: float
    tool_origin: np.ndarray
    tool_rot: np.ndarray
    tool_pts: np.ndarray
    tool_radii: np.ndarray


def read(states) -> Reading | None:
    try:
        b, s, t = states[BOX], states[SUPPLY], states[TOOL]
        cover = float(b.joints["cover"])
        rocker = float(s.joints["rocker"])
        o = np.asarray(t.origin, dtype=float).ravel()
        r = np.asarray(t.rotation(), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    if o.size != 3 or r.shape != (3, 3) or not (math.isfinite(cover) and math.isfinite(rocker)
                                                 and np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
        return None
    pts, radii = L.tool_axis_points(o, r)
    return Reading(L.COVER_OPEN_DEG - math.degrees(cover), math.degrees(rocker), o, r, pts, radii)


def tool_in(r: Reading) -> bool:
    local = r.tool_pts - np.array([0.0, 0.0, L.TOP])
    return L.tool_inside(local, r.tool_radii)


def on_table(r: Reading) -> bool:
    c = r.tool_pts.mean(axis=0)
    low = float(np.min(r.tool_pts[:, 2] - r.tool_radii))
    high = float(np.max(r.tool_pts[:, 2] + r.tool_radii))
    return (L.TABLE_X[0] <= c[0] <= L.TABLE_X[1] and L.TABLE_Y[0] <= c[1] <= L.TABLE_Y[1]
            and low >= L.TOP - L.OFF_TABLE_DROP and high <= L.TOP + L.MAX_REST_HEIGHT)


def tool_released(r: Reading, arms) -> bool:
    """Both grippers have let go of the tool: every finger zone at least RELEASE_M from its surface."""
    pts, rad = r.tool_pts, r.tool_radii

    def distance(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        d = np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=2) - rad[None, :]
        return np.maximum(np.min(d, axis=1), 0.0)

    return rigstate.released(arms, distance, L.RELEASE_M)


def shut(r: Reading) -> bool:
    return r.open_deg <= L.SHUT_DEG + L.EPS


GATE_ZONE = rigstate.boxes_distance([L.gate_box()])


def fingers_in_box(arms) -> bool:
    """A finger zone inside the box's outline below 14 cm (unreadable arms: True)."""
    return rigstate.keep_out_breached(arms, GATE_ZONE, margin=0.0)


def tier(tool_out: bool, table: bool, still: bool, latched_cover: bool, on: bool, clear: bool, zero: str) -> float:
    if zero or not (tool_out and table and still):
        return 0.0
    if not latched_cover:
        return 0.25
    if on:
        return 1.0 if clear else 0.75
    return 0.5


class Tracker:
    def __init__(self) -> None:
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.on = False
        self.switch_ons: list = []
        self.zero = ""
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.prev: Reading | None = None
        self.last = Result()

    @staticmethod
    def at_start(r: Reading) -> bool:
        return (float(np.linalg.norm(r.tool_origin - np.array([L.TOOL_ORIGIN[0], L.TOOL_ORIGIN[1],
                                                                L.TOP + L.TOOL_ORIGIN[2]]))) < 0.005
                and r.open_deg > L.COVER_OPEN_DEG - 1.0 and r.rocker_deg < 1.0)

    @staticmethod
    def teleported(prev: Reading, now: Reading) -> bool:
        """Put back at the start with a jump no trial makes in one step (a reset the hooks did not report)."""
        if not Tracker.at_start(now):
            return False
        return (float(np.linalg.norm(now.tool_origin - prev.tool_origin)) >= L.RESET_JUMP_M
                or prev.open_deg < L.COVER_OPEN_DEG - 20.0 or prev.rocker_deg > L.ROCKER_CENTRE_DEG)

    def _zero(self, why: str) -> None:
        if not self.zero:
            self.zero = why

    def update(self, states, arms, latched_cover: bool) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.win.clear()
            self.prev = None
            self.last = Result(0.0, False, {"unreadable": "box, supply or tool", "zero": self.zero})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.prev = r
        was_on = self.on
        if r.rocker_deg >= L.ROCKER_ON_DEG - L.EPS:
            self.on = True
        elif r.rocker_deg <= L.ROCKER_OFF_DEG + L.EPS:
            self.on = False
        t_in = tool_in(r)
        latched_now = bool(latched_cover) and shut(r)
        gate = None
        if self.on and not was_on:
            hands = fingers_in_box(arms)
            gate = {"step": self.steps, "tool_in": t_in, "latched": latched_now, "fingers_in_box": hands}
            self.switch_ons.append(gate)
            if t_in:
                self._zero("switched on with the tool in the box")
            elif not latched_now:
                self._zero("switched on with the cover not latched")
            elif hands:
                self._zero("switched on with a hand in the box")
        if self.on and r.open_deg >= L.OPEN_LIVE_DEG - L.EPS:
            self._zero("the cover opened with the supply on")
        try:
            dist = states[BOX].distance
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance: not clear
            gap = None
        clear_now = gap is not None and gap >= L.CLEAR_M - 1e-9
        self.win.append((clear_now, r.tool_origin.copy()))
        full = len(self.win) == L.WIN_STEPS
        clear = full and all(w[0] for w in self.win)
        moved = None
        if full:
            moved = max(float(np.linalg.norm(w[1] - self.win[0][1])) for w in self.win)
        still = moved is not None and moved <= L.STILL_M + 1e-9
        table = on_table(r) and tool_released(r, arms)
        score = tier(not t_in, table, still, latched_now, self.on, clear, self.zero)
        details = {"open_deg": round(r.open_deg, 2), "rocker_deg": round(r.rocker_deg, 2), "on": self.on,
                   "tool_in": t_in, "on_table": on_table(r), "released": tool_released(r, arms), "still_1s": still,
                   "moved_mm": None if moved is None else round(moved * 1000, 2), "cover_latched": latched_now,
                   "clear_1s": clear, "gap_mm": None if gap is None else round(gap * 1000, 1), "zero": self.zero,
                   "switch_ons": len(self.switch_ons)}
        if gate is not None:
            details["gate"] = gate
        self.last = Result(score, score >= 1.0, details)
        return self.last
