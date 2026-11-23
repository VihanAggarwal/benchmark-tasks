"""Offline checker only: where the grippers would put the drip tray and the dial. The simulator never uses this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics and no joints: every mechkit joint stays
at 0. So when task.py's hooks see states whose source is "offline", the scorer is fed the tray's travel and the
dial's angle from this small kinematic model instead, using the pad-tip points and openings the checker feeds to
rigstate and each hand's commanded turn about the vertical, read from the scripted plan at the current step:

- a gripper closed (opening under 0.25) with its pad-tip point inside the tray's pull tab (4 mm of slack round it)
  holds the tray: while held, the tray's travel follows the hand along y, between its stops (0 and 0.21 m);
- a gripper closed with its pad-tip point within 8 mm of the dial's axis at the grip bar's height holds the dial: it
  turns with the hand (clockwise seen from above positive), between its stops (0 and 120 degrees);
- let go, both stay where they are. Nothing slips or binds.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

ARMS = ("left", "right")
CLOSED = 0.25


def smoothstep(s: float) -> float:
    s = min(max(float(s), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def commanded_yaw(waypoints, t: float) -> float | None:
    """The hand's commanded turn about the vertical at ``t`` s into its plan (as yam_scripted's ArmPlan interpolates:
    smoothstep in time, the shorter way round). None before the plan's first waypoint is reached."""
    if not waypoints:
        return None
    prev = None
    for w in waypoints:
        yaw = float(getattr(w, "yaw", 0.0))
        d = float(w.duration)
        if t < d:
            if prev is None:
                return None
            return prev + smoothstep(t / d) * math.remainder(yaw - prev, 2 * math.pi)
        t -= d
        prev = yaw
    return prev


def _arm(arms, side):
    try:
        a = arms[side]
        tips = np.asarray(a.tips, dtype=float).ravel()
        opening = float(a.opening)
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    if tips.size != 3 or not np.all(np.isfinite(tips)) or not math.isfinite(opening):
        return None
    return tips, opening


class GriddleModel:
    """The tray's travel (m) and the dial's angle (deg)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.travel = 0.0
        self.dial = 0.0
        self.tray_hold: dict = {}  # arm -> (hand y then, travel then)
        self.dial_hold: dict = {}  # arm -> (hand yaw then, dial then)
        self.events: list = []

    def _on_tab(self, tips) -> bool:
        ty = L.tab_centre_y(self.travel)
        return (abs(tips[0] - L.TRAY_X) <= L.TAB[0] / 2 + 0.004 and abs(tips[1] - ty) <= L.TAB[1] / 2 + 0.004
                and L.TRAY_Z0 - 0.004 <= tips[2] <= L.TRAY_Z0 + L.TAB[2])

    @staticmethod
    def _on_dial(tips) -> bool:
        return (math.hypot(tips[0] - L.DIAL[0], tips[1] - L.DIAL[1]) <= 0.008
                and L.PROBE_Z1 + L.DIAL_SKIRT_H - 0.002 <= tips[2] <= L.DIAL_TOP)

    def update(self, arms, yaws: dict, step: int) -> None:
        seen = {}
        if isinstance(arms, dict):
            for side in ARMS:
                a = _arm(arms, side)
                if a is not None:
                    tips, opening = a
                    seen[side] = (tips - np.array([0.0, 0.0, L.TOP]), opening)
        for side in ARMS:
            if side not in seen or seen[side][1] >= CLOSED or yaws.get(side) is None:
                if side in self.tray_hold or side in self.dial_hold:
                    self.events.append((step, side, "let go"))
                self.tray_hold.pop(side, None)
                self.dial_hold.pop(side, None)
                continue
            tips, _ = seen[side]
            if side in self.tray_hold or side in self.dial_hold:
                continue
            if self._on_dial(tips):
                self.dial_hold[side] = (yaws[side], self.dial)
                self.events.append((step, side, "holds the dial"))
            elif self._on_tab(tips):
                self.tray_hold[side] = (float(tips[1]), self.travel)
                self.events.append((step, side, "holds the tray"))
        for side, (y0, a0) in self.dial_hold.items():
            a = a0 - math.degrees(math.remainder(yaws[side] - y0, 2 * math.pi))
            self.dial = min(max(a, 0.0), math.degrees(L.DIAL_UPPER))
        for side, (hy0, t0) in self.tray_hold.items():
            self.travel = min(max(t0 + float(seen[side][0][1]) - hy0, 0.0), L.TRAVEL)
