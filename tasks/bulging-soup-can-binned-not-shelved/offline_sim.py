"""Kinematics of the trash bin's lid for use ONLY where there is no simulator: tools/offline/check_task.py and the
unit tests. The simulator never reads from here: in Isaac Sim the lid is a PhysX hinge moved by the fingers'
contacts and by gravity. The cans are carried and dropped by the checker's own kinematic proxy.

Offline model: a CLOSED gripper (pads under 6 mm apart) is a pad of half thickness 8 mm, 27 mm wide across and 38 mm
tall from its tips. A flap is pushed shut by it when the pad, having been outside the flap (further from flat), comes
to lie where the flap is: the flap turns to the pad's leading face. Past upright (90 degrees) a flap left alone falls
shut under its own weight (6 degrees per control step); before upright it stays (it leans on its stop).
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

CLOSED_GAP = 0.006
PAD_HALF_T = 0.008
PAD_HALF_W = 0.0135
PAD_HEIGHTS = (0.004, 0.012, 0.020, 0.028, 0.036)  # sample points up the pad from its tips
FALL_DEG = 6.0


class OfflineFlaps:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.theta = {n: L.PANELS[n][5] for n in L.PANELS}
        self.outside: dict = {}
        self.events: list[str] = []

    def joints(self) -> dict[str, float]:
        return {n: math.radians(L.PANELS[n][5] - self.theta[n]) for n in L.PANELS}

    @staticmethod
    def _polar(name: str, point) -> tuple[float, float]:
        """A world point's radius and angle (degrees from flat) about a flap's hinge, the pad's leading face taken."""
        hinge_x, inward, hinge_z = L.PANELS[name][:3]
        u = inward * (float(point[0]) - hinge_x) + PAD_HALF_T
        v = float(point[2]) - L.TOP - hinge_z
        return math.hypot(u, v), math.degrees(math.atan2(v, u))

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        pushed = set()
        if read is not None:
            for side, (_tcp, tips, opening) in zip(rigstate.ARMS, read):
                if float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP > CLOSED_GAP:
                    for k in [k for k in self.outside if k[0] == side]:
                        del self.outside[k]
                    continue
                for name, (_hx, _inw, _hz, length, (y0, y1), _o) in L.PANELS.items():
                    if not (y0 - PAD_HALF_W <= float(tips[1]) <= y1 + PAD_HALF_W):
                        continue
                    pts = [np.array([tips[0], tips[1], tips[2] + h]) for h in PAD_HEIGHTS]
                    polar = [self._polar(name, p) for p in pts]
                    on = [(r, a) for r, a in polar if 0.005 <= r <= length + PAD_HALF_T]
                    key = (side, name)
                    if not on:
                        self.outside[key] = False
                        continue
                    was_out = self.outside.get(key, False)
                    a = min(a for _, a in on)
                    if was_out and a < self.theta[name]:
                        self.theta[name] = max(0.0, a)
                        pushed.add(name)
                    self.outside[key] = was_out or a >= self.theta[name] - 1e-6
        for name in L.PANELS:
            if name not in pushed and self.theta[name] < L.BALANCE_DEG:
                self.theta[name] = max(0.0, self.theta[name] - FALL_DEG)
