"""Offline checker only: how far the grippers push the spray arm round. The simulator never calls anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics: it sees the spray assembly as mechkit's
assembly at rest, its joint at 0, whatever the grippers do (``Assembly.offline_state``). So when task.py's hooks get a
spray state whose ``source`` is "offline", the scorer is fed the arm's angle from this small kinematic model instead,
using the finger-tip points the checker feeds to rigstate:

- a gripper pushes the arm while its finger-tip point is 45 to 105 mm above the table (its fingers then cover the
  arm's 85 to 105 mm band, or meet it from below or above) and 20 to 110 mm from the hub's axis (inside the arm's
  length, outside the hub);
- the arm is a line through the axis (both ends, so angles are taken modulo half a turn); a pushing gripper keeps
  that line at least atan(24 mm / r) from its finger-tip point's angle about the axis (the bar's 10 mm half-width
  plus a closed finger pair's 14 mm half-width, at the tips' distance r), on the side the gripper was on when it
  last stood clear of the line, so the arm is pushed ahead of the fingers either way;
- the angle stops at the axle's stops (+-6.2 rad). Nothing else turns it: no coasting, no drag, no slip, and an arm's
  links or a carried item do not push it (the demo never does that, and it must not count offline).

The real contact (the fingers' 27 to 40 mm width, the soft wrist, the arm coasting a few degrees after each push)
decides the real angle on the GPU: README.md says what to read in the trace. The model resets with rigstate's reset
hook and at every new offline run.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

ARMS = ("left", "right")
HUB = np.array(L.HUB, dtype=float)


def _tips(arms, side):
    try:
        t = np.asarray(arms[side].tips, dtype=float).ravel()
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    return t if t.size == 3 and np.all(np.isfinite(t)) else None


class ArmModel:
    """The spray arm's angle (rad, anticlockwise seen from above; the red end's direction) as the grippers would push
    it."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.theta = 0.0
        self.side = {s: None for s in ARMS}  # which side of the arm's line each gripper is on (-1 behind, +1 ahead)
        self.pushes = 0  # control steps on which some gripper pushed the arm (for the trace and the tests)

    def in_band(self, tips) -> bool:
        r = float(np.hypot(*(tips[:2] - HUB)))
        return bool(L.TOP + L.PUSH_Z[0] <= float(tips[2]) <= L.TOP + L.PUSH_Z[1]
                    and L.PUSH_RADII[0] <= r <= L.PUSH_RADII[1])

    def update(self, arms) -> float:
        """Advance one control step with the grippers as rigstate reports them; returns the angle."""
        if not isinstance(arms, dict):
            self.side = {s: None for s in ARMS}
            return self.theta
        for s in ARMS:
            tips = _tips(arms, s)
            if tips is None or not self.in_band(tips):
                self.side[s] = None
                continue
            rel = tips[:2] - HUB
            r = float(np.hypot(*rel))
            phi = math.atan2(rel[1], rel[0])
            half = math.atan(L.PUSH_HALF / r)
            d = math.remainder(phi - self.theta, math.pi)  # the tips' angle from the nearer end of the line
            if self.side[s] is None or abs(d) >= half:
                self.side[s] = -1 if d < 0 else 1  # clear of the line (or just arrived): which side it is on
            if abs(d) < half:  # pushing: the line goes ahead of the fingers, ``half`` from the tips
                theta = self.theta + d - self.side[s] * half
                self.theta = min(max(theta, L.ARM_LOWER), L.ARM_UPPER)
                self.pushes += 1
        return self.theta
