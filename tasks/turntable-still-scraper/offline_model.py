"""Offline checker only: how far the grippers turn the turntable. The simulator never calls anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics: it sees the turntable as mechkit's
assembly at rest, its joint at 0, whatever the grippers do (``Assembly.offline_state``). So when task.py's hooks get
a turntable state whose ``source`` is "offline", the scorer is fed the plate's angle from this small kinematic model
instead, using the finger-tip points and openings the checker feeds to rigstate:

- a gripper that closes past a peg's 20 mm width (its opening falling through 20 mm) with its finger-tip point
  within 10 mm of that peg's centre (seen from above) and between the plate's top and the peg's top grips the peg;
- while it stays closed on it (opening at most 22 mm) with its tips in that height band and within 15 mm of the
  pegs' circle, the plate turns with it: the peg keeps the angle about the axis it had in the fingers when they
  closed, so the plate's angle follows the finger-tip point's angle about the axis;
- it lets go when the gripper opens, lifts its tips above the peg or drifts off the circle; the plate then stays
  where it is (no spring; the real damping is small and the scripted strokes end at rest);
- the angle stops at the joint's travel (-0.2 to 6.1 rad). Nothing else moves it: open fingers, the smoother, the
  cake or an arm's links pushing it are not modelled (the demo never does that, and it must not count offline).

No lag, slip, sag or coasting: the arm's tracking on the GPU decides the real angle (the README says what to read in
the trace). The model resets with rigstate's reset hook and at every new offline run.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

ARMS = ("left", "right")
AXIS = np.array(L.AXIS, dtype=float)
PEG_W = 2 * L.PEG_R
Z_PLATE = L.TOP + L.PLATE_TOP
Z_PEG_TOP = L.TOP + L.PEG_TOP


def _arms_ok(arms) -> bool:
    try:
        return all(np.all(np.isfinite(np.asarray(arms[s].tips, dtype=float))) and math.isfinite(float(arms[s].opening))
                   for s in ARMS)
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def _in_band(tips) -> bool:
    return bool(Z_PLATE <= float(tips[2]) <= Z_PEG_TOP)


class PlateModel:
    """The plate's angle (rad, anticlockwise seen from above) as the grippers would turn it by its pegs."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.theta = 0.0
        self.omega = 0.0
        self.hold = None  # (arm, peg index, angle offset of the tips from that peg when it was gripped)
        self.prev_gap = {s: None for s in ARMS}
        self.grips = 0  # how many times a peg was gripped (for the trace and the tests)

    def peg_angle(self, k: int) -> float:
        return math.radians(L.PEG_ANGLES_DEG[k]) + self.theta

    def update(self, arms) -> float:
        """Advance one control step with the grippers as rigstate reports them; returns the angle."""
        before = self.theta
        if _arms_ok(arms):
            for side in ARMS:
                a = arms[side]
                tips = np.asarray(a.tips, dtype=float)
                gap = float(np.clip(a.opening, 0.0, 1.0)) * L.FULL_GAP
                rel = tips[:2] - AXIS
                ang, rad = math.atan2(rel[1], rel[0]), float(np.hypot(*rel))
                if self.hold is not None and self.hold[0] == side:
                    _, k, off = self.hold
                    if gap > PEG_W + 0.002 or not _in_band(tips) or abs(rad - L.PEG_CIRCLE) > L.GRIP_SLIP:
                        self.hold = None  # it let go
                    else:
                        want = ang - off - math.radians(L.PEG_ANGLES_DEG[k])
                        theta = self.theta + math.remainder(want - self.theta, 2 * math.pi)
                        self.theta = min(max(theta, L.PLATE_LOWER), L.PLATE_UPPER)
                elif self.hold is None:
                    prev = self.prev_gap[side]
                    if prev is not None and prev > PEG_W >= gap and _in_band(tips):
                        best = None
                        for k in range(len(L.PEG_ANGLES_DEG)):
                            c = AXIS + L.PEG_CIRCLE * np.array([math.cos(self.peg_angle(k)),
                                                                math.sin(self.peg_angle(k))])
                            d = float(np.hypot(*(tips[:2] - c)))
                            if d <= L.GRIP_CAPTURE and (best is None or d < best[0]):
                                best = (d, k)
                        if best is not None:
                            k = best[1]
                            self.hold = (side, k, math.remainder(ang - self.peg_angle(k), 2 * math.pi))
                            self.grips += 1
                self.prev_gap[side] = gap
        else:
            self.prev_gap = {s: None for s in ARMS}
        self.omega = (self.theta - before) / self.dt
        return self.theta
