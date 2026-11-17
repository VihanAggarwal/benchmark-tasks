"""The cover's two snap catches and the rocker's over-centre spring. NumPy free.

THE CATCHES (``lever_target``, ``Catch``): each catch is its own hinged spring toggle lever on the box's right wall
(joints ``catch_a`` at x 17 and ``catch_b`` at x 32), like the rocker. A lever stands raised 10 deg (open). The robot
has to press its tab down with a pad of its own gripper: the spring gives 5 N at the press point at the lever's
centre (layout.LEV_K), after which the lever snaps closed by itself. Nothing reads the arm: the lever
is a physical part and its joint angle is what is read. The one declared rule is when the lever may stay closed: the
simulator has no hook, so every control step a lever's drive target is set past its stop on the side it is on, but
only while the cover is shut (open under 5 deg); with the cover open the lever springs back, as a snap catch that has
nothing to hook. A lever reads CLOSED 2 deg past its centre and OPEN again 2 deg before it. The cover is LATCHED when
BOTH levers read closed and the cover is within 2 deg of shut, so each catch needs its own press of about 5 N. The
cover is not held shut by the levers (its own weight holds it); the catches count as undone once the cover is 5 deg
or more open (the levers spring open).

THE ROCKER (``rocker_target``): a real rocker power switch is bistable, held on its stops by an over-centre spring.
The rocker is a plain hinge with a spring; every control step its target is set 3 deg past the stop on the side the
rocker is on (OFF below its 10 deg centre, ON above it), so pushing the ON end down needs 4 N at the centre (the
1550 series sheet: 3 to 5 N) and the rocker then snaps to ON by itself. Nothing reads the box, the cover or the tool.
"""

from __future__ import annotations

import math

import layout as L


def rocker_target(joint_rad: float) -> float:
    """The rocker spring's target (rad) for the rocker's joint reading (rad): past the stop on its side."""
    deg = math.degrees(joint_rad)
    if deg >= L.ROCKER_CENTRE_DEG:
        return math.radians(L.ROCKER_TRAVEL_DEG + L.ROCKER_PRELOAD_DEG)
    return math.radians(-L.ROCKER_PRELOAD_DEG)


def lever_target(joint_rad: float, open_deg: float) -> float:
    """A catch lever's spring target (rad): past its closed stop when it is past its centre AND the cover is shut
    (open under 5 deg); otherwise past its open stop (a catch with nothing to hook springs back)."""
    deg = math.degrees(joint_rad) if math.isfinite(joint_rad) else 0.0
    hooked = math.isfinite(open_deg) and open_deg < L.UNLATCH_DEG - L.EPS
    if deg >= L.LEV_CENTRE_DEG and hooked:
        return math.radians(L.LEV_TRAVEL_DEG + L.LEV_PRELOAD_DEG)
    return math.radians(-L.LEV_PRELOAD_DEG)


class Catch:
    """The latch state from the two levers' joint readings (deg) and the cover's open angle (deg)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.closed = [False] * len(L.LEV_NAMES)
        self.latched = False
        self.pressed = 0  # how many levers read closed
        self.events: list = []
        self.steps = 0

    def update(self, open_deg: float, lever_degs) -> bool:
        """One control step: the cover's open angle (deg; NaN unreadable) and each lever's joint reading in deg (None
        or a NaN when unreadable: that lever reads open, so an unreadable state is never latched). Returns the
        latched state."""
        self.steps += 1
        readings = list(lever_degs) if lever_degs is not None else []
        readings += [math.nan] * (len(self.closed) - len(readings))
        for i, deg in enumerate(readings[:len(self.closed)]):
            if deg is None or not math.isfinite(deg):
                self.closed[i] = False
            elif deg >= L.LEV_ON_DEG - L.EPS:
                self.closed[i] = True
            elif deg <= L.LEV_OFF_DEG + L.EPS:
                self.closed[i] = False
        self.pressed = sum(self.closed)
        shut = math.isfinite(open_deg) and open_deg <= L.SHUT_DEG + L.EPS
        now = all(self.closed) and shut
        if now != self.latched:
            self.events.append(("latched" if now else "undone", self.steps))
        self.latched = now
        return self.latched
