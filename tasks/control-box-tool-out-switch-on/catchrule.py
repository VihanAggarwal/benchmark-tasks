"""The cover's snap catches (the one declared software rule) and the rocker's over-centre spring. NumPy free.

THE CATCH RULE (``Catch``): the simulator has no snap catch, so the catches are a state switched by a measured pose of
the robot's own hand. The cover is LATCHED from the step on which a gripper's pad-tip point has rested on the shut
cover (cover within 2 deg of shut; the tips over the cover's outline, within 6 cm of its catch edge and no more than
3 mm above its top) for 0.25 s (5 control steps). The 5 N the real catches need is not measured: the demo commands
its press 8 mm past contact (the arm gives about 1 mm per newton). The cover is not held shut by the rule (its own
weight holds it); the catches count as undone once the cover is 5 deg or more open. On a bench: the catches click
and the cover's edge does not lift.

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


def in_catch_zone(tips) -> bool:
    """A pad-tip point (world, m) over the shut cover's outline, within 6 cm of its catch edge, resting on its top."""
    x, y, z = float(tips[0]), float(tips[1]), float(tips[2]) - L.TOP
    return (L.CATCH_ZONE_X[0] <= x <= L.CATCH_ZONE_X[1] and L.CATCH_ZONE_Y[0] <= y <= L.CATCH_ZONE_Y[1]
            and z <= L.BOX_H + L.PRESS_ABOVE + L.EPS and z >= L.RIM - 0.010)


class Catch:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.latched = False
        self.pressed = 0
        self.events: list = []
        self.steps = 0

    def update(self, open_deg: float, tips_list) -> bool:
        """One control step: the cover's open angle (deg; NaN unreadable) and the pad-tip points of both grippers
        (None when the arms cannot be read). Returns the latched state."""
        self.steps += 1
        if not math.isfinite(open_deg):
            self.pressed = 0
            return self.latched
        if self.latched and open_deg >= L.UNLATCH_DEG - L.EPS:
            self.latched = False
            self.events.append(("undone", self.steps))
        shut = open_deg <= L.SHUT_DEG + L.EPS
        press = shut and tips_list is not None and any(in_catch_zone(t) for t in tips_list)
        self.pressed = self.pressed + 1 if press else 0
        if not self.latched and self.pressed >= L.PRESS_STEPS:
            self.latched = True
            self.events.append(("latched", self.steps))
        return self.latched
