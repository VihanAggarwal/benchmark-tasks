"""The front guard seating spring, top retaining clip interlock, and rear rocker switch spring. NumPy free.

THE TOP RETAINING CLIP (``clip_target``, ``Latch``): a hinged spring-steel toggle clip at the top of the rear shroud
(joint ``clip``; 0 = open +12 deg, +24 deg = closed -12 deg; ``clip_closed_deg = degrees(q)``). Pushing its rear press
tab down past its 12.0 deg centre needs about 3.0 N at the 32 mm press point (layout.CLIP_K), after which it snaps
closed and its forward hook lip (8 mm downward catch at y +0.8 cm) engages over the front guard's top rim.
One declared interlock rule: the clip stays closed ONLY while the front guard is seated within 3 degrees of closed
(``open_deg <= 3.0``); with the guard ajar the clip has nothing to hook and springs back open.

THE FRONT GUARD SEATING SPRING (``guard_target``): while unlatched, the guard's spring pushes it open toward 22 deg
(about 2.5 N at the push point when held flush), so releasing the guard before latching the top clip lets
it spring back open. Once the top clip is latched on a seated guard, the guard's target switches past its closed stop
to hold it flush shut (representing the perimeter clip's front hook engaging the wire rim).

THE REAR ROCKER SWITCH (``rocker_target``): a plain hinge with an over-centre spring (3.5 N at 11 mm). Its target is
set 3 deg past the stop on whichever side of its 12 deg centre it is on. Nothing reads the guard or clip.
"""

from __future__ import annotations

import math

import layout as L


def clip_closed_deg(joint_rad: float) -> float:
    """Convert the ``clip`` joint reading (rad, >= 0) to positive closed travel in degrees [0..35]."""
    return math.degrees(joint_rad) if math.isfinite(joint_rad) else 0.0


def clip_target(joint_rad: float, open_deg: float) -> float:
    """The top retaining clip's spring target (rad): past its closed stop (+35 + preload deg) when past its centre
    AND the front guard is seated within LATCH_SEAT_DEG; otherwise past its open stop (-preload deg)."""
    cdeg = clip_closed_deg(joint_rad)
    hooked = math.isfinite(open_deg) and open_deg <= L.LATCH_SEAT_DEG + L.EPS
    if cdeg >= L.CLIP_CENTRE_DEG and hooked:
        return math.radians(L.CLIP_TRAVEL_DEG + L.CLIP_PRELOAD_DEG)
    return -math.radians(L.CLIP_PRELOAD_DEG)


def guard_target(latched: bool) -> float:
    """The front guard's spring target (rad): when latched, past the closed stop (holds flush); when unlatched, 0
    (springs back open to GUARD_OPEN_DEG)."""
    if latched:
        return math.radians(L.GUARD_OPEN_DEG + L.GUARD_HOLD_PRELOAD_DEG)
    return 0.0


def rocker_target(joint_rad: float) -> float:
    """The rocker spring's target (rad) for the rocker's joint reading (rad): past the stop on its side."""
    deg = math.degrees(joint_rad) if math.isfinite(joint_rad) else 0.0
    if deg >= L.ROCKER_CENTRE_DEG:
        return math.radians(L.ROCKER_TRAVEL_DEG + L.ROCKER_PRELOAD_DEG)
    return math.radians(-L.ROCKER_PRELOAD_DEG)


class Latch:
    """Tracks whether the top retaining clip is closed and latched onto the seated front guard."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.clip_closed = False
        self.latched = False
        self.events: list = []
        self.steps = 0

    def update(self, open_deg: float, cdeg: float | None) -> bool:
        """One control step: guard open angle (deg; NaN unreadable) and clip closed travel (deg; None/NaN unreadable).
        Returns the latched state."""
        self.steps += 1
        if cdeg is None or not math.isfinite(cdeg):
            self.clip_closed = False
        elif cdeg >= L.CLIP_ON_DEG - L.EPS:
            self.clip_closed = True
        elif cdeg <= L.CLIP_OFF_DEG + L.EPS:
            self.clip_closed = False
        seated = math.isfinite(open_deg) and open_deg <= L.LATCH_SEAT_DEG + L.EPS
        now = self.clip_closed and seated
        if now != self.latched:
            self.events.append(("latched" if now else "undone", self.steps))
        self.latched = now
        return self.latched
