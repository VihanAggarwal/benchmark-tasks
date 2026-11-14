"""Tilted grips for the scripted demo: a waypoint whose gripper is the stock straight-down grip turned ``yaw`` about the
vertical and then rolled about world x.

Both arms here grip with their pads closing along world x (yaw -pi/2 on the left, +pi/2 on the right). Rolled about
world x, the pads stay normal to x, so the lock's hole axis and the cord stay along y while the two wrists lean away
from each other: two straight-down grippers cannot come within about 170 mm of each other along y, and the cord must
reach the lock in one push.

How it works. The stock ``yam_arena.demo.ArmPlan`` builds every waypoint's rotation by calling the module's
``grip_rotation(w.yaw)`` at run time. A tilted waypoint stores its full rotation on its ``yaw`` value (a float subclass
that still equals the yaw), and importing this module wraps ``grip_rotation`` so that it returns the stored rotation
when there is one and calls the original otherwise. Plain floats behave exactly as before. tools/offline's stand-in
``yam_arena.demo`` builds rotations through its kinematics module (``demo._k.grip_rotation``); that one is wrapped the
same way, so the offline checker plays the same tilted plan. Importing checks that the plan honours a tilted waypoint
and raises if it does not. The idea follows the program repository's tasks that patch ``grip_rotation`` for tilted
and side grips; this is an independent implementation.
"""

from __future__ import annotations

import math

import numpy as np
import yam_arena.demo as _demo

MARK = "_tiltgrip_original"


class TiltedYaw(float):
    """A yaw (rad) that also carries the full TCP rotation of a tilted waypoint."""

    rotation: np.ndarray

    def __new__(cls, yaw: float, rotation: np.ndarray):
        obj = super().__new__(cls, yaw)
        obj.rotation = np.array(rotation, dtype=float)
        return obj

    def __reduce__(self):
        return (TiltedYaw, (float(self), self.rotation))


def rx(angle: float) -> np.ndarray:
    """Rotation about world x by ``angle`` rad (right-handed: +angle turns +y toward +z)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _wrap(module, name: str) -> None:
    fn = getattr(module, name)
    original = getattr(fn, MARK, fn)

    def grip_rotation(yaw: float = 0.0) -> np.ndarray:
        rot = getattr(yaw, "rotation", None)
        return rot.copy() if rot is not None else original(yaw)

    setattr(grip_rotation, MARK, original)
    setattr(module, name, grip_rotation)


_ORIGINAL = getattr(_demo.grip_rotation, MARK, _demo.grip_rotation)
_wrap(_demo, "grip_rotation")
_KIN = getattr(_demo, "_k", None)  # tools/offline's stand-in only
if _KIN is not None and hasattr(_KIN, "grip_rotation"):
    _wrap(_KIN, "grip_rotation")


def rotation(yaw: float, roll: float) -> np.ndarray:
    """The straight-down grip turned ``yaw`` about the vertical, then rolled ``roll`` rad about world x."""
    return rx(roll) @ np.asarray(_ORIGINAL(float(yaw)), dtype=float)


def tilted(yaw: float, roll: float) -> float:
    """A waypoint ``yaw`` value for that rotation (a plain float when ``roll`` is 0)."""
    return float(yaw) if roll == 0.0 else TiltedYaw(yaw, rotation(yaw, roll))


def approach(yaw: float, roll: float) -> np.ndarray:
    """The approach axis (TCP z: from the TCP toward the pad tips) of that grip, world frame."""
    return rotation(yaw, roll)[:, 2]


def _self_check() -> None:
    w = _demo.Waypoint(1.0, (0.30, 0.10, 0.80), 0.0, "check", tilted(-math.pi / 2, -0.5))
    plan = _demo.ArmPlan([w], (0.0, 0.305, 0.75))
    _, rot, _, _ = plan.target(10.0)
    if not np.allclose(rot, w.yaw.rotation, atol=1e-9):
        raise RuntimeError("tiltgrip.py: yam_arena.demo.ArmPlan ignores tilted waypoints on this stack; do not use it")


_self_check()
