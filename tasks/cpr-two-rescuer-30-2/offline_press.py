"""Offline checker only: how far the finger tips hold the chest and the bag down. The simulator never calls anything
in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics: it sees the manikin as mechkit's
assembly at rest, every joint at 0, whatever the grippers do (``Assembly.offline_state``). So when task.py's hooks
get a manikin state whose ``source`` is "offline", the scorer is fed the joints from this small static model
instead, using the finger-tip points the checker feeds to rigstate:

- each slider rests at its sag under its own weight (spring and gravity balance: 2.0 mm for the chest, 3.7 mm for
  the cap);
- finger tips over a slider's top face (the plate's 115 x 115 mm, 1 mm higher on the 30 mm mark; the cap's
  60 x 60 mm) and below it hold it down: the slider's spring and the arm's give (about 1.9 mm per newton at the tips
  at both stations, from J K^-1 J^T of the arm's joint gains) act as two springs in series, so the slider goes
  less far down than the tips would without it;
- each slider stops at its travel limit; of the two grippers, the one pressing deeper counts.

No lag, no dynamics: the arm's tracking of the 1.8 Hz stroke on the GPU decides the real depths (the README says
how to tune them).
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

ARMS = ("left", "right")


def _valid(arms) -> bool:
    try:
        return all(np.all(np.isfinite(np.asarray(arms[s].tips, dtype=float))) for s in ARMS)
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def pressed(tip, cx, cy, half, top, spring, rest, travel) -> float:
    """How far (m) finger tips at ``tip`` hold down a spring slider whose top face (``half`` each way around
    (cx, cy)) is ``top`` above the table as drawn."""
    if abs(tip[0] - cx) > half or abs(tip[1] - cy) > half:
        return rest
    overlap = L.TOP + top - float(tip[2])
    if overlap <= rest:
        return rest
    return min(travel, (L.ARM_STIFFNESS * overlap + spring * rest) / (L.ARM_STIFFNESS + spring))


def kinematic_joints(arms) -> dict:
    """The chest's and the cap's travel (m, positive pressed down) for the arms' finger tips (rigstate.arms())."""
    chest, bag = L.CHEST_REST, L.BAG_REST
    if _valid(arms):
        for s in ARMS:
            tip = np.asarray(arms[s].tips, dtype=float)
            on_mark = math.hypot(tip[0] - L.MARK_X, tip[1] - L.MARK_Y) <= L.MARK_RADIUS
            chest = max(chest, pressed(tip, L.MARK_X, L.MARK_Y, L.PLATE_SIZE[0] / 2,
                                       L.MARK_TOP if on_mark else L.PLATE_TOP, L.CHEST_SPRING, L.CHEST_REST,
                                       L.CHEST_TRAVEL))
            bag = max(bag, pressed(tip, L.BAG_X, L.BAG_Y, L.CAP_SIZE[0] / 2, L.CAP_TOP, L.BAG_SPRING, L.BAG_REST,
                                   L.BAG_TRAVEL))
    return {"chest": chest, "bag": bag}
