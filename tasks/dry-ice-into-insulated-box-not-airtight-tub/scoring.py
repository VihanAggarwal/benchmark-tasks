"""The scorer, as plain functions of the block's, the box's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), one observation per control step. It reads the
mechkit AssemblyState of the block (loose), the insulated box (fixed, two flap hinges) and the airtight tub (fixed,
its lid's hinge).

    IN TUB      the block's world-aligned box inside the airtight tub's inner outline (2 mm allowed into a wall) with
                its lowest point below the tub's rim
    SEALED      the tub's gasketed lid within 5 degrees of shut with its press-to-seal button pushed 2.5 mm (it then
                stays sealed until the lid is lifted 10 degrees or more)
    IN BOX      the block's whole box inside the insulated box's inner walls and wholly below its rim
    FLAP SHUT   a flap within 20 degrees of flat (its joint 80 degrees or more past as drawn)
    LATCH (0 for the rest of the trial)
                the block in the tub while its lid is sealed, on any step (dry ice sealed in an airtight container
                builds pressure until it bursts), or the block off the table
    RELEASED    both grippers open (>= 0.85) and every finger zone 4 cm or more from the block and from the box
    STILL       over the last 1 s the block moved 3 mm or less

    1.00  in box, both flaps shut, released and still on every step of the last 1.0 s
    0.75  in box, both flaps shut
    0.50  in box
    0.25  the block carried over the box's opening (its centre inside the inner outline, any height)
    0.00  anything else, latched, or unreadable
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BLOCK, BOX, TUB = "dry_ice", "insulated_box", "airtight_tub"


def _finite(v) -> bool:
    try:
        return bool(np.all(np.isfinite(np.asarray(v, dtype=float))))
    except (TypeError, ValueError):
        return False


def inside(low, high, in_x, in_y, rim: float, wholly: bool = True) -> bool:
    """A world-aligned box (low, high; z above the table) inside an inner outline (2 mm allowed into a wall) and
    wholly below a rim (``wholly``) or with its lowest point below it."""
    a = L.IN_ALLOW
    top = high[2] if wholly else low[2]
    return (in_x[0] - a <= low[0] and high[0] <= in_x[1] + a and in_y[0] - a <= low[1] and high[1] <= in_y[1] + a
            and top <= rim + a)


def flap_open_deg(q_rad: float) -> float:
    """A flap's opening from flat (degrees) for its joint value (0 as drawn, at 100 degrees open)."""
    return L.FLAP_OPEN_DEG - math.degrees(float(q_rad))


def _off_table(centre, low_z: float) -> bool:
    x, y = float(centre[0]), float(centre[1])
    return not (L.TABLE_X[0] <= x <= L.TABLE_X[1] and L.TABLE_Y[0] <= y <= L.TABLE_Y[1]) or low_z < -L.OFF_TABLE_DROP


@dataclass
class Result:
    score: float = 0.0
    success: bool = False
    details: dict = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.last = Result()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.win: deque = deque(maxlen=L.WIN_STEPS)
        self.latched = ""
        self.latch_step: int | None = None
        self.start: np.ndarray | None = None
        self.sealed = False
        self.last = Result()

    def _latch(self, why: str) -> None:
        if not self.latched:
            self.latched = why
            self.latch_step = self.steps

    def _fail(self, why: str) -> Result:
        self.win.clear()
        self.last = Result(0.0, False, {"latched": self.latched, "unreadable": why})
        return self.last

    def update(self, states, arms) -> Result:
        self.steps += 1
        try:
            blk, box, tub = states[BLOCK], states[BOX], states[TUB]
            low = np.asarray(blk.low, dtype=float) - np.array([0.0, 0.0, L.TOP])
            high = np.asarray(blk.high, dtype=float) - np.array([0.0, 0.0, L.TOP])
            centre = (low + high) / 2.0
            flaps = [flap_open_deg(box.joint(n)) for n in L.FLAP_NAMES]
            lid = L.LID_OPEN_DEG - math.degrees(tub.joint(L.LID_NAME))
            button = float(tub.joint(L.BUTTON))
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._fail("block or box")
        if not (_finite(low) and _finite(high) and _finite(flaps) and _finite(lid) and _finite(button)):
            return self._fail("numbers")
        if self.start is None:
            self.start = centre.copy()

        in_tub = inside(low, high, L.TUB_IN_X, L.TUB_IN_Y, L.TUB_RIM, wholly=False)
        in_box = inside(low, high, L.BOX_IN_X, L.BOX_IN_Y, L.RIM)
        shut = [f <= L.FLAP_SHUT_DEG + 1e-9 for f in flaps]
        if lid <= L.LID_SEALED_DEG + 1e-9 and button >= L.BUTTON_SEAL_AT - L.EPS:
            self.sealed = True
        elif lid > 10.0:
            self.sealed = False
        sealed = self.sealed
        if in_tub and sealed:
            self._latch("dry ice sealed in the airtight tub")
        if _off_table(centre, float(low[2])):
            self._latch("dry ice off the table")
        over_box = (L.BOX_IN_X[0] <= centre[0] <= L.BOX_IN_X[1] and L.BOX_IN_Y[0] <= centre[1] <= L.BOX_IN_Y[1])

        read = rigstate.read_arms(arms)
        arms_open = read is not None and all(float(a[2]) >= L.OPEN_THRESHOLD - L.EPS for a in read)
        released = (arms_open and rigstate.released(arms, blk.distance, L.RELEASE_CLEAR)
                    and rigstate.released(arms, box.distance, L.RELEASE_CLEAR))

        self.win.append((in_box and all(shut) and released, centre))
        full = still = False
        if len(self.win) == L.WIN_STEPS:
            full = all(w[0] for w in self.win)
            c0 = self.win[0][1]
            still = max(float(np.linalg.norm(w[1] - c0)) for w in self.win) <= L.STILL_MOVE + 1e-12

        if self.latched:
            score = 0.0
        elif full and still:
            score = 1.0
        elif in_box and all(shut):
            score = 0.75
        elif in_box:
            score = 0.5
        elif over_box:
            score = 0.25
        else:
            score = 0.0
        details = {
            "in_box": in_box, "in_tub": in_tub, "flaps_open_deg": [round(f, 1) for f in flaps], "shut": shut,
            "over_box": bool(over_box), "tub_lid_open_deg": round(lid, 1), "sealed": sealed, "released": released, "still": still,
            "block_centre_mm": [round(float(v) * 1000, 1) for v in centre],
            "latched": self.latched, "latch_step": self.latch_step,
        }
        self.last = Result(score, score >= 1.0, details)
        return self.last
