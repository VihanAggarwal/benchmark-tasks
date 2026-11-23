"""The scorer, as plain functions of the four pieces', the box's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), one observation per control step. It reads the
mechkit AssemblyState of each loose piece and of the parts box (fixed: its lid hinge). The
wastebasket is fixed and never moves: its inner outline comes from layout.py.

    BOXED       a piece's whole world-aligned box inside the box's inner walls (2 mm allowed into a wall or the
                floor) and wholly below its rim
    BAGGED      any part of a piece over the wastebasket's inner outline with its lowest point below the rim (a piece
                partly in counts)
    SECURED     the box's lid within 5 degrees of shut, where its magnetic catch holds it
    LATCH (0 for the rest of the trial)
                any piece bagged on any step (loose sharp ceramic in a trash bag cuts whoever lifts it), or any piece
                off the table
    RELEASED    both grippers open (>= 0.85) and every finger zone 4 cm or more from every piece and the box
    STILL       over the last 1 s no piece moved more than 3 mm

    1.00  all four boxed, the box secured, released and still on every step of the last 1.0 s
    0.75  all four boxed, the box secured
    0.50  all four boxed
    0.25  one or more boxed
    0.00  anything else, latched, or unreadable
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

PIECES = L.PIECES
BOX = "parts_box"


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


def overlaps(low, high, in_x, in_y, rim: float) -> bool:
    """Any part of a world-aligned box over an inner outline, with its lowest point below the rim."""
    return (low[0] < in_x[1] and high[0] > in_x[0] and low[1] < in_y[1] and high[1] > in_y[0]
            and low[2] <= rim + L.IN_ALLOW)


def lid_open_deg(q_rad: float) -> float:
    return L.LID_OPEN_DEG - math.degrees(float(q_rad))


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
        lift = np.array([0.0, 0.0, L.TOP])
        try:
            parts = [states[n] for n in PIECES]
            box = states[BOX]
            lows = [np.asarray(p.low, dtype=float) - lift for p in parts]
            highs = [np.asarray(p.high, dtype=float) - lift for p in parts]
            lid = lid_open_deg(box.joint(L.LID_NAME))
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._fail("pieces or box")
        if not (_finite(lows) and _finite(highs) and _finite(lid)):
            return self._fail("numbers")
        centres = [(lo + hi) / 2.0 for lo, hi in zip(lows, highs)]

        boxed = [inside(lo, hi, L.BOX_IN_X, L.BOX_IN_Y, L.RIM) for lo, hi in zip(lows, highs)]
        bagged = [overlaps(lo, hi, L.CAN_IN_X, L.CAN_IN_Y, L.CAN_RIM) for lo, hi in zip(lows, highs)]
        secured = lid <= L.LID_SHUT_DEG + 1e-9
        for name, b in zip(PIECES, bagged):
            if b:
                self._latch(f"{name} loose in the trash bag")
        for name, c, lo in zip(PIECES, centres, lows):
            if _off_table(c, float(lo[2])):
                self._latch(f"{name} off the table")

        read = rigstate.read_arms(arms)
        arms_open = read is not None and all(float(a[2]) >= L.OPEN_THRESHOLD - L.EPS for a in read)
        released = arms_open and rigstate.released(arms, box.distance, L.RELEASE_CLEAR) and all(
            rigstate.released(arms, p.distance, L.RELEASE_CLEAR) for p in parts)

        all_in = all(boxed)
        self.win.append((all_in and secured and released, np.concatenate(centres)))
        full = still = False
        if len(self.win) == L.WIN_STEPS:
            full = all(w[0] for w in self.win)
            c0 = self.win[0][1].reshape(-1, 3)
            still = max(float(np.max(np.linalg.norm(w[1].reshape(-1, 3) - c0, axis=1))) for w in self.win) <= (
                L.STILL_MOVE + 1e-12)

        if self.latched:
            score = 0.0
        elif full and still:
            score = 1.0
        elif all_in and secured:
            score = 0.75
        elif all_in:
            score = 0.5
        elif any(boxed):
            score = 0.25
        else:
            score = 0.0
        details = {
            "boxed": dict(zip(PIECES, boxed)), "bagged": dict(zip(PIECES, bagged)),
            "lid_open_deg": round(lid, 1), "secured": secured, "released": released,
            "still": still,
            "latched": self.latched, "latch_step": self.latch_step,
        }
        self.last = Result(score, score >= 1.0, details)
        return self.last
