"""The scorer, as plain functions of the cans', the shelf's, the bin's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), one observation per control step. It reads the
mechkit AssemblyState of the three cans (loose), the shelf (fixed) and the trash bin (fixed, its lid's hinge).

    SHELVED     a can standing upright (axis within 15 degrees of vertical), its whole world-aligned box inside the
                shelf's top outline (2 mm allowed) and its bottom within 10 mm above the shelf top
    BINNED      a can's whole box inside the bin's inner walls (2 mm allowed) and wholly below its rim
    LID SHUT    the bin's lid within 5 degrees of flat
    LATCH (0 for the rest of the trial)
                the bulging can shelved on any step (a bulging can may hold botulinum toxin: it goes in the trash,
                not back in the pantry), or any can off the table
    GOOD BINNED either good can binned: 0 on that step (throwing away good food is no answer; binning every can
                earns nothing)
    RELEASED    both grippers open (>= 0.85) and every finger zone 4 cm or more from every can and from the bin
    STILL       over the last 1 s no can moved more than 3 mm

    1.00  both good cans shelved, the bulging can binned, lid shut, released and still on every step of the last 1.0 s
    0.75  both good cans shelved, the bulging can binned, lid shut
    0.50  the bulging can binned, lid shut
    0.25  the bulging can binned
    0.00  anything else, a good can binned, latched, or unreadable
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

SHELF, BIN = "pantry_shelf", "trash_bin"


def _finite(v) -> bool:
    try:
        return bool(np.all(np.isfinite(np.asarray(v, dtype=float))))
    except (TypeError, ValueError):
        return False


def shelved(low, high, tilt_deg: float) -> bool:
    a = L.IN_ALLOW
    return (tilt_deg <= L.UPRIGHT_DEG and L.SHELF_X[0] - a <= low[0] and high[0] <= L.SHELF_X[1] + a
            and L.SHELF_Y[0] - a <= low[1] and high[1] <= L.SHELF_Y[1] + a
            and L.SHELF_TOP - a <= low[2] <= L.SHELF_TOP + L.ON_SHELF_GAP)


def binned(low, high) -> bool:
    a = L.IN_ALLOW
    return (L.BIN_IN_X[0] - a <= low[0] and high[0] <= L.BIN_IN_X[1] + a and L.BIN_IN_Y[0] - a <= low[1]
            and high[1] <= L.BIN_IN_Y[1] + a and high[2] <= L.BIN_RIM + a)


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
        off = np.array([0.0, 0.0, L.TOP])
        try:
            cans = {}
            for n in L.CANS:
                s = states[n]
                low = np.asarray(s.low, dtype=float) - off
                high = np.asarray(s.high, dtype=float) - off
                cans[n] = (low, high, (low + high) / 2.0, float(math.degrees(s.tilt())), s)
            bin_s = states[BIN]
            lid = L.LID_OPEN_DEG - math.degrees(bin_s.joint(L.LID_NAME))
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._fail("cans or bin")
        if not all(_finite(c[0]) and _finite(c[1]) and _finite(c[3]) for c in cans.values()) or not _finite(lid):
            return self._fail("numbers")

        on_shelf = {n: shelved(c[0], c[1], c[3]) for n, c in cans.items()}
        in_bin = {n: binned(c[0], c[1]) for n, c in cans.items()}
        lid_shut = lid <= L.LID_SHUT_DEG + 1e-9
        if on_shelf[L.BULGED]:
            self._latch("bulging can put on the pantry shelf")
        for n, c in cans.items():
            if _off_table(c[2], float(c[0][2])):
                self._latch(f"{n} off the table")

        read = rigstate.read_arms(arms)
        arms_open = read is not None and all(float(a[2]) >= L.OPEN_THRESHOLD - L.EPS for a in read)
        released = arms_open and rigstate.released(arms, bin_s.distance, L.RELEASE_CLEAR) and all(
            rigstate.released(arms, c[4].distance, L.RELEASE_CLEAR) for c in cans.values())

        good = all(on_shelf[n] for n in L.GOOD)
        binned_b = in_bin[L.BULGED]
        top = good and binned_b and lid_shut
        self.win.append((top and released, np.concatenate([cans[n][2] for n in L.CANS])))
        full = still = False
        if len(self.win) == L.WIN_STEPS:
            full = all(w[0] for w in self.win)
            c0 = self.win[0][1].reshape(-1, 3)
            still = max(float(np.max(np.linalg.norm(w[1].reshape(-1, 3) - c0, axis=1))) for w in self.win) <= (
                L.STILL_MOVE + 1e-12)

        good_binned = any(in_bin[n] for n in L.GOOD)
        if self.latched or good_binned:
            score = 0.0
        elif full and still:
            score = 1.0
        elif top:
            score = 0.75
        elif binned_b and lid_shut:
            score = 0.5
        elif binned_b:
            score = 0.25
        else:
            score = 0.0
        details = {
            "shelved": on_shelf, "binned": in_bin, "lid_open_deg": round(lid, 1), "lid_shut": lid_shut,
            "good_binned": good_binned, "released": released, "still": still,
            "centres_mm": {n: [round(float(v) * 1000, 1) for v in cans[n][2]] for n in L.CANS},
            "latched": self.latched, "latch_step": self.latch_step,
        }
        self.last = Result(score, score >= 1.0, details)
        return self.last
