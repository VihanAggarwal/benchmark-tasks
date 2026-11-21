"""The score, from the parts' poses and joints and the arms' fingers (rigstate). NumPy only (plus mechkit/rigstate).

Read every control step by a Tracker (task.py feeds it from success()):
- live: the plug drawn out less than layout.UNPLUGGED_AT (16 mm, the blades' length);
- the wrench on the nut: its jaw's centre within layout.WRENCH_ON_R of the spindle axis, at nut height;
- a hand near the bit: any finger zone (rigstate) within layout.NEAR_BIT of the bit's shapes.
Latched 0 for the episode: the wrench on the nut or a hand near the bit while live; the bit or the wrench off the
bench. Otherwise, at the latest step: 0 live; 0.25 unplugged; 0.5 unplugged and the nut turned layout.BIT_FREE_AT
(0.7 rad, 40 deg) or more loose on the spindle; 0.75 the bit out of the collet (or anywhere but in it) but not done;
1.0 on every step of the last 1 s: the bit standing shank-down in a block hole, the wrench lying flat off the router,
both hands 50 mm or more from the bit, bit and wrench still.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

ROUTER, LOCK, BIT, WRENCH, BENCH = "router", "lock", "bit", "wrench", "bench"
PARTS = (ROUTER, LOCK, BIT, WRENCH, BENCH)
WRENCH_ON_R = 0.015  # the jaw's centre this close to the axis: its prongs reach the nut's flats
BIT_LEN = L.CUTTER_Z[1] - L.SHANK_Z[0]


def plug_out(bench) -> float:
    return float(bench.joints.get("plug", 0.0))


def live(bench) -> bool:
    return plug_out(bench) < L.UNPLUGGED_AT


def nut_loose(router) -> float:
    """The nut's turn on the spindle (rad, + loosening)."""
    return float(router.joints.get("nut", 0.0))


def axis_world() -> np.ndarray:
    return np.array([L.RX, L.RY, L.TOP])


def wrench_on_nut(wrench) -> bool:
    jaw = np.asarray(wrench.origin, dtype=float)
    a = axis_world()
    zmid = L.TOP + (L.NUT_Z[0] + L.NUT_Z[1]) / 2
    return bool(math.hypot(jaw[0] - a[0], jaw[1] - a[1]) <= WRENCH_ON_R and abs(jaw[2] - zmid) <= L.ENGAGE_DZ)


def bit_ends(bit) -> tuple[np.ndarray, np.ndarray]:
    """(shank's bottom end, cutter's top end), world."""
    o = np.asarray(bit.origin, dtype=float)
    up = np.asarray(bit.rotation(), dtype=float)[:, 2]
    return o, o + BIT_LEN * up


def bit_out(bit) -> bool:
    bottom, _ = bit_ends(bit)
    a = axis_world()
    off_axis = math.hypot(bottom[0] - a[0], bottom[1] - a[1]) > 0.010
    return bool(off_axis or bottom[2] > L.TOP + L.NUT_Z[1] + L.OUT_OF_COLLET)


def bit_in_block(bit) -> bool:
    bottom, _ = bit_ends(bit)
    if bit.tilt() > L.HOLE_TILT:
        return False
    top = L.TOP + L.BLOCK_SIZE[2]
    if bottom[2] > top - L.IN_HOLE_DEPTH:
        return False
    return any(math.hypot(bottom[0] - h[0], bottom[1] - h[1]) <= L.IN_HOLE_XY for h in L.HOLES)


def wrench_down(wrench) -> bool:
    """Lying flat on the bench (or back on its rest block), its jaw clear of the router."""
    o = np.asarray(wrench.origin, dtype=float)
    if wrench.tilt() > math.radians(10.0):
        return False
    on_bench = o[2] <= L.TOP + L.WRENCH_T / 2 + L.WRENCH_DOWN_Z
    on_rest = abs(o[2] - (L.TOP + L.WRENCH_Z_REST)) <= L.WRENCH_DOWN_Z and L.REST_X[0] - 0.02 <= o[0] <= L.REST_X[1]
    clear = math.hypot(o[0] - L.RX, o[1] - L.RY) >= L.WRENCH_CLEAR_R
    return bool((on_bench or on_rest) and clear)


def fell(part) -> bool:
    return bool(float(np.asarray(part.origin, dtype=float)[2]) < L.TOP + L.FELL_Z)


def hand_gap(arms, part) -> float | None:
    """The nearest finger zone's gap (m) to the part's shapes, None if the arms are unreadable."""
    if arms is None or not rigstate.readable(arms):
        return None
    return rigstate.hands_gap(arms, part.distance)


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.latched: str | None = None
        self.window: list = []  # (done, bit bottom, wrench origin) for the last HOLD_STEPS steps
        self.last = Result(0.0, False, {})
        self.unplugged_ever = False

    def update(self, parts: dict, arms) -> Result:
        self.steps += 1
        router, bench, bit, wrench = parts[ROUTER], parts[BENCH], parts[BIT], parts[WRENCH]
        is_live = live(bench)
        on_nut = wrench_on_nut(wrench)
        gap = hand_gap(arms, bit)
        near = gap is not None and gap < L.NEAR_BIT
        if self.latched is None:
            if is_live and on_nut:
                self.latched = f"wrench on the nut while plugged in (step {self.steps})"
            elif is_live and near:
                self.latched = f"a finger within 10 mm of the bit while plugged in (step {self.steps})"
            elif fell(bit) or fell(wrench):
                self.latched = f"the bit or the wrench fell off the bench (step {self.steps})"
        loose = nut_loose(router)
        out = bit_out(bit)
        in_block = bit_in_block(bit)
        down = wrench_down(wrench)
        away = gap is not None and gap >= L.HANDS_AWAY
        done = bool(in_block and down and away and not is_live)
        bottom, _ = bit_ends(bit)
        self.window.append((done, bottom.copy(), np.asarray(wrench.origin, dtype=float).copy()))
        if len(self.window) > L.HOLD_STEPS:
            self.window.pop(0)
        held = (len(self.window) == L.HOLD_STEPS and all(w[0] for w in self.window)
                and max(np.linalg.norm(w[1] - self.window[0][1]) for w in self.window) < L.STILL
                and max(np.linalg.norm(w[2] - self.window[0][2]) for w in self.window) < L.STILL)
        if self.latched is not None or is_live:
            s = 0.0
        elif held:
            s = 1.0
        elif out:
            s = 0.75
        elif loose >= L.BIT_FREE_AT:
            s = 0.5
        else:
            s = 0.25
        details = {"live": is_live, "plug_out_mm": round(plug_out(bench) * 1000, 1), "nut_loose_rad": round(loose, 3),
                   "button_mm": round(float(parts[LOCK].joints.get("button", 0.0)) * 1000, 2),
                   "spindle_rad": round(float(router.joints.get("spindle", 0.0)), 3), "wrench_on_nut": on_nut,
                   "bit_out": out, "bit_in_block": in_block, "wrench_down": down,
                   "hand_gap_mm": None if gap is None else round(gap * 1000, 1), "latched": self.latched}
        self.last = Result(s, s >= 1.0, details)
        return self.last
