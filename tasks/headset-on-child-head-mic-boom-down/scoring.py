"""The scorer, as plain functions of the headset's state and the arms. NumPy only; no simulator needed.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step by
``task.success()``. It reads one mechkit AssemblyState, the headset (base: the headband; links cup_p, the +y cup on
the child's right ear; cup_n, the -y cup on the child's left ear; boom, the microphone boom on cup_n). The head form
and the stand are fixed bodies: their poses are layout constants.

    LATCH     the headset left the work surface: any part of it past the table's edge in plan, or any part 10 mm or
              more below the table top (it fell off), at any step; a headset set down on the table top and picked up
              again does not latch (the 10 mm allows for contact penetration)
    OFF STAND the band's underside 15 mm or more above the cradle's top, or its centre 30 mm or more outside the
              cradle's 30 x 90 mm top in plan (sliding it along the cradle is not off)
    SEATED    the band's origin within 15 mm of its seated pose (centred over the head's axis, its underside on the
              crown) and tilted under 0.25 rad
    ON EARS   each cushion's face centre within 15 mm of its own ear's centre across the ear (x and z), and the face
              no more than 3 mm out from the ear's outer face (nor 4 mm into it): the cups are clamped on, not held
              spread (the right cup on the right ear, the left cup with the boom on the left ear)
    MIC DOWN  the boom within 0.25 rad of its lower stop and the microphone's tip within 50 mm of the left corner of
              the mouth
    RESTING   the band's and every link's origin moving under 5 mm/s
    HANDS OFF both grippers' finger zones 30 mm or more from every part of the headset (unreadable arms: not off)

    0      a latch fired, or the headset still on its stand
    0.25   off the stand, not seated with both cups on their ears
    0.5    seated with both cups on their ears, the microphone not down
    0.75   seated, on the ears and the microphone down at the last step, but resting or hands off (or any of the
           three) failed on a step of the last 1 s
    1.0    seated, on the ears, microphone down, resting and hands off on every one of the last 20 steps (1 s)

Every end state falls in exactly one tier. Nothing reads the order of moves.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

HEADSET = "headset"
PARTS = (HEADSET,)
CUSH_LOCAL = {"cup_p": np.array([0.0, L.CUSH_Y[0], L.CUP_ZC]),  # the centre of each cushion's face (inner side)
              "cup_n": np.array([0.0, -L.CUSH_Y[0], L.CUP_ZC])}
SIDE = {"cup_p": 1.0, "cup_n": -1.0}
EAR_FACE = {"cup_p": np.array([L.HX, L.HY + L.EAR_Y1, L.TOP + L.EAR_ZC]),
            "cup_n": np.array([L.HX, L.HY - L.EAR_Y1, L.TOP + L.EAR_ZC])}
MIC_LOCAL = np.array([L.BOOM_BAR_X, L.TIP_Y1, L.CUP_ZC + L.BOOM_LEN])
STAND_REST = np.array(L.HEADSET_ORIGIN_START, dtype=float)
SEAT = np.asarray(L.HEADSET_ORIGIN_SEATED, dtype=float)


@dataclass(frozen=True)
class Reading:
    origin: np.ndarray
    tilt: float
    cush: dict
    ear_gap: dict
    ear_out: dict
    mic: np.ndarray
    mic_gap: float
    boom: float
    origins: np.ndarray
    low: np.ndarray
    high: np.ndarray
    state: object


def read(states) -> Reading | None:
    if not isinstance(states, dict):
        return None
    try:
        hs = states.get(HEADSET)
        if hs is None or hs.spec is None:
            return None
        o = np.asarray(hs.origin, dtype=float).reshape(3)
        tilt = float(hs.tilt())
        cush = {c: hs.link_to_world(c, CUSH_LOCAL[c][None, :])[0] for c in CUSH_LOCAL}
        ear_gap = {c: float(np.hypot(*(cush[c] - EAR_FACE[c])[[0, 2]])) for c in cush}
        ear_out = {c: float(SIDE[c] * (cush[c] - EAR_FACE[c])[1]) for c in cush}
        mic = hs.link_to_world("boom", MIC_LOCAL[None, :])[0]
        mic_gap = float(np.linalg.norm(mic - L.MOUTH_CORNER_LEFT))
        boom = float(hs.joints["boom"])
        origins = np.vstack([o] + [np.asarray(hs.links[j.name][:3], dtype=float) for j in hs.spec.joints])
        low, high = np.asarray(hs.low, dtype=float), np.asarray(hs.high, dtype=float)
        nums = [o, origins, mic, low, high] + list(cush.values())
        if not all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in nums) or not all(
                math.isfinite(v) for v in (tilt, mic_gap, boom)):
            return None
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return None
    return Reading(o, tilt, cush, ear_gap, ear_out, mic, mic_gap, boom, origins, low, high, hs)


def seated(r: Reading) -> bool:
    return float(np.linalg.norm(r.origin - SEAT)) <= L.SEAT_TOL and r.tilt < L.TILT_MAX


def on_ears(r: Reading) -> bool:
    return all(r.ear_gap[c] <= L.EAR_NEAR and -L.EAR_IN <= r.ear_out[c] <= L.EAR_OFF for c in r.ear_gap)


def mic_down(r: Reading) -> bool:
    return r.boom <= L.BOOM_DOWN + L.BOOM_DOWN_TOL and r.mic_gap <= L.MIC_NEAR


def off_stand(r: Reading) -> bool:
    lift = float(r.origin[2] - STAND_REST[2])
    d = np.maximum(np.abs(r.origin[:2] - STAND_REST[:2]) - np.array(L.CRADLE_HALF), 0.0)
    return lift >= L.OFF_LIFT or float(np.linalg.norm(d)) >= L.OFF_SIDE


def left_table(r: Reading) -> bool:
    """The latch: any part of the headset past the table's edge in plan, or below the table top (fell off). Read
    from the world-aligned box around every headset shape as it is now (exact for the lowest point and the edges)."""
    return bool(r.low[0] < L.TABLE_X[0] or r.high[0] > L.TABLE_X[1] or r.low[1] < L.TABLE_Y[0] or
                r.high[1] > L.TABLE_Y[1] or r.low[2] <= L.TOP - L.BELOW_TOP)


def tier(off: bool, on: bool, mic: bool, window_ok: bool) -> float:
    if not off:
        return 0.0
    if not on:
        return 0.25
    if not mic:
        return 0.5
    if not window_ok:
        return 0.75
    return 1.0


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
        self.resets = getattr(self, "resets", -1) + 1
        self.latch = False
        self.prev: Reading | None = None
        self.window: deque = deque(maxlen=L.WINDOW)
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.window.append(False)
            self.prev = None
            self.last = Result(min(self.last.score, 0.75), False, {"bad_state": True, "step": self.steps})
            return self.last
        if left_table(r):
            self.latch = True
        if self.prev is not None:
            speed = float(np.max(np.linalg.norm(r.origins - self.prev.origins, axis=1))) * L.CONTROL_HZ
        else:
            speed = math.inf
        resting = speed < L.STILL_SPEED
        gap = rigstate.hands_gap(arms, r.state.distance)
        hands_off = gap is not None and gap >= L.HANDS_OFF - 1e-9
        off, st, ears, mic = off_stand(r), seated(r), on_ears(r), mic_down(r)
        on = st and ears
        self.window.append(bool(off and on and mic and resting and hands_off))
        window_ok = len(self.window) == L.WINDOW and all(self.window)
        score = 0.0 if self.latch else tier(off, on, mic, window_ok)
        details = {"step": self.steps, "origin_mm": [round((float(v) - b) * 1000, 1) for v, b in
                                                      zip(r.origin, (L.HX, L.HY, L.TOP))],
                   "tilt": round(r.tilt, 3), "ear_gap_mm": {k: round(v * 1000, 1) for k, v in r.ear_gap.items()},
                   "ear_out_mm": {k: round(v * 1000, 1) for k, v in r.ear_out.items()},
                   "mic_gap_mm": round(r.mic_gap * 1000, 1), "boom": round(r.boom, 3),
                   "cups_mm": {k: round(float(r.state.joints[k]) * 1000, 1) for k in ("cup_p", "cup_n")},
                   "off_stand": off, "seated": st, "on_ears": ears, "mic_down": mic,
                   "speed_mm_s": None if not math.isfinite(speed) else round(speed * 1000, 1), "resting": resting,
                   "hands_gap_mm": None if gap is None or not math.isfinite(gap) else round(gap * 1000, 1),
                   "hands_off": hands_off, "lowest_mm": round((float(r.low[2]) - L.TOP) * 1000, 1),
                   "latch": self.latch}
        self.prev = r
        self.last = Result(score, score >= 1.0, details)
        return self.last
