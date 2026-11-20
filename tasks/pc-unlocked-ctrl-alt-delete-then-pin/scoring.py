"""The scorer, as plain functions of the keyboard's and the arms' state. NumPy only.

The keyboard is one mechkit articulation: the case (loose, 550 g) and one sprung cap on a slider per modelled key
(joint value: travel down, metres). A key is DOWN while its travel is 2.0 mm or more. The PC's response is a rule
(the device's own output, as on a real PC): the lock screen until Delete goes down while a Ctrl and an Alt were both
already down on the step before and are still down (Windows' secure attention sequence); then the PIN prompt, which logs number-pad
digits; Enter with the right PIN opens the desktop.

    Z  latched 0: the keyboard was lifted 5 mm or tilted 5 degrees (pulled off the desk; a slide is harmless)
    S  stray: any modelled key (a Windows key included) went down while it was not part of the act (before the prompt: any key but Ctrl,
       Alt, Delete; after it: any key but digits, Enter and the still-held Ctrl and Alt), or Enter with a wrong PIN

    0     Z, or nothing below
    0.25  a Ctrl and an Alt down together for 0.5 s
    0.5   the PIN prompt shown (Ctrl+Alt+Delete in order), or anything higher while S
    0.75  the prompt shown and the PIN typed right (2, 5, 8, 0, nothing else), not yet entered
    1.0   the desktop shown (Enter after the right PIN), every key up, both grippers 30 mm or more from the keyboard
          and the keyboard still over the last 1 s
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

KEYBOARD = "keyboard"
KEY_NAMES = tuple(L.KEYS)
LOCK, PROMPT, DESKTOP = "lock", "prompt", "desktop"


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def turn_deg(r0, r1) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class Reading:
    origin: np.ndarray  # world, table top at z 0
    rot: np.ndarray
    down: frozenset  # keys at or past actuation


def read(states) -> Reading | None:
    if not isinstance(states, dict) or states.get(KEYBOARD) is None:
        return None
    s = states[KEYBOARD]
    try:
        o = np.asarray(s.origin, dtype=float).ravel() - np.array([0.0, 0.0, L.TOP])
        r = rotation(s.quat_xyzw)
        travel = {k: float(s.joints[k]) for k in KEY_NAMES}
    except (AttributeError, TypeError, ValueError, KeyError):
        return None
    if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))) or \
            not all(math.isfinite(v) for v in travel.values()):
        return None
    return Reading(o, r, frozenset(k for k, v in travel.items() if v >= L.KEY_ACT - L.EPS))


class PC:
    """The PC's response to key events (a rule standing for the real PC's own behaviour)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.screen = LOCK
        self.typed = ""
        self.prev: frozenset = frozenset()
        self.win = False
        self.stray: list = []
        self.wrong_enter = False
        self.pair_steps = 0
        self.pair_ok = False
        self.pin_ok = False

    def step(self, down: frozenset, hold_steps: int) -> None:
        pressed = down - self.prev  # keys that went down this step
        prev = self.prev
        ctrl_prev = any(k in prev for k in L.CTRL_KEYS)
        alt_prev = any(k in prev for k in L.ALT_KEYS)
        ctrl_now = any(k in down for k in L.CTRL_KEYS)
        alt_now = any(k in down for k in L.ALT_KEYS)
        self.pair_steps = self.pair_steps + 1 if (ctrl_now and alt_now) else 0
        if self.pair_steps >= hold_steps:
            self.pair_ok = True
        for k in sorted(pressed):
            if k in L.WIN_KEYS:
                self.win = True  # a stray keystroke like any other (the lock screen ignores it)
            if self.screen == LOCK:
                if k in L.CTRL_KEYS or k in L.ALT_KEYS:
                    continue
                if k == "delete":  # without the held pair the lock screen ignores it
                    if ctrl_prev and alt_prev and ctrl_now and alt_now:
                        self.screen = PROMPT
                    continue
                self.stray.append(k)
            elif self.screen == PROMPT:
                if k in L.DIGIT_KEYS:
                    self.typed += L.DIGIT_KEYS[k]
                    self.pin_ok = self.typed == L.PIN and not self.wrong_enter
                    continue
                if k == "npenter":
                    if self.typed == L.PIN and not self.wrong_enter:
                        self.screen = DESKTOP
                    else:
                        self.wrong_enter = True
                        self.typed = ""
                        self.pin_ok = False
                    continue
                if k in L.CTRL_KEYS or k in L.ALT_KEYS:
                    self.stray.append(k)
                    continue
                self.stray.append(k)
            else:  # desktop: any key now is a stray input to the desktop
                self.stray.append(k)
        self.prev = down


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def tier(pc: PC, latched: bool, settled: bool, all_up: bool) -> float:
    if latched:
        return 0.0
    if pc.screen == LOCK:
        return 0.25 if pc.pair_ok else 0.0
    capped = bool(pc.stray) or pc.wrong_enter
    if capped:
        return 0.5
    if pc.screen == DESKTOP and settled and all_up:
        return 1.0
    if pc.screen == DESKTOP or pc.pin_ok:
        return 0.75
    return 0.5


class Tracker:
    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.hold_steps = max(1, int(round(L.HOLD_PAIR_S * control_hz)))
        self.resets = 0
        self.pc = PC()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.latch = None
        self.origin0 = None
        self.rot0 = None
        self.pc.reset()
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.released.clear()
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if self.origin0 is None:
            self.origin0, self.rot0 = r.origin.copy(), r.rot.copy()
        if self.prev is not None and float(np.linalg.norm(r.origin - self.prev.origin)) >= L.RESET_JUMP and \
                float(np.linalg.norm(r.origin[:2] - np.array([L.KB_X, L.KB_Y]))) < 0.003:
            self.reset()
            self.steps, self.resets = 1, self.resets + 1
            self.origin0, self.rot0 = r.origin.copy(), r.rot.copy()
        self.prev = r
        self.hist.append(r)
        self.pc.step(r.down, self.hold_steps)
        move = float(np.linalg.norm(r.origin[:2] - self.origin0[:2]))
        lift = float(r.origin[2] - self.origin0[2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(r.rot[2, 2])))))
        if self.latch is None and (lift > L.LIFT_LATCH or tilt > L.TILT_LATCH_DEG):
            self.latch = self.steps
        try:
            dist = states[KEYBOARD].distance
            ok = bool(rigstate.released(arms, dist, L.LET_GO))
            gap = rigstate.hands_gap(arms, dist)
        except Exception:  # noqa: BLE001 - no distance or arms: not released
            ok, gap = False, None
        self.released.append(ok)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m[0] <= L.STILL_MOVE + L.EPS and m[1] <= L.STILL_TURN_DEG + L.EPS)
        all_up = len(r.down) == 0
        level = tier(self.pc, self.latch is not None, released and still, all_up)
        self.last = Result(level, level == 1.0, {
            "valid_state": True, "score": level, "screen": self.pc.screen, "typed": self.pc.typed,
            "pair_ok": self.pc.pair_ok, "pin_ok": self.pc.pin_ok, "win": self.pc.win, "stray": list(self.pc.stray),
            "wrong_enter": self.pc.wrong_enter, "down": sorted(r.down), "latch": self.latch,
            "move_mm": round(move * 1000, 2), "lift_mm": round(lift * 1000, 2), "tilt_deg": round(tilt, 3),
            "released_1s": released, "still_1s": still,
            "hands_gap_mm": None if gap is None or not math.isfinite(gap) else round(gap * 1000, 1)})
        return self.last

    def motion(self):
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        move = max(float(np.linalg.norm(x.origin - h[0].origin)) for x in h[1:])
        turn = max(turn_deg(h[0].rot, x.rot) for x in h[1:])
        return move, turn
