"""The machine's program (a rule) and the scorer, as plain functions of the objects' states. NumPy only.

``Program`` is the claw machine's controller, fed the joystick's two angles and the button's travel every control step:

    PLAY     while a direction's switch is closed (the stick tilted past 8 deg that way), the claw travels that way at
             50 mm/s (8-way: both axes at once on a diagonal); the 30 s timer starts at the stick's first move
    DROP     the button pressed past 3 mm (or the timer run out): the claw drops at 80 mm/s to just above a capsule's
             top, closes (0.6 s), and holds a capsule if its centre is within 12 mm of the claw's axis; it rises,
             travels back over the chute at 50 mm/s and opens. One credit: one drop.

``Tracker.update(states, arms, program) -> Result(score, success, details)``:

    WON      a capsule has fallen through the chute (inside the base, below the pit's floor)
    NEAR     the claw closed with its axis within 25 mm of a capsule's centre
    TRIED    the claw was moved at least 50 mm from its start before it dropped
    HAND     a gripper's TCP or pad tips inside the cabinet, above the pit's floor, at any time

    1.0   WON                         (success, held 1 s)
    0.5   NEAR, not won
    0.25  TRIED
    0     anything else, and every trial with HAND
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

MACHINE = "claw_machine"
STICK_FB, STICK_LR, BUTTON = "stick_fb", "stick_lr", "button"
CARRIAGE_X, CARRIAGE_Y, CLAW = "carriage_x", "carriage_y", "claw"
CAPSULES = [f"capsule_{i}" for i in range(len(L.CAPSULES))]


class Program:
    """The claw machine's controller. ``pos`` (x, y) and ``z`` (the hub's bottom) are in the machine's frame."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = L.HOME.astype(float).copy()
        self.z = L.HUB_Z
        self.state = "play"
        self.t = 0.0  # time in the current state
        self.timer: float | None = None  # game time since the stick's first move
        self.dropped = False
        self.moved = 0.0  # the claw's distance from its start when it dropped
        self.closed_at: np.ndarray | None = None
        self.near = False
        self.grabbed: int | None = None
        self.events: list[str] = []

    @staticmethod
    def direction(angle: float) -> int:
        return int(math.copysign(1, angle)) if abs(angle) > L.STICK_SWITCH else 0

    def step(self, fb: float, lr: float, button: float, capsules_xy, dt: float = L.DT) -> list[str]:
        """Advance one control step; returns this step's events ("grab <i>", "release")."""
        self.events = []
        self.t += dt
        if self.state == "play":
            dx, dy = self.direction(fb), -self.direction(lr)  # stick top forward: +x; tilted right (+lr): -y
            if self.timer is None and (dx or dy):
                self.timer = 0.0
            if self.timer is not None:
                self.timer += dt
            self.pos[0] = float(np.clip(self.pos[0] + dx * L.SPEED * dt, *L.CLAW_X))
            self.pos[1] = float(np.clip(self.pos[1] + dy * L.SPEED * dt, *L.CLAW_Y))
            if button >= L.BUTTON_ON or (self.timer is not None and self.timer >= L.GAME_S):
                self.dropped = True
                self.moved = float(np.linalg.norm(self.pos - L.HOME))
                self._go("drop")
        elif self.state == "drop":
            self.z = max(L.DROP_TO, self.z - L.DROP_SPEED * dt)
            if self.z <= L.DROP_TO + 1e-9:
                self._go("close")
        elif self.state == "close":
            if self.t >= L.CLOSE_S:
                self.closed_at = self.pos.copy()
                d = [float(np.linalg.norm(np.asarray(c, dtype=float) - self.pos)) for c in capsules_xy]
                if d:
                    k = int(np.argmin(d))
                    self.near = d[k] <= L.NEAR_R
                    if d[k] <= L.GRAB_R:
                        self.grabbed = k
                        self.events.append(f"grab {k}")
                self._go("rise")
        elif self.state == "rise":
            self.z = min(L.HUB_Z, self.z + L.DROP_SPEED * dt)
            if self.z >= L.HUB_Z - 1e-9:
                self._go("return")
        elif self.state == "return":
            d = L.HOME - self.pos
            n = float(np.linalg.norm(d))
            step = L.SPEED * dt
            self.pos = L.HOME.astype(float).copy() if n <= step else self.pos + d / n * step
            if n <= step:
                self.events.append("release")
                self._go("open")
        elif self.state == "open" and self.t >= L.OPEN_WAIT_S:
            self._go("done")
        return self.events

    def _go(self, state: str) -> None:
        self.state, self.t = state, 0.0

    def targets(self) -> dict:
        """The gantry's joint targets (m), each 0 at the claw's start."""
        return {CARRIAGE_X: float(self.pos[0] - L.HOME[0]), CARRIAGE_Y: float(self.pos[1] - L.HOME[1]),
                CLAW: float(L.HUB_Z - self.z)}


def controls(states) -> tuple[float, float, float] | None:
    try:
        j = states[MACHINE].joints
        out = (float(j[STICK_FB]), float(j[STICK_LR]), float(j[BUTTON]))
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def capsule_positions(states) -> list[np.ndarray] | None:
    """Each capsule's centre in the machine's frame (x, y, z)."""
    out = []
    try:
        for name in CAPSULES:
            p = np.asarray(states[name].pos, dtype=float).ravel()
            if p.size != 3 or not np.all(np.isfinite(p)):
                return None
            out.append(p - np.array([L.ORIGIN[0], L.ORIGIN[1], L.TOP]))
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        return None
    return out


def won(capsules) -> bool:
    return any(c[2] < L.BASE_H - 0.03 and 0.0 < c[0] < L.CAB_D and abs(c[1]) < L.CAB_W / 2 for c in capsules)


def tier(won_: bool, near: bool, dropped: bool, moved: float, hand: bool) -> float:
    if hand:
        return 0.0
    if won_:
        return 1.0
    if near:
        return 0.5
    if dropped and moved >= L.MOVED:
        return 0.25
    return 0.0


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """Only an episode reset (task.py: rigstate.on_reset and mechkit's episode counter) clears the history."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hand = False
        self.won = False
        self.last = Result(0.0, False, {})

    def update(self, states: dict, arms, program: Program) -> Result:
        self.steps += 1
        caps = capsule_positions(states) if isinstance(states, dict) else None
        if caps is None:
            self.bad_steps += 1
            self.last = Result(self.last.score, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        if rigstate.readable(arms):
            for s in ("left", "right"):
                if L.in_cabinet(arms[s].tcp) or L.in_cabinet(arms[s].tips):
                    self.hand = True
        self.won = self.won or won(caps)
        level = tier(self.won, program.near, program.dropped, program.moved, self.hand)
        details = {
            "valid_state": True, "score": level, "state": program.state, "claw_mm": [round(v * 1000, 1) for v in program.pos],
            "claw_z_mm": round(program.z * 1000, 1), "timer": None if program.timer is None else round(program.timer, 2),
            "dropped": program.dropped, "moved_mm": round(program.moved * 1000, 1), "near": program.near,
            "grabbed": program.grabbed, "won": self.won, "hand": self.hand, "arms_readable": rigstate.readable(arms),
        }
        self.last = Result(level, bool(level == 1.0), details)
        return self.last
