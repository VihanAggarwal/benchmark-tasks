"""Offline checker only: where the work and the pins would be, from the grippers' motion. The simulator never calls
anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics. It sees the work as mechkit's assembly
at rest, its feed joint always at 0 (``Assembly.offline_state``), and it carries a loose object only when the pads
close across it where it started, so it never moves the plate and never carries a pin that rode the plate to the
needle. When task.py's hooks get a work state whose ``source`` is "offline" (mechkit's own marker for the checker's
states; the simulator's say "sim"), they feed the scorer the feed and the pins from this small kinematic model
instead, built from the tool points, finger-tip points and openings the checker feeds to rigstate:

- The feed follows a gripper that grips the handle: its finger-tip point inside the handle's box grown by 5 mm and
  the gripper closed onto it (opening 0.36 or less: fingers within 2 mm of its 30 mm faces). While it grips, the
  feed changes by the tool point's motion along -y, clamped to the rail's travel (0 .. 180 mm) and stopped where a
  pin riding in the work would touch the needle (pin and needle radii, 11 mm); at a stop the grip slips (it is
  re-anchored), so moving back does not drag the work further back than the gripper goes. Nothing else moves the
  work: pushing the plate, or dragging it by a pin, is not modelled and moves nothing.
- A pin rides in the work (its position at the start of the rail, minus the feed along y) until a gripper closes on
  it: its opening falls through the pin's 18 mm in one step with its tool point within 8 mm of the pin's axis
  horizontally and its pads (the 22 mm from the finger tips up) overlapping the pin's height by 10 mm or more. A
  held pin moves with that tool point. When the gripper opens past 20 mm the pin drops straight down and stands on
  whatever is under its centre: the dish's floor, a pocket's floor or the plate's top (then it rides with the work
  again), the bed, the needle head's bar, the table, or the floor (off the table). It is reported moving for its
  fall time plus 0.15 s.

Upright pins only, no lag, no slip in the hand, no sag, no rattle in the pockets, no pushing or knocking: the GPU
runs decide those (the README says what to read in the trace). An arm that does not grip the handle or close on a
pin moves nothing here. Reset with every new episode (task.py calls ``reset()`` from rigstate's and mechkit's
reset signals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import layout as L

ARMS = ("left", "right")
GRIP_OPENING = 0.36  # a gripper this closed (or more) on the handle grips it (fingers within 2 mm of its faces)
GRIP_MARGIN = 0.005  # its finger-tip point inside the handle's box grown by this
PIN_W = 2 * L.PIN_R
CLOSE_PIN = PIN_W / L.FULL_GAP  # 0.192: the pads reach the pin's width
RELEASE_PIN = (PIN_W + 0.002) / L.FULL_GAP  # 0.213: opened 2 mm past it, the pin is let go
CENTRE_TOL = 0.008  # the tool point this close to a pin's axis (horizontally) to close on it
PAD_LEN = 0.022  # the pads' flat faces run 22 mm up from the finger tips
PAD_OVERLAP = 0.010  # ... and must overlap the pin's height by this much
STOP = L.PIN_R + L.NEEDLE_R  # a pin's centre this far from the needle's axis touches it
FALL_EXTRA = 0.15  # s a dropped pin counts as moving after its fall
_HALF = np.array([L.PIN_R, L.PIN_R, L.PIN_H / 2])


@dataclass
class PinState:
    """What the scorer reads for a pin (the fields of yam_arena's ObjectState)."""

    pos: np.ndarray
    quat_xyzw: np.ndarray
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    low: np.ndarray
    high: np.ndarray


def _arms_ok(arms) -> bool:
    try:
        return all(np.all(np.isfinite(np.asarray(arms[a].tcp, float))) and np.all(np.isfinite(np.asarray(
            arms[a].tips, float))) and math.isfinite(float(arms[a].opening)) for a in ARMS)
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def _handle_box(feed: float, margin: float = 0.0):
    c = np.array([L.X_SEAM, L.HANDLE_Y - feed, L.TOP + L.HANDLE_Z])
    h = np.array(L.HANDLE, dtype=float) / 2 + margin
    return c - h, c + h


def _inside(p, lo, hi) -> bool:
    p = np.asarray(p, dtype=float)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _support(x: float, y: float, bottom: float, feed: float) -> tuple[float, str]:
    """Height above the table of what a pin dropped at (x, y) from ``bottom`` comes to stand on, and its name."""
    if abs(x - L.DISH_X) <= L.DISH_HALF[0] and abs(y - L.DISH_Y) <= L.DISH_HALF[1]:
        return L.DISH_FLOOR_T, "dish"
    if abs(x - L.DISH_X) <= L.DISH_HALF[0] + L.DISH_WALL and abs(y - L.DISH_Y) <= L.DISH_HALF[1] + L.DISH_WALL:
        return L.DISH_RIM, "dish wall"
    if (abs(x - L.X_SEAM) <= L.HEAD_HALF and abs(y - L.HEAD_Y) <= L.POST[1] / 2
            and bottom >= L.BAR_Z1 - 1e-9):
        return L.BAR_Z1, "needle bar"
    if abs(x - L.X_SEAM) <= L.PLATE_W / 2 and L.PLATE_Y0 - feed <= y <= L.PLATE_Y1 - feed:
        slack = L.POCKET / 2 - L.PIN_R
        for d0 in L.PIN_D0.values():
            if abs(x - L.X_SEAM) <= slack and abs(y + feed - (L.Y_NEEDLE + d0)) <= slack:
                return L.POCKET_FLOOR, "plate"
        return L.PLATE_TOP, "plate"
    if abs(x - L.X_SEAM) <= L.BED_W / 2 and L.BED_Y0 <= y <= L.BED_Y1:
        return L.BED_TOP, "bed"
    if L.TABLE_X[0] <= x <= L.TABLE_X[1] and L.TABLE_Y[0] <= y <= L.TABLE_Y[1]:
        return 0.0, "table"
    return -L.TOP, "floor"


class _Pin:
    def __init__(self, start) -> None:
        self.mode = "work"  # "work" (riding the plate), "held" (by self.arm), "rest" (standing somewhere still)
        self.plate = np.array(start, dtype=float)  # in "work": its world centre at feed 0
        self.pos = np.array(start, dtype=float)
        self.prev = self.pos.copy()
        self.arm: str | None = None
        self.offset = np.zeros(3)
        self.falling_until = -1  # step until which it counts as moving (a drop)


class OfflineWork:
    """The work's feed and the pins, stepped once per control step from rigstate's arms."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.feed = 0.0
        self.steps = 0
        self.driver: str | None = None  # the arm whose grip is moving the work
        self.anchor = (0.0, 0.0)  # (feed, tool point y) when that grip began (or last slipped)
        self.prev_opening = {a: 1.0 for a in ARMS}
        self.pins = {n: _Pin(L.PIN_START[n]) for n in L.PINS}
        self.events: list[dict] = []

    # -- the work --------------------------------------------------------------------------------------------------
    def needle_stop(self) -> float:
        """The largest feed before a pin riding in the work touches the needle (FEED_MAX if none would)."""
        stop = L.FEED_MAX
        for p in self.pins.values():
            if p.mode != "work" or abs(p.plate[0] - L.X_SEAM) >= STOP:
                continue
            if p.plate[1] - self.feed >= L.Y_NEEDLE + STOP - 1e-9:  # still before the needle
                stop = min(stop, p.plate[1] - L.Y_NEEDLE - STOP)
        return stop

    def _drive(self, arms) -> None:
        lo, hi = _handle_box(self.feed, GRIP_MARGIN)
        holding = {p.arm for p in self.pins.values() if p.mode == "held"}
        gripping = [a for a in ARMS if a not in holding and float(arms[a].opening) <= GRIP_OPENING
                    and _inside(arms[a].tips, lo, hi)]
        if self.driver not in gripping:
            self.driver = gripping[0] if gripping else None
            if self.driver is not None:
                self.anchor = (self.feed, float(arms[self.driver].tcp[1]))
                self.events.append({"step": self.steps, "event": "grip handle", "arm": self.driver})
        if self.driver is None:
            return
        y = float(arms[self.driver].tcp[1])
        target = self.anchor[0] + (self.anchor[1] - y)
        feed = min(max(target, 0.0), self.needle_stop())
        if feed != target:  # at the end of the rail or against the needle: the grip slips
            self.anchor = (feed, y)
        self.feed = feed

    # -- the pins --------------------------------------------------------------------------------------------------
    def _drop(self, name: str, p: _Pin) -> None:
        bottom = float(p.pos[2]) - L.TOP - L.PIN_H / 2
        z, what = _support(float(p.pos[0]), float(p.pos[1]), bottom, self.feed)
        p.pos = np.array([p.pos[0], p.pos[1], L.TOP + z + L.PIN_H / 2])
        fall = max(bottom - z, 0.0)
        p.falling_until = self.steps + int(math.ceil((math.sqrt(2 * fall / L.GRAVITY) + FALL_EXTRA) / self.dt))
        p.arm = None
        if what == "plate":
            p.mode, p.plate = "work", p.pos + np.array([0.0, self.feed, 0.0])
        else:
            p.mode = "rest"
        self.events.append({"step": self.steps, "event": "drop", "pin": name, "onto": what,
                            "fall_mm": round(fall * 1000, 1)})

    def _grasp(self, arm: str, a) -> None:
        tcp, tips = np.asarray(a.tcp, dtype=float), np.asarray(a.tips, dtype=float)
        pad_lo, pad_hi = min(tips[2], tips[2] + PAD_LEN), max(tips[2], tips[2] + PAD_LEN)
        best = None
        for name, p in self.pins.items():
            if p.mode == "held":
                continue
            off = float(np.hypot(p.pos[0] - tcp[0], p.pos[1] - tcp[1]))
            overlap = min(pad_hi, p.pos[2] + L.PIN_H / 2) - max(pad_lo, p.pos[2] - L.PIN_H / 2)
            if off <= CENTRE_TOL and overlap >= PAD_OVERLAP and (best is None or off < best[0]):
                best = (off, name, p)
        if best is not None:
            _, name, p = best
            p.mode, p.arm, p.offset = "held", arm, p.pos - tcp
            self.events.append({"step": self.steps, "event": "grasp", "pin": name, "arm": arm,
                                "d_mm": round((float(p.pos[1]) - L.Y_NEEDLE) * 1000, 1)})

    def step(self, arms) -> tuple[float, dict]:
        """Advance one control step with the arms rigstate holds for it; returns (feed, pin states)."""
        self.steps += 1
        for p in self.pins.values():
            p.prev = p.pos.copy()
        if not _arms_ok(arms):  # nothing to go on: nothing moves
            return self.state()
        self._drive(arms)
        for p in self.pins.values():
            if p.mode == "work":
                p.pos = p.plate - np.array([0.0, self.feed, 0.0])
        for name, p in self.pins.items():
            if p.mode == "held":
                a = arms[p.arm]
                if float(a.opening) > RELEASE_PIN:
                    self._drop(name, p)
                else:
                    p.pos = np.asarray(a.tcp, dtype=float) + p.offset
        holding = {p.arm for p in self.pins.values() if p.mode == "held"}
        for arm in ARMS:
            o = float(arms[arm].opening)
            if arm not in holding and self.prev_opening[arm] >= CLOSE_PIN > o:
                self._grasp(arm, arms[arm])
            self.prev_opening[arm] = o
        return self.state()

    def state(self) -> tuple[float, dict]:
        """The current feed and pin states (no step)."""
        out = {}
        for name, p in self.pins.items():
            if self.steps <= p.falling_until:
                vel = np.array([0.0, 0.0, -0.2])
            else:
                vel = (p.pos - p.prev) / self.dt
            out[name] = PinState(p.pos.copy(), np.array([0.0, 0.0, 0.0, 1.0]), vel, np.zeros(3), p.pos - _HALF,
                                 p.pos + _HALF)
        return self.feed, out
