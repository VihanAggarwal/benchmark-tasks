"""An offline stand-in for the die's platen and the safety block, for tools/offline/check_task.py only.

The offline checker simulates nothing: it hands the hooks the die with its joint at 0 for the whole trial, and it
sees the block only as the box round it (98 x 90 mm, wider than an open gripper), so it never moves it. Scored as
they are, the demo's reach for the part would look like a reach into a shut die. ``task.py`` therefore swaps in this
stand-in's states, and only when mechkit marks the states as offline (``source == "offline"``); in the simulator
(``source == "sim"``) the scorer reads the real joint and the real block and this module does nothing.

What it models, from the arms' tool points and openings (rigstate) alone:
- the platen: a gripper closing (opening under 0.6) with its tool point within 6 mm of the handle post's box takes
  hold of it; while held, the post sits at the tool point's x (the closing pads centre it), so j = 0.273 - x,
  limited to 0..0.075 and never below the block's bar if the bar lies on the bed between the faces. Let go, the
  platen closes (the spring never opens it) toward the bar, or shut, like a first-order lag with the spring's time
  constant (damping / stiffness = 0.075 s), so letting go of an open die shows as a fast close;
- the block: a gripper closing with its tool point within 6 mm of the handle arm's box takes hold of it and carries
  it rigidly, without turning it (nothing stops it: there are no collisions here); let go, it drops straight down
  onto the highest thing under it (the platens as they are now, the bed's top, the stand, the tray's floor or the
  table), and a drop of more than 2 mm shows as a fall at the free-fall speed. A block let go on the shut die rests
  on the moving platen, not on the bed.

The part is left to the checker's own carry-and-drop proxy.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk

ARMS = ("left", "right")
_SUPPORTS = (  # (x0, x1, y0, y1, top above the table)
    (L.BED_X0, L.BED_X1, L.BED_Y0, L.BED_Y1, L.BED_TOP_Z),
    (L.STAND_X - L.STAND_SIZE[0] / 2, L.STAND_X + L.STAND_SIZE[0] / 2,
     L.STAND_Y - L.STAND_SIZE[1] / 2, L.STAND_Y + L.STAND_SIZE[1] / 2, L.STAND_TOP_Z),
    (L.TRAY_X - L.TRAY_INNER[0] / 2, L.TRAY_X + L.TRAY_INNER[0] / 2,
     L.TRAY_Y - L.TRAY_INNER[1] / 2, L.TRAY_Y + L.TRAY_INNER[1] / 2, L.TRAY_FLOOR_Z),
)


def _box_dist(p, lo, hi) -> float:
    p = np.asarray(p, dtype=float)
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - p, p - np.asarray(hi)))
    return float(np.linalg.norm(d))


def _post_box(j: float):
    (x0, x1), (y0, y1), (z0, z1) = L.POST
    return np.array([x0 - j, y0, L.TOP + z0]), np.array([x1 - j, y1, L.TOP + z1])


def _shape_box(origin, center, size):
    c = np.asarray(origin, dtype=float) + np.asarray(center, dtype=float)
    h = np.asarray(size, dtype=float) / 2
    return c - h, c + h


class StandIn:
    """Kinematic platen and block, fed once per control step (``step``); ``states`` gives the latest without
    advancing. Keyed by the die and block specs from task.py."""

    def __init__(self, die_spec, block_spec, control_hz: float = L.CONTROL_HZ) -> None:
        self.die_spec, self.block_spec = die_spec, block_spec
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.j, self.jv = 0.0, 0.0
        self.post_holder: str | None = None
        self.origin = np.array(L.BLOCK_ORIGIN, dtype=float)
        self.vel = np.zeros(3)
        self.block_holder: str | None = None
        self.offset = np.zeros(3)
        self.steps = 0
        self.events: list[tuple[int, str]] = []

    # -- geometry ------------------------------------------------------------------------------------------------
    def _bar_box(self, origin=None):
        return _shape_box(self.origin if origin is None else origin, L.BAR_CENTER, L.BAR_SIZE)

    def _arm_box(self):
        return _shape_box(self.origin, L.ARM_CENTER, L.ARM_SIZE)

    def _stop(self) -> float:
        """The opening the platen closes to: shut, or onto the bar if the bar lies on the bed in the open gap."""
        lo, hi = self._bar_box()
        between = (hi[1] > L.FACE_PLATE[1][0] and lo[1] < L.FACE_PLATE[1][1]
                   and abs(lo[2] - (L.TOP + L.BED_TOP_Z)) <= 0.004
                   and lo[0] >= L.MOVING_FACE_X0 - self.j - 0.002 and hi[0] <= L.FIXED_FACE_X + 0.002)
        if not between:
            return 0.0
        return float(np.clip(L.MOVING_FACE_X0 - lo[0], 0.0, L.PLATEN_TRAVEL))

    def _support(self) -> float:
        """World height of the highest thing under the block's outline (the table if nothing else)."""
        boxes = [self._bar_box(), self._arm_box()]
        tops = list(_SUPPORTS)
        for part in (L.FACE_PLATE, L.CHEEK_LEFT, L.CHEEK_RIGHT, L.BRACKET, L.POST):  # the moving platen, open j
            (x0, x1), (y0, y1), (_, z1) = part
            tops.append((x0 - self.j, x1 - self.j, y0, y1, z1))
        (x0, x1), (y0, y1), (_, z1) = L.FIXED_PLATEN
        tops.append((x0, x1, y0, y1, z1))
        top = 0.0
        for x0, x1, y0, y1, h in tops:
            if any(lo[0] < x1 and hi[0] > x0 and lo[1] < y1 and hi[1] > y0 for lo, hi in boxes):
                top = max(top, h)
        return L.TOP + top

    # -- one control step ----------------------------------------------------------------------------------------
    def step(self, arms):
        self.steps += 1
        j0, origin0 = self.j, self.origin.copy()
        if isinstance(arms, dict):
            self._post(arms)
            self._block(arms)
        if self.post_holder is None:
            stop = self._stop()
            if self.j > stop:  # the spring only closes it
                self.j = stop + (self.j - stop) * math.exp(-self.dt / L.STANDIN_TAU)
        self.jv = (self.j - j0) / self.dt
        if self.block_holder is not None:
            self.vel = (self.origin - origin0) / self.dt
        return self.states()

    def _closing(self, arm) -> bool:
        return float(arm.opening) < L.STANDIN_GRIP

    def _post(self, arms) -> None:
        if self.post_holder is not None and not self._closing(arms[self.post_holder]):
            self.events.append((self.steps, f"{self.post_holder} lets go of the post at j {self.j:.4f}"))
            self.post_holder = None
        if self.post_holder is None:
            for side in ARMS:
                a = arms[side]
                if side != self.block_holder and self._closing(a) and \
                        _box_dist(a.tcp, *_post_box(self.j)) <= L.STANDIN_REACH:
                    self.post_holder = side
                    self.events.append((self.steps, f"{side} takes the post at j {self.j:.4f}"))
                    break
        if self.post_holder is not None:
            x = float(np.asarray(arms[self.post_holder].tcp)[0])
            self.j = float(np.clip(L.POST_X0 - x, self._stop(), L.PLATEN_TRAVEL))

    def _block(self, arms) -> None:
        if self.block_holder is not None and not self._closing(arms[self.block_holder]):
            side, self.block_holder = self.block_holder, None
            rest = self._support()
            drop = float(self.origin[2] - rest)
            self.origin[2] = rest  # (up onto a platen it was pushed through, here where nothing collides)
            self.vel = np.array([0.0, 0.0, -math.sqrt(2 * L.GRAVITY * drop) if drop > 0.002 else 0.0])
            self.events.append((self.steps, f"{side} lets go of the block, drop {drop * 1000:.1f} mm"))
            return
        if self.block_holder is None:
            self.vel = np.zeros(3)
            lo, hi = self._arm_box()
            for side in ARMS:
                a = arms[side]
                if side != self.post_holder and self._closing(a) and \
                        _box_dist(a.tcp, lo, hi + [0.0, 0.0, 0.010]) <= L.STANDIN_REACH:
                    self.block_holder = side
                    self.offset = self.origin - np.asarray(a.tcp, dtype=float)
                    self.events.append((self.steps, f"{side} takes the block"))
                    break
        if self.block_holder is not None:
            self.origin = np.asarray(arms[self.block_holder].tcp, dtype=float) + self.offset

    def states(self):
        """(die, block): mechkit AssemblyStates marked ``source="offline"``."""
        die = mk.state_from_poses(self.die_spec, (*L.DIE_ORIGIN, 0.0, 0.0, 0.0, 1.0), None,
                                  joints={"platen": self.j}, joint_vel={"platen": self.jv}, source="offline")
        block = mk.state_from_poses(self.block_spec, (*self.origin, 0.0, 0.0, 0.0, 1.0),
                                    (*self.vel, 0.0, 0.0, 0.0), source="offline")
        return die, block
