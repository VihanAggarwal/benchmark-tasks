"""Kinematics of the pliers for use ONLY where there is no simulator: tools/offline/check_task.py (which by itself can
carry and drop a gripped box, nothing else) and the tests. The simulator never takes a result from here: there the
slide and the hinge are PhysX joints, the bench contacts are PhysX contacts, and the groove rule (task.py) writes
the slider's drive target every control step.

What this file assumes is what the design claims, so the offline prediction is the design's prediction, not a
measurement. It moves the pliers from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed):

- a gripper holds a handle once its pads have straddled it open (pad tips no more than 9 mm above the bench, the
  gripper's axis within 4 mm of the handle's centre line, 10 mm or more inside either end of the handle) and then
  close below the handle's 13 mm width; it lets go once open wider than 14 mm;
- a held slotted half moves with its gripper (translation only);
- the rivet half, held, follows its gripper about the rivet: its spread is the gripper's bearing from the rivet;
  while the spread is 72 deg or more and the slotted half is held, the rivet also slides along the slot (from groove 1
  at the slot's end nearest the jaws back toward the handles, layout.SLIDE_DIR), but only once the gripper is more than
  ARM_GIVE (17 mm, more than the GPU showed: there, 13 Nov, the open joint slid with the command about 12 mm ahead
  of the rivet) ahead of or behind it, and only between 0 and 24 mm;
- with the spread under 72 deg the rivet stays in its groove; when the spread drops below 72 deg it drops into the
  nearest groove;
- with the slotted half not held, the rivet never slides (a simplification: on the GPU, 13 Nov, the right arm alone
  dragged the loose pliers 16 mm while the joint slid 19 mm, and it locked in groove 4): a gripper that pulls or
  pushes the rivet half radially more than TOW_GIVE (10 mm of the arm's give) past where it is drags the whole
  pliers instead;
- nothing is lifted, nothing tips, and the pliers stay flat on the bench.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

HOLD_MAX_TIP_H = 0.009  # pad tips at most this high: 3 mm or more of pad beside the 12 mm handle
HOLD_AXIS_TOL = 0.004  # m: the gripper's axis this close to the handle's centre line
HOLD_END_MARGIN = 0.010  # m: inside either end of the handle (half the offline checker's 20 mm pad proxy)
HANDLE_W = L.HANDLE_SECTION[0] * L.MM  # 0.013
RELEASE_GAP = HANDLE_W + 0.001
ARM_GIVE = 0.017  # m: more than the GPU showed (the open joint slid with the command about 12 mm ahead of the rivet)
TOW_GIVE = 0.010  # m


def _wrap(a: float) -> float:
    return math.remainder(a, 2 * math.pi)


class OfflinePliers:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.origin = np.array(L.ORIGIN[:2], dtype=float)  # the slot frame's origin (world xy); its yaw is fixed
        self.q = 0.0  # m
        self.sigma = L.SIGMA0_DEG  # deg
        self.hold = {"slotted": None, "rivet": None}  # which arm holds which half
        self.grip_local: dict[str, np.ndarray] = {}  # the held point, in the half's own frame (m)
        self.around = {("left", "slotted"): False, ("left", "rivet"): False,
                       ("right", "slotted"): False, ("right", "rivet"): False}
        self.prev_tips: dict[str, np.ndarray] = {}
        self.towed = 0.0
        self.max_q = 0.0

    # -- frames ---------------------------------------------------------------------------------------------------
    def _slot_point(self, world_xy) -> np.ndarray:
        return L.to_slot_xy(world_xy, (*self.origin, L.TOP))

    def _half_turn(self, half: str) -> float:
        return L.SLOTTED_TURN_DEG + (self.sigma if half == "rivet" else 0.0)

    def _half_origin(self, half: str) -> np.ndarray:
        return L.rivet_xy(self.q) if half == "rivet" else np.zeros(2)

    def _in_half(self, half: str, world_xy) -> np.ndarray:
        """A world point in the half's own frame (x along its handle line), m."""
        p = self._slot_point(world_xy) - self._half_origin(half)
        return L.rot2(-self._half_turn(half)) @ p

    def _on_handle(self, half: str, tips) -> bool:
        local = self._in_half(half, tips[:2])
        u0, u1 = (L.HANDLE_U if half == "slotted" else L.RIVET_HANDLE_R)
        off = -L.SLOTTED_OFFSET if half == "slotted" else L.RIVET_OFFSET
        h = float(tips[2] - L.TOP)
        return (u0 * L.MM + HOLD_END_MARGIN <= local[0] <= u1 * L.MM - HOLD_END_MARGIN
                and abs(local[1] - off) <= HOLD_AXIS_TOL and -0.002 <= h <= HOLD_MAX_TIP_H)

    # -- one control step -----------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return  # nothing can be judged from unreadable arms: the pliers stay as they are
        grips = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            grips[side] = (np.asarray(tips, dtype=float), gap)

        # holds start and end
        for side, (tips, gap) in grips.items():
            for half in ("slotted", "rivet"):
                if self.hold[half] == side:
                    if gap > RELEASE_GAP + 1e-9:
                        self.hold[half] = None
                    continue
                on = self._on_handle(half, tips)
                if self.hold[half] is None and on and self.around[(side, half)] and gap < HANDLE_W - 1e-9 \
                        and side not in self.hold.values():
                    self.hold[half] = side
                    self.grip_local[half] = self._in_half(half, tips[:2])
                self.around[(side, half)] = bool(on and gap > HANDLE_W)

        # the slotted half moves with its gripper
        ls = self.hold["slotted"]
        if ls is not None and ls in self.prev_tips:
            self.origin = self.origin + (grips[ls][0][:2] - self.prev_tips[ls][:2])

        # the rivet half follows its gripper about the rivet
        rs = self.hold["rivet"]
        if rs is not None:
            self._follow(grips[rs][0][:2], slotted_held=ls is not None)

        for side, (tips, _) in grips.items():
            self.prev_tips[side] = tips.copy()
        self.max_q = max(self.max_q, self.q)

    def _follow(self, target_xy, slotted_held: bool) -> None:
        g = self.grip_local["rivet"]  # the held point in the rivet half's frame
        glen = float(np.linalg.norm(g))
        gang = math.atan2(g[1], g[0])
        d = self._slot_point(target_xy)  # where the gripper is, slot frame
        was_locked = L.locked(self.sigma)
        if slotted_held and not was_locked:
            # free joint: the rivet slides (along SLIDE_DIR, toward the handles) to put the held point under the
            # gripper, past the arms' give
            along = float(d @ L.SLIDE_DIR)
            across = float(d @ np.array([-L.SLIDE_DIR[1], L.SLIDE_DIR[0]]))
            disc = glen * glen - across * across
            if disc >= 0.0:
                roots = (along - math.sqrt(disc), along + math.sqrt(disc))
                q_hand = min(roots, key=lambda r: abs(r - self.q))  # the branch the held point is on
                if q_hand > self.q + ARM_GIVE:
                    self.q = q_hand - ARM_GIVE
                elif q_hand < self.q - ARM_GIVE:
                    self.q = q_hand + ARM_GIVE
                self.q = float(np.clip(self.q, 0.0, L.SLIDE_TRAVEL))
        rel = d - L.rivet_xy(self.q)
        sigma = math.degrees(_wrap(math.atan2(rel[1], rel[0]) - gang)) - L.SLOTTED_TURN_DEG
        sigma = (sigma + 180.0) % 360.0 - 180.0
        self.sigma = float(np.clip(sigma, L.CLOSED_DEG, L.STOP_DEG))
        if not slotted_held:
            radial = float(np.linalg.norm(rel)) - glen
            if abs(radial) > TOW_GIVE:
                step = (abs(radial) - TOW_GIVE) * math.copysign(1.0, radial)
                move_slot = step * rel / max(float(np.linalg.norm(rel)), 1e-9)
                c, s = math.cos(L.SLOT_YAW), math.sin(L.SLOT_YAW)
                self.origin = self.origin + np.array([c * move_slot[0] - s * move_slot[1],
                                                      s * move_slot[0] + c * move_slot[1]])
                self.towed += abs(step)
        if not was_locked and L.locked(self.sigma):
            self.q = L.nearest_groove_target(self.q)

    def joints(self) -> dict:
        return {"slip": self.q, "spread": L.hinge_of(self.sigma)}

    def pose(self) -> tuple[np.ndarray, float]:
        return np.array([self.origin[0], self.origin[1], L.TOP]), L.SLOT_YAW
