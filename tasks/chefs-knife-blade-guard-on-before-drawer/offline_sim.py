"""Offline kinematics of the chef's knife and blade guard for use ONLY where there is no simulator
(tools/offline/check_task.py and offline tests). Isaac Sim never reads from this file: on the GPU
every part is a PhysX rigid body or hinged link moved by gripper contacts and the mechkit weld latch.

Models the physical mechanics claimed by the design from rigstate's offline arm snapshots:
- Grasping the knife handle (left/right gripper closing < 30 mm while straddling the handle) or the
  blade guard (closing < 60 mm while straddling the guard).
- Insertion resistance from the guard's sprung retention leaf (~2 N normal force, pinching the blade onto
  the bottom plate: more drag than the ~1 N static friction of the 180 g knife on the cutting board): if the guard slides into the blade tip while the knife is NOT
  held by a gripper, the knife slides forward along +y ahead of the guard instead of sheathing.
- Heel stop: the handle face at the heel stops a guard on the blade line at the seat (mouth flush with the heel).
- Heel retention catch: once the robot has slid the guard onto a stabilized blade from the tip to within
  CATCH_TOL (1 mm) of the heel, or against the heel stop, the guard locks onto the blade and moves with the knife.
- Support surfaces on release: cutting board top (BOARD_TOP), utensil drawer floor (DRAWER_FLOOR_Z),
  or bench top (TOP).
"""

from __future__ import annotations

import types

import numpy as np

import layout as L
import rigstate
import scoring as S

Q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
SEAT_OFFSET = np.array([L.GUARD_SEAT_X, L.GUARD_SEAT_Y, L.GUARD_SEAT_Z], dtype=float)


def _support_z(x: float, y: float) -> float:
    bx0 = L.BOARD_CENTER[0] - L.BOARD_SIZE[0] / 2
    bx1 = L.BOARD_CENTER[0] + L.BOARD_SIZE[0] / 2
    by0 = L.BOARD_CENTER[1] - L.BOARD_SIZE[1] / 2
    by1 = L.BOARD_CENTER[1] + L.BOARD_SIZE[1] / 2
    if bx0 <= x <= bx1 and by0 <= y <= by1:
        return float(L.BOARD_TOP)
    if L.DRAWER_INNER_X[0] <= x <= L.DRAWER_INNER_X[1] and L.DRAWER_INNER_Y[0] <= y <= L.DRAWER_INNER_Y[1]:
        return float(L.DRAWER_FLOOR_Z)
    return float(L.TOP)


class OfflineKnifeSim:
    """Pure-Python kinematic mirror for check_task.py when no physics simulator is attached."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.knife = np.array([float(L.KNIFE_START[0]), float(L.KNIFE_START[1]), L.BOARD_TOP], dtype=float)
        self.guard = np.array([float(L.GUARD_START[0]), float(L.GUARD_START[1]), L.BOARD_TOP + L.GUARD_H / 2 + 0.0002],
                              dtype=float)
        self.prev_knife = self.knife.copy()
        self.prev_guard = self.guard.copy()
        self.knife_vel = np.zeros(3, dtype=float)
        self.guard_vel = np.zeros(3, dtype=float)
        self.held: dict[str, tuple[str, np.ndarray]] = {}  # arm -> (obj_name, offset from tcp)
        self.welded = False
        self.on_from_tip = False

    def _holders(self, name: str) -> list[str]:
        return [a for a, (obj, _) in self.held.items() if obj == name]

    def step(self, arms) -> dict:
        self.prev_knife = self.knife.copy()
        self.prev_guard = self.guard.copy()

        if rigstate.readable(arms):
            for side in ("left", "right"):
                arm = arms[side]
                tcp = np.asarray(arm.tcp, dtype=float)
                tips = np.asarray(arm.tips, dtype=float)
                gap = max(0.0, min(1.0, float(arm.opening))) * L.FINGER_GAP

                if side in self.held:
                    obj, _ = self.held[side]
                    release_gap = 0.036 if obj == S.KNIFE else 0.066
                    if gap > release_gap:
                        del self.held[side]
                else:
                    # Try grasping knife handle
                    if (gap < 0.030
                            and abs(tcp[0] - self.knife[0]) <= 0.016
                            and self.knife[1] - 0.005 <= tcp[1] <= self.knife[1] + L.HANDLE_Y[1] + 0.005
                            and abs(tips[2] - self.knife[2]) <= 0.025):
                        self.held[side] = (S.KNIFE, self.knife - tcp)
                    # Try grasping blade guard
                    elif (gap < 0.060
                          and abs(tcp[0] - self.guard[0]) <= 0.022
                          and abs(tcp[1] - self.guard[1]) <= L.GUARD_L / 2 + 0.005
                          and abs(tips[2] - (self.guard[2] - L.GUARD_H / 2)) <= 0.022):
                        self.held[side] = (S.GUARD, self.guard - tcp)

            # Move held objects with their primary holder
            k_holders = self._holders(S.KNIFE)
            g_holders = self._holders(S.GUARD)
            if k_holders:
                arm = k_holders[0]
                self.knife = np.asarray(arms[arm].tcp, dtype=float) + self.held[arm][1]
            if g_holders and not self.welded:
                arm = g_holders[0]
                self.guard = np.asarray(arms[arm].tcp, dtype=float) + self.held[arm][1]
                # When sliding onto a held blade within the guard slot's capture width, the slot self-centers on the blade
                if k_holders:
                    target_x = self.knife[0] + L.GUARD_SEAT_X
                    target_z = self.knife[2] + L.GUARD_SEAT_Z
                    if abs(self.guard[0] - target_x) <= 0.036 and abs(self.guard[2] - target_z) <= 0.010:
                        self.guard[0] = target_x
                        self.guard[2] = target_z

            # Insertion resistance: pushing guard into unheld knife slides knife forward along +y
            if g_holders and not k_holders and not self.welded:
                dx = abs(self.guard[0] - (self.knife[0] + L.GUARD_SEAT_X))
                dz = abs(self.guard[2] - (self.knife[2] + L.GUARD_SEAT_Z))
                mouth_y = self.guard[1] + L.GUARD_L / 2
                tip_y = self.knife[1] + L.BLADE_Y[0]
                if dx <= L.GUARD_W / 2 and dz <= L.GUARD_H and mouth_y > tip_y + 0.015:
                    self.knife[1] = mouth_y - L.BLADE_Y[0] - 0.015

            # Heel stop: the guard's top and bottom plates meet the handle face at the heel (y = 0), so a guard on the
            # blade line cannot slide past the seat; a hand pushing further only presses it against the stop.
            if not self.welded:
                _, gy0 = S.guard_on(S.Body(self.knife, np.eye(3), np.zeros(3)),
                                    S.Body(self.guard, np.eye(3), np.zeros(3)))
                if gy0 is not None and L.GUARD_SEAT_Y < gy0 <= L.GUARD_SEAT_Y + 0.050:
                    self.guard[1] = self.knife[1] + L.GUARD_SEAT_Y

            # Settle unheld objects onto support surface
            if not k_holders:
                sup_k = _support_z(float(self.knife[0]), float(self.knife[1] + 0.065))
                if self.knife[2] > sup_k:
                    self.knife[2] = sup_k
            if self.welded:
                self.guard = self.knife + SEAT_OFFSET
            elif not g_holders:
                sup_g = _support_z(float(self.guard[0]), float(self.guard[1])) + L.GUARD_H / 2
                if self.guard[2] > sup_g:
                    self.guard[2] = sup_g

            # Evaluate catch_ready weld rule
            if not self.welded:
                k_body = S.Body(self.knife, np.eye(3), np.zeros(3))
                g_body = S.Body(self.guard, np.eye(3), np.zeros(3))
                covered, gy = S.guard_on(k_body, g_body)
                ready, self.on_from_tip = S.catch_ready(covered, gy, self.on_from_tip)
                if ready:
                    self.welded = True
                    self.guard = self.knife + SEAT_OFFSET

        self.knife_vel = (self.knife - self.prev_knife) / L.DT
        self.guard_vel = (self.guard - self.prev_guard) / L.DT

        return {
            S.KNIFE: types.SimpleNamespace(origin=self.knife.copy(), quat_xyzw=Q0.copy(), lin_vel=self.knife_vel.copy()),
            S.GUARD: types.SimpleNamespace(origin=self.guard.copy(), quat_xyzw=Q0.copy(), lin_vel=self.guard_vel.copy()),
        }
