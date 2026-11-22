"""Kinematics of the bypass pruning shears for use ONLY where there is no simulator: tools/offline/check_task.py
and the unit tests. The simulator never reads from this file: there the hinge, slider, springs and bench/tote
contacts are PhysX physics and the interlock rule (task.py) sets joint drive targets each control step.

Model:
- Handle squeeze & carry: a gripper straddling the handle grip zone (x_local in [-0.140, -0.070] m, pad tips
  within the handle height z_local in [-0.004, +0.022] m, y_local within 25 mm of the handles' midpoint) whose
  gap closes below the current open outer handle span squeezes the handles toward 0 deg (open_deg scales down
  linearly with gap, reaching 0 deg once gap <= 42 mm). While clamping the closed/pinched handles (gap <= 46 mm),
  that gripper carries the shears (translation in x, y, z). When the gripper opens past 56 mm, it releases and
  the shears settle onto the bench (z = TOP) or tote floor (z = TOP + TOTE_FLOOR if horizontally inside the tote).
- Return spring: whenever no gripper is squeezing the handles AND the interlock is NOT locked, open_deg springs
  back to THETA_OPEN_DEG (22.0 deg).
- Thumb safety catch: a closed/narrow gripper (gap <= 20 mm) whose pad tips are at catch height
  (z_local in [0.016, 0.036] m) and x_local in [+0.002, +0.026] m pushes the catch tab along -Y (increasing
  q_catch up to CATCH_TRAVEL = 12 mm) as its -Y pad face (y_local - 0.008) advances past the tab's +Y face
  (CATCH_ANCHOR[1] + CATCH_SIZE[1]/2 - q_catch). When not pushed and not locked, q_catch springs back to 0.
- Interlock: uses the exact same Interlock.step(q_jaw, q_catch) rule as the simulator.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate
from interlock import Interlock

GRIP_X_RANGE = (-0.145, -0.070)
GRIP_Z_RANGE = (-0.005, 0.024)
GRIP_Y_TOL = 0.028
CLAMP_GAP_MAX = 0.046  # m: holding both closed handles (38 mm outer span)
CLOSED_GAP_M = 0.042  # m: gap at or below which handles are fully closed (0 deg)
RELEASE_GAP_M = 0.056  # m: opening wider than this lets go

CATCH_X_RANGE = (0.000, 0.028)
CATCH_Z_RANGE = (0.016, 0.036)
PAD_HALF_THICK = 0.008  # closed finger pad outer half-thickness along Y


class OfflineShears:
    def __init__(self) -> None:
        self.interlock = Interlock()
        self.reset()

    def reset(self) -> None:
        self.origin = np.array(L.SHEARS_ORIGIN, dtype=float)
        self.yaw = float(L.SHEARS_YAW)
        self.open_deg = float(L.THETA_OPEN_DEG)
        self.catch_m = 0.0
        self.interlock.reset()
        self.holder: str | None = None
        self.prev_tips: dict[str, np.ndarray] = {}

    def _to_local(self, p_world: np.ndarray) -> np.ndarray:
        d = p_world - self.origin
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]], dtype=float)

    def _in_grip_zone(self, local: np.ndarray) -> bool:
        y_mid, _, _ = L.handle_outer_span_at(L.GRIP_X_LOCAL, self.open_deg)
        return (GRIP_X_RANGE[0] <= local[0] <= GRIP_X_RANGE[1]
                and abs(local[1] - y_mid) <= GRIP_Y_TOL
                and GRIP_Z_RANGE[0] <= local[2] <= GRIP_Z_RANGE[1])

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        grips: dict[str, tuple[np.ndarray, float]] = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            grips[side] = (np.asarray(tips, dtype=float), gap)

        # 1. Handle grip / squeeze / carry
        squeezed = False
        if self.holder is not None:
            tips, gap = grips[self.holder]
            if gap > RELEASE_GAP_M:
                self.holder = None
            else:
                squeezed = True
                if self.holder in self.prev_tips:
                    self.origin = self.origin + (tips - self.prev_tips[self.holder])
                # Map gap to open_deg
                _, span_open, _ = L.handle_outer_span_at(L.GRIP_X_LOCAL, L.THETA_OPEN_DEG)
                if gap <= CLOSED_GAP_M:
                    self.open_deg = 0.0
                else:
                    frac = np.clip((gap - CLOSED_GAP_M) / max(span_open - CLOSED_GAP_M, 1e-6), 0.0, 1.0)
                    self.open_deg = float(frac * L.THETA_OPEN_DEG)
        else:
            for side, (tips, gap) in grips.items():
                local = self._to_local(tips)
                _, span_now, _ = L.handle_outer_span_at(L.GRIP_X_LOCAL, self.open_deg)
                if self._in_grip_zone(local) and gap < span_now - 0.002:
                    squeezed = True
                    _, span_open, _ = L.handle_outer_span_at(L.GRIP_X_LOCAL, L.THETA_OPEN_DEG)
                    if gap <= CLOSED_GAP_M:
                        self.open_deg = 0.0
                    else:
                        frac = np.clip((gap - CLOSED_GAP_M) / max(span_open - CLOSED_GAP_M, 1e-6), 0.0, 1.0)
                        self.open_deg = float(frac * L.THETA_OPEN_DEG)
                    if gap <= CLAMP_GAP_MAX:
                        self.holder = side
                    break

        # Return spring if not held/squeezed and not locked
        if not squeezed and not self.interlock.locked:
            self.open_deg = float(L.THETA_OPEN_DEG)

        # Gravity settle when not held
        if self.holder is None:
            # Check if shears horizontally inside tote
            in_tote_xy = (L.TOTE_IN_X[0] <= self.origin[0] <= L.TOTE_IN_X[1]
                          and L.TOTE_IN_Y[0] <= self.origin[1] <= L.TOTE_IN_Y[1])
            floor_z = L.TOP + (L.TOTE_FLOOR if in_tote_xy else 0.0)
            self.origin[2] = floor_z
        else:
            # Cannot sink below bench/tote floor
            in_tote_xy = (L.TOTE_IN_X[0] <= self.origin[0] <= L.TOTE_IN_X[1]
                          and L.TOTE_IN_Y[0] <= self.origin[1] <= L.TOTE_IN_Y[1])
            floor_z = L.TOP + (L.TOTE_FLOOR if in_tote_xy else 0.0)
            if self.origin[2] < floor_z:
                self.origin[2] = floor_z

        # 2. Thumb safety catch push (by any non-holding gripper)
        pushed = False
        for side, (tips, gap) in grips.items():
            if side == self.holder:
                continue
            local = self._to_local(tips)
            if (gap <= 0.020
                    and CATCH_X_RANGE[0] <= local[0] <= CATCH_X_RANGE[1]
                    and CATCH_Z_RANGE[0] <= local[2] <= CATCH_Z_RANGE[1]):
                # -Y face of the closed pusher pad is at local[1] - PAD_HALF_THICK
                pad_neg_y = local[1] - PAD_HALF_THICK
                tab_start_pos_y = L.CATCH_ANCHOR[1] + L.CATCH_SIZE[1] / 2.0  # +0.012 m
                if pad_neg_y < tab_start_pos_y and local[1] > L.CATCH_ANCHOR[1] - L.CATCH_TRAVEL - 0.015:
                    pushed = True
                    target_q = float(np.clip(tab_start_pos_y - pad_neg_y, 0.0, L.CATCH_TRAVEL))
                    if target_q > self.catch_m:
                        self.catch_m = target_q

        # 3. Interlock rule (identical to simulator @mk.on_step)
        locked, _, catch_tgt = self.interlock.step(L.q_jaw_of(self.open_deg), self.catch_m)
        if locked:
            self.catch_m = catch_tgt  # snaps to 12 mm locked stop
            self.open_deg = 0.0
        elif not pushed:
            self.catch_m = 0.0  # detent spring returns unlocked catch to 0

        for side, (tips, _) in grips.items():
            self.prev_tips[side] = tips.copy()

    def joints(self) -> dict[str, float]:
        return {"jaw": L.q_jaw_of(self.open_deg), "catch": self.catch_m}

    def pose(self) -> tuple[np.ndarray, float]:
        return self.origin.copy(), self.yaw
