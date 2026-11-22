"""Kinematics of the shredder switches and jammed mailer for use ONLY where there is no simulator:
tools/offline/check_task.py and unit tests. The simulator never reads from here: in Isaac Sim every body and joint
is simulated by PhysX and read directly from the articulation/rigid-body states.

Offline model:
- Power rocker (`power_rocker`, 0 deg = ON, 24 deg = OFF): a closed gripper pad over the OFF (-x) half presses it down;
  past 12 deg centre it snaps to 24 deg (OFF), before centre it springs back to 0 deg (ON).
- Orange Safety Lock (`safety_lock`, 0 mm = UNLOCKED, 20 mm = LOCKED): a closed gripper pad pushing the tab's rear face
  along +X slides it forward; past 10 mm midpoint it snaps to 20 mm (LOCKED), before 10 mm it springs back to 0 mm.
- Corrugated mailer: starts upright in the feed slot. An open gripper descending astride the 4 mm thick mailer and
  closing below 12 mm grips it; moving up lifts it out of the slot. In the discard tray, opening the gripper and
  nudging toward -y tips it flat onto the tray floor.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

CLOSED_GAP = 0.006
GRAB_GAP = 0.015
LET_GO_GAP = 0.025
PAD_HALF_W = 0.0135
PAD_HALF_T = 0.008


def _quat_from_rot(r: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (x, y, z, w) quaternion (mechkit.state_from_poses uses xyzw? Check!)."""
    # mechkit.state_from_poses uses (x, y, z, qx, qy, qz, qw) — verified in control-box offline_box.py:
    # (BO[0], BO[1], L.TOP, 0.0, 0.0, 0.0, 1.0)
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(roll: float) -> np.ndarray:
    c, s = math.cos(roll), math.sin(roll)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class OfflineSim:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rocker_deg = 0.0  # joint angle: 0 = ON (+12 deg), 24 = OFF (-12 deg)
        self.lock_m = 0.0  # slider displacement: 0 = UNLOCKED, 0.020 = LOCKED
        self.mailer_pos = np.array([L.MAILER_ORIGIN[0], L.MAILER_ORIGIN[1], L.TOP + L.MAILER_ORIGIN[2]])
        self.mailer_rot = np.eye(3)
        self.mailer_flat = False
        self.held: tuple[str, np.ndarray, float] | None = None  # (side, offset, grab_y)
        self.around = {s: False for s in rigstate.ARMS}
        self.events: list[str] = []

    def shredder_joints(self) -> dict[str, float]:
        return {
            "power_rocker": math.radians(self.rocker_deg),
            "safety_lock": self.lock_m,
        }

    def mailer_pose(self) -> tuple[float, ...]:
        qx, qy, qz, qw = _quat_from_rot(self.mailer_rot)
        return (float(self.mailer_pos[0]), float(self.mailer_pos[1]), float(self.mailer_pos[2]), qx, qy, qz, qw)

    def _step_rocker(self, read) -> None:
        # Check if any closed gripper pad presses the OFF (-x) half of the power rocker
        press_x = L.ROCKER_XY[0] - L.ROCKER_PRESS_U
        press_y = L.ROCKER_XY[1]
        pressed = False
        for _, tips, opening in read:
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            if gap > CLOSED_GAP:
                continue
            if abs(tips[0] - press_x) <= 0.016 and abs(tips[1] - press_y) <= 0.020:
                tip_h = float(tips[2]) - L.TOP
                top_now = L.rocker_top_z(L.ROCKER_PRESS_U, self.rocker_deg)
                if tip_h < top_now:
                    # Push down proportionally to overlap
                    for deg in np.linspace(self.rocker_deg, L.ROCKER_TRAVEL_DEG, 49):
                        if L.rocker_top_z(L.ROCKER_PRESS_U, float(deg)) <= tip_h + 0.002:
                            self.rocker_deg = max(self.rocker_deg, float(deg))
                            pressed = True
                            break
                    else:
                        self.rocker_deg = L.ROCKER_TRAVEL_DEG
                        pressed = True
        if not pressed:
            if self.rocker_deg >= L.ROCKER_CENTRE_DEG:
                self.rocker_deg = min(L.ROCKER_TRAVEL_DEG, self.rocker_deg + 4.0)
            else:
                self.rocker_deg = max(0.0, self.rocker_deg - 4.0)

    def _step_lock(self, read) -> None:
        pushed = False
        tab_y = L.LOCK_XY[1]
        for _, tips, opening in read:
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            if gap > CLOSED_GAP:
                continue
            tip_h = float(tips[2]) - L.TOP
            if abs(tips[1] - tab_y) <= 0.022 and L.SHRED_DECK_Z - 0.005 <= tip_h <= L.LOCK_TAB_Z1:
                front_x = float(tips[0]) + PAD_HALF_T
                rear_face = L.LOCK_XY[0] - L.LOCK_TAB[0] / 2 + self.lock_m
                if front_x > rear_face and float(tips[0]) < L.LOCK_XY[0] + L.LOCK_TRAVEL + 0.015:
                    new_q = min(L.LOCK_TRAVEL, max(self.lock_m, front_x - (L.LOCK_XY[0] - L.LOCK_TAB[0] / 2)))
                    if new_q > self.lock_m:
                        self.lock_m = new_q
                        pushed = True
        if not pushed:
            if self.lock_m >= L.LOCK_MID:
                self.lock_m = min(L.LOCK_TRAVEL, self.lock_m + 0.004)
            else:
                self.lock_m = max(0.0, self.lock_m - 0.004)

    def _step_mailer(self, arms) -> None:
        if not rigstate.readable(arms):
            return
        for side in rigstate.ARMS:
            a = arms[side]
            tips = np.asarray(a.tips, dtype=float)
            gap = float(np.clip(a.opening, 0.0, 1.0)) * rigstate.FULL_GAP
            if self.held is not None and self.held[0] == side:
                if gap > LET_GO_GAP:
                    self.held = None
                    # If released over the discard tray, tip flat into the tray!
                    mx, my = float(self.mailer_pos[0]), float(self.mailer_pos[1])
                    if L.TRAY_IN_X[0] - 0.02 <= mx <= L.TRAY_IN_X[1] + 0.02 and L.TRAY_IN_Y[0] - 0.02 <= my <= L.TRAY_IN_Y[1] + 0.02:
                        self.mailer_flat = True
                        # Rotated 90 deg around Z (160 mm along X), then 90 deg around X (120 mm along Y, 4 mm along Z)
                        self.mailer_rot = _rot_x(math.pi / 2) @ _rot_z(math.pi / 2)
                        flat_y = max(
                            L.TRAY_IN_Y[0] + L.MAILER_SIZE[2] / 2 + 0.005,
                            min(L.TRAY_IN_Y[1] - L.MAILER_SIZE[2] / 2 - 0.005, my - 0.045),
                        )
                        flat_x = max(
                            L.TRAY_IN_X[0] + L.MAILER_SIZE[1] / 2 + 0.005,
                            min(L.TRAY_IN_X[1] - L.MAILER_SIZE[1] / 2 - 0.005, mx),
                        )
                        self.mailer_pos = np.array([flat_x, flat_y, L.TOP + L.TRAY_FLOOR + L.MAILER_SIZE[0] / 2])
                        self.events.append("mailer_laid_flat")
                    else:
                        # Drop vertically onto table or slot
                        self.mailer_pos[2] = max(L.TOP + L.MAILER_SIZE[2] / 2, self.mailer_pos[2] - 0.02)
                else:
                    offset = self.held[1]
                    self.mailer_pos = tips + offset
                    # When carried over to the tray (y < -0.14), the right arm rotates yaw by 90 deg
                    if float(tips[1]) < -0.14 and not self.mailer_flat:
                        self.mailer_rot = _rot_z(math.pi / 2)
                return

        if self.held is None and not self.mailer_flat:
            for side in rigstate.ARMS:
                a = arms[side]
                tips = np.asarray(a.tips, dtype=float)
                gap = float(np.clip(a.opening, 0.0, 1.0)) * rigstate.FULL_GAP
                near_xy = abs(tips[0] - self.mailer_pos[0]) <= 0.020 and abs(tips[1] - self.mailer_pos[1]) <= 0.070
                near_z = self.mailer_pos[2] - 0.020 <= tips[2] <= self.mailer_pos[2] + 0.065
                if near_xy and near_z and gap > GRAB_GAP:
                    self.around[side] = True
                elif not (near_xy and near_z):
                    self.around[side] = False
                if self.around[side] and gap <= GRAB_GAP:
                    self.held = (side, self.mailer_pos - tips, float(tips[1]))
                    self.events.append(f"mailer_grabbed_{side}")
                    break

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        self._step_rocker(read)
        self._step_lock(read)
        self._step_mailer(arms)
