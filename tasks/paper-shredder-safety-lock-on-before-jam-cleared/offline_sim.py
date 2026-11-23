"""Kinematics of the shredder switches and jammed mailer for use ONLY where there is no simulator:
tools/offline/check_task.py and unit tests. The simulator never reads from here: in Isaac Sim every body and joint
is simulated by PhysX and read directly from the articulation/rigid-body states.

Offline model:
- Disconnect power rocker (`power_rocker`, 0 deg = ON, 24 deg = OFF): a closed gripper pad over the OFF (-x) half
  presses it down; past 12 deg centre it snaps to 24 deg (OFF), before centre it springs back to 0 deg (ON).
- Control slide switch (`control_switch`, 0 mm Auto-On, 10 mm O, 20 mm Reverse): a closed gripper pad pushing the
  knob's -y face along +Y slides it, or pads closed across x onto the 24 mm knob carry it along y both ways;
  released, it moves to its detent (below 4 mm Auto-On, 4 to 16 mm O, else Reverse).
- Mains plug (`plug` on the outlet, 0 = seated, 35 mm travel along +Y): pinched across x (26 mm body, gap <= 29 mm)
  with the pad tips on the body, it follows the TCP along +Y (never pushed back in past 0).
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
    # mechkit.state_from_poses uses (x, y, z, qx, qy, qz, qw), verified in control-box offline_box.py:
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
        self.ctrl_m = 0.0  # knob displacement: 0 Auto-On, 0.010 O, 0.020 Reverse
        self.ctrl_grip: tuple[int, float] | None = None  # (arm index, tips_y - knob_centre_y) while the knob is pinched
        self.plug_out = 0.0  # plug pulled out of the outlet (m)
        self.plug_hold: tuple[str, float] | None = None  # (side, tcp_y - plug_centre_y)
        self.mailer_pos = np.array([L.MAILER_ORIGIN[0], L.MAILER_ORIGIN[1], L.TOP + L.MAILER_ORIGIN[2]])
        self.mailer_rot = np.eye(3)
        self.mailer_flat = False
        self.held: tuple[str, np.ndarray, float] | None = None  # (side, offset, grab_y)
        self.around = {s: False for s in rigstate.ARMS}
        self.events: list[str] = []

    def shredder_joints(self) -> dict[str, float]:
        return {
            "power_rocker": math.radians(self.rocker_deg),
            "control_switch": self.ctrl_m,
        }

    def outlet_joints(self) -> dict[str, float]:
        return {"plug": self.plug_out}

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

    def _step_ctrl(self, read) -> None:
        pushed = False
        face0 = L.CTRL_XY[1] - L.CTRL_KNOB[1] / 2  # knob's -y face at Auto-On
        # Pinched knob: pads across x closed onto the 24 mm knob (gap <= 26 mm) with the tips beside it -> the knob
        # follows the TCP along y both ways (clamped to its travel) until the pads open again.
        if self.ctrl_grip is not None:
            idx, off = self.ctrl_grip
            row = read[idx] if idx < len(read) else None
            gap = float(np.clip(row[2], 0.0, 1.0)) * rigstate.FULL_GAP if row is not None else 1.0
            if row is None or gap > L.CTRL_KNOB[0] + 0.002:
                self.ctrl_grip = None
            else:
                self.ctrl_m = float(np.clip(float(row[1][1]) - off - L.CTRL_XY[1], 0.0, L.CTRL_TRAVEL))
                return
        for idx, (_, tips, opening) in enumerate(read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            tip_h = float(tips[2]) - L.TOP
            knob_y = L.CTRL_XY[1] + self.ctrl_m
            if (gap <= L.CTRL_KNOB[0] + 0.002 and abs(tips[0] - L.CTRL_XY[0]) <= 0.008
                    and abs(tips[1] - knob_y) <= 0.006 and L.SHRED_DECK_Z <= tip_h <= L.CTRL_KNOB_Z1 - 0.002):
                self.ctrl_grip = (idx, float(tips[1]) - knob_y)
                return
            if gap > CLOSED_GAP:
                continue
            tip_h = float(tips[2]) - L.TOP
            if abs(tips[0] - L.CTRL_XY[0]) <= 0.020 and L.SHRED_DECK_Z - 0.005 <= tip_h <= L.CTRL_KNOB_Z1:
                front_y = float(tips[1]) + PAD_HALF_T
                if front_y > face0 + self.ctrl_m and float(tips[1]) < L.CTRL_XY[1] + L.CTRL_TRAVEL + 0.015:
                    new_q = min(L.CTRL_TRAVEL, front_y - face0)
                    if new_q > self.ctrl_m:
                        self.ctrl_m = new_q
                        pushed = True
        if not pushed:
            if self.ctrl_m < L.CTRL_AUTO_EDGE:
                goal = 0.0
            elif self.ctrl_m < L.CTRL_REV_EDGE:
                goal = L.CTRL_OFF_POS
            else:
                goal = L.CTRL_TRAVEL
            self.ctrl_m += float(np.clip(goal - self.ctrl_m, -0.004, 0.004))

    def plug_centre(self) -> np.ndarray:
        return np.array([L.OUTLET_C[0], L.OUTLET_FACE_Y + L.PLUG_SIZE[1] / 2 + self.plug_out, L.TOP + L.PLUG_Z])

    def _step_plug(self, arms) -> None:
        if not rigstate.readable(arms):
            return
        for side in rigstate.ARMS:
            a = arms[side]
            tips = np.asarray(a.tips, dtype=float)
            tcp = np.asarray(a.tcp, dtype=float)
            gap = float(np.clip(a.opening, 0.0, 1.0)) * rigstate.FULL_GAP
            if self.plug_hold is not None and self.plug_hold[0] == side:
                if gap > L.PLUG_SIZE[0] + 0.006:
                    self.plug_hold = None
                else:
                    target_y = float(tcp[1]) - self.plug_hold[1]
                    dy = target_y - (L.OUTLET_FACE_Y + L.PLUG_SIZE[1] / 2)
                    self.plug_out = float(np.clip(max(dy, self.plug_out), 0.0, L.PLUG_UPPER))
                continue
            pc = self.plug_centre()
            if (self.plug_hold is None and gap <= L.PLUG_SIZE[0] + 0.003
                    and abs(tips[0] - pc[0]) <= 0.010
                    and abs(tips[1] - pc[1]) <= L.PLUG_SIZE[1] / 2 + 0.010
                    and pc[2] - L.PLUG_SIZE[2] / 2 - 0.003 <= tips[2] <= pc[2] + L.PLUG_SIZE[2] / 2 + 0.003):
                self.plug_hold = (side, float(tcp[1] - pc[1]))
                self.events.append(f"plug_grabbed_{side}")

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
        self._step_ctrl(read)
        self._step_plug(arms)
        self._step_mailer(arms)
