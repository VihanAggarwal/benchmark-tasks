"""Offline physics stand-in for tools/offline/check_task.py and unit tests (never used in the simulator).

Models:
1. NEMA 1-15P plug slider along +Y: pinched across x (26 mm body, gap <= 29 mm) with pad tips covering the body;
   follows TCP along +Y up to PLUG_UPPER (35 mm), never pushes back in past 0.
2. Spring-loaded release button slider along -Z: a closed gripper (gap < 12 mm) positioned over the 18 x 18 mm cap
   pushes it down by (BUTTON_TOP_Z - tips_z) clamped to [0, BUTTON_TRAVEL]; otherwise the spring returns it to 0.
3. Blending wand coupling bond: locked to the motor body at spawn. Releases while button >= BUTTON_RELEASE_AT (4 mm);
   if button drops below 4 mm while the spigot is withdrawn < WAND_CLEAR_LATCH (5 mm), re-engages the lock.
4. Coupling socket: while the spigot is still in the socket (withdrawn < SPIGOT_LEN), the unlocked wand moves only
   straight out along -y with the gripper (never sideways, never further in) and stays put when let go.
5. Blending wand carry & drop: pinched across the 18 mm stainless shaft (gap <= 22 mm) with pad tips near the shaft;
   once out of the socket, follows the gripper TCP (aligning along X when carried over the sink wash tray y < -0.15). When
   released, settles flat onto the tray floor (if over the tray interior) or onto the bench top.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk

CAPTURE = 0.010


def _quat_rz(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class BlenderModel:
    def __init__(self, motor_spec, wand_spec, bench_spec) -> None:
        self.specs = {"motor": motor_spec, "wand": wand_spec, "bench": bench_spec}
        self.reset()

    def reset(self) -> None:
        self.plug_out = 0.0
        self.button = 0.0
        self.bonded = True
        self.wand_o = L.WAND_SEATED_ORIGIN.copy()
        self.wand_yaw = 0.0  # 0: wand extends along world -Y; -pi/2: extends along world -X
        self.holds: dict = {}  # side -> (part, offset)
        self.mode = "rest"

    def plug_centre(self) -> np.ndarray:
        return np.array([
            float(L.OUTLET_C[0]),
            L.OUTLET_FACE_Y + L.PLUG_SIZE[1] / 2 + self.plug_out,
            L.TOP + L.PLUG_Z,
        ], dtype=float)

    def wand_grip_world(self) -> np.ndarray:
        c, s = math.cos(self.wand_yaw), math.sin(self.wand_yaw)
        dx = -s * L.WAND_GRIP_LOCAL_Y
        dy = c * L.WAND_GRIP_LOCAL_Y
        return self.wand_o + np.array([dx, dy, 0.0], dtype=float)

    def _wand_disp(self) -> float:
        """Axial withdrawal of the spigot out of the socket (m); the model only moves it along -y while engaged."""
        if self.wand_yaw != 0.0:
            return math.inf
        return float(L.WAND_SEATED_ORIGIN[1] - self.wand_o[1])

    def _engaged(self) -> bool:
        return self._wand_disp() < L.SPIGOT_LEN

    def step(self, arms) -> None:
        if arms is None:
            return

        # 1. Check release button press from any closed gripper over the button cap
        max_press = 0.0
        bx, by = L.BUTTON_X, L.BUTTON_Y
        top_z = L.TOP + L.BUTTON_TOP_Z
        for side in ("left", "right"):
            a = arms.get(side)
            if a is None:
                continue
            gap = float(a.opening) * L.FULL_GAP
            tips = np.asarray(a.tips, dtype=float)
            if gap < 0.012 and abs(tips[0] - bx) <= 0.014 and abs(tips[1] - by) <= 0.014:
                if tips[2] < top_z and tips[2] >= L.TOP + L.MOTOR_TOP_Z - 0.015:
                    press = min(L.BUTTON_TRAVEL, max(0.0, top_z - tips[2]))
                    if press > max_press:
                        max_press = press
        self.button = max_press

        # 2. Update wand coupling bond
        if self.bonded and self.button >= L.BUTTON_RELEASE_AT:
            self.bonded = False
        elif not self.bonded and self.button < L.BUTTON_RELEASE_AT and self._wand_disp() < L.WAND_CLEAR_LATCH:
            self.bonded = True

        # 3. Process grasps & carries
        for side in ("left", "right"):
            a = arms.get(side)
            if a is None:
                continue
            gap = float(a.opening) * L.FULL_GAP
            tips = np.asarray(a.tips, dtype=float)
            tcp = np.asarray(a.tcp, dtype=float)

            if side in self.holds:
                part, off = self.holds[side]
                width = L.PLUG_SIZE[0] if part == "plug" else 2 * L.SHAFT_R
                if gap > width + 0.006:
                    del self.holds[side]
                    if part == "wand":
                        self._drop_wand()
                else:
                    if part == "plug":
                        target_y = float(tcp[1] - off[1])
                        dy = target_y - (L.OUTLET_FACE_Y + L.PLUG_SIZE[1] / 2)
                        self.plug_out = float(np.clip(dy, 0.0, L.PLUG_UPPER))
                        self.mode = "pull_plug"
                    elif part == "wand":
                        if not self.bonded and self._engaged():
                            target_y = float(tcp[1] - off[1])
                            self.wand_o = L.WAND_SEATED_ORIGIN.copy()
                            self.wand_o[1] = min(float(L.WAND_SEATED_ORIGIN[1]), target_y)
                            self.mode = "withdraw_wand"
                        elif not self.bonded:
                            self.wand_yaw = -math.pi / 2 if float(tcp[1]) < -0.150 else 0.0
                            c, s = math.cos(self.wand_yaw), math.sin(self.wand_yaw)
                            gx = -s * L.WAND_GRIP_LOCAL_Y
                            gy = c * L.WAND_GRIP_LOCAL_Y
                            self.wand_o = np.array([
                                float(tcp[0]) - gx,
                                float(tcp[1]) - gy,
                                float(tcp[2] - off[2]),
                            ], dtype=float)
                            self.mode = "carry_wand"
                continue

            # New plug grasp?
            pc = self.plug_centre()
            if (gap <= L.PLUG_SIZE[0] + 0.003
                    and abs(tips[0] - pc[0]) <= CAPTURE
                    and abs(tips[1] - pc[1]) <= L.PLUG_SIZE[1] / 2 + 0.010
                    and pc[2] - L.PLUG_SIZE[2] / 2 - 0.003 <= tips[2] <= pc[2] + L.PLUG_SIZE[2] / 2 + 0.003):
                self.holds[side] = ("plug", tcp - pc)
                continue

            # New wand shaft grasp?
            wg = self.wand_grip_world()
            if (gap <= 2 * L.SHAFT_R + 0.004
                    and math.hypot(tips[0] - wg[0], tips[1] - wg[1]) <= 0.012
                    and wg[2] - L.SHAFT_R - 0.006 <= tips[2] <= wg[2] + L.SHAFT_R + 0.004):
                self.holds[side] = ("wand", tcp - self.wand_o)
                continue

    def _drop_wand(self) -> None:
        if self.bonded or self._engaged():
            return
        c, s = math.cos(self.wand_yaw), math.sin(self.wand_yaw)
        mid_y = (L.SPIGOT_LEN - L.WAND_LEN) / 2
        mid = self.wand_o + np.array([-s * mid_y, c * mid_y, 0.0], dtype=float)
        over_tray = (L.TRAY_INNER_X[0] <= mid[0] <= L.TRAY_INNER_X[1]
                     and L.TRAY_INNER_Y[0] <= mid[1] <= L.TRAY_INNER_Y[1])
        support_z = L.TOP + (L.TRAY_FLOOR if over_tray else 0.0)
        self.wand_o[2] = support_z + L.BELL_R
        self.mode = "in_tray" if over_tray else "dropped"

    def states(self) -> dict:
        s = self.specs
        motor_s = mk.state_from_poses(
            s["motor"],
            (L.BLENDER_X, L.SEAM_Y, L.TOP, 0.0, 0.0, 0.0, 1.0),
            joints={"release": self.button},
            source="offline",
        )
        bench_s = mk.state_from_poses(
            s["bench"],
            (0.0, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
            joints={"plug": self.plug_out},
            source="offline",
        )
        wand_s = mk.state_from_poses(
            s["wand"],
            (*self.wand_o.tolist(), *_quat_rz(self.wand_yaw)),
            source="offline",
        )
        return {"motor": motor_s, "wand": wand_s, "bench": bench_s}
