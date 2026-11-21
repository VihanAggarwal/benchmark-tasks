"""Offline stand-in for the physics (never used in the simulator). The checker's proxy carries a gripped object rigidly and
drops it straight down; it knows no pinch that squares a board, no pistol grip and no ratchet. This model gives the scorer,
from the arms' fingertip readings only:

STUB A    pinched across its width (the pads closed on its 38 mm, tips under its top, the grip over it): the pads' faces are
          parallel to the wrist's yaw, so A is squared to the bench's x axis and centred on the gripper's axis (that is how a
          pinch squares a board), and follows the hand along x down to B's face. Let go, it stays.
CLAMP     hung over the corner from the start, pads clear of both boards; held at its pistol grip (pads on the handle's far
          face and the trigger's far face) it follows the hand until the jaw has taken its first step, then stays put.
TRIGGER   the pads' gap, down to the stop 46 mm, turns the trigger; a gap above 64 mm leaves it at rest.
RATCHET   a squeeze past 80% of the travel pulls the moving jaw one 3 mm step (the pawl), once, until the trigger has been
          back under 30%. The jaw follows the pull until its pad meets A; then A is pushed to B (squared by the flat pad) and B
          to the fixed pad, and the pull's excess is the clamp force through the stiffness.
The release tab is not touched by any demo and is not modelled (a hand on it is a scorer test).
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk
import rigstate
import scoring as S

ARMS = rigstate.ARMS  # ("left", "right")
HOLD_A_GAP = 0.044
HOLD_C_GAP = 0.068
HAND_REACH = 0.025


def _quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class ClampModel:
    def __init__(self, a_part, b_part, clamp_part) -> None:
        self.parts = {S.A: a_part, S.B: b_part, S.CLAMP: clamp_part}
        self.reset()

    def reset(self) -> None:
        self.a = [L.A_CX + L.A_START_DX, L.A_CY + L.A_START_DY, L.A_START_YAW]  # x, y, yaw
        self.b = [L.B_CX, L.B_CY]
        self.c = [L.CLAMP_XF, L.CLAMP_YC0]
        self.jaw = 0.0
        self.target = 0.0
        self.armed = True
        self.trig = 0.0
        self.a_off = None
        self.grab: dict = {}  # arm -> offset of the hand from the clamp's grip point when it took hold
        self.holds: dict = {}

    # -- poses for the scorer --
    def _pose(self, x, y, yaw) -> S.Pose:
        c, s = math.cos(yaw), math.sin(yaw)
        return S.Pose(np.array([x, y, 0.0]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))

    def _geom(self):
        a, b, c = self._pose(*self.a), self._pose(*self.b, 0.0), self._pose(*self.c, 0.0)
        return S.clamp_geometry(a, b, c, self.jaw)

    def engaged(self) -> bool:
        g = self._geom()
        return bool(g["cover_a"] >= L.PAD_OVERLAP_MIN and g["cover_b"] >= L.PAD_OVERLAP_MIN
                    and g["fixed_gap"] >= -0.0005 and g["moving_gap"] >= -0.0005)

    # -- one control step --
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        self.holds = {}
        pinned = self.target > 0.0  # once the jaw has taken a step the clamp is on the work and stays there
        f = 0.0
        for side, (tcp, tips, opening) in zip(ARMS, read):
            gap = float(opening) * L.GRIPPER_FULL
            u_tip = float(tips[2]) - L.TOP
            # a pinch on A
            if (gap <= HOLD_A_GAP and abs(tcp[0] - self.a[0]) <= 0.045 and abs(tcp[1] - self.a[1]) <= 0.015
                    and 0.002 <= u_tip <= 0.017 and tips[2] < tcp[2] - 0.005):
                self.holds[side] = "a"
                if self.a_off is None:
                    self.a_off = self.a[0] - float(tcp[0])  # where on A the hand took it
                self.a[1] = float(tcp[1])
                self.a[2] = 0.0
                self.a[0] = max(float(tcp[0]) + self.a_off, L.B_X1 + L.A_LEN / 2 - 0.0005)
                g = self._geom()
                if g["cover_a"] >= L.PAD_OVERLAP_MIN:  # the moving pad is on A's end: the hand cannot push A through it
                    self.a[0] = min(self.a[0], self.c[0] + L.G0 - self.jaw - L.A_LEN / 2)
                continue
            # the pistol grip
            hx, hy = self.c[0] + L.GRIP_LX - self.jaw, self.c[1] + L.GRIP_LY
            if (gap <= HOLD_C_GAP and abs(tcp[0] - hx) <= HAND_REACH and abs(tcp[1] - hy) <= HAND_REACH
                    and 0.003 <= u_tip <= 0.026 and tips[2] < tcp[2] - 0.005):
                self.holds[side] = "clamp"
                if side not in self.grab:
                    self.grab[side] = (float(tcp[0]) - hx, float(tcp[1]) - hy)
                span = max(gap, L.SPAN_STOP)
                f = max(f, float(np.clip((L.SPAN_REST - span) / L.TRIG_TRAVEL, 0.0, 1.0)))
                if not pinned:
                    ox, oy = self.grab[side]
                    self.c[0] = float(tcp[0]) - ox - L.GRIP_LX + self.jaw
                    self.c[1] = float(tcp[1]) - oy - L.GRIP_LY
            else:
                self.grab.pop(side, None)
        self.trig = f * L.TRIG_MAX
        if f >= L.TRIG_ON and self.armed:
            self.target = min(self.target + L.STEP, L.JAW_UPPER)
            self.armed = False
        elif f <= L.TRIG_OFF:
            self.armed = True
        self._resolve()

    def _resolve(self) -> None:
        """The jaw follows its pull; on the boards the pad pushes A to B and B to the fixed pad."""
        if not self.engaged():
            self.jaw = min(self.target, L.JAW_UPPER)
            return
        xf = self.c[0]
        xm_goal = xf + L.G0 - self.target
        a_end = self.a[0] + L.A_LEN / 2
        if xm_goal < a_end:
            self.a[0] = max(xm_goal - L.A_LEN / 2, xf + L.SPAN_BOARDS - L.A_LEN / 2)
            self.a[2] = 0.0  # the flat pad squares A's end against B
            self.b[0] = min(self.b[0], self.a[0] - L.A_LEN / 2 - L.BOARD_W / 2)
            self.b[0] = max(self.b[0], xf + L.BOARD_W / 2)
            xm = max(xm_goal, self.a[0] + L.A_LEN / 2)
            self.jaw = xf + L.G0 - xm
        else:
            self.jaw = min(self.target, L.JAW_UPPER)

    # -- states --
    def states(self) -> dict:
        q = _quat(self.a[2])
        out = {
            S.A: mk.state_from_poses(self.parts[S.A], (self.a[0], self.a[1], L.TOP, *q), source="offline"),
            S.B: mk.state_from_poses(self.parts[S.B], (self.b[0], self.b[1], L.TOP, 0.0, 0.0, 0.0, 1.0),
                                     source="offline"),
            S.CLAMP: mk.state_from_poses(self.parts[S.CLAMP], (self.c[0], self.c[1], L.TOP, 0.0, 0.0, 0.0, 1.0),
                                         joints={"jaw": self.jaw, "trigger": self.trig, "release": 0.0},
                                         source="offline"),
        }
        return out
