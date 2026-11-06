"""Offline checker only: how far the straightedge pushes the motor. The simulator never calls anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics: it carries a gripped object rigidly with
the gripper and drops it straight down when let go, and nothing pushes anything. So when task.py's hooks get a motor
state whose ``source`` is "offline", the scorer is fed the motor's pose from this small model instead:

- the motor keeps its own pose from step to step; whenever the checker moves the motor's box (a gripper carries it,
  or it drops when let go), the model takes the checker's pose;
- a gripper holds the straightedge when it is at most 30 mm open with its pad-tip point within 15 mm of the bar;
- when the straightedge's face toward +x, at the motor pulley's face centre, comes past that face, the bar pushes the
  motor along x, its face flat on the bar (its heading becomes the bar's):
  - held on both sides of the blower pulley (one gripper's tips beyond its outer rim, y < -0.195, another's beyond
    its inner rim, y > -0.105), the bar lies flat on the blower pulley's face at both rims, so it stops there: the
    motor's face goes to the bar's face, but never past the line x 0.230;
  - held only on the motor's side of the blower's outer rim (y > -0.195), the bar pivots on the blower pulley's rim
    and its far end swings on: the motor's face goes to the bar's face, unclipped (the one-arm overshoot);
  - held only beyond the outer rim (y < -0.195), the bar pivots on that rim and its far end swings back, away from
    the motor: nothing is pushed.
- Nothing else moves the motor: the bar never pulls it back, and open fingers, the bar's ends or links pushing it are
  not modelled (the demo never does that).

No forces, lag, slip, sag, bar pitch or friction: the arms' give and the press force on the GPU decide the real end
state (the README says what to read in the trace). The model resets with rigstate's and mechkit's reset hooks and at
every new offline run.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import scoring as S


def holds(arm, bar) -> bool:
    """A gripper at most 30 mm open with its pad-tip point within 15 mm of the straightedge (in the bar's frame)."""
    try:
        gap = float(np.clip(arm.opening, 0.0, 1.0)) * L.FULL_GAP
        tips = np.asarray(arm.tips, dtype=float)
        if gap > L.HOLD_GAP or not np.all(np.isfinite(tips)):
            return False
        r = S.rotation(bar.quat_xyzw)
        d = r.T @ (tips - np.asarray(bar.pos, dtype=float))
        return bool(np.linalg.norm(np.maximum(np.abs(d) - np.array(L.BAR_HALF), 0.0)) <= L.HOLD_REACH)
    except (AttributeError, TypeError, ValueError):
        return False


def push_mode(arms, bar) -> str | None:
    """How the straightedge is held: "both sides" (it lies flat on the blower pulley), "motor side" (it pivots and
    overshoots), "outboard" (it swings away from the motor) or None (not held)."""
    if not S.arms_valid(arms) or not S.valid(bar):
        return None
    ys = [float(arms[s].tips[1]) for s in S.ARMS if holds(arms[s], bar)]
    if not ys:
        return None
    if any(y < L.OUTER_RIM_Y for y in ys) and any(y > L.INNER_RIM_Y for y in ys):
        return "both sides"
    if any(y >= L.OUTER_RIM_Y for y in ys):
        return "motor side"
    return "outboard"


class MotorModel:
    """The motor's pose as the straightedge would push it, offline."""

    def __init__(self, spec, control_hz: float = L.CONTROL_HZ) -> None:
        self.spec = spec  # the motor's mechkit Assembly (to build its state)
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.origin = None  # the model's motor origin (world) and heading (rad)
        self.yaw = 0.0
        self.seen = None  # the checker's last (origin, heading) for the motor's box
        self.prev_origin = None
        self.pushes = 0  # steps on which the bar pushed the motor
        self.mode = None
        self.state = None

    @staticmethod
    def _pose(m) -> tuple[np.ndarray, float]:
        r, o = S.motor_pose(m)
        return o.copy(), math.atan2(r[1, 0], r[0, 0])

    def update(self, checker_state, bar, arms):
        """Advance one control step; returns the motor's AssemblyState as the model has it."""
        if not S.valid(checker_state):
            return checker_state
        o, yaw = self._pose(checker_state)
        if (self.origin is None or self.seen is None or np.linalg.norm(o - self.seen[0]) > 1e-7
                or abs(math.remainder(yaw - self.seen[1], 2 * math.pi)) > 1e-7):
            self.origin, self.yaw = o, yaw  # the checker carried or dropped it: its pose wins
        self.seen = (o, yaw)
        self.mode = push_mode(arms, bar)
        if self.mode in ("both sides", "motor side"):
            face = self.origin[0] + L.FACE_LOCAL[0] * math.cos(self.yaw) - L.FACE_LOCAL[1] * math.sin(self.yaw)
            face_y = self.origin[1] + L.FACE_LOCAL[0] * math.sin(self.yaw) + L.FACE_LOCAL[1] * math.cos(self.yaw)
            bar_x = S.bar_face_x(bar, face_y)
            lo, hi = np.asarray(bar.low, dtype=float), np.asarray(bar.high, dtype=float)
            fz = self.origin[2] + L.FACE_LOCAL[2]
            level = lo[2] < fz + L.MOTOR_PULLEY_R and hi[2] > fz - L.MOTOR_PULLEY_R  # the bar meets the pulley's face
            if bar_x is not None and level and bar_x > face:
                target = min(bar_x, L.PLANE_X) if self.mode == "both sides" else bar_x
                if target > face:
                    self.yaw = math.radians(S.bar_yaw_deg(bar))  # the face lies flat on the bar
                    self.origin = self.origin.copy()
                    self.origin[0] = target - L.FACE_LOCAL[0] * math.cos(self.yaw) + L.FACE_LOCAL[1] * math.sin(self.yaw)
                    self.pushes += 1
        vel = np.zeros(6)
        if self.prev_origin is not None:
            vel[:3] = (self.origin - self.prev_origin) / self.dt
        self.prev_origin = self.origin.copy()
        quat = (0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2))
        import mechkit as mk  # here, not at the top: it needs yam_arena, which the scorer's own tests do without

        self.state = mk.state_from_poses(self.spec, (*self.origin, *quat), vel, source="offline model")
        return self.state
