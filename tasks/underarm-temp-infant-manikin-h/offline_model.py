"""Offline only (tools/offline/check_task.py, the tests; never in the simulator): a rigid, kinematic stand-in for the
physics, so that the scorer sees the doll's arm, the thermometer and its button move as the demo moves the grippers.

- The doll's arm: a gripper closed under 30 mm with its TCP within 10 mm of the hand's centre (as it is now) holds
  the hand; the arm then turns with the TCP about the shoulder axis (the pinch's closing axis is parallel to it).
  Let go, it falls to the lowest angle at which it clears the thermometer (resting on it) or to its stop at 0; above
  75 deg (where the joint's 0.02 N m friction holds its weight) it stays.
- The thermometer: a gripper closed under 26 mm across its body (TCP within 4 mm of its centre line, over its
  straight part) carries it rigidly. Held more than 8 mm from its centre of mass with nothing under it, it swings
  about the pinch (which holds no torque about its closing axis): the model drops it. It never goes into the mat or
  into the doll: a move that would is cut short where it touches, and the pinch slips. Let go, it lies on the mat.
- The button: a closed gripper's pad tips within 6 mm (top view) of its centre push it in, up to its 0.8 mm travel.
- The display face follows the device rule (scoring.Unit) as in the simulator.
No forces, no friction, no arm give: a predicted 1.0 means the plan's geometry is right, not that the physics works.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk
import parts as P
import rigstate

HOLD_HAND_GAP = 2 * L.HAND_R * L.MM + 0.004
HOLD_HAND_DIST = 0.010
HOLD_TH_GAP = L.TH_W * L.MM + 0.004
PRESS_GAP = 0.006
FRICTION_HOLD = math.radians(75.0)
COM = P.THERMO_SPEC.total_center_of_mass()
CHECK_POINTS = P.TH_POINTS[::3]
REST_GAP = 0.0001  # the arm comes to rest 0.1 mm off the thermometer (resting surfaces read 0 to 0.2 mm on the GPU)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.theta = 0.0
        self.hand_by = None  # (arm, offset TCP - hand)
        self.th_origin = np.asarray(P.TH_ORIGIN, dtype=float).copy()
        self.th_yaw = 0.0
        self.carry = None  # (arm, origin - TCP)
        self.button = 0.0
        self.display = 0.0
        self._support = {}
        self.events: list = []

    # -- states ----------------------------------------------------------------------------------------------------
    def doll(self, theta=None):
        return P.doll_state(self.theta if theta is None else theta)

    def thermo(self, origin=None):
        return P.thermo_state(self.th_origin if origin is None else origin, self.th_yaw, self.button, self.display)

    def states(self) -> dict:
        return {P.DOLL: self.doll(), P.THERMO: self.thermo(), P.MAT: P.MAT_SPEC.rest_state()}

    # -- geometry ----------------------------------------------------------------------------------------------------
    def support(self) -> float:
        key = tuple(np.round(self.th_origin, 4)) + (round(self.th_yaw, 4),)
        if key not in self._support:
            self._support[key] = P.support_angle(self.thermo(), clear=REST_GAP)
        return self._support[key]

    def _blocked(self, origin) -> bool:
        if origin[2] < L.TOP + L.MAT_H - 1e-6:
            return True
        pts = P.thermo_state(origin, self.th_yaw).to_world(CHECK_POINTS)
        return bool(np.min(self.doll().distance(pts)) < 0.0003)

    def _move_thermo(self, target) -> None:
        target = np.asarray(target, dtype=float).copy()
        target[2] = max(target[2], L.TOP + L.MAT_H)
        if not self._blocked(target):
            self.th_origin = target
            return
        lo, hi = 0.0, 1.0
        start = self.th_origin.copy()
        for _ in range(12):
            mid = (lo + hi) / 2
            if self._blocked(start + mid * (target - start)):
                hi = mid
            else:
                lo = mid
        self.th_origin = start + lo * (target - start)

    def _grips_thermo(self, tcp) -> bool:
        loc = self.thermo().to_local(np.asarray(tcp, dtype=float))
        return bool(abs(loc[1]) <= 0.004 and 0.005 <= loc[0] <= (L.TH_BODY_END - 4.0) * L.MM
                    and -0.004 <= loc[2] - 0.013 <= 0.016)

    # -- step ----------------------------------------------------------------------------------------------------------
    def step(self, arms, display_on: bool) -> None:
        read = rigstate.read_arms(arms)
        grippers = {}
        if read is not None:
            for name, a in zip(rigstate.ARMS, read):
                grippers[name] = (np.asarray(a[0], dtype=float), np.asarray(a[1], dtype=float),
                                  float(a[2]) * rigstate.FULL_GAP)
        # the doll's arm
        if self.hand_by is not None:
            arm, off = self.hand_by
            g = grippers.get(arm)
            if g is None or g[2] >= HOLD_HAND_GAP:
                self.hand_by = None
        if self.hand_by is None:
            for arm, (tcp, _tips, gap) in grippers.items():
                if gap < HOLD_HAND_GAP and float(np.linalg.norm(tcp - P.hand_at(self.theta))) <= HOLD_HAND_DIST:
                    self.hand_by = (arm, tcp - P.hand_at(self.theta))
                    self.events.append(("hand gripped", arm, round(math.degrees(self.theta), 1)))
                    break
        if self.hand_by is not None:
            arm, off = self.hand_by
            want = max(0.0, min(math.pi, P.angle_of(grippers[arm][0] - off)))
            if P.thermo_to_arm(self.doll(want), self.thermo()) < 0.0003:
                want = max(want, self.support())
            self.theta = want
        elif self.theta <= FRICTION_HOLD:
            self.theta = self.support()
        # the thermometer
        if self.carry is not None:
            arm, off = self.carry
            g = grippers.get(arm)
            if g is None or g[2] >= HOLD_TH_GAP:
                self.carry = None
                self.events.append(("thermometer let go", arm))
        if self.carry is None:
            for arm, (tcp, _tips, gap) in grippers.items():
                if gap < HOLD_TH_GAP and gap > PRESS_GAP and self._grips_thermo(tcp):
                    self.carry = (arm, self.th_origin - tcp)
                    self.events.append(("thermometer gripped", arm,
                                        round(float(self.thermo().to_local(tcp)[0]) * 1000, 1)))
                    break
        if self.carry is not None:
            arm, off = self.carry
            tcp = grippers[arm][0]
            self._move_thermo(tcp + off)
            self.carry = (arm, self.th_origin - tcp)  # a blocked move slips in the pinch
            grip_x = float(self.thermo().to_local(tcp)[0])
            supported = self.th_origin[2] <= L.TOP + L.MAT_H + 0.003
            if abs(grip_x - COM[0]) > 0.008 and not supported:
                self.carry = None
                self.events.append(("thermometer swung out of the pinch", arm, round(grip_x * 1000, 1)))
        if self.carry is None:
            self.th_origin[2] = L.TOP + L.MAT_H
        # the button
        self.button = 0.0
        if self.carry is None:
            top = self.thermo().to_world((L.BUTTON_X * L.MM, 0.0, (L.TH_H + L.BUTTON_PROUD) * L.MM))
            for _arm, (_tcp, tips, gap) in grippers.items():
                if gap < PRESS_GAP and math.hypot(tips[0] - top[0], tips[1] - top[1]) <= 0.006:
                    self.button = max(self.button, min(L.BUTTON_TRAVEL, max(0.0, float(top[2] - tips[2]))))
        self.display = L.DISPLAY_TRAVEL if display_on else 0.0

    def state_of(self, name):
        return self.states()[name]


def rest_states() -> dict:
    return {P.DOLL: P.doll_state(0.0), P.THERMO: P.thermo_state(P.TH_ORIGIN), P.MAT: P.MAT_SPEC.rest_state()}


__all__ = ["OfflineModel", "rest_states", "mk"]
