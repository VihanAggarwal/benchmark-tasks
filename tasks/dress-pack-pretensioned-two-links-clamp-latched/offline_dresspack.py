"""The offline stand-in for the physics (tools/offline/check_task.py only; never used in the simulator).

The checker's proxy carries loose parts and never moves a joint, so without a simulator task.py steps this model
from the arms' commanded TCPs, pad tips and openings:

1. The hose. A gripper whose TCP is within 6 mm (in x and z) of the hose's axis, over the hose's length, with its
   pads commanded under 45 mm (the 43 mm crests plus 2), holds it: the hose then follows the TCP's -y travel from
   where it was taken, less the wrist's give under the unit's pull (layout.WRIST_C), between 0 and the end stop.
   Let go (pads past 45 mm), it runs back into the unit: to 0, or, with the protector shut on a waist, until the
   waist's wall meets the rib and the protector meets the exit face (RIB_PLAY + RING_GAP short of where it shut).
2. The protector. It slides along its cradle only: on its own it stays put; shut on the hose it rides with it within
   the rib's play, never past the exit face.
3. The lid. Turned only by pushing: wherever the shut finger pair's box (layout.FINGER_HX either side of the pad-tip
   point along x, FINGER_LEN up from it; across y it must overlap the lid) would cut into the plate, the lid is
   turned shut just far enough to clear it. It cannot go below about 8 deg (the rib on a crest) unless the rib is
   over a waist. Within 2.5 deg of shut it latches and stays shut; unlatched and not pushed, it springs back open.

No friction, inertia or contact forces. Every number is a design value, not a measurement.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk

GAP_PER_OPENING = 0.0939  # pad gap (m) at opening 1.0
RIB_STOP = math.radians(7.7)  # the lid with the rib's underside on a crest (tests: computed)


def _plate_hit(phi: float, tip_xz, hx: float = L.FINGER_HX) -> bool:
    """Whether the finger box (pad-tip point ``tip_xz``, protector frame) cuts into the plate with the lid ``phi``
    open (a separating-axis test of two rectangles in the x-z plane)."""
    ux, uz = L.lid_dir(phi)
    nx, nz = L.lid_normal(phi)
    corners = [(tip_xz[0] + sx * hx, tip_xz[1] + h) for sx in (-1, 1) for h in (0.0, L.FINGER_LEN)]
    pc = (L.HINGE_X + ux * L.LID_LEN / 2, L.HINGE_Z + uz * L.LID_LEN / 2)
    plate = [(pc[0] + a * ux * L.LID_LEN / 2 + b * nx * L.LID_T / 2, pc[1] + a * uz * L.LID_LEN / 2
              + b * nz * L.LID_T / 2) for a in (-1, 1) for b in (-1, 1)]
    for ax in ((1.0, 0.0), (0.0, 1.0), (ux, uz), (nx, nz)):
        p1 = [c[0] * ax[0] + c[1] * ax[1] for c in corners]
        p2 = [c[0] * ax[0] + c[1] * ax[1] for c in plate]
        if max(p1) <= min(p2) + 1e-6 or max(p2) <= min(p1) + 1e-6:
            return False
    return True


class DressPackModel:
    def __init__(self, robot_spec, ring_spec) -> None:
        self.robot_spec, self.ring_spec = robot_spec, ring_spec
        self.reset()

    def reset(self) -> None:
        self.q = 0.0
        self.phi = L.LID_OPEN
        self.ring = 0.0  # the protector's travel toward -y from its start (m; -RING_GAP: on the exit face)
        self.latched = False
        self.qc = None  # the pull that centres the rib's waist with the protector at its start
        self.holder = None
        self.grab = (0.0, 0.0)  # (tcp y, q) when taken
        self.vel = 0.0

    def _gap(self, arm) -> float:
        return float(arm.opening) * GAP_PER_OPENING

    def step(self, arms) -> None:
        prev = self.q
        if arms is None:
            return
        if self.holder is None:
            for a in ("left", "right"):
                arm = arms[a]
                t = np.asarray(arm.tcp, dtype=float) - np.array(L.ORIGIN)
                on = L.HOSE_Y1 - self.q <= t[1] <= L.HOSE_Y0 - self.q
                if (abs(t[0]) <= 0.006 and abs(t[2] - L.HOSE_Z) <= 0.006 and on
                        and self._gap(arm) < 2 * L.HOSE_R + 0.002):
                    self.holder, self.grab = a, (float(t[1]), self.q)
                    break
        if self.holder is not None:
            arm = arms[self.holder]
            if self._gap(arm) >= 2 * L.HOSE_R + 0.002:
                self.holder = None
            else:
                t = np.asarray(arm.tcp, dtype=float) - np.array(L.ORIGIN)
                want = self.grab[1] + (self.grab[0] - float(t[1])) / (1.0 + L.WRIST_C * L.PULL_K)
                self.q = float(np.clip(want, 0.0, L.PULL_MAX))
        if self.latched:
            lo = self.qc - L.RIB_PLAY - L.RING_GAP  # the protector on the exit face, the waist's wall on the rib
            self.q = max(self.q, lo) if self.holder is not None else lo
            d = self.q - self.qc
            self.ring = float(np.clip(self.ring, d - L.RIB_PLAY, d + L.RIB_PLAY))
            self.ring = max(self.ring, -L.RING_GAP)
        elif self.holder is None:
            self.q = 0.0
        self._lid(arms)
        self.vel = (self.q - prev) * L.CONTROL_HZ

    def _lid(self, arms) -> None:
        if self.latched:
            self.phi = 0.0
            return
        ring_y = L.RING_Y - self.ring
        pushed = False
        for a in ("left", "right"):
            arm = arms[a]
            tip = np.asarray(arm.tips, dtype=float) - np.array(L.ORIGIN)
            if abs(tip[1] - ring_y) > L.RING_W / 2 + 0.008 + self._gap(arm) / 2:
                continue
            txz = (float(tip[0]), float(tip[2]))
            if not _plate_hit(min(self.phi + math.radians(1.5), L.LID_OPEN), txz):
                continue  # not touching the lid
            pushed = True
            phi = self.phi
            while phi > 0.0 and _plate_hit(phi, txz):
                phi -= math.radians(0.25)
            self.phi = max(phi, 0.0)
        floor = 0.0 if L.aligned(self.q, ring_y) else RIB_STOP
        self.phi = max(self.phi, floor)
        if self.phi <= L.LATCH_AT:
            w = min(L.waists(self.q), key=lambda y: abs(y - ring_y))
            self.latched, self.phi = True, 0.0
            self.qc = self.q + (w - ring_y) - self.ring
        elif not pushed:
            self.phi = L.LID_OPEN

    def states(self) -> dict:
        import scoring as S
        q0 = (0.0, 0.0, 0.0, 1.0)
        ro = (L.ORIGIN[0], L.ORIGIN[1] - self.ring, L.ORIGIN[2])
        return {
            S.ROBOT: mk.state_from_poses(self.robot_spec, (*L.ORIGIN, *q0), joints={"pull": self.q},
                                         joint_vel={"pull": self.vel}, source="offline"),
            S.RING: mk.state_from_poses(self.ring_spec, (*ro, *q0), joints={"lid": self.phi - L.LID_OPEN},
                                        source="offline"),
        }
