"""Kinematics of the front wire guard, top retaining clip and rear rocker switch for use ONLY where there is no
simulator: tools/offline/check_task.py and unit tests. The simulator never takes a result from here: in PhysX every
part is a real body or joint moved by the grippers' contacts, and latchrule.py drives the springs.

What this file models:
- A CLOSED gripper (pads under 4 mm apart) is represented as a finger wedge (4 mm half-thickness along its closing
  axis at the tips, widening 0.183 mm per mm of height up to 60 mm; 13.5 mm half-width across).
- Front wire guard: while unlatched, its seating spring returns it toward 22 deg open whenever no wedge holds it shut
  (up to 5 deg per step). While a wedge overlaps the guard's outer face, the guard turns to the smallest open angle
  that clears the wedge (down to 0 deg). Once latched by the top clip, it stays held flush at 0 deg.
- Top retaining clip: a wedge over the clip's top surface presses it down to clear the pad tips; past its 17.5 deg
  centre it snaps closed (35 deg) ONLY while the front guard is seated within 3 deg of closed; otherwise it springs
  back open to 0 deg.
- Rear rocker switch: a wedge over the rocker's ON half (+x from pivot) presses it down to clear the pad tips; past
  its 12 deg centre it snaps ON (24 deg); before it, it springs back OFF (0 deg).
"""

from __future__ import annotations

import math

import numpy as np

import latchrule
import layout as L
import rigstate

CLOSED_GAP = 0.004
FACE = 0.004
SLANT = 0.183
WEDGE_H = 0.060
HALF_W = 0.0135
CLOSE_AXIS = {"left": 1, "right": 0}  # world axis the closed fingers close along
SPRING_DEG = 5.0  # guard spring return rate per control step when unlatched
STEP_DEG = 0.25


class OfflineFan:
    def __init__(self) -> None:
        self.latch = latchrule.Latch()
        self.reset()

    def reset(self) -> None:
        self.open_deg = L.GUARD_OPEN_DEG
        self.clip_deg = 0.0  # positive closed travel [0..35]
        self.rocker_deg = 0.0
        self.events: list = []
        self.latch.reset()

    @staticmethod
    def in_wedge(pts, tips, axis: int, gap: float) -> np.ndarray:
        """Which world points lie inside a closed gripper's wedge."""
        d = np.asarray(pts, dtype=float) - np.asarray(tips, dtype=float)
        dz = d[:, 2]
        across = 1 - axis
        half = FACE + SLANT * np.clip(dz, 0.0, None) + gap / 2.0
        return (dz >= 0.0) & (dz <= WEDGE_H) & (np.abs(d[:, axis]) <= half) & (np.abs(d[:, across]) <= HALF_W)

    def _guard_hit(self, open_deg: float, wedges) -> bool:
        pts = L.guard_face_points(open_deg) + np.array([0.0, 0.0, L.TOP])
        return any(bool(np.any(self.in_wedge(pts, t, a, g))) for t, a, g in wedges)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            self.latch.update(self.open_deg, None)
            return
        wedges = []
        for side, (_tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            if gap < CLOSED_GAP:
                wedges.append((tips, CLOSE_AXIS[side], gap))
        self._guard(wedges)
        self._clip(wedges)
        self._rocker(wedges)
        was = self.latch.latched
        self.latch.update(self.open_deg, self.clip_deg)
        if self.latch.latched != was:
            self.events.append(("latched" if self.latch.latched else "undone", round(self.open_deg, 2)))

    def _guard(self, wedges) -> None:
        # Check if any wedge is pushing the guard toward closed (into -y)
        if wedges and self._guard_hit(self.open_deg, wedges):
            a = self.open_deg
            while a > 0.0 and self._guard_hit(a, wedges):
                a = max(0.0, a - STEP_DEG)
            self.open_deg = a
            return
        if self.latch.latched:
            self.open_deg = 0.0
            return
        # Unlatched seating spring pushes the guard back open toward GUARD_OPEN_DEG unless blocked by a wedge
        target = min(L.GUARD_OPEN_DEG, self.open_deg + SPRING_DEG)
        a = self.open_deg
        while a < target - 1e-9:
            nxt = min(target, a + STEP_DEG)
            if wedges and self._guard_hit(nxt, wedges):
                break
            a = nxt
        self.open_deg = a

    def _clip(self, wedges) -> None:
        q_min = 0.0
        for tips, axis, gap in wedges:
            half = (FACE + gap / 2.0, HALF_W) if axis == 0 else (HALF_W, FACE + gap / 2.0)
            if tips[0] + half[0] < L.CLIP_X - L.CLIP_W / 2.0 or tips[0] - half[0] > L.CLIP_X + L.CLIP_W / 2.0:
                continue
            u0 = max(L.CLIP_PIVOT_Y - (tips[1] + half[1]), 0.0)
            u1 = min(L.CLIP_PIVOT_Y - (tips[1] - half[1]), L.CLIP_LEN)
            if u1 <= u0:
                continue
            z = float(tips[2]) - L.TOP
            q = self.clip_deg
            while q < L.CLIP_TRAVEL_DEG and max(L.clip_top_z(u, q) for u in (u0, u1)) > z + 1e-9:
                q = min(L.CLIP_TRAVEL_DEG, q + STEP_DEG)
            q_min = max(q_min, q)
        hooked = self.open_deg <= L.LATCH_SEAT_DEG + L.EPS
        free = L.CLIP_TRAVEL_DEG if (self.clip_deg >= L.CLIP_CENTRE_DEG and hooked) else 0.0
        self.clip_deg = max(q_min, free)
        if not hooked and q_min < L.CLIP_CENTRE_DEG:
            self.clip_deg = q_min

    def _rocker(self, wedges) -> None:
        q_min = 0.0
        rx, ry = L.ROCKER_XY
        for tips, axis, gap in wedges:
            half = (FACE + gap / 2.0, HALF_W) if axis == 0 else (HALF_W, FACE + gap / 2.0)
            u0 = max(tips[0] - half[0] - rx, 0.0)
            u1 = min(tips[0] + half[0] - rx, L.ROCKER[0] / 2.0)
            v0, v1 = tips[1] - half[1] - ry, tips[1] + half[1] - ry
            if u1 <= u0 or v1 <= -L.ROCKER[1] / 2.0 or v0 >= L.ROCKER[1] / 2.0:
                continue
            z = float(tips[2]) - L.TOP
            q = self.rocker_deg
            while q < L.ROCKER_TRAVEL_DEG and max(L.rocker_top_z(u, q) for u in (u0, u1)) > z + 1e-9:
                q = min(L.ROCKER_TRAVEL_DEG, q + STEP_DEG)
            q_min = max(q_min, q)
        free = L.ROCKER_TRAVEL_DEG if self.rocker_deg >= L.ROCKER_CENTRE_DEG else 0.0
        self.rocker_deg = max(q_min, free)

    def fan_joints(self) -> dict:
        return {
            "guard": math.radians(L.GUARD_OPEN_DEG - self.open_deg),
            "clip": math.radians(self.clip_deg),
            "rocker": math.radians(self.rocker_deg),
        }
