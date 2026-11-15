"""OFFLINE ONLY: a small model of the picture frame for tools/offline/check_task.py, which has no physics.

The simulator never uses this file. Offline, the checker carries a gripped object rigidly and drops it straight down,
and it cannot tip, lean or fall anything; so task.py replaces the frame's state each control step with this model's:

- The frame turns about its bottom back edge (the pivot) by an angle ``theta`` from upright (positive: the top toward
  +x, the wall and the carton). Gravity tips it over that edge past the balance angle (4.8 degrees back) and stands it
  back up below it; it stops (no bounce) on the first support its back meets: the carton (its near face and top),
  the wall, or the table (90 degrees: lying on its back).
- The carton supports it only while the carton spans the frame's middle along y (within 25 mm).
- A gripper whose pads close to the frame's depth across its top rail holds it: it stands upright between the pads,
  its faces centred on the TCP, its bottom on the table or carried with the hand. Opened, it lets go.
- Closed fingers in front of the frame push its face: its angle is at least what keeps the face behind them.

It integrates in 20 substeps per control step. Numbers are design values; nothing here was measured.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

G = 9.81
FULL_GAP = 0.0939
GRIP_GAP = L.FR_T + 0.004  # pads at most this far apart: holding the 18 mm rail
CLOSED_GAP = 0.006  # fingers this nearly closed: a pusher
FINGER_HALF_T = 0.008  # closed fingers: each pad 8 mm thick, so 8 mm either side of the TCP
FINGER_LEN = 0.022  # the pad's flat face along the approach axis, up from the tips
FINGER_HALF_W = 0.0135
SUPPORT_HALF = 0.025  # the carton holds the frame while its span along y reaches within 25 mm of the frame's middle
SUBSTEPS = 20
K2 = (L.FR_H ** 2 + L.FR_T ** 2) / 3.0  # the frame's radius of gyration squared about its bottom edge (a slab)


def rel_pose(theta: float):
    """Rotation of the U frame in the world when tipped ``theta`` about its pivot."""
    return L.ry(theta)


def com_offset(theta: float) -> np.ndarray:
    """The centre of mass from the pivot (x, z) at ``theta``."""
    cx, cz = L.COM_U[0], L.COM_U[2]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([cx * c + cz * s, -cx * s + cz * c])


def front_x(theta: float, z: float) -> float:
    """The glass side's x (from the pivot) at height ``z`` when tipped ``theta`` (0 <= theta < 90 deg)."""
    return -L.FR_T / math.cos(theta) + z * math.tan(theta)


def carton_support(d: float, zb: float, zt: float, depth: float) -> float | None:
    """The angle at which the frame's back first meets a box whose near face is ``d`` ahead of the pivot, spanning
    heights ``zb``..``zt`` and ``depth`` deep. None: it never does before lying flat."""
    if d <= 0.0 or zt <= 0.0:
        return None
    cands = []
    for z in (zt, zb):  # the back face's line through one of the near face's edges
        if z > 0.0 and math.hypot(d, z) <= L.FR_H:
            cands.append(math.atan2(d, z))
    if d <= L.FR_H:  # the top back corner meets the near face
        th = math.asin(d / L.FR_H)
        if zb - 1e-9 <= L.FR_H * math.cos(th) <= zt + 1e-9:
            cands.append(th)
    if 0.0 < zt < L.FR_H:  # the top back corner lands on the box's top
        th = math.acos(zt / L.FR_H)
        if d - 1e-9 <= L.FR_H * math.sin(th) <= d + depth + 1e-9:
            cands.append(th)
    return min(cands) if cands else None


class FrameModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pivot = np.array([L.PIVOT_X, L.FRAME_Y])
        self.bottom = 0.0  # height of the bottom edge (0: on the table)
        self.vz = 0.0
        self.theta = L.LEAN0  # resting on the carton (the 0.5 mm spawn gap closes at once)
        self.omega = 0.0
        self.held_by: str | None = None
        self.dz = 0.0
        self.prev_gap: dict = {}
        self.events: list = []
        self.max_down = 0.0

    # -- world pose ----------------------------------------------------------------------------------------------
    def pose(self):
        """(origin, rotation) of the frame BODY (the declared assembly), z above the table."""
        rot = L.ry(self.theta - L.LEAN0)
        return np.array([self.pivot[0], self.pivot[1], self.bottom]), rot

    def com(self) -> np.ndarray:
        o = com_offset(self.theta)
        return np.array([self.pivot[0] + o[0], self.pivot[1], self.bottom + o[1]])

    def top_mid(self) -> np.ndarray:
        """The middle of the top rail's top face."""
        r = L.ry(self.theta)
        p = r @ np.array([-L.FR_T / 2, 0.0, L.FR_H])
        return np.array([self.pivot[0] + p[0], self.pivot[1], self.bottom + p[2]])

    # -- one control step ----------------------------------------------------------------------------------------
    def step(self, dt: float, carton, arms) -> None:
        """``carton``: (low, high) of the carton's box, z above the table, or None. ``arms``: {side: (tcp, tips,
        opening)} with z above the table, or None."""
        arms = arms or {}
        gaps = {a: float(np.clip(v[2], 0.0, 1.0)) * FULL_GAP for a, v in arms.items()}
        if self.held_by is not None:
            a = self.held_by
            if a not in arms or gaps[a] > GRIP_GAP:
                self.events.append(("let_go", a))
                self.held_by = None
            else:
                self._follow(arms[a][0])
        if self.held_by is None:
            for a, (tcp, tips, _) in arms.items():
                was = self.prev_gap.get(a, FULL_GAP)
                if gaps[a] <= GRIP_GAP < was and self._straddles(tcp, tips):
                    self.held_by = a
                    self.theta, self.omega = 0.0, 0.0
                    self.pivot = np.array([tcp[0] + L.FR_T / 2, tcp[1]])
                    top = max(L.FR_H, tcp[2] + 0.004)
                    self.dz = top - tcp[2]
                    self.bottom = top - L.FR_H
                    self.events.append(("grasp", a))
                    break
        self.prev_gap = gaps
        if self.held_by is None:
            self._free(dt, carton, arms, gaps)

    def _straddles(self, tcp, tips) -> bool:
        t = self.top_mid()
        if abs(tcp[1] - self.pivot[1]) > L.FR_W / 2 - 0.010:
            return False
        if abs(tcp[0] - t[0]) > FULL_GAP / 2 - 0.007:  # its top between the open pads
            return False
        return tips[2] < t[2] - 0.003 and tcp[2] > t[2] - 0.030

    def _follow(self, tcp) -> None:
        self.theta, self.omega, self.vz = 0.0, 0.0, 0.0
        self.pivot = np.array([tcp[0] + L.FR_T / 2, tcp[1]])
        bottom = tcp[2] + self.dz - L.FR_H
        if bottom < 0.0:  # pushed onto the table: it slides up in the grip
            bottom = 0.0
            self.dz = L.FR_H - tcp[2]
        self.bottom = bottom

    def _supports(self, carton) -> float:
        hi = math.pi / 2
        if carton is not None:
            lo, top = carton
            y0, y1 = max(lo[1], self.pivot[1] - L.FR_W / 2), min(top[1], self.pivot[1] + L.FR_W / 2)
            if y0 < y1 and y0 <= self.pivot[1] + SUPPORT_HALF and y1 >= self.pivot[1] - SUPPORT_HALF:
                th = carton_support(lo[0] - self.pivot[0], lo[2] - self.bottom, top[2] - self.bottom, top[0] - lo[0])
                if th is not None:
                    hi = min(hi, th)
        y0, y1 = max(L.WALL_Y[0], self.pivot[1] - L.FR_W / 2), min(L.WALL_Y[1], self.pivot[1] + L.FR_W / 2)
        d = L.WALL_FACE_X - self.pivot[0]
        if y0 < y1 and 0.0 < d < L.FR_H:
            hi = min(hi, math.asin(d / L.FR_H))
        return hi

    def _push_floor(self, arms, gaps) -> float:
        floor = 0.0
        for a, (tcp, tips, _) in arms.items():
            if gaps[a] > CLOSED_GAP or tcp[0] >= self.pivot[0]:
                continue
            if abs(tcp[1] - self.pivot[1]) > L.FR_W / 2 + FINGER_HALF_W:
                continue
            z_lo = tips[2] - self.bottom
            if z_lo > L.FR_H * math.cos(self.theta) or z_lo + FINGER_LEN < 0.0:
                continue
            z = max(z_lo, 0.0)
            need = tcp[0] + FINGER_HALF_T - self.pivot[0]
            if front_x(self.theta, z) >= need:
                floor = max(floor, 0.0)
                continue
            lo_t, hi_t = self.theta, math.radians(89.0)
            if front_x(hi_t, z) < need:
                floor = max(floor, hi_t)
                continue
            for _ in range(50):
                mid = (lo_t + hi_t) / 2
                lo_t, hi_t = (mid, hi_t) if front_x(mid, z) < need else (lo_t, mid)
            floor = max(floor, hi_t)
        return floor

    def _free(self, dt: float, carton, arms, gaps) -> None:
        h = dt / SUBSTEPS
        z0 = self.com()[2]
        if self.bottom > 0.0:  # let go in the air: it drops onto the table
            for _ in range(SUBSTEPS):
                self.vz -= G * h
                self.bottom = max(0.0, self.bottom + self.vz * h)
                if self.bottom == 0.0:
                    self.vz = 0.0
                    break
        floor = self._push_floor(arms, gaps)
        for _ in range(SUBSTEPS):
            ceiling = self._supports(carton)
            acc = G * com_offset(self.theta)[0] / K2
            self.omega += acc * h
            self.theta += self.omega * h
            if self.theta < floor:
                self.theta, self.omega = floor, max(self.omega, 0.0)
            if self.theta <= 0.0:
                self.theta, self.omega = 0.0, 0.0
            if self.theta >= ceiling:
                self.theta, self.omega = ceiling, 0.0
        z1 = self.com()[2]
        self.max_down = max(self.max_down, (z0 - z1) / dt)
