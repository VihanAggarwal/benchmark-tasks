"""Kinematics of the screwdriver, the cover, the catches and the rocker for use ONLY where there is no simulator:
tools/offline/check_task.py and the tests. The simulator never takes a result from here: there every part is a PhysX
body or joint moved by the fingers' contacts (the catch levers too), and catchrule.py reads the levers' joints.

What this file assumes is what the design claims, so the offline prediction is the design's, not a measurement. It
moves the parts from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed). Every gripper points
straight down, as the demo has it: the right one with its fingers closing along world y while it holds the tool and
along world x when it is closed for the rocker; the left one closed, its fingers closing along y.

- a CLOSED gripper (pads under 4 mm apart) is a wedge, as the GPU measured the fingers' convex hulls on earlier
  tasks: along the closing direction its face is 4 mm from the axis at the pad tips and widens 0.183 mm per mm of
  height, up to 60 mm over the tips; across it, 13.5 mm each side (half a pad's width). Where the wedge overlaps the
  cover, the cover is turned the least that clears it, the way the wedge is moving (offline the push is made only by
  the left wedge moving toward -y, so it turns the cover shut);
- with nothing on it the cover falls shut once it is past its balance (91.2 deg open) and back onto its stop before;
- the same wedge's bottom (the pad tips, 8 mm by 27 mm) over the rocker's ON half presses it down to clear; past its
  centre the rocker snaps ON, before it it springs back OFF;
- an open gripper whose pads straddle the handle (tips within the handle's length, 8 mm of its axis across, between
  12 mm under and 5 mm over its axis) and then close below its 28 mm holds the tool; the tool follows the pad tips;
  above 32 mm it is let go and drops straight down onto what is under its centre of mass (tray floor, terminal row,
  mounting plate, the supply's top or the table);
- the same wedge's bottom over a catch lever's tab presses that lever down to clear it (each lever on its own: it
  needs the wedge over its own tab); past its centre it snaps closed while the cover is shut, before it, or with the
  cover open 5 degrees or more, it springs back open; catchrule.Catch reads the two levers.
"""

from __future__ import annotations

import math

import numpy as np

import catchrule
import layout as L
import rigstate

CLOSED_GAP = 0.004
FACE = 0.004
SLANT = 0.183
WEDGE_H = 0.060
HALF_W = 0.0135
CLOSE_AXIS = {"left": 1, "right": 0}  # world axis the closed fingers close along (the demo's yaw)
FALL_DEG = 6.0  # per control step
STEP_DEG = 0.25
GRAB_GAP = 0.028
LET_GO_GAP = 0.032
PLATE_PTS_STEP = 0.008


class OfflineBox:
    def __init__(self) -> None:
        self.catch = catchrule.Catch()
        self.plate_local = None
        self.reset()

    def reset(self) -> None:
        self.open_deg = L.COVER_OPEN_DEG
        self.rocker_deg = 0.0
        self.lever_deg = [0.0] * len(L.LEV_NAMES)
        self.tool = np.array([L.TOOL_ORIGIN[0], L.TOOL_ORIGIN[1], L.TOP + L.TOOL_ORIGIN[2]])
        self.held = None  # (side, tool origin minus pad tips)
        self.around = {s: False for s in rigstate.ARMS}
        self.prev_tips: dict = {}
        self.events: list = []
        self.catch.reset()

    # -- geometry ---------------------------------------------------------------------------------------------------
    @staticmethod
    def in_wedge(pts, tips, axis: int, gap: float) -> np.ndarray:
        """Which points (world) lie inside a closed gripper's wedge."""
        d = np.asarray(pts, dtype=float) - np.asarray(tips, dtype=float)
        dz = d[:, 2]
        across = 1 - axis
        half = FACE + SLANT * np.clip(dz, 0.0, None) + gap / 2
        return (dz >= 0.0) & (dz <= WEDGE_H) & (np.abs(d[:, axis]) <= half) & (np.abs(d[:, across]) <= HALF_W)

    def plate(self, open_deg: float) -> np.ndarray:
        p = L.cover_plate_points(open_deg, PLATE_PTS_STEP)
        return p + np.array([0.0, 0.0, L.TOP])

    def _cover_hit(self, open_deg, wedges) -> bool:
        pts = self.plate(open_deg)
        return any(bool(np.any(self.in_wedge(pts, t, a, g))) for t, a, g in wedges)

    # -- the tool ---------------------------------------------------------------------------------------------------
    def _tool_grip(self, grips) -> None:
        for side, (tips, gap) in grips.items():
            if self.held is not None and self.held[0] == side:
                if gap > LET_GO_GAP:
                    self.held = None
                    self._drop()
                    self.events.append(("let go", side))
                continue
            rel = tips - self.tool
            hx0, hx1 = L.HANDLE_X[0] - L.TOOL_ORIGIN[0], L.HANDLE_X[1] - L.TOOL_ORIGIN[0]
            near = (hx0 + HALF_W <= rel[0] <= hx1 - HALF_W and abs(rel[1]) <= 0.008
                    and -0.012 <= rel[2] <= 0.005)
            if self.held is None and self.around[side] and near and gap < GRAB_GAP:
                self.held = (side, self.tool - tips)
                self.events.append(("grip", side))
            if gap > GRAB_GAP + 0.004:
                self.around[side] = near
            else:
                self.around[side] = bool(self.around[side] and near)
        if self.held is not None:
            side, off = self.held
            self.tool = grips[side][0] + off

    def support_z(self, x: float, y: float) -> float:
        """The height (over the table) of the highest support under a point."""
        if L.TRAY_IN_X[0] <= x <= L.TRAY_IN_X[1] and L.TRAY_IN_Y[0] <= y <= L.TRAY_IN_Y[1]:
            return L.TRAY_FLOOR
        if L.IN_X[0] <= x <= L.IN_X[1] and L.IN_Y[0] <= y <= L.IN_Y[1]:
            if L.ROW_X[0] <= x <= L.ROW_X[1] and L.ROW_Y[0] <= y <= L.ROW_Y[1]:
                return L.ROW_TOP
            return L.PLATE_TOP
        sx, sy = L.SUP_XY
        if abs(x - sx) <= L.SUP_SIZE[0] / 2 and abs(y - sy) <= L.SUP_SIZE[1] / 2:
            return L.SUP_TOP
        return 0.0

    def _drop(self) -> None:
        com = self.tool[0] - L.TOOL_ORIGIN[0] + L.tool_com_x()
        self.tool[2] = L.TOP + self.support_z(com, self.tool[1]) + L.HANDLE_R

    # -- one control step -------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            self.catch.update(self.open_deg, None)
            return
        grips = {}
        for side, (_tcp, tips, opening) in zip(rigstate.ARMS, read):
            grips[side] = (tips, float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP)
        self._tool_grip(grips)
        holder = None if self.held is None else self.held[0]
        wedges, motions = [], []
        for side, (tips, gap) in grips.items():
            prev = self.prev_tips.get(side)
            self.prev_tips[side] = tips.copy()
            if gap < CLOSED_GAP and side != holder:
                wedges.append((tips, CLOSE_AXIS[side], gap))
                motions.append(np.zeros(3) if prev is None else tips - prev)
        self._cover(wedges, motions)
        self._rocker(wedges)
        self._levers(wedges)
        self.catch.update(self.open_deg, self.lever_deg)

    def _cover(self, wedges, motions) -> None:
        if wedges and self._cover_hit(self.open_deg, wedges):
            closing = any(v[1] < -1e-6 for v in motions)  # a wedge moving toward -y turns it shut
            if closing:
                a = self.open_deg
                while a > 0.0 and self._cover_hit(a, wedges):
                    a = max(0.0, a - STEP_DEG)
                if not self._cover_hit(a, wedges):
                    self.open_deg = a
            return
        rest = 0.0 if self.open_deg < L.cover_balance_deg() else L.COVER_OPEN_DEG
        d = float(np.clip(rest - self.open_deg, -FALL_DEG, FALL_DEG))
        if d and not (wedges and self._cover_hit(self.open_deg + d, wedges)):
            self.open_deg += d

    def _rocker(self, wedges) -> None:
        q_min = 0.0
        sx, sy = L.SUP_XY
        for tips, axis, gap in wedges:
            half = (FACE + gap / 2, HALF_W) if axis == 0 else (HALF_W, FACE + gap / 2)
            u0, u1 = tips[0] - half[0] - sx, tips[0] + half[0] - sx
            v0, v1 = tips[1] - half[1] - sy, tips[1] + half[1] - sy
            u0, u1 = max(u0, 0.0), min(u1, L.ROCKER[0] / 2)  # the ON half only
            if u1 <= u0 or v1 <= -L.ROCKER[1] / 2 or v0 >= L.ROCKER[1] / 2:
                continue
            z = float(tips[2]) - L.TOP
            q = self.rocker_deg
            while q < L.ROCKER_TRAVEL_DEG and max(L.rocker_top_z(u, q) for u in (u0, u1)) > z + 1e-9:
                q = min(L.ROCKER_TRAVEL_DEG, q + STEP_DEG)
            q_min = max(q_min, q)
        free = L.ROCKER_TRAVEL_DEG if self.rocker_deg >= L.ROCKER_CENTRE_DEG else 0.0
        self.rocker_deg = max(q_min, free)  # past the centre it snaps ON on the next step

    def _levers(self, wedges) -> None:
        for i, xc in enumerate(L.CATCH_XS):
            q_min = 0.0
            for tips, axis, gap in wedges:
                half = (FACE + gap / 2, HALF_W) if axis == 0 else (HALF_W, FACE + gap / 2)
                if tips[0] + half[0] < xc - L.LEV_W / 2 or tips[0] - half[0] > xc + L.LEV_W / 2:
                    continue
                u0 = max(L.LEV_PIVOT_Y - (tips[1] + half[1]), 0.0)  # outward from the hinge
                u1 = min(L.LEV_PIVOT_Y - (tips[1] - half[1]), L.LEV_LEN)
                if u1 <= u0:
                    continue
                z = float(tips[2]) - L.TOP
                q = self.lever_deg[i]
                while q < L.LEV_TRAVEL_DEG and max(L.lever_top_z(u, q) for u in (u0, u1)) > z + 1e-9:
                    q = min(L.LEV_TRAVEL_DEG, q + STEP_DEG)
                q_min = max(q_min, q)
            hooked = self.open_deg < L.UNLATCH_DEG - L.EPS
            free = L.LEV_TRAVEL_DEG if (self.lever_deg[i] >= L.LEV_CENTRE_DEG and hooked) else 0.0
            self.lever_deg[i] = max(q_min, free)
            if not hooked and q_min < L.LEV_CENTRE_DEG:
                self.lever_deg[i] = q_min

    # -- outputs ----------------------------------------------------------------------------------------------------
    def box_joints(self) -> dict:
        out = {"cover": math.radians(L.COVER_OPEN_DEG - self.open_deg)}
        out.update({n: math.radians(d) for n, d in zip(L.LEV_NAMES, self.lever_deg)})
        return out

    def supply_joints(self) -> dict:
        return {"rocker": math.radians(self.rocker_deg)}

    def tool_pose(self) -> tuple:
        return (*self.tool, 0.0, 0.0, 0.0, 1.0)
