"""Kinematics and statics of the chest, its drawers and the screwdriver for use ONLY where there is no simulator:
tools/offline/check_task.py (whose own carry proxy cannot push or pull a drawer or tip a chest) and the tests. The
simulator never takes a result from here: there the chest is a loose articulation with two prismatic drawers and real
masses, the screwdriver a loose body, and the tip is free-body physics.

What this file assumes is what the design claims, so the offline prediction is the design's, not a measurement. It
moves the parts from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed):

- PUSH: a closed gripper (opening under 6 mm) whose pad tips are at a bar's height (from 15 mm under it to its top),
  within its length (plus 10 mm), and whose block face (10 mm in front of the tips' axis, ``PUSH_FACE``) has reached
  the bar's front face, pushes that drawer in: the bar's front face never stays behind the block's face;
- PULL: a gripper that has straddled a bar open (tips within 6 mm of the bar's middle in x, within its length, from
  12 mm under the bar's underside to its middle) and then closes below 12 mm holds the bar; while held the drawer
  follows the gripper along x, between shut and full travel; it lets go once open wider than 13.5 mm;
- PRESS: a closed gripper whose tips are on the chest's top (within its outline, at most 4 mm above it) presses down
  with 8 N, the robot's limit;
- the SCREWDRIVER is held once open pads have straddled its handle (tips within 8 mm of the handle's axis in x, within
  45 mm along it, from 20 mm under to 10 mm over the axis) and then close below 32 mm; it lets go above 33.5 mm. Held,
  it moves with the gripper. Let go over the right drawer's bay it drops onto the bay's floor; over a drawer's foam it
  lies on the foam; elsewhere on the table. In a drawer it rides with the drawer;
- STATICS: every step, ``layout.moment`` with the two openings, any press and the screwdriver; below zero the chest
  TIPS: it turns forward about its tip line until it rests on an open drawer's front (``layout.rest_angle``) and stays
  there.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

PUSH_FACE = 0.010  # a closed pad pair's face from the gripper's axis along x (the pads' half width; set on the GPU)
PUSH_GAP = 0.006
BAR_GRASP, BAR_RELEASE = L.BAR_SEC, L.BAR_SEC + 0.0015
TOOL_GRASP, TOOL_RELEASE = L.HANDLE_ACROSS, L.HANDLE_ACROSS + 0.0015
PRESS_N = 8.0


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class OfflineChest:
    def __init__(self, chest_spec, tool_spec) -> None:
        self.chest_spec, self.tool_spec = chest_spec, tool_spec
        self.reset()

    def reset(self) -> None:
        import mechkit as mk

        self.open = dict(L.START_OPEN)
        to, tq = self.tool_spec.spawn_pose()
        self.tool_o, self.tool_r = np.array(to, dtype=float), mk.quat_to_matrix(tq)
        self.tool_in: str | None = None  # the drawer the screwdriver rides in
        self.tool_dx = 0.0  # its x in that drawer, relative to the drawer's opening
        self.held_tool: str | None = None
        self.held_bar: dict = {}  # side -> drawer
        self.rel = np.zeros(3)
        self.bar_rel = {}
        self.around = {}
        self.tipped = False
        self.angle = 0.0
        self.events: list = []
        self.margin = L.moment(self.open["left"], self.open["right"])

    # -- geometry (world, the chest level) --------------------------------------------------------------------------
    def bar_front(self, name: str) -> float:
        return L.CHEST_X + L.BAR_FRONT - self.open[name]

    def bar_mid(self, name: str) -> float:
        return L.CHEST_X + L.BAR_CENTRE - self.open[name]

    def _on_bar_line(self, name: str, tips) -> bool:
        return abs(tips[1] - L.DRAWER_Y[name]) <= L.BAR_LEN / 2 + 0.010

    def _straddles_bar(self, name: str, tips) -> bool:
        dz = tips[2] - L.TOP
        return (abs(tips[0] - self.bar_mid(name)) <= 0.006 and abs(tips[1] - L.DRAWER_Y[name]) <= L.BAR_LEN / 2
                and L.BAR_Z - L.BAR_SEC / 2 - 0.012 <= dz <= L.BAR_Z)

    def _straddles_tool(self, tips) -> bool:
        local = self.tool_r.T @ (np.asarray(tips) - self.tool_o)
        return abs(local[0]) <= 0.008 and abs(local[1]) <= 0.045 and -0.020 <= local[2] <= 0.010

    def _push(self, name: str, tips) -> None:
        dz = tips[2] - L.TOP
        if not (self._on_bar_line(name, tips) and L.BAR_Z - L.BAR_SEC / 2 - 0.015 <= dz <= L.BAR_Z + L.BAR_SEC / 2):
            return
        face = tips[0] + PUSH_FACE
        front = self.bar_front(name)
        if front < face <= front + 0.020:
            self.open[name] = float(np.clip(self.open[name] - (face - front), 0.0, L.TRAVEL))

    def _drop_tool(self) -> None:
        """Let go: onto the right drawer's bay floor, a drawer's foam, or the table."""
        corners = L.tool_corners() @ self.tool_r.T + self.tool_o
        z_axis = L.TOP + L.TOOL_R
        self.tool_in = None
        for name in L.DRAWERS:
            q = self.open[name]
            x0 = L.CHEST_X - q
            cy = L.DRAWER_Y[name]
            in_y = np.all(np.abs(corners[:, 1] - cy) <= L.INNER_HALF_Y)
            in_x = np.all((corners[:, 0] >= x0 + L.INNER_X[0]) & (corners[:, 0] <= L.CHEST_X))
            if not (in_y and in_x):
                continue
            bay = (name == "right" and np.all(np.abs(corners[:, 1] - cy) <= L.BAY_HALF_Y)
                   and np.all((corners[:, 0] >= x0 + L.BAY_X[0]) & (corners[:, 0] <= x0 + L.BAY_X[1])))
            z_axis = L.TOP + (L.BALLAST_Z[1] if bay else L.FOAM_Z[1]) + L.TOOL_R
            self.tool_in = name
            self.tool_dx = float(self.tool_o[0] - x0)
            break
        self.tool_o = np.array([self.tool_o[0], self.tool_o[1], z_axis])
        self.events.append(("drop", self.tool_in, [round(float(v), 4) for v in self.tool_o]))

    # -- one control step -------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        grips = {}
        for side, (tcp, tips, o) in zip(rigstate.ARMS, read):
            grips[side] = (np.asarray(tcp, dtype=float), np.asarray(tips, dtype=float),
                           float(np.clip(o, 0.0, 1.0)) * rigstate.FULL_GAP)
        press = []
        for side, (tcp, tips, gap) in grips.items():
            # bars: hold, let go
            if side in self.held_bar:
                if gap > BAR_RELEASE:
                    self.events.append(("let go of bar", self.held_bar.pop(side), side))
                continue
            for name in L.DRAWERS:
                near = self._straddles_bar(name, tips)
                key = ("bar", name, side)
                if near and self.around.get(key) and gap < BAR_GRASP and name not in self.held_bar.values():
                    self.held_bar[side] = name
                    self.bar_rel[side] = self.bar_mid(name) - tcp[0]
                    self.events.append(("grip bar", name, side))
                    break
                if not near:
                    self.around[key] = False
                elif gap > BAR_RELEASE + 0.004:
                    self.around[key] = True
            # the screwdriver
            if self.held_tool == side:
                if gap > TOOL_RELEASE:
                    self.held_tool = None
                    self.events.append(("let go of screwdriver", side))
                    self._drop_tool()
            elif self.held_tool is None and side not in self.held_bar:
                near = self._straddles_tool(tips)
                key = ("tool", side)
                if near and self.around.get(key) and gap < TOOL_GRASP:
                    self.held_tool = side
                    self.tool_in = None
                    self.rel = self.tool_o - tcp
                    self.events.append(("grip screwdriver", side))
                elif not near:
                    self.around[key] = False
                elif gap > TOOL_RELEASE + 0.004:
                    self.around[key] = True
            if gap < PUSH_GAP and side not in self.held_bar:
                for name in L.DRAWERS:
                    self._push(name, tips)
                on_top = (L.CHEST_X <= tips[0] <= L.CHEST_X + L.CHEST_D and abs(tips[1]) <= L.CHEST_W / 2
                          and tips[2] - L.TOP <= L.CHEST_H + 0.004)
                if on_top:
                    press.append(tips[0] - L.CHEST_X)
        for side, name in self.held_bar.items():
            tcp = grips[side][0]
            mid = tcp[0] + self.bar_rel[side]
            self.open[name] = float(np.clip(L.CHEST_X + L.BAR_CENTRE - mid, 0.0, L.TRAVEL))
        if self.held_tool is not None:
            self.tool_o = grips[self.held_tool][0] + self.rel
        elif self.tool_in is not None:
            self.tool_o = np.array([L.CHEST_X - self.open[self.tool_in] + self.tool_dx, self.tool_o[1], self.tool_o[2]])
        tool_x = None if self.tool_in is None else float(self.tool_o[0] - L.CHEST_X)
        m = L.moment(self.open["left"], self.open["right"], tool_x=tool_x)
        for px in press:
            m += PRESS_N / L.G * (px - L.TIP_X)
        self.margin = m
        if not self.tipped and m < 0.0:
            self.tipped = True
            self.angle = L.rest_angle(self.open["left"], self.open["right"])
            self.events.append(("TIP", {n: round(v, 4) for n, v in self.open.items()}, round(m, 4)))

    # -- the states the scorer sees ---------------------------------------------------------------------------------
    def states(self, objects: dict, chest_name: str, tool_name: str) -> dict:
        import mechkit as mk

        origin = np.array([L.CHEST_X, 0.0, L.TOP])
        rot = np.eye(3)
        tool_o, tool_r = self.tool_o.copy(), self.tool_r.copy()
        if self.tipped:  # forward about the tip line: the back rises
            rot = ry(-self.angle)
            pivot = np.array([L.CHEST_X + L.TIP_X, 0.0, L.TOP])
            origin = pivot + rot @ (origin - pivot)
            if self.tool_in is not None and self.held_tool is None:
                tool_o = pivot + rot @ (tool_o - pivot)
                tool_r = rot @ tool_r
        joints = {"left": self.open["left"] - L.TRAVEL, "right": self.open["right"]}
        out = dict(objects)
        out[chest_name] = mk.state_from_poses(self.chest_spec, (*origin, *mk.matrix_to_quat(rot)), np.zeros(6),
                                              joints=joints, source="offline")
        out[tool_name] = mk.state_from_poses(self.tool_spec, (*tool_o, *mk.matrix_to_quat(tool_r)), np.zeros(6),
                                             source="offline")
        return out
