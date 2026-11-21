"""Kinematics of the socket, the pack and the rocker for use ONLY where there is no simulator:
tools/offline/check_task.py and the tests. The simulator never takes a result from here: there every part is a PhysX
body or joint moved by the fingers' contacts.

What this file assumes is what the design claims, so the offline prediction is the design's, not a measurement. It
moves the parts from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed). Every gripper points
straight down, as the demo has it: the right one with its fingers closing along world y (socket and pack), the left
one closed and turned a quarter turn, its fingers closing along world x, for the rocker.

- an open gripper whose pad tips come down astride the socket (tips within 8 mm of its centre along x, 4 mm across,
  from 10 mm under to 2 mm over its axis) and then close below its 28 mm length holds it; above 32 mm it is let go
  and drops straight down onto what is under its centre (tray floor, tongue, deck, strip or table);
- the same for the pack: tips within 10 mm of its centre of mass along x, 4 mm across, 10 mm or more inside its top
  and bottom, closing below its 76 mm width; let go above 80 mm, it drops onto the deck (over the charger) or the
  table. While held the pack cannot pass the end stop: its +x face stops at the stop's face (the hand slips);
- a CLOSED gripper (pads under 4 mm apart) is a block 8 mm (along its closing axis) by 27 mm under its pad tips (the
  GPU-measured face is 4 mm from the axis at the tips); where it overlaps a half of the rocker it presses that half
  down until the rocker's top there is under the tips. Past its centre the rocker snaps to that side, before it it
  springs back;
- the right gripper CLOSED and holding nothing is a block 27 mm along x (its fingers' width) by 8 mm along y under
  its pad tips, 38 mm tall. Come down behind the pack's rear face (the block wholly behind it) and moved along +x
  into it while it overlaps the pack's height and width, it pushes the pack along x (on the deck, up to the end
  stop's face; the hand slips there). A closed block that comes down onto or into the pack from above is logged as
  ``("into the pack", ...)`` and moves nothing.
"""

from __future__ import annotations

import numpy as np

import layout as L
import rigstate

CLOSED_GAP = 0.004
FACE = 0.004
HALF_W = 0.0135
STEP_DEG = 0.25
SOCK_GRAB, SOCK_LET_GO = 0.028, 0.032
BAT_GRAB, BAT_LET_GO = 0.076, 0.080
PUSHER = "right"


class OfflineCharger:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rocker_deg = 0.0
        self.sock = np.array([L.SOCK_START[0], L.SOCK_START[1], L.TOP + L.SOCK_START[2]])
        self.bat = np.array([L.BAT_ORIGIN[0], L.BAT_ORIGIN[1], L.TOP + L.BAT_ORIGIN[2]])
        self.held: dict = {}  # part -> (side, part origin minus pad tips)
        self.around = {(p, s): False for p in ("socket", "battery") for s in rigstate.ARMS}
        self.events: list = []
        self.steps = 0
        self.push_prev = None  # the closed right gripper's +x edge and whether it was at the pack's height

    # -- supports -----------------------------------------------------------------------------------------------------
    @staticmethod
    def support_z(x: float, y: float) -> float:
        """The height (over the table) of the highest support under a point."""
        if L.TRAY_IN_X[0] <= x <= L.TRAY_IN_X[1] and L.TRAY_IN_Y[0] <= y <= L.TRAY_IN_Y[1]:
            return L.TRAY_FLOOR
        if L.on_charger_xy((x, y)):
            if L.TONGUE_X[0] <= x <= L.TONGUE_X[1] and abs(y - L.SEAT_YC) <= L.TONGUE_HALF_Y:
                return L.TONGUE_TOP
            if x >= L.STOP_X:
                return L.STOP_TOP
            if abs(y - L.SEAT_YC) >= L.RAIL_IN:
                return L.RAIL_TOP
            return L.DECK
        sx, sy = L.STRIP_XY
        if abs(x - sx) <= L.STRIP[0] / 2 and abs(y - sy) <= L.STRIP[1] / 2:
            return L.STRIP_TOP
        return 0.0

    def _drop(self, part: str) -> None:
        if part == "socket":
            self.sock[2] = L.TOP + self.support_z(self.sock[0], self.sock[1]) + L.SOCK_R
        else:
            base = L.DECK if L.on_charger_xy(self.bat) else 0.0
            self.bat[2] = L.TOP + base + L.BAT[2] / 2

    # -- grips --------------------------------------------------------------------------------------------------------
    def _near(self, part: str, tips) -> bool:
        if part == "socket":
            rel = tips - self.sock
            return abs(rel[0]) <= 0.008 and abs(rel[1]) <= 0.004 and -0.010 <= rel[2] <= 0.002
        com = self.bat + L.bat_com_local()
        rel = tips - com
        z = tips[2] - self.bat[2]
        return abs(rel[0]) <= 0.010 and abs(rel[1]) <= 0.004 and abs(z) <= L.BAT[2] / 2 - 0.010

    def _grips(self, grips) -> None:
        for part, grab, let_go in (("socket", SOCK_GRAB, SOCK_LET_GO), ("battery", BAT_GRAB, BAT_LET_GO)):
            for side, (tips, gap) in grips.items():
                key = (part, side)
                if part in self.held and self.held[part][0] == side:
                    if gap > let_go:
                        del self.held[part]
                        self._drop(part)
                        self.events.append(("let go", part, side, self.steps))
                    continue
                near = self._near(part, tips)
                if part not in self.held and self.around[key] and near and gap < grab:
                    origin = self.sock if part == "socket" else self.bat
                    self.held[part] = (side, origin - tips)
                    self.events.append(("grip", part, side, self.steps))
                if gap > grab + 0.004:
                    self.around[key] = near
                else:
                    self.around[key] = bool(self.around[key] and near)
        for part, (side, off) in self.held.items():
            p = grips[side][0] + off
            if part == "socket":
                self.sock = p
            else:
                if L.on_charger_xy(p) and p[2] - L.BAT[2] / 2 - L.TOP < L.STOP_TOP:  # low enough to meet the stop
                    p[0] = min(p[0], L.STOP_X - L.BAT[0] / 2)
                self.bat = p

    # -- the rocker ---------------------------------------------------------------------------------------------------
    def _rocker(self, blocks) -> None:
        q = self.rocker_deg
        pressed = False
        for tips, gap in blocks:
            hx, hy = FACE + gap / 2, HALF_W  # closing along x
            u0, u1 = tips[0] - hx - L.ROCK_XY[0], tips[0] + hx - L.ROCK_XY[0]
            v0, v1 = tips[1] - hy - L.ROCK_XY[1], tips[1] + hy - L.ROCK_XY[1]
            if v1 <= -L.ROCKER[1] / 2 or v0 >= L.ROCKER[1] / 2:
                continue
            z = float(tips[2]) - L.TOP
            half = L.ROCKER[0] / 2
            off_side = (max(u0, 0.0), min(u1, half))  # the +x "O" half
            on_side = (max(u0, -half), min(u1, 0.0))  # the -x "I" half
            if off_side[1] > off_side[0]:
                while q < L.TRAVEL_DEG and max(L.rocker_top_z(u, q) for u in off_side) > z + 1e-9:
                    q = min(L.TRAVEL_DEG, q + STEP_DEG)
                    pressed = True
            if on_side[1] > on_side[0]:
                while q > 0.0 and max(L.rocker_top_z(u, q) for u in on_side) > z + 1e-9:
                    q = max(0.0, q - STEP_DEG)
                    pressed = True
        if not pressed:
            q = L.TRAVEL_DEG if q >= L.CENTRE_DEG else 0.0
        self.rocker_deg = q

    # -- one control step ---------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        self.steps += 1
        read = rigstate.read_arms(arms)
        if read is None:
            return
        grips = {}
        for side, (_tcp, tips, opening) in zip(rigstate.ARMS, read):
            grips[side] = (np.asarray(tips, dtype=float), float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP)
        self._grips(grips)
        holders = {s for s, _o in self.held.values()}
        self._rocker([g for s, g in grips.items() if g[1] < CLOSED_GAP and s not in holders])
        if PUSHER in grips and PUSHER not in holders and grips[PUSHER][1] < CLOSED_GAP:
            self._push(*grips[PUSHER])
        else:
            self.push_prev = None

    # -- a push on the pack's rear face -------------------------------------------------------------------------------
    def _push(self, tips, gap) -> None:
        """One step of the closed right gripper against the pack. It pushes only if on the step before it was behind
        the rear face (its +x edge at or behind it) at the pack's height; otherwise an overlap is a hand that came
        down onto or into the pack and is logged."""
        hx, hy = rigstate.FINGER_HALF_WIDTH, FACE + gap / 2  # closing along y
        bx, by, bz = self.bat
        lo_z, hi_z = bz - L.BAT[2] / 2, bz + L.BAT[2] / 2
        z0, z1 = float(tips[2]), float(tips[2]) + rigstate.FINGER_LEN
        rear = bx - L.BAT[0] / 2
        x0, x1 = float(tips[0]) - hx, float(tips[0]) + hx
        level = z0 < hi_z and z1 > lo_z and abs(float(tips[1]) - by) < L.BAT[1] / 2 + hy
        prev, self.push_prev = self.push_prev, (x1, level)
        if not level or x1 <= rear or x0 >= bx + L.BAT[0] / 2:
            return
        if prev is None or not prev[1] or prev[0] > rear + 1e-6:
            if not self.events or self.events[-1][0] != "into the pack":
                self.events.append(("into the pack", "battery", PUSHER, self.steps))
            return
        x = x1 + L.BAT[0] / 2
        if L.on_charger_xy((x, by)) and lo_z - L.TOP < L.STOP_TOP:  # low enough to meet the stop
            x = min(x, L.STOP_X - L.BAT[0] / 2)
        if x > bx:
            if not self.events or self.events[-1][0] != "push":
                self.events.append(("push", "battery", PUSHER, self.steps))
            self.bat[0] = x
        self.push_prev = (min(x1, self.bat[0] - L.BAT[0] / 2), level)

    # -- outputs ------------------------------------------------------------------------------------------------------
    def strip_joints(self) -> dict:
        return {"rocker": float(np.radians(self.rocker_deg))}

    def socket_pose(self) -> tuple:
        return (*self.sock, 0.0, 0.0, 0.0, 1.0)

    def battery_pose(self) -> tuple:
        return (*self.bat, 0.0, 0.0, 0.0, 1.0)
