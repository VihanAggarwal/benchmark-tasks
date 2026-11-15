"""Kinematics of the iron, its cord and the board for use ONLY where there is no simulator: tools/offline/check_task.py
(whose carry proxy cannot tilt a part, slide it into an inclined cage or drag a cord) and the tests. The simulator
never takes a result from here: there the iron, cord, board and holder are free bodies.

What this file assumes is what the design claims, so the offline prediction is the design's, not a measurement. It
moves the parts from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed) and each gripper's
commanded rotation (read from a shadow of the demo plan, because rigstate carries no wrist angle):

- a gripper holds the iron once its pad tips have straddled one of its sections open (on the section, within 12 mm
  of the axis) and then close below that section's diameter; it lets go once open 1.5 mm wider. Held, the iron moves
  rigidly with the gripper (turning with it);
- let go with its metal front in the cage and coaxial within 15 deg, the iron slides down the cage onto its stop
  (the grip's front on the step) and lies on the cage's lower side (axis 2 mm low); let go anywhere else it lies
  flat (axis level, heading kept) on the mat or the bench with its grip under the gripper;
- a closed gripper (pads one block, under 4 mm open) beside the grip at its height pushes it along -x (the
  push-aside variant), never pulls it;
- the board is held once the pads straddle its 60 mm width (pad tips within 6 mm of its centre, under the plate)
  and close below 60 mm; let go above 61.5 mm it drops straight down onto the mat or the bench, or onto the lying
  iron if that is under it, a foot raised 5 mm if the cord is under it;
- the cord is a rope: laid as it was from the station up to the last point from which the rest of its length still
  reaches the iron's rear, then straight to the rear (it never stretches; slack is not drawn).
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

PUSH_GAP = 0.004
PAD_BLOCK_HALF = 0.0085  # two 8 mm pads and the 1 mm commanded gap, along the closing axis


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


class OfflineRig:
    def __init__(self, iron_spec, board_spec, holder_spec, shadow=None) -> None:
        self.iron_spec, self.board_spec, self.holder_spec = iron_spec, board_spec, holder_spec
        self.shadow = shadow  # callable(side, t) -> 3x3 TCP rotation or None
        self.laid = np.asarray(iron_spec.ends, dtype=float)
        self.reset()

    def reset(self) -> None:
        tip, r = self.iron_spec.tool_frame()
        self.tip, self.r = np.asarray(tip, dtype=float).copy(), np.asarray(r, dtype=float).copy()
        o, _ = self.board_spec.spawn_pose()
        self.board_o, self.board_yaw = np.asarray(o, dtype=float).copy(), float(self.board_spec.yaw)
        self.held = {"iron": None, "board": None}
        self.around = {(o_, s): False for o_ in ("iron", "board") for s in rigstate.ARMS}
        self.width = {"iron": 0.0}
        self.rel: dict = {}
        self.events: list = []
        self.t = 0.0
        self.cord_points = np.asarray([c for c, _ in self.iron_spec.segment_frames()])

    # -- helpers ----------------------------------------------------------------------------------------------
    @property
    def axis(self) -> np.ndarray:
        return self.r[:, 0]

    def _rot(self, side: str) -> np.ndarray:
        if self.shadow is not None:
            r = self.shadow(side, self.t)
            if r is not None:
                return np.asarray(r, dtype=float)
        return np.diag([1.0, -1.0, -1.0])  # straight down

    def _section_at(self, tips):
        p = self.r.T @ (np.asarray(tips) - self.tip)
        s, radial = float(p[0]), math.hypot(p[1], p[2])
        if radial > 0.012:
            return None
        for name, s0, s1, r, _ in L.IRON_SECTIONS:
            if s0 + 0.005 <= s <= s1 - 0.005:
                return name, 2 * r
        return None

    def _board_near(self, tips) -> bool:
        c = self.board_o
        plate_bottom = c[2] + L.STANDOFF_H
        return (math.hypot(tips[0] - c[0], tips[1] - c[1]) <= 0.006 and c[2] - 0.001 <= tips[2] <= plate_bottom)

    def _seat_if_in_cage(self) -> bool:
        a = L.CAGE_AXIS
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(-self.axis @ a)))))
        mouth = np.array(L.world(L.MOUTH))
        ok = ang <= 15.0
        for s in (0.0, L.METAL_END):
            rel = self.tip + s * self.axis - mouth
            u = float(rel @ a)
            ok = ok and -0.010 <= u <= L.CAGE_LEN and np.linalg.norm(rel - u * a) <= L.MOUTH_R_IN
        if not ok:
            return False
        self.tip = mouth + (L.STEP_U + L.METAL_END) * a + 0.002 * L.CAGE_DOWN
        x = -a
        y = np.array([1.0, 0.0, 0.0])
        self.r = np.column_stack([x, y, np.cross(x, y)])
        self.events.append(("seated", round(self.t, 2)))
        return True

    def _lay_down(self, grip_xy) -> None:
        x = self.axis.copy()
        x[2] = 0.0
        x = x / np.linalg.norm(x)
        y = np.cross([0.0, 0.0, 1.0], x)
        self.r = np.column_stack([x, y, np.cross(x, y)])
        p = self.r.T @ (np.array([grip_xy[0], grip_xy[1], self.tip[2]]) - self.tip)
        s_hold = min(max(float(p[0]), 0.0), L.IRON_LEN)
        support = L.MAT_TOP if L.over_mat(grip_xy[0], grip_xy[1]) else 0.0
        hold = np.array([grip_xy[0], grip_xy[1], L.TOP + support + L.GRIP_R])
        self.tip = hold - s_hold * x
        self.events.append(("laid", round(self.t, 2), "mat" if support else "bench"))

    def _push(self, tcp, tips) -> None:
        ax = self.axis
        if abs(ax[2]) > 0.2 or abs(ax[0]) > 0.2:
            return
        p = self.r.T @ (np.asarray(tips) - self.tip)
        if not (L.GRIP_S[0] <= p[0] <= L.GRIP_S[1]):
            return
        axis_z = self.tip[2]
        if not (axis_z - 0.012 <= tips[2] <= axis_z + 0.005):
            return
        axis_x = self.tip[0] + p[0] * ax[0]
        face = tcp[0] - PAD_BLOCK_HALF
        if tcp[0] > axis_x and face < axis_x + L.GRIP_R:
            self.tip = self.tip + np.array([face - (axis_x + L.GRIP_R), 0.0, 0.0])

    def _drop_board(self) -> None:
        c = self.board_o
        rz = np.array([[math.cos(self.board_yaw), -math.sin(self.board_yaw)],
                       [math.sin(self.board_yaw), math.cos(self.board_yaw)]])
        feet = [c[:2] + rz @ np.array(f) for f in L.STANDOFF_XY]
        z = max((L.MAT_TOP if L.over_mat(f[0], f[1]) else 0.0) for f in feet)
        half = np.array([L.BOARD_SIZE[0], L.BOARD_SIZE[1]]) / 2 + L.GRIP_R
        for s in np.linspace(0.0, L.IRON_LEN, 44):
            q = self.tip + s * self.axis
            loc = rz.T @ (q[:2] - c[:2])
            if np.all(np.abs(loc) <= half):
                z = max(z, q[2] - L.TOP + L.GRIP_R - L.STANDOFF_H)  # the plate rests on the iron
        for f in feet:
            d = np.linalg.norm(self.cord_points[:, :2] - f, axis=1)
            if np.any(d <= L.STANDOFF_R + L.CORD_R):
                z = max(z, (L.MAT_TOP if L.over_mat(f[0], f[1]) else 0.0) + 2 * L.CORD_R)
        self.board_o = np.array([c[0], c[1], L.TOP + z])

    def _rope(self) -> None:
        rear = self.tip + L.IRON_LEN * self.axis
        e, n, seg = self.laid, len(self.laid) - 1, self.iron_spec.seg_len
        k = n
        while k > 0 and (n - k) * seg < np.linalg.norm(e[k] - rear):
            k -= 1
        straight = np.linspace(e[k], rear, n - k + 1)
        ends = np.vstack([e[:k + 1], straight[1:]])
        self.cord_points = (ends[:-1] + ends[1:]) / 2

    # -- one control step -------------------------------------------------------------------------------------
    def step(self, arms, t: float) -> None:
        self.t = t
        read = rigstate.read_arms(arms)
        if read is None:
            return
        grips = {side: (np.asarray(tcp, float), np.asarray(tips, float),
                        float(np.clip(op, 0.0, 1.0)) * rigstate.FULL_GAP)
                 for side, (tcp, tips, op) in zip(rigstate.ARMS, read)}
        # the iron
        for side, (tcp, tips, gap) in grips.items():
            if self.held["iron"] == side:
                if gap > self.width["iron"] + 0.0015:
                    self.held["iron"] = None
                    self.events.append(("release", "iron", side, round(t, 2)))
                    if not self._seat_if_in_cage():
                        self._lay_down(tcp[:2])
                continue
            sec = self._section_at(tips)
            if sec is None:
                self.around[("iron", side)] = False
                continue
            name, width = sec
            if gap > width + 0.002:
                self.around[("iron", side)] = True
            elif self.around[("iron", side)] and gap < width and self.held["iron"] is None:
                rt = self._rot(side)
                self.held["iron"] = side
                self.width["iron"] = width
                self.rel["iron"] = (rt.T @ (self.tip - tcp), rt.T @ self.r)
                self.events.append(("grasp", "iron", side, name, round(t, 2)))
        # the board
        for side, (tcp, tips, gap) in grips.items():
            if self.held["board"] == side:
                if gap > L.BOARD_SIZE[0] + 0.0015:
                    self.held["board"] = None
                    self.events.append(("release", "board", side, round(t, 2)))
                    self._drop_board()
                continue
            if not self._board_near(tips):
                self.around[("board", side)] = False
                continue
            if gap > L.BOARD_SIZE[0] + 0.002:
                self.around[("board", side)] = True
            elif self.around[("board", side)] and gap < L.BOARD_SIZE[0] and self.held["board"] is None:
                self.held["board"] = side
                self.rel["board"] = self.board_o - tcp
                self.events.append(("grasp", "board", side, round(t, 2)))
        if self.held["iron"] is None:
            for side, (tcp, tips, gap) in grips.items():
                if gap < PUSH_GAP:
                    self._push(tcp, tips)
        side = self.held["iron"]
        if side is not None:
            tcp = grips[side][0]
            rt = self._rot(side)
            p, rr = self.rel["iron"]
            self.tip, self.r = tcp + rt @ p, rt @ rr
        side = self.held["board"]
        if side is not None:
            self.board_o = grips[side][0] + self.rel["board"]
        self._rope()

    # -- the states the scorer sees ---------------------------------------------------------------------------
    def states(self, objects: dict) -> dict:
        import ironcord
        import mechkit as mk
        from scoring import BOARD, HOLDER, IRON

        out = dict(objects)
        out[IRON] = ironcord.IronCordState.build(self.iron_spec, self.cord_points, self.tip, self.axis,
                                                 source="offline")
        out[BOARD] = mk.state_from_poses(self.board_spec, (*self.board_o, *yaw_quat(self.board_yaw)), np.zeros(6),
                                         source="offline")
        out[HOLDER] = mk.state_from_poses(self.holder_spec, (*self.holder_spec.spawn_pose()[0],
                                                             *self.holder_spec.spawn_pose()[1]), np.zeros(6),
                                          source="offline")
        return out
