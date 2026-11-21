"""The offline stand-in for the physics (tools/offline/check_task.py only; never used in the simulator).

The checker's own proxy only carries parts between closed pads. Without a simulator, task.py steps this model from the
arms' commanded TCPs, pad tips and openings, and scores what it gives:

1. The pencil. A gripper whose pad-tip point lies within 6 mm of the pencil's body axis, along its body, with its pads
   commanded under the body's 18 mm plus 2 mm, holds it; the pencil then follows the TCP rigidly (translation only, it
   stays upright) until the pads open past that width. Let go over the block's slot (the body's axis within 3 mm of
   the slot's), it drops into the block with its tip on the slot's floor; let go anywhere else it drops straight down
   until its tip meets the board's top face or the table. Once the held pencil has drawn (its tip on the board's
   face) and is lifted 5 mm off it, its tip shifts L.PEN_SLIP_Y along y from the commanded TCP (the net of what the
   GPU runs showed: the grasp's 1.25 mm offset in the pads, the TCP's error over the block and the stroke's drag
   turning the pencil in the pads; see layout.PEN_SLIP_Y), so the return into the block must allow for it.
2. The pencil block is fixed to the table; its slot lets the pencil lift out freely.
3. The square. A gripper whose pad-tip point lies within 6.5 mm of the grip point on the blade's tail (layout.GRIP_LOCAL)
   and between 4 and 18 mm over the table (pads beside the tail's edges), with its pads commanded under the blade's
   25.4 mm plus 2 mm, holds the blade: while it holds,
   the square slides with the pad tips in x and y, except that its face stops at the edge (the board does not give).
   Otherwise it stays where it is: the model never moves it for the pencil's push, so a one-handed stroke is not
   modelled as slipping (the one-arm check drops a whole arm). Pressing on the head's top moves nothing (the first
   GPU run showed it: closed fingertips pushed over the top slid over it and the square stayed put).
4. The board never moves.
Every number is a design value, not a measurement.
"""

from __future__ import annotations

import numpy as np

import layout as L
import mechkit as mk
import parts as P
import rigstate

HOLD_W = L.PN_W + 0.002
GRIP_W = L.SQ_BLADE_W + 0.002
GRIP_R = 0.0065  # the pinch's reach about the grip point (covers PRESS_X_BIAS: the hand runs on past the stop)
X_SEAT = L.NEAR_X - L.SEAT_GAP  # the face's x when pushed home


class Model:
    def __init__(self, board_spec, square_spec, pencil_spec, block_spec) -> None:
        self.specs = (board_spec, square_spec, pencil_spec, block_spec)
        self.reset()

    def reset(self) -> None:
        self.pencil = np.array(L.PN_START, dtype=float)
        self.square = np.array(L.SQ_START, dtype=float)
        self.hold: tuple | None = None  # (arm, offset pencil - tcp)
        self.drawn = False  # the held pencil's tip has been on the board's face
        self.slipped = False
        self.press: tuple | None = None  # (arm, last pad-tip x, last pad-tip y)

    def _on_blade(self, tips, gap) -> bool:
        s = np.asarray(tips, dtype=float) - self.square
        return bool(gap <= GRIP_W and np.hypot(float(s[0]) - L.GRIP_LOCAL[0], float(s[1]) - L.GRIP_LOCAL[1]) <= GRIP_R
                    and 0.004 <= float(s[2]) <= L.SQ_BLADE_Z - 0.001)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        g = {side: (np.asarray(t, float), np.asarray(p, float), float(o) * rigstate.FULL_GAP)
             for side, (t, p, o) in zip(rigstate.ARMS, read)}
        # the pencil
        if self.hold is not None:
            side, off = self.hold
            tcp, _, gap = g[side]
            if gap > HOLD_W:
                self.hold = None
                self._drop()
            else:
                self.pencil = tcp + off
                face = L.TOP + L.BOARD_H
                if float(self.pencil[2]) <= face + L.INK_Z:
                    self.drawn = True
                elif self.drawn and not self.slipped and float(self.pencil[2]) >= face + 0.005:
                    self.slipped = True
                    off = off + np.array([0.0, L.PEN_SLIP_Y, 0.0])
                    self.hold = (side, off)
                    self.pencil = tcp + off
        if self.hold is None:
            for side, (tcp, tips, gap) in g.items():
                rel = tips - (self.pencil + np.array([L.PN_CENTER_X, 0.0, 0.0]))
                along = float(rel[2])
                if (gap <= HOLD_W and np.hypot(rel[0], rel[1]) <= 0.006
                        and 0.010 <= along <= L.PN_LEN - 0.005):
                    self.hold = (side, self.pencil - tcp)
                    break
        # the square
        if self.press is not None:
            side, lx, ly = self.press
            _, tips, gap = g[side]
            if self._on_blade(tips, gap):
                self.square[0] = min(self.square[0] + float(tips[0]) - lx, X_SEAT)
                self.square[1] += float(tips[1]) - ly
                self.press = (side, float(tips[0]), float(tips[1]))
            else:
                self.press = None
        if self.press is None:
            for side, (_, tips, gap) in g.items():
                if self._on_blade(tips, gap):
                    self.press = (side, float(tips[0]), float(tips[1]))
                    break

    def _drop(self) -> None:
        x, y = float(self.pencil[0]) + L.PN_CENTER_X, float(self.pencil[1])
        if np.hypot(x - L.BL_X, y - L.BL_Y) <= 0.003:
            self.pencil = np.array([L.BL_X - L.PN_CENTER_X, L.BL_Y, L.TOP + L.BL_FLOOR])
        elif L.NEAR_X <= x <= L.FAR_X and abs(y - L.BOARD_Y) <= L.BOARD_L / 2:
            self.pencil = np.array([self.pencil[0], self.pencil[1], L.TOP + L.BOARD_H])
        else:
            self.pencil = np.array([self.pencil[0], self.pencil[1], L.TOP])

    def states(self, names) -> dict:
        board, square, pencil, block = names
        bs, ss, ps, cs = self.specs
        q = (0.0, 0.0, 0.0, 1.0)
        return {board: mk.state_from_poses(bs, (*L.BOARD_ORIGIN, *q), source="offline"),
                square: mk.state_from_poses(ss, (*self.square, *q), source="offline"),
                pencil: mk.state_from_poses(ps, (*self.pencil, *q), source="offline"),
                block: mk.state_from_poses(cs, (*L.BLOCK_ORIGIN, *q), source="offline")}


__all__ = ["Model", "P"]
