"""Offline checker only: floorboards pushed along the floor by a closed gripper. The simulator never calls anything in
this file.

Our offline checker (tools/offline/check_task.py, NumPy only) carries a board rigidly with a gripper that closes on
it and drops it when the gripper opens, but it does not push anything: on its own it would leave each board where it
was set down, 18-22 mm short of its neighbour, while the closed fingers pass through it. The checker lets an object
build its own state (an ``offline_state(ObjectState)`` attribute on its spec), so here each board gets a small
quasi-static push model, and the scorer is fed the boards' poses from it:

- only a gripper with its fingers closed pushes (rigstate's opening at most 0.15). Its fingers fill a block
  (gap + 16 mm) deep along x, 20 mm wide along y, from the finger tips to 25 mm above the tool point (rigstate gives
  no heading; the demo pushes only at yaw +-90 degrees, along +x, and every other use of the grippers is a grasp);
- a block that overlaps a board from behind (its centre at smaller x than the board's) pushes it along +x, and only
  along x: the boards never turn here;
- the arm is a spring: pushing a load of N boards (each sliding on friction 1.0 under 0.11 kg), the block's leading
  face trails its commanded place by ``PUSH_COMPLIANCE`` (6 mm/N, the J K^-1 J^T estimate at the push poses) times
  the load's friction. So a board pushed into its neighbour stops against it unless the commanded face is far
  enough past it to slide both (the chain then packs tight and moves together); the far wall stops everything;
- a board the checker moves (carried, or dropped when let go) follows the checker and forgets its pushes; a pushed
  board stays where it was pushed.

So this predicts only what the scripted demo and its one-arm variants do. A block that came down onto a board from
above, pushed a board from in front, moved one more than 20 mm in one step, or drove the floor into the far wall is
recorded in ``BoardPush.warnings``; the task's offline tests require none in the demo.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np

import layout as L
import rigstate

JUMP_WARN = 0.020  # m: a single step's push larger than this is recorded
TOL = 1e-9


class BoardPush:
    """The boards' poses, offline, from the grippers rigstate is fed. One instance per task module."""

    def __init__(self, names=L.BOARD_NAMES, dt: float = 1.0 / L.CONTROL_HZ,
                 compliance: float = L.PUSH_COMPLIANCE) -> None:
        self.names = tuple(names)
        self.dt = dt
        self.compliance = compliance
        self.friction = L.FRICTION * L.BOARD_MASS * L.GRAVITY  # N to slide one board
        self.reset()

    def reset(self) -> None:
        self.raw: dict[str, tuple] = {}  # board -> (pos, quat) as the checker last reported it
        self.low: dict[str, np.ndarray] = {}  # board -> the checker's bounding box, low corner
        self.high: dict[str, np.ndarray] = {}
        self.off = {n: 0.0 for n in self.names}  # board -> how far pushes have moved it along x
        self.prev_off = dict(self.off)
        self.moving: set[str] = set()  # boards the checker moved at their last report (carried or dropping)
        self.prev_bottom: dict[str, float] = {}  # each gripper's block bottom at the previous step
        self.t = 0.0
        self.warnings: list[dict] = []
        self.log: list[dict] = []  # every push: time, arm, board, mm

    # -- hooks the offline checker calls (never the simulator) -------------------------------------------------
    def attach(self, specs) -> None:
        """Give each board's spec the offline hook: a plain attribute on the (frozen) spec; the simulator never
        reads it."""
        try:
            for spec in specs:
                object.__setattr__(spec, "offline_state", self._hook(spec.name))
        except (AttributeError, TypeError):  # a spec class that takes no attributes: nothing to do offline
            pass

    def _hook(self, name: str):
        def hook(state):
            return self.board_state(name, state)
        return hook

    def board_state(self, name: str, state):
        """The board's state this control step. The checker calls the boards in their order in OBJECTS; the
        first one's call advances the model by a step."""
        key = (tuple(round(float(v), 9) for v in state.pos), tuple(round(float(v), 9) for v in state.quat_xyzw))
        if self.raw.get(name) != key:  # the checker moved it: follow, forget the pushes
            if name in self.raw:
                self.moving.add(name)
            self.off[name] = 0.0
            self.prev_off[name] = 0.0
            self.raw[name] = key
        else:
            self.moving.discard(name)
        self.low[name] = np.asarray(state.low, dtype=float)
        self.high[name] = np.asarray(state.high, dtype=float)
        if name == self.names[0]:
            self.t += self.dt
            self._step()
        d = np.array([self.off[name], 0.0, 0.0])
        vel = np.asarray(state.lin_vel, dtype=float) + np.array([(self.off[name] - self.prev_off[name]) / self.dt,
                                                                   0.0, 0.0])
        self.prev_off[name] = self.off[name]
        fields = {"pos": np.asarray(state.pos, dtype=float) + d, "low": self.low[name] + d,
                  "high": self.high[name] + d, "lin_vel": vel}
        if dataclasses.is_dataclass(state):  # the checker's ObjectState
            return dataclasses.replace(state, **fields)
        out = SimpleNamespace(**vars(state)) if hasattr(state, "__dict__") else SimpleNamespace()
        for k, v in fields.items():
            setattr(out, k, v)
        if not hasattr(out, "quat_xyzw"):
            out.quat_xyzw = state.quat_xyzw
        if not hasattr(out, "ang_vel"):
            out.ang_vel = state.ang_vel
        return out

    # -- the model ---------------------------------------------------------------------------------------------
    def box(self, name: str) -> tuple[float, float, float, float, float, float]:
        """(x0, x1, y0, y1, z0, z1) of the board as the model has it."""
        lo, hi, o = self.low[name], self.high[name], self.off[name]
        return (float(lo[0]) + o, float(hi[0]) + o, float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))

    def _static(self) -> list[str]:
        return [n for n in self.names if n in self.low and n not in self.moving]

    def _step(self) -> None:
        arms = rigstate.arms()
        if not isinstance(arms, dict):
            return
        for side in ("left", "right"):
            a = arms.get(side)
            if a is None:
                continue
            try:
                tcp, tips, opening = (np.asarray(a.tcp, dtype=float), np.asarray(a.tips, dtype=float),
                                      float(a.opening))
            except (AttributeError, TypeError, ValueError):
                continue
            if not (np.all(np.isfinite(tcp)) and np.all(np.isfinite(tips)) and math.isfinite(opening)):
                continue
            half = float(np.clip(opening, 0.0, 1.0)) * L.FULL_GAP / 2 + L.PAD_THICK
            bx0, bx1 = float(tcp[0]) - half, float(tcp[0]) + half
            by0, by1 = float(tcp[1]) - L.PAD_HALF_WIDTH, float(tcp[1]) + L.PAD_HALF_WIDTH
            bz0, bz1 = float(min(tips[2], tcp[2])), float(tcp[2]) + L.PAD_ABOVE_TCP
            was = self.prev_bottom.get(side, -math.inf)
            self.prev_bottom[side] = bz0
            if opening > L.PUSH_OPENING_MAX:
                continue  # open fingers grasp and let go; they do not push
            hit = []
            for n in self._static():
                x0, x1, y0, y1, z0, z1 = self.box(n)
                if bx1 <= x0 + TOL or bx0 >= x1 - TOL or by1 <= y0 or by0 >= y1 or bz1 <= z0 or bz0 >= z1:
                    continue
                if was >= z1 - TOL:
                    self.warnings.append({"t": round(self.t, 2), "arm": side, "board": n,
                                          "issue": "came down onto the board"})
                    continue
                if float(tcp[0]) >= (x0 + x1) / 2:
                    self.warnings.append({"t": round(self.t, 2), "arm": side, "board": n,
                                          "issue": "pushed the board from in front"})
                    continue
                hit.append((x0, n))
            if hit:
                self._push(min(hit)[1], bx1, side)

    def _lane(self, p: str) -> list[str]:
        """The pushed board and every static board ahead of it that it could drive (overlapping it across y and in
        height), nearest first."""
        x0, _, y0, y1, z0, z1 = self.box(p)
        ahead = []
        for n in self._static():
            if n == p:
                continue
            a0, _, b0, b1, c0, c1 = self.box(n)
            if a0 >= x0 - TOL and b1 > y0 and b0 < y1 and c1 > z0 and c0 < z1:
                ahead.append((a0, n))
        return [p] + [n for _, n in sorted(ahead)]

    def _push(self, p: str, lead: float, side: str) -> None:
        """The quasi-static response to a block whose commanded leading face is at x ``lead``, behind board p."""
        lane = self._lane(p)
        x0 = {n: self.box(n)[0] for n in lane}
        w = {n: self.box(n)[1] - self.box(n)[0] for n in lane}

        def pack(k: int, front: float) -> dict[str, float]:
            pos, cur = {}, front
            for n in lane[:k]:
                pos[n] = max(x0[n], cur)
                cur = pos[n] + w[n]
            return pos

        k = 1
        while True:
            pos = pack(k, lead - self.compliance * self.friction * k)
            if pos[p] <= x0[p] + TOL:
                return  # the spring's pull is under the friction: nothing moves
            last = lane[k - 1]
            if k < len(lane) and pos[last] + w[last] > x0[lane[k]] + TOL:
                pos2 = pack(k + 1, lead - self.compliance * self.friction * (k + 1))
                if pos2[lane[k]] > x0[lane[k]] + TOL:
                    k += 1  # enough to slide the next board as well
                    continue
                limit = x0[lane[k]]  # not enough: the moving boards pack up against it
                for n in reversed(lane[:k]):
                    pos[n] = max(x0[n], min(pos[n], limit - w[n]))
                    limit = pos[n]
            break
        front = lane[k - 1]
        if pos[front] + w[front] > L.FAR_X + TOL:  # the far wall stops the floor
            self.warnings.append({"t": round(self.t, 2), "arm": side, "board": front,
                                  "issue": "drove the floor into the far wall"})
            limit = L.FAR_X
            for n in reversed(lane[:k]):
                pos[n] = max(x0[n], min(pos[n], limit - w[n]))
                limit = pos[n]
        for n, x in pos.items():
            d = x - x0[n]
            if d <= TOL:
                continue
            if d > JUMP_WARN:
                self.warnings.append({"t": round(self.t, 2), "arm": side, "board": n,
                                      "issue": "pushed it more than 20 mm in one step", "mm": round(d * 1000, 1)})
            self.off[n] += d
            self.log.append({"t": round(self.t, 2), "arm": side, "board": n, "mm": round(d * 1000, 3)})
