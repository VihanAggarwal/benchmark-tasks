"""Offline checker only: the crate riding on the fork bars. The simulator never calls anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) carries an object rigidly with a gripper that closes on
it and drops it when the gripper opens, but it does not move what rests on a carried object: on its own it would
leave the crate on the floor bearers while the bars rise through it. The checker lets an object build its own state
(an ``offline_state(ObjectState)`` attribute on its spec, the hook tools/kit/mechkit.py uses for its assemblies), so
here the crate gets a small quasi-static model, and the scorer is fed the crate's pose from it:

- each of the crate's two pocket lines (y +-0.120 in its frame) is held at the higher of the fixed support under the
  crate (floor bearers 0.036, rack platform 0.036, rack bearers 0.072, else the table) and the top of a bar that is
  under that line (within 15 mm across, overlapping the crate's length by 10 mm or more, its top no higher than
  10 mm above the crate's underside there, so a bar can only come up from below);
- both lines on bars: the crate is carried, level or rolled between them, and moves horizontally with the bars' mean
  motion (it rides on them by friction);
- one line on a bar: the crate pivots about its opposite edge, which stays on the fixed support (it tilts); it does
  not move horizontally;
- neither: it rests on the fixed support.

The crate's yaw never changes and it never pitches about its width. A bar driven into the crate's box by more than
2 mm (from the side, or from above) is recorded in ``CrateRide.warnings``; the task's offline tests require none in
the demo. Offline, the checker also sees the crate as its starting box flagged approximate (``offline_box``), so its
own finger and carried-object interference checks, which would run against that stale box after the lift, skip it;
the scorer's touch rule (finger zones against the modelled crate) and these warnings take their place.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np

import layout as L

FIXED = (  # (x from, x to, top above the table): what the crate can stand on
    (L.FLOOR_X0, L.FLOOR_X1, L.FLOOR_TOP),
    (L.RACK_X0, L.RACK_X1, L.RACK_PLATFORM_H),
    (L.RACK_BEARER_X0, L.RACK_BEARER_X1, L.RACK_TOP),
)
MIN_OVERLAP = 0.010  # m of the crate's length over a support (or a bar) for it to bear on it
COME_FROM_BELOW = 0.010  # a bar may bear on a pocket line only if its top is at most this above the underside there
PENETRATION_WARN = 0.002
HALF = np.array(L.CRATE_SIZE, dtype=float) / 2
BAR_HALF = np.array(L.BAR_SIZE, dtype=float) / 2
_SIGNS = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _quat_xyzw(r: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return np.array([x, y, z, w])


def _yaw(quat_xyzw) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class CrateRide:
    """The crate's pose, offline, from the bars the checker carries. One instance per task module."""

    def __init__(self, bar_names: tuple[str, str], dt: float = 1.0 / L.CONTROL_HZ) -> None:
        self.bar_names = tuple(bar_names)
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.bars: dict = {}  # the bars' states this step (from their own offline_state hooks)
        self.prev_bar_xy: dict = {}
        self.pose = None  # (x, y, yaw, underside height at the centre (world z), roll)
        self.prev_centre = None
        self.prev_roll = 0.0
        self.t = 0.0
        self.mode = "rest"
        self.warnings: list[dict] = []

    # -- hooks the offline checker calls (never the simulator) -------------------------------------------------
    def attach(self, crate_spec, bar_specs) -> None:
        """Give the specs the offline hooks: the bars report their states, the crate builds its own. Plain
        attributes on the (frozen) specs; the simulator never reads them."""
        try:
            for spec in bar_specs:
                object.__setattr__(spec, "offline_state", self._bar_hook(spec.name))
            object.__setattr__(crate_spec, "offline_state", self.crate_state)
            object.__setattr__(crate_spec, "offline_box", lambda: (tuple(crate_spec.pos), tuple(crate_spec.size),
                                                                   float(crate_spec.yaw)))
        except (AttributeError, TypeError):  # a spec class that takes no attributes: nothing to do offline
            pass

    def _bar_hook(self, name: str):
        def hook(state):
            self.bars[name] = state
            return state
        return hook

    # -- the model ---------------------------------------------------------------------------------------------
    @staticmethod
    def fixed_top(x: float, yaw: float) -> float:
        """Height above the table of the highest fixed support under the crate's length."""
        reach = abs(math.cos(yaw)) * HALF[0] + abs(math.sin(yaw)) * HALF[1]
        lo, hi = x - reach, x + reach
        tops = [top for x0, x1, top in FIXED if min(hi, x1) - max(lo, x0) >= MIN_OVERLAP]
        return max(tops, default=0.0)

    def _bar_top_under(self, side: int, x: float, y: float, yaw: float, under_z: float) -> float | None:
        """Top (world z) of the highest bar bearing on the pocket line ``side`` (+1: the crate's +y), or None."""
        r = _rot_z(yaw)
        best = None
        for name in self.bar_names:
            s = self.bars.get(name)
            if s is None:
                continue
            p = r.T @ (np.asarray(s.pos, dtype=float) - np.array([x, y, 0.0]))
            u = r.T @ (_rot_z(_yaw(s.quat_xyzw)) @ np.array([1.0, 0.0, 0.0]))
            ends_x = sorted((p[0] - u[0] * BAR_HALF[0], p[0] + u[0] * BAR_HALF[0]))
            overlap = min(ends_x[1], HALF[0]) - max(ends_x[0], -HALF[0])
            top = float(s.pos[2]) + BAR_HALF[2]
            if (abs(p[1] - side * L.POCKET_Y) <= L.POCKET_BAND and overlap >= MIN_OVERLAP
                    and top <= under_z + COME_FROM_BELOW):
                best = top if best is None else max(best, top)
        return best

    def crate_state(self, state):
        """The crate's state this control step (called by the checker after the bars' hooks)."""
        if self.pose is None:
            yaw = _yaw(state.quat_xyzw)
            self.pose = (float(state.pos[0]), float(state.pos[1]), yaw, float(state.pos[2]) - HALF[2], 0.0)
            self.prev_bar_xy = {n: np.asarray(s.pos[:2], dtype=float) for n, s in self.bars.items()}
        x, y, yaw, under, roll = self.pose
        base = L.TOP + self.fixed_top(x, yaw)
        heights = {}
        on_bar = {}
        for side in (+1, -1):
            here = under + side * L.POCKET_Y * math.tan(roll)  # the underside over this pocket line, as it was
            bar = self._bar_top_under(side, x, y, yaw, here)
            on_bar[side] = bar is not None and bar > base + 1e-4
            heights[side] = bar if on_bar[side] else base
        if on_bar[+1] and on_bar[-1]:
            mode = "carried"
            under = 0.5 * (heights[+1] + heights[-1])
            roll = math.atan2(heights[+1] - heights[-1], 2 * L.POCKET_Y)
            moves = [np.asarray(self.bars[n].pos[:2], dtype=float) - self.prev_bar_xy[n]
                     for n in self.bar_names if n in self.bars and n in self.prev_bar_xy]
            if moves:
                dx, dy = np.mean(moves, axis=0)
                x, y = x + float(dx), y + float(dy)
        elif on_bar[+1] or on_bar[-1]:
            mode = "pivot"
            side = +1 if on_bar[+1] else -1
            lever = L.POCKET_Y + HALF[1]  # from the bar's line to the crate's opposite edge
            roll = side * math.atan2(heights[side] - base, lever)
            under = base + HALF[1] * math.tan(abs(roll))
        else:
            mode = "rest"
            under, roll = base, 0.0
        self.pose = (x, y, yaw, under, roll)
        self.mode = mode
        self.prev_bar_xy = {n: np.asarray(s.pos[:2], dtype=float) for n, s in self.bars.items()}
        r = _rot_z(yaw) @ _rot_x(roll)
        centre = np.array([x, y, under]) + r @ np.array([0.0, 0.0, HALF[2]])
        vel = np.zeros(3) if self.prev_centre is None else (centre - self.prev_centre) / self.dt
        spin = _rot_z(yaw) @ np.array([1.0, 0.0, 0.0]) * ((roll - self.prev_roll) / self.dt)
        self.prev_centre, self.prev_roll = centre, roll
        cs = centre + (_SIGNS * HALF) @ r.T
        self.t += self.dt
        self._check_penetration(centre, r)
        fields = {"pos": centre, "quat_xyzw": _quat_xyzw(r), "lin_vel": vel, "ang_vel": spin,
                  "low": cs.min(axis=0), "high": cs.max(axis=0)}
        if dataclasses.is_dataclass(state):  # the checker's ObjectState
            return dataclasses.replace(state, **fields)
        return SimpleNamespace(**fields)

    def _check_penetration(self, centre: np.ndarray, r: np.ndarray) -> None:
        """Record any bar more than 2 mm inside the crate's box (a bar driven into it from the side or above)."""
        for name in self.bar_names:
            s = self.bars.get(name)
            if s is None:
                continue
            rb = _rot_z(_yaw(s.quat_xyzw))
            pts = np.asarray(s.pos, dtype=float) + (_SIGNS * BAR_HALF) @ rb.T
            # sample the bar's volume (corners, edge midpoints, centre) in the crate's frame
            samples = np.vstack([pts, 0.5 * (pts[:, None, :] + pts[None, :, :]).reshape(-1, 3)])
            local = (samples - centre) @ r
            depth = np.min(HALF - np.abs(local), axis=1).max()
            if depth > PENETRATION_WARN:
                self.warnings.append({"t": round(self.t, 2), "bar": name, "depth_mm": round(float(depth) * 1000, 1)})
