"""Offline checker only: the cart pushed by a closed gripper. The simulator never calls anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) carries an object rigidly with a gripper that closes on
it and drops it when the gripper opens, but it does not push anything: on its own it would leave the cart on the dock
while the grippers pass through it. The checker lets an object build its own state (an ``offline_state(ObjectState)``
attribute on its spec, the hook tools/kit/mechkit.py uses for its assemblies), so here the cart gets a small
quasi-static model, and the scorer is fed the cart's pose from it:

- a gripper pushes like the demo's: its fingers closing along y (yaw 0), so the fingers fill a block 27 mm wide
  along x, (gap + 16 mm) deep along y, from the finger tips to 25 mm above the tool point (rigstate gives no heading;
  the demo pushes only at yaw 0, and every other use of the grippers stays far from the cart);
- where that block overlaps the cart, the cart is moved along y, the shortest way out of the block (a push from
  behind moves it toward the truck bed); it never turns and never moves along x;
- it rests on the highest surface under its outline (dock 40.0 mm, a seated runner 39.5 mm, bed 39.0 mm; a runner
  that is not seated carries nothing here), and it stays up while its centre lies inside the outline of everything
  within 1 mm of that height under it; otherwise it falls into the gap: it is then left lying tipped 30 degrees with
  a corner on the table, and nothing moves it again.

So this predicts only what the scripted demo and its one-arm variants do (a cart pushed at an open gap falls in).
Moves of the cart into a block coming down from above, or by more than 20 mm in one step, are recorded in
``CartPush.warnings``; the task's offline tests require none in the demo. Offline, the checker also sees the cart as
its starting box flagged approximate (``offline_box``), so its own finger interference checks, which would run
against that stale box after the push, skip it.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np

import layout as L
import rigstate
import scoring as S

PUSH_HALF_X = L.PAD_HALF_WIDTH  # the fingers' half width across their closing axis
PUSH_ABOVE_TCP = 0.025  # the fingers' block reaches this far above the tool point
OVERLAP = 0.001  # m of the cart's outline over a surface for it to bear on it
CONTACT_BAND = 0.001  # surfaces within this of the highest one under the cart all carry it
JUMP_WARN = 0.020  # m: a single step's push larger than this is recorded
TIPPED_DEG = 30.0
HALF = np.array(L.CART_HALF, dtype=float)
_SIGNS = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)


def _yaw(quat_xyzw) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _hull(points) -> list:
    """Convex hull of 2-D points, counter-clockwise (monotone chain)."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(pts) <= 2:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(list(reversed(pts)))
    return lower[:-1] + upper[:-1]


def _inside_hull(p, hull, tol: float = 1e-9) -> bool:
    if len(hull) < 3:
        return False
    for i in range(len(hull)):
        (x0, y0), (x1, y1) = hull[i], hull[(i + 1) % len(hull)]
        if (x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0) < -tol:
            return False
    return True


class CartPush:
    """The cart's pose, offline, from the grippers rigstate is fed. One instance per task module."""

    def __init__(self, runner_names=S.RUNNER_NAMES, dt: float = 1.0 / L.CONTROL_HZ) -> None:
        self.runner_names = tuple(runner_names)
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.runners: dict = {}  # the runners' states this step (from their own offline_state hooks)
        self.pose = None  # (x, y, underside height, world z)
        self.fallen = False
        self.prev_centre = None
        self.prev_bottom: dict = {}  # each gripper's fingers' lowest point at the previous step
        self.t = 0.0
        self.mode = "rest"
        self.warnings: list[dict] = []

    # -- hooks the offline checker calls (never the simulator) -------------------------------------------------
    def attach(self, cart_spec, runner_specs) -> None:
        """Give the specs the offline hooks: the runners report their states, the cart builds its own. Plain
        attributes on the (frozen) specs; the simulator never reads them."""
        try:
            for spec in runner_specs:
                object.__setattr__(spec, "offline_state", self._runner_hook(spec.name))
            object.__setattr__(cart_spec, "offline_state", self.cart_state)
            object.__setattr__(cart_spec, "offline_box", lambda: (tuple(cart_spec.pos), tuple(cart_spec.size),
                                                                  float(cart_spec.yaw)))
        except (AttributeError, TypeError):  # a spec class that takes no attributes: nothing to do offline
            pass

    def _runner_hook(self, name: str):
        def hook(state):
            self.runners[name] = state
            return state
        return hook

    # -- the model ---------------------------------------------------------------------------------------------
    def surfaces(self) -> list[tuple]:
        """(x0, x1, y0, y1, top above the table) of everything the cart can rest on: the dock's and the bed's
        blocks at its level, and each seated runner's top."""
        out = list(L.CART_SURFACES)
        for name in self.runner_names:
            s = self.runners.get(name)
            if s is not None and S.valid(s) and S.seat_of(s) is not None:
                c = S.corners(s, S.RUNNER_HALF)
                out.append((float(c[:, 0].min()), float(c[:, 0].max()), float(c[:, 1].min()), float(c[:, 1].max()),
                            float(c[:, 2].max()) - L.TOP))
        return out

    def support(self, x: float, y: float) -> tuple[float | None, bool]:
        """(the height it rests at above the table, whether its centre is carried) for the cart centred at x, y."""
        cx0, cx1, cy0, cy1 = x - HALF[0], x + HALF[0], y - HALF[1], y + HALF[1]
        under = []
        for x0, x1, y0, y1, top in self.surfaces():
            ox0, ox1, oy0, oy1 = max(cx0, x0), min(cx1, x1), max(cy0, y0), min(cy1, y1)
            if ox1 - ox0 >= OVERLAP and oy1 - oy0 >= OVERLAP:
                under.append((top, (ox0, ox1, oy0, oy1)))
        if not under:
            return None, False
        rest = max(top for top, _ in under)
        pts = [(a, b) for top, (ox0, ox1, oy0, oy1) in under if top >= rest - CONTACT_BAND
               for a in (ox0, ox1) for b in (oy0, oy1)]
        return rest, _inside_hull((x, y), _hull(pts))

    def _push(self, x: float, y: float, bottom: float) -> float:
        """The cart's y after the grippers' finger blocks have pushed it out of their way."""
        arms = rigstate.arms()
        if not S.arms_valid(arms):
            return y
        for side in S.ARMS:
            a = arms[side]
            tcp, tips = np.asarray(a.tcp, dtype=float), np.asarray(a.tips, dtype=float)
            half_y = S.finger_gap(a) / 2 + L.PAD_THICK
            z0, z1 = float(min(tips[2], tcp[2])), float(tcp[2]) + PUSH_ABOVE_TCP
            bx0, bx1 = float(tcp[0]) - PUSH_HALF_X, float(tcp[0]) + PUSH_HALF_X
            by0, by1 = float(tcp[1]) - half_y, float(tcp[1]) + half_y
            top = bottom + L.CART_SIZE[2]
            was_above = self.prev_bottom.get(side, -math.inf) >= top
            self.prev_bottom[side] = z0
            overlap_x = min(bx1, x + HALF[0]) - max(bx0, x - HALF[0])
            overlap_y = min(by1, y + HALF[1]) - max(by0, y - HALF[1])
            if overlap_x <= 0 or overlap_y <= 0 or z0 >= top or z1 <= bottom:
                continue
            new = by1 + HALF[1] if (by0 + by1) / 2 <= y else by0 - HALF[1]
            if was_above:
                self.warnings.append({"t": round(self.t, 2), "arm": side, "issue": "came down onto the cart"})
            if abs(new - y) > JUMP_WARN:
                self.warnings.append({"t": round(self.t, 2), "arm": side, "issue": "pushed it more than 20 mm in "
                                      "one step", "mm": round(abs(new - y) * 1000, 1)})
            y = new
        return y

    def cart_state(self, state):
        """The cart's state this control step (called by the checker after the runners' hooks)."""
        self.t += self.dt
        if self.pose is None:
            self.pose = (float(state.pos[0]), float(state.pos[1]), float(state.pos[2]) - HALF[2])
        x, y, bottom = self.pose
        if self.fallen:
            r, centre = self._fallen_pose(x, y)
            self.mode = "fallen"
        else:
            y = self._push(x, y, bottom)
            rest, carried = self.support(x, y)
            if rest is None or not carried:
                self.fallen = True
                self.mode = "fallen"
                r, centre = self._fallen_pose(x, y)
            else:
                bottom = L.TOP + rest
                self.mode = "rest"
                r, centre = np.eye(3), np.array([x, y, bottom + HALF[2]])
        self.pose = (x, y, bottom)
        vel = np.zeros(3) if self.prev_centre is None else (centre - self.prev_centre) / self.dt
        self.prev_centre = centre
        cs = centre + (_SIGNS * HALF) @ r.T
        q = _quat_about_x(math.radians(TIPPED_DEG)) if self.fallen else np.array([0.0, 0.0, 0.0, 1.0])
        fields = {"pos": centre, "quat_xyzw": q, "lin_vel": vel, "ang_vel": np.zeros(3),
                  "low": cs.min(axis=0), "high": cs.max(axis=0)}
        if dataclasses.is_dataclass(state):  # the checker's ObjectState
            return dataclasses.replace(state, **fields)
        return SimpleNamespace(**fields)

    @staticmethod
    def _fallen_pose(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Tipped 30 degrees about x with its lowest corner on the table: fallen into the gap."""
        a = math.radians(TIPPED_DEG)
        r = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
        centre = np.array([x, y, 0.0])
        low = float(((_SIGNS * HALF) @ r.T)[:, 2].min())
        centre[2] = L.TOP - low
        return r, centre


def _quat_about_x(a: float) -> np.ndarray:
    return np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])
