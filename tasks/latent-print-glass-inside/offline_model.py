"""Offline checker only: where the glass goes when a gripper holds it from the inside. The simulator never calls
anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics, and its own grasp proxy carries an
object only when the pads close ACROSS it, so it never moves the glass for an internal grasp (the fingers go in
nearly closed and open against the inner walls). It hands task.py's hooks the glass as mechkit's assembly at its
start (``Assembly.offline_state``: ``source`` "offline") and feeds rigstate the arms from its kinematics (TCP,
finger-tip point and commanded opening). When task.py sees ``source == "offline"`` it asks this model, once per
control step, for the glass state (and the arms as the physics would leave them), and the SAME scorer runs on that.

The model, step by step:

- a gripper has ENTERED the glass when its tip point goes from above the rim to below it with its fingers (at their
  current opening) fitting through the mouth at some turn, without touching the inner walls (the scorer's finger
  test with no allowance); it stays entered while its tips are below the rim and its fingers are 'inside' (the
  scorer's test), and leaves when either stops;
- an entered gripper whose tips are 25 mm or more below the rim (so the fingers' 10 mm-thick band is inside) and
  above the inner floor, opened to the contact gap (42 mm: outer faces on both inner walls) or more, GRIPS the glass
  (first come, first served); from then on the glass keeps its offset from that gripper's tips, keeps its yaw and
  stays upright, and the reported opening is held at the contact gap (the walls stop the fingers there, as the
  simulator's joint readings would show);
- the glass cannot go below what is under its outline (table, start mark, exam sheet, a box wall's top): if the
  gripper pushes it lower, the fingers slide down inside it (its offset from the tips changes);
- when the gripper's fingers close below 40 mm (off the walls), it lets go: the glass falls freely (9.81 m/s2, one
  step at a time, with its real downward speed reported) onto whatever is under its outline;
- an arm that never enters the glass, or that grips it from outside, moves nothing here: if the checker's own proxy
  carries the glass (pads closed across its outside), the checker's state is used instead and the model lets go.

It cannot model lag, slip under acceleration, wrist sag, the glass shifting as the fingers open (it does not centre
the glass between them), a grip from a tilted gripper tilting the glass, or anything a policy other than the
scripted demo does with contacts: those are for the GPU runs.
"""

from __future__ import annotations

import numpy as np

import layout as L
import scoring as S

RELEASE_GAP = L.CONTACT_GAP - 0.002  # 0.040: the fingers are off the walls
MIN_DEPTH = 0.025  # tips this far below the rim before a grip counts
MOVED_TOL = 0.002  # the checker's own state this far from the start: its proxy moved the glass


class _Arm:
    """An arm as rigstate reports it, with the opening the walls allow."""

    __slots__ = ("tcp", "tips", "opening")

    def __init__(self, tcp, tips, opening: float) -> None:
        self.tcp, self.tips, self.opening = tcp, tips, opening


class _State:
    """The glass's state as the scorer reads it (ObjectState's fields), plus where it came from."""

    def __init__(self, pos, vel, source: str) -> None:
        h = np.array([L.GLASS_HALF, L.GLASS_HALF, L.GLASS_HALF_H])
        self.pos = np.asarray(pos, dtype=float)
        self.quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self.lin_vel = np.asarray(vel, dtype=float)
        self.ang_vel = np.zeros(3)
        self.low, self.high = self.pos - h, self.pos + h
        self.source = source


class GlassModel:
    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.centre = None  # the glass's centre (world); None until the first step
        self.vz = 0.0  # downward speed while falling (m/s, negative)
        self.holder = None  # the arm gripping it
        self.offset = None  # glass centre minus the holder's tips
        self.entered = {s: False for s in S.ARMS}
        self.above = {s: True for s in S.ARMS}  # the arm's tips were above the rim at the last step
        self.key = None  # the arms snapshot this step was computed for (success() then score() reuse it)
        self.out = None
        self.events: list[dict] = []
        self.steps = 0

    # -- geometry --------------------------------------------------------------------------------------------------
    def _glass(self, centre, vel=(0.0, 0.0, 0.0)):
        return _State(centre, vel, "offline_model")

    @staticmethod
    def support(centre) -> float:
        """Height of whatever is highest under the glass's outline (it stands upright here)."""
        x, y = float(centre[0]), float(centre[1])
        h = L.GLASS_HALF
        poly = (x - h, x + h, y - h, y + h)

        def overlaps(r) -> bool:
            return poly[0] < r[1] and r[0] < poly[1] and poly[2] < r[3] and r[2] < poly[3]

        top = L.TOP if (L.TABLE_X[0] <= x <= L.TABLE_X[1] and L.TABLE_Y[0] <= y <= L.TABLE_Y[1]) else 0.0
        for r, z in ((S.MARK, L.MARK_TOP), (S.SHEET, L.SHEET_TOP), *((w, L.BOX_TOP) for w in S.BOX_WALLS.values())):
            if overlaps(r):
                top = max(top, z)
        return top

    # -- one control step ------------------------------------------------------------------------------------------
    def step(self, checker_state, arms):
        """The glass state and the arms for this control step. ``checker_state`` is what the offline checker made
        of the glass (its start, unless its own proxy carried it); ``arms`` is rigstate.arms()."""
        if arms is not None and arms is self.key and self.out is not None:
            return self.out  # the same step again (score() after success())
        self.key = arms
        self.steps += 1
        ck = np.asarray(checker_state.pos, dtype=float)
        if self.centre is None:
            self.centre = ck.copy()
            self.start = ck.copy()
        if float(np.linalg.norm(ck - self.start)) > MOVED_TOL:  # the checker's proxy carries it (outside grip)
            if self.holder is not None:
                self.events.append({"step": self.steps, "event": "checker_moved_it", "arm": self.holder})
            self.holder, self.offset, self.vz = None, None, 0.0
            self.centre = ck.copy()
            self.out = (checker_state, arms)
            return self.out
        if not S.arms_valid(arms):
            self.out = (self._glass(self.centre), arms)
            return self.out
        prev = self.centre.copy()
        tips = {s: np.asarray(arms[s].tips, dtype=float) for s in S.ARMS}
        glass = self._glass(self.centre)
        seen = dict(arms)
        for s in S.ARMS:
            below = S.to_local(glass, tips[s])[2] < L.GLASS_HALF_H
            if below and self.entered[s]:  # already in: the walls stop the fingers at the contact gap
                seen[s] = _Arm(arms[s].tcp, arms[s].tips, min(float(arms[s].opening), L.CONTACT_GAP / L.FULL_GAP))
                self.entered[s] = S.finger_check(seen[s], glass)["status"] == "inside"
            elif below and self.above[s]:  # crossing the rim now: only fingers that fit through the mouth go in
                self.entered[s] = S.finger_check(arms[s], glass, tol=0.0)["status"] == "inside"
            else:
                self.entered[s] = False
            if not self.entered[s]:
                seen[s] = arms[s]
            self.above[s] = not below
        gaps = {s: S.finger_gap(arms[s]) for s in S.ARMS}  # as commanded
        # let go
        if self.holder is not None and gaps[self.holder] < RELEASE_GAP:
            self.events.append({"step": self.steps, "event": "release", "arm": self.holder})
            self.holder, self.offset = None, None
        # grip
        if self.holder is None:
            for s in S.ARMS:
                local = S.to_local(glass, tips[s])
                deep = (L.GLASS_HALF_H - local[2] >= MIN_DEPTH and local[2] > -L.GLASS_HALF_H + L.GLASS_FLOOR)
                if self.entered[s] and deep and gaps[s] >= L.CONTACT_GAP:
                    self.holder, self.offset, self.vz = s, self.centre - tips[s], 0.0
                    self.events.append({"step": self.steps, "event": "grip", "arm": s})
                    break
        # move
        if self.holder is not None:
            want = tips[self.holder] + self.offset
            floor = self.support(want) + L.GLASS_HALF_H
            if want[2] < floor:  # pushed onto what is under it: the fingers slide down inside the glass
                want[2] = floor
                self.offset = want - tips[self.holder]
            self.centre = want
        else:
            floor = self.support(self.centre) + L.GLASS_HALF_H
            if self.centre[2] > floor + 1e-9 or self.vz < 0.0:
                self.vz -= L.GRAVITY * self.dt
                self.centre = self.centre + np.array([0.0, 0.0, self.vz * self.dt])
                if self.centre[2] <= floor:
                    self.centre[2], self.vz = floor, 0.0
                    self.events.append({"step": self.steps, "event": "landed", "z_mm": round(
                        (floor - L.GLASS_HALF_H - L.TOP) * 1000, 1)})
        vel = (self.centre - prev) / self.dt
        if self.holder is None and self.vz < 0.0:
            vel[2] = self.vz
        self.out = (self._glass(self.centre, vel), seen)
        return self.out

