"""Kinematics of the tongs for use ONLY where there is no simulator: tools/offline/check_task.py (which by itself can
carry and drop a gripped box, nothing else) and the tests. The simulator never takes a result from here: there the
hinge spring, the tab's slider and the lock block under the heel lugs are PhysX joints and contacts.

What this file assumes is what the design claims, so the offline prediction is the design's prediction, not a
measurement. It moves the tongs from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed):

- a gripper whose pads straddle both arms (its centre between them, open wider than they are) 20 to 140 mm from
  the pin, with the pad tips no more than 22 mm above the worktop, holds them: as it closes, the arms shut until
  their outer span (28 mm shut) fits between the pads, and the pin stays put while the two arms close symmetrically
  about the pads' centre line (the root turns by half the closing). The hold ends once the pads are wider than the
  arms at their rest angle;
- a gripper whose pads straddle the ring (pad tips no more than 24 mm up) and close below 9 mm holds the ring. While
  the arms are held, the ring's motion along the tab draws the tab, up to what the lugs allow at that angle
  (layout.travel_limit: 12 mm only within 2.86 deg of shut, 3.8 to 6.3 mm with the arms open); while they are not,
  the tab's 1 N of friction is more than the worktop's 0.79 N, so the whole tongs go with the ring (towed);
- arms that nobody holds come to rest where layout.rest_angle puts them: on the block (2.86 deg with the tab at its
  stop, 3.0 to 4.1 deg on its edge with the tab 5.9 to 11 mm out), or fully open with the tab pushed back to 3.8 mm;
- nothing is lifted, nothing tips, and the tongs stay on the worktop.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

HOLD_U = (20.0, 140.0)  # mm from the pin: where pads hold the arms (not on the heads or the hinge)
HOLD_MAX_TIP_H = 0.022  # pad tips at most this high: 3 mm of pad beside the arms (they span 5 to 25 mm up)
RING_MAX_TIP_H = 0.024  # the ring spans 3 to 27 mm up
RING_GRIP_GAP = 0.009  # m: pads this close on the 8 mm ring hold it
STRADDLE_TOL = 0.5  # mm


class OfflineTongs:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.origin = np.array(L.ORIGIN[:2], dtype=float)
        self.yaw = L.YAW
        self.theta = L.OPEN  # rad from shut
        self.tab = 0.0  # m out
        self.hold: str | None = None  # the arm that holds the arms shut
        self.hold_u = L.HOLD_U
        self.ring: str | None = None  # the arm that holds the ring
        self.around = {"left": False, "right": False}
        self.around_ring = {"left": False, "right": False}
        self.prev_tips: dict[str, np.ndarray] = {}
        self.towed = 0.0  # m the tongs were dragged in all

    # -- geometry -------------------------------------------------------------------------------------------------
    def _root(self, xy) -> np.ndarray:
        return L.to_root(xy, self.origin, self.yaw)[0]

    def _rest(self) -> tuple[float, bool]:
        return L.rest_angle(self.tab)

    def _settle_free(self) -> None:
        """Arms nobody holds: onto the block, or fully open with the tab pushed back."""
        theta, on_block = self._rest()
        self.theta = theta
        if not on_block:
            self.tab = min(self.tab, L.travel_limit(L.OPEN))

    # -- one control step -----------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return  # nothing can be judged from unreadable arms: the tongs stay as they are
        grips = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            grips[side] = (np.asarray(tips, dtype=float), gap)

        # the ring first, with the hold as it was: its motion this step
        for side, (tips, gap) in grips.items():
            prev = self.prev_tips.get(side)
            delta = np.zeros(2) if prev is None else tips[:2] - prev[:2]
            if self.ring == side:
                if gap > RING_GRIP_GAP + 1e-9:
                    self.ring = None
                else:
                    out = -L.base_rotation(self.yaw)[:, 1]  # world direction the tab is drawn in (-u)
                    if self.hold is not None:
                        self.tab = float(np.clip(self.tab + float(delta @ out), 0.0, L.travel_limit(self.theta)))
                    else:
                        self.origin = self.origin + delta
                        self.towed += float(np.linalg.norm(delta))
            if self.ring is None and self.hold != side:
                u, w = self._root(tips[:2])
                h = float(tips[2] - L.TOP)
                gap_mm = gap / L.MM
                t_mm = self.tab / L.MM
                in_ring = (L.RING_U[0] - t_mm - 1.0 <= u <= L.RING_U[1] - t_mm + 1.0 and -0.002 <= h <= RING_MAX_TIP_H)
                straddle = (w - gap_mm / 2 <= L.RING_W[0] + STRADDLE_TOL
                            and w + gap_mm / 2 >= L.RING_W[1] - STRADDLE_TOL)
                if in_ring and self.around_ring[side] and gap <= RING_GRIP_GAP:
                    self.ring = side
                self.around_ring[side] = bool(in_ring and straddle)

        # the hold on the arms
        for side, (tips, gap) in grips.items():
            if self.ring == side:
                self.around[side] = False
                continue
            u, w = self._root(tips[:2])
            h = float(tips[2] - L.TOP)
            gap_mm = gap / L.MM
            on_arms = HOLD_U[0] <= u <= HOLD_U[1] and -0.002 <= h <= HOLD_MAX_TIP_H
            if self.hold == side:
                if not on_arms:
                    self.hold = None
                else:
                    rest, on_block = self._rest()
                    limit = rest if on_block else L.OPEN
                    fit = L.theta_for_span(self.hold_u, gap_mm)
                    if fit >= limit - 1e-9:
                        self.hold = None  # the pads no longer touch the arms
                    else:
                        before = self.theta
                        self.theta = fit
                        self.yaw -= (before - self.theta) / 2.0
                continue
            span = L.outer_span(u, self.theta) if on_arms else math.inf
            # The pads' centre between the two arms, both arms between the pads (outer span, measured square to
            # the pin arm: the pads close along world x, at most 13.75 deg off that square)
            straddle = on_arms and 0.0 < w < span and gap_mm >= span - STRADDLE_TOL
            if self.hold is None and self.around[side] and on_arms and gap_mm < span:
                self.hold, self.hold_u = side, float(u)
                before = self.theta
                self.theta = min(self.theta, L.theta_for_span(self.hold_u, gap_mm))
                self.yaw -= (before - self.theta) / 2.0
            self.around[side] = bool(straddle)

        if self.hold is None:
            self._settle_free()
        for side, (tips, _) in grips.items():
            self.prev_tips[side] = tips.copy()

    def joints(self) -> dict:
        return {"free_arm": L.OPEN - self.theta, "tab": self.tab}

    def pose(self) -> tuple[np.ndarray, float]:
        return np.array([self.origin[0], self.origin[1], L.TOP]), self.yaw
