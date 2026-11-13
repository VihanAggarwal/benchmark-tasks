"""The clip's kinematics for use ONLY where there is no simulator: tools/offline/check_task.py (which by itself can
carry and drop a gripped box, nothing else) and the tests. The simulator never takes a result from here: there the
pinch, the spring, the cushions, the stop and the hold on the finger are PhysX contacts and one sprung joint.

What this file assumes is what the design claims, so the offline prediction is the design's prediction, not a
measurement (all from layout.py's geometry):

- a gripper (straight down, fingers closing along y, as the demo's) whose pads are either side of the oximeter and
  overlap its grip pads in height presses its two shells. Its pads bear on the grip pads when the TCP is between
  10.5 mm behind the back end and 23.5 mm ahead of it (t, measured from the back end); further forward they bear on
  the body only, which holds the jaws shut;
- with the pads on the grip pads the pivot angle is where the finger drive (5 N per mm of each finger's error, 50 N
  cap) balances the spring through the grip pad's lever, up to the open stop (28 deg), and no more than the pad's
  position allows: a pad whose front edge is e mm ahead of the pivot also bears on the body (e = t - 6.5): 28 deg up
  to e = -1.2, 22 at 2.9, 18.9 at 5.0, 16.3 at 7.5, 12.1 at 13.5, falling to 9.7 at 17 (the spec's tapered-pad
  numbers, joined by straight lines);
- each shell turns about its grip pad's front edge, which stays on the gripper pad, so the pivot sits the pad's lever
  (10 mm shut, 14.2 mm fully open) ahead of that edge: the unit slides forward 4.2 mm as it opens; it is otherwise
  carried rigidly with the TCP, its centre line along x;
- carried onto the finger, the unit goes on until the fingertip meets the stop or the finger is wider than the gap
  between the cushions or the lips (``layout.deepest_s``), and no further: the hand's further travel loads the wrist
  ("over-travel"); with the fingertip more than 3.5 mm off the centre line along the pivot, or more than the mouth's
  play across it, the lips butt on the fingertip;
- let go (no gripper pressing it) with the fingertip 10 mm or more inside and the unit along the finger, the
  cushions close on the finger at 16.3 deg and the unit drops 3.5 mm until its upper side wall rests on the finger;
  anywhere else it falls flat onto the table where it is, jaws shut;
- nothing springs, swings or slips otherwise.

Only the GPU runs can check those claims; see the README's watch list.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

PAD_SPAN = (-0.025, 0.013)  # a pad from 25 mm above the TCP to its tip 13 mm below it
GRIP_PAD_Z = 0.010  # the grip pads span Z -10 to +10 mm about the unit's centre
REAR_T = -10.5  # mm: the pad's front edge overlaps the grip pad by 3 mm
BODY_T = 23.5  # mm: from here forward the pad misses the grip pads
E_TABLE = ((-1.2, 28.0), (2.9, 22.0), (5.0, 18.9), (7.5, 16.3), (13.5, 12.1), (17.0, 9.65))
SAG = 0.0035  # the unit drops this far onto the finger once let go
W_PLAY = 3.5  # mm: the chamber's play along the pivot (22 mm between the side walls, 15 mm finger)
HOLD = math.radians(L.HOLD_DEG)


def q_max_for(e: float) -> float:
    """The largest pivot angle (rad) a pad whose front edge is ``e`` mm ahead of the pivot allows."""
    if e <= E_TABLE[0][0]:
        return L.OPEN_LIMIT
    for (e0, q0), (e1, q1) in zip(E_TABLE, E_TABLE[1:]):
        if e <= e1:
            return math.radians(q0 + (q1 - q0) * (e - e0) / (e1 - e0))
    return math.radians(E_TABLE[-1][1])


class OfflineClip:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pivot = np.array([L.START[0] + L.PIVOT_X, L.START[1], L.TOP + L.START[2]])  # world
        self.q = 0.0
        self.mode = "rest"  # rest | held | on_finger
        self.holder: str | None = None
        self.rel: np.ndarray | None = None  # held: pivot minus (TCP + lever along x), at the grasp
        self.t: float | None = None  # held: the TCP's place along the shells, mm from the back end (kept)
        self.overtravel = 0.0  # mm the hand is commanded past where the unit stopped

    # -- geometry ---------------------------------------------------------------------------------------------
    def origin(self) -> np.ndarray:
        c, s = math.cos(self.yaw()), math.sin(self.yaw())
        return self.pivot - np.array([c * L.PIVOT_X, s * L.PIVOT_X, 0.0])

    def yaw(self) -> float:
        return -self.q / 2  # the base (battery) shell, its centre line along +x

    def _local(self, point) -> tuple[float, float, float]:
        """A world point in the unit's centre-line frame: mm along it from the pivot, across, along the pivot."""
        d = (np.asarray(point, dtype=float) - self.pivot) * 1000.0
        return float(d[0]), float(d[1]), float(d[2])

    def _contact(self, side: str, arm) -> tuple[str, float] | None:
        """('pads' or 'body', t) when this gripper presses the shells, else None; t is the TCP's place along the
        shells in mm from the back end (kept from the grasp while the same gripper holds it)."""
        tcp, tips, opening = arm
        gap = float(np.clip(opening, 0.0, 1.0)) * L.FULL_GAP * 1000.0
        x, y, z = self._local(tcp)
        if abs(y) > 6.0:
            return None
        lo, hi = float(tips[2]), float(tips[2]) - PAD_SPAN[0] + PAD_SPAN[1]
        if min(hi, self.pivot[2] + GRIP_PAD_Z) - max(lo, self.pivot[2] - GRIP_PAD_Z) < 0.003:
            return None
        if self.mode == "held" and self.holder == side and self.t is not None:
            t = self.t
        else:
            t = x + L.PIVOT_U_MM * math.cos(self.q / 2)  # TCP from the back end, along the centre line
        if t < REAR_T or t - 13.5 > L.UNIT_LEN * 1000.0:
            return None
        a = self.q / 2
        if t < BODY_T:
            kind, across = "pads", L.pads_across(a)
        else:
            kind, across = "body", 2 * (16.0 * math.cos(a) + max(0.0, t - L.PIVOT_U_MM) * math.sin(a))
        if gap >= across - 1e-9:
            return None
        return kind, t

    def _finger_floor(self) -> float:
        """Pressed shut on the finger, the jaws cannot close past where both cushions touch it."""
        depth = L.lip_line(self.q / 2) - self._local(L_P)[0]
        return HOLD if depth >= L.OVER_DEPTH_MM and self.mode != "rest" else 0.0

    # -- one control step ---------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        contact = None
        if read is not None:
            for side, arm in zip(rigstate.ARMS, read):
                c = self._contact(side, arm)
                if c is not None and (self.holder in (None, side)):
                    contact = (side, arm, c)
                    break
        if contact is None:
            if self.mode == "held":
                self._let_go()
            self.holder = None
            return
        side, (tcp, tips, opening), (kind, t) = contact
        gap = float(np.clip(opening, 0.0, 1.0)) * L.FULL_GAP * 1000.0
        floor = self._finger_floor()
        if kind == "pads":
            q = min(L.finger_force_gap(gap), q_max_for(t - 6.5))
        else:
            q = 0.0
        q = max(q, floor)
        if self.mode != "held" or self.holder != side:
            self.mode, self.holder, self.t = "held", side, t
            lever = L.pad_lever(self.q / 2) * L.MM if kind == "pads" else 0.0
            self.rel = self.pivot - (np.asarray(tcp, dtype=float) + np.array([lever, 0.0, 0.0]))
        self.q = q
        lever = L.pad_lever(q / 2) * L.MM if kind == "pads" else 0.0
        want = np.asarray(tcp, dtype=float) + np.array([lever, 0.0, 0.0]) + self.rel
        want[2] = max(want[2], L.TOP + L.UNIT_W / 2)  # cannot be pressed into the table
        self.pivot = self._block(want)

    def _block(self, want: np.ndarray) -> np.ndarray:
        """Where the unit stops on its way to ``want``: on the stop, jammed on the finger, or butting on its lips."""
        a = self.q / 2
        s = (L.P[0] - (want[0])) * 1000.0
        v = (L.P[1] - want[1]) * 1000.0
        w = (L.TOP + L.P[2] - want[2]) * 1000.0
        lip = L.lip_line(a)
        self.overtravel = 0.0
        if s >= lip:  # the fingertip is not yet at the lips
            return want
        if abs(w) > W_PLAY + 1e-6 or abs(v) > max(0.0, L.half_gap_lips(a) - 5.0) + 1e-6:
            s_min = lip  # butts on the lips or the shell's face
        else:
            s_min = L.deepest_s(a)
        if s >= s_min:
            return want
        self.overtravel = s_min - s
        out = want.copy()
        out[0] = L.P[0] - s_min * L.MM
        return out

    def _let_go(self) -> None:
        depth = L.lip_line(self.q / 2) - self._local(L_P)[0]
        x, y, z = self._local(L_P)
        if depth >= L.OVER_DEPTH_MM and abs(y) <= 5.0 and abs(z) <= W_PLAY + 1e-6:
            self.mode = "on_finger"
            self.q = HOLD
            self.pivot = self.pivot + np.array([0.0, 0.0, -SAG])
        else:
            self.mode = "rest"
            self.q = 0.0
            self.pivot = np.array([self.pivot[0], self.pivot[1], L.TOP + L.START[2]])
        self.rel = None
        self.t = None
        self.overtravel = 0.0

    def pose(self) -> tuple[np.ndarray, float, float]:
        """(base origin, base yaw, pivot angle) for mechkit.state_from_poses."""
        return self.origin(), self.yaw(), self.q


L_P = np.array([L.P[0], L.P[1], L.TOP + L.P[2]])
