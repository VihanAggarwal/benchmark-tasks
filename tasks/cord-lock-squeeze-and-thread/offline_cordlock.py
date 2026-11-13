"""The lock's and the cord's kinematics for use ONLY where there is no simulator: tools/offline/check_task.py (which by
itself can carry and drop a gripped box, nothing else) and the tests. The simulator never takes a result from here:
there the squeeze, the spring, the passage, the pinch and the cord are PhysX contacts, one sprung slider and
cablekit's capsule chain.

What this file assumes is what the design claims, so the offline prediction is the design's prediction, not a
measurement (all from layout.py's geometry):

- a gripper whose pads close along x across the lock's length (41 mm with the button out) grasps it when the pads
  reach that length with the TCP within 6 mm of the lock's middle along x, within 10 mm of it along y and 8 to 26 mm
  above its underside. While held, the travel is where the finger drive (5 N per mm of each finger's error) balances
  the spring (1.5 N/mm), q = 0.625 (41 mm - command), up to the 7 mm stop; the lock is centred between the pads and
  carried with the TCP, resting on the table when the TCP goes lower;
- a gripper closing to the cord's 7.9 mm with the cord's centre 6 to 22 mm below its TCP, within 4 mm of it along x
  and 10 mm along y, grasps it; the held cord is carried rigidly with the TCP and stays straight;
- the cord's end 0 (pointing to +y) meeting the lock's near face (y -6.5 mm in the lock's frame) goes in only when it
  fits the passage: centre line within (4.4 mm + q) along x and the plunger hole's 13 mm along z, less the cord's
  radius each side, to 0.3 mm. Otherwise it goes no further than 2.7 mm into the face hole (the 2 mm face and the end's
  rounded cap), or stops at the face outside the hole. A blocked held cord pushes a loose lock away along +y; a held
  lock moved onto a loose cord pushes the cord along with it; both held, the cord slips back in the hand;
- with the cord through the passage, the plunger cannot spring back past the cord's width: a lock that no gripper
  squeezes sits at q = 3.5 mm on the cord; with nothing in it, at 0;
- a cord let go drops straight onto the table, except where it runs through the lock: there it stays at the passage's
  height, and it slopes down to the table over 30 mm either side;
- nothing springs, rolls, slides or bends otherwise.

Only the GPU runs can check those claims; see the README's watch list.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L

FULL_GAP = 0.0939  # the pads' gap fully open (rigstate's opening 1.0)
TOL = 0.0003
FACE_ENTRY = 0.0027  # a 7.9 mm end into the 11.4 mm face hole with the passage shut: the 2 mm face and 0.67 mm of cap
DROP_RAMP = 0.030


def squeeze_travel(gap_cmd: float) -> float:
    """The travel a squeeze commanded to ``gap_cmd`` (m) holds: 5 N/mm per finger against 1.5 N/mm, to the stop."""
    return float(min(L.TRAVEL, max(0.0, 0.625 * (L.LOCK_LEN_REST - gap_cmd))))


def start_line() -> np.ndarray:
    """The cord's centre line at the start: end 0, the 16 segment centres, end 1 (cablekit's layout)."""
    n = int(round(L.CORD_LEN / L.CORD_SEGMENT))
    s = [0.0, *[(i + 0.5) * L.CORD_SEGMENT for i in range(n)], L.CORD_LEN]
    z = L.TOP + L.CORD_R
    return np.array([(L.CORD_X, L.CORD_END0_Y - v, z) for v in s])


class OfflineRig:
    """Steps the lock and the cord from the arms (rigstate snapshots) once per scored step."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.origin = np.array([L.LOCK_ORIGIN[0], L.LOCK_ORIGIN[1], L.TOP])
        self.q = 0.0
        self.lock_holder: str | None = None
        self.lock_rel = np.zeros(3)
        self.line = start_line()
        self.cord_holder: str | None = None
        self.cord_rel = np.zeros_like(self.line)
        self.threaded = False
        self.prev_gap = {"left": FULL_GAP, "right": FULL_GAP}
        self.events: list[str] = []

    # -- geometry ------------------------------------------------------------------------------------------------
    def lock_len(self) -> float:
        return L.LOCK_LEN_REST - self.q

    def lock_mid_x(self) -> float:
        """World x of the middle of the lock's length (closed end to the button's end)."""
        return float(self.origin[0] + (L.LOCK_EXTENT_X[0] + L.LOCK_EXTENT_X[1] - self.q) / 2)

    def fits(self, tip_local) -> bool:
        lo, hi = L.passage_x(self.q)
        x, z = float(tip_local[0]), float(tip_local[2])
        return (x - L.CORD_R >= lo - TOL and x + L.CORD_R <= hi + TOL
                and z - L.CORD_R >= L.PLUNGER_HOLE_Z[0] - TOL and z + L.CORD_R <= L.PLUNGER_HOLE_Z[1] + TOL)

    @staticmethod
    def in_face_hole(tip_local) -> bool:
        x, z = float(tip_local[0]), float(tip_local[2])
        return (x - L.CORD_R >= L.FACE_HOLE_X[0] - TOL and x + L.CORD_R <= L.FACE_HOLE_X[1] + TOL
                and z - L.CORD_R >= L.FACE_HOLE_Z[0] - TOL and z + L.CORD_R <= L.FACE_HOLE_Z[1] + TOL)

    def in_front(self, tip_local) -> bool:
        """End 0 within the lock's outline seen along y (it can meet the near face)."""
        x, z = float(tip_local[0]), float(tip_local[2])
        return (L.LOCK_EXTENT_X[0] - L.CORD_R <= x <= L.LOCK_EXTENT_X[1] - self.q + L.CORD_R
                and -L.CORD_R <= z <= L.WIDTH + L.CORD_R)

    # -- one step ------------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        if arms is None:
            return
        for arm in ("left", "right"):
            a = arms.get(arm) if isinstance(arms, dict) else None
            if a is None:
                continue
            tcp = np.asarray(a.tcp, dtype=float)
            gap = float(np.clip(a.opening, 0.0, 1.0)) * FULL_GAP
            self._lock_hand(arm, tcp, gap)
            self._cord_hand(arm, tcp, gap)
            self.prev_gap[arm] = gap
        self._passage()

    def _lock_hand(self, arm, tcp, gap) -> None:
        if self.lock_holder == arm:
            if gap > self.lock_len() + 0.001:
                self.lock_holder = None
                self.events.append(f"{arm} let go of the lock")
                self.q = L.PINCH_Q if self.threaded else 0.0
                self.origin[2] = L.TOP
                return
            self.q = squeeze_travel(gap)
            if self.threaded:
                self.q = max(self.q, L.PINCH_Q)
            mid = tcp + self.lock_rel
            self.origin = np.array([mid[0] - (L.LOCK_EXTENT_X[0] + L.LOCK_EXTENT_X[1] - self.q) / 2, mid[1],
                                    max(L.TOP, mid[2])])
            return
        if self.lock_holder is None and gap < self.lock_len() <= self.prev_gap[arm] + 1e-9:
            dx = float(tcp[0]) - self.lock_mid_x()
            dy = float(tcp[1] - self.origin[1])
            dz = float(tcp[2] - self.origin[2])
            if abs(dx) <= 0.006 and abs(dy) <= 0.010 and 0.008 <= dz <= 0.026:
                self.lock_holder = arm
                self.lock_rel = np.array([0.0, -dy, -dz])
                self.events.append(f"{arm} squeezed the lock")

    def _cord_hand(self, arm, tcp, gap) -> None:
        if self.cord_holder == arm:
            if gap > L.CORD_D + 0.001:
                self.cord_holder = None
                self.events.append(f"{arm} let go of the cord")
                self._drop()
                return
            self.line = tcp + self.cord_rel
            return
        if self.cord_holder is None and gap < L.CORD_D + 0.0005 <= self.prev_gap[arm] + 1e-9:
            d = np.linalg.norm(self.line[:, :2] - tcp[:2], axis=1)
            k = int(np.argmin(d))
            p = self.line[k]
            if abs(p[0] - tcp[0]) <= 0.004 and abs(p[1] - tcp[1]) <= 0.010 and 0.006 <= tcp[2] - p[2] <= 0.022:
                self.cord_holder = arm
                self.cord_rel = self.line - tcp
                self.events.append(f"{arm} gripped the cord")

    def _passage(self) -> None:
        local = self.line - self.origin  # the lock's frame is the world's turned by yaw 0
        tip = local[0]
        if self.threaded and tip[1] < -L.HALF_DEPTH - 0.001:
            self.threaded = False
            self.events.append("the cord came back out")
        if not self.threaded and tip[1] > -L.HALF_DEPTH and self.in_front(tip) and tip[1] < L.HALF_DEPTH:
            if self.fits(tip):
                self.threaded = True
                self.events.append(f"the cord's end entered the passage (travel {self.q * 1000:.2f} mm)")
            else:
                allowed = FACE_ENTRY if self.in_face_hole(tip) else 0.0
                pen = float(tip[1] + L.HALF_DEPTH - allowed)
                if pen > 0:
                    self._resolve(pen)
        if self.threaded and self.lock_holder is None:
            self.q = L.PINCH_Q
        elif not self.threaded and self.lock_holder is None:
            self.q = 0.0

    def _resolve(self, pen: float) -> None:
        if self.lock_holder is not None and self.cord_holder is None:
            self.line[:, 1] -= pen  # the held lock pushes the loose cord along
        elif self.lock_holder is not None and self.cord_holder is not None:
            self.line[:, 1] -= pen  # both held: the cord slips back in the hand
            self.cord_rel[:, 1] -= pen
        else:
            self.origin[1] += pen  # the held (or pushed) cord shoves the loose lock away

    def _drop(self) -> None:
        bottom = L.TOP + L.CORD_R
        if not self.threaded:
            self.line[:, 2] = bottom
            return
        local = self.line - self.origin
        inside = np.abs(local[:, 1]) <= L.HALF_DEPTH
        z_pass = float(np.mean(self.line[inside, 2])) if np.any(inside) else float(self.line[0, 2])
        out = np.maximum(np.abs(local[:, 1]) - L.HALF_DEPTH, 0.0)
        self.line[:, 2] = bottom + (z_pass - bottom) * np.maximum(0.0, 1.0 - out / DROP_RAMP)

    # -- states for the scorer -------------------------------------------------------------------------------------
    def cord_arrays(self):
        """(segment centres, start_dir, end_dir, seg_len) as cablekit's CableState holds them."""
        pts = self.line[1:-1].copy()
        sd = self.line[0] - self.line[1]
        ed = self.line[-1] - self.line[-2]
        return pts, sd / np.linalg.norm(sd), ed / np.linalg.norm(ed), L.CORD_SEGMENT
