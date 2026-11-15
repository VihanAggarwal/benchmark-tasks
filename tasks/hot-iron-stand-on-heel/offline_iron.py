"""Kinematics of the iron for use ONLY where there is no simulator: tools/offline/check_task.py (whose own carry proxy
never picks up a part wider than the open gripper, and never tips one) and the tests. On the GPU the iron is a free
rigid body and nothing here runs.

What this file assumes is what the design claims, so the offline prediction is the design's, not a measurement. It
moves the iron from the arms' COMMANDED TCP, pad-tip point and opening (rigstate's offline feed):

- a gripper holds the iron once its pad tips have straddled the handle open (the pad-tip point within 15 mm of the
  handle's middle plane, which the closing pads then centre between them; inside the grip bar 10 mm or more from
  its ends at the bar's height, or inside the front post (the top of the handle loop when the iron stands on its
  heel) 10 mm or more from its ends) and then close below the handle's 28 mm; it lets go once open wider than
  29.5 mm;
- held, the iron hangs on the pinch as on a pin across its bar (the GPU run of 2026-11-14 showed the pads do not
  resist a turn about their own axis): its heading and roll stay as gripped, and its nose pitch is set by where the
  pin is, not by the hand's pitch. With the pin no higher over what the heel corner is over than on the flat iron,
  the iron lies flat (pressed); up to the pin's distance from the heel corner, the heel corner rests there and the
  iron is pitched about it; higher, it hangs with its centre of mass under the pin. The pin is 11 mm past the TCP
  along the approach (the bar's centre line); its place on the iron is taken at the grasp;
- let go, it settles by its centre of mass (layout.COM, mass spread over the boxes): nose raised past the balance
  angle (layout.BALANCE_DEG, about 66.5 degrees) about its heel corner, it rocks onto its heel; otherwise it falls
  back face down about the heel corner (or drops face down when it was let go level). Its heading is kept;
- not held, a closed gripper (pads one block, opening under 4 mm) whose block face (PAD_FACE from the TCP, across
  the approach, toward the iron's -y) comes from outside onto the shell's +y side face (over its length and height,
  give or take half the 27 mm pad) pushes the iron along its -y axis: the face never stays inside the side. Its
  pose is kept (standing or lying) and it sits on what is under its heel's middle (standing) or its sole's middle
  (lying): pushed off the rest it drops 5 mm onto the cover. The design claims (task.py PUSH_*) that the push is low
  enough not to tip it; this model takes that as given.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

BAR_W = L.HANDLE_W * L.MM
PIN_PAST_TCP = 0.011  # the pinch axis (the bar's centre line) past the TCP along the approach
HANG_MM = 3.0  # the pin this much farther over the heel's support than the pin's reach: the heel leaves it
CAPTURE_MM = 15.0  # pads closing within this of the handle's middle plane centre it between them (the open pads'
# inner faces are 47 mm off their middle, the handle 14 mm: real capture about 33 mm; 15 kept well inside that)
GRASP_GAP, RELEASE_GAP = BAR_W - 0.0001, BAR_W + 0.0015
PUSH_GAP = 0.004  # "closed" for a pusher
PAD_FACE = 0.004  # a closed block's face from the TCP across the approach (GPU-measured, task.PAD_FACE)
PAD_HALF = 13.5  # mm: half the pad's width
SIDE_Y = L.SHELL[2] / 2  # the shell's side face, mm off the iron's middle plane
PUSH_DEPTH_MM = 30.0  # a face deeper inside the side than this did not come from the side: no push
_CORNERS = L.iron_corners_mm() * L.MM


def heading(r) -> float:
    y = np.asarray(r)[:, 1]
    return math.atan2(-float(y[0]), float(y[1]))


def nose_pitch_deg(r) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, float(np.asarray(r)[2, 0])))))


class OfflineIron:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self.o = np.array([L.START_O[0], L.START_O[1], L.TOP + L.START_O[2]], dtype=float)
        self.r = np.eye(3)
        self.held = None
        self.around = {s: False for s in rigstate.ARMS}
        self.outside = {s: False for s in rigstate.ARMS}
        self.grip = None  # (the pin in the iron frame (m), heading, roll matrix)
        self.phi = 0.0
        self.events: list = []
        self.steps = 0

    # -- helpers ------------------------------------------------------------------------------------------------------
    def _near_bar(self, tips) -> bool:
        p = self.r.T @ (np.asarray(tips, dtype=float) - self.o) / L.MM
        if abs(p[1]) > CAPTURE_MM:
            return False
        x0, x1, z0, z1 = L.BAR
        if x0 + 10.0 <= p[0] <= x1 - 10.0 and z0 - 2.0 <= p[2] <= z1 + 2.0:
            return True
        x0, x1, z0, z1 = L.FRONT_POST
        return x0 - 2.0 <= p[0] <= x1 + 2.0 and z0 + 10.0 <= p[2] <= z1 - 10.0

    def _lift_clear(self) -> None:
        """Raise the iron so no corner is inside what lies under it."""
        pts = self.o + _CORNERS @ self.r.T
        lift = max(L.support_top(float(p[0]), float(p[1])) + L.TOP - float(p[2]) for p in pts)
        if lift > 0.0:
            self.o = self.o + np.array([0.0, 0.0, lift])

    def _settle(self) -> None:
        phi = nose_pitch_deg(self.r)
        yaw = heading(self.r)
        if phi >= L.BALANCE_DEG:
            self.r = L.pitched(90.0, yaw)
            heel = self.o + self.r @ (L.HEEL_CENTRE * L.MM)
            self.o = np.array([self.o[0], self.o[1], L.TOP + L.support_top(float(heel[0]), float(heel[1]))])
            kind = "rocked onto its heel"
        else:
            self.r = L.pitched(0.0, yaw)
            c = self.o + self.r @ np.array([L.SOLE_LEN / 2 * L.MM, 0.0, 0.0])
            self.o = np.array([self.o[0], self.o[1], L.TOP + L.support_top(float(c[0]), float(c[1]))])
            kind = "fell back face down" if phi > 2.0 else "set down face down"
        self.events.append((self.steps, "settle", kind, round(phi, 1)))

    # -- one control step ---------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        self.steps += 1
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(np.clip(opening, 0.0, 1.0)) * rigstate.FULL_GAP
            if self.held == side:
                if gap > RELEASE_GAP:
                    self.held, self.grip = None, None
                    self.events.append((self.steps, "release", side))
                    self._settle()
                continue
            if self.held is None:
                self._push(side, tcp, tips, gap)
            near = self._near_bar(tips)
            if near and self.around[side] and gap < GRASP_GAP and self.held is None:
                self.held = side
                off = (self.r.T @ (np.asarray(tips, dtype=float) - self.o))[1]
                self.o = self.o + self.r[:, 1] * off  # the closing pads centre the handle between them
                pin = self._pin(tcp, tips)
                yaw = heading(self.r)
                self.phi = nose_pitch_deg(self.r)
                tilt = L.pitched(self.phi, yaw).T @ self.r  # what is left after heading and pitch (roll)
                self.grip = (self.r.T @ (pin - self.o), yaw, tilt)
                self.events.append((self.steps, "grasp", side))
            if not near:
                self.around[side] = False
            elif gap > GRASP_GAP + 0.004:
                self.around[side] = True
        if self.held is not None:
            tcp, tips, _ = read[rigstate.ARMS.index(self.held)]
            self._hang(self._pin(tcp, tips))
            self._lift_clear()

    def _push(self, side, tcp, tips, gap) -> None:
        """A closed block's face on the shell's +y side pushes the iron along its -y axis (see the module text)."""
        if gap >= PUSH_GAP:
            self.outside[side] = False
            return
        tcp, tips = np.asarray(tcp, dtype=float), np.asarray(tips, dtype=float)
        a = tips - tcp
        a = a / (float(np.linalg.norm(a)) or 1.0)
        d = -self.r[:, 1]
        n = d - float(d @ a) * a
        if float(np.linalg.norm(n)) < 0.5:
            self.outside[side] = False
            return
        face = tips + PAD_FACE * n / float(np.linalg.norm(n))
        p = self.r.T @ (face - self.o) / L.MM
        x0, x1, _, z0, z1 = L.SHELL
        over = x0 - PAD_HALF <= p[0] <= x1 + PAD_HALF and z0 - PAD_HALF <= p[2] <= z1 + PAD_HALF
        if not over or p[1] >= SIDE_Y:
            self.outside[side] = over
            return
        if not self.outside[side] or p[1] < SIDE_Y - PUSH_DEPTH_MM:
            return
        self.o = self.o + self.r[:, 1] * (p[1] - SIDE_Y) * L.MM
        upright = nose_pitch_deg(self.r) > 45.0
        mid = self.o + self.r @ ((L.HEEL_CENTRE if upright else np.array([L.SOLE_LEN / 2, 0.0, 0.0])) * L.MM)
        self.o[2] = L.TOP + L.support_top(float(mid[0]), float(mid[1]))
        if not self.events or self.events[-1][1] != "push":
            self.events.append((self.steps, "push", side))

    @staticmethod
    def _pin(tcp, tips) -> np.ndarray:
        a = np.asarray(tips, dtype=float) - np.asarray(tcp, dtype=float)
        a = a / (float(np.linalg.norm(a)) or 1.0)
        return np.asarray(tcp, dtype=float) + PIN_PAST_TCP * a

    def _hang(self, pin) -> None:
        """Pose the held iron on the pin: heel corner on what it is over, flat, or hanging (see the module text)."""
        p_loc, yaw, tilt = self.grip
        px, pz = float(p_loc[0]), float(p_loc[2])
        rad, beta = math.hypot(px, pz), math.atan2(pz, px)
        h = float(pin[2]) - (L.TOP + L.support_top(float(self.o[0]), float(self.o[1])))
        if h <= pz:
            phi = 0.0
        elif h > rad + HANG_MM * L.MM:
            c = L.COM * L.MM - p_loc  # the centre of mass from the pin, iron frame (below it and behind)
            phi = math.degrees(math.atan(float(c[0]) / float(c[2])))  # turned until it is under the pin: 22.6
        else:  # the heel corner on its support, on the side of the pin nearest where it was (it does not jump)
            d = math.sqrt(max(0.0, rad * rad - h * h))
            u = np.array([math.cos(yaw), math.sin(yaw)])
            back = float(np.linalg.norm(np.asarray(pin[:2]) - d * u - self.o[:2]))
            ahead = float(np.linalg.norm(np.asarray(pin[:2]) + d * u - self.o[:2]))
            phi = math.degrees(math.atan2(h, d if back <= ahead else -d) - beta)
        self.phi = phi
        self.r = L.pitched(phi, yaw) @ tilt
        self.o = np.asarray(pin, dtype=float) - self.r @ p_loc

    def pose7(self) -> tuple:
        import mechkit as mk

        return (*self.o, *mk.matrix_to_quat(self.r))

    def state(self):
        import mechkit as mk

        return mk.state_from_poses(self.spec, self.pose7(), np.zeros(6), source="offline")
