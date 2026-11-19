"""The offline stand-in for the physics (tools/offline/check_task.py only; never used in the simulator).

The checker's proxy knows no sprung cups or boom. Without a simulator, task.py steps this model from the arms'
pad tips and openings, and scores what it gives:

1. Grips. A gripper whose pad-tip point lies within 8 mm (x), 12 mm (y) and 12 mm (z) of a cup shell's centre less
   5 mm, with its pads commanded under the shell's 60 mm plus 2 mm, holds that cup; the cup then follows the pad
   tips (translation only).
2. Both cups held: the band rides between them (its origin at the cups' mean x, y; at the cups' height plus 90 mm),
   each cup's slide is how far its hand has pulled it out from rest (0 to 35 mm). On the stand the band cannot sink
   under the cradle; over the head, a cushion pressed into its ear stops at the ear's outer face.
3. One cup held: the other hangs at rest (slide 0). If that free cushion, or any cup, would pass into the head's
   cylinder or an ear, the hand slips and the headset falls: it lands upright on its cups on the table top where it
   was (off the stand, not on the head: the scorer reads 0.25 unless it is back over the cradle).
4. Let go: within 20 mm of the seated pose (xy) and 12 mm of its height, with both cushions at or outside the ears,
   it seats (cups on the ears, band on the crown). Near its stand rest, it hangs on the stand again. Anywhere else
   it falls to the table.
5. The boom: a gripper with its pads under 20 mm apart, its pad tips within 16 mm of the boom's plane (y) and 20 to
   105 mm from the boom's hinge, pushes the boom forward when it is at or ahead of the boom's angle; once 0.05 rad
   past upright the boom falls to its lower stop. A gripper never lifts the boom back.
No friction, no inertia, no contact forces. Every number is a design value, not a measurement.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk
import rigstate

CUPS = ("cup_p", "cup_n")
SIGN = {"cup_p": 1.0, "cup_n": -1.0}
SHELL_YC = (L.SHELL_Y[0] + L.SHELL_Y[1]) / 2
Q0 = (0.0, 0.0, 0.0, 1.0)
PAD_LEAD = -0.013  # closed pads: their front face 13 mm ahead (toward -x) of the pad-tip point


class HeadsetModel:
    def __init__(self, headset_spec, head_spec, stand_spec) -> None:
        self.headset_spec, self.head_spec, self.stand_spec = headset_spec, head_spec, stand_spec
        self.reset()

    def reset(self) -> None:
        self.origin = np.array(L.HEADSET_ORIGIN_START, dtype=float)
        self.q = {"cup_p": 0.0, "cup_n": 0.0, "boom": 0.0}
        self.mode = "stand"  # stand | held | seated | fallen
        self.holds: dict = {}  # arm -> cup
        self.slips = 0

    def cup_centre(self, cup: str) -> np.ndarray:
        s = SIGN[cup]
        return self.origin + np.array([0.0, s * (SHELL_YC + self.q[cup]), L.CUP_ZC])

    def _grip_point(self, cup: str) -> np.ndarray:
        return self.cup_centre(cup) + np.array([0.0, 0.0, L.GRIP_DZ])

    def _gap(self, arm) -> float:
        return float(arm.opening) * rigstate.FULL_GAP

    def _hits_head(self, origin, q) -> bool:
        """A cup's inner face inside the head's cranium (radius 70 mm) or past an ear's outer face, at these
        values, with the cup level with the head (it overlaps z 85 to 160 mm and x within reach of the head)."""
        for cup in CUPS:
            s = SIGN[cup]
            cz = origin[2] - L.TOP + L.CUP_ZC
            dx = abs(origin[0] - L.HX)
            if not (cz - L.CUP_R < 0.160 and cz + L.CUP_R > 0.085 and dx < L.CUP_R + 0.070):
                continue
            inner = s * (origin[1] - L.HY) + L.REST_INNER + q[cup]  # the inner face's distance out on its side
            lim = L.EAR_Y1 if dx < L.CUP_R + L.EAR_X / 2 else 0.070
            if inner < lim - 1e-6:
                return True
        return False

    @staticmethod
    def _on_table(o) -> np.ndarray:
        """Where a fallen headset rests: upright on its cups' rims on the table top, below where it was let go."""
        return np.array([o[0], o[1], L.TOP - (L.CUP_ZC - L.CUP_R)])

    def step(self, arms) -> None:
        if self.mode == "fallen" or arms is None:
            return
        held = {}
        for a in ("left", "right"):
            arm = arms.get(a) if isinstance(arms, dict) else None
            if arm is None:
                continue
            tips, gap = np.asarray(arm.tips, float), self._gap(arm)
            if a in self.holds and gap <= 2 * L.CUP_R + 0.002:
                held[a] = self.holds[a]
                continue
            if gap > 2 * L.CUP_R + 0.002:
                continue
            for cup in CUPS:
                if cup in held.values():
                    continue
                d = tips - self._grip_point(cup)
                if abs(d[0]) <= 0.008 and abs(d[1]) <= 0.012 and abs(d[2]) <= 0.012 and gap > 0.030:
                    held[a] = cup
                    break
        self.holds = held
        tips = {a: np.asarray(arms[a].tips, float) for a in held}
        if len(held) == 2:
            by_cup = {cup: tips[a] - np.array([0.0, 0.0, L.GRIP_DZ]) for a, cup in held.items()}
            cp, cn = by_cup["cup_p"], by_cup["cup_n"]
            o = np.array([(cp[0] + cn[0]) / 2, (cp[1] + cn[1]) / 2, (cp[2] + cn[2]) / 2 - L.CUP_ZC])
            half = (cp[1] - cn[1]) / 2
            sl = float(np.clip(half - SHELL_YC, 0.0, L.CUP_UPPER))
            q = {"cup_p": sl, "cup_n": sl, "boom": self.q["boom"]}
            if self.mode == "stand" and o[2] <= L.TOP + L.STAND_TOP + 0.002:
                o[2] = L.TOP + L.STAND_TOP
                o[0], o[1] = L.SX, L.SY
            else:
                self.mode = "held"
            near_seat = np.linalg.norm(o[:2] - L.HEADSET_ORIGIN_SEATED[:2]) < 0.02 and \
                abs(o[2] - L.HEADSET_ORIGIN_SEATED[2]) < 0.015
            if near_seat:
                q["cup_p"] = q["cup_n"] = max(sl, L.Q_SEAT)
                o[2] = max(o[2], L.HEADSET_ORIGIN_SEATED[2])
            if self._hits_head(o, q):
                self.mode, self.slips = "fallen", self.slips + 1
                self.origin = self._on_table(o)
                return
            self.origin, self.q = o, q
        elif len(held) == 1:
            (a, cup), = held.items()
            c = tips[a] - np.array([0.0, 0.0, L.GRIP_DZ])
            q = {k: 0.0 for k in CUPS}
            q["boom"] = self.q["boom"]
            o = c - np.array([0.0, SIGN[cup] * SHELL_YC, L.CUP_ZC])
            if self.mode == "stand" and o[2] <= L.TOP + L.STAND_TOP + 0.002:
                return  # one hand only takes the slack: it still hangs on the stand
            if self.mode == "seated":
                return  # the clamp holds it: one hand touching a cup does not move it in this model
            self.mode = "held"
            if self._hits_head(o, q):
                self.mode, self.slips = "fallen", self.slips + 1
                self.origin = self._on_table(o)
                return
            self.origin, self.q = o, q
        else:
            if self.mode == "held":
                o = self.origin
                if np.linalg.norm(o[:2] - L.HEADSET_ORIGIN_SEATED[:2]) < 0.02 and \
                        abs(o[2] - L.HEADSET_ORIGIN_SEATED[2]) < 0.012 and \
                        min(self.q["cup_p"], self.q["cup_n"]) >= L.Q_SEAT - 1e-6:
                    self.mode = "seated"
                    self.origin = np.array(L.HEADSET_ORIGIN_SEATED, dtype=float)
                    self.q["cup_p"] = self.q["cup_n"] = L.Q_SEAT
                elif np.linalg.norm(o - np.array(L.HEADSET_ORIGIN_START)) < 0.02:
                    self.mode = "stand"
                    self.origin = np.array(L.HEADSET_ORIGIN_START, dtype=float)
                    self.q["cup_p"] = self.q["cup_n"] = 0.0
                else:
                    self.mode = "fallen"
                    self.origin = self._on_table(o)
        self._boom(arms)

    def boom_pivot(self) -> np.ndarray:
        return self.origin + np.array([L.BOOM_PIVOT[0], L.BOOM_PIVOT[1] - self.q["cup_n"], L.BOOM_PIVOT[2]])

    def _boom(self, arms) -> None:
        if self.mode == "fallen" or self.q["boom"] <= L.BOOM_DOWN + 1e-9:
            return
        piv = self.boom_pivot()
        theta = -self.q["boom"]  # forward (toward -x) from upright
        for a in ("left", "right"):
            arm = arms.get(a) if isinstance(arms, dict) else None
            if arm is None or self._gap(arm) > 0.020 or a in self.holds:
                continue
            t = np.asarray(arm.tips, float)
            if abs(t[1] - (piv[1] - 0.002)) > 0.016:
                continue
            dx, dz = t[0] - piv[0] + PAD_LEAD, t[2] - piv[2]  # the pads' front face leads the tip point
            r = math.hypot(dx, dz)
            if not 0.020 <= r <= 0.105:
                continue
            th = math.atan2(-dx, dz)
            if theta - 0.15 <= th <= theta + 0.30:
                theta = max(theta, th + 0.02)
        if theta > 0.05:
            self.q["boom"] = L.BOOM_DOWN
        else:
            self.q["boom"] = -max(0.0, theta)

    def states(self) -> dict:
        hs = mk.state_from_poses(self.headset_spec, (*self.origin, *Q0), joints=dict(self.q), source="offline")
        head = mk.state_from_poses(self.head_spec, (*self.head_spec.pos, *Q0), source="offline")
        stand = mk.state_from_poses(self.stand_spec, (*self.stand_spec.pos, *Q0), source="offline")
        return {"headset": hs, "head": head, "stand": stand}
