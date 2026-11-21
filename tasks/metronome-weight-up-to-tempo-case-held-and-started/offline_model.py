"""The offline stand-in for the physics (tools/offline/check_task.py and the tests only; never used in the simulator).

The checker's own proxy knows no joints. Without a simulator, task.py steps this model from the arms' commanded
TCPs, pad-tip points and openings, and scores what it gives:

1. Grips. A gripper holds the WEIGHT when its TCP is over the weight's middle (within 4 mm across base y, inside its
   20 mm depth along base x; positions in the case's base frame), the weight's centre lies along its pads (between the pad tips and 22 mm above them) and its
   pads are commanded under 18 mm apart (the weight is 16 mm across). It holds the CASE when its TCP is over the
   case's back half (within 6 mm of the middle across base y, base x 0 to 22.5 mm behind the
   centre), its pad tips are under
   the lower tier's top (60 mm) and its pads are commanded 60 to 72 mm apart (the lower tier is 70 mm across).
2. The weight's clip (clip.py in the sim: held below 3 N, sliding at 3 N). While a hand holds the weight, the rod
   cannot swing. If the other hand holds the case (or the table does, pushed down), the weight slides along the rod
   with the hand, but only once the hand has moved layout.DRAG_GIVE (14.2 mm, measured on the GPU: the arm's give
   under the clip's 3 N, the case's rise, the pads' slip and the clip's set) farther than the weight: a dead band
   either way, so a hand that stops leaves the weight 14.2 mm short of it, and a hand eased back less than 28 mm
   leaves the weight where it is. If
   nobody holds the case and the hand lifts, the clip's 3 N beats the whole metronome's 1.68 N weight: the case
   rides up with the weight (and is dropped back on the table when let go).
3. The case follows a hand that holds it (translation only); it never sinks into the table.
4. The pendulum, free: the exact equation of motion of the rod, counterweight and weight from pendulum.py (no
   damping), integrated in 20 substeps per control step. A finger (the closed pads, 8 mm either side of the TCP
   across base y, 27 mm wide along base x, from the pad tips 30 mm up) that meets the rod above the pivot or the
   weight (each a box along the rod: its depth along base x, its half width across base y, its length along the
   rod) pushes the pendulum (quasi-static: the nearest edge stays against the finger's face, at rest) until the
   finger no longer overlaps it (lifted clear, or slid off it along base x).

No friction at the table, no inertia of the case, no contact forces. Every number is a design value.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk
import pendulum as P

PAD_LEN = 0.022
FINGER_HALF = 0.008  # closed pads, either side of the TCP across y
FINGER_X = 0.0135
FINGER_SPAN = 0.030
SUB = 20


class MetronomeModel:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        self.origin = np.array(L.M_ORIGIN, dtype=float)
        self.s = 0.0  # the weight's slide (0: the 160 mark)
        self.theta = 0.0
        self.omega = 0.0
        self.prev: dict = {}
        self.holds: dict = {}  # arm -> "weight" | "case"
        self.side: dict = {}  # arm -> -1 / +1: which side of the rod a pushing finger is on
        self.give = 0.0  # the holding hand's travel along the rod beyond the weight's (the arm's give), +- DRAG_GIVE
        self.events: list = []

    # -- geometry ------------------------------------------------------------------------------------------------
    ROT = np.array([[math.cos(L.M_YAW), -math.sin(L.M_YAW), 0.0], [math.sin(L.M_YAW), math.cos(L.M_YAW), 0.0],
                    [0.0, 0.0, 1.0]])

    def local(self, p) -> np.ndarray:
        """A world point in the base frame (the case is never turned in this model, only moved)."""
        return (np.asarray(p, dtype=float) - self.origin) @ self.ROT

    def r(self) -> float:
        return P.R_START + self.s

    def weight_local(self) -> np.ndarray:
        st = self.state()
        return self.local(st.link_to_world("weight", np.array(st.spec.body_offset("weight"))))

    @staticmethod
    def _gap(arm) -> float:
        return float(np.clip(arm.opening, 0.0, 1.0)) * 0.0939

    def _holds_weight(self, arm) -> bool:
        w = self.weight_local()
        tcp, tips = self.local(arm.tcp), self.local(arm.tips)
        return (abs(tcp[1] - w[1]) <= 0.004 and abs(tcp[0] - w[0]) <= L.W_SIZE[0] / 2
                and tips[2] - 0.002 <= w[2] <= tips[2] + PAD_LEN and self._gap(arm) < L.W_SIZE[1] + 0.002)

    def _holds_case(self, arm) -> bool:
        tcp, tips = self.local(arm.tcp), self.local(arm.tips)
        gap = self._gap(arm)
        return (abs(tcp[1]) <= 0.006 and 0.0 <= tcp[0] <= 0.0225 and tips[2] < L.TIERS[0][4]
                and 0.060 <= gap <= 0.072)

    # -- one control step ------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        if not arms:
            self._free()
            return
        tcps = {a: np.asarray(arms[a].tcp, float) for a in arms}
        d = {a: (tcps[a] - self.prev[a]) if a in self.prev else np.zeros(3) for a in arms}
        # grips start or end
        for a in arms:
            held = self.holds.get(a)
            if held == "weight" and self._gap(arms[a]) >= L.W_SIZE[1] + 0.002:
                del self.holds[a]
                self.events.append((a, "let go of the weight"))
            elif held == "case" and self._gap(arms[a]) > 0.072:
                del self.holds[a]
                self.events.append((a, "let go of the case"))
            elif held is None:
                if self._holds_weight(arms[a]):
                    self.holds[a] = "weight"
                    self.give = 0.0
                    self.events.append((a, "holds the weight"))
                elif self._holds_case(arms[a]):
                    self.holds[a] = "case"
                    self.events.append((a, "holds the case"))
        case_arm = next((a for a, h in self.holds.items() if h == "case"), None)
        weight_arm = next((a for a, h in self.holds.items() if h == "weight"), None)
        if case_arm is not None:
            self.origin = self.origin + d[case_arm]
        if weight_arm is not None:
            dz = float(d[weight_arm][2]) - (float(d[case_arm][2]) if case_arm is not None else 0.0)
            if case_arm is None and dz > 0:
                self.origin = self.origin + np.array([d[weight_arm][0], d[weight_arm][1], dz])
            elif case_arm is None and dz < 0 and self.origin[2] > L.TOP + 1e-9:
                drop = min(-dz, self.origin[2] - L.TOP)
                self.origin[2] -= drop
                self._slide(dz + drop)
            else:
                self._slide(dz)
            self.s = float(np.clip(self.s, L.W_R_MIN - P.R_START, L.W_R_MAX - P.R_START))
            self.omega = 0.0
        elif case_arm is None and self.origin[2] > L.TOP:
            self.origin[2] = L.TOP  # dropped back onto the table
        self.origin[2] = max(self.origin[2], L.TOP)
        if weight_arm is None:
            self._free()
            for a in arms:
                if self.holds.get(a) is None:
                    self._push(arms[a], a)
        self.prev = tcps

    def _slide(self, dz: float) -> None:
        """The held weight with the case held (or on the table): it moves only past the arm's give."""
        self.give += dz
        if self.give > L.DRAG_GIVE:
            self.s += self.give - L.DRAG_GIVE
            self.give = L.DRAG_GIVE
        elif self.give < -L.DRAG_GIVE:
            self.s += self.give + L.DRAG_GIVE
            self.give = -L.DRAG_GIVE

    def _free(self) -> None:
        r = self.r()
        h = L.DT / SUB
        th, om = self.theta, self.omega
        for _ in range(SUB):
            k1t, k1o = om, P.accel(th, om, r)
            k2t, k2o = om + h / 2 * k1o, P.accel(th + h / 2 * k1t, 0.0, r)
            k3t, k3o = om + h / 2 * k2o, P.accel(th + h / 2 * k2t, 0.0, r)
            k4t, k4o = om + h * k3o, P.accel(th + h * k3t, 0.0, r)
            th += h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
            om += h / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
        if abs(th) > L.SWING_LIMIT:
            th, om = math.copysign(L.SWING_LIMIT, th), 0.0
        self.theta, self.omega = th, om

    def _bodies(self) -> list:
        """The pendulum's parts a finger can push: (name, base x range, half width across base y, along-rod range from
        the pivot)."""
        r = self.r()
        return [
            ("rod", (L.ROD_X - L.ROD_D / 2, L.ROD_X + L.ROD_D / 2), L.ROD_W / 2, (0.0, L.ROD_Z1)),
            ("weight", (L.ROD_X + L.W_X0, L.ROD_X + L.W_X0 + L.W_SIZE[0]), L.W_SIZE[1] / 2,
             (r - L.W_SIZE[2] / 2, r + L.W_SIZE[2] / 2)),
        ]

    @staticmethod
    def _edge_y(theta: float, side: int, w: float, srange, zlo: float, zhi: float):
        """The body's edge toward a finger on ``side`` (-1: the finger at smaller base y), at rod angle ``theta``: the
        edge point nearest the finger among those between heights zlo and zhi (from the pivot), or None."""
        best = None
        for s in np.linspace(srange[0], srange[1], 41):
            y = -s * math.sin(theta) + side * w * math.cos(theta)
            z = s * math.cos(theta) + side * w * math.sin(theta)
            if zlo <= z <= zhi:
                best = y if best is None else (min(best, y) if side < 0 else max(best, y))
        return best

    def _push(self, arm, name: str) -> None:
        tcp, tips = self.local(arm.tcp), self.local(arm.tips)
        piv = np.array([L.ROD_X, 0.0, L.PIVOT_Z])
        zlo, zhi = tips[2] - piv[2], tips[2] - piv[2] + FINGER_SPAN
        sides = self.side.setdefault(name, {})
        touched = False
        for body, (x0, x1), w, srange in self._bodies():
            if tcp[0] + FINGER_X < x0 or tcp[0] - FINGER_X > x1:
                sides.pop(body, None)
                continue
            mid = 0.5 * (srange[0] + srange[1])
            y_c = -mid * math.sin(self.theta)
            if self._edge_y(self.theta, -1, w, srange, zlo, zhi) is None:
                sides.pop(body, None)
                continue
            side = sides.get(body)
            if side is None:
                if abs(tcp[1] - y_c) < FINGER_HALF + w:
                    continue  # came down onto it from above: no side yet (the demo never does this)
                side = -1 if tcp[1] < y_c else 1
                sides[body] = side
            face = tcp[1] - piv[1] - side * FINGER_HALF  # the finger's face toward the body
            edge = self._edge_y(self.theta, side, w, srange, zlo, zhi)
            if edge is None or (side < 0 and edge >= face) or (side > 0 and edge <= face):
                continue
            # the angle at which the edge just meets the face: a finger at smaller y pushes toward +y (theta < 0)
            lim = L.SWING_LIMIT
            lo, hi = (-lim, self.theta) if side < 0 else (self.theta, lim)
            for _ in range(50):
                mid_t = 0.5 * (lo + hi)
                e = self._edge_y(mid_t, side, w, srange, zlo, zhi)
                clear = e is None or (e >= face if side < 0 else e <= face)
                if side < 0:
                    lo, hi = (mid_t, hi) if clear else (lo, mid_t)
                else:
                    lo, hi = (lo, mid_t) if clear else (mid_t, hi)
            self.theta = lo if side < 0 else hi
            self.omega = 0.0
            touched = True
        if not sides and not touched:
            self.side.pop(name, None)

    # -- what the scorer sees ----------------------------------------------------------------------------------
    def state(self):
        q = mk.yaw_quat(L.M_YAW)
        return mk.state_from_poses(self.spec, (*self.origin, *q),
                                   joints={"rod": self.theta, "weight": self.s},
                                   joint_vel={"rod": self.omega, "weight": 0.0}, source="offline")
