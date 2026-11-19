"""The offline stand-in for the physics (tools/offline/check_task.py only; never used in the simulator).

The checker's proxy knows no prongs, hooks or webbing. Without a simulator, task.py steps this model from the arms'
commanded TCPs, pad tips and openings, and scores what it gives:

1. Grips. A gripper whose pad-tip point lies on a half's centre line (within 6 mm across, along its length, at its
   height) with its pads commanded under that half's width plus 2 mm holds it; the half then follows the TCP
   rigidly (translation only, the demo never turns a held part) until the pads open past that width.
2. Tabs. A gripper holding the female presses each prong's tab that lies within its pad's 20 mm (plus the tab's
   8 mm): the tab goes in by (32 mm - max(pad gap, 29 mm)) / 2, at most the 1.5 mm to flush; the prong turns by
   that over its 18 mm lever. The hook clears its rib once both prongs turn RELEASE_RAD (the tab 0.89 mm in).
3. The buckle. Clicked (male within 0.3 mm of home) and either prong short of RELEASE_RAD: the two halves move as
   one, following whichever is held; if both are held by different hands and the hands part by over 2 mm, the hand
   on the male slips off. Hooks clear: the male slides along the female's axis toward the hand that holds it; once
   26.3 mm out no part of it is inside, and the two halves are free of each other. Let go while still in with the
   hooks back behind their ribs (under 0.8 mm out): clicked again.
4. The webbing: 14 nodes (the female's back, 12 hinges, the male's bar) 25 mm apart at most (a rope: never
   stretched, free to bunch), plus each free half as a rigid segment (34 and 48.3 mm). Every step the free nodes
   sink 4 mm (gravity, quasi-static), then 40 passes of the length rules and of pushing nodes out of the neck, head
   and chest cylinders and up onto the table. A half that is neither held nor free (clicked, nobody holding it)
   stays where it is. If the lengths cannot be met to within 4 mm (a closed collar pulled further than its loop
   reaches round the neck), the hand on the male slips off. No friction, no inertia, no contact forces.

The manikin never moves in this model. Every number is a design value, not a measurement.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk
import rigstate

GRAVITY_STEP = 0.004
PASSES = 40
SLIP_EXCESS = 0.004
NODE_R_WEB = L.WEB_T / 2
NODE_R_F = L.F_H / 2
NODE_R_M = (L.M_Z1 - L.M_Z0) / 2
# cylinders along x the webbing can lie on (table frame): (x0, x1, y, z, r)
CYLS = [(L.NECK_X0, L.NECK_X1, 0.0, L.NECK_Z, L.NECK_R), (L.HEAD_X0, L.HEAD_X1, 0.0, L.HEAD_R, L.HEAD_R),
        (0.380, 0.770, -0.020, 0.127, 0.127)]


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roll_of(u) -> float:
    return math.atan2(float(u[2]), float(u[1]))


B0 = np.array(L.B_ORIGIN, dtype=float)
F_BACK = np.array([0.0, L.F_Y1, L.END_Z])  # female points, buckle frame
F_MOUTH = np.array([0.0, L.F_Y0, L.END_Z])
M_BAR = np.array([0.0, L.M_Y0, L.END_Z])
M_TIP = np.array([0.0, L.M_TIP_Y, L.END_Z])
TAB_C = (L.TAB_Y[0] + L.TAB_Y[1]) / 2


class CollarModel:
    def __init__(self, collar_spec, male_spec, dog_spec) -> None:
        self.collar_spec, self.male_spec, self.dog_spec = collar_spec, male_spec, dog_spec
        self.rest_roll = [L.link_angle(L.VERTS[i - 1], L.VERTS[i]) for i in range(1, L.N_LINKS + 1)]
        self.reset()

    def reset(self) -> None:
        self.f_rot = np.eye(3)
        self.f_o = B0.copy()
        self.m_rot = np.eye(3)
        self.m_o = B0.copy()
        top = np.array([0.0, 0.0, L.TOP])
        self.nodes = np.array([v + top for v in L.VERTS[1:-1]])  # the 12 hinges, world
        self.mode = "clicked"  # clicked | sliding | apart
        self.prongs = [0.0, 0.0]
        self.holds: dict = {}  # arm -> (part, offset)
        self.slips = 0

    # -- geometry helpers -------------------------------------------------------------------------------------------
    def _f_pt(self, p):
        return self.f_o + self.f_rot @ p

    def _m_pt(self, p):
        return self.m_o + self.m_rot @ p

    def slide(self) -> float:
        return -float(((self.m_o - self.f_o) @ self.f_rot)[1])

    def _straddles(self, part, tip, gap) -> bool:
        rot, o = (self.f_rot, self.f_o) if part == "female" else (self.m_rot, self.m_o)
        lx, ly, lz = (np.asarray(tip) - o) @ rot
        if part == "female":
            return (abs(lx) <= 0.006 and L.F_Y0 - 0.002 <= ly <= L.F_Y1 and -0.010 <= lz <= L.F_H + 0.002
                    and gap <= 2 * L.F_HALF_W + 0.002)
        return (abs(lx) <= 0.006 and L.M_Y0 - 0.004 <= ly <= L.M_Y1 + 0.004 and -0.010 <= lz <= 0.016
                and gap <= 2 * L.M_HALF_W + 0.002)

    # -- one control step -----------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        grips = {}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(opening) * rigstate.FULL_GAP
            grips[side] = (np.asarray(tcp, float), np.asarray(tips, float), gap)
        # let go of what the pads opened from
        for side in list(self.holds):
            part, _ = self.holds[side]
            width = 2 * (L.F_HALF_W if part == "female" else L.M_HALF_W) + 0.002
            if grips[side][2] > width:
                del self.holds[side]
        # take hold
        for side, (tcp, tips, gap) in grips.items():
            if side in self.holds:
                continue
            for part in ("female", "male"):
                if any(p == part for p, _ in self.holds.values()):
                    continue
                if self._straddles(part, tips, gap):
                    o = self.f_o if part == "female" else self.m_o
                    self.holds[side] = (part, o - tcp)
                    break
        # tabs pressed by a hand on the female
        press = [0.0, 0.0]
        for side, (part, _) in self.holds.items():
            if part != "female":
                continue
            tcp, tips, gap = grips[side]
            ly = float(((tips - self.f_o) @ self.f_rot)[1])
            if self.mode != "apart" and abs((TAB_C - self.slide()) - ly) <= 0.010 + 0.004:
                d = min(L.TAB_PROUD, max(0.0, (2 * L.TAB_X1 - max(gap, 2 * L.F_HALF_W)) / 2))
                press = [d, d]
        self.prongs = [p / L.TAB_LEVER for p in press]
        clear = min(self.prongs) >= L.RELEASE_RAD
        targets = {part: grips[side][0] + off for side, (part, off) in self.holds.items()}
        holder = {part: side for side, (part, _) in self.holds.items()}
        if self.mode == "clicked" and clear:
            self.mode = "sliding"
        if self.mode == "sliding" and not clear and self.slide() <= L.PLAY + 0.0005:
            self.mode = "clicked"
        if self.mode == "clicked":
            if "female" in targets and "male" in targets:
                if np.linalg.norm((targets["male"] - targets["female"]) - (self.m_o - self.f_o)) > 0.002:
                    del self.holds[holder["male"]]
                    self.slips += 1
                    targets.pop("male")
            if "female" in targets:
                d = targets["female"] - self.f_o
            elif "male" in targets:
                d = targets["male"] - self.m_o
            else:
                d = np.zeros(3)
            self.f_o, self.m_o = self.f_o + d, self.m_o + d
        elif self.mode == "sliding":
            if "female" in targets:
                self.f_o = targets["female"]
                if "male" in targets:
                    s = -float(((targets["male"] - self.f_o) @ self.f_rot)[1])
                    s = max(0.0, s)
                    self.m_o = self.f_o - self.f_rot[:, 1] * s
                else:
                    self.m_o = self.f_o - self.f_rot[:, 1] * self.slide()
            elif "male" in targets:
                d = targets["male"] - self.m_o
                self.f_o, self.m_o = self.f_o + d, self.m_o + d
            if self.slide() >= L.OUT_FULL:
                self.mode = "apart"
        else:
            if "female" in targets:
                self.f_o = targets["female"]
            if "male" in targets:
                self.m_o = targets["male"]
        self._rope(set(targets))

    def _rope(self, held: set) -> None:
        free_f = self.mode == "apart" and "female" not in held
        free_m = self.mode == "apart" and "male" not in held
        # node list: female mouth, female back, 12 hinges, male bar, male tip
        pts = np.vstack([self._f_pt(F_MOUTH), self._f_pt(F_BACK), self.nodes, self._m_pt(M_BAR), self._m_pt(M_TIP)])
        n = len(pts)
        fixed = np.zeros(n, dtype=bool)
        if not free_f:
            fixed[0] = fixed[1] = True
        if not free_m:
            fixed[-1] = fixed[-2] = True
        radius = np.full(n, NODE_R_WEB)
        radius[:2] = NODE_R_F
        radius[-2:] = NODE_R_M
        seg = [(0, 1, L.F_LEN, True)] + [(i, i + 1, L.PITCH, False) for i in range(1, n - 2)] + [
            (n - 2, n - 1, L.M_LEN, True)]
        pts[~fixed, 2] -= GRAVITY_STEP
        for _ in range(PASSES):
            for a, b, length, rigid in seg:
                d = pts[b] - pts[a]
                dist = float(np.linalg.norm(d))
                if dist < 1e-9 or (not rigid and dist <= length):
                    continue
                corr = (dist - length) * d / dist
                if fixed[a] and fixed[b]:
                    continue
                if fixed[a]:
                    pts[b] -= corr
                elif fixed[b]:
                    pts[a] += corr
                else:
                    pts[a] += corr / 2
                    pts[b] -= corr / 2
            self._collide(pts, fixed, radius)
        excess = max(float(np.linalg.norm(pts[b] - pts[a])) - length for a, b, length, _ in seg)
        if excess > SLIP_EXCESS:
            for side, (part, _) in list(self.holds.items()):
                if part == "male" or self.mode != "apart":
                    del self.holds[side]
                    self.slips += 1
                    break
        self.nodes = pts[2:-2].copy()
        if free_f:
            u = pts[1] - pts[0]
            self.f_rot = rx(roll_of(u))
            self.f_o = pts[1] - self.f_rot @ F_BACK
        if free_m:
            u = pts[-1] - pts[-2]
            self.m_rot = rx(roll_of(u))
            self.m_o = pts[-2] - self.m_rot @ M_BAR

    @staticmethod
    def _collide(pts, fixed, radius) -> None:
        for k in range(len(pts)):
            if fixed[k]:
                continue
            p = pts[k]
            for x0, x1, cy, cz, r in CYLS:
                if x0 <= p[0] <= x1:
                    dy, dz = p[1] - cy, p[2] - (L.TOP + cz)
                    d = math.hypot(dy, dz)
                    need = r + radius[k]
                    if d < need:
                        if d < 1e-9:
                            dy, dz, d = 0.0, 1.0, 1.0
                        p[1] = cy + dy * need / d
                        p[2] = L.TOP + cz + dz * need / d
            if p[2] < L.TOP + radius[k]:
                p[2] = L.TOP + radius[k]

    # -- what the scorer reads --------------------------------------------------------------------------------------
    def link_poses(self) -> dict:
        pts = np.vstack([self._f_pt(F_BACK), self.nodes, self._m_pt(M_BAR)])
        out = {}
        floor = L.TOP + L.WEB_T / 2
        for i in range(1, L.N_LINKS + 1):
            p0, p1 = pts[i - 1], pts[i]
            d = p1 - p0
            a = roll_of(d)
            if p0[2] + L.LINK_LEN * math.sin(a) < floor:
                # a rigid link reaching from p0 toward p1 would pass under the table (the rope has bunched there):
                # it lies tilted only as far as the table lets it, the way the real piece folds over
                c = 1.0 if d[1] >= 0.0 else -1.0
                sn = max(-1.0, min(0.0, (floor - p0[2]) / L.LINK_LEN))
                a = math.atan2(sn, c * math.sqrt(1.0 - sn * sn))
            r = rx(a - self.rest_roll[i - 1])
            out[f"w{i}"] = (*p0, *mk.matrix_to_quat(r))
        return out

    def states(self, names) -> dict:
        collar, male, dog = names
        fq, mq = mk.matrix_to_quat(self.f_rot), mk.matrix_to_quat(self.m_rot)
        return {collar: mk.state_from_poses(self.collar_spec, (*self.f_o, *fq), link_poses=self.link_poses(),
                                            source="offline"),
                male: mk.state_from_poses(self.male_spec, (*self.m_o, *mq),
                                          joints={"prong_p": self.prongs[0], "prong_n": self.prongs[1]},
                                          source="offline"),
                dog: mk.state_from_poses(self.dog_spec, (*self.dog_spec.pos, 0.0, 0.0, 0.0, 1.0), source="offline")}
