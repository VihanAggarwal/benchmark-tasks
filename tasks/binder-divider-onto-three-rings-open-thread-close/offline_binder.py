"""The offline stand-in for the physics (tools/offline/check_task.py only; never used in the simulator).

The checker's proxy knows no hinge, detent or hole. Without a simulator, task.py steps this model from the arms'
commanded TCPs, pad tips and openings, and scores what it gives (all poses are table-world, z includes the table):

1. The lever. A closed gripper (opening under 0.2) whose pad tip is over the lever bar (x, y within 5 mm of it) and
   lower than the bar's top at the pad's x presses the gate open: its angle is what puts the bar's top at the pad
   tip's height. The detent catches the gate at OPEN from 0.55 rad, and shuts it from under 0.25.
2. The push. A closed gripper beside a ring (within 14 mm in y), its pad (27 mm across x, 22 mm of it over the tip)
   on the arm side of the movable half's wire, pushes the gate shut as far as the wire's surface allows.
3. The divider. Leaning in the holder until a gripper, closed (opening under 0.1), has its pad tip on the divider's
   upper 35 mm (within 12 mm of its mid-plane) and holds it there: the divider turns with the hand (its plane holds
   the hand's approach axis, its holes along y). Held by two hands it follows both; held by ONE hand it swings about
   the pad axis until its centre of mass hangs under the pad (a pinch holds no torque about its closing axis), so the
   hole row turns away from the rings. Let go: with all three holes 8 mm or more along their wires it slides down the half
   rings and hangs there, leaning back 25 degrees with its far part on the sorter (and follows the gate when that
   moves); else it falls flat on the table.
4. The shut by the divider. Held by both hands with every hole round its ring's movable half, a divider moved so that
   a hole's edge reaches its wire (centre more than 2 mm off) drives the gate shut until the wires are back inside the
   holes' clearance (only shutting; at the stop the gate stays at 0 while the wires are still inside the holes).
No friction, no inertia, no contact forces, no collision of the divider with the wires. Every number is a design
value, not a measurement.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import mechkit as mk
import rigstate
import scoring as S

CLOSED_OPEN = 0.20  # a gripper counts as closed (pushing) below this opening
PINCH_OPEN = 0.10  # a gripper counts as pinching (closed on the sheet) below this opening
RELEASE_OPEN = 0.15
DRIVE_CLEAR = L.HOLE_SIZE / 2 - L.WIRE / 2  # a hole's edge reaches the wire once its centre is this far off
SNAP_OPEN = 0.55
SNAP_SHUT = 0.25
TOPV = np.array([0.0, 0.0, L.TOP])
FLAT_R = np.column_stack([L.U_VEC, np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]) @ L.SHEET_R0.T


def rot_axis(axis, ang) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def sheet_rotation(approach, u=L.U_VEC) -> np.ndarray:
    """Base-frame -> world rotation of a divider whose plane contains the hand's approach axis and ``u``."""
    d = -np.asarray(approach, float)
    d = d / np.linalg.norm(d)
    n = np.cross(u, d)
    return np.column_stack([u, d, n]) @ L.SHEET_R0.T


class BinderModel:
    def __init__(self, binder_spec, div_spec) -> None:
        self.binder_spec, self.div_spec = binder_spec, div_spec
        self.reset()

    def reset(self) -> None:
        self.q = 0.0
        self.o = np.array([L.SHEET_O[0], 0.0, L.TOP + L.SHEET_O[2]])
        self.r = np.eye(3)
        self.mode = "rest"  # rest | held | threaded | dropped
        self.holds: dict = {}  # arm -> the pad tip's sheet coordinates (u, v, w)
        self.swing = 0.0
        self.nominal = (self.o.copy(), self.r.copy())  # the divider's pose before any one-hand swing

    # -- the divider's frame ----------------------------------------------------------------------------------------
    def sheet_coords(self, p, pose=None) -> np.ndarray:
        o, r = (self.o, self.r) if pose is None else pose
        return ((np.asarray(p, float) - o) @ r) @ L.SHEET_R0

    def world_of(self, loc) -> np.ndarray:
        return self.o + self.r @ (L.SHEET_R0 @ np.asarray(loc, float))

    def states(self, names) -> dict:
        binder, div = names
        bq = mk.matrix_to_quat(np.eye(3))
        dq = mk.matrix_to_quat(self.r)
        return {binder: mk.state_from_poses(self.binder_spec, (L.XS, 0.0, L.TOP, *bq), joints={"gate": self.q},
                                            source="offline"),
                div: mk.state_from_poses(self.div_spec, (*self.o, *dq), source="offline")}

    def _threaded(self) -> bool:
        st = self.states((S.BINDER, S.DIVIDER))
        r = S.read(st)
        return r is not None and S.threaded(r)

    # -- one control step -------------------------------------------------------------------------------------------
    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        g = {side: (np.asarray(t, float), np.asarray(p, float), float(o)) for side, (t, p, o) in zip(rigstate.ARMS, read)}
        pressing = self._lever(g)
        self._push(g)
        self._detent(pressing)
        self._sheet(g)

    def _lever(self, g) -> bool:
        pressing = False
        for tcp, tip, opening in g.values():
            if opening >= CLOSED_OPEN:
                continue
            t = tip - TOPV
            if (L.LEVER_Y[0] - 0.005 <= t[1] <= L.LEVER_Y[1] + 0.005
                    and L.LEVER_X[0] - 0.005 <= t[0] <= L.LEVER_X[1] + 0.005):
                qp = L.gate_for_press(float(t[2]))
                if qp > self.q:
                    self.q = qp
                if qp > 0:
                    pressing = True
        return pressing

    def _push(self, g) -> None:
        for tcp, tip, opening in g.values():
            if opening >= CLOSED_OPEN or self.q <= 0.0:
                continue
            t = tip - TOPV
            if min(abs(t[1] - y) for y in L.RING_Y) > 0.014:
                continue
            face = t[0] + 0.0135
            z0, z1 = float(t[2]), float(t[2]) + 0.022
            pts = L.half_points("movable", 60)

            def ok(q):
                p = L.rotate_gate(pts, q)
                sel = (p[:, 1] >= z0) & (p[:, 1] <= z1)
                return not np.any(sel) or bool(np.all(p[sel, 0] - 0.002 >= face))
            if ok(self.q):
                continue
            q = self.q
            while q > 0.0 and not ok(q):
                q = max(0.0, q - 0.01)
            self.q = q

    def _detent(self, pressing: bool) -> None:
        if self.q >= SNAP_OPEN:
            self.q = L.OPEN
        elif self.q < SNAP_SHUT and not pressing:
            self.q = 0.0

    # -- the divider --------------------------------------------------------------------------------------------------
    def _sheet(self, g) -> None:
        if self.mode == "threaded":
            self._rest_pose()
            return
        for side in list(self.holds):
            if g[side][2] > RELEASE_OPEN:
                del self.holds[side]
        for side, (tcp, tip, opening) in g.items():
            if side in self.holds or opening >= PINCH_OPEN or self.mode == "dropped":
                continue
            u, v, w = self.sheet_coords(tip, self.nominal)
            if abs(w) <= 0.012 and L.SHEET_V - 0.035 <= v <= L.SHEET_V + 0.004 and abs(u) <= L.SHEET_U / 2:
                self.holds[side] = np.array([u, v, w])
        if self.holds:
            self.mode = "held"
            self._follow(g)
            if len(self.holds) == 2:
                self._drag()
        elif self.mode == "held":
            self._let_go()

    def _follow(self, g) -> None:
        sides = list(self.holds)
        if len(sides) == 2:
            a = np.mean([g[s][1] - g[s][0] for s in sides], axis=0)
            u = g[sides[0]][1] - g[sides[1]][1]
            u = u / (np.linalg.norm(u) or 1.0)
            if u[1] < 0:
                u = -u
            self.r = sheet_rotation(a, u)
            self.swing = 0.0
        else:
            s = sides[0]
            a = g[s][1] - g[s][0]
            r2 = sheet_rotation(a)
            p = g[s][1]
            loc = self.holds[s]
            o2 = p - r2 @ (L.SHEET_R0 @ loc)
            com = o2 + r2 @ (L.SHEET_R0 @ np.array([0.0, L.SHEET_V / 2, 0.0]))
            n = r2 @ L.N_VEC
            c = com - p
            gz = np.array([0.0, 0.0, -1.0])
            cp, gp = c - n * (c @ n), gz - n * (gz @ n)
            ang = math.atan2(float(n @ np.cross(cp, gp)), float(cp @ gp))
            self.swing = ang
            self.r = rot_axis(n, ang) @ r2
        pts = [g[s][1] - self.r @ (L.SHEET_R0 @ self.holds[s]) for s in sides]
        self.o = np.mean(pts, axis=0)
        if len(sides) == 2:  # the carry sag measured on the GPU (layout.CARRY_SAG_N, CARRY_SAG_Z)
            self.o = self.o - L.CARRY_SAG_N * (self.r @ L.N_VEC) - np.array([0.0, 0.0, L.CARRY_SAG_Z])
        if len(sides) == 2:
            self.nominal = (self.o.copy(), self.r.copy())
        else:  # the pose the divider would have with no swing, for a second hand arriving
            r2 = sheet_rotation(g[sides[0]][1] - g[sides[0]][0])
            self.nominal = (g[sides[0]][1] - r2 @ (L.SHEET_R0 @ self.holds[sides[0]]), r2)

    def _drag(self) -> None:
        """Held by both hands with its holes round the movable half rings: a hole's edge that reaches its wire pushes
        the gate shut (the first angle, going shut, at which every wire is back inside its hole's clearance)."""
        if self.q <= 0.0:
            return
        off = self._off_movable(self.q)
        if off is not None and off <= DRIVE_CLEAR:
            return
        q = self.q
        while q > 0.0:
            q = max(0.0, q - 0.005)
            off = self._off_movable(q)
            if off is not None and off <= DRIVE_CLEAR:
                self.q = q
                return
        off = self._off_movable(0.0)
        if off is not None and off <= L.HOLE_SIZE / 2:
            self.q = 0.0

    def _off_movable(self, q):
        """The largest distance (m, in the sheet's plane) of a movable wire's crossing from its hole's centre with the
        gate at ``q``, or None if a movable half does not cross the sheet near its hole."""
        q_keep = self.q
        self.q = q
        st = self.states((S.BINDER, S.DIVIDER))
        self.q = q_keep
        b, dv = st[S.BINDER], st[S.DIVIDER]
        sheet = S.to_sheet(dv)
        worst = 0.0
        for i in range(3):
            offs = [max(abs(u - L.HOLE_U[i]), abs(v - L.HOLE_V))
                    for u, v, _ in S.crossings(S.wire_points(b, i, "movable"), sheet)]
            offs = [o for o in offs if o <= 0.010]
            if not offs:
                return None
            worst = max(worst, min(offs))
        return worst

    def _rest_pose(self) -> None:
        """Threaded and let go: the holes hang REST_DEPTH along the wire from its tip, the divider leaning back
        REST_TILT from horizontal (its far part on the sorter); it follows the gate when that moves."""
        d = np.array([-math.cos(L.REST_TILT), 0.0, math.sin(L.REST_TILT)])
        n = np.cross(L.U_VEC, d)
        self.r = np.column_stack([L.U_VEC, d, n]) @ L.SHEET_R0.T
        hole = L.wire_at(self.q, L.REST_DEPTH) + np.array([0.0, 0.0, L.TOP])
        self.o = hole - L.HOLE_V * d

    def _let_go(self) -> None:
        if self._threaded():
            self.mode = "threaded"
            self._rest_pose()
        else:
            self.mode = "dropped"
            self.r = FLAT_R.copy()
            self.o = np.array([self.o[0], self.o[1], L.TOP + L.SHEET_T / 2 + 0.0005])
