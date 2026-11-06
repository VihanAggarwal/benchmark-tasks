"""Offline checker only: where a released ring comes to rest. The simulator never calls anything in this file.

Our offline checker (tools/offline/check_task.py, NumPy only) has no physics. It treats each ring as its bounding
box (mechkit's offline box: a solid 60 x 60 mm block), carries it rigidly when the pads close across it and, when they
open, drops that box onto the highest thing under its outline. Under a ring released over the former that is the
sprue's top (the solid box cannot pass round the wax), so the checker's own drop would leave the ring 26 mm up, where
a real tube, open at both ends, slides down round the wax onto the base. It hands task.py's hooks the ring as mechkit's
assembly state (``source`` "offline") and feeds rigstate the arms from its kinematics (TCP, finger-tip point and the
commanded opening). When task.py sees ``source == "offline"`` it asks this model, once per control step, for the
rings' states, and the SAME scorer runs on them.

The model, step by step, for each ring:

- a gripper GRIPS a ring when its commanded gap closes to the ring's 60 mm width plus 1 mm with its tip point over
  the ring's outline (within 12 mm of it) and its pads (the 38 mm above the tips) beside the ring's walls; while it
  grips, the ring is where the checker carries it (it cannot be pushed below what is under its walls), and the
  reported opening is held at 60 mm (the walls stop the fingers, as the simulator's joint readings would show);
- when that gripper opens past 61 mm (the checker's own release), it lets go: the ring falls freely (9.81 m/s2, one
  step at a time, with its real downward speed reported), straight down and level, onto the highest thing under its
  four WALLS: the table, the former's 88 mm base, the step's 36 mm circle, the sprue, the crown, or another ring's
  walls. What lies inside its opening (the wax, the step if it is centred) is never under a wall;
- a ring nobody grips stays where it is (if the checker moved it sideways by a grasp this model did not see, its new
  place seen from above is taken over).

It cannot model a ring tipping onto the step's edge (it lands level, 16 mm up, which the scorer also counts as not
seated), lag, slip, wrist sag, contacts during the descent, or anything a policy other than the scripted demo does
with contacts: those are for the GPU runs.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import scoring as S

GRIP_GAP = L.RING_OUTER + 0.001  # 0.061: the checker carries a ring from here down, and lets go above it
TIP_SLACK = 0.012  # the tip point within this of the ring's outline seen from above
PAD_SPAN = 0.038  # the pads run this far up from the tips
MOVED_TOL = 0.001


def _origin(state, h: float) -> np.ndarray:
    """The ring's base-frame origin (the centre of its bottom face): mechkit's ``origin``, else from its centre."""
    o = getattr(state, "origin", None)
    if o is not None:
        return np.array(o, dtype=float)
    return np.asarray(state.pos, dtype=float) - np.array([0.0, 0.0, h / 2])


def _yaw(quat_xyzw) -> float:
    r = S.rotation(quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def _rect(cx: float, cy: float, hx: float, hy: float, yaw: float) -> np.ndarray:
    """Corners (4 x 2) of a rectangle of half sizes (hx, hy) centred on (cx, cy), turned yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    return pts @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def _rects_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() + 1e-12 or pb.max() <= pa.min() + 1e-12:
                return False
    return True


def _rect_circle(a: np.ndarray, cx: float, cy: float, r: float) -> bool:
    """Does the convex quadrilateral ``a`` overlap the circle?"""
    p = np.array([cx, cy])
    inside = True
    best = math.inf
    for i in range(4):
        q0, q1 = a[i], a[(i + 1) % 4]
        e = q1 - q0
        t = float(np.clip((p - q0) @ e / (e @ e), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(q0 + t * e - p)))
        if (e[0] * (p[1] - q0[1]) - e[1] * (p[0] - q0[0])) < 0:
            inside = False
    return inside or best < r


def wall_rects(x: float, y: float, yaw: float) -> list[np.ndarray]:
    """The four walls of an upright ring at (x, y) turned yaw, seen from above."""
    off = L.RING_OUTER_HALF - L.RING_WALL / 2
    hw = L.RING_WALL / 2
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for (dx, dy), (hx, hy) in (((off, 0.0), (hw, L.RING_OUTER_HALF)), ((-off, 0.0), (hw, L.RING_OUTER_HALF)),
                               ((0.0, off), (L.RING_INNER_HALF, hw)), ((0.0, -off), (L.RING_INNER_HALF, hw))):
        out.append(_rect(x + c * dx - s * dy, y + s * dx + c * dy, hx, hy, yaw))
    return out


class _Arm:
    """An arm as rigstate reports it, with the opening the ring's walls allow."""

    __slots__ = ("tcp", "tips", "opening")

    def __init__(self, tcp, tips, opening: float) -> None:
        self.tcp, self.tips, self.opening = tcp, tips, opening


class _State:
    """A ring's state as the scorer reads it (ObjectState's fields), upright and turned yaw."""

    def __init__(self, origin, yaw: float, vel, h: float) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.pos = self.origin + np.array([0.0, 0.0, h / 2])
        self.quat_xyzw = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        self.lin_vel = np.asarray(vel, dtype=float)
        self.ang_vel = np.zeros(3)
        ext = np.array([L.RING_OUTER_HALF * (abs(math.cos(yaw)) + abs(math.sin(yaw)))] * 2 + [h / 2])
        self.low, self.high = self.pos - ext, self.pos + ext
        self.source = "offline_model"


class RingModel:
    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.dt = 1.0 / control_hz
        self.reset()

    def reset(self) -> None:
        self.rings: dict[str, dict] = {}  # name -> {"origin", "yaw", "vz", "holder"}
        self.key = None  # the arms snapshot this step was computed for (success() then score() reuse it)
        self.out = None
        self.events: list[dict] = []
        self.steps = 0

    # -- geometry ------------------------------------------------------------------------------------------------
    def support(self, name: str, x: float, y: float, yaw: float, bottom: float) -> tuple[float, str]:
        """Height (world z) and name of the highest thing under the ring's four walls that is not above its
        bottom (a ring falls onto it)."""
        cand = [(L.TOP, "table")]
        for w in wall_rects(x, y, yaw):
            if _rects_overlap(w, _rect(L.FORMER_X, L.FORMER_Y, L.BASE_HALF, L.BASE_HALF, 0.0)):
                cand.append((L.BASE_TOP, "former_base"))
            if _rect_circle(w, L.FORMER_X, L.FORMER_Y, L.STEP_RADIUS):
                cand.append((L.STEP_TOP, "former_step"))
            if _rect_circle(w, L.WAX_X, L.WAX_Y, L.SPRUE_RADIUS):
                cand.append((L.SPRUE_TOP, "sprue"))
            if _rects_overlap(w, _rect(L.WAX_X, L.WAX_Y, L.CROWN_SIZE[0] / 2, L.CROWN_SIZE[1] / 2, 0.0)):
                cand.append((L.CROWN_TOP, "wax_crown"))
            for other, st in self.rings.items():
                if other == name:
                    continue
                o = st["origin"]
                if any(_rects_overlap(w, ow) for ow in wall_rects(o[0], o[1], st["yaw"])):
                    cand.append((float(o[2]) + L.ring_height(other), other))
        ok = [c for c in cand if c[0] <= bottom + 0.002]
        return max(ok) if ok else (L.TOP, "table")

    def _grips(self, arm, st: dict, h: float) -> bool:
        if S.finger_gap(arm) > GRIP_GAP:
            return False
        tips = np.asarray(arm.tips, dtype=float)
        o, yaw = st["origin"], st["yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        d = tips[:2] - o[:2]
        lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        over = abs(lx) <= L.RING_OUTER_HALF + TIP_SLACK and abs(ly) <= L.RING_OUTER_HALF + TIP_SLACK
        beside = tips[2] < o[2] + h and tips[2] + PAD_SPAN > o[2]
        return bool(over and beside)

    # -- one control step ----------------------------------------------------------------------------------------
    def step(self, checker_rings: dict, arms):
        """The rings' states and the arms for this control step. ``checker_rings`` is what the offline checker made
        of each ring (mechkit's offline state); ``arms`` is rigstate.arms()."""
        if arms is not None and arms is self.key and self.out is not None:
            return self.out  # the same step again (score() after success())
        self.key = arms
        self.steps += 1
        for name in L.RING_NAMES:
            ck = checker_rings[name]
            h = L.ring_height(name)
            if name not in self.rings:
                self.rings[name] = {"origin": _origin(ck, h), "yaw": _yaw(ck.quat_xyzw), "vz": 0.0, "holder": None,
                                    "vel": np.zeros(3)}
        if not S.arms_valid(arms):
            states = {n: _State(st["origin"], st["yaw"], np.zeros(3), L.ring_height(n)) for n, st in self.rings.items()}
            self.out = (states, arms)
            return self.out
        seen = dict(arms)
        for name in L.RING_NAMES:
            st, h, ck = self.rings[name], L.ring_height(name), checker_rings[name]
            ck_origin = _origin(ck, h)
            prev = st["origin"].copy()
            if st["holder"] is not None and S.finger_gap(arms[st["holder"]]) > GRIP_GAP:
                self.events.append({"step": self.steps, "event": "release", "ring": name, "arm": st["holder"]})
                st["holder"] = None
            if st["holder"] is None:
                for a in S.ARMS:
                    if self._grips(arms[a], st, h) and not any(o["holder"] == a for o in self.rings.values()):
                        st["holder"], st["vz"] = a, 0.0
                        self.events.append({"step": self.steps, "event": "grip", "ring": name, "arm": a})
                        break
            if st["holder"] is not None:
                want, yaw = ck_origin.copy(), _yaw(ck.quat_xyzw)
                floor, _ = self.support(name, want[0], want[1], yaw, float(prev[2]))
                if want[2] < floor:  # pushed onto what is under its walls: it stops there
                    want[2] = floor
                st["origin"], st["yaw"] = want, yaw
                a = st["holder"]
                seen[a] = _Arm(arms[a].tcp, arms[a].tips, max(float(arms[a].opening), L.RING_OUTER / L.FULL_GAP))
            else:
                if float(np.hypot(*(ck_origin[:2] - st["origin"][:2]))) > MOVED_TOL:
                    st["origin"][:2] = ck_origin[:2]  # moved sideways by a grasp the model did not see
                    st["yaw"] = _yaw(ck.quat_xyzw)
                floor, onto = self.support(name, st["origin"][0], st["origin"][1], st["yaw"], float(st["origin"][2]))
                if st["origin"][2] > floor + 1e-9 or st["vz"] < 0.0:
                    st["vz"] -= L.GRAVITY * self.dt
                    st["origin"][2] += st["vz"] * self.dt
                    if st["origin"][2] <= floor:
                        st["origin"][2], st["vz"] = floor, 0.0
                        self.events.append({"step": self.steps, "event": "landed", "ring": name, "onto": onto,
                                            "z_mm": round((floor - L.TOP) * 1000, 1)})
            vel = (st["origin"] - prev) / self.dt
            if st["holder"] is None and st["vz"] < 0.0:
                vel[2] = st["vz"]
            st["vel"] = vel
        states = {n: _State(st["origin"], st["yaw"], st["vel"], L.ring_height(n)) for n, st in self.rings.items()}
        self.out = (states, seen)
        return self.out
