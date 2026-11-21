"""On the station table, real size: a steel fixture plate (160 x 320 x 12 mm) with two datum stops (S1 across its far
side, S2 along its right side) and two hold-down toggle clamps (C1 on the near side, C2 on the left side), both open
and resting on their stops; a splash guard behind the plate; and, to the left of the plate, a staging block with a
aluminium housing (the casting, 60 x 100 x 30 mm, 0.18 kg) lying on it.

Goal: the casting carried into the fixture and pushed hard into its corner (far face on S1, right face on S2, flat on
the plate), then both clamps closed over it (the pad on its top face, the lever flat and past over-centre), hands off.

The parts are mechkit compounds (mechkit.py, next to this file). Each clamp is articulated: a handle and a clamp arm
on their own hinges on the base and a connecting link on the handle (a four-bar toggle linkage, layout.Clamp). One
declared software rule (``_linkage_rule``, a mechkit on_step callback, ``linkage_targets``) closes the linkage's loop,
which a tree of joints cannot: every control step it sets the arm's joint target to the angle the link gives for the
handle's angle, the link's to the angle that keeps it on the arm, and passes the arm's load (the pad's force on the
casting) back to the handle through the linkage's velocity ratio. Nothing drives the clamp: the robot closes it by
moving the handle, and it locks because past dead centre the pad's force pushes the handle onto its stop. The arms are
read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Hold the casting seated (never the submitted
demo, a check that a held casting scores 0): DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON
line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import CASTING, CLAMP1, CLAMP2  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Set the casting hard into the fixture's corner stops, then close both hold-down clamps over it."


# -- the parts --------------------------------------------------------------------------------------------------------
def fixture_shapes():
    """Plate, S1, S2 in the plate's frame (origin at its centre on the table)."""
    def at(x0, x1, y0, y1, z0, z1):
        return mk.box((((x0 + x1) / 2) - L.PLATE_X, ((y0 + y1) / 2) - L.PLATE_Y, (z0 + z1) / 2),
                      (x1 - x0, y1 - y0, z1 - z0), L.STEEL)
    return [at(L.PLATE_X0, L.PLATE_X1, L.PLATE_Y0, L.PLATE_Y1, 0.0, L.PLATE_H),
            at(L.S1_X0, L.S1_X1, L.S1_Y0, L.S1_Y1, L.PLATE_H, L.PLATE_H + L.STOP_H),
            at(L.S2_X0, L.S2_X1, L.S2_Y0, L.S2_Y1, L.PLATE_H, L.PLATE_H + L.STOP_H)]


def _xy(c, along, across):
    return (along, across) if abs(c.dir[0]) > 0.5 else (across, along)


def _piece(c, p0, p1, thick, width, color, w=0.0):
    """A box in clamp ``c``'s plane from plane point p0 to p1 (its length), ``thick`` in the plane, ``width`` across,
    centred ``w`` across; in the clamp's base frame (origin on the table under the arm pivot)."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    length = float(np.hypot(d[0], d[1]))
    ang = math.atan2(d[1], d[0])
    mid = (p0 + p1) / 2
    centre = c.world(mid[0], mid[1], w) - np.array([c.px, c.py, 0.0])
    quat = tuple(float(v) for v in mk.matrix_to_quat(c.plane_frame(ang)))
    return mk.box(tuple(float(v) for v in centre), (length, width, thick), color, quat=quat)


def _square(c, p, size, width, color, w=0.0):
    p = np.asarray(p, dtype=float)
    return _piece(c, p - np.array([size / 2, 0.0]), p + np.array([size / 2, 0.0]), size, width, color, w)


def base_shapes(c):
    """A foot plate on the fixture plate under both pivots, a post under the arm pivot, two side cheeks holding the
    handle pivot, and the two pivot pins (frame: origin on the table under the arm pivot)."""
    top_post = L.Z_PIVOT - 0.010
    foot0, foot1 = L.PLATE_H, L.PLATE_H + L.FOOT_H
    w = 2 * L.BASE_HALF_ACROSS
    cheek_top = c.A[1] + 0.008
    shapes = [_piece(c, (L.BASE_REAR, (foot0 + foot1) / 2), (L.BASE_HALF_ALONG, (foot0 + foot1) / 2), L.FOOT_H, w,
                     L.BASE_COLOR),
              _piece(c, (-L.BASE_HALF_ALONG, (foot1 + top_post) / 2), (L.BASE_HALF_ALONG, (foot1 + top_post) / 2),
                     top_post - foot1, w, L.BASE_COLOR)]
    for side in (-1.0, 1.0):
        shapes.append(_piece(c, (L.BASE_REAR, (foot1 + cheek_top) / 2), (-L.BASE_HALF_ALONG, (foot1 + cheek_top) / 2),
                             cheek_top - foot1, L.CHEEK_T, L.BASE_COLOR, w=side * (L.BASE_HALF_ACROSS - L.CHEEK_T / 2)))
    shapes += [_square(c, c.B, 0.006, w + 0.004, L.STEEL), _square(c, c.A, 0.006, w + 0.004, L.STEEL)]
    return shapes


def handle_shapes(c):
    """The handle drawn open (qh = 0): hub on the pivot A, the bar, the black grip on its last 45 mm, and the crank lug
    out to the link's pin C."""
    end = c.handle_point(c.handle, 0.0, 0.0)
    grip0 = c.handle_point(c.handle - L.GRIP_L, 0.0, 0.0)
    return [_square(c, c.A, 0.016, 0.014, L.LEVER_COLOR),
            _piece(c, c.A, end, L.HANDLE_T, L.HANDLE_WIDTH, L.LEVER_COLOR),
            _piece(c, grip0, end, L.GRIP_T, L.GRIP_WIDTH, L.GRIP_COLOR),
            _piece(c, c.A, c.crank_pin(0.0), 0.008, 0.014, L.LEVER_COLOR)]


def arm_shapes(c):
    """The clamp arm drawn open (qa = 0, standing 99 degrees up): hub on the pivot B, the bar, the red pad under its
    end, and the tail lug out to the link's pin D."""
    return [_square(c, c.B, 0.014, L.ARM_WIDTH, L.LEVER_COLOR),
            _piece(c, c.B, c.arm_point(c.arm, 0.0, 0.0), L.ARM_T, L.ARM_WIDTH, L.LEVER_COLOR),
            _piece(c, c.arm_point(c.pad_s - L.PAD_SIZE / 2, -(L.ARM_T / 2 + L.PAD_H / 2), 0.0),
                   c.arm_point(c.pad_s + L.PAD_SIZE / 2, -(L.ARM_T / 2 + L.PAD_H / 2), 0.0), L.PAD_H, L.PAD_SIZE,
                   L.PAD_RED),
            _piece(c, c.B, c.tail_pin(0.0), 0.008, L.ARM_WIDTH, L.LEVER_COLOR)]


def link_shapes(c):
    """The connecting link drawn open, from the crank pin C to the arm's tail pin D, beside the arm and the handle."""
    return [_piece(c, c.crank_pin(0.0), c.tail_pin(0.0), L.LINK_T, L.LINK_W, L.LINK_COLOR, w=L.LINK_OFFSET)]


def link_limits(c):
    vals = [c.link_joint(L.QH_UPPER * k / 200) for k in range(201)]
    return min(0.0, min(vals)) - 0.05, max(0.0, max(vals)) + 0.05


def make_clamp(c):
    axis = "Y" if c is L.C1 else "X"
    org = np.array([c.px, c.py, 0.0])
    a_anchor = tuple(float(v) for v in c.world(*c.A) - org)
    c_anchor = tuple(float(v) for v in c.world(*c.crank_pin(0.0)) - org)
    lo, hi = link_limits(c)
    handle = mk.hinge(S.HANDLE, axis, anchor=a_anchor, shapes=handle_shapes(c), lower=0.0, upper=L.QH_UPPER,
                      mass=L.HANDLE_MASS, spring=c.handle_k, damping=L.HANDLE_DAMPING)
    arm = mk.hinge(S.ARM, axis, anchor=(0.0, 0.0, L.Z_PIVOT), shapes=arm_shapes(c), lower=0.0, upper=c.arm_upper,
                   mass=L.ARM_MASS, spring=c.k_arm, damping=c.arm_damping)
    link = mk.hinge(S.LINK, axis, anchor=c_anchor, shapes=link_shapes(c), lower=lo, upper=hi, mass=L.LINK_MASS,
                    spring=L.LINK_K, damping=L.LINK_DAMPING, parent=S.HANDLE)
    return mk.assembly(c.name, origin=(c.px, c.py, L.TOP), shapes=base_shapes(c), joints=[handle, arm, link],
                       color=L.BASE_COLOR, friction=L.LEVER_FRICTION, tight_fit=True)


FIXTURE = mk.assembly("fixture", origin=(L.PLATE_X, L.PLATE_Y, L.TOP), shapes=fixture_shapes(), color=L.STEEL,
                      friction=L.FIXTURE_FRICTION, tight_fit=True)
GUARD = mk.assembly("guard", origin=((L.GUARD_X0 + L.GUARD_X1) / 2, 0.0, L.TOP), color=L.STEEL,
                    shapes=[mk.box((0.0, 0.0, L.GUARD_H / 2), (L.GUARD_X1 - L.GUARD_X0, L.GUARD_Y1 - L.GUARD_Y0,
                                   L.GUARD_H), L.STEEL)])
STAGING = mk.assembly("staging", origin=(L.ST_X, L.ST_Y, L.TOP), color=L.ST_COLOR, friction=L.CS_FRICTION,
                      shapes=[mk.box((0.0, 0.0, L.ST_H / 2), (L.ST_L, L.ST_W, L.ST_H), L.ST_COLOR)])
CLAMP1_PART = make_clamp(L.C1)
CLAMP2_PART = make_clamp(L.C2)
CASTING_PART = mk.assembly(CASTING, origin=(L.START_X, L.START_Y, L.TOP + L.START_Z), mass=L.CS_MASS,
                           shapes=[mk.box((0.0, 0.0, 0.0), (L.CS_L, L.CS_W, L.CS_H), L.CS_COLOR)],
                           color=L.CS_COLOR, friction=L.CS_FRICTION, tight_fit=True)
OBJECTS = [FIXTURE, GUARD, STAGING, CLAMP1_PART, CLAMP2_PART, CASTING_PART]
SPECS = {CASTING: CASTING_PART, CLAMP1: CLAMP1_PART, CLAMP2: CLAMP2_PART}


# -- the one declared rule: the linkage's loop, closed every control step ------------------------------------------
def linkage_targets(c, qh, qa, vh, prev):
    """One control step of the linkage rule for clamp ``c`` (layout.Clamp): handle at ``qh`` (rad) moving at ``vh``
    (rad/s), arm at ``qa``, ``prev`` the arm target set last step (None at first). Returns ({joint: target}, the arm
    target now).

    arm     the angle the link puts it at for this handle angle (layout.Clamp.arm_target); its joint spring is the
            spindle's stiffness, so an arm held up by the casting presses it with that spring's force
    link    the angle that keeps the link's far end on the arm's tail pin
    handle  the arm's spring load now, passed back through the linkage (virtual work: arm torque times d(arm)/d(handle))
            as a torque on the handle, applied through the handle's joint spring: target = qh + torque / k (+ a lead of
            half a step of its own speed, so a moving handle feels no drag from the 20 Hz update). Short of dead
            centre it pushes the handle open; past it, onto its over-centre stop. With no load it is zero: the handle
            then only has its weight and its damping.
    The handle's target stays within HANDLE_TARGET_MAX of the handle (a torque of 0.5 rad times its spring at most,
    beyond any the pad gives). A state that cannot be read sets nothing."""
    try:
        qh, qa = float(qh), float(qa)
        vh = float(vh) if vh is not None else 0.0
    except (TypeError, ValueError):
        return {}, prev
    if not (math.isfinite(qh) and math.isfinite(qa)):
        return {}, prev
    if not math.isfinite(vh):
        vh = 0.0
    qh_c = min(max(qh, 0.0), L.QH_UPPER)
    at = c.arm_target(qh_c)
    torque = 0.0 if prev is None else c.handle_torque(qh_c, qa, prev)
    lead = max(-L.HANDLE_LEAD_MAX, min(L.HANDLE_LEAD_MAX, L.HANDLE_LEAD * vh / L.CONTROL_HZ))
    push = max(-L.HANDLE_TARGET_MAX, min(L.HANDLE_TARGET_MAX, lead + torque / c.handle_k))
    return {S.HANDLE: qh + push, S.ARM: at, S.LINK: c.link_joint(qh_c)}, at


_PREV = {CLAMP1: None, CLAMP2: None}


@mk.on_step
def _linkage_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    for name, part, c in ((CLAMP1, CLAMP1_PART, L.C1), (CLAMP2, CLAMP2_PART, L.C2)):
        s = states.get(name)
        try:
            qh, qa = s.joints[S.HANDLE], s.joints[S.ARM]
            vh = (getattr(s, "joint_vel", None) or {}).get(S.HANDLE, 0.0)
        except (AttributeError, KeyError, TypeError):
            continue
        targets, _PREV[name] = linkage_targets(c, qh, qa, vh, _PREV.get(name))
        for joint, value in targets.items():
            mk.set_targets(part, joint, value)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it neither pushes the casting nor
# turns a handle. Offline only (never in the simulator), this model gives the scorer: (1) the casting carried by a
# gripper that holds it across its 60 mm (follows the TCP) and dropped onto the highest support under it when let go;
# (2) pushed by a closed gripper's block (its faces PAD_FACE ahead of the TCP along the closing direction and
# FIST_HALF across it) toward +x until S1 or toward -y until S2; (3) a handle held by a gripper closed on its grip at
# the grasp point follows the TCP round its pivot (layout.Clamp.grasp_point); let go, a handle past dead centre goes
# onto its over-centre stop (the pad's force through the linkage), one short of it but past Clamp.balance_qh settles
# with the pad just touching a casting under it (the arm's weight against the pad's force; with nothing under the
# pad it closes onto its stop, shut on nothing), and one nearer its open stop falls back open (the handle's weight);
# (4) a closed gripper's block behind a grip, pushing its back face forward, turns the handle with it (onto its stop
# at most); (5) the arm where the linkage puts it, or flat on the casting's top when the casting is seated under its
# pad.
class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.cx, self.cy, self.cz = L.START_X, L.START_Y, L.START_Z
        self.carry: dict = {}  # arm -> (dx, dy, dz) of the casting's centre from the TCP
        self.qh = {CLAMP1: 0.0, CLAMP2: 0.0}
        self.held: dict = {}  # arm -> clamp name whose handle it holds

    # support height under the casting's footprint at its centre (x, y)
    def support(self, x, y) -> float:
        best = 0.0
        if abs(x - L.ST_X) < (L.ST_L + L.CS_L) / 2 and abs(y - L.ST_Y) < (L.ST_W + L.CS_W) / 2:
            best = max(best, L.ST_H)
        if (L.PLATE_X0 - L.CS_L / 2 < x < L.PLATE_X1 + L.CS_L / 2
                and L.PLATE_Y0 - L.CS_W / 2 < y < L.PLATE_Y1 + L.CS_W / 2):
            best = max(best, L.PLATE_H)
        return best

    def _yaw(self, side: str) -> float:
        t = self.steps / L.CONTROL_HZ
        acc = 0.0
        yaw = None
        for w in PLAN_YAWS.get(side, ()):
            acc += w[0]
            yaw = w[1]
            if t <= acc + 1e-9:
                break
        return 0.0 if yaw is None else yaw

    @staticmethod
    def handle_from_tcp(c, tcp) -> float:
        """The handle angle that puts its grasp point under this TCP (projected into the clamp's plane)."""
        p = np.array([float(tcp[0]), float(tcp[1]), float(tcp[2]) - L.TOP - L.GRASP_TCP_UP])
        rel = p[:2] - np.array([c.px, c.py])
        a = float(rel @ c.dir)
        psi = math.atan2(a - c.A[0], p[2] - c.A[1])
        return min(max(psi + L.HANDLE_OPEN, 0.0), L.QH_UPPER)

    def _grasp(self, side, tcp, opening) -> None:
        lo, hi = L.GRASP_OPENING
        if not (lo < opening < hi):
            self.held.pop(side, None)
            return
        if side in self.held:
            return
        for name, c in ((CLAMP1, L.C1), (CLAMP2, L.C2)):
            g = c.grasp_point(c.handle - L.GRASP_BACK, self.qh[name])
            want = np.array([g[0], g[1], g[2] + L.GRASP_TCP_UP])
            here = np.array([float(tcp[0]), float(tcp[1]), float(tcp[2]) - L.TOP])
            if float(np.linalg.norm(here - want)) <= L.GRASP_TOL:
                self.held[side] = name
                return

    def step(self, arms) -> None:
        self.steps += 1
        read = rigstate.read_arms(arms)
        if read is None:
            return
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            x, y, z = float(tcp[0]), float(tcp[1]), float(tcp[2]) - L.TOP
            tip_z = float(tips[2]) - L.TOP
            holding = (0.30 < opening < 0.75 and abs(x - self.cx) < 0.012 and abs(y - self.cy) < 0.020
                       and self.cz - L.CS_H / 2 + 0.003 < tip_z < self.cz + L.CS_H / 2)
            if holding and side not in self.carry:
                self.carry[side] = (self.cx - x, self.cy - y, self.cz - z)
            if not holding and side in self.carry:
                self.carry.pop(side)
                self.cz = self.support(self.cx, self.cy) + L.CS_H / 2
        if self.carry:
            side = next(iter(self.carry))
            tcp = read[rigstate.ARMS.index(side)][0]
            dx, dy, dz = self.carry[side]
            self.cx, self.cy, self.cz = float(tcp[0]) + dx, float(tcp[1]) + dy, float(tcp[2]) - L.TOP + dz
            if self.cz - L.CS_H / 2 < L.PLATE_H + L.STOP_H and abs(self.cy) < L.PLATE_Y1:
                # a held casting low over the plate meets the stops' faces; the wrist gives, the grip does not slip
                if L.S2_X0 - L.CS_L / 2 < self.cx < L.S2_X1 + L.CS_L / 2:
                    self.cy = max(self.cy, L.SEAT_Y)
                if L.S1_Y0 - L.CS_W / 2 < self.cy < L.S1_Y1 + L.CS_W / 2:
                    self.cx = min(self.cx, L.SEAT_X)
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            if side in self.carry:
                continue
            self._grasp(side, tcp, opening)
            if opening > 0.2 or side in self.held:
                continue
            x, y = float(tcp[0]), float(tcp[1])
            tip_z = float(tips[2]) - L.TOP
            x_closing = abs(math.sin(self._yaw(side))) > 0.5
            face_x = L.PAD_FACE if x_closing else L.FIST_HALF  # the block's face ahead of the TCP along x
            face_y = L.FIST_HALF if x_closing else L.PAD_FACE  # and along y
            if not (abs(tip_z + L.FIST_H / 2 - self.cz) < L.CS_H / 2 + L.FIST_H / 2 - 0.002
                    and tip_z < self.cz + L.CS_H / 2):
                continue
            if abs(y - self.cy) < L.CS_W / 2 + face_y - 0.004 and x < self.cx - L.CS_L / 2:
                behind = x - face_x <= self.cx - L.CS_L / 2 + 0.0005
                face = x + face_x
                if behind and face > self.cx - L.CS_L / 2:
                    self.cx = min(face + L.CS_L / 2, L.SEAT_X)
            elif abs(x - self.cx) < L.CS_L / 2 + face_x - 0.004 and y > self.cy + L.CS_W / 2:
                behind = y + face_y >= self.cy + L.CS_W / 2 - 0.0005
                face = y - face_y
                if behind and face < self.cy + L.CS_W / 2:
                    self.cy = max(face - L.CS_W / 2, L.SEAT_Y)
        for name, c in ((CLAMP1, L.C1), (CLAMP2, L.C2)):
            holders = [s for s, n in self.held.items() if n == name]
            pushed = [q for q in (self.block_push(c, name, side, tcp, tips, opening)
                                  for side, (tcp, tips, opening) in zip(rigstate.ARMS, read)
                                  if side not in self.carry and side not in self.held) if q is not None]
            if holders:
                tcp = read[rigstate.ARMS.index(holders[0])][0]
                self.qh[name] = self.handle_from_tcp(c, tcp)
            elif pushed:
                self.qh[name] = max(pushed)  # a block pushing the grip's back face carries the handle forward
            elif self.qh[name] >= L.QH_DC:
                self.qh[name] = L.QH_UPPER  # past dead centre: onto the over-centre stop
            elif self.qh[name] > c.balance_qh():  # the arm's weight closes it: onto the casting, short of vertical
                self.qh[name] = c.touch_qh() if self.under_pad(c) else L.QH_UPPER
            else:
                self.qh[name] = 0.0  # near its open stop: the handle's weight takes it back open

    def block_push(self, c, name, side, tcp, tips, opening):
        """The handle angle a closed gripper's block puts clamp ``c``'s handle at by pushing its grip's back face
        forward (toward the casting), or None if it does not touch it from behind. The block: its flat faces PAD_FACE
        ahead of the TCP along the fingers' closing direction and FIST_HALF across it, from the pad tips up FIST_H; it
        meets the grip (the handle's last GRIP_L) where they overlap in height and across the clamp's plane."""
        if opening > 0.2:
            return None
        x_closing = abs(math.sin(self._yaw(side))) > 0.5
        along_x = abs(c.dir[0]) > 0.5
        face = L.PAD_FACE if x_closing == along_x else L.FIST_HALF  # the block's face ahead of the TCP along c.dir
        half_across = L.FIST_HALF if x_closing == along_x else L.PAD_FACE
        rel = np.array([float(tcp[0]) - c.px, float(tcp[1]) - c.py])
        a = float(rel @ c.dir)
        w = float(rel @ c.across()[:2])
        if abs(w) > half_across + L.GRIP_WIDTH / 2 - 0.002:
            return None
        z0 = float(tips[2]) - L.TOP
        z1 = z0 + L.FIST_H
        qh = self.qh[name]
        psi = qh - L.HANDLE_OPEN
        g0 = c.A[1] + (c.handle - L.GRIP_L) * math.cos(psi)  # the grip's height span now
        g1 = c.A[1] + c.handle * math.cos(psi)
        lo, hi = max(z0, g0), min(z1, g1)
        if hi <= lo:
            return None

        def back(p):  # the grip's back face, its rearmost point over the overlap, along c.dir
            return min(c.A[0] + (z - c.A[1]) * math.tan(p) - (L.GRIP_T / 2) / math.cos(p) for z in (lo, hi))

        if a > back(psi):  # the block's centre is not behind the grip: it is not pushing it forward
            return None
        if a + face <= back(psi):
            return None
        p_lo, p_hi = psi, L.QH_UPPER - L.HANDLE_OPEN
        if back(p_hi) <= a + face:
            return L.QH_UPPER  # onto its over-centre stop
        for _ in range(50):
            mid = (p_lo + p_hi) / 2
            if back(mid) < a + face:
                p_lo = mid
            else:
                p_hi = mid
        return min(max(p_lo + L.HANDLE_OPEN, qh), L.QH_UPPER)

    def under_pad(self, c) -> bool:
        p = c.pad_bottom(L.QA_FLAT)
        return bool(abs(p[0] - self.cx) <= L.CS_L / 2 - L.PAD_INSET and abs(p[1] - self.cy) <= L.CS_W / 2 - L.PAD_INSET
                    and abs(self.cz - L.CS_H / 2 - L.PLATE_H) < 0.002)

    def qa(self, name: str) -> float:
        c = L.C1 if name == CLAMP1 else L.C2
        at = c.arm_target(self.qh[name])
        return min(at, L.QA_FLAT) if self.under_pad(c) else at

    def states(self) -> dict:
        cs = mk.state_from_poses(CASTING_PART, (self.cx, self.cy, L.TOP + self.cz, 0.0, 0.0, 0.0, 1.0),
                                 source="offline")
        out = {CASTING: cs}
        for name, part, c in ((CLAMP1, CLAMP1_PART, L.C1), (CLAMP2, CLAMP2_PART, L.C2)):
            qh = self.qh[name]
            out[name] = mk.state_from_poses(part, (c.px, c.py, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                            joints={S.HANDLE: qh, S.ARM: self.qa(name), S.LINK: c.link_joint(qh)},
                                            source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(CASTING), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, gap S1 mm, gap S2 mm, q1 deg, q2 deg]


@rigstate.on_reset
def _new_episode() -> None:
    _PREV.update({CLAMP1: None, CLAMP2: None})
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@mk.on_reset
def _kit_reset() -> None:
    _PREV.update({CLAMP1: None, CLAMP2: None})
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the casting's pose, both levers, what the scorer
    saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """Seated, both clamps locked on the casting, both grippers 30 mm clear and every part still for 1 s; held 1 s.
    This runs first every control step, so it also feeds the tracker (and, without a simulator, the offline
    stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    offline = no_simulator(parts)
    if offline:
        _OFFLINE.step(arms)
        parts = _OFFLINE.states()
    result = _TRACKER.update(parts, arms)
    row = {"episode": mk.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(CASTING), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["gap_s1_mm"], d["gap_s2_mm"], d["q1_deg"], d["q2_deg"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless every part is let go and still and the casting is seated; else 1.0 both clamps locked, 0.5 one, 0.25
    none. The score of the latest scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap.
PAD_TIP = L.TIPS_BELOW
OPEN = -0.04695  # 94 mm
OPEN_L = -(L.CS_L + 0.024) / 2  # 84 mm round the casting's 60 mm: 12 mm each side
GRIP = -(L.CS_L - 0.008) / 2  # 52 mm commanded across it: 8 mm under its width
CLOSED = 0.0
YAW_XL, YAW_XR, YAW_Y = -math.pi / 2, math.pi / 2, 0.0  # closing along x (left, right arm) and along y

# the left arm: pick, carry, place, push into the corner
GRIP_DY = 0.010  # the pads take the casting 10 mm left (+y) of its centre: clear of C1's base when they open
GRIP_DOWN = 0.018  # pad tips 18 mm under the casting's top
GRIP_Z = L.ST_H + L.CS_H - GRIP_DOWN + PAD_TIP  # 0.045: TCP height at the pick
CARRY_Z = 0.095  # casting's underside 70 mm up: clear of the clamps' bases (40 mm)
GRIP_TO_BOTTOM = GRIP_Z - L.ST_H  # TCP height above the casting's underside while held (25 mm)
LANE_X = 0.225
SHIFT_Y = 0.045  # the casting's centre while it crosses toward the plate: clear of C1's lever (y -37 to -13 mm)
SHIFT_Z = 0.040 + GRIP_TO_BOTTOM  # its underside 40 mm up: over the stops (24 mm)
PLACE_Z = L.PLATE_H + 0.003 + GRIP_TO_BOTTOM  # set down 3 mm over the plate
PUSH_TIPS = 0.026  # pad tips 26 mm up: on the casting's side face from 26 to 42 mm, 14 mm over the plate
PUSH_Z = PUSH_TIPS + PAD_TIP
PUSH1_Y = 0.030  # inside S1's 90 mm face, clear of C1's base (y -42 to -8 mm)
PUSH_GAP = 0.012
PUSH1_X0 = L.PLACE_NEAR_X - PUSH_GAP - L.PAD_FACE  # block face 12 mm short of the near face
OVERTRAVEL = 0.008
HELD_OVERTRAVEL = 0.004  # the held casting is carried 4 mm past S2's face: the wrist gives
PUSH_SPEED = 0.020  # m/s
HOVER_L = 0.12
PARK_L = (0.22, 0.30, 0.160)
PARK_NC = (0.14, 0.33, 0.200)  # no_clamps: aside, well clear of C2's open handle

# the clamps, in two parts. (1) The free stroke: a gripper closed on each handle's grip, 15 mm short of its end,
# carries it round its pivot until the pad comes down onto the casting. The fingers close across the handle, so the
# pinch drives it by friction only: enough for the free stroke (about 0.3 N), not for the press (GPU, 21 Nov: pinched
# through the press, both handles stalled 2 to 3 degrees short of vertical, the fingers slipping along the grip with
# the arm 8 to 10 mm behind its command). (2) The press: the gripper lets go, closes into a block behind the grip
# with its fingers along the stroke and pushes the grip's back face, square on, through the press and dead centre
# onto the over-centre stop, its face commanded well past where the stop puts the grip (the arm trails a pushed load).
PINCH = -0.006  # 12 mm commanded across the 24 mm grip: squeezed
ARC_FREE = (-57.0, -45.0, -33.0, -21.0, -11.0)  # handle angle from vertical (deg): the free part of the stroke
ARC_FREE_S = 1.0  # s per step of it
ARC_LAND = (-6.0, -2.0)  # slowly on till the pad lands (it touches 2.8 (C2) / 3.4 (C1) degrees short of vertical)
ARC_LAND_S = 0.8
ARC_YAW = {L.C1.name: 0.0, L.C2.name: -math.pi / 2}  # fingers closing across each handle (along its pivot axis)
PUSH_YAW = {L.C1.name: YAW_XR, L.C2.name: YAW_Y}  # fingers closing along each stroke: the block's face leads
PUSH_START_PSI = -9.0  # the block comes down 10 mm behind the grip as it stands at this angle (it rests at about -3)
PUSH_CLEAR = 0.010
PUSH_LOAD_PSI = -3.0  # the block's face up to the grip as it stands with the pad just on the casting
OVERSHOOT = 0.020  # the block's face commanded 20 mm past the grip's back face on the stop (+1.5 degrees)
PUSH_LOAD_S = 1.0
PRESS_S = 3.0  # slowly through the press, dead centre and onto the stop
FIST_Z_UP = 0.002  # the block's middle (TCP - 2 mm) at the grip's middle: 11.5 mm in from both its ends


def grasp_tcp(c, psi_deg):
    """TCP (x, y, z above the table) on the handle's grip with the handle ``psi_deg`` from vertical."""
    g = c.grasp_point(c.handle - L.GRASP_BACK, L.QH_DC + math.radians(psi_deg))
    return (float(g[0]), float(g[1]), float(g[2]) + L.GRASP_TCP_UP)


def back_face_a(c, psi_deg, z):
    """Where the grip's back face is, along the clamp's direction from its arm pivot (m), at height ``z`` above the
    table, with the handle ``psi_deg`` from vertical (+ toward the casting)."""
    p = math.radians(psi_deg)
    return float(c.A[0] + (z - c.A[1]) * math.tan(p) - (L.GRIP_T / 2) / math.cos(p))


def push_z(c):
    """TCP height for the push: the block (pad tips 13 mm under the TCP, flat 22 mm above them) on the grip's
    middle."""
    return float(c.A[1] + c.handle - L.GRIP_L / 2 + FIST_Z_UP)


def push_points(c):
    """The block's face positions (along the clamp's direction) for the push: start (10 mm behind the grip at
    PUSH_START_PSI, measured at the handle's end, which leans furthest back), up to the grip (PUSH_LOAD_PSI, at the
    block's top) and the end (OVERSHOOT past the grip's back face on the stop, at the block's bottom, which meets it
    first past vertical)."""
    z = push_z(c)
    tips, top = z - PAD_TIP, z - PAD_TIP + L.FIST_H
    end_z = c.A[1] + c.handle * math.cos(math.radians(PUSH_START_PSI))
    start = back_face_a(c, PUSH_START_PSI, end_z) - PUSH_CLEAR
    load = back_face_a(c, PUSH_LOAD_PSI, top)
    end = back_face_a(c, math.degrees(L.OVER_CENTRE), tips) + OVERSHOOT
    return start, load, end


def push_tcp(c, face, z):
    """TCP (x, y, z above the table) with the block's face at ``face`` along the clamp's direction."""
    p = c.world(face - L.PAD_FACE, z)
    return (float(p[0]), float(p[1]), float(p[2]))


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def left_seat_moves():
    """The left arm: pick the casting off the staging block, carry it over the clamps' bases to the plate, set it down
    6 mm short of S1 and 12 mm off S2, slide it (still held, 3 mm over the plate) -y onto S2, let go, close the fingers
    into a block and push it +x into S1 (it slides along S2). The pads hold it 10 mm left of its centre, clear of C1's
    base when they open. Ends 18 cm over the plate with the block closed. Seating
    against S2 while it is held keeps the gripper's housing out of the open C2 arm, which stands 99 degrees up just
    beyond the casting's left face."""
    x, y = L.START_X, L.START_Y + GRIP_DY
    shift_y, place_y = SHIFT_Y + GRIP_DY, L.PLACE_Y + GRIP_DY
    s2_y, seat_y = L.SEAT_Y - HELD_OVERTRAVEL + GRIP_DY, L.SEAT_Y + GRIP_DY
    return [
        _w(2.0, x, y, 0.16, OPEN_L, "over the casting on the staging block", YAW_XL),
        _w(1.0, x, y, GRIP_Z + 0.030, OPEN_L, "down to just above it", YAW_XL),
        _w(1.5, x, y, GRIP_Z, OPEN_L, "slowly down round it", YAW_XL),
        _w(0.5, x, y, GRIP_Z, OPEN_L, "settle", YAW_XL),
        _w(1.5, x, y, GRIP_Z, GRIP, "grip the casting across its 60 mm", YAW_XL),
        _w(2.0, x, y, CARRY_Z, GRIP, "lift it 70 mm clear", YAW_XL),
        _w(1.0, LANE_X, y, CARRY_Z, GRIP, "slide it to the lane clear of the left clamp", YAW_XL),
        _w(3.0, LANE_X, shift_y, CARRY_Z, GRIP, "carry it toward the plate", YAW_XL),
        _w(1.5, L.PLACE_X, shift_y, CARRY_Z, GRIP, "over the plate", YAW_XL),
        _w(1.0, L.PLACE_X, shift_y, SHIFT_Z, GRIP, "down over the stops", YAW_XL),
        _w(1.5, L.PLACE_X, place_y, SHIFT_Z, GRIP, "across to its place", YAW_XL),
        _w(1.5, L.PLACE_X, place_y, PLACE_Z, GRIP, "set it down", YAW_XL),
        _w(0.5, L.PLACE_X, place_y, PLACE_Z, GRIP, "settle", YAW_XL),
        _w((L.PLACE_GAP_Y + HELD_OVERTRAVEL) / PUSH_SPEED, L.PLACE_X, s2_y, PLACE_Z, GRIP,
           "slide it -y onto S2, still held", YAW_XL),
        _w(0.5, L.PLACE_X, s2_y, PLACE_Z, GRIP, "settle against S2", YAW_XL),
        _w(0.5, L.PLACE_X, seat_y - 0.001, PLACE_Z, GRIP, "ease off to a light press", YAW_XL),
        _w(1.5, L.PLACE_X, seat_y - 0.001, PLACE_Z, OPEN_L, "let go", YAW_XL),
        _w(1.0, L.PLACE_X, seat_y, HOVER_L, OPEN_L, "straight up", YAW_XL),
        _w(2.0, PUSH1_X0, PUSH1_Y, HOVER_L, CLOSED, "close the fingers into a block, over the casting's near face",
           YAW_XL),
        _w(2.0, PUSH1_X0, PUSH1_Y, PUSH_Z, CLOSED, "down beside its near face", YAW_XL),
        _w(abs(L.PLACE_GAP_X + PUSH_GAP + OVERTRAVEL) / PUSH_SPEED, PUSH1_X0 + PUSH_GAP + L.PLACE_GAP_X + OVERTRAVEL,
           PUSH1_Y, PUSH_Z, CLOSED, "push it +x into S1", YAW_XL),
        _w(0.5, PUSH1_X0 + PUSH_GAP + L.PLACE_GAP_X + OVERTRAVEL, PUSH1_Y, PUSH_Z, CLOSED, "settle against S1", YAW_XL),
        _w(1.0, PUSH1_X0, PUSH1_Y, PUSH_Z, CLOSED, "back off", YAW_XL),
        _w(1.0, PUSH1_X0, PUSH1_Y, HOVER_L + 0.06, CLOSED, "up", YAW_XL),
    ]


def clamp_moves(c, start_hover, park):
    """A gripper closing clamp ``c``: over the open handle's grip, down round it, pinch it, carry it round its pivot
    (the free stroke) until the pad comes down on the casting, let go and rise clear; then, fingers closed into a block
    along the stroke, down behind the grip and push its back face slowly through the press and dead centre onto the
    over-centre stop, hold, back off, up, aside."""
    yaw = ARC_YAW[c.name]
    g0 = grasp_tcp(c, ARC_FREE[0])
    plan = [
        _w(2.0, g0[0], g0[1], start_hover, OPEN, f"over {c.name}'s open handle", yaw),
        _w(1.2, g0[0], g0[1], g0[2] + 0.030, OPEN, "down to just above its grip", yaw),
        _w(1.0, g0[0], g0[1], g0[2], OPEN, "down round the grip", yaw),
        _w(1.5, g0[0], g0[1], g0[2], PINCH, f"pinch {c.name}'s grip", yaw),
    ]
    for deg in ARC_FREE[1:]:
        p = grasp_tcp(c, deg)
        plan.append(_w(ARC_FREE_S, *p, PINCH, f"swing {c.name}'s handle to {deg:.0f} deg", yaw))
    for deg in ARC_LAND:
        p = grasp_tcp(c, deg)
        plan.append(_w(ARC_LAND_S, *p, PINCH, f"lower {c.name}'s pad onto the casting: handle to {deg:.0f} deg", yaw))
    last = grasp_tcp(c, ARC_LAND[-1])
    clear_z = c.A[1] + c.handle + PAD_TIP + 0.015  # pad tips 15 mm over the handle's end
    plan.append(_w(0.5, *last, PINCH, "settle with the pad on the casting", yaw))
    plan.append(_w(1.0, *last, OPEN, f"let go of {c.name}'s grip", yaw))
    plan.append(_w(1.0, last[0], last[1], clear_z, OPEN, f"straight up off {c.name}'s grip", yaw))
    start, load, end = push_points(c)
    z, pyaw = push_z(c), PUSH_YAW[c.name]
    p0, p1, p2 = push_tcp(c, start, clear_z), push_tcp(c, start, z), push_tcp(c, load, z)
    p3, p4 = push_tcp(c, end, z), push_tcp(c, start, z)
    plan += [
        _w(1.5, *p0, CLOSED, f"behind {c.name}'s handle, the fingers closed into a block along the stroke", pyaw),
        _w(1.5, *p1, CLOSED, f"down behind {c.name}'s grip", pyaw),
        _w(PUSH_LOAD_S, *p2, CLOSED, f"up to {c.name}'s grip", pyaw),
        _w(PRESS_S, *p3, CLOSED, f"push {c.name}'s handle through dead centre onto its stop", pyaw),
        _w(0.5, *p3, CLOSED, "hold it on the over-centre stop", pyaw),
        _w(0.8, *p4, CLOSED, f"back off {c.name}'s handle", pyaw),
        _w(0.8, p4[0], p4[1], clear_z, CLOSED, f"straight up off {c.name}", pyaw),
        _w(1.5, park[0], park[1], park[2], OPEN, "aside", pyaw),
    ]
    return plan


PARK_R = (0.14, -0.20, 0.150)
WAIT_R = (0.18, -0.24, 0.150)


def plans(variant: str = "") -> dict:
    """Both arms' plans. Order of work: the left arm seats the casting (pick, carry, set down, slide it onto S2 while
    held, push +x into S1), then goes round to the left clamp and closes it; the right arm waits aside until the left
    gripper has let go of C2's grip and is rising (the two wrists stay apart: the left then works behind C2's handle,
    y over +14 cm, the right at C1, y -2.5 cm), then closes the near clamp."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    seat = left_seat_moves()
    seated_at = _duration(seat)
    left = list(seat)
    if variant == "no_clamps":
        left.append(_w(2.0, *PARK_NC, OPEN, "aside", YAW_Y))
        return {"left": left}
    left += clamp_moves(L.C2, 0.20, PARK_L)
    c2_done = _until(left, f"straight up off {L.C2.name}")  # the left gripper is off C2's handle and rising
    right = [_w(c2_done, *WAIT_R[:2], WAIT_R[2], OPEN, "wait aside", ARC_YAW[L.C1.name])]
    right += clamp_moves(L.C1, 0.17, PARK_R)
    if variant == "left_only":
        return {"left": left}
    return {"left": left, "right": right}


VARIANTS = ("no_clamps", "left_only")
PLAN_YAWS = {side: [(w.duration, w.yaw) for w in plan] for side, plan in plans("").items()}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_clamps  the left arm seats the casting and the clamps stay open (0.25)
      left_only  the left arm seats the casting and shuts C2; C1 stays open (0.5)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
