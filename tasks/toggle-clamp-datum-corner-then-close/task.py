"""On the station table, real size: a steel fixture plate (160 x 320 x 12 mm) with two datum stops (S1 across its far
side, S2 along its right side) and two hold-down toggle clamps (C1 on the near side, C2 on the left side), both open
and resting on their stops; a splash guard behind the plate; and, to the left of the plate, a staging block with a
aluminium housing (the casting, 60 x 100 x 30 mm, 0.18 kg) lying on it.

Goal: the casting carried into the fixture and pushed hard into its corner (far face on S1, right face on S2, flat on
the plate), then both clamps closed over it (the pad on its top face, the lever flat and past over-centre), hands off.

The parts are mechkit compounds (mechkit.py, next to this file). One declared software rule (``_toggle_rule``, a
mechkit on_step callback, ``toggle_target``): each clamp's lever is on a stiff joint drive whose target trails the
lever while its handle is short of over-centre (Q_OC: a constant 1.5 N pull back toward open at the grip) and runs
on to 40 degrees past flat once beyond it, so the lever snaps shut and its pad presses the casting with 25 N; it
takes about 27 to 34 N lifting the grip to reopen it. The arms are read through rigstate.py (the let-go check).

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


def base_shapes(c):
    """A foot plate on the fixture plate and a post under the pivot (frame: origin on the table under the pivot)."""
    fx, fy = _xy(c, 0.032, 0.034)
    px, py = _xy(c, 2 * L.BASE_HALF_ALONG, 2 * L.BASE_HALF_ACROSS)
    top = L.Z_PIVOT - 0.014
    return [mk.box((0.0, 0.0, L.PLATE_H + L.FOOT_H / 2), (fx, fy, L.FOOT_H), L.BASE_COLOR),
            mk.box((0.0, 0.0, (L.PLATE_H + L.FOOT_H + top) / 2), (px, py, top - L.PLATE_H - L.FOOT_H), L.BASE_COLOR)]


def lever_shapes(c):
    """The lever drawn open (q = 0): built shut (arm along the clamp's direction, handle above it) and turned back by
    Q_FLAT about the pivot. Clamp frame: origin on the table under the pivot."""
    frame = c.closed_frame()
    turn0 = c.turn(0.0)
    pv = c.pivot
    org = np.array([c.px, c.py, 0.0])
    quat = tuple(float(v) for v in mk.matrix_to_quat(turn0 @ frame))

    def piece(s0, s1, t0, t1, width, color):
        centre_closed = pv + frame @ np.array([(s0 + s1) / 2, 0.0, (t0 + t1) / 2])
        centre = pv + turn0 @ (centre_closed - pv) - org
        return mk.box(tuple(float(v) for v in centre), (s1 - s0, width, t1 - t0), color, quat=quat)

    h0 = L.HANDLE_OFFSET - L.HANDLE_T / 2
    h1 = L.HANDLE_OFFSET + L.HANDLE_T / 2
    return [
        piece(-0.010, 0.014, -L.ARM_T / 2, h1, L.ARM_WIDTH, L.LEVER_COLOR),  # hub joining arm and handle
        piece(0.0, c.arm, -L.ARM_T / 2, L.ARM_T / 2, L.ARM_WIDTH, L.LEVER_COLOR),  # arm bar
        piece(c.pad_s - L.PAD_SIZE / 2, c.pad_s + L.PAD_SIZE / 2, L.PAD_T_BOTTOM, -L.ARM_T / 2, L.PAD_SIZE,
              L.PAD_RED),  # the red pad
        piece(0.0, c.handle, h0, h1, L.HANDLE_WIDTH, L.LEVER_COLOR),  # handle bar
        piece(c.handle - L.GRIP_L, c.handle, L.HANDLE_OFFSET - L.GRIP_T / 2, L.HANDLE_OFFSET + L.GRIP_T / 2,
              L.GRIP_WIDTH, L.GRIP_COLOR),  # the grip
    ]


def make_clamp(c):
    lever = mk.hinge("handle", "Y" if c is L.C1 else "X", anchor=(0.0, 0.0, L.Z_PIVOT), shapes=lever_shapes(c),
                     lower=0.0, upper=L.Q_UPPER, mass=L.LEVER_MASS, spring=c.press_k, damping=L.DAMPING)
    return mk.assembly(c.name, origin=(c.px, c.py, L.TOP), shapes=base_shapes(c), joints=[lever], color=L.BASE_COLOR,
                       friction=L.LEVER_FRICTION, tight_fit=True)


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


# -- the one declared rule: the toggle's over-centre action, through a stiff joint drive ---------------------------
def toggle_target(c, q, ramp):
    """One control step of the toggle rule for clamp ``c`` (layout.Clamp) with its lever at ``q`` (rad): returns
    (drive target, ramp). ``ramp`` is None while the clamp is open; once the handle passes over-centre (Q_OC) it
    holds the drive's target, which runs on at SNAP_RATE to Q_TARGET_SHUT (40 degrees past flat, beyond the far stop),
    so the lever shuts and presses whatever is under its pad with up to PRESS_FORCE; a lever forced back below
    Q_RELEASE is open again. Open, the target trails the lever by OPEN_BIAS: a constant pull toward the open stop
    (OPEN_FORCE at the grip). A state that cannot be read pulls to open."""
    try:
        q = float(q)
    except (TypeError, ValueError):
        return 0.0, None
    if not math.isfinite(q):
        return 0.0, None
    if ramp is None and q >= L.Q_OC:
        ramp = q
    elif ramp is not None and q < L.Q_RELEASE:
        ramp = None
    if ramp is None:
        return max(0.0, q - c.open_bias), None
    ramp = min(L.Q_TARGET_SHUT, ramp + L.SNAP_RATE / L.CONTROL_HZ)
    return ramp, ramp


_RAMP = {CLAMP1: None, CLAMP2: None}


@mk.on_step
def _toggle_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    for name, part, c in ((CLAMP1, CLAMP1_PART, L.C1), (CLAMP2, CLAMP2_PART, L.C2)):
        s = states.get(name)
        try:
            q = s.joints[S.HANDLE] if s is not None else 0.0
        except (AttributeError, KeyError, TypeError):
            q = 0.0
        target, _RAMP[name] = toggle_target(c, q, _RAMP.get(name))
        mk.set_targets(part, S.HANDLE, target)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part rigidly and drops it straight down; it neither pushes the casting nor
# turns a lever. Offline only (never in the simulator), this model gives the scorer: (1) the casting carried by a
# gripper that holds it across its 60 mm (follows the TCP) and dropped onto the highest support under it when let go;
# (2) pushed by a closed gripper's block (its face PAD_FACE ahead of the TCP along the closing direction) toward +x
# until S1 or toward -y until S2; (3) a lever turned by the block's bottom front corner riding on the handle's back
# face (layout.contact_psi); past over-centre it goes on to flat by itself (on the casting's top, or to its far stop
# when nothing is under the pad), short of it it springs back to open.
class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.cx, self.cy, self.cz = L.START_X, L.START_Y, L.START_Z
        self.carry: dict = {}  # arm -> (dx, dy, dz) of the casting's centre from the TCP
        self.psi = {CLAMP1: -math.radians(30.0), CLAMP2: -math.radians(30.0)}  # lever angle from vertical
        self.shut = {CLAMP1: False, CLAMP2: False}

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
        pushed = {CLAMP1: None, CLAMP2: None}
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            if opening > 0.2 or side in self.carry:
                continue
            x, y = float(tcp[0]), float(tcp[1])
            tip_z = float(tips[2]) - L.TOP
            x_closing = abs(math.sin(self._yaw(side))) > 0.5
            # the casting
            if abs(tip_z + L.FIST_H / 2 - self.cz) < L.CS_H / 2 + L.FIST_H / 2 - 0.002 and tip_z < self.cz + L.CS_H / 2:
                if x_closing and abs(y - self.cy) < L.CS_W / 2 + L.FIST_HALF:
                    behind = x - L.PAD_FACE <= self.cx - L.CS_L / 2 + 0.0005
                    face = x + L.PAD_FACE
                    if behind and face > self.cx - L.CS_L / 2:
                        self.cx = min(face + L.CS_L / 2, L.SEAT_X)
                elif (not x_closing) and abs(x - self.cx) < L.CS_L / 2 + L.FIST_HALF:
                    behind = y + L.PAD_FACE >= self.cy + L.CS_W / 2 - 0.0005
                    face = y - L.PAD_FACE
                    if behind and face < self.cy + L.CS_W / 2:
                        self.cy = max(face - L.CS_W / 2, L.SEAT_Y)
            # the levers
            for name, c in ((CLAMP1, L.C1), (CLAMP2, L.C2)):
                if self.shut[name]:
                    continue
                if x_closing != (abs(c.dir[0]) > 0.5):
                    continue
                lateral = abs((y - c.py) if x_closing else (x - c.px))
                if lateral > L.GRIP_WIDTH / 2 + L.FIST_HALF:
                    continue
                front = (x + L.PAD_FACE) if x_closing else (y - L.PAD_FACE)
                a = L.plan_ab(c, (front, c.py) if x_closing else (c.px, front))
                # the block's corner must be on the handle's back face now, or inside the handle up to 40 mm
                # (the arm's step); a corner far in front of the open handle pushes nothing
                p0 = self.psi[name]
                d = (a, tip_z - L.Z_PIVOT)
                t_c = -d[0] * math.cos(p0) + d[1] * math.sin(p0)
                s_c = d[0] * math.sin(p0) + d[1] * math.cos(p0)
                ft = L.face_t(c, s_c)
                if not (0.0 <= s_c <= c.handle + 0.004 and ft - 0.040 <= t_c <= ft + 0.0005):
                    continue
                psi, s = L.contact_psi(c, a, tip_z)
                if psi is not None and psi > p0 - 1e-9:
                    pushed[name] = psi
        for name, c in ((CLAMP1, L.C1), (CLAMP2, L.C2)):
            if self.shut[name]:
                continue
            if pushed[name] is not None:
                self.psi[name] = min(pushed[name], math.radians(100.0))
            elif self.psi[name] < L.Q_OC - L.LEAN:
                self.psi[name] = -L.LEAN  # springs back to the open stop
            if self.psi[name] >= L.Q_OC - L.LEAN:
                self.shut[name] = True
        for name, c in ((CLAMP1, L.C1), (CLAMP2, L.C2)):
            if self.shut[name]:
                over = (abs(c.pad_bottom(L.Q_FLAT)[0] - self.cx) <= L.CS_L / 2 - L.PAD_INSET
                        and abs(c.pad_bottom(L.Q_FLAT)[1] - self.cy) <= L.CS_W / 2 - L.PAD_INSET
                        and abs(self.cz - L.CS_H / 2 - L.PLATE_H) < 0.002)
                self.psi[name] = (L.Q_FLAT - L.LEAN) if over else (L.Q_UPPER - L.LEAN)

    def q(self, name: str) -> float:
        return self.psi[name] + L.LEAN

    def states(self) -> dict:
        cs = mk.state_from_poses(CASTING_PART, (self.cx, self.cy, L.TOP + self.cz, 0.0, 0.0, 0.0, 1.0),
                                 source="offline")
        out = {CASTING: cs}
        for name, part, c in ((CLAMP1, CLAMP1_PART, L.C1), (CLAMP2, CLAMP2_PART, L.C2)):
            out[name] = mk.state_from_poses(part, (c.px, c.py, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                            joints={S.HANDLE: self.q(name)}, source="offline")
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
    _RAMP.update({CLAMP1: None, CLAMP2: None})
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@mk.on_reset
def _kit_reset() -> None:
    _RAMP.update({CLAMP1: None, CLAMP2: None})
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
GRIP_DOWN = 0.018  # pad tips 18 mm under the casting's top
GRIP_Z = L.ST_H + L.CS_H - GRIP_DOWN + PAD_TIP  # 0.045: TCP height at the pick
CARRY_Z = 0.095  # casting's underside 70 mm up: clear of the clamps' bases (40 mm)
GRIP_TO_BOTTOM = GRIP_Z - L.ST_H  # TCP height above the casting's underside while held (25 mm)
LANE_X = 0.240
SHIFT_Y = 0.045  # the casting's centre while it crosses toward the plate: clear of C1's lever (y -37 to -13 mm)
SHIFT_Z = 0.040 + GRIP_TO_BOTTOM  # its underside 40 mm up: over the stops (24 mm)
PLACE_Z = L.PLATE_H + 0.003 + GRIP_TO_BOTTOM  # set down 3 mm over the plate
PUSH_TIPS = 0.026  # pad tips 26 mm up: on the casting's side face from 26 to 42 mm, 14 mm over the plate
PUSH_Z = PUSH_TIPS + PAD_TIP
PUSH1_Y = 0.030  # inside S1's 90 mm face, clear of C1's base (y -42 to -8 mm)
PUSH1_X0 = L.PLACE_NEAR_X - 0.012 - L.PAD_FACE  # block face 12 mm short of the near face
PUSH2_X = L.SEAT_X
PUSH2_Y0 = L.PLACE_LEFT_Y + 0.012 + L.PAD_FACE  # block face 12 mm short of the left face
OVERTRAVEL = 0.008
PUSH_SPEED = 0.020  # m/s
HOVER_L = 0.12
PARK_L = (0.22, 0.30, 0.200)

# the levers: the closed fingers' block rides the handle's back face
S_PUSH = {L.C1.name: 0.045, L.C2.name: 0.060}
ARC_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 72.0)
ARC_STEP_S = 0.9
LEAD = 0.012  # the block starts this far short of the handle's face


def arc_tcp(c, psi_deg):
    """TCP (x, y, z above the table) that puts the block's bottom front corner on the handle's back face at ``psi``."""
    a, z = L.block_corner(c, math.radians(psi_deg), S_PUSH[c.name])
    if abs(c.dir[0]) > 0.5:
        return (c.px + a - L.PAD_FACE, c.py, z + PAD_TIP)
    return (c.px, c.py - a + L.PAD_FACE, z + PAD_TIP)


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
    12 mm short of S1 and 12 mm off S2, let go, close the fingers into a block and push it +x into S1, then -y into
    S2. Ends 10 cm over the plate with the block closed."""
    x, y = L.START_X, L.START_Y
    return [
        _w(2.0, x, y, 0.16, OPEN_L, "over the casting on the staging block", YAW_XL),
        _w(1.0, x, y, GRIP_Z + 0.030, OPEN_L, "down to just above it", YAW_XL),
        _w(1.5, x, y, GRIP_Z, OPEN_L, "slowly down round it", YAW_XL),
        _w(0.5, x, y, GRIP_Z, OPEN_L, "settle", YAW_XL),
        _w(1.5, x, y, GRIP_Z, GRIP, "grip the casting across its 60 mm", YAW_XL),
        _w(2.0, x, y, CARRY_Z, GRIP, "lift it 70 mm clear", YAW_XL),
        _w(1.0, LANE_X, y, CARRY_Z, GRIP, "slide it to the lane clear of the left clamp", YAW_XL),
        _w(3.0, LANE_X, SHIFT_Y, CARRY_Z, GRIP, "carry it toward the plate", YAW_XL),
        _w(1.5, L.PLACE_X, SHIFT_Y, CARRY_Z, GRIP, "over the plate", YAW_XL),
        _w(1.0, L.PLACE_X, SHIFT_Y, SHIFT_Z, GRIP, "down over the stops", YAW_XL),
        _w(1.5, L.PLACE_X, L.PLACE_Y, SHIFT_Z, GRIP, "across to its place", YAW_XL),
        _w(1.5, L.PLACE_X, L.PLACE_Y, PLACE_Z, GRIP, "set it down", YAW_XL),
        _w(0.5, L.PLACE_X, L.PLACE_Y, PLACE_Z, GRIP, "settle", YAW_XL),
        _w(1.5, L.PLACE_X, L.PLACE_Y, PLACE_Z, OPEN_L, "let go", YAW_XL),
        _w(1.0, L.PLACE_X, L.PLACE_Y, HOVER_L, OPEN_L, "straight up", YAW_XL),
        _w(2.0, PUSH1_X0, PUSH1_Y, HOVER_L, CLOSED, "close the fingers into a block, over the casting's near face",
           YAW_XL),
        _w(2.0, PUSH1_X0, PUSH1_Y, PUSH_Z, CLOSED, "down beside its near face", YAW_XL),
        _w(abs(L.PLACE_GAP_X + 0.012 + OVERTRAVEL) / PUSH_SPEED, PUSH1_X0 + 0.012 + L.PLACE_GAP_X + OVERTRAVEL,
           PUSH1_Y, PUSH_Z, CLOSED, "push it +x into S1", YAW_XL),
        _w(0.5, PUSH1_X0 + 0.012 + L.PLACE_GAP_X + OVERTRAVEL, PUSH1_Y, PUSH_Z, CLOSED, "settle against S1", YAW_XL),
        _w(1.0, PUSH1_X0, PUSH1_Y, PUSH_Z, CLOSED, "back off", YAW_XL),
        _w(1.0, PUSH1_X0, PUSH1_Y, HOVER_L, CLOSED, "up", YAW_XL),
        _w(2.0, PUSH2_X, PUSH2_Y0, HOVER_L, CLOSED, "over the casting's left face", YAW_Y),
        _w(2.0, PUSH2_X, PUSH2_Y0, PUSH_Z, CLOSED, "down beside its left face", YAW_Y),
        _w((0.012 + L.PLACE_GAP_Y + OVERTRAVEL) / PUSH_SPEED, PUSH2_X,
           PUSH2_Y0 - 0.012 - L.PLACE_GAP_Y - OVERTRAVEL, PUSH_Z, CLOSED, "push it -y into S2", YAW_Y),
        _w(0.5, PUSH2_X, PUSH2_Y0 - 0.012 - L.PLACE_GAP_Y - OVERTRAVEL, PUSH_Z, CLOSED, "settle against S2", YAW_Y),
        _w(1.0, PUSH2_X, PUSH2_Y0, PUSH_Z, CLOSED, "back off", YAW_Y),
        _w(1.0, PUSH2_X, PUSH2_Y0, HOVER_L + 0.06, CLOSED, "up", YAW_Y),
    ]


def arm_moves(c, side_yaw, start_hover, park):
    """A block pushing clamp ``c`` shut: hover over the air in front of the open handle, down, then the arc."""
    first = arc_tcp(c, ARC_DEG[0])
    if abs(c.dir[0]) > 0.5:
        lead = (first[0] - LEAD, first[1], first[2])
    else:
        lead = (first[0], first[1] + LEAD, first[2])
    plan = [
        _w(2.0, lead[0], lead[1], start_hover, CLOSED, f"over the air in front of {c.name}'s open handle", side_yaw),
        _w(2.0, lead[0], lead[1], lead[2], CLOSED, "down beside the handle", side_yaw),
    ]
    # first contact: the block comes up to the handle's face at the open stop
    plan.append(_w(1.0, first[0], first[1], first[2], CLOSED, f"to {c.name}'s handle", side_yaw))
    for deg in ARC_DEG[1:]:
        p = arc_tcp(c, deg)
        plan.append(_w(ARC_STEP_S, p[0], p[1], p[2], CLOSED, f"push {c.name}'s handle to {deg:.0f} deg", side_yaw))
    last = arc_tcp(c, ARC_DEG[-1])
    plan.append(_w(0.5, last[0], last[1], last[2], CLOSED, "settle", side_yaw))
    plan.append(_w(1.0, last[0], last[1], last[2] + 0.060, CLOSED, f"straight up off {c.name}", side_yaw))
    plan.append(_w(2.0, park[0], park[1], park[2], OPEN, "aside", side_yaw))
    return plan


PARK_R = (0.14, -0.20, 0.150)
WAIT_R = (0.20, -0.17, 0.200)


def plans(variant: str = "") -> dict:
    """Both arms' plans. Order of work: the left arm seats the casting (pick, carry, set down, push +x, push -y), then
    goes round to the left clamp and shuts it; the right arm waits aside, and shuts the near clamp once the left arm
    has moved off the casting."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    seat = left_seat_moves()
    seated_at = _duration(seat)
    left = list(seat)
    if variant == "no_clamps":
        left.append(_w(2.0, *PARK_L[:2], PARK_L[2], OPEN, "aside", YAW_Y))
        return {"left": left}
    to_c2 = arm_moves(L.C2, YAW_Y, 0.20, PARK_L)
    left += to_c2
    left_done = seated_at + _duration(to_c2[:-1])  # the left arm starts moving aside here
    right = [_w(left_done, *WAIT_R[:2], WAIT_R[2], OPEN, "wait aside", YAW_XR)]
    right += arm_moves(L.C1, YAW_XR, 0.15, PARK_R)
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
