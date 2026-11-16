"""On the station table, real size: a bench micrometer stand (a steel base 185 x 90 x 30 mm with a lapped work plate
flush with the anvil's face, two guide rails, a rear post with a clamp) holds an outside micrometer upright, anvil down,
its C-frame open toward the robot. The spindle (0.5 mm per thimble turn) starts open to 20.30 mm. A grade 0 gauge block,
20.000 mm thick (35 x 9 mm faces), stands on the plate in the rails' channel 55 mm in front of the faces.

Goal: the block slid in between the faces, the spindle run down by the thimble and finished with the ratchet knob until
it clicks at least three times, the lock lever flipped, hands off. The thimble is never used to squeeze the block.

The stand, frame and controls are one mechkit articulation (mechkit.py, next to this file); the gauge block is a mechkit
compound. Declared software rules (``_rules``, a mechkit on_step callback, written down in task.json's fidelity note):
the screw (the spindle's drive target is the thimble's turn at 0.5 mm per turn, stopped at the block), the ratchet (the
knob turns the thimble only until the faces touch the block; after that the thimble holds and the knob slips, one click
per 15 degrees) and the lever's cam (a two-notch detent that also holds the thimble and knob). The arms are read through
rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Counterfactuals (never the submitted demo):
DEMO_VARIANT=<name>, see VARIANTS.  Trace: TASK_TRACE=<file> (one JSON line per control step).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import BLOCK, MIC  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = ("Put the gauge block between the micrometer's faces, close the spindle on it with the ratchet knob "
               "until it clicks three times, and set the lock lever. Do not use the thimble to squeeze the block.")

# -- the parts --------------------------------------------------------------------------------------------------------
SPINDLE = mk.slider(L.SPINDLE, "-Z", L.spindle_shapes(mk), upper=L.SP_HI, lower=L.SP_LO, mass=L.SP_MASS,
                    spring=L.SP_K, damping=L.SP_D, color=L.CARBIDE)
THIMBLE = mk.hinge(L.THIMBLE, "-Z", (0.0, 0.0, 0.5 * (L.THIM_Z0 + L.THIM_Z1)), L.thimble_shapes(mk), upper=L.THIM_HI,
                   lower=L.THIM_LO, mass=L.THIM_MASS, spring=L.THIM_K, damping=L.THIM_D, color=L.SILVER)
RATCHET = mk.hinge(L.RATCHET, "-Z", (0.0, 0.0, 0.5 * (L.RAT_Z0 + L.RAT_Z1)), L.ratchet_shapes(mk), upper=L.RAT_HI,
                   lower=L.RAT_LO, mass=L.RAT_MASS, spring=L.RAT_K, damping=L.RAT_D, parent=L.THIMBLE,
                   color=L.KNOB_BLACK)
LEVER = mk.hinge(L.LEVER, "-Z", (0.0, 0.0, 0.5 * (L.LEV_Z0 + L.LEV_Z1)), L.lever_shapes(mk), upper=L.LEV_HI,
                 lower=L.LEV_LO, mass=L.LEV_MASS, spring=L.LEV_K, damping=L.LEV_D, color=L.LEVER_RED)
MIC_PART = mk.assembly(MIC, origin=(L.X_A, L.Y_A, L.TOP), shapes=L.base_shapes(mk),
                       joints=[SPINDLE, THIMBLE, RATCHET, LEVER], color=L.STAND_GREY, friction=L.FRICTION,
                       tight_fit=True)
BLOCK_PART = mk.assembly(BLOCK, origin=(L.BL_X0, L.Y_A, L.TOP + L.BL_Z0), mass=L.BLOCK_MASS,
                         shapes=[mk.box((0.0, 0.0, 0.0), (L.BL_X, L.BL_Y, L.BL_Z), L.BLOCK_STEEL)],
                         color=L.BLOCK_STEEL, friction=L.FRICTION, tight_fit=True)
OBJECTS = [MIC_PART, BLOCK_PART]
SPECS = {MIC: MIC_PART, BLOCK: BLOCK_PART}


# -- the declared rules: the screw, the ratchet and the lever's cam ---------------------------------------------------
def lever_target(q: float) -> float:
    """The lever's drive target (rad) at ``q`` (0 = as found, LEV_Q_LOCK = flipped): a two-notch cam. Past LEV_SNAP_ON
    the spring pulls it to the lock stop, preloaded LEV_PRELOAD past it; short of LEV_SNAP_OFF it pulls it back to the
    free stop; between the two it follows the joint (lets go)."""
    if q >= L.LEV_SNAP_ON:
        return L.LEV_Q_LOCK + L.LEV_PRELOAD
    if q <= L.LEV_SNAP_OFF:
        return -L.LEV_PRELOAD
    return q


def coupling_targets(theta: float, phi: float, between: bool, lock_theta: float | None) -> tuple[float, float]:
    """The thimble's and the ratchet knob's drive targets (rad): ``theta`` the thimble's turn from as found, ``phi`` the
    knob's turn on the thimble. Locked: both hold. Free: the thimble follows the knob (and the knob the thimble: the
    knob is on the thimble) up to the turn at which the faces touch the block (only when the block is between the
    faces). Faces touching and the knob ahead of the thimble: the thimble holds and the knob is free: it slips."""
    if lock_theta is not None:
        return lock_theta, phi
    stop = L.THETA_C if between else None
    if stop is not None and phi > 0.0 and theta >= stop - L.SLIP_EPS:
        return stop, phi  # held at the touch turn, not where it is now: the knob's drag must not creep it on
    target = theta + phi
    if stop is not None:
        target = min(target, stop)
    return float(np.clip(target, L.THIM_LO, L.THIM_HI)), 0.0


def spindle_target(theta: float, between: bool) -> float:
    """The spindle's drive target (m toward the anvil): 0.5 mm per thimble turn, stopped where the faces touch the block
    (nothing is pressed on the block by the drive: the squeeze is the thimble's turn past touch, see scoring), less the
    drive's sag under the spindle's weight."""
    turn = min(theta, L.THETA_C) if between else theta
    return float(np.clip(L.adv_of(turn) - L.SP_SAG, L.SP_LO, L.SP_HI))


_LOCK: dict = {"theta": None}


def _lock_state(theta: float, lever: float) -> float | None:
    """The lever's lock: once it is flipped (80 degrees) the thimble's turn is held; let off when it is flipped back."""
    if lever >= L.LOCK_Q:
        if _LOCK["theta"] is None:
            _LOCK["theta"] = theta
    elif lever < L.LOCK_Q - 0.2:
        _LOCK["theta"] = None
    return _LOCK["theta"]


@mk.on_step
def _rules(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    try:
        r = S.read(states)
        if r is None:
            return
        btw = S.between(r)
        lock = _lock_state(r.theta, r.lever)
        tt, tp = coupling_targets(r.theta, r.phi, btw, lock)
        mk.set_targets(MIC_PART, L.THIMBLE, tt)
        mk.set_targets(MIC_PART, L.RATCHET, float(np.clip(tp, L.RAT_LO, L.RAT_HI)))
        mk.set_targets(MIC_PART, L.SPINDLE, spindle_target(r.theta, btw))
        mk.set_targets(MIC_PART, L.LEVER, lever_target(r.lever))
    except (AttributeError, KeyError, TypeError, ValueError):
        return


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy knows neither the screw nor the ratchet nor the lever, and carries a gripped part rigidly. Offline
# only (never in the simulator), this model gives the scorer: (1) the block pushed along -x by a closed gripper whose
# flat face meets its end (stopped by the frame's column); (2) the knob (or, gripped low, the thimble) turned by a
# gripper closed on it by as much as that gripper's yaw turns, read from the demo's own plan (the checker's snapshot has
# no yaw); (3) the ratchet and the screw by the same rules as in the simulator; (4) the lever pushed over by a closed
# gripper's flat face and snapped by its cam.
PAD_FACE = 0.005  # a closed pad block's face ahead of the TCP along the closing direction, less the arm's trail
# (the shim task's GPU run: the face 5.9 mm ahead of the TCP, the TCP 0.9 mm behind its command at the end of a push)
PAD_TIP = 0.013  # the pad tips are this far past the TCP
DT = 1.0 / L.CONTROL_HZ
KNOB_GAP = 0.0185  # fingers closed on the 18 mm knob (or the 20 mm thimble) when the pads are under this far apart
KNOB_AXIS_TOL = 0.006
KNOB_TIPS = (0.0555, 0.0625)  # the pad tips' height over the plate: on the knob only
THIM_TIPS = (0.0345, 0.0545)  # lower: the pads take the thimble (and the knob above it)
PUSH_TIPS = (0.002, 0.018)  # a closed gripper's tips over the plate: against the block's end
LEV_TIPS = (0.0255, 0.0345)  # against the lever's slab


def _smooth(s: float) -> float:
    s = min(max(float(s), 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def plan_yaw(wps, t: float) -> float:
    """The grip's yaw (rad) at ``t`` s into a plan: the stub's and the real ArmPlan's one-axis smoothstep."""
    prev = wps[0].yaw
    for w in wps:
        if t < w.duration:
            return prev + _smooth(t / w.duration) * (w.yaw - prev)
        t -= w.duration
        prev = w.yaw
    return prev


def lever_angle_for(face_x: float) -> float:
    """The lever's turn (rad) when a flat face at ``face_x`` (m from the axis, on its -x side) has pushed its tip round:
    the -x edge of the slab's tip is at LEV_L sin(a) - LEV_W/2 cos(a)."""
    lo, hi = 0.0, math.pi / 2
    f = lambda a: L.LEV_L * math.sin(a) - L.LEV_W / 2 * math.cos(a)  # noqa: E731
    if face_x <= f(lo):
        return 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) < face_x:
            lo = mid
        else:
            hi = mid
    return hi


def lever_tip_y(a: float) -> float:
    return L.LEV_L * math.cos(a) + L.LEV_W / 2 * math.sin(a)


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.u = L.BL_START_DX  # the block's centre in front of the axis
        self.theta = 0.0
        self.phi = 0.0
        self.lever = 0.0
        self.lock_theta: float | None = None
        self._plan = None

    def plan(self) -> dict:
        if self._plan is None:
            self._plan = demo()
        return self._plan

    def between(self) -> bool:
        return S.block_between((self.u, 0.0, L.BL_Z / 2), np.eye(3))

    def _dyaw(self, side: str) -> float:
        wps = self.plan().get(side)
        if not wps:
            return 0.0
        t = self.k * DT
        return plan_yaw(wps, t) - plan_yaw(wps, max(0.0, t - DT))

    def _turn(self, dpsi: float, thimble: bool) -> None:
        """The hand turns what it holds by ``dpsi`` rad (+ closing)."""
        if self.lock_theta is not None or dpsi == 0.0:
            return
        if thimble:
            self.theta = float(np.clip(self.theta + dpsi, L.THIM_LO, L.THIM_HI))
            return
        stop = L.THETA_C if self.between() else L.THIM_HI
        if dpsi > 0.0:
            if self.phi > 0.0 or (self.between() and self.theta >= stop - L.SLIP_EPS):
                self.phi = min(self.phi + dpsi, L.RAT_HI)  # slipping
            else:
                room = stop - self.theta
                if dpsi <= room:
                    self.theta += dpsi
                else:
                    self.theta = stop
                    self.phi = min(self.phi + dpsi - room, L.RAT_HI)
        else:
            take = min(self.phi, -dpsi)
            self.phi -= take
            self.theta = max(L.THIM_LO, self.theta - (-dpsi - take))

    def step(self, arms) -> None:
        self.k += 1
        read = rigstate.read_arms(arms)
        if read is None:
            return
        axis = np.array([L.X_A, L.Y_A])
        plate = L.TOP + L.PL
        for side, (tcp, tips, opening) in zip(rigstate.ARMS, read):
            gap = float(opening) * rigstate.FULL_GAP
            tz = float(tips[2]) - plate
            tcp = np.asarray(tcp, dtype=float)
            if gap <= KNOB_GAP and float(np.linalg.norm(tcp[:2] - axis)) <= KNOB_AXIS_TOL:
                if KNOB_TIPS[0] <= tz <= KNOB_TIPS[1]:
                    self._turn(-self._dyaw(side), False)
                elif THIM_TIPS[0] <= tz < THIM_TIPS[1]:
                    self._turn(-self._dyaw(side), True)
                continue
            if gap > 0.004:
                continue
            if PUSH_TIPS[0] <= tz <= PUSH_TIPS[1] and abs(float(tcp[1]) - L.Y_A) <= 0.012:
                face = float(tcp[0]) - PAD_FACE  # a closed block pushing toward -x
                front = L.X_A + self.u + L.BL_X / 2
                if float(tcp[0]) > L.X_A + self.u and face < front:
                    self.u = max(face - L.BL_X / 2 - L.X_A, -L.THROAT + L.BL_X / 2)
            elif LEV_TIPS[0] <= tz <= LEV_TIPS[1] and 0.012 <= float(tcp[1]) - L.Y_A <= 0.050 \
                    and abs(float(tcp[0]) - L.X_A) <= 0.045:
                face_x = float(tcp[0]) + PAD_FACE - L.X_A  # a closed block pushing toward +x
                a = lever_angle_for(face_x)
                if a > 0.0 and abs(lever_tip_y(a) - (float(tcp[1]) - L.Y_A)) <= 0.010:
                    self.lever = max(self.lever, a)
        if self.lever >= L.LEV_SNAP_ON:
            self.lever = L.LEV_Q_LOCK + L.LEV_PRELOAD
        elif self.lever <= L.LEV_SNAP_OFF:
            self.lever = 0.0
        if self.lever >= L.LOCK_Q and self.lock_theta is None:
            self.lock_theta = self.theta
        elif self.lever < L.LOCK_Q - 0.2:
            self.lock_theta = None

    def states(self) -> dict:
        q = spindle_target(self.theta, self.between()) + L.SP_SAG
        joints = {L.SPINDLE: q, L.THIMBLE: self.theta, L.RATCHET: self.phi, L.LEVER: self.lever}
        q0 = (0.0, 0.0, 0.0, 1.0)
        mic = mk.state_from_poses(MIC_PART, (L.X_A, L.Y_A, L.TOP, *q0), joints=joints, source="offline")
        blk = mk.state_from_poses(BLOCK_PART, (L.X_A + self.u, L.Y_A, L.TOP + L.BL_Z0, *q0), source="offline")
        return {MIC: mic, BLOCK: blk}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(MIC), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _reset_all() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _LOCK["theta"] = None


@rigstate.on_reset
def _new_episode() -> None:
    _reset_all()


@mk.on_reset
def _kit_reset() -> None:
    _reset_all()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: the controls, the block, what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The block between the faces, the gap 20.000 mm within 0.03 mm, three or more clicks, the lever locked, both
    grippers 30 mm clear and every part still for 1 s, no latched fall or squeeze; held 1 s. This runs first every
    control step, so it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _reset_all()
        _EPISODE["mechkit"] = mk.episode()
    parts = {n: objects.get(n) for n in S.PARTS}
    arms = rigstate.arms()
    if no_simulator(parts):
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
    row["source"] = getattr(parts.get(MIC), "source", None)
    row["lock_theta"] = _LOCK["theta"]
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the block left the plate or the thimble squeezed it, or unless every part is let go and still; 1.0 the block
    between the faces, the gap within 0.03 mm of 20.000, three or more clicks, locked; 0.75 the same unlocked; 0.5 the
    same with one or two clicks; 0.25 between the faces within 0.20 mm otherwise; 0 otherwise. The score of the latest
    scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points; the pad tips are
# 13 mm below. Finger targets: the joint value is minus half the commanded gap (8 mm narrower than the part).
OPEN = -0.04695  # 94 mm
CLOSED = 0.0
GRIP_K = -(2 * L.RAT_R - 0.008) / 2  # 10 mm commanded across the 18 mm knob
YAW_R = math.pi / 2  # right: fingers close along x, to reach the midline
YAW_L = -math.pi / 2  # left
Z0 = L.TOP  # waypoints are heights over the table top


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (x, y, L.TOP + z), finger, label, yaw)


def _duration(plan) -> float:
    return sum(w.duration for w in plan)


# the right arm: push the block in along its channel with a closed pad block
PUSH_TCP = L.PL + 0.0055 + PAD_TIP  # pad tips 5.5 mm over the plate: 3 mm over the rails
BLOCK_FRONT0 = L.X_A + L.BL_START_DX + L.BL_X / 2
PUSH_START_X = BLOCK_FRONT0 + PAD_FACE + 0.012  # the closed block comes down 12 mm in front of the block (in the air)
BLOCK_END_U = -0.002  # the block's centre 2 mm behind the axis at the end of the push (1 mm short of the column)
PUSH_END_X = L.X_A + BLOCK_END_U + L.BL_X / 2 + PAD_FACE
PUSH_SPEED = 0.025  # m/s
HOVER_R = 0.110
BACK_OFF = 0.030
PARK_R = (0.14, -0.20, 0.150)

# the left arm: grip the ratchet knob and turn it; flip the lever
KNOB_TIPS_Z = L.RAT_Z0 + 0.0012  # pad tips 1.2 mm over the knob's foot (the thimble's top is 0.3 mm lower)
KNOB_TCP = KNOB_TIPS_Z + PAD_TIP
THIM_TIPS_Z = L.PL + 0.045  # the thimble variant: the pads take the thimble's upper 9.5 mm and the knob
THIM_TCP = THIM_TIPS_Z + PAD_TIP
HOVER_L = 0.150
WAIT_L = (0.22, 0.26, 0.150)
KA0, KA1 = math.radians(40.0), math.radians(-125.0)  # first grip: 165 degrees clockwise (seen from above)
KB0, KB1 = math.radians(40.0), math.radians(-100.0)  # second grip: 140 degrees (305 in all: 216 to touch, 89 slip)
LEV_TIPS_Z = L.PL + 0.0285
LEV_TCP = LEV_TIPS_Z + PAD_TIP
LEV_Y = 0.030
LEV_X0 = L.X_A - 0.0135  # the closed block's -x pad 5.5 mm short of the lever's -x face
LEV_X1 = L.X_A + 0.020
LEV_SPEED = 0.015
PARK_L = (0.22, 0.26, 0.150)


def push_moves():
    y = L.Y_A
    d = PUSH_START_X - PUSH_END_X
    return [
        _w(2.0, PUSH_START_X, y, HOVER_R, OPEN, "over the stand's front, in the air ahead of the block", YAW_R),
        _w(1.5, PUSH_START_X, y, HOVER_R, CLOSED, "close the fingers into a block", YAW_R),
        _w(1.5, PUSH_START_X, y, PUSH_TCP + 0.030, CLOSED, "down to just above the plate", YAW_R),
        _w(1.5, PUSH_START_X, y, PUSH_TCP, CLOSED, "slowly down: pad tips 5.5 mm over the plate", YAW_R),
        _w(d / PUSH_SPEED, PUSH_END_X, y, PUSH_TCP, CLOSED, "push the block in between the faces", YAW_R),
        _w(0.5, PUSH_END_X, y, PUSH_TCP, CLOSED, "settle", YAW_R),
        _w(1.0, PUSH_END_X + BACK_OFF, y, PUSH_TCP, CLOSED, "back off 30 mm", YAW_R),
        _w(1.5, PUSH_END_X + BACK_OFF, y, HOVER_R, OPEN, "straight up, opening", YAW_R),
        _w(1.5, *PARK_R[:2], PARK_R[2], OPEN, "aside", YAW_R),
    ]


def _twist(y0, y1, tcp, label_stem, parts=2, dur=1.7):
    x, y = L.X_A, L.Y_A
    out = []
    for i in range(parts):
        yaw = y0 + (y1 - y0) * (i + 1) / parts
        out.append(_w(dur, x, y, tcp, GRIP_K, f"{label_stem} ({i + 1} of {parts})", yaw))
    return out


def turn_moves(tcp=KNOB_TCP, b1=KB1, a1=KA1):
    """The left arm: grip the knob, turn it 165 degrees clockwise, let go, turn back, grip it again, turn it 140 more,
    let go, up."""
    x, y = L.X_A, L.Y_A
    out = [
        _w(2.0, x, y, HOVER_L, OPEN, "over the knob", KA0),
        _w(1.2, x, y, tcp + 0.020, OPEN, "down to just above it", KA0),
        _w(1.5, x, y, tcp, OPEN, "slowly down: pad tips on the knob", KA0),
        _w(0.5, x, y, tcp, OPEN, "settle", KA0),
        _w(1.5, x, y, tcp, GRIP_K, "grip the knob across its 18 mm", KA0),
    ]
    out += _twist(KA0, a1, tcp, "turn the knob clockwise")
    out += [_w(1.5, x, y, tcp, OPEN, "let go", a1)]
    out += [_w(1.0, x, y, tcp, OPEN, "turn the open hand back (1 of 2)", (a1 + KB0) / 2),
            _w(1.0, x, y, tcp, OPEN, "turn the open hand back (2 of 2)", KB0),
            _w(1.5, x, y, tcp, GRIP_K, "grip the knob again", KB0)]
    out += _twist(KB0, b1, tcp, "turn the knob clockwise again")
    out += [_w(1.5, x, y, tcp, OPEN, "let go of the knob", b1),
            _w(1.5, x, y, HOVER_L, OPEN, "straight up", b1)]
    return out


def lever_moves():
    """The left arm: closed fingers beside the lever, push its tip round 90 degrees (the cam snaps it), up, aside."""
    y = LEV_Y
    return [
        _w(2.0, LEV_X0, y, HOVER_L, CLOSED, "over the lever's side, closing the fingers into a block", YAW_L),
        _w(1.2, LEV_X0, y, LEV_TCP + 0.030, CLOSED, "down to just above the lever's height", YAW_L),
        _w(1.2, LEV_X0, y, LEV_TCP, CLOSED, "slowly down: pad tips on the lever's slab", YAW_L),
        _w((LEV_X1 - LEV_X0) / LEV_SPEED, LEV_X1, y, LEV_TCP, CLOSED, "push the lever round to the lock", YAW_L),
        _w(0.5, LEV_X1, y, LEV_TCP, CLOSED, "settle", YAW_L),
        _w(1.5, LEV_X1, y, HOVER_L, OPEN, "straight up, opening", YAW_L),
        _w(1.5, *PARK_L[:2], PARK_L[2], OPEN, "aside", YAW_L),
    ]


# Wrong ways (counterfactuals), each one the submitted demo with one change.
VARIANTS = ("no_lock", "two_clicks", "no_clicks", "thimble")


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the right arm pushes the block in between the faces and goes aside; the left arm, waiting over its
    own side, grips the knob, turns it (run-up on the knob, then the clicks), lets go, and flips the lever; then goes
    aside."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = push_moves()
    b1, a1, tcp = KB1, KA1, KNOB_TCP
    if variant == "two_clicks":
        b1 = math.radians(-50.0)  # 255 degrees in all: 39 degrees of slip, two clicks
    elif variant == "no_clicks":
        b1 = math.radians(-10.0)  # 215 degrees: the faces 1.4 um short of the block, nothing slips
    elif variant == "thimble":
        tcp = THIM_TCP  # the pads take the thimble: it is turned straight on past touch
    left = [_w(2.0, *WAIT_L[:2], WAIT_L[2], OPEN, "over its own side", KA0)]
    pushed = _duration(right) - 1.5
    left.append(_w(max(pushed - _duration(left), 0.5), *WAIT_L[:2], WAIT_L[2], OPEN, "wait for the block", KA0))
    left += turn_moves(tcp, b1, a1)
    if variant != "no_lock":
        left += lever_moves()
    else:
        left.append(_w(1.5, *PARK_L[:2], PARK_L[2], OPEN, "aside", YAW_L))
    return {"left": left, "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      no_lock     the lever never flipped (0.75)
      two_clicks  the knob turned 90 degrees on the second grip: 39 degrees of slip, 2 clicks (0.5)
      no_clicks   the knob stopped 1 degree before touch: the right gap, no click (0.25)
      thimble     the pads take the thimble and turn it straight on 89 degrees past touch: squeezed (0)"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
