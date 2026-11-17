"""On the station table, real size: an entry-level DSLR (a Canon EOS 2000D / Rebel T7 body, 129 x 101 x 78 mm, with its
kit lens, the EF-S 18-55mm f/3.5-5.6 IS II, 68.5 mm across and 70 mm long; 0.675 kg) stands on its base at x 340 mm,
its lens pointing toward -y with the lens's front face at y -30 mm. Its front cap, a Canon E-58 II centre-pinch cap
(62 mm across, 12 g, est), lies face up beside it at (250, -130) mm. The cap has two finger tabs on its front; each is
one sprung slider link (2.5 mm travel, 2 N spring, est) that carries a latch tooth on the cap's back. Pinching the
tabs pulls both teeth in; let go, they spring out behind the lip of the lens's 58 mm filter thread.

Goal: the cap back on the lens, both latches behind the lip, the camera standing on its base on the table, hands off.

The parts are mechkit compounds (mechkit.py, next to this file): the camera is one loose rigid body, the cap one loose
articulation (its plate and two fixed centring lugs, and two sprung latch links). There is no declared software rule:
the latches are geometry (a tooth's flat catch face behind the lip's rear face) and springs. The arms are read through
rigstate.py (the let-go check), and the scripted demo's sideways grip comes from tiltgrip.py. None changes what the
embodiment does.

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
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the parts are built as compounds)
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import tiltgrip  # noqa: E402
from scoring import CAMERA, CAP, LATCH_N, LATCH_P  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0  # 1.0 must hold for 1 s on top of the scorer's own 1 s of let-go and stillness

INSTRUCTION = "Put the lens cap back on the camera and leave the camera standing on the table."


# -- the parts --------------------------------------------------------------------------------------------------------
def _latch(name: str, axis: str, s: int):
    return mk.slider(name, axis, P.latch_shapes(s), upper=L.LATCH_OUT, lower=-L.TRAVEL, mass=L.LATCH_MASS,
                     spring=L.LATCH_SPRING, damping=L.LATCH_DAMP, friction=0.0, color=L.TAB_BLACK)


CAMERA_PART = mk.assembly(CAMERA, origin=L.CAM_ORIGIN, mass=L.CAM_MASS, shapes=P.camera_shapes(), color=L.CAM_BLACK,
                          friction=L.CAM_FRICTION, tight_fit=True)
CAP_PART = mk.assembly(CAP, origin=L.CAP_ORIGIN, mass=L.CAP_BASE_MASS, shapes=P.cap_base_shapes(),
                       joints=[_latch(LATCH_P, "X", +1), _latch(LATCH_N, "-X", -1)], color=L.CAP_BLACK,
                       friction=L.CAP_FRICTION, tight_fit=True)
OBJECTS = [CAMERA_PART, CAP_PART]
SPECS = {CAMERA: CAMERA_PART, CAP: CAP_PART}


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy carries a gripped part upright and drops it straight down; it has no joints, lips or latches.
# Offline only (never in the simulator), this model gives the scorer: (1) the camera carried by a gripper closed on its
# hump, turned with the gripper and, once lifted, hanging from it as the GPU measured (``hang``: the left arm bends
# under the camera's weight); (2) the cap carried rigidly (turned with the gripper) by a gripper closed on its tabs,
# which pulls both latches in; (3) a carried cap that meets the lens square and within the lugs' capture slides on
# to the face and stops there, one outside the capture stops with its teeth on the lip; (4) a cap let go on the face
# with its latches pinched in stays on (latched) and moves with the camera, any other cap drops face up to the table;
# a camera let go drops to the table, upright.
TAB_CATCH = 0.015  # a gripper closes on the tabs: the TCP within 15 mm of the tabs' centre
CAPTURE_X = 0.0045  # the pinched latches' ramps bring the cap in within 4.5 mm across the tabs (x)
CAPTURE_Z = 0.0030  # and the lugs' ramps within 3 mm the other way
CAPTURE_DEG = 4.0
SIDE_TOL = 0.010


def _rot_to_quat(r: np.ndarray) -> tuple:
    return tuple(float(v) for v in mk.matrix_to_quat(r))


def _grip_frame(tcp, tips) -> np.ndarray:
    """The gripper's frame from its TCP and pad tips: z the approach, x the closing direction (world x for every grip
    of this demo, made square to the approach)."""
    u = np.asarray(tips, dtype=float) - np.asarray(tcp, dtype=float)
    u = u / (np.linalg.norm(u) or 1.0)
    x = np.array([1.0, 0.0, 0.0]) - u * u[0]
    x = x / (np.linalg.norm(x) or 1.0)
    return np.column_stack([x, np.cross(u, x), u])


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cam = np.array(L.CAM_ORIGIN, dtype=float)
        self.cam_rot = np.eye(3)
        self.cap = np.array(L.CAP_ORIGIN, dtype=float)
        self.cap_rot = np.eye(3)
        self.q = 0.0
        self.on = False  # the cap latched on the lens
        self.cap_off = (np.zeros(3), np.eye(3))  # latched: the cap's pose in the camera's frame
        self.hold: dict = {}  # arm side -> (part, data)

    def _tab_centre(self) -> np.ndarray:
        return self.cap + self.cap_rot @ np.array([0.0, 0.0, (L.PLATE_T + L.TAB_Z1) / 2])

    def _on_hump(self, tcp, tips) -> bool:
        c = self.cam
        hy0, hy1 = c[1] + L.HUMP_Y0, c[1] + L.HUMP_Y1
        top, plate = c[2] + L.HUMP_Z1, c[2] + L.TOP_PLATE_Z
        return bool(abs(tcp[0] - c[0]) <= SIDE_TOL and hy0 <= tcp[1] <= hy1
                    and tips[2] <= top and tips[2] + rigstate.FINGER_LEN >= plate)

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            return
        cur = {side: (tcp, tips, opening * rigstate.FULL_GAP) for side, (tcp, tips, opening) in zip(rigstate.ARMS, read)}
        tabs_w = 2 * L.TAB_X1
        # let go
        for side in list(self.hold):
            name, _ = self.hold[side]
            gap = cur[side][2]
            if name == CAMERA and gap > L.HUMP_W + 0.004:
                del self.hold[side]
            elif name == CAP and gap > tabs_w + 0.004:
                del self.hold[side]
                self.q = 0.0
                f = self._fit()
                if f["p"] <= 0.0003 and f["lat"] <= 0.0015 and f["ang"] <= 1.0:
                    self.on = True
                    self.cap_off = (self.cam_rot.T @ (self.cap - self.cam), self.cam_rot.T @ self.cap_rot)
        # take hold
        for side, (tcp, tips, gap) in cur.items():
            if side in self.hold:
                continue
            if gap <= L.HUMP_W + 0.002 and self._on_hump(tcp, tips):
                g = _grip_frame(tcp, tips)
                self.hold[side] = (CAMERA, (g.T @ self.cam_rot, g.T @ (self.cam - tcp), float(tcp[2])))
            elif (not self.on and gap <= tabs_w + 0.002
                  and np.linalg.norm(tcp - self._tab_centre()) <= TAB_CATCH):
                g = _grip_frame(tcp, tips)
                self.hold[side] = (CAP, (g.T @ self.cap_rot, g.T @ (self.cap - tcp)))
        held = {name: (side, data) for side, (name, data) in self.hold.items()}
        # the camera: carried by the hand, hanging from it in proportion to the lift (the full hang at LIFT)
        if CAMERA in held:
            side, (r_rel, o_rel, z_grip) = held[CAMERA]
            tcp, tips, _ = cur[side]
            g = _grip_frame(tcp, tips)
            s = min(1.0, max(0.0, (float(tcp[2]) - z_grip) / LIFT))
            self.cam, self.cam_rot = hang(tcp, tcp + g @ o_rel, g @ r_rel, s)
        else:
            self.cam = np.array([self.cam[0], self.cam[1], L.TOP + L.CAM_AXIS_Z])
            self.cam_rot = np.eye(3)
        # the cap
        if self.on:
            off, rot = self.cap_off
            self.cap = self.cam + self.cam_rot @ off
            self.cap_rot = self.cam_rot @ rot
            return
        if CAP in held:
            side, (r_rel, o_rel) = held[CAP]
            tcp, tips, gap = cur[side]
            g = _grip_frame(tcp, tips)
            self.q = -min(L.TRAVEL, max(0.0, (tabs_w - gap) / 2))
            self.cap_rot = g @ r_rel
            self.cap = tcp + g @ o_rel
            self._meet_lens()
            return
        self.q = 0.0
        if self.cap[2] > L.CAP_ORIGIN[2] + 1e-6 or not np.allclose(self.cap_rot, np.eye(3)):
            self.cap_rot = np.eye(3)
            self.cap = np.array([self.cap[0], self.cap[1], L.CAP_ORIGIN[2]])

    def _fit(self) -> dict:
        rel = self.cam_rot.T @ (self.cap - self.cam)  # in the camera's frame, as scoring.fit
        return {"p": -float(rel[1]), "x": float(rel[0]), "z": float(rel[2]),
                "lat": math.hypot(float(rel[0]), float(rel[2])),
                "ang": S.angle_deg(self.cap_rot[:, 2], -self.cam_rot[:, 1])}

    def _meet_lens(self) -> None:
        f = self._fit()
        if f["p"] >= -L.TEETH_Z or f["ang"] > 30.0 or f["lat"] > L.LENS_R:
            return  # the teeth not yet at the lens's face
        if abs(f["x"]) <= CAPTURE_X and abs(f["z"]) <= CAPTURE_Z and f["ang"] <= CAPTURE_DEG:
            # the ramps centre it and it slides on to the face, square
            p = max(0.0, f["p"])
            self.cap = self.cam + self.cam_rot @ np.array([0.0, -p, 0.0])
            self.cap_rot = self.cam_rot @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        else:  # its teeth stop on the lip
            self.cap = self.cap + self.cam_rot @ np.array([0.0, f["p"] + L.TEETH_Z, 0.0])

    def states(self) -> dict:
        return {CAMERA: mk.state_from_poses(CAMERA_PART, (*self.cam, *_rot_to_quat(self.cam_rot)), source="offline"),
                CAP: mk.state_from_poses(CAP_PART, (*self.cap, *_rot_to_quat(self.cap_rot)),
                                         joints={LATCH_P: self.q, LATCH_N: self.q}, source="offline")}


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(CAMERA), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}
_HIST: deque = deque(maxlen=60)  # trace only: every 2nd step [step, latch_p mm, latch_n mm, cap p mm, cap lat mm]


@rigstate.on_reset
def _new_episode() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


@mk.on_reset
def _kit_reset() -> None:
    _TRACKER.reset()
    _OFFLINE.reset()
    _HIST.clear()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: every part's pose and what the scorer saw."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def success(objects) -> bool:
    """The cap on the lens with both latches behind the lip, the camera on its base on the table, both grippers 30 mm
    clear and both parts still for 1 s, no latched fall; held 1 s. This runs first every control step, so it also
    feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a second, independent reset signal
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
            _OFFLINE.reset()
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
    row["source"] = getattr(parts.get(CAMERA), "source", None)
    if result.details.get("valid_state") and _TRACKER.steps % 2 == 0:
        d = result.details
        _HIST.append([_TRACKER.steps, d["latch_p_mm"], d["latch_n_mm"], d["cap_p_mm"], d["cap_lat_mm"]])
    row["hist"] = list(_HIST)  # the last 3 s of the key numbers in every row: one sampled row tells the story
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if a part fell off the table, or unless both parts are let go and still, or the cap is not home on the lens;
    1.0 the cap latched (home, both latches behind the lip) and the camera on its base on the table; 0.5 latched with
    the camera not on its base; 0.25 home with a latch held in and the camera on its base. The score of the latest
    scored step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout and the camera's measured hang (below), nothing read from the
# simulator during a run. TCP points; the pad tips are 13 mm along the approach. Finger targets: the joint value is minus half the commanded gap.
OPEN = -0.04695  # 94 mm
GRIP_HUMP = -(L.HUMP_W - 0.008) / 2  # 50 mm across the 58 mm hump: about 20 N a pad
PINCH = -0.0135  # 27 mm across the tabs (38 mm out, 33 mm pinched home): about 15 N a pad, latches fully in
YAW_L, YAW_R = -math.pi / 2, math.pi / 2  # both close along x
ROLL = math.radians(90.0)  # the right hand turned about world x: approach +y, the cap's front toward -y
R_FLAT = YAW_R

# the right hand on the cap: pad tips 2 mm above the plate's front face
Z_PICK = -L.TEETH_Z + L.PLATE_T + 0.002 + 0.013  # 23.5 mm
CAP_OFF = Z_PICK + L.TEETH_Z  # the cap's origin 17.5 mm from the TCP along the approach
HOVER_R = 0.100
Z_ROLL = 0.140

# the left hand on the hump, closing across its 58 mm, leaned 0.4 rad about world x (its wrist toward its own base,
# tiltgrip.py): straight down, joint4 runs out of range with the camera lifted. Pad tips 9 mm above the top plate
# (the leaned pads' lower corners about 4 mm above it), the pads centred on the hump halfway down its flank.
L_ROLL = -0.4
L_GRIP = tiltgrip.tilted(YAW_L, L_ROLL)
Z_HUMP = L.CAM_AXIS_Z + L.TOP_PLATE_Z + 0.009 + 0.013 * math.cos(L_ROLL)  # 97 mm
Y_HUMP = L.CAM_Y0 + (L.HUMP_Y0 + L.HUMP_Y1) / 2 + 0.006 * math.tan(-L_ROLL)
HOVER_L = 0.160
LIFT = 0.080  # the hand lifted 80 mm (the camera, hanging from it, about 57 mm: HANG below)

# The camera hanging from the left hand. The arms are joint-PD position control that compensates gravity for their
# own links only (run log: joints 1 to 3 at 80 Nm/rad, joints 4 to 6 at 10 Nm/rad), so the 0.675 kg camera bends the
# left arm. Measured on the GPU (L40S, 17 Nov 2026), with the hand leaned 0.4 rad and lifted 80 mm: the left TCP stood
# (-2.4, +8.3, -21.3) mm off its target and the camera hung turned 4.2, 3.2 and 1.0 degrees about world x (nose
# down), y and z: the lens's face at (336.2, -17.7, 96.0) mm instead of (340, -30, 124), its axis 4.2 degrees off -y.
# HANG_TCP_1 / HANG_DEG_1 are that measurement, as a displacement of the TCP and a turn about it.
HANG_TCP_1 = (-0.0024, 0.0083, -0.0213)
HANG_DEG_1 = (4.2, 3.2, 1.0)
# The demo turns the left hand 4.7 degrees nose up as it lifts (PITCH_COMP), so the camera hangs level. A joint-
# stiffness model of the arm (those gains, the camera's weight at its centre of mass; it brackets the measurement
# above within about 2.5 mm) adds (-0.5, +1.4, -1.6) mm and (0.5, 0.15, 0.15) degrees for the pre-turned hand.
# HANG_TCP / HANG_DEG, which the demo aims by and the offline model hangs the camera by, are the measurement plus that
# difference (est): the lens's face at about (336.1, -20.4, 100.9) mm, its axis level and 1.2 degrees off -y.
PITCH_COMP = math.radians(4.7)
L_LIFT = tiltgrip.tilted(YAW_L, L_ROLL - PITCH_COMP)
HANG_TCP = (-0.0029, 0.0097, -0.0229)
HANG_DEG = (4.7, 3.35, 1.15)


def _turn(deg_xyz, s: float = 1.0) -> np.ndarray:
    w = np.radians(np.asarray(deg_xyz, dtype=float)) * s
    n = float(np.linalg.norm(w))
    return np.eye(3) if n < 1e-12 else mk.axis_angle_matrix(w, n)


def hang(tcp, cam_rigid, rot_rigid, s: float = 1.0, tcp_off=HANG_TCP, turn_deg=HANG_DEG):
    """The camera's (origin, rotation) when the hand whose TCP is commanded to ``tcp`` would carry it rigidly at
    (``cam_rigid``, ``rot_rigid``), bent by its weight: the TCP displaced by ``s`` x ``tcp_off`` and the camera turned
    by ``s`` x ``turn_deg`` (world x, y, z) about the TCP."""
    t = np.asarray(tcp, dtype=float)
    w = _turn(turn_deg, s)
    return t + s * np.asarray(tcp_off, dtype=float) + w @ (np.asarray(cam_rigid, dtype=float) - t), w @ rot_rigid


def held_lens(comp: float = PITCH_COMP, tcp_off=HANG_TCP, turn_deg=HANG_DEG):
    """(origin, rotation) of the camera (its lens's front face) held up by the left hand: gripped standing at Y_HUMP,
    Z_HUMP, lifted LIFT with the hand turned ``comp`` rad nose up, hanging as ``hang``."""
    t_grip = np.array([L.CAM_X, Y_HUMP, L.TOP + Z_HUMP])
    t_lift = t_grip + np.array([0.0, 0.0, LIFT])
    rel = tiltgrip.rx(-comp)  # the hand's turn from the grip to the lift (both are rolls about world x)
    return hang(t_lift, t_lift + rel @ (np.asarray(L.CAM_ORIGIN) - t_grip), rel, 1.0, tcp_off, turn_deg)


LENS_FACE, LENS_ROT = held_lens()  # where the right hand presents the cap (world, m)
LENS_OUT = -LENS_ROT[:, 1]  # the lens's axis, out of the lens (about 1.3 degrees off -y, toward +x)
# The right hand's own error in the turned pose (first GPU run, end of the push): the cap sat 1.2 mm short in x and
# 1.0 mm low of the command. The right hand aims that much the other way.
R_TRIM = np.array([0.0012, 0.0, 0.0010])


def _square_to_lens(r: np.ndarray) -> np.ndarray:
    """``r`` (the hand on its side, the cap's front toward -y) turned so that the cap's front points along LENS_OUT."""
    a, b = np.array([0.0, -1.0, 0.0]), LENS_OUT / np.linalg.norm(LENS_OUT)
    ax = np.cross(a, b)
    n = float(np.linalg.norm(ax))
    return r if n < 1e-12 else mk.axis_angle_matrix(ax, math.atan2(n, float(a @ b))) @ r


R_PRESS = tiltgrip.TiltedYaw(YAW_R, _square_to_lens(tiltgrip.rotation(YAW_R, ROLL)))


def _on_axis(d: float) -> tuple:
    """The right TCP (table-top frame) that puts the cap's seat face ``d`` m out from the lens's face, on its axis."""
    p = LENS_FACE + (CAP_OFF + d) * LENS_OUT + R_TRIM - np.array([0.0, 0.0, L.TOP])
    return tuple(float(v) for v in p)


PRE = 0.030  # the cap's seat face 30 mm in front of the lens before the push
OVERSHOOT = 0.0035  # commanded past the stop (about 1 N of the arms' give; the face's place is known to about 1 mm)
PRELOAD = 0.0015  # then eased back to 1.5 mm past it before letting go
PUSH_SPEED = 0.015  # m/s
BACK = 0.045
PARK_L = (0.25, 0.32, 0.140)
PARK_R = (0.20, -0.30, 0.140)


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


def right_plan(variant: str = "") -> list:
    """The right arm: pinch the cap's tabs from above (both latches in), lift it, turn the hand onto its side so the
    cap faces the lens, push it square onto the lens's front, let go (the latches spring out behind the lip), back
    away."""
    x0, y0 = L.CAP_X, L.CAP_Y
    d_end = 0.002 if variant == "short_push" else -OVERSHOOT
    d_rel = 0.002 if variant == "short_push" else -PRELOAD
    return [
        _w(2.5, x0, y0, HOVER_R, OPEN, "over the cap", R_FLAT),
        _w(1.2, x0, y0, Z_PICK + 0.020, OPEN, "down to just above the tabs", R_FLAT),
        _w(1.0, x0, y0, Z_PICK, OPEN, "slowly down round the tabs", R_FLAT),
        _w(0.3, x0, y0, Z_PICK, OPEN, "settle", R_FLAT),
        _w(1.5, x0, y0, Z_PICK, PINCH, "pinch the tabs: both latches in", R_FLAT),
        _w(0.5, x0, y0, Z_PICK, PINCH, "hold", R_FLAT),
        _w(1.5, x0, y0, Z_ROLL, PINCH, "lift the cap", R_FLAT),
        _w(3.0, *_on_axis(PRE), PINCH, "turn the hand onto its side and bring the cap in front of the lens", R_PRESS),
        _w(1.0, *_on_axis(PRE), PINCH, "settle", R_PRESS),
        _w((PRE - d_end) / PUSH_SPEED, *_on_axis(d_end), PINCH, "push the cap square onto the lens", R_PRESS),
        _w(0.5, *_on_axis(d_end), PINCH, "settle", R_PRESS),
        _w(0.5, *_on_axis(d_rel), PINCH, "ease off to a light press", R_PRESS),
        _w(1.5, *_on_axis(d_rel), OPEN, "let go of the tabs: the latches spring out behind the lip", R_PRESS),
        _w(1.5, *_on_axis(d_rel + BACK), OPEN, "straight back", R_PRESS),
        _w(2.5, *PARK_R, OPEN, "aside, the hand upright again", R_FLAT),
    ]


def left_plan(variant: str, right: list) -> list:
    """The left arm: grip the camera by its pentaprism hump, lift it (the hand 80 mm, turning 4.7 degrees nose up so
    the hanging camera stays level) and hold it while the cap goes on, set it back down on its base, let go."""
    x, y = L.CAM_X, Y_HUMP
    plan = [
        _w(2.5, x, y, HOVER_L, OPEN, "over the hump", L_GRIP),
        _w(1.2, x, y, Z_HUMP + 0.020, OPEN, "down to just above the hump", L_GRIP),
        _w(1.0, x, y, Z_HUMP, OPEN, "slowly down beside it", L_GRIP),
        _w(0.3, x, y, Z_HUMP, OPEN, "settle", L_GRIP),
        _w(1.5, x, y, Z_HUMP, GRIP_HUMP, "grip the hump", L_GRIP),
        _w(0.5, x, y, Z_HUMP, GRIP_HUMP, "hold", L_GRIP),
        _w(2.0, x, y, Z_HUMP + LIFT, GRIP_HUMP, "lift the camera, the hand turning 4.7 deg nose up", L_LIFT),
    ]
    clear = _until(right, "straight back")
    plan.append(_w(max(0.5, clear - _duration(plan)), x, y, Z_HUMP + LIFT, GRIP_HUMP,
                   "hold it while the cap goes on", L_LIFT))
    plan += [
        _w(2.0, x, y, Z_HUMP + 0.001, GRIP_HUMP, "set the camera down on its base", L_GRIP),
        _w(0.5, x, y, Z_HUMP + 0.001, GRIP_HUMP, "settle", L_GRIP),
        _w(1.5, x, y, Z_HUMP + 0.001, OPEN, "let go", L_GRIP),
        _w(1.0, x, y, HOVER_L, OPEN, "straight up", L_GRIP),
        _w(1.5, *PARK_L, OPEN, "aside", L_GRIP),
    ]
    return plan


# Wrong ways (counterfactuals), each with the score it earns.
VARIANTS = ("short_push",)


def plans(variant: str = "") -> dict:
    """Both arms' plans for the demo or a counterfactual variant.

    Order of work: the left hand grips the camera's hump and lifts the camera while the right pinches the cap's tabs
    and lifts it; the right turns onto its side, brings the cap in front of the lens and pushes it on, lets go and
    backs away; the left sets the camera down and lets go."""
    if variant not in ("",) + VARIANTS:
        raise ValueError(f"DEMO_VARIANT={variant!r} is not one of {', '.join(VARIANTS)}")
    right = right_plan(variant)
    return {"left": left_plan(variant, right), "right": right}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check).
    DEMO_VARIANT (counterfactual runs for the README, never the submitted demo):
      short_push  the cap let go 2 mm short of the lens's face: the latches spring out against the lip, the cap
                  is not home: 0.0"""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans(os.environ.get("DEMO_VARIANT", ""))
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
