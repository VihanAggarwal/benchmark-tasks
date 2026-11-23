"""On the station table, real size: a Bio-Rad Mini-PROTEAN Tetra electrode assembly (the 2-gel clamping frame), fixed,
its two green gaskets facing +y and -y and both green clamping arms swung open; two Mini-PROTEAN TGX precast gel
cassettes (10 x 8 cm, 4.6 mm thick) standing in two wet-storage racks, one left, one right, each a quarter turn from
how it goes in, and facing opposite ways.

Goal: each cassette on the gel supports against its gasket with its short plate facing in, its short plate's top just
under the gasket's notch, and each green arm swung over its cassette and clicked shut, hands off.

The parts are mechkit compounds (mechkit.py, next to this file): each cassette is one loose body (long plate, short
plate with its gel, a printed band); the electrode assembly is a fixed body with two hinged arms. The gel supports
slope outward, so a cassette let go before its arm is shut tips out onto the bench. One declared software rule
(``_clip_rule``, a mechkit on_step callback, ``clip_targets``) is each arm's clip: with a cassette of the right
thickness pressed flat on the gasket under it, the arm's cam meets resistance over the last 12 degrees (it is pushed
back open if let go there) and once pushed to within 2 degrees of shut the clip catches and holds it shut, pressing
the cassette. With nothing of that thickness there, it catches nothing and the arm swings on. Nothing moves an arm
toward shut or seats a cassette: the robot does both. The arms are read through rigstate.py (the let-go check).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
Settle:  DEMO_HOLD_ONLY=1.  One arm: DEMO_IDLE_ARM=left (or right).  Trace: TASK_TRACE=<file>.
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
from scoring import FRAME, GEL_A, GEL_B  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Clamp both gel cassettes in the electrode assembly: each against a green gasket, short plate facing "
               "in and its top just below the gasket's notch, then swing each green arm over its gel until it "
               "locks. Let go of everything.")


# -- the parts --------------------------------------------------------------------------------------------------------
def _box(x0, x1, y0, y1, z0, z1, color, quat=None):
    return mk.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color, quat=quat)


def cassette_shapes():
    """In the cassette's frame (origin at its bottom edge's middle): the long plate on +y, the short plate (with its
    gel) on -y, 10 mm shorter, and a green band printed on the long plate's foot."""
    t0 = -L.CS_T / 2
    return [_box(-L.CS_W / 2, L.CS_W / 2, t0 + L.CS_TS, L.CS_T / 2, 0.0, L.CS_H, L.CS_LONG_COLOR),
            _box(-L.CS_W / 2, L.CS_W / 2, t0, t0 + L.CS_TS, 0.0, L.CS_HS, L.CS_SHORT_COLOR),
            _box(-L.CS_W / 2 + 0.004, L.CS_W / 2 - 0.004, L.CS_T / 2, L.CS_T / 2 + 0.0002, 0.004, 0.010,
                 L.CS_BAND_COLOR)]


def support_shapes(s):
    """Side ``s`` (+1 / -1): the gel support, a groove 6 mm wide and 6 mm deep along the gasket's foot: its floor at
    L0 and a lip on its outer edge. Frame: the assembly's (origin on the table at the frame's middle)."""
    f0, f1 = sorted((s * L.GY, s * (L.GY + L.GROOVE_W)))
    l0, l1 = sorted((s * (L.GY + L.GROOVE_W), s * (L.GY + L.GROOVE_W + L.LIP_W)))
    return [_box(-L.LEG_X1, L.LEG_X1, f0, f1, 0.0, L.L0, L.CORE_COLOR),
            _box(-L.LEG_X1, L.LEG_X1, l0, l1, 0.0, L.LIP_TOP, L.CORE_COLOR)]


def frame_shapes():
    """The U core (two legs, a bottom bar), green gasket faces on both sides down to the support floor, the gel
    supports, and the two arm pillars at the far end on a foot."""
    g_in = L.GY - L.GASKET_T
    out = [_box(-L.LEG_X1, L.LEG_X1, -g_in, g_in, 0.0, L.BOTTOM_Z, L.CORE_COLOR)]
    for sx in (-1, 1):
        x0, x1 = sorted((sx * L.LEG_X0, sx * L.LEG_X1))
        out.append(_box(x0, x1, -g_in, g_in, L.BOTTOM_Z, L.NOTCH_Z, L.CORE_COLOR))
    for s in (-1, 1):
        y0, y1 = sorted((s * g_in, s * L.GY))
        for sx in (-1, 1):  # the gasket on each leg and along the bar
            x0, x1 = sorted((sx * L.LEG_X0, sx * L.LEG_X1))
            out.append(_box(x0, x1, y0, y1, L.BOTTOM_Z, L.NOTCH_Z, L.GASKET_GREEN))
        out.append(_box(-L.LEG_X1, L.LEG_X1, y0, y1, L.L0, L.BOTTOM_Z, L.GASKET_GREEN))
        out.append(_box(-L.LEG_X1, L.LEG_X1, y0, y1, 0.0, L.L0, L.CORE_COLOR))
        out += support_shapes(s)
        out.append(mk.cylinder((L.PIV_X - L.FX, s * L.PIV_Y, (L.ARM_Z0 - 0.001) / 2), L.PILLAR_R, L.ARM_Z0 - 0.001,
                               color=L.CORE_COLOR))
    out.append(_box(L.LEG_X1, L.PIV_X - L.FX + 0.008, -(L.PIV_Y + 0.008), L.PIV_Y + 0.008, 0.0, 0.006,
                    L.CORE_COLOR))
    return out


def arm_shapes(side):
    """The green arm drawn open (q = 0, pointing straight out): hub on its pin, the bar and a handle grip on its last
    40 mm, in the assembly's frame."""
    s = L.arm_sign(side)
    px, py = L.PIV_X - L.FX, s * L.PIV_Y
    y0, y1 = sorted((py, py + s * L.ARM_L))
    g0, g1 = sorted((py + s * (L.ARM_L - 0.040), py + s * L.ARM_L))
    return [mk.cylinder((px, py, L.ARM_ZC), L.HUB_R, L.AH, color=L.ARM_GREEN),
            _box(px - L.AW / 2, px + L.AW / 2, y0, y1, L.ARM_Z0, L.ARM_Z1, L.ARM_GREEN),
            _box(px - L.AW / 2 - 0.001, px + L.AW / 2 + 0.001, g0, g1, L.ARM_Z0 + 0.001, L.ARM_Z1 + 0.003,
                 (0.10, 0.45, 0.22))]


def make_frame():
    joints = []
    for side, axis in (("pos", "Z"), ("neg", "-Z")):
        s = L.arm_sign(side)
        joints.append(mk.hinge(S.ARM_JOINT[side], axis, anchor=(L.PIV_X - L.FX, s * L.PIV_Y, L.ARM_ZC),
                               shapes=arm_shapes(side), lower=0.0, upper=L.Q_UPPER, mass=L.ARM_MASS, spring=L.ARM_K,
                               damping=L.ARM_DAMPING))
    return mk.assembly(FRAME, origin=(L.FX, 0.0, L.TOP), shapes=frame_shapes(), joints=joints, color=L.CORE_COLOR,
                       friction=L.FRAME_FRICTION, tight_fit=True)


def rack_shapes():
    """A white block with one slot along y (frame: its middle on the table)."""
    hx, hy = L.RACK_L / 2, L.RACK_W / 2
    sx, sy = L.SLOT_W / 2, L.SLOT_L / 2
    f, h = L.SLOT_FLOOR, L.RACK_H
    return [_box(-hx, hx, -hy, hy, 0.0, f, L.RACK_COLOR),
            _box(-hx, -sx, -hy, hy, f, h, L.RACK_COLOR), _box(sx, hx, -hy, hy, f, h, L.RACK_COLOR),
            _box(-sx, sx, -hy, -sy, f, h, L.RACK_COLOR), _box(-sx, sx, sy, hy, f, h, L.RACK_COLOR)]


def make_gel(name, start):
    x, y, z, yaw = start
    return mk.assembly(name, origin=(x, y, L.TOP + z), yaw=yaw, mass=L.CS_MASS, shapes=cassette_shapes(),
                       color=L.CS_LONG_COLOR, friction=L.CS_FRICTION, tight_fit=True)


FRAME_PART = make_frame()
RACK_LEFT = mk.assembly("rack_left", origin=(L.RACK_X, L.RACK_Y, L.TOP), shapes=rack_shapes(), color=L.RACK_COLOR,
                        tight_fit=True)
RACK_RIGHT = mk.assembly("rack_right", origin=(L.RACK_X, -L.RACK_Y, L.TOP), shapes=rack_shapes(),
                         color=L.RACK_COLOR, tight_fit=True)
GEL_A_PART = make_gel(GEL_A, L.A_START)
GEL_B_PART = make_gel(GEL_B, L.B_START)
OBJECTS = [FRAME_PART, RACK_LEFT, RACK_RIGHT, GEL_A_PART, GEL_B_PART]
SPECS = {GEL_A: GEL_A_PART, GEL_B: GEL_B_PART, FRAME: FRAME_PART}


# -- the one declared rule: each arm's clip --------------------------------------------------------------------------
def clip_targets(q, v, latched, any_present):
    """One control step of an arm's clip: arm at ``q`` (rad) moving at ``v`` (rad/s), ``latched`` its clip state,
    ``any_present`` whether a cassette of the right thickness lies flat on its gasket (scoring.present). Returns
    (spring target, latched now).
      latched                 caught: target 0.05 rad past shut (the arm rests on the cassette, pressing it with
                              0.1 N m); pulled back 5 degrees past the catch, it lets go
      cassette, cam zone      over the last 12 degrees short of the catch the cam rides on the cassette: target 0.05
                              rad behind the arm (an opening torque of 0.1 N m: let go there, it is pushed back open)
      cassette, at the catch  within 2 degrees of shut: caught
      otherwise               free: the target leads the arm by one control step (no torque, no drag)
    A state that cannot be read sets nothing."""
    try:
        q = float(q)
        v = float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return None, latched
    if not math.isfinite(q):
        return None, latched
    if not math.isfinite(v):
        v = 0.0
    if latched and q < L.Q_UNLATCH:
        latched = False
    if not latched and any_present and q >= L.Q_LATCH:
        latched = True
    if latched:
        return L.Q_CLOSED + L.PRELOAD, True
    if any_present and q >= L.Q_CAM0:
        return q - L.CAM_BACK, False
    return q + v * L.ARM_LEAD_S, False


def any_present(gels: dict, side: str) -> bool:
    return any(S.present(S.fit(g, side)) for g in gels.values())


_LATCH = {"pos": False, "neg": False}


@mk.on_step
def _clip_rule(states) -> None:
    """In the simulator, at the start of every control step (before the action and the physics)."""
    fr = states.get(FRAME)
    gels = {}
    for n in S.GELS:
        g = S.gel_from_state(states.get(n)) if states.get(n) is not None else None
        if g is not None:
            gels[n] = g
    for side in S.SIDES:
        joint = S.ARM_JOINT[side]
        try:
            q = fr.joints[joint]
            v = (getattr(fr, "joint_vel", None) or {}).get(joint, 0.0)
        except (AttributeError, KeyError, TypeError):
            continue
        target, _LATCH[side] = clip_targets(q, v, _LATCH[side], any_present(gels, side))
        if target is not None:
            mk.set_targets(FRAME_PART, joint, target)


# -- the offline stand-in for the physics ----------------------------------------------------------------------------
# The checker's proxy neither turns a hinge nor tips a part. Offline only (never in the simulator) this model gives the
# scorer: (1) a cassette pinched across its thickness near its top by a closed gripper follows the TCP and its yaw; (2)
# let go with its foot in a gasket's support groove it leans out 12 degrees onto the lip, unless that side's arm is
# caught (it then stands upright, clamped); anywhere else but its rack it lies flat on the bench; (3) a gripper
# straddling both cassettes with its pads squeezed to the pair's thickness stands both upright while it squeezes;
# (4) an arm is pushed by a closed gripper's block behind it (the TCP PUSH_BEHIND behind the arm's centre line),
# stopping at shut on an upright cassette, about 7 degrees short of shut on a leaning one (the bar meets the leaning
# cassette's foot), and the clip rule above runs on it (springs back to the cam's start if let go in the cam zone).
Q_BLOCKED = L.Q_CLOSED - math.radians(7.0)


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _yaw_at(side: str, t: float) -> float:
    prev, acc = 0.0, 0.0
    for d, yaw in PLAN_YAWS.get(side, ()):
        if t <= acc + d:
            return prev + _smooth((t - acc) / d if d > 0 else 1.0) * (yaw - prev)
        acc += d
        prev = yaw
    return prev


class _Gel:
    def __init__(self, start) -> None:
        self.x, self.y, self.z, self.yaw = start
        self.mode = "rack"  # rack, held, groove (on a support: leaning or upright), fallen
        self.holder = None
        self.grip = None  # (local x, local y, dz, dyaw)

    def rot2(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s], [s, c]])


class OfflineModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.gels = {GEL_A: _Gel(L.A_START), GEL_B: _Gel(L.B_START)}
        self.q = {"pos": 0.0, "neg": 0.0}
        self.latched = {"pos": False, "neg": False}
        self.squeezed = False

    @staticmethod
    def side_of(g: _Gel) -> str:
        return "pos" if g.y > 0 else "neg"

    def upright(self, g: _Gel) -> bool:
        return g.mode == "groove" and (self.squeezed or self.latched[self.side_of(g)])

    def gel_obj(self, g: _Gel) -> S.Gel:
        if g.mode == "fallen":
            s = 1.0 if g.y >= 0 else -1.0
            r = mk.quat_to_matrix(mk.quat_from_ypr(g.yaw, 0.0, -math.radians(85.0)))
            return S.Gel(np.array([g.x, g.y + s * 0.05, 0.003]), r)
        if g.mode == "groove" and not self.upright(g):
            s = L.arm_sign(self.side_of(g))
            r = mk.quat_to_matrix(mk.quat_from_ypr(g.yaw, 0.0, -math.radians(L.LEAN_DEG)))
            corner = np.array([g.x, s * L.GY, L.L0])
            return S.Gel(corner - r @ np.array([0.0, -L.CS_T / 2, 0.0]), r)
        r = mk.quat_to_matrix(np.array([0.0, 0.0, math.sin(g.yaw / 2), math.cos(g.yaw / 2)]))
        return S.Gel(np.array([g.x, g.y, g.z]), r)

    @staticmethod
    def in_groove(g: _Gel) -> bool:
        s = 1.0 if g.y > 0 else -1.0
        return bool(abs(math.sin(g.yaw)) < 0.05 and abs(g.x - L.FX) < 0.008
                    and abs(s * g.y - (L.GY + L.CS_T / 2)) < 0.0015 and abs(g.z - L.L0) < 0.003)

    def step(self, arms) -> None:
        self.steps += 1
        t = self.steps / L.CONTROL_HZ
        read = rigstate.read_arms(arms)
        if read is None:
            return
        hands = {side: (tcp, tips, op) for side, (tcp, tips, op) in zip(rigstate.ARMS, read)}
        for side, (tcp, tips, op) in hands.items():
            x, y, z = float(tcp[0]), float(tcp[1]), float(tcp[2]) - L.TOP
            yaw = _yaw_at(side, t)
            held = [g for g in self.gels.values() if g.holder == side]
            if held and op > 0.12:
                g = held[0]
                g.holder, g.grip = None, None
                if self.in_groove(g):
                    s = 1.0 if g.y > 0 else -1.0
                    g.y, g.z, g.mode = s * (L.GY + L.CS_T / 2), L.L0, "groove"
                elif abs(g.x - L.RACK_X) < 0.004 and abs(abs(g.y) - L.RACK_Y) < 0.01 and g.z < L.SLOT_FLOOR + 0.004:
                    g.mode = "rack"
                else:
                    g.mode = "fallen"
            elif not held and op < 0.06:
                for g in self.gels.values():
                    if g.holder is not None or g.mode != "rack":
                        continue
                    loc = g.rot2().T @ np.array([x - g.x, y - g.y])
                    tip_z = float(tips[2]) - L.TOP
                    top = g.z + L.CS_H
                    if abs(loc[1]) < 0.004 and abs(loc[0]) < 0.040 and top - 0.030 < tip_z < top - 0.008:
                        g.holder, g.mode = side, "held"
                        g.grip = (float(loc[0]), float(loc[1]), z - g.z, g.yaw - yaw)
                        break
        for g in self.gels.values():
            if g.holder is None:
                continue
            tcp = hands[g.holder][0]
            lx, ly, dz, dyaw = g.grip
            g.yaw = _yaw_at(g.holder, t) + dyaw
            off = g.rot2() @ np.array([lx, ly])
            g.x, g.y = float(tcp[0]) - off[0], float(tcp[1]) - off[1]
            g.z = float(tcp[2]) - L.TOP - dz
            if abs(math.sin(g.yaw)) < 0.05 and abs(g.x - L.FX) < L.LEG_X1 + L.CS_W / 2:
                s = 1.0 if g.y > 0 else -1.0  # a held cassette is stopped by the gasket face and the support floor
                if s * g.y < L.GY + L.CS_T / 2 + 0.004 and g.z < L.NOTCH_Z:
                    g.y = s * max(s * g.y, L.GY + L.CS_T / 2)
                    g.z = max(g.z, L.L0)
        # the squeeze: a gripper straddling the pair at their tops, pads at the pair's thickness
        self.squeezed = False
        for side, (tcp, tips, op) in hands.items():
            gap = op * rigstate.FULL_GAP
            tip_z = float(tips[2]) - L.TOP
            if (abs(float(tcp[0]) - L.FX) < L.CS_W / 2 - 0.005 and abs(float(tcp[1])) < 0.004
                    and L.L0 + L.CS_H - 0.020 < tip_z < L.L0 + L.CS_H - 0.004 and gap < 2 * (L.GY + L.CS_T) + 0.001
                    and gap > 0.020):
                self.squeezed = True
        # the arms
        gels = {n: self.gel_obj(g) for n, g in self.gels.items()}
        for side in S.SIDES:
            pres = any_present(gels, side)
            leaning = any(g.mode == "groove" and self.side_of(g) == side and not self.upright(g)
                          for g in self.gels.values())
            pushed = False
            for hside, (tcp, tips, op) in hands.items():
                if op > 0.06 or any(g.holder == hside for g in self.gels.values()):
                    continue
                tip_z = float(tips[2]) - L.TOP
                if not (tip_z < L.ARM_Z1 and tip_z + L.FIST_H > L.ARM_Z0):
                    continue
                px, py = L.pivot(side)
                rel = np.array([float(tcp[0]) - px, float(tcp[1]) - py])
                s = L.arm_sign(side)
                alpha = math.atan2(-rel[0], s * rel[1])
                rad = float(np.hypot(*rel))
                if not (0.03 < rad < L.ARM_L + 0.01):
                    continue
                phi = alpha + math.atan2(L.PUSH_BEHIND, math.sqrt(max(rad * rad - L.PUSH_BEHIND ** 2, 1e-9)))
                if alpha <= self.q[side] + 0.01 and phi > self.q[side] - 0.002:
                    stop = L.Q_CLOSED if pres else (Q_BLOCKED if leaning else L.Q_UPPER)
                    self.q[side] = min(max(self.q[side], phi), stop)
                    pushed = True
            _, self.latched[side] = clip_targets(self.q[side], 0.0, self.latched[side], pres)
            if self.latched[side]:
                self.q[side] = L.Q_CLOSED
            elif pres and self.q[side] >= L.Q_CAM0 and not pushed:
                self.q[side] = L.Q_CAM0 - 0.05  # let go on the cam: pushed back open

    def states(self) -> dict:
        out = {}
        for n, g in self.gels.items():
            o = self.gel_obj(g)
            out[n] = mk.state_from_poses(SPECS[n], (o.o[0], o.o[1], L.TOP + o.o[2], *mk.matrix_to_quat(o.r)),
                                         source="offline")
        out[FRAME] = mk.state_from_poses(FRAME_PART, (L.FX, 0.0, L.TOP, 0.0, 0.0, 0.0, 1.0),
                                         joints={S.ARM_JOINT[s]: self.q[s] for s in S.SIDES}, source="offline")
        return out


_OFFLINE = OfflineModel()


def no_simulator(objects) -> bool:
    return mk.env() is None and getattr(objects.get(GEL_A), "source", "sim") != "sim"


# -- scoring hooks ---------------------------------------------------------------------------------------------------
_TRACKER = S.Tracker()
_EPISODE = {"mechkit": None}


def _new():
    _LATCH.update({"pos": False, "neg": False})
    _TRACKER.reset()
    _OFFLINE.reset()


@rigstate.on_reset
def _new_episode() -> None:
    _new()


@mk.on_reset
def _kit_reset() -> None:
    _new()


def _trace(row: dict) -> None:
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:
            pass


def success(objects) -> bool:
    """1.0 (both cassettes seated short plate in, both arms shut over them), let go and still for 1 s; held 1 s.
    Runs first every control step: it also feeds the tracker (and, without a simulator, the offline stand-in)."""
    if _EPISODE["mechkit"] != mk.episode():
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
           "latch": dict(_LATCH), "fallback_resets": _TRACKER.resets, **result.details}
    if arms is not None:
        try:
            row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                       openings={a: round(float(arms[a].opening), 3) for a in ("left", "right")})
        except (AttributeError, KeyError, TypeError, ValueError):
            row["rigstate_error"] = "unreadable snapshot"
    else:
        row["rigstate_error"] = rigstate.error()
    row["source"] = getattr(parts.get(GEL_A), "source", None)
    _trace(row)
    return result.success


def score(objects) -> float:
    """The sum of each side's credit (0.15 against short plate in, 0.15 seated under the notch, 0.20 arm shut over
    it), if everything is let go and still and no cassette went off the bench; the score of the latest step."""
    return _TRACKER.last.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop: timed waypoints from the known start layout, nothing read from the simulator. TCP points (heights above
# the table); the pad tips are 13 mm below. Finger targets: the joint value is minus half the commanded gap.
OPEN = -0.04695
PINCH = 0.0  # fingers commanded shut on the 4.6 mm cassette: squeezed
LET_GO_GAP = -0.008  # 16 mm: opened just enough to let go, the inner finger clear of the other cassette
CLOSED = 0.0
SQUEEZE = -L.SQUEEZE_GAP / 2
HOVER = 0.150
PUSH_HOVER = 0.095
SQUEEZE_HOVER = 0.125  # pad tips 22 mm over the cassettes' tops
GRIP_Z_RACK = L.SLOT_FLOOR + L.CS_H - L.GRIP_DOWN + L.TIPS_BELOW  # 0.076
GRIP_Z_SEAT = L.L0 + L.CS_H - L.GRIP_DOWN + L.TIPS_BELOW  # 0.081
SQUEEZE_Z = L.SQUEEZE_TIPS + L.TIPS_BELOW
OUT = 0.016  # the cassette is lowered 16 mm outboard of its gasket, then moved in
ABOVE = 0.004  # its foot 4 mm over the support's lip while it moves in, then down the gasket face into the groove
PRESS = 0.0005  # 0.5 mm into the gasket
SWEEP_DEG = (-7.0, 15.0, 35.0, 55.0, 70.0, 78.0, 84.0, 88.0, 90.0, 92.0)  # the pushing block's arm angle
SWEEP_S = (1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 1.0, 1.0, 1.0, 1.0)
START_DEG = -7.0
PARK = {"left": (0.16, 0.25, 0.15), "right": (0.16, -0.25, 0.15)}


def _w(d, x, y, z, finger, label, yaw):
    return Waypoint(d, (float(x), float(y), L.TOP + float(z)), finger, label, float(yaw))


def _dur(plan) -> float:
    return sum(w.duration for w in plan)


def _until(plan, prefix) -> float:
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(prefix):
            return t
    raise KeyError(prefix)


def _wait_until(plan, t_abs, label):
    """Hold the last waypoint until ``t_abs``."""
    last = plan[-1]
    d = t_abs - _dur(plan)
    if d > 0.05:
        plan.append(Waypoint(d, last.pos, last.finger, label, last.yaw))


def place_moves(start, side, hand):
    """Pick a racked cassette at its middle (fingers closing across its thickness, yaw -pi/2), lift it, turn it a
    quarter turn (+90 degrees) on the way over, lower it outboard of the ``side`` gasket, move it in against the
    gasket face, slide it down the face into the support groove, open just enough to let go (it leans out onto the
    lip) and rise."""
    s = L.arm_sign(side)
    x, y = start[0], start[1]
    oy = s * (L.GY + L.CS_T / 2)  # the cassette's middle plane, upright against the gasket
    return [
        _w(2.0, x, y, HOVER, OPEN, f"{hand}: over its cassette in the rack", -math.pi / 2),
        _w(1.2, x, y, GRIP_Z_RACK + 0.025, OPEN, "down to just above it", -math.pi / 2),
        _w(1.0, x, y, GRIP_Z_RACK, OPEN, "down round its top edge", -math.pi / 2),
        _w(1.2, x, y, GRIP_Z_RACK, PINCH, "pinch it across its thickness", -math.pi / 2),
        _w(1.5, x, y, HOVER, PINCH, "lift it out of the rack", -math.pi / 2),
        _w(3.0, L.FX, oy + s * OUT, HOVER, PINCH, "carry it over, turning it a quarter turn", 0.0),
        _w(1.2, L.FX, oy + s * OUT, GRIP_Z_SEAT + 0.020, PINCH, "down outboard of the gasket", 0.0),
        _w(0.8, L.FX, oy + s * OUT, GRIP_Z_SEAT + L.LIP_TOP - L.L0 + ABOVE, PINCH, "its foot just over the lip", 0.0),
        _w(1.2, L.FX, oy - s * PRESS, GRIP_Z_SEAT + L.LIP_TOP - L.L0 + ABOVE, PINCH, "in against the gasket", 0.0),
        _w(1.2, L.FX, oy - s * PRESS, GRIP_Z_SEAT + 0.0003, PINCH, "slide it down into the groove", 0.0),
        _w(0.8, L.FX, oy - s * PRESS, GRIP_Z_SEAT + 0.0003, LET_GO_GAP, "let go: it leans out onto the lip", 0.0),
        _w(1.5, L.FX, oy, HOVER, LET_GO_GAP, "straight up", 0.0),
    ]


def squeeze_moves(hand):
    """Straddle both cassettes near their far ends, pads open, down to 12 mm under their tops, and close to 4 mm
    inside the pair's thickness: both stand upright, flat on their gaskets (the manual: squeeze them toward each
    other)."""
    x = L.SQUEEZE_X
    return [_w(1.8, x, 0.0, SQUEEZE_HOVER, OPEN, f"{hand}: over the pair, fingers open across them", 0.0),
            _w(1.5, x, 0.0, SQUEEZE_Z, OPEN, "down astride both cassettes", 0.0),
            _w(1.5, x, 0.0, SQUEEZE_Z, SQUEEZE, "squeeze them upright onto their gaskets", 0.0)]


def unsqueeze_moves(hand):
    x = L.SQUEEZE_X
    return [_w(1.0, x, 0.0, SQUEEZE_Z, OPEN, f"{hand}: open off the pair", 0.0),
            _w(1.2, x, 0.0, SQUEEZE_HOVER, OPEN, "straight up", 0.0)]


def push_tcp(side, q_deg, extra=0.0):
    q = math.radians(q_deg)
    px, py = L.pivot(side)
    dx, dy = L.arm_dir(side, q)
    mx, my = L.arm_motion(side, q)
    b = L.PUSH_BEHIND + extra
    return px + L.PUSH_R * dx - b * mx, py + L.PUSH_R * dy - b * my


def push_yaw(side, q_deg):
    """Fingers closing along the arm's motion, so the closed block's flat face leads."""
    q = math.radians(q_deg)
    return q - math.pi / 2 if side == "pos" else math.pi / 2 - q


def push_moves(side, hand, park):
    """A closed gripper's block comes down behind the open arm's handle, 180 mm out from its pin, and pushes it round
    its pin to shut, slowly over the cam, 2 degrees past shut (the arm stops on the cassette: the block presses it),
    holds, backs off outward, rises and goes aside."""
    x0, y0 = push_tcp(side, START_DEG, 0.004)
    plan = [_w(1.8, x0, y0, PUSH_HOVER, CLOSED, f"{hand}: behind the {side} arm's handle, fingers closed",
               push_yaw(side, START_DEG)),
            _w(1.5, x0, y0, L.PUSH_Z, CLOSED, "down behind the open arm", push_yaw(side, START_DEG))]
    for deg, d in zip(SWEEP_DEG, SWEEP_S):
        x, y = push_tcp(side, deg)
        plan.append(_w(d, x, y, L.PUSH_Z, CLOSED, f"swing the {side} arm to {deg:.0f} deg", push_yaw(side, deg)))
    x, y = push_tcp(side, SWEEP_DEG[-1])
    s = L.arm_sign(side)
    plan += [_w(0.8, x, y, L.PUSH_Z, CLOSED, "hold it shut", push_yaw(side, 90.0)),
             _w(0.8, x, y + s * 0.015, L.PUSH_Z, CLOSED, "back off outward", push_yaw(side, 90.0)),
             _w(1.0, x, y + s * 0.015, PUSH_HOVER, CLOSED, "up off the arm", push_yaw(side, 90.0)),
             _w(1.5, *park, OPEN, f"{hand}: aside", 0.0)]
    return plan


def plans() -> dict:
    """Left: gel A from the left rack onto the +y support (it leans out). Right: gel B from the right rack onto the -y
    support. Left: straddle the pair and squeeze both upright; right: push the -y arm shut and go aside; left lets go.
    Right: straddle and squeeze; left: push the +y arm shut and go aside; right lets go. Each arm pushes only the arm
    on its own side; a hand works over the frame only while the other is aside or low beside it."""
    left = place_moves(L.A_START, "pos", "left")
    left.append(_w(1.5, *PARK["left"], OPEN, "left: aside", 0.0))
    right = [_w(2.0, *PARK["right"], OPEN, "right: aside", -math.pi / 2)]
    b = place_moves(L.B_START, "neg", "right")
    t_carry = _dur(left) - 0.5  # the right hand carries gel B over the frame once the left hand is aside
    pre = b[:5]
    _wait_until(right, t_carry - _dur(pre), "right: wait aside")
    right += pre
    right += b[5:]
    right.append(_w(1.5, *PARK["right"], OPEN, "right: aside", 0.0))
    _wait_until(left, _dur(right) - 1.5, "left: wait aside")
    left += squeeze_moves("left")
    _wait_until(right, _dur(left) - 1.5, "right: wait while the left hand squeezes")
    right += push_moves("neg", "right", PARK["right"])
    _wait_until(left, _until(right, "up off the arm"), "left: hold the squeeze while the -y arm is shut")
    left += unsqueeze_moves("left")
    left.append(_w(1.5, *PARK["left"], OPEN, "left: aside", 0.0))
    _wait_until(right, _dur(left) - 1.5, "right: wait aside")
    right += squeeze_moves("right")
    _wait_until(left, _dur(right) - 1.5, "left: wait while the right hand squeezes")
    left += push_moves("pos", "left", PARK["left"])
    _wait_until(right, _until(left, "up off the arm"), "right: hold the squeeze while the +y arm is shut")
    right += unsqueeze_moves("right")
    right.append(_w(1.5, *PARK["right"], OPEN, "park", 0.0))
    return {"left": left, "right": right}


PLAN_YAWS = {side: [(w.duration, w.yaw) for w in plan] for side, plan in plans().items()}


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
